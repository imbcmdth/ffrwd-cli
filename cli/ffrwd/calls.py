"""A call's arguments against the filter or macro it names.

Stream inputs first, then the filter's positional options, then named ones:
this is where an argument list is matched to that signature, where an array
argument's length settles how many nodes the call mints, and where the
rejections that say what went wrong are built.
"""

from __future__ import annotations

import difflib
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from sqlglot import exp

from ffrwd.errors import ErrorCode, FfrwdError
from ffrwd.expressions import _error, _number, _sql_text, _unwrap
from ffrwd.filter_options import _NamedArg
from ffrwd.ir import FrameRef, StreamType
from ffrwd.macros import Macro, macro_names
from ffrwd.parser import (
    FILTER_NAMESPACE,
    MACRO_NAMESPACE,
    _pos,
    kwarg_name,
    null_variable,
)
from ffrwd.parser import (
    _ident_name as _fold,
)
from ffrwd.registry import FilterOption
from ffrwd.values import (
    _PASSTHROUGH_ONLY,
    _agreed_source,
    _array,
    _scalar,
    _Stream,
    _stream_count,
    _Value,
)
from ffrwd.vars import unset_error

if TYPE_CHECKING:
    from ffrwd.filters import _ArrayFilter, _BadCount, _NInputFilter


_ZIP_HINT = (
    "broadcast arrays zip elementwise, one output per element; "
    "subscript one of them to pair a single stream with the other, e.g. a.audio[1]"
)


_PASSTHROUGH_HINT = (
    "subtitle and data streams can only be selected (and copied), never filtered; "
    "drop them from the call and select them as their own column"
)


@dataclass(frozen=True)
class _Call:
    """A function call as lowering sees it: a name, positional args, named args.

    `namespaced` marks the ``ffmpeg.<filter>(...)`` spelling, which
    resolves in the registry under a name no Postgres grammar can claim.
    `is_macro` marks the ``ffrwd.<name>(...)`` spelling, which
    resolves against :data:`MACROS` and never touches the registry. The two
    are mutually exclusive (different Dot qualifiers).

    `variadic` is the array expression inside a trailing ``VARIADIC <array>``
    argument, already unwrapped from ``exp.Variadic`` and excluded from
    `args` -- Postgres allows at most one, and it is always last, which
    :func:`_split_args` enforces at parse time.
    """

    name: str
    args: list[exp.Expr]
    named: list[_NamedArg]
    namespaced: bool = False
    is_macro: bool = False
    variadic: exp.Expr | None = None

    @property
    def display(self) -> str:
        """The call as the user spelled it, for error messages."""
        if self.namespaced:
            return f"{FILTER_NAMESPACE}.{self.name}"
        if self.is_macro:
            return f"{MACRO_NAMESPACE}.{self.name}"
        return self.name


def _namespaced_call(node: exp.Expr) -> exp.Anonymous | None:
    """The ``exp.Anonymous`` inside ``ffmpeg.<filter>(...)``, else None.

    VERIFIED (sqlglot 30.17, ``read="postgres"``): a qualified call parses as
    ``exp.Dot(this=Identifier(ffmpeg), expression=exp.Anonymous(...))`` for
    EVERY filter name, with its positional arguments and its ``=>`` kwargs
    intact inside the ``Anonymous``. Postgres's special-form grammars —
    ``OVERLAY(x PLACING y ...)``, ``TRIM``, ``FORMAT``, ``MEDIAN``, ... — key
    on a BARE name, so qualifying the call bypasses all of them at once. That
    is the whole point of the namespace: it is the one spelling of a filter
    name that no SQL grammar has an opinion about.
    """
    if not isinstance(node, exp.Dot):
        return None
    qualifier = node.this
    if not isinstance(qualifier, exp.Identifier) or _fold(qualifier) != FILTER_NAMESPACE:
        return None
    inner = node.args.get("expression")
    return inner if isinstance(inner, exp.Anonymous) else None


def _macro_call(node: exp.Expr) -> exp.Anonymous | None:
    """The ``exp.Anonymous`` inside ``ffrwd.<name>(...)``, else None.

    Mirrors :func:`_namespaced_call` exactly, and VERIFIED to parse
    to the identical shape under sqlglot 30.17 ``read="postgres"`` for all
    three macro names: ``exp.Dot(this=Identifier(ffrwd),
    expression=exp.Anonymous(this=<macro>, expressions=[...]))``, symmetric
    with the ffmpeg namespace's.
    """
    if not isinstance(node, exp.Dot):
        return None
    qualifier = node.this
    if not isinstance(qualifier, exp.Identifier) or _fold(qualifier) != MACRO_NAMESPACE:
        return None
    inner = node.args.get("expression")
    return inner if isinstance(inner, exp.Anonymous) else None


def _call_parts(node: exp.Expr) -> _Call | None:
    """The call `node` is, else None.

    ``exp.Overlay`` is normalized back to the four positional arguments the
    SQL surface uses; sqlglot parks them under named keys because Postgres
    spells the builtin ``OVERLAY(x PLACING y FROM n FOR m)``. (That builtin
    grammar also means a BARE ``overlay(...)`` cannot take named arguments at
    all: sqlglot rejects ``=>`` inside it at PARSE time. Its options are still
    reachable positionally, and ``ffmpeg.overlay(base, top, x => 20, y => 20)``
    reaches every one of them by name.)

    Named arguments arrive as ``exp.Kwarg`` among the positional ones and are
    split out here. Their TRAILING position is enforced by resolve; the check
    is repeated defensively because a Kwarg among positional args would
    otherwise silently shift every parameter after it.
    """
    inner = _namespaced_call(node)
    if inner is not None:
        return _split_args(str(inner.this), inner, namespaced=True)
    macro_inner = _macro_call(node)
    if macro_inner is not None:
        return _split_args(str(macro_inner.this), macro_inner, is_macro=True)
    if isinstance(node, exp.Overlay):
        parts = [
            node.this,
            node.args.get("expression"),
            node.args.get("from_"),
            node.args.get("for_"),
        ]
        return _Call("overlay", [arg for arg in parts if isinstance(arg, exp.Expr)], [])
    if isinstance(node, exp.Anonymous):
        return _split_args(str(node.this), node)
    if isinstance(node, exp.Func):
        return _split_args(node.sql_name().lower(), node)
    return None


def _split_args(
    name: str, call: exp.Expr, *, namespaced: bool = False, is_macro: bool = False
) -> _Call:
    positional: list[exp.Expr] = []
    named: list[_NamedArg] = []
    variadic: exp.Expr | None = None
    for arg in call.expressions:
        if not isinstance(arg, exp.Expr):
            continue
        if isinstance(arg, exp.Kwarg):
            value = arg.args.get("expression")
            if not isinstance(value, exp.Expr):  # resolve already rejected this
                raise _error(
                    ErrorCode.UNSUPPORTED_SQL,
                    f"named argument '{kwarg_name(arg)}' has no value",
                    arg,
                )
            named.append(_NamedArg(name=kwarg_name(arg), value=value))
            continue
        if named:
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                "positional arguments must come before named arguments",
                arg,
                fallback=call,
            )
        if variadic is not None:
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                "VARIADIC must be the last argument",
                arg,
                fallback=call,
            )
        if isinstance(arg, exp.Variadic):
            inner = arg.this
            if not isinstance(inner, exp.Expr):
                raise _error(
                    ErrorCode.UNSUPPORTED_SQL,
                    "VARIADIC needs an array expression",
                    arg,
                    fallback=call,
                )
            variadic = inner
            continue
        positional.append(arg)
    return _Call(name, positional, named, namespaced, is_macro, variadic)


def _macro_options(
    macro: Macro, call: _Call, node: exp.Expr
) -> dict[str, object]:
    """A macro's named-only options: every one optional, none repeated.

    Returned in the MACRO's declared order, not the order they were
    written, so the rendered filter is the same whichever way round the
    query spells them. An omitted option is left out entirely -- the
    expansion renders only what was written, and ffmpeg's own default
    covers the rest. Repeats need no check here: resolve rejects a
    duplicate `name =>` on any call before lowering starts.
    """
    written: dict[str, object] = {}
    for argument in call.named:
        if argument.name not in macro.options:
            raise _error(
                ErrorCode.UDF_ARG_TYPE,
                f"{call.display}() has no '{argument.name}' option",
                argument.value,
                fallback=node,
                hint=f"its signature is {macro.signature}",
            )
        try:
            written[argument.name] = _number(argument.value)
        except FfrwdError as exc:
            raise _error(
                exc.code,
                f"{call.display}()'s '{argument.name}' option must be a "
                "numeric literal",
                argument.value,
                fallback=node,
                hint=f"its signature is {macro.signature}",
            ) from None
    return {name: written[name] for name in macro.options if name in written}


def _macro_function_hint(name: str) -> str:
    """Did-you-mean over :data:`MACROS`, the small-by-design macro set."""
    matches = difflib.get_close_matches(name, macro_names(), n=1, cutoff=0.6)
    if matches:
        return f"did you mean {MACRO_NAMESPACE}.{matches[0]}()?"
    return (
        f"{MACRO_NAMESPACE}.<name> is one of ffrwd's own macros -- "
        f"{', '.join(macro_names())} -- not an ffmpeg filter; filters live "
        f"bare or under {FILTER_NAMESPACE}.<filter>(...)"
    )


def _n_input_count(
    spec: _NInputFilter,
    option_name: str,
    args: dict[str, object],
    options: dict[str, FilterOption],
) -> int:
    """What the count option says, written or introspected-default or fallback.

    `args` has already been validated against the option table, so a
    written value is a number in range; only the DEFAULT needs care, since
    `FilterOption.default` is verbatim ffmpeg text that is documented as
    never re-typed (it can be a constant name, or absent entirely). Called
    only when `spec.option` is not None; `option_name` is that narrowed
    value, passed separately so mypy sees a plain `str`.
    """
    written = args.get(option_name)
    if isinstance(written, (int, float)) and not isinstance(written, bool):
        return int(written)
    option = options.get(option_name)
    if option is not None and option.default is not None:
        try:
            return int(float(option.default))
        except ValueError:
            pass
    return spec.fallback


def _bad_count(
    spec: _ArrayFilter,
    bad: _BadCount,
    call: _Call,
    node: exp.Expr,
    select: exp.Select,
) -> FfrwdError:
    """A count rule's rejection, anchored on the argument that caused it.

    The offending option is normally one the query wrote, and that is the
    token worth pointing at; a rule can only reject a DEFAULT if the table
    itself is wrong, so falling back to the call keeps that case anchored
    rather than unanchored.
    """
    written = next((arg for arg in call.named if arg.name == bad.option), None)
    anchor = written.value if written is not None else node
    return _error(
        ErrorCode.FILTER_OPTION_TYPE,
        f"option '{bad.option}' of filter '{spec.name}' decides how many "
        f"streams the call returns, so it must be {bad.expected}, "
        f"got {bad.value!r}",
        anchor,
        fallback=select,
        hint=bad.hint,
    )


def _reject_passthrough_args(
    name: str,
    kinds: list[str],
    call: _Call,
    node: exp.Expr,
) -> None:
    """No function takes a subtitle or data stream.

    An ffmpeg filtergraph carries video and audio only, so a caption or
    timed-metadata stream can never be a filter INPUT — in either tier.
    Tier 1 would otherwise report it as a generic signature mismatch and
    tier 2 as "expects gblur(video)"; both are true but neither says the
    thing that actually matters, which is that no signature could ever
    accept it. ``ParamKind`` and ``DynamicFilter.inputs`` are deliberately
    left alone ("ParamKind is UNCHANGED"), so this is the one
    place that knows it.
    """
    for position, kind in enumerate(kinds):
        if kind not in _PASSTHROUGH_ONLY:
            continue
        raise _error(
            ErrorCode.UDF_ARG_TYPE,
            f"{name}() cannot take a {kind} stream: {kind} streams cannot be "
            "filtered, only selected",
            call.args[position],
            fallback=node,
            hint=_PASSTHROUGH_HINT,
        )


def _bad_streams(
    call: _Call,
    node: exp.Expr,
    select: exp.Select,
    expected: list[StreamType],
    got: list[str],
    *,
    twin_stem: str | None = None,
) -> FfrwdError:
    """The stream-signature rejection — UDF_ARG_TYPE's remaining job.

    Option problems never reach here: they are ``UNKNOWN_FILTER_OPTION`` /
    ``FILTER_OPTION_TYPE`` uniformly, positional or named. ``twin_stem``,
    when given, names the bare video filter whose audio twin the caller
    hand-spelled -- the hint then points at calling the stem instead of
    repeating the ordinary calling-convention reminder.
    """
    shown = call.display
    hint = (
        f"call {twin_stem}(...) on either kind: the compiler picks the "
        f"audio twin, {shown}, from the column's type"
        if twin_stem is not None
        else f"stream inputs come first, then options in the filter's own order, "
        f"then named options: {shown}({', '.join(expected)}, <option>, "
        f"<option> => <value>)"
    )
    return _error(
        ErrorCode.UDF_ARG_TYPE,
        f"{shown}() is an ffmpeg filter: it takes {', '.join(expected)} as its "
        f"stream input{'' if len(expected) == 1 else 's'}, "
        f"got ({', '.join(got) or 'nothing'})",
        node,
        fallback=select,
        hint=hint,
    )


def _reject_null_stream(
    display: str, arg: exp.Expr, select: exp.Select
) -> None:
    """A NULL where a stream input belongs: absence has no stream to offer."""
    inner = _unwrap(arg)
    if not isinstance(inner, exp.Null):
        return
    variable = null_variable(inner)
    if variable is not None:
        line, col = _pos(inner, select)
        raise unset_error(
            ErrorCode.UDF_ARG_TYPE,
            variable,
            what=f"{display}() needs a stream in this position",
            line=line,
            col=col,
        )
    raise _error(
        ErrorCode.UDF_ARG_TYPE,
        f"{display}() takes a stream in this position, got NULL",
        inner,
        fallback=select,
        hint="a stream input cannot be absent; pass one, e.g. f.video[1]",
    )


def _zip_length(
    name: str,
    node: exp.Expr,
    arg_nodes: list[exp.Expr],
    streams: dict[int, _Value],
    select: exp.Select,
    rows: int | None = None,
) -> int | None:
    """The element count this call expands to, or None if it expands to one.

    Arrays zip (no cross products): they must all have the same length, and
    scalar arguments repeat into every element.

    `rows` is how many rows an argument that READS one runs over -- a
    filter option computed per row makes one node per row, the way an
    array argument makes one per element. An array and a per-row option in
    the same call zip too: both count the same relation, so a disagreement
    is the same mismatch a pair of arrays of different lengths is.
    """
    first: tuple[int, int] | None = None  # (argument position, length)
    for position, value in sorted(streams.items()):
        if not value.is_array:
            continue
        length = len(value.streams)
        if first is None:
            first = (position, length)
            continue
        if length == first[1]:
            continue
        raise _error(
            ErrorCode.BROADCAST_MISMATCH,
            f"{name}() cannot broadcast over arrays of different lengths: "
            f"{_sql_text(arg_nodes[first[0]])} has {_stream_count(first[1])}, "
            f"{_sql_text(arg_nodes[position])} has {_stream_count(length)}",
            node,
            fallback=select,
            hint=_ZIP_HINT,
        )
    if first is None:
        return rows
    if rows is not None and rows != first[1]:
        raise _error(
            ErrorCode.BROADCAST_MISMATCH,
            f"{name}() cannot broadcast over arrays of different lengths: "
            f"{_sql_text(arg_nodes[first[0]])} has "
            f"{_stream_count(first[1])}, and its options are read once per "
            f"row over {rows} rows",
            node,
            fallback=select,
            hint=_ZIP_HINT,
        )
    return first[1]


def _expand_call(
    name: str,
    node: exp.Expr,
    arg_nodes: list[exp.Expr],
    select: exp.Select,
    *,
    streams: dict[int, _Value],
    literals: dict[int, object],
    arity: int,
    positions: list[int],
    returns: StreamType,
    build: Callable[[list[object], int], FrameRef],
    rows: int | None = None,
) -> _Value:
    """Broadcast `build` over the array arguments, if there are any.

    Type-driven and tier-agnostic: `positions` is where the stream
    arguments are (always the LEADING positions, from the pad signature or
    from an N-input call's own count) and `build` is what turns one
    element's argument values into a subgraph. `build` also gets the
    ELEMENT INDEX, which is what lets a filter option computed per row
    pick out the row that element came from.

    `rows` broadcasts over the RELATION rather than over an array: a call
    whose options are read per row is one node per row, even where every
    stream argument it takes is a single stream.
    """
    length = _zip_length(name, node, arg_nodes, streams, select, rows)
    expanded: list[_Stream] = []
    for element in range(1 if length is None else length):
        values: list[object] = [
            streams[position].at(element).ref
            if position in streams
            else literals[position]
            for position in range(arity)
        ]
        # A single-stream-input function is 1:1, so its result inherits
        # that input's provenance unconditionally. A call over two or more
        # streams (amix, overlay, xfade) is a join like concat's: it
        # threads provenance only when every input agrees
        # (`_agreed_source`).
        if len(positions) == 1:
            source = streams[positions[0]].at(element).source
        elif len(positions) >= 2:
            source = _agreed_source([streams[p].at(element) for p in positions])
        else:
            source = None
        expanded.append(_Stream(ref=build(values, element), type=returns, source=source))
    if length is None:
        return _scalar(expanded[0])
    return _array(returns, expanded)
