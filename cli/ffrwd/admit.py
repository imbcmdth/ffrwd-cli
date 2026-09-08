"""The checks a lowered construct has to pass, decided from its arguments.

Each takes what it judges and either returns or raises; none reads the graph
being built, which is what will let them run as one admission pass over it.
Together in one module so a rule written for one construct is visible to the
next, instead of a check in one method being invisible to a check in another.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlglot import exp

from ffrwd.bindings import _Env, _has_track_rows, _RowBinding
from ffrwd.destinations import _PER_TRACK_OPTION_HINT
from ffrwd.errors import ErrorCode, FfrwdError
from ffrwd.expressions import _coalesce_label, _error, _unwrap
from ffrwd.filter_options import (
    _ENABLE,
    REQUIRED_OPTIONS,
    _enable_value,
    _NamedArg,
    _option_hint,
    _option_value,
)
from ffrwd.functions import WASM_STREAM_NAMES, Annotation, WasmFunction
from ffrwd.ir import Output
from ffrwd.modules import (
    _annotation_fields,
    _annotation_matches,
    _vector_field,
    _written_json_fields,
)
from ffrwd.parser import RawSink, _pos, null_variable, star_node
from ffrwd.registry import FilterOption
from ffrwd.sink import SINK_OPTIONS
from ffrwd.types import (
    CONTAINER_READONLY_FIELDS,
    DISPOSITION_COLUMN,
    DISPOSITION_KEYS,
    is_array,
)
from ffrwd.values import _PASSTHROUGH_ONLY, _Column, _signature, _stream_count, _Value
from ffrwd.vars import unset_error
from ffrwd.wasm import (
    ANNOTATION_TYPES,
    Described,
    DescribedFunction,
    rows_arms,
    rows_vector_dims,
)

if TYPE_CHECKING:
    from ffrwd.destinations import _VariantRow
    from ffrwd.lower import _Call


def _check_per_track_options(
    options: dict[str, object],
    option_nodes: dict[str, exp.Expr],
    outputs: list[Output],
    raw: RawSink,
) -> None:
    """A per-row option against the tracks it binds to, one apiece.

    The rows were gathered into this destination, so it holds one track
    per row of the option's own scope -- and a count that does not line up
    is a query saying two different things about how many renditions it
    writes.
    """
    for name, value in options.items():
        if not isinstance(value, list):
            continue
        scope = SINK_OPTIONS[name].scope
        tracks = sum(1 for output in outputs if output.type == scope)
        if tracks == len(value):
            continue
        anchor = option_nodes.get(name)
        line, col = _pos(anchor, raw.path_node) if anchor else _pos(raw.path_node)
        raise FfrwdError(
            ErrorCode.ROW_COUNT_MISMATCH,
            f"sink option {name!r} is read once per row over "
            f"{len(value)} rows, and this destination has "
            f"{tracks} {scope} track{'' if tracks == 1 else 's'}",
            line=line,
            col=col,
            hint=_PER_TRACK_OPTION_HINT,
        )


def _check_concat_columns(
    branches: list[exp.Select], flattened: list[list[_Column]]
) -> None:
    """No UNION ALL branch may carry a subtitle/data column.

    ``concat`` is a filtergraph filter and takes ``v`` video plus ``a``
    audio pads — there is no ``s``/``d`` half — so a caption column in a
    concatenated branch has nowhere to go. Checked before
    :meth:`_check_concat_signature` so the rejection names the real reason
    rather than a column-count mismatch.
    """
    for index, columns in enumerate(flattened):
        for column in columns:
            if column.value.type in _PASSTHROUGH_ONLY:
                raise _error(
                    ErrorCode.UNSUPPORTED_SQL,
                    f"UNION ALL concatenates video and audio only: branch "
                    f"{index + 1} selects a {column.value.type} stream",
                    branches[index],
                    hint="select subtitle and data streams outside the UNION ALL "
                    "(they are copied, never concatenated)",
                )


def _check_concat_signature(
    branches: list[exp.Select],
    lowered: list[list[_Column]],
    flattened: list[list[_Column]],
) -> None:
    """Every UNION ALL branch must agree on column count, types and order.

    On the FLATTENED signature: an array column contributes one concat
    column per element, so branches must agree on element counts too. The
    message renders each column as written (``audio[2]`` for an array), so
    a pure length mismatch reads as one.
    """
    expected = [column.value.type for column in flattened[0]]
    for index in range(1, len(flattened)):
        got = [column.value.type for column in flattened[index]]
        if got == expected:
            continue
        raise _error(
            ErrorCode.CONCAT_MISMATCH,
            "UNION ALL branches must select the same stream types in the same "
            f"order: branch 1 selects ({_signature(lowered[0])}), "
            f"branch {index + 1} selects ({_signature(lowered[index])})",
            branches[index],
            hint="ffmpeg concat needs identical segments; reorder or add columns",
        )


def _check_tag_key(
    key: str, anchor: exp.Expr, env: _Env, select: exp.Select
) -> None:
    """A tags field names a tag KEY, never a probed field or the flag map.

    The reserved set is the read-only field names of whatever the column
    sits over: the file reports those, so a query cannot claim them.
    """
    if key == DISPOSITION_COLUMN:
        raise _error(
            ErrorCode.UNSUPPORTED_SQL,
            f"'{DISPOSITION_COLUMN}' is the stream's flag map, not a tag",
            anchor,
            fallback=select,
            hint="write it as its own column, e.g. "
            f"'{DISPOSITION_KEYS[0]}' AS {DISPOSITION_COLUMN}",
        )
    if _has_track_rows(env) and not env.grouped:
        what = "track row"
        reserved = frozenset(
            name
            for binding in env.bindings.values()
            if isinstance(binding, _RowBinding)
            for name in binding.readonly
        )
    else:
        what = "container"
        reserved = CONTAINER_READONLY_FIELDS
    if key not in reserved:
        return
    raise _error(
        ErrorCode.UNSUPPORTED_SQL,
        f"'{key}' is a probed field of the {what}, not something a query "
        "can set",
        anchor,
        fallback=select,
        hint=f"the file reports {key}; a tags field is a free-form key, "
        "e.g. STRUCT('eng' AS language) AS tags",
    )


def _check_star_table_mode(anchor: exp.Expr, select: exp.Select) -> None:
    """EXCEPT/REPLACE narrow a MEDIA star's stream expansion; a table
    query's star prints record fields instead, which they do not reach."""
    star = star_node(anchor)
    if star is not None and (star.args.get("except_") or star.args.get("replace")):
        raise _error(
            ErrorCode.UNSUPPORTED_SQL,
            "EXCEPT/REPLACE are not supported on a table query's '*'",
            anchor,
            fallback=select,
            hint="write the columns out, or drop the modifier",
        )


def _check_rows_schema(
    declared: WasmFunction,
    annotation: Annotation,
    arms: tuple[tuple[tuple[str, str], ...], ...] | None,
    *,
    reads: bool,
    node: exp.Expr,
    select: exp.Select,
) -> None:
    """One end of a rows declaration against the schema the module publishes.

    The same field-for-field match a per-frame annotation gets, run twice:
    once for the rows the module reads, once for the rows it writes.
    """
    side = "reads" if reads else "writes"
    if arms is None:
        raise _error(
            ErrorCode.UNSUPPORTED_SQL,
            f"the module '{declared.module}' names no shape for the rows "
            f"it {side}",
            node,
            fallback=select,
            hint="a rows function needs a module that publishes both row "
            "schemas; rebuild it declaring them",
        )
    fields = _annotation_fields(annotation)
    if any(_annotation_matches(fields, arm) for arm in arms):
        return
    raise _error(
        ErrorCode.UDF_ARG_TYPE,
        f"function '{declared.name}' declares the rows it {side} as "
        f"{annotation.written}, and the module '{declared.module}' {side} "
        f"{' or '.join(_written_json_fields(arm) for arm in arms)}",
        node,
        fallback=select,
        hint="a rows record names the module's own row columns, with a "
        "type each value fits",
    )


def _check_rows_argument(
    declared: WasmFunction,
    emitted: Annotation,
    written: exp.Expr,
    node: exp.Expr,
    select: exp.Select,
) -> None:
    """That the rows arriving are the rows this function declares reading."""
    if _annotation_fields(declared.rows_param) == _annotation_fields(emitted):
        return
    raise _error(
        ErrorCode.UDF_ARG_TYPE,
        f"{declared.name}() reads '{declared.rows_param.name}' as "
        f"{declared.rows_param.written}, and its argument carries "
        f"'{emitted.name}' as {emitted.written}",
        written,
        fallback=node,
        hint="the two annotation records have to name the same fields, "
        "with the same types",
    )


def _check_coalesce_width(
    value: _Value, cardinality: int, node: exp.Expr, select: exp.Select
) -> None:
    """A COALESCE argument is one cell per row, or one stream every row
    repeats -- nothing in between."""
    if len(value.streams) in (1, cardinality):
        return
    raise _error(
        ErrorCode.ROW_COUNT_MISMATCH,
        f"'{_coalesce_label(node)}' carries {len(value.streams)} streams, "
        f"and COALESCE is over {cardinality} rows",
        node,
        fallback=select,
        hint="every argument is one cell per row, or one stream the rows "
        "repeat",
    )


def _check_sink_shape(
    declared: WasmFunction,
    described: Described,
    node: exp.Expr,
    select: exp.Select,
) -> None:
    """What the signature says it reads against what the module declares.

    Per kind, since a sink reads each independently. A declaration naming
    one stream of a kind always hands over one, so a module reading one,
    many or any accepts it; anything that can hand over SEVERAL -- an
    array, or several parameters of one kind -- needs a module that reads
    many or any. A module reading ANY works without the kind entirely, so
    a declaration naming none of it is a shape it accepts too. The refusal
    lands here, at the call, where the run-time one it forestalls can be
    said before anything runs.
    """
    for kind in WASM_STREAM_NAMES:
        params = [
            param
            for param, declared_kind in zip(
                declared.stream_params, declared.stream_kinds, strict=True
            )
            if declared_kind == kind
        ]
        reads = described.sink_streams(kind)
        if not params:
            if reads not in ("none", "any"):
                raise _error(
                    ErrorCode.UNSUPPORTED_SQL,
                    f"the module '{declared.module}' reads {reads} "
                    f"{kind} stream(s), and '{declared.name}' declares no "
                    f"{WASM_STREAM_NAMES[kind]} parameter",
                    node,
                    fallback=select,
                    hint=f"declare a {WASM_STREAM_NAMES[kind]} parameter "
                    f"before the module's value parameters",
                )
            continue
        several = len(params) > 1 or any(is_array(p.type) for p in params)
        if reads == "none" or (several and reads not in ("many", "any")):
            written = ", ".join(f"{p.name} {p.type}" for p in params)
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"function '{declared.name}' declares ({written}), and the "
                f"module '{declared.module}' reads {reads} {kind} stream"
                f"{'' if reads == 'one' else 's'}",
                node,
                fallback=select,
                hint="declare what the module reads: "
                + (
                    f"no {WASM_STREAM_NAMES[kind]} parameter"
                    if reads == "none"
                    else f"one {WASM_STREAM_NAMES[kind]} parameter"
                ),
            )


def _check_stream_arity(
    declared: WasmFunction,
    described: Described,
    node: exp.Expr,
    select: exp.Select,
) -> None:
    """The streams the signature declares against the streams the module reads.

    A module reading several at once is a WINDOWED-interface export that
    hands one frame back per frame in: a windowing one has no answer for
    which frame of which input pairs with which, and a per-frame one
    predates the declaration entirely.
    """
    reads = described.inputs
    if reads > 1 and not described.windowed:
        raise _error(
            ErrorCode.UNSUPPORTED_SQL,
            f"the module '{declared.module}' reads {reads} streams, and "
            "declares the per-frame interface",
            node,
            fallback=select,
            hint="rebuild the module against the windowed interface, which "
            "is where a module reads more than one stream",
        )
    if reads > 1 and (described.window != 1 or described.stride != 1):
        raise _error(
            ErrorCode.UNSUPPORTED_SQL,
            f"the module '{declared.module}' reads {reads} streams over a "
            f"window of {described.window} every {described.stride}",
            node,
            fallback=select,
            hint="a module reading several streams takes one frame off each "
            "and hands one back; rebuild it with a window and stride of 1",
        )
    if declared.stream_arity != reads:
        raise _error(
            ErrorCode.UNSUPPORTED_SQL,
            f"function '{declared.name}' declares {declared.stream_arity} "
            f"stream parameter{'' if declared.stream_arity == 1 else 's'}, "
            f"and the module '{declared.module}' reads {reads}",
            node,
            fallback=select,
            hint=f"declare one {declared.returns} parameter per stream the "
            f"module reads, {reads} of them, before its value parameters",
        )


def _check_annotation_schema(
    declared: WasmFunction,
    annotation: Annotation,
    described: Described,
    node: exp.Expr,
    select: exp.Select,
) -> None:
    """The declared annotation record against the rows the module says it emits.

    Field for field, by name and by type. A module publishing SEVERAL row
    shapes is matched against each in turn, and one arm fitting is enough.
    A module that declares no rows has nothing to annotate with, so
    declaring an annotation return over one is a rejection of its own.
    """
    arms = rows_arms(described)
    if arms is None:
        raise _error(
            ErrorCode.UNSUPPORTED_SQL,
            f"function '{declared.name}' returns the annotation column "
            f"'{annotation.name}', and the module '{declared.module}' emits no rows",
            node,
            fallback=select,
            hint=f"declare '{declared.name}' as RETURNS {declared.returns}; a "
            "module that reads nothing off its frames has no annotations "
            "to return",
        )
    fields = _annotation_fields(annotation)
    if any(_annotation_matches(fields, arm) for arm in arms):
        return
    raise _error(
        ErrorCode.UDF_ARG_TYPE,
        f"function '{declared.name}' declares '{annotation.name}' as "
        f"{annotation.written}, and the module '{declared.module}' "
        f"emits {' or '.join(_written_json_fields(arm) for arm in arms)}",
        node,
        fallback=select,
        hint="an annotation record names the module's own row columns, "
        "with a type each value fits",
    )


def _check_vector_dims(
    declared: WasmFunction,
    annotation: Annotation,
    described: Described,
    node: exp.Expr,
    select: exp.Select,
) -> None:
    """A vector-typed annotation field's length, fixed by the module's schema.

    The declaration says a field is a vector; only the module's own
    schema says how many numbers one carries, and that is what tags the
    track the field becomes (:meth:`_rows_output`). A field the schema
    does not fix a length for is refused here, rather than minting a
    track later with nothing to tag it.
    """
    field = _vector_field(annotation)
    if field is None:
        return
    if rows_vector_dims(described, field) is None:
        raise _error(
            ErrorCode.UNSUPPORTED_SQL,
            f"function '{declared.name}' declares '{field}' as vector, and "
            f"the module '{declared.module}' does not fix its length",
            node,
            fallback=select,
            hint="a vector track needs its dimension: declare minItems and "
            "maxItems on the field",
        )


def _check_wasm_result_type(
    declared: WasmFunction,
    described: DescribedFunction,
    node: exp.Expr,
    select: exp.Select,
) -> None:
    """RETURNS against the module's own ``result_schema``, once per declaration."""
    schema = described.result_schema
    wanted = schema.get("type") if isinstance(schema, dict) else None
    if isinstance(wanted, str) and wanted in ANNOTATION_TYPES.get(declared.returns, ()):
        return
    raise _error(
        ErrorCode.UDF_ARG_TYPE,
        f"function '{declared.name}' declares RETURNS {declared.returns}, and "
        f"the module's function '{declared.export}' returns "
        f"{wanted if isinstance(wanted, str) else 'nothing'}",
        node,
        fallback=select,
        hint="a value-returning wasm function's RETURNS matches what the "
        "module's function declares",
    )


def _check_row_sink_arity(
    declared: WasmFunction,
    described: Described,
    rows: list[_VariantRow],
    node: exp.Expr,
    select: exp.Select,
) -> None:
    """A sink whose arity holds one row at a time, handed several.

    `video_streams`/`audio_streams` say how many pads of each kind a
    sink reads; neither declaring "many" or "any" means it is
    file-shaped, and a multi-row relation reaching it is the one-row
    rule again (:meth:`_check_one_row_per_file`, which a row-reading
    sink skips), this time naming the sink instead of a path.
    """
    if len(rows) <= 1:
        return
    if described.video_streams in ("many", "any") or described.audio_streams in (
        "many",
        "any",
    ):
        return
    raise _error(
        ErrorCode.ROW_COUNT_MISMATCH,
        f"this query has {len(rows)} rows, and '{declared.name}()' reads one",
        node,
        fallback=select,
        hint="narrow to one row with WHERE or ORDER BY ... LIMIT, or use "
        "a sink whose module reads many",
    )


def _check_variadic_count(
    option_name: str | None,
    count: int,
    args: dict[str, object],
    call: _Call,
    node: exp.Expr,
    select: exp.Select,
    hint: str,
) -> None:
    """A WRITTEN count option must agree with the array's length.

    Unlike the positional call (:meth:`_lower_n_input_call`), there is no
    "did you forget to write it" ambiguity here: the array's length IS the
    count, full stop, so an unwritten option is simply set to it below --
    only a value the query itself wrote can possibly disagree.
    """
    if option_name is None:
        return
    written = args.get(option_name)
    if not isinstance(written, (int, float)) or isinstance(written, bool):
        return
    if int(written) == count:
        return
    anchor = next((arg.value for arg in call.named if arg.name == option_name), node)
    raise _error(
        ErrorCode.UDF_ARG_TYPE,
        f"{call.display}() was given {_stream_count(count)} but its "
        f"'{option_name}' option says {int(written)}",
        anchor,
        fallback=select,
        hint=hint,
    )


def _check_named_args(
    filter_name: str,
    options: dict[str, FilterOption],
    named: list[_NamedArg],
    call: exp.Expr,
    *,
    owner: str,
    occupied: set[str],
    timeline: bool = False,
    dropped: dict[str, exp.Expr] | None = None,
) -> dict[str, object]:
    """Validate every named argument against `options`, in written order.

    A NULL value -- an unset variable's, or a literal one -- means the
    option is not written: it is recorded in `dropped` (when the caller
    passes one) and never reaches `_option_value`. An UNKNOWN name still
    rejects whatever its value, NULL included: the name is wrong before
    the value matters.

    `occupied` holds the option names this call already bound
    POSITIONALLY, so ``crop(f, 100, 50, 10, 20, out_w => 5)`` reads as the
    conflict it is rather than silently overriding what the call itself
    said. A collision is ``FILTER_OPTION_TYPE``, an option problem like any
    other, and the fix is to drop one of the two spellings.

    The collision check comes FIRST so the message names the conflict
    rather than whatever the registry would say about the name.

    `timeline` is the target's ``DynamicFilter.timeline`` flag, and it is a
    PARAMETER because this method cannot look filters up: every caller
    already holds the registry entry (or, for a generated source, knows
    there is no such field to hold — a source is never timeline-capable, so
    the default rejects). It admits ``enable`` BEFORE `options` is consulted
: ffmpeg implements ``enable`` in the filter framework, so
    it is in no filter's option table and a registry lookup would always
    call it unknown.
    """
    checked: dict[str, object] = {}
    for arg in named:
        if arg.name in occupied:
            raise _error(
                ErrorCode.FILTER_OPTION_TYPE,
                f"option '{arg.name}' of filter '{filter_name}' is already set "
                f"positionally by {owner}()",
                arg.value,
                fallback=call,
                hint="a named argument never overrides what the call itself "
                "set; drop one of the two spellings",
            )
        is_null = isinstance(_unwrap(arg.value), exp.Null)
        if arg.name == _ENABLE:
            if is_null:
                if dropped is not None:
                    dropped[_ENABLE] = arg.value
                continue
            checked[_ENABLE] = _enable_value(filter_name, arg, call, timeline)
            continue
        option = options.get(arg.name)
        if option is None:
            raise _error(
                ErrorCode.UNKNOWN_FILTER_OPTION,
                f"filter '{filter_name}' has no option '{arg.name}'",
                arg.value,
                fallback=call,
                hint=_option_hint(arg.name, options),
            )
        if is_null:
            if dropped is not None:
                dropped[arg.name] = arg.value
            continue
        checked[arg.name] = _option_value(filter_name, option, arg, call)
    return checked


def _check_required_options(
    filter_name: str,
    bound: dict[str, object],
    dropped: dict[str, exp.Expr],
    node: exp.Expr,
    select: exp.Select,
) -> None:
    """The curated :data:`REQUIRED_OPTIONS` check, on what was WRITTEN.

    A NULL dropped the option before this runs, so an unset variable and
    an omitted option fail the same way -- ffmpeg's init() would refuse
    both at run time, and this says so at compile time, naming the
    variable when the NULL came from one.
    """
    required: list[tuple[tuple[str, ...], str]] = [
        (group, "") for group in REQUIRED_OPTIONS.get(filter_name, ())
    ]
    if filter_name == "xfade" and bound.get("transition") == "custom":
        required.append((("expr",), " when transition is 'custom'"))
    for group, because in required:
        if any(option in bound for option in group):
            continue
        option_name = next((o for o in group if o in dropped), None)
        if option_name is not None:
            anchor = dropped[option_name]
            variable = null_variable(_unwrap(anchor))
            if variable is not None:
                line, col = _pos(anchor, node)
                raise unset_error(
                    ErrorCode.FILTER_OPTION_TYPE,
                    variable,
                    what=f"option '{option_name}' of filter "
                    f"'{filter_name}' is required{because}",
                    line=line,
                    col=col,
                )
            raise _error(
                ErrorCode.FILTER_OPTION_TYPE,
                f"option '{option_name}' of filter '{filter_name}' is "
                f"required{because}, got NULL",
                anchor,
                fallback=node,
                hint="NULL is absence, and this filter cannot run "
                "without the option; write a value",
            )
        spelled = " or ".join(f"'{option}'" for option in group)
        raise _error(
            ErrorCode.FILTER_OPTION_TYPE,
            f"filter '{filter_name}' requires option {spelled}{because}",
            node,
            fallback=select,
            hint=f"ffmpeg would refuse the filter at run time; write "
            f"{group[0]} => <value>",
        )
