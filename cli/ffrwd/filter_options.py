"""A written option against the filter's own option table.

ffmpeg's AVOptions, introspected by :mod:`ffrwd.registry`, are what a named
argument is judged against: the name has to exist, the value has to fit the
option's type and range, and a filter that cannot run without an option says
so before the command is printed.
"""

from __future__ import annotations

import difflib
from collections.abc import Iterable
from dataclasses import dataclass

from sqlglot import exp

from ffrwd.errors import ErrorCode, FfrwdError
from ffrwd.expressions import _describe, _error, _unwrap
from ffrwd.registry import FilterOption

# `enable` is FRAMEWORK-level: ffmpeg implements it in the filter framework,
# not in any filter, so it never appears in a filter's `-help` AVOptions and no
# options table can contain it. Which filters honour it is the `T` column of
# `ffmpeg -filters`, captured as DynamicFilter.timeline; that flag admits the
# name here.
_ENABLE = "enable"


_ENABLE_HINT = (
    "enable takes a single-quoted ffmpeg timeline expression over t (seconds), "
    "n (frame number) or pos, e.g. enable => 'between(t,2,5)'"
)


_NO_TIMELINE_HINT = (
    "enable is only accepted by filters your ffmpeg flags with timeline support "
    "(the T column of `ffmpeg -filters`: gblur has it, scale does not); drop it, "
    "or express the timing with a WHERE window over the input"
)


# Options a filter cannot run without. ffmpeg has no required-option
# metadata -- AVOption carries no such flag; each filter enforces its own in
# init() -- so this table is hand-kept, curated knowledge like MACROS. Each
# value is a tuple of groups; a group is satisfied when any one of its names
# is written (drawtext runs on `text` OR `textfile`). xfade's conditional
# requirement -- `expr`, only when `transition` is 'custom' -- cannot be a
# name list and lives in `_check_required_options` directly.
REQUIRED_OPTIONS: dict[str, tuple[tuple[str, ...], ...]] = {
    "subtitles": (("filename",),),
    "lut3d": (("file",),),
    "frei0r": (("filter_name",),),
    # plugin only matters when the library holds more than one; file always.
    "ladspa": (("file",),),
    "movie": (("filename",),),
    "amovie": (("filename",),),
    "drawtext": (("text", "textfile"),),
}


# Longest option/constant list a hint or message renders before it stops
# counting (xfade's `transition` alone has 59 constants).
_MAX_LISTED = 12


@dataclass(frozen=True)
class _NamedArg:
    """One ``name => value`` call argument.

    `name` is verbatim (ffmpeg AVOption names are case-sensitive) and `value` is
    the raw sqlglot node — the option table this is checked against comes from
    the installed ffmpeg, so nothing is interpreted before the registry says
    what the option's type is.
    """

    name: str
    value: exp.Expr


def _listed(names: Iterable[str]) -> str:
    """Comma-list at most ``_MAX_LISTED`` names, then count the rest."""
    items = list(names)
    if len(items) <= _MAX_LISTED:
        return ", ".join(items)
    rest = len(items) - _MAX_LISTED
    return ", ".join(items[:_MAX_LISTED]) + f", ... ({rest} more)"


def _option_hint(name: str, options: dict[str, FilterOption]) -> str:
    """Did-you-mean over the filter's REAL option names, else list them."""
    matches = difflib.get_close_matches(name, sorted(options), n=1, cutoff=0.6)
    if matches:
        return f"did you mean {matches[0]} => ...?"
    if not options:
        return "this filter has no options ffrwd can set"
    return "its options: " + _listed(sorted(options))


def _number_text(value: float) -> str:
    """A range bound as ffmpeg meant it: ``1024`` rather than ``1024.0``."""
    if value == int(value):
        return str(int(value))
    return str(value)


def _range_text(option: FilterOption) -> str | None:
    if option.minimum is not None and option.maximum is not None:
        return f"from {_number_text(option.minimum)} to {_number_text(option.maximum)}"
    if option.minimum is not None:
        return f"at least {_number_text(option.minimum)}"
    if option.maximum is not None:
        return f"at most {_number_text(option.maximum)}"
    return None


def _literal_value(node: exp.Expr) -> object | None:
    """A named argument's value as a python scalar, or None if it is not a literal.

    Deliberately separate from :func:`ffrwd.expressions._number`: that raises its own message,
    and an option's expected type is only known after the registry has been
    consulted, so reading the value and judging it are two steps here.
    """
    node = _unwrap(node)
    if isinstance(node, exp.Boolean):
        return bool(node.this)
    negated = False
    if isinstance(node, exp.Neg) and isinstance(node.this, exp.Expr):
        negated = True
        node = _unwrap(node.this)
    if not isinstance(node, exp.Literal):
        return None
    if node.is_string:
        return None if negated else str(node.this)
    try:
        value = node.to_py()
    except (ArithmeticError, TypeError, ValueError):
        return None
    if isinstance(value, bool):  # sqlglot never does this; be explicit anyway
        return None
    number = value if isinstance(value, int) else float(value)
    return -number if negated else number


def _option_got(node: exp.Expr, value: object) -> str:
    """How a rejected option value is echoed back in the message."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)
    if isinstance(value, str):
        return repr(value)
    inner = _unwrap(node)
    if isinstance(inner, exp.Literal):
        # A literal `_literal_value` could not read (sqlglot tokenizes `1e` as a
        # number but `to_py()` raises): echo what was written, not its shape.
        return repr(str(inner.this))
    return _describe(inner)


def _option_error(
    filter_name: str, option: FilterOption, arg: _NamedArg, call: exp.Expr, what: str, hint: str
) -> FfrwdError:
    return _error(
        ErrorCode.FILTER_OPTION_TYPE,
        f"option '{option.name}' of filter '{filter_name}' {what}",
        arg.value,
        fallback=call,
        hint=hint,
    )


def _enable_value(
    filter_name: str, arg: _NamedArg, call: exp.Expr, timeline: bool
) -> str:
    """The timeline ``enable`` expression, or the rejection for this filter.

    Two ways it fails. The filter has no timeline support at all, which is a
    property of the FILTER and so reads as an unknown option on it — flavoured
    with the reason, because "gblur has no option 'enable'" would be a lie
    about gblur. Or the value is not a string: an ffmpeg timeline expression is
    text, and a bare number would silently mean "always on"/"never on" rather
    than the window the writer had in mind.

    The expression's CONTENT is deliberately unchecked: the variable
    vocabulary is per-filter and not introspectable, so it is ffmpeg's to
    validate at run time.
    """
    if not timeline:
        raise _error(
            ErrorCode.UNKNOWN_FILTER_OPTION,
            f"filter '{filter_name}' has no option 'enable': your ffmpeg does "
            f"not flag '{filter_name}' as supporting timeline editing",
            arg.value,
            fallback=call,
            hint=_NO_TIMELINE_HINT,
        )
    value = _literal_value(arg.value)
    if not isinstance(value, str):
        raise _error(
            ErrorCode.FILTER_OPTION_TYPE,
            f"option 'enable' of filter '{filter_name}' expects an ffmpeg "
            f"timeline expression, got {_option_got(arg.value, value)}",
            arg.value,
            fallback=call,
            hint=_ENABLE_HINT,
        )
    return value


def _option_value(
    filter_name: str, option: FilterOption, arg: _NamedArg, call: exp.Expr
) -> object:
    """One named argument's value, checked against its introspected AVOption.

    Numeric AVOptions take a bare number (range checked whenever ffmpeg
    printed a parseable one), booleans take ``true`` / ``false``, an enum
    takes one of its named constants, and everything else takes a string — or
    a bare number, since an ffmpeg option value is text on the command line
    either way and ``duration``/``video_rate``/expression options
    (``xfade``'s ``duration``, ``crop``'s ``x``) are routinely numeric.

    ``binary`` and ``dictionary`` AVOptions are strings here too: ffmpeg reads
    a dictionary as its own ``key=value`` list and a binary as hex, both from
    the same text an option value always is.
    """
    value = _literal_value(arg.value)
    got = _option_got(arg.value, value)
    if option.type == "num":
        if not isinstance(value, int | float) or isinstance(value, bool):
            bounds = _range_text(option)
            raise _option_error(
                filter_name,
                option,
                arg,
                call,
                f"expects a number, got {got}",
                f"write a bare numeric literal ({bounds})" if bounds else
                "write a bare numeric literal, e.g. sigma => 5",
            )
        bounds = _range_text(option)
        below = option.minimum is not None and value < option.minimum
        above = option.maximum is not None and value > option.maximum
        if (below or above) and bounds is not None:
            raise _option_error(
                filter_name,
                option,
                arg,
                call,
                f"accepts a number {bounds}, got {got}",
                f"pick a value {bounds}",
            )
        return value
    if option.type == "bool":
        if not isinstance(value, bool):
            raise _option_error(
                filter_name,
                option,
                arg,
                call,
                f"expects true or false, got {got}",
                "write the bare word true or false, with no quotes",
            )
        return value
    if option.constants:
        if not isinstance(value, str) or value not in option.constants:
            constants = _listed(option.constants)
            matches = (
                difflib.get_close_matches(value, list(option.constants), n=1, cutoff=0.6)
                if isinstance(value, str)
                else []
            )
            raise _option_error(
                filter_name,
                option,
                arg,
                call,
                f"expects one of its named constants ({constants}), got {got}",
                f"did you mean '{matches[0]}'?"
                if matches
                else "the value is a single-quoted constant name, not a number",
            )
        return value
    if isinstance(value, str) or (isinstance(value, int | float) and not isinstance(value, bool)):
        return value
    raise _option_error(
        filter_name,
        option,
        arg,
        call,
        f"expects a string, got {got}",
        "write a single-quoted string literal, e.g. flags => 'lanczos'",
    )
