"""What stands in for a row a join did not match.

A FULL or LEFT join leaves gaps, and ``COALESCE(<row alias>, <fill>)`` puts a
generated track in one. The fill inherits what it can from the row that DID
match on the same result row, so a silence-filled French mix stays French.
"""

from __future__ import annotations

from sqlglot import exp

from ffrwd.bindings import (
    _Env,
    _RowBinding,
    _RowRelation,
    _RowTuple,
    _track_of,
    _TrackRow,
)
from ffrwd.errors import ErrorCode
from ffrwd.expressions import _error, _unwrap
from ffrwd.ir import StreamType
from ffrwd.parser import FILTER_NAMESPACE, MACRO_NAMESPACE, ROW_STREAM
from ffrwd.parser import _ident_name as _fold

# The fill each track type takes when an outer join leaves a gap. Quoted
# verbatim in the NULL-track hint, so it is spelled the way a user would paste
# it. `data` is absent deliberately: nothing generates a data track, so there
# is no fill to suggest.
_FILL_SPELLINGS: dict[StreamType, str] = {
    "audio": f"{FILTER_NAMESPACE}.anullsrc()",
    "video": f"{FILTER_NAMESPACE}.color()",
    "subtitle": f"{MACRO_NAMESPACE}.empty_captions()",
}


_COALESCE_HINT = (
    "COALESCE fills an outer join's gaps: COALESCE(b, "
    f"{FILTER_NAMESPACE}.anullsrc(duration => 2)) for audio, "
    f"{FILTER_NAMESPACE}.color() for video, "
    f"{MACRO_NAMESPACE}.empty_captions() for captions"
)

def _paired_row(
    relation: _RowRelation,
    row: _RowTuple,
    alias: str,
) -> tuple[str | None, _TrackRow | None]:
    """The counterpart of a gap: the first row table that DID match here.

    A fill's provenance is the paired (non-NULL counterpart) row's metadata,
    and its inherited options come from that same row, so a silence-filled
    French mix stays French.
    Relation order (FROM order) breaks the tie when three tables joined.
    """
    for other in relation.aliases:
        if other == alias:
            continue
        track = _track_of(row, other)
        if track is not None:
            return other, track
    return None, None


def _coalesce_arguments(node: exp.Expr, select: exp.Select) -> list[exp.Expr]:
    """COALESCE's two arguments, or a rejection. Creates no nodes, which
    is what lets :meth:`_classify` read the type without lowering it."""
    arguments = [
        argument
        for argument in [node.this, *node.expressions]
        if isinstance(argument, exp.Expr)
    ]
    if len(arguments) != 2:
        raise _error(
            ErrorCode.UNSUPPORTED_SQL,
            f"COALESCE takes a track column and one fill, got "
            f"{len(arguments)} argument{'' if len(arguments) == 1 else 's'}",
            node,
            fallback=select,
            hint=_COALESCE_HINT,
        )
    return arguments


def _coalesce_binding(argument: exp.Expr, env: _Env) -> _RowBinding | None:
    """The track-row table a bare ``<alias>`` first argument names, whose
    own relation says which rows are gaps. None for every other nullable
    cell, which carries its gaps in the value itself."""
    column = _unwrap(argument)
    if not isinstance(column, exp.Column):
        return None
    table_node = column.args.get("table")
    binding = env.bindings.get(_fold(table_node)) if table_node is not None else None
    if not isinstance(binding, _RowBinding) or _fold(column.this) != ROW_STREAM:
        return None
    return binding


def _fill_hint(kind: StreamType, label: str) -> str:
    spelling = _FILL_SPELLINGS.get(kind)
    if spelling is None:
        return (
            f"nothing generates a {kind} track, so there is no fill "
            f"for '{label}'; select it from a "
            "join that always matches"
        )
    return (
        f"the fill for a {kind} track is {spelling}; its options "
        "inherit from the paired row unless you give them"
    )


def _inherited_fill_options(
    stream_type: StreamType, paired: _TrackRow | None
) -> dict[str, object]:
    """What the fill copies from the row it stands beside.

    Audio inherits DURATION only in v1 — a silent track's sample rate and
    layout are ffmpeg's own defaults, and amix resamples anyway, so
    inventing them would put options in the command nobody wrote. Video
    inherits size, rate and duration, because a black frame of the wrong
    size or rate is not a stand-in for the picture that is missing.
    An option the query set explicitly always wins (the caller only fills
    the ones it did not).
    """
    if paired is None:
        return {}
    columns = paired.columns
    if stream_type == "audio":
        return {"duration": columns.get("duration")}
    if stream_type == "video":
        width = columns.get("width")
        height = columns.get("height")
        size = (
            f"{int(width)}x{int(height)}"
            if isinstance(width, int | float) and isinstance(height, int | float)
            else None
        )
        return {
            "size": size,
            "rate": columns.get("fps"),
            "duration": columns.get("duration"),
        }
    return {}
