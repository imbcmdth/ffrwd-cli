"""The typed stream value, and the column a projection lowers to.

Every value flowing through lowering is a stream of a known type -- one of
them, or an array -- carrying the IR ref it renders as and the probed stream
it came from. A branch's SELECT list is a list of these.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from ffrwd.ir import FrameRef, StreamType
from ffrwd.probe import RenditionMeta, StreamMeta
from ffrwd.table import CellValue, StreamCell

# The array-typed pseudo-columns an input exposes, and their element type.
# subtitle/data have the identical array/subscript/splat surface but are
# passthrough-only (see `_PASSTHROUGH_ONLY` below).
_ARRAY_COLUMNS: dict[str, StreamType] = {
    "video": "video",
    "audio": "audio",
    "subtitle": "subtitle",
    "data": "data",
}


_TYPE_MARKERS: dict[StreamType, str] = {
    "video": "v",
    "audio": "a",
    "subtitle": "s",
    "data": "d",
}


# Stream types an ffmpeg filtergraph cannot carry: they may only become an
# Output (a bare `-map`), never a filter argument and never a WHERE trim's
# input.
_PASSTHROUGH_ONLY: frozenset[StreamType] = frozenset({"subtitle", "data"})


@dataclass(frozen=True)
class _Stream:
    """One typed stream: an IR ref, the pad type it carries, and its origin.

    `source` is the probed :class:`~ffrwd.probe.StreamMeta` this stream comes
    from 1:1 — directly (a passthrough subscript) or through a chain of
    single-stream-input filters, the WHERE trim included — and is threaded
    unconditionally. A call over two or more streams (``amix``, ``overlay``)
    and a ``concat`` pad are the other kind of join: each keeps `source` only
    when every stream feeding it agrees on what it says (:func:`ffrwd.lower._agreed_source`);
    otherwise it is None, same as an unprobed input. :func:`ffrwd.lower._provenance` turns
    it into ``Output.metadata``.

    `rendition` is the :class:`~ffrwd.probe.RenditionMeta` a rendition row's
    cell was built from, set only where :meth:`_Lowerer._rendition_row` mints
    the stream. It threads through nothing else: a filter output is a freshly
    built ``_Stream`` that never copies it, so an UNMODIFIED read of a
    rendition row's ``video``/``audio`` cell is the only way to still carry
    it by the time a manifest destination or ``Output.metadata`` reads it.
    """

    ref: FrameRef
    type: StreamType
    source: StreamMeta | None = None
    rendition: RenditionMeta | None = None


# Table mode only: the sentinel `_Stream.ref` for an
# outer join's NULL row, read back by `_value_to_cells`. Never a real
# FrameRef -- every well-formed one is non-empty (a node id or a "src:..."
# ref) -- so there is no ambiguity with an actual stream.
_NULL_STREAM_REF: FrameRef = ""


@dataclass(frozen=True)
class _Value:
    """What every expression lowers to: one stream, or a whole array of them.

    `is_array` is deliberately not ``len(streams) != 1``: a one-element array
    is still an array — it splats, broadcasts and subscripts — and on a
    single-track file that is the ONLY thing separating ``a.audio`` from
    ``a.audio[1]``.
    """

    type: StreamType  # element type; every element of an array agrees on it
    streams: tuple[_Stream, ...]
    is_array: bool

    def at(self, index: int) -> _Stream:
        """Element `index` of an array; the one stream of a scalar (it repeats)."""
        return self.streams[index] if self.is_array else self.streams[0]


def _scalar(stream: _Stream) -> _Value:
    return _Value(type=stream.type, streams=(stream,), is_array=False)


def _array(stream_type: StreamType, streams: Iterable[_Stream]) -> _Value:
    return _Value(type=stream_type, streams=tuple(streams), is_array=True)


def _is_null(value: _Value) -> bool:
    """True for a single NULL cell: a gap, or an aggregate over no rows."""
    if value.is_array or not value.streams:
        return False
    return value.streams[0].ref == _NULL_STREAM_REF


def _writes_nothing(columns: list[_Column]) -> bool:
    """True when every column of a branch is a NULL cell.

    Only a NULL cell counts. A branch with no columns, or one whose column is
    an empty array, is a different shape -- a module sink, a rows file --
    that writes through something other than these columns.
    """
    return bool(columns) and all(_is_null(column.value) for column in columns)


@dataclass(frozen=True)
class _Column:
    """One SELECT column of a branch (or of a CTE body): its name and value.

    An array column carries every one of its streams here, so a CTE records an
    array column's LENGTH statically and a later ``<cte>.<name>[k]`` is
    bounds-checked without re-probing anything.

    `splat` matters only when `value.is_array`: True means the array IS a row
    set (a row alias's stream column, or a call over one) that a table query
    prints one row per element, like :meth:`_Lowerer._value_to_cells` already
    does outside a CTE. False means the array is a single unit -- an
    ``array_agg`` or a bare input array column -- that a table query
    broadcasts as ONE cell instead (see :meth:`_Lowerer._array_cell_broadcast`).
    Ignored for a scalar column.
    """

    name: str | None
    value: _Value
    splat: bool = True


def _signature(columns: list[_Column]) -> str:
    """Branch column types for a CONCAT_MISMATCH message, arrays as ``audio[2]``."""
    parts = [
        f"{column.value.type}[{len(column.value.streams)}]"
        if column.value.is_array
        else column.value.type
        for column in columns
    ]
    return ", ".join(parts) or "nothing"


def _stream_count(count: int) -> str:
    return f"{count} stream" + ("" if count == 1 else "s")

def _stream_to_cell(stream: _Stream) -> CellValue:
    """One stream as a cell, carrying its REF until `_render_specs` runs."""
    if stream.ref == _NULL_STREAM_REF:
        return None
    return StreamCell(type=stream.type, spec=stream.ref)
