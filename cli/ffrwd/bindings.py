"""What a FROM item binds, and the rows behind one.

Four kinds of binding -- an ``input()`` alias, a CTE, a generated source and a
compile-time row table -- plus the joined row set a branch's row tables share
and the environment a branch resolves names against.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from sqlglot import exp

from ffrwd.functions import Annotation
from ffrwd.ir import FrameRef, StreamType
from ffrwd.merge import RowValue
from ffrwd.parser import FILTER_NAMESPACE, RawValuesTable
from ffrwd.table import ArrayCell, CellValue, RecordCell, VectorCell
from ffrwd.types import (
    DISPOSITION_KEYS,
    RECORD_ARRAY_COLUMNS,
    RECORD_ELEMENTS,
    ROW_READONLY_FIELDS,
    ROW_SCHEMAS,
    ROW_STAR_COLUMNS,
    RowColumnType,
)
from ffrwd.values import _Column, _Stream

# The pseudo `_RowBinding.column` a manifest's ABR ladder binds under: one row
# per `RenditionMeta`, read straight off an `input()` alias with no `unnest`
# to ask for it. Its schema is not a view over `container` like ROW_SCHEMAS is
# -- a rendition is not a container array field, only ever a probed fact.
RENDITION_COLUMN = "rendition"


_RENDITION_SCHEMA: dict[str, RowColumnType] = {
    "bandwidth": "number",
    "width": "number",
    "height": "number",
    "codecs": "text",
    "name": "text",
    "language": "text",
}


# Every rendition column is a probed fact, never an assertion a query can make.
_RENDITION_READONLY: frozenset[str] = frozenset(_RENDITION_SCHEMA)


@dataclass(frozen=True)
class _InputBinding:
    """``FROM input('x.mp4') a`` — exposes ``a.video[k]`` / ``a.audio[k]``."""

    alias: str


@dataclass(frozen=True)
class _CteBinding:
    """``FROM <cte>`` — a TABLE of the rows its body produced.

    `columns` is what the body's SELECT list named, each column holding every
    stream it carries. `rows` is the body's ROW count, which a splat array
    column carries one element per; `relation` is the branch's joined row set,
    so a column of that width reads back one element per result row and a
    cross join with a second source repeats it honestly.
    """

    name: str
    columns: tuple[_Column, ...]
    rows: int = 1
    relation: _RowRelation | None = None
    # Scalar columns of the body, name -> one value per body row. Read back
    # by position, the same way a stream column is.
    values: dict[str, tuple[RowValue, ...]] = field(default_factory=dict)
    # Columns whose value is a run-time annotation -- the producer's own
    # rows, or another rows function's result read one module later -- name
    # -> the same (producer, kind, record) triple `_rows_source` returns for
    # the inline spelling. A rows function reading `<cte>.<name>` resolves
    # through here instead of re-lowering the producer, so the producer node
    # is not minted twice.
    rows_columns: dict[str, tuple[FrameRef, StreamType, Annotation]] = field(
        default_factory=dict
    )
    # Columns whose value is a compile-time cue array (a written document,
    # not a module's rows), named so a rows function refuses one by what it
    # is instead of mistaking it for a stream.
    cue_columns: frozenset[str] = frozenset()


@dataclass
class _SourceBinding:
    """``FROM ffmpeg.<source>(...) a`` — exposes ONE statically-typed stream.

    Everything about the stream is known before any projection lowers: the
    registry's :class:`~ffrwd.registry.SourceFilter` says which
    type the source's single output pad carries, so ``a.video[1]``
    (video sources), ``a.audio[1]`` (audio ones), the bare array
    ``a.video`` (length 1, statically), and ``a.*`` are all answered without
    a probe — there is no file to probe, and no ``-i``: the source is a
    ZERO-INPUT filter node.

    `options` is already validated against the source's introspected option
    table (the exact same ``Registry.options`` path a tier-2 call's named
    arguments take), because that happens when the FROM clause binds, not
    when a column is read.

    Mutable on purpose: `ref` memoizes the node, which is minted lazily on
    the FIRST column access and shared by every later one. Fan-out beyond
    that is the split pass's job, exactly as for any other node, so
    ``SELECT a.video[1], hflip(a.video[1]) FROM ffmpeg.testsrc(...) a`` is one
    ``testsrc`` plus a ``split``, never two generators.
    """

    alias: str
    name: str  # the ffmpeg source filter's name, e.g. "testsrc"
    output: StreamType
    options: dict[str, object]
    ref: FrameRef | None = None

    @property
    def display(self) -> str:
        """The source as the user spelled it, for error messages."""
        return f"{FILTER_NAMESPACE}.{self.name}"


@dataclass(frozen=True)
class _TrackRow:
    """One row of an ``unnest`` table: the track, plus its metadata columns.

    `stream` IS the row's stream, and its ``_Stream.source`` is the very
    ``StreamMeta`` `columns` was read from — a row's provenance and its columns
    are the same probed fact, seen twice.

    `kinds` holds every stream of a RENDITION row, by type — a manifest's
    variant may carry video and audio together — with `stream` staying the
    row's primary one (its first video stream, else its first audio one).
    Empty for every other row kind, unnest rows included.
    """

    stream: _Stream
    columns: dict[str, RowValue]
    kinds: dict[StreamType, _Stream] = field(default_factory=dict)


@dataclass(frozen=True)
class _CteRow:
    """One row of a CTE source: which row of the body's row set it is.

    The position is all a result tuple needs: it indexes both the stream
    columns' arrays and the body's value columns, which is what makes
    ``x.n`` read the value this very row computed.
    """

    position: int


# What one result row holds per FROM alias: a track (or a gap, where an outer
# join found no counterpart) for a row table, a position for a CTE source.
_RowTuple = dict[str, "_TrackRow | _CteRow | None"]


@dataclass
class _RowRelation:
    """One branch's joined row set: every row source, aligned.

    `tuples` is the relation itself — one dict per result ROW, mapping each row
    alias to that row's track, or to ``None`` where an outer join left a gap,
    and each CTE alias to the body row it took. All of a branch's row sources
    share this one object, which is what keeps
    ``a`` and ``b`` aligned: element `i` of each is the pair the
    join made, so the existing zip/broadcast machinery wires the right streams
    together without learning that joins exist.

    Row order is the join's, never sorted implicitly: the
    LEFT side's order, then — for a FULL join only — the unmatched right rows
    in their own order. `keys` remembers which columns each side was matched
    on, so a NULL track can say what it failed to match.
    """

    aliases: list[str] = field(default_factory=list)
    tuples: list[_RowTuple] = field(default_factory=list)
    keys: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class _RowBinding:
    """``FROM ..., unnest(<input>.<type>) t`` — a compile-time TABLE.

    `rows` is this alias's column of the branch's joined relation, in ROW
    ORDER: the surviving row set, one entry per result row, ``None`` where an
    outer join found no counterpart. It is what the WHERE predicate and the
    ORDER BY rewrite (both act on the shared :class:`_RowRelation`, so every
    alias stays aligned), and both happen once per branch before any projection
    lowers. Selecting ``t`` over N surviving rows is an N-element array in
    that order, which is the same array value a bare ``f.audio`` produces — the
    row model and the array model are one mechanism.

    `source` is the INPUT alias the tracks belong to. Everything downstream
    (the ``-i``, its WHERE window, provenance) keys off THAT alias, not the row
    one: a row table takes no input slot of its own. `values` is set instead
    for a WRITTEN row source (a struct row table in FROM), whose rows come
    from the query rather than from a probe; it has no input alias and no
    streams.

    `extra` is set for a RENDITION table whose rows a module wrote: the value
    columns it named beside the six rendition ones, each with the type its
    values settled on. Empty for a manifest's own rows, which carry the six
    and no more.
    """

    alias: str
    source: str
    column: str  # the array that was unnested: video/audio/subtitle/data
    type: StreamType
    relation: _RowRelation
    values: RawValuesTable | None = None
    extra: Mapping[str, RowColumnType] = field(default_factory=dict)

    @property
    def rows(self) -> tuple[_TrackRow | None, ...]:
        return tuple(_track_of(row, self.alias) for row in self.relation.tuples)

    @property
    def streamless(self) -> bool:
        """True for rows that carry no track: records, and written rows."""
        return self.values is not None or self.column in RECORD_ARRAY_COLUMNS

    @property
    def record(self) -> str:
        """The record these rows are one of. Only a record array has one."""
        return RECORD_ELEMENTS[self.column]

    @property
    def schema(self) -> dict[str, RowColumnType]:
        """The columns these rows expose, in declaration (or written) order."""
        if self.column == RENDITION_COLUMN:
            return {**_RENDITION_SCHEMA, **self.extra}
        return (
            self.values.schema() if self.values is not None else ROW_SCHEMAS[self.column]
        )

    @property
    def star(self) -> tuple[str, ...]:
        """What ``<alias>.*`` expands to: the scalar columns, in order."""
        if self.column == RENDITION_COLUMN:
            return (*_RENDITION_SCHEMA, *self.extra)
        if self.values is not None:
            return self.values.columns
        return ROW_STAR_COLUMNS[self.column]

    @property
    def readonly(self) -> frozenset[str]:
        """The columns a query may not assert. A written row has none."""
        if self.column == RENDITION_COLUMN:
            return _RENDITION_READONLY | frozenset(self.extra)
        if self.values is not None:
            return frozenset()
        return ROW_READONLY_FIELDS[self.column]

    @property
    def exposes(self) -> str:
        """How a rejection names this row source's column list."""
        listed = ", ".join(sorted(self.schema))
        if self.values is not None or self.extra:
            return f"'{self.alias}' exposes {listed}"
        return f"{self.column} track rows expose {listed}"


_Binding = _InputBinding | _CteBinding | _SourceBinding | _RowBinding


def _track_of(row: _RowTuple, alias: str) -> _TrackRow | None:
    """One result row's track for a row alias; None for a gap or a CTE row."""
    entry = row.get(alias)
    return entry if isinstance(entry, _TrackRow) else None


def _tags_to_cell(tags: dict[str, str]) -> ArrayCell:
    """One tag map as an array cell of ``(key,value)`` records, in key order."""
    return ArrayCell(
        elements=tuple(RecordCell(fields=(key, tags[key])) for key in sorted(tags))
    )


def _tag_cell(row: _TrackRow | None) -> CellValue:
    """One row's whole tag map as a cell; NULL for an outer join's gap."""
    if row is None:
        return None
    source = row.stream.source
    return _tags_to_cell({} if source is None else source.metadata)


def _flags_to_cell(flags: dict[str, bool]) -> ArrayCell:
    """One disposition as an array cell of ``(key,set)`` records, in flag order.

    The key set is CLOSED, so every declared flag is an entry; one this ffmpeg
    did not report reads NULL, the way an absent tag does.
    """
    return ArrayCell(
        elements=tuple(
            RecordCell(fields=(key, flags.get(key))) for key in DISPOSITION_KEYS
        )
    )


def _disposition_cell(row: _TrackRow | None) -> CellValue:
    """One row's whole flag map as a cell; NULL where nothing was probed."""
    if row is None or row.stream.source is None:
        return None
    return _flags_to_cell(row.stream.source.disposition)


def _row_value_as_cell(value: RowValue) -> CellValue:
    """A row value as a printable cell: a vector wraps, everything else already is one."""
    return VectorCell(values=value) if isinstance(value, tuple) else value


@dataclass
class _Env:
    """Everything one SELECT branch resolves names against."""

    bindings: dict[str, _Binding] = field(default_factory=dict)
    # CTE name -> its WHERE window. CTE-ONLY: an INPUT alias's window is a
    # property of its `-i`, not of this branch, so `_collect_trims` records it
    # in `Graph.input_trims` instead and no filter trim is ever spliced for it.
    # Either half may be None (an open-ended window).
    trims: dict[str, tuple[int | float | None, int | float | None]] = field(
        default_factory=dict
    )
    # base stream ref -> its trimmed ref, so one filter trim is shared by every
    # consumer of that stream inside this branch (CTE-only, as above).
    trimmed: dict[FrameRef, FrameRef] = field(default_factory=dict)
    # The branch's joined row set, or None until its first
    # `unnest` binds. There is at most ONE: every row table of a branch joins
    # into it, comma sources included (the comma between two unnests is the
    # bounded cross join), so all row aliases stay aligned by construction.
    relation: _RowRelation | None = None
    # True for a branch that aggregates -- a GROUP BY, an `array_agg`, or both.
    # Its scalar columns are group-constants (resolve's grouping check proves
    # that), so they tag the CONTAINER rather than the tracks.
    grouped: bool = False
    # The GROUP BY keys that read a track-row column: the ones that actually
    # partition the relation. An input-level or constant key has the same value
    # for every tuple and leaves one group.
    group_keys: tuple[exp.Expr, ...] = ()
    # Input alias -> the `-i` each ROW of `relation` seeks, in row order. Set
    # only for a window whose bounds read a row column with no fan-out TO: the
    # rows stay in one graph, so each takes its own copy of the input with its
    # own `-ss`/`-to`, and every stream column of the alias reads one stream
    # per row.
    row_inputs: dict[str, list[str]] = field(default_factory=dict)
    # The rendition row alias `array_agg(<expression over its [1] columns>)`
    # is currently lowering, if any (:meth:`_lower_rendition_agg_expr`). While
    # set, that alias's `.video[1]` / `.audio[1]` reads the array every
    # surviving row contributes instead of picking one row out of it -- the
    # same reading a bare `array_agg(r.video[1])` already gets.
    rendition_agg: str | None = None


def _has_track_rows(env: _Env) -> bool:
    """True when the branch has rows carrying a track to tag per stream.

    Chapter rows and written rows carry none, so a branch holding only those
    tags the CONTAINER, exactly as one with no rows at all does.
    """
    return any(
        isinstance(binding, _RowBinding) and not binding.streamless
        for binding in env.bindings.values()
    )
