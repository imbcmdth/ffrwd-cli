"""Reading a compile-time row table.

``unnest`` over a track array, a rendition ladder and a written VALUES table
all bind the same way: a relation whose rows are known before ffmpeg runs.
This is what reads one -- which expressions touch it, what a row's columns
and cells hold, and which ``-i`` each row seeks.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import replace

from sqlglot import exp

from ffrwd.bindings import (
    _RENDITION_SCHEMA,
    RENDITION_COLUMN,
    _CteBinding,
    _CteRow,
    _disposition_cell,
    _Env,
    _InputBinding,
    _row_value_as_cell,
    _RowBinding,
    _RowRelation,
    _RowTuple,
    _tag_cell,
    _track_of,
    _TrackRow,
)
from ffrwd.ctes import _cte_cell_column, _cte_column, _cte_column_ref
from ffrwd.errors import ErrorCode, FfrwdError
from ffrwd.evaluate import _eval_row, _eval_value, _EvalContext, _sort_key
from ffrwd.expressions import _describe, _error, _unwrap
from ffrwd.fills import _paired_row
from ffrwd.functions import Annotation, WasmFunction
from ffrwd.ir import FrameRef, StreamType, is_src, src_parts
from ffrwd.merge import RowValue, merge_rows
from ffrwd.parser import (
    ROW_STREAM,
    RawRowJoin,
    RawTrackRows,
    RawValuesTable,
    _order_by_alias_expr,
    _time_bounds,
    column_label,
    map_ref,
    record_cast_type,
)
from ffrwd.parser import _ident_name as _fold
from ffrwd.table import ArrayCell, CellValue
from ffrwd.types import CUE_TYPE, DISPOSITION_COLUMN, TAGS_COLUMN, TIME_COLUMN
from ffrwd.values import (
    _NULL_STREAM_REF,
    _TYPE_MARKERS,
    _array,
    _Column,
    _Stream,
    _stream_to_cell,
    _Value,
)


def _reads_unbound_rendition_column(expression: exp.Expr, env: _Env) -> bool:
    """True if `expression` reads a rendition column off an input alias
    `env` binds, but only as a plain `_InputBinding` -- a ladder column
    resolve admitted on spec, that this alias's own probe turned up no
    renditions for."""
    for sub in expression.walk():
        if not isinstance(sub, exp.Column):
            continue
        table_node = sub.args.get("table")
        if table_node is None:
            continue
        binding = env.bindings.get(_fold(table_node))
        if isinstance(binding, _InputBinding) and _fold(sub.this) in _RENDITION_SCHEMA:
            return True
    return False


def _from_rendition_table(env: _Env | None) -> bool:
    """True when `env` binds a rendition ladder's row table.

    The fix for "too many rows/streams for one slot" is different for a
    ladder than for a joined CTE or an unnest table: narrow it to one
    rendition (WHERE, or ORDER BY ... LIMIT 1), not restructure the query.
    """
    if env is None:
        return False
    return any(
        isinstance(binding, _RowBinding) and binding.column == RENDITION_COLUMN
        for binding in env.bindings.values()
    )


def _reads_row_alias(node: exp.Expr, env: _Env) -> bool:
    """True when `node` reads a column of any row table of this branch."""
    for sub in node.walk():
        if not isinstance(sub, exp.Column):
            continue
        table_node = sub.args.get("table")
        if table_node is not None and isinstance(
            env.bindings.get(_fold(table_node)), _RowBinding
        ):
            return True
    return False


def _is_row_window(conjunct: exp.Expr, env: _Env) -> bool:
    """True for a time window on a non-row alias bounded by row columns."""
    parsed = _time_bounds(conjunct)
    if parsed is None:
        return False
    table_node = parsed[0].args.get("table")
    if table_node is None or _fold(parsed[0].this) != TIME_COLUMN:
        return False
    return not isinstance(env.bindings.get(_fold(table_node)), _RowBinding)


def _row_binding_of(
    node: exp.Expr, env: _Env, select: exp.Select
) -> _RowBinding:
    """The single row table `node`'s columns belong to (checked upstream)."""
    for sub in node.walk():
        if not isinstance(sub, exp.Column):
            continue
        table_node = sub.args.get("table")
        if table_node is None:
            continue
        binding = env.bindings.get(_fold(table_node))
        if isinstance(binding, _RowBinding):
            return binding
    raise _error(  # defensive: the caller only passes row expressions
        ErrorCode.UNSUPPORTED_SQL,
        "unsupported track-row expression",
        node,
        fallback=select,
    )


def _row_bound(
    node: exp.Expr | None, clause: str, select: exp.Select
) -> int:
    """One LIMIT/OFFSET count as a python int (defensive: resolve checked)."""
    value = _unwrap(node) if isinstance(node, exp.Expr) else None
    if isinstance(value, exp.Literal) and not value.is_string:
        text = str(value.this)
        if text.isascii() and text.isdigit():
            return int(text)
    raise _error(
        ErrorCode.UNSUPPORTED_SQL,
        f"{clause} must be an integer literal",
        node if isinstance(node, exp.Expr) else None,
        fallback=select,
    )


def _rendition_row_cells(binding: _RowBinding, kind: StreamType) -> _Value:
    """``<alias>.video``/``.audio`` as one cell per ladder rung, in row
    order -- a bare column and its ``[1]`` subscript read the same thing,
    since a rung carries at most one stream of a kind. A rung without
    that kind (an audio-only rendition's ``.video``) contributes the NULL
    sentinel a manifest's variant map already knows how to read as an
    absent cell, exactly like an unmatched FULL JOIN row.
    """
    return _array(
        kind,
        (
            row.kinds[kind]
            if row is not None and kind in row.kinds
            else _Stream(ref=_NULL_STREAM_REF, type=kind, source=None)
            for row in binding.rows
        ),
    )


def _per_row_seeks(
    binding: _RowBinding, streams: list[_Stream], env: _Env
) -> list[_Stream]:
    """Re-point a row table's tracks at the ``-i`` each row seeks.

    A row-bounded window on the input the tracks came from gives every
    result row a copy of the file with its own ``-ss``/``-to``, and a row's
    track belongs to the copy that row seeks: the same stream of the same
    file, read through a different input slot.
    """
    row_inputs = env.row_inputs.get(binding.source)
    if row_inputs is None:
        return streams
    reseeked: list[_Stream] = []
    for position, stream in enumerate(streams):
        if position >= len(row_inputs) or not is_src(stream.ref):
            reseeked.append(stream)
            continue
        _, stream_type, index = src_parts(stream.ref)
        marker = _TYPE_MARKERS[stream_type]
        reseeked.append(
            replace(stream, ref=f"src:{row_inputs[position]}:{marker}:{index}")
        )
    return reseeked


def _row_elements(per_row: bool, env: _Env) -> int | None:
    """How many rows a per-row call runs over, or None when it reads none.

    The relation as this branch left it: a grouped branch has already been
    cut to the group being gathered, and a fan-out to the one row this
    command writes -- so both keep making the one node they always made.
    """
    if not per_row or env.relation is None:
        return None
    return len(env.relation.tuples)


def _row_metadata_cells(
    binding: _RowBinding, name: str, anchor: exp.Expr, select: exp.Select
) -> list[CellValue]:
    """A row alias's metadata column, one value per row (NULL for a gap)."""
    schema = binding.schema
    if name == TAGS_COLUMN:
        return [_tag_cell(row) for row in binding.rows]
    if name == DISPOSITION_COLUMN and name in schema:
        return [_disposition_cell(row) for row in binding.rows]
    if name not in schema and map_ref(name) is None:
        raise _error(
            ErrorCode.UNSUPPORTED_SQL,
            f"unknown column '{binding.alias}.{column_label(name)}'",
            anchor,
            fallback=select,
            hint=binding.exposes,
        )
    return [
        None if row is None else _row_value_as_cell(row.columns.get(name))
        for row in binding.rows
    ]


def _join_rows(
    ctx: _EvalContext,
    relation: _RowRelation,
    alias: str,
    rows: Sequence[_TrackRow | _CteRow],
    join: RawRowJoin | None,
    env: _Env,
    select: exp.Select,
) -> None:
    """Fold one freshly bound row source into the branch's relation.

    Ordinary SQL join semantics, evaluated here because every column is
    probed metadata ("the joins never reach ffmpeg"):

    * the FIRST row table simply becomes the relation;
    * a comma between two row tables is the bounded CROSS join;
    * ``ON`` is 061's three-valued evaluator, and a pair is kept only when
      it comes back TRUE — so a NULL key matches nothing, without that
      being a rule of ours;
    * multiplicity is real: a left row matching two right rows pairs with
      BOTH (two result rows, hence two output streams). The fix, when that
      is not wanted, is a wider key, not an error;
    * LEFT keeps an unmatched left row with a NULL right side, FULL also
      appends the unmatched RIGHT rows, in their own order, after every
      left row -- which is the whole of the row-order rule.
    """
    kind = join.kind if join is not None else "cross"
    if not relation.aliases:
        relation.aliases.append(alias)
        relation.tuples = [{alias: row} for row in rows]
        return
    if join is not None and join.on is not None:
        for key_alias, names in _join_keys(join.on).items():
            for name in names:
                if name not in relation.keys.setdefault(key_alias, []):
                    relation.keys[key_alias].append(name)

    combined: list[_RowTuple] = []
    matched: set[int] = set()
    for left in relation.tuples:
        paired = False
        for position, row in enumerate(rows):
            candidate: _RowTuple = {**left, alias: row}
            if kind != "cross" and (
                join is None
                or join.on is None
                or _eval_row(ctx, join.on, env, candidate, select) is not True
            ):
                continue
            combined.append(candidate)
            matched.add(position)
            paired = True
        if not paired and kind in ("left", "full"):
            combined.append({**left, alias: None})
    if kind == "full":
        empty: _RowTuple = {name: None for name in relation.aliases}
        combined += [
            {**empty, alias: row}
            for position, row in enumerate(rows)
            if position not in matched
        ]
    relation.aliases.append(alias)
    relation.tuples = combined


def _filter_rows(
    ctx: _EvalContext, conjuncts: list[exp.Expr], env: _Env, select: exp.Select
) -> None:
    """Keep the rows whose predicate is TRUE; drop UNKNOWN and FALSE alike.

    Standard SQL: WHERE admits TRUE only, so a row whose metadata field was
    never probed simply does not match — no new rule, and no silent guess.
    The surviving set is written back onto the branch's relation, so every
    later ``t`` sees it and an unselected row's stream is never
    touched. Filtering happens AFTER the joins, which is where
    SQL puts it: dropping a row of an outer join's nullable side before the
    join would silently turn it into an inner one.
    """
    for conjunct in conjuncts:
        relation = _predicate_relation(conjunct, env, select)
        relation.tuples = [
            row
            for row in relation.tuples
            if _eval_row(ctx, conjunct, env, row, select) is True
        ]


def _merged_rows(
    ctx: _EvalContext,
    raw: RawTrackRows,
    rows: list[_TrackRow],
    env: _Env,
    select: exp.Select,
) -> list[_TrackRow]:
    """`rows` narrowed by the gather that read them, then merged into runs.

    The gather's predicate reads these rows and nothing else, so it runs
    over a relation of its own -- one tuple per row, under the name the
    gather gave them -- before the runs collapse. Without a merge the rows
    come back as they were.
    """
    merge = raw.merge
    if merge is None:
        return rows
    if merge.alias is not None and merge.where is not None:
        relation = _RowRelation(
            aliases=[merge.alias], tuples=[{merge.alias: row} for row in rows]
        )
        inner = _Env(
            bindings={
                **env.bindings,
                merge.alias: _RowBinding(
                    alias=merge.alias,
                    source=raw.source,
                    column=raw.column,
                    type="data",
                    relation=relation,
                ),
            },
            relation=relation,
        )
        rows = [
            row
            for row, tuple_ in zip(rows, relation.tuples)
            if _eval_row(ctx, merge.where, inner, tuple_, select) is True
        ]
    return [
        _TrackRow(stream=_STREAMLESS_ROW, columns=columns)
        for columns in merge_rows([row.columns for row in rows], merge.max_distance)
    ]


def _order_rows(ctx: _EvalContext, select: exp.Select, env: _Env) -> None:
    """Re-sort a row table explicitly -- the ORDER BY carve-out.

    Row order is deterministic WITHOUT this — it is the file's track order,
    which is player-visible surface nothing resorts implicitly — so an
    ORDER BY is the user saying otherwise, and it applies at compile time
    to the row list, never to frames.

    Multi-key sorting is done one key at a time from LAST to FIRST over
    python's stable sort, which is exactly SQL's key precedence. NULLs are
    partitioned out rather than sorted, because they have no order: their
    position is ``nulls_first``, which sqlglot fills in from the Postgres
    defaults (ASC -> NULLS LAST, DESC -> NULLS FIRST) whether or not the
    query spelled it.

    Every key goes through the one value evaluator, so a CTE's value
    column sorts the branch's rows exactly as a track row's own column
    does, and a name the body never selected is the unknown-column
    refusal that names what it did.
    """
    order = select.args.get("order")
    if not isinstance(order, exp.Order):
        return
    if env.relation is None:
        # The parser admitted ORDER BY on the strength of an `input(...)`
        # alias that MIGHT have been a ladder; the probe just settled it
        # wasn't, so this branch has no row table after all.
        raise _error(
            ErrorCode.NO_STREAMING_EQUIVALENT,
            "ORDER BY has no streaming equivalent",
            order,
            fallback=select,
            hint="remove the ORDER BY clause",
        )
    relation = env.relation
    for ordered in reversed(order.expressions):
        if not isinstance(ordered, exp.Ordered):
            raise _error(
                ErrorCode.UNSUPPORTED_SQL, "malformed ORDER BY", fallback=order
            )
        key = _unwrap(ordered.this)
        if isinstance(key, exp.Column) and key.args.get("table") is None:
            # A bare SELECT-list alias: resolve already chased this back
            # to its aliased expression (:meth:`ffrwd.parser._Resolver.
            # _check_order_key`), so sorting runs against that expression
            # instead of a column no binding owns.
            resolved = _order_by_alias_expr(_fold(key.this), select)
            if resolved is not None:
                key = _unwrap(resolved)
        # A bare column, or a computed key -- a built-in text/number
        # function, a value wasm function's result -- over the same row
        # columns a bare one reads; resolve checked its shape and type
        # (:meth:`ffrwd.parser._Resolver._check_order`).
        def value_of(row: _RowTuple, key: exp.Expr = key) -> RowValue:
            return _eval_value(ctx, key, env, row, select)

        nulls = [row for row in relation.tuples if value_of(row) is None]
        rest = [row for row in relation.tuples if value_of(row) is not None]
        rest.sort(
            key=lambda row: _sort_key(value_of(row)),
            reverse=bool(ordered.args.get("desc")),
        )
        relation.tuples = (
            nulls + rest if ordered.args.get("nulls_first") else rest + nulls
        )


def _limit_rows(select: exp.Select, env: _Env) -> None:
    """Narrow the resolved row set: OFFSET skips rows, LIMIT caps them.

    Applies to the branch's shared relation after WHERE and ORDER BY and
    before grouping, the fan-out pin, and the one-row rule -- so ``ORDER
    BY t.width DESC LIMIT 1`` IS the top row, no aggregate needed. Both
    counts are integer literals (resolve checked, LIMIT 0 included); the
    one judgment only this pass can make is an OFFSET that skips every
    row, since only the resolved relation knows its own size -- the same
    selects-nothing mistake LIMIT 0 names at resolve.
    """
    limit = select.args.get("limit")
    offset = select.args.get("offset")
    if env.relation is None:
        # Same story as `_order_rows`: the parser could not yet tell a
        # renditionless input from a ladder, so this rejection waited for
        # the probe instead of firing at parse time.
        if isinstance(limit, exp.Limit):
            raise _error(
                ErrorCode.NO_STREAMING_EQUIVALENT,
                "LIMIT has no streaming equivalent",
                limit,
                fallback=select,
                hint=_RENDITIONLESS_ROW_CLAUSE_HINT,
            )
        if isinstance(offset, exp.Offset):
            raise _error(
                ErrorCode.NO_STREAMING_EQUIVALENT,
                "OFFSET has no streaming equivalent",
                offset,
                fallback=select,
                hint=_RENDITIONLESS_ROW_CLAUSE_HINT,
            )
    take = (
        _row_bound(limit.args.get("expression"), "LIMIT", select)
        if isinstance(limit, exp.Limit)
        else None
    )
    skip = (
        _row_bound(offset.args.get("expression"), "OFFSET", select)
        if isinstance(offset, exp.Offset)
        else None
    )
    if take is None and skip is None:
        return
    relation = env.relation
    count = len(relation.tuples) if relation is not None else 1
    if skip is not None and skip >= count:
        have = f"{count} row" + ("" if count == 1 else "s")
        raise _error(
            ErrorCode.UNSUPPORTED_SQL,
            f"OFFSET {skip} skips every row: this query has {have}",
            offset,
            fallback=select,
            hint="a query that selects nothing is a mistake worth "
            "naming; skip fewer rows, or drop the clause",
        )
    if relation is None:
        return
    end = None if take is None else (skip or 0) + take
    relation.tuples = relation.tuples[skip or 0 : end]


def _add_values_rows(
    ctx: _EvalContext,
    local: str,
    values: RawValuesTable,
    env: _Env,
    select: exp.Select,
    join: RawRowJoin | None = None,
) -> None:
    """Bind one written row table: its rows join the branch's relation.

    The same join :meth:`_add_track_rows` builds, with the rows read off
    the query instead of a probe -- so a comma between a written row
    table and anything else is the ordinary cross join, an explicit
    `join` matches rows the same way it does between unnest tables, and
    ``array_agg`` over it aggregates the same way. No stream and no
    ``-i``: the rows are values. Each cell takes the ordinary
    compile-time value grammar (:meth:`_eval_value`), evaluated once
    over the branch's representative row -- a ``generate_series`` cell
    is always a literal, so this is the identity for it; a struct row
    table's cell may be an expression over one.
    """
    if env.relation is None:
        env.relation = _RowRelation()
    group_row = _group_row(env)
    rows = [
        _TrackRow(
            stream=_STREAMLESS_ROW,
            columns={
                name: _eval_value(ctx, cell, env, group_row, select)
                for name, cell in zip(values.columns, entry, strict=True)
            },
        )
        for entry in values.rows
    ]
    env.bindings[local] = _RowBinding(
        alias=local,
        source="",
        column=local,
        type="data",  # filler: a written row has no track
        relation=env.relation,
        values=values,
    )
    _join_rows(ctx, env.relation, local, rows, join, env, select)


def _add_cte_rows(
    ctx: _EvalContext,
    local: str,
    columns: tuple[_Column, ...],
    values: dict[str, tuple[RowValue, ...]],
    rows_columns: dict[str, tuple[FrameRef, StreamType, Annotation]],
    cue_columns: frozenset[str],
    env: _Env,
    select: exp.Select,
    join: RawRowJoin | None = None,
) -> None:
    """Bind one CTE reference: its body's rows join the branch's relation.

    One body row is one outer row, so a comma between two CTEs (or between
    a CTE and an unnest table) is the ordinary cross join
    :meth:`_join_rows` already builds, multiplicity and all -- and an
    explicit `join` matches, keeps and gaps rows exactly as it does
    between unnest tables. A single-row body is a shape no-op, which is
    what keeps the one-input CTE shapes compiling exactly as they did.
    """
    if env.relation is None:
        env.relation = _RowRelation()
    rows = _cte_row_count(columns, values)
    env.bindings[local] = _CteBinding(
        name=local,
        columns=columns,
        rows=rows,
        relation=env.relation,
        values=values,
        rows_columns=rows_columns,
        cue_columns=cue_columns,
    )
    _join_rows(
        ctx,
        env.relation,
        local,
        [_CteRow(position=position) for position in range(rows)],
        join,
        env,
        select,
    )


def _add_series_rows(
    ctx: _EvalContext,
    local: str,
    values: tuple[int, ...],
    node: exp.Expr,
    env: _Env,
    select: exp.Select,
    join: RawRowJoin | None = None,
) -> None:
    """Bind one ``generate_series`` table: its computed rows join the
    branch's relation exactly like a struct row table's written ones.

    `values` is the whole computed sequence -- resolve already did the
    arithmetic and rejected a zero step or an empty/descending range,
    since bounds and step are literals by the time it runs. No stream and
    no ``-i``: the rows are computed, not read.
    """
    if env.relation is None:
        env.relation = _RowRelation()
    rows = [_TrackRow(stream=_STREAMLESS_ROW, columns={local: v}) for v in values]
    env.bindings[local] = _RowBinding(
        alias=local,
        source="",
        column=local,
        type="data",  # filler: a computed row has no track
        relation=env.relation,
        values=RawValuesTable(
            alias=local, columns=(local,), rows=(), node=node, types=("number",)
        ),
    )
    _join_rows(ctx, env.relation, local, rows, join, env, select)


def _grouped_partitions(
    ctx: _EvalContext, env: _Env, select: exp.Select
) -> list[list[_RowTuple]]:
    """The relation's tuples partitioned into the groups a table query
    prints, one row each.

    A row-referencing GROUP BY key partitions in FIRST-APPEARANCE order,
    the same partition a media fan-out builds. With no such key the whole
    relation is ONE group -- Postgres's own rule for an aggregate with
    nothing to partition by (unlike a media fan-out's ungrouped case,
    where every row writes its own file).

    An EMPTY relation partitions into NO groups either way: a table query
    prints the same zero rows an ungrouped branch does, and a media query
    falls through to the empty-row-set rejection.
    """
    relation = env.relation
    tuples = relation.tuples if relation is not None else []
    if not tuples:
        return []
    if not env.group_keys:
        return [list(tuples)]
    groups: dict[tuple[RowValue, ...], list[_RowTuple]] = {}
    for row in tuples:
        key = tuple(_key_value(ctx, node, env, row, select) for node in env.group_keys)
        groups.setdefault(key, []).append(row)
    return list(groups.values())


def _fanout_groups(
    ctx: _EvalContext, env: _Env, select: exp.Select
) -> list[list[_RowTuple]]:
    """The relation's tuples partitioned into the files they write.

    One group per distinct GROUP BY key, in FIRST-APPEARANCE order (the
    dict's own insertion order), so the command sequence follows the row
    order the query built. With no row-level key every tuple is its own
    group, which is the ungrouped fan-out unchanged.
    """
    relation = env.relation
    tuples = relation.tuples if relation is not None else []
    if not env.group_keys:
        return [[row] for row in tuples]
    groups: dict[tuple[RowValue, ...], list[_RowTuple]] = {}
    for row in tuples:
        key = tuple(_key_value(ctx, node, env, row, select) for node in env.group_keys)
        groups.setdefault(key, []).append(row)
    return list(groups.values())


def _key_value(
    ctx: _EvalContext, node: exp.Expr, env: _Env, row: _RowTuple, select: exp.Select
) -> RowValue:
    """One GROUP BY key, read out of one result tuple.

    A stream column -- a CTE's, or a row table's ``track`` -- has no
    metadata value to compare, so what identifies the group is the stream
    itself: its ref, which two tuples share exactly when they carry the
    same stream.
    """
    stream = _key_stream(node, env, row)
    if stream is not None:
        return stream.ref
    return _eval_value(ctx, node, env, row, select)


def _key_stream(node: exp.Expr, env: _Env, row: _RowTuple) -> _Stream | None:
    """The stream a GROUP BY key names in this tuple, else None."""
    column_node = _unwrap(node)
    if not isinstance(column_node, exp.Column):
        return None
    table_node = column_node.args.get("table")
    if table_node is None:
        return None
    binding = env.bindings.get(_fold(table_node))
    name = _fold(column_node.this)
    if isinstance(binding, _RowBinding):
        if name != ROW_STREAM:
            return None
        track = _track_of(row, binding.alias)
        return track.stream if track is not None else None
    if not isinstance(binding, _CteBinding):
        return None
    column = _cte_column(binding, name)
    if column is None or not column.value.streams:
        return None
    entry = row.get(binding.name)
    if (
        isinstance(entry, _CteRow)
        and column.splat
        and len(column.value.streams) == binding.rows
    ):
        return column.value.streams[entry.position]
    # A broadcast column is one unit: every tuple reads the same stream.
    return column.value.streams[0]


def _predicate_relation(
    conjunct: exp.Expr, env: _Env, select: exp.Select
) -> _RowRelation:
    """The relation one WHERE predicate filters.

    A predicate over a CTE's value column filters the branch's own
    relation -- the CTE's rows are already joined into it.
    """
    for sub in conjunct.walk():
        if not isinstance(sub, exp.Column):
            continue
        table_node = sub.args.get("table")
        if table_node is None:
            continue
        binding = env.bindings.get(_fold(table_node))
        if isinstance(binding, _CteBinding) and binding.relation is not None:
            return binding.relation
    return _row_binding_of(conjunct, env, select).relation


def _value_cells(
    ctx: _EvalContext, node: exp.Expr, env: _Env, select: exp.Select, cardinality: int
) -> list[CellValue]:
    """A CASE / ``||`` column, evaluated once per row.

    The same expression a media query writes back as a tag, PRINTED
    instead: a table query is how you check what the tag would say before
    writing it.
    """
    relation = env.relation
    if relation is None:
        value = _row_value_as_cell(_eval_value(ctx, node, env, {}, select))
        return [value] * cardinality
    return [
        _row_value_as_cell(_eval_value(ctx, node, env, row, select))
        for row in relation.tuples
    ]


def _value_to_cells(
    value: _Value, cardinality: int, splat: bool = True
) -> list[CellValue]:
    """A lowered stream `_Value` as one cell per row: a scalar broadcasts,
    and a row column's array (``t`` over N surviving rows) splats
    one stream cell per row -- the array IS the row set, not one cell.

    `splat` False marks an array that is NOT a row set -- a call broadcast
    over a bare input array, whose length is the file's track count and
    has nothing to do with the row count. That one prints as a single
    array cell per row, exactly as the bare array column does.
    """
    if value.is_array and splat:
        return [_stream_to_cell(stream) for stream in value.streams]
    if value.is_array:
        array_cell = ArrayCell(
            elements=tuple(_stream_to_cell(stream) for stream in value.streams)
        )
        return [array_cell] * cardinality
    cell = _stream_to_cell(value.streams[0])
    return [cell] * cardinality


def _not_rows(
    declared: WasmFunction, written: exp.Expr, node: exp.Expr, env: _Env
) -> FfrwdError:
    """Why this argument is not the rows a rows function reads.

    Compile-time rows are their own answer: a caption file's cues are
    known before anything runs, so a module hosted beside a producer has
    nothing to run against, and the value grammar is where that work
    already happens. A CTE column is named rather than merely typed --
    'a stream' alone would not say WHICH one -- and classified against
    what its own body actually bound it to, not against `written`'s own
    shape (a Column, whatever it carries).
    """
    cte_ref = _cte_column_ref(written, env)
    if cte_ref is not None:
        binding, name = cte_ref
        label = f"'{binding.name}.{name}'"
        if name in binding.cue_columns:
            return _error(
                ErrorCode.UDF_ARG_TYPE,
                f"{declared.name}() reads a module's rows, and {label} is a "
                "compile-time cue array",
                written,
                fallback=node,
                hint=f"cues the compiler already holds are rewritten one row "
                f"at a time by the value form, e.g. {declared.name}(<row>.text) "
                "inside a STRUCT(...)::cue",
            )
        said = "a value" if name in binding.values else "a stream"
        return _error(
            ErrorCode.UDF_ARG_TYPE,
            f"{declared.name}() reads rows, and {label} is {said}",
            written,
            fallback=node,
            hint=f"call it over the annotation column a module produces, e.g. "
            f"{declared.name}(<producer>(<stream>).<column>)",
        )
    if _is_cue_array(written):
        return _error(
            ErrorCode.UDF_ARG_TYPE,
            f"{declared.name}() reads a module's rows, and its argument is "
            "a compile-time cue array",
            written,
            fallback=node,
            hint=f"cues the compiler already holds are rewritten one row at "
            f"a time by the value form, e.g. {declared.name}(<row>.text) "
            "inside a STRUCT(...)::cue",
        )
    said = (
        "a stream"
        if isinstance(written, exp.Column | exp.Bracket | exp.Anonymous | exp.Dot)
        else _describe(written)
    )
    return _error(
        ErrorCode.UDF_ARG_TYPE,
        f"{declared.name}() reads rows, and its argument is {said}",
        written,
        fallback=node,
        hint=f"call it over the annotation column a module produces, e.g. "
        f"{declared.name}(<producer>(<stream>).<column>)",
    )


def _reads_row_set(node: exp.Expr, env: _Env) -> bool:
    """True when `node` reads a row alias's column, or a CTE column that
    is itself a row set."""
    for sub in node.walk():
        if not isinstance(sub, exp.Column):
            continue
        table_node = sub.args.get("table")
        if table_node is None:
            continue
        binding = env.bindings.get(_fold(table_node))
        if isinstance(binding, _RowBinding):
            return True
        if isinstance(binding, _InputBinding) and _fold(table_node) in env.row_inputs:
            return True
        if isinstance(binding, _CteBinding):
            column = _cte_column(binding, _fold(sub.this))
            if column is not None and _cte_cell_column(binding, column):
                return True
    return False


def _is_splat_projection(projection: exp.Expr, env: _Env) -> bool:
    """True when this stream column's array value (if it turns out to be
    one) is a row set rather than a single broadcast unit.

    Computed here (at the projection's OWN scope, CTE body or bare SELECT)
    because that is the only place its AST shape is still visible -- an
    outer table query sees just ``<cte>.<name>`` and has to trust what got
    recorded.

    A column is a row set exactly when it READS one: a row alias's stream
    column, a call over one, another CTE's row-set column (which it
    inherits), or an input alias a row-bounded window gave one ``-i`` per
    row. A bare input/source array (``f.audio``) and anything broadcast
    over one is a single row carrying an array VALUE, and an ``array_agg``
    is one unit by definition.
    """
    expr = _unwrap(projection)
    if isinstance(expr, exp.ArrayAgg):
        return False
    return _reads_row_set(expr, env)


def _is_cue_array_column(node: exp.Expr, env: _Env) -> bool:
    """True for a compile-time cue array, spelled inline or read off a
    CTE column that is one."""
    if _is_cue_array(node):
        return True
    cte_ref = _cte_column_ref(node, env)
    return cte_ref is not None and cte_ref[1] in cte_ref[0].cue_columns


def _unmatched_text(binding: _RowBinding, position: int) -> str:
    """What the missing row failed to match, named from its paired row."""
    relation = binding.relation
    row = relation.tuples[position]
    paired_alias, paired = _paired_row(relation, row, binding.alias)
    keys = relation.keys.get(paired_alias or "", [])
    if paired is None or not keys:
        return f"the join found no {binding.column} row of '{binding.alias}'"
    described = ", ".join(
        f"{paired_alias}.{column_label(key)}={paired.columns.get(key)!r}"
        for key in keys
    )
    return f"no '{binding.alias}' row matched {described}"


# The parser admits ORDER BY/LIMIT/OFFSET over any `input(...)` alias, since
# whether it turns out to be an ABR ladder is a probed fact, not a syntactic
# one -- so a renditionless input reaches here needing the same rejection
# the parser used to raise for it, word for word.
_RENDITIONLESS_ROW_CLAUSE_HINT = (
    "it is legal only over a compile-time row table -- a branch whose FROM "
    "has unnest(...), generate_series(...), or a CTE or view name -- where "
    "it narrows the resolved rows, exactly like ORDER BY"
)


# `_TrackRow.stream` for a row that carries no track -- a chapter row, or a
# written row. Never a real stream (neither exposes a stream column at
# all), only a dataclass filler. Its ref deliberately fails `is_src()` (no
# "src:" prefix) and is not a node id either, so anything that somehow did try
# to render it fails fast with "unknown node" rather than silently wiring up
# the wrong stream.
_STREAMLESS_ROW = _Stream(ref="rows:no-stream", type="data", source=None)


def _is_cue_array(node: exp.Expr) -> bool:
    """True for the two spellings of a compile-time cue list.

    The SHAPE only -- nothing is evaluated here, since this answers a
    rejection's question rather than building a track.
    """
    if isinstance(node, exp.ArrayAgg):
        inner = node.this
        return isinstance(inner, exp.Expr) and record_cast_type(_unwrap(inner)) == CUE_TYPE
    if isinstance(node, exp.Array):
        elements = [item for item in node.expressions if isinstance(item, exp.Expr)]
        return bool(elements) and record_cast_type(_unwrap(elements[0])) == CUE_TYPE
    return False


def _cte_row_count(
    columns: Iterable[_Column], values: dict[str, tuple[RowValue, ...]]
) -> int:
    """How many rows a CTE body produced: the width of its row-set columns.

    A splat array column carries one stream per body row, so its length IS the
    body's row count; a body with none (a single input row, a UNION ALL's
    concat, a broadcast array) is one row.
    """
    widths = [
        len(column.value.streams)
        for column in columns
        if column.splat and column.value.is_array
    ]
    widths += [len(one) for one in values.values()]
    return max(widths) if widths else 1


def _join_keys(on: exp.Expr) -> dict[str, list[str]]:
    """Which columns each row alias was matched on, from a JOIN's ON predicate.

    Bookkeeping for one message only: a NULL track says what it failed to
    match (``no 'b' row matched a.tags.language='fra'``), and that needs the key
    columns of the side that DID match. Order is written order, deduplicated.
    """
    keys: dict[str, list[str]] = {}
    for sub in on.walk():
        if not isinstance(sub, exp.Column):
            continue
        table_node = sub.args.get("table")
        if table_node is None:
            continue
        names = keys.setdefault(_fold(table_node), [])
        name = _fold(sub.this)
        if name not in names:
            names.append(name)
    return keys


def _group_row(env: _Env) -> _RowTuple:
    """The one tuple a FILE-level value reads, or no row at all.

    A container tag and a chapter list belong to the file, not to a row, so
    they are evaluated over a single representative tuple: the group's first
    where the branch groups, the relation's first otherwise (an ungrouped
    branch that survives the one-row rule has exactly one).
    """
    relation = env.relation
    if relation is None or not relation.tuples:
        return {}
    return relation.tuples[0]
