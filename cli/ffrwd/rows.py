"""Reading a compile-time row table.

``unnest`` over a track array, a rendition ladder and a written VALUES table
all bind the same way: a relation whose rows are known before ffmpeg runs.
This is what reads one -- which expressions touch it, what a row's columns
and cells hold, and which ``-i`` each row seeks.
"""

from __future__ import annotations

from dataclasses import replace

from sqlglot import exp

from ffrwd.bindings import (
    _RENDITION_SCHEMA,
    RENDITION_COLUMN,
    _disposition_cell,
    _Env,
    _InputBinding,
    _row_value_as_cell,
    _RowBinding,
    _tag_cell,
)
from ffrwd.errors import ErrorCode
from ffrwd.expressions import _error, _number, _unwrap
from ffrwd.ir import StreamType, is_src, src_parts
from ffrwd.merge import RowValue
from ffrwd.parser import _ident_name as _fold
from ffrwd.parser import _time_bounds, column_label, map_ref
from ffrwd.table import CellValue
from ffrwd.types import DISPOSITION_COLUMN, TAGS_COLUMN, TIME_COLUMN
from ffrwd.values import _NULL_STREAM_REF, _TYPE_MARKERS, _array, _Stream, _Value


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


def _literal_of(node: exp.Expr | None, select: exp.Select) -> RowValue:
    """A row predicate's literal operand as a python scalar."""
    value = _unwrap(node) if isinstance(node, exp.Expr) else None
    if isinstance(value, exp.Neg) and isinstance(value.this, exp.Expr):
        return -_number(_unwrap(value.this), ErrorCode.UNSUPPORTED_SQL)
    if isinstance(value, exp.Literal):
        if value.is_string:
            return str(value.this)
        return _number(value, ErrorCode.UNSUPPORTED_SQL)
    raise _error(
        ErrorCode.UNSUPPORTED_SQL,
        "a track-row predicate compares a row column against a literal",
        value,
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
