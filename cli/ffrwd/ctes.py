"""Reading a CTE's recorded columns.

A CTE body's SELECT list is fixed when the body lowers, so ``FROM <cte>``
answers every column question off that record -- what a name carries, whether
it reads one cell per row or one unit, and what the alias exposes when a name
is wrong.
"""

from __future__ import annotations

from sqlglot import exp

from ffrwd.bindings import _CteBinding, _Env
from ffrwd.parser import _ident_name as _fold
from ffrwd.values import _Column


def _cte_column_ref(
    node: exp.Expr, env: _Env
) -> tuple[_CteBinding, str] | None:
    """``<alias>.<column>``, when `alias` names a CTE in scope.

    The binding and the column's folded name, so a caller can look up
    what that column carries. None for everything else -- no error, no
    lowering, a pure probe over `env`.
    """
    if not isinstance(node, exp.Column):
        return None
    table_node = node.args.get("table")
    if table_node is None:
        return None
    binding = env.bindings.get(_fold(table_node))
    if not isinstance(binding, _CteBinding):
        return None
    return binding, _fold(node.this)


def _cte_cell_column(binding: _CteBinding, column: _Column) -> bool:
    """True when a CTE's stream column reads one cell per row of the
    branch's relation: a row set, one stream per body row, or a single
    value every row repeats. A gathered array is one unit instead."""
    value = column.value
    if column.splat and value.is_array:
        return len(value.streams) == binding.rows
    return not value.is_array and len(value.streams) == 1


def _cte_column(binding: _CteBinding, name: str) -> _Column | None:
    for column in binding.columns:
        if column.name == name:
            return column
    return None


def _cte_columns_hint(binding: _CteBinding) -> str:
    names = {column.name for column in binding.columns if column.name is not None}
    names |= set(binding.values)
    if not names:
        return (
            f"'{binding.name}' has no named columns; name them with AS "
            "inside its SELECT"
        )
    return f"'{binding.name}' exposes: {', '.join(sorted(names))}"
