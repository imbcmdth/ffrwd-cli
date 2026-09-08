"""What a wasm module publishes, and the values it hands back.

A declared annotation record is matched field for field against the row
schemas the module's ``describe`` reports, and the JSON a call returns is
read back as a compile-time value of the declared type.
"""

from __future__ import annotations

from collections.abc import Sequence

from sqlglot import exp

from ffrwd.errors import ErrorCode, FfrwdError
from ffrwd.expressions import _error
from ffrwd.functions import Annotation, WasmFunction
from ffrwd.merge import RowValue
from ffrwd.wasm import ANNOTATION_TYPES


def _annotation_fields(annotation: Annotation) -> tuple[tuple[str, str], ...]:
    """One annotation record's fields, name-ordered, for comparing two of them."""
    return tuple(sorted((f.name, f.type) for f in annotation.fields))


def _vector_field(annotation: Annotation) -> str | None:
    """The name of `annotation`'s vector-typed field, or None if it has none."""
    for f in annotation.fields:
        if f.type == "vector":
            return f.name
    return None


def _annotation_matches(
    declared: Sequence[tuple[str, str]], emitted: Sequence[tuple[str, str]]
) -> bool:
    """Whether a declared annotation record and a module's rows are the same shape.

    Same column names, and each declared type covering the JSON type the
    module gave that column. Order says nothing: the rows travel keyed by name.
    """
    if len(declared) != len(emitted):
        return False
    return all(
        name == emitted_name and json_type in ANNOTATION_TYPES.get(kind, ())
        for (name, kind), (emitted_name, json_type) in zip(declared, emitted)
    )


def _written_json_fields(fields: Sequence[tuple[str, str]]) -> str:
    """A module's row schema as a message spells it."""
    if not fields:
        return "rows with no columns"
    return "rows of " + ", ".join(f"{name} ({kind or 'no type'})" for name, kind in fields)


def _is_number_list(result: object) -> bool:
    """True for a JSON array of numbers -- a vector-returning module's answer."""
    return isinstance(result, list) and all(
        isinstance(v, int | float) and not isinstance(v, bool) for v in result
    )

def _bad_wasm_param(
    name: str,
    value: RowValue,
    wanted: str,
    anchor: exp.Expr,
    select: exp.Select,
) -> FfrwdError:
    return _error(
        ErrorCode.UDF_ARG_TYPE,
        f"the module's parameter '{name}' is {wanted}, got {value!r}",
        anchor,
        fallback=select,
        hint=f"write a value the module can take: '{name}' is {wanted}",
    )


def _folded_result(
    declared: WasmFunction, result: object, node: exp.Expr, select: exp.Select
) -> RowValue:
    """The module's JSON answer as this call's compile-time value.

    Checked against the DECLARED return type, not the schema -- the
    schema was already checked once, at :meth:`_described_value`, but the
    module could still hand back a value of the wrong JSON type at this
    particular call.
    """
    if declared.returns == "boolean":
        if isinstance(result, bool):
            return result
    elif not isinstance(result, bool):
        if declared.returns == "text" and isinstance(result, str):
            return result
        if declared.returns == "number" and isinstance(result, int | float):
            return result
        if declared.returns == "vector" and _is_number_list(result):
            assert isinstance(result, list)
            return tuple(float(v) for v in result)
    raise _error(
        ErrorCode.UDF_ARG_TYPE,
        f"function '{declared.name}' declares RETURNS {declared.returns}, and "
        f"the module '{declared.module}' returned {result!r}",
        node,
        fallback=select,
        hint=f"the module's result must be {declared.returns}",
    )
