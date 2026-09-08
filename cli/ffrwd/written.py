"""Values a query writes, checked as they are read.

A ``STRUCT`` matched to the fields a record type declares, the width every
row of a vector track must agree on, and the closed flag set a disposition
may name.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlglot import exp

from ffrwd.errors import ErrorCode
from ffrwd.expressions import _error, _struct_fields
from ffrwd.merge import RowValue
from ffrwd.parser import article, flag_error
from ffrwd.types import DISPOSITION_COLUMN, DISPOSITION_KEYS, EMBEDDING_TYPE, Field

if TYPE_CHECKING:
    from ffrwd.lower import _Embedding


def _named_record_cells(
    struct: exp.Struct,
    record: str,
    fields: tuple[Field, ...],
    select: exp.Select,
) -> dict[str, exp.Expr]:
    """One ``STRUCT(... AS name)`` matched to a record's declared fields.

    Order-free: the field NAME picks the slot. A field the struct leaves
    out is NULL, which the per-field checks then accept or reject exactly
    as a written NULL is accepted or rejected.
    """
    written = _struct_fields(struct)
    declared = {field.name for field in fields}
    for name in written:
        if name in declared:
            continue
        listed = ", ".join(field.name for field in fields)
        raise _error(
            ErrorCode.UNSUPPORTED_SQL,
            f"{article(record)} {record} has no field '{name}'",
            written[name],
            fallback=select,
            hint=f"the fields a {record} takes are {listed}",
        )
    return {
        field.name: written.get(field.name, exp.Null()) for field in fields
    }


def _flag_spec(
    value: RowValue, anchor: exp.Expr, select: exp.Select
) -> tuple[str, ...]:
    """One written disposition value as the flags it sets, in declared order.

    ``'default+forced'`` sets those two, ``'0'`` and NULL set none, and a
    name outside the closed set is a rejection naming the ones that are in
    it. Order is the type's, not the writer's, so one flag map has one
    spelling however it was typed.
    """
    if value is None:
        return ()
    if not isinstance(value, str):
        raise _error(
            ErrorCode.UNSUPPORTED_SQL,
            f"'{DISPOSITION_COLUMN}' takes ffmpeg's flag spec, not a number",
            anchor,
            fallback=select,
            hint=f"quote the flags, e.g. '{DISPOSITION_KEYS[0]}' or '0' to "
            "clear them",
        )
    if value == "0":
        return ()
    named = set()
    for part in value.split("+"):
        key = part.strip().lower()
        if not key or key.startswith(("+", "-")):
            # ffmpeg's own `+flag`/`-flag` adjusts what the source carries;
            # this column says what the whole map is, so there is nothing
            # for a relative spec to adjust.
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"'{value}' is not a flag list",
                anchor,
                fallback=select,
                hint="name every flag the track should have, joined with "
                f"'+', e.g. '{DISPOSITION_KEYS[0]}+{DISPOSITION_KEYS[6]}'; "
                "'0' clears them all",
            )
        if key not in DISPOSITION_KEYS:
            raise flag_error(part.strip(), key, anchor, select)
        named.add(key)
    return tuple(key for key in DISPOSITION_KEYS if key in named)


def _embedding_dims(
    rows: list[_Embedding], node: exp.Expr, select: exp.Select
) -> int:
    """How many numbers every row of one vector track carries.

    A track has ONE width -- it is a stream tag, not a per-row field --
    so two rows of different lengths are a rejection naming both.
    """
    dims = len(rows[0].vector)
    for position, row in enumerate(rows[1:], start=2):
        if len(row.vector) != dims:
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"{EMBEDDING_TYPE} {position} carries {len(row.vector)} "
                f"numbers, and {EMBEDDING_TYPE} 1 carries {dims}",
                row.vector_node,
                fallback=select,
                hint="one track holds vectors of one length; write the rows "
                "of one embedder per track",
            )
    if dims == 0:
        raise _error(
            ErrorCode.UNSUPPORTED_SQL,
            f"{EMBEDDING_TYPE} 1 carries an empty vector",
            rows[0].vector_node,
            fallback=select,
            hint="a vector track holds the numbers an embedder wrote; an "
            "empty one says nothing",
        )
    return dims
