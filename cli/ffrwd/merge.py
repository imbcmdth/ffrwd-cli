"""The merge `merge_cues` is, over rows that carry a span.

Rows are taken in `start_t` order; a row whose `start_t` is no more than
`max_distance` past the run's end joins the run, and a run becomes one row
from the first start to the furthest end its rows reached. A `text` field
joins with one space; every other field is the first row's.

The sidecar's `rowmerge` node is the same merge over rows as they stream.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

__all__ = [
    "END_FIELD",
    "SPAN_FIELDS",
    "START_FIELD",
    "TEXT_FIELD",
    "RowValue",
    "merge_rows",
]

# A row cell: NULL (unprobed input, or a field this file does not carry) or
# the probed scalar. A disposition flag is the boolean case. A vector cell (an
# annotation field, or a value function's result) is a tuple of floats --
# immutable and hashable, unlike a list, which is what lets one memoize a wasm
# value call and key a GROUP BY on its result. Never a stream -- the row IS
# that.
RowValue = str | int | float | bool | tuple[float, ...] | None

START_FIELD = "start_t"
END_FIELD = "end_t"

# The fields a record has to carry for its rows to merge at all.
SPAN_FIELDS = (START_FIELD, END_FIELD)

# The one field a run joins rather than taking the first row's.
TEXT_FIELD = "text"

Row = Mapping[str, RowValue]


def _bound(row: Row, name: str) -> float:
    """One span bound of a row, as a number."""
    value = row.get(name)
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else 0.0


def _joined(run: Sequence[Row], end: float) -> dict[str, RowValue]:
    """One run as the row that stands for it."""
    merged: dict[str, RowValue] = dict(run[0])
    if TEXT_FIELD in merged:
        merged[TEXT_FIELD] = " ".join(
            str(row[TEXT_FIELD]) for row in run if row.get(TEXT_FIELD) is not None
        )
    merged[END_FIELD] = end
    return merged


def merge_rows(rows: Sequence[Row], max_distance: float) -> list[dict[str, RowValue]]:
    """`rows` collapsed into one row per run. An empty array stays empty."""
    ordered = sorted(rows, key=lambda row: _bound(row, START_FIELD))
    merged: list[dict[str, RowValue]] = []
    run: list[Row] = []
    end = 0.0
    for row in ordered:
        if run and _bound(row, START_FIELD) - end <= max_distance:
            run.append(row)
            end = max(end, _bound(row, END_FIELD))
            continue
        if run:
            merged.append(_joined(run, end))
        run = [row]
        end = _bound(row, END_FIELD)
    if run:
        merged.append(_joined(run, end))
    return merged
