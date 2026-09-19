"""Table/CSV rendering for ffrwd.

``ffrwd.lower.lower_table`` builds a :class:`TableResult` per sink -- column
names and already-resolved cell values -- and this module turns one into
either the psql-style ASCII table or a CSV document. No SQL, no IR graph
knowledge: pure formatting over already-typed data.

Format is PINNED byte-for-byte by cookbook recipes 30 (docs/examples.md) and
31 (docs/corpus.md):
one leading space per cell, cells left-justified to ``max(header width, every
value's width)``, columns joined with ``" | "``, a dashed rule with ``"+"`` at
the column separators (``width + 2`` dashes each), a ``"(N rows)"`` /
``"(1 row)"`` footer, and every line RSTRIPPED -- so a row whose last cell is
empty ends at its ``"|"``. Plain ASCII, no unicode box-drawing: it pins
cleanly and survives any pipe.
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from typing import Literal

from .ir import StreamType

__all__ = [
    "VECTOR_CELL_CAP",
    "StreamCell",
    "ArrayCell",
    "RecordCell",
    "VectorCell",
    "CellValue",
    "TableResult",
    "TableSink",
    "render_table",
    "TableFormat",
    "render_csv",
    "render_json",
]


@dataclass(frozen=True)
class StreamCell:
    """A stream-valued table cell: what would have been wired, not run.

    ``spec`` arrives fully resolved -- the ffmpeg stream spec (``"0:a:0"``)
    for a source passthrough, or a filtergraph node id (``"n2"``) for a
    filtered stream -- because only ``ffrwd.lower`` has ``Graph.sources`` to
    convert it. Renders as ``<video 0:v:0>`` / ``<audio n2>``.
    """

    type: StreamType
    spec: str


@dataclass(frozen=True)
class RecordCell:
    """One record of a record array (a chapter), one nested cell.

    Postgres record-literal style over the fields' own cell text, in schema
    order -- ``(1,Intro,0.0,1.0)``. Nothing is quoted or escaped: these are
    printable data, not a re-parseable literal.
    """

    fields: tuple[CellValue, ...]


@dataclass(frozen=True)
class ArrayCell:
    """A bare input array column's full element list, one table cell.

    Postgres array-literal style over the elements' own cell text --
    ``{<audio 0:a:0>,<audio 0:a:1>}`` -- braces even for one element.
    """

    elements: tuple[CellValue, ...]


# What a table query's result is rendered as: the psql-style ASCII table a
# person reads, or one of the two a program does.
TableFormat = Literal["table", "csv", "json"]


# The most values a vector cell prints in the ASCII TABLE before summarizing
# the rest -- a table is read by a person, and a full embedding would blow out
# every column's width. Nothing a program reads is capped: see `_cell_text`.
VECTOR_CELL_CAP = 4


@dataclass(frozen=True)
class VectorCell:
    """A vector row column's value, one table cell.

    In the ASCII table it prints capped: the first :data:`VECTOR_CELL_CAP`
    values, then the vector's own length in place of the rest -- ``[0.12,
    -0.03, 0.5, 0.77, ... (384)]``. A vector no longer than the cap prints
    whole, with no ellipsis. Not Postgres array-literal style (no braces) --
    a vector is read, never re-parsed, so the numeric-array spelling is the
    plainer one.

    Everywhere a PROGRAM reads the result -- csv, json -- it is written in
    full. A capped vector is not a vector, and a file written for a machine
    that silently held four of three hundred numbers would be worse than one
    that refused.
    """

    values: tuple[float, ...]


# One table cell: NULL (empty, psql-style), a probed scalar, a stream
# placeholder, a record, an array of cells, or a vector. Never a raw
# FrameRef -- that would leak IR shape into printable data.
CellValue = str | int | float | bool | None | StreamCell | RecordCell | ArrayCell | VectorCell


@dataclass(frozen=True)
class TableResult:
    """One table query's result set: column names, then rows in row order."""

    columns: list[str]
    rows: list[list[CellValue]]


@dataclass(frozen=True)
class TableSink:
    """One table query's destination.

    Mirrors ``ffrwd.ir.SinkUnit`` for the non-media path. A bare SELECT is
    exactly one of these with ``format="table"``, ``path=None`` (``run``
    prints the ASCII table to stdout; there is no file form). A ``COPY ...
    WITH (FORMAT csv)`` has ``format="csv"``, and ``FORMAT json`` or a
    ``.json`` path ``format="json"``; either has ``path`` None for ``TO
    STDOUT`` or the file path for ``TO '<path>'``. ``header`` is the csv
    ``HEADER`` option's value, irrelevant for the other two.
    """

    result: TableResult
    path: str | None
    format: TableFormat
    header: bool

    @property
    def csv(self) -> bool:
        """True for the csv spelling. Kept for callers that only ask that."""
        return self.format == "csv"


def _cell_text(cell: CellValue, *, capped: bool = True) -> str:
    """One cell as text. `capped` is the ASCII table's vector rule.

    Uncapped, a vector writes every value: what a program reads is the
    vector, not a summary of it.
    """
    if cell is None:
        return ""
    if isinstance(cell, bool):
        return "true" if cell else "false"
    if isinstance(cell, StreamCell):
        return f"<{cell.type} {cell.spec}>"
    if isinstance(cell, ArrayCell):
        return (
            "{"
            + ",".join(_cell_text(element, capped=capped) for element in cell.elements)
            + "}"
        )
    if isinstance(cell, RecordCell):
        return (
            "("
            + ",".join(_cell_text(field, capped=capped) for field in cell.fields)
            + ")"
        )
    if isinstance(cell, VectorCell):
        if not capped or len(cell.values) <= VECTOR_CELL_CAP:
            return "[" + ", ".join(str(v) for v in cell.values) + "]"
        shown = ", ".join(str(v) for v in cell.values[:VECTOR_CELL_CAP])
        return f"[{shown}, ... ({len(cell.values)})]"
    return str(cell)


def render_table(result: TableResult) -> str:
    """The psql-style ASCII table for `result`, newline-joined, no trailing "\\n".

    Callers ``print()`` it, which supplies the final newline the pinned
    recipes' code blocks show.
    """
    headers = result.columns
    text_rows = [[_cell_text(cell) for cell in row] for row in result.rows]
    widths = [
        max(len(headers[i]), max((len(row[i]) for row in text_rows), default=0))
        for i in range(len(headers))
    ]

    def _line(cells: list[str]) -> str:
        return (" " + " | ".join(cell.ljust(width) for cell, width in zip(cells, widths))).rstrip()

    lines = [_line(headers), "+".join("-" * (width + 2) for width in widths)]
    lines += [_line(row) for row in text_rows]
    count = len(result.rows)
    lines.append(f"({count} row)" if count == 1 else f"({count} rows)")
    return "\n".join(lines)


def render_csv(result: TableResult, *, header: bool) -> str:
    """`result` as CSV text (stock ``csv`` module defaults, LF line endings).

    Quoting only when a value needs it (``csv.writer``'s own default,
    ``QUOTE_MINIMAL``). Ends with a trailing newline after the last row, like
    any well-formed CSV file/stream -- callers write or print it verbatim,
    never through ``print()`` (which would double it).
    """
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    if header:
        writer.writerow(result.columns)
    for row in result.rows:
        writer.writerow([_cell_text(cell, capped=False) for cell in row])
    return buffer.getvalue()


def _cell_json(cell: CellValue) -> object:
    """One cell as JSON. What has a JSON shape keeps it; the rest is text.

    A number is a number, a boolean a boolean, NULL is null and a vector is
    an array of numbers -- the wire shape a reader wants. A stream, a record
    and an array of cells have no JSON of their own, so they carry the text
    the csv writes, which is the only spelling of them there is.
    """
    if cell is None:
        return None
    if isinstance(cell, bool | int | float):
        return cell
    if isinstance(cell, VectorCell):
        return list(cell.values)
    if isinstance(cell, StreamCell | RecordCell | ArrayCell):
        return _cell_text(cell, capped=False)
    return str(cell)


def render_json(result: TableResult) -> str:
    """`result` as a JSON array of objects, one per row, keyed by column.

    Ends with a trailing newline, as `render_csv` does, so a caller writes
    or prints it verbatim. Objects rather than arrays of values: a column
    name is what makes a row readable, and the columns are right there.
    """
    rows = [
        {name: _cell_json(cell) for name, cell in zip(result.columns, row)}
        for row in result.rows
    ]
    return json.dumps(rows, indent=2, ensure_ascii=False) + "\n"
