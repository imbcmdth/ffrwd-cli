"""Tests for the table/CSV renderer.

Format is PINNED byte-for-byte by cookbook recipes 30-32 (see
tests/test_examples.py, which runs those through the CLI); this file is the
renderer's own unit coverage -- widths, rstrip, empty cells, the "(1 row)"
singular, and the stream-placeholder forms -- built directly against
:class:`~ffrwd.table.TableResult` rather than a compiled query.
"""

from __future__ import annotations

from ffrwd.table import (
    ArrayCell,
    RecordCell,
    StreamCell,
    TableResult,
    render_csv,
    render_table,
)


def test_widths_are_max_of_header_and_values() -> None:
    result = TableResult(columns=["a", "bb"], rows=[["x", "y"], ["long", "z"]])
    text = render_table(result)
    lines = text.split("\n")
    assert lines[0] == " a    | bb"
    assert lines[1] == "------+----"


def test_every_line_is_rstripped() -> None:
    result = TableResult(columns=["a", "b"], rows=[["x", ""]])
    text = render_table(result)
    for line in text.split("\n"):
        assert line == line.rstrip()
    # An empty last cell ends the row right at its "|".
    assert text.split("\n")[2] == " x |"


def test_null_cell_is_empty() -> None:
    result = TableResult(columns=["a"], rows=[[None]])
    text = render_table(result)
    # A single-column NULL row's only cell IS the last one, so the whole
    # line rstrips away to nothing.
    assert text.split("\n")[2] == ""


def test_footer_uses_singular_for_one_row() -> None:
    result = TableResult(columns=["a"], rows=[["x"]])
    assert render_table(result).endswith("(1 row)")


def test_footer_uses_plural_for_zero_and_many_rows() -> None:
    assert render_table(TableResult(columns=["a"], rows=[])).endswith("(0 rows)")
    assert render_table(
        TableResult(columns=["a"], rows=[["x"], ["y"], ["z"]])
    ).endswith("(3 rows)")


def test_stream_cell_source_passthrough_placeholder() -> None:
    result = TableResult(columns=["t"], rows=[[StreamCell(type="video", spec="0:v:0")]])
    text = render_table(result)
    assert "<video 0:v:0>" in text


def test_stream_cell_filtered_node_placeholder() -> None:
    result = TableResult(columns=["t"], rows=[[StreamCell(type="audio", spec="n2")]])
    text = render_table(result)
    assert "<audio n2>" in text


def test_int_and_float_cells_render_via_str() -> None:
    result = TableResult(columns=["n", "f"], rows=[[1, 2.5]])
    text = render_table(result)
    assert "1" in text.split("\n")[2]
    assert "2.5" in text.split("\n")[2]


def test_recipe_30_shape_byte_for_byte() -> None:
    """Cross-check against the pinned recipe 30 bytes without compiling."""
    result = TableResult(
        columns=["index", "language", "codec", "channel_layout"],
        rows=[[1, "eng", "aac", "mono"], [2, "fra", "aac", "mono"]],
    )
    assert render_table(result) == (
        " index | language | codec | channel_layout\n"
        "-------+----------+-------+----------------\n"
        " 1     | eng      | aac   | mono\n"
        " 2     | fra      | aac   | mono\n"
        "(2 rows)"
    )


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------


def test_csv_header_true_emits_column_row() -> None:
    result = TableResult(columns=["a", "b"], rows=[["x", "y"]])
    assert render_csv(result, header=True) == "a,b\nx,y\n"


def test_csv_header_false_omits_column_row() -> None:
    result = TableResult(columns=["a", "b"], rows=[["x", "y"]])
    assert render_csv(result, header=False) == "x,y\n"


def test_csv_null_cell_is_empty_field() -> None:
    result = TableResult(columns=["a", "b"], rows=[["x", None]])
    assert render_csv(result, header=False) == "x,\n"


def test_csv_quotes_only_when_needed() -> None:
    result = TableResult(columns=["a"], rows=[["plain"], ["has,comma"], ['has"quote']])
    text = render_csv(result, header=False)
    lines = text.splitlines()
    assert lines[0] == "plain"
    assert lines[1] == '"has,comma"'
    assert lines[2] == '"has""quote"'


def test_csv_uses_lf_not_crlf() -> None:
    result = TableResult(columns=["a"], rows=[["x"], ["y"]])
    text = render_csv(result, header=False)
    assert "\r" not in text


def test_csv_stream_cell_placeholder() -> None:
    result = TableResult(columns=["t"], rows=[[StreamCell(type="audio", spec="0:a:0")]])
    assert render_csv(result, header=False) == "<audio 0:a:0>\n"


# ---------------------------------------------------------------------------
# ArrayCell
# ---------------------------------------------------------------------------


def test_array_cell_braces_the_comma_joined_elements() -> None:
    result = TableResult(
        columns=["a"],
        rows=[
            [
                ArrayCell(
                    elements=(
                        StreamCell(type="audio", spec="0:a:0"),
                        StreamCell(type="audio", spec="0:a:1"),
                    )
                )
            ]
        ],
    )
    text = render_table(result)
    assert "{<audio 0:a:0>,<audio 0:a:1>}" in text.split("\n")[2]


def test_array_cell_single_element_still_braces() -> None:
    result = TableResult(
        columns=["v"],
        rows=[[ArrayCell(elements=(StreamCell(type="video", spec="0:v:0"),))]],
    )
    text = render_table(result)
    assert text.split("\n")[2].strip() == "{<video 0:v:0>}"


def test_csv_array_cell_placeholder() -> None:
    result = TableResult(
        columns=["a"],
        rows=[
            [
                ArrayCell(
                    elements=(
                        StreamCell(type="audio", spec="0:a:0"),
                        StreamCell(type="audio", spec="0:a:1"),
                    )
                )
            ]
        ],
    )
    # The comma inside the braces is a real field separator to `csv`, which
    # quotes the whole field -- stock `QUOTE_MINIMAL`, same as any other value.
    assert render_csv(result, header=False) == '"{<audio 0:a:0>,<audio 0:a:1>}"\n'


# ---------------------------------------------------------------------------
# RecordCell
# ---------------------------------------------------------------------------


_CHAPTER_ARRAY = ArrayCell(
    elements=(
        RecordCell(fields=(1, "Intro", 0.0, 1.0)),
        RecordCell(fields=(2, "Chapter 1", 1.0, 2.0)),
    )
)


def test_record_cell_parenthesizes_the_comma_joined_fields() -> None:
    result = TableResult(columns=["chapters"], rows=[[_CHAPTER_ARRAY]])
    text = render_table(result)
    assert text.split("\n")[2].strip() == "{(1,Intro,0.0,1.0),(2,Chapter 1,1.0,2.0)}"


def test_record_cell_null_field_is_empty() -> None:
    result = TableResult(columns=["c"], rows=[[RecordCell(fields=(1, None, 0.0, 1.0))]])
    assert render_table(result).split("\n")[2].strip() == "(1,,0.0,1.0)"


def test_csv_record_array_is_one_quoted_field() -> None:
    result = TableResult(columns=["chapters"], rows=[[_CHAPTER_ARRAY]])
    assert render_csv(result, header=False) == (
        '"{(1,Intro,0.0,1.0),(2,Chapter 1,1.0,2.0)}"\n'
    )
