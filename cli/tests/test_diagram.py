"""Tests for the mermaid renderer behind ``explain --mermaid``.

Bare-machine by rule: synthetic graphs, a stubbed ``describe`` for the
partitioned plan, no ffmpeg and no fixture files. Assertions are structural
-- the node and edge lines, the subgraph count, balanced delimiters -- never
a screenshot of the whole text.
"""

from __future__ import annotations

import re

from ffrwd.compiler import Compiled, compile_all
from ffrwd.diagram import render_diagram
from ffrwd.ir import Graph, Node, Output, SinkUnit
from ffrwd.wasm import Described


def _lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines()]


def _chain_graph(path: str = "out.mp4") -> Graph:
    """input -> scale -> sink, the smallest complete shape."""
    return Graph(
        input_paths=["x.mp4"],
        sources={"a": 0},
        nodes={
            "n1": Node(
                id="n1",
                filter="scale",
                args={"width": 640, "height": 480},
                inputs=["src:a:v:0"],
                outputs=["video"],
            ),
        },
        sinks=[
            SinkUnit(
                outputs=[Output(ref="n1", type="video", name=None, metadata={})],
                path=path,
            )
        ],
    )


def _split_graph() -> Graph:
    """One split node feeding two output files."""
    return Graph(
        input_paths=["x.mp4"],
        sources={"a": 0},
        nodes={
            "n1": Node(
                id="n1",
                filter="split",
                args={"n": 2},
                inputs=["src:a:v:0"],
                outputs=["video", "video"],
            ),
        },
        sinks=[
            SinkUnit(
                outputs=[Output(ref="n1:0", type="video", name=None, metadata={})],
                path="one.mp4",
            ),
            SinkUnit(
                outputs=[Output(ref="n1:1", type="video", name=None, metadata={})],
                path="two.mp4",
            ),
        ],
    )


def _balanced(text: str) -> None:
    """Every bracket the renderer writes closes, subgraphs included.

    Quoted label content is stripped first: a label may carry anything,
    an elided JSON predicate included, and only the syntax around it has
    to balance.
    """
    assert text.count("subgraph ") == sum(1 for line in _lines(text) if line == "end")
    for line in _lines(text):
        assert line.count('"') % 2 == 0, line
    syntax = re.sub(r'"[^"]*"', '""', text)
    for open_char, close_char in ("[]", "()", "{}"):
        assert syntax.count(open_char) == syntax.count(close_char), text


# ---------------------------------------------------------------------------
# one command, flat
# ---------------------------------------------------------------------------


def test_a_plain_chain_draws_input_filter_and_sink() -> None:
    text = render_diagram([_chain_graph()])
    lines = _lines(text)
    assert lines[0] == "flowchart LR"
    assert 'in0(["x.mp4"])' in lines
    assert 'n1["scale(width=640, height=480)"]' in lines
    assert "in0 --> n1" in lines
    assert 'out0(["out.mp4"])' in lines
    assert "n1 --> out0" in lines
    assert "subgraph" not in text
    _balanced(text)


def test_a_split_draws_as_its_fan_out() -> None:
    lines = _lines(render_diagram([_split_graph()]))
    assert 'n1["split(n=2)"]' in lines
    assert "n1 --> out0" in lines
    assert "n1 --> out1" in lines


def test_a_sink_with_no_destination_still_draws() -> None:
    graph = _chain_graph()
    graph.sinks[0].path = None
    assert 'out0(["no destination"])' in _lines(render_diagram([graph]))


def test_a_long_option_list_is_elided() -> None:
    graph = _chain_graph()
    graph.nodes["n1"].args = {f"option{i}": "value" * 4 for i in range(6)}
    label_line = next(line for line in _lines(render_diagram([graph])) if line.startswith("n1["))
    assert label_line.endswith(', ...)"]')
    assert len(label_line) < 80


def test_reserved_characters_leave_labels_as_entities() -> None:
    graph = _chain_graph()
    graph.nodes["n1"].args = {"pred": '{"a"|<b>&}'}
    label_line = next(line for line in _lines(render_diagram([graph])) if line.startswith("n1["))
    for entity in ("#quot;", "#124;", "#lt;", "#gt;", "#amp;"):
        assert entity in label_line
    # The label's own two delimiters are the only raw quotes left.
    assert label_line.count('"') == 2
    assert "|" not in label_line


# ---------------------------------------------------------------------------
# a command sequence
# ---------------------------------------------------------------------------


def test_a_sequence_draws_one_subgraph_per_command() -> None:
    text = render_diagram([_chain_graph("pass1.mkv"), _chain_graph("out.mp4")])
    lines = _lines(text)
    assert text.count("subgraph ") == 2
    assert 'subgraph cmd0 ["command 1"]' in lines
    assert 'subgraph cmd1 ["command 2"]' in lines
    # Ids are per-command, so the same node name never collides.
    assert "cmd0_in0 --> cmd0_n1" in lines
    assert "cmd1_in0 --> cmd1_n1" in lines
    _balanced(text)


# ---------------------------------------------------------------------------
# a partitioned plan, rowfilter included
# ---------------------------------------------------------------------------

SEGMENTER = "modules/segment.wasm"
SELECTOR = "modules/mask_select.wasm"
MASKER = "modules/blur_mask.wasm"
OBJECT = (
    "STRUCT(id number, class text, score number, x number, y number, "
    "w number, h number)[]"
)

_OBJECT_ROWS: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["id", "class", "score", "x", "y", "w", "h"],
    "properties": {
        "id": {"type": "integer"},
        "class": {"type": "string"},
        "score": {"type": "number"},
        "x": {"type": "integer"},
        "y": {"type": "integer"},
        "w": {"type": "integer"},
        "h": {"type": "integer"},
    },
}

_DESCRIBED = {
    SEGMENTER: Described(
        world="ffrwd:av@0.9.0",
        name="segment",
        version="0.1.0",
        params_schema={"type": "object", "properties": {}},
        rows_schema=_OBJECT_ROWS,
        pixel_formats=("yuv420p", "rgba"),
    ),
    SELECTOR: Described(
        world="ffrwd:av@0.9.0",
        name="mask_select",
        version="0.1.0",
        params_schema={"type": "object", "properties": {}},
        pixel_formats=("yuv420p", "rgba"),
        meta=True,
    ),
    MASKER: Described(
        world="ffrwd:av@0.9.0",
        name="blur_mask",
        version="0.1.0",
        params_schema={"type": "object", "properties": {}},
        pixel_formats=("yuv420p", "rgba"),
        inputs=2,
        windowed=True,
    ),
}


def _segment_compiled() -> Compiled:
    """The blur-where-masked showcase, compiled with `describe` from memory."""
    sql = (
        f"CREATE FUNCTION segment(v video_stream)\n"
        f"RETURNS STRUCT(map video_stream, objects {OBJECT})\n"
        f"  AS '{SEGMENTER}', 'segment' LANGUAGE wasm;\n"
        f"CREATE FUNCTION mask_select(v video_stream, objects {OBJECT})\n"
        f"RETURNS video_stream AS '{SELECTOR}', 'mask_select' LANGUAGE wasm;\n"
        f"CREATE FUNCTION blur_mask(v video_stream, m video_stream)\n"
        f"RETURNS video_stream AS '{MASKER}', 'blur_mask' LANGUAGE wasm;\n"
        "COPY (SELECT blur_mask(s.video[1], mask_select(segment(s.video[1]).map,\n"
        "      ARRAY(SELECT o FROM unnest(segment(s.video[1]).objects) o\n"
        "            WHERE o.class = 'person')))\n"
        "      FROM input('a.mp4') s) TO 'out.mp4'"
    )
    return compile_all(sql, describe=lambda path: _DESCRIBED[path])


def test_a_partitioned_plan_draws_one_subgraph_per_process() -> None:
    compiled = _segment_compiled()
    plan = compiled.plan
    assert plan is not None
    text = render_diagram(compiled.graphs, plan)
    for process in plan.processes:
        assert f"subgraph {process.id} " in text
    assert text.count("subgraph ") == len(plan.processes)
    _balanced(text)


def test_the_sidecar_subgraph_holds_its_module_chain() -> None:
    compiled = _segment_compiled()
    text = render_diagram(compiled.graphs, compiled.plan)
    for module in ("segment", "mask_select", "blur_mask"):
        assert f'["{module}"]' in text
    # The hosted row filter is a node like any other, its predicate elided
    # into the label.
    assert "rowfilter(pred=" in text


def test_pipe_edges_are_labelled_with_what_crosses_them() -> None:
    compiled = _segment_compiled()
    plan = compiled.plan
    assert plan is not None
    lines = _lines(render_diagram(compiled.graphs, plan))
    for edge in plan.stream_edges:
        wanted = f"{edge.source} -->|{edge.format.container} {edge.format.codec}| {edge.target}"
        assert wanted in lines
    assert any("-->|nut rawvideo|" in line for line in lines)


def test_pipe_ends_stay_out_of_the_subgraphs() -> None:
    """The plan's own edges draw the pipes, so no stadium node repeats them."""
    compiled = _segment_compiled()
    text = render_diagram(compiled.graphs, compiled.plan)
    assert '(["pipe:"])' not in text
    assert '(["a.mp4"])' in text
    assert '(["out.mp4"])' in text
