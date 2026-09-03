"""Tests for the emit pass.

Graphs are hand-built with ir.Node/ir.Graph/ir.Output -- emit must not depend
on the parser/lower/split modules.

The ``@pytest.mark.exec`` test at the bottom runs real ffmpeg against the
generated fixtures (``python scripts/gen_fixtures.py``); it is excluded from
the default run by ``addopts = -m "not exec"``.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from ffrwd.emit import (
    Emitted,
    OutputGroup,
    OutputMap,
    _drop_unused_url_inputs,
    _escape_value,
    _render_command,
    build_ffmpeg_args,
    build_ffmpeg_commands,
    emit,
)
from ffrwd.errors import ErrorCode, FfrwdError
from ffrwd.ir import (
    NO_CHAPTERS,
    NO_METADATA,
    Attachment,
    Graph,
    Node,
    Output,
    SinkUnit,
    StreamType,
    UrlSource,
    UrlSourceRow,
)


def _node(
    node_id: str,
    filter_name: str,
    args: dict[str, object],
    inputs: list[str],
    outputs: list[StreamType] | None = None,
) -> Node:
    return Node(
        id=node_id,
        filter=filter_name,
        args=dict(args),
        inputs=list(inputs),
        outputs=list(outputs) if outputs is not None else ["video"],
    )


def _out(
    ref: str,
    type: StreamType = "video",
    *,
    name: str | None = None,
    metadata: dict[str, str] | None = None,
) -> Output:
    return Output(ref=ref, type=type, name=name, metadata=dict(metadata or {}))


def _sink(
    path: str,
    options: dict[str, object] | None = None,
    tags: dict[str, str | None] | None = None,
    window: tuple[float | None, float | None] | None = None,
    chapters: int | None = None,
    metadata: int | None = None,
    attachments: list[Attachment] | None = None,
) -> SinkUnit:
    """A destination with no outputs yet -- `_graph` fills them in."""
    return SinkUnit(
        outputs=[],
        path=path,
        options=dict(options or {}),
        tags=dict(tags or {}),
        window=window,
        chapters=chapters,
        metadata=metadata,
        attachments=list(attachments or []),
    )


def _graph(
    nodes: list[Node],
    outputs: list[Output],
    *,
    input_paths: list[str] | None = None,
    sources: dict[str, int] | None = None,
    sink: SinkUnit | None = None,
    sinks: list[SinkUnit] | None = None,
    input_trims: dict[str, tuple[float | None, float | None]] | None = None,
    input_options: dict[str, dict[str, object]] | None = None,
    url_sources: dict[str, UrlSource] | None = None,
) -> Graph:
    """A ONE-sink graph carrying `outputs` (`sink` names its destination).

    `sinks` is the multi-sink escape hatch: pass whole
    :class:`SinkUnit`s, each with its own outputs, and `outputs` is ignored.
    """
    if sinks is not None:
        units = list(sinks)
    elif sink is not None:
        units = [
            SinkUnit(
                outputs=list(outputs),
                path=sink.path,
                options=dict(sink.options),
                tags=dict(sink.tags),
                window=sink.window,
                chapters=sink.chapters,
                metadata=sink.metadata,
                attachments=list(sink.attachments),
            )
        ]
    else:
        units = [SinkUnit(outputs=list(outputs))]
    return Graph(
        input_paths=list(input_paths or ["a.mp4"]),
        sources=dict(sources or {"a": 0}),
        nodes={n.id: n for n in nodes},
        sinks=units,
        input_trims=dict(input_trims or {}),
        input_options={k: dict(v) for k, v in (input_options or {}).items()},
        url_sources=dict(url_sources or {}),
    )


# ---------------------------------------------------------------------------
# chains, labels, semicolons
# ---------------------------------------------------------------------------


def test_single_node_graph() -> None:
    g = _graph(
        [_node("n1", "crop", {"w": 600, "h": 200, "x": 1200, "y": 50}, ["src:a:v:0"])],
        [_out("n1")],
    )
    e = emit(g)
    assert e.filter_complex == "[0:v:0]crop=w=600:h=200:x=1200:y=50[out0]"
    assert e.inputs == ["a.mp4"]
    assert e.maps == [OutputMap(target="[out0]", type="video", copy=False, metadata={})]
    assert isinstance(e, Emitted)


def test_readme_example_shape() -> None:
    """WITH pip AS (scale(crop(b.video[1],...), 0.5)) SELECT overlay(a.video[1], pip.frame, 20, 20).

    `a` and `b` are both untrimmed, option-free aliases of the SAME path, so
    input dedup folds them onto one `-i`: both source refs render
    as `[0:v:0]` (see emit.py's "Input dedup" docstring section).
    """
    g = _graph(
        [
            _node("n1", "crop", {"w": 600, "h": 200, "x": 1200, "y": 50}, ["src:b:v:0"]),
            _node("n2", "scale", {"w": "iw*0.5", "h": "-2"}, ["n1"]),
            _node("n3", "overlay", {"x": 20, "y": 20}, ["src:a:v:0", "n2"]),
        ],
        [_out("n3")],
        input_paths=["game.mp4", "game.mp4"],
        sources={"a": 0, "b": 1},
    )
    e = emit(g)
    assert e.filter_complex == (
        "[0:v:0]crop=w=600:h=200:x=1200:y=50,scale=w=iw*0.5:h=-2[n2];"
        "[0:v:0][n2]overlay=x=20:y=20[out0]"
    )
    assert e.inputs == ["game.mp4"]  # deduped: one -i, not two
    # one comma-chain (crop,scale) and one semicolon between the two chains
    assert e.filter_complex.count(";") == 1
    assert ",scale=" in e.filter_complex


def test_long_linear_run_merges_into_one_chain() -> None:
    g = _graph(
        [
            _node("n1", "trim", {"start": 1, "end": 5}, ["src:a:v:0"]),
            _node("n2", "setpts", {"expr": "PTS-STARTPTS"}, ["n1"]),
            _node("n3", "hflip", {}, ["n2"]),
            _node("n4", "gblur", {"sigma": 2.5}, ["n3"]),
        ],
        [_out("n4")],
    )
    e = emit(g)
    assert e.filter_complex == (
        "[0:v:0]trim=start=1:end=5,setpts=PTS-STARTPTS,hflip,gblur=sigma=2.5[out0]"
    )
    assert ";" not in e.filter_complex


def test_audio_source_ref_renders_typed_index() -> None:
    g = _graph(
        [_node("n1", "volume", {"volume": 0.5}, ["src:a:a:1"], ["audio"])],
        [_out("n1", "audio")],
    )
    e = emit(g)
    assert e.filter_complex == "[0:a:1]volume=volume=0.5[out0]"
    assert e.maps == [OutputMap(target="[out0]", type="audio", copy=False, metadata={})]


def test_diamond_post_split_labels_and_semicolons() -> None:
    """src:a:v:0 split in two, one branch blurred, both recombined by overlay."""
    g = _graph(
        [
            _node("src_a_split", "split", {"n": 2}, ["src:a:v:0"], ["video", "video"]),
            _node("n1", "gblur", {"sigma": 5}, ["src_a_split:1"]),
            _node("n2", "overlay", {"x": 0, "y": 0}, ["src_a_split:0", "n1"]),
        ],
        [_out("n2")],
    )
    e = emit(g)
    assert e.filter_complex == (
        "[0:v:0]split=2[src_a_split0][src_a_split1];"
        "[src_a_split1]gblur=sigma=5[n1];"
        "[src_a_split0][n1]overlay=x=0:y=0[out0]"
    )


def test_split_node_gets_one_label_per_declared_output_pad() -> None:
    """A split whose producer is not adjacent renders [n1]split=2[n1_split0][n1_split1]."""
    g = _graph(
        [
            _node("n1", "hflip", {}, ["src:a:v:0"]),
            _node("n2", "vflip", {}, ["src:b:v:0"]),
            _node("n1_split", "split", {"n": 2}, ["n1"], ["video", "video"]),
            _node("n3", "overlay", {"x": 0, "y": 0}, ["n1_split:0", "n2"]),
            _node("n4", "overlay", {"x": 10, "y": 10}, ["n1_split:1", "n3"]),
        ],
        [_out("n4")],
        input_paths=["a.mp4", "b.mp4"],
        sources={"a": 0, "b": 1},
    )
    e = emit(g)
    assert e.filter_complex == (
        "[0:v:0]hflip[n1];"
        "[1:v:0]vflip[n2];"
        "[n1]split=2[n1_split0][n1_split1];"
        "[n1_split0][n2]overlay=x=0:y=0[n3];"
        "[n1_split1][n3]overlay=x=10:y=10[out0]"
    )


def test_pad_count_comes_from_outputs_not_from_split_args() -> None:
    """``n`` is only an argument now; ``outputs`` decides how many pads render."""
    g = _graph(
        [
            _node("s", "split", {"n": 3}, ["src:a:v:0"], ["video", "video", "video"]),
            _node("n1", "hflip", {}, ["s:0"]),
            _node("n2", "vflip", {}, ["s:1"]),
            _node("n3", "gblur", {"sigma": 1}, ["s:2"]),
            _node("n4", "overlay", {"x": 0, "y": 0}, ["n1", "n2"]),
            _node("n5", "overlay", {"x": 0, "y": 0}, ["n4", "n3"]),
        ],
        [_out("n5")],
    )
    e = emit(g)
    assert e.filter_complex.startswith("[0:v:0]split=3[s0][s1][s2];")
    assert "[s0]hflip[n1];" in e.filter_complex
    assert "[s1]vflip[n2];" in e.filter_complex
    assert "[s2]gblur=sigma=1[n3];" in e.filter_complex
    assert e.filter_complex.endswith("[n4][n3]overlay=x=0:y=0[out0]")


def test_asplit_renders_value_only_args_and_audio_pads() -> None:
    g = _graph(
        [
            _node("s", "asplit", {"n": 2}, ["src:a:a:0"], ["audio", "audio"]),
            _node("n1", "volume", {"volume": 2}, ["s:0"], ["audio"]),
            _node("n2", "amix", {"inputs": 2}, ["s:1", "n1"], ["audio"]),
        ],
        [_out("n2", "audio")],
    )
    e = emit(g)
    assert e.filter_complex == (
        "[0:a:0]asplit=2[s0][s1];"
        "[s0]volume=volume=2[n1];"
        "[s1][n1]amix=inputs=2[out0]"
    )


def test_split_as_chain_tail_merges_with_its_producer() -> None:
    g = _graph(
        [
            _node("n1", "hflip", {}, ["src:a:v:0"]),
            _node("n1_split", "split", {"n": 2}, ["n1"], ["video", "video"]),
            _node("n2", "gblur", {"sigma": 3}, ["n1_split:0"]),
            _node("n3", "overlay", {"x": 0, "y": 0}, ["n1_split:1", "n2"]),
        ],
        [_out("n3")],
    )
    e = emit(g)
    assert e.filter_complex == (
        "[0:v:0]hflip,split=2[n1_split0][n1_split1];"
        "[n1_split0]gblur=sigma=3[n2];"
        "[n1_split1][n2]overlay=x=0:y=0[out0]"
    )


# ---------------------------------------------------------------------------
# multi-output maps, passthrough, metadata
# ---------------------------------------------------------------------------


def test_pure_passthrough_graph_has_empty_filter_complex() -> None:
    """SELECT a.video[1], a.audio[2] -- a remap, nothing re-encoded."""
    g = _graph([], [_out("src:a:v:0"), _out("src:a:a:1", "audio")])
    e = emit(g)
    assert e.filter_complex == ""
    assert e.maps == [
        OutputMap(target="0:v:0", type="video", copy=True, metadata={}),
        OutputMap(target="0:a:1", type="audio", copy=True, metadata={}),
    ]


def test_passthrough_from_second_input_uses_its_index() -> None:
    g = _graph(
        [],
        [_out("src:b:a:2", "audio")],
        input_paths=["a.mp4", "b.mp4"],
        sources={"a": 0, "b": 1},
    )
    assert emit(g).maps == [OutputMap(target="1:a:2", type="audio", copy=True, metadata={})]


def test_mixed_filtered_and_passthrough() -> None:
    """SELECT hflip(a.video[1]), a.audio[1] -- audio never enters the graph."""
    g = _graph(
        [_node("n1", "hflip", {}, ["src:a:v:0"])],
        [_out("n1"), _out("src:a:a:0", "audio", name="orig")],
    )
    e = emit(g)
    assert e.filter_complex == "[0:v:0]hflip[out0]"
    assert e.maps == [
        OutputMap(target="[out0]", type="video", copy=False, metadata={}),
        OutputMap(target="0:a:0", type="audio", copy=True, metadata={}),
    ]


def test_passthrough_first_still_labels_filtered_output_by_its_index() -> None:
    """Labels track the output index, not the count of filtered outputs."""
    g = _graph(
        [_node("n1", "volume", {"volume": 0.5}, ["src:a:a:0"], ["audio"])],
        [_out("src:a:v:0"), _out("n1", "audio")],
    )
    e = emit(g)
    assert e.filter_complex == "[0:a:0]volume=volume=0.5[out1]"
    assert e.maps == [
        OutputMap(target="0:v:0", type="video", copy=True, metadata={}),
        OutputMap(target="[out1]", type="audio", copy=False, metadata={}),
    ]


def test_two_filtered_outputs_are_labelled_in_graph_output_order() -> None:
    g = _graph(
        [
            _node("n1", "hflip", {}, ["src:a:v:0"]),
            _node("n2", "volume", {"volume": 2}, ["src:a:a:0"], ["audio"]),
        ],
        [_out("n2", "audio"), _out("n1")],  # deliberately not node order
    )
    e = emit(g)
    assert e.filter_complex == "[0:v:0]hflip[out1];[0:a:0]volume=volume=2[out0]"
    assert [m.target for m in e.maps] == ["[out0]", "[out1]"]
    assert [m.type for m in e.maps] == ["audio", "video"]


def test_concat_node_with_video_and_audio_pads() -> None:
    """UNION ALL of (video, audio) branches: concat n=2:v=1:a=1 -> two pads."""
    g = _graph(
        [
            _node(
                "n1",
                "concat",
                {"n": 2, "v": 1, "a": 1},
                ["src:a:v:0", "src:a:a:0", "src:b:v:0", "src:b:a:0"],
                ["video", "audio"],
            )
        ],
        [_out("n1"), _out("n1:1", "audio")],
        input_paths=["a.mp4", "b.mp4"],
        sources={"a": 0, "b": 1},
    )
    e = emit(g)
    assert e.filter_complex == (
        "[0:v:0][0:a:0][1:v:0][1:a:0]concat=n=2:v=1:a=1[out0][out1]"
    )
    assert e.maps == [
        OutputMap(target="[out0]", type="video", copy=False, metadata={}),
        OutputMap(target="[out1]", type="audio", copy=False, metadata={}),
    ]


def test_metadata_is_carried_onto_the_map() -> None:
    g = _graph(
        [_node("n1", "aecho", {"expr": "0.8:0.9:60:0.3"}, ["src:a:a:1"], ["audio"])],
        [_out("n1", "audio", name="fra", metadata={"language": "fra", "title": "VF"})],
    )
    e = emit(g)
    assert e.maps[0].metadata == {"language": "fra", "title": "VF"}
    assert e.maps[0].copy is False


def test_map_metadata_is_a_copy_not_the_output_dict() -> None:
    output = _out("src:a:a:0", "audio", metadata={"language": "eng"})
    g = _graph([], [output])
    e = emit(g)
    e.maps[0].metadata["language"] = "mangled"
    assert output.metadata == {"language": "eng"}


def test_output_label_wins_over_generated_label() -> None:
    """A node id of 'out0' does not steal the reserved output label."""
    g = _graph(
        [
            _node("out0", "hflip", {}, ["src:a:v:0"]),
            _node("n2", "vflip", {}, ["out0"]),
        ],
        [_out("n2")],
    )
    e = emit(g)
    # the 'out0' node is merged into the chain, so its label is elided entirely
    assert e.filter_complex == "[0:v:0]hflip,vflip[out0]"


def test_label_collision_is_broken() -> None:
    g = _graph(
        [
            _node("out0", "hflip", {}, ["src:a:v:0"]),
            _node("n2", "vflip", {}, ["src:b:v:0"]),
            _node("n3", "overlay", {"x": 0, "y": 0}, ["out0", "n2"]),
        ],
        [_out("n3")],
        input_paths=["a.mp4", "b.mp4"],
        sources={"a": 0, "b": 1},
    )
    e = emit(g)
    assert e.filter_complex == (
        "[0:v:0]hflip[out0_];[1:v:0]vflip[n2];[out0_][n2]overlay=x=0:y=0[out0]"
    )


def test_unused_output_index_is_still_reserved() -> None:
    """out0 belongs to the passthrough column even though it is never rendered."""
    g = _graph(
        [_node("out0", "hflip", {}, ["src:a:v:0"]), _node("n2", "vflip", {}, ["src:b:v:0"])],
        [_out("src:a:a:0", "audio"), _out("out0"), _out("n2")],
        input_paths=["a.mp4", "b.mp4"],
        sources={"a": 0, "b": 1},
    )
    e = emit(g)
    assert e.filter_complex == "[0:v:0]hflip[out1];[1:v:0]vflip[out2]"
    assert [m.target for m in e.maps] == ["0:a:0", "[out1]", "[out2]"]


def test_labels_are_sanitized() -> None:
    g = _graph(
        [
            _node("a.b-c", "hflip", {}, ["src:a:v:0"]),
            _node("n2", "vflip", {}, ["src:b:v:0"]),
            _node("n3", "overlay", {"x": 0, "y": 0}, ["a.b-c", "n2"]),
        ],
        [_out("n3")],
        input_paths=["a.mp4", "b.mp4"],
        sources={"a": 0, "b": 1},
    )
    assert "[a_b_c]" in emit(g).filter_complex


# ---------------------------------------------------------------------------
# zero-input (generated source) nodes
# ---------------------------------------------------------------------------
#
# `FROM ffmpeg.testsrc(duration => 2) t` lowers to a node with `inputs=[]`.
# Nothing in this pass needed changing for it (verified, not assumed): a chain
# head with no inputs renders no input labels, `_extends` refuses to
# comma-append a node that does not take exactly one input (so a source always
# STARTS a chain), and `_verify_topological` / `_check_fanout` have nothing to
# say about an empty input list. These tests pin that.


def _source_graph(nodes: list[Node], outputs: list[Output]) -> Graph:
    """A graph with NO inputs at all -- `_graph`'s defaults would add one."""
    return Graph(
        input_paths=[],
        sources={},
        nodes={n.id: n for n in nodes},
        sinks=[SinkUnit(outputs=list(outputs))],
    )


def test_a_zero_input_node_renders_as_a_chain_head_with_no_input_labels() -> None:
    g = _source_graph([_node("n1", "testsrc", {"duration": 2}, [])], [_out("n1")])
    assert emit(g).filter_complex == "testsrc=duration=2[out0]"


def test_a_graph_of_only_sources_needs_no_input_at_all() -> None:
    """No `-i` anywhere: the whole command is a filtergraph and its maps."""
    g = _source_graph(
        [
            _node("n1", "testsrc", {"duration": 2}, []),
            _node("n2", "anullsrc", {"duration": 2}, [], ["audio"]),
        ],
        [_out("n1"), _out("n2", "audio")],
    )
    e = emit(g)
    assert e.filter_complex == "testsrc=duration=2[out0];anullsrc=duration=2[out1]"
    assert build_ffmpeg_args(e, "out.mp4") == [
        "ffmpeg",
        "-filter_complex",
        "testsrc=duration=2[out0];anullsrc=duration=2[out1]",
        "-map",
        "[out0]",
        "-map",
        "[out1]",
        "out.mp4",
    ]


def test_a_zero_input_node_merges_into_a_chain_with_what_follows_it() -> None:
    g = _source_graph(
        [
            _node("n1", "sine", {"frequency": 440}, [], ["audio"]),
            _node("n2", "volume", {"volume": 0.5}, ["n1"], ["audio"]),
        ],
        [_out("n2", "audio")],
    )
    assert emit(g).filter_complex == "sine=frequency=440,volume=volume=0.5[out0]"


def test_a_zero_input_node_never_extends_the_previous_chain() -> None:
    """A source takes no inputs, so it can never be the comma-continuation of
    anything -- it always starts its own chain."""
    g = _graph(
        [
            _node("n1", "hflip", {}, ["src:a:v:0"]),
            _node("n2", "testsrc", {"duration": 1}, []),
        ],
        [_out("n1"), _out("n2")],
    )
    assert emit(g).filter_complex == "[0:v:0]hflip[out0];testsrc=duration=1[out1]"


def test_a_split_of_a_zero_input_node_chains_off_it() -> None:
    g = _source_graph(
        [
            _node("n1", "testsrc", {"duration": 2}, []),
            _node("n1_split", "split", {"n": 2}, ["n1"], ["video", "video"]),
            _node("n2", "hflip", {}, ["n1_split:0"]),
        ],
        [_out("n2"), _out("n1_split:1")],
    )
    assert emit(g).filter_complex == (
        "testsrc=duration=2,split=2[n1_split0][out1];[n1_split0]hflip[out0]"
    )


def test_zero_input_nodes_feed_concat_alongside_real_inputs() -> None:
    """The silent-audio-for-concat shape, generated sources' headline."""
    g = _graph(
        [
            _node("n1", "testsrc2", {"duration": 1}, []),
            _node("n2", "anullsrc", {"duration": 1}, [], ["audio"]),
            _node(
                "n3",
                "concat",
                {"n": 2, "v": 1, "a": 1},
                ["src:a:v:0", "src:a:a:0", "n1", "n2"],
                ["video", "audio"],
            ),
        ],
        [_out("n3:0"), _out("n3:1", "audio")],
    )
    assert emit(g).filter_complex == (
        "testsrc2=duration=1[n1];anullsrc=duration=1[n2];"
        "[0:v:0][0:a:0][n1][n2]concat=n=2:v=1:a=1[out0][out1]"
    )


# ---------------------------------------------------------------------------
# argument rendering
# ---------------------------------------------------------------------------


def test_bare_filter_renders_without_equals() -> None:
    g = _graph([_node("n1", "hflip", {}, ["src:a:v:0"])], [_out("n1")])
    assert emit(g).filter_complex == "[0:v:0]hflip[out0]"


def test_expr_key_renders_value_only() -> None:
    g = _graph([_node("n1", "setpts", {"expr": "PTS/2"}, ["src:a:v:0"])], [_out("n1")])
    assert emit(g).filter_complex == "[0:v:0]setpts=PTS/2[out0]"


def test_concat_renders_named_args() -> None:
    g = _graph(
        [_node("n1", "concat", {"n": 2, "v": 1, "a": 0}, ["src:a:v:0", "src:b:v:0"])],
        [_out("n1")],
        input_paths=["a.mp4", "b.mp4"],
        sources={"a": 0, "b": 1},
    )
    assert emit(g).filter_complex == "[0:v:0][1:v:0]concat=n=2:v=1:a=0[out0]"


def test_args_render_in_insertion_order_not_sorted() -> None:
    g = _graph([_node("n1", "crop", {"w": 1, "h": 2, "x": 3, "y": 4}, ["src:a:v:0"])], [_out("n1")])
    assert emit(g).filter_complex == "[0:v:0]crop=w=1:h=2:x=3:y=4[out0]"
    g2 = _graph(
        [_node("n1", "crop", {"y": 4, "x": 3, "h": 2, "w": 1}, ["src:a:v:0"])], [_out("n1")]
    )
    assert emit(g2).filter_complex == "[0:v:0]crop=y=4:x=3:h=2:w=1[out0]"


def test_scalar_types_render() -> None:
    g = _graph(
        [_node("n1", "f", {"i": 3, "f": 0.5, "s": "auto", "b": True, "z": False}, ["src:a:v:0"])],
        [_out("n1")],
    )
    assert emit(g).filter_complex == "[0:v:0]f=i=3:f=0.5:s=auto:b=1:z=0[out0]"


def test_unrenderable_arg_type_is_internal() -> None:
    g = _graph([_node("n1", "f", {"k": [1, 2]}, ["src:a:v:0"])], [_out("n1")])
    with pytest.raises(FfrwdError) as excinfo:
        emit(g)
    assert excinfo.value.code is ErrorCode.INTERNAL


def test_drawtext_value_goes_through_the_escaper() -> None:
    g = _graph(
        [_node("n1", "drawtext", {"text": "12:30, take 'one'", "x": 10}, ["src:a:v:0"])],
        [_out("n1")],
    )
    assert emit(g).filter_complex == (
        r"[0:v:0]drawtext=text=12\\:30\,\ take\ \\\'one\\\':x=10[out0]"
    )


# ---------------------------------------------------------------------------
# escaping (_escape_value unit tests)
# ---------------------------------------------------------------------------


def test_escaping_maps_every_value_to_its_two_level_graph_form() -> None:
    # ':' and '=' separate filter options (level 1): '\:' there, so '\\:' in
    # the graph. ',' ';' '[' ']' space and '#' only matter at the filtergraph
    # level. A "'" is turned into "\'" at level 1, then both characters are
    # escaped again. Values with no special character pass through untouched.
    escaped = {
        "hflip": "hflip",
        "1200": "1200",
        "-2": "-2",
        "iw*0.5": "iw*0.5",
        "PTS-STARTPTS": "PTS-STARTPTS",
        "0x00FF00@0.5": "0x00FF00@0.5",
        "in_w/2": "in_w/2",
        "a.b+c%d": "a.b+c%d",
        "12:30": r"12\\:30",
        "w=2": r"w\\=2",
        "it's": r"it\\\'s",
        "a,b": r"a\,b",
        "a;b[c]d": r"a\;b\[c\]d",
        "C:\\clips": r"C\\:\\\\clips",
        "hello world #1": r"hello\ world\ \#1",
        # The worked example from ffmpeg-filters "Notes on filtergraph
        # escaping". The docs stop at \\\'string\\\' / \\: / \, ; we also
        # backslash-escape spaces, which is redundant-but-valid.
        "this is a 'string': may contain one, or more, special characters": (
            r"this\ is\ a\ \\\'string\\\'\\:\ may\ contain\ one\,\ or\ more\,\ "
            r"special\ characters"
        ),
    }
    assert {plain: _escape_value(plain) for plain in escaped} == escaped
    # ffmpeg unescapes '\ ' back to a plain space, leaving the documented form.
    doc_example = escaped["this is a 'string': may contain one, or more, special characters"]
    assert doc_example.replace("\\ ", " ") == (
        r"this is a \\\'string\\\'\\: may contain one\, or more\, special characters"
    )


def test_escape_empty_string_is_quoted() -> None:
    assert _escape_value("") == "''"


# ---------------------------------------------------------------------------
# malformed graphs -> INTERNAL
# ---------------------------------------------------------------------------


def _assert_internal(g: Graph) -> FfrwdError:
    with pytest.raises(FfrwdError) as excinfo:
        emit(g)
    assert excinfo.value.code is ErrorCode.INTERNAL
    return excinfo.value


def test_cycle_is_internal_error() -> None:
    g = _graph(
        [_node("n1", "hflip", {}, ["n2"]), _node("n2", "vflip", {}, ["n1"])],
        [_out("n2")],
    )
    err = _assert_internal(g)
    assert "topological" in err.message


def test_forward_reference_is_internal_error() -> None:
    g = _graph(
        [_node("n1", "vflip", {}, ["n2"]), _node("n2", "hflip", {}, ["src:a:v:0"])],
        [_out("n1")],
    )
    _assert_internal(g)


def test_unknown_node_ref_is_internal_error() -> None:
    g = _graph([_node("n1", "hflip", {}, ["nope"])], [_out("n1")])
    err = _assert_internal(g)
    assert "unknown node" in err.message


def test_unknown_source_alias_is_internal_error() -> None:
    g = _graph([_node("n1", "hflip", {}, ["src:zz:v:0"])], [_out("n1")])
    err = _assert_internal(g)
    assert "unknown source alias" in err.message


def test_malformed_source_ref_is_internal_error() -> None:
    g = _graph([_node("n1", "hflip", {}, ["src:a:x:0"])], [_out("n1")])
    err = _assert_internal(g)
    assert "malformed source ref" in err.message


def test_untyped_v1_source_ref_is_internal_error() -> None:
    """v0/v1's untyped ``src:a`` is retired and must not silently render."""
    _assert_internal(_graph([_node("n1", "hflip", {}, ["src:a"])], [_out("n1")]))


def test_pad_index_past_output_count_is_internal_error() -> None:
    g = _graph(
        [
            _node("s", "split", {"n": 2}, ["src:a:v:0"], ["video", "video"]),
            _node("n1", "hflip", {}, ["s:0"]),
            _node("n2", "vflip", {}, ["s:5"]),
            _node("n3", "overlay", {"x": 0, "y": 0}, ["n1", "n2"]),
        ],
        [_out("n3")],
    )
    _assert_internal(g)


def test_node_without_output_pads_is_internal_error() -> None:
    _assert_internal(_graph([_node("n1", "hflip", {}, ["src:a:v:0"], [])], [_out("n1")]))


def test_unsplit_fanout_is_internal_error() -> None:
    """emit refuses to render a pad consumed twice -- ffmpeg pads are consume-once."""
    g = _graph(
        [
            _node("n1", "hflip", {}, ["src:a:v:0"]),
            _node("n2", "gblur", {"sigma": 1}, ["n1"]),
            _node("n3", "overlay", {"x": 0, "y": 0}, ["n1", "n2"]),
        ],
        [_out("n3")],
    )
    err = _assert_internal(g)
    assert "consume-once" in err.message


def test_pad_feeding_both_a_node_and_an_output_is_internal_error() -> None:
    """An Output counts as a consumer, so this needs a split too."""
    g = _graph(
        [
            _node("n1", "hflip", {}, ["src:a:v:0"]),
            _node("n2", "gblur", {"sigma": 1}, ["n1"]),
        ],
        [_out("n1"), _out("n2")],
    )
    err = _assert_internal(g)
    assert "consume-once" in err.message


def test_src_pad_used_by_a_node_and_an_output_is_internal_error() -> None:
    g = _graph(
        [_node("n1", "hflip", {}, ["src:a:v:0"])],
        [_out("n1"), _out("src:a:v:0")],
    )
    err = _assert_internal(g)
    assert "consume-once" in err.message


def test_two_outputs_naming_the_same_src_pad_is_internal_error() -> None:
    g = _graph([], [_out("src:a:a:0", "audio"), _out("src:a:a:0", "audio")])
    err = _assert_internal(g)
    assert "consume-once" in err.message


def test_two_outputs_naming_the_same_node_pad_is_internal_error() -> None:
    g = _graph([_node("n1", "hflip", {}, ["src:a:v:0"])], [_out("n1"), _out("n1:0")])
    err = _assert_internal(g)
    assert "consume-once" in err.message


def test_empty_output_ref_is_internal_error() -> None:
    _assert_internal(_graph([_node("n1", "hflip", {}, ["src:a:v:0"])], [_out("")]))


def test_no_outputs_is_internal_error() -> None:
    err = _assert_internal(_graph([_node("n1", "hflip", {}, ["src:a:v:0"])], []))
    assert "no outputs" in err.message


# ---------------------------------------------------------------------------
# build_ffmpeg_args
# ---------------------------------------------------------------------------


def test_build_ffmpeg_args_exact_list() -> None:
    """`a`/`b` dedup onto one `-i`: see test_readme_example_shape."""
    g = _graph(
        [
            _node("n1", "crop", {"w": 600, "h": 200, "x": 1200, "y": 50}, ["src:b:v:0"]),
            _node("n2", "scale", {"w": "iw*0.5", "h": "-2"}, ["n1"]),
            _node("n3", "overlay", {"x": 20, "y": 20}, ["src:a:v:0", "n2"]),
        ],
        [_out("n3")],
        input_paths=["game.mp4", "game.mp4"],
        sources={"a": 0, "b": 1},
    )
    e = emit(g)
    assert build_ffmpeg_args(e, "out.mp4") == [
        "ffmpeg",
        "-i",
        "game.mp4",
        "-filter_complex",
        "[0:v:0]crop=w=600:h=200:x=1200:y=50,scale=w=iw*0.5:h=-2[n2];"
        "[0:v:0][n2]overlay=x=20:y=20[out0]",
        "-map",
        "[out0]",
        "out.mp4",
    ]


def test_build_ffmpeg_args_single_input() -> None:
    e = Emitted(
        inputs=["a.mp4"],
        filter_complex="[0:v:0]hflip[out0]",
        groups=[
            OutputGroup(
                maps=[OutputMap(target="[out0]", type="video", copy=False, metadata={})]
            )
        ],
    )
    assert build_ffmpeg_args(e, "/tmp/o.mkv") == [
        "ffmpeg",
        "-i",
        "a.mp4",
        "-filter_complex",
        "[0:v:0]hflip[out0]",
        "-map",
        "[out0]",
        "/tmp/o.mkv",
    ]


def test_build_ffmpeg_args_omits_filter_complex_when_empty() -> None:
    """SELECT a.video[1], a.audio[2] -- a pure remap, no filtergraph at all."""
    g = _graph([], [_out("src:a:v:0"), _out("src:a:a:1", "audio")])
    assert build_ffmpeg_args(emit(g), "out.mp4") == [
        "ffmpeg",
        "-i",
        "a.mp4",
        "-map",
        "0:v:0",
        "-c:0",
        "copy",
        "-map",
        "0:a:1",
        "-c:1",
        "copy",
        "out.mp4",
    ]


def test_build_ffmpeg_args_mixed_with_metadata() -> None:
    g = _graph(
        [_node("n1", "hflip", {}, ["src:a:v:0"])],
        [
            _out("n1", metadata={"title": "flipped"}),
            _out("src:a:a:0", "audio", metadata={"language": "eng"}),
        ],
    )
    assert build_ffmpeg_args(emit(g), "out.mp4") == [
        "ffmpeg",
        "-i",
        "a.mp4",
        "-filter_complex",
        "[0:v:0]hflip[out0]",
        "-map",
        "[out0]",
        "-metadata:s:0",
        "title=flipped",
        "-map",
        "0:a:0",
        "-c:1",
        "copy",
        "-metadata:s:1",
        "language=eng",
        "out.mp4",
    ]


def test_build_ffmpeg_args_sorts_metadata_keys() -> None:
    e = Emitted(
        inputs=["a.mp4"],
        filter_complex="",
        groups=[
            OutputGroup(
                maps=[
                    OutputMap(
                        target="0:a:0",
                        type="audio",
                        copy=True,
                        metadata={
                            "title": "Commentary",
                            "language": "fra",
                            "artist": "x",
                        },
                    )
                ]
            )
        ],
    )
    args = build_ffmpeg_args(e, "out.mp4")
    assert args[args.index("-map") :] == [
        "-map",
        "0:a:0",
        "-c:0",
        "copy",
        "-metadata:s:0",
        "artist=x",
        "-metadata:s:0",
        "language=fra",
        "-metadata:s:0",
        "title=Commentary",
        "out.mp4",
    ]


def test_build_ffmpeg_args_metadata_values_are_passed_raw() -> None:
    """Metadata goes through argv, not the filtergraph parser -- no escaping."""
    e = Emitted(
        inputs=["a.mp4"],
        filter_complex="",
        groups=[
            OutputGroup(
                maps=[
                    OutputMap(
                        target="0:a:0",
                        type="audio",
                        copy=True,
                        metadata={"title": "12:30, take 'one'"},
                    )
                ]
            )
        ],
    )
    assert "title=12:30, take 'one'" in build_ffmpeg_args(e, "out.mp4")


def test_disposition_renders_as_its_own_flag_after_the_metadata() -> None:
    """The disposition is a FIELD of its own, not a metadata key: it renders as
    ffmpeg's own `+`-joined spec, after the `-metadata:s:<i> k=v` block."""
    e = Emitted(
        inputs=["a.mp4"],
        filter_complex="",
        groups=[
            OutputGroup(
                maps=[
                    OutputMap(
                        target="0:a:0",
                        type="audio",
                        copy=True,
                        metadata={"language": "eng"},
                        disposition=("default", "forced"),
                    )
                ]
            )
        ],
    )
    args = build_ffmpeg_args(e, "out.mp4")
    assert args[args.index("-map") :] == [
        "-map",
        "0:a:0",
        "-c:0",
        "copy",
        "-metadata:s:0",
        "language=eng",
        "-disposition:0",
        "default+forced",
        "out.mp4",
    ]
    assert "-metadata:s:0" not in args[args.index("-disposition:0") :]


def test_no_flags_renders_the_clearing_spec() -> None:
    e = Emitted(
        inputs=["a.mp4"],
        filter_complex="",
        groups=[
            OutputGroup(
                maps=[
                    OutputMap(
                        target="0:a:0", type="audio", copy=True, metadata={}, disposition=()
                    )
                ]
            )
        ],
    )
    args = build_ffmpeg_args(e, "out.mp4")
    assert "-metadata:s:0" not in args
    assert args[args.index("-disposition:0") + 1] == "0"


def test_an_unflagged_output_renders_no_disposition_at_all() -> None:
    e = Emitted(
        inputs=["a.mp4"],
        filter_complex="",
        groups=[
            OutputGroup(
                maps=[OutputMap(target="0:a:0", type="audio", copy=True, metadata={})]
            )
        ],
    )
    assert "-disposition:0" not in build_ffmpeg_args(e, "out.mp4")


# ---------------------------------------------------------------------------
# sink -- Graphs are hand-built with sink=_sink(...) here,
# i.e. one `SinkUnit` / one `Emitted.groups` entry.
# ---------------------------------------------------------------------------


def test_emitted_group_path_defaults_to_none() -> None:
    g = _graph([], [_out("src:a:v:0")])
    e = emit(g)
    assert len(e.groups) == 1
    assert e.groups[0].path is None
    assert e.groups[0].options == {}


def test_emit_copies_sink_options() -> None:
    """`emit` copies each unit's options, not aliases them."""
    sink = _sink(path="out.mkv", options={"video_codec": "libx264"})
    g = _graph([], [_out("src:a:v:0")], sink=sink)
    e = emit(g)
    assert e.groups[0].path == "out.mkv"
    assert e.groups[0].options == sink.options
    assert e.groups[0].options is not sink.options
    e.groups[0].options["video_codec"] = "mangled"
    assert g.sinks[0].options["video_codec"] == "libx264"


def test_build_ffmpeg_args_without_out_path_or_sink_raises() -> None:
    g = _graph([], [_out("src:a:v:0")])
    e = emit(g)
    with pytest.raises(ValueError, match="no output path"):
        build_ffmpeg_args(e)


def test_build_ffmpeg_args_uses_sink_path_when_no_out_path() -> None:
    sink = _sink(path="sink.mkv", options={})
    g = _graph([], [_out("src:a:v:0")], sink=sink)
    e = emit(g)
    assert build_ffmpeg_args(e)[-1] == "sink.mkv"


def test_build_ffmpeg_args_out_path_overrides_sink_path() -> None:
    sink = _sink(path="sink.mkv", options={})
    g = _graph([], [_out("src:a:v:0")], sink=sink)
    e = emit(g)
    assert build_ffmpeg_args(e, "override.mp4")[-1] == "override.mp4"


def test_build_ffmpeg_args_no_sink_graph_unchanged_byte_for_byte() -> None:
    """Regression pin: a sinkless graph's args, byte for byte.

    `a`/`b` dedup onto one `-i`: see test_readme_example_shape.
    """
    g = _graph(
        [
            _node("n1", "crop", {"w": 600, "h": 200, "x": 1200, "y": 50}, ["src:b:v:0"]),
            _node("n2", "scale", {"w": "iw*0.5", "h": "-2"}, ["n1"]),
            _node("n3", "overlay", {"x": 20, "y": 20}, ["src:a:v:0", "n2"]),
        ],
        [_out("n3")],
        input_paths=["game.mp4", "game.mp4"],
        sources={"a": 0, "b": 1},
    )
    e = emit(g)
    assert e.groups[0].path is None
    assert build_ffmpeg_args(e, "out.mp4") == [
        "ffmpeg",
        "-i",
        "game.mp4",
        "-filter_complex",
        "[0:v:0]crop=w=600:h=200:x=1200:y=50,scale=w=iw*0.5:h=-2[n2];"
        "[0:v:0][n2]overlay=x=20:y=20[out0]",
        "-map",
        "[out0]",
        "out.mp4",
    ]


def test_sink_video_codec_and_crf_render_per_video_output() -> None:
    sink = _sink(path="out.mp4", options={"video_codec": "libx264", "crf": 20})
    g = _graph(
        [
            _node("n1", "hflip", {}, ["src:a:v:0"]),
            _node("n2", "vflip", {}, ["src:b:v:0"]),
        ],
        [_out("n1"), _out("n2")],
        input_paths=["a.mp4", "b.mp4"],
        sources={"a": 0, "b": 1},
        sink=sink,
    )
    e = emit(g)
    args = build_ffmpeg_args(e)
    assert args[args.index("-map") :] == [
        "-map",
        "[out0]",
        "-map",
        "[out1]",
        "-c:0",
        "libx264",
        "-c:1",
        "libx264",
        "-crf:0",
        "20",
        "-crf:1",
        "20",
        "out.mp4",
    ]


def test_sink_video_codec_and_frames_render_per_video_output() -> None:
    """Recipe 34's shape: -c:0 png -frames:0 1, in WITH-clause order."""
    sink = _sink(path="poster.png", options={"video_codec": "png", "frames": 1})
    g = _graph(
        [_node("n1", "hflip", {}, ["src:a:v:0"])],
        [_out("n1")],
        sink=sink,
    )
    e = emit(g)
    args = build_ffmpeg_args(e)
    assert args[args.index("-map") :] == [
        "-map",
        "[out0]",
        "-c:0",
        "png",
        "-frames:0",
        "1",
        "poster.png",
    ]


def test_sink_audio_bitrate_renders_per_audio_output() -> None:
    sink = _sink(path="out.mp4", options={"audio_bitrate": "192k"})
    g = _graph(
        [
            _node("n1", "volume", {"volume": 1}, ["src:a:a:0"], ["audio"]),
            _node("n2", "volume", {"volume": 2}, ["src:a:a:1"], ["audio"]),
        ],
        [_out("n1", "audio"), _out("n2", "audio")],
        sink=sink,
    )
    e = emit(g)
    args = build_ffmpeg_args(e)
    assert args[args.index("-map") :] == [
        "-map",
        "[out0]",
        "-map",
        "[out1]",
        "-b:0",
        "192k",
        "-b:1",
        "192k",
        "out.mp4",
    ]


def test_sink_format_and_faststart_render_once_at_container_level() -> None:
    sink = _sink(path="out.mp4", options={"format": "mp4", "faststart": True})
    g = _graph([], [_out("src:a:v:0")], sink=sink)
    e = emit(g)
    args = build_ffmpeg_args(e)
    assert args[args.index("-map") :] == [
        "-map",
        "0:v:0",
        "-c:0",
        "copy",
        "-f",
        "mp4",
        "-movflags",
        "+faststart",
        "out.mp4",
    ]


def test_sink_faststart_false_omits_movflags_entirely() -> None:
    sink = _sink(path="out.mp4", options={"faststart": False})
    g = _graph([], [_out("src:a:v:0")], sink=sink)
    e = emit(g)
    args = build_ffmpeg_args(e)
    assert "-movflags" not in args
    assert "+faststart" not in args


def test_sink_shortest_renders_as_a_bare_flag() -> None:
    sink = _sink(path="out.mp4", options={"shortest": True})
    g = _graph([], [_out("src:a:v:0")], sink=sink)
    args = build_ffmpeg_args(emit(g))
    assert args[args.index("-map") :] == ["-map", "0:v:0", "-c:0", "copy", "-shortest", "out.mp4"]


def test_sink_shortest_false_emits_nothing() -> None:
    sink = _sink(path="out.mp4", options={"shortest": False})
    g = _graph([], [_out("src:a:v:0")], sink=sink)
    args = build_ffmpeg_args(emit(g))
    assert "-shortest" not in args


def test_sink_duration_renders_a_fractional_number() -> None:
    sink = _sink(path="out.mp4", options={"duration": 30.5})
    g = _graph([], [_out("src:a:v:0")], sink=sink)
    args = build_ffmpeg_args(emit(g))
    assert args[args.index("-t") :] == ["-t", "30.5", "out.mp4"]


def test_sink_movflags_renders_its_raw_value() -> None:
    sink = _sink(path="out.mp4", options={"movflags": "+faststart+frag_keyframe"})
    g = _graph([], [_out("src:a:v:0")], sink=sink)
    args = build_ffmpeg_args(emit(g))
    assert args[args.index("-movflags") :] == [
        "-movflags",
        "+faststart+frag_keyframe",
        "out.mp4",
    ]


def test_container_tags_render_as_global_metadata_flags_keys_sorted() -> None:
    sink = _sink(path="out.mp4", tags={"title": "Director's Cut", "comment": "ripped"})
    g = _graph([], [_out("src:a:v:0")], sink=sink)
    args = build_ffmpeg_args(emit(g))
    assert args[args.index("-map") :] == [
        "-map",
        "0:v:0",
        "-c:0",
        "copy",
        "-metadata",
        "comment=ripped",
        "-metadata",
        "title=Director's Cut",
        "out.mp4",
    ]


def test_container_tag_with_a_null_value_renders_an_empty_assignment() -> None:
    """A cleared key is written out: ffmpeg copies input globals by default."""
    sink = _sink(path="out.mp4", tags={"artist": None})
    g = _graph([], [_out("src:a:v:0")], sink=sink)
    args = build_ffmpeg_args(emit(g))
    assert args[args.index("-metadata") :] == ["-metadata", "artist=", "out.mp4"]


def test_container_tags_render_after_the_metadata_source() -> None:
    """ffmpeg applies -metadata after -map_metadata, so a named key wins."""
    sink = _sink(path="out.mp4", metadata=0, tags={"title": "Cut"})
    g = _graph([], [_out("src:a:v:0")], sink=sink)
    args = build_ffmpeg_args(emit(g))
    assert args[args.index("-map_metadata") :] == [
        "-map_metadata",
        "0",
        "-metadata",
        "title=Cut",
        "out.mp4",
    ]


def test_a_copied_metadata_source_renders_its_input_index() -> None:
    sink = _sink(path="out.mp4", metadata=0)
    g = _graph([], [_out("src:a:v:0")], sink=sink)
    args = build_ffmpeg_args(emit(g))
    assert args[args.index("-map_metadata") :] == ["-map_metadata", "0", "out.mp4"]


def test_no_metadata_source_renders_map_metadata_negative_one() -> None:
    sink = _sink(path="out.mp4", metadata=NO_METADATA)
    g = _graph([], [_out("src:a:v:0")], sink=sink)
    args = build_ffmpeg_args(emit(g))
    assert args[args.index("-map_metadata") :] == ["-map_metadata", "-1", "out.mp4"]


def test_an_unset_metadata_source_emits_nothing() -> None:
    sink = _sink(path="out.mp4")
    g = _graph([], [_out("src:a:v:0")], sink=sink)
    args = build_ffmpeg_args(emit(g))
    assert "-map_metadata" not in args


def test_a_chapter_list_renders_map_chapters_with_its_input_index() -> None:
    sink = _sink(path="out.mp4", chapters=1)
    g = _graph([], [_out("src:a:v:0")], sink=sink)
    args = build_ffmpeg_args(emit(g))
    assert args[args.index("-map_chapters") :] == ["-map_chapters", "1", "out.mp4"]


def test_a_cleared_chapter_list_renders_map_chapters_negative_one() -> None:
    sink = _sink(path="out.mp4", chapters=NO_CHAPTERS)
    g = _graph([], [_out("src:a:v:0")], sink=sink)
    args = build_ffmpeg_args(emit(g))
    assert args[args.index("-map_chapters") :] == ["-map_chapters", "-1", "out.mp4"]


def test_no_chapter_list_renders_no_map_chapters_at_all() -> None:
    g = _graph([], [_out("src:a:v:0")], sink=_sink(path="out.mp4"))
    assert "-map_chapters" not in build_ffmpeg_args(emit(g))


def test_an_attachment_renders_attach_ahead_of_the_maps_and_tags_after() -> None:
    """ffmpeg numbers an attachment after every mapped stream of that file."""
    sink = _sink(
        path="out.mkv",
        attachments=[Attachment("f.ttf", "font.ttf", "application/x-truetype-font")],
    )
    g = _graph([], [_out("src:a:v:0"), _out("src:a:a:0", "audio")], sink=sink)
    args = build_ffmpeg_args(emit(g))
    assert args == [
        "ffmpeg", "-i", "a.mp4",
        "-attach", "f.ttf",
        "-map", "0:v:0", "-c:0", "copy",
        "-map", "0:a:0", "-c:1", "copy",
        "-metadata:s:2", "mimetype=application/x-truetype-font",
        "-metadata:s:2", "filename=font.ttf",
        "out.mkv",
    ]


def test_two_attachments_take_consecutive_output_stream_indices() -> None:
    sink = _sink(
        path="out.mkv",
        attachments=[
            Attachment("f.ttf", "font.ttf", "application/x-truetype-font"),
            Attachment("n.txt", "notes.txt", "text/plain"),
        ],
    )
    g = _graph([], [_out("src:a:v:0")], sink=sink)
    args = build_ffmpeg_args(emit(g))
    assert args[args.index("-attach") : args.index("-map")] == [
        "-attach", "f.ttf", "-attach", "n.txt",
    ]
    assert args[args.index("-metadata:s:1") :] == [
        "-metadata:s:1", "mimetype=application/x-truetype-font",
        "-metadata:s:1", "filename=font.ttf",
        "-metadata:s:2", "mimetype=text/plain",
        "-metadata:s:2", "filename=notes.txt",
        "out.mkv",
    ]


def test_an_attachment_with_no_tags_renders_attach_alone() -> None:
    """ffmpeg then names it after the file and guesses its type."""
    sink = _sink(path="out.mkv", attachments=[Attachment("f.ttf")])
    g = _graph([], [_out("src:a:v:0")], sink=sink)
    args = build_ffmpeg_args(emit(g))
    assert args == [
        "ffmpeg", "-i", "a.mp4", "-attach", "f.ttf",
        "-map", "0:v:0", "-c:0", "copy", "out.mkv",
    ]


def test_each_output_file_carries_its_own_attachments() -> None:
    """`-attach` applies to the output file that follows it."""
    g = _graph(
        [],
        [],
        sinks=[
            SinkUnit(
                outputs=[_out("src:a:v:0")],
                path="one.mkv",
                attachments=[Attachment("f.ttf", "font.ttf", "font/ttf")],
            ),
            SinkUnit(outputs=[_out("src:a:a:0", "audio")], path="two.mkv"),
        ],
    )
    args = build_ffmpeg_args(emit(g))
    assert args.index("-attach") < args.index("one.mkv")
    assert args.count("-attach") == 1


def test_sink_codec_params_derives_its_flag_from_video_codec() -> None:
    sink = _sink(
        path="out.mp4", options={"video_codec": "libx264", "codec_params": "keyint=48"}
    )
    g = _graph([_node("n1", "hflip", {}, ["src:a:v:0"])], [_out("n1")], sink=sink)
    args = build_ffmpeg_args(emit(g))
    assert args[args.index("-c:0") :] == [
        "-c:0",
        "libx264",
        "-x264-params:0",
        "keyint=48",
        "out.mp4",
    ]


@pytest.mark.parametrize(
    "codec, flag",
    [("libx264", "-x264-params"), ("libx265", "-x265-params"), ("libsvtav1", "-svtav1-params")],
)
def test_sink_codec_params_flag_per_codec(codec: str, flag: str) -> None:
    sink = _sink(path="out.mp4", options={"video_codec": codec, "codec_params": "k=1"})
    g = _graph([_node("n1", "hflip", {}, ["src:a:v:0"])], [_out("n1")], sink=sink)
    args = build_ffmpeg_args(emit(g))
    assert f"{flag}:0" in args


def test_sink_codec_params_with_no_matching_video_codec_reaching_emit_is_internal() -> None:
    """Defensive: lower already rejects this shape; a hand-built graph that
    skips lower's check surfaces it as an internal-error backstop instead."""
    sink = _sink(path="out.mp4", options={"codec_params": "keyint=48"})
    g = _graph([_node("n1", "hflip", {}, ["src:a:v:0"])], [_out("n1")], sink=sink)
    with pytest.raises(FfrwdError) as excinfo:
        build_ffmpeg_args(emit(g))
    assert excinfo.value.code == ErrorCode.INTERNAL


def test_sink_options_render_in_insertion_order_not_table_order() -> None:
    """SINK_OPTIONS lists video_codec before crf; Sink.options insertion order wins."""
    sink = _sink(path="out.mp4", options={"crf": 20, "video_codec": "libx264"})
    g = _graph([_node("n1", "hflip", {}, ["src:a:v:0"])], [_out("n1")], sink=sink)
    e = emit(g)
    args = build_ffmpeg_args(e)
    assert args[args.index("-map") :] == [
        "-map",
        "[out0]",
        "-crf:0",
        "20",
        "-c:0",
        "libx264",
        "out.mp4",
    ]


def test_sink_audio_codec_suppresses_passthrough_copy_for_audio_only() -> None:
    """passthrough audio + audio_codec set -> no -c:<i> copy, codec rendered instead."""
    sink = _sink(path="out.mp4", options={"audio_codec": "aac"})
    g = _graph(
        [_node("n1", "hflip", {}, ["src:a:v:0"])],
        [_out("n1"), _out("src:a:a:0", "audio")],
        sink=sink,
    )
    e = emit(g)
    args = build_ffmpeg_args(e)
    assert args[args.index("-map") :] == [
        "-map",
        "[out0]",
        "-map",
        "0:a:0",
        "-c:1",
        "aac",
        "out.mp4",
    ]


def test_sink_video_codec_suppresses_only_video_passthrough_copy() -> None:
    """A video-scoped codec option leaves an untouched audio passthrough's copy alone."""
    sink = _sink(path="out.mp4", options={"video_codec": "libx264"})
    g = _graph([], [_out("src:a:v:0"), _out("src:a:a:0", "audio")], sink=sink)
    e = emit(g)
    args = build_ffmpeg_args(e)
    assert args[args.index("-map") :] == [
        "-map",
        "0:v:0",
        "-map",
        "0:a:0",
        "-c:1",
        "copy",
        "-c:0",
        "libx264",
        "out.mp4",
    ]


# ---------------------------------------------------------------------------
# subtitle / data streams -- bare -map only, repeatable
# ---------------------------------------------------------------------------


def test_subtitle_and_data_src_refs_render_as_s_and_d_stream_specs() -> None:
    g = _graph(
        [],
        [_out("src:a:s:0", "subtitle"), _out("src:b:d:2", "data")],
        input_paths=["a.mkv", "b.mkv"],
        sources={"a": 0, "b": 1},
    )
    e = emit(g)
    assert e.filter_complex == ""
    assert e.maps == [
        OutputMap(target="0:s:0", type="subtitle", copy=True, metadata={}),
        OutputMap(target="1:d:2", type="data", copy=True, metadata={}),
    ]


def test_the_same_subtitle_ref_may_be_mapped_twice() -> None:
    """Consume-once is a filtergraph pad rule; a bare -map may repeat."""
    g = _graph(
        [],
        [_out("src:a:s:0", "subtitle"), _out("src:a:s:0", "subtitle")],
        input_paths=["a.mkv"],
    )
    args = build_ffmpeg_args(emit(g), "out.mkv")
    assert args[args.index("-map") :] == [
        "-map",
        "0:s:0",
        "-c:0",
        "copy",
        "-map",
        "0:s:0",
        "-c:1",
        "copy",
        "out.mkv",
    ]


def test_the_same_data_ref_may_be_mapped_twice() -> None:
    g = _graph(
        [], [_out("src:a:d:0", "data"), _out("src:a:d:0", "data")], input_paths=["a.mkv"]
    )
    assert [m.target for m in emit(g).maps] == ["0:d:0", "0:d:0"]


def test_a_duplicated_video_pad_is_still_a_consume_once_error() -> None:
    """The exemption is for subtitle/data ONLY -- video still needs the split pass."""
    g = _graph([], [_out("src:a:v:0"), _out("src:a:v:0")])
    with pytest.raises(FfrwdError) as excinfo:
        emit(g)
    assert excinfo.value.code is ErrorCode.INTERNAL
    assert "consume-once" in excinfo.value.message


def test_subtitle_output_keeps_its_language_metadata() -> None:
    g = _graph(
        [],
        [_out("src:a:s:0", "subtitle", metadata={"language": "eng"})],
        input_paths=["a.mkv"],
    )
    args = build_ffmpeg_args(emit(g), "out.mkv")
    assert "-metadata:s:0" in args
    assert args[args.index("-metadata:s:0") + 1] == "language=eng"


def test_sink_subtitle_codec_renders_per_subtitle_output() -> None:
    sink = _sink(path="out.mp4", options={"subtitle_codec": "mov_text"})
    g = _graph(
        [],
        [_out("src:a:v:0"), _out("src:a:a:0", "audio"), _out("src:a:s:0", "subtitle")],
        input_paths=["a.mkv"],
        sink=sink,
    )
    args = build_ffmpeg_args(emit(g))
    assert args[args.index("-map") :] == [
        "-map",
        "0:v:0",
        "-c:0",
        "copy",
        "-map",
        "0:a:0",
        "-c:1",
        "copy",
        "-map",
        "0:s:0",
        # copy suppressed on the SUBTITLE output only: the option table's
        # scope drives it, so subtitle_codec generalizes for free.
        "-c:2",
        "mov_text",
        "out.mp4",
    ]


def test_sink_subtitle_codec_leaves_video_and_audio_copies_alone() -> None:
    sink = _sink(path="out.mkv", options={"subtitle_codec": "srt"})
    g = _graph(
        [],
        [_out("src:a:a:0", "audio"), _out("src:a:s:0", "subtitle")],
        input_paths=["a.mkv"],
        sink=sink,
    )
    args = build_ffmpeg_args(emit(g))
    assert "-c:0" in args and args[args.index("-c:0") + 1] == "copy"
    assert "-c:1" in args and args[args.index("-c:1") + 1] == "srt"


def test_sink_video_codec_does_not_suppress_a_subtitle_copy() -> None:
    sink = _sink(path="out.mkv", options={"video_codec": "libx264"})
    g = _graph(
        [],
        [_out("src:a:v:0"), _out("src:a:s:0", "subtitle")],
        input_paths=["a.mkv"],
        sink=sink,
    )
    args = build_ffmpeg_args(emit(g))
    assert args[args.index("-map") :] == [
        "-map",
        "0:v:0",
        "-map",
        "0:s:0",
        "-c:1",
        "copy",
        "-c:0",
        "libx264",
        "out.mkv",
    ]


# ---------------------------------------------------------------------------
# output groups: one ffmpeg command, one file per sink
# ---------------------------------------------------------------------------


def _ladder_graph() -> Graph:
    """Two scaled video renditions off one split, plus an audio-only file."""
    return _graph(
        [
            _node("n1", "split", {"n": 2}, ["src:a:v:0"], ["video", "video"]),
            _node("n2", "scale", {"w": 1280, "h": -2}, ["n1:0"]),
            _node("n3", "scale", {"w": 640, "h": -2}, ["n1:1"]),
        ],
        [],
        sinks=[
            SinkUnit(
                outputs=[_out("n2"), _out("src:a:a:0", "audio")],
                path="720.mp4",
                options={"video_codec": "libx264", "crf": 21},
            ),
            SinkUnit(
                outputs=[_out("n3"), _out("src:a:a:0", "audio")],
                path="360.mp4",
                options={"crf": 26},
            ),
        ],
    )


def test_groups_mirror_the_graphs_sinks() -> None:
    e = emit(_ladder_graph())
    assert [group.path for group in e.groups] == ["720.mp4", "360.mp4"]
    assert [len(group.maps) for group in e.groups] == [2, 2]
    assert [group.options for group in e.groups] == [
        {"video_codec": "libx264", "crf": 21},
        {"crf": 26},
    ]


def test_output_labels_stay_unique_across_the_whole_command() -> None:
    """Labels are graph-scoped: out<i> is indexed over the OUTPUT UNION."""
    e = emit(_ladder_graph())
    assert [m.target for group in e.groups for m in group.maps] == [
        "[out0]",
        "0:a:0",
        "[out2]",
        "0:a:0",
    ]
    assert "[out2]" in e.filter_complex


def test_maps_is_the_concatenation_of_every_group() -> None:
    e = emit(_ladder_graph())
    assert e.maps == e.groups[0].maps + e.groups[1].maps


def test_stream_indices_restart_in_every_output_file() -> None:
    """ffmpeg numbers output streams per FILE, so -c:<i> restarts at 0."""
    args = build_ffmpeg_args(emit(_ladder_graph()))
    assert args[args.index("-map") :] == [
        "-map",
        "[out0]",
        "-map",
        "0:a:0",
        "-c:1",
        "copy",
        "-c:0",
        "libx264",
        "-crf:0",
        "21",
        "720.mp4",
        # group 2 starts over at 0 -- its audio passthrough is -c:1 again,
        # not -c:3, and its own crf applies to ITS video index.
        "-map",
        "[out2]",
        "-map",
        "0:a:0",
        "-c:1",
        "copy",
        "-crf:0",
        "26",
        "360.mp4",
    ]


def test_metadata_indices_are_per_group_too() -> None:
    g = _graph(
        [_node("n1", "hflip", {}, ["src:a:v:0"])],
        [],
        sinks=[
            SinkUnit(outputs=[_out("n1")], path="one.mp4"),
            SinkUnit(
                outputs=[
                    _out("src:a:v:1"),
                    _out("src:a:a:0", "audio", metadata={"language": "fra"}),
                ],
                path="two.mkv",
            ),
        ],
    )
    args = build_ffmpeg_args(emit(g))
    # The tag rides group 2's SECOND stream, which is index 1 of that file.
    assert "-metadata:s:1" in args
    assert args[args.index("-metadata:s:1") + 1] == "language=fra"
    assert "-metadata:s:2" not in args


def test_copy_suppression_is_per_group() -> None:
    """One file may re-encode the stream the other copies."""
    g = _graph(
        [],
        [],
        sinks=[
            SinkUnit(outputs=[_out("src:a:a:0", "audio")], path="copy.m4a"),
            SinkUnit(
                outputs=[_out("src:a:a:0", "audio")],
                path="encoded.m4a",
                options={"audio_codec": "aac"},
            ),
        ],
    )
    args = build_ffmpeg_args(emit(g))
    assert args == [
        "ffmpeg",
        "-i",
        "a.mp4",
        "-map",
        "0:a:0",
        "-c:0",
        "copy",
        "copy.m4a",
        "-map",
        "0:a:0",
        "-c:0",
        "aac",
        "encoded.m4a",
    ]


def test_two_groups_may_bare_map_the_same_source_stream() -> None:
    """The cross-group exemption: a repeated -map is not a fan-out bug."""
    g = _graph(
        [],
        [],
        sinks=[
            SinkUnit(outputs=[_out("src:a:a:0", "audio")], path="one.m4a"),
            SinkUnit(outputs=[_out("src:a:a:0", "audio")], path="two.m4a"),
        ],
    )
    e = emit(g)  # no INTERNAL: the pad is a bare -map, not a filtergraph pad
    assert [group.maps[0].target for group in e.groups] == ["0:a:0", "0:a:0"]


def test_one_group_mapping_a_source_stream_twice_is_still_a_fanout_bug() -> None:
    g = _graph(
        [],
        [_out("src:a:a:0", "audio"), _out("src:a:a:0", "audio")],
        sink=_sink(path="one.m4a"),
    )
    err = _assert_internal(g)
    assert "consume-once" in err.message


def test_one_node_pad_read_by_two_groups_is_still_a_fanout_bug() -> None:
    g = _graph(
        [_node("n1", "hflip", {}, ["src:a:v:0"])],
        [],
        sinks=[
            SinkUnit(outputs=[_out("n1")], path="one.mp4"),
            SinkUnit(outputs=[_out("n1")], path="two.mp4"),
        ],
    )
    err = _assert_internal(g)
    assert "consume-once" in err.message


def test_out_path_may_not_override_a_multi_group_command() -> None:
    e = emit(_ladder_graph())
    with pytest.raises(ValueError, match="writes 2"):
        build_ffmpeg_args(e, "override.mp4")


def test_a_group_without_a_path_and_without_out_path_raises() -> None:
    g = _graph(
        [],
        [],
        sinks=[
            SinkUnit(outputs=[_out("src:a:v:0")], path="one.mp4"),
            SinkUnit(outputs=[_out("src:a:a:0", "audio")]),
        ],
    )
    with pytest.raises(ValueError, match="no output path"):
        build_ffmpeg_args(emit(g))


# ---------------------------------------------------------------------------
# input seek: -ss/-to in front of the owning -i
# ---------------------------------------------------------------------------


def test_emitted_input_trims_default_to_all_none() -> None:
    """Always parallel to `inputs`, so a consumer can index it directly."""
    g = _graph([], [_out("src:a:v:0")], input_paths=["a.mp4", "b.mp4"], sources={"a": 0, "b": 1})
    assert emit(g).input_trims == [None, None]


def test_alias_keyed_windows_resolve_to_input_positions() -> None:
    """`Graph.input_trims` is alias-keyed; `Emitted.input_trims` is -i-ordered."""
    g = _graph(
        [],
        [_out("src:b:v:0")],
        input_paths=["a.mp4", "b.mp4", "c.mp4"],
        sources={"a": 0, "b": 1, "c": 2},
        input_trims={"c": (7, 8), "a": (1, 2)},
    )
    assert emit(g).input_trims == [(1, 2), None, (7, 8)]


def test_build_ffmpeg_args_puts_ss_and_to_before_the_owning_input() -> None:
    g = _graph(
        [],
        [_out("src:a:v:0"), _out("src:b:a:0", "audio")],
        input_paths=["a.mp4", "b.mp4"],
        sources={"a": 0, "b": 1},
        input_trims={"b": (2, 4)},
    )
    assert build_ffmpeg_args(emit(g), "out.mp4") == [
        "ffmpeg",
        "-i",
        "a.mp4",
        "-ss",
        "2",
        "-to",
        "4",
        "-i",
        "b.mp4",
        "-map",
        "0:v:0",
        "-c:0",
        "copy",
        "-map",
        "1:a:0",
        "-c:1",
        "copy",
        "out.mp4",
    ]


def test_seek_times_render_by_the_scalar_rules() -> None:
    """Same rendering a filter argument gets: 12.5 -> "12.5", 5 -> "5"."""
    g = _graph([], [_out("src:a:v:0")], input_trims={"a": (12.5, 60)})
    args = build_ffmpeg_args(emit(g), "out.mp4")
    assert args[:6] == ["ffmpeg", "-ss", "12.5", "-to", "60", "-i"]


def test_a_seeked_input_still_stream_copies() -> None:
    """The new capability: a window no longer forces a filter node, so the
    output stays a passthrough (`-c:<i> copy`) with no filtergraph at all."""
    g = _graph([], [_out("src:a:v:0")], input_trims={"a": (5, 60)})
    e = emit(g)
    assert e.filter_complex == ""
    assert e.maps[0].copy is True
    assert "-filter_complex" not in build_ffmpeg_args(e, "out.mp4")


def test_a_hand_built_emitted_needs_no_input_trims() -> None:
    """The empty default means "no input is seeked", so an Emitted built
    without windows renders with no seek flags at all."""
    e = Emitted(
        inputs=["a.mp4"],
        filter_complex="",
        groups=[
            OutputGroup(
                maps=[OutputMap(target="0:v:0", type="video", copy=True, metadata={})]
            )
        ],
    )
    assert build_ffmpeg_args(e, "out.mp4")[:3] == ["ffmpeg", "-i", "a.mp4"]


# ---------------------------------------------------------------------------
# open-ended windows: either half of a window may be None
# ---------------------------------------------------------------------------


def test_tail_only_window_renders_ss_with_no_to() -> None:
    g = _graph([], [_out("src:a:v:0")], input_trims={"a": (5, None)})
    args = build_ffmpeg_args(emit(g), "out.mp4")
    assert args == [
        "ffmpeg",
        "-ss",
        "5",
        "-i",
        "a.mp4",
        "-map",
        "0:v:0",
        "-c:0",
        "copy",
        "out.mp4",
    ]
    assert "-to" not in args


def test_head_only_window_renders_to_with_no_ss() -> None:
    g = _graph([], [_out("src:a:v:0")], input_trims={"a": (None, 60)})
    args = build_ffmpeg_args(emit(g), "out.mp4")
    assert args == [
        "ffmpeg",
        "-to",
        "60",
        "-i",
        "a.mp4",
        "-map",
        "0:v:0",
        "-c:0",
        "copy",
        "out.mp4",
    ]
    assert "-ss" not in args


def test_open_window_still_resolves_alias_to_input_position() -> None:
    g = _graph(
        [],
        [_out("src:b:v:0")],
        input_paths=["a.mp4", "b.mp4"],
        sources={"a": 0, "b": 1},
        input_trims={"b": (None, 8)},
    )
    assert emit(g).input_trims == [None, (None, 8)]


def test_open_window_still_stream_copies() -> None:
    """Same as a closed window: no bound forces a filter node into the graph."""
    g = _graph([], [_out("src:a:v:0")], input_trims={"a": (5, None)})
    e = emit(g)
    assert e.filter_complex == ""
    assert e.maps[0].copy is True


def test_input_trim_on_an_unknown_alias_is_internal_error() -> None:
    g = _graph([], [_out("src:a:v:0")], input_trims={"nope": (1, 2)})
    err = _assert_internal(g)
    assert "unknown source alias" in err.message


def test_two_aliases_disagreeing_on_one_inputs_window_is_internal_error() -> None:
    """Aliases each own an -i, so this cannot come from lowering -- but a
    hand-built graph that says otherwise must not silently drop a window."""
    g = _graph(
        [],
        [_out("src:a:v:0")],
        sources={"a": 0, "b": 0},
        input_trims={"a": (1, 2), "b": (3, 4)},
    )
    err = _assert_internal(g)
    assert "two different trim windows" in err.message


# ---------------------------------------------------------------------------
# output windows: -ss/-to on the output file, ahead of its maps
# ---------------------------------------------------------------------------


def test_a_sink_window_renders_before_that_outputs_maps() -> None:
    g = _graph([], [_out("src:a:v:0")], sink=_sink("out.mkv", window=(1.0, 2.5)))
    args = build_ffmpeg_args(emit(g))
    assert args == [
        "ffmpeg", "-i", "a.mp4",
        "-ss", "1.0", "-to", "2.5", "-map", "0:v:0", "out.mkv",
    ]


def test_a_sink_window_drops_the_forced_copy() -> None:
    """An output seek re-encodes, and `-c copy` under one writes a corrupt
    file, so the passthrough map takes the default encoder instead."""
    g = _graph([], [_out("src:a:v:0")], sink=_sink("out.mkv", window=(None, 2.0)))
    args = build_ffmpeg_args(emit(g))
    assert "copy" not in args
    assert args[3:6] == ["-to", "2.0", "-map"]


def test_an_open_sink_window_renders_only_the_bound_it_has() -> None:
    g = _graph([], [_out("src:a:v:0")], sink=_sink("out.mkv", window=(3, None)))
    args = build_ffmpeg_args(emit(g))
    assert "-to" not in args
    assert args[3:5] == ["-ss", "3"]


def test_each_output_file_carries_its_own_window() -> None:
    units = [
        SinkUnit(outputs=[_out("src:a:v:0")], path="one.mkv", window=(0.0, 1.0)),
        SinkUnit(
            outputs=[_out("src:a:a:0", "audio")], path="two.mkv", window=(1.0, 2.0)
        ),
    ]
    g = _graph([], [], sinks=units)
    args = build_ffmpeg_args(emit(g))
    assert args == [
        "ffmpeg", "-i", "a.mp4",
        "-ss", "0.0", "-to", "1.0", "-map", "0:v:0", "one.mkv",
        "-ss", "1.0", "-to", "2.0", "-map", "0:a:0", "two.mkv",
    ]


def test_a_windowed_output_still_takes_the_codec_its_options_name() -> None:
    g = _graph(
        [],
        [_out("src:a:v:0")],
        sink=_sink("out.mkv", {"video_codec": "libx264"}, window=(0.0, 1.0)),
    )
    args = build_ffmpeg_args(emit(g))
    assert args[-3:] == ["-c:0", "libx264", "out.mkv"]
    assert "copy" not in args


# ---------------------------------------------------------------------------
# input options: rendered before -ss/-to, before -i
# ---------------------------------------------------------------------------


def test_emitted_input_options_default_to_all_empty() -> None:
    """Always parallel to `inputs`, so a consumer can index it directly."""
    g = _graph([], [_out("src:a:v:0")], input_paths=["a.mp4", "b.mp4"], sources={"a": 0, "b": 1})
    assert emit(g).input_options == [{}, {}]


def test_alias_keyed_options_resolve_to_input_positions() -> None:
    """`Graph.input_options` is alias-keyed; `Emitted.input_options` is -i-ordered."""
    g = _graph(
        [],
        [_out("src:b:v:0")],
        input_paths=["a.mp4", "b.mp4", "c.mp4"],
        sources={"a": 0, "b": 1, "c": 2},
        input_options={"c": {"hwaccel": "cuda"}, "a": {"loop": True}},
    )
    assert emit(g).input_options == [{"loop": True}, {}, {"hwaccel": "cuda"}]


def test_build_ffmpeg_args_renders_options_before_the_owning_input() -> None:
    g = _graph(
        [],
        [_out("src:a:v:0"), _out("src:b:a:0", "audio")],
        input_paths=["a.mp4", "b.mp4"],
        sources={"a": 0, "b": 1},
        input_options={"b": {"loop": True, "framerate": 15}},
    )
    assert build_ffmpeg_args(emit(g), "out.mp4") == [
        "ffmpeg",
        "-i",
        "a.mp4",
        "-loop",
        "1",
        "-framerate",
        "15",
        "-i",
        "b.mp4",
        "-map",
        "0:v:0",
        "-c:0",
        "copy",
        "-map",
        "1:a:0",
        "-c:1",
        "copy",
        "out.mp4",
    ]


def test_input_options_render_before_seek_flags() -> None:
    """Verified order (see emit.py's docstring): options, -ss/-to, then -i."""
    g = _graph(
        [],
        [_out("src:a:v:0")],
        input_options={"a": {"loop": True}},
        input_trims={"a": (0, 2)},
    )
    args = build_ffmpeg_args(emit(g), "out.mp4")
    assert args[:7] == ["ffmpeg", "-loop", "1", "-ss", "0", "-to", "2"]


def test_loop_false_is_never_rendered() -> None:
    g = _graph([], [_out("src:a:v:0")], input_options={"a": {"loop": False}})
    args = build_ffmpeg_args(emit(g), "out.mp4")
    assert "-loop" not in args


def test_itsoffset_renders_a_negative_number() -> None:
    g = _graph([], [_out("src:a:v:0")], input_options={"a": {"itsoffset": -1.5}})
    args = build_ffmpeg_args(emit(g), "out.mp4")
    assert args[:3] == ["ffmpeg", "-itsoffset", "-1.5"]


def test_stream_loop_renders_as_a_bare_int() -> None:
    g = _graph([], [_out("src:a:v:0")], input_options={"a": {"stream_loop": -1}})
    args = build_ffmpeg_args(emit(g), "out.mp4")
    assert args[:3] == ["ffmpeg", "-stream_loop", "-1"]


def test_hwaccel_renders_as_a_bare_string() -> None:
    g = _graph([], [_out("src:a:v:0")], input_options={"a": {"hwaccel": "cuda"}})
    args = build_ffmpeg_args(emit(g), "out.mp4")
    assert args[:3] == ["ffmpeg", "-hwaccel", "cuda"]


def test_realtime_renders_as_a_bare_re_flag() -> None:
    g = _graph([], [_out("src:a:v:0")], input_options={"a": {"realtime": True}})
    args = build_ffmpeg_args(emit(g), "out.mp4")
    assert args[:3] == ["ffmpeg", "-re", "-i"]


def test_realtime_false_is_never_rendered() -> None:
    g = _graph([], [_out("src:a:v:0")], input_options={"a": {"realtime": False}})
    args = build_ffmpeg_args(emit(g), "out.mp4")
    assert "-re" not in args


def test_seek_end_renders_negated() -> None:
    g = _graph([], [_out("src:a:v:0")], input_options={"a": {"seek_end": 60}})
    args = build_ffmpeg_args(emit(g), "out.mp4")
    assert args[:3] == ["ffmpeg", "-sseof", "-60"]


def test_seek_end_renders_a_negated_fraction() -> None:
    g = _graph([], [_out("src:a:v:0")], input_options={"a": {"seek_end": 12.5}})
    args = build_ffmpeg_args(emit(g), "out.mp4")
    assert args[:3] == ["ffmpeg", "-sseof", "-12.5"]


def test_user_format_renders_the_f_flag_before_the_i() -> None:
    g = _graph([], [_out("src:a:v:0")], input_options={"a": {"format": "v4l2"}})
    args = build_ffmpeg_args(emit(g), "out.mp4")
    assert args[:3] == ["ffmpeg", "-f", "v4l2"]


def test_subtitle_decoder_renders_before_the_i() -> None:
    g = _graph([], [_out("src:a:v:0")], input_options={"a": {"subtitle_decoder": "webvtt"}})
    args = build_ffmpeg_args(emit(g), "out.mp4")
    assert args[:3] == ["ffmpeg", "-c:s", "webvtt"]


def test_sub_charenc_and_start_number_render_as_bare_strings() -> None:
    g = _graph(
        [],
        [_out("src:a:v:0")],
        input_options={"a": {"sub_charenc": "CP1250", "start_number": 3}},
    )
    args = build_ffmpeg_args(emit(g), "out.mp4")
    assert args[:5] == ["ffmpeg", "-sub_charenc", "CP1250", "-start_number", "3"]


def test_input_options_render_in_insertion_order() -> None:
    g = _graph(
        [],
        [_out("src:a:v:0")],
        input_options={"a": {"framerate": 15, "loop": True}},
    )
    args = build_ffmpeg_args(emit(g), "out.mp4")
    assert args[:5] == ["ffmpeg", "-framerate", "15", "-loop", "1"]


def test_a_hand_built_emitted_needs_no_input_options() -> None:
    """The empty default means "no input set options", same contract as
    `input_trims`'s hand-built-Emitted case."""
    e = Emitted(
        inputs=["a.mp4"],
        filter_complex="",
        groups=[
            OutputGroup(
                maps=[OutputMap(target="0:v:0", type="video", copy=True, metadata={})]
            )
        ],
    )
    assert build_ffmpeg_args(e, "out.mp4")[:3] == ["ffmpeg", "-i", "a.mp4"]


def test_input_options_on_an_unknown_alias_is_internal_error() -> None:
    g = _graph([], [_out("src:a:v:0")], input_options={"nope": {"loop": True}})
    err = _assert_internal(g)
    assert "unknown source alias" in err.message


def test_two_aliases_disagreeing_on_one_inputs_option_set_is_internal_error() -> None:
    g = _graph(
        [],
        [_out("src:a:v:0")],
        sources={"a": 0, "b": 0},
        input_options={"a": {"loop": True}, "b": {"hwaccel": "cuda"}},
    )
    err = _assert_internal(g)
    assert "two different option sets" in err.message


# ---------------------------------------------------------------------------
# input dedup: untrimmed same-path same-options aliases
# share one -i; a mismatched option set or a trim window blocks it.
# ---------------------------------------------------------------------------


def test_dedup_fires_for_untrimmed_same_path_no_options() -> None:
    """Two aliases, same path, no options, no trim -> one -i."""
    g = _graph(
        [],
        [_out("src:x:v:0"), _out("src:y:v:0")],
        input_paths=["a.mp4", "a.mp4"],
        sources={"x": 0, "y": 1},
    )
    e = emit(g)
    assert e.inputs == ["a.mp4"]
    args = build_ffmpeg_args(e, "out.mp4")
    assert args.count("-i") == 1
    assert args[:3] == ["ffmpeg", "-i", "a.mp4"]


def test_dedup_fires_for_matching_options_and_renders_them_once() -> None:
    """Same path, IDENTICAL options on both aliases -> one -i, options once."""
    g = _graph(
        [],
        [_out("src:x:v:0"), _out("src:y:v:0")],
        input_paths=["a.mp4", "a.mp4"],
        sources={"x": 0, "y": 1},
        input_options={"x": {"loop": True}, "y": {"loop": True}},
    )
    e = emit(g)
    assert e.inputs == ["a.mp4"]
    args = build_ffmpeg_args(e, "out.mp4")
    assert args.count("-i") == 1
    assert args.count("-loop") == 1
    assert args[:5] == ["ffmpeg", "-loop", "1", "-i", "a.mp4"]


def test_dedup_is_blocked_by_mismatched_options() -> None:
    """Same path, DIFFERENT (or absent-vs-present) options -> two -i's."""
    g = _graph(
        [],
        [_out("src:x:v:0"), _out("src:y:v:0")],
        input_paths=["a.mp4", "a.mp4"],
        sources={"x": 0, "y": 1},
        input_options={"x": {"loop": True}},
    )
    e = emit(g)
    assert e.inputs == ["a.mp4", "a.mp4"]
    assert build_ffmpeg_args(e, "out.mp4").count("-i") == 2


def test_dedup_is_blocked_by_a_trim_window() -> None:
    """Same path, one alias trimmed -> two -i's, even though nothing conflicts.

    Mirrors cookbook recipe 17 (the concat splice): the same file appears
    under two aliases with two different windows and must keep two `-i`'s.
    This is the minimal version -- only one side is trimmed at all.
    """
    g = _graph(
        [],
        [_out("src:x:v:0"), _out("src:y:v:0")],
        input_paths=["a.mp4", "a.mp4"],
        sources={"x": 0, "y": 1},
        input_trims={"x": (0, 5)},
    )
    e = emit(g)
    assert e.inputs == ["a.mp4", "a.mp4"]
    assert build_ffmpeg_args(e, "out.mp4").count("-i") == 2


def test_dedup_is_blocked_by_two_different_trim_windows() -> None:
    """The splice shape itself: same file, two DIFFERENT windows -> two -i's."""
    g = _graph(
        [],
        [_out("src:x:v:0"), _out("src:y:v:0")],
        input_paths=["a.mp4", "a.mp4"],
        sources={"x": 0, "y": 1},
        input_trims={"x": (None, 5), "y": (5, None)},
    )
    e = emit(g)
    assert e.inputs == ["a.mp4", "a.mp4"]
    args = build_ffmpeg_args(e, "out.mp4")
    assert args.count("-i") == 2
    assert args[:6] == ["ffmpeg", "-to", "5", "-i", "a.mp4", "-ss"]


def test_dedup_only_merges_matching_pairs_in_a_three_input_graph() -> None:
    """x/z share a.mp4 untrimmed and merge; y's b.mp4 stays on its own -i."""
    g = _graph(
        [],
        [_out("src:x:v:0"), _out("src:y:v:0"), _out("src:z:v:0")],
        input_paths=["a.mp4", "b.mp4", "a.mp4"],
        sources={"x": 0, "y": 1, "z": 2},
    )
    e = emit(g)
    assert e.inputs == ["a.mp4", "b.mp4"]
    assert build_ffmpeg_args(e, "out.mp4").count("-i") == 2


def test_dedup_renumbers_source_specs_onto_the_shared_slot() -> None:
    """Both aliases' stream specs resolve to the SAME (merged) input index."""
    g = _graph(
        [_node("n1", "hflip", {}, ["src:x:v:0"])],
        [_out("n1"), _out("src:y:v:0")],
        input_paths=["a.mp4", "a.mp4"],
        sources={"x": 0, "y": 1},
    )
    e = emit(g)
    assert e.filter_complex == "[0:v:0]hflip[out0]"
    assert e.maps[1].target == "0:v:0"  # y's passthrough map, same merged index


# ---------------------------------------------------------------------------
# unused URL-source inputs: a `FROM files(...) s` row WHERE/LIMIT filtered
# out still mints an `-i` today; emit drops it and renumbers everything else
# that names an input index.
# ---------------------------------------------------------------------------


def _url_source(alias: str, rows: list[tuple[str, int]]) -> UrlSource:
    """A `UrlSource` with one row per `(url, input)` pair, no module columns."""
    return UrlSource(
        alias=alias,
        module="modules/files.wasm",
        params="{}",
        document=None,
        rows=tuple(UrlSourceRow(url=url, input=index) for url, index in rows),
    )


def test_dropping_an_unused_url_input_renumbers_maps_and_pad_labels() -> None:
    """Three URL-source inputs (a, b, c); only a and c are ever read (a's and
    c's AUDIO feed a concat, c's VIDEO is a bare passthrough map). b's `-i`
    is dropped and c's index shifts from 2 to 1 everywhere it appears: the
    concat chain's pad labels and the passthrough `-map` alike."""
    g = _graph(
        [
            _node(
                "cc", "concat", {"n": 2, "v": 0, "a": 1},
                ["src:x0:a:0", "src:x2:a:0"], ["audio"],
            ),
        ],
        [_out("cc", "audio"), _out("src:x2:v:0")],
        input_paths=["a.mp4", "b.mp4", "c.mp4"],
        sources={"x0": 0, "x1": 1, "x2": 2},
        url_sources={"s": _url_source("s", [("a.mp4", 0), ("b.mp4", 1), ("c.mp4", 2)])},
    )
    e = emit(g)
    assert e.inputs == ["a.mp4", "c.mp4"]
    assert e.filter_complex == "[0:a:0][1:a:0]concat=n=2:v=0:a=1[out0]"
    assert e.maps[1].target == "1:v:0"
    args = build_ffmpeg_args(e, "out.mkv")
    assert args[:5] == ["ffmpeg", "-i", "a.mp4", "-i", "c.mp4"]


def test_an_unmapped_ordinary_input_is_not_dropped() -> None:
    """Only a URL-source row's own input is ever a drop candidate: a plain
    `input()` alias the query names but never maps keeps its slot exactly as
    it does today -- that pin is unrelated and lives elsewhere."""
    g = _graph(
        [],
        [_out("src:x0:v:0")],
        input_paths=["a.mp4", "b.mp4"],
        sources={"x0": 0, "plain": 1},
        url_sources={"s": _url_source("s", [("a.mp4", 0)])},
    )
    e = emit(g)
    assert e.inputs == ["a.mp4", "b.mp4"]


def test_dropping_an_unused_url_input_shifts_windows_and_options_with_it() -> None:
    """c's trim window and input options are alias-keyed, so they follow c's
    alias to its new (shifted) input position without being told to."""
    g = _graph(
        [],
        [_out("src:x0:v:0"), _out("src:x2:v:0")],
        input_paths=["a.mp4", "b.mp4", "c.mp4"],
        sources={"x0": 0, "x1": 1, "x2": 2},
        input_trims={"x2": (0.0, 5.0)},
        input_options={"x2": {"loop": True}},
        url_sources={"s": _url_source("s", [("a.mp4", 0), ("b.mp4", 1), ("c.mp4", 2)])},
    )
    e = emit(g)
    assert e.inputs == ["a.mp4", "c.mp4"]
    assert e.input_trims == [None, (0.0, 5.0)]
    assert e.input_options == [{}, {"loop": True}]


def test_dropping_an_unused_url_input_updates_everything_else_that_names_it() -> None:
    """The renumbering reaches every OTHER place a raw input index lives, not
    just `Graph.sources`: a sink's `-map_chapters` source (here the only
    thing keeping c's `-i` alive at all -- nothing maps it) and the
    surviving row's own `UrlSourceRow.input`. b's row, backing nothing at
    all, is dropped outright rather than left pointing at a removed slot."""
    g = _graph(
        [],
        [_out("src:x0:v:0")],
        sink=_sink("o.mp4", chapters=2),
        input_paths=["a.mp4", "b.mp4", "c.mp4"],
        sources={"x0": 0, "x1": 1, "x2": 2},
        url_sources={"s": _url_source("s", [("a.mp4", 0), ("b.mp4", 1), ("c.mp4", 2)])},
    )
    dropped = _drop_unused_url_inputs(g)
    assert dropped.input_paths == ["a.mp4", "c.mp4"]
    assert dropped.sources == {"x0": 0, "x2": 1}
    assert dropped.sinks[0].chapters == 1
    assert [(row.url, row.input) for row in dropped.url_sources["s"].rows] == [
        ("a.mp4", 0), ("c.mp4", 1),
    ]


# ---------------------------------------------------------------------------
# command sequences: build_ffmpeg_commands and the two_pass sink option
# ---------------------------------------------------------------------------


def _two_pass_sink(**extra: object) -> SinkUnit:
    options: dict[str, object] = {
        "video_codec": "libx264",
        "video_bitrate": "2500k",
        "two_pass": True,
    }
    options.update(extra)
    return _sink(path="out.mp4", options=options)


def test_an_ordinary_query_is_a_sequence_of_one() -> None:
    """Every non-two_pass compile: one command, byte-identical to
    `build_ffmpeg_args`."""
    g = _graph([_node("n1", "hflip", {}, ["src:a:v:0"])], [_out("n1")])
    e = emit(g)
    commands = build_ffmpeg_commands(e, "out.mp4")
    assert commands == [build_ffmpeg_args(e, "out.mp4")]


def test_a_multi_sink_query_is_still_a_sequence_of_one() -> None:
    """Several output FILES stay ONE ffmpeg command; only two_pass splits."""
    g = _graph(
        [],
        [],
        sinks=[
            SinkUnit(outputs=[_out("src:a:v:0")], path="one.mp4"),
            SinkUnit(outputs=[_out("src:a:a:0", "audio")], path="two.m4a"),
        ],
    )
    e = emit(g)
    assert len(build_ffmpeg_commands(e)) == 1


def test_two_pass_emits_two_commands() -> None:
    g = _graph(
        [],
        [_out("src:a:v:0"), _out("src:a:a:0", "audio")],
        sink=_two_pass_sink(audio_codec="aac"),
    )
    first, second = build_ffmpeg_commands(emit(g))
    assert first == [
        "ffmpeg",
        "-i", "a.mp4",
        "-map", "0:v:0",
        "-c:0", "libx264",
        "-b:0", "2500k",
        "-pass", "1",
        "-passlogfile", "out.mp4",
        "-f", "null",
        "-",
    ]
    assert second == [
        "ffmpeg",
        "-i", "a.mp4",
        "-map", "0:v:0",
        "-map", "0:a:0",
        "-c:0", "libx264",
        "-b:0", "2500k",
        "-pass", "2",
        "-passlogfile", "out.mp4",
        "-c:1", "aac",
        "out.mp4",
    ]


def test_pass_one_drops_passthrough_non_video_maps() -> None:
    """Subtitles and copied audio are bare -maps: nothing needs them in pass 1."""
    g = _graph(
        [],
        [
            _out("src:a:v:0"),
            _out("src:a:a:0", "audio"),
            _out("src:a:s:0", "subtitle"),
        ],
        sink=_two_pass_sink(),
    )
    first = build_ffmpeg_commands(emit(g))[0]
    assert [first[i + 1] for i, tok in enumerate(first) if tok == "-map"] == ["0:v:0"]


def test_pass_one_keeps_a_filtered_audio_map() -> None:
    """A filtergraph output pad with no consumer is a hard ffmpeg error, so a
    FILTERED audio output stays mapped in pass 1 and encodes into the null
    muxer; only passthrough maps are dropped."""
    g = _graph(
        [
            _node("n1", "hflip", {}, ["src:a:v:0"]),
            _node("n2", "volume", {"volume": 0.5}, ["src:a:a:0"], outputs=["audio"]),
        ],
        [_out("n1"), _out("n2", "audio")],
        sink=_two_pass_sink(audio_codec="aac"),
    )
    first = build_ffmpeg_commands(emit(g))[0]
    assert [first[i + 1] for i, tok in enumerate(first) if tok == "-map"] == [
        "[out0]",
        "[out1]",
    ]
    assert first[first.index("-c:1") : first.index("-c:1") + 2] == ["-c:1", "aac"]


def test_pass_one_renumbers_stream_indices_over_its_own_maps() -> None:
    """Audio first in the SELECT: pass 2 calls the video stream 1, pass 1
    calls it 0, because indices are per FILE and pass 1 writes fewer."""
    g = _graph(
        [],
        [_out("src:a:a:0", "audio"), _out("src:a:v:0")],
        sink=_two_pass_sink(),
    )
    first, second = build_ffmpeg_commands(emit(g))
    assert "-c:0" in first and "libx264" == first[first.index("-c:0") + 1]
    assert "-c:1" in second and "libx264" == second[second.index("-c:1") + 1]


def test_passlogfile_follows_the_out_path_override() -> None:
    """`out_path` renames the destination, so the stats file moves with it."""
    g = _graph([], [_out("src:a:v:0")], sink=_two_pass_sink())
    for command in build_ffmpeg_commands(emit(g), "elsewhere/final.mkv"):
        assert command[command.index("-passlogfile") + 1] == "elsewhere/final.mkv"
    assert build_ffmpeg_commands(emit(g))[0][-5:] == [
        "-passlogfile",
        "out.mp4",
        "-f",
        "null",
        "-",
    ]


def test_pass_one_overrides_an_explicit_format_option() -> None:
    """`format 'mp4'` is the pass-2 container; pass 1 muxes to null regardless."""
    g = _graph([], [_out("src:a:v:0")], sink=_two_pass_sink(format="mp4"))
    first, second = build_ffmpeg_commands(emit(g))
    assert first[-3:] == ["-f", "null", "-"]
    assert first.count("-f") == 1
    assert second[second.index("-f") + 1] == "mp4"


def test_two_pass_false_emits_one_command_and_no_pass_flags() -> None:
    g = _graph([], [_out("src:a:v:0")], sink=_sink("out.mp4", {"two_pass": False}))
    commands = build_ffmpeg_commands(emit(g))
    assert len(commands) == 1
    assert "-pass" not in commands[0]


def test_build_ffmpeg_args_refuses_a_two_pass_emitted() -> None:
    """The single-command seam never silently drops a pass."""
    g = _graph([], [_out("src:a:v:0")], sink=_two_pass_sink())
    with pytest.raises(ValueError, match="build_ffmpeg_commands"):
        build_ffmpeg_args(emit(g))


def test_two_pass_across_several_groups_raises() -> None:
    """lower rejects this shape; a hand-built Emitted gets the ValueError."""
    g = _graph(
        [],
        [],
        sinks=[
            SinkUnit(
                outputs=[_out("src:a:v:0")],
                path="one.mp4",
                options={"video_codec": "libx264", "two_pass": True},
            ),
            SinkUnit(outputs=[_out("src:a:a:0", "audio")], path="two.m4a"),
        ],
    )
    with pytest.raises(ValueError, match="two_pass writes one file"):
        build_ffmpeg_commands(emit(g))


def test_two_pass_with_no_destination_anywhere_raises() -> None:
    g = _graph(
        [],
        [],
        sinks=[SinkUnit(outputs=[_out("src:a:v:0")], options={"two_pass": True})],
    )
    with pytest.raises(ValueError, match="no output path"):
        build_ffmpeg_commands(emit(g))


def test_two_pass_reaching_the_renderer_with_no_pass_number_is_internal() -> None:
    """Defensive: both public entry points supply one or refuse."""
    e = emit(_graph([], [_out("src:a:v:0")], sink=_two_pass_sink()))
    with pytest.raises(FfrwdError) as excinfo:
        _render_command(e, "out.mp4", None)
    assert excinfo.value.code == ErrorCode.INTERNAL


# ---------------------------------------------------------------------------
# real ffmpeg sanity check (exec-marked)
# ---------------------------------------------------------------------------

_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
_AV = _FIXTURES_DIR / "av.mp4"
_SUBPROCESS_TIMEOUT = 60.0


def _require_ffmpeg_and_fixture() -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg/ffprobe not found on PATH")
    if not _AV.exists():
        pytest.skip(f"fixture missing: {_AV} (run scripts/gen_fixtures.py first)")


def _run_ffmpeg(args: list[str]) -> None:
    args = list(args)
    args.insert(1, "-y")
    result = subprocess.run(args, capture_output=True, text=True, timeout=_SUBPROCESS_TIMEOUT)
    assert result.returncode == 0, f"{args}\n{result.stderr}"


def _probe_stream_tags(path: Path) -> list[dict[str, str]]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type:stream_tags=filename,mimetype",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=_SUBPROCESS_TIMEOUT,
    )
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    return [dict(stream.get("tags") or {}) for stream in data["streams"]]


def _probe_codec_types(path: Path) -> list[str]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=_SUBPROCESS_TIMEOUT,
    )
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    return [str(stream["codec_type"]) for stream in data["streams"]]


@pytest.mark.exec
def test_multi_map_command_runs_under_real_ffmpeg(tmp_path: Path) -> None:
    """hflip(a.video[1]) + volume(a.audio[1], 0.5) -> a real two-stream file."""
    _require_ffmpeg_and_fixture()
    g = _graph(
        [
            _node("n1", "hflip", {}, ["src:a:v:0"]),
            _node("n2", "volume", {"volume": 0.5}, ["src:a:a:0"], ["audio"]),
        ],
        [_out("n1"), _out("n2", "audio", metadata={"language": "eng"})],
        input_paths=[str(_AV)],
    )
    e = emit(g)
    assert e.filter_complex == "[0:v:0]hflip[out0];[0:a:0]volume=volume=0.5[out1]"

    out_path = tmp_path / "multimap.mp4"
    _run_ffmpeg(build_ffmpeg_args(e, str(out_path)))

    assert out_path.exists()
    assert _probe_codec_types(out_path) == ["video", "audio"]


@pytest.mark.exec
def test_passthrough_map_runs_under_real_ffmpeg(tmp_path: Path) -> None:
    """hflip(a.video[1]) + a.audio[1] -- the audio stream is copied, not filtered."""
    _require_ffmpeg_and_fixture()
    g = _graph(
        [_node("n1", "hflip", {}, ["src:a:v:0"])],
        [_out("n1"), _out("src:a:a:0", "audio", metadata={"language": "eng"})],
        input_paths=[str(_AV)],
    )
    e = emit(g)
    out_path = tmp_path / "passthrough.mp4"
    _run_ffmpeg(build_ffmpeg_args(e, str(out_path)))

    assert _probe_codec_types(out_path) == ["video", "audio"]


@pytest.mark.exec
def test_an_attached_file_runs_under_real_ffmpeg(tmp_path: Path) -> None:
    """The attachment lands after the mapped streams, tagged as written."""
    _require_ffmpeg_and_fixture()
    font = _FIXTURES_DIR / "font.ttf"
    if not font.exists():
        pytest.skip(f"fixture missing: {font} (run scripts/gen_fixtures.py first)")
    sink = _sink(
        path="out.mkv",
        attachments=[
            Attachment(str(font), "renamed.ttf", "application/x-truetype-font")
        ],
    )
    g = _graph(
        [],
        [_out("src:a:v:0"), _out("src:a:a:0", "audio")],
        input_paths=[str(_AV)],
        sink=sink,
    )
    out_path = tmp_path / "attached.mkv"
    _run_ffmpeg(build_ffmpeg_args(emit(g), str(out_path)))

    assert _probe_codec_types(out_path) == ["video", "audio", "attachment"]
    assert _probe_stream_tags(out_path)[2] == {
        "filename": "renamed.ttf",
        "mimetype": "application/x-truetype-font",
    }


@pytest.mark.exec
def test_a_deduped_compile_runs_under_real_ffmpeg(tmp_path: Path) -> None:
    """Two untrimmed, option-free aliases of the SAME real file:
    one decoded input feeds both a filtered chain and a passthrough map."""
    _require_ffmpeg_and_fixture()
    g = _graph(
        [_node("n1", "hflip", {}, ["src:x:v:0"])],
        [_out("n1"), _out("src:y:a:0", "audio", metadata={"language": "eng"})],
        input_paths=[str(_AV), str(_AV)],
        sources={"x": 0, "y": 1},
    )
    e = emit(g)
    assert e.inputs == [str(_AV)]  # deduped onto one -i
    args = build_ffmpeg_args(e, str(tmp_path / "deduped.mp4"))
    assert args.count("-i") == 1

    out_path = tmp_path / "deduped.mp4"
    _run_ffmpeg(build_ffmpeg_args(e, str(out_path)))

    assert _probe_codec_types(out_path) == ["video", "audio"]
