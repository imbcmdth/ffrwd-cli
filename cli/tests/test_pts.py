from __future__ import annotations

from ffrwd.ir import Graph, ModuleSource, Node, Output, SinkUnit, SourceTrack, StreamType
from ffrwd.pts import insert_pts_resets


def _out(ref: str, type_: StreamType = "video", name: str | None = None) -> Output:
    return Output(ref=ref, type=type_, name=name, metadata={})


def _no_trim_graph() -> Graph:
    g = Graph(input_paths=["a.mp4"], sources={"a": 0})
    g.nodes["n0"] = Node(id="n0", filter="hflip", args={}, inputs=["src:a:v:0"], outputs=["video"])
    g.sinks = [SinkUnit(outputs=[_out("n0")])]
    return g


def _trim_into_filter_graph() -> Graph:
    """A trimmed stream that nothing has reset: hflip is not setpts."""
    g = Graph(input_paths=["a.mp4"], sources={"a": 0})
    g.nodes["n1"] = Node(
        id="n1", filter="trim", args={"start": 5}, inputs=["src:a:v:0"], outputs=["video"]
    )
    g.nodes["n2"] = Node(id="n2", filter="hflip", args={}, inputs=["n1"], outputs=["video"])
    g.sinks = [SinkUnit(outputs=[_out("n2")])]
    return g


def _trim_direct_to_output_graph() -> Graph:
    """One trim, written to a file with no filter in between at all."""
    g = Graph(input_paths=["a.mp4"], sources={"a": 0})
    g.nodes["n1"] = Node(
        id="n1", filter="trim", args={"start": 5}, inputs=["src:a:v:0"], outputs=["video"]
    )
    g.sinks = [SinkUnit(outputs=[_out("n1")])]
    return g


def _atrim_graph() -> Graph:
    g = Graph(input_paths=["a.mp4"], sources={"a": 0})
    g.nodes["n1"] = Node(
        id="n1", filter="atrim", args={"start": 5}, inputs=["src:a:a:0"], outputs=["audio"]
    )
    g.nodes["n2"] = Node(id="n2", filter="highpass", args={}, inputs=["n1"], outputs=["audio"])
    g.sinks = [SinkUnit(outputs=[_out("n2", "audio")])]
    return g


def _mixed_audio_video_trim_graph() -> Graph:
    """One query, one video trim and one audio trim, each unreset."""
    g = Graph(input_paths=["a.mp4"], sources={"a": 0})
    g.nodes["v1"] = Node(
        id="v1", filter="trim", args={"start": 5}, inputs=["src:a:v:0"], outputs=["video"]
    )
    g.nodes["v2"] = Node(id="v2", filter="hflip", args={}, inputs=["v1"], outputs=["video"])
    g.nodes["a1"] = Node(
        id="a1", filter="atrim", args={"start": 5}, inputs=["src:a:a:0"], outputs=["audio"]
    )
    g.nodes["a2"] = Node(id="a2", filter="highpass", args={}, inputs=["a1"], outputs=["audio"])
    g.sinks = [SinkUnit(outputs=[_out("v2", "video"), _out("a2", "audio")])]
    return g


def _author_written_setpts_graph() -> Graph:
    """The author's own setpts sits directly on the trim's only edge."""
    g = Graph(input_paths=["a.mp4"], sources={"a": 0})
    g.nodes["n1"] = Node(
        id="n1", filter="trim", args={"start": 5}, inputs=["src:a:v:0"], outputs=["video"]
    )
    g.nodes["n2"] = Node(
        id="n2", filter="setpts", args={"expr": "PTS-STARTPTS"}, inputs=["n1"], outputs=["video"]
    )
    g.sinks = [SinkUnit(outputs=[_out("n2")])]
    return g


def _speed_macro_graph() -> Graph:
    """ffrwd.speed's own expansion: a setpts with a DIFFERENT expr, still
    counting as the author taking control of timing."""
    g = Graph(input_paths=["a.mp4"], sources={"a": 0})
    g.nodes["n1"] = Node(
        id="n1", filter="trim", args={"start": 5}, inputs=["src:a:v:0"], outputs=["video"]
    )
    g.nodes["n2"] = Node(
        id="n2", filter="setpts", args={"expr": "PTS/2"}, inputs=["n1"], outputs=["video"]
    )
    g.sinks = [SinkUnit(outputs=[_out("n2")])]
    return g


def _mixed_protection_graph() -> Graph:
    """One trim, two consumers: one already has its own setpts, one does not."""
    g = Graph(input_paths=["a.mp4"], sources={"a": 0})
    g.nodes["n1"] = Node(
        id="n1", filter="trim", args={"start": 5}, inputs=["src:a:v:0"], outputs=["video"]
    )
    g.nodes["n2"] = Node(id="n2", filter="hflip", args={}, inputs=["n1"], outputs=["video"])
    g.nodes["n3"] = Node(
        id="n3", filter="setpts", args={"expr": "PTS-STARTPTS"}, inputs=["n1"], outputs=["video"]
    )
    g.sinks = [SinkUnit(outputs=[_out("n2", name="a"), _out("n3", name="b")])]
    return g


def _two_trims_into_concat_graph() -> Graph:
    """Recipe 77's shape: two trims of the same source, joined by concat."""
    g = Graph(input_paths=["a.mp4"], sources={"a": 0})
    g.nodes["n1"] = Node(
        id="n1", filter="trim", args={"start": 0, "end": 1}, inputs=["src:a:v:0"],
        outputs=["video"],
    )
    g.nodes["n2"] = Node(
        id="n2", filter="trim", args={"start": 2, "end": 3}, inputs=["src:a:v:0"],
        outputs=["video"],
    )
    g.nodes["n3"] = Node(
        id="n3", filter="concat", args={"n": 2, "v": 1, "a": 0}, inputs=["n1", "n2"],
        outputs=["video"],
    )
    g.sinks = [SinkUnit(outputs=[_out("n3")])]
    return g


def test_graph_with_no_trims_is_unchanged() -> None:
    g = _no_trim_graph()
    out = insert_pts_resets(g)
    assert out.to_dict() == g.to_dict()


def test_insert_pts_resets_does_not_mutate_input_graph() -> None:
    g = _trim_into_filter_graph()
    before = g.to_dict()
    insert_pts_resets(g)
    after = g.to_dict()
    assert after == before
    # sanity: this graph needs a reset, so a no-op result would be a weak test.
    assert insert_pts_resets(g).to_dict() != before


def test_trim_into_an_ordinary_filter_gets_a_reset_spliced_in_front() -> None:
    g = _trim_into_filter_graph()
    out = insert_pts_resets(g)

    assert list(out.nodes.keys()) == ["n1", "n1_pts", "n2"]

    reset = out.nodes["n1_pts"]
    assert reset.filter == "setpts"
    assert reset.args == {"expr": "PTS-STARTPTS"}
    assert reset.inputs == ["n1"]
    assert reset.outputs == ["video"]

    assert out.nodes["n2"].inputs == ["n1_pts"]
    assert out.outputs == [_out("n2")]


def test_trim_mapped_straight_to_an_output_gets_a_reset() -> None:
    g = _trim_direct_to_output_graph()
    out = insert_pts_resets(g)

    assert list(out.nodes.keys()) == ["n1", "n1_pts"]
    assert out.nodes["n1_pts"].filter == "setpts"
    assert out.outputs == [_out("n1_pts")]


def test_atrim_gets_asetpts_not_setpts() -> None:
    g = _atrim_graph()
    out = insert_pts_resets(g)

    assert list(out.nodes.keys()) == ["n1", "n1_pts", "n2"]
    reset = out.nodes["n1_pts"]
    assert reset.filter == "asetpts"
    assert reset.args == {"expr": "PTS-STARTPTS"}
    assert reset.outputs == ["audio"]
    assert out.nodes["n2"].inputs == ["n1_pts"]


def test_one_query_with_a_video_and_an_audio_trim_gets_both_reset_filters() -> None:
    g = _mixed_audio_video_trim_graph()
    out = insert_pts_resets(g)

    assert out.nodes["v1_pts"].filter == "setpts"
    assert out.nodes["a1_pts"].filter == "asetpts"
    assert out.nodes["v2"].inputs == ["v1_pts"]
    assert out.nodes["a2"].inputs == ["a1_pts"]


def test_author_written_setpts_means_exactly_one_setpts_on_that_path() -> None:
    g = _author_written_setpts_graph()
    out = insert_pts_resets(g)

    # No reset inserted: n2 (the author's own setpts) still reads n1 directly.
    assert list(out.nodes.keys()) == ["n1", "n2"]
    assert out.nodes["n2"].inputs == ["n1"]
    setpts_nodes = [n for n in out.nodes.values() if n.filter == "setpts"]
    assert len(setpts_nodes) == 1


def test_speed_macros_setpts_suppresses_the_reset_too() -> None:
    """`ffrwd.speed` expands to its own setpts (a different expr) -- that
    still counts as the author taking control of timing."""
    g = _speed_macro_graph()
    out = insert_pts_resets(g)

    assert list(out.nodes.keys()) == ["n1", "n2"]
    assert out.nodes["n2"].inputs == ["n1"]
    assert out.nodes["n2"].args == {"expr": "PTS/2"}


def test_protection_is_per_consumer_not_per_trim() -> None:
    """One trim, two consumers: only the unprotected edge gets a reset."""
    g = _mixed_protection_graph()
    out = insert_pts_resets(g)

    assert list(out.nodes.keys()) == ["n1", "n1_pts", "n2", "n3"]
    # n2 (hflip) had no setpts on its edge -- rewired to the new reset.
    assert out.nodes["n2"].inputs == ["n1_pts"]
    # n3 IS the author's own setpts -- its edge stays pointed at n1 directly,
    # not doubled up with the new reset node.
    assert out.nodes["n3"].inputs == ["n1"]
    setpts_nodes = [n for n in out.nodes.values() if n.filter == "setpts"]
    assert len(setpts_nodes) == 2  # the inserted one (n1_pts) and n3, distinct


def test_two_trims_concatenated_each_get_their_own_reset() -> None:
    g = _two_trims_into_concat_graph()
    out = insert_pts_resets(g)

    # Reset nodes are inserted immediately before their first consumer (n3),
    # same positioning rule as `split.py`'s own split nodes -- not right
    # after the trim they reset.
    assert list(out.nodes.keys()) == ["n1", "n2", "n1_pts", "n2_pts", "n3"]
    assert out.nodes["n1_pts"].inputs == ["n1"]
    assert out.nodes["n2_pts"].inputs == ["n2"]
    assert out.nodes["n3"].inputs == ["n1_pts", "n2_pts"]


def test_insert_pts_resets_is_idempotent() -> None:
    """Every trim's consumer is a reset (or already protected) after one
    pass, so a second pass is a no-op."""
    g = _two_trims_into_concat_graph()
    once = insert_pts_resets(g)
    twice = insert_pts_resets(once)
    assert once.to_dict() == twice.to_dict()


def test_module_source_survives_the_pass() -> None:
    """A ``RETURNS source`` binding carries no trim/atrim of its own, but the
    pass must not drop it while rebuilding the graph."""
    g = _no_trim_graph()
    g.module_sources["s"] = ModuleSource(
        alias="s",
        module="replay.wasm",
        params="{}",
        tracks=(
            SourceTrack(
                ref="src:s:v:0", kind="video", codec="h264", time_base=(1, 90000), row=0
            ),
        ),
        bounded=True,
    )
    out = insert_pts_resets(g)
    assert out.module_sources == g.module_sources


def test_a_rows_edge_survives_the_pass() -> None:
    """Rows carry no timestamps to reset, but the pass rebuilds every node,
    so a rows module's edge has to ride through naming the same producer."""
    g = _no_trim_graph()
    g.nodes["rows"] = Node(
        id="rows", filter="fauxlate.wasm", args={}, inputs=[], outputs=[],
        rows_inputs=["n0"],
    )
    out = insert_pts_resets(g)
    assert out.nodes["rows"].rows_inputs == ["n0"]
    assert out.nodes["rows"].inputs == [] and out.nodes["rows"].outputs == []
