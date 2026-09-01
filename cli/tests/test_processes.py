"""Partitioning a logical graph into the processes that run it.

Unit tier: synthetic :class:`~ffrwd.probe.ProbeResult`s, no fixtures, no
ffmpeg. Every graph here is written by hand in the shape lowering produces --
split-complete, topologically ordered.
"""

from __future__ import annotations

import pytest

from ffrwd.errors import ErrorCode, FfrwdError
from ffrwd.ir import Graph, Node, Output, SinkUnit, StreamType
from ffrwd.probe import ProbeResult, StreamMeta
from ffrwd.processes import (
    COPY_CODEC,
    NUT,
    PCM_F32LE,
    PIPE,
    RAWVIDEO,
    AudioFormat,
    FfmpegProcess,
    FileEdge,
    ModuleShape,
    ProcessPlan,
    SidecarProcess,
    StreamEdge,
    VideoFormat,
    check_spellable,
    external_ids,
    from_commands,
    is_live,
    partition,
)


def _out(ref: str, type_: StreamType = "video", name: str | None = None) -> Output:
    return Output(ref=ref, type=type_, name=name, metadata={})


def _video_probe() -> ProbeResult:
    return ProbeResult(
        streams=[
            StreamMeta(
                type="video",
                index=0,
                metadata={},
                width=1920,
                height=1080,
                fps="30000/1001",
                sample_rate=None,
            ),
            StreamMeta(
                type="audio",
                index=0,
                metadata={},
                width=None,
                height=None,
                fps=None,
                sample_rate=48000,
                channels=2,
            ),
        ],
        duration=12.0,
    )


def _plain_graph() -> Graph:
    """No external node anywhere: crop then scale, into one file."""
    g = Graph(input_paths=["a.mp4"], sources={"a": 0})
    g.nodes["n0"] = Node(
        id="n0",
        filter="crop",
        args={"w": 600, "h": 200},
        inputs=["src:a:v:0"],
        outputs=["video"],
    )
    g.nodes["n1"] = Node(
        id="n1", filter="scale", args={"w": 320, "h": -2}, inputs=["n0"], outputs=["video"]
    )
    g.sinks = [SinkUnit(outputs=[_out("n1")], path="out.mp4")]
    return g


def _series_graph() -> Graph:
    """Two external nodes in series."""
    g = Graph(input_paths=["a.mp4"], sources={"a": 0})
    g.nodes["e0"] = Node(
        id="e0", filter="denoise", args={}, inputs=["src:a:v:0"], outputs=["video"]
    )
    g.nodes["e1"] = Node(id="e1", filter="grade", args={}, inputs=["e0"], outputs=["video"])
    g.sinks = [SinkUnit(outputs=[_out("e1")], path="out.mp4")]
    return g


def _sandwich_graph() -> Graph:
    """An ordinary filter between two external nodes."""
    g = Graph(input_paths=["a.mp4"], sources={"a": 0})
    g.nodes["e0"] = Node(
        id="e0", filter="denoise", args={}, inputs=["src:a:v:0"], outputs=["video"]
    )
    g.nodes["n0"] = Node(id="n0", filter="hue", args={"h": 30}, inputs=["e0"], outputs=["video"])
    g.nodes["e1"] = Node(id="e1", filter="grade", args={}, inputs=["n0"], outputs=["video"])
    g.sinks = [SinkUnit(outputs=[_out("e1")], path="out.mp4")]
    return g


def _passthrough_audio_graph() -> Graph:
    """The video visits an external node; the audio is mapped straight through."""
    g = Graph(input_paths=["a.mp4"], sources={"a": 0})
    g.nodes["e0"] = Node(
        id="e0", filter="denoise", args={}, inputs=["src:a:v:0"], outputs=["video"]
    )
    g.sinks = [
        SinkUnit(outputs=[_out("e0"), _out("src:a:a:0", "audio")], path="out.mp4")
    ]
    return g


def _fanout_graph() -> Graph:
    """One decode feeds two chains, each ending at an external node."""
    g = Graph(input_paths=["a.mp4"], sources={"a": 0})
    g.nodes["sp"] = Node(
        id="sp",
        filter="split",
        args={"n": 2},
        inputs=["src:a:v:0"],
        outputs=["video", "video"],
    )
    g.nodes["n0"] = Node(id="n0", filter="hue", args={}, inputs=["sp:0"], outputs=["video"])
    g.nodes["n1"] = Node(id="n1", filter="eq", args={}, inputs=["sp:1"], outputs=["video"])
    g.nodes["e0"] = Node(id="e0", filter="left", args={}, inputs=["n0"], outputs=["video"])
    g.nodes["e1"] = Node(id="e1", filter="right", args={}, inputs=["n1"], outputs=["video"])
    g.sinks = [
        SinkUnit(outputs=[_out("e0")], path="left.mp4"),
        SinkUnit(outputs=[_out("e1")], path="right.mp4"),
    ]
    return g


def _mixed_consumers_graph() -> Graph:
    """One stream feeds an ffmpeg filter and an external node: a split across
    two processes."""
    g = Graph(input_paths=["a.mp4"], sources={"a": 0})
    g.nodes["sp"] = Node(
        id="sp",
        filter="split",
        args={"n": 2},
        inputs=["src:a:v:0"],
        outputs=["video", "video"],
    )
    g.nodes["e0"] = Node(id="e0", filter="invert", args={}, inputs=["sp:1"], outputs=["video"])
    g.nodes["n0"] = Node(
        id="n0", filter="hstack", args={}, inputs=["sp:0", "e0"], outputs=["video"]
    )
    g.sinks = [SinkUnit(outputs=[_out("n0")], path="out.mp4")]
    return g


def _three_way_graph() -> Graph:
    """A split of three: two pads stay in one ffmpeg, one goes external."""
    g = Graph(input_paths=["a.mp4"], sources={"a": 0})
    g.nodes["sp"] = Node(
        id="sp",
        filter="split",
        args={"n": 3},
        inputs=["src:a:v:0"],
        outputs=["video", "video", "video"],
    )
    g.nodes["n0"] = Node(id="n0", filter="hue", args={}, inputs=["sp:0"], outputs=["video"])
    g.nodes["n1"] = Node(id="n1", filter="eq", args={}, inputs=["sp:1"], outputs=["video"])
    g.nodes["e0"] = Node(id="e0", filter="invert", args={}, inputs=["sp:2"], outputs=["video"])
    g.nodes["n2"] = Node(
        id="n2", filter="blend", args={}, inputs=["n0", "n1"], outputs=["video"]
    )
    g.nodes["n3"] = Node(
        id="n3", filter="hstack", args={}, inputs=["n2", "e0"], outputs=["video"]
    )
    g.sinks = [SinkUnit(outputs=[_out("n3")], path="out.mp4")]
    return g


# ---------------------------------------------------------------- degenerate


def test_no_external_node_is_one_ffmpeg_process() -> None:
    g = _plain_graph()
    plan = partition(g)

    assert len(plan.processes) == 1
    assert plan.sidecars == ()
    assert plan.edges == ()
    only = plan.processes[0]
    assert isinstance(only, FfmpegProcess)
    assert only.graph.to_dict() == g.to_dict()


def test_partition_does_not_mutate_the_graph() -> None:
    g = _sandwich_graph()
    before = g.to_dict()
    partition(g, external=external_ids("e0", "e1"))
    assert g.to_dict() == before


def test_no_external_node_is_a_single_stage() -> None:
    plan = partition(_plain_graph())
    assert [stage.processes for stage in plan.stages] == [("ffmpeg0",)]


# ---------------------------------------------------------------- structure


def test_two_external_nodes_in_series_are_one_sidecar_process() -> None:
    plan = partition(_series_graph(), external=external_ids("e0", "e1"))

    assert len(plan.sidecars) == 1
    region = plan.sidecars[0]
    assert region.nodes == ("e0", "e1")
    # No NUT hop between them: the region's only edges are its boundary.
    assert {(e.source, e.target) for e in plan.stream_edges} == {
        ("ffmpeg1", region.id),
        (region.id, "ffmpeg0"),
    }


def test_a_contracted_region_reads_and_writes_only_its_boundary() -> None:
    plan = partition(_series_graph(), external=external_ids("e0", "e1"))
    region = plan.sidecars[0]

    assert region.inputs == ("src:a:v:0",)
    assert region.outputs == ("video",)
    assert region.module == "denoise"  # the entry module keeps naming it
    assert [b.name for b in region.modules] == ["denoise", "grade"]


def test_filter_between_two_external_nodes_becomes_its_own_ffmpeg() -> None:
    plan = partition(_sandwich_graph(), external=external_ids("e0", "e1"))

    first = next(s for s in plan.sidecars if s.node == "e0")
    second = next(s for s in plan.sidecars if s.node == "e1")
    middle = next(e for e in plan.stream_edges if e.target == second.id).source
    assert isinstance(plan.process(middle), FfmpegProcess)

    hosted = plan.process(middle)
    assert isinstance(hosted, FfmpegProcess)
    assert list(hosted.graph.nodes) == ["n0"]
    # It reads the first sidecar and writes the second: sidecar -> ffmpeg ->
    # sidecar.
    assert {e.source for e in plan.stream_edges if e.target == middle} == {first.id}


def test_the_filter_between_reads_a_pipe_and_writes_a_pipe() -> None:
    plan = partition(_sandwich_graph(), external=external_ids("e0", "e1"))
    hosted = next(p for p in plan.ffmpeg if list(p.graph.nodes) == ["n0"])

    assert hosted.graph.input_paths == [PIPE]
    alias = next(iter(hosted.graph.sources))
    assert hosted.graph.nodes["n0"].inputs == [f"src:{alias}:v:0"]
    assert [unit.path for unit in hosted.graph.sinks] == [PIPE]
    assert hosted.graph.sinks[0].outputs[0].ref == "n0"


def test_a_stream_that_meets_no_external_node_stays_in_its_ffmpeg() -> None:
    plan = partition(_passthrough_audio_graph(), external=external_ids("e0"))

    writer = next(p for p in plan.ffmpeg if p.graph.sinks[0].path == "out.mp4")
    # The audio is still a plain map off the original file, and that file is
    # one of this process's own inputs.
    audio = writer.graph.sinks[0].outputs[1]
    assert audio.ref == "src:a:a:0"
    assert writer.graph.input_paths[writer.graph.sources["a"]] == "a.mp4"
    # No edge carries audio anywhere.
    assert all(isinstance(e.format, VideoFormat) for e in plan.stream_edges)
    # The video arrives on a pipe from the sidecar.
    video = writer.graph.sinks[0].outputs[0]
    assert video.ref.startswith("src:")
    assert writer.graph.input_paths[writer.graph.sources["e0"]] == PIPE


def test_the_decode_ahead_of_a_sidecar_carries_no_nodes() -> None:
    plan = partition(_passthrough_audio_graph(), external=external_ids("e0"))
    sidecar = plan.sidecars[0]
    feeder = next(e for e in plan.stream_edges if e.target == sidecar.id).source
    decode = plan.process(feeder)

    assert isinstance(decode, FfmpegProcess)
    assert decode.graph.nodes == {}
    assert decode.graph.input_paths == ["a.mp4"]
    assert [unit.path for unit in decode.graph.sinks] == [PIPE]
    assert decode.graph.sinks[0].outputs[0].ref == "src:a:v:0"


def test_a_sidecar_process_names_its_module_and_arguments() -> None:
    g = _series_graph()
    g.nodes["e0"].args = {"strength": 3}
    plan = partition(g, external=external_ids("e0", "e1"))

    first = next(s for s in plan.sidecars if s.node == "e0")
    assert isinstance(first, SidecarProcess)
    assert first.module == "denoise"
    assert first.args == {"strength": 3}
    assert first.inputs == ("src:a:v:0",)
    assert first.outputs == ("video",)


# ---------------------------------------------------------------- fan-out


def test_a_raw_stream_that_fans_out_duplicates_its_producer() -> None:
    plan = partition(_fanout_graph(), external=external_ids("e0", "e1"))

    producers = [p for p in plan.ffmpeg if any(u.path == PIPE for u in p.graph.sinks)]
    assert len(producers) == 2
    # Two decodes of the same input rather than one shared pipe.
    assert [p.graph.input_paths for p in producers] == [["a.mp4"], ["a.mp4"]]
    # One consumer each, so neither duplicate carries the split.
    assert {tuple(p.graph.nodes) for p in producers} == {("n0",), ("n1",)}
    assert [p.graph.nodes["n0"].inputs for p in producers if "n0" in p.graph.nodes] == [
        ["src:a:v:0"]
    ]


def test_no_ffmpeg_process_writes_more_than_one_raw_stream() -> None:
    plan = partition(_fanout_graph(), external=external_ids("e0", "e1"))

    for process in plan.ffmpeg:
        outgoing = [e for e in plan.stream_edges if e.source == process.id]
        assert len(outgoing) <= 1
        assert len([u for u in process.graph.sinks if u.path == PIPE]) <= 1


# ---------------------------------------------------------------- sibling legs


def _masked_graph() -> Graph:
    """One input, three legs: a passthrough, a blur, and a module's matte.

    The shape a masked blur lowers to. The merge reads all three; the module
    reads its own pad of the same split.
    """
    g = Graph(input_paths=["a.mp4"], sources={"a": 0})
    g.nodes["sp"] = Node(
        id="sp",
        filter="split",
        args={"n": 3},
        inputs=["src:a:v:0"],
        outputs=["video", "video", "video"],
    )
    g.nodes["n0"] = Node(
        id="n0", filter="format", args={"pix_fmts": "gbrp"}, inputs=["sp:0"], outputs=["video"]
    )
    g.nodes["n1"] = Node(
        id="n1", filter="gblur", args={"sigma": 12}, inputs=["sp:1"], outputs=["video"]
    )
    g.nodes["e0"] = Node(id="e0", filter="seg", args={}, inputs=["sp:2"], outputs=["video"])
    g.nodes["n2"] = Node(
        id="n2", filter="maskedmerge", args={}, inputs=["n0", "n1", "e0"], outputs=["video"]
    )
    g.sinks = [SinkUnit(outputs=[_out("n2")], path="out.mp4")]
    return g


def test_sibling_legs_into_one_consumer_share_a_decode() -> None:
    """The passthrough and blur legs are one process; the module's feeder is not."""
    plan = partition(_masked_graph(), external=external_ids("e0"))

    decoders = [p for p in plan.ffmpeg if "a.mp4" in p.graph.input_paths]
    assert len(decoders) == 2
    assert len(plan.ffmpeg) == 3
    assert len(plan.sidecars) == 1


def test_the_shared_feeder_carries_the_split_and_both_branches() -> None:
    plan = partition(_masked_graph(), external=external_ids("e0"))

    shared = next(
        p for p in plan.ffmpeg if sum(u.path == PIPE for u in p.graph.sinks) == 2
    )
    assert list(shared.graph.nodes) == ["sp", "n0", "n1"]
    # The split is cut to the two pads read here; the module's pad left.
    assert shared.graph.nodes["sp"].args["n"] == 2
    assert shared.graph.nodes["sp"].outputs == ["video", "video"]
    assert shared.graph.nodes["n0"].inputs == ["sp:0"]
    assert shared.graph.nodes["n1"].inputs == ["sp:1"]
    assert [u.outputs[0].ref for u in shared.graph.sinks] == ["n0", "n1"]

    merge = next(p for p in plan.ffmpeg if "n2" in p.graph.nodes)
    outgoing = [e for e in plan.stream_edges if e.source == shared.id]
    assert [e.ref for e in outgoing] == ["n0", "n1"]
    assert {e.target for e in outgoing} == {merge.id}


def test_the_modules_feeder_is_never_folded_into_the_shared_one() -> None:
    """Its consumer is the sidecar, not the merge, so it stays a decode of its own."""
    plan = partition(_masked_graph(), external=external_ids("e0"))

    sidecar = plan.sidecars[0]
    feeder_id = next(e.source for e in plan.stream_edges if e.target == sidecar.id)
    feeder = plan.process(feeder_id)
    assert isinstance(feeder, FfmpegProcess)
    assert feeder.graph.nodes == {}
    assert feeder.graph.sinks[0].outputs[0].ref == "src:a:v:0"
    assert sum(u.path == PIPE for u in feeder.graph.sinks) == 1


def test_legs_over_different_seek_windows_stay_two_processes() -> None:
    """Same file, but each alias carries its own window: nothing to share."""
    g = Graph(
        input_paths=["a.mp4", "a.mp4"],
        sources={"a": 0, "b": 1},
        input_trims={"a": (0.0, 5.0), "b": (5.0, 10.0)},
    )
    g.nodes["sp"] = Node(
        id="sp",
        filter="split",
        args={"n": 2},
        inputs=["src:a:v:0"],
        outputs=["video", "video"],
    )
    g.nodes["n0"] = Node(id="n0", filter="hue", args={}, inputs=["sp:0"], outputs=["video"])
    g.nodes["n1"] = Node(id="n1", filter="eq", args={}, inputs=["src:b:v:0"], outputs=["video"])
    g.nodes["e0"] = Node(id="e0", filter="seg", args={}, inputs=["sp:1"], outputs=["video"])
    g.nodes["n2"] = Node(
        id="n2", filter="maskedmerge", args={}, inputs=["n0", "n1", "e0"], outputs=["video"]
    )
    g.sinks = [SinkUnit(outputs=[_out("n2")], path="out.mp4")]

    plan = partition(g, external=external_ids("e0"))
    merge = next(p for p in plan.ffmpeg if "n2" in p.graph.nodes)
    feeders = [
        p
        for p in plan.ffmpeg
        if any(e.source == p.id and e.target == merge.id for e in plan.stream_edges)
    ]
    assert len(feeders) == 2
    assert all(sum(u.path == PIPE for u in p.graph.sinks) == 1 for p in feeders)


def test_the_shared_feeder_leaves_no_pad_unconnected() -> None:
    for process in partition(_masked_graph(), external=external_ids("e0")).ffmpeg:
        assert _dangling(process) == []


# ---------------------------------------------------------------- splits


def _dangling(process: FfmpegProcess) -> list[str]:
    """Pads this payload's nodes produce that nothing in it reads."""
    read = {ref for node in process.graph.nodes.values() for ref in node.inputs}
    read |= {o.ref for unit in process.graph.sinks for o in unit.outputs}
    loose: list[str] = []
    for node in process.graph.nodes.values():
        for pad in range(len(node.outputs)):
            if node.id not in read and f"{node.id}:{pad}" not in read:
                loose.append(f"{node.id}:{pad}")
    return loose


def test_no_payload_leaves_a_pad_unconnected() -> None:
    """An ffmpeg refuses to start with a filter output nothing reads."""
    for graph, external in (
        (_mixed_consumers_graph(), external_ids("e0")),
        (_three_way_graph(), external_ids("e0")),
        (_fanout_graph(), external_ids("e0", "e1")),
    ):
        for process in partition(graph, external=external).ffmpeg:
            assert _dangling(process) == []


def test_a_split_read_once_in_a_payload_dissolves() -> None:
    """Producer duplication leaves one consumer each, so the split goes."""
    plan = partition(_mixed_consumers_graph(), external=external_ids("e0"))

    producers = [p for p in plan.ffmpeg if any(u.path == PIPE for u in p.graph.sinks)]
    assert len(producers) == 2
    for producer in producers:
        assert producer.graph.nodes == {}
        assert producer.graph.input_paths == ["a.mp4"]
        # The consumer reads what the split read.
        assert producer.graph.sinks[0].outputs[0].ref == "src:a:v:0"

    stacker = next(p for p in plan.ffmpeg if "n0" in p.graph.nodes)
    assert list(stacker.graph.nodes) == ["n0"]
    assert stacker.graph.input_paths == [PIPE, PIPE]


def test_a_split_keeps_only_the_pads_its_own_payload_reads() -> None:
    plan = partition(_three_way_graph(), external=external_ids("e0"))

    hosted = next(p for p in plan.ffmpeg if "n2" in p.graph.nodes)
    split = hosted.graph.nodes["sp"]
    assert split.args["n"] == 2
    assert split.outputs == ["video", "video"]
    assert hosted.graph.nodes["n0"].inputs == ["sp:0"]
    assert hosted.graph.nodes["n1"].inputs == ["sp:1"]

    # The external node's producer reads the source straight off.
    sidecar = plan.sidecars[0]
    feeder = next(e for e in plan.stream_edges if e.target == sidecar.id).source
    decode = plan.process(feeder)
    assert isinstance(decode, FfmpegProcess)
    assert decode.graph.nodes == {}
    assert decode.graph.sinks[0].outputs[0].ref == "src:a:v:0"


def test_a_split_whose_consumers_share_a_process_is_untouched() -> None:
    """In-graph fan-out inside one ffmpeg keeps its split exactly as written."""
    g = _three_way_graph()
    del g.nodes["n3"]
    g.nodes["e0"] = Node(
        id="e0", filter="invert", args={}, inputs=["sp:2"], outputs=["video"]
    )
    g.sinks = [
        SinkUnit(outputs=[_out("n2")], path="out.mp4"),
        SinkUnit(outputs=[_out("e0")], path="inverted.mp4"),
    ]
    before = g.to_dict()

    plan = partition(g)  # nothing external: one process, the graph itself
    assert len(plan.processes) == 1
    only = plan.processes[0]
    assert isinstance(only, FfmpegProcess)
    assert only.graph.to_dict() == before


def test_fan_in_at_an_ffmpeg_process_is_left_alone() -> None:
    """Two sidecars feeding one ffmpeg is two pipes into one process."""
    g = Graph(input_paths=["a.mp4"], sources={"a": 0})
    g.nodes["sp"] = Node(
        id="sp",
        filter="split",
        args={"n": 2},
        inputs=["src:a:v:0"],
        outputs=["video", "video"],
    )
    g.nodes["e0"] = Node(id="e0", filter="left", args={}, inputs=["sp:0"], outputs=["video"])
    g.nodes["e1"] = Node(id="e1", filter="right", args={}, inputs=["sp:1"], outputs=["video"])
    g.nodes["n0"] = Node(
        id="n0", filter="hstack", args={}, inputs=["e0", "e1"], outputs=["video"]
    )
    g.sinks = [SinkUnit(outputs=[_out("n0")], path="out.mp4")]

    plan = partition(g, external=external_ids("e0", "e1"))
    stacker = next(p for p in plan.ffmpeg if "n0" in p.graph.nodes)
    incoming = [e for e in plan.stream_edges if e.target == stacker.id]
    assert len(incoming) == 2
    assert stacker.graph.input_paths == [PIPE, PIPE]
    assert stacker.graph.nodes["n0"].inputs == ["src:e0:v:0", "src:e1:v:0"]


# ---------------------------------------------------------------- formats


def test_a_video_edge_carries_the_probed_parameters() -> None:
    plan = partition(
        _series_graph(),
        external=external_ids("e0", "e1"),
        probes={"a": _video_probe()},
    )
    edge = plan.stream_edges[0]
    assert edge.format == VideoFormat(
        pix_fmt="yuv420p",
        width=1920,
        height=1080,
        timebase="1001/30000",
        container=NUT,
        codec=RAWVIDEO,
    )
    assert isinstance(edge.format, VideoFormat)
    assert edge.format.size == "1920x1080"


def test_a_video_edge_without_a_probe_still_names_its_format() -> None:
    plan = partition(_series_graph(), external=external_ids("e0", "e1"))
    edge = plan.stream_edges[0]
    assert isinstance(edge.format, VideoFormat)
    assert (edge.format.container, edge.format.codec) == (NUT, RAWVIDEO)
    assert edge.format.size is None
    assert edge.format.timebase is None


def test_an_audio_edge_carries_the_probed_rate_and_channels() -> None:
    g = Graph(input_paths=["a.mp4"], sources={"a": 0})
    g.nodes["e0"] = Node(
        id="e0", filter="deverb", args={}, inputs=["src:a:a:0"], outputs=["audio"]
    )
    g.sinks = [SinkUnit(outputs=[_out("e0", "audio")], path="out.m4a")]

    plan = partition(g, external=external_ids("e0"), probes={"a": _video_probe()})
    formats = {e.format for e in plan.stream_edges}
    assert formats == {AudioFormat(rate=48000, channels=2, container=NUT, codec=PCM_F32LE)}


def test_a_format_survives_the_filters_between_it_and_its_input() -> None:
    """The parameters come from the input the ref traces back to."""
    plan = partition(
        _sandwich_graph(),
        external=external_ids("e0", "e1"),
        probes={"a": _video_probe()},
    )
    edge = next(e for e in plan.stream_edges if e.ref == "n0")
    assert isinstance(edge.format, VideoFormat)
    assert edge.format.size == "1920x1080"


# ---------------------------------------------------------------- stages


def test_processes_wired_by_stream_edges_are_one_stage() -> None:
    plan = partition(_sandwich_graph(), external=external_ids("e0", "e1"))
    stages = plan.stages

    assert len(stages) == 1
    assert set(stages[0].processes) == {p.id for p in plan.processes}
    assert stages[0].index == 0


def test_file_edges_order_the_stages() -> None:
    first = Graph(input_paths=["a.mp4"], sources={"a": 0})
    first.sinks = [SinkUnit(outputs=[_out("src:a:v:0")], path="mid.mkv")]
    second = Graph(input_paths=["mid.mkv"], sources={"m": 0})
    second.sinks = [SinkUnit(outputs=[_out("src:m:v:0")], path="out.mp4")]

    plan = from_commands([first, second])

    assert [p.id for p in plan.processes] == ["ffmpeg0", "ffmpeg1"]
    assert [stage.processes for stage in plan.stages] == [("ffmpeg0",), ("ffmpeg1",)]
    edge = plan.file_edges[0]
    assert isinstance(edge, FileEdge)
    assert edge.format.content == "media"
    assert edge.format.path == "mid.mkv"


def test_a_command_pair_with_no_artifact_hands_over_rows() -> None:
    measure = Graph(input_paths=["a.mp4"], sources={"a": 0})
    measure.sinks = [SinkUnit(outputs=[_out("src:a:a:0", "audio")], path=None)]
    correct = Graph(input_paths=["a.mp4"], sources={"a": 0})
    correct.sinks = [SinkUnit(outputs=[_out("src:a:a:0", "audio")], path="out.m4a")]

    plan = from_commands([measure, correct])

    assert plan.stream_edges == ()
    assert plan.file_edges[0].format.content == "rows"
    assert plan.file_edges[0].format.path is None
    assert [stage.index for stage in plan.stages] == [0, 1]


def test_one_command_is_one_stage() -> None:
    plan = from_commands([_plain_graph()])
    assert plan.edges == ()
    assert [stage.processes for stage in plan.stages] == [("ffmpeg0",)]


def test_a_plan_describes_itself() -> None:
    plan = partition(
        _passthrough_audio_graph(),
        external=external_ids("e0"),
        probes={"a": _video_probe()},
    )
    described = plan.to_dict()

    kinds = [p["kind"] for p in described["processes"]]  # type: ignore[index]
    assert kinds.count("sidecar") == 1
    assert kinds.count("ffmpeg") == 2
    assert all(e["kind"] == "stream" for e in described["edges"])  # type: ignore[index]
    assert len(described["stages"]) == 1  # type: ignore[arg-type]


def test_two_edges_never_leave_one_process_ambiguous() -> None:
    """Every edge names a process the plan holds."""
    plan = partition(_fanout_graph(), external=external_ids("e0", "e1"))
    ids = {p.id for p in plan.processes}
    for edge in plan.edges:
        assert edge.source in ids
        assert edge.target in ids
    assert isinstance(plan.stream_edges[0], StreamEdge)


# ---------------------------------------------------------------- regions


def _region_fanout_graph() -> Graph:
    """One module's frames reach two others through the split lowering inserts."""
    g = Graph(input_paths=["a.mp4"], sources={"a": 0})
    g.nodes["e0"] = Node(
        id="e0", filter="detect", args={}, inputs=["src:a:v:0"], outputs=["video"]
    )
    g.nodes["sp"] = Node(
        id="sp",
        filter="split",
        args={"n": 2},
        inputs=["e0"],
        outputs=["video", "video"],
    )
    g.nodes["e1"] = Node(id="e1", filter="left", args={}, inputs=["sp:0"], outputs=["video"])
    g.nodes["e2"] = Node(id="e2", filter="right", args={}, inputs=["sp:1"], outputs=["video"])
    g.nodes["n0"] = Node(
        id="n0", filter="hstack", args={}, inputs=["e1", "e2"], outputs=["video"]
    )
    g.sinks = [SinkUnit(outputs=[_out("n0")], path="out.mp4")]
    return g


def _detour_graph() -> Graph:
    """A module reading one leg of its own stream back through an ffmpeg filter.

    The one shape a region could not contract -- an ffmpeg process would sit
    inside it -- and the one the lockstep rule refuses before it can arise.
    """
    g = Graph(input_paths=["a.mp4"], sources={"a": 0})
    g.nodes["e0"] = Node(
        id="e0", filter="detect", args={}, inputs=["src:a:v:0"], outputs=["video"]
    )
    g.nodes["sp"] = Node(
        id="sp",
        filter="split",
        args={"n": 2},
        inputs=["e0"],
        outputs=["video", "video"],
    )
    g.nodes["n0"] = Node(id="n0", filter="hue", args={}, inputs=["sp:0"], outputs=["video"])
    g.nodes["e1"] = Node(
        id="e1",
        filter="blur",
        args={},
        inputs=["sp:1", "n0"],
        outputs=["video"],
        reads_annotations=True,
    )
    g.sinks = [SinkUnit(outputs=[_out("e1")], path="out.mp4")]
    return g


def test_a_regions_fan_out_carries_no_split_node() -> None:
    plan = partition(_region_fanout_graph(), external=external_ids("e0", "e1", "e2"))

    assert len(plan.sidecars) == 1
    region = plan.sidecars[0]
    assert region.graph is not None
    # The split is absorbed: three module nodes, and both readers take e0.
    assert list(region.graph.nodes) == ["e0", "e1", "e2"]
    assert region.graph.nodes["e1"].inputs == ["e0"]
    assert region.graph.nodes["e2"].inputs == ["e0"]


def test_a_region_leaving_on_two_edges_is_not_spellable() -> None:
    """Partitioning says what the region's shape is; a module process
    writes one stream, so this plan is one nothing can spawn."""
    plan = partition(_region_fanout_graph(), external=external_ids("e0", "e1", "e2"))
    with pytest.raises(FfrwdError) as caught:
        check_spellable(plan)
    assert caught.value.code is ErrorCode.UNSUPPORTED_SQL
    assert "2 streams leaving it" in caught.value.message
    assert "one COPY per stream" in (caught.value.hint or "")


def test_a_region_leaving_on_one_edge_is_spellable() -> None:
    check_spellable(partition(_series_graph(), external=external_ids("e0", "e1")))


def test_a_regions_fan_out_leaves_the_region_on_two_edges() -> None:
    plan = partition(_region_fanout_graph(), external=external_ids("e0", "e1", "e2"))
    region = plan.sidecars[0]

    assert region.outputs == ("video", "video")
    outgoing = [e.ref for e in plan.stream_edges if e.source == region.id]
    assert sorted(outgoing) == ["e1", "e2"]


def test_a_region_that_would_swallow_an_ffmpeg_process_never_arises() -> None:
    """It needs a module reading two streams, and those two are not lockstep."""
    with pytest.raises(FfrwdError) as caught:
        partition(_detour_graph(), external=external_ids("e0", "e1"))
    assert "the module 'blur'" in caught.value.message


def test_two_adjacent_modules_always_share_a_process() -> None:
    """Which is what leaves no annotation edge for a process boundary to cross."""
    plan = partition(_series_graph(), external=external_ids("e0", "e1"))
    assert len(plan.sidecars) == 1
    assert not any(e.annotations for e in plan.stream_edges)


def test_a_regions_lookahead_sums_along_its_longest_path() -> None:
    shapes = {
        "denoise": ModuleShape(window=3),  # 2 frames
        "grade": ModuleShape(window=5),  # 4 more
    }
    plan = partition(
        _series_graph(), external=external_ids("e0", "e1"), shapes=shapes
    )
    assert plan.sidecars[0].lookahead == 6


def test_a_region_of_windowless_modules_promises_nothing() -> None:
    plan = partition(_series_graph(), external=external_ids("e0", "e1"))
    assert plan.sidecars[0].lookahead == 0


def test_a_regions_latency_is_written_out() -> None:
    plan = partition(
        _series_graph(),
        external=external_ids("e0", "e1"),
        shapes={"grade": ModuleShape(window=4)},
    )
    described = plan.sidecars[0].to_dict()
    assert described["lookahead"] == 3
    assert described["modules"] == [
        {"name": "denoise", "path": "denoise"},
        {"name": "grade", "path": "grade"},
    ]
    assert "graph" in described


def test_a_single_module_region_writes_no_graph() -> None:
    plan = partition(_passthrough_audio_graph(), external=external_ids("e0"))
    described = plan.sidecars[0].to_dict()

    assert not plan.sidecars[0].network
    assert "graph" not in described
    assert described["modules"] == [{"name": "denoise", "path": "denoise"}]


# ---------------------------------------------------------------- lockstep


def _lockstep_graph(*, common: bool) -> Graph:
    """A two-input module, fed either from one split or from two decodes."""
    g = Graph(input_paths=["a.mp4"], sources={"a": 0})
    if common:
        g.nodes["sp"] = Node(
            id="sp",
            filter="split",
            args={"n": 2},
            inputs=["src:a:v:0"],
            outputs=["video", "video"],
        )
        g.nodes["m0"] = Node(
            id="m0", filter="track", args={}, inputs=["sp:0"], outputs=["video"]
        )
        legs = ["m0", "sp:1"]
    else:
        g.nodes["m0"] = Node(
            id="m0", filter="track", args={}, inputs=["src:a:v:0"], outputs=["video"]
        )
        legs = ["m0", "src:a:v:1"]
    g.nodes["m1"] = Node(id="m1", filter="pair", args={}, inputs=legs, outputs=["video"])
    g.sinks = [SinkUnit(outputs=[_out("m1")], path="out.mp4")]
    return g


def test_a_multi_input_module_over_one_point_is_accepted() -> None:
    plan = partition(_lockstep_graph(common=True), external=external_ids("m0", "m1"))
    assert len(plan.sidecars) == 1
    assert plan.sidecars[0].inputs == ("src:a:v:0",)


def test_a_multi_input_module_over_two_streams_is_refused() -> None:
    with pytest.raises(FfrwdError) as caught:
        partition(_lockstep_graph(common=False), external=external_ids("m0", "m1"))

    error = caught.value
    assert error.code is ErrorCode.UNSUPPORTED_SQL
    assert "the module 'pair'" in error.message
    assert "a.video[2]" in error.message
    assert error.hint is not None and "lockstep" not in error.hint


def test_a_module_that_declares_itself_not_one_to_one_breaks_the_lockstep() -> None:
    """Its output no longer pairs with the stream it split from."""
    with pytest.raises(FfrwdError) as caught:
        partition(
            _lockstep_graph(common=True),
            external=external_ids("m0", "m1"),
            shapes={"track": ModuleShape(one_to_one=False)},
        )
    assert "the module 'pair'" in caught.value.message


# ---------------------------------------------------------------- live inputs


LIVE = "srt://camera.local:9000"


def _live_probe(width: int = 640, height: int = 360) -> ProbeResult:
    return ProbeResult(
        streams=[
            StreamMeta(
                type="video",
                index=0,
                metadata={},
                width=width,
                height=height,
                fps="30/1",
                sample_rate=None,
                codec="h264",
            ),
            StreamMeta(
                type="audio",
                index=0,
                metadata={},
                width=None,
                height=None,
                fps=None,
                sample_rate=48000,
                channels=2,
                codec="aac",
            ),
        ],
        duration=None,
    )


def _merge_graph(path: str = LIVE) -> Graph:
    """One source, split two ways: a module on one leg, the picture on the other."""
    g = Graph(input_paths=[path], sources={"a": 0})
    g.nodes["sp"] = Node(
        id="sp",
        filter="split",
        args={"n": 2},
        inputs=["src:a:v:0"],
        outputs=["video", "video"],
    )
    g.nodes["e0"] = Node(
        id="e0", filter="invert", args={}, inputs=["sp:0"], outputs=["video"]
    )
    g.nodes["n0"] = Node(
        id="n0", filter="hstack", args={}, inputs=["sp:1", "e0"], outputs=["video"]
    )
    g.sinks = [SinkUnit(outputs=[_out("n0")], path="out.mp4")]
    return g


def _merged(
    path: str = LIVE,
    *,
    width: int = 640,
    height: int = 360,
    shape: ModuleShape | None = None,
    graph: Graph | None = None,
) -> ProcessPlan:
    return partition(
        graph if graph is not None else _merge_graph(path),
        external=external_ids("e0"),
        probes={"a": _live_probe(width, height)},
        pix_fmts={"invert": "rgba"},
        shapes={"invert": shape or ModuleShape()},
        anchors={"a": (7, 14)},
    )


def _opens(plan: ProcessPlan, path: str) -> list[str]:
    """The ffmpeg processes of `plan` whose own inputs name `path`."""
    return [p.id for p in plan.ffmpeg if path in p.graph.input_paths]


def _edge(plan: ProcessPlan, source: str, target: str) -> StreamEdge:
    return next(e for e in plan.stream_edges if (e.source, e.target) == (source, target))


def test_one_process_reads_a_live_input_however_the_graph_splits() -> None:
    """A socket opens once, so both legs come off one reader's two pipes."""
    plan = _merged()

    assert _opens(plan, LIVE) == ["ffmpeg1"]
    reader = plan.process("ffmpeg1")
    assert isinstance(reader, FfmpegProcess)
    # The split moved into the reader, and each of its pads leaves on a pipe.
    assert list(reader.graph.nodes) == ["sp"]
    assert [unit.path for unit in reader.graph.sinks] == [PIPE, PIPE]
    assert [(e.source, e.target) for e in plan.stream_edges] == [
        ("ffmpeg1", "ffmpeg0"),
        ("sidecar0", "ffmpeg0"),
        ("ffmpeg1", "sidecar0"),
    ]


def test_a_plain_file_still_decodes_once_per_leg() -> None:
    """A file opens as many times as the graph asks, and the plan shows it."""
    plan = _merged("clip.mp4")

    assert _opens(plan, "clip.mp4") == ["ffmpeg1", "ffmpeg2"]
    assert all(edge.bound == 0 and edge.buffer is None for edge in plan.stream_edges)


def test_the_bound_counts_the_frames_the_quick_path_holds() -> None:
    """The module's leg goes through a process; the direct leg waits for it."""
    plan = _merged()

    # One process stands between the reader and the merge, holding the frame it
    # is working on, so the direct edge holds one while that frame is in flight.
    assert {(e.source, e.target): e.bound for e in plan.stream_edges} == {
        ("ffmpeg1", "ffmpeg0"): 1,
        ("sidecar0", "ffmpeg0"): 0,
        ("ffmpeg1", "sidecar0"): 0,
    }


def test_a_declared_window_raises_the_bound_by_what_it_reads_ahead() -> None:
    # 1 for the frame the sidecar holds, 8 more for the window it reads ahead.
    assert _edge(_merged(shape=ModuleShape(window=9)), "ffmpeg1", "ffmpeg0").bound == 9


@pytest.mark.parametrize(
    ("size", "road", "expected"),
    [
        # 640x360 yuv420p is 460800 bytes; two frames round up to 15 whole
        # 64 KiB steps, well under the limit a pipe's own buffer is given.
        ((640, 360), "pipe", (983040, 0)),
        # 1920x1080 is 3110400; two frames is over that limit, so the depth
        # moves into the producing ffmpeg's own fifo queue.
        ((1920, 1080), "fifo", (0, 2)),
    ],
)
def test_the_frame_size_picks_which_road_the_depth_takes(
    size: tuple[int, int], road: str, expected: tuple[int, int]
) -> None:
    plan = _merged(width=size[0], height=size[1])
    buffer = _edge(plan, "ffmpeg1", "ffmpeg0").buffer

    assert buffer is not None
    assert (buffer.road, buffer.frames) == (road, 2)
    assert (buffer.size, buffer.packets) == expected


def _passthrough_graph() -> Graph:
    """A module over the picture, and the sound mapped straight through."""
    g = Graph(input_paths=[LIVE], sources={"a": 0})
    g.nodes["e0"] = Node(
        id="e0", filter="invert", args={}, inputs=["src:a:v:0"], outputs=["video"]
    )
    g.sinks = [
        SinkUnit(outputs=[_out("e0"), _out("src:a:a:0", "audio")], path="out.mkv")
    ]
    return g


def test_a_stream_no_process_filters_crosses_its_pipe_as_a_copy() -> None:
    """The sink process would have opened the socket for the sound. Now it does
    not, and what it takes off the reader is what it was mapping."""
    plan = _merged(graph=_passthrough_graph())

    assert _opens(plan, LIVE) == ["ffmpeg1"]
    sound = next(e for e in plan.stream_edges if e.ref == "src:a:a:0")
    assert (sound.source, sound.target) == ("ffmpeg1", "ffmpeg0")
    assert sound.format.codec == COPY_CODEC


def _rate_changing_graph() -> Graph:
    """The same merge, with a frame rate change on the leg that does not filter."""
    g = _merge_graph()
    nodes = dict(g.nodes)
    nodes["fps"] = Node(
        id="fps", filter="fps", args={"fps": 15}, inputs=["sp:1"], outputs=["video"]
    )
    nodes["n0"].inputs = ["fps", "e0"]
    g.nodes = {name: nodes[name] for name in ("sp", "e0", "fps", "n0")}
    return g


def _stranded_graph() -> Graph:
    """The same merge, with a subtitle track the sink process would have to read."""
    g = _merge_graph()
    g.sinks = [
        SinkUnit(outputs=[_out("n0"), _out("src:a:s:0", "subtitle")], path="out.mkv")
    ]
    return g


@pytest.mark.parametrize(
    ("graph", "shape", "needle"),
    [
        (
            _rate_changing_graph(),
            None,
            "with 'fps' between them handing on a different number of frames",
        ),
        (
            _merge_graph(),
            ModuleShape(one_to_one=False),
            "with 'invert' between them handing on a different number of frames",
        ),
        (
            _stranded_graph(),
            None,
            "a.subtitle[1] is read by one of those others",
        ),
    ],
)
def test_a_live_input_this_compiler_cannot_wire_is_refused(
    graph: Graph, shape: ModuleShape | None, needle: str
) -> None:
    with pytest.raises(FfrwdError) as caught:
        _merged(graph=graph, shape=shape)

    error = caught.value
    assert error.code is ErrorCode.UNBOUNDED_LIVE_INPUT
    assert (error.line, error.col) == (7, 14)  # the input() the anchors named
    assert LIVE in error.message
    assert needle in error.message, error.message
    assert error.hint is not None


def test_a_stream_edge_writes_its_bound_and_the_buffer_it_bought() -> None:
    edges = _merged().to_dict()["edges"]
    assert isinstance(edges, list)
    written = {(e["source"], e["target"]): e for e in edges}

    assert written[("ffmpeg1", "ffmpeg0")]["bound"] == 1
    assert written[("ffmpeg1", "ffmpeg0")]["buffer"] == {
        "road": "pipe",
        "frames": 2,
        "size": 983040,
    }
    # An edge with nothing to hold keeps the smaller shape.
    assert "bound" not in written[("sidecar0", "ffmpeg0")]
    assert "buffer" not in written[("sidecar0", "ffmpeg0")]


def test_which_inputs_can_only_be_opened_once() -> None:
    cases: list[tuple[str, dict[str, object]]] = [
        ("clip.mp4", {}),
        ("/media/clip.mp4", {"realtime": True}),
        ("srt://camera.local:9000", {}),
        ("udp://239.0.0.1:1234", {}),
        ("https://example.com/live.m3u8", {}),
        ("video=Integrated Camera", {"format": "dshow"}),
        ("testsrc2=size=640x360", {"format": "lavfi"}),
        ("data:text/vtt,WEBVTT", {"format": "webvtt"}),
    ]

    assert [is_live(spec, options) for spec, options in cases] == [
        False,
        False,
        True,
        True,
        True,
        True,
        True,
        False,
    ]


def _no_pure_reader_graph() -> Graph:
    """Two output files, and only pipe-reading processes touching the socket.

    The pictures come from a generated source through two modules, so nothing
    reads the socket at depth 0; each output file maps one of its sound tracks
    straight through, and both of those processes also read a pipe.
    """
    g = Graph(input_paths=[LIVE, "testsrc"], sources={"a": 0, "t": 1})
    g.nodes["e0"] = Node(
        id="e0", filter="invert", args={}, inputs=["src:t:v:0"], outputs=["video"]
    )
    g.nodes["e1"] = Node(
        id="e1", filter="invert", args={}, inputs=["e0"], outputs=["video"]
    )
    g.sinks = [
        SinkUnit(outputs=[_out("e0"), _out("src:a:a:0", "audio")], path="one.mkv"),
        SinkUnit(outputs=[_out("e1"), _out("src:a:a:1", "audio")], path="two.mkv"),
    ]
    return g


def test_a_reader_is_made_where_no_process_could_be_the_one() -> None:
    """Every process reading the socket also reads a pipe, so none may be
    joined without risking a cycle: a reader of nothing else is made instead."""
    plan = partition(
        _no_pure_reader_graph(),
        external=external_ids("e0", "e1"),
        probes={"a": _live_probe()},
        anchors={"a": (7, 14)},
    )

    reading = _opens(plan, LIVE)
    assert len(reading) == 1
    reader = plan.process(reading[0])
    assert isinstance(reader, FfmpegProcess)
    # It filters nothing: both tracks are mapped out to pipes as they came.
    assert reader.graph.nodes == {}
    assert [e.ref for e in plan.stream_edges if e.source == reader.id] == [
        "src:a:a:0",
        "src:a:a:1",
    ]
    assert all(
        e.format.codec == COPY_CODEC
        for e in plan.stream_edges
        if e.source == reader.id
    )
