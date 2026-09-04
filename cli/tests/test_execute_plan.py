"""Rendering and wiring a process plan into the argv that runs it.

Unit tier: plans built by hand or by :func:`~ffrwd.processes.partition`, and
nothing spawned. Which transport an edge takes and how each end spells its
pipe are decided before any process exists, so both are testable without one.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Sequence
from dataclasses import replace

import pytest

from ffrwd import pipes
from ffrwd.errors import ErrorCode, FfrwdError
from ffrwd.execute import (
    _CHUNK,
    STDIN,
    STDOUT,
    Flow,
    _End,
    _pipe_buffer,
    _pump,
    _read_ahead,
    overflow_error,
    overflowed,
    plan_argv,
    wires,
)
from ffrwd.ir import Graph, Node, Output, RowsSink, SinkUnit, StreamType
from ffrwd.probe import ProbeResult, StreamMeta
from ffrwd.processes import (
    PIPE,
    AudioFormat,
    EdgeBuffer,
    FfmpegProcess,
    ProcessPlan,
    RowsDocument,
    RowsEdge,
    SidecarProcess,
    StreamEdge,
    VideoFormat,
    external_ids,
    partition,
)


def _out(ref: str, type_: StreamType = "video") -> Output:
    return Output(ref=ref, type=type_, name=None, metadata={})


def _stand_in(
    process: SidecarProcess, reads: Sequence[str] = (), writes: Sequence[str] = ()
) -> list[str]:
    """An ffmpeg standing in for a sidecar, reading the paths it was given."""
    argv = ["ffmpeg"]
    for path in reads or ("pipe:0",):
        argv += ["-f", "nut", "-i", path]
    return [*argv, "-vf", process.module, "-c:v", "rawvideo", "-f", "nut", "pipe:1"]


def _named(edge: StreamEdge, side: str) -> str:
    return f"/pipes/{edge.source}-{edge.target}-{side}"


# ---------------------------------------------------------------- plans


def _chain() -> ProcessPlan:
    """decode -> one external node -> encode: three processes in a line."""
    g = Graph(input_paths=["a.mp4"], sources={"a": 0})
    g.nodes["e0"] = Node(
        id="e0", filter="negate", args={}, inputs=["src:a:v:0"], outputs=["video"]
    )
    g.sinks = [SinkUnit(outputs=[_out("e0")], path="out.mp4")]
    return partition(g, external=external_ids("e0"))


def _fan_in() -> ProcessPlan:
    """One video and one audio stream, each through its own external node."""
    g = Graph(input_paths=["a.mp4"], sources={"a": 0})
    g.nodes["e0"] = Node(
        id="e0", filter="negate", args={}, inputs=["src:a:v:0"], outputs=["video"]
    )
    g.nodes["e1"] = Node(
        id="e1", filter="boost", args={}, inputs=["src:a:a:0"], outputs=["audio"]
    )
    g.sinks = [
        SinkUnit(outputs=[_out("e0"), _out("e1", "audio")], path="out.mkv")
    ]
    return partition(g, external=external_ids("e0", "e1"))


def _two_producers() -> ProcessPlan:
    """Two ffmpeg processes feeding one that muxes -- no sidecar anywhere."""
    video = Graph(input_paths=["a.mp4"], sources={"s": 0})
    video.sinks = [SinkUnit(outputs=[_out("src:s:v:0")], path=PIPE)]
    audio = Graph(input_paths=["a.mp4"], sources={"s": 0})
    audio.sinks = [SinkUnit(outputs=[_out("src:s:a:0", "audio")], path=PIPE)]
    mux = Graph(input_paths=[PIPE, PIPE], sources={"v": 0, "a": 1})
    mux.sinks = [
        SinkUnit(
            outputs=[_out("src:v:v:0"), _out("src:a:a:0", "audio")], path="out.mkv"
        )
    ]
    return ProcessPlan(
        processes=(
            FfmpegProcess(id="video", graph=video),
            FfmpegProcess(id="audio", graph=audio),
            FfmpegProcess(id="mux", graph=mux),
        ),
        edges=(
            StreamEdge(
                source="video", target="mux", ref="src:s:v:0", format=VideoFormat()
            ),
            StreamEdge(
                source="audio", target="mux", ref="src:s:a:0", format=AudioFormat()
            ),
        ),
    )


def _packet_source_plan(tracks: int) -> ProcessPlan:
    """A packet source riding alone, `tracks` outgoing edges to one reader.

    The mirror of a packet sink's several reads: no inputs, and its several
    pads are each their own outgoing :class:`StreamEdge`, catalog order
    (video ahead of the trailing audio track) matching `outputs`.
    """
    kinds = ["video"] * max(tracks - 1, 0) + (["audio"] if tracks else [])
    reader = Graph(input_paths=[PIPE] * tracks, sources={f"t{i}": i for i in range(tracks)})
    reader.sinks = [
        SinkUnit(
            outputs=[
                _out(f"src:t{i}:{'v' if kind == 'video' else 'a'}:0", kind)
                for i, kind in enumerate(kinds)
            ],
            path="out.mkv",
        )
    ]
    return ProcessPlan(
        processes=(
            SidecarProcess(
                id="sidecar0",
                module="relay.wasm",
                node="relay",
                outputs=tuple(kinds),
                packet_source=True,
                # The whole catalog is read here, so every track is an output.
                tracks=tuple(range(tracks)),
            ),
            FfmpegProcess(id="reader", graph=reader),
        ),
        edges=tuple(
            StreamEdge(
                source="sidecar0",
                target="reader",
                ref=f"track{i}",
                format=VideoFormat() if kind == "video" else AudioFormat(),
            )
            for i, kind in enumerate(kinds)
        ),
    )


def _named_by_ref(edge: StreamEdge | RowsEdge, side: str) -> str:
    """`_named`, disambiguated by what the edge carries -- several of a
    packet source's edges share one source and one target, so the pair alone
    will not tell them apart."""
    carried = edge.ref if isinstance(edge, StreamEdge) else edge.alias
    return f"/pipes/{edge.source}-{edge.target}-{carried}-{side}"


def _stand_in_writes(
    process: SidecarProcess, reads: Sequence[str] = (), writes: Sequence[str] = ()
) -> list[str]:
    """A stand-in spelling one write path per entry it is given, the way a
    packet source's real argv spells one ``-f nut`` per track."""
    argv = ["ffmpeg", "-m", process.module]
    for path in writes:
        argv += ["-f", "nut", path]
    return argv


def _shared_feeder() -> ProcessPlan:
    """Two legs of one merge sharing a decode, plus a module on a third leg."""
    g = Graph(input_paths=["a.mp4"], sources={"a": 0})
    g.nodes["sp"] = Node(
        id="sp",
        filter="split",
        args={"n": 3},
        inputs=["src:a:v:0"],
        outputs=["video", "video", "video"],
    )
    g.nodes["n0"] = Node(id="n0", filter="hue", args={}, inputs=["sp:0"], outputs=["video"])
    g.nodes["n1"] = Node(id="n1", filter="gblur", args={}, inputs=["sp:1"], outputs=["video"])
    g.nodes["e0"] = Node(id="e0", filter="seg", args={}, inputs=["sp:2"], outputs=["video"])
    g.nodes["n2"] = Node(
        id="n2", filter="maskedmerge", args={}, inputs=["n0", "n1", "e0"], outputs=["video"]
    )
    g.sinks = [SinkUnit(outputs=[_out("n2")], path="out.mp4")]
    return partition(g, external=external_ids("e0"))


# ---------------------------------------------------------------- transport


def test_a_chain_puts_every_edge_on_stdio() -> None:
    assigned = {
        (wire.edge.source, wire.edge.target): (wire.read_stdio, wire.write_stdio)
        for wire in wires(_chain())
    }
    assert assigned == {
        ("ffmpeg1", "sidecar0"): (True, True),
        ("sidecar0", "ffmpeg0"): (True, True),
    }


def test_fan_in_takes_named_pipes_only_where_it_must() -> None:
    """The reader of two streams has one stdin; its producers still write theirs."""
    assigned = {
        (wire.edge.source, wire.edge.target): (wire.read_stdio, wire.write_stdio)
        for wire in wires(_two_producers())
    }
    assert assigned == {
        ("video", "mux"): (False, True),
        ("audio", "mux"): (False, True),
    }


def test_a_shared_feeders_fan_takes_named_pipes_on_both_ends() -> None:
    """It writes two streams and its reader takes three: no stdio on either side.

    The module's leg still chains where a single stream remains: one decode
    into the sidecar's stdin, the sidecar's stdout being one of the three the
    merge reads.
    """
    plan = _shared_feeder()
    shared = next(
        p.id
        for p in plan.ffmpeg
        if len([e for e in plan.stream_edges if e.source == p.id]) == 2
    )
    merge = next(e.target for e in plan.stream_edges if e.source == shared)
    assigned = {
        (wire.edge.source, wire.edge.target, wire.edge.ref): (
            wire.read_stdio,
            wire.write_stdio,
        )
        for wire in wires(plan)
    }
    feeder = next(e.source for e in plan.stream_edges if e.target == "sidecar0")
    assert assigned == {
        (shared, merge, "n0"): (False, False),
        (shared, merge, "n1"): (False, False),
        (feeder, "sidecar0", "sp:2"): (True, True),
        ("sidecar0", merge, "e0"): (False, True),
    }


def test_a_chained_edge_is_the_one_popen_can_wire() -> None:
    chained = {wire.edge.target: wire.chained for wire in wires(_chain())}
    assert chained == {"sidecar0": True, "ffmpeg0": True}
    assert not any(wire.chained for wire in wires(_two_producers()))


# ---------------------------------------------------------------- rendering


def test_a_chain_spells_its_pipes_as_stdio() -> None:
    argv = plan_argv(_chain(), sidecar_argv=_stand_in)
    assert argv["ffmpeg1"] == [
        "ffmpeg", "-i", "a.mp4",
        "-map", "0:v:0", "-c:0", "rawvideo", "-pix_fmt:0", "yuv420p",
        "-f", "nut", STDOUT,
    ]  # fmt: skip
    assert argv["ffmpeg0"][:5] == ["ffmpeg", "-f", "nut", "-i", STDIN]


def test_a_video_edge_carries_rawvideo_and_an_audio_edge_pcm() -> None:
    argv = plan_argv(_two_producers(), pipe_path=_named)
    assert argv["video"][-7:] == [
        "-c:0", "rawvideo", "-pix_fmt:0", "yuv420p", "-f", "nut", STDOUT,
    ]  # fmt: skip
    assert argv["audio"][-5:] == ["-c:0", "pcm_f32le", "-f", "nut", STDOUT]


def test_a_fan_in_reader_names_one_pipe_per_input() -> None:
    argv = plan_argv(_two_producers(), pipe_path=_named)
    assert argv["mux"][:9] == [
        "ffmpeg",
        "-f", "nut", "-i", "/pipes/video-mux-read",
        "-f", "nut", "-i", "/pipes/audio-mux-read",
    ]  # fmt: skip
    # Both producers keep their own stdout: only the reading end fans in.
    assert argv["video"][-1] == STDOUT
    assert argv["audio"][-1] == STDOUT


def test_two_piped_inputs_stay_two_inputs() -> None:
    """Identical ``pipe:`` paths would otherwise be folded onto one ``-i``."""
    argv = plan_argv(_two_producers(), pipe_path=_named)
    assert argv["mux"].count("-i") == 2
    assert "-map" in argv["mux"]
    assert argv["mux"][argv["mux"].index("-map") + 1] == "0:v:0"
    assert argv["mux"][argv["mux"].index("-map", 10) + 1] == "1:a:0"


def test_a_sidecar_between_two_ffmpegs_reads_and_writes_stdio() -> None:
    argv = plan_argv(_fan_in(), sidecar_argv=_stand_in, pipe_path=_named)
    assert argv["sidecar0"] == [
        "ffmpeg", "-f", "nut", "-i", STDIN,
        "-vf", "negate", "-c:v", "rawvideo", "-f", "nut", STDOUT,
    ]  # fmt: skip
    assert argv["sidecar1"][6] == "boost"


# ---------------------------------------------------------------- refusals


def test_a_sidecar_with_no_argv_is_refused() -> None:
    with pytest.raises(FfrwdError) as caught:
        plan_argv(_chain())
    assert caught.value.code is ErrorCode.INTERNAL
    assert "sidecar0" in caught.value.message
    assert caught.value.hint is not None
    assert "sidecar_argv" in caught.value.hint


def test_a_named_pipe_with_no_name_is_refused() -> None:
    with pytest.raises(FfrwdError) as caught:
        plan_argv(_two_producers())
    assert caught.value.code is ErrorCode.INTERNAL
    assert caught.value.hint is not None
    assert "pipe_path" in caught.value.hint


def test_a_sidecar_reading_two_streams_is_refused() -> None:
    """Its argv comes from a hook that cannot be told a named pipe's path.

    Its two inputs are two pads of one split, which is what keeps them in
    lockstep -- a module reading streams that are not is a rejection long
    before this one. The split's third pad is read outside the region, so the
    region cannot absorb it and reads two pipes instead of one.
    """
    g = Graph(input_paths=["a.mp4"], sources={"a": 0})
    g.nodes["sp"] = Node(
        id="sp",
        filter="split",
        args={"n": 3},
        inputs=["src:a:v:0"],
        outputs=["video", "video", "video"],
    )
    g.nodes["n0"] = Node(id="n0", filter="hue", args={}, inputs=["sp:2"], outputs=["video"])
    g.nodes["e0"] = Node(
        id="e0",
        filter="overlay",
        args={},
        inputs=["sp:0", "sp:1"],
        outputs=["video"],
    )
    g.sinks = [
        SinkUnit(outputs=[_out("e0")], path="out.mp4"),
        SinkUnit(outputs=[_out("n0")], path="side.mp4"),
    ]
    plan = partition(g, external=external_ids("e0"))
    with pytest.raises(FfrwdError) as caught:
        plan_argv(plan, sidecar_argv=_stand_in, pipe_path=_named)
    assert caught.value.code is ErrorCode.INTERNAL
    assert "sidecar0" in caught.value.message


def test_a_non_source_sidecar_writing_two_streams_still_refuses() -> None:
    """The `streams > 1` exemption is for a packet source only.

    An ordinary module still has one stdout, however many consumers a plan
    wires it to -- the same shape :func:`plan_argv` refuses whether the two
    edges reach the plan through a live relay or, as here, by hand.
    """
    decode = Graph(input_paths=["a.mp4"], sources={"a": 0})
    decode.sinks = [SinkUnit(outputs=[_out("src:a:v:0")], path=PIPE)]
    sink1 = Graph(input_paths=[PIPE], sources={"s": 0})
    sink1.sinks = [SinkUnit(outputs=[_out("src:s:v:0")], path="one.mp4")]
    sink2 = Graph(input_paths=[PIPE], sources={"s": 0})
    sink2.sinks = [SinkUnit(outputs=[_out("src:s:v:0")], path="two.mp4")]
    plan = ProcessPlan(
        processes=(
            FfmpegProcess(id="decode", graph=decode),
            SidecarProcess(id="sidecar0", module="m0.wasm", node="m0"),
            FfmpegProcess(id="sink1", graph=sink1),
            FfmpegProcess(id="sink2", graph=sink2),
        ),
        edges=(
            StreamEdge(source="decode", target="sidecar0", ref="src:a:v:0", format=VideoFormat()),
            StreamEdge(source="sidecar0", target="sink1", ref="m0", format=VideoFormat()),
            StreamEdge(source="sidecar0", target="sink2", ref="m0", format=VideoFormat()),
        ),
    )
    with pytest.raises(FfrwdError) as caught:
        plan_argv(plan, sidecar_argv=_stand_in, pipe_path=_named)
    assert caught.value.code is ErrorCode.INTERNAL
    assert caught.value.message == (
        "process 'sidecar0' writes 2 streams, but only its own stdout is wired"
    )


# ---------------------------------------------------------------- packet sources


def test_a_packet_sources_tracks_each_take_their_own_named_pipe() -> None:
    """No longer refused, and each of a three-track source's edges gets its
    own pipe -- the same paths its reader's ``-i`` list names."""
    plan = _packet_source_plan(3)
    argv = plan_argv(plan, sidecar_argv=_stand_in_writes, pipe_path=_named_by_ref)
    edges = [e for e in plan.stream_edges if e.source == "sidecar0"]
    assert len(edges) == 3

    written = [_named_by_ref(edge, "write") for edge in edges]
    assert len(set(written)) == 3  # three distinct pipes, not one reused
    assert argv["sidecar0"] == [
        "ffmpeg", "-m", "relay.wasm",
        "-f", "nut", written[0],
        "-f", "nut", written[1],
        "-f", "nut", written[2],
    ]  # fmt: skip

    read = [_named_by_ref(edge, "read") for edge in edges]
    seen = [argv["reader"][i + 1] for i, tok in enumerate(argv["reader"]) if tok == "-i"]
    assert seen == read


def test_a_packet_sources_writes_put_its_tracks_before_its_rows_document() -> None:
    """Order matters: a packet source's tracks lead `writes`, its rows
    documents follow -- the same "streams first" order
    :func:`~ffrwd.wasm._network_args` gives a region's several sinks, so
    :func:`~ffrwd.wasm._split_writes` can peel the tracks back off the front.
    """
    plan = _packet_source_plan(2)
    doc = RowsDocument(sink=RowsSink(container="webvtt", alias="cues"), node="relay")
    docs = Graph(input_paths=[PIPE], sources={"cues": 0})
    docs.sinks = [SinkUnit(outputs=[_out("src:cues:s:0", "subtitle")], path="cues.vtt")]
    plan = replace(
        plan,
        processes=(
            *(
                replace(p, rows=(doc,)) if p.id == "sidecar0" else p for p in plan.processes
            ),
            FfmpegProcess(id="docs", graph=docs),
        ),
        edges=(
            *plan.edges,
            RowsEdge(source="sidecar0", target="docs", alias="cues", container="webvtt"),
        ),
    )
    captured: list[str] = []

    def _capture(
        process: SidecarProcess, reads: Sequence[str] = (), writes: Sequence[str] = ()
    ) -> list[str]:
        captured.extend(writes)
        return ["ffmpeg"]

    plan_argv(plan, sidecar_argv=_capture, pipe_path=_named_by_ref)

    tracks = [e for e in plan.stream_edges if e.source == "sidecar0"]
    (rows_edge,) = [e for e in plan.rows_edges if e.source == "sidecar0"]
    assert captured == [
        *(_named_by_ref(edge, "write") for edge in tracks),
        _named_by_ref(rows_edge, "write"),
    ]


# ---------------------------------------------------------------- live inputs


LIVE = "srt://camera.local:9000"


def _live_probe(width: int, height: int) -> ProbeResult:
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
            )
        ],
        duration=None,
    )


def _live_merge(width: int = 640, height: int = 360) -> ProcessPlan:
    """A live source split two ways, the module's leg merged back with the picture."""
    g = Graph(input_paths=[LIVE], sources={"a": 0})
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
    return partition(
        g,
        external=external_ids("e0"),
        probes={"a": _live_probe(width, height)},
        pix_fmts={"invert": "rgba"},
    )


def test_the_one_reader_of_a_live_input_writes_a_pipe_per_consumer() -> None:
    """The whole argv: one -i over the socket, its split, and two outputs.

    The module's pipe is the FIRST output: the sidecar drains it from the
    moment it exists, where the merge drains nothing until the module has
    spoken -- so the module's feed cannot sit behind the merge's.
    """
    plan = _live_merge()
    argv = plan_argv(plan, sidecar_argv=_stand_in, pipe_path=_named)

    assert argv["ffmpeg1"] == [
        "ffmpeg",
        "-i", LIVE,
        "-filter_complex", "[0:v:0]split=2[out0][out1]",
        "-map", "[out0]",
        "-c:0", "rawvideo", "-pix_fmt:0", "rgba", "-f", "nut",
        "/pipes/ffmpeg1-sidecar0-write",
        "-map", "[out1]",
        "-c:0", "rawvideo", "-pix_fmt:0", "yuv420p", "-f", "nut",
        "/pipes/ffmpeg1-ffmpeg0-write",
    ]  # fmt: skip
    assert argv["ffmpeg0"] == [
        "ffmpeg",
        "-f", "nut", "-i", "/pipes/ffmpeg1-ffmpeg0-read",
        "-f", "nut", "-i", "/pipes/sidecar0-ffmpeg0-read",
        "-filter_complex", "[0:v:0][1:v:0]hstack[out0]",
        "-map", "[out0]",
        "out.mp4",
    ]  # fmt: skip
    assert [words.count(LIVE) for words in argv.values()] == [0, 1, 0]


def test_a_depth_too_deep_for_a_pipe_puts_the_fifo_muxer_on_the_edge() -> None:
    """Two 1080p frames outgrow a pipe's buffer, so the queue moves into ffmpeg."""
    argv = plan_argv(_live_merge(1920, 1080), sidecar_argv=_stand_in, pipe_path=_named)
    words = argv["ffmpeg1"]
    at = words.index("/pipes/ffmpeg1-ffmpeg0-write")

    assert words[at - 6 : at] == [
        "-fifo_format", "nut", "-queue_size", "2", "-f", "fifo",
    ]  # fmt: skip
    # The other edge holds nothing, so it is a plain pipe as it always was.
    sidecar = words.index("/pipes/ffmpeg1-sidecar0-write")
    assert words[sidecar - 2 : sidecar] == ["-f", "nut"]


def test_a_bounded_edge_asks_for_a_pipe_sized_from_its_bound() -> None:
    edges = {(e.source, e.target): e for e in _live_merge().stream_edges}

    assert _pipe_buffer(edges[("ffmpeg1", "ffmpeg0")]) == 983040
    assert _pipe_buffer(edges[("sidecar0", "ffmpeg0")]) == pipes.DEFAULT_BUFFER
    # A fifo edge holds its depth inside ffmpeg; its pipe stays ordinary.
    deep = {(e.source, e.target): e for e in _live_merge(1920, 1080).stream_edges}
    assert _pipe_buffer(deep[("ffmpeg1", "ffmpeg0")]) == pipes.DEFAULT_BUFFER


# ---------------------------------------------------------------- overflow


def _flow(name: str, at: float, *, moved: int = 1, writing: bool = False, bound: int = 0) -> Flow:
    edge = StreamEdge(
        source="ffmpeg1",
        target=name,
        ref="src_a_v_0_split:1",
        format=VideoFormat(width=640, height=360),
        bound=bound,
        buffer=EdgeBuffer("pipe", frames=bound * 2, size=983040) if bound else None,
    )
    return Flow(edge=edge, at=at, moved=moved, writing=writing)


def test_a_stage_whose_pipes_have_all_stopped_with_one_waiting_is_full() -> None:
    stuck = _flow("sidecar0", at=0.0, writing=True, bound=1)

    assert overflowed([stuck, _flow("ffmpeg0", at=0.0)], now=31.0, stall=30.0) is stuck


@pytest.mark.parametrize(
    ("flows", "why"),
    [
        ([], "a stage with no pipes at all"),
        ([_flow("ffmpeg0", at=0.0, moved=0, writing=True)], "nothing has crossed yet"),
        ([_flow("ffmpeg0", at=30.0, writing=True)], "one pipe is still moving"),
        ([_flow("ffmpeg0", at=0.0)], "no copy is waiting to hand anything over"),
    ],
)
def test_a_stage_that_is_merely_quiet_is_not_full(flows: list[Flow], why: str) -> None:
    assert overflowed(flows, now=31.0, stall=30.0) is None, why


def test_the_deepest_bound_among_the_waiting_edges_is_the_one_named() -> None:
    shallow = _flow("ffmpeg0", at=0.0, writing=True, bound=1)
    deep = _flow("sidecar0", at=0.0, writing=True, bound=9)

    assert overflowed([shallow, deep], now=31.0, stall=30.0) is deep


def test_a_full_buffer_names_the_edge_and_the_depth_it_was_given() -> None:
    error = overflow_error(_flow("sidecar0", at=0.0, writing=True, bound=1), 30.0)

    assert error.code is ErrorCode.BUFFER_OVERFLOW
    assert "'src_a_v_0_split:1' from ffmpeg1 to sidecar0" in error.message
    assert "sized for the 2 frames the compiler bounded it at" in error.message
    assert "30s" in error.message
    assert error.hint is not None


# ---------------------------------------------------------------- the copy


class _Pieces:
    """A stream handing back one queued piece per read, EOF once it is empty."""

    def __init__(self, pieces: list[bytes]) -> None:
        self.pieces = list(pieces)
        self.drained = threading.Event()

    def read(self, size: int) -> bytes:
        if self.pieces:
            return self.pieces.pop(0)
        self.drained.set()
        return b""


class _Trickle:
    """A stream taking at most `most` bytes per write, the way a pipe does."""

    def __init__(self, most: int) -> None:
        self.most = most
        self.writes: list[bytes] = []

    def write(self, data: bytes) -> int:
        self.writes.append(data[: self.most])
        return len(self.writes[-1])

    def flush(self) -> None:
        pass


class _Held(_End):
    """An end already open, so a copy can be driven without a process."""

    def __init__(self, stream: object) -> None:
        self.stream = stream

    def open(self, deadline: float) -> object:  # type: ignore[override]
        return self.stream

    def close(self) -> None:
        pass


def test_the_copy_writes_each_read_on_as_it_comes() -> None:
    """Every read goes on as its own write, whatever its size.

    Accumulating two of them into a bigger write would wedge a stage where one
    process feeds two paths that meet again: the process that would supply the
    rest is waiting for the frame the held-back bytes finish.
    """
    dest = _Trickle(1 << 16)
    flow = _flow("ffmpeg0", at=0.0, moved=0)

    _pump(_Held(_Pieces([b"one", b"two"])), _Held(dest), math.inf, flow)

    assert dest.writes == [b"one", b"two"]
    assert flow.moved == 6
    assert not flow.writing


def test_the_copy_finishes_a_write_the_other_end_only_partly_took() -> None:
    """An unbuffered write takes what it takes; the rest is written again."""
    dest = _Trickle(2)

    _pump(_Held(_Pieces([b"abcde"])), _Held(dest), math.inf)

    assert dest.writes == [b"ab", b"cd", b"e"]


class _Unhurried:
    """A stream whose writes wait until it is let go."""

    def __init__(self) -> None:
        self.taking = threading.Event()
        self.writes: list[bytes] = []

    def write(self, data: bytes) -> int:
        assert self.taking.wait(10), "the copy was never let go"
        self.writes.append(data)
        return len(data)

    def flush(self) -> None:
        pass


def test_a_bounded_edge_is_read_on_while_the_consumer_is_not_taking() -> None:
    """The producer on such an edge is never the one waiting.

    An edge the compiler gave a depth is the direct leg of a live source split
    two ways. The producer has to be able to run to the end of its stream and
    close, whatever the consumer is doing: the process on the SLOWER leg only
    hands on the frames its own pipeline holds when it sees the end of its
    input, and the consumer is waiting for exactly those.
    """
    source = _Pieces([b"one", b"two", b"three"])
    dest = _Unhurried()
    flow = _flow("ffmpeg0", at=0.0, moved=0, bound=1)
    copy = threading.Thread(
        target=_pump, args=(_Held(source), _Held(dest), math.inf, flow), daemon=True
    )

    copy.start()
    read_on = source.drained.wait(10)
    dest.taking.set()
    copy.join(10)

    assert read_on, "the copy stopped reading while the consumer was not taking"
    assert dest.writes == [b"one", b"two", b"three"]
    assert flow.moved == 11


def test_an_edge_with_no_depth_hands_each_read_straight_on() -> None:
    """Only an edge the compiler bounded reads ahead; the rest cost nothing."""
    assert _read_ahead(None) == _CHUNK
    assert _read_ahead(_flow("ffmpeg0", at=0.0)) == _CHUNK
    assert _read_ahead(_flow("ffmpeg0", at=0.0, bound=1)) > _CHUNK
