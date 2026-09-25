"""Tests for feeder arguments: a stream a frame module reads itself, over a
loopback connection the host delivers, instead of being handed it as a pad.

Bare-machine, the way tests/test_wasm.py is: every module is a synthetic
:class:`~ffrwd.wasm.Described`, every input path is one nobody has, and no
sidecar or ffmpeg is spawned. The port a compile picks is the operating
system's, so every test here picks from a counter instead. Running a real
feeder is tests/exec/test_exec_feeders.py's.
"""

from __future__ import annotations

import functools
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from ffrwd import wasm
from ffrwd.compiler import compile_all
from ffrwd.errors import ErrorCode, FfrwdError
from ffrwd.execute import plan_argv, render_plan
from ffrwd.ir import FeederCall, Graph
from ffrwd.parser import parse, resolve
from ffrwd.probe import ProbeResult, StreamMeta
from ffrwd.processes import FfmpegProcess, ProcessPlan
from ffrwd.registry import Registry, load_reference
from ffrwd.split import insert_splits
from ffrwd.wasm import WORLDS, Described, Feeder

execute = sys.modules["ffrwd.execute"]
lower = sys.modules["ffrwd.lower"]

SNAPSHOT_PATH = Path(__file__).resolve().parent / "data" / "reference_registry.json"

PROBE = "modules/feed_probe.wasm"
VIDEO = "modules/switch_video.wasm"
AUDIO = "modules/switch_audio.wasm"
PAIR = "modules/pair.wasm"

_DECLARATIONS = {
    "probe": "CREATE FUNCTION probe(v video_stream, feed video_stream DEFAULT NULL, "
    "port number DEFAULT 9000, lead number DEFAULT 0.3, tag text DEFAULT 'x') "
    f"RETURNS video_stream AS '{PROBE}', 'feed-probe' LANGUAGE wasm;",
    "video": "CREATE FUNCTION video(v video_stream, feed video_stream DEFAULT NULL, "
    "port number DEFAULT 9000, lead number DEFAULT 0.3) "
    f"RETURNS video_stream AS '{VIDEO}', 'video' LANGUAGE wasm;",
    "audio": "CREATE FUNCTION audio(a audio_stream, feed audio_stream DEFAULT NULL, "
    f"port number DEFAULT 9000) RETURNS audio_stream AS '{AUDIO}', 'audio' LANGUAGE wasm;",
}


def _probe_module(*feeders: Feeder, port_type: str = "integer") -> Described:
    return Described(
        world=WORLDS[-1],
        name="feed-probe",
        version="0.1.0",
        params_schema={
            "type": "object",
            "properties": {
                "port": {"type": port_type},
                "lead": {"type": "number"},
                "tag": {"type": "string"},
            },
        },
        pixel_formats=("yuv420p",),
        windowed=True,
        pure=False,
        tcp=True,
        feeders=feeders or (Feeder(input=1, port_param="port", kind="video"),),
    )


_MODULES = {
    PROBE: _probe_module(),
    VIDEO: Described(
        world=WORLDS[-1],
        name="video",
        params_schema={
            "type": "object",
            "properties": {"port": {"type": "integer"}, "lead": {"type": "number"}},
        },
        pixel_formats=("yuv420p",),
        windowed=True,
        pure=False,
        tcp=True,
        feeders=(Feeder(input=1, port_param="port", kind="video", group="switch"),),
    ),
    AUDIO: Described(
        world=WORLDS[-1],
        name="audio",
        params_schema={"type": "object", "properties": {"port": {"type": "integer"}}},
        sample_formats=("s16",),
        sample_rates=(48000,),
        channel_counts=(2,),
        windowed=True,
        window=1024,
        stride=1024,
        pure=False,
        tcp=True,
        feeders=(Feeder(input=1, port_param="port", kind="audio", group="switch"),),
    ),
}

_FROM = " FROM input('prog.mp4') p, input('ad.mp4') a) TO 'out.mp4'"


@pytest.fixture(autouse=True)
def _ports(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Ports picked from 50000 up, one compile at a time."""
    picked = iter(range(50000, 60000))
    monkeypatch.setattr(lower, "free_loopback_port", lambda: next(picked))
    yield


@functools.cache
def _registry() -> Registry:
    return load_reference(SNAPSHOT_PATH)


def _declared(query: str) -> str:
    called = [text for name, text in _DECLARATIONS.items() if f"{name}(" in query]
    return "\n".join([*called, query])


def _video(width: int, height: int, fps: str) -> StreamMeta:
    return StreamMeta(
        type="video", index=0, metadata={}, width=width, height=height, fps=fps,
        sample_rate=None, codec="h264",
    )


def _audio(rate: int, channels: int) -> StreamMeta:
    return StreamMeta(
        type="audio", index=0, metadata={}, width=None, height=None, fps=None,
        sample_rate=rate, codec="aac", channels=channels,
    )


def _probes() -> dict[str, ProbeResult | None]:
    """A 640x360 programme at 25 fps with 44.1 kHz stereo, and a smaller ad."""
    return {
        "p": ProbeResult(streams=[_video(640, 360, "25/1"), _audio(44100, 2)]),
        "a": ProbeResult(streams=[_video(320, 240, "30/1"), _audio(48000, 1)]),
    }


def _lowered(
    query: str,
    modules: dict[str, Described] | None = None,
    probes: dict[str, ProbeResult | None] | None = None,
) -> Graph:
    return insert_splits(
        lower.lower(
            resolve(parse(_declared(query))),
            _probes() if probes is None else probes,
            registry=_registry(),
            describes=modules or _MODULES,
        )
    )


def _refused(query: str, modules: dict[str, Described] | None = None) -> FfrwdError:
    with pytest.raises(FfrwdError) as caught:
        _lowered(query, modules)
    return caught.value


def _plan(query: str, modules: dict[str, Described] | None = None) -> ProcessPlan:
    found = modules or _MODULES
    plan = compile_all(_declared(query), describe=lambda path: found[path]).plan
    assert plan is not None
    return plan


def _module_nodes(graph: Graph, module: str) -> list[str]:
    return [name for name, node in graph.nodes.items() if node.filter == module]


def _chain(graph: Graph, ref: str) -> list[tuple[str, dict[str, object]]]:
    """The filters from `ref` back to the input it reads."""
    found: list[tuple[str, dict[str, object]]] = []
    while ref in graph.nodes:
        node = graph.nodes[ref]
        found.append((node.filter, node.args))
        ref = node.inputs[0]
    return found[::-1]


# -- a stream in the feeder's place --------------------------------------------


def test_a_stream_feeder_is_no_pad_and_is_delivered_on_the_port_the_host_picked() -> None:
    graph = _lowered("COPY (SELECT probe(p.video[1], a.video[1])" + _FROM)
    (name,) = _module_nodes(graph, PROBE)
    node = graph.nodes[name]
    assert node.inputs == ["src:p:v:0"]
    assert node.args == {"port": 50000, "lead": 0.3, "tag": "x"}
    unit = graph.sinks[-1]
    assert (unit.path, unit.options) == (
        "tcp://127.0.0.1:50000",
        {"video_codec": "rawvideo", "format": "nut"},
    )
    assert graph.feeders == {
        "tcp://127.0.0.1:50000": (FeederCall(node=name, function="probe", param="feed"),)
    }
    assert Graph.from_dict(graph.to_dict()).feeders == graph.feeders
    # Conformed to the programme's probed size and rate, in the module's format.
    (output,) = unit.outputs
    assert _chain(graph, output.ref) == [
        ("scale", {"width": 640, "height": 360}),
        ("fps", {"fps": "25/1"}),
        ("format", {"pix_fmts": "yuv420p"}),
    ]
    assert graph.nodes[_chain_start(graph, output.ref)].inputs == ["src:a:v:0"]


def _chain_start(graph: Graph, ref: str) -> str:
    while graph.nodes[ref].inputs[0] in graph.nodes:
        ref = graph.nodes[ref].inputs[0]
    return ref


def test_an_unprobed_programme_conforms_the_feeder_to_the_format_alone() -> None:
    graph = _lowered("COPY (SELECT probe(p.video[1], a.video[1])" + _FROM, probes={})
    (output,) = graph.sinks[-1].outputs
    assert _chain(graph, output.ref) == [("format", {"pix_fmts": "yuv420p"})]


def test_the_values_after_a_stream_feeder_fill_from_the_port_on() -> None:
    """The port is the host's; a positional after the stream is the port's."""
    graph = _lowered("COPY (SELECT probe(p.video[1], a.video[1], lead => 0.5)" + _FROM)
    (name,) = _module_nodes(graph, PROBE)
    assert graph.nodes[name].args == {"port": 50000, "lead": 0.5, "tag": "x"}


@pytest.mark.parametrize(
    "call",
    [
        "probe(p.video[1], a.video[1], 9000)",
        "probe(p.video[1], a.video[1], port => 9000)",
    ],
)
def test_a_port_written_beside_a_stream_feeder_is_refused(call: str) -> None:
    error = _refused(f"COPY (SELECT {call}" + _FROM)
    assert (error.code, error.message) == (
        ErrorCode.UDF_ARG_TYPE,
        "probe() writes 'port', the port its feeder is delivered on, beside a "
        "stream in the feeder's place",
    )


def test_the_plan_starts_the_feeders_writer_alone_and_lists_it() -> None:
    plan = _plan("COPY (SELECT probe(p.video[1], a.video[1])" + _FROM)
    (edge,) = plan.feeder_edges
    writer = plan.process(edge.source)
    assert isinstance(writer, FfmpegProcess)
    assert (edge.target, edge.port, edge.calls[0].param) == ("sidecar0", 50000, "feed")
    argv = plan_argv(plan, sidecar_argv=wasm.shown_argv)
    assert argv[writer.id] == [
        "ffmpeg", "-i", "ad.mp4", "-filter_complex",
        "[0:v:0]format=pix_fmts=yuv420p[out0]", "-map", "[out0]",
        "-c:0", "rawvideo", "-f", "nut", "tcp://127.0.0.1:50000",
    ]
    # The writer and the module run together, with nothing piped between.
    (stage,) = plan.stages
    assert set(stage.processes) == {p.id for p in plan.processes}
    assert all(edge.source != writer.id for edge in plan.stream_edges)
    shown = render_plan(plan, sidecar_argv=wasm.shown_argv)
    assert shown.splitlines()[-2] == (
        f"# feeder: {writer.id} writes tcp://127.0.0.1:50000 for probe(feed) in "
        "sidecar0, started once the port accepts"
    )


def _writer_argv(plan: ProcessPlan) -> list[str]:
    """The argv of the ffmpeg writing the plan's one feeder connection."""
    (source,) = {edge.source for edge in plan.feeder_edges}
    return plan_argv(plan, sidecar_argv=wasm.shown_argv)[source]


def test_sound_carried_as_f32_is_conformed_in_ffmpegs_name_for_it() -> None:
    """The wire says f32 and s16; ffmpeg's aformat reads flt and s16."""
    modules = {**_MODULES, AUDIO: replace(_MODULES[AUDIO], sample_formats=("f32",))}
    argv = _writer_argv(_plan("COPY (SELECT audio(p.audio[1], a.audio[1]) AS s" + _FROM, modules))
    graph = argv[argv.index("-filter_complex") + 1]
    assert "aformat=sample_fmts=flt:sample_rates=48000:channel_layouts=stereo" in graph
    assert argv[argv.index("-c:0") + 1] == "pcm_f32le"


# -- a number in the feeder's place ----------------------------------------------


def test_a_number_in_the_feeders_place_is_the_port_and_wires_nothing() -> None:
    """The old spelling: the positionals after the number carry on past the port."""
    graph = _lowered("COPY (SELECT probe(p.video[1], 9100, 0.5, 'y')" + _FROM)
    (name,) = _module_nodes(graph, PROBE)
    assert graph.nodes[name].args == {"port": 9100, "lead": 0.5, "tag": "y"}
    assert graph.feeders == {}
    assert [unit.path for unit in graph.sinks] == ["out.mp4"]


def test_the_port_spelling_compiles_to_the_argv_it_always_did() -> None:
    """The same call against the declaration from before feeders reads the same."""
    query = "COPY (SELECT probe(p.video[1], 9100, 0.5) FROM input('prog.mp4') p) TO 'out.mp4'"
    old = (
        "CREATE FUNCTION probe(v video_stream, port number DEFAULT 9000, "
        "lead number DEFAULT 0.3, tag text DEFAULT 'x') "
        f"RETURNS video_stream AS '{PROBE}', 'feed-probe' LANGUAGE wasm;\n"
    )
    unfed = replace(_probe_module(), feeders=())
    was = compile_all(old + query, describe=lambda path: unfed).plan
    assert was is not None
    now = render_plan(_plan(query), sidecar_argv=wasm.shown_argv)
    assert now == render_plan(was, sidecar_argv=wasm.shown_argv)
    assert """-params '{"lead": 0.5, "port": 9100, "tag": "x"}'""" in now


def test_a_value_after_the_port_spelling_is_still_checked() -> None:
    error = _refused("COPY (SELECT probe(p.video[1], 9100, 0.5, 'y', 'z')" + _FROM)
    assert (error.code, error.message) == (
        ErrorCode.UDF_ARG_TYPE,
        "probe() got 3 value arguments where it declares 2",
    )
    twice = _refused("COPY (SELECT probe(p.video[1], 9100, port => 9200)" + _FROM)
    assert twice.message == "probe() gets 'port' twice: positionally and by name"


# -- nothing in the feeder's place -----------------------------------------------


@pytest.mark.parametrize("call", ["probe(p.video[1])", "probe(p.video[1], NULL)"])
def test_no_feeder_wires_nothing_and_the_port_keeps_its_default(call: str) -> None:
    graph = _lowered(f"COPY (SELECT {call}" + _FROM)
    (name,) = _module_nodes(graph, PROBE)
    assert graph.nodes[name].args == {"port": 9000, "lead": 0.3, "tag": "x"}
    assert graph.feeders == {}


# -- DEFAULT NULL ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("declaration", "needle"),
    [
        (
            "CREATE FUNCTION f(v video_stream DEFAULT NULL) RETURNS video_stream "
            f"AS '{PROBE}', 'feed-probe' LANGUAGE wasm;",
            "gives the stream parameter 'v' DEFAULT NULL",
        ),
        (
            "CREATE FUNCTION f(v video_stream, w video_stream DEFAULT NULL) "
            f"RETURNS sink AS '{PROBE}', 'feed-probe' LANGUAGE wasm;",
            "gives the stream parameter 'w' DEFAULT NULL",
        ),
        (
            "CREATE FUNCTION f(d data_stream, v video_stream DEFAULT NULL) "
            f"RETURNS data_stream AS '{PROBE}', 'feed-probe' LANGUAGE wasm;",
            "gives the stream parameter 'v' DEFAULT NULL",
        ),
    ],
)
def test_only_a_feeder_may_default_to_null_at_the_declaration(
    declaration: str, needle: str
) -> None:
    with pytest.raises(FfrwdError) as caught:
        resolve(parse(declaration + "\nCOPY (SELECT f(p.video[1]) FROM input('p.mp4') p) "
                      "TO 'out.mp4'"))
    assert caught.value.code is ErrorCode.UNSUPPORTED_SQL
    assert needle in caught.value.message


def test_a_null_default_the_module_reads_as_a_pad_is_refused() -> None:
    pair = Described(
        world=WORLDS[-1], name="pair", pixel_formats=("yuv420p",), windowed=True,
        inputs=2, params_schema={"type": "object", "properties": {}},
    )
    sql = (
        "CREATE FUNCTION pair(v video_stream, w video_stream DEFAULT NULL) "
        f"RETURNS video_stream AS '{PAIR}', 'pair' LANGUAGE wasm;\n"
        "COPY (SELECT pair(p.video[1], a.video[1])" + _FROM
    )
    with pytest.raises(FfrwdError) as caught:
        lower.lower(resolve(parse(sql)), {}, registry=_registry(), describes={PAIR: pair})
    assert caught.value.message == (
        f"function 'pair' defaults 'w' to NULL, and the module '{PAIR}' reads it as a pad"
    )


# -- what the module declares against the declaration ----------------------------


@pytest.mark.parametrize(
    ("module", "needle"),
    [
        (
            _probe_module(Feeder(input=2, port_param="port", kind="video")),
            "reads a feeder at stream argument 2, and function 'probe' declares 2 "
            "stream parameters",
        ),
        (
            _probe_module(Feeder(input=1, port_param="port", kind="audio")),
            "reads a feeder at stream argument 1, reading audio, and function 'probe' "
            "takes 'feed' there as video_stream",
        ),
        (
            _probe_module(Feeder(input=1, port_param="sock", kind="video")),
            "takes the port of its feeder 'feed' in 'sock', and function 'probe' "
            "declares no such parameter",
        ),
        (
            _probe_module(Feeder(input=1, port_param="tag", kind="video")),
            "takes the port of its feeder 'feed' in 'tag', and function 'probe' "
            "declares it as text",
        ),
    ],
)
def test_a_feeder_the_declaration_does_not_match_is_refused(
    module: Described, needle: str
) -> None:
    error = _refused("COPY (SELECT probe(p.video[1], a.video[1])" + _FROM, {PROBE: module})
    assert error.code is ErrorCode.UNSUPPORTED_SQL
    assert needle in error.message


def test_a_feeder_of_the_wrong_kind_is_refused_at_the_call() -> None:
    error = _refused("COPY (SELECT probe(p.video[1], a.audio[1])" + _FROM)
    assert (error.code, error.message) == (
        ErrorCode.UDF_ARG_TYPE,
        "probe() takes 'feed' as one video_stream the module reads itself, or the "
        "port it listens on, and its argument is a audio stream",
    )


# -- groups --------------------------------------------------------------------------


_GROUP = (
    "COPY (SELECT video(p.video[1], a.video[1]) AS v, audio(p.audio[1], a.audio[1]) AS s"
)


def test_one_group_fed_by_one_source_is_one_connection_video_first() -> None:
    graph = _lowered(_GROUP + _FROM)
    (video,) = _module_nodes(graph, VIDEO)
    (audio,) = _module_nodes(graph, AUDIO)
    assert graph.nodes[video].args["port"] == graph.nodes[audio].args["port"] == 50000
    (unit,) = [unit for unit in graph.sinks if unit.path in graph.feeders]
    assert [o.type for o in unit.outputs] == ["video", "audio"]
    assert unit.options == {
        "video_codec": "rawvideo", "audio_codec": "pcm_s16le", "format": "nut",
    }
    # Sound is conformed to what the sound module asks for.
    assert _chain(graph, unit.outputs[1].ref) == [
        ("aformat", {"sample_fmts": "s16", "sample_rates": 48000, "channel_layouts": "stereo"}),
    ]
    assert graph.feeders[unit.path] == (
        FeederCall(node=video, function="video", param="feed"),
        FeederCall(node=audio, function="audio", param="feed"),
    )


def test_one_connection_reaches_both_processes_of_its_group() -> None:
    plan = _plan(_GROUP + _FROM)
    edges = plan.feeder_edges
    assert len({edge.source for edge in edges}) == 1
    assert sorted((edge.target, edge.port) for edge in edges) == [
        ("sidecar0", 50000), ("sidecar1", 50000),
    ]
    (line,) = [
        line for line in render_plan(plan, sidecar_argv=wasm.shown_argv).splitlines()
        if line.startswith("# feeder:")
    ]
    assert line == (
        f"# feeder: {edges[0].source} writes tcp://127.0.0.1:50000 for video(feed) in "
        "sidecar0 and audio(feed) in sidecar1, started once the port accepts"
    )


def test_one_group_fed_by_two_sources_is_refused() -> None:
    error = _refused(
        "COPY (SELECT video(p.video[1], a.video[1]) AS v, audio(p.audio[1], b.audio[1]) "
        "AS s FROM input('prog.mp4') p, input('ad.mp4') a, input('bd.mp4') b) TO 'out.mp4'"
    )
    assert (error.code, error.message) == (
        ErrorCode.UNSUPPORTED_SQL,
        "the feeder 'feed' reads 'b', and another feeder of the group 'switch' reads "
        "'a': the group shares one connection, which carries one source",
    )


def test_feeders_naming_no_group_each_have_a_connection() -> None:
    graph = _lowered(
        "COPY (SELECT probe(p.video[1], a.video[1]) AS x, probe(p.video[1], a.video[1]) "
        "AS y" + _FROM
    )
    ports = sorted(graph.nodes[name].args["port"] for name in _module_nodes(graph, PROBE))
    assert ports == [50000, 50001]
    assert sorted(graph.feeders) == ["tcp://127.0.0.1:50000", "tcp://127.0.0.1:50001"]


# -- the run waits for the port ------------------------------------------------------


class _Running:
    """A member whose process has not ended, or has, with `code`."""

    def __init__(self, code: int | None = None) -> None:
        self.proc = self
        self.code = code

    def poll(self) -> int | None:
        return self.code


def _members(**codes: int | None) -> list[execute._Member]:
    return [
        execute._Member(id=pid, argv=[], proc=cast("subprocess.Popen[bytes]", _Running(code)))
        for pid, code in codes.items()
    ]


def test_the_writer_waits_for_the_port_to_accept() -> None:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]

        def listen_late() -> None:
            time.sleep(0.3)
            listener.listen()

        opener = threading.Thread(target=listen_late)
        opener.start()
        reader = _members(sidecar0=None)
        assert execute._await_port(port, reader, reader, time.monotonic() + 10) is True
        opener.join()


def test_a_port_that_never_opens_fails_the_run_after_the_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(execute, "FEEDER_WAIT", 0.3)
    with socket.socket() as held:
        held.bind(("127.0.0.1", 0))
        port = held.getsockname()[1]
        reader = _members(sidecar0=None)
        assert execute._await_port(port, reader, reader, time.monotonic() + 10) is False
    error = execute.unheard_error("ffmpeg0", ["sidecar0"], port)
    assert error.code is ErrorCode.INPUT_NEVER_OPENED
    assert error.message.startswith(
        f"sidecar0 never listened on tcp://127.0.0.1:{port} for its feeder, so ffmpeg0"
    )


def test_no_writer_is_started_once_its_readers_have_gone() -> None:
    later = time.monotonic() + 10
    gone = _members(sidecar0=0)
    assert execute._await_port(1, gone, gone, later) is None
    running, failed = _members(sidecar0=None, ffmpeg1=1)
    assert execute._await_port(1, [running], [running, failed], later) is None


@pytest.mark.parametrize("code", [0, 1])
def test_a_writer_ends_nothing_once_its_reader_has_gone(code: int) -> None:
    """The stage is done once everything but the writer is, and a writer that
    lost its reader failed for that reason alone."""
    members = _members(sidecar0=0, ffmpeg0=code)
    watched = execute._watch(
        members, time.monotonic() + 5, stall=None, writers={"ffmpeg0": ["sidecar0"]}
    )
    assert watched == (None, False, None)
    running = _members(sidecar0=0, ffmpeg0=None)
    assert execute._watch(
        running, time.monotonic() + 5, stall=None, writers={"ffmpeg0": ["sidecar0"]}
    ) == (None, False, None)


def test_a_writer_failing_while_its_reader_runs_ends_the_stage() -> None:
    members = _members(sidecar0=None, ffmpeg0=1)
    failed, _, _ = execute._watch(
        members, time.monotonic() + 5, stall=None, writers={"ffmpeg0": ["sidecar0"]}
    )
    assert failed == "ffmpeg0"


def test_a_number_where_a_pad_goes_is_refused() -> None:
    """Only a feeder's place takes a port."""
    pair = Described(
        world=WORLDS[-1], name="pair", pixel_formats=("yuv420p",), windowed=True,
        inputs=2, params_schema={"type": "object", "properties": {}},
    )
    sql = (
        "CREATE FUNCTION pair(v video_stream, w video_stream) "
        f"RETURNS video_stream AS '{PAIR}', 'pair' LANGUAGE wasm;\n"
        "COPY (SELECT pair(p.video[1], 9000) FROM input('p.mp4') p) TO 'out.mp4'"
    )
    with pytest.raises(FfrwdError) as caught:
        lower.lower(resolve(parse(sql)), {}, registry=_registry(), describes={PAIR: pair})
    assert (caught.value.code, caught.value.message) == (
        ErrorCode.UDF_ARG_TYPE,
        "pair() takes video_stream as its 'w' argument, got a number",
    )
