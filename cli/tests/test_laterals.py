"""Tests for a run-time lateral: a sql table function over a data stream,
started once per message while the query runs, its streams feeding a feeder.

Bare-machine, the way tests/test_feeders.py is: every module is a synthetic
:class:`~ffrwd.wasm.Described`, every input path is one nobody has, the
ports a compile picks come from a counter, and no sidecar or ffmpeg is
spawned. Instances really run in tests/exec/test_exec_laterals.py.
"""

from __future__ import annotations

import functools
import socket
import sys
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import replace
from pathlib import Path

import pytest

from ffrwd import shapes, wasm
from ffrwd.compiler import compile_all
from ffrwd.errors import ErrorCode, FfrwdError
from ffrwd.execute import render_plan
from ffrwd.ir import FeederCall, Graph, Lateral, LateralConnection, LateralValue, Node
from ffrwd.parser import parse, resolve
from ffrwd.probe import ProbeResult, StreamMeta
from ffrwd.processes import ProcessPlan, SidecarProcess
from ffrwd.registry import Registry, load_reference
from ffrwd.split import insert_splits
from ffrwd.wasm import WORLDS, Described, Feeder

execute = sys.modules["ffrwd.execute"]
lower = sys.modules["ffrwd.lower"]
compiler = sys.modules["ffrwd.compiler"]

SNAPSHOT_PATH = Path(__file__).resolve().parent / "data" / "reference_registry.json"

AUCTION = "modules/auction.wasm"
VIDEO = "modules/switch_video.wasm"
AUDIO = "modules/switch_audio.wasm"
PROBE = "modules/feed_probe.wasm"
PUBLISH = "modules/publish.wasm"

# The body a run-time lateral runs per message: vast's play, reading a file
# where vast reads a VAST tag's media file.
_PLAY = """CREATE FUNCTION play(launch data_stream, url text, start_pts number,
                     duration number, width number, height number, fps number,
                     pix_fmt text, rate number, channels number DEFAULT 2)
RETURNS TABLE(video video_stream, audio audio_stream) AS $$
  SELECT setpts(ffmpeg.format(fps(scale(m.video[1], width, height), fps),
                              pix_fmts => pix_fmt),
                'PTS+' || start_pts::text || '/TB') AS video,
         asetpts(ffmpeg.aformat(aresample(m.audio[1], rate),
                                channel_layouts => channels::text || 'c'),
                 'PTS+' || start_pts::text || '/TB') AS audio,
         STRUCT('1' AS smart_timed) AS tags
  FROM input(url) m
$$ LANGUAGE sql;"""

_DECLARATIONS = {
    "auction": "CREATE FUNCTION auction(d data_stream, clock video_stream, "
    "cohort text DEFAULT 'es-ES', viewers number DEFAULT 0) "
    "RETURNS STRUCT(d data_stream, launch data_stream) "
    f"AS '{AUCTION}', 'auction' LANGUAGE wasm;",
    "video": "CREATE FUNCTION video(v video_stream, feed video_stream DEFAULT NULL, "
    "port number DEFAULT 9000, lead number DEFAULT 0.3) "
    f"RETURNS video_stream AS '{VIDEO}', 'video' LANGUAGE wasm;",
    "audio": "CREATE FUNCTION audio(a audio_stream, feed audio_stream DEFAULT NULL, "
    f"port number DEFAULT 9000) RETURNS audio_stream AS '{AUDIO}', 'audio' LANGUAGE wasm;",
    "probe": "CREATE FUNCTION probe(v video_stream, feed video_stream DEFAULT NULL, "
    f"port number DEFAULT 9000) RETURNS video_stream AS '{PROBE}', 'feed-probe' "
    "LANGUAGE wasm;",
    "publish": "CREATE FUNCTION publish(relay text, name text) RETURNS sink "
    f"AS '{PUBLISH}', 'publish' LANGUAGE wasm;",
    "play": _PLAY,
}

_MODULES = {
    AUCTION: Described(
        world=WORLDS[-1],
        name="auction",
        version="0.1.0",
        params_schema={
            "type": "object",
            "properties": {"cohort": {"type": "string"}, "viewers": {"type": "number"}},
        },
        data_filter=True,
        data_outputs=("json", "json"),
        data_time_base=(1, 1_000_000),
    ),
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
        sample_formats=("f32",),
        sample_rates=(48000,),
        channel_counts=(2,),
        windowed=True,
        window=1024,
        stride=1024,
        pure=False,
        tcp=True,
        feeders=(Feeder(input=1, port_param="port", kind="audio", group="switch"),),
    ),
    PROBE: Described(
        world=WORLDS[-1],
        name="feed-probe",
        params_schema={"type": "object", "properties": {"port": {"type": "integer"}}},
        pixel_formats=("yuv420p",),
        windowed=True,
        pure=False,
        tcp=True,
        feeders=(Feeder(input=1, port_param="port", kind="video"),),
    ),
    PUBLISH: Described(
        world=WORLDS[-1],
        name="publish",
        params_schema={
            "type": "object",
            "properties": {"relay": {"type": "string"}, "name": {"type": "string"}},
        },
        video_codecs=("h264",),
        audio_codecs=("aac",),
        video_streams="many",
        audio_streams="many",
        data_streams="one",
    ),
}


def _stream(kind: str, **values: object) -> StreamMeta:
    fields: dict[str, object] = {
        "type": kind, "index": 0, "metadata": {}, "width": None, "height": None,
        "fps": None, "sample_rate": None, "codec": "h264",
    }
    fields.update(values)
    return StreamMeta(**fields)  # type: ignore[arg-type]


# The programme: a subscribed broadcast's picture, sound and deal track, as a
# file the probe reads; and a file of nothing but launch messages.
_PROBES: dict[str, ProbeResult | None] = {
    "path": ProbeResult(
        streams=[
            _stream("video", width=1280, height=720, fps="25/1"),
            _stream("audio", sample_rate=44100, channels=2, codec="aac"),
            _stream("data", codec="json"),
        ]
    ),
}


def _probe(path: str, *_: object, **__: object) -> ProbeResult | None:
    return _PROBES["path"]


@pytest.fixture(autouse=True)
def _ports(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Ports picked from 50000 up, and every input probed as the programme."""
    picked = iter(range(50000, 60000))
    monkeypatch.setattr(lower, "free_loopback_port", lambda: next(picked))
    monkeypatch.setattr(compiler, "probe_path", _probe)
    yield


@functools.cache
def _registry() -> Registry:
    return load_reference(SNAPSHOT_PATH)


def _declared(query: str) -> str:
    called = [text for name, text in _DECLARATIONS.items() if f"{name}(" in query]
    return "\n".join([*called, query])


def _lowered(query: str) -> Graph:
    return insert_splits(
        lower.lower(
            resolve(parse(_declared(query))),
            {alias: _PROBES["path"] for alias in ("p", "f", "s")},
            registry=_registry(),
            describes=_MODULES,
        )
    )


def _plan(query: str) -> ProcessPlan:
    plan = compile_all(_declared(query), describe=lambda path: _MODULES[path]).plan
    assert plan is not None
    return plan


def _refused(query: str) -> FfrwdError:
    with pytest.raises(FfrwdError) as caught:
        compile_all(_declared(query), describe=lambda path: _MODULES[path])
    return caught.value


# The leaf of a SMART tree: a subscribed programme, this node's auction on its
# deal track, an ad started per launch message, switched in, and the deal
# published on.
_LEAF = """COPY (
  WITH prog AS (SELECT s.video[1] AS v, s.audio[1] AS a, s.data[1] AS d
                FROM input('leaf.nut') s),
       awards AS (SELECT auction(prog.d, prog.v, cohort => 'es-ES').d AS d,
                         auction(prog.d, prog.v, cohort => 'es-ES').launch AS launch
                  FROM prog),
       ads AS (SELECT ad.video, ad.audio FROM awards, LATERAL play(awards.launch) ad)
  SELECT video(prog.v, ads.video), audio(prog.a, ads.audio), awards.d AS deal
  FROM prog, ads, awards
) TO publish('https://relay.example', 'leaf')"""


# -- typing and the plan ------------------------------------------------------


def test_a_lateral_over_a_data_stream_is_listed_as_its_own_block() -> None:
    """Nothing of the body is lowered: the data stream is written to the host,
    and the listing says what each message starts."""
    plan = _plan(_LEAF)
    (lateral,) = plan.laterals
    shown = render_plan(plan, sidecar_argv=wasm.shown_argv).splitlines()
    tap = next(line for line in shown if line.endswith("-f data tcp://127.0.0.1:50002"))
    assert tap.startswith("1. ffmpeg: ffmpeg -copyts -f nut")
    assert "-map 0:d:0 -c:0 copy" in tap
    at = shown.index(next(line for line in shown if line.startswith("# run-time:")))
    assert shown[at:-1] == [
        "# run-time: for each message of awards.launch (ffmpeg0 writes it to "
        "tcp://127.0.0.1:50002 for the host), play",
        "#   feeds video(feed) in sidecar1 and audio(feed) in sidecar2 at "
        "tcp://127.0.0.1:50000",
        "#   binds by name, from each message: url, start_pts, duration, "
        "width or 1280, height or 720, fps or 25, pix_fmt or yuv420p, rate or 48000, "
        "channels or 2",
        "#   template:",
        "#     COPY (SELECT ad.video, ad.audio FROM play(NULL, url => <url>, "
        "start_pts => <start_pts>, duration => <duration>, width => <width>, "
        "height => <height>, fps => <fps>, pix_fmt => <pix_fmt>, rate => <rate>, "
        "channels => <channels>) ad) TO 'tcp://127.0.0.1:50000' WITH (format 'nut', "
        "video_codec 'rawvideo', pix_fmt 'yuv420p', audio_codec 'pcm_f32le')",
        "#   one instance at a time; a message that would overlap the running one "
        "is refused with a row",
    ]
    assert lateral.writer == "ffmpeg0" and lateral.definitions == _PLAY
    # Its writer starts once the switch listens, and joins the switch's stage.
    assert {(edge.source, edge.target, edge.port) for edge in plan.feeder_edges} == {
        ("ffmpeg0", "sidecar1", 50000),
        ("ffmpeg0", "sidecar2", 50000),
    }
    assert not any(line.startswith("# feeder:") for line in shown)


def test_the_launch_messages_a_lateral_reads_can_be_published_too() -> None:
    """The auction writes its launch output once: an ffmpeg copies it both to
    the host, which starts an instance per message, and to the publisher,
    which carries it on to the next node down the tree."""
    publisher = replace(_MODULES[PUBLISH], data_streams="many")
    query = _LEAF.replace("awards.d AS deal", "awards.d AS deal, awards.launch AS launch")
    plan = compile_all(
        _declared(query), describe=lambda path: {**_MODULES, PUBLISH: publisher}[path]
    ).plan
    assert plan is not None
    auction = next(s for s in plan.sidecars if s.module == AUCTION)
    sink = next(s for s in plan.sidecars if s.packet_sink)
    (lateral,) = plan.laterals
    launch = next(e for e in plan.stream_edges if e.source == auction.id and e.ref.endswith(":1"))
    readers = {e.target for e in plan.stream_edges if e.source == launch.target}
    assert readers == {lateral.writer, sink.id}
    assert len([e for e in plan.stream_edges if e.source == auction.id]) == 2


def test_a_lateral_rides_the_graph_and_the_plan_whole() -> None:
    graph = _lowered(_LEAF)
    (lateral,) = graph.laterals
    assert Graph.from_dict(graph.to_dict()).laterals == [lateral]
    video, audio = (
        name for name, node in graph.nodes.items() if node.filter in (VIDEO, AUDIO)
    )
    assert lateral.connections == (
        LateralConnection(
            port=50000,
            calls=(
                FeederCall(node=video, function="video", param="feed"),
                FeederCall(node=audio, function="audio", param="feed"),
            ),
        ),
    )
    (written,) = _plan(_LEAF).to_dict()["laterals"]  # type: ignore[misc]
    assert Lateral.from_dict(written).writer == "ffmpeg0"


# A linear channel's gapless playout: two laterals, A and B, each playing
# into its own switch pair, the pairs cascaded.
_CHANNEL = """COPY (
  WITH prog AS (SELECT s.video[1] AS v, s.audio[1] AS a, s.data[1] AS d
                FROM input('leaf.nut') s)
  SELECT video(video(prog.v, ia.video), ib.video), audio(audio(prog.a, ia.audio), ib.audio)
  FROM prog, LATERAL play(prog.d) ia, LATERAL play(prog.d) ib
) TO 'o.mp4'"""


def test_two_laterals_each_play_into_a_switch_pair_of_their_own() -> None:
    graph = _lowered(_CHANNEL)
    first, second = graph.laterals
    (one,) = first.connections
    (two,) = second.connections
    assert one.port != two.port and abs(one.port - two.port) >= 2
    for connection in (one, two):
        assert sorted(call.function for call in connection.calls) == ["audio", "video"]
    # Each switch reads its own lateral's port; the sound pairs with the
    # picture of the same lane.
    by_node = {
        call.node: connection.port for connection in (one, two) for call in connection.calls
    }
    for node, port in by_node.items():
        assert graph.nodes[node].args["port"] == port
    plan = _plan(_CHANNEL)
    assert len(plan.laterals) == 2
    ports = {edge.port for edge in plan.feeder_edges}
    assert ports == {c.port for lateral in plan.laterals for c in lateral.connections}
    assert len(ports) == 2
    for port in ports:
        assert len([edge for edge in plan.feeder_edges if edge.port == port]) == 2


def test_one_instance_is_the_body_inlined_its_tags_with_it() -> None:
    """NULL in the data stream's place and every value written: one instance,
    what the host compiles per message, its tags on what it writes."""
    graph = _lowered(
        "COPY (SELECT ad.video, ad.audio FROM play(NULL, url => 'ad.mp4', "
        "start_pts => 2.5, duration => 1, width => 640, height => 360, fps => 25, "
        "pix_fmt => 'yuv420p', rate => 48000) ad) TO 'tcp://127.0.0.1:9000' "
        "WITH (format 'nut')"
    )
    (unit,) = graph.sinks
    assert unit.tags == {"smart_timed": "1"}
    filters = {node.filter: node.args for node in graph.nodes.values()}
    assert filters["setpts"] == {"expr": "PTS+2.5/TB"}
    assert filters["scale"] == {"width": 640, "height": 360}
    # channels was left to its DEFAULT.
    assert filters["aformat"] == {"channel_layouts": "2c"}
    assert graph.laterals == []


def test_named_arguments_reach_a_sql_functions_parameters() -> None:
    graph = _lowered(
        "COPY (SELECT ad.video FROM play(NULL, 'ad.mp4', 1, 1, 320, 240, 25, "
        "pix_fmt => 'rgba', rate => 44100, channels => 1) ad) TO 'o.nut'"
    )
    filters = {node.filter: node.args for node in graph.nodes.values()}
    assert filters["format"] == {"pix_fmts": "rgba"}


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (
            "play(NULL, 'ad.mp4', nope => 1)",
            "play() has no parameter 'nope'",
        ),
        (
            "play(NULL, 'ad.mp4', url => 'b.mp4')",
            "play() gets 'url' twice: positionally and by name",
        ),
        (
            "play(NULL, url => 'ad.mp4', 1)",
            "positional arguments must come before named arguments",
        ),
        (
            "play(NULL, url => 'ad.mp4', start_pts => 1)",
            "play() leaves its parameter 'duration' unwritten, which has no DEFAULT",
        ),
    ],
)
def test_a_named_argument_is_refused_at_the_call(call: str, message: str) -> None:
    query = f"COPY (SELECT ad.video\nFROM {call} ad) TO 'o.nut'"
    error = _refused(query)
    at = _declared(query).splitlines().index(f"FROM {call} ad) TO 'o.nut'") + 1
    assert (error.message, error.line) == (message, at)


def test_a_data_stream_parameter_is_refused_a_picture() -> None:
    query = (
        "CREATE FUNCTION pass(d data_stream) RETURNS data_stream AS $$ SELECT d $$ "
        "LANGUAGE sql;\nCOPY (SELECT pass(s.video[1]) FROM input('leaf.nut') s) TO 'o.nut'"
    )
    with pytest.raises(FfrwdError) as caught:
        compile_all(query)
    assert (caught.value.code, caught.value.message) == (
        ErrorCode.UDF_ARG_TYPE,
        "pass() takes data_stream as its 'd' argument, got a video stream",
    )
    assert caught.value.line == 2


_ADS = "COPY (SELECT {select} FROM input('leaf.nut') s, LATERAL play({stream}) ad) TO 'o.mp4'"


@pytest.mark.parametrize(
    ("query", "code", "message"),
    [
        (
            _ADS.format(select="scale(ad.video, 320, 240)", stream="s.data[1]"),
            ErrorCode.UNSUPPORTED_SQL,
            "'ad.video' comes from LATERAL play(s.data[1]), which starts a source per "
            "row of a data stream: it is empty between rows, and scale reads a frame "
            "on every tick. A run-time lateral's streams can only go to a module "
            "input declared 'feeder', or a node's input held on a port",
        ),
        (
            _ADS.format(select="ad.audio", stream="s.data[1]"),
            ErrorCode.UNSUPPORTED_SQL,
            "'ad.audio' comes from LATERAL play(s.data[1]), which starts a source per "
            "row of a data stream: it is empty between rows, and a COPY reads a frame "
            "on every tick. A run-time lateral's streams can only go to a module "
            "input declared 'feeder', or a node's input held on a port",
        ),
        (
            _ADS.format(select="probe(ad.video)", stream="s.data[1]"),
            ErrorCode.UNSUPPORTED_SQL,
            "'ad.video' comes from LATERAL play(s.data[1]), which starts a source per "
            "row of a data stream: it is empty between rows, and probe reads a frame "
            "on every tick. A run-time lateral's streams can only go to a module "
            "input declared 'feeder', or a node's input held on a port",
        ),
        (
            _ADS.format(select="probe(s.video[1], ad.video)", stream="s.video[1]"),
            ErrorCode.UDF_ARG_TYPE,
            "play() takes data_stream as its 'launch' argument, got a video stream",
        ),
        (
            "COPY (SELECT probe(s.video[1], ad.video) FROM input('leaf.nut') s, "
            "play('x') ad) TO 'o.mp4'",
            ErrorCode.UDF_ARG_TYPE,
            "play() takes data_stream as its 'launch' argument, got a string",
        ),
        (
            _ADS.format(select="probe(s.video[1], ad.video)",
                        stream="s.data[1], url => s.tags.title"),
            ErrorCode.UDF_ARG_TYPE,
            "play() is started once per message, and its 'url' argument reads "
            "'s.tags.title': a value written in the call is the same for every "
            "message",
        ),
        (
            "COPY (SELECT video(s.video[1], one.video), audio(s.audio[1], two.audio) "
            "FROM input('leaf.nut') s, LATERAL play(s.data[1]) one, "
            "LATERAL play(s.data[1]) two) TO 'o.mp4'",
            ErrorCode.UNSUPPORTED_SQL,
            "the group 'switch' reads 'one (play#1)' in video(feed) and "
            "'two (play#2)' in audio(feed): "
            "each source a group reads is a connection of its own, and each has to "
            "reach the same calls",
        ),
    ],
)
def test_a_run_time_lateral_is_refused_by_what_it_gets_wrong(
    query: str, code: ErrorCode, message: str
) -> None:
    error = _refused(query)
    assert (error.code, error.message) == (code, message)


@pytest.mark.parametrize(
    ("declaration", "message"),
    [
        (
            "CREATE FUNCTION play(launch data_stream, url text) RETURNS TABLE(d "
            "data_stream) AS $$ SELECT m.data[1] AS d FROM input(url) m $$ LANGUAGE sql;",
            "play() is started once per message of its data stream, and returns "
            "'d data_stream': it returns the picture and sound a feeder takes, one "
            "of each at most",
        ),
        (
            "CREATE FUNCTION play(launch data_stream, v video_stream) RETURNS "
            "TABLE(video video_stream) AS $$ SELECT v AS video $$ LANGUAGE sql;",
            "play() is started once per message of its data stream, and takes 'v' "
            "as video_stream: every parameter after the data stream is a value "
            "bound per message",
        ),
        (
            "CREATE FUNCTION play(launch data_stream, url text) RETURNS "
            "TABLE(video video_stream) AS $$ SELECT m.video[1] AS video FROM "
            "input(url) m WHERE launch IS NULL $$ LANGUAGE sql;",
            "the body of play() reads 'launch', the data stream it is started from",
        ),
    ],
)
def test_a_run_time_laterals_declaration_is_refused_at_the_call(
    declaration: str, message: str
) -> None:
    with pytest.raises(FfrwdError) as caught:
        compile_all(
            declaration + "\n" + _DECLARATIONS["probe"] + "\nCOPY (SELECT "
            "probe(s.video[1], ad.video) FROM input('leaf.nut') s, "
            "LATERAL play(s.data[1]) ad) TO 'o.mp4'",
            describe=lambda path: _MODULES[path],
        )
    assert caught.value.message == message


def test_a_body_that_cannot_compile_is_refused_before_the_run() -> None:
    query = _ADS.format(select="probe(s.video[1], ad.video)", stream="s.data[1]")
    broken = _declared(query).replace("FROM input(url) m", "FROM input(url) m, nowhere n")
    with pytest.raises(FfrwdError) as caught:
        compile_all(broken, describe=lambda path: _MODULES[path])
    assert caught.value.message.startswith(
        "an instance of play(s.data[1]) does not compile: "
    )
    assert caught.value.line == broken.splitlines().index(
        next(line for line in broken.splitlines() if line.startswith("COPY"))
    ) + 1


# -- run time -------------------------------------------------------------------


_LATERAL = Lateral(
    function="play",
    call="play(s.data[1])",
    stream="s.data[1]",
    tap=0,
    template="COPY (SELECT ad.video FROM play(NULL, url => :'url', "
    "start_pts => :start_pts, width => :width, loud => :loud, "
    "channels => :channels) ad) TO 'tcp://127.0.0.1:9000' WITH (format 'nut')",
    values=(
        LateralValue("url", "text"),
        LateralValue("start_pts", "number"),
        LateralValue("width", "number", shape=640),
        LateralValue("loud", "boolean", default=True),
        LateralValue("channels", "number", shape=2, default=True),
    ),
    connections=(LateralConnection(port=9000, calls=()),),
)


@pytest.mark.parametrize(
    ("message", "bound"),
    [
        (
            {"url": "ad.mp4", "start_pts": 2.5, "loud": True, "extra": [1]},
            ({"url": "ad.mp4", "start_pts": "2.5", "width": "640", "loud": "true",
              "channels": "2"}, None),
        ),
        (
            {"url": "ad.mp4", "start_pts": 3, "width": 1280, "channels": 1},
            ({"url": "ad.mp4", "start_pts": "3", "width": "1280", "channels": "1"}, None),
        ),
        (
            {"url": "ad.mp4", "start_pts": "3"},
            ({}, "'start_pts' is number, and the message's 'start_pts' is a string"),
        ),
        (
            {"url": "ad.mp4", "start_pts": 3, "loud": 1},
            ({}, "'loud' is boolean, and the message's 'loud' is a number"),
        ),
        (
            {"start_pts": 3},
            ({}, "nothing binds 'url': the message has no field of that name, and it "
             "has no DEFAULT"),
        ),
    ],
)
def test_a_message_binds_by_name_then_the_feeder_then_the_default(
    message: Mapping[str, object], bound: tuple[dict[str, str], str | None]
) -> None:
    assert execute._bind(_LATERAL, message) == bound


def test_messages_are_read_whole_however_the_bytes_arrive() -> None:
    """One object after another, a heartbeat's blank payload between them."""
    messages = execute._Messages()
    assert messages.feed(b' {"a": 1}{"b": "\xc3') == [{"a": 1}]
    assert messages.feed(b'\xa9"} \n') == [{"b": "é"}]
    assert messages.feed(b"[1]{") == [[1]]
    assert messages.rest() == "{"


def _until(done: Callable[[], bool]) -> None:
    """Wait up to ten seconds for `done`."""
    deadline = time.monotonic() + 10
    while not done() and time.monotonic() < deadline:
        time.sleep(0.01)


def test_each_message_starts_one_instance_at_a_time() -> None:
    """The loop without a process: the compile of the first instance is held
    while the rest arrive, and every message ends with a row."""
    held = threading.Event()
    compiling = threading.Event()
    texts: list[str] = []
    rows: list[Mapping[str, object]] = []

    def compile_instance(text: str, unset: Mapping[tuple[int, int], str]) -> ProcessPlan:
        texts.append(text)
        compiling.set()
        held.wait(10)
        raise FfrwdError(ErrorCode.UNSUPPORTED_SQL, "no such file")

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        tap = probe.getsockname()[1]
    run = execute._LateralRun(
        replace(_LATERAL, tap=tap), compile_instance, None, rows.append, None, None
    )
    try:
        with socket.create_connection(("127.0.0.1", tap)) as writer:
            writer.sendall(b' {"url": "a.mp4", "start_pts": 1.0, "duration": 1.0}')
            assert compiling.wait(10)
            writer.sendall(
                b'{"url": "b.mp4", "start_pts": 1.5, "duration": 1.0} '
                b'{"start_pts": 4.0, "duration": 1.0}'
                b'{"url": "c.mp4", "start_pts": 2.0, "duration": 1.0}'
            )
        _until(lambda: len(rows) == 2)
        held.set()
        _until(lambda: len(rows) == 4)
    finally:
        held.set()
        run.stop()
        run.join()
    assert rows == [
        {"event": "feeder", "row": 2, "start_pts": 1.5,
         "refused": "it plays from 1.5 to 2.5, over the instance of message 1, "
         "from 1.0 to 2.0: one instance at a time"},
        {"event": "feeder", "row": 3, "start_pts": 4.0,
         "refused": "nothing binds 'url': the message has no field of that name, "
         "and it has no DEFAULT"},
        {"event": "feeder", "row": 1, "start_pts": 1.0, "refused": "no such file"},
        {"event": "feeder", "row": 4, "start_pts": 2.0, "refused": "no such file"},
    ]
    assert texts[0] == (
        "COPY (SELECT ad.video FROM play(NULL, url => 'a.mp4', start_pts => 1.0, "
        "width => 640, loud => NULL, channels => 2) ad) TO 'tcp://127.0.0.1:9000' "
        "WITH (format 'nut')"
    )



# Two views whose laterals share an alias, as a SMART tree written as one
# script does: two laterals all the same, each on a connection of its own.
_SAME_ALIAS = """CREATE VIEW prog AS
  SELECT s.video[1] AS v, s.audio[1] AS a, s.data[1] AS d FROM input('leaf.nut') s;
CREATE VIEW one AS
  SELECT video(prog.v, ad.video) AS v, audio(prog.a, ad.audio) AS a
  FROM prog, LATERAL play(prog.d) ad;
CREATE VIEW two AS
  SELECT video(one.v, ad.video) AS v, audio(one.a, ad.audio) AS a
  FROM one, prog, LATERAL play(prog.d) ad;
COPY (SELECT two.v, two.a FROM two) TO 'o.mp4'"""


def test_laterals_sharing_an_alias_each_feed_a_connection_of_their_own() -> None:
    plan = _plan(_SAME_ALIAS)
    assert len(plan.laterals) == 2
    ports = {edge.port for edge in plan.feeder_edges}
    assert len(ports) == 2
    for port in ports:
        assert len([edge for edge in plan.feeder_edges if edge.port == port]) == 2


# -- a node holding its feed on a port ----------------------------------------

SWITCH = "modules/switch_node.wasm"
_SWITCH = (
    "CREATE FUNCTION switch(v video_stream, a audio_stream DEFAULT NULL, "
    "feed video_stream DEFAULT NULL, feed_audio audio_stream DEFAULT NULL, "
    "port number DEFAULT 9000) RETURNS STRUCT(v video_stream, a audio_stream) "
    f"AS '{SWITCH}', 'switch' LANGUAGE wasm;"
)
_HELD = {
    "kind": "hold",
    "anchor": {"kind": "tagged", "tag": "smart_timed"},
    "group": "switch",
    "lead": 0.3,
    "port_param": "port",
}


def _switch_shape(
    module: str, params: str, bound: Sequence[str], grants: Sequence[str] = ()
) -> shapes.NodeShape:
    """ffrwd/switch 0.5.0's shape: the programme, and a feed group held on `port`."""

    def port(name: str, kind: str, pairing: dict[str, object]) -> dict[str, object]:
        return {
            "name": name, "kind": kind, "required": name == "v", "many": False,
            "pairing": pairing, "rows": "ignore", "window": 1, "stride": 1, "accepts": {},
        }

    lockstep: dict[str, object] = {"kind": "lockstep"}
    return shapes.node_shape(
        module,
        {
            "inputs": [
                port("v", "video", lockstep),
                port("a", "audio", lockstep),
                port("feed", "video", _HELD),
                port("feed_audio", "audio", _HELD),
            ],
            "outputs": [
                {"name": "v", "kind": "video", "latency": 0,
                 "format": {"kind": "like", "port": "v"}},
                {"name": "a", "kind": "audio", "latency": 0,
                 "format": {"kind": "like", "port": "a"}},
            ],
            "clock": {"kind": "input", "port": "v"},
            "pure": True,
            "one_to_one": True,
            "bounded": True,
            "relation": [],
        },
    )


def test_a_lateral_feeds_a_nodes_held_inputs_on_one_connection() -> None:
    """The lateral's picture and sound go to the port the host listens on for
    the node's feed group, written into the node's port param, as a feeder's
    connection is."""
    modules = {
        **_MODULES,
        SWITCH: Described(
            world="node-module",
            name="switch",
            params_schema={"type": "object", "properties": {"port": {"type": "integer"}}},
            node=True,
        ),
    }
    query = """COPY (
  WITH prog AS (SELECT s.video[1] AS v, s.audio[1] AS a, s.data[1] AS d
                FROM input('leaf.nut') s),
       awards AS (SELECT auction(prog.d, prog.v).d AS d,
                         auction(prog.d, prog.v).launch AS launch FROM prog),
       sw AS (SELECT (switch(prog.v, prog.a, ad.video, ad.audio)).*
              FROM prog, awards, LATERAL play(awards.launch) ad)
  SELECT sw.v, sw.a, awards.d AS deal FROM sw, awards
) TO publish('https://relay.example', 'leaf')"""
    plan = compile_all(
        _declared(query).replace("COPY (", _SWITCH + "\nCOPY (", 1),
        describe=lambda path: modules[path],
        shape=_switch_shape,
    ).plan
    assert plan is not None
    (lateral,) = plan.laterals
    (connection,) = lateral.connections
    assert [call.param for call in connection.calls] == ["feed", "feed_audio"]
    (switch,) = [
        node
        for process in plan.sidecars
        if process.graph is not None
        for node in process.graph.nodes.values()
        if node.filter == "switch_node"
    ]
    assert switch.args["port"] == connection.port
    assert switch.ports == ["v", "a"]


# -- a node's data output over a source read row by row --------------------------

SUB = "modules/sub_node.wasm"
NODE_AUCTION = "modules/auction_node.wasm"


def _source_and_auction_shape(
    module: str, params: str, bound: Sequence[str], grants: Sequence[str] = ()
) -> shapes.NodeShape:
    """A subscription's one rendition, and an auction node reading its deals."""
    data = {"kind": "data", "codec": "json"}
    if module == NODE_AUCTION:
        def port(name: str, kind: str) -> dict[str, object]:
            pairing = {"kind": "lockstep"} if kind == "video" else {"kind": "arrival"}
            return {
                "name": name, "kind": kind, "required": True, "many": False,
                "pairing": pairing, "rows": "ignore", "window": 1, "stride": 1,
                "accepts": {},
            }

        return shapes.node_shape(module, {
            "inputs": [port("d", "data"), port("clock", "video")],
            "outputs": [
                {"name": "d", "kind": "data", "latency": 0, "format": data},
                {"name": "launch", "kind": "data", "latency": 0, "format": data},
            ],
            "clock": {"kind": "input", "port": "clock"},
            "pure": False, "one_to_one": False, "bounded": False, "relation": [],
        })
    picture = {"kind": "video", "width": 1280, "height": 720, "pix_fmt": "yuv420p"}
    return shapes.node_shape(module, {
        "inputs": [],
        "outputs": [
            {"name": "video", "kind": "video", "latency": 0, "row": 0, "format": picture},
            {"name": "deals", "kind": "data", "latency": 0, "row": 0, "format": data},
        ],
        "clock": {"kind": "self_clocked"},
        "pure": False, "one_to_one": False, "bounded": False,
        "relation": ['{"name": "720p"}'],
    })


def test_a_nodes_data_over_a_source_read_by_row_starts_a_lateral() -> None:
    """The auction runs once over the one row the source reads, so its
    launch column is one stream however the body reads it."""
    modules = {
        **_MODULES,
        SUB: Described(
            world="node-module", name="sub",
            params_schema={"type": "object", "properties": {"relay": {"type": "string"}}},
            node=True,
        ),
        NODE_AUCTION: Described(
            world="node-module", name="auction",
            params_schema={"type": "object", "properties": {}}, node=True,
        ),
    }
    query = (
        f"CREATE FUNCTION sub(relay text) RETURNS source AS '{SUB}', 'sub' LANGUAGE wasm;\n"
        "CREATE FUNCTION auction(d data_stream, clock video_stream) "
        "RETURNS STRUCT(d data_stream, launch data_stream) "
        f"AS '{NODE_AUCTION}', 'auction' LANGUAGE wasm;\n"
        + _DECLARATIONS["play"] + "\n" + _DECLARATIONS["video"] + "\n"
        + """COPY (
  WITH prog AS (SELECT s.video[1] AS v, s.data[1] AS d FROM sub('r') s),
       awards AS (SELECT (auction(prog.d, prog.v)).* FROM prog),
       ads AS (SELECT ad.video FROM awards, LATERAL play(awards.launch) ad)
  SELECT video(prog.v, ads.video), awards.d AS deal FROM prog, ads, awards
) TO 'out.nut'"""
    )
    plan = compile_all(
        query, describe=lambda path: modules[path], shape=_source_and_auction_shape
    ).plan
    assert plan is not None
    (lateral,) = plan.laterals
    assert lateral.stream == "awards.launch"


# -- two laterals on one held many-port ---------------------------------------

COMPOSE = "modules/compose_node.wasm"
_COMPOSE = (
    "CREATE FUNCTION compose(v video_stream, inputs video_stream[] DEFAULT NULL, "
    "port number DEFAULT NULL) RETURNS video_stream "
    f"AS '{COMPOSE}', 'compose' LANGUAGE wasm;"
)
_TWO_ADS = """COPY (
  WITH prog AS (SELECT s.video[1] AS v, s.data[1] AS d FROM input('leaf.nut') s)
  SELECT compose(prog.v, ARRAY[ad.video, lbar.video])
  FROM prog, LATERAL play(prog.d) ad, LATERAL play(prog.d) lbar
) TO 'out.nut'"""


def _compose_shape(
    module: str, params: str, bound: Sequence[str], grants: Sequence[str] = ()
) -> shapes.NodeShape:
    """ffrwd/blitz's compose: the programme, and pictures held on `port` when
    `inputs` is bound, so a call binding none listens on nothing."""
    held = {
        "kind": "hold",
        "anchor": {"kind": "tagged", "tag": "smart_timed"},
        "lead": 0.3,
        "port_param": "port" if "inputs" in bound else None,
    }

    def port(name: str, pairing: dict[str, object], many: bool) -> dict[str, object]:
        return {
            "name": name, "kind": "video", "required": name == "v", "many": many,
            "pairing": pairing, "rows": "ignore", "window": 1, "stride": 1, "accepts": {},
        }

    return shapes.node_shape(module, {
        "inputs": [port("v", {"kind": "lockstep"}, False), port("inputs", held, True)],
        "outputs": [{"name": "v", "kind": "video", "latency": 0,
                     "format": {"kind": "like", "port": "v"}}],
        "clock": {"kind": "input", "port": "v"},
        "pure": True, "one_to_one": True, "bounded": True, "relation": [],
    })


def _compose_modules(port: dict[str, object]) -> dict[str, Described]:
    return {
        **_MODULES,
        COMPOSE: Described(
            world="node-module", name="compose",
            params_schema={"type": "object", "properties": {"port": port}}, node=True,
        ),
    }


def _composed(port: dict[str, object]) -> ProcessPlan:
    plan = compile_all(
        _declared(_TWO_ADS).replace("COPY (", _COMPOSE + "\nCOPY (", 1),
        describe=lambda path: _compose_modules(port)[path],
        shape=_compose_shape,
    ).plan
    assert plan is not None
    return plan


def test_two_laterals_on_one_held_many_port_are_two_connections() -> None:
    """Each lateral is a connection of its own, so a module taking its port
    param as an array is given one port per lateral, in the order written."""
    plan = _composed({"type": "array", "items": {"type": "integer"}})
    assert len(plan.laterals) == 2
    ports = [lateral.connections[0].port for lateral in plan.laterals]
    assert len(set(ports)) == 2
    (compose,) = [
        node
        for process in plan.sidecars
        if process.graph is not None
        for node in process.graph.nodes.values()
        if node.filter == "compose_node"
    ]
    assert compose.args["port"] == ports
    assert compose.ports == ["v"]


def test_two_laterals_on_a_held_many_port_taking_one_port_are_refused() -> None:
    with pytest.raises(FfrwdError) as caught:
        _composed({"type": "integer", "minimum": 1, "maximum": 65535})
    assert caught.value.message.endswith(
        f"hands 'inputs' 2 run-time laterals, each a connection on a port of its own, "
        f"and the module '{COMPOSE}' takes one port in 'port'"
    )


def test_a_lateral_counts_as_bound_where_the_shape_says_which_port_it_holds() -> None:
    """compose names `port` as its feed's port only with `inputs` bound."""
    query = _TWO_ADS.replace("ARRAY[ad.video, lbar.video]", "ARRAY[ad.video]").replace(
        ", LATERAL play(prog.d) lbar", ""
    )
    modules = _compose_modules({"type": "integer"})
    plan = compile_all(
        _declared(query).replace("COPY (", _COMPOSE + "\nCOPY (", 1),
        describe=lambda path: modules[path],
        shape=_compose_shape,
    ).plan
    assert plan is not None
    (lateral,) = plan.laterals
    (compose,) = [
        node
        for process in plan.sidecars
        if process.graph is not None
        for node in process.graph.nodes.values()
        if node.filter == "compose_node"
    ]
    assert compose.args["port"] == lateral.connections[0].port


# compose's own `port`: one port the host may be handed, or the compiler's list.
_ONE_OR_MANY_PORTS: dict[str, object] = {
    "type": ["array", "integer"],
    "items": {"type": "integer", "minimum": 1, "maximum": 65535},
    "minimum": 1,
    "maximum": 65535,
}


def _compose_node(plan: ProcessPlan) -> Node:
    (compose,) = [
        node
        for process in plan.sidecars
        if process.graph is not None
        for node in process.graph.nodes.values()
        if node.filter == "compose_node"
    ]
    return compose


def test_a_port_param_taking_one_or_many_is_handed_a_list_for_two_laterals() -> None:
    plan = _composed(_ONE_OR_MANY_PORTS)
    ports = [lateral.connections[0].port for lateral in plan.laterals]
    assert len(set(ports)) == 2
    assert _compose_node(plan).args["port"] == ports


def test_a_port_written_by_hand_stays_one_integer_where_a_list_also_fits() -> None:
    query = """COPY (
  SELECT compose(s.video[1], port => 9100.0) FROM input('leaf.nut') s
) TO 'out.nut'"""
    modules = _compose_modules(_ONE_OR_MANY_PORTS)
    plan = compile_all(
        _COMPOSE + "\n" + query,
        describe=lambda path: modules[path],
        shape=_compose_shape,
    ).plan
    assert plan is not None
    port = _compose_node(plan).args["port"]
    assert port == 9100 and isinstance(port, int)


# -- a node's data, read by several and by a lateral, with no copying ffmpeg ---

SELL = "modules/sell_node.wasm"
_SOLD = """COPY (
  WITH prog AS (SELECT s.video[1] AS v, s.data[1] AS d FROM input('leaf.nut') s),
       sold AS (SELECT (sell(prog.d, prog.v)).* FROM prog),
       ads AS (SELECT ad.video FROM sold, LATERAL play(sold.launch) ad)
  SELECT video(prog.v, ads.video),
         auction(sold.d, prog.v, cohort => 'es').d AS es,
         auction(sold.d, prog.v, cohort => 'es').launch AS es_launch,
         auction(sold.d, prog.v, cohort => 'fr').d AS fr,
         auction(sold.d, prog.v, cohort => 'fr').launch AS fr_launch,
         sold.d AS deal
  FROM prog, ads, sold
) TO 'out.nut'"""


def test_a_nodes_data_reaches_every_reader_and_the_host_from_its_own_region() -> None:
    """sell's deals go to two auctions and the destination, one output of the
    network apiece, and its launches to the host as NDJSON: no ffmpeg copies a
    message. The same plan had eleven processes, seven of them ffmpeg, with a
    copy per reader and one to the tap."""
    modules = {
        **_MODULES,
        SELL: Described(
            world="node-module", name="sell",
            params_schema={"type": "object", "properties": {}}, node=True,
        ),
    }
    plan = compile_all(
        "CREATE FUNCTION sell(d data_stream, clock video_stream) "
        "RETURNS STRUCT(d data_stream, launch data_stream) "
        f"AS '{SELL}', 'sell' LANGUAGE wasm;\n" + _declared(_SOLD),
        describe=lambda path: modules[path],
        # sell takes deals and a clock and writes deals and launches, as
        # the node auction does.
        shape=lambda module, params, bound, grants=(): _source_and_auction_shape(
            NODE_AUCTION, params, bound, grants
        ),
    ).plan
    assert plan is not None
    ffmpegs = [p for p in plan.processes if not isinstance(p, SidecarProcess)]
    assert (len(plan.processes), len(ffmpegs)) == (9, 5)
    assert all(process.graph.nodes or process.graph.sinks for process in ffmpegs)
    (lateral,) = plan.laterals
    seller = next(
        process for process in plan.sidecars if process.module == SELL
    )
    assert lateral.pipe and lateral.writer == seller.id
    argv = render_plan(plan, sidecar_argv=wasm.shown_argv)
    line = next(one for one in argv.splitlines() if "sell_node=" in one)
    assert line.count("-map '[out2]'") == 3
    assert line.endswith(f"-f ndjson ffrwd:tap:{lateral.tap}")


def test_a_piped_tap_is_read_off_the_pipe_the_relay_hands_the_host(tmp_path: Path) -> None:
    """The host reads the region's NDJSON as it reads the messages off a port."""
    rows: list[Mapping[str, object]] = []

    def compile_instance(text: str, unset: Mapping[tuple[int, int], str]) -> ProcessPlan:
        raise FfrwdError(ErrorCode.UNSUPPORTED_SQL, "no such file")

    tap = tmp_path / "tap.ndjson"
    tap.write_text(
        '{"url": "a.mp4", "start_pts": 1.0, "duration": 1.0}\n'
        '{"url": "b.mp4", "start_pts": 3.0, "duration": 1.0}\n',
        encoding="utf-8",
    )
    run = execute._LateralRun(
        replace(_LATERAL, pipe=True), compile_instance, None, rows.append, None, None,
        str(tap),
    )
    try:
        _until(lambda: len(rows) == 2)
    finally:
        run.stop()
        run.join()
    assert [(row["row"], row["refused"]) for row in rows] == [
        (1, "no such file"),
        (2, "no such file"),
    ]
