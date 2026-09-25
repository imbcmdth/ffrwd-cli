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
from collections.abc import Callable, Iterator, Mapping
from dataclasses import replace
from pathlib import Path

import pytest

from ffrwd import wasm
from ffrwd.compiler import compile_all
from ffrwd.errors import ErrorCode, FfrwdError
from ffrwd.execute import render_plan
from ffrwd.ir import FeederCall, Graph, Lateral, LateralConnection, LateralValue
from ffrwd.parser import parse, resolve
from ffrwd.probe import ProbeResult, StreamMeta
from ffrwd.processes import ProcessPlan
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
            "input declared 'feeder'",
        ),
        (
            _ADS.format(select="ad.audio", stream="s.data[1]"),
            ErrorCode.UNSUPPORTED_SQL,
            "'ad.audio' comes from LATERAL play(s.data[1]), which starts a source per "
            "row of a data stream: it is empty between rows, and a COPY reads a frame "
            "on every tick. A run-time lateral's streams can only go to a module "
            "input declared 'feeder'",
        ),
        (
            _ADS.format(select="probe(ad.video)", stream="s.data[1]"),
            ErrorCode.UNSUPPORTED_SQL,
            "'ad.video' comes from LATERAL play(s.data[1]), which starts a source per "
            "row of a data stream: it is empty between rows, and probe reads a frame "
            "on every tick. A run-time lateral's streams can only go to a module "
            "input declared 'feeder'",
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
            "the feeder 'feed' reads 'two', and another feeder of the group 'switch' "
            "reads 'one': the group shares one connection, which carries one source",
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
