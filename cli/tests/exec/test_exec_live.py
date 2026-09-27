"""End-to-end exec tests for live inputs: a feed read once, its edges sized.

Marked ``@pytest.mark.exec`` and excluded from the default run. Run explicitly::

    python -m pytest -m exec tests/exec/test_exec_live.py -q

The feed is 1080p, conformed to 720p ahead of the reader's split the way the
SMART demo conforms whatever it is sent, and stamped onto a wall clock with
``setpts``. One leg goes through the ``invert`` module, the other straight to
the ``hstack`` that meets it. Three feeds carry the same pictures: a paced
lavfi graph, an SRT listener and an RTMP listener, the last two fed by an
ffmpeg sender this module starts and stops itself, and each declared with
``shape`` so nothing is probed.

The edge that waits for the module is forced onto ffmpeg's fifo muxer by a
lower pipe limit. That is the road the demo's 1080p runs took, where every
frame was dropped: the fifo muxer declares no variable frame rate, so ffmpeg
ran it at a constant one, found billions of frames missing between zero and
the wall clock, and would not duplicate them.

A listener's sender may also arrive late. The ``feed-probe`` module reads a
feeder beside the feed, and its sender dials only after the whole feeder
wait has gone: the wait counts from the feed's first bytes, not the launch.

A 1080p60 feed with sound, conformed to 720p30, keeps up with the wall: its
picture and sound chains run as two filtergraphs, where one would hold the
picture to about 23 frames a second.

A feed handed on a second at a time over TCP, the way a MoQ relay hands a
subscriber each group of pictures at once, passes through ``ffrwd.leaky``
whole: the leaky learns the spread of its delivery and drops next to
nothing, where learning none it drops about half.

Requires ``ffmpeg``/``ffprobe`` on PATH with libx264, libsrt and ffv1, the
``ffrwd-wasm`` sidecar, and the sidecar fleet's ``invert`` and ``feed-probe``
modules built for ``wasm32-wasip2``. Tests skip cleanly when any of those is
missing; the 720p30 conform needs ffmpeg alone.
"""

from __future__ import annotations

import importlib
import json
import shutil
import socket
import subprocess
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from ffrwd import binaries, compiler, processes, wasm
from ffrwd.compiler import compile_all, compile_sql, emitted_commands
from ffrwd.console import Work
from ffrwd.emit import build_ffmpeg_args, emit
from ffrwd.execute import execute, execute_plan, plan_argv
from ffrwd.processes import StreamEdge, VideoFormat

pytestmark = pytest.mark.exec

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_MODULE = (
    _REPO_ROOT / "sidecar" / "modules" / "target" / "wasm32-wasip2" / "release" / "invert.wasm"
)
_TIMEOUT = 120.0
_SECONDS = 2
_RATE = 30
_FRAMES = _SECONDS * _RATE
_EPOCH = 1790351579
_SOURCE = f"testsrc2=size=1920x1080:rate={_RATE}:duration={_SECONDS}"
_SHAPE = f"shape => STRUCT(1920 AS width, 1080 AS height, {_RATE} AS fps)"


@pytest.fixture(autouse=True)
def _require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg/ffprobe not found on PATH")


@pytest.fixture
def _require_modules() -> None:
    if binaries.ffrwd_wasm_path() is None:
        pytest.skip("ffrwd-wasm not found (uv sync --extra wasm)")
    if not _MODULE.exists():
        pytest.skip(f"module missing: {_MODULE}")


def _free_port(kind: int) -> int:
    """A port nothing holds right now, of the socket `kind` given."""
    with socket.socket(socket.AF_INET, kind) as probe:
        probe.bind(("127.0.0.1", 0))
        port: int = probe.getsockname()[1]
        return port


class _Sender:
    """An ffmpeg that publishes the feed, dialling until the listener answers.

    Every process it starts is its own, and :meth:`stop` ends each one still
    running, whatever it started with it.
    """

    def __init__(self, destination: list[str], feed: list[str] | None = None) -> None:
        """`feed` is what is read and sent; the 1080p pictures by default."""
        self.argv = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            *(feed or [
                "-re", "-f", "lavfi", "-i", _SOURCE,
                "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
                "-g", str(_RATE), "-pix_fmt", "yuv420p",
            ]),
            *destination,
        ]  # fmt: skip
        self.started: list[subprocess.Popen[str]] = []
        self.delivered = False
        self._after = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._dial, daemon=True)

    def start(self, after: float = 0.0) -> None:
        """Start dialling, `after` seconds from now: a sender that is late."""
        self._after = after
        self._thread.start()

    def _dial(self) -> None:
        if self._stop.wait(self._after):
            return
        deadline = time.monotonic() + _TIMEOUT / 2
        while not self._stop.is_set() and time.monotonic() < deadline:
            sender = subprocess.Popen(
                self.argv,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            self.started.append(sender)
            if sender.wait() == 0:
                self.delivered = True
                return
            time.sleep(0.3)

    def stop(self) -> None:
        self._stop.set()
        for sender in self.started:
            if sender.poll() is None:
                binaries.end_tree(sender)
                sender.wait(timeout=10)
        if self._thread.ident is not None:
            self._thread.join(timeout=10)


def _query(spelled: str, out_path: Path) -> str:
    return (
        "CREATE FUNCTION invert(v video_stream) RETURNS video_stream\n"
        f"  AS '{_MODULE.as_posix()}', 'invert' LANGUAGE wasm;\n"
        "COPY (\n"
        "  WITH feed AS (\n"
        f"    SELECT setpts(scale(s.video[1], 1280, 720), 'PTS+{_EPOCH}/TB') AS v\n"
        f"    FROM {spelled} s\n"
        "  )\n"
        "  SELECT ffmpeg.hstack(feed.v, invert(feed.v)) FROM feed\n"
        f") TO '{out_path.as_posix()}' WITH (video_codec 'ffv1')"
    )


def _listener(
    protocol: str, shape: str = _SHAPE, feed: list[str] | None = None
) -> tuple[str, _Sender]:
    """A listening input as the query spells it, and the sender to dial it."""
    if protocol == "srt":
        port = _free_port(socket.SOCK_DGRAM)
        spelled = f"input('srt://127.0.0.1:{port}?mode=listener&latency=200000', {shape})"
        return spelled, _Sender(
            ["-f", "mpegts", f"srt://127.0.0.1:{port}?mode=caller&latency=200000"], feed
        )
    port = _free_port(socket.SOCK_STREAM)
    spelled = f"input('rtmp://127.0.0.1:{port}/live/test', listen => true, {shape})"
    return spelled, _Sender(["-f", "flv", f"rtmp://127.0.0.1:{port}/live/test"], feed)


@pytest.fixture(params=["realtime", "srt", "rtmp"])
def _feed(request: pytest.FixtureRequest) -> Iterator[tuple[str, _Sender | None]]:
    """The input as the query spells it, and the sender feeding it, if any."""
    if request.param == "realtime":
        yield f"input('{_SOURCE}', format => 'lavfi', realtime => true)", None
        return
    spelled, sender = _listener(request.param)
    try:
        yield spelled, sender
    finally:
        sender.stop()


def _written(path: Path) -> dict[str, object]:
    done = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
            "-show_entries", "stream=nb_read_frames,width,height,start_time",
            "-of", "json", str(path),
        ],
        capture_output=True,
        text=True,
        timeout=_TIMEOUT,
        check=False,
    )  # fmt: skip
    assert done.returncode == 0, done.stderr
    streams = json.loads(done.stdout)["streams"]
    assert streams, f"{path} carries no video stream"
    found: dict[str, object] = streams[0]
    return found


@pytest.mark.usefixtures("_require_modules")
def test_a_720p_conform_of_a_1080p_feed_crosses_the_fifo_road_frame_for_frame(
    _feed: tuple[str, _Sender | None],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spelled, sender = _feed

    def no_probe(*args: object, **kwargs: object) -> None:
        raise AssertionError(f"probed {args}: a declared shape must not be")

    if sender is not None:
        monkeypatch.setattr(compiler, "probe_path", no_probe)
    # Two 720p frames are 2.7 MB: a limit under that puts the edge on the
    # fifo road, where the 1080p demo runs lost every frame.
    monkeypatch.setattr(processes, "PIPE_BUFFER_LIMIT", 1 << 20)
    out_path = tmp_path / "live.mkv"
    compiled = compile_all(_query(spelled, out_path))
    plan = compiled.plan
    assert plan is not None

    # The one process that opens the feed, and its edge to the hstack.
    reader = next(p for p in plan.ffmpeg if "pipe:" not in p.graph.input_paths)
    direct = next(
        e
        for e in plan.stream_edges
        if isinstance(e, StreamEdge) and e.source == reader.id and e.target.startswith("ffmpeg")
    )
    # Sized as the 720p it carries, not the 1080p it was sent, and still over
    # the lowered limit.
    assert isinstance(direct.format, VideoFormat)
    assert (direct.format.width, direct.format.height) == (1280, 720)
    assert direct.buffer is not None and direct.buffer.road == "fifo"
    argv = plan_argv(
        plan,
        sidecar_argv=wasm.sidecar_argv,
        pipe_path=lambda edge, side: f"{edge.source}-{edge.target}-{side}",
    )
    assert argv[reader.id].count("passthrough") == 2

    if sender is not None:
        sender.start()
    result = execute_plan(
        plan, sidecar_argv=wasm.sidecar_argv, overwrite=True, timeout=_TIMEOUT
    )
    assert result.exit_code == 0, "\n".join(
        f"{member.id} exited {member.exit_code}: {member.stderr_tail}"
        for stage in result.stages
        for member in stage.members
    )
    assert not result.timed_out
    assert result.overflow is None, str(result.overflow)
    if sender is not None:
        assert sender.delivered, "the sender never got through to the listener"

    written = _written(out_path)
    assert (written["width"], written["height"]) == (2560, 720)
    frames = int(str(written["nb_read_frames"]))
    if sender is None:
        assert frames == _FRAMES
    else:
        # A network feed may lose the tail a sender closes on; nothing else.
        assert _FRAMES - 10 <= frames <= _FRAMES, frames
    # On the wall clock the query stamped, which is what the fifo road lost.
    assert float(str(written["start_time"])) >= _EPOCH


@pytest.mark.usefixtures("_require_modules")
def test_an_rtmp_listener_waits_for_its_publisher_and_ends_with_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A query with no module is one ffmpeg: ``-listen 1 -timeout 30`` before
    its ``-i``, waiting for the publisher, taking every frame it sends, and
    done when the publisher is."""

    def no_probe(*args: object, **kwargs: object) -> None:
        raise AssertionError(f"probed {args}: a listening input must not be")

    monkeypatch.setattr(compiler, "probe_path", no_probe)
    port = _free_port(socket.SOCK_STREAM)
    url = f"rtmp://127.0.0.1:{port}/live/test"
    out_path = tmp_path / "rtmp.mkv"
    graph = compile_sql(
        f"COPY (SELECT s.video[1] FROM input('{url}', listen => true, "
        f"listen_timeout => 30, {_SHAPE}) s) "
        f"TO '{out_path.as_posix()}' WITH (video_codec 'ffv1')"
    )
    argv = build_ffmpeg_args(emit(graph))
    at = argv.index("-i")
    assert argv[at - 4 : at + 2] == ["-listen", "1", "-timeout", "30", "-i", url]

    run = subprocess.Popen(
        [*argv[:1], "-hide_banner", "-y", *argv[1:]],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    sender = _Sender(["-f", "flv", url])
    try:
        sender.start()
        _, err = run.communicate(timeout=_TIMEOUT)
    finally:
        if run.poll() is None:
            binaries.end_tree(run)
            run.wait(timeout=10)
        sender.stop()

    assert run.returncode == 0, err[-2000:]
    assert sender.delivered
    written = _written(out_path)
    assert (written["width"], written["height"]) == (1920, 1080)
    assert int(str(written["nb_read_frames"])) == _FRAMES


_PROBE = (
    _REPO_ROOT / "sidecar" / "modules" / "target" / "wasm32-wasip2" / "release"
    / "feed_probe.wasm"
)
# The feeder wait this test runs with, and how long after the run starts the
# sender dials: later than the whole wait.
_FEEDER_WAIT = 4.0
_LATE = 8.0
# The feeder: a second at the feed's rate, smaller than the feed.
_FEEDER_SPEC = f"testsrc2=size=320x180:rate={_RATE}:duration=1"


@pytest.mark.usefixtures("_require_modules")
@pytest.mark.parametrize("protocol", ["srt", "rtmp"])
def test_a_feeder_waits_for_a_listener_whose_sender_is_late(
    protocol: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A presenter starts sending whenever they are ready. The module cannot
    listen for its feeder before the programme's header reaches it, so the
    feeder wait counts from the first bytes the listener delivers, and a
    sender dialling after the whole wait has gone still gets a full run."""
    if not _PROBE.exists():
        pytest.skip(f"module missing: {_PROBE}")
    monkeypatch.setattr(importlib.import_module("ffrwd.execute"), "FEEDER_WAIT", _FEEDER_WAIT)
    feeder = tmp_path / "feeder.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", _FEEDER_SPEC,
         "-pix_fmt", "yuv420p", str(feeder)],
        check=True,
        timeout=_TIMEOUT,
    )  # fmt: skip
    spelled, sender = _listener(protocol)
    rows_path = tmp_path / "rows.ndjson"
    compiled = compile_all(
        "CREATE FUNCTION probe(v video_stream, feed video_stream DEFAULT NULL, "
        "port number DEFAULT 9000)\n"
        "RETURNS STRUCT(v video_stream, feeds STRUCT(feed_pts number, w number, "
        "h number)[])\n"
        f"AS '{_PROBE.as_posix()}', 'feed-probe' LANGUAGE wasm;\n"
        f"COPY (SELECT probe(s.video[1], a.video[1]).feeds FROM {spelled} s, "
        f"input('{feeder.as_posix()}') a) TO '{rows_path.as_posix()}'"
    )
    plan = compiled.plan
    assert plan is not None and plan.feeder_edges

    started = time.monotonic()
    try:
        sender.start(after=_LATE)
        result = execute_plan(
            plan, sidecar_argv=wasm.sidecar_argv, overwrite=True, timeout=_TIMEOUT
        )
    finally:
        sender.stop()
    assert result.overflow is None, str(result.overflow)
    assert result.exit_code == 0, "\n".join(
        f"{member.id} exited {member.exit_code}: {member.stderr_tail}"
        for stage in result.stages
        for member in stage.members
    )
    assert not result.timed_out
    assert sender.delivered, "the sender never got through to the listener"
    assert time.monotonic() - started >= _LATE

    rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line]
    # Every picture of the feeder, at the programme's size.
    assert len(rows) == _RATE
    assert {(row["w"], row["h"]) for row in rows} == {(1920, 1080)}



# A 1080p60 feed with sound, conformed to 720p30 the way the SMART demo's
# reader conforms it: long enough for a reader that falls behind to show it.
_CONFORM_SECONDS = 20
_CONFORM_RATE = 30
_CONFORM_SHAPE = (
    "shape => STRUCT(1920 AS width, 1080 AS height, 60 AS fps, 48000 AS rate, 2 AS channels)"
)
# How far the pictures may fall behind the wall: the sender's start-up burst
# and the encoder's own delay, and nothing that grows with the run.
_BEHIND = 2.0


@pytest.fixture(scope="module")
def _feed_1080p60(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The feed as MPEG-TS: 1080p60 H.264 and 48 kHz stereo AAC."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not found on PATH")
    path = tmp_path_factory.mktemp("feed") / "feed1080p60.ts"
    seconds = f"duration={_CONFORM_SECONDS}"
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y",
            "-f", "lavfi", "-i", f"testsrc2=size=1920x1080:rate=60:{seconds}",
            "-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=48000:{seconds}",
            "-ac", "2", "-c:v", "libx264", "-preset", "veryfast", "-g", "120",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k", str(path),
        ],
        check=True,
        timeout=_TIMEOUT,
    )  # fmt: skip
    return path


def _conform_query(spelled: str, out_path: Path) -> str:
    return (
        "COPY (\n"
        "  SELECT ffmpeg.format(scale(ffmpeg.fps(s.video[1], 30), 1280, 720), 'yuv420p'),\n"
        "         ffmpeg.aformat(aresample(s.audio[1], 48000, async => 1, first_pts => 0),\n"
        "                        channel_layouts => 'stereo')\n"
        f"  FROM {spelled} s\n"
        f") TO '{out_path.as_posix()}'\n"
        "  WITH (video_codec 'libx264', preset 'ultrafast', tune 'zerolatency',\n"
        "        audio_codec 'aac')"
    )


def _behind(argv: list[str]) -> float:
    """Run `argv`, and how far its pictures fell behind the wall at worst.

    ffmpeg reports its picture count twice a second. Behind is the wall time
    since the first report with a picture in it, less the media time the
    pictures written since then cover.
    """
    run = subprocess.Popen(
        [argv[0], "-hide_banner", "-y", "-nostats", "-progress", "pipe:1",
         "-stats_period", "0.5", *argv[1:]],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )  # fmt: skip
    errors: list[str] = []
    drain = threading.Thread(target=lambda: errors.extend(run.stderr or ()), daemon=True)
    drain.start()
    first: tuple[float, int] | None = None
    worst = 0.0
    try:
        for line in run.stdout or ():
            key, _, value = line.strip().partition("=")
            if key != "frame" or int(value) == 0:
                continue
            now, frames = time.monotonic(), int(value)
            if first is None:
                first = (now, frames)
            worst = max(worst, now - first[0] - (frames - first[1]) / _CONFORM_RATE)
        run.wait(timeout=_TIMEOUT)
    finally:
        if run.poll() is None:
            binaries.end_tree(run)
            run.wait(timeout=10)
    drain.join(timeout=10)
    assert run.returncode == 0, "".join(errors)[-2000:]
    return worst


@pytest.mark.parametrize("feed", ["realtime", "srt", "rtmp"])
def test_a_1080p60_feed_conformed_to_720p30_keeps_up_with_the_wall(
    feed: str, _feed_1080p60: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reader's picture chain (60 to 30 fps) and sound chain share no node,
    so each runs as a filtergraph of its own. Held in one, an MPEG-TS feed's
    picture falls to about 23 frames a second and the rest pile up until the
    input ends; apart, every picture comes out as its time comes."""
    out_path = tmp_path / "conformed.mkv"
    sender: _Sender | None = None
    if feed == "realtime":
        spelled = f"input('{_feed_1080p60.as_posix()}', realtime => true)"
    else:

        def no_probe(*args: object, **kwargs: object) -> None:
            raise AssertionError(f"probed {args}: a declared shape must not be")

        monkeypatch.setattr(compiler, "probe_path", no_probe)
        copied = ["-re", "-i", str(_feed_1080p60), "-c", "copy"]
        spelled, sender = _listener(feed, _CONFORM_SHAPE, copied)
    argv = build_ffmpeg_args(emit(compile_sql(_conform_query(spelled, out_path))))
    assert argv.count("-filter_complex") == 2

    try:
        if sender is not None:
            sender.start()
        behind = _behind(argv)
    finally:
        if sender is not None:
            sender.stop()
    if sender is not None:
        assert sender.delivered, "the sender never got through to the listener"

    done = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
         "-show_entries", "stream=nb_read_packets,width,height",
         "-of", "json", str(out_path)],
        capture_output=True, text=True, timeout=_TIMEOUT, check=False,
    )  # fmt: skip
    assert done.returncode == 0, done.stderr
    written = json.loads(done.stdout)["streams"][0]
    assert (written["width"], written["height"]) == (1280, 720)
    pictures = int(written["nb_read_packets"])
    due = _CONFORM_SECONDS * _CONFORM_RATE
    if sender is None:
        assert pictures == due
    else:
        # A network feed may lose the tail a sender closes on; nothing else.
        assert due - 10 <= pictures <= due, pictures
    assert behind < _BEHIND, f"the pictures fell {behind:.1f} s behind the wall"


# A 720p30 feed with sound, stamped onto the wall clock the way the demo's
# heads stamp it, into a picture path that cannot keep up: nlmeans at these
# settings manages about a third of the feed's rate on the machine this was
# written on. Through `ffrwd.leaky` the picture stays near the wall and the
# sound arrives whole; without it the picture falls further behind the
# longer the feed runs.
_LEAKY_SECONDS = 10
_MAX_LATENESS = 0.5
# What the picture may trail by past max_lateness: the frames the slow
# ffmpeg's own queues hold past the leaky, and the half second between two
# of its progress readings. Those frames are counted at the slow stage's own
# rate, so how far behind the picture SETTLES depends on the machine (about
# 1.1 s here, 2.3 s on a slower CI runner); what the leaky guarantees is that
# it settles. _LEAKY_SETTLED is how much the lag may still move over the
# second half of the feed.
_LEAKY_MARGIN = 1.0
_LEAKY_SETTLED = 0.75
_SLOW = "nlmeans({}, s => 4, p => 5, r => 9)"
_SLOW_FILTER = "nlmeans=s=4:p=5:r=9"
# The reader's own probe of the feed is bounded: what it reads while it
# probes reaches the leaky as one late burst ahead of everything else.
_LEAKY_SHAPE = (
    "shape => STRUCT(1280 AS width, 720 AS height, 30 AS fps, 48000 AS rate, "
    "2 AS channels), analyzeduration => 500000"
)
_LEAKY_FEED = [
    "-re", "-f", "lavfi", "-i", f"testsrc2=size=1280x720:rate=30:duration={_LEAKY_SECONDS}",
    "-re", "-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=48000:duration={_LEAKY_SECONDS}",
    "-ac", "2", "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
    "-g", "30", "-pix_fmt", "yuv420p", "-c:a", "aac",
]  # fmt: skip


@pytest.fixture(scope="module")
def _slow_path() -> None:
    """Skip where the slow path keeps up after all: it has to be slower than
    the feed for either test to say anything."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not found on PATH")
    frames = 60
    started = time.monotonic()
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
         f"testsrc2=size=1280x720:rate=30:duration={frames // 30}",
         "-vf", _SLOW_FILTER, "-f", "null", "-"],
        check=True,
        timeout=_TIMEOUT,
    )  # fmt: skip
    rate = frames / (time.monotonic() - started)
    if rate > 20:
        pytest.skip(f"{_SLOW_FILTER} runs at {rate:.0f} fps here, near enough to keep up")


def _leaky_query(spelled: str, out_path: Path, *, leaky: bool) -> str:
    epoch = int(time.time())
    picture = f"setpts(s.video[1], 'PTS-STARTPTS+{epoch}/TB')"
    if leaky:
        picture = f"ffrwd.leaky({picture}, max_lateness => {_MAX_LATENESS})"
    return (
        "COPY (\n"
        f"  SELECT {_SLOW.format(picture)},\n"
        f"         asetpts(s.audio[1], 'PTS-STARTPTS+{epoch}/TB')\n"
        f"  FROM {spelled} s\n"
        f") TO '{out_path.as_posix()}'\n"
        "  WITH (video_codec 'libx264', preset 'ultrafast', tune 'zerolatency',\n"
        "        gop 30, audio_codec 'pcm_s16le')"
    )


def _lags(readings: list[tuple[float, float]]) -> list[float]:
    """How much further behind the wall each progress reading is than the
    first: wall time gone by, less output time written meanwhile."""
    written = [(at, out) for at, out in readings if out > 0]
    assert len(written) > 4, f"too few progress readings: {readings}"
    first_at, first_out = written[0]
    return [(at - first_at) - (out - first_out) for at, out in written]


def _run_slow(
    protocol: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, leaky: bool
) -> tuple[list[float], list[dict[str, object]], Path]:
    """The feed through the slow path, with or without a leaky: the lags its
    progress readings show, the rows the run reported, and the file."""

    def no_probe(*args: object, **kwargs: object) -> None:
        raise AssertionError(f"probed {args}: a declared shape must not be")

    monkeypatch.setattr(compiler, "probe_path", no_probe)
    spelled, sender = _listener(protocol, _LEAKY_SHAPE, _LEAKY_FEED)
    out_path = tmp_path / "slow.mkv"
    compiled = compile_all(_leaky_query(spelled, out_path, leaky=leaky))
    readings: list[tuple[float, float]] = []
    rows: list[dict[str, object]] = []

    def work(reading: Work) -> None:
        readings.append((time.monotonic(), reading.out_time))

    try:
        sender.start()
        if leaky:
            assert compiled.plan is not None
            result = execute_plan(
                compiled.plan,
                sidecar_argv=wasm.sidecar_argv,
                overwrite=True,
                timeout=_TIMEOUT,
                rows=lambda row: rows.append(dict(row)),
                work=work,
            )
            assert result.overflow is None, str(result.overflow)
            assert result.exit_code == 0, "\n".join(
                f"{member.id} exited {member.exit_code}: {member.stderr_tail}"
                for stage in result.stages
                for member in stage.members
            )
        else:
            # No module and no leaky: the one ffmpeg command it always was.
            assert compiled.plan is None
            ran = execute(
                emitted_commands(compiled.graphs),
                overwrite=True,
                timeout=_TIMEOUT,
                capture_stderr=True,
                work=work,
            )
            assert ran.exit_code == 0
    finally:
        sender.stop()
    assert sender.delivered, "the sender never got through to the listener"
    return _lags(readings), rows, out_path


def _keyframes(path: Path) -> list[float]:
    """When each keyframe of the picture in `path` is shown, in seconds."""
    done = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "packet=pts_time,flags", "-of", "json", str(path)],
        capture_output=True, text=True, timeout=_TIMEOUT, check=False,
    )  # fmt: skip
    assert done.returncode == 0, done.stderr
    return [
        float(packet["pts_time"])
        for packet in json.loads(done.stdout)["packets"]
        if "K" in packet["flags"]
    ]


def _audio_packets(path: Path) -> list[tuple[float, float]]:
    done = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "packet=pts_time,duration_time", "-of", "json", str(path)],
        capture_output=True, text=True, timeout=_TIMEOUT, check=False,
    )  # fmt: skip
    assert done.returncode == 0, done.stderr
    return [
        (float(packet["pts_time"]), float(packet["duration_time"]))
        for packet in json.loads(done.stdout)["packets"]
    ]


@pytest.mark.usefixtures("_slow_path")
@pytest.mark.parametrize("protocol", ["srt", "rtmp"])
def test_a_leaky_keeps_a_slow_picture_near_the_wall_and_the_sound_whole(
    protocol: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if binaries.ffrwd_wasm_path() is None:
        pytest.skip("ffrwd-wasm not found (uv sync --extra wasm)")
    lags, rows, out_path = _run_slow(protocol, tmp_path, monkeypatch, leaky=True)

    # (a) Once the leaky has taken up its budget and the slow stage's queues
    # are full, the picture stops losing ground: over the second half of the
    # feed the lag holds still, where without the leaky it keeps growing.
    settled = lags[len(lags) // 2 :]
    assert max(settled) - min(settled) <= _LEAKY_SETTLED, lags

    # (b) What it could not keep up with, it dropped, and said so.
    assert rows and all(row["kind"] == "leaky" for row in rows)
    assert set(rows[0]) == {
        "kind", "node", "passed", "dropped", "lateness_s", "baseline_s", "spread_s",
    }  # fmt: skip
    assert sum(int(str(row["dropped"])) for row in rows) > 0, rows

    # (c) A group stays a second long however many pictures the leaky drops,
    # where gop 30 alone would make one every 30 of the few that pass,
    # seconds apart: a keyframe is forced a second after the last, so two
    # are at most a second and a picture's gap apart.
    keyframes = _keyframes(out_path)
    assert len(keyframes) >= _LEAKY_SECONDS - 2, keyframes
    assert max(b - a for a, b in zip(keyframes, keyframes[1:])) < 1.5, keyframes

    # (d) The sound is never dropped: one unbroken run of it, as long as the
    # feed less the tail a sender closes on.
    packets = _audio_packets(out_path)
    gaps = [b[0] - (a[0] + a[1]) for a, b in zip(packets, packets[1:])]
    assert max(gaps) < 0.005, max(gaps)
    heard = packets[-1][0] + packets[-1][1] - packets[0][0]
    assert heard >= _LEAKY_SECONDS - 0.5, heard


@pytest.mark.usefixtures("_slow_path")
@pytest.mark.parametrize("protocol", ["srt", "rtmp"])
def test_without_a_leaky_the_same_slow_picture_falls_further_behind(
    protocol: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lags, rows, _ = _run_slow(protocol, tmp_path, monkeypatch, leaky=False)

    assert not rows
    # Behind by more at the end than the leaky ever lets it be, and still
    # losing ground over the second half.
    assert lags[-1] > _MAX_LATENESS + _LEAKY_MARGIN + 1.0, lags
    assert lags[-1] - lags[len(lags) // 2] > 1.0, lags


# A feed that arrives a second at a time, the way a MoQ relay hands a
# subscriber each group of pictures at once: 30 pictures within a few
# milliseconds, once a second, so a group's first picture is a second later
# than its last. Nothing slow follows the leaky. It learns that spread from
# the feed's own delivery and drops next to nothing; learning none
# (max_spread => 0), the older half of every second is past half a second.
_BURST_SECONDS = 12
_BURST_SHAPE = (
    "shape => STRUCT(640 AS width, 360 AS height, 30 AS fps), analyzeduration => 500000"
)


@pytest.fixture(scope="module")
def _groups(tmp_path_factory: pytest.TempPathFactory) -> list[bytes]:
    """The feed as one MPEG-TS piece per second: a group of 30 pictures each."""
    folder = tmp_path_factory.mktemp("groups")
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
         f"testsrc2=size=640x360:rate=30:duration={_BURST_SECONDS}",
         "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
         "-g", "30", "-keyint_min", "30", "-sc_threshold", "0", "-pix_fmt", "yuv420p",
         "-f", "segment", "-segment_time", "1", "-segment_format", "mpegts",
         "-reset_timestamps", "0", str(folder / "g%03d.ts")],
        check=True,
        timeout=_TIMEOUT,
    )  # fmt: skip
    groups = [path.read_bytes() for path in sorted(folder.glob("g*.ts"))]
    assert len(groups) == _BURST_SECONDS, len(groups)
    return groups


class _Bursts:
    """Dials a TCP listener and hands it each second of the feed at once,
    once that second is over, as a relay hands on a group."""

    def __init__(self, port: int, groups: list[bytes]) -> None:
        self.port = port
        self.groups = groups
        self.sent = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._pump, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _pump(self) -> None:
        deadline = time.monotonic() + _TIMEOUT / 2
        while not self._stop.is_set() and time.monotonic() < deadline:
            try:
                connection = socket.create_connection(("127.0.0.1", self.port), timeout=1)
            except OSError:
                time.sleep(0.2)
                continue
            with connection:
                began = time.monotonic()
                for index, group in enumerate(self.groups):
                    if self._stop.wait(max(0.0, began + index + 1 - time.monotonic())):
                        return
                    connection.sendall(group)
                    self.sent += 1
            return

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=10)


@pytest.mark.parametrize(("max_spread", "learns"), [(None, True), (0, False)])
def test_a_leaky_learns_a_bursty_feeds_spread_and_drops_next_to_nothing(
    max_spread: int | None,
    learns: bool,
    _groups: list[bytes],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if binaries.ffrwd_wasm_path() is None:
        pytest.skip("ffrwd-wasm not found (uv sync --extra wasm)")

    def no_probe(*args: object, **kwargs: object) -> None:
        raise AssertionError(f"probed {args}: a declared shape must not be")

    monkeypatch.setattr(compiler, "probe_path", no_probe)
    port = _free_port(socket.SOCK_STREAM)
    epoch = int(time.time())
    limits = "max_lateness => 0.5"
    if max_spread is not None:
        limits += f", max_spread => {max_spread}"
    out_path = tmp_path / "bursts.mkv"
    query = (
        "COPY (\n"
        f"  SELECT ffrwd.leaky(setpts(s.video[1], 'PTS-STARTPTS+{epoch}/TB'), {limits})\n"
        f"  FROM input('tcp://127.0.0.1:{port}?listen=1', format => 'mpegts', "
        f"{_BURST_SHAPE}) s\n"
        f") TO '{out_path.as_posix()}' WITH (video_codec 'libx264', preset 'ultrafast')"
    )
    compiled = compile_all(query)
    assert compiled.plan is not None
    rows: list[dict[str, object]] = []
    pump = _Bursts(port, _groups)
    try:
        pump.start()
        result = execute_plan(
            compiled.plan,
            sidecar_argv=wasm.sidecar_argv,
            overwrite=True,
            timeout=_TIMEOUT,
            rows=lambda row: rows.append(dict(row)),
        )
    finally:
        pump.stop()
    assert result.exit_code == 0, "\n".join(
        f"{member.id} exited {member.exit_code}: {member.stderr_tail}"
        for stage in result.stages
        for member in stage.members
    )
    assert pump.sent == _BURST_SECONDS

    assert rows and all(row["kind"] == "leaky" for row in rows)
    passed = sum(int(str(row["passed"])) for row in rows)
    dropped = sum(int(str(row["dropped"])) for row in rows)
    assert passed + dropped >= _BURST_SECONDS * 30 - 30, rows
    spreads = [float(str(row["spread_s"])) for row in rows]
    if learns:
        # A second's pictures, less the time they took to arrive. The reader
        # hands on its probe backlog first, and the pipeline starting up may
        # cut that into pieces, so the second group can lose a few of its
        # oldest pictures; from then on nothing is dropped, and every row
        # carries the groups' own spread.
        assert all(0.85 <= spread <= 1.1 for spread in spreads[2:]), rows
        assert dropped < 15, rows
        assert all(row["dropped"] == 0 for row in rows[2:]), rows
    else:
        assert spreads == [0.0] * len(spreads)
        assert dropped > (passed + dropped) // 4, rows
