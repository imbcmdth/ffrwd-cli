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

Requires ``ffmpeg``/``ffprobe`` on PATH with libx264, libsrt and ffv1, the
``ffrwd-wasm`` sidecar, and the sidecar fleet's ``invert`` module built for
``wasm32-wasip2``. Tests skip cleanly when any of those is missing.
"""

from __future__ import annotations

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
from ffrwd.compiler import compile_all, compile_sql
from ffrwd.emit import build_ffmpeg_args, emit
from ffrwd.execute import execute_plan, plan_argv
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
def _require_everything() -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg/ffprobe not found on PATH")
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

    def __init__(self, destination: list[str]) -> None:
        self.argv = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-re", "-f", "lavfi", "-i", _SOURCE,
            "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
            "-g", str(_RATE), "-pix_fmt", "yuv420p",
            *destination,
        ]  # fmt: skip
        self.started: list[subprocess.Popen[str]] = []
        self.delivered = False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._dial, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _dial(self) -> None:
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


@pytest.fixture(params=["realtime", "srt", "rtmp"])
def _feed(request: pytest.FixtureRequest) -> Iterator[tuple[str, _Sender | None]]:
    """The input as the query spells it, and the sender feeding it, if any."""
    if request.param == "realtime":
        yield f"input('{_SOURCE}', format => 'lavfi', realtime => true)", None
        return
    if request.param == "srt":
        port = _free_port(socket.SOCK_DGRAM)
        spelled = f"input('srt://127.0.0.1:{port}?mode=listener&latency=200000', {_SHAPE})"
        sender = _Sender(
            ["-f", "mpegts", f"srt://127.0.0.1:{port}?mode=caller&latency=200000"]
        )
    else:
        port = _free_port(socket.SOCK_STREAM)
        spelled = f"input('rtmp://127.0.0.1:{port}/live/test', listen => true, {_SHAPE})"
        sender = _Sender(["-f", "flv", f"rtmp://127.0.0.1:{port}/live/test"])
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
