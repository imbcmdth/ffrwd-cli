"""A sidecar writing its NUT to a ``tcp://`` or ``udp://`` destination itself.

Marked ``@pytest.mark.exec``: the query compiles against the installed ffmpeg
and the sidecar's own ``--describe``, runs for real, and a listener here
takes what arrives. Requires ``ffmpeg``/``ffprobe`` on PATH, the generated
fixtures, the ``ffrwd-wasm`` sidecar, and the ``matte``, ``dim`` and
``source_replay`` modules built for ``wasm32-wasip2``. Tests skip cleanly
when any of those is missing.
"""

from __future__ import annotations

import json
import shutil
import socket
import subprocess
import threading
from pathlib import Path

import pytest

from ffrwd import binaries, cli
from ffrwd.compiler import compile_all

pytestmark = pytest.mark.exec

_CLI_ROOT = Path(__file__).resolve().parent.parent.parent
_RELEASE = _CLI_ROOT.parent / "sidecar" / "modules" / "target" / "wasm32-wasip2" / "release"
_SOURCE = _CLI_ROOT / "tests" / "fixtures" / "testsrc.mp4"
_MODULES = {name: _RELEASE / f"{name}.wasm" for name in ("matte", "dim", "source_replay")}
_TIMEOUT = 120.0

# `source_replay`'s one track: seven h264 packets, 32x24.
_REPLAY_PACKETS = 7
_REPLAY_HEIGHT = 24


@pytest.fixture(autouse=True)
def _require_everything() -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg/ffprobe not found on PATH")
    if not _SOURCE.exists():
        pytest.skip(f"fixture missing: {_SOURCE} (run scripts/gen_fixtures.py first)")
    if binaries.ffrwd_wasm_path() is None:
        pytest.skip("ffrwd-wasm not found (uv sync --extra wasm)")
    for module in _MODULES.values():
        if not module.exists():
            pytest.skip(f"module missing: {module} (cargo build --target wasm32-wasip2 --release)")


def _matte_query(destination: str) -> str:
    """Recipe 172: `dim`'s picture and `matte`'s mask, both to `destination`."""
    return (
        "CREATE FUNCTION matte(v video_stream, every number DEFAULT 30)\n"
        "RETURNS STRUCT(mask video_stream, spots STRUCT(start_t number, id number,\n"
        "  x number, y number, w number, h number)[])\n"
        f"  AS '{_MODULES['matte'].as_posix()}', 'matte' LANGUAGE wasm;\n"
        "CREATE FUNCTION dim(v video_stream, boxes STRUCT(x number, y number, w number,\n"
        "  h number)[], amount number DEFAULT 0.5) RETURNS video_stream\n"
        f"  AS '{_MODULES['dim'].as_posix()}', 'dim' LANGUAGE wasm;\n"
        "COPY (\n"
        "  WITH m AS (SELECT f.video[1] AS v, (matte(f.video[1])).*\n"
        f"             FROM input('{_SOURCE.as_posix()}') f)\n"
        "  SELECT dim(m.v, m.spots), m.mask FROM m\n"
        f") TO '{destination}' WITH (format 'nut')"
    )


def _replay_query(destination: str) -> str:
    return (
        "CREATE FUNCTION replay() RETURNS source\n"
        f"  AS '{_MODULES['source_replay'].as_posix()}', 'source_replay' LANGUAGE wasm;\n"
        f"COPY (SELECT s.video[1] FROM replay() s) TO '{destination}' WITH (format 'nut')"
    )


def _streams(path: Path) -> list[dict[str, object]]:
    """Every stream `path` carries, frame-counted."""
    done = subprocess.run(
        [
            "ffprobe", "-v", "error", "-count_frames",
            "-show_entries", "stream=codec_type,codec_name,width,height,nb_read_frames",
            "-of", "json", str(path),
        ],
        capture_output=True, text=True, timeout=_TIMEOUT, check=False,
    )
    assert done.returncode == 0, done.stderr
    streams: list[dict[str, object]] = json.loads(done.stdout)["streams"]
    return streams


def _frame_hashes(path: Path) -> list[str]:
    """Each frame of `path` as ffmpeg's framemd5 lists it: stream, times and hash."""
    done = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-map", "0", "-c", "copy",
         "-f", "framemd5", "-"],
        capture_output=True, text=True, timeout=_TIMEOUT, check=False,
    )
    assert done.returncode == 0, done.stderr
    return [line for line in done.stdout.splitlines() if not line.startswith("#")]


def test_a_tcp_destination_takes_the_sidecars_nut_with_both_pictures(tmp_path: Path) -> None:
    """The plan runs no ffmpeg after the sidecar; what reaches the listener is
    one NUT with `dim`'s picture and the mask, every frame of the source in
    each, frame for frame what a NUT file written through ffmpeg holds."""
    listener = socket.create_server(("127.0.0.1", 0))
    url = f"tcp://127.0.0.1:{listener.getsockname()[1]}"
    received = tmp_path / "received.nut"

    def take() -> None:
        connection, _ = listener.accept()
        with connection, received.open("wb") as out:
            while chunk := connection.recv(1 << 16):
                out.write(chunk)

    plan = compile_all(_matte_query(url)).plan
    assert plan is not None
    assert not any(e.source == plan.sidecars[0].id for e in plan.stream_edges)
    taker = threading.Thread(target=take)
    taker.start()
    try:
        assert cli.main(["run", _matte_query(url), "-y"]) == 0
    finally:
        taker.join(_TIMEOUT)
        listener.close()

    source_frames = int(str(_streams(_SOURCE)[0]["nb_read_frames"]))
    streams = _streams(received)
    assert [(s["codec_type"], s["codec_name"]) for s in streams] == [("video", "rawvideo")] * 2
    assert [int(str(s["nb_read_frames"])) for s in streams] == [source_frames] * 2

    written = tmp_path / "written.nut"
    assert cli.main(["run", _matte_query(written.as_posix()), "-y"]) == 0
    assert _frame_hashes(received) == _frame_hashes(written)


def test_a_udp_destination_takes_the_sources_packets_in_datagrams_of_pkt_size(
    tmp_path: Path,
) -> None:
    """A packet source's track goes straight to the address, each datagram at
    most `pkt_size` bytes and none empty; together they are a NUT ffprobe
    reads every packet of."""
    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.bind(("127.0.0.1", 0))
    url = f"udp://127.0.0.1:{receiver.getsockname()[1]}?pkt_size=200"
    datagrams: list[bytes] = []

    def take() -> None:
        receiver.settimeout(_TIMEOUT)
        while True:
            try:
                datagrams.append(receiver.recv(1 << 16))
            except OSError:
                return
            receiver.settimeout(2.0)

    plan = compile_all(_replay_query(url)).plan
    assert plan is not None and plan.ffmpeg == ()
    taker = threading.Thread(target=take)
    taker.start()
    try:
        assert cli.main(["run", _replay_query(url), "-y"]) == 0
    finally:
        taker.join(_TIMEOUT)
        receiver.close()

    assert datagrams and all(1 <= len(one) <= 200 for one in datagrams)
    received = tmp_path / "received.nut"
    received.write_bytes(b"".join(datagrams))
    (stream,) = _streams(received)
    assert (stream["codec_name"], stream["height"]) == ("h264", _REPLAY_HEIGHT)
    assert int(str(stream["nb_read_frames"])) == _REPLAY_PACKETS
