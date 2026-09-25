"""End-to-end exec tests for a data filter.

Marked ``@pytest.mark.exec`` and excluded from the default run. Run
explicitly::

    python -m pytest -m exec tests/exec/test_exec_data_filter.py -q

The module under test is ``data-stamp``: it copies each message on its data
pad with ``"node"`` set to its `node` parameter, at the same pts, and with a
clock pad and `every_s` above 0 it also writes ``{"kind": "tick", "node":
<node>}`` at each multiple of `every_s` seconds. Its outputs are timed in
microseconds. Every expectation is read off that description and off
`tests/data/deal.nut`, whose three messages tests/exec/test_exec_data.py
lists: a two second picture at ten frames a second, and messages at 0, 0.4
and 1.2 seconds. A data filter's output also carries heartbeats, empty
packets saying how far time has got, which are no messages.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import time
from pathlib import Path

import pytest

from ffrwd import binaries, cli

pytestmark = pytest.mark.exec

_CLI_ROOT = Path(__file__).resolve().parent.parent.parent
_REPO_ROOT = _CLI_ROOT.parent
_SIDECAR_MODULES = _REPO_ROOT / "sidecar" / "modules"
_STAMP = _SIDECAR_MODULES / "target" / "wasm32-wasip2" / "release" / "data_stamp.wasm"
_DEAL = _CLI_ROOT / "tests" / "data" / "deal.nut"
_MESSAGES = [
    (0.0, {"kind": "break", "id": 1, "start_pts": 0.5, "duration": 1.0}),
    (0.4, {"kind": "award", "break": 1, "node": "root", "price": 11.42}),
    (1.2, {"kind": "break", "id": 2, "start_pts": 1.6, "duration": 0.3}),
]
# The picture's own span: ten frames a second, the last at 1.9 seconds.
_PICTURE_END = 2.0
_TIMEOUT = 120.0

_DECLARE = {
    "stamp": "CREATE FUNCTION stamp(d data_stream, node text, every_s number "
    "DEFAULT 0) RETURNS data_stream AS '{module}', 'data_stamp' LANGUAGE wasm;",
    "stamp_clock": "CREATE FUNCTION stamp_clock(clock video_stream, node text, "
    "every_s number DEFAULT 0) RETURNS data_stream "
    "AS '{module}', 'data_stamp' LANGUAGE wasm;",
    "stamp_both": "CREATE FUNCTION stamp_both(d data_stream, clock video_stream, "
    "node text, every_s number DEFAULT 0) RETURNS data_stream "
    "AS '{module}', 'data_stamp' LANGUAGE wasm;",
}


@pytest.fixture(autouse=True)
def _require_everything() -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg/ffprobe not found on PATH")
    if binaries.ffrwd_wasm_path() is None:
        pytest.skip("the ffrwd-wasm sidecar is not installed")
    if not _STAMP.exists():
        pytest.skip(
            f"module missing: {_STAMP} (cargo build --target wasm32-wasip2 "
            f"--release, from {_SIDECAR_MODULES})"
        )


def _run(query: str, out: Path) -> list[tuple[float, dict[str, object]]]:
    """Run `query`, its declarations ahead of it, and read back what it wrote."""
    declared = [
        text.format(module=_STAMP.as_posix())
        for name, text in _DECLARE.items()
        if f"{name}(" in query
    ]
    sql = "\n".join(
        [*declared, query.format(deal=_DEAL.as_posix(), out=out.as_posix())]
    )
    assert cli.main(["run", sql, "-y", "-q"]) == 0
    return _messages(out)


def _messages(path: Path) -> list[tuple[float, dict[str, object]]]:
    """Every message on `path`'s data stream: its time and its object. A
    heartbeat carries no bytes and is left out."""
    done = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "d",
            "-show_entries", "stream=codec_tag_string:packet=pts_time,data",
            "-show_data", "-of", "json", str(path),
        ],
        capture_output=True,
        text=True,
        timeout=_TIMEOUT,
        check=True,
    )
    parsed = json.loads(done.stdout)
    assert [s["codec_tag_string"] for s in parsed["streams"]] == ["JSON"]
    return [
        (float(packet["pts_time"]), json.loads(body))
        for packet in parsed["packets"]
        if (body := _hexdump_bytes(packet.get("data", "")))
    ]


def _hexdump_bytes(dump: str) -> bytes:
    """The bytes of ffprobe's `-show_data` hexdump."""
    body = bytearray()
    for line in dump.splitlines():
        line = line.strip()
        if line:
            body += bytes.fromhex(line.split(":", 1)[1][:41].replace(" ", ""))
    return bytes(body)


def _stamped(node: str) -> list[tuple[float, dict[str, object]]]:
    return [(at, {**message, "node": node}) for at, message in _MESSAGES]


def _ticks(found: list[tuple[float, dict[str, object]]], node: str, every: float) -> None:
    """The ticks among `found`: each at a multiple of `every` inside the
    picture, every such multiple after the first frame present."""
    ticks = [at for at, message in found if message == {"kind": "tick", "node": node}]
    assert all(math.isclose(at / every, round(at / every)) for at in ticks), ticks
    assert all(0 <= at <= _PICTURE_END for at in ticks), ticks
    inside = [k * every for k in range(1, math.ceil(_PICTURE_END / every))]
    assert all(any(math.isclose(at, want) for at in ticks) for want in inside), ticks


def test_a_data_filter_rewrites_every_message_at_its_own_time(tmp_path: Path) -> None:
    found = _run(
        "COPY (SELECT stamp(f.data[1], 'es') FROM input('{deal}') f) TO '{out}'",
        tmp_path / "stamped.nut",
    )
    assert found == _stamped("es")


def test_a_clock_alone_times_what_a_data_filter_writes(tmp_path: Path) -> None:
    found = _run(
        "COPY (SELECT stamp_clock(f.video[1], 'root', 0.5) FROM input('{deal}') f) "
        "TO '{out}'",
        tmp_path / "ticks.nut",
    )
    assert found and all(message["kind"] == "tick" for _, message in found)
    _ticks(found, "root", 0.5)


def test_a_data_pad_and_a_clock_together(tmp_path: Path) -> None:
    """The messages come through stamped, and the ticks fall between them:
    one stream in pts order."""
    found = _run(
        "COPY (SELECT stamp_both(f.data[1], f.video[1], 'es', 0.5) "
        "FROM input('{deal}') f) TO '{out}'",
        tmp_path / "both.nut",
    )
    assert [m for m in found if m[1].get("kind") != "tick"] == _stamped("es")
    _ticks(found, "es", 0.5)
    assert [at for at, _ in found] == sorted(at for at, _ in found)


def test_two_data_filters_chain(tmp_path: Path) -> None:
    """The second reads what the first wrote: the node it sets is the one
    that stays, on every message, at every message's own time."""
    found = _run(
        "COPY (SELECT stamp(stamp(f.data[1], 'root'), 'es') FROM input('{deal}') f) "
        "TO '{out}'",
        tmp_path / "chained.nut",
    )
    assert found == _stamped("es")


def test_the_messages_ride_beside_the_picture(tmp_path: Path) -> None:
    """A data filter's output muxed with a stream copied off the input keeps
    the times the module wrote."""
    out = tmp_path / "beside.nut"
    found = _run(
        "COPY (SELECT f.video[1], stamp(f.data[1], 'es') FROM input('{deal}') f) "
        "TO '{out}'",
        out,
    )
    assert found == _stamped("es")


def test_a_live_picture_beside_its_own_sparse_clock_runs_in_real_time(
    tmp_path: Path,
) -> None:
    """The picture and the ticks its own clock times, into one file, off a
    paced lavfi graph. The graph starts half a second in, so the first tick
    is 5 s away, as a head's first break is: the file's writer opens the
    data pipe on its first heartbeat rather than waiting those seconds with
    the clock's producer blocked behind it, and heartbeats keep the picture
    from waiting on each tick."""
    out = tmp_path / "live.nut"
    query = (
        "COPY (SELECT f.video[1], stamp_clock(f.video[1], 'root', 5) FROM "
        "input('testsrc2=size=320x240:rate=30:duration=20', format => 'lavfi', "
        "realtime => true, itsoffset => 0.5) f) TO '{out}'"
    )
    started = time.monotonic()
    found = _run(query, out)
    took = time.monotonic() - started
    assert took < 1.3 * 20, f"20 s of picture took {took:.1f} s"
    frames = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v", "-count_packets",
            "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(out),
        ],
        capture_output=True,
        text=True,
        timeout=_TIMEOUT,
        check=True,
    )
    assert int(frames.stdout.strip()) == 20 * 30
    tick = {"kind": "tick", "node": "root"}
    assert found == [(5.0, tick), (10.0, tick), (15.0, tick), (20.0, tick)]
