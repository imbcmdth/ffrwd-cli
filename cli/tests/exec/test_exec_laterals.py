"""End-to-end exec tests for a run-time lateral: an instance of a sql table
function started per message of a data stream, writing a feeder.

Marked ``@pytest.mark.exec`` and excluded from the default run. Run
explicitly::

    python -m pytest -m exec tests/exec/test_exec_laterals.py -q

`tests/data/launch.nut` is a data stream of launch messages and nothing
else, committed because no ffmpeg on its own writes JSON messages. It was
made with ffrwd-nut's `json_nut` example, fed one message a line as its pts
in microseconds, a tab, and the payload::

    0         (a heartbeat: one space)
    100000    {"url": "ad.mkv", "start_pts": 1.0, "duration": 1.0}
    200000    {"url": "ad.mkv", "start_pts": 1.5, "duration": 1.0}
    300000    {"start_pts": 2.5, "duration": 1.0}
    400000    (a heartbeat: one space)
    500000    {"url": "ad.mkv", "start_pts": 3.0, "duration": 1.0}

    cargo run --example json_nut < launch.txt > launch.nut     # in ffrwd-nut

The first and last messages each start an instance; the second would play
over the first and is refused; the third names no url and is refused. The
url is relative, so each test runs where it writes `ad.mkv`: a second of a
160x120 picture at 15 frames a second and of 48 kHz mono sound. The
programme is six seconds read in real time, long enough for both instances
to run while its module still reads the feeder. The modules reading the
feeder are the fleet's ``feed-probe`` and ``feed-probe-audio``, which write
a row per picture, or per packet of sound, the feeder sends.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from ffrwd import binaries, cli

pytestmark = pytest.mark.exec

_CLI_ROOT = Path(__file__).resolve().parent.parent.parent
_RELEASE = _CLI_ROOT.parent / "sidecar" / "modules" / "target" / "wasm32-wasip2" / "release"
_PROBE = _RELEASE / "feed_probe.wasm"
_HEAR = _RELEASE / "feed_probe_audio.wasm"
_STAMP = _RELEASE / "data_stamp.wasm"
_LAUNCH = _CLI_ROOT / "tests" / "data" / "launch.nut"
_TIMEOUT = 120.0

_PLAY = """CREATE FUNCTION play(launch data_stream, url text, start_pts number,
                     duration number, width number, height number, fps number,
                     pix_fmt text)
RETURNS TABLE(video video_stream) AS $$
  SELECT setpts(ffmpeg.format(fps(scale(m.video[1], width, height), fps),
                              pix_fmts => pix_fmt),
                'PTS+' || start_pts::text || '/TB') AS video,
         STRUCT('1' AS smart_timed) AS tags
  FROM input(url) m
$$ LANGUAGE sql;
"""

_HEAR_PLAY = """CREATE FUNCTION play(launch data_stream, url text, start_pts number,
                     rate number, channels number, sample_fmt text)
RETURNS TABLE(audio audio_stream) AS $$
  SELECT asetpts(ffmpeg.aformat(aresample(m.audio[1], rate), sample_fmts => sample_fmt,
                                channel_layouts => channels::text || 'c'),
                 'PTS+' || start_pts::text || '/TB') AS audio,
         STRUCT('1' AS smart_timed) AS tags
  FROM input(url) m
$$ LANGUAGE sql;
"""


@pytest.fixture(autouse=True)
def _require_everything() -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg/ffprobe not found on PATH")
    if binaries.ffrwd_wasm_path() is None:
        pytest.skip("the ffrwd-wasm sidecar is not installed")
    for module in (_PROBE, _HEAR):
        if not module.exists():
            pytest.skip(
                f"module missing: {module} (cargo build --target wasm32-wasip2 "
                "--release, in sidecar/modules)"
            )


def _ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True, timeout=_TIMEOUT)


def _media(where: Path) -> Path:
    """The ad every instance plays, and the programme; the programme's path."""
    _ffmpeg(
        "-f", "lavfi", "-i", "testsrc2=size=160x120:rate=15:duration=1",
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=1",
        "-c:v", "ffv1", "-c:a", "pcm_s16le", str(where / "ad.mkv"),
    )
    programme = where / "programme.mkv"
    _ffmpeg(
        "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=15:duration=6",
        "-f", "lavfi", "-i", "sine=frequency=220:sample_rate=48000:duration=6",
        "-c:v", "ffv1", "-c:a", "pcm_s16le", "-ac", "2", str(programme),
    )
    return programme


def _run(sql: str, capsys: pytest.CaptureFixture[str]) -> list[dict[str, object]]:
    """Run `sql`, and the rows it said about its instances."""
    capsys.readouterr()
    assert cli.main(["run", sql, "-y", "-q"]) == 0
    out = capsys.readouterr().out
    return [json.loads(line) for line in out.splitlines() if line.startswith("{")]


def _check_feeder_rows(rows: list[dict[str, object]]) -> None:
    """One row per message, heartbeats none: the first and last ran, the
    second overlapped the first and the third named no url."""
    by_row = {row["row"]: row for row in rows}
    assert sorted(by_row) == [1, 2, 3, 4] and len(rows) == 4
    assert all(row["event"] == "feeder" for row in rows)
    assert (by_row[1]["start_pts"], by_row[1]["exit"]) == (1.0, 0)
    assert (by_row[4]["start_pts"], by_row[4]["exit"]) == (3.0, 0)
    assert "over the instance of message 1" in str(by_row[2]["refused"])
    assert "nothing binds 'url'" in str(by_row[3]["refused"])


def test_each_message_starts_an_instance_whose_frames_reach_the_feeder(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    programme = _media(tmp_path)
    monkeypatch.chdir(tmp_path)
    dump = tmp_path / "dump"
    monkeypatch.setenv("FFRWD_DUMP_STDERR", str(dump))
    probed = tmp_path / "probed.ndjson"
    sql = (
        "CREATE FUNCTION probe(v video_stream, feed video_stream DEFAULT NULL, "
        "port number DEFAULT 9000)\nRETURNS STRUCT(v video_stream, feeds "
        "STRUCT(feed_pts number, w number, h number)[])\n"
        f"AS '{_PROBE.as_posix()}', 'feed-probe' LANGUAGE wasm;\n" + _PLAY
        + "COPY (SELECT probe(p.video[1], ad.video).feeds FROM "
        f"input('{programme.as_posix()}', realtime => true) p, "
        f"input('{_LAUNCH.as_posix()}') f, LATERAL play(f.data[1]) ad) "
        f"TO '{probed.as_posix()}'"
    )
    _check_feeder_rows(_run(sql, capsys))
    fed = [json.loads(line) for line in probed.read_text().splitlines() if line]
    # Each instance's second of pictures, conformed to the programme's size,
    # the second stamped two seconds after the first, as its start_pts says.
    assert len(fed) == 30
    assert {(row["w"], row["h"]) for row in fed} == {(320, 240)}
    pts = [int(row["feed_pts"]) for row in fed]
    assert pts == sorted(pts)
    assert pts[15] == 3 * pts[0]
    # Each instance's members wrote their stderr where the run's did, and the
    # tag the body sets reached what it wrote.
    for row in (1, 4):
        (log,) = dump.glob(f"feeder{row}.*.stderr")
        assert "smart_timed" in log.read_text(encoding="utf-8")


def test_an_instance_of_sound_reaches_a_sound_feeder_in_its_format(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """``feed-probe-audio`` reads f32 at 48 kHz in stereo, and every sample of
    each instance's second of mono sound arrives in that format."""
    programme = _media(tmp_path)
    monkeypatch.chdir(tmp_path)
    heard = tmp_path / "heard.ndjson"
    sql = (
        "CREATE FUNCTION hear(a audio_stream, feed audio_stream DEFAULT NULL, "
        "port number DEFAULT 9000)\nRETURNS STRUCT(a audio_stream, feeds "
        "STRUCT(feed_pts number, rate number, channels number, sample_fmt text, "
        "samples number)[])\n"
        f"AS '{_HEAR.as_posix()}', 'feed-probe-audio' LANGUAGE wasm;\n" + _HEAR_PLAY
        + "COPY (SELECT hear(p.audio[1], ad.audio).feeds FROM "
        f"input('{programme.as_posix()}', realtime => true) p, "
        f"input('{_LAUNCH.as_posix()}') f, LATERAL play(f.data[1]) ad) "
        f"TO '{heard.as_posix()}'"
    )
    _check_feeder_rows(_run(sql, capsys))
    fed = [json.loads(line) for line in heard.read_text().splitlines() if line]
    assert {(row["rate"], row["channels"], row["sample_fmt"]) for row in fed} == {
        (48000, 2, "f32")
    }
    assert sum(int(row["samples"]) for row in fed) == 2 * 48000


def _data_messages(path: Path) -> list[tuple[float, dict[str, object]]]:
    """Every message on `path`'s data stream, heartbeats left out."""
    done = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "d", "-show_entries",
            "packet=pts_time,data", "-show_data", "-of", "json", str(path),
        ],
        capture_output=True,
        text=True,
        timeout=_TIMEOUT,
        check=True,
    )
    found = []
    for packet in json.loads(done.stdout)["packets"]:
        body = b"".join(
            bytes.fromhex(line.split(":", 1)[1][:41].replace(" ", ""))
            for line in packet.get("data", "").splitlines()
            if line.strip()
        )
        if body.strip():
            found.append((float(packet["pts_time"]), json.loads(body)))
    return found


def test_a_data_filters_output_reaches_the_lateral_and_the_file_both(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The launch messages pass through ``data-stamp`` first, and its one
    output is read twice: by the lateral, which starts an instance per message
    as ever, and by the file, which gets every message with its node set."""
    if not _STAMP.exists():
        pytest.skip(f"module missing: {_STAMP}")
    programme = _media(tmp_path)
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "out.nut"
    sql = (
        "CREATE FUNCTION probe(v video_stream, feed video_stream DEFAULT NULL, "
        "port number DEFAULT 9000) RETURNS video_stream "
        f"AS '{_PROBE.as_posix()}', 'feed-probe' LANGUAGE wasm;\n"
        "CREATE FUNCTION stamp(d data_stream, node text) RETURNS data_stream "
        f"AS '{_STAMP.as_posix()}', 'data_stamp' LANGUAGE wasm;\n" + _PLAY
        + "COPY (WITH w AS (SELECT stamp(f.data[1], 'leaf') AS s "
        f"FROM input('{_LAUNCH.as_posix()}') f), "
        "ads AS (SELECT ad.video FROM w, LATERAL play(w.s) ad) "
        "SELECT probe(p.video[1], ads.video), w.s "
        f"FROM input('{programme.as_posix()}', realtime => true) p, ads, w) "
        f"TO '{out.as_posix()}'"
    )
    _check_feeder_rows(_run(sql, capsys))
    written = _data_messages(out)
    assert [at for at, _ in written] == [0.1, 0.2, 0.3, 0.5]
    assert {message["node"] for _, message in written} == {"leaf"}
