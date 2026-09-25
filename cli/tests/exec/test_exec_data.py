"""End-to-end exec tests for a data stream of JSON messages.

Marked ``@pytest.mark.exec`` and excluded from the default run. Run
explicitly::

    python -m pytest -m exec tests/exec/test_exec_data.py -q

`tests/data/deal.nut` is a two second 64x48 h264 picture and a data stream
of three deal messages, at 0, 0.4 and 1.2 seconds. It is committed rather
than generated, since no ffmpeg on its own writes JSON messages at chosen
times. It was made with ffrwd-nut's `json_nut` example, fed `deal.txt`, one
message a line as its pts in microseconds, a tab, and the object::

    0         {"kind":"break","id":1,"start_pts":0.5,"duration":1.0}
    400000    {"kind":"award","break":1,"node":"root","price":11.42}
    1200000   {"kind":"break","id":2,"start_pts":1.6,"duration":0.3}

and one ffmpeg merge::

    cargo run --example json_nut < deal.txt > deal-only.nut     # in ffrwd-nut
    ffmpeg -f lavfi -i testsrc2=size=64x48:rate=10:duration=2 \
           -c:v libx264 -bf 0 -g 10 -pix_fmt yuv420p video.mp4
    ffmpeg -i video.mp4 -copyts -f nut -i deal-only.nut \
           -map 0:v -map 1:d -c copy -f nut deal.nut
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from ffrwd import cli

pytestmark = pytest.mark.exec

_CLI_ROOT = Path(__file__).resolve().parent.parent.parent
_DEAL = _CLI_ROOT / "tests" / "data" / "deal.nut"
_MESSAGES = [
    (0.0, {"kind": "break", "id": 1, "start_pts": 0.5, "duration": 1.0}),
    (0.4, {"kind": "award", "break": 1, "node": "root", "price": 11.42}),
    (1.2, {"kind": "break", "id": 2, "start_pts": 1.6, "duration": 0.3}),
]


@pytest.fixture(autouse=True)
def _require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg/ffprobe not found on PATH")


def _messages(path: Path) -> list[tuple[float, dict[str, object]]]:
    """Every message on `path`'s data stream: its time and its object."""
    done = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "d",
            "-show_entries", "stream=codec_tag_string:packet=pts_time,data",
            "-show_data", "-of", "json", str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    parsed = json.loads(done.stdout)
    assert [s["codec_tag_string"] for s in parsed["streams"]] == ["JSON"]
    out = []
    for packet in parsed["packets"]:
        out.append((float(packet["pts_time"]), json.loads(_hexdump_bytes(packet["data"]))))
    return out


def _hexdump_bytes(dump: str) -> bytes:
    """The bytes of ffprobe's `-show_data` hexdump."""
    body = bytearray()
    for line in dump.splitlines():
        line = line.strip()
        if not line:
            continue
        # "00000000: 7b22 6b69 ...  {"ki..."
        hexpart = line.split(":", 1)[1][:41]
        body += bytes.fromhex(hexpart.replace(" ", ""))
    return bytes(body)


def test_a_json_data_stream_is_copied_into_nut_with_its_times(tmp_path: Path) -> None:
    out = tmp_path / "copy.nut"
    sql = (
        f"COPY (SELECT f.video[1], f.data[1] FROM input('{_DEAL.as_posix()}') f) "
        f"TO '{out.as_posix()}'"
    )
    assert cli.main(["run", sql, "-y", "-q"]) == 0
    assert _messages(out) == _MESSAGES


def test_a_json_data_stream_is_copied_on_its_own(tmp_path: Path) -> None:
    out = tmp_path / "deal-only.nut"
    sql = (
        f"COPY (SELECT f.data[1] FROM input('{_DEAL.as_posix()}') f) "
        f"TO '{out.as_posix()}'"
    )
    assert cli.main(["run", sql, "-y", "-q"]) == 0
    assert _messages(out) == _MESSAGES
