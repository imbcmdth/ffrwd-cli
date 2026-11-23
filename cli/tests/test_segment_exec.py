"""A destination cut into files, run through real ffmpeg: each file a
fragmented MP4 that starts on a keyframe at zero, and the list naming each
file as the muxer finishes it. Reads the ``keys.mp4`` fixture, 4 s of
320x240 at 15 frames a second with a keyframe every 7 frames and a tone.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from ffrwd import cli

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = PROJECT_ROOT / "tests" / "fixtures"


@pytest.fixture(scope="module")
def _fixtures() -> None:
    subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "gen_fixtures.py")],
        check=True,
    )


def _boxes(path: Path) -> list[str]:
    """The types of a file's top-level boxes, in order."""
    data = path.read_bytes()
    types: list[str] = []
    at = 0
    while at + 8 <= len(data):
        size = int.from_bytes(data[at : at + 4], "big")
        types.append(data[at + 4 : at + 8].decode("latin-1"))
        if size < 8:
            break
        at += size
    return types


def _first_frames(path: Path) -> list[dict[str, object]]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_entries",
            "packet=stream_index,pts_time,flags",
            "-read_intervals",
            "%+#4",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    packets: list[dict[str, object]] = json.loads(result.stdout)["packets"]
    return packets


@pytest.mark.exec
def test_a_cut_copy_writes_fragmented_files_from_a_keyframe_at_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _fixtures: None
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "rec").mkdir()
    keys = (FIXTURES_DIR / "keys.mp4").as_posix()
    sql = (
        f"COPY (SELECT s.* FROM input('{keys}') s) TO 'rec/part-%03d.mp4' "
        "WITH (format 'segment', segment_time 1, segment_list 'rec/list.csv', "
        "movflags '+frag_keyframe+empty_moov+default_base_moof')"
    )
    assert cli.main(["run", sql, "-y"]) == 0
    parts = sorted((tmp_path / "rec").glob("part-*.mp4"))
    assert len(parts) >= 3
    listed = (tmp_path / "rec" / "list.csv").read_text(encoding="utf-8").splitlines()
    assert [line.split(",")[0] for line in listed] == [part.name for part in parts]
    for part in parts:
        assert "moof" in _boxes(part), part.name
        video = [one for one in _first_frames(part) if one["stream_index"] == 0]
        assert video, part.name
        assert "K" in str(video[0]["flags"]), part.name
        assert abs(float(str(video[0]["pts_time"]))) < 0.2, part.name
