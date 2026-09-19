"""Exec tests for titled metadata tracks: written by the compiler, muxed by
real ffmpeg, and read back by the compiler's own probe.

Marked ``@pytest.mark.exec`` like the rest of the tier; needs ffmpeg/ffprobe
on PATH and ``tests/fixtures/described.mkv`` (``python
scripts/gen_fixtures.py``).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from ffrwd.compiler import compile_sql, compile_table_sql
from ffrwd.emit import build_ffmpeg_args, emit
from ffrwd.table import CellValue

pytestmark = pytest.mark.exec

_FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
_DESCRIBED = _FIXTURES_DIR / "described.mkv"

_SUBPROCESS_TIMEOUT = 60.0


@pytest.fixture(autouse=True)
def _require_tools() -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg/ffprobe not found on PATH")
    if not _DESCRIBED.exists():
        pytest.skip(f"fixture missing: {_DESCRIBED} (run scripts/gen_fixtures.py)")


def _path(path: Path) -> str:
    """`path` as a SQL string literal's body -- forward slashes everywhere."""
    return path.resolve().as_posix()


def _run(query: str, out_path: Path) -> None:
    args = build_ffmpeg_args(emit(compile_sql(query)), str(out_path))
    args.insert(1, "-y")
    result = subprocess.run(
        args, capture_output=True, text=True, timeout=_SUBPROCESS_TIMEOUT
    )
    assert result.returncode == 0, result.stderr
    assert out_path.exists()


def _rows(path: Path, column: str, columns: str) -> list[list[CellValue]]:
    """What the compiler reads back out of `path`, as its printed rows."""
    sinks = compile_table_sql(
        f"SELECT {columns} FROM input('{_path(path)}') f, unnest(f.{column}) r"
    )
    return sinks[0].result.rows


def _stream_tags(path: Path) -> list[dict[str, str]]:
    args = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "s",
        "-show_entries",
        "stream_tags",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(
        args, capture_output=True, text=True, timeout=_SUBPROCESS_TIMEOUT
    )
    assert result.returncode == 0, result.stderr
    streams: list[dict[str, object]] = json.loads(result.stdout)["streams"]
    tagged: list[dict[str, str]] = []
    for stream in streams:
        tags = stream.get("tags")
        tagged.append(
            {str(key).lower(): str(value) for key, value in tags.items()}
            if isinstance(tags, dict)
            else {}
        )
    return tagged


def test_a_titled_caption_track_reads_back_under_its_title(tmp_path: Path) -> None:
    out = tmp_path / "captioned.mkv"
    _run(
        "COPY (SELECT f.video[1], array_agg(STRUCT(c.text AS text, "
        "c.start_t AS start_t, c.end_t AS end_t)::cue) AS speech "
        f"FROM input('{_path(_DESCRIBED)}') f, unnest(f.cues) c "
        "GROUP BY f.video[1]) TO 'out.mkv'",
        out,
    )
    assert _stream_tags(out)[0].get("title") == "speech"
    assert _rows(out, "cues['speech']", "r.track, r.text") == _rows(
        _DESCRIBED, "cues", "r.track, r.text"
    )


def test_a_remux_through_ffmpeg_alone_keeps_the_tracks_and_their_tags(
    tmp_path: Path,
) -> None:
    """`ffmpeg -i out.mkv -c copy again.mkv` is what a user does next."""
    remuxed = tmp_path / "remuxed.mkv"
    result = subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(_DESCRIBED), "-map", "0", "-c",
         "copy", str(remuxed)],
        capture_output=True,
        text=True,
        timeout=_SUBPROCESS_TIMEOUT,
    )
    assert result.returncode == 0, result.stderr
    assert _stream_tags(remuxed) == _stream_tags(_DESCRIBED)
    assert _rows(remuxed, "cues", "r.track, r.text") == _rows(
        _DESCRIBED, "cues", "r.track, r.text"
    )
