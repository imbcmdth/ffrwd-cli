"""Exec tests for titled metadata tracks: written by the compiler, muxed by
real ffmpeg, and read back by the compiler's own probe.

Marked ``@pytest.mark.exec`` like the rest of the tier; needs ffmpeg/ffprobe
on PATH and ``tests/fixtures/described.mkv`` (``python
scripts/gen_fixtures.py``). The written-from-scratch half also needs the
sidecar and a built module, and skips without them, the same courtesy the
cookbook harness extends.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from ffrwd import binaries
from ffrwd.compiler import compile_sql, compile_table_sql
from ffrwd.emit import build_ffmpeg_args, emit
from ffrwd.table import CellValue, VectorCell

pytestmark = pytest.mark.exec

_FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
_DESCRIBED = _FIXTURES_DIR / "described.mkv"
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_FAUXLATE = (
    _PROJECT_ROOT
    / ".."
    / "sidecar"
    / "modules"
    / "target"
    / "wasm32-wasip2"
    / "release"
    / "fauxlate.wasm"
)

_SUBPROCESS_TIMEOUT = 60.0
# Two f32 values agree if they agree to this: the payload is exact, so the
# tolerance only covers the float/repr round trip.
_TOLERANCE = 1e-6


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


# One vector row, read back: its track, its bounds, and its numbers.
_VectorRow = tuple[CellValue, CellValue, CellValue, tuple[float, ...]]


def _vectors(path: Path) -> list[_VectorRow]:
    """Every vector row of `path`: its track, its span, and its numbers."""
    sinks = compile_table_sql(
        "SELECT v.track, v.start_t, v.end_t, v.vector "
        f"FROM input('{_path(path)}') f, unnest(f.embeddings) v"
    )
    rows: list[_VectorRow] = []
    for track, start, end, cell in sinks[0].result.rows:
        assert isinstance(cell, VectorCell), cell
        rows.append((track, start, end, cell.values))
    return rows


def _same_vectors(written: list[_VectorRow], read: list[_VectorRow]) -> None:
    assert [row[:3] for row in written] == [row[:3] for row in read]
    for left, right in zip(written, read):
        assert len(left[3]) == len(right[3])
        assert all(abs(a - b) <= _TOLERANCE for a, b in zip(left[3], right[3]))


def test_a_files_vector_rows_survive_being_written_into_another(
    tmp_path: Path,
) -> None:
    """Read -> write -> read: the same numbers, spans and title come back."""
    out = tmp_path / "again.mkv"
    _run(
        "COPY (SELECT f.video[1], array_agg(STRUCT(v.start_t AS start_t, "
        "v.end_t AS end_t, v.vector AS vector)::embedding) AS clip_vectors "
        f"FROM input('{_path(_DESCRIBED)}') f, unnest(f.embeddings) v "
        "GROUP BY f.video[1]) TO 'out.mkv'",
        out,
    )
    _same_vectors(_vectors(_DESCRIBED), _vectors(out))
    assert {"title": "clip_vectors", "vector_dims": "8"}.items() <= _stream_tags(out)[
        0
    ].items()


def test_a_titled_caption_track_reads_back_under_its_title(tmp_path: Path) -> None:
    out = tmp_path / "captioned.mkv"
    _run(
        "COPY (SELECT f.video[1], array_agg(STRUCT(c.text AS text, "
        "c.start_t AS start_t, c.end_t AS end_t)::cue) AS speech "
        f"FROM input('{_path(_DESCRIBED)}') f, unnest(f.cues) c "
        "GROUP BY f.video[1]) TO 'out.mkv'",
        out,
    )
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
    _same_vectors(_vectors(_DESCRIBED), _vectors(remuxed))
    assert _rows(remuxed, "cues", "r.track, r.text") == _rows(
        _DESCRIBED, "cues", "r.track, r.text"
    )


def test_a_vector_track_written_from_an_embedder_reads_back(tmp_path: Path) -> None:
    """The other direction: vectors a module computed at compile time."""
    if binaries.ffrwd_wasm_path() is None:
        pytest.skip("ffrwd-wasm not found (uv sync --extra wasm)")
    if not _FAUXLATE.exists():
        pytest.skip(f"module missing: {_FAUXLATE}")
    out = tmp_path / "embedded.mkv"
    _run(
        f"CREATE FUNCTION embed_text(prompt text) RETURNS vector AS "
        f"'{_path(_FAUXLATE)}', 'embed_text' LANGUAGE wasm;\n"
        "COPY (SELECT f.video[1], array_agg(STRUCT(r.start_t AS start_t, "
        "r.end_t AS end_t, embed_text(r.line) AS vector)::embedding) AS lines "
        f"FROM input('{_path(_DESCRIBED)}') f, unnest(ARRAY["
        "STRUCT('a cat sat on the mat' AS line, 0 AS start_t, 1.5 AS end_t), "
        "STRUCT('a dog ran in the yard' AS line, 1.5 AS start_t, 3 AS end_t)"
        "]) r GROUP BY f.video[1]) TO 'out.mkv'",
        out,
    )
    rows = _vectors(out)
    assert [row[:3] for row in rows] == [("lines", 0.0, 1.5), ("lines", 1.5, 3.0)]
    assert len({row[3] for row in rows}) == 2
    assert {"title": "lines", "vector_dims": str(len(rows[0][3]))}.items() <= (
        _stream_tags(out)[0].items()
    )
