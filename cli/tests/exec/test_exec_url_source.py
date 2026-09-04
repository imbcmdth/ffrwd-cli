"""End-to-end exec proof for a URL source (recipes 114-115 in
`docs/examples.md`): `RETURNS source` over a values-world module, invoked
once at compile time. Each row it answers names a `url`; the compiler mints
that url its own `-i` and probes it like any `input()` path, so the output
of these queries is a single, ordinary ffmpeg command with no sidecar
involved at run time.

Marked ``@pytest.mark.exec`` and excluded from the default run, same as
``test_exec.py``. Run explicitly::

    python -m pytest -m exec tests/exec/test_exec_url_source.py -q

Requires ``ffmpeg``/``ffprobe`` on PATH, the generated fixtures, and the
``source_files`` module built for ``wasm32-wasip2`` (``cargo build --target
wasm32-wasip2 --release -p source-files``, from ``sidecar/modules``). Tests
skip cleanly when any of those is missing. The module itself is invoked
through ``ffrwd-wasm --invoke`` (see ``ffrwd/wasm.py``), so the sidecar
binary (``uv sync --extra wasm``, or ``FFRWD_WASM``) has to be findable too.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from ffrwd import binaries
from ffrwd.compiler import compile_sql
from ffrwd.emit import build_ffmpeg_args, emit

pytestmark = pytest.mark.exec

_CLI_ROOT = Path(__file__).resolve().parent.parent.parent
_REPO_ROOT = _CLI_ROOT.parent
_FIXTURES_DIR = _CLI_ROOT / "tests" / "fixtures"
_AV = _FIXTURES_DIR / "av.mp4"
_AV2 = _FIXTURES_DIR / "av2.mp4"
_TESTSRC = _FIXTURES_DIR / "testsrc.mp4"
_MODULE = (
    _REPO_ROOT / "sidecar" / "modules" / "target" / "wasm32-wasip2" / "release"
    / "source_files.wasm"
)

_SUBPROCESS_TIMEOUT = 60.0


def _path(path: Path) -> str:
    """`path` as a SQL string-literal body: forward slashes, absolute."""
    return path.resolve().as_posix()


def _declare_files() -> str:
    return (
        "CREATE FUNCTION files(paths text) RETURNS source\n"
        f"  AS '{_path(_MODULE)}', 'files' LANGUAGE wasm;\n"
    )


@pytest.fixture(autouse=True)
def _require_everything() -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg/ffprobe not found on PATH")
    for fixture in (_AV, _AV2, _TESTSRC):
        if not fixture.exists():
            pytest.skip(f"fixture missing: {fixture} (run scripts/gen_fixtures.py first)")
    if binaries.ffrwd_wasm_path() is None:
        pytest.skip("ffrwd-wasm not found (uv sync --extra wasm)")
    if not _MODULE.exists():
        pytest.skip(
            f"module missing: {_MODULE} (cargo build --target wasm32-wasip2 "
            "--release -p source-files, from sidecar/modules)"
        )


def _run(query: str, out_path: Path) -> None:
    args = build_ffmpeg_args(emit(compile_sql(query)), str(out_path))
    args.insert(1, "-y")
    result = subprocess.run(
        args, capture_output=True, text=True, timeout=_SUBPROCESS_TIMEOUT
    )
    assert result.returncode == 0, result.stderr


def _ffprobe_duration(path: Path) -> float:
    args = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "json", str(path),
    ]
    result = subprocess.run(
        args, capture_output=True, text=True, timeout=_SUBPROCESS_TIMEOUT
    )
    assert result.returncode == 0, result.stderr
    return float(json.loads(result.stdout)["format"]["duration"])


def _ffprobe_stream_types(path: Path) -> list[str]:
    args = [
        "ffprobe", "-v", "error", "-show_entries", "stream=codec_type",
        "-of", "json", str(path),
    ]
    result = subprocess.run(
        args, capture_output=True, text=True, timeout=_SUBPROCESS_TIMEOUT
    )
    assert result.returncode == 0, result.stderr
    streams = json.loads(result.stdout)["streams"]
    return [s["codec_type"] for s in streams]


# ---------------------------------------------------------------------------
# recipe 114 -- a list of files as one relation, stitched in row order
# ---------------------------------------------------------------------------


def test_url_source_114_stitches_files_into_one_output(tmp_path: Path) -> None:
    """`files('av.mp4,av2.mp4')` mints two hidden `-i`s in row order; the
    `concat(VARIADIC array_agg(...))` over each stitches them into one
    output whose duration is the sum of the two, one video and one audio
    track."""
    query = (
        _declare_files()
        + "COPY (\n"
        "  SELECT concat(VARIADIC array_agg(s.video[1])), "
        "concat(VARIADIC array_agg(s.audio[1]))\n"
        f"  FROM files('{_path(_AV)},{_path(_AV2)}') s\n"
        f") TO '{_path(tmp_path / 'joined.mp4')}'"
    )
    out_path = tmp_path / "joined.mp4"

    _run(query, out_path)

    assert out_path.exists()
    assert _ffprobe_stream_types(out_path) == ["video", "audio"]
    assert _ffprobe_duration(out_path) == pytest.approx(
        _ffprobe_duration(_AV) + _ffprobe_duration(_AV2), abs=0.1
    )


# ---------------------------------------------------------------------------
# recipe 115 -- WHERE drops a row before the stitch, then re-encode to a
# ladder
# ---------------------------------------------------------------------------


def test_url_source_115_stitches_kept_rows_then_re_encodes_to_a_ladder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`WHERE s.sequence != 2` drops `testsrc.mp4` (row 2) before the
    `concat`, so only `av.mp4` and `av2.mp4` are stitched; the result is
    re-encoded to a two-rung HLS ladder. The master and both variant
    playlists exist, and the top rung's own duration (the 800k/320-wide
    rung, `v0/index.m3u8`) is the sum of the two kept files' durations --
    proof `testsrc.mp4` never reached the command at all.

    Runs from `tmp_path` with a relative destination: ffmpeg's hls muxer
    cannot create `%v` directories on a drive other than the working one
    (Windows), same as the manifest-ladder exec tests in test_exec.py.
    """
    monkeypatch.chdir(tmp_path)
    dest = tmp_path / "out" / "master.m3u8"
    dest.parent.mkdir()
    query = (
        _declare_files()
        + "COPY (\n"
        "  WITH pod AS (\n"
        "    SELECT concat(VARIADIC array_agg(scale(s.video[1], 320, -2))) AS v,\n"
        "           concat(VARIADIC array_agg(s.audio[1])) AS a\n"
        f"    FROM files('{_path(_AV)},{_path(_TESTSRC)},{_path(_AV2)}') s\n"
        "    WHERE s.sequence != 2\n"
        "  ),\n"
        "  vid AS (\n"
        "    SELECT scale(pod.v, ARRAY[320, 160][i.i], -2) AS v, i.i AS rung\n"
        "    FROM pod, generate_series(1, 2) i\n"
        "  ),\n"
        "  aud AS (\n"
        "    SELECT pod.a AS t, 3 AS rung\n"
        "    FROM pod\n"
        "  )\n"
        "  SELECT vid.v, aud.t\n"
        "  FROM vid FULL JOIN aud ON vid.rung = aud.rung\n"
        ") TO 'out/master.m3u8'\n"
        "  WITH (format 'hls', hls_time 2, hls_playlist_type 'vod',\n"
        "        video_codec 'libx264', video_bitrate ARRAY['800k', '300k'][vid.rung],\n"
        "        audio_codec 'aac')"
    )

    graph = compile_sql(query)
    printed = " ".join(build_ffmpeg_args(emit(graph)))
    assert "testsrc" not in printed

    args = build_ffmpeg_args(emit(graph))
    args.insert(1, "-y")
    result = subprocess.run(
        args, capture_output=True, text=True, timeout=_SUBPROCESS_TIMEOUT
    )
    assert result.returncode == 0, result.stderr

    assert dest.exists()
    top_rung = dest.parent / "v0" / "index.m3u8"
    other_rung = dest.parent / "v1" / "index.m3u8"
    assert top_rung.exists()
    assert other_rung.exists()
    assert _ffprobe_duration(top_rung) == pytest.approx(
        _ffprobe_duration(_AV) + _ffprobe_duration(_AV2), abs=0.2
    )
