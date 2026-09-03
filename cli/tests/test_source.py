"""End-to-end exec proof for a `RETURNS source` wasm function in FROM.

`source-replay` (the sidecar's own fleet module,
`sidecar/modules/source-replay`) is a packet source with no filesystem
input: it compiles in a fixed seven-packet h264-shaped stream (one
video track, 32x24, decode order I0 P3 B1 B2 P6 B4 B5, a reorder depth
of 2) and hands it out through the same `packet-source` wit interface
any packet source implements. `probe`/`open` both publish a one-row,
one-track, bounded catalog, so `FROM replay() s` reads it exactly like
a probed manifest's rendition rows -- `s.height`, `s.video[1]`, WHERE,
the one-row rule.

Real throughout: the sidecar runs the module for real (`ffrwd-wasm -m
... -f nut pipe:1`, no `-i` before it -- a packet source rides alone,
reading nothing), and the far ffmpeg decodes and re-encodes what it
wrote. Requires ffmpeg/ffprobe on PATH, the `ffrwd-wasm` sidecar (`uv
sync --extra wasm`), and `source-replay` built for `wasm32-wasip2`.
Tests skip cleanly when any of those is missing.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from ffrwd import binaries, cli
from ffrwd.compiler import compile_sql
from ffrwd.errors import ErrorCode, FfrwdError

_CLI_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = _CLI_ROOT.parent
_SIDECAR_MODULES = _REPO_ROOT / "sidecar" / "modules"
_MODULE = _SIDECAR_MODULES / "target" / "wasm32-wasip2" / "release" / "source_replay.wasm"

# The catalog `source-replay` publishes: one video track, 32x24, 7 coded
# packets -- see sidecar/modules/source-replay/src/lib.rs.
_HEIGHT = 24
_PACKET_COUNT = 7
_SUBPROCESS_TIMEOUT = 120.0


@pytest.fixture(autouse=True)
def _require_everything() -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg/ffprobe not found on PATH")
    if binaries.ffrwd_wasm_path() is None:
        pytest.skip("ffrwd-wasm not found (uv sync --extra wasm)")
    if not _MODULE.exists():
        pytest.skip(
            f"module missing: {_MODULE} (cargo build --target wasm32-wasip2 "
            f"--release, from {_SIDECAR_MODULES})"
        )


def _query(out_path: Path, *, where: str = "") -> str:
    return (
        "CREATE FUNCTION replay() RETURNS source\n"
        f"  AS '{_MODULE.as_posix()}', 'source_replay' LANGUAGE wasm;\n"
        "COPY (\n"
        "  SELECT s.video[1]\n"
        f"  FROM replay() s{where}\n"
        f") TO '{out_path.as_posix()}' WITH (video_codec 'libx264', crf 20)"
    )


def _video_streams(path: Path) -> list[dict[str, object]]:
    """Every video stream `path` carries, frame-counted."""
    done = subprocess.run(
        [
            "ffprobe", "-v", "error", "-count_frames",
            "-show_entries", "stream=codec_type,nb_read_frames,width,height",
            "-of", "json", str(path),
        ],
        capture_output=True,
        text=True,
        timeout=_SUBPROCESS_TIMEOUT,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    streams: list[dict[str, object]] = json.loads(done.stdout)["streams"]
    return [s for s in streams if s["codec_type"] == "video"]


@pytest.mark.exec
def test_a_replayed_source_decodes_and_reencodes_all_seven_packets(tmp_path: Path) -> None:
    """`FROM replay() s` reads the module's compiled-in stream like any
    input: the sidecar rides alone (no `-i` feeds it) and writes the seven
    packets as one NUT pipe, the far ffmpeg decodes and re-encodes them --
    one video stream, seven frames, none dropped or duplicated by the
    two-deep reorder the decode order needs."""
    out = tmp_path / "out.mp4"
    assert cli.main(["run", _query(out), "-y"]) == 0
    assert out.exists()

    video_streams = _video_streams(out)
    assert len(video_streams) == 1
    assert int(video_streams[0]["nb_read_frames"]) == _PACKET_COUNT
    assert int(video_streams[0]["height"]) == _HEIGHT


@pytest.mark.exec
def test_where_on_source_height_keeps_the_row_and_a_wrong_height_refuses(
    tmp_path: Path,
) -> None:
    """`WHERE s.height = 24` narrows the catalog's one row to itself, so
    `s.video[1]` still reads -- the same one-row rule a probed manifest's
    rendition rows already obey. A height nothing in the catalog carries
    narrows the row set to zero, and reading a stream off zero rows is a
    compile-time refusal, not a runtime one."""
    out = tmp_path / "out.mp4"
    assert cli.main(["run", _query(out, where=f" WHERE s.height = {_HEIGHT}"), "-y"]) == 0
    assert out.exists()
    video_streams = _video_streams(out)
    assert len(video_streams) == 1
    assert int(video_streams[0]["height"]) == _HEIGHT

    wrong_height = _HEIGHT + 1
    with pytest.raises(FfrwdError) as excinfo:
        compile_sql(_query(tmp_path / "wrong.mp4", where=f" WHERE s.height = {wrong_height}"))
    err = excinfo.value
    assert err.code is ErrorCode.STREAM_NOT_FOUND
    assert err.message == "'s.video[1]' does not exist: 0 rows carry a video track"


@pytest.mark.exec
def test_compile_prints_the_source_riding_alone(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """A packet source reads nothing, so its sidecar is fed no `-i` at
    all -- `compile`'s printed pipeline puts `-f nut` on the sidecar's own
    output and no `-i` anywhere before its `-m`, the mirror of every other
    module-bearing recipe where the sidecar instead reads a decoded pipe."""
    out = tmp_path / "out.mp4"
    code = cli.main(["compile", _query(out)])
    printed = capsys.readouterr().out
    assert code == 0

    sidecar_segment = printed.split(" | ", 1)[0]
    assert "-m " in sidecar_segment
    assert "-f nut" in sidecar_segment
    assert "-i" not in sidecar_segment
