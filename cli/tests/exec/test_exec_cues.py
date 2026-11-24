"""End-to-end exec tests for cue rows a source writes beside its picture.

Marked ``@pytest.mark.exec`` and excluded from the default run. Run explicitly::

    python -m pytest -m exec tests/exec/test_exec_cues.py -q

The source is the ``speaker`` module: a 320x180 picture and the cue rows a
streaming transcriber writes, an utterance every 12 s, sent growing and then
whole 3 s after it ends. Its picture and its cues reach ffmpeg over two pipes
from one sidecar, as a live program's do from ``ffrwd.moq.subscribe``. ffmpeg
reads no input more than a tenth of a second ahead of an output's
furthest-behind stream, so between two utterances the cue stream would hold
the picture back until its pipe filled and the sidecar could write no more:
the run would end wedged, reported as an overflow. The sidecar's heartbeats keep the cue stream
level with the picture, and the output drops them before its muxer.

Requires ``ffmpeg``/``ffprobe`` on PATH with libx264, the ``ffrwd-wasm``
sidecar, and the sidecar fleet's ``speaker`` module built for
``wasm32-wasip2``. Tests skip cleanly when any of those is missing.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from ffrwd import binaries, wasm
from ffrwd.compiler import compile_all
from ffrwd.execute import execute_plan

pytestmark = pytest.mark.exec

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_SPEAKER = (
    _REPO_ROOT / "sidecar" / "modules" / "target" / "wasm32-wasip2" / "release" / "speaker.wasm"
)
_TIMEOUT = 120.0
_FRAME = 1 / 15
# The utterances said in 40 s whose whole rows come before the end: each at
# its own start, by who says it.
_SAID = {0.25: "Ann", 12.25: "Bob", 24.25: "Ann"}
_TIMING = re.compile(r"((?:\d+:)?\d+:\d+\.\d+) --> ((?:\d+:)?\d+:\d+\.\d+)")


@pytest.fixture(autouse=True)
def _require_tools() -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg/ffprobe not found on PATH")
    if binaries.ffrwd_wasm_path() is None:
        pytest.skip("ffrwd-wasm not found")
    if not _SPEAKER.exists():
        pytest.skip(f"module missing: {_SPEAKER} (cargo build --target wasm32-wasip2 --release)")


def _run(destination: str, with_block: str) -> None:
    sql = (
        "CREATE FUNCTION speaker(fps number DEFAULT 15, every number DEFAULT 12,\n"
        "                        late number DEFAULT 0, seconds number DEFAULT 0)\n"
        f"  RETURNS source AS '{_SPEAKER.as_posix()}', 'speaker' LANGUAGE wasm;\n"
        "COPY (\n"
        "  SELECT s.video[1], s.data[1] AS captions\n"
        "  FROM speaker(late => 3, seconds => 40) s\n"
        f") TO '{destination}' WITH ({with_block})"
    )
    compiled = compile_all(sql)
    assert compiled.plan is not None
    result = execute_plan(
        compiled.plan, sidecar_argv=wasm.sidecar_argv, overwrite=True, timeout=_TIMEOUT
    )
    assert result.overflow is None, result.overflow
    assert result.exit_code == 0, "\n".join(
        f"{member.id} exited {member.exit_code}: {member.stderr_tail}"
        for stage in result.stages
        for member in stage.members
    )


def _seconds(stamp: str) -> float:
    parts = [float(part) for part in stamp.split(":")]
    return sum(part * 60**power for power, part in enumerate(reversed(parts)))


def _cues(webvtt: str) -> list[tuple[float, float, str]]:
    """Each cue of a WebVTT document: its start, its end and its text."""
    cues: list[tuple[float, float, str]] = []
    for block in webvtt.split("\n\n"):
        lines = block.strip().splitlines()
        timing = next((line for line in lines if _TIMING.search(line)), None)
        if timing is None:
            continue
        found = _TIMING.search(timing)
        assert found is not None
        text = "\n".join(lines[lines.index(timing) + 1 :])
        cues.append((_seconds(found[1]), _seconds(found[2]), text))
    return cues


def _first_picture(path: Path) -> float:
    """The first picture's presentation time in `path`, in seconds."""
    args = [
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-read_intervals", "%+#1",
        "-show_entries", "packet=pts_time", "-of", "json", str(path),
    ]
    result = subprocess.run(args, capture_output=True, text=True, timeout=_TIMEOUT)
    assert result.returncode == 0, result.stderr
    return float(json.loads(result.stdout)["packets"][0]["pts_time"])


def _check_said(cues: list[tuple[float, float, str]], first: float) -> None:
    """The finished cues, each on its own start against the picture, said by
    the one who said it; nothing still growing, and nothing lasting no time."""
    assert all(end > start for start, end, _ in cues), cues
    assert not any(text.endswith(" so") for _, _, text in cues), cues
    starts = {round(start - first, 2): text for start, _, text in cues}
    assert len(starts) == len(_SAID), cues
    for (said, voice), (start, text) in zip(sorted(_SAID.items()), sorted(starts.items())):
        assert abs(start - said) <= _FRAME, (start, said)
        assert text.startswith(f"<v {voice}>cue "), text


def test_a_quiet_cue_stream_into_a_file_holds_no_picture_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _run("out.mkv", "video_codec 'libx264', preset 'ultrafast'")
    extracted = subprocess.run(
        ["ffmpeg", "-v", "error", "-copyts", "-i", "out.mkv", "-map", "0:s:0",
         "-c:s", "copy", "-f", "webvtt", "-"],
        capture_output=True, text=True, timeout=_TIMEOUT,
    )
    assert extracted.returncode == 0, extracted.stderr
    _check_said(_cues(extracted.stdout), _first_picture(tmp_path / "out.mkv"))


def test_a_quiet_cue_stream_into_hls_holds_no_picture_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "out").mkdir()
    _run(
        "out/master.m3u8",
        "format 'hls', hls_time 4, hls_playlist_type 'vod', video_codec 'libx264', "
        "preset 'ultrafast'",
    )
    variant = tmp_path / "out" / "180p"
    segments = sorted(variant.glob("index*.vtt"), key=lambda p: int(p.stem[5:] or 0))
    assert len(segments) >= 9
    # Each segment's map puts a cue time of zero 1.4 s before the MPEG-TS
    # timestamps, which is where the picture's own times are read from.
    first = _first_picture(variant / "segment_0.ts") - 1.4
    cues = []
    for segment in segments:
        text = segment.read_text(encoding="utf-8")
        assert text.startswith("WEBVTT\nX-TIMESTAMP-MAP=MPEGTS:126000,LOCAL:00:00:00.000\n")
        cues += _cues(text)
    _check_said(cues, first)
