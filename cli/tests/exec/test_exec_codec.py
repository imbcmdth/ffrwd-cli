"""End-to-end exec tests for a codec package's encoder and decoder.

Marked ``@pytest.mark.exec`` and excluded from the default run. Run
explicitly::

    python -m pytest -m exec tests/exec/test_exec_codec.py -q

The module under test is ``testcodec``: a lossless run-length codec, tag
``FTST``, whose encoder writes one keyframe packet a frame and whose decoder
inverts it exactly. So a clip coded by it and decoded again is the clip,
frame for frame, and every expectation here is that.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from ffrwd import binaries, cli

pytestmark = pytest.mark.exec

_CLI_ROOT = Path(__file__).resolve().parent.parent.parent
_REPO_ROOT = _CLI_ROOT.parent
_SIDECAR_MODULES = _REPO_ROOT / "sidecar" / "modules"
_CODEC = _SIDECAR_MODULES / "target" / "wasm32-wasip2" / "release" / "testcodec.wasm"
_TIMEOUT = 120.0

_ENCODER = (
    "CREATE FUNCTION enc(level number DEFAULT NULL) RETURNS encoder "
    "AS '{module}', 'encode' LANGUAGE wasm;"
)
_DECODER = "CREATE FUNCTION dec() RETURNS decoder AS '{module}', 'decode' LANGUAGE wasm;"


@pytest.fixture(autouse=True)
def _require_everything() -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg/ffprobe not found on PATH")
    if binaries.ffrwd_wasm_path() is None:
        pytest.skip("the ffrwd-wasm sidecar is not installed")
    if not _CODEC.exists():
        pytest.skip(
            f"module missing: {_CODEC} (cargo build --target wasm32-wasip2 "
            f"--release, from {_SIDECAR_MODULES})"
        )


def _clip(path: Path) -> Path:
    """One second of a moving picture and a tone, as an ordinary mp4."""
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y",
            "-f", "lavfi", "-i", "testsrc2=size=64x48:rate=10",
            "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000",
            "-t", "1", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
            "-shortest", str(path),
        ],
        check=True,
        timeout=_TIMEOUT,
    )
    return path


def _frames(path: Path, pix_fmt: str = "yuv420p") -> list[tuple[str, str]]:
    """Each picture's time and the md5 of its pixels in `pix_fmt`, decoded by
    ffmpeg."""
    done = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0",
            "-pix_fmt", pix_fmt, "-f", "framemd5", "-",
        ],
        capture_output=True,
        text=True,
        timeout=_TIMEOUT,
        check=True,
    )
    return [
        (fields[2].strip(), fields[5].strip())
        for line in done.stdout.splitlines()
        if not line.startswith("#") and len(fields := line.split(",")) == 6
    ]


def _run(sql: str) -> None:
    sql = sql.replace("{module}", _CODEC.as_posix())
    assert cli.main(["run", sql, "-y", "-q"]) == 0


def _tag(path: Path) -> str:
    done = subprocess.run(
        [
            "ffprobe", "-v", "quiet", "-select_streams", "v:0",
            "-show_entries", "stream=codec_tag_string", "-of", "csv=p=0", str(path),
        ],
        capture_output=True,
        text=True,
        timeout=_TIMEOUT,
    )
    return done.stdout.strip()


def test_a_clip_coded_and_decoded_again_is_the_clip(tmp_path: Path) -> None:
    clip = _clip(tmp_path / "clip.mp4")
    coded = tmp_path / "coded.nut"
    back = tmp_path / "back.nut"
    _run(
        f"{_ENCODER}\nCOPY (SELECT f.video[1], f.audio[1] FROM input('{clip.as_posix()}') f) "
        f"TO '{coded.as_posix()}' WITH (video_codec enc(level => 3), audio_codec 'aac')"
    )
    assert _tag(coded) == "FTST"
    _run(
        f"{_DECODER}\nCOPY (SELECT f.video[1] FROM input('{coded.as_posix()}') f) "
        f"TO '{back.as_posix()}' WITH (video_codec 'rawvideo')"
    )
    assert _frames(back) == _frames(clip)


def test_a_coded_stream_copied_to_matroska_stays_coded(tmp_path: Path) -> None:
    clip = _clip(tmp_path / "clip.mp4")
    coded = tmp_path / "coded.nut"
    moved = tmp_path / "moved.mkv"
    back = tmp_path / "back.nut"
    _run(
        f"{_ENCODER}\nCOPY (SELECT f.video[1] FROM input('{clip.as_posix()}') f) "
        f"TO '{coded.as_posix()}' WITH (video_codec enc())"
    )
    _run(
        f"{_DECODER}\nCOPY (SELECT f.video[1] FROM input('{coded.as_posix()}') f) "
        f"TO '{moved.as_posix()}'"
    )
    assert _tag(moved) == "FTST"
    _run(
        f"{_DECODER}\nCOPY (SELECT f.video[1] FROM input('{moved.as_posix()}') f) "
        f"TO '{back.as_posix()}' WITH (video_codec 'rawvideo')"
    )
    assert [md5 for _, md5 in _frames(back)] == [md5 for _, md5 in _frames(clip)]


def test_a_decoded_stream_is_filtered_like_any_other(tmp_path: Path) -> None:
    clip = _clip(tmp_path / "clip.mp4")
    coded = tmp_path / "coded.nut"
    flipped = tmp_path / "flipped.nut"
    expected = tmp_path / "expected.nut"
    _run(
        f"{_ENCODER}\nCOPY (SELECT f.video[1] FROM input('{clip.as_posix()}') f) "
        f"TO '{coded.as_posix()}' WITH (video_codec enc())"
    )
    _run(
        f"{_DECODER}\nCOPY (SELECT hflip(f.video[1]) FROM input('{coded.as_posix()}') f) "
        f"TO '{flipped.as_posix()}' WITH (video_codec 'rawvideo')"
    )
    _run(
        f"COPY (SELECT hflip(f.video[1]) FROM input('{clip.as_posix()}') f) "
        f"TO '{expected.as_posix()}' WITH (video_codec 'rawvideo')"
    )
    assert _frames(flipped) == _frames(expected)


def _picture(path: Path, entries: str) -> dict[str, str]:
    """ffprobe's `entries` of the first picture stream, by name."""
    done = subprocess.run(
        [
            "ffprobe", "-v", "quiet", "-select_streams", "v:0",
            "-show_entries", f"stream={entries}", "-of", "default=nw=1", str(path),
        ],
        capture_output=True,
        text=True,
        timeout=_TIMEOUT,
        check=True,
    )
    return dict(line.split("=", 1) for line in done.stdout.splitlines() if "=" in line)


def _source(path: Path, pix_fmt: str) -> Path:
    """One second of a moving picture in `pix_fmt`, coded losslessly, and
    declared bt709, tv range, left-sited."""
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y",
            "-f", "lavfi", "-i", "testsrc2=size=64x48:rate=10", "-t", "1",
            "-vf", f"format={pix_fmt},setparams=range=tv:color_primaries=bt709"
            ":color_trc=bt709:colorspace=bt709:chroma_location=left",
            "-c:v", "ffv1", str(path),
        ],
        check=True,
        timeout=_TIMEOUT,
    )
    return path


_COLOUR = "color_range,color_primaries,color_transfer,color_space,chroma_location"
_BT709 = {
    "color_range": "tv",
    "color_primaries": "bt709",
    "color_transfer": "bt709",
    "color_space": "bt709",
    "chroma_location": "left",
}


def test_a_444_clip_is_coded_in_444_and_decoded_back_to_it(tmp_path: Path) -> None:
    clip = _source(tmp_path / "clip.mkv", "yuv444p")
    coded = tmp_path / "coded.nut"
    back = tmp_path / "back.nut"
    _run(
        f"{_ENCODER}\nCOPY (SELECT f.video[1] FROM input('{clip.as_posix()}') f) "
        f"TO '{coded.as_posix()}' WITH (video_codec enc())"
    )
    _run(
        f"{_DECODER}\nCOPY (SELECT f.video[1] FROM input('{coded.as_posix()}') f) "
        f"TO '{back.as_posix()}' WITH (video_codec 'rawvideo')"
    )
    assert _picture(back, "pix_fmt") == {"pix_fmt": "yuv444p"}
    assert _frames(back, "yuv444p") == _frames(clip, "yuv444p")


def test_a_coded_files_colour_is_the_sources_and_its_decode_carries_it(
    tmp_path: Path,
) -> None:
    clip = _source(tmp_path / "clip.mkv", "yuv420p")
    coded = tmp_path / "coded.mkv"
    quick = tmp_path / "coded.mov"
    back = tmp_path / "back.mkv"
    for path in (coded, quick):
        _run(
            f"{_ENCODER}\nCOPY (SELECT f.video[1] FROM input('{clip.as_posix()}') f) "
            f"TO '{path.as_posix()}' WITH (video_codec enc())"
        )
    assert _picture(coded, _COLOUR) == _BT709
    # QuickTime's colour atom for a codec it does not know holds no range,
    # and the container has no field for the siting.
    assert _picture(quick, _COLOUR) == {
        **_BT709,
        "color_range": "unknown",
        "chroma_location": "unspecified",
    }
    _run(
        f"{_DECODER}\nCOPY (SELECT f.video[1] FROM input('{coded.as_posix()}') f) "
        f"TO '{back.as_posix()}' WITH (video_codec 'ffv1')"
    )
    assert _picture(back, _COLOUR) == _BT709
    assert _frames(back) == _frames(clip)
