"""End-to-end exec proof for a ROWS module: rows in, rows out, no stream.

`captions` (`sidecar/modules/captions`) is a window module that reads a
video stream and hands back one cue per frame; `fauxlate`
(`sidecar/modules/fauxlate`) is a rows module that reads those cues and
writes them back with every word carrying `-a` and `-o` in turn. Written
one inside the other -- `fauxlate(captions(f.video[1]).cues)` -- the two
run in ONE sidecar, the rows crossing between them in memory over a rows
edge, and the translated cues come back out as the output's subtitle
track.

Real throughout: the sidecar runs both modules for real and the far ffmpeg
muxes the WebVTT document it writes. Requires ffmpeg/ffprobe on PATH, the
`ffrwd-wasm` sidecar (`uv sync --extra wasm`), and both modules built for
`wasm32-wasip2`. Tests skip cleanly when any of those is missing.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from ffrwd import binaries, cli
from ffrwd.compiler import compile_table_sql

_CLI_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = _CLI_ROOT.parent
_SIDECAR_MODULES = _REPO_ROOT / "sidecar" / "modules"
_BUILT = _SIDECAR_MODULES / "target" / "wasm32-wasip2" / "release"
_CAPTIONS = _BUILT / "captions.wasm"
_FAUXLATE = _BUILT / "fauxlate.wasm"
_FIXTURE = _CLI_ROOT / "tests" / "fixtures" / "av.mp4"

_SUBPROCESS_TIMEOUT = 300.0

# What the word rule leaves on every word: `-a` and `-o` in turn, before
# whatever punctuation trailed it. See sidecar/modules/fauxlate/src/lib.rs.
_TRANSLATED = re.compile(r"-[ao]\W*$")

# One cue's payload in a WebVTT document: the lines after a timing line, up
# to the blank line that ends the cue. WebVTT's hours field is optional, and
# ffmpeg's muxer leaves it out under an hour.
_TIMING = re.compile(r"^(\d\d:)?\d\d:\d\d\.\d\d\d --> ")


@pytest.fixture(autouse=True)
def _require_everything() -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg/ffprobe not found on PATH")
    if binaries.ffrwd_wasm_path() is None:
        pytest.skip("ffrwd-wasm not found (uv sync --extra wasm)")
    for module in (_CAPTIONS, _FAUXLATE):
        if not module.exists():
            pytest.skip(
                f"module missing: {module} (cargo build --target wasm32-wasip2 "
                f"--release, from {_SIDECAR_MODULES})"
            )


def _query(out_path: Path, *, rows: str, translating: bool) -> str:
    """The cookbook's own declarations, with `rows` as the selected column.

    A script declares only what it calls, so the untranslated run leaves the
    rows function out entirely.
    """
    fauxlate = (
        "CREATE FUNCTION fauxlate(cues cue[]) RETURNS cue[]\n"
        f"  AS '{_FAUXLATE.as_posix()}', 'fauxlate' LANGUAGE wasm;\n"
    )
    return (
        "CREATE FUNCTION captions(v video_stream)\n"
        "RETURNS STRUCT(v video_stream, cues cue[])\n"
        f"  AS '{_CAPTIONS.as_posix()}', 'captions' LANGUAGE wasm;\n"
        f"{fauxlate if translating else ''}"
        "COPY (\n"
        f"  SELECT f.video[1], f.audio[1], {rows}\n"
        f"  FROM input('{_FIXTURE.as_posix()}') f\n"
        f") TO '{out_path.as_posix()}'"
    )


def _two_document_query(out_path: Path) -> str:
    """The cookbook's own two-track query: the rows written, and translated.

    One `captions` node produces both columns -- the CTE is what lets the
    second read what the first named -- so one sidecar writes two documents.
    """
    return f"""CREATE FUNCTION captions(v video_stream)
RETURNS STRUCT(v video_stream, cues cue[])
  AS '{_CAPTIONS.as_posix()}', 'captions' LANGUAGE wasm;
CREATE FUNCTION fauxlate(cues cue[]) RETURNS cue[]
  AS '{_FAUXLATE.as_posix()}', 'fauxlate' LANGUAGE wasm;
COPY (
  WITH d AS (
    SELECT f.video[1] AS v, captions(f.video[1]).cues AS speech
    FROM input('{_FIXTURE.as_posix()}') f
  )
  SELECT d.v, d.speech, fauxlate(d.speech) AS translated
  FROM d
) TO '{out_path.as_posix()}'"""


def _track_texts(path: Path, title: str) -> list[str]:
    """Every cue of the track `title` names, read back by the compiler's probe."""
    sinks = compile_table_sql(
        f"SELECT c.text FROM input('{path.as_posix()}') f, "
        f"unnest(f.cues['{title}']) c"
    )
    return [str(row[0]) for row in sinks[0].result.rows]


def _cue_texts(path: Path) -> list[str]:
    """Every cue payload in `path`'s first subtitle track, in document order."""
    done = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:s:0", "-f", "webvtt", "-"],
        capture_output=True,
        text=True,
        timeout=_SUBPROCESS_TIMEOUT,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    texts: list[str] = []
    lines = done.stdout.splitlines()
    for index, line in enumerate(lines):
        if _TIMING.match(line) is None:
            continue
        payload: list[str] = []
        for following in lines[index + 1 :]:
            if not following.strip():
                break
            payload.append(following)
        texts.append("\n".join(payload))
    return texts


def _subtitle_streams(path: Path) -> list[dict[str, object]]:
    done = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "stream=codec_type,codec_name",
            "-of", "json", str(path),
        ],
        capture_output=True,
        text=True,
        timeout=_SUBPROCESS_TIMEOUT,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    streams: list[dict[str, object]] = json.loads(done.stdout)["streams"]
    return [s for s in streams if s["codec_type"] == "subtitle"]


@pytest.mark.exec
def test_a_rows_module_translates_every_cue_its_producer_wrote(tmp_path: Path) -> None:
    """The rows edge carries every cue and only the cues: the output's one
    subtitle track holds as many cues as `captions` alone writes, each of
    them run through the word rule, so no cue is dropped, duplicated, or
    left untranslated on the way between the two modules."""
    plain = tmp_path / "plain.mkv"
    written_query = _query(
        plain, rows="captions(f.video[1]).cues", translating=False
    )
    assert cli.main(["run", written_query, "-y"]) == 0
    written = _cue_texts(plain)
    assert written, "the captions module wrote no cues to translate"

    translated_path = tmp_path / "translated.mkv"
    query = _query(
        translated_path,
        rows="fauxlate(captions(f.video[1]).cues)",
        translating=True,
    )
    assert cli.main(["run", query, "-y"]) == 0
    assert translated_path.exists()

    assert len(_subtitle_streams(translated_path)) == 1
    translated = _cue_texts(translated_path)
    assert len(translated) == len(written)
    untouched = [text for text in translated if _TRANSLATED.search(text) is None]
    assert untouched == []


@pytest.mark.exec
def test_one_producers_rows_are_written_twice_once_translated(tmp_path: Path) -> None:
    """Two rows documents off ONE region, muxed as two subtitle tracks: the
    `speech` track is what `captions` wrote and `translated` the same cues a
    module later, cue for cue, with neither document holding the other's
    rows."""
    out = tmp_path / "spoken.mkv"
    assert cli.main(["run", _two_document_query(out), "-y"]) == 0
    assert len(_subtitle_streams(out)) == 2

    speech = _track_texts(out, "speech")
    translated = _track_texts(out, "translated")
    assert speech, "the captions module wrote no cues"
    assert len(translated) == len(speech)
    assert [text for text in translated if _TRANSLATED.search(text) is None] == []
    assert [text for text in speech if _TRANSLATED.search(text) is not None] == []
