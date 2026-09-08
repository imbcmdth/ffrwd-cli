"""The maintainer's shape matrix, compiled cell by cell.

`tests/data/shape_matrix.json` is his workbook: 21 input shapes down the
side, 21 output shapes across the top, and in each cell the query that turns
one into the other. The workbook is where a human edits it; the JSON is a
copy, and each cell's SQL is that cell's text unchanged.

What this measures is the count of cells the compiler REFUSES, and the rule
on that number is one-directional:

* a cell that stopped compiling is a regression, and fails;
* a cell that started compiling is the win, and fails too -- saying the
  baseline is stale and to regenerate it, because an improvement nobody
  records is how a count stops meaning anything;
* a cell that changed which side of the count it is on -- newly run, or no
  longer run -- fails the same way, because the number is only comparable
  over the same cells;
* a refusal whose text moved warns rather than fails. The number did not
  change, and the local ffmpeg is in some of these messages, so a machine
  with another build must not go red for it.

`tests/data/shape_sweep_baseline.json` holds one row per cell: its
coordinates, whether it compiled, and, for a refusal, the whole typed error
-- every field of `FfrwdError.to_dict()`, captured the way
`tests/test_refusal_snapshot.py` captures one, through that module's own
runner rather than a second copy of it.

A row is "unrun" where no fixture stands for its input shape. Those cells are
counted and named, never guessed at: `SOURCES` below is the whole of the
judgement here, one line per shape saying which fixture stands for it and
why, and a shape with nothing to stand for it says so instead.

Exec-marked: every runnable cell reads real media.

Regenerate with `FFRWD_WRITE_SHAPE_BASELINE` set::

    FFRWD_WRITE_SHAPE_BASELINE=1 pytest tests/test_shape_sweep.py -m exec

Then read the diff: this writes whatever the compiler currently does, and
cannot tell a lifted restriction from a broken one.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import pytest

from . import test_refusal_snapshot as snapshot

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MATRIX_PATH = PROJECT_ROOT / "tests" / "data" / "shape_matrix.json"
BASELINE_PATH = PROJECT_ROOT / "tests" / "data" / "shape_sweep_baseline.json"

# Set to regenerate rather than compare; see the module docstring.
_WRITE_ENV = "FFRWD_WRITE_SHAPE_BASELINE"

_REGEN_HINT = (
    f"set {_WRITE_ENV} and rerun this file with -m exec, then review the diff "
    f"before committing"
)

# Every fixture is four seconds long, so a cell that synthesizes a source
# beside one makes a source of the same length.
_DURATION = 4

COMPILED = "compiled"
REFUSED = "refused"
UNRUN = "unrun"

Row = dict[str, object]


# ---------------------------------------------------------------------------
# what stands for each input shape
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Stand:
    """What a cell in one input row reads: a fixture, or why nothing does.

    `height` is the value a ``WHERE r.height = :h`` predicate is given and
    `name` the value a ``WHERE r.name = :'audio'`` predicate is given -- both
    read off the fixture itself, so a predicate the compiler supports selects
    a row that is really there rather than emptying the relation.
    """

    shape: str
    fixture: str | None
    height: int | None
    name: str | None
    why: str


_NO_VIDEO_ROW = "no video rendition, so a height predicate matches nothing whatever it is given"
_NO_AUDIO_ROW = "no audio-only rendition, so a name predicate matches nothing whatever it is given"
_FILE_ROWS = "a file has renditions of neither kind, so both values are placeholders"

SOURCES: tuple[Stand, ...] = (
    Stand(
        "1 file 0 video track 0 audio track", None, None, None,
        "nothing stands for it: ffmpeg writes no container with no streams in it",
    ),
    Stand(
        "1 file 1 video track 0 audio track", "tests/fixtures/testsrc.mp4", 240, "audio_0",
        f"one testsrc2 track, 320x240, no audio; {_FILE_ROWS}",
    ),
    Stand(
        "1 file 0 video track 1 audio track", "tests/fixtures/audio.m4a", 240, "audio_0",
        f"one sine track, no video; {_FILE_ROWS}, and the height is the set's",
    ),
    Stand(
        "1 file M video track 0 audio track", "tests/fixtures/video2.mkv", 240, "audio_0",
        f"testsrc2 beside smptebars, no audio; {_FILE_ROWS}",
    ),
    Stand(
        "1 file 0 video track M audio track", "tests/fixtures/audio2.mka", 240, "audio_0",
        f"two language-tagged sine tracks, no video; {_FILE_ROWS}",
    ),
    Stand(
        "1 file 1 video track 1 audio track", "tests/fixtures/av.mp4", 240, "audio_0",
        f"the simplest A/V file, 320x240; {_FILE_ROWS}",
    ),
    Stand(
        "1 file 1 video track M audio track", "tests/fixtures/av2.mp4", 240, "audio_0",
        f"one video track, two language-tagged audio tracks; {_FILE_ROWS}",
    ),
    Stand(
        "1 file M video track 1 audio track", "tests/fixtures/av-2v.mkv", 240, "audio_0",
        f"two video tracks, one audio track; {_FILE_ROWS}",
    ),
    Stand(
        "1 file M video track N audio track", "tests/fixtures/av-2v2a.mkv", 240, "audio_0",
        f"two video tracks, two audio tracks -- both columns arrays; {_FILE_ROWS}",
    ),
    Stand(
        "1 manifest 1 video", None, None, None,
        "nothing stands for it: every ladder here carries audio somewhere, and the "
        "one video-only manifest would be a new fixture rather than a flag on an "
        "existing one",
    ),
    Stand(
        "1 manifest 1 audio", "tests/fixtures/ladder-audio-only/master.m3u8", 720, "audio_0",
        f"one audio rendition and no video row at all; {_NO_VIDEO_ROW}",
    ),
    Stand(
        "1 manifest - muxed 1 video + audio", None, None, None,
        "nothing stands for it: the muxed ladder has two rungs, and a one-rung "
        "ladder is another real encode rather than a flag on it",
    ),
    Stand(
        "1 manifest - muxed M video + audio", "tests/fixtures/ladder/master.m3u8", 720, "audio_0",
        f"two variants, each carrying its own video and audio; {_NO_AUDIO_ROW}",
    ),
    Stand(
        "1 manifest - demuxed 1 video 1 audio", None, None, None,
        "nothing stands for it: the demuxed ladders have two video rungs, and a "
        "one-rung one is another real encode",
    ),
    Stand(
        "1 manifest - demuxed M video 1 audio",
        "tests/fixtures/ladder-demuxed-hls/master.m3u8", 720, "audio_2",
        "two video-only variants and the audio group's own row. The HLS ladder "
        "rather than the DASH one of the same shape: every output column of the "
        "matrix names format 'hls', so this keeps the container out of the reading",
    ),
    Stand(
        "1 manifest - demuxed 1 video M audio", None, None, None,
        "nothing stands for it: no ladder here has two audio renditions",
    ),
    Stand(
        "1 manifest - demuxed M video N audio", None, None, None,
        "nothing stands for it: no ladder here has two audio renditions",
    ),
    Stand(
        "1 manifest - hybrid 1 video + audio 1 audio", None, None, None,
        "nothing stands for it: the hybrid ladder has two muxed variants, and a "
        "one-variant one is another composed playlist",
    ),
    Stand(
        "1 manifest - hybrid M video + audio 1 audio",
        "tests/fixtures/ladder-hybrid/master.m3u8", 720, "audio_2",
        "two variants that mux their own audio AND name an audio group, plus that "
        "group's own row",
    ),
    Stand(
        "1 manifest - hybrid 1 video + audio M audio", None, None, None,
        "nothing stands for it: no ladder here has two audio renditions",
    ),
    Stand(
        "1 manifest - hybrid M video + audio N audio", None, None, None,
        "nothing stands for it: no ladder here has two audio renditions",
    ),
)


def _destination(shape: str) -> str:
    """Where an output shape's cell writes.

    Each manifest column names ``format 'hls'`` in its own SQL, so a manifest
    destination only has to be an HLS master. A file column carries no format,
    so its extension is the whole of the choice: audio alone where the column
    says no video track, and a general container otherwise.
    """
    if "manifest" in shape:
        return "out/master.m3u8"
    return "out.m4a" if shape.startswith("1 file 0 video track") else "out.mp4"


# ---------------------------------------------------------------------------
# the matrix
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Cell:
    """One cell of the matrix: where it sits, and the query pinned there."""

    key: str
    input: str
    output: str
    kind: str
    sql: str


def _matrix() -> tuple[list[str], list[str], list[Cell]]:
    data = json.loads(MATRIX_PATH.read_text(encoding="utf-8"))
    inputs: list[str] = data["inputs"]
    outputs: list[str] = data["outputs"]
    cells = [
        Cell(
            key=f"{inputs.index(cell['input']):02d}->{outputs.index(cell['output']):02d}",
            input=cell["input"],
            output=cell["output"],
            kind=cell["kind"],
            sql=cell["sql"],
        )
        for cell in data["cells"]
    ]
    return inputs, outputs, cells


_INPUTS, _OUTPUTS, _CELLS = _matrix()
_QUERIES = [cell for cell in _CELLS if cell.kind == "query"]


# ---------------------------------------------------------------------------
# rows
# ---------------------------------------------------------------------------


def _bindings(sql: str, stand: Stand, destination: str) -> list[str]:
    """``-v name=value`` for exactly the variables `sql` references.

    The CLI refuses a variable the query never names, so this passes the ones
    it does and no others. Both of psql's spellings count as a reference:
    ``:'name'`` interpolates a string literal, bare ``:name`` a number.
    """
    values: dict[str, object] = {
        "src": stand.fixture,
        "dest": destination,
        "duration": _DURATION,
        "h": stand.height,
        "audio": stand.name,
    }
    argv: list[str] = []
    for name, value in values.items():
        if f":'{name}'" in sql or _references_bare(sql, name):
            argv += ["-v", f"{name}={value}"]
    return argv


def _references_bare(sql: str, name: str) -> bool:
    """Whether `sql` names ``:name`` -- the bare, unquoted spelling."""
    marker = f":{name}"
    at = sql.find(marker)
    while at != -1:
        after = at + len(marker)
        if after == len(sql) or not (sql[after].isalnum() or sql[after] == "_"):
            return True
        at = sql.find(marker, after)
    return False


def _row(cell: Cell, verdict: str, **extra: object) -> Row:
    return {"input": cell.input, "output": cell.output, "verdict": verdict, **extra}


def _compile(
    cell: Cell, stand: Stand, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Row:
    """Compile one cell's query against the fixture standing for its input row."""
    argv = ["compile", "-f", "query.sql", *_bindings(cell.sql, stand, _destination(cell.output))]
    code, _, refusals = snapshot._run(argv, cell.sql, tmp_path, monkeypatch)
    if code == 0:
        return _row(cell, COMPILED)
    assert refusals, (
        f"{cell.key}: `ffrwd {' '.join(argv)}` exited {code} without a typed error. "
        f"A cell is a compile or an FfrwdError, and this is neither -- most likely "
        f"the bindings above are not the ones its query names"
    )
    error = snapshot._refused(snapshot.EXEC, refusals[0], tmp_path)["error"]
    return _row(cell, REFUSED, error=error)


def _sweep(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Row]:
    """Every cell of the matrix, compiled where a fixture stands for its input."""
    stands = {stand.shape: stand for stand in SOURCES}
    built: dict[str, Row] = {}
    for cell in _CELLS:
        stand = stands[cell.input]
        if cell.kind != "query":
            built[cell.key] = _row(cell, UNRUN, why=cell.sql)
        elif stand.fixture is None:
            built[cell.key] = _row(cell, UNRUN, why=stand.why)
        else:
            built[cell.key] = _compile(cell, stand, tmp_path, monkeypatch)
    return built


# ---------------------------------------------------------------------------
# the file
# ---------------------------------------------------------------------------

_ABOUT = (
    "One row per cell of tests/data/shape_matrix.json: whether the compiler "
    "took the query pinned there, and the whole typed error where it did not. "
    "The count of refused rows may only go down; see tests/test_shape_sweep.py."
)


def _load() -> dict[str, Row]:
    assert BASELINE_PATH.exists(), f"{BASELINE_PATH.name} is missing -- {_REGEN_HINT}"
    data = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    stored: dict[str, Row] = data["cells"]
    return stored


def _write(built: dict[str, Row]) -> None:
    payload = {"about": _ABOUT, "cells": {key: built[key] for key in sorted(built)}}
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    BASELINE_PATH.write_text(text, encoding="utf-8", newline="\n")


def _writing() -> bool:
    return bool(os.environ.get(_WRITE_ENV))


def _counts(rows: dict[str, Row]) -> tuple[int, int, int]:
    verdicts = [row["verdict"] for row in rows.values()]
    return (
        verdicts.count(COMPILED),
        verdicts.count(REFUSED),
        verdicts.count(UNRUN),
    )


def _say(request: pytest.FixtureRequest, line: str) -> None:
    """One line into the run's own report, so the number is said out loud."""
    reporter = request.config.pluginmanager.get_plugin("terminalreporter")
    if reporter is None:
        print(line)
        return
    reporter.write_line(line)


@pytest.fixture(scope="module")
def _fixtures() -> None:
    """The generated media every runnable cell reads."""
    subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "gen_fixtures.py")],
        check=True,
    )


# ---------------------------------------------------------------------------
# the checks
# ---------------------------------------------------------------------------


def test_a_fixture_is_named_for_every_input_shape() -> None:
    """`SOURCES` is the matrix's own side labels, in the matrix's own order."""
    assert [stand.shape for stand in SOURCES] == _INPUTS
    for stand in SOURCES:
        if stand.fixture is None:
            assert stand.height is None and stand.name is None, stand.shape
        else:
            assert stand.height is not None, stand.shape


@pytest.mark.exec
def test_every_named_fixture_is_one_the_generator_writes(_fixtures: None) -> None:
    """A stand names a file `scripts/gen_fixtures.py` really produces. Exec-marked
    because `tests/fixtures/` is generated: a bare checkout has none of it."""
    for stand in SOURCES:
        if stand.fixture is not None:
            assert (PROJECT_ROOT / stand.fixture).exists(), stand.fixture


def test_the_baseline_covers_every_cell() -> None:
    """Checked on a bare machine, so a cell added to the matrix without a row
    fails in the default suite rather than only where ffmpeg is installed."""
    assert set(_load()) == {cell.key for cell in _CELLS}, (
        f"the baseline and the matrix name different cells -- {_REGEN_HINT}"
    )


def test_the_baseline_is_portable() -> None:
    data = BASELINE_PATH.read_bytes()
    assert b"\r" not in data
    text = data.decode("utf-8")
    for spelling in (str(snapshot.REPO_ROOT), snapshot.REPO_ROOT.as_posix()):
        assert spelling not in text


@pytest.mark.exec
def test_the_refused_count_only_goes_down(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
    _fixtures: None,
) -> None:
    built = _sweep(tmp_path, monkeypatch)
    if _writing():
        _write(built)
    stored = _load()

    compiled, refused, unrun = _counts(built)
    _say(request, f"shape matrix: {compiled} compiled, {refused} refused, {unrun} unrun")

    regressed: list[str] = []
    unmeasured: list[str] = []
    improved: list[str] = []
    covered: list[str] = []
    reworded: list[str] = []
    for key in sorted(built):
        was, now = str(stored[key]["verdict"]), str(built[key]["verdict"])
        if was == now:
            if now == REFUSED and stored[key].get("error") != built[key].get("error"):
                reworded.append(key)
        elif was == UNRUN:
            covered.append(key)
        elif now == UNRUN:
            unmeasured.append(f"{key} (was {was})")
        elif now == REFUSED:
            regressed.append(key)
        else:
            improved.append(key)

    if reworded:
        warnings.warn(
            f"{len(reworded)} refusals say something new without changing the "
            f"count: {', '.join(reworded)}. Regenerating records the new text.",
            stacklevel=1,
        )
    assert not regressed, (
        f"{len(regressed)} cells stopped compiling: {', '.join(regressed)}. "
        f"The count of refusals may only go down; each of these is a restriction "
        f"the compiler did not have before"
    )
    assert not unmeasured, (
        f"{len(unmeasured)} cells are no longer measured: {', '.join(unmeasured)}. "
        f"A fixture that stood for an input shape is gone, so the count is taken "
        f"over fewer cells than the baseline was"
    )
    assert not improved, (
        f"{len(improved)} cells now compile that the baseline records as refused: "
        f"{', '.join(improved)}. That is the win, and the baseline is stale -- "
        f"{_REGEN_HINT}"
    )
    assert not covered, (
        f"{len(covered)} cells now run that the baseline records as unrun: "
        f"{', '.join(covered)}. An input shape gained a fixture, so the matrix "
        f"measures more than it did -- {_REGEN_HINT}"
    )
