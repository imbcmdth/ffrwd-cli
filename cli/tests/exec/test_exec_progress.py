"""Drawing a real encode's progress, and reporting a real failure's log.

Marked ``@pytest.mark.exec`` and excluded from the default run, same as
``test_exec.py``. Run explicitly::

    python -m pytest -m exec tests/exec/test_exec_progress.py -q

Requires ``ffmpeg``/``ffprobe`` on PATH and the fixtures already generated
(``python scripts/gen_fixtures.py``). Tests skip cleanly if either is missing.

The console writes to a stream that claims to be a terminal, since that is
what a redrawn line needs and a test run does not have. The encode is
deliberately slow -- a large scale at a slow preset -- so that ffmpeg writes
several progress blocks rather than only the one it ends with.
"""

from __future__ import annotations

import io
import re
import shutil
from pathlib import Path

import pytest

from ffrwd.compiler import compile_all, emitted_commands
from ffrwd.console import Console
from ffrwd.execute import execute

pytestmark = pytest.mark.exec

_FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
_AV = _FIXTURES_DIR / "av.mp4"

# What gen_fixtures.py writes: 4 seconds of it.
_SOURCE_SECONDS = 4.0
_TIMEOUT = 120.0
# A progress line, as `ffrwd.execute` tells one from a line of the log.
_PROGRESS = re.compile(r"^[a-z_][a-z0-9_]*=")


@pytest.fixture(autouse=True)
def _require_ffmpeg_tools() -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg/ffprobe not found on PATH")


def _require_fixture(path: Path) -> None:
    if not path.exists():
        pytest.skip(f"fixture missing: {path} (run scripts/gen_fixtures.py first)")


class _Tty(io.StringIO):
    """A stream that claims to be a terminal."""

    def isatty(self) -> bool:
        return True


def _query(out_path: Path) -> str:
    return (
        f"COPY (SELECT scale(f.video[1], 1920, -2) "
        f"FROM input('{_AV.resolve().as_posix()}') f) "
        f"TO '{out_path.resolve().as_posix()}' "
        "WITH (video_codec 'libx264', crf 0, preset 'veryslow')"
    )


def test_a_transcode_draws_its_progress_and_clears_the_line(tmp_path: Path) -> None:
    _require_fixture(_AV)
    out_path = tmp_path / "encoded.mp4"
    stream = _Tty()
    console = Console(stream)
    compiled = compile_all(_query(out_path))

    result = execute(
        emitted_commands(compiled.graphs),
        timeout=_TIMEOUT,
        overwrite=True,
        work=console.work("encoding", compiled.duration),
    )

    assert result.exit_code == 0, result.commands[-1].stderr
    assert out_path.exists()
    assert compiled.duration == _SOURCE_SECONDS
    written = stream.getvalue()
    drawn = [line.rstrip() for line in written.split("\r") if line.strip()]
    assert drawn, "nothing was drawn"
    assert all(line.startswith("encoding  [") for line in drawn)
    # A bounded input, so every line is a fraction of its length.
    assert all(f"/ 0:0{int(_SOURCE_SECONDS)}" in line for line in drawn)
    # The last thing written blanks the line and returns to its start.
    assert written.endswith("\r")
    assert written.split("\r")[-2].strip() == ""


def test_a_failing_command_reports_its_log_and_none_of_the_progress(
    tmp_path: Path,
) -> None:
    """A destination in a directory that is not there ends the run, and what
    ffmpeg said is the report -- with none of the progress in it."""
    _require_fixture(_AV)
    out_path = tmp_path / "nowhere" / "encoded.mp4"
    console = Console(_Tty())
    compiled = compile_all(_query(out_path))

    result = execute(
        emitted_commands(compiled.graphs),
        timeout=_TIMEOUT,
        overwrite=True,
        work=console.work("encoding", compiled.duration),
    )

    assert result.exit_code != 0
    log = result.commands[-1].stderr
    assert result.commands[-1].captured
    assert out_path.name in log
    # Everything ffmpeg said, from the input it opened to the error it ended
    # on -- and not one line of the progress.
    assert "Input #0" in log
    assert [line for line in log.splitlines() if _PROGRESS.match(line)] == []
