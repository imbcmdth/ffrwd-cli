"""Tests for the CLI's narration: lines, sizes, and the spinner.

The spinner is exercised against two injected streams -- a plain StringIO,
which is what a pipe or a captured test run looks like, and one whose
``isatty`` answers True. No thread timing is relied on: the first frame is
drawn synchronously when a status opens, and the stop clears synchronously,
so what the stream holds is deterministic apart from extra frames.
"""

from __future__ import annotations

import io

from ffrwd.console import Console, written_size


class _Tty(io.StringIO):
    """A stream that claims to be a terminal."""

    def isatty(self) -> bool:
        return True


def test_written_size_picks_the_readable_unit() -> None:
    assert [
        written_size(count)
        for count in (0, 640, 310_000, 12 * 1024 * 1024, 86_000_000, 1_400_000_000)
    ] == ["0 bytes", "640 bytes", "303 KB", "12 MB", "82 MB", "1.3 GB"]


def test_say_writes_one_line_to_the_stream() -> None:
    stream = io.StringIO()
    Console(stream).say("fetching broadcast/tracks 1.0.0 (303 KB)")
    assert stream.getvalue() == "fetching broadcast/tracks 1.0.0 (303 KB)\n"


def test_quiet_silences_lines_and_spinner_both() -> None:
    stream = _Tty()
    console = Console(stream, quiet=True)
    with console.status("compiling"):
        console.say("fetching something")
    assert stream.getvalue() == ""


def test_the_spinner_never_writes_off_a_tty() -> None:
    stream = io.StringIO()
    console = Console(stream)
    with console.status("compiling"):
        pass
    assert stream.getvalue() == ""


def test_the_spinner_draws_and_clears_on_a_tty() -> None:
    stream = _Tty()
    console = Console(stream)
    with console.status("compiling"):
        pass
    written = stream.getvalue()
    assert written.startswith("\r- compiling")
    # The stop blanks the line and returns the cursor, so whatever prints
    # next starts on a clean column.
    assert written.endswith("\r" + " " * len("- compiling") + "\r")


def test_a_line_said_while_spinning_lands_on_its_own_clean_line() -> None:
    stream = _Tty()
    console = Console(stream)
    with console.status("installing"):
        console.say("fetching broadcast/tracks 1.0.0 (303 KB)")
    written = stream.getvalue()
    clear = "\r" + " " * len("- installing") + "\r"
    line = "fetching broadcast/tracks 1.0.0 (303 KB)\n"
    assert clear + line in written
    # And nothing of the spinner rides on the narration line itself.
    assert written.split(line)[0].endswith(clear)
