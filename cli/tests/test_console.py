"""Tests for the CLI's narration: lines, sizes, the spinner, and the progress bar.

The spinner is exercised against two injected streams -- a plain StringIO,
which is what a pipe or a captured test run looks like, and one whose
``isatty`` answers True. The first frame is drawn synchronously when a status
opens and the stop clears synchronously, so what the stream holds is
deterministic apart from extra frames; the one check that waits on the
spinner's thread waits to see it write NOTHING.

The bar reads its clock through ``console._now``, and the `clock` fixture
holds that still: the redraw cap and the ETA are both timing, and a stopped
clock is what makes them assertable.
"""

from __future__ import annotations

import io
import time

import pytest

from ffrwd import console as console_module
from ffrwd.console import Console, written_size

MB = 1024 * 1024


class _Tty(io.StringIO):
    """A stream that claims to be a terminal."""

    def isatty(self) -> bool:
        return True


class _Clock:
    """A monotonic clock the check moves itself."""

    def __init__(self) -> None:
        self.reading = 0.0

    def __call__(self) -> float:
        return self.reading


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> _Clock:
    held = _Clock()
    monkeypatch.setattr(console_module, "_now", held)
    return held


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


def test_transient_redraws_in_place_and_end_transient_clears_it() -> None:
    stream = _Tty()
    console = Console(stream)
    console.transient("[####--------------------]  10% running")
    console.transient("[########----------------]  30% running")
    console.end_transient()
    written = stream.getvalue()
    assert written.startswith("\r[####--------------------]  10% running")
    assert "\r[########----------------]  30% running" in written
    last_line = "[########----------------]  30% running"
    assert written.endswith("\r" + " " * len(last_line) + "\r")


def test_transient_pads_over_a_shorter_previous_line() -> None:
    stream = _Tty()
    console = Console(stream)
    console.transient("[####--------------------]  10% running")
    console.transient("done")
    written = stream.getvalue()
    assert written.split("\r")[-1] == "done" + " " * (
        len("[####--------------------]  10% running") - len("done")
    )


def test_transient_never_writes_off_a_tty() -> None:
    stream = io.StringIO()
    console = Console(stream)
    console.transient("[####--------------------]  10% running")
    console.end_transient()
    assert stream.getvalue() == ""


def test_transient_is_silenced_by_quiet() -> None:
    stream = _Tty()
    console = Console(stream, quiet=True)
    console.transient("[####--------------------]  10% running")
    console.end_transient()
    assert stream.getvalue() == ""


def test_progress_draws_a_bar_with_figures_and_an_eta_then_clears(clock: _Clock) -> None:
    stream = _Tty()
    report = Console(stream).progress("yolo26n.onnx")
    total = 128 * MB
    report(1 * MB, total)
    clock.reading = 2.0
    report(61 * MB, total)
    report(total, total)

    written = stream.getvalue()
    assert "\ryolo26n.onnx  [>                   ]   0%  1 MB / 128 MB" in written
    # 67 of 128 MB left at the 30.5 MB a second measured so far: two more.
    line = "yolo26n.onnx  [=========>          ]  47%  61 MB / 128 MB  eta 0:02"
    assert "\r" + line in written
    assert len(line) < 80
    # Reaching the total blanks the line, so the next narration starts clean.
    assert written.endswith("\r" + " " * len(line) + "\r")


def test_progress_leaves_the_eta_blank_until_a_second_is_measured(clock: _Clock) -> None:
    stream = _Tty()
    report = Console(stream).progress("yolo26n.onnx")
    clock.reading = 0.5
    report(61 * MB, 128 * MB)
    assert stream.getvalue().endswith("61 MB / 128 MB")


def test_progress_draws_nothing_under_a_megabyte(clock: _Clock) -> None:
    stream = _Tty()
    report = Console(stream).progress("small.onnx")
    report(300_000, 900_000)
    clock.reading = 2.0
    report(900_000, 900_000)
    assert stream.getvalue() == ""


def test_progress_never_writes_off_a_tty(clock: _Clock) -> None:
    stream = io.StringIO()
    report = Console(stream).progress("yolo26n.onnx")
    report(1 * MB, 128 * MB)
    clock.reading = 2.0
    report(128 * MB, 128 * MB)
    assert stream.getvalue() == ""


def test_progress_redraws_at_most_ten_times_a_second(clock: _Clock) -> None:
    stream = _Tty()
    report = Console(stream).progress("big.bin")
    for number in range(1, 11):
        clock.reading = number * 0.02
        report(number * MB, 128 * MB)
    # Ten blocks over a fifth of a second: two frames, not ten.
    assert stream.getvalue().count("\rbig.bin") == 2


def test_progress_without_a_total_shows_what_has_arrived(clock: _Clock) -> None:
    stream = _Tty()
    report = Console(stream).progress("archive")
    report(2 * MB, None)
    clock.reading = 1.0
    report(5 * MB, None)
    report(5 * MB, 5 * MB)

    written = stream.getvalue()
    assert "\rarchive  2 MB" in written
    assert "\rarchive  5 MB" in written
    assert written.endswith("\r" + " " * len("archive  5 MB") + "\r")


def test_progress_times_the_next_download_from_its_own_first_byte(clock: _Clock) -> None:
    stream = _Tty()
    report = Console(stream).progress("downloading")
    total = 4 * MB
    report(2 * MB, total)
    report(total, total)
    clock.reading = 5.0
    report(1 * MB, total)
    clock.reading = 6.0
    report(2 * MB, total)

    written = stream.getvalue()
    # Half of 4 MB in the one second since this download's first byte.
    assert "eta 0:01" in written
    assert "eta 0:06" not in written


def test_progress_shortens_a_label_that_would_run_past_the_terminal(clock: _Clock) -> None:
    stream = _Tty()
    report = Console(stream).progress("onnxruntime-win-x64-gpu-cuda12-1.20.1.zip")
    clock.reading = 2.0
    report(61 * MB, 128 * MB)
    drawn = stream.getvalue().lstrip("\r")
    assert len(drawn) < 80
    assert drawn.startswith("onnxruntime-win-x64-gpu-cuda12-...  [")


def test_the_spinner_leaves_a_standing_progress_line_alone(clock: _Clock) -> None:
    stream = _Tty()
    console = Console(stream)
    with console.status("installing"):
        console.progress("downloading")(2 * MB, 4 * MB)
        standing = stream.getvalue()
        # Several of the spinner thread's own ticks: it draws none of them,
        # so the bar is not flickered over while it stands.
        time.sleep(4 * console_module._TICK)
        assert stream.getvalue() == standing


def test_a_line_said_over_a_standing_progress_line_starts_clean(clock: _Clock) -> None:
    stream = _Tty()
    console = Console(stream)
    console.progress("downloading")(2 * MB, 4 * MB)
    console.say("model model.onnx (4 MB) from imbcmdth/yolo26-onnx")
    written = stream.getvalue()
    bar = "downloading  [=========>          ]  50%  2 MB / 4 MB"
    assert written.endswith(
        "\r" + " " * len(bar) + "\rmodel model.onnx (4 MB) from imbcmdth/yolo26-onnx\n"
    )
