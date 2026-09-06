"""What the CLI says while it works: narration lines, and a spinner between them.

Narration is one line per meaningful step, present tense, to stderr -- never
stdout, which carries what a script would parse. The library modules stay
silent by contract; they take an ``Announce`` callback and the CLI wires it
to :meth:`Console.say`. ``--quiet`` drops the lines and the spinner both.

The spinner is plain ASCII over carriage returns, and runs only when the
stream is a TTY: a pipe or a CI log carries the narration lines alone. A
narration line printed while it spins clears the spinner's line first, so
nothing interleaves.

A transfer -- a download, or an upload of a file input -- reports through a
``Progress`` the same way, and :meth:`Console.progress` turns one into a bar
redrawn over :meth:`Console.transient`.
"""

from __future__ import annotations

import itertools
import sys
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import TextIO

__all__ = ["Announce", "Console", "Progress", "written_size"]

# What a step tells the caller, so nothing in the library decides where a
# line prints.
Announce = Callable[[str], None]

# How a transfer reports itself: the bytes so far, and the total when it is
# known. The last call of a transfer names the total it ended with -- a
# transfer whose total was never known names the count it reached -- and that
# is what ends the line.
Progress = Callable[[int, int | None], None]

_FRAMES = "-\\|/"
_TICK = 0.1

# The clock every bar reads. A seam, so a check can hold time still.
_now = time.monotonic

# Nothing draws for a transfer smaller than this.
_PROGRESS_FLOOR = 1024 * 1024
_BAR_WIDTH = 20
_REDRAW = 0.1
_ETA_AFTER = 1.0
_COLUMNS = 79
# How far back the rate behind the ETA is measured.
_RATE_WINDOW = 5.0


def written_size(count: int) -> str:
    """`count` bytes as a short figure: 640 bytes, 303 KB, 82 MB, 1.2 GB."""
    if count < 1024:
        return f"{count} bytes"
    if count < 1024 * 1024:
        return f"{count / 1024:.0f} KB"
    if count < 1024 * 1024 * 1024:
        return f"{count / (1024 * 1024):.0f} MB"
    return f"{count / (1024 * 1024 * 1024):.1f} GB"


def _elapsed_time(seconds: float) -> str:
    """`seconds` as ``0:12`` or ``1:02:03``."""
    whole = int(seconds)
    hours, rest = divmod(whole, 3600)
    minutes, second = divmod(rest, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{second:02d}"
    return f"{minutes}:{second:02d}"


def _bar(fraction: float) -> str:
    """The fixed-width bar for `fraction`: ``=========>          ``."""
    filled = min(_BAR_WIDTH, int(_BAR_WIDTH * fraction))
    if filled >= _BAR_WIDTH:
        return "=" * _BAR_WIDTH
    return "=" * filled + ">" + " " * (_BAR_WIDTH - filled - 1)


class _Bar:
    """One transfer's line: where it started, when it last drew, and its recent rate.

    Reused across transfers -- one `Progress` covers a whole install -- so a
    count that does not continue the previous one starts the timing again.
    The rate comes off the counts of the last few redraws rather than the
    whole transfer, so a connection that bursts and then settles reports the
    rate it settled at.
    """

    def __init__(self, label: str) -> None:
        self.label = label
        self._start = 0.0
        self._last_draw = 0.0
        self._counted = 0
        self._samples: deque[tuple[float, int]] = deque()
        self.drawn = False
        self._ended = True

    def _restart(self) -> None:
        self._start = _now()
        self._last_draw = 0.0
        self._counted = 0
        self._samples.clear()
        self.drawn = False
        self._ended = False

    def ends(self, done: int, total: int | None) -> bool:
        """Whether this call is the transfer's last, the state carried forward."""
        if self._ended or done < self._counted:
            self._restart()
        self._counted = done
        finished = total is not None and done >= total
        self._ended = finished
        return finished

    def line(self, done: int, total: int | None) -> str | None:
        """The line to redraw, or None when this call draws nothing."""
        known_large = total is not None and total > _PROGRESS_FLOOR
        if not known_large and done <= _PROGRESS_FLOOR:
            return None
        now = _now()
        if self.drawn and now - self._last_draw < _REDRAW:
            return None
        self._last_draw = now
        self.drawn = True
        self._sample(now, done)
        return _fit(self.label, self._tail(done, total, now - self._start))

    def _sample(self, now: float, done: int) -> None:
        """Record this redraw's count, dropping what has fallen out of the window.

        One sample older than the window is kept, so the span it measures
        covers the whole window rather than stopping just inside it.
        """
        self._samples.append((now, done))
        while len(self._samples) > 1 and now - self._samples[1][0] > _RATE_WINDOW:
            self._samples.popleft()

    def _rate(self, done: int, elapsed: float) -> float:
        """Bytes a second across the window, or the average while it holds one sample."""
        first_at, first_done = self._samples[0]
        last_at, last_done = self._samples[-1]
        span = last_at - first_at
        if len(self._samples) > 1 and span > 0:
            return (last_done - first_done) / span
        return done / elapsed if elapsed else 0.0

    def _tail(self, done: int, total: int | None, elapsed: float) -> str:
        if total is None:
            return f"  {written_size(done)}"
        percent = min(100, int(100 * done / total)) if total else 100
        sizes = f"{written_size(done)} / {written_size(total)}"
        eta = ""
        rate = self._rate(done, elapsed)
        if elapsed >= _ETA_AFTER and rate > 0:
            eta = f"  eta {_elapsed_time(max(0, total - done) / rate)}"
        return f"  [{_bar(done / total if total else 1.0)}] {percent:3d}%  {sizes}{eta}"


def _fit(label: str, tail: str) -> str:
    """`label` and `tail` in one line, the label shortened to keep it in the terminal."""
    room = max(0, _COLUMNS - len(tail))
    if len(label) <= room:
        return label + tail
    return (label[: max(0, room - 3)] + "...")[:room] + tail


class _Spinner(threading.Thread):
    """One spinner line, redrawn in place until stopped.

    Every write happens under the caller's lock, which is how a narration
    line and a frame never share a line: whoever holds the lock clears first.
    `held` says a redrawn line of someone else's stands there, and the
    spinner leaves it alone until it is gone.
    """

    def __init__(
        self,
        stream: TextIO,
        label: str,
        lock: threading.Lock,
        held: Callable[[], bool] = lambda: False,
    ) -> None:
        super().__init__(daemon=True)
        self._stream = stream
        self._label = label
        self._lock = lock
        self._held = held
        self._done = threading.Event()
        self._frames = itertools.cycle(_FRAMES)
        self._width = 0

    def draw(self) -> None:
        """Write the next frame over the current line. Caller holds the lock."""
        line = f"{next(self._frames)} {self._label}" if self._label else next(self._frames)
        self._stream.write("\r" + line)
        self._stream.flush()
        self._width = max(self._width, len(line))

    def clear(self) -> None:
        """Blank the spinner's line. Caller holds the lock."""
        if self._width:
            self._stream.write("\r" + " " * self._width + "\r")
            self._stream.flush()
            self._width = 0

    def run(self) -> None:
        while not self._done.wait(_TICK):
            with self._lock:
                if not self._done.is_set() and not self._held():
                    self.draw()

    def stop(self) -> None:
        self._done.set()
        self.join(timeout=2.0)
        with self._lock:
            self.clear()


class Console:
    """The CLI's one voice on stderr: `say` a line, or `status` a spinner.

    `quiet` silences both. The stream defaults to ``sys.stderr`` read at
    write time, so a captured stderr is the one written to.
    """

    def __init__(self, stream: TextIO | None = None, *, quiet: bool = False) -> None:
        self._stream = stream
        self.quiet = quiet
        self._lock = threading.Lock()
        self._spinner: _Spinner | None = None
        self._transient_width = 0

    @property
    def stream(self) -> TextIO:
        return self._stream if self._stream is not None else sys.stderr

    def say(self, line: str) -> None:
        """Print one narration line, whatever stands on the line cleared first."""
        if self.quiet:
            return
        with self._lock:
            if self._spinner is not None:
                self._spinner.clear()
            self._clear_transient()
            self.stream.write(line + "\n")
            self.stream.flush()

    @contextmanager
    def status(self, label: str) -> Iterator[None]:
        """A spinner naming `label` while the body runs.

        Silent when quiet, off a TTY, or already spinning: the lines still
        print, the animation just does not.
        """
        if self.quiet or self._spinner is not None or not self._is_tty():
            yield
            return
        spinner = _Spinner(
            self.stream, label, self._lock, lambda: self._transient_width > 0
        )
        self._spinner = spinner
        with self._lock:
            spinner.draw()
        spinner.start()
        try:
            yield
        finally:
            self._spinner = None
            spinner.stop()

    def transient(self, line: str) -> None:
        """Redraw one line in place over ``\\r`` -- a poll's progress line.

        Silent when quiet or off a TTY, the same as the spinner: a pipe or a
        CI log gets nothing until the final result prints.
        """
        if self.quiet or not self._is_tty():
            return
        with self._lock:
            if self._spinner is not None:
                self._spinner.clear()
            pad = max(0, self._transient_width - len(line))
            self.stream.write("\r" + line + " " * pad)
            self.stream.flush()
            self._transient_width = len(line)

    def end_transient(self) -> None:
        """Blank the last `transient` line so whatever prints next starts clean."""
        if self.quiet or not self._is_tty():
            return
        with self._lock:
            self._clear_transient()

    def _clear_transient(self) -> None:
        """Blank the redrawn line, if one stands. Caller holds the lock."""
        if self._transient_width:
            self.stream.write("\r" + " " * self._transient_width + "\r")
            self.stream.flush()
            self._transient_width = 0

    def progress(self, label: str) -> Progress:
        """A `Progress` drawing `label`'s bar in place, cleared when it finishes.

        ``yolo26n.onnx  [=========>          ]  47%  61 MB / 128 MB  eta 0:12``.
        Nothing draws below a megabyte, and nothing off a TTY or under
        `quiet`, which `transient` already answers for. Redraws are capped at
        ten a second, and the ETA -- the rate over the last few seconds --
        appears once a second of it has been measured. One `Progress` covers
        however many transfers a command makes, each drawing its own line.
        """
        bar = _Bar(label)

        def report(done: int, total: int | None) -> None:
            if bar.ends(done, total):
                if bar.drawn:
                    self.end_transient()
                return
            line = bar.line(done, total)
            if line is not None:
                self.transient(line)

        return report

    def _is_tty(self) -> bool:
        try:
            return bool(self.stream.isatty())
        except (AttributeError, ValueError):
            return False
