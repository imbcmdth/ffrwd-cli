"""What the CLI says while it works: narration lines, and a spinner between them.

Narration is one line per meaningful step, present tense, to stderr -- never
stdout, which carries what a script would parse. The library modules stay
silent by contract; they take an ``Announce`` callback and the CLI wires it
to :meth:`Console.say`. ``--quiet`` drops the lines and the spinner both.

The spinner is plain ASCII over carriage returns, and runs only when the
stream is a TTY: a pipe or a CI log carries the narration lines alone. A
narration line printed while it spins clears the spinner's line first, so
nothing interleaves.
"""

from __future__ import annotations

import itertools
import sys
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import TextIO

__all__ = ["Announce", "Console", "written_size"]

# What a step tells the caller, so nothing in the library decides where a
# line prints.
Announce = Callable[[str], None]

_FRAMES = "-\\|/"
_TICK = 0.1


def written_size(count: int) -> str:
    """`count` bytes as a short figure: 640 bytes, 303 KB, 82 MB, 1.2 GB."""
    if count < 1024:
        return f"{count} bytes"
    if count < 1024 * 1024:
        return f"{count / 1024:.0f} KB"
    if count < 1024 * 1024 * 1024:
        return f"{count / (1024 * 1024):.0f} MB"
    return f"{count / (1024 * 1024 * 1024):.1f} GB"


class _Spinner(threading.Thread):
    """One spinner line, redrawn in place until stopped.

    Every write happens under the caller's lock, which is how a narration
    line and a frame never share a line: whoever holds the lock clears first.
    """

    def __init__(self, stream: TextIO, label: str, lock: threading.Lock) -> None:
        super().__init__(daemon=True)
        self._stream = stream
        self._label = label
        self._lock = lock
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
                if not self._done.is_set():
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
        """Print one narration line, the spinner cleared out of its way first."""
        if self.quiet:
            return
        with self._lock:
            if self._spinner is not None:
                self._spinner.clear()
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
        spinner = _Spinner(self.stream, label, self._lock)
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
            if self._transient_width:
                self.stream.write("\r" + " " * self._transient_width + "\r")
                self.stream.flush()
                self._transient_width = 0

    def _is_tty(self) -> bool:
        try:
            return bool(self.stream.isatty())
        except (AttributeError, ValueError):
            return False
