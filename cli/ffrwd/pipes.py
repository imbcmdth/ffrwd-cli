"""Pipes: the names of the ones the relay serves, and the ones with none.

Two processes wired end to end need nothing more than stdio: one writes its
stdout, the next reads its stdin, through a pipe with no name
(:func:`anonymous`). Any other edge is a NAMED pipe at each end, a path the
process on that end opens like any other file, and between the two is the
relay (:mod:`ffrwd.relay`), which makes both pipes, serves them and copies
one into the other. Nothing here makes a named pipe: :func:`path` is what
one is called, which a process's argv names before the relay makes it.

A pipe's BUFFER is how far the process on the other end runs ahead before its
writes wait, and it is a parameter: a plan whose edges carry a depth bound
sizes each pipe from that bound. Windows takes the size at creation; Linux
takes it afterwards with ``F_SETPIPE_SZ``, no more than
``/proc/sys/fs/pipe-max-size`` (a larger ask is refused outright, not
shortened, so it is capped first); the rest take what they are given.
Best effort everywhere -- a pipe that ends up smaller than asked still
carries the stream, and what a too-small one costs is the run-time overflow
:mod:`ffrwd.execute` reports. The relay sizes the named ones the same way.

``subprocess.PIPE`` gives a pipe with no name the platform's default buffer,
which on Windows is what caps raw video through a chain, not the processes on
either end; :func:`anonymous` makes one of the size asked for.
"""

from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path

if sys.platform != "win32":
    import fcntl

__all__ = ["DEFAULT_BUFFER", "anonymous", "path"]

# The pipe's own buffer, per direction, for a pipe nothing asked to size.
DEFAULT_BUFFER = 1 << 16
# Where Linux says how big a process without privileges may make a pipe.
_PIPE_MAX_SIZE = Path("/proc/sys/fs/pipe-max-size")


if sys.platform == "win32":

    def _set_size(fd: int, buffer: int) -> None:
        """Windows sizes a pipe when it is made, and never afterwards."""

else:

    def _set_size(fd: int, buffer: int) -> None:
        """Ask Linux for a pipe of `buffer` bytes, no more than it allows; a
        platform with no ``F_SETPIPE_SZ`` sizes its pipes itself. Best
        effort: the pipe carries its stream at whatever size it ends up."""
        setter = getattr(fcntl, "F_SETPIPE_SZ", None)
        if setter is None or buffer <= DEFAULT_BUFFER:
            return
        try:
            buffer = min(buffer, int(_PIPE_MAX_SIZE.read_text().strip()))
        except (OSError, ValueError):
            pass
        with contextlib.suppress(OSError, ValueError):
            fcntl.fcntl(fd, setter, buffer)


def anonymous(buffer: int = DEFAULT_BUFFER) -> tuple[int, int]:
    """A pipe with no name holding `buffer` bytes: ``(read, write)``, two
    file descriptors this process owns, neither inherited. Handed to
    :class:`subprocess.Popen` as a child's stdin or stdout, an end becomes
    the child's own; the parent then closes its copy."""
    if sys.platform == "win32":
        import _winapi
        import msvcrt

        read, write = _winapi.CreatePipe(None, buffer)
        return msvcrt.open_osfhandle(read, os.O_RDONLY), msvcrt.open_osfhandle(write, 0)
    read, write = os.pipe()
    _set_size(write, buffer)
    return read, write


def path(directory: Path, name: str) -> str:
    """What the named pipe called `name` is called: a pipe in Windows's own
    namespace, unique to this process, or a FIFO inside `directory`."""
    if sys.platform == "win32":
        return rf"\\.\pipe\ffrwd-{os.getpid()}-{name}"
    return str(directory / f"ffrwd-{name}")
