"""Named pipes, the transport stdio cannot provide.

Two processes wired end to end need nothing more than stdio: one writes its
stdout, the next reads its stdin. A process reading TWO streams has one stdin
and two producers, and that is what a named pipe is for -- a path the other
process opens like any other file. Windows makes one with ``CreateNamedPipeW``,
POSIX with :func:`os.mkfifo`: one interface, two creation calls.

This process is always the pipe's server and the process on the other end
always its client, so the bytes pass through here. :meth:`NamedPipe.wait`
blocks until that client opens the path and returns the stream to copy to or
from; :meth:`NamedPipe.close` releases everything the platform allocated and
unblocks a :meth:`wait` that is still waiting.

A pipe's BUFFER is how far the process on the other end runs ahead before its
writes wait, and it is a parameter: a plan whose edges carry a depth bound
sizes each pipe from that bound. Windows takes the size at creation; Linux
takes it afterwards with ``F_SETPIPE_SZ`` and caps it at its own maximum; the
rest take what they are given. Best effort everywhere -- a pipe that ends up
smaller than asked still carries the stream, and what a too-small one costs is
the run-time overflow :mod:`ffrwd.execute` reports.

The streams :meth:`NamedPipe.wait` hands back are UNBUFFERED, and that is a
correctness rule rather than a preference. Python's buffered reader returns
only when it has the whole size asked for, so it would sit on bytes that have
already crossed the pipe until more arrive -- and in a plan where one process
feeds two paths that meet again, the process that would send those extra bytes
is waiting on the very frame the withheld tail completes. Nothing here may hold
a byte it has been given.
"""

from __future__ import annotations

import contextlib
import errno
import os
import sys
import threading
import time
from pathlib import Path
from typing import BinaryIO

if sys.platform != "win32":
    import fcntl

__all__ = ["DEFAULT_BUFFER", "NamedPipe", "create"]

# How often a wait re-checks whether the client has arrived.
_POLL = 0.01
# The pipe's own buffer, per direction, for a pipe nothing asked to size.
DEFAULT_BUFFER = 1 << 16


class NamedPipe:
    """One end of a named pipe served by this process.

    `path` is what the process on the other end is given as a file name.
    `writing` is this end's direction: True when this process writes into the
    pipe and the client reads it, False when the client writes and this
    process reads.

    `buffer` is how many bytes the pipe holds before a write waits for a read.
    A plan sizes it from what the edge's own bound said it must hold; where
    the platform has no say in it, the attribute is what was asked for and not
    a promise.
    """

    def __init__(self, path: str, *, writing: bool, buffer: int = DEFAULT_BUFFER) -> None:
        self.path = path
        self.writing = writing
        self.buffer = buffer
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._stream: BinaryIO | None = None

    def wait(self, deadline: float) -> BinaryIO:
        """Block until the client opens `path`; the stream to copy through.

        The stream is unbuffered: a read hands back whatever has arrived, and a
        write may take only part of what it is given.

        Raises :class:`TimeoutError` at `deadline` and :class:`BrokenPipeError`
        if :meth:`close` ran first.
        """
        raise NotImplementedError

    def close(self) -> None:
        """Release the pipe. Safe to call twice, and from another thread."""
        raise NotImplementedError

    def _closed(self) -> BrokenPipeError:
        return BrokenPipeError(f"named pipe {self.path} was closed")

    def _timeout(self) -> TimeoutError:
        return TimeoutError(f"nothing opened {self.path}")


if sys.platform == "win32":
    import ctypes
    import msvcrt
    from ctypes import wintypes

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _kernel32.CreateNamedPipeW.restype = wintypes.HANDLE
    _kernel32.CreateNamedPipeW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.c_void_p,
    ]
    _kernel32.ConnectNamedPipe.restype = wintypes.BOOL
    _kernel32.ConnectNamedPipe.argtypes = [wintypes.HANDLE, ctypes.c_void_p]
    _kernel32.SetNamedPipeHandleState.restype = wintypes.BOOL
    _kernel32.SetNamedPipeHandleState.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.DWORD),
        ctypes.POINTER(wintypes.DWORD),
        ctypes.POINTER(wintypes.DWORD),
    ]
    _kernel32.FlushFileBuffers.restype = wintypes.BOOL
    _kernel32.FlushFileBuffers.argtypes = [wintypes.HANDLE]
    _kernel32.DisconnectNamedPipe.restype = wintypes.BOOL
    _kernel32.DisconnectNamedPipe.argtypes = [wintypes.HANDLE]
    _kernel32.CloseHandle.restype = wintypes.BOOL
    _kernel32.CloseHandle.argtypes = [wintypes.HANDLE]

    _PIPE_ACCESS_INBOUND = 0x00000001
    _PIPE_ACCESS_OUTBOUND = 0x00000002
    _PIPE_ACCESS_DUPLEX = 0x00000003
    _PIPE_WAIT = 0x00000000
    _PIPE_NOWAIT = 0x00000001
    _ERROR_PIPE_CONNECTED = 535
    _ERROR_PIPE_LISTENING = 536
    _ERROR_NO_DATA = 232
    _INVALID_HANDLE = ctypes.c_void_p(-1).value

    class _Pipe(NamedPipe):
        """A Windows named pipe: this process is the server, ffmpeg the client.

        Created non-blocking so a wait can give up, then switched to blocking
        once the client is there, which is what lets ordinary reads and writes
        move the stream afterwards.
        """

        def __init__(self, path: str, *, writing: bool, buffer: int = DEFAULT_BUFFER) -> None:
            super().__init__(path, writing=writing, buffer=buffer)
            # The reading end is DUPLEX, not INBOUND: switching the handle to
            # blocking mode below needs write-attributes access, which an
            # inbound handle lacks -- the switch fails, the handle stays
            # non-blocking, and a read of a momentarily empty pipe errors
            # instead of waiting.
            access = _PIPE_ACCESS_OUTBOUND if writing else _PIPE_ACCESS_DUPLEX
            handle = _kernel32.CreateNamedPipeW(
                path, access, _PIPE_NOWAIT, 1, buffer, buffer, 0, None
            )
            if handle == _INVALID_HANDLE:
                raise ctypes.WinError(ctypes.get_last_error())
            self._handle: int | None = handle
            self._owned = True

        def wait(self, deadline: float) -> BinaryIO:
            while True:
                with self._lock:
                    handle = self._handle
                    if self._stop.is_set() or handle is None:
                        raise self._closed()
                    if self._connect(handle):
                        break
                if time.monotonic() >= deadline:
                    raise self._timeout()
                time.sleep(_POLL)
            with self._lock:
                handle = self._handle
                if self._stop.is_set() or handle is None:
                    raise self._closed()
                mode = wintypes.DWORD(_PIPE_WAIT)
                if not _kernel32.SetNamedPipeHandleState(
                    handle, ctypes.byref(mode), None, None
                ):
                    raise ctypes.WinError(ctypes.get_last_error())
                flags = 0 if self.writing else os.O_RDONLY
                stream = os.fdopen(
                    msvcrt.open_osfhandle(handle, flags),
                    "wb" if self.writing else "rb",
                    buffering=0,
                )
                # The file object owns the handle from here on.
                self._owned = False
                self._stream = stream
                return stream

        @staticmethod
        def _connect(handle: int) -> bool:
            if _kernel32.ConnectNamedPipe(handle, None):
                return True
            code = ctypes.get_last_error()
            if code in (_ERROR_PIPE_CONNECTED, _ERROR_NO_DATA):
                # NO_DATA is a client that already came and went; what it
                # wrote still sits in the pipe, so the connect happened.
                return True
            if code == _ERROR_PIPE_LISTENING:
                return False
            raise ctypes.WinError(code)

        def close(self) -> None:
            self._stop.set()
            with self._lock:
                stream, handle, owned = self._stream, self._handle, self._owned
                self._stream, self._handle, self._owned = None, None, False
            if stream is not None:
                try:
                    if self.writing:
                        stream.flush()
                        # Blocks until the client has read everything written.
                        if handle is not None:
                            _kernel32.FlushFileBuffers(handle)
                            _kernel32.DisconnectNamedPipe(handle)
                except OSError:
                    pass
                try:
                    stream.close()
                except OSError:
                    pass
            elif handle is not None and owned:
                _kernel32.CloseHandle(handle)

    def create(
        directory: Path, name: str, *, writing: bool, buffer: int = DEFAULT_BUFFER
    ) -> NamedPipe:
        """A named pipe called `name`. `directory` is unused on Windows."""
        return _Pipe(
            rf"\\.\pipe\ffrwd-{os.getpid()}-{name}", writing=writing, buffer=buffer
        )

else:

    class _Pipe(NamedPipe):
        """A POSIX FIFO, opened once the process on the other end is there.

        The writing end can poll, since opening a FIFO for writing fails with
        ENXIO until a reader arrives. The reading end has no such answer and
        blocks in ``open``; :meth:`close` releases it by opening the FIFO
        read-write for an instant, which counts as the writer it was waiting
        for.
        """

        def __init__(self, path: str, *, writing: bool, buffer: int = DEFAULT_BUFFER) -> None:
            super().__init__(path, writing=writing, buffer=buffer)
            os.mkfifo(path, 0o600)

        def wait(self, deadline: float) -> BinaryIO:
            fd = self._open_write(deadline) if self.writing else self._open_read()
            os.set_blocking(fd, True)
            self._resize(fd)
            with self._lock:
                if self._stop.is_set():
                    os.close(fd)
                    raise self._closed()
                stream = os.fdopen(fd, "wb" if self.writing else "rb", buffering=0)
                self._stream = stream
                return stream

        def _resize(self, fd: int) -> None:
            """Ask for the buffer this pipe was sized for, where the kernel has one.

            Linux takes ``F_SETPIPE_SZ`` and caps it at
            ``/proc/sys/fs/pipe-max-size``; every other platform sizes its
            FIFOs itself and has nothing to set. Best effort either way -- the
            pipe works at whatever size it ends up with.
            """
            setter = getattr(fcntl, "F_SETPIPE_SZ", None)
            if setter is None or self.buffer <= DEFAULT_BUFFER:
                return
            with contextlib.suppress(OSError, ValueError):
                fcntl.fcntl(fd, setter, self.buffer)

        def _open_write(self, deadline: float) -> int:
            while True:
                if self._stop.is_set():
                    raise self._closed()
                try:
                    return os.open(self.path, os.O_WRONLY | os.O_NONBLOCK)
                except OSError as err:
                    if err.errno != errno.ENXIO:
                        raise
                if time.monotonic() >= deadline:
                    raise self._timeout()
                time.sleep(_POLL)

        def _open_read(self) -> int:
            if self._stop.is_set():
                raise self._closed()
            return os.open(self.path, os.O_RDONLY)

        def close(self) -> None:
            self._stop.set()
            with self._lock:
                stream, self._stream = self._stream, None
            if stream is None and not self.writing:
                # Releases a `_open_read` still waiting for its first writer.
                try:
                    os.close(os.open(self.path, os.O_RDWR | os.O_NONBLOCK))
                except OSError:
                    pass
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass
            try:
                os.unlink(self.path)
            except OSError:
                pass

    def create(
        directory: Path, name: str, *, writing: bool, buffer: int = DEFAULT_BUFFER
    ) -> NamedPipe:
        """A named pipe called `name`, made inside `directory`."""
        return _Pipe(str(directory / f"ffrwd-{name}"), writing=writing, buffer=buffer)
