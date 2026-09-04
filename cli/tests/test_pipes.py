"""The named pipe transport, driven from plain Python threads.

Unit tier: no ffmpeg and no fixtures -- the client on the other end is this
process opening the path as a file, which is exactly what ffmpeg does.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import pytest

from ffrwd import pipes


def _drain(pipe: pipes.NamedPipe, into: dict[str, object]) -> None:
    try:
        stream = pipe.wait(time.monotonic() + 10)
        total = 0
        while True:
            chunk = stream.read(1 << 16)
            if not chunk:
                break
            total += len(chunk)
        into["bytes"] = total
    except Exception as err:  # noqa: BLE001 -- the assertion reports it whole
        into["error"] = repr(err)


def test_a_reading_end_waits_out_an_empty_pipe(tmp_path: Path) -> None:
    """A slow writer leaves the pipe empty between bursts; the read blocks
    until the next burst instead of erroring, however the pipe was made."""
    pipe = pipes.create(tmp_path, "slow", writing=False)
    result: dict[str, object] = {}
    reader = threading.Thread(target=_drain, args=(pipe, result), daemon=True)
    reader.start()
    time.sleep(0.1)
    with open(pipe.path, "wb", buffering=0) as client:
        for _ in range(4):
            client.write(b"a" * 200_000)  # past the pipe's own buffer
            time.sleep(0.1)
    reader.join(10)
    pipe.close()
    assert result == {"bytes": 800_000}


@pytest.mark.skipif(
    sys.platform != "win32",
    reason="a FIFO writer waits for its reader, so the client cannot leave first",
)
def test_a_client_that_came_and_went_still_hands_over_its_bytes(tmp_path: Path) -> None:
    """A fast client connects, writes less than the buffer, and closes before
    the server ever looks; what it wrote is read, then EOF."""
    pipe = pipes.create(tmp_path, "fast", writing=False)
    with open(pipe.path, "wb", buffering=0) as client:
        client.write(b"b" * 1000)
    result: dict[str, object] = {}
    reader = threading.Thread(target=_drain, args=(pipe, result), daemon=True)
    reader.start()
    reader.join(10)
    pipe.close()
    assert result == {"bytes": 1000}


def _first_read(pipe: pipes.NamedPipe, into: dict[str, object]) -> None:
    try:
        into["chunk"] = pipe.wait(time.monotonic() + 10).read(1 << 16)
    except Exception as err:  # noqa: BLE001 -- the assertion reports it whole
        into["error"] = repr(err)


def test_a_read_hands_back_what_has_arrived_rather_than_a_full_chunk(
    tmp_path: Path,
) -> None:
    """A reading end holds nothing back.

    The client writes far less than the size the read asks for and then stops,
    the pipe still open. Waiting for the rest would wedge a plan whose writer
    is itself waiting on the frame those bytes finish, so they come back as
    they are.
    """
    pipe = pipes.create(tmp_path, "partial", writing=False)
    result: dict[str, object] = {}
    reader = threading.Thread(target=_first_read, args=(pipe, result), daemon=True)
    reader.start()
    with open(pipe.path, "wb", buffering=0) as client:
        client.write(b"c" * 300)
        reader.join(10)
        assert result == {"chunk": b"c" * 300}
    pipe.close()


@pytest.mark.skipif(
    sys.platform != "win32",
    reason="pins the Windows connect-at-creation ordering; POSIX opens a FIFO instead",
)
def test_a_windows_pipe_starts_connecting_as_soon_as_it_is_made(tmp_path: Path) -> None:
    """The connect begins at `create`, not at the first `wait()`.

    A plan makes every named pipe before spawning anything (`plan_argv` runs
    ahead of `execute_plan`'s stage loop), but the SERVER used to only start
    listening once a pump thread got around to calling `wait()` -- after every
    member of the stage was already spawned. A client fast enough to open and
    operate on the path before that pump thread ran found a pipe nobody had
    told to accept a connection yet. Starting the connect in `create` itself
    closes that window: `wait()` now only waits for a connect already under
    way, so a fast client is never ahead of it.
    """
    pipe = pipes.create(tmp_path, "eager", writing=True)
    try:
        assert pipe._connector.is_alive()  # type: ignore[attr-defined]
    finally:
        pipe.close()


@pytest.mark.skipif(
    sys.platform != "win32",
    reason="the Windows connect has a deadline; a POSIX FIFO open blocks for its peer",
)
def test_a_wait_with_no_client_times_out_at_its_deadline(tmp_path: Path) -> None:
    """Nobody ever opens the path: `wait` gives up at `deadline` rather than
    hanging on a connect that has nothing to wait for."""
    pipe = pipes.create(tmp_path, "lonely", writing=False)
    try:
        with pytest.raises(TimeoutError):
            pipe.wait(time.monotonic() + 0.2)
    finally:
        pipe.close()


@pytest.mark.skipif(
    sys.platform != "win32",
    reason="the Windows connect is cancelled by close; a POSIX FIFO open blocks for its peer",
)
def test_a_close_unblocks_a_wait_that_has_no_client_coming(tmp_path: Path) -> None:
    """A stage tears down mid-run: `close`, called from another thread, ends
    a `wait` still parked for a client that was never going to arrive."""
    pipe = pipes.create(tmp_path, "abandoned", writing=False)
    result: dict[str, object] = {}

    def block() -> None:
        try:
            pipe.wait(time.monotonic() + 30)
        except Exception as err:  # noqa: BLE001 -- the assertion reports it whole
            result["error"] = repr(err)

    waiter = threading.Thread(target=block, daemon=True)
    waiter.start()
    time.sleep(0.2)
    pipe.close()
    waiter.join(10)
    assert not waiter.is_alive()
    assert isinstance(result.get("error"), str) and "BrokenPipeError" in result["error"]


def test_a_writing_end_hands_its_bytes_over_without_being_flushed(
    tmp_path: Path,
) -> None:
    """The mirror: what this process writes reaches the client at once, rather
    than waiting in a buffer for enough more to be worth sending."""
    pipe = pipes.create(tmp_path, "unflushed", writing=True)
    result: dict[str, object] = {}

    def serve() -> None:
        try:
            pipe.wait(time.monotonic() + 10).write(b"d" * 300)
        except Exception as err:  # noqa: BLE001 -- the assertion reports it whole
            result["error"] = repr(err)

    def receive() -> None:
        try:
            with open(pipe.path, "rb", buffering=0) as client:
                result["chunk"] = client.read(1 << 16)
        except Exception as err:  # noqa: BLE001 -- the assertion reports it whole
            result["error"] = repr(err)

    server = threading.Thread(target=serve, daemon=True)
    client = threading.Thread(target=receive, daemon=True)
    server.start()
    client.start()
    server.join(10)
    client.join(10)
    # Read before closing: closing flushes, and a flush would hide the very
    # buffer this is about.
    handed = dict(result)
    pipe.close()
    assert handed == {"chunk": b"d" * 300}
