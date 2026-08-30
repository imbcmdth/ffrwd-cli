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
