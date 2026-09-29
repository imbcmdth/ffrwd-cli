"""Stdio pipes made to a size: raw video between two processes gets a pipe
of its own making, since the platform's default is what caps it
(:func:`ffrwd.pipes.anonymous` and the stdio wiring in ffrwd.execute).

Unit tier: the members are this Python, standing in for ffmpeg and a sidecar.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from ffrwd import pipes
from ffrwd.execute import _run_stage, _stdio_buffer, wires
from ffrwd.processes import (
    PIPE_BUFFER_LIMIT,
    AudioFormat,
    DataFormat,
    ProcessPlan,
    SidecarProcess,
    Stage,
    StreamEdge,
    VideoFormat,
)

_EXECUTE = sys.modules["ffrwd.execute"]

# More than a default pipe holds on any platform (4 KiB on Windows, 64 KiB on
# Linux), and less than Linux lets a process without privileges ask for.
_AHEAD = 512 << 10


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        view = view[os.write(fd, view) :]


def _holds(read: int, write: int) -> bool:
    """Whether `_AHEAD` bytes go into the pipe with nothing reading it."""
    done = threading.Event()

    def fill() -> None:
        _write_all(write, bytes(_AHEAD))
        done.set()

    threading.Thread(target=fill, daemon=True).start()
    held = done.wait(2)
    while not done.is_set():  # let a writer that is stuck finish
        os.read(read, 1 << 20)
        done.wait(0.05)
    return held


def test_a_sized_pipe_holds_what_a_default_one_cannot() -> None:
    for make, holds in ((lambda: pipes.anonymous(1 << 20), True), (os.pipe, False)):
        read, write = make()
        try:
            assert _holds(read, write) is holds
        finally:
            os.close(read)
            os.close(write)


def _edge(format: VideoFormat | AudioFormat | DataFormat) -> StreamEdge:
    return StreamEdge(source="s0", target="s1", ref="m0", format=format)


def test_raw_video_gets_a_sized_pipe_and_everything_else_the_default() -> None:
    assert (_stdio_buffer(_edge(VideoFormat(pix_fmt="rgba"))) or 0) >= PIPE_BUFFER_LIMIT
    assert _stdio_buffer(_edge(VideoFormat(codec="h264"))) is None
    assert _stdio_buffer(_edge(AudioFormat())) is None
    assert _stdio_buffer(_edge(DataFormat())) is None


# 4 MiB with no two neighbouring bytes alike, so a lost or reordered chunk shows.
_PAYLOAD = bytes(range(251)) * ((4 << 20) // 251)

_PRODUCE = (
    "import sys; out = sys.stdout.buffer\n"
    "data = bytes(range(251)) * ((4 << 20) // 251)\n"
    "for at in range(0, len(data), 1 << 16): out.write(data[at:at + (1 << 16)])\n"
    "out.flush()\n"
)


@pytest.mark.parametrize(
    ("format", "sized"),
    [(VideoFormat(pix_fmt="rgba"), True), (VideoFormat(codec="h264"), False)],
    ids=["rawvideo", "coded"],
)
def test_a_chain_carries_every_byte_through_the_pipe_it_was_given(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    format: VideoFormat,
    sized: bool,
) -> None:
    """Producer | consumer, chained through stdio: raw video crosses a pipe
    this runner made (a file descriptor, where Popen's own is PIPE), coded
    video the platform's default; either way every byte arrives in order."""
    python = getattr(sys, "_base_executable", None) or sys.executable
    received = tmp_path / "received.bin"
    consume = f"import sys; open({str(received)!r}, 'wb').write(sys.stdin.buffer.read())"
    edge = _edge(format)
    plan = ProcessPlan(
        processes=(
            SidecarProcess(id="s0", module="x", node="x"),
            SidecarProcess(id="s1", module="y", node="y"),
        ),
        edges=(edge,),
    )
    handed: dict[str, object] = {}
    real_spawn = _EXECUTE._spawn

    def _spawn_and_record(
        command: list[str], stdin: object, stdout: object, env: object = None
    ) -> subprocess.Popen[bytes]:
        pid = "s0" if command[-1] == _PRODUCE else "s1"
        handed[pid] = stdout if pid == "s0" else stdin
        return real_spawn(command, stdin, stdout, env=env)  # type: ignore[arg-type]

    monkeypatch.setattr(_EXECUTE, "_spawn", _spawn_and_record)
    result = _run_stage(
        plan,
        Stage(index=0, processes=("s0", "s1")),
        {"s0": [python, "-c", _PRODUCE], "s1": [python, "-c", consume]},
        served={},
        assigned=wires(plan),
        timeout=60,
        overwrite=False,
        echo=None,
        players={},
    )

    assert result.exit_code == 0, result.failures
    assert received.read_bytes() == _PAYLOAD
    if sized:
        assert isinstance(handed["s0"], int)
    else:
        assert handed["s0"] == subprocess.PIPE
