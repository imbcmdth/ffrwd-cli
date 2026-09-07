"""What ffmpeg is asked for while it works, and what is read back off it.

Unit tier: the parser runs over bytes the check hands it, and the argv is
compared as a list. Nothing is spawned and no ffmpeg is needed -- the reader
sees the same stream either way, and a chunk boundary in the middle of a line
is something a check can arrange and a real run cannot.
"""

from __future__ import annotations

import importlib
import subprocess

import pytest

from ffrwd.compiler import compile_all, emitted_commands
from ffrwd.console import Work
from ffrwd.execute import (
    _PROGRESS_ARGS,
    _spawn_argv,
    _WorkReader,
    execute,
    terminal_member,
)
from ffrwd.ir import Graph, Node, Output, SinkUnit, StreamType
from ffrwd.processes import (
    PIPE,
    AudioFormat,
    FfmpegProcess,
    ProcessPlan,
    StreamEdge,
    VideoFormat,
    external_ids,
    partition,
)

# The package exports `execute` the function, so the module it lives in is
# reached the way `test_public_api` reaches one.
execute_module = importlib.import_module("ffrwd.execute")

QUERY = "COPY (SELECT a.video[1] FROM input('a.mp4') a) TO 'out.mp4'"

# One period of what `-progress` writes, as ffmpeg writes it: the per-stream
# quality key among the rest, and the same instant spelled three ways.
_BLOCK = (
    "frame=486\n"
    "fps=61.00\n"
    "stream_0_0_q=28.0\n"
    "bitrate=2100.0kbits/s\n"
    "total_size=524288\n"
    "out_time_us=20150000\n"
    "out_time_ms=20150000\n"
    "out_time=00:00:20.150000\n"
    "dup_frames=0\n"
    "drop_frames=0\n"
    "speed=1.94x\n"
    "progress=continue\n"
)
_LAST_BLOCK = (
    "frame=960\n"
    "fps=59.00\n"
    "bitrate=N/A\n"
    "total_size=1048576\n"
    "out_time=00:00:40.000000\n"
    "speed=1.9x\n"
    "progress=end\n"
)
_OPENING = "Input #0, matroska,webm, from 'psych.mkv':\n"
_WARNING = "[libx264 @ 0000021d] non-strictly-monotonic PTS\n"
_CLOSING = "video:1024kB audio:64kB muxing overhead: 0.4%\n"


def _out(ref: str, type_: StreamType = "video") -> Output:
    return Output(ref=ref, type=type_, name=None, metadata={})


def _read(text: str, *, chunk: int) -> tuple[list[Work], str]:
    """`text` fed to a reader `chunk` bytes at a time; its readings and its log."""
    seen: list[Work] = []
    log: list[bytes] = []
    reader = _WorkReader(log, seen.append)
    raw = text.encode()
    for start in range(0, len(raw), chunk):
        reader.feed(raw[start : start + chunk])
    reader.finish()
    return seen, b"".join(log).decode()


# --- reading a member's stderr ----------------------------------------------


def test_the_blocks_are_read_and_the_log_keeps_everything_else() -> None:
    """13 bytes at a time, so a line and a block both span chunks."""
    seen, log = _read(_OPENING + _BLOCK + _WARNING + _LAST_BLOCK + _CLOSING, chunk=13)

    assert seen == [
        Work(
            out_time=20.15,
            fps=61.0,
            speed=1.94,
            bitrate=2100.0,
            total_size=524288,
            done=False,
        ),
        Work(
            out_time=40.0,
            fps=59.0,
            speed=1.9,
            bitrate=None,
            total_size=1048576,
            done=True,
        ),
    ]
    assert log == _OPENING + _WARNING + _CLOSING


def test_a_stream_that_stops_short_still_ends_the_run() -> None:
    """A killed member writes no ``progress=end``, and leaves no line standing."""
    seen, log = _read(_OPENING + _BLOCK + "frame=500\nfps=60.00", chunk=4096)

    assert [work.done for work in seen] == [False, True]
    # The last reading holds where the run had got to, the block it was in
    # having named no time of its own.
    assert seen[-1].out_time == 20.15
    assert log == _OPENING


# --- which command, and which member ----------------------------------------


def _chain() -> ProcessPlan:
    """decode -> one external node -> encode: three processes in a line."""
    g = Graph(input_paths=["a.mp4"], sources={"a": 0})
    g.nodes["e0"] = Node(
        id="e0", filter="negate", args={}, inputs=["src:a:v:0"], outputs=["video"]
    )
    g.sinks = [SinkUnit(outputs=[_out("e0")], path="out.mp4")]
    return partition(g, external=external_ids("e0"))


def _two_producers() -> ProcessPlan:
    """Two ffmpeg processes feeding a third that muxes what they write."""
    video = Graph(input_paths=["a.mp4"], sources={"s": 0})
    video.sinks = [SinkUnit(outputs=[_out("src:s:v:0")], path=PIPE)]
    audio = Graph(input_paths=["a.mp4"], sources={"s": 0})
    audio.sinks = [SinkUnit(outputs=[_out("src:s:a:0", "audio")], path=PIPE)]
    mux = Graph(input_paths=[PIPE, PIPE], sources={"v": 0, "a": 1})
    mux.sinks = [
        SinkUnit(outputs=[_out("src:v:v:0"), _out("src:a:a:0", "audio")], path="out.mkv")
    ]
    return ProcessPlan(
        processes=(
            FfmpegProcess(id="video", graph=video),
            FfmpegProcess(id="audio", graph=audio),
            FfmpegProcess(id="mux", graph=mux),
        ),
        edges=(
            StreamEdge(source="video", target="mux", ref="src:s:v:0", format=VideoFormat()),
            StreamEdge(source="audio", target="mux", ref="src:s:a:0", format=AudioFormat()),
        ),
    )


def test_the_terminal_member_is_the_one_writing_the_destination() -> None:
    plan = _chain()
    writing = [
        p.id for p in plan.ffmpeg if any(sink.path == "out.mp4" for sink in p.graph.sinks)
    ]

    assert [terminal_member(plan)] == writing


def test_a_member_feeding_another_is_never_the_terminal_one() -> None:
    """Three ffmpegs at once, and only the muxer writes a file."""
    assert terminal_member(_two_producers()) == "mux"


def test_only_the_member_asked_for_progress_carries_the_flags() -> None:
    process = FfmpegProcess(id="mux", graph=Graph(input_paths=["a.mp4"], sources={"a": 0}))
    argv = ["ffmpeg", "-i", "a.mp4", "out.mp4"]

    assert _spawn_argv(process, argv, overwrite=True, progress=True) == [
        "ffmpeg",
        "-hide_banner",
        "-y",
        *_PROGRESS_ARGS,
        "-i",
        "a.mp4",
        "out.mp4",
    ]
    assert _spawn_argv(process, argv, overwrite=True) == [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-i",
        "a.mp4",
        "out.mp4",
    ]


def test_a_watched_command_asks_ffmpeg_for_its_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The flags are global options, so they go ahead of the first input."""
    monkeypatch.setattr(
        execute_module, "_run_watched", lambda argv, timeout, **kw: (0, "")
    )
    emitted = emitted_commands(compile_all(QUERY).graphs)

    result = execute(emitted, work=lambda work: None)

    argv = result.commands[0].argv
    assert argv[:3] == ["ffmpeg", "-hide_banner", "-n"]
    assert argv[3 : 3 + len(_PROGRESS_ARGS)] == list(_PROGRESS_ARGS)
    assert argv[3 + len(_PROGRESS_ARGS)] == "-i"


def test_a_command_nobody_is_watching_is_asked_for_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda argv, **kw: subprocess.CompletedProcess(args=argv, returncode=0),
    )
    emitted = emitted_commands(compile_all(QUERY).graphs)

    result = execute(emitted)

    assert result.commands[0].argv[:4] == ["ffmpeg", "-hide_banner", "-n", "-i"]
