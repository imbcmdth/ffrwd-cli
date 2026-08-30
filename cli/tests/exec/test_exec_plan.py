"""End-to-end exec tests for running a process plan through real ffmpeg.

Marked ``@pytest.mark.exec`` and excluded from the default run, same as
``test_exec.py``. Run explicitly::

    python -m pytest -m exec tests/exec/test_exec_plan.py -q

Requires ``ffmpeg``/``ffprobe`` on PATH and the fixtures already generated
(``python scripts/gen_fixtures.py``). Tests skip cleanly if either is missing.

An ffmpeg stands in for the sidecar, since the real sidecar argv lands with
the sidecar itself: ``execute_plan`` takes the argv for a sidecar process from
a hook, and here that hook returns an ffmpeg reading its stdin and writing its
stdout.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from ffrwd.errors import ErrorCode
from ffrwd.execute import execute_plan
from ffrwd.ir import Graph, Node, Output, SinkUnit, StreamType
from ffrwd.processes import (
    PIPE,
    AudioFormat,
    FfmpegProcess,
    ProcessPlan,
    SidecarProcess,
    StreamEdge,
    VideoFormat,
    external_ids,
    partition,
)

pytestmark = pytest.mark.exec

_FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
_AV = _FIXTURES_DIR / "av.mp4"

# What gen_fixtures.py writes: 4 seconds at 15 frames a second.
_SRC_FRAMES = 60

_SUBPROCESS_TIMEOUT = 60.0
# Long enough for a four-second clip to cross three processes, short enough
# that a stage which cannot finish does not hold the suite up.
_STAGE_TIMEOUT = 60.0
_HUNG_TIMEOUT = 2.0
# Long enough for a member that outlived the stage to write more.
_SETTLE = 1.0


@pytest.fixture(autouse=True)
def _require_ffmpeg_tools() -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg/ffprobe not found on PATH")


def _require_fixture(path: Path) -> None:
    if not path.exists():
        pytest.skip(f"fixture missing: {path} (run scripts/gen_fixtures.py first)")


def _sql_path(path: Path) -> str:
    return path.resolve().as_posix()


def _out(ref: str, type_: StreamType = "video") -> Output:
    return Output(ref=ref, type=type_, name=None, metadata={})


def _live_pipes() -> list[str]:
    """The named pipes this process still serves.

    Windows lists them under ``\\\\.\\pipe``; a POSIX FIFO lives in a
    temporary directory that goes with the directory itself, so there is
    nothing to enumerate and the answer is always empty.
    """
    if sys.platform != "win32":
        return []
    prefix = f"ffrwd-{os.getpid()}-"
    return [name for name in os.listdir("\\\\.\\pipe\\") if name.startswith(prefix)]


def _negate(process: SidecarProcess) -> list[str]:
    return [
        "ffmpeg", "-hide_banner",
        "-f", "nut", "-i", "pipe:0",
        "-vf", "negate",
        "-c:v", "rawvideo", "-f", "nut", "pipe:1",
    ]  # fmt: skip


def _broken(process: SidecarProcess) -> list[str]:
    """An ffmpeg whose filtergraph names a filter that does not exist."""
    return [
        "ffmpeg", "-hide_banner",
        "-f", "nut", "-i", "pipe:0",
        "-vf", "ffrwd_no_such_filter",
        "-c:v", "rawvideo", "-f", "nut", "pipe:1",
    ]  # fmt: skip


def _frame_count(path: Path) -> int:
    args = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-count_frames", "-show_entries", "stream=nb_read_frames",
        "-of", "json", str(path),
    ]  # fmt: skip
    done = subprocess.run(
        args, capture_output=True, text=True, timeout=_SUBPROCESS_TIMEOUT
    )
    assert done.returncode == 0, done.stderr
    return int(json.loads(done.stdout)["streams"][0]["nb_read_frames"])


def _codec_types(path: Path) -> list[str]:
    args = [
        "ffprobe", "-v", "error",
        "-show_entries", "stream=codec_type",
        "-of", "json", str(path),
    ]  # fmt: skip
    done = subprocess.run(
        args, capture_output=True, text=True, timeout=_SUBPROCESS_TIMEOUT
    )
    assert done.returncode == 0, done.stderr
    return [stream["codec_type"] for stream in json.loads(done.stdout)["streams"]]


# ---------------------------------------------------------------- plans


def _chain(out_path: Path) -> ProcessPlan:
    """One input decoded, handed to an external node, encoded on the far side."""
    g = Graph(input_paths=[_sql_path(_AV)], sources={"a": 0})
    g.nodes["e0"] = Node(
        id="e0", filter="negate", args={}, inputs=["src:a:v:0"], outputs=["video"]
    )
    g.sinks = [
        SinkUnit(
            outputs=[_out("e0")],
            path=str(out_path),
            options={"video_codec": "libx264", "pix_fmt": "yuv420p"},
        )
    ]
    return partition(g, external=external_ids("e0"))


def _fan_in(out_path: Path) -> ProcessPlan:
    """Two processes decoding one file each, one muxing both of their streams."""
    video = Graph(input_paths=[_sql_path(_AV)], sources={"s": 0})
    video.sinks = [SinkUnit(outputs=[_out("src:s:v:0")], path=PIPE)]
    audio = Graph(input_paths=[_sql_path(_AV)], sources={"s": 0})
    audio.sinks = [SinkUnit(outputs=[_out("src:s:a:0", "audio")], path=PIPE)]
    mux = Graph(input_paths=[PIPE, PIPE], sources={"v": 0, "a": 1})
    mux.sinks = [
        SinkUnit(
            outputs=[_out("src:v:v:0"), _out("src:a:a:0", "audio")],
            path=str(out_path),
            options={
                "video_codec": "libx264",
                "pix_fmt": "yuv420p",
                "audio_codec": "aac",
            },
        )
    ]
    return ProcessPlan(
        processes=(
            FfmpegProcess(id="video", graph=video),
            FfmpegProcess(id="audio", graph=audio),
            FfmpegProcess(id="mux", graph=mux),
        ),
        edges=(
            StreamEdge(
                source="video", target="mux", ref="src:s:v:0", format=VideoFormat()
            ),
            StreamEdge(
                source="audio", target="mux", ref="src:s:a:0", format=AudioFormat()
            ),
        ),
    )


def _endless(out_path: Path) -> ProcessPlan:
    """A producer that never reaches an end, and a consumer waiting on it."""
    source = Graph(
        input_paths=["testsrc2=size=64x48:rate=25"],
        sources={"s": 0},
        input_options={"s": {"format": "lavfi"}},
    )
    source.sinks = [SinkUnit(outputs=[_out("src:s:v:0")], path=PIPE)]
    sink = Graph(input_paths=[PIPE], sources={"p": 0})
    sink.sinks = [
        SinkUnit(
            outputs=[_out("src:p:v:0")],
            path=str(out_path),
            options={"video_codec": "libx264", "pix_fmt": "yuv420p"},
        )
    ]
    return ProcessPlan(
        processes=(
            FfmpegProcess(id="source", graph=source),
            FfmpegProcess(id="sink", graph=sink),
        ),
        edges=(
            StreamEdge(
                source="source", target="sink", ref="src:s:v:0", format=VideoFormat()
            ),
        ),
    )


# ---------------------------------------------------------------- tests


def test_a_three_process_chain_writes_every_frame(tmp_path: Path) -> None:
    _require_fixture(_AV)
    out_path = tmp_path / "chain.mp4"
    result = execute_plan(
        _chain(out_path),
        sidecar_argv=_negate,
        timeout=_STAGE_TIMEOUT,
        overwrite=True,
    )

    assert result.exit_code == 0, result.failure
    assert result.failure is None
    assert len(result.stages) == 1
    assert len(result.stages[0].members) == 3
    assert all(member.exit_code == 0 for member in result.stages[0].members)
    assert out_path.exists()
    assert _frame_count(out_path) == _SRC_FRAMES
    assert _live_pipes() == []


def test_a_fan_in_stage_muxes_two_piped_producers(tmp_path: Path) -> None:
    _require_fixture(_AV)
    out_path = tmp_path / "fan-in.mkv"
    result = execute_plan(_fan_in(out_path), timeout=_STAGE_TIMEOUT, overwrite=True)

    assert result.exit_code == 0, result.failure
    assert len(result.stages) == 1
    assert {member.id for member in result.stages[0].members} == {
        "video",
        "audio",
        "mux",
    }
    assert out_path.exists()
    assert _codec_types(out_path) == ["video", "audio"]
    assert _frame_count(out_path) == _SRC_FRAMES
    assert _live_pipes() == []


def test_a_member_that_dies_takes_the_stage_with_it(tmp_path: Path) -> None:
    _require_fixture(_AV)
    out_path = tmp_path / "broken.mp4"
    result = execute_plan(
        _chain(out_path),
        sidecar_argv=_broken,
        timeout=_STAGE_TIMEOUT,
        overwrite=True,
    )

    assert result.exit_code != 0
    assert not result.timed_out
    assert result.failure is not None
    # The member that died is named, with the argv and the stderr that say
    # what it was. Its pipe neighbours may be named beside it: losing it shut
    # their pipes, and which of the three is seen first is milliseconds.
    named = {member.id: member for member in result.failures}
    assert "sidecar0" in named
    assert "ffrwd_no_such_filter" in named["sidecar0"].summary
    assert named["sidecar0"].stderr_tail != ""
    # Every member is accounted for, and the two that were still going were
    # told to stop rather than left behind.
    members = result.stages[0].members
    assert len(members) == 3
    assert all(member.exit_code != 0 or member.terminated for member in members)
    assert _live_pipes() == []


def test_a_stage_that_cannot_finish_hits_its_timeout(tmp_path: Path) -> None:
    out_path = tmp_path / "endless.mp4"
    result = execute_plan(_endless(out_path), timeout=_HUNG_TIMEOUT, overwrite=True)

    assert result.timed_out
    assert result.exit_code != 0
    assert result.failure is not None
    assert result.failure.terminated
    assert len(result.stages[0].members) == 2
    assert all(member.terminated for member in result.stages[0].members)
    assert _live_pipes() == []
    # Nothing survived the stage: an encoder left running would still be
    # writing, and the file it writes would still be growing.
    written = out_path.stat().st_size
    time.sleep(_SETTLE)
    assert out_path.stat().st_size == written


# --- a buffer that fills ------------------------------------------------------

# How long every pipe of the stage may stand still before the buffer one of
# them waits on is called full. Short, so the wedge is reported in seconds.
_STALL = 2.0


def _sleeper(process: SidecarProcess) -> list[str]:
    """A stand-in that never reads its pipe: whatever is written to it stays."""
    return [sys.executable, "-c", "import time; time.sleep(30)"]


def _live_merge(out_path: Path, duration: int) -> ProcessPlan:
    """A one-open source split two ways, both legs merged back together.

    A lavfi graph is one-open the way a camera is, so ONE process reads it and
    hands each leg a pipe -- and the buffers on those pipes are sized from the
    bound the compiler computed for them.
    """
    g = Graph(
        input_paths=[f"testsrc2=size=1920x1080:rate=25:duration={duration}"],
        sources={"a": 0},
        input_options={"a": {"format": "lavfi"}},
    )
    g.nodes["sp"] = Node(
        id="sp",
        filter="split",
        args={"n": 2},
        inputs=["src:a:v:0"],
        outputs=["video", "video"],
    )
    g.nodes["e0"] = Node(
        id="e0", filter="negate", args={}, inputs=["sp:0"], outputs=["video"]
    )
    g.nodes["n0"] = Node(
        id="n0", filter="hstack", args={}, inputs=["sp:1", "e0"], outputs=["video"]
    )
    g.sinks = [
        SinkUnit(
            outputs=[_out("n0")],
            path=str(out_path),
            options={"video_codec": "libx264", "pix_fmt": "yuv420p"},
        )
    ]
    return partition(g, external=external_ids("e0"))


def test_a_buffer_that_fills_ends_the_stage_naming_the_edge(tmp_path: Path) -> None:
    """A consumer that never reads wedges the whole stage, and the run says so.

    Not a timeout: the report names the edge whose buffer filled and the depth
    the compiler gave it, so what stopped is a fact rather than a guess.
    """
    plan = _live_merge(tmp_path / "wedged.mp4", duration=10)
    result = execute_plan(
        plan,
        sidecar_argv=_sleeper,
        timeout=_STAGE_TIMEOUT,
        overwrite=True,
        stall=_STALL,
    )

    assert result.overflow is not None, "the wedged stage was not reported"
    assert result.overflow.code is ErrorCode.BUFFER_OVERFLOW
    assert result.exit_code != 0
    assert not result.timed_out, "a full buffer is not a timeout"
    assert "from ffmpeg1 to " in result.overflow.message
    assert f"for {_STALL:.0f}s" in result.overflow.message
    assert result.overflow.hint is not None
    assert all(member.terminated for member in result.stages[0].members)
    assert _live_pipes() == []


def test_a_stage_that_keeps_moving_is_never_called_full(tmp_path: Path) -> None:
    """The same shape with a consumer that reads: the stall window never fires."""
    out_path = tmp_path / "merged.mp4"
    plan = _live_merge(out_path, duration=1)
    result = execute_plan(
        plan,
        sidecar_argv=_negate,
        timeout=_STAGE_TIMEOUT,
        overwrite=True,
        stall=_STALL,
    )

    assert result.overflow is None, str(result.overflow)
    assert result.exit_code == 0, "\n".join(
        f"{m.id} exited {m.exit_code}: {m.stderr_tail}"
        for stage in result.stages
        for m in stage.members
    )
    assert out_path.exists()
