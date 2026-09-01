"""Printing a process plan instead of running it.

Unit tier: plans built by hand or by :func:`~ffrwd.processes.partition`,
nothing spawned, nothing read off disk. Every expected pipeline string is
composed FROM :func:`~ffrwd.execute.plan_argv`'s own output rather than typed
by hand, so a test cannot drift from the renderer it is checking.
"""

from __future__ import annotations

import shlex
from collections.abc import Sequence

from ffrwd.execute import CHAIN, PIPELINE, plan_argv, render_plan
from ffrwd.ir import Graph, Node, Output, SinkUnit, StreamType
from ffrwd.processes import (
    FfmpegProcess,
    ProcessPlan,
    SidecarProcess,
    external_ids,
    partition,
)


def _out(ref: str, type_: StreamType = "video") -> Output:
    return Output(ref=ref, type=type_, name=None, metadata={})


def _stand_in(process: SidecarProcess, reads: Sequence[str] = ()) -> list[str]:
    """An ffmpeg standing in for a sidecar, reading the paths it was given."""
    argv = ["ffmpeg"]
    for path in reads or ("pipe:0",):
        argv += ["-f", "nut", "-i", path]
    return [*argv, "-vf", process.module, "-c:v", "rawvideo", "-f", "nut", "pipe:1"]


def _chain_plan() -> ProcessPlan:
    """decode -> one external node -> encode: three processes in a line."""
    g = Graph(input_paths=["a.mp4"], sources={"a": 0})
    g.nodes["e0"] = Node(
        id="e0", filter="negate", args={}, inputs=["src:a:v:0"], outputs=["video"]
    )
    g.sinks = [SinkUnit(outputs=[_out("e0")], path="out.mp4")]
    return partition(g, external=external_ids("e0"))


def _fan_in_plan() -> ProcessPlan:
    """One video and one audio stream, each through its own external node --
    both feed the muxing ffmpeg, which reads two streams: a genuine fan-in."""
    g = Graph(input_paths=["a.mp4"], sources={"a": 0})
    g.nodes["e0"] = Node(
        id="e0", filter="negate", args={}, inputs=["src:a:v:0"], outputs=["video"]
    )
    g.nodes["e1"] = Node(
        id="e1", filter="boost", args={}, inputs=["src:a:a:0"], outputs=["audio"]
    )
    g.sinks = [SinkUnit(outputs=[_out("e0"), _out("e1", "audio")], path="out.mkv")]
    return partition(g, external=external_ids("e0", "e1"))


def _single_process_plan() -> ProcessPlan:
    """No sidecars, no edges: the plain compile of one query, one command."""
    graph = Graph(input_paths=["a.mp4"], sources={"a": 0})
    graph.sinks = [SinkUnit(outputs=[_out("src:a:v:0")], path="out.mp4")]
    return ProcessPlan(processes=(FfmpegProcess(id="ffmpeg0", graph=graph),))


# ---------------------------------------------------------------- pipelines


def test_a_chain_renders_as_one_pipeline() -> None:
    plan = _chain_plan()
    argv = plan_argv(plan, sidecar_argv=_stand_in)
    expected = PIPELINE.join(
        shlex.join(argv[pid]) for pid in ("ffmpeg1", "sidecar0", "ffmpeg0")
    )
    assert render_plan(plan, sidecar_argv=_stand_in) == expected
    assert CHAIN not in expected  # one stage: no `&&` needed


def test_a_single_process_plan_renders_as_the_plain_command() -> None:
    plan = _single_process_plan()
    argv = plan_argv(plan)
    expected = shlex.join(argv["ffmpeg0"])
    rendered = render_plan(plan)
    assert rendered == expected
    assert PIPELINE not in rendered
    assert CHAIN not in rendered


# ---------------------------------------------------------------- fan-in


def test_a_fan_in_plan_renders_as_a_run_only_listing() -> None:
    plan = _fan_in_plan()
    rendered = render_plan(plan, sidecar_argv=_stand_in)
    lines = rendered.split("\n")

    # every process is present, one per line, ordinal-prefixed
    assert len(lines) == len(plan.processes) + 2  # header + members + trailing note
    for index, process in enumerate(plan.processes, start=1):
        role = "sidecar" if isinstance(process, SidecarProcess) else "ffmpeg"
        assert lines[index].startswith(f"{index}. {role}: ")

    # the courtesy line names the actual way to run this shape
    assert "ffrwd run" in lines[-1]
    assert "not a shell command" in lines[-1]

    # this is not, and does not claim to be, a pipeline
    assert PIPELINE not in rendered
    assert CHAIN not in rendered


def test_a_fan_in_listing_never_raises_for_missing_pipe_names() -> None:
    """`plan_argv` refuses a fan-in edge with no `pipe_path`; the courtesy
    listing must not inherit that refusal -- it has a placeholder instead."""
    rendered = render_plan(_fan_in_plan(), sidecar_argv=_stand_in)
    assert "ffmpeg" in rendered
