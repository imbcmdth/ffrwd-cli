"""Run a compiled sequence of ffmpeg commands.

:func:`execute` is the half of ``run`` that is not printing: it walks the
:class:`~ffrwd.emit.Emitted` list, builds each command's argv
(:func:`~ffrwd.emit.build_ffmpeg_commands`), inserts ``-hide_banner`` and
``-y``/``-n``, runs it as a subprocess with a per-command timeout, and stops
at the first nonzero exit -- whose code becomes the run's. No shell, on any
platform: the argv list goes straight to :func:`subprocess.run`. A timeout of
None runs with none at all, which is what an input of unknown duration -- a
device, a live URL -- gets, since nothing bounds how long it should run.

``ffrwd.loudnorm2`` is the one graph whose commands are not independent.
Its measuring pass prints the measurements as a JSON block on stderr, so that
pass is ALWAYS captured, parsed in process (:func:`ffrwd.loudnorm.parse`)
and substituted into the correction pass's argv
(:func:`ffrwd.loudnorm.substitute`) -- the ``eval "$(...)"`` the printed
command line shows is only for a pasted command.

Two stderr modes, because two callers want opposite things:

* ``capture_stderr=False`` (the default, and what the CLI passes) leaves
  ffmpeg's stderr inherited -- progress lines land on the user's terminal as
  they are written, and :attr:`CommandResult.stderr` is empty for every
  command but loudnorm2's measuring pass;
* ``capture_stderr=True`` pipes every command's stderr into its
  :class:`CommandResult`, for a library or server caller that has no terminal
  to share.

Nothing here prints or raises: the caller reads :class:`ExecutionResult` --
the argv actually run per command, the exit code, the captured stderr, and
which of the two non-ffmpeg failures (a timeout, an unparseable measuring
pass) ended the run -- and words its own messages.

Process plans
-------------
:func:`execute_plan` is the other half, and runs a
:class:`~ffrwd.processes.ProcessPlan` rather than a command list. A plan's
stages run in order, exactly as commands do; a stage's members run AT ONCE,
because they hand each other frames over pipes and ffmpeg opens its inputs
one at a time -- feeding a stage member by member deadlocks.

:func:`wires` decides how each stream edge travels. A process the plan hands
at most one stream reads it on its own stdin, and one the plan takes at most
one stream from writes it on its own stdout; a chain of those needs nothing
but :class:`subprocess.Popen`. Fan-in is what stdio cannot spell -- one stdin,
two producers -- and there the consumer reads named pipes instead
(:mod:`ffrwd.pipes`), with this process copying between each pipe and the
producer that feeds it. Every stream such a copy runs through is unbuffered,
and the copy hands on whatever has arrived rather than waiting to fill a
chunk: where one process feeds two paths that meet again, the process that
would round the chunk up is waiting for the frame the held-back bytes finish.

Members are judged by EXIT CODE only. A raw demuxer logs an error at the
pipe's EOF and exits 0 anyway, so stderr says nothing about whether a member
worked; it is captured to be reported, not to be read.

One failure is not an exit code: a stage whose pipes have all stopped moving
while every member still runs and one copy still waits to hand its bytes over.
That is a buffer the plan sized from a bound (:attr:`StreamEdge.bound`) that
the run outgrew, and it ends the stage with a typed
:attr:`PlanResult.overflow` naming the edge and the depth it was given -- not
with a timeout that would say only that something hung.

A plan shows the same way a command list does. A member ``players`` names
gets an ffplay of its own, reading the display output off that member's
stdout -- free, because a process writing a file hands its frames to nobody.
The forwarding is the same drain-tolerant copy, so a closed window ends the
window and not the run -- unless the windows are the ONLY thing the run
feeds (``show_only``), where closing the last of them ends the run cleanly
rather than leaving a camera and an encoder busy for nobody.

Printing a plan
----------------
:func:`render_plan` is what a caller prints instead of running a plan. A
plan :func:`wires` chains end to end -- every stream edge chained, no fan-in
or fan-out -- is exactly what a shell pipe can spell: each stage's members
joined by ``|``, and a stage after another (a file edge, as a command
sequence already prints) joined by ``&&``. A plan with any fan needs a named
pipe on at least one edge end, which no pipe operator carries, so that plan
prints as a numbered listing instead -- one line per process, run-only. The
piped form is POSIX-shell only, the same caveat ``docs/known_gaps.md``
already carries for the printed ``loudnorm2`` chain.
"""

from __future__ import annotations

import contextlib
import heapq
import math
import os
import shlex
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Literal

from . import loudnorm, pipes
from .emit import Emitted, build_ffmpeg_commands, build_process_args
from .errors import ErrorCode, FfrwdError
from .pipes import NamedPipe
from .processes import (
    FfmpegProcess,
    Process,
    ProcessPlan,
    RowsEdge,
    SidecarProcess,
    Stage,
    StreamEdge,
)

__all__ = [
    "CHAIN",
    "DEFAULT_STALL",
    "DEFAULT_TIMEOUT",
    "PIPELINE",
    "STDIN",
    "STDOUT",
    "CommandResult",
    "ExecutionResult",
    "Flow",
    "PipeEdge",
    "PipeNamer",
    "PlanResult",
    "ProcessResult",
    "Side",
    "SidecarArgv",
    "StageResult",
    "Wire",
    "execute",
    "execute_plan",
    "overflow_error",
    "overflowed",
    "plan_argv",
    "render_plan",
    "wires",
]

# Per command, not per run, and the floor of the input-scaled default a
# compile carries (`Compiled.default_timeout`).
DEFAULT_TIMEOUT = 600

# What a timeout and an unparseable measuring pass report, neither being an
# ffmpeg exit code.
_FAILED = 1

# How a process spells the stream edge its own stdio carries.
STDIN = "pipe:0"
STDOUT = "pipe:1"

# How `render_plan` joins a stage's members into a shell pipeline, and one
# stage after another -- the same separator a printed command sequence uses.
PIPELINE = " | "
CHAIN = " && "

# How often a running stage is re-checked.
_POLL = 0.02
# How long every pipe of a stage may stand still, with every member running and
# one of them waiting to write, before the buffer it waits on is called full.
DEFAULT_STALL = 30.0
# How long a member that was told to stop is given before it is killed.
_GRACE = 5.0
# How long a helper thread is waited for once its process has gone.
_JOIN = 5.0
# Bytes moved per copy between a named pipe and a process's stdio.
_CHUNK = 1 << 16
# What `ProcessResult.summary` and `.stderr_tail` keep.
_SUMMARY_CHARS = 160
_TAIL_LINES = 20


@dataclass(frozen=True)
class CommandResult:
    """One ffmpeg subprocess: the argv actually run, and how it ended."""

    argv: list[str]
    exit_code: int
    # The command's stderr when it was captured, "" when it went to the
    # caller's own stderr. `captured` tells the two empties apart.
    stderr: str = ""
    captured: bool = False


@dataclass(frozen=True)
class ExecutionResult:
    """Every command run, in order, plus the run's own outcome."""

    commands: list[CommandResult] = field(default_factory=list)
    # The first nonzero ffmpeg exit, or 1 for a timeout / measuring-pass
    # failure, or 0.
    exit_code: int = 0
    # True when a command hit the timeout; its argv is the last `commands`
    # entry, and no later command started.
    timed_out: bool = False
    # The `loudnorm.parse` failure text when a measuring pass printed no
    # loudnorm JSON block; None otherwise.
    measure_error: str | None = None


def execute(
    emitted: Sequence[Emitted],
    *,
    timeout: float | None = DEFAULT_TIMEOUT,
    overwrite: bool = False,
    capture_stderr: bool = False,
    echo: Callable[[list[str]], None] | None = None,
    players: Sequence[list[str] | None] = (),
    show_only: bool = False,
) -> ExecutionResult:
    """Run every command of `emitted`, in order, stopping at the first failure.

    A two-pass sink compiles to two commands, a loudnorm2 graph to two, a
    fan-out COPY to one per row, every other query to one. `timeout` is per
    command; None runs without one, for a live input no duration bounds.
    `overwrite` picks ffmpeg's ``-y`` over ``-n``. `echo` is called
    with each argv just before its subprocess starts, so a caller that prints
    the command line interleaves with ffmpeg's own output the way the CLI
    does.

    `players` is parallel to `emitted`: entry `i` is the ffplay reading that
    command's display output, or None for one with no window. Only the LAST
    command of a multi-pass sink shows -- an earlier pass measures rather
    than writes -- and the others send their stdout nowhere.

    `show_only` says the windows are all that consumes the run: a command
    whose window the viewer closes ends CLEANLY, exit 0, instead of feeding a
    display nobody is watching. Without it a closed window never ends a
    command, since its files are still being written.
    """
    results: list[CommandResult] = []
    measured: dict[str, str] = {}

    for position, e in enumerate(emitted):
        commands = build_ffmpeg_commands(e)
        measures = bool(e.measure_filter_complex)
        player = players[position] if position < len(players) else None
        for index, command in enumerate(commands):
            # The measuring pass is captured whatever the caller asked for:
            # parsing its stderr is the only reason it runs.
            measuring = measures and index == 0
            capture = capture_stderr or measuring
            last = index == len(commands) - 1

            argv = [loudnorm.substitute(word, measured) for word in command]
            argv.insert(1, "-y" if overwrite else "-n")
            argv.insert(1, "-hide_banner")

            if echo is not None:
                echo(argv)

            try:
                code, captured = _run_ffmpeg(
                    argv,
                    timeout,
                    capture=capture,
                    player=player if last else None,
                    mute=player is not None and not last,
                    show_only=show_only,
                )
            except subprocess.TimeoutExpired as err:
                # Whatever the killed child had written by then, if captured.
                partial = err.stderr if isinstance(err.stderr, str) else ""
                results.append(CommandResult(argv, _FAILED, partial, capture))
                return ExecutionResult(results, _FAILED, timed_out=True)

            results.append(CommandResult(argv, code, captured, capture))
            if code != 0:
                return ExecutionResult(results, code)

            if measuring:
                try:
                    measured = loudnorm.parse(captured)
                except ValueError as err:
                    return ExecutionResult(results, _FAILED, measure_error=str(err))

    return ExecutionResult(results)


def _run_ffmpeg(
    argv: list[str],
    timeout: float | None,
    *,
    capture: bool,
    player: list[str] | None = None,
    mute: bool = False,
    show_only: bool = False,
) -> tuple[int, str]:
    """Run one ffmpeg command; ``(exit code, its stderr)``.

    Uncaptured stderr writes straight through to the caller's terminal,
    progress lines included, and comes back as "".

    `player` is the ffplay reading this command's display output; it is
    spawned first, takes ffmpeg's stdout as its stdin, and is torn down with
    the run. Closing its window does NOT end the run: ffmpeg's writes to the
    dead pipe are what would kill it, so the display output is written to a
    pipe ffrwd holds open and forwards, and a forward that fails is dropped.
    Under `show_only` it DOES end the command, cleanly -- the window was the
    only thing the run fed.

    `mute` sends stdout nowhere -- a shown command's earlier pass writes a
    display output no window is reading, and it must not reach the terminal.
    """
    if player is None:
        stdout = subprocess.DEVNULL if mute else None
        if not capture:
            return subprocess.run(argv, timeout=timeout, stdout=stdout).returncode, ""
        done = subprocess.run(
            argv, timeout=timeout, stdout=stdout, stderr=subprocess.PIPE, text=True
        )
        return done.returncode, done.stderr
    return _run_with_player(
        argv, timeout, capture=capture, player=player, show_only=show_only
    )


def _run_with_player(
    argv: list[str],
    timeout: float | None,
    *,
    capture: bool,
    player: list[str],
    show_only: bool = False,
) -> tuple[int, str]:
    """One ffmpeg command whose stdout an ffplay window is reading."""
    stderr = subprocess.PIPE if capture else None
    watching = subprocess.Popen(player, stdin=subprocess.PIPE, bufsize=0)
    ffmpeg = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=stderr, bufsize=0)
    # Not communicate(): it would close the stdout this thread is reading.
    forward = threading.Thread(
        target=_forward, args=(ffmpeg.stdout, watching.stdin), daemon=True
    )
    forward.start()
    collected: list[bytes] = []
    draining: threading.Thread | None = None
    if ffmpeg.stderr is not None:
        draining = threading.Thread(
            target=lambda: collected.append(_read_all(ffmpeg.stderr)), daemon=True
        )
        draining.start()
    try:
        code = _await_ffmpeg(ffmpeg, timeout, watching if show_only else None)
    except subprocess.TimeoutExpired:
        _end_tree(ffmpeg)
        _stop_player(watching)
        raise
    finally:
        forward.join(_JOIN)
        if draining is not None:
            draining.join(_JOIN)
        _stop_player(watching)
    captured = b"".join(collected)
    return code, captured.decode(errors="replace")


def _await_ffmpeg(
    ffmpeg: subprocess.Popen[bytes],
    timeout: float | None,
    watching: subprocess.Popen[bytes] | None,
) -> int:
    """Wait for `ffmpeg`; its exit code, or 0 when `watching` closed first.

    `watching` is passed only for a run whose window is its only consumer:
    the viewer closing it ends the command, and ending it that way IS the
    run succeeding. Passed None, this is a plain wait. `timeout` of None
    waits indefinitely, and otherwise raises
    :class:`subprocess.TimeoutExpired` the way :meth:`Popen.wait` does.
    """
    if watching is None:
        ffmpeg.wait(timeout=timeout)
        return ffmpeg.returncode
    started = time.monotonic()
    while True:
        code = ffmpeg.poll()
        if code is not None:
            return code
        if watching.poll() is not None:
            _end_tree(ffmpeg)
            with contextlib.suppress(subprocess.TimeoutExpired):
                ffmpeg.wait(_GRACE)
            return 0
        if timeout is not None and time.monotonic() - started >= timeout:
            raise subprocess.TimeoutExpired(ffmpeg.args, timeout)
        time.sleep(_POLL)


def _read_all(stream: IO[bytes] | None) -> bytes:
    """Everything left on `stream`, or b"" if it fails part way."""
    if stream is None:
        return b""
    try:
        return stream.read()
    except OSError:
        return b""


def _forward(source: IO[bytes] | None, target: IO[bytes] | None) -> None:
    """Copy the display output to the player, tolerating the window closing.

    A closed window breaks the pipe on the next write. That ends the WINDOW,
    not the run: the exception stops this thread, ffmpeg keeps writing into a
    pipe that is still drained here, and the file output finishes.
    """
    if source is None:
        return
    try:
        while chunk := source.read(_CHUNK):
            if target is None:
                continue
            try:
                _write_all(target, chunk)
            except OSError:
                target = None  # window gone; keep draining so ffmpeg runs on
    except OSError:
        pass
    finally:
        if target is not None:
            with contextlib.suppress(OSError):
                target.close()


def _stop_player(player: subprocess.Popen[bytes]) -> None:
    """End one display window, whether or not the viewer already closed it."""
    if player.poll() is None:
        _end_tree(player)
        with contextlib.suppress(subprocess.TimeoutExpired):
            player.wait(_GRACE)
    if player.stdin is not None:
        with contextlib.suppress(OSError):
            player.stdin.close()


# ---------------------------------------------------------------- process plans

# Which end of a pipe edge a named pipe is serving: the consumer's `read`,
# or the producer's `write`.
Side = Literal["read", "write"]

# An edge one process hands another over a pipe: frames, or the rows a module
# read off them.
PipeEdge = StreamEdge | RowsEdge

# Names the named pipe one end of an edge needs. Called only for an end stdio
# cannot carry.
PipeNamer = Callable[[PipeEdge, Side], str]

# Renders one sidecar process as the argv that runs it. The real one lands
# with the sidecar itself; until then a caller supplies it.
SidecarArgv = Callable[[SidecarProcess], list[str]]


@dataclass(frozen=True)
class Wire:
    """One pipe edge and the transport each of its ends takes.

    `read_stdio` is True when the consuming process reads this edge on its own
    stdin, `write_stdio` when the producing one writes it on its own stdout.
    An end that is neither takes a named pipe.
    """

    edge: PipeEdge
    read_stdio: bool
    write_stdio: bool

    @property
    def chained(self) -> bool:
        """True when stdio carries both ends and one Popen feeds the next."""
        return self.read_stdio and self.write_stdio


@dataclass(frozen=True)
class ProcessResult:
    """One member of a stage: the argv actually run, and how it ended."""

    id: str
    argv: list[str]
    exit_code: int
    stderr: str = ""
    # True when this member was still running and was told to stop, so its
    # exit code says how it was ended rather than how it ended on its own.
    terminated: bool = False

    @property
    def summary(self) -> str:
        """This member's argv, shortened enough to name it in a message."""
        text = " ".join(self.argv)
        if len(text) <= _SUMMARY_CHARS:
            return text
        return text[: _SUMMARY_CHARS - 3] + "..."

    @property
    def stderr_tail(self) -> str:
        """The last lines of this member's stderr, or "" if it wrote none."""
        lines = self.stderr.splitlines()
        return "\n".join(lines[-_TAIL_LINES:])


@dataclass
class Flow:
    """One pipe edge while the stage runs: what has crossed it, and when.

    `at` is when the last bytes moved, and `writing` is True while the copy is
    waiting for the consuming end to take what it was handed -- which is what
    a full buffer looks like from here.
    """

    edge: PipeEdge
    at: float
    moved: int = 0
    writing: bool = False

    @property
    def bound(self) -> int:
        """The frames the compiler said this edge would have to hold."""
        return self.edge.bound if isinstance(self.edge, StreamEdge) else 0

    @property
    def held(self) -> int:
        """The frames its buffer was actually sized for; 0 where none was."""
        buffer = self.edge.buffer if isinstance(self.edge, StreamEdge) else None
        return 0 if buffer is None else buffer.frames

    @property
    def road(self) -> str:
        buffer = self.edge.buffer if isinstance(self.edge, StreamEdge) else None
        return "pipe" if buffer is None else buffer.road


def overflow_error(flow: Flow, stall: float) -> FfrwdError:
    """The typed failure for a stage wedged on one edge's full buffer.

    Names the edge, the depth the compiler sized it for, and what stopped:
    never a bare timeout, and never a dropped frame nobody was told about.
    """
    carried = flow.edge.ref if isinstance(flow.edge, StreamEdge) else flow.edge.alias
    sized = (
        f"sized for the {flow.held} frames the compiler bounded it at"
        if flow.held
        else "left at the transport's own size, the compiler having found no "
        "depth it had to hold"
    )
    return FfrwdError(
        ErrorCode.BUFFER_OVERFLOW,
        f"the {flow.road} buffer carrying '{carried}' from {flow.edge.source} to "
        f"{flow.edge.target} overflowed: it was {sized}, and with every process "
        f"still running nothing has crossed any pipe of this stage for "
        f"{stall:.0f}s",
        hint="the paths out of the one process reading the input drifted "
        "further apart than the compiler counted them: record the input to a "
        "file and run this query over the file, or take the slower path's "
        "work out of the pipeline",
    )


@dataclass(frozen=True)
class StageResult:
    """One stage: every member it ran, and the outcome of running them."""

    index: int
    members: list[ProcessResult] = field(default_factory=list)
    # The failing member's exit code, or 1 for a timeout, or 0.
    exit_code: int = 0
    timed_out: bool = False
    # The member that ended the stage: the first SEEN to exit nonzero, or the
    # one still running when the timeout struck. None when every member exited
    # 0. Which one is seen first is a matter of milliseconds, so a report that
    # names a single member should say `failures` instead.
    failure: ProcessResult | None = None
    # Every member that failed on its own rather than being told to stop --
    # the whole truth, since one member dying takes its pipe neighbours with
    # it and nothing distinguishes the cause from the consequence. On a
    # timeout, the member that was still running.
    failures: list[ProcessResult] = field(default_factory=list)
    # The full buffer that wedged this stage, when that is what ended it. Set
    # instead of `timed_out`, since the two answer the same question and this
    # one names the edge.
    overflow: FfrwdError | None = None


@dataclass(frozen=True)
class PlanResult:
    """Every stage run, in order, plus the run's own outcome."""

    stages: list[StageResult] = field(default_factory=list)
    exit_code: int = 0
    timed_out: bool = False
    failure: ProcessResult | None = None
    failures: list[ProcessResult] = field(default_factory=list)
    overflow: FfrwdError | None = None


def wires(plan: ProcessPlan) -> tuple[Wire, ...]:
    """Every pipe edge of `plan`, with the transport each end takes.

    One rule, read per process: a process the plan hands at most one thing
    reads it on its own stdin, and one the plan takes at most one thing from
    writes it on its own stdout. Anything else is a fan, which stdio has no
    second handle for, and every edge on that side takes a named pipe. A rows
    edge counts on both sides: it occupies stdio exactly as frames do.
    """
    edges = _pipe_edges(plan)
    incoming: dict[str, int] = {}
    outgoing: dict[str, int] = {}
    for edge in edges:
        incoming[edge.target] = incoming.get(edge.target, 0) + 1
        outgoing[edge.source] = outgoing.get(edge.source, 0) + 1
    return tuple(
        Wire(
            edge=edge,
            read_stdio=incoming[edge.target] <= 1,
            write_stdio=outgoing[edge.source] <= 1,
        )
        for edge in edges
    )


def _pipe_edges(plan: ProcessPlan) -> tuple[PipeEdge, ...]:
    """`plan`'s edges that run over a pipe, rows ahead of frames per process.

    The order is what pairs an edge with the ``pipe:`` slot it fills: a
    reading process's own inputs come first in its ``-i`` list, and a rows
    track is one of those, where a frame edge is appended after them.
    """
    return (*plan.rows_edges, *plan.stream_edges)


def plan_argv(
    plan: ProcessPlan,
    *,
    sidecar_argv: SidecarArgv | None = None,
    pipe_path: PipeNamer | None = None,
) -> dict[str, list[str]]:
    """The argv that runs each process of `plan`, keyed by process id.

    An ffmpeg process renders through :func:`~ffrwd.emit.build_process_args`
    with its edges spelled as :func:`wires` assigned them, a rows track ahead
    of the frame edges because the reading graph carries it as an ``-i`` of
    its own. A sidecar process renders through `sidecar_argv`; a plan carrying
    one without it is refused, since nothing else here knows how to spawn a
    wasm module.

    `pipe_path` names a named pipe, and is called once per end that needs one.
    """
    read: dict[PipeEdge, str] = {}
    write: dict[PipeEdge, str] = {}
    for wire in wires(plan):
        read[wire.edge] = (
            STDIN if wire.read_stdio else _named(pipe_path, wire.edge, "read")
        )
        write[wire.edge] = (
            STDOUT if wire.write_stdio else _named(pipe_path, wire.edge, "write")
        )

    argv: dict[str, list[str]] = {}
    for process in plan.processes:
        incoming = _once_per_ref(
            [e for e in plan.stream_edges if e.target == process.id]
        )
        outgoing = [e for e in plan.stream_edges if e.source == process.id]
        rows_out = [e for e in plan.rows_edges if e.source == process.id]
        if isinstance(process, SidecarProcess):
            argv[process.id] = _sidecar_args(
                process, sidecar_argv, len(incoming), len(outgoing) + len(rows_out)
            )
            continue
        rows_in = _rows_inputs(process, plan)
        argv[process.id] = build_process_args(
            process.graph,
            pipe_inputs=[(read[edge], edge.container) for edge in rows_in]
            + [(read[edge], edge.format.container) for edge in incoming],
            pipe_outputs=[(write[edge], edge.format) for edge in outgoing],
            pipe_buffers=[edge.buffer for edge in outgoing],
        )
    return argv


def _rows_inputs(process: FfmpegProcess, plan: ProcessPlan) -> list[RowsEdge]:
    """The rows tracks `process` reads, in the ``-i`` order its graph gives them.

    A minted track is one of the reading graph's own inputs, so its alias's
    place in ``sources`` is the ``pipe:`` slot it fills -- and every such slot
    comes before the frame edges, which are appended after the graph's own
    inputs.
    """
    found = [e for e in plan.rows_edges if e.target == process.id]
    return sorted(found, key=lambda e: process.graph.sources.get(e.alias, 0))


def render_plan(
    plan: ProcessPlan,
    *,
    sidecar_argv: SidecarArgv | None = None,
    pipe_path: PipeNamer | None = None,
) -> str:
    """The text a compile of `plan` prints in place of running it.

    A plan with no fan -- every stream edge :func:`wires` calls chained --
    prints as a shell pipeline: a stage's members joined by `` | ``, and one
    stage after another (a file edge) joined by `` && ``, matching how a
    plain command sequence already prints. A single process with no stream
    edges at all is the trivial pipeline of one: its own argv, unchanged.

    Any fan-in or fan-out puts a named pipe on at least one edge end, and no
    pipe operator can spell that; such a plan prints as a numbered, run-only
    listing instead (:func:`_render_listing`) -- ``ffrwd run`` is what
    actually executes it. `pipe_path` names those pipes as :func:`plan_argv`
    would for a real run; nothing has made one yet at print time, so a plan
    that needs one and is given no `pipe_path` gets a placeholder that says
    so, rather than :func:`plan_argv`'s refusal.
    """
    argv = plan_argv(plan, sidecar_argv=sidecar_argv, pipe_path=pipe_path or _placeholder_pipe)
    if _is_pipeline(plan):
        return _render_pipeline(plan, argv)
    return _render_listing(plan, argv)


def _placeholder_pipe(edge: PipeEdge, side: Side) -> str:
    """The named pipe path `render_plan` shows when it was given no real one.

    Unique per edge, not per process pair: two streams one process hands
    another are two pipes, and an equal spelling would fold their ``-i``s.
    """
    carried = edge.ref if isinstance(edge, StreamEdge) else edge.alias
    return f"<named pipe {edge.source}-{edge.target} {carried} {side}>"


def _is_pipeline(plan: ProcessPlan) -> bool:
    """True when every pipe edge chains through stdio -- no fan anywhere."""
    return all(wire.chained for wire in wires(plan))


def _render_pipeline(plan: ProcessPlan, argv: Mapping[str, list[str]]) -> str:
    edges = _pipe_edges(plan)
    stages = [
        PIPELINE.join(shlex.join(argv[pid]) for pid in _chain_order(stage.processes, edges))
        for stage in plan.stages
    ]
    return CHAIN.join(stages)


def _chain_order(members: Sequence[str], edges: Sequence[PipeEdge]) -> list[str]:
    """`members`, already known to form one stdio chain, producer to consumer."""
    inside = set(members)
    next_of = {
        e.source: e.target for e in edges if e.source in inside and e.target in inside
    }
    fed = set(next_of.values())
    order = [next(m for m in members if m not in fed)]
    while order[-1] in next_of:
        order.append(next_of[order[-1]])
    return order


# `render_plan`'s honest fallback: no shell can paste this, and it says so.
_COURTESY_NOTE = "# this listing is not a shell command -- run the plan with `ffrwd run`"


def _render_listing(plan: ProcessPlan, argv: Mapping[str, list[str]]) -> str:
    lines = [_listing_header(plan)]
    lines += [
        f"{index}. {_role(process)}: {shlex.join(argv[process.id])}"
        for index, process in enumerate(plan.processes, start=1)
    ]
    lines.append(_COURTESY_NOTE)
    return "\n".join(lines)


def _listing_header(plan: ProcessPlan) -> str:
    """One line naming which processes fan in or out over named pipes."""
    incoming: dict[str, list[str]] = {}
    outgoing: dict[str, list[str]] = {}
    for edge in _pipe_edges(plan):
        incoming.setdefault(edge.target, []).append(edge.source)
        outgoing.setdefault(edge.source, []).append(edge.target)
    fans = [
        f"{target} reads {', '.join(sources)}"
        for target, sources in incoming.items()
        if len(sources) > 1
    ]
    fans += [
        f"{source} feeds {', '.join(targets)}"
        for source, targets in outgoing.items()
        if len(targets) > 1
    ]
    return f"# named pipes: {'; '.join(fans)}"


def _role(process: Process) -> str:
    return "ffmpeg" if isinstance(process, FfmpegProcess) else "sidecar"


def execute_plan(
    plan: ProcessPlan,
    *,
    sidecar_argv: SidecarArgv | None = None,
    timeout: float | None = DEFAULT_TIMEOUT,
    overwrite: bool = False,
    echo: Callable[[str, list[str]], None] | None = None,
    players: Mapping[str, list[str]] | None = None,
    show_only: bool = False,
    stall: float | None = DEFAULT_STALL,
) -> PlanResult:
    """Run `plan`, stage by stage, stopping at the first stage that fails.

    A stage's members are spawned together and watched together: the first to
    exit nonzero stops the rest, and everything those members started
    (:func:`subprocess.Popen.terminate` reaches one process, and the ffmpeg on
    PATH is often a shim around the real one). `timeout` does the same, and is
    per STAGE the way :func:`execute`'s is per command; None runs the stage
    without one, for a live input no duration bounds. Every member's stderr
    is captured -- several ffmpegs sharing one terminal interleave into
    nothing readable -- and reported with it.

    A failed stage reports every member that failed on its own, not one:
    losing a member closes the pipes around it, so its neighbours fail too and
    nothing tells the cause from the consequence.

    `overwrite` picks ffmpeg's ``-y`` over ``-n`` for the files the plan
    writes; a member writing into a named pipe always gets ``-y``, since that
    pipe is one this process just made. `echo` is called with each member's id
    and argv just before it is spawned.

    `players` is keyed by process id: the ffplay reading that member's display
    output off its stdout. A member absent from it sends its stdout where the
    plan's own wiring says. Closing a window does NOT end the run, exactly as
    it does not for a command list -- unless `show_only` says the windows are
    all the run feeds, and closing the LAST of a stage's ends that stage
    cleanly.

    `stall` is how long every pipe of a stage may stand still, with every
    member running and one of them waiting to write, before the buffer it is
    waiting on is reported full: :attr:`PlanResult.overflow` names the edge and
    the depth it was sized for, rather than leaving a wedged run to the
    timeout. None turns that off.

    Named pipes and any temporary directory holding them are removed before
    this returns, whether the plan finished or failed.
    """
    stack = contextlib.ExitStack()
    try:
        served: dict[tuple[PipeEdge, Side], NamedPipe] = {}
        home: list[Path] = []

        def pipe_path(edge: PipeEdge, side: Side) -> str:
            if not home:
                home.append(
                    Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="ffrwd-")))
                )
            # A named pipe on the consumer's side is one this process writes.
            pipe = pipes.create(
                home[0],
                str(len(served)),
                writing=side == "read",
                buffer=_pipe_buffer(edge),
            )
            stack.callback(pipe.close)
            served[(edge, side)] = pipe
            return pipe.path

        argv = plan_argv(plan, sidecar_argv=sidecar_argv, pipe_path=pipe_path)
        assigned = wires(plan)

        stages: list[StageResult] = []
        for stage in plan.stages:
            result = _run_stage(
                plan,
                stage,
                argv,
                served,
                assigned,
                timeout,
                overwrite,
                echo,
                players or {},
                show_only,
                stall,
            )
            stages.append(result)
            if result.exit_code != 0:
                return PlanResult(
                    stages,
                    result.exit_code,
                    result.timed_out,
                    result.failure,
                    result.failures,
                    result.overflow,
                )
        return PlanResult(stages)
    finally:
        stack.close()


# -- argv helpers


def _pipe_buffer(edge: PipeEdge) -> int:
    """How big the named pipes for `edge` are made.

    Only a stream edge whose plan put its depth on the PIPE road asks for more
    than the default: the fifo road holds its depth in the producing ffmpeg,
    and a rows edge carries no frames to count.
    """
    if not isinstance(edge, StreamEdge) or edge.buffer is None:
        return pipes.DEFAULT_BUFFER
    if edge.buffer.road != "pipe":
        return pipes.DEFAULT_BUFFER
    return max(edge.buffer.size, pipes.DEFAULT_BUFFER)


def _named(namer: PipeNamer | None, edge: PipeEdge, side: Side) -> str:
    if namer is None:
        raise FfrwdError(
            ErrorCode.INTERNAL,
            f"the stream {edge.source} hands {edge.target} needs a named pipe, "
            "and no way to make one was given",
            hint="pass pipe_path, which names the named pipe an edge end needs",
        )
    return namer(edge, side)


def _once_per_ref(edges: Sequence[StreamEdge]) -> list[StreamEdge]:
    """`edges` with one entry per ref, the input list a partitioned graph has."""
    kept: list[StreamEdge] = []
    seen: set[str] = set()
    for edge in edges:
        if edge.ref in seen:
            continue
        seen.add(edge.ref)
        kept.append(edge)
    return kept


def _sidecar_args(
    process: SidecarProcess,
    hook: SidecarArgv | None,
    reads: int,
    writes: int,
) -> list[str]:
    if hook is None:
        raise FfrwdError(
            ErrorCode.INTERNAL,
            f"process {process.id!r} hosts the module {process.module!r}, and "
            "nothing was given to spawn it",
            hint="pass sidecar_argv, which renders one sidecar process as argv",
        )
    if reads > 1 or writes > 1:
        raise FfrwdError(
            ErrorCode.INTERNAL,
            f"process {process.id!r} reads {reads} streams and writes {writes}, "
            "but only its own stdin and stdout are wired",
            hint="a sidecar reading or writing more than one stream needs argv "
            "that can spell a named pipe path",
        )
    return list(hook(process))


def _spawn_argv(
    process: FfmpegProcess | SidecarProcess,
    argv: Sequence[str],
    *,
    overwrite: bool,
) -> list[str]:
    """`argv` as it is actually spawned. A sidecar's is the hook's, verbatim."""
    if not isinstance(process, FfmpegProcess):
        return list(argv)
    command = list(argv)
    command.insert(1, "-y" if overwrite else "-n")
    command.insert(1, "-hide_banner")
    return command


# -- running a stage


@dataclass
class _Member:
    """One spawned member of a stage, and what has been collected from it."""

    id: str
    argv: list[str]
    proc: subprocess.Popen[bytes]
    stderr: list[bytes] = field(default_factory=list)
    terminated: bool = False


class _End:
    """One end of a wire: the stream to copy through, and its release."""

    def open(self, deadline: float) -> IO[bytes]:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class _StdioEnd(_End):
    """A member's own stdin or stdout, already open."""

    def __init__(self, stream: IO[bytes]) -> None:
        self._stream = stream

    def open(self, deadline: float) -> IO[bytes]:
        return self._stream

    def close(self) -> None:
        try:
            self._stream.close()
        except (OSError, ValueError):
            pass


class _PipeEnd(_End):
    """A named pipe, open once the member on the other side connects."""

    def __init__(self, pipe: NamedPipe) -> None:
        self._pipe = pipe

    def open(self, deadline: float) -> IO[bytes]:
        return self._pipe.wait(deadline)

    def close(self) -> None:
        self._pipe.close()


def _run_stage(
    plan: ProcessPlan,
    stage: Stage,
    argv: dict[str, list[str]],
    served: dict[tuple[PipeEdge, Side], NamedPipe],
    assigned: Sequence[Wire],
    timeout: float | None,
    overwrite: bool,
    echo: Callable[[str, list[str]], None] | None,
    players: Mapping[str, list[str]],
    show_only: bool = False,
    stall: float | None = DEFAULT_STALL,
) -> StageResult:
    """Spawn every member of `stage` at once, watch them, and report."""
    ids = list(stage.processes)
    inside = set(ids)
    stage_wires = [
        wire
        for wire in assigned
        if wire.edge.source in inside and wire.edge.target in inside
    ]
    deadline = math.inf if timeout is None else time.monotonic() + timeout
    members: dict[str, _Member] = {}
    watching: dict[str, subprocess.Popen[bytes]] = {}
    helpers: list[threading.Thread] = []
    ends: list[_End] = []
    flows: list[Flow] = []
    failed: str | None = None
    timed_out = False
    full: Flow | None = None
    try:
        for pid in _spawn_order(ids, stage_wires):
            process = plan.process(pid)
            reads = [w for w in stage_wires if w.edge.target == pid]
            writes = [w for w in stage_wires if w.edge.source == pid]
            chained = next((w for w in reads if w.chained), None)
            player = players.get(pid)
            stdin: int | IO[bytes] = subprocess.DEVNULL
            if chained is not None:
                stdin = _stream(members[chained.edge.source].proc.stdout)
            elif any(w.read_stdio for w in reads):
                stdin = subprocess.PIPE
            stdout: int | None = (
                subprocess.PIPE
                if any(w.write_stdio for w in writes) or player is not None
                else subprocess.DEVNULL
            )
            if _writes_rows_to_stdout(process):
                # A packet sink's rows are its product, and nothing else in
                # the plan reads them: they reach the caller's own stdout.
                stdout = None
            # A named pipe this process just made is never a file to protect.
            command = _spawn_argv(
                process,
                argv[pid],
                overwrite=overwrite or any(not w.write_stdio for w in writes),
            )
            if echo is not None:
                echo(pid, command)
            if player is not None:
                watching[pid] = subprocess.Popen(
                    player, stdin=subprocess.PIPE, bufsize=0
                )
            members[pid] = _Member(
                id=pid, argv=command, proc=_spawn(command, stdin, stdout)
            )
            if chained is not None and not isinstance(stdin, int):
                stdin.close()  # the spawned member owns it now

        for pid, window in watching.items():
            helpers.append(_start(_forward, members[pid].proc.stdout, window.stdin))

        for wire in stage_wires:
            if wire.chained:
                continue
            source: _End = (
                _StdioEnd(_stream(members[wire.edge.source].proc.stdout))
                if wire.write_stdio
                else _PipeEnd(served[(wire.edge, "write")])
            )
            dest: _End = (
                _StdioEnd(_stream(members[wire.edge.target].proc.stdin))
                if wire.read_stdio
                else _PipeEnd(served[(wire.edge, "read")])
            )
            ends += [source, dest]
            flow = Flow(edge=wire.edge, at=time.monotonic())
            flows.append(flow)
            helpers.append(_start(_pump, source, dest, deadline, flow))

        for member in members.values():
            helpers.append(_start(_drain, _stream(member.proc.stderr), member.stderr))

        failed, timed_out, full = _watch(
            members.values(),
            deadline,
            list(watching.values()) if show_only and watching else None,
            flows,
            stall,
        )
    finally:
        # Whatever ended the stage, the rest of it goes too.
        _stop(members.values())
        for window in watching.values():
            _stop_player(window)
        # A pump whose members have all gone finishes on its own; one still
        # waiting for a member that never arrived is released by its end.
        for helper in helpers:
            helper.join(_POLL)
        for end in ends:
            end.close()
        for helper in helpers:
            helper.join(_JOIN)

    results = [_result(members[pid]) for pid in ids if pid in members]
    failure = next((r for r in results if r.id == failed), None)
    failures = (
        [r for r in results if r.id == failed]
        if timed_out
        else [r for r in results if r.exit_code != 0 and not r.terminated]
    )
    code = 0 if failure is None else (_FAILED if timed_out else failure.exit_code)
    return StageResult(
        index=stage.index,
        members=results,
        exit_code=code,
        timed_out=timed_out and full is None,
        failure=failure,
        failures=failures,
        overflow=(
            None
            if full is None
            else overflow_error(full, stall if stall is not None else DEFAULT_STALL)
        ),
    )


def _spawn_order(ids: Sequence[str], stage_wires: Sequence[Wire]) -> list[str]:
    """`ids` reordered so a chained producer is spawned before its consumer.

    Only a wire stdio carries at both ends constrains anything: the consumer is
    handed the producer's stdout, which does not exist until the producer is
    spawned. Everything else keeps stage order.
    """
    pending = dict.fromkeys(ids, 0)
    after: dict[str, list[str]] = {}
    for wire in stage_wires:
        if not wire.chained:
            continue
        pending[wire.edge.target] += 1
        after.setdefault(wire.edge.source, []).append(wire.edge.target)

    position = {name: index for index, name in enumerate(ids)}
    ready = [position[name] for name in ids if pending[name] == 0]
    heapq.heapify(ready)
    order: list[str] = []
    while ready:
        name = ids[heapq.heappop(ready)]
        order.append(name)
        for follower in after.get(name, []):
            pending[follower] -= 1
            if pending[follower] == 0:
                heapq.heappush(ready, position[follower])

    placed = set(order)
    order.extend(name for name in ids if name not in placed)
    return order


def _watch(
    members: Iterable[_Member],
    deadline: float,
    windows: Sequence[subprocess.Popen[bytes]] | None = None,
    flows: Sequence[Flow] = (),
    stall: float | None = DEFAULT_STALL,
) -> tuple[str | None, bool, Flow | None]:
    """Watch a running stage.

    ``(the member that ended it, whether it timed out, the edge that filled)``.

    Exit codes only: a raw demuxer writes an error to stderr at the pipe's EOF
    and exits 0, so what a member wrote says nothing about whether it worked.

    `flows` are the stage's own pipes, and a stage where every one of them has
    stood still while a copy waits to write is one whose buffers were too
    small (:func:`overflowed`) -- caught here rather than left to the timeout,
    which would report a wedge without naming what wedged it. `stall` of None
    turns that off.

    `windows` are passed only for a stage whose display windows are all it
    feeds: once every one of them has been closed the stage is done, and
    ending it that way is no failure -- the caller stops the members still
    running, and a member it stopped is not counted against the run.
    """
    watched = list(members)
    while True:
        for member in watched:
            code = member.proc.poll()
            if code is not None and code != 0:
                return member.id, False, None
        if all(member.proc.poll() is not None for member in watched):
            return None, False, None
        if windows is not None and all(w.poll() is not None for w in windows):
            return None, False, None
        if stall is not None:
            full = overflowed(flows, time.monotonic(), stall)
            if full is not None:
                held = next(m for m in watched if m.proc.poll() is None)
                return held.id, True, full
        if time.monotonic() >= deadline:
            hung = next(m for m in watched if m.proc.poll() is None)
            return hung.id, True, None
        time.sleep(_POLL)


def _writes_rows_to_stdout(process: Process) -> bool:
    """True for a sink region whose rows name no file: they ride its stdout."""
    return (
        isinstance(process, SidecarProcess)
        and process.sink
        and process.rows is not None
        and not process.rows.alias
        and not process.rows.path
    )


def _spawn(
    command: list[str], stdin: int | IO[bytes], stdout: int | None
) -> subprocess.Popen[bytes]:
    """Spawn one member, in a process group of its own where there are any.

    ``bufsize=0`` for the same reason :mod:`ffrwd.pipes` opens its streams
    unbuffered: a stdio end a copy runs through must hand on what it has,
    since the process that would fill a buffer up to its size is waiting on
    what the buffer holds.
    """
    if sys.platform == "win32":
        return subprocess.Popen(
            command, stdin=stdin, stdout=stdout, stderr=subprocess.PIPE, bufsize=0
        )
    return subprocess.Popen(
        command,
        stdin=stdin,
        stdout=stdout,
        stderr=subprocess.PIPE,
        bufsize=0,
        start_new_session=True,
    )


def _end_tree(proc: subprocess.Popen[bytes]) -> None:
    """End `proc` AND anything it started.

    Ending a process does not end its children, and the ffmpeg on PATH is
    often a shim that runs the real binary as one -- ending only the shim
    leaves an encoder running and a file growing.
    """
    if sys.platform == "win32":
        with contextlib.suppress(OSError, subprocess.SubprocessError):
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=_GRACE,
                check=False,
            )
    else:
        with contextlib.suppress(OSError):
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    with contextlib.suppress(OSError):
        proc.terminate()


def _stop(members: Iterable[_Member]) -> None:
    """End every member still running, and everything those members started."""
    running = [member for member in members if member.proc.poll() is None]
    for member in running:
        member.terminated = True
        _end_tree(member.proc)
    for member in running:
        try:
            member.proc.wait(timeout=_GRACE)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(OSError):
                member.proc.kill()
            with contextlib.suppress(subprocess.TimeoutExpired):
                member.proc.wait(timeout=_GRACE)


def _result(member: _Member) -> ProcessResult:
    code = member.proc.poll()
    return ProcessResult(
        id=member.id,
        argv=member.argv,
        exit_code=_FAILED if code is None else code,
        stderr=b"".join(member.stderr).decode("utf-8", "replace"),
        terminated=member.terminated,
    )


def _stream(stream: IO[bytes] | None) -> IO[bytes]:
    if stream is None:  # defensive: every one of these was asked for as a pipe
        raise FfrwdError(
            ErrorCode.INTERNAL,
            "a stage member was spawned without the pipe its wiring needs",
            hint="this is a compiler bug; please report the query that produced it",
        )
    return stream


def _start(target: Callable[..., None], *args: object) -> threading.Thread:
    thread = threading.Thread(target=target, args=args, daemon=True)
    thread.start()
    return thread


def _pump(source: _End, dest: _End, deadline: float, flow: Flow | None = None) -> None:
    """Copy one stream edge's bytes end to end until the producer stops.

    `_CHUNK` is a ceiling and not a quantum: both ends are unbuffered, so a
    read hands back whatever has arrived and the copy passes exactly that on.
    Waiting for a full chunk would be a deadlock in a plan where one process
    feeds two paths that meet again -- the process that would round the chunk
    up is itself waiting for the frame the held-back tail completes.

    `flow` is where the copy records what it has moved and when, and marks
    itself as waiting on the consuming end -- which is what makes a full
    buffer visible to :func:`overflowed` rather than a stage that simply hangs.
    """
    try:
        reader = source.open(deadline)
        writer = dest.open(deadline)
        while True:
            chunk = reader.read(_CHUNK)
            if not chunk:
                break
            if flow is not None:
                flow.writing = True
            _write_all(writer, chunk)
            if flow is not None:
                flow.writing = False
                flow.moved += len(chunk)
                flow.at = time.monotonic()
        writer.flush()
    except (OSError, ValueError):
        pass  # the other end went away; exit codes are what judge that
    finally:
        if flow is not None:
            flow.writing = False
        dest.close()
        source.close()


def _write_all(writer: IO[bytes], chunk: bytes) -> None:
    """Hand the whole chunk over. An unbuffered write takes what it takes."""
    sent = writer.write(chunk)
    while sent < len(chunk):
        sent += writer.write(chunk[sent:])


def overflowed(flows: Sequence[Flow], now: float, stall: float) -> Flow | None:
    """The edge whose buffer is full, out of a stage that has stopped moving.

    A stage is wedged when nothing has crossed ANY of its pipes for `stall`
    seconds -- the producer waiting on a full buffer stops writing to its other
    edges too, so their consumers starve and the whole stage goes still at
    once. The edge to name is one the copy is waiting to hand over, and the
    deepest bound among those, since that is the one the compiler promised the
    most about. None while anything is still moving, and for a stage where no
    copy is waiting -- that one is idle, not full.
    """
    moving = [flow for flow in flows if flow.moved]
    if not moving or any(now - flow.at < stall for flow in moving):
        return None
    waiting = [flow for flow in moving if flow.writing]
    if not waiting:
        return None
    return max(waiting, key=lambda flow: (flow.bound, flow.held))


def _drain(stream: IO[bytes], into: list[bytes]) -> None:
    """Collect one member's stderr so its pipe never fills and stalls it."""
    try:
        while True:
            chunk = stream.read(_CHUNK)
            if not chunk:
                break
            into.append(chunk)
    except (OSError, ValueError):
        pass
    finally:
        with contextlib.suppress(OSError, ValueError):
            stream.close()
