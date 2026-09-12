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

A `work` callback adds a third, for one command of each graph: the one that
writes the destinations is asked for ffmpeg's own ``-progress`` blocks in
place of its status line, and its stderr is piped so the blocks can be read
off it as they arrive and reported through the callback. What comes back is
the log alone -- the progress lines are taken out of it, so a failure report
carries the warnings and none of them.

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
On an edge the compiler gave a depth, that copy is also where the depth is
actually held -- it reads on while the consumer is behind, so a producer with
frames still to hand over can always finish and close its outputs.

Members are judged by EXIT CODE only. A raw demuxer logs an error at the
pipe's EOF and exits 0 anyway, so stderr says nothing about whether a member
worked; it is captured to be reported, not to be read.

Exit codes alone name the wrong member, though, because losing one member
takes its pipe neighbours with it. Every member's exit TIME is recorded as
well, and the one reported (:attr:`StageResult.failure`) is the one that
ended first while the others were still running -- whatever its code, since
a member that exits 0 while a producer is still writing to it is exactly the
member that ended the run. The members that then died writing into a pipe
nobody was reading are :attr:`StageResult.consequences`, reported as such
rather than mixed in with the failures. A stage that has lost a member waits
`_CASCADE` seconds for those to arrive, so a broken pipe is reported as one
instead of as a member this process stopped.

One failure is not an exit code: a stage whose pipes have all stopped moving
while every member still runs and one copy still waits on the far end. Where
that copy waits to WRITE, the buffer the plan sized from a bound
(:attr:`StreamEdge.bound`) is one the run outgrew; where it waits to OPEN, the
process that edge goes to never asked for it. Either ends the stage with a
typed :attr:`PlanResult.overflow` naming the edge -- not with a timeout that
would say only that something hung.

A ROWS edge is the one edge nothing paces. ffmpeg reads a rows document whole
when it opens that input, and opens its inputs in order, so a producer writing
a second document is writing into an input that will not be opened until the
first one ends: the copy takes the whole document off the producer instead --
in memory, and in a temporary file past a few megabytes -- and hands it over
once the consumer opens.

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
import ctypes
import heapq
import math
import os
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Literal

from . import loudnorm, pipes
from .console import Work, WorkProgress
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
    "terminal_member",
    "unopened",
    "unopened_error",
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
# How long every pipe of a stage may stand still and every member sit idle,
# with all of them running and one copy still waiting on the far end, before
# that edge is called wedged.
DEFAULT_STALL = 30.0
# The 100 ns units Windows reports a process's CPU time in, per second.
_FILETIME_TICKS = 1e7
# How long a member that was told to stop is given before it is killed.
_GRACE = 5.0
# How long a stage that has lost a member waits for the members around it to
# end on their own, so their broken pipes are reported rather than stopped.
_CASCADE = 1.0
# ffmpeg's EPIPE, which is AVERROR(EPIPE) == -32, as each platform's exit
# status spells it: POSIX keeps the low byte, Windows the whole word.
_BROKEN_PIPE = frozenset({224, 0xFFFFFFE0})
# How long a helper thread is waited for once its process has gone.
_JOIN = 5.0
# Bytes moved per copy between a named pipe and a process's stdio.
_CHUNK = 1 << 16
# How far ahead of the consumer a copy reads, on an edge the compiler gave a
# depth. Generous enough for several frames of the largest one a plan carries,
# and bounded, so a run whose paths really have drifted apart still stops.
_READ_AHEAD = 1 << 25
# How much of a spooled rows document is held in memory before the rest of it
# goes to a temporary file.
_SPOOL_MEMORY = 1 << 22
# What `ProcessResult.stderr_tail` keeps.
_TAIL_LINES = 20

# What the one ffmpeg whose progress is drawn is asked for instead of its own
# status line: a key=value block on its stderr, twice a second.
_PROGRESS_ARGS = ("-nostats", "-progress", "pipe:2", "-stats_period", "0.5")
# Where those flags go: after `-hide_banner` and `-y`/`-n`, still ahead of the
# first input, since they are global options.
_PROGRESS_AT = 3
# A line of one of those blocks, as against a line of ffmpeg's log. The digits
# are for the per-stream keys, `stream_0_0_q` and its siblings.
_PROGRESS_LINE = re.compile(rb"^[a-z_][a-z0-9_]*=")


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
    # True when Ctrl-C ended the run: not a failure, so `exit_code` stays 0.
    interrupted: bool = False


def execute(
    emitted: Sequence[Emitted],
    *,
    timeout: float | None = DEFAULT_TIMEOUT,
    overwrite: bool = False,
    capture_stderr: bool = False,
    echo: Callable[[list[str]], None] | None = None,
    players: Sequence[list[str] | None] = (),
    show_only: bool = False,
    work: WorkProgress | None = None,
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

    `work` is where ffmpeg's own progress is reported, for the command of each
    graph that writes its destinations -- the last, a measuring pass being a
    pass and not the work. That command's stderr is piped, whatever
    `capture_stderr` says, and comes back with the progress lines taken out. A
    command feeding a window reports nothing: its output is the window.
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
            last = index == len(commands) - 1
            reports = work if last and not measuring and player is None else None
            capture = capture_stderr or measuring

            argv = [loudnorm.substitute(word, measured) for word in command]
            argv.insert(1, "-y" if overwrite else "-n")
            argv.insert(1, "-hide_banner")
            if reports is not None:
                argv[_PROGRESS_AT:_PROGRESS_AT] = _PROGRESS_ARGS

            if echo is not None:
                echo(argv)

            kept = capture or reports is not None
            try:
                code, captured = _run_ffmpeg(
                    argv,
                    timeout,
                    capture=capture,
                    player=player if last else None,
                    mute=player is not None and not last,
                    show_only=show_only,
                    work=reports,
                )
            except subprocess.TimeoutExpired as err:
                # Whatever the killed child had written by then, if captured.
                partial = err.stderr if isinstance(err.stderr, str) else ""
                results.append(CommandResult(argv, _FAILED, partial, kept))
                return ExecutionResult(results, _FAILED, timed_out=True)
            except KeyboardInterrupt:
                # The command itself has already killed its ffmpeg by the time
                # this is caught (`_run_watched`, `_run_with_player`, and
                # `subprocess.run` all do); nothing to append for one that
                # never finished.
                return ExecutionResult(results, 0, interrupted=True)

            results.append(CommandResult(argv, code, captured, kept))
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
    work: WorkProgress | None = None,
) -> tuple[int, str]:
    """Run one ffmpeg command; ``(exit code, its stderr)``.

    Uncaptured stderr writes straight through to the caller's terminal,
    progress lines included, and comes back as "". `work` pipes it instead,
    and comes back with the log alone (:func:`_run_watched`).

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
        if work is not None:
            return _run_watched(argv, timeout, stdout=stdout, work=work)
        if not capture:
            return subprocess.run(argv, timeout=timeout, stdout=stdout).returncode, ""
        done = subprocess.run(
            argv, timeout=timeout, stdout=stdout, stderr=subprocess.PIPE, text=True
        )
        return done.returncode, done.stderr
    return _run_with_player(
        argv, timeout, capture=capture, player=player, show_only=show_only
    )


def _run_watched(
    argv: list[str],
    timeout: float | None,
    *,
    stdout: int | None,
    work: WorkProgress,
) -> tuple[int, str]:
    """Run one ffmpeg command, reporting its progress as it writes it.

    Its stderr is piped and read in a thread of its own, so a block reaches
    `work` as ffmpeg writes it rather than when the command ends; what comes
    back is everything else on that stream, which is the log a failure is
    reported with.
    """
    proc = subprocess.Popen(
        argv, stdout=stdout, stderr=subprocess.PIPE, bufsize=0, start_new_session=_own_session()
    )
    log: list[bytes] = []
    reading = _start(_drain_work, _stream(proc.stderr), log, work)

    def _abort() -> None:
        _end_tree(proc)
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(_GRACE)
        reading.join(_JOIN)

    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired as err:
        _abort()
        raise subprocess.TimeoutExpired(
            argv, err.timeout, stderr=_text(log)
        ) from err
    except KeyboardInterrupt:
        # Unlike `subprocess.run`, a bare `Popen.wait()` does not kill its
        # child on an interrupt -- Ctrl-C would otherwise leave this ffmpeg
        # running past the command that was watching it.
        _abort()
        raise
    reading.join(_JOIN)
    return proc.returncode, _text(log)


def _text(log: Sequence[bytes]) -> str:
    return b"".join(log).decode("utf-8", "replace")


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
    watching = subprocess.Popen(
        player, stdin=subprocess.PIPE, bufsize=0, start_new_session=_own_session()
    )
    ffmpeg = subprocess.Popen(
        argv, stdout=subprocess.PIPE, stderr=stderr, bufsize=0, start_new_session=_own_session()
    )
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
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
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

# Renders one sidecar process as the argv that runs it, given the path each
# stream it reads arrives on and the path each rows document it writes goes
# to. The real one lands with the sidecar itself; until then a caller
# supplies it.
SidecarArgv = Callable[[SidecarProcess, Sequence[str], Sequence[str]], list[str]]


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
    def command(self) -> str:
        """This member's argv as one line, whole: a report shows all of it."""
        return " ".join(self.argv)

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
    a full buffer looks like from here. `opening` is True earlier than that:
    the copy is waiting for the consuming process to open its end at all.
    """

    edge: PipeEdge
    at: float
    moved: int = 0
    writing: bool = False
    opening: bool = False

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
        f"still running, nothing has crossed any pipe of this stage and every "
        f"process of it has sat idle for {stall:.0f}s",
        hint="the paths out of the one process reading the input drifted "
        "further apart than the compiler counted them: record the input to a "
        "file and run this query over the file, or take the slower path's "
        "work out of the pipeline",
    )


def unopened_error(flow: Flow, stall: float) -> FfrwdError:
    """The typed failure for a stage wedged on an end its consumer never opened.

    Names the edge and what the copy was waiting for, the same way
    :func:`overflow_error` does: a copy holding everything the producer gave
    it, and a consuming process that has not asked for any of it.
    """
    carried = flow.edge.ref if isinstance(flow.edge, StreamEdge) else flow.edge.alias
    return FfrwdError(
        ErrorCode.INPUT_NEVER_OPENED,
        f"the pipe carrying '{carried}' from {flow.edge.source} to "
        f"{flow.edge.target} has nowhere to go: the consumer never opened its "
        f"input, and with every process still running, nothing has crossed any "
        f"pipe of this stage and every process of it has sat idle for "
        f"{stall:.0f}s",
        hint="the process reading it opens its inputs in order and is still "
        "waiting on an earlier one of its own, which cannot end while this "
        "one waits: hand it that earlier input first, or write this one to a "
        "file and run the query over the file",
    )


@dataclass(frozen=True)
class StageResult:
    """One stage: every member it ran, and the outcome of running them."""

    index: int
    members: list[ProcessResult] = field(default_factory=list)
    # The failing member's exit code, or 1 where there is none to report --
    # a timeout, or a member that ended the stage by exiting 0 -- or 0.
    exit_code: int = 0
    timed_out: bool = False
    # The member that ended the stage: the first to end while the others were
    # still running -- a nonzero exit, or a 0 from a member a producer was
    # still writing to in a stage something else failed in. None when nothing
    # failed, whatever order the members ended in. On a timeout, the member
    # still running when it struck.
    failure: ProcessResult | None = None
    # Every member that failed on its own rather than being told to stop, and
    # not because the member before it had gone: `failure` first, then any
    # member that failed independently. On a timeout, the member still running.
    failures: list[ProcessResult] = field(default_factory=list)
    # The members that ended after `failure` writing into a pipe nobody was
    # reading -- consequences of it, in the order they were spawned.
    consequences: list[ProcessResult] = field(default_factory=list)
    # The edge that wedged this stage -- a buffer that filled, or an end its
    # consumer never opened -- when that is what ended it. Set instead of
    # `timed_out`, since the two answer the same question and this one names
    # the edge.
    overflow: FfrwdError | None = None
    # True when Ctrl-C ended the stage: every member stopped the way `_stop`
    # always stops one, not a failure -- `exit_code` stays 0 and `failure`
    # stays None.
    interrupted: bool = False


@dataclass(frozen=True)
class PlanResult:
    """Every stage run, in order, plus the run's own outcome."""

    stages: list[StageResult] = field(default_factory=list)
    exit_code: int = 0
    timed_out: bool = False
    failure: ProcessResult | None = None
    failures: list[ProcessResult] = field(default_factory=list)
    overflow: FfrwdError | None = None
    consequences: list[ProcessResult] = field(default_factory=list)
    # True when Ctrl-C ended the run at the stage named last in `stages`.
    interrupted: bool = False


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
        if isinstance(process, SidecarProcess):
            argv[process.id] = _sidecar_args(
                process,
                sidecar_argv,
                [read[edge] for edge in incoming],
                _sidecar_writes(process, plan, write, outgoing),
                len(outgoing),
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


def _sidecar_writes(
    process: SidecarProcess,
    plan: ProcessPlan,
    write: Mapping[PipeEdge, str],
    outgoing: Sequence[StreamEdge],
) -> list[str]:
    """Where `process` writes go: a packet source's tracks in `outgoing`'s
    own order, which is its catalog order, and then its rows documents.
    Everything else hands its frames on over the one stdout instead."""
    streams = [write[edge] for edge in outgoing] if process.packet_source else []
    return streams + _rows_writes(process, plan, write)


def _rows_writes(
    process: SidecarProcess, plan: ProcessPlan, write: Mapping[PipeEdge, str]
) -> list[str]:
    """Where each rows document `process` writes goes, in document order.

    A document another process reads as a track goes to that edge's own
    path -- stdout where the plan takes one thing from this process, a named
    pipe where it takes several. A document the query sent to a file names
    its own destination and takes no edge, so it takes no path here either.
    """
    fed = iter(e for e in plan.rows_edges if e.source == process.id)
    return [
        write[next(fed)] if document.sink.alias else "" for document in process.rows
    ]


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


def terminal_member(plan: ProcessPlan) -> str | None:
    """The member that writes `plan`'s destinations, whose progress is the run's.

    The last stage's ffmpeg that hands nothing on: everything else feeds a
    pipe, and a run is waiting on the one actually writing files. Several
    destinations out of one ffmpeg are still one member. None for a plan whose
    last stage ends in a sidecar instead, which reports nothing of the kind.
    """
    stages = plan.stages
    if not stages:
        return None
    handing_on = {edge.source for edge in plan.edges}
    writing = [
        pid
        for pid in stages[-1].processes
        if pid not in handing_on and isinstance(plan.process(pid), FfmpegProcess)
    ]
    return writing[-1] if writing else None


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
    work: WorkProgress | None = None,
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
    member running and one copy still waiting on the far end, before that
    edge is reported: :attr:`PlanResult.overflow` names it, and whether it was
    a buffer that filled or an input nobody opened, rather than leaving a
    wedged run to the timeout. None turns that off.

    `work` is where ffmpeg's own progress is reported, for the one member that
    writes the plan's destinations (:func:`terminal_member`). Every other
    member's stderr is collected as it always was.

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
        terminal = terminal_member(plan) if work is not None else None

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
                work=work,
                terminal=terminal,
            )
            stages.append(result)
            if result.interrupted:
                return PlanResult(stages, interrupted=True)
            if result.exit_code != 0:
                return PlanResult(
                    stages,
                    result.exit_code,
                    result.timed_out,
                    result.failure,
                    result.failures,
                    result.overflow,
                    result.consequences,
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
    reads: Sequence[str],
    writes: Sequence[str],
    streams: int,
) -> list[str]:
    """One sidecar process as argv, its ends spelled as the plan names them.

    `reads` is one path per incoming edge, in the order the module's own pads
    take them: stdin where the plan hands this process one thing, a named pipe
    per edge where it fans in. Only a SINK reads several -- everything else
    takes its pads out of one input. `writes` is one path per outgoing stream
    edge a packet SOURCE hands on, ahead of one path per rows document, which
    a process may write several of, each to a path of its own; `streams` is
    how many streams of frames it hands on, and one stdout is all there is to
    carry those -- except for a packet SOURCE, whose several tracks are each
    their own named pipe by construction, the mirror of a packet sink's
    several reads.
    """
    if hook is None:
        raise FfrwdError(
            ErrorCode.INTERNAL,
            f"process {process.id!r} hosts the module {process.module!r}, and "
            "nothing was given to spawn it",
            hint="pass sidecar_argv, which renders one sidecar process as argv",
        )
    if streams > 1 and not process.packet_source:
        raise FfrwdError(
            ErrorCode.INTERNAL,
            f"process {process.id!r} writes {streams} streams, but only its own "
            "stdout is wired",
            hint="a sidecar writing more than one stream needs argv that can "
            "spell a named pipe path",
        )
    if len(reads) > 1 and not process.packet_sink:
        raise FfrwdError(
            ErrorCode.INTERNAL,
            f"process {process.id!r} reads {len(reads)} streams and hosts no "
            "packet sink",
            hint="a frame module takes its pads out of one input, wired by the "
            "network string",
        )
    return list(hook(process, reads, writes))


def _spawn_argv(
    process: FfmpegProcess | SidecarProcess,
    argv: Sequence[str],
    *,
    overwrite: bool,
    progress: bool = False,
) -> list[str]:
    """`argv` as it is actually spawned. A sidecar's is the hook's, verbatim.

    `progress` asks this one for the blocks a drawn line reads, in place of
    the status line it would write.
    """
    if not isinstance(process, FfmpegProcess):
        return list(argv)
    command = list(argv)
    command.insert(1, "-y" if overwrite else "-n")
    command.insert(1, "-hide_banner")
    if progress:
        command[_PROGRESS_AT:_PROGRESS_AT] = _PROGRESS_ARGS
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
    # When the watch first saw this member had exited; None while it runs, and
    # for one still running when the stage was stopped.
    ended_at: float | None = None


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
    *,
    work: WorkProgress | None = None,
    terminal: str | None = None,
) -> StageResult:
    """Spawn every member of `stage` at once, watch them, and report.

    `terminal` is the member whose progress `work` draws, which is in the last
    stage; every other member, and every stage before it, is untouched.

    :func:`_watch` says WHETHER a member ended the stage; :func:`_attribute`
    then says which member is the cause and whether there was one at all, read
    off the exit times it recorded rather than off the exit codes. A stage
    nothing ended -- every member finished, or every one was stopped because
    the last display window closed -- is asked neither question.
    """
    ids = list(stage.processes)
    inside = set(ids)
    stage_wires = [
        wire
        for wire in assigned
        if wire.edge.source in inside and wire.edge.target in inside
    ]
    feeds = [(wire.edge.source, wire.edge.target) for wire in stage_wires]
    deadline = math.inf if timeout is None else time.monotonic() + timeout
    members: dict[str, _Member] = {}
    watching: dict[str, subprocess.Popen[bytes]] = {}
    helpers: list[threading.Thread] = []
    ends: list[_End] = []
    flows: list[Flow] = []
    failed: str | None = None
    timed_out = False
    wedge: FfrwdError | None = None
    interrupted = False
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
                progress=work is not None and pid == terminal,
            )
            if echo is not None:
                echo(pid, command)
            if player is not None:
                watching[pid] = subprocess.Popen(
                    player, stdin=subprocess.PIPE, bufsize=0
                )
            members[pid] = _Member(
                id=pid,
                argv=command,
                proc=_spawn(command, stdin, stdout, env=_nn_runtime_env(command)),
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
            stderr = _stream(member.proc.stderr)
            if work is not None and member.id == terminal:
                helpers.append(_start(_drain_work, stderr, member.stderr, work))
            else:
                helpers.append(_start(_drain, stderr, member.stderr))

        failed, timed_out, wedge = _watch(
            members.values(),
            deadline,
            list(watching.values()) if show_only and watching else None,
            flows,
            stall,
            feeds,
        )
    except KeyboardInterrupt:
        # `failed`/`timed_out`/`wedge` stay at their unstruck defaults: the
        # stage below reads as a clean stop, not a failure.
        interrupted = True
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
    consequences: list[ProcessResult] = []
    failure: ProcessResult | None = None
    failures: list[ProcessResult] = []
    code = 0
    if timed_out:
        failure = next((r for r in results if r.id == failed), None)
        failures = [r for r in results if r.id == failed]
        code = 0 if failure is None else _FAILED
    elif failed is not None:
        ended = {
            member.id: member.ended_at
            for member in members.values()
            if member.ended_at is not None
        }
        failure, consequences = _attribute(results, ended, feeds)
        blamed = {r.id for r in consequences}
        failures = [
            r
            for r in results
            if r.exit_code != 0 and not r.terminated and r.id not in blamed
        ]
        if failure is not None and failure.id not in {r.id for r in failures}:
            failures.insert(0, failure)
        # A member that ended the stage by exiting 0 leaves no code to report,
        # so the stage carries the one a timeout does.
        code = 0 if failure is None else (failure.exit_code or _FAILED)
    return StageResult(
        index=stage.index,
        members=results,
        exit_code=code,
        timed_out=timed_out and wedge is None,
        failure=failure,
        failures=failures,
        consequences=consequences,
        overflow=wedge,
        interrupted=interrupted,
    )


def _attribute(
    results: Sequence[ProcessResult],
    ended: Mapping[str, float],
    feeds: Sequence[tuple[str, str]] = (),
) -> tuple[ProcessResult | None, list[ProcessResult]]:
    """Which member ended the stage, and which ones broke because it did.

    The cause is the first member to end while others were still running and
    to have no business ending: any nonzero exit, or -- where something in the
    stage did fail -- a 0 from a member one of `feeds`' producers was still
    writing to. `ended` is when each member was first seen to have
    exited, and a member missing from it was still running when the stage was
    stopped. The consequences are the members that ended from that moment on
    writing into a pipe nobody was reading.

    A stage every member of which exited 0, none of them stopped, has no
    cause and names nobody: nothing there was cut short, since a producer
    whose reader had gone would have died of a broken pipe rather than
    exited 0. A consumer reaching the end of its own graph while its
    producers are still draining theirs is how a finite output legitimately
    ends, and only a real failure elsewhere -- a nonzero exit, a member told
    to stop -- makes such a 0 the reason for it.

    Called only for a stage something ended, which is what lets two members
    seen ending in the SAME poll be read as the cascade they are: one poll
    cannot order two exits inside it, so a producer seen ending no earlier
    than its consumer still counts as writing to it, and a broken pipe loses
    the tie to anything else -- it is never the reason a pipe closed.
    """
    producers: dict[str, list[str]] = {}
    for source, target in feeds:
        producers.setdefault(target, []).append(source)

    def writing_at(pid: str, when: float) -> bool:
        at = ended.get(pid)
        return at is None or at >= when

    # Whether anything in this stage failed at all, which is what a 0 needs
    # behind it before it can be read as the end of anything.
    any_failed = any(result.exit_code != 0 or result.terminated for result in results)
    candidates: list[tuple[float, bool, ProcessResult]] = []
    for result in results:
        at = ended.get(result.id)
        if at is None or result.terminated:
            continue
        writing = producers.get(result.id, ())
        if result.exit_code != 0 or (
            any_failed and any(writing_at(pid, at) for pid in writing)
        ):
            candidates.append((at, _broken_pipe(result), result))
    if not candidates:
        return None, []
    cause = min(candidates, key=lambda entry: (entry[0], entry[1]))[2]
    since = ended[cause.id]
    consequences = [
        result
        for result in results
        if result.id != cause.id
        and _broken_pipe(result)
        and ended.get(result.id, math.inf) >= since
    ]
    return cause, consequences


def _broken_pipe(result: ProcessResult) -> bool:
    """True when this member died writing to a pipe nobody was reading."""
    if result.terminated or result.exit_code == 0:
        return False
    return result.exit_code in _BROKEN_PIPE or "Broken pipe" in result.stderr


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


def _ends_the_stage(
    member: _Member,
    by_id: Mapping[str, _Member],
    producers: Mapping[str, Sequence[str]],
) -> bool:
    """True when this member's ending is the end of the stage.

    Any nonzero exit, and a 0 from a member one of its producers is still
    writing to -- the muxer that stops early leaves everything upstream
    writing into a closed pipe, and its 0 is the cause of every broken pipe
    that follows.
    """
    if member.proc.poll() != 0:
        return True
    return any(
        by_id[pid].ended_at is None
        for pid in producers.get(member.id, ())
        if pid in by_id
    )


def _watch(
    members: Iterable[_Member],
    deadline: float,
    windows: Sequence[subprocess.Popen[bytes]] | None = None,
    flows: Sequence[Flow] = (),
    stall: float | None = DEFAULT_STALL,
    feeds: Sequence[tuple[str, str]] = (),
) -> tuple[str | None, bool, FfrwdError | None]:
    """Watch a running stage, recording when each member ends.

    ``(the member that ended it, whether it timed out, what wedged it)``.

    Exit codes only: a raw demuxer writes an error to stderr at the pipe's EOF
    and exits 0, so what a member wrote says nothing about whether it worked.
    A 0 is not always a finish, though: `feeds` are the stage's own
    ``(producer, consumer)`` pairs, and a member that exits 0 while one of its
    producers is still writing to it ends the stage the same way a nonzero
    exit does -- it is about to leave everything upstream writing into a
    closed pipe. From there the stage is given `_CASCADE` seconds for those
    members to end on their own, so their broken pipes are recorded rather
    than stopped, and :attr:`_Member.ended_at` orders cause before
    consequence.

    `flows` are the stage's own pipes. A stage where every one of them has
    stood still is wedged in one of two ways, and both are named rather than
    left to the timeout, which would report a wedge without saying what wedged
    it: a copy waiting to WRITE means the buffer it waits on was too small
    (:func:`overflowed`), and one waiting to OPEN means the consumer never
    asked for the edge at all (:func:`unopened`). `stall` of None turns both
    off.

    Neither is asked while the stage is working: the pipes a stage pumps are
    not all the pipes it has, so a member computing between two writes stands
    still on every pumped edge without being wedged. Both detectors wait for
    the members' own CPU time to stop advancing for `stall` seconds too.

    `windows` are passed only for a stage whose display windows are all it
    feeds: once every one of them has been closed the stage is done, and
    ending it that way is no failure -- the caller stops the members still
    running, and a member it stopped is not counted against the run.
    """
    watched = list(members)
    by_id = {member.id: member for member in watched}
    producers: dict[str, list[str]] = {}
    for source, target in feeds:
        producers.setdefault(target, []).append(source)
    used: dict[str, float | None] = {m.id: _cpu_seconds(m.proc) for m in watched}
    working = time.monotonic()
    ended: str | None = None
    settled = math.inf
    while True:
        now = time.monotonic()
        just_ended = [
            member
            for member in watched
            if member.ended_at is None and member.proc.poll() is not None
        ]
        for member in just_ended:
            member.ended_at = now
        if ended is None:
            ender = next(
                (m for m in just_ended if _ends_the_stage(m, by_id, producers)), None
            )
            if ender is not None:
                ended = ender.id
                settled = now + _CASCADE
        if all(member.ended_at is not None for member in watched):
            return ended, False, None
        if ended is not None:
            if time.monotonic() >= settled:
                return ended, False, None
            time.sleep(_POLL)
            continue
        if windows is not None and all(w.poll() is not None for w in windows):
            return None, False, None
        if stall is not None:
            now = time.monotonic()
            readable = False
            for member in watched:
                seconds = _cpu_seconds(member.proc)
                if seconds is None:
                    continue
                readable = True
                before = used[member.id]
                if before is None or seconds > before:
                    working = now
                used[member.id] = seconds
            # Where no member's CPU can be read the pipes are all there is to
            # go on, and the stall detectors answer on them alone.
            if not readable or now - working >= stall:
                full = overflowed(flows, now, stall)
                if full is not None:
                    held = next(m for m in watched if m.proc.poll() is None)
                    return held.id, True, overflow_error(full, stall)
                stuck = unopened(flows, now, stall)
                if stuck is not None:
                    held = next(m for m in watched if m.proc.poll() is None)
                    return held.id, True, unopened_error(stuck, stall)
        if time.monotonic() >= deadline:
            hung = next(m for m in watched if m.proc.poll() is None)
            return hung.id, True, None
        time.sleep(_POLL)


def _cpu_seconds(proc: subprocess.Popen[bytes]) -> float | None:
    """The CPU time one member has used so far, kernel and user, in seconds.

    None where this platform, or this process, cannot be read -- and a member
    that has already exited reads as None on Linux, its ``/proc`` entry being
    gone the moment :meth:`Popen.poll` reaps it.
    """
    if sys.platform == "win32":
        handle = getattr(proc, "_handle", None)
        if handle is None:
            return None
        created = ctypes.c_uint64()
        exited = ctypes.c_uint64()
        kernel = ctypes.c_uint64()
        user = ctypes.c_uint64()
        ok = ctypes.windll.kernel32.GetProcessTimes(
            ctypes.c_void_p(int(handle)),
            ctypes.byref(created),
            ctypes.byref(exited),
            ctypes.byref(kernel),
            ctypes.byref(user),
        )
        if not ok:
            return None
        return (kernel.value + user.value) / _FILETIME_TICKS
    if sys.platform.startswith("linux"):
        pid = getattr(proc, "pid", None)
        if pid is None:
            return None
        try:
            stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        except OSError:
            return None
        # The command name is parenthesised and may hold spaces, so the fields
        # are counted from after it: utime and stime are the 12th and 13th.
        fields = stat.rpartition(")")[2].split()
        if len(fields) < 13:
            return None
        try:
            ticks = int(fields[11]) + int(fields[12])
        except ValueError:
            return None
        return ticks / os.sysconf("SC_CLK_TCK")
    return None


def _writes_rows_to_stdout(process: Process) -> bool:
    """True for a sink region whose rows name no file: they ride its stdout."""
    return (
        isinstance(process, SidecarProcess)
        and process.sink
        and any(
            not document.sink.alias and not document.sink.path
            for document in process.rows
        )
    )


_NN_RUNTIME_FLAG = "-nn-runtime"


def _nn_runtime_env(command: Sequence[str]) -> dict[str, str] | None:
    """The environment a spawn needs beyond what it would inherit, if any.

    A sidecar given ``-nn-runtime <dir>`` fetched ONNX Runtime's CUDA provider
    into that directory, but the provider then dlopens its own CUDA/cuDNN
    libraries by soname -- not by any path the sidecar passes it. On Windows
    the sidecar's own loader adds the directory itself; elsewhere only
    LD_LIBRARY_PATH puts it where the platform loader resolves those sonames,
    so a non-Windows spawn gets the directory prepended to it. None where
    there is nothing to add, which leaves the member spawned with the
    inherited environment untouched.
    """
    if sys.platform == "win32":
        return None
    try:
        runtime = command[command.index(_NN_RUNTIME_FLAG) + 1]
    except (ValueError, IndexError):
        return None
    env = dict(os.environ)
    existing = env.get("LD_LIBRARY_PATH")
    env["LD_LIBRARY_PATH"] = f"{runtime}{os.pathsep}{existing}" if existing else runtime
    return env


def _spawn(
    command: list[str],
    stdin: int | IO[bytes],
    stdout: int | None,
    env: Mapping[str, str] | None = None,
) -> subprocess.Popen[bytes]:
    """Spawn one member, in a process group of its own where there are any.

    ``bufsize=0`` for the same reason :mod:`ffrwd.pipes` opens its streams
    unbuffered: a stdio end a copy runs through must hand on what it has,
    since the process that would fill a buffer up to its size is waiting on
    what the buffer holds. `env` of None inherits this process's own
    environment, exactly as leaving it out of :class:`subprocess.Popen` would.
    """
    if sys.platform == "win32":
        return subprocess.Popen(
            command,
            stdin=stdin,
            stdout=stdout,
            stderr=subprocess.PIPE,
            bufsize=0,
            env=env,
        )
    return subprocess.Popen(
        command,
        stdin=stdin,
        stdout=stdout,
        stderr=subprocess.PIPE,
        bufsize=0,
        start_new_session=True,
        env=env,
    )


def _own_session() -> bool:
    """Whether a child is put in a session of its own: off Windows, always.

    What lets :func:`_end_tree` signal the child's whole group without
    signalling this process and its shell, which share the inherited group.
    """
    return sys.platform != "win32"


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
            group = os.getpgid(proc.pid)
            # A child sharing this process's group is ended alone: signalling
            # the group would end this process and its shell with it.
            if group != os.getpgid(0):
                os.killpg(group, signal.SIGTERM)
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


class _Ahead:
    """Bytes a copy has taken from the producer and not yet handed on.

    Bounded by `limit`: once that much is held the reading side waits, so a
    run whose paths really have drifted apart still stops rather than growing
    without end -- what is held here is the depth, not a reservoir.
    """

    def __init__(self, limit: int) -> None:
        self._limit = limit
        self._held: deque[bytes] = deque()
        self._size = 0
        self._ended = False
        self._change = threading.Condition()

    def put(self, chunk: bytes) -> bool:
        """Hold `chunk`, waiting for room. False once the copy has finished."""
        with self._change:
            while self._size >= self._limit and not self._ended:
                self._change.wait()
            if self._ended:
                return False
            self._held.append(chunk)
            self._size += len(chunk)
            self._change.notify_all()
            return True

    def take(self) -> bytes | None:
        """The next bytes to write; None once nothing more will arrive."""
        with self._change:
            while not self._held and not self._ended:
                self._change.wait()
            if not self._held:
                return None
            chunk = self._held.popleft()
            self._size -= len(chunk)
            self._change.notify_all()
            return chunk

    def finish(self) -> None:
        """Nothing more will be read. Releases whichever side is waiting."""
        with self._change:
            self._ended = True
            self._change.notify_all()


class _Spool:
    """Bytes taken off the producer and held until the consumer opens.

    A rows document is read WHOLE by the process it goes to, at the moment
    that process opens the input, so there is nothing to pace and everything
    to hold: a producer left blocked on a consumer that has not opened yet is
    a stage that never moves again. Held in memory up to `_SPOOL_MEMORY`, and
    in a temporary file past that, which goes when the copy does.

    `flow` counts the bytes as they arrive rather than as they are handed on,
    so a stage watching its pipes sees this edge moving while its consumer is
    still opening.
    """

    def __init__(self, flow: Flow | None = None) -> None:
        self._flow = flow
        self._held: deque[bytes] = deque()
        self._size = 0
        self._file: IO[bytes] | None = None
        self._written = 0
        self._read = 0
        self._ended = False
        self._change = threading.Condition()

    def put(self, chunk: bytes) -> bool:
        """Hold `chunk`, never waiting. False once the copy has finished."""
        with self._change:
            if self._ended:
                return False
            if self._file is None and self._size + len(chunk) > _SPOOL_MEMORY:
                self._file = tempfile.TemporaryFile(prefix="ffrwd-rows-")
            if self._file is None:
                self._held.append(chunk)
                self._size += len(chunk)
            else:
                self._file.seek(self._written)
                self._file.write(chunk)
                self._written += len(chunk)
            if self._flow is not None:
                self._flow.moved += len(chunk)
                self._flow.at = time.monotonic()
            self._change.notify_all()
            return True

    def take(self) -> bytes | None:
        """The next bytes to write; None once nothing more will arrive."""
        with self._change:
            while not self._ready() and not self._ended:
                self._change.wait()
            if self._held:
                chunk = self._held.popleft()
                self._size -= len(chunk)
                return chunk
            if self._file is not None and self._read < self._written:
                self._file.seek(self._read)
                chunk = self._file.read(min(_CHUNK, self._written - self._read))
                self._read += len(chunk)
                return chunk
            return None

    def _ready(self) -> bool:
        """True while there are bytes to hand on: memory first, then file."""
        spilled = self._file is not None and self._read < self._written
        return bool(self._held) or spilled

    def finish(self) -> None:
        """Nothing more will be read. Releases a take that is waiting."""
        with self._change:
            self._ended = True
            self._change.notify_all()

    def release(self) -> None:
        """Drop what is still held, and the temporary file with it."""
        with self._change:
            self._ended = True
            self._held.clear()
            self._size = 0
            if self._file is not None:
                with contextlib.suppress(OSError):
                    self._file.close()
                self._file = None
            self._change.notify_all()


def _fill(reader: IO[bytes], ahead: _Ahead | _Spool) -> None:
    """Take from the producer as fast as it writes, as far as `ahead` holds."""
    try:
        while (chunk := reader.read(_CHUNK)) and ahead.put(chunk):
            continue
    except (OSError, ValueError):
        pass
    finally:
        ahead.finish()


def _read_ahead(flow: Flow | None) -> int:
    """How far ahead of the consumer this copy reads.

    An edge the compiler gave a depth is read ahead in earnest. It has to be:
    the depth exists because one process feeds two paths that meet again, and
    the only place it does any good is HERE. Held in the producing ffmpeg --
    its fifo queue, or the pipe it writes -- it sits behind the write that
    blocks, so a producer with frames still to hand over cannot finish and
    close its outputs, the process on the slower path never sees the end of
    its input, and the frames it holds for its own pipeline never come out.
    Every other edge hands each read straight on.
    """
    return _CHUNK if flow is None or not flow.held else _READ_AHEAD


def _pump(source: _End, dest: _End, deadline: float, flow: Flow | None = None) -> None:
    """Copy one stream edge's bytes end to end until the producer stops.

    `_CHUNK` is a ceiling and not a quantum: both ends are unbuffered, so a
    read hands back whatever has arrived and the copy passes exactly that on.
    Waiting for a full chunk would be a deadlock in a plan where one process
    feeds two paths that meet again -- the process that would round the chunk
    up is itself waiting for the frame the held-back tail completes.

    Reading runs in a thread of its own, and starts as soon as the PRODUCING
    end is open rather than once both are: the consumer may still be opening
    an earlier input of its own, and a producer left unread until then fills
    its pipe and stops before it writes the output that earlier input is
    waiting for. How far it reads ahead of the consumer is the depth the
    compiler counted (:func:`_read_ahead`).

    A ROWS edge is spooled instead (:class:`_Spool`): the process it goes to
    reads the whole document when it opens the input, so there is nothing to
    pace, and the producer must never be the one waiting.

    `flow` is where the copy records what it has moved and when, and marks
    itself as waiting on the consuming end -- which is what makes a full
    buffer visible to :func:`overflowed`, and an end nobody opened visible to
    :func:`unopened`, rather than a stage that simply hangs.
    """
    spooled = flow is not None and isinstance(flow.edge, RowsEdge)
    ahead: _Ahead | _Spool = _Spool(flow) if spooled else _Ahead(_read_ahead(flow))
    reading: threading.Thread | None = None
    try:
        reader = source.open(deadline)
        if flow is not None:
            flow.opening = True
        # Reading starts before the consuming end is even open: ffmpeg opens
        # its inputs one at a time, so a producer whose first output nobody is
        # taking yet stops before it reaches the output the consumer is
        # actually waiting on.
        reading = _start(_fill, reader, ahead)
        writer = dest.open(deadline)
        if flow is not None:
            flow.opening = False
        while (chunk := ahead.take()) is not None:
            if flow is not None:
                flow.writing = True
            _write_all(writer, chunk)
            if flow is not None:
                flow.writing = False
                if not spooled:  # a spool counted these as it took them
                    flow.moved += len(chunk)
                flow.at = time.monotonic()
        writer.flush()
    except (OSError, ValueError):
        pass  # the other end went away; exit codes are what judge that
    finally:
        ahead.finish()
        if flow is not None:
            flow.writing = False
            flow.opening = False
        dest.close()
        source.close()
        if reading is not None:
            reading.join(_JOIN)
        if isinstance(ahead, _Spool):
            ahead.release()


def _write_all(writer: IO[bytes], chunk: bytes) -> None:
    """Hand the whole chunk over. An unbuffered write takes what it takes."""
    sent = writer.write(chunk)
    while sent < len(chunk):
        sent += writer.write(chunk[sent:])


def overflowed(flows: Sequence[Flow], now: float, stall: float) -> Flow | None:
    """The edge whose buffer is full, out of a stage that has stopped moving.

    Nothing across ANY of the stage's pipes for `stall` seconds is the half of
    a wedge this answers -- the producer waiting on a full buffer stops writing
    to its other edges too, so their consumers starve and the whole stage goes
    still at once. :func:`_watch` asks only once the members have gone idle as
    well. The edge to name is one the copy is waiting to hand over, and the
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


def unopened(flows: Sequence[Flow], now: float, stall: float) -> Flow | None:
    """The edge whose consumer never opened it, out of a stage that has stopped.

    The stillness test :func:`overflowed` makes -- nothing has crossed ANY
    pipe of the stage for `stall` seconds -- over the copies waiting for the
    consuming process to open its end rather than to take what it was handed.
    A copy still in its open holds everything the producer gave it and can
    hand none of it on, so the process it reads is not going to finish either.
    None for a stage where every copy is through its open: that one is idle,
    not wedged. The edge to name is the one holding the most.
    """
    moving = [flow for flow in flows if flow.moved]
    if not moving or any(now - flow.at < stall for flow in moving):
        return None
    stuck = [flow for flow in flows if flow.opening]
    if not stuck:
        return None
    return max(stuck, key=lambda flow: flow.moved)


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


def _drain_work(stream: IO[bytes], into: list[bytes], report: WorkProgress) -> None:
    """Drain the one member whose progress is drawn, `into` keeping its log.

    The same drain as every other member's, with a reader in front of it: the
    progress blocks go to `report` as they arrive, the rest to `into`. The
    run's line is ended whatever ends the stream, so a member that was killed
    leaves no line standing.
    """
    reader = _WorkReader(into, report)
    try:
        while True:
            chunk = stream.read(_CHUNK)
            if not chunk:
                break
            reader.feed(chunk)
    except (OSError, ValueError):
        pass
    finally:
        reader.finish()
        with contextlib.suppress(OSError, ValueError):
            stream.close()


class _WorkReader:
    """One member's stderr split into ffmpeg's progress blocks and its log.

    ``-progress`` writes a block of ``key=value`` lines per period, ending in
    ``progress=continue`` or ``progress=end``. Everything else on the stream is
    the log a failure is reported with, and the progress lines never join it.
    Chunks arrive as the pipe hands them over, so a block -- and a line --
    routinely spans two of them.
    """

    def __init__(self, log: list[bytes], report: WorkProgress) -> None:
        self._log = log
        self._report = report
        self._rest = b""
        self._block: dict[str, str] = {}
        self._reached = 0.0
        self._ended = False

    def feed(self, chunk: bytes) -> None:
        """Take what has arrived, reporting every block it completes."""
        lines = (self._rest + chunk).split(b"\n")
        self._rest = lines.pop()
        for line in lines:
            self._line(line)

    def finish(self) -> None:
        """Take what the last chunk left over, and end the run's line."""
        if self._rest:
            self._line(self._rest)
            self._rest = b""
        self._end()

    def _line(self, line: bytes) -> None:
        if not _PROGRESS_LINE.match(line.rstrip(b"\r")):
            self._log.append(line + b"\n")
            return
        key, _, value = line.strip().decode("utf-8", "replace").partition("=")
        if key != "progress":
            self._block[key] = value
            return
        if value == "end":
            self._end()
        else:
            self._report(self._work(done=False))
        self._block.clear()

    def _end(self) -> None:
        if self._ended:
            return
        self._ended = True
        self._report(self._work(done=True))

    def _work(self, *, done: bool) -> Work:
        self._reached = self._out_time()
        return Work(
            out_time=self._reached,
            fps=_figure(self._block.get("fps")),
            speed=_figure(self._block.get("speed"), "x"),
            bitrate=_figure(self._block.get("bitrate"), "kbits/s"),
            total_size=_count(self._block.get("total_size")),
            done=done,
        )

    def _out_time(self) -> float:
        """How far into the output this block reached, in seconds.

        ``out_time`` is a timestamp; the two counts beside it are both in
        microseconds, ``out_time_ms`` included. A block that named none of
        them -- the last one of a run that wrote nothing -- holds where the
        run had got to.
        """
        stamp = self._block.get("out_time")
        if stamp is not None:
            seconds = _timestamp(stamp)
            if seconds is not None:
                return seconds
        for key in ("out_time_us", "out_time_ms"):
            micros = _figure(self._block.get(key))
            if micros is not None:
                return micros / 1e6
        return self._reached


def _figure(value: str | None, suffix: str = "") -> float | None:
    """One progress value as a number, or None where ffmpeg said ``N/A``."""
    if value is None:
        return None
    text = value.strip().removesuffix(suffix)
    try:
        return float(text)
    except ValueError:
        return None


def _count(value: str | None) -> int | None:
    number = _figure(value)
    return None if number is None else int(number)


def _timestamp(value: str) -> float | None:
    """``00:20:15.150000`` as seconds, or None for anything else."""
    parts = value.strip().split(":")
    if len(parts) != 3:
        return None
    try:
        hours, minutes, seconds = (float(part) for part in parts)
    except ValueError:
        return None
    return hours * 3600 + minutes * 60 + seconds
