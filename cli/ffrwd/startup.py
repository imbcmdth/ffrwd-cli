"""Whether every process of a plan can reach its first frame.

ffmpeg opens its inputs ONE AT A TIME: input 1's header is not read until
input 0's has arrived. It writes its outputs interleaved by dts, so an output
whose pipe nobody is draining stops the whole process -- the outputs after it
included. Between those two facts a plan can wedge before a frame moves: a
consumer blocked opening its first pipe, and the producer that would have
filled that pipe blocked writing an earlier output to a consumer that has not
opened yet.

Which output a process writes first, and which input each consumer opens
first, are the plan's to choose, and here they are chosen. :func:`arrange`
walks the plan forward from the processes that read nothing: a process writes
its output headers once every pipe it reads carries frames, and the FRAMES of
one output at a time. The walk it finds IS the order -- each producer's pipe
outputs in the order their frames were reached, each ``-i`` list in the order
the headers it waits for were written. A plan whose own order already works
comes back unchanged, since the walk prefers the order the plan arrived in.

What a producer may write past
------------------------------
Exactly one thing: an edge into a SIDECAR. A sidecar spawns a reader per
input before its module opens, so nothing writing to one ever waits -- that
edge is CARRIED, and its producer moves straight on to its next output.

Nothing else is. An edge the compiler gave a depth to is drained early by
:mod:`ffrwd.execute`'s copy, but into a FINITE buffer: a camera fills it in
fractions of a second while a module is still loading its model, and then the
producer stops exactly as it would have without the depth. Slack delays the
wait; it does not remove the arc. So every edge into an ffmpeg process makes
its producer wait, sooner or later, for that consumer to be reading -- and
the order has to be right on its own, with the depths as headroom rather than
as an excuse.

:func:`relation` is that reasoning as data -- each milestone and what must
happen before it -- and :func:`stalled` is the cycle in it, if there is one.
:func:`check` refuses a plan that has one, naming the processes that wait on
each other rather than leaving the run to report a full buffer on a pipe that
never carried a byte.

Same contract as the split and partition passes: plan in, plan out, nothing
mutated.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace

from .errors import ErrorCode, FfrwdError
from .ir import PIPE
from .processes import (
    Edge,
    FfmpegProcess,
    ProcessPlan,
    RowsEdge,
    SidecarProcess,
    StreamEdge,
)

__all__ = ["Milestone", "arrange", "check", "relation", "stalled"]

# One thing that has to happen for a plan to start, and the process it happens
# in: ``("open", pid, k)`` is that process taking the header off its k-th pipe
# input, ``("run", pid, 0)`` its having every input open -- which is when it
# starts draining them -- ``("head", pid, j)`` its writing the header of its
# j-th pipe output, and ``("write", pid, j)`` the frames that follow.
Milestone = tuple[str, str, int]

# One edge, named by the two processes and the stream it carries -- an
# identity that survives the copies `replace` makes.
Wire = tuple[str, ...]

# How far a search for a workable order will look before giving up. Reached
# only by a plan far larger than anything the compiler produces; a plan that
# reaches it is refused as though no order existed, which is the safe answer.
_STATES = 20000


def relation(plan: ProcessPlan) -> dict[Milestone, tuple[Milestone, ...]]:
    """Each milestone of `plan`, and the milestones that must come first.

    Read straight off the argv the plan renders: a process's ``-i`` list is
    the order it opens, its pipe outputs the order it writes. Nothing runs.

    A header is small enough for any pipe to hold, so one output's header
    never waits on another's. Frames are not -- a raw video frame is larger
    than a pipe -- so an output whose frames nobody is taking stops the frames
    of every output after it.
    """
    ids = [process.id for process in plan.processes]
    inputs = {pid: _pipe_inputs(plan, pid) for pid in ids}
    outputs = {pid: _pipe_outputs(plan, pid) for pid in ids}
    headers = {
        _key(edge): ("head", pid, index)
        for pid, edges in outputs.items()
        for index, edge in enumerate(edges)
    }
    frames = {
        _key(edge): ("write", pid, index)
        for pid, edges in outputs.items()
        for index, edge in enumerate(edges)
    }
    concurrent = {
        process.id for process in plan.processes if isinstance(process, SidecarProcess)
    }

    waits: dict[Milestone, tuple[Milestone, ...]] = {}
    for pid in ids:
        queued: list[Milestone] = []
        for index, edge in enumerate(inputs[pid]):
            found = headers.get(_key(edge))
            # A sidecar reads every input at once, so its opens do not queue
            # behind each other; an ffmpeg's do.
            behind = () if pid in concurrent else tuple(queued)
            waits[("open", pid, index)] = behind if found is None else (*behind, found)
            queued.append(("open", pid, index))
        waits[("run", pid, 0)] = tuple(queued)

        # Nothing leaves a process before it is reading every input and every
        # one of them carries frames: an output header names what the first
        # frame through the graph settled.
        fed = tuple(frames[_key(e)] for e in inputs[pid] if _key(e) in frames)
        held: Milestone | None = None
        for index, edge in enumerate(outputs[pid]):
            waits[("head", pid, index)] = (("run", pid, 0), *fed)
            after: list[Milestone] = [("head", pid, index)]
            if index:
                after.append(("write", pid, index - 1))
            if held is not None:
                after.append(held)
            waits[("write", pid, index)] = tuple(after)
            held = None if _carried(plan, edge) else ("run", edge.target, 0)
    return waits


def stalled(plan: ProcessPlan) -> tuple[Milestone, ...] | None:
    """The milestones of `plan` that wait on each other, or None.

    A cycle here is a plan that cannot start: every step in it is waiting for
    the next, and the last for the first.
    """
    return _cycle(relation(plan))


def arrange(plan: ProcessPlan) -> ProcessPlan:
    """`plan` with each process's pipe ``-i`` and output order decided.

    The order comes from a forward walk of the plan rather than from the
    numbering its nodes happened to get, so two spellings of one query -- a
    reordered SELECT list, a CTE -- arrive at the same order.

    An output order is what the walk really chooses: a producer reaches its
    second output only once the first is being taken, so the output feeding
    the longest way round to a consumer comes before the one feeding that
    consumer directly. Each ``-i`` list then follows the order the headers it
    waits for are written in, so no process sits in an open that cannot yet
    return.

    Only an ffmpeg process's frame pipes move. A sidecar reads its inputs at
    once and writes one stream, and a rows track is the head of the ``-i``
    list it arrives on, so neither has an order to choose. A plan no walk
    finishes comes back unchanged, for :func:`check` to word.
    """
    order = _walk(plan)
    if order is None:
        return plan
    return _reordered(plan, order)


def check(plan: ProcessPlan) -> None:
    """Refuse a plan whose processes wait on each other to start.

    Unanchored, for the caller to re-anchor on the declaration that put a
    module in the query -- the same contract
    :func:`~ffrwd.processes.check_spellable` follows.
    """
    cycle = stalled(plan)
    if cycle is None:
        return
    raise FfrwdError(ErrorCode.STARTUP_DEADLOCK, _wording(plan, cycle), hint=_HINT)


_HINT = (
    "each of those processes is waiting for the one after it before it can "
    "read or write anything: give the query one fewer place where the streams "
    "off a single input split apart and come back together, or record the "
    "input to a file and run the query over the file"
)


def _wording(plan: ProcessPlan, cycle: Sequence[Milestone]) -> str:
    """The cycle as a sentence: who waits on whom, in the order they wait."""
    return "these processes cannot start, each waiting on the next: " + ", ".join(
        _phrase(plan, milestone) for milestone in cycle
    ) + f", and {_names(plan, cycle[0][1])} again"


def _phrase(plan: ProcessPlan, milestone: Milestone) -> str:
    kind, pid, index = milestone
    named = _names(plan, pid)
    if kind == "run":
        return f"{named} waits for every input it reads"
    if kind == "open":
        return f"{named} waits to open input {index}"
    return f"{named} waits to write output {index}"


def _names(plan: ProcessPlan, pid: str) -> str:
    """A process as a message names it: its id, plus the module it hosts."""
    process = next((p for p in plan.processes if p.id == pid), None)
    if isinstance(process, SidecarProcess):
        return f"{pid} (the module '{process.module}')"
    return pid


# ---------------------------------------------------------------- the walk


def _walk(plan: ProcessPlan) -> list[Edge] | None:
    """Every pipe edge in an order that starts, or None if none does.

    A process writes its output headers once every pipe it reads carries
    frames, and writes the FRAMES of one output at a time -- reaching the
    next only when the one before it is carried or its consumer is reading
    everything. Which output comes next is the one thing chosen here, and the
    search tries each process's outputs in the order the plan already has
    them, so a plan that starts as it stands keeps that order.
    """
    ids = [process.id for process in plan.processes]
    inputs = {pid: [_key(e) for e in _pipe_inputs(plan, pid)] for pid in ids}
    outputs = {pid: [_key(e) for e in _pipe_outputs(plan, pid)] for pid in ids}
    # A sidecar's own order is not this pass's to change: it reads every input
    # at once and writes one stream, and a rows track's place is its reader's
    # first input either way.
    fixed = {
        process.id for process in plan.processes if isinstance(process, SidecarProcess)
    }
    edge_of = {_key(edge): edge for pid in ids for edge in _pipe_outputs(plan, pid)}
    carried = {key: _carried(plan, edge_of[key]) for key in edge_of}
    target = {key: edge_of[key].target for key in edge_of}
    producer = {key: pid for pid, keys in outputs.items() for key in keys}
    total = len(edge_of)

    def flowing(done: frozenset[Wire], pid: str) -> bool:
        """True once every pipe `pid` reads is carrying frames."""
        return all(key in done for key in inputs[pid])

    def reading(done: frozenset[Wire], pid: str) -> bool:
        """True once `pid` has opened every pipe it reads, and so drains them.

        Its last open returns when the header does, and a header is written
        by a process whose own inputs are already carrying frames.
        """
        return all(flowing(done, producer[key]) for key in inputs[pid])

    seen: set[tuple[object, ...]] = set()
    budget = [_STATES]

    def descend(
        done: frozenset[Wire], held: dict[str, Wire | None]
    ) -> list[Wire] | None:
        if len(done) == total:
            return []
        state = (done, tuple(sorted(k for k, v in held.items() if v)), tuple(sorted(
            v for v in held.values() if v
        )))
        if state in seen or budget[0] <= 0:
            return None
        seen.add(state)
        budget[0] -= 1
        for pid in ids:
            if not flowing(done, pid):
                continue
            waiting = held.get(pid)
            if waiting is not None and not reading(done, target[waiting]):
                continue
            remaining = [key for key in outputs[pid] if key not in done]
            if not remaining:
                continue
            # Two outputs to ONE consumer are interchangeable, and swapping
            # them would only take the plan's edges out of step with the
            # inputs they fill: the plan-earliest of each such pair is tried.
            earliest = {target[key]: key for key in reversed(remaining)}
            for key in remaining[:1] if pid in fixed else remaining:
                if earliest[target[key]] != key:
                    continue
                onward = descend(
                    done | {key}, {**held, pid: None if carried[key] else key}
                )
                if onward is not None:
                    return [key, *onward]
        return None

    found = descend(frozenset(), dict.fromkeys(ids, None))
    return None if found is None else [edge_of[key] for key in found]


def _reordered(plan: ProcessPlan, order: Sequence[Edge]) -> ProcessPlan:
    """`plan` with its edges and every ffmpeg's pipes in the walk's order."""
    when = {_key(edge): index for index, edge in enumerate(order)}
    streams = sorted(plan.stream_edges, key=lambda edge: when[_key(edge)])
    moved = iter(streams)
    edges = tuple(
        next(moved) if isinstance(edge, StreamEdge) else edge for edge in plan.edges
    )
    after = ProcessPlan(processes=plan.processes, edges=edges)
    return ProcessPlan(
        processes=tuple(
            _respell(plan, after, process)
            if isinstance(process, FfmpegProcess)
            else process
            for process in plan.processes
        ),
        edges=edges,
    )


def _respell(
    before: ProcessPlan, after: ProcessPlan, process: FfmpegProcess
) -> FfmpegProcess:
    """`process` with its frame pipes where the reordered edges now say.

    A frame pipe is a slot at the TAIL of the ``-i`` list and a sink at the
    tail of the sink list -- the rows tracks and the files this process opens
    itself come first, and neither moves. Permuting the tail leaves every
    other input index, and so every ``-map``, ``-map_chapters`` and
    ``-map_metadata`` pointing at one, where it was.
    """
    graph = process.graph
    was_read = _once_per_ref(_reads(before, process.id))
    now_read = _once_per_ref(_reads(after, process.id))
    was_written = _writes(before, process.id)
    now_written = _writes(after, process.id)

    slots = [index for index, path in enumerate(graph.input_paths) if path == PIPE]
    frames = slots[len(slots) - len(was_read) :] if was_read else []
    at = {_key(edge): slot for edge, slot in zip(was_read, frames)}
    taken = set(at.values())
    alias_at = {
        slot: alias for alias, slot in graph.sources.items() if slot in taken
    }
    sources = dict(graph.sources)
    for slot, edge in zip(frames, now_read):
        alias = alias_at.get(at[_key(edge)])
        if alias is not None:
            sources[alias] = slot

    units = list(graph.sinks)
    pipes = [index for index, unit in enumerate(units) if unit.path == PIPE]
    tail = pipes[len(pipes) - len(was_written) :] if was_written else []
    held = {_key(edge): units[slot] for edge, slot in zip(was_written, tail)}
    for slot, edge in zip(tail, now_written):
        units[slot] = held[_key(edge)]

    return replace(process, graph=replace(graph, sources=sources, sinks=units))


def _reads(plan: ProcessPlan, pid: str) -> list[StreamEdge]:
    return [e for e in plan.stream_edges if e.target == pid]


def _writes(plan: ProcessPlan, pid: str) -> list[StreamEdge]:
    return [e for e in plan.stream_edges if e.source == pid]


def _once_per_ref(edges: Sequence[StreamEdge]) -> list[StreamEdge]:
    """One edge per ref: two edges of one ref share a single ``-i``."""
    seen: set[str] = set()
    kept: list[StreamEdge] = []
    for edge in edges:
        if edge.ref in seen:
            continue
        seen.add(edge.ref)
        kept.append(edge)
    return kept


# ---------------------------------------------------------------- reading a plan


def _pipe_inputs(plan: ProcessPlan, pid: str) -> list[Edge]:
    """What `pid` opens, in ``-i`` order: rows tracks first, then frames."""
    rows: list[Edge] = [e for e in plan.rows_edges if e.target == pid]
    frames: list[Edge] = list(
        _once_per_ref([e for e in plan.stream_edges if e.target == pid])
    )
    return rows + frames


def _pipe_outputs(plan: ProcessPlan, pid: str) -> list[Edge]:
    """What `pid` writes, in the order its outputs are rendered."""
    frames: list[Edge] = [e for e in plan.stream_edges if e.source == pid]
    rows: list[Edge] = [e for e in plan.rows_edges if e.source == pid]
    return frames + rows


def _carried(plan: ProcessPlan, edge: Edge) -> bool:
    """True when the producer may write past this edge and go on.

    Only an edge into a SIDECAR: it reads every pipe handed to it from the
    moment it exists -- a reader per input, before any module opens -- so
    nothing writing to one ever waits.

    A depth the compiler counted does NOT carry an edge. The executor's copy
    drains it early, but into a finite buffer that a live source fills while
    the consumer is still starting; then the producer stops exactly where it
    would have. Slack, not an exemption.
    """
    return isinstance(plan.process(edge.target), SidecarProcess)


def _key(edge: Edge) -> Wire:
    """An edge's identity, stable across the copies `replace` makes."""
    if isinstance(edge, RowsEdge):
        return ("rows", edge.source, edge.target, edge.alias)
    if isinstance(edge, StreamEdge):
        return ("stream", edge.source, edge.target, edge.ref)
    return ("file", edge.source, edge.target)


def _cycle(
    waits: Mapping[Milestone, Sequence[Milestone]],
) -> tuple[Milestone, ...] | None:
    """One cycle of `waits`, in waiting order, or None if it is acyclic."""
    colour: dict[Milestone, int] = {}
    path: list[Milestone] = []

    def walk(node: Milestone) -> tuple[Milestone, ...] | None:
        colour[node] = 1
        path.append(node)
        for nxt in waits.get(node, ()):
            state = colour.get(nxt, 0)
            if state == 1:
                return tuple(path[path.index(nxt) :])
            if state == 0:
                found = walk(nxt)
                if found is not None:
                    return found
        path.pop()
        colour[node] = 2
        return None

    for node in waits:
        if colour.get(node, 0) == 0:
            found = walk(node)
            if found is not None:
                return found
    return None
