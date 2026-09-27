"""Which node each process of a plan runs on, and the edges between nodes.

A :class:`~ffrwd.processes.ProcessPlan` says nothing about machines. One
node runs every process today; a placement maps each process to a node
instead, and :func:`split` turns the placed plan into what each node runs
(:class:`NodePlan`): its own processes, and the stream edges it shares with
another node (:class:`Cut`), which travel over TCP between the two nodes'
runners. The consumer's node listens and the producer's node dials.

Some processes cannot be parted (:func:`colocation_groups`):

- a feeder's writer and the module listening on its loopback port, and so
  every module reading one port (the switch's pair meets on the port above
  its own), and the processes a run-time lateral's instances feed, which run
  beside the writer of the lateral's data stream;
- the one reader of a live input, which is a single process by construction
  and is pinned where the input can be opened.

A stdio chain (one process's stdout handed to the next one's stdin) is not
among them: the runner already copies a stdio edge through itself where it
has to see the bytes, and a chain cut between two nodes is copied the same
way, stdout to TCP to stdin, with no argv changed. The ``one`` placement
keeps every chain whole, since it keeps everything on one node.

Strategies:

- ``one``: every process on node 0.
- ``per-module``: each sidecar region on a node of its own, each ffmpeg
  process on the node of the first region it feeds, else the first region
  feeding it, else the node of an ffmpeg process it is wired to, else node
  0. A node holding a region that computes on a GPU (a ``gpu`` grant, or a
  model bound with ``-nn``) is marked as needing one.
- ``per-process``: every process on a node of its own, the groups above
  kept whole. Not a product strategy: it cuts every edge it can, which is
  what a test of the cut edges wants.

Node 0 is the node of the plan's first process.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from .errors import ErrorCode, FfrwdError
from .ir import is_rows_document
from .processes import (
    FeederEdge,
    FfmpegProcess,
    FileEdge,
    ProcessPlan,
    RowsEdge,
    SidecarProcess,
    StreamEdge,
    is_live,
)

__all__ = [
    "STRATEGIES",
    "Cut",
    "Group",
    "NodePlan",
    "Placement",
    "Strategy",
    "check_placement",
    "colocation_groups",
    "cut_key",
    "pipe_edges",
    "place",
    "split",
    "stdio_chains",
]

Strategy = Literal["one", "per-module", "per-process"]
STRATEGIES: tuple[Strategy, ...] = ("one", "per-module", "per-process")

# The effect grant that puts a region on a GPU.
_GPU = "gpu"

# Why a group cannot be parted, as a refusal names it.
FEEDER = "feeder connection"
LIVE_READER = "live input's one reader"


@dataclass(frozen=True)
class Group:
    """Processes that run on one node, and why they must."""

    members: tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class Placement:
    """The node each process runs on. `gpu` names the nodes needing a GPU."""

    nodes: Mapping[str, int]
    gpu: frozenset[int] = frozenset()

    @property
    def count(self) -> int:
        return max(self.nodes.values(), default=-1) + 1

    def node(self, process: str) -> int:
        return self.nodes[process]


@dataclass(frozen=True)
class Cut:
    """One pipe edge between processes on two nodes.

    `edge` is the edge's position among the plan's pipe edges
    (:func:`pipe_edges`), and `key` the name both nodes know it by. The
    `consumer` node listens for it and the `producer` node dials.
    """

    edge: int
    key: str
    source: str
    target: str
    carried: str
    producer: int
    consumer: int

    def to_dict(self) -> dict[str, object]:
        return {
            "edge": self.edge,
            "key": self.key,
            "source": self.source,
            "target": self.target,
            "carried": self.carried,
            "producer": self.producer,
            "consumer": self.consumer,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Cut:
        edge, producer, consumer = d["edge"], d["producer"], d["consumer"]
        if not (
            isinstance(edge, int) and isinstance(producer, int) and isinstance(consumer, int)
        ):
            raise ValueError("a cut edge's 'edge', 'producer' and 'consumer' are whole numbers")
        return cls(
            edge=edge,
            key=str(d["key"]),
            source=str(d["source"]),
            target=str(d["target"]),
            carried=str(d["carried"]),
            producer=producer,
            consumer=consumer,
        )


@dataclass(frozen=True)
class NodePlan:
    """What one node runs: its processes, and the cut edges at its border.

    `processes` are in plan order. `listens` are the cut edges into this
    node's processes, `dials` the ones out of them.
    """

    node: int
    processes: tuple[str, ...]
    gpu: bool = False
    listens: tuple[Cut, ...] = ()
    dials: tuple[Cut, ...] = ()

    def view(self, plan: ProcessPlan) -> ProcessPlan:
        """`plan` as this node sees it: its processes, and every edge
        touching one of them (a cut edge among them)."""
        mine = set(self.processes)
        return ProcessPlan(
            processes=tuple(p for p in plan.processes if p.id in mine),
            edges=tuple(e for e in plan.edges if e.source in mine or e.target in mine),
            laterals=tuple(one for one in plan.laterals if one.writer in mine),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "node": self.node,
            "processes": list(self.processes),
            "gpu": self.gpu,
            "listens": [cut.to_dict() for cut in self.listens],
            "dials": [cut.to_dict() for cut in self.dials],
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> NodePlan:
        node, processes = d["node"], d["processes"]
        listens, dials = d.get("listens", []), d.get("dials", [])
        if not isinstance(node, int) or not isinstance(processes, list):
            raise ValueError("a node plan names its node and lists its processes")
        if not isinstance(listens, list) or not isinstance(dials, list):
            raise ValueError("a node plan's cut edges are lists")
        return cls(
            node=node,
            processes=tuple(str(pid) for pid in processes),
            gpu=d.get("gpu") is True,
            listens=tuple(Cut.from_dict(one) for one in listens if isinstance(one, dict)),
            dials=tuple(Cut.from_dict(one) for one in dials if isinstance(one, dict)),
        )


# -- what the plan implies


def pipe_edges(plan: ProcessPlan) -> tuple[StreamEdge | RowsEdge, ...]:
    """The plan's edges that run over a pipe, in the order the runner pairs
    them with pipe slots: rows edges ahead of stream edges."""
    return (*plan.rows_edges, *plan.stream_edges)


def cut_key(index: int) -> str:
    """The name a cut edge at this position among the pipe edges goes by."""
    return f"e{index}"


class _Union:
    def __init__(self, names: Sequence[str]) -> None:
        self._parent = {name: name for name in names}

    def root(self, name: str) -> str:
        while self._parent[name] != name:
            self._parent[name] = self._parent[self._parent[name]]
            name = self._parent[name]
        return name

    def holds(self, name: str) -> bool:
        return name in self._parent

    def join(self, left: str, right: str) -> None:
        a, b = self.root(left), self.root(right)
        if a != b:
            self._parent[b] = a


def live_readers(plan: ProcessPlan) -> tuple[str, ...]:
    """The processes that open a live input: each reads it alone.

    An ffmpeg whose own inputs include one :func:`~ffrwd.processes.is_live`
    names, and the producer of any edge the partitioner marked as a live
    reader's (which also covers a manifest only its probe knew was live).
    """
    found: list[str] = []
    marked = {edge.source for edge in plan.stream_edges if edge.live}
    for process in plan.processes:
        if not isinstance(process, FfmpegProcess):
            continue
        graph = process.graph
        opens = any(
            is_live(graph.input_paths[index], graph.input_options.get(alias))
            for alias, index in graph.sources.items()
            if index < len(graph.input_paths)
        )
        if opens or process.id in marked:
            found.append(process.id)
    return tuple(found)


def colocation_groups(plan: ProcessPlan) -> tuple[Group, ...]:
    """The groups of processes no placement may part, in plan order.

    One group per feeder connection's processes (merged where they share a
    process or a port), and one per live input's reader that no feeder group
    already holds.
    """
    ids = [p.id for p in plan.processes]
    union = _Union(ids)
    joined: set[str] = set()
    by_port: dict[int, list[str]] = {}
    for edge in plan.feeder_edges:
        if union.holds(edge.source) and union.holds(edge.target):
            union.join(edge.source, edge.target)
            joined.update((edge.source, edge.target))
            by_port.setdefault(edge.port, []).append(edge.target)
    for readers in by_port.values():
        for reader in readers[1:]:
            union.join(readers[0], reader)
    members: dict[str, list[str]] = {}
    for pid in ids:
        if pid in joined:
            members.setdefault(union.root(pid), []).append(pid)
    groups = [Group(tuple(found), FEEDER) for found in members.values()]
    groups += [Group((pid,), LIVE_READER) for pid in live_readers(plan) if pid not in joined]
    order = {pid: index for index, pid in enumerate(ids)}
    return tuple(sorted(groups, key=lambda group: order[group.members[0]]))


def stdio_chains(plan: ProcessPlan) -> tuple[tuple[str, ...], ...]:
    """Each run of processes one stdout hands the next one's stdin, in order.

    Not a group: a chain is cut by copying the edge through the runners
    (see the module docstring). Listed so a placement can say what it cut.
    """
    from .execute import wires

    chained = [wire.edge for wire in wires(plan) if wire.chained]
    after = {edge.source: edge.target for edge in chained}
    fed = {edge.target for edge in chained}
    runs: list[tuple[str, ...]] = []
    for process in plan.processes:
        if process.id in after and process.id not in fed:
            run = [process.id]
            while run[-1] in after:
                run.append(after[run[-1]])
            runs.append(tuple(run))
    return tuple(runs)


def _needs_gpu(process: SidecarProcess) -> bool:
    return bool(process.models) or any(grant.effect == _GPU for grant in process.grants)


# -- strategies


def place(plan: ProcessPlan, strategy: Strategy = "one") -> Placement:
    """`plan` placed by `strategy`, every co-location group kept whole."""
    ids = [p.id for p in plan.processes]
    union = _Union(ids)
    if strategy == "one":
        for pid in ids[1:]:
            union.join(ids[0], pid)
    elif strategy == "per-module":
        _attach_ffmpeg(plan, union)
    elif strategy != "per-process":
        raise FfrwdError(
            ErrorCode.PLACEMENT_REFUSED,
            f"no placement is called '{strategy}'",
            hint=f"use one of: {', '.join(STRATEGIES)}",
        )
    for group in colocation_groups(plan):
        for pid in group.members[1:]:
            union.join(group.members[0], pid)
    numbered: dict[str, int] = {}
    nodes: dict[str, int] = {}
    for pid in ids:
        root = union.root(pid)
        nodes[pid] = numbered.setdefault(root, len(numbered))
    gpu = frozenset(
        nodes[p.id] for p in plan.processes if isinstance(p, SidecarProcess) and _needs_gpu(p)
    )
    return Placement(nodes=nodes, gpu=gpu)


def _attach_ffmpeg(plan: ProcessPlan, union: _Union) -> None:
    """Put each ffmpeg process with the region it feeds (over a pipe or a
    feeder connection), else the region feeding it, else an ffmpeg it is
    wired to; one wired to nothing stays on a node of its own."""
    kinds = {p.id: isinstance(p, SidecarProcess) for p in plan.processes}
    edges = [e for e in plan.edges if isinstance(e, StreamEdge | RowsEdge | FeederEdge)]
    for process in plan.ffmpeg:
        feeds = [e.target for e in edges if e.source == process.id and kinds.get(e.target)]
        fed = [e.source for e in edges if e.target == process.id and kinds.get(e.source)]
        if feeds or fed:
            union.join((feeds or fed)[0], process.id)
    for process in plan.ffmpeg:
        if any(
            kinds.get(e.target if e.source == process.id else e.source)
            for e in edges
            if process.id in (e.source, e.target)
        ):
            continue
        wired = [
            e.target if e.source == process.id else e.source
            for e in edges
            if process.id in (e.source, e.target)
        ]
        if wired:
            union.join(wired[0], process.id)


# -- refusals


def check_placement(
    plan: ProcessPlan, placement: Placement, *, shown: Sequence[str] = ()
) -> None:
    """Refuse a placement this runner cannot carry out, naming what is wrong.

    - Every process placed.
    - No co-location group parted.
    - No file one process hands another across two nodes: a later stage's
      input, a rows document, a two-pass log. Nothing carries files between
      nodes, so the reader would not find it.
    - No display window (`shown` names the processes with one) on a node
      other than node 0, the one on this machine's screen.
    """
    missing = [p.id for p in plan.processes if p.id not in placement.nodes]
    if missing:
        raise FfrwdError(
            ErrorCode.PLACEMENT_REFUSED,
            f"the placement puts {', '.join(missing)} on no node",
            hint="place every process of the plan",
        )
    for group in colocation_groups(plan):
        nodes = {placement.node(pid) for pid in group.members}
        if len(nodes) > 1:
            where = ", ".join(f"{pid} on node {placement.node(pid)}" for pid in group.members)
            raise FfrwdError(
                ErrorCode.PLACEMENT_REFUSED,
                f"the placement splits a {group.reason}: {where}",
                hint="a feeder's writer and the modules listening on its loopback "
                "port run on one machine; place them on one node",
            )
    for edge in plan.file_edges:
        _check_file_edge(plan, placement, edge)
    _check_rows_documents(plan, placement)
    away = [pid for pid in shown if placement.node(pid) != 0]
    if away:
        raise FfrwdError(
            ErrorCode.PLACEMENT_REFUSED,
            f"{', '.join(away)} would open a display window on node "
            f"{placement.node(away[0])}, which is not this machine",
            hint="drop --show, or place the shown processes on node 0",
        )


def _check_file_edge(plan: ProcessPlan, placement: Placement, edge: FileEdge) -> None:
    producer, consumer = placement.node(edge.source), placement.node(edge.target)
    if producer == consumer:
        return
    what = f"'{edge.format.path}'" if edge.format.path else "a file"
    stages = len(plan.stages)
    raise FfrwdError(
        ErrorCode.PLACEMENT_REFUSED,
        f"this plan runs in {stages} stage{'s' if stages != 1 else ''}, and "
        f"{edge.source} on node {producer} hands {what} ({edge.format.content}) to "
        f"{edge.target} on node {consumer}: nothing carries a file between nodes",
        hint="place the two processes on one node, or run the plan on one node",
    )


def _check_rows_documents(plan: ProcessPlan, placement: Placement) -> None:
    """A rows document a packet filter reads by path, written on another node."""
    written: dict[str, str] = {}
    for process in plan.sidecars:
        for document in process.rows:
            if document.sink.path and is_rows_document(document.sink.path):
                written.setdefault(document.sink.path, process.id)
    for process in plan.sidecars:
        for read in process.rows_in:
            writer = written.get(read.path)
            if writer is None or placement.node(writer) == placement.node(process.id):
                continue
            raise FfrwdError(
                ErrorCode.PLACEMENT_REFUSED,
                f"{process.id} on node {placement.node(process.id)} reads the rows "
                f"document {writer} writes on node {placement.node(writer)}: nothing "
                "carries a file between nodes",
                hint="place the two processes on one node, or run the plan on one node",
            )


# -- the split


def split(plan: ProcessPlan, placement: Placement) -> tuple[NodePlan, ...]:
    """What each node of `placement` runs, node 0 first, once
    :func:`check_placement` has passed it."""
    cuts: list[Cut] = []
    for index, edge in enumerate(pipe_edges(plan)):
        producer, consumer = placement.node(edge.source), placement.node(edge.target)
        if producer == consumer:
            continue
        carried = edge.ref if isinstance(edge, StreamEdge) else edge.alias
        cuts.append(
            Cut(
                edge=index,
                key=cut_key(index),
                source=edge.source,
                target=edge.target,
                carried=carried,
                producer=producer,
                consumer=consumer,
            )
        )
    nodes: list[NodePlan] = []
    for node in range(placement.count):
        nodes.append(
            NodePlan(
                node=node,
                processes=tuple(p.id for p in plan.processes if placement.node(p.id) == node),
                gpu=node in placement.gpu,
                listens=tuple(cut for cut in cuts if cut.consumer == node),
                dials=tuple(cut for cut in cuts if cut.producer == node),
            )
        )
    return tuple(nodes)

