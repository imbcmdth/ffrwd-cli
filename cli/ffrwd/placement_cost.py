"""Placing a plan by cost: fewest links, GPU jobs capped. Experimental.

The strategies of :mod:`ffrwd.placement` part a plan by what its processes
are (``by-hardware``: each connected run of one class on a node of its own).
These part it by what it costs to run: each process's load in cores and in
GPU jobs, each edge's bandwidth, and a node's capacity. A plan that fits one
node runs on one node, with no links at all; one that does not is split
where the split costs least.

Nothing here is a default. A strategy is a set of parameters
(:class:`CostStrategy`), spelled ``cost:gpu_jobs=2,link=50,search=refine``
or by a preset's name, and :func:`report` places one plan under several and
says how each came out, so the default can be chosen from numbers.

What the search may not do is the same as for every strategy: the groups of
:func:`ffrwd.placement.colocation_groups` stay whole and a file handoff stays
on one node, so the search runs over the groups those leave (:func:`groups`),
and every placement it makes passes :func:`ffrwd.placement.check_placement`.

The load model is an estimate, and a coarse one (:func:`process_load`):
- an encode by its codec and pixel rate (x264 about two cores for 1080p30);
  an NVIDIA encode is not GPU compute but a session on the card's own
  encoder, counted against its own cap, plus a fifth of a core;
- a hardware decode likewise, against a decode cap;
- a software decode of an input half a core;
- each video filter a fifth of a core at 1080p30, audio next to nothing;
- a wasm region a third of a core per module; a region binding a model, or
  a CUDA filter chain, one GPU job (compute).

The defaults are the owner's (2026-09-28): a node loaded to 60% of its
cores, at most 2 GPU jobs, 2 encodes and 2 hardware decodes, each counted on
its own.

Each process's measured CPU time, from earlier runs of the same plan, is the
estimate's replacement; nothing records that yet.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Literal

from .errors import ErrorCode, FfrwdError
from .placement import (
    Placement,
    check_placement,
    colocation_groups,
    file_handoffs,
    gpu_processes,
    needs_gpu,
)
from .processes import (
    AudioFormat,
    FeederEdge,
    FfmpegProcess,
    ProcessPlan,
    RowsEdge,
    SidecarProcess,
    StreamEdge,
    VideoFormat,
)

__all__ = [
    "PRESETS",
    "CostStrategy",
    "check_strategy_name",
    "Load",
    "cost_place",
    "edge_mbps",
    "groups",
    "parse_strategy",
    "process_load",
    "report",
]

Search = Literal["greedy", "refine", "exhaustive"]

# 1080p30 in pixels a second: the unit the load table is written in.
_REFERENCE_RATE = 1920 * 1080 * 30
# Bits a pixel of raw video carries, by pixel format; anything unlisted is
# taken as 4:2:0 at 8 bits.
_BITS_PER_PIXEL = {"yuv420p": 12, "nv12": 12, "yuv422p": 16, "yuv444p": 24, "rgb24": 24,
                   "rgba": 32, "bgra": 32, "gray": 8}
# An edge whose codec is not raw carries about this much, unless its
# encoder's options say.
_CODED_VIDEO_MBPS = 6.0
_CODED_AUDIO_MBPS = 0.192
_DATA_MBPS = 0.05
# What a coded edge's encoder costs, in cores at 1080p30, by the codec's name.
_ENCODE_CORES = {"libx264": 2.0, "libx265": 4.0, "libsvtav1": 3.0, "libvpx-vp9": 4.0}
_GPU_ENCODE_CORES = 0.2
_DECODE_CORES = 0.5
_FILTER_CORES = 0.2
_AUDIO_CORES = 0.02
_REGION_CORES = 0.33
_MODEL_CORES = 0.5
# The most groups the exhaustive search tries every partition of.
_EXHAUSTIVE_LIMIT = 10
_REFINE_PASSES = 50


@dataclass(frozen=True)
class CostStrategy:
    """One experimental strategy: the parameters of the search.

    - `gpu_jobs`: GPU compute jobs one node may hold (a model, a CUDA filter
      chain).
    - `encodes`: hardware encode sessions (NVENC) one node may hold.
    - `decodes`: hardware decode sessions (NVDEC) one node may hold.
    - `node_cores`: the cores a node has (the container's reservation).
    - `cores`: the share of those a node may be loaded to, leaving headroom.
    - `link`: the fixed cost of one cut edge, in Mbit/s it is worth.
    - `raw_penalty`: the weight on a raw-video edge's bandwidth, which is
      what moves a cut after an encoder.
    - `search`: greedy, greedy then refine, or every partition (small plans).
    - `fps`: the frame rate a video edge is taken to run at.
    """

    name: str = "cost"
    gpu_jobs: int = 2
    encodes: int = 2
    decodes: int = 2
    node_cores: float = 8.0
    cores: float = 0.6
    link: float = 50.0
    raw_penalty: float = 2.0
    search: Search = "refine"
    fps: float = 30.0

    @property
    def capacity(self) -> float:
        return self.node_cores * self.cores


# Named combinations worth comparing; all experimental.
PRESETS: dict[str, CostStrategy] = {
    # The owner's defaults.
    "cost": CostStrategy(name="cost"),
    "cost-lean": CostStrategy(
        name="cost-lean", gpu_jobs=3, encodes=3, decodes=3, cores=0.8, link=200.0
    ),
    "cost-spread": CostStrategy(
        name="cost-spread", gpu_jobs=1, encodes=1, decodes=1, cores=0.5, link=10.0
    ),
}

_NUMBERS = {"gpu_jobs": int, "encodes": int, "decodes": int, "node_cores": float,
            "cores": float, "link": float, "raw_penalty": float, "fps": float}


def check_strategy_name(text: str) -> None:
    """Refuse a placement strategy name that means nothing, before a run."""
    from .placement import STRATEGIES

    if text in STRATEGIES:
        return
    parse_strategy(text)


def parse_strategy(text: str) -> CostStrategy:
    """A preset's name, or ``cost:key=value,...`` over the defaults."""
    if text in PRESETS:
        return PRESETS[text]
    head, _, rest = text.partition(":")
    if head != "cost":
        raise FfrwdError(
            ErrorCode.PLACEMENT_REFUSED,
            f"no cost strategy is called '{text}'",
            hint=f"use cost:key=value,... or one of: {', '.join(PRESETS)}",
        )
    strategy = CostStrategy(name=text)
    for pair in filter(None, rest.split(",")):
        key, _, value = pair.partition("=")
        key = key.strip()
        if key == "search":
            if value not in ("greedy", "refine", "exhaustive"):
                raise FfrwdError(
                    ErrorCode.PLACEMENT_REFUSED,
                    f"search={value} is not greedy, refine or exhaustive",
                )
            strategy = replace(strategy, search=value)  # type: ignore[arg-type]
        elif key in _NUMBERS:
            try:
                strategy = replace(strategy, **{key: _NUMBERS[key](value)})
            except ValueError as err:
                raise FfrwdError(
                    ErrorCode.PLACEMENT_REFUSED, f"{key}={value} is not a number"
                ) from err
        else:
            raise FfrwdError(
                ErrorCode.PLACEMENT_REFUSED,
                f"a cost strategy has no parameter '{key}'",
                hint=f"its parameters: search, {', '.join(_NUMBERS)}",
            )
    return strategy


# -- the cost model


@dataclass(frozen=True)
class Load:
    """What running one process, or one group, takes: cores, GPU compute
    jobs, and hardware encode and decode sessions."""

    cores: float = 0.0
    gpu_jobs: int = 0
    encodes: int = 0
    decodes: int = 0

    def __add__(self, other: Load) -> Load:
        return Load(
            self.cores + other.cores,
            self.gpu_jobs + other.gpu_jobs,
            self.encodes + other.encodes,
            self.decodes + other.decodes,
        )

    @property
    def on_gpu(self) -> bool:
        """Whether this needs a node with a GPU at all."""
        return bool(self.gpu_jobs or self.encodes or self.decodes)


def _pixel_scale(format: VideoFormat, fps: float) -> float:
    """A video format's pixel rate against 1080p30; an unknown size is 1080p."""
    width = format.width or 1920
    height = format.height or 1080
    return (width * height * fps) / _REFERENCE_RATE


def _is_raw(format: object) -> bool:
    return isinstance(format, VideoFormat) and format.codec == "rawvideo"


def edge_mbps(edge: object, fps: float = 30.0) -> float:
    """The megabits a second `edge` carries, from its format."""
    if isinstance(edge, StreamEdge):
        format = edge.format
        if isinstance(format, VideoFormat):
            if format.codec == "rawvideo":
                width = format.width or 1920
                height = format.height or 1080
                bits = _BITS_PER_PIXEL.get(format.pix_fmt, 12)
                return width * height * bits * fps / 1e6
            options = dict(format.options)
            said = options.get("video_bitrate") or options.get("b:v")
            return _bitrate_mbps(said) if said else _CODED_VIDEO_MBPS
        if isinstance(format, AudioFormat):
            if format.codec.startswith("pcm_"):
                rate = format.rate or 48000
                channels = format.channels or 2
                bits = 32 if format.codec.endswith(("f32le", "s32le")) else 16
                return rate * channels * bits / 1e6
            return _CODED_AUDIO_MBPS
        return _DATA_MBPS
    if isinstance(edge, RowsEdge | FeederEdge):
        return _DATA_MBPS
    return 0.0


def _bitrate_mbps(value: object) -> float:
    """``2000k``, ``6M`` or a number of bits, in Mbit/s."""
    text = str(value).strip().lower()
    scale = {"k": 1e-3, "m": 1.0, "g": 1e3}.get(text[-1:], 1e-6)
    try:
        number = float(text[:-1] if text[-1:] in "kmg" else text)
    except ValueError:
        return _CODED_VIDEO_MBPS
    return number * scale


def process_load(process: object, plan: ProcessPlan, fps: float = 30.0) -> Load:
    """The estimated load of one process (see the module's table)."""
    if isinstance(process, SidecarProcess):
        modules = max(1, len(process.modules) or 1)
        cores = _REGION_CORES * modules
        if needs_gpu(process):
            # A model or a gpu grant: compute on the card.
            return Load(cores + (_MODEL_CORES if process.models else 0.0), gpu_jobs=1)
        return Load(cores)
    if not isinstance(process, FfmpegProcess):
        return Load()
    cores = 0.0
    jobs = 0
    encodes = 0
    decodes = 0
    graph = process.graph
    for options in graph.input_options.values():
        if str(options.get("hwaccel", "")).lower() in ("cuda", "cuvid", "nvdec"):
            decodes += 1
    # Decoding what it opens itself (a file, a url, a device), not its pipes.
    opened = [path for path in graph.input_paths if not path.startswith("pipe:")]
    cores += _DECODE_CORES * len(opened)
    for node in graph.nodes.values():
        if node.filter.endswith(("_cuda", "_npp")):
            jobs = 1  # a CUDA filter chain is one compute job, however long
            continue
        video = any(kind == "video" for kind in node.outputs)
        cores += _FILTER_CORES if video else _AUDIO_CORES
    for edge in plan.stream_edges:
        if edge.source != process.id:
            continue
        format = edge.format
        if isinstance(format, VideoFormat) and format.codec != "rawvideo":
            scale = _pixel_scale(format, fps)
            if format.codec.endswith(("_nvenc", "_qsv", "_vaapi")):
                encodes += 1
                cores += _GPU_ENCODE_CORES
            else:
                cores += _ENCODE_CORES.get(format.codec, 1.0) * scale
        elif isinstance(format, AudioFormat) and not format.codec.startswith("pcm_"):
            cores += _AUDIO_CORES
    for unit in graph.sinks:
        codec = unit.options.get("video_codec")
        if isinstance(codec, str) and codec:
            if codec.endswith(("_nvenc", "_qsv", "_vaapi")):
                encodes += 1
                cores += _GPU_ENCODE_CORES
            else:
                cores += _ENCODE_CORES.get(codec, 1.0)
    return Load(round(cores, 3), jobs, encodes, decodes)


# -- the groups the hard rules leave


def groups(plan: ProcessPlan) -> list[tuple[str, ...]]:
    """The units the search moves: every co-location group whole, and the two
    ends of a file handoff together, merged where they share a process; every
    other process alone. In plan order."""
    ids = [p.id for p in plan.processes]
    parent = {pid: pid for pid in ids}

    def root(pid: str) -> str:
        while parent[pid] != pid:
            parent[pid] = parent[parent[pid]]
            pid = parent[pid]
        return pid

    def join(a: str, b: str) -> None:
        ra, rb = root(a), root(b)
        if ra != rb:
            parent[rb] = ra

    for group in colocation_groups(plan):
        for pid in group.members[1:]:
            join(group.members[0], pid)
    for writer, reader in file_handoffs(plan):
        join(writer, reader)
    found: dict[str, list[str]] = {}
    for pid in ids:
        found.setdefault(root(pid), []).append(pid)
    return [tuple(members) for members in found.values()]


# -- the search


@dataclass
class _Problem:
    units: list[tuple[str, ...]]
    load: list[Load]
    # (unit a, unit b) -> the cost of cutting every edge between them
    weight: dict[tuple[int, int], float]
    strategy: CostStrategy

    def fits(self, total: Load) -> bool:
        return _fits(total, self.strategy)

    def cost(self, assign: Sequence[int]) -> float:
        return sum(w for (a, b), w in self.weight.items() if assign[a] != assign[b])

    def totals(self, assign: Sequence[int]) -> dict[int, Load]:
        found: dict[int, Load] = {}
        for unit, node in enumerate(assign):
            found[node] = found.get(node, Load()) + self.load[unit]
        return found

    def feasible(self, assign: Sequence[int]) -> bool:
        return all(self.fits(total) for total in self.totals(assign).values())


def _fits(load: Load, strategy: CostStrategy) -> bool:
    return (
        load.cores <= strategy.capacity + 1e-9
        and load.gpu_jobs <= strategy.gpu_jobs
        and load.encodes <= strategy.encodes
        and load.decodes <= strategy.decodes
    )


def _problem(plan: ProcessPlan, strategy: CostStrategy) -> _Problem:
    units = groups(plan)
    unit_of = {pid: index for index, members in enumerate(units) for pid in members}
    loads = [
        sum((process_load(plan.process(pid), plan, strategy.fps) for pid in members), Load())
        for members in units
    ]
    weight: dict[tuple[int, int], float] = {}
    for edge in plan.edges:
        if not isinstance(edge, StreamEdge | RowsEdge | FeederEdge):
            continue
        a, b = unit_of[edge.source], unit_of[edge.target]
        if a == b:
            continue
        key = (min(a, b), max(a, b))
        mbps = edge_mbps(edge, strategy.fps)
        raw = isinstance(edge, StreamEdge) and _is_raw(edge.format)
        weight[key] = weight.get(key, 0.0) + strategy.link + mbps * (
            strategy.raw_penalty if raw else 1.0
        )
    for index, load in enumerate(loads):
        if not _fits(load, strategy):
            raise FfrwdError(
                ErrorCode.PLACEMENT_REFUSED,
                f"the processes {', '.join(units[index])} must run on one node and need "
                f"{load.cores:.1f} cores, {load.gpu_jobs} GPU job(s), {load.encodes} "
                f"encode(s) and {load.decodes} decode(s), more than one node holds "
                f"({strategy.capacity:.1f} cores, {strategy.gpu_jobs} GPU jobs, "
                f"{strategy.encodes} encodes, {strategy.decodes} decodes)",
                hint="raise gpu_jobs, encodes, decodes, node_cores or cores",
            )
    return _Problem(units, loads, weight, strategy)


def _affinity(problem: _Problem, unit: int, members: Iterable[int]) -> float:
    return sum(
        problem.weight.get((min(unit, other), max(unit, other)), 0.0) for other in members
    )


def _greedy(problem: _Problem) -> list[int]:
    n = len(problem.units)
    if problem.fits(sum(problem.load, Load())):
        return [0] * n
    assign = [-1] * n
    nodes: list[list[int]] = []
    totals: list[Load] = []

    def put(unit: int, node: int | None) -> None:
        if node is None:
            nodes.append([])
            totals.append(Load())
            node = len(nodes) - 1
        nodes[node].append(unit)
        totals[node] = totals[node] + problem.load[unit]
        assign[unit] = node

    # GPU work first (compute, encodes, decodes), packed by branch: each onto
    # the L4 node it is most connected to that has room.
    gpu = sorted(
        (u for u in range(n) if problem.load[u].on_gpu),
        key=lambda u: (
            -(problem.load[u].gpu_jobs + problem.load[u].encodes + problem.load[u].decodes),
            -problem.load[u].cores,
            u,
        ),
    )
    for unit in gpu:
        best, best_affinity = None, -1.0
        for node, members in enumerate(nodes):
            if problem.fits(totals[node] + problem.load[unit]):
                affinity = _affinity(problem, unit, members)
                if affinity > best_affinity:
                    best, best_affinity = node, affinity
        put(unit, best)
    # Everything else, the unit most tied to what is placed first, onto the
    # node it is most tied to that has room.
    rest = [u for u in range(n) if assign[u] < 0]
    while rest:
        rest.sort(
            key=lambda u: (
                -_affinity(problem, u, (v for v in range(n) if assign[v] >= 0)),
                -problem.load[u].cores,
                u,
            )
        )
        unit = rest.pop(0)
        best, best_affinity = None, -1.0
        for node, members in enumerate(nodes):
            if problem.fits(totals[node] + problem.load[unit]):
                affinity = _affinity(problem, unit, members)
                if affinity > best_affinity:
                    best, best_affinity = node, affinity
        put(unit, best)
    return assign


def _refine(problem: _Problem, assign: list[int]) -> list[int]:
    """Move one unit at a time to another node while the cut cost falls."""
    assign = list(assign)
    for _ in range(_REFINE_PASSES):
        improved = False
        for unit in range(len(assign)):
            here = problem.cost(assign)
            for node in sorted(set(assign)):
                if node == assign[unit]:
                    continue
                trial = list(assign)
                trial[unit] = node
                if problem.feasible(trial) and problem.cost(trial) < here - 1e-9:
                    assign, here, improved = trial, problem.cost(trial), True
        if not improved:
            break
    return assign


def _exhaustive(problem: _Problem) -> list[int] | None:
    """Every partition of the units, the cheapest that fits; None when there
    are too many units to try them all."""
    n = len(problem.units)
    if n > _EXHAUSTIVE_LIMIT:
        return None
    best: tuple[float, int, list[int]] | None = None

    def partitions(prefix: list[int], top: int) -> Iterable[list[int]]:
        if len(prefix) == n:
            yield prefix
            return
        for node in range(top + 2):
            yield from partitions(prefix + [node], max(top, node))

    for assign in partitions([0], 0):
        if not problem.feasible(assign):
            continue
        key = (round(problem.cost(assign), 6), len(set(assign)))
        if best is None or key < best[:2]:
            best = (key[0], key[1], assign)
    return best[2] if best else None


def cost_place(plan: ProcessPlan, strategy: CostStrategy) -> Placement:
    """`plan` placed by `strategy`: node 0 is the node of the plan's first
    process, and a node holding GPU work is marked as needing a GPU."""
    problem = _problem(plan, strategy)
    assign = _greedy(problem)
    if strategy.search == "refine":
        assign = _refine(problem, assign)
    elif strategy.search == "exhaustive":
        assign = _exhaustive(problem) or _refine(problem, assign)
    numbered: dict[int, int] = {}
    nodes: dict[str, int] = {}
    unit_of = {pid: index for index, members in enumerate(problem.units) for pid in members}
    for process in plan.processes:
        node = assign[unit_of[process.id]]
        nodes[process.id] = numbered.setdefault(node, len(numbered))
    gpu = frozenset(nodes[pid] for pid in gpu_processes(plan))
    placement = Placement(nodes=nodes, gpu=gpu)
    check_placement(plan, placement)
    return placement


# -- the report


@dataclass
class Row:
    """How one strategy placed one plan."""

    strategy: str
    nodes: int
    gpu_nodes: int
    cut_edges: int
    cut_mbps: float
    raw_cuts: int
    largest_raw_cut_mbps: float
    per_node: list[dict[str, object]] = field(default_factory=list)


def _row(name: str, plan: ProcessPlan, placement: Placement, fps: float) -> Row:
    cut = [
        edge
        for edge in plan.edges
        if isinstance(edge, StreamEdge | RowsEdge | FeederEdge)
        and placement.nodes[edge.source] != placement.nodes[edge.target]
    ]
    raw = [edge for edge in cut if isinstance(edge, StreamEdge) and _is_raw(edge.format)]
    count = len(set(placement.nodes.values()))
    per_node = []
    for node in range(count):
        members = [pid for pid, where in placement.nodes.items() if where == node]
        load = sum((process_load(plan.process(pid), plan, fps) for pid in members), Load())
        per_node.append(
            {
                "node": node,
                "gpu": node in placement.gpu,
                "processes": len(members),
                "gpu_jobs": load.gpu_jobs,
                "encodes": load.encodes,
                "decodes": load.decodes,
                "cores": round(load.cores, 2),
            }
        )
    return Row(
        strategy=name,
        nodes=count,
        gpu_nodes=len(placement.gpu),
        cut_edges=len(cut),
        cut_mbps=round(sum(edge_mbps(edge, fps) for edge in cut), 1),
        raw_cuts=len(raw),
        largest_raw_cut_mbps=round(max((edge_mbps(edge, fps) for edge in raw), default=0.0), 1),
        per_node=per_node,
    )


def report(
    plan: ProcessPlan, strategies: Sequence[str], fps: float = 30.0
) -> list[Row]:
    """`plan` placed under each of `strategies` (the named strategies of
    :mod:`ffrwd.placement` and cost strategies alike), one row each."""
    rows: list[Row] = []
    for name in strategies:
        try:
            from .placement import place

            placement = place(plan, name)
        except FfrwdError as err:
            rows.append(Row(strategy=f"{name} (refused: {err.message})", nodes=0, gpu_nodes=0,
                            cut_edges=0, cut_mbps=0.0, raw_cuts=0, largest_raw_cut_mbps=0.0))
            continue
        rows.append(_row(name, plan, placement, fps))
    return rows


def format_rows(title: str, rows: Sequence[Row]) -> str:
    """The rows as a plain table."""
    lines = [title]
    header = f"  {'strategy':<44} {'nodes':>5} {'L4':>3} {'cuts':>5} {'Mbit/s':>8} " \
             f"{'raw':>4} {'max raw':>8}  per node (processes/gpu jobs/encodes/cores)"
    lines.append(header)
    for row in rows:
        nodes = "  ".join(
            f"{'L4' if one['gpu'] else 'cpu'}:{one['processes']}/{one['gpu_jobs']}/"
            f"{one['encodes']}/{one['cores']}"
            for one in row.per_node
        )
        lines.append(
            f"  {row.strategy:<44} {row.nodes:>5} {row.gpu_nodes:>3} {row.cut_edges:>5} "
            f"{row.cut_mbps:>8} {row.raw_cuts:>4} {row.largest_raw_cut_mbps:>8}  {nodes}"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m ffrwd.placement_cost QUERY [-v k=v ...] [--strategy S ...]``:
    compile each query and print how each strategy places it."""
    from .compiler import compile_all
    from .project import discover
    from .vars import referenced, substitute

    parser = argparse.ArgumentParser(prog="python -m ffrwd.placement_cost")
    parser.add_argument("queries", nargs="+", type=Path)
    parser.add_argument("-v", dest="values", action="append", default=[], metavar="K=V")
    parser.add_argument("--strategy", action="append", default=[])
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    values: dict[str, str] = dict(one.split("=", 1) for one in args.values)
    strategies = args.strategy or ["one", "by-hardware", "per-module", *PRESETS]
    out: dict[str, list[dict[str, object]]] = {}
    for query in args.queries:
        text = query.read_text(encoding="utf-8")
        needed = {name: values.get(name, "1") for name in referenced(text)}
        compiled = compile_all(substitute(text, needed).text, packages=discover(query.parent))
        if compiled.plan is None:
            print(f"{query.name}: one ffmpeg process, nothing to place")
            continue
        rows = report(compiled.plan, strategies, args.fps)
        out[query.name] = [asdict(row) for row in rows]
        if not args.json:
            print(format_rows(f"{query.name} ({len(compiled.plan.processes)} processes)", rows))
            print()
    if args.json:
        print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
