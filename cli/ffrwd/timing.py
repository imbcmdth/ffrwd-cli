"""How late each node module's outputs run behind the source, and why.

Every node declares the window its clock input reads and how late each output
may leave; every input says how it pairs with the clock. Summed along the
paths of a graph, those say how far behind its source each stream a query
writes runs (:func:`timing`):

- a source's stream runs at its source's time, 0;
- a node is ready for a tick once every input it waits for has arrived:
  the clock, each lockstep input, and each interval input with its `ahead`,
  an interval input waiting no longer than its own bound past the clock.
  A held input and one delivered on arrival hold nothing up, and an
  interval input the host re-times onto the clock (anchored `first-frame`
  or `tagged`) waits its bound and no more, since its producer counts from
  another origin;
- a node's window is how long its tick takes to fill, a tumbling 2 s
  window 2 s; a node's output adds the output's declared latency;
- the host's span reducer adds its `max_span`.

Where streams that run at different delays meet, in one node or in one file
written, the earlier one waits: :attr:`OutputTiming.holds` says for how long.
A live query cannot give a node an interval input later than the bound the
node set on it, which is :data:`~ffrwd.errors.ErrorCode.LIVE_LEAD`
(:func:`check_live_leads`).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from fractions import Fraction

from .errors import ErrorCode, FfrwdError
from .ir import (
    LEAKY,
    MAX_SPAN,
    ROWMERGE,
    FrameRef,
    Graph,
    StreamType,
    is_src,
    src_parts,
)
from .probe import ProbeResult
from .shapes import InputPort, NodeShape, node_shape, window_words

__all__ = [
    "HOLD_LIMIT",
    "NodeTiming",
    "OutputTiming",
    "Paths",
    "Timing",
    "check_live_leads",
    "paths_of",
    "stream_rate",
    "summary",
    "timing",
]

# How many bytes a stream may wait for the one written beside it before the
# compile says so: a picture held behind 30 s of late words is about 2 GB of
# 1080p, and that is worth a line.
HOLD_LIMIT = 512 * 1024 * 1024

# Bytes per picture element of a raw picture on an edge, yuv420p's; and of
# one sample of one channel, f32's.
_PICTURE_BYTES = Fraction(3, 2)
_SAMPLE_BYTES = 4


@dataclass(frozen=True)
class Wait:
    """One input of a node: how it pairs, and how late what it reads runs.

    `timing` marks an input read for its frames' times alone; `retimed` an
    interval input the host re-stamps onto the clock.
    """

    port: str
    pairing: str
    delay: float | None
    bound: float | None = None
    timing: bool = False
    retimed: bool = False


@dataclass(frozen=True)
class NodeTiming:
    """One node module: its window in streaming words, and when it is ready.

    `delay` is how far behind the source a tick's outputs leave, before each
    output's own latency; None where something on the way has no size.
    """

    node: str
    module: str
    window: str
    waits: tuple[Wait, ...]
    delay: float | None
    # What the query calls it.
    called: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "node": self.node,
            "module": self.module,
            "called": self.called,
            "window": self.window,
            "inputs": [
                {
                    "port": wait.port,
                    "pairing": wait.pairing,
                    "delay": wait.delay,
                    **({"bound": wait.bound} if wait.bound is not None else {}),
                    **({"wants": "timing"} if wait.timing else {}),
                    **({"retimed": True} if wait.retimed else {}),
                }
                for wait in self.waits
            ],
            "delay": self.delay,
        }


@dataclass(frozen=True)
class OutputTiming:
    """One stream a query writes: how late it runs, how long it waits for
    the latest one written beside it, and about how many bytes that wait
    holds, None where the stream's size is not known."""

    ref: FrameRef
    delay: float | None
    holds: float
    held: int | None = None
    # Where it is written: the file, the stream's place in it and its kind;
    # a rows file names no stream.
    path: str = ""
    index: int | None = None
    kind: str = ""

    def to_dict(self) -> dict[str, object]:
        written: dict[str, object] = {
            "ref": self.ref,
            "path": self.path,
            "index": self.index,
            "kind": self.kind,
            "delay": self.delay,
            "holds": self.holds,
        }
        if self.held:
            written["held_bytes"] = self.held
        return written


@dataclass(frozen=True)
class Timing:
    nodes: tuple[NodeTiming, ...]
    outputs: tuple[OutputTiming, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "nodes": [node.to_dict() for node in self.nodes],
            "outputs": [output.to_dict() for output in self.outputs],
        }


class Paths:
    """Delays over one graph, each ref's counted once."""

    def __init__(self, graph: Graph, probes: Mapping[str, ProbeResult | None]) -> None:
        self.graph = graph
        self.probes = probes
        self.shapes = {
            name: node_shape(graph.nodes[name].filter, raw)
            for name, raw in graph.node_shapes.items()
            if name in graph.nodes
        }
        self._delays: dict[FrameRef, float | None] = {}
        self._ready: dict[str, tuple[float | None, tuple[Wait, ...]]] = {}

    def delay(self, ref: FrameRef) -> float | None:
        if ref not in self._delays:
            self._delays[ref] = self._count(ref)
        return self._delays[ref]

    def _count(self, ref: FrameRef) -> float | None:
        if is_src(ref):
            return 0.0
        name, _, pad = ref.partition(":")
        node = self.graph.nodes.get(name)
        if node is None:
            return 0.0
        shape = self.shapes.get(name)
        if shape is not None:
            ready = self.ready(name)[0]
            output = shape.outputs[int(pad) if pad else 0] if shape.outputs else None
            latency = output.latency if output is not None else 0.0
            return None if ready is None else ready + latency
        found: list[float | None] = [self.delay(one) for one in node.inputs]
        if any(one is None for one in found):
            return None
        above = max((one for one in found if one is not None), default=0.0)
        span = node.args.get(MAX_SPAN) if node.filter == ROWMERGE else None
        if isinstance(span, int | float) and not isinstance(span, bool):
            return above + float(span)
        return above

    def ready(self, name: str) -> tuple[float | None, tuple[Wait, ...]]:
        """When node `name`'s tick has all it waits for, its window filled."""
        if name in self._ready:
            return self._ready[name]
        node = self.graph.nodes[name]
        shape = self.shapes[name]
        by_port: dict[str, list[FrameRef]] = {}
        for bound, ref in zip(node.ports, node.inputs):
            by_port.setdefault(bound, []).append(ref)
        clock = shape.clock_input
        clock_delay: float | None = 0.0
        if clock is not None:
            clock_delay = self._latest(by_port.get(clock.name, []))
        waits: list[Wait] = []
        ready = clock_delay
        for port in shape.inputs:
            refs = by_port.get(port.name, [])
            if not refs:
                continue
            arrived = self._latest(refs)
            pairing = port.pairing
            interval = pairing.interval
            timed = port.accepts.wants == "timing"
            if pairing.kind == "lockstep":
                waited = arrived
                said = "clock" if clock is not None and port.name == clock.name else "lockstep"
            elif interval is not None and interval.retimed:
                waits.append(
                    Wait(port.name, "interval", arrived, interval.latency, retimed=True)
                )
                if interval.latency is None or clock_delay is None:
                    continue
                waited = clock_delay + interval.latency + interval.ahead
                ready = None if ready is None else max(ready, waited)
                continue
            elif interval is not None:
                said = "interval"
                waited = None if arrived is None else arrived + interval.ahead
                if interval.latency is not None and clock_delay is not None:
                    limit = clock_delay + interval.latency + interval.ahead
                    waited = limit if waited is None else min(waited, limit)
            else:
                waits.append(Wait(port.name, pairing.kind, arrived, timing=timed))
                continue
            waits.append(
                Wait(
                    port.name,
                    said,
                    arrived,
                    interval.latency if interval else None,
                    timing=timed,
                )
            )
            ready = None if ready is None or waited is None else max(ready, waited)
        window = self.window_seconds(name, shape, by_port)
        total = None if ready is None or window is None else ready + window
        self._ready[name] = (total, tuple(waits))
        return self._ready[name]

    def latest(self, refs: list[FrameRef]) -> float | None:
        """The latest of `refs`, or None where one of them has no size."""
        return self._latest(refs)

    def _latest(self, refs: list[FrameRef]) -> float | None:
        found = [self.delay(ref) for ref in refs]
        if any(one is None for one in found):
            return None
        return max((one for one in found if one is not None), default=0.0)

    def window_seconds(
        self, name: str, shape: NodeShape, by_port: Mapping[str, list[FrameRef]]
    ) -> float | None:
        """How long the clock input's window takes to fill; 0 for one frame."""
        clock = shape.clock_input
        if clock is None or clock.window <= 1:
            return 0.0
        rate = self.clock_rate(clock, by_port.get(clock.name, []))
        return None if rate is None else float(Fraction(clock.window) / rate)

    def clock_rate(self, port: InputPort, refs: list[FrameRef]) -> Fraction | None:
        """Items per second of what the clock input reads: frames, or samples."""
        if port.kind == "audio" and port.accepts.sample_rates:
            return Fraction(port.accepts.sample_rates[0])
        return self.rate(refs[0]) if refs else None

    def bytes_per_second(self, ref: FrameRef) -> Fraction | None:
        """About how many bytes a second of the raw stream `ref` is on an edge."""
        origin = self.origin(ref)
        if origin is None:
            return None
        alias, kind, index = origin
        probe = self.probes.get(alias)
        streams = probe.by_type(kind) if probe is not None else []
        if index >= len(streams):
            return None
        stream = streams[index]
        if kind == "video":
            rate = _fraction(stream.fps)
            if rate is None or not stream.width or not stream.height:
                return None
            return stream.width * stream.height * _PICTURE_BYTES * rate
        if kind == "audio" and stream.sample_rate and stream.channels:
            return Fraction(stream.sample_rate * stream.channels * _SAMPLE_BYTES)
        return None

    def origin(self, ref: FrameRef) -> tuple[str, StreamType, int] | None:
        """The source stream `ref` was made from, along first inputs."""
        while not is_src(ref):
            node = self.graph.nodes.get(ref.partition(":")[0])
            if node is None or not node.inputs:
                return None
            ref = node.inputs[0]
        alias, kind, index = src_parts(ref)
        return alias, kind, index

    def rate(self, ref: FrameRef) -> Fraction | None:
        """The rate of the stream `ref` is (:func:`stream_rate`)."""
        return stream_rate(self.graph, self.probes, self.shapes.get, ref)


# Filters whose pictures or sound leave at a rate their args do not say.
_RATE_LOST = frozenset(
    {
        "ainterleave",
        "aselect",
        "asetpts",
        "atempo",
        "decimate",
        "framestep",
        "interleave",
        "minterpolate",
        "mpdecimate",
        "select",
        "setpts",
        "thumbnail",
        "tile",
    }
)

# The args a filter setting a rate says it in; a filter with no input is a
# source, and says its rate in one of `_SOURCE_RATE_ARGS`.
_RATE_ARGS: Mapping[str, tuple[str, ...]] = {
    "fps": ("fps", "rate", "r"),
    "framerate": ("fps", "rate", "r"),
    "aresample": ("osr", "out_sample_rate", "sample_rate"),
    "asetrate": ("sample_rate", "r"),
}
_SOURCE_RATE_ARGS = ("rate", "r", "framerate", "fps", "sample_rate")


def stream_rate(
    graph: Graph,
    probes: Mapping[str, ProbeResult | None],
    shape_of: Callable[[str], NodeShape | None],
    ref: FrameRef,
    *,
    through_nodes: bool = True,
) -> Fraction | None:
    """The rate of the stream `ref` names, as known before the run.

    A picture's frame rate, a sound's sample rate; None for data and coded
    streams a node writes, and wherever nothing settles it: a self-clocked
    node's output, a stream with no probe (a feed by port), a filter whose
    rate its args do not say. A node's picture leaves once per tick, so at
    its clock's rate: a rate clock's own, the rate of the port a `rate-of`
    clock names, the clock input's rate over its stride. A node's sound is
    at the rate its format says, else the rate of the input it follows.

    Without `through_nodes`, a stream a node writes, a node read in FROM
    included, has none: the host hints such a stream with nothing, since a
    network binds it by its label.
    """
    seen: set[str] = set()
    while True:
        if is_src(ref):
            alias, kind, index = src_parts(ref)
            if not through_nodes and alias in graph.node_sources:
                return None
            probe = probes.get(alias)
            streams = probe.by_type(kind) if probe is not None else []
            if index >= len(streams):
                return None
            stream = streams[index]
            if kind == "audio":
                return Fraction(stream.sample_rate) if stream.sample_rate else None
            return _fraction(stream.fps) if kind == "video" else None
        name, _, pad_text = ref.partition(":")
        node = graph.nodes.get(name)
        if node is None or name in seen:
            return None
        seen.add(name)
        pad = int(pad_text) if pad_text.isdigit() else 0
        kind = node.outputs[pad] if pad < len(node.outputs) else "video"
        if kind not in ("video", "audio"):
            return None
        shape = shape_of(name)
        if not through_nodes and (shape is not None or node.filter == LEAKY):
            return None
        if shape is not None:
            return _node_output_rate(graph, probes, shape_of, name, shape, pad, kind)
        if node.filter in _RATE_LOST:
            return None
        said = _RATE_ARGS.get(node.filter, () if node.inputs else _SOURCE_RATE_ARGS)
        for key in said:
            rate = _rate_arg(node.args.get(key))
            if rate is not None:
                return rate
        if node.filter in _RATE_ARGS or not node.inputs:
            return None
        ref = next(
            (one for one in node.inputs if _kind_of(graph, one) == kind), node.inputs[0]
        )


def _node_output_rate(
    graph: Graph,
    probes: Mapping[str, ProbeResult | None],
    shape_of: Callable[[str], NodeShape | None],
    name: str,
    shape: NodeShape,
    pad: int,
    kind: str,
) -> Fraction | None:
    """The rate of what node `name` writes on output `pad`, of `kind`."""
    node = graph.nodes[name]
    output = shape.outputs[pad] if pad < len(shape.outputs) else None
    if output is None or output.kind not in ("video", "audio"):
        return None

    def first(port: str) -> FrameRef | None:
        return next((ref for bound, ref in zip(node.ports, node.inputs) if bound == port), None)

    found = output.format
    if kind == "audio":
        if found is not None and found.kind == "audio":
            return Fraction(found.sample_rate) if found.sample_rate else None
        follows = (
            found.port
            if found is not None and found.kind == "like"
            else shape.clock.port
            if shape.clock.kind == "input"
            else None
        )
        read = first(follows) if follows else None
        if read is None or _kind_of(graph, read) != "audio":
            return None
        return stream_rate(graph, probes, shape_of, read)
    clock = shape.clock
    if clock.kind == "rate" and clock.rate is not None:
        return Fraction(*clock.rate)
    if clock.kind not in ("input", "rate-of"):
        return None
    read = first(clock.port)
    rate = stream_rate(graph, probes, shape_of, read) if read is not None else None
    port = shape.input(clock.port)
    if rate is None or clock.kind == "rate-of" or port is None:
        return rate
    return rate / port.stride


def _kind_of(graph: Graph, ref: FrameRef) -> str:
    if is_src(ref):
        return src_parts(ref)[1]
    name, _, pad_text = ref.partition(":")
    node = graph.nodes.get(name)
    pad = int(pad_text) if pad_text.isdigit() else 0
    if node is None or pad >= len(node.outputs):
        return "video"
    return node.outputs[pad]


def _rate_arg(value: object) -> Fraction | None:
    """A rate as a filter's arg says it: ``30``, ``29.97`` or ``30000/1001``."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return Fraction(value) if value > 0 else None
    if isinstance(value, float):
        return Fraction(value).limit_denominator(1001) if value > 0 else None
    if not isinstance(value, str):
        return None
    text = value.strip()
    try:
        found = Fraction(text) if "/" in text else Fraction(text).limit_denominator(1001)
    except (ValueError, ZeroDivisionError):
        return None
    return found if found > 0 else None


def paths_of(graph: Graph, probes: Mapping[str, ProbeResult | None]) -> Paths:
    """How late each ref of `graph` runs, counted as it is asked for."""
    return Paths(graph, probes)


def _fraction(fps: str | None) -> Fraction | None:
    numerator, _, denominator = (fps or "").partition("/")
    try:
        found = Fraction(int(numerator), int(denominator) if denominator else 1)
    except (ValueError, ZeroDivisionError):
        return None
    return found if found > 0 else None


def timing(
    graph: Graph,
    probes: Mapping[str, ProbeResult | None],
    called: Mapping[str, str] | None = None,
) -> Timing | None:
    """Each node module's window and readiness, and each written stream's
    delay; None for a graph with no node module in it. `called` names each
    module path the way the query calls it."""
    if not graph.node_shapes:
        return None
    paths = Paths(graph, probes)
    names = called or {}
    # A track minted from rows runs as late as the node writing them.
    rows_of = {sink.alias: ref for ref, sink in graph.rows_sinks.items() if sink.alias}
    nodes: list[NodeTiming] = []
    for name, shape in paths.shapes.items():
        node = graph.nodes[name]
        by_port: dict[str, list[FrameRef]] = {}
        for bound, ref in zip(node.ports, node.inputs):
            by_port.setdefault(bound, []).append(ref)
        clock = shape.clock_input
        if clock is not None:
            rate = paths.clock_rate(clock, by_port.get(clock.name, []))
            window = window_words(clock, rate)
        elif shape.clock.kind == "rate" and shape.clock.rate is not None:
            window = f"rate {_rate_words(shape.clock.rate)}"
        elif shape.clock.kind == "rate-of":
            window = f"at the rate of {shape.clock.port}"
        else:
            window = shape.clock.kind
        ready, waits = paths.ready(name)
        called_as = names.get(node.filter, node.filter)
        nodes.append(NodeTiming(name, node.filter, window, waits, ready, called_as))
    outputs: list[OutputTiming] = []
    for unit in graph.sinks:
        refs = [
            rows_of.get(src_parts(output.ref)[0], output.ref) if is_src(output.ref)
            else output.ref
            for output in unit.outputs
        ]
        delays = [paths.delay(ref) for ref in refs]
        latest = max((d for d in delays if d is not None), default=0.0)
        for index, (output, delay) in enumerate(zip(unit.outputs, delays)):
            holds = 0.0 if delay is None else latest - delay
            per_second = paths.bytes_per_second(output.ref) if holds else None
            held = None if per_second is None else int(per_second * Fraction(holds))
            outputs.append(
                OutputTiming(
                    output.ref, delay, holds, held, unit.path or "", index, output.type
                )
            )
    for ref, sink in graph.rows_sinks.items():
        if sink.path:
            outputs.append(OutputTiming(ref, paths.delay(ref), 0.0, path=sink.path, kind="rows"))
    return Timing(tuple(nodes), tuple(outputs))


def summary(timed: Timing) -> str:
    """What ``explain --delays`` prints: a line per node, then per output.

    A node says its window in streaming words, each input it reads for its
    timing alone, and how each input it waits for by interval is bounded,
    one the host re-times said to be on its own clock; an output how far
    behind the source it runs, and how long it waits for the latest stream
    written beside it.
    """
    lines: list[str] = []
    for node in timed.nodes:
        said = [node.window]
        for wait in node.waits:
            if wait.timing:
                said.append(f"{wait.port} for its timing")
            if wait.pairing != "interval":
                continue
            bound = "no bound" if wait.bound is None else f"at most {_seconds(wait.bound)}"
            own = " on its own clock" if wait.retimed else ""
            said.append(f"{wait.port} by interval{own}, {bound}")
        lines.append(f"{node.called}: {'; '.join(said)}")
    for output in timed.outputs:
        where = (
            f"{output.path} ({output.kind})"
            if output.index is None
            else f"{output.path} stream {output.index} ({output.kind})"
        )
        late = "no known delay" if output.delay is None else (
            f"{_seconds(output.delay)} behind the source"
        )
        waits = f", waits {_seconds(output.holds)}" if output.holds else ""
        lines.append(f"{where}: {late}{waits}")
    return "\n".join(lines)


def _rate_words(rate: tuple[int, int]) -> str:
    num, den = rate
    return f"{num}/s" if den == 1 else f"{num}/{den}/s"


def check_live_leads(
    graph: Graph,
    probes: Mapping[str, ProbeResult | None],
    anchors: Mapping[str, tuple[int, int, str]],
) -> None:
    """Refuse a node whose interval input runs later than the bound it set.

    A node bounding how long it waits for an input (`interval.latency`) is
    one that acts ahead of the rows it reads, and in a live run what arrives
    past the bound is late for good. `anchors` maps a module path to where
    the query named it and the function's name.
    """
    paths = Paths(graph, probes)
    for name, shape in paths.shapes.items():
        node = graph.nodes[name]
        clock = shape.clock_input
        by_port: dict[str, list[FrameRef]] = {}
        for bound, ref in zip(node.ports, node.inputs):
            by_port.setdefault(bound, []).append(ref)
        clock_delay = paths._latest(by_port.get(clock.name, [])) if clock is not None else 0.0
        for port in shape.inputs:
            interval = port.pairing.interval
            if interval is None or interval.latency is None or interval.retimed:
                continue
            arrived = paths._latest(by_port.get(port.name, []))
            if arrived is None or clock_delay is None:
                continue
            late = arrived - clock_delay
            if late <= interval.latency:
                continue
            line, col, called = anchors.get(node.filter, (1, 1, node.filter))
            raise FfrwdError(
                ErrorCode.LIVE_LEAD,
                f"{called}() needs '{port.name}' {_seconds(interval.latency)} ahead of "
                f"its clock, and the path feeding it runs {_seconds(late)} behind",
                line=line,
                col=col,
                hint=f"feed '{port.name}' from a path no later than "
                f"{_seconds(interval.latency)}, or give {called}() a longer lead",
            )


def _seconds(value: float) -> str:
    return f"{round(value, 3):g} s"
