"""How late each node module's outputs run behind the source, and why.

Every node declares the window its clock input reads and how late each output
may leave; every input says how it pairs with the clock. Summed along the
paths of a graph, those say how far behind its source each stream a query
writes runs (:func:`timing`):

- a source's stream runs at its source's time, 0;
- a node is ready for a tick once every input it waits for has arrived:
  the clock, each lockstep input, and each interval input with its `ahead`,
  an interval input waiting no longer than its own bound past the clock.
  A held input and one delivered on arrival hold nothing up;
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

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction

from .errors import ErrorCode, FfrwdError
from .ir import (
    MAX_SPAN,
    MERGE_SPANS,
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
    "Timing",
    "check_live_leads",
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
    """One input of a node: how it pairs, and how late what it reads runs."""

    port: str
    pairing: str
    delay: float | None
    bound: float | None = None


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

    def to_dict(self) -> dict[str, object]:
        return {
            "node": self.node,
            "module": self.module,
            "window": self.window,
            "inputs": [
                {
                    "port": wait.port,
                    "pairing": wait.pairing,
                    "delay": wait.delay,
                    **({"bound": wait.bound} if wait.bound is not None else {}),
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

    def to_dict(self) -> dict[str, object]:
        written: dict[str, object] = {
            "ref": self.ref,
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


class _Paths:
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
        if node.filter == ROWMERGE and node.args.get(MERGE_SPANS):
            span = node.args.get(MAX_SPAN)
            return above + float(span) if isinstance(span, int | float) else None
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
            if pairing.kind == "lockstep":
                waited = arrived
                said = "clock" if clock is not None and port.name == clock.name else "lockstep"
            elif interval is not None:
                said = "interval"
                waited = None if arrived is None else arrived + interval.ahead
                if interval.latency is not None and clock_delay is not None:
                    limit = clock_delay + interval.latency + interval.ahead
                    waited = limit if waited is None else min(waited, limit)
            else:
                waits.append(Wait(port.name, pairing.kind, arrived))
                continue
            waits.append(
                Wait(port.name, said, arrived, interval.latency if interval else None)
            )
            ready = None if ready is None or waited is None else max(ready, waited)
        window = self.window_seconds(name, shape, by_port)
        total = None if ready is None or window is None else ready + window
        self._ready[name] = (total, tuple(waits))
        return self._ready[name]

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
        return self.rate(refs[0], port.kind) if refs else None

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

    def rate(self, ref: FrameRef, kind: str) -> Fraction | None:
        """The rate of the stream `ref` is, read back to where it came from."""
        if is_src(ref):
            alias, stream_kind, index = src_parts(ref)
            probe = self.probes.get(alias)
            streams = probe.by_type(stream_kind) if probe is not None else []
            if index >= len(streams):
                return None
            stream = streams[index]
            if kind == "audio":
                return Fraction(stream.sample_rate) if stream.sample_rate else None
            return _fraction(stream.fps)
        name, _, _ = ref.partition(":")
        node = self.graph.nodes.get(name)
        if node is None:
            return None
        shape = self.shapes.get(name)
        if shape is not None and shape.clock.rate is not None and kind == "video":
            return Fraction(*shape.clock.rate)
        return self.rate(node.inputs[0], kind) if node.inputs else None


def _fraction(fps: str | None) -> Fraction | None:
    numerator, _, denominator = (fps or "").partition("/")
    try:
        found = Fraction(int(numerator), int(denominator) if denominator else 1)
    except (ValueError, ZeroDivisionError):
        return None
    return found if found > 0 else None


def timing(graph: Graph, probes: Mapping[str, ProbeResult | None]) -> Timing | None:
    """Each node module's window and readiness, and each written stream's
    delay; None for a graph with no node module in it."""
    if not graph.node_shapes:
        return None
    paths = _Paths(graph, probes)
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
        else:
            window = shape.clock.kind
        ready, waits = paths.ready(name)
        nodes.append(NodeTiming(name, node.filter, window, waits, ready))
    outputs: list[OutputTiming] = []
    for unit in graph.sinks:
        delays = [paths.delay(output.ref) for output in unit.outputs]
        latest = max((d for d in delays if d is not None), default=0.0)
        for output, delay in zip(unit.outputs, delays):
            holds = 0.0 if delay is None else latest - delay
            per_second = paths.bytes_per_second(output.ref) if holds else None
            held = None if per_second is None else int(per_second * Fraction(holds))
            outputs.append(OutputTiming(output.ref, delay, holds, held))
    for ref in graph.rows_sinks:
        if not any(output.ref == ref for output in outputs):
            outputs.append(OutputTiming(ref, paths.delay(ref), 0.0))
    return Timing(tuple(nodes), tuple(outputs))


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
    paths = _Paths(graph, probes)
    for name, shape in paths.shapes.items():
        node = graph.nodes[name]
        clock = shape.clock_input
        by_port: dict[str, list[FrameRef]] = {}
        for bound, ref in zip(node.ports, node.inputs):
            by_port.setdefault(bound, []).append(ref)
        clock_delay = paths._latest(by_port.get(clock.name, [])) if clock is not None else 0.0
        for port in shape.inputs:
            interval = port.pairing.interval
            if interval is None or interval.latency is None:
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
