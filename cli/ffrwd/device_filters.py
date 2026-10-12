"""Run a query's filters on the sidecar's device, inside the region that
holds its pictures there.

A node module that holds a picture on the sidecar's GPU puts its region on
the device: the region decodes onto it and, with an encoder the sidecar
hosts, encodes from it. An ffmpeg filter between such a module and the rest
of its region would run in ffmpeg, so the picture would leave the device
for it. Where the sidecar's libavfilter has a variant of the filter for its
device, the filter becomes the sidecar's own ``filter`` node instead, which
runs the variant on the device inside the region: ``scale`` becomes
``scale_d3d12``. A run of such filters is one graph.

``deinterlace_d3d12`` stands in for nothing: creating its video processor
removes the device on NVIDIA's driver (DXGI_ERROR_DEVICE_REMOVED,
0x887A0005), ffmpeg's own command line included, so ``yadif`` and ``bwdif``
stay in ffmpeg.

A filter qualifies when its picture comes from the device (a module output
held on the GPU, another filter that qualifies, or an input's own picture,
which the region then decodes onto the device) and every reader of what it
makes is on the device too (a module input held on the GPU, another filter
that qualifies, or a file whose encoder reads the device's pictures), and
the run of such filters touches a module holding its pictures on the GPU,
since without one there is no region to join. Which variants exist is what
the sidecar the compile finds says its libavfilter has (``ffrwd-wasm
--av-info``), the way the card pass reads the local ffmpeg's filters; where
that sidecar loads no libav at all, the command is written for a reader's
sidecar that carries the bundled libavfilter, as the card pass writes for a
reader's ffmpeg it cannot see. A filter with no variant, or a call naming
an option the variant lacks, stays in ffmpeg.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, replace

from .emit import _escape_value
from .ir import DEVICE_FILTER, GRAPH, FrameRef, Graph, Node, is_src, src_parts
from .shapes import node_shape

__all__ = ["DEVICE_PIX_FMTS", "filters_on_the_device"]

# The pixel formats a ``filter`` node writes its pictures in: yuv420p is the
# device's NV12 repacked, rgba is made on the device from it.
DEVICE_PIX_FMTS = ("yuv420p", "rgba")

# The encoders that read the device's pictures as they are.
_DEVICE_ENCODER_SUFFIX = "_d3d12va"


@dataclass(frozen=True)
class _Variant:
    """A filter of the sidecar's libavfilter standing in for an ffmpeg one."""

    name: str
    # The call's argument -> the variant's option.
    options: Mapping[str, str]


_VARIANTS: Mapping[str, _Variant] = {
    "scale": _Variant(
        name="scale_d3d12",
        options={"width": "w", "height": "h", "w": "w", "h": "h"},
    ),
}


def filters_on_the_device(g: Graph, available: Callable[[], Collection[str] | None]) -> Graph:
    """`g` with each filter that qualifies (see the module docstring) made
    a ``filter`` node running its variant. `available` answers the filters
    the sidecar's libavfilter has, None where that cannot be known, and is
    asked only where a module holds a picture on the GPU and a filter with a
    variant is there to move.

    Returns a new Graph, and `g` itself where nothing qualifies.
    """
    shapes = {
        name: node_shape(g.nodes[name].filter, raw)
        for name, raw in g.node_shapes.items()
        if name in g.nodes
    }
    on_the_gpu = any(
        any(port.accepts.memory == "gpu" for port in shape.inputs)
        or any(port.memory == "gpu" for port in shape.outputs)
        for shape in shapes.values()
    )
    if not on_the_gpu or not any(node.filter in _VARIANTS for node in g.nodes.values()):
        return g
    answered = available()
    has = (
        frozenset(variant.name for variant in _VARIANTS.values())
        if answered is None
        else frozenset(answered)
    )
    graphs = {
        name: graph
        for name, node in g.nodes.items()
        if (graph := _variant_graph(node, has)) is not None
    }
    if not graphs:
        return g
    producers: dict[FrameRef, tuple[str, int]] = {}
    readers: dict[FrameRef, list[tuple[str, int]]] = {}
    for node in g.nodes.values():
        for pad in range(len(node.outputs)):
            producers[f"{node.id}:{pad}"] = (node.id, pad)
            if pad == 0:
                producers[node.id] = (node.id, 0)
    for node in g.nodes.values():
        for position, ref in enumerate(node.inputs):
            readers.setdefault(_slot(ref, producers), []).append((node.id, position))
    sinks: dict[FrameRef, list[str]] = {}
    for unit in g.sinks:
        codec = unit.options.get("video_codec")
        for output in unit.outputs:
            sinks.setdefault(_slot(output.ref, producers), []).append(
                codec if isinstance(codec, str) else ""
            )

    def held_out(ref: FrameRef) -> bool:
        """Whether `ref` is a module output held on the GPU."""
        found = producers.get(ref)
        if found is None:
            return False
        name, pad = found
        shape = shapes.get(name)
        node = g.nodes[name]
        if shape is None or pad >= len(node.out_ports):
            return False
        port = shape.output(node.out_ports[pad])
        return port is not None and port.memory == "gpu"

    def held_in(name: str, position: int) -> bool:
        """Whether input `position` of node `name` is held on the GPU."""
        shape = shapes.get(name)
        node = g.nodes[name]
        if shape is None or position >= len(node.ports):
            return False
        port = shape.input(node.ports[position])
        return port is not None and port.accepts.memory == "gpu"

    # The largest set whose members each read from the device and are read
    # only there, found by dropping members that do not until none is left.
    chosen = set(graphs)
    changed = True
    while changed:
        changed = False
        for name in sorted(chosen):
            node = g.nodes[name]
            source = node.inputs[0]
            reads = (
                held_out(source)
                or _producer(source, producers) in chosen
                or (is_src(source) and src_parts(source)[1] == "video")
            )
            slot = f"{name}:0"
            read_by = readers.get(slot, [])
            files = sinks.get(slot, [])
            read = (
                bool(read_by or files)
                and all(
                    reader in chosen or held_in(reader, position) for reader, position in read_by
                )
                and all(codec.endswith(_DEVICE_ENCODER_SUFFIX) for codec in files)
            )
            if not (reads and read):
                chosen.discard(name)
                changed = True

    # Each run of chosen filters joins a region only where it touches a
    # module holding its pictures on the GPU.
    def touches(name: str, seen: set[str]) -> bool:
        if name in seen:
            return False
        seen.add(name)
        node = g.nodes[name]
        if held_out(node.inputs[0]):
            return True
        if any(held_in(reader, position) for reader, position in readers.get(f"{name}:0", [])):
            return True
        neighbours = [_producer(node.inputs[0], producers)] + [
            reader for reader, _ in readers.get(f"{name}:0", [])
        ]
        return any(n in chosen and touches(n, seen) for n in neighbours if n is not None)

    chosen = {name for name in chosen if touches(name, set())}
    if not chosen:
        return g

    def wire(name: str) -> str:
        """The pixel format the readers of node `name` take."""
        for reader, position in readers.get(f"{name}:0", []):
            if reader in chosen:
                continue
            shape = shapes[reader]
            port = shape.input(g.nodes[reader].ports[position])
            accepted = port.accepts.pixel_formats if port is not None else ()
            if accepted and DEVICE_PIX_FMTS[0] not in accepted:
                return next((f for f in DEVICE_PIX_FMTS if f in accepted), DEVICE_PIX_FMTS[0])
        return DEVICE_PIX_FMTS[0]

    # A chosen filter read by nothing but the next one joins its graph, so
    # a run of them is one graph and the picture is handed in once.
    inputs = {name: list(g.nodes[name].inputs) for name in chosen}
    merged: set[str] = set()
    for name in [n for n in g.nodes if n in chosen]:
        source = inputs[name][0]
        before = _producer(source, producers)
        if (
            before in chosen
            and before not in merged
            and readers.get(f"{before}:0") == [(name, 0)]
            and not sinks.get(f"{before}:0")
        ):
            graphs[name] = f"{graphs[before]},{graphs[name]}"
            inputs[name] = inputs[before]
            merged.add(before)

    nodes = {
        name: (
            replace(
                node,
                filter=DEVICE_FILTER,
                args={GRAPH: graphs[name], "format": wire(name)},
                inputs=inputs[name],
                outputs=list(node.outputs),
            )
            if name in chosen
            else node
        )
        for name, node in g.nodes.items()
        if name not in merged
    }
    return replace(g, nodes=nodes)


def _variant_graph(node: Node, available: Collection[str]) -> str | None:
    """The libavfilter graph running `node` as its device variant, or None
    where it has none in `available` or names an option the variant lacks."""
    variant = _VARIANTS.get(node.filter)
    if variant is None or variant.name not in available:
        return None
    if len(node.inputs) != 1 or node.outputs != ["video"]:
        return None
    options: list[str] = []
    for key, value in node.args.items():
        option = variant.options.get(key)
        if option is None or isinstance(value, bool) or not isinstance(value, str | int | float):
            return None
        options.append(f"{option}={_escape_value(str(value))}")
    return f"{variant.name}={':'.join(options)}" if options else variant.name


def _slot(ref: FrameRef, producers: Mapping[FrameRef, tuple[str, int]]) -> FrameRef:
    """`ref` as ``<node>:<pad>`` where a node writes it, else as it is."""
    found = producers.get(ref)
    return f"{found[0]}:{found[1]}" if found is not None else ref


def _producer(ref: FrameRef, producers: Mapping[FrameRef, tuple[str, int]]) -> str | None:
    found = producers.get(ref)
    return found[0] if found is not None else None
