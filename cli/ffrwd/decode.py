"""A decode in front of everything that reads a coded stream as frames.

A node writing packets (a node source such as a subscribe) or a packet filter
hands on a coded stream, and the sidecar decodes nothing: it binds a coded
stream to a packets port alone. So where a coded stream is read by something
taking frames -- a node's video or audio port, timing ones included, or a
frame module the sidecar hosts -- a decode goes in front of it, which the
partitioner puts in an ffmpeg. Whatever takes packets keeps the packets, a
leaky is left to :func:`~ffrwd.leaky.decode_ahead_of_leaky`, and an ffmpeg
filter already runs where it decodes.

Runs before :func:`~ffrwd.split.insert_splits` (``compiler.py``), so a coded
stream's readers are its consumers themselves: one decode serves them all, and
the split pass fans it out.

Pure: returns a new Graph, never mutates `g`.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import replace

from .ir import LEAKY, FrameRef, Graph, Node, StreamType
from .processes import ref_type
from .shapes import node_shape

# The filters that stand for a decode: each hands on every frame as it is, so
# the ffmpeg running it decodes the packets and nothing else.
DECODE_FILTERS: Mapping[StreamType, str] = {"video": "null", "audio": "anull"}


def coded(g: Graph, ref: FrameRef) -> bool:
    """Whether `ref` is a coded stream: what a node writes as packets, or
    what a packet filter hands back."""
    name, _, pad = ref.partition(":")
    if name in g.packet_filters:
        return True
    raw = g.node_shapes.get(name)
    node = g.nodes.get(name)
    if raw is None or node is None:
        return False
    outputs = node_shape(node.filter, raw).outputs
    index = int(pad) if pad.isdigit() else 0
    return index < len(outputs) and outputs[index].kind == "packets"


def takes_packets(g: Graph, reader: Node, ref: FrameRef) -> bool:
    """Whether `reader` takes `ref` as packets: a node port reading packets,
    or a packet sink or filter."""
    if reader.id in g.packet_sinks or reader.id in g.packet_filters:
        return True
    raw = g.node_shapes.get(reader.id)
    if raw is None or ref not in reader.inputs:
        return False
    position = reader.inputs.index(ref)
    if position >= len(reader.ports):
        return False
    port = node_shape(reader.filter, raw).input(reader.ports[position])
    return port is not None and port.kind == "packets"


def _takes_frames(g: Graph, reader: Node, ref: FrameRef, modules: Collection[str]) -> bool:
    """Whether the sidecar hands `reader` the frames of `ref`: a node port of
    kind video or audio, or a frame module of `modules`. A data filter reads
    its clock for the time alone, and a codec's decoder reads the packets."""
    if (
        reader.filter == LEAKY
        or reader.id in g.data_filters
        or reader.id in g.decoders
        or takes_packets(g, reader, ref)
    ):
        return False
    raw = g.node_shapes.get(reader.id)
    if raw is None:
        return reader.filter in modules
    position = reader.inputs.index(ref)
    if position >= len(reader.ports):
        return False
    port = node_shape(reader.filter, raw).input(reader.ports[position])
    return port is not None and port.kind in ("video", "audio")


def decode_ahead_of_frame_readers(g: Graph, modules: Collection[str]) -> Graph:
    """A decode in front of every reader of a coded stream that takes frames.

    `modules` are the paths of the modules the sidecar hosts, the filters of
    the nodes ffmpeg cannot run. One decode per coded stream, ahead of its
    first such reader, and every such reader reads it.
    """
    decodes: dict[FrameRef, Node] = {}
    nodes: dict[str, Node] = {}
    for name, node in g.nodes.items():
        inputs = list(node.inputs)
        for position, ref in enumerate(node.inputs):
            kind = ref_type(g, ref)
            if (
                kind not in DECODE_FILTERS
                or not coded(g, ref)
                or not _takes_frames(g, node, ref, modules)
            ):
                continue
            decode = decodes.get(ref)
            if decode is None:
                decode = Node(
                    id=f"{ref.replace(':', '_')}_decode",
                    filter=DECODE_FILTERS[kind],
                    args={},
                    inputs=[ref],
                    outputs=[kind],
                )
                decodes[ref] = decode
                nodes[decode.id] = decode
            inputs[position] = decode.id
        nodes[name] = node if inputs == node.inputs else replace(node, inputs=inputs)
    if not decodes:
        return g
    return replace(g, nodes=nodes)
