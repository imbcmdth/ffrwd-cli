"""The form a leaky takes: over coded packets, or over decoded pictures.

``ffrwd.leaky`` over a coded picture drops whole groups, from a late packet
to the next keyframe, which is right only where everything after it hands
the packets on as they are (a publish, a file the stream is copied into).
Where anything decodes the picture after it, dropping single pictures is
cheaper on the viewer: a stall costs a picture or two rather than the wait
for a keyframe. So a leaky reading packets keeps them only when every reader
of its output takes packets; otherwise a decode goes ahead of it, and it
leaks over the pictures.

Runs before :func:`~ffrwd.split.insert_splits` (``compiler.py``), so a
leaky's readers are its consumers themselves.

Pure: returns a new Graph, never mutates `g`.
"""

from __future__ import annotations

from dataclasses import replace

from .ir import LEAKY, FrameRef, Graph, Node
from .shapes import node_shape

# The filter that stands for the decode: it hands on every picture as it is,
# so the ffmpeg running it decodes the packets and nothing else.
DECODE_FILTER = "null"


def _coded(g: Graph, ref: FrameRef) -> bool:
    """Whether `ref` is a coded picture: what a node writes as packets, or
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


def _takes_packets(g: Graph, reader: Node, ref: FrameRef) -> bool:
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


def _readers_take_packets(g: Graph, ref: FrameRef) -> bool:
    """Whether everything reading `ref` takes packets: nodes, and files that
    copy the stream rather than encode it."""
    for node in g.nodes.values():
        if ref in node.inputs and not _takes_packets(g, node, ref):
            return False
    for sink in g.sinks:
        codec = sink.options.get("video_codec")
        if any(output.ref == ref for output in sink.outputs) and codec not in (None, "copy"):
            return False
    return True


def decode_ahead_of_leaky(g: Graph) -> Graph:
    """A decode in front of each leaky reading packets whose output something
    decodes anyway."""
    decodes: dict[str, Node] = {}
    for node in g.nodes.values():
        if node.filter != LEAKY or not node.inputs:
            continue
        read = node.inputs[0]
        if not _coded(g, read) or _readers_take_packets(g, node.id):
            continue
        decodes[node.id] = Node(
            id=f"{node.id}_decode",
            filter=DECODE_FILTER,
            args={},
            inputs=[read],
            outputs=["video"],
        )
    if not decodes:
        return g
    nodes: dict[str, Node] = {}
    for name, node in g.nodes.items():
        decode = decodes.get(name)
        if decode is not None:
            nodes[decode.id] = decode
            node = replace(node, inputs=[decode.id, *node.inputs[1:]])
        nodes[name] = node
    return replace(g, nodes=nodes)
