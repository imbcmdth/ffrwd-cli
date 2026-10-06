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

from .decode import DECODE_FILTERS, coded, takes_packets
from .ir import LEAKY, FrameRef, Graph, Node


def _readers_take_packets(g: Graph, ref: FrameRef) -> bool:
    """Whether everything reading `ref` takes packets: nodes, and files that
    copy the stream rather than encode it."""
    for node in g.nodes.values():
        if ref in node.inputs and not takes_packets(g, node, ref):
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
        if not coded(g, read) or _readers_take_packets(g, node.id):
            continue
        decodes[node.id] = Node(
            id=f"{node.id}_decode",
            filter=DECODE_FILTERS["video"],
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
