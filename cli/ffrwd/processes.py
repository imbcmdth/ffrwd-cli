"""The level above one ffmpeg command: a graph of processes.

A logical :class:`~ffrwd.ir.Graph` is what the query means; it is not always
what one ffmpeg can run. A node ffmpeg cannot host runs in the sidecar
instead, and the streams around it move between processes over pipes.
:func:`partition` contracts every ffmpeg-hostable part of a graph into one
ffmpeg process and returns the process DAG that falls out: ffmpeg processes,
sidecar processes, and typed edges.

The contraction is symmetric. A run of adjacent external nodes -- any DAG
shape, chain, fan-out or fan-in -- becomes ONE sidecar process carrying that
region's own subgraph, the way an ffmpeg process carries its own. The sidecar
hosts a network of modules, so frames cross between two modules of one region
in memory and no pipe, no NUT hop and no split node stands between them; only
the region's BOUNDARY is edges. A region is contracted only while it stays
convex -- no path may leave it and come back -- since a region an ffmpeg
process sits inside would make the process DAG a cycle.

Same contract as the split pass -- plan in, plan out, nothing mutated.

Edges
-----
A STREAM edge is one stream on a pipe, always NUT-wrapped so the reader finds
the parameters in the header: ``rawvideo`` for video, pcm for audio.
A FILE edge is an artifact on disk: a media file, or rows as NDJSON.

Stream edges wire processes that run at the same time; file edges order one
group of processes after the previous one. :attr:`ProcessPlan.stages` is that
grouping.

Which process gets which node
-----------------------------
Every node has a DEPTH: how many sidecar hops its inputs came through. A
source ref is depth 0; a node's depth is the largest depth over its inputs,
plus one for each input a sidecar produced. Depth never decreases along an
edge, so grouping by it cannot produce a cycle -- which grouping by plain
connectivity can, when a hostable node feeds a sidecar AND the hostable node
that reads that sidecar back.

An ffmpeg process holds exactly the depth-`d` nodes its own outputs need. Two
kinds exist: one per output FILE group (the sinks at depth `d`), and one per
raw stream edge. A node no output needs is in no process at all -- lowering
does not emit one.

Deadlock
--------
An ffmpeg process hands raw streams to at most ONE consumer. Two readers of
one ffmpeg would each wait on the other's pace, so a raw stream that fans out
to two processes DUPLICATES its producer instead: a second ffmpeg process
decodes the input again. That is a real cost, and the plan shows it -- both
processes are there, both reading the same file. Several streams read by ONE
process are different: that consumer takes them at its own interleaving, so
neither branch can outrun the other, and sibling legs reading exactly the
same inputs share a producer -- one decode, a `split` fanning the branches,
one output per branch. Fan-IN is free: an ffmpeg process reads as many pipes
as it likes, and so does a sidecar.

Live inputs
-----------
Duplicating the producer is only available to an input that can be opened
twice. A protocol -- srt, udp, rtmp -- and a capture device cannot: the second
open takes a socket already bound, or a camera already held. Neither can a
still-live HLS or DASH manifest, whose second read would resume mid-stream
rather than start over. Such an input (:func:`is_live`, :func:`is_live_probe`)
is therefore read by exactly ONE process whatever the graph shape. Every leg
over it joins that reader instead of opening the input again,
and the reader writes each consumer's stream to a pipe of its own -- one
ffmpeg with several outputs. A process that would have opened the input
directly reads one of those pipes instead.

Now the two readers of one ffmpeg the rule above forbids are exactly what the
plan has, so the buffers between them have to be big enough that neither waits
on the other. :attr:`StreamEdge.bound` is that size, in frames: how far ahead
of its siblings an edge runs while the slowest of them reaches the process
where they meet again. It is the difference between two path delays, where a
process on a path holds one frame plus whatever its modules declare they read
ahead. :attr:`StreamEdge.buffer` is what the bound buys -- a sized named pipe
where the bytes fit under :data:`PIPE_BUFFER_LIMIT`, ffmpeg's own fifo muxer
where they do not.

A path whose delay cannot be counted -- a module that does not hand on one
frame per frame it reads, an ffmpeg filter that changes the frame count -- has
no such difference, and a live input feeding one is refused at compile time.

A payload holds only the split it needs. A `split`/`asplit` whose consumers
landed in other processes is cut to the pads still read here: one left and the
split dissolves, its consumer reading the split's own input; several and the
split keeps that count, its surviving pads renumbered. An ffmpeg refuses to
start with an unconnected filter output, so the pads have to go with the nodes.
"""

from __future__ import annotations

import heapq
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Literal

from .errors import ErrorCode, FfrwdError
from .ir import (
    PIPE,
    ROWFILTER,
    FrameRef,
    Graph,
    Node,
    Output,
    RowsSink,
    SinkUnit,
    StreamType,
    is_src,
    src_parts,
)
from .probe import ProbeResult, StreamMeta, is_url

__all__ = [
    "COPY_CODEC",
    "FIFO",
    "NUT",
    "PCM_F32LE",
    "PCM_S16LE",
    "PIPE",
    "PIPE_BUFFER_LIMIT",
    "RAWVIDEO",
    "SAFETY",
    "AudioFormat",
    "Edge",
    "EdgeBuffer",
    "EffectGrant",
    "FileContent",
    "FfmpegProcess",
    "FileEdge",
    "FileFormat",
    "IsExternal",
    "ModelBinding",
    "ModuleBinding",
    "ModuleShape",
    "Process",
    "ProcessPlan",
    "RowsEdge",
    "SidecarProcess",
    "Stage",
    "StreamEdge",
    "StreamFormat",
    "VideoFormat",
    "external_filters",
    "external_ids",
    "from_commands",
    "is_live",
    "is_live_probe",
    "nothing_external",
    "check_spellable",
    "partition",
]

# The container every raw stream edge is wrapped in, and the codecs inside it.
NUT = "nut"
RAWVIDEO = "rawvideo"
PCM_F32LE = "pcm_f32le"
PCM_S16LE = "pcm_s16le"

# ffmpeg's own spelling for a stream handed on untouched: what an edge carries
# when nothing on the far side filters it, so the packets cross as they came.
COPY_CODEC = "copy"

# ffprobe reports no pixel format, so the wire format is one the compiler
# picks and the producing ffmpeg is told to write.
DEFAULT_PIX_FMT = "yuv420p"

# The filters whose output pads are interchangeable copies of one input, and so
# the ones a payload can cut down to the consumers it actually holds.
SPLIT_FILTERS = frozenset({"split", "asplit"})

# The muxer that queues packets ahead of the real one, and the options that
# name what it wraps and how deep it goes.
FIFO = "fifo"
FIFO_FORMAT = "fifo_format"
QUEUE_SIZE = "queue_size"

# How much bigger than the computed bound a buffer is actually made. A bound
# counts the frames one path runs ahead of its siblings; the doubling covers
# the frame each process holds while it works on it.
SAFETY = 2

# The most a named pipe's own buffer is sized to. Above this the fifo muxer
# queues in the producing ffmpeg's own memory instead, where a large queue
# costs ordinary heap rather than the kernel's.
PIPE_BUFFER_LIMIT = 4 << 20

# The smallest a named pipe's buffer is made, and the step it is rounded up by.
PIPE_BUFFER_STEP = 1 << 16

# Bytes one pixel takes on the wire, per pixel format ffrwd carries.
_PIXEL_BYTES: Mapping[str, int] = {"rgba": 4, "rgb24": 3, "yuv420p": 2, "yuva420p": 3}

# Bytes one sample of one channel takes, per pcm codec.
_SAMPLE_BYTES: Mapping[str, int] = {PCM_F32LE: 4, PCM_S16LE: 2}

# Filters whose output holds a different number of frames than their input,
# and by a margin nothing at compile time counts. A path through one of these
# drifts from its siblings without limit.
RATE_CHANGING_FILTERS = frozenset(
    {
        "aloop",
        "areverse",
        "aselect",
        "asetrate",
        "atempo",
        "atrim",
        "concat",
        "decimate",
        "fps",
        "framerate",
        "framestep",
        "interleave",
        "ainterleave",
        "loop",
        "minterpolate",
        "mpdecimate",
        "reverse",
        "select",
        "thumbnail",
        "tile",
        "trim",
    }
)

# Filters that read a window of frames to produce one: the option naming the
# window, and ffmpeg's own default for it. The delay is that window less one;
# an empty option name is a filter whose window is fixed.
FILTER_WINDOWS: Mapping[str, tuple[str, int]] = {
    "deflicker": ("size", 5),
    "tblend": ("", 2),
    "tmix": ("frames", 3),
}

# The frames a decoder holds back before it can emit its first, for an edge
# carrying encoded packets rather than raw frames. Conservative: it covers the
# reorder depth of every encoder ffrwd puts on an edge without being told the
# actual b-frame count.
ENCODED_EDGE_DELAY = 8

# Nodes the sidecar hosts itself. They belong to a region the way a module
# does -- ffmpeg cannot run them -- but no ``-m`` entry binds their name.
HOSTED_FILTERS = frozenset({ROWFILTER})

# What a module's bound name may not contain: a network string names modules
# where a filtergraph names filters.
_MODULE_NAME_UNSAFE = re.compile(r"[^A-Za-z0-9_]")

# An input carried in the query text itself rather than opened from anywhere.
_DATA_URI = "data:"

# The stream types a pipe carries between processes. A subtitle or data track
# is a bare ``-map`` and NUT has nowhere to put it.
_PIPED_TYPES: frozenset[StreamType] = frozenset({"video", "audio"})


# ---------------------------------------------------------------- formats


@dataclass(frozen=True)
class VideoFormat:
    """One video stream on a pipe: NUT carrying rawvideo frames.

    `width`, `height` and `timebase` are what the probe said, and None where
    it said nothing -- NUT's header carries them at run time either way.
    `pix_fmt` is not an observation: ffprobe reports none, so this is the
    format the producing ffmpeg is told to write.

    The edge into a PACKET SINK carries the encoder's output instead: `codec`
    is then the encoder, and `options` the validated sink options shaping it
    (crf, preset and kin), rendered on the producing ffmpeg's output.
    """

    pix_fmt: str = DEFAULT_PIX_FMT
    width: int | None = None
    height: int | None = None
    timebase: str | None = None
    container: str = NUT
    codec: str = RAWVIDEO
    options: tuple[tuple[str, object], ...] = ()

    @property
    def size(self) -> str | None:
        """``"<width>x<height>"``, ffmpeg's spelling, or None if unprobed."""
        if self.width is None or self.height is None:
            return None
        return f"{self.width}x{self.height}"

    def to_dict(self) -> dict[str, object]:
        written: dict[str, object] = {
            "container": self.container,
            "codec": self.codec,
            "pix_fmt": self.pix_fmt,
            "width": self.width,
            "height": self.height,
            "timebase": self.timebase,
        }
        if self.options:
            written["options"] = dict(self.options)
        return written


@dataclass(frozen=True)
class AudioFormat:
    """One audio stream on a pipe: NUT carrying pcm samples.

    `rate` and `channels` are what the probe said, None where it said nothing.
    `codec` is the pcm the negotiated sample format travels as, the audio
    counterpart of :attr:`VideoFormat.pix_fmt` and prescriptive the same way.

    `required_rate` and `required_channels` are what the MODULE on this edge
    accepts, and are what the producing ffmpeg is told to write; None where
    nothing on the edge names one, and the stream then travels as it is.

    The edge into a PACKET SINK carries the encoder's output instead: `codec`
    is then the audio encoder, and `options` the validated sink options
    shaping it, the way :attr:`VideoFormat.options` are.
    """

    rate: int | None = None
    channels: int | None = None
    container: str = NUT
    codec: str = PCM_F32LE
    required_rate: int | None = None
    required_channels: int | None = None
    options: tuple[tuple[str, object], ...] = ()

    def to_dict(self) -> dict[str, object]:
        written: dict[str, object] = {
            "container": self.container,
            "codec": self.codec,
            "rate": self.rate,
            "channels": self.channels,
        }
        if self.options:
            written["options"] = dict(self.options)
        if self.required_rate is not None:
            written["required_rate"] = self.required_rate
        if self.required_channels is not None:
            written["required_channels"] = self.required_channels
        return written


StreamFormat = VideoFormat | AudioFormat

# A media artifact, or rows -- one JSON object per line.
FileContent = Literal["media", "rows"]

# Where a stream edge's depth is held: the named pipe's own buffer, or the
# fifo muxer's queue inside the producing ffmpeg.
BufferRoad = Literal["pipe", "fifo"]


@dataclass(frozen=True)
class FileFormat:
    """A file one process hands the next.

    `path` is None when the handoff has no file of its own and the edge only
    says which process runs first.
    """

    content: FileContent = "media"
    path: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {"content": self.content, "path": self.path}


# ---------------------------------------------------------------- edges


@dataclass(frozen=True)
class EdgeBuffer:
    """How deep one stream edge is buffered, and where the depth is held.

    `road` is ``"pipe"`` when the named pipe's own buffer is sized to
    `size` bytes, and ``"fifo"`` when the producing ffmpeg queues `packets`
    of them ahead of the muxer instead. `frames` is what the two were sized
    from: the edge's bound times :data:`SAFETY`.
    """

    road: BufferRoad
    frames: int
    size: int = 0
    packets: int = 0

    def to_dict(self) -> dict[str, object]:
        written: dict[str, object] = {"road": self.road, "frames": self.frames}
        if self.road == "pipe":
            written["size"] = self.size
        else:
            written["packets"] = self.packets
        return written


@dataclass(frozen=True)
class StreamEdge:
    """One stream on a pipe, from one process to another.

    `ref` is the logical graph ref this stream carries, which is how the
    consuming process's payload names it.

    `annotations` marks an edge whose frames travel with the rows the
    producing module read off them. Only an edge between two sidecars is ever
    one -- ffmpeg neither writes nor reads the annotation stream -- and it is
    what puts ``-annotations`` on both ends.

    `bound` is how many frames this edge must hold while its sibling edges'
    paths reach the process they all meet at again: the difference between the
    slowest sibling's path delay and this one's, in frames. It is 0 for an
    edge with no sibling to outrun, which is every edge of a plan whose inputs
    are all plain files. `buffer` is what the bound was turned into, and None
    where the bound is 0 and the transport's own defaults suffice.
    """

    source: str
    target: str
    ref: FrameRef
    format: StreamFormat
    annotations: bool = False
    bound: int = 0
    buffer: EdgeBuffer | None = None

    def to_dict(self) -> dict[str, object]:
        written: dict[str, object] = {
            "kind": "stream",
            "source": self.source,
            "target": self.target,
            "ref": self.ref,
            "format": self.format.to_dict(),
        }
        if self.annotations:
            written["annotations"] = True
        if self.bound:
            written["bound"] = self.bound
        if self.buffer is not None:
            written["buffer"] = self.buffer.to_dict()
        return written


@dataclass(frozen=True)
class FileEdge:
    """A file handed from one process to another, ordering the two."""

    source: str
    target: str
    format: FileFormat

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": "file",
            "source": self.source,
            "target": self.target,
            "format": self.format.to_dict(),
        }


@dataclass(frozen=True)
class RowsEdge:
    """The rows one sidecar writes, read by another process as an input.

    `alias` is the compiler-minted ``-i`` of the reading process that the
    track arrives on, which is what says WHICH pipe slot of that process this
    edge fills. `container` is what the sidecar writes and the reader reads.
    """

    source: str
    target: str
    alias: str
    container: str

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": "rows",
            "source": self.source,
            "target": self.target,
            "alias": self.alias,
            "container": self.container,
        }


Edge = StreamEdge | FileEdge | RowsEdge


# ---------------------------------------------------------------- processes


@dataclass(frozen=True)
class FfmpegProcess:
    """One ffmpeg invocation, described by a graph of its own.

    `graph` is a complete :class:`~ffrwd.ir.Graph`: an incoming stream edge is
    one of its inputs, at :data:`PIPE`, and an outgoing stream edge is one of
    its sinks, also at :data:`PIPE`. The edge, not the payload, is where the
    wire format lives.
    """

    id: str
    graph: Graph

    def to_dict(self) -> dict[str, object]:
        return {"id": self.id, "kind": "ffmpeg", "graph": self.graph.to_dict()}


@dataclass(frozen=True)
class ModuleShape:
    """What one module declares about its own frame timing.

    `window` is how many frames the export reads to produce one, `stride` how
    far it advances between them, `one_to_one` whether it emits exactly one
    frame per frame it is handed -- the property a multi-input module's inputs
    need in common to run in lockstep -- and `pure` whether a call depends only
    on the frames it was handed, which is what lets several instances of it run
    at once. A module that declares none of them takes these defaults, which is
    what a plain filter module does.
    """

    window: int = 1
    stride: int = 1
    one_to_one: bool = True
    pure: bool = True

    @property
    def lookahead(self) -> int:
        """Frames this module reads ahead before it can emit its first."""
        return max(self.window - 1, 0)


@dataclass(frozen=True)
class ModuleBinding:
    """One ``-m <name>=<path>``: the name a network names, and what it loads."""

    name: str
    path: str

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "path": self.path}


@dataclass(frozen=True)
class ModelBinding:
    """One ``-nn <name>=<path>``: the name a module loads a model by, and the file."""

    name: str
    path: str

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "path": self.path}


@dataclass(frozen=True)
class EffectGrant:
    """One ``-http <path>`` or ``-net <path>``: an effect one module is allowed.

    `effect` is the capability name -- ``http`` for outbound HTTP requests,
    ``udp`` for UDP sockets -- and `module` the path of the module granted
    it. The sidecar denies both by default; the argv is the grant.
    """

    effect: str
    module: str

    def to_dict(self) -> dict[str, object]:
        return {"effect": self.effect, "module": self.module}


@dataclass(frozen=True)
class SidecarProcess:
    """One REGION of wasm modules the sidecar hosts, reading and writing pipes.

    `graph` is the region's own :class:`~ffrwd.ir.Graph`, exactly as an
    :class:`FfmpegProcess` carries one: an incoming stream edge is one of its
    inputs, at :data:`PIPE`, and an outgoing edge one of its sinks, also at
    :data:`PIPE`. Its nodes call modules by the NAME `modules` binds, not by
    path, so the graph renders as a filtergraph and the ``-m`` table says what
    each name loads. `lookahead` is the region's total declared latency: the
    largest sum of per-module lookahead along any path through it.

    `module` is the path the region's ENTRY module loads and `node` the
    logical node it came from -- the whole region for the common case of one
    module, which keeps its simpler argv. `inputs` are the refs the region
    reads from outside, in read order, and `outputs` the stream types it hands
    back out, one outgoing stream edge each.

    `reads_rows` and `writes_rows` are which sides of this process carry
    annotations, read off the edges around it: the argv the sidecar is run
    with says so, and an edge with ffmpeg at the far end never does. Rows
    between two modules of ONE region never appear here -- they never leave
    the process.

    `rows` is the region's ROWS output, for a query that selects a module's
    annotation column: the rows leave as a document of their own rather than
    riding frames, and the region's frames end at the module that read them.

    `impure` names the modules of this region that declared they carry state
    between calls, in region order. It is empty for a region every module of
    which is pure. The sidecar's own scheduler is what acts on purity: an
    impure module's calls run one at a time in order, and a pure one spreads
    across the worker pool, whatever this plan says.
    """

    id: str
    module: str
    node: str
    args: dict[str, object] = field(default_factory=dict)
    inputs: tuple[FrameRef, ...] = ()
    outputs: tuple[StreamType, ...] = ()
    reads_rows: bool = False
    writes_rows: bool = False
    graph: Graph | None = None
    modules: tuple[ModuleBinding, ...] = ()
    models: tuple[ModelBinding, ...] = ()
    grants: tuple[EffectGrant, ...] = ()
    lookahead: int = 0
    rows: RowsSink | None = None
    impure: tuple[str, ...] = ()
    # True for a region holding a SINK MODULE: the region consumes its pipes
    # and writes nothing back into the pipeline -- its stream output is a
    # null output, and the module's own effects are the product.
    sink: bool = False
    # True for a region whose module consumes ENCODED PACKETS rather than
    # frames.
    packet_sink: bool = False

    @property
    def nodes(self) -> tuple[str, ...]:
        """The logical nodes this region holds, in dependency order."""
        return (self.node,) if self.graph is None else tuple(self.graph.nodes)

    @property
    def network(self) -> bool:
        """True when this process needs the filtergraph spelling.

        More than one module node, or one FRAME module reading more than one
        stream: the short ``-m <path>`` form says one module, and its pads
        come out of one input wired by the network string.

        A packet sink is the exception, and never a network: its pads are
        whole encoded streams rather than pads cut out of one, so the short
        form spells them as the inputs they are, one ``-i`` apiece.
        """
        if self.graph is None:
            return False
        if self.packet_sink:
            return False
        return len(self.graph.nodes) > 1 or any(
            len(node.inputs) > 1 for node in self.graph.nodes.values()
        )

    def to_dict(self) -> dict[str, object]:
        written: dict[str, object] = {
            "id": self.id,
            "kind": "sidecar",
            "module": self.module,
            "node": self.node,
            "args": dict(self.args),
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "modules": [binding.to_dict() for binding in self.modules],
            "lookahead": self.lookahead,
        }
        if self.models:
            written["models"] = [binding.to_dict() for binding in self.models]
        if self.grants:
            written["grants"] = [grant.to_dict() for grant in self.grants]
        if self.reads_rows:
            written["reads_rows"] = True
        if self.writes_rows:
            written["writes_rows"] = True
        if self.rows is not None:
            written["rows"] = self.rows.to_dict()
        if self.sink:
            written["sink"] = True
        if self.network and self.graph is not None:
            written["graph"] = self.graph.to_dict()
        return written


Process = FfmpegProcess | SidecarProcess


@dataclass(frozen=True)
class Stage:
    """Processes that run at the same time, and their place in the order."""

    index: int
    processes: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {"index": self.index, "processes": list(self.processes)}


@dataclass(frozen=True)
class ProcessPlan:
    """Every process one query needs, and what runs between them."""

    processes: tuple[Process, ...] = ()
    edges: tuple[Edge, ...] = ()

    @property
    def ffmpeg(self) -> tuple[FfmpegProcess, ...]:
        """The ffmpeg processes, in plan order."""
        return tuple(p for p in self.processes if isinstance(p, FfmpegProcess))

    @property
    def sidecars(self) -> tuple[SidecarProcess, ...]:
        """The sidecar processes, in plan order."""
        return tuple(p for p in self.processes if isinstance(p, SidecarProcess))

    @property
    def stream_edges(self) -> tuple[StreamEdge, ...]:
        return tuple(e for e in self.edges if isinstance(e, StreamEdge))

    @property
    def file_edges(self) -> tuple[FileEdge, ...]:
        return tuple(e for e in self.edges if isinstance(e, FileEdge))

    @property
    def rows_edges(self) -> tuple[RowsEdge, ...]:
        return tuple(e for e in self.edges if isinstance(e, RowsEdge))

    def process(self, id: str) -> Process:
        """The process with this id."""
        for candidate in self.processes:
            if candidate.id == id:
                return candidate
        raise KeyError(id)

    @property
    def stages(self) -> tuple[Stage, ...]:
        """The processes grouped into what runs together, in run order.

        Processes stream edges connect are one stage: they run at the same
        time, passing frames as they are produced -- and a rows edge is a pipe
        like any other, so it groups the same way. File edges put one stage
        after another. A file edge inside a stage says nothing about order and
        is skipped.
        """
        return _stages(self.processes, self.edges)

    def to_dict(self) -> dict[str, object]:
        return {
            "processes": [p.to_dict() for p in self.processes],
            "edges": [e.to_dict() for e in self.edges],
            "stages": [s.to_dict() for s in self.stages],
        }


# ---------------------------------------------------------------- marking

# How a node ffmpeg cannot host enters the partition. No SQL surface produces
# one yet, so a caller -- today, a test -- supplies the predicate.
IsExternal = Callable[[Node], bool]


def nothing_external(node: Node) -> bool:
    """No node is external: the graph partitions to one ffmpeg process."""
    return False


def external_ids(*ids: str) -> IsExternal:
    """Mark the nodes with these ids external."""
    marked = frozenset(ids)
    return lambda node: node.id in marked


def external_filters(*names: str) -> IsExternal:
    """Mark every node calling one of these filters external."""
    marked = frozenset(names)
    return lambda node: node.filter in marked


# ---------------------------------------------------------------- ref helpers


def _ref_node(ref: FrameRef) -> str | None:
    """The node id `ref` names, or None for a source ref."""
    if is_src(ref):
        return None
    node_id, _, pad = ref.rpartition(":")
    return node_id if node_id and pad.isdigit() else ref


def _ref_pad(ref: FrameRef) -> int:
    """The output pad `ref` names; 0 for the unqualified form."""
    _, _, pad = ref.rpartition(":")
    return int(pad) if pad.isdigit() else 0


def ref_type(g: Graph, ref: FrameRef) -> StreamType:
    """The stream type `ref` carries in `g`."""
    if is_src(ref):
        return src_parts(ref)[1]
    node_id = _ref_node(ref)
    node = g.nodes.get(node_id) if node_id is not None else None
    if node is None:
        return "video"
    pad = _ref_pad(ref)
    return node.outputs[pad] if pad < len(node.outputs) else "video"


def _unique_alias(ref: FrameRef, taken: Iterable[str]) -> str:
    """An input alias for a piped `ref` that collides with nothing."""
    used = set(taken)
    alias = ref.replace(":", "_")
    while alias in used:
        alias += "_"
    return alias


def is_live(path: str, options: Mapping[str, object] | None = None) -> bool:
    """True when this input cannot be opened a second time.

    A protocol spec -- srt, udp, rtmp, rtsp -- names a socket, and an input
    whose demuxer the query named itself (``format => 'dshow'``,
    ``format => 'lavfi'``) names a device or a graph rather than a file. A
    plain file path is neither and reads as many times as it is asked to. A
    ``data:`` document is in memory and reads twice for nothing.
    """
    if path.startswith(_DATA_URI):
        return False
    return is_url(path) or "format" in (options or {})


def is_live_probe(result: ProbeResult | None) -> bool:
    """True when a probed manifest is still live: no end, no second read.

    :func:`is_live` answers from a path and options alone, which a plain HLS
    or DASH URL cannot settle -- only ffprobe knows whether the manifest it
    read carries ``#EXT-X-ENDLIST`` or ``type="dynamic"``.
    """
    return result is not None and result.live


def _named_ref(ref: FrameRef) -> str:
    """A stream ref as a message names it: its input, or the node it comes from."""
    if is_src(ref):
        alias, kind, index = src_parts(ref)
        return f"{alias}.{kind}[{index + 1}]"
    return f"the output of '{_ref_node(ref)}'"


def _bindings(paths: Iterable[str]) -> tuple[ModuleBinding, ...]:
    """One ``-m`` entry per distinct module path, named after the file.

    The name is what a filtergraph can spell -- the module's own file name,
    with anything outside ``[A-Za-z0-9_]`` replaced -- made unique inside the
    process, since two modules of one region may share a file name.
    """
    bound: dict[str, ModuleBinding] = {}
    used: set[str] = set()
    for path in paths:
        if path in bound:
            continue
        base = _MODULE_NAME_UNSAFE.sub("_", path.replace("\\", "/").rpartition("/")[2])
        base = base.removesuffix("_wasm") or "module"
        while base in used or base[0].isdigit():
            base += "_"
        used.add(base)
        bound[path] = ModuleBinding(name=base, path=path)
    return tuple(bound.values())


def _encoded(wire: StreamFormat) -> bool:
    """True when this edge carries packets rather than raw frames."""
    if isinstance(wire, AudioFormat):
        return wire.codec not in (PCM_F32LE, PCM_S16LE)
    return wire.codec != RAWVIDEO


def _encodes(wire: StreamFormat) -> bool:
    """True when the producing process ENCODES onto this edge.

    A copied stream is not one: its packets were encoded before this run and
    cross untouched, so nothing holds frames back to reorder them.
    """
    return _encoded(wire) and wire.codec != COPY_CODEC


def _frame_bytes(wire: StreamFormat) -> int | None:
    """One frame's size on this edge, or None where it cannot be counted.

    Raw video is width by height by the pixel format's own byte count. An
    encoded packet's size is not its frame's, and an audio packet holds
    however many samples the muxer put in it: neither has an answer here.
    """
    if isinstance(wire, AudioFormat) or _encoded(wire):
        return None
    if wire.width is None or wire.height is None:
        return None
    per_pixel = _PIXEL_BYTES.get(wire.pix_fmt)
    return None if per_pixel is None else wire.width * wire.height * per_pixel


def _rounded(size: int) -> int:
    """`size` rounded up to whole buffer steps, and never below one."""
    steps = max((size + PIPE_BUFFER_STEP - 1) // PIPE_BUFFER_STEP, 1)
    return steps * PIPE_BUFFER_STEP


def _timebase(fps: str | None) -> str | None:
    """A frame rate inverted, one tick per frame; None when there is no rate."""
    if fps is None:
        return None
    numerator, _, denominator = fps.partition("/")
    try:
        num = int(numerator)
        den = int(denominator) if denominator else 1
    except ValueError:
        return None
    if num == 0 or den == 0:
        return None
    return f"{den}/{num}"


def _topological(g: Graph) -> list[str]:
    """Node ids in dependency order, ties broken by insertion order.

    A graph already in topological order comes back in exactly its own order,
    which is what keeps a payload byte-equal to the graph it came from.
    """
    names = list(g.nodes)
    position = {name: index for index, name in enumerate(names)}
    pending = dict.fromkeys(names, 0)
    consumers: dict[str, list[str]] = {}
    for name, node in g.nodes.items():
        for ref in node.inputs:
            producer = _ref_node(ref)
            if producer is None or producer not in g.nodes or producer == name:
                continue
            pending[name] += 1
            consumers.setdefault(producer, []).append(name)

    ready = [position[name] for name in names if pending[name] == 0]
    heapq.heapify(ready)
    order: list[str] = []
    while ready:
        name = names[heapq.heappop(ready)]
        order.append(name)
        for consumer in consumers.get(name, []):
            pending[consumer] -= 1
            if pending[consumer] == 0:
                heapq.heappush(ready, position[consumer])

    # A cycle is malformed IR; those nodes keep their insertion order.
    placed = set(order)
    order.extend(name for name in names if name not in placed)
    return order


# ---------------------------------------------------------------- the pass


@dataclass
class _Pending:
    """One ffmpeg process before its edges -- and so its inputs -- are known."""

    id: str
    depth: int
    nodes: list[str]
    sinks: list[SinkUnit]
    pipes: list[FrameRef]


class _Partitioner:
    """One run of :func:`partition`."""

    def __init__(
        self,
        g: Graph,
        external: IsExternal,
        probes: Mapping[str, ProbeResult | None] | None,
        pix_fmts: Mapping[str, str] | None = None,
        shapes: Mapping[str, ModuleShape] | None = None,
        audio_wires: Mapping[str, AudioFormat] | None = None,
        models: Mapping[str, ModelBinding] | None = None,
        effects: Mapping[str, tuple[str, ...]] | None = None,
        anchors: Mapping[str, tuple[int, int]] | None = None,
    ) -> None:
        self.g = g
        self.probes = probes or {}
        self.pix_fmts = pix_fmts or {}
        self.shapes = shapes or {}
        self.audio_wires = audio_wires or {}
        self.models = models or {}
        self.effects = effects or {}
        self.anchors = anchors or {}
        self.live = frozenset(
            alias
            for alias, index in g.sources.items()
            if index < len(g.input_paths)
            and (
                is_live(g.input_paths[index], g.input_options.get(alias))
                or is_live_probe(self.probes.get(alias))
            )
        )
        self.order = _topological(g)
        # A hosted node is external whoever asked: only the sidecar runs it.
        self.external = {
            name: bool(external(g.nodes[name]))
            or g.nodes[name].filter in HOSTED_FILTERS
            for name in self.order
        }
        self.depth: dict[str, int] = {}
        self._compute_depths()
        self.sink_depth = [self._unit_depth(unit) for unit in g.sinks]
        self.pending: list[_Pending] = []
        self.sidecars: list[SidecarProcess] = []
        self.sidecar_of: dict[str, str] = {}  # node id -> process id
        self.members: dict[str, list[str]] = {}  # process id -> its node ids
        self.consumer_of: dict[str, str] = {}  # feeder process id -> its reader
        self.edges: list[StreamEdge] = []
        # Kept apart from the stream edges: everything reading `edges` reads
        # frame pipes, and rows are neither frames nor a ref.
        self.rows: list[RowsEdge] = []

    # -- depths

    def _ref_depth(self, ref: FrameRef) -> int:
        producer = _ref_node(ref)
        if producer is None or producer not in self.depth:
            return 0
        return self.depth[producer] + (1 if self.external[producer] else 0)

    def _compute_depths(self) -> None:
        """Fill :attr:`depth` in dependency order; each node reads its inputs'."""
        for name in self.order:
            node = self.g.nodes[name]
            self.depth[name] = max(
                (self._ref_depth(ref) for ref in node.inputs),
                default=0,
            )

    def _unit_depth(self, unit: SinkUnit) -> int:
        return max((self._ref_depth(o.ref) for o in unit.outputs), default=0)

    # -- process construction

    def _ffmpeg_id(self) -> str:
        return f"ffmpeg{len(self.pending)}"

    def _ancestors(self, refs: Iterable[FrameRef], depth: int) -> list[str]:
        """Node ids at `depth` that `refs` need, in topological order."""
        keep: set[str] = set()
        stack = list(refs)
        while stack:
            name = _ref_node(stack.pop())
            if name is None or name not in self.g.nodes or name in keep:
                continue
            if self.external[name] or self.depth[name] != depth:
                continue
            keep.add(name)
            stack.extend(self.g.nodes[name].inputs)
        return [name for name in self.order if name in keep]

    def _consumed(self, process: _Pending) -> list[FrameRef]:
        """Refs `process` must be handed by another process, in read order."""
        inside = set(process.nodes)
        refs: list[FrameRef] = []
        for name in process.nodes:
            refs.extend(self.g.nodes[name].inputs)
        for unit in process.sinks:
            refs.extend(o.ref for o in unit.outputs)
        refs.extend(process.pipes)
        wanted: list[FrameRef] = []
        for ref in refs:
            producer = _ref_node(ref)
            if producer is None or producer in inside:
                continue  # a file this process opens itself, or its own node
            if ref not in wanted:
                wanted.append(ref)
        return wanted

    def _sibling_feeder(
        self, target: str, depth: int, nodes: Sequence[str], ref: FrameRef
    ) -> _Pending | None:
        """The feeder this leg joins instead of becoming a process of its own.

        Raw streams ONE consumer reads may leave one producer: that consumer
        takes them at its own interleaving, so neither branch can outrun the
        other and the fan is deadlock-free. The legs must read exactly alike
        -- one consumer, one depth, the same input aliases and the same refs
        handed over -- so joining them changes nothing about what is opened
        or consumed, and the join adds no demand of its own.

        A SIDECAR consumer joins them like any other: a sink reads each pad
        on a reader of its own, so no pad's pace holds another's back and the
        fan is deadlock-free. Joining is what makes a rendition ladder one
        decode feeding N encodes rather than N decodes of the same input.
        """
        leg = self._feeder_reads(nodes, [ref])
        for process in self.pending:
            if self.consumer_of.get(process.id) != target or process.depth != depth:
                continue
            if self._feeder_reads(process.nodes, process.pipes) == leg:
                return process
        return None

    def _feeder_reads(
        self, nodes: Iterable[str], pipes: Iterable[FrameRef]
    ) -> tuple[frozenset[str], frozenset[FrameRef]]:
        """What one feeder reads: its input aliases, and the refs handed to it.

        The alias, not the path: one alias is one ``-i``, carrying its own
        seek window and input options, so equal aliases are equal opens.

        `pipes` are the refs the feeder hands over. One a node inside
        produced adds nothing the nodes did not already say; a BARE source
        ref has no producing node, and its alias is a read all the same --
        an audio track mapped straight off the input reads the input exactly
        as a scaled video leg does.
        """
        inside = set(nodes)
        aliases: set[str] = set()
        outside: set[FrameRef] = set()
        for name in inside:
            for ref in self.g.nodes[name].inputs:
                producer = _ref_node(ref)
                if producer is None:
                    aliases.add(src_parts(ref)[0])
                elif producer not in inside:
                    outside.add(ref)
        for ref in pipes:
            producer = _ref_node(ref)
            if producer is None:
                aliases.add(src_parts(ref)[0])
            elif producer not in inside:
                outside.add(ref)
        return frozenset(aliases), frozenset(outside)

    # -- one reader per live input

    def _live_reader(
        self, depth: int, nodes: Sequence[str], ref: FrameRef
    ) -> _Pending | None:
        """The reader this leg joins because both read one live input.

        A live input is opened by exactly ONE process, so a second leg over it
        joins the first rather than opening it again -- and the joined reader
        then hands each consumer a pipe of its own. Both sides must read
        nothing but inputs: a process that reads a pipe may sit downstream of
        the one it would join, and the join would make the process DAG a cycle.
        """
        aliases, outside = self._feeder_reads(nodes, [ref])
        wanted = aliases & self.live
        if not wanted or outside:
            return None
        for process in self.pending:
            if process.depth != depth:
                continue
            theirs, others = self._feeder_reads(process.nodes, process.pipes)
            if others or not theirs & wanted:
                continue
            return process
        return None

    def _opened(self, process: _Pending) -> dict[str, list[FrameRef]]:
        """The input aliases `process` opens itself, and the refs it reads off each.

        Only video and audio: a subtitle or data track never travels a pipe,
        so it is not something a reader could hand over.
        """
        found: dict[str, list[FrameRef]] = {}
        refs: list[FrameRef] = [ref for name in process.nodes
                                for ref in self.g.nodes[name].inputs]
        for unit in process.sinks:
            refs.extend(o.ref for o in unit.outputs)
        refs.extend(process.pipes)
        for ref in refs:
            if not is_src(ref) or ref_type(self.g, ref) not in _PIPED_TYPES:
                continue
            alias = src_parts(ref)[0]
            if ref not in found.setdefault(alias, []):
                found[alias].append(ref)
        return found

    def _reads(self, process: _Pending, alias: str) -> list[FrameRef]:
        """Every stream of `alias` this process opens itself, of any type."""
        refs: list[FrameRef] = [ref for name in process.nodes
                                for ref in self.g.nodes[name].inputs]
        for unit in process.sinks:
            refs.extend(o.ref for o in unit.outputs)
        refs.extend(process.pipes)
        found: list[FrameRef] = []
        for ref in refs:
            if is_src(ref) and src_parts(ref)[0] == alias and ref not in found:
                found.append(ref)
        return found

    def _unpiped(self, process: _Pending, alias: str) -> FrameRef | None:
        """A stream of `alias` this process reads that no pipe could carry."""
        return next(
            (
                ref
                for ref in self._reads(process, alias)
                if ref_type(self.g, ref) not in _PIPED_TYPES
            ),
            None,
        )

    def _redirect_live_reads(self) -> None:
        """Leave one process opening each live input; the rest read its pipes.

        A process that would open a live input another already reads has that
        read turned into a stream edge off the reader. Where no process reads
        the input on its own -- every one of them also reads a pipe, so none
        may be joined without risking a cycle -- a reader of nothing else is
        made for it.
        """
        for alias in sorted(self.live):
            readers = [p for p in self.pending if self._reads(p, alias)]
            if len(readers) < 2:
                continue
            reader = next((p for p in readers if not self._consumed(p)), None)
            if reader is None:
                reader = _Pending(
                    id=self._ffmpeg_id(), depth=0, nodes=[], sinks=[], pipes=[]
                )
                self.pending.append(reader)
            for process in readers:
                if process is reader:
                    continue
                stranded = self._unpiped(process, alias)
                if stranded is not None:
                    raise self._live_refusal(
                        alias,
                        f"so one process reads it and hands every other one a "
                        f"pipe -- and {_named_ref(stranded)} is read by one of "
                        "those others, where a subtitle or data track cannot "
                        "follow",
                        hint="a pipe between two processes carries pictures and "
                        "sound and nothing else: map that track from a separate "
                        "input() over a recording, or drop it from the SELECT",
                    )
                for ref in self._opened(process).get(alias, []):
                    if ref not in reader.pipes:
                        reader.pipes.append(ref)
                    self._add_edge(
                        reader.id, process.id, ref, copy=not self._filters(process, ref)
                    )

    def _filters(self, process: _Pending, ref: FrameRef) -> bool:
        """True when a node inside `process` reads `ref` rather than mapping it."""
        return any(ref in self.g.nodes[name].inputs for name in process.nodes)

    def _live_refusal(self, alias: str, why: str, *, hint: str) -> FfrwdError:
        """The rejection for a live input this compiler cannot wire safely."""
        index = self.g.sources.get(alias)
        path = self.g.input_paths[index] if index is not None else alias
        line, col = self.anchors.get(alias, (None, None))
        return FfrwdError(
            ErrorCode.UNBOUNDED_LIVE_INPUT,
            f"'{path}' can only be opened once, {why}",
            line=line,
            col=col,
            hint=hint,
        )

    # -- regions

    def _reachable(self) -> dict[str, frozenset[str]]:
        """Each node's descendants, computed once for the convexity test."""
        consumers: dict[str, list[str]] = {}
        for name, node in self.g.nodes.items():
            for ref in node.inputs:
                producer = _ref_node(ref)
                if producer is not None and producer in self.g.nodes:
                    consumers.setdefault(producer, []).append(name)
        found: dict[str, frozenset[str]] = {}
        for name in reversed(self.order):  # a node's consumers are already done
            below: set[str] = set()
            for consumer in consumers.get(name, []):
                below.add(consumer)
                below |= found.get(consumer, frozenset())
            found[name] = frozenset(below)
        return found

    def _convex(self, members: Iterable[str], reach: Mapping[str, frozenset[str]]) -> bool:
        """True when no path leaves `members` and comes back into it.

        A region an outside node sits inside cannot be one process: the
        process feeding that node would have to run after the process reading
        it back.
        """
        inside = set(members)
        below: set[str] = set()
        for name in inside:
            below |= reach[name]
        below -= inside
        return not any(reach[name] & inside for name in below)

    def _pad_consumers(self) -> dict[str, list[str] | None]:
        """Each node's consumers, or None where a sink output reads a pad of it.

        A sink output is a consumer no region can hold, so a split feeding one
        cannot be absorbed into a region and is marked here.
        """
        reading: dict[str, list[str]] = {}
        for name, node in self.g.nodes.items():
            for ref in node.inputs:
                producer = _ref_node(ref)
                if producer is not None and producer in self.g.nodes:
                    reading.setdefault(producer, []).append(name)
        sunk = {
            producer
            for unit in self.g.sinks
            for output in unit.outputs
            if (producer := _ref_node(output.ref)) is not None
        }
        return {
            name: None if name in sunk else found for name, found in reading.items()
        }

    def _regions(self) -> list[list[str]]:
        """Maximal convex groups of adjacent external nodes, in graph order.

        Two external nodes join when one reads the other directly. A `split`
        between them joins too, and dissolves when the region is built: the
        network hands one module's frames to several, so a region's own
        fan-out needs no split node at all. Every merge is taken only while
        the group stays convex, and the whole thing runs to a fixed point --
        merging two groups can make a third mergeable.
        """
        reach = self._reachable()
        consumers = self._pad_consumers()
        home = {name: name for name in self.order if self.external[name]}
        groups = {name: [name] for name in home}

        def join(*names: str) -> bool:
            """Merge the groups these nodes are in, if the result stays convex."""
            reps = {home.get(name) for name in names}
            if None not in reps and len(reps) == 1:
                return False  # already one group
            wanted: set[str] = set()
            for name in names:
                wanted |= set(groups[home[name]]) if name in home else {name}
            joined = [n for n in self.order if n in wanted]
            if not self._convex(joined, reach):
                return False
            for rep in reps:
                if rep is not None:
                    groups.pop(rep, None)
            groups[joined[0]] = joined
            for name in joined:
                home[name] = joined[0]
            return True

        # A packet sink joins no region but its own: it reads the encoder's
        # output, and only an ffmpeg on its own side of the pipe encodes.
        alone = set(self.g.packet_sinks)
        links = [
            (producer, name)
            for name in self.order
            if self.external[name] and name not in alone
            for ref in self.g.nodes[name].inputs
            if (producer := _ref_node(ref)) is not None
            and self.external.get(producer, False)
            and producer not in alone
        ]
        merged = True
        while merged:
            merged = False
            for producer, consumer in links:
                if join(producer, consumer):
                    merged = True
            for name in self.order:
                if name in home or self.g.nodes[name].filter not in SPLIT_FILTERS:
                    continue
                inputs = self.g.nodes[name].inputs
                if not inputs:
                    continue
                reads = consumers.get(name)
                if not reads or any(reader not in home for reader in reads):
                    continue
                if any(reader in alone for reader in reads):
                    continue  # a packet sink's edge stays an ffmpeg's to encode
                # The split's producer joins too when it is a module; otherwise
                # the split's own input becomes a boundary read of the region.
                feeds = _ref_node(inputs[0])
                feeder = [feeds] if feeds is not None and feeds in home else []
                if join(name, *reads, *feeder):
                    merged = True
        return [groups[name] for name in self.order if name in groups]

    def _region_reads(self, members: Sequence[str]) -> list[tuple[FrameRef, str]]:
        """Refs this region reads from outside, each with the node reading it."""
        inside = set(members)
        wanted: list[tuple[FrameRef, str]] = []
        seen: set[FrameRef] = set()
        for name in members:
            for ref in self.g.nodes[name].inputs:
                producer = _ref_node(ref)
                if producer is not None and producer in inside:
                    continue
                if ref in seen:
                    continue
                seen.add(ref)
                wanted.append((ref, name))
        return wanted

    def _region_writes(self, members: Sequence[str]) -> list[tuple[FrameRef, StreamType]]:
        """Pads this region produces that something outside it reads."""
        inside = set(members)
        outside: dict[tuple[str, int], FrameRef] = {}
        refs: list[FrameRef] = []
        for name, node in self.g.nodes.items():
            if name not in inside:
                refs.extend(node.inputs)
        for unit in self.g.sinks:
            refs.extend(o.ref for o in unit.outputs)
        for ref in refs:
            producer = _ref_node(ref)
            if producer is not None and producer in inside:
                outside[(producer, _ref_pad(ref))] = ref
        written: list[tuple[FrameRef, StreamType]] = []
        for name in members:
            node = self.g.nodes[name]
            for pad, kind in enumerate(node.outputs):
                exit_ref = outside.get((name, pad))
                if exit_ref is not None:
                    written.append((exit_ref, kind))
        return written

    def _shape(self, name: str) -> ModuleShape:
        return self.shapes.get(self.g.nodes[name].filter, ModuleShape())

    def _lookahead(self, members: Sequence[str]) -> int:
        """The region's declared latency: the longest lookahead path through it."""
        inside = set(members)
        best: dict[str, int] = {}
        for name in members:  # topological
            above = max(
                (
                    best[producer]
                    for ref in self.g.nodes[name].inputs
                    if (producer := _ref_node(ref)) is not None and producer in inside
                ),
                default=0,
            )
            best[name] = above + self._shape(name).lookahead
        return max(best.values(), default=0)

    # -- the depth bound

    def _node_delay(self, name: str) -> int | None:
        """Frames this node reads before it can hand the first one on.

        None where the node's output holds a different number of frames than
        its input: nothing at compile time counts how far such a leg drifts
        from its siblings.
        """
        node = self.g.nodes[name]
        if self.external.get(name, False):
            shape = self._shape(name)
            return shape.lookahead if shape.one_to_one else None
        if node.filter in SPLIT_FILTERS:
            return 0
        if node.filter in RATE_CHANGING_FILTERS:
            return None
        window = FILTER_WINDOWS.get(node.filter)
        if window is None:
            return 0
        option, default = window
        written = node.args.get(option) if option else None
        size = written if isinstance(written, int) and not isinstance(written, bool) else default
        return max(size - 1, 0)

    def _node_delays(self, names: Sequence[str]) -> dict[str, int | None]:
        """Each node's delay from this process's own inputs, over its longest path."""
        inside = set(names)
        best: dict[str, int | None] = {}
        for name in names:  # topological
            step = self._node_delay(name)
            above = 0
            for ref in self.g.nodes[name].inputs:
                producer = _ref_node(ref)
                if producer is None or producer not in inside:
                    continue
                found = best[producer]
                if found is None:
                    step = None
                    break
                above = max(above, found)
            best[name] = None if step is None else above + step
        return best

    def _unbounded_node(self, names: Sequence[str]) -> str | None:
        """The filter of the first node in `names` whose delay has no size."""
        return next(
            (self.g.nodes[name].filter for name in names if self._node_delay(name) is None),
            None,
        )

    def _leg_delay(self, process: _Pending, ref: FrameRef) -> int | None:
        """The delay of the chain inside `process` that produces `ref`."""
        name = _ref_node(ref)
        if name is None or name not in process.nodes:
            return 0
        return self._node_delays(process.nodes)[name]

    def _process_delay(self, pid: str) -> int | None:
        """Frames a process holds: the one it is working on, plus its own chains."""
        region = next((s for s in self.sidecars if s.id == pid), None)
        if region is not None:
            members = self.members[pid]
            if any(self._node_delay(name) is None for name in members):
                return None
            return 1 + region.lookahead
        process = next((p for p in self.pending if p.id == pid), None)
        if process is None:
            return None
        delays = self._node_delays(process.nodes).values()
        if any(found is None for found in delays):
            return None
        return 1 + max((found for found in delays if found is not None), default=0)

    def _process_blame(self, pid: str) -> str | None:
        """The filter inside `pid` whose delay has no size, if one is there."""
        region = next((s for s in self.sidecars if s.id == pid), None)
        names = self.members[pid] if region is not None else next(
            (p.nodes for p in self.pending if p.id == pid), []
        )
        return self._unbounded_node(names)

    def _downstream(self) -> dict[str, frozenset[str]]:
        """Each process and everything its streams reach, itself included."""
        ahead: dict[str, list[str]] = {}
        for edge in self.edges:
            ahead.setdefault(edge.source, []).append(edge.target)
        found: dict[str, frozenset[str]] = {}

        def walk(pid: str, path: frozenset[str]) -> frozenset[str]:
            if pid in found:
                return found[pid]
            if pid in path:  # malformed plan; stop rather than recurse forever
                return frozenset({pid})
            below = {pid}
            for target in ahead.get(pid, []):
                below |= walk(target, path | {pid})
            found[pid] = frozenset(below)
            return found[pid]

        ids = [p.id for p in self.pending] + [s.id for s in self.sidecars]
        for pid in ids:
            walk(pid, frozenset())
        return found

    def _cost(self, start: str, target: str) -> int | None:
        """The longest path delay from `start` to `target`, `target` excluded.

        The processes a frame passes THROUGH on the way, each holding what
        :meth:`_process_delay` says. 0 when `start` is `target` itself.
        """
        ahead: dict[str, list[str]] = {}
        for edge in self.edges:
            ahead.setdefault(edge.source, []).append(edge.target)
        best: dict[str, int | None] = {}

        def walk(pid: str) -> int | None:
            if pid == target:
                return 0
            if pid in best:
                return best[pid]
            best[pid] = None  # cycle guard, and the answer for a dead end
            own = self._process_delay(pid)
            if own is None:
                return None
            found: int | None = None
            for follower in ahead.get(pid, []):
                below = walk(follower)
                if below is None:
                    continue
                found = below if found is None else max(found, below)
            best[pid] = None if found is None else own + found
            return best[pid]

        return walk(start)

    def _bound_edges(self) -> None:
        """Give every edge leaving a live reader the frames it must hold.

        The reader writes one frame to each of its edges at once, and the
        process where two of those edges meet again consumes one from each at
        once -- so the edge whose path is quicker holds frames until the slower
        one arrives. The bound is that difference, counted over the process
        delays between the reader and the meeting point, plus the reader's own
        chain for each edge.
        """
        opened = {p.id: self._opened(p) for p in self.pending}
        readers = [
            process
            for process in self.pending
            if any(alias in self.live for alias in opened[process.id])
            and len([e for e in self.edges if e.source == process.id]) > 1
        ]
        if not readers:
            return
        reach = self._downstream()
        bounds: dict[int, int] = {}
        for reader in readers:
            outs = [
                (index, edge)
                for index, edge in enumerate(self.edges)
                if edge.source == reader.id
            ]
            legs = {index: self._edge_leg(reader, edge) for index, edge in outs}
            meeting = {
                pid
                for pid in reach
                if sum(1 for _, edge in outs if pid in reach[edge.target]) > 1
            }
            for pid in sorted(meeting):
                delays: dict[int, int] = {}
                for index, edge in outs:
                    if pid not in reach[edge.target]:
                        continue
                    leg, cost = legs[index], self._cost(edge.target, pid)
                    if leg is None or cost is None:
                        raise self._unbounded_refusal(reader, edge, pid)
                    delays[index] = leg + cost
                if len(delays) < 2:
                    continue
                slowest = max(delays.values())
                for index, delay in delays.items():
                    bounds[index] = max(bounds.get(index, 0), slowest - delay)
        for index, bound in bounds.items():
            edge = self.edges[index]
            self.edges[index] = replace(
                edge, bound=bound, buffer=self._sized(edge, bound)
            )

    def _edge_leg(self, reader: _Pending, edge: StreamEdge) -> int | None:
        """One edge's delay inside the reader: its chain, plus any encoder on it."""
        leg = self._leg_delay(reader, edge.ref)
        if leg is None:
            return None
        return leg + (ENCODED_EDGE_DELAY if _encodes(edge.format) else 0)

    def _unbounded_refusal(
        self, reader: _Pending, edge: StreamEdge, meeting: str
    ) -> FfrwdError:
        """The rejection for a live input whose paths cannot be counted.

        The stage to name is the first whose frame count does not follow its
        input: the reader's own chains, then the processes between this edge
        and where the paths meet. Either path's is worth naming -- the count
        that has no size is the DIFFERENCE between the two.
        """
        alias = next(iter(sorted(self._opened(reader).keys() & self.live)), "")
        blame = self._unbounded_node(reader.nodes)
        if blame is None:
            between = sorted(self._downstream()[edge.target] - {meeting})
            blame = next(
                (found for pid in between if (found := self._process_blame(pid))),
                None,
            )
        named = f"'{blame}'" if blame else "a stage on one of them"
        return self._live_refusal(
            alias,
            "so one process reads it and hands every other one a pipe -- and "
            f"two of those paths come back together, with {named} between them "
            "handing on a different number of frames than it reads",
            hint="the buffer between two such paths is sized from how far one "
            f"runs ahead of the other, and {named} makes that distance "
            "uncountable: apply it after the two paths come together rather "
            "than before, or record the input to a file and run the query over "
            "the file",
        )

    def _sized(self, edge: StreamEdge, bound: int) -> EdgeBuffer | None:
        """The buffer `bound` frames of this edge buys, and where it is held.

        The pipe's own buffer where the bytes fit under the limit, and the
        producing ffmpeg's fifo queue where they do not, or where an audio
        packet's size is not a number anything here can name. Only an edge
        leaving a READER is ever sized, and a reader is always an ffmpeg
        process, so the fifo muxer is always available to it.
        """
        if bound <= 0:
            return None
        frames = bound * SAFETY
        width = _frame_bytes(edge.format)
        if width is not None and frames * width <= PIPE_BUFFER_LIMIT:
            return EdgeBuffer("pipe", frames, size=_rounded(frames * width))
        return EdgeBuffer("fifo", frames, packets=frames)

    # -- the fan-in rule

    def _one_to_one(self, name: str) -> bool:
        """True when this node hands on exactly one frame per frame it reads.

        A module says so itself. A ``split`` says so structurally -- its pads
        are copies of its input. Nothing else does: an ffmpeg filter declares
        nothing about its frame timing, and this will not assume.
        """
        if self.external.get(name, False):
            return self._shape(name).one_to_one
        return self.g.nodes[name].filter in SPLIT_FILTERS

    def _anchor(self, ref: FrameRef) -> str:
        """The point `ref` is in lockstep with: back through one-to-one nodes."""
        seen: set[str] = set()
        while True:
            if is_src(ref):
                return ref
            name = _ref_node(ref)
            slot = ref if name is None else f"{name}:{_ref_pad(ref)}"
            if name is None or name not in self.g.nodes or slot in seen:
                return slot
            seen.add(slot)
            node = self.g.nodes[name]
            if not node.inputs or not self._one_to_one(name):
                return slot
            ref = node.inputs[0]

    def _check_lockstep(self) -> None:
        """Refuse a multi-input module whose inputs do not share a timeline.

        Legal only when every input traces back to ONE point through nodes
        that emit a frame per frame; anything else hands the module streams
        whose PTS drift apart, and it has no way to say which frames pair up.

        A PACKET SINK is exempt: packets are not frames, nothing pairs one
        pad's packet with another's, and its pads are separate encodes of the
        same source by construction.
        """
        for name in self.order:
            if not self.external[name] or name in self.g.packet_sinks:
                continue
            node = self.g.nodes[name]
            if len(node.inputs) < 2:
                continue
            first = self._anchor(node.inputs[0])
            offender = next(
                (ref for ref in node.inputs[1:] if self._anchor(ref) != first), None
            )
            if offender is None:
                continue
            raise FfrwdError(
                ErrorCode.UNSUPPORTED_SQL,
                f"the module '{node.filter}' reads several streams, and "
                f"{_named_ref(node.inputs[0])} and {_named_ref(offender)} do not "
                "run in lockstep: they reach it from different points",
                hint="feed every stream of a multi-stream module from one "
                "stream, through modules that declare one frame out per frame in",
            )

    def run(self) -> ProcessPlan:
        self._check_lockstep()
        for members in self._regions():
            first = next(name for name in members if self.external[name])
            entry = self.g.nodes[first]
            sidecar = SidecarProcess(
                id=f"sidecar{len(self.sidecars)}",
                module=entry.filter,
                node=first,
                args=dict(entry.args),
                inputs=tuple(ref for ref, _ in self._region_reads(members)),
                outputs=tuple(kind for _, kind in self._region_writes(members)),
                modules=_bindings(
                    self.g.nodes[name].filter
                    for name in members
                    if self.external[name]
                    and self.g.nodes[name].filter not in HOSTED_FILTERS
                ),
                models=self._region_models(members),
                grants=self._region_grants(members),
                lookahead=self._lookahead(members),
                rows=self._region_rows(members),
                impure=self._region_impure(members),
                sink=any(name in self.g.module_sinks for name in members),
                packet_sink=any(name in self.g.packet_sinks for name in members),
            )
            self.sidecars.append(sidecar)
            self.members[sidecar.id] = list(members)
            for name in members:
                self.sidecar_of[name] = sidecar.id

        # One ffmpeg process per output-file group, then one per raw stream
        # edge, created as the demand for it is found.
        demands: list[tuple[str, FrameRef, int]] = []
        for depth in sorted(set(self.sink_depth)):
            units = [
                unit for unit, at in zip(self.g.sinks, self.sink_depth) if at == depth
            ]
            refs = [o.ref for unit in units for o in unit.outputs]
            process = _Pending(
                id=self._ffmpeg_id(),
                depth=depth,
                nodes=self._ancestors(refs, depth),
                sinks=[_copy_unit(unit) for unit in units],
                pipes=[],
            )
            self.pending.append(process)
            demands.extend((process.id, ref, depth) for ref in self._consumed(process))

        for sidecar in self.sidecars:
            demands.extend(
                (sidecar.id, ref, self.depth[reader])
                for ref, reader in self._region_reads(self.members[sidecar.id])
            )

        while demands:
            target, ref, depth = demands.pop(0)
            producer = _ref_node(ref)
            if producer is not None and self.external.get(producer, False):
                consumer = self._reader(target, ref)
                if consumer is not None and consumer in self.g.packet_sinks:
                    # A packet sink consumes the encoder's output, and a
                    # module region emits decoded frames: an encoding ffmpeg
                    # stands between them, reading the region's pipe and
                    # writing the encoded stream the sink's edge names --
                    # the same fronting encoder the sink gets when its feed
                    # is an ffmpeg filter, shaped by the same options.
                    stage = _Pending(
                        id=self._ffmpeg_id(),
                        depth=depth,
                        nodes=[],
                        sinks=[],
                        pipes=[ref],
                    )
                    self.pending.append(stage)
                    self.consumer_of[stage.id] = target
                    self._add_edge(stage.id, target, ref)
                    self._add_edge(self.sidecar_of[producer], stage.id, ref)
                    continue
                self._add_edge(self.sidecar_of[producer], target, ref)
                continue
            at = self.depth[producer] if producer in self.depth else depth
            nodes = self._ancestors([ref], at)
            sibling = self._sibling_feeder(target, at, nodes, ref)
            if sibling is None:
                sibling = self._live_reader(at, nodes, ref)
            if sibling is not None:
                wanted = set(sibling.nodes) | set(nodes)
                sibling.nodes = [name for name in self.order if name in wanted]
                sibling.pipes.append(ref)
                self._add_edge(sibling.id, target, ref)
                continue
            process = _Pending(
                id=self._ffmpeg_id(),
                depth=at,
                nodes=nodes,
                sinks=[],
                pipes=[ref],
            )
            self.pending.append(process)
            self.consumer_of[process.id] = target
            self._add_edge(process.id, target, ref)
            demands.extend((process.id, r, at) for r in self._consumed(process))

        self._redirect_live_reads()
        self._add_rows_edges()
        self._bound_edges()
        processes: list[Process] = [self._materialize(p) for p in self.pending]
        processes.extend(self._materialize_region(sidecar) for sidecar in self.sidecars)
        return ProcessPlan(
            processes=tuple(processes), edges=(*self.edges, *self.rows)
        )

    def _region_models(self, members: Sequence[str]) -> tuple[ModelBinding, ...]:
        """The ``-nn`` binding each module of this region needs, in graph order."""
        found: dict[str, ModelBinding] = {}
        for name in members:
            path = self.g.nodes[name].filter
            binding = self.models.get(path)
            if binding is not None and path not in found:
                found[path] = binding
        return tuple(found.values())

    def _region_grants(self, members: Sequence[str]) -> tuple[EffectGrant, ...]:
        """The effect grants each module of this region needs, in graph order."""
        found: dict[tuple[str, str], EffectGrant] = {}
        for name in members:
            path = self.g.nodes[name].filter
            for effect in self.effects.get(path, ()):
                key = (effect, path)
                if key not in found:
                    found[key] = EffectGrant(effect=effect, module=path)
        return tuple(found.values())

    def _region_impure(self, members: Sequence[str]) -> tuple[str, ...]:
        """The modules of this region that carry state, in graph order.

        A module that declared nothing about its shape is pure, which is what
        a plain filter module is.
        """
        found: list[str] = []
        for name in members:
            if not self.external[name]:
                continue
            path = self.g.nodes[name].filter
            if not self._shape(name).pure and path not in found:
                found.append(path)
        return tuple(found)

    def _region_rows(self, members: Sequence[str]) -> RowsSink | None:
        """Where this region's rows go, for a region that writes any.

        One region writes at most one rows document: a module whose rows are
        selected reads no rows itself, so no two members can both have one.
        """
        found = [self.g.rows_sinks[name] for name in members if name in self.g.rows_sinks]
        if not found:
            return None
        if len(found) > 1:
            raise FfrwdError(
                ErrorCode.INTERNAL,
                f"a region of {len(found)} modules writes rows, and a process "
                "writes one rows document",
                hint="please report this query as a bug",
            )
        return found[0]

    def _add_rows_edges(self) -> None:
        """One edge per region whose rows an ffmpeg process reads as a track.

        A region writing rows to a FILE writes it itself and reaches no other
        process, so it earns no edge and orders nothing.
        """
        for sidecar in self.sidecars:
            rows = sidecar.rows
            if rows is None or not rows.alias:
                continue
            target = self._rows_reader(rows.alias)
            if target is None:
                raise FfrwdError(
                    ErrorCode.INTERNAL,
                    f"the rows '{sidecar.module}' writes reach no output",
                    hint="please report this query as a bug",
                )
            self.rows.append(
                RowsEdge(
                    source=sidecar.id,
                    target=target,
                    alias=rows.alias,
                    container=rows.container,
                )
            )

    def _rows_reader(self, alias: str) -> str | None:
        """The pending ffmpeg process that maps the minted input `alias`."""
        for process in self.pending:
            for unit in process.sinks:
                for output in unit.outputs:
                    if is_src(output.ref) and src_parts(output.ref)[0] == alias:
                        return process.id
        return None

    def _add_edge(
        self, source: str, target: str, ref: FrameRef, *, copy: bool = False
    ) -> None:
        # The consuming NODE, where a sidecar process is what consumes: the
        # format an edge carries is the module's, not the process id's.
        consumer = self._reader(target, ref)
        wire = self._format(ref, consumer)
        if copy:
            # Nothing on the far side filters this stream, so it travels as it
            # arrived: NUT carries the packets and both ends copy them, which
            # is the passthrough the query asked for and not a decode.
            wire = replace(wire, codec=COPY_CODEC)
        self.edges.append(
            StreamEdge(
                source=source,
                target=target,
                ref=ref,
                format=wire,
                annotations=self._carries_annotations(ref, consumer),
            )
        )

    def _reader(self, target: str, ref: FrameRef) -> str | None:
        """The node inside sidecar process `target` that reads `ref`, if any."""
        for name in self.members.get(target, []):
            if ref in self.g.nodes[name].inputs:
                return name
        return None

    def _carries_annotations(self, ref: FrameRef, consumer: str | None) -> bool:
        """Whether this edge's frames travel with the producer's rows.

        The consumer reads annotations off its first input, and the producer
        has to be a module: ffmpeg writes no annotation stream. Both ends
        external is what makes it true, and nothing else.
        """
        if consumer is None or not self.g.nodes[consumer].reads_annotations:
            return False
        if self.g.nodes[consumer].inputs[:1] != [ref]:
            return False
        producer = _ref_node(ref)
        return producer is not None and self.external.get(producer, False)

    # -- edge formats

    def _origin(self, ref: FrameRef) -> tuple[str, StreamType, int] | None:
        """The input stream `ref` traces back to, or None if it traces to none.

        Follows the input carrying the same stream type at each node, which is
        the one whose parameters survive the filter.
        """
        wanted = ref_type(self.g, ref)
        seen: set[str] = set()
        current = ref
        while True:
            if is_src(current):
                return src_parts(current)
            name = _ref_node(current)
            if name is None or name not in self.g.nodes or name in seen:
                return None
            seen.add(name)
            inputs = self.g.nodes[name].inputs
            if not inputs:
                return None
            current = next(
                (r for r in inputs if ref_type(self.g, r) == wanted), inputs[0]
            )

    def _origin_meta(self, ref: FrameRef) -> StreamMeta | None:
        origin = self._origin(ref)
        if origin is None:
            return None
        alias, kind, index = origin
        result = self.probes.get(alias)
        if result is None:
            return None
        return next((s for s in result.by_type(kind) if s.index == index), None)

    def _format(self, ref: FrameRef, target: str | None = None) -> StreamFormat:
        meta = self._origin_meta(ref)
        pads = self.g.packet_sinks.get(target) if target is not None else None
        if ref_type(self.g, ref) == "audio":
            if pads is not None:
                # The consumer is a packet sink: this edge carries the audio
                # encoder's output, not the pcm every other audio edge does.
                assert target is not None  # `pads` came from it
                rest = dict(pads[self.g.nodes[target].inputs.index(ref)])
                return AudioFormat(
                    rate=meta.sample_rate if meta else None,
                    channels=meta.channels if meta else None,
                    codec=str(rest.pop("audio_codec")),
                    options=tuple(sorted(rest.items())),
                )
            return replace(
                self._audio_wire(ref, target),
                rate=meta.sample_rate if meta else None,
                channels=meta.channels if meta else None,
            )
        if pads is not None:
            # The consumer is a packet sink: the edge carries the encoder's
            # output, shaped by the COPY's own options -- this PAD's, since a
            # ladder shapes every rendition differently.
            assert target is not None  # `pads` came from it
            rest = dict(pads[self.g.nodes[target].inputs.index(ref)])
            codec = str(rest.pop("video_codec"))
            pix_fmt = rest.pop("pix_fmt", DEFAULT_PIX_FMT)
            return VideoFormat(
                pix_fmt=str(pix_fmt),
                width=meta.width if meta else None,
                height=meta.height if meta else None,
                timebase=_timebase(meta.fps) if meta else None,
                codec=codec,
                options=tuple(sorted(rest.items())),
            )
        return VideoFormat(
            pix_fmt=self._pix_fmt(ref, target),
            width=meta.width if meta else None,
            height=meta.height if meta else None,
            timebase=_timebase(meta.fps) if meta else None,
        )

    def _pix_fmt(self, ref: FrameRef, target: str | None) -> str:
        """The pixel format this edge carries.

        An edge with an external node at either end carries what THAT node
        accepts: it is the one reading or writing the frames, and it takes one
        format. Every other edge takes the default, which is what the
        producing ffmpeg is told to write.
        """
        for name in (_ref_node(ref), target):
            if name is None or not self.external.get(name, False):
                continue
            found = self.pix_fmts.get(self.g.nodes[name].filter)
            if found is not None:
                return found
        return DEFAULT_PIX_FMT

    def _audio_wire(self, ref: FrameRef, target: str | None) -> AudioFormat:
        """The pcm and the conformance this edge carries.

        The audio mirror of :meth:`_pix_fmt`: an edge with an external node at
        either end carries what THAT node accepts. Every other edge takes the
        defaults, which constrain nothing.
        """
        for name in (_ref_node(ref), target):
            if name is None or not self.external.get(name, False):
                continue
            found = self.audio_wires.get(self.g.nodes[name].filter)
            if found is not None:
                return found
        return AudioFormat()

    # -- payloads

    def _split_pads(self, process: _Pending) -> dict[str, dict[int, FrameRef]]:
        """Each split this payload holds, and the pads read inside it.

        Keyed by node id, then by pad -- the ref that reads it. Pads a consumer
        in another process reads are absent, which is what makes them droppable.
        """
        refs: list[FrameRef] = []
        for name in process.nodes:
            refs.extend(self.g.nodes[name].inputs)
        for unit in process.sinks:
            refs.extend(o.ref for o in unit.outputs)
        refs.extend(process.pipes)

        read: dict[str, dict[int, FrameRef]] = {}
        for ref in refs:
            producer = _ref_node(ref)
            if producer is None or producer not in process.nodes:
                continue
            if self.g.nodes[producer].filter not in SPLIT_FILTERS:
                continue
            read.setdefault(producer, {})[_ref_pad(ref)] = ref
        return read

    def _shrink_splits(
        self, process: _Pending
    ) -> tuple[list[str], dict[str, Node], Callable[[FrameRef], FrameRef]]:
        """This payload's splits cut to the consumers it actually holds.

        A split read once here dissolves: its consumer reads the split's own
        input instead. One read several times keeps a split that size, its
        surviving pads renumbered. One read not at all is dropped. Returns the
        node ids that remain, the splits that changed size, and the
        substitution the payload's refs need.
        """
        read = self._split_pads(process)
        keep: list[str] = []
        resized: dict[str, Node] = {}
        dissolved: dict[FrameRef, FrameRef] = {}
        renumbered: dict[FrameRef, FrameRef] = {}

        def undissolve(ref: FrameRef) -> FrameRef:
            seen: set[FrameRef] = set()
            while ref in dissolved and ref not in seen:
                seen.add(ref)
                ref = dissolved[ref]
            return ref

        for name in process.nodes:  # topological: a split precedes its readers
            node = self.g.nodes[name]
            pads = read.get(name)
            if pads is None:
                if node.filter in SPLIT_FILTERS:
                    continue  # nothing here reads it
                keep.append(name)
                continue
            if len(pads) >= len(node.outputs):
                keep.append(name)
                continue
            if len(pads) == 1:
                dissolved[next(iter(pads.values()))] = undissolve(node.inputs[0])
                continue
            keep.append(name)
            for new_pad, old_pad in enumerate(sorted(pads)):
                ref = pads[old_pad]
                if _ref_pad(ref) != new_pad:
                    renumbered[ref] = f"{name}:{new_pad}"
            resized[name] = Node(
                id=node.id,
                filter=node.filter,
                args={**node.args, "n": len(pads)},
                inputs=list(node.inputs),
                outputs=list(node.outputs[: len(pads)]),
                reads_annotations=node.reads_annotations,
            )

        def substitute(ref: FrameRef) -> FrameRef:
            resolved = undissolve(ref)
            return renumbered.get(resolved, resolved)

        return keep, resized, substitute

    def _materialize(self, process: _Pending) -> FfmpegProcess:
        """`process` as a complete graph, its pipes now inputs and sinks."""
        kept, resized, substitute = self._shrink_splits(process)
        incoming = [e for e in self.edges if e.target == process.id]
        alias_of: dict[FrameRef, str] = {}
        marker_of: dict[FrameRef, str] = {}
        taken = set(self.g.sources)
        for edge in incoming:
            if edge.ref in alias_of:
                continue
            alias = _unique_alias(edge.ref, taken)
            taken.add(alias)
            alias_of[edge.ref] = alias
            marker_of[edge.ref] = "a" if isinstance(edge.format, AudioFormat) else "v"

        def rewrite(ref: FrameRef) -> FrameRef:
            ref = substitute(ref)
            alias = alias_of.get(ref)
            return ref if alias is None else f"src:{alias}:{marker_of[ref]}:0"

        nodes: dict[str, Node] = {}
        for name in kept:
            node = resized.get(name, self.g.nodes[name])
            nodes[name] = Node(
                id=node.id,
                filter=node.filter,
                args=dict(node.args),
                inputs=[rewrite(ref) for ref in node.inputs],
                outputs=list(node.outputs),
                reads_annotations=node.reads_annotations,
            )

        sinks = [_rewrite_unit(unit, rewrite) for unit in process.sinks]
        sinks.extend(
            SinkUnit(
                outputs=[
                    Output(
                        ref=rewrite(ref),
                        type=ref_type(self.g, ref),
                        name=None,
                        metadata={},
                    )
                ],
                path=PIPE,
            )
            for ref in process.pipes
        )

        paths, sources, trims, options = self._inputs(nodes, sinks)
        for edge in incoming:
            alias = alias_of[edge.ref]
            if alias in sources:
                continue
            sources[alias] = len(paths)
            paths.append(PIPE)

        return FfmpegProcess(
            id=process.id,
            graph=Graph(
                input_paths=paths,
                sources=sources,
                nodes=nodes,
                sinks=sinks,
                input_trims=trims,
                input_options=options,
            ),
        )

    def _materialize_region(self, sidecar: SidecarProcess) -> SidecarProcess:
        """`sidecar` given the region's own graph and its annotation sides.

        The region reads its boundary streams as pipes and writes them as
        pipes, exactly as an ffmpeg process does, and its nodes call modules
        by the name the ``-m`` table binds. Everything between two nodes here
        stays a plain node ref: the network hands those frames over in memory,
        annotations included, so no edge and no split node spells them.
        """
        members = self.members[sidecar.id]
        incoming: list[StreamEdge] = []
        seen: set[FrameRef] = set()
        for edge in self.edges:
            if edge.target == sidecar.id and edge.ref not in seen:
                seen.add(edge.ref)
                incoming.append(edge)
        outgoing = [e for e in self.edges if e.source == sidecar.id]

        alias_of: dict[FrameRef, str] = {}
        marker_of: dict[FrameRef, str] = {}
        taken = set(self.g.sources)
        for edge in incoming:
            alias = _unique_alias(edge.ref, taken)
            taken.add(alias)
            alias_of[edge.ref] = alias
            marker_of[edge.ref] = "a" if isinstance(edge.format, AudioFormat) else "v"

        names = {binding.path: binding.name for binding in sidecar.modules}
        dissolved: dict[str, FrameRef] = {}

        def rewrite(ref: FrameRef) -> FrameRef:
            slot = ref if is_src(ref) else f"{_ref_node(ref)}:{_ref_pad(ref)}"
            ref = dissolved.get(slot, ref)
            if ref in alias_of:
                return f"src:{alias_of[ref]}:{marker_of[ref]}:0"
            return ref

        nodes: dict[str, Node] = {}
        for name in members:  # topological: a split precedes its readers
            node = self.g.nodes[name]
            if not self.external[name]:
                # An absorbed split: its readers take its own input instead.
                source = rewrite(node.inputs[0])
                for pad in range(len(node.outputs)):
                    dissolved[f"{name}:{pad}"] = source
                continue
            nodes[name] = Node(
                id=node.id,
                # A hosted node keeps its reserved name; no ``-m`` binds one.
                filter=names.get(node.filter, node.filter),
                args=dict(node.args),
                inputs=[rewrite(ref) for ref in node.inputs],
                outputs=list(node.outputs),
                reads_annotations=node.reads_annotations,
            )
        sinks = [
            SinkUnit(
                outputs=[
                    Output(
                        ref=edge.ref,
                        type=ref_type(self.g, edge.ref),
                        name=None,
                        metadata={},
                    )
                ],
                path=PIPE,
            )
            for edge in outgoing
        ]
        # The module whose rows leave is a sink of the region too: the network
        # string has to name the pad they were read off, even though its
        # frames go no further than the rows document.
        rows_node = next((name for name in members if name in self.g.rows_sinks), None)
        if rows_node is not None:
            sinks.append(
                SinkUnit(
                    outputs=[
                        Output(
                            ref=rows_node,
                            type=ref_type(self.g, rows_node),
                            name=None,
                            metadata={},
                        )
                    ],
                    path=PIPE,
                )
            )
        # A SINK MODULE is one too: the network string names its pad, and the
        # null output that pad is mapped to carries nothing.
        for name in members:
            if name not in self.g.module_sinks:
                continue
            sinks.append(
                SinkUnit(
                    outputs=[
                        Output(
                            ref=name,
                            type=ref_type(self.g, name),
                            name=None,
                            metadata={},
                        )
                    ],
                    path=PIPE,
                )
            )
        return replace(
            sidecar,
            reads_rows=any(e.annotations for e in self.edges if e.target == sidecar.id),
            writes_rows=any(e.annotations for e in self.edges if e.source == sidecar.id),
            graph=Graph(
                input_paths=[PIPE] * len(incoming),
                sources={alias_of[e.ref]: index for index, e in enumerate(incoming)},
                nodes=nodes,
                sinks=sinks,
            ),
        )

    def _inputs(
        self, nodes: Mapping[str, Node], sinks: Sequence[SinkUnit]
    ) -> tuple[
        list[str],
        dict[str, int],
        dict[str, tuple[float | None, float | None]],
        dict[str, dict[str, object]],
    ]:
        """This process's own input table: the inputs it reads, in ``-i`` order.

        Kept by INDEX, so two aliases over one input stay together and a sink's
        ``chapters``/``metadata`` input index still points where it did.
        """
        kept: set[int] = set()
        refs: list[FrameRef] = [ref for node in nodes.values() for ref in node.inputs]
        for unit in sinks:
            refs.extend(o.ref for o in unit.outputs)
            for index in (unit.chapters, unit.metadata):
                if index is not None and index >= 0:
                    kept.add(index)
        for ref in refs:
            if not is_src(ref):
                continue
            index = self.g.sources.get(src_parts(ref)[0])
            if index is not None:
                kept.add(index)

        indices = sorted(kept)
        renumber = {old: new for new, old in enumerate(indices)}
        paths = [self.g.input_paths[old] for old in indices]
        sources = {
            alias: renumber[index]
            for alias, index in self.g.sources.items()
            if index in renumber
        }
        trims = {
            alias: bounds
            for alias, bounds in self.g.input_trims.items()
            if alias in sources
        }
        options = {
            alias: dict(values)
            for alias, values in self.g.input_options.items()
            if alias in sources
        }
        return paths, sources, trims, options


def _copy_unit(unit: SinkUnit) -> SinkUnit:
    return _rewrite_unit(unit, lambda ref: ref)


def _rewrite_unit(unit: SinkUnit, rewrite: Callable[[FrameRef], FrameRef]) -> SinkUnit:
    return SinkUnit(
        outputs=[
            Output(
                ref=rewrite(o.ref),
                type=o.type,
                name=o.name,
                metadata=dict(o.metadata),
                disposition=o.disposition,
            )
            for o in unit.outputs
        ],
        path=unit.path,
        options=dict(unit.options),
        tags=dict(unit.tags),
        window=unit.window,
        chapters=unit.chapters,
        metadata=unit.metadata,
        attachments=list(unit.attachments),
    )


def check_spellable(plan: ProcessPlan) -> None:
    """Refuse a plan whose sidecar frames have no wire to leave on.

    A module process writes its own stdout and nothing else, so a region
    whose frames leave on more than one pad is a plan nothing can spawn.
    That happens when a query builds an INSTANCE PER ROW -- gathered into one
    destination or fanned out to several, the instances still share the
    process -- and when one module's frames reach two others that each leave
    the region.

    Checked on the finished plan rather than while it is being built, so
    partitioning stays free to say what the region's shape actually is.
    Unanchored, for the caller to re-anchor on the declaration.
    """
    for sidecar in plan.sidecars:
        if len(sidecar.outputs) <= 1:
            continue
        raise FfrwdError(
            ErrorCode.UNSUPPORTED_SQL,
            f"the module '{sidecar.module}' has {len(sidecar.outputs)} streams "
            "leaving it, and a module process writes one",
            hint="write one COPY per stream the module produces, each naming "
            "its own destination; a module running once per row is one COPY "
            "per row",
        )


def partition(
    g: Graph,
    *,
    external: IsExternal = nothing_external,
    probes: Mapping[str, ProbeResult | None] | None = None,
    pix_fmts: Mapping[str, str] | None = None,
    shapes: Mapping[str, ModuleShape] | None = None,
    audio_wires: Mapping[str, AudioFormat] | None = None,
    models: Mapping[str, ModelBinding] | None = None,
    effects: Mapping[str, tuple[str, ...]] | None = None,
    anchors: Mapping[str, tuple[int, int]] | None = None,
) -> ProcessPlan:
    """Partition a logical graph into the processes that run it.

    `external` marks the nodes ffmpeg cannot host; the default marks none, and
    `g` comes back as a single ffmpeg process whose payload is `g` itself.
    Adjacent external nodes contract into ONE sidecar process carrying their
    region's subgraph. `probes` is keyed by input ALIAS, the same map lowering
    takes, and is what fills in a stream edge's parameters; without it an edge
    still names its container and codec.

    `pix_fmts` is keyed by an external node's FILTER -- the module it hosts --
    and is the pixel format the edges touching that node carry. Every other
    edge, and every edge whose module is not listed, takes
    :data:`DEFAULT_PIX_FMT`.

    `audio_wires` is the same for an AUDIO module: the pcm its edges carry and
    the rate and channel count they are conformed to. An unlisted module's
    audio edges take :class:`AudioFormat`'s defaults and are conformed to
    nothing.

    `shapes` is keyed the same way and is what each module declares about its
    frame timing: it is what a region's `lookahead` sums, and what decides
    whether a multi-input module's streams run in lockstep. An unlisted module
    takes :class:`ModuleShape`'s defaults.

    `models` is keyed the same way and is the model each module runs, which
    the sidecar is told to load before anything is instantiated. A module that
    runs none is absent and binds nothing.

    `effects` is keyed the same way and is what each module needs granted --
    ``http``, ``udp`` -- which the sidecar denies without the matching argv.
    A module needing neither is absent and is granted nothing.

    `anchors` is where each input ALIAS was written, ``(line, col)``, and is
    what a rejection about a live input points at. An alias missing from it
    gets an unanchored rejection.

    Raises ``FfrwdError`` -- unanchored, for the caller to re-anchor on the
    declaration -- for a multi-input module whose streams cannot pair up, and
    ``UNBOUNDED_LIVE_INPUT`` -- already anchored, from `anchors` -- for a live
    input feeding two paths whose difference has no compile-time size.

    The finished plan's pipes are then put in an order every process can
    start from (:func:`ffrwd.startup.arrange`), and a plan no order starts is
    refused there.

    Pure: returns a new plan and never mutates `g`.
    """
    # Imported here: startup reads a finished plan, so it is built on top of
    # this module rather than beside it.
    from . import startup

    plan = _Partitioner(
        g, external, probes, pix_fmts, shapes, audio_wires, models, effects, anchors
    ).run()
    plan = startup.arrange(plan)
    startup.check(plan)
    return plan


def from_commands(graphs: Sequence[Graph]) -> ProcessPlan:
    """The existing command list as a plan: one ffmpeg process per command.

    The shape with no sidecars and no stream edges, where every process is a
    stage of its own and file edges hold the order the list already has. The
    edge names the artifact the earlier command wrote when the later one reads
    it back; a two-pass sink hands over measurements rather than a file, so
    that edge carries rows and no path.
    """
    processes: tuple[Process, ...] = tuple(
        FfmpegProcess(id=f"ffmpeg{index}", graph=graph)
        for index, graph in enumerate(graphs)
    )
    edges: list[Edge] = []
    for index in range(1, len(graphs)):
        earlier, later = graphs[index - 1], graphs[index]
        handoff = next(
            (
                unit.path
                for unit in earlier.sinks
                if unit.path is not None and unit.path in later.input_paths
            ),
            None,
        )
        edges.append(
            FileEdge(
                source=f"ffmpeg{index - 1}",
                target=f"ffmpeg{index}",
                format=(
                    FileFormat("media", handoff)
                    if handoff is not None
                    else FileFormat("rows", None)
                ),
            )
        )
    return ProcessPlan(processes=processes, edges=tuple(edges))


def _stages(processes: Sequence[Process], edges: Sequence[Edge]) -> tuple[Stage, ...]:
    """Group by pipe edge, then order the groups by file edge."""
    ids = [p.id for p in processes]
    parent = {name: name for name in ids}

    def root(name: str) -> str:
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    for edge in edges:
        if not isinstance(edge, StreamEdge | RowsEdge):
            continue
        if edge.source not in parent or edge.target not in parent:
            continue
        left, right = root(edge.source), root(edge.target)
        if left != right:
            parent[right] = left

    members: dict[str, list[str]] = {}
    for name in ids:
        members.setdefault(root(name), []).append(name)
    groups = list(members)  # first-appearance order
    group_of = {name: root(name) for name in ids}

    pending = dict.fromkeys(groups, 0)
    after: dict[str, list[str]] = {}
    for edge in edges:
        if not isinstance(edge, FileEdge):
            continue
        if edge.source not in group_of or edge.target not in group_of:
            continue
        source, target = group_of[edge.source], group_of[edge.target]
        if source == target:
            continue  # inside one stage: no ordering to read from it
        pending[target] += 1
        after.setdefault(source, []).append(target)

    position = {group: index for index, group in enumerate(groups)}
    ready = [position[group] for group in groups if pending[group] == 0]
    heapq.heapify(ready)
    ordered: list[str] = []
    while ready:
        group = groups[heapq.heappop(ready)]
        ordered.append(group)
        for follower in after.get(group, []):
            pending[follower] -= 1
            if pending[follower] == 0:
                heapq.heappush(ready, position[follower])

    placed = set(ordered)
    ordered.extend(group for group in groups if group not in placed)
    return tuple(
        Stage(index=index, processes=tuple(members[group]))
        for index, group in enumerate(ordered)
    )
