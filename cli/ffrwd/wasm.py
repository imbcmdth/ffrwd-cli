"""What a wasm module declares, read from the sidecar at compile time.

A ``LANGUAGE wasm`` function names a module file and one export in it
(:mod:`ffrwd.functions`). Neither the compiler nor ffmpeg can read a
component's interface, so the sidecar is asked: ``ffrwd-wasm --describe
<module>`` prints one JSON object saying which world the module targets, what
the export is called, what parameters it takes and which pixel or sample
formats it accepts. :func:`describe` runs that; :class:`Described` is what
comes back.

Describing is the same shape as probing. One call per distinct module path per
compile, results keyed by path and handed to lowering, which never shells out
itself -- so a lowering test supplies a synthetic :class:`Described` and spawns
nothing, exactly as it supplies a synthetic ``ProbeResult``.

Unlike a probe, a describe that fails is a REJECTION, not a None. A probe is
opportunistic -- an unreadable input costs the query some validation and
compiles anyway -- but a module that cannot be described cannot be wired at
all: its pixel format and its parameters are what the surrounding ffmpeg
processes are built from.

Pixel and sample formats
------------------------
The sidecar carries frames over NUT, and NUT here spells two pixel formats
(:data:`WIRE_PIX_FMTS`) and two sample formats (:data:`WIRE_SAMPLE_FMTS`). A
module accepts some set of its own, so the wire format for the edges touching
it is the intersection, and an empty intersection is a rejection naming both
lists.

Which of the two lists a module fills is what says whether it filters video or
audio (:attr:`Described.kind`). An audio module may also name the sample rates
and channel counts it accepts, and the edges into it are conformed to the first
of each the wire can offer.

Running a module while compiling
--------------------------------
Three calls here run a module rather than describing one, each for an answer
the compiler needs before it can build anything: :func:`invoke` folds a value
function, :func:`probe_source` reads a packet source's catalog, and
:func:`read_packet_rows` reads a packet sink's rows off one stream of a file.
The last is the run-time arrangement run early -- ffmpeg stream-copying the
stream as coded NUT into the sidecar hosting the sink -- and, like the other
two, it is a seam a lowering test replaces, so the unit tier spawns nothing.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol, cast

from . import binaries, nn, probe
from .emit import build_network_graph
from .errors import ErrorCode, FfrwdError
from .execute import STDIN, STDOUT
from .ir import RowsSink, StreamType
from .probe import ProbeResult, RenditionMeta, StreamMeta
from .processes import (
    NUT,
    PCM_F32LE,
    PCM_S16LE,
    AudioFormat,
    EffectGrant,
    ModelBinding,
    ModuleShape,
    PadMeta,
    RowsDocument,
    SidecarProcess,
)

__all__ = [
    "ANNOTATION_TYPES",
    "ANNOTATIONS_IN",
    "ANNOTATIONS_OUT",
    "AUDIO_CODEC_ENCODERS",
    "CODEC_ENCODERS",
    "DEFAULT_TIMEOUT_SECONDS",
    "LANGUAGE_TAGS",
    "MODEL_SUFFIX",
    "PACKET_FILTER_WORLD",
    "PACKET_SOURCE_WORLD",
    "SAMPLE_FMT_CODECS",
    "TIMEOUT_ENV",
    "WIRE_AUDIO_CODECS",
    "WIRE_PIX_FMTS",
    "WIRE_SAMPLE_FMTS",
    "WIRE_VIDEO_CODECS",
    "WIT_PACKAGE",
    "WORLDS",
    "WORLD_VERSION",
    "SinkArity",
    "SinkWants",
    "Describe",
    "Described",
    "DescribedFunction",
    "Invoke",
    "PacketRead",
    "ReadPackets",
    "SourceCatalog",
    "SourceRendition",
    "SourceTrack",
    "audio_encoder_codec",
    "catalog_as_probe",
    "describe",
    "encoder_codec",
    "hosts_packet_filter",
    "hosts_packet_sink",
    "hosts_packet_source",
    "hosts_rows_module",
    "input_rows_arms",
    "invoke",
    "language_tag",
    "model_binding",
    "model_path",
    "copy_argv",
    "probe_source",
    "read_packet_rows",
    "rows_arms",
    "rows_fields",
    "shown_argv",
    "sidecar_argv",
    "timeout_seconds",
    "wire_audio",
    "wire_pix_fmt",
    "wire_sample_fmt",
]

# What the sidecar calls the container every edge it reads or writes is in.
EDGE_FORMAT = NUT

# The output a sink module's pad is mapped to: the sidecar opens nothing for
# it, since the module's own effects are the product.
_NULL_FORMAT = "null"

# The wit worlds this compiler knows how to wire. More than one, because a
# module built against an earlier one still describes and runs the same: the
# worlds have only ever added to what a module may export. A bump is one entry.
# How many streams of one kind a packet sink reads.
SinkArity = Literal["none", "one", "many", "any"]
_SINK_ARITIES = ("none", "one", "many", "any")

# How much of a stream a packet sink asks to be handed.
SinkWants = Literal["all", "keyframes", "first"]
_SINK_WANTS = ("all", "keyframes", "first")

WORLDS: tuple[str, ...] = (
    "ffrwd:av@0.2.0",
    "ffrwd:av@0.3.0",
    "ffrwd:av@0.4.0",
    "ffrwd:av@0.5.0",
    "ffrwd:av@0.6.0",
    "ffrwd:av@0.7.0",
    "ffrwd:av@0.8.0",
    "ffrwd:av@0.9.0",
    "ffrwd:av@0.10.0",
    "ffrwd:av@0.11.0",
    "ffrwd:av@0.12.0",
    "ffrwd:av@0.13.0",
    "ffrwd:av@0.14.0",
    "ffrwd:av@0.15.0",
    "ffrwd:av@0.16.0",
)

# The world a module scaffolded today is built against: the newest of those,
# and the version of the `ffrwd/wasm` package that carries its wit.
WORLD_VERSION = WORLDS[-1].partition("@")[2]

# The package carrying the wit, whose version is the world's.
WIT_PACKAGE = "ffrwd/wasm"

# The pixel formats a stream edge into or out of the sidecar can carry.
WIRE_PIX_FMTS: tuple[str, ...] = ("rgba", "yuv420p")

# The coded video streams a stream edge can carry to a packet sink: the ones
# the sidecar's NUT reader hands through untouched.
WIRE_VIDEO_CODECS: tuple[str, ...] = ("h264", "hevc", "av1")

# The coded audio streams the same edge carries, whose codec tag and
# out-of-band header the sidecar's NUT reader hands through untouched.
WIRE_AUDIO_CODECS: tuple[str, ...] = ("aac",)

# The encoder each of those codecs is reached by when the COPY names none.
CODEC_ENCODERS: Mapping[str, str] = {
    "h264": "libx264",
    "hevc": "libx265",
    "av1": "libsvtav1",
}

# The same, for the audio the edge carries.
AUDIO_CODEC_ENCODERS: Mapping[str, str] = {"aac": "aac"}

# The codec each software encoder writes, for the names that do not carry it
# themselves the way a hardware encoder's `<codec>_<vendor>` spelling does.
_ENCODER_CODECS: Mapping[str, str] = {
    "libx264": "h264",
    "libx264rgb": "h264",
    "libopenh264": "h264",
    "libx265": "hevc",
    "libsvtav1": "av1",
    "libaom-av1": "av1",
    "librav1e": "av1",
}

# The codec each audio encoder writes, the same way.
_AUDIO_ENCODER_CODECS: Mapping[str, str] = {
    "aac": "aac",
    "libfdk_aac": "aac",
}

# The first world whose sidecar hosts a packet sink.
_PACKET_SINK_WORLD = "ffrwd:av@0.10.0"

# The first world whose packet source this sidecar hosts. `open` takes the
# tracks the plan mapped from 0.15.0 on and has not moved since, so that
# world and every later one load; no older spelling is adapted, since a
# source that cannot be told what to pull would subscribe to tracks nobody
# reads.
PACKET_SOURCE_WORLD = "ffrwd:av@0.15.0"

# The first world whose sidecar hosts a rows module.
_ROWS_MODULE_WORLD = "ffrwd:av@0.14.0"

# The first world whose sidecar hosts a packet filter: encoded packets in,
# encoded packets out, with rows arriving beside them.
PACKET_FILTER_WORLD = "ffrwd:av@0.16.0"

# The sample formats one can carry, and the pcm each of them travels as.
WIRE_SAMPLE_FMTS: tuple[str, ...] = ("f32", "s16")
SAMPLE_FMT_CODECS: Mapping[str, str] = {"f32": PCM_F32LE, "s16": PCM_S16LE}

# The JSON Schema types each declared annotation field type covers. `number`
# covers integer as well: the dialect has one numeric type, and a module
# counting pixels declares the same column a module measuring them does.
# `vector` is a plain JSON array of numbers -- no base64: rows already travel
# as JSON text, which the sidecar stamps and filters as JSON, and a number
# array is exactly what a module's own `{"type": "array", "items": {"type":
# "number"}}` rows_schema entry already says.
ANNOTATION_TYPES: Mapping[str, tuple[str, ...]] = {
    "boolean": ("boolean",),
    "number": ("number", "integer"),
    "text": ("string",),
    "vector": ("array",),
}

# The container tag each language code a transcribing module names stands for:
# ISO 639-1 to 639-2/B, which is what a media container records. Covers the
# set whisper-shaped modules accept, including the three codes it spells
# outside 639-1 (`haw`, `yue`, and `jw` for Javanese).
LANGUAGE_TAGS: Mapping[str, str] = {
    "af": "afr", "am": "amh", "ar": "ara", "as": "asm", "az": "aze",
    "ba": "bak", "be": "bel", "bg": "bul", "bn": "ben", "bo": "tib",
    "br": "bre", "bs": "bos", "ca": "cat", "cs": "cze", "cy": "wel",
    "da": "dan", "de": "ger", "el": "gre", "en": "eng", "es": "spa",
    "et": "est", "eu": "baq", "fa": "per", "fi": "fin", "fo": "fao",
    "fr": "fre", "gl": "glg", "gu": "guj", "ha": "hau", "haw": "haw",
    "he": "heb", "hi": "hin", "hr": "hrv", "ht": "hat", "hu": "hun",
    "hy": "arm", "id": "ind", "is": "ice", "it": "ita", "ja": "jpn",
    "jv": "jav", "jw": "jav", "ka": "geo", "kk": "kaz", "km": "khm",
    "kn": "kan", "ko": "kor", "la": "lat", "lb": "ltz", "ln": "lin",
    "lo": "lao", "lt": "lit", "lv": "lav", "mg": "mlg", "mi": "mao",
    "mk": "mac", "ml": "mal", "mn": "mon", "mr": "mar", "ms": "may",
    "mt": "mlt", "my": "bur", "ne": "nep", "nl": "dut", "nn": "nno",
    "no": "nor", "oc": "oci", "pa": "pan", "pl": "pol", "ps": "pus",
    "pt": "por", "ro": "rum", "ru": "rus", "sa": "san", "sd": "snd",
    "si": "sin", "sk": "slo", "sl": "slv", "sn": "sna", "so": "som",
    "sq": "alb", "sr": "srp", "su": "sun", "sv": "swe", "sw": "swa",
    "ta": "tam", "te": "tel", "tg": "tgk", "th": "tha", "tk": "tuk",
    "tl": "tgl", "tr": "tur", "tt": "tat", "uk": "ukr", "ur": "urd",
    "uz": "uzb", "vi": "vie", "yi": "yid", "yo": "yor", "yue": "yue",
    "zh": "chi",
}

# 639-2 codes for what is not a language: `zxx` for content with none at all
# (a module's machine-readable payloads -- decoded barcodes, telemetry), `und`
# for speech whose language nobody determined, `mul` for several at once.
_NON_LANGUAGE_TAGS = frozenset({"zxx", "und", "mul"})

_LANGUAGE_TAG_VALUES = frozenset(LANGUAGE_TAGS.values()) | _NON_LANGUAGE_TAGS


def _local_use_tag(written: str) -> bool:
    """Whether `written` is in 639-2's `qaa`-`qtz` range, reserved for local use.

    A container may carry one, and the standard promises no code will ever be
    assigned there, so a module tagging its own kind of track picks from here.
    """
    return (
        len(written) == 3
        and written[0] == "q"
        and "a" <= written[1] <= "t"
        and "a" <= written[2] <= "z"
    )

_DESCRIBE_FLAG = "--describe"
_INVOKE_FLAG = "--invoke"

# How long ONE compile-time sidecar call may take: describing a module,
# running a value function, probing a source, reading a sink's rows. It is
# not an inference budget. A module that runs a model spends most of it
# before the first tensor is computed -- reading a file that is hundreds of
# megabytes, loading the ONNX Runtime, and starting the execution provider,
# which on a GPU is tens of seconds the first time. Half a gigabyte is a real
# model size here, and that is twenty seconds of reading on its own where the
# volume is busy, so the default is minutes rather than the second a warm
# call takes. `probe.RETRY_TIMEOUT_SECONDS` is 60 for a plain ffprobe of a
# file; this covers the same read plus a model, and doubles it.
DEFAULT_TIMEOUT_SECONDS = 120.0

# Where a machine says it wants longer, or shorter.
TIMEOUT_ENV = "FFRWD_WASM_TIMEOUT"


def timeout_seconds() -> float:
    """The compile-time budget: what `TIMEOUT_ENV` names, or the default.

    A variable holding something that is not a positive number of seconds --
    left empty by a shell, or filled with ``30s`` -- takes the default, the
    way every other setting this CLI reads out of the environment does. A
    stray variable is not a reason to refuse a compile.
    """
    try:
        asked = float(os.environ.get(TIMEOUT_ENV, ""))
    except ValueError:
        return DEFAULT_TIMEOUT_SECONDS
    return asked if asked > 0 else DEFAULT_TIMEOUT_SECONDS


def _budget_hint(budget: float) -> str:
    """Why a compile-time call ran out of time, and how to give it more."""
    return (
        f"one compile-time call to the sidecar is budgeted {budget:g}s, and "
        "reading the module and loading any model it runs count against it: "
        f"set {TIMEOUT_ENV} to a larger number of seconds"
    )

# How the sidecar is told a side carries annotations beside the frames.
_ANNOTATIONS_FLAG = "-annotations"

# How a rows module is told whose rows it reads: the position, among this
# command line's ``-m`` flags, of the module producing them.
_ROWS_FROM_FLAG = "-rows-from"

# How a rows DOCUMENT is told whose rows it holds, counted the same way. One
# document on the line needs no such flag: it is the only one there is.
_ROWS_FLAG = "-rows"

# Where a packet FILTER's rows come from, repeated once per rows argument:
# ``-rows-in <arg>=<path>``, the argument named so the sidecar can write it
# onto every row that input delivers.
_ROWS_IN_FLAG = "-rows-in"

# How one of a packet source's outputs is told which track of the module's
# catalog it carries, 0-based. The tracks the outputs name are what the
# source subscribes to.
_TRACK_FLAG = "-track"

# How a model file is bound to the name the module loads it by, and what the
# file is called: the export's own name, beside the module.
_NN_FLAG = "-nn"
_NN_EXCLUDE_FLAG = "-nn-exclude"

# The sidecar's worker-thread cap. Unwritten, the sidecar sizes its own pool.
_JOBS_FLAG = "-jobs"

# The effects a module's own imports can ask the host for, in the order the
# argv writes their grants. `nn` is not among them: a model is bound to a
# name rather than granted.
EFFECTS: tuple[str, ...] = ("http", "udp", "tcp")

# The sidecar flag that grants each effect to one module: one flag per
# capability, named for it.
_GRANT_FLAGS: Mapping[str, str] = {"http": "-http", "udp": "-udp", "tcp": "-tcp"}
MODEL_SUFFIX = ".onnx"
ANNOTATIONS_IN = "in"
ANNOTATIONS_OUT = "out"

INSTALL_HINT = (
    "reinstall ffrwd (the sidecar comes with it on supported platforms), or point "
    f"{binaries.FFRWD_WASM_ENV} at an ffrwd-wasm binary"
)


@dataclass(frozen=True)
class DescribedFunction:
    """One value function inside a module's ``functions`` list.

    `params_schema` is the JSON Schema its arguments take, keyed by SQL
    parameter name; `result_schema` is the JSON Schema of what it returns,
    or None when the sidecar named none.
    """

    name: str
    params_schema: Mapping[str, object] = field(default_factory=dict)
    result_schema: Mapping[str, object] | None = None


@dataclass(frozen=True)
class Described:
    """One wasm module's declared interface, as the sidecar reports it.

    `world` is the wit world the module targets. A STREAM module names its
    single export as `name`, that export's own version as `version`,
    `params_schema` the JSON Schema its parameters take, `rows_schema` the
    schema of the rows it emits or None when it emits none, and either
    `pixel_formats` or `sample_formats` — whichever kind of stream it
    filters; a VALUES-only module leaves all of them at their defaults and
    lists what it offers in `functions` instead. `sample_rates` and
    `channel_counts` are what an audio module accepts beside its sample
    format, empty where it accepts anything. `reads_rows` is the sidecar's
    own answer to the question lowering asks — does this module ACT on
    incoming rows — read through `reads_annotations`; `meta` is the older
    flag it falls back to for a sidecar that predates the key.

    `window`, `stride`, `pure` and `one_to_one` are the export's declared
    SHAPE: how many frames it reads to produce one, how far it advances
    between them, whether it carries no state across calls, and whether it
    emits exactly one frame per frame it is handed. `windowed` is whether the
    description declared any of them. One that declared none -- every module
    built before the sidecar published them -- takes these defaults, which is
    what a plain filter module does anyway.

    `inputs` is how many streams the export reads at once, 1 for a module
    that names none. `nn` is whether the export runs a model, which is what
    puts a ``-nn`` binding on the sidecar's own command line. `http`, `udp`
    and `tcp` are the effects the module imports, each of which puts the
    matching grant -- ``-http``, ``-udp``, ``-tcp`` -- on that command line.
    The two socket effects are read apart: `wasi:sockets` splits its
    protocols into interfaces of their own, so a module that imports
    ``wasi:sockets/udp`` needs `udp` and one that imports
    ``wasi:sockets/tcp`` needs `tcp`, and neither implies the other.

    `video_codecs` is present exactly for a PACKET SINK -- a sink module
    that consumes the encoder's own output rather than decoded frames --
    and lists the ffmpeg VIDEO codec names it accepts, most preferred
    first, empty for every codec; `audio_codecs` says the same of audio. A packet
    sink fills neither format list, so its `kind` is None; what it consumes
    is said by the COPY that encodes for it.

    `video_streams` and `audio_streams` say how many streams of each kind the
    sink reads -- ``"none"``, ``"one"`` or ``"many"``. A sink built against a
    world before 0.12.0 read exactly one video stream, which is what a
    description with neither key means.

    `source` marks a PACKET SOURCE -- a module that PRODUCES coded packets
    rather than consuming or filtering them -- the same boolean-flag
    convention `nn`/`http`/`udp` already use, false for every module built
    before 0.13.0.

    `rows_module` marks a ROWS MODULE -- one that reads JSON rows and writes
    JSON rows with no stream anywhere -- and `input_rows_schema` is the shape
    it READS, beside `rows_schema`'s shape it writes. False and None for
    every other kind.
    """

    world: str
    name: str = ""
    version: str = ""
    params_schema: Mapping[str, object] = field(default_factory=dict)
    rows_schema: Mapping[str, object] | None = None
    pixel_formats: tuple[str, ...] = ()
    sample_formats: tuple[str, ...] = ()
    sample_rates: tuple[int, ...] = ()
    channel_counts: tuple[int, ...] = ()
    functions: tuple[DescribedFunction, ...] = ()
    # The module's own parameters, in preference order, whose value says what
    # language its rows are in. Empty for a module that names none, and for
    # every module built before the declaration existed.
    rows_language: tuple[str, ...] = ()
    meta: bool = False
    reads_rows: bool | None = None
    # Whether upstream rows may appear on this module's output frames. None
    # where the sidecar said nothing: an older one that predates the
    # declaration, and every module with no frames for a row to leave on --
    # a packet sink, a source, a rows module, a packet filter -- each of
    # which carries rows its own way and has no answer to give here.
    forwards_rows: bool | None = None
    window: int = 1
    stride: int = 1
    pure: bool = True
    one_to_one: bool = True
    windowed: bool = False
    inputs: int = 1
    nn: bool = False
    # Whether the module imports wasi:http, wasi:sockets/udp or
    # wasi:sockets/tcp, and so runs only under the sidecar's matching
    # ``-http`` / ``-udp`` / ``-tcp`` grant.
    http: bool = False
    udp: bool = False
    tcp: bool = False
    video_codecs: tuple[str, ...] | None = None
    audio_codecs: tuple[str, ...] = ()
    video_streams: SinkArity = "one"
    audio_streams: SinkArity = "none"
    # How much of a stream a packet sink has to be handed: a request the
    # host may exceed and must not fall short of. "all" for a sink built
    # against a world with no field for it.
    wants: SinkWants = "all"
    source: bool = False
    # Whether the module's export is a packet FILTER rather than a sink: the
    # two declare the same codecs and arities, and this is what tells them
    # apart.
    packet_filter: bool = False
    rows_module: bool = False
    input_rows_schema: Mapping[str, object] | None = None

    @property
    def packet_sink(self) -> bool:
        """True for a module whose export CONSUMES encoded packets.

        A packet filter fills the same codec and arity fields, so the flag
        beside them is what keeps a filter out of every sink path.
        """
        return self.video_codecs is not None and not self.packet_filter

    def sink_streams(self, kind: StreamType) -> SinkArity:
        """How many streams of `kind` this sink reads."""
        return self.audio_streams if kind == "audio" else self.video_streams

    def sink_codecs(self, kind: StreamType) -> tuple[str, ...]:
        """The codecs this sink accepts for `kind`; empty is every codec."""
        return self.audio_codecs if kind == "audio" else (self.video_codecs or ())

    @property
    def kind(self) -> StreamType | None:
        """The stream kind this module filters, or None when it names neither.

        Which format list the description filled says it: pixel formats are a
        video module's, sample formats an audio one's. A module that filled
        BOTH is a rejection its caller raises, not an answer here.
        """
        if self.pixel_formats and self.sample_formats:
            return None
        if self.sample_formats:
            return "audio"
        return "video" if self.pixel_formats else None

    @property
    def both_kinds(self) -> bool:
        """True for a module that named pixel formats AND sample formats."""
        return bool(self.pixel_formats and self.sample_formats)

    @property
    def shape(self) -> ModuleShape:
        """This export's frame timing, as partitioning reads it."""
        return ModuleShape(
            window=self.window,
            stride=self.stride,
            one_to_one=self.one_to_one,
            pure=self.pure,
        )

    @property
    def reads_annotations(self) -> bool:
        """True for a module built to CONSUME annotations off its frames.

        The sidecar declares it as `reads_rows`; an older sidecar without
        the key falls back to `meta`, which windowed modules also set for
        mere availability — hence the windowed exclusion there.
        """
        if self.reads_rows is not None:
            return self.reads_rows
        return self.meta and not self.windowed


# Reads one module path and returns what it declares, or raises FfrwdError.
# :func:`describe` is the real one; a lowering test passes its own.
Describe = Callable[[str], Described]

# Runs one module's function with a JSON-object argument and returns its
# parsed JSON result, or raises FfrwdError. :func:`invoke` is the real one; a
# lowering test passes its own, so folding a value needs no sidecar.
#
# `described` is optional and defaults to None (no grant) so a caller that
# has not read the module's description yet -- or a lowering test's own
# fake -- need not pass one; the real `invoke` reads `described.http` and
# `described.udp` off it to grant the run the effects the module imports.
class Invoke(Protocol):
    def __call__(
        self,
        path: str,
        function: str,
        args: Mapping[str, object],
        *,
        described: Described | None = None,
    ) -> object: ...


def _reject(message: str, hint: str) -> FfrwdError:
    """A describe failure, unanchored -- the caller re-anchors on the declaration."""
    return FfrwdError(ErrorCode.UNSUPPORTED_SQL, message, hint=hint)


def _strings(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, str))


def _positives(value: object) -> tuple[int, ...]:
    """A declared list of rates or channel counts, ignoring anything else."""
    if not isinstance(value, list):
        return ()
    return tuple(
        item
        for item in value
        if isinstance(item, int) and not isinstance(item, bool) and item > 0
    )


def _functions(value: object) -> tuple[DescribedFunction, ...]:
    """The ``functions`` list a values module's describe carries, else ``()``."""
    if not isinstance(value, list):
        return ()
    found: list[DescribedFunction] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        fn_name = item.get("name")
        if not isinstance(fn_name, str):
            continue
        params = item.get("params_schema")
        result = item.get("result_schema")
        found.append(
            DescribedFunction(
                name=fn_name,
                params_schema=params if isinstance(params, dict) else {},
                result_schema=result if isinstance(result, dict) else None,
            )
        )
    return tuple(found)


def _described(path: str, payload: object) -> Described:
    """One ``--describe`` document as a :class:`Described`.

    A STREAM module names its export as `name`; a values-only module leaves
    `name` null and lists what it offers in `functions` instead. Neither is a
    rejection on its own -- only a document with NEITHER is one, since it
    says nothing this compiler can wire.
    """
    if not isinstance(payload, dict):
        raise _reject(
            f"the sidecar described {path} with something that is not an object",
            hint="the module may be built against a sidecar this ffrwd does not know",
        )
    world = payload.get("world")
    if not isinstance(world, str):
        raise _reject(
            f"the sidecar's description of {path} names no world",
            hint="the module may be built against a sidecar this ffrwd does not know",
        )
    name = payload.get("name")
    functions = _functions(payload.get("functions"))
    if not isinstance(name, str) and not functions:
        raise _reject(
            f"the sidecar's description of {path} names no export and no functions",
            hint="the module may be built against a sidecar this ffrwd does not know",
        )
    version = payload.get("version")
    params = payload.get("params_schema")
    rows = payload.get("rows_schema")
    return Described(
        world=world,
        name=name if isinstance(name, str) else "",
        version=version if isinstance(version, str) else "",
        params_schema=params if isinstance(params, dict) else {},
        rows_schema=rows if isinstance(rows, dict) else None,
        pixel_formats=_strings(payload.get("pixel_formats")),
        sample_formats=_strings(payload.get("sample_formats")),
        sample_rates=_positives(payload.get("sample_rates")),
        channel_counts=_positives(payload.get("channel_counts")),
        functions=functions,
        rows_language=_strings(payload.get("rows_language")),
        meta=payload.get("meta") is True,
        reads_rows=payload["reads_rows"]
        if isinstance(payload.get("reads_rows"), bool)
        else None,
        forwards_rows=payload["forwards_rows"]
        if isinstance(payload.get("forwards_rows"), bool)
        else None,
        window=_count(payload.get("window")),
        stride=_count(payload.get("stride")),
        pure=payload.get("pure") is not False,
        one_to_one=payload.get("one_to_one") is not False,
        windowed=any(key in payload for key in _SHAPE_KEYS),
        inputs=_count(payload.get("inputs")),
        nn=payload.get("nn") is True,
        http=payload.get("http") is True,
        udp=payload.get("udp") is True,
        tcp=payload.get("tcp") is True,
        # Present only for a packet sink; its ABSENCE is what marks every
        # other module, so an absent key stays None rather than ().
        video_codecs=_strings(payload["video_codecs"])
        if isinstance(payload.get("video_codecs"), list)
        else None,
        audio_codecs=_strings(payload.get("audio_codecs")),
        # A sink built before the counts existed read one video stream.
        video_streams=_sink_arity(payload.get("video_streams"), "one"),
        audio_streams=_sink_arity(payload.get("audio_streams"), "none"),
        wants=cast(SinkWants, payload["wants"])
        if payload.get("wants") in _SINK_WANTS
        else "all",
        source=payload.get("source") is True,
        packet_filter=payload.get("packet_filter") is True,
        rows_module=payload.get("rows_module") is True,
        input_rows_schema=payload["input_rows_schema"]
        if isinstance(payload.get("input_rows_schema"), dict)
        else None,
    )


def _sink_arity(value: object, absent: SinkArity) -> SinkArity:
    """A sink's declared stream count for one kind, or what an older one meant."""
    if value in _SINK_ARITIES:
        return cast(SinkArity, value)
    return absent


# The keys a windowed export's description carries, and no other export's.
_SHAPE_KEYS = ("window", "stride", "pure", "one_to_one")


def _count(value: object) -> int:
    """A declared frame count, or 1 for a description that names none."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return 1
    return value


def describe(path: str) -> Described:
    """Ask the sidecar what the module at `path` declares.

    Raises ``FfrwdError`` -- and nothing else -- when the sidecar is not
    installed, cannot read the module, or answers with something that is not
    a description. The rejection carries no position; the caller anchors it on
    the declaration that named the module.
    """
    sidecar = binaries.ffrwd_wasm_path()
    if sidecar is None:
        raise _reject(
            "the ffrwd-wasm sidecar is not installed, and a LANGUAGE wasm "
            "function needs it to read the module",
            hint=INSTALL_HINT,
        )
    budget = timeout_seconds()
    try:
        done = subprocess.run(
            [sidecar, _DESCRIBE_FLAG, path],
            capture_output=True,
            # The sidecar writes UTF-8; the locale codec would be wrong AND
            # brittle -- a byte it cannot decode kills the reader thread and
            # hands back None instead of text.
            encoding="utf-8",
            errors="replace",
            timeout=budget,
            check=False,
        )
    except (OSError, ValueError) as err:
        # ValueError is the spawn refusing the argv itself (an embedded NUL
        # in a written path); the same rejection serves.
        raise _reject(
            f"could not run the ffrwd-wasm sidecar at {sidecar}: "
            f"{getattr(err, 'strerror', None) or err}",
            hint=INSTALL_HINT,
        ) from err
    except subprocess.TimeoutExpired as err:
        raise _reject(
            f"the ffrwd-wasm sidecar did not describe {path} within "
            f"{budget:g}s",
            hint=_budget_hint(budget),
        ) from err
    if done.returncode != 0:
        raise _reject(
            f"the ffrwd-wasm sidecar could not describe {path}: "
            f"{_first_line(done.stderr)}",
            hint="check the path names a wasm component built for the ffrwd world",
        )
    try:
        payload = json.loads(done.stdout)
    except ValueError as err:
        raise _reject(
            f"the ffrwd-wasm sidecar's description of {path} is not JSON",
            hint="the sidecar on PATH may be a different version than this ffrwd",
        ) from err
    return _described(path, payload)


def _grant_args(described: Described, path: str) -> list[str]:
    """The ``-http``/``-udp``/``-tcp`` grants `described`'s own imports need
    for `path`, which :func:`invoke` and :func:`probe_source` both put ahead
    of the flag that dispatches their call."""
    argv: list[str] = []
    for effect in EFFECTS:
        if getattr(described, effect):
            argv += [_GRANT_FLAGS[effect], path]
    return argv


def invoke(
    path: str,
    function: str,
    args: Mapping[str, object],
    *,
    described: Described | None = None,
) -> object:
    """Run `function` inside the module at `path` on `args`, and return its result.

    ``ffrwd-wasm --invoke <module> <function> '<args-json>'`` prints one JSON
    value on stdout and exits 0, or writes a message to stderr and exits
    nonzero. `args` is marshalled to one JSON object keyed by the module's own
    parameter names.

    `described` is the module's own declared interface, when the caller has
    already read one -- the same :class:`Described` :func:`describe` returns.
    A module that imports `wasi:http` or `wasi:sockets` needs its effect
    granted the same way a run grants it: ``-http``/``-udp <path>`` ahead of
    ``--invoke``, the sidecar's own argv order. A module that runs a model
    gets the same ``-nn name=path`` binding, plus ``-nn-runtime``/
    ``-nn-target``, ahead of those grants -- :func:`model_binding` names the
    file, and raises the same "runs a model, and '<path>' is not there"
    rejection a hosted module's compile-time check does when it is missing.
    `described` left at its default of None grants nothing, which is what a
    caller with no description in hand -- one invoking a module it knows
    imports nothing and runs no model -- gets.

    Raises ``FfrwdError`` -- and nothing else -- when the sidecar is not
    installed, the module runs a model with no file beside it, the module
    rejects the call, or the answer is not JSON. The rejection carries no
    position; the caller anchors it on the call site.
    """
    sidecar = binaries.ffrwd_wasm_path()
    if sidecar is None:
        raise _reject(
            "the ffrwd-wasm sidecar is not installed, and a LANGUAGE wasm "
            "value function needs it to run",
            hint=INSTALL_HINT,
        )
    payload = json.dumps(args, sort_keys=True)
    argv = [sidecar]
    if described is not None:
        if described.nn:
            argv += _nn_args((model_binding(described, path),), nn.spawn_args())
        argv += _grant_args(described, path)
    argv += [_INVOKE_FLAG, path, function, payload]
    budget = timeout_seconds()
    try:
        done = subprocess.run(
            argv,
            capture_output=True,
            encoding="utf-8",  # what the sidecar writes; see describe()
            errors="replace",
            timeout=budget,
            check=False,
        )
    except (OSError, ValueError) as err:
        raise _reject(
            f"could not run the ffrwd-wasm sidecar at {sidecar}: "
            f"{getattr(err, 'strerror', None) or err}",
            hint=INSTALL_HINT,
        ) from err
    except subprocess.TimeoutExpired as err:
        raise _reject(
            f"the ffrwd-wasm sidecar did not run {function}() within "
            f"{budget:g}s",
            hint=_budget_hint(budget),
        ) from err
    if done.returncode != 0:
        raise _reject(
            f"the module '{path}' rejected the call to {function}(): "
            f"{_first_line(done.stderr)}",
            hint="check the arguments match what the module declares",
        )
    try:
        return json.loads(done.stdout)
    except ValueError as err:
        raise _reject(
            f"the ffrwd-wasm sidecar's result from {function}() is not JSON",
            hint="the sidecar on PATH may be a different version than this ffrwd",
        ) from err


# -- what a packet source publishes ----------------------------------------
#
# A packet source is a module that PRODUCES coded packets: `probe_source`
# reads its compile-time catalog, `catalog_as_probe` bridges that catalog
# into the shape everything downstream already reads a probed FILE as.

_PROBE_FLAG = "--probe"
_PARAMS_FLAG = "-params"


@dataclass(frozen=True)
class SourceRendition:
    """One track's ABR row, as a packet source's ``probe`` reports it.

    The wire's own ``rendition`` object -- name, bandwidth and codecs string
    exactly as the manifest or catalog said them, None where nothing did --
    not the fuller :class:`ffrwd.probe.RenditionMeta`, whose `streams`,
    `width`, `height` and `program_id` a packet source never reports per
    track.
    """

    name: str | None
    bandwidth: int | None
    codecs: str | None
    language: str | None


@dataclass(frozen=True)
class SourceTrack:
    """One coded track a packet source's ``probe`` reports.

    `kind` is read off which arm -- ``video`` or ``audio`` -- the wire's
    ``format`` filled; `width`/`height` come from the video arm,
    `sample_rate`/`channels` from the audio one, and the pair for the other
    kind stays None. `extradata` is the codec's out-of-band header, decoded
    from the wire's hex. `row` is which relation row this track belongs to;
    `rendition` is what the source read of that row.
    """

    codec: str
    time_base: tuple[int, int]
    kind: StreamType
    width: int | None
    height: int | None
    sample_rate: int | None
    channels: int | None
    extradata: bytes
    profile: int | None
    level: int | None
    row: int
    rendition: SourceRendition


@dataclass(frozen=True)
class SourceCatalog:
    """What a packet source module's ``probe(params)`` answers.

    `tracks` is every coded track it publishes, in the source's own order;
    `bounded` is whether it ever ends -- False is what :func:`catalog_as_probe`
    reads as `live`.
    """

    tracks: tuple[SourceTrack, ...]
    bounded: bool


def _int_or_none(value: object) -> int | None:
    """An optional declared int off a probe payload, treating a bool as absent."""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _source_rendition(value: object) -> SourceRendition:
    """A track's ``rendition`` object, or one naming nothing when it is absent."""
    if not isinstance(value, dict):
        return SourceRendition(name=None, bandwidth=None, codecs=None, language=None)
    name = value.get("name")
    codecs = value.get("codecs")
    language = value.get("language")
    return SourceRendition(
        name=name if isinstance(name, str) else None,
        bandwidth=_int_or_none(value.get("bandwidth")),
        codecs=codecs if isinstance(codecs, str) else None,
        language=language if isinstance(language, str) else None,
    )


def _source_format(
    module: str, index: int, value: object
) -> tuple[StreamType, int | None, int | None, int | None, int | None]:
    """A track's ``format`` arm as ``(kind, width, height, sample_rate, channels)``.

    Exactly one of the video pair and the audio pair is filled; the other
    stays ``(None, None)``. Neither arm present, or the named arm not an
    object, is a rejection naming the module and the track.
    """
    if isinstance(value, dict) and isinstance(value.get("video"), dict):
        video = value["video"]
        return (
            "video",
            _int_or_none(video.get("width")),
            _int_or_none(video.get("height")),
            None,
            None,
        )
    if isinstance(value, dict) and isinstance(value.get("audio"), dict):
        audio = value["audio"]
        return (
            "audio",
            None,
            None,
            _int_or_none(audio.get("sample_rate")),
            _int_or_none(audio.get("channels")),
        )
    raise _reject(
        f"track {index} of the sidecar's probe of {module} names a format "
        "that is neither video nor audio",
        hint="the module may be built against a sidecar this ffrwd does not know",
    )


def _extradata(module: str, index: int, value: object) -> bytes:
    """A track's ``extradata`` hex string as bytes, or a rejection naming it."""
    if isinstance(value, str):
        try:
            return bytes.fromhex(value)
        except ValueError:
            pass
    raise _reject(
        f"track {index} of the sidecar's probe of {module} names extradata "
        "that is not hex",
        hint="the module may be built against a sidecar this ffrwd does not know",
    )


def _time_base(module: str, index: int, value: object) -> tuple[int, int]:
    """A track's ``time_base`` pair, or a rejection naming the module and track."""
    if (
        isinstance(value, list)
        and len(value) == 2
        and all(isinstance(n, int) and not isinstance(n, bool) for n in value)
    ):
        return (value[0], value[1])
    raise _reject(
        f"track {index} of the sidecar's probe of {module} names no time_base",
        hint="the module may be built against a sidecar this ffrwd does not know",
    )


def _source_track(module: str, index: int, raw: object) -> SourceTrack:
    """One entry of the probe payload's ``tracks`` list as a :class:`SourceTrack`."""
    if not isinstance(raw, dict):
        raise _reject(
            f"track {index} of the sidecar's probe of {module} is not an object",
            hint="the module may be built against a sidecar this ffrwd does not know",
        )
    codec = raw.get("codec")
    if not isinstance(codec, str):
        raise _reject(
            f"track {index} of the sidecar's probe of {module} names no codec",
            hint="the module may be built against a sidecar this ffrwd does not know",
        )
    row = raw.get("row")
    if not isinstance(row, int) or isinstance(row, bool):
        raise _reject(
            f"track {index} of the sidecar's probe of {module} names no row",
            hint="the module may be built against a sidecar this ffrwd does not know",
        )
    kind, width, height, sample_rate, channels = _source_format(
        module, index, raw.get("format")
    )
    return SourceTrack(
        codec=codec,
        time_base=_time_base(module, index, raw.get("time_base")),
        kind=kind,
        width=width,
        height=height,
        sample_rate=sample_rate,
        channels=channels,
        extradata=_extradata(module, index, raw.get("extradata")),
        profile=_int_or_none(raw.get("profile")),
        level=_int_or_none(raw.get("level")),
        row=row,
        rendition=_source_rendition(raw.get("rendition")),
    )


def _source_catalog(module: str, payload: object) -> SourceCatalog:
    """One ``probe`` document as a :class:`SourceCatalog`."""
    if not isinstance(payload, dict):
        raise _reject(
            f"the sidecar probed {module} with something that is not an object",
            hint="the module may be built against a sidecar this ffrwd does not know",
        )
    raw_tracks = payload.get("tracks")
    if not isinstance(raw_tracks, list):
        raise _reject(
            f"the sidecar's probe of {module} names no tracks",
            hint="the module may be built against a sidecar this ffrwd does not know",
        )
    tracks = tuple(_source_track(module, i, raw) for i, raw in enumerate(raw_tracks))
    return SourceCatalog(tracks=tracks, bounded=payload.get("bounded") is True)


# Runs one packet-source module's `probe` for its compile-time catalog:
# :func:`probe_source` is the real one, and a lowering test passes its own.
class ProbeSource(Protocol):
    def __call__(
        self,
        module: str,
        params: str,
        *,
        described: Described | None = None,
    ) -> SourceCatalog: ...


def probe_source(module: str, params: str, *, described: Described | None = None) -> SourceCatalog:
    """Ask the sidecar what the packet-source module at `module` publishes for `params`.

    ``ffrwd-wasm --probe <module> -params '<json>'`` prints one JSON line naming
    every coded track the module would produce and whether the source ever
    ends. `params` travels verbatim -- already marshalled JSON, the same
    convention :func:`invoke` follows for its own argument.

    A source that imports `wasi:http` or `wasi:sockets` needs that effect
    granted to answer its own probe, the same way :func:`invoke` grants it,
    so `described` carries the module's declared interface. Left at None it
    grants nothing.

    Raises ``FfrwdError`` -- and nothing else -- when the sidecar is not
    installed, cannot probe the module, or answers with something that is not
    the documented shape. The rejection carries no position; the caller
    anchors it on the call that named the module.
    """
    sidecar = binaries.ffrwd_wasm_path()
    if sidecar is None:
        raise _reject(
            f"the ffrwd-wasm sidecar is not installed, and probing '{module}' "
            "needs it to read the module",
            hint=INSTALL_HINT,
        )
    argv = [sidecar]
    if described is not None:
        argv += _grant_args(described, module)
    argv += [_PROBE_FLAG, module, _PARAMS_FLAG, params]
    budget = timeout_seconds()
    try:
        done = subprocess.run(
            argv,
            capture_output=True,
            encoding="utf-8",  # what the sidecar writes; see describe()
            errors="replace",
            timeout=budget,
            check=False,
        )
    except (OSError, ValueError) as err:
        raise _reject(
            f"could not run the ffrwd-wasm sidecar at {sidecar}: "
            f"{getattr(err, 'strerror', None) or err}",
            hint=INSTALL_HINT,
        ) from err
    except subprocess.TimeoutExpired as err:
        raise _reject(
            f"the ffrwd-wasm sidecar did not probe {module} within "
            f"{budget:g}s",
            hint=_budget_hint(budget),
        ) from err
    if done.returncode != 0:
        raise _reject(
            f"the ffrwd-wasm sidecar could not probe {module}: "
            f"{_first_line(done.stderr)}",
            hint="check the path names a wasm component built for the ffrwd world",
        )
    try:
        payload = json.loads(done.stdout)
    except ValueError as err:
        raise _reject(
            f"the ffrwd-wasm sidecar's probe of {module} is not JSON",
            hint="the sidecar on PATH may be a different version than this ffrwd",
        ) from err
    return _source_catalog(module, payload)


def catalog_as_probe(alias: str, catalog: SourceCatalog) -> ProbeResult:
    """A packet source's catalog as a :class:`~ffrwd.probe.ProbeResult`.

    The bridge from a compile-time SOURCE probe to everything downstream
    that already reads a :class:`ProbeResult` the way ffprobe hands one
    over. One `StreamMeta` per track, catalog order, its per-type `index`
    counted the way ffprobe counts one -- 0-based, video and audio counted
    separately. One `RenditionMeta` per distinct `row`, first-seen order,
    holding that row's own streams plus the attributes its FIRST track's
    `rendition` named -- muxed tracks of one row agree on them in practice.

    `live` is the negation of `bounded`: an unbounded source reads exactly
    like a live manifest to everything that reads `ProbeResult.live`.
    `format_name` names the packet-source kind, the same way `format_name`
    already says a probed file is a webvtt document. `alias` is accepted for
    symmetry with a call site that keys its inputs by alias; nothing here
    reads its value.
    """
    video_index = 0
    audio_index = 0
    streams: list[StreamMeta] = []
    row_order: list[int] = []
    streams_by_row: dict[int, list[StreamMeta]] = {}
    rendition_by_row: dict[int, SourceRendition] = {}
    for track in catalog.tracks:
        if track.kind == "video":
            index, video_index = video_index, video_index + 1
        else:
            index, audio_index = audio_index, audio_index + 1
        stream = StreamMeta(
            type=track.kind,
            index=index,
            metadata={},
            width=track.width,
            height=track.height,
            fps=None,
            sample_rate=track.sample_rate,
            codec=track.codec,
            channels=track.channels,
        )
        streams.append(stream)
        if track.row not in streams_by_row:
            row_order.append(track.row)
            streams_by_row[track.row] = []
            rendition_by_row[track.row] = track.rendition
        streams_by_row[track.row].append(stream)

    renditions: list[RenditionMeta] = []
    for row in row_order:
        row_streams = streams_by_row[row]
        video = next((s for s in row_streams if s.type == "video"), None)
        rendition = rendition_by_row[row]
        renditions.append(
            RenditionMeta(
                streams=row_streams,
                bandwidth=rendition.bandwidth,
                width=video.width if video is not None else None,
                height=video.height if video is not None else None,
                codecs=rendition.codecs,
                name=rendition.name,
                language=rendition.language,
                program_id=None,
            )
        )

    return ProbeResult(
        streams=streams,
        format_name="packet-source",
        renditions=renditions,
        live=not catalog.bounded,
    )


def _first_line(text: str) -> str:
    """The first non-blank line the sidecar wrote, for a message."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return "it wrote nothing"


def _last_line(text: str) -> str:
    """The last non-blank line a process wrote, for a message.

    ffmpeg says what it was doing first and what went wrong last, and a
    sidecar that fails mid-run does the same: the last line is the cause.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else "it wrote nothing"


# -- reading a packet sink's rows while compiling --------------------------
#
# The run-time arrangement, run now: one ffmpeg stream-copying one stream as
# coded NUT onto a pipe, one sidecar hosting the sink on the other end,
# writing its rows as NDJSON. `read_packet_rows` runs the pair; `lower` is
# what decides there is a question to ask.

# The rows document a packet sink writes, and the specifier every flag on the
# copy's one output stream carries.
_ROWS_FORMAT = "ndjson"
_OUTPUT_STREAM = "0"

# How a `-map` names one stream of a kind, ffmpeg's own letters.
_TYPE_SPECIFIERS: Mapping[StreamType, str] = {"video": "v", "audio": "a"}

# The keyframes, without relying on a demuxer to skip: `noise` drops every
# packet decoding cannot start at. `-discard nokey` reads less of the file,
# but mov is the only demuxer that honours it AND its keyframe pts come back
# wrong on a stream that reorders, so this is the one path for every
# container.
_KEYFRAMES_FILTER = "noise=drop=not(key)"
# What puts the copy on the same clock the query's own times are on. A stream
# that reorders frames opens with a negative dts, and the muxer's
# `avoid_negative_ts` would then shift every timestamp -- pts included -- by
# that lead-in. Lifting the dts instead leaves pts alone, so what a sink
# reports is what `trim` means.
_CLOCK_FILTER = "setts=dts=DTS-STARTDTS"


@dataclass(frozen=True)
class PacketRead:
    """One compile-time read: which stream of which file, through which module.

    `spec` and `input_args` are the input exactly as the query wrote it -- the
    same path and the same per-input options the probe was taken under, so
    the demuxer sees what it will see at run time. `kind` and `index` name
    the stream, `index` 0-based within its kind, the way a ``-map`` counts.
    `params` is the module's own parameters, already marshalled the way
    :func:`invoke` marshals its argument. `wants` is how much of the stream
    the module asked for.
    """

    spec: str
    input_args: tuple[str, ...]
    kind: StreamType
    index: int
    module: str
    params: str
    wants: SinkWants


# Runs one packet sink over one stream and returns the rows it wrote:
# :func:`read_packet_rows` is the real one, and a lowering test passes its own
# so the unit tier spawns neither ffmpeg nor the sidecar.
class ReadPackets(Protocol):
    def __call__(
        self,
        read: PacketRead,
        *,
        described: Described | None = None,
    ) -> tuple[dict[str, object], ...]: ...


def _bsf_chain(wants: SinkWants, attempt: int) -> str:
    """The bitstream filters this attempt puts on the copied stream.

    Attempt 0 is what `wants` asked for. Each later attempt drops one filter,
    widening the copy rather than failing: an ffmpeg whose `noise` cannot take
    the expression hands over the whole stream, which `wants` allows, and one
    with no `setts` at all hands it over on whatever clock it writes.
    """
    filters = []
    if wants == "keyframes" and attempt < 1:
        filters.append(_KEYFRAMES_FILTER)
    if attempt < 2:
        filters.append(_CLOCK_FILTER)
    return ",".join(filters)


def copy_argv(ffmpeg: str, read: PacketRead, attempt: int = 0) -> list[str]:
    """The ffmpeg command that copies `read`'s stream as coded NUT onto a pipe.

    One stream, stream-copied, no decode. `wants` shapes what is copied and
    nothing else: ``first`` stops after one packet, ``keyframes`` drops what
    decoding cannot start at, and ``all`` copies the stream whole. A host may
    hand a sink more than it asked for, so every widening is legal and only
    handing over less would not be.
    """
    argv = [ffmpeg, "-v", "error", *read.input_args, "-i", read.spec]
    argv += ["-map", f"0:{_TYPE_SPECIFIERS[read.kind]}:{read.index}"]
    if read.wants == "first":
        argv += [f"-frames:{_OUTPUT_STREAM}", "1"]
    argv += ["-c", "copy"]
    chain = _bsf_chain(read.wants, attempt)
    if chain:
        argv += [f"-bsf:{_OUTPUT_STREAM}", chain]
    return [*argv, "-f", EDGE_FORMAT, STDOUT]


def _reader_argv(binary: str, read: PacketRead, described: Described | None) -> list[str]:
    """The sidecar command that hosts the sink on the other end of the pipe.

    Built as a one-module region reading stdin and writing one rows document,
    which is the same argv a run-time packet sink takes -- the arrangement is
    not a second one, it is this one run early.
    """
    process = SidecarProcess(
        id="read",
        module=read.module,
        node="read",
        args=_params_object(read.params),
        sink=True,
        packet_sink=True,
        rows=(RowsDocument(sink=RowsSink(container=_ROWS_FORMAT), node="read"),),
        grants=tuple(
            EffectGrant(effect=effect, module=read.module)
            for effect in EFFECTS
            if described is not None and getattr(described, effect)
        ),
    )
    return _argv(binary, process, reads=(STDIN,), writes=(STDOUT,))


def _params_object(params: str) -> dict[str, object]:
    """A marshalled params string back as the object the argv writes."""
    if not params:
        return {}
    loaded = json.loads(params)
    return loaded if isinstance(loaded, dict) else {}


def read_packet_rows(
    read: PacketRead, *, described: Described | None = None
) -> tuple[dict[str, object], ...]:
    """Run `read` and return the rows the module wrote, one dict per row.

    ffmpeg copies the stream as coded NUT onto a pipe and the sidecar hosts
    the sink on the other end, writing NDJSON -- the same two processes a
    run-time packet sink is, run while the query compiles.

    A copy ffmpeg refuses is retried WIDER (:func:`_bsf_chain`) rather than
    given up on, since every widening hands the sink more of the stream than
    it asked for and that is always allowed. The ceilings are the probe's own:
    one demux of a track, retried once with room.

    Raises ``FfrwdError`` -- and nothing else -- when ffmpeg or the sidecar is
    missing, the copy fails however it is written, the module rejects the
    stream, or a row is not JSON. The rejection carries no position; the
    caller anchors it on the call that named the module.
    """
    ffmpeg = binaries.ffmpeg_path()
    if ffmpeg is None:
        raise _reject(
            f"ffmpeg is not installed, and reading '{read.spec}' at compile "
            "time needs it to copy the stream",
            hint="install ffmpeg, or put one on PATH",
        )
    binary = binaries.ffrwd_wasm_path()
    if binary is None:
        raise _reject(
            f"the ffrwd-wasm sidecar is not installed, and reading '{read.module}' "
            "needs it to host the module",
            hint=INSTALL_HINT,
        )
    sidecar_command = _reader_argv(binary, read, described)
    failure: FfrwdError | None = None
    for attempt in range(3):
        command = copy_argv(ffmpeg, read, attempt)
        if attempt and command == copy_argv(ffmpeg, read, attempt - 1):
            continue  # nothing left to widen; the previous attempt was this one
        try:
            return _run_read(command, sidecar_command, read)
        except _CopyRefused as refused:
            failure = refused.error
    assert failure is not None  # the loop runs at least one attempt
    raise failure


class _CopyRefused(Exception):
    """ffmpeg would not write the copy: retryable with a wider one."""

    def __init__(self, error: FfrwdError) -> None:
        super().__init__(error.message)
        self.error = error


def _run_read(
    command: Sequence[str], sidecar_command: Sequence[str], read: PacketRead
) -> tuple[dict[str, object], ...]:
    """One attempt: spawn the pair, collect the rows, or raise.

    A failure the copy may be responsible for comes back as
    :class:`_CopyRefused`, which the caller retries with a wider copy; a
    module that rejected a stream it was handed whole is final.

    The second ceiling is the wider of the probe's retry and the module
    budget: a sink is a module like any other, so a machine that raised
    `TIMEOUT_ENV` for the model one of them loads raises this too.
    """
    ceilings = (
        probe.EXTRACT_TIMEOUT_SECONDS,
        max(probe.RETRY_TIMEOUT_SECONDS, timeout_seconds()),
    )
    for ceiling in ceilings:
        try:
            copy, copy_error, rows_text, rows_error, code = _spawn_read(
                command, sidecar_command, ceiling
            )
            break
        except subprocess.TimeoutExpired:
            if ceiling == ceilings[-1]:
                raise _reject(
                    f"reading '{read.spec}' through '{read.module}' did not "
                    f"finish within {ceiling:g}s",
                    hint="a compile-time read copies one stream of the file "
                    "into the module: check the input is reachable and not far "
                    f"slower to read than it is to probe, or raise {TIMEOUT_ENV}",
                ) from None
        except OSError as err:
            raise _reject(
                f"could not read '{read.spec}' through '{read.module}': "
                f"{getattr(err, 'strerror', None) or err}",
                hint=INSTALL_HINT,
            ) from err
    if copy != 0 and code != 0:
        # Both ends died and neither exit code says which one started it: a
        # copy that never wrote leaves the module reading an empty stream,
        # and a module that quit leaves the copy writing into a closed pipe.
        # Saying both is the honest answer, and the copy is still widened in
        # case it was the one that could not start.
        raise _CopyRefused(
            _reject(
                f"reading '{read.spec}' through '{read.module}' failed at both "
                f"ends: ffmpeg said '{_last_line(copy_error)}' and the module "
                f"said '{_last_line(rows_error)}'",
                hint="a compile-time read is one ffmpeg copying the stream "
                "into one sidecar hosting the module; the end that failed "
                "first is the one to look at",
            )
        )
    if code != 0:
        raise _reject(
            f"the module '{read.module}' rejected the stream: "
            f"{_last_line(rows_error)}",
            hint="check the module reads the codec the stream carries, and "
            "that its parameters are the ones it declares",
        )
    if copy != 0:
        raise _CopyRefused(
            _reject(
                f"ffmpeg could not copy the stream out of '{read.spec}': "
                f"{_last_line(copy_error)}",
                hint="a compile-time read stream-copies one stream of the "
                "input; check the file carries that stream",
            )
        )
    return _packet_rows(read.module, rows_text)


def _spawn_read(
    command: Sequence[str], sidecar_command: Sequence[str], ceiling: float
) -> tuple[int, str, str, str, int]:
    """The pair, run to completion: ffmpeg's exit and stderr, then the sidecar's.

    ffmpeg's stderr is drained by a thread of its own: it writes at ``-v
    error`` and so says very little, but a pipe nobody reads is a pipe that
    fills, and a copy blocked on its own diagnostics would never reach the
    ceiling as a timeout.
    """
    with subprocess.Popen(
        list(command),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ) as copy:
        assert copy.stdout is not None and copy.stderr is not None
        diagnostics = copy.stderr
        drained: list[bytes] = []
        drain = threading.Thread(target=lambda: drained.append(diagnostics.read()))
        drain.start()
        try:
            with subprocess.Popen(
                list(sidecar_command),
                stdin=copy.stdout,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            ) as reader:
                # This process's own handle on the pipe closes here, so the
                # sidecar sees the end of the stream when ffmpeg is done.
                copy.stdout.close()
                try:
                    out, err = reader.communicate(timeout=ceiling)
                except subprocess.TimeoutExpired:
                    reader.kill()
                    copy.kill()
                    drain.join()
                    raise
                code = reader.returncode
        finally:
            copy.wait()
            drain.join()
    written = drained[0] if drained else b""
    return (
        copy.returncode,
        written.decode("utf-8", errors="replace"),
        out.decode("utf-8", errors="replace"),
        err.decode("utf-8", errors="replace"),
        code,
    )


def _packet_rows(module: str, text: str) -> tuple[dict[str, object], ...]:
    """The sidecar's NDJSON as one dict per row, or a rejection naming it."""
    rows: list[dict[str, object]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            row = json.loads(stripped)
        except ValueError as err:
            raise _reject(
                f"the module '{module}' wrote a row that is not JSON",
                hint="the sidecar on PATH may be a different version than this ffrwd",
            ) from err
        if not isinstance(row, dict):
            raise _reject(
                f"the module '{module}' wrote a row that is not an object",
                hint="a packet sink's rows are JSON objects, one per line",
            )
        rows.append(row)
    return tuple(rows)


def _nn_args(models: Sequence[ModelBinding], runtime: Sequence[str]) -> list[str]:
    """The ``-nn-runtime``/``-nn-target`` pair, ``-nn-exclude`` per denied provider,
    then one ``-nn name=path`` per model.

    Shared by a sidecar process's argv and a compile-time :func:`invoke` of a
    module that runs one -- both bind a model the same way. `runtime` is
    written only when there is at least one model to bind. ``-nn-exclude`` is
    the union of every bound model's own ``not_on``, sorted: every model on
    one process shares the one provider walk, so one model's pin narrows it
    for the process, not for its own binding alone.
    """
    argv: list[str] = []
    if models:
        argv += list(runtime)
    excluded: set[str] = {provider for model in models for provider in model.not_on}
    for provider in sorted(excluded):
        argv += [_NN_EXCLUDE_FLAG, provider]
    for model in models:
        argv += [_NN_FLAG, f"{model.name}={model.path}"]
    return argv


def _argv(
    binary: str,
    process: SidecarProcess,
    reads: Sequence[str] = (),
    writes: Sequence[str] = (),
    runtime: Sequence[str] = (),
    jobs: int | None = None,
) -> list[str]:
    """One sidecar process as argv, run by `binary`.

    The frame parameters are not on the command line: they ride the NUT
    header on the input, which is why the sidecar refuses ``-pix_fmt``,
    ``-s`` and ``-r``.

    ONE module is spelled the short way: ``-m <path>``, its parameters as
    ``-params``, and a module given none is passed no ``-params`` at all. A
    region of SEVERAL is a network, and is configured the way ffmpeg is --
    ``-m <name>=<path>`` per distinct module and a ``-filter_complex`` string
    wiring the names together, each node's parameters written into it as
    ``k=v:k2=v2``.

    A ROWS module is neither: it carries no stream, so no filtergraph names
    it. It comes after the module table as an ``-m <path>`` of its own,
    followed by ``-rows-from <index>`` -- the 0-based position, among every
    ``-m`` this command line writes, of the module whose rows it reads. The
    stream modules are written first, so a rows module reading another rows
    module's output names it the same way.

    A rows DOCUMENT is written per rows-bearing node the region holds, and
    where there are several each says whose rows it holds with ``-rows
    <index>``, counting the same ``-m`` flags (:func:`rows_args`).

    ``-annotations`` says which sides carry the rows a module read off its
    frames -- ``in`` beside the input it reads them from, ``out`` beside the
    output it writes them to. A process with ffmpeg on both sides gets
    neither: ffmpeg has no annotation stream to hand over, and rows between
    two modules of one network never reach a side at all.

    ``-nn`` binds one model file per module that runs one, ahead of the
    module table: the host loads them before anything is instantiated.
    `runtime` is what it takes to load them -- the fetched runtime directory
    and the target to run on -- and is empty for a printed command, whose
    reader has their own machine's. ``-nn-exclude`` names a provider a bound
    model's own pin denies, ahead of the ``-nn`` table the same way.

    ``-http`` and ``-udp`` grant one module its effects, also ahead of the
    module table: the sidecar denies both to any module the argv never
    names.

    `reads` is one path per stream the process is handed, each written as its
    own ``-f nut -i <path>``: stdin for the ordinary one-input process, and a
    named pipe apiece where a SINK reads several. Empty means stdin, which is
    what a caller with no plan in hand (a describe, a test) gets. A packet
    SINK pad carrying rendition metadata (:attr:`SidecarProcess.pads`) gets a
    ``-pad '<json>'`` right after its own ``-i``, ``{"row": ..., "rendition":
    {...}}`` with absent attributes omitted -- a pad with none gets no flag.

    `writes` is the mirror on the other side: one path per rows document the
    process writes, in document order, since a process writing several of
    them has only one stdout to put the first on. A packet SOURCE leads
    `writes` with one path per track, ahead of any rows document, and its
    `reads` are empty, since it takes no ``-i`` at all.

    A packet SOURCE writes one ``-f nut <pipe>`` per track the plan takes
    from it, in catalog order, each preceded by ``-track <index>`` saying
    which track of the module's catalog it carries. A source with no output
    is refused: it would have nothing to write.

    ``-jobs`` caps the sidecar's worker threads, and is written on every
    sidecar process whenever the run gave one -- the sidecar itself decides
    what each module's lane admits. Absent, the sidecar sizes its pool to
    the machine, so a run at the default carries no ``-jobs`` at all.
    """
    if process.packet_source and not process.outputs:
        raise _reject(
            f"the module '{process.module}' produces no track to write",
            hint="a source with nothing to write is not one; give it at "
            "least one track, or drop it from the query",
        )
    if process.packet_source and len(process.tracks) != len(process.outputs):
        raise FfrwdError(
            ErrorCode.INTERNAL,
            f"process {process.id!r} writes {len(process.outputs)} track(s) and "
            f"names {len(process.tracks)} of its catalog",
            hint="a packet source's `tracks` says which catalog track each of "
            "its outputs carries, one per output",
        )
    argv = [binary]
    if not process.packet_source:
        for index, path in enumerate(reads or (STDIN,)):
            argv += ["-f", EDGE_FORMAT, "-i", path]
            meta: PadMeta | None = process.pads[index] if index < len(process.pads) else None
            if meta is not None:
                argv += ["-pad", json.dumps(meta.to_dict())]
    if jobs is not None:
        argv += [_JOBS_FLAG, str(jobs)]
    if process.reads_rows:
        argv += [_ANNOTATIONS_FLAG, ANNOTATIONS_IN]
    argv += _nn_args(process.models, runtime)
    for grant in process.grants:
        argv += [_GRANT_FLAGS[grant.effect], grant.module]
    if process.network:
        argv += _network_args(process, writes)
    else:
        argv += ["-m", process.module]
        if process.args:
            argv += ["-params", json.dumps(process.args, sort_keys=True)]
        argv += _rows_in_args(process)
        argv += _rows_module_args(process)
        tracks, documents = _split_writes(process, writes)
        argv += rows_args(process, documents) or _stream_output(process, tracks)
    if process.writes_rows:
        argv += [_ANNOTATIONS_FLAG, ANNOTATIONS_OUT]
    return argv


def _rows_in_args(process: SidecarProcess) -> list[str]:
    """One ``-rows-in <arg>=<path>`` per rows document this filter reads.

    In declaration order, and named: the sidecar writes the argument's name
    onto every row it delivers from that input, which is how a filter
    reading several rows arguments tells them apart.
    """
    argv: list[str] = []
    for read in process.rows_in:
        argv += [_ROWS_IN_FLAG, f"{read.arg}={read.path}"]
    return argv


def _rows_module_args(process: SidecarProcess) -> list[str]:
    """The ``-m``/``-rows-from`` pair each rows module of this region takes.

    A rows module is in no filtergraph -- it reads rows and writes rows, and
    nothing about it is a pad -- so the edge is a flag instead: ``-rows-from``
    counts ``-m`` flags from the start of this command line, the stream
    modules first, and names the one whose rows arrive here.
    """
    argv: list[str] = []
    for module in process.rows_modules:
        argv += ["-m", module.path, _ROWS_FROM_FLAG, str(module.source)]
    return argv


def _split_writes(
    process: SidecarProcess, writes: Sequence[str]
) -> tuple[Sequence[str], Sequence[str]]:
    """`writes` as a packet source's or a packet filter's stream paths and the
    rows paths after them, which :func:`~ffrwd.execute._sidecar_writes` puts in
    that order. Every other process writes rows documents and nothing else."""
    if not (process.packet_source or process.packet_filter):
        return (), writes
    count = len(process.outputs)
    return writes[:count], writes[count:]


def _stream_output(process: SidecarProcess, writes: Sequence[str] = ()) -> list[str]:
    """The argv tail one mapped stream output takes.

    NUT to stdout for a region whose frames feed the next process; a null
    output for a SINK region, whose module consumes its frames and whose
    effects are the product -- nothing rides its stdout. A packet FILTER
    writes one ``-f nut`` per pad it reads, in pad order, `writes` naming
    each: it hands every stream it was given back, still encoded. A packet
    SOURCE
    writes one ``-track <index> -f nut <pipe>`` per track the plan takes
    instead of the single stdout every other region gets, `writes` naming
    each in catalog order -- a printed command with none given still numbers
    them the way a single output's own ``pipe:1`` already does. The selector
    is what lets a source write fewer tracks than its catalog holds: the
    sidecar subscribes to the named ones and to nothing else.
    """
    if process.sink:
        return ["-f", _NULL_FORMAT, "-"]
    if process.packet_filter:
        # One output per pad, in pad order: the filter hands every stream it
        # was given back, and each travels its own pipe the way a source's
        # tracks do. The order is the order the reader opens them in.
        paths = writes or tuple(f"pipe:{i + 1}" for i in range(len(process.outputs)))
        return [
            token for path in paths for token in ("-f", EDGE_FORMAT, path)
        ]
    if process.packet_source:
        paths = writes or tuple(f"pipe:{i + 1}" for i in range(len(process.outputs)))
        argv: list[str] = []
        for track, path in zip(process.tracks, paths):
            argv += [_TRACK_FLAG, str(track), "-f", EDGE_FORMAT, path]
        return argv
    return ["-f", EDGE_FORMAT, STDOUT]


def rows_args(process: SidecarProcess, writes: Sequence[str] = ()) -> list[str]:
    """The argv tail that writes this region's rows, or ``[]`` for one with none.

    A region whose rows the query selects writes them as a DOCUMENT rather
    than as frames: cue timing does not survive the NUT edge, so the sidecar
    gathers the rows and writes a finished subtitle file. Its stream output
    is what the rows were read off, and nothing downstream maps it.

    ``-f <format> <path>`` per document, the spelling an ``-f nut`` output
    already takes, with the network form's ``-map`` supplied by the caller.
    A document goes to the named file when the query wrote the rows
    themselves, and otherwise to what `writes` names for it -- the region's
    own stdout for the one document a process ordinarily writes, a named
    pipe apiece where it writes several. A printed command with none given
    numbers them the way a single output's own ``pipe:1`` already does.

    Several documents each carry ``-rows <index>``, saying which module's
    rows this one holds by its position among the line's ``-m`` flags: a
    module's own rows, or a rows module's output. ONE document needs no
    such flag -- it is the only rows the line writes -- which is the
    spelling a region with one has always taken.
    """
    return [token for tail in _rows_tails(process, writes) for token in tail]


def _rows_tails(process: SidecarProcess, writes: Sequence[str] = ()) -> list[list[str]]:
    """:func:`rows_args`, one tail per document rather than one flat list."""
    tails: list[list[str]] = []
    several = len(process.rows) > 1
    for index, document in enumerate(process.rows):
        given = writes[index] if index < len(writes) else ""
        tail = (
            [_ROWS_FLAG, str(document.source)]
            if several and document.source is not None
            else []
        )
        path = document.sink.path or given or f"pipe:{index + 1}"
        tails.append([*tail, "-f", document.sink.container, path])
    return tails


def _network_args(process: SidecarProcess, writes: Sequence[str] = ()) -> list[str]:
    """The ``-m`` table, the network string and the maps of a module region.

    One output per sink, in the region's own sink order: its stream outputs
    first, then a rows document apiece. A rows document names a path of its
    own, so several of them are spellable; a stream leaves over stdout, and
    only one of those is.
    """
    graph = process.graph
    if graph is None:  # `network` is False without one
        raise _reject(
            f"process '{process.id}' hosts several modules and carries no graph",
            hint="the plan was built without partitioning; recompile the query",
        )
    streams = len(graph.sinks) - len(process.rows)
    if len(graph.input_paths) != 1 or streams > 1:
        raise _reject(
            f"process '{process.id}' reads {len(graph.input_paths)} streams and "
            f"writes {streams}, and only its own stdin and stdout are wired",
            hint="a module network on more than one stream in or out needs argv "
            "that can spell a named pipe path",
        )
    network, targets = build_network_graph(graph, pipe_inputs=[STDIN])
    rows = _rows_tails(process, writes)
    # The region's stream sinks come first, its rows documents after them.
    tails = [_stream_output(process)] * (len(targets) - len(rows)) + rows
    argv: list[str] = []
    for binding in process.modules:
        argv += ["-m", f"{binding.name}={binding.path}"]
    # After the table the network names, so a rows module's ``-rows-from``
    # index counts the same ``-m`` flags in both spellings.
    argv += _rows_module_args(process)
    argv += ["-filter_complex", network]
    for target, tail in zip(targets, tails):
        argv += ["-map", target, *tail]
    return argv


def sidecar_argv(
    process: SidecarProcess,
    reads: Sequence[str] = (),
    writes: Sequence[str] = (),
    jobs: int | None = None,
) -> list[str]:
    """The argv that RUNS one sidecar process, with the binary located.

    A wheel installs the sidecar into the environment's scripts directory,
    which need not be on PATH, so what is spawned is the located path rather
    than the program name.

    A process binding a model is also told where the fetched ONNX Runtime is
    and what to run it on, which a printed command has no business spelling:
    the directory is under THIS machine's cache.

    `jobs` is the worker-thread cap the run asked for, passed to every
    sidecar; None is the default, the sidecar sizing itself to the machine.

    Raises ``FfrwdError`` when the sidecar is not installed -- the same
    rejection, and the same hint, a describe gives.
    """
    binary = binaries.ffrwd_wasm_path()
    if binary is None:
        raise _reject(
            f"the ffrwd-wasm sidecar is not installed, and '{process.module}' "
            "needs it to run",
            hint=INSTALL_HINT,
        )
    return _argv(
        binary, process, reads, writes, nn.spawn_args() if process.models else (), jobs
    )


def shown_argv(
    process: SidecarProcess,
    reads: Sequence[str] = (),
    writes: Sequence[str] = (),
    jobs: int | None = None,
) -> list[str]:
    """The argv a PRINTED command line shows, naming the sidecar by program name.

    The same convention a printed ffmpeg command follows: what is shown is
    what a reader would type, resolved by PATH, not the absolute path this
    machine happens to have found -- and so with no runtime directory either.
    """
    return _argv(binaries.SIDECAR_EXECUTABLE, process, reads, writes, (), jobs)


def model_path(module: str, export: str) -> str:
    """The model file a module's export loads: ``<export>.onnx`` beside it.

    The module's own directory, spelled the way the declaration spelled it --
    a path already written with backslashes keeps them.
    """
    cut = max(module.rfind("/"), module.rfind("\\"))
    return f"{module[: cut + 1]}{export}{MODEL_SUFFIX}"


def model_binding(
    described: Described, module: str, *, not_on: Sequence[str] = ()
) -> ModelBinding:
    """The ``-nn`` entry `module` needs, or a rejection naming the missing file.

    The path is written into the argv at compile time, so the file has to be
    there now: a run that would fail to load it fails here instead, where the
    declaration can be pointed at. `not_on` is the providers the model's own
    manifest pin denies, when the caller has one in hand -- empty for a
    module bound outside any package.
    """
    path = model_path(module, described.name)
    if not Path(path).is_file():
        raise _reject(
            f"the module '{module}' runs a model, and '{path}' is not there",
            hint=f"a module that runs one expects '{described.name}{MODEL_SUFFIX}' "
            "beside its wasm file; `ffrwd install` in the package fetches "
            "what its manifest pins",
        )
    return ModelBinding(name=described.name, path=path, not_on=tuple(not_on))


def rows_fields(described: Described) -> tuple[tuple[str, str], ...] | None:
    """Each column the module's rows carry, as ``(name, JSON Schema type)``.

    None when the module declares no rows at all, which is a different answer
    from an empty tuple: a module emitting rows with no properties is a
    schema this compiler has nothing to match a declaration against. A module
    publishing SEVERAL row shapes answers with its first; :func:`rows_arms`
    is what reads all of them.
    """
    arms = rows_arms(described)
    return None if arms is None else arms[0]


def rows_arms(described: Described) -> tuple[tuple[tuple[str, str], ...], ...] | None:
    """Every shape the module's rows may take, each as :func:`rows_fields` does.

    One arm for a plain schema. A module whose rows take one of several
    shapes publishes them as ``oneOf``, and each branch is an arm of its own:
    a declaration matching ANY of them is a declaration the module can fill.
    None when the module declares no rows at all.
    """
    return _arms(described.rows_schema)


def input_rows_arms(
    described: Described,
) -> tuple[tuple[tuple[str, str], ...], ...] | None:
    """Every shape the rows a ROWS MODULE reads may take, as :func:`rows_arms`.

    The mirror of `rows_schema` for the other end of a rows module: what it
    consumes rather than what it emits. None when the module names no shape
    in particular.
    """
    return _arms(described.input_rows_schema)


def _arms(
    schema: Mapping[str, object] | None,
) -> tuple[tuple[tuple[str, str], ...], ...] | None:
    """One row schema read as its arms: `oneOf`'s branches, else itself."""
    if schema is None:
        return None
    branches = schema.get("oneOf")
    if isinstance(branches, list) and branches:
        arms = [_arm(branch) for branch in branches if isinstance(branch, dict)]
        if arms:
            return tuple(arms)
    return (_arm(schema),)


def _arm(schema: Mapping[str, object]) -> tuple[tuple[str, str], ...]:
    """One row schema's columns, name-ordered."""
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return ()
    found: list[tuple[str, str]] = []
    for name, member in properties.items():
        kind = member.get("type") if isinstance(member, dict) else None
        found.append((str(name), kind if isinstance(kind, str) else ""))
    return tuple(sorted(found))


def language_tag(code: str) -> str | None:
    """The container tag for a language `code`, or None for one not known.

    A written 639-1 code maps to the 639-2/B tag a container records; a code
    already spelled as that tag is itself. `zxx`, `und` and `mul` pass for the
    tracks that are not one language, and so does the `qaa`-`qtz` range the
    standard reserves for local use.
    """
    written = code.strip().lower()
    if written in _LANGUAGE_TAG_VALUES or _local_use_tag(written):
        return written
    return LANGUAGE_TAGS.get(written)


def wire_pix_fmt(described: Described) -> str:
    """The pixel format the edges around this module carry.

    The module's own formats, in ITS order -- a module lists what it prefers
    first -- narrowed to what a stream edge can carry. No overlap is a
    rejection naming both lists.
    """
    for candidate in described.pixel_formats:
        if candidate in WIRE_PIX_FMTS:
            return candidate
    raise _reject(
        f"the module '{described.name}' accepts "
        f"{_listed(described.pixel_formats, 'no pixel format')}, and a stream "
        f"edge carries {_listed(WIRE_PIX_FMTS, 'no pixel format')}",
        hint="the module has to accept one of the formats the sidecar's frames "
        "travel in",
    )


def wire_sample_fmt(described: Described) -> str:
    """The sample format the edges around this module carry.

    The audio mirror of :func:`wire_pix_fmt`: the module's own formats, in ITS
    order, narrowed to what a stream edge can carry. No overlap is a rejection
    naming both lists.
    """
    for candidate in described.sample_formats:
        if candidate in WIRE_SAMPLE_FMTS:
            return candidate
    raise _reject(
        f"the module '{described.name}' accepts "
        f"{_listed(described.sample_formats, 'no sample format')}, and a stream "
        f"edge carries {_listed(WIRE_SAMPLE_FMTS, 'no sample format')}",
        hint="the module has to accept one of the formats the sidecar's samples "
        "travel in",
    )


def wire_audio(described: Described) -> AudioFormat:
    """What the edges around this audio module carry.

    The negotiated sample format as the pcm codec it travels as, plus the rate
    and channel count the producing ffmpeg is told to conform to -- the first
    of each the module accepts. A module that names neither constrains
    neither, and the stream reaches it as it is.
    """
    return AudioFormat(
        codec=SAMPLE_FMT_CODECS[wire_sample_fmt(described)],
        required_rate=described.sample_rates[0] if described.sample_rates else None,
        required_channels=(
            described.channel_counts[0] if described.channel_counts else None
        ),
    )


def encoder_codec(encoder: str) -> str | None:
    """The codec `encoder` writes, or None for one this compiler cannot name.

    A hardware encoder spells its codec itself (``h264_nvenc``); the software
    encoders are table entries. A bare codec name is no encoder, so it maps
    to nothing.
    """
    head, sep, _ = encoder.partition("_")
    if sep and head in WIRE_VIDEO_CODECS:
        return head
    return _ENCODER_CODECS.get(encoder)


def audio_encoder_codec(encoder: str) -> str | None:
    """The audio codec `encoder` writes, or None for one this cannot name.

    The audio half of :func:`encoder_codec`. A hardware encoder spells its
    codec itself (``aac_mf``); the rest are table entries.
    """
    head, sep, _ = encoder.partition("_")
    if sep and head in WIRE_AUDIO_CODECS:
        return head
    return _AUDIO_ENCODER_CODECS.get(encoder)


def hosts_packet_filter(world: str) -> bool:
    """True when `world`'s sidecar can host a packet filter."""
    return world in WORLDS and WORLDS.index(world) >= WORLDS.index(PACKET_FILTER_WORLD)


def hosts_packet_sink(world: str) -> bool:
    """True when `world`'s sidecar can host a packet sink."""
    return world in WORLDS and WORLDS.index(world) >= WORLDS.index(_PACKET_SINK_WORLD)


def hosts_packet_source(world: str) -> bool:
    """True when `world`'s sidecar can host a packet source."""
    return world in WORLDS and WORLDS.index(world) >= WORLDS.index(PACKET_SOURCE_WORLD)


def hosts_rows_module(world: str) -> bool:
    """True when `world`'s sidecar can host a rows module."""
    return world in WORLDS and WORLDS.index(world) >= WORLDS.index(_ROWS_MODULE_WORLD)


def _listed(names: Sequence[str], empty: str) -> str:
    """A format list as a message says it."""
    if not names:
        return empty
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " or " + names[-1]
