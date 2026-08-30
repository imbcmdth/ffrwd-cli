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
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from . import binaries, nn
from .emit import build_network_graph
from .errors import ErrorCode, FfrwdError
from .execute import STDIN, STDOUT
from .ir import StreamType
from .processes import (
    NUT,
    PCM_F32LE,
    PCM_S16LE,
    AudioFormat,
    ModelBinding,
    ModuleShape,
    SidecarProcess,
)

__all__ = [
    "ANNOTATION_TYPES",
    "ANNOTATIONS_IN",
    "ANNOTATIONS_OUT",
    "CODEC_ENCODERS",
    "LANGUAGE_TAGS",
    "MODEL_SUFFIX",
    "SAMPLE_FMT_CODECS",
    "WIRE_PIX_FMTS",
    "WIRE_SAMPLE_FMTS",
    "WIRE_VIDEO_CODECS",
    "WIT_PACKAGE",
    "WORLDS",
    "WORLD_VERSION",
    "Describe",
    "Described",
    "DescribedFunction",
    "Invoke",
    "describe",
    "encoder_codec",
    "hosts_packet_sink",
    "invoke",
    "language_tag",
    "model_binding",
    "model_path",
    "rows_arms",
    "rows_fields",
    "shown_argv",
    "sidecar_argv",
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

# The encoder each of those codecs is reached by when the COPY names none.
CODEC_ENCODERS: Mapping[str, str] = {
    "h264": "libx264",
    "hevc": "libx265",
    "av1": "libsvtav1",
}

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

# The first world whose sidecar hosts a packet sink.
_PACKET_SINK_WORLD = "ffrwd:av@0.10.0"

# The sample formats one can carry, and the pcm each of them travels as.
WIRE_SAMPLE_FMTS: tuple[str, ...] = ("f32", "s16")
SAMPLE_FMT_CODECS: Mapping[str, str] = {"f32": PCM_F32LE, "s16": PCM_S16LE}

# The JSON Schema types each declared annotation field type covers. `number`
# covers integer as well: the dialect has one numeric type, and a module
# counting pixels declares the same column a module measuring them does.
ANNOTATION_TYPES: Mapping[str, tuple[str, ...]] = {
    "boolean": ("boolean",),
    "number": ("number", "integer"),
    "text": ("string",),
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

_LANGUAGE_TAG_VALUES = frozenset(LANGUAGE_TAGS.values())

_DESCRIBE_FLAG = "--describe"
_INVOKE_FLAG = "--invoke"
_TIMEOUT_SECONDS = 20.0

# How the sidecar is told a side carries annotations beside the frames.
_ANNOTATIONS_FLAG = "-annotations"

# How a model file is bound to the name the module loads it by, and what the
# file is called: the export's own name, beside the module.
_NN_FLAG = "-nn"

# The sidecar flag that grants each effect to one module, per capability name.
_GRANT_FLAGS: Mapping[str, str] = {"http": "-http", "udp": "-net"}
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
    puts a ``-nn`` binding on the sidecar's own command line. `http` and
    `udp` are the effects the module imports, each of which puts the
    matching grant -- ``-http``, ``-net`` -- on that command line.

    `codecs` is present exactly for a PACKET SINK -- a sink module that
    consumes the encoder's own output rather than decoded frames -- and
    lists the ffmpeg codec names it accepts, most preferred first, empty for
    every codec. A packet sink fills neither format list, so its `kind` is
    None; what it consumes is said by the COPY that encodes for it.
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
    # Whether upstream rows may appear on this module's output frames; None
    # for a sidecar that predates the declaration.
    forwards_rows: bool | None = None
    window: int = 1
    stride: int = 1
    pure: bool = True
    one_to_one: bool = True
    windowed: bool = False
    inputs: int = 1
    nn: bool = False
    # Whether the module imports wasi:http / wasi:sockets, and so runs only
    # under the sidecar's matching ``-http`` / ``-net`` grant.
    http: bool = False
    udp: bool = False
    codecs: tuple[str, ...] | None = None

    @property
    def packet_sink(self) -> bool:
        """True for a module whose export consumes encoded packets."""
        return self.codecs is not None

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
            window=self.window, stride=self.stride, one_to_one=self.one_to_one
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
Invoke = Callable[[str, str, Mapping[str, object]], object]


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
        # Present only for a packet sink; its ABSENCE is what marks every
        # other module, so an absent key stays None rather than ().
        codecs=_strings(payload["codecs"])
        if isinstance(payload.get("codecs"), list)
        else None,
    )


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
    try:
        done = subprocess.run(
            [sidecar, _DESCRIBE_FLAG, path],
            capture_output=True,
            # The sidecar writes UTF-8; the locale codec would be wrong AND
            # brittle -- a byte it cannot decode kills the reader thread and
            # hands back None instead of text.
            encoding="utf-8",
            errors="replace",
            timeout=_TIMEOUT_SECONDS,
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
            f"{_TIMEOUT_SECONDS:.0f}s",
            hint="check the module is a wasm component and not something much larger",
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


def invoke(path: str, function: str, args: Mapping[str, object]) -> object:
    """Run `function` inside the module at `path` on `args`, and return its result.

    ``ffrwd-wasm --invoke <module> <function> '<args-json>'`` prints one JSON
    value on stdout and exits 0, or writes a message to stderr and exits
    nonzero. `args` is marshalled to one JSON object keyed by the module's own
    parameter names.

    Raises ``FfrwdError`` -- and nothing else -- when the sidecar is not
    installed, the module rejects the call, or the answer is not JSON. The
    rejection carries no position; the caller anchors it on the call site.
    """
    sidecar = binaries.ffrwd_wasm_path()
    if sidecar is None:
        raise _reject(
            "the ffrwd-wasm sidecar is not installed, and a LANGUAGE wasm "
            "value function needs it to run",
            hint=INSTALL_HINT,
        )
    payload = json.dumps(args, sort_keys=True)
    try:
        done = subprocess.run(
            [sidecar, _INVOKE_FLAG, path, function, payload],
            capture_output=True,
            encoding="utf-8",  # what the sidecar writes; see describe()
            errors="replace",
            timeout=_TIMEOUT_SECONDS,
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
            f"{_TIMEOUT_SECONDS:.0f}s",
            hint="check the module is a wasm component and not something much larger",
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


def _first_line(text: str) -> str:
    """The first non-blank line the sidecar wrote, for a message."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return "it wrote nothing"


def _argv(
    binary: str, process: SidecarProcess, runtime: Sequence[str] = ()
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

    ``-annotations`` says which sides carry the rows a module read off its
    frames -- ``in`` beside the input it reads them from, ``out`` beside the
    output it writes them to. A process with ffmpeg on both sides gets
    neither: ffmpeg has no annotation stream to hand over, and rows between
    two modules of one network never reach a side at all.

    ``-nn`` binds one model file per module that runs one, ahead of the
    module table: the host loads them before anything is instantiated.
    `runtime` is what it takes to load them -- the fetched runtime directory
    and the target to run on -- and is empty for a printed command, whose
    reader has their own machine's.

    ``-http`` and ``-net`` grant one module its effects, also ahead of the
    module table: the sidecar denies both to any module the argv never
    names.
    """
    argv = [binary, "-f", EDGE_FORMAT, "-i", STDIN]
    if process.reads_rows:
        argv += [_ANNOTATIONS_FLAG, ANNOTATIONS_IN]
    if process.models:
        argv += list(runtime)
    for model in process.models:
        argv += [_NN_FLAG, f"{model.name}={model.path}"]
    for grant in process.grants:
        argv += [_GRANT_FLAGS[grant.effect], grant.module]
    if process.network:
        argv += _network_args(process)
    else:
        argv += ["-m", process.module]
        if process.args:
            argv += ["-params", json.dumps(process.args, sort_keys=True)]
        argv += rows_args(process) or _stream_output(process)
    if process.writes_rows:
        argv += [_ANNOTATIONS_FLAG, ANNOTATIONS_OUT]
    return argv


def _stream_output(process: SidecarProcess) -> list[str]:
    """The argv tail one mapped stream output takes.

    NUT to stdout for a region whose frames feed the next process; a null
    output for a SINK region, whose module consumes its frames and whose
    effects are the product -- nothing rides its stdout.
    """
    if process.sink:
        return ["-f", _NULL_FORMAT, "-"]
    return ["-f", EDGE_FORMAT, STDOUT]


def rows_args(process: SidecarProcess) -> list[str]:
    """The argv tail that writes this region's rows, or ``[]`` for one with none.

    A region whose rows the query selects writes them as a DOCUMENT rather
    than as frames: cue timing does not survive the NUT edge, so the sidecar
    gathers the rows and writes a finished subtitle file. Its stream output
    is what the rows were read off, and nothing downstream maps it.

    ``-f <format> <path>``, the spelling an ``-f nut`` output already takes,
    with the network form's ``-map`` supplied by the caller. The document
    goes to the region's own stdout when an ffmpeg process reads it as a
    track, and to the named file when the query wrote the rows themselves.
    """
    rows = process.rows
    if rows is None:
        return []
    return ["-f", rows.container, rows.path or STDOUT]


def _network_args(process: SidecarProcess) -> list[str]:
    """The ``-m`` table, the network string and the maps of a module region."""
    graph = process.graph
    if graph is None:  # `network` is False without one
        raise _reject(
            f"process '{process.id}' hosts several modules and carries no graph",
            hint="the plan was built without partitioning; recompile the query",
        )
    if len(graph.input_paths) != 1 or len(graph.sinks) != 1:
        raise _reject(
            f"process '{process.id}' reads {len(graph.input_paths)} streams and "
            f"writes {len(graph.sinks)}, and only its own stdin and stdout are wired",
            hint="a module network on more than one stream in or out needs argv "
            "that can spell a named pipe path",
        )
    network, targets = build_network_graph(graph, pipe_inputs=[STDIN])
    rows = rows_args(process)
    argv: list[str] = []
    for binding in process.modules:
        argv += ["-m", f"{binding.name}={binding.path}"]
    argv += ["-filter_complex", network]
    for target in targets:
        argv += ["-map", target, *(rows or _stream_output(process))]
    return argv


def sidecar_argv(process: SidecarProcess) -> list[str]:
    """The argv that RUNS one sidecar process, with the binary located.

    A wheel installs the sidecar into the environment's scripts directory,
    which need not be on PATH, so what is spawned is the located path rather
    than the program name.

    A process binding a model is also told where the fetched ONNX Runtime is
    and what to run it on, which a printed command has no business spelling:
    the directory is under THIS machine's cache.

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
    return _argv(binary, process, nn.spawn_args() if process.models else ())


def shown_argv(process: SidecarProcess) -> list[str]:
    """The argv a PRINTED command line shows, naming the sidecar by program name.

    The same convention a printed ffmpeg command follows: what is shown is
    what a reader would type, resolved by PATH, not the absolute path this
    machine happens to have found -- and so with no runtime directory either.
    """
    return _argv(binaries.SIDECAR_EXECUTABLE, process)


def model_path(module: str, export: str) -> str:
    """The model file a module's export loads: ``<export>.onnx`` beside it.

    The module's own directory, spelled the way the declaration spelled it --
    a path already written with backslashes keeps them.
    """
    cut = max(module.rfind("/"), module.rfind("\\"))
    return f"{module[: cut + 1]}{export}{MODEL_SUFFIX}"


def model_binding(described: Described, module: str) -> ModelBinding:
    """The ``-nn`` entry `module` needs, or a rejection naming the missing file.

    The path is written into the argv at compile time, so the file has to be
    there now: a run that would fail to load it fails here instead, where the
    declaration can be pointed at.
    """
    path = model_path(module, described.name)
    if not Path(path).is_file():
        raise _reject(
            f"the module '{module}' runs a model, and '{path}' is not there",
            hint=f"a module that runs one expects '{described.name}{MODEL_SUFFIX}' "
            "beside its wasm file; run the module's own fetch script to "
            "download it",
        )
    return ModelBinding(name=described.name, path=path)


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
    schema = described.rows_schema
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
    already spelled as that tag is itself.
    """
    written = code.strip().lower()
    if written in _LANGUAGE_TAG_VALUES:
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


def hosts_packet_sink(world: str) -> bool:
    """True when `world`'s sidecar can host a packet sink."""
    return world in WORLDS and WORLDS.index(world) >= WORLDS.index(_PACKET_SINK_WORLD)


def _listed(names: Sequence[str], empty: str) -> str:
    """A format list as a message says it."""
    if not names:
        return empty
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " or " + names[-1]
