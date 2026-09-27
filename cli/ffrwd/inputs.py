"""Input option table for ffrwd.

Guardrail #4: ``INPUT_OPTIONS`` is DATA, the single source of truth driving
``input('path', <name> => <value>, ...)`` validation, emit, docs and the LLM
prompt. Input-side mirror of ``ffrwd.sink.SINK_OPTIONS``, deliberately the
same shape.

No ``extra_args`` escape hatch: arbitrary flag passthrough would break
"reject, never approximate"; the table grows instead.

``InputOptionSpec`` has no ``scope``/``per_stream`` (unlike ``SinkOptionSpec``)
-- a demuxer-level flag applies once to the input's ``-i``, with no per-stream
axis. ``"num"`` (int or float, never bool) covers ``framerate``/``itsoffset``/
``seek_end``, which are routinely fractional and, for ``itsoffset``, legally
negative (ffmpeg shifts a stream earlier). ``seek_end`` renders NEGATED
(``-sseof -<v>``, value written as seconds from the end); ``realtime`` renders
as a bare ``-re`` flag, no value.

Every spec also says whether it ``probes``: ffmpeg and ffprobe share a
demuxer but not a purpose, and several options here (``realtime``,
``stream_loop``, ``hwaccel``, ``itsoffset``, ``seek_end``) are ffprobe-only
rejections on the build this was measured against. :func:`probe_options`
reads that field to narrow what a probe sends, so the decision lives beside
each option rather than as a second list that could drift from this one.

An option can also be the COMPILER's alone (``compile_time``): it changes what
the compile knows about the input and reaches neither ffprobe nor ffmpeg.
``shape`` is one. It declares the input's streams, which stand in for a probe
(:func:`declared_probe`), so a listener that a probe would consume, or a feed
that would not answer in time, compiles with its shape known. Its value is a
``STRUCT``, checked key by key against :data:`SHAPE_KEYS`.
"""

from __future__ import annotations

import difflib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from typing import Literal

from ffrwd.errors import ErrorCode, FfrwdError
from ffrwd.ir import StreamType
from ffrwd.probe import ProbeResult, StreamMeta

InputOptionType = Literal["str", "int", "bool", "num", "size", "struct"]


@dataclass(frozen=True)
class InputOptionSpec:
    name: str
    type: InputOptionType
    doc: str  # one line; drives docs + prompt
    flag: str  # e.g. "-loop", "-framerate"
    # Whether this option belongs on the ffprobe invocation too, not just
    # ffmpeg's. ffmpeg and ffprobe share the demuxer, so an option that
    # shapes what the demuxer reads -- a forced format, a frame rate, a raw
    # input's geometry -- changes what a probe reports and must reach it, or
    # the probed stream list would describe a different read than ffmpeg is
    # about to do. An option that only affects decode, playback pacing, or a
    # timestamp presented downstream leaves nothing for ffprobe to read
    # differently, and several of those are options ffprobe rejects outright
    # on the build this table was measured against.
    probes: bool
    # True -> a boolean flag with no value (e.g. "-re"); rendered flag-only
    # when the value is True, omitted entirely when False.
    bare: bool = False
    # True -> the compiler's alone: read at compile time, never rendered to
    # ffprobe or ffmpeg, and kept out of the graph's rendered input options
    # (`rendered_options`). `flag` is then empty.
    compile_time: bool = False


INPUT_OPTIONS: dict[str, InputOptionSpec] = {
    "loop": InputOptionSpec(
        name="loop",
        type="bool",
        doc="Loop a single-frame input (e.g. a still image) indefinitely.",
        flag="-loop",
        # Turns a single frame into an endless stream -- changes the
        # demuxer's reported duration and frame count, so ffprobe needs it
        # to describe the same read ffmpeg is about to do.
        probes=True,
    ),
    "stream_loop": InputOptionSpec(
        name="stream_loop",
        type="int",
        doc="Loop the whole input this many extra times (-1 loops forever).",
        flag="-stream_loop",
        # Repeats the file at the OUTPUT side of the demuxer, after ffprobe
        # would already have reported the stream list; ffprobe rejects it
        # outright.
        probes=False,
    ),
    "framerate": InputOptionSpec(
        name="framerate",
        type="num",
        doc="Force the input's frame rate, e.g. for a looped still image.",
        flag="-framerate",
        # Changes the demuxer's own reported fps.
        probes=True,
    ),
    "itsoffset": InputOptionSpec(
        name="itsoffset",
        type="num",
        doc="Shift the input's timestamps by this many seconds (negative shifts earlier).",
        flag="-itsoffset",
        # A presentation-time shift applied downstream of the demuxer --
        # the file's own metadata is unchanged; ffprobe rejects it outright.
        probes=False,
    ),
    "hwaccel": InputOptionSpec(
        name="hwaccel",
        type="str",
        doc="Request a hardware decoder for this input, e.g. 'cuda'.",
        flag="-hwaccel",
        # Names a decoder, not a demuxer; ffprobe rejects it outright.
        probes=False,
    ),
    "seek_end": InputOptionSpec(
        name="seek_end",
        type="num",
        doc="Seek this many seconds before the end of the file (rendered negated).",
        flag="-sseof",
        # ffprobe rejects it outright, so it cannot shape a probe even
        # though seeking does move where the demuxer starts reading.
        probes=False,
    ),
    "format": InputOptionSpec(
        name="format",
        type="str",
        doc="Force the demuxer, e.g. for a capture device, rawvideo, or image2.",
        flag="-f",
        # Names the demuxer itself; without it neither tool can even open
        # a device, a lavfi graph, or a raw/image2 spec.
        probes=True,
    ),
    "realtime": InputOptionSpec(
        name="realtime",
        type="bool",
        doc="Read the input at its native frame rate, e.g. for a live source.",
        flag="-re",
        bare=True,
        # Paces ffmpeg's own read loop; the file's metadata does not
        # change, and ffprobe rejects the flag outright.
        probes=False,
    ),
    "sub_charenc": InputOptionSpec(
        name="sub_charenc",
        type="str",
        doc="Character encoding of a text subtitle input, e.g. 'CP1250'.",
        flag="-sub_charenc",
        # The demuxer needs it to decode the cue text at all when the file
        # is not already UTF-8.
        probes=True,
    ),
    "start_number": InputOptionSpec(
        name="start_number",
        type="int",
        doc="First index of an image2-sequence input.",
        flag="-start_number",
        # Picks which file on disk the image2 demuxer opens first; changes
        # what a probe of the sequence would even read.
        probes=True,
    ),
    "subtitle_decoder": InputOptionSpec(
        name="subtitle_decoder",
        type="str",
        doc="Force the subtitle decoder for this input, e.g. 'webvtt'.",
        flag="-c:s",
        # Names a decoder, not a demuxer; -show_streams reports from the
        # demuxer without invoking one.
        probes=False,
    ),
    "video_size": InputOptionSpec(
        name="video_size",
        type="str",
        doc="Frame size a capture device or raw input is read at, e.g. '960x540'.",
        flag="-video_size",
        # A raw/device demuxer cannot report width/height at all without
        # being told them first.
        probes=True,
    ),
    "pixel_format": InputOptionSpec(
        name="pixel_format",
        type="str",
        doc="Pixel format a capture device or raw input is read as, e.g. 'yuyv422'.",
        flag="-pixel_format",
        # Same raw/device geometry the demuxer needs to make sense of the
        # bytes at all, same as `video_size`.
        probes=True,
    ),
    "sample_rate": InputOptionSpec(
        name="sample_rate",
        type="int",
        doc="Sample rate an audio capture device is read at, e.g. 48000.",
        flag="-sample_rate",
        # A raw/device audio demuxer has no sample rate of its own to
        # report until this names one.
        probes=True,
    ),
    "channels": InputOptionSpec(
        name="channels",
        type="int",
        doc="Channel count an audio capture device is read at, e.g. 2.",
        flag="-channels",
        # Same raw/device audio geometry as `sample_rate`.
        probes=True,
    ),
    "rtbufsize": InputOptionSpec(
        name="rtbufsize",
        type="size",
        doc="Buffer size held for a live source before frames drop, e.g. '256M'.",
        flag="-rtbufsize",
        # Sizes a buffer against frame drops during a long read; the
        # stream list it would report is the same either way.
        probes=False,
    ),
    "probesize": InputOptionSpec(
        name="probesize",
        type="size",
        doc="Bytes read before the demuxer decides the stream list, e.g. 5000000 or '32M'.",
        flag="-probesize",
        # Literally how much ffprobe itself reads before it can answer;
        # omitting it risks the exact under-read this option exists to fix.
        probes=True,
    ),
    "analyzeduration": InputOptionSpec(
        name="analyzeduration",
        type="size",
        doc="Microseconds analysed before the demuxer decides, e.g. 10000000 or '10M'.",
        flag="-analyzeduration",
        # The probing counterpart of `probesize`, same reasoning.
        probes=True,
    ),
    "rtsp_transport": InputOptionSpec(
        name="rtsp_transport",
        type="str",
        doc="Transport an RTSP input negotiates, e.g. 'tcp'.",
        flag="-rtsp_transport",
        # Whether the demuxer can even connect to negotiate a stream list
        # at all, e.g. a UDP transport a firewall drops.
        probes=True,
    ),
    "user_agent": InputOptionSpec(
        name="user_agent",
        type="str",
        doc="User-Agent an HTTP input sends.",
        flag="-user_agent",
        # Whether the HTTP demuxer's request is even served, same
        # reachability concern as `rtsp_transport`.
        probes=True,
    ),
    "listen": InputOptionSpec(
        name="listen",
        type="bool",
        doc="Wait for an rtmp(s) or http(s) sender to connect instead of dialling it.",
        flag="-listen",
        # A listening input declares its `shape` and is never probed: a probe
        # would take the publisher's one connection.
        probes=False,
    ),
    "listen_timeout": InputOptionSpec(
        name="listen_timeout",
        type="num",
        doc="Whole seconds a listening rtmp(s) input waits for its publisher.",
        # ffmpeg's rtmp `timeout`, documented in seconds (`ffmpeg -h
        # protocol=rtmp`); lowering allows it on rtmp(s) alone.
        flag="-timeout",
        probes=False,
    ),
    "shape": InputOptionSpec(
        name="shape",
        type="struct",
        doc="The input's streams, declared instead of probed, e.g. "
        "STRUCT(1920 AS width, 1080 AS height, 30 AS fps, 48000 AS rate, 2 AS channels).",
        flag="",
        # The compiler's alone: a declared shape means the input is not
        # probed at all, so there is no probe for it to reach.
        probes=False,
        compile_time=True,
    ),
}


# Flags the COMPILER sets on an input it minted itself, for a name no user
# SQL can also bind to `input()`. Kept out of `INPUT_OPTIONS` so the
# user-facing surface (docs, prompt, validation) never learns of them:
# `validate_option` still rejects these names as unknown, only `option_spec`
# (which emit renders through) resolves them.
#
# `format` used to live here for `ffrwd.empty_captions()` (a `data:` URI
# carries no extension, so the demuxer has to be named -- `-f webvtt -i
# "data:..."`), but `format` is now also a user-facing option (capture
# devices, rawvideo, image2 need it too), so `INPUT_OPTIONS` alone already
# resolves it -- `option_spec` never reaches this table for it.
# `empty_captions` itself still bypasses `validate_option` entirely (its
# option dict is built directly, not parsed from SQL).
_INTERNAL_INPUT_OPTIONS: dict[str, InputOptionSpec] = {
    "fpsprobesize": InputOptionSpec(
        name="fpsprobesize",
        type="int",
        doc="Frames read to settle an input's frame rate.",
        flag="-fpsprobesize",
        # Compiler-only: what bounds the analysis of a PIPE input in FRAMES
        # rather than in bytes, which is the difference between an open that
        # returns and one that outlasts the process filling the pipe. A real
        # file has no such producer to outlast, so no query needs to say it.
        probes=False,
    ),
}


def option_spec(name: str) -> InputOptionSpec | None:
    """The spec emit renders `name` with: user-facing table first, then internal."""
    return INPUT_OPTIONS.get(name) or _INTERNAL_INPUT_OPTIONS.get(name)


def _unknown_option_hint(name: str) -> str:
    matches = difflib.get_close_matches(name, sorted(INPUT_OPTIONS), n=1, cutoff=0.6)
    if matches:
        return f"did you mean {matches[0]!r}?"
    return "known options: " + ", ".join(sorted(INPUT_OPTIONS))


def validate_option(
    name: str,
    value: object,
    *,
    line: int | None = None,
    col: int | None = None,
) -> object:
    """Validate one ``input('path', name => value)`` pair against INPUT_OPTIONS.

    Returns the normalized value. Raises ``UNKNOWN_INPUT_OPTION`` for a name
    not in the table, ``INPUT_OPTION_TYPE`` for a value whose type doesn't
    match the spec. ``str``/``int``/``bool`` mirror
    ``ffrwd.sink.validate_option``; ``"num"`` accepts any int or float,
    never a bool, and negatives are legal (``itsoffset``). ``"size"`` is an
    ffmpeg integer that also reads a suffix: a bare integer, or a string
    such as ``'32M'``.
    """
    spec = INPUT_OPTIONS.get(name)
    if spec is None:
        raise FfrwdError(
            ErrorCode.UNKNOWN_INPUT_OPTION,
            f"unknown input option {name!r}",
            line=line,
            col=col,
            hint=_unknown_option_hint(name),
        )

    if spec.type == "bool":
        if isinstance(value, bool):
            return value
        raise FfrwdError(
            ErrorCode.INPUT_OPTION_TYPE,
            f"option {name!r} expects a bool, got {value!r}",
            line=line,
            col=col,
            hint=f"{name} accepts true or false, e.g. {name} => true",
        )

    if spec.type == "int":
        if isinstance(value, bool) or not isinstance(value, int):
            raise FfrwdError(
                ErrorCode.INPUT_OPTION_TYPE,
                f"option {name!r} expects an int, got {value!r}",
                line=line,
                col=col,
                hint=f"{name} takes a bare integer literal, e.g. {name} => 2",
            )
        return value

    if spec.type == "num":
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise FfrwdError(
                ErrorCode.INPUT_OPTION_TYPE,
                f"option {name!r} expects a number, got {value!r}",
                line=line,
                col=col,
                hint=f"{name} takes a bare numeric literal, e.g. {name} => 15",
            )
        return value

    if spec.type == "struct":
        return _validate_shape(value, line=line, col=col)

    if spec.type == "size":
        if isinstance(value, str) or (
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
        ):
            return value
        raise FfrwdError(
            ErrorCode.INPUT_OPTION_TYPE,
            f"option {name!r} expects a size, got {value!r}",
            line=line,
            col=col,
            hint=f"{name} takes a whole number, or a string with a suffix: "
            f"{_example(spec)}",
        )

    # spec.type == "str"
    if isinstance(value, bool) or not isinstance(value, str):
        raise FfrwdError(
            ErrorCode.INPUT_OPTION_TYPE,
            f"option {name!r} expects a str, got {value!r}",
            line=line,
            col=col,
            hint=f"{name} takes a single-quoted string literal, {_example(spec)}",
        )
    return value


# The example a spec's doc gives, after its "e.g.".
_EXAMPLE = re.compile(r"e\.g\. ('[^']*'|\d+)(?: or ('[^']*'))?")


def _example(spec: InputOptionSpec) -> str:
    """How to write `spec`'s value, from the example its own doc gives."""
    found = _EXAMPLE.search(spec.doc)
    if found is None:
        return f"e.g. {spec.name} => '...'"
    written = [f"{spec.name} => {value}" for value in found.groups() if value]
    return "e.g. " + " or ".join(written)


def render_value(spec: InputOptionSpec, name: str, value: object) -> str | None:
    """Render one input option's value per its spec, or None to omit entirely.

    A bool value of False is always omitted (e.g. plain ``loop => false``);
    ``num``/``int`` render through ``str``, so a negative ``itsoffset``
    renders as ``-1``, not ``-1.0``; a bool True renders ``"1"`` (ffmpeg's
    own spelling for e.g. ``-loop 1``); ``str`` renders as-is. ``seek_end``
    is the one NEGATED value: it is written as seconds from the end but
    ``-sseof`` wants a negative offset.
    """
    if spec.type == "bool":
        return "1" if value is True else None
    if spec.type in ("int", "num", "size") and isinstance(value, int | float):
        return str(-value if name == "seek_end" else value)
    return str(value)


def rendered_options(options: Mapping[str, object]) -> dict[str, object]:
    """`options` without the compiler's own (``compile_time``): what ffmpeg is
    handed for the input, and what a graph carries."""
    return {name: value for name, value in options.items() if not _compile_time(name)}


def _compile_time(name: str) -> bool:
    spec = option_spec(name)
    return spec is not None and spec.compile_time


def render_options(options: Mapping[str, object]) -> list[str]:
    """Validated input options as argv, in written order.

    The ONE renderer both the probe and the decode go through, so a flag
    that reaches both renders identically for each -- the decode gets every
    option, the probe only the ones :func:`probe_options` kept. A ``bare``
    spec (``realtime`` -> ``-re``) renders the flag alone, only when True.
    Raises ``ValueError`` for a name no table resolves -- callers have
    already validated every name they wrote.
    """
    args: list[str] = []
    for name, value in options.items():
        spec = option_spec(name)
        if spec is None:
            raise ValueError(f"unknown input option {name!r}")
        if spec.compile_time:
            continue
        if spec.bare:
            if value is True:
                args.append(spec.flag)
            continue
        rendered = render_value(spec, name, value)
        if rendered is None:
            continue
        args += [spec.flag, rendered]
    return args


def probe_options(options: Mapping[str, object]) -> dict[str, object]:
    """`options` narrowed to the ones a probe should see too.

    ffmpeg and ffprobe share the demuxer, so an option that shapes what the
    demuxer reads belongs on both invocations, or a probe would describe a
    different read than the one ffmpeg is about to do. An option that only
    shapes decode, playback pacing, or a downstream timestamp has nothing
    for ffprobe to read differently -- see ``InputOptionSpec.probes`` for the
    call made per option, and why.
    """
    return {name: value for name, value in options.items() if _probes(name)}


def _probes(name: str) -> bool:
    spec = option_spec(name)
    return spec is not None and spec.probes


def forces_demuxer(options: Mapping[str, object]) -> bool:
    """Whether these options name the demuxer that reads the input.

    A forced demuxer makes the spec ITS to interpret -- a device name, a
    lavfi graph, an image pattern -- so the spec need not name a file at all.
    """
    return "format" in options


# -- a declared shape --------------------------------------------------------

ShapeKeyType = Literal["count", "rate", "str"]


@dataclass(frozen=True)
class ShapeKey:
    """One key of ``shape => STRUCT(...)``: the stream it declares and what it holds.

    A `required` key is required of the stream it belongs to, once any key of
    that stream is written: which streams exist follows from the keys.
    """

    name: str
    stream: StreamType
    type: ShapeKeyType
    required: bool
    doc: str


SHAPE_KEYS: dict[str, ShapeKey] = {
    "width": ShapeKey("width", "video", "count", True, "picture width in pixels"),
    "height": ShapeKey("height", "video", "count", True, "picture height in pixels"),
    "fps": ShapeKey(
        "fps", "video", "rate", True, "frame rate, a number or text such as '30000/1001'"
    ),
    "pix_fmt": ShapeKey("pix_fmt", "video", "str", False, "pixel format, e.g. 'yuv420p'"),
    "video_codec": ShapeKey(
        "video_codec", "video", "str", False, "codec the picture arrives in, e.g. 'h264'"
    ),
    "rate": ShapeKey("rate", "audio", "count", True, "sample rate in Hz"),
    "channels": ShapeKey("channels", "audio", "count", True, "channel count"),
    "channel_layout": ShapeKey(
        "channel_layout", "audio", "str", False, "channel layout, e.g. 'stereo'"
    ),
    "audio_codec": ShapeKey(
        "audio_codec", "audio", "str", False, "codec the sound arrives in, e.g. 'aac'"
    ),
}

_SHAPE_EXAMPLE = (
    "shape => STRUCT(1920 AS width, 1080 AS height, 30 AS fps, "
    "48000 AS rate, 2 AS channels)"
)


def _shape_error(message: str, hint: str, line: int | None, col: int | None) -> FfrwdError:
    return FfrwdError(
        ErrorCode.INPUT_OPTION_TYPE, message, line=line, col=col, hint=hint
    )


def _validate_shape(
    value: object, *, line: int | None, col: int | None
) -> dict[str, object]:
    """``shape``'s STRUCT, checked key by key: a dict of the keys written.

    Refused by name: a value that is no STRUCT, a key :data:`SHAPE_KEYS` does
    not have, a value of the wrong type, a stream written without the keys it
    requires, and a STRUCT declaring no stream at all. A ``fps`` comes back as
    the text ffprobe writes a rate in, ``'30/1'``.
    """
    if not isinstance(value, dict):
        raise _shape_error(
            f"option 'shape' expects a STRUCT of the input's streams, got {value!r}",
            f"name each value with AS, e.g. {_SHAPE_EXAMPLE}",
            line,
            col,
        )
    checked: dict[str, object] = {}
    for key, written in value.items():
        spec = SHAPE_KEYS.get(key)
        if spec is None:
            close = difflib.get_close_matches(key, sorted(SHAPE_KEYS), n=1, cutoff=0.6)
            raise _shape_error(
                f"shape has no key {key!r}",
                f"did you mean {close[0]!r}?"
                if close
                else "known keys: " + ", ".join(SHAPE_KEYS),
                line,
                col,
            )
        checked[key] = _shape_value(spec, written, line=line, col=col)
    streams = {SHAPE_KEYS[key].stream for key in checked}
    if not streams:
        raise _shape_error(
            "shape declares no stream: it names none of " + ", ".join(SHAPE_KEYS),
            f"declare a picture, a sound or both, e.g. {_SHAPE_EXAMPLE}",
            line,
            col,
        )
    for stream in ("video", "audio"):
        if stream not in streams:
            continue
        wanted = [k.name for k in SHAPE_KEYS.values() if k.stream == stream and k.required]
        missing = [name for name in wanted if name not in checked]
        if missing:
            written_keys = [key for key in checked if SHAPE_KEYS[key].stream == stream]
            what = "a picture" if stream == "video" else "a sound"
            raise _shape_error(
                f"shape declares {what} ({', '.join(written_keys)}) without its "
                f"{' and '.join(missing)}",
                f"{what} needs {', '.join(wanted)}, e.g. {_SHAPE_EXAMPLE}",
                line,
                col,
            )
    return checked


def _shape_value(
    spec: ShapeKey, value: object, *, line: int | None, col: int | None
) -> object:
    """One key's value, of its declared type."""
    if spec.type == "count":
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
        raise _shape_error(
            f"shape key {spec.name!r} expects a whole number above zero, got {value!r}",
            f"{spec.name} is the {spec.doc}, e.g. {_SHAPE_EXAMPLE}",
            line,
            col,
        )
    if spec.type == "rate":
        found = _rate_text(value)
        if found is not None:
            return found
        raise _shape_error(
            f"shape key {spec.name!r} expects a frame rate, got {value!r}",
            "write a number, 30 or 29.97, or text, '30000/1001'",
            line,
            col,
        )
    if isinstance(value, str) and value:
        return value
    raise _shape_error(
        f"shape key {spec.name!r} expects text, got {value!r}",
        f"{spec.name} is the {spec.doc}",
        line,
        col,
    )


def _rate_text(value: object) -> str | None:
    """A rate as ffprobe writes one, ``'<num>/<den>'``, or None."""
    if isinstance(value, bool):
        return None
    try:
        if isinstance(value, int):
            rate = Fraction(value)
        elif isinstance(value, float):
            rate = Fraction(str(value))
        elif isinstance(value, str):
            rate = Fraction(value.strip())
        else:
            return None
    except (ValueError, ZeroDivisionError):
        return None
    if rate <= 0:
        return None
    return f"{rate.numerator}/{rate.denominator}"


def declared_probe(shape: Mapping[str, object]) -> ProbeResult:
    """What a probe would have said of an input, from its validated ``shape``.

    One picture stream, one sound stream or both, the way ffprobe reports a
    live feed: no duration, no tags. A codec the shape does not name is None,
    as it is for a stream no probe read, so a stream that would otherwise be
    copied is decoded instead.
    """
    streams: list[StreamMeta] = []

    def text(key: str) -> str | None:
        found = shape.get(key)
        return found if isinstance(found, str) else None

    def count(key: str) -> int | None:
        found = shape.get(key)
        return found if isinstance(found, int) else None

    if any(SHAPE_KEYS[key].stream == "video" for key in shape):
        streams.append(
            StreamMeta(
                type="video",
                index=0,
                metadata={},
                width=count("width"),
                height=count("height"),
                fps=text("fps"),
                sample_rate=None,
                codec=text("video_codec"),
                pix_fmt=text("pix_fmt"),
                declared=True,
            )
        )
    if any(SHAPE_KEYS[key].stream == "audio" for key in shape):
        streams.append(
            StreamMeta(
                type="audio",
                index=0,
                metadata={},
                width=None,
                height=None,
                fps=None,
                sample_rate=count("rate"),
                codec=text("audio_codec"),
                channels=count("channels"),
                channel_layout=text("channel_layout"),
                declared=True,
            )
        )
    return ProbeResult(streams=streams)
