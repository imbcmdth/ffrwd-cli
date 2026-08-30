"""Sink option table for ffrwd.

Guardrail #4: the option table is DATA, not code. ``SINK_OPTIONS`` is the
single source of truth driving ``COPY ... TO 'path' WITH (...)`` validation,
emit, docs and the LLM prompt. No option-specific logic lives anywhere else --
every sink-visible option's behavior is a ``SinkOptionSpec`` field.

No ``extra_args`` escape hatch: arbitrary flag passthrough would break
"reject, never approximate"; the table grows instead.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from typing import Literal

from ffrwd.errors import ErrorCode, FfrwdError

OptionScope = Literal["video", "audio", "subtitle", "container"]
OptionType = Literal["str", "int", "bool", "num"]  # "num" = int or float, never bool


@dataclass(frozen=True)
class SinkOptionSpec:
    name: str
    scope: OptionScope
    type: OptionType
    doc: str  # one line; drives docs + prompt
    # rendering data for emit (no logic outside the table):
    flag: str  # e.g. "-c", "-crf", "-b", "-f", "-movflags"
    per_stream: bool  # True -> rendered as f"{flag}:{i}" per output
    value_template: str = "{v}"  # e.g. "+faststart" for the bool movflags case
    # True -> a boolean flag with no value (e.g. "-shortest"); rendered flag-
    # only when the value is True, omitted entirely when False. Mutually
    # exclusive with per_stream (no bare option in the table is per-stream).
    bare: bool = False


# `codec_params`'s flag is the one derived-at-render-time exception to "flag
# is static table data": its spec.flag carries a `{codec}` placeholder that
# emit fills in from the SAME group's `video_codec` value. Verified against
# real ffmpeg (9.0.1): `-x264-params`, `-x265-params`, `-svtav1-params` all
# apply to their matching libx264/libx265/libsvtav1 encoder.
CODEC_PARAMS_FLAGS: dict[str, str] = {
    "libx264": "x264",
    "libx265": "x265",
    "libsvtav1": "svtav1",
}


# Encoders whose `-pass 1` / `-pass 2` rate control `two_pass` drives.
TWO_PASS_CODECS: frozenset[str] = frozenset({"libx264", "libx265"})

# `two_pass`'s second rendered flag. Its value is the destination path, which
# only emit knows, so it cannot be a `value_template` on the spec.
PASSLOGFILE_FLAG = "-passlogfile"


SINK_OPTIONS: dict[str, SinkOptionSpec] = {
    "video_codec": SinkOptionSpec(
        name="video_codec",
        scope="video",
        type="str",
        doc="Video codec name, e.g. 'libx264'.",
        flag="-c",
        per_stream=True,
    ),
    "audio_codec": SinkOptionSpec(
        name="audio_codec",
        scope="audio",
        type="str",
        doc="Audio codec name, e.g. 'aac'.",
        flag="-c",
        per_stream=True,
    ),
    "crf": SinkOptionSpec(
        name="crf",
        scope="video",
        type="int",
        doc="Constant rate factor (encoder-dependent quality target).",
        flag="-crf",
        per_stream=True,
    ),
    "preset": SinkOptionSpec(
        name="preset",
        scope="video",
        type="str",
        doc="Encoder speed/quality preset, e.g. 'slow'.",
        flag="-preset",
        per_stream=True,
    ),
    "pix_fmt": SinkOptionSpec(
        name="pix_fmt",
        scope="video",
        type="str",
        doc="Pixel format, e.g. 'yuv420p'.",
        flag="-pix_fmt",
        per_stream=True,
    ),
    "video_bitrate": SinkOptionSpec(
        name="video_bitrate",
        scope="video",
        type="str",
        doc="Target video bitrate, e.g. '4M'.",
        flag="-b",
        per_stream=True,
    ),
    "audio_bitrate": SinkOptionSpec(
        name="audio_bitrate",
        scope="audio",
        type="str",
        doc="Target audio bitrate, e.g. '192k'.",
        flag="-b",
        per_stream=True,
    ),
    "sample_rate": SinkOptionSpec(
        name="sample_rate",
        scope="audio",
        type="int",
        doc="Output audio sample rate in Hz, e.g. 48000.",
        flag="-ar",
        per_stream=True,
    ),
    "subtitle_codec": SinkOptionSpec(
        name="subtitle_codec",
        scope="subtitle",
        type="str",
        doc="Subtitle codec name, e.g. 'mov_text', 'webvtt', 'srt'.",
        flag="-c",
        per_stream=True,
    ),
    "frames": SinkOptionSpec(
        name="frames",
        scope="video",
        type="int",
        doc="Stop the video output after N frames.",
        flag="-frames",
        per_stream=True,
    ),
    "format": SinkOptionSpec(
        name="format",
        scope="container",
        type="str",
        doc="Container format, e.g. 'mp4' (else inferred from the path extension).",
        flag="-f",
        per_stream=False,
    ),
    "faststart": SinkOptionSpec(
        name="faststart",
        scope="container",
        type="bool",
        doc="Move the MP4 moov atom to the front of the file for progressive playback.",
        flag="-movflags",
        per_stream=False,
        value_template="+faststart",
    ),
    "duration": SinkOptionSpec(
        name="duration",
        scope="container",
        type="num",
        doc="Stop the output after this many seconds (fractional allowed).",
        flag="-t",
        per_stream=False,
    ),
    "max_size": SinkOptionSpec(
        name="max_size",
        scope="container",
        type="str",
        doc="Stop the output once the file reaches this size, e.g. '10M'.",
        flag="-fs",
        per_stream=False,
    ),
    "shortest": SinkOptionSpec(
        name="shortest",
        scope="container",
        type="bool",
        doc="Stop the output as soon as its shortest stream ends.",
        flag="-shortest",
        per_stream=False,
        bare=True,
    ),
    "maxrate": SinkOptionSpec(
        name="maxrate",
        scope="video",
        type="str",
        doc="Rate-control ceiling for a VBV-constrained encode, e.g. '2675k'.",
        flag="-maxrate",
        per_stream=True,
    ),
    "bufsize": SinkOptionSpec(
        name="bufsize",
        scope="video",
        type="str",
        doc="VBV buffer size paired with maxrate, e.g. '5350k'.",
        flag="-bufsize",
        per_stream=True,
    ),
    "gop": SinkOptionSpec(
        name="gop",
        scope="video",
        type="int",
        doc="Group-of-pictures size: the max distance between keyframes.",
        flag="-g",
        per_stream=True,
    ),
    "profile": SinkOptionSpec(
        name="profile",
        scope="video",
        type="str",
        doc="Encoder profile, e.g. 'baseline', 'main', 'high'.",
        flag="-profile",
        per_stream=True,
    ),
    "level": SinkOptionSpec(
        name="level",
        scope="video",
        type="str",
        doc="Encoder level, e.g. '3.1', '4.0'.",
        flag="-level",
        per_stream=True,
    ),
    "tune": SinkOptionSpec(
        name="tune",
        scope="video",
        type="str",
        doc="Encoder tuning, e.g. 'film', 'animation', 'zerolatency'.",
        flag="-tune",
        per_stream=True,
    ),
    "codec_params": SinkOptionSpec(
        name="codec_params",
        scope="video",
        type="str",
        doc=(
            "Encoder-private key=value:key=value passthrough. Only libx264/"
            "libx265/libsvtav1; needs a matching video_codec."
        ),
        flag="-{codec}-params",  # placeholder filled from video_codec; see CODEC_PARAMS_FLAGS
        per_stream=True,
    ),
    # The one option whose value changes the SHAPE of the compile: a two_pass
    # sink emits two chained ffmpeg commands, not one. emit renders it as
    # `-pass <n> -passlogfile <dest>`, both derived per pass, so neither the
    # value nor the second flag can come from this row alone.
    "two_pass": SinkOptionSpec(
        name="two_pass",
        scope="container",
        type="bool",
        doc=(
            "Two-pass encode: compiles to two chained ffmpeg commands, a "
            "video-only analysis pass then the real write. Needs video_bitrate "
            "and a video_codec of libx264/libx265; conflicts with crf. One "
            "COPY per script."
        ),
        flag="-pass",
        per_stream=False,
    ),
    "movflags": SinkOptionSpec(
        name="movflags",
        scope="container",
        type="str",
        doc="Raw -movflags value, e.g. '+faststart+frag_keyframe'. Conflicts with faststart.",
        flag="-movflags",
        per_stream=False,
    ),
}

# Options that were sink options and are SELECT columns now, with the spelling
# that replaced each. Named separately so the rejection can say where to go.
REPLACED_OPTIONS = {
    "metadata_from": "copy an input's global tags with a tags column, "
    "e.g. SELECT ..., f.tags AS tags",
    "strip_metadata": "drop the tags the muxer would copy with an empty tags "
    "column, e.g. SELECT ..., STRUCT() AS tags",
}


# CSV option table: a COPY ... WITH (FORMAT csv, ...) sink takes
# exactly these two. A media option in a csv COPY is rejected against THIS
# table, not SINK_OPTIONS; `header` in a media COPY is rejected against
# SINK_OPTIONS, which never held it.
CSV_OPTIONS: dict[str, SinkOptionSpec] = {
    "format": SinkOptionSpec(
        name="format",
        scope="container",
        type="str",
        doc="Must be 'csv' -- this is what makes a COPY a table sink.",
        flag="",
        per_stream=False,
    ),
    "header": SinkOptionSpec(
        name="header",
        scope="container",
        type="bool",
        doc="Emit a header row of column names (default false).",
        flag="",
        per_stream=False,
    ),
}


# Flags the COMPILER sets on a sink it minted itself -- a stream edge between
# two of its own processes -- under names no user SQL may write. Kept out of
# ``SINK_OPTIONS`` so the user-facing surface (docs, prompt, validation) never
# learns of them: ``validate_option`` still rejects these names as unknown,
# and only ``option_spec``, which emit renders through, resolves them.
_INTERNAL_SINK_OPTIONS: dict[str, SinkOptionSpec] = {
    "fifo_format": SinkOptionSpec(
        name="fifo_format",
        scope="container",
        type="str",
        doc="The muxer the fifo muxer wraps.",
        flag="-fifo_format",
        per_stream=False,
    ),
    "queue_size": SinkOptionSpec(
        name="queue_size",
        scope="container",
        type="int",
        doc="Packets the fifo muxer queues before the writer waits.",
        flag="-queue_size",
        per_stream=False,
    ),
}


def option_spec(name: str) -> SinkOptionSpec | None:
    """The spec emit renders `name` with: user-facing table first, then internal."""
    return SINK_OPTIONS.get(name) or _INTERNAL_SINK_OPTIONS.get(name)


def copy_suppressed_scopes(options: dict[str, object]) -> set[str]:
    """Stream scopes for which `options` names an explicit codec.

    A passthrough output in one of these scopes re-encodes, so nothing may
    force ``-c:<i> copy`` on it. Read straight off the table's ``flag``/
    ``scope`` fields, and shared by emit (what it renders) and lower (whether
    a windowed fan-out can keep the stream-copy chain).
    """
    return {
        spec.scope
        for name in options
        if (spec := option_spec(name)) is not None and spec.flag == "-c"
    }


def _unknown_option_hint(name: str, table: dict[str, SinkOptionSpec] = SINK_OPTIONS) -> str:
    matches = difflib.get_close_matches(name, sorted(table), n=1, cutoff=0.6)
    if matches:
        return f"did you mean {matches[0]!r}?"
    return "known options: " + ", ".join(sorted(table))


def validate_option(
    name: str,
    value: object,
    *,
    line: int | None = None,
    col: int | None = None,
) -> object:
    """Validate one COPY ... WITH (name value) pair against SINK_OPTIONS.

    Returns the normalized value. Raises ``UNKNOWN_SINK_OPTION`` for a name
    not in the table, ``SINK_OPTION_TYPE`` for a value whose type doesn't
    match the spec. Bools accept `true`/`false`; ints reject floats, strings
    and bools (a Python bool is an int subclass, so the bool case is checked
    first and `isinstance(value, bool)` guards the int case).
    """
    return _validate_against(SINK_OPTIONS, name, value, line=line, col=col)


def validate_csv_option(
    name: str,
    value: object,
    *,
    line: int | None = None,
    col: int | None = None,
) -> object:
    """Validate one COPY ... WITH (name value) pair against CSV_OPTIONS.

    A separate table from ``SINK_OPTIONS``: a media option like
    ``video_codec`` is unknown here and gets its own typed rejection.
    """
    return _validate_against(CSV_OPTIONS, name, value, line=line, col=col)


def _validate_against(
    table: dict[str, SinkOptionSpec],
    name: str,
    value: object,
    *,
    line: int | None,
    col: int | None,
) -> object:
    spec = table.get(name)
    if spec is None:
        replacement = REPLACED_OPTIONS.get(name)
        raise FfrwdError(
            ErrorCode.UNKNOWN_SINK_OPTION,
            f"unknown sink option {name!r}",
            line=line,
            col=col,
            hint=replacement or _unknown_option_hint(name, table),
        )

    if spec.type == "bool":
        if isinstance(value, bool):
            return value
        raise FfrwdError(
            ErrorCode.SINK_OPTION_TYPE,
            f"option {name!r} expects a bool, got {value!r}",
            line=line,
            col=col,
            hint=f"{name} accepts true or false",
        )

    if spec.type == "int":
        if isinstance(value, bool) or not isinstance(value, int):
            raise FfrwdError(
                ErrorCode.SINK_OPTION_TYPE,
                f"option {name!r} expects an int, got {value!r}",
                line=line,
                col=col,
                hint=f"{name} takes a bare integer literal, e.g. {name} 20",
            )
        return value

    if spec.type == "num":
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise FfrwdError(
                ErrorCode.SINK_OPTION_TYPE,
                f"option {name!r} expects a number, got {value!r}",
                line=line,
                col=col,
                hint=f"{name} takes a bare numeric literal, e.g. {name} 30",
            )
        return value

    # spec.type == "str"
    if isinstance(value, bool) or not isinstance(value, str):
        raise FfrwdError(
            ErrorCode.SINK_OPTION_TYPE,
            f"option {name!r} expects a str, got {value!r}",
            line=line,
            col=col,
            hint=f"{name} takes a single-quoted string literal, e.g. {name} 'libx264'",
        )
    return value
