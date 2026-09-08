"""What a COPY's destination writes.

A destination's options, the codecs its streams travel in, and -- for a sink
a wasm module implements -- the SELECT list's streams matched to the pads
that module reads.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from sqlglot import exp

from ffrwd.errors import ErrorCode, FfrwdError
from ffrwd.expressions import _error
from ffrwd.functions import WASM_STREAM_NAMES, WasmFunction
from ffrwd.ir import StreamType
from ffrwd.parser import RawSink, _pos
from ffrwd.types import is_array
from ffrwd.values import _Stream, _stream_count
from ffrwd.wasm import (
    AUDIO_CODEC_ENCODERS,
    WIRE_AUDIO_CODECS,
    Described,
    audio_encoder_codec,
)

_PER_TRACK_OPTION_HINT = (
    "a per-row option binds one TRACK per row: gather the rows into this "
    "destination with array_agg(...) so it has a track for each; or give each "
    "row a file of its own with a TO expression, e.g. TO (:'names'[i.i] || "
    "'.mp4')"
)


def _each(value: object) -> list[object]:
    """One option's values: every element of a per-pad list, or the one value.

    Empty for an option that is not written at all, so a caller loops over
    nothing rather than checking first.
    """
    if value is None:
        return []
    return list(value) if isinstance(value, list) else [value]


@dataclass(frozen=True)
class _VariantRow:
    """One row of a manifest destination: the streams its cells hold.

    A NULL cell is None -- that kind absent from this variant. A video-only
    row is a variant drawing from the audio group, an audio-only row a
    rendition in it, a both-cells row a muxed variant.

    `name`/`language` are what this row's cells carry from an UNMODIFIED read
    of a rendition row (:attr:`_Stream.rendition`), read off in
    :meth:`_Lowerer._row_cells`: `name` from the video cell, or the audio
    cell on an audio-only row; `language` from the audio cell, only where its
    own stream tags none. Both None where the row carries no such cell, which
    is where :meth:`_Lowerer._variant_names` falls back to a computed name.
    """

    video: _Stream | None
    audio: _Stream | None
    name: str | None = None
    language: str | None = None


def _join_codecs(names: Sequence[str]) -> str:
    """A codec list as a message says it."""
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " or " + names[-1]

def _packet_sink_audio_codec(
    declared: WasmFunction,
    described: Described,
    options: dict[str, object],
    option_nodes: dict[str, exp.Expr],
    raw: RawSink,
) -> None:
    """`audio_codec` settled the way `video_codec` is, against the audio
    the stream edge carries and the codecs the module accepts."""
    accepted = described.sink_codecs("audio")
    for written in _each(options.get("audio_codec")):
        assert isinstance(written, str)  # validated as a str above
        codec = audio_encoder_codec(written)
        line, col = _pos(option_nodes["audio_codec"], raw.path_node)
        if codec is None:
            raise FfrwdError(
                ErrorCode.UNSUPPORTED_SQL,
                f"the audio stream into a packet sink travels as "
                f"{_join_codecs(WIRE_AUDIO_CODECS)}, and '{written}' "
                "encodes none of them",
                line=line,
                col=col,
                hint="name an encoder for one of them, e.g. "
                + ", ".join(AUDIO_CODEC_ENCODERS.values()),
            )
        if accepted and codec not in accepted:
            raise FfrwdError(
                ErrorCode.UNSUPPORTED_SQL,
                f"'{written}' writes {codec}, and the module "
                f"'{declared.module}' consumes {_join_codecs(accepted)} "
                "audio",
                line=line,
                col=col,
                hint=f"name an encoder for {_join_codecs(accepted)}, or "
                "drop audio_codec to take the module's preference",
            )
    if "audio_codec" not in options:
        codec = next(
            (c for c in accepted if c in WIRE_AUDIO_CODECS),
            WIRE_AUDIO_CODECS[0] if not accepted else None,
        )
        if codec is None:
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"the module '{declared.module}' consumes "
                f"{_join_codecs(accepted)} audio, and the stream edge "
                f"carries {_join_codecs(WIRE_AUDIO_CODECS)}",
                raw.path_node,
                hint="the module has to accept one of the codecs the "
                "sidecar's packets travel in",
            )
        options["audio_codec"] = AUDIO_CODEC_ENCODERS[codec]


def _default_audio_row(variant_rows: list[_VariantRow]) -> int | None:
    """Which rendition row gets ``default:yes``.

    The probed disposition wins when a track carries one; otherwise the
    first audio-only row. A muxed row is a variant, not a rendition, and
    never takes the flag.
    """
    renditions = [
        position
        for position, row in enumerate(variant_rows)
        if row.audio is not None and row.video is None
    ]
    for position in renditions:
        audio = variant_rows[position].audio
        source = audio.source if audio is not None else None
        if source is not None and source.disposition.get("default"):
            return position
    return renditions[0] if renditions else None


def _disable_scene_cuts(options: dict[str, object]) -> None:
    """Scene cuts off, in the encoder's own spelling.

    libx264 reads ``-sc_threshold 0``; libx265 and libsvtav1 read a
    private param, carried on the codec_params road the table already
    renders. Any other encoder is left alone: a knob it does not have
    would be a silent no-op.
    """
    codec = options.get("video_codec")
    if codec == "libx264":
        options["sc_threshold"] = 0
        return
    param = {"libx265": "scenecut=0", "libsvtav1": "scd=0"}.get(
        codec if isinstance(codec, str) else ""
    )
    if param is None:
        return
    key = param.partition("=")[0]
    written = options.get("codec_params")
    if written is None:
        options["codec_params"] = param
    elif isinstance(written, str):
        if key not in written:
            options["codec_params"] = f"{written}:{param}"
    elif isinstance(written, list):
        options["codec_params"] = [
            element
            if not isinstance(element, str) or key in element
            else f"{element}:{param}"
            for element in written
        ]


def _bind_sink_streams(
    declared: WasmFunction,
    gathered: list[tuple[_Stream, exp.Expr]],
    node: exp.Expr,
    select: exp.Select,
) -> list[_Stream]:
    """The SELECT's streams against the sink's parameters, matched by kind.

    Each parameter takes streams of its own kind in SELECT order: a bare
    one exactly one, an ARRAY one every remaining stream of that kind. The
    pads come back in DECLARATION order, which is the order `init` names
    them, so a module reading video and audio knows which is which without
    being told.
    """
    remaining: dict[StreamType, list[tuple[_Stream, exp.Expr]]] = {}
    for entry in gathered:
        remaining.setdefault(entry[0].type, []).append(entry)
    pads: list[_Stream] = []
    for param, kind in zip(
        declared.stream_params, declared.stream_kinds, strict=True
    ):
        waiting = remaining.get(kind, [])
        wanted = len(waiting) if is_array(param.type) else 1
        if len(waiting) < max(wanted, 1):
            raise _error(
                ErrorCode.UDF_ARG_TYPE,
                f"{declared.name}() takes '{param.name}' as {param.type}, "
                f"and this query's SELECT list carries "
                f"{_stream_count(len(waiting))} of that kind",
                node,
                fallback=select,
                hint=f"a sink reads the streams its SELECT list names: "
                f"COPY (SELECT <{kind} stream>, ...) TO {declared.called}"
                f"(<values>)",
            )
        pads += [stream for stream, _ in waiting[:wanted]]
        remaining[kind] = waiting[wanted:]
    left = [entry for entries in remaining.values() for entry in entries]
    if left:
        stream, anchor = left[0]
        raise _error(
            ErrorCode.UDF_ARG_TYPE,
            f"{declared.name}() reads {_stream_count(len(pads))}, and this "
            f"query's SELECT list carries "
            f"{_stream_count(len(pads) + len(left))}",
            anchor,
            fallback=node,
            hint=f"declare the parameter as {WASM_STREAM_NAMES[stream.type]}[] "
            "to read every stream of its kind the SELECT carries, or drop "
            "the extra columns",
        )
    return pads
