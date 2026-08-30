"""The ``ffrwd.<name>`` macro namespace.

Macros expand to a small ffmpeg filter subgraph no single filter provides.
Unlike a registry filter call, a macro's signature is OURS: fixed positional
parameters, checked without consulting the installed ffmpeg's option tables --
macros work OFFLINE, with no registry. Named arguments are rejected outright
(``lower._lower_macro_call``) unless the macro declares ``options``: a macro's
option surface is its own, closed list, not something read off a filter.

:data:`MACROS` is keyed by lowercased macro name and ffrwd/lower.py is its
only reader, resolving ``ffrwd.<name>(...)`` here and nowhere else (not the
registry). Each entry's ``expand`` builds its filter chain through the ``node``
callback it is handed, so lower.py's node-minting stays the only place
FrameRefs are created. ``expand`` never validates -- by the time it runs,
lower.py has checked arity, argument kinds and the macro's stream-type rule.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from ffrwd import loudnorm
from ffrwd.ir import StreamType

__all__ = [
    "INPUT_MACROS",
    "MACROS",
    "Expander",
    "InputMacro",
    "Macro",
    "MacroParam",
    "NodeBuilder",
    "macro_names",
]

# One node-minting call: filter name, args, input refs, output pad types -> the
# new node's ref (pad 0). Exactly `_NodeFactory.node`'s shape, so `self.ctx.node`
# passes through with no adapter.
NodeBuilder = Callable[[str, dict[str, object], list[str], list[StreamType]], str]

# One expansion: positional values, the node minter, the named options that
# were actually written ({} for a positional-only macro).
Expander = Callable[[list[object], NodeBuilder, dict[str, object]], str]


@dataclass(frozen=True)
class MacroParam:
    """One position of a macro's OWN signature: a stream pad, or a number.

    `kind` is ``"stream"`` (its pad type is fixed per macro, `stream_type`)
    or ``"num"`` (a numeric literal, sign included).
    """

    name: str
    kind: str  # "stream" | "num"
    stream_type: StreamType | None = None


@dataclass(frozen=True)
class Macro:
    """One ``ffrwd.<name>`` macro: its signature, output type, expansion.

    `kind_hints` is keyed by the kind actually passed to a stream position,
    and supplies a hint more specific than the generic stream-signature
    message. Only ``delay`` uses it: audio into a video-only macro is common
    enough to earn a hint naming the bare filter that does the job.

    `options` are named-only numeric options, all optional, in render order.
    An empty tuple (every macro but ``loudnorm2``) means the signature is
    positional only and any ``=>`` argument is rejected.
    """

    name: str
    params: tuple[MacroParam, ...]
    output: StreamType
    expand: Expander
    kind_hints: dict[str, str] = field(default_factory=dict)
    options: tuple[str, ...] = ()

    @property
    def signature(self) -> str:
        parts = [p.name for p in self.params]
        parts += [f"{name} => ..." for name in self.options]
        return f"ffrwd.{self.name}(" + ", ".join(parts) + ")"

    @property
    def stream_positions(self) -> list[int]:
        return [index for index, param in enumerate(self.params) if param.kind == "stream"]


def _blur_regions(values: list[object], node: NodeBuilder, _options: dict[str, object]) -> str:
    """crop -> gblur -> overlay, region named once; `f` feeds crop AND overlay."""
    f, x, y, w, h, sigma = values
    cropped = node("crop", {"out_w": w, "out_h": h, "x": x, "y": y}, [str(f)], ["video"])
    blurred = node("gblur", {"sigma": sigma}, [cropped], ["video"])
    return node("overlay", {"x": x, "y": y}, [str(f), blurred], ["video"])


def _speed(values: list[object], node: NodeBuilder, _options: dict[str, object]) -> str:
    f, factor = values
    return node("setpts", {"expr": f"PTS/{factor}"}, [str(f)], ["video"])


def _delay(values: list[object], node: NodeBuilder, _options: dict[str, object]) -> str:
    """The transparent-canvas macro: format to yuva420p, then pad the start."""
    f, seconds = values
    formatted = node("format", {"pix_fmts": "yuva420p"}, [str(f)], ["video"])
    return node(
        "tpad",
        {"start_duration": seconds, "stop": 1, "color": "black@0"},
        [formatted],
        ["video"],
    )


def _loudnorm2(values: list[object], node: NodeBuilder, options: dict[str, object]) -> str:
    """One node carrying only what was written; emit renders it once per phase.

    Not a subgraph at all -- the expansion this macro exists for is in TIME
    (measure, then correct), not in the filtergraph, and the whole command's
    shape follows from this node being present.
    """
    (f,) = values
    return node(loudnorm.FILTER, dict(options), [str(f)], ["audio"])


_ADELAY_HINT = (
    "ffrwd.delay() is the video (transparent-canvas) macro; delay an audio "
    "stream with the bare filter directly, in milliseconds, e.g. "
    "adelay(a.audio[1], delays => '2000')"
)

MACROS: dict[str, Macro] = {
    "blur_regions": Macro(
        name="blur_regions",
        params=(
            MacroParam("f", "stream", "video"),
            MacroParam("x", "num"),
            MacroParam("y", "num"),
            MacroParam("w", "num"),
            MacroParam("h", "num"),
            MacroParam("sigma", "num"),
        ),
        output="video",
        expand=_blur_regions,
    ),
    "speed": Macro(
        name="speed",
        params=(MacroParam("f", "stream", "video"), MacroParam("factor", "num")),
        output="video",
        expand=_speed,
    ),
    "delay": Macro(
        name="delay",
        params=(MacroParam("f", "stream", "video"), MacroParam("seconds", "num")),
        output="video",
        expand=_delay,
        kind_hints={"audio": _ADELAY_HINT},
    ),
    "loudnorm2": Macro(
        name="loudnorm2",
        params=(MacroParam("stream", "stream", "audio"),),
        output="audio",
        expand=_loudnorm2,
        options=loudnorm.OPTIONS,
    ),
}


@dataclass(frozen=True)
class InputMacro:
    """A macro that lowers to an extra ``-i``, not to a filter node.

    ffmpeg has no "empty subtitle source" filter, because a filtergraph carries
    no subtitle pads at all (the passthrough-only rule). So the stand-in
    for a missing caption track is minted as an INPUT: `format` forces the
    demuxer, `path` is a self-contained ``data:`` URI, and the result is an
    ordinary passthrough subtitle stream that takes tags like any other.

    `format` is validated against the same ``ffrwd.inputs.INPUT_OPTIONS``
    entry a user's own ``input('x', format => ...)`` would be (it is a
    legitimate user-facing option too, e.g. for capture devices), but this
    macro's own value is set directly on the minted alias's option dict,
    bypassing ``validate_option`` -- the compiler already knows it is
    well-formed, having written it itself.
    """

    name: str
    output: StreamType
    path: str
    format: str


# "WEBVTT\n\n", base64'd: a valid WebVTT file with zero cues. Measured
# 2026-08-17 against ffmpeg 7.1 -- `-f webvtt -i "data:text/vtt;base64,
# V0VCVlRUCgo="` muxes a real, taggable subtitle stream carrying no cues, and
# ffmpeg's own `data:` protocol keeps the compiled command self-contained (no
# temp file is written, nothing is shipped alongside the command).
_EMPTY_VTT_URI = "data:text/vtt;base64,V0VCVlRUCgo="

INPUT_MACROS: dict[str, InputMacro] = {
    "empty_captions": InputMacro(
        name="empty_captions",
        output="subtitle",
        path=_EMPTY_VTT_URI,
        format="webvtt",
    ),
}


def macro_names() -> list[str]:
    """Every ``ffrwd.<name>`` there is, both kinds, sorted."""
    return sorted({*MACROS, *INPUT_MACROS})
