"""Keep an ffmpeg-only command on the card.

A query that asks an input for ``hwaccel => 'cuda'`` and writes its picture
with an ``_nvenc`` encoder decodes on the card, and as lowered then brings
every frame down for the filters between and sends it up again for the
encoder. This pass has ffmpeg keep the frames where they are: the input
gets ``hwaccel_output_format cuda``, each filter on the way that the build
has a ``_cuda`` variant of becomes that variant, and the filters it has
none of run on the CPU as one segment, between one ``hwdownload`` and one
``hwupload_cuda``, never a pair around each filter. Every other reader of a
frame on the card -- a software encoder, a module, a second file -- gets
the one download it needs.

Only the picture of an input that reaches such an encoder is kept on the
card; an input whose picture goes nowhere near one is left as it was, and
so is the whole command when the build's ffmpeg is known and has no
``_cuda`` filters.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

from .ir import FrameRef, Graph, Node, SinkUnit, is_src, src_parts
from .registry import Registry

__all__ = ["keep_on_the_card"]

# The hwaccel names that decode onto a CUDA device.
_HWACCELS = frozenset({"cuda", "cuvid", "nvdec"})
# What the input is told so decoded frames stay on the device.
_OUTPUT_FORMAT = "cuda"
# The encoder family that takes frames on the device.
_ENCODER_SUFFIX = "_nvenc"

# The filters a `_cuda` variant stands in for, with the options the variant
# takes under the name the CPU filter spells them. A call with an option
# outside this list stays on the CPU: the variant does not take it.
_VARIANTS: Mapping[str, Mapping[str, str]] = {
    "scale": {
        "width": "w",
        "height": "h",
        "w": "w",
        "h": "h",
        "force_original_aspect_ratio": "force_original_aspect_ratio",
        "force_divisible_by": "force_divisible_by",
    },
    "overlay": {
        name: name
        for name in ("x", "y", "eof_action", "eval", "shortest", "repeatlast", "ts_sync_mode")
    },
    "pad": {name: name for name in ("width", "height", "x", "y", "color", "eval", "aspect")},
    "transpose": {name: name for name in ("dir", "passthrough")},
    "yadif": {name: name for name in ("mode", "parity", "deint")},
    "bwdif": {name: name for name in ("mode", "parity", "deint")},
}

# Filters that read frames wherever they are and hand them on there: they
# touch timing, counts and pads, never pixels.
_PASSTHROUGH = frozenset(
    {"split", "null", "copy", "setpts", "settb", "trim", "fps", "select", "setsar", "setdar"}
)


def keep_on_the_card(g: Graph, registry: Registry) -> Graph:
    """`g` with every CUDA-decoded picture that reaches an ``_nvenc`` sink
    kept on the device as far as ffmpeg's own filters allow.

    Pure: returns a new Graph, `g` left as it was, and `g` itself where
    nothing in it qualifies. Runs after the split pass, so every fan-out is
    a ``split`` node already, which rides on the card like any other.
    """
    aliases = _aliases_on_the_card(g)
    if not aliases or not _build_has_variants(registry):
        return g

    nodes: dict[str, Node] = {}
    # The refs whose frames sit on the device after the pass.
    resident: set[FrameRef] = set()
    # The refs whose frames came down from the device somewhere upstream:
    # a filter with a variant sends them up again rather than staying down.
    descended: set[FrameRef] = set()
    for alias in aliases:
        for ref in _video_refs(g, alias):
            resident.add(ref)

    def downloaded(ref: FrameRef) -> FrameRef:
        """`ref` read on the CPU: through one ``hwdownload`` per card ref."""
        node_id = f"{_slug(ref)}_hwdownload"
        if node_id not in nodes:
            nodes[node_id] = Node(
                id=node_id, filter="hwdownload", args={}, inputs=[ref], outputs=["video"]
            )
            descended.add(node_id)
        return node_id

    def uploaded(ref: FrameRef) -> FrameRef:
        """`ref` read on the card: through one ``hwupload_cuda`` per ref."""
        node_id = f"{_slug(ref)}_hwupload"
        if node_id not in nodes:
            nodes[node_id] = Node(
                id=node_id, filter="hwupload_cuda", args={}, inputs=[ref], outputs=["video"]
            )
            resident.add(node_id)
        return node_id

    for node in g.nodes.values():
        on_card = [ref in resident for ref in node.inputs]
        lineage = [ref in resident or ref in descended for ref in node.inputs]
        variant = _variant_of(node)
        if variant is not None and any(lineage):
            # On the card, every input sent up that is not there already.
            inputs = [
                ref if is_resident else uploaded(ref)
                for ref, is_resident in zip(node.inputs, on_card, strict=True)
            ]
            nodes[node.id] = replace(node, filter=variant[0], args=variant[1], inputs=inputs)
            _mark(node, resident)
        elif node.filter in _PASSTHROUGH and all(on_card):
            nodes[node.id] = replace(node, inputs=list(node.inputs), args=dict(node.args))
            _mark(node, resident)
        else:
            inputs = [
                downloaded(ref) if is_resident else ref
                for ref, is_resident in zip(node.inputs, on_card, strict=True)
            ]
            nodes[node.id] = replace(node, inputs=inputs, args=dict(node.args))
            if any(lineage):
                _mark(node, descended)

    sinks = [
        replace(
            unit,
            outputs=[
                replace(output, ref=downloaded(output.ref))
                if output.ref in resident and not _takes_the_card(unit)
                else output
                for output in unit.outputs
            ],
        )
        for unit in g.sinks
    ]
    options = {
        alias: _with_output_format(values) if alias in aliases else dict(values)
        for alias, values in g.input_options.items()
    }
    return replace(g, nodes=nodes, sinks=sinks, input_options=options)


def _aliases_on_the_card(g: Graph) -> frozenset[str]:
    """The aliases decoded onto a CUDA device whose picture reaches an
    ``_nvenc`` sink."""
    decoded = {
        alias
        for alias, options in g.input_options.items()
        if str(options.get("hwaccel", "")).lower() in _HWACCELS
    }
    if not decoded:
        return frozenset()
    wanted = {
        output.ref for unit in g.sinks if _takes_the_card(unit) for output in unit.outputs
    }
    if not wanted:
        return frozenset()
    # Everything upstream of a wanted ref, by walking producers.
    producers: dict[FrameRef, Node] = {}
    for node in g.nodes.values():
        producers[node.id] = node
        for pad in range(len(node.outputs)):
            producers[f"{node.id}:{pad}"] = node
    upstream: set[FrameRef] = set()
    stack = list(wanted)
    while stack:
        ref = stack.pop()
        if ref in upstream:
            continue
        upstream.add(ref)
        producer = producers.get(ref)
        if producer is not None:
            stack.extend(producer.inputs)
    return frozenset(
        alias
        for alias in decoded
        if any(ref in upstream for ref in _video_refs(g, alias)) or _bare(g, alias, upstream)
    )


def _bare(g: Graph, alias: str, upstream: set[FrameRef]) -> bool:
    """Whether a source ref of `alias` is itself a wanted sink ref: a picture
    mapped straight from the decoder to the encoder."""
    return any(
        is_src(ref) and src_parts(ref)[0] == alias and src_parts(ref)[1] == "video"
        for ref in upstream
    )


def _video_refs(g: Graph, alias: str) -> set[FrameRef]:
    """Every source ref of `alias`'s picture that anything in `g` reads."""
    refs: set[FrameRef] = set()
    for node in g.nodes.values():
        refs.update(node.inputs)
    refs.update(output.ref for output in g.outputs)
    return {
        ref
        for ref in refs
        if is_src(ref) and src_parts(ref)[0] == alias and src_parts(ref)[1] == "video"
    }


def _takes_the_card(unit: SinkUnit) -> bool:
    """Whether the unit's encoder reads frames on the device."""
    codec = unit.options.get("video_codec")
    return isinstance(codec, str) and codec.endswith(_ENCODER_SUFFIX)


def _variant_of(node: Node) -> tuple[str, dict[str, object]] | None:
    """The ``_cuda`` filter and args standing in for `node`, or None where
    the build has none or the call names an option the variant lacks."""
    spelled = _VARIANTS.get(node.filter)
    if spelled is None:
        return None
    args: dict[str, object] = {}
    for name, value in node.args.items():
        renamed = spelled.get(name)
        if renamed is None:
            return None
        args[renamed] = value
    return f"{node.filter}_cuda", args


def _build_has_variants(registry: Registry) -> bool:
    """Whether the ffmpeg this compile sees has the ``_cuda`` filters: yes
    where it is known to, and where no ffmpeg is known at all, since the
    command is then written for the reader's own."""
    if not registry.available():
        return True
    return all(registry.get(f"{name}_cuda") is not None for name in _VARIANTS)


def _mark(node: Node, into: set[FrameRef]) -> None:
    into.add(node.id)
    for pad in range(len(node.outputs)):
        into.add(f"{node.id}:{pad}")


def _slug(ref: FrameRef) -> str:
    return ref.replace(":", "_")


def _with_output_format(options: Mapping[str, object]) -> dict[str, object]:
    """`options` with ``hwaccel_output_format`` right after ``hwaccel``,
    unless the query set one itself."""
    if "hwaccel_output_format" in options:
        return dict(options)
    written: dict[str, object] = {}
    for name, value in options.items():
        written[name] = value
        if name == "hwaccel":
            written["hwaccel_output_format"] = _OUTPUT_FORMAT
    return written
