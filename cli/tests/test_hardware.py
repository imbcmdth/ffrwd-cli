"""The pass that keeps an ffmpeg command's frames on the card: a picture
decoded with a CUDA hwaccel that reaches an ``_nvenc`` encoder stays on the
device through the filters ffmpeg has ``_cuda`` variants of, and comes down
once around the ones it has not."""

from __future__ import annotations

from pathlib import Path

from ffrwd.hardware import keep_on_the_card
from ffrwd.ir import Graph, Node, Output, SinkUnit, StreamType
from ffrwd.registry import Registry, load_reference

SNAPSHOT = Path(__file__).resolve().parent / "data" / "reference_registry.json"


def _reference() -> Registry:
    """The pinned build, which has the `_cuda` filters."""
    return load_reference(SNAPSHOT)


def _unknown() -> Registry:
    """No ffmpeg known at all: an empty registry that spawns nothing."""
    return load_reference(SNAPSHOT.parent / "no-such-snapshot.json")


def _node(
    node_id: str,
    filter_name: str,
    args: dict[str, object],
    inputs: list[str],
    outputs: list[StreamType] | None = None,
) -> Node:
    return Node(
        id=node_id,
        filter=filter_name,
        args=dict(args),
        inputs=list(inputs),
        outputs=list(outputs) if outputs is not None else ["video"],
    )


def _sink(path: str, refs: list[str], codec: str | None) -> SinkUnit:
    outputs = [Output(ref=ref, type="video", name=None, metadata={}) for ref in refs]
    options: dict[str, object] = {"video_codec": codec} if codec else {}
    return SinkUnit(outputs=outputs, path=path, options=options)


def _graph(
    nodes: list[Node],
    sinks: list[SinkUnit],
    *,
    paths: list[str] | None = None,
    options: dict[str, dict[str, object]] | None = None,
) -> Graph:
    paths = paths or ["film.mp4"]
    return Graph(
        input_paths=list(paths),
        sources={chr(ord("f") + i): i for i in range(len(paths))},
        nodes={node.id: node for node in nodes},
        sinks=sinks,
        input_options=options if options is not None else {"f": {"hwaccel": "cuda"}},
    )


def _filters(g: Graph) -> list[tuple[str, str, dict[str, object], list[str]]]:
    return [(n.id, n.filter, n.args, n.inputs) for n in g.nodes.values()]


def test_a_chain_to_an_nvenc_sink_stays_on_the_card() -> None:
    g = _graph(
        [_node("n1", "scale", {"width": 1280, "height": -2}, ["src:f:v:0"])],
        [_sink("small.mp4", ["n1"], "h264_nvenc")],
    )
    kept = keep_on_the_card(g, _reference())
    assert kept.input_options == {"f": {"hwaccel": "cuda", "hwaccel_output_format": "cuda"}}
    assert _filters(kept) == [("n1", "scale_cuda", {"w": 1280, "h": -2}, ["src:f:v:0"])]
    assert kept.sinks[0].outputs[0].ref == "n1"


def test_a_filter_with_no_variant_is_one_cpu_segment_between_a_download_and_an_upload() -> None:
    g = _graph(
        [
            _node("n1", "hflip", {}, ["src:f:v:0"]),
            _node("n2", "vflip", {}, ["n1"]),
            _node("n3", "scale", {"width": 1280, "height": -2}, ["n2"]),
        ],
        [_sink("small.mp4", ["n3"], "h264_nvenc")],
    )
    kept = keep_on_the_card(g, _reference())
    assert _filters(kept) == [
        ("src_f_v_0_hwdownload", "hwdownload", {}, ["src:f:v:0"]),
        ("n1", "hflip", {}, ["src_f_v_0_hwdownload"]),
        ("n2", "vflip", {}, ["n1"]),
        ("n2_hwupload", "hwupload_cuda", {}, ["n2"]),
        ("n3", "scale_cuda", {"w": 1280, "h": -2}, ["n2_hwupload"]),
    ]


def test_every_other_reader_of_a_card_frame_gets_one_download() -> None:
    """A software encoder and a module read the same scaled picture as the
    hardware encoder: the split rides on the card, and each of the two gets
    its own download, while the hardware encoder reads the card's frames."""
    g = _graph(
        [
            _node("n1", "scale", {"width": 1280, "height": -2}, ["src:f:v:0"]),
            _node("n1_split", "split", {"n": 3}, ["n1"], ["video", "video", "video"]),
            _node("n2", "invert", {}, ["n1_split:2"]),
        ],
        [
            _sink("card.mp4", ["n1_split:0"], "h264_nvenc"),
            _sink("soft.mp4", ["n1_split:1"], "libx264"),
            _sink("module.mp4", ["n2"], "libx264"),
        ],
    )
    kept = keep_on_the_card(g, _reference())
    assert [(n.id, n.filter) for n in kept.nodes.values()] == [
        ("n1", "scale_cuda"),
        ("n1_split", "split"),
        ("n1_split_2_hwdownload", "hwdownload"),
        ("n2", "invert"),
        ("n1_split_1_hwdownload", "hwdownload"),
    ]
    assert [unit.outputs[0].ref for unit in kept.sinks] == [
        "n1_split:0",
        "n1_split_1_hwdownload",
        "n2",
    ]


def test_an_overlay_sends_its_other_picture_up_and_an_option_without_a_variant_stays_down() -> None:
    g = _graph(
        [
            _node("n1", "overlay", {"x": 20, "y": 20}, ["src:f:v:0", "src:g:v:0"]),
            _node("n2", "scale", {"width": 1280, "height": -2, "flags": "lanczos"}, ["n1"]),
        ],
        [_sink("out.mp4", ["n2"], "h264_nvenc")],
        paths=["film.mp4", "logo.png"],
        options={"f": {"hwaccel": "cuda"}, "g": {"loop": True}},
    )
    kept = keep_on_the_card(g, _reference())
    assert _filters(kept) == [
        ("src_g_v_0_hwupload", "hwupload_cuda", {}, ["src:g:v:0"]),
        ("n1", "overlay_cuda", {"x": 20, "y": 20}, ["src:f:v:0", "src_g_v_0_hwupload"]),
        ("n1_hwdownload", "hwdownload", {}, ["n1"]),
        ("n2", "scale", {"width": 1280, "height": -2, "flags": "lanczos"}, ["n1_hwdownload"]),
    ]
    assert kept.input_options["g"] == {"loop": True}, "only the decoded input is told"


def test_nothing_changes_without_a_hardware_encoder_or_a_hwaccel_or_the_filters() -> None:
    chain = [_node("n1", "scale", {"width": 1280, "height": -2}, ["src:f:v:0"])]
    software = _graph(chain, [_sink("small.mp4", ["n1"], "libx264")])
    assert keep_on_the_card(software, _reference()) is software
    plain = _graph(chain, [_sink("small.mp4", ["n1"], "h264_nvenc")], options={"f": {}})
    assert keep_on_the_card(plain, _reference()) is plain
    # A picture mapped straight to the encoder has nothing to spell, and
    # still stays on the card.
    bare = _graph([], [_sink("small.mp4", ["src:f:v:0"], "h264_nvenc")])
    told = keep_on_the_card(bare, _reference()).input_options["f"]
    assert told["hwaccel_output_format"] == "cuda"
    # Where ffmpeg is known, its filter set decides: the reference build has
    # the variants, a build without them is written as it was lowered.
    card = _graph(chain, [_sink("small.mp4", ["n1"], "h264_nvenc")])
    assert keep_on_the_card(card, _unknown()).nodes["n1"].filter == "scale_cuda"
    without = _reference()
    without._filters = {k: v for k, v in without._filters.items() if not k.endswith("_cuda")}
    assert keep_on_the_card(card, without) is card
