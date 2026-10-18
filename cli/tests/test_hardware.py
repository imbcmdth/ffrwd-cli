"""The pass that keeps an ffmpeg command's frames on the card: a picture
decoded with a CUDA hwaccel that reaches an ``_nvenc`` encoder stays on the
device through the filters ffmpeg has ``_cuda`` variants of, and comes down
once around the ones it has not, through a ``format`` naming the frames'
software format."""

from __future__ import annotations

from pathlib import Path

from ffrwd.hardware import keep_on_the_card
from ffrwd.ir import Graph, Node, Output, SinkUnit, StreamType
from ffrwd.probe import ProbeResult, StreamMeta
from ffrwd.registry import Registry, load_reference

SNAPSHOT = Path(__file__).resolve().parent / "data" / "reference_registry.json"


def _reference() -> Registry:
    """The pinned build, which has the `_cuda` filters."""
    return load_reference(SNAPSHOT)


def _unknown() -> Registry:
    """No ffmpeg known at all: an empty registry that spawns nothing."""
    return load_reference(SNAPSHOT.parent / "no-such-snapshot.json")


def _probe(pix_fmt: str | None, codec: str = "h264") -> ProbeResult:
    """One picture stream, as ffprobe would describe it."""
    stream = StreamMeta(
        type="video",
        index=0,
        metadata={},
        width=1920,
        height=1080,
        fps="30/1",
        sample_rate=None,
        codec=codec,
        pix_fmt=pix_fmt,
    )
    return ProbeResult(streams=[stream])


# An 8-bit 4:2:0 h264 film, the common case.
_FILM = {"f": _probe("yuv420p")}


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
    kept = keep_on_the_card(g, _reference(), _FILM)
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
    kept = keep_on_the_card(g, _reference(), _FILM)
    assert _filters(kept) == [
        ("src_f_v_0_hwdownload", "hwdownload", {}, ["src:f:v:0"]),
        ("src_f_v_0_hwdownload_format", "format", {"pix_fmts": "nv12"}, ["src_f_v_0_hwdownload"]),
        ("n1", "hflip", {}, ["src_f_v_0_hwdownload_format"]),
        ("n2", "vflip", {}, ["n1"]),
        ("n2_hwupload", "hwupload_cuda", {}, ["n2"]),
        ("n3", "scale_cuda", {"w": 1280, "h": -2}, ["n2_hwupload"]),
    ]


def test_the_download_names_the_software_format_of_the_probed_stream() -> None:
    """One row per probed pixel format the card keeps, CUDA's and Vulkan's."""
    rows = [
        ("cuda", "h264", "yuv420p", "nv12"),
        ("cuda", "h264", "yuvj420p", "nv12"),
        ("cuda", "hevc", "yuv420p10le", "p010le"),
        ("cuda", "hevc", "yuv444p", "yuv444p"),
        ("vulkan", "h264", "yuv420p", "nv12"),
        ("vulkan", "hevc", "yuv420p10le", "p010le"),
    ]
    for hwaccel, codec, pix_fmt, expected in rows:
        g = _graph(
            [_node("n1", "drawgrid", {}, ["src:f:v:0"])],
            [_sink("out.mp4", ["n1"], f"h264_{'nvenc' if hwaccel == 'cuda' else 'vulkan'}")],
            options={"f": {"hwaccel": hwaccel}},
        )
        kept = keep_on_the_card(g, _reference(), {"f": _probe(pix_fmt, codec)})
        assert kept.nodes["src_f_v_0_hwdownload_format"].args == {"pix_fmts": expected}, (
            hwaccel,
            codec,
            pix_fmt,
        )


def test_a_picture_whose_software_format_is_unknown_stays_off_the_card() -> None:
    """4:2:2, a format with no entry, 4:4:4 from a decoder that does not keep
    it, 10-bit 4:4:4, Vulkan 4:4:4, a stream the probe does not describe, and
    no probe at all: each command is what it was before the pass."""
    for hwaccel, probe in [
        ("cuda", _probe("yuv422p")),
        ("cuda", _probe("gbrp")),
        ("cuda", _probe("yuv444p", "h264")),
        ("cuda", _probe("yuv444p10le", "hevc")),
        ("vulkan", _probe("yuv444p", "hevc")),
        ("cuda", _probe(None)),
        ("cuda", ProbeResult(streams=[])),
        ("cuda", None),
    ]:
        g = _graph(
            [_node("n1", "hflip", {}, ["src:f:v:0"])],
            [_sink("out.mp4", ["n1"], f"h264_{'nvenc' if hwaccel == 'cuda' else 'vulkan'}")],
            options={"f": {"hwaccel": hwaccel}},
        )
        assert keep_on_the_card(g, _reference(), {"f": probe}) is g, (hwaccel, probe)
    g = _graph([], [_sink("out.mp4", ["src:f:v:0"], "h264_nvenc")])
    assert keep_on_the_card(g, _reference(), {}) is g


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
    kept = keep_on_the_card(g, _reference(), _FILM)
    assert [(n.id, n.filter) for n in kept.nodes.values()] == [
        ("n1", "scale_cuda"),
        ("n1_split", "split"),
        ("n1_split_2_hwdownload", "hwdownload"),
        ("n1_split_2_hwdownload_format", "format"),
        ("n2", "invert"),
        ("n1_split_1_hwdownload", "hwdownload"),
        ("n1_split_1_hwdownload_format", "format"),
    ]
    assert [unit.outputs[0].ref for unit in kept.sinks] == [
        "n1_split:0",
        "n1_split_1_hwdownload_format",
        "n2",
    ]


def test_a_download_after_card_filters_takes_the_format_of_the_picture_it_carries() -> None:
    """A 10-bit picture comes down for `hflip`, goes up for `scale_cuda`,
    and comes down again for a software encoder: both downloads name the
    film's software format, carried through the CPU segment and the upload."""
    g = _graph(
        [
            _node("n1", "hflip", {}, ["src:f:v:0"]),
            _node("n2", "scale", {"width": 1280, "height": -2}, ["n1"]),
            _node("n2_split", "split", {"n": 2}, ["n2"], ["video", "video"]),
        ],
        [
            _sink("card.mp4", ["n2_split:0"], "hevc_nvenc"),
            _sink("soft.mp4", ["n2_split:1"], "libx265"),
        ],
    )
    kept = keep_on_the_card(g, _reference(), {"f": _probe("yuv420p10le", "hevc")})
    assert _filters(kept) == [
        ("src_f_v_0_hwdownload", "hwdownload", {}, ["src:f:v:0"]),
        (
            "src_f_v_0_hwdownload_format",
            "format",
            {"pix_fmts": "p010le"},
            ["src_f_v_0_hwdownload"],
        ),
        ("n1", "hflip", {}, ["src_f_v_0_hwdownload_format"]),
        ("n1_hwupload", "hwupload_cuda", {}, ["n1"]),
        ("n2", "scale_cuda", {"w": 1280, "h": -2}, ["n1_hwupload"]),
        ("n2_split", "split", {"n": 2}, ["n2"]),
        ("n2_split_1_hwdownload", "hwdownload", {}, ["n2_split:1"]),
        (
            "n2_split_1_hwdownload_format",
            "format",
            {"pix_fmts": "p010le"},
            ["n2_split_1_hwdownload"],
        ),
    ]


def test_an_overlay_sends_its_other_picture_up_and_an_option_without_a_variant_stays_down() -> None:
    """The overlay's output carries the format of its first picture, the
    film's, whatever the logo's is."""
    g = _graph(
        [
            _node("n1", "overlay", {"x": 20, "y": 20}, ["src:f:v:0", "src:g:v:0"]),
            _node("n2", "scale", {"width": 1280, "height": -2, "flags": "lanczos"}, ["n1"]),
        ],
        [_sink("out.mp4", ["n2"], "h264_nvenc")],
        paths=["film.mp4", "logo.png"],
        options={"f": {"hwaccel": "cuda"}, "g": {"loop": True}},
    )
    probes = {"f": _probe("yuv420p10le", "hevc"), "g": _probe("rgba", "png")}
    kept = keep_on_the_card(g, _reference(), probes)
    assert _filters(kept) == [
        ("src_g_v_0_hwupload", "hwupload_cuda", {}, ["src:g:v:0"]),
        ("n1", "overlay_cuda", {"x": 20, "y": 20}, ["src:f:v:0", "src_g_v_0_hwupload"]),
        ("n1_hwdownload", "hwdownload", {}, ["n1"]),
        ("n1_hwdownload_format", "format", {"pix_fmts": "p010le"}, ["n1_hwdownload"]),
        (
            "n2",
            "scale",
            {"width": 1280, "height": -2, "flags": "lanczos"},
            ["n1_hwdownload_format"],
        ),
    ]
    assert kept.input_options["g"] == {"loop": True}, "only the decoded input is told"


def test_nothing_changes_without_a_hardware_encoder_or_a_hwaccel_or_the_filters() -> None:
    chain = [_node("n1", "scale", {"width": 1280, "height": -2}, ["src:f:v:0"])]
    software = _graph(chain, [_sink("small.mp4", ["n1"], "libx264")])
    assert keep_on_the_card(software, _reference(), _FILM) is software
    plain = _graph(chain, [_sink("small.mp4", ["n1"], "h264_nvenc")], options={"f": {}})
    assert keep_on_the_card(plain, _reference(), _FILM) is plain
    # A picture mapped straight to the encoder has nothing to spell, and
    # still stays on the card.
    bare = _graph([], [_sink("small.mp4", ["src:f:v:0"], "h264_nvenc")])
    told = keep_on_the_card(bare, _reference(), _FILM).input_options["f"]
    assert told["hwaccel_output_format"] == "cuda"
    # Where ffmpeg is known, its filter set decides: the reference build has
    # the variants, a build without them is written as it was lowered.
    card = _graph(chain, [_sink("small.mp4", ["n1"], "h264_nvenc")])
    assert keep_on_the_card(card, _unknown(), _FILM).nodes["n1"].filter == "scale_cuda"
    without = _reference()
    without._filters = {k: v for k, v in without._filters.items() if not k.endswith("_cuda")}
    assert keep_on_the_card(card, without, _FILM) is card


def test_the_vulkan_family_sends_a_picture_up_again_before_its_encoder() -> None:
    """The flips have Vulkan variants and ride the card; `pad` has none, so
    the picture comes down for it and, the encoder reading only the card's
    frames, goes up again ahead of the file. `hwupload` makes no device of
    its own, so the command names one for the filtergraph."""
    g = _graph(
        [
            _node("n1", "hflip", {}, ["src:f:v:0"]),
            _node("n2", "scale", {"width": 1280, "height": -2}, ["n1"]),
            _node("n3", "pad", {"width": 1920, "height": 1080}, ["n2"]),
        ],
        [_sink("out.mp4", ["n3"], "h264_vulkan")],
        options={"f": {"hwaccel": "vulkan"}},
    )
    kept = keep_on_the_card(g, _reference(), _FILM)
    assert kept.input_options == {
        "f": {
            "init_hw_device": "vulkan=up",
            "filter_hw_device": "up",
            "hwaccel": "vulkan",
            "hwaccel_output_format": "vulkan",
        }
    }
    assert _filters(kept) == [
        ("n1", "hflip_vulkan", {}, ["src:f:v:0"]),
        ("n2", "scale_vulkan", {"w": 1280, "h": -2}, ["n1"]),
        ("n2_hwdownload", "hwdownload", {}, ["n2"]),
        ("n2_hwdownload_format", "format", {"pix_fmts": "nv12"}, ["n2_hwdownload"]),
        ("n3", "pad", {"width": 1920, "height": 1080}, ["n2_hwdownload_format"]),
        ("n3_hwupload", "hwupload", {}, ["n3"]),
    ]
    assert kept.sinks[0].outputs[0].ref == "n3_hwupload"
    # The families do not mix: a CUDA decode reaching a Vulkan encoder is
    # left as it was lowered.
    mixed = _graph(
        [_node("n1", "hflip", {}, ["src:f:v:0"])], [_sink("out.mp4", ["n1"], "h264_vulkan")]
    )
    assert keep_on_the_card(mixed, _reference(), _FILM) is mixed


def test_the_videotoolbox_family_uploads_to_a_device_it_names() -> None:
    """On macOS `scale` and `transpose` have VideoToolbox variants; `hflip`
    has none, so the picture comes down for it and goes back up through
    `hwupload`, which makes no device of its own, so the command names one
    for the filtergraph. The encoder reads either memory, so nothing goes up
    for it."""
    g = _graph(
        [
            _node("n1", "scale", {"width": 1280, "height": -2}, ["src:f:v:0"]),
            _node("n2", "hflip", {}, ["n1"]),
            _node("n3", "transpose", {"dir": 1}, ["n2"]),
            _node("n4", "hflip", {}, ["n3"]),
        ],
        [_sink("out.mp4", ["n4"], "h264_videotoolbox")],
        options={"f": {"hwaccel": "videotoolbox"}},
    )
    kept = keep_on_the_card(g, _unknown(), _FILM)
    assert kept.input_options == {
        "f": {
            "init_hw_device": "videotoolbox=up",
            "filter_hw_device": "up",
            "hwaccel": "videotoolbox",
            "hwaccel_output_format": "videotoolbox_vld",
        }
    }
    assert _filters(kept) == [
        ("n1", "scale_vt", {"w": 1280, "h": -2}, ["src:f:v:0"]),
        ("n1_hwdownload", "hwdownload", {}, ["n1"]),
        ("n1_hwdownload_format", "format", {"pix_fmts": "nv12"}, ["n1_hwdownload"]),
        ("n2", "hflip", {}, ["n1_hwdownload_format"]),
        ("n2_hwupload", "hwupload", {}, ["n2"]),
        ("n3", "transpose_vt", {"dir": 1}, ["n2_hwupload"]),
        ("n3_hwdownload", "hwdownload", {}, ["n3"]),
        ("n3_hwdownload_format", "format", {"pix_fmts": "nv12"}, ["n3_hwdownload"]),
        ("n4", "hflip", {}, ["n3_hwdownload_format"]),
    ]
    assert kept.sinks[0].outputs[0].ref == "n4"
    # With nothing going up, no device is named.
    plain = _graph(
        [_node("n1", "scale", {"width": 1280, "height": -2}, ["src:f:v:0"])],
        [_sink("out.mp4", ["n1"], "h264_videotoolbox")],
        options={"f": {"hwaccel": "videotoolbox"}},
    )
    assert keep_on_the_card(plain, _unknown(), _FILM).input_options == {
        "f": {"hwaccel": "videotoolbox", "hwaccel_output_format": "videotoolbox_vld"}
    }
