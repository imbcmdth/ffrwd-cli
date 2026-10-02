"""Tests for calls to node modules: declarations, shapes, ports and outputs.

Bare-machine, as tests/test_data_filter.py is: every module is a synthetic
:class:`~ffrwd.wasm.Described` describing a node, every shape is a JSON
document read by :func:`ffrwd.shapes.node_shape` exactly as the sidecar's
``--shape`` answer is, and nothing is spawned.
"""

from __future__ import annotations

import functools
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import pytest

from ffrwd import shapes, wasm
from ffrwd.compiler import compile_all
from ffrwd.errors import ErrorCode, FfrwdError
from ffrwd.execute import plan_argv
from ffrwd.ir import Graph
from ffrwd.lower import lower
from ffrwd.parser import parse, resolve
from ffrwd.probe import ProbeResult, StreamMeta
from ffrwd.registry import Registry, load_reference
from ffrwd.timing import check_live_leads, summary, timing
from ffrwd.warnings import FfrwdWarning, WarningCode
from ffrwd.wasm import WORLDS, Described

SNAPSHOT_PATH = Path(__file__).resolve().parent / "data" / "reference_registry.json"

_ROWS = {
    "type": "object",
    "properties": {
        name: {"type": "number"} for name in ("start_t", "id", "x", "y", "w", "h")
    },
}
_BOX = {
    "type": "object",
    "properties": {name: {"type": "number"} for name in ("x", "y", "w", "h")},
}
_CUE = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "start_t": {"type": "number"},
        "end_t": {"type": "number"},
    },
}


def _clock(name: str, kind: str = "video", window: int = 1) -> dict[str, object]:
    return {
        "name": name,
        "kind": kind,
        "required": True,
        "many": False,
        "pairing": {"kind": "lockstep"},
        "rows": "ignore",
        "window": window,
        "stride": window,
        "accepts": {},
    }


def _input(
    name: str,
    kind: str,
    pairing: dict[str, object],
    *,
    required: bool = False,
    many: bool = False,
    schema: Mapping[str, object] | None = None,
) -> dict[str, object]:
    port: dict[str, object] = {
        "name": name,
        "kind": kind,
        "required": required,
        "many": many,
        "pairing": pairing,
        "rows": "per-frame" if kind == "data" else "ignore",
        "window": 1,
        "stride": 1,
        "accepts": {},
    }
    if schema is not None:
        port["schema"] = json.dumps(schema)
    return port


def _output(
    name: str, kind: str, *, schema: Mapping[str, object] | None = None, latency: float = 0
) -> dict[str, object]:
    port: dict[str, object] = {"name": name, "kind": kind, "latency": latency}
    if kind == "data":
        port["format"] = {"kind": "data", "codec": "json"}
    if schema is not None:
        port["schema"] = json.dumps(schema)
    return port


def _shape(
    inputs: list[dict[str, object]],
    outputs: list[dict[str, object]],
    clock: dict[str, object],
    **rest: object,
) -> dict[str, object]:
    return {
        "inputs": inputs,
        "outputs": outputs,
        "clock": clock,
        "pure": True,
        "one_to_one": False,
        "bounded": True,
        "relation": [],
        **rest,
    }


def _spot(params: Mapping[str, object], bound: Sequence[str]) -> dict[str, object]:
    return _shape(
        [_clock("v")], [_output("spots", "data", schema=_ROWS)], {"kind": "input", "port": "v"}
    )


def _reader(port: str, schema: Mapping[str, object]) -> Callable[..., dict[str, object]]:
    def shaped(params: Mapping[str, object], bound: Sequence[str]) -> dict[str, object]:
        return _shape(
            [_clock("v"), _input(port, "data", {"kind": "lockstep"}, required=True,
                                 schema=schema)],
            [_output("v", "video")],
            {"kind": "input", "port": "v"},
        )

    return shaped


def _hear(params: Mapping[str, object], bound: Sequence[str]) -> dict[str, object]:
    return _shape(
        [_clock("a", "audio", window=96000)],
        [_output("cues", "data", schema=_CUE)],
        {"kind": "input", "port": "a"},
    )


def _burn(params: Mapping[str, object], bound: Sequence[str]) -> dict[str, object]:
    return _shape(
        [
            _clock("v"),
            _input("a", "audio", {"kind": "lockstep"}),
            _input("words", "data", {"kind": "interval", "ahead": 0}, schema=_CUE),
        ],
        [_output("v", "video")],
        {"kind": "input", "port": "v"},
    )


def _inset(params: Mapping[str, object], bound: Sequence[str]) -> dict[str, object]:
    hold = {
        "kind": "hold",
        "anchor": {"kind": "first-frame"},
        "lead": params.get("lead", 0.5),
        "port_param": "port",
    }
    return _shape(
        [_clock("v"), _input("feed", "video", hold)],
        [_output("v", "video")],
        {"kind": "input", "port": "v"},
    )


def _tile(params: Mapping[str, object], bound: Sequence[str]) -> dict[str, object]:
    hold = {"kind": "hold", "anchor": {"kind": "shared-clock"}, "lead": 0}
    clock: dict[str, object] = (
        {"kind": "rate", "rate": {"num": int(str(params["fps"])), "den": 1}}
        if "fps" in params
        else {"kind": "rate-of", "port": "v"}
    )
    return _shape(
        [_input("v", "video", hold, required=True, many=True)],
        [_output("v", "video")],
        clock,
    )


def _matte(params: Mapping[str, object], bound: Sequence[str]) -> dict[str, object]:
    return _shape(
        [_clock("v")],
        [_output("mask", "video"), _output("spots", "data", schema=_ROWS)],
        {"kind": "input", "port": "v"},
    )


def _ticker(params: Mapping[str, object], bound: Sequence[str]) -> dict[str, object]:
    canvas = {
        "kind": "video",
        "width": params.get("width", 1280),
        "height": params.get("height", 720),
        "pix_fmt": "rgba",
    }
    return _shape(
        [],
        [{"name": "v", "kind": "video", "format": canvas, "latency": 0}],
        {"kind": "rate", "rate": {"num": int(str(params.get("fps", 30))), "den": 1}},
        bounded=False,
    )


SHAPES: dict[str, Callable[..., dict[str, object]]] = {
    "spot.wasm": _spot,
    "ring.wasm": _reader("spots", _ROWS),
    "dim.wasm": _reader("boxes", _BOX),
    "hear.wasm": _hear,
    "burn.wasm": _burn,
    "inset.wasm": _inset,
    "tile.wasm": _tile,
    "matte.wasm": _matte,
    "ticker.wasm": _ticker,
}

_PARAMS = {
    "spot.wasm": {"every": {"type": "number"}},
    "dim.wasm": {"amount": {"type": "number"}},
    "inset.wasm": {"port": {"type": "integer"}, "lead": {"type": "number"}},
    "tile.wasm": {"columns": {"type": "number"}, "fps": {"type": "number"}},
    "matte.wasm": {"every": {"type": "number"}},
    "ticker.wasm": {
        name: {"type": "number" if name != "text" else "string"}
        for name in ("text", "width", "height", "fps")
    },
    "sub.wasm": {"relay": {"type": "string"}},
}


def _node(path: str) -> Described:
    return Described(
        world="node-module",
        name=path.removesuffix(".wasm"),
        version="0.1.0",
        params_schema={"type": "object", "properties": _PARAMS.get(path, {})},
        node=True,
    )


class _Asked:
    """A fake `shape` seam that counts what it is asked."""

    def __init__(self) -> None:
        self.asked: list[tuple[str, dict[str, object], tuple[str, ...]]] = []

    def __call__(
        self, module: str, params: str, bound: Sequence[str], grants: Sequence[str] = ()
    ) -> shapes.NodeShape:
        decoded = json.loads(params)
        self.asked.append((module, decoded, tuple(bound)))
        return shapes.node_shape(module, SHAPES[module](decoded, bound))


_DECLARATIONS = {
    "spot": "CREATE FUNCTION spot(v video_stream, every number DEFAULT 30) "
    "RETURNS STRUCT(start_t number, id number, x number, y number, w number, h number)[] "
    "AS 'spot.wasm', 'spot' LANGUAGE wasm;",
    "ring": "CREATE FUNCTION ring(v video_stream, spots STRUCT(start_t number, id number, "
    "x number, y number, w number, h number)[]) RETURNS video_stream "
    "AS 'ring.wasm', 'ring' LANGUAGE wasm;",
    "dim": "CREATE FUNCTION dim(v video_stream, boxes STRUCT(x number, y number, "
    "w number, h number)[], amount number DEFAULT 0.5) RETURNS video_stream "
    "AS 'dim.wasm', 'dim' LANGUAGE wasm;",
    "hear": "CREATE FUNCTION hear(a audio_stream) RETURNS cue[] "
    "AS 'hear.wasm', 'hear' LANGUAGE wasm;",
    "burn": "CREATE FUNCTION burn(v video_stream, a audio_stream DEFAULT NULL, "
    "words cue[] DEFAULT NULL) RETURNS video_stream AS 'burn.wasm', 'burn' LANGUAGE wasm;",
    "inset": "CREATE FUNCTION inset(v video_stream, feed video_stream DEFAULT NULL, "
    "port number DEFAULT 9000, lead number DEFAULT 0.5) RETURNS video_stream "
    "AS 'inset.wasm', 'inset' LANGUAGE wasm;",
    "tile": "CREATE FUNCTION tile(v video_stream[], columns number DEFAULT 2, "
    "fps number DEFAULT NULL) RETURNS video_stream AS 'tile.wasm', 'tile' LANGUAGE wasm;",
    "matte": "CREATE FUNCTION matte(v video_stream, every number DEFAULT 30) "
    "RETURNS STRUCT(mask video_stream, spots STRUCT(start_t number, id number, "
    "x number, y number, w number, h number)[]) AS 'matte.wasm', 'matte' LANGUAGE wasm;",
    "ticker": "CREATE FUNCTION ticker(text text, width number DEFAULT 1280, "
    "height number DEFAULT 720, fps number DEFAULT 30) RETURNS source "
    "AS 'ticker.wasm', 'ticker' LANGUAGE wasm;",
}


def _declared(query: str) -> str:
    called = [text for name, text in _DECLARATIONS.items() if f"{name}(" in query]
    return "\n".join([*called, query])


@functools.cache
def _registry() -> Registry:
    return load_reference(SNAPSHOT_PATH)


def _probes(
    rate: int = 48000, colour: Mapping[str, str] | None = None
) -> dict[str, ProbeResult | None]:
    said = colour or {}

    def media() -> ProbeResult:
        return ProbeResult(
            streams=[
                StreamMeta(
                    type="video", index=0, metadata={}, width=320, height=240,
                    fps="25/1", sample_rate=None, codec="h264",
                    color_range=said.get("color_range"),
                    color_primaries=said.get("color_primaries"),
                    color_transfer=said.get("color_transfer"),
                    color_space=said.get("color_space"),
                ),
                StreamMeta(
                    type="audio", index=0, metadata={}, width=None, height=None,
                    fps=None, sample_rate=rate, codec="aac", channels=2,
                ),
            ]
        )

    return {"f": media(), "a": media(), "b": media(), "c": media()}


def _lowered(
    query: str,
    modules: Mapping[str, Described] | None = None,
    asked: _Asked | None = None,
    probes: Mapping[str, ProbeResult | None] | None = None,
) -> Graph:
    return lower(
        resolve(parse(_declared(query))),
        dict(probes) if probes is not None else _probes(),
        registry=_registry(),
        describes=dict(modules) if modules is not None else {p: _node(p) for p in SHAPES},
        shapes=asked if asked is not None else _Asked(),
    )


def _refused(query: str, modules: Mapping[str, Described] | None = None) -> FfrwdError:
    with pytest.raises(FfrwdError) as caught:
        _lowered(query, modules)
    return caught.value


_FROM = " FROM input('f.mp4') f) TO 'out.mkv'"


# -- the shape document ------------------------------------------------------


def test_a_shape_document_reads_every_field_the_wit_names() -> None:
    read = shapes.node_shape(
        "m.wasm",
        {
            "inputs": [
                _clock("v"),
                _input(
                    "feed",
                    "video",
                    {
                        "kind": "hold",
                        "hold": {
                            "anchor": {"kind": "tagged", "tagged": "smart_timed"},
                            "lead": 0.3,
                            "linger": 1.5,
                            "group": "switch",
                            "port_param": "port",
                        },
                    },
                ),
                _input("words", "data", {"kind": "interval", "latency": 2, "ahead": 0.1},
                       schema=_CUE),
            ],
            "outputs": [
                {
                    "name": "mask",
                    "kind": "video",
                    "format": {"kind": "like", "port": "v", "pixel_format": "gray"},
                    "latency": 0,
                },
                {
                    "name": "clock",
                    "kind": "data",
                    "format": {"kind": "data", "data": "json"},
                    "time_base": {"num": 1, "den": 1000000},
                    "latency": 0.25,
                    "row": 0,
                },
            ],
            "clock": {"kind": "rate", "rate": {"num": 30000, "den": 1001}},
            "pure": False,
            "one_to_one": True,
            "bounded": False,
            "relation": ['{"name": "720p"}'],
        },
    )
    feed, words = read.inputs[1], read.inputs[2]
    assert feed.pairing.hold == shapes.Hold(
        anchor=shapes.Anchor("tagged", "smart_timed"),
        lead=0.3,
        linger=1.5,
        group="switch",
        port_param="port",
    )
    assert words.pairing.interval == shapes.Interval(latency=2.0, ahead=0.1)
    assert words.schema == _CUE
    assert read.outputs[0].format == shapes.OutputFormat(
        "like", port="v", pixel_format="gray"
    )
    assert read.outputs[1].format == shapes.OutputFormat("data", codec="json")
    assert (read.outputs[1].time_base, read.outputs[1].latency, read.outputs[1].row) == (
        (1, 1_000_000),
        0.25,
        0,
    )
    assert read.clock == shapes.Clock("rate", rate=(30000, 1001))
    assert (read.pure, read.one_to_one, read.bounded) == (False, True, False)
    assert read.relation == ({"name": "720p"},)


def test_a_shape_document_missing_its_ports_is_refused_naming_the_module() -> None:
    with pytest.raises(FfrwdError) as caught:
        shapes.node_shape("m.wasm", {"clock": {"kind": "self-clocked"}})
    assert "m.wasm" in caught.value.message


@pytest.mark.parametrize(
    ("port", "words"),
    [
        ((1, 1), "per-frame"),
        ((96000, 96000), "tumbling 2 s"),
        ((96000, 48000), "hopping 2 s every 1 s"),
        ((15, 1), "sliding 0.6 s"),
    ],
)
def test_a_window_is_said_in_streaming_words(port: tuple[int, int], words: str) -> None:
    window, stride = port
    kind = "audio" if window > 100 else "video"
    rate = 48000 if kind == "audio" else 25
    read = shapes.node_shape(
        "m.wasm",
        _shape([{**_clock("x", kind, window), "stride": stride}], [],
               {"kind": "input", "port": "x"}),
    )
    from fractions import Fraction

    assert shapes.window_words(read.inputs[0], Fraction(rate)) == words


def test_rows_match_by_their_fields_and_extra_fields_pass() -> None:
    assert shapes.row_mismatch(_BOX, _ROWS) is None
    integer = {"type": "object", "properties": {"x": {"type": "integer"}}}
    assert shapes.row_mismatch(_BOX, integer | {"properties": {
        name: {"type": "integer"} for name in ("x", "y", "w", "h")}}) is None
    assert shapes.row_mismatch(integer, _BOX) == ("x", "integer", "number")
    assert shapes.row_mismatch(_CUE, _ROWS) == ("text", "string", "nothing")


def test_one_shape_is_asked_once_per_module_params_and_bound_ports() -> None:
    asked = _Asked()
    cache = shapes.ShapeCache(asked)
    cache("spot.wasm", "{}", ["v"])
    cache("spot.wasm", "{}", ["v"])
    cache("spot.wasm", '{"every": 5}', ["v"])
    assert [one[1] for one in asked.asked] == [{}, {"every": 5}]


# -- declarations ------------------------------------------------------------


def test_a_signature_only_a_node_reads_resolves_and_waits_for_the_module() -> None:
    res = resolve(parse(_declared("COPY (SELECT burn(f.video[1], f.audio[1])" + _FROM)))
    burn = res.wasm["burn"]
    assert burn.is_node_only
    assert [port.name for port in burn.ports] == ["v", "a", "words"]
    assert burn.refusal is not None
    old = Described(world=WORLDS[-1], name="burn", pixel_formats=("rgba",))
    error = _refused("COPY (SELECT burn(f.video[1], f.audio[1])" + _FROM, {"burn.wasm": old})
    assert error.message == burn.refusal.message


def test_a_module_of_an_older_world_refuses_a_rows_call_outside_from() -> None:
    old = Described(world=WORLDS[-1], name="spot", video_codecs=("h264",))
    error = _refused("COPY (SELECT spot(f.video[1])" + _FROM, {"spot.wasm": old})
    assert "this call is not in FROM" in error.message


# -- ports and outputs -------------------------------------------------------


def test_a_detector_returns_rows_and_the_reader_takes_the_picture_from_the_source() -> None:
    graph = _lowered("COPY (SELECT ring(f.video[1], spot(f.video[1]))" + _FROM)
    nodes = {node.filter: node for node in graph.nodes.values()}
    spot, ring = nodes["spot.wasm"], nodes["ring.wasm"]
    assert (spot.inputs, spot.ports, spot.outputs, spot.out_ports) == (
        ["src:f:v:0"], ["v"], ["data"], ["spots"]
    )
    assert (ring.inputs, ring.ports) == (["src:f:v:0", spot.id], ["v", "spots"])
    assert graph.node_shapes[spot.id]["clock"] == {"kind": "input", "port": "v"}


def test_a_reader_naming_a_field_the_producer_lacks_is_refused_naming_both() -> None:
    SHAPES["dim.wasm"] = _reader("boxes", {
        "type": "object", "properties": {"x": {"type": "number"}, "z": {"type": "number"}},
    })
    try:
        error = _refused("COPY (SELECT dim(f.video[1], spot(f.video[1]))" + _FROM)
    finally:
        SHAPES["dim.wasm"] = _reader("boxes", _BOX)
    assert error.message == (
        "dim() reads 'z' as number on its 'boxes' input, and spot() does not write it"
    )


def test_one_call_read_twice_is_one_node() -> None:
    graph = _lowered(
        "COPY (SELECT burn(f.video[1], words => hear(f.audio[1])), f.audio[1], "
        "hear(f.audio[1])" + _FROM
    )
    hears = [node for node in graph.nodes.values() if node.filter == "hear.wasm"]
    assert len(hears) == 1
    (burn,) = [node for node in graph.nodes.values() if node.filter == "burn.wasm"]
    assert burn.inputs == ["src:f:v:0", hears[0].id]
    assert burn.ports == ["v", "words"]
    assert graph.rows_sinks[hears[0].id].container == "webvtt"


def test_a_nodes_rows_under_a_pinned_track_row_are_a_track_of_the_file() -> None:
    graph = _lowered(
        "COPY (SELECT v, spot(v) FROM input('f.mp4') f, unnest(f.video) v "
        "WHERE v.index = 1) TO 'out.mkv'"
    )
    (spot,) = [node for node in graph.nodes.values() if node.filter == "spot.wasm"]
    assert graph.rows_sinks[spot.id].container == "webvtt"
    assert [output.type for output in graph.sinks[0].outputs] == ["video", "subtitle"]


def test_kinds_mix_in_one_call_and_a_left_out_port_is_unbound() -> None:
    asked = _Asked()
    graph = _lowered(
        "COPY (SELECT burn(f.video[1], f.audio[1], hear(f.audio[1])), "
        "burn(f.video[1]) AS plain" + _FROM,
        asked=asked,
    )
    burns = [node for node in graph.nodes.values() if node.filter == "burn.wasm"]
    assert [node.ports for node in burns] == [["v", "a", "words"], ["v"]]
    assert [one[2] for one in asked.asked if one[0] == "burn.wasm"] == [
        ("v", "a", "words"),
        ("v",),
    ]


def _with_silent_source() -> dict[str, ProbeResult | None]:
    probes = _probes()
    media = probes["f"]
    assert media is not None
    probes["s"] = ProbeResult(streams=[one for one in media.streams if one.type == "video"])
    return probes


def _outer_joined(source: str, call: str) -> str:
    return (
        f"COPY (SELECT {call} FROM input('{source}.mp4') {source}, unnest({source}.video) v "
        f"LEFT JOIN unnest({source}.audio) a ON v.index = a.index) TO 'out.mkv'"
    )


@pytest.mark.parametrize(
    ("source", "bound"), [("s", ["v"]), ("f", ["v", "a"])], ids=["silent", "with-sound"]
)
def test_a_stream_an_outer_join_leaves_null_leaves_a_default_null_port_unbound(
    source: str, bound: list[str]
) -> None:
    graph = _lowered(_outer_joined(source, "burn(v, a)"), probes=_with_silent_source())
    (burn,) = [node for node in graph.nodes.values() if node.filter == "burn.wasm"]
    assert burn.ports == bound


def test_a_stream_an_outer_join_leaves_null_is_still_refused_to_a_required_port() -> None:
    with pytest.raises(FfrwdError) as caught:
        _lowered(_outer_joined("s", "burn(v, words => hear(a))"), probes=_with_silent_source())
    assert caught.value.message.startswith("'a' is NULL in row 1")


def test_a_held_input_left_unbound_keeps_its_port_param() -> None:
    graph = _lowered("COPY (SELECT inset(f.video[1], port => 9100)" + _FROM)
    (inset,) = [node for node in graph.nodes.values() if node.filter == "inset.wasm"]
    assert (inset.ports, inset.args) == (["v"], {"port": 9100, "lead": 0.5})


def test_a_port_number_where_a_held_input_goes_writes_its_port_param() -> None:
    graph = _lowered("COPY (SELECT inset(f.video[1], 9200)" + _FROM)
    (inset,) = [node for node in graph.nodes.values() if node.filter == "inset.wasm"]
    assert (inset.ports, inset.args["port"]) == (["v"], 9200)


def test_an_array_fills_a_port_taking_many() -> None:
    graph = _lowered(
        "COPY (SELECT tile(ARRAY[a.video[1], b.video[1], c.video[1]], 3) "
        "FROM input('a.mp4') a, input('b.mp4') b, input('c.mp4') c) TO 'out.mkv'"
    )
    (tile,) = [node for node in graph.nodes.values() if node.filter == "tile.wasm"]
    assert tile.inputs == ["src:a:v:0", "src:b:v:0", "src:c:v:0"]
    assert tile.ports == ["v", "v", "v"]


def test_every_field_of_a_struct_return_is_an_output_of_one_node() -> None:
    graph = _lowered(
        "COPY (WITH m AS (SELECT (matte(f.video[1])).* FROM input('f.mp4') f) "
        "SELECT dim(m.mask, m.spots), m.spots FROM m) TO 'out.mkv'"
    )
    (matte,) = [node for node in graph.nodes.values() if node.filter == "matte.wasm"]
    (dim,) = [node for node in graph.nodes.values() if node.filter == "dim.wasm"]
    assert matte.out_ports == ["mask", "spots"]
    assert dim.inputs == [f"{matte.id}:0", f"{matte.id}:1"]
    assert graph.rows_sinks[f"{matte.id}:1"].container == "webvtt"


def test_a_declared_port_the_shape_has_none_for_is_refused_bound() -> None:
    SHAPES["burn.wasm"] = lambda params, bound: _shape(
        [_clock("v")], [_output("v", "video")], {"kind": "input", "port": "v"}
    )
    try:
        unbound = _lowered("COPY (SELECT burn(f.video[1])" + _FROM)
        error = _refused("COPY (SELECT burn(f.video[1], f.audio[1])" + _FROM)
    finally:
        SHAPES["burn.wasm"] = _burn
    assert any(node.filter == "burn.wasm" for node in unbound.nodes.values())
    assert error.message == (
        "burn() binds 'a', and for these params the module 'burn.wasm' reads no input 'a'"
    )


def test_a_port_the_module_requires_is_refused_unbound() -> None:
    SHAPES["burn.wasm"] = lambda params, bound: _shape(
        [_clock("v"), _input("a", "audio", {"kind": "lockstep"}, required=True)],
        [_output("v", "video")],
        {"kind": "input", "port": "v"},
    )
    try:
        error = _refused("COPY (SELECT burn(f.video[1])" + _FROM)
    finally:
        SHAPES["burn.wasm"] = _burn
    assert error.message == "burn() leaves 'a' out, and the module 'burn.wasm' requires it"


def test_a_gather_over_a_nodes_rows_narrows_them_on_the_data_edge() -> None:
    graph = _lowered(
        "COPY (SELECT dim(f.video[1], ARRAY(SELECT s FROM unnest(spot(f.video[1])) s "
        "WHERE s.w > 20))" + _FROM
    )
    (spot,) = [node for node in graph.nodes.values() if node.filter == "spot.wasm"]
    (narrow,) = [node for node in graph.nodes.values() if node.filter == "rowfilter"]
    (dim,) = [node for node in graph.nodes.values() if node.filter == "dim.wasm"]
    assert (narrow.inputs, narrow.outputs) == ([spot.id], ["data"])
    assert dim.inputs == ["src:f:v:0", narrow.id]


def test_spans_reduce_a_nodes_rows_into_a_rows_file() -> None:
    graph = _lowered(
        "COPY (SELECT ffrwd.merge_spans(spot(f.video[1]), max_span => 10) "
        "FROM input('f.mp4') f) TO 'spots.ndjson'"
    )
    (spot,) = [node for node in graph.nodes.values() if node.filter == "spot.wasm"]
    (spans,) = [node for node in graph.nodes.values() if node.filter == "rowmerge"]
    assert (spans.inputs, spans.args) == ([spot.id], {"max_span": 10})
    assert graph.rows_sinks[spans.id].path == "spots.ndjson"


def test_a_node_reading_nothing_is_a_source_in_from() -> None:
    graph = _lowered(
        "COPY (SELECT s.video[1] FROM ticker('Nothing to see here') s WHERE s.t < 10) "
        "TO 'ticker.mp4'"
    )
    (ticker,) = [node for node in graph.nodes.values() if node.filter == "ticker.wasm"]
    assert (ticker.inputs, ticker.out_ports) == ([], ["v"])
    assert ticker.args == {"text": "Nothing to see here", "width": 1280, "height": 720,
                           "fps": 30}
    assert graph.node_sources == {"s": ticker.id}
    assert [output.ref for output in graph.sinks[0].outputs] == [ticker.id]


def test_a_sql_function_returning_a_stream_and_rows_hands_a_node_both() -> None:
    graph = _lowered(
        "CREATE FUNCTION spotted(v video_stream) RETURNS STRUCT(v video_stream, "
        "spots STRUCT(start_t number, id number, x number, y number, w number, h number)[]) "
        "AS $$ SELECT v, spot(v) AS spots $$ LANGUAGE sql;\n"
        "COPY (SELECT ring(spotted(f.video[1]))" + _FROM
    )
    (spot,) = [node for node in graph.nodes.values() if node.filter == "spot.wasm"]
    (ring,) = [node for node in graph.nodes.values() if node.filter == "ring.wasm"]
    assert (ring.inputs, ring.ports) == (["src:f:v:0", spot.id], ["v", "spots"])


def test_a_node_making_a_stream_and_its_rows_hands_a_reader_both() -> None:
    graph = _lowered("COPY (SELECT ring(matte(f.video[1]))" + _FROM)
    (matte,) = [node for node in graph.nodes.values() if node.filter == "matte.wasm"]
    (ring,) = [node for node in graph.nodes.values() if node.filter == "ring.wasm"]
    assert ring.inputs == [f"{matte.id}:0", f"{matte.id}:1"]


def test_a_struct_a_sql_function_returns_is_a_stream_and_rows_only() -> None:
    with pytest.raises(FfrwdError) as caught:
        resolve(parse(
            "CREATE FUNCTION two(v video_stream) RETURNS STRUCT(a video_stream, "
            "b video_stream) AS $$ SELECT v, v $$ LANGUAGE sql;\n"
            "COPY (SELECT two(f.video[1]).a" + _FROM
        ))
    assert caught.value.message == "function 'two' returns a struct that is not a stream and rows"


# -- what each node waits for ------------------------------------------------


def test_each_nodes_window_and_each_outputs_delay_add_up_along_the_path() -> None:
    graph = _lowered(
        "COPY (SELECT burn(f.video[1], f.audio[1], hear(f.audio[1])), f.audio[1]" + _FROM
    )
    timed = timing(graph, _probes())
    assert timed is not None
    by_module = {node.module: node for node in timed.nodes}
    assert (by_module["hear.wasm"].window, by_module["hear.wasm"].delay) == (
        "tumbling 2 s",
        2.0,
    )
    burn = by_module["burn.wasm"]
    assert (burn.window, burn.delay) == ("per-frame", 2.0)
    assert [(wait.port, wait.pairing, wait.delay) for wait in burn.waits] == [
        ("v", "clock", 0.0),
        ("a", "lockstep", 0.0),
        ("words", "interval", 2.0),
    ]
    picture, sound = timed.outputs
    assert (picture.delay, picture.holds) == (2.0, 0.0)
    assert (sound.ref, sound.delay, sound.holds, sound.held) == ("src:f:a:0", 0.0, 2.0, 768000)


def test_a_live_node_fed_later_than_its_bound_is_refused() -> None:
    bounded = {"kind": "interval", "latency": 1.0, "ahead": 0}

    def burn(params: Mapping[str, object], bound: Sequence[str]) -> dict[str, object]:
        return _shape(
            [_clock("v"), _input("words", "data", bounded, schema=_CUE)],
            [_output("v", "video")],
            {"kind": "input", "port": "v"},
        )

    SHAPES["burn.wasm"] = burn
    try:
        graph = _lowered("COPY (SELECT burn(f.video[1], words => hear(f.audio[1]))" + _FROM)
    finally:
        SHAPES["burn.wasm"] = _burn
    with pytest.raises(FfrwdError) as caught:
        check_live_leads(graph, _probes(), {"burn.wasm": (3, 4, "burn")})
    assert caught.value.code is ErrorCode.LIVE_LEAD
    assert caught.value.message == (
        "burn() needs 'words' 1 s ahead of its clock, and the path feeding it runs 2 s behind"
    )
    assert (caught.value.line, caught.value.col) == (3, 4)


# -- on the sidecar's command line -------------------------------------------


def _plan_argv(
    query: str,
    monkeypatch: pytest.MonkeyPatch,
    rate: int = 48000,
    colour: Mapping[str, str] | None = None,
) -> dict[str, list[str]]:
    """Each process of the compiled plan as the printed command shows it."""
    probes = _probes(rate, colour)
    monkeypatch.setattr(
        "ffrwd.compiler.probe_path", lambda path, args=(), **kw: probes[path[0]]
    )
    compiled = compile_all(_declared(query), describe=_node, shape=_Asked())
    assert compiled.plan is not None
    return plan_argv(
        compiled.plan,
        sidecar_argv=wasm.shown_argv,
        pipe_path=lambda edge, side: f"<{edge.source}-{edge.target} {side}>",
    )


def test_a_node_network_names_the_port_each_pad_binds(monkeypatch: pytest.MonkeyPatch) -> None:
    argv = _plan_argv(
        "COPY (SELECT ring(f.video[1], spot(f.video[1])) FROM input('f.mp4') f) "
        "TO 'ringed.mp4'",
        monkeypatch,
    )
    sidecar = argv["sidecar0"]
    assert sidecar[sidecar.index("-filter_complex") + 1] == (
        "[v=0:v]spot=every=30[spots=n1];[v=0:v][spots=n1]ring[v=out0]"
    )
    assert sidecar[sidecar.index("-map") :] == ["-map", "[out0]", "-f", "nut", "pipe:1"]


def test_every_stream_one_process_hands_a_node_network_rides_one_nut(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    argv = _plan_argv(
        "COPY (SELECT burn(f.video[1], f.audio[1], hear(f.audio[1])), f.audio[1] "
        "FROM input('f.mp4') f) TO 'burned.mp4'",
        monkeypatch,
    )
    sidecar = argv["sidecar0"]
    assert sidecar.count("-i") == 1
    assert sidecar[sidecar.index("-filter_complex") + 1] == (
        "[a=0:a]hear[cues=n1];[v=0:v][a=0:a][words=n1]burn[v=out0]"
    )
    (feeder,) = [
        words for pid, words in argv.items() if pid.startswith("ffmpeg") and "nut" in words
        and words[-1] == "pipe:1"
    ]
    assert feeder.count("-map") == 2, "both ports take the sound as it is, so it crosses once"
    assert feeder[-3:] == ["-f", "nut", "pipe:1"]


def _taking(
    shaped: Callable[..., dict[str, object]], kind: str, accepts: Mapping[str, object]
) -> Callable[..., dict[str, object]]:
    """`shaped` with every input of `kind` accepting `accepts`."""

    def taking(params: Mapping[str, object], bound: Sequence[str]) -> dict[str, object]:
        shape = shaped(params, bound)
        inputs = shape["inputs"]
        assert isinstance(inputs, list)
        for port in inputs:
            if port["kind"] == kind:
                port["accepts"] = dict(accepts)
        return shape

    return taking


def _feeder(argv: Mapping[str, list[str]]) -> list[str]:
    (feeder,) = [
        words for pid, words in argv.items() if pid.startswith("ffmpeg") and "nut" in words
        and words[-1] == "pipe:1"
    ]
    return feeder


def test_a_stream_two_nodes_read_crosses_in_the_format_both_take(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rgba = {"pixel_formats": ["rgba"]}
    monkeypatch.setitem(SHAPES, "spot.wasm", _taking(_spot, "video", rgba))
    monkeypatch.setitem(SHAPES, "ring.wasm", _taking(_reader("spots", _ROWS), "video", rgba))
    feeder = _feeder(_plan_argv(
        "COPY (SELECT ring(f.video[1], spot(f.video[1])) FROM input('f.mp4') f) "
        "TO 'ringed.mp4'",
        monkeypatch,
    ))
    assert feeder[feeder.index("-pix_fmt:0") + 1] == "rgba"


def test_a_picture_written_beside_the_nodes_reading_it_crosses_to_them_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    argv = _plan_argv(
        "COPY (SELECT ring(f.video[1], spot(f.video[1])), f.video[1] "
        "FROM input('f.mp4') f) TO 'both.mkv'",
        monkeypatch,
    )
    assert _feeder(argv).count("-map") == 1
    sidecar = argv["sidecar0"]
    assert sidecar[sidecar.index("-filter_complex") + 1] == (
        "[v=0:v]spot=every=30[spots=n1];[v=0:v][spots=n1]ring[v=out0]"
    )


@pytest.mark.parametrize("beside", ["", ", f.video[1]"], ids=["alone", "written-beside"])
def test_nodes_taking_one_picture_in_different_formats_get_a_stream_each(
    monkeypatch: pytest.MonkeyPatch, beside: str
) -> None:
    yuv, rgba = {"pixel_formats": ["yuv420p"]}, {"pixel_formats": ["rgba"]}
    monkeypatch.setitem(SHAPES, "spot.wasm", _taking(_spot, "video", yuv))
    monkeypatch.setitem(SHAPES, "ring.wasm", _taking(_reader("spots", _ROWS), "video", rgba))
    argv = _plan_argv(
        f"COPY (SELECT ring(f.video[1], spot(f.video[1])){beside} "
        "FROM input('f.mp4') f) TO 'both.mkv'",
        monkeypatch,
    )
    feeder = _feeder(argv)
    assert [feeder[at + 1] for at, word in enumerate(feeder) if word.startswith("-pix_fmt")] == [
        "yuv420p",
        "rgba",
    ]
    sidecar = argv["sidecar0"]
    assert sidecar[sidecar.index("-filter_complex") + 1] == (
        "[v=0:v]spot=every=30[spots=n1];[v=0:v:1][spots=n1]ring[v=out0]"
    )


def test_each_stream_of_one_nut_is_conformed_to_the_port_it_feeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(SHAPES, "hear.wasm", _taking(
        _hear, "audio", {"sample_formats": ["f32"], "sample_rates": [48000]}
    ))
    monkeypatch.setitem(SHAPES, "burn.wasm", _taking(_burn, "audio", {"sample_formats": ["f32"]}))
    feeder = _feeder(_plan_argv(
        "COPY (SELECT burn(f.video[1], f.audio[1], hear(f.audio[1])), f.audio[1] "
        "FROM input('f.mp4') f) TO 'burned.mp4'",
        monkeypatch,
        rate=44100,
    ))
    assert [word for word in feeder if word.startswith("-ar")] == ["-ar:0"]
    assert feeder[feeder.index("-ar:0") + 1] == "48000"


def test_a_node_read_in_from_has_no_input_and_its_reader_ends_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    argv = _plan_argv(
        "COPY (SELECT s.video[1] FROM ticker('Nothing to see here') s WHERE s.t < 10) "
        "TO 'ticker.mp4'",
        monkeypatch,
    )
    assert "-i" not in argv["sidecar0"]
    reader = argv["ffmpeg0"]
    assert reader[reader.index("-to") + 1] == "10"


def test_a_node_source_whose_relation_is_its_renditions_ends_where_its_reader_says(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def ticker(params: Mapping[str, object], bound: Sequence[str]) -> dict[str, object]:
        shape = _ticker(params, bound)
        outputs = shape["outputs"]
        assert isinstance(outputs, list)
        outputs[0]["row"] = 0
        shape["relation"] = [json.dumps({"width": 1280, "height": 720})]
        return shape

    monkeypatch.setitem(SHAPES, "ticker.wasm", ticker)
    argv = _plan_argv(
        "COPY (SELECT s.video[1] FROM ticker('Nothing to see here') s "
        "WHERE s.height = 720 AND s.t < 10) TO 'ticker.mp4'",
        monkeypatch,
    )
    reader = argv["ffmpeg0"]
    assert reader[reader.index("-to") + 1] == "10"


_BT709_PC = {
    "color_range": "pc",
    "color_primaries": "bt709",
    "color_transfer": "bt709",
    "color_space": "bt709",
}
_RING = "COPY (SELECT ring(f.video[1], spot(f.video[1])) FROM input('f.mp4') f) TO 'ringed.mp4'"


def _pad_after_input(sidecar: Sequence[str]) -> object:
    """The ``-pad`` JSON written right after the sidecar's one ``-i``."""
    at = sidecar.index("-i") + 2
    assert sidecar[at] == "-pad"
    return json.loads(sidecar[at + 1])


def _taking_pictures(monkeypatch: pytest.MonkeyPatch, pixel_format: str) -> None:
    taken = {"pixel_formats": [pixel_format]}
    monkeypatch.setitem(SHAPES, "spot.wasm", _taking(_spot, "video", taken))
    monkeypatch.setitem(SHAPES, "ring.wasm", _taking(_reader("spots", _ROWS), "video", taken))


def test_a_yuv_picture_into_a_node_network_carries_the_probed_colour(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _taking_pictures(monkeypatch, "yuv420p")
    argv = _plan_argv(_RING, monkeypatch, colour=_BT709_PC)
    assert _pad_after_input(argv["sidecar0"]) == {
        "color": {"range": "pc", "primaries": "bt709", "trc": "bt709", "space": "bt709"}
    }


def test_a_picture_the_probe_says_nothing_of_carries_unknown_colour(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _taking_pictures(monkeypatch, "yuv420p")
    argv = _plan_argv(_RING, monkeypatch)
    assert _pad_after_input(argv["sidecar0"]) == {
        "color": {"range": "unknown", "primaries": "unknown", "trc": "unknown",
                  "space": "unknown"}
    }


def test_a_picture_converted_to_rgb_on_its_way_carries_what_the_conversion_wrote(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _taking_pictures(monkeypatch, "rgba")
    argv = _plan_argv(_RING, monkeypatch, colour=_BT709_PC)
    assert _pad_after_input(argv["sidecar0"]) == {
        "color": {"range": "pc", "primaries": "bt709", "trc": "bt709", "space": "gbr"}
    }


def test_a_node_network_hands_one_process_every_stream_it_reads_on_one_nut(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    argv = _plan_argv(
        "COPY (WITH m AS (SELECT f.video[1] AS v, (matte(f.video[1])).* "
        "FROM input('f.mp4') f) SELECT dim(m.v, m.spots), m.mask FROM m) TO 'matte.mkv'",
        monkeypatch,
    )
    sidecar = argv["sidecar0"]
    assert sidecar.count("-i") == 1
    assert sidecar[sidecar.index("-map") :] == [
        "-map", "[out0]", "-map", "[out1]", "-f", "nut", "pipe:1",
    ]
    assert _feeder(argv).count("-map") == 1


def _gray_matte(params: Mapping[str, object], bound: Sequence[str]) -> dict[str, object]:
    shape = _matte(params, bound)
    outputs = shape["outputs"]
    assert isinstance(outputs, list)
    outputs[0]["format"] = {"kind": "like", "port": "v", "pixel_format": "gray"}
    return shape


def test_a_node_handed_another_nodes_output_in_a_format_it_does_not_take_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rgba = {"pixel_formats": ["rgba"]}
    monkeypatch.setitem(SHAPES, "matte.wasm", _taking(_gray_matte, "video", rgba))
    monkeypatch.setitem(SHAPES, "dim.wasm", _taking(_reader("boxes", _BOX), "video", rgba))
    with pytest.raises(FfrwdError) as caught:
        _plan_argv(
            "COPY (WITH m AS (SELECT (matte(f.video[1])).* FROM input('f.mp4') f) "
            "SELECT dim(m.mask, m.spots) FROM m) TO 'dimmed.mkv'",
            monkeypatch,
        )
    assert caught.value.message == (
        "function 'dim': the module 'dim.wasm' takes rgba on 'v', and the 'mask' "
        "output of matte hands it gray"
    )
    assert caught.value.hint is not None
    assert "ffmpeg.format(<stream>, pix_fmts => 'rgba')" in caught.value.hint


def test_an_ffmpeg_filter_between_two_nodes_hands_the_reader_its_format(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rgba = {"pixel_formats": ["rgba"]}
    monkeypatch.setitem(SHAPES, "matte.wasm", _taking(_gray_matte, "video", rgba))
    monkeypatch.setitem(SHAPES, "dim.wasm", _taking(_reader("boxes", _BOX), "video", rgba))
    argv = _plan_argv(
        "COPY (WITH m AS (SELECT (matte(f.video[1])).* FROM input('f.mp4') f) "
        "SELECT dim(ffmpeg.format(m.mask, pix_fmts => 'rgba'), m.spots) FROM m) "
        "TO 'dimmed.mkv'",
        monkeypatch,
    )
    (converts,) = [words for words in argv.values() if "format=pix_fmts=rgba" in str(words)]
    assert converts[converts.index("-pix_fmt:0") + 1] == "rgba"


def test_spans_are_the_hosts_rowmerge_with_its_span_written_as_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    argv = _plan_argv(
        "COPY (SELECT ffrwd.merge_spans(spot(f.video[1]), max_span => 10) "
        "FROM input('f.mp4') f) TO 'spots.ndjson'",
        monkeypatch,
    )
    sidecar = argv["sidecar0"]
    assert sidecar[sidecar.index("-filter_complex") + 1] == (
        "[v=0:v]spot=every=30[spots=n1];[n1]rowmerge=max_span=10[out0]"
    )
    assert sidecar[-5:] == ["-map", "[out0]", "-f", "ndjson", "spots.ndjson"]


def test_spans_without_a_span_are_refused() -> None:
    with pytest.raises(FfrwdError) as caught:
        _lowered("COPY (SELECT ffrwd.merge_spans(spot(f.video[1])) FROM input('f.mp4') f) "
                 "TO 'spots.ndjson'")
    assert caught.value.message == "ffrwd.merge_spans() needs 'max_span'"


_SPANS = {
    "type": "object",
    "properties": {"start_t": {"type": "number"}, "end_t": {"type": "number"},
                   "id": {"type": "integer"}},
}
_MASK = (
    "CREATE FUNCTION mask(v video_stream, spans STRUCT(start_t number, end_t number, "
    "id number)[]) RETURNS video_stream AS 'mask.wasm', 'mask' LANGUAGE wasm;\n"
)


def test_spans_are_read_with_the_end_and_id_the_merge_writes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(SHAPES, "mask.wasm", _reader("spans", _SPANS))
    graph = _lowered(
        _MASK + "COPY (SELECT mask(f.video[1], ffrwd.merge_spans(spot(f.video[1]), "
        "max_span => 10))" + _FROM
    )
    (spans,) = [node for node in graph.nodes.values() if node.filter == "rowmerge"]
    (mask,) = [node for node in graph.nodes.values() if node.filter == "mask.wasm"]
    assert mask.inputs == ["src:f:v:0", spans.id]


def test_rows_read_with_an_end_they_do_not_carry_are_still_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(SHAPES, "mask.wasm", _reader("spans", _SPANS))
    with pytest.raises(FfrwdError) as caught:
        _lowered(_MASK + "COPY (SELECT mask(f.video[1], spot(f.video[1]))" + _FROM)
    assert caught.value.message == (
        "mask() reads 'end_t' as number on its 'spans' input, and spot() does not write it"
    )


def test_explain_delays_says_each_window_and_each_outputs_delay() -> None:
    graph = _lowered(
        "COPY (SELECT burn(f.video[1], f.audio[1], hear(f.audio[1])), f.audio[1]" + _FROM
    )
    timed = timing(graph, _probes(), {"hear.wasm": "hear", "burn.wasm": "burn"})
    assert timed is not None
    assert summary(timed).splitlines() == [
        "hear: tumbling 2 s",
        "burn: per-frame; words by interval, no bound",
        "out.mkv stream 0 (video): 2 s behind the source",
        "out.mkv stream 1 (audio): 0 s behind the source, waits 2 s",
    ]


# -- coded packets -----------------------------------------------------------


def _subscribe(params: Mapping[str, object], bound: Sequence[str]) -> dict[str, object]:
    def coded(name: str, codec: str, row: int, carried: dict[str, object]) -> dict[str, object]:
        return {
            "name": name,
            "kind": "packets",
            "format": {
                "kind": "packets",
                "codec": codec,
                "time_base": {"num": 1, "den": 90000},
                "format": carried,
                "extradata": "",
                "profile": None,
                "level": None,
            },
            "latency": 0,
            "row": row,
        }

    return _shape(
        [],
        [
            coded("hd", "h264", 0, {"kind": "video", "width": 1280, "height": 720}),
            coded("hd_audio", "aac", 0, {"kind": "audio", "sample_rate": 48000, "channels": 2}),
            coded("sd", "h264", 1, {"kind": "video", "width": 640, "height": 360}),
        ],
        {"kind": "self_clocked"},
        bounded=False,
        relation=['{"name": "720p", "bandwidth": 3000000}', '{"name": "360p"}'],
    )


def _remux(params: Mapping[str, object], bound: Sequence[str]) -> dict[str, object]:
    coded = {**_clock("v"), "kind": "packets", "accepts": {"codecs": ["h264"]}}
    return _shape(
        [coded],
        [{"name": "v", "kind": "packets", "format": {"kind": "like", "port": "v"}, "latency": 0}],
        {"kind": "input", "port": "v"},
    )


def test_a_node_source_writing_coded_packets_binds_one_row_per_rendition() -> None:
    SHAPES["sub.wasm"] = _subscribe
    try:
        graph = _lowered(
            "CREATE FUNCTION sub(relay text) RETURNS source AS 'sub.wasm', 'sub' LANGUAGE wasm;\n"
            "COPY (SELECT v.video[1] FROM sub('r') v WHERE v.height = 720) TO 'out.mkv'",
            {"sub.wasm": _node("sub.wasm")},
        )
    finally:
        del SHAPES["sub.wasm"]
    (sub,) = [node for node in graph.nodes.values() if node.filter == "sub.wasm"]
    assert sub.outputs == ["video", "audio", "video"]
    assert [output.ref for output in graph.sinks[0].outputs] == [f"{sub.id}:0"]


def test_a_node_reading_coded_packets_is_handed_the_stream_as_it_was_coded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    SHAPES["remux.wasm"] = _remux
    _DECLARATIONS["remux"] = (
        "CREATE FUNCTION remux(v video_stream) RETURNS video_stream "
        "AS 'remux.wasm', 'remux' LANGUAGE wasm;"
    )
    try:
        argv = _plan_argv(
            "COPY (SELECT remux(f.video[1]) FROM input('f.mp4') f) TO 'out.mkv'", monkeypatch
        )
    finally:
        del SHAPES["remux.wasm"]
        del _DECLARATIONS["remux"]
    feeder = argv["ffmpeg1"]
    assert feeder[feeder.index("-c:0") + 1] == "copy"
    reader = argv["ffmpeg0"]
    assert reader[reader.index("-c:0") + 1] == "copy"


def test_params_too_long_for_a_command_line_are_read_from_a_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    long = "x" * (shapes.PARAMS_INLINE_LIMIT + 1)
    argv = _plan_argv(
        f"COPY (SELECT s.video[1] FROM ticker('{long}') s) TO 'ticker.mp4'", monkeypatch
    )
    sidecar = argv["sidecar0"]
    assert sidecar[sidecar.index("-filter_complex") + 1] == "ticker[v=out0]"
    assert sidecar[sidecar.index("-params-from") + 1] == "ticker=ffrwd:params:sidecar0:n1"


def test_a_packets_function_over_a_node_hands_back_the_stream_still_coded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    SHAPES["remux.wasm"] = _remux
    _DECLARATIONS["remux"] = (
        "CREATE FUNCTION remux(v video_stream) RETURNS packets "
        "AS 'remux.wasm', 'remux' LANGUAGE wasm;"
    )
    try:
        argv = _plan_argv(
            "COPY (SELECT remux(f.video[1]) FROM input('f.mp4') f) TO 'out.mkv'", monkeypatch
        )
    finally:
        del SHAPES["remux.wasm"]
        del _DECLARATIONS["remux"]
    sidecar = argv["sidecar0"]
    assert sidecar[sidecar.index("-filter_complex") + 1] == "[v=0:v]remux[v=out0]"
    assert argv["ffmpeg1"][argv["ffmpeg1"].index("-c:0") + 1] == "copy"


def test_a_live_query_feeding_a_node_later_than_its_bound_is_refused_at_compile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bounded = {"kind": "interval", "latency": 1.0, "ahead": 0}

    def burn(params: Mapping[str, object], bound: Sequence[str]) -> dict[str, object]:
        return _shape(
            [_clock("v"), _input("words", "data", bounded, schema=_CUE)],
            [_output("v", "video")],
            {"kind": "input", "port": "v"},
        )

    live = _probes()["f"]
    monkeypatch.setattr("ffrwd.compiler.probe_path", lambda path, args=(), **kw: live)
    SHAPES["burn.wasm"] = burn
    try:
        with pytest.raises(FfrwdError) as caught:
            compile_all(
                _declared(
                    "COPY (SELECT burn(f.video[1], words => hear(f.audio[1])) "
                    "FROM input('srt://127.0.0.1:9000') f) TO 'out.mkv'"
                ),
                describe=_node,
                shape=_Asked(),
            )
    finally:
        SHAPES["burn.wasm"] = _burn
    assert caught.value.code is ErrorCode.LIVE_LEAD
    assert (caught.value.line, caught.value.col) == (2, 17)


def test_a_stream_waiting_long_beside_a_later_one_is_warned_about(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def hear(params: Mapping[str, object], bound: Sequence[str]) -> dict[str, object]:
        return _shape(
            [_clock("a", "audio", window=48000 * 1500)],
            [_output("cues", "data", schema=_CUE)],
            {"kind": "input", "port": "a"},
        )

    probes = _probes()
    monkeypatch.setattr(
        "ffrwd.compiler.probe_path", lambda path, args=(), **kw: probes[path[0]]
    )
    said: list[FfrwdWarning] = []
    SHAPES["hear.wasm"] = hear
    try:
        compile_all(
            _declared(
                "COPY (SELECT burn(f.video[1], words => hear(f.audio[1])), f.audio[1] "
                "FROM input('f.mp4') f) TO 'out.mkv'"
            ),
            describe=_node,
            shape=_Asked(),
            on_warning=said.append,
        )
    finally:
        SHAPES["hear.wasm"] = _hear
    assert [warning.code for warning in said] == [WarningCode.HELD_STREAM]


def test_a_field_only_a_node_makes_is_refused_for_a_module_of_an_older_world() -> None:
    old = Described(world=WORLDS[-1], name="matte", pixel_formats=("rgba",))
    error = _refused("COPY (SELECT matte(f.video[1]).mask" + _FROM, {"matte.wasm": old})
    assert error.message == (
        "'.mask' is the stream matte() was handed, and a stream is not read back off a struct"
    )


def _weave(params: Mapping[str, object], bound: Sequence[str]) -> dict[str, object]:
    coded = {**_clock("v"), "kind": "packets", "accepts": {"codecs": ["h264"]}}
    clip = _input("clip", "data", {"kind": "interval", "ahead": 0}, schema=_ROWS)
    return _shape(
        [coded, *([clip] if "clip" in bound else [])],
        [{"name": "v", "kind": "packets", "format": {"kind": "like", "port": "v"}, "latency": 0}],
        {"kind": "input", "port": "v"},
    )


def test_a_node_reading_packets_reads_what_its_destination_encodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    SHAPES["weave.wasm"] = _weave
    _DECLARATIONS["weave"] = (
        "CREATE FUNCTION weave(v video_stream, clip STRUCT(start_t number, id number, "
        "x number, y number, w number, h number)[] DEFAULT NULL) RETURNS packets "
        "AS 'weave.wasm', 'weave' LANGUAGE wasm;"
    )
    try:
        argv = _plan_argv(
            "COPY (SELECT weave(f.video[1], clip => spot(f.video[1])) FROM input('f.mp4') f) "
            "TO 'out.mkv' WITH (video_codec 'libx264', crf 20)",
            monkeypatch,
        )
    finally:
        del SHAPES["weave.wasm"]
        del _DECLARATIONS["weave"]
    (weave,) = [words for words in argv.values() if "weave=weave.wasm" in words]
    assert weave[weave.index("-filter_complex") + 1] == "[v=0:v][clip=1:d]weave[v=out0]"
    (encoder,) = [words for words in argv.values() if "libx264" in words]
    assert encoder[encoder.index("-c:0") + 1] == "libx264"
    (writer,) = [words for words in argv.values() if "out.mkv" in words]
    assert writer[writer.index("-c:0") + 1] == "copy"
