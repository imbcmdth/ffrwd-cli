"""Tests for a data filter: a wasm module reading data streams of messages and
writing them, with clock pads beside them.

Bare-machine, the way tests/test_wasm.py is: every module is a synthetic
:class:`~ffrwd.wasm.Described`, every input path is one nobody has, and no
sidecar or ffmpeg is spawned. Running the ``data-stamp`` module for real is
tests/exec/test_exec_data_filter.py's.
"""

from __future__ import annotations

import functools
from dataclasses import replace
from pathlib import Path

import pytest
import sqlglot

from ffrwd import wasm
from ffrwd.compiler import Compiled, compile_all
from ffrwd.emit import build_ffmpeg_args, emit
from ffrwd.errors import ErrorCode, FfrwdError
from ffrwd.execute import keeps_clock, plan_argv, wires
from ffrwd.functions import WASM_SOURCE, Parameter, WasmFunction
from ffrwd.ir import Graph
from ffrwd.lower import lower
from ffrwd.parser import Resolved, parse, resolve
from ffrwd.probe import ProbeResult, StreamMeta
from ffrwd.processes import (
    CLOCK_SIZE,
    DataFormat,
    PadMeta,
    ProcessPlan,
    SidecarProcess,
    VideoFormat,
    external_filters,
    partition,
)
from ffrwd.registry import Registry, load_reference
from ffrwd.split import insert_splits
from ffrwd.wasm import WORLDS, Described

SNAPSHOT_PATH = Path(__file__).resolve().parent / "data" / "reference_registry.json"

STAMP = "modules/data_stamp.wasm"
AUCTION = "modules/ortb.wasm"
PUBLISH = "modules/publish.wasm"

_DECLARATIONS = {
    "stamp": "CREATE FUNCTION stamp(d data_stream, node text DEFAULT 'x', "
    "every_s number DEFAULT 0) RETURNS data_stream "
    f"AS '{STAMP}', 'data_stamp' LANGUAGE wasm;",
    "stamp_clock": "CREATE FUNCTION stamp_clock(clock video_stream, node text "
    "DEFAULT 'x', every_s number DEFAULT 0) RETURNS data_stream "
    f"AS '{STAMP}', 'data_stamp' LANGUAGE wasm;",
    "stamp_heard": "CREATE FUNCTION stamp_heard(clock audio_stream, node text "
    "DEFAULT 'x') RETURNS data_stream "
    f"AS '{STAMP}', 'data_stamp' LANGUAGE wasm;",
    "stamp_both": "CREATE FUNCTION stamp_both(d data_stream, clock video_stream, "
    "node text DEFAULT 'x', every_s number DEFAULT 0) RETURNS data_stream "
    f"AS '{STAMP}', 'data_stamp' LANGUAGE wasm;",
    "auction": "CREATE FUNCTION auction(d data_stream, clock video_stream, "
    "cohort text DEFAULT 'x', viewers number DEFAULT 0) "
    "RETURNS STRUCT(d data_stream, launch data_stream) "
    f"AS '{AUCTION}', 'auction' LANGUAGE wasm;",
    "publish": "CREATE FUNCTION publish(relay text) RETURNS sink "
    f"AS '{PUBLISH}', 'publish' LANGUAGE wasm;",
}


def _declared(query: str) -> str:
    """`query` behind the declarations it calls, and no others: a function
    no statement calls is refused."""
    called = [text for name, text in _DECLARATIONS.items() if f"{name}(" in query]
    return "\n".join([*called, query])


def _data_filter(
    name: str = "data_stamp",
    *,
    outputs: tuple[str, ...] = ("json",),
    world: str = WORLDS[-1],
    rows: dict[str, object] | None = None,
) -> Described:
    """A data filter's description, shaped like what ``--describe`` prints."""
    return Described(
        world=world,
        name=name,
        version="0.1.0",
        params_schema={
            "type": "object",
            "properties": {
                "node": {"type": "string"},
                "every_s": {"type": "number"},
                "cohort": {"type": "string"},
                "viewers": {"type": "number"},
            },
        },
        rows_schema=rows,
        data_filter=True,
        data_outputs=outputs,
        data_time_base=(1, 1_000_000),
    )


def _publisher(data_streams: wasm.SinkArity = "one") -> Described:
    return Described(
        world=WORLDS[-1],
        name="publish",
        params_schema={"type": "object", "properties": {"relay": {"type": "string"}}},
        video_codecs=("h264",),
        audio_codecs=("aac",),
        video_streams="many",
        audio_streams="many",
        data_streams=data_streams,
    )


_MODULES = {
    STAMP: _data_filter(),
    AUCTION: _data_filter("auction", outputs=("json", "json")),
    PUBLISH: _publisher(),
}


@functools.cache
def _registry() -> Registry:
    return load_reference(SNAPSHOT_PATH)


def _deal() -> dict[str, ProbeResult | None]:
    """`deal.nut` as the probe reads it: a picture, and JSON messages."""
    return {
        "f": ProbeResult(
            streams=[
                StreamMeta(
                    type="video", index=0, metadata={}, width=64, height=48,
                    fps="10/1", sample_rate=None, codec="h264",
                ),
                StreamMeta(
                    type="audio", index=0, metadata={}, width=None, height=None,
                    fps=None, sample_rate=48000, codec="aac", channels=2,
                ),
                StreamMeta(
                    type="data", index=0, metadata={}, width=None, height=None,
                    fps=None, sample_rate=None, codec="json",
                ),
            ]
        )
    }


def _lowered(query: str, modules: dict[str, Described] | None = None) -> Graph:
    return insert_splits(
        lower(
            resolve(parse(_declared(query))),
            _deal(),
            registry=_registry(),
            describes=modules or _MODULES,
        )
    )


def _refused(query: str, modules: dict[str, Described] | None = None) -> FfrwdError:
    with pytest.raises(FfrwdError) as caught:
        _lowered(query, modules)
    return caught.value


def _compiled(query: str, modules: dict[str, Described] | None = None) -> Compiled:
    found = modules or _MODULES
    return compile_all(_declared(query), describe=lambda path: found[path])


def _plan(query: str, modules: dict[str, Described] | None = None) -> ProcessPlan:
    plan = _compiled(query, modules).plan
    assert plan is not None
    return plan


def _argv(plan: ProcessPlan) -> dict[str, list[str]]:
    return plan_argv(
        plan,
        sidecar_argv=lambda process, reads, writes: wasm.shown_argv(
            process, reads, writes
        ),
        pipe_path=lambda edge, side: f"<{edge.source}-{edge.target} {side}>",
    )


def _filters(plan: ProcessPlan) -> list[SidecarProcess]:
    return [s for s in plan.sidecars if s.data_filter]


_FROM = " FROM input('deal.nut') f) TO 'out.nut'"


# -- what a module describes -----------------------------------------------


def test_the_describe_keys_of_a_data_filter_and_a_feeder_are_read() -> None:
    described = wasm._described(
        "m.wasm",
        {
            "world": "ffrwd:av@0.17.0",
            "name": "auction",
            "data_filter": True,
            "data_outputs": ["json", "json"],
            "data_time_base": [1, 1000000],
            "data_streams": "one",
            "feeders": [
                {"input": 1, "port_param": "port", "kind": "video", "group": "switch"},
                {"input": 2, "port_param": "sound", "kind": "audio"},
            ],
        },
    )
    assert described.data_filter
    assert described.data_outputs == ("json", "json")
    assert described.data_time_base == (1, 1_000_000)
    assert described.sink_streams("data") == "one"
    assert described.feeders == (
        wasm.Feeder(input=1, port_param="port", kind="video", group="switch"),
        wasm.Feeder(input=2, port_param="sound", kind="audio", group=""),
    )
    plain = wasm._described("m.wasm", {"world": "ffrwd:av@0.16.0", "name": "invert"})
    assert (plain.data_filter, plain.data_outputs, plain.data_time_base) == (
        False,
        (),
        None,
    )
    assert (plain.sink_streams("data"), plain.feeders) == ("none", ())


_CATALOG = wasm._source_catalog(
    "subscribe.wasm",
    {
        "bounded": False,
        "tracks": [
            {"codec": "h264", "time_base": [1, 90000], "row": 0,
             "extradata": "", "format": {"video": {"width": 64, "height": 48}}},
            {"codec": "json", "time_base": [1, 1000000], "row": 0,
             "extradata": "", "format": {"data": {}}},
        ],
    },
)


def test_a_packet_sources_data_track_is_a_json_data_stream_of_its_alias() -> None:
    """The catalog's data arm is read, counted apart from video and audio,
    and `s.data[1]` selects it like any probed track."""
    catalog = _CATALOG
    assert [track.kind for track in catalog.tracks] == ["video", "data"]
    declared = WasmFunction(
        name="subscribe",
        module="subscribe.wasm",
        export="subscribe",
        params=(Parameter("relay", "text"),),
        returns=WASM_SOURCE,
        line=1,
        col=1,
    )
    tree = parse("SELECT s.data[1] FROM subscribe('relay') s")
    assert isinstance(tree, sqlglot.exp.Select)
    graph = lower(
        Resolved(select=tree, input_paths=[], sources={}, branches=[tree],
                 wasm={"subscribe": declared}),
        {},
        registry=_registry(),
        describes={
            "subscribe.wasm": Described(
                world=WORLDS[-1], name="subscribe", source=True,
                params_schema={"properties": {"relay": {"type": "string"}}},
            )
        },
        probe_source=lambda module, params, **_: catalog,
    )
    (unit,) = graph.sinks
    assert [(o.ref, o.type) for o in unit.outputs] == [("src:s:d:0", "data")]
    assert [(t.ref, t.kind, t.codec) for t in graph.module_sources["s"].tracks] == [
        ("src:s:v:0", "video", "h264"),
        ("src:s:d:0", "data", "json"),
    ]


# -- the declaration -------------------------------------------------------


def test_the_shapes_a_data_filter_is_declared_in() -> None:
    """One output is the call itself; several are a struct's fields, in
    the module's output order. A clock alone is a data source in effect."""
    found = resolve(
        parse(
            _declared(
                "COPY (SELECT stamp(f.data[1]), stamp_clock(f.video[1]), "
                "auction(f.data[1], f.video[1]).d, auction(f.data[1], f.video[1]).launch"
                + _FROM
            )
        )
    ).wasm
    assert [
        (d.is_data_filter, d.is_value, d.data_fields, d.stream_kinds, d.data_output_count)
        for d in (found["stamp"], found["stamp_clock"], found["auction"])
    ] == [
        (True, False, (), ("data",), 1),
        (True, False, (), ("video",), 1),
        (True, False, ("d", "launch"), ("data", "video"), 2),
    ]
    assert found["auction"].signature == (
        "auction(d data_stream, clock video_stream, cohort text DEFAULT 'x', "
        "viewers number DEFAULT 0) RETURNS STRUCT(d data_stream, launch data_stream)"
    )
    assert [p.name for p in found["auction"].value_params] == ["cohort", "viewers"]


@pytest.mark.parametrize(
    ("signature", "needle"),
    [
        ("(n text) RETURNS data_stream", "returns data_stream and takes no stream"),
        (
            "(d data_stream) RETURNS STRUCT(d data_stream, v video_stream)",
            "returns the field 'v' as 'video_stream' beside data streams",
        ),
        (
            "(d data_stream) RETURNS STRUCT(d data_stream, d data_stream)",
            "returns the field 'd' twice",
        ),
        (
            "(s subtitle_stream) RETURNS data_stream",
            "takes 's' as subtitle_stream, and a data filter reads data_stream",
        ),
        (
            "(d data_stream[]) RETURNS data_stream",
            "takes 'd' as data_stream[], and a data filter reads data_stream",
        ),
        (
            "(d data_stream, n text, c video_stream) RETURNS data_stream",
            "takes a stream, 'c', after its values",
        ),
        (
            "(d data_stream, r STRUCT(id number)[]) RETURNS data_stream",
            "takes the annotation column 'r', and a data filter reads messages",
        ),
    ],
)
def test_a_data_filter_declaration_is_refused_by_what_it_gets_wrong(
    signature: str, needle: str
) -> None:
    with pytest.raises(FfrwdError) as caught:
        resolve(
            parse(
                f"CREATE FUNCTION f{signature} AS 'm.wasm', 'm' LANGUAGE wasm;\n"
                "COPY (SELECT f(f.data[1])" + _FROM
            )
        )
    assert caught.value.code is ErrorCode.UNSUPPORTED_SQL
    assert needle in caught.value.message


def test_a_field_a_data_filter_does_not_return_is_refused_at_resolve() -> None:
    with pytest.raises(FfrwdError) as caught:
        resolve(parse(_declared("COPY (SELECT auction(f.data[1], f.video[1]).x" + _FROM)))
    assert "auction() returns no field 'x'" in caught.value.message
    assert caught.value.hint == "it returns 'd', 'launch'"


# -- lowering ---------------------------------------------------------------


def test_a_data_filter_is_one_node_over_the_data_stream() -> None:
    graph = _lowered("COPY (SELECT stamp(f.data[1], 'es')" + _FROM)
    (name,) = graph.data_filters
    node = graph.nodes[name]
    assert (node.filter, node.args, node.inputs, node.outputs) == (
        STAMP,
        {"node": "es", "every_s": 0},
        ["src:f:d:0"],
        ["data"],
    )
    (unit,) = graph.sinks
    assert [(o.ref, o.type) for o in unit.outputs] == [(name, "data")]


def test_a_picture_clock_is_scaled_down_first_and_a_sound_clock_is_not() -> None:
    """The module reads a clock pad's time and nothing else, so only the
    frames' existence has to cross: every one, at a size that costs nothing."""
    graph = _lowered("COPY (SELECT stamp_clock(f.video[1], 'root', 0.5)" + _FROM)
    (name,) = graph.data_filters
    (scale,) = graph.nodes[name].inputs
    assert (graph.nodes[scale].filter, graph.nodes[scale].args) == (
        "scale",
        {"width": CLOCK_SIZE, "height": CLOCK_SIZE},
    )
    assert graph.nodes[scale].inputs == ["src:f:v:0"]
    heard = _lowered("COPY (SELECT stamp_heard(f.audio[1])" + _FROM)
    (name,) = heard.data_filters
    assert heard.nodes[name].inputs == ["src:f:a:0"]


def test_a_structs_fields_are_one_instance_and_another_filter_reads_one() -> None:
    """Both fields read off one call are ONE module, in a CTE as anywhere,
    and a field is a data stream another data filter takes."""
    graph = _lowered(
        "COPY (WITH w AS (SELECT auction(f.data[1], f.video[1]).d AS d, "
        "auction(f.data[1], f.video[1]).launch AS l FROM input('deal.nut') f) "
        "SELECT stamp(w.d, 'es'), w.l FROM w) TO 'out.nut'"
    )
    auction, stamp = graph.data_filters
    assert graph.nodes[auction].outputs == ["data", "data"]
    assert graph.nodes[stamp].inputs == [f"{auction}:0"]
    (unit,) = graph.sinks
    assert [o.ref for o in unit.outputs] == [stamp, f"{auction}:1"]


def test_a_data_filter_takes_its_values_by_name() -> None:
    """The leaf's call as the target query writes it, both fields one instance."""
    call = "auction(f.data[1], f.video[1], cohort => 'es-ES', viewers => 18000)"
    graph = _lowered(f"COPY (SELECT {call}.d, {call}.launch" + _FROM)
    (auction,) = graph.data_filters
    assert graph.nodes[auction].args == {"cohort": "es-ES", "viewers": 18000}
    error = _refused(
        "COPY (SELECT stamp(f.data[1], 'es', node => 'fr')" + _FROM
    )
    assert (error.code, error.message) == (
        ErrorCode.UDF_ARG_TYPE,
        "stamp() gets 'node' twice: positionally and by name",
    )


def _subscribe(call: str) -> Graph:
    declare = (
        "CREATE FUNCTION subscribe(relay text, broadcast text DEFAULT 'live') "
        "RETURNS source AS 'subscribe.wasm', 'subscribe' LANGUAGE wasm;\n"
    )
    return lower(
        resolve(parse(declare + f"SELECT s.video[1] FROM {call} s")),
        {},
        registry=_registry(),
        describes={
            "subscribe.wasm": Described(
                world=WORLDS[-1], name="subscribe", source=True,
                params_schema={
                    "properties": {
                        "relay": {"type": "string"},
                        "broadcast": {"type": "string"},
                    }
                },
            )
        },
        probe_source=lambda module, params, **_: _CATALOG,
    )


def test_a_packet_source_in_from_takes_its_values_by_name() -> None:
    """A call in FROM is bound by lowering alone, so lowering refuses what
    a call elsewhere is refused before it."""
    params = _subscribe("subscribe(broadcast => 'news', relay => 'r')").module_sources["s"]
    assert params.params == '{"broadcast": "news", "relay": "r"}'
    with pytest.raises(FfrwdError) as caught:
        _subscribe("subscribe('r', relays => 'x')")
    assert caught.value.message == "subscribe() has no parameter 'relays'"
    assert caught.value.hint == "its value parameters are 'relay', 'broadcast'"
    with pytest.raises(FfrwdError) as caught:
        _subscribe("subscribe(broadcast => 'news')")
    assert caught.value.message == "subscribe() does not write 'relay', which has no DEFAULT"


def test_a_data_stream_read_twice_is_not_split() -> None:
    """No filter splits a data stream: each reader maps it on its own."""
    graph = _lowered("COPY (SELECT f.data[1], stamp(f.data[1])" + _FROM)
    assert not any(node.filter in ("split", "asplit") for node in graph.nodes.values())


_REFUSALS = [
    (
        "COPY (SELECT stamp(f.video[1])" + _FROM,
        None,
        ErrorCode.UDF_ARG_TYPE,
        "stamp() takes 'd' as data_stream, and its argument is a video stream",
    ),
    (
        "COPY (SELECT stamp_clock(f.data[1])" + _FROM,
        None,
        ErrorCode.UDF_ARG_TYPE,
        "stamp_clock() takes 'clock' as video_stream, and its argument is a data stream",
    ),
    (
        "COPY (SELECT auction(f.data[1], f.video[1])" + _FROM,
        None,
        ErrorCode.UNSUPPORTED_SQL,
        "a struct is not a stream",
    ),
    (
        "COPY (SELECT f.video[1], auction(f.data[1], f.video[1]).d" + _FROM,
        None,
        ErrorCode.UNSUPPORTED_SQL,
        "'auction(...).launch' is read by nothing",
    ),
    (
        "COPY (SELECT stamp(f.data[1]) FROM input('deal.nut') f) TO 'out.mkv'",
        None,
        ErrorCode.UNSUPPORTED_SQL,
        "a data stream of JSON messages keeps what it is only in NUT",
    ),
    (
        "COPY (SELECT stamp(f.data[1])" + _FROM,
        {STAMP: replace(_data_filter(), data_filter=False, data_outputs=())},
        ErrorCode.UNSUPPORTED_SQL,
        "and the module 'modules/data_stamp.wasm' is not a data filter",
    ),
    (
        "COPY (SELECT stamp(f.data[1])" + _FROM,
        {STAMP: _data_filter(world="ffrwd:av@0.16.0")},
        ErrorCode.UNSUPPORTED_SQL,
        "cannot host one",
    ),
    (
        "COPY (SELECT stamp(f.data[1])" + _FROM,
        {STAMP: _data_filter(outputs=("json", "json"))},
        ErrorCode.UNSUPPORTED_SQL,
        "returns one, and the module 'modules/data_stamp.wasm' writes 2 data streams",
    ),
    (
        "COPY (SELECT stamp(f.data[1])" + _FROM,
        {STAMP: _data_filter(outputs=("cbor",))},
        ErrorCode.UNSUPPORTED_SQL,
        "writes a data stream of 'cbor'",
    ),
    (
        "COPY (SELECT f.video[1], f.data[1] FROM input('deal.nut') f) TO publish('r')",
        {PUBLISH: _publisher("none")},
        ErrorCode.UNSUPPORTED_SQL,
        "hands 'publish()' 1 data stream, and the module 'modules/publish.wasm' "
        "reads none",
    ),
]


@pytest.mark.parametrize(("query", "modules", "code", "needle"), _REFUSALS)
def test_a_data_filter_call_is_refused_by_what_it_gets_wrong(
    query: str, modules: dict[str, Described] | None, code: ErrorCode, needle: str
) -> None:
    error = _refused(query, modules)
    assert error.code is code, error
    assert needle in error.message, error.message


def test_a_frame_declaration_over_a_data_filter_module_is_refused() -> None:
    """The module says what it is, and a declaration that disagrees is
    refused by name rather than wired as a frame filter."""
    sql = (
        "CREATE FUNCTION invert(v video_stream) RETURNS video_stream "
        f"AS '{STAMP}', 'data_stamp' LANGUAGE wasm;\n"
        "COPY (SELECT invert(f.video[1])" + _FROM
    )
    with pytest.raises(FfrwdError) as caught:
        lower(resolve(parse(sql)), _deal(), registry=_registry(), describes=_MODULES)
    assert "returns video_stream, and the module" in caught.value.message
    assert "is a data filter" in caught.value.message


# -- the plan ----------------------------------------------------------------


def test_a_data_filter_is_a_sidecar_of_its_own_between_data_edges() -> None:
    """Messages cross as NUT data streams both ways, and the ffmpeg that
    only maps what came back keeps the programme's clock."""
    plan = _plan("COPY (SELECT stamp(f.data[1], 'es')" + _FROM)
    (sidecar,) = _filters(plan)
    assert (sidecar.module, sidecar.outputs, sidecar.network) == (STAMP, ("data",), False)
    formats = {(e.source, e.target): e.format for e in plan.stream_edges}
    assert set(formats.values()) == {DataFormat()}
    argv = _argv(plan)
    assert argv[sidecar.id] == [
        "ffrwd-wasm", "-f", "nut", "-i", "pipe:0", "-m", STAMP,
        "-params", '{"every_s": 0, "node": "es"}', "-f", "nut", "pipe:1",
    ]
    (reader,) = (e.target for e in plan.stream_edges if e.source == sidecar.id)
    assert all(keeps_clock(e, plan) for e in plan.stream_edges if e.target == reader)
    assert argv[reader][:2] == ["ffmpeg", "-copyts"]
    assert argv[reader][-5:-1] == ["-map", "0:d:0", "-c:0", "copy"]


def test_a_clock_crosses_as_a_tiny_picture_beside_the_data_it_times() -> None:
    """One ffmpeg reads the input once and hands the data filter both pads,
    each on a pipe of its own, in the call's own order."""
    plan = _plan("COPY (SELECT stamp_both(f.data[1], f.video[1], 'es', 0.5)" + _FROM)
    (sidecar,) = _filters(plan)
    reads = [e for e in plan.stream_edges if e.target == sidecar.id]
    assert [type(e.format) for e in reads] == [DataFormat, VideoFormat]
    clock = reads[1].format
    assert isinstance(clock, VideoFormat)
    assert (clock.width, clock.height, clock.pix_fmt, clock.codec) == (
        CLOCK_SIZE, CLOCK_SIZE, "yuv420p", "rawvideo"
    )
    assert len({e.source for e in reads}) == 1
    argv = _argv(plan)[sidecar.id]
    assert argv[: argv.index("-m")] == [
        "ffrwd-wasm",
        "-f", "nut", "-i", f"<{reads[0].source}-{sidecar.id} read>",
        "-f", "nut", "-i", f"<{reads[1].source}-{sidecar.id} read>",
    ]


def test_two_data_filters_chain_sidecar_to_sidecar() -> None:
    plan = _plan("COPY (SELECT stamp(stamp(f.data[1], 'a'), 'b')" + _FROM)
    first, second = _filters(plan)
    (between,) = [e for e in plan.stream_edges if e.source == first.id]
    assert (between.target, between.format) == (second.id, DataFormat())


def test_a_structs_outputs_leave_in_the_modules_own_order() -> None:
    """Output 0 is the first ``-f nut`` whatever reads it, and the startup
    walk orders the readers around that."""
    plan = _plan(
        "COPY (WITH w AS (SELECT auction(f.data[1], f.video[1]).d AS d, "
        "auction(f.data[1], f.video[1]).launch AS l FROM input('deal.nut') f) "
        "SELECT w.l, stamp(w.d, 'es') FROM w) TO 'out.nut'"
    )
    auction = next(s for s in _filters(plan) if s.module == AUCTION)
    leaving = [e for e in plan.stream_edges if e.source == auction.id]
    assert [e.ref.rpartition(":")[2] for e in leaving] == ["0", "1"]
    argv = _argv(plan)[auction.id]
    written = [argv[i + 1] for i, token in enumerate(argv) if token == "nut"][2:]
    assert written == [f"<{auction.id}-{e.target} write>" for e in leaving]


def test_a_data_filters_rows_ride_stdout_and_its_outputs_a_pipe_of_their_own() -> None:
    plan = _plan(
        "COPY (SELECT stamp(f.data[1])" + _FROM,
        {**_MODULES, STAMP: _data_filter(rows={"type": "object"})},
    )
    (sidecar,) = _filters(plan)
    (out,) = [w for w in wires(plan) if w.edge.source == sidecar.id]
    assert not out.write_stdio
    assert _argv(plan)[sidecar.id][-6:] == [
        "-f", "nut", f"<{sidecar.id}-{out.edge.target} write>", "-f", "ndjson", "pipe:1",
    ]


def test_a_data_stream_is_a_packet_sinks_data_pad_after_its_picture() -> None:
    plan = _plan(
        "COPY (SELECT f.video[1], stamp(f.data[1], 'es') FROM input('deal.nut') f) "
        "TO publish('r')"
    )
    sink = next(s for s in plan.sidecars if s.packet_sink)
    stamp = next(s for s in _filters(plan))
    reads = [e for e in plan.stream_edges if e.target == sink.id]
    assert [(e.source == stamp.id, type(e.format)) for e in reads] == [
        (False, VideoFormat),
        (True, DataFormat),
    ]
    # A data pad is no rendition's, so it carries no ``-pad``.
    assert [type(pad) for pad in sink.pads] == [PadMeta, type(None)]


def test_a_packet_sink_destination_takes_its_values_by_name() -> None:
    query = "COPY (SELECT f.video[1], f.data[1] FROM input('deal.nut') f) TO publish({})"
    sink = next(s for s in _plan(query.format("relay => 'r'")).sidecars if s.packet_sink)
    assert sink.args == {"relay": "r"}
    with pytest.raises(FfrwdError) as caught:
        _plan(query.format("relays => 'r'"))
    assert caught.value.message == "publish() has no parameter 'relays'"


def _subscribed(query: str) -> ProcessPlan:
    """`query` over a live source publishing a picture and a data track."""
    sql = _declared(query).replace(
        "input('deal.nut') f", "subscribe('relay') f"
    )
    declare = (
        "CREATE FUNCTION subscribe(relay text) RETURNS source "
        "AS 'subscribe.wasm', 'subscribe' LANGUAGE wasm;\n"
    )
    modules = {
        **_MODULES,
        "subscribe.wasm": Described(
            world=WORLDS[-1], name="subscribe", source=True,
            params_schema={"properties": {"relay": {"type": "string"}}},
        ),
    }
    graph = lower(
        resolve(parse(declare + sql)),
        {},
        registry=_registry(),
        describes=modules,
        probe_source=lambda module, params, **_: _CATALOG,
    )
    return partition(insert_splits(graph), external=external_filters(*modules))


def test_a_sources_data_track_crosses_straight_to_the_sidecars_reading_it() -> None:
    """No ffmpeg stands between two sidecars just to carry messages: the
    source writes the track to the data filter, and the data filter's reads
    are still in its argument order."""
    plan = _subscribed("COPY (SELECT stamp_both(f.data[1], f.video[1], 'es')" + _FROM)
    source = next(s for s in plan.sidecars if s.packet_source)
    (stamp,) = _filters(plan)
    (direct,) = [e for e in plan.stream_edges if e.source == source.id and e.target == stamp.id]
    assert (direct.ref, direct.format) == ("src:f:d:0", DataFormat())
    argv = _argv(plan)[stamp.id]
    assert argv[4] == f"<{source.id}-{stamp.id} read>"
    published = _subscribed(
        "COPY (SELECT f.video[1], f.data[1] FROM input('deal.nut') f) TO publish('r')"
    )
    sink = next(s for s in published.sidecars if s.packet_sink)
    source = next(s for s in published.sidecars if s.packet_source)
    assert [(e.source, e.ref) for e in published.stream_edges if e.target == sink.id][
        -1
    ] == (source.id, "src:f:d:0")


def test_a_sources_data_track_a_file_takes_too_goes_through_its_reader() -> None:
    """The source writes each track once, so a track two things read is
    handed round by the one ffmpeg reading the source."""
    plan = _subscribed("COPY (SELECT f.data[1], stamp(f.data[1], 'es')" + _FROM)
    source = next(s for s in plan.sidecars if s.packet_source)
    assert {e.target for e in plan.stream_edges if e.source == source.id} <= {
        p.id for p in plan.ffmpeg
    }


@pytest.mark.parametrize(
    ("columns", "waits"),
    [
        ("f.video[1], f.data[1]", True),
        ("f.audio[1], f.data[1]", True),
        ("f.data[1]", False),
        ("f.video[1], f.audio[1]", False),
    ],
)
def test_a_file_carrying_data_beside_media_waits_on_it_a_tenth_of_a_second(
    columns: str, waits: bool
) -> None:
    """A sparse data stream otherwise holds the picture back up to ten
    seconds while the muxer waits for its next message."""
    graph = lower(
        resolve(parse(f"COPY (SELECT {columns}" + _FROM)),
        _deal(),
        registry=_registry(),
    )
    argv = build_ffmpeg_args(emit(insert_splits(graph)))
    assert ("-max_interleave_delta" in argv) is waits
    if waits:
        at = argv.index("-max_interleave_delta")
        assert argv[at + 1 :] == ["100000", "out.nut"]


def test_a_data_filters_output_beside_the_picture_waits_on_it_too() -> None:
    plan = _plan("COPY (SELECT f.video[1], stamp(f.data[1], 'es')" + _FROM)
    (writer,) = [p for p in plan.ffmpeg if any(u.path == "out.nut" for u in p.graph.sinks)]
    assert _argv(plan)[writer.id][-3:] == ["-max_interleave_delta", "100000", "out.nut"]
