"""The addresses a hosted live run may not reach (ffrwd.reach)."""

from __future__ import annotations

import pytest

from ffrwd.compiler import compile_all
from ffrwd.reach import private_destinations, private_reason

_SHAPE = "shape => STRUCT(640 AS width, 360 AS height, 30 AS fps)"


@pytest.mark.parametrize(
    ("url", "reason"),
    [
        ("rtmp://10.0.0.5/live/k", "10.0.0.5 is a private address"),
        ("srt://192.168.1.2:9000", "192.168.1.2 is a private address"),
        ("http://127.0.0.1:8080/x", "127.0.0.1 is a loopback address"),
        ("http://169.254.169.254/latest", "169.254.169.254 is a link-local address"),
        ("tcp://[fdaa:0:1::2]:9000", "fdaa:0:1::2 is a private address"),
        ("udp://0.0.0.0:5000", "0.0.0.0 is an unspecified address, which reaches this machine"),
        ("udp://239.1.1.1:5000", "239.1.1.1 is a multicast address"),
        ("http://[::ffff:10.1.2.3]/", "::ffff:10.1.2.3 is a private address"),
        ("http://localhost:9000/", "'localhost' names a private host"),
        ("http://metadata.google.internal/", "'metadata.google.internal' names a private host"),
        ("tcp://node.modal.local:9000", "'node.modal.local' names a private host"),
    ],
)
def test_a_private_place_is_named_with_why(url: str, reason: str) -> None:
    assert private_reason(url) == reason


@pytest.mark.parametrize(
    "url",
    [
        "rtmp://relay.example.com/live/k",
        "https://8.8.8.8/x",
        "srt://[2606:4700::1]:9000",
        "out.mkv",
        "pipe:1",
        "data:text/plain,x",
    ],
)
def test_a_public_place_or_no_place_passes(url: str) -> None:
    assert private_reason(url) is None


def test_a_compiled_query_names_the_private_destination_and_not_its_listener() -> None:
    compiled = compile_all(
        f"COPY (SELECT v.video[1] FROM input('rtmp://0.0.0.0:1935/in', listen => true, "
        f"{_SHAPE}) v) TO 'rtmp://10.0.0.5/live/out' WITH (format 'flv')"
    )
    graphs = (
        list(compiled.graphs)
        if compiled.plan is None
        else [process.graph for process in compiled.plan.ffmpeg]
    )
    assert private_destinations(graphs, compiled.plan) == [
        ("rtmp://10.0.0.5/live/out", "10.0.0.5 is a private address")
    ]


def test_a_dialled_private_input_is_named() -> None:
    compiled = compile_all(
        f"COPY (SELECT v.video[1] FROM input('srt://10.1.1.1:9000', {_SHAPE}) v) "
        "TO 'rtmp://relay.example.com/live/out' WITH (format 'flv')"
    )
    graphs = (
        list(compiled.graphs)
        if compiled.plan is None
        else [process.graph for process in compiled.plan.ffmpeg]
    )
    assert private_destinations(graphs, compiled.plan) == [
        ("srt://10.1.1.1:9000", "10.1.1.1 is a private address")
    ]


@pytest.mark.parametrize(
    "source", ["srt://0.0.0.0:9000?mode=listener", "udp://0.0.0.0:5000", "udp://239.1.1.1:5000"]
)
def test_an_input_binding_its_own_address_is_not_a_place_it_reaches(source: str) -> None:
    compiled = compile_all(
        f"COPY (SELECT v.video[1] FROM input('{source}', {_SHAPE}) v) "
        "TO 'rtmp://relay.example.com/live/out' WITH (format 'flv')"
    )
    graphs = (
        list(compiled.graphs)
        if compiled.plan is None
        else [process.graph for process in compiled.plan.ffmpeg]
    )
    assert private_destinations(graphs, compiled.plan) == []


def test_the_compilers_own_loopback_connections_are_not_places_a_run_reaches() -> None:
    """A feeder's connection and a run-time lateral's tap and data ports are
    loopback the compiler made between the plan's own processes; a loopback
    address the query wrote itself is still refused."""
    from ffrwd.ir import FeederCall, Graph, Lateral, LateralConnection, SinkUnit, feeder_path

    feeder, tap, data = feeder_path(21723), feeder_path(21725), feeder_path(21727)
    graph = Graph(
        input_paths=["rtmp://0.0.0.0:1935/in"],
        sources={"v": 0},
        input_options={"v": {"listen": True}},
        sinks=[
            SinkUnit(outputs=[], path=feeder, options={"format": "nut"}),
            SinkUnit(outputs=[], path=tap, options={"format": "data"}),
            SinkUnit(outputs=[], path="tcp://127.0.0.1:9", options={"format": "nut"}),
        ],
        feeders={feeder: (FeederCall(node="sw", function="video", param="feed"),)},
        laterals=[
            Lateral(
                function="vast.play",
                call="ffrwd.vast.play(...)",
                stream="launch",
                tap=21725,
                template="",
                values=(),
                connections=(LateralConnection(port=21727, calls=()),),
            )
        ],
    )
    assert data not in {unit.path for unit in graph.sinks}
    assert private_destinations([graph]) == [
        ("tcp://127.0.0.1:9", "127.0.0.1 is a loopback address")
    ]


def test_a_plans_feeder_edges_and_laterals_are_its_own_loopback() -> None:
    """On a plan the feeder connections are edges and the laterals the
    plan's own, not any process's graph's: they are skipped all the same."""
    from ffrwd.ir import FeederCall, Graph, Lateral, LateralConnection, SinkUnit, feeder_path
    from ffrwd.processes import FeederEdge, FfmpegProcess, ProcessPlan, SidecarProcess

    feeder, tap = feeder_path(32442), feeder_path(32444)
    writer = FfmpegProcess(
        id="ffmpeg0",
        graph=Graph(
            input_paths=[],
            sources={},
            sinks=[
                SinkUnit(outputs=[], path=feeder, options={"format": "nut"}),
                SinkUnit(outputs=[], path=tap, options={"format": "data"}),
            ],
        ),
    )
    plan = ProcessPlan(
        processes=(writer, SidecarProcess(id="sidecar0", module="sw.wasm", node="sw")),
        edges=(
            FeederEdge(
                source="ffmpeg0",
                target="sidecar0",
                port=32442,
                calls=(FeederCall(node="sw", function="video", param="feed"),),
            ),
        ),
        laterals=(
            Lateral(
                function="vast.play",
                call="ffrwd.vast.play(...)",
                stream="launch",
                tap=32444,
                template="",
                values=(),
                connections=(LateralConnection(port=32446, calls=()),),
            ),
        ),
    )
    assert private_destinations([writer.graph], plan) == []
