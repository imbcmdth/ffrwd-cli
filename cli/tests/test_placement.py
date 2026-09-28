"""Placing a process plan on nodes: the groups no node boundary may part, the
strategies, the split into what each node runs, and what is refused.

Bare-machine: every plan is built by hand, nothing is probed or spawned.
Running a split plan is tests/test_nodes.py's and the exec tier's.
"""

from __future__ import annotations

import pytest

from ffrwd.errors import ErrorCode, FfrwdError
from ffrwd.ir import FeederCall, Graph, Lateral, LateralConnection, SinkUnit
from ffrwd.placement import (
    Cut,
    NodePlan,
    Placement,
    check_placement,
    colocation_groups,
    place,
    split,
    stdio_chains,
)
from ffrwd.processes import (
    DataFormat,
    EffectGrant,
    FeederEdge,
    FfmpegProcess,
    FileEdge,
    FileFormat,
    ProcessPlan,
    SidecarProcess,
    StreamEdge,
    VideoFormat,
)


def _ffmpeg(pid: str, *paths: str, options: dict[str, object] | None = None) -> FfmpegProcess:
    aliases = [f"in{index}" for index in range(len(paths))]
    return FfmpegProcess(
        id=pid,
        graph=Graph(
            input_paths=list(paths),
            sources={alias: index for index, alias in enumerate(aliases)},
            input_options={aliases[0]: options} if options and aliases else {},
        ),
    )


def _sidecar(pid: str, *, gpu: bool = False) -> SidecarProcess:
    grants = (EffectGrant(effect="gpu", module="m.wasm"),) if gpu else ()
    return SidecarProcess(id=pid, module=f"{pid}.wasm", node=pid, grants=grants)


def _video(source: str, target: str, ref: str = "v", *, live: bool = False) -> StreamEdge:
    return StreamEdge(source=source, target=target, ref=ref, format=VideoFormat(), live=live)


def _chain() -> ProcessPlan:
    """A file decoded, two regions in a row, and an encode: one stdio chain."""
    return ProcessPlan(
        processes=(
            _ffmpeg("ffmpeg0", "a.mp4"),
            _sidecar("sidecar0"),
            _sidecar("sidecar1", gpu=True),
            _ffmpeg("ffmpeg1"),
        ),
        edges=(
            _video("ffmpeg0", "sidecar0", "src:a:v:0"),
            _video("sidecar0", "sidecar1", "n0"),
            _video("sidecar1", "ffmpeg1", "n1"),
        ),
    )


def _head() -> ProcessPlan:
    """A SMART-like head: a live reader feeding a seller and a switch pair,
    the seller's launch stream written to the host by a lateral's writer
    that also feeds the pair on one port, and an encode into a publisher."""
    calls = (FeederCall(node="sw", function="video", param="feed"),)
    return ProcessPlan(
        processes=(
            _ffmpeg("ffmpeg0", "srt://0.0.0.0:9000?mode=listener"),
            _sidecar("sidecar0"),
            _sidecar("sidecar1"),
            _sidecar("sidecar2"),
            _ffmpeg("ffmpeg1"),
            _ffmpeg("ffmpeg2"),
            _sidecar("sidecar3"),
        ),
        edges=(
            _video("ffmpeg0", "sidecar0", "v0", live=True),
            _video("ffmpeg0", "sidecar1", "v1", live=True),
            _video("ffmpeg0", "sidecar2", "a0", live=True),
            StreamEdge(source="sidecar0", target="ffmpeg1", ref="d", format=DataFormat()),
            FeederEdge(source="ffmpeg1", target="sidecar1", port=9000, calls=calls),
            FeederEdge(source="ffmpeg1", target="sidecar2", port=9000, calls=calls),
            _video("sidecar1", "ffmpeg2", "sw"),
            _video("sidecar2", "ffmpeg2", "sa"),
            _video("ffmpeg2", "sidecar3", "enc"),
        ),
        laterals=(
            Lateral(
                function="play",
                call="play(d.launch)",
                stream="d.launch",
                tap=9100,
                template="SELECT 1",
                values=(),
                connections=(LateralConnection(port=9000, calls=calls),),
                writer="ffmpeg1",
            ),
        ),
    )


def test_groups_hold_a_feeders_writer_with_every_module_on_its_port_and_a_live_reader() -> None:
    groups = colocation_groups(_head())
    assert [(g.members, g.reason) for g in groups] == [
        (("ffmpeg0",), "live input's one reader"),
        (("sidecar1", "sidecar2", "ffmpeg1"), "feeder connection"),
    ]


def test_a_stdio_chain_is_listed_and_is_no_group() -> None:
    plan = _chain()
    assert stdio_chains(plan) == (("ffmpeg0", "sidecar0", "sidecar1", "ffmpeg1"),)
    assert colocation_groups(plan) == ()


def test_one_places_everything_on_node_zero() -> None:
    placement = place(_head(), "one")
    assert set(placement.nodes.values()) == {0}
    (only,) = split(_head(), placement)
    assert only.listens == only.dials == ()


def test_per_module_gives_each_region_a_node_and_each_ffmpeg_its_neighbours() -> None:
    plan = _chain()
    placement = place(plan, "per-module")
    assert placement.nodes == {"ffmpeg0": 0, "sidecar0": 0, "sidecar1": 1, "ffmpeg1": 1}
    assert placement.gpu == frozenset({1})
    nodes = split(plan, placement)
    cut = Cut(
        edge=1,
        key="e1",
        source="sidecar0",
        target="sidecar1",
        carried="n0",
        producer=0,
        consumer=1,
    )
    assert nodes == (
        NodePlan(node=0, processes=("ffmpeg0", "sidecar0"), dials=(cut,)),
        NodePlan(node=1, processes=("sidecar1", "ffmpeg1"), gpu=True, listens=(cut,)),
    )
    assert [e.target for e in nodes[1].view(plan).edges] == ["sidecar1", "ffmpeg1"]


def test_per_module_puts_a_head_on_three_nodes_ingest_switch_and_publish() -> None:
    plan = _head()
    placement = place(plan, "per-module")
    assert placement.nodes == {
        "ffmpeg0": 0,
        "sidecar0": 0,
        "sidecar1": 1,
        "sidecar2": 1,
        "ffmpeg1": 1,
        "ffmpeg2": 2,
        "sidecar3": 2,
    }
    nodes = split(plan, placement)
    assert [(c.source, c.target, c.producer, c.consumer) for c in nodes[1].listens] == [
        ("ffmpeg0", "sidecar1", 0, 1),
        ("ffmpeg0", "sidecar2", 0, 1),
        ("sidecar0", "ffmpeg1", 0, 1),
    ]
    assert [(c.source, c.target) for c in nodes[2].listens] == [
        ("sidecar1", "ffmpeg2"),
        ("sidecar2", "ffmpeg2"),
    ]
    assert nodes[1].view(plan).laterals == plan.laterals
    assert nodes[0].view(plan).laterals == ()


def test_per_process_cuts_every_edge_but_keeps_the_groups() -> None:
    placement = place(_head(), "per-process")
    assert placement.count == 5
    assert len({placement.node(pid) for pid in ("ffmpeg1", "sidecar1", "sidecar2")}) == 1


def test_a_node_plan_and_its_cuts_round_trip() -> None:
    for node in split(_head(), place(_head(), "per-process")):
        assert NodePlan.from_dict(node.to_dict()) == node


def _refused(plan: ProcessPlan, placement: Placement, *, shown: tuple[str, ...] = ()) -> str:
    with pytest.raises(FfrwdError) as caught:
        check_placement(plan, placement, shown=shown)
    assert caught.value.code is ErrorCode.PLACEMENT_REFUSED
    return caught.value.message


def test_a_placement_parting_a_feeder_group_is_refused() -> None:
    plan = _head()
    nodes = dict(place(plan, "per-module").nodes)
    nodes["sidecar2"] = 2
    message = _refused(plan, Placement(nodes=nodes))
    assert message == (
        "the placement splits a feeder connection: sidecar1 on node 1, "
        "sidecar2 on node 2, ffmpeg1 on node 1"
    )


def test_a_file_handed_between_stages_on_two_nodes_is_refused() -> None:
    plan = ProcessPlan(
        processes=(_ffmpeg("ffmpeg0", "a.mp4"), _ffmpeg("ffmpeg1", "pass.log")),
        edges=(
            FileEdge(source="ffmpeg0", target="ffmpeg1", format=FileFormat(path="pass.log")),
        ),
    )
    assert place(plan, "per-process").count == 1
    assert _refused(plan, Placement(nodes={"ffmpeg0": 0, "ffmpeg1": 1})) == (
        "this plan runs in 2 stages, and ffmpeg0 on node 0 hands 'pass.log' (media) to "
        "ffmpeg1 on node 1: nothing carries a file between nodes"
    )
    check_placement(plan, place(plan, "one"))


def test_a_window_on_a_node_other_than_this_one_is_refused() -> None:
    plan = _chain()
    placement = place(plan, "per-module")
    assert _refused(plan, placement, shown=("ffmpeg1",)) == (
        "ffmpeg1 would open a display window on node 1, which is not this machine"
    )
    check_placement(plan, placement, shown=("ffmpeg0",))


def test_an_unknown_strategy_is_refused() -> None:
    with pytest.raises(FfrwdError, match="no placement is called 'per-sink'"):
        place(_chain(), "per-sink")  # type: ignore[arg-type]


# -- by hardware


def test_by_hardware_puts_each_run_of_one_class_on_a_node_of_its_own() -> None:
    """A decode and a CPU region, a model on a GPU, and an encode after it:
    the cores before the model, the GPU, and the cores after it."""
    placement = place(_chain(), "by-hardware")
    assert placement.nodes == {"ffmpeg0": 0, "sidecar0": 0, "sidecar1": 1, "ffmpeg1": 2}
    assert placement.gpu == frozenset({1})


def test_an_nvenc_encode_joins_the_gpu_region_it_reads() -> None:
    plan = _chain()
    nvenc = FfmpegProcess(
        id="ffmpeg1",
        graph=Graph(
            input_paths=[],
            sources={},
            sinks=[SinkUnit(outputs=[], path="out.mp4", options={"video_codec": "h264_nvenc"})],
        ),
    )
    plan = ProcessPlan(
        processes=(*plan.processes[:3], nvenc),
        edges=plan.edges,
    )
    placement = place(plan, "by-hardware")
    assert placement.nodes["ffmpeg1"] == placement.nodes["sidecar1"]
    assert placement.gpu == frozenset({placement.nodes["sidecar1"]})


def test_a_cuda_hwaccel_marks_an_ffmpeg_process_as_needing_a_gpu() -> None:
    from ffrwd.placement import needs_gpu

    assert needs_gpu(_ffmpeg("ffmpeg0", "a.mp4", options={"hwaccel": "cuda"}))
    assert not needs_gpu(_ffmpeg("ffmpeg0", "a.mp4"))


def test_an_nvenc_encode_onto_an_edge_is_gpu_work() -> None:
    """An encode feeding a module (a publisher) names its codec on the edge,
    not on a file: that ffmpeg process computes on a GPU all the same."""
    plan = ProcessPlan(
        processes=(_ffmpeg("ffmpeg0", "a.mp4"), _sidecar("sidecar0")),
        edges=(
            StreamEdge(
                source="ffmpeg0",
                target="sidecar0",
                ref="v",
                format=VideoFormat(codec="h264_nvenc"),
            ),
        ),
    )
    placement = place(plan, "by-hardware")
    assert placement.gpu == frozenset({placement.nodes["ffmpeg0"]})
    assert placement.nodes["ffmpeg0"] != placement.nodes["sidecar0"]
