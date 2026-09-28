"""Placing a plan by cost (ffrwd.placement_cost): the load and bandwidth
model, the search over the groups the hard rules leave, and the strategies'
spelling. Bare-machine: every plan is built by hand."""

from __future__ import annotations

import pytest

from ffrwd.errors import ErrorCode, FfrwdError
from ffrwd.ir import Graph
from ffrwd.placement import check_placement
from ffrwd.placement_cost import (
    PRESETS,
    CostStrategy,
    cost_place,
    edge_mbps,
    groups,
    parse_strategy,
    report,
)
from ffrwd.processes import (
    AudioFormat,
    FfmpegProcess,
    ProcessPlan,
    SidecarProcess,
    StreamEdge,
    VideoFormat,
)

_RAW_720 = VideoFormat(width=1280, height=720)


def _ffmpeg(pid: str, *paths: str) -> FfmpegProcess:
    return FfmpegProcess(
        id=pid,
        graph=Graph(
            input_paths=list(paths),
            sources={f"in{index}": index for index in range(len(paths))},
        ),
    )


def _region(pid: str) -> SidecarProcess:
    return SidecarProcess(id=pid, module=f"/m/{pid}.wasm", node=pid)


def _edge(source: str, target: str, format: object, ref: str = "v") -> StreamEdge:
    return StreamEdge(source=source, target=target, ref=ref, format=format)  # type: ignore[arg-type]


def _branches(count: int) -> ProcessPlan:
    """A decode feeding `count` branches, each a region and an nvenc encode
    into a publishing region."""
    processes: list[object] = [_ffmpeg("dec", "in.mp4")]
    edges: list[StreamEdge] = []
    for n in range(count):
        processes += [_region(f"fx{n}"), _ffmpeg(f"enc{n}"), _region(f"pub{n}")]
        edges += [
            _edge("dec", f"fx{n}", _RAW_720, f"d{n}"),
            _edge(f"fx{n}", f"enc{n}", _RAW_720, f"f{n}"),
            _edge(
                f"enc{n}",
                f"pub{n}",
                VideoFormat(width=1280, height=720, codec="h264_nvenc"),
                f"e{n}",
            ),
        ]
    return ProcessPlan(processes=tuple(processes), edges=tuple(edges))  # type: ignore[arg-type]


def _nodes(placement: object) -> int:
    return len(set(placement.nodes.values()))  # type: ignore[attr-defined]


def test_raw_720p30_is_about_330_mbit_and_audio_and_coded_are_small() -> None:
    assert edge_mbps(_edge("a", "b", _RAW_720)) == pytest.approx(331.776)
    assert edge_mbps(_edge("a", "b", AudioFormat(rate=48000, channels=2))) == pytest.approx(
        3.072
    )
    coded = VideoFormat(codec="h264_nvenc", options=(("video_bitrate", "2000k"),))
    assert edge_mbps(_edge("a", "b", coded)) == pytest.approx(2.0)


def test_a_plan_that_fits_one_node_runs_on_one_with_no_links() -> None:
    plan = _branches(3)
    placement = cost_place(plan, CostStrategy(encodes=3, cores=1.0))
    assert _nodes(placement) == 1
    assert placement.gpu == frozenset({0})


def test_the_encode_cap_splits_by_branch_and_never_after_an_encoder() -> None:
    """One encode a node: three L4 nodes. Each publisher stays with its
    encoder, so no coded edge is cut; each branch away from the decode costs
    one raw cut, before or after its region, whichever."""
    plan = _branches(3)
    placement = cost_place(plan, CostStrategy(encodes=1, cores=1.0))
    assert _nodes(placement) == 3
    assert len(placement.gpu) == 3
    cut = [e for e in plan.stream_edges if placement.nodes[e.source] != placement.nodes[e.target]]
    assert all(edge.format.codec == "rawvideo" for edge in cut)  # type: ignore[union-attr]
    assert len(cut) == 2


def test_every_cost_placement_keeps_the_hard_rules() -> None:
    from tests.test_placement import _head

    plan = _head()
    for strategy in (*PRESETS.values(), CostStrategy(node_cores=1.0, cores=1.0)):
        try:
            placement = cost_place(plan, strategy)
        except FfrwdError as err:
            assert err.code is ErrorCode.PLACEMENT_REFUSED
            continue
        check_placement(plan, placement)
        for group in groups(plan):
            assert len({placement.nodes[pid] for pid in group}) == 1


def test_exhaustive_is_never_worse_than_refine_on_a_small_plan() -> None:
    plan = _branches(2)
    strategy = CostStrategy(encodes=1, cores=1.0)
    refined = report(plan, ["cost:encodes=1,cores=1.0,search=refine"])[0]
    exhaustive = report(plan, ["cost:encodes=1,cores=1.0,search=exhaustive"])[0]
    assert exhaustive.cut_mbps <= refined.cut_mbps
    assert cost_place(plan, strategy).nodes


def test_a_group_bigger_than_a_node_is_refused_by_name() -> None:
    plan = _branches(1)
    with pytest.raises(FfrwdError) as caught:
        cost_place(plan, CostStrategy(encodes=0))
    assert caught.value.code is ErrorCode.PLACEMENT_REFUSED
    assert "more than one node holds" in caught.value.message


def test_a_strategy_is_a_preset_or_cost_with_parameters() -> None:
    assert parse_strategy("cost-lean") is PRESETS["cost-lean"]
    # The owner's defaults: 60% of a node's cores, 2 GPU jobs, 2 encodes and
    # 4 decodes, an L4's engines.
    owner = parse_strategy("cost")
    assert (owner.cores, owner.gpu_jobs, owner.encodes, owner.decodes) == (0.6, 2, 2, 4)
    spelled = parse_strategy("cost:gpu_jobs=2,link=50,search=greedy,cores=0.5")
    assert (spelled.gpu_jobs, spelled.link, spelled.search, spelled.cores) == (
        2,
        50.0,
        "greedy",
        0.5,
    )
    for wrong, said in (
        ("cost:depth=3", "has no parameter 'depth'"),
        ("cost:gpu_jobs=many", "is not a number"),
        ("cost:search=fast", "is not greedy, refine or exhaustive"),
        ("cheap", "no cost strategy is called 'cheap'"),
    ):
        with pytest.raises(FfrwdError) as caught:
            parse_strategy(wrong)
        assert said in caught.value.message


def test_an_nvenc_encode_is_an_encode_not_a_gpu_job() -> None:
    from ffrwd.placement_cost import process_load

    plan = _branches(1)
    load = process_load(plan.process("enc0"), plan)
    assert (load.gpu_jobs, load.encodes) == (0, 1)
    assert load.on_gpu


def test_place_takes_a_cost_strategy_and_a_placement_survives_its_trip() -> None:
    from ffrwd.placement import Placement, place

    plan = _branches(3)
    placed = place(plan, "cost:encodes=1,cores=1.0")
    assert _nodes(placed) == 3
    assert Placement.from_dict(placed.to_dict()) == placed


def test_a_strategy_name_that_means_nothing_is_refused_before_a_run() -> None:
    from ffrwd.placement_cost import check_strategy_name

    for fine in ("by-hardware", "cost", "cost-spread", "cost:encodes=1"):
        check_strategy_name(fine)
    with pytest.raises(FfrwdError):
        check_strategy_name("fastest")


def test_balance_evens_the_nodes_load_by_weight() -> None:
    """Two encodes a node forces two nodes for three branches; with balance
    the lighter node takes more of the load than the cut alone would give it."""
    plan = _branches(3)
    from ffrwd.placement_cost import Load, process_load

    def loads(strategy: CostStrategy) -> list[float]:
        placement = cost_place(plan, strategy)
        totals: dict[int, Load] = {}
        for pid, node in placement.nodes.items():
            totals[node] = totals.get(node, Load()) + process_load(plan.process(pid), plan)
        return sorted(total.cores for total in totals.values())

    even = loads(CostStrategy(encodes=2, cores=1.0, balance=250.0))
    cut_only = loads(CostStrategy(encodes=2, cores=1.0, balance=0.0))
    assert len(even) == len(cut_only) == 2
    assert even[1] - even[0] <= cut_only[1] - cut_only[0]
