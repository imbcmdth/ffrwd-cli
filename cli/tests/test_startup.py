"""The order a plan's processes start in, read off the argv they render.

Unit tier: every shape here lowers symbolically -- a synthetic
:class:`~ffrwd.probe.ProbeResult` for the camera, a synthetic
:class:`~ffrwd.wasm.Described` for the module, paths that do not exist, and
nothing spawned. What is asserted is what the emitted argv says: which pipe
each process opens first, and which pipe each producer writes first.

The shapes at the top are the ones that wedged: one camera feeding a module
and two merge legs, the same with an audio leg, and the same query written
through a CTE. They differ only in the order a SELECT list happens to be
written in, which is exactly the thing that must not decide whether a plan
can run.
"""

from __future__ import annotations

import functools
from dataclasses import replace
from pathlib import Path

import pytest

from ffrwd.compiler import compile_all
from ffrwd.errors import ErrorCode, FfrwdError
from ffrwd.execute import plan_argv
from ffrwd.ir import Graph, Node, Output, SinkUnit, StreamType
from ffrwd.probe import ProbeResult, StreamMeta
from ffrwd.processes import (
    ModuleShape,
    ProcessPlan,
    SidecarProcess,
    StreamEdge,
    external_ids,
    partition,
)
from ffrwd.registry import Registry, load_reference
from ffrwd.startup import arrange, check, relation, stalled
from ffrwd.wasm import Described

SNAPSHOT_PATH = Path(__file__).resolve().parent / "data" / "reference_registry.json"

CAMERA = "srt://camera.local:9000"
MODULE = "modules/invert.wasm"
DECLARE = (
    "CREATE FUNCTION invert(v video_stream) RETURNS video_stream\n"
    f"  AS '{MODULE}', 'invert' LANGUAGE wasm;\n"
)


@functools.cache
def _snapshot_registry() -> Registry:
    return load_reference(SNAPSHOT_PATH)


def _described(_path: str) -> Described:
    return Described(
        world="ffrwd:av@0.3.0",
        name="invert",
        version="0.1.0",
        params_schema={"type": "object", "additionalProperties": False, "properties": {}},
        rows_schema=None,
        pixel_formats=("rgba",),
        window=1,
        windowed=False,
    )


def _live_probe() -> ProbeResult:
    return ProbeResult(
        streams=[
            StreamMeta(
                type="video",
                index=0,
                metadata={},
                width=640,
                height=360,
                fps="30/1",
                sample_rate=None,
                codec="h264",
            ),
            StreamMeta(
                type="audio",
                index=0,
                metadata={},
                width=None,
                height=None,
                fps=None,
                sample_rate=48000,
                channels=2,
                codec="aac",
            ),
        ],
        duration=None,
    )


@pytest.fixture(autouse=True)
def _offline_camera(monkeypatch: pytest.MonkeyPatch) -> None:
    """Nothing here asks the network what the camera is.

    A ``scheme://`` input is ffprobe's to interpret, so a compile over one
    would open a socket -- which the unit tier has no business doing and a
    bare CI machine has no ffprobe for. The answer is supplied instead.
    """
    monkeypatch.setattr(
        "ffrwd.compiler.probe_path", lambda path, args=(), **kw: _live_probe()
    )


def _planned(body: str) -> ProcessPlan:
    """The plan for a COPY over `body`, compiled with nothing on disk."""
    plan = compile_all(DECLARE + f"COPY ({body}) TO 'out.mkv'", describe=_described).plan
    assert plan is not None
    return plan


# -- reading the argv -----------------------------------------------------


def _stand_in(process: SidecarProcess, reads: list[str]) -> list[str]:
    """An argv standing in for the sidecar, naming every pipe it is given."""
    argv = ["ffrwd-wasm"]
    for path in reads or ["pipe:0"]:
        argv += ["-i", path]
    return [*argv, "-m", process.module, "-o", "pipe:1"]


def _spelling(edge: StreamEdge, side: str) -> str:
    # A ref carries colons of its own, so the parts are joined by something
    # else and `_leg` reads them back.
    return f"<{edge.source}|{edge.ref}|{edge.target}|{side}>"


def _argv(plan: ProcessPlan) -> dict[str, list[str]]:
    return plan_argv(plan, sidecar_argv=_stand_in, pipe_path=_spelling)


def _opens(argv: list[str]) -> list[str]:
    """The pipes this argv opens, in ``-i`` order."""
    return [
        word
        for before, word in zip(argv, argv[1:])
        if before == "-i" and _is_pipe(word)
    ]


def _writes(argv: list[str]) -> list[str]:
    """The pipes this argv writes, in output order."""
    return [
        word
        for before, word in zip(argv, argv[1:])
        if before != "-i" and _is_pipe(word)
    ]


def _is_pipe(word: str) -> bool:
    return word.startswith("<") and word.endswith(">")


def _leg(spelling: str) -> tuple[str, str]:
    """The ``(source, target)`` a pipe spelling names."""
    source, _, target, _ = spelling.strip("<>").split("|")
    return source, target


# -- the three shapes that wedged -----------------------------------------

# Both legs of one merge, written the two ways round, and the second written
# through a CTE. Every one of them means the same picture.
SHAPES = {
    "direct leg first": (
        f"SELECT hstack(c.video[1], invert(c.video[1])) FROM input('{CAMERA}') c"
    ),
    "module leg first": (
        f"SELECT hstack(invert(c.video[1]), c.video[1]) FROM input('{CAMERA}') c"
    ),
    "module leg first, through a CTE": (
        "WITH m AS (SELECT invert(c.video[1]) AS v, c.video[1] AS raw "
        f"FROM input('{CAMERA}') c) SELECT hstack(m.v, m.raw) FROM m"
    ),
    "an audio leg as well": (
        "SELECT hstack(invert(c.video[1]), c.video[1]), c.audio[1] "
        f"FROM input('{CAMERA}') c"
    ),
    "the audio leg written first": (
        "SELECT c.audio[1], hstack(invert(c.video[1]), c.video[1]) "
        f"FROM input('{CAMERA}') c"
    ),
}


@pytest.mark.parametrize("body", SHAPES.values(), ids=list(SHAPES))
def test_the_camera_shapes_start(body: str) -> None:
    """Nothing in any of them waits on something that waits on it."""
    assert stalled(_planned(body)) is None


@pytest.mark.parametrize("body", SHAPES.values(), ids=list(SHAPES))
def test_the_merge_opens_the_camera_leg_before_the_module_leg(body: str) -> None:
    """The argv says so: the reader's own pipe is the merge's first ``-i``.

    The module cannot write a header before it has frames, and the frames are
    the reader's to hand over -- so a merge that opened the module's pipe
    first would sit in that open while the reader filled the pipe nobody was
    taking and stopped.
    """
    plan = _planned(body)
    argv = _argv(plan)
    merge = _opens(argv["ffmpeg0"])
    assert [_leg(pipe)[0] for pipe in merge][0] == "ffmpeg1"
    assert "sidecar0" in [_leg(pipe)[0] for pipe in merge]


@pytest.mark.parametrize("body", SHAPES.values(), ids=list(SHAPES))
def test_the_reader_writes_the_module_leg_before_the_merge_leg(body: str) -> None:
    """The module's feed is the reader's first output, whatever the SELECT said.

    The sidecar is the one consumer that drains its pipes from the moment it
    exists; the merge drains nothing until the module has spoken. A reader
    that put the merge's leg first would fill it -- the depth the compiler
    counted is finite, and a camera fills it while a module is still loading
    -- and stop before ever reaching the leg the module was waiting on.
    """
    plan = _planned(body)
    argv = _argv(plan)
    written = [_leg(pipe)[1] for pipe in _writes(argv["ffmpeg1"])]
    assert written.index("sidecar0") < written.index("ffmpeg0")


def test_the_merge_leg_still_carries_the_depth_the_compiler_counted() -> None:
    """The counted depth stays on the merge-bound edge, as slack.

    It is what lets the reader run ahead while the module works -- headroom
    for the run, not part of the reason the plan can start.
    """
    plan = _planned(SHAPES["direct leg first"])
    merge_bound = next(
        e
        for e in plan.stream_edges
        if (e.source, e.target) == ("ffmpeg1", "ffmpeg0")
    )
    assert merge_bound.bound == 1
    assert merge_bound.buffer is not None


def test_every_spelling_of_the_query_compiles_to_the_same_pipes() -> None:
    """A reordered SELECT list and a CTE reach one plan, not two.

    This is the property the original defect broke: three of these wedged and
    one streamed, and what separated them was only which argument of the
    merge had been written first.
    """
    orders = {
        name: (
            [_leg(pipe) for pipe in _opens(_argv(_planned(body))["ffmpeg0"])],
            [_leg(pipe) for pipe in _writes(_argv(_planned(body))["ffmpeg1"])],
        )
        for name, body in SHAPES.items()
        if "audio" not in name
    }
    assert len(set(map(str, orders.values()))) == 1


def test_the_query_still_says_which_way_round_the_merge_reads() -> None:
    """Reordering the pipes never reorders the picture: the labels follow."""
    direct = " ".join(_argv(_planned(SHAPES["direct leg first"]))["ffmpeg0"])
    module = " ".join(_argv(_planned(SHAPES["module leg first"]))["ffmpeg0"])
    assert "[0:v:0][1:v:0]hstack" in direct
    assert "[1:v:0][0:v:0]hstack" in module


# -- the relation itself --------------------------------------------------


def _out(ref: str, type_: StreamType = "video") -> Output:
    return Output(ref=ref, type=type_, name=None, metadata={})


def _merge_graph(module_first: bool) -> Graph:
    """One camera, split two ways: a module on one leg, the picture on the other."""
    g = Graph(input_paths=[CAMERA], sources={"a": 0})
    g.nodes["sp"] = Node(
        id="sp",
        filter="split",
        args={"n": 2},
        inputs=["src:a:v:0"],
        outputs=["video", "video"],
    )
    g.nodes["e0"] = Node(
        id="e0", filter="invert", args={}, inputs=["sp:0"], outputs=["video"]
    )
    g.nodes["n0"] = Node(
        id="n0",
        filter="hstack",
        args={},
        inputs=["e0", "sp:1"] if module_first else ["sp:1", "e0"],
        outputs=["video"],
    )
    g.sinks = [SinkUnit(outputs=[_out("n0")], path="out.mp4")]
    return g


def _merged(module_first: bool) -> ProcessPlan:
    return partition(
        _merge_graph(module_first),
        external=external_ids("e0"),
        probes={"a": _live_probe()},
        pix_fmts={"invert": "rgba"},
        shapes={"invert": ModuleShape()},
        anchors={"a": (7, 14)},
    )


def test_the_relation_names_every_step_of_every_process() -> None:
    """One open per pipe read, one head and one write per pipe written."""
    waits = relation(_merged(module_first=False))
    assert sorted(key for key in waits if key[1] == "ffmpeg1") == [
        ("head", "ffmpeg1", 0),
        ("head", "ffmpeg1", 1),
        ("run", "ffmpeg1", 0),
        ("write", "ffmpeg1", 0),
        ("write", "ffmpeg1", 1),
    ]
    # The reader opens no pipe of its own, so it is running from the start.
    assert waits[("run", "ffmpeg1", 0)] == ()


def test_a_merge_opens_its_pipes_in_order() -> None:
    """Input 1's header is not taken until input 0's has arrived."""
    waits = relation(_merged(module_first=False))
    assert ("open", "ffmpeg0", 0) in waits[("open", "ffmpeg0", 1)]


def test_a_module_takes_every_pipe_at_once() -> None:
    """A sidecar spawns a reader per input, so its opens do not queue."""
    plan = _merged(module_first=False)
    waits = relation(plan)
    opens = [key for key in waits if key[0] == "open" and key[1] == "sidecar0"]
    assert all(
        ("open", "sidecar0", key[2] - 1) not in waits[key] for key in opens
    )


def test_the_shape_that_cannot_start_is_read_as_a_cycle() -> None:
    """The merge opening the module's pipe first, and why nothing moves.

    The module has nothing to say until the reader reaches its second output;
    the reader cannot, because its first output goes to the merge, which is
    not reading anything until the module has spoken.
    """
    wedged = _wedged(_merged(module_first=False))
    cycle = stalled(wedged)
    assert cycle is not None
    assert ("run", "ffmpeg0", 0) in cycle
    assert ("write", "ffmpeg1", 1) in cycle


def test_arranging_that_plan_is_what_unwedges_it() -> None:
    wedged = _wedged(_merged(module_first=False))
    assert stalled(wedged) is not None
    assert stalled(arrange(wedged)) is None


def test_a_plan_that_already_starts_is_left_alone() -> None:
    plan = _merged(module_first=False)
    assert arrange(plan).to_dict() == plan.to_dict()


def _wedged(plan: ProcessPlan) -> ProcessPlan:
    """`plan` with its pipes back in the order node numbering used to give them.

    The module's edge ahead of the reader's own, and the depth the compiler
    counted taken off that one -- which together are the plan that could not
    start. Nothing the compiler builds is shaped this way any more, so the
    shape is written out here rather than compiled.
    """
    order = {
        ("sidecar0", "ffmpeg0"): 0,  # the merge opens the module's pipe first
        ("ffmpeg1", "ffmpeg0"): 1,  # and the reader writes the merge's first
        ("ffmpeg1", "sidecar0"): 2,
    }
    edges = sorted(
        (replace(edge, bound=0, buffer=None) for edge in plan.stream_edges),
        key=lambda edge: order[(edge.source, edge.target)],
    )
    return ProcessPlan(processes=plan.processes, edges=tuple(edges))


# -- what no order carries ------------------------------------------------


def test_a_plan_no_order_starts_is_refused_by_name() -> None:
    """The refusal every plan is checked against, on a plan that earns it."""
    with pytest.raises(FfrwdError) as caught:
        check(_wedged(_merged(module_first=False)))
    error = caught.value
    assert error.code is ErrorCode.STARTUP_DEADLOCK
    assert "each waiting on the next" in error.message
    assert error.hint is not None


def test_the_refusal_names_the_processes_that_wait_on_each_other() -> None:
    message = _refusal().message
    assert "ffmpeg1 waits to write output 1" in message
    assert "ffmpeg0 waits for every input it reads" in message
    assert message.endswith(" again")


def test_the_refusal_names_the_module_a_process_hosts() -> None:
    assert "the module 'invert'" in _refusal().message


def _refusal() -> FfrwdError:
    with pytest.raises(FfrwdError) as caught:
        check(_wedged(_merged(module_first=False)))
    return caught.value


def test_a_plan_that_starts_is_not_refused() -> None:
    check(_merged(module_first=True))


@pytest.mark.parametrize("body", SHAPES.values(), ids=list(SHAPES))
def test_no_camera_shape_is_refused(body: str) -> None:
    check(_planned(body))
