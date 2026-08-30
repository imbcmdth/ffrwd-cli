"""Tests for the display output `run --show` adds.

Unit tier: graphs and plans built by hand, rendered through the same emit the
run uses, and nothing spawned -- what each mode puts on the ffmpeg command
line is decided before any process exists, so it is testable without one.
Expected argv is always read off the rendered command rather than typed out.
"""

from __future__ import annotations

import pytest

from ffrwd import show
from ffrwd.emit import build_ffmpeg_args, emit
from ffrwd.execute import plan_argv
from ffrwd.ir import PIPE, Graph, Node, Output, RowsSink, SinkUnit, StreamType
from ffrwd.processes import ProcessPlan, SidecarProcess, external_ids, partition
from ffrwd.split import insert_splits


def _out(ref: str, type_: StreamType = "video") -> Output:
    return Output(ref=ref, type=type_, name=None, metadata={})


def _filtered(path: str = "out.mp4") -> Graph:
    """One video through a filter, plus a passthrough audio track."""
    g = Graph(input_paths=["a.mp4"], sources={"a": 0})
    g.nodes["n0"] = Node(
        id="n0", filter="negate", args={}, inputs=["src:a:v:0"], outputs=["video"]
    )
    g.sinks = [SinkUnit(outputs=[_out("n0"), _out("src:a:a:0", "audio")], path=path)]
    return g


def _audio_only() -> Graph:
    g = Graph(input_paths=["a.mp4"], sources={"a": 0})
    g.sinks = [SinkUnit(outputs=[_out("src:a:a:0", "audio")], path="out.m4a")]
    return g


def _rendered(g: Graph, *, only: bool) -> list[str]:
    """`g` with its display output, through the same emit a run goes through."""
    return build_ffmpeg_args(emit(insert_splits(show.with_display(g, only=only))))


# --- what each mode puts on the command line -------------------------------


def test_show_keeps_the_file_and_adds_a_display_output() -> None:
    """--show is save AND show: the file output is untouched and the display
    output is a second one, on the process's own stdout."""
    argv = _rendered(_filtered(), only=False)
    assert "out.mp4" in argv
    assert argv[-1] == show.DISPLAY_PATH
    assert argv.count("-map") == 4  # two streams, twice


def test_the_display_output_is_raw_nut() -> None:
    """Raw so the window costs no encoder: rawvideo for the picture,
    pcm_s16le so sound survives, NUT to carry both down one pipe."""
    argv = _rendered(_filtered(), only=False)
    tail = argv[argv.index("out.mp4") :]
    assert "rawvideo" in tail
    assert "pcm_s16le" in tail
    assert tail[-3:] == ["-f", "nut", show.DISPLAY_PATH]


def test_show_only_writes_no_file_and_runs_no_encoder() -> None:
    """--show-only drops the file sink entirely, which is what makes it the
    cheap live path: nothing reaches disk and no encoder is configured."""
    argv = _rendered(_filtered(), only=True)
    assert "out.mp4" not in argv
    assert "libx264" not in argv
    assert argv[-1] == show.DISPLAY_PATH
    assert argv.count("-map") == 2  # the display output's, and no others


def test_show_splits_a_pad_two_outputs_now_read() -> None:
    """A filtered pad is consume-once, so the second reader gets a split --
    the ordinary pass does it, because the display unit is added before it."""
    argv = _rendered(_filtered(), only=False)
    graph = argv[argv.index("-filter_complex") + 1]
    assert "split=2" in graph


def test_show_only_needs_no_split() -> None:
    """One consumer, one pad: nothing to fan out."""
    argv = _rendered(_filtered(), only=True)
    graph = argv[argv.index("-filter_complex") + 1]
    assert "split" not in graph


# --- which outputs get a window --------------------------------------------


def test_a_video_output_names_its_window() -> None:
    """The window is titled with the destination it stands for."""
    assert show.shown_path(_filtered("film.mkv")) == "film.mkv"


def test_an_audio_only_output_has_nothing_to_show() -> None:
    assert show.shown_path(_audio_only()) is None


def test_a_plan_pipe_is_not_a_file_to_show() -> None:
    """A sink writing a plan's pipe is a wire, not a destination."""
    g = _filtered(PIPE)
    assert show.shown_path(g) is None


def test_show_only_suppresses_an_output_it_cannot_show() -> None:
    """Nothing is written under --show-only, showable or not: a graph left
    with no sink emits no command at all."""
    assert show.with_display(_audio_only(), only=True).sinks == []


def test_show_leaves_an_unshowable_graph_alone() -> None:
    """Without `only`, a graph with nothing to show still writes its file."""
    g = _audio_only()
    assert show.with_display(g, only=False).sinks == g.sinks


# --- the player -------------------------------------------------------------


@pytest.mark.parametrize("title", ["out.mp4", "clips/ch 1.mkv"])
def test_the_player_reads_nut_off_its_stdin(title: str) -> None:
    argv = show.ffplay_argv("/usr/bin/ffplay", title)
    assert argv[0] == "/usr/bin/ffplay"
    assert argv[-3:] == ["-f", "nut", "-"]
    assert argv[argv.index("-window_title") + 1] == title


# --- a process plan ---------------------------------------------------------


def _stand_in(process: SidecarProcess) -> list[str]:
    """The argv a sidecar is spawned with, standing in for the real one."""
    return ["ffrwd-wasm", "-m", process.module, "-f", "nut", "pipe:0", "pipe:1"]


def _module(id_: str, ref: str, type_: StreamType = "video") -> Node:
    return Node(
        id=id_, filter=f"{id_}.wasm", args={}, inputs=[ref], outputs=[type_]
    )


def _chain(path: str = "out.mp4") -> ProcessPlan:
    """decode -> module -> encode, the shape a wasm query partitions into."""
    g = Graph(input_paths=["a.mp4"], sources={"a": 0})
    g.nodes["m0"] = _module("m0", "src:a:v:0")
    g.sinks = [
        SinkUnit(
            outputs=[_out("m0"), _out("src:a:a:0", "audio")],
            path=path,
            options={"video_codec": "libx264"},
        )
    ]
    return partition(g, external=external_ids("m0"))


def _two_destinations() -> ProcessPlan:
    """A video file through a module, and an audio-only file that is not shown."""
    g = Graph(input_paths=["a.mp4"], sources={"a": 0})
    g.nodes["m0"] = _module("m0", "src:a:a:0", "audio")
    g.sinks = [
        SinkUnit(outputs=[_out("src:a:v:0")], path="vid.mp4"),
        SinkUnit(outputs=[_out("m0", "audio")], path="snd.m4a"),
    ]
    return partition(g, external=external_ids("m0"))


def _with_rows() -> ProcessPlan:
    """A module filtering video and writing its rows as a document of its own."""
    g = Graph(input_paths=["a.mp4"], sources={"a": 0})
    g.nodes["m0"] = _module("m0", "src:a:v:0")
    g.sinks = [SinkUnit(outputs=[_out("m0")], path="out.mp4")]
    g.rows_sinks = {"m0": RowsSink(container="ndjson", path="rows.ndjson")}
    return partition(g, external=external_ids("m0"))


def _terminal(plan: ProcessPlan, *, only: bool) -> list[str]:
    """The argv of the process a window stands for, with its display output."""
    pid = next(iter(show.shown_processes(plan)))
    shown = show.with_plan_display(plan, only=only)
    return plan_argv(shown, sidecar_argv=_stand_in)[pid]


def test_a_shown_plan_keeps_the_file_and_adds_a_display_output() -> None:
    """The terminal ffmpeg reads its frames on stdin and its stdout is free,
    so the display output is a second output on that same command."""
    argv = _terminal(_chain(), only=False)
    assert "out.mp4" in argv
    assert argv[-1] == show.DISPLAY_PATH
    assert argv.count("-map") == 4  # two streams, twice


def test_a_shown_plan_display_output_is_raw_nut() -> None:
    argv = _terminal(_chain(), only=False)
    tail = argv[argv.index("out.mp4") :]
    assert "rawvideo" in tail
    assert "pcm_s16le" in tail
    assert tail[-3:] == ["-f", "nut", show.DISPLAY_PATH]


def test_show_only_on_a_plan_writes_no_file_and_runs_no_encoder() -> None:
    """Nothing reaches disk and no encoder is configured -- the same cheap
    live path a single command gets."""
    argv = _terminal(_chain(), only=True)
    assert "out.mp4" not in argv
    assert "libx264" not in argv
    assert argv[-1] == show.DISPLAY_PATH
    assert argv.count("-map") == 2  # the display output's, and no others


def test_only_a_plan_process_that_writes_video_shows() -> None:
    """A process writing a pipe is a wire, not a destination; the one writing
    the file is what the window stands for."""
    assert show.shown_processes(_chain("film.mkv")) == {"ffmpeg0": "film.mkv"}


def test_a_shown_plan_leaves_the_processes_around_it_alone() -> None:
    """Only the shown process changes: the pipes the plan wires are untouched,
    so the module still reads and writes exactly what it did."""
    plan = _chain()
    before = plan_argv(plan, sidecar_argv=_stand_in)
    after = plan_argv(show.with_plan_display(plan), sidecar_argv=_stand_in)
    assert {pid: argv for pid, argv in after.items() if pid != "ffmpeg0"} == {
        pid: argv for pid, argv in before.items() if pid != "ffmpeg0"
    }


def test_show_only_suppresses_a_plan_output_it_cannot_show() -> None:
    """The audio-only file is written by a process of its own, and under
    --show-only that process has nothing left to do."""
    shown = show.with_plan_display(_two_destinations(), only=True)
    argv = plan_argv(shown, sidecar_argv=_stand_in)
    assert [process.id for process in shown.processes] == ["ffmpeg0"]
    assert "snd.m4a" not in argv["ffmpeg0"]
    assert argv["ffmpeg0"][-1] == show.DISPLAY_PATH


def test_show_only_drops_what_fed_a_suppressed_output() -> None:
    """The module and the ffmpeg decoding for it produced frames only that
    file read, so they go with it."""
    plan = _two_destinations()
    assert {process.id for process in plan.processes} == {
        "ffmpeg0",
        "ffmpeg1",
        "ffmpeg2",
        "sidecar0",
    }
    shown = show.with_plan_display(plan, only=True)
    assert shown.sidecars == ()
    assert shown.edges == ()


def test_show_only_leaves_a_rows_document_alone() -> None:
    """Rows are a document the sidecar writes itself, not a file the flag
    suppresses."""
    shown = show.with_plan_display(_with_rows(), only=True)
    assert [s.rows for s in shown.sidecars] == [
        RowsSink(container="ndjson", path="rows.ndjson")
    ]


def test_a_plan_writing_only_rows_has_nothing_to_show() -> None:
    g = Graph(input_paths=["a.mp4"], sources={"a": 0})
    g.nodes["m0"] = _module("m0", "src:a:v:0")
    g.rows_sinks = {"m0": RowsSink(container="ndjson", path="rows.ndjson")}
    plan = partition(g, external=external_ids("m0"))
    assert show.shown_processes(plan) == {}
