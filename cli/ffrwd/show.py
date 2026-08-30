"""The display output ``run --show`` adds, and the ffplay that reads it.

A shown output is a SECOND output on the same ffmpeg command: the file's own
video and audio maps again, raw NUT on the process's stdout, with an ffplay of
its own reading that pipe. Raw because the display path should cost no
encoder -- the file output already pays for one, and under ``--show-only``
there is no file output at all, which is what makes that mode the cheap live
path.

The display unit is added BEFORE the split pass, so a pad two outputs now read
gets its ``split`` the ordinary way. ffmpeg numbers output streams per file,
so the display group's codecs are its own and touch nothing the file writes.

One window per ffmpeg command: a command has one stdout, so the FIRST
video-bearing output of each is the one shown.

A process plan shows the same way. Its terminal ffmpeg reads NUT on stdin and
writes a file, which leaves its stdout free for exactly this; the streams
already crossing between processes travel on pipes of their own.
"""

from __future__ import annotations

from dataclasses import replace

from .ir import PIPE, Graph, Output, SinkUnit
from .processes import (
    FfmpegProcess,
    Process,
    ProcessPlan,
    RowsEdge,
    SidecarProcess,
    StreamEdge,
)
from .split import insert_splits

__all__ = [
    "DISPLAY_PATH",
    "ffplay_argv",
    "shown_path",
    "shown_processes",
    "suppressed",
    "with_display",
    "with_plan_display",
]

# ffmpeg's spelling for "this process's stdout".
DISPLAY_PATH = "pipe:1"

# What the display output carries. rawvideo/pcm_s16le so neither end encodes;
# NUT because it frames both in one stream and ffplay reads it from a pipe.
_VIDEO_CODEC = "rawvideo"
_AUDIO_CODEC = "pcm_s16le"
_CONTAINER = "nut"

# Only these reach a window: NUT carries them raw, and ffplay renders them.
_SHOWN_TYPES = frozenset({"video", "audio"})


def _is_file(unit: SinkUnit) -> bool:
    """Whether this unit writes a real destination rather than a plan's pipe."""
    return unit.path is not None and unit.path != PIPE


def _has_video(unit: SinkUnit) -> bool:
    return any(output.type == "video" for output in unit.outputs)


def shown_path(g: Graph) -> str | None:
    """The destination this graph's window stands for, or None if it has none.

    The first video-bearing output file, which is the one whose streams the
    display output repeats.
    """
    for unit in g.sinks:
        if _is_file(unit) and _has_video(unit):
            return unit.path
    return None


def _display_unit(unit: SinkUnit) -> SinkUnit:
    """`unit`'s video and audio streams again, as raw NUT on stdout.

    Its maps are the same FrameRefs, which is what gives the split pass a
    second consumer to fan out. Tags, chapters, attachments and the file's own
    options are left behind: they describe a document, and this is a window.
    """
    return SinkUnit(
        outputs=[
            Output(ref=output.ref, type=output.type, name=output.name, metadata={})
            for output in unit.outputs
            if output.type in _SHOWN_TYPES
        ],
        path=DISPLAY_PATH,
        options={
            "video_codec": _VIDEO_CODEC,
            "audio_codec": _AUDIO_CODEC,
            "format": _CONTAINER,
        },
        window=unit.window,
    )


def suppressed(g: Graph) -> Graph:
    """`g` with the files it writes dropped, and its plan pipes kept.

    What ``--show-only`` leaves behind: a pipe sink is a wire between two
    processes, not a destination, and suppressing it would cut the graph.
    """
    return replace(g, sinks=[unit for unit in g.sinks if not _is_file(unit)])


def with_display(g: Graph, *, only: bool = False) -> Graph:
    """`g` with a display output on stdout, or `g` itself if it has none to show.

    `only` suppresses every file this graph writes, leaving the display output
    alone -- nothing reaches disk, and no encoder runs. Otherwise the display
    output joins the files, which keep writing exactly what they would have.

    Pure. Run the split pass over the result: the display unit re-reads pads
    the file output already reads.
    """
    shown = next((unit for unit in g.sinks if _is_file(unit) and _has_video(unit)), None)
    if shown is None:
        # Nothing to show. Under `only` that still means nothing is written:
        # a graph left with no sink emits no command at all.
        return suppressed(g) if only else g
    display = _display_unit(shown)
    return replace(g, sinks=[*suppressed(g).sinks, display] if only else [*g.sinks, display])


def shown_processes(plan: ProcessPlan) -> dict[str, str]:
    """Each ffmpeg process of `plan` a window stands for, id to destination.

    One window per command holds here too: a process has one stdout, so the
    first video-bearing output file of each is the one shown.
    """
    found: dict[str, str] = {}
    for process in plan.ffmpeg:
        path = shown_path(process.graph)
        if path is not None:
            found[process.id] = path
    return found


def with_plan_display(plan: ProcessPlan, *, only: bool = False) -> ProcessPlan:
    """`plan` with a display output on the stdout of every process that shows one.

    Each ffmpeg process goes through :func:`with_display` and the split pass,
    exactly as a standalone graph does. The pipes the plan already wires are
    untouched: they are sinks and inputs of their own, and a process writing
    one has no file to show.

    `only` suppresses the files, and a process then left with nothing to write
    is dropped along with whatever fed it and nothing else.

    Pure: returns a new plan and never mutates `plan`.
    """
    processes = tuple(
        replace(p, graph=insert_splits(with_display(p.graph, only=only)))
        if isinstance(p, FfmpegProcess)
        else p
        for p in plan.processes
    )
    shown = replace(plan, processes=processes)
    return _without_idle(shown) if only else shown


def _writes(process: Process) -> bool:
    """Whether `process` still has a destination of its own to write.

    A display output counts: it is what the window reads. A pipe does not --
    that is a wire to another process, which is judged on its own.
    """
    if isinstance(process, SidecarProcess):
        if process.sink:  # a sink module's effects are its destination
            return True
        return process.rows is not None and bool(process.rows.path)
    return any(_is_file(unit) for unit in process.graph.sinks)


def _without_idle(plan: ProcessPlan) -> ProcessPlan:
    """`plan` without the processes suppressing its files left nothing to do.

    A process with no destination writes nothing and is not run; one that fed
    only such a process now produces frames nobody reads, and goes too, until
    nothing more can be dropped. Only whole processes go: a process hands its
    frames to at most one other, so no survivor is left holding a pipe sink
    whose reader has gone.
    """
    live = {p.id for p in plan.processes}
    pipe_edges: tuple[StreamEdge | RowsEdge, ...] = (
        *plan.stream_edges,
        *plan.rows_edges,
    )
    while True:
        idle = {
            p.id
            for p in plan.processes
            if p.id in live
            and not _writes(p)
            and not any(e.source == p.id and e.target in live for e in pipe_edges)
        }
        if not idle:
            break
        live -= idle
    return ProcessPlan(
        processes=tuple(p for p in plan.processes if p.id in live),
        edges=tuple(e for e in plan.edges if e.source in live and e.target in live),
    )


def ffplay_argv(ffplay: str, title: str) -> list[str]:
    """The ffplay that reads one display output off its stdin.

    Titled with the destination it stands for, so several windows are told
    apart. ``-fflags nobuffer`` keeps the window near the encode rather than
    a buffer behind it.
    """
    return [
        ffplay,
        "-hide_banner",
        "-loglevel",
        "error",
        "-window_title",
        title,
        "-fflags",
        "nobuffer",
        "-f",
        _CONTAINER,
        "-",
    ]
