"""Emit pass: IR ``Graph`` -> ffmpeg ``-filter_complex`` string and argv.

This is pass 4 of the compiler (see "Architecture" in ffrwd-project.md).
It assumes the graph has already been through the split pass, so every pad
has exactly one consumer.

FrameRef grammar consumed here (authoritative source: ``ir.py``)::

    "<node-id>"          -> output pad 0 of that node
    "<node-id>:<p>"      -> output pad p of that node
    "src:<alias>:v:<k>"  -> the k-th (0-based) video stream of the input bound
                            to <alias>; rendered as "[<index>:v:<k>]"
    "src:<alias>:a:<k>"  -> same for audio; rendered as "[<index>:a:<k>]"
    "src:<alias>:s:<k>"  -> same for subtitle; rendered as "<index>:s:<k>"
    "src:<alias>:d:<k>"  -> same for data; rendered as "<index>:d:<k>"

Subtitle and data refs are passthrough-only: they only ever appear
as sink-unit outputs and are only ever rendered as bare ``-map`` targets,
never as filtergraph labels. Repeating one is legal ffmpeg, so they are exempt
from the consume-once check (:func:`_check_fanout`).

Output groups
-------------
``Graph.sinks`` is one :class:`~ffrwd.ir.SinkUnit` per output FILE, and
:attr:`Emitted.groups` mirrors it one-for-one. The whole command is a single
ffmpeg invocation: the inputs are rendered once, the ``-filter_complex`` once,
and then each group contributes its own ``-map`` block, per-stream options,
sink options and path, in that order — ffmpeg's own native multi-output form.

Two scopes matter here and they are NOT the same:

* output STREAM indices restart at 0 in every output file, so ``-c:<i>`` and
  ``-metadata:s:<i>`` are indexed within a group;
* filtergraph LABELS are graph-scoped — one filter_complex serves the whole
  command — so ``out<i>`` is indexed over ``Graph.outputs``, the union of
  every group's outputs, and stays unique across the command.

Each ``Output`` becomes an :class:`OutputMap`:

* **passthrough** -- ``Output.ref`` is a source ref with zero node consumers.
  The stream never enters the filtergraph: the map target is the bare ffmpeg
  stream spec (``"0:a:1"``) and ``copy`` is True, so ``build_ffmpeg_args``
  adds ``-c:<i> copy``.
* **filtered** -- ``Output.ref`` names a node pad. That pad's label is
  ``out<i>``, where ``i`` is the output's index in ``Graph.outputs``, and the
  map target is ``"[out<i>]"``. Labels are always indexed, even for a
  one-column SELECT.

A graph whose outputs are all passthrough has NO nodes and therefore an empty
``filter_complex``; ``build_ffmpeg_args`` omits ``-filter_complex`` entirely
in that case.

Command sequences
-----------------
A compile is a SEQUENCE of ffmpeg commands: :func:`build_ffmpeg_commands` is
the entry point and returns a list of argv lists, one entry for every query
except a ``two_pass`` sink, which returns two. :func:`build_ffmpeg_args`
renders the single command and stays the seam every other caller uses; it
refuses a ``two_pass`` :class:`Emitted` rather than silently dropping a pass.

A ``ffrwd.loudnorm2`` node returns two as well, for a different reason:
its two commands share every input and every label but NOT the filtergraph.
:func:`emit` therefore renders the chains twice, once per phase, and keeps
the measuring one in :attr:`Emitted.measure_filter_complex` (empty when the
graph has no such node). Pass 1 is that filtergraph, the FILTERED maps only,
and ``-f null -``: it measures, encodes nothing, and carries none of the
sink's options -- a codec on the measuring pass would be work thrown away.
Pass 2 is the ordinary command, with ``${FFRWD_LN_*}`` references sitting
in its filtergraph where the measurements go (see ``ffrwd.loudnorm``).

Process plans
-------------
:func:`build_process_args` renders one ffmpeg process of a
:class:`~ffrwd.processes.ProcessPlan`. Such a process is an ordinary graph
except that some of its inputs and sinks are ``pipe:`` -- the plan's stream
edges -- and those take the spelling and wire format the plan assigned. The
rest of the rendering, filtergraph and all, is what
:func:`build_ffmpeg_args` does.

``two_pass`` renders as ``-pass <n> -passlogfile <dest>``, where ``<dest>`` is
the resolved destination path of the WRITING pass (so ffmpeg's own stats file
lands beside the output as ``<dest>-0.log``, and is left there — ffmpeg owns
its temp files). Pass 1 is the same inputs, the same ``-filter_complex`` and
the same options, writing ``-f null -`` with a reduced ``-map`` list: video
outputs, plus any FILTERED output of another type. Dropping a passthrough
audio/subtitle map costs nothing, but a filtergraph pad with no consumer is a
hard ffmpeg error ("has output N unconnected"), so a filtered pad is kept and
encoded into the null muxer. Per-file stream indices are recomputed from that
reduced list, so ``-c:<i>`` still names the right stream in both passes.

Input seeking
-------------
``Graph.input_trims`` maps an alias to a ``(start, end)`` window, either bound
possibly ``None`` (an open-ended ``WHERE <alias>.t >= x`` / ``<= y``). Since an
alias owns exactly one ``-i`` slot, emit resolves it against ``Graph.sources``
into :attr:`Emitted.input_trims`, a list parallel to :attr:`Emitted.inputs`,
and ``build_ffmpeg_args`` renders ``-ss <start>`` and/or ``-to <end>``
immediately BEFORE the owning ``-i`` — an input option, so the demuxer seeks
and every stream of that input is cut coherently, subtitle and data included —
omitting whichever half is ``None`` (a tail-only window has no ``-to``, so
ffmpeg reads to EOF). Both bounds are on the input's own timeline. A seeked
input can still be stream-copied, and then the cut snaps back to the preceding
keyframe; a decoded stream is frame-accurate.

Output seeking
--------------
``SinkUnit.window`` is the other half: one FILE's own ``(start, end)``,
rendered as ``-ss``/``-to`` ahead of that output's ``-map`` list rather than
before an ``-i``. One input can then feed several differently-cut files from a
single decode, which is what a fan-out over chapters compiles to. Output-side
seeking re-encodes, so a group carrying a window never renders ``-c:<i> copy``:
its passthrough streams take whatever codec the sink options name, else
ffmpeg's default encoder for the container.

Input options
-------------
``Graph.input_options`` maps an alias to its validated ``input('path', name =>
value, ...)`` options, resolved like ``input_trims`` into
:attr:`Emitted.input_options`, parallel to :attr:`Emitted.inputs`.
``build_ffmpeg_args`` renders them immediately before that input's ``-i`` and
BEFORE any ``-ss``/``-to``. ffmpeg input options are order-INSENSITIVE among
themselves, but this fixed order -- options, seek flags, ``-i`` -- is what
ffrwd always emits; measured against ffmpeg 7.1, ``-loop 1 -framerate 15
-to 2 -i frame.png`` runs correctly. A ``False`` bool (``loop => false``) is
never rendered, same as a sink option's False bool. A ``bare`` spec
(``realtime`` -> ``-re``) renders the flag alone with no value, only when
True. ``seek_end`` renders its value NEGATED (``-sseof -60`` for
``seek_end => 60``).

An option name resolves through ``ffrwd.inputs.option_spec``, not the
user-facing table alone: lower mints inputs of its own
(``ffrwd.empty_captions()``, whose ``data:`` URI needs ``-f webvtt``) and
carries their flags the same way, whether or not the name (like ``format``)
also has a user-facing entry in ``INPUT_OPTIONS``.

Pad label scheme
----------------
Every node output pad gets exactly one label, derived from the node id
(sanitized: any character outside ``[A-Za-z0-9_]`` becomes ``_``):

* single-output node  -> ``[<id>]``            e.g. ``[n2]``
* multi-output node   -> ``[<id><p>]``         e.g. a ``split`` node whose
  id is ``n1_split`` and whose ``outputs`` list has two entries produces
  ``[n1_split0][n1_split1]``, so a full chain reads
  ``[n1]split=2[n1_split0][n1_split1]``
* a pad named by ``Graph.outputs[i]`` -> ``[out<i>]``, whichever node/pad it is

The out-pad count of a node is ``len(node.outputs)`` -- no filter is special
cased. ``out0..out<n-1>`` are reserved up front, and any label collision is
broken by appending ``_`` until the label is unique. Labels of pads that are
consumed inside a merged comma-chain are never rendered (that is what chain
merging means).

Chain merging
-------------
Nodes are walked in graph (topological) order. A node extends the chain built
so far when it is the *sole* consumer of the immediately preceding node's only
output pad and has no other inputs; such runs are joined with ``,`` and only
the head's input labels and the tail's output labels are rendered. Chains are
joined with ``;`` (no whitespace). The README example therefore emits::

    [0:v:0]crop=w=600:h=200:x=1200:y=50,scale=w=iw*0.5:h=-2[n2];[0:v:0][n2]overlay=x=20:y=20[out0]

(both source refs read ``[0:v:0]``: the README's two aliases name the same
untrimmed ``game.mp4``, so input dedup, below, folds them onto one ``-i``.)

``emit(g, network=True)`` renders the subset a module network reads instead:
every node is its own chain, so nothing merges and every label is written
out, and a source ref drops its per-type index (``[0:v]``) since a network
input carries one stream. Labels and escaping are the same in both.

Input dedup
-----------
Two ``INPUT`` aliases over the SAME path, with the SAME (validated,
normalized) ``input_options`` and NO ``input_trims`` window on either, share a
single ``-i``: :func:`~ffrwd.ir.dedup_inputs` runs first in :func:`emit` and folds such
aliases' ``Graph.sources`` entries onto one ``input_paths`` slot (first
occurrence wins the position). A trimmed alias is NEVER merged, even against
an identical window -- the concat splice (two aliases, two DIFFERENT windows,
same file; cookbook recipe 17) needs two ``-i``'s to seek independently, and
this pass does not try to tell "two windows that happen to match" apart from
"two windows that must stay apart".

It is a POST-LOWER normalization: trims are only known once lower has resolved
every ``WHERE <alias>.t`` window into ``Graph.input_trims``, so the "no trim
window" test cannot run earlier. It re-keys ``sources``/``input_paths`` only
-- exactly the resolution :func:`_src_spec` does per source ref -- so no
FrameRef, node or label changes. The pre-dedup graph (``ffrwd explain``)
still shows one input slot per alias; only the rendered ``-i`` list and the
indices baked into the filtergraph are deduped.

Zero-input nodes
----------------
A generated source (``FROM ffmpeg.testsrc(duration => 2) t``) lowers to a node
with ``inputs=[]``. It renders as a chain head with no input labels
(``testsrc=duration=2[out0]``); :func:`_extends` refuses to comma-append any
node that does not take exactly one input, so a source always STARTS a chain
and never gets swallowed into the previous one, while what FOLLOWS it merges
normally (``sine=frequency=440,volume=volume=0.5[out0]``).
:func:`_verify_topological` and :func:`_check_fanout` both iterate
``node.inputs``, so an empty list contributes nothing. A graph of only sources
has no ``-i`` at all: ``ffmpeg -filter_complex ... -map [out0] out.mp4`` is a
complete, valid command.

Argument rendering
------------------
``args`` renders in dict insertion order as ``filter=k1=v1:k2=v2``. Special
cases: a filter with no args renders bare (``hflip``, no ``=``); the key
``"expr"`` renders value-only (``setpts=PTS-STARTPTS``); every arg of a
``split``/``asplit`` node renders value-only (``split=2``). ``concat`` needs
nothing special: ``{"n": 2, "v": 1, "a": 0}`` -> ``concat=n=2:v=1:a=0``.

Escaping
--------
All filter values go through :func:`_escape_value` -- the single place where
ffmpeg filtergraph escaping happens (drawtext ``text=`` included). Metadata
values do NOT: they are passed as argv words, not parsed as a filtergraph.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field, replace

from . import loudnorm
from .errors import ErrorCode, FfrwdError
from .inputs import render_options
from .ir import (
    Attachment,
    FrameRef,
    Graph,
    Node,
    Output,
    SinkUnit,
    StreamType,
    dedup_inputs,
    is_src,
    src_parts,
)
from .processes import (
    COPY_CODEC,
    FIFO,
    FIFO_FORMAT,
    PIPE,
    QUEUE_SIZE,
    AudioFormat,
    EdgeBuffer,
    StreamFormat,
)
from .sink import (
    CODEC_PARAMS_FLAGS,
    PASSLOGFILE_FLAG,
    SINK_OPTIONS,
    SinkOptionSpec,
    copy_suppressed_scopes,
    option_spec,
)

OUTPUT_LABEL_PREFIX = "out"
"""Filtered outputs are labelled ``out0``, ``out1``, ... (without brackets)."""

_SPLIT_FILTERS = frozenset({"split", "asplit"})
_VALUE_ONLY_KEYS = frozenset({"expr"})

_TYPE_MARKERS: dict[StreamType, str] = {
    "video": "v",
    "audio": "a",
    "subtitle": "s",
    "data": "d",
}

# Stream types no filtergraph can carry: they are only ever bare
# `-map`s, so they are exempt from the consume-once rule below.
_PASSTHROUGH_ONLY: frozenset[StreamType] = frozenset({"subtitle", "data"})

# Level 1 (filter-option) escaping: these characters would otherwise separate
# options / quote inside a single filter's argument list.
_LEVEL1_ESCAPES = {
    "\\": "\\\\",
    "'": "\\'",
    ":": "\\:",
    "=": "\\=",
}

# Level 2 (filtergraph) escaping: applied on top of level 1, so the backslashes
# introduced above are themselves escaped -- exactly the composition described
# in ffmpeg's "Notes on filtergraph escaping".
_LEVEL2_SPECIAL = re.compile(r"[\\'\[\],;#\s]")

_LABEL_UNSAFE = re.compile(r"[^A-Za-z0-9_]")

_TWO_PASS = "two_pass"
_FORMAT = "format"
# Pass 1 analyses and throws the muxed result away.
_ANALYSIS_PATH = "-"
_ANALYSIS_FORMAT = "null"
# The chapter list is an output COLUMN, not a sink option, so its flag is not
# table data. It names the input the chapters come from and renders last.
MAP_CHAPTERS_FLAG = "-map_chapters"
# What an audio wire edge is conformed to for the module reading it. No query
# writes either, so their flags are not table data either.
SAMPLE_RATE_FLAG = "-ar"
CHANNELS_FLAG = "-ac"

# The input a file's GLOBAL tags are copied from, or -1 for none. Rendered
# ahead of `-metadata`, which ffmpeg applies after it whatever the argv order.
MAP_METADATA_FLAG = "-map_metadata"
# One attached file. An output option: it applies to the output file that
# follows it, and adds one output stream after that file's mapped ones.
ATTACH_FLAG = "-attach"


@dataclass
class OutputMap:
    """One ``-map`` argument: a filtered pad label or a raw stream spec."""

    target: str  # "[out0]" (filtered) or "0:a:1" (passthrough)
    type: StreamType
    copy: bool  # passthrough -> True -> -c:<i> copy
    metadata: dict[str, str]
    # The flags the query asserted, or None where it asserted none.
    disposition: tuple[str, ...] | None = None


@dataclass
class OutputGroup:
    """One output FILE of the command: its maps, its path, its options.

    Mirrors one :class:`~ffrwd.ir.SinkUnit`. `maps` is that file's
    ``-map`` list in SELECT order, and its INDEX is the ffmpeg output stream
    index — per file, since ffmpeg restarts numbering at every output — so
    ``-c:<i>``/``-metadata:s:<i>`` are computed from it and nothing wider.
    `path` is None only for a bare-SELECT graph, where the caller supplies the
    destination. `options` are the validated ``WITH (...)`` sink options, and
    `tags` that file's container tags (a None value clears its key).
    `window` is this file's own ``(start, end)`` trim, rendered as ``-ss``/
    ``-to`` ahead of the maps; it re-encodes, so a group carrying one never
    renders ``-c:<i> copy``. `chapters` is the input index the file's chapter
    list comes from (``ir.NO_CHAPTERS`` for none), rendered last as
    ``-map_chapters``; None leaves ffmpeg's default alone. `metadata` is the
    input the file's global tags are copied from (``ir.NO_METADATA`` for
    none), rendered as ``-map_metadata``; None leaves ffmpeg's default alone.
    `attachments` are
    the files this one carries, rendered as ``-attach`` ahead of the maps;
    each takes an output stream index after every map of this group.
    `wire` is the stream format set when this group writes a pipe edge rather
    than a file; its codec and pixel format ride `options`, and what an audio
    edge is conformed to is rendered off it.
    """

    maps: list[OutputMap]
    path: str | None = None
    options: dict[str, object] = field(default_factory=dict)
    tags: dict[str, str | None] = field(default_factory=dict)
    window: tuple[float | None, float | None] | None = None
    chapters: int | None = None
    metadata: int | None = None
    attachments: list[Attachment] = field(default_factory=list)
    wire: StreamFormat | None = None


@dataclass
class Emitted:
    inputs: list[str]  # file paths in -i order
    filter_complex: str  # "" when every output is a passthrough
    groups: list[OutputGroup]  # one per Graph.sinks entry, same order
    # Entry `i` is the (start, end) window of the `-i` at index `i`, or None if
    # that input is not seeked; either half of a window may itself be None
    # (open-ended). :func:`emit` always fills this parallel to `inputs`. The
    # empty default (hand-built Emitted objects) means the same: nothing seeked.
    input_trims: list[tuple[float | None, float | None] | None] = field(default_factory=list)
    # Entry `i` is the validated, insertion-ordered options of the `-i` at
    # index `i`, {} if it set none. Same convention as `input_trims`.
    input_options: list[dict[str, object]] = field(default_factory=list)
    # The same graph rendered in loudnorm2's MEASURING phase, or "" when the
    # graph has no loudnorm2 node. Non-empty means this compile is a sequence
    # (`build_ffmpeg_commands`), same as a two_pass group does.
    measure_filter_complex: str = ""

    @property
    def maps(self) -> list[OutputMap]:
        """Every group's maps concatenated, in command order (read-only).

        A convenience view of the whole command's ``-map`` list. Its index is
        an ffmpeg output STREAM index only when there is exactly one group:
        ffmpeg numbers output streams per FILE, which is why
        ``build_ffmpeg_args`` walks ``groups`` and never this.
        """
        return [mapping for group in self.groups for mapping in group.maps]


def emit(g: Graph, *, network: bool = False) -> Emitted:
    """Render `g` as an ffmpeg filtergraph plus its output map list.

    Raises ``FfrwdError(INTERNAL)`` on a malformed graph: a cycle or
    non-topological node ordering, a dangling FrameRef, no outputs, or a pad
    with more than one consumer -- which means the split pass did not run. An
    ``Output`` counts as a consumer whichever group it sits in, so a pad
    feeding both a node and an output, or two outputs naming one pad, is a
    split-pass bug.

    `network` renders the grammar a module NETWORK reads instead: one node per
    chain, joined by ``;``, so every label is written out, and an input named
    by stream type alone (``[0:v]``) rather than by its per-type index. A
    label there may be read by several nodes -- the network hands one module's
    frames to every reader, where ffmpeg needs a ``split`` -- so the
    consume-once rule does not apply. Labels are allocated and values escaped
    identically either way.

    Runs :func:`~ffrwd.ir.dedup_inputs` first, before anything below reads
    ``g.sources``/``g.input_paths``.
    """
    g = dedup_inputs(g)
    _verify_topological(g)

    nodes = list(g.nodes.values())
    pads = {node.id: _out_pad_count(node) for node in nodes}
    if not network:
        _check_fanout(nodes, g)
    labels = _assign_labels(nodes, pads, g.outputs)

    chains = [[node] for node in nodes] if network else _build_chains(nodes, pads)
    filter_complex = _render_chains(
        chains, g, pads, labels, measure=False, network=network
    )
    two_phase = any(node.filter == loudnorm.FILTER for node in nodes)

    groups = [_output_group(g, unit, labels) for unit in g.sinks]

    return Emitted(
        inputs=list(g.input_paths),
        filter_complex=filter_complex,
        groups=groups,
        input_trims=_input_windows(g),
        input_options=_input_option_list(g),
        measure_filter_complex=(
            _render_chains(chains, g, pads, labels, measure=True, network=network)
            if two_phase
            else ""
        ),
    )


def _output_group(g: Graph, unit: SinkUnit, labels: dict[str, str]) -> OutputGroup:
    return OutputGroup(
        maps=[_output_map(g, output, labels) for output in unit.outputs],
        path=unit.path,
        options=dict(unit.options),
        tags=dict(unit.tags),
        window=unit.window,
        chapters=unit.chapters,
        metadata=unit.metadata,
        attachments=list(unit.attachments),
    )


def _input_windows(g: Graph) -> list[tuple[float | None, float | None] | None]:
    """``g.input_trims`` (alias-keyed) resolved into a per-``-i`` list.

    Each alias's window applies to whichever input slot ``g.sources`` sends it
    to; entry `i` is that input's ``(start, end)``, or None. ``_dedup_inputs``
    never merges a trimmed alias's slot, so two aliases sharing a slot here
    can only come from a hand-built ``Graph``, and are tolerated only when
    their windows agree exactly -- a real mismatch is the lower/split bug this
    guards against.
    """
    windows: list[tuple[float | None, float | None] | None] = [None] * len(g.input_paths)
    for alias, window in g.input_trims.items():
        index = g.sources.get(alias)
        if index is None:
            raise _internal(f"input trim names unknown source alias {alias!r}")
        if not 0 <= index < len(g.input_paths):
            raise _internal(
                f"input trim on alias {alias!r} maps to out-of-range input index {index}"
            )
        previous = windows[index]
        if previous is not None and previous != window:
            raise _internal(
                f"input {index} carries two different trim windows, {previous} and "
                f"{window}; each alias must own its own -i (lower/split bug)"
            )
        windows[index] = window
    return windows


def _input_option_list(g: Graph) -> list[dict[str, object]]:
    """``g.input_options`` (alias-keyed) resolved into a per-``-i`` list.

    Same resolution ``_input_windows`` does for trims: entry `i` is that
    input's insertion-ordered options dict, empty if it set none. Two aliases
    sharing a slot (what dedup produces) is fine exactly when their option
    sets agree; a real mismatch is a lower/split bug, as in `_input_windows`.
    """
    options: list[dict[str, object]] = [{} for _ in g.input_paths]
    seen: list[bool] = [False for _ in g.input_paths]
    for alias, alias_options in g.input_options.items():
        index = g.sources.get(alias)
        if index is None:
            raise _internal(f"input options name unknown source alias {alias!r}")
        if not 0 <= index < len(g.input_paths):
            raise _internal(
                f"input options on alias {alias!r} map to out-of-range input index {index}"
            )
        candidate = dict(alias_options)
        if seen[index] and options[index] != candidate:
            raise _internal(
                f"input {index} carries two different option sets; each alias "
                "must own its own -i (lower/split bug)"
            )
        options[index] = candidate
        seen[index] = True
    return options


@dataclass(frozen=True)
class _Pass:
    """Which half of a two-pass encode is being rendered, and its stats file."""

    number: int
    logfile: str


def _two_pass_groups(e: Emitted) -> list[OutputGroup]:
    return [group for group in e.groups if group.options.get(_TWO_PASS) is True]


def build_ffmpeg_commands(e: Emitted, out_path: str | None = None) -> list[list[str]]:
    """Every ffmpeg command `e` runs, in order — the compile's real shape.

    One entry for an ordinary query (:func:`build_ffmpeg_args` verbatim), two
    for a ``two_pass`` sink: the video-only analysis pass, then the write. A
    caller runs them in order and stops at the first nonzero exit.

    ``two_pass`` is single-sink only (lower rejects it in a multi-COPY
    script), so a multi-group `e` carrying it is a hand-built object and
    raises ``ValueError``, as does one with no destination to derive the
    ``-passlogfile`` path from. Path precedence is
    :func:`build_ffmpeg_args`'s: `out_path`, else the group's own.

    A ``loudnorm2`` `e` is the other sequence: the measuring pass, then the
    write. Lower rejects it together with ``two_pass``, so an `e` carrying
    both is hand-built and raises ``ValueError``.
    """
    if e.measure_filter_complex:
        if _two_pass_groups(e):
            raise ValueError("loudnorm2 and two_pass are two different command sequences")
        return [
            _render_command(_measure_emitted(e), _ANALYSIS_PATH, None),
            _render_command(e, out_path, None),
        ]
    two_pass = _two_pass_groups(e)
    if not two_pass:
        return [_render_command(e, out_path, None)]
    if len(e.groups) > 1:
        raise ValueError(
            f"two_pass writes one file, but this command writes {len(e.groups)}"
        )
    dest = out_path if out_path is not None else e.groups[0].path
    if dest is None:
        raise ValueError("no output path given and the query has no sink")
    return [
        _render_command(_analysis_emitted(e), _ANALYSIS_PATH, _Pass(1, dest)),
        _render_command(e, out_path, _Pass(2, dest)),
    ]


def _measure_emitted(e: Emitted) -> Emitted:
    """`e` reduced to loudnorm2's measuring pass.

    One group holding every FILTERED map -- the measured stream itself plus
    any other pad, since a filtergraph output with no consumer is a hard
    ffmpeg error -- muxed to ``-f null -``. Passthrough maps are dropped
    (copying a stream teaches the measurement nothing) and so is every sink
    option: this pass writes no file.
    """
    group = OutputGroup(
        maps=[mapping for mapping in e.maps if not mapping.copy],
        path=None,
        options={_FORMAT: _ANALYSIS_FORMAT},
    )
    return replace(e, filter_complex=e.measure_filter_complex, groups=[group])


def _analysis_emitted(e: Emitted) -> Emitted:
    """`e` reduced to what pass 1 encodes: see "Command sequences" above."""
    group = e.groups[0]
    options = {name: value for name, value in group.options.items() if name != _FORMAT}
    options[_FORMAT] = _ANALYSIS_FORMAT
    analysis = OutputGroup(
        maps=[m for m in group.maps if m.type == "video" or not m.copy],
        path=group.path,
        options=options,
    )
    return replace(e, groups=[analysis])


def build_ffmpeg_args(e: Emitted, out_path: str | None = None) -> list[str]:
    """Full ffmpeg argv for `e`: one invocation, one output file per group.

    The single-command seam. A ``two_pass`` or ``loudnorm2`` `e` compiles to a
    SEQUENCE and is refused here with ``ValueError``;
    :func:`build_ffmpeg_commands` renders it.

    Path precedence per group: `out_path`, else that group's own path, else
    ``ValueError``. `out_path` OVERRIDES, so it is legal only for a
    single-group `e` — overriding several files with one path would write them
    on top of each other — and a multi-group `e` with `out_path` set raises.
    A library/programmer contract, not something user SQL can trigger: the
    CLI never passes `out_path` at all, so every sink writes its own path.

    The SELECT list is authoritative: exactly one ``-map`` per
    :class:`OutputMap`, in order, with ``-c:<i> copy`` for passthrough streams
    and ``-metadata:s:<i> k=v`` (keys sorted) for provenance metadata, then
    ``-disposition:<i> <flags>`` for a stream the query flagged (ffmpeg's own
    ``+``-joined spec, ``0`` for none). Those ``<i>`` are PER GROUP — ffmpeg restarts output
    stream numbering at each file — so group 2's first map is ``-c:0``, not
    ``-c:<n>``.
    ``-filter_complex`` is rendered once for the command, omitted when the
    graph is pure passthrough.

    A group's own ``window`` renders ``-ss``/``-to`` FIRST, ahead of its maps,
    and suppresses every ``-c:<i> copy`` in that group (see "Output seeking").

    Per-``-i`` flags come first in a fixed order (see the module docstring's
    "Input options"): `e.input_options` flags (``-loop 1``, ``-framerate 15``)
    FIRST, then `e.input_trims`' ``-ss <start>`` and/or ``-to <end>``, then
    the ``-i``. Short or empty `input_options`/`input_trims` lists leave the
    remaining inputs plain, so a hand-built :class:`Emitted` needs neither.

    Sink options render after each group's -map/-metadata block, in
    insertion order, purely from ``ffrwd.sink.SINK_OPTIONS`` table data --
    no option-specific logic here, with three derived exceptions: a ``bare``
    spec (``shortest`` -> ``-shortest``) renders the flag alone with no
    value, only when True; ``codec_params``'s flag has a ``{codec}``
    placeholder filled from THIS GROUP's own ``video_codec`` value via
    ``ffrwd.sink.CODEC_PARAMS_FLAGS`` (``libx264`` -> ``-x264-params``);
    and ``two_pass`` renders a pair of flags whose values come from the pass
    being rendered (see "Command sequences").
    ``per_stream=True`` specs render ``f"{flag}:{i}"`` for every output index
    `i` of THAT GROUP whose type matches the spec's scope; ``per_stream=False``
    renders ``flag`` once. A False bool is never rendered (``faststart false``
    emits nothing).

    Copy suppression: a group whose options set a codec option (``flag ==
    "-c"``) scoped to video drops ``-c:<i> copy`` on its video passthrough
    outputs, since the explicit codec re-encodes them; same for audio and, as
    the rule reads the table's ``scope`` rather than naming types, for
    ``subtitle_codec`` (webvtt -> mov_text on a passthrough subtitle
    output). Per group, so one file may re-encode what another copies.
    """
    reason = None
    if _two_pass_groups(e):
        reason = _TWO_PASS
    elif e.measure_filter_complex:
        reason = loudnorm.FILTER
    if reason is not None:
        raise ValueError(
            f"this query compiles to a sequence of commands ({reason}); "
            "call build_ffmpeg_commands"
        )
    return _render_command(e, out_path, None)


def build_process_args(
    g: Graph,
    *,
    pipe_inputs: Sequence[tuple[str, str]] = (),
    pipe_outputs: Sequence[tuple[str, StreamFormat]] = (),
    pipe_buffers: Sequence[EdgeBuffer | None] = (),
) -> list[str]:
    """Full ffmpeg argv for one ffmpeg process of a process plan.

    Everything :func:`build_ffmpeg_args` renders, with one difference: the
    ``pipe:`` inputs and sinks a partitioned graph carries are the plan's
    edges, and here they take the spelling and the wire format the plan gave
    them. `pipe_inputs` is one ``(spelling, container)`` per ``pipe:`` entry
    of ``g.input_paths``, in ``-i`` order; `pipe_outputs` one ``(spelling,
    format)`` per ``pipe:`` sink, in sink order. A spelling is
    ``pipe:0``/``pipe:1`` for an edge stdio carries, and the named pipe's own
    path otherwise. An input the graph already gave a container of its own --
    a subtitle track a module writes -- keeps it. `pipe_buffers` is parallel to
    `pipe_outputs`: the depth the plan gave each of those edges, which puts the
    fifo muxer in front of the real one where the depth is held there. A short
    or empty list leaves the remaining edges plain.

    An input renders ``-f <container> -i <spelling>``. An output renders the
    wire's codec for each stream it carries, its pixel format when it carries
    video, the rate and channel count it is conformed to when it carries audio
    a module constrains, ``-f <container>``, and `spelling` as the destination
    -- so the reader finds every parameter in the NUT header.

    Input paths are substituted BEFORE the graph is emitted: two ``pipe:``
    inputs are two different streams, and only their real spellings keep
    :func:`~ffrwd.ir.dedup_inputs` from folding them onto one ``-i``.
    """
    slots = [index for index, path in enumerate(g.input_paths) if path == PIPE]
    if len(slots) != len(pipe_inputs):
        raise _internal(
            f"process reads {len(slots)} pipes but {len(pipe_inputs)} were wired"
        )
    paths = list(g.input_paths)
    options = {alias: dict(values) for alias, values in g.input_options.items()}
    for slot, (spelling, container) in zip(slots, pipe_inputs):
        paths[slot] = spelling
        for alias, index in g.sources.items():
            if index == slot:
                options[alias] = {_FORMAT: container, **options.get(alias, {})}

    e = emit(replace(g, input_paths=paths, input_options=options))

    pipes = [index for index, group in enumerate(e.groups) if group.path == PIPE]
    if len(pipes) != len(pipe_outputs):
        raise _internal(
            f"process writes {len(pipes)} pipes but {len(pipe_outputs)} were wired"
        )
    groups = list(e.groups)
    for slot, (index, (spelling, wire)) in enumerate(zip(pipes, pipe_outputs)):
        group = groups[index]
        buffer = pipe_buffers[slot] if slot < len(pipe_buffers) else None
        groups[index] = replace(
            group,
            path=spelling,
            wire=wire,
            options={**_wire_options(wire, buffer), **group.options},
        )
    return build_ffmpeg_args(replace(e, groups=groups))


def build_network_graph(
    g: Graph, *, pipe_inputs: Sequence[str] = ()
) -> tuple[str, list[str]]:
    """One sidecar region's filtergraph string and its ``-map`` targets.

    The sidecar hosts a network of modules configured the way ffmpeg is: a
    ``-filter_complex`` string over module NAMES, and one ``-map`` target per
    output. Both come from :func:`emit`, so a region's labels are allocated
    and its argument values escaped by exactly the rules an ffmpeg process's
    are. Targets come back in sink order.

    `pipe_inputs` is the spelling of each ``pipe:`` input in ``-i`` order,
    substituted before the graph is emitted for the same reason
    :func:`build_process_args` substitutes them: two pipes are two streams,
    and only their real spellings keep :func:`~ffrwd.ir.dedup_inputs` from
    folding them onto one ``-i``.
    """
    slots = [index for index, path in enumerate(g.input_paths) if path == PIPE]
    if len(slots) != len(pipe_inputs):
        raise _internal(
            f"network reads {len(slots)} pipes but {len(pipe_inputs)} were wired"
        )
    paths = list(g.input_paths)
    for slot, spelling in zip(slots, pipe_inputs):
        paths[slot] = spelling
    e = emit(replace(g, input_paths=paths), network=True)
    return e.filter_complex, [m.target for group in e.groups for m in group.maps]


def _render_conformance(group: OutputGroup) -> list[str]:
    """``-ar:<i>``/``-ac:<i>`` for each audio stream a wire edge constrains.

    Rendered per audio map of the group, the way a per-stream sink option is,
    and ahead of the codec that follows it. A module naming neither a rate nor
    a channel count renders nothing and the stream is left alone.
    """
    wire = group.wire
    if not isinstance(wire, AudioFormat):
        return []
    args: list[str] = []
    for index, mapping in enumerate(group.maps):
        if mapping.type != "audio":
            continue
        if wire.required_rate is not None:
            args += [f"{SAMPLE_RATE_FLAG}:{index}", str(wire.required_rate)]
        if wire.required_channels is not None:
            args += [f"{CHANNELS_FLAG}:{index}", str(wire.required_channels)]
    return args


def _wire_options(wire: StreamFormat, buffer: EdgeBuffer | None = None) -> dict[str, object]:
    """The sink options that write one stream edge's wire format.

    An encoded edge -- the one into a packet sink -- carries its encoder
    options too, validated at lowering, rendered exactly as a file sink's.

    A `buffer` on the fifo road puts the fifo muxer in front of the real one:
    ``-f fifo`` with the wire's own container as ``-fifo_format`` and a queue
    the edge's bound sized. The pipe road is the named pipe's own buffer and
    renders nothing here.
    """
    if isinstance(wire, AudioFormat):
        written: dict[str, object] = {"audio_codec": wire.codec}
    elif wire.codec == COPY_CODEC:
        # A copied stream keeps the pixel format it was encoded with; naming
        # one would be an instruction to a decoder that never runs.
        written = {"video_codec": wire.codec, **dict(wire.options)}
    else:
        written = {
            "video_codec": wire.codec,
            "pix_fmt": wire.pix_fmt,
            **dict(wire.options),
        }
    if buffer is not None and buffer.road == FIFO:
        written[FIFO_FORMAT] = wire.container
        written[QUEUE_SIZE] = buffer.packets
        written[_FORMAT] = FIFO
    else:
        written[_FORMAT] = wire.container
    return written


def _render_command(e: Emitted, out_path: str | None, pass_: _Pass | None) -> list[str]:
    """One ffmpeg invocation's argv. `pass_` is set only for a two-pass half."""
    if not e.groups:
        raise ValueError("no output path given and the query has no sink")
    if out_path is not None and len(e.groups) > 1:
        raise ValueError(
            f"out_path overrides a single output file, but this command writes "
            f"{len(e.groups)}"
        )

    args = ["ffmpeg"]
    for index, input_path in enumerate(e.inputs):
        options = e.input_options[index] if index < len(e.input_options) else {}
        args += _render_input_options(options)
        window = e.input_trims[index] if index < len(e.input_trims) else None
        if window is not None:
            start, end = window
            if start is not None:
                args += ["-ss", _render_number(start)]
            if end is not None:
                args += ["-to", _render_number(end)]
        args += ["-i", input_path]
    if e.filter_complex:
        args += ["-filter_complex", e.filter_complex]
    for group in e.groups:
        path = out_path if out_path is not None else group.path
        if path is None:
            raise ValueError("no output path given and the query has no sink")
        suppressed = copy_suppressed_scopes(group.options)
        # This file's own seeks, ahead of its maps: an output-side window
        # re-encodes everything it covers, which is why no map below forces
        # `copy` under one.
        if group.window is not None:
            start, end = group.window
            if start is not None:
                args += ["-ss", _render_number(start)]
            if end is not None:
                args += ["-to", _render_number(end)]
        for attachment in group.attachments:
            args += [ATTACH_FLAG, attachment.path]
        # Output stream indices restart at 0 in every output file.
        for index, mapping in enumerate(group.maps):
            args += ["-map", mapping.target]
            if mapping.copy and mapping.type not in suppressed and group.window is None:
                args += [f"-c:{index}", "copy"]
            for key in sorted(mapping.metadata):
                args += [f"-metadata:s:{index}", f"{key}={mapping.metadata[key]}"]
            if mapping.disposition is not None:
                args += [
                    f"-disposition:{index}",
                    "+".join(mapping.disposition) or "0",
                ]
        args += _render_attachment_tags(group.attachments, len(group.maps))
        # Ahead of the sink options. ffmpeg applies `-metadata` after
        # `-map_metadata` whatever the argv order, so a named key wins over
        # whatever the map was copied from either way.
        if group.metadata is not None:
            args += [MAP_METADATA_FLAG, str(group.metadata)]
        args += _render_container_tags(group.tags)
        args += _render_conformance(group)
        args += _render_sink_options(group, pass_)
        if group.chapters is not None:
            args += [MAP_CHAPTERS_FLAG, str(group.chapters)]
        args.append(path)
    return args


# input option rendering


def _render_input_options(options: dict[str, object]) -> list[str]:
    """This input's options as argv, through the renderer the probe shares."""
    try:
        return render_options(options)
    except ValueError as err:  # defensive: lower validated every name it wrote
        raise _internal(f"{err} reached emit") from err


def _render_attachment_tags(attachments: list[Attachment], maps: int) -> list[str]:
    """``-metadata:s:<i>`` for each attached file, after this group's maps.

    ffmpeg appends an attachment to the output as a stream of its own, in
    ``-attach`` order and after every mapped stream, so the index is the map
    count plus the attachment's position. A None tag is left unset: ffmpeg
    then names the attachment after the file's basename and guesses its type.
    """
    args: list[str] = []
    for offset, attachment in enumerate(attachments):
        index = maps + offset
        # mimetype first, the order ffmpeg's own -attach example writes.
        tags = (("mimetype", attachment.mimetype), ("filename", attachment.filename))
        for key, value in tags:
            if value is not None:
                args += [f"-metadata:s:{index}", f"{key}={value}"]
    return args


# container tag rendering


def _render_container_tags(tags: dict[str, str | None]) -> list[str]:
    """One file's container tags as ``-metadata key=value``, keys sorted.

    A None value renders ``-metadata key=``, which is how ffmpeg is told to
    clear a key it would otherwise copy from the input.
    """
    args: list[str] = []
    for key in sorted(tags):
        value = tags[key]
        args += ["-metadata", f"{key}={'' if value is None else value}"]
    return args


# sink option rendering


def _render_option_value(spec: SinkOptionSpec, value: object) -> str | None:
    """Render one option's value per its spec, or None to omit it entirely.

    A bool value of False is always omitted (e.g. plain ``faststart false``);
    every other value renders via ``spec.value_template.format(v=value)``.
    """
    if spec.type == "bool" and value is not True:
        return None
    return spec.value_template.format(v=value)


def _codec_params_flag(options: dict[str, object]) -> str:
    """The derived ``-<codec>-params`` flag for THIS group's ``video_codec``.

    Defensive only: ``lower`` already rejected ``codec_params`` set without a
    ``video_codec`` in :data:`ffrwd.sink.CODEC_PARAMS_FLAGS`, so a miss here
    means that check was bypassed (hand-built ``Emitted``/graph).
    """
    codec = options.get("video_codec")
    codec_key = CODEC_PARAMS_FLAGS.get(codec) if isinstance(codec, str) else None
    if codec_key is None:
        raise _internal(f"codec_params has no matching video_codec (got {codec!r})")
    return SINK_OPTIONS["codec_params"].flag.format(codec=codec_key)


def _render_two_pass(value: object, pass_: _Pass | None) -> list[str]:
    """``two_pass true`` -> ``-pass <n> -passlogfile <dest>``; False -> nothing.

    The pass number and the stats-file path are properties of the COMMAND, not
    of the option's value, which is why this is the second derived exception
    to "flag is static table data" (``codec_params`` is the first).
    """
    if value is not True:
        return []
    if pass_ is None:  # defensive: build_ffmpeg_args refuses a two_pass Emitted
        raise _internal("two_pass reached emit with no pass number")
    return [SINK_OPTIONS[_TWO_PASS].flag, str(pass_.number), PASSLOGFILE_FLAG, pass_.logfile]


def _render_sink_options(group: OutputGroup, pass_: _Pass | None = None) -> list[str]:
    args: list[str] = []
    for name, value in group.options.items():
        spec = option_spec(name)
        if spec is None:
            raise _internal(f"unknown sink option {name!r}")
        if name == _TWO_PASS:
            args += _render_two_pass(value, pass_)
            continue
        if spec.bare:
            if value is True:
                args.append(spec.flag)
            continue
        rendered = _render_option_value(spec, value)
        if rendered is None:
            continue
        flag = _codec_params_flag(group.options) if name == "codec_params" else spec.flag
        if spec.per_stream:
            # Per-FILE stream indices, same as the -c/-metadata block above.
            for index, mapping in enumerate(group.maps):
                if mapping.type == spec.scope:
                    args += [f"{flag}:{index}", rendered]
        else:
            args += [flag, rendered]
    return args


# escaping


def _escape_value(s: str) -> str:
    """Escape `s` for use as a filter option value inside a filtergraph.

    A filtergraph description is unescaped twice: once when the graph is
    split into filters (``[ ] , ;`` separate, ``\\`` escapes, ``'`` quotes),
    then again when a filter's option string is split into ``k=v`` pairs
    (``:`` separates, ``\\`` escapes, ``'`` quotes). This applies both levels
    in that order, matching the worked example in ffmpeg's "Notes on
    filtergraph escaping": ``:`` becomes ``\\\\:``, ``'`` becomes ``\\\\\\'``,
    ``,`` becomes ``\\,``.

    Whitespace and ``#`` are backslash-escaped too: ffmpeg trims unquoted
    leading/trailing whitespace from tokens, and ``#`` can start a comment.
    Values made only of ordinary characters (alphanumerics, ``. - + * / % @``,
    expression punctuation like ``( ) < >``) contain nothing special at either
    level and therefore pass through unchanged and unquoted.
    """
    if not s:
        return "''"
    level1 = "".join(_LEVEL1_ESCAPES.get(char, char) for char in s)
    return _LEVEL2_SPECIAL.sub(lambda m: "\\" + m.group(0), level1)


# refs, labels, validation


def _internal(message: str) -> FfrwdError:
    return FfrwdError(
        ErrorCode.INTERNAL,
        message,
        hint="this is a compiler bug; please report the query that produced it",
    )


def _parse_node_ref(ref: FrameRef) -> tuple[str, int]:
    """Split a non-source FrameRef into ``(node_id, pad)``."""
    node_id, sep, pad_text = ref.partition(":")
    if not sep:
        return node_id, 0
    try:
        pad = int(pad_text)
    except ValueError:
        raise _internal(f"malformed frame ref {ref!r}: pad index is not an integer") from None
    if pad < 0:
        raise _internal(f"malformed frame ref {ref!r}: negative pad index")
    return node_id, pad


def _parse_src_ref(ref: FrameRef) -> tuple[str, StreamType, int]:
    try:
        alias, stream_type, index = src_parts(ref)
    except ValueError as exc:
        raise _internal(f"malformed source ref {ref!r}: {exc}") from None
    if index < 0:
        raise _internal(f"malformed source ref {ref!r}: negative stream index")
    return alias, stream_type, index


def _slot(ref: FrameRef) -> str:
    """Canonical key for the pad a FrameRef points at.

    ``"n1"`` and ``"n1:0"`` are the same pad, so both map to ``"n1:0"``;
    source refs are already canonical and keep their ``"src:<alias>:v:<k>"``
    form.
    """
    if is_src(ref):
        return ref
    node_id, pad = _parse_node_ref(ref)
    return f"{node_id}:{pad}"


def _out_pad_count(node: Node) -> int:
    count = len(node.outputs)
    if count < 1:
        raise _internal(f"node {node.id!r} declares no output pads")
    return count


def _pad_type(g: Graph, ref: FrameRef) -> StreamType:
    """Stream type of the pad `ref` points at (src marker or node.outputs[pad])."""
    if is_src(ref):
        return _parse_src_ref(ref)[1]
    node_id, pad = _parse_node_ref(ref)
    node = g.nodes.get(node_id)
    if node is None:
        raise _internal(f"ref {ref!r} names unknown node {node_id!r}")
    return node.outputs[pad]


def _src_spec(g: Graph, ref: FrameRef) -> str:
    """Render a source ref as an ffmpeg stream spec, e.g. ``0:a:1``."""
    alias, stream_type, index = _parse_src_ref(ref)
    if alias not in g.sources:
        raise _internal(f"source ref {ref!r} names unknown source alias {alias!r}")
    return f"{g.sources[alias]}:{_TYPE_MARKERS[stream_type]}:{index}"


def _verify_topological(g: Graph) -> None:
    """Check every ref resolves and points backwards; a cycle cannot pass."""
    defined: set[str] = set()
    for node_id, node in g.nodes.items():
        if node.id != node_id:
            raise _internal(f"node key {node_id!r} does not match node id {node.id!r}")
        for ref in node.inputs:
            _check_ref(g, ref, defined, f"node {node.id!r}")
        defined.add(node_id)
    if not g.outputs:
        raise _internal("graph has no outputs")
    for index, output in enumerate(g.outputs):
        if not output.ref:
            raise _internal(f"graph output {index} has an empty ref")
        _check_ref(g, output.ref, defined, f"graph output {index}")


def _check_ref(g: Graph, ref: FrameRef, defined: set[str], where: str) -> None:
    if is_src(ref):
        alias, _stream_type, _index = _parse_src_ref(ref)
        if alias not in g.sources:
            raise _internal(f"{where} references unknown source alias {alias!r}")
        input_index = g.sources[alias]
        if not 0 <= input_index < len(g.input_paths):
            raise _internal(
                f"source alias {alias!r} maps to out-of-range input index {input_index}"
            )
        return
    node_id, pad = _parse_node_ref(ref)
    if node_id not in g.nodes:
        raise _internal(f"{where} references unknown node {node_id!r}")
    if node_id not in defined:
        raise _internal(
            f"{where} references {node_id!r}, which is not defined earlier: "
            "the node graph is cyclic or not in topological order"
        )
    if pad >= _out_pad_count(g.nodes[node_id]):
        raise _internal(f"{where} references pad {pad} of {node_id!r}, which has fewer outputs")


def _check_fanout(nodes: list[Node], g: Graph) -> None:
    """Every pad has at most one consumer, except the ones that are not pads.

    Two exemptions, both about refs that are bare ``-map``s rather than
    filtergraph pads — and both mirrored in the split pass, which is what
    leaves them fanned out on purpose:

    * a subtitle/data source ref never enters the filtergraph, and repeating
      one is legal ffmpeg (``SELECT f.subtitle[1], f.subtitle[1]`` writes the
      same track twice). There is no split filter for such a stream at all.
    * a video/audio SOURCE ref that no node consumes and that each
      output FILE maps at most once. Two output groups bare-mapping the same
      input stream is a repeated ``-map``, which is equally legal, and it is
      what lets both files stream-copy that track. A repeat within ONE group,
      or any pad a filter also consumes, is still a split-pass bug.

    Real filtergraph pads (``"<node-id>[:<pad>]"``) are consume-once in every
    direction, including across groups: two sinks reading one view's pad must
    have been split.
    """
    counts = _count_consumers(nodes, g.outputs)
    exempt = _exempt_refs(g)
    for slot, count in counts.items():
        if count > 1 and slot not in exempt:
            raise _internal(
                f"pad {slot!r} has {count} consumers; ffmpeg pads are consume-once "
                "(the split pass must run before emit)"
            )


def _is_passthrough_only(ref: FrameRef) -> bool:
    """True for a subtitle/data source ref (a bare -map, never a pad)."""
    if not is_src(ref):
        return False
    try:
        return src_parts(ref)[1] in _PASSTHROUGH_ONLY
    except ValueError:  # malformed: _parse_src_ref reports it properly later
        return False


def _exempt_refs(g: Graph) -> set[FrameRef]:
    """Source refs allowed to keep more than one consumer (see `_check_fanout`).

    Deliberately the same rule ``ffrwd.split._exempt_refs`` applies, stated
    independently: emit never imports the split pass, it CHECKS its output.
    """
    exempt: set[FrameRef] = set()
    filtered: set[FrameRef] = {
        ref for node in g.nodes.values() for ref in node.inputs if is_src(ref)
    }
    for unit in g.sinks:
        seen: dict[FrameRef, int] = {}
        for output in unit.outputs:
            if is_src(output.ref):
                seen[output.ref] = seen.get(output.ref, 0) + 1
        for ref, count in seen.items():
            if _is_passthrough_only(ref):
                exempt.add(ref)
            elif count == 1 and ref not in filtered:
                exempt.add(ref)
            else:
                exempt.discard(ref)
                filtered.add(ref)
    return exempt


def _count_consumers(nodes: list[Node], outputs: list[Output]) -> dict[str, int]:
    """Count consumers per pad. Every sink unit's Output is a consumer too."""
    counts: dict[str, int] = {}
    for node in nodes:
        for ref in node.inputs:
            slot = _slot(ref)
            counts[slot] = counts.get(slot, 0) + 1
    for output in outputs:
        slot = _slot(output.ref)
        counts[slot] = counts.get(slot, 0) + 1
    return counts


def _sanitize_label(node_id: str) -> str:
    label = _LABEL_UNSAFE.sub("_", node_id)
    return label or "pad"


def _assign_labels(
    nodes: list[Node], pads: dict[str, int], outputs: list[Output]
) -> dict[str, str]:
    """Map every node pad slot to its rendered label (without brackets)."""
    output_labels: dict[str, str] = {}
    for index, output in enumerate(outputs):
        if is_src(output.ref):
            continue  # passthrough: never enters the filtergraph
        output_labels[_slot(output.ref)] = f"{OUTPUT_LABEL_PREFIX}{index}"

    # Reserve every out<i> up front, passthrough indices included, so a node
    # id that happens to be "out1" can never steal an output label.
    used = {f"{OUTPUT_LABEL_PREFIX}{index}" for index in range(len(outputs))}
    labels: dict[str, str] = {}
    for node in nodes:
        count = pads[node.id]
        for pad in range(count):
            slot = f"{node.id}:{pad}"
            output_label = output_labels.get(slot)
            if output_label is not None:
                labels[slot] = output_label
                continue
            base = _sanitize_label(node.id)
            if count > 1:
                base = f"{base}{pad}"
            while base in used:
                base += "_"
            used.add(base)
            labels[slot] = base
    return labels


def _output_map(g: Graph, output: Output, labels: dict[str, str]) -> OutputMap:
    produced = _pad_type(g, output.ref)
    if produced != output.type:
        raise _internal(
            f"output {output.ref!r} is declared {output.type} but the producing "
            f"pad is {produced} (lower/split bug)"
        )
    if is_src(output.ref):
        # Passthrough: zero node consumers is guaranteed by _check_fanout,
        # which counts this Output itself as the pad's single consumer.
        return OutputMap(
            target=_src_spec(g, output.ref),
            type=output.type,
            copy=True,
            metadata=dict(output.metadata),
            disposition=output.disposition,
        )
    slot = _slot(output.ref)
    label = labels.get(slot)
    if label is None:
        raise _internal(f"no pad label was assigned for {slot!r}")
    return OutputMap(
        target=f"[{label}]",
        type=output.type,
        copy=False,
        metadata=dict(output.metadata),
        disposition=output.disposition,
    )


# chain building and rendering


def _build_chains(nodes: list[Node], pads: dict[str, int]) -> list[list[Node]]:
    chains: list[list[Node]] = []
    for node in nodes:
        if chains and _extends(chains[-1][-1], node, pads):
            chains[-1].append(node)
        else:
            chains.append([node])
    return chains


def _extends(prev: Node, node: Node, pads: dict[str, int]) -> bool:
    """True if `node` can be comma-appended after `prev` in one chain.

    `node` must directly follow `prev` (guaranteed by the caller), take
    exactly one input, and that input must be `prev`'s only output pad. Pads
    are consume-once (checked in :func:`_check_fanout`), so `node` being a
    consumer of that pad already makes it the sole consumer, and rules out the
    pad also being named by a ``Graph.outputs`` entry.
    """
    if pads[prev.id] != 1:
        return False
    if len(node.inputs) != 1:
        return False
    ref = node.inputs[0]
    if is_src(ref):
        return False
    return _slot(ref) == f"{prev.id}:0"


def _render_chains(
    chains: list[list[Node]],
    g: Graph,
    pads: dict[str, int],
    labels: dict[str, str],
    *,
    measure: bool,
    network: bool = False,
) -> str:
    """The whole ``-filter_complex`` string for one loudnorm2 phase.

    `measure` only ever changes a loudnorm2 node's own arguments: the chains,
    the labels and every other filter render identically in both phases.
    """
    return ";".join(
        _render_chain(chain, g, pads, labels, measure, network) for chain in chains
    )


def _render_chain(
    chain: list[Node],
    g: Graph,
    pads: dict[str, int],
    labels: dict[str, str],
    measure: bool,
    network: bool = False,
) -> str:
    head, tail = chain[0], chain[-1]
    prefix = "".join(f"[{_input_label(g, ref, labels, network)}]" for ref in head.inputs)
    body = ",".join(_render_filter(node, measure) for node in chain)
    suffix = "".join(f"[{labels[f'{tail.id}:{pad}']}]" for pad in range(pads[tail.id]))
    return f"{prefix}{body}{suffix}"


def _input_label(
    g: Graph, ref: FrameRef, labels: dict[str, str], network: bool = False
) -> str:
    if is_src(ref):
        spec = _src_spec(g, ref)
        # A network input carries one stream, so its per-type index says
        # nothing and the subset grammar leaves it off.
        return spec.rpartition(":")[0] if network else spec
    slot = _slot(ref)
    label = labels.get(slot)
    if label is None:
        raise _internal(f"no pad label was assigned for {slot!r}")
    return label


def _render_filter(node: Node, measure: bool) -> str:
    name, args = node.filter, node.args
    if name == loudnorm.FILTER:
        # The one node whose rendered arguments depend on which command of the
        # sequence is being written.
        name = loudnorm.FFMPEG_FILTER
        args = loudnorm.phase_args(args, measure=measure)
    if not args:
        return name
    value_only_filter = name in _SPLIT_FILTERS
    parts: list[str] = []
    for key, value in args.items():
        rendered = _escape_value(_render_scalar(node, key, value))
        if value_only_filter or key in _VALUE_ONLY_KEYS:
            parts.append(rendered)
        else:
            parts.append(f"{key}={rendered}")
    return f"{name}={':'.join(parts)}"


def _render_number(value: int | float) -> str:
    """Render a numeric scalar: ``12.5`` -> ``"12.5"``, ``5`` -> ``"5"``.

    The one place numbers become argv/filtergraph text, so a filter argument
    and an ``-ss``/``-to`` seek time render by the same rule.
    """
    return str(value)


def _render_scalar(node: Node, key: str, value: object) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int | float):
        return _render_number(value)
    if isinstance(value, str):
        return value
    raise _internal(
        f"node {node.id!r} arg {key!r} has unrenderable type {type(value).__name__}"
    )
