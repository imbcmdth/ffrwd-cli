"""Lower pass: a resolved query becomes an IR :class:`~ffrwd.ir.Graph`.

This is pass 2 of the compiler (see "Architecture" in ffrwd-project.md). It
assumes :func:`ffrwd.parser.resolve` already accepted the query, so every
rejection raised here is either a check resolve deliberately left to lowering
(CTE column names, function names, argument types, probed stream bounds) or a
defensive re-check.

The top-level SELECT list IS the output stream list, and every value flowing
through lowering is a *typed* stream (``video``, ``audio``, ``subtitle`` or
``data``), never an untyped "frame".

Passthrough-only stream types
-----------------------------
``subtitle`` and ``data`` streams get the exact same surface as video/audio —
``a.subtitle[1]``, the bare array ``a.data``, a CTE column, a star expansion —
but an ffmpeg filtergraph carries video and audio only, so they may never be a
filter input. Three rejections enforce that, all keyed off ``_PASSTHROUGH_ONLY``:

* as a function argument, in EITHER tier -> ``UDF_ARG_TYPE``
  (:meth:`_Lowerer._reject_passthrough_args`);
* under a **CTE's** WHERE time range that is actually consumed in that branch ->
  ``UNSUPPORTED_SQL`` (:meth:`_Lowerer._access`). An INPUT alias's WHERE is not
  a filtergraph trim at all any more (see the WHERE bullet below), so it carries
  captions perfectly well; a CTE's window IS a filtergraph trim, so for it the
  rejection is permanent;
* as a UNION ALL branch column -> ``UNSUPPORTED_SQL``
  (:meth:`_Lowerer._check_concat_columns`; ``concat`` has ``v``/``a`` pads only).

Everything else about them is ordinary: they lower to ``"src:<alias>:s:<k>"`` /
``"src:<alias>:d:<k>"`` refs, carry provenance (a caption track's ``language``
tag rides the same passthrough metadata path an audio track's does), and become
``Output`` rows that split and emit treat as bare ``-map``s.

``SELECT *`` and ``<alias>.*``
------------------------------
A star is a column GENERATOR, not an expression: :meth:`_Lowerer._expand_star`
turns it into one passthrough column per stream. A bare ``*`` covers every FROM
alias in FROM order; ``<alias>.*`` covers one. Within an INPUT alias the order
is FILE order (probe order, all four stream types interleaved as the container
has them) and the expansion is splat tier — it needs a probe, so an unreadable
input is ``INPUT_NOT_FOUND``, the same policy a bare ``a.audio`` has. Within a
CTE it is column order, array columns splatting, and no probe is consulted at
all: the CTE's shape was fixed when its body lowered.

What lowering does, in order:

* CTE bodies lower first, in definition order, into the *same* graph. A CTE
  records a list of ``(name, type, ref)`` columns — its SELECT list — and
  ``FROM <cte>`` later exposes those columns by their ``AS`` names. A script's
  VIEWS are CTEs here: ``Resolved.ctes`` holds both, so the whole
  binding table is lowered exactly ONCE no matter how many COPYs read it.
* Then one :class:`~ffrwd.ir.SinkUnit` per ``COPY``, in script order, each
  from that COPY's own query — or, for a bare SELECT, a single unit
  with ``path=None``. Every unit shares this graph's nodes, so a view read by
  three COPYs is decoded and filtered once and fanned out by the split pass.
* Inside a branch, ``FROM`` builds a typed environment: an ``input()`` alias
  exposes per-type stream access (``a.video[1]`` -> ``"src:a:v:0"``; SQL
  subscripts are 1-based, IR indices 0-based), a CTE alias exposes its
  recorded columns (under its own name, or under a branch-local alias:
  ``FROM master m``), and a ``ffmpeg.<source>(...)`` alias exposes exactly one
  statically-typed stream (see below).
* ``WHERE <alias>.t BETWEEN x AND y`` records a per-alias time range; where
  that window lands depends on what the alias is:

  - an INPUT alias owns its own ``-i`` slot and has at most one window in the
    whole query, so the window is recorded as ``Graph.input_trims[alias]`` and
    emit renders it as ``-ss <x> -to <y>`` in front of that ``-i``. NO filter
    node is spliced: the stream refs come out of lowering untouched, so a
    trimmed column that nothing else filters stays a PASSTHROUGH and is
    stream-copied. The seek applies to the WHOLE input — every stream of that
    alias, including subtitle/data streams and streams the SELECT list never
    mentions (harmless: an unselected stream is never ``-map``ped) — which is
    exactly what makes a trimmed caption track possible. Accuracy: decoded
    (filtered/re-encoded) streams are frame-accurate; stream-copied ones snap
    back to the preceding keyframe and may start up to a GOP early.
  - a CTE alias names a filtergraph pad, not an input, so its window still
    lowers to a filter trim: spliced lazily, the first time a stream of that
    CTE is consumed, and memoized per stream, so every consumer of the same
    stream shares one ``trim``+``setpts`` (video) / ``atrim``+``asetpts``
    (audio) pair. Being a filtergraph trim, it cannot carry captions.
  - under a fan-out ``TO (<expression>)`` an input alias's window is per-FILE
    rather than per-``-i``: the rows name different windows over one input, so
    each lands on its own ``SinkUnit.window`` and emit seeks that OUTPUT. The
    exception is a fan-out that stream-copies everything it maps, where an
    output seek would write a corrupt file: that one goes back to one graph
    (one command) per file, each with its own ``Graph.input_trims``.
* Each projection lowers bottom-up to one :class:`~ffrwd.ir.Output` per
  stream it carries (an array column splats into consecutive Outputs). A call
  type-checks its stream arguments against the filter's pad signature and its
  option arguments against that filter's introspected AVOptions (see "One
  calling convention" below).

Generated sources: ``FROM ffmpeg.<source>(...) a``
--------------------------------------------------
A source alias is the third kind of binding (:class:`_SourceBinding`), and
it is the registry surface in TABLE position: the name resolves through
``Registry.get_source`` alone (never ``get``), and its options through the
same ``Registry.options`` path a call's named arguments take, with the same
two codes.

What makes it different from an ``input()`` alias is that there is no FILE:

* no ``-i``, so no input index — a source appears in neither
  ``Graph.input_paths`` nor ``Graph.sources``, and ``compile_sql`` never
  probes it (it probes ``Resolved.sources``, which a source alias is not in);
* it lowers to a ZERO-INPUT node, ``Node(filter=<source>, args=<options>,
  inputs=[], outputs=[<type>])``, minted lazily on first column access and
  memoized on the binding, so fan-out is the split pass's ordinary business
  and never a second generator;
* one output pad means one stream of one statically-known type, so every
  column rule is answered without a probe: ``a.video[1]`` on a
  video source, ``a.audio[1]`` on an audio one, a bare ``a.video``/``a.audio``
  that is an array of LENGTH 1, ``a.*`` = that one column, and
  ``STREAM_NOT_FOUND`` (naming the source and what it produces) for the other
  type or any subscript but ``[1]``;
* ``WHERE a.t`` is rejected: nothing was read, so there is no timeline to
  seek — a source's length is its own ``duration =>`` option;
* provenance is always empty, for the same reason (nothing probed).

Everything else is ordinary. A source is legal in a CTE body and in a UNION
ALL branch — silent-audio-for-concat, ``SELECT t.video[1], s.audio[1] FROM
ffmpeg.testsrc2(...) t, ffmpeg.anullsrc(...) s`` as the second branch of a
concat, is the motivating case — and the node it builds is one split, emit
and the goldens cannot tell apart from any other.

One calling convention
----------------------
Every call is an ffmpeg filter, spelled the way ffmpeg's own filtergraph
syntax spells it::

    <name>(<stream inputs...>, <positional options...>, <named options...>)

There is no curated stdlib and no tier system: a name resolves in the
``registry`` — the filter set of the ffmpeg on PATH — and nowhere else. What
compiles therefore depends on what that ffmpeg reports, and an empty registry
(no ffmpeg) means every call name is simply UNKNOWN, not an INTERNAL error.

* STREAM INPUTS come first, count and types straight from the pad signature
  (``gblur`` is ``V->V``, ``xfade`` is ``VV->V``). A count or type mismatch
  against that signature is ``UDF_ARG_TYPE`` — the code's whole remaining job.
* POSITIONAL OPTIONS follow, binding to the filter's options in REGISTRY
  ORDER, which is ffmpeg's AVOption declaration order and therefore exactly
  the order ``gblur=5:2`` binds in a filtergraph (see
  ``ffrwd/registry.py``'s docstring for why the deduped list is that order).
  ``crop(f, 100, 50, 10, 20)``
  is ``crop=out_w=100:out_h=50:x=10:y=20``; ``scale(f, 640, 480)`` is
  ``scale=width=640:height=480``. A positional binds AS the option it lands
  on and is validated as that option — same type/range/enum checks, same two
  codes — so option problems are uniformly ``UNKNOWN_FILTER_OPTION`` /
  ``FILTER_OPTION_TYPE`` whether the option was written positionally or by
  name. More positionals than the filter has options is ``UDF_ARG_TYPE``,
  naming that count.
* NAMED OPTIONS (``sigma => 5``) come last. Mixing rules: a positional after
  a named is ``UNSUPPORTED_SQL`` (:func:`_split_args`, resolve's rule), and a
  named that collides with an option already bound positionally is
  ``FILTER_OPTION_TYPE`` — a named argument never silently overrides one.
* ``enable`` stays NAMED-ONLY and framework-level: it is in no filter's option
  table, so it can never be reached positionally, and it is admitted by the
  ``T`` flag alone.
* ``ffmpeg.<filter>(...)`` is the same call under a name no SQL grammar can
  claim: identical semantics, but it bypasses Postgres's special
  forms, so ``ffmpeg.overlay(base, top, x => 20, eof_action => 'pass')``
  reaches the option set the ``OVERLAY..PLACING`` grammar hides, and
  ``ffmpeg.trim(...)`` / ``ffmpeg.format(...)`` arrive with their arguments
  intact. It is REQUIRED for the census's eleven collided names and optional
  everywhere else. The node it builds carries the FILTER's name, so nothing
  downstream knows the namespace exists.
* Three ``->N`` filters are callable through that namespace despite the pad
  scope check, because their output COUNT is fixed by an option: ``channelsplit``,
  ``acrossover`` and ``extractplanes`` (:data:`ARRAY_RETURNING`). Each lowers
  to ONE node with N output pads and RETURNS an array, so its result splats
  into a SELECT list, subscripts out of a CTE column and broadcasts
  elementwise like any other array. The table is
  consulted before the registry's verdict, since the registry has no entry to
  give; every other excluded name keeps its ``UNKNOWN_FUNCTION``.
* The mirror shape, ``N->1``, is an ORDINARY registry filter
  (``DynamicFilter.n_input``, e.g. ``amix``, ``hstack``, ``xstack`` — ~31 on
  ffmpeg 9.0.1): a variable number of INPUT pads, all of the filter's own
  output stream type, fixed by one OPTION. :data:`_NInputFilter` and its
  per-call derivation (:func:`_n_input_spec`) read that option off the
  filter's own table — ``inputs`` for most, ``nb_inputs`` where that is the
  longer name ``interleave``/``ainterleave`` dedup to. Their leading stream
  arguments ARE the input pads and the count option must agree with how many
  were supplied (``UDF_ARG_TYPE`` naming both numbers when it does not).
  Reachable BARE as well as namespaced — no Postgres grammar claims their
  names. ``ladspa`` (``N->A``) has no count option at all — its pad count is
  whatever the loaded LADSPA plugin's own ports say, so the streams supplied
  ARE the count, nothing to cross-check and nothing to write back.
  ``emit_default``/an absent count option's ``fallback`` are the two things no
  single ffmpeg build can answer about itself; a small override table
  (:data:`_N_INPUT_OVERRIDES`) covers those.
* ``ffrwd.<name>(...)`` is a THIRD namespace, resolved against
  :data:`ffrwd.macros.MACROS` and NEVER the registry -- macros work offline,
  with no ffmpeg on PATH at all. A macro owns its own fixed
  positional signature (no named arguments, no option table) and expands to a
  small filter subgraph (:data:`ffrwd.macros.Macro.expand`); its one stream
  argument broadcasts elementwise through the same :meth:`_expand_call` every
  other call uses.
* Broadcasting and zipping run off the stream-argument POSITIONS, which are
  always the leading ones, so ``volume(a.audio, 0.5)`` and
  ``anlmdn(a.audio, s => 0.01)`` expand identically.
* A UNION ALL (top level or inside a CTE) lowers each branch and joins them
  with one ``concat`` node. Branch column counts, types and order must match
  exactly (``CONCAT_MISMATCH``); concat inputs interleave per ffmpeg's segment
  contract — all of segment 1's videos, then its audios, then segment 2's, ...
  — and its output pads are ``["video"]*v + ["audio"]*a``, mapped back to the
  branch's own column order.

Broadcasting makes a bare ``a.video`` / ``a.audio`` the WHOLE array of that
input's streams, in probe order. Splatted into a SELECT list it becomes
one Output per element; handed to a function it expands the call elementwise
(a fresh subgraph per element); stored in a CTE column it keeps its length, so
``<cte>.<name>`` splats or broadcasts again and ``<cte>.<name>[k]`` picks one
element (1-based, bounds-checked statically — no probe needed at that point).
Arrays are purely a lowering concept: the spread happens here, so the IR, the
split pass and emit only ever see scalar streams.

Probing (``probes``, keyed by alias) only ever ADDS validation: an explicit
subscript lowers to the same ref whether or not the input could be probed, but
a probed input bounds-checks it (``STREAM_NOT_FOUND``). Enumerating an array is
the one thing that cannot be done symbolically — a bare array over an input
that could not be probed is ``INPUT_NOT_FOUND``. Two arrays in one call zip and
must agree on length (``BROADCAST_MISMATCH``); scalar arguments repeat.

Provenance: a stream derived 1:1 from one probed source stream — a passthrough,
or a chain of single-stream-input calls, WHERE trims included — carries that
stream's language/title tags into ``Output.metadata`` (an ffmpeg-stamped
``language=und`` carries no information and is dropped), so a broadcast
``reverb(a.audio, 0.3)`` keeps every track's language tag. A call over two or
more streams (``amix``, ``overlay``) and a ``concat`` pad (fed by one stream
per UNION ALL segment) are the other kind of join: each threads the tag only
when EVERY stream feeding it carries the same non-empty one, so mixing two
English tracks keeps ``language=eng``, but mixing English with French, or with
an untagged stream, keeps neither. Same rule, one function: ``_agreed_source``.

Node ids are ``n1, n2, ...`` in creation order across the whole graph, minted
by :class:`_NodeFactory`.

sqlglot notes that matter here
------------------------------
* Postgres has a builtin ``OVERLAY(x PLACING y FROM n FOR m)``, so
  ``overlay(a, b, x, y)`` parses to :class:`sqlglot.exp.Overlay` with *named*
  args (``this``, ``expression``, ``from_``, ``for_``) rather than to
  ``exp.Anonymous``; :func:`_call_parts` normalizes it back to four
  positionals. A ``=>`` inside that grammar is a PARSE_ERROR before lowering
  sees the call, so a BARE ``overlay`` can take its options positionally but
  never by name. Eleven registry names collide with a Postgres special form
  this way (census in docs/dynamic-filters.md); ``ffmpeg.<filter>(...)``
  reaches every one of them, because the special-form grammars key on a BARE
  name and a qualified call parses as ``Dot(Identifier(ffmpeg),
  Anonymous(...))`` whatever the filter is called.
* A subscript arrives as ``exp.Bracket`` wrapping the ``exp.Column``, and
  sqlglot REBASES the index at parse time (postgres ``INDEX_OFFSET = 1``), so
  ``a.video[1]`` holds ``Literal(0)``. Never read ``Bracket.expressions``
  here: :func:`ffrwd.parser.subscript_index` undoes the rebase and returns
  the 1-based number the user wrote.
* Neither ``Bracket`` nor ``Column`` carries a token position of its own;
  ``_pos`` walks the subtree and anchors on the qualifier identifier, which is
  the best line/col a stream error can get.
* ``exp.Literal.to_py()`` returns ``decimal.Decimal`` for non-integer numbers,
  which neither ``emit`` nor JSON can render, so numeric literals are coerced
  to ``int``/``float`` here. ``-1.5`` parses as ``exp.Neg(Literal)``.
* A named argument is an ``exp.Kwarg(this=Var(name), expression=value)``. The
  ``Var`` carries NO token position (the same gap sink option names have), so
  every rejection about one anchors on the VALUE — a literal, which does have a
  position — and falls back to the call itself for a ``Boolean`` value, which
  does not.
* A COPY option value (``WITH (crf 20)``) is NOT always a ``Literal``: ``true``
  / ``false`` arrive as ``exp.Boolean``, a bare word as ``exp.Var``, a
  double-quoted word as ``exp.Identifier``, ``NULL`` as ``exp.Null``.
  :func:`_sink_value` normalizes the first three shapes to python values and
  hands everything else to the option table as an unrepresentable value, so
  the SINK_OPTION_TYPE message and hint still come from the table.
"""

from __future__ import annotations

import base64
import difflib
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Literal

from sqlglot import exp

from ffrwd import binaries, loudnorm
from ffrwd.errors import ErrorCode, FfrwdError
from ffrwd.functions import WASM_STREAM_NAMES, Annotation, WasmFunction
from ffrwd.inputs import validate_option as validate_input_option
from ffrwd.ir import (
    NO_CHAPTERS,
    NO_METADATA,
    PIPE,
    PREDICATE,
    ROWFILTER,
    Attachment,
    FrameRef,
    Graph,
    ModuleSource,
    Node,
    Output,
    RowsSink,
    SinkUnit,
    StreamType,
    dedup_inputs,
    is_src,
    src_alias,
    src_parts,
)
from ffrwd.ir import (
    SourceTrack as IrSourceTrack,
)
from ffrwd.macros import INPUT_MACROS, MACROS, InputMacro, Macro, macro_names
from ffrwd.parser import (
    _ARITHMETIC,
    _ARITHMETIC_NAMES,
    _BUILTIN_VALUE_FUNCS,
    _REMOVED_FRAME,
    FILTER_NAMESPACE,
    MACRO_NAMESPACE,
    MAP_COLUMNS,
    ROW_PREDICATE,
    ROW_STREAM,
    SINK_STREAMS,
    RawInputOption,
    RawRowJoin,
    RawSink,
    RawSinkOption,
    RawSource,
    RawTrackRows,
    RawValuesTable,
    Resolved,
    _pos,
    _projection_expr,
    _time_bounds,
    annotation_projection,
    article,
    column_label,
    flag_error,
    from_entries,
    group_keys,
    is_grouped,
    is_value_expr,
    kwarg_name,
    map_example,
    map_noun,
    map_path,
    map_ref,
    null_variable,
    record_cast_type,
    record_unnest_hint,
    references_row_alias,
    star_except_entries,
    star_node,
    star_qualifier,
    star_replace_entries,
    subscript_index,
    subscript_metadata_shape,
    tag_key,
    tag_path,
    union_branches,
)
from ffrwd.parser import _ident_name as _fold
from ffrwd.probe import (
    WEBVTT_FORMAT,
    ProbeFailure,
    ProbeResult,
    RenditionMeta,
    StreamMeta,
    is_url,
)
from ffrwd.processes import COPY_CODEC, ref_type
from ffrwd.registry import DynamicFilter, FilterOption, Registry, SourceFilter
from ffrwd.sink import (
    CODEC_PARAMS_FLAGS,
    MANIFEST_DEFAULT_SEGMENT,
    MANIFEST_FORMATS,
    MANIFEST_MAP_OPTION,
    MANIFEST_OPTION_FORMATS,
    MANIFEST_SEGMENT_OPTION,
    SINK_OPTIONS,
    TWO_PASS_CODECS,
    copy_suppressed_scopes,
    validate_csv_option,
)
from ffrwd.sink import validate_option as validate_sink_option
from ffrwd.table import (
    ArrayCell,
    CellValue,
    RecordCell,
    StreamCell,
    TableResult,
    TableSink,
)
from ffrwd.types import (
    ATTACHMENT_TYPE,
    ATTACHMENTS_COLUMN,
    CHAPTER_TYPE,
    CHAPTERS_COLUMN,
    CONTAINER_READONLY_FIELDS,
    CUE_TYPE,
    CUES_COLUMN,
    DISPOSITION_COLUMN,
    DISPOSITION_KEYS,
    INPUT_DURATION_COLUMN,
    RECORD_ARRAY_COLUMNS,
    RECORD_ELEMENTS,
    RECORD_FIELDS,
    ROW_READONLY_FIELDS,
    ROW_SCHEMAS,
    ROW_STAR_COLUMNS,
    STAR_COLUMNS,
    STREAM_ARRAY_COLUMNS,
    STREAM_TAG_COLUMNS,
    TAGS_COLUMN,
    TIME_COLUMN,
    Field,
    RowColumnType,
    is_array,
)
from ffrwd.vars import unset_error
from ffrwd.warnings import FfrwdWarning, OnWarning, WarningCode
from ffrwd.wasm import (
    ANNOTATION_TYPES,
    AUDIO_CODEC_ENCODERS,
    CODEC_ENCODERS,
    WIRE_AUDIO_CODECS,
    WIRE_VIDEO_CODECS,
    WORLDS,
    Described,
    DescribedFunction,
    Invoke,
    SourceCatalog,
    audio_encoder_codec,
    catalog_as_probe,
    encoder_codec,
    hosts_packet_sink,
    hosts_packet_source,
    language_tag,
    rows_arms,
)
from ffrwd.wasm import invoke as wasm_invoke
from ffrwd.wasm import probe_source as wasm_probe_source

__all__ = ["lower", "lower_table"]

# Runs one packet-source module's `probe` and returns its compile-time
# catalog, or raises FfrwdError. :func:`ffrwd.wasm.probe_source` is the real
# one; a lowering test passes its own, so binding a `RETURNS source` call
# needs no sidecar.
ProbeSource = Callable[[str, str], SourceCatalog]

# The python types each JSON Schema type a module parameter may declare
# accepts. A schema naming anything else is left alone: what the module
# takes is the module's business, and only the shapes named here are ones a
# written argument can be judged against.
_JSON_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "number": (int, float),
    "integer": (int, float),
    "boolean": (bool,),
}

# A miss in `_Lowerer._invoke_cache`: distinct from every JSON value a module
# could hand back, `None` (JSON null) included.
_UNCACHED = object()


def _declares_params(known: dict[str, object]) -> str:
    """What a module's parameters are, for a message; said when it has none."""
    if not known:
        return "the module declares no parameters"
    return "the module declares " + ", ".join(sorted(known))

# The array-typed pseudo-columns an input exposes, and their element type.
# subtitle/data have the identical array/subscript/splat surface but are
# passthrough-only (see `_PASSTHROUGH_ONLY` below).
_ARRAY_COLUMNS: dict[str, StreamType] = {
    "video": "video",
    "audio": "audio",
    "subtitle": "subtitle",
    "data": "data",
}

# The pseudo `_RowBinding.column` a manifest's ABR ladder binds under: one row
# per `RenditionMeta`, read straight off an `input()` alias with no `unnest`
# to ask for it. Its schema is not a view over `container` like ROW_SCHEMAS is
# -- a rendition is not a container array field, only ever a probed fact.
RENDITION_COLUMN = "rendition"
_RENDITION_SCHEMA: dict[str, RowColumnType] = {
    "bandwidth": "number",
    "width": "number",
    "height": "number",
    "codecs": "text",
    "name": "text",
    "language": "text",
}
# Every rendition column is a probed fact, never an assertion a query can make.
_RENDITION_READONLY: frozenset[str] = frozenset(_RENDITION_SCHEMA)

# The container array columns a MEDIA query's `SELECT *` expands: the stream
# ones, in declaration order. `chapters` is an array column too, but a chapter
# is not a stream, so it takes no output position.
_STREAM_STAR_COLUMNS: tuple[str, ...] = tuple(
    name for name in STAR_COLUMNS if name in STREAM_ARRAY_COLUMNS
)

_TYPE_MARKERS: dict[StreamType, str] = {
    "video": "v",
    "audio": "a",
    "subtitle": "s",
    "data": "d",
}

# Stream types an ffmpeg filtergraph cannot carry: they may only become an
# Output (a bare `-map`), never a filter argument and never a WHERE trim's
# input.
_PASSTHROUGH_ONLY: frozenset[StreamType] = frozenset({"subtitle", "data"})

# Kind label used in UDF_ARG_TYPE "got" lists for anything that is neither a
# literal nor a stream-typed subexpression (e.g. `1 + 2`, NULL, TRUE). The
# angle brackets keep it from ever colliding with a StreamType name.
_UNSUPPORTED_KIND = "<expr>"

# Kind labels :meth:`_Lowerer._classify` gives a stream-valued argument -- the
# only kinds that may occupy a call's leading (stream input) positions.
_STREAM_KINDS: frozenset[str] = frozenset({"video", "audio", "subtitle", "data"})

# What an mp4 muxer stamps on an untagged stream: no information, so it is
# never copied onto a passthrough Output.
_UNDEFINED_LANGUAGE = "und"

_TIME_HINT = (
    "<alias>.t is only usable as WHERE <alias>.t BETWEEN <start> AND <end>, "
    "<alias>.t >= <start>, or <alias>.t <= <end>"
)
_STREAM_HINT = (
    "a SELECT column must be a stream, e.g. a.video[1] or scale(a.video[1], 640, -2)"
)
_SUBSCRIPT_HINT = "stream subscripts are 1-based: a.video[1] is the first video stream"
_FROM_ITEM_MESSAGE = (
    "only input('path'), unnest(...), ffmpeg.<source>(...), "
    "generate_series(...), and CTE or view names are allowed in FROM"
)
_ZIP_HINT = (
    "broadcast arrays zip elementwise, one output per element; "
    "subscript one of them to pair a single stream with the other, e.g. a.audio[1]"
)
_NO_REGISTRY_HINT = (
    f"ffrwd's function surface IS your installed ffmpeg's filter set; {binaries.INSTALL_HINT}"
)
_PASSTHROUGH_HINT = (
    "subtitle and data streams can only be selected (and copied), never filtered; "
    "drop them from the call and select them as their own column"
)
_SOURCE_DURATION_HINT = (
    "a generated source has no timeline to seek into; give it a length with "
    "its own option instead, e.g. ffmpeg.anullsrc(duration => 30) s"
)
_MODULE_SOURCE_SEEK_HINT = (
    "a module source paces itself -- there is no file offset to seek into; "
    "drop the WHERE window on it"
)
_ROW_METADATA_HINT = (
    "a track row's metadata columns are what you FILTER, JOIN and SORT rows by; "
    "the only column that is a stream — and therefore the only one that can be "
    "an output — is the row itself, <alias>. Give the column an alias to write "
    "it back as a TAG instead, e.g. SELECT t, t.tags.language AS language"
)
_ARRAY_AGG_HINT = (
    "array_agg takes one track-row stream expression, e.g. array_agg(t) "
    "over FROM input('f.mkv') f, unnest(f.audio) t"
)
_ONE_FILE_PER_ROW_HINT = (
    "gather the rows into that one file with array_agg(...), adding GROUP BY "
    "the column they share when they share one; or give each row a file of its "
    "own with a TO expression, e.g. TO (t.tags.language || '.mka')"
)
_ROW_WINDOW_FILE_HINT = (
    "a row-bounded window is one seek per row: gather the rows into that one "
    "file with ffmpeg.concat(VARIADIC array_agg(<column>)), or give each row a "
    "file of its own with a TO expression, e.g. TO ('clip' || i.i::text || '.mp4')"
)
# The same two ways out, spelled for rows a CTE body produced: the value the
# TO expression names has to be a column of that body.
_CTE_ROW_FILE_HINT = (
    "gather the rows into that one file with array_agg(...), or give each row a "
    "file of its own with a TO expression over a value the CTE body selected, "
    "e.g. SELECT ..., i.i AS n in the body and TO ('clip' || x.n::text || '.mp4')"
)
_PER_TRACK_OPTION_HINT = (
    "a per-row option binds one TRACK per row: gather the rows into this "
    "destination with array_agg(...) so it has a track for each; or give each "
    "row a file of its own with a TO expression, e.g. TO (:'names'[i.i] || "
    "'.mp4')"
)
_ONE_FILE_PER_GROUP_HINT = (
    "one group is one file, so the destination has to name the group, e.g. "
    "TO (t.tags.language || '.mka'); group by a column every row agrees on to write "
    "a single file instead"
)
_GROUPED_CTE_HINT = (
    "a CTE with several rows varies inside the group: wrap the column in "
    "array_agg(...), or add it to the GROUP BY to make it the group's key"
)
# The parser admits ORDER BY/LIMIT/OFFSET over any `input(...)` alias, since
# whether it turns out to be an ABR ladder is a probed fact, not a syntactic
# one -- so a renditionless input reaches here needing the same rejection
# the parser used to raise for it, word for word.
_RENDITIONLESS_ROW_CLAUSE_HINT = (
    "it is legal only over a compile-time row table -- a branch whose FROM "
    "has unnest(...), generate_series(...), or a CTE or view name -- where "
    "it narrows the resolved rows, exactly like ORDER BY"
)
# The hint a "too many rows/streams for one slot" refusal takes when the
# offending relation is a ladder: the fix is narrowing it to one rendition,
# not restructuring the query into rows.
_RENDITION_PICK_HINT = (
    "pick a rendition: WHERE on height, bandwidth or name, or ORDER BY "
    "bandwidth DESC LIMIT 1"
)
_CHAPTER_LITERAL = f"STRUCT(... AS title, ... AS start_t, ... AS end_t)::{CHAPTER_TYPE}"
_CHAPTER_EXAMPLE = f"STRUCT('Intro' AS title, 0 AS start_t, 60 AS end_t)::{CHAPTER_TYPE}"
_CHAPTERS_COLUMN_HINT = (
    f"a {CHAPTERS_COLUMN} column is an array of chapter records, e.g. "
    f"ARRAY[{_CHAPTER_EXAMPLE}] AS {CHAPTERS_COLUMN}, or "
    f"array_agg(STRUCT(c.title AS title, c.start_t AS start_t, c.end_t AS "
    f"end_t)::{CHAPTER_TYPE}) AS {CHAPTERS_COLUMN} over rows"
)
_CUE_LITERAL = f"STRUCT(... AS text, ... AS start_t, ... AS end_t)::{CUE_TYPE}"
_CUE_EXAMPLE = f"STRUCT('Hello' AS text, 0 AS start_t, 2.5 AS end_t)::{CUE_TYPE}"
_CUE_ARRAY_HINT = (
    f"an array of cue records IS a WebVTT subtitle track, e.g. "
    f"ARRAY[{_CUE_EXAMPLE}], or "
    f"array_agg(STRUCT(c.title AS text, c.start_t AS start_t, c.end_t AS "
    f"end_t)::{CUE_TYPE}) over chapter rows"
)
_ATTACHMENT_LITERAL = (
    f"STRUCT(... AS filename, ... AS mimetype, ... AS path)::{ATTACHMENT_TYPE}"
)
_ATTACHMENT_EXAMPLE = (
    f"STRUCT('font.ttf' AS filename, 'application/x-truetype-font' AS mimetype, "
    f"'fonts/font.ttf' AS path)::{ATTACHMENT_TYPE}"
)
_ATTACHMENTS_COLUMN_HINT = (
    f"an {ATTACHMENTS_COLUMN} column is an array of attachment records, e.g. "
    f"ARRAY[{_ATTACHMENT_EXAMPLE}] AS {ATTACHMENTS_COLUMN}"
)
_WRITTEN_ROW_HINT = (
    "a written row carries values, never a stream: filter, group and aggregate "
    "by its columns, e.g. array_agg(STRUCT(m.title AS title, m.start_t AS "
    "start_t, m.end_t AS end_t)::chapter) AS chapters"
)
_CAPTION_TRIM_HINT = (
    "trim the video/audio without selecting the subtitle/data columns, or select "
    "them in a query without a WHERE time range; to caption a trimmed clip, join "
    "an external subtitle file whose cues are timed for the cut"
)

# `enable` is FRAMEWORK-level: ffmpeg implements it in the filter framework,
# not in any filter, so it never appears in a filter's `-help` AVOptions and no
# options table can contain it. Which filters honour it is the `T` column of
# `ffmpeg -filters`, captured as DynamicFilter.timeline; that flag admits the
# name here.
_ENABLE = "enable"
_ENABLE_HINT = (
    "enable takes a single-quoted ffmpeg timeline expression over t (seconds), "
    "n (frame number) or pos, e.g. enable => 'between(t,2,5)'"
)
_NO_TIMELINE_HINT = (
    "enable is only accepted by filters your ffmpeg flags with timeline support "
    "(the T column of `ffmpeg -filters`: gblur has it, scale does not); drop it, "
    "or express the timing with a WHERE window over the input"
)

# Options a filter cannot run without. ffmpeg has no required-option
# metadata -- AVOption carries no such flag; each filter enforces its own in
# init() -- so this table is hand-kept, curated knowledge like MACROS. Each
# value is a tuple of groups; a group is satisfied when any one of its names
# is written (drawtext runs on `text` OR `textfile`). xfade's conditional
# requirement -- `expr`, only when `transition` is 'custom' -- cannot be a
# name list and lives in `_check_required_options` directly.
REQUIRED_OPTIONS: dict[str, tuple[tuple[str, ...], ...]] = {
    "subtitles": (("filename",),),
    "lut3d": (("file",),),
    "frei0r": (("filter_name",),),
    # plugin only matters when the library holds more than one; file always.
    "ladspa": (("file",),),
    "movie": (("filename",),),
    "amovie": (("filename",),),
    "drawtext": (("text", "textfile"),),
}

# Longest option/constant list a hint or message renders before it stops
# counting (xfade's `transition` alone has 59 constants).
_MAX_LISTED = 12


# array-RETURNING filters.
#
# Three ffmpeg filters take ONE input pad and produce a number of output pads
# fixed statically by one of their options. Their `-filters` spec is `A->N` /
# `V->N`, so the pad scope check excludes all three and `Registry.get` says None.
# This table re-admits exactly those three. It lives here, not in the registry,
# because the count rule is a property of the OPTION SEMANTICS, which nothing
# ffmpeg prints exposes: the registry keeps saying `A->N`, lowering keeps the
# arithmetic.
#
# Re-admitted through the `ffmpeg.<filter>(...)` namespace ONLY. A bare
# `channelsplit(...)` stays UNKNOWN_FUNCTION like every other excluded name.
#
# The result is an ARRAY value: `Node(outputs=[element]*N)` plus one `_Stream`
# per pad, `is_array=True` even when N == 1 (a one-element array still splats,
# subscripts through a CTE column, and broadcasts). Its pads are ordinary
# consume-once pads, so a pad read by two sinks gets an `asplit` like any other.


@dataclass(frozen=True)
class _BadCount:
    """A count rule's rejection: which option said what, and what was expected."""

    option: str
    value: str
    expected: str
    hint: str


@dataclass(frozen=True)
class _ArrayFilter:
    """One array-returning filter: its pads, and how an option fixes its count."""

    name: str
    input: StreamType  # its single input pad
    element: StreamType  # what every one of its output pads carries
    count: Callable[[dict[str, object]], int | _BadCount]


# `ffmpeg -layouts` (7.1), "Standard channel layouts": name -> how many
# channels its decomposition lists. Data, verbatim -- the whole table ffmpeg
# printed, not a curated subset of it, so the only layouts a query can be
# rejected for are the ones this ffmpeg would reject too.
_CHANNEL_LAYOUTS: dict[str, int] = {
    "mono": 1,
    "stereo": 2,
    "2.1": 3,
    "3.0": 3,
    "3.0(back)": 3,
    "4.0": 4,
    "quad": 4,
    "quad(side)": 4,
    "3.1": 4,
    "5.0": 5,
    "5.0(side)": 5,
    "4.1": 5,
    "5.1": 6,
    "5.1(side)": 6,
    "6.0": 6,
    "6.0(front)": 6,
    "3.1.2": 6,
    "hexagonal": 6,
    "6.1": 7,
    "6.1(back)": 7,
    "6.1(front)": 7,
    "7.0": 7,
    "7.0(front)": 7,
    "7.1": 8,
    "7.1(wide)": 8,
    "7.1(wide-side)": 8,
    "5.1.2": 8,
    "octagonal": 8,
    "cube": 8,
    "5.1.4": 10,
    "7.1.2": 10,
    "7.1.4": 12,
    "7.2.3": 12,
    "9.1.4": 14,
    "hexadecagonal": 16,
    "downmix": 2,
    "22.2": 24,
}

# `ffmpeg -layouts` (7.1), "Individual channels": the names a custom layout is
# composed of with `+` (`FL+FR`, `FC+LFE`), which ffmpeg accepts anywhere a
# standard layout name is accepted.
_CHANNEL_NAMES: frozenset[str] = frozenset(
    {
        "FL", "FR", "FC", "LFE", "BL", "BR", "FLC", "FRC", "BC", "SL", "SR",
        "TC", "TFL", "TFC", "TFR", "TBL", "TBC", "TBR", "DL", "DR", "WL", "WR",
        "SDL", "SDR", "LFE2", "TSL", "TSR", "BFC", "BFL", "BFR", "SSL", "SSR",
        "TTL", "TTR",
    }
)

_LAYOUT_HINT = (
    "a channel layout is one of ffmpeg's standard names (see `ffmpeg -layouts`) "
    "or a '+'-joined list of channel names, e.g. 'stereo', '5.1', 'FL+FR'"
)
_SPLIT_HINT = (
    "acrossover splits at a list of positive frequencies separated by spaces or "
    "'|', e.g. split => '500' (2 bands) or split => '500|3000' (3 bands)"
)
_PLANES_HINT = (
    "planes names the planes to extract, e.g. planes => 'y'; your ffmpeg types "
    "it as an enum, so only ONE plane per call is accepted here"
)


def _record_row_hint(record: str) -> str:
    """What a record row can be asked for, when a query asked it for a stream."""
    named = f"{article(record)} {record}"
    return (
        f"{named} row has no stream column at all — {named} is not a "
        "track — so it can only be read as a metadata query, e.g. no COPY, or "
        "COPY ... WITH (FORMAT csv)"
    )


def _option_text(value: object) -> str:
    """A validated option value as the text ffmpeg will be handed."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _channel_count(text: str) -> int | None:
    """How many channels a layout spelling describes, or None if unrecognized."""
    standard = _CHANNEL_LAYOUTS.get(text)
    if standard is not None:
        return standard
    parts = text.split("+")
    if parts and all(part in _CHANNEL_NAMES for part in parts):
        return len(parts)
    return None


def _channelsplit_count(args: dict[str, object]) -> int | _BadCount:
    """One output pad per channel channelsplit is asked to extract.

    `channels` (default "all") wins when it is set to anything else: it is
    itself a layout spelling naming the SUBSET to split out, so
    `channels => 'FL'` is one pad however wide `channel_layout` is. Verified
    against ffmpeg 7.1 -- a graph that labels more pads than the filter has is
    a hard "More output link labels specified ... than it has outputs" error,
    so the count has to follow both options, not just the documented one.
    """
    channels = _option_text(args.get("channels", "all"))
    if channels != "all":
        count = _channel_count(channels)
        if count is None:
            return _BadCount("channels", channels, "a channel layout", _LAYOUT_HINT)
        return count
    layout = _option_text(args.get("channel_layout", "stereo"))
    count = _channel_count(layout)
    if count is None:
        return _BadCount(
            "channel_layout",
            layout,
            f"one of {_listed(_CHANNEL_LAYOUTS)}",
            _LAYOUT_HINT,
        )
    return count


def _acrossover_count(args: dict[str, object]) -> int | _BadCount:
    """One band per split frequency, plus the band below the lowest one."""
    split = _option_text(args.get("split", "500"))
    parts = split.replace("|", " ").split()
    ok = bool(parts)
    for part in parts:
        try:
            frequency = float(part)
        except ValueError:
            ok = False
            break
        if not frequency > 0:
            ok = False
            break
    if not ok:
        return _BadCount("split", split, "a list of positive frequencies", _SPLIT_HINT)
    return len(parts) + 1


def _extractplanes_count(args: dict[str, object]) -> int | _BadCount:
    """One output pad per requested plane.

    ffmpeg's own option is a `flags` set (`y+u+v`), but the registry types an
    option that lists constants as an enum, so `_option_value` accepts exactly
    one of them and a `+`-joined value is FILTER_OPTION_TYPE before this rule
    ever runs. The `+` arithmetic is written out anyway: it is what the option
    means, and it is what a later plan widening flags handling will need.
    """
    planes = _option_text(args.get("planes", "r"))
    parts = planes.split("+")
    if not parts or not all(parts):
        return _BadCount("planes", planes, "one or more plane names", _PLANES_HINT)
    return len(parts)


ARRAY_RETURNING: dict[str, _ArrayFilter] = {
    "channelsplit": _ArrayFilter(
        name="channelsplit",
        input="audio",
        element="audio",
        count=_channelsplit_count,
    ),
    "acrossover": _ArrayFilter(
        name="acrossover",
        input="audio",
        element="audio",
        count=_acrossover_count,
    ),
    "extractplanes": _ArrayFilter(
        name="extractplanes",
        input="video",
        element="video",
        count=_extractplanes_count,
    ),
}

_ARRAY_INPUT_HINT = (
    "an array-returning filter takes exactly one stream, because its own result "
    "is the array; subscript the argument, e.g. a.audio[1]"
)


# N-input filters: registry.py now includes every `N->A`/`N->V` filter as an
# ordinary member of `Registry.names()`/`Registry.get()`, marked
# `DynamicFilter.n_input`. `_NInputFilter` is the per-call shape lowering
# needs on top of that -- which option (if any) carries the count, and what
# to do when it is unwritten -- derived per name by `_n_input_spec` rather
# than hand-listed.
#
# Reachable under BOTH spellings, bare and namespaced: none of these names
# collides with a Postgres special form. Every entry also takes VARIADIC
# (`_lower_variadic_n_input_call`), and `concat` (`N->N`, its own `n` option,
# still excluded from the registry on the OUTPUT side) joins them under
# VARIADIC only -- see `_lower_concat_call`.


@dataclass(frozen=True)
class _NInputFilter:
    """One N-input filter's call shape: its pads, and the option fixing the count."""

    name: str
    stream: StreamType  # what every one of its INPUT pads carries
    output: StreamType  # its single output pad
    option: str | None  # the option whose value IS the input-pad count; None
    # when there is no such option (ladspa: the plugin's own ports decide) --
    # then the supplied stream count is never checked against anything and
    # never written back.
    fallback: int  # count when the option is neither written nor introspectable
    # Write the count onto the node even when it equals the fallback. True for
    # the filters that are N-input on EVERY ffmpeg (amix: pins carry
    # `inputs=2`); False for ones that grew the option in a later ffmpeg
    # (acrossfade, N->A since ffmpeg 9) -- omitting the defaulted count keeps
    # the compiled command valid on builds whose acrossfade has no such
    # option at all.
    emit_default: bool = True


# What no single ffmpeg build's introspection can answer about itself:
# whether writing the DEFAULTED count is safe on an older build that lacks
# the option entirely (acrossfade, N->A only since ffmpeg 9), and ladspa's
# fallback/emit_default, whose "count" is never a real ffmpeg option value.
# Everything else is derived from the registry -- see `_n_input_spec`.
@dataclass(frozen=True)
class _NInputOverride:
    fallback: int | None = None
    emit_default: bool | None = None


_N_INPUT_OVERRIDES: dict[str, _NInputOverride] = {
    "acrossfade": _NInputOverride(emit_default=False),
    "ladspa": _NInputOverride(fallback=0, emit_default=False),
}

# The count option's name, in the order to look for it: `inputs` for most
# N-input filters, `nb_inputs` where that is the longer name the registry's
# adjacent-alias dedup keeps (interleave/ainterleave -- `n` is the alias it
# drops). A filter with neither has no count option (`option=None`).
_N_INPUT_OPTION_NAMES = ("inputs", "nb_inputs")


def _n_input_spec(
    name: str, dynamic: DynamicFilter, options: dict[str, FilterOption]
) -> _NInputFilter:
    """One N-input filter's call shape, derived from what this registry reports.

    `stream`/`output` both come from `dynamic.output`: ffmpeg's pad notation
    for an N-input filter is just `N->V`/`N->A`, one letter, and every filter
    observed takes input pads of that same kind. `option`/`fallback` come from
    the filter's own option table; `_N_INPUT_OVERRIDES` covers the two things
    no single build can answer about itself.
    """
    option_name = next((n for n in _N_INPUT_OPTION_NAMES if n in options), None)
    fallback = 2
    emit_default = True
    if option_name is not None:
        default = options[option_name].default
        if default is not None:
            try:
                fallback = int(float(default))
            except ValueError:
                pass
    else:
        fallback = 0
        emit_default = False
    override = _N_INPUT_OVERRIDES.get(name)
    if override is not None:
        if override.fallback is not None:
            fallback = override.fallback
        if override.emit_default is not None:
            emit_default = override.emit_default
    return _NInputFilter(
        name=name,
        stream=dynamic.output,
        output=dynamic.output,
        option=option_name,
        fallback=fallback,
        emit_default=emit_default,
    )


_N_INPUT_HINT = (
    "the number of streams you pass IS the filter's input count; either pass "
    "that many streams, or set the count explicitly, e.g. amix(a, b, c, inputs => 3)"
)

# `concat` stays excluded from the registry (dynamic on the OUTPUT side too,
# `N->N` -- see registry.py), but VARIADIC gives its count a source, so it is
# callable on those terms alone -- never without VARIADIC.
_CONCAT_NAME = "concat"

_CONCAT_VARIADIC_HINT = (
    "concat has a variable pad count: call it with VARIADIC, e.g. "
    "concat(VARIADIC array_agg(v))"
)

_VARIADIC_HINT = (
    "VARIADIC spreads an array as the call's argument list, and only a filter "
    "whose pad count follows its argument count takes it -- an N-input filter "
    "(amix, hstack, xstack, ...) or concat"
)


# errors


def _error(
    code: ErrorCode,
    message: str,
    node: exp.Expr | None = None,
    *,
    fallback: exp.Expr | None = None,
    hint: str | None = None,
) -> FfrwdError:
    line, col = _pos(node, fallback)
    return FfrwdError(code, message, line=line, col=col, hint=hint)


def _computed_segments(expression: exp.Expr, row_aliases: set[str]) -> list[exp.Expr]:
    """The pieces of a path expression whose text comes from row metadata.

    A ``||`` chain is split at its operands, so the literal directory in
    ``'out/' || t.tags.language`` stays a literal and only ``t.tags.language`` is
    checked. Anything else is one segment, computed if it reads a row at all.
    """
    node = _unwrap(expression)
    if isinstance(node, exp.DPipe):
        expression_node = node.args.get("expression")
        sides = [node.this, expression_node if isinstance(expression_node, exp.Expr) else None]
        return [
            segment
            for side in sides
            if isinstance(side, exp.Expr)
            for segment in _computed_segments(side, row_aliases)
        ]
    return [node] if references_row_alias(node, row_aliases) else []


def _projects_annotations(node: exp.Expr, column: str) -> bool:
    """True when `column` is read off `node`, through any wrapping parens."""
    inner, parent = node, node.parent
    while isinstance(parent, exp.Paren):
        inner, parent = parent, parent.parent
    if not isinstance(parent, exp.Dot) or parent.this is not inner:
        return False
    field = parent.args.get("expression")
    return isinstance(field, exp.Identifier) and _fold(field) == column


def _stream_projection(
    node: exp.Expr, wasm: Mapping[str, WasmFunction]
) -> exp.Anonymous | None:
    """``<module call>.<stream field>``, as the call the field is read off.

    None for every other expression. Resolve has already refused this
    projection everywhere but beside the same struct's annotation column.
    """
    dot = _unwrap(node)
    if not isinstance(dot, exp.Dot):
        return None
    base = _unwrap(dot.this) if isinstance(dot.this, exp.Expr) else None
    field = dot.args.get("expression")
    if not isinstance(base, exp.Anonymous) or not isinstance(field, exp.Identifier):
        return None
    declared = wasm.get(str(base.name).lower())
    if declared is None or declared.emits is None:
        return None
    return base if _fold(field) == declared.stream_field else None


def _annotation_fields(annotation: Annotation) -> tuple[tuple[str, str], ...]:
    """One annotation record's fields, name-ordered, for comparing two of them."""
    return tuple(sorted((f.name, f.type) for f in annotation.fields))


def _annotation_matches(
    declared: Sequence[tuple[str, str]], emitted: Sequence[tuple[str, str]]
) -> bool:
    """Whether a declared annotation record and a module's rows are the same shape.

    Same column names, and each declared type covering the JSON type the
    module gave that column. Order says nothing: the rows travel keyed by name.
    """
    if len(declared) != len(emitted):
        return False
    return all(
        name == emitted_name and json_type in ANNOTATION_TYPES.get(kind, ())
        for (name, kind), (emitted_name, json_type) in zip(declared, emitted)
    )


def _written_json_fields(fields: Sequence[tuple[str, str]]) -> str:
    """A module's row schema as a message spells it."""
    if not fields:
        return "rows with no columns"
    return "rows of " + ", ".join(f"{name} ({kind or 'no type'})" for name, kind in fields)


def _describe(node: exp.Expr) -> str:
    """Short human name for an expression that cannot produce a stream."""
    if isinstance(node, exp.Literal):
        return "a string literal" if node.is_string else "a numeric literal"
    if isinstance(node, exp.Neg):
        return "a numeric literal"
    if isinstance(node, exp.Null):
        return "NULL"
    if isinstance(node, exp.Boolean):
        return "a boolean literal"
    if isinstance(node, exp.Case):
        return "a CASE expression"
    if isinstance(node, exp.DPipe):
        return "a '||' expression"
    return f"a {node.__class__.__name__.upper()} expression"


# small AST helpers


def _unwrap(node: exp.Expr) -> exp.Expr:
    """Strip projection aliases and redundant parentheses."""
    while True:
        if isinstance(node, exp.Alias | exp.Paren):
            inner = node.this
            if isinstance(inner, exp.Expr):
                node = inner
                continue
        return node


def _projection_name(node: exp.Expr) -> str | None:
    """The ``AS`` name of a projection, folded Postgres-style, else None."""
    if not isinstance(node, exp.Alias):
        return None
    alias = node.args.get("alias")
    if not isinstance(alias, exp.Expr):
        return None
    name = _fold(alias)
    return name or None


def _table_column_name(node: exp.Expr) -> str:
    """A table/csv column's header: the ``AS`` alias, else its natural name.

    The SELECT alias when given, else the column expression's natural name
    (``language``, ``codec``, ...). A bare row/input column names itself, and
    a bare row alias names the alias, as Postgres does for a whole-row
    reference; a subscript metadata accessor names the metadata field it
    reads (``f.audio[1].codec`` -> ``codec``, matching a row table's
    own column of the same name); anything else (a filter call, COALESCE,
    ...) has no single name to fall back to. A tag path names its KEY
    (``a.tags.language`` -> ``language``): the last part of the path, the way
    Postgres names any field reference.
    """
    alias = _projection_name(node)
    if alias is not None:
        return alias
    inner = _unwrap(node)
    if isinstance(inner, exp.Column):
        name = _fold(inner.this)
        if name == ROW_STREAM:
            return _fold(inner.args.get("table"))
        return _map_key(name)
    if isinstance(inner, exp.ArrayAgg):
        return "array_agg"  # Postgres's own convention for the unaliased column
    shape = subscript_metadata_shape(inner)
    if shape is not None:
        return _map_key(shape[1])
    return "column"


def _flatten_and(node: exp.Expr | None) -> list[exp.Expr]:
    """Flatten an AND tree into its conjuncts, left to right."""
    out: list[exp.Expr] = []
    stack: list[exp.Expr | None] = [node]
    while stack:
        current = stack.pop(0)
        if current is None:
            continue
        while isinstance(current, exp.Paren) and isinstance(current.this, exp.Expr):
            current = current.this
        if isinstance(current, exp.And):
            expression = current.args.get("expression")
            stack.insert(0, expression if isinstance(expression, exp.Expr) else None)
            stack.insert(0, current.this if isinstance(current.this, exp.Expr) else None)
            continue
        out.append(current)
    return out


@dataclass(frozen=True)
class _NamedArg:
    """One ``name => value`` call argument.

    `name` is verbatim (ffmpeg AVOption names are case-sensitive) and `value` is
    the raw sqlglot node — the option table this is checked against comes from
    the installed ffmpeg, so nothing is interpreted before the registry says
    what the option's type is.
    """

    name: str
    value: exp.Expr


@dataclass(frozen=True)
class _Call:
    """A function call as lowering sees it: a name, positional args, named args.

    `namespaced` marks the ``ffmpeg.<filter>(...)`` spelling, which
    resolves in the registry under a name no Postgres grammar can claim.
    `is_macro` marks the ``ffrwd.<name>(...)`` spelling, which
    resolves against :data:`MACROS` and never touches the registry. The two
    are mutually exclusive (different Dot qualifiers).

    `variadic` is the array expression inside a trailing ``VARIADIC <array>``
    argument, already unwrapped from ``exp.Variadic`` and excluded from
    `args` -- Postgres allows at most one, and it is always last, which
    :func:`_split_args` enforces at parse time.
    """

    name: str
    args: list[exp.Expr]
    named: list[_NamedArg]
    namespaced: bool = False
    is_macro: bool = False
    variadic: exp.Expr | None = None

    @property
    def display(self) -> str:
        """The call as the user spelled it, for error messages."""
        if self.namespaced:
            return f"{FILTER_NAMESPACE}.{self.name}"
        if self.is_macro:
            return f"{MACRO_NAMESPACE}.{self.name}"
        return self.name


def _namespaced_call(node: exp.Expr) -> exp.Anonymous | None:
    """The ``exp.Anonymous`` inside ``ffmpeg.<filter>(...)``, else None.

    VERIFIED (sqlglot 30.17, ``read="postgres"``): a qualified call parses as
    ``exp.Dot(this=Identifier(ffmpeg), expression=exp.Anonymous(...))`` for
    EVERY filter name, with its positional arguments and its ``=>`` kwargs
    intact inside the ``Anonymous``. Postgres's special-form grammars —
    ``OVERLAY(x PLACING y ...)``, ``TRIM``, ``FORMAT``, ``MEDIAN``, ... — key
    on a BARE name, so qualifying the call bypasses all of them at once. That
    is the whole point of the namespace: it is the one spelling of a filter
    name that no SQL grammar has an opinion about.
    """
    if not isinstance(node, exp.Dot):
        return None
    qualifier = node.this
    if not isinstance(qualifier, exp.Identifier) or _fold(qualifier) != FILTER_NAMESPACE:
        return None
    inner = node.args.get("expression")
    return inner if isinstance(inner, exp.Anonymous) else None


def _macro_call(node: exp.Expr) -> exp.Anonymous | None:
    """The ``exp.Anonymous`` inside ``ffrwd.<name>(...)``, else None.

    Mirrors :func:`_namespaced_call` exactly, and VERIFIED to parse
    to the identical shape under sqlglot 30.17 ``read="postgres"`` for all
    three macro names: ``exp.Dot(this=Identifier(ffrwd),
    expression=exp.Anonymous(this=<macro>, expressions=[...]))``, symmetric
    with the ffmpeg namespace's.
    """
    if not isinstance(node, exp.Dot):
        return None
    qualifier = node.this
    if not isinstance(qualifier, exp.Identifier) or _fold(qualifier) != MACRO_NAMESPACE:
        return None
    inner = node.args.get("expression")
    return inner if isinstance(inner, exp.Anonymous) else None


def _call_parts(node: exp.Expr) -> _Call | None:
    """The call `node` is, else None.

    ``exp.Overlay`` is normalized back to the four positional arguments the
    SQL surface uses; sqlglot parks them under named keys because Postgres
    spells the builtin ``OVERLAY(x PLACING y FROM n FOR m)``. (That builtin
    grammar also means a BARE ``overlay(...)`` cannot take named arguments at
    all: sqlglot rejects ``=>`` inside it at PARSE time. Its options are still
    reachable positionally, and ``ffmpeg.overlay(base, top, x => 20, y => 20)``
    reaches every one of them by name.)

    Named arguments arrive as ``exp.Kwarg`` among the positional ones and are
    split out here. Their TRAILING position is enforced by resolve; the check
    is repeated defensively because a Kwarg among positional args would
    otherwise silently shift every parameter after it.
    """
    inner = _namespaced_call(node)
    if inner is not None:
        return _split_args(str(inner.this), inner, namespaced=True)
    macro_inner = _macro_call(node)
    if macro_inner is not None:
        return _split_args(str(macro_inner.this), macro_inner, is_macro=True)
    if isinstance(node, exp.Overlay):
        parts = [
            node.this,
            node.args.get("expression"),
            node.args.get("from_"),
            node.args.get("for_"),
        ]
        return _Call("overlay", [arg for arg in parts if isinstance(arg, exp.Expr)], [])
    if isinstance(node, exp.Anonymous):
        return _split_args(str(node.this), node)
    if isinstance(node, exp.Func):
        return _split_args(node.sql_name().lower(), node)
    return None


def _split_args(
    name: str, call: exp.Expr, *, namespaced: bool = False, is_macro: bool = False
) -> _Call:
    positional: list[exp.Expr] = []
    named: list[_NamedArg] = []
    variadic: exp.Expr | None = None
    for arg in call.expressions:
        if not isinstance(arg, exp.Expr):
            continue
        if isinstance(arg, exp.Kwarg):
            value = arg.args.get("expression")
            if not isinstance(value, exp.Expr):  # resolve already rejected this
                raise _error(
                    ErrorCode.UNSUPPORTED_SQL,
                    f"named argument '{kwarg_name(arg)}' has no value",
                    arg,
                )
            named.append(_NamedArg(name=kwarg_name(arg), value=value))
            continue
        if named:
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                "positional arguments must come before named arguments",
                arg,
                fallback=call,
            )
        if variadic is not None:
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                "VARIADIC must be the last argument",
                arg,
                fallback=call,
            )
        if isinstance(arg, exp.Variadic):
            inner = arg.this
            if not isinstance(inner, exp.Expr):
                raise _error(
                    ErrorCode.UNSUPPORTED_SQL,
                    "VARIADIC needs an array expression",
                    arg,
                    fallback=call,
                )
            variadic = inner
            continue
        positional.append(arg)
    return _Call(name, positional, named, namespaced, is_macro, variadic)


# literal coercion


def _number(node: exp.Expr, code: ErrorCode = ErrorCode.UDF_ARG_TYPE) -> int | float:
    """Python value of a numeric literal, negation included.

    ``to_py()`` hands back ``decimal.Decimal`` for non-integers (the IR only
    carries JSON/ffmpeg-renderable scalars, so that is narrowed to float here)
    and raises ``ValueError`` on malformed literals sqlglot still tokenized as
    numbers, e.g. ``1e`` — which must surface as a typed rejection, not a panic.
    """
    node = _unwrap(node)
    sign = 1
    if isinstance(node, exp.Neg) and isinstance(node.this, exp.Expr):
        sign = -1
        node = node.this
    if not isinstance(node, exp.Literal) or node.is_string:
        raise _error(code, "expected a numeric literal", node)
    try:
        value = node.to_py()
        if isinstance(value, bool):
            raise ValueError(value)
        return sign * value if isinstance(value, int) else sign * float(value)
    except (ArithmeticError, TypeError, ValueError):
        raise _error(code, f"could not read {str(node.this)!r} as a number", node) from None


@dataclass(frozen=True)
class _Unrepresentable:
    """A COPY option value that is no python scalar at all (``NULL``, a bare word).

    Handed to :func:`ffrwd.sink.validate_option` AS the value: it is never a
    ``str``/``int``/``bool``, so every declared option type rejects it and the
    SINK_OPTION_TYPE message plus its per-type hint still come from the option
    table — guardrail #4, no option knowledge is duplicated here. ``__repr__``
    is what the message interpolates, so it reads back as what the user wrote.
    """

    text: str

    def __repr__(self) -> str:
        return self.text


def _sink_describe(node: exp.Expr) -> str:
    if isinstance(node, exp.Var):
        return f"the bare word {node.name}"
    if isinstance(node, exp.Identifier):
        return f'the identifier "{node.name}"'
    return _describe(node)


def _each(value: object) -> list[object]:
    """One option's values: every element of a per-pad list, or the one value.

    Empty for an option that is not written at all, so a caller loops over
    nothing rather than checking first.
    """
    if value is None:
        return []
    return list(value) if isinstance(value, list) else [value]


def _sink_stream_count(node: exp.Expr, arguments: int) -> int:
    """How many of a sink call's leading arguments came out of the SELECT list.

    Recorded by the rewrite that made the call, which is the only place the
    split is known: a sink reading an ARRAY of streams has no fixed argument
    count to work it out from.
    """
    written = node.meta.get(SINK_STREAMS)
    return written if isinstance(written, int) else arguments


def _validated_option(name: str, written: object, *, line: int, col: int) -> object:
    """One option value through the option table -- element by element when it
    is a per-TRACK list, so a bad element is refused where a bad scalar is.
    A None element is that row's NULL read: absence, never a type error."""
    if isinstance(written, list):
        return [
            element
            if element is None
            else validate_sink_option(name, element, line=line, col=col)
            for element in written
        ]
    return validate_sink_option(name, written, line=line, col=col)


def _manifest_format(raw: RawSink) -> str | None:
    """The manifest format ('hls'/'dash') a COPY's WITH block names, else None.

    Read off the raw option shape, like ``RawSink.is_csv``, because it changes
    how the wrapped query is allowed to lower -- a manifest destination takes
    a multi-row relation -- and that is decided before option values are
    otherwise interpreted.
    """
    if raw.is_csv:
        return None
    for option in raw.options:
        if option.name != "format":
            continue
        value = _sink_value(_unwrap(option.value))
        if isinstance(value, str) and value in MANIFEST_FORMATS:
            return value
    return None


@dataclass(frozen=True)
class _VariantRow:
    """One row of a manifest destination: the streams its cells hold.

    A NULL cell is None -- that kind absent from this variant. A video-only
    row is a variant drawing from the audio group, an audio-only row a
    rendition in it, a both-cells row a muxed variant.
    """

    video: _Stream | None
    audio: _Stream | None


def _compress_manifest_lists(
    options: dict[str, object], variant_rows: list[_VariantRow]
) -> None:
    """Per-row option lists, cut to the rows that hold the option's kind.

    A per-row option was read once per ROW; at a manifest destination the
    rows of the option's scope are the ones that carry a stream of that kind
    (a video option read over an audio-only row read NULL through its NULL
    subscript anyway). After the cut the list is one element per track, the
    shape the per-track check and emit already speak.
    """
    for name, value in options.items():
        if not isinstance(value, list) or len(value) != len(variant_rows):
            continue
        scope = SINK_OPTIONS[name].scope
        if scope == "video":
            options[name] = [
                element
                for element, row in zip(value, variant_rows)
                if row.video is not None
            ]
        elif scope == "audio":
            options[name] = [
                element
                for element, row in zip(value, variant_rows)
                if row.audio is not None
            ]


def _fallback_names(candidates: list[str | None], prefix: str) -> list[str]:
    """Each candidate name, or its positional fallback where it fails.

    A name fails when it is missing, ``und``, or shared with another of its
    kind -- the colliding ones ALL fall back, since neither owns the name.
    """
    counts: dict[str, int] = {}
    for name in candidates:
        if name is not None:
            counts[name] = counts.get(name, 0) + 1
    return [
        name
        if name is not None and name != _UNDEFINED_LANGUAGE and counts[name] == 1
        else f"{prefix}{position}"
        for position, name in enumerate(candidates)
    ]


def _stream_language(stream: _Stream) -> str | None:
    """The probed language tag a stream carries, None for none or ``und``."""
    source = stream.source
    if source is None:
        return None
    language = source.metadata.get("language")
    if language is None or language == _UNDEFINED_LANGUAGE:
        return None
    return str(language)


# The rate-setting args a chain walk reads: the fps filter's own, and a
# generated source's.
_RATE_ARGS = ("fps", "rate", "r", "framerate")


def _node_rate(node: Node) -> float | None:
    """The frame rate `node` imposes on what flows through it, if any."""
    if node.filter not in ("fps", "framerate") and node.inputs:
        return None
    for key in _RATE_ARGS:
        rate = _parse_rate(node.args.get(key))
        if rate is not None:
            return rate
    return None


def _parse_rate(value: object) -> float | None:
    """A frame rate as a float: ``30``, ``29.97``, or ffprobe's ``30000/1001``."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value) if value > 0 else None
    if not isinstance(value, str):
        return None
    text = value.strip()
    numerator, slash, denominator = text.partition("/")
    try:
        if slash:
            top, bottom = float(numerator), float(denominator)
            return top / bottom if top > 0 and bottom > 0 else None
        rate = float(text)
        return rate if rate > 0 else None
    except ValueError:
        return None


def _int_arg(node: Node, *keys: str) -> int | None:
    """A node arg as an int, under whichever of `keys` it sits; None otherwise."""
    for key in keys:
        value = node.args.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                continue
    return None


def _text_number(value: int | float) -> str:
    """A number the way a message spells it: ``6``, not ``6.0``."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _sink_value(node: exp.Expr) -> object:
    """One ``COPY ... WITH (name value)`` value as a python scalar.

    Never raises and never validates: an unusable shape comes back as an
    :class:`_Unrepresentable`, and a well-formed value of the WRONG type (a
    float for ``crf``, a string for ``faststart``) comes back as itself. The
    option table decides in both cases.
    """
    if isinstance(node, exp.Boolean):
        return bool(node.this)
    if isinstance(node, exp.Literal):
        if node.is_string:
            return str(node.this)
        try:
            value = node.to_py()
        except (ArithmeticError, TypeError, ValueError):
            return _Unrepresentable(repr(str(node.this)))
        if isinstance(value, bool):  # sqlglot never does this; be explicit anyway
            return _Unrepresentable(repr(value))
        # Decimal renders neither to JSON nor to an ffmpeg arg; float does, and
        # a float is a type error for every v1 option anyway.
        return value if isinstance(value, int) else float(value)
    return _Unrepresentable(_sink_describe(node))


# Characters ffmetadata's own escaping would need (`\`, `=`, `;`, `#`, a
# newline) -- rejected outright rather than silently writing a file ffmpeg
# cannot parse back.
_UNSAFE_CHAPTER_TITLE = frozenset("\\=;#\n\r")


def _struct_node(node: exp.Expr) -> exp.Struct | None:
    """The ``STRUCT(...)`` a cast wraps, else None."""
    inner = _unwrap(node.this) if isinstance(node.this, exp.Expr) else None
    return inner if isinstance(inner, exp.Struct) else None


def _struct_fields(node: exp.Struct) -> dict[str, exp.Expr]:
    """One ``STRUCT(value AS name, ...)`` as its fields, by name.

    Every field is named: a positional entry has no name to match against the
    record's own, so it is rejected rather than silently taken in order.
    """
    fields: dict[str, exp.Expr] = {}
    for entry in node.expressions:
        if not isinstance(entry, exp.PropertyEQ):
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"a STRUCT field is named, got {_describe(entry)}",
                entry if isinstance(entry, exp.Expr) else node,
                fallback=node,
                hint="name every field with AS, e.g. STRUCT('Intro' AS title)",
            )
        name = _fold(entry.this)
        value = entry.expression
        if not isinstance(value, exp.Expr):
            continue
        if name in fields:
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"STRUCT names the field '{name}' twice",
                entry,
                fallback=node,
                hint="one value per field name",
            )
        fields[name] = value
    return fields


@dataclass(frozen=True)
class _Chapter:
    """One written chapter: its span, its title, and where they were written.

    `start_node` / `end_node` are the expressions the bounds came from, so a
    span rejection anchors on the number the query typed.
    """

    start: int | float
    end: int | float
    title: str | None
    start_node: exp.Expr
    end_node: exp.Expr


@dataclass(frozen=True)
class _Cue:
    """One written cue: its span, its text, and where the bounds were written."""

    start: int | float
    end: int | float
    text: str
    start_node: exp.Expr
    end_node: exp.Expr


def _chapters_ffmetadata(chapters: Sequence[_Chapter]) -> str:
    """One evaluated ``chapter[]`` as an ffmetadata document's text.

    ``;FFMETADATA1`` plus one ``[CHAPTER]`` block per record, in written
    order. `title` is nullable, and a NULL one omits the line entirely.
    """
    scale = _chapter_timebase(chapters)
    lines = [";FFMETADATA1"]
    previous: tuple[int | float, int | float] | None = None
    for position, chapter in enumerate(chapters, start=1):
        _check_chapter_span(
            CHAPTERS_COLUMN,
            position,
            chapter.start,
            chapter.end,
            previous,
            chapter.start_node,
            chapter.end_node,
        )
        previous = (chapter.start, chapter.end)
        lines.append("[CHAPTER]")
        lines.append(f"TIMEBASE=1/{scale}")
        lines.append(f"START={round(chapter.start * scale)}")
        lines.append(f"END={round(chapter.end * scale)}")
        if chapter.title is not None:
            lines.append(f"title={chapter.title}")
    return "\n".join(lines) + "\n"


def _chapter_timebase(chapters: Sequence[_Chapter]) -> int:
    """The ffmetadata timebase this chapter list needs, as its denominator.

    ffmetadata's ``START``/``END`` are INTEGERS counted in the block's
    timebase, so ``1/1`` reads whole seconds and would truncate a bound of
    0.6 down to 0. A list every bound of which is a whole number keeps
    ``1/1`` -- the plainest thing to read -- and one with a fraction anywhere
    counts in milliseconds instead, which is as fine as either a chapter mark
    or a WebVTT cue is written.
    """
    bounds = [bound for chapter in chapters for bound in (chapter.start, chapter.end)]
    return 1 if all(float(bound).is_integer() for bound in bounds) else 1000


def _check_chapter_span(
    alias: str,
    position: int,
    start: int | float,
    end: int | float,
    previous: tuple[int | float, int | float] | None,
    start_cell: exp.Expr,
    end_cell: exp.Expr,
) -> None:
    """One written chapter against the three rules a chapter list obeys.

    A chapter runs forward, the list runs forward, and two chapters never cover
    the same second: a player reads them in written order and has no way to
    show a span that goes backwards or sits inside its neighbour.
    """
    if start >= end:
        raise _error(
            ErrorCode.UNSUPPORTED_SQL,
            f"'{alias}' chapter {position} ends at {end}, which is not after "
            f"its start {start}",
            end_cell,
            hint="a chapter runs from start_t to end_t: end_t must be larger",
        )
    if previous is None:
        return
    previous_start, previous_end = previous
    if start < previous_start:
        raise _error(
            ErrorCode.UNSUPPORTED_SQL,
            f"'{alias}' chapter {position} starts at {start}, before chapter "
            f"{position - 1} at {previous_start}",
            start_cell,
            hint="chapters are written in ascending order; reorder the rows",
        )
    if start < previous_end:
        raise _error(
            ErrorCode.UNSUPPORTED_SQL,
            f"'{alias}' chapter {position} starts at {start}, inside chapter "
            f"{position - 1} which ends at {previous_end}",
            start_cell,
            hint=f"chapters may not overlap: start this one at or after {previous_end}",
        )


def _span_number(
    value: RowValue, label: str, column: str, node: exp.Expr, example: str
) -> int | float:
    """One evaluated ``start_t``/``end_t`` as the number it must be, never NULL.

    `label` is how the rejection names the field the value was written for --
    a chapter's belongs to the column that holds the list, a cue's to the
    record itself, since a cue array is a stream rather than a column.
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        got = "NULL" if value is None else repr(value)
        raise _error(
            ErrorCode.UNSUPPORTED_SQL,
            f"'{label}' must be a number, got {got}",
            node,
            hint=f"{column} is a number of seconds, e.g. {example}",
        )
    return value


def _chapter_title(value: RowValue, node: exp.Expr) -> str | None:
    """One evaluated ``title`` as text, or None for NULL (ffmetadata omits it)."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise _error(
            ErrorCode.UNSUPPORTED_SQL,
            f"'{CHAPTERS_COLUMN}.title' must be a string or NULL, got {value!r}",
            node,
            hint=f"title is text or NULL, e.g. {_CHAPTER_EXAMPLE}",
        )
    if any(char in _UNSAFE_CHAPTER_TITLE for char in value):
        raise _error(
            ErrorCode.UNSUPPORTED_SQL,
            f"'{CHAPTERS_COLUMN}.title' {value!r} contains a character "
            "ffmetadata cannot represent unescaped",
            node,
            hint=r"avoid \ = ; # and newlines in a chapter title",
        )
    return value


# What the sidecar writes a module's rows as when the query writes them
# itself, and the destination that asks for it.
_ROWS_CONTAINER = "ndjson"
_ROWS_SUFFIX = ".ndjson"

# The subtitle codec a minted rows track is written with, per container, and
# the ffmpeg option that names it. A container missing here carries WebVTT as
# it stands, and the track is copied.
_ROWS_TRACK_CODECS: Mapping[str, str] = {
    "mp4": "mov_text",
    "m4v": "mov_text",
    "mov": "mov_text",
    "srt": "srt",
    "ass": "ass",
    "ssa": "ass",
}
_SUBTITLE_CODEC_OPTION = "subtitle_codec"


def _rows_meta(tag: str | None) -> StreamMeta | None:
    """The provenance a minted rows track carries: its language, or nothing.

    A track the query gave no language is a track with nothing to say about
    itself, which is exactly an unprobed stream.
    """
    if tag is None:
        return None
    return StreamMeta(
        type="subtitle",
        index=0,
        metadata={"language": tag},
        width=None,
        height=None,
        fps=None,
        sample_rate=None,
    )


# The document's first word, the separator between a cue's two bounds, and
# the two characters WebVTT reads as markup inside a cue.
_WEBVTT_MAGIC = "WEBVTT"
_CUE_ARROW = "-->"
_WEBVTT_ESCAPES = (("&", "&amp;"), ("<", "&lt;"))


def _cues_webvtt(cues: Sequence[_Cue]) -> str:
    """One evaluated ``cue[]`` as a WebVTT document's text.

    ``WEBVTT`` then one block per cue, in written order, blocks separated by
    a blank line: the format `ffrwd.empty_captions()` already writes, with
    cues in it. Bounds render as ``HH:MM:SS.mmm``, which is the only
    timestamp spelling WebVTT has.
    """
    blocks = [_WEBVTT_MAGIC]
    previous: int | float | None = None
    for position, cue in enumerate(cues, start=1):
        _check_cue_span(position, cue.start, cue.end, previous, cue.start_node, cue.end_node)
        previous = cue.start
        timing = f"{_cue_timestamp(cue.start)} {_CUE_ARROW} {_cue_timestamp(cue.end)}"
        blocks.append(f"{timing}\n{cue.text}")
    return "\n\n".join(blocks) + "\n"


def _cue_timestamp(seconds: int | float) -> str:
    """One cue bound as WebVTT's ``HH:MM:SS.mmm``."""
    total = round(seconds * 1000)
    hours, total = divmod(total, 3_600_000)
    minutes, total = divmod(total, 60_000)
    whole, milliseconds = divmod(total, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole:02d}.{milliseconds:03d}"


def _check_cue_span(
    position: int,
    start: int | float,
    end: int | float,
    previous: int | float | None,
    start_cell: exp.Expr,
    end_cell: exp.Expr,
) -> None:
    """One written cue against the two rules a WebVTT document obeys.

    A cue runs forward and the document lists its cues in ascending order.
    Overlap is NOT a rule here, unlike a chapter list: WebVTT is allowed to
    show two captions at once, and a player draws both.
    """
    if start >= end:
        raise _error(
            ErrorCode.UNSUPPORTED_SQL,
            f"cue {position} ends at {end}, which is not after its start {start}",
            end_cell,
            hint="a cue runs from start_t to end_t: end_t must be larger",
        )
    if previous is not None and start < previous:
        raise _error(
            ErrorCode.UNSUPPORTED_SQL,
            f"cue {position} starts at {start}, before cue {position - 1} at "
            f"{previous}",
            start_cell,
            hint="a WebVTT document lists its cues in ascending order; reorder "
            "the rows. Two cues MAY overlap",
        )


def _cue_text(value: RowValue, node: exp.Expr) -> str:
    """One evaluated ``text`` as the payload the cue block carries.

    WebVTT ends a cue at the next blank line and reads ``&`` and ``<`` as
    markup, so the two are escaped the way the format says and a payload that
    would break the block out is rejected instead of quietly truncating it.
    """
    if not isinstance(value, str):
        got = "NULL" if value is None else repr(value)
        raise _error(
            ErrorCode.UNSUPPORTED_SQL,
            f"'{CUE_TYPE}.text' must be a string, got {got}",
            node,
            hint=f"text is what the cue shows, e.g. {_CUE_EXAMPLE}",
        )
    if not value.strip():
        raise _error(
            ErrorCode.UNSUPPORTED_SQL,
            f"'{CUE_TYPE}.text' is empty, and a cue with nothing to show is "
            "not a cue",
            node,
            hint=f"write what the cue says, e.g. {_CUE_EXAMPLE}",
        )
    if _CUE_ARROW in value or "\r" in value or "\n\n" in value:
        raise _error(
            ErrorCode.UNSUPPORTED_SQL,
            f"'{CUE_TYPE}.text' {value!r} contains a character WebVTT cannot "
            "represent inside a cue",
            node,
            hint="a cue's text runs to the next blank line, so it may hold no "
            "blank line and no arrow (-->)",
        )
    for character, escape in _WEBVTT_ESCAPES:
        value = value.replace(character, escape)
    return value


def _attachment_path(value: RowValue, node: exp.Expr) -> str:
    """One evaluated ``path`` as the file ffmpeg attaches.

    The one field that may not be NULL: ffmpeg reads the bytes from this
    file, so an attachment without one names nothing to attach.
    """
    if not isinstance(value, str) or not value.strip():
        got = "NULL" if value is None else repr(value)
        raise _error(
            ErrorCode.UNSUPPORTED_SQL,
            f"'{ATTACHMENTS_COLUMN}.path' must name a file, got {got}",
            node,
            hint=f"path is the file to attach, e.g. {_ATTACHMENT_EXAMPLE}",
        )
    return value


def _attachment_text(value: RowValue, field_name: str, node: exp.Expr) -> str | None:
    """One evaluated ``filename``/``mimetype`` as text, or None for NULL.

    NULL leaves ffmpeg's own default in place: it names the attachment after
    the file's basename and guesses the type from it.
    """
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise _error(
            ErrorCode.UNSUPPORTED_SQL,
            f"'{ATTACHMENTS_COLUMN}.{field_name}' must be a string or NULL, "
            f"got {value!r}",
            node,
            hint=f"{field_name} is text or NULL, e.g. {_ATTACHMENT_EXAMPLE}",
        )
    return value


def _input_value(node: exp.Expr) -> object:
    """One `input('path', name => value)` value as a python scalar.

    Mirrors :func:`_sink_value`, with one addition: INPUT_OPTIONS has a
    ``"num"`` type (``framerate``, ``itsoffset``) whose value may carry a
    leading ``-`` -- ``itsoffset`` legitimately takes a negative offset.
    ``exp.Neg`` is unwrapped first, the same rule :func:`_number` applies to
    positional numeric literals. Never raises: an unusable shape comes back
    as an :class:`_Unrepresentable`, exactly like `_sink_value`, and the
    option table decides.
    """
    node = _unwrap(node)
    if not (isinstance(node, exp.Neg) and isinstance(node.this, exp.Expr)):
        return _sink_value(node)
    inner = _sink_value(_unwrap(node.this))
    if isinstance(inner, int | float) and not isinstance(inner, bool):
        return -inner
    return _Unrepresentable(_sink_describe(node))


def input_option_values(raw_options: Sequence[RawInputOption]) -> dict[str, object]:
    """One ``input()``'s trailing named options as validated scalars.

    In written order, which is the order they reach both ffprobe and ffmpeg.
    A NULL value is absence: the option is not written. The anchor falls back
    through the name node to the value node to the input()'s own path
    literal, since neither a Kwarg's ``Var`` name nor a
    ``Boolean``/``Var``/``Null`` value carries a token position.

    Public because the probe pass needs the same options lowering will write,
    and it runs first.
    """
    options: dict[str, object] = {}
    for option in raw_options:
        if isinstance(_unwrap(option.value), exp.Null):
            continue
        line, col = _pos(option.name_node, option.value, option.path_node)
        options[option.name] = validate_input_option(
            option.name, _input_value(option.value), line=line, col=col
        )
    return options


# The sink options that shape the encoder feeding a packet sink: the video
# half of the file-output vocabulary, read off the table's own scopes.
_PACKET_SINK_OPTIONS = frozenset(
    name for name, spec in SINK_OPTIONS.items() if spec.scope == "video"
)


def _join_codecs(names: Sequence[str]) -> str:
    """A codec list as a message says it."""
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " or " + names[-1]


def _check_sink_option_conflicts(
    options: dict[str, object],
    option_nodes: dict[str, exp.Expr],
    path_node: exp.Expr,
) -> None:
    """Reject two sink options that cannot both hold, once all are validated.

    ``faststart``/``movflags`` both set: -movflags either way, so one would
    silently win over the other's spelling.
    ``codec_params`` with no matching ``video_codec``: its rendered flag (see
    ``ffrwd.sink.CODEC_PARAMS_FLAGS``) is derived FROM ``video_codec``, so
    it has nothing to derive from.
    """
    if "faststart" in options and "movflags" in options:
        raise _error(
            ErrorCode.SINK_OPTION_TYPE,
            "'faststart' and 'movflags' both set -movflags",
            option_nodes["movflags"],
            fallback=path_node,
            hint="use 'faststart true' for the common case, or 'movflags' "
            "directly for anything else -- not both",
        )
    if "codec_params" in options:
        codec = options.get("video_codec")
        if not isinstance(codec, str) or codec not in CODEC_PARAMS_FLAGS:
            raise _error(
                ErrorCode.SINK_OPTION_TYPE,
                f"'codec_params' needs a matching video_codec, got {codec!r}",
                option_nodes["codec_params"],
                fallback=path_node,
                hint="set video_codec to one of: "
                + ", ".join(sorted(CODEC_PARAMS_FLAGS)),
            )
    if options.get("two_pass") is True:
        _check_two_pass(options, option_nodes, path_node)


def _check_two_pass(
    options: dict[str, object],
    option_nodes: dict[str, exp.Expr],
    path_node: exp.Expr,
) -> None:
    """The rate-control rules a ``two_pass true`` sink must satisfy.

    Two-pass exists to hit a target bitrate with a codec that has a ``-pass``
    mode, so it needs both and cannot coexist with ``crf``, which is the other
    rate control entirely.
    """
    anchor = option_nodes.get("two_pass")
    if "crf" in options:
        raise _error(
            ErrorCode.SINK_OPTION_TYPE,
            "'crf' and 'two_pass' are two different rate controls",
            option_nodes.get("crf", anchor),
            fallback=path_node,
            hint="two-pass targets a bitrate (video_bitrate); crf targets a "
            "quality level -- pick one",
        )
    codec = options.get("video_codec")
    if not isinstance(codec, str) or codec not in TWO_PASS_CODECS:
        raise _error(
            ErrorCode.SINK_OPTION_TYPE,
            f"'two_pass' needs a video_codec with a -pass mode, got {codec!r}",
            option_nodes.get("video_codec", anchor),
            fallback=path_node,
            hint="set video_codec to one of: " + ", ".join(sorted(TWO_PASS_CODECS)),
        )
    if "video_bitrate" not in options:
        raise _error(
            ErrorCode.SINK_OPTION_TYPE,
            "'two_pass' needs a video_bitrate to target",
            anchor,
            fallback=path_node,
            hint="two-pass exists to hit a bitrate, e.g. video_bitrate '2500k'",
        )


def _check_two_pass_outputs(
    options: dict[str, object], outputs: list[Output], path_node: exp.Expr
) -> None:
    """A ``two_pass`` sink must have a video output for pass 1 to analyse."""
    if options.get("two_pass") is not True:
        return
    if not any(output.type == "video" for output in outputs):
        raise _error(
            ErrorCode.SINK_OPTION_TYPE,
            "'two_pass' analyses a video stream, but this COPY selects none",
            path_node,
            hint="select a video column, or drop two_pass",
        )


def _check_two_pass_is_single_sink(sinks: list[SinkUnit], raws: list[RawSink]) -> None:
    """``two_pass`` is one COPY per script: nothing sequences per-COPY passes."""
    if len(sinks) <= 1:
        return
    for unit, raw in zip(sinks, raws):
        if unit.options.get("two_pass") is True:
            raise _error(
                ErrorCode.SINK_OPTION_TYPE,
                f"'two_pass' is not supported in a {len(sinks)}-COPY script",
                raw.path_node,
                hint="a script's COPYs share one ffmpeg command; two_pass "
                "splits it in two -- write the two-pass COPY on its own",
            )


# typed values, bindings, per-branch environment


@dataclass(frozen=True)
class _Stream:
    """One typed stream: an IR ref, the pad type it carries, and its origin.

    `source` is the probed :class:`~ffrwd.probe.StreamMeta` this stream comes
    from 1:1 — directly (a passthrough subscript) or through a chain of
    single-stream-input filters, the WHERE trim included — and is threaded
    unconditionally. A call over two or more streams (``amix``, ``overlay``)
    and a ``concat`` pad are the other kind of join: each keeps `source` only
    when every stream feeding it agrees on what it says (:func:`_agreed_source`);
    otherwise it is None, same as an unprobed input. :func:`_provenance` turns
    it into ``Output.metadata``.
    """

    ref: FrameRef
    type: StreamType
    source: StreamMeta | None = None


# Table mode only: the sentinel `_Stream.ref` for an
# outer join's NULL row, read back by `_value_to_cells`. Never a real
# FrameRef -- every well-formed one is non-empty (a node id or a "src:..."
# ref) -- so there is no ambiguity with an actual stream.
_NULL_STREAM_REF: FrameRef = ""


@dataclass(frozen=True)
class _Value:
    """What every expression lowers to: one stream, or a whole array of them.

    `is_array` is deliberately not ``len(streams) != 1``: a one-element array
    is still an array — it splats, broadcasts and subscripts — and on a
    single-track file that is the ONLY thing separating ``a.audio`` from
    ``a.audio[1]``.
    """

    type: StreamType  # element type; every element of an array agrees on it
    streams: tuple[_Stream, ...]
    is_array: bool

    def at(self, index: int) -> _Stream:
        """Element `index` of an array; the one stream of a scalar (it repeats)."""
        return self.streams[index] if self.is_array else self.streams[0]


def _scalar(stream: _Stream) -> _Value:
    return _Value(type=stream.type, streams=(stream,), is_array=False)


def _array(stream_type: StreamType, streams: Iterable[_Stream]) -> _Value:
    return _Value(type=stream_type, streams=tuple(streams), is_array=True)


@dataclass(frozen=True)
class _Column:
    """One SELECT column of a branch (or of a CTE body): its name and value.

    An array column carries every one of its streams here, so a CTE records an
    array column's LENGTH statically and a later ``<cte>.<name>[k]`` is
    bounds-checked without re-probing anything.

    `splat` matters only when `value.is_array`: True means the array IS a row
    set (a row alias's stream column, or a call over one) that a table query
    prints one row per element, like :meth:`_Lowerer._value_to_cells` already
    does outside a CTE. False means the array is a single unit -- an
    ``array_agg`` or a bare input array column -- that a table query
    broadcasts as ONE cell instead (see :meth:`_Lowerer._array_cell_broadcast`).
    Ignored for a scalar column.
    """

    name: str | None
    value: _Value
    splat: bool = True


@dataclass(frozen=True)
class _InputBinding:
    """``FROM input('x.mp4') a`` — exposes ``a.video[k]`` / ``a.audio[k]``."""

    alias: str


@dataclass(frozen=True)
class _CteBinding:
    """``FROM <cte>`` — a TABLE of the rows its body produced.

    `columns` is what the body's SELECT list named, each column holding every
    stream it carries. `rows` is the body's ROW count, which a splat array
    column carries one element per; `relation` is the branch's joined row set,
    so a column of that width reads back one element per result row and a
    cross join with a second source repeats it honestly.
    """

    name: str
    columns: tuple[_Column, ...]
    rows: int = 1
    relation: _RowRelation | None = None
    # Scalar columns of the body, name -> one value per body row. Read back
    # by position, the same way a stream column is.
    values: dict[str, tuple[RowValue, ...]] = field(default_factory=dict)


@dataclass
class _SourceBinding:
    """``FROM ffmpeg.<source>(...) a`` — exposes ONE statically-typed stream.

    Everything about the stream is known before any projection lowers: the
    registry's :class:`~ffrwd.registry.SourceFilter` says which
    type the source's single output pad carries, so ``a.video[1]``
    (video sources), ``a.audio[1]`` (audio ones), the bare array
    ``a.video`` (length 1, statically), and ``a.*`` are all answered without
    a probe — there is no file to probe, and no ``-i``: the source is a
    ZERO-INPUT filter node.

    `options` is already validated against the source's introspected option
    table (the exact same ``Registry.options`` path a tier-2 call's named
    arguments take), because that happens when the FROM clause binds, not
    when a column is read.

    Mutable on purpose: `ref` memoizes the node, which is minted lazily on
    the FIRST column access and shared by every later one. Fan-out beyond
    that is the split pass's job, exactly as for any other node, so
    ``SELECT a.video[1], hflip(a.video[1]) FROM ffmpeg.testsrc(...) a`` is one
    ``testsrc`` plus a ``split``, never two generators.
    """

    alias: str
    name: str  # the ffmpeg source filter's name, e.g. "testsrc"
    output: StreamType
    options: dict[str, object]
    ref: FrameRef | None = None

    @property
    def display(self) -> str:
        """The source as the user spelled it, for error messages."""
        return f"{FILTER_NAMESPACE}.{self.name}"


# A track-row metadata value: NULL (unprobed input, or a field this file does
# not carry) or the probed scalar. A disposition flag is the boolean case.
# Never a stream — the row IS that.
RowValue = str | int | float | bool | None

# `_TrackRow.stream` for a row that carries no track -- a chapter row, or a
# written row. Never a real stream (neither exposes a stream column at
# all), only a dataclass filler. Its ref deliberately fails `is_src()` (no
# "src:" prefix) and is not a node id either, so anything that somehow did try
# to render it fails fast with "unknown node" rather than silently wiring up
# the wrong stream.
_STREAMLESS_ROW = _Stream(ref="rows:no-stream", type="data", source=None)


@dataclass(frozen=True)
class _TrackRow:
    """One row of an ``unnest`` table: the track, plus its metadata columns.

    `stream` IS the row's stream, and its ``_Stream.source`` is the very
    ``StreamMeta`` `columns` was read from — a row's provenance and its columns
    are the same probed fact, seen twice.

    `kinds` holds every stream of a RENDITION row, by type — a manifest's
    variant may carry video and audio together — with `stream` staying the
    row's primary one (its first video stream, else its first audio one).
    Empty for every other row kind, unnest rows included.
    """

    stream: _Stream
    columns: dict[str, RowValue]
    kinds: dict[StreamType, _Stream] = field(default_factory=dict)


@dataclass(frozen=True)
class _CteRow:
    """One row of a CTE source: which row of the body's row set it is.

    The position is all a result tuple needs: it indexes both the stream
    columns' arrays and the body's value columns, which is what makes
    ``x.n`` read the value this very row computed.
    """

    position: int


# What one result row holds per FROM alias: a track (or a gap, where an outer
# join found no counterpart) for a row table, a position for a CTE source.
_RowTuple = dict[str, "_TrackRow | _CteRow | None"]


@dataclass
class _RowRelation:
    """One branch's joined row set: every row source, aligned.

    `tuples` is the relation itself — one dict per result ROW, mapping each row
    alias to that row's track, or to ``None`` where an outer join left a gap,
    and each CTE alias to the body row it took. All of a branch's row sources
    share this one object, which is what keeps
    ``a`` and ``b`` aligned: element `i` of each is the pair the
    join made, so the existing zip/broadcast machinery wires the right streams
    together without learning that joins exist.

    Row order is the join's, never sorted implicitly: the
    LEFT side's order, then — for a FULL join only — the unmatched right rows
    in their own order. `keys` remembers which columns each side was matched
    on, so a NULL track can say what it failed to match.
    """

    aliases: list[str] = field(default_factory=list)
    tuples: list[_RowTuple] = field(default_factory=list)
    keys: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class _RowBinding:
    """``FROM ..., unnest(<input>.<type>) t`` — a compile-time TABLE.

    `rows` is this alias's column of the branch's joined relation, in ROW
    ORDER: the surviving row set, one entry per result row, ``None`` where an
    outer join found no counterpart. It is what the WHERE predicate and the
    ORDER BY rewrite (both act on the shared :class:`_RowRelation`, so every
    alias stays aligned), and both happen once per branch before any projection
    lowers. Selecting ``t`` over N surviving rows is an N-element array in
    that order, which is the same array value a bare ``f.audio`` produces — the
    row model and the array model are one mechanism.

    `source` is the INPUT alias the tracks belong to. Everything downstream
    (the ``-i``, its WHERE window, provenance) keys off THAT alias, not the row
    one: a row table takes no input slot of its own. `values` is set instead
    for a WRITTEN row source (a struct row table in FROM), whose rows come
    from the query rather than from a probe; it has no input alias and no
    streams.
    """

    alias: str
    source: str
    column: str  # the array that was unnested: video/audio/subtitle/data
    type: StreamType
    relation: _RowRelation
    values: RawValuesTable | None = None

    @property
    def rows(self) -> tuple[_TrackRow | None, ...]:
        return tuple(_track_of(row, self.alias) for row in self.relation.tuples)

    @property
    def streamless(self) -> bool:
        """True for rows that carry no track: records, and written rows."""
        return self.values is not None or self.column in RECORD_ARRAY_COLUMNS

    @property
    def record(self) -> str:
        """The record these rows are one of. Only a record array has one."""
        return RECORD_ELEMENTS[self.column]

    @property
    def schema(self) -> dict[str, RowColumnType]:
        """The columns these rows expose, in declaration (or written) order."""
        if self.column == RENDITION_COLUMN:
            return _RENDITION_SCHEMA
        return (
            self.values.schema() if self.values is not None else ROW_SCHEMAS[self.column]
        )

    @property
    def star(self) -> tuple[str, ...]:
        """What ``<alias>.*`` expands to: the scalar columns, in order."""
        if self.column == RENDITION_COLUMN:
            return tuple(_RENDITION_SCHEMA)
        if self.values is not None:
            return self.values.columns
        return ROW_STAR_COLUMNS[self.column]

    @property
    def readonly(self) -> frozenset[str]:
        """The columns a query may not assert. A written row has none."""
        if self.column == RENDITION_COLUMN:
            return _RENDITION_READONLY
        if self.values is not None:
            return frozenset()
        return ROW_READONLY_FIELDS[self.column]

    @property
    def exposes(self) -> str:
        """How a rejection names this row source's column list."""
        listed = ", ".join(sorted(self.schema))
        if self.values is not None:
            return f"'{self.alias}' exposes {listed}"
        return f"{self.column} track rows expose {listed}"


_Binding = _InputBinding | _CteBinding | _SourceBinding | _RowBinding

# Metadata tag overrides for one query: probed StreamMeta identity -> the keys
# its output streams set, with None for a key the query clears.
_TagOverrides = dict[int, dict[str, str | None]]

# What a query being lowered can tag. "sink" is a query that writes a file:
# per-stream tags where it has track rows, container tags where it has none.
# "rows" is a CTE body -- per-stream tags only, since a CTE has no container.
_TagScope = Literal["sink", "rows"]

# One query's per-track disposition writes: probed StreamMeta identity -> the
# flags that stream's output sets, in declared order. An empty tuple is the
# written `'0'`: every flag off.
_DispositionOverrides = dict[int, tuple[str, ...]]


def _track_of(row: _RowTuple, alias: str) -> _TrackRow | None:
    """One result row's track for a row alias; None for a gap or a CTE row."""
    entry = row.get(alias)
    return entry if isinstance(entry, _TrackRow) else None


def _cte_row_count(
    columns: Iterable[_Column], values: dict[str, tuple[RowValue, ...]]
) -> int:
    """How many rows a CTE body produced: the width of its row-set columns.

    A splat array column carries one stream per body row, so its length IS the
    body's row count; a body with none (a single input row, a UNION ALL's
    concat, a broadcast array) is one row.
    """
    widths = [
        len(column.value.streams)
        for column in columns
        if column.splat and column.value.is_array
    ]
    widths += [len(one) for one in values.values()]
    return max(widths) if widths else 1


def _map_key(name: str) -> str:
    """The key a folded map path names, else the column name itself."""
    ref = map_ref(name)
    return ref[1] if ref is not None else name


def _tags_to_cell(tags: dict[str, str]) -> ArrayCell:
    """One tag map as an array cell of ``(key,value)`` records, in key order."""
    return ArrayCell(
        elements=tuple(RecordCell(fields=(key, tags[key])) for key in sorted(tags))
    )


def _tag_cell(row: _TrackRow | None) -> CellValue:
    """One row's whole tag map as a cell; NULL for an outer join's gap."""
    if row is None:
        return None
    source = row.stream.source
    return _tags_to_cell({} if source is None else source.metadata)


def _flags_to_cell(flags: dict[str, bool]) -> ArrayCell:
    """One disposition as an array cell of ``(key,set)`` records, in flag order.

    The key set is CLOSED, so every declared flag is an entry; one this ffmpeg
    did not report reads NULL, the way an absent tag does.
    """
    return ArrayCell(
        elements=tuple(
            RecordCell(fields=(key, flags.get(key))) for key in DISPOSITION_KEYS
        )
    )


def _disposition_cell(row: _TrackRow | None) -> CellValue:
    """One row's whole flag map as a cell; NULL where nothing was probed."""
    if row is None or row.stream.source is None:
        return None
    return _flags_to_cell(row.stream.source.disposition)


def _row_star_error(
    binding: _RowBinding, anchor: exp.Expr, select: exp.Select
) -> FfrwdError:
    """``*`` over rows in a MEDIA query: fields are not output streams.

    A star over a row table expands the record's fields, the same as it does in
    a table query. Fields have nowhere to go on an ffmpeg command line; for a
    track row the stream the query means is the bare alias, and a chapter row
    has no stream at all.
    """
    printed = "a bare SELECT prints the fields as a table"
    hint = (
        f"these rows carry no stream; {printed}"
        if binding.streamless
        else f"the row is the stream: select {binding.alias}; {printed}"
    )
    return _error(
        ErrorCode.UNSUPPORTED_SQL,
        f"'*' over the rows of '{binding.alias}' expands their fields, and a "
        "SELECT column is an output stream",
        anchor,
        fallback=select,
        hint=hint,
    )


def _row_columns(meta: StreamMeta, column: str) -> dict[str, RowValue]:
    """One probed stream's row columns.

    Three sources, one table: the stream's own tags (``StreamMeta.metadata``)
    and its disposition flags each become one column per key under a folded
    path name, everything else comes from a field of the StreamMeta itself. An
    absent field is NULL, and so is a key the file does not carry — which is
    the whole NULL story, there is no other way for a row column to be null.

    ``index`` is +1'd: ``StreamMeta.index`` is the 0-based per-type index the
    IR ref uses, and the SQL surface is 1-based everywhere (``f.audio[1]``), so
    ``WHERE t.index = 1`` and ``f.audio[1]`` name the same track.

    The enriched fields (``codec``, ``channels``, ``channel_layout``,
    ``bitrate``, ``duration``, ``color_transfer``) are read through
    :func:`getattr` deliberately: a StreamMeta built without them yields NULL
    columns rather than an AttributeError -- exactly what an unprobed field
    yields anyway.
    """
    schema = ROW_SCHEMAS[column]
    values: dict[str, RowValue] = {
        "index": meta.index + 1,
        "width": meta.width,
        "height": meta.height,
        "fps": meta.fps,
        "sample_rate": meta.sample_rate,
    }
    for name in ("codec", "channels", "channel_layout", "bitrate", "duration",
                 "color_transfer"):
        probed = getattr(meta, name, None)
        values[name] = probed if isinstance(probed, str | int | float) else None
    columns = {name: values.get(name) for name in schema if name not in MAP_COLUMNS}
    columns.update({tag_path(key): value for key, value in meta.metadata.items()})
    columns.update(
        {
            map_path(DISPOSITION_COLUMN, key): meta.disposition[key]
            for key in DISPOSITION_KEYS
            if key in meta.disposition
        }
    )
    return columns


def _record_columns(result: ProbeResult, column: str) -> list[dict[str, RowValue]]:
    """One container record array's rows, each keyed by the record's fields.

    The row tables that are not built from a `StreamMeta`: a chapter comes
    from ffprobe's own chapter list, an attachment from the streams ffprobe
    types as attachments, and a cue from the WebVTT document ffrwd parsed
    (:func:`ffrwd.probe.parse_webvtt`). A file carrying none of a kind
    reads zero rows.
    """
    if column == CHAPTERS_COLUMN:
        return [
            {
                "index": chapter.index,
                "title": chapter.title,
                "start_t": chapter.start_t,
                "end_t": chapter.end_t,
            }
            for chapter in result.chapters
        ]
    if column == ATTACHMENTS_COLUMN:
        return [
            {
                "index": attachment.index,
                "filename": attachment.filename,
                "mimetype": attachment.mimetype,
            }
            for attachment in result.attachments
        ]
    return [
        {"index": cue.index, "text": cue.text, "start_t": cue.start_t, "end_t": cue.end_t}
        for cue in result.cues
    ]


def _join_keys(on: exp.Expr) -> dict[str, list[str]]:
    """Which columns each row alias was matched on, from a JOIN's ON predicate.

    Bookkeeping for one message only: a NULL track says what it failed to
    match (``no 'b' row matched a.tags.language='fra'``), and that needs the key
    columns of the side that DID match. Order is written order, deduplicated.
    """
    keys: dict[str, list[str]] = {}
    for sub in on.walk():
        if not isinstance(sub, exp.Column):
            continue
        table_node = sub.args.get("table")
        if table_node is None:
            continue
        names = keys.setdefault(_fold(table_node), [])
        name = _fold(sub.this)
        if name not in names:
            names.append(name)
    return keys


# The fill each track type takes when an outer join leaves a gap. Quoted
# verbatim in the NULL-track hint, so it is spelled the way a user would paste
# it. `data` is absent deliberately: nothing generates a data track, so there
# is no fill to suggest.
_FILL_SPELLINGS: dict[StreamType, str] = {
    "audio": f"{FILTER_NAMESPACE}.anullsrc()",
    "video": f"{FILTER_NAMESPACE}.color()",
    "subtitle": f"{MACRO_NAMESPACE}.empty_captions()",
}

_COALESCE_HINT = (
    "COALESCE fills an outer join's gaps: COALESCE(b, "
    f"{FILTER_NAMESPACE}.anullsrc(duration => 2)) for audio, "
    f"{FILTER_NAMESPACE}.color() for video, "
    f"{MACRO_NAMESPACE}.empty_captions() for captions"
)


# The compile-time row predicate evaluator.
#
# Every column of a track row is PROBED metadata, so a predicate over rows is
# decidable here, at compile time, and never reaches ffmpeg -- the way a
# `WHERE t BETWEEN` vanishes into `-ss`/`-to`. Standard SQL three-valued logic
# throughout: a comparison against NULL is UNKNOWN (python `None`), AND/OR/NOT
# are Kleene, and WHERE keeps a row only when its predicate came back TRUE, so
# "NULL matches nothing" falls out rather than being a rule of ours.
#
# `resolve` already shape- and type-checked everything below; the rejections
# here are defensive re-checks raising the same FfrwdError resolve would.


def _kleene_and(left: bool | None, right: bool | None) -> bool | None:
    if left is False or right is False:
        return False
    if left is None or right is None:
        return None
    return True


def _kleene_or(left: bool | None, right: bool | None) -> bool | None:
    if left is True or right is True:
        return True
    if left is None or right is None:
        return None
    return False


# `<literal> OP <column>` is the same predicate as `<column> OP' <literal>`
# with the ordering operators inverted; the two equality ones are their own
# mirror. sqlglot does NOT normalize operand order at parse time (the same
# thing `_time_bounds` handles for time bounds), so the mirror is explicit.
_MIRRORED_COMPARISONS: dict[type[exp.Expr], type[exp.Expr]] = {
    exp.EQ: exp.EQ,
    exp.NEQ: exp.NEQ,
    exp.GT: exp.LT,
    exp.GTE: exp.LTE,
    exp.LT: exp.GT,
    exp.LTE: exp.GTE,
}


def _sort_key(value: RowValue) -> tuple[int, str, float]:
    """A total, type-stable sort key for one non-NULL row-column value.

    A column's type is static, so the two branches never actually compete
    within one sort — the tuple shape is what keeps the comparison total
    anyway, rather than letting a surprising value raise a TypeError deep
    inside ``list.sort``.
    """
    if isinstance(value, str):
        return (0, value, 0.0)
    return (1, "", float(value if value is not None else 0))


def _compare(node: exp.Expr, left: RowValue, right: RowValue) -> bool | None:
    """One comparison under SQL NULL semantics; None is UNKNOWN, never False."""
    if left is None or right is None:
        return None
    if isinstance(node, exp.EQ):
        return left == right
    if isinstance(node, exp.NEQ):
        return left != right
    if isinstance(left, str) != isinstance(right, str):
        # Unreachable via resolve (a column's type is static and the literal
        # was checked against it), and an ordering comparison across the two
        # would be a python TypeError rather than an answer.
        return None
    if isinstance(node, exp.GT):
        return left > right  # type: ignore[operator]
    if isinstance(node, exp.GTE):
        return left >= right  # type: ignore[operator]
    if isinstance(node, exp.LT):
        return left < right  # type: ignore[operator]
    return left <= right  # type: ignore[operator]


@dataclass
class _Env:
    """Everything one SELECT branch resolves names against."""

    bindings: dict[str, _Binding] = field(default_factory=dict)
    # CTE name -> its WHERE window. CTE-ONLY: an INPUT alias's window is a
    # property of its `-i`, not of this branch, so `_collect_trims` records it
    # in `Graph.input_trims` instead and no filter trim is ever spliced for it.
    # Either half may be None (an open-ended window).
    trims: dict[str, tuple[int | float | None, int | float | None]] = field(
        default_factory=dict
    )
    # base stream ref -> its trimmed ref, so one filter trim is shared by every
    # consumer of that stream inside this branch (CTE-only, as above).
    trimmed: dict[FrameRef, FrameRef] = field(default_factory=dict)
    # The branch's joined row set, or None until its first
    # `unnest` binds. There is at most ONE: every row table of a branch joins
    # into it, comma sources included (the comma between two unnests is the
    # bounded cross join), so all row aliases stay aligned by construction.
    relation: _RowRelation | None = None
    # True for a branch that aggregates -- a GROUP BY, an `array_agg`, or both.
    # Its scalar columns are group-constants (resolve's grouping check proves
    # that), so they tag the CONTAINER rather than the tracks.
    grouped: bool = False
    # The GROUP BY keys that read a track-row column: the ones that actually
    # partition the relation. An input-level or constant key has the same value
    # for every tuple and leaves one group.
    group_keys: tuple[exp.Expr, ...] = ()
    # Input alias -> the `-i` each ROW of `relation` seeks, in row order. Set
    # only for a window whose bounds read a row column with no fan-out TO: the
    # rows stay in one graph, so each takes its own copy of the input with its
    # own `-ss`/`-to`, and every stream column of the alias reads one stream
    # per row.
    row_inputs: dict[str, list[str]] = field(default_factory=dict)


# ExpandCtx


class _NodeFactory:
    """Mints ``n1, n2, ...`` node ids into a graph, in creation order."""

    def __init__(self, graph: Graph) -> None:
        self._graph = graph
        self._counter = 0

    def node(
        self,
        filter: str,
        args: dict[str, object],
        inputs: list[FrameRef],
        outputs: list[StreamType],
        *,
        reads_annotations: bool = False,
    ) -> FrameRef:
        self._counter += 1
        node_id = f"n{self._counter}"
        self._graph.nodes[node_id] = Node(
            id=node_id,
            filter=filter,
            args=dict(args),
            inputs=list(inputs),
            outputs=list(outputs),
            reads_annotations=reads_annotations,
        )
        return node_id


# the lowering walk


class _Lowerer:
    def __init__(
        self,
        res: Resolved,
        probes: dict[str, ProbeResult | None],
        registry: Registry | None,
        fanout_index: int = 0,
        *,
        fanout_sinks: bool = False,
        on_warning: OnWarning | None = None,
        describes: dict[str, Described] | None = None,
        invoke: Invoke = wasm_invoke,
        probe_failures: Mapping[str, ProbeFailure | None] | None = None,
        probe_source: ProbeSource = wasm_probe_source,
    ) -> None:
        self.res = res
        self.probes = probes
        # Why an alias in `probes` maps to None, when there is a specific
        # answer -- unset (or no answer for this alias) reads the same as an
        # explicit None, both meaning "say the old, unqualified thing".
        self.probe_failures: Mapping[str, ProbeFailure | None] = probe_failures or {}
        self.registry = registry
        self.describes = describes or {}
        self.invoke = invoke
        self.probe_source = probe_source
        # (module, function, sorted args) -> result, so two calls with the
        # same arguments run the module once per compile.
        self._invoke_cache: dict[tuple[str, str, tuple[tuple[str, object], ...]], object] = {}
        self.on_warning = on_warning
        self.graph = Graph(input_paths=list(res.input_paths), sources=dict(res.sources))
        self.ctx = _NodeFactory(self.graph)
        self.cte_columns: dict[str, tuple[_Column, ...]] = {}
        # The VALUE columns of each CTE body, name -> column -> one value per
        # body row. Filled as each body lowers, read when its alias binds.
        self.cte_values: dict[str, dict[str, tuple[RowValue, ...]]] = {}
        # The value columns the query being lowered has collected so far.
        self.branch_values: dict[str, tuple[RowValue, ...]] = {}
        # Inputs this pass minted itself (`ffrwd.empty_captions()`),
        # alias -> its INTERNAL input options. Merged into `Graph.input_options`
        # by `_lower_input_options`, which is the only writer of that field.
        self.minted_input_options: dict[str, dict[str, object]] = {}
        # The tag columns of the query being lowered; reset per query, since
        # two COPYs may tag the same track differently.
        self.tags: _TagOverrides = {}
        # The tag columns of every CTE body, harvested as each one lowers and
        # kept for the whole pass: a CTE's streams carry their tags into
        # whichever sink maps them, under that sink's own tags.
        self.cte_tags: _TagOverrides = {}
        # The disposition columns of the query being lowered, and of every CTE
        # body, on the same two-scope plan the tags follow.
        self.dispositions: _DispositionOverrides = {}
        self.cte_dispositions: _DispositionOverrides = {}
        # The same for the CONTAINER tags of the file being written, key ->
        # value, None meaning "clear this key".
        self.container_tags: dict[str, str | None] = {}
        # The chapter list of the file being written: the ffmpeg input index
        # its chapters come from, `ir.NO_CHAPTERS` for a written NULL, and None
        # while no `chapters` column has been read. Reset per COPY.
        self.chapters: int | None = None
        # The global tags of the file being written: the ffmpeg input index
        # they are copied from, `ir.NO_METADATA` for none, and None while no
        # `tags` column has named a source. Reset per COPY.
        self.metadata: int | None = None
        # The files the file being written carries, in written order. Empty
        # while no `attachments` column has been read. Reset per COPY.
        self.attachments: list[Attachment] = []
        # Output fan-out: which row of the sink's relation THIS run binds, the
        # sink's TO expression once it is known to reference a row column, and
        # the pinned row / its branch environment once `_pin_fanout_row` runs.
        # `fanout_count` is the relation's surviving row count, i.e. how many
        # FILES the query writes; None until a pin happens, which is what tells
        # `lower_commands` this was not a fan-out query at all.
        self.fanout_index = fanout_index
        # True -> every fan-out row becomes a SinkUnit of THIS graph (one
        # command, several output files) and its time window rides that unit
        # instead of the shared `-i`. False -> `fanout_index` alone binds, one
        # graph per row, which is the `&&` chain a stream-copy trim needs.
        self.fanout_sinks = fanout_sinks
        # The input windows the row being lowered named, alias -> (start, end).
        # Reset per row; harvested into that row's `SinkUnit.window`.
        self.fanout_windows: dict[str, tuple[float | None, float | None]] = {}
        # True once a row named two windows at once: no single output seek
        # says that, so the fan-out falls back to the chain.
        self.fanout_window_conflict = False
        self.fanout_expr: exp.Expr | None = None
        # Sticky across sinks, unlike `fanout_expr`: the loudnorm2 limits ask
        # whether ANY COPY of the script fanned out.
        self.fanout_seen = False
        self.fanout_row: _RowTuple = {}
        self.fanout_env: _Env | None = None
        # The branch relation a WITH option read once per row runs over, and
        # the env it is evaluated against. Both are the LAST branch lowered,
        # which is the one this COPY's options belong to.
        self.sink_rows: list[_RowTuple] = []
        self.sink_env: _Env | None = None
        self.fanout_count: int | None = None
        # True when the pin partitioned by a GROUP BY key rather than by row,
        # so a collision message names groups.
        self.fanout_grouped = False
        # The COPY whose query is lowering: the node its row-count rejection
        # anchors on, and the path it names. Both None for a bare SELECT,
        # which names no destination at all.
        self.sink_anchor: exp.Expr | None = None
        self.sink_path: str | None = None
        # The rows file this COPY writes, for a destination that IS one, and
        # "" for every other sink. A rows file has no ffmpeg output at all:
        # the sidecar writes it, and the COPY makes no unit.
        self.rows_file = ""
        # Every track minted from a module's rows, so a sink holding one can
        # be given the subtitle codec its container needs.
        self.rows_tracks: list[FrameRef] = []
        # A per-row `-i` this pass minted for a row-bounded window: the minted
        # alias -> the input alias it copies. Path, probe and options are that
        # alias's; only the window differs.
        self.row_input_source: dict[str, str] = {}
        # True once any branch minted one, so the one-row-per-file rejection
        # can name the two ways a windowed row set reaches a destination --
        # including when the windows are in a CTE body.
        self.row_window_seen = False
        # True for the whole duration of `run_table()`; `run()` never sets it.
        # Table mode changes exactly one thing about the stream machinery it
        # otherwise reuses verbatim: an outer join's NULL row is an empty cell
        # rather than a rejection (see `_row_stream`).
        self.table_mode = False
        # The manifest format ('hls'/'dash') of the COPY currently lowering,
        # or None for every other destination. Under it a multi-row relation
        # is accepted -- each row one variant map entry -- and an outer
        # join's NULL row is that entry's absent stream kind.
        self.manifest: str | None = None
        # True while the COPY currently lowering targets a sink that reads
        # rows off the SELECT list rather than named stream parameters: a
        # multi-row relation is accepted here too, the way a manifest's is.
        self.row_reading_sink = False
        # Node id -> one {"row": int, "rendition": {...}} per pad, in the
        # same row-major order `_packet_sink_pads` builds a row-reading
        # sink's pads in. Read there to fold `row`/`rendition` into each
        # pad's dict; empty for the old, stream-parameter sink form.
        self.row_reading_sink_pads: dict[str, list[dict[str, object]]] = {}
        # The same sinks' rows, for cutting a per-row option list to the rows
        # that carry the option's kind.
        self.row_reading_sink_rows: dict[str, list[_VariantRow]] = {}

    # -- entry point ------------------------------------------------------

    def run(self) -> Graph:
        """Lower every CTE/view once, then one :class:`SinkUnit` per COPY.

        The bindings come first and are shared: ``res.ctes`` holds a script's
        views AND every COPY's own ``WITH``, in written order, and
        each is lowered into THIS graph exactly once. A view read by three
        COPYs therefore mints its nodes once and hands the same refs to all
        three — the fan-out is the split pass's ordinary business, which is
        the whole point of the ABR ladder compiling to one ffmpeg command.

        ``res.select`` / ``res.branches`` are read for the BARE-SELECT case
        only (a query with no COPY at all, which is the one unit whose path
        is None). When there are sinks they are just a mirror of ``sinks[0]``
        and walking them again would lower the first group twice.
        """
        for name, body in self.res.ctes.items():
            self.branch_values = {}
            self.cte_columns[name] = tuple(
                self._lower_query(union_branches(body), body, tags="rows")
            )
            self.cte_values[name] = self.branch_values
            self.branch_values = {}
            self._harvest_cte_tags(body)
            self._harvest_cte_dispositions(body)
        if self.res.sinks:
            self.graph.sinks = self._lower_sinks()
            if self.fanout_count is None:
                _check_two_pass_is_single_sink(self.graph.sinks, self.res.sinks)
        else:
            columns = self._lower_query(self.res.branches, self.res.select, tags="sink")
            self.graph.sinks = [
                SinkUnit(
                    outputs=_outputs(
                        columns, self._layered_tags(), self._layered_dispositions()
                    ),
                    tags=dict(self.container_tags),
                    chapters=self.chapters,
                    metadata=self.metadata,
                    attachments=list(self.attachments),
                )
            ]
        self._check_loudnorm2()
        self.graph.input_options = self._lower_input_options()
        return self.graph

    def _check_loudnorm2(self) -> None:
        """The v1 limits on ``ffrwd.loudnorm2``.

        It is not one filter among others: its presence turns the whole
        compile into a two-command sequence with a shell handoff in the
        middle. Everything that would need a SECOND sequencing rule on top of
        that -- a second loudnorm2, a ``two_pass`` sink, a fan-out TO -- is
        closed rather than guessed at. Counted over NODES, so a call
        broadcast across an audio array is caught as the several it is.

        The fan-out rejection comes FIRST: a fan-out mints the call once per
        file it writes, so the count would otherwise report a multiplicity the
        query text does not show.
        """
        anchors = [(raw.path_expr, raw.path_node) for raw in self.res.sinks]
        anchor, fallback = anchors[0] if anchors else (self.res.select, self.res.select)
        count = sum(1 for n in self.graph.nodes.values() if n.filter == loudnorm.FILTER)
        if count == 0:
            return
        if self.fanout_seen:
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                "ffrwd.loudnorm2() and a fan-out TO cannot both be set",
                anchor,
                fallback=fallback,
                hint="a TO expression writes one file per row, each needing its "
                "own measuring pass; write a quoted TO path",
            )
        if count > 1:
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"one ffrwd.loudnorm2() per query, got {count}",
                anchor,
                fallback=fallback,
                hint="each one needs its own measuring pass; write one query per "
                "stream you are normalizing",
            )
        for unit, raw in zip(self.graph.sinks, self.res.sinks):
            if unit.options.get("two_pass") is True:
                raise _error(
                    ErrorCode.UNSUPPORTED_SQL,
                    "'two_pass' and ffrwd.loudnorm2() cannot both be set",
                    raw.path_node,
                    hint="both compile to a command sequence of their own; "
                    "normalize the audio in a separate COPY",
                )

    # -- the COPY sink ----------------------------------

    def _lower_sinks(self) -> list[SinkUnit]:
        """One :class:`SinkUnit` per COPY — or per fan-out ROW/GROUP.

        A fan-out COPY is alone in its script (the parser sees to that) and
        writes one file per surviving row, so under `fanout_sinks` it lowers
        once per row into THIS graph: shared streams are minted once and the
        split pass fans them out across the units, exactly as it does for a
        view several COPYs read. The row COUNT is a property of the probed
        relation, so it comes back from the first pass rather than being known
        up front.
        """
        units: list[SinkUnit] = []
        for raw in self.res.sinks:
            first = self._lower_sink(raw)
            if first is not None:
                units.append(first)
            count = self.fanout_count
            if not self.fanout_sinks or count is None:
                continue
            for index in range(1, count):
                self.fanout_index = index
                more = self._lower_sink(raw)
                if more is not None:
                    units.append(more)
            self.fanout_index = 0
        return units

    def _lower_sink(self, raw: RawSink) -> SinkUnit | None:
        """One COPY: its own query lowered, its options validated.

        Each COPY carries a whole query of its own (``RawSink.query`` /
        ``.branches``, already validated by resolve), so a sink unit is that
        query's SELECT list plus the destination it names.

        Anchoring, VERIFIED against sqlglot 30.17: the option NAME (an
        ``exp.Var``) carries no token position, and neither does a ``Boolean``
        / ``Var`` / ``Null`` value, so the anchor falls back through the name
        node to the value node to the path literal — which at least keeps
        every rejection on (or just above) the ``WITH`` block.

        A ``TO (<expression>)`` reaching here is a fan-out sink exactly when it
        reads a row column -- any row source, ``unnest``, a struct row table,
        or ``generate_series``; that decision is made FIRST, since it changes
        how the wrapped query lowers (one pinned row, per-row seek bounds).

        None for a COPY that writes a module's ROWS: the sidecar writes that
        file itself, so there is no ffmpeg output and no unit. None for a
        SINK MODULE destination too: the module's own effects are the output,
        and lowering the query is what put its node in the graph.
        """
        self.fanout_expr = (
            raw.path_expr
            if raw.path_expr is not None
            and references_row_alias(raw.path_expr, set(self.res.row_aliases))
            else None
        )
        self.fanout_seen = self.fanout_seen or self.fanout_expr is not None
        self.sink_anchor = raw.path_expr if raw.path_expr is not None else raw.path_node
        self.sink_path = raw.path
        self.fanout_windows = {}
        self.chapters = None
        self.metadata = None
        self.attachments = []
        self.rows_file = self._rows_file(raw)
        self.manifest = _manifest_format(raw)
        self._check_manifest_target(raw)
        self.row_reading_sink = bool(raw.module_sink) and self.res.wasm[
            raw.module_sink
        ].reads_rows_from_select
        first_sink = len(self.graph.module_sinks)
        columns = self._lower_query(list(raw.branches), raw.query, tags="sink")
        if self.rows_file:
            return None
        if raw.module_sink:
            self._lower_module_sink(raw, self.graph.module_sinks[first_sink:])
            return None
        variant_rows: list[_VariantRow] | None = None
        if self.manifest is not None:
            variant_rows, columns = self._manifest_rows(columns, raw)
        options: dict[str, object] = {}
        option_nodes: dict[str, exp.Expr] = {}
        for option in raw.options:
            if variant_rows is not None and option.name in MANIFEST_MAP_OPTION.values():
                raise self._hand_written_map_error(option, columns, variant_rows, raw)
            if isinstance(_unwrap(option.value), exp.Null):
                # NULL is absence: the option is not written, the encoder's /
                # muxer's own default applies, and the option table never
                # sees the value.
                continue
            written = self._sink_option_value(option, raw)
            if written is None:
                continue  # a NULL element, absence like any other NULL
            line, col = _pos(option.name_node, option.value, raw.path_node)
            options[option.name] = _validated_option(
                option.name, written, line=line, col=col
            )
            option_nodes[option.name] = option.value
        _check_sink_option_conflicts(options, option_nodes, raw.path_node)
        self._check_manifest_options(options, option_nodes, columns, variant_rows, raw)
        outputs = _outputs(columns, self._layered_tags(), self._layered_dispositions())
        if not outputs:
            # An empty column contributes nothing and only warns, but a sink
            # left with nothing at all would write a file with no streams in
            # it, which is never what anyone meant.
            raise _error(
                ErrorCode.STREAM_NOT_FOUND,
                f"'{raw.path}' would have no streams: every column selected is empty",
                raw.path_node,
                hint="the file has none of the tracks this query names; check the "
                "input, or select * to take whatever it holds",
            )
        _check_two_pass_outputs(options, outputs, raw.path_node)
        if variant_rows is not None:
            _compress_manifest_lists(options, variant_rows)
        self._check_per_track_options(options, option_nodes, outputs, raw)
        path = raw.path
        if raw.path_expr is not None:
            self._check_fanout_options(options, raw)
            path = self._sink_path(raw)
        if variant_rows is not None and path is not None:
            path = self._derive_manifest(
                options, option_nodes, columns, variant_rows, outputs, path, raw
            )
        self._codec_for_rows_track(options, outputs, path)
        return SinkUnit(
            outputs=outputs,
            path=path,
            options=options,
            tags=dict(self.container_tags),
            window=self._fanout_window(),
            chapters=self.chapters,
            metadata=self.metadata,
            attachments=list(self.attachments),
        )

    def _rows_file(self, raw: RawSink) -> str:
        """The rows file this COPY writes, or "" for a COPY that writes media.

        A destination spelled as a rows file writes ONE thing: the annotation
        column a module's call projects. Anything else in the SELECT list is a
        rejection -- a rows file has no track to put a stream in.
        """
        path = raw.path
        if path is None or not path.lower().endswith(_ROWS_SUFFIX):
            return ""
        written = [
            column
            for branch in (raw.branches or [raw.query])
            if isinstance(branch, exp.Select)
            for column in branch.expressions
            if isinstance(column, exp.Expr)
        ]
        sole = _unwrap(written[0]) if len(written) == 1 else None
        if sole is not None and self._rows_projection(sole) is not None:
            return path
        raise _error(
            ErrorCode.UNSUPPORTED_SQL,
            f"'{path}' is a rows file, and this query writes "
            f"{len(written)} columns to it",
            raw.path_node,
            hint="a rows file holds one module's annotation column and nothing "
            "else; write the streams to a media file of their own",
        )

    def _lower_module_sink(self, raw: RawSink, sink_nodes: list[str]) -> None:
        """The destination side of a COPY whose TO names a sink function.

        A FRAME sink takes no WITH options: the call's own value arguments
        configure it, and the frames reach it decoded. A PACKET sink consumes
        the encoder's own output, so the COPY's video encoder options -- the
        same spellings a file sink takes -- shape the stream the feeding
        ffmpeg encodes onto the edge, and the codec answers to the list the
        module's describe names. `sink_nodes` are the sink's graph nodes this
        COPY just lowered.
        """
        declared = self.res.wasm[raw.module_sink]
        described = self.describes.get(declared.module)
        if described is None or not described.packet_sink:
            if raw.options:
                raise _error(
                    ErrorCode.UNSUPPORTED_SQL,
                    "a sink destination takes no WITH options",
                    raw.options[0].name_node,
                    fallback=raw.path_node,
                    hint=f"the sink's own value arguments configure it: "
                    f"{declared.name}(<values>)",
                )
            return
        if declared.reads_rows_from_select:
            present = {
                ref_type(self.graph, ref)
                for node in sink_nodes
                for ref in self.graph.nodes[node].inputs
            }
            scopes = tuple(kind for kind in ("video", "audio") if kind in present)
        else:
            scopes = ("video", "audio") if "audio" in declared.stream_kinds else ("video",)
        options = self._packet_sink_options(declared, raw, scopes)
        for node in sink_nodes:
            per_node = dict(options)
            rows = self.row_reading_sink_rows.get(node)
            if rows is not None:
                # A per-row value was read over every row; the pads of its
                # kind are the rows that carry one, as at a manifest.
                _compress_manifest_lists(per_node, rows)
            self.graph.packet_sinks[node] = self._packet_sink_pads(
                node, per_node, raw, declared, described
            )
            if described.rows_schema is not None:
                # The rows are the sink's product, and no path names a home
                # for them: they ride the hosting process's own stdout.
                self.graph.rows_sinks[node] = RowsSink(container=_ROWS_CONTAINER)

    def _packet_sink_pads(
        self,
        node: str,
        options: dict[str, object],
        raw: RawSink,
        declared: WasmFunction,
        described: Described,
    ) -> list[dict[str, object]]:
        """The sink's encoder options resolved PAD BY PAD.

        A value read once per row is a list, one element per rendition in the
        order the rows were gathered; every other value shapes every pad the
        same way. The counts have to line up, for the same reason a file's do:
        the rows and the pads are two accounts of one ladder -- counted over
        the pads of the option's OWN kind, since a video option says nothing
        about an audio pad standing beside them.

        A pad of a kind the WITH wrote no option for, whose stream is an
        unmodified probed one the sink already accepts, copies instead of
        re-encoding (:meth:`_copies_onto_sink`) -- a written encoder option
        of that kind is a request to encode; a row-reading sink's pads
        also carry `row` and `rendition` (:attr:`row_reading_sink_pads`),
        empty for the old, stream-parameter form.
        """
        inputs = self.graph.nodes[node].inputs
        kinds = [ref_type(self.graph, ref) for ref in inputs]
        for name, value in options.items():
            if not isinstance(value, list):
                continue
            scope = SINK_OPTIONS[name].scope
            of_kind = sum(1 for kind in kinds if kind == scope)
            if len(value) != of_kind:
                raise _error(
                    ErrorCode.ROW_COUNT_MISMATCH,
                    f"sink option {name!r} is read once per row over "
                    f"{len(value)} rows, and this sink reads "
                    f"{of_kind} {scope} "
                    f"{'stream' if of_kind == 1 else 'streams'}",
                    raw.path_node,
                    hint=_PER_TRACK_OPTION_HINT,
                )
        written = {option.name for option in raw.options}
        row_meta = self.row_reading_sink_pads.get(node)
        # Each pad takes the options shaping ITS OWN encoder, a per-row value
        # indexed by the pad's position among the pads of that kind -- which
        # is the order the rows were gathered in.
        seen: dict[str, int] = {}
        pads: list[dict[str, object]] = []
        for position, kind in enumerate(kinds):
            index = seen.get(kind, 0)
            seen[kind] = index + 1
            pad: dict[str, object] = {
                name: value[index] if isinstance(value, list) else value
                for name, value in options.items()
                if SINK_OPTIONS[name].scope == kind
            }
            asked = any(SINK_OPTIONS[name].scope == kind for name in written)
            if not asked and self._copies_onto_sink(inputs[position], kind, described):
                pad = {f"{kind}_codec": COPY_CODEC}
            if row_meta is not None:
                meta = row_meta[position]
                pad["row"] = meta["row"]
                rendition = meta.get("rendition")
                if rendition:
                    pad["rendition"] = rendition
            pads.append(pad)
        return pads

    def _copies_onto_sink(
        self, ref: FrameRef, kind: StreamType, described: Described
    ) -> bool:
        """True when `ref` is an unmodified probed stream in a codec the
        sink already accepts, and so may travel onto the edge as a stream
        copy instead of paying for an encode nothing asked for.
        """
        if not is_src(ref):
            return False
        alias, stream_type, index = src_parts(ref)
        if stream_type != kind:
            return False
        result = self.probes.get(alias)
        if result is None:
            return False
        streams = result.by_type(stream_type)
        if not 0 <= index < len(streams):
            return False
        codec = streams[index].codec
        if codec is None:
            return False
        accepted = described.sink_codecs(kind)
        return not accepted or codec in accepted

    def _packet_sink_options(
        self, declared: WasmFunction, raw: RawSink, scopes: tuple[str, ...]
    ) -> dict[str, object]:
        """The COPY's WITH options as the encoder a packet sink reads.

        The video half of the file-output vocabulary -- and the audio half
        too, for a sink that reads audio streams (`scopes`, the caller's:
        the declared stream parameters for the old sink form, the SELECT
        list's actual cells for a row-reading one) -- validated against the
        same table; anything else has no encoder to shape and is refused by
        name. `video_codec` is always present on the way out: written, or
        filled from the module's declared preference, h264 when it names
        none -- and checked against that list either way. `audio_codec` is
        filled the same way, and only where an audio stream reaches the sink.
        """
        described = self.describes[declared.module]
        options: dict[str, object] = {}
        option_nodes: dict[str, exp.Expr] = {}
        for option in raw.options:
            if isinstance(_unwrap(option.value), exp.Null):
                continue  # absence: the encoder's own default applies
            line, col = _pos(option.name_node, option.value, raw.path_node)
            value = _validated_option(
                option.name, self._sink_option_value(option, raw), line=line, col=col
            )
            if SINK_OPTIONS[option.name].scope not in scopes:
                allowed = sorted(
                    name
                    for name, spec in SINK_OPTIONS.items()
                    if spec.scope in scopes
                )
                raise FfrwdError(
                    ErrorCode.UNKNOWN_SINK_OPTION,
                    f"option {option.name!r} does not shape the encoder "
                    f"'{declared.name}' reads",
                    line=line,
                    col=col,
                    hint=f"a packet sink takes the {' and '.join(scopes)} "
                    "encoder options: " + ", ".join(allowed),
                )
            options[option.name] = value
            option_nodes[option.name] = option.value
        _check_sink_option_conflicts(options, option_nodes, raw.path_node)
        accepted = described.video_codecs or ()
        for written in _each(options.get("video_codec")):
            assert isinstance(written, str)  # validated as a str above
            codec = encoder_codec(written)
            line, col = _pos(option_nodes["video_codec"], raw.path_node)
            if codec is None:
                raise FfrwdError(
                    ErrorCode.UNSUPPORTED_SQL,
                    f"the stream into a packet sink travels as "
                    f"{_join_codecs(WIRE_VIDEO_CODECS)}, and '{written}' "
                    "encodes none of them",
                    line=line,
                    col=col,
                    hint="name an encoder for one of them, e.g. "
                    + ", ".join(CODEC_ENCODERS.values()),
                )
            if accepted and codec not in accepted:
                raise FfrwdError(
                    ErrorCode.UNSUPPORTED_SQL,
                    f"'{written}' writes {codec}, and the module "
                    f"'{declared.module}' consumes {_join_codecs(accepted)}",
                    line=line,
                    col=col,
                    hint=f"name an encoder for {_join_codecs(accepted)}, or "
                    "drop video_codec to take the module's preference",
                )
        if "video_codec" not in options:
            codec = next(
                (c for c in accepted if c in WIRE_VIDEO_CODECS),
                WIRE_VIDEO_CODECS[0] if not accepted else None,
            )
            if codec is None:
                raise _error(
                    ErrorCode.UNSUPPORTED_SQL,
                    f"the module '{declared.module}' consumes "
                    f"{_join_codecs(accepted)}, and the stream edge carries "
                    f"{_join_codecs(WIRE_VIDEO_CODECS)}",
                    raw.path_node,
                    hint="the module has to accept one of the codecs the "
                    "sidecar's packets travel in",
                )
            options["video_codec"] = CODEC_ENCODERS[codec]
        if "audio" in scopes:
            self._packet_sink_audio_codec(declared, described, options, option_nodes, raw)
        return options

    def _packet_sink_audio_codec(
        self,
        declared: WasmFunction,
        described: Described,
        options: dict[str, object],
        option_nodes: dict[str, exp.Expr],
        raw: RawSink,
    ) -> None:
        """`audio_codec` settled the way `video_codec` is, against the audio
        the stream edge carries and the codecs the module accepts."""
        accepted = described.sink_codecs("audio")
        for written in _each(options.get("audio_codec")):
            assert isinstance(written, str)  # validated as a str above
            codec = audio_encoder_codec(written)
            line, col = _pos(option_nodes["audio_codec"], raw.path_node)
            if codec is None:
                raise FfrwdError(
                    ErrorCode.UNSUPPORTED_SQL,
                    f"the audio stream into a packet sink travels as "
                    f"{_join_codecs(WIRE_AUDIO_CODECS)}, and '{written}' "
                    "encodes none of them",
                    line=line,
                    col=col,
                    hint="name an encoder for one of them, e.g. "
                    + ", ".join(AUDIO_CODEC_ENCODERS.values()),
                )
            if accepted and codec not in accepted:
                raise FfrwdError(
                    ErrorCode.UNSUPPORTED_SQL,
                    f"'{written}' writes {codec}, and the module "
                    f"'{declared.module}' consumes {_join_codecs(accepted)} "
                    "audio",
                    line=line,
                    col=col,
                    hint=f"name an encoder for {_join_codecs(accepted)}, or "
                    "drop audio_codec to take the module's preference",
                )
        if "audio_codec" not in options:
            codec = next(
                (c for c in accepted if c in WIRE_AUDIO_CODECS),
                WIRE_AUDIO_CODECS[0] if not accepted else None,
            )
            if codec is None:
                raise _error(
                    ErrorCode.UNSUPPORTED_SQL,
                    f"the module '{declared.module}' consumes "
                    f"{_join_codecs(accepted)} audio, and the stream edge "
                    f"carries {_join_codecs(WIRE_AUDIO_CODECS)}",
                    raw.path_node,
                    hint="the module has to accept one of the codecs the "
                    "sidecar's packets travel in",
                )
            options["audio_codec"] = AUDIO_CODEC_ENCODERS[codec]

    def _codec_for_rows_track(
        self, options: dict[str, object], outputs: list[Output], path: str | None
    ) -> None:
        """Give a file holding a minted rows track the codec its container reads.

        The rows arrive as WebVTT, which not every container carries, so the
        destination's own extension picks what the track is written as. A
        query naming a subtitle codec itself has already said which, and one
        writing a container that carries WebVTT needs nothing.
        """
        if _SUBTITLE_CODEC_OPTION in options or path is None:
            return
        minted = set(self.rows_tracks)
        if not any(output.ref in minted for output in outputs):
            return
        suffix = path.rsplit(".", 1)[-1].lower() if "." in path else ""
        codec = _ROWS_TRACK_CODECS.get(suffix)
        if codec is not None:
            options[_SUBTITLE_CODEC_OPTION] = codec

    # -- the fan-out TO expression ------------------------------------

    def _fanout_window(self) -> tuple[float | None, float | None] | None:
        """This row's OUTPUT window: the one input window its WHERE named.

        An output seek is a property of the FILE, so every alias the row
        trimmed has to agree on it. Two disagreeing windows are not a
        rejection — they are recorded and send the whole fan-out back to one
        command per row, where each alias seeks its own ``-i`` again.
        """
        windows = set(self.fanout_windows.values())
        if not windows:
            return None
        if len(windows) > 1:
            self.fanout_window_conflict = True
            return None
        return windows.pop()

    def _sink_option_value(self, option: RawSinkOption, raw: RawSink) -> object:
        """One ``WITH (name value)`` value as a python scalar, or a LIST of them.

        ``ARRAY[<literals>][<subscript>]`` -- what a subscripted list variable
        substitutes to -- is read here, so each file a fan-out COPY writes
        carries its own element of the list. A subscript that reads a track
        row picks off the pinned row, exactly as the ``TO`` expression does.

        With no fan-out the rows are gathered into one destination, one track
        apiece, and the option is read once per row: the value is then a LIST,
        one element per track of the option's own scope, in row order.
        ffmpeg spells that ``-b:v:0``, ``-b:v:1``, and so on.

        An option is settled before ffmpeg runs, so those two shapes and the
        constants :func:`_sink_value` reads are all that may stand here; a
        subscript over anything else is refused by name rather than left to
        the option table's type message, which would say only "a BRACKET
        expression".
        """
        node = _unwrap(option.value)
        if not isinstance(node, exp.Bracket):
            return _sink_value(option.value)
        if not isinstance(node.this, exp.Array):
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"sink option '{option.name}' takes a literal or a subscripted "
                f"list variable, got {_describe(node)}",
                node,
                fallback=raw.path_node,
                hint="an option is settled before ffmpeg runs; write the value "
                "out, or pass a list on the command line and subscript it, "
                "e.g. video_bitrate :'rates'[i.i]",
            )
        anchor = raw.branches[0] if raw.branches else exp.Select()
        if references_row_alias(node, set(self.res.row_aliases)) and self.fanout_expr is None:
            # A gathered destination holds one track per row, so the option
            # binds per TRACK, in the order the rows were gathered.
            if not self.sink_rows:
                raise _error(
                    ErrorCode.UNSUPPORTED_SQL,
                    f"sink option '{option.name}' reads a track row, and this "
                    "COPY has no rows to read",
                    node,
                    fallback=raw.path_node,
                    hint=_PER_TRACK_OPTION_HINT,
                )
            env = self.sink_env if self.sink_env is not None else _Env()
            return [
                self._eval_list_element(node, env, row, anchor)
                for row in self.sink_rows
            ]
        env = self.fanout_env if self.fanout_env is not None else _Env()
        return self._eval_list_element(node, env, self.fanout_row, anchor)

    def _check_per_track_options(
        self,
        options: dict[str, object],
        option_nodes: dict[str, exp.Expr],
        outputs: list[Output],
        raw: RawSink,
    ) -> None:
        """A per-row option against the tracks it binds to, one apiece.

        The rows were gathered into this destination, so it holds one track
        per row of the option's own scope -- and a count that does not line up
        is a query saying two different things about how many renditions it
        writes.
        """
        for name, value in options.items():
            if not isinstance(value, list):
                continue
            scope = SINK_OPTIONS[name].scope
            tracks = sum(1 for output in outputs if output.type == scope)
            if tracks == len(value):
                continue
            anchor = option_nodes.get(name)
            line, col = _pos(anchor, raw.path_node) if anchor else _pos(raw.path_node)
            raise FfrwdError(
                ErrorCode.ROW_COUNT_MISMATCH,
                f"sink option {name!r} is read once per row over "
                f"{len(value)} rows, and this destination has "
                f"{tracks} {scope} track{'' if tracks == 1 else 's'}",
                line=line,
                col=col,
                hint=_PER_TRACK_OPTION_HINT,
            )

    def _check_fanout_options(self, options: dict[str, object], raw: RawSink) -> None:
        """The sink options a fan-out COPY does not take, v1.

        ``two_pass`` already compiles to a command SEQUENCE of its own, a
        matrix left closed rather than guessed at.
        """
        if self.fanout_expr is None:
            return
        # Only `two_pass false` is a set option that asks for nothing.
        for name in ("two_pass",):
            if name not in options or options[name] is False:
                continue
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"'{name}' and a fan-out TO cannot both be set",
                raw.path_expr,
                fallback=raw.path_node,
                hint=f"a TO expression writes one file per row; drop {name}, or "
                "write a quoted TO path",
            )

    # -- the manifest destination ------------------------------------------

    def _check_manifest_target(self, raw: RawSink) -> None:
        """The two destination shapes a manifest format cannot take.

        A fan-out ``TO (<expression>)`` writes one file per row, and a
        manifest binds many outputs under ONE written name -- the two answers
        to a multi-row relation cannot both hold. ``UNION ALL`` concatenates
        branches in time, so it carries no rows for the variant map to
        transcribe.
        """
        if self.manifest is None:
            return
        if self.fanout_expr is not None:
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"format '{self.manifest}' binds every output under one "
                "written name, and a TO expression writes one file per row",
                raw.path_expr,
                fallback=raw.path_node,
                hint="name the manifest with a quoted TO path; its rows "
                "become the variant map",
            )
        if len(raw.branches) > 1:
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"UNION ALL concatenates branches in time, and a "
                f"format '{self.manifest}' destination transcribes one "
                "relation's rows",
                raw.path_node,
                hint="concatenate in a CTE and select its columns, or write "
                "the variants as rows of one SELECT",
            )

    def _manifest_rows(
        self, columns: list[_Column], raw: RawSink
    ) -> tuple[list[_VariantRow], list[_Column]]:
        """The variant map's rows, transcribed off this COPY's relation.

        Each SELECT column must carry one stream (or NULL) per row -- at most
        one video column and one audio column, since a var_stream_map entry
        holds one stream of each kind. Returns the rows plus the columns with
        every NULL cell dropped, which is what the output list is built from:
        an absent cell maps nothing, it only shapes the transcription.
        """
        cardinality = max(len(self.sink_rows), 1)
        rows, stripped = self._row_cells(
            columns,
            cardinality,
            raw.path_node,
            f"a format '{self.manifest}' destination",
        )
        self._check_no_null_stream_feeds_a_filter(raw.path_node)
        return rows, stripped

    def _row_cells(
        self,
        columns: list[_Column],
        cardinality: int,
        anchor: exp.Expr,
        subject: str,
    ) -> tuple[list[_VariantRow], list[_Column]]:
        """Every column's cells, read into rows of at most a video and an
        audio one -- the shape a manifest destination and a row-reading
        sink share, `subject` naming which one a rejection is about.

        Returns the rows plus the columns with every NULL cell dropped,
        which is what the caller's output list is built from: an absent
        cell maps nothing, it only shapes the transcription.
        """
        cells: dict[StreamType, list[_Stream | None]] = {}
        stripped: list[_Column] = []
        for column in columns:
            value = column.value
            streams = list(value.streams)
            if not streams:
                stripped.append(column)  # empty column: contributes nothing
                continue
            label = column.name or value.type
            if value.type not in ("video", "audio"):
                raise _error(
                    ErrorCode.UNSUPPORTED_SQL,
                    f"{subject} takes video and audio columns, and '{label}' "
                    f"is {value.type}",
                    anchor,
                    hint="write subtitle and data tracks to a file of their own",
                )
            if value.type in cells:
                raise _error(
                    ErrorCode.UNSUPPORTED_SQL,
                    f"two {value.type} columns at {subject}",
                    anchor,
                    hint="each row is one variant map entry, so it holds at "
                    "most one video and one audio cell; put the second "
                    f"{value.type} in a row of its own",
                )
            row_set = value.is_array and column.splat and len(streams) == cardinality
            if cardinality > 1 and not row_set:
                raise _error(
                    ErrorCode.ROW_COUNT_MISMATCH,
                    f"'{label}' does not carry one stream per row, and each "
                    f"row of {subject} is one variant map entry",
                    anchor,
                    hint="select a row column (a joined CTE's, an unnest "
                    "table's); a gathered array belongs to a single file",
                )
            if cardinality == 1 and len(streams) > 1:
                raise _error(
                    ErrorCode.ROW_COUNT_MISMATCH,
                    f"'{label}' is {len(streams)} streams in one row, and "
                    f"each row of {subject} is one variant map entry",
                    anchor,
                    hint=_RENDITION_PICK_HINT
                    if self._from_rendition_table(self.sink_env)
                    else "one variant per row: unnest or join the tracks "
                    "into rows instead of selecting an array",
                )
            cells[value.type] = [
                None if stream.ref == _NULL_STREAM_REF else stream
                for stream in streams
            ]
            kept = [s for s in streams if s.ref != _NULL_STREAM_REF]
            if len(kept) != len(streams):
                stripped.append(replace(column, value=_array(value.type, kept)))
            else:
                stripped.append(column)
        rows: list[_VariantRow] = []
        empty: list[_Stream | None] = [None] * cardinality
        for position in range(cardinality):
            video = cells.get("video", empty)[position]
            audio = cells.get("audio", empty)[position]
            if video is None and audio is None:
                raise _error(
                    ErrorCode.STREAM_NOT_FOUND,
                    f"row {position + 1} of {subject} has no stream in "
                    "any column",
                    anchor,
                    hint="every row is one variant map entry; drop the empty "
                    "row with a WHERE, or use a join that leaves no all-NULL "
                    "row",
                )
            rows.append(_VariantRow(video=video, audio=audio))
        return rows, stripped

    def _check_no_null_stream_feeds_a_filter(self, anchor: exp.Expr) -> None:
        """A NULL cell can be SELECTED at a manifest destination, not filtered.

        A call over a nullable column would hand a filter a stream that is
        not there; the sentinel ref would reach the graph as a dangling
        input. Caught here, by scanning what this COPY's lowering minted.
        """
        for node in self.graph.nodes.values():
            if any(ref == _NULL_STREAM_REF for ref in node.inputs):
                raise _error(
                    ErrorCode.STREAM_NOT_FOUND,
                    f"a NULL stream feeds {node.filter}(): an outer join's "
                    "gap can be selected at a manifest destination, not "
                    "filtered",
                    anchor,
                    hint="apply the filter inside the CTE, where the row "
                    "exists, and select the joined column bare",
                )

    def _hand_written_map_error(
        self,
        option: RawSinkOption,
        columns: list[_Column],
        variant_rows: list[_VariantRow],
        raw: RawSink,
    ) -> FfrwdError:
        """The refusal for a hand-written variant map, naming the compiler's own."""
        name = MANIFEST_MAP_OPTION[self.manifest or "hls"]
        derived = self._manifest_map(columns, variant_rows, self.manifest or "hls")
        line, col = _pos(option.name_node, option.value, raw.path_node)
        return FfrwdError(
            ErrorCode.UNKNOWN_SINK_OPTION,
            f"'{option.name}' is the compiler's to write: it is a "
            "transcription of this COPY's rows",
            line=line,
            col=col,
            hint=f"drop it; the compiler writes {name} '{derived}'",
        )

    def _check_manifest_options(
        self,
        options: dict[str, object],
        option_nodes: dict[str, exp.Expr],
        columns: list[_Column],
        variant_rows: list[_VariantRow] | None,
        raw: RawSink,
    ) -> None:
        """The format each manifest option belongs to, and the map smuggle.

        A manifest option under the other format -- or under no manifest
        format at all -- is refused by name. ``codec_params`` carrying a
        hand-written map spelling is refused the way the option itself is.
        """
        for name in options:
            wanted = MANIFEST_OPTION_FORMATS.get(name)
            if wanted is None or wanted == self.manifest:
                continue
            have = (
                f"format '{self.manifest}'"
                if self.manifest is not None
                else "no manifest format"
            )
            raise _error(
                ErrorCode.SINK_OPTION_TYPE,
                f"option '{name}' belongs to format '{wanted}', and this "
                f"COPY has {have}",
                option_nodes.get(name),
                fallback=raw.path_node,
                hint=f"set format '{wanted}', or drop {name}",
            )
        if variant_rows is None or self.manifest is None:
            return
        for element in _each(options.get("codec_params")):
            if not isinstance(element, str):
                continue
            for smuggled in MANIFEST_MAP_OPTION.values():
                if smuggled in element:
                    derived = self._manifest_map(columns, variant_rows, self.manifest)
                    name = MANIFEST_MAP_OPTION[self.manifest]
                    raise _error(
                        ErrorCode.SINK_OPTION_TYPE,
                        f"'{smuggled}' inside codec_params: the variant map "
                        "is the compiler's to write",
                        option_nodes.get("codec_params"),
                        fallback=raw.path_node,
                        hint=f"drop it; the compiler writes {name} '{derived}'",
                    )

    def _manifest_map(
        self, columns: list[_Column], variant_rows: list[_VariantRow], format_name: str
    ) -> str:
        """The variant map, transcribed: each row one entry, in row order.

        HLS spells it ``var_stream_map`` -- ``v:N``/``a:N`` per-kind indices
        in output order, an ``agroup`` binding the demuxed shape together,
        names from the streams, ``default:yes`` on one rendition. DASH spells
        the same analysis ``adaptation_sets``: video and audio each one set,
        one representation per rung, by flat output stream index.
        """
        if format_name == "dash":
            sets: list[str] = []
            offset = 0
            by_type: dict[StreamType, list[int]] = {}
            for column in columns:
                for stream in column.value.streams:
                    by_type.setdefault(column.value.type, []).append(offset)
                    offset += 1
            for set_id, stream_type in enumerate(
                t for t in ("video", "audio") if t in by_type
            ):
                indices = ",".join(str(i) for i in by_type[stream_type])
                sets.append(f"id={set_id},streams={indices}")
            return " ".join(sets)

        video_names, audio_names = self._variant_names(variant_rows)
        demuxed = any(row.audio is not None and row.video is None for row in variant_rows)
        default_row = self._default_audio_row(variant_rows)
        entries: list[str] = []
        video_seen = 0
        audio_seen = 0
        for position, row in enumerate(variant_rows):
            parts: list[str] = []
            if row.video is not None:
                parts.append(f"v:{video_seen}")
            if row.audio is not None:
                parts.append(f"a:{audio_seen}")
            if demuxed:
                parts.append("agroup:aud")
            if row.video is not None:
                parts.append(f"name:{video_names[video_seen]}")
                video_seen += 1
            elif row.audio is not None:
                parts.append(f"name:{audio_names[audio_seen]}")
                language = _stream_language(row.audio)
                if language is not None:
                    parts.append(f"language:{language}")
                if position == default_row:
                    parts.append("default:yes")
            if row.audio is not None:
                audio_seen += 1
            entries.append(",".join(parts))
        return " ".join(entries)

    def _default_audio_row(self, variant_rows: list[_VariantRow]) -> int | None:
        """Which rendition row gets ``default:yes``.

        The probed disposition wins when a track carries one; otherwise the
        first audio-only row. A muxed row is a variant, not a rendition, and
        never takes the flag.
        """
        renditions = [
            position
            for position, row in enumerate(variant_rows)
            if row.audio is not None and row.video is None
        ]
        for position in renditions:
            audio = variant_rows[position].audio
            source = audio.source if audio is not None else None
            if source is not None and source.disposition.get("default"):
                return position
        return renditions[0] if renditions else None

    def _variant_names(
        self, variant_rows: list[_VariantRow]
    ) -> tuple[list[str], list[str]]:
        """Names for the map's entries, per kind, in output order.

        Video takes its height (``1080p``); audio its language tag. A name
        that cannot be computed, or that collides with another of its kind,
        falls back to its position (``v0``, ``a1``) -- files carry ``und``
        more often than not, and ``%v`` becomes a directory name, so names
        must exist and must not collide.
        """
        video_streams = [row.video for row in variant_rows if row.video is not None]
        audio_streams = [
            row.audio for row in variant_rows if row.audio is not None and row.video is None
        ]
        # A muxed row's audio never names anything, but it still numbers.
        heights = [self._output_height(stream.ref) for stream in video_streams]
        video = _fallback_names(
            [None if h is None else f"{h}p" for h in heights], "v"
        )
        languages = [_stream_language(stream) for stream in audio_streams]
        named = _fallback_names(languages, "a")
        # Audio names index by RENDITION order among audio cells; muxed rows
        # consume an audio index without a name of their own.
        audio: list[str] = []
        taken = 0
        for row in variant_rows:
            if row.audio is None:
                continue
            if row.video is None:
                audio.append(named[taken])
                taken += 1
            else:
                audio.append("")  # a muxed row's audio: numbered, never named
        return video, audio

    def _derive_manifest(
        self,
        options: dict[str, object],
        option_nodes: dict[str, exp.Expr],
        columns: list[_Column],
        variant_rows: list[_VariantRow],
        outputs: list[Output],
        path: str,
        raw: RawSink,
    ) -> str:
        """Everything ``format 'hls'``/``'dash'`` owns: the keyframe
        discipline, the variant map, and (for hls) the whole layout.

        Returns the positional output path -- the variant playlist pattern
        for hls, where the written destination names the MASTER playlist and
        ffmpeg's positional output is the variant pattern; the ``.mpd``
        itself for dash, whose muxer already writes everything beside it.
        Every derived option lands in `options` under its ordinary name, so
        writing any of them by hand simply pre-empts the derivation.
        """
        format_name = self.manifest or "hls"
        self._derive_keyframes(options, option_nodes, outputs, raw, format_name)
        options[MANIFEST_MAP_OPTION[format_name]] = self._manifest_map(
            columns, variant_rows, format_name
        )
        if format_name != "hls":
            return path
        normalized = path.replace("\\", "/")
        directory, _, filename = normalized.rpartition("/")
        prefix = f"{directory}/" if directory else ""
        options.setdefault("master_pl_name", filename)
        extension = "m4s" if options.get("hls_segment_type") == "fmp4" else "ts"
        options.setdefault(
            "hls_segment_filename", f"{prefix}v%v/segment_%d.{extension}"
        )
        if options.get("hls_segment_type") == "fmp4":
            options.setdefault("hls_fmp4_init_filename", "init.mp4")
        return f"{prefix}v%v/index.m3u8"

    def _derive_keyframes(
        self,
        options: dict[str, object],
        option_nodes: dict[str, exp.Expr],
        outputs: list[Output],
        raw: RawSink,
        format_name: str,
    ) -> None:
        """The keyframe discipline a manifest's segments need.

        A segment boundary must be a keyframe in every rung, so the gop is
        the segment length times the frame rate (written by the query's
        ``fps()``, probed otherwise), ``keyint_min`` is pinned to it, and
        scene cuts are disabled in the encoder's own spelling. An explicit
        gop that does not divide the segment is refused, naming the nearest
        ones that would.
        """
        video_maps = [output for output in outputs if output.type == "video"]
        if not video_maps:
            return
        encoded = "video" in copy_suppressed_scopes(options) or any(
            not is_src(output.ref) for output in video_maps
        )
        if not encoded:
            return
        segment_name = MANIFEST_SEGMENT_OPTION[format_name]
        segment = options.get(segment_name)
        if not isinstance(segment, int | float):
            segment = MANIFEST_DEFAULT_SEGMENT[format_name]
        rates = [
            self._output_rate(output.ref, raw, format_name) for output in video_maps
        ]
        targets = [max(1, round(segment * rate)) for rate in rates]
        written = options.get("gop")
        if written is not None:
            gops = written if isinstance(written, list) else [written] * len(targets)
            for gop, target, rate in zip(gops, targets, rates):
                if not isinstance(gop, int) or target % gop == 0:
                    continue
                divisors = [d for d in range(1, target + 1) if target % d == 0]
                nearest = sorted(
                    {
                        max((d for d in divisors if d <= gop), default=1),
                        min((d for d in divisors if d >= gop), default=target),
                    }
                )
                line, col = _pos(option_nodes.get("gop"), raw.path_node)
                raise FfrwdError(
                    ErrorCode.SINK_OPTION_TYPE,
                    f"gop {gop} does not divide the {target}-frame segment "
                    f"({segment_name} {_text_number(segment)} x "
                    f"{_text_number(rate)} fps)",
                    line=line,
                    col=col,
                    hint="the nearest that would: "
                    + ", ".join(str(d) for d in nearest),
                )
        else:
            options["gop"] = targets[0] if len(set(targets)) == 1 else list(targets)
        keyint = options["gop"]
        options["keyint_min"] = list(keyint) if isinstance(keyint, list) else keyint
        self._disable_scene_cuts(options)

    def _disable_scene_cuts(self, options: dict[str, object]) -> None:
        """Scene cuts off, in the encoder's own spelling.

        libx264 reads ``-sc_threshold 0``; libx265 and libsvtav1 read a
        private param, carried on the codec_params road the table already
        renders. Any other encoder is left alone: a knob it does not have
        would be a silent no-op.
        """
        codec = options.get("video_codec")
        if codec == "libx264":
            options["sc_threshold"] = 0
            return
        param = {"libx265": "scenecut=0", "libsvtav1": "scd=0"}.get(
            codec if isinstance(codec, str) else ""
        )
        if param is None:
            return
        key = param.partition("=")[0]
        written = options.get("codec_params")
        if written is None:
            options["codec_params"] = param
        elif isinstance(written, str):
            if key not in written:
                options["codec_params"] = f"{written}:{param}"
        elif isinstance(written, list):
            options["codec_params"] = [
                element
                if not isinstance(element, str) or key in element
                else f"{element}:{param}"
                for element in written
            ]

    def _output_rate(self, ref: FrameRef, raw: RawSink, format_name: str) -> float:
        """One video output's frame rate, walked back through its chain.

        The nearest ``fps()`` on the way to the source wins -- it is what
        the stream actually plays at; failing one, the probed rate of the
        source stream the chain reads. A rate the compiler cannot know is a
        refusal: the keyframe discipline is derived from it.
        """
        current = ref
        while current and not is_src(current):
            node = self.graph.nodes.get(current.partition(":")[0])
            if node is None:
                break
            rate = _node_rate(node)
            if rate is not None:
                return rate
            if not node.inputs:
                break
            current = node.inputs[0]
        if current and is_src(current):
            alias, stream_type, index = src_parts(current)
            result = self.probes.get(alias)
            if result is not None:
                streams = result.by_type(stream_type)
                if 0 <= index < len(streams):
                    rate = _parse_rate(streams[index].fps)
                    if rate is not None:
                        return rate
        raise _error(
            ErrorCode.UNSUPPORTED_SQL,
            f"format '{format_name}' derives the keyframe interval from the "
            "frame rate, and this video stream's rate is unknown",
            raw.path_node,
            hint="pin the rate in the query, e.g. fps(<stream>, 30)",
        )

    def _output_height(self, ref: FrameRef) -> int | None:
        """One video output's height, walked back through its chain.

        The nearest size-setting filter wins; a proportional height (``-2``/
        ``-1``/``0``) is computed from its width and the source's probed
        dimensions. None when nothing on the way says -- the variant name
        then falls back to its position.
        """
        current = ref
        width: int | None = None  # a pending proportional scale's width
        while current and not is_src(current):
            node = self.graph.nodes.get(current.partition(":")[0])
            if node is None:
                return None
            height = _int_arg(node, "h", "height")
            if height is not None and height > 0:
                return height
            if height is not None:  # proportional: need the source's aspect
                width = _int_arg(node, "w", "width")
                if width is None or width <= 0:
                    return None
            if not node.inputs:
                return None
            current = node.inputs[0]
        if not current or not is_src(current):
            return None
        alias, stream_type, index = src_parts(current)
        result = self.probes.get(alias)
        if result is None:
            return None
        streams = result.by_type(stream_type)
        if not 0 <= index < len(streams):
            return None
        meta = streams[index]
        if width is None:
            return meta.height
        if not meta.width or not meta.height:
            return None
        return round(width * meta.height / meta.width / 2) * 2

    def _sink_path(self, raw: RawSink) -> str:
        """``TO (<expression>)`` evaluated: this command's destination.

        A constant expression is an ordinary path. A fan-out one is the pinned
        row's, and is checked for the two things a name built from file
        metadata must not smuggle in: a NULL (an unprobed column, named), and a
        path separator or ``..`` inside a COMPUTED segment.
        """
        expression = raw.path_expr
        if expression is None:  # defensive: the caller checked it
            raise _error(ErrorCode.INTERNAL, "sink path expression is missing")
        env = self.fanout_env if self.fanout_env is not None else _Env()
        # The wrapped query's first branch is the anchor every rejection below
        # falls back to; `raw.query` may be a Union, which `_eval_value` is not
        # typed for.
        anchor = raw.branches[0] if raw.branches else exp.Select()
        if self.fanout_expr is not None and self.fanout_env is None:
            # No row relation ever formed for this branch -- ordinarily a
            # foreign row alias this COPY's own FROM does not bind, but a
            # rendition column read off a plain (non-ladder) input alias
            # lands here too, since `_bind_renditions` leaves that alias an
            # `_InputBinding` with no relation at all. That one has its own,
            # more specific rejection (the file's own, from `_row_value_of`),
            # so it is evaluated against the branch's real bindings instead
            # of raising the generic message below.
            bound_env = self.sink_env if self.sink_env is not None else _Env()
            if not self._reads_unbound_rendition_column(expression, bound_env):
                raise _error(
                    ErrorCode.UNSUPPORTED_SQL,
                    "a TO expression reads a track-row table this COPY's FROM "
                    "does not bind",
                    expression,
                    fallback=raw.path_node,
                    hint="unnest the rows in the COPY's own FROM, e.g. FROM "
                    "input(:'src') f, unnest(f.audio) t",
                )
            env = bound_env
        for segment in _computed_segments(expression, set(self.res.row_aliases)):
            self._check_path_segment(segment, env, raw, anchor)
        value = self._eval_value(expression, env, self.fanout_row, anchor)
        if value is None:
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                "the TO expression is NULL for this row: "
                + self._null_field(expression, env, anchor),
                expression,
                fallback=raw.path_node,
                hint="COALESCE the column, or filter the rows that lack it",
            )
        return _tag_text(value)

    def _reads_unbound_rendition_column(self, expression: exp.Expr, env: _Env) -> bool:
        """True if `expression` reads a rendition column off an input alias
        `env` binds, but only as a plain `_InputBinding` -- a ladder column
        resolve admitted on spec, that this alias's own probe turned up no
        renditions for."""
        for sub in expression.walk():
            if not isinstance(sub, exp.Column):
                continue
            table_node = sub.args.get("table")
            if table_node is None:
                continue
            binding = env.bindings.get(_fold(table_node))
            if isinstance(binding, _InputBinding) and _fold(sub.this) in _RENDITION_SCHEMA:
                return True
        return False

    def _check_path_segment(
        self, segment: exp.Expr, env: _Env, raw: RawSink, anchor: exp.Select
    ) -> None:
        """One computed piece of a path: no separator, no ``..``."""
        value = self._eval_value(segment, env, self.fanout_row, anchor)
        if not isinstance(value, str):
            return
        found = next((bad for bad in ("/", "\\", "..") if bad in value), None)
        if found is None:
            return
        raise _error(
            ErrorCode.UNSUPPORTED_SQL,
            f"a computed path segment may not contain {found!r}, got {value!r}",
            segment,
            fallback=raw.path_node,
            hint="write the directory as a literal: 'out/' || t.tags.language "
            "|| '.m4a'; metadata never chooses the directory",
        )

    def _null_field(self, expression: exp.Expr, env: _Env, anchor: exp.Select) -> str:
        """Which column of the path expression read NULL, for the message."""
        for sub in expression.walk():
            variable = null_variable(sub)
            if variable is not None:
                return f"':{variable}' was not set"
        for sub in expression.walk():
            if not isinstance(sub, exp.Column):
                continue
            if self._eval_value(sub, env, self.fanout_row, anchor) is None:
                table_node = sub.args.get("table")
                prefix = f"{_fold(table_node)}." if table_node is not None else ""
                return f"'{prefix}{column_label(_fold(sub.this))}' was never probed"
        return "no column of it has a value"

    # -- the chapters output column ------------------------------------

    def _collect_chapters(
        self, projection: exp.Expr, env: _Env, select: exp.Select, *, scope: _TagScope
    ) -> None:
        """``... AS chapters``: the file's chapter list, from one of three sources.

        A literal ``ARRAY[STRUCT(...)::chapter, ...]`` and an ``array_agg`` over
        rows both become one self-contained ffmetadata ``data:`` input;
        ``<input>.chapters`` names that input's own list; NULL writes none.
        The value is the FILE's, not a row's, so it is read once per COPY and
        two branches of a UNION ALL have to agree on it.
        """
        if scope != "sink":
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"'{CHAPTERS_COLUMN}' is the file's chapter list, and a CTE "
                "body writes no file",
                projection,
                fallback=select,
                hint="build the chapter list in the outer SELECT, e.g. "
                "array_agg(STRUCT(c.title AS title, c.start_t AS start_t, "
                "c.end_t AS end_t)::chapter) AS chapters",
            )
        if self.fanout_expr is not None:
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"'{CHAPTERS_COLUMN}' and a fan-out TO cannot both be set",
                projection,
                fallback=select,
                hint="a TO expression writes one file per row; drop the "
                "chapters column, or write a quoted TO path",
            )
        index = self._chapters_input(_unwrap(projection), env, select)
        if self.chapters is not None and self.chapters != index:
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"'{CHAPTERS_COLUMN}' takes two different chapter lists",
                projection,
                fallback=select,
                hint="a file has one chapter list, so write the column once; "
                "the branches of a UNION ALL write one file between them",
            )
        self.chapters = index

    def _chapters_input(self, value: exp.Expr, env: _Env, select: exp.Select) -> int:
        """The ffmpeg input index a ``chapters`` column resolves to."""
        if isinstance(value, exp.Null):
            return NO_CHAPTERS
        copied = self._copied_chapters(value, env)
        if copied is not None:
            return copied
        records = self._chapter_records(value, env, select)
        if not records:
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"'{CHAPTERS_COLUMN}' is an empty list",
                value,
                fallback=select,
                hint=f"write at least one chapter, or NULL AS {CHAPTERS_COLUMN} "
                "for a file with none",
            )
        text = _chapters_ffmetadata(records)
        uri = "data:text/plain;base64," + base64.b64encode(text.encode()).decode()
        return self._mint_chapters_input(uri)

    def _copied_chapters(self, value: exp.Expr, env: _Env) -> int | None:
        """The input index behind ``<input>.chapters``, else None."""
        if not isinstance(value, exp.Column) or isinstance(value.this, exp.Star):
            return None
        table_node = value.args.get("table")
        if table_node is None or _fold(value.this) != CHAPTERS_COLUMN:
            return None
        binding = env.bindings.get(_fold(table_node))
        if not isinstance(binding, _InputBinding):
            return None
        return self.graph.sources.get(binding.alias)

    def _chapter_records(
        self, value: exp.Expr, env: _Env, select: exp.Select
    ) -> list[_Chapter]:
        """The chapter records a ``chapters`` column lists, in written order.

        A literal array is evaluated ONCE, over the branch's first row -- the
        list belongs to the file, not to a row -- so it may read an input's
        ``duration`` or a variable. ``array_agg`` is the per-row form: one
        record per surviving row, in row order.
        """
        if isinstance(value, exp.ArrayAgg):
            inner = value.this
            relation = env.relation
            if not isinstance(inner, exp.Expr) or relation is None:
                raise _error(
                    ErrorCode.UNSUPPORTED_SQL,
                    "array_agg() aggregates rows, and this query has none",
                    value,
                    fallback=select,
                    hint=_CHAPTERS_COLUMN_HINT,
                )
            return [
                self._chapter_record(inner, env, row, select) for row in relation.tuples
            ]
        if isinstance(value, exp.Array):
            row = _group_row(env)
            return [
                self._chapter_record(element, env, row, select)
                for element in value.expressions
                if isinstance(element, exp.Expr)
            ]
        raise _error(
            ErrorCode.UNSUPPORTED_SQL,
            f"'{CHAPTERS_COLUMN}' takes an array of chapter records, got "
            f"{_describe(value)}",
            value,
            fallback=select,
            hint=_CHAPTERS_COLUMN_HINT,
        )

    def _written_record(
        self,
        node: exp.Expr,
        record: str,
        literal: str,
        hint: str,
        env: _Env,
        row: _RowTuple,
        select: exp.Select,
    ) -> dict[str, tuple[exp.Expr, RowValue]]:
        """One ``STRUCT(...)::<record>``, evaluated: each field's cell and value.

        Fields are named (:data:`~ffrwd.types.RECORD_FIELDS` lists them
        for the record); a query supplies the writable ones, by name, and
        never a probed one like ``index``. Each value takes the ordinary
        compile-time value grammar. The cell is kept beside the value so a
        rejection anchors on what the query typed.

        A ``SELECT AS STRUCT`` gather's struct carries no cast -- there is
        nowhere in that spelling to write one -- so it is marked instead
        (``ARRAY(...)``'s own resolve-time rewrite) and accepted here on that
        mark alone; an ordinary bare ``STRUCT(...)`` still needs its
        ``::<record>`` cast exactly as before.
        """
        node = _unwrap(node)
        fields = RECORD_FIELDS[record]
        matches = record_cast_type(node) == record
        struct = _struct_node(node) if matches else None
        if struct is None and isinstance(node, exp.Struct) and node.meta.get(
            "gathered_struct"
        ):
            struct = node
        if struct is None:
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"{article(record)} {record} is written as {literal}, got "
                f"{_describe(node)}",
                node,
                fallback=select,
                hint=hint,
            )
        cells = self._named_record_cells(struct, record, fields, select)
        return {
            name: (cell, self._eval_value(cell, env, row, select))
            for name, cell in cells.items()
        }

    def _named_record_cells(
        self,
        struct: exp.Struct,
        record: str,
        fields: tuple[Field, ...],
        select: exp.Select,
    ) -> dict[str, exp.Expr]:
        """One ``STRUCT(... AS name)`` matched to a record's declared fields.

        Order-free: the field NAME picks the slot. A field the struct leaves
        out is NULL, which the per-field checks then accept or reject exactly
        as a written NULL is accepted or rejected.
        """
        written = _struct_fields(struct)
        declared = {field.name for field in fields}
        for name in written:
            if name in declared:
                continue
            listed = ", ".join(field.name for field in fields)
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"{article(record)} {record} has no field '{name}'",
                written[name],
                fallback=select,
                hint=f"the fields a {record} takes are {listed}",
            )
        return {
            field.name: written.get(field.name, exp.Null()) for field in fields
        }

    def _chapter_record(
        self, node: exp.Expr, env: _Env, row: _RowTuple, select: exp.Select
    ) -> _Chapter:
        """One ``STRUCT(title, start_t, end_t)::chapter``, evaluated and checked."""
        cells = self._written_record(
            node, CHAPTER_TYPE, _CHAPTER_LITERAL, _CHAPTERS_COLUMN_HINT, env, row, select
        )
        title_cell, title = cells["title"]
        start_cell, start = cells["start_t"]
        end_cell, end = cells["end_t"]
        return _Chapter(
            start=_span_number(
                start, f"{CHAPTERS_COLUMN}.start_t", "start_t", start_cell, _CHAPTER_EXAMPLE
            ),
            end=_span_number(
                end, f"{CHAPTERS_COLUMN}.end_t", "end_t", end_cell, _CHAPTER_EXAMPLE
            ),
            title=_chapter_title(title, title_cell),
            start_node=start_cell,
            end_node=end_cell,
        )

    # -- the attachments output column ---------------------------------

    def _collect_attachments(
        self, projection: exp.Expr, env: _Env, select: exp.Select, *, scope: _TagScope
    ) -> None:
        """``... AS attachments``: the files this output carries.

        Each record names a file to read, so the column emits one ``-attach``
        per record rather than minting an input the way a chapter list does.
        The list belongs to the FILE, so two branches of a UNION ALL have to
        agree on it.
        """
        if scope != "sink":
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"'{ATTACHMENTS_COLUMN}' is the file's attachment list, and a "
                "CTE body writes no file",
                projection,
                fallback=select,
                hint="build the list in the outer SELECT, e.g. "
                f"ARRAY[{_ATTACHMENT_EXAMPLE}] AS {ATTACHMENTS_COLUMN}",
            )
        written = self._attachment_records(_unwrap(projection), env, select)
        if self.attachments and self.attachments != written:
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"'{ATTACHMENTS_COLUMN}' takes two different attachment lists",
                projection,
                fallback=select,
                hint="a file has one attachment list, so write the column once; "
                "the branches of a UNION ALL write one file between them",
            )
        self.attachments = written

    def _attachment_records(
        self, value: exp.Expr, env: _Env, select: exp.Select
    ) -> list[Attachment]:
        """The attachments an ``attachments`` column lists, in written order.

        The two spellings a record list takes everywhere: a literal array,
        read element by element over the branch's first row, and an
        ``array_agg`` read once per surviving row. ``NULL`` writes a file
        carrying none, which is also what an omitted column writes -- ffmpeg
        attaches nothing on its own.
        """
        if isinstance(value, exp.Null):
            return []
        if isinstance(value, exp.ArrayAgg):
            inner = value.this
            relation = env.relation
            if not isinstance(inner, exp.Expr) or relation is None:
                raise _error(
                    ErrorCode.UNSUPPORTED_SQL,
                    "array_agg() aggregates rows, and this query has none",
                    value,
                    fallback=select,
                    hint=_ATTACHMENTS_COLUMN_HINT,
                )
            return [
                self._attachment_record(inner, env, row, select)
                for row in relation.tuples
            ]
        if isinstance(value, exp.Array):
            row = _group_row(env)
            written = [
                self._attachment_record(element, env, row, select)
                for element in value.expressions
                if isinstance(element, exp.Expr)
            ]
            if not written:
                raise _error(
                    ErrorCode.UNSUPPORTED_SQL,
                    f"'{ATTACHMENTS_COLUMN}' is an empty list",
                    value,
                    fallback=select,
                    hint="write at least one attachment, or NULL AS "
                    f"{ATTACHMENTS_COLUMN} for a file carrying none",
                )
            return written
        raise _error(
            ErrorCode.UNSUPPORTED_SQL,
            f"'{ATTACHMENTS_COLUMN}' takes an array of attachment records, got "
            f"{_describe(value)}",
            value,
            fallback=select,
            hint=_ATTACHMENTS_COLUMN_HINT,
        )

    def _attachment_record(
        self, node: exp.Expr, env: _Env, row: _RowTuple, select: exp.Select
    ) -> Attachment:
        """One ``STRUCT(filename, mimetype, path)::attachment``, evaluated."""
        cells = self._written_record(
            node,
            ATTACHMENT_TYPE,
            _ATTACHMENT_LITERAL,
            _ATTACHMENTS_COLUMN_HINT,
            env,
            row,
            select,
        )
        filename_cell, filename = cells["filename"]
        mimetype_cell, mimetype = cells["mimetype"]
        path_cell, path = cells["path"]
        return Attachment(
            path=_attachment_path(path, path_cell),
            filename=_attachment_text(filename, "filename", filename_cell),
            mimetype=_attachment_text(mimetype, "mimetype", mimetype_cell),
        )

    def _cue_record(
        self, node: exp.Expr, env: _Env, row: _RowTuple, select: exp.Select
    ) -> _Cue:
        """One ``STRUCT(text, start_t, end_t)::cue``, evaluated and checked."""
        cells = self._written_record(
            node, CUE_TYPE, _CUE_LITERAL, _CUE_ARRAY_HINT, env, row, select
        )
        text_cell, text = cells["text"]
        start_cell, start = cells["start_t"]
        end_cell, end = cells["end_t"]
        return _Cue(
            start=_span_number(
                start, f"{CUE_TYPE}.start_t", "start_t", start_cell, _CUE_EXAMPLE
            ),
            end=_span_number(end, f"{CUE_TYPE}.end_t", "end_t", end_cell, _CUE_EXAMPLE),
            text=_cue_text(text, text_cell),
            start_node=start_cell,
            end_node=end_cell,
        )

    def _mint_chapters_input(self, uri: str) -> int:
        """Add one ffmetadata ``data:`` URI as an extra ``-i``; return its index.

        Mirrors :meth:`_mint_input` (the ``empty_captions`` mechanism): the
        alias is spelled so no query can ever collide with it, and it exists
        only to carry the slot in the graph's alias-keyed tables. Unlike
        ``_mint_input`` this returns the plain ffmpeg input INDEX, not a
        stream ref -- ``-map_chapters`` names an input, not a stream.
        """
        index = len(self.graph.input_paths)
        alias = f"{MACRO_NAMESPACE}.chapters#{index + 1}"
        self.graph.input_paths.append(uri)
        self.graph.sources[alias] = index
        self.minted_input_options[alias] = {"format": "ffmetadata"}
        return index

    # -- input() named options ---------------------

    def _check_realtime_option(
        self,
        alias: str,
        options: dict[str, object],
        raw_options: Sequence[RawInputOption],
    ) -> None:
        """Refuse `realtime => true` on a socket: it is already paced by reality.

        `ffrwd.processes.is_live` also calls a `format =>`-forced input live
        (a capture device cannot be opened twice, same as a socket), but that
        rule conflates a device with a SYNTHETIC one -- `format => 'lavfi'`
        generates frames as fast as it is asked to, and pacing it with
        `realtime => true` is exactly the documented idiom (recipe 101, 102 in
        `../docs/examples.md`). Telling a capture device from a generator by
        its `format` value needs a name list this table does not carry, so
        that half stays unrefused -- only a URL (`is_url`: udp, srt, rtmp,
        rtsp, http(s), ...) is unambiguous enough to reject here.
        """
        if options.get("realtime") is not True:
            return
        index = self.res.sources.get(alias)
        path = self.res.input_paths[index] if index is not None else ""
        if not is_url(path):
            return
        value_node = next((o.value for o in raw_options if o.name == "realtime"), None)
        path_node = raw_options[0].path_node if raw_options else None
        line, col = _pos(value_node, path_node)
        raise FfrwdError(
            ErrorCode.INPUT_OPTION_TYPE,
            f"'{alias}' is already live -- realtime => true would pace it a second time",
            line=line,
            col=col,
            hint="drop realtime; a socket is already paced by its own clock",
        )

    def _lower_input_options(self) -> dict[str, dict[str, object]]:
        """Validate every `input('path', name => value, ...)`'s trailing options.

        Mirrors `_lower_sink`: anchor falls back through the name node to the
        value node to the input()'s own path literal, since neither a
        Kwarg's `Var` name nor a `Boolean`/`Var`/`Null` value carries a token
        position (same gap sink option names have).
        """
        result: dict[str, dict[str, object]] = {}
        for alias, raw_options in self.res.input_options.items():
            options = input_option_values(raw_options)
            if options:
                self._check_realtime_option(alias, options, raw_options)
                result[alias] = options
        # A per-row `-i` repeats its origin's options: same file, same demuxer,
        # only the seek differs.
        for minted, origin in self.row_input_source.items():
            origin_options = result.get(origin)
            if origin_options:
                result[minted] = dict(origin_options)
        # Compiler-minted inputs last: their options are INTERNAL (`-f webvtt`
        # for an `empty_captions` data: URI), already validated by construction,
        # and their aliases cannot collide with a user one.
        result.update(self.minted_input_options)
        return result

    # -- a query (one SELECT, or a UNION ALL of them) ----------------------

    def _lower_query(
        self, branches: list[exp.Select], anchor: exp.Expr, *, tags: _TagScope
    ) -> list[_Column]:
        if not branches:
            raise _error(ErrorCode.UNSUPPORTED_SQL, "query has no SELECT", anchor)
        self.tags = {}
        self.dispositions = {}
        self.container_tags = {}
        lowered = [self._lower_branch(branch, tags=tags) for branch in branches]
        if len(lowered) == 1:
            # A single branch keeps its arrays: a CTE body's array column stays
            # an array for `<cte>.<name>` to splat, broadcast over, or subscript.
            return lowered[0]
        # concat maps one input pad per column, so arrays are flattened to
        # one column per element BEFORE it sees them.
        flattened = [_flatten(columns) for columns in lowered]
        self._check_concat_columns(branches, flattened)
        self._check_concat_signature(branches, lowered, flattened)
        return self._concat(flattened)

    def _check_concat_columns(
        self, branches: list[exp.Select], flattened: list[list[_Column]]
    ) -> None:
        """No UNION ALL branch may carry a subtitle/data column.

        ``concat`` is a filtergraph filter and takes ``v`` video plus ``a``
        audio pads — there is no ``s``/``d`` half — so a caption column in a
        concatenated branch has nowhere to go. Checked before
        :meth:`_check_concat_signature` so the rejection names the real reason
        rather than a column-count mismatch.
        """
        for index, columns in enumerate(flattened):
            for column in columns:
                if column.value.type in _PASSTHROUGH_ONLY:
                    raise _error(
                        ErrorCode.UNSUPPORTED_SQL,
                        f"UNION ALL concatenates video and audio only: branch "
                        f"{index + 1} selects a {column.value.type} stream",
                        branches[index],
                        hint="select subtitle and data streams outside the UNION ALL "
                        "(they are copied, never concatenated)",
                    )

    def _check_concat_signature(
        self,
        branches: list[exp.Select],
        lowered: list[list[_Column]],
        flattened: list[list[_Column]],
    ) -> None:
        """Every UNION ALL branch must agree on column count, types and order.

        On the FLATTENED signature: an array column contributes one concat
        column per element, so branches must agree on element counts too. The
        message renders each column as written (``audio[2]`` for an array), so
        a pure length mismatch reads as one.
        """
        expected = [column.value.type for column in flattened[0]]
        for index in range(1, len(flattened)):
            got = [column.value.type for column in flattened[index]]
            if got == expected:
                continue
            raise _error(
                ErrorCode.CONCAT_MISMATCH,
                "UNION ALL branches must select the same stream types in the same "
                f"order: branch 1 selects ({_signature(lowered[0])}), "
                f"branch {index + 1} selects ({_signature(lowered[index])})",
                branches[index],
                hint="ffmpeg concat needs identical segments; reorder or add columns",
            )

    def _concat(self, lowered: list[list[_Column]]) -> list[_Column]:
        """Join branches with one ``concat`` node, interleaved as ffmpeg wants.

        ffmpeg's concat filter takes its inputs per SEGMENT — for ``v=1:a=1``
        that is ``[seg1 v][seg1 a][seg2 v][seg2 a]`` — and produces ``v``
        video pads followed by ``a`` audio pads. The SELECT list of branch 1
        defines the output COLUMN order, which may interleave types
        differently, so the pads are mapped back onto it here.
        """
        first = lowered[0]
        video_positions = [i for i, column in enumerate(first) if column.value.type == "video"]
        audio_positions = [i for i, column in enumerate(first) if column.value.type == "audio"]
        video_count, audio_count = len(video_positions), len(audio_positions)

        inputs: list[FrameRef] = []
        for columns in lowered:
            inputs += [columns[i].value.streams[0].ref for i in video_positions]
            inputs += [columns[i].value.streams[0].ref for i in audio_positions]

        video_pads: list[StreamType] = ["video"] * video_count
        audio_pads: list[StreamType] = ["audio"] * audio_count
        node_id = self.ctx.node(
            "concat",
            {"n": len(lowered), "v": video_count, "a": audio_count},
            inputs,
            video_pads + audio_pads,
        )

        pad_of: dict[int, int] = {}
        for pad, position in enumerate(video_positions):
            pad_of[position] = pad
        for pad, position in enumerate(audio_positions):
            pad_of[position] = video_count + pad

        total = video_count + audio_count
        # A concat pad is fed by one stream per segment; it inherits provenance
        # only where all of them say the same thing (see `_agreed_source`).
        return [
            _Column(
                name=column.name,
                value=_scalar(
                    _Stream(
                        ref=node_id if total == 1 else f"{node_id}:{pad_of[position]}",
                        type=column.value.type,
                        source=_agreed_source(
                            [columns[position].value.streams[0] for columns in lowered]
                        ),
                    )
                ),
            )
            for position, column in enumerate(first)
        ]

    # -- one SELECT branch ------------------------------------------------

    def _lower_branch(self, select: exp.Select, *, tags: _TagScope) -> list[_Column]:
        env = self._scope(select)
        env.grouped = is_grouped(select)
        env.group_keys = _partition_keys(select, env)
        self._check_grouped_cte_columns(select, env)
        # One WHERE clause, three languages. A conjunct over track-row columns
        # is decided HERE and never reaches ffmpeg; a subscript metadata
        # conjunct is a compile-time ASSERTION (nothing to filter -- the SELECT
        # list already names the exact stream the subscript picked); a time
        # window is a seek on an input. Resolve already rejected a conjunct
        # mixing any two, so the split is total -- except for the one admitted
        # mix, a time window bounded by row columns.
        time_conjuncts, row_conjuncts, assertion_conjuncts = self._split_where(select, env)
        fanout = self.fanout_expr is not None
        # A bound naming a row column is one window per row, so it -- like a
        # fan-out's, which waits for the pin -- is read off the relation the
        # WHERE and the ORDER BY leave behind.
        per_row = any(self._is_row_window(conjunct, env) for conjunct in time_conjuncts)
        if not fanout and not per_row:
            self._collect_trims(select, env, time_conjuncts)
        self._filter_rows(row_conjuncts, env, select)
        self._check_assertions(assertion_conjuncts, select)
        self._order_rows(select, env)
        self._limit_rows(select, env)
        self._pin_fanout_row(env, select)
        # What a WITH option read once per row runs over. The pin has already
        # cut a fan-out to its one row, so this is the gathered case alone.
        self.sink_env = env
        self.sink_rows = list(env.relation.tuples) if env.relation is not None else []
        if fanout or per_row:
            self._collect_trims(select, env, time_conjuncts)

        projections = select.expressions
        if not projections:
            raise _error(
                ErrorCode.UNSUPPORTED_SQL, "SELECT has no output column", fallback=select
            )
        columns: list[_Column] = []
        for projection in projections:
            # A star is not an expression, it is a column GENERATOR: it
            # contributes as many columns as the aliases it names have
            # streams, so it expands here rather than in `_lower_expr`.
            qualifier = star_qualifier(projection)
            if qualifier is not None:
                columns += self._expand_star(qualifier, projection, env, select)
                continue
            # The chapter list is a column of the FILE, not a stream and not a
            # tag: one array of chapter records, whatever the branch's rows.
            if _projection_name(projection) == CHAPTERS_COLUMN:
                self._collect_chapters(projection, env, select, scope=tags)
                continue
            # An attached file is not a stream either: ffmpeg reads it by
            # path, so the column is a list of files the output carries.
            if _projection_name(projection) == ATTACHMENTS_COLUMN:
                self._collect_attachments(projection, env, select, scope=tags)
                continue
            # The metadata map produces no stream, so it never becomes an
            # output. With track rows its keys are per-stream, without them
            # they are the container's -- which a CTE body ("rows" scope) has
            # no way to name. A GROUPED branch has rows but no per-row scope,
            # so its map is the group's container.
            if _projection_name(projection) == TAGS_COLUMN:
                self._collect_tags(projection, env, select, scope=tags)
                continue
            # The flag map is the stream's own field, not metadata: it says
            # what the whole map is and emits -disposition.
            if _projection_name(projection) == DISPOSITION_COLUMN and _is_value_column(
                projection, env
            ):
                self._collect_disposition(projection, env, select, scope=tags)
                continue
            # Every other compile-time scalar is a VALUE column: a column of
            # the rows a CTE body produces, readable downstream. A sink writes
            # streams, so one there has nowhere to go.
            if _is_value_column(projection, env):
                if tags == "sink":
                    raise _error(
                        ErrorCode.UNSUPPORTED_SQL,
                        f"'{_projection_name(projection)}' is a value, and a "
                        "SELECT column of a media query is an output stream",
                        projection,
                        fallback=select,
                        hint="metadata is written by a tags column, e.g. "
                        f"STRUCT(... AS {_projection_name(projection)}) AS "
                        f"{TAGS_COLUMN}; a value read by a TO expression or a "
                        "WHERE needs no SELECT column at all",
                    )
                self._collect_value_column(projection, env, select)
                continue
            columns.append(
                _Column(
                    name=_projection_name(projection),
                    value=self._branch_value(projection, env, select),
                    splat=self._is_splat_projection(projection, env),
                )
            )
        if not columns:
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                "every SELECT column is metadata, so the query selects no "
                "stream",
                fallback=select,
                hint="metadata rides on the file the query writes; select its "
                "tracks too, e.g. SELECT t, STRUCT('Main' AS title) AS tags",
            )
        if tags == "sink":
            self._check_one_row_per_file(select, env)
        return columns

    # -- one row, one file -------------------------------------------------

    def _from_rendition_table(self, env: _Env | None) -> bool:
        """True when `env` binds a rendition ladder's row table.

        The fix for "too many rows/streams for one slot" is different for a
        ladder than for a joined CTE or an unnest table: narrow it to one
        rendition (WHERE, or ORDER BY ... LIMIT 1), not restructure the query.
        """
        if env is None:
            return False
        return any(
            isinstance(binding, _RowBinding) and binding.column == RENDITION_COLUMN
            for binding in env.bindings.values()
        )

    def _check_one_row_per_file(self, select: exp.Select, env: _Env) -> None:
        """One row is one file, so a single destination needs a single row.

        The count is the RESOLVED one -- the relation as the WHERE clause and
        the joins left it, partitioned into groups where the branch groups --
        so a row table a predicate narrows to one track writes its one file,
        and rows are combined only where the query says to combine them:
        ``array_agg`` (with ``GROUP BY`` when they share a key), or a fan-out
        ``TO (<expression>)`` that gives each row a destination of its own.

        A fan-out has already been pinned to the one group this command
        writes, so it never reaches the count. A manifest destination is the
        third answer: its one written name binds many outputs, so the rows
        stand -- each becomes a variant map entry. A row-reading sink is a
        fourth: its own arity, not this rule, says how many rows it takes
        (:meth:`_check_row_sink_arity`).
        """
        if (
            self.fanout_expr is not None
            or self.manifest is not None
            or self.row_reading_sink
        ):
            return
        rendition_rows = self._from_rendition_table(env)
        if env.grouped:
            count = len(self._grouped_partitions(env, select))
            what = "group" if count == 1 else "groups"
            hint = _RENDITION_PICK_HINT if rendition_rows else _ONE_FILE_PER_GROUP_HINT
        else:
            count = len(env.relation.tuples) if env.relation is not None else 1
            what = "row" if count == 1 else "rows"
            if rendition_rows:
                hint = _RENDITION_PICK_HINT
            elif any(isinstance(b, _CteBinding) for b in env.bindings.values()):
                hint = _CTE_ROW_FILE_HINT
            elif self.row_window_seen:
                hint = _ROW_WINDOW_FILE_HINT
            else:
                hint = _ONE_FILE_PER_ROW_HINT
        if count <= 1:
            return
        destination = (
            f"'{self.sink_path}' is one file" if self.sink_path else "it writes one file"
        )
        raise _error(
            ErrorCode.ROW_COUNT_MISMATCH,
            f"this query has {count} {what}, and {destination}",
            self.sink_anchor,
            fallback=select,
            hint=hint,
        )

    def _check_grouped_cte_columns(self, select: exp.Select, env: _Env) -> None:
        """Postgres's grouping rule for the columns only lowering can judge.

        Resolve enforces the rule wherever the SQL text settles it -- a track
        row's columns vary within a group, an input alias's do not. A CTE
        column is neither until its body has been lowered: it varies exactly
        when the body produced more than one row and the column carries one
        stream per row. So the same rejection is raised here, with the same
        wording, for the shape resolve could not see.
        """
        if not env.grouped:
            return
        key_texts = {key.sql() for key in group_keys(select)}
        for projection in select.expressions:
            if not isinstance(projection, exp.Expr):
                continue
            star = star_node(projection)
            if star is None:
                self._check_grouped_cte_expr(
                    _projection_expr(projection), env, select, key_texts
                )
            else:
                for _, _, expr in star_replace_entries(star):
                    self._check_grouped_cte_expr(expr, env, select, key_texts)

    def _check_grouped_cte_expr(
        self, node: exp.Expr, env: _Env, select: exp.Select, key_texts: set[str]
    ) -> None:
        """One expression of a grouped branch, recursively."""
        if node.sql() in key_texts or isinstance(node, exp.ArrayAgg):
            return
        if isinstance(node, exp.Column) and not isinstance(node.this, exp.Star):
            table_node = node.args.get("table")
            binding = (
                env.bindings.get(_fold(table_node)) if table_node is not None else None
            )
            name = _fold(node.this)
            if isinstance(binding, _CteBinding) and self._varies_per_row(binding, name):
                raise _error(
                    ErrorCode.UNSUPPORTED_SQL,
                    f"'{binding.name}.{name}' is neither aggregated nor a GROUP "
                    "BY key",
                    node,
                    fallback=select,
                    hint=_GROUPED_CTE_HINT,
                )
            return
        for value in node.args.values():
            items = value if isinstance(value, list) else [value]
            for item in items:
                if isinstance(item, exp.Expr):
                    self._check_grouped_cte_expr(item, env, select, key_texts)

    def _varies_per_row(self, binding: _CteBinding, name: str) -> bool:
        """True when a CTE column carries a stream per body row, and there is
        more than one of them -- the shape that differs tuple by tuple."""
        if binding.rows <= 1:
            return False
        column = self._cte_column(binding, name)
        return column is not None and column.splat and column.value.is_array

    def _branch_value(
        self, projection: exp.Expr, env: _Env, select: exp.Select
    ) -> _Value:
        """One SELECT column's streams, group by group where that matters.

        A grouped branch gathers each group in turn: an aggregate sees its
        whole group, every other column the group's first tuple -- which is
        what makes ``SELECT vid, array_agg(aud) ... GROUP BY vid`` map the
        video once and every audio of its group after it.
        With no partitioning key there is a single group, and the same split
        holds inside it: a group-constant column is mapped ONCE however many
        tuples the relation carries (:meth:`_lower_grouped_table_branch` reads
        it the same way, which is what keeps a table preview and its COPY
        agreeing). Under a fan-out ``TO`` the pin already cut the relation to
        one group. An ungrouped branch lowers over the relation as it stands.
        """
        if not env.grouped:
            return self._lower_expr(projection, env, select)
        relation = env.relation
        if relation is None:  # a query with no rows has nothing to partition
            return self._lower_expr(projection, env, select)
        groups = self._grouped_partitions(env, select)
        if not groups:
            # No row survived: lower the column as it stands, which is where
            # the empty-row-set rejection lives.
            return self._lower_expr(projection, env, select)
        aggregate = _contains_array_agg(_unwrap(projection))
        original = relation.tuples
        gathered: list[_Stream] = []
        stream_type: StreamType = "video"  # every pass overwrites it
        try:
            for group in groups:
                relation.tuples = list(group) if aggregate else group[:1]
                value = self._lower_expr(projection, env, select)
                gathered += value.streams
                stream_type = value.type
        finally:
            relation.tuples = original
        return _array(stream_type, gathered)

    def _is_splat_projection(self, projection: exp.Expr, env: _Env) -> bool:
        """True when this stream column's array value (if it turns out to be
        one) is a row set rather than a single broadcast unit.

        Computed here (at the projection's OWN scope, CTE body or bare SELECT)
        because that is the only place its AST shape is still visible -- an
        outer table query sees just ``<cte>.<name>`` and has to trust what got
        recorded.

        A column is a row set exactly when it READS one: a row alias's stream
        column, a call over one, another CTE's row-set column (which it
        inherits), or an input alias a row-bounded window gave one ``-i`` per
        row. A bare input/source array (``f.audio``) and anything broadcast
        over one is a single row carrying an array VALUE, and an ``array_agg``
        is one unit by definition.
        """
        expr = _unwrap(projection)
        if isinstance(expr, exp.ArrayAgg):
            return False
        return self._reads_row_set(expr, env)

    def _reads_row_set(self, node: exp.Expr, env: _Env) -> bool:
        """True when `node` reads a row alias's column, or a CTE column that
        is itself a row set."""
        for sub in node.walk():
            if not isinstance(sub, exp.Column):
                continue
            table_node = sub.args.get("table")
            if table_node is None:
                continue
            binding = env.bindings.get(_fold(table_node))
            if isinstance(binding, _RowBinding):
                return True
            if isinstance(binding, _InputBinding) and _fold(table_node) in env.row_inputs:
                return True
            if isinstance(binding, _CteBinding):
                column = self._cte_column(binding, _fold(sub.this))
                if column is not None and column.splat and column.value.is_array:
                    return True
        return False

    # -- metadata tag columns ---------------------------------------------

    def _harvest_cte_tags(self, body: exp.Expr) -> None:
        """Move the tags one CTE body just recorded into the carry-over dict.

        ``_lower_query`` clears ``self.tags`` at entry, so a CTE's tags would be
        gone by the time a sink's ``_outputs`` reads them. The clearing itself
        is right -- two COPYs may tag one track differently -- so what the CTE
        recorded moves somewhere that outlives the reset instead.

        The CTE bodies of one script all pour into the SAME dict, though, so
        unlike two COPYs they cannot disagree: whatever any of them says about a
        track is what every sink reading that track sees.
        """
        for source_id, overrides in self.tags.items():
            carried = self.cte_tags.setdefault(source_id, {})
            for key, value in overrides.items():
                if key in carried and carried[key] != value:
                    raise _error(
                        ErrorCode.UNSUPPORTED_SQL,
                        f"tag '{key}' takes two different values on the same track",
                        body,
                        hint="two CTE bodies tag one track's '"
                        f"{key}' differently; give it a single value, or set it "
                        "in the outer SELECT, which overrides them both",
                    )
                carried[key] = value

    def _harvest_cte_dispositions(self, body: exp.Expr) -> None:
        """Move the dispositions one CTE body just recorded into the carry-over
        dict, exactly as `_harvest_cte_tags` does for its tags."""
        for source_id, flags in self.dispositions.items():
            carried = self.cte_dispositions.get(source_id)
            if carried is not None and carried != flags:
                raise _error(
                    ErrorCode.UNSUPPORTED_SQL,
                    "the disposition takes two different values on the same track",
                    body,
                    hint="two CTE bodies flag one track differently; give it a "
                    "single value, or set it in the outer SELECT, which "
                    "overrides them both",
                )
            self.cte_dispositions[source_id] = flags

    def _layered_dispositions(self) -> _DispositionOverrides:
        """The CTE bodies' dispositions with this sink's laid over them."""
        return {**self.cte_dispositions, **self.dispositions}

    def _layered_tags(self) -> _TagOverrides:
        """The CTE bodies' tags with this sink's laid over them, per track.

        Two scopes, written inner to outer, so on a key both set the sink wins.
        That is layering, not the disagreement ``_record_tag`` rejects: that
        check stays inside one query.
        """
        merged: _TagOverrides = {
            source_id: dict(overrides) for source_id, overrides in self.cte_tags.items()
        }
        for source_id, overrides in self.tags.items():
            merged.setdefault(source_id, {}).update(overrides)
        return merged

    def _collect_tags(
        self, projection: exp.Expr, env: _Env, select: exp.Select, *, scope: _TagScope
    ) -> None:
        """``... AS tags``: the metadata keys this column sets.

        A tags column MERGES: it sets the keys it names and leaves every other
        key alone. Over track rows the keys land on that row's streams, over
        input rows on the container. Naming an input's own ``tags`` map copies
        that input's globals through, and an empty ``STRUCT()`` writes none.
        """
        node = _unwrap(projection)
        spec = self._read_tags(node, env, select)
        for key, value_node in spec.entries.items():
            self._check_tag_key(key, value_node, env, select)
        if _has_track_rows(env) and not env.grouped:
            self._collect_stream_tags(spec, node, env, select)
            return
        if scope != "sink":
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"a '{TAGS_COLUMN}' column in a CTE body has no track row to tag",
                projection,
                fallback=select,
                hint="a CTE tags the rows it selects, e.g. FROM "
                "input('f.mkv') f, unnest(f.audio) t; the container's own tags "
                "belong in the outer SELECT",
            )
        self._collect_container_tags(spec, node, env, select)

    def _collect_stream_tags(
        self, spec: _Tags, node: exp.Expr, env: _Env, select: exp.Select
    ) -> None:
        """One tags column over track rows: its keys, per result row, per track.

        A stream keeps the tags it already carries, so there is nothing here
        for an empty map to mean -- only keys to set.
        """
        relation = env.relation
        if relation is None:
            raise _error(
                ErrorCode.UNSUPPORTED_SQL, "malformed tags column", node, fallback=select
            )
        if spec.stripped or not spec.entries:
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"this '{TAGS_COLUMN}' column sets no key",
                node,
                fallback=select,
                hint="per-stream tags set the keys they name, and this map "
                "names none, e.g. STRUCT('Main' AS title) AS tags",
            )
        for key, value_node in spec.entries.items():
            for row in relation.tuples:
                value = self._eval_value(value_node, env, row, select)
                text = None if value is None else _tag_text(value)
                for track in row.values():
                    # A CTE row carries no track of its own: its streams were
                    # tagged by the body that named them.
                    if isinstance(track, _TrackRow):
                        self._record_tag(track.stream.source, key, text, node, select)

    def _collect_container_tags(
        self, spec: _Tags, node: exp.Expr, env: _Env, select: exp.Select
    ) -> None:
        """One tags column over input rows: the file's own global tags.

        ffmpeg copies the first input's globals by default, so naming a source
        (or naming none) is what writes ``-map_metadata``; the keys layer over
        whichever applies.
        """
        if spec.copy_alias is not None and spec.stripped:
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"this '{TAGS_COLUMN}' column both copies and writes no tags",
                node,
                fallback=select,
                hint="copy an input's globals with f.tags, or write none with "
                "STRUCT() -- not both",
            )
        if spec.copy_alias is not None:
            index = self.graph.sources.get(spec.copy_alias)
            if index is None:
                raise _error(
                    ErrorCode.UNSUPPORTED_SQL,
                    f"'{spec.copy_alias}.{TAGS_COLUMN}' names no input()",
                    node,
                    fallback=select,
                    hint="the copied map is an input alias's own, e.g. "
                    "f.tags || STRUCT('Cut' AS title) AS tags",
                )
            self.metadata = index
        elif spec.stripped:
            self.metadata = NO_METADATA
        for key, value_node in spec.entries.items():
            value = self._eval_value(value_node, env, _group_row(env), select)
            text = None if value is None else _tag_text(value)
            if key in self.container_tags and self.container_tags[key] != text:
                raise _error(
                    ErrorCode.UNSUPPORTED_SQL,
                    f"container tag '{key}' takes two different values",
                    value_node,
                    fallback=select,
                    hint="one value per key; a file has one set of container tags",
                )
            self.container_tags[key] = text

    def _read_tags(self, node: exp.Expr, env: _Env, select: exp.Select) -> _Tags:
        """One tags EXPRESSION, read: what it copies and which keys it sets.

        ``a || b`` is the merge, left to right, so a key b names wins over the
        same key in a. An operand is either a struct literal or an alias's own
        ``tags`` map.
        """
        entries: dict[str, exp.Expr] = {}
        copy_alias: str | None = None
        stripped = False
        for operand in _merge_operands(node):
            if isinstance(operand, exp.Struct):
                fields = _struct_fields(operand)
                if not fields:
                    stripped = True
                entries.update(fields)
                continue
            alias = _tags_map_alias(operand)
            if alias is None:
                raise _error(
                    ErrorCode.UNSUPPORTED_SQL,
                    f"a '{TAGS_COLUMN}' column is a map, got {_describe(operand)}",
                    operand,
                    fallback=select,
                    hint="write the keys with STRUCT('Main' AS title) AS tags, "
                    "or copy an input's own map with f.tags || STRUCT(...) AS tags",
                )
            binding = env.bindings.get(alias)
            # A CTE exposes what its body named, and the metadata map is not
            # one of those: it rode the body's streams and is already spent.
            if isinstance(binding, _CteBinding):
                raise _error(
                    ErrorCode.UNSUPPORTED_SQL,
                    f"unknown column '{alias}.{TAGS_COLUMN}'",
                    operand,
                    fallback=select,
                    hint=self._cte_columns_hint(binding),
                )
            # A row alias's map is what already rides through to the output,
            # so copying it names nothing new; only an input's globals do.
            if isinstance(binding, _InputBinding):
                copy_alias = alias
        return _Tags(entries=entries, copy_alias=copy_alias, stripped=stripped)

    def _check_tag_key(
        self, key: str, anchor: exp.Expr, env: _Env, select: exp.Select
    ) -> None:
        """A tags field names a tag KEY, never a probed field or the flag map.

        The reserved set is the read-only field names of whatever the column
        sits over: the file reports those, so a query cannot claim them.
        """
        if key == DISPOSITION_COLUMN:
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"'{DISPOSITION_COLUMN}' is the stream's flag map, not a tag",
                anchor,
                fallback=select,
                hint="write it as its own column, e.g. "
                f"'{DISPOSITION_KEYS[0]}' AS {DISPOSITION_COLUMN}",
            )
        if _has_track_rows(env) and not env.grouped:
            what = "track row"
            reserved = frozenset(
                name
                for binding in env.bindings.values()
                if isinstance(binding, _RowBinding)
                for name in binding.readonly
            )
        else:
            what = "container"
            reserved = CONTAINER_READONLY_FIELDS
        if key not in reserved:
            return
        raise _error(
            ErrorCode.UNSUPPORTED_SQL,
            f"'{key}' is a probed field of the {what}, not something a query "
            "can set",
            anchor,
            fallback=select,
            hint=f"the file reports {key}; a tags field is a free-form key, "
            "e.g. STRUCT('eng' AS language) AS tags",
        )

    def _collect_value_column(
        self, projection: exp.Expr, env: _Env, select: exp.Select
    ) -> None:
        """One VALUE column of a CTE body: its value, once per body row.

        The rows are the branch's relation, so a body cross-joined against a
        series carries one value per series row and a downstream fan-out reads
        the one its pinned row computed.
        """
        name = _projection_name(projection)
        if name is None:  # defensive: `_is_value_column` checked
            raise _error(
                ErrorCode.UNSUPPORTED_SQL, "malformed value column", projection,
                fallback=select,
            )
        node = _unwrap(projection)
        relation = env.relation
        tuples = relation.tuples if relation is not None and relation.tuples else [{}]
        self.branch_values[name] = tuple(
            self._eval_value(node, env, row, select) for row in tuples
        )

    def _collect_disposition(
        self, projection: exp.Expr, env: _Env, select: exp.Select, *, scope: _TagScope
    ) -> None:
        """``... AS disposition``: ffmpeg's own flag spec, per result row.

        The value is the spec ffmpeg takes on the command line -- flag names
        joined by ``+``, or ``'0'`` for none -- and it is ABSOLUTE: it says what
        the output stream's whole flag map is, so every flag it does not name
        is off. NULL says the same as ``'0'``, the way a NULL tag clears its
        key. A container has no disposition, so a branch with no track row to
        flag is a rejection rather than a container write.
        """
        relation = env.relation
        if not _has_track_rows(env) or env.grouped or relation is None:
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"'{DISPOSITION_COLUMN}' is a stream field, not a container one",
                projection,
                fallback=select,
                hint="a disposition rides on a track row, e.g. SELECT t, "
                f"'{DISPOSITION_KEYS[0]}' AS {DISPOSITION_COLUMN} FROM "
                "input('f.mkv') f, unnest(f.audio) t"
                if scope == "sink"
                else "flag the rows a CTE body selects, then gather them outside it",
            )
        value_node = _unwrap(projection)
        for row in relation.tuples:
            flags = self._flag_spec(
                self._eval_value(value_node, env, row, select), projection, select
            )
            for track in row.values():
                if isinstance(track, _TrackRow):
                    self._record_disposition(track.stream.source, flags, projection, select)

    def _flag_spec(
        self, value: RowValue, anchor: exp.Expr, select: exp.Select
    ) -> tuple[str, ...]:
        """One written disposition value as the flags it sets, in declared order.

        ``'default+forced'`` sets those two, ``'0'`` and NULL set none, and a
        name outside the closed set is a rejection naming the ones that are in
        it. Order is the type's, not the writer's, so one flag map has one
        spelling however it was typed.
        """
        if value is None:
            return ()
        if not isinstance(value, str):
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"'{DISPOSITION_COLUMN}' takes ffmpeg's flag spec, not a number",
                anchor,
                fallback=select,
                hint=f"quote the flags, e.g. '{DISPOSITION_KEYS[0]}' or '0' to "
                "clear them",
            )
        if value == "0":
            return ()
        named = set()
        for part in value.split("+"):
            key = part.strip().lower()
            if not key or key.startswith(("+", "-")):
                # ffmpeg's own `+flag`/`-flag` adjusts what the source carries;
                # this column says what the whole map is, so there is nothing
                # for a relative spec to adjust.
                raise _error(
                    ErrorCode.UNSUPPORTED_SQL,
                    f"'{value}' is not a flag list",
                    anchor,
                    fallback=select,
                    hint="name every flag the track should have, joined with "
                    f"'+', e.g. '{DISPOSITION_KEYS[0]}+{DISPOSITION_KEYS[6]}'; "
                    "'0' clears them all",
                )
            if key not in DISPOSITION_KEYS:
                raise flag_error(part.strip(), key, anchor, select)
            named.add(key)
        return tuple(key for key in DISPOSITION_KEYS if key in named)

    def _record_disposition(
        self,
        source: StreamMeta | None,
        flags: tuple[str, ...],
        anchor: exp.Expr,
        select: exp.Select,
    ) -> None:
        """Note one track's disposition; disagreement is a rejection.

        Keyed like `_record_tag`, by the identity of the probed StreamMeta, so
        the flags find their track through any chain of filters.
        """
        if source is None:
            return
        recorded = self.dispositions.get(id(source))
        if recorded is not None and recorded != flags:
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                "the disposition takes two different values on the same track",
                anchor,
                fallback=select,
                hint="a disposition is row-scoped, so a track selected by "
                "several result rows must get the same flags in each",
            )
        self.dispositions[id(source)] = flags

    def _record_tag(
        self,
        source: StreamMeta | None,
        key: str,
        value: str | None,
        anchor: exp.Expr,
        select: exp.Select,
    ) -> None:
        """Note one track's override for one key; disagreement is a rejection.

        Keyed by the identity of the probed :class:`StreamMeta`, which is the
        same thing :func:`_provenance` reads off an output stream — so an
        override finds its track through any chain of filters that threads
        provenance, not just a passthrough. The probes hold every StreamMeta for
        the whole lowering, so the ids stay valid.
        """
        if source is None:
            return
        overrides = self.tags.setdefault(id(source), {})
        if key in overrides and overrides[key] != value:
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"tag '{key}' takes two different values on the same track",
                anchor,
                fallback=select,
                hint="a tag is row-scoped, so a track selected by several result "
                "rows must get the same value in each",
            )
        overrides[key] = value

    # -- SELECT * / <alias>.* ------------------------------------

    def _expand_star(
        self, qualifier: str, anchor: exp.Expr, env: _Env, select: exp.Select
    ) -> list[_Column]:
        """Every stream a star stands for, as passthrough columns.

        A bare ``*`` takes every alias of the FROM clause in FROM order
        (``_Env.bindings`` is insertion-ordered and built by `_scope` in exactly
        that order); ``<alias>.*`` takes one. Within an alias: the container's
        stream array columns in v/a/s/d order for an input, COLUMN order for a
        CTE, with array columns splatting.

        The WHERE window of each alias still applies: for an input alias it is
        already on the ``-i`` (so ``SELECT *`` under a WHERE seeks every stream
        of the file, captions included), for a CTE it is the filter trim
        `_access` splices — which is also where a trimmed CTE caption column is
        rejected.

        EXCEPT/REPLACE (borrowed from BigQuery) narrow or override the result
        by IDENTITY: an input or generated-source stream's identity is its
        kind (``video``/``audio``/``subtitle``/``data`` -- passthrough columns
        carry no name of their own, so a kind is all EXCEPT/REPLACE has to
        aim at, and both drop or replace EVERY stream of a repeated kind), a
        CTE column's is the name its body gave it with ``AS``. A REPLACE
        expression lowers once PER MATCHING SLOT, same as writing it out that
        many times by hand -- two streams of one kind sharing a REPLACE are
        two independent nodes, split downstream like any other reused source.
        """
        star = star_node(anchor)
        except_entries = star_except_entries(star) if star is not None else []
        replace_entries = star_replace_entries(star) if star is not None else []
        except_names = {name for name, _ in except_entries}
        replace_map = {name: expr for name, _, expr in replace_entries}
        # The star's VOCABULARY, not just what this file happens to hold: an
        # input's four kinds are always nameable, a video-less file included --
        # EXCEPT(subtitle) on one with none is a no-op, exactly like a bare
        # `*` already silently skips a kind with nothing in it.
        vocabulary: set[str] = set()
        columns: list[_Column] = []

        def slot(identity: str, build: Callable[[], _Column]) -> None:
            if identity in except_names:
                return
            if identity in replace_map:
                columns.append(
                    _Column(name=None, value=self._lower_expr(replace_map[identity], env, select))
                )
                return
            columns.append(build())

        def input_thunk(alias: str, meta: StreamMeta) -> Callable[[], _Column]:
            return lambda: self._star_input_column(alias, meta, anchor, env, select)

        def source_thunk(binding: _SourceBinding) -> Callable[[], _Column]:
            return lambda: _Column(name=None, value=_scalar(self._source_stream_of(binding)))

        def cte_thunk(column: _Column) -> Callable[[], _Column]:
            return lambda: column

        for binding in self._star_bindings(qualifier, anchor, env, select):
            if isinstance(binding, _RowBinding):
                raise _row_star_error(binding, anchor, select)
            if isinstance(binding, _InputBinding):
                vocabulary |= set(_STREAM_STAR_COLUMNS)
                for kind, meta in self._star_input(binding.alias, anchor, env, select):
                    slot(kind, input_thunk(binding.alias, meta))
            elif isinstance(binding, _SourceBinding):
                # A source has exactly one stream, so its star is that one
                # column -- statically, like everything else about it.
                vocabulary.add(binding.output)
                slot(binding.output, source_thunk(binding))
            else:
                for name, column in self._star_cte(binding, anchor, env, select):
                    vocabulary.add(name)
                    slot(name, cte_thunk(column))

        for name, item_anchor in except_entries + [(n, a) for n, a, _ in replace_entries]:
            if name not in vocabulary:
                raise _error(
                    ErrorCode.UNSUPPORTED_SQL,
                    f"'{name}' is not a column '*' expands to here",
                    item_anchor,
                    fallback=select,
                    hint=self._star_holdings_hint(vocabulary),
                )
        return columns

    def _star_holdings_hint(self, vocabulary: set[str]) -> str:
        if not vocabulary:
            return "'*' expands to nothing here"
        return f"'*' holds: {', '.join(sorted(vocabulary))}"

    def _star_probe(self, alias: str, anchor: exp.Expr, select: exp.Select) -> ProbeResult:
        """The probe a star over an input alias needs, or INPUT_NOT_FOUND.

        Splat tier, same policy as a bare ``a.audio``: how many streams a
        file has, and of which types, is a property of the file, so an input
        that could not be probed is a rejection rather than a guess.
        """
        result = self.probes.get(alias)
        if result is None:
            path = self.res.input_paths[self.graph.sources[alias]]
            raise self._unreadable_error(
                ErrorCode.INPUT_NOT_FOUND,
                alias,
                f"cannot expand '*' over '{path}'",
                anchor,
                select,
                hint="'*' is every stream of the input, and only a readable input "
                f"can list them; name the streams instead, e.g. {alias}.video[1]",
            )
        return result

    def _star_input(
        self, alias: str, anchor: exp.Expr, env: _Env, select: exp.Select
    ) -> list[tuple[str, StreamMeta]]:
        """Every stream of one input alias: the stream arrays, in v/a/s/d order.

        The container's array columns are what a star stands for, and a media
        SELECT column is an output stream, so the four stream arrays expand and
        `chapters` does not -- a chapter is not a stream, and ffmpeg's own
        default already carries an input's chapters through a remux.

        Returns ``(kind, probed metadata)`` pairs rather than built columns:
        building one is split out to :meth:`_star_input_column` so a stream
        EXCEPT drops, or REPLACE overrides, never reaches the codecless check
        or the WHERE-window access at all.
        """
        result = self._star_probe(alias, anchor, select)
        path = self.res.input_paths[self.graph.sources[alias]]
        streams = [
            meta
            for column in _STREAM_STAR_COLUMNS
            for meta in result.by_type(_ARRAY_COLUMNS[column])
        ]
        if not streams:
            raise _error(
                ErrorCode.STREAM_NOT_FOUND,
                f"'*' over '{path}' selects nothing: it has no video, audio, "
                "subtitle or data streams",
                anchor,
                fallback=select,
                hint="an empty expansion would select nothing; drop the star",
            )
        return [(meta.type, meta) for meta in streams]

    def _star_input_column(
        self,
        alias: str,
        meta: StreamMeta,
        anchor: exp.Expr,
        env: _Env,
        select: exp.Select,
    ) -> _Column:
        """One passthrough column of one input alias's star expansion."""
        self._reject_codecless(
            meta,
            f"'{alias}.*' includes '{alias}.{meta.type}[{meta.index + 1}]', which",
            anchor,
            select,
        )
        row_inputs = env.row_inputs.get(alias)
        return _Column(
            name=None,
            value=self._access(
                env,
                alias,
                _scalar(self._source_stream(alias, meta.type, meta.index))
                if row_inputs is None
                else _array(
                    meta.type,
                    [
                        self._source_stream(source, meta.type, meta.index)
                        for source in row_inputs
                    ],
                ),
                anchor,
                select,
            ),
        )

    def _star_cte(
        self, binding: _CteBinding, anchor: exp.Expr, env: _Env, select: exp.Select
    ) -> list[tuple[str, _Column]]:
        """A CTE's columns, in order, arrays splatted. No probe is involved.

        A CTE's shape was fixed when its body lowered, so this is static — the
        same information `<cte>.<name>` already reads. Column names are kept:
        the star selects the columns the CTE named, not anonymous streams --
        and are the identity EXCEPT/REPLACE match here.
        """
        return [
            (
                column.name or "",
                _Column(
                    name=column.name,
                    value=self._access(
                        env, binding.name, _scalar(stream), anchor, select
                    ),
                ),
            )
            for column in binding.columns
            for stream in self._cte_column_value(
                binding, column, anchor, select
            ).streams
        ]

    def _star_bindings(
        self, qualifier: str, anchor: exp.Expr, env: _Env, select: exp.Select
    ) -> list[_Binding]:
        """What a star stands for: one named alias, or every FROM alias."""
        if not qualifier:
            return list(env.bindings.values())
        binding = env.bindings.get(qualifier)
        if binding is None:
            raise _error(
                ErrorCode.UNKNOWN_ALIAS,
                f"unknown alias '{qualifier}'",
                anchor,
                fallback=select,
                hint=self._known_hint(),
            )
        return [binding]

    def _check_star_table_mode(self, anchor: exp.Expr, select: exp.Select) -> None:
        """EXCEPT/REPLACE narrow a MEDIA star's stream expansion; a table
        query's star prints record fields instead, which they do not reach."""
        star = star_node(anchor)
        if star is not None and (star.args.get("except_") or star.args.get("replace")):
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                "EXCEPT/REPLACE are not supported on a table query's '*'",
                anchor,
                fallback=select,
                hint="write the columns out, or drop the modifier",
            )

    def _star_names(
        self, qualifier: str, anchor: exp.Expr, env: _Env, select: exp.Select
    ) -> list[str]:
        """A table star's column headers. Static: no probe is consulted.

        A container names its array columns, a row table its record's scalar
        fields, a CTE the columns its body named, and a generated source the
        one array column its output type fills. `_star_cells` walks the very
        same lists in the same order.
        """
        self._check_star_table_mode(anchor, select)
        names: list[str] = []
        for binding in self._star_bindings(qualifier, anchor, env, select):
            if isinstance(binding, _RowBinding):
                names += binding.star
            elif isinstance(binding, _InputBinding):
                names += STAR_COLUMNS
            elif isinstance(binding, _SourceBinding):
                names.append(binding.output)
            else:
                names += [column.name or "column" for column in binding.columns]
        return names

    def _star_cells(
        self,
        qualifier: str,
        anchor: exp.Expr,
        env: _Env,
        select: exp.Select,
        cardinality: int,
    ) -> list[list[CellValue]]:
        """A table star's columns, each already one cell per printed row."""
        columns: list[list[CellValue]] = []
        for binding in self._star_bindings(qualifier, anchor, env, select):
            if isinstance(binding, _RowBinding):
                columns += [
                    self._row_metadata_cells(binding, name, anchor, select)
                    for name in binding.star
                ]
            elif isinstance(binding, _InputBinding):
                columns += [
                    self._input_array_cells(
                        binding.alias, name, anchor, env, select, cardinality
                    )
                    for name in STAR_COLUMNS
                ]
            elif isinstance(binding, _SourceBinding):
                cell = ArrayCell(
                    elements=(self._stream_to_cell(self._source_stream_of(binding)),)
                )
                columns.append([cell] * cardinality)
            else:
                columns += [
                    self._value_to_cells(
                        self._access(
                            env,
                            binding.name,
                            self._cte_column_value(binding, column, anchor, select),
                            anchor,
                            select,
                        ),
                        cardinality,
                        splat=column.splat,
                    )
                    for column in binding.columns
                ]
        return columns

    def _input_array_cells(
        self,
        alias: str,
        column: str,
        anchor: exp.Expr,
        env: _Env,
        select: exp.Select,
        cardinality: int,
    ) -> list[CellValue]:
        """One container array column as ONE array cell, broadcast to each row.

        The same cell a bare ``f.audio`` / ``f.chapters`` prints on its own: an
        array column is a value inside the input's single row, not a row set.
        Unless a row-bounded window gave the alias an ``-i`` per row, in which
        case each row prints the streams IT reads -- the same thing
        ``SELECT f.audio`` prints for that row.
        """
        if column in RECORD_ARRAY_COLUMNS:
            return self._record_cells(alias, column, anchor, select, cardinality)
        result = self._star_probe(alias, anchor, select)
        stream_type = _ARRAY_COLUMNS[column]
        indices = [meta.index for meta in result.by_type(stream_type)]

        def cell_of(source: str) -> CellValue:
            streams = [
                self._source_stream(source, stream_type, index) for index in indices
            ]
            if streams:
                streams = list(
                    self._access(
                        env, alias, _array(stream_type, streams), anchor, select
                    ).streams
                )
            return ArrayCell(
                elements=tuple(self._stream_to_cell(stream) for stream in streams)
            )

        row_inputs = env.row_inputs.get(alias)
        if row_inputs is not None and len(row_inputs) == cardinality:
            return [cell_of(source) for source in row_inputs]
        return [cell_of(alias)] * cardinality

    # -- FROM -------------------------------------------------------------

    def _scope(self, select: exp.Select) -> _Env:
        env = _Env()
        from_ = select.args.get("from_")
        if not isinstance(from_, exp.From):
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                "SELECT requires a FROM clause",
                fallback=select,
                hint="add FROM input('clip.mp4') a",
            )
        for item, join in from_entries(select):
            if isinstance(item, exp.Unnest):
                alias_node = item.args.get("alias")
                alias = (
                    _fold(alias_node.this)
                    if isinstance(alias_node, exp.TableAlias)
                    and alias_node.this is not None
                    else ""
                )
                struct_values = self.res.struct_rows.get(alias)
                if struct_values is not None:
                    self._add_values_rows(alias, struct_values, env, select, join)
                else:
                    self._add_track_rows(item, join, env, select)
            else:
                self._add_table(item, join, env, select)
        return env

    # -- FROM unnest(<input>.<type>) alias -------------

    def _add_track_rows(
        self,
        unnest: exp.Unnest,
        join: RawRowJoin | None,
        env: _Env,
        select: exp.Select,
    ) -> None:
        """Bind one track-row table: every track of the array becomes a row.

        This is the one binding that MUST probe. A row's columns are probed
        metadata and its row COUNT is a property of the file, so an input that
        could not be read cannot be unnested at all -- the same policy, and the
        same code, a bare ``f.audio`` has: the streams of a file that cannot be
        read cannot be enumerated.

        No node is minted and no ``-i`` is taken: the rows' streams are the
        INPUT alias's streams, already probed and already mapped, so a row
        table is pure bookkeeping until ``t`` is actually selected. That
        is what makes the consume-once rule fall out of ordinary column
        selection -- an unmatched row's stream is simply never read.
        """
        alias_node = unnest.args.get("alias")
        alias = (
            _fold(alias_node.this)
            if isinstance(alias_node, exp.TableAlias) and alias_node.this is not None
            else ""
        )
        raw = self.res.track_rows.get(alias)
        if raw is None:  # defensive: resolve records every row alias
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                "malformed unnest in FROM",
                unnest,
                fallback=select,
                hint="unnest one input's stream array, e.g. unnest(f.audio) t",
            )
        if raw.column in RECORD_ARRAY_COLUMNS:
            stream_type: StreamType = "data"  # filler: a record row has no track
            rows = self._record_rows(raw, unnest, select)
        else:
            stream_type = _ARRAY_COLUMNS[raw.column]
            result = self.probes.get(raw.source)
            if result is None:
                raise self._unreadable_error(
                    ErrorCode.INPUT_NOT_FOUND,
                    raw.source,
                    f"cannot unnest '{raw.source}.{raw.column}' of "
                    f"'{self._path_of(raw.source)}'",
                    unnest,
                    select,
                    hint=f"unnest lists the tracks of a file and reads their "
                    f"metadata, and only a readable input has either; subscript "
                    f"one stream instead, e.g. {raw.source}.{raw.column}[1]",
                )
            rows = [
                _TrackRow(
                    stream=self._source_stream(raw.source, stream_type, position),
                    columns=_row_columns(meta, raw.column),
                )
                for position, meta in enumerate(result.by_type(stream_type))
            ]
        if env.relation is None:
            env.relation = _RowRelation()
        env.bindings[alias] = _RowBinding(
            alias=alias,
            source=raw.source,
            column=raw.column,
            type=stream_type,
            relation=env.relation,
        )
        self._join_rows(env.relation, alias, rows, join, env, select)

    # -- FROM input(<manifest>) alias, over an ABR ladder ------------------

    def _bind_renditions(
        self,
        alias: str,
        join: RawRowJoin | None,
        env: _Env,
        select: exp.Select,
    ) -> None:
        """An input alias whose probe found renditions is ALSO a track-row
        table: one row per ``RenditionMeta``, no ``unnest`` needed to ask for
        it -- an ABR ladder's variants are rows the same way a plain file's
        tracks are once ``unnest(<input>.<type>)`` names them.

        Replaces the plain ``_InputBinding`` bound just above: once an alias
        has rendition rows, the row table IS what ``alias.<column>`` means,
        not a second thing beside it. `source` names itself, so the ``-i``,
        its WHERE window and its provenance stay keyed off the same alias a
        plain input would use. An input with no renditions leaves the plain
        binding untouched.
        """
        result = self.probes.get(alias)
        if result is None or not result.renditions:
            return
        rows = [self._rendition_row(alias, rendition) for rendition in result.renditions]
        if env.relation is None:
            env.relation = _RowRelation()
        env.bindings[alias] = _RowBinding(
            alias=alias,
            source=alias,
            column=RENDITION_COLUMN,
            type=rows[0].stream.type,
            relation=env.relation,
        )
        self._join_rows(env.relation, alias, rows, join, env, select)

    def _rendition_row(self, alias: str, rendition: RenditionMeta) -> _TrackRow:
        """One ladder rung as a track row: its streams by kind, plus the
        ABR metadata ``RenditionMeta`` itself carries.

        `stream` is the row's primary track -- its first video stream, else
        its first audio one -- so a bare row still means one thing, exactly
        as an unnest row's does. Every stream is built the same way the
        unnest path builds one (:meth:`_source_stream`, keyed by the
        `StreamMeta`'s own per-type `index`), so emission maps it identically.
        """
        kinds: dict[StreamType, _Stream] = {}
        for meta in rendition.streams:
            kinds.setdefault(meta.type, self._source_stream(alias, meta.type, meta.index))
        primary = kinds.get("video") or kinds.get("audio") or next(iter(kinds.values()), None)
        return _TrackRow(
            stream=primary if primary is not None else _STREAMLESS_ROW,
            columns={
                "bandwidth": rendition.bandwidth,
                "width": rendition.width,
                "height": rendition.height,
                "codecs": rendition.codecs,
                "name": rendition.name,
                "language": rendition.language,
            },
            kinds=kinds,
        )

    # -- FROM <source>(<values>) alias: a RETURNS source call --------------

    def _add_module_source(
        self,
        alias: str,
        inner: exp.Anonymous,
        declared: WasmFunction,
        join: RawRowJoin | None,
        env: _Env,
        select: exp.Select,
    ) -> None:
        """``FROM <source>(<values>) alias`` -- the mirror of a sink call,
        bound exactly as ``input()`` binds.

        The call's value arguments fold into the module's own parameters the
        same way a sink's do (:meth:`_wasm_params`), and the sidecar is asked
        ONCE, at compile time, for the catalog those parameters describe
        (:func:`~ffrwd.wasm.probe_source`). The catalog becomes this alias's
        :class:`~ffrwd.probe.ProbeResult`
        (:func:`~ffrwd.wasm.catalog_as_probe`) -- one row per rendition,
        never zero -- so it binds through :meth:`_bind_renditions` exactly as
        a probed manifest does: no new relation kind, ``s.video``,
        ``s.bandwidth``, WHERE/ORDER BY/LIMIT and the one-row rule all read
        it the same way. :attr:`Graph.module_sources` records the same
        catalog as IR, the mirror of :attr:`Graph.packet_sinks`.
        """
        call = _call_parts(inner)
        assert call is not None  # inner is exp.Anonymous; _call_parts always answers
        if call.named:
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"{declared.name}() does not take named arguments",
                call.named[0].value,
                fallback=inner,
                hint=f"a wasm function's parameters are positional: "
                f"{declared.signature}",
            )
        described = self._described_source(declared, inner, select)
        params = self._wasm_params(
            declared, described, call, inner, select, env, {}, first=0
        )
        params_json = json.dumps(params, sort_keys=True)
        try:
            catalog = self.probe_source(declared.module, params_json)
        except FfrwdError as err:
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"cannot read '{alias}': {err.message}",
                inner,
                fallback=select,
                hint=err.hint,
            ) from err
        result = catalog_as_probe(alias, catalog)
        self.probes[alias] = result
        env.bindings[alias] = _InputBinding(alias=alias)
        self._bind_renditions(alias, join, env, select)
        self.graph.module_sources[alias] = ModuleSource(
            alias=alias,
            module=declared.module,
            params=params_json,
            tracks=tuple(
                IrSourceTrack(
                    ref=f"src:{alias}:{_TYPE_MARKERS[stream.type]}:{stream.index}",
                    kind=track.kind,
                    codec=track.codec,
                    time_base=track.time_base,
                    row=track.row,
                    name=track.rendition.name,
                    bandwidth=track.rendition.bandwidth,
                    codecs=track.rendition.codecs,
                    language=track.rendition.language,
                )
                for track, stream in zip(catalog.tracks, result.streams, strict=True)
            ),
            bounded=catalog.bounded,
        )

    def _described_source(
        self, declared: WasmFunction, node: exp.Expr, select: exp.Select
    ) -> Described:
        """What a ``RETURNS source`` call's module declares, checked.

        The source mirror of :meth:`_described`, checked against its OWN
        rules rather than reused whole: a source reads no streams and emits
        no per-frame annotations, so the filter-shaped checks
        :meth:`_described` runs after the world/export match --
        :meth:`_check_stream_arity` chief among them, which would read
        ``described.inputs`` as if it were a filter's pad count -- have
        nothing to check here and would misjudge a module that correctly
        reads none at all.
        """
        described = self.describes.get(declared.module)
        if described is None:
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"the module '{declared.module}' was never described",
                node,
                fallback=select,
                hint="this is a compiler bug; please report the query that "
                "produced it",
            )
        if described.world not in WORLDS:
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"the module '{declared.module}' targets {described.world}, and "
                f"this ffrwd hosts {' or '.join(WORLDS)}",
                node,
                fallback=select,
                hint="rebuild the module against a world this ffrwd hosts, or "
                "upgrade ffrwd",
            )
        if described.name != declared.export:
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"function '{declared.name}' names the export '{declared.export}', "
                f"and '{declared.module}' exports '{described.name}'",
                node,
                fallback=select,
                hint=f"a module carries one filter; write '{described.name}' as "
                "the export",
            )
        if not hosts_packet_source(described.world):
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"the module '{declared.module}' produces packets, and the "
                f"sidecar's {described.world} cannot host one",
                node,
                fallback=select,
                hint="packet sources arrived with ffrwd:av@0.13.0; upgrade "
                "ffrwd, or point at a newer ffrwd-wasm",
            )
        if not described.source:
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"function '{declared.name}' declares RETURNS source, and the "
                f"module '{declared.module}' is not a packet source",
                node,
                fallback=select,
                hint=f"'{declared.module}' has to export a packet source built "
                "RETURNS source; check the module and the export named",
            )
        return described

    # -- joining two row tables ------------------------

    def _join_rows(
        self,
        relation: _RowRelation,
        alias: str,
        rows: Sequence[_TrackRow | _CteRow],
        join: RawRowJoin | None,
        env: _Env,
        select: exp.Select,
    ) -> None:
        """Fold one freshly bound row source into the branch's relation.

        Ordinary SQL join semantics, evaluated here because every column is
        probed metadata ("the joins never reach ffmpeg"):

        * the FIRST row table simply becomes the relation;
        * a comma between two row tables is the bounded CROSS join;
        * ``ON`` is 061's three-valued evaluator, and a pair is kept only when
          it comes back TRUE — so a NULL key matches nothing, without that
          being a rule of ours;
        * multiplicity is real: a left row matching two right rows pairs with
          BOTH (two result rows, hence two output streams). The fix, when that
          is not wanted, is a wider key, not an error;
        * LEFT keeps an unmatched left row with a NULL right side, FULL also
          appends the unmatched RIGHT rows, in their own order, after every
          left row -- which is the whole of the row-order rule.
        """
        kind = join.kind if join is not None else "cross"
        if not relation.aliases:
            relation.aliases.append(alias)
            relation.tuples = [{alias: row} for row in rows]
            return
        if join is not None and join.on is not None:
            for key_alias, names in _join_keys(join.on).items():
                for name in names:
                    if name not in relation.keys.setdefault(key_alias, []):
                        relation.keys[key_alias].append(name)

        combined: list[_RowTuple] = []
        matched: set[int] = set()
        for left in relation.tuples:
            paired = False
            for position, row in enumerate(rows):
                candidate: _RowTuple = {**left, alias: row}
                if kind != "cross" and (
                    join is None
                    or join.on is None
                    or self._eval_row(join.on, env, candidate, select) is not True
                ):
                    continue
                combined.append(candidate)
                matched.add(position)
                paired = True
            if not paired and kind in ("left", "full"):
                combined.append({**left, alias: None})
        if kind == "full":
            empty: _RowTuple = {name: None for name in relation.aliases}
            combined += [
                {**empty, alias: row}
                for position, row in enumerate(rows)
                if position not in matched
            ]
        relation.aliases.append(alias)
        relation.tuples = combined

    def _path_of(self, alias: str) -> str:
        """The path behind an input alias, for a message about its file."""
        index = self.graph.sources.get(alias)
        if index is None or not 0 <= index < len(self.res.input_paths):
            return alias
        return self.res.input_paths[index]

    def _unreadable_error(
        self,
        code: ErrorCode,
        alias: str,
        lead: str,
        anchor: exp.Expr | None,
        select: exp.Select,
        hint: str,
    ) -> FfrwdError:
        """The rejection for an input `self.probes` has no result for.

        Two stories, chosen by whether the input's path actually exists:
        one that never had a file to read -- missing, a bad permission, a
        typo -- keeps the familiar "file not found or unreadable" and the
        caller's own `hint`. One that DOES exist, whose probe failed anyway,
        says so honestly instead: ffprobe's own last line when there is one,
        and a hint that points at the probe rather than at an existence
        `hint` would wrongly imply is in question.
        """
        failure = self.probe_failures.get(alias)
        if failure is None:
            return _error(code, f"{lead}: file not found or unreadable", anchor,
                           fallback=select, hint=hint)
        detail = failure.stderr or "ffprobe exited without reporting why"
        return _error(
            code,
            f"{lead}: the probe failed ({detail})",
            anchor,
            fallback=select,
            hint="the input exists but ffprobe could not read it with the "
            "options given; run ffprobe on it directly, with the same "
            "options, to see why",
        )

    def _record_rows(
        self, raw: RawTrackRows, unnest: exp.Expr, select: exp.Select
    ) -> list[_TrackRow]:
        """The rows of ``unnest(<input>.chapters)`` / ``unnest(<input>.cues)``.

        The array columns whose elements are not streams, so every row carries
        `_STREAMLESS_ROW` in place of a track and only the record's own
        metadata columns are ever read.
        """
        result = self._record_probe(
            raw.source,
            raw.column,
            unnest,
            select,
            hint=f"unnest({raw.source}.{raw.column}) lists a file's "
            f"{raw.column}, and only a readable input has any",
        )
        return [
            _TrackRow(stream=_STREAMLESS_ROW, columns=columns)
            for columns in _record_columns(result, raw.column)
        ]

    def _record_probe(
        self,
        alias: str,
        column: str,
        anchor: exp.Expr,
        select: exp.Select,
        *,
        hint: str,
    ) -> ProbeResult:
        """The probe a record array column reads, or the rejection for it.

        Cues are the one column ffprobe does not answer: it reports a WebVTT
        file's single subtitle stream and never the cues in it, so ffrwd
        parses the document itself and only a WebVTT input has any. A
        container that merely CARRIES a webvtt track is not read.
        """
        result = self.probes.get(alias)
        if result is None:
            raise self._unreadable_error(
                ErrorCode.INPUT_NOT_FOUND,
                alias,
                f"cannot read {column} of '{self._path_of(alias)}'",
                anchor,
                select,
                hint=hint,
            )
        if column == CUES_COLUMN and result.format_name != WEBVTT_FORMAT:
            reported = result.format_name or "unreadable"
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"'{self._path_of(alias)}' is {reported}, not WebVTT, so it "
                "has no cues to read",
                anchor,
                fallback=select,
                hint="ffrwd reads cues out of a WebVTT document, so they "
                "come from a .vtt input, e.g. input('subs.en.vtt'); a webvtt "
                "track inside a container is not read",
            )
        return result

    def _add_table(
        self,
        table: exp.Expr | None,
        join: RawRowJoin | None,
        env: _Env,
        select: exp.Select,
    ) -> None:
        if not isinstance(table, exp.Table):
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                _FROM_ITEM_MESSAGE,
                table,
                fallback=select,
            )
        inner = table.this
        alias_node = table.args.get("alias")
        db = table.args.get("db")
        if isinstance(db, exp.Expr) and _fold(db) == FILTER_NAMESPACE:
            # `FROM ffmpeg.<source>(...) alias`: resolve already
            # shape-checked it and parked the record in `res.source_filters`.
            self._add_source(table, alias_node, env, select)
            return
        if isinstance(inner, exp.GenerateSeries):
            if not isinstance(alias_node, exp.TableAlias) or alias_node.this is None:
                raise _error(  # defensive: resolve already required one
                    ErrorCode.UNSUPPORTED_SQL,
                    "generate_series(...) requires an alias",
                    table,
                    fallback=select,
                )
            alias = _fold(alias_node.this)
            series_values = self.res.series.get(alias)
            if series_values is None:  # defensive: resolve records every series alias
                raise _error(
                    ErrorCode.UNKNOWN_ALIAS,
                    f"unknown alias '{alias}'",
                    alias_node,
                    fallback=table,
                    hint=self._known_hint(),
                )
            self._add_series_rows(alias, series_values, inner, env, select, join)
            return
        if isinstance(inner, exp.Anonymous):
            declared = self.res.wasm.get(str(inner.this).lower())
            if declared is not None and declared.is_source:
                if not isinstance(alias_node, exp.TableAlias) or alias_node.this is None:
                    raise _error(
                        ErrorCode.UNSUPPORTED_SQL,
                        f"{declared.name}() requires an alias",
                        table,
                        fallback=select,
                        hint=f"add an alias, e.g. FROM {declared.name}(...) s",
                    )
                alias = _fold(alias_node.this)
                self._add_module_source(alias, inner, declared, join, env, select)
                return
            if not isinstance(alias_node, exp.TableAlias) or alias_node.this is None:
                raise _error(
                    ErrorCode.UNSUPPORTED_SQL,
                    "input() requires an alias",
                    table,
                    fallback=select,
                    hint="add an alias, e.g. FROM input('clip.mp4') a",
                )
            alias = _fold(alias_node.this)
            if alias not in self.graph.sources:
                raise _error(
                    ErrorCode.UNKNOWN_ALIAS, f"unknown alias '{alias}'", alias_node, fallback=table
                )
            env.bindings[alias] = _InputBinding(alias=alias)
            self._bind_renditions(alias, join, env, select)
            return
        if isinstance(inner, exp.Identifier):
            name = _fold(inner)
            columns = self.cte_columns.get(name)
            body_values = self.cte_values.get(name, {})
            if columns is None:
                raise _error(
                    ErrorCode.UNKNOWN_ALIAS,
                    f"unknown table '{name}'",
                    inner,
                    fallback=table,
                    hint=self._known_hint(),
                )
            # `FROM master m` binds the view/CTE under a BRANCH-LOCAL name
            # (resolve checked it shadows nothing in the flat namespace). The
            # binding records the local name, so `m.v` resolves and messages
            # read back as written; the columns -- and therefore the graph
            # refs -- are the same objects either way, which is what makes the
            # shared subgraph shared.
            local = name
            if isinstance(alias_node, exp.TableAlias) and alias_node.this is not None:
                local = _fold(alias_node.this)
            self._add_cte_rows(local, columns, body_values, env, select, join)
            return
        raise _error(
            ErrorCode.UNSUPPORTED_SQL,
            _FROM_ITEM_MESSAGE,
            table,
            fallback=select,
        )

    def _add_values_rows(
        self,
        local: str,
        values: RawValuesTable,
        env: _Env,
        select: exp.Select,
        join: RawRowJoin | None = None,
    ) -> None:
        """Bind one written row table: its rows join the branch's relation.

        The same join :meth:`_add_track_rows` builds, with the rows read off
        the query instead of a probe -- so a comma between a written row
        table and anything else is the ordinary cross join, an explicit
        `join` matches rows the same way it does between unnest tables, and
        ``array_agg`` over it aggregates the same way. No stream and no
        ``-i``: the rows are values. Each cell takes the ordinary
        compile-time value grammar (:meth:`_eval_value`), evaluated once
        over the branch's representative row -- a ``generate_series`` cell
        is always a literal, so this is the identity for it; a struct row
        table's cell may be an expression over one.
        """
        if env.relation is None:
            env.relation = _RowRelation()
        group_row = _group_row(env)
        rows = [
            _TrackRow(
                stream=_STREAMLESS_ROW,
                columns={
                    name: self._eval_value(cell, env, group_row, select)
                    for name, cell in zip(values.columns, entry, strict=True)
                },
            )
            for entry in values.rows
        ]
        env.bindings[local] = _RowBinding(
            alias=local,
            source="",
            column=local,
            type="data",  # filler: a written row has no track
            relation=env.relation,
            values=values,
        )
        self._join_rows(env.relation, local, rows, join, env, select)

    def _add_series_rows(
        self,
        local: str,
        values: tuple[int, ...],
        node: exp.Expr,
        env: _Env,
        select: exp.Select,
        join: RawRowJoin | None = None,
    ) -> None:
        """Bind one ``generate_series`` table: its computed rows join the
        branch's relation exactly like a struct row table's written ones.

        `values` is the whole computed sequence -- resolve already did the
        arithmetic and rejected a zero step or an empty/descending range,
        since bounds and step are literals by the time it runs. No stream and
        no ``-i``: the rows are computed, not read.
        """
        if env.relation is None:
            env.relation = _RowRelation()
        rows = [_TrackRow(stream=_STREAMLESS_ROW, columns={local: v}) for v in values]
        env.bindings[local] = _RowBinding(
            alias=local,
            source="",
            column=local,
            type="data",  # filler: a computed row has no track
            relation=env.relation,
            values=RawValuesTable(
                alias=local, columns=(local,), rows=(), node=node, types=("number",)
            ),
        )
        self._join_rows(env.relation, local, rows, join, env, select)

    def _add_cte_rows(
        self,
        local: str,
        columns: tuple[_Column, ...],
        values: dict[str, tuple[RowValue, ...]],
        env: _Env,
        select: exp.Select,
        join: RawRowJoin | None = None,
    ) -> None:
        """Bind one CTE reference: its body's rows join the branch's relation.

        One body row is one outer row, so a comma between two CTEs (or between
        a CTE and an unnest table) is the ordinary cross join
        :meth:`_join_rows` already builds, multiplicity and all -- and an
        explicit `join` matches, keeps and gaps rows exactly as it does
        between unnest tables. A single-row body is a shape no-op, which is
        what keeps the one-input CTE shapes compiling exactly as they did.
        """
        if env.relation is None:
            env.relation = _RowRelation()
        rows = _cte_row_count(columns, values)
        env.bindings[local] = _CteBinding(
            name=local,
            columns=columns,
            rows=rows,
            relation=env.relation,
            values=values,
        )
        self._join_rows(
            env.relation,
            local,
            [_CteRow(position=position) for position in range(rows)],
            join,
            env,
            select,
        )

    def _known_hint(self) -> str:
        known = sorted(
            set(self.cte_columns)
            | set(self.graph.sources)
            | set(self.res.source_filters)
            | set(self.res.track_rows)
        )
        return f"known names: {', '.join(known)}" if known else "no aliases are in scope"

    # -- FROM ffmpeg.<source>(...) ------------------

    def _add_source(
        self,
        table: exp.Table,
        alias_node: exp.Expr | None,
        env: _Env,
        select: exp.Select,
    ) -> None:
        """Bind one generated-source alias, options validated, no node yet.

        Resolution and option validation happen HERE, when the FROM clause
        binds, rather than at first column access: a source's options are
        checked against the installed ffmpeg exactly like a tier-2 call's
        named arguments, and that check is a property of the query, not of
        how many times a column of it is read. The NODE is what is deferred
        (:meth:`_source_stream_of`) — an alias no projection ever mentions
        contributes no filter, which is the one respect in which a source
        alias differs from an ``input()`` one (that always gets its ``-i``).
        """
        if not isinstance(alias_node, exp.TableAlias) or alias_node.this is None:
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"{FILTER_NAMESPACE}.<source>() requires an alias",
                table,
                fallback=select,
                hint=f"add an alias, e.g. FROM {FILTER_NAMESPACE}.testsrc"
                "(duration => 2) t",
            )
        alias = _fold(alias_node.this)
        raw = self.res.source_filters.get(alias)
        if raw is None:  # defensive: resolve records every source alias
            raise _error(
                ErrorCode.UNKNOWN_ALIAS,
                f"unknown alias '{alias}'",
                alias_node,
                fallback=table,
                hint=self._known_hint(),
            )
        source = self._source_filter(raw, select)
        named = [_NamedArg(name=option.name, value=option.value) for option in raw.options]
        options = (
            self._filter_options(raw.name, raw.call_node, select) if named else {}
        )
        # No `timeline=`: SourceFilter has no such field, because a generator
        # is never timeline-capable -- there is no upstream frame to switch
        # on/off. `enable => ...` on a source rejects unconditionally.
        dropped: dict[str, exp.Expr] = {}
        args = self._check_named_args(
            raw.name,
            options,
            named,
            raw.call_node,
            owner=f"{FILTER_NAMESPACE}.{raw.name}",
            occupied=set(),
            dropped=dropped,
        )
        self._check_required_options(raw.name, args, dropped, raw.call_node, select)
        env.bindings[alias] = _SourceBinding(
            alias=alias, name=raw.name, output=source.output, options=args
        )

    def _source_filter(self, raw: RawSource, select: exp.Select) -> SourceFilter:
        """The registry's entry for ``ffmpeg.<name>`` in FROM position, or a rejection.

        Three ways this fails, in the order they are told apart:

        * the name is a REGULAR filter of this ffmpeg (``ffmpeg.gblur``) — it
          has input pads, so it is a call, not a table: UNSUPPORTED_SQL saying
          so, the one excluded case that is positively identifiable;
        * there is no registry at all (no ffmpeg) — the standard
          unavailability wording, same as a namespaced CALL's;
        * the name is unknown to both tables — UNKNOWN_FUNCTION with a
          did-you-mean over ``source_names()``. Sources the v1 scope check
          excluded (``avsynctest``'s ``|->AV``, ``movie``/``amovie``'s
          ``|->N``) are NOT retained by the registry at all, so they are
          indistinguishable from a typo here and land on the same rejection —
          which is why its fallback hint states the exclusion explicitly rather
          than only listing near-misses.
        """
        registry = self.registry
        source = registry.get_source(raw.name) if registry is not None else None
        if source is not None:
            return source
        if registry is not None and registry.get(raw.name) is not None:
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"{FILTER_NAMESPACE}.{raw.name} is an ffmpeg filter, not a source: "
                "it takes stream inputs, so it cannot stand in FROM",
                raw.call_node,
                fallback=select,
                hint=f"call it over a stream instead, e.g. SELECT "
                f"{FILTER_NAMESPACE}.{raw.name}(a.video[1]) FROM input('clip.mp4') a",
            )
        raise _error(
            ErrorCode.UNKNOWN_FUNCTION,
            f"unknown generated source {FILTER_NAMESPACE}.{raw.name}()",
            raw.call_node,
            fallback=select,
            hint=self._unknown_source_hint(raw.name),
        )

    def _unknown_source_hint(self, name: str) -> str:
        """Did-you-mean over ``source_names()``, then why the set might be missing.

        Mirrors :meth:`_namespaced_function_hint` branch for branch — the
        namespace is the same one, and a source is unavailable for exactly the
        same reasons a namespaced call is — but suggests only SOURCES, since
        a regular filter would not be usable in FROM either way.
        """
        registry = self.registry
        if registry is not None and registry.available():
            matches = difflib.get_close_matches(
                name, sorted(registry.source_names()), n=1, cutoff=0.6
            )
            if matches:
                return f"did you mean {FILTER_NAMESPACE}.{matches[0]}()?"
            return (
                f"FROM {FILTER_NAMESPACE}.<source>(...) takes a zero-input filter of "
                "your installed ffmpeg, and this is not one of them; sources with "
                "more than one output pad (avsynctest) or a variable pad count "
                "(movie, amovie) are not usable"
            )
        return (
            f"FROM {FILTER_NAMESPACE}.<source>(...) generates a stream with your "
            "installed ffmpeg; the provisioner failed to supply one"
        )

    def _source_stream_of(self, binding: _SourceBinding) -> _Stream:
        """The source's one stream, minting its node on first use only.

        The node is ``Node(filter=<source>, args=<validated options>,
        inputs=[], outputs=[<type>])`` — a chain head with no input labels
        (emit renders it as ``testsrc=duration=2[out0]``). Provenance is
        always empty: nothing was probed, because nothing was read.
        """
        if binding.ref is None:
            binding.ref = self.ctx.node(
                binding.name, dict(binding.options), [], [binding.output]
            )
        return _Stream(ref=binding.ref, type=binding.output, source=None)

    # -- WHERE ------------------------------------------------------------

    # -- WHERE, split into its three halves ------------

    def _split_where(
        self, select: exp.Select, env: _Env
    ) -> tuple[list[exp.Expr], list[exp.Expr], list[exp.Expr]]:
        """This branch's WHERE conjuncts, as ``(time windows, row predicates,
        subscript metadata assertions)``.

        A conjunct is a ROW predicate exactly when it mentions a track-row
        alias, or a CTE's value column -- both are columns of the rows this
        branch joins, and an alias is unambiguous: one name cannot be two
        things. A subscript metadata accessor (``Dot`` over
        ``Bracket``) is told apart by SHAPE instead, since its alias
        is an ordinary input one -- checked first, so a conjunct never falls
        through to the row/time split. Resolve rejected every mixed case but
        one -- a time window whose bounds are row columns, which is a window
        per row and lands in the time half.
        """
        where = select.args.get("where")
        if not isinstance(where, exp.Where):
            return [], [], []
        time_conjuncts: list[exp.Expr] = []
        row_conjuncts: list[exp.Expr] = []
        assertion_conjuncts: list[exp.Expr] = []
        for conjunct in _flatten_and(where.this):
            if any(
                isinstance(sub, exp.Dot) and subscript_metadata_shape(sub) is not None
                for sub in conjunct.walk()
            ):
                assertion_conjuncts.append(conjunct)
                continue
            aliases = {
                _fold(sub.args["table"])
                for sub in conjunct.walk()
                if isinstance(sub, exp.Column) and sub.args.get("table") is not None
            }
            rows = {
                alias
                for alias in aliases
                if isinstance(env.bindings.get(alias), _RowBinding)
                or _reads_cte_value(alias, conjunct, env)
            }
            if not rows:
                time_conjuncts.append(conjunct)
                continue
            if aliases - rows and len(rows) == 1 and self._is_row_window(conjunct, env):
                # A time window whose BOUNDS are row columns: one seek per row.
                # A fan-out TO gives each row a file; without one the rows stay
                # in this graph and each seeks its own `-i` of the same file.
                self._check_row_window_seeks_a_file(conjunct, where, env)
                time_conjuncts.append(conjunct)
                continue
            if aliases - rows or len(rows) > 1:  # defensive: resolve rejected both
                raise _error(
                    ErrorCode.UNSUPPORTED_SQL,
                    "a WHERE predicate may reference only one track-row table",
                    conjunct,
                    fallback=where,
                    hint="filter each unnest separately",
                )
            row_conjuncts.append(conjunct)
        return time_conjuncts, row_conjuncts, assertion_conjuncts

    def _check_row_window_seeks_a_file(
        self, conjunct: exp.Expr, where: exp.Where, env: _Env
    ) -> None:
        """A row-bounded window with no fan-out ``TO`` needs an ``-i`` per row,
        so the alias it windows has to own one.

        A CTE name is a filtergraph pad, not a file: its window is a
        ``trim``/``atrim`` pair on one stream, and there is nothing to mint one
        of per row.
        """
        if self.fanout_expr is not None:
            return
        parsed = _time_bounds(conjunct)
        table_node = parsed[0].args.get("table") if parsed is not None else None
        alias = _fold(table_node) if table_node is not None else ""
        if not isinstance(env.bindings.get(alias), _CteBinding):
            return
        raise _error(
            ErrorCode.UNSUPPORTED_SQL,
            f"'{alias}' is a filtergraph stream, so a trim bound reading a row "
            "column has no input to seek per row",
            conjunct,
            fallback=where,
            hint=f"window the input() alias '{alias}' was built from, or write "
            "TO ('clip' || i.i::text || '.mp4') for one command per row",
        )

    def _reads_row_alias(self, node: exp.Expr, env: _Env) -> bool:
        """True when `node` reads a column of any row table of this branch."""
        for sub in node.walk():
            if not isinstance(sub, exp.Column):
                continue
            table_node = sub.args.get("table")
            if table_node is not None and isinstance(
                env.bindings.get(_fold(table_node)), _RowBinding
            ):
                return True
        return False

    def _is_row_window(self, conjunct: exp.Expr, env: _Env) -> bool:
        """True for a time window on a non-row alias bounded by row columns."""
        parsed = _time_bounds(conjunct)
        if parsed is None:
            return False
        table_node = parsed[0].args.get("table")
        if table_node is None or _fold(parsed[0].this) != TIME_COLUMN:
            return False
        return not isinstance(env.bindings.get(_fold(table_node)), _RowBinding)

    # -- compile-time row filtering / ordering -------------------

    def _filter_rows(
        self, conjuncts: list[exp.Expr], env: _Env, select: exp.Select
    ) -> None:
        """Keep the rows whose predicate is TRUE; drop UNKNOWN and FALSE alike.

        Standard SQL: WHERE admits TRUE only, so a row whose metadata field was
        never probed simply does not match — no new rule, and no silent guess.
        The surviving set is written back onto the branch's relation, so every
        later ``t`` sees it and an unselected row's stream is never
        touched. Filtering happens AFTER the joins, which is where
        SQL puts it: dropping a row of an outer join's nullable side before the
        join would silently turn it into an inner one.
        """
        for conjunct in conjuncts:
            relation = self._predicate_relation(conjunct, env, select)
            relation.tuples = [
                row
                for row in relation.tuples
                if self._eval_row(conjunct, env, row, select) is True
            ]

    def _pin_fanout_row(self, env: _Env, select: exp.Select) -> None:
        """Cut the branch's relation down to the ONE group this command writes.

        Ungrouped, a group is a single row and this is the per-row pin it has
        always been. Under a GROUP BY over row columns the relation partitions
        into one group per distinct key, and the pinned group keeps ALL its
        tuples: everything downstream then works unchanged, since ``t``
        over the surviving tuples is exactly the array ``array_agg`` asked for,
        and the trim bounds and the path expression read `fanout_row` -- the
        group's first tuple, which stands for the whole group because the key
        is what every tuple in it agrees on.

        `fanout_count` is recorded so :func:`lower_commands` knows how many
        more runs to make.
        """
        if self.fanout_expr is None or env.relation is None:
            return
        groups = self._fanout_groups(env, select)
        if not groups:
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                "a TO expression writes one file per row, and no row survives "
                "the WHERE clause",
                self.fanout_expr,
                fallback=select,
                hint="loosen the filter, or write a quoted TO path",
            )
        if not 0 <= self.fanout_index < len(groups):
            raise _error(
                ErrorCode.INTERNAL,
                f"fan-out index {self.fanout_index} is outside the "
                f"{len(groups)} files this query writes",
                fallback=select,
                hint="please report this query as a bug",
            )
        group = groups[self.fanout_index]
        self.fanout_count = len(groups)
        self.fanout_grouped = bool(env.group_keys)
        self.fanout_row = group[0]
        self.fanout_env = env
        env.relation.tuples = list(group)

    def _fanout_groups(
        self, env: _Env, select: exp.Select
    ) -> list[list[_RowTuple]]:
        """The relation's tuples partitioned into the files they write.

        One group per distinct GROUP BY key, in FIRST-APPEARANCE order (the
        dict's own insertion order), so the command sequence follows the row
        order the query built. With no row-level key every tuple is its own
        group, which is the ungrouped fan-out unchanged.
        """
        relation = env.relation
        tuples = relation.tuples if relation is not None else []
        if not env.group_keys:
            return [[row] for row in tuples]
        groups: dict[tuple[RowValue, ...], list[_RowTuple]] = {}
        for row in tuples:
            key = tuple(self._key_value(node, env, row, select) for node in env.group_keys)
            groups.setdefault(key, []).append(row)
        return list(groups.values())

    def _key_value(
        self, node: exp.Expr, env: _Env, row: _RowTuple, select: exp.Select
    ) -> RowValue:
        """One GROUP BY key, read out of one result tuple.

        A stream column -- a CTE's, or a row table's ``track`` -- has no
        metadata value to compare, so what identifies the group is the stream
        itself: its ref, which two tuples share exactly when they carry the
        same stream.
        """
        stream = self._key_stream(node, env, row)
        if stream is not None:
            return stream.ref
        return self._eval_value(node, env, row, select)

    def _key_stream(self, node: exp.Expr, env: _Env, row: _RowTuple) -> _Stream | None:
        """The stream a GROUP BY key names in this tuple, else None."""
        column_node = _unwrap(node)
        if not isinstance(column_node, exp.Column):
            return None
        table_node = column_node.args.get("table")
        if table_node is None:
            return None
        binding = env.bindings.get(_fold(table_node))
        name = _fold(column_node.this)
        if isinstance(binding, _RowBinding):
            if name != ROW_STREAM:
                return None
            track = _track_of(row, binding.alias)
            return track.stream if track is not None else None
        if not isinstance(binding, _CteBinding):
            return None
        column = self._cte_column(binding, name)
        if column is None or not column.value.streams:
            return None
        entry = row.get(binding.name)
        if (
            isinstance(entry, _CteRow)
            and column.splat
            and len(column.value.streams) == binding.rows
        ):
            return column.value.streams[entry.position]
        # A broadcast column is one unit: every tuple reads the same stream.
        return column.value.streams[0]

    def _predicate_relation(
        self, conjunct: exp.Expr, env: _Env, select: exp.Select
    ) -> _RowRelation:
        """The relation one WHERE predicate filters.

        A predicate over a CTE's value column filters the branch's own
        relation -- the CTE's rows are already joined into it.
        """
        for sub in conjunct.walk():
            if not isinstance(sub, exp.Column):
                continue
            table_node = sub.args.get("table")
            if table_node is None:
                continue
            binding = env.bindings.get(_fold(table_node))
            if isinstance(binding, _CteBinding) and binding.relation is not None:
                return binding.relation
        return self._row_binding_of(conjunct, env, select).relation

    def _row_binding_of(
        self, node: exp.Expr, env: _Env, select: exp.Select
    ) -> _RowBinding:
        """The single row table `node`'s columns belong to (checked upstream)."""
        for sub in node.walk():
            if not isinstance(sub, exp.Column):
                continue
            table_node = sub.args.get("table")
            if table_node is None:
                continue
            binding = env.bindings.get(_fold(table_node))
            if isinstance(binding, _RowBinding):
                return binding
        raise _error(  # defensive: the caller only passes row expressions
            ErrorCode.UNSUPPORTED_SQL,
            "unsupported track-row expression",
            node,
            fallback=select,
        )

    def _eval_row(
        self,
        node: exp.Expr,
        env: _Env,
        rows: _RowTuple,
        select: exp.Select,
    ) -> bool | None:
        """One predicate against one result row: TRUE, FALSE, UNKNOWN (``None``).

        `rows` maps every row alias in scope to that result row's track, or to
        None where an outer join left a gap — one evaluator for WHERE (which
        sees a single alias) and for a JOIN's ON (which sees both sides), plan
        062 generalizing 061's single binding.

        Kleene three-valued logic, which is what makes the NULL story a
        non-story: a comparison with a NULL operand is UNKNOWN, UNKNOWN
        propagates through AND/OR/NOT the SQL way, and both callers keep TRUE
        only. A gap row reads NULL in every column, so "NULL matches nothing"
        covers the gaps too, for free.
        """
        node = _unwrap(node)
        if isinstance(node, exp.And | exp.Or):
            left = self._eval_row(node.this, env, rows, select)
            expression = node.args.get("expression")
            if not isinstance(expression, exp.Expr):
                raise _error(
                    ErrorCode.UNSUPPORTED_SQL, "malformed row predicate", node,
                    fallback=select,
                )
            right = self._eval_row(expression, env, rows, select)
            return (
                _kleene_and(left, right)
                if isinstance(node, exp.And)
                else _kleene_or(left, right)
            )
        if isinstance(node, exp.Not) and isinstance(node.this, exp.Expr):
            inner = self._eval_row(node.this, env, rows, select)
            return None if inner is None else not inner
        if isinstance(node, exp.Is):
            value = self._row_value_of(node.this, env, rows, select)
            is_null = value is None
            return not is_null if node.args.get("negate") else is_null
        if isinstance(node, exp.Between):
            value = self._eval_value(node.this, env, rows, select)
            low = self._eval_value(node.args.get("low"), env, rows, select)
            high = self._eval_value(node.args.get("high"), env, rows, select)
            return _kleene_and(
                _compare(exp.GTE(), value, low), _compare(exp.LTE(), value, high)
            )
        if isinstance(node, exp.EQ | exp.NEQ | exp.GT | exp.GTE | exp.LT | exp.LTE):
            # Both sides go through one value evaluator, so the operands stay in
            # written order and `'eng' = t.tags.language` needs no mirroring.
            return _compare(
                node,
                self._eval_value(node.this, env, rows, select),
                self._eval_value(node.args.get("expression"), env, rows, select),
            )
        if isinstance(node, exp.Boolean | exp.Column):
            # A boolean value IS the condition, as it is in Postgres; resolve
            # already turned away a column of any other type.
            value = self._eval_value(node, env, rows, select)
            return None if value is None else bool(value)
        raise _error(  # defensive: resolve accepted only the shapes above
            ErrorCode.UNSUPPORTED_SQL,
            "unsupported row predicate",
            node,
            fallback=select,
        )

    def _cte_value_of(
        self,
        binding: _CteBinding,
        column: exp.Column,
        rows: _RowTuple,
        select: exp.Select,
    ) -> RowValue:
        """One ``<cte>.<value column>`` reference, read out of this result row.

        The tuple holds the body row this result row came from, so the value
        is the one THAT row computed.
        """
        name = _fold(column.this)
        values = binding.values.get(name)
        if values is None:
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"unknown column '{binding.name}.{column.name}'",
                column,
                fallback=select,
                hint=self._cte_columns_hint(binding),
            )
        if binding.name in rows and rows[binding.name] is None:
            return None  # an outer join's gap reads NULL in every column
        entry = rows.get(binding.name)
        position = entry.position if isinstance(entry, _CteRow) else 0
        return values[position] if position < len(values) else None

    def _row_value_of(
        self,
        node: exp.Expr | None,
        env: _Env,
        rows: _RowTuple,
        select: exp.Select,
    ) -> RowValue:
        """One ``<row alias>.<column>`` reference, read out of this result row.

        A gap (the alias maps to None, because an outer join found no
        counterpart) reads NULL in every column — the one thing an absent row
        can honestly say about itself.

        ``<input alias>.duration`` and the container tags come from no row at
        all: they are probed off the input itself.
        """
        column = _unwrap(node) if isinstance(node, exp.Expr) else None
        if not isinstance(column, exp.Column):
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                "a track-row predicate compares a row column against a literal "
                "or another row column",
                column,
                fallback=select,
            )
        table_node = column.args.get("table")
        binding = env.bindings.get(_fold(table_node)) if table_node is not None else None
        if isinstance(binding, _InputBinding):
            name = _fold(column.this)
            if name == INPUT_DURATION_COLUMN:
                return self._input_duration(binding.alias, column, select)
            key = tag_key(name)
            if key is not None:
                return self._input_tag(binding.alias, key, column, select)
            if name in _RENDITION_SCHEMA:
                # Resolve admitted this name on spec (`RENDITION_COLUMNS`),
                # since only a probe can say whether `alias` is a ladder --
                # this alias's probe found none, so `_bind_renditions` left
                # it a plain `_InputBinding` rather than a rendition row
                # table, and this is that file's own rejection.
                raise _error(
                    ErrorCode.UNSUPPORTED_SQL,
                    f"'{binding.alias}' is a single file, not a ladder: "
                    f"input('{self._path_of(binding.alias)}') has no renditions",
                    column,
                    fallback=select,
                    hint="rendition columns (bandwidth, width, height, "
                    "codecs, name, language) read from an HLS master or "
                    "DASH manifest",
                )
        if isinstance(binding, _CteBinding):
            return self._cte_value_of(binding, column, rows, select)
        if not isinstance(binding, _RowBinding):  # defensive: resolve checked it
            raise _error(
                ErrorCode.UNKNOWN_ALIAS,
                f"unknown track-row alias '{_fold(table_node)}'",
                column,
                fallback=select,
                hint=self._known_hint(),
            )
        name = _fold(column.this)
        if name in MAP_COLUMNS:
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"'{binding.alias}.{name}' is the whole {map_noun(name)} map, "
                "not a single value",
                column,
                fallback=select,
                hint=f"name the key: '{binding.alias}.{name}.{map_example(name)}'",
            )
        if name not in binding.schema and map_ref(name) is None:
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"unknown column '{binding.alias}.{column.name}'",
                column,
                fallback=select,
                hint=binding.exposes,
            )
        row = _track_of(rows, binding.alias)
        return None if row is None else row.columns.get(name)

    def _eval_value(
        self,
        node: exp.Expr | None,
        env: _Env,
        rows: _RowTuple,
        select: exp.Select,
    ) -> RowValue:
        """One compile-time value over a result row.

        The whole value grammar: a literal, NULL, a row's metadata column, an
        input's probed ``duration``, ``CASE``, ``||``, arithmetic and
        ``::text``. Shared by the predicate evaluator (a comparison's operands,
        a BETWEEN bound), by tag columns, by trim bounds and by computed call
        arguments, so every one of them speaks the same language.
        """
        value = _unwrap(node) if isinstance(node, exp.Expr) else None
        if isinstance(value, exp.Null):
            return None
        if isinstance(value, exp.Boolean):
            return bool(value.this)
        if isinstance(value, exp.Column):
            return self._row_value_of(value, env, rows, select)
        if isinstance(value, exp.Case):
            return self._eval_case(value, env, rows, select)
        if isinstance(value, exp.Bracket) and isinstance(value.this, exp.Array):
            return self._eval_list_element(value, env, rows, select)
        if isinstance(value, exp.Coalesce) and is_value_expr(value):
            # A value COALESCE (first argument a value, never a stream): the
            # first non-NULL argument, or NULL when every one is absent.
            for argument in [value.this, *value.args.get("expressions", [])]:
                result = self._eval_value(argument, env, rows, select)
                if result is not None:
                    return result
            return None
        if isinstance(value, exp.DPipe):
            return self._eval_concat(value, env, rows, select)
        if isinstance(value, _ARITHMETIC):
            return self._eval_arithmetic(value, env, rows, select)
        if isinstance(value, exp.Cast):
            return self._eval_cast(value, env, rows, select)
        if isinstance(value, _BUILTIN_VALUE_FUNCS):
            return self._eval_builtin_call(value, env, rows, select)
        if isinstance(value, exp.Neg) and not isinstance(_unwrap(value.this), exp.Literal):
            operand = self._eval_number(value.this, "'-'", value, env, rows, select)
            return None if operand is None else -operand
        if isinstance(value, exp.Expr):
            call = _call_parts(value)
            if call is not None and not call.namespaced and not call.is_macro:
                declared = self.res.wasm.get(call.name.lower())
                if declared is not None and declared.is_value:
                    return self._eval_wasm_value(declared, call, value, env, rows, select)
        return self._literal_of(value, select)

    def _eval_arithmetic(
        self,
        node: exp.Expr,
        env: _Env,
        rows: _RowTuple,
        select: exp.Select,
    ) -> RowValue:
        """``+ - * /`` with Postgres' own typing, at compile time.

        int op int stays an int and ``/`` TRUNCATES toward zero, any float
        operand makes the result a float, and NULL on either side propagates.
        Dividing by a zero is a typed rejection: the value is knowable here, so
        shipping an ffmpeg command built on it is not an option.
        """
        operator = _ARITHMETIC_NAMES[type(node)]
        left = self._eval_number(node.this, operator, node, env, rows, select)
        right = self._eval_number(node.args.get("expression"), operator, node, env, rows, select)
        if left is None or right is None:
            return None
        if isinstance(node, exp.Add):
            return left + right
        if isinstance(node, exp.Sub):
            return left - right
        if isinstance(node, exp.Mul):
            return left * right
        if right == 0:
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                "division by zero",
                node,
                fallback=select,
                hint="the divisor is known at compile time, and it is zero",
            )
        if isinstance(left, int) and isinstance(right, int):
            quotient = abs(left) // abs(right)
            return -quotient if (left < 0) != (right < 0) else quotient
        return left / right

    def _eval_number(
        self,
        node: exp.Expr | None,
        operator: str,
        anchor: exp.Expr,
        env: _Env,
        rows: _RowTuple,
        select: exp.Select,
    ) -> int | float | None:
        """One arithmetic operand's value; text is a typed rejection."""
        value = self._eval_value(node, env, rows, select)
        if value is None or isinstance(value, int | float):
            return value
        raise _error(
            ErrorCode.UNSUPPORTED_SQL,
            f"{operator} needs numbers, but one side is text",
            node if isinstance(node, exp.Expr) else anchor,
            fallback=select,
        )

    def _eval_cast(
        self,
        node: exp.Cast,
        env: _Env,
        rows: _RowTuple,
        select: exp.Select,
    ) -> RowValue:
        """``x::text``: the number spelled out, NULL left NULL.

        One spelling rule, shared with the filtergraph and the seek times --
        an int prints without a point, a float in python's shortest form that
        reads back as the same float.
        """
        value = self._eval_value(node.this, env, rows, select)
        return None if value is None else _tag_text(value)

    def _eval_builtin_call(
        self,
        node: exp.Expr,
        env: _Env,
        rows: _RowTuple,
        select: exp.Select,
    ) -> RowValue:
        """``upper``/``lower``/``length``/``round``/``replace``/``substring``,
        over a literal or a row column alike -- the same value grammar every
        other operator here uses, so a row column reads exactly as a literal
        would. NULL propagates from any argument, as it does through ``||``
        and arithmetic; :meth:`ffrwd.parser._Resolver._check_builtin_call`
        already typed every argument, so this only evaluates.
        """
        name = node.__class__.__name__.lower()
        if isinstance(node, exp.Upper | exp.Lower):
            text = self._eval_text(node.this, name, env, rows, select)
            if text is None:
                return None
            return text.upper() if isinstance(node, exp.Upper) else text.lower()
        if isinstance(node, exp.Length):
            text = self._eval_text(node.this, name, env, rows, select)
            return None if text is None else len(text)
        if isinstance(node, exp.Round):
            number = self._eval_number(node.this, f"{name}()", node, env, rows, select)
            if number is None:
                return None
            decimals_node = node.args.get("decimals")
            places = 0
            if decimals_node is not None:
                decimals = self._eval_number(
                    decimals_node, f"{name}()", node, env, rows, select
                )
                if decimals is None:
                    return None
                places = int(decimals)
            rounded = round(number, places)
            return int(rounded) if places <= 0 else rounded
        if isinstance(node, exp.Replace):
            text = self._eval_text(node.this, name, env, rows, select)
            target = self._eval_text(node.args.get("expression"), name, env, rows, select)
            replacement_node = node.args.get("replacement")
            replacement = (
                self._eval_text(replacement_node, name, env, rows, select)
                if replacement_node is not None
                else ""
            )
            if text is None or target is None or replacement is None:
                return None
            return text.replace(target, replacement)
        # exp.Substring: the string, then a 1-based start and an optional length.
        text = self._eval_text(node.this, name, env, rows, select)
        if text is None:
            return None
        start_node = node.args.get("start")
        start = 1
        if start_node is not None:
            value = self._eval_number(start_node, f"{name}()", node, env, rows, select)
            if value is None:
                return None
            start = int(value)
        length_node = node.args.get("length")
        if length_node is None:
            return text[max(start - 1, 0) :]
        value = self._eval_number(length_node, f"{name}()", node, env, rows, select)
        if value is None:
            return None
        end = start - 1 + int(value)
        return text[max(start - 1, 0) : max(end, 0)]

    def _eval_text(
        self,
        node: exp.Expr | None,
        name: str,
        env: _Env,
        rows: _RowTuple,
        select: exp.Select,
    ) -> str | None:
        """One text-function operand's value; a number or boolean is a typed rejection."""
        value = self._eval_value(node, env, rows, select)
        if value is None or isinstance(value, str):
            return value
        raise _error(
            ErrorCode.UNSUPPORTED_SQL,
            f"{name}() needs text, but the argument is "
            + ("boolean" if isinstance(value, bool) else "number"),
            node if isinstance(node, exp.Expr) else select,
            fallback=select,
        )

    def _eval_case(
        self,
        node: exp.Case,
        env: _Env,
        rows: _RowTuple,
        select: exp.Select,
    ) -> RowValue:
        """CASE, searched and simple: the first TRUE branch, else ELSE, else NULL.

        A searched branch's condition is an ordinary row predicate, so its
        three-valued logic carries straight over: only TRUE takes a branch, and
        UNKNOWN falls through exactly as FALSE does. The simple form compares
        the operand with ``=``, which makes a NULL operand match no WHEN — SQL's
        rule, and the same 3VL again.
        """
        operand_node = node.this if isinstance(node.this, exp.Expr) else None
        operand = (
            self._eval_value(operand_node, env, rows, select)
            if operand_node is not None
            else None
        )
        for branch in node.args.get("ifs") or []:
            if not isinstance(branch, exp.If) or not isinstance(branch.this, exp.Expr):
                raise _error(  # defensive: resolve checked the shape
                    ErrorCode.UNSUPPORTED_SQL, "malformed CASE", node, fallback=select
                )
            matched = (
                self._eval_row(branch.this, env, rows, select)
                if operand_node is None
                else _compare(
                    exp.EQ(),
                    operand,
                    self._eval_value(branch.this, env, rows, select),
                )
            )
            if matched is True:
                return self._eval_value(branch.args.get("true"), env, rows, select)
        default = node.args.get("default")
        if not isinstance(default, exp.Expr):
            return None
        return self._eval_value(default, env, rows, select)

    def _eval_list_element(
        self,
        node: exp.Bracket,
        env: _Env,
        rows: _RowTuple,
        select: exp.Select,
    ) -> RowValue:
        """``ARRAY[<literals>][<subscript>]``: one element, picked per row.

        What a subscripted list variable substitutes to when its subscript is
        a row column, and equally writable by hand. The subscript is 1-based;
        NULL propagates as everywhere in the value grammar; a subscript past
        either end is a typed rejection naming the list's length, because a
        row that quietly picks nothing would ship the wrong command.
        """
        array = node.this
        if not isinstance(array, exp.Array) or len(node.expressions) != 1:
            raise _error(  # defensive: resolve checked the shape
                ErrorCode.UNSUPPORTED_SQL, "malformed array element", node, fallback=select
            )
        elements = array.expressions
        index = subscript_index(node)
        if index is None:
            picked = self._eval_value(node.expressions[0], env, rows, select)
            if picked is None:
                return None
            if isinstance(picked, bool) or not isinstance(picked, int):
                raise _error(
                    ErrorCode.UNSUPPORTED_SQL,
                    f"an array subscript is a whole number, got {_tag_text(picked)}",
                    node.expressions[0],
                    fallback=select,
                    hint="subscripts are 1-based integers; a row column like "
                    "a generate_series value fits as it is",
                )
            index = picked
        if index < 1:
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"subscript {index} is before the first element",
                node,
                fallback=select,
                hint="list subscripts are 1-based: [1] is the first element",
            )
        if index > len(elements):
            have = f"{len(elements)} element" + ("" if len(elements) == 1 else "s")
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"subscript {index} is past the end: the list has {have}",
                node,
                fallback=select,
                hint=f"subscript from 1 to {len(elements)}",
            )
        element = elements[index - 1]
        if isinstance(_unwrap(element), exp.Null):
            return None  # a NULL element is absence, like a NULL subscript
        return self._literal_of(element, select)

    def _eval_concat(
        self,
        node: exp.DPipe,
        env: _Env,
        rows: _RowTuple,
        select: exp.Select,
    ) -> RowValue:
        """``a || b``: NULL when either side is NULL, else the two texts joined."""
        left = self._eval_value(node.this, env, rows, select)
        right = self._eval_value(node.args.get("expression"), env, rows, select)
        if left is None or right is None:
            return None
        return f"{left}{right}"

    def _literal_of(self, node: exp.Expr | None, select: exp.Select) -> RowValue:
        """A row predicate's literal operand as a python scalar."""
        value = _unwrap(node) if isinstance(node, exp.Expr) else None
        if isinstance(value, exp.Neg) and isinstance(value.this, exp.Expr):
            return -_number(_unwrap(value.this), ErrorCode.UNSUPPORTED_SQL)
        if isinstance(value, exp.Literal):
            if value.is_string:
                return str(value.this)
            return _number(value, ErrorCode.UNSUPPORTED_SQL)
        raise _error(
            ErrorCode.UNSUPPORTED_SQL,
            "a track-row predicate compares a row column against a literal",
            value,
            fallback=select,
        )

    # -- subscript metadata WHERE assertions --
    #
    # `<alias>.<type>[k].<column>` names ONE probed track deterministically
    # (the subscript is bounds-checked, not filtered), so a WHERE conjunct over
    # it has nothing to DROP the way a row predicate drops rows. It is an
    # ASSERTION, checked once at compile time against the probed file: TRUE
    # proceeds unchanged, FALSE or UNKNOWN (3VL -- a field that was never
    # probed) is a typed rejection, because an ffmpeg command line cannot
    # encode "select nothing" (recipe 29 of docs/examples.md).
    #
    # The boolean algebra is the row evaluator's, reused wholesale; the only
    # new piece is where a leaf's VALUE comes from (`_accessor_value`, probed
    # off the input through the same `_row_columns` a track-row table uses).

    def _check_assertions(self, conjuncts: list[exp.Expr], select: exp.Select) -> None:
        for conjunct in conjuncts:
            if self._eval_assertion(conjunct, select) is not True:
                raise _error(
                    ErrorCode.UNSUPPORTED_SQL,
                    "WHERE assertion failed at compile time: "
                    f"{conjunct.sql(dialect='postgres')}",
                    conjunct,
                    fallback=select,
                    hint="a subscript metadata predicate is checked once, "
                    "against the probed file, and a false or unprobed ('NULL') "
                    "result refuses to compile rather than silently shipping "
                    "the wrong track; fix the query or the input",
                )

    def _eval_assertion(self, node: exp.Expr, select: exp.Select) -> bool | None:
        """One subscript metadata predicate, Kleene three-valued, like `_eval_row`."""
        node = _unwrap(node)
        if isinstance(node, exp.And | exp.Or):
            left = self._eval_assertion(node.this, select)
            expression = node.args.get("expression")
            if not isinstance(expression, exp.Expr):
                raise _error(
                    ErrorCode.UNSUPPORTED_SQL, "malformed WHERE predicate", node,
                    fallback=select,
                )
            right = self._eval_assertion(expression, select)
            return (
                _kleene_and(left, right)
                if isinstance(node, exp.And)
                else _kleene_or(left, right)
            )
        if isinstance(node, exp.Not) and isinstance(node.this, exp.Expr):
            inner = self._eval_assertion(node.this, select)
            return None if inner is None else not inner
        if isinstance(node, exp.Is):
            value = self._accessor_value(node.this, select)
            is_null = value is None
            return not is_null if node.args.get("negate") else is_null
        if isinstance(node, exp.Between):
            value = self._accessor_value(node.this, select)
            low = self._literal_of(node.args.get("low"), select)
            high = self._literal_of(node.args.get("high"), select)
            return _kleene_and(
                _compare(exp.GTE(), value, low), _compare(exp.LTE(), value, high)
            )
        if isinstance(node, exp.EQ | exp.NEQ | exp.GT | exp.GTE | exp.LT | exp.LTE):
            left_node = node.this
            right_node = node.args.get("expression")
            left_shape = (
                subscript_metadata_shape(_unwrap(left_node))
                if isinstance(left_node, exp.Expr)
                else None
            )
            if left_shape is not None:
                return _compare(
                    node,
                    self._accessor_value(left_node, select),
                    self._literal_of(right_node, select),
                )
            mirrored = _MIRRORED_COMPARISONS[type(node)]()
            return _compare(
                mirrored,
                self._accessor_value(right_node, select),
                self._literal_of(left_node, select),
            )
        raise _error(
            ErrorCode.UNSUPPORTED_SQL, "unsupported WHERE predicate", node,
            fallback=select,
        )

    def _input_duration(
        self, alias: str, anchor: exp.Expr, select: exp.Select
    ) -> int | float:
        """``<input>.duration``: the probed container length, in seconds.

        Probed-only, and a rejection when it is not there — an unreadable file
        has no length, and neither does a container that declares none, so
        there is nothing to guess an expression's value from.
        """
        result = self.probes.get(alias)
        duration = None if result is None else result.duration
        if duration is None:
            raise _error(
                ErrorCode.INPUT_NOT_FOUND,
                f"'{alias}.{INPUT_DURATION_COLUMN}' is unknown: "
                f"'{self._path_of(alias)}' reports no container duration",
                anchor,
                fallback=select,
                hint="the duration is probed from the file; only a readable "
                "input that declares one has it",
            )
        return duration

    def _input_tag(
        self, alias: str, key: str, anchor: exp.Expr, select: exp.Select
    ) -> str | None:
        """``<input>.<tag>``: one probed container tag, NULL when absent.

        An absent key is NULL — that is what lets a CASE fill it — but an input
        this compile could not probe is a rejection, the same rule
        ``duration`` follows: a file nobody read says nothing about its tags.
        """
        result = self.probes.get(alias)
        if result is None:
            raise _error(
                ErrorCode.INPUT_NOT_FOUND,
                f"'{alias}.{TAGS_COLUMN}.{key}' is unknown: "
                f"'{self._path_of(alias)}' could not be probed",
                anchor,
                fallback=select,
                hint="container tags are read from the file; only a readable "
                "input has them",
            )
        return result.tags.get(key)

    def _accessor_value(self, node: exp.Expr | None, select: exp.Select) -> RowValue:
        """The probed value one ``<alias>.<type>[k].<column>`` accessor names.

        Resolve already confined this shape to an ordinary INPUT alias (never
        a row or CTE one), so this reads the SAME probed ``StreamMeta`` a bare
        ``<alias>.<type>[k]`` would select, through the SAME `_row_columns` a
        track-row table's columns come from -- one metadata table,
        two ways to name a row of it.
        """
        shape = (
            subscript_metadata_shape(_unwrap(node)) if isinstance(node, exp.Expr) else None
        )
        if shape is None:
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                "a subscript metadata predicate compares an accessor against "
                "a literal",
                node if isinstance(node, exp.Expr) else None,
                fallback=select,
            )
        bracket, name = shape
        inner = bracket.this
        if not isinstance(inner, exp.Column):  # defensive: resolve checked the shape
            raise _error(
                ErrorCode.UNSUPPORTED_SQL, "malformed subscript metadata accessor",
                bracket, fallback=select,
            )
        table_node = inner.args.get("table")
        alias = _fold(table_node) if table_node is not None else ""
        array_column = _fold(inner.this)
        stream_type = _ARRAY_COLUMNS.get(array_column)
        if stream_type is None:  # defensive: resolve checked the array column
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"'{alias}.{array_column}' has no per-track metadata",
                inner,
                fallback=select,
            )
        index = subscript_index(bracket)
        if index is None:  # defensive: resolve checked the subscript
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                "stream subscript must be a positive integer literal",
                bracket,
                fallback=select,
                hint=_SUBSCRIPT_HINT,
            )
        result = self.probes.get(alias)
        if result is None:
            path = self._path_of(alias)
            raise self._unreadable_error(
                ErrorCode.INPUT_NOT_FOUND,
                alias,
                f"cannot check '{alias}.{array_column}[{index}].{name}' of '{path}'",
                bracket,
                select,
                hint="subscript metadata is probed from the file, and only a "
                "readable input has any; the WHERE assertion cannot be checked",
            )
        streams = result.by_type(stream_type)
        if not 1 <= index <= len(streams):
            have = f"{len(streams)} {stream_type} stream" + ("" if len(streams) == 1 else "s")
            raise _error(
                ErrorCode.STREAM_NOT_FOUND,
                f"'{alias}.{array_column}[{index}]' does not exist: "
                f"'{self._path_of(alias)}' has {have}",
                bracket,
                fallback=select,
                hint=_SUBSCRIPT_HINT,
            )
        meta = streams[index - 1]
        columns = _row_columns(meta, array_column)
        return columns.get(name)

    def _order_rows(self, select: exp.Select, env: _Env) -> None:
        """Re-sort a row table explicitly -- the ORDER BY carve-out.

        Row order is deterministic WITHOUT this — it is the file's track order,
        which is player-visible surface nothing resorts implicitly — so an
        ORDER BY is the user saying otherwise, and it applies at compile time
        to the row list, never to frames.

        Multi-key sorting is done one key at a time from LAST to FIRST over
        python's stable sort, which is exactly SQL's key precedence. NULLs are
        partitioned out rather than sorted, because they have no order: their
        position is ``nulls_first``, which sqlglot fills in from the Postgres
        defaults (ASC -> NULLS LAST, DESC -> NULLS FIRST) whether or not the
        query spelled it.
        """
        order = select.args.get("order")
        if not isinstance(order, exp.Order):
            return
        if env.relation is None:
            # The parser admitted ORDER BY on the strength of an `input(...)`
            # alias that MIGHT have been a ladder; the probe just settled it
            # wasn't, so this branch has no row table after all.
            raise _error(
                ErrorCode.NO_STREAMING_EQUIVALENT,
                "ORDER BY has no streaming equivalent",
                order,
                fallback=select,
                hint="remove the ORDER BY clause",
            )
        for ordered in reversed(order.expressions):
            if not isinstance(ordered, exp.Ordered):
                raise _error(
                    ErrorCode.UNSUPPORTED_SQL, "malformed ORDER BY", fallback=order
                )
            binding = self._row_binding_of(ordered, env, select)
            key = _unwrap(ordered.this)
            if not isinstance(key, exp.Column):  # defensive: resolve checked it
                raise _error(
                    ErrorCode.NO_STREAMING_EQUIVALENT,
                    "ORDER BY has no streaming equivalent",
                    ordered,
                    fallback=order,
                )
            name = _fold(key.this)
            relation = binding.relation

            def value_of(
                row: _RowTuple,
                alias: str = binding.alias,
                name: str = name,
            ) -> RowValue:
                track = _track_of(row, alias)
                return None if track is None else track.columns.get(name)

            nulls = [row for row in relation.tuples if value_of(row) is None]
            rest = [row for row in relation.tuples if value_of(row) is not None]
            rest.sort(
                key=lambda row: _sort_key(value_of(row)),
                reverse=bool(ordered.args.get("desc")),
            )
            relation.tuples = (
                nulls + rest if ordered.args.get("nulls_first") else rest + nulls
            )

    def _limit_rows(self, select: exp.Select, env: _Env) -> None:
        """Narrow the resolved row set: OFFSET skips rows, LIMIT caps them.

        Applies to the branch's shared relation after WHERE and ORDER BY and
        before grouping, the fan-out pin, and the one-row rule -- so ``ORDER
        BY t.width DESC LIMIT 1`` IS the top row, no aggregate needed. Both
        counts are integer literals (resolve checked, LIMIT 0 included); the
        one judgment only this pass can make is an OFFSET that skips every
        row, since only the resolved relation knows its own size -- the same
        selects-nothing mistake LIMIT 0 names at resolve.
        """
        limit = select.args.get("limit")
        offset = select.args.get("offset")
        if env.relation is None:
            # Same story as `_order_rows`: the parser could not yet tell a
            # renditionless input from a ladder, so this rejection waited for
            # the probe instead of firing at parse time.
            if isinstance(limit, exp.Limit):
                raise _error(
                    ErrorCode.NO_STREAMING_EQUIVALENT,
                    "LIMIT has no streaming equivalent",
                    limit,
                    fallback=select,
                    hint=_RENDITIONLESS_ROW_CLAUSE_HINT,
                )
            if isinstance(offset, exp.Offset):
                raise _error(
                    ErrorCode.NO_STREAMING_EQUIVALENT,
                    "OFFSET has no streaming equivalent",
                    offset,
                    fallback=select,
                    hint=_RENDITIONLESS_ROW_CLAUSE_HINT,
                )
        take = (
            self._row_bound(limit.args.get("expression"), "LIMIT", select)
            if isinstance(limit, exp.Limit)
            else None
        )
        skip = (
            self._row_bound(offset.args.get("expression"), "OFFSET", select)
            if isinstance(offset, exp.Offset)
            else None
        )
        if take is None and skip is None:
            return
        relation = env.relation
        count = len(relation.tuples) if relation is not None else 1
        if skip is not None and skip >= count:
            have = f"{count} row" + ("" if count == 1 else "s")
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"OFFSET {skip} skips every row: this query has {have}",
                offset,
                fallback=select,
                hint="a query that selects nothing is a mistake worth "
                "naming; skip fewer rows, or drop the clause",
            )
        if relation is None:
            return
        end = None if take is None else (skip or 0) + take
        relation.tuples = relation.tuples[skip or 0 : end]

    def _row_bound(
        self, node: exp.Expr | None, clause: str, select: exp.Select
    ) -> int:
        """One LIMIT/OFFSET count as a python int (defensive: resolve checked)."""
        value = _unwrap(node) if isinstance(node, exp.Expr) else None
        if isinstance(value, exp.Literal) and not value.is_string:
            text = str(value.this)
            if text.isascii() and text.isdigit():
                return int(text)
        raise _error(
            ErrorCode.UNSUPPORTED_SQL,
            f"{clause} must be an integer literal",
            node if isinstance(node, exp.Expr) else None,
            fallback=select,
        )

    def _collect_trims(
        self, select: exp.Select, env: _Env, conjuncts: list[exp.Expr]
    ) -> None:
        """Record each aliased time range, on the input or on the branch.

        The binding decides where the window goes. An INPUT alias owns its own
        ``-i`` and is globally unique, so at most one window can ever apply to
        it: it is recorded on the GRAPH
        (``Graph.input_trims``) and becomes ``-ss``/``-to``, seeking every
        stream of that input coherently — captions and unselected streams
        included. A CTE name is a filtergraph pad, so its window is recorded on
        the BRANCH (``_Env.trims``) and the ``trim``/``atrim`` pair is spliced
        lazily by :meth:`_access`, the first time a stream of that CTE is
        consumed.

        A conjunct may supply only a lower bound (``<alias>.t >= x``) or only
        an upper one (``<alias>.t <= y``),
        via :func:`ffrwd.parser._time_bounds`, which also normalizes the
        mirrored operand order (``x <= <alias>.t`` etc.) and flags a strict
        ``>``/``<`` so it is rejected here too. Two conjuncts for the same
        alias MERGE into one window (``t >= 1 AND t <= 2`` behaves exactly
        like ``t BETWEEN 1 AND 2``) — resolve already rejected a second bound
        of the same kind, so this only ever fills in the other half. Every
        check below duplicates one resolve already made (defensive re-check,
        as elsewhere in this pass).

        `conjuncts` is the TIME half of the WHERE clause
        (:meth:`_split_where`), not the whole of it: row predicates share the
        clause and are decided on rows, not on the timeline.
        """
        where = select.args.get("where")
        if not isinstance(where, exp.Where) or not conjuncts:
            return
        # alias -> its (lower, upper) bound EXPRESSIONS; the numbers come
        # after, once it is known how many rows each one is evaluated against.
        bounds: dict[str, tuple[exp.Expr | None, exp.Expr | None]] = {}
        for conjunct in conjuncts:
            parsed = _time_bounds(conjunct)
            if parsed is None:
                raise _error(
                    ErrorCode.UNSUPPORTED_SQL,
                    "unsupported WHERE predicate",
                    conjunct,
                    fallback=where,
                    hint=_TIME_HINT,
                )
            column, low, high, strict = parsed
            if strict:
                raise _error(
                    ErrorCode.UNSUPPORTED_SQL,
                    "strict inequalities are not supported",
                    conjunct,
                    fallback=where,
                    hint=_TIME_HINT,
                )
            table_node = column.args.get("table")
            if table_node is None:
                raise _error(
                    ErrorCode.UNSUPPORTED_SQL,
                    f"unqualified column '{column.name}' in WHERE",
                    column,
                    fallback=where,
                    hint=_TIME_HINT,
                )
            alias = _fold(table_node)
            if _fold(column.this) != TIME_COLUMN:
                raise _error(
                    ErrorCode.UNSUPPORTED_SQL,
                    f"only the time column '{alias}.t' can be filtered, "
                    f"got '{alias}.{column.name}'",
                    column,
                    fallback=where,
                    hint=_TIME_HINT,
                )
            if alias not in env.bindings:
                raise _error(
                    ErrorCode.UNKNOWN_ALIAS,
                    f"unknown alias '{alias}'",
                    table_node,
                    fallback=where,
                    hint=self._known_hint(),
                )
            binding = env.bindings[alias]
            if isinstance(binding, _SourceBinding):
                # A generated source has no input file to seek and no
                # timeline to trim: it is a filter that MAKES a stream, and
                # how long a stream it makes is one of its own options.
                raise _error(
                    ErrorCode.UNSUPPORTED_SQL,
                    f"'{alias}' is a generated source, so 'WHERE {alias}.t' has "
                    "nothing to seek",
                    conjunct,
                    fallback=where,
                    hint=_SOURCE_DURATION_HINT,
                )
            if isinstance(binding, _InputBinding) and alias in self.graph.module_sources:
                # A module source is a pull loop the sidecar paces itself,
                # not a file with an offset: there is nothing for -ss/-to to
                # seek, unlike a probed input's own -i.
                raise _error(
                    ErrorCode.UNSUPPORTED_SQL,
                    f"'{alias}' is a module source, so 'WHERE {alias}.t' has "
                    "nothing to seek",
                    conjunct,
                    fallback=where,
                    hint=_MODULE_SOURCE_SEEK_HINT,
                )
            low_node, high_node = bounds.get(alias, (None, None))
            bounds[alias] = (
                low if low is not None else low_node,
                high if high is not None else high_node,
            )

        for alias, (low_node, high_node) in bounds.items():
            per_row = self._is_per_row_window(alias, low_node, high_node, env)
            rows = (
                env.relation.tuples
                if per_row and env.relation is not None
                else [self.fanout_row]
            )
            windows = [
                self._window_of(alias, low_node, high_node, env, row, select)
                for row in rows
            ]
            if isinstance(env.bindings[alias], _InputBinding):
                if any(
                    opt.name == "seek_end"
                    for opt in self.res.input_options.get(alias, ())
                ):
                    raise _error(
                        ErrorCode.UNSUPPORTED_SQL,
                        f"'{alias}' sets seek_end and is also seeked by "
                        f"'WHERE {alias}.t' -- one input, two seek origins",
                        fallback=select,
                        hint=f"drop seek_end from {alias}'s input(), or drop "
                        f"the WHERE window on '{alias}'",
                    )
                if per_row:
                    # Every row seeks its own copy of the file, all in this one
                    # graph, so the alias reads one stream per row from here on.
                    env.row_inputs[alias] = [
                        self._row_input(alias, window) for window in windows
                    ]
                    self.row_window_seen = True
                elif self.fanout_sinks and self.fanout_expr is not None:
                    # A fan-out row's window belongs to the FILE that row
                    # writes, not to the `-i` every one of them reads.
                    self.fanout_windows[alias] = windows[0]
                else:
                    self.graph.input_trims[alias] = windows[0]
            else:
                env.trims[alias] = windows[0]

    def _is_per_row_window(
        self,
        alias: str,
        low: exp.Expr | None,
        high: exp.Expr | None,
        env: _Env,
    ) -> bool:
        """True when this window is one seek PER ROW inside a single graph.

        A bound that reads a row column names a different number for every
        row. Under a fan-out ``TO`` that is one command per row and the pinned
        row answers for all of them; without one the rows share this graph, so
        each needs an ``-i`` of its own -- which only an input alias has.
        """
        if self.fanout_expr is not None or env.relation is None:
            return False
        if not isinstance(env.bindings.get(alias), _InputBinding):
            return False
        return any(
            node is not None and self._reads_row_alias(node, env)
            for node in (low, high)
        )

    def _window_of(
        self,
        alias: str,
        low: exp.Expr | None,
        high: exp.Expr | None,
        env: _Env,
        rows: _RowTuple,
        select: exp.Select,
    ) -> tuple[int | float | None, int | float | None]:
        """One alias's window as `rows` reads it, start strictly before end."""
        start = self._time_bound(low, env, select, rows) if low is not None else None
        end = self._time_bound(high, env, select, rows) if high is not None else None
        if start is not None and end is not None and start >= end:
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"empty time window for alias '{alias}': start ({start}) "
                f"is not before end ({end})",
                fallback=select,
                hint="the start bound must be strictly before the end bound",
            )
        return start, end

    def _row_input(
        self, alias: str, window: tuple[int | float | None, int | float | None]
    ) -> str:
        """The ``-i`` one row's window seeks: `alias` itself for the first
        window, a copy of it for each further one.

        One input per DISTINCT window, so two rows naming the same one share a
        slot (and the split pass shares its decode). The copy's alias carries a
        ``#``, which no unquoted identifier may, because nothing resolves it --
        it exists so the graph's alias-keyed input tables can hold the slot.
        """
        recorded = self.graph.input_trims.get(alias)
        if recorded is None or recorded == window:
            self.graph.input_trims[alias] = window
            return alias
        for minted, origin in self.row_input_source.items():
            if origin == alias and self.graph.input_trims.get(minted) == window:
                return minted
        index = len(self.graph.input_paths)
        minted = f"{alias}#{index + 1}"
        self.graph.input_paths.append(self.graph.input_paths[self.graph.sources[alias]])
        self.graph.sources[minted] = index
        self.graph.input_trims[minted] = window
        self.row_input_source[minted] = alias
        return minted

    def _time_bound(
        self,
        bound: exp.Expr,
        env: _Env,
        select: exp.Select,
        rows: _RowTuple | None = None,
    ) -> int | float:
        """One trim bound in seconds: a literal, or the value grammar's answer.

        A computed bound is still a SEEK, so it must come out a number. The
        one way it could come out NULL — an input whose duration was never
        probed — is already a rejection naming that field
        (:meth:`_input_duration`), so the raise below is the defensive floor.

        `rows` is the result row the bound reads its row columns off, which is
        what makes ``WHERE f.t BETWEEN c.start_t AND c.end_t`` a per-row seek:
        the pinned row under a fan-out ``TO``, each surviving row without one.
        """
        value = self._eval_value(
            bound, env, self.fanout_row if rows is None else rows, select
        )
        if isinstance(value, int | float):
            return value
        raise _error(
            ErrorCode.UNSUPPORTED_SQL,
            f"time bound '{bound.sql(dialect='postgres')}' is "
            + ("NULL" if value is None else "text"),
            bound,
            fallback=select,
            hint="a trim bound is a number of seconds",
        )

    def _access(
        self,
        env: _Env,
        alias: str,
        value: _Value,
        anchor: exp.Expr,
        select: exp.Select,
    ) -> _Value:
        """Apply `alias`'s FILTER trim to every stream of `value`.

        ``_Env.trims`` is CTE-only (see :meth:`_collect_trims`), so this is a
        no-op for every input alias: an input's window is already on its ``-i``
        as ``-ss``/``-to``, and the stream refs pass through untouched — which
        is what lets a trimmed column stay a passthrough and be stream-copied.

        For a CTE window the trim is spliced elementwise over an array and
        memoized per stream, so each element of a broadcast array gets exactly
        one trim, shared by all its consumers.

        This is also where a trimmed caption is rejected: the WHERE
        window is collected before any projection lowers, so "is this CTE's
        subtitle/data actually CONSUMED under a trim" is only knowable here, at
        the point the trim would be applied. A CTE's trim is a filtergraph
        ``trim``/``atrim`` pair, which cannot carry subtitle or data streams at
        all, so for a CTE the rejection is permanent; on an input
        alias it does not arise, because there is no filter node to feed.
        """
        window = env.trims.get(alias)
        if window is None:
            trimmed = alias in self.graph.input_trims or alias in self.fanout_windows
            if value.type in _PASSTHROUGH_ONLY and trimmed:
                # MEASURED 2026-08-15, not theoretical: ffmpeg does not retime
                # subtitle/data packets under an input -ss (copy OR transcode;
                # cue times stay near-original while video rebases to zero), so
                # a seeked caption track plays out of sync by the seek amount.
                # Reject rather than ship silent desync.
                raise _error(
                    ErrorCode.UNSUPPORTED_SQL,
                    f"'WHERE {alias}.t' cannot trim a selected {value.type} stream: "
                    "ffmpeg does not retime caption packets under an input seek, so "
                    "they would play out of sync with the trimmed video",
                    anchor,
                    fallback=select,
                    hint=_CAPTION_TRIM_HINT,
                )
            return value
        if value.type in _PASSTHROUGH_ONLY:
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"a CTE's captions cannot be trimmed: 'WHERE {alias}.t' would have "
                f"to trim a {value.type} stream, which no filtergraph can carry",
                anchor,
                fallback=select,
                hint=_CAPTION_TRIM_HINT,
            )
        return _Value(
            type=value.type,
            streams=tuple(self._trim(env, window, stream) for stream in value.streams),
            is_array=value.is_array,
        )

    def _trim(
        self,
        env: _Env,
        window: tuple[int | float | None, int | float | None],
        stream: _Stream,
    ) -> _Stream:
        """The trimmed counterpart of one stream; a trim is spliced once per stream.

        `window` may have either half absent (open-ended), so the
        ``trim``/``atrim`` node gets only the args it has: ``start=X``,
        ``end=Y``, or both.
        """
        cached = env.trimmed.get(stream.ref)
        if cached is not None:
            return _Stream(ref=cached, type=stream.type, source=stream.source)
        start, end = window
        args: dict[str, object] = {}
        if start is not None:
            args["start"] = start
        if end is not None:
            args["end"] = end
        if stream.type == "video":
            trimmed = self.ctx.node("trim", args, [stream.ref], ["video"])
            rebased = self.ctx.node(
                "setpts", {"expr": "PTS-STARTPTS"}, [trimmed], ["video"]
            )
        else:
            trimmed = self.ctx.node("atrim", args, [stream.ref], ["audio"])
            rebased = self.ctx.node(
                "asetpts", {"expr": "PTS-STARTPTS"}, [trimmed], ["audio"]
            )
        env.trimmed[stream.ref] = rebased
        # A trim is 1:1, so it threads provenance through unchanged.
        return _Stream(ref=rebased, type=stream.type, source=stream.source)

    # -- expressions ------------------------------------------------------

    def _lower_expr(self, node: exp.Expr, env: _Env, select: exp.Select) -> _Value:
        node = _unwrap(node)
        # An array of cue records IS a subtitle track, so it lowers here, in a
        # stream position, and not as an output column the way `chapters` does.
        cues = self._lower_cue_array(node, env, select)
        if cues is not None:
            return cues
        # A module's annotation column is a track too, minted from the rows
        # instead of from a written document.
        rows = self._lower_rows_projection(node, env, select)
        if rows is not None:
            return rows
        if isinstance(node, exp.Struct):
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                "a STRUCT is a record, not a stream",
                node,
                fallback=select,
                hint=f"name it to write metadata, e.g. STRUCT('Main' AS title) "
                f"AS {TAGS_COLUMN}, or cast it to a record type, e.g. "
                f"{_CHAPTER_EXAMPLE}",
            )
        if isinstance(node, exp.Bracket | exp.Column):
            alias, value = self._base_stream(node, env, select)
            return self._access(env, alias, value, node, select)
        if isinstance(node, exp.ArrayAgg):
            return self._lower_array_agg(node, env, select)
        if isinstance(node, exp.Cast):
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                "casts are not supported",
                node,
                fallback=select,
                hint="a stream has exactly one type",
            )
        if isinstance(node, exp.Array):
            # A chapter list is a column of the file rather than a stream; a
            # cue array is a stream, and was taken above.
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                "an array literal is not a stream expression",
                node,
                fallback=select,
                hint=_CHAPTERS_COLUMN_HINT,
            )
        if isinstance(node, exp.Coalesce):
            # Not a call: COALESCE resolves against the ROW model, not the
            # registry -- it is how a nullable track column is spelled.
            return self._lower_coalesce(node, env, select)
        if is_value_expr(node):
            # A value expression, never a stream. Reaching here means it is not
            # a tag column either: unaliased, or inside a CTE body.
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"every SELECT column must be a stream expression, got "
                f"{_describe(node)}",
                node,
                fallback=select,
                hint="a value expression names a metadata TAG: give it an alias "
                "for the tag key",
            )
        call = _call_parts(node)
        if call is not None:
            return self._lower_call(node, call, env, select)
        raise _error(
            ErrorCode.UNSUPPORTED_SQL,
            "every SELECT column must be a stream expression, got "
            f"{_describe(node)}",
            node,
            fallback=select,
            hint=_STREAM_HINT,
        )

    def _lower_array_agg(
        self, node: exp.ArrayAgg, env: _Env, select: exp.Select
    ) -> _Value:
        """``array_agg(<stream expression>)``: the explicit splat.

        The argument lowers over the branch's surviving tuples exactly as it
        would on its own -- ``t`` is already the N-element array of the
        rows in row order (:meth:`_row_value`), and a filter call over it
        already broadcasts elementwise -- so the aggregate is the identity on
        the value, and the sugar and the spelled-out form emit the same bytes
        by construction rather than by agreement.

        A rendition row's own kind columns (``r.video``, ``r.audio``) are the
        one exception: bare, they already read that way, one stream per
        surviving row that carries the kind (:meth:`_rendition_kind_value`),
        but subscripted -- ``r.video[1]`` -- the same column instead picks a
        SINGLE rendition out of the ones carrying the kind, the right
        reading for a bare, non-aggregated column. `array_agg` names every
        surviving row's own kind, not one row picked out of them, and a
        rendition never carries more than one stream of a kind, so ``[1]``
        is the only index that can ever name one -- :meth:`_rendition_array_agg`
        reads it straight off `kinds`, the way :meth:`_row_value` reads an
        unnest row's own stream.
        """
        inner = node.this
        if not isinstance(inner, exp.Expr):
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                "array_agg() takes one stream expression",
                node,
                fallback=select,
                hint=_ARRAY_AGG_HINT,
            )
        if env.relation is None:
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                "array_agg() aggregates track rows, and this query has none",
                node,
                fallback=select,
                hint=_ARRAY_AGG_HINT,
            )
        rendition = self._rendition_array_agg(inner, env, select)
        if rendition is not None:
            return rendition
        return self._lower_expr(inner, env, select)

    def _rendition_array_agg(
        self, inner: exp.Expr, env: _Env, select: exp.Select
    ) -> _Value | None:
        """``array_agg(<row>.video[1])`` / ``.audio[1]``: every surviving
        row's own stream of the kind, gathered in row order.

        None for anything else -- a plain container column, an unnest row, a
        bare (unsubscripted) rendition column, or a subscript other than
        ``[1]`` -- so :meth:`_lower_array_agg` falls back to lowering the
        argument exactly as it would outside the aggregate.
        """
        bracket = _unwrap(inner)
        if not isinstance(bracket, exp.Bracket):
            return None
        column = bracket.this
        if not isinstance(column, exp.Column):
            return None
        table_node = column.args.get("table")
        if table_node is None:
            return None
        binding = env.bindings.get(_fold(table_node))
        if not isinstance(binding, _RowBinding) or binding.column != RENDITION_COLUMN:
            return None
        name = _fold(column.this)
        if name not in _ARRAY_COLUMNS or subscript_index(bracket) != 1:
            return None
        return self._rendition_kind_value(binding, name, None, bracket, select)

    # -- a cue array as a subtitle track -----------------------------------

    def _lower_cue_array(
        self, node: exp.Expr, env: _Env, select: exp.Select
    ) -> _Value | None:
        """``ARRAY[STRUCT(...)::cue, ...]`` / ``array_agg(STRUCT(...)::cue)`` as a track.

        None when the expression is not one, so every other stream expression
        falls through untouched. The cues become one self-contained WebVTT
        ``data:`` input -- the mechanism ``ffrwd.empty_captions()`` already
        uses, with cues in the document -- and the value is that input's one
        subtitle stream, mapped and passed through like any other.
        """
        cues = self._cue_records(node, env, select)
        if cues is None:
            return None
        if not cues:
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                "a cue array is empty, so there is no subtitle track to write",
                node,
                fallback=select,
                hint=f"write at least one cue, e.g. ARRAY[{_CUE_EXAMPLE}], or "
                "drop the column",
            )
        text = _cues_webvtt(cues)
        uri = "data:text/vtt;base64," + base64.b64encode(text.encode()).decode()
        ref = self._mint_stream_input(CUES_COLUMN, uri, WEBVTT_FORMAT, "subtitle")
        return _scalar(_Stream(ref=ref, type="subtitle", source=None))

    def _cue_records(
        self, node: exp.Expr, env: _Env, select: exp.Select
    ) -> list[_Cue] | None:
        """The cues a cue array lists, in written order; None if it is not one.

        A literal array is read element by element and an ``array_agg`` once
        per surviving row, exactly as a chapter list is -- the two spellings
        of "a list of records" are the same two here.
        """
        if isinstance(node, exp.ArrayAgg):
            inner = node.this
            relation = env.relation
            if not isinstance(inner, exp.Expr) or record_cast_type(
                _unwrap(inner)
            ) != CUE_TYPE:
                return None
            if relation is None:
                raise _error(
                    ErrorCode.UNSUPPORTED_SQL,
                    "array_agg() aggregates rows, and this query has none",
                    node,
                    fallback=select,
                    hint=_CUE_ARRAY_HINT,
                )
            return [
                self._cue_record(inner, env, row, select) for row in relation.tuples
            ]
        if isinstance(node, exp.Array):
            elements = [item for item in node.expressions if isinstance(item, exp.Expr)]
            if not elements or record_cast_type(_unwrap(elements[0])) != CUE_TYPE:
                return None
            row = _group_row(env)
            return [self._cue_record(element, env, row, select) for element in elements]
        return None

    # -- a module's rows as a track ----------------------------------------

    def _rows_projection(
        self, node: exp.Expr
    ) -> tuple[exp.Anonymous, WasmFunction] | None:
        """``<module call>.<annotation column>``, as the call and what declares it.

        None for every other expression. Resolve has already refused a field
        read that is not the annotation column, so a projection reaching here
        names one.
        """
        if not isinstance(node, exp.Dot):
            return None
        field = node.args.get("expression")
        base = _unwrap(node.this) if isinstance(node.this, exp.Expr) else None
        if not isinstance(field, exp.Identifier) or not isinstance(base, exp.Anonymous):
            return None
        declared = self.res.wasm.get(str(base.name).lower())
        if declared is None or declared.emits is None:
            return None
        return (base, declared) if _fold(field) == declared.emits.name else None

    def _lower_rows_projection(
        self, node: exp.Expr, env: _Env, select: exp.Select
    ) -> _Value | None:
        """A module's annotation column, as the track its rows become.

        The call lowers exactly as it would on its own -- the module reads the
        stream and the rows come off its frames -- and the frames go no
        further: the rows are what was selected, so the module's own output
        feeds the rows document and nothing maps it. The value is a subtitle
        stream on a compiler-minted ``-i`` the sidecar writes, tagged with the
        language the call named.

        A COPY whose destination IS a rows file writes the rows themselves
        instead, and mints no track.
        """
        found = self._rows_projection(node)
        if found is None:
            return None
        call, declared = found
        module = self._row_filtered(self._lower_expr(call, env, select), node)
        producer = module.streams[0].ref
        if self.rows_file:
            self.graph.rows_sinks[producer] = RowsSink(
                container=_ROWS_CONTAINER, path=self.rows_file
            )
            return _Value(type=module.type, streams=(), is_array=False)
        tag = self._rows_language(declared, call, node, env, select)
        ref = self._mint_stream_input(
            CUES_COLUMN, PIPE, WEBVTT_FORMAT, "subtitle"
        )
        self.graph.rows_sinks[producer] = RowsSink(
            container=WEBVTT_FORMAT, alias=src_alias(ref)
        )
        self.rows_tracks.append(ref)
        return _scalar(_Stream(ref=ref, type="subtitle", source=_rows_meta(tag)))

    def _rows_language(
        self,
        declared: WasmFunction,
        call: exp.Anonymous,
        node: exp.Expr,
        env: _Env,
        select: exp.Select,
    ) -> str | None:
        """The container tag a minted track carries, or None for an untagged one.

        The module names which of its parameters say what language its rows
        are in, best first. The first of them the CALL gives a value -- written
        or filled from a DEFAULT -- is the one, and a module naming none leaves
        the track untagged. A value that is no language this compiler knows is
        a rejection naming the parameter and what it was given.
        """
        described = self.describes.get(declared.module)
        wanted = described.rows_language if described is not None else ()
        parts = _call_parts(call)
        if not wanted or parts is None:
            return None
        positions = {
            param.name: (index, param)
            for index, param in enumerate(
                declared.value_params, start=declared.stream_arity
            )
        }
        row = env.relation.tuples[0] if env.relation and env.relation.tuples else {}
        for name in wanted:
            found = positions.get(name)
            if found is None:
                continue
            index, param = found
            written = parts.args[index] if index < len(parts.args) else param.default
            if written is None:
                continue
            value = self._eval_value(written, env, row, select)
            if value is None:
                continue
            tag = language_tag(value) if isinstance(value, str) else None
            if tag is not None:
                return tag
            raise _error(
                ErrorCode.UDF_ARG_TYPE,
                f"{declared.name}() was given '{param.name}' as {value!r}, which "
                "is no language a container can be tagged with",
                written,
                fallback=node,
                hint="name the language as its two-letter code, e.g. 'es' or "
                "'en'; rows that are no language at all are 'zxx'",
            )
        return None

    # -- stream references -------------------------------------------------

    def _base_stream(
        self, node: exp.Expr, env: _Env, select: exp.Select
    ) -> tuple[str, _Value]:
        """Resolve a column / subscript to ``(alias, untrimmed value)``.

        The value is an ARRAY for a bare ``a.video`` / ``a.audio`` (or a bare
        reference to an array-typed CTE column) and a scalar for anything
        subscripted. Pure: creates no nodes, so the type checker
        (:meth:`_classify`) can call it on an argument before deciding whether
        to lower it — which is also why enumerating an unprobeable input fails
        here, before the graph has grown.
        """
        if isinstance(node, exp.Bracket):
            inner = node.this
            if not isinstance(inner, exp.Column):
                raise _error(
                    ErrorCode.UNSUPPORTED_SQL,
                    "only stream columns can be subscripted",
                    node,
                    fallback=select,
                    hint=_SUBSCRIPT_HINT,
                )
            index = subscript_index(node)
            if index is None:
                raise _error(
                    ErrorCode.UNSUPPORTED_SQL,
                    "stream subscript must be a positive integer literal",
                    node,
                    fallback=select,
                    hint=_SUBSCRIPT_HINT,
                )
            return self._resolve_column(inner, index, node, env, select)
        if isinstance(node, exp.Column):
            return self._resolve_column(node, None, node, env, select)
        raise _error(
            ErrorCode.UNSUPPORTED_SQL,
            f"expected a stream expression, got {_describe(node)}",
            node,
            fallback=select,
            hint=_STREAM_HINT,
        )

    def _resolve_column(
        self,
        column: exp.Column,
        index: int | None,
        anchor: exp.Expr,
        env: _Env,
        select: exp.Select,
    ) -> tuple[str, _Value]:
        table_node = column.args.get("table")
        if table_node is None:
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"unqualified column '{column.name}'",
                anchor,
                fallback=select,
                hint="qualify the column with its alias, e.g. a.video[1]",
            )
        alias = _fold(table_node)
        name = _fold(column.this)
        binding = env.bindings.get(alias)
        if binding is None:
            raise _error(
                ErrorCode.UNKNOWN_ALIAS,
                f"unknown alias '{alias}'",
                table_node,
                fallback=select,
                hint=self._known_hint(),
            )
        if isinstance(binding, _InputBinding):
            return alias, self._input_value(alias, name, index, env, anchor, select)
        if isinstance(binding, _SourceBinding):
            return alias, self._source_value(binding, name, index, anchor, select)
        if isinstance(binding, _RowBinding):
            if binding.column == RENDITION_COLUMN and name in _ARRAY_COLUMNS:
                # A rendition row's own track-kind columns: one stream per
                # SURVIVING row that carries the kind, not one per row -- see
                # `_rendition_kind_value`.
                return binding.source, self._rendition_kind_value(
                    binding, name, index, anchor, select
                )
            # Under the INPUT alias, not the row one: a row table has no window
            # of its own, and every rule about the streams (`-i`, `-ss`, the
            # caption-seek rejection) is a property of the file they came from.
            return binding.source, self._row_value(
                binding, name, index, env, anchor, select
            )
        return alias, self._cte_value(binding, name, index, anchor, select)

    def _rendition_kind_value(
        self,
        binding: _RowBinding,
        name: str,
        index: int | None,
        anchor: exp.Expr,
        select: exp.Select,
    ) -> _Value:
        """One track-kind column of a rendition row table (``r.video``,
        ``r.audio``, ...): one stream per surviving row that carries the
        kind, in row order.

        Unlike the row's own stream (``r``, one per surviving row
        unconditionally), an audio-only rendition contributes NOTHING to
        ``r.video`` -- the array is shorter than the row count whenever the
        ladder mixes muxed and audio-only rungs. Same array/subscript surface
        as every other stream column: bare ``r.video`` is the whole array,
        ``r.video[k]`` names one element of it.

        A manifest destination reads this differently (:meth:`_rendition_row_cells`):
        each ladder rung is its own variant map entry, so the column has to
        stay one cell PER ROW -- an audio-only rung's video cell is NULL,
        not absent, the same gap a FULL JOIN's unmatched row leaves.
        """
        kind = _ARRAY_COLUMNS[name]
        if self.manifest is not None or self.row_reading_sink:
            return self._rendition_row_cells(binding, kind, index)
        streams = [
            row.kinds[kind] for row in binding.rows if row is not None and kind in row.kinds
        ]
        if index is None:
            return _array(kind, streams)
        if not 1 <= index <= len(streams):
            have = f"{len(streams)} row" + ("" if len(streams) == 1 else "s")
            raise _error(
                ErrorCode.STREAM_NOT_FOUND,
                f"'{binding.alias}.{name}[{index}]' does not exist: "
                f"{have} carry a {kind} track",
                anchor,
                fallback=select,
                hint=_SUBSCRIPT_HINT,
            )
        return _scalar(streams[index - 1])

    def _rendition_row_cells(
        self, binding: _RowBinding, kind: StreamType, index: int | None
    ) -> _Value:
        """``<alias>.video``/``.audio`` at a manifest destination: one cell
        per ladder rung, in row order -- a bare column and its ``[1]``
        subscript read the same thing, since a rung carries at most one
        stream of a kind. A rung without that kind (an audio-only rendition's
        ``.video``) contributes the NULL sentinel a manifest's variant map
        already knows how to read as an absent cell, exactly like an
        unmatched FULL JOIN row.
        """
        return _array(
            kind,
            (
                row.kinds[kind]
                if row is not None and kind in row.kinds and index in (None, 1)
                else _Stream(ref=_NULL_STREAM_REF, type=kind, source=None)
                for row in binding.rows
            ),
        )

    def _row_value(
        self,
        binding: _RowBinding,
        name: str,
        index: int | None,
        env: _Env,
        anchor: exp.Expr,
        select: exp.Select,
    ) -> _Value:
        """One column of a track-row table — and only the row itself is a stream.

        ``t`` over N surviving rows is an N-element ARRAY in row order, exactly
        what a bare ``f.audio`` is, so
        every existing array rule (splat, broadcast, subscript, zip) applies to
        it unchanged and the downstream passes learn nothing new.

        A metadata column is not an output: streams are the only outputs there
        are, and ``SELECT t.tags.language`` names a string. That is a typed
        rejection rather than a stringly-typed output, and its hint says what
        metadata columns ARE for.
        """
        schema = binding.schema
        if name != ROW_STREAM:
            if name not in schema and map_ref(name) is None:
                raise _error(
                    ErrorCode.UNSUPPORTED_SQL,
                    f"unknown column '{binding.alias}.{column_label(name)}'",
                    anchor,
                    fallback=select,
                    hint=binding.exposes,
                )
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"'{binding.alias}.{column_label(name)}' is track metadata, not "
                "a stream, and a SELECT column is an output stream",
                anchor,
                fallback=select,
                hint=_WRITTEN_ROW_HINT
                if binding.values is not None
                else _record_row_hint(binding.record)
                if binding.streamless
                else _ROW_METADATA_HINT,
            )
        if binding.values is not None:
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"'{binding.alias}' is a written row, not a stream",
                anchor,
                fallback=select,
                hint=_WRITTEN_ROW_HINT,
            )
        if binding.column in RECORD_ARRAY_COLUMNS:
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"'{binding.alias}' is {article(binding.record)} "
                f"{binding.record} row, not a stream",
                anchor,
                fallback=select,
                hint=_record_row_hint(binding.record),
            )
        if not binding.rows:
            raise _error(
                ErrorCode.STREAM_NOT_FOUND,
                f"'{binding.alias}' selects nothing: no "
                f"{binding.column} track of '{self._path_of(binding.source)}' "
                "survived",
                anchor,
                fallback=select,
                hint="an empty row set would select no streams; widen the WHERE, "
                "or check that the file has the tracks you expect",
            )
        streams = self._per_row_seeks(
            binding,
            [
                self._row_stream(binding, row, position, anchor, select)
                for position, row in enumerate(binding.rows)
            ],
            env,
        )
        if index is None:
            return _array(binding.type, streams)
        if not 1 <= index <= len(streams):
            have = f"{len(streams)} row" + ("" if len(streams) == 1 else "s")
            raise _error(
                ErrorCode.STREAM_NOT_FOUND,
                f"'{binding.alias}[{index}]' does not exist: "
                f"'{binding.alias}' has {have}",
                anchor,
                fallback=select,
                hint=_SUBSCRIPT_HINT,
            )
        return _scalar(streams[index - 1])

    def _per_row_seeks(
        self, binding: _RowBinding, streams: list[_Stream], env: _Env
    ) -> list[_Stream]:
        """Re-point a row table's tracks at the ``-i`` each row seeks.

        A row-bounded window on the input the tracks came from gives every
        result row a copy of the file with its own ``-ss``/``-to``, and a row's
        track belongs to the copy that row seeks: the same stream of the same
        file, read through a different input slot.
        """
        row_inputs = env.row_inputs.get(binding.source)
        if row_inputs is None:
            return streams
        reseeked: list[_Stream] = []
        for position, stream in enumerate(streams):
            if position >= len(row_inputs) or not is_src(stream.ref):
                reseeked.append(stream)
                continue
            _, stream_type, index = src_parts(stream.ref)
            marker = _TYPE_MARKERS[stream_type]
            reseeked.append(
                replace(stream, ref=f"src:{row_inputs[position]}:{marker}:{index}")
            )
        return reseeked

    def _row_stream(
        self,
        binding: _RowBinding,
        row: _TrackRow | None,
        position: int,
        anchor: exp.Expr,
        select: exp.Select,
    ) -> _Stream:
        """One result row's track — and the NULL rejection when there isn't one.

        Selecting a nullable track column (outer join) without COALESCE is a
        typed rejection naming the row that was NULL, never a silently missing
        output. An outer join is the user saying the
        counterpart may be absent, so what to put there instead is a decision
        only they can make; ``COALESCE(<column>, <fill>)`` is where they make
        it, and the hint says so with the fill this column's type takes.

        In table mode there is no ffmpeg command to be
        missing an input for — the NULL row is exactly what an outer join's
        gap IS, and it prints as an empty cell, psql-style, same as any other
        NULL. ``_NULL_STREAM`` is the sentinel :meth:`_value_to_cells` reads
        back into that empty cell; its empty ref can never collide with a
        real one (every real ref is non-empty).
        """
        if row is not None:
            self._reject_codecless(
                row.stream.source,
                f"'{binding.alias}' (row {position + 1})",
                anchor,
                select,
            )
            return row.stream
        if self.table_mode or self.manifest is not None or self.row_reading_sink:
            return _Stream(ref=_NULL_STREAM_REF, type=binding.type, source=None)
        fill = _FILL_SPELLINGS.get(binding.type)
        hint = (
            f"an outer join leaves gaps; fill them with "
            f"COALESCE({binding.alias}, {fill})"
            if fill is not None
            # data rows have no fill spelling at all: nothing can stand in
            # for a missing data track, so the join itself must not leave
            # the gap.
            else "data tracks have no fill; use an INNER or LEFT join so "
            "every selected row has one"
        )
        raise _error(
            ErrorCode.STREAM_NOT_FOUND,
            f"'{binding.alias}' is NULL in row {position + 1}: "
            f"{self._unmatched_text(binding, position)}",
            anchor,
            fallback=select,
            hint=hint,
        )

    def _unmatched_text(self, binding: _RowBinding, position: int) -> str:
        """What the missing row failed to match, named from its paired row."""
        relation = binding.relation
        row = relation.tuples[position]
        paired_alias, paired = self._paired_row(relation, row, binding.alias)
        keys = relation.keys.get(paired_alias or "", [])
        if paired is None or not keys:
            return f"the join found no {binding.column} row of '{binding.alias}'"
        described = ", ".join(
            f"{paired_alias}.{column_label(key)}={paired.columns.get(key)!r}"
            for key in keys
        )
        return f"no '{binding.alias}' row matched {described}"

    def _paired_row(
        self,
        relation: _RowRelation,
        row: _RowTuple,
        alias: str,
    ) -> tuple[str | None, _TrackRow | None]:
        """The counterpart of a gap: the first row table that DID match here.

        A fill's provenance is the paired (non-NULL counterpart) row's metadata,
        and its inherited options come from that same row, so a silence-filled
        French mix stays French.
        Relation order (FROM order) breaks the tie when three tables joined.
        """
        for other in relation.aliases:
            if other == alias:
                continue
            track = _track_of(row, other)
            if track is not None:
                return other, track
        return None, None

    # -- COALESCE(<row>, <fill>) -----------------

    def _lower_coalesce(self, node: exp.Expr, env: _Env, select: exp.Select) -> _Value:
        """The accepted spelling for a nullable track column: fill its gaps.

        The result is the same N-element array ``<alias>`` is, in the
        same row order — every gap replaced by a generated stand-in. Only the
        gaps mint anything: a join with no unmatched rows compiles to exactly
        the command the bare column would -- consume-once here means "generate
        nothing nobody needed".
        """
        binding, fill = self._coalesce_parts(node, env, select)
        relation = binding.relation
        if not relation.tuples:
            raise _error(
                ErrorCode.STREAM_NOT_FOUND,
                f"'{binding.alias}' selects nothing: no "
                f"{binding.column} track of '{self._path_of(binding.source)}' "
                "survived",
                node,
                fallback=select,
                hint="an empty row set would select no streams; widen the WHERE, "
                "or check that the file has the tracks you expect",
            )
        streams: list[_Stream] = []
        for row in relation.tuples:
            track = _track_of(row, binding.alias)
            if track is not None:
                # The real track goes through `_access` exactly as a bare
                # a bare `<alias>` would, so the input's WHERE window (and the
                # caption-seek rejection) still applies to it.
                streams.append(
                    self._access(
                        env, binding.source, _scalar(track.stream), node, select
                    ).streams[0]
                )
                continue
            _, paired = self._paired_row(relation, row, binding.alias)
            streams.append(self._lower_fill(fill, binding, paired, node, select))
        return _array(binding.type, streams)

    def _coalesce_parts(
        self, node: exp.Expr, env: _Env, select: exp.Select
    ) -> tuple[_RowBinding, exp.Expr]:
        """``(the track column's row table, the fill expression)``, or a rejection.

        Deliberately narrow: COALESCE exists in this dialect for exactly one
        job — standing something in for an outer join's missing track — so it
        takes a track-row stream column and one fill, and nothing else. It
        creates no nodes, which is what lets :meth:`_classify` call it on an
        argument before deciding whether to lower it.
        """
        arguments = [
            argument
            for argument in [node.this, *node.expressions]
            if isinstance(argument, exp.Expr)
        ]
        if len(arguments) != 2:
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"COALESCE takes a track column and one fill, got "
                f"{len(arguments)} argument{'' if len(arguments) == 1 else 's'}",
                node,
                fallback=select,
                hint=_COALESCE_HINT,
            )
        column = _unwrap(arguments[0])
        binding: _Binding | None = None
        if isinstance(column, exp.Column):
            table_node = column.args.get("table")
            if table_node is not None:
                binding = env.bindings.get(_fold(table_node))
        if not isinstance(binding, _RowBinding) or _fold(column.this) != ROW_STREAM:
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                "COALESCE's first argument is a track-row stream column, got "
                f"{_describe(arguments[0])}",
                arguments[0],
                fallback=select,
                hint=_COALESCE_HINT,
            )
        return binding, arguments[1]

    def _lower_fill(
        self,
        node: exp.Expr,
        binding: _RowBinding,
        paired: _TrackRow | None,
        anchor: exp.Expr,
        select: exp.Select,
    ) -> _Stream:
        """Mint the stand-in for one missing track, per the per-type table.

        Two mechanisms, one rule. ``ffmpeg.<source>()`` is a zero-input filter
        node (``anullsrc`` for audio, ``color`` for video), option-checked
        against the installed ffmpeg exactly like a source in FROM;
        ``ffrwd.empty_captions()`` is an INPUT, because a filtergraph carries
        no subtitle pads to generate one on. Either way the fill inherits from
        the PAIRED row — the counterpart that did match — both its options
        (:meth:`_inherited_fill_options`) and its provenance, so a
        silence-filled French mix is still tagged French.
        """
        call = _call_parts(node) if isinstance(node, exp.Expr) else None
        if call is None or call.args or not (call.namespaced or call.is_macro):
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"a COALESCE fill is a generated stand-in, got {_describe(node)}",
                node,
                fallback=select,
                hint=self._fill_hint(binding),
            )
        source_meta = paired.stream.source if paired is not None else None
        name = call.name.lower()
        if call.is_macro:
            return self._lower_macro_fill(node, name, call, binding, source_meta, select)
        source = self._source_filter(
            RawSource(alias="", name=name, options=(), call_node=node), select
        )
        self._check_fill_type(source.output, call.display, binding, node, select)
        options = self._filter_options(name, node, select)
        dropped: dict[str, exp.Expr] = {}
        args = self._check_named_args(
            name,
            options,
            call.named,
            node,
            owner=f"{FILTER_NAMESPACE}.{name}",
            occupied=set(),
            dropped=dropped,
        )
        self._check_required_options(name, args, dropped, node, select)
        for option, value in self._inherited_fill_options(binding.type, paired).items():
            if value is None or option in args or option not in options:
                continue
            args[option] = value
        if "duration" in options and "duration" not in args:
            # A generator with no duration runs forever, and "forever" is not
            # what a missing 2-second track means. Inheriting it is the only
            # correct default, so when the paired row was never
            # probed for one, the query has to say it.
            raise _error(
                ErrorCode.UDF_ARG_TYPE,
                f"{call.display}() has no duration to stand in for: the paired "
                f"track's duration was never probed",
                node,
                fallback=select,
                hint=f"give the fill one, e.g. {call.display}(duration => 2)",
            )
        return _Stream(
            ref=self.ctx.node(name, args, [], [source.output]),
            type=source.output,
            source=source_meta,
        )

    def _lower_macro_fill(
        self,
        node: exp.Expr,
        name: str,
        call: _Call,
        binding: _RowBinding,
        source_meta: StreamMeta | None,
        select: exp.Select,
    ) -> _Stream:
        """``ffrwd.empty_captions()`` as a fill: an input, with the pair's tags."""
        macro = INPUT_MACROS.get(name)
        if macro is None:
            raise _error(
                ErrorCode.UNSUPPORTED_SQL
                if name in MACROS
                else ErrorCode.UNKNOWN_FUNCTION,
                f"{call.display}() cannot stand in for a missing track"
                if name in MACROS
                else f"unknown function {call.display}()",
                node,
                fallback=select,
                hint=self._fill_hint(binding),
            )
        if call.named:
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"{call.display}() takes no arguments: an empty caption track "
                "has nothing to configure",
                call.named[0].value,
                fallback=node,
                hint=f"write {call.display}()",
            )
        self._check_fill_type(macro.output, call.display, binding, node, select)
        return _Stream(
            ref=self._mint_input(macro), type=macro.output, source=source_meta
        )

    def _check_fill_type(
        self,
        output: StreamType,
        display: str,
        binding: _RowBinding,
        node: exp.Expr,
        select: exp.Select,
    ) -> None:
        """A fill stands in for a track, so it has to BE one of the same type."""
        if output == binding.type:
            return
        raise _error(
            ErrorCode.UDF_ARG_TYPE,
            f"{display}() generates a {output} stream, but "
            f"'{binding.alias}' is {binding.type}",
            node,
            fallback=select,
            hint=self._fill_hint(binding),
        )

    def _fill_hint(self, binding: _RowBinding) -> str:
        spelling = _FILL_SPELLINGS.get(binding.type)
        if spelling is None:
            return (
                f"nothing generates a {binding.type} track, so there is no fill "
                f"for '{binding.alias}'; select it from a "
                "join that always matches"
            )
        return (
            f"the fill for a {binding.type} track is {spelling}; its options "
            "inherit from the paired row unless you give them"
        )

    def _inherited_fill_options(
        self, stream_type: StreamType, paired: _TrackRow | None
    ) -> dict[str, object]:
        """What the fill copies from the row it stands beside.

        Audio inherits DURATION only in v1 — a silent track's sample rate and
        layout are ffmpeg's own defaults, and amix resamples anyway, so
        inventing them would put options in the command nobody wrote. Video
        inherits size, rate and duration, because a black frame of the wrong
        size or rate is not a stand-in for the picture that is missing.
        An option the query set explicitly always wins (the caller only fills
        the ones it did not).
        """
        if paired is None:
            return {}
        columns = paired.columns
        if stream_type == "audio":
            return {"duration": columns.get("duration")}
        if stream_type == "video":
            width = columns.get("width")
            height = columns.get("height")
            size = (
                f"{int(width)}x{int(height)}"
                if isinstance(width, int | float) and isinstance(height, int | float)
                else None
            )
            return {
                "size": size,
                "rate": columns.get("fps"),
                "duration": columns.get("duration"),
            }
        return {}

    def _mint_input(self, macro: InputMacro) -> FrameRef:
        """Add the macro's own ``-i`` to the graph and ref its single stream."""
        return self._mint_stream_input(
            macro.name, macro.path, macro.format, macro.output
        )

    def _mint_stream_input(
        self, name: str, path: str, format_: str, output: StreamType
    ) -> FrameRef:
        """Add one compiler-minted ``-i`` and ref its single stream.

        The alias is spelled so no query can ever collide with it (a dot AND a
        ``#``, neither legal in an unquoted identifier), because it is not a
        name anything resolves — it exists only so the graph's alias-keyed
        input tables (``sources``, ``input_options``) can carry the slot. The
        internal ``format`` option is what puts ``-f webvtt`` before the
        ``data:`` URI; see ``ffrwd.inputs.option_spec``.
        """
        index = len(self.graph.input_paths)
        alias = f"{MACRO_NAMESPACE}.{name}#{index + 1}"
        self.graph.input_paths.append(path)
        self.graph.sources[alias] = index
        self.minted_input_options[alias] = {"format": format_}
        return f"src:{alias}:{_TYPE_MARKERS[output]}:0"

    def _source_value(
        self,
        binding: _SourceBinding,
        name: str,
        index: int | None,
        anchor: exp.Expr,
        select: exp.Select,
    ) -> _Value:
        """One column of a generated-source alias — all of it statically known.

        A source has exactly ONE output pad, of exactly one type, so the whole
        column surface is decided by ``binding.output`` with no probe
        anywhere:

        * ``a.video[1]`` / ``a.audio[1]`` — the stream, when the type matches.
        * bare ``a.video`` / ``a.audio`` — an ARRAY of length 1, so it splats
          into one Output and broadcasts a call exactly once. (Not a scalar:
          a length-1 array is still an array, the same distinction a
          single-track file's ``a.audio`` has.)
        * a subscript other than ``[1]``, or a column of the other type
          (``subtitle``/``data`` included) — STREAM_NOT_FOUND stating what the
          source does produce.
        * anything else — an unknown column.
        """
        produces = f"{binding.display} produces 1 {binding.output} stream"
        if name == TIME_COLUMN:
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"'{binding.alias}.t' is a time column, not a stream",
                anchor,
                fallback=select,
                hint=_SOURCE_DURATION_HINT,
            )
        if name == _REMOVED_FRAME:
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"'{binding.alias}.{_REMOVED_FRAME}' is not a column",
                anchor,
                fallback=select,
                hint=f"'{binding.display}' produces one {binding.output} "
                f"stream: use '{binding.alias}.{binding.output}[1]'",
            )
        array_type = _ARRAY_COLUMNS.get(name)
        if array_type is None:
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"unknown column '{binding.alias}.{name}'",
                anchor,
                fallback=select,
                hint=self._source_columns_hint(binding),
            )
        if array_type != binding.output:
            raise _error(
                ErrorCode.STREAM_NOT_FOUND,
                f"'{binding.alias}.{name}' does not exist: {produces}",
                anchor,
                fallback=select,
                hint=self._source_columns_hint(binding),
            )
        if index is None:
            return _array(binding.output, (self._source_stream_of(binding),))
        if index != 1:
            raise _error(
                ErrorCode.STREAM_NOT_FOUND,
                f"'{binding.alias}.{name}[{index}]' does not exist: {produces}",
                anchor,
                fallback=select,
                hint=_SUBSCRIPT_HINT,
            )
        return _scalar(self._source_stream_of(binding))

    def _source_columns_hint(self, binding: _SourceBinding) -> str:
        return f"'{binding.display}' exposes {binding.alias}.{binding.output}"

    def _input_value(
        self,
        alias: str,
        name: str,
        index: int | None,
        env: _Env,
        anchor: exp.Expr,
        select: exp.Select,
    ) -> _Value:
        """One column of an input alias.

        A row-bounded window makes the alias a ROW SET: it holds one ``-i``
        per surviving row (``_Env.row_inputs``), so every stream column reads
        one stream per row, in row order, and a subscript names that stream in
        each of them rather than a single one.
        """
        if name == TIME_COLUMN:
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"'{alias}.t' is a time column, not a stream",
                anchor,
                fallback=select,
                hint=_TIME_HINT,
            )
        if name == INPUT_DURATION_COLUMN:
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"'{alias}.{INPUT_DURATION_COLUMN}' is a number of seconds, "
                "not a stream",
                anchor,
                fallback=select,
                hint=f"'{alias}.{INPUT_DURATION_COLUMN}' belongs in an "
                f"expression, e.g. WHERE {alias}.t <= {alias}."
                f"{INPUT_DURATION_COLUMN} - 60",
            )
        key = tag_key(name)
        if key is not None:
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"'{alias}.{column_label(name)}' is a text tag, not a stream",
                anchor,
                fallback=select,
                hint=f"give it an alias to write it back, e.g. SELECT "
                f"{alias}.video[1], {alias}.{column_label(name)} AS {key}",
            )
        if name == TAGS_COLUMN:
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"'{alias}.{TAGS_COLUMN}' carries no streams: it is the "
                "container's tag map, and a SELECT column of a media query is "
                "an output stream",
                anchor,
                fallback=select,
                hint=f"read one key as a value, e.g. {alias}.{TAGS_COLUMN}.title",
            )
        if name in RECORD_ARRAY_COLUMNS:
            record = RECORD_ELEMENTS[name]
            if index is not None:
                raise _error(
                    ErrorCode.UNSUPPORTED_SQL,
                    f"'{alias}.{name}' cannot be subscripted: "
                    f"{article(record)} {record} is not a stream",
                    anchor,
                    fallback=select,
                    hint=record_unnest_hint(alias, name),
                )
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"'{alias}.{name}' carries no streams: it is an "
                f"array of {record} records, and a SELECT column of a media "
                "query is an output stream",
                anchor,
                fallback=select,
                hint=record_unnest_hint(alias, name),
            )
        if name in _RENDITION_SCHEMA:
            # A rendition column is real SQL, just not over THIS input: its
            # probe never turned up an ABR ladder, so `_bind_renditions` left
            # `alias` a plain `_InputBinding` rather than replacing it with a
            # rendition row table.
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"'{alias}' is a single file, not a ladder: "
                f"input('{self._path_of(alias)}') has no renditions",
                anchor,
                fallback=select,
                hint="rendition columns (bandwidth, width, height, codecs, "
                "name, language) read from an HLS master or DASH manifest",
            )
        array_type = _ARRAY_COLUMNS.get(name)
        if array_type is None:
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"unknown column '{alias}.{name}'",
                anchor,
                fallback=select,
                hint=f"an input exposes the streams {alias}.video, "
                f"{alias}.audio, {alias}.subtitle and {alias}.data, plus the "
                f"values {alias}.t, {alias}.{INPUT_DURATION_COLUMN} and its "
                f"container tags ({alias}.{TAGS_COLUMN}.title, ...)",
            )
        row_inputs = env.row_inputs.get(alias)
        if index is None:
            return self._enumerate(alias, array_type, anchor, select, row_inputs)
        stream_type: StreamType = array_type
        zero_based = index - 1

        self._check_bounds(alias, stream_type, zero_based, anchor, select)
        self._reject_codecless(
            self._stream_meta(alias, stream_type, zero_based),
            f"'{alias}.{stream_type}[{zero_based + 1}]'",
            anchor,
            select,
        )
        if row_inputs is None:
            return _scalar(self._source_stream(alias, stream_type, zero_based))
        return _array(
            stream_type,
            [
                self._source_stream(source, stream_type, zero_based)
                for source in row_inputs
            ],
        )

    def _source_stream(self, alias: str, stream_type: StreamType, index: int) -> _Stream:
        """One raw input stream, tagged with its probed metadata when there is any."""
        marker = _TYPE_MARKERS[stream_type]
        return _Stream(
            ref=f"src:{alias}:{marker}:{index}",
            type=stream_type,
            source=self._stream_meta(alias, stream_type, index),
        )

    def _stream_meta(
        self, alias: str, stream_type: StreamType, index: int
    ) -> StreamMeta | None:
        result = self.probes.get(self.row_input_source.get(alias, alias))
        if result is None:
            return None
        streams = result.by_type(stream_type)
        if not 0 <= index < len(streams):
            return None
        return streams[index]

    def _reject_codecless(
        self,
        meta: StreamMeta | None,
        display: str,
        anchor: exp.Expr,
        select: exp.Select,
    ) -> None:
        """A probed stream ffmpeg could not IDENTIFY cannot reach a media sink.

        ffprobe reporting no codec at all (e.g. a DASH manifest's WebVTT
        AdaptationSets, which ffmpeg's demuxer sees but cannot name) means
        ffmpeg can neither copy the stream (no tag to write) nor transcode it
        (no decoder to invoke) -- the run is GUARANTEED to die at header-write
        with "Could not find tag for codec none". We know at compile time, so
        we say so at compile time. Table queries are exempt on purpose: rows
        with a NULL codec column are how you DISCOVER these tracks. An
        unprobed input (meta None) is exempt too -- nothing is known, so
        nothing is knowably broken.
        """
        if self.table_mode or meta is None or meta.codec is not None:
            return
        raise _error(
            ErrorCode.UNSUPPORTED_SQL,
            f"{display} has no identifiable codec: ffmpeg's demuxer reports "
            f"none, so the stream can be neither copied nor transcoded and no "
            f"container can carry it",
            anchor,
            fallback=select,
            hint="drop it from the SELECT (a query with no COPY can still "
            "inspect it as a table row, codec column NULL); if it is a "
            "subtitle track, extract it with a tool that can read it and mux "
            "the resulting file as its own input() instead",
        )

    def _enumerate(
        self,
        alias: str,
        stream_type: StreamType,
        anchor: exp.Expr,
        select: exp.Select,
        row_inputs: list[str] | None = None,
    ) -> _Value:
        """The whole array of `alias`'s `stream_type` streams, in file order.

        The one thing lowering cannot do symbolically: an array's LENGTH is a
        property of the file, so an input that could not be probed fails here
        -- the streams of a file that cannot be read cannot be enumerated, a
        natural error rather than a policy one.

        `row_inputs` is the per-row ``-i`` list of a row-bounded window: the
        array then runs row by row, the file's own tracks inside each.
        """
        result = self.probes.get(alias)
        if result is None:
            path = self.res.input_paths[self.graph.sources[alias]]
            raise self._unreadable_error(
                ErrorCode.INPUT_NOT_FOUND,
                alias,
                f"cannot enumerate the streams of '{path}'",
                anchor,
                select,
                hint=f"'{alias}.{stream_type}' is the whole stream array, and only a "
                f"readable input can size it; subscript one stream, "
                f"e.g. {alias}.{stream_type}[1]",
            )
        count = len(result.by_type(stream_type))
        if count == 0:
            # An empty array is a column the file has no tracks for, and
            # selecting it contributes no streams - what `unnest` of it already
            # does, and what `SELECT *` already does. Worth saying, not worth
            # refusing; a sink left with no streams at all is the rejection.
            path = self.res.input_paths[self.graph.sources[alias]]
            self._warn(
                WarningCode.EMPTY_STREAM_ARRAY,
                f"{alias}.{stream_type}",
                f"'{alias}.{stream_type}' is empty: '{path}' has no "
                f"{stream_type} streams, so this column contributes nothing",
                anchor,
                hint="name the column only when the file has those tracks, or "
                "select * to take whatever it holds",
            )
            return _array(stream_type, [])
        for k in range(count):
            self._reject_codecless(
                self._stream_meta(alias, stream_type, k),
                f"'{alias}.{stream_type}[{k + 1}]'",
                anchor,
                select,
            )
        sources = [alias] if row_inputs is None else row_inputs
        return _array(
            stream_type,
            [
                self._source_stream(source, stream_type, k)
                for source in sources
                for k in range(count)
            ],
        )

    def _warn(
        self,
        code: WarningCode,
        about: str,
        message: str,
        anchor: exp.Expr,
        *,
        hint: str | None = None,
    ) -> None:
        """Say something about the compile without refusing it."""
        if self.on_warning is None:
            return
        line, col = _pos(anchor)
        self.on_warning(FfrwdWarning(code, about, message, line=line, col=col, hint=hint))

    def _cte_value(
        self,
        binding: _CteBinding,
        name: str,
        index: int | None,
        anchor: exp.Expr,
        select: exp.Select,
    ) -> _Value:
        column = self._cte_column(binding, name)
        if column is None:
            if name in binding.values:
                raise _error(
                    ErrorCode.UNSUPPORTED_SQL,
                    f"'{binding.name}.{name}' is a value, and a SELECT column "
                    "of a media query is an output stream",
                    anchor,
                    fallback=select,
                    hint="read it where values are read -- a TO expression, a "
                    f"WHERE, a GROUP BY -- without selecting it; or write it as "
                    f"metadata, e.g. STRUCT({binding.name}.{name} AS {name}) AS "
                    f"{TAGS_COLUMN}",
                )
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"unknown column '{binding.name}.{name}'",
                anchor,
                fallback=select,
                hint=self._cte_columns_hint(binding),
            )
        if index is None:
            return self._cte_column_value(binding, column, anchor, select)
        # A subscript names one element of the BODY's array, whatever the
        # branch's relation did with the rows around it.
        value = column.value
        if not value.is_array:
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"'{binding.name}.{name}' is a single stream and cannot be subscripted",
                anchor,
                fallback=select,
                hint=f"drop the subscript: '{binding.name}.{name}' already names one stream",
            )
        # The length was recorded when the CTE body lowered, so this bound is
        # STATIC: no probe is consulted here, whatever produced the array.
        if not 1 <= index <= len(value.streams):
            have = f"{len(value.streams)} stream" + ("" if len(value.streams) == 1 else "s")
            raise _error(
                ErrorCode.STREAM_NOT_FOUND,
                f"'{binding.name}.{name}[{index}]' does not exist: "
                f"column '{binding.name}.{name}' has {have}",
                anchor,
                fallback=select,
                hint=_SUBSCRIPT_HINT,
            )
        return _scalar(value.streams[index - 1])

    def _cte_column_value(
        self,
        binding: _CteBinding,
        column: _Column,
        anchor: exp.Expr,
        select: exp.Select,
    ) -> _Value:
        """One CTE column as this branch's relation reads it.

        A row-set column carries one stream per body row, so the value is that
        column re-read through the result tuples: a cross join repeats the
        stream once per partner row, and a filtered relation drops the rows it
        dropped. Everything else -- a scalar, an ``array_agg``, a bare input
        array re-exposed -- is one unit that broadcasts, exactly as before.
        """
        relation = binding.relation
        if (
            relation is None
            or not (column.splat and column.value.is_array)
            or len(column.value.streams) != binding.rows
        ):
            return column.value
        streams: list[_Stream] = []
        for position, entry in enumerate(
            tuple_.get(binding.name) for tuple_ in relation.tuples
        ):
            if isinstance(entry, _CteRow):
                streams.append(column.value.streams[entry.position])
                continue
            # An outer join's gap: a NULL cell, sentinel where a NULL is
            # allowed to stand (a table's empty cell, a manifest row's
            # absent stream kind), a rejection everywhere else.
            if self.table_mode or self.manifest is not None or self.row_reading_sink:
                streams.append(
                    _Stream(ref=_NULL_STREAM_REF, type=column.value.type, source=None)
                )
                continue
            raise _error(
                ErrorCode.STREAM_NOT_FOUND,
                f"'{binding.name}.{column.name}' is NULL in row "
                f"{position + 1}: no '{binding.name}' row matched here",
                anchor,
                fallback=select,
                hint="an outer join leaves gaps, and only a manifest "
                "destination (WITH (format 'hls'), format 'dash') takes them "
                "as absent variants; use an INNER or LEFT join so every "
                "selected row has one",
            )
        if not any(stream.ref != _NULL_STREAM_REF for stream in streams):
            raise _error(
                ErrorCode.STREAM_NOT_FOUND,
                f"'{binding.name}.{column.name}' selects nothing: no row of "
                f"'{binding.name}' survived",
                anchor,
                fallback=select,
                hint="an empty row set would select no streams; widen the WHERE",
            )
        return _array(column.value.type, streams)

    def _cte_column(self, binding: _CteBinding, name: str) -> _Column | None:
        for column in binding.columns:
            if column.name == name:
                return column
        return None

    def _cte_columns_hint(self, binding: _CteBinding) -> str:
        names = {column.name for column in binding.columns if column.name is not None}
        names |= set(binding.values)
        if not names:
            return (
                f"'{binding.name}' has no named columns; name them with AS "
                "inside its SELECT"
            )
        return f"'{binding.name}' exposes: {', '.join(sorted(names))}"

    def _check_bounds(
        self,
        alias: str,
        stream_type: StreamType,
        zero_based: int,
        anchor: exp.Expr,
        select: exp.Select,
    ) -> None:
        """Bounds-check a subscript — only possible when the input was probed."""
        result = self.probes.get(alias)
        if result is None:
            return
        available = len(result.by_type(stream_type))
        if zero_based < available:
            return
        path = self.res.input_paths[self.graph.sources[alias]]
        have = f"{available} {stream_type} stream" + ("" if available == 1 else "s")
        raise _error(
            ErrorCode.STREAM_NOT_FOUND,
            f"'{alias}.{stream_type}[{zero_based + 1}]' does not exist: "
            f"'{path}' has {have}",
            anchor,
            fallback=select,
            hint=_SUBSCRIPT_HINT,
        )

    # -- calls -------------------------------------------------------------

    def _lower_call(
        self, node: exp.Expr, call: _Call, env: _Env, select: exp.Select
    ) -> _Value:
        """Resolve a call in the registry, and nowhere else.

        One convention, three shapes of filter, tried in the order that makes
        each reachable at all:

        * :data:`ARRAY_RETURNING` (namespaced spelling ONLY) comes first: the
          v1 pad scope check keeps its names OUT of the registry entirely, so
          asking ``get`` about one first would answer "unknown".
        * an N-input filter (``DynamicFilter.n_input``) comes next, even
          though it IS an ordinary registry member now: its pad count is not
          the fixed arity ``dynamic.inputs`` gives the registry's own path,
          so it needs :func:`_n_input_spec`'s derived shape instead of
          reaching the registry proper.
        * then the registry proper, whose pad signature is the call's stream
          signature.

        A ``VARIADIC`` call is dispatched separately (:meth:`_lower_variadic_call`)
        before any of that: it only ever means "spread this array as the pad
        list", which is meaningless for a fixed-arity filter or a macro.

        ``ffmpeg.<filter>(...)`` differs from the bare spelling only in what a
        message calls the function (``call.display``) and in skipping the
        Postgres special forms at PARSE time.
        """
        name = call.name.lower()
        if call.variadic is not None and not call.is_macro:
            return self._lower_variadic_call(node, name, call, env, select)
        if not call.namespaced and not call.is_macro:
            declared = self.res.wasm.get(name)
            if declared is not None:
                if declared.is_value:
                    raise _error(
                        ErrorCode.UNSUPPORTED_SQL,
                        f"{call.display}() returns {declared.returns}, not a stream",
                        node,
                        fallback=select,
                        hint="a value-returning wasm function belongs in a "
                        "compile-time value position, e.g. inside a metadata STRUCT",
                    )
                return self._lower_wasm_call(node, declared, call, env, select)
        if call.is_macro:
            if call.variadic is not None:
                raise _error(
                    ErrorCode.UNSUPPORTED_SQL,
                    f"{call.display}() does not take VARIADIC: ffrwd macros "
                    "take a fixed number of streams",
                    node,
                    fallback=select,
                    hint=_VARIADIC_HINT,
                )
            return self._lower_macro_call(node, name, call, env, select)
        if call.namespaced:
            options = self._array_options(name)
            if options is not None:
                return self._lower_array_call(
                    node, ARRAY_RETURNING[name], options, call, env, select
                )
        n_input = self._n_input_call(name)
        if n_input is not None:
            spec, options = n_input
            return self._lower_n_input_call(node, spec, options, call, env, select)
        dynamic = self.registry.get(name) if self.registry is not None else None
        if dynamic is None:
            raise _error(
                ErrorCode.UNKNOWN_FUNCTION,
                f"unknown function {call.display}()",
                node,
                fallback=select,
                hint=self._namespaced_function_hint(name)
                if call.namespaced
                else self._unknown_function_hint(name),
            )
        target_name, target = self._dispatch_audio(name, dynamic, call, env, select)
        return self._lower_dynamic_call(node, target_name, target, call, env, select)

    # -- wasm calls --

    def _described(
        self, declared: WasmFunction, node: exp.Expr, select: exp.Select
    ) -> Described:
        """What the module a declaration names turned out to declare, checked.

        The describe itself happened before lowering, once per module path
        (:mod:`ffrwd.wasm`); what happens here is comparing it against the
        declaration that named it. Both rejections anchor on the CALL, since
        the declaration's own position is not in the query being lowered by
        the time a rejection is worth reporting.
        """
        described = self.describes.get(declared.module)
        if described is None:
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"the module '{declared.module}' was never described",
                node,
                fallback=select,
                hint="this is a compiler bug; please report the query that "
                "produced it",
            )
        if described.world not in WORLDS:
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"the module '{declared.module}' targets {described.world}, and "
                f"this ffrwd hosts {' or '.join(WORLDS)}",
                node,
                fallback=select,
                hint="rebuild the module against a world this ffrwd hosts, or "
                "upgrade ffrwd",
            )
        if described.name != declared.export:
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"function '{declared.name}' names the export '{declared.export}', "
                f"and '{declared.module}' exports '{described.name}'",
                node,
                fallback=select,
                hint=f"a module carries one filter; write '{described.name}' as "
                "the export",
            )
        if described.packet_sink:
            self._check_packet_sink(declared, described, node, select)
        if described.both_kinds:
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"the module '{declared.module}' accepts both pixel formats and "
                "sample formats",
                node,
                fallback=select,
                hint="a module filters video or audio; rebuild it declaring one "
                "of the two",
            )
        # A module naming NEITHER list has nothing to compare against, and is
        # refused where its wire format is negotiated instead.
        if described.kind is not None and described.kind != declared.stream_kind:
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"function '{declared.name}' takes {declared.returns}, and the "
                f"module '{declared.module}' filters {described.kind}",
                node,
                fallback=select,
                hint=f"declare the stream and the return as "
                f"{WASM_STREAM_NAMES[described.kind]}, or name a module that "
                f"filters {declared.stream_kind}",
            )
        # A packet sink has no frame interface to read a window over: how many
        # streams of each kind it takes is what it declares, and that is
        # checked against the signature in `_check_sink_shape`.
        if not described.packet_sink:
            self._check_stream_arity(declared, described, node, select)
        if declared.emits is not None:
            self._check_annotation_schema(declared, declared.emits, described, node, select)
        # A windowed module is handed each frame's rows either way and reads
        # them at its own option, so a declaration without an annotation
        # column just wires none in. A per-frame consumer exists only to read
        # them, so there the bare declaration is a mistake.
        if (
            described.reads_annotations
            and declared.reads is None
            and not described.windowed
        ):
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"the module '{declared.module}' reads annotations off its "
                f"frames, and '{declared.name}' takes none",
                node,
                fallback=select,
                hint="declare an annotation column right after the stream: "
                f"{declared.name}(<stream> {declared.returns}, <name> "
                "STRUCT(<field> <type>, ...)[])",
            )
        if not described.reads_annotations and declared.reads is not None:
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"function '{declared.name}' takes the annotation column "
                f"'{declared.reads.name}', and the module '{declared.module}' "
                "does not read annotations",
                node,
                fallback=select,
                hint="drop the annotation column, or use a module built to "
                "consume them",
            )
        # Only a windowed module can be handed no rows: a per-frame consumer
        # reads them on every frame, so its column cannot be optional.
        if (
            declared.reads is not None
            and declared.reads_optional
            and not described.windowed
        ):
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"function '{declared.name}' defaults the annotation column "
                f"'{declared.reads.name}', and the module '{declared.module}' "
                "reads rows on every frame",
                node,
                fallback=select,
                hint="drop the DEFAULT; a per-frame consumer always needs a "
                "producer under it",
            )
        return described

    def _check_packet_sink(
        self,
        declared: WasmFunction,
        described: Described,
        node: exp.Expr,
        select: exp.Select,
    ) -> None:
        """A packet-sink module against the declaration that named it.

        The module consumes the encoder's own output: it is a COPY
        destination over one video stream, hosted only by a sidecar new
        enough to hand packets through. Each mismatch is refused here, where
        the run-time refusal it forestalls can be said at the call.
        """
        if not hosts_packet_sink(described.world):
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"the module '{declared.module}' consumes encoded packets, and "
                f"the sidecar's {described.world} cannot hand them through",
                node,
                fallback=select,
                hint="packet sinks arrived with ffrwd:av@0.10.0; upgrade "
                "ffrwd, or point at a newer ffrwd-wasm",
            )
        if not declared.is_sink:
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"function '{declared.name}' returns {declared.returns}, and "
                f"the module '{declared.module}' consumes encoded packets and "
                "hands nothing back",
                node,
                fallback=select,
                hint=f"declare '{declared.name}' as RETURNS sink and write it "
                "as a COPY destination",
            )
        # An audio pad reaches a sink only where the module accepts a codec
        # the stream edge can carry: the edge is what the sidecar's NUT reader
        # hands through, and it hands through nothing else.
        if "audio" in declared.stream_kinds:
            accepted = described.sink_codecs("audio")
            if accepted and not any(c in WIRE_AUDIO_CODECS for c in accepted):
                raise _error(
                    ErrorCode.UNSUPPORTED_SQL,
                    f"the module '{declared.module}' consumes "
                    f"{_join_codecs(accepted)} audio, and the stream edge into "
                    f"a packet sink carries {_join_codecs(WIRE_AUDIO_CODECS)}",
                    node,
                    fallback=select,
                    hint="the module has to accept one of the codecs the "
                    "sidecar's packets travel in",
                )
        # A row-reading sink declares no stream parameters at all -- its
        # shape is judged against the SELECT list's actual rows instead,
        # once they are known (:meth:`_check_row_sink_arity`), not here
        # against a signature that names none.
        if not declared.reads_rows_from_select:
            self._check_sink_shape(declared, described, node, select)

    def _check_sink_shape(
        self,
        declared: WasmFunction,
        described: Described,
        node: exp.Expr,
        select: exp.Select,
    ) -> None:
        """What the signature says it reads against what the module declares.

        Per kind, since a sink reads each independently. A declaration naming
        one stream of a kind always hands over one, so a module reading one,
        many or any accepts it; anything that can hand over SEVERAL -- an
        array, or several parameters of one kind -- needs a module that reads
        many or any. A module reading ANY works without the kind entirely, so
        a declaration naming none of it is a shape it accepts too. The refusal
        lands here, at the call, where the run-time one it forestalls can be
        said before anything runs.
        """
        for kind in WASM_STREAM_NAMES:
            params = [
                param
                for param, declared_kind in zip(
                    declared.stream_params, declared.stream_kinds, strict=True
                )
                if declared_kind == kind
            ]
            reads = described.sink_streams(kind)
            if not params:
                if reads not in ("none", "any"):
                    raise _error(
                        ErrorCode.UNSUPPORTED_SQL,
                        f"the module '{declared.module}' reads {reads} "
                        f"{kind} stream(s), and '{declared.name}' declares no "
                        f"{WASM_STREAM_NAMES[kind]} parameter",
                        node,
                        fallback=select,
                        hint=f"declare a {WASM_STREAM_NAMES[kind]} parameter "
                        f"before the module's value parameters",
                    )
                continue
            several = len(params) > 1 or any(is_array(p.type) for p in params)
            if reads == "none" or (several and reads not in ("many", "any")):
                written = ", ".join(f"{p.name} {p.type}" for p in params)
                raise _error(
                    ErrorCode.UNSUPPORTED_SQL,
                    f"function '{declared.name}' declares ({written}), and the "
                    f"module '{declared.module}' reads {reads} {kind} stream"
                    f"{'' if reads == 'one' else 's'}",
                    node,
                    fallback=select,
                    hint="declare what the module reads: "
                    + (
                        f"no {WASM_STREAM_NAMES[kind]} parameter"
                        if reads == "none"
                        else f"one {WASM_STREAM_NAMES[kind]} parameter"
                    ),
                )

    def _check_stream_arity(
        self,
        declared: WasmFunction,
        described: Described,
        node: exp.Expr,
        select: exp.Select,
    ) -> None:
        """The streams the signature declares against the streams the module reads.

        A module reading several at once is a WINDOWED-interface export that
        hands one frame back per frame in: a windowing one has no answer for
        which frame of which input pairs with which, and a per-frame one
        predates the declaration entirely.
        """
        reads = described.inputs
        if reads > 1 and not described.windowed:
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"the module '{declared.module}' reads {reads} streams, and "
                "declares the per-frame interface",
                node,
                fallback=select,
                hint="rebuild the module against the windowed interface, which "
                "is where a module reads more than one stream",
            )
        if reads > 1 and (described.window != 1 or described.stride != 1):
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"the module '{declared.module}' reads {reads} streams over a "
                f"window of {described.window} every {described.stride}",
                node,
                fallback=select,
                hint="a module reading several streams takes one frame off each "
                "and hands one back; rebuild it with a window and stride of 1",
            )
        if declared.stream_arity != reads:
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"function '{declared.name}' declares {declared.stream_arity} "
                f"stream parameter{'' if declared.stream_arity == 1 else 's'}, "
                f"and the module '{declared.module}' reads {reads}",
                node,
                fallback=select,
                hint=f"declare one {declared.returns} parameter per stream the "
                f"module reads, {reads} of them, before its value parameters",
            )

    def _check_annotation_schema(
        self,
        declared: WasmFunction,
        annotation: Annotation,
        described: Described,
        node: exp.Expr,
        select: exp.Select,
    ) -> None:
        """The declared annotation record against the rows the module says it emits.

        Field for field, by name and by type. A module publishing SEVERAL row
        shapes is matched against each in turn, and one arm fitting is enough.
        A module that declares no rows has nothing to annotate with, so
        declaring an annotation return over one is a rejection of its own.
        """
        arms = rows_arms(described)
        if arms is None:
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"function '{declared.name}' returns the annotation column "
                f"'{annotation.name}', and the module '{declared.module}' emits no rows",
                node,
                fallback=select,
                hint=f"declare '{declared.name}' as RETURNS {declared.returns}; a "
                "module that reads nothing off its frames has no annotations "
                "to return",
            )
        fields = _annotation_fields(annotation)
        if any(_annotation_matches(fields, arm) for arm in arms):
            return
        raise _error(
            ErrorCode.UDF_ARG_TYPE,
            f"function '{declared.name}' declares '{annotation.name}' as "
            f"{annotation.written}, and the module '{declared.module}' "
            f"emits {' or '.join(_written_json_fields(arm) for arm in arms)}",
            node,
            fallback=select,
            hint="an annotation record names the module's own row columns, "
            "with a type each value fits",
        )

    def _wasm_params(
        self,
        declared: WasmFunction,
        described: Described,
        call: _Call,
        node: exp.Expr,
        select: exp.Select,
        env: _Env,
        row: _RowTuple,
        *,
        first: int,
    ) -> dict[str, object]:
        """The value arguments as the module's own parameters, schema-checked.

        `first` is the index the value arguments start at: past the streams,
        and past the annotation column when the call wrote it explicitly.
        A parameter left NULL or unwritten is OMITTED, the way absence works
        everywhere else in the dialect -- the module then sees its own
        default. What is written is checked against the schema the module
        declares, by name and by type.
        """
        properties = described.params_schema.get("properties")
        known = properties if isinstance(properties, dict) else {}
        params: dict[str, object] = {}
        for index, param in enumerate(declared.value_params, start=first):
            written = call.args[index] if index < len(call.args) else param.default
            if written is None:
                continue
            value = self._eval_value(written, env, row, select)
            if value is None:
                continue
            anchor = call.args[index] if index < len(call.args) else node
            schema = known.get(param.name)
            if schema is None:
                raise _error(
                    ErrorCode.UDF_ARG_TYPE,
                    f"the module '{declared.module}' has no parameter "
                    f"'{param.name}'",
                    anchor,
                    fallback=select,
                    hint=_declares_params(known),
                )
            self._check_wasm_param(param.name, value, schema, anchor, select)
            params[param.name] = value
        return params

    def _check_wasm_param(
        self,
        name: str,
        value: RowValue,
        schema: object,
        anchor: exp.Expr,
        select: exp.Select,
    ) -> None:
        """One written parameter against the JSON Schema type the module gave it."""
        wanted = schema.get("type") if isinstance(schema, dict) else None
        if not isinstance(wanted, str) or wanted not in _JSON_TYPES:
            return  # a schema shape this compiler does not judge
        allowed = _JSON_TYPES[wanted]
        # bool is an int in Python, and a module asking for a number does not
        # mean true.
        if isinstance(value, bool) != (wanted == "boolean"):
            raise self._bad_wasm_param(name, value, wanted, anchor, select)
        if not isinstance(value, allowed):
            raise self._bad_wasm_param(name, value, wanted, anchor, select)
        if wanted == "integer" and isinstance(value, float) and value != int(value):
            raise self._bad_wasm_param(name, value, wanted, anchor, select)

    def _bad_wasm_param(
        self,
        name: str,
        value: RowValue,
        wanted: str,
        anchor: exp.Expr,
        select: exp.Select,
    ) -> FfrwdError:
        return _error(
            ErrorCode.UDF_ARG_TYPE,
            f"the module's parameter '{name}' is {wanted}, got {value!r}",
            anchor,
            fallback=select,
            hint=f"write a value the module can take: '{name}' is {wanted}",
        )

    # -- wasm value calls --

    def _described_value(
        self, declared: WasmFunction, node: exp.Expr, select: exp.Select
    ) -> DescribedFunction:
        """What the module's own function turned out to declare, checked.

        Mirrors :meth:`_described`, for a VALUE function instead of a stream
        one: the export named in the ``functions`` list, its parameters
        matched name-for-name against the declaration, and its result type
        against RETURNS.
        """
        described = self.describes.get(declared.module)
        if described is None:
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"the module '{declared.module}' was never described",
                node,
                fallback=select,
                hint="this is a compiler bug; please report the query that "
                "produced it",
            )
        if described.world not in WORLDS:
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"the module '{declared.module}' targets {described.world}, and "
                f"this ffrwd hosts {' or '.join(WORLDS)}",
                node,
                fallback=select,
                hint="rebuild the module against a world this ffrwd hosts, or "
                "upgrade ffrwd",
            )
        if not described.functions:
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"the module '{declared.module}' declares no functions",
                node,
                fallback=select,
                hint=f"a value-returning wasm function needs '{declared.export}' "
                "in the module's own function list",
            )
        found = next(
            (fn for fn in described.functions if fn.name == declared.export), None
        )
        if found is None:
            names = ", ".join(sorted(fn.name for fn in described.functions))
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"function '{declared.name}' names the export '{declared.export}', "
                f"and '{declared.module}' offers {names}",
                node,
                fallback=select,
                hint=f"a module's function list names what it offers; write one "
                f"of {names} as the export",
            )
        self._check_wasm_result_type(declared, found, node, select)
        return found

    def _check_wasm_result_type(
        self,
        declared: WasmFunction,
        described: DescribedFunction,
        node: exp.Expr,
        select: exp.Select,
    ) -> None:
        """RETURNS against the module's own ``result_schema``, once per declaration."""
        schema = described.result_schema
        wanted = schema.get("type") if isinstance(schema, dict) else None
        if isinstance(wanted, str) and wanted in ANNOTATION_TYPES.get(declared.returns, ()):
            return
        raise _error(
            ErrorCode.UDF_ARG_TYPE,
            f"function '{declared.name}' declares RETURNS {declared.returns}, and "
            f"the module's function '{declared.export}' returns "
            f"{wanted if isinstance(wanted, str) else 'nothing'}",
            node,
            fallback=select,
            hint="a value-returning wasm function's RETURNS matches what the "
            "module's function declares",
        )

    def _eval_wasm_value(
        self,
        declared: WasmFunction,
        call: _Call,
        node: exp.Expr,
        env: _Env,
        rows: _RowTuple,
        select: exp.Select,
    ) -> RowValue:
        """A call to a value-returning wasm function: run it now, fold the result.

        Every argument is itself a compile-time value, through this same
        grammar -- which is what lets ``brand(f.tags.title, ...)`` read a
        probed tag. NULL, written or omitted, drops the argument the same way
        absence works everywhere else in the dialect; the module then sees no
        key for it. The module runs once per distinct (module, function,
        arguments) within this compile (:attr:`_invoke_cache`).
        """
        if call.named:
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"{declared.name}() does not take named arguments",
                call.named[0].value,
                fallback=node,
                hint=f"a wasm function's parameters are positional: "
                f"{declared.signature}",
            )
        described = self._described_value(declared, node, select)
        properties = described.params_schema.get("properties")
        known = properties if isinstance(properties, dict) else {}
        args: dict[str, object] = {}
        for param, argument in zip(declared.value_params, call.args):
            value = self._eval_value(argument, env, rows, select)
            if value is None:
                continue
            schema = known.get(param.name)
            if schema is None:
                raise _error(
                    ErrorCode.UDF_ARG_TYPE,
                    f"the module '{declared.module}' has no parameter "
                    f"'{param.name}'",
                    argument,
                    fallback=select,
                    hint=_declares_params(known),
                )
            self._check_wasm_param(param.name, value, schema, argument, select)
            args[param.name] = value
        key = (declared.module, declared.export, tuple(sorted(args.items())))
        cached = self._invoke_cache.get(key, _UNCACHED)
        if cached is _UNCACHED:
            try:
                result = self.invoke(declared.module, declared.export, args)
            except FfrwdError as err:
                raise _error(
                    ErrorCode.UNSUPPORTED_SQL,
                    f"function '{declared.name}': {err.message}",
                    node,
                    fallback=select,
                    hint=err.hint,
                ) from err
            self._invoke_cache[key] = result
        else:
            result = cached
        return self._folded_result(declared, result, node, select)

    def _folded_result(
        self, declared: WasmFunction, result: object, node: exp.Expr, select: exp.Select
    ) -> RowValue:
        """The module's JSON answer as this call's compile-time value.

        Checked against the DECLARED return type, not the schema -- the
        schema was already checked once, at :meth:`_described_value`, but the
        module could still hand back a value of the wrong JSON type at this
        particular call.
        """
        if declared.returns == "boolean":
            if isinstance(result, bool):
                return result
        elif not isinstance(result, bool):
            if declared.returns == "text" and isinstance(result, str):
                return result
            if declared.returns == "number" and isinstance(result, int | float):
                return result
        raise _error(
            ErrorCode.UDF_ARG_TYPE,
            f"function '{declared.name}' declares RETURNS {declared.returns}, and "
            f"the module '{declared.module}' returned {result!r}",
            node,
            fallback=select,
            hint=f"the module's result must be {declared.returns}",
        )

    def _annotating_call(self, node: exp.Expr) -> WasmFunction | None:
        """The annotation-returning wasm function `node` calls, if it calls one."""
        call = _call_parts(_unwrap(node))
        if call is None or call.namespaced or call.is_macro:
            return None
        found = self.res.wasm.get(call.name.lower())
        return found if found is not None and found.emits is not None else None

    def _reads_annotations(self, node: exp.Expr) -> WasmFunction | None:
        """The annotation-taking wasm function `node` is written as an argument of.

        Through a field read as well as directly: a call writing both halves
        of a struct names each of them, and each name is one of its arguments.
        """
        inner, parent = node, node.parent
        while isinstance(parent, exp.Paren) or (
            isinstance(parent, exp.Dot) and parent.this is inner
        ):
            inner, parent = parent, parent.parent
        if not isinstance(parent, exp.Expr):
            return None
        call = _call_parts(parent)
        if call is None or call.namespaced or call.is_macro:
            return None
        found = self.res.wasm.get(call.name.lower())
        return found if found is not None and found.reads is not None else None

    def _check_annotation_argument(
        self, declared: WasmFunction, call: _Call, node: exp.Expr, select: exp.Select
    ) -> None:
        """That the annotation columns at a call site line up, both ways.

        A function taking annotations is written over the call that produces
        them, or writes the column itself; either way their records have to be
        the same shape. A function RETURNING them has to be written under one
        that takes them: the struct it produces is not a stream, and nothing
        else in the dialect reads one.
        """
        # Any stream argument may be the producer: a module reading several
        # streams is handed annotations by whichever of them returns some. A
        # call writing the column names its producer there instead.
        anchor, producer = next(
            (
                (argument, found)
                for argument in call.args[: max(declared.stream_arity, 1)]
                if (found := self._annotating_call(argument)) is not None
            ),
            (call.args[0] if call.args else node, None),
        )
        at = declared.stream_arity
        gathered = (
            annotation_projection(_unwrap(call.args[at]), self.res.wasm)
            if declared.reads is not None and len(call.args) > at
            else None
        )
        if gathered is not None:
            anchor, producer = call.args[at], gathered[1]
        if (
            declared.emits is not None
            and self._reads_annotations(node) is None
            and not _projects_annotations(node, declared.emits.name)
        ):
            raise _error(
                ErrorCode.UDF_ARG_TYPE,
                f"{declared.name}() returns the annotation column "
                f"'{declared.emits.name}', and nothing here reads it",
                node,
                fallback=select,
                hint=f"read the column off the call, {declared.name}"
                f"(...).{declared.emits.name}, or pass {declared.name}(...) to a "
                "function that takes an annotation column; a struct is not a "
                "stream and cannot be selected, trimmed or written",
            )
        if declared.reads is None:
            if producer is None or producer.emits is None:
                return
            raise _error(
                ErrorCode.UDF_ARG_TYPE,
                f"{declared.name}() takes {declared.returns}, and {producer.name}() "
                f"returns it with the annotation column '{producer.emits.name}'",
                anchor,
                fallback=node,
                hint=f"declare {declared.name}() with an annotation column after "
                f"its stream, or call it over a plain {declared.returns}",
            )
        if producer is None:
            if declared.reads_optional:
                return
            raise _error(
                ErrorCode.UDF_ARG_TYPE,
                f"{declared.name}() takes the annotation column "
                f"'{declared.reads.name}', and its argument produces none",
                anchor,
                fallback=node,
                hint=f"call {declared.name}() over a function that returns "
                "annotations, or declare the column DEFAULT NULL to make it "
                "optional",
            )
        assert producer.emits is not None  # what _annotating_call selected on
        if _annotation_fields(declared.reads) == _annotation_fields(producer.emits):
            return
        raise _error(
            ErrorCode.UDF_ARG_TYPE,
            f"{declared.name}() takes '{declared.reads.name}' as "
            f"{declared.reads.written}, and {producer.name}() returns "
            f"'{producer.emits.name}' as {producer.emits.written}",
            anchor,
            fallback=node,
            hint="the two annotation records have to name the same fields, "
            "with the same types",
        )

    def _written_annotation(
        self,
        declared: WasmFunction,
        call: _Call,
        env: _Env,
        node: exp.Expr,
        select: exp.Select,
    ) -> _Value | None:
        """What a consumer reads when its call WRITES the annotation column.

        The rows and the stream are two halves of one struct, so the producer
        is lowered ONCE and the consumer reads what comes out of it -- through
        the row filter node, where the gather narrowed the rows. None for the
        implicit spelling, where a single argument covers both halves.

        A stream argument naming a different producer is a rejection: the rows
        ride the stream they were read off, and no other.
        """
        at = declared.stream_arity
        if declared.reads is None or len(call.args) <= at:
            return None
        written = _unwrap(call.args[at])
        found = annotation_projection(written, self.res.wasm)
        if found is None:
            return None
        rows_call = found[0]
        stream_call = _stream_projection(call.args[0], self.res.wasm)
        if stream_call is None or stream_call.sql() != rows_call.sql():
            raise _error(
                ErrorCode.UDF_ARG_TYPE,
                f"{declared.name}() reads rows off a stream "
                f"{found[1].name}() did not produce",
                call.args[0],
                fallback=node,
                hint=f"write the same call for both halves, e.g. {declared.name}"
                f"({found[1].name}(...).{found[1].stream_field}, <rows off "
                f"{found[1].name}(...)>)",
            )
        produced = self._lower_expr(rows_call, env, select)
        return self._row_filtered(produced, written)

    def _row_filtered(self, value: _Value, node: exp.Expr) -> _Value:
        """`value` through the row filter its gather wrote, or `value` itself.

        The predicate rides the frames' own chain: the node narrows the rows
        the stream carries and hands both on, so a filtered column and an
        unfiltered one are the same value with one more node in front of it.
        """
        predicate = node.meta.get(ROW_PREDICATE)
        if not isinstance(predicate, str):
            return value
        return replace(
            value,
            streams=tuple(
                replace(
                    stream,
                    ref=self.ctx.node(
                        ROWFILTER,
                        {PREDICATE: predicate},
                        [stream.ref],
                        [stream.type],
                        reads_annotations=True,
                    ),
                )
                for stream in value.streams
            ),
        )

    def _lower_wasm_call(
        self, node: exp.Expr, declared: WasmFunction, call: _Call, env: _Env,
        select: exp.Select,
    ) -> _Value:
        """A call to a ``LANGUAGE wasm`` function: one node the sidecar hosts.

        The node's FILTER is the module path, which is what marks it as one
        ffmpeg cannot run; its ARGS are the module's own parameters. Its
        stream signature is what the declaration named: one stream per leading
        parameter, all of one kind, and one of that kind out. Each stream
        argument becomes an input edge of its own, so the same stream written
        twice reaches the module through the ordinary split.

        A call that WRITES its annotation column names the producer twice --
        once for each half of the struct -- and the two are one chain:
        :meth:`_written_annotation` lowers the producer once and hands back
        the pad the module reads.
        """
        described = self._described(declared, node, select)
        if call.named:
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"{declared.name}() does not take named arguments",
                call.named[0].value,
                fallback=node,
                hint=f"a wasm function's parameters are positional: "
                f"{declared.signature}",
            )
        if declared.is_sink:
            return self._lower_sink_call(node, declared, described, call, env, select)
        kind = declared.stream_kind
        arity = declared.stream_arity
        expected: list[StreamType] = [kind] * arity
        kinds = self._stream_kinds(call, env, select, arity)
        if kinds != expected:
            raise self._bad_streams(call, node, select, expected, kinds)
        self._check_annotation_argument(declared, call, node, select)
        positions = list(range(arity))
        wired = self._written_annotation(declared, call, env, node, select)
        streams = {
            position: (
                wired
                if position == 0 and wired is not None
                else self._lower_expr(call.args[position], env, select)
            )
            for position in positions
        }
        tuples = env.relation.tuples if env.relation is not None else []
        first = arity + (1 if wired is not None else 0)
        # A module parameter read off a row is one instance per row, the way a
        # filter option read off one is one node per row.
        per_row = any(_reads_row_column(arg, env) for arg in call.args[first:])

        def build(values: list[object], element: int) -> FrameRef:
            row = tuples[element] if element < len(tuples) else {}
            params = self._wasm_params(
                declared,
                described,
                call,
                node,
                select,
                env,
                row,
                first=first,
            )
            ref = self.ctx.node(
                declared.module,
                params,
                [_as_ref(values[position]) for position in positions],
                [kind],
                reads_annotations=declared.reads is not None,
            )
            return ref

        lowered = self._expand_call(
            declared.name,
            node,
            call.args[:arity],
            select,
            streams=streams,
            literals={},
            arity=arity,
            positions=positions,
            returns=kind,
            build=build,
            rows=self._row_elements(per_row, env),
        )
        return lowered

    def _lower_sink_call(
        self,
        node: exp.Expr,
        declared: WasmFunction,
        described: Described,
        call: _Call,
        env: _Env,
        select: exp.Select,
    ) -> _Value:
        """A ``RETURNS sink`` call: every stream the SELECT list carries, into
        ONE instance.

        A sink is a filter with no output pads and reads streams the way one
        does -- except that how MANY is the query's to say, not the
        declaration's. The COPY's SELECT list is rewritten into the leading
        arguments of this call (:meth:`_rewrite_sink_copy`), each of which may
        itself carry a whole array of streams, so the pads are the FLATTENED
        run of them in SELECT order, bound to the declared parameters by KIND.

        One node, always: N calls would be N instances with nothing shared
        between them, which for a sink over a rendition ladder is the whole
        difference.

        A sink declaring no stream parameters (:attr:`WasmFunction.
        reads_rows_from_select`) reads the SELECT list's cells instead --
        dispatched to :meth:`_lower_row_reading_sink_call`, which is the
        manifest destination's own reading of a relation, applied to a
        module instead of a written map.
        """
        if declared.reads_rows_from_select:
            return self._lower_row_reading_sink_call(
                node, declared, described, call, env, select
            )
        at = _sink_stream_count(node, len(call.args))
        gathered: list[tuple[_Stream, exp.Expr]] = []
        for argument in call.args[:at]:
            value = self._lower_expr(argument, env, select)
            gathered += [(stream, argument) for stream in value.streams]
        pads = self._bind_sink_streams(declared, gathered, node, select)
        # One instance, so its parameters are read once. A row-varying value
        # argument has no one row to read here; the FIRST is what a call over
        # a gathered relation means everywhere else.
        tuples = env.relation.tuples if env.relation is not None else []
        params = self._wasm_params(
            declared,
            described,
            call,
            node,
            select,
            env,
            tuples[0] if tuples else {},
            first=at,
        )
        ref = self.ctx.node(
            declared.module,
            params,
            [stream.ref for stream in pads],
            [pads[0].type],
        )
        # The module is the destination: its pad reaches no file and no other
        # node, and the record here is what says so.
        self.graph.module_sinks.append(ref)
        # Nothing downstream reads a sink: the value carries no streams, the
        # way a rows projection's does not.
        return _Value(type=pads[0].type, streams=(), is_array=False)

    def _lower_row_reading_sink_call(
        self,
        node: exp.Expr,
        declared: WasmFunction,
        described: Described,
        call: _Call,
        env: _Env,
        select: exp.Select,
    ) -> _Value:
        """A sink declaring no stream parameters: the SELECT list's cells,
        read the way a manifest destination reads its rows.

        N rows, each a video cell and an audio cell, either NULL
        (:meth:`_row_cells`, the same cardinality and shape logic a
        manifest destination runs); flattened here into the sink's pads in
        ROW-MAJOR order -- row 0's video, row 0's audio, row 1's video, ...
        -- skipping NULL cells, each pad remembering the row it came from
        and what the relation said of that row (:meth:`_row_renditions`).
        """
        at = _sink_stream_count(node, len(call.args))
        columns = [
            _Column(
                name=None,
                value=self._lower_expr(argument, env, select),
                splat=self._is_splat_projection(argument, env),
            )
            for argument in call.args[:at]
        ]
        cardinality = max(len(self.sink_rows), 1)
        rows, _ = self._row_cells(columns, cardinality, node, f"'{declared.name}'")
        self._check_no_null_stream_feeds_a_filter(node)
        self._check_row_sink_arity(declared, described, rows, node, select)
        renditions = self._row_renditions(rows, env)
        pads: list[_Stream] = []
        meta: list[dict[str, object]] = []
        for position, row in enumerate(rows):
            for stream in (row.video, row.audio):
                if stream is None:
                    continue
                pads.append(stream)
                entry: dict[str, object] = {"row": position}
                rendition = renditions[position]
                if rendition:
                    entry["rendition"] = rendition
                meta.append(entry)
        tuples = env.relation.tuples if env.relation is not None else []
        params = self._wasm_params(
            declared,
            described,
            call,
            node,
            select,
            env,
            tuples[0] if tuples else {},
            first=at,
        )
        ref = self.ctx.node(
            declared.module,
            params,
            [stream.ref for stream in pads],
            [pads[0].type],
        )
        self.row_reading_sink_pads[ref] = meta
        self.row_reading_sink_rows[ref] = rows
        self.graph.module_sinks.append(ref)
        return _Value(type=pads[0].type, streams=(), is_array=False)

    def _check_row_sink_arity(
        self,
        declared: WasmFunction,
        described: Described,
        rows: list[_VariantRow],
        node: exp.Expr,
        select: exp.Select,
    ) -> None:
        """A sink whose arity holds one row at a time, handed several.

        `video_streams`/`audio_streams` say how many pads of each kind a
        sink reads; neither declaring "many" or "any" means it is
        file-shaped, and a multi-row relation reaching it is the one-row
        rule again (:meth:`_check_one_row_per_file`, which a row-reading
        sink skips), this time naming the sink instead of a path.
        """
        if len(rows) <= 1:
            return
        if described.video_streams in ("many", "any") or described.audio_streams in (
            "many",
            "any",
        ):
            return
        raise _error(
            ErrorCode.ROW_COUNT_MISMATCH,
            f"this query has {len(rows)} rows, and '{declared.name}()' reads one",
            node,
            fallback=select,
            hint="narrow to one row with WHERE or ORDER BY ... LIMIT, or use "
            "a sink whose module reads many",
        )

    def _row_renditions(
        self, rows: list[_VariantRow], env: _Env
    ) -> list[dict[str, object]]:
        """Rendition attributes per row, aligned with `rows`.

        Read straight off the rendition table's own columns when the
        relation came from one (a manifest or a module source alias);
        otherwise derived exactly as ``var_stream_map`` names rows today
        (:meth:`_variant_names`) -- height for a video row, nothing else.
        """
        alias = next(
            (
                a
                for a, binding in env.bindings.items()
                if isinstance(binding, _RowBinding) and binding.column == RENDITION_COLUMN
            ),
            None,
        )
        if alias is not None:
            out: list[dict[str, object]] = []
            for position in range(len(rows)):
                track = (
                    self.sink_rows[position].get(alias)
                    if position < len(self.sink_rows)
                    else None
                )
                columns = track.columns if isinstance(track, _TrackRow) else {}
                out.append(
                    {
                        name: columns[name]
                        for name in ("name", "bandwidth", "codecs", "language")
                        if columns.get(name) is not None
                    }
                )
            return out
        video_names, audio_names = self._variant_names(rows)
        out = []
        video_seen = audio_seen = 0
        for row in rows:
            if row.video is not None:
                out.append({"name": video_names[video_seen]})
                video_seen += 1
            elif row.audio is not None:
                out.append({"name": audio_names[audio_seen]})
                audio_seen += 1
            else:
                out.append({})
        return out

    def _bind_sink_streams(
        self,
        declared: WasmFunction,
        gathered: list[tuple[_Stream, exp.Expr]],
        node: exp.Expr,
        select: exp.Select,
    ) -> list[_Stream]:
        """The SELECT's streams against the sink's parameters, matched by kind.

        Each parameter takes streams of its own kind in SELECT order: a bare
        one exactly one, an ARRAY one every remaining stream of that kind. The
        pads come back in DECLARATION order, which is the order `init` names
        them, so a module reading video and audio knows which is which without
        being told.
        """
        remaining: dict[StreamType, list[tuple[_Stream, exp.Expr]]] = {}
        for entry in gathered:
            remaining.setdefault(entry[0].type, []).append(entry)
        pads: list[_Stream] = []
        for param, kind in zip(
            declared.stream_params, declared.stream_kinds, strict=True
        ):
            waiting = remaining.get(kind, [])
            wanted = len(waiting) if is_array(param.type) else 1
            if len(waiting) < max(wanted, 1):
                raise _error(
                    ErrorCode.UDF_ARG_TYPE,
                    f"{declared.name}() takes '{param.name}' as {param.type}, "
                    f"and this query's SELECT list carries "
                    f"{_stream_count(len(waiting))} of that kind",
                    node,
                    fallback=select,
                    hint=f"a sink reads the streams its SELECT list names: "
                    f"COPY (SELECT <{kind} stream>, ...) TO {declared.called}"
                    f"(<values>)",
                )
            pads += [stream for stream, _ in waiting[:wanted]]
            remaining[kind] = waiting[wanted:]
        left = [entry for entries in remaining.values() for entry in entries]
        if left:
            stream, anchor = left[0]
            raise _error(
                ErrorCode.UDF_ARG_TYPE,
                f"{declared.name}() reads {_stream_count(len(pads))}, and this "
                f"query's SELECT list carries "
                f"{_stream_count(len(pads) + len(left))}",
                anchor,
                fallback=node,
                hint=f"declare the parameter as {WASM_STREAM_NAMES[stream.type]}[] "
                "to read every stream of its kind the SELECT carries, or drop "
                "the extra columns",
            )
        return pads

    def _lower_variadic_call(
        self, node: exp.Expr, name: str, call: _Call, env: _Env, select: exp.Select
    ) -> _Value:
        """Dispatch a call carrying ``VARIADIC``: the pad count follows the array.

        Only an N-input filter (``DynamicFilter.n_input``) and ``concat`` have
        a pad count that can follow anything -- every other filter's arity is
        fixed by its pad signature, so ``VARIADIC`` on one of those is a
        rejection naming that, and an unknown name is the ordinary
        ``UNKNOWN_FUNCTION`` either way.
        """
        n_input = self._n_input_call(name)
        if n_input is not None:
            spec, options = n_input
            return self._lower_variadic_n_input_call(node, spec, options, call, env, select)
        concat = self._concat_options(name)
        if concat is not None:
            return self._lower_concat_call(node, concat, call, env, select)
        dynamic = self.registry.get(name) if self.registry is not None else None
        array_returning = call.namespaced and self._array_options(name) is not None
        if dynamic is not None or array_returning:
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"{call.display}() takes a fixed number of streams: VARIADIC "
                "only spreads an array over a filter whose pad count follows it",
                node,
                fallback=select,
                hint=_VARIADIC_HINT,
            )
        raise _error(
            ErrorCode.UNKNOWN_FUNCTION,
            f"unknown function {call.display}()",
            node,
            fallback=select,
            hint=self._namespaced_function_hint(name)
            if call.namespaced
            else self._unknown_function_hint(name),
        )

    # -- the ffrwd macro namespace -----------------------------

    def _lower_macro_call(
        self, node: exp.Expr, name: str, call: _Call, env: _Env, select: exp.Select
    ) -> _Value:
        """Resolve ``ffrwd.<name>(...)`` against :data:`MACROS`, and nowhere
        else -- the registry is never consulted, so a macro compiles OFFLINE
        (``which() -> None``) exactly as well as it does against a live ffmpeg.

        A macro owns its OWN positional signature: there is no option table to
        bind against, so named arguments are rejected outright (UNSUPPORTED_SQL,
        the same shape-violation code resolve's own named-only/positional-only
        argument rules use) unless the macro declares its own closed option
        list, and arity/kind mismatches are UDF_ARG_TYPE naming the macro's
        signature -- mirroring the registry call's stream-signature message,
        but there is exactly one stream position (always index 0) to check, so
        no `_bind_options` machinery is involved.

        Broadcasting reuses :meth:`_expand_call` unchanged: it is type-driven
        off `positions`/`streams`, so a macro's single stream argument
        broadcasts elementwise exactly like any registry call's would.
        """
        input_macro = INPUT_MACROS.get(name)
        if input_macro is not None:
            # An input-minting macro: no filter node, no arguments,
            # one passthrough stream of the type it mints.
            if call.args or call.named:
                raise _error(
                    ErrorCode.UNSUPPORTED_SQL,
                    f"{call.display}() takes no arguments",
                    node,
                    fallback=select,
                    hint=f"write {call.display}()",
                )
            return _scalar(
                _Stream(
                    ref=self._mint_input(input_macro), type=input_macro.output
                )
            )
        macro = MACROS.get(name)
        if macro is None:
            raise _error(
                ErrorCode.UNKNOWN_FUNCTION,
                f"unknown function {call.display}()",
                node,
                fallback=select,
                hint=self._macro_function_hint(name),
            )
        if call.named and not macro.options:
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"{call.display}() is a ffrwd macro: its arguments are "
                "positional only, in the documented order",
                call.named[0].value,
                fallback=node,
                hint=f"its signature is {macro.signature}",
            )
        if macro.name == loudnorm.FILTER and self.table_mode:
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"{call.display}() is a filter, and a table query filters nothing",
                node,
                fallback=select,
                hint="print the tracks with a table query, normalize them with "
                "a COPY that writes a file",
            )
        if len(call.args) != len(macro.params):
            raise _error(
                ErrorCode.UDF_ARG_TYPE,
                f"{call.display}() takes {len(macro.params)} argument"
                f"{'' if len(macro.params) == 1 else 's'}, got {len(call.args)}",
                node,
                fallback=select,
                hint=f"its signature is {macro.signature}",
            )
        stream_pos = macro.stream_positions[0]
        stream_param = macro.params[stream_pos]
        self._reject_null_stream(call.display, call.args[stream_pos], select)
        kind = self._classify(call.args[stream_pos], env, select)
        self._reject_passthrough_args(call.display, [kind], call, call.args[stream_pos])
        if kind != stream_param.stream_type:
            hint = macro.kind_hints.get(
                kind,
                f"stream inputs come first, then options in the macro's own "
                f"order: {macro.signature}",
            )
            raise _error(
                ErrorCode.UDF_ARG_TYPE,
                f"{call.display}() takes a {stream_param.stream_type} stream as "
                f"its '{stream_param.name}' argument, got {kind}",
                call.args[stream_pos],
                fallback=node,
                hint=hint,
            )
        literals: dict[int, object] = {}
        for position, param in enumerate(macro.params):
            if param.kind != "num":
                continue
            arg = call.args[position]
            try:
                literals[position] = _number(arg)
            except FfrwdError as exc:
                raise _error(
                    exc.code,
                    f"{call.display}()'s '{param.name}' argument must be a "
                    "numeric literal",
                    arg,
                    fallback=node,
                    hint=f"its signature is {macro.signature}",
                ) from None
        options = self._macro_options(macro, call, node)
        streams = {stream_pos: self._lower_expr(call.args[stream_pos], env, select)}

        def build(values: list[object], _element: int) -> FrameRef:
            return macro.expand(values, self.ctx.node, options)

        return self._expand_call(
            call.display,
            node,
            call.args,
            select,
            streams=streams,
            literals=literals,
            arity=len(macro.params),
            positions=[stream_pos],
            returns=macro.output,
            build=build,
        )

    def _macro_options(
        self, macro: Macro, call: _Call, node: exp.Expr
    ) -> dict[str, object]:
        """A macro's named-only options: every one optional, none repeated.

        Returned in the MACRO's declared order, not the order they were
        written, so the rendered filter is the same whichever way round the
        query spells them. An omitted option is left out entirely -- the
        expansion renders only what was written, and ffmpeg's own default
        covers the rest. Repeats need no check here: resolve rejects a
        duplicate `name =>` on any call before lowering starts.
        """
        written: dict[str, object] = {}
        for argument in call.named:
            if argument.name not in macro.options:
                raise _error(
                    ErrorCode.UDF_ARG_TYPE,
                    f"{call.display}() has no '{argument.name}' option",
                    argument.value,
                    fallback=node,
                    hint=f"its signature is {macro.signature}",
                )
            try:
                written[argument.name] = _number(argument.value)
            except FfrwdError as exc:
                raise _error(
                    exc.code,
                    f"{call.display}()'s '{argument.name}' option must be a "
                    "numeric literal",
                    argument.value,
                    fallback=node,
                    hint=f"its signature is {macro.signature}",
                ) from None
        return {name: written[name] for name in macro.options if name in written}

    def _macro_function_hint(self, name: str) -> str:
        """Did-you-mean over :data:`MACROS`, the small-by-design macro set."""
        matches = difflib.get_close_matches(name, macro_names(), n=1, cutoff=0.6)
        if matches:
            return f"did you mean {MACRO_NAMESPACE}.{matches[0]}()?"
        return (
            f"{MACRO_NAMESPACE}.<name> is one of ffrwd's own macros -- "
            f"{', '.join(macro_names())} -- not an ffmpeg filter; filters live "
            f"bare or under {FILTER_NAMESPACE}.<filter>(...)"
        )

    # -- the ordinary case: any filter the installed ffmpeg reports --------

    def _dispatch_audio(
        self,
        name: str,
        dynamic: DynamicFilter,
        call: _Call,
        env: _Env,
        select: exp.Select,
    ) -> tuple[str, DynamicFilter]:
        """A bare video-only name dispatches to its audio twin over audio input.

        ``ffmpeg.<name>(...)`` is untouched -- only a bare call dispatches.
        Eligibility comes straight from the registry, not a curated list:
        ``name`` takes video-only input, ``a<name>`` exists and takes
        audio-only input. That excludes a pair that only shares a stem, like
        ``interleave``/``ainterleave`` or ``mix``/``amix`` (both N-input, so
        neither has a fixed pad type to compare).
        """
        if call.namespaced or self.registry is None:
            return name, dynamic
        if dynamic.n_input or not dynamic.inputs or any(k != "video" for k in dynamic.inputs):
            return name, dynamic
        twin_name = "a" + name
        twin = self.registry.get(twin_name)
        if twin is None or twin.n_input or not twin.inputs:
            return name, dynamic
        if any(k != "audio" for k in twin.inputs):
            return name, dynamic
        kinds = self._stream_kinds(call, env, select, len(dynamic.inputs))
        if kinds[:1] == ["audio"]:
            return twin_name, twin
        return name, dynamic

    def _lower_dynamic_call(
        self,
        node: exp.Expr,
        name: str,
        dynamic: DynamicFilter,
        call: _Call,
        env: _Env,
        select: exp.Select,
    ) -> _Value:
        """A call resolved from the registry: streams, then options.

        The pad signature IS the stream signature: ``gblur`` (``V->V``) takes
        exactly one video argument, ``xfade`` (``VV->V``) exactly two. Every
        positional argument after those binds to one of the filter's OPTIONS,
        in ffmpeg's own declared order (:meth:`_bind_options`).

        Reached by both spellings — a bare filter name and
        ``ffmpeg.<filter>(...)`` — which differ only in ``call.display``. The
        NODE always carries the filter's own name, so the IR, split and emit
        never learn that the namespace exists.
        """
        expected = list(dynamic.inputs)
        kinds = self._stream_kinds(call, env, select, len(expected))
        if kinds != expected:
            raise self._bad_streams(call, node, select, expected, kinds)
        args_at, per_row = self._option_binder(
            name,
            call,
            node,
            select,
            env,
            options=self._options_for(name, call, len(expected), node, select),
            extras=call.args[len(expected) :],
            timeline=dynamic.timeline,
        )
        streams = {
            position: self._lower_expr(arg, env, select)
            for position, arg in enumerate(call.args[: len(expected)])
        }
        output = dynamic.output

        def build(values: list[object], element: int) -> FrameRef:
            return self.ctx.node(
                name, dict(args_at(element)), [_as_ref(value) for value in values], [output]
            )

        return self._expand_call(
            call.display,
            node,
            call.args,
            select,
            streams=streams,
            literals={},
            arity=len(expected),
            positions=list(range(len(expected))),
            returns=output,
            build=build,
            rows=self._row_elements(per_row, env),
        )

    def _row_elements(self, per_row: bool, env: _Env) -> int | None:
        """How many rows a per-row call runs over, or None when it reads none.

        The relation as this branch left it: a grouped branch has already been
        cut to the group being gathered, and a fan-out to the one row this
        command writes -- so both keep making the one node they always made.
        """
        if not per_row or env.relation is None:
            return None
        return len(env.relation.tuples)

    # -- N-input filters ----------------------------

    def _n_input_call(self, name: str) -> tuple[_NInputFilter, dict[str, FilterOption]] | None:
        """`name`'s derived call shape and option table, if THIS registry has
        it as a callable N-input filter (``DynamicFilter.n_input``).

        An n-input filter is an ordinary registry member now (see
        registry.py), so this is a membership check plus the same option
        fetch every other callable filter goes through -- options are
        fetched even for a call that passes none, since the spec's
        `option`/`fallback` are derived from the table's own content
        (:func:`_n_input_spec`). ``acrossfade`` is the case this matters for:
        on a build where it is still an ordinary ``AA->A`` filter,
        ``dynamic.n_input`` is False and this returns None, so the registry's
        own pad signature wins over any N-input treatment.
        """
        if self.registry is None:
            return None
        dynamic = self.registry.get(name)
        if dynamic is None or not dynamic.n_input:
            return None
        options = self.registry.options(name)
        if options is None:
            return None
        return _n_input_spec(name, dynamic, options), options

    def _lower_n_input_call(
        self,
        node: exp.Expr,
        spec: _NInputFilter,
        options: dict[str, FilterOption],
        call: _Call,
        env: _Env,
        select: exp.Select,
    ) -> _Value:
        """One node with N input pads, N being what the count option says.

        The stream/option split cannot come from a fixed pad signature here
        (there is none — the pad count is dynamic, which is exactly what
        marks this filter n-input), so it comes from the arguments
        themselves: the LEADING RUN of
        stream-valued arguments are the input pads, and everything after them
        is an option. That is unambiguous because an option value is always a
        literal and a pad is never one.

        The count option is then read back and must AGREE with how many
        streams were supplied — `amix(a, b)` (2 streams, `inputs` defaulted to
        2) and `amix(a, b, c, inputs => 3)` are both consistent;
        `amix(a, b, c)` is not, and says so with both numbers.
        """
        kinds = [self._classify(arg, env, select) for arg in call.args]
        self._reject_passthrough_args(call.display, kinds, call, node)
        count = 0
        for kind in kinds:
            if kind not in _STREAM_KINDS:
                break
            count += 1
        supplied = kinds[:count]
        if not supplied or any(kind != spec.stream for kind in supplied):
            raise _error(
                ErrorCode.UDF_ARG_TYPE,
                f"{call.display}() is an ffmpeg filter: its stream inputs are all "
                f"{spec.stream}, got ({', '.join(supplied) or 'no streams'})",
                node,
                fallback=select,
                hint=_N_INPUT_HINT,
            )
        args = self._bind_options(
            spec.name,
            call,
            node,
            select,
            env,
            options=options,
            extras=call.args[count:],
            timeline=False,
        )
        # A filter with no count option (ladspa) has nothing to cross-check the
        # supplied stream count against and nothing to write back -- the
        # streams themselves ARE the count, decided by the loaded plugin.
        option_name = spec.option
        if option_name is not None:
            declared = self._n_input_count(spec, option_name, args, options)
            if declared != count:
                anchor = next(
                    (arg.value for arg in call.named if arg.name == option_name), node
                )
                raise _error(
                    ErrorCode.UDF_ARG_TYPE,
                    f"{call.display}() was given {_stream_count(count)} but its "
                    f"'{option_name}' option says {declared}",
                    anchor,
                    fallback=select,
                    hint=_N_INPUT_HINT,
                )
            # Write the count onto the node unless this spec omits a defaulted
            # one (`emit_default`): ffmpeg only NEEDS `inputs=N` to grow pads
            # beyond the option's default of 2, and for a filter that is
            # variadic only on newer builds the omitted default is what keeps
            # the command portable.
            if spec.emit_default or option_name in args or count != spec.fallback:
                args[option_name] = count
        streams = {
            position: self._lower_expr(arg, env, select)
            for position, arg in enumerate(call.args[:count])
        }

        def build(values: list[object], _element: int) -> FrameRef:
            return self.ctx.node(
                spec.name, dict(args), [_as_ref(value) for value in values], [spec.output]
            )

        return self._expand_call(
            call.display,
            node,
            call.args,
            select,
            streams=streams,
            literals={},
            arity=count,
            positions=list(range(count)),
            returns=spec.output,
            build=build,
        )

    def _n_input_count(
        self,
        spec: _NInputFilter,
        option_name: str,
        args: dict[str, object],
        options: dict[str, FilterOption],
    ) -> int:
        """What the count option says, written or introspected-default or fallback.

        `args` has already been validated against the option table, so a
        written value is a number in range; only the DEFAULT needs care, since
        `FilterOption.default` is verbatim ffmpeg text that is documented as
        never re-typed (it can be a constant name, or absent entirely). Called
        only when `spec.option` is not None; `option_name` is that narrowed
        value, passed separately so mypy sees a plain `str`.
        """
        written = args.get(option_name)
        if isinstance(written, (int, float)) and not isinstance(written, bool):
            return int(written)
        option = options.get(option_name)
        if option is not None and option.default is not None:
            try:
                return int(float(option.default))
            except ValueError:
                pass
        return spec.fallback

    # -- VARIADIC: an array IS the argument list --------------------------

    def _variadic_array(
        self, call: _Call, node: exp.Expr, env: _Env, select: exp.Select
    ) -> _Value:
        """The array a call's ``VARIADIC`` argument lowers to, validated.

        Every VARIADIC caller wants the same three checks: an array (a bare
        array is broadcast, never spread -- see the module's own rules), a
        non-empty one (a filter call with no inputs is not a filter call, and
        this is the one place that says so), and the streams themselves,
        already lowered.
        """
        variadic = call.variadic
        assert variadic is not None  # callers only reach here when it is
        value = self._lower_expr(variadic, env, select)
        if not value.is_array:
            raise _error(
                ErrorCode.UDF_ARG_TYPE,
                f"VARIADIC needs an array: {_sql_text(variadic)} is a single "
                f"{value.type} stream",
                variadic,
                fallback=node,
                hint="drop VARIADIC to pass it as one ordinary stream argument",
            )
        if not value.streams:
            raise _error(
                ErrorCode.UDF_ARG_TYPE,
                f"{call.display}() has no inputs: {_sql_text(variadic)} is empty",
                variadic,
                fallback=node,
                hint="VARIADIC spreads the array as the call's argument list; "
                "an empty array leaves the filter nothing to run on",
            )
        return value

    def _lower_variadic_n_input_call(
        self,
        node: exp.Expr,
        spec: _NInputFilter,
        options: dict[str, FilterOption],
        call: _Call,
        env: _Env,
        select: exp.Select,
    ) -> _Value:
        """``spec.name(<streams...>, VARIADIC <array>)``: the array supplies
        every pad past the positional streams.

        Positional streams ahead of VARIADIC bind exactly as they do without
        it; the array's elements follow them in argument order, so
        ``concat(intro, VARIADIC array_agg(v))`` feeds ``intro`` then every
        element of the aggregate. No positional OPTION can follow a variable
        number of streams, so every option here is named -- unlike the
        leading-run count guess :meth:`_lower_n_input_call` makes, the array's
        length already IS the count, checked against a written ``option =>``
        the same way a plain call's count is.
        """
        kinds = self._stream_kinds(call, env, select, len(call.args))
        if any(kind != spec.stream for kind in kinds):
            raise _error(
                ErrorCode.UDF_ARG_TYPE,
                f"{call.display}() is an ffmpeg filter: its stream inputs are "
                f"all {spec.stream}, got ({', '.join(kinds) or 'no streams'})",
                node,
                fallback=select,
                hint=_N_INPUT_HINT,
            )
        array_value = self._variadic_array(call, node, env, select)
        if array_value.type != spec.stream:
            raise _error(
                ErrorCode.UDF_ARG_TYPE,
                f"{call.display}() is an ffmpeg filter: its stream inputs are "
                f"all {spec.stream}, got VARIADIC {array_value.type}",
                call.variadic if call.variadic is not None else node,
                fallback=node,
                hint=_N_INPUT_HINT,
            )
        prefix = [self._lower_expr(arg, env, select).at(0) for arg in call.args]
        streams = prefix + list(array_value.streams)
        count = len(streams)
        args = self._bind_options(
            spec.name, call, node, select, env, options=options, extras=[], timeline=False,
        )
        self._check_variadic_count(spec.option, count, args, call, node, select, _N_INPUT_HINT)
        if spec.option is not None and (
            spec.emit_default or spec.option in args or count != spec.fallback
        ):
            args[spec.option] = count
        node_id = self.ctx.node(
            spec.name, args, [stream.ref for stream in streams], [spec.output]
        )
        source = streams[0].source if count == 1 else _agreed_source(streams)
        return _scalar(_Stream(ref=node_id, type=spec.output, source=source))

    def _check_variadic_count(
        self,
        option_name: str | None,
        count: int,
        args: dict[str, object],
        call: _Call,
        node: exp.Expr,
        select: exp.Select,
        hint: str,
    ) -> None:
        """A WRITTEN count option must agree with the array's length.

        Unlike the positional call (:meth:`_lower_n_input_call`), there is no
        "did you forget to write it" ambiguity here: the array's length IS the
        count, full stop, so an unwritten option is simply set to it below --
        only a value the query itself wrote can possibly disagree.
        """
        if option_name is None:
            return
        written = args.get(option_name)
        if not isinstance(written, (int, float)) or isinstance(written, bool):
            return
        if int(written) == count:
            return
        anchor = next((arg.value for arg in call.named if arg.name == option_name), node)
        raise _error(
            ErrorCode.UDF_ARG_TYPE,
            f"{call.display}() was given {_stream_count(count)} but its "
            f"'{option_name}' option says {int(written)}",
            anchor,
            fallback=select,
            hint=hint,
        )

    # -- VARIADIC concat: N segments of one stream type --------------------

    def _concat_options(self, name: str) -> dict[str, FilterOption] | None:
        """``concat``'s option table, but ONLY for a call under VARIADIC.

        Mirrors :meth:`_n_input_options`: ``concat`` is ``N->N`` and excluded
        from the registry's own table by the pad-scope check (see
        registry.py), and ``excluded_options`` is the one door back in --
        also the evidence that this ffmpeg actually ships the filter at all.
        """
        if name != _CONCAT_NAME or self.registry is None:
            return None
        return self.registry.excluded_options(name)

    def _lower_concat_call(
        self,
        node: exp.Expr,
        options: dict[str, FilterOption],
        call: _Call,
        env: _Env,
        select: exp.Select,
    ) -> _Value:
        """``concat(<streams...>, VARIADIC <array>)``: one segment per stream.

        ffmpeg's ``concat`` multiplexes video AND audio pads per segment, set
        by its own ``v``/``a`` options; VARIADIC only ever spreads ONE
        homogeneous array, so this is ``concat`` run as a plain N-input
        filter over whichever type the array carries. ``v``/``a`` follow that
        type unconditionally -- writing either is rejected, the same
        ``UNKNOWN_FILTER_OPTION`` a made-up option name gets, since a call
        with one array has nothing for a split segment shape to mean. ``n``
        stays an ordinary count option: written or not, it is checked against
        the array's length exactly as an N-input filter's is.
        """
        array_value = self._variadic_array(call, node, env, select)
        stream_type = array_value.type
        kinds = self._stream_kinds(call, env, select, len(call.args))
        if any(kind != stream_type for kind in kinds):
            raise _error(
                ErrorCode.UDF_ARG_TYPE,
                f"{call.display}() is an ffmpeg filter: its stream inputs are "
                f"all one type, got ({', '.join([*kinds, stream_type])})",
                node,
                fallback=select,
                hint=_CONCAT_VARIADIC_HINT,
            )
        prefix = [self._lower_expr(arg, env, select).at(0) for arg in call.args]
        streams = prefix + list(array_value.streams)
        count = len(streams)
        bindable = {key: option for key, option in options.items() if key not in ("v", "a")}
        args = self._bind_options(
            "concat", call, node, select, env, options=bindable, extras=[], timeline=False,
        )
        self._check_variadic_count("n", count, args, call, node, select, _CONCAT_VARIADIC_HINT)
        args["n"] = count
        args["v"] = 1 if stream_type == "video" else 0
        args["a"] = 1 if stream_type == "audio" else 0
        node_id = self.ctx.node("concat", args, [stream.ref for stream in streams], [stream_type])
        source = streams[0].source if count == 1 else _agreed_source(streams)
        return _scalar(_Stream(ref=node_id, type=stream_type, source=source))

    # -- array-returning filters -----------------------

    def _array_options(self, name: str) -> dict[str, FilterOption] | None:
        """`name`'s option table if it is a callable array-returning filter.

        Three questions, one answer, because they have the same shape: is the
        name in :data:`ARRAY_RETURNING`, is there a registry at all, and does
        THIS ffmpeg actually have the filter. The last one is why the
        options are fetched even for a call with no named arguments: an excluded
        name is in no registry table, so its option block is the only evidence
        this build has it (see ``Registry.excluded_options``). None means "not
        callable", and the caller falls through to the ordinary namespaced
        rejection, hint and all.
        """
        if name not in ARRAY_RETURNING or self.registry is None:
            return None
        return self.registry.excluded_options(name)

    def _lower_array_call(
        self,
        node: exp.Expr,
        spec: _ArrayFilter,
        options: dict[str, FilterOption],
        call: _Call,
        env: _Env,
        select: exp.Select,
    ) -> _Value:
        """One node with N output pads, returned as an N-element array value.

        The pad COUNT comes from the table's count rule, run over the validated
        named arguments — so the option's own type, range and constant checks
        have already happened, and a value that is well-typed but not a count
        this filter could produce (``channel_layout => 'nonsense'``) is the
        rule's own ``FILTER_OPTION_TYPE``, anchored on that argument.

        Provenance is a 1:N fan: the single input stream's source is threaded
        to every element (not ``_agreed_source``, which answers the opposite
        question), so splitting a ``language=eng`` track gives N eng channels.
        """
        expected = [spec.input]
        kinds = self._stream_kinds(call, env, select, 1)
        if kinds != expected:
            raise self._bad_streams(call, node, select, expected, kinds)
        args = self._bind_options(
            spec.name,
            call,
            node,
            select,
            env,
            options=options,
            extras=call.args[1:],
            timeline=False,
        )
        count = spec.count(args)
        if isinstance(count, _BadCount):
            raise self._bad_count(spec, count, call, node, select)

        value = self._lower_expr(call.args[0], env, select)
        if value.is_array:
            raise _error(
                ErrorCode.UDF_ARG_TYPE,
                f"{call.display}() returns an array, so it cannot also broadcast "
                f"over one: {_sql_text(call.args[0])} is "
                f"{_stream_count(len(value.streams))}",
                call.args[0],
                fallback=node,
                hint=_ARRAY_INPUT_HINT,
            )
        stream = value.streams[0]
        node_id = self.ctx.node(
            spec.name, dict(args), [stream.ref], [spec.element] * count
        )
        return _array(
            spec.element,
            [
                _Stream(ref=f"{node_id}:{pad}", type=spec.element, source=stream.source)
                for pad in range(count)
            ],
        )

    def _bad_count(
        self,
        spec: _ArrayFilter,
        bad: _BadCount,
        call: _Call,
        node: exp.Expr,
        select: exp.Select,
    ) -> FfrwdError:
        """A count rule's rejection, anchored on the argument that caused it.

        The offending option is normally one the query wrote, and that is the
        token worth pointing at; a rule can only reject a DEFAULT if the table
        itself is wrong, so falling back to the call keeps that case anchored
        rather than unanchored.
        """
        written = next((arg for arg in call.named if arg.name == bad.option), None)
        anchor = written.value if written is not None else node
        return _error(
            ErrorCode.FILTER_OPTION_TYPE,
            f"option '{bad.option}' of filter '{spec.name}' decides how many "
            f"streams the call returns, so it must be {bad.expected}, "
            f"got {bad.value!r}",
            anchor,
            fallback=select,
            hint=bad.hint,
        )

    # -- shared call machinery --------------------------------------------

    def _reject_passthrough_args(
        self,
        name: str,
        kinds: list[str],
        call: _Call,
        node: exp.Expr,
    ) -> None:
        """No function takes a subtitle or data stream.

        An ffmpeg filtergraph carries video and audio only, so a caption or
        timed-metadata stream can never be a filter INPUT — in either tier.
        Tier 1 would otherwise report it as a generic signature mismatch and
        tier 2 as "expects gblur(video)"; both are true but neither says the
        thing that actually matters, which is that no signature could ever
        accept it. ``ParamKind`` and ``DynamicFilter.inputs`` are deliberately
        left alone ("ParamKind is UNCHANGED"), so this is the one
        place that knows it.
        """
        for position, kind in enumerate(kinds):
            if kind not in _PASSTHROUGH_ONLY:
                continue
            raise _error(
                ErrorCode.UDF_ARG_TYPE,
                f"{name}() cannot take a {kind} stream: {kind} streams cannot be "
                "filtered, only selected",
                call.args[position],
                fallback=node,
                hint=_PASSTHROUGH_HINT,
            )

    def _stream_kinds(
        self, call: _Call, env: _Env, select: exp.Select, arity: int
    ) -> list[str]:
        """Kind labels for the call's LEADING `arity` arguments, checked for captions.

        Only the leading run is classified: everything after it is an option
        value, which is a literal that the OPTION table judges, not the
        classifier. A short call classifies what it has, so the caller's
        comparison against the pad signature reports the missing argument.
        """
        for arg in call.args[:arity]:
            self._reject_null_stream(call.display, arg, select)
        kinds = [self._classify(arg, env, select) for arg in call.args[:arity]]
        if kinds:
            self._reject_passthrough_args(call.display, kinds, call, call.args[0])
        return kinds

    def _bad_streams(
        self,
        call: _Call,
        node: exp.Expr,
        select: exp.Select,
        expected: list[StreamType],
        got: list[str],
    ) -> FfrwdError:
        """The stream-signature rejection — UDF_ARG_TYPE's remaining job.

        Option problems never reach here: they are ``UNKNOWN_FILTER_OPTION`` /
        ``FILTER_OPTION_TYPE`` uniformly, positional or named.
        """
        shown = call.display
        return _error(
            ErrorCode.UDF_ARG_TYPE,
            f"{shown}() is an ffmpeg filter: it takes {', '.join(expected)} as its "
            f"stream input{'' if len(expected) == 1 else 's'}, "
            f"got ({', '.join(got) or 'nothing'})",
            node,
            fallback=select,
            hint=f"stream inputs come first, then options in the filter's own order, "
            f"then named options: {shown}({', '.join(expected)}, <option>, "
            f"<option> => <value>)",
        )

    def _options_for(
        self,
        filter_name: str,
        call: _Call,
        stream_arity: int,
        node: exp.Expr,
        select: exp.Select,
    ) -> dict[str, FilterOption]:
        """The filter's option table, fetched only when the call actually needs it.

        ``-help filter=X`` is a subprocess, and a call that passes no options
        at all (``hflip(a.video[1])``) has nothing to validate — so the table stays
        unfetched, exactly as it did before positional options existed.
        """
        if len(call.args) <= stream_arity and not call.named:
            return {}
        return self._filter_options(filter_name, node, select)

    def _reject_stream_option(
        self,
        filter_name: str,
        option: FilterOption,
        arg: exp.Expr,
        node: exp.Expr,
        env: _Env,
        select: exp.Select,
    ) -> None:
        """A stream where an option value belongs, said plainly.

        Classifying first is also what keeps a TYPO in a nested call readable:
        `gblur(a.video[1], nope(a.video[1]))` is UNKNOWN_FUNCTION for `nope`, raised
        by the classifier, rather than a puzzled complaint about `sigma`'s type.
        Only stream-SHAPED arguments are classified -- a literal is the option
        validator's business and is left to it.
        """
        inner = _unwrap(arg)
        if not isinstance(inner, exp.Bracket | exp.Column) and _call_parts(inner) is None:
            return
        kind = self._classify(arg, env, select)
        if kind not in _STREAM_KINDS:
            return
        raise _error(
            ErrorCode.FILTER_OPTION_TYPE,
            f"option '{option.name}' of filter '{filter_name}' takes a value, "
            f"got a {kind} stream",
            arg,
            fallback=node,
            hint="stream inputs come first and are counted by the filter's pad "
            "signature; everything after them is an option value",
        )

    def _option_binder(
        self,
        filter_name: str,
        call: _Call,
        node: exp.Expr,
        select: exp.Select,
        env: _Env,
        *,
        options: dict[str, FilterOption],
        extras: list[exp.Expr],
        timeline: bool,
    ) -> tuple[Callable[[int], dict[str, object]], bool]:
        """This call's option dict as a function of the element, and whether
        that function actually reads the row.

        An option written as a compile-time expression
        (``scale(t, t.width / 2, -2)``) is evaluated against the row that
        element came from and REPLACED BY THE LITERAL it computes to, so a
        per-row option and a written one bind through the same
        :meth:`_bind_options` and are validated by the same option table.
        A bare column bound to a non-boolean option counts too --
        ``ffmpeg.trim(f, starti => f.duration)``, ``scale(f.video[1], r.w,
        -2)`` -- because a probed/row scalar may be exactly what the option
        wants (the option table still rejects it once evaluated if it is
        not); a bare column bound to a boolean option, or to a stream
        argument, stays untouched, since it may be a stream.

        A call with no computed option binds exactly once and hands the same
        dict to every element -- which is every call that existed before
        arithmetic did.
        """
        order = list(options)

        def positional_target(index: int) -> FilterOption | None:
            return options[order[index]] if index < len(order) else None

        def countable(arg: exp.Expr, option: FilterOption | None) -> bool:
            if is_value_expr(arg):
                return True
            return (
                option is not None
                and option.type != "bool"
                and _is_row_scalar(arg, env)
            )

        extras_countable = [
            countable(arg, positional_target(i)) for i, arg in enumerate(extras)
        ]
        named_countable = [
            countable(arg.value, options.get(arg.name)) for arg in call.named
        ]
        if not any(extras_countable) and not any(named_countable):
            args = self._bind_options(
                filter_name, call, node, select, env,
                options=options, extras=extras, timeline=timeline,
            )
            return (lambda _element: args), False
        per_row = any(
            _reads_row_column(arg, env)
            for arg, countable_here in [
                *zip(extras, extras_countable, strict=True),
                *[(a.value, c) for a, c in zip(call.named, named_countable, strict=True)],
            ]
            if countable_here
        )
        tuples = env.relation.tuples if env.relation is not None else []
        cache: dict[int, dict[str, object]] = {}

        def bound(element: int) -> dict[str, object]:
            if element not in cache:
                row = tuples[element] if element < len(tuples) else {}
                cache[element] = self._bind_options(
                    filter_name,
                    replace(
                        call,
                        named=[
                            _NamedArg(
                                arg.name,
                                self._computed_arg(arg.value, env, row, select, evaluate=eval_it),
                            )
                            for arg, eval_it in zip(call.named, named_countable, strict=True)
                        ],
                    ),
                    node,
                    select,
                    env,
                    options=options,
                    extras=[
                        self._computed_arg(arg, env, row, select, evaluate=eval_it)
                        for arg, eval_it in zip(extras, extras_countable, strict=True)
                    ],
                    timeline=timeline,
                )
            return cache[element]

        return bound, per_row

    def _computed_arg(
        self,
        node: exp.Expr,
        env: _Env,
        row: _RowTuple,
        select: exp.Select,
        *,
        evaluate: bool,
    ) -> exp.Expr:
        """One option argument as `row` makes it; anything else, untouched."""
        if not evaluate:
            return node
        return _literal_node(self._eval_value(node, env, row, select), node)

    def _bind_options(
        self,
        filter_name: str,
        call: _Call,
        node: exp.Expr,
        select: exp.Select,
        env: _Env,
        *,
        options: dict[str, FilterOption],
        extras: list[exp.Expr],
        timeline: bool,
    ) -> dict[str, object]:
        """Positional options first, then named ones — one merged arg dict.

        `extras` is every positional argument past the stream inputs. Each
        binds to the option at its own index in ``options``, whose insertion
        order IS ffmpeg's AVOption declaration order and therefore its own
        positional binding order (verified against ffmpeg 7.1 for the whole
        registry; see ``ffrwd/registry.py``). Having landed on an option, a
        positional is validated AS that option by the very same
        :func:`_option_value` a named argument goes through, which is what
        makes option errors uniform across the two spellings.

        Named arguments are then checked with the positionally-bound names
        marked `occupied`, so a named argument never silently overrides one the
        call already set.
        """
        order = list(options)
        if len(extras) > len(order):
            raise _error(
                ErrorCode.UDF_ARG_TYPE,
                f"{call.display}() got {len(extras)} positional option"
                f"{'' if len(extras) == 1 else 's'}, but the '{filter_name}' filter "
                f"has {len(order)}",
                extras[len(order)] if len(order) < len(extras) else node,
                fallback=select,
                hint="its options, in the order they bind: " + _listed(order)
                if order
                else f"the '{filter_name}' filter has no options ffrwd can set",
            )
        bound: dict[str, object] = {}
        dropped: dict[str, exp.Expr] = {}
        for index, arg in enumerate(extras):
            option = options[order[index]]
            self._reject_stream_option(filter_name, option, arg, node, env, select)
            if isinstance(_unwrap(arg), exp.Null):
                # NULL is absence: the option is not written and ffmpeg's own
                # default applies. The position stays occupied, so later
                # positionals keep their slots and a named repeat still
                # collides. `_option_value` never sees a NULL.
                dropped[option.name] = arg
                continue
            bound[option.name] = _option_value(
                filter_name, option, _NamedArg(name=option.name, value=arg), node
            )
        bound.update(
            self._check_named_args(
                filter_name,
                options,
                call.named,
                node,
                owner=call.display,
                occupied=set(bound) | set(dropped),
                timeline=timeline,
                dropped=dropped,
            )
        )
        self._check_required_options(filter_name, bound, dropped, node, select)
        return bound

    def _expand_call(
        self,
        name: str,
        node: exp.Expr,
        arg_nodes: list[exp.Expr],
        select: exp.Select,
        *,
        streams: dict[int, _Value],
        literals: dict[int, object],
        arity: int,
        positions: list[int],
        returns: StreamType,
        build: Callable[[list[object], int], FrameRef],
        rows: int | None = None,
    ) -> _Value:
        """Broadcast `build` over the array arguments, if there are any.

        Type-driven and tier-agnostic: `positions` is where the stream
        arguments are (always the LEADING positions, from the pad signature or
        from an N-input call's own count) and `build` is what turns one
        element's argument values into a subgraph. `build` also gets the
        ELEMENT INDEX, which is what lets a filter option computed per row
        pick out the row that element came from.

        `rows` broadcasts over the RELATION rather than over an array: a call
        whose options are read per row is one node per row, even where every
        stream argument it takes is a single stream.
        """
        length = self._zip_length(name, node, arg_nodes, streams, select, rows)
        expanded: list[_Stream] = []
        for element in range(1 if length is None else length):
            values: list[object] = [
                streams[position].at(element).ref
                if position in streams
                else literals[position]
                for position in range(arity)
            ]
            # A single-stream-input function is 1:1, so its result inherits
            # that input's provenance unconditionally. A call over two or more
            # streams (amix, overlay, xfade) is a join like concat's: it
            # threads provenance only when every input agrees
            # (`_agreed_source`).
            if len(positions) == 1:
                source = streams[positions[0]].at(element).source
            elif len(positions) >= 2:
                source = _agreed_source([streams[p].at(element) for p in positions])
            else:
                source = None
            expanded.append(_Stream(ref=build(values, element), type=returns, source=source))
        if length is None:
            return _scalar(expanded[0])
        return _array(returns, expanded)

    # -- named argument validation --

    def _filter_options(
        self, filter_name: str, anchor: exp.Expr, fallback: exp.Expr
    ) -> dict[str, FilterOption]:
        """The introspected options of `filter_name`, or a typed rejection.

        One rule: options ARE the installed ffmpeg. Without a registry there is
        nothing to validate them against, and guessing is exactly what this
        compiler does not do. (A CALL cannot reach this with a None registry —
        its name would already be UNKNOWN_FUNCTION — but a generated source in
        FROM position can, so the branch stays.)

        ``Registry.options`` returns None only for a filter this ffmpeg does not
        have (or that the v1 scope check excluded); an empty dict is a real
        answer (a filter with no options) and is passed through as one.
        """
        if self.registry is None:
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                "options are validated against your installed ffmpeg; "
                "the provisioner failed to supply one",
                anchor,
                fallback=fallback,
                hint=_NO_REGISTRY_HINT,
            )
        options = self.registry.options(filter_name)
        if options is None:
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"options are validated against the ffmpeg filter "
                f"'{filter_name}', which your ffmpeg does not provide",
                anchor,
                fallback=fallback,
                hint="drop the options, or install an ffmpeg that has "
                f"the '{filter_name}' filter",
            )
        return options

    def _check_named_args(
        self,
        filter_name: str,
        options: dict[str, FilterOption],
        named: list[_NamedArg],
        call: exp.Expr,
        *,
        owner: str,
        occupied: set[str],
        timeline: bool = False,
        dropped: dict[str, exp.Expr] | None = None,
    ) -> dict[str, object]:
        """Validate every named argument against `options`, in written order.

        A NULL value -- an unset variable's, or a literal one -- means the
        option is not written: it is recorded in `dropped` (when the caller
        passes one) and never reaches `_option_value`. An UNKNOWN name still
        rejects whatever its value, NULL included: the name is wrong before
        the value matters.

        `occupied` holds the option names this call already bound
        POSITIONALLY, so ``crop(f, 100, 50, 10, 20, out_w => 5)`` reads as the
        conflict it is rather than silently overriding what the call itself
        said. A collision is ``FILTER_OPTION_TYPE``, an option problem like any
        other, and the fix is to drop one of the two spellings.

        The collision check comes FIRST so the message names the conflict
        rather than whatever the registry would say about the name.

        `timeline` is the target's ``DynamicFilter.timeline`` flag, and it is a
        PARAMETER because this method cannot look filters up: every caller
        already holds the registry entry (or, for a generated source, knows
        there is no such field to hold — a source is never timeline-capable, so
        the default rejects). It admits ``enable`` BEFORE `options` is consulted
: ffmpeg implements ``enable`` in the filter framework, so
        it is in no filter's option table and a registry lookup would always
        call it unknown.
        """
        checked: dict[str, object] = {}
        for arg in named:
            if arg.name in occupied:
                raise _error(
                    ErrorCode.FILTER_OPTION_TYPE,
                    f"option '{arg.name}' of filter '{filter_name}' is already set "
                    f"positionally by {owner}()",
                    arg.value,
                    fallback=call,
                    hint="a named argument never overrides what the call itself "
                    "set; drop one of the two spellings",
                )
            is_null = isinstance(_unwrap(arg.value), exp.Null)
            if arg.name == _ENABLE:
                if is_null:
                    if dropped is not None:
                        dropped[_ENABLE] = arg.value
                    continue
                checked[_ENABLE] = _enable_value(filter_name, arg, call, timeline)
                continue
            option = options.get(arg.name)
            if option is None:
                raise _error(
                    ErrorCode.UNKNOWN_FILTER_OPTION,
                    f"filter '{filter_name}' has no option '{arg.name}'",
                    arg.value,
                    fallback=call,
                    hint=_option_hint(arg.name, options),
                )
            if is_null:
                if dropped is not None:
                    dropped[arg.name] = arg.value
                continue
            checked[arg.name] = _option_value(filter_name, option, arg, call)
        return checked

    def _check_required_options(
        self,
        filter_name: str,
        bound: dict[str, object],
        dropped: dict[str, exp.Expr],
        node: exp.Expr,
        select: exp.Select,
    ) -> None:
        """The curated :data:`REQUIRED_OPTIONS` check, on what was WRITTEN.

        A NULL dropped the option before this runs, so an unset variable and
        an omitted option fail the same way -- ffmpeg's init() would refuse
        both at run time, and this says so at compile time, naming the
        variable when the NULL came from one.
        """
        required: list[tuple[tuple[str, ...], str]] = [
            (group, "") for group in REQUIRED_OPTIONS.get(filter_name, ())
        ]
        if filter_name == "xfade" and bound.get("transition") == "custom":
            required.append((("expr",), " when transition is 'custom'"))
        for group, because in required:
            if any(option in bound for option in group):
                continue
            option_name = next((o for o in group if o in dropped), None)
            if option_name is not None:
                anchor = dropped[option_name]
                variable = null_variable(_unwrap(anchor))
                if variable is not None:
                    line, col = _pos(anchor, node)
                    raise unset_error(
                        ErrorCode.FILTER_OPTION_TYPE,
                        variable,
                        what=f"option '{option_name}' of filter "
                        f"'{filter_name}' is required{because}",
                        line=line,
                        col=col,
                    )
                raise _error(
                    ErrorCode.FILTER_OPTION_TYPE,
                    f"option '{option_name}' of filter '{filter_name}' is "
                    f"required{because}, got NULL",
                    anchor,
                    fallback=node,
                    hint="NULL is absence, and this filter cannot run "
                    "without the option; write a value",
                )
            spelled = " or ".join(f"'{option}'" for option in group)
            raise _error(
                ErrorCode.FILTER_OPTION_TYPE,
                f"filter '{filter_name}' requires option {spelled}{because}",
                node,
                fallback=select,
                hint=f"ffmpeg would refuse the filter at run time; write "
                f"{group[0]} => <value>",
            )

    def _reject_null_stream(
        self, display: str, arg: exp.Expr, select: exp.Select
    ) -> None:
        """A NULL where a stream input belongs: absence has no stream to offer."""
        inner = _unwrap(arg)
        if not isinstance(inner, exp.Null):
            return
        variable = null_variable(inner)
        if variable is not None:
            line, col = _pos(inner, select)
            raise unset_error(
                ErrorCode.UDF_ARG_TYPE,
                variable,
                what=f"{display}() needs a stream in this position",
                line=line,
                col=col,
            )
        raise _error(
            ErrorCode.UDF_ARG_TYPE,
            f"{display}() takes a stream in this position, got NULL",
            inner,
            fallback=select,
            hint="a stream input cannot be absent; pass one, e.g. f.video[1]",
        )

    def _unknown_function_hint(self, name: str) -> str:
        """Did-you-mean over the registry (there is nothing else)."""
        registry = self.registry
        if registry is not None and registry.available():
            if name == _CONCAT_NAME:
                return _CONCAT_VARIADIC_HINT
            if registry.get_source(name) is not None:
                return (
                    f"{name} is a generated source, not a function: put it in FROM, "
                    f"e.g. FROM {FILTER_NAMESPACE}.{name}(duration => 2) s"
                )
            # An n-input filter (amix, hstack, xstack, ...) is already in
            # `registry.names()` -- an ordinary registry member now -- so
            # only `concat` (excluded on the OUTPUT side) needs adding by hand.
            candidates = sorted((set(registry.names()) | {_CONCAT_NAME}) - {name})
            matches = difflib.get_close_matches(name, candidates, n=1, cutoff=0.6)
            if matches:
                return f"did you mean {matches[0]}()?"
            return (
                "every function is a filter of your installed ffmpeg, and this is "
                "not one of them; filters with a variable pad count, more than one "
                "output, or no input at all are not callable"
            )
        return _NO_REGISTRY_HINT

    def _namespaced_function_hint(self, name: str) -> str:
        """Did-you-mean for ``ffmpeg.<filter>()``, keeping the namespace spelling.

        Suggestions keep the ``ffmpeg.`` prefix, which is the one spelling that
        works for every filter name whatever Postgres thinks of it.
        """
        registry = self.registry
        if registry is not None and registry.available():
            if name == _CONCAT_NAME:
                return _CONCAT_VARIADIC_HINT
            if registry.get_source(name) is not None:
                # A generated source IS usable -- in FROM, where it belongs
                #. Say where rather than "unknown".
                return (
                    f"{FILTER_NAMESPACE}.{name} is a generated source, not a "
                    f"function: put it in FROM, e.g. FROM {FILTER_NAMESPACE}."
                    f"{name}(duration => 2) s"
                )
            candidates = sorted(
                (set(registry.names()) | set(ARRAY_RETURNING) | {_CONCAT_NAME}) - {name}
            )
            matches = difflib.get_close_matches(name, candidates, n=1, cutoff=0.6)
            if matches:
                return f"did you mean {FILTER_NAMESPACE}.{matches[0]}()?"
            return (
                f"{FILTER_NAMESPACE}.<filter> is a filter of your installed ffmpeg, "
                "and this is not one of them; filters with a variable pad count, "
                "more than one output, or no input at all are not callable"
            )
        return (
            f"the {FILTER_NAMESPACE}.<filter> namespace is your installed ffmpeg's "
            "filter set; the provisioner failed to supply one"
        )

    def _zip_length(
        self,
        name: str,
        node: exp.Expr,
        arg_nodes: list[exp.Expr],
        streams: dict[int, _Value],
        select: exp.Select,
        rows: int | None = None,
    ) -> int | None:
        """The element count this call expands to, or None if it expands to one.

        Arrays zip (no cross products): they must all have the same length, and
        scalar arguments repeat into every element.

        `rows` is how many rows an argument that READS one runs over -- a
        filter option computed per row makes one node per row, the way an
        array argument makes one per element. An array and a per-row option in
        the same call zip too: both count the same relation, so a disagreement
        is the same mismatch a pair of arrays of different lengths is.
        """
        first: tuple[int, int] | None = None  # (argument position, length)
        for position, value in sorted(streams.items()):
            if not value.is_array:
                continue
            length = len(value.streams)
            if first is None:
                first = (position, length)
                continue
            if length == first[1]:
                continue
            raise _error(
                ErrorCode.BROADCAST_MISMATCH,
                f"{name}() cannot broadcast over arrays of different lengths: "
                f"{_sql_text(arg_nodes[first[0]])} has {_stream_count(first[1])}, "
                f"{_sql_text(arg_nodes[position])} has {_stream_count(length)}",
                node,
                fallback=select,
                hint=_ZIP_HINT,
            )
        if first is None:
            return rows
        if rows is not None and rows != first[1]:
            raise _error(
                ErrorCode.BROADCAST_MISMATCH,
                f"{name}() cannot broadcast over arrays of different lengths: "
                f"{_sql_text(arg_nodes[first[0]])} has "
                f"{_stream_count(first[1])}, and its options are read once per "
                f"row over {rows} rows",
                node,
                fallback=select,
                hint=_ZIP_HINT,
            )
        return first[1]

    def _classify(self, node: exp.Expr, env: _Env, select: exp.Select) -> str:
        """Kind label for one call argument: a stream type, ``num``/``str``, or
        :data:`_UNSUPPORTED_KIND`.

        Stream arguments resolve to ``video``/``audio`` without creating any
        node, so a mismatch is reported before the graph grows. Nested calls
        to unknown functions are reported here rather than being labelled a
        stream and swallowed by an outer arity error, and a nested call
        resolves exactly the way a top-level one does, so
        ``scale(gblur(a.video[1], 2), 640, 480)`` sees the inner call's output
        pad type.
        """
        node = _unwrap(node)
        if isinstance(node, exp.Literal):
            return "str" if node.is_string else "num"
        if (
            isinstance(node, exp.Neg)
            and isinstance(node.this, exp.Literal)
            and not node.this.is_string
        ):
            return "num"
        # The stream half of a struct return is the stream the module wrote.
        streamed = _stream_projection(node, self.res.wasm)
        if streamed is not None:
            return self.res.wasm[str(streamed.name).lower()].stream_kind
        if isinstance(node, exp.Bracket | exp.Column):
            return self._base_stream(node, env, select)[1].type
        if isinstance(node, exp.Cast):
            return _UNSUPPORTED_KIND
        if isinstance(node, exp.Coalesce):
            # A filled track column is a stream of the row table's own type,
            # which is knowable without lowering anything (no fill is minted).
            return self._coalesce_parts(node, env, select)[0].type
        call = _call_parts(node)
        if call is not None:
            name = call.name.lower()
            if call.is_macro:
                if name in INPUT_MACROS:
                    return INPUT_MACROS[name].output
                macro = MACROS.get(name)
                if macro is None:
                    raise _error(
                        ErrorCode.UNKNOWN_FUNCTION,
                        f"unknown function {call.display}()",
                        node,
                        fallback=select,
                        hint=self._macro_function_hint(name),
                    )
                return macro.output
            # A module's output type is its declaration's, not the registry's:
            # the registry has never heard of it. A VALUE function has no pad
            # at all, and its call is refused where it is lowered.
            declared = None if call.namespaced else self.res.wasm.get(name)
            if declared is not None:
                return "video" if declared.is_value else declared.stream_kind
            # An array-returning call is classified by its ELEMENT type, which
            # is what makes it a legal argument: `volume(ffmpeg.channelsplit(
            # a.audio[1]), 0.5)` broadcasts over the channels.
            if call.namespaced and self._array_options(name) is not None:
                return ARRAY_RETURNING[name].element
            n_input = self._n_input_call(name)
            if n_input is not None:
                return n_input[0].output
            dynamic = self.registry.get(name) if self.registry is not None else None
            if dynamic is None:
                raise _error(
                    ErrorCode.UNKNOWN_FUNCTION,
                    f"unknown function {call.display}()",
                    node,
                    fallback=select,
                    hint=self._namespaced_function_hint(name)
                    if call.namespaced
                    else self._unknown_function_hint(name),
                )
            _, target = self._dispatch_audio(name, dynamic, call, env, select)
            return target.output
        return _UNSUPPORTED_KIND

    # -- table/csv queries --
    #
    # A table query never reaches ffmpeg -- the row model holds every cell at
    # compile time -- so this is a second top-level entry point (`run_table`,
    # parallel to `run`), not a mode bolted onto the streaming one. It reuses
    # the streaming machinery for anything STREAM-shaped (a row alias, a filtered
    # stream, COALESCE's fill) by calling into `_lower_expr` with
    # `self.table_mode` set; the one behavior that changes under it is
    # `_row_stream`'s NULL-row rejection, which becomes an empty cell. Metadata
    # columns have no streaming representation, so those shapes are intercepted
    # before `_lower_expr` sees them.

    def run_table(self) -> list[TableSink]:
        """One :class:`~ffrwd.table.TableSink` per COPY, or one bare-select."""
        for name, body in self.res.ctes.items():
            self.branch_values = {}
            self.cte_columns[name] = tuple(
                self._lower_query(union_branches(body), body, tags="rows")
            )
            self.cte_values[name] = self.branch_values
            self.branch_values = {}
            self._harvest_cte_tags(body)
            self._harvest_cte_dispositions(body)
        self.table_mode = True
        sinks: list[TableSink] = []
        if self.res.sinks:
            for raw in self.res.sinks:
                sinks.append(self._lower_table_sink(raw))
        else:
            result = self._lower_table_query(self.res.branches, self.res.select)
            sinks.append(TableSink(result=result, path=None, csv=False, header=False))
        self.graph.input_options = self._lower_input_options()
        return self._render_specs(sinks)

    def _lower_table_sink(self, raw: RawSink) -> TableSink:
        """One csv COPY: its query lowered, ``FORMAT``/``HEADER`` validated.

        Against ``ffrwd.sink.CSV_OPTIONS``, a separate table from
        ``SINK_OPTIONS``, so a media option like ``video_codec`` here is
        UNKNOWN rather than silently accepted.
        """
        result = self._lower_table_query(list(raw.branches), raw.query)
        header = False
        for option in raw.options:
            if isinstance(_unwrap(option.value), exp.Null):
                continue  # NULL is absence: the option is not written
            line, col = _pos(option.name_node, option.value, raw.path_node)
            value = validate_csv_option(option.name, _sink_value(option.value), line=line, col=col)
            if option.name == "header":
                assert isinstance(value, bool)
                header = value
        return TableSink(result=result, path=raw.path, csv=True, header=header)

    def _lower_table_query(self, branches: list[exp.Select], anchor: exp.Expr) -> TableResult:
        if not branches:
            raise _error(ErrorCode.UNSUPPORTED_SQL, "query has no SELECT", anchor)
        if len(branches) > 1:
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                "a table/csv query does not support UNION ALL",
                branches[1],
                fallback=anchor,
                hint="run each branch as its own query",
            )
        return self._lower_table_branch(branches[0])

    def _lower_table_branch(self, select: exp.Select) -> TableResult:
        """One table/csv branch: row cardinality, then every column, per row.

        Cardinality is the branch's shared row relation -- every row source
        stays aligned to it, joins and CTE references included -- and 1 for a
        branch with no rows at all (a plain metadata/stream SELECT has exactly
        one row, the same way a bare scalar broadcasts). A GROUPED branch (a
        GROUP BY, an ``array_agg``, or both) prints one row per group instead
        -- see :meth:`_lower_grouped_table_branch`.
        """
        env = self._scope(select)
        env.grouped = is_grouped(select)
        env.group_keys = _partition_keys(select, env)
        self._check_grouped_cte_columns(select, env)
        time_conjuncts, row_conjuncts, assertion_conjuncts = self._split_where(select, env)
        per_row = any(self._is_row_window(conjunct, env) for conjunct in time_conjuncts)
        if not per_row:
            self._collect_trims(select, env, time_conjuncts)
        self._filter_rows(row_conjuncts, env, select)
        self._check_assertions(assertion_conjuncts, select)
        self._order_rows(select, env)
        self._limit_rows(select, env)
        if per_row:
            self._collect_trims(select, env, time_conjuncts)

        projections = select.expressions
        if not projections:
            raise _error(
                ErrorCode.UNSUPPORTED_SQL, "SELECT has no output column", fallback=select
            )
        names: list[str] = []
        for projection in projections:
            qualifier = star_qualifier(projection)
            if qualifier is not None:
                names += self._star_names(qualifier, projection, env, select)
            else:
                names.append(_table_column_name(projection))

        if env.grouped:
            return self._lower_grouped_table_branch(select, env, projections, names)

        cardinality = len(env.relation.tuples) if env.relation is not None else 1
        per_column = self._table_columns(projections, env, select, cardinality)
        rows = [[per_column[c][r] for c in range(len(names))] for r in range(cardinality)]
        return TableResult(columns=names, rows=rows)

    def _table_columns(
        self,
        projections: list[exp.Expr],
        env: _Env,
        select: exp.Select,
        cardinality: int,
    ) -> list[list[CellValue]]:
        """Every printed column of a branch, in SELECT order, stars expanded."""
        columns: list[list[CellValue]] = []
        for projection in projections:
            qualifier = star_qualifier(projection)
            if qualifier is not None:
                columns += self._star_cells(
                    qualifier, projection, env, select, cardinality
                )
            else:
                columns.append(
                    self._table_projection(projection, env, select, cardinality)
                )
        return columns

    def _lower_grouped_table_branch(
        self,
        select: exp.Select,
        env: _Env,
        projections: list[exp.Expr],
        names: list[str],
    ) -> TableResult:
        """One printed row per GROUP BY group; no fan-out sink involved.

        Reuses the exact per-row machinery: for each group, the relation's
        tuples are pinned to that group (the array_agg column sees every tuple,
        so it collects the whole group; every other column -- a key or a
        constant, the only shapes grouping validity admits -- sees just the
        first, since it is the same value for the whole group by construction)
        and every projection lowers as one ordinary, single-row column.
        """
        relation = env.relation
        assert relation is not None  # is_grouped implies rows; resolve enforced it
        groups = self._grouped_partitions(env, select)
        original = relation.tuples
        rows: list[list[CellValue]] = []
        try:
            for group in groups:
                row: list[CellValue] = []
                for projection in projections:
                    aggregate = _contains_array_agg(_projection_expr(projection))
                    relation.tuples = list(group) if aggregate else group[:1]
                    row += [
                        cells[0]
                        for cells in self._table_columns([projection], env, select, 1)
                    ]
                rows.append(row)
        finally:
            relation.tuples = original
        return TableResult(columns=names, rows=rows)

    def _grouped_partitions(
        self, env: _Env, select: exp.Select
    ) -> list[list[_RowTuple]]:
        """The relation's tuples partitioned into the groups a table query
        prints, one row each.

        A row-referencing GROUP BY key partitions in FIRST-APPEARANCE order,
        the same partition a media fan-out builds. With no such key the whole
        relation is ONE group -- Postgres's own rule for an aggregate with
        nothing to partition by (unlike a media fan-out's ungrouped case,
        where every row writes its own file).

        An EMPTY relation partitions into NO groups either way: a table query
        prints the same zero rows an ungrouped branch does, and a media query
        falls through to the empty-row-set rejection.
        """
        relation = env.relation
        tuples = relation.tuples if relation is not None else []
        if not tuples:
            return []
        if not env.group_keys:
            return [list(tuples)]
        groups: dict[tuple[RowValue, ...], list[_RowTuple]] = {}
        for row in tuples:
            key = tuple(self._key_value(node, env, row, select) for node in env.group_keys)
            groups.setdefault(key, []).append(row)
        return list(groups.values())

    def _table_projection(
        self, projection: exp.Expr, env: _Env, select: exp.Select, cardinality: int
    ) -> list[CellValue]:
        """One SELECT column, per row: a metadata value, or a stream cell."""
        expr = _unwrap(projection)
        if isinstance(expr, exp.ArrayAgg):
            return self._array_cell_broadcast(expr, env, select, cardinality)
        if isinstance(expr, exp.Column):
            table_node = expr.args.get("table")
            if table_node is not None:
                binding = env.bindings.get(_fold(table_node))
                if isinstance(binding, _RowBinding):
                    name = _fold(expr.this)
                    if name != ROW_STREAM:
                        return self._row_metadata_cells(binding, name, expr, select)
                elif (
                    isinstance(binding, _InputBinding)
                    and _fold(expr.this) in RECORD_ARRAY_COLUMNS
                ):
                    return self._record_cells(
                        binding.alias, _fold(expr.this), expr, select, cardinality
                    )
                elif (
                    isinstance(binding, _InputBinding)
                    and _fold(expr.this) == TAGS_COLUMN
                ):
                    return self._container_tag_cells(
                        binding.alias, expr, select, cardinality
                    )
                elif (
                    isinstance(binding, _InputBinding | _SourceBinding)
                    and _fold(expr.this) in _ARRAY_COLUMNS
                ):
                    return self._array_cell_broadcast(expr, env, select, cardinality)
                elif isinstance(binding, _CteBinding):
                    if _fold(expr.this) in binding.values:
                        # A value column of the body prints as plain data.
                        return self._value_cells(expr, env, select, cardinality)
                    column = self._cte_column(binding, _fold(expr.this))
                    # A splat column falls through to `_value_to_cells` below,
                    # which is where its per-row cardinality is already
                    # honored; a non-splat one (array_agg / a bare input
                    # array, re-exposed through the CTE) stays ONE cell.
                    if column is not None and column.value.is_array and not column.splat:
                        return self._array_cell_broadcast(expr, env, select, cardinality)
        if is_value_expr(expr) or _is_input_value_column(expr, env):
            return self._value_cells(expr, env, select, cardinality)
        shape = subscript_metadata_shape(expr)
        if shape is not None:
            metadata_value = self._accessor_value(expr, select)
            return [metadata_value] * cardinality
        stream_value = self._lower_expr(projection, env, select)
        splat = self._is_splat_projection(projection, env)
        return self._value_to_cells(stream_value, cardinality, splat=splat)

    def _value_cells(
        self, node: exp.Expr, env: _Env, select: exp.Select, cardinality: int
    ) -> list[CellValue]:
        """A CASE / ``||`` column, evaluated once per row.

        The same expression a media query writes back as a tag, PRINTED
        instead: a table query is how you check what the tag would say before
        writing it.
        """
        relation = env.relation
        if relation is None:
            return [self._eval_value(node, env, {}, select)] * cardinality
        return [self._eval_value(node, env, row, select) for row in relation.tuples]

    def _row_metadata_cells(
        self, binding: _RowBinding, name: str, anchor: exp.Expr, select: exp.Select
    ) -> list[CellValue]:
        """A row alias's metadata column, one value per row (NULL for a gap)."""
        schema = binding.schema
        if name == TAGS_COLUMN:
            return [_tag_cell(row) for row in binding.rows]
        if name == DISPOSITION_COLUMN and name in schema:
            return [_disposition_cell(row) for row in binding.rows]
        if name not in schema and map_ref(name) is None:
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"unknown column '{binding.alias}.{column_label(name)}'",
                anchor,
                fallback=select,
                hint=binding.exposes,
            )
        return [None if row is None else row.columns.get(name) for row in binding.rows]

    def _container_tag_cells(
        self, alias: str, anchor: exp.Expr, select: exp.Select, cardinality: int
    ) -> list[CellValue]:
        """A bare ``<input>.tags`` as ONE array cell, broadcast to every row.

        The map's entries print as key/value records in key order:
        ``{(artist,Nobody),(title,Clip)}``. Name a key to read one of them.
        """
        result = self.probes.get(alias)
        if result is None:
            raise self._unreadable_error(
                ErrorCode.INPUT_NOT_FOUND,
                alias,
                f"cannot read tags of '{self._path_of(alias)}'",
                anchor,
                select,
                hint=f"'{alias}.{TAGS_COLUMN}' is the container's own tag map, "
                "and only a readable input has one",
            )
        return [_tags_to_cell(result.tags)] * cardinality

    def _record_cells(
        self,
        alias: str,
        column: str,
        anchor: exp.Expr,
        select: exp.Select,
        cardinality: int,
    ) -> list[CellValue]:
        """A bare ``<input>.chapters`` / ``<input>.cues`` as ONE array cell,
        broadcast to every row.

        The array's records print in schema order (a chapter is index, title,
        start_t, end_t): ``{(1,Intro,0.0,1.0),(2,Chapter 1,1.0,2.0)}``. Unnest
        it to read the fields as columns.
        """
        result = self._record_probe(
            alias,
            column,
            anchor,
            select,
            hint=f"'{alias}.{column}' is the container's own {column}, and "
            "only a readable input has any",
        )
        names = ROW_STAR_COLUMNS[column]
        cell = ArrayCell(
            elements=tuple(
                RecordCell(fields=tuple(row[name] for name in names))
                for row in _record_columns(result, column)
            )
        )
        return [cell] * cardinality

    def _array_cell_broadcast(
        self, node: exp.Expr, env: _Env, select: exp.Select, cardinality: int
    ) -> list[CellValue]:
        """A bare input array column (``f.video``/``f.audio``/...) or a whole
        ``array_agg(...)`` column: every element as ONE array cell,
        broadcasting to every row -- the value does not depend on which row
        (or, grouped, which group) is printing it."""
        value = self._lower_expr(node, env, select)
        cell = ArrayCell(
            elements=tuple(self._stream_to_cell(stream) for stream in value.streams)
        )
        return [cell] * cardinality

    def _value_to_cells(
        self, value: _Value, cardinality: int, splat: bool = True
    ) -> list[CellValue]:
        """A lowered stream `_Value` as one cell per row: a scalar broadcasts,
        and a row column's array (``t`` over N surviving rows) splats
        one stream cell per row -- the array IS the row set, not one cell.

        `splat` False marks an array that is NOT a row set -- a call broadcast
        over a bare input array, whose length is the file's track count and
        has nothing to do with the row count. That one prints as a single
        array cell per row, exactly as the bare array column does.
        """
        if value.is_array and splat:
            return [self._stream_to_cell(stream) for stream in value.streams]
        if value.is_array:
            array_cell = ArrayCell(
                elements=tuple(self._stream_to_cell(stream) for stream in value.streams)
            )
            return [array_cell] * cardinality
        cell = self._stream_to_cell(value.streams[0])
        return [cell] * cardinality

    def _stream_to_cell(self, stream: _Stream) -> CellValue:
        """One stream as a cell, carrying its REF until `_render_specs` runs."""
        if stream.ref == _NULL_STREAM_REF:
            return None
        return StreamCell(type=stream.type, spec=stream.ref)

    def _render_specs(self, sinks: list[TableSink]) -> list[TableSink]:
        """Turn every cell's stream ref into the spec the command will name.

        Which ``-i`` an alias reads is settled only once every input option
        and trim window is known, and two aliases over one untrimmed path
        share a slot. A table previews the command, so it names the same
        input: the refs wait here for the final input list.
        """
        self.graph = dedup_inputs(self.graph)
        return [
            replace(
                sink,
                result=TableResult(
                    columns=sink.result.columns,
                    rows=[[self._render_cell(cell) for cell in row] for row in sink.result.rows],
                ),
            )
            for sink in sinks
        ]

    def _render_cell(self, cell: CellValue) -> CellValue:
        if isinstance(cell, StreamCell):
            return StreamCell(type=cell.type, spec=self._stream_spec(cell.spec))
        if isinstance(cell, ArrayCell):
            return ArrayCell(
                elements=tuple(self._render_cell(element) for element in cell.elements)
            )
        return cell

    def _stream_spec(self, ref: FrameRef) -> str:
        """The ffmpeg stream spec (``"0:a:0"``) for a source ref, else the
        filtergraph node id verbatim (``"n2"``) for a filtered one."""
        if is_src(ref):
            alias, stream_type, index = src_parts(ref)
            return f"{self.graph.sources[alias]}:{_TYPE_MARKERS[stream_type]}:{index}"
        return ref

# provenance & small value helpers


def _provenance(stream: _Stream) -> dict[str, str]:
    """Language/title tags of the source stream an output is derived 1:1 from.

    `_Stream.source` is what threads them: it survives a passthrough, the WHERE
    trim, and any chain of single-stream-input calls unconditionally; a call
    over two or more streams (``amix``, ``overlay``) and a concat pad thread it
    only when every stream feeding them agrees (:func:`_agreed_source`).
    ``language=und`` is what an mp4 muxer stamps on an untagged stream, so it
    carries no information and is not copied.

    Only STREAM_TAG_COLUMNS ride, not every key the source carries: a file's
    ``encoder`` or ``handler_name`` tag riding through a filter would emit
    ``-metadata`` ffmpeg does not emit today.
    """
    source = stream.source
    if source is None:
        return {}
    metadata: dict[str, str] = {}
    for key in STREAM_TAG_COLUMNS:
        value = source.metadata.get(key)
        if value is None:
            continue
        if key == "language" and value == _UNDEFINED_LANGUAGE:
            continue
        metadata[key] = value
    return metadata


def _agreed_source(segments: list[_Stream]) -> StreamMeta | None:
    """The provenance an N:1 join inherits from the streams feeding it.

    Used by both kinds of join that take more than one input stream: a concat
    pad (`segments` is one stream per UNION ALL branch, in branch order) and a
    multi-stream call like ``amix``/``overlay`` (`segments` is its stream
    arguments, in argument order, one element already picked out of each). The
    result is only still "that stream" when every segment says the SAME thing
    about it: the comparison is on the FILTERED provenance dicts, not on the
    raw ``StreamMeta``, so two segments that differ in sample rate or index but
    agree on ``language=fra`` do agree, and two "und"-tagged segments both
    filter down to ``{}`` — nothing to say, so nothing survives. Any
    disagreement, or an empty dict, gives None.

    The first segment's ``StreamMeta`` is what gets threaded: it and the others
    render identically, and it keeps ``_Stream.source`` a real probed stream.
    """
    agreed = _provenance(segments[0])
    if not agreed:
        return None
    if any(_provenance(segment) != agreed for segment in segments[1:]):
        return None
    return segments[0].source


def _outputs(
    columns: list[_Column],
    tags: _TagOverrides,
    dispositions: _DispositionOverrides,
) -> list[Output]:
    """One :class:`~ffrwd.ir.Output` per stream a SELECT list carries.

    The SELECT list IS the output stream list, and an array column is several
    streams, so it splats into consecutive Outputs. Every element of an
    aliased array column keeps that alias VERBATIM (no ordinal suffix): the
    alias names the column, not the stream.
    """
    return [
        Output(
            ref=stream.ref,
            type=stream.type,
            name=column.name,
            metadata=_metadata(stream, tags),
            disposition=_disposition(stream, dispositions),
        )
        for column in columns
        for stream in column.value.streams
    ]


def _metadata(stream: _Stream, tags: _TagOverrides) -> dict[str, str]:
    """One output's tags: its provenance, with this query's overrides applied.

    An override REPLACES the provenance value for its key, a NULL one removes
    the key, and a key nothing overrode passes through untouched.
    """
    metadata = _provenance(stream)
    if stream.source is None:
        return metadata
    for key, value in tags.get(id(stream.source), {}).items():
        if value is None:
            metadata.pop(key, None)
        else:
            metadata[key] = value
    return metadata


def _disposition(
    stream: _Stream, dispositions: _DispositionOverrides
) -> tuple[str, ...] | None:
    """The flags one output asserts, or None where the query asserted none.

    Nothing rides through from the source: ffmpeg copies a stream's own
    disposition already, so only a written column puts `-disposition:<i>` on
    the command line.
    """
    if stream.source is None:
        return None
    return dispositions.get(id(stream.source))


@dataclass(frozen=True)
class _Tags:
    """One ``tags`` column, read: which keys it sets and what it copies.

    `entries` is key -> the expression that computes it, in merge order, so a
    later operand of ``||`` has already overwritten an earlier one's key.
    `copy_alias` is the input whose globals the column copies, and `stripped`
    says an empty map was written -- the two things that decide
    ``-map_metadata``.
    """

    entries: dict[str, exp.Expr]
    copy_alias: str | None
    stripped: bool


def _merge_operands(node: exp.Expr) -> list[exp.Expr]:
    """The operands of a ``||`` chain, left to right; a lone node is one."""
    inner = _unwrap(node)
    if not isinstance(inner, exp.DPipe):
        return [inner]
    left = inner.this
    right = inner.expression
    operands: list[exp.Expr] = []
    if isinstance(left, exp.Expr):
        operands += _merge_operands(left)
    if isinstance(right, exp.Expr):
        operands += _merge_operands(right)
    return operands


def _tags_map_alias(node: exp.Expr) -> str | None:
    """The alias whose whole ``tags`` map `node` names, else None."""
    if not isinstance(node, exp.Column):
        return None
    table_node = node.args.get("table")
    if table_node is None:
        return None
    return _fold(table_node) if _fold(node.this) == TAGS_COLUMN else None


def _tag_text(value: str | int | float | bool) -> str:
    """A tag value as the text ffmpeg receives; a boolean spells itself out."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return value if isinstance(value, str) else str(value)


def _partition_keys(select: exp.Select, env: _Env) -> tuple[exp.Expr, ...]:
    """The GROUP BY keys that actually partition the branch's relation.

    A key reading a row source -- a track row, a chapter row, a CTE row --
    varies from tuple to tuple. An input-level or constant key has the same
    value everywhere and leaves one group.
    """
    return tuple(key for key in group_keys(select) if _reads_row_source(key, env))


def _reads_row_source(node: exp.Expr, env: _Env) -> bool:
    for sub in node.walk():
        if not isinstance(sub, exp.Column):
            continue
        table_node = sub.args.get("table")
        if table_node is None:
            continue
        if isinstance(env.bindings.get(_fold(table_node)), _RowBinding | _CteBinding):
            return True
    return False


def _has_track_rows(env: _Env) -> bool:
    """True when the branch has rows carrying a track to tag per stream.

    Chapter rows and written rows carry none, so a branch holding only those
    tags the CONTAINER, exactly as one with no rows at all does.
    """
    return any(
        isinstance(binding, _RowBinding) and not binding.streamless
        for binding in env.bindings.values()
    )


def _group_row(env: _Env) -> _RowTuple:
    """The one tuple a FILE-level value reads, or no row at all.

    A container tag and a chapter list belong to the file, not to a row, so
    they are evaluated over a single representative tuple: the group's first
    where the branch groups, the relation's first otherwise (an ungrouped
    branch that survives the one-row rule has exactly one).
    """
    relation = env.relation
    if relation is None or not relation.tuples:
        return {}
    return relation.tuples[0]


def _is_value_column(projection: exp.Expr, env: _Env) -> bool:
    """True for a SELECT column that is a compile-time VALUE, not a stream.

    A value column is aliased — the alias names it — and its value is a
    compile-time expression over the row: a literal, NULL, a row's metadata
    column, an input's ``duration`` or container tag, CASE, ``||``, arithmetic
    or ``::text``. Everything else is a stream expression and lowers as one.
    """
    if _projection_name(projection) is None:
        return False
    value = _unwrap(projection)
    if isinstance(value, exp.Null | exp.Literal | exp.Neg) or is_value_expr(value):
        return True
    if _is_input_value_column(value, env):
        return True
    if _is_cte_value_column(value, env):
        return True
    return _row_metadata_column(value, env) is not None


def _reads_cte_value(alias: str, conjunct: exp.Expr, env: _Env) -> bool:
    """True when `conjunct` reads a VALUE column off this CTE alias."""
    binding = env.bindings.get(alias)
    if not isinstance(binding, _CteBinding):
        return False
    return any(
        isinstance(sub, exp.Column)
        and sub.args.get("table") is not None
        and _fold(sub.args["table"]) == alias
        and _fold(sub.this) in binding.values
        for sub in conjunct.walk()
    )


def _is_cte_value_column(node: exp.Expr, env: _Env) -> bool:
    """True for a reference to a CTE's own VALUE column."""
    if not isinstance(node, exp.Column):
        return False
    table_node = node.args.get("table")
    if table_node is None:
        return False
    binding = env.bindings.get(_fold(table_node))
    return isinstance(binding, _CteBinding) and _fold(node.this) in binding.values


def _is_input_value_column(node: exp.Expr, env: _Env) -> bool:
    """True for an input alias's scalar column — ``duration`` or a container
    tag — a value, never a stream."""
    if not isinstance(node, exp.Column):
        return False
    table_node = node.args.get("table")
    if table_node is None:
        return False
    if not isinstance(env.bindings.get(_fold(table_node)), _InputBinding):
        return False
    name = _fold(node.this)
    return name == INPUT_DURATION_COLUMN or tag_key(name) is not None


def _row_metadata_column(node: exp.Expr, env: _Env) -> str | None:
    """The metadata column `node` reads off a row alias, else None (``track``
    is a stream, not metadata).

    A STREAMLESS row -- a chapter row, a written row -- counts here exactly as
    a track row does: `_has_track_rows` sends a branch holding only those to
    the CONTAINER tag and `_group_row` hands the value the one representative
    tuple it reads, so ``c.title AS title`` writes the same tag ``'Ch: ' ||
    c.title AS title`` already did. Without an alias the column is no tag
    column at all and still falls through to `_row_value`'s ordinary "not an
    output" rejection.
    """
    if not isinstance(node, exp.Column):
        return None
    table_node = node.args.get("table")
    if table_node is None:
        return None
    if not isinstance(env.bindings.get(_fold(table_node)), _RowBinding):
        return None
    name = _fold(node.this)
    return None if name == ROW_STREAM else name


def _reads_row_column(node: exp.Expr, env: _Env) -> bool:
    """True when `node` reads a metadata column off a row alias.

    What makes an option's value differ from row to row: ``t.width``,
    ``:'widths'[i.i]``, anything arithmetic over one. An input alias's probed
    column (``f.duration``) is the same for every row and is not one.
    """
    for sub in _unwrap(node).walk():
        if isinstance(sub, exp.Expr) and _row_metadata_column(sub, env) is not None:
            return True
    return False


def _is_row_scalar(node: exp.Expr, env: _Env) -> bool:
    """True for a bare column that is a compile-time VALUE, never a stream --
    an input's probed duration/tag, or a row table's metadata field.

    Lets a ``duration``-typed filter option accept ``starti => f.duration``
    the way it already accepts arithmetic over one (`_option_binder`).
    """
    inner = _unwrap(node)
    return _is_input_value_column(inner, env) or _row_metadata_column(inner, env) is not None


def _flatten(columns: list[_Column]) -> list[_Column]:
    """One column per stream: arrays are gone, every column is a scalar.

    An aliased array column hands its alias to each of its elements, exactly as
    the SELECT-list splat does.
    """
    return [
        _Column(name=column.name, value=_scalar(stream))
        for column in columns
        for stream in column.value.streams
    ]


def _signature(columns: list[_Column]) -> str:
    """Branch column types for a CONCAT_MISMATCH message, arrays as ``audio[2]``."""
    parts = [
        f"{column.value.type}[{len(column.value.streams)}]"
        if column.value.is_array
        else column.value.type
        for column in columns
    ]
    return ", ".join(parts) or "nothing"


def _as_ref(value: object) -> FrameRef:
    """A lowered argument value as a stream ref (dynamic calls take only those)."""
    if not isinstance(value, str):  # pragma: no cover -- structurally impossible
        raise FfrwdError(
            ErrorCode.INTERNAL,
            "a dynamic filter argument lowered to something that is not a stream",
            line=1,
            col=1,
            hint="please report this query as a bug",
        )
    return value


def _listed(names: Iterable[str]) -> str:
    """Comma-list at most ``_MAX_LISTED`` names, then count the rest."""
    items = list(names)
    if len(items) <= _MAX_LISTED:
        return ", ".join(items)
    rest = len(items) - _MAX_LISTED
    return ", ".join(items[:_MAX_LISTED]) + f", ... ({rest} more)"


def _option_hint(name: str, options: dict[str, FilterOption]) -> str:
    """Did-you-mean over the filter's REAL option names, else list them."""
    matches = difflib.get_close_matches(name, sorted(options), n=1, cutoff=0.6)
    if matches:
        return f"did you mean {matches[0]} => ...?"
    if not options:
        return "this filter has no options ffrwd can set"
    return "its options: " + _listed(sorted(options))


def _number_text(value: float) -> str:
    """A range bound as ffmpeg meant it: ``1024`` rather than ``1024.0``."""
    if value == int(value):
        return str(int(value))
    return str(value)


def _range_text(option: FilterOption) -> str | None:
    if option.minimum is not None and option.maximum is not None:
        return f"from {_number_text(option.minimum)} to {_number_text(option.maximum)}"
    if option.minimum is not None:
        return f"at least {_number_text(option.minimum)}"
    if option.maximum is not None:
        return f"at most {_number_text(option.maximum)}"
    return None


def _literal_value(node: exp.Expr) -> object | None:
    """A named argument's value as a python scalar, or None if it is not a literal.

    Deliberately separate from :func:`_number`: that raises its own message,
    and an option's expected type is only known after the registry has been
    consulted, so reading the value and judging it are two steps here.
    """
    node = _unwrap(node)
    if isinstance(node, exp.Boolean):
        return bool(node.this)
    negated = False
    if isinstance(node, exp.Neg) and isinstance(node.this, exp.Expr):
        negated = True
        node = _unwrap(node.this)
    if not isinstance(node, exp.Literal):
        return None
    if node.is_string:
        return None if negated else str(node.this)
    try:
        value = node.to_py()
    except (ArithmeticError, TypeError, ValueError):
        return None
    if isinstance(value, bool):  # sqlglot never does this; be explicit anyway
        return None
    number = value if isinstance(value, int) else float(value)
    return -number if negated else number


def _literal_node(value: RowValue, source: exp.Expr) -> exp.Expr:
    """A computed value back as the literal node the option binder reads.

    The synthesized node inherits `source`'s position, so an option that
    rejects what a row computed still points at the expression that wrote it.
    """
    node: exp.Expr
    if value is None:
        node = exp.Null()
    elif isinstance(value, str):
        node = exp.Literal.string(value)
    elif value < 0:
        node = exp.Neg(this=exp.Literal.number(str(-value)))
    else:
        node = exp.Literal.number(str(value))
    line, col = _pos(source)
    node.meta.update({"line": line, "col": col})
    return node


def _option_got(node: exp.Expr, value: object) -> str:
    """How a rejected option value is echoed back in the message."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)
    if isinstance(value, str):
        return repr(value)
    inner = _unwrap(node)
    if isinstance(inner, exp.Literal):
        # A literal `_literal_value` could not read (sqlglot tokenizes `1e` as a
        # number but `to_py()` raises): echo what was written, not its shape.
        return repr(str(inner.this))
    return _describe(inner)


def _option_error(
    filter_name: str, option: FilterOption, arg: _NamedArg, call: exp.Expr, what: str, hint: str
) -> FfrwdError:
    return _error(
        ErrorCode.FILTER_OPTION_TYPE,
        f"option '{option.name}' of filter '{filter_name}' {what}",
        arg.value,
        fallback=call,
        hint=hint,
    )


def _enable_value(
    filter_name: str, arg: _NamedArg, call: exp.Expr, timeline: bool
) -> str:
    """The timeline ``enable`` expression, or the rejection for this filter.

    Two ways it fails. The filter has no timeline support at all, which is a
    property of the FILTER and so reads as an unknown option on it — flavoured
    with the reason, because "gblur has no option 'enable'" would be a lie
    about gblur. Or the value is not a string: an ffmpeg timeline expression is
    text, and a bare number would silently mean "always on"/"never on" rather
    than the window the writer had in mind.

    The expression's CONTENT is deliberately unchecked: the variable
    vocabulary is per-filter and not introspectable, so it is ffmpeg's to
    validate at run time.
    """
    if not timeline:
        raise _error(
            ErrorCode.UNKNOWN_FILTER_OPTION,
            f"filter '{filter_name}' has no option 'enable': your ffmpeg does "
            f"not flag '{filter_name}' as supporting timeline editing",
            arg.value,
            fallback=call,
            hint=_NO_TIMELINE_HINT,
        )
    value = _literal_value(arg.value)
    if not isinstance(value, str):
        raise _error(
            ErrorCode.FILTER_OPTION_TYPE,
            f"option 'enable' of filter '{filter_name}' expects an ffmpeg "
            f"timeline expression, got {_option_got(arg.value, value)}",
            arg.value,
            fallback=call,
            hint=_ENABLE_HINT,
        )
    return value


def _option_value(
    filter_name: str, option: FilterOption, arg: _NamedArg, call: exp.Expr
) -> object:
    """One named argument's value, checked against its introspected AVOption.

    The type map is the RFC's: numeric AVOptions take a bare number (range
    checked whenever ffmpeg printed a parseable one), booleans take ``true`` /
    ``false``, an enum takes one of its named constants, and everything else
    takes a string — or a bare number, since an ffmpeg option value is text on
    the command line either way and ``duration``/``video_rate``/expression
    options (``xfade``'s ``duration``, ``crop``'s ``x``) are routinely numeric.
    """
    value = _literal_value(arg.value)
    got = _option_got(arg.value, value)
    if option.unusable:
        raise _option_error(
            filter_name,
            option,
            arg,
            call,
            "has an ffmpeg type (binary/dictionary) ffrwd cannot set",
            "drop it; ffrwd sets numeric, string and boolean options only",
        )
    if option.type == "num":
        if not isinstance(value, int | float) or isinstance(value, bool):
            bounds = _range_text(option)
            raise _option_error(
                filter_name,
                option,
                arg,
                call,
                f"expects a number, got {got}",
                f"write a bare numeric literal ({bounds})" if bounds else
                "write a bare numeric literal, e.g. sigma => 5",
            )
        bounds = _range_text(option)
        below = option.minimum is not None and value < option.minimum
        above = option.maximum is not None and value > option.maximum
        if (below or above) and bounds is not None:
            raise _option_error(
                filter_name,
                option,
                arg,
                call,
                f"accepts a number {bounds}, got {got}",
                f"pick a value {bounds}",
            )
        return value
    if option.type == "bool":
        if not isinstance(value, bool):
            raise _option_error(
                filter_name,
                option,
                arg,
                call,
                f"expects true or false, got {got}",
                "write the bare word true or false, with no quotes",
            )
        return value
    if option.constants:
        if not isinstance(value, str) or value not in option.constants:
            constants = _listed(option.constants)
            matches = (
                difflib.get_close_matches(value, list(option.constants), n=1, cutoff=0.6)
                if isinstance(value, str)
                else []
            )
            raise _option_error(
                filter_name,
                option,
                arg,
                call,
                f"expects one of its named constants ({constants}), got {got}",
                f"did you mean '{matches[0]}'?"
                if matches
                else "the value is a single-quoted constant name, not a number",
            )
        return value
    if isinstance(value, str) or (isinstance(value, int | float) and not isinstance(value, bool)):
        return value
    raise _option_error(
        filter_name,
        option,
        arg,
        call,
        f"expects a string, got {got}",
        "write a single-quoted string literal, e.g. flags => 'lanczos'",
    )


def _sql_text(node: exp.Expr) -> str:
    """The argument as the user wrote it, for a BROADCAST_MISMATCH message.

    ``dialect="postgres"`` matters: it re-adds the ``INDEX_OFFSET`` sqlglot
    subtracted at parse time, so ``a.audio[2]`` renders as ``a.audio[2]``.
    """
    return str(node.sql(dialect="postgres"))


def _stream_count(count: int) -> str:
    return f"{count} stream" + ("" if count == 1 else "s")


def _contains_array_agg(node: exp.Expr) -> bool:
    """True if `node` is, or contains anywhere, an ``array_agg(...)`` call.

    A grouped branch's projection needs the WHOLE group's tuples exactly when
    it contains an aggregate somewhere -- ``array_agg(t)`` at the top, same as
    always, and now also under ``VARIADIC`` (``concat(VARIADIC array_agg(t))``)
    or alongside a positional stream (``concat(intro, VARIADIC array_agg(t))``).
    Grouping validity already forbids a bare row-column reference OUTSIDE an
    array_agg in the same projection, so "contains one anywhere" and "needs
    the group, not just its first tuple" are the same question.
    """
    return any(isinstance(sub, exp.ArrayAgg) for sub in node.walk())


# public entry point


def lower(
    res: Resolved,
    probes: dict[str, ProbeResult | None],
    *,
    registry: Registry | None = None,
    on_warning: OnWarning | None = None,
    describes: dict[str, Described] | None = None,
    invoke: Invoke = wasm_invoke,
    probe_failures: Mapping[str, ProbeFailure | None] | None = None,
    probe_source: ProbeSource = wasm_probe_source,
) -> Graph:
    """Lower a resolved query into an IR graph -- its FIRST command's.

    The whole query except for the one fan-out shape that compiles to a
    command sequence (see :func:`lower_commands`, which returns them all).

    `probes` is keyed by input ALIAS (``compiler.compile_sql`` builds it, one
    ``probe()`` per distinct path); a missing or ``None`` entry means that
    input could not be read, and lowering stays symbolic for it.

    `registry` IS the function surface: the filter set of the ffmpeg
    on PATH, introspected lazily. It is a PARAMETER rather than a module lookup
    so that a caller — ``compile_sql``, or a test — decides which ffmpeg (or
    which captured snapshot) this compile resolves against. None, or an empty
    one, means every call name is UNKNOWN_FUNCTION.

    `describes` is keyed by MODULE PATH, one entry per path a ``LANGUAGE
    wasm`` declaration names (``compiler.compile_commands`` builds it, one
    ``describe()`` per distinct path). It is a parameter for the same reason
    `probes` is: a lowering test hands over a synthetic one and spawns
    nothing. `invoke` runs one VALUE function's module, once per distinct
    call site's arguments, to fold its result -- a parameter for the same
    reason. `probe_source` runs one ``RETURNS source`` module's ``probe``,
    once per FROM alias that calls one, to bind its catalog -- a parameter
    for the same reason.

    Raises ``FfrwdError`` — and nothing else — on every rejection.
    """
    return lower_commands(
        res, probes, registry=registry, on_warning=on_warning, describes=describes,
        invoke=invoke, probe_failures=probe_failures, probe_source=probe_source,
    )[0]


def lower_commands(
    res: Resolved,
    probes: dict[str, ProbeResult | None],
    *,
    registry: Registry | None = None,
    on_warning: OnWarning | None = None,
    describes: dict[str, Described] | None = None,
    invoke: Invoke = wasm_invoke,
    probe_failures: Mapping[str, ProbeFailure | None] | None = None,
    probe_source: ProbeSource = wasm_probe_source,
) -> list[Graph]:
    """Lower a resolved query into one IR graph per ffmpeg COMMAND.

    Usually ONE graph, a fan-out COPY included: ffmpeg takes several output
    files per invocation, so a ``TO (<expression>)`` lowers each surviving
    row into a :class:`SinkUnit` of a single graph, sharing one decode of the
    inputs. The row COUNT is a property of the probed file, so it comes back
    from the lowering rather than being known up front.

    The exception is a fan-out that TRIMS and stream-copies every stream it
    maps (:func:`_fanout_keeps_chain`): that one lowers again, one graph per
    row, and the caller chains the commands.

    Same probing/registry contract as :func:`lower`; raises ``FfrwdError``
    -- and nothing else -- on every rejection.
    """
    try:
        shared = _Lowerer(
            res, probes, registry, fanout_sinks=True, on_warning=on_warning,
            describes=describes, invoke=invoke, probe_failures=probe_failures,
            probe_source=probe_source,
        )
        graph = shared.run()
        count = shared.fanout_count
        if count is None:
            return [graph]
        _check_distinct_paths(
            [unit.path for unit in graph.sinks], res, grouped=shared.fanout_grouped
        )
        if not _fanout_keeps_chain(graph, conflict=shared.fanout_window_conflict):
            return [graph]
        return [
            _Lowerer(
                res, probes, registry, fanout_index=index, on_warning=on_warning,
                describes=describes, invoke=invoke, probe_failures=probe_failures,
                probe_source=probe_source,
            ).run()
            for index in range(count)
        ]
    except FfrwdError:
        raise
    except Exception as err:  # backstop: guardrail #7, no panics on user input
        raise FfrwdError(
            ErrorCode.INTERNAL,
            f"internal error while lowering ({err.__class__.__name__}: {err})",
            line=1,
            col=1,
            hint="please report this query as a bug",
        ) from err


def _fanout_keeps_chain(graph: Graph, *, conflict: bool) -> bool:
    """True when this fan-out has to stay one ffmpeg command per file.

    An output-side seek re-encodes, and ffmpeg writes a corrupt file when one
    meets a stream copy, so a windowed fan-out whose every mapped stream is a
    copy keeps the ``&&`` chain and seeks its inputs instead. Anything that
    re-encodes -- a filtered stream, a codec the sink names -- takes the
    single invocation, and the streams that would have been copies re-encode
    along with it. `conflict` is the other way back to the chain: one file
    wanting two different windows, which only an ``-i`` seek can say.
    """
    if conflict:
        return True
    if all(unit.window is None for unit in graph.sinks):
        return False
    return all(
        is_src(output.ref) and output.type not in copy_suppressed_scopes(unit.options)
        for unit in graph.sinks
        for output in unit.outputs
    )


def _check_distinct_paths(
    paths: list[str | None], res: Resolved, *, grouped: bool = False
) -> None:
    """No two fan-out files may share a destination.

    Rows sharing a destination is the typo guard; GROUP BY is how a query ASKS
    for them to share one, so the hint says so. Two distinct GROUPS colliding
    is still a rejection -- the key told them apart, the name did not.
    """
    what = "groups" if grouped else "rows"
    hint = (
        "add a column that tells the groups apart to the TO expression"
        if grouped
        else "add a column that tells the rows apart, e.g. t.index::text, to "
        "the TO expression, or GROUP BY the column they share to write one "
        "file per group"
    )
    seen: dict[str, int] = {}
    anchor = res.sinks[0].path_expr if res.sinks else None
    fallback = res.sinks[0].path_node if res.sinks else None
    for index, path in enumerate(paths):
        if path is None:
            continue
        if path in seen:
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"{what} {seen[path] + 1} and {index + 1} both name '{path}'",
                anchor,
                fallback=fallback,
                hint=hint,
            )
        seen[path] = index


def lower_table(
    res: Resolved,
    probes: dict[str, ProbeResult | None],
    *,
    registry: Registry | None = None,
    on_warning: OnWarning | None = None,
    describes: dict[str, Described] | None = None,
    invoke: Invoke = wasm_invoke,
    probe_failures: Mapping[str, ProbeFailure | None] | None = None,
    probe_source: ProbeSource = wasm_probe_source,
) -> list[TableSink]:
    """Lower a resolved TABLE query into its printable result set(s).

    The sibling of :func:`lower` for a query with no media destination -- a
    bare SELECT, or every COPY a ``FORMAT csv`` one. Never
    executes ffmpeg, never inserts splits (there is no filtergraph fan-out to
    consume-once here, only cells). Same probing/registry contract as
    :func:`lower`; raises ``FfrwdError`` -- and nothing else -- on every
    rejection.
    """
    try:
        return _Lowerer(
        res, probes, registry, on_warning=on_warning, describes=describes, invoke=invoke,
        probe_failures=probe_failures, probe_source=probe_source,
    ).run_table()
    except FfrwdError:
        raise
    except Exception as err:  # backstop: guardrail #7, no panics on user input
        raise FfrwdError(
            ErrorCode.INTERNAL,
            f"internal error while lowering ({err.__class__.__name__}: {err})",
            line=1,
            col=1,
            hint="please report this query as a bug",
        ) from err
