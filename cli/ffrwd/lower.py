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
  (:func:`ffrwd.calls._reject_passthrough_args`);
* under a **CTE's** WHERE time range that is actually consumed in that branch ->
  ``UNSUPPORTED_SQL`` (:meth:`_Lowerer._access`). An INPUT alias's WHERE is not
  a filtergraph trim at all any more (see the WHERE bullet below), so it carries
  captions perfectly well; a CTE's window IS a filtergraph trim, so for it the
  rejection is permanent;
* as a UNION ALL branch column -> ``UNSUPPORTED_SQL``
  (:func:`ffrwd.admit._check_concat_columns`; ``concat`` has ``v``/``a`` pads only).

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
A source alias is the third kind of binding (:class:`ffrwd.bindings._SourceBinding`), and
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
import binascii
import json
import struct
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Literal

from sqlglot import exp

from ffrwd import loudnorm
from ffrwd.admit import (
    _check_annotation_argument,
    _check_coalesce_fill,
    _check_coalesce_width,
    _check_concat_columns,
    _check_concat_signature,
    _check_fill_type,
    _check_grouped_cte_columns,
    _check_named_args,
    _check_per_track_options,
    _check_realtime_option,
    _check_required_options,
    _check_row_sink_arity,
    _check_rows_argument,
    _check_star_table_mode,
    _check_tag_key,
    _check_variadic_count,
    _check_vector_dims,
    _described,
    _described_rows,
    _described_source,
)
from ffrwd.bindings import (
    _RENDITION_SCHEMA,
    RENDITION_COLUMN,
    _Binding,
    _CteBinding,
    _CteRow,
    _Env,
    _has_track_rows,
    _InputBinding,
    _row_value_as_cell,
    _RowBinding,
    _RowRelation,
    _RowTuple,
    _SourceBinding,
    _tags_to_cell,
    _track_of,
    _TrackRow,
)
from ffrwd.calls import (
    _bad_count,
    _bad_streams,
    _Call,
    _call_parts,
    _expand_call,
    _macro_function_hint,
    _macro_options,
    _n_input_count,
    _reject_null_stream,
    _reject_passthrough_args,
)
from ffrwd.ctes import (
    _cte_cell_column,
    _cte_column,
    _cte_column_ref,
    _cte_columns_hint,
)
from ffrwd.destinations import (
    _PER_TRACK_OPTION_HINT,
    _bind_sink_streams,
    _default_audio_row,
    _disable_scene_cuts,
    _each,
    _join_codecs,
    _nothing_to_write_error,
    _packet_sink_audio_codec,
    _rows_call,
    _rows_file,
    _rows_projection,
    _VariantRow,
)
from ffrwd.errors import ErrorCode, FfrwdError
from ffrwd.evaluate import (
    _compare,
    _computed_arg,
    _eval_list_element,
    _eval_value,
    _EvalContext,
    _kleene_and,
    _kleene_or,
    _literal_of,
    _tag_text,
    _wasm_params,
)
from ffrwd.expressions import (
    _coalesce_label,
    _describe,
    _error,
    _number,
    _sql_text,
    _unwrap,
)
from ffrwd.fills import (
    _COALESCE_HINT,
    _FILL_SPELLINGS,
    _coalesce_arguments,
    _coalesce_binding,
    _fill_hint,
    _inherited_fill_options,
    _paired_row,
)
from ffrwd.filter_options import (
    _listed,
    _NamedArg,
    _option_value,
)
from ffrwd.filters import (
    _ARRAY_INPUT_HINT,
    _CONCAT_VARIADIC_HINT,
    _N_INPUT_HINT,
    ARRAY_RETURNING,
    _array_options,
    _ArrayFilter,
    _BadCount,
    _concat_options,
    _filter_options,
    _n_input_call,
    _namespaced_function_hint,
    _NInputFilter,
    _options_for,
    _source_filter,
    _twin_dispatch_stem,
    _twin_pair,
    _unknown_function_hint,
)
from ffrwd.functions import Annotation, WasmFunction
from ffrwd.inputs import render_options
from ffrwd.inputs import validate_option as validate_input_option
from ffrwd.ir import (
    MAX_DISTANCE,
    NO_CHAPTERS,
    NO_METADATA,
    PIPE,
    PREDICATE,
    ROWFILTER,
    ROWMERGE,
    Attachment,
    FrameRef,
    Graph,
    ModuleSource,
    Node,
    Output,
    RowsSink,
    SinkUnit,
    StreamType,
    UrlSource,
    UrlSourceRow,
    dedup_inputs,
    is_src,
    src_alias,
    src_parts,
)
from ffrwd.ir import (
    SourceTrack as IrSourceTrack,
)
from ffrwd.macros import INPUT_MACROS, MACROS, InputMacro
from ffrwd.merge import RowValue
from ffrwd.modules import (
    _UNCACHED,
    _vector_field,
)
from ffrwd.parser import (
    _BUILTIN_VALUE_FUNCS,
    _REMOVED_FRAME,
    _VECTOR_BUILTIN_ARITY,
    FILTER_NAMESPACE,
    MACRO_NAMESPACE,
    MAP_COLUMNS,
    MERGE_CUES,
    ROW_MERGE,
    ROW_PREDICATE,
    ROW_STREAM,
    SINK_STREAMS,
    RawInputOption,
    RawRowJoin,
    RawSink,
    RawSinkOption,
    RawSource,
    RawTrackRows,
    Resolved,
    _pos,
    _projection_expr,
    _time_bounds,
    annotation_projection,
    article,
    column_label,
    from_entries,
    group_keys,
    is_grouped,
    is_value_expr,
    map_path,
    map_ref,
    null_variable,
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
    CueMeta,
    ProbeFailure,
    ProbeResult,
    RenditionMeta,
    StreamMeta,
    track_cues,
)
from ffrwd.probe import probe as probe_one_path
from ffrwd.processes import COPY_CODEC, ref_type
from ffrwd.registry import DynamicFilter, FilterOption, Registry
from ffrwd.rows import (
    _STREAMLESS_ROW,
    _add_cte_rows,
    _add_series_rows,
    _add_values_rows,
    _fanout_groups,
    _filter_rows,
    _from_rendition_table,
    _group_row,
    _grouped_partitions,
    _is_cue_array_column,
    _is_row_window,
    _is_splat_projection,
    _join_rows,
    _limit_rows,
    _merged_rows,
    _not_rows,
    _order_rows,
    _per_row_seeks,
    _reads_row_alias,
    _reads_unbound_rendition_column,
    _rendition_row_cells,
    _row_elements,
    _row_metadata_cells,
    _unmatched_text,
    _value_cells,
    _value_to_cells,
)
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
from ffrwd.sources import (
    _source_columns_hint,
    _url_source_payload,
)
from ffrwd.table import (
    ArrayCell,
    CellValue,
    RecordCell,
    StreamCell,
    TableResult,
    TableSink,
)
from ffrwd.types import (
    ATTACHMENTS_COLUMN,
    CHAPTERS_COLUMN,
    CUE_TYPE,
    CUES_COLUMN,
    DISPOSITION_COLUMN,
    DISPOSITION_KEYS,
    EMBEDDING_TYPE,
    EMBEDDINGS_COLUMN,
    INPUT_DURATION_COLUMN,
    RECORD_ARRAY_COLUMNS,
    RECORD_ELEMENTS,
    ROW_SCHEMAS,
    ROW_STAR_COLUMNS,
    STAR_COLUMNS,
    STREAM_ARRAY_COLUMNS,
    TAGS_COLUMN,
    TIME_COLUMN,
    TRACK_RECORD_COLUMNS,
    RowColumnType,
)
from ffrwd.values import (
    _ARRAY_COLUMNS,
    _NULL_STREAM_REF,
    _PASSTHROUGH_ONLY,
    _TYPE_MARKERS,
    _agreed_source,
    _array,
    _Column,
    _is_null,
    _provenance,
    _scalar,
    _Stream,
    _stream_count,
    _stream_to_cell,
    _Value,
    _writes_nothing,
)
from ffrwd.warnings import FfrwdWarning, OnWarning, WarningCode
from ffrwd.wasm import (
    CODEC_ENCODERS,
    WIRE_VIDEO_CODECS,
    Described,
    Invoke,
    ProbeSource,
    catalog_as_probe,
    encoder_codec,
    language_tag,
    rows_vector_dims,
)
from ffrwd.wasm import invoke as wasm_invoke
from ffrwd.wasm import probe_source as wasm_probe_source
from ffrwd.written import (
    _ATTACHMENT_EXAMPLE,
    _CHAPTER_EXAMPLE,
    _CHAPTERS_COLUMN_HINT,
    _CUE_ARROW,
    _CUE_EXAMPLE,
    _EMBEDDING_EXAMPLE,
    _attachment_records,
    _Chapter,
    _chapter_records,
    _Cue,
    _cue_records,
    _embedding_dims,
    _embedding_records,
    _flag_spec,
    _read_tags,
    _Tags,
)

__all__ = ["lower", "lower_table"]

# Probes ONE path a URL source's row named, the way the pre-lowering pass
# probes an `input()` path -- which one is only known once the module has
# answered, so it happens here instead. :func:`ffrwd.probe.probe` is the real
# one; a lowering test passes its own, so binding a URL source needs no file.
ProbePath = Callable[[str], ProbeResult | None]

# The container array columns a MEDIA query's `SELECT *` expands: the stream
# ones, in declaration order. `chapters` is an array column too, but a chapter
# is not a stream, so it takes no output position.
_STREAM_STAR_COLUMNS: tuple[str, ...] = tuple(
    name for name in STAR_COLUMNS if name in STREAM_ARRAY_COLUMNS
)

# A rendition row's `SELECT *` expands to its two stream arrays, video then
# audio -- the same two columns a manifest destination reads off a row
# (:meth:`_Lowerer._row_cells`); a rendition never carries subtitle/data.
_RENDITION_STAR_COLUMNS: tuple[StreamType, ...] = ("video", "audio")

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
_ONE_FILE_PER_GROUP_HINT = (
    "one group is one file, so the destination has to name the group, e.g. "
    "TO (t.tags.language || '.mka'); group by a column every row agrees on to write "
    "a single file instead"
)
# The hint a "too many rows/streams for one slot" refusal takes when the
# offending relation is a ladder: the fix is narrowing it to one rendition,
# not restructuring the query into rows.
_RENDITION_PICK_HINT = (
    "pick a rendition: WHERE on height, bandwidth or name, or ORDER BY "
    "bandwidth DESC LIMIT 1"
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


def _record_row_hint(record: str) -> str:
    """What a record row can be asked for, when a query asked it for a stream."""
    named = f"{article(record)} {record}"
    return (
        f"{named} row has no stream column at all — {named} is not a "
        "track — so it can only be read as a metadata query, e.g. no COPY, or "
        "COPY ... WITH (FORMAT csv)"
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


_VARIADIC_HINT = (
    "VARIADIC spreads an array as the call's argument list, and only a filter "
    "whose pad count follows its argument count takes it -- an N-input filter "
    "(amix, hstack, xstack, ...) or concat"
)


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


def _fill_call(node: exp.Expr) -> _Call | None:
    """The generated stand-in `node` spells -- ``ffmpeg.<source>()``,
    ``ffrwd.empty_captions()`` -- or None, which makes it a stream
    expression like any other. A source takes no stream, so a call with a
    positional argument is a filter over one, not a stand-in."""
    call = _call_parts(node)
    if call is None or call.args or not (call.namespaced or call.is_macro):
        return None
    return call


# small AST helpers


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


def _name_counts(*candidate_lists: list[str | None]) -> dict[str, int]:
    """How many times each name appears, across every list given.

    `%v` is ONE directory namespace regardless of kind -- a video row named
    ``720p`` and an audio row also named ``720p`` (the same rendition's
    own name, read off its audio cell) would both land in ``out/720p/``
    -- so a collision is counted across video and audio together, not per
    kind.
    """
    counts: dict[str, int] = {}
    for candidates in candidate_lists:
        for name in candidates:
            if name is not None:
                counts[name] = counts.get(name, 0) + 1
    return counts


def _fallback_names(
    candidates: list[str | None], prefix: str, counts: dict[str, int]
) -> list[str]:
    """Each candidate name, or its positional fallback where it fails.

    A name fails when it is missing, ``und``, or shared with another --
    `counts` (:func:`_name_counts`) says how many times, across every kind
    sharing the ``%v`` namespace, not just this list -- the colliding ones
    ALL fall back, since none of them owns the name.
    """
    return [
        name
        if name is not None and name != _UNDEFINED_LANGUAGE and counts.get(name, 0) == 1
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


def _rendition_name(stream: _Stream | None) -> str | None:
    """A cell's rendition NAME -- HLS NAME, a DASH Representation's @id, or a
    variant playlist's directory -- when it is an unmodified rendition-row
    read, else None."""
    if stream is None or stream.rendition is None:
        return None
    return stream.rendition.name


def _rendition_language(stream: _Stream | None) -> str | None:
    """A cell's rendition LANGUAGE/@lang, None for none, ``und``, or a cell
    that carries no rendition provenance."""
    if stream is None or stream.rendition is None:
        return None
    language = stream.rendition.language
    if language is None or language == _UNDEFINED_LANGUAGE:
        return None
    return language


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


# What the sidecar writes a module's rows as when the query writes them
# itself, and the destination that asks for it.
_ROWS_CONTAINER = "ndjson"

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

# What a metadata track is written to: the names Matroska goes by, as an
# extension or as a written `format` option.
_MATROSKA_FORMATS = frozenset({"mkv", "mka", "mks", "matroska"})


def _container_of(options: Mapping[str, object], path: str) -> str:
    """The container a destination writes: its `format` option, else its
    extension. An extensionless path with no format reads as "unnamed"."""
    written = options.get("format")
    if isinstance(written, str) and written:
        return written.lower()
    suffix = path.rsplit(".", 1)[-1].lower() if "." in path else ""
    return suffix or "unnamed"


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


# The stream tag a metadata track takes its SELECT alias as, and the one a
# track read back is named by.
STREAM_TITLE_TAG = "title"

# A vector track's own stream tag: how many numbers each of its blocks
# carries. It is what tells a vector track from a caption track when a file
# is read back, and how to read the payload.
VECTOR_DIMS_TAG = "vector_dims"
_VECTOR_ITEM_BYTES = 4

# The document's first word, the separator between a cue's two bounds, and
# the two characters WebVTT reads as markup inside a cue.
_WEBVTT_MAGIC = "WEBVTT"


def _cues_webvtt(cues: Sequence[_Cue], noun: str = CUE_TYPE) -> str:
    """One evaluated ``cue[]`` as a WebVTT document's text.

    ``WEBVTT`` then one block per cue, in written order, blocks separated by
    a blank line: the format `ffrwd.empty_captions()` already writes, with
    cues in it. Bounds render as ``HH:MM:SS.mmm``, which is the only
    timestamp spelling WebVTT has -- so a bound is written to the
    millisecond and reads back to the millisecond.

    `noun` names the record a rejection is about: an embedding's rows travel
    in this same document, its vectors written as the text of each block.
    """
    blocks = [_WEBVTT_MAGIC]
    previous: int | float | None = None
    for position, cue in enumerate(cues, start=1):
        _check_cue_span(
            noun, position, cue.start, cue.end, previous, cue.start_node, cue.end_node
        )
        previous = cue.start
        timing = f"{_cue_timestamp(cue.start)} {_CUE_ARROW} {_cue_timestamp(cue.end)}"
        blocks.append(f"{timing}\n{cue.text}")
    return "\n\n".join(blocks) + "\n"


def _vector_payload(values: Sequence[float]) -> str:
    """One vector as the text its block carries: little-endian f32, base64.

    The row's other fields are not in the payload -- its bounds are the
    block's own timing, and the track's title names the column it came from.
    """
    return base64.b64encode(struct.pack(f"<{len(values)}f", *values)).decode()


def _vector_values(payload: str, dims: int) -> tuple[float, ...] | None:
    """One block's text read back as `dims` floats, or None if it is not that.

    The converse of :func:`_vector_payload`: text that is not base64, or
    whose bytes are not exactly `dims` little-endian f32, is not a vector
    this wrote.
    """
    try:
        raw = base64.b64decode(payload.strip(), validate=True)
    except (ValueError, binascii.Error):
        return None
    if len(raw) != dims * _VECTOR_ITEM_BYTES:
        return None
    return struct.unpack(f"<{dims}f", raw)


def _cue_timestamp(seconds: int | float) -> str:
    """One cue bound as WebVTT's ``HH:MM:SS.mmm``."""
    total = round(seconds * 1000)
    hours, total = divmod(total, 3_600_000)
    minutes, total = divmod(total, 60_000)
    whole, milliseconds = divmod(total, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole:02d}.{milliseconds:03d}"


def _check_cue_span(
    noun: str,
    position: int,
    start: int | float,
    end: int | float,
    previous: int | float | None,
    start_cell: exp.Expr,
    end_cell: exp.Expr,
) -> None:
    """One written row against the two rules a WebVTT document obeys.

    A row runs forward and the document lists its rows in ascending order.
    Overlap is NOT a rule here, unlike a chapter list: WebVTT is allowed to
    show two captions at once, and a player draws both.
    """
    if start >= end:
        raise _error(
            ErrorCode.UNSUPPORTED_SQL,
            f"{noun} {position} ends at {end}, which is not after its start {start}",
            end_cell,
            hint=f"a {noun} runs from start_t to end_t: end_t must be larger",
        )
    if previous is not None and start < previous:
        raise _error(
            ErrorCode.UNSUPPORTED_SQL,
            f"{noun} {position} starts at {start}, before {noun} {position - 1} at "
            f"{previous}",
            start_cell,
            hint=f"a WebVTT document lists its rows in ascending order; reorder "
            f"them. Two {noun} rows MAY overlap",
        )


def _cue_rows(
    cues: Sequence[CueMeta], title: str | None
) -> list[dict[str, RowValue]]:
    """One track's cues as its rows. A WebVTT document's track has no title."""
    return [
        {
            "index": cue.index,
            "track": title,
            "text": cue.text,
            "start_t": cue.start_t,
            "end_t": cue.end_t,
        }
        for cue in cues
    ]


def _record_tracks(column: str, result: ProbeResult) -> list[StreamMeta]:
    """The tracks one record column reads, in file order.

    A container's caption streams, each belonging to `cues` or to
    `embeddings` by its ``vector_dims`` tag and never to both.
    """
    wants_vectors = column == EMBEDDINGS_COLUMN
    return [
        meta
        for meta in result.by_type("subtitle")
        if (VECTOR_DIMS_TAG in meta.metadata) == wants_vectors
    ]


def _track_name(title: str | None, meta: StreamMeta) -> str:
    """How a message names one track: its title, else its position."""
    return f"'{title}'" if title is not None else f"{meta.type}[{meta.index + 1}]"


def _titled_track_hint(source: str, column: str, titles: Sequence[str]) -> str:
    """What to write instead of a title the file does not carry."""
    if not titles:
        return (
            f"a {column} track is named by its title; this file's carry none, "
            f"so read them all: unnest({source}.{column})"
        )
    listed = ", ".join(f"'{title}'" for title in titles)
    return f"the {column} tracks it carries are titled {listed}"


def _input_value(node: exp.Expr) -> object:
    """One `input('path', name => value)` value as a python scalar.

    Mirrors :func:`_sink_value`, with one addition: INPUT_OPTIONS has a
    ``"num"`` type (``framerate``, ``itsoffset``) whose value may carry a
    leading ``-`` -- ``itsoffset`` legitimately takes a negative offset.
    ``exp.Neg`` is unwrapped first, the same rule :func:`ffrwd.expressions._number` applies to
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


def _scalar_columns(
    columns: Mapping[str, RowValue],
) -> dict[str, str | int | float | bool | None]:
    """A URL source row's columns as the scalars they are.

    A source's catalog carries no vector - `_url_source_types` refused one
    before this is reached - so the narrowing only tells the type checker
    what the rows already guarantee.
    """
    return {
        name: value for name, value in columns.items() if not isinstance(value, tuple)
    }


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


def _map_key(name: str) -> str:
    """The key a folded map path names, else the column name itself."""
    ref = map_ref(name)
    return ref[1] if ref is not None else name


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
        rows_inputs: Sequence[str] = (),
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
            rows_inputs=list(rows_inputs),
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
        probe_path: ProbePath = probe_one_path,
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
        self.probe_path = probe_path
        # (module, function, sorted args) -> result, so two calls with the
        # same arguments run the module once per compile.
        self._invoke_cache: dict[tuple[str, str, tuple[tuple[str, object], ...]], object] = {}
        # What a compile-time value may read: no graph, and no mutable state
        # but the memo. `path_of` and `known_hint` answer from the graph as it
        # stands, which is why they are callbacks and the graph is not a field.
        self._eval_ctx = _EvalContext(
            res=res,
            probes=probes,
            describes=self.describes,
            invoke=invoke,
            invoke_cache=self._invoke_cache,
            path_of=self._path_of,
            known_hint=self._known_hint,
        )
        # id(VARIADIC array expression) -> its lowered value. The classifier
        # (:meth:`_classify`) needs a VARIADIC call's real element type to
        # answer a nested `concat`'s kind, which means lowering the array
        # once there; the cache is what keeps the call's own later lowering
        # (:meth:`_variadic_array`) from doing that work, and any node it
        # creates, a second time.
        self._variadic_array_cache: dict[int, _Value] = {}
        # The refusals this branch's VARIADIC calls deferred by lowering an
        # aggregate over no rows to a NULL cell (:meth:`_variadic_array`).
        self.empty_aggregates: list[FfrwdError] = []
        # The `input()` aliases this branch's own FROM bound directly
        # (`_InputBinding`), read by `_lower_query` once the branch's columns
        # say whether it wrote anything.
        self.branch_input_aliases: frozenset[str] = frozenset()
        self.on_warning = on_warning
        self.graph = Graph(input_paths=list(res.input_paths), sources=dict(res.sources))
        self.ctx = _NodeFactory(self.graph)
        self.cte_columns: dict[str, tuple[_Column, ...]] = {}
        # The VALUE columns of each CTE body, name -> column -> one value per
        # body row. Filled as each body lowers, read when its alias binds.
        self.cte_values: dict[str, dict[str, tuple[RowValue, ...]]] = {}
        # The value columns the query being lowered has collected so far.
        self.branch_values: dict[str, tuple[RowValue, ...]] = {}
        # The rows-bearing stream columns of each CTE body, name -> its
        # (producer, kind, record) triple. Filled as each body lowers, read
        # when its alias binds -- the `_CteBinding.rows_columns` a rows
        # function's argument resolves through.
        self.cte_rows_columns: dict[
            str, dict[str, tuple[FrameRef, StreamType, Annotation]]
        ] = {}
        # The rows-bearing columns the query being lowered has collected so
        # far, keyed by column name -- `_lower_rows_projection` and
        # `_lower_rows_call` record here (via `rows_producers`, below) as
        # each stream column of the branch lowers.
        self.branch_rows_columns: dict[str, tuple[FrameRef, StreamType, Annotation]] = {}
        # unwrapped SELECT-list node id -> its (producer, kind, record)
        # triple, set by `_lower_rows_projection` / `_lower_rows_call` on
        # success and consumed once, in `_lower_branch`, by the projection
        # that produced it -- the bridge from a stream column's OWN lowering
        # back to the column loop, which alone knows the column's name.
        self.rows_producers: dict[int, tuple[FrameRef, StreamType, Annotation]] = {}
        # The compile-time cue-array columns of each CTE body -- the names
        # only, so a rows function fed one refuses it as what it is (a
        # written document) rather than as a stream.
        self.cte_cue_columns: dict[str, frozenset[str]] = {}
        self.branch_cue_columns: set[str] = set()
        # Inputs this pass minted itself (`ffrwd.empty_captions()`),
        # alias -> its INTERNAL input options. Merged into `Graph.input_options`
        # by `_lower_input_options`, which is the only writer of that field.
        self.minted_input_options: dict[str, dict[str, object]] = {}
        # The metadata tracks this pass minted -- a written cue or embedding
        # array, a module's rows -- each mapped to the `-metadata:s:` keys it
        # carries. A SELECT alias over one adds `title`; a vector track
        # already carries its `vector_dims`. Read by `_outputs`.
        self.minted_track_meta: dict[FrameRef, dict[str, str]] = {}
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
        # True while a CTE body lowers. The bodies are lowered once, before
        # any COPY, so the reader is not known here: a stream column records
        # one cell per row of the body's relation -- NULL where the row
        # carries no track of the kind -- and each sink decides what a NULL
        # cell means (an absent variant, an empty table cell, a rejection).
        self.cte_body = False
        # True while `array_agg`'s own argument lowers: a NULL cell of the
        # CTE column it names is dropped rather than refused, the one place
        # this dialect departs from Postgres's own array_agg (which keeps
        # NULLs) -- an outer join's gap is not a track.
        self.array_agg_reads_nulls = False
        # Node id -> one {"row": int, "rendition": {...}} per pad, in the
        # same row-major order `_packet_sink_pads` builds a row-reading
        # sink's pads in. Read there to fold `row`/`rendition` into each
        # pad's dict; empty for the old, stream-parameter sink form.
        self.row_reading_sink_pads: dict[str, list[dict[str, object]]] = {}
        # The same sinks' rows, for cutting a per-row option list to the rows
        # that carry the option's kind.
        self.row_reading_sink_rows: dict[str, list[_VariantRow]] = {}

    # -- entry point ------------------------------------------------------

    def _lower_ctes(self) -> None:
        """Lower every CTE body once, recording what each one exposes.

        The bindings come first and are shared: ``res.ctes`` holds a script's
        views AND every COPY's own ``WITH``, in written order, and each is
        lowered into THIS graph exactly once. A view read by three COPYs
        therefore mints its nodes once and hands the same refs to all three --
        the fan-out is the split pass's ordinary business, which is the whole
        point of the ABR ladder compiling to one ffmpeg command.

        `cte_body` is set for the duration: lowered once, before any COPY, a
        body has no reader to ask, so its stream columns record one cell per
        row of its own relation.
        """
        self.cte_body = True
        try:
            for name, body in self.res.ctes.items():
                self.branch_values = {}
                self.branch_rows_columns = {}
                self.branch_cue_columns = set()
                self.cte_columns[name] = tuple(
                    self._lower_query(union_branches(body), body, tags="rows")
                )
                self.cte_values[name] = self.branch_values
                self.cte_rows_columns[name] = self.branch_rows_columns
                self.cte_cue_columns[name] = frozenset(self.branch_cue_columns)
                self.branch_values = {}
                self.branch_rows_columns = {}
                self.branch_cue_columns = set()
                self._harvest_cte_tags(body)
                self._harvest_cte_dispositions(body)
        finally:
            self.cte_body = False

    def run(self) -> Graph:
        """Lower every CTE/view once, then one :class:`SinkUnit` per COPY.

        ``res.select`` / ``res.branches`` are read for the BARE-SELECT case
        only (a query with no COPY at all, which is the one unit whose path
        is None). When there are sinks they are just a mirror of ``sinks[0]``
        and walking them again would lower the first group twice.
        """
        self._lower_ctes()
        if self.res.sinks:
            self.graph.sinks = self._lower_sinks()
            if self.fanout_count is None:
                _check_two_pass_is_single_sink(self.graph.sinks, self.res.sinks)
        else:
            columns = self._lower_query(self.res.branches, self.res.select, tags="sink")
            self.graph.sinks = [
                SinkUnit(
                    outputs=_outputs(
                        columns,
                        self._layered_tags(),
                        self._layered_dispositions(),
                        self.minted_track_meta,
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
        self.rows_file = _rows_file(self.res, raw)
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
        outputs = _outputs(
            columns,
            self._layered_tags(),
            self._layered_dispositions(),
            self.minted_track_meta,
        )
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
        _check_per_track_options(options, option_nodes, outputs, raw)
        path = raw.path
        if raw.path_expr is not None:
            self._check_fanout_options(options, raw)
            path = self._sink_path(raw)
        if variant_rows is not None and path is not None:
            path = self._derive_manifest(
                options, option_nodes, columns, variant_rows, outputs, path, raw
            )
        self._check_metadata_track_container(options, outputs, path, raw)
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
            _packet_sink_audio_codec(declared, described, options, option_nodes, raw)
        return options

    def _check_metadata_track_container(
        self,
        options: dict[str, object],
        outputs: list[Output],
        path: str | None,
        raw: RawSink,
    ) -> None:
        """Refuse a titled or vector track into a container that loses it.

        A metadata track is written to be READ again, and what makes that
        possible is the tags beside it: the title the column's alias wrote,
        and a vector track's ``vector_dims``. Matroska keeps both verbatim;
        mp4 renames a title and drops any tag it has no field for, so the
        file would come back nameless or unreadable. An UNTITLED caption
        track is unaffected -- it says nothing about itself to lose.
        """
        if path is None:
            return
        container = _container_of(options, path)
        if container in _MATROSKA_FORMATS:
            return
        for output in outputs:
            metadata = self.minted_track_meta.get(output.ref)
            if not metadata:
                continue
            if VECTOR_DIMS_TAG in metadata:
                raise _error(
                    ErrorCode.UNSUPPORTED_SQL,
                    f"'{path}' is {container}, and a vector track is "
                    f"Matroska's: no other container keeps its "
                    f"{VECTOR_DIMS_TAG} tag, so the rows could not be read "
                    "back",
                    raw.path_node,
                    hint="write the file as .mkv, or drop the vector column",
                )
            title = metadata.get(STREAM_TITLE_TAG)
            if title is not None:
                raise _error(
                    ErrorCode.UNSUPPORTED_SQL,
                    f"'{path}' is {container}, and a titled track is "
                    f"Matroska's: no other container keeps '{title}' as the "
                    "title the track is found again by",
                    raw.path_node,
                    hint=f"write the file as .mkv, or drop the '{title}' "
                    "alias -- an untitled track writes to any container that "
                    "carries captions",
                )

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
                _eval_list_element(self._eval_ctx, node, env, row, anchor)
                for row in self.sink_rows
            ]
        env = self.fanout_env if self.fanout_env is not None else _Env()
        return _eval_list_element(self._eval_ctx, node, env, self.fanout_row, anchor)

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
                if value.is_array or len(streams) != 1:
                    raise _error(
                        ErrorCode.ROW_COUNT_MISMATCH,
                        f"'{label}' does not carry one stream per row, and each "
                        f"row of {subject} is one variant map entry",
                        anchor,
                        hint="select a row column (a joined CTE's, an unnest "
                        "table's), or a single stream, which every row "
                        "repeats; a gathered array belongs to a single file",
                    )
                # One value over N rows is that value on every row, which is
                # what puts the same audio under every video rung.
                streams = streams * cardinality
                column = replace(column, value=_array(value.type, streams), splat=True)
            if cardinality == 1 and len(streams) > 1:
                raise _error(
                    ErrorCode.ROW_COUNT_MISMATCH,
                    f"'{label}' is {len(streams)} streams in one row, and "
                    f"each row of {subject} is one variant map entry",
                    anchor,
                    hint=_RENDITION_PICK_HINT
                    if _from_rendition_table(self.sink_env)
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
            name = _rendition_name(video) if video is not None else _rendition_name(audio)
            language = (
                (_stream_language(audio) or _rendition_language(audio))
                if audio is not None
                else None
            )
            rows.append(_VariantRow(video=video, audio=audio, name=name, language=language))
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
        default_row = _default_audio_row(variant_rows)
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
                if row.language is not None:
                    parts.append(f"language:{row.language}")
                if position == default_row:
                    parts.append("default:yes")
            if row.audio is not None:
                audio_seen += 1
            entries.append(",".join(parts))
        return " ".join(entries)

    def _variant_names(
        self, variant_rows: list[_VariantRow]
    ) -> tuple[list[str], list[str]]:
        """Names for the map's entries, per kind, in output order.

        A row whose cell is an unmodified read of a rendition row (`name`,
        set in :meth:`_row_cells`) copies that row's own name verbatim.
        Everything else keeps the computed name it always had: video its
        height (``1080p``), audio its language tag. A name that cannot be
        computed, or that collides with another one -- video or audio, both
        share the one ``%v`` directory namespace -- falls back to its
        position (``v0``, ``a1``): files carry ``und`` more often than not,
        and a copied name can repeat across kinds (an audio-only row's cell
        reading the same rendition a video row's name came from), so names
        must exist and must not collide with ANY other name in the map.
        """
        video_rows = [row for row in variant_rows if row.video is not None]
        audio_only_rows = [
            row for row in variant_rows if row.audio is not None and row.video is None
        ]
        # A muxed row's audio never names anything, but it still numbers.
        heights = [self._output_height(row.video.ref) for row in video_rows if row.video]
        video_candidates = [
            row.name if row.name is not None else (None if h is None else f"{h}p")
            for row, h in zip(video_rows, heights)
        ]
        audio_candidates = [
            row.name if row.name is not None else _stream_language(row.audio)
            for row in audio_only_rows
            if row.audio is not None
        ]
        counts = _name_counts(video_candidates, audio_candidates)
        video = _fallback_names(video_candidates, "v", counts)
        named = _fallback_names(audio_candidates, "a", counts)
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
            "hls_segment_filename", f"{prefix}%v/segment_%d.{extension}"
        )
        if options.get("hls_segment_type") == "fmp4":
            options.setdefault("hls_fmp4_init_filename", "init.mp4")
        return f"{prefix}%v/index.m3u8"

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
        _disable_scene_cuts(options)

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
            if not _reads_unbound_rendition_column(expression, bound_env):
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
        value = _eval_value(self._eval_ctx, expression, env, self.fanout_row, anchor)
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

    def _check_path_segment(
        self, segment: exp.Expr, env: _Env, raw: RawSink, anchor: exp.Select
    ) -> None:
        """One computed piece of a path: no separator, no ``..``."""
        value = _eval_value(self._eval_ctx, segment, env, self.fanout_row, anchor)
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
            if _eval_value(self._eval_ctx, sub, env, self.fanout_row, anchor) is None:
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
        records = _chapter_records(self._eval_ctx, value, env, select)
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
        written = _attachment_records(self._eval_ctx, _unwrap(projection), env, select)
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
                _check_realtime_option(self.res, alias, options, raw_options)
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
        """Every branch, joined by ``concat`` where there is more than one.

        A branch whose every column is NULL -- one whose ``WHERE`` kept no
        row, so its aggregates gathered nothing -- contributes no segment and
        is dropped here, leaving the surviving branches to compile exactly as
        they would written alone. Under a sink, every branch dropping means
        the file would be empty, which is a refusal rather than a written
        file; a CTE body keeps its NULL columns instead, since recording the
        gaps is what a body is for.

        A dropped branch's own ``input()`` aliases are recorded onto
        ``Graph.dropped_aliases`` -- nothing in a branch that writes nothing
        ever reaches an alias's stream (an empty aggregate iterates zero
        rows), so its ``-i`` is a candidate for :func:`~ffrwd.emit._drop_dropped_branch_inputs`
        to prune once emit knows nothing ELSE still points at it.
        """
        if not branches:
            raise _error(ErrorCode.UNSUPPORTED_SQL, "query has no SELECT", anchor)
        self.tags = {}
        self.dispositions = {}
        self.container_tags = {}
        lowered: list[list[_Column]] = []
        branch_aliases: list[frozenset[str]] = []
        for branch in branches:
            lowered.append(self._lower_branch(branch, tags=tags))
            branch_aliases.append(self.branch_input_aliases)
        for columns, aliases in zip(lowered, branch_aliases, strict=True):
            if _writes_nothing(columns):
                self.graph.dropped_aliases |= aliases
        kept = [
            (branch, columns)
            for branch, columns in zip(branches, lowered, strict=True)
            if not _writes_nothing(columns)
        ]
        if not kept:
            if tags == "sink":
                raise _nothing_to_write_error(branches, anchor)
            return lowered[0]
        if len(kept) == 1:
            # A single branch keeps its arrays: a CTE body's array column stays
            # an array for `<cte>.<name>` to splat, broadcast over, or subscript.
            return kept[0][1]
        written = [branch for branch, _ in kept]
        surviving = [columns for _, columns in kept]
        # concat maps one input pad per column, so arrays are flattened to
        # one column per element BEFORE it sees them.
        flattened = [_flatten(columns) for columns in surviving]
        _check_concat_columns(written, flattened)
        _check_concat_signature(written, surviving, flattened)
        return self._concat(flattened)

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
        self.empty_aggregates = []
        self.branch_input_aliases = frozenset()
        env = self._scope(select)
        self.branch_input_aliases = frozenset(
            alias for alias, binding in env.bindings.items() if isinstance(binding, _InputBinding)
        )
        env.grouped = is_grouped(select)
        env.group_keys = _partition_keys(select, env)
        _check_grouped_cte_columns(select, env)
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
        per_row = any(_is_row_window(conjunct, env) for conjunct in time_conjuncts)
        if not fanout and not per_row:
            self._collect_trims(select, env, time_conjuncts)
        _filter_rows(self._eval_ctx, row_conjuncts, env, select)
        self._check_assertions(assertion_conjuncts, select)
        _order_rows(self._eval_ctx, select, env)
        _limit_rows(select, env)
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
            if (
                _projection_name(projection) == DISPOSITION_COLUMN
                and _value_column_name(projection, env, natural=False) is not None
            ):
                self._collect_disposition(projection, env, select, scope=tags)
                continue
            # Every other compile-time scalar is a VALUE column: a column of
            # the rows a CTE body produces, readable downstream. A body names
            # a bare one after the column it reads; a sink writes streams, so
            # one there has nowhere to go.
            value_name = _value_column_name(projection, env, natural=tags == "rows")
            if value_name is not None:
                if tags == "sink":
                    raise _error(
                        ErrorCode.UNSUPPORTED_SQL,
                        f"'{value_name}' is a value, and a SELECT column of a "
                        "media query is an output stream",
                        projection,
                        fallback=select,
                        hint="metadata is written by a tags column, e.g. "
                        f"STRUCT(... AS {value_name}) AS {TAGS_COLUMN}; a value "
                        "read by a TO expression or a WHERE needs no SELECT "
                        "column at all",
                    )
                self._collect_value_column(value_name, projection, env, select)
                continue
            column = _Column(
                name=_projection_name(projection),
                value=self._branch_value(projection, env, select),
                splat=_is_splat_projection(projection, env),
            )
            if column.name is not None:
                node = _unwrap(projection)
                source = self._rows_column_source(node, env)
                if source is not None:
                    self.branch_rows_columns[column.name] = source
                elif _is_cue_array_column(node, env):
                    self.branch_cue_columns.add(column.name)
            self._title_minted_track(column)
            columns.append(column)
        if not columns:
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                "every SELECT column is metadata, so the query selects no "
                "stream",
                fallback=select,
                hint="metadata rides on the file the query writes; select its "
                "tracks too, e.g. SELECT t, STRUCT('Main' AS title) AS tags",
            )
        if self.empty_aggregates and not _writes_nothing(columns):
            # Some columns of this branch still write, so the ones whose
            # aggregate gathered nothing are a hole rather than an empty
            # branch: the refusal deferred in `_variadic_array` stands.
            raise self.empty_aggregates[0]
        if tags == "sink":
            self._check_one_row_per_file(select, env)
        return columns

    def _title_minted_track(self, column: _Column) -> None:
        """A metadata track's alias is the title the output writes for it.

        The one place a SELECT alias means anything to a STREAM column: a
        written cue or embedding array is a track this pass minted, and the
        name the query gave the column is the only name that track has. A
        column over anything else keeps the alias as documentation.
        """
        if column.name is None:
            return
        for stream in column.value.streams:
            metadata = self.minted_track_meta.get(stream.ref)
            if metadata is not None:
                metadata[STREAM_TITLE_TAG] = column.name

    # -- one row, one file -------------------------------------------------

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
        (:func:`ffrwd.admit._check_row_sink_arity`).
        """
        if (
            self.fanout_expr is not None
            or self.manifest is not None
            or self.row_reading_sink
        ):
            return
        rendition_rows = _from_rendition_table(env)
        if env.grouped:
            count = len(_grouped_partitions(self._eval_ctx, env, select))
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
        groups = _grouped_partitions(self._eval_ctx, env, select)
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
        spec = _read_tags(node, env, select)
        for key, value_node in spec.entries.items():
            _check_tag_key(key, value_node, env, select)
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
                value = _eval_value(self._eval_ctx, value_node, env, row, select)
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
            value = _eval_value(self._eval_ctx, value_node, env, _group_row(env), select)
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

    def _collect_value_column(
        self, name: str, projection: exp.Expr, env: _Env, select: exp.Select
    ) -> None:
        """One VALUE column of a CTE body: its value, once per body row.

        The rows are the branch's relation, so a body cross-joined against a
        series carries one value per series row and a downstream fan-out reads
        the one its pinned row computed. `name` is what the body called it
        (:func:`_value_column_name`).
        """
        node = _unwrap(projection)
        relation = env.relation
        tuples = relation.tuples if relation is not None and relation.tuples else [{}]
        self.branch_values[name] = tuple(
            _eval_value(self._eval_ctx, node, env, row, select) for row in tuples
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
            flags = _flag_spec(
                _eval_value(self._eval_ctx, value_node, env, row, select), projection, select
            )
            for track in row.values():
                if isinstance(track, _TrackRow):
                    self._record_disposition(track.stream.source, flags, projection, select)

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
        stream array columns in v/a/s/d order for an input, video then audio
        for a rendition row (one cell per surviving row, NULL where the rung
        lacks the kind -- identical to spelling ``r.video, r.audio`` out),
        COLUMN order for a CTE, with array columns splatting. Every other row
        table (track rows, chapters, ...) refuses a star here: its fields are
        not streams (:func:`_row_star_error`).

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

        def rendition_thunk(binding: _RowBinding, kind: StreamType) -> Callable[[], _Column]:
            return lambda: _Column(name=None, value=_rendition_row_cells(binding, kind))

        for binding in self._star_bindings(qualifier, anchor, env, select):
            if isinstance(binding, _RowBinding):
                if binding.column != RENDITION_COLUMN:
                    raise _row_star_error(binding, anchor, select)
                # A rendition alias's star is its two stream arrays, one cell
                # per row -- the same expansion an input alias gets, applied
                # per row (:func:`ffrwd.rows._rendition_row_cells`, already how a
                # manifest destination and a CTE body read `r.video`/`r.audio`).
                vocabulary |= set(_RENDITION_STAR_COLUMNS)
                for rendition_kind in _RENDITION_STAR_COLUMNS:
                    slot(rendition_kind, rendition_thunk(binding, rendition_kind))
                continue
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

    def _star_names(
        self, qualifier: str, anchor: exp.Expr, env: _Env, select: exp.Select
    ) -> list[str]:
        """A table star's column headers. Static: no probe is consulted.

        A container names its array columns, a row table its record's scalar
        fields, a CTE the columns its body named, and a generated source the
        one array column its output type fills. `_star_cells` walks the very
        same lists in the same order.
        """
        _check_star_table_mode(anchor, select)
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
                    _row_metadata_cells(binding, name, anchor, select)
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
                    elements=(_stream_to_cell(self._source_stream_of(binding)),)
                )
                columns.append([cell] * cardinality)
            else:
                columns += [
                    _value_to_cells(
                        self._access(
                            env,
                            binding.name,
                            self._cte_column_value(binding, column, anchor, select),
                            anchor,
                            select,
                        ),
                        cardinality,
                        splat=_cte_cell_column(binding, column),
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
                elements=tuple(_stream_to_cell(stream) for stream in streams)
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
                    _add_values_rows(self._eval_ctx, alias, struct_values, env, select, join)
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
            rows = _merged_rows(
                self._eval_ctx, raw, self._record_rows(raw, unnest, select), env, select
            )
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
        _join_rows(self._eval_ctx, env.relation, alias, rows, join, env, select)

    # -- FROM input(<manifest>) alias, over an ABR ladder ------------------

    def _bind_renditions(
        self,
        alias: str,
        join: RawRowJoin | None,
        env: _Env,
        select: exp.Select,
        *,
        stream_aliases: Sequence[str] | None = None,
        extra: Mapping[str, RowColumnType] | None = None,
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

        `stream_aliases` names the input alias each ROW's streams belong to,
        for a URL source whose rows are each their own ``-i``; None means
        every row's streams are this alias's own, which is what a manifest's
        renditions are. `extra` is the value columns a module named beside
        the six.
        """
        result = self.probes.get(alias)
        if result is None or not result.renditions:
            return
        rows = [
            self._rendition_row(
                alias if stream_aliases is None else stream_aliases[position],
                rendition,
            )
            for position, rendition in enumerate(result.renditions)
        ]
        if env.relation is None:
            env.relation = _RowRelation()
        env.bindings[alias] = _RowBinding(
            alias=alias,
            source=alias,
            column=RENDITION_COLUMN,
            type=rows[0].stream.type,
            relation=env.relation,
            extra=dict(extra or {}),
        )
        _join_rows(self._eval_ctx, env.relation, alias, rows, join, env, select)

    def _rendition_row(self, alias: str, rendition: RenditionMeta) -> _TrackRow:
        """One ladder rung as a track row: its streams by kind, plus the
        ABR metadata ``RenditionMeta`` itself carries.

        `stream` is the row's primary track -- its first video stream, else
        its first audio one -- so a bare row still means one thing, exactly
        as an unnest row's does. Every stream is built the same way the
        unnest path builds one (:meth:`_source_stream`, keyed by the
        `StreamMeta`'s own per-type `index`), so emission maps it identically.

        `alias` is the input the row's STREAMS belong to, which is the row
        table's own alias for a manifest and the row's own minted ``-i`` for
        a URL source -- the rows of one table may come from several inputs.
        """
        kinds: dict[StreamType, _Stream] = {}
        for meta in rendition.streams:
            kinds.setdefault(
                meta.type,
                replace(self._source_stream(alias, meta.type, meta.index), rendition=rendition),
            )
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
                **rendition.extra,
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

        A module that is not a packet source at all but offers the export
        among its own ``functions`` is a URL SOURCE
        (:meth:`_add_url_source`): it names files rather than producing
        packets, and binds through the same rendition rows.
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
        described = _described_source(self.describes, declared, inner, select)
        if not described.source:
            self._add_url_source(
                alias, inner, declared, described, call, join, env, select
            )
            return
        params = _wasm_params(
            self._eval_ctx, declared, described, call, inner, select, env, {}, first=0
        )
        params_json = json.dumps(params, sort_keys=True)
        try:
            catalog = self.probe_source(declared.module, params_json, described=described)
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

    # -- FROM <source>(<values>) alias over a VALUES module: a URL table ----

    def _add_url_source(
        self,
        alias: str,
        inner: exp.Anonymous,
        declared: WasmFunction,
        described: Described,
        call: _Call,
        join: RawRowJoin | None,
        env: _Env,
        select: exp.Select,
    ) -> None:
        """``FROM <source>(<values>) alias`` over a module that names FILES.

        The other half of ``RETURNS source``: the module produces no packets
        and takes no sidecar, it answers with a list of urls, and ffmpeg
        opens each of them itself. The call's value arguments fold and
        type-check exactly as a value call's do, the module runs ONCE per
        distinct arguments (:attr:`_invoke_cache`), and every row of its
        answer mints one hidden ``-i`` which is then PROBED the way an
        ``input()`` path is.

        The rows bind through :meth:`_bind_renditions`, so ``s.video[1]``,
        ``s.height``, WHERE/ORDER BY/LIMIT and the one-row rule read them
        like a manifest's -- except that each row's streams belong to its own
        minted alias rather than to `alias`, which is what makes N rows N
        inputs. The alias's own :class:`~ffrwd.probe.ProbeResult` carries the
        rendition list every one of those rules reads, its streams the union
        of the rows' in row order.
        """
        found = next(fn for fn in described.functions if fn.name == declared.export)
        params = _wasm_params(
            self._eval_ctx,
            declared, described, call, inner, select, env, {},
            first=0, params_schema=found.params_schema,
        )
        answered = self._url_source_answer(
            alias, declared, described, params, inner, select
        )
        payload = _url_source_payload(alias, declared, params, answered, inner, select)
        minted: list[str] = []
        renditions: list[RenditionMeta] = []
        streams: list[StreamMeta] = []
        rows: list[UrlSourceRow] = []
        for position, row in enumerate(payload.rows, start=1):
            probed = self.probe_path(row.url)
            if probed is None:
                raise _error(
                    ErrorCode.INPUT_NOT_FOUND,
                    f"cannot read row {position} of '{alias}': '{row.url}' "
                    "could not be probed",
                    inner,
                    fallback=select,
                    hint=f"'{declared.name}()' names inputs, read exactly as "
                    "input('<path>') reads one; check the url it produced",
                )
            minted_alias, index = self._mint_input_slot(alias, row.url)
            self.probes[minted_alias] = (
                probed if payload.bounded else replace(probed, live=True)
            )
            minted.append(minted_alias)
            video = next((s for s in probed.streams if s.type == "video"), None)
            renditions.append(
                RenditionMeta(
                    streams=list(probed.streams),
                    bandwidth=row.bandwidth,
                    width=video.width if video is not None else None,
                    height=video.height if video is not None else None,
                    codecs=row.codecs,
                    name=row.name,
                    language=row.language,
                    program_id=None,
                    extra=_scalar_columns(row.columns),
                )
            )
            streams.extend(probed.streams)
            rows.append(
                UrlSourceRow(url=row.url, input=index, columns=_scalar_columns(row.columns))
            )
        self.probes[alias] = ProbeResult(
            streams=streams, renditions=renditions, live=not payload.bounded
        )
        env.bindings[alias] = _InputBinding(alias=alias)
        self._bind_renditions(
            alias, join, env, select, stream_aliases=minted, extra=payload.types
        )
        self.graph.url_sources[alias] = UrlSource(
            alias=alias,
            module=declared.module,
            params=json.dumps(params, sort_keys=True),
            document=payload.document,
            rows=tuple(rows),
        )

    def _url_source_answer(
        self,
        alias: str,
        declared: WasmFunction,
        described: Described,
        params: Mapping[str, object],
        node: exp.Expr,
        select: exp.Select,
    ) -> object:
        """Run the module for this call's arguments, once per compile.

        The same cache a value call uses, keyed the same way, so two reads of
        one call -- two branches, two COPYs -- cost one run. `described` is
        what grants the module its network for the run, when it imports one.
        """
        key = (declared.module, declared.export, tuple(sorted(params.items())))
        cached = self._invoke_cache.get(key, _UNCACHED)
        if cached is not _UNCACHED:
            return cached
        try:
            answered = self.invoke(
                declared.module, declared.export, dict(params), described=described
            )
        except FfrwdError as err:
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"cannot read '{alias}': {err.message}",
                node,
                fallback=select,
                hint=err.hint,
            ) from err
        self._invoke_cache[key] = answered
        return answered

    # -- joining two row tables ------------------------

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
        columns = (
            self._track_record_columns(
                raw.source, raw.column, raw.title, result, unnest, select
            )
            if raw.column in TRACK_RECORD_COLUMNS
            else _record_columns(result, raw.column)
        )
        return [
            _TrackRow(stream=_STREAMLESS_ROW, columns=row) for row in columns
        ]

    def _track_record_columns(
        self,
        source: str,
        column: str,
        title_wanted: str | None,
        result: ProbeResult,
        anchor: exp.Expr,
        select: exp.Select,
    ) -> list[dict[str, RowValue]]:
        """The rows of a ``cues`` / ``embeddings`` column, track by track.

        ffprobe reports that a file CARRIES caption tracks and never what is
        in them, so each one is demuxed to WebVTT text and read from there
        (:func:`ffrwd.probe.track_cues`). A WebVTT DOCUMENT is its own single
        nameless track, already read when it was probed, and carries no
        vectors.

        A track's ``vector_dims`` tag is what says which column it belongs
        to: with it, the blocks are vectors and the rows are `embeddings`;
        without it, they are captions and the rows are `cues`. Rows keep
        file order, and `index` counts within each track. `title_wanted` is
        a subscript naming ONE track, which the file has to carry.
        """
        path = self._path_of(source)
        rows: list[dict[str, RowValue]] = []
        titles: list[str] = []
        document = result.format_name == WEBVTT_FORMAT
        if document and title_wanted is None:
            return (
                []
                if column == EMBEDDINGS_COLUMN
                else _cue_rows(result.cues, None)
            )
        for meta in [] if document else _record_tracks(column, result):
            title = meta.metadata.get(STREAM_TITLE_TAG)
            if title is not None:
                titles.append(title)
            if title_wanted is not None and title != title_wanted:
                continue
            cues = track_cues(path, meta.index, self._input_flags(source))
            rows += self._track_rows(source, column, meta, title, cues, anchor, select)
        if title_wanted is not None and not rows:
            raise _error(
                ErrorCode.STREAM_NOT_FOUND,
                f"'{path}' carries no {column} track titled '{title_wanted}'",
                anchor,
                fallback=select,
                hint=_titled_track_hint(source, column, titles),
            )
        return rows

    def _track_rows(
        self,
        source: str,
        column: str,
        meta: StreamMeta,
        title: str | None,
        cues: Sequence[CueMeta],
        anchor: exp.Expr,
        select: exp.Select,
    ) -> list[dict[str, RowValue]]:
        """One track's blocks as its rows: captions, or the vectors in them."""
        if column == CUES_COLUMN:
            return _cue_rows(cues, title)
        dims = self._track_dims(source, meta, title, anchor, select)
        rows: list[dict[str, RowValue]] = []
        for cue in cues:
            values = _vector_values(cue.text, dims)
            if values is None:
                raise _error(
                    ErrorCode.UNSUPPORTED_SQL,
                    f"'{self._path_of(source)}' track {_track_name(title, meta)} "
                    f"row {cue.index} does not hold {dims} numbers",
                    anchor,
                    fallback=select,
                    hint=f"a vector track's rows are its {VECTOR_DIMS_TAG} numbers "
                    "as little-endian f32, base64; this one was written by "
                    "something else",
                )
            rows.append(
                {
                    "index": cue.index,
                    "track": title,
                    "start_t": cue.start_t,
                    "end_t": cue.end_t,
                    "vector": values,
                }
            )
        return rows

    def _track_dims(
        self,
        source: str,
        meta: StreamMeta,
        title: str | None,
        anchor: exp.Expr,
        select: exp.Select,
    ) -> int:
        """How many numbers one vector track's rows carry, from its own tag."""
        written = meta.metadata.get(VECTOR_DIMS_TAG, "")
        if not written.isdigit() or int(written) == 0:
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"'{self._path_of(source)}' track {_track_name(title, meta)} says "
                f"{VECTOR_DIMS_TAG}='{written}'",
                anchor,
                fallback=select,
                hint=f"{VECTOR_DIMS_TAG} counts the numbers in each row's vector, "
                "so it is a positive whole number",
            )
        return int(written)

    def _input_flags(self, alias: str) -> tuple[str, ...]:
        """One input's own options as argv, for a second read of the file.

        The DECODE's options, not the probe's: extracting a track runs
        ffmpeg, so the file is opened the way the command opens it. An
        option lowering is about to reject reads as no flags at all, the
        same best effort probing makes.
        """
        raw_options = self.res.input_options.get(alias)
        if not raw_options:
            return ()
        try:
            return tuple(render_options(input_option_values(raw_options)))
        except (FfrwdError, ValueError):
            return ()

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

        Only that the file was readable at all: what a `cues` or
        `embeddings` column then finds in it is a property of its TRACKS,
        which :meth:`_track_record_columns` reads one at a time.
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
        return result

    def _add_table(
        self,
        table: exp.Expr | None,
        join: RawRowJoin | None,
        env: _Env,
        select: exp.Select,
    ) -> None:
        if isinstance(table, exp.Subquery):
            # `FROM (<select>) alias`: resolve bound the body under the alias,
            # so it reads back as the CTE it is.
            alias_node = table.args.get("alias")
            if not isinstance(alias_node, exp.TableAlias) or alias_node.this is None:
                raise _error(  # defensive: resolve already required a name
                    ErrorCode.UNSUPPORTED_SQL,
                    "a subquery in FROM needs a name",
                    table,
                    fallback=select,
                )
            local = _fold(alias_node.this)
            self._add_cte_table(local, local, table, select, join, env, select)
            return
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
            _add_series_rows(self._eval_ctx, alias, series_values, inner, env, select, join)
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
            # `FROM master m` binds the view/CTE under a BRANCH-LOCAL name
            # (resolve checked it shadows nothing in the flat namespace). The
            # binding records the local name, so `m.v` resolves and messages
            # read back as written; the columns -- and therefore the graph
            # refs -- are the same objects either way, which is what makes the
            # shared subgraph shared.
            name = _fold(inner)
            local = name
            if isinstance(alias_node, exp.TableAlias) and alias_node.this is not None:
                local = _fold(alias_node.this)
            self._add_cte_table(name, local, inner, table, join, env, select)
            return
        raise _error(
            ErrorCode.UNSUPPORTED_SQL,
            _FROM_ITEM_MESSAGE,
            table,
            fallback=select,
        )

    def _add_cte_table(
        self,
        name: str,
        local: str,
        anchor: exp.Expr,
        fallback: exp.Expr,
        join: RawRowJoin | None,
        env: _Env,
        select: exp.Select,
    ) -> None:
        """Bind one named relation's rows: a view, a CTE, or a FROM subquery."""
        columns = self.cte_columns.get(name)
        if columns is None:
            raise _error(
                ErrorCode.UNKNOWN_ALIAS,
                f"unknown table '{name}'",
                anchor,
                fallback=fallback,
                hint=self._known_hint(),
            )
        _add_cte_rows(
            self._eval_ctx,
            local,
            columns,
            self.cte_values.get(name, {}),
            self.cte_rows_columns.get(name, {}),
            self.cte_cue_columns.get(name, frozenset()),
            env,
            select,
            join,
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
        source = _source_filter(self.registry, raw, select)
        named = [_NamedArg(name=option.name, value=option.value) for option in raw.options]
        options = (
            _filter_options(self.registry, raw.name, raw.call_node, select) if named else {}
        )
        # No `timeline=`: SourceFilter has no such field, because a generator
        # is never timeline-capable -- there is no upstream frame to switch
        # on/off. `enable => ...` on a source rejects unconditionally.
        dropped: dict[str, exp.Expr] = {}
        args = _check_named_args(
            raw.name,
            options,
            named,
            raw.call_node,
            owner=f"{FILTER_NAMESPACE}.{raw.name}",
            occupied=set(),
            dropped=dropped,
        )
        _check_required_options(raw.name, args, dropped, raw.call_node, select)
        env.bindings[alias] = _SourceBinding(
            alias=alias, name=raw.name, output=source.output, options=args
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
            if aliases - rows and len(rows) == 1 and _is_row_window(conjunct, env):
                # A time window whose BOUNDS are row columns: one seek per row.
                # A fan-out TO gives each row a file; without one the rows stay
                # in this graph and each seeks its own `-i` of the same file.
                self._check_row_window_seeks_a_file(conjunct, where, env)
                time_conjuncts.append(conjunct)
                continue
            if aliases - rows:  # defensive: resolve rejected the mix already
                raise _error(
                    ErrorCode.UNSUPPORTED_SQL,
                    "a WHERE predicate cannot mix track-row columns with other aliases",
                    conjunct,
                    fallback=where,
                    hint="write them as separate AND conjuncts",
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

    # -- compile-time row filtering / ordering -------------------

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
        groups = _fanout_groups(self._eval_ctx, env, select)
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

    # -- subscript metadata WHERE assertions --
    #
    # `<alias>.<type>[k].<column>` names ONE probed track deterministically
    # (the subscript is bounds-checked, not filtered), so a WHERE conjunct over
    # it has nothing to DROP the way a row predicate drops rows. It is an
    # ASSERTION, checked once at compile time against the probed file: TRUE
    # proceeds unchanged, FALSE or UNKNOWN (3VL -- a field that was never
    # probed) is a typed rejection, because an ffmpeg command line cannot
    # encode "select nothing" (recipe 29 of docs/corpus.md).
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
            low = _literal_of(node.args.get("low"), select)
            high = _literal_of(node.args.get("high"), select)
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
                    _literal_of(right_node, select),
                )
            mirrored = _MIRRORED_COMPARISONS[type(node)]()
            return _compare(
                mirrored,
                self._accessor_value(right_node, select),
                _literal_of(left_node, select),
            )
        raise _error(
            ErrorCode.UNSUPPORTED_SQL, "unsupported WHERE predicate", node,
            fallback=select,
        )

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
            node is not None and _reads_row_alias(node, env)
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
        value = _eval_value(
            self._eval_ctx, bound, env, self.fanout_row if rows is None else rows, select
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
        # An array of embedding records is a track the same way, its rows
        # carrying vectors instead of captions.
        vectors = self._lower_embedding_array(node, env, select)
        if vectors is not None:
            return vectors
        # A module's annotation column is a track too, minted from the rows
        # instead of from a written document.
        rows = self._lower_rows_projection(node, env, select)
        if rows is not None:
            return rows
        # A rows function's result is that same column, one module later.
        rewritten = self._lower_rows_call(node, env, select)
        if rewritten is not None:
            return rewritten
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
        if isinstance(node, exp.Filter) and isinstance(node.this, exp.ArrayAgg):
            # FILTER (WHERE <col> IS NOT NULL): parser already confirmed the
            # predicate names the aggregated column, so it adds nothing here
            # -- array_agg skips that NULL cell on its own.
            return self._lower_array_agg(node.this, env, select)
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
        unnest row's own stream. A ``[1]`` buried inside a larger expression
        (``scale(r.video[1], 320, -2)``) gets the same reading one layer
        down, in :meth:`_lower_rendition_agg_expr`.

        A bare CTE stream column (``array_agg(vid.v)``) is the other
        exception: outside the aggregate a NULL cell -- an outer join's gap
        -- is a typed rejection, but `array_agg` skips it instead
        (:meth:`_cte_array_agg`), the one place this dialect departs from
        Postgres's own array_agg, which keeps NULLs. ``FILTER (WHERE
        vid.v IS NOT NULL)`` on the same column is the explicit spelling of
        the same thing -- parser admits only that predicate, and lowering
        never distinguishes it from the bare form.

        An aggregate that gathers NOTHING -- a branch whose ``WHERE`` kept no
        row, or a join every one of whose cells is a gap -- is NULL, as in
        Postgres, rather than a zero-length array. A NULL column is one
        ``COPY`` does not write (:meth:`_lower_query`).
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
        gathered = self._rendition_array_agg(inner, env, select)
        if gathered is None:
            gathered = self._cte_array_agg(inner, env, select)
        if gathered is None:
            gathered = self._lower_rendition_agg_expr(inner, env, select)
        if gathered.is_array and not gathered.streams:
            return _scalar(_Stream(ref=_NULL_STREAM_REF, type=gathered.type))
        return gathered

    def _cte_array_agg(
        self, inner: exp.Expr, env: _Env, select: exp.Select
    ) -> _Value | None:
        """``array_agg(<CTE stream column>)``: every surviving row's cell,
        an outer join's gap dropped instead of refused.

        None for anything else -- a plain column outside a CTE binding, a
        subscript, a call -- so :meth:`_lower_array_agg` falls back to the
        ordinary lowering, where a NULL cell still refuses. That narrower
        reach is deliberate: a NULL cell buried inside a filter call
        (``array_agg(scale(vid.v, 320, -2))``) would otherwise feed the
        filter a stream that is not there.
        """
        column = _unwrap(inner)
        if not isinstance(column, exp.Column):
            return None
        table_node = column.args.get("table")
        if table_node is None:
            return None
        binding = env.bindings.get(_fold(table_node))
        if not isinstance(binding, _CteBinding):
            return None
        previous = self.array_agg_reads_nulls
        self.array_agg_reads_nulls = True
        try:
            value = self._lower_expr(column, env, select)
        finally:
            self.array_agg_reads_nulls = previous
        if not value.is_array:
            return value
        streams = [stream for stream in value.streams if stream.ref != _NULL_STREAM_REF]
        return _array(value.type, streams)

    def _lower_rendition_agg_expr(
        self, inner: exp.Expr, env: _Env, select: exp.Select
    ) -> _Value:
        """``array_agg(<expression over r.video[1] / r.audio[1]>)``: the
        expression evaluated once per surviving row, the same broadcast a
        track row's bare ``array_agg(volume(a, 0.5))`` already gets.

        Outside an aggregate, ``r.video[1]`` picks ONE rendition out of the
        surviving set (:meth:`_rendition_kind_value`) -- the right reading
        for a bare column. Under `array_agg`, the sole rendition row alias in
        scope (if there is exactly one) is marked in `env` for the length of
        this lowering, so that reading flips to the array `array_agg(r.video[1])`
        already gets: every surviving row's own stream, in row order. From
        there the ordinary broadcast machinery (:meth:`_expand_call`) is what
        turns ``scale(r.video[1], 320, -2)`` into one `scale` node per
        element, exactly as it already does for any other array argument --
        no new mechanism, just the same subscript reading a bare
        ``array_agg(r.video[1])`` uses.

        More than one rendition row alias in scope is ambiguous (which one is
        `array_agg` naming?), so it is left alone -- the expression lowers
        exactly as it would outside the aggregate.
        """
        aliases = [
            alias
            for alias, binding in env.bindings.items()
            if isinstance(binding, _RowBinding) and binding.column == RENDITION_COLUMN
        ]
        if len(aliases) != 1:
            return self._lower_expr(inner, env, select)
        previous = env.rendition_agg
        env.rendition_agg = aliases[0]
        try:
            return self._lower_expr(inner, env, select)
        finally:
            env.rendition_agg = previous

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
        cues = _cue_records(self._eval_ctx, node, env, select)
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
        ref = self._mint_webvtt_input(CUES_COLUMN, text, {})
        return _scalar(_Stream(ref=ref, type="subtitle", source=None))

    def _mint_webvtt_input(
        self, name: str, text: str, metadata: dict[str, str]
    ) -> FrameRef:
        """One written WebVTT document as a self-contained ``-i``, and its ref.

        `metadata` is what the track says about itself beyond its title --
        a vector track's ``vector_dims`` -- and is recorded for the output to
        emit; the title itself lands later, from the SELECT column's alias.
        """
        uri = "data:text/vtt;base64," + base64.b64encode(text.encode()).decode()
        ref = self._mint_stream_input(name, uri, WEBVTT_FORMAT, "subtitle")
        self.minted_track_meta[ref] = metadata
        return ref

    # -- an embedding array as a vector track -------------------------------

    def _lower_embedding_array(
        self, node: exp.Expr, env: _Env, select: exp.Select
    ) -> _Value | None:
        """``ARRAY[STRUCT(...)::embedding, ...]`` / ``array_agg(...)`` as a track.

        A vector track rides in the same WebVTT document a caption track does
        -- one block per row, the row's bounds as the block's timing -- with
        each block's text the vector itself, little-endian f32 in base64. The
        track's ``vector_dims`` tag says how many numbers that is, which is
        what reads them back.
        """
        rows = _embedding_records(self._eval_ctx, node, env, select)
        if rows is None:
            return None
        if not rows:
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"{article(EMBEDDING_TYPE)} {EMBEDDING_TYPE} array is empty, so "
                "there is no vector track to write",
                node,
                fallback=select,
                hint=f"write at least one row, e.g. ARRAY[{_EMBEDDING_EXAMPLE}], "
                "or drop the column",
            )
        dims = _embedding_dims(rows, node, select)
        text = _cues_webvtt(
            [
                _Cue(
                    start=row.start,
                    end=row.end,
                    text=_vector_payload(row.vector),
                    start_node=row.start_node,
                    end_node=row.end_node,
                )
                for row in rows
            ],
            noun=EMBEDDING_TYPE,
        )
        ref = self._mint_webvtt_input(
            EMBEDDINGS_COLUMN, text, {VECTOR_DIMS_TAG: str(dims)}
        )
        return _scalar(_Stream(ref=ref, type="subtitle", source=None))

    # -- a module's rows as a track ----------------------------------------

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
        found = _rows_projection(self.res, node)
        if found is None:
            return None
        call, declared = found
        module = self._row_filtered(self._lower_expr(call, env, select), node)
        assert declared.emits is not None  # what `_rows_projection` selected on
        self.rows_producers[id(node)] = (module.streams[0].ref, module.type, declared.emits)
        return self._rows_output(
            module.streams[0].ref,
            module.type,
            declared,
            declared.emits,
            call,
            node,
            env,
            select,
        )

    def _rows_output(
        self,
        producer: FrameRef,
        kind: StreamType,
        declared: WasmFunction,
        annotation: Annotation,
        call: exp.Anonymous,
        node: exp.Expr,
        env: _Env,
        select: exp.Select,
    ) -> _Value:
        """Where the rows `producer` writes go: a rows file, or a minted track.

        The one place a row column becomes an output, whichever node ends the
        rows -- the module that read them off its frames, or the rows module
        that ran over them afterwards. `annotation` is that end's own record
        -- `declared.emits` or `declared.returns_rows`. A vector field passed
        module to module, never written to a track, names no length at all;
        one reaching a track here needs one fixed, which is what
        :func:`ffrwd.admit._check_vector_dims` checks -- only at this, its one path to an
        output, and not at every call a vector annotation happens to pass
        through.
        """
        if self.rows_file:
            self.graph.rows_sinks[producer] = RowsSink(
                container=_ROWS_CONTAINER, path=self.rows_file
            )
            return _Value(type=kind, streams=(), is_array=False)
        described = self.describes.get(declared.module)
        assert described is not None  # the caller already described it
        _check_vector_dims(declared, annotation, described, node, select)
        tag = self._rows_language(declared, call, node, env, select)
        ref = self._mint_stream_input(
            CUES_COLUMN, PIPE, WEBVTT_FORMAT, "subtitle"
        )
        meta: dict[str, str] = {}
        field = _vector_field(annotation)
        if field is not None:
            dims = rows_vector_dims(described, field)
            assert dims is not None  # _check_vector_dims just fixed one
            meta[VECTOR_DIMS_TAG] = str(dims)
        self.minted_track_meta[ref] = meta
        self.graph.rows_sinks[producer] = RowsSink(
            container=WEBVTT_FORMAT, alias=src_alias(ref)
        )
        self.rows_tracks.append(ref)
        return _scalar(_Stream(ref=ref, type="subtitle", source=_rows_meta(tag)))

    # -- a rows function: rows in, rows out ---------------------------------

    def _lower_rows_call(
        self, node: exp.Expr, env: _Env, select: exp.Select
    ) -> _Value | None:
        """A rows function over a module's annotation column.

        The producer lowers exactly as it would on its own; the rows function
        becomes a node of its own beside it, fed by a ROWS edge naming that
        producer and carrying no frames. Its value is a row column of the
        declared return type, which is the producer's own value read one
        module later: a track where the query projects it, the rows
        themselves at a rows-file destination.
        """
        found = _rows_call(self.res, node)
        if found is None:
            return None
        call, declared = found
        ref, kind, emitted = self._rows_node(node, env, select)
        self.rows_producers[id(node)] = (ref, kind, emitted)
        return self._rows_output(ref, kind, declared, emitted, call, node, env, select)

    def _rows_node(
        self, node: exp.Expr, env: _Env, select: exp.Select
    ) -> tuple[FrameRef, StreamType, Annotation]:
        """One rows-function call as its node: the rows edge, wired and typed.

        `node` names a rows function, which the caller has already checked.
        Its argument's own node comes back from :meth:`_rows_source`, and the
        record that node emits is matched against what this declaration says
        it reads before the edge is drawn.
        """
        found = _rows_call(self.res, node)
        assert found is not None  # the caller selected on it
        call, declared = found
        _described_rows(self.describes, declared, node, select)
        arguments = [a for a in call.expressions if isinstance(a, exp.Expr)]
        if len(arguments) != 1:
            raise _error(
                ErrorCode.UDF_ARG_TYPE,
                f"{declared.name}() takes 1 argument, got {len(arguments)}",
                node,
                fallback=select,
                hint=f"a rows function reads one row column: {declared.signature}",
            )
        written = _unwrap(arguments[0])
        source = self._rows_source(written, env, select)
        if source is None:
            raise _not_rows(declared, written, node, env)
        producer, kind, emitted = source
        _check_rows_argument(declared, emitted, written, node, select)
        assert declared.returns_rows is not None  # what is_rows selected on
        ref = self.ctx.node(declared.module, {}, [], [], rows_inputs=[producer])
        return ref, kind, declared.returns_rows

    def _rows_column_source(
        self, node: exp.Expr, env: _Env
    ) -> tuple[FrameRef, StreamType, Annotation] | None:
        """A stream expression's run-time annotation source, if it has one.

        Either half of `_lower_expr`'s two rows-bearing matches, read back
        from `rows_producers` rather than re-lowered (the producer already
        ran, when the expression first lowered as a stream column) -- or a
        bare reference to a CTE column that carries one, chased through
        `_CteBinding.rows_columns`. That dict is filled the same way one CTE
        level down, so a chain of CTEs resolves exactly as one does.
        """
        source = self.rows_producers.pop(id(node), None)
        if source is not None:
            return source
        cte_ref = _cte_column_ref(node, env)
        if cte_ref is None:
            return None
        binding, name = cte_ref
        return binding.rows_columns.get(name)

    def _rows_source(
        self, written: exp.Expr, env: _Env, select: exp.Select
    ) -> tuple[FrameRef, StreamType, Annotation] | None:
        """The node whose rows `written` names, and the record they carry.

        The dialect's two row producers, spelled inline: the annotation
        column a stream module reads off its frames, or another rows
        function's result. A CTE column carrying either -- directly, or
        through another CTE column that does -- resolves to the SAME
        producer `_rows_column_source` already recorded when that column
        first lowered, so the producer runs once whether its rows go to a
        track, to a rows function, or to both. None for everything else.
        """
        produced = _rows_projection(self.res, written)
        if produced is not None:
            producer_call, producer = produced
            if written.meta.get(ROW_PREDICATE) is not None:
                raise _error(
                    ErrorCode.UNSUPPORTED_SQL,
                    f"'{producer.name}' rows are narrowed before a rows function "
                    "reads them",
                    written,
                    fallback=select,
                    hint="a rows function reads every row the module produced; "
                    "drop the WHERE, or narrow the rows the function returns",
                )
            if written.meta.get(ROW_MERGE) is not None:
                raise _error(
                    ErrorCode.UNSUPPORTED_SQL,
                    f"'{producer.name}' rows are merged before a rows function "
                    "reads them",
                    written,
                    fallback=select,
                    hint="a rows function reads every row the module produced; "
                    f"drop the {MERGE_CUES}(), or merge what the function "
                    "returns",
                )
            module = self._lower_expr(producer_call, env, select)
            assert producer.emits is not None  # what _rows_projection selected on
            return module.streams[0].ref, module.type, producer.emits
        if _rows_call(self.res, written) is not None:
            return self._rows_node(written, env, select)
        return self._rows_column_source(written, env)

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
            value = _eval_value(self._eval_ctx, written, env, row, select)
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
                # `_rendition_kind_value`. Under `array_agg` over this same
                # alias (`env.rendition_agg`), a `[1]` subscript reads that
                # same array instead of picking one row out of it -- see
                # `_lower_rendition_agg_expr`.
                agg_index = None if index == 1 and env.rendition_agg == alias else index
                return binding.source, self._rendition_kind_value(
                    binding, name, agg_index, anchor, select
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

        A reader that counts rows reads it per ROW instead
        (:func:`ffrwd.rows._rendition_row_cells`): a manifest destination, whose every
        row is one variant map entry; a row-reading sink; and a CTE body,
        which carries its rows to whichever of those reads it later. An
        audio-only rung's video cell is NULL there, not absent -- the same
        gap a FULL JOIN's unmatched row leaves.

        A rendition carries at most one stream of a kind, so only ``[1]``
        can name one per row; every other subscript keeps the reading above,
        which is where its bounds check lives.
        """
        kind = _ARRAY_COLUMNS[name]
        per_row = self.manifest is not None or self.row_reading_sink or self.cte_body
        if per_row and index in (None, 1):
            return _rendition_row_cells(binding, kind)
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
        streams = _per_row_seeks(
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
            f"{_unmatched_text(binding, position)}",
            anchor,
            fallback=select,
            hint=hint,
        )

    # -- COALESCE(<nullable cell>, <stand-in>) -----------------

    def _lower_coalesce(self, node: exp.Expr, env: _Env, select: exp.Select) -> _Value:
        """The accepted spelling for a nullable stream cell: fill its gaps.

        Two readings of one rule, the first non-NULL per row. A track-row
        alias knows its own relation, so its gaps are filled row by row
        against the row that DID match (:meth:`_lower_row_coalesce`). Every
        other nullable cell -- a CTE's or subquery's stream column, a
        rendition array subscript -- carries its gaps in the value itself
        (:meth:`_lower_cell_coalesce`), which is where a second nullable
        column can stand in for the first.
        """
        arguments = _coalesce_arguments(node, select)
        binding = _coalesce_binding(arguments[0], env)
        if binding is None:
            return self._lower_cell_coalesce(arguments, env, node, select)
        return self._lower_row_coalesce(binding, arguments[1], env, node, select)

    def _lower_row_coalesce(
        self,
        binding: _RowBinding,
        fill: exp.Expr,
        env: _Env,
        node: exp.Expr,
        select: exp.Select,
    ) -> _Value:
        """A track-row alias's gaps, filled row by row.

        The result is the same N-element array ``<alias>`` is, in the
        same row order — every gap replaced by what stands in for it: a cell
        of another stream column, or a generated stand-in. Only the gaps mint
        anything: a join with no unmatched rows compiles to exactly the
        command the bare column would -- consume-once here means "generate
        nothing nobody needed".
        """
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
        cardinality = len(relation.tuples)
        matched = [
            self._matched_stream(binding, row, env, node, select)
            for row in relation.tuples
        ]
        fill_call = _fill_call(fill)
        if fill_call is None:
            stand_in = self._lower_expr(fill, env, select)
            _check_coalesce_fill(
                binding.type, stand_in, binding.alias, fill, select
            )
            _check_coalesce_width(stand_in, cardinality, fill, select)
            streams = [
                stream
                if stream is not None
                else stand_in.at(min(position, len(stand_in.streams) - 1))
                for position, stream in enumerate(matched)
            ]
        else:
            streams = [
                stream
                if stream is not None
                else self._lower_fill(
                    fill,
                    fill_call,
                    binding.type,
                    binding.alias,
                    _paired_row(relation, row, binding.alias)[1],
                    node,
                    select,
                )
                for row, stream in zip(relation.tuples, matched, strict=True)
            ]
        return _array(binding.type, streams)

    def _matched_stream(
        self,
        binding: _RowBinding,
        row: _RowTuple,
        env: _Env,
        node: exp.Expr,
        select: exp.Select,
    ) -> _Stream | None:
        """One row's own track, or None where the join left a gap.

        The real track goes through `_access` exactly as a bare ``<alias>``
        would, so the input's WHERE window (and the caption-seek rejection)
        still applies to it.
        """
        track = _track_of(row, binding.alias)
        if track is None:
            return None
        return self._access(
            env, binding.source, _scalar(track.stream), node, select
        ).streams[0]

    def _lower_cell_coalesce(
        self,
        arguments: list[exp.Expr],
        env: _Env,
        node: exp.Expr,
        select: exp.Select,
    ) -> _Value:
        """Any other nullable stream cell's gaps: the first non-NULL per row.

        Both arguments lower as they stand, so each is already one cell per
        row (or one value the rows repeat), and the result is the first
        column with every gap taken from the second. An all-NULL row stays
        NULL -- a manifest reads it as an absent stream kind, the same as any
        other gap. A generated stand-in is still a legal second argument, and
        then only the gaps mint one.
        """
        first = self._lower_expr(arguments[0], env, select)
        label = _coalesce_label(arguments[0])
        if not first.streams:
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                "COALESCE's first argument is a nullable stream cell, got "
                f"{_describe(arguments[0])}",
                arguments[0],
                fallback=select,
                hint=_COALESCE_HINT,
            )
        second = arguments[1]
        fill_call = _fill_call(second)
        cardinality = len(first.streams)
        filled: list[_Stream] = []
        if fill_call is not None:
            filled = [
                cell
                if cell.ref != _NULL_STREAM_REF
                else self._lower_fill(
                    second, fill_call, first.type, label, None, node, select
                )
                for cell in first.streams
            ]
        else:
            other = self._lower_expr(second, env, select)
            _check_coalesce_fill(first.type, other, label, second, select)
            cardinality = max(cardinality, len(other.streams))
            _check_coalesce_width(first, cardinality, arguments[0], select)
            _check_coalesce_width(other, cardinality, second, select)
            for position in range(cardinality):
                cell = first.at(min(position, len(first.streams) - 1))
                if cell.ref == _NULL_STREAM_REF:
                    cell = other.at(min(position, len(other.streams) - 1))
                filled.append(cell)
        if cardinality == 1 and not first.is_array:
            return _scalar(filled[0])
        return _array(first.type, filled)

    def _lower_fill(
        self,
        node: exp.Expr,
        call: _Call,
        kind: StreamType,
        label: str,
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
        (:func:`ffrwd.fills._inherited_fill_options`) and its provenance, so a
        silence-filled French mix is still tagged French.
        """
        source_meta = paired.stream.source if paired is not None else None
        name = call.name.lower()
        if call.is_macro:
            return self._lower_macro_fill(
                node, name, call, kind, label, source_meta, select
            )
        source = _source_filter(
            self.registry, RawSource(alias="", name=name, options=(), call_node=node), select
        )
        _check_fill_type(source.output, call.display, kind, label, node, select)
        options = _filter_options(self.registry, name, node, select)
        dropped: dict[str, exp.Expr] = {}
        args = _check_named_args(
            name,
            options,
            call.named,
            node,
            owner=f"{FILTER_NAMESPACE}.{name}",
            occupied=set(),
            dropped=dropped,
        )
        _check_required_options(name, args, dropped, node, select)
        for option, value in _inherited_fill_options(kind, paired).items():
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
        kind: StreamType,
        label: str,
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
                hint=_fill_hint(kind, label),
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
        _check_fill_type(macro.output, call.display, kind, label, node, select)
        return _Stream(
            ref=self._mint_input(macro), type=macro.output, source=source_meta
        )

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
        alias, _ = self._mint_input_slot(name, path)
        self.minted_input_options[alias] = {"format": format_}
        return f"src:{alias}:{_TYPE_MARKERS[output]}:0"

    def _mint_input_slot(self, name: str, path: str) -> tuple[str, int]:
        """One compiler-minted ``-i``: its alias and its ffmpeg input index.

        The slot alone, with no options on it -- the input is opened exactly
        as ``input(path)`` opens one. See :meth:`_mint_stream_input` for why
        the alias is spelled the way it is.
        """
        index = len(self.graph.input_paths)
        alias = f"{MACRO_NAMESPACE}.{name}#{index + 1}"
        self.graph.input_paths.append(path)
        self.graph.sources[alias] = index
        return alias, index

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
                hint=_source_columns_hint(binding),
            )
        if array_type != binding.output:
            raise _error(
                ErrorCode.STREAM_NOT_FOUND,
                f"'{binding.alias}.{name}' does not exist: {produces}",
                anchor,
                fallback=select,
                hint=_source_columns_hint(binding),
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
        column = _cte_column(binding, name)
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
                hint=_cte_columns_hint(binding),
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

        A stream column's cardinality follows the relation, so the value is
        the column re-read through the result tuples: a cross join repeats a
        stream once per partner row, a filtered relation drops the rows it
        dropped, and an outer join's gap is a NULL cell. A row-set column
        carries one stream per body row to read by position; a single value
        is that value on every row, so a one-row body joined into several
        rows is present where its row matched and NULL where it did not. A
        gathered array (an ``array_agg``, a bare input array re-exposed) is
        one unit and reads as itself.
        """
        relation = binding.relation
        if relation is None or not _cte_cell_column(binding, column):
            return column.value
        cells = (
            column.value.streams
            if column.value.is_array
            else column.value.streams * binding.rows
        )
        streams: list[_Stream] = []
        for position, entry in enumerate(
            tuple_.get(binding.name) for tuple_ in relation.tuples
        ):
            cell = (
                cells[entry.position]
                if isinstance(entry, _CteRow)
                # An outer join's gap reads as the same NULL a row carrying
                # no track of the kind leaves.
                else _Stream(ref=_NULL_STREAM_REF, type=column.value.type, source=None)
            )
            if cell.ref != _NULL_STREAM_REF or self._null_cells_are_read:
                streams.append(cell)
                continue
            missed = (
                f"no '{binding.name}' row matched here"
                if not isinstance(entry, _CteRow)
                else f"that row carries no {column.value.type} track"
            )
            raise _error(
                ErrorCode.STREAM_NOT_FOUND,
                f"'{binding.name}.{column.name}' is NULL in row "
                f"{position + 1}: {missed}",
                anchor,
                fallback=select,
                hint="only a manifest destination (WITH (format 'hls'), "
                "format 'dash') reads a NULL cell, as an absent variant; "
                "elsewhere narrow the rows with a WHERE, or join so every "
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
        if not column.value.is_array and len(streams) == 1:
            return _scalar(streams[0])  # one row in, one row out: still scalar
        return _array(column.value.type, streams)

    @property
    def _null_cells_are_read(self) -> bool:
        """True where a NULL stream cell is something the reader understands:
        a table's empty cell, a manifest row's absent stream kind, a
        row-reading sink's, a CTE body, which records its rows for one of
        those to read later, and `array_agg`'s own argument, which drops the
        cell instead of keeping it."""
        return (
            self.table_mode
            or self.manifest is not None
            or self.row_reading_sink
            or self.cte_body
            or self.array_agg_reads_nulls
        )

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
            options = _array_options(self.registry, name)
            if options is not None:
                return self._lower_array_call(
                    node, ARRAY_RETURNING[name], options, call, env, select
                )
        n_input = _n_input_call(self.registry, name)
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
                hint=_namespaced_function_hint(self.registry, name)
                if call.namespaced
                else _unknown_function_hint(self.registry, name),
            )
        target_name, target = self._dispatch_audio(name, dynamic, call, env, select)
        return self._lower_dynamic_call(node, target_name, target, call, env, select)

    # -- wasm calls --

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
        """`value` through the rows nodes written over it, or `value` itself.

        Both ride the frames' own chain: each node changes the rows the stream
        carries and hands both on, so a narrowed or merged column and a plain
        one are the same value with one more node in front of it. A gather
        narrows before ``merge_cues`` collapses, which is the order they are
        written in.
        """
        predicate = node.meta.get(ROW_PREDICATE)
        if isinstance(predicate, str):
            value = self._hosted_rows_node(value, ROWFILTER, {PREDICATE: predicate})
        distance = node.meta.get(ROW_MERGE)
        if isinstance(distance, int | float) and not isinstance(distance, bool):
            value = self._hosted_rows_node(value, ROWMERGE, {MAX_DISTANCE: distance})
        return value

    def _hosted_rows_node(
        self, value: _Value, filter: str, args: dict[str, object]
    ) -> _Value:
        """One host-provided rows node in front of every stream of `value`."""
        return replace(
            value,
            streams=tuple(
                replace(
                    stream,
                    ref=self.ctx.node(
                        filter,
                        args,
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
        described = _described(self.describes, declared, node, select)
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
        # A stream position holding a row column: the rows travel beside the
        # frames a module filters, and there are none here to travel beside.
        rows_written = next(
            (
                argument
                for argument in call.args[:arity]
                if _rows_projection(self.res, _unwrap(argument)) is not None
                or _rows_call(self.res, _unwrap(argument)) is not None
            ),
            None,
        )
        if rows_written is not None:
            raise _error(
                ErrorCode.UDF_ARG_TYPE,
                f"{declared.name}() takes {declared.params[0].type}, and its "
                "argument is rows",
                rows_written,
                fallback=node,
                hint=f"give it the stream those rows came off, and its rows "
                f"beside it: {declared.name}(<producer>(<stream>)...)"
                if declared.reads is not None
                else f"give it a stream: {declared.signature}",
            )
        expected: list[StreamType] = [kind] * arity
        kinds = self._stream_kinds(call, env, select, arity)
        if kinds != expected:
            raise _bad_streams(call, node, select, expected, kinds)
        _check_annotation_argument(self.res, declared, call, node, select)
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
            params = _wasm_params(
                self._eval_ctx,
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

        lowered = _expand_call(
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
            rows=_row_elements(per_row, env),
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
        pads = _bind_sink_streams(declared, gathered, node, select)
        # One instance, so its parameters are read once. A row-varying value
        # argument has no one row to read here; the FIRST is what a call over
        # a gathered relation means everywhere else.
        tuples = env.relation.tuples if env.relation is not None else []
        params = _wasm_params(
            self._eval_ctx,
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
                splat=_is_splat_projection(argument, env),
            )
            for argument in call.args[:at]
        ]
        cardinality = max(len(self.sink_rows), 1)
        rows, _ = self._row_cells(columns, cardinality, node, f"'{declared.name}'")
        self._check_no_null_stream_feeds_a_filter(node)
        _check_row_sink_arity(declared, described, rows, node, select)
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
        params = _wasm_params(
            self._eval_ctx,
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
        n_input = _n_input_call(self.registry, name)
        if n_input is not None:
            spec, options = n_input
            return self._lower_variadic_n_input_call(node, spec, options, call, env, select)
        concat = _concat_options(self.registry, name)
        if concat is not None:
            return self._lower_concat_call(node, concat, call, env, select)
        dynamic = self.registry.get(name) if self.registry is not None else None
        array_returning = call.namespaced and _array_options(self.registry, name) is not None
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
            hint=_namespaced_function_hint(self.registry, name)
            if call.namespaced
            else _unknown_function_hint(self.registry, name),
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
                hint=_macro_function_hint(name),
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
        _reject_null_stream(call.display, call.args[stream_pos], select)
        kind = self._classify(call.args[stream_pos], env, select)
        _reject_passthrough_args(call.display, [kind], call, call.args[stream_pos])
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
        options = _macro_options(macro, call, node)
        streams = {stream_pos: self._lower_expr(call.args[stream_pos], env, select)}

        def build(values: list[object], _element: int) -> FrameRef:
            return macro.expand(values, self.ctx.node, options)

        return _expand_call(
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
        """
        if call.namespaced or self.registry is None:
            return name, dynamic
        twin = _twin_pair(self.registry, name, dynamic)
        if twin is None:
            return name, dynamic
        kinds = self._stream_kinds(call, env, select, len(dynamic.inputs))
        if kinds[:1] == ["audio"]:
            return "a" + name, twin
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
            twin_stem = _twin_dispatch_stem(self.registry, name, call, kinds)
            raise _bad_streams(call, node, select, expected, kinds, twin_stem=twin_stem)
        args_at, per_row = self._option_binder(
            name,
            call,
            node,
            select,
            env,
            options=_options_for(self.registry, name, call, len(expected), node, select),
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

        return _expand_call(
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
            rows=_row_elements(per_row, env),
        )

    # -- N-input filters ----------------------------

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
        _reject_passthrough_args(call.display, kinds, call, node)
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
            declared = _n_input_count(spec, option_name, args, options)
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

        return _expand_call(
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

        Gathering nothing is the one shape that is not a refusal outright:
        with no positional stream ahead of it the call has nothing to run on
        and nothing to write, so it lowers to a NULL cell. The refusal it
        stands in for is recorded rather than dropped -- a branch that ends up
        writing its OTHER columns raises it after all (:meth:`_lower_branch`),
        since only a branch that is NULL throughout has nothing to write.
        Positional streams ahead of the array keep the empty refusal outright.

        Cached by the array expression's identity: the classifier
        (:meth:`_classify`) needs this same array's element type to answer a
        nested `concat`'s kind, ahead of the call's own lowering reaching
        here -- so the second call reuses what the first already lowered
        instead of building it, and any node under it, twice.
        """
        variadic = call.variadic
        assert variadic is not None  # callers only reach here when it is
        cached = self._variadic_array_cache.get(id(variadic))
        if cached is not None:
            return cached
        value = self._lower_expr(variadic, env, select)
        if _is_null(value) or (value.is_array and not value.streams):
            empty = _error(
                ErrorCode.UDF_ARG_TYPE,
                f"{call.display}() has no inputs: {_sql_text(variadic)} is empty",
                variadic,
                fallback=node,
                hint="VARIADIC spreads the array as the call's argument list; "
                "an empty array leaves the filter nothing to run on",
            )
            if call.args:
                raise empty
            self.empty_aggregates.append(empty)
            value = _scalar(_Stream(ref=_NULL_STREAM_REF, type=value.type))
        elif not value.is_array:
            raise _error(
                ErrorCode.UDF_ARG_TYPE,
                f"VARIADIC needs an array: {_sql_text(variadic)} is a single "
                f"{value.type} stream",
                variadic,
                fallback=node,
                hint="drop VARIADIC to pass it as one ordinary stream argument",
            )
        self._variadic_array_cache[id(variadic)] = value
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
        if _is_null(array_value):
            return array_value  # the aggregate gathered nothing: a NULL cell
        prefix = [self._lower_expr(arg, env, select).at(0) for arg in call.args]
        streams = prefix + list(array_value.streams)
        count = len(streams)
        args = self._bind_options(
            spec.name, call, node, select, env, options=options, extras=[], timeline=False,
        )
        _check_variadic_count(spec.option, count, args, call, node, select, _N_INPUT_HINT)
        if spec.option is not None and (
            spec.emit_default or spec.option in args or count != spec.fallback
        ):
            args[spec.option] = count
        node_id = self.ctx.node(
            spec.name, args, [stream.ref for stream in streams], [spec.output]
        )
        source = streams[0].source if count == 1 else _agreed_source(streams)
        return _scalar(_Stream(ref=node_id, type=spec.output, source=source))

    # -- VARIADIC concat: N segments of one stream type --------------------

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
        if _is_null(array_value):
            return array_value  # the aggregate gathered nothing: a NULL cell
        prefix = [self._lower_expr(arg, env, select).at(0) for arg in call.args]
        streams = prefix + list(array_value.streams)
        count = len(streams)
        bindable = {key: option for key, option in options.items() if key not in ("v", "a")}
        args = self._bind_options(
            "concat", call, node, select, env, options=bindable, extras=[], timeline=False,
        )
        _check_variadic_count("n", count, args, call, node, select, _CONCAT_VARIADIC_HINT)
        args["n"] = count
        args["v"] = 1 if stream_type == "video" else 0
        args["a"] = 1 if stream_type == "audio" else 0
        node_id = self.ctx.node("concat", args, [stream.ref for stream in streams], [stream_type])
        source = streams[0].source if count == 1 else _agreed_source(streams)
        return _scalar(_Stream(ref=node_id, type=stream_type, source=source))

    # -- array-returning filters -----------------------

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
            raise _bad_streams(call, node, select, expected, kinds)
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
            raise _bad_count(spec, count, call, node, select)

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

    # -- shared call machinery --------------------------------------------

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
            _reject_null_stream(call.display, arg, select)
        kinds = [self._classify(arg, env, select) for arg in call.args[:arity]]
        if kinds:
            _reject_passthrough_args(call.display, kinds, call, call.args[0])
        return kinds

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
        ``ffmpeg.trim(f, start => f.duration)``, ``scale(f.video[1], r.w,
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
                                _computed_arg(
                                    self._eval_ctx, arg.value, env, row, select,
                                    evaluate=eval_it,
                                ),
                            )
                            for arg, eval_it in zip(call.named, named_countable, strict=True)
                        ],
                    ),
                    node,
                    select,
                    env,
                    options=options,
                    extras=[
                        _computed_arg(self._eval_ctx, arg, env, row, select, evaluate=eval_it)
                        for arg, eval_it in zip(extras, extras_countable, strict=True)
                    ],
                    timeline=timeline,
                )
            return cache[element]

        return bound, per_row

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
        :func:`ffrwd.filter_options._option_value` a named argument goes through, which is what
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
            _check_named_args(
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
        _check_required_options(filter_name, bound, dropped, node, select)
        return bound

    # -- named argument validation --

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
            # A filled track column is a stream of its first argument's type,
            # which is knowable without lowering anything (no fill is minted).
            first = _coalesce_arguments(node, select)[0]
            binding = _coalesce_binding(first, env)
            return binding.type if binding is not None else self._classify(
                first, env, select
            )
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
                        hint=_macro_function_hint(name),
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
            if call.namespaced and _array_options(self.registry, name) is not None:
                return ARRAY_RETURNING[name].element
            n_input = _n_input_call(self.registry, name)
            if n_input is not None:
                return n_input[0].output
            # `concat` is excluded from the registry on BOTH sides (`N->N`),
            # so it is unreachable here unless VARIADIC gave its pad count a
            # source -- exactly the condition :meth:`_lower_call` dispatches
            # on before ever reaching this classifier. Mirrored here so a
            # nested VARIADIC call (`arealtime(concat(VARIADIC array_agg(t)))`)
            # is classified the same way a top-level one lowers, instead of
            # falling through to the ordinary registry lookup, where a call
            # this shape can never be found.
            if call.variadic is not None:
                concat_options = _concat_options(self.registry, name)
                if concat_options is not None:
                    return self._variadic_array(call, node, env, select).type
            dynamic = self.registry.get(name) if self.registry is not None else None
            if dynamic is None:
                raise _error(
                    ErrorCode.UNKNOWN_FUNCTION,
                    f"unknown function {call.display}()",
                    node,
                    fallback=select,
                    hint=_namespaced_function_hint(self.registry, name)
                    if call.namespaced
                    else _unknown_function_hint(self.registry, name),
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
        self._lower_ctes()
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

        A sink call has no table to print: the module is the destination and
        writes nothing back, so it is refused here rather than lowered as a
        csv COPY whose cells the sink already consumed.
        """
        if raw.module_sink:
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"{raw.module_sink}() is a sink and prints no table",
                raw.path_node,
                fallback=raw.query,
                hint="drop the TO to print the relation, or run it as a media "
                "COPY with `ffrwd compile`",
            )
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
        _check_grouped_cte_columns(select, env)
        time_conjuncts, row_conjuncts, assertion_conjuncts = self._split_where(select, env)
        per_row = any(_is_row_window(conjunct, env) for conjunct in time_conjuncts)
        if not per_row:
            self._collect_trims(select, env, time_conjuncts)
        _filter_rows(self._eval_ctx, row_conjuncts, env, select)
        self._check_assertions(assertion_conjuncts, select)
        _order_rows(self._eval_ctx, select, env)
        _limit_rows(select, env)
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
        if relation is None:
            raise self._grouped_no_relation_error(env, select)
        groups = _grouped_partitions(self._eval_ctx, env, select)
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

    def _grouped_no_relation_error(self, env: _Env, select: exp.Select) -> FfrwdError:
        """The refusal for a grouped branch with no rows to gather: an input
        whose probe failed never got its rendition table, and a probed one
        without renditions is the same shape error the media path gives."""
        for binding in env.bindings.values():
            if isinstance(binding, _InputBinding) and self.probes.get(binding.alias) is None:
                return self._unreadable_error(
                    ErrorCode.INPUT_NOT_FOUND,
                    binding.alias,
                    f"cannot gather rows for '{binding.alias}' from "
                    f"'{self._path_of(binding.alias)}'",
                    select,
                    select,
                    hint=f"array_agg(...) over '{binding.alias}' reads its rendition "
                    "rows, and only a readable input has any",
                )
        return _error(
            ErrorCode.UNSUPPORTED_SQL,
            "array_agg() aggregates track rows, and this query has none",
            select,
            fallback=select,
            hint=_ARRAY_AGG_HINT,
        )

    def _is_printable_value(self, node: exp.Expr) -> bool:
        """True for a shape a table column prints as a computed value cell,
        the same grammar :meth:`_eval_value` evaluates: the parser's own
        value shapes (``is_value_expr``), a vector builtin call
        (``cos_similarity``/``vector_length``), a declared wasm value
        function call, or a built-in text/number function (``round``,
        ``upper``, ...) wrapping one of those -- what lets
        ``round(cos_similarity(...), 4)`` print a rounded score the same
        way ``round(a + b, 2)`` prints a rounded sum.
        """
        if is_value_expr(node):
            return True
        call = _call_parts(node)
        if call is not None and not call.namespaced and not call.is_macro:
            name = call.name.lower()
            if name in _VECTOR_BUILTIN_ARITY:
                return True
            declared = self.res.wasm.get(name)
            if declared is not None and declared.is_value:
                return True
        if isinstance(node, _BUILTIN_VALUE_FUNCS):
            return self._is_printable_value(node.this)
        return False

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
                        return _row_metadata_cells(binding, name, expr, select)
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
                        return _value_cells(self._eval_ctx, expr, env, select, cardinality)
                    column = _cte_column(binding, _fold(expr.this))
                    # A splat column falls through to `_value_to_cells` below,
                    # which is where its per-row cardinality is already
                    # honored; a non-splat one (array_agg / a bare input
                    # array, re-exposed through the CTE) stays ONE cell.
                    if column is not None and column.value.is_array and not column.splat:
                        return self._array_cell_broadcast(expr, env, select, cardinality)
        if self._is_printable_value(expr) or _is_input_value_column(expr, env):
            return _value_cells(self._eval_ctx, expr, env, select, cardinality)
        shape = subscript_metadata_shape(expr)
        if shape is not None:
            metadata_value = self._accessor_value(expr, select)
            return [_row_value_as_cell(metadata_value)] * cardinality
        stream_value = self._lower_expr(projection, env, select)
        if not stream_value.streams:
            # A column that consumed its streams rather than producing one --
            # a sink call, a module's rows at a rows destination -- has no
            # cell to print.
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"'{expr.sql(dialect='postgres')}' carries no streams, so there "
                "is nothing to print",
                projection,
                fallback=select,
                hint="a table query prints streams and metadata; a module that "
                "consumes them writes to a destination instead",
            )
        splat = _is_splat_projection(projection, env)
        return _value_to_cells(stream_value, cardinality, splat=splat)

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
        rows = (
            self._track_record_columns(alias, column, None, result, anchor, select)
            if column in TRACK_RECORD_COLUMNS
            else _record_columns(result, column)
        )
        cell = ArrayCell(
            elements=tuple(
                RecordCell(
                    fields=tuple(_row_value_as_cell(row[name]) for name in names)
                )
                for row in rows
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
            elements=tuple(_stream_to_cell(stream) for stream in value.streams)
        )
        return [cell] * cardinality

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


def _outputs(
    columns: list[_Column],
    tags: _TagOverrides,
    dispositions: _DispositionOverrides,
    minted: Mapping[FrameRef, dict[str, str]] | None = None,
) -> list[Output]:
    """One :class:`~ffrwd.ir.Output` per stream a SELECT list carries.

    The SELECT list IS the output stream list, and an array column is several
    streams, so it splats into consecutive Outputs. Every element of an
    aliased array column keeps that alias VERBATIM (no ordinal suffix): the
    alias names the column, not the stream.

    `minted` is what a compiler-minted metadata track says about itself --
    its title, a vector track's dimensions -- which no probe reported and
    which therefore rides here rather than through provenance.
    """
    carried = minted or {}
    return [
        Output(
            ref=stream.ref,
            type=stream.type,
            name=column.name,
            metadata={**_metadata(stream, tags), **carried.get(stream.ref, {})},
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


def _value_column_name(projection: exp.Expr, env: _Env, *, natural: bool) -> str | None:
    """The name a VALUE column takes, or None when the projection is not one.

    A value column's expression is a compile-time one over the row: a
    literal, NULL, a row's metadata column, an input's ``duration`` or
    container tag, a CTE's own value column, CASE, ``||``, arithmetic or
    ``::text``. Everything else is a stream expression and lowers as one.

    The name is the ``AS`` alias. In a CTE BODY (`natural`) a bare column
    reference names itself instead, as Postgres names any unaliased one, so
    ``SELECT w.start_t`` gives the CTE's rows a ``start_t`` the outer query
    reads back. Under a sink an unaliased value stays unnamed and keeps the
    rejection it already had — there is no output column for it to be.
    """
    value = _unwrap(projection)
    if not (
        isinstance(value, exp.Null | exp.Literal | exp.Neg)
        or is_value_expr(value)
        or _is_input_value_column(value, env)
        or _is_cte_value_column(value, env)
        or _row_metadata_column(value, env) is not None
    ):
        return None
    name = _projection_name(projection)
    if name is not None:
        return name
    return _table_column_name(value) if natural and isinstance(value, exp.Column) else None


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
    ``:'widths'[i.i]``, a CTE's own value column, anything arithmetic over
    one. An input alias's probed column (``f.duration``) is the same for
    every row and is not one.
    """
    for sub in _unwrap(node).walk():
        if not isinstance(sub, exp.Expr):
            continue
        if _row_metadata_column(sub, env) is not None or _is_cte_value_column(sub, env):
            return True
    return False


def _is_row_scalar(node: exp.Expr, env: _Env) -> bool:
    """True for a bare column that is a compile-time VALUE, never a stream --
    an input's probed duration/tag, a row table's metadata field, or a CTE's
    value column.

    Lets a ``duration``-typed filter option accept ``start => f.duration``
    the way it already accepts arithmetic over one (`_option_binder`), and
    ``start => b.start_t`` off the CTE that carried the bound out.
    """
    inner = _unwrap(node)
    return (
        _is_input_value_column(inner, env)
        or _is_cte_value_column(inner, env)
        or _row_metadata_column(inner, env) is not None
    )


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
    probe_path: ProbePath = probe_one_path,
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
        probe_path=probe_path,
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
    probe_path: ProbePath = probe_one_path,
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
            probe_source=probe_source, probe_path=probe_path,
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
                probe_source=probe_source, probe_path=probe_path,
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
    probe_path: ProbePath = probe_one_path,
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
        probe_failures=probe_failures, probe_source=probe_source, probe_path=probe_path,
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
