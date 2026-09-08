"""The checks a lowered construct has to pass, decided from its arguments.

Each takes what it judges and either returns or raises; none reads the graph
being built, which is what will let them run as one admission pass over it.
Together in one module so a rule written for one construct is visible to the
next, instead of a check in one method being invisible to a check in another.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from sqlglot import exp

from ffrwd.bindings import _CteBinding, _Env, _has_track_rows, _RowBinding
from ffrwd.calls import _Call, _call_parts
from ffrwd.ctes import _cte_column
from ffrwd.destinations import _PER_TRACK_OPTION_HINT, _join_codecs
from ffrwd.errors import ErrorCode, FfrwdError
from ffrwd.expressions import _coalesce_label, _error, _unwrap
from ffrwd.fills import _fill_hint
from ffrwd.filter_options import (
    _ENABLE,
    REQUIRED_OPTIONS,
    _enable_value,
    _NamedArg,
    _option_hint,
    _option_value,
)
from ffrwd.functions import WASM_STREAM_NAMES, Annotation, WasmFunction
from ffrwd.ir import Output, StreamType
from ffrwd.modules import (
    _annotation_fields,
    _annotation_matches,
    _vector_field,
    _written_json_fields,
)
from ffrwd.parser import (
    RawInputOption,
    RawSink,
    Resolved,
    _pos,
    _projection_expr,
    annotation_projection,
    group_keys,
    null_variable,
    star_node,
    star_replace_entries,
)
from ffrwd.parser import _ident_name as _fold
from ffrwd.probe import is_url
from ffrwd.registry import FilterOption
from ffrwd.sink import SINK_OPTIONS
from ffrwd.types import (
    CONTAINER_READONLY_FIELDS,
    DISPOSITION_COLUMN,
    DISPOSITION_KEYS,
    is_array,
)
from ffrwd.values import _PASSTHROUGH_ONLY, _Column, _signature, _stream_count, _Value
from ffrwd.vars import unset_error
from ffrwd.wasm import (
    ANNOTATION_TYPES,
    WIRE_AUDIO_CODECS,
    WORLDS,
    Described,
    DescribedFunction,
    hosts_packet_sink,
    hosts_packet_source,
    hosts_rows_module,
    input_rows_arms,
    rows_arms,
    rows_vector_dims,
)

if TYPE_CHECKING:
    from ffrwd.destinations import _VariantRow


def _check_per_track_options(
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


def _check_concat_columns(
    branches: list[exp.Select], flattened: list[list[_Column]]
) -> None:
    """No UNION ALL branch may carry a subtitle/data column.

    ``concat`` is a filtergraph filter and takes ``v`` video plus ``a``
    audio pads — there is no ``s``/``d`` half — so a caption column in a
    concatenated branch has nowhere to go. Checked before
    :func:`_check_concat_signature` so the rejection names the real reason
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


def _check_tag_key(
    key: str, anchor: exp.Expr, env: _Env, select: exp.Select
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


def _check_star_table_mode(anchor: exp.Expr, select: exp.Select) -> None:
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


def _check_rows_schema(
    declared: WasmFunction,
    annotation: Annotation,
    arms: tuple[tuple[tuple[str, str], ...], ...] | None,
    *,
    reads: bool,
    node: exp.Expr,
    select: exp.Select,
) -> None:
    """One end of a rows declaration against the schema the module publishes.

    The same field-for-field match a per-frame annotation gets, run twice:
    once for the rows the module reads, once for the rows it writes.
    """
    side = "reads" if reads else "writes"
    if arms is None:
        raise _error(
            ErrorCode.UNSUPPORTED_SQL,
            f"the module '{declared.module}' names no shape for the rows "
            f"it {side}",
            node,
            fallback=select,
            hint="a rows function needs a module that publishes both row "
            "schemas; rebuild it declaring them",
        )
    fields = _annotation_fields(annotation)
    if any(_annotation_matches(fields, arm) for arm in arms):
        return
    raise _error(
        ErrorCode.UDF_ARG_TYPE,
        f"function '{declared.name}' declares the rows it {side} as "
        f"{annotation.written}, and the module '{declared.module}' {side} "
        f"{' or '.join(_written_json_fields(arm) for arm in arms)}",
        node,
        fallback=select,
        hint="a rows record names the module's own row columns, with a "
        "type each value fits",
    )


def _check_rows_argument(
    declared: WasmFunction,
    emitted: Annotation,
    written: exp.Expr,
    node: exp.Expr,
    select: exp.Select,
) -> None:
    """That the rows arriving are the rows this function declares reading."""
    if _annotation_fields(declared.rows_param) == _annotation_fields(emitted):
        return
    raise _error(
        ErrorCode.UDF_ARG_TYPE,
        f"{declared.name}() reads '{declared.rows_param.name}' as "
        f"{declared.rows_param.written}, and its argument carries "
        f"'{emitted.name}' as {emitted.written}",
        written,
        fallback=node,
        hint="the two annotation records have to name the same fields, "
        "with the same types",
    )


def _check_coalesce_width(
    value: _Value, cardinality: int, node: exp.Expr, select: exp.Select
) -> None:
    """A COALESCE argument is one cell per row, or one stream every row
    repeats -- nothing in between."""
    if len(value.streams) in (1, cardinality):
        return
    raise _error(
        ErrorCode.ROW_COUNT_MISMATCH,
        f"'{_coalesce_label(node)}' carries {len(value.streams)} streams, "
        f"and COALESCE is over {cardinality} rows",
        node,
        fallback=select,
        hint="every argument is one cell per row, or one stream the rows "
        "repeat",
    )


def _check_sink_shape(
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


def _check_vector_dims(
    declared: WasmFunction,
    annotation: Annotation,
    described: Described,
    node: exp.Expr,
    select: exp.Select,
) -> None:
    """A vector-typed annotation field's length, fixed by the module's schema.

    The declaration says a field is a vector; only the module's own
    schema says how many numbers one carries, and that is what tags the
    track the field becomes (:meth:`_rows_output`). A field the schema
    does not fix a length for is refused here, rather than minting a
    track later with nothing to tag it.
    """
    field = _vector_field(annotation)
    if field is None:
        return
    if rows_vector_dims(described, field) is None:
        raise _error(
            ErrorCode.UNSUPPORTED_SQL,
            f"function '{declared.name}' declares '{field}' as vector, and "
            f"the module '{declared.module}' does not fix its length",
            node,
            fallback=select,
            hint="a vector track needs its dimension: declare minItems and "
            "maxItems on the field",
        )


def _check_wasm_result_type(
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


def _check_row_sink_arity(
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


def _check_variadic_count(
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


def _check_named_args(
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


def _described_source(
    describes: Mapping[str, Described], declared: WasmFunction, node: exp.Expr, select: exp.Select
) -> Described:
    """What a ``RETURNS source`` call's module declares, checked.

    The source mirror of :meth:`_described`, checked against its OWN
    rules rather than reused whole: a source reads no streams and emits
    no per-frame annotations, so the filter-shaped checks
    :meth:`_described` runs after the world/export match --
    :func:`ffrwd.admit._check_stream_arity` chief among them, which would read
    ``described.inputs`` as if it were a filter's pad count -- have
    nothing to check here and would misjudge a module that correctly
    reads none at all.

    Two module shapes answer a ``RETURNS source`` call. A PACKET source
    names the export as its own single export and reports ``source``; a
    URL source is a values module that offers the export in its
    ``functions`` list and names files instead of producing packets.
    Anything else is refused.
    """
    described = describes.get(declared.module)
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
    # A values module names no single export, so there is nothing to
    # match; the export it has to offer is checked against `functions`
    # below instead.
    if described.name and described.name != declared.export:
        raise _error(
            ErrorCode.UNSUPPORTED_SQL,
            f"function '{declared.name}' names the export '{declared.export}', "
            f"and '{declared.module}' exports '{described.name}'",
            node,
            fallback=select,
            hint=f"a module carries one filter; write '{described.name}' as "
            "the export",
        )
    if described.source:
        if not hosts_packet_source(described.world):
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"the module '{declared.module}' produces packets, and the "
                f"sidecar's {described.world} cannot host one",
                node,
                fallback=select,
                hint=f"a packet source is told which tracks to pull from "
                f"{WORLDS[-1]} on; upgrade ffrwd, or point at a newer "
                "ffrwd-wasm",
            )
        return described
    if any(fn.name == declared.export for fn in described.functions):
        return described
    raise _error(
        ErrorCode.UNSUPPORTED_SQL,
        f"function '{declared.name}' declares RETURNS source, and the "
        f"module '{declared.module}' is not a packet source",
        node,
        fallback=select,
        hint=f"'{declared.module}' has to export a packet source built "
        f"RETURNS source, or offer '{declared.export}' among its own "
        "functions; check the module and the export named",
    )


def _described_rows(
    describes: Mapping[str, Described], declared: WasmFunction, node: exp.Expr, select: exp.Select
) -> Described:
    """What a ROWS function's module turned out to declare, checked.

    The rows mirror of :meth:`_described`: the world has to host a rows
    module, the module has to BE one, and both ends of the declaration --
    the column it reads and the record it returns -- are matched against
    the two schemas the module publishes.
    """
    described = describes.get(declared.module)
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
    if not described.rows_module:
        raise _error(
            ErrorCode.UNSUPPORTED_SQL,
            f"function '{declared.name}' returns {declared.returns}, and the "
            f"module '{declared.module}' reads no rows",
            node,
            fallback=select,
            hint="a rows function needs a module that reads rows and writes "
            "rows; declare a stream and a return to filter a stream instead",
        )
    if not hosts_rows_module(described.world):
        raise _error(
            ErrorCode.UNSUPPORTED_SQL,
            f"the module '{declared.module}' is a rows module, and the "
            f"sidecar's {described.world} cannot host one",
            node,
            fallback=select,
            hint="rebuild the module against a world whose sidecar runs "
            "rows modules, or upgrade ffrwd",
        )
    if described.name != declared.export:
        raise _error(
            ErrorCode.UNSUPPORTED_SQL,
            f"function '{declared.name}' names the export '{declared.export}', "
            f"and '{declared.module}' exports '{described.name}'",
            node,
            fallback=select,
            hint=f"a module carries one rows export; write '{described.name}' "
            "as the export",
        )
    _check_rows_schema(
        declared,
        declared.rows_param,
        input_rows_arms(described),
        reads=True,
        node=node,
        select=select,
    )
    assert declared.returns_rows is not None  # what is_rows selected on
    _check_rows_schema(
        declared,
        declared.returns_rows,
        rows_arms(described),
        reads=False,
        node=node,
        select=select,
    )
    return described


def _described(
    describes: Mapping[str, Described], declared: WasmFunction, node: exp.Expr, select: exp.Select
) -> Described:
    """What the module a declaration names turned out to declare, checked.

    The describe itself happened before lowering, once per module path
    (:mod:`ffrwd.wasm`); what happens here is comparing it against the
    declaration that named it. Both rejections anchor on the CALL, since
    the declaration's own position is not in the query being lowered by
    the time a rejection is worth reporting.
    """
    described = describes.get(declared.module)
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
        _check_packet_sink(declared, described, node, select)
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
        _check_stream_arity(declared, described, node, select)
    if declared.emits is not None:
        _check_annotation_schema(declared, declared.emits, described, node, select)
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
    # once they are known (:func:`ffrwd.admit._check_row_sink_arity`), not here
    # against a signature that names none.
    if not declared.reads_rows_from_select:
        _check_sink_shape(declared, described, node, select)


def _varies_per_row(binding: _CteBinding, name: str) -> bool:
    """True when a CTE column carries a stream per body row, and there is
    more than one of them -- the shape that differs tuple by tuple."""
    if binding.rows <= 1:
        return False
    column = _cte_column(binding, name)
    return column is not None and column.splat and column.value.is_array


def _check_grouped_cte_expr(
    node: exp.Expr, env: _Env, select: exp.Select, key_texts: set[str]
) -> None:
    """One expression of a grouped branch, recursively."""
    if node.sql() in key_texts or isinstance(node, exp.ArrayAgg):
        return
    if isinstance(node, exp.Filter) and isinstance(node.this, exp.ArrayAgg):
        # A FILTER over array_agg only ever names the same column the
        # aggregate reads -- parser confirmed the predicate -- so its
        # WHERE clause raises nothing new here.
        return
    if isinstance(node, exp.Column) and not isinstance(node.this, exp.Star):
        table_node = node.args.get("table")
        binding = (
            env.bindings.get(_fold(table_node)) if table_node is not None else None
        )
        name = _fold(node.this)
        if isinstance(binding, _CteBinding) and _varies_per_row(binding, name):
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
                _check_grouped_cte_expr(item, env, select, key_texts)


def _check_grouped_cte_columns(select: exp.Select, env: _Env) -> None:
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
            _check_grouped_cte_expr(
                _projection_expr(projection), env, select, key_texts
            )
        else:
            for _, _, expr in star_replace_entries(star):
                _check_grouped_cte_expr(expr, env, select, key_texts)


def _check_fill_type(
    output: StreamType,
    display: str,
    kind: StreamType,
    label: str,
    node: exp.Expr,
    select: exp.Select,
) -> None:
    """A fill stands in for a track, so it has to BE one of the same type."""
    if output == kind:
        return
    raise _error(
        ErrorCode.UDF_ARG_TYPE,
        f"{display}() generates a {output} stream, but "
        f"'{label}' is {kind}",
        node,
        fallback=select,
        hint=_fill_hint(kind, label),
    )


def _check_coalesce_fill(
    kind: StreamType,
    other: _Value,
    label: str,
    node: exp.Expr,
    select: exp.Select,
) -> None:
    """What stands in for a gap is a stream of the SAME kind as the cell
    it fills: another column's, or a generated one."""
    if not other.streams:
        raise _error(
            ErrorCode.UNSUPPORTED_SQL,
            f"a COALESCE fill is a stream or a generated stand-in, and "
            f"'{_coalesce_label(node)}' produces neither",
            node,
            fallback=select,
            hint=_fill_hint(kind, label),
        )
    if other.type != kind:
        raise _error(
            ErrorCode.UDF_ARG_TYPE,
            f"COALESCE stands in for one track: '{label}' is {kind}, "
            f"and '{_coalesce_label(node)}' is {other.type}",
            node,
            fallback=select,
            hint=_fill_hint(kind, label),
        )


def _check_realtime_option(
    res: Resolved,
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
    `../docs/corpus.md`). Telling a capture device from a generator by
    its `format` value needs a name list this table does not carry, so
    that half stays unrefused -- only a URL (`is_url`: udp, srt, rtmp,
    rtsp, http(s), ...) is unambiguous enough to reject here.
    """
    if options.get("realtime") is not True:
        return
    index = res.sources.get(alias)
    path = res.input_paths[index] if index is not None else ""
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


def _annotating_call(res: Resolved, node: exp.Expr) -> WasmFunction | None:
    """The annotation-returning wasm function `node` calls, if it calls one."""
    call = _call_parts(_unwrap(node))
    if call is None or call.namespaced or call.is_macro:
        return None
    found = res.wasm.get(call.name.lower())
    return found if found is not None and found.emits is not None else None


def _reads_annotations(res: Resolved, node: exp.Expr) -> WasmFunction | None:
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
    found = res.wasm.get(call.name.lower())
    return found if found is not None and found.reads is not None else None


def _check_annotation_argument(
    res: Resolved, declared: WasmFunction, call: _Call, node: exp.Expr, select: exp.Select
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
            if (found := _annotating_call(res, argument)) is not None
        ),
        (call.args[0] if call.args else node, None),
    )
    at = declared.stream_arity
    gathered = (
        annotation_projection(_unwrap(call.args[at]), res.wasm)
        if declared.reads is not None and len(call.args) > at
        else None
    )
    if gathered is not None:
        anchor, producer = call.args[at], gathered[1]
    if (
        declared.emits is not None
        and _reads_annotations(res, node) is None
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


_GROUPED_CTE_HINT = (
    "a CTE with several rows varies inside the group: wrap the column in "
    "array_agg(...), or add it to the GROUP BY to make it the group's key"
)


def _projects_annotations(node: exp.Expr, column: str) -> bool:
    """True when `column` is read off `node`, through any wrapping parens."""
    inner, parent = node, node.parent
    while isinstance(parent, exp.Paren):
        inner, parent = parent, parent.parent
    if not isinstance(parent, exp.Dot) or parent.this is not inner:
        return False
    field = parent.args.get("expression")
    return isinstance(field, exp.Identifier) and _fold(field) == column
