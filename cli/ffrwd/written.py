"""Values a query writes, checked as they are read.

A ``STRUCT`` matched to the fields a record type declares, the width every
row of a vector track must agree on, and the closed flag set a disposition
may name.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlglot import exp

from ffrwd.bindings import _CteBinding, _Env, _InputBinding, _RowTuple
from ffrwd.ctes import _cte_columns_hint
from ffrwd.errors import ErrorCode
from ffrwd.evaluate import _eval_value, _EvalContext
from ffrwd.expressions import _describe, _error, _struct_fields, _unwrap
from ffrwd.ir import Attachment
from ffrwd.merge import RowValue
from ffrwd.parser import _ident_name as _fold
from ffrwd.parser import article, flag_error, record_cast_type
from ffrwd.rows import _group_row
from ffrwd.types import (
    ATTACHMENT_TYPE,
    ATTACHMENTS_COLUMN,
    CHAPTER_TYPE,
    CHAPTERS_COLUMN,
    CUE_TYPE,
    DISPOSITION_COLUMN,
    DISPOSITION_KEYS,
    EMBEDDING_TYPE,
    EMBEDDINGS_COLUMN,
    RECORD_FIELDS,
    TAGS_COLUMN,
    Field,
)

_ATTACHMENT_EXAMPLE = (
    f"STRUCT('font.ttf' AS filename, 'application/x-truetype-font' AS mimetype, "
    f"'fonts/font.ttf' AS path)::{ATTACHMENT_TYPE}"
)

_CUE_ARROW = "-->"

# Characters ffmetadata's own escaping would need (`\`, `=`, `;`, `#`, a
# newline) -- rejected outright rather than silently writing a file ffmpeg
# cannot parse back.
_UNSAFE_CHAPTER_TITLE = frozenset("\\=;#\n\r")

_WEBVTT_ESCAPES = (("&", "&amp;"), ("<", "&lt;"))


@dataclass(frozen=True)
class _Embedding:
    """One written embedding: its span, its vector, and where both were written."""

    start: int | float
    end: int | float
    vector: tuple[float, ...]
    start_node: exp.Expr
    end_node: exp.Expr
    vector_node: exp.Expr


def _named_record_cells(
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


def _flag_spec(
    value: RowValue, anchor: exp.Expr, select: exp.Select
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


def _embedding_dims(
    rows: list[_Embedding], node: exp.Expr, select: exp.Select
) -> int:
    """How many numbers every row of one vector track carries.

    A track has ONE width -- it is a stream tag, not a per-row field --
    so two rows of different lengths are a rejection naming both.
    """
    dims = len(rows[0].vector)
    for position, row in enumerate(rows[1:], start=2):
        if len(row.vector) != dims:
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"{EMBEDDING_TYPE} {position} carries {len(row.vector)} "
                f"numbers, and {EMBEDDING_TYPE} 1 carries {dims}",
                row.vector_node,
                fallback=select,
                hint="one track holds vectors of one length; write the rows "
                "of one embedder per track",
            )
    if dims == 0:
        raise _error(
            ErrorCode.UNSUPPORTED_SQL,
            f"{EMBEDDING_TYPE} 1 carries an empty vector",
            rows[0].vector_node,
            fallback=select,
            hint="a vector track holds the numbers an embedder wrote; an "
            "empty one says nothing",
        )
    return dims


def _written_record(
    ctx: _EvalContext,
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
    cells = _named_record_cells(struct, record, fields, select)
    return {
        name: (cell, _eval_value(ctx, cell, env, row, select))
        for name, cell in cells.items()
    }


def _chapter_record(
    ctx: _EvalContext, node: exp.Expr, env: _Env, row: _RowTuple, select: exp.Select
) -> _Chapter:
    """One ``STRUCT(title, start_t, end_t)::chapter``, evaluated and checked."""
    cells = _written_record(
        ctx, node, CHAPTER_TYPE, _CHAPTER_LITERAL, _CHAPTERS_COLUMN_HINT, env, row, select
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


def _chapter_records(
    ctx: _EvalContext, value: exp.Expr, env: _Env, select: exp.Select
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
            _chapter_record(ctx, inner, env, row, select) for row in relation.tuples
        ]
    if isinstance(value, exp.Array):
        row = _group_row(env)
        return [
            _chapter_record(ctx, element, env, row, select)
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


def _attachment_record(
    ctx: _EvalContext, node: exp.Expr, env: _Env, row: _RowTuple, select: exp.Select
) -> Attachment:
    """One ``STRUCT(filename, mimetype, path)::attachment``, evaluated."""
    cells = _written_record(
        ctx,
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


def _attachment_records(
    ctx: _EvalContext, value: exp.Expr, env: _Env, select: exp.Select
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
            _attachment_record(ctx, inner, env, row, select)
            for row in relation.tuples
        ]
    if isinstance(value, exp.Array):
        row = _group_row(env)
        written = [
            _attachment_record(ctx, element, env, row, select)
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


def _cue_record(
    ctx: _EvalContext, node: exp.Expr, env: _Env, row: _RowTuple, select: exp.Select
) -> _Cue:
    """One ``STRUCT(text, start_t, end_t)::cue``, evaluated and checked."""
    cells = _written_record(
        ctx, node, CUE_TYPE, _CUE_LITERAL, _CUE_ARRAY_HINT, env, row, select
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


def _cue_records(
    ctx: _EvalContext, node: exp.Expr, env: _Env, select: exp.Select
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
            _cue_record(ctx, inner, env, row, select) for row in relation.tuples
        ]
    if isinstance(node, exp.Array):
        elements = [item for item in node.expressions if isinstance(item, exp.Expr)]
        if not elements or record_cast_type(_unwrap(elements[0])) != CUE_TYPE:
            return None
        row = _group_row(env)
        return [_cue_record(ctx, element, env, row, select) for element in elements]
    return None


def _embedding_record(
    ctx: _EvalContext, node: exp.Expr, env: _Env, row: _RowTuple, select: exp.Select
) -> _Embedding:
    """One ``STRUCT(start_t, end_t, vector)::embedding``, evaluated."""
    cells = _written_record(
        ctx,
        node,
        EMBEDDING_TYPE,
        _EMBEDDING_LITERAL,
        _EMBEDDING_ARRAY_HINT,
        env,
        row,
        select,
    )
    start_cell, start = cells["start_t"]
    end_cell, end = cells["end_t"]
    vector_cell, vector = cells["vector"]
    return _Embedding(
        start=_span_number(
            start,
            f"{EMBEDDING_TYPE}.start_t",
            "start_t",
            start_cell,
            _EMBEDDING_EXAMPLE,
        ),
        end=_span_number(
            end, f"{EMBEDDING_TYPE}.end_t", "end_t", end_cell, _EMBEDDING_EXAMPLE
        ),
        vector=_written_vector(vector, vector_cell),
        start_node=start_cell,
        end_node=end_cell,
        vector_node=vector_cell,
    )


def _embedding_records(
    ctx: _EvalContext, node: exp.Expr, env: _Env, select: exp.Select
) -> list[_Embedding] | None:
    """The rows an embedding array lists, in written order; None if it is
    not one. The two spellings a record list takes, exactly as a cue
    array's."""
    if isinstance(node, exp.ArrayAgg):
        inner = node.this
        relation = env.relation
        if not isinstance(inner, exp.Expr) or record_cast_type(
            _unwrap(inner)
        ) != EMBEDDING_TYPE:
            return None
        if relation is None:
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                "array_agg() aggregates rows, and this query has none",
                node,
                fallback=select,
                hint=_EMBEDDING_ARRAY_HINT,
            )
        return [
            _embedding_record(ctx, inner, env, row, select)
            for row in relation.tuples
        ]
    if isinstance(node, exp.Array):
        elements = [item for item in node.expressions if isinstance(item, exp.Expr)]
        if not elements or record_cast_type(_unwrap(elements[0])) != EMBEDDING_TYPE:
            return None
        row = _group_row(env)
        return [
            _embedding_record(ctx, element, env, row, select)
            for element in elements
        ]
    return None


def _read_tags(node: exp.Expr, env: _Env, select: exp.Select) -> _Tags:
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
                hint=_cte_columns_hint(binding),
            )
        # A row alias's map is what already rides through to the output,
        # so copying it names nothing new; only an input's globals do.
        if isinstance(binding, _InputBinding):
            copy_alias = alias
    return _Tags(entries=entries, copy_alias=copy_alias, stripped=stripped)


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


_CHAPTER_LITERAL = f"STRUCT(... AS title, ... AS start_t, ... AS end_t)::{CHAPTER_TYPE}"


_CHAPTER_EXAMPLE = f"STRUCT('Intro' AS title, 0 AS start_t, 60 AS end_t)::{CHAPTER_TYPE}"


_CHAPTERS_COLUMN_HINT = (
    f"a {CHAPTERS_COLUMN} column is an array of chapter records, e.g. "
    f"ARRAY[{_CHAPTER_EXAMPLE}] AS {CHAPTERS_COLUMN}, or "
    f"array_agg(STRUCT(c.title AS title, c.start_t AS start_t, c.end_t AS "
    f"end_t)::{CHAPTER_TYPE}) AS {CHAPTERS_COLUMN} over rows"
)


_ATTACHMENT_LITERAL = (
    f"STRUCT(... AS filename, ... AS mimetype, ... AS path)::{ATTACHMENT_TYPE}"
)


_ATTACHMENTS_COLUMN_HINT = (
    f"an {ATTACHMENTS_COLUMN} column is an array of attachment records, e.g. "
    f"ARRAY[{_ATTACHMENT_EXAMPLE}] AS {ATTACHMENTS_COLUMN}"
)


_CUE_LITERAL = f"STRUCT(... AS text, ... AS start_t, ... AS end_t)::{CUE_TYPE}"


_CUE_EXAMPLE = f"STRUCT('Hello' AS text, 0 AS start_t, 2.5 AS end_t)::{CUE_TYPE}"


_CUE_ARRAY_HINT = (
    f"an array of cue records IS a WebVTT subtitle track, e.g. "
    f"ARRAY[{_CUE_EXAMPLE}], or "
    f"array_agg(STRUCT(c.title AS text, c.start_t AS start_t, c.end_t AS "
    f"end_t)::{CUE_TYPE}) over chapter rows"
)


_EMBEDDING_LITERAL = (
    f"STRUCT(... AS start_t, ... AS end_t, ... AS vector)::{EMBEDDING_TYPE}"
)


_EMBEDDING_EXAMPLE = (
    f"STRUCT(v.start_t AS start_t, v.end_t AS end_t, v.vector AS vector)"
    f"::{EMBEDDING_TYPE}"
)


_EMBEDDING_ARRAY_HINT = (
    f"an array of {EMBEDDING_TYPE} records IS a vector track, e.g. "
    f"array_agg({_EMBEDDING_EXAMPLE}) over the rows of "
    f"unnest(<input>.{EMBEDDINGS_COLUMN})"
)


def _struct_node(node: exp.Expr) -> exp.Struct | None:
    """The ``STRUCT(...)`` a cast wraps, else None."""
    inner = _unwrap(node.this) if isinstance(node.this, exp.Expr) else None
    return inner if isinstance(inner, exp.Struct) else None


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


def _written_vector(value: RowValue, node: exp.Expr) -> tuple[float, ...]:
    """One evaluated ``vector`` as the numbers its block carries.

    A vector has no literal, so the value here came from a row column or a
    value function's own RETURNS; anything else -- a number, text, NULL --
    is not one.
    """
    if not isinstance(value, tuple):
        got = "NULL" if value is None else repr(value)
        raise _error(
            ErrorCode.UNSUPPORTED_SQL,
            f"'{EMBEDDING_TYPE}.vector' must be a vector, got {got}",
            node,
            hint="a vector comes from a vector row column or a RETURNS vector "
            f"function, e.g. {_EMBEDDING_EXAMPLE}",
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
