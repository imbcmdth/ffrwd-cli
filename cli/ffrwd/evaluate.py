"""The compile-time value evaluator.

Every column of a track row is PROBED metadata, so a predicate over rows is
decidable here, at compile time, and never reaches ffmpeg -- the way a
``WHERE t BETWEEN`` vanishes into ``-ss``/``-to``. Standard SQL three-valued
logic throughout: a comparison against NULL is UNKNOWN (python ``None``),
AND/OR/NOT are Kleene, and WHERE keeps a row only when its predicate came
back TRUE, so "NULL matches nothing" falls out rather than being a rule of
ours.

`resolve` already shape- and type-checked everything here; the rejections are
defensive re-checks raising the same FfrwdError resolve would.

Nothing below names the graph. :class:`_EvalContext` is the whole of what a
value may read -- what resolve settled, what the probes found, what the
modules describe -- so a value is a function of what the compile already
knows, and evaluating one mints no node. Two of its fields are bound methods
that consult the graph to answer, and each answers with a string; the graph
itself never crosses. That is a discipline the type does not enforce, and
widening either callback's return would end it.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from sqlglot import exp

from ffrwd.admit import _check_wasm_result_type
from ffrwd.bindings import (
    _RENDITION_SCHEMA,
    _CteBinding,
    _CteRow,
    _Env,
    _InputBinding,
    _RowBinding,
    _RowTuple,
    _track_of,
)
from ffrwd.calls import _Call, _call_parts
from ffrwd.ctes import _cte_columns_hint
from ffrwd.errors import ErrorCode, FfrwdError
from ffrwd.expressions import _error, _number, _unwrap
from ffrwd.functions import WasmFunction
from ffrwd.merge import RowValue
from ffrwd.modules import (
    _UNCACHED,
    _check_wasm_param,
    _declares_params,
    _folded_result,
)
from ffrwd.parser import (
    _ARITHMETIC,
    _ARITHMETIC_NAMES,
    _BUILTIN_VALUE_FUNCS,
    _VECTOR_BUILTIN_ARITY,
    MAP_COLUMNS,
    Resolved,
    _pos,
    is_value_expr,
    map_example,
    map_noun,
    map_ref,
    subscript_index,
    tag_key,
)
from ffrwd.parser import _ident_name as _fold
from ffrwd.probe import ProbeResult
from ffrwd.types import INPUT_DURATION_COLUMN, TAGS_COLUMN
from ffrwd.wasm import WORLDS, Described, DescribedFunction, Invoke


@dataclass(frozen=True)
class _EvalContext:
    """Everything a compile-time value is allowed to read.

    What resolve settled, what the probes found and what the modules
    describe; the module invoker and the memo it shares with the rest of
    lowering; and the two answers that need the graph as it stands, which is
    why they arrive as callbacks rather than as the graph itself.

    `probes` is the lowering's own dict and grows as inputs are probed, so a
    value read early and one read late may see different entries. It is an
    input, not a snapshot.
    """

    res: Resolved
    probes: Mapping[str, ProbeResult | None]
    describes: Mapping[str, Described]
    invoke: Invoke
    # (module, function, sorted args) -> result, so two calls with the same
    # arguments run the module once per compile.
    invoke_cache: dict[tuple[str, str, tuple[tuple[str, object], ...]], object]
    # The path behind an input alias, for a message about its file.
    path_of: Callable[[str], str]
    # The alias names in scope, for a message about one that is not.
    known_hint: Callable[[], str]


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


def _tag_text(value: str | int | float | bool | tuple[float, ...]) -> str:
    """A tag value as the text ffmpeg receives; a boolean spells itself out.

    A vector never reaches here in practice -- resolve refuses one at every
    call site (a tag, a ``::text`` cast, a fan-out path) -- but the
    signature is total, not partial, so a defensive caller gets a message
    instead of a crash.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, tuple):
        return f"a vector of {len(value)} values"
    return value if isinstance(value, str) else str(value)


def _literal_of(node: exp.Expr | None, select: exp.Select) -> RowValue:
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


def _eval_row(
    ctx: _EvalContext,
    node: exp.Expr,
    env: _Env,
    rows: _RowTuple,
    select: exp.Select,
) -> bool | None:
    """One predicate against one result row: TRUE, FALSE, UNKNOWN (``None``).

    `rows` maps every row alias in scope to that result row's track, or to
    None where an outer join left a gap — one evaluator for WHERE (which
    sees a single alias) and for a JOIN's ON (which sees both sides).

    Kleene three-valued logic, which is what makes the NULL story a
    non-story: a comparison with a NULL operand is UNKNOWN, UNKNOWN
    propagates through AND/OR/NOT the SQL way, and both callers keep TRUE
    only. A gap row reads NULL in every column, so "NULL matches nothing"
    covers the gaps too, for free.
    """
    node = _unwrap(node)
    if isinstance(node, exp.And | exp.Or):
        left = _eval_row(ctx, node.this, env, rows, select)
        expression = node.args.get("expression")
        if not isinstance(expression, exp.Expr):
            raise _error(
                ErrorCode.UNSUPPORTED_SQL, "malformed row predicate", node,
                fallback=select,
            )
        right = _eval_row(ctx, expression, env, rows, select)
        return (
            _kleene_and(left, right)
            if isinstance(node, exp.And)
            else _kleene_or(left, right)
        )
    if isinstance(node, exp.Not) and isinstance(node.this, exp.Expr):
        inner = _eval_row(ctx, node.this, env, rows, select)
        return None if inner is None else not inner
    if isinstance(node, exp.Is):
        value = _row_value_of(ctx, node.this, env, rows, select)
        is_null = value is None
        return not is_null if node.args.get("negate") else is_null
    if isinstance(node, exp.Between):
        value = _eval_value(ctx, node.this, env, rows, select)
        low = _eval_value(ctx, node.args.get("low"), env, rows, select)
        high = _eval_value(ctx, node.args.get("high"), env, rows, select)
        return _kleene_and(
            _compare(exp.GTE(), value, low), _compare(exp.LTE(), value, high)
        )
    if isinstance(node, exp.EQ | exp.NEQ | exp.GT | exp.GTE | exp.LT | exp.LTE):
        # Both sides go through one value evaluator, so the operands stay in
        # written order and `'eng' = t.tags.language` needs no mirroring.
        return _compare(
            node,
            _eval_value(ctx, node.this, env, rows, select),
            _eval_value(ctx, node.args.get("expression"), env, rows, select),
        )
    if isinstance(node, exp.Boolean | exp.Column):
        # A boolean value IS the condition, as it is in Postgres; resolve
        # already turned away a column of any other type.
        value = _eval_value(ctx, node, env, rows, select)
        return None if value is None else bool(value)
    raise _error(  # defensive: resolve accepted only the shapes above
        ErrorCode.UNSUPPORTED_SQL,
        "unsupported row predicate",
        node,
        fallback=select,
    )


def _cte_value_of(
    ctx: _EvalContext,
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
            hint=_cte_columns_hint(binding),
        )
    if binding.name in rows and rows[binding.name] is None:
        return None  # an outer join's gap reads NULL in every column
    entry = rows.get(binding.name)
    position = entry.position if isinstance(entry, _CteRow) else 0
    return values[position] if position < len(values) else None


def _row_value_of(
    ctx: _EvalContext,
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
            return _input_duration(ctx, binding.alias, column, select)
        key = tag_key(name)
        if key is not None:
            return _input_tag(ctx, binding.alias, key, column, select)
        if name in _RENDITION_SCHEMA:
            # Resolve admitted this name on spec (`RENDITION_COLUMNS`),
            # since only a probe can say whether `alias` is a ladder --
            # this alias's probe found none, so `_Lowerer._bind_renditions`
            # left it a plain `_InputBinding` rather than a rendition row
            # table, and this is that file's own rejection.
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"'{binding.alias}' is a single file, not a ladder: "
                f"input('{ctx.path_of(binding.alias)}') has no renditions",
                column,
                fallback=select,
                hint="rendition columns (bandwidth, width, height, "
                "codecs, name, language) read from an HLS master or "
                "DASH manifest",
            )
    if isinstance(binding, _CteBinding):
        return _cte_value_of(ctx, binding, column, rows, select)
    if not isinstance(binding, _RowBinding):  # defensive: resolve checked it
        raise _error(
            ErrorCode.UNKNOWN_ALIAS,
            f"unknown track-row alias '{_fold(table_node)}'",
            column,
            fallback=select,
            hint=ctx.known_hint(),
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
    ctx: _EvalContext,
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
        return _row_value_of(ctx, value, env, rows, select)
    if isinstance(value, exp.Case):
        return _eval_case(ctx, value, env, rows, select)
    if isinstance(value, exp.Bracket) and isinstance(value.this, exp.Array):
        return _eval_list_element(ctx, value, env, rows, select)
    if isinstance(value, exp.Coalesce) and is_value_expr(value):
        # A value COALESCE (first argument a value, never a stream): the
        # first non-NULL argument, or NULL when every one is absent.
        for argument in [value.this, *value.args.get("expressions", [])]:
            result = _eval_value(ctx, argument, env, rows, select)
            if result is not None:
                return result
        return None
    if isinstance(value, exp.DPipe):
        return _eval_concat(ctx, value, env, rows, select)
    if isinstance(value, _ARITHMETIC):
        return _eval_arithmetic(ctx, value, env, rows, select)
    if isinstance(value, exp.Cast):
        return _eval_cast(ctx, value, env, rows, select)
    if isinstance(value, _BUILTIN_VALUE_FUNCS):
        return _eval_builtin_call(ctx, value, env, rows, select)
    if isinstance(value, exp.Neg) and not isinstance(_unwrap(value.this), exp.Literal):
        operand = _eval_number(ctx, value.this, "'-'", value, env, rows, select)
        return None if operand is None else -operand
    if isinstance(value, exp.Expr):
        call = _call_parts(value)
        if call is not None and not call.namespaced and not call.is_macro:
            name = call.name.lower()
            if name in _VECTOR_BUILTIN_ARITY:
                return _eval_vector_builtin(ctx, name, call, value, env, rows, select)
            declared = ctx.res.wasm.get(name)
            if declared is not None and declared.is_value:
                return _eval_wasm_value(ctx, declared, call, value, env, rows, select)
    return _literal_of(value, select)


def _eval_vector(
    ctx: _EvalContext,
    node: exp.Expr,
    name: str,
    env: _Env,
    rows: _RowTuple,
    select: exp.Select,
) -> tuple[float, ...] | None:
    """One vector-builtin argument's value; anything else is a typed rejection.

    Resolve already checked the STATIC type of every argument
    (:meth:`ffrwd.parser._Resolver._check_vector_builtin_call`); this is
    the runtime mirror for a value resolve could not see through -- a
    CTE's own value column, whose type only lowering knows.
    """
    value = _eval_value(ctx, node, env, rows, select)
    if value is None or isinstance(value, tuple):
        return value
    raise _error(
        ErrorCode.UNSUPPORTED_SQL,
        f"{name}() needs a vector",
        node,
        fallback=select,
        hint=f"{name}() takes a row column or a value function's result, "
        "either typed vector",
    )


def _eval_vector_builtin(
    ctx: _EvalContext,
    name: str,
    call: _Call,
    node: exp.Expr,
    env: _Env,
    rows: _RowTuple,
    select: exp.Select,
) -> RowValue:
    """``cos_similarity``/``vector_length``, evaluated once per row.

    Every argument folds through the same value grammar every other
    builtin call does, memoized wasm calls included. A length mismatch
    between two vectors is the one thing resolve could not already
    reject -- lengths are data, knowable only against the actual
    vectors -- so it is refused here, by name, with both lengths.
    """
    vectors: list[tuple[float, ...]] = []
    for argument in call.args:
        vector = _eval_vector(ctx, argument, name, env, rows, select)
        if vector is None:
            return None
        vectors.append(vector)
    if name == "vector_length":
        return len(vectors[0])
    left, right = vectors
    if len(left) != len(right):
        raise _error(
            ErrorCode.UNSUPPORTED_SQL,
            f"cos_similarity() compares vectors of length {len(left)} and "
            f"{len(right)}",
            node,
            fallback=select,
            hint="cos_similarity() needs two vectors of the same length",
        )
    dot = sum(a * b for a, b in zip(left, right))
    left_mag = math.sqrt(sum(a * a for a in left))
    right_mag = math.sqrt(sum(a * a for a in right))
    if left_mag == 0 or right_mag == 0:
        return 0.0
    return dot / (left_mag * right_mag)


def _eval_arithmetic(
    ctx: _EvalContext,
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
    left = _eval_number(ctx, node.this, operator, node, env, rows, select)
    right = _eval_number(ctx, node.args.get("expression"), operator, node, env, rows, select)
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
    ctx: _EvalContext,
    node: exp.Expr | None,
    operator: str,
    anchor: exp.Expr,
    env: _Env,
    rows: _RowTuple,
    select: exp.Select,
) -> int | float | None:
    """One arithmetic operand's value; text is a typed rejection."""
    value = _eval_value(ctx, node, env, rows, select)
    if value is None or isinstance(value, int | float):
        return value
    raise _error(
        ErrorCode.UNSUPPORTED_SQL,
        f"{operator} needs numbers, but one side is text",
        node if isinstance(node, exp.Expr) else anchor,
        fallback=select,
    )


def _eval_cast(
    ctx: _EvalContext,
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
    value = _eval_value(ctx, node.this, env, rows, select)
    return None if value is None else _tag_text(value)


def _eval_builtin_call(
    ctx: _EvalContext,
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
        text = _eval_text(ctx, node.this, name, env, rows, select)
        if text is None:
            return None
        return text.upper() if isinstance(node, exp.Upper) else text.lower()
    if isinstance(node, exp.Length):
        text = _eval_text(ctx, node.this, name, env, rows, select)
        return None if text is None else len(text)
    if isinstance(node, exp.Round):
        number = _eval_number(ctx, node.this, f"{name}()", node, env, rows, select)
        if number is None:
            return None
        decimals_node = node.args.get("decimals")
        places = 0
        if decimals_node is not None:
            decimals = _eval_number(
                ctx, decimals_node, f"{name}()", node, env, rows, select
            )
            if decimals is None:
                return None
            places = int(decimals)
        rounded = round(number, places)
        return int(rounded) if places <= 0 else rounded
    if isinstance(node, exp.Replace):
        text = _eval_text(ctx, node.this, name, env, rows, select)
        target = _eval_text(ctx, node.args.get("expression"), name, env, rows, select)
        replacement_node = node.args.get("replacement")
        replacement = (
            _eval_text(ctx, replacement_node, name, env, rows, select)
            if replacement_node is not None
            else ""
        )
        if text is None or target is None or replacement is None:
            return None
        return text.replace(target, replacement)
    # exp.Substring: the string, then a 1-based start and an optional length.
    text = _eval_text(ctx, node.this, name, env, rows, select)
    if text is None:
        return None
    start_node = node.args.get("start")
    start = 1
    if start_node is not None:
        value = _eval_number(ctx, start_node, f"{name}()", node, env, rows, select)
        if value is None:
            return None
        start = int(value)
    length_node = node.args.get("length")
    if length_node is None:
        return text[max(start - 1, 0) :]
    value = _eval_number(ctx, length_node, f"{name}()", node, env, rows, select)
    if value is None:
        return None
    end = start - 1 + int(value)
    return text[max(start - 1, 0) : max(end, 0)]


def _eval_text(
    ctx: _EvalContext,
    node: exp.Expr | None,
    name: str,
    env: _Env,
    rows: _RowTuple,
    select: exp.Select,
) -> str | None:
    """One text-function operand's value; a number or boolean is a typed rejection."""
    value = _eval_value(ctx, node, env, rows, select)
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
    ctx: _EvalContext,
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
        _eval_value(ctx, operand_node, env, rows, select)
        if operand_node is not None
        else None
    )
    for branch in node.args.get("ifs") or []:
        if not isinstance(branch, exp.If) or not isinstance(branch.this, exp.Expr):
            raise _error(  # defensive: resolve checked the shape
                ErrorCode.UNSUPPORTED_SQL, "malformed CASE", node, fallback=select
            )
        matched = (
            _eval_row(ctx, branch.this, env, rows, select)
            if operand_node is None
            else _compare(
                exp.EQ(),
                operand,
                _eval_value(ctx, branch.this, env, rows, select),
            )
        )
        if matched is True:
            return _eval_value(ctx, branch.args.get("true"), env, rows, select)
    default = node.args.get("default")
    if not isinstance(default, exp.Expr):
        return None
    return _eval_value(ctx, default, env, rows, select)


def _eval_list_element(
    ctx: _EvalContext,
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
        picked = _eval_value(ctx, node.expressions[0], env, rows, select)
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
    return _literal_of(element, select)


def _eval_concat(
    ctx: _EvalContext,
    node: exp.DPipe,
    env: _Env,
    rows: _RowTuple,
    select: exp.Select,
) -> RowValue:
    """``a || b``: NULL when either side is NULL, else the two texts joined."""
    left = _eval_value(ctx, node.this, env, rows, select)
    right = _eval_value(ctx, node.args.get("expression"), env, rows, select)
    if left is None or right is None:
        return None
    return f"{left}{right}"


def _input_duration(
    ctx: _EvalContext, alias: str, anchor: exp.Expr, select: exp.Select
) -> int | float:
    """``<input>.duration``: the probed container length, in seconds.

    Probed-only, and a rejection when it is not there — an unreadable file
    has no length, and neither does a container that declares none, so
    there is nothing to guess an expression's value from.
    """
    result = ctx.probes.get(alias)
    duration = None if result is None else result.duration
    if duration is None:
        raise _error(
            ErrorCode.INPUT_NOT_FOUND,
            f"'{alias}.{INPUT_DURATION_COLUMN}' is unknown: "
            f"'{ctx.path_of(alias)}' reports no container duration",
            anchor,
            fallback=select,
            hint="the duration is probed from the file; only a readable "
            "input that declares one has it",
        )
    return duration


def _input_tag(
    ctx: _EvalContext, alias: str, key: str, anchor: exp.Expr, select: exp.Select
) -> str | None:
    """``<input>.<tag>``: one probed container tag, NULL when absent.

    An absent key is NULL — that is what lets a CASE fill it — but an input
    this compile could not probe is a rejection, the same rule
    ``duration`` follows: a file nobody read says nothing about its tags.
    """
    result = ctx.probes.get(alias)
    if result is None:
        raise _error(
            ErrorCode.INPUT_NOT_FOUND,
            f"'{alias}.{TAGS_COLUMN}.{key}' is unknown: "
            f"'{ctx.path_of(alias)}' could not be probed",
            anchor,
            fallback=select,
            hint="container tags are read from the file; only a readable "
            "input has them",
        )
    return result.tags.get(key)


def _described_value(
    ctx: _EvalContext, declared: WasmFunction, node: exp.Expr, select: exp.Select
) -> DescribedFunction:
    """What the module's own function turned out to declare, checked.

    Mirrors :func:`ffrwd.admit._described`, for a VALUE function
    instead of a stream one: the export named in the ``functions`` list,
    its parameters matched name-for-name against the declaration, and its
    result type against RETURNS.
    """
    described = ctx.describes.get(declared.module)
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
    _check_wasm_result_type(declared, found, node, select)
    return found


def _eval_wasm_value(
    ctx: _EvalContext,
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
    arguments) within this compile (:attr:`_EvalContext.invoke_cache`).
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
    described = _described_value(ctx, declared, node, select)
    properties = described.params_schema.get("properties")
    known = properties if isinstance(properties, dict) else {}
    args: dict[str, object] = {}
    for param, argument in zip(declared.value_params, call.args):
        value = _eval_value(ctx, argument, env, rows, select)
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
        _check_wasm_param(param.name, value, schema, argument, select)
        args[param.name] = value
    key = (declared.module, declared.export, tuple(sorted(args.items())))
    cached = ctx.invoke_cache.get(key, _UNCACHED)
    if cached is _UNCACHED:
        try:
            result = ctx.invoke(
                declared.module,
                declared.export,
                args,
                described=ctx.describes.get(declared.module),
            )
        except FfrwdError as err:
            raise _error(
                ErrorCode.UNSUPPORTED_SQL,
                f"function '{declared.name}': {err.message}",
                node,
                fallback=select,
                hint=err.hint,
            ) from err
        ctx.invoke_cache[key] = result
    else:
        result = cached
    return _folded_result(declared, result, node, select)


def _sort_key(value: RowValue) -> tuple[int, str, float]:
    """A total, type-stable sort key for one non-NULL row-column value.

    A column's type is static, so the two branches never actually compete
    within one sort — the tuple shape is what keeps the comparison total
    anyway, rather than letting a surprising value raise a TypeError deep
    inside ``list.sort``.
    """
    if isinstance(value, str):
        return (0, value, 0.0)
    if isinstance(value, tuple):  # defensive: resolve never admits a vector sort key
        return (2, "", 0.0)
    return (1, "", float(value if value is not None else 0))


def _computed_arg(
    ctx: _EvalContext,
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
    return _literal_node(_eval_value(ctx, node, env, row, select), node)


def _wasm_params(
    ctx: _EvalContext,
    declared: WasmFunction,
    described: Described,
    call: _Call,
    node: exp.Expr,
    select: exp.Select,
    env: _Env,
    row: _RowTuple,
    *,
    first: int,
    params_schema: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """The value arguments as the module's own parameters, schema-checked.

    `first` is the index the value arguments start at: past the streams,
    and past the annotation column when the call wrote it explicitly.
    A parameter left NULL or unwritten is OMITTED, the way absence works
    everywhere else in the dialect -- the module then sees its own
    default. What is written is checked against the schema the module
    declares, by name and by type.

    `params_schema` overrides where that schema is read from, for a call
    whose parameters belong to one FUNCTION of the module rather than to
    the module's single export.
    """
    schema_source = (
        described.params_schema if params_schema is None else params_schema
    )
    properties = schema_source.get("properties")
    known = properties if isinstance(properties, dict) else {}
    params: dict[str, object] = {}
    for index, param in enumerate(declared.value_params, start=first):
        written = call.args[index] if index < len(call.args) else param.default
        if written is None:
            continue
        value = _eval_value(ctx, written, env, row, select)
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
        _check_wasm_param(param.name, value, schema, anchor, select)
        params[param.name] = value
    return params


def _literal_node(value: RowValue, source: exp.Expr) -> exp.Expr:
    """A computed value back as the literal node the option binder reads.

    The synthesized node inherits `source`'s position, so an option that
    rejects what a row computed still points at the expression that wrote it.
    """
    if isinstance(value, tuple):  # defensive: resolve never admits a vector option
        raise _error(
            ErrorCode.UNSUPPORTED_SQL,
            "a vector cannot be an option value",
            source,
            hint="read vector_length(...) or cos_similarity(...) instead",
        )
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
