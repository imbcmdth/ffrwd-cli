"""Reading a written expression, and refusing one.

The bottom of lowering's vocabulary: what a node was written as, what it
reads back to, and the node-anchored :class:`~ffrwd.errors.FfrwdError` every
rejection is built from. Nothing here consults a probe, a registry or a
graph, so every lowering module can depend on it.
"""

from __future__ import annotations

from sqlglot import exp

from ffrwd.errors import ErrorCode, FfrwdError
from ffrwd.parser import _ident_name as _fold
from ffrwd.parser import _pos


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


def _coalesce_label(node: exp.Expr) -> str:
    """How a rejection names a COALESCE argument: as it was written."""
    written = _unwrap(node)
    if isinstance(written, exp.Column | exp.Bracket):
        return written.sql()
    return _describe(written)


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


def _unwrap(node: exp.Expr) -> exp.Expr:
    """Strip projection aliases and redundant parentheses."""
    while True:
        if isinstance(node, exp.Alias | exp.Paren):
            inner = node.this
            if isinstance(inner, exp.Expr):
                node = inner
                continue
        return node


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


def _sql_text(node: exp.Expr) -> str:
    """The argument as the user wrote it, for a BROADCAST_MISMATCH message.

    ``dialect="postgres"`` matters: it re-adds the ``INDEX_OFFSET`` sqlglot
    subtracted at parse time, so ``a.audio[2]`` renders as ``a.audio[2]``.
    """
    return str(node.sql(dialect="postgres"))
