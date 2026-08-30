"""psql-style variable substitution for query text, and the header that declares them.

``substitute(text, variables)`` scans `text` for three reference forms --
``:'name'`` (quoted string literal), ``:"name"`` (quoted identifier), and
bare ``:name`` (raw text) -- and replaces each with a value from
`variables`. The scan skips the same opaque spans the SQL lexer itself
does: ``'...'`` strings, ``"..."`` identifiers, ``--`` line comments, and
``/* */`` block comments. A ``::`` cast and a lone ``:`` pass through
unchanged.

Each form takes an optional list subscript, ``:name[k]``: the value splits
on commas and the reference substitutes to ONE element, 1-based, quoted
the way the form asks. A literal subscript picks its element right here; a
row-column subscript (``:widths[i.i]``) cannot be known until lowering, so
the reference substitutes to an ``ARRAY[...]`` element access -- raw
elements for ``:name``, string literals for ``:'name'`` -- and lowering
reads the element per row. An identifier reference names a compile-time
name, so ``:"name"[...]`` takes a literal subscript only. A subscript past
the end of the list is a rejection naming the list's length. Without a
subscript, a comma-carrying value stays exactly the one raw text it
always was.

An UNSET reference substitutes to the bare keyword ``NULL`` -- absence, which
every binding site treats as "not written" and the required positions reject.
The returned :class:`Substitution` maps each such NULL's (line, col) in the
substituted text back to the variable's name, so a rejection at the NULL's
point of use can say ``':source' was not set`` instead of "NULL is not a
path". The map exists for messages only.

``referenced(text)`` reports which names `text` references, over the same
scan. ``declared_variables(text)`` reads the other direction: the
``-- variables:`` header a runnable query carries, naming what the reader
has to supply::

    -- variables: source (input media path), prefix (output name prefix)

It is a comment, so nothing enforces it and a query without one declares
nothing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .errors import ErrorCode, FfrwdError

__all__ = [
    "Substitution",
    "Variable",
    "declared_variables",
    "referenced",
    "substitute",
    "unset_error",
    "unset_variable",
]

_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

# A list subscript's two accepted bodies: a positive integer literal, or a
# dot-qualified row column (`i.i`, `r.name`). Anything else inside `[...]`
# after a reference is a rejection, never silently left in the text.
_INT_BODY_RE = re.compile(r"[0-9]+\Z")
_COLUMN_BODY_RE = re.compile(
    r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)+\Z"
)

_NULL = "NULL"


@dataclass(frozen=True)
class Substitution:
    """Substituted query text, plus where each unset variable's NULL landed.

    `unset` is keyed by the NULL keyword's 1-based (line, col) in `text` --
    the same coordinates sqlglot records on the parsed node, which is how a
    later rejection finds the variable's name.
    """

    text: str
    unset: dict[tuple[int, int], str] = field(default_factory=dict)


def substitute(text: str, variables: dict[str, str]) -> Substitution:
    out: list[str] = []
    length = 0  # of the output built so far
    nulls: list[tuple[int, str]] = []  # (output offset, variable name)
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "'" or ch == '"':
            end = _scan_quoted(text, i, ch)
            out.append(text[i:end])
            length += end - i
            i = end
            continue
        if text.startswith("--", i):
            end = text.find("\n", i)
            end = n if end == -1 else end
            out.append(text[i:end])
            length += end - i
            i = end
            continue
        if text.startswith("/*", i):
            end = text.find("*/", i + 2)
            end = n if end == -1 else end + 2
            out.append(text[i:end])
            length += end - i
            i = end
            continue
        if ch == ":" and text.startswith("::", i):
            out.append("::")
            length += 2
            i += 2
            continue
        if ch == ":":
            found = _match_reference(text, i)
            if found is not None:
                name, ref_end = found
                subscript = _match_subscript(text, ref_end)
                if subscript is not None:
                    ref_end = subscript[1]
                if name not in variables:
                    # Unset stays NULL-is-absence, subscripted or not.
                    replacement = _NULL
                    nulls.append((length, name))
                elif subscript is not None:
                    replacement = _element(
                        text[i + 1],
                        name,
                        variables[name],
                        subscript[0],
                        at=_line_col(text, i),
                    )
                else:
                    replacement = _replacement(text[i + 1], variables[name])
                out.append(replacement)
                length += len(replacement)
                i = ref_end
                continue
        out.append(ch)
        length += 1
        i += 1
    result = "".join(out)
    return Substitution(
        text=result,
        unset={_line_col(result, offset): name for offset, name in nulls},
    )


def referenced(text: str) -> set[str]:
    """Every variable name `text` references, over the same scan as `substitute`."""
    names: set[str] = set()
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "'" or ch == '"':
            i = _scan_quoted(text, i, ch)
            continue
        if text.startswith("--", i):
            end = text.find("\n", i)
            i = n if end == -1 else end
            continue
        if text.startswith("/*", i):
            end = text.find("*/", i + 2)
            i = n if end == -1 else end + 2
            continue
        if ch == ":" and text.startswith("::", i):
            i += 2
            continue
        if ch == ":":
            found = _match_reference(text, i)
            if found is not None:
                name, ref_end = found
                subscript = _match_subscript(text, ref_end)
                if subscript is not None:
                    ref_end = subscript[1]
                names.add(name)
                i = ref_end
                continue
        i += 1
    return names


def _scan_quoted(text: str, start: int, quote: str) -> int:
    """End offset (exclusive) of the ``'...'``/``"..."`` run at `start`; a
    doubled quote (``''``, ``\"\"``) stays inside the run."""
    i = start + 1
    n = len(text)
    while i < n:
        if text[i] == quote:
            if i + 1 < n and text[i + 1] == quote:
                i += 2
                continue
            return i + 1
        i += 1
    return n


def _match_reference(text: str, start: int) -> tuple[str, int] | None:
    """The `:name`/`:'name'`/`:"name"` reference at `start` as (name, end
    offset), or None if nothing there fits the shape (the caller copies the
    colon as-is)."""
    next_ch = text[start + 1] if start + 1 < len(text) else ""
    quote = next_ch if next_ch in ("'", '"') else ""
    name_start = start + 2 if quote else start + 1
    match = _NAME_RE.match(text, name_start)
    if match is None:
        return None
    name_end = match.end()
    if quote:
        if name_end >= len(text) or text[name_end] != quote:
            return None
        ref_end = name_end + 1
    else:
        ref_end = name_end
    return match.group(), ref_end


def _match_subscript(text: str, start: int) -> tuple[str, int] | None:
    """The ``[<body>]`` list subscript at `start` as (body, end offset), or
    None when the reference has none (no ``[`` directly after it, or no
    ``]`` on the same line -- a real subscript never spans one)."""
    if start >= len(text) or text[start] != "[":
        return None
    close = text.find("]", start + 1)
    if close == -1:
        return None
    body = text[start + 1 : close]
    if "\n" in body:
        return None
    return body.strip(), close + 1


def _element(
    quote: str, name: str, value: str, body: str, at: tuple[int, int]
) -> str:
    """One element of a SET list variable, or the ``ARRAY[...]`` access that
    reads it per row; a subscript the grammar has no reading for is a
    rejection, never text left in the query."""
    line, col = at
    elements = value.split(",")
    if _INT_BODY_RE.match(body):
        index = int(body)
        if index == 0:
            raise FfrwdError(
                ErrorCode.UNSUPPORTED_SQL,
                f"':{name}[0]' subscripts below the start of the list",
                line=line,
                col=col,
                hint="list subscripts are 1-based: :"
                f"{name}[1] is the first element",
            )
        if index > len(elements):
            raise FfrwdError(
                ErrorCode.UNSUPPORTED_SQL,
                f"':{name}[{index}]' is past the end: the list has "
                f"{_count(len(elements))}",
                line=line,
                col=col,
                hint=f"-v {name}=... splits on commas; subscript from 1 to "
                f"{len(elements)}",
            )
        return _replacement(quote, elements[index - 1])
    if _COLUMN_BODY_RE.match(body):
        if quote == '"':
            raise FfrwdError(
                ErrorCode.UNSUPPORTED_SQL,
                f'\':"{name}"[{body}]\' picks an identifier per row',
                line=line,
                col=col,
                hint="an identifier is a compile-time name, so its subscript "
                f'must be an integer literal, e.g. :"{name}"[1]',
            )
        if quote == "'":
            listed = ",".join(
                "'" + element.replace("'", "''") + "'" for element in elements
            )
        else:
            listed = value
        return f"ARRAY[{listed}][{body}]"
    raise FfrwdError(
        ErrorCode.UNSUPPORTED_SQL,
        f"':{name}[{body}]' is not a list subscript",
        line=line,
        col=col,
        hint="a list subscript is a positive integer literal or a row "
        f"column, e.g. :{name}[1] or :{name}[i.i]",
    )


def _count(n: int) -> str:
    """``n`` elements, spelled for a message."""
    return f"{n} element" + ("" if n == 1 else "s")


def _replacement(quote: str, value: str) -> str:
    """A set variable's value, quoted the way the reference form asks."""
    if quote == "'":
        return "'" + value.replace("'", "''") + "'"
    if quote == '"':
        return '"' + value.replace('"', '""') + '"'
    return value


def _line_col(text: str, offset: int) -> tuple[int, int]:
    """1-indexed (line, col) of `offset` in `text`."""
    line = text.count("\n", 0, offset) + 1
    last_newline = text.rfind("\n", 0, offset)
    col = offset - last_newline
    return line, col


# -- the unset-variable rejection, built and read back in one place --------

_UNSET_RE = re.compile(r"^':([A-Za-z_][A-Za-z0-9_]*)' was not set\b")


def unset_error(
    code: ErrorCode,
    name: str,
    *,
    what: str,
    line: int | None,
    col: int | None,
) -> FfrwdError:
    """The rejection for an unset variable's NULL landing where a value is
    required. `what` says what the position needed, and lands in the hint."""
    return FfrwdError(
        code,
        f"':{name}' was not set",
        line=line,
        col=col,
        hint=f"{what}; set it with -v {name}=<value>",
    )


def unset_variable(err: FfrwdError) -> str | None:
    """The variable name an :func:`unset_error` rejection is about, or None."""
    match = _UNSET_RE.match(err.message)
    return match.group(1) if match is not None else None


# -- the declaring header --------------------------------------------------

_HEADER_RE = re.compile(r"^--\s*variables:\s*(?P<body>.+)$", re.MULTILINE)


@dataclass(frozen=True)
class Variable:
    """One variable a query declares: its name, and what the header says it is."""

    name: str
    description: str = ""


def declared_variables(text: str) -> tuple[Variable, ...]:
    """The variables `text`'s ``-- variables:`` header declares, in written order.

    Empty for a query with no such header: the header is documentation, and a
    query is free not to carry one. A description is whatever the parentheses
    after a name hold, commas and all; a name written without them declares
    itself and nothing more.
    """
    header = _HEADER_RE.search(text)
    if header is None:
        return ()
    body = header.group("body")
    found: list[Variable] = []
    at = 0
    while at < len(body):
        match = _NAME_RE.search(body, at)
        if match is None:
            break
        description, at = _description(body, match.end())
        found.append(Variable(name=match.group(), description=description))
        # Past the separating comma, so a description's own words are not read
        # as further names.
        comma = body.find(",", at)
        at = len(body) if comma == -1 else comma + 1
    return tuple(found)


def _description(body: str, start: int) -> tuple[str, int]:
    """The ``(...)`` description at `start`, and where it ends. ("", start) if there is none."""
    at = start
    while at < len(body) and body[at].isspace():
        at += 1
    if at >= len(body) or body[at] != "(":
        return "", start
    depth = 0
    for end in range(at, len(body)):
        if body[end] == "(":
            depth += 1
        elif body[end] == ")":
            depth -= 1
            if depth == 0:
                return body[at + 1 : end].strip(), end + 1
    return body[at + 1 :].strip(), len(body)  # unclosed: the rest of the line
