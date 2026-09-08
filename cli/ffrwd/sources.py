"""A FROM item with no file behind it.

Two kinds: an ffmpeg generator (``ffmpeg.testsrc(...)``), which is one
statically-typed stream and no ``-i`` at all, and a wasm module that answers
with the urls to open, whose every row is checked here before anything is
probed.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from sqlglot import exp

from ffrwd.bindings import _SourceBinding
from ffrwd.errors import ErrorCode, FfrwdError
from ffrwd.expressions import _error
from ffrwd.functions import WasmFunction
from ffrwd.merge import RowValue
from ffrwd.types import RowColumnType
from ffrwd.values import _ARRAY_COLUMNS

# What a URL source's row names its input with, the attributes a row MAY
# carry beside it, and the ones it may not: width, height and the stream
# arrays are what the probe of that url reports, so a row naming one would
# be overruled by the file itself.
_URL_SOURCE_URL = "url"
_URL_SOURCE_TEXT = ("codecs", "name", "language")
_URL_SOURCE_PROBED = frozenset({"width", "height"}) | frozenset(_ARRAY_COLUMNS)


# A value column's name: a plain lowercase identifier, so every column a
# module names is one a query can spell.
_URL_SOURCE_NAME = re.compile(r"[a-z][a-z0-9_]*")


_URL_SOURCE_SHAPE_HINT = (
    'a source answers with {"rows": [{"url": "clip.mp4", ...}, ...]}, one '
    "object per row"
)


@dataclass(frozen=True)
class _UrlRow:
    """One checked row of a URL source's answer.

    The four rendition attributes a row may declare (a probe cannot report
    them: they describe the ladder, not the file), and `columns` -- every
    other key it named, which is a value column of the alias.
    """

    url: str
    bandwidth: int | None
    codecs: str | None
    name: str | None
    language: str | None
    columns: dict[str, RowValue]

def _url_source_row(
    alias: str,
    position: int,
    entry: object,
    refuse: Callable[[str, str], FfrwdError],
) -> _UrlRow:
    """One row of a URL source's answer: its url, attributes and columns."""
    if not isinstance(entry, dict):
        raise refuse(
            f"row {position} of '{alias}' is {entry!r}",
            _URL_SOURCE_SHAPE_HINT,
        )
    url = entry.get("url")
    if not isinstance(url, str) or not url:
        raise refuse(
            f"row {position} of '{alias}' names no url",
            "every row a source produces names the url to open, e.g. "
            '{"url": "clip.mp4"}',
        )
    bandwidth: int | None = None
    text: dict[str, str | None] = dict.fromkeys(_URL_SOURCE_TEXT)
    columns: dict[str, RowValue] = {}
    for key, value in entry.items():
        if key == _URL_SOURCE_URL:
            continue
        if key in _URL_SOURCE_PROBED:
            raise refuse(
                f"row {position} of '{alias}' names '{key}'",
                f"'{key}' is what the probe reports; drop it from the row",
            )
        if key == "bandwidth":
            if value is None or (
                isinstance(value, int) and not isinstance(value, bool)
            ):
                bandwidth = value
                continue
            raise refuse(
                f"row {position} of '{alias}' gives 'bandwidth' {value!r}",
                "'bandwidth' is a whole number of bits per second, or null",
            )
        if key in _URL_SOURCE_TEXT:
            if value is None or isinstance(value, str):
                text[key] = value
                continue
            raise refuse(
                f"row {position} of '{alias}' gives '{key}' {value!r}",
                f"'{key}' is a string, or null",
            )
        if _URL_SOURCE_NAME.fullmatch(key) is None:
            raise refuse(
                f"row {position} of '{alias}' names the column '{key}'",
                "a source's column names are lowercase identifiers, e.g. "
                "sequence",
            )
        if value is not None and not isinstance(value, str | int | float | bool):
            raise refuse(
                f"row {position} of '{alias}' gives '{key}' {value!r}",
                "a source's column value is a string, a number, a boolean "
                "or null",
            )
        columns[key] = value
    return _UrlRow(
        url=url,
        bandwidth=bandwidth,
        codecs=text["codecs"],
        name=text["name"],
        language=text["language"],
        columns=columns,
    )


def _source_columns_hint(binding: _SourceBinding) -> str:
    return f"'{binding.display}' exposes {binding.alias}.{binding.output}"


def _url_source_payload(
    alias: str,
    declared: WasmFunction,
    params: Mapping[str, object],
    answered: object,
    node: exp.Expr,
    select: exp.Select,
) -> _UrlPayload:
    """The module's JSON answer as this alias's rows, checked.

    One object: ``rows`` (required, at least one), ``document`` and
    ``bounded`` beside it. Each row names a ``url`` and may name the
    rendition attributes a probe cannot report -- everything else it
    names is a value column of the alias, which is why the rows all have
    to name the same ones.
    """

    def refuse(message: str, hint: str) -> FfrwdError:
        return _error(
            ErrorCode.UNSUPPORTED_SQL, message, node, fallback=select, hint=hint
        )

    if not isinstance(answered, dict):
        raise refuse(
            f"'{declared.name}()' returned {answered!r}, and a source "
            "returns an object of rows",
            _URL_SOURCE_SHAPE_HINT,
        )
    written = answered.get("rows")
    if not isinstance(written, list):
        raise refuse(
            f"'{declared.name}()' returned no 'rows' list",
            _URL_SOURCE_SHAPE_HINT,
        )
    if not written:
        raise refuse(
            f"'{alias}' produced no rows",
            f"'{declared.name}()' answered nothing for "
            f"{_written_params(params)}; a source that produces no rows "
            "selects nothing",
        )
    document = answered.get("document")
    if document is not None and not isinstance(document, str):
        raise refuse(
            f"'{declared.name}()' returned a 'document' that is "
            f"{document!r}",
            "a source's 'document' is the text it wrote beside its rows",
        )
    bounded = answered.get("bounded", True)
    if not isinstance(bounded, bool):
        raise refuse(
            f"'{declared.name}()' returned a 'bounded' that is {bounded!r}",
            "a source's 'bounded' says whether its rows end; it is true "
            "or false, and defaults to true",
        )
    rows = [
        _url_source_row(alias, position, entry, refuse)
        for position, entry in enumerate(written, start=1)
    ]
    return _UrlPayload(
        document=document,
        bounded=bounded,
        rows=rows,
        types=_url_source_types(alias, rows, refuse),
    )


@dataclass(frozen=True)
class _UrlPayload:
    """A URL source's whole answer, checked: its rows and what came beside."""

    document: str | None
    bounded: bool
    rows: list[_UrlRow]
    types: dict[str, RowColumnType]


def _written_params(params: Mapping[str, object]) -> str:
    """A call's folded arguments, as a message names them."""
    written = ", ".join(f"{name} = {value!r}" for name, value in sorted(params.items()))
    return written or "no arguments"


def _url_column_type(value: str | int | float | bool) -> RowColumnType:
    """The row-column type one JSON scalar reads as."""
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int | float):
        return "number"
    return "text"


def _url_source_types(
    alias: str,
    rows: Sequence[_UrlRow],
    refuse: Callable[[str, str], FfrwdError],
) -> dict[str, RowColumnType]:
    """The value columns a URL source's rows expose, and the type of each.

    A written row table's own two rules: every row names the same columns,
    and every row of a column carries the same type, with null fitting any
    of them and an all-null column reading as text the way Postgres types
    one.
    """
    columns = tuple(rows[0].columns)
    known = set(columns)
    for position, row in enumerate(rows[1:], start=2):
        missing = known - set(row.columns)
        unexpected = set(row.columns) - known
        if missing or unexpected:
            odd = sorted(missing | unexpected)[0]
            raise refuse(
                f"row {position} of '{alias}' does not name the same columns "
                f"row 1 does ({', '.join(columns) or 'none'}): '{odd}' "
                f"{'is missing' if odd in missing else 'is unexpected'}",
                "every row a source produces names the same columns",
            )
    types: dict[str, RowColumnType] = {}
    for column in columns:
        settled: RowColumnType | None = None
        for row in rows:
            value = row.columns[column]
            if value is None:
                continue
            if isinstance(value, tuple):
                raise refuse(
                    f"column '{alias}.{column}' is a vector",
                    "a source's columns are text, number or boolean; a vector "
                    "rides in a row a module writes, not in a source's catalog",
                )
            written = _url_column_type(value)
            if settled is not None and written != settled:
                raise refuse(
                    f"column '{alias}.{column}' holds both {settled} and "
                    f"{written}",
                    "every row of a column carries the same type; null fits "
                    "any of them",
                )
            settled = written
        types[column] = settled or "text"
    return types
