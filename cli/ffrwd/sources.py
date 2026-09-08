"""A FROM item with no file behind it.

Two kinds: an ffmpeg generator (``ffmpeg.testsrc(...)``), which is one
statically-typed stream and no ``-i`` at all, and a wasm module that answers
with the urls to open, whose every row is checked here before anything is
probed.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from ffrwd.bindings import _SourceBinding
from ffrwd.errors import FfrwdError
from ffrwd.merge import RowValue
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
