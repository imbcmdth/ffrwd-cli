"""Fuzz test for guardrail #7 ("no panics on user input").

Seed corpus: the ``.sql`` text of every fixture under ``tests/golden/`` (both
the accepted queries and the ones that are expected to be rejected -- the
mutation strategy does not care which) plus every recipe under
``packages/*/*/recipes/``, which is where the row/grouping surface lives:
``unnest``, ``array_agg`` +
``GROUP BY``, chapter fan-out, CSV to STDOUT, and the ``:'name'`` variable
forms. Each example takes a seed, applies a handful of random mutations to it
-- slice deletion, slice duplication, single-character swap, random-token
injection, random-unicode injection -- and runs the result through
``ffrwd.vars.substitute`` before compiling, so the variable scanner is
fuzzed by the same mutants.

Property (spec: "Fuzz" in the Testing section, guardrail #7): on arbitrarily
mutated SQL text ``compile_sql`` returns a ``Graph`` and ``compile_table_sql``
a list of ``TableSink``, or either raises ``FfrwdError`` -- and the error
code is NEVER ``INTERNAL``. ``INTERNAL`` is the bug backstop: every one it
fires on is an unhandled Python exception somewhere in parse/resolve/lower,
to be fixed at its source with a typed rejection.

Probing is stubbed, drawn per example between a rich synthetic
``ProbeResult`` (which is what makes track rows, chapters and subscript
bounds reachable) and ``None`` (the unreadable-input path). That keeps the
test hermetic and off the network -- a mutant can grow a ``://`` and a real
probe would try to fetch it.

Long hunt (local, not CI): raise the example count and vary the seed::

    FFRWD_FUZZ_EXAMPLES=20000 pytest tests/test_fuzz.py --hypothesis-seed=1
    ($env:FFRWD_FUZZ_EXAMPLES = 20000 in PowerShell)
"""

from __future__ import annotations

import contextlib
import os
import re
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st

from ffrwd import compiler
from ffrwd.compiler import compile_sql, compile_table_sql
from ffrwd.errors import ErrorCode, FfrwdError
from ffrwd.ir import Graph
from ffrwd.probe import ChapterMeta, ProbeResult, StreamMeta
from ffrwd.table import TableSink
from ffrwd.vars import substitute

from .test_examples import _parse as parse_examples

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = _PROJECT_ROOT.parent
_GOLDEN_DIR = _PROJECT_ROOT / "tests" / "golden"
_COOKBOOK = (_REPO_ROOT / "docs" / "examples.md", _REPO_ROOT / "docs" / "corpus.md")
_CORPUS: list[str] = sorted(
    {
        *(p.read_text(encoding="utf-8") for p in _GOLDEN_DIR.glob("*.sql")),
        # The cookbook and its corpus are where the row and grouping surface
        # lives, and both are already kept honest by their own pins.
        *(
            example.sql
            for page in _COOKBOOK
            for example in parse_examples(page.read_text(encoding="utf-8"))
        ),
    }
)
assert _CORPUS, f"no seed queries found under {_GOLDEN_DIR}"

_MAX_EXAMPLES = int(os.environ.get("FFRWD_FUZZ_EXAMPLES", "300"))
_MAX_MUTATIONS_PER_EXAMPLE = 5

# Values for the `:name` references the corpus makes, so a seed reaches the
# compiler instead of stopping at an undefined variable. The named ones match
# the synthetic probe below (so the metadata predicates select a row rather
# than an empty one); anything else is a path.
_VARIABLE_VALUES = {
    "width": "1920",
    "height": "1080",
    "codec": "aac",
    "language": "eng",
    "title": "My Film",
    "artist": "Me",
    "ext": "m4a",
    "prefix": "ch",
    "dir": "clock",
    "crf": "23",
    "rate": "1",
    "gain": "0.5",
    "factor": "2",
    "start": "5",
    "end": "60",
    "cut": "120",
    "at": "10",
    "duration": "1",
    "w": "640",
    "h": "480",
    "x": "100",
    "y": "50",
    "high_w": "1920",
    "mid_w": "1280",
    "low_w": "854",
    "vcodec": "libx264",
    "acodec": "aac",
    "video_bitrate": "4M",
    "audio_bitrate": "192k",
    "preset": "slow",
    "scale": "iw/2",
    "i": "-23",
    "tp": "-2",
    "lra": "7",
    "fps": "12",
}
_REFERENCE_RE = re.compile(r"(?<![:\w]):['\"]?([A-Za-z_][A-Za-z0-9_]*)")
_VARIABLES = {
    name: _VARIABLE_VALUES.get(name, "in.mp4")
    for text in _CORPUS
    for name in _REFERENCE_RE.findall(text)
}

# Tokens drawn from the SQL surface (spec.md dialect + stdlib) plus a few
# generic troublemakers, injected at random positions.
_TOKENS: list[str] = [
    "SELECT",
    "FROM",
    "WHERE",
    "UNION ALL",
    "WITH",
    "AS",
    "BETWEEN",
    "AND",
    "GROUP BY",
    "ORDER BY",
    "COPY",
    "TO",
    "STDOUT",
    "CREATE VIEW",
    "JOIN",
    "ON",
    "COALESCE(",
    "unnest(",
    "array_agg(",
    "input(",
    "chapters",
    "::text",
    "||",
    "=>",
    "format 'csv'",
    "header true",
    ":'source'",
    ':"source"',
    ":rate",
    ")",
    "(",
    ",",
    ".",
    "'",
    '"',
    ";",
    "--",
    "/*",
    "*/",
    "\n",
    "\t",
    "scale",
    "crop",
    "overlay",
    "hflip",
    "blur_regions",
    "nope",
    "NULL",
    "TRUE",
    "0",
    "-1",
    "0.5",
    "1e400",
    "🎬",  # movie camera emoji
    "™",  # trademark sign
    " ",  # line separator
    "﻿",  # byte order mark
    "İ",  # dotted capital I (Turkish)
]

_Draw = Callable[[st.SearchStrategy[Any]], Any]

# One video, one audio, one subtitle track and two chapters: enough metadata
# for the track-row, chapter and subscript paths to lower for real.
_PROBE = ProbeResult(
    streams=[
        StreamMeta(
            type="video",
            index=0,
            metadata={},
            width=1920,
            height=1080,
            fps="30/1",
            sample_rate=None,
            codec="h264",
        ),
        StreamMeta(
            type="audio",
            index=0,
            metadata={"language": "eng"},
            width=None,
            height=None,
            fps=None,
            sample_rate=48000,
            codec="aac",
            channels=2,
            channel_layout="stereo",
        ),
        StreamMeta(
            type="subtitle",
            index=0,
            metadata={"language": "eng"},
            width=None,
            height=None,
            fps=None,
            sample_rate=None,
            codec="srt",
        ),
    ],
    chapters=[
        ChapterMeta(index=1, start_t=0.0, end_t=30.0, title="Intro"),
        ChapterMeta(index=2, start_t=30.0, end_t=90.0, title="Credits"),
    ],
)


def _delete_slice(text: str, draw: _Draw) -> str:
    n = len(text)
    start = draw(st.integers(min_value=0, max_value=n - 1))
    length = draw(st.integers(min_value=1, max_value=n - start))
    return text[:start] + text[start + length :]


def _duplicate_slice(text: str, draw: _Draw) -> str:
    n = len(text)
    start = draw(st.integers(min_value=0, max_value=n - 1))
    length = draw(st.integers(min_value=1, max_value=n - start))
    end = start + length
    return text[:end] + text[start:end] + text[end:]


def _swap_chars(text: str, draw: _Draw) -> str:
    n = len(text)
    a = draw(st.integers(min_value=0, max_value=n - 1))
    b = draw(st.integers(min_value=0, max_value=n - 1))
    chars = list(text)
    chars[a], chars[b] = chars[b], chars[a]
    return "".join(chars)


def _inject_token(text: str, draw: _Draw) -> str:
    pos = draw(st.integers(min_value=0, max_value=len(text)))
    token = draw(st.sampled_from(_TOKENS))
    return text[:pos] + token + text[pos:]


def _inject_unicode(text: str, draw: _Draw) -> str:
    pos = draw(st.integers(min_value=0, max_value=len(text)))
    ch = draw(st.characters())
    return text[:pos] + ch + text[pos:]


def _mutate_once(text: str, draw: _Draw) -> str:
    ops = ["inject_token", "inject_unicode"]
    if len(text) >= 1:
        ops += ["delete", "duplicate"]
    if len(text) >= 2:
        ops += ["swap"]
    op = draw(st.sampled_from(ops))
    if op == "delete":
        return _delete_slice(text, draw)
    if op == "duplicate":
        return _duplicate_slice(text, draw)
    if op == "swap":
        return _swap_chars(text, draw)
    if op == "inject_unicode":
        return _inject_unicode(text, draw)
    return _inject_token(text, draw)


@contextlib.contextmanager
def _stubbed_probe(result: ProbeResult | None) -> Iterator[None]:
    """`compiler.probe_path` fixed to `result` -- no file, no network."""
    original = compiler.probe_path
    compiler.probe_path = lambda path, args=(), **kw: result
    try:
        yield
    finally:
        compiler.probe_path = original


def _mutated(data: st.DataObject) -> str | None:
    """A mutated seed with its variable references substituted, or None when
    substitution itself rejected it (an undefined variable a mutation made up).

    Anything but a `FfrwdError` out of `substitute` fails the example, which
    is half of what these tests are for.
    """
    text = data.draw(st.sampled_from(_CORPUS), label="seed")
    num_mutations = data.draw(
        st.integers(min_value=1, max_value=_MAX_MUTATIONS_PER_EXAMPLE),
        label="num_mutations",
    )
    for _ in range(num_mutations):
        text = _mutate_once(text, data.draw)
    try:
        return substitute(text, _VARIABLES).text
    except FfrwdError as err:
        assert err.code is not ErrorCode.INTERNAL, f"INTERNAL on: {text!r}"
        return None


def _drawn_probe(data: st.DataObject) -> ProbeResult | None:
    return _PROBE if data.draw(st.booleans(), label="probed") else None


@given(data=st.data())
@settings(max_examples=_MAX_EXAMPLES, deadline=None)
def test_compile_sql_never_panics_on_mutated_corpus(data: st.DataObject) -> None:
    text = _mutated(data)
    if text is None:
        return
    with _stubbed_probe(_drawn_probe(data)):
        try:
            result = compile_sql(text)
        except FfrwdError as err:
            assert err.code is not ErrorCode.INTERNAL, f"INTERNAL on: {text!r}"
            return
    assert isinstance(result, Graph)


@given(data=st.data())
@settings(max_examples=_MAX_EXAMPLES, deadline=None)
def test_compile_table_sql_never_panics_on_mutated_corpus(data: st.DataObject) -> None:
    """The same promise for the table/csv half of the pipeline, which the CLI
    reaches for every bare SELECT."""
    text = _mutated(data)
    if text is None:
        return
    with _stubbed_probe(_drawn_probe(data)):
        try:
            sinks = compile_table_sql(text)
        except FfrwdError as err:
            assert err.code is not ErrorCode.INTERNAL, f"INTERNAL on: {text!r}"
            return
    assert all(isinstance(sink, TableSink) for sink in sinks)
