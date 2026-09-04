"""End-to-end exec proof for `embed_text`, `fauxlate`'s `RETURNS vector` export.

`fauxlate` (the sidecar's own fleet module, `sidecar/modules/fauxlate`) is
otherwise a fake translator; `embed_text` is its third export, a fake
embedder standing in for a real one -- see
`sidecar/modules/fauxlate/src/lib.rs` for the letter-count rule. This is the
module docs/examples.md recipe 114, "Rank rows by a vector", runs for real.

Every test here reaches a bare `SELECT` -- no `COPY`, so `compile_table_sql`
is the path lowering takes, not `compile_sql`. That path is what these tests
guard: `compile_table_sql` once built its result set without describing any
`LANGUAGE wasm` module first, so a value function reached from a bare SELECT
(exactly what recipe 114 is) failed the same way regardless of the module or
its RETURNS.

Requires the `ffrwd-wasm` sidecar (`uv sync --extra wasm`) and `fauxlate`
built for `wasm32-wasip2`; skips cleanly when either is missing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ffrwd import binaries
from ffrwd.compiler import compile_table_sql

_CLI_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = _CLI_ROOT.parent
_MODULE = (
    _REPO_ROOT / "sidecar" / "modules" / "target" / "wasm32-wasip2" / "release" / "fauxlate.wasm"
)


@pytest.fixture(autouse=True)
def _require_everything() -> None:
    if binaries.ffrwd_wasm_path() is None:
        pytest.skip("ffrwd-wasm not found (uv sync --extra wasm)")
    if not _MODULE.exists():
        pytest.skip(
            f"module missing: {_MODULE} (cargo build --target wasm32-wasip2 "
            f"--release -p fauxlate, from sidecar/modules)"
        )


_DECLARE = (
    "CREATE FUNCTION embed_text(prompt text) RETURNS vector\n"
    f"  AS '{_MODULE.as_posix()}', 'embed_text' LANGUAGE wasm;\n"
)


_ONE_ROW = "FROM unnest(ARRAY[STRUCT(1 AS x)]) t"  # a one-row source; no column it carries matters


def _score(expression: str) -> float:
    """The one cell a single-row, single-column bare SELECT prints."""
    sinks = compile_table_sql(_DECLARE + f"SELECT {expression} AS score {_ONE_ROW}")
    assert len(sinks) == 1
    rows = sinks[0].result.rows
    assert len(rows) == 1
    (cell,) = rows[0]
    assert isinstance(cell, float | int)
    return float(cell)


@pytest.mark.exec
def test_a_bare_select_reaches_embed_text() -> None:
    """The regression this file exists for: a bare SELECT (no COPY) is
    `compile_table_sql`'s path, and it used to reach a value-returning wasm
    function without ever describing the module that runs it."""
    assert _score("vector_length(embed_text('a cat sat on the mat'))") == 8.0


@pytest.mark.exec
def test_the_same_argument_embeds_to_the_same_vector() -> None:
    """Two calls, same literal argument: memoized to the same vector, so
    the angle between them is nothing at all."""
    assert _score("cos_similarity(embed_text('a cat sat on the mat'), "
                   "embed_text('a cat sat on the mat'))") == pytest.approx(1.0)


@pytest.mark.exec
def test_texts_sharing_letters_score_higher_than_ones_that_share_none() -> None:
    """`cat` and `bat` share every letter but one; `cat` and `xyz` share
    none. The fake embedder is a letter-count, so cosine ranks the first
    pair closer."""
    shared = _score("cos_similarity(embed_text('cat'), embed_text('bat'))")
    disjoint = _score("cos_similarity(embed_text('cat'), embed_text('xyz'))")
    assert shared > disjoint


@pytest.mark.exec
def test_text_with_no_letters_still_embeds_to_eight_dimensions() -> None:
    """Digits and punctuation alone hit no letter bucket, so the module
    hands back the zero vector (see `fauxlate`'s own unit tests) -- but
    `vector_length` reads the vector's dimension, not its magnitude, and
    that stays 8 whether or not anything landed in a bucket."""
    assert _score("vector_length(embed_text('123 !?'))") == 8.0
