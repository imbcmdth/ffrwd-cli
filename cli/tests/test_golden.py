"""Golden IR tests.

Every `tests/golden/NNN-name.sql` fixture is compiled with `compile_sql` and
checked against a blessed expectation file sitting next to it:

* `NNN-name.ir.json`    -- exact `Graph.to_dict()` for queries that compile.
* `NNN-name.error.json` -- `FfrwdError.to_dict()` for queries that are
  rejected; only the `code` and `line` fields are compared (guardrail #5:
  golden-test the IR/error shape, not message text, which is free to improve
  wording without breaking tests).

Expectation files are generated -- never hand-written -- by
`python -m tests.regen_golden`. If you add or change a fixture, run that,
then review the diff yourself: the regen helper blesses whatever the
compiler currently does, it does not know what the query *should* mean.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import sqlglot

from ffrwd.compiler import compile_sql
from ffrwd.emit import emit
from ffrwd.errors import FfrwdError

GOLDEN_DIR = Path(__file__).resolve().parent / "golden"


def _fixtures() -> list[Path]:
    paths = sorted(GOLDEN_DIR.glob("*.sql"))
    assert paths, f"no .sql fixtures found in {GOLDEN_DIR}"
    return paths


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)  # type: ignore[no-any-return]


_REGEN_HINT = "run `python -m tests.regen_golden`, then review the diff before committing"


@pytest.mark.parametrize("sql_path", _fixtures(), ids=lambda p: p.stem)
def test_golden_fixture(sql_path: Path) -> None:
    stem = sql_path.stem
    ir_path = GOLDEN_DIR / f"{stem}.ir.json"
    error_path = GOLDEN_DIR / f"{stem}.error.json"

    if ir_path.exists() and error_path.exists():
        pytest.fail(
            f"{stem}: both {ir_path.name} and {error_path.name} exist; "
            "a fixture must have exactly one expectation file"
        )
    if not ir_path.exists() and not error_path.exists():
        pytest.fail(
            f"{stem}: no expectation file (.ir.json or .error.json) found -- {_REGEN_HINT}"
        )

    text = sql_path.read_text(encoding="utf-8")

    if ir_path.exists():
        expected = _load_json(ir_path)
        graph = compile_sql(text)
        assert graph.to_dict() == expected, (
            f"{stem}: compiled IR no longer matches {ir_path.name} -- "
            f"if this is intentional, {_REGEN_HINT}"
        )
        return

    expected = _load_json(error_path)
    with pytest.raises(FfrwdError) as excinfo:
        compile_sql(text)
    err = excinfo.value
    assert err.code.value == expected["code"], (
        f"{stem}: expected error code {expected['code']!r}, got {err.code.value!r}"
    )
    assert err.line == expected["line"], (
        f"{stem}: expected error on line {expected['line']!r}, got {err.line!r}"
    )


# ---------------------------------------------------------------------------
# smoke test: the README example emits a sane filter_complex
# ---------------------------------------------------------------------------


def test_readme_example_emits_crop_and_overlay() -> None:
    """Not a byte-exact golden (guardrail #5) -- just the load-bearing pieces."""
    text = (GOLDEN_DIR / "010-readme-pip.sql").read_text(encoding="utf-8")
    emitted = emit(compile_sql(text))
    assert "crop=out_w=1200:out_h=50:x=600:y=200" in emitted.filter_complex
    assert "overlay=" in emitted.filter_complex


# ---------------------------------------------------------------------------
# dialect test (guardrail #2): every fixture stays valid Postgres SQL
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sql_path", _fixtures(), ids=lambda p: p.stem)
def test_fixture_parses_as_postgres(sql_path: Path) -> None:
    text = sql_path.read_text(encoding="utf-8")
    sqlglot.parse_one(text, read="postgres")
