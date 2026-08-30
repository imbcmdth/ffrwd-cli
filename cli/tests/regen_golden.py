"""Regenerate `tests/golden/*.ir.json` and `*.error.json` from the compiler.

Run as a module from the repo root::

    python -m tests.regen_golden

For every ``tests/golden/NNN-name.sql`` fixture this compiles the query and
writes:

* ``NNN-name.ir.json`` -- ``Graph.to_dict()``, if compilation succeeds, or
* ``NNN-name.error.json`` -- ``FfrwdError.to_dict()``, if it raises.

Only one of the two expectation files is written per fixture; if the
compilation outcome flips (a fixture that used to error now compiles, or vice
versa), the stale expectation file is removed so it cannot be picked up by
:mod:`tests.test_golden` by accident.

IMPORTANT: this is a code generator, not a test oracle. It blesses whatever
the compiler currently does. After running it, review every changed file in
``git diff`` (or, for a fresh checkout, read them) and confirm the IR is
actually what the query should mean -- the regen helper cannot tell a
correct graph from a compiler bug.
"""

from __future__ import annotations

import json
from pathlib import Path

from ffrwd.compiler import compile_sql
from ffrwd.errors import FfrwdError

GOLDEN_DIR = Path(__file__).resolve().parent / "golden"


def _write_json(path: Path, payload: dict[str, object]) -> None:
    text = json.dumps(payload, indent=2, sort_keys=False) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")


def regen() -> None:
    fixtures = sorted(GOLDEN_DIR.glob("*.sql"))
    if not fixtures:
        print(f"no .sql fixtures found in {GOLDEN_DIR}")
        return

    for sql_path in fixtures:
        stem = sql_path.stem
        ir_path = GOLDEN_DIR / f"{stem}.ir.json"
        error_path = GOLDEN_DIR / f"{stem}.error.json"
        text = sql_path.read_text(encoding="utf-8")

        try:
            graph = compile_sql(text)
        except FfrwdError as err:
            _write_json(error_path, err.to_dict())
            if ir_path.exists():
                ir_path.unlink()
            print(f"wrote {error_path.name} (code={err.code.value}, line={err.line})")
            continue

        _write_json(ir_path, graph.to_dict())
        if error_path.exists():
            error_path.unlink()
        print(f"wrote {ir_path.name}")


if __name__ == "__main__":
    regen()
