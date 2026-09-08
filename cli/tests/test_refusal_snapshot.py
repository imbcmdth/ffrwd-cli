"""A snapshot of what the compiler says to every query this repo pins.

It is a change detector, not a specification: a row records what a refusal IS
today, never that it is right.

Every source of pinned queries feeds it. The ```sql and ```pgsql recipes of
`../docs/examples.md` and `../docs/corpus.md`, parsed by `tests/test_examples.py`'s
own parser -- imported, not reimplemented, so the two nets can never disagree
about what the corpus is -- and every `tests/golden/*.sql` fixture. A query
that compiles contributes its command; a query the compiler refuses
contributes the WHOLE typed error: code, message, hint, line and column, every
field of `FfrwdError.to_dict()`, not the code and line `tests/test_golden.py`
compares. Nothing else pins a refusal's identity, so a moved anchor or a
reworded hint is invisible until this file moves with it.

Tiers follow the corpus, the same split test_examples.py makes:

* A ```sql recipe compiles with ffmpeg made unavailable and probing stubbed,
  and a golden fixture names paths that do not exist, so both are checked in
  the default suite.
* A ```pgsql recipe needs this machine's ffmpeg and the generated fixtures, so
  its rows are checked in an `exec`-marked test.

`../docs/errors.md` is pinned by the page itself and so has no row below:
every ```sql query there is followed by the ```json its rejection prints, and
the check is that the compiler still prints exactly that -- line, col, code,
message and hint. A query on that page that COMPILES is a failure; the page
is a list of refusals, and one that stopped refusing documents nothing. Its
tiers split on the query rather than on the block's language: one naming
`tests/fixtures/` media or a wasm module is the exec tier's, and every other
one compiles against the committed registry snapshot with probing stubbed,
so the default suite reads no media and runs no binary.

The file is `tests/data/refusal_snapshot.json`: one row per query, keyed by the
repo-relative source and the recipe's heading (or the fixture's name), sorted,
LF-only, with machine-specific path prefixes stripped, so two runs over the
same corpus on the same machine produce the same bytes.

Regenerate by running this module with `FFRWD_WRITE_REFUSAL_SNAPSHOT` set, once
per tier -- each run rewrites only the rows it can compute::

    FFRWD_WRITE_REFUSAL_SNAPSHOT=1 pytest tests/test_refusal_snapshot.py
    FFRWD_WRITE_REFUSAL_SNAPSHOT=1 pytest tests/test_refusal_snapshot.py -m exec

Then read the diff: this writes whatever the compiler currently does, and
cannot tell a better refusal from a broken one.
"""

from __future__ import annotations

import io
import json
import os
import shlex
import subprocess
import sys
import warnings
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from ffrwd import build_ffmpeg_commands, cli, compile_sql, emit
from ffrwd.errors import FfrwdError

from . import test_examples as examples

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PROJECT_ROOT.parent
SNAPSHOT_PATH = PROJECT_ROOT / "tests" / "data" / "refusal_snapshot.json"
GOLDEN_DIR = PROJECT_ROOT / "tests" / "golden"
ERRORS_PATH = REPO_ROOT / "docs" / "errors.md"

OFFLINE = "offline"
EXEC = "exec"

# Set to regenerate rather than compare; see the module docstring.
_WRITE_ENV = "FFRWD_WRITE_REFUSAL_SNAPSHOT"

# A golden fixture with no COPY destination has no command line of its own.
# It is rendered to one fixed name so the row records something to diff.
_NO_SINK_DESTINATION = "out.mp4"

_REGEN_HINT = (
    f"set {_WRITE_ENV} and rerun this file (once per tier), then review the diff "
    f"before committing"
)

Entry = dict[str, object]


# ---------------------------------------------------------------------------
# the corpus
# ---------------------------------------------------------------------------


def _repo_path(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def _doc_corpus() -> list[tuple[str, examples.Example]]:
    """`(key, example)` for every recipe the two docs pin, in test_examples' order."""
    pairs = [
        (path, example)
        for path in examples.DOC_PATHS
        for example in examples._parse(path.read_text(encoding="utf-8"))
    ]
    parsed = [example for _, example in pairs]
    # Same parser, same pooling, same order -- so the ids below are the ones
    # test_examples names its own cases with.
    assert parsed == examples._EXAMPLES
    ids = examples._ids(parsed)
    return [
        (f"{_repo_path(path)}::{name}", example)
        for (path, example), name in zip(pairs, ids)
    ]


def _golden_corpus() -> list[tuple[str, Path]]:
    """`(key, path)` for every golden fixture."""
    return [
        (_repo_path(path.with_suffix("")), path) for path in sorted(GOLDEN_DIR.glob("*.sql"))
    ]


def _corpus_keys() -> set[str]:
    return {key for key, _ in _doc_corpus()} | {key for key, _ in _golden_corpus()}


def _needs_this_machine(sql: str) -> bool:
    """Whether `sql` names generated media or a wasm module.

    The unit tier reads neither, so this is the whole of the errors.md split.
    """
    return "tests/fixtures/" in sql or f"language {examples._WASM}" in sql.lower()


def _documented_refusals() -> list[examples.Example]:
    """Every query docs/errors.md pins a refusal for, carrying that pin.

    Read with test_examples' own block regex and heading lookup, so the two
    pages are cut into blocks the same way. What differs is the pairing -- a
    query there is followed by the ```json of its rejection, not by a command
    block, and that JSON becomes the example's `command` -- and the tier,
    which the query text decides rather than the block's language.
    """
    text = ERRORS_PATH.read_text(encoding="utf-8")
    blocks = list(examples._BLOCK_RE.finditer(text))
    documented: list[examples.Example] = []
    for index, block in enumerate(blocks):
        if block.group("info").strip() != "sql":
            continue
        following = blocks[index + 1] if index + 1 < len(blocks) else None
        pinned = (
            following.group("body")
            if following is not None and following.group("info").strip() == "json"
            else None
        )
        sql = block.group("body")
        documented.append(
            examples.Example(
                heading=examples._heading_before(text, block.start()),
                tier=EXEC if _needs_this_machine(sql) else OFFLINE,
                sql=sql,
                command=pinned,
            )
        )
    return documented


_DOCUMENTED = _documented_refusals()
_DOCUMENTED_IDS = examples._ids(_DOCUMENTED)


def _documented_of(tier: str) -> tuple[list[examples.Example], list[str]]:
    """One tier's queries, and their ids -- named once over the whole page, so
    a query answers to the same id in every check below."""
    chosen = [
        (example, name)
        for example, name in zip(_DOCUMENTED, _DOCUMENTED_IDS)
        if example.tier == tier
    ]
    return [example for example, _ in chosen], [name for _, name in chosen]


_DOCUMENTED_OFFLINE, _OFFLINE_IDS = _documented_of(OFFLINE)
_DOCUMENTED_EXEC, _EXEC_IDS = _documented_of(EXEC)

_MISSING_PIN_HELP = (
    "every query in docs/errors.md is followed by the ```json block its "
    "rejection prints -- the whole object, as `ffrwd validate --json` emits it"
)

# The command the page documents as the structured form of a rejection.
_VALIDATE = ["validate", "--json", "-f", "query.sql"]


# ---------------------------------------------------------------------------
# rows
# ---------------------------------------------------------------------------


def _scrub(text: str, temp_dir: Path | None = None) -> str:
    """Machine-specific prefixes out: the repo root, and the directory a query
    file was written to."""
    roots = [REPO_ROOT] if temp_dir is None else [temp_dir.resolve(), REPO_ROOT]
    for root in roots:
        for spelling in (str(root), root.as_posix()):
            for prefix in (spelling + os.sep, spelling + "/", spelling):
                text = text.replace(prefix, "")
    return text


def _accepted(tier: str, command: list[str], temp_dir: Path | None = None) -> Entry:
    return {
        "tier": tier,
        "verdict": "accepted",
        "command": [_scrub(line, temp_dir) for line in command],
    }


def _refused(tier: str, err: FfrwdError, temp_dir: Path | None = None) -> Entry:
    return {
        "tier": tier,
        "verdict": "refused",
        "error": {
            field: _scrub(value, temp_dir) if isinstance(value, str) else value
            for field, value in err.to_dict().items()
        },
    }


def _run(
    argv: list[str], sql: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[int, str, list[FfrwdError]]:
    """Run `argv` over `sql`: its exit code, what it printed, what it refused with.

    `-f <name>` is rewritten to a file under `tmp_path` holding `sql`, so the
    query under test is the one the page shows.
    """
    argv = list(argv)
    for index, token in enumerate(argv):
        if token in ("-f", "--file"):
            query = tmp_path / argv[index + 1]
            query.write_text(sql, encoding="utf-8")
            argv[index + 1] = str(query)

    refusals: list[FfrwdError] = []

    def _record(err: FfrwdError, **_: object) -> None:
        refusals.append(err)

    printed = io.StringIO()
    with monkeypatch.context() as patched:
        # The one place the CLI renders a rejection, so the typed error is
        # taken whole instead of read back out of a printed line.
        patched.setattr(cli, "_print_error", _record)
        with redirect_stdout(printed):
            code = cli.main(argv)
    return code, printed.getvalue(), refusals


def _recipe_row(
    example: examples.Example, tier: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Entry:
    """Run a recipe's own command line; record what it printed, or what it refused."""
    argv, _ = examples._split_command(example)
    code, printed, refusals = _run(argv, example.sql, tmp_path, monkeypatch)
    if code == 0:
        return _accepted(tier, printed.rstrip("\n").split("\n"), tmp_path)
    assert refusals, (
        f"{example.heading}: `{shlex.join(argv)}` exited {code} without a typed error; "
        f"a row is a command or an FfrwdError, and this is neither"
    )
    return _refused(tier, refusals[0], tmp_path)


def _golden_row(path: Path) -> Entry:
    """Compile a golden fixture the way tests/test_golden.py does, and render it."""
    try:
        graph = compile_sql(path.read_text(encoding="utf-8"))
    except FfrwdError as err:
        return _refused(OFFLINE, err)
    emitted = emit(graph)
    sinkless = len(emitted.groups) == 1 and emitted.groups[0].path is None
    commands = build_ffmpeg_commands(emitted, _NO_SINK_DESTINATION if sinkless else None)
    return _accepted(OFFLINE, [shlex.join(argv) for argv in commands])


# ---------------------------------------------------------------------------
# building each tier
# ---------------------------------------------------------------------------


def _build_offline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Entry]:
    """Every row the default suite can compute: the ```sql recipes, then the fixtures.

    The recipes compile inside `_go_offline`'s context and the fixtures
    outside it -- a fixture resolves against the captured registry snapshot,
    as it does in tests/test_golden.py, and an offline recipe against no
    registry at all.
    """
    built: dict[str, Entry] = {}
    with monkeypatch.context() as offline:
        examples._go_offline(offline)
        for key, example in _doc_corpus():
            if example.tier == OFFLINE:
                built[key] = _recipe_row(example, OFFLINE, tmp_path, offline)
    for key, path in _golden_corpus():
        built[key] = _golden_row(path)
    return built


def _build_exec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[str, Entry], set[str]]:
    """The ```pgsql rows, and the keys this machine could not compute."""
    built: dict[str, Entry] = {}
    unreachable: set[str] = set()
    for key, example in _doc_corpus():
        if example.tier != EXEC:
            continue
        if examples.missing_module(example) is not None:
            unreachable.add(key)
            continue
        built[key] = _recipe_row(example, EXEC, tmp_path, monkeypatch)
    return built, unreachable


# ---------------------------------------------------------------------------
# the file
# ---------------------------------------------------------------------------


def _load() -> dict[str, Entry]:
    assert SNAPSHOT_PATH.exists(), f"{SNAPSHOT_PATH.name} is missing -- {_REGEN_HINT}"
    stored: dict[str, Entry] = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    return stored


def _rows_of(stored: dict[str, Entry], tier: str) -> dict[str, Entry]:
    return {key: entry for key, entry in stored.items() if entry["tier"] == tier}


def _write(built: dict[str, Entry], tier: str, unreachable: set[str]) -> None:
    """Replace this tier's rows with `built`, keeping the ones it could not compute."""
    stored = _load() if SNAPSHOT_PATH.exists() else {}
    keys = _corpus_keys()
    kept = {
        key: entry
        for key, entry in stored.items()
        if key in keys and (entry["tier"] != tier or key in unreachable)
    }
    text = json.dumps({**kept, **built}, indent=2, sort_keys=True) + "\n"
    SNAPSHOT_PATH.write_text(text, encoding="utf-8", newline="\n")


def _writing() -> bool:
    return bool(os.environ.get(_WRITE_ENV))


@pytest.fixture(scope="module")
def _fixtures() -> None:
    """The generated media the exec-tier recipes name verbatim."""
    subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "gen_fixtures.py")],
        check=True,
    )


# ---------------------------------------------------------------------------
# the checks
# ---------------------------------------------------------------------------


def test_the_snapshot_covers_every_pinned_query() -> None:
    """Coverage is checked on a bare machine, so a recipe added without a row
    fails in the default suite rather than only where ffmpeg is installed."""
    assert set(_load()) == _corpus_keys(), (
        f"the snapshot and the pinned corpus name different queries -- {_REGEN_HINT}"
    )


def test_the_snapshot_is_portable() -> None:
    data = SNAPSHOT_PATH.read_bytes()
    assert b"\r" not in data
    text = data.decode("utf-8")
    for spelling in (str(REPO_ROOT), REPO_ROOT.as_posix()):
        assert spelling not in text


def test_offline_queries_still_say_what_the_snapshot_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    built = _build_offline(tmp_path, monkeypatch)
    if _writing():
        _write(built, OFFLINE, unreachable=set())
    assert _rows_of(_load(), OFFLINE) == built, (
        f"a pinned query's command or refusal moved -- if that is the intent, {_REGEN_HINT}"
    )


@pytest.mark.exec
def test_exec_queries_still_say_what_the_snapshot_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _fixtures: None
) -> None:
    built, unreachable = _build_exec(tmp_path, monkeypatch)
    if _writing():
        _write(built, EXEC, unreachable)
    if unreachable:
        warnings.warn(
            f"{len(unreachable)} exec rows not checked, their wasm modules "
            f"missing: {', '.join(sorted(unreachable))}",
            stacklevel=1,
        )
    stored = _rows_of(_load(), EXEC)
    assert {key: entry for key, entry in stored.items() if key not in unreachable} == built, (
        f"a pinned query's command or refusal moved -- if that is the intent, {_REGEN_HINT}"
    )


# ---------------------------------------------------------------------------
# docs/errors.md: the page carries its own pins
# ---------------------------------------------------------------------------


def _assert_refuses_as_documented(
    example: examples.Example, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Compile the query; compare its rejection to the ```json below it."""
    assert example.command is not None, f"{example.heading}: {_MISSING_PIN_HELP}"
    # `--json` prints the object rather than rendering it, so the comparison is
    # against stdout: the bytes the page says it captured.
    code, printed, _ = _run(_VALIDATE, example.sql, tmp_path, monkeypatch)
    assert code != 0, (
        f"{example.heading}: this query compiled. Every query on the error page "
        f"is one the compiler refuses; one that stopped refusing documents nothing"
    )
    assert printed.strip(), (
        f"{example.heading}: exited {code} printing no error object; "
        f"`ffrwd validate --json` answers a rejection with one"
    )
    assert json.loads(printed) == json.loads(example.command), (
        f"{example.heading}: the rejection and the ```json below the query "
        f"disagree. The page pins line, col, code, message and hint; regenerate "
        f"it by running the compiler over the query above it, never by editing "
        f"the field that moved"
    )


@pytest.mark.parametrize("example", _DOCUMENTED, ids=_DOCUMENTED_IDS)
def test_every_documented_query_shows_its_error_json(example: examples.Example) -> None:
    """Checked for BOTH tiers in the default suite: an exec-tier query missing
    its pin would otherwise go unnoticed until someone ran `-m exec`."""
    assert example.command is not None, f"{example.heading}: {_MISSING_PIN_HELP}"


@pytest.mark.parametrize("example", _DOCUMENTED_OFFLINE, ids=_OFFLINE_IDS)
def test_documented_query_refuses_as_the_page_pins_it(
    example: examples.Example, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Filters resolve against the committed registry snapshot, as everything
    # in this tier does (tests/conftest.py); probing is stubbed on top of that,
    # so a query naming 'x.mp4' or a socket reads nothing and reaches no host.
    monkeypatch.setattr("ffrwd.compiler.probe_path", lambda path, args=(), **kw: None)
    _assert_refuses_as_documented(example, tmp_path, monkeypatch)


@pytest.mark.exec
@pytest.mark.parametrize("example", _DOCUMENTED_EXEC, ids=_EXEC_IDS)
def test_documented_query_over_real_media_refuses_as_the_page_pins_it(
    example: examples.Example,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    _fixtures: None,
) -> None:
    reason = examples.missing_module(example)
    if reason is not None:
        pytest.skip(reason)
    _assert_refuses_as_documented(example, tmp_path, monkeypatch)
