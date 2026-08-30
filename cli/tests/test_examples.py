"""Tests for docs/examples.md -- the cookbook.

Data-driven, not one test per recipe: every ```sql / ```pgsql code block in
the cookbook is compiled by running the exact command line the code block
below it shows, and the printed ffmpeg command must match that block byte for
byte.
Adding a recipe to the cookbook therefore adds its test here automatically,
and a recipe whose command drifts fails with the diff in front of it.

The two tiers are the ones the cookbook's own "How to read this file" section
documents:

* ```sql -- OFFLINE. Compiled with ffmpeg made unavailable (``shutil.which``
  returns None, so the filter registry loads empty) and probing stubbed out,
  which is exactly what a machine with no ffmpeg and no readable input file
  sees. Byte-reproducible anywhere, so these run in the default suite.
* ```pgsql -- needs this machine: a tier-2 filter, a named option, a
  generated source, or a bare-array broadcast whose length only the file
  knows. Compiled (never executed) against the real installed ffmpeg in an
  ``exec``-marked test.

The command block is run through :func:`ffrwd.cli.main` rather than the
library API, so the ``$ ffrwd ...`` line a reader would paste is itself
under test -- flags included. ``-f query.sql`` is rewritten to a temp file
holding that recipe's SQL; the path never reaches the printed command.
"""

from __future__ import annotations

import io
import re
import shlex
import subprocess
import sys
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path

import pytest

from ffrwd import binaries, cli
from ffrwd.errors import FfrwdError
from ffrwd.parser import parse, resolve
from ffrwd.registry import Registry

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PROJECT_ROOT.parent
EXAMPLES_PATH = REPO_ROOT / "docs" / "examples.md"

_TIERS = {"sql": "offline", "pgsql": "exec"}

# The language whose recipes need more than ffmpeg on the machine.
_WASM = "wasm"

_BLOCK_RE = re.compile(r"^```(?P<info>[^\n]*)\n(?P<body>.*?)^```$", re.DOTALL | re.MULTILINE)
_HEADING_RE = re.compile(r"^#{1,6} (?P<text>.+)$", re.MULTILINE)


@dataclass(frozen=True)
class Example:
    """One cookbook recipe: its heading, its query, and the code block below it."""

    heading: str
    tier: str
    sql: str
    command: str | None  # the whole ``` block body ("$ ...\n<command>\n"), if any


def _heading_before(text: str, position: int) -> str:
    """The nearest markdown heading above `position` (the recipe's title)."""
    found = "(no heading)"
    for match in _HEADING_RE.finditer(text, 0, position):
        found = match.group("text").strip()
    return found


def _parse(text: str) -> list[Example]:
    """Every ```sql / ```pgsql code block, paired with the block that follows it.

    Pairing is positional and strict: the command block must be the very next
    code block in the file, with no info string. Anything else (a second query
    block, or nothing at all) leaves ``command`` None, which the tests below
    turn into a failure that says what to add.
    """
    blocks = list(_BLOCK_RE.finditer(text))
    examples: list[Example] = []
    for index, block in enumerate(blocks):
        tier = _TIERS.get(block.group("info").strip())
        if tier is None:
            continue
        following = blocks[index + 1] if index + 1 < len(blocks) else None
        command = (
            following.group("body")
            if following is not None and following.group("info").strip() == ""
            else None
        )
        examples.append(
            Example(
                heading=_heading_before(text, block.start()),
                tier=tier,
                sql=block.group("body"),
                command=command,
            )
        )
    return examples


def _ids(examples: list[Example]) -> list[str]:
    """Stable, readable test ids: the heading, slugified, deduplicated."""
    ids: list[str] = []
    for example in examples:
        slug = re.sub(r"[^a-z0-9]+", "-", example.heading.lower()).strip("-")
        suffix = 2
        candidate = slug
        while candidate in ids:
            candidate = f"{slug}-{suffix}"
            suffix += 1
        ids.append(candidate)
    return ids


_EXAMPLES = _parse(EXAMPLES_PATH.read_text(encoding="utf-8"))
_OFFLINE = [e for e in _EXAMPLES if e.tier == "offline"]
_EXEC = [e for e in _EXAMPLES if e.tier == "exec"]

_MISSING_BLOCK_HELP = (
    "every query in docs/examples.md is followed by a code block holding the real "
    "compiled command: a ``` block whose first line is the `$ ffrwd compile "
    "...` invocation and whose remaining lines are exactly what it prints"
)


def _split_command(example: Example) -> tuple[list[str], str]:
    """``(argv, expected stdout)`` from the recipe's command block."""
    assert example.command is not None, f"{example.heading}: {_MISSING_BLOCK_HELP}"
    shown, _, expected = example.command.partition("\n")
    assert shown.startswith("$ ffrwd "), (
        f"{example.heading}: the command block must start with a `$ ffrwd ...` "
        f"line, got {shown!r}"
    )
    assert expected, f"{example.heading}: {_MISSING_BLOCK_HELP}"
    argv = shlex.split(shown[2:])[1:]
    assert "-f" in argv or "--file" in argv, (
        f"{example.heading}: the shown command must read the query with -f, so that the "
        f"```sql block above it is the text being compiled; inline SQL could drift from it"
    )
    return argv, expected


def _run(example: Example, tmp_path: Path) -> str:
    """Run the recipe's own command line; return what it printed."""
    argv, _ = _split_command(example)
    for index, token in enumerate(argv):
        if token in ("-f", "--file"):
            query = tmp_path / argv[index + 1]
            query.write_text(example.sql, encoding="utf-8")
            argv[index + 1] = str(query)
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        code = cli.main(argv)
    assert code == 0, f"{example.heading}: `{shlex.join(argv)}` exited {code}"
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# wrapping: printed `ffmpeg ...` lines are pinned <= 90 columns in the blocks
# ---------------------------------------------------------------------------

_WRAP_WIDTH = 90
_INDENT = "  "
_SPLIT_DELIMS = (";", ",")
# space + trailing backslash: reserved on every packing decision so that a
# line which turns out to need a break, decided one token later, never grows
# past `width` once that " \" lands on it
_BREAK_RESERVE = 2
# The separators `compile` prints between commands: `&&` between two chained
# commands, `|` between two members of one pipeline. Each packs like any other
# token but must NOT be quoted: `'&&'` would be an argument to the first ffmpeg
# rather than the shell operator that runs the second.
_SHELL_OPERATORS = frozenset({"&&", "|"})

# One member of the numbered listing a plan with a named pipe prints, and the
# `N. role: ` that stays put while its argv wraps.
_LISTING_RE = re.compile(r"^(?P<lead>\d+\. (?:ffmpeg|sidecar): )\S")


def wrap_command(line: str, width: int = _WRAP_WIDTH) -> str:
    """Wrap a printed command line to `width` columns with bash `\\`
    continuations. Any other line (a table/CSV row, a `$ ...` line) is
    returned unchanged.

    Two shapes are wrapped: a whole `ffmpeg ...` command, and one numbered
    member of the listing a plan with a named pipe prints (`3. sidecar:
    ffrwd-wasm ...`), whose argv is wrapped and whose `N. role: ` lead stays
    on the first line.

    Tokens pack greedily onto each line. A break between two tokens puts the
    space before the `\\`, so once the shell deletes `\\<newline>` the two
    tokens are still separated by whitespace, and the new token gets the
    usual 2-space indent. A single token still too wide for a line of its
    own -- and shaped like a plain `'...'` quote with no embedded quote --
    moves to its own indented line same as any other token, then splits at
    `;` (falling back to `,`) into adjacent quoted chunks: no space and no
    indent between one chunk's `\\` and the next chunk's open quote, so the
    shell's own adjacent-quote concatenation glues them back into one token.
    A token with no safe split point is left long.
    """
    listed = _LISTING_RE.match(line)
    if listed is None and not line.startswith("ffmpeg "):
        return line
    lead = listed.group("lead") if listed is not None else ""
    tokens = [_quote(token) for token in shlex.split(line[len(lead):])]
    lines: list[str] = []
    current = f"{lead}{tokens[0]}"
    for token in tokens[1:]:
        if len(current) + 1 + len(token) <= width - _BREAK_RESERVE:
            current = f"{current} {token}"
            continue
        if len(_INDENT) + len(token) <= width - _BREAK_RESERVE:
            lines.append(f"{current} \\")
            current = f"{_INDENT}{token}"
            continue
        # reserve room for a possible outer token-boundary break too: the
        # last produced chunk re-enters normal packing and may itself need
        # a trailing " \" once the next token doesn't fit after it
        chunks = (
            _split_quoted_token(token, len(_INDENT), width - _BREAK_RESERVE)
            if _is_simple_quoted(token)
            else [token]
        )
        if len(chunks) > 1:
            lines.append(f"{current} \\")
            lines.append(f"{_INDENT}{chunks[0]}\\")
            lines.extend(f"{chunk}\\" for chunk in chunks[1:-1])
            current = chunks[-1]
            continue
        lines.append(f"{current} \\")
        current = f"{_INDENT}{token}"
    lines.append(current)
    return "\n".join(lines)


def _quote(token: str) -> str:
    """`shlex.quote`, except a shell separator stays bare (see `_SHELL_OPERATORS`)."""
    return token if token in _SHELL_OPERATORS else shlex.quote(token)


def _is_simple_quoted(token: str) -> bool:
    """True for a `'...'` token with no embedded quote to escape."""
    return len(token) >= 2 and token[0] == "'" and token[-1] == "'" and token.count("'") == 2


def _split_quoted_token(token: str, prefix_width: int, width: int) -> list[str]:
    """`token`'s content packed into `'chunk'` pieces that concatenate back
    to it exactly. `prefix_width` is what's already on the line before the
    first chunk starts, so its budget is smaller by that much."""
    content = token[1:-1]
    pieces = _pack(content, width - prefix_width, width)
    return [f"'{piece}'" for piece in pieces]


def _pack(
    content: str,
    first_budget: int,
    rest_budget: int,
    delims: tuple[str, ...] = _SPLIT_DELIMS,
) -> list[str]:
    """`content` split into pieces (concatenating back to it exactly), each
    sized to fit as a quoted, backslash-continued chunk within the given
    per-line budgets. Splits at the first delimiter in `delims` present in
    `content`; a piece that still doesn't fit recurses on the next
    delimiter. No delimiter anywhere: one piece, however long that is."""
    atoms: list[str] | None = None
    remaining_delims: tuple[str, ...] = ()
    for index, delim in enumerate(delims):
        if delim in content:
            parts = content.split(delim)
            atoms = [part + delim for part in parts[:-1]] + [parts[-1]]
            remaining_delims = delims[index + 1 :]
            break
    if atoms is None:
        return [content]

    pieces: list[str] = []
    budget = first_budget
    current = ""
    for atom in atoms:
        candidate = current + atom
        # reserve room for the surrounding quotes (2) and a possible
        # trailing backslash (1)
        if current and len(candidate) + 3 > budget:
            pieces.append(current)
            current = ""
            budget = rest_budget
            candidate = atom
        if len(candidate) + 3 > budget:
            sub = _pack(atom, budget, rest_budget, remaining_delims)
            pieces.extend(sub[:-1])
            current = sub[-1]
            budget = rest_budget
        else:
            current = candidate
    pieces.append(current)
    return pieces


def _shell_tokens(text: str) -> list[str]:
    """`text` tokenized the way a shell would see it: `\\<newline>` is a
    lexical deletion in bash, done before any tokenizing, so it's stripped
    here by hand; shlex.split does the rest, including the adjacent-quote
    concatenation (`'a''b'` -> one token) the chunk split relies on."""
    return shlex.split(text.replace("\\\n", ""))


def _assert_shlex_invariant(actual: str, expected: str) -> None:
    """For a code block that wrapped a single `ffmpeg` line, prove the wrap kept
    the same shell command: the wrapped block text and the original
    unwrapped line must tokenize identically."""
    actual_line = actual.rstrip("\n")
    if "\n" in actual_line or not actual_line.startswith("ffmpeg "):
        return
    assert _shell_tokens(expected) == shlex.split(actual_line)


def _go_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the compile see a machine with no ffmpeg and no readable inputs.

    A fresh, unloadable ``Registry`` rather than the process-wide singleton
    (another test may already have loaded the real one), and ``probe_path``
    stubbed rather than relying on an unreadable path, because the command
    block shows a plain ``ffrwd compile`` -- the flags under test are the
    ones a reader would type, so the isolation has to come from underneath
    them. ``binaries.ffmpeg_path`` is stubbed (not just ``shutil.which``) so
    this stays offline even when the ``static-ffmpeg`` provisioner
    is actually installed and already has a cached binary. Patched by dotted
    path rather than by attribute, since the two modules re-export their
    imports and ``--strict`` will not read through a re-export.
    """
    monkeypatch.setattr("ffrwd.registry.binaries.ffmpeg_path", lambda: None)
    monkeypatch.setattr("ffrwd.compiler.registry_module.load", Registry)
    monkeypatch.setattr("ffrwd.compiler.probe_path", lambda path, args=(), **kw: None)


@pytest.fixture(scope="module")
def _fixtures() -> None:
    """The generated media the exec-tier recipes name verbatim."""
    subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "gen_fixtures.py")],
        check=True,
    )


# ---------------------------------------------------------------------------
# structure
# ---------------------------------------------------------------------------


def test_the_cookbook_has_examples() -> None:
    assert len(_OFFLINE) > 1 and len(_EXEC) > 1


@pytest.mark.parametrize("example", _EXAMPLES, ids=_ids(_EXAMPLES))
def test_every_example_shows_its_compiled_command(example: Example) -> None:
    """Checked for BOTH tiers in the default suite: an exec recipe missing its
    command block would otherwise go unnoticed until someone ran `-m exec`."""
    _split_command(example)


def test_examples_md_uses_lf_newlines_only() -> None:
    assert b"\r" not in EXAMPLES_PATH.read_bytes()


# ---------------------------------------------------------------------------
# the recipes themselves
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("example", _OFFLINE, ids=_ids(_OFFLINE))
def test_offline_example_compiles_to_the_shown_command(
    example: Example, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _go_offline(monkeypatch)
    actual = _run(example, tmp_path)
    expected = _split_command(example)[1]
    assert "\n".join(wrap_command(line) for line in actual.split("\n")) == expected
    _assert_shlex_invariant(actual, expected)


def _require_modules(example: Example) -> None:
    """Skip a recipe naming a wasm module the machine cannot describe.

    The same courtesy the rest of the exec tier extends to a missing ffmpeg:
    describing a module needs the sidecar installed and the module itself
    built, and neither ships with the Python package. Resolve is enough to
    learn which modules a recipe names -- it reads the declarations without
    running anything.
    """
    if f"language {_WASM}" not in example.sql.lower():
        return
    if binaries.ffrwd_wasm_path() is None:
        pytest.skip("ffrwd-wasm not found (uv sync --extra wasm)")
    try:
        declared = resolve(parse(example.sql)).wasm
    except FfrwdError:  # a rejection the compile below will report properly
        return
    for function in declared.values():
        if not (PROJECT_ROOT / function.module).exists():
            pytest.skip(f"module missing: {function.module}")


@pytest.mark.exec
@pytest.mark.parametrize("example", _EXEC, ids=_ids(_EXEC))
def test_exec_example_compiles_to_the_shown_command(
    example: Example, tmp_path: Path, _fixtures: None
) -> None:
    _require_modules(example)
    actual = _run(example, tmp_path)
    expected = _split_command(example)[1]
    assert "\n".join(wrap_command(line) for line in actual.split("\n")) == expected
    _assert_shlex_invariant(actual, expected)


# ---------------------------------------------------------------------------
# wrap_command
# ---------------------------------------------------------------------------


def test_wrap_command_leaves_short_line_untouched() -> None:
    line = "ffmpeg -i film.mkv -map 0:v:0 -c:0 copy film.mp4"
    assert wrap_command(line) == line


def test_wrap_command_leaves_non_ffmpeg_lines_untouched() -> None:
    assert wrap_command("$ ffrwd compile -f query.sql") == "$ ffrwd compile -f query.sql"
    assert wrap_command(" a | b") == " a | b"


def test_wrap_command_passes_an_eval_line_through_unwrapped() -> None:
    """The loudnorm2 chain (recipe 49) is pinned as ONE long line on purpose:
    its quoting is adjacent-quote concatenation around `${...}` references,
    which the token packer would split in the wrong places. Keying on
    `ffmpeg ` is what leaves it alone -- do not "fix" that to match `eval `
    too."""
    line = (
        'eval "$(ffmpeg -i film.mkv -filter_complex '
        "'[0:a:0]loudnorm=I=-16:print_format=json[out0]' -map '[out0]' -f null - "
        '2>&1 | ffrwd loudnorm2env)" && ffmpeg -i film.mkv -filter_complex '
        "'[0:a:0]loudnorm=I=-16:measured_I='\"${FFRWD_LN_I}\"':linear=true[out0]' "
        "-map '[out0]' -c:0 aac out.m4a"
    )
    assert len(line) > _WRAP_WIDTH
    assert wrap_command(line) == line


def test_wrap_command_packs_tokens_at_the_width_boundary() -> None:
    line = "ffmpeg -i in.mp4 -map 0:v:0 -map 0:a:0 -c:0 copy -c:1 copy out.mp4"
    wrapped = wrap_command(line, width=30)
    lines = wrapped.split("\n")
    assert len(lines) > 1
    assert all(len(ln) <= 30 for ln in lines)
    # every non-final line breaks at a token boundary: space then backslash
    for ln in lines[:-1]:
        assert ln.endswith(" \\")
    # continuation lines (all but the first) are indented 2 spaces
    for ln in lines[1:]:
        assert ln.startswith("  ")
    assert shlex.split(wrapped.replace("\\\n", "")) == shlex.split(line)


def test_wrap_command_splits_a_quoted_filtergraph_at_semicolons() -> None:
    line = (
        "ffmpeg -i song.m4a -filter_complex "
        "'[0:a:0]acrossover=split=300 3000[n10][n11][n12];"
        "[n10]acompressor=threshold=0.1:ratio=4[n2];"
        "[n11]acompressor=threshold=0.05:ratio=6[n3];"
        "[n2][n3]amix=inputs=2[n4];"
        "[n12]acompressor=threshold=0.1:ratio=4[n5];"
        "[n4][n5]amix=inputs=2[out0]' "
        "-map '[out0]' mastered.m4a"
    )
    wrapped = wrap_command(line, width=90)
    lines = wrapped.split("\n")
    assert len(lines) > 1
    assert all(len(ln) <= 90 for ln in lines)
    # adjacent quoted chunks: no space, no indent between a chunk and the
    # backslash that continues it, or before the next chunk's open quote
    assert "'\\\n'" in wrapped
    assert shlex.split(wrapped.replace("\\\n", "")) == shlex.split(line)


def test_wrap_command_leaves_a_token_with_no_safe_split_point_long() -> None:
    # brackets force shlex to keep this quoted; no `;` or `,` inside means no
    # safe split point, so the whole quoted token must stay on one line
    token = "'[" + "x" * 200 + "]'"
    line = f"ffmpeg -i in.mp4 -filter_complex {token} out.mp4"
    wrapped = wrap_command(line, width=60)
    assert any(token in ln for ln in wrapped.split("\n"))
    assert shlex.split(wrapped.replace("\\\n", "")) == shlex.split(line)


def test_wrap_command_is_deterministic() -> None:
    line = (
        "ffmpeg -i song.m4a -filter_complex "
        "'[0:a:0]acrossover=split=300:3000[n1][n2];"
        "[n1]acompressor=threshold=0.1:ratio=4[n3]' "
        "-map '[n3]' out.m4a"
    )
    assert wrap_command(line) == wrap_command(line)
    assert wrap_command(line, width=45) == wrap_command(line, width=45)
