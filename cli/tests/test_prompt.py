"""Tests for the LLM system prompt.

The load-bearing one is :func:`test_every_sql_example_compiles`: the prompt is
a contract with a model, so every query it shows off must actually compile.
The marker convention is a ``sql``-tagged code block -- prompt.py only ever puts
complete, accepted queries inside one.

Content-keyed rather than full-text pinned: these assert the prompt SAYS the
load-bearing facts, not that it says them in one exact wording, since prose is
expected to keep improving.

``build_system_prompt`` REQUIRES a ``Registry`` (ffmpeg is always there,
PATH or the provisioner). ``PROMPT`` here is rendered from the committed,
version-pinned reference snapshot
(:func:`ffrwd.registry.load_reference`, no ffmpeg or subprocess
involved), so every assertion is machine-independent.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from ffrwd import cli
from ffrwd.compiler import compile_sql
from ffrwd.errors import ErrorCode, FfrwdError
from ffrwd.inputs import INPUT_OPTIONS
from ffrwd.parser import ROW_SCHEMAS
from ffrwd.prompt import NOT_REPAIRABLE, build_system_prompt
from ffrwd.registry import Registry, load, load_reference
from ffrwd.sink import SINK_OPTIONS

ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT_PATH = ROOT / "tests" / "data" / "reference_registry.json"

_SQL_BLOCK_RE = re.compile(r"```sql\n(.*?)\n```", re.DOTALL)
_SQL_PROBED_BLOCK_RE = re.compile(r"```sql-probed\n(.*?)\n```", re.DOTALL)

_REFERENCE_REGISTRY = load_reference(SNAPSHOT_PATH)
PROMPT = build_system_prompt(_REFERENCE_REGISTRY)


def _sql_examples(text: str) -> list[str]:
    return _SQL_BLOCK_RE.findall(text)


def _probed_sql_examples(text: str) -> list[str]:
    return _SQL_PROBED_BLOCK_RE.findall(text)


# ---------------------------------------------------------------------------
# the examples are real
# ---------------------------------------------------------------------------


def test_prompt_has_examples() -> None:
    assert len(_sql_examples(PROMPT)) >= 8


@pytest.mark.parametrize("query", _sql_examples(PROMPT))
def test_every_sql_example_compiles(query: str) -> None:
    try:
        graph = compile_sql(query)
    except FfrwdError as err:  # pragma: no cover - failure path
        pytest.fail(f"prompt example does not compile: {err}\n{query}")
    assert graph.outputs


def test_examples_cover_the_headline_constructs() -> None:
    joined = "\n".join(_sql_examples(PROMPT))
    for construct in (
        "WITH ",
        "UNION ALL",
        "WHERE ",
        "BETWEEN ",
        "overlay(",
        "ffrwd.blur_regions(",
        "ffrwd.speed(",
    ):
        assert construct in joined, construct


# ---------------------------------------------------------------------------
# the calling convention: three namespaces, the macro trio's exact signatures
# ---------------------------------------------------------------------------


def test_calling_convention_names_the_three_namespaces() -> None:
    assert "### Calling convention" in PROMPT
    assert "ffmpeg.<filter>" in PROMPT
    assert "ffrwd.<name>" in PROMPT


def test_macro_signatures_are_documented_exactly() -> None:
    assert "ffrwd.blur_regions(f, x, y, w, h, sigma)" in PROMPT
    assert "ffrwd.speed(f, factor)" in PROMPT
    assert "ffrwd.delay(f, seconds)" in PROMPT


def test_macros_are_documented_as_positional_only() -> None:
    assert "POSITIONAL ONLY" in PROMPT
    assert "UNSUPPORTED_SQL" in PROMPT


def test_loudnorm2_signature_and_limits_are_documented() -> None:
    """The one macro with named options, and the one whose presence changes
    the shape of the compile -- so the prompt has to carry its limits too."""
    assert "ffrwd.loudnorm2(stream, I => ..., TP => ..., LRA => ...)" in PROMPT
    assert "one `loudnorm2` per query" in PROMPT
    assert "two chained ffmpeg commands" in PROMPT


def test_delay_audio_hint_is_documented() -> None:
    assert "adelay(a.audio[1], 2000)" in PROMPT
    assert "VIDEO ONLY" in PROMPT


def test_n_input_functions_are_documented() -> None:
    assert "amix" in PROMPT and "hstack" in PROMPT and "vstack" in PROMPT
    assert "ANY NUMBER" in PROMPT


def test_array_returning_trio_is_documented() -> None:
    for name in ("channelsplit", "acrossover", "extractplanes"):
        assert f"ffmpeg.{name}" in PROMPT


def test_ffrwd_and_ffmpeg_are_documented_as_reserved() -> None:
    assert "`ffmpeg` is a reserved name" in PROMPT
    assert "`ffrwd` is a reserved name" in PROMPT


# ---------------------------------------------------------------------------
# generated from the real surface
# ---------------------------------------------------------------------------


def test_every_error_code_has_repair_guidance() -> None:
    """Every code a COMPILE can raise carries guidance; the handful that only
    refuse a `run` flag are excluded by name, so neither list can drift."""
    for code in ErrorCode:
        if code in NOT_REPAIRABLE:
            assert f"`{code.value}`" not in PROMPT, code.value
        else:
            assert f"`{code.value}`" in PROMPT, code.value


def test_every_sink_option_is_documented() -> None:
    """The Output section's table is rendered from SINK_OPTIONS --
    a new option cannot silently go undocumented in the prompt."""
    for spec in SINK_OPTIONS.values():
        assert f"`{spec.name}`" in PROMPT, spec.name
        assert spec.doc in PROMPT, spec.name


def test_every_input_option_is_documented() -> None:
    """The Dialect > Sources input-options bullets are rendered from
    INPUT_OPTIONS -- a new option cannot silently go undocumented."""
    for spec in INPUT_OPTIONS.values():
        assert f"`{spec.name}`" in PROMPT, spec.name
        assert spec.doc in PROMPT, spec.name


def test_generated_sources_are_documented() -> None:
    """The FROM-position half of the `ffmpeg.` namespace: its column surface,
    and the silent-audio pattern it exists for."""
    assert "FROM ffmpeg.<source>(<name> => <value>, ...) alias" in PROMPT
    assert "ffmpeg.anullsrc(duration => 1) s" in PROMPT
    assert "duration => <seconds>" in PROMPT
    # A source is NOT a column function; the prompt has to say where it goes.
    assert "belongs in `FROM`" in PROMPT


# ---------------------------------------------------------------------------
# track rows: unnest, row columns, WHERE/ORDER BY, joins,
# COALESCE fills
# ---------------------------------------------------------------------------


def test_track_rows_section_present() -> None:
    assert "### Track rows" in PROMPT
    assert "unnest(" in PROMPT


def test_track_row_columns_match_row_schemas() -> None:
    """The column lists are rendered from the real schema, not guessed --
    every column ROW_SCHEMAS actually exposes must be named in the prompt."""
    for schema in ROW_SCHEMAS.values():
        for column in schema:
            assert f"`{column}`" in PROMPT, column


def test_track_row_where_and_order_by_documented() -> None:
    assert "BETWEEN" in PROMPT
    assert "IS [NOT] NULL" in PROMPT
    assert "ORDER BY" in PROMPT
    assert "NULLS FIRST" in PROMPT


def test_track_row_joins_documented() -> None:
    joined = PROMPT
    for construct in ("INNER JOIN", "LEFT [OUTER] JOIN", "FULL OUTER JOIN"):
        assert construct in joined, construct
    assert "RIGHT [OUTER] JOIN" in joined
    assert "row order is the LEFT side's track order" in joined


def test_track_row_fills_documented() -> None:
    assert "ffmpeg.anullsrc(...)" in PROMPT
    assert "ffmpeg.color(...)" in PROMPT
    assert "ffrwd.empty_captions()" in PROMPT
    assert "COALESCE(<alias>, <fill>)" in PROMPT
    assert "Nothing generates a `data` track" in PROMPT


def test_track_row_examples_are_present_but_not_compiled() -> None:
    """unnest needs a probeable input to size its rows (same rule as a bare
    array), so the worked track-row examples cannot be proven to compile
    without a real file -- they are ```sql-probed, same convention as the
    bare-array broadcast examples, and therefore NOT run by
    test_every_sql_example_compiles above."""
    probed = "\n".join(_probed_sql_examples(PROMPT))
    assert "unnest(" in probed
    assert "FULL OUTER JOIN" in probed
    assert "COALESCE(" in probed
    assert "unnest(" not in "\n".join(_sql_examples(PROMPT))


def test_prompt_is_deterministic() -> None:
    """Pure given the same registry -- no clock, no environment of its own."""
    assert build_system_prompt(_REFERENCE_REGISTRY) == build_system_prompt(_REFERENCE_REGISTRY)


def test_prompt_is_ascii_and_has_no_trailing_newline() -> None:
    PROMPT.encode("ascii")
    assert PROMPT == PROMPT.rstrip("\n")
    assert "\r" not in PROMPT


def test_prompt_states_the_output_rule() -> None:
    assert "Output only the query text." in PROMPT
    assert "validate --json" in PROMPT


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_prompt_prints_it(capsys: pytest.CaptureFixture[str]) -> None:
    code = cli.main(["prompt"])
    captured = capsys.readouterr()
    assert code == 0
    assert captured.err == ""
    # The CLI always loads the live registry now (no --dynamic flag left),
    # so its printed prompt need not equal the no-registry PROMPT constant --
    # only the note-vs-filter-list part of the Functions section can differ.
    assert captured.out.startswith(PROMPT.split("## Functions", 1)[0])


def test_cli_prompt_takes_no_query_argument() -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["prompt", "query.sql"])
    assert exc_info.value.code == 2


def test_cli_prompt_dynamic_flag_is_gone() -> None:
    """The base/--dynamic split collapsed into one prompt; the
    flag is deleted outright (pre-1.0), not kept-but-ignored."""
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["prompt", "--dynamic"])
    assert exc_info.value.code == 2


# ---------------------------------------------------------------------------
# the Functions section: rendered from a real Registry, or a fallback note
# for a broken one (guardrail #7 -- not a normal mode)
# ---------------------------------------------------------------------------


def test_functions_section_is_rendered_from_the_reference_registry() -> None:
    """PROMPT always has a real filter list now
    -- the "no registry" default is not a documented mode."""
    assert "## Functions" in PROMPT
    assert "gblur(video) -> video" in PROMPT
    assert "-> video`" in PROMPT
    assert "-> audio`" in PROMPT


def test_functions_section_with_an_unavailable_registry_is_a_provisioner_note(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    empty = Registry()
    monkeypatch.setattr(empty, "available", lambda: False)
    text = build_system_prompt(empty)
    assert "## Functions" in text
    assert "provisioner failed" in text
    # No per-machine filter list leaked into the fallback note.
    assert "-> video`" not in text
    assert "-> audio`" not in text
    assert text != PROMPT


@pytest.mark.exec
def test_functions_section_with_a_live_registry_lists_gblur() -> None:
    text = build_system_prompt(load())
    assert "gblur(video) -> video" in text


def test_cli_prompt_loads_the_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unlike the old --dynamic-guarded CLI, `ffrwd prompt` always renders the
    Functions section from the live registry now -- so it always loads one."""
    called = False

    def _tracked() -> Registry:
        nonlocal called
        called = True
        empty = Registry()
        monkeypatch.setattr(empty, "available", lambda: False)
        return empty

    monkeypatch.setattr(cli.registry_module, "load", _tracked)
    code = cli.main(["prompt"])
    assert code == 0
    assert called
