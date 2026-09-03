"""Tests for user-defined SQL functions: definitions, calls, expansion.

Everything goes through the real parser: `resolve` is where expansion happens,
so a hand-built AST would test a shape that cannot occur. Paths deliberately
do not exist -- probing degrades to symbolic lowering -- and the two tests
that need probed metadata (a tag read off a track row, an array's length)
hand `lower` a synthetic ``ProbeResult``, exactly as tests/test_lower.py does.

The filter surface is the captured snapshot, so `volume` resolves on a machine
with no ffmpeg.
"""

from __future__ import annotations

import functools
import json
import re
from pathlib import Path

import pytest

from ffrwd.emit import build_ffmpeg_args, emit
from ffrwd.errors import ErrorCode, FfrwdError
from ffrwd.lower import lower, lower_table
from ffrwd.parser import Resolved, parse, resolve
from ffrwd.probe import ProbeResult, StreamMeta
from ffrwd.project import discover
from ffrwd.registry import Registry, load_reference
from ffrwd.split import insert_splits
from ffrwd.wasm import Described, SourceCatalog, SourceRendition
from ffrwd.wasm import SourceTrack as WasmSourceTrack

SNAPSHOT_PATH = Path(__file__).resolve().parent / "data" / "reference_registry.json"


@functools.cache
def _snapshot_registry() -> Registry:
    return load_reference(SNAPSHOT_PATH)


def _audio_probe(*tags: dict[str, str], channels: int | None = None) -> ProbeResult:
    """One audio stream per `tags` entry, in file order."""
    return ProbeResult(
        streams=[
            StreamMeta(
                type="audio",
                index=index,
                metadata=dict(entry),
                width=None,
                height=None,
                fps=None,
                sample_rate=44100,
                codec="aac",
                channels=channels,
                channel_layout=None,
                bitrate=None,
                duration=None,
                color_transfer=None,
            )
            for index, entry in enumerate(tags)
        ]
    )


def _resolved(sql: str) -> Resolved:
    return resolve(parse(sql))


def _argv(sql: str, probes: dict[str, ProbeResult | None] | None = None) -> list[str]:
    graph = lower(_resolved(sql), probes or {}, registry=_snapshot_registry())
    return build_ffmpeg_args(emit(insert_splits(graph)))


def _rows(sql: str, probes: dict[str, ProbeResult | None] | None = None) -> list[list[object]]:
    sinks = lower_table(_resolved(sql), probes or {}, registry=_snapshot_registry())
    return sinks[0].result.rows


def _rejects(sql: str, code: ErrorCode, needle: str) -> FfrwdError:
    """Compile `sql` far enough to fail, and pin the code and the wording."""
    with pytest.raises(FfrwdError) as caught:
        graph = lower(_resolved(sql), {}, registry=_snapshot_registry())
        insert_splits(graph)
    error = caught.value
    assert error.code is code, f"{error.code} != {code}: {error}"
    assert needle in error.message, error.message
    return error


# The plan's two target functions, verbatim except for the `IN` the value
# grammar does not have yet (see test_a_body_rejection_lands_on_the_call_site).
NORMALIZE_LANG = (
    "CREATE FUNCTION normalize_lang(raw text) RETURNS text AS $$\n"
    "  SELECT CASE WHEN raw = 'en' OR raw = 'english' THEN 'eng' ELSE raw END\n"
    "$$ LANGUAGE sql;\n"
)
QUIETER = (
    "CREATE FUNCTION quieter(track audio_stream, factor number) RETURNS audio_stream AS $$\n"
    "  SELECT volume(track, factor)\n"
    "$$ LANGUAGE sql;\n"
)


# ---------------------------------------------------------------------------
# the target queries
# ---------------------------------------------------------------------------


def test_a_scalar_function_binds_its_argument_per_row() -> None:
    """The language normalizer, read as data: one expansion, two rows."""
    sql = NORMALIZE_LANG + (
        "SELECT t.index, normalize_lang(t.tags.language) AS language\n"
        "FROM input('a.mka') f, unnest(f.audio) t"
    )
    probes = {"f": _audio_probe({"language": "en"}, {"language": "de"})}
    assert _rows(sql, probes) == [[1, "eng"], [2, "de"]]


def test_a_scalar_function_writes_the_tag_it_computes() -> None:
    sql = NORMALIZE_LANG + (
        "COPY (SELECT t, STRUCT(normalize_lang(t.tags.language) AS language) AS tags\n"
        "      FROM input('a.mka') f, unnest(f.audio) t)\n"
        "TO 'out.mka'"
    )
    probes = {"f": _audio_probe({"language": "english"})}
    assert _argv(sql, probes) == [
        "ffmpeg", "-i", "a.mka",
        "-map", "0:a:0", "-c:0", "copy", "-metadata:s:0", "language=eng",
        "out.mka",
    ]


def test_a_stream_function_wraps_a_filter() -> None:
    sql = QUIETER + (
        "COPY (SELECT f.video[1], quieter(f.audio[1], 0.5) FROM input('film.mkv') f) "
        "TO 'out.mkv'"
    )
    args = _argv(sql)
    assert "[0:a:0]volume=volume=0.5[out1]" in " ".join(args)
    assert args[-1] == "out.mkv"


# ---------------------------------------------------------------------------
# hygiene
# ---------------------------------------------------------------------------


FIRST_TRACK = (
    "CREATE FUNCTION first_track(path text) RETURNS audio_stream AS $$\n"
    "  SELECT g.audio[1] FROM input(path) g\n"
    "$$ LANGUAGE sql;\n"
)


def test_two_calls_to_one_function_get_their_own_aliases() -> None:
    sql = FIRST_TRACK + (
        "COPY (SELECT first_track('a.mka'), first_track('b.mka')) TO 'out.mka'"
    )
    res = _resolved(sql)
    assert sorted(res.input_paths) == ["a.mka", "b.mka"]
    assert len(res.sources) == 2
    assert all(re.fullmatch(r"first_track_\d+_g", alias) for alias in res.sources), res.sources


def test_a_body_input_is_minted_once_per_call_site() -> None:
    """Input identity is the ALIAS: two calls are two inputs, folded onto one
    -i by the same dedup two hand-written input() items over one path get."""
    sql = FIRST_TRACK + (
        "COPY (SELECT first_track('a.mka'), first_track('a.mka')) TO 'out.mka'"
    )
    assert _resolved(sql).input_paths == ["a.mka", "a.mka"]
    assert _argv(sql) == [
        "ffmpeg", "-i", "a.mka",
        "-map", "0:a:0", "-c:0", "copy",
        "-map", "0:a:0", "-c:1", "copy",
        "out.mka",
    ]


def test_two_body_inputs_over_two_paths_are_two_entries() -> None:
    sql = FIRST_TRACK + (
        "COPY (SELECT first_track('a.mka'), first_track('b.mka')) TO 'out.mka'"
    )
    assert _argv(sql) == [
        "ffmpeg", "-i", "a.mka", "-i", "b.mka",
        "-map", "0:a:0", "-c:0", "copy",
        "-map", "1:a:0", "-c:1", "copy",
        "out.mka",
    ]


def test_two_calls_to_a_scalar_function_stay_apart() -> None:
    sql = NORMALIZE_LANG + (
        "SELECT normalize_lang(t.tags.language) AS a, "
        "normalize_lang(t.tags.title) AS b\n"
        "FROM input('a.mka') f, unnest(f.audio) t"
    )
    probes = {"f": _audio_probe({"language": "english", "title": "en"})}
    assert _rows(sql, probes) == [["eng", "eng"]]


def test_a_body_may_not_read_an_alias_of_the_calling_query() -> None:
    sql = (
        "CREATE FUNCTION peek(x number) RETURNS number AS $$\n"
        "  SELECT f.duration\n"
        "$$ LANGUAGE sql;\n"
        "SELECT peek(1) AS d FROM input('a.mka') f"
    )
    _rejects(sql, ErrorCode.UNSUPPORTED_SQL, "references 'f'")


def test_a_body_alias_may_not_shadow_a_parameter() -> None:
    sql = (
        "CREATE FUNCTION shadow(g text) RETURNS audio_stream AS $$\n"
        "  SELECT g.audio[1] FROM input('a.mka') g\n"
        "$$ LANGUAGE sql;\n"
        "COPY (SELECT shadow('x')) TO 'out.mka'"
    )
    _rejects(sql, ErrorCode.UNSUPPORTED_SQL, "shadows")


# ---------------------------------------------------------------------------
# every position a value is legal in
# ---------------------------------------------------------------------------


def test_a_call_is_a_predicate_in_where() -> None:
    sql = (
        "CREATE FUNCTION wanted(lang text) RETURNS boolean AS $$\n"
        "  SELECT lang = 'eng'\n"
        "$$ LANGUAGE sql;\n"
        "COPY (SELECT t FROM input('a.mka') f, unnest(f.audio) t "
        "WHERE wanted(t.tags.language)) TO 'out.mka'"
    )
    probes = {"f": _audio_probe({"language": "eng"}, {"language": "fra"})}
    assert _argv(sql, probes) == [
        "ffmpeg", "-i", "a.mka",
        "-map", "0:a:0", "-c:0", "copy", "-metadata:s:0", "language=eng",
        "out.mka",
    ]


def test_an_array_return_splats_like_a_bare_array_column() -> None:
    sql = (
        "CREATE FUNCTION every_track(tracks audio_stream[]) RETURNS audio_stream[] AS $$\n"
        "  SELECT tracks\n"
        "$$ LANGUAGE sql;\n"
        "COPY (SELECT every_track(f.audio) FROM input('a.mka') f) TO 'out.mka'"
    )
    probes = {"f": _audio_probe({}, {})}
    assert _argv(sql, probes) == [
        "ffmpeg", "-i", "a.mka",
        "-map", "0:a:0", "-c:0", "copy",
        "-map", "0:a:1", "-c:1", "copy",
        "out.mka",
    ]


def test_a_function_takes_no_parameters_at_all() -> None:
    sql = (
        "CREATE FUNCTION house_style() RETURNS text AS $$\n"
        "  SELECT 'eng'\n"
        "$$ LANGUAGE sql;\n"
        "COPY (SELECT f.audio[1], STRUCT(house_style() AS language) AS tags "
        "FROM input('a.mka') f) "
        "TO 'out.mka'"
    )
    assert "language=eng" in _argv(sql)


def test_a_function_reads_a_command_line_variable() -> None:
    """`-v` substitution is textual and runs before the parse, so an argument
    carrying one needs nothing of its own."""
    from ffrwd.vars import substitute

    sql = substitute(
        FIRST_TRACK + "COPY (SELECT first_track(:'src')) TO 'out.mka'",
        {"src": "a.mka"},
    ).text
    assert _argv(sql) == ["ffmpeg", "-i", "a.mka", "-map", "0:a:0", "-c:0", "copy", "out.mka"]


def test_a_function_calls_another_function() -> None:
    sql = (
        NORMALIZE_LANG
        + "CREATE FUNCTION shout(raw text) RETURNS text AS $$\n"
        "  SELECT normalize_lang(raw) || '!'\n"
        "$$ LANGUAGE sql;\n"
        "SELECT shout(t.tags.language) AS language\n"
        "FROM input('a.mka') f, unnest(f.audio) t"
    )
    probes = {"f": _audio_probe({"language": "english"})}
    assert _rows(sql, probes) == [["eng!"]]


def test_a_call_nests_inside_a_call_to_the_same_function() -> None:
    """An argument is the caller's text, so f(f(x)) is nesting, not recursion."""
    sql = QUIETER + (
        "COPY (SELECT quieter(quieter(f.audio[1], 0.5), 0.5) FROM input('a.mka') f) "
        "TO 'out.mka'"
    )
    assert " ".join(_argv(sql)).count("volume=volume=0.5") == 2


def test_a_view_body_may_call_a_function() -> None:
    sql = QUIETER + (
        "CREATE VIEW soft AS SELECT quieter(f.audio[1], 0.5) FROM input('a.mka') f;\n"
        "COPY (SELECT * FROM soft) TO 'out.mka'"
    )
    assert "volume=volume=0.5" in " ".join(_argv(sql))


# ---------------------------------------------------------------------------
# the signature
# ---------------------------------------------------------------------------


def test_too_many_arguments_is_rejected() -> None:
    sql = NORMALIZE_LANG + "SELECT normalize_lang('en', 'de') AS language"
    error = _rejects(sql, ErrorCode.UDF_ARG_TYPE, "got 2 arguments")
    assert error.hint is not None and "normalize_lang(raw text)" in error.hint


def test_too_few_arguments_is_rejected() -> None:
    sql = QUIETER + "COPY (SELECT quieter(f.audio[1]) FROM input('a.mka') f) TO 'o.mka'"
    _rejects(sql, ErrorCode.UDF_ARG_TYPE, "got 1 argument")


def test_a_number_where_text_is_declared_is_rejected() -> None:
    sql = NORMALIZE_LANG + "SELECT normalize_lang(5) AS language"
    _rejects(sql, ErrorCode.UDF_ARG_TYPE, "'raw' argument")


def test_text_where_a_number_is_declared_is_rejected() -> None:
    sql = QUIETER + (
        "COPY (SELECT quieter(f.audio[1], 'half') FROM input('a.mka') f) TO 'o.mka'"
    )
    _rejects(sql, ErrorCode.UDF_ARG_TYPE, "'factor' argument")


@pytest.mark.parametrize(
    ("argument", "language"),
    [
        ("t.tags.language", "eng"),
        ("CASE WHEN t.index = 1 THEN 'en' ELSE t.tags.language END", "eng"),
        ("t.tags.language || ''", "eng"),
        ("t.index::text", "1"),
    ],
)
def test_a_value_whose_type_the_probe_decides_is_taken_as_written(
    argument: str, language: str
) -> None:
    """Only a shape that says its own type is checked here; the rest is resolve's."""
    sql = NORMALIZE_LANG + (
        f"SELECT normalize_lang({argument}) AS language\n"
        "FROM input('a.mka') f, unnest(f.audio) t"
    )
    probes = {"f": _audio_probe({"language": "en"})}
    assert _rows(sql, probes) == [[language]]


def test_a_body_reads_a_path_off_a_stream_parameter() -> None:
    sql = (
        "CREATE FUNCTION lang_of(track audio_stream) RETURNS text AS $$\n"
        "  SELECT track.tags.language\n"
        "$$ LANGUAGE sql;\n"
        "SELECT lang_of(t) AS language FROM input('a.mka') f, unnest(f.audio) t"
    )
    probes = {"f": _audio_probe({"language": "eng"})}
    assert _rows(sql, probes) == [["eng"]]


def test_a_filter_call_where_text_is_declared_is_rejected() -> None:
    sql = NORMALIZE_LANG + (
        "COPY (SELECT normalize_lang(volume(f.audio[1], 0.5)) AS language, f.audio[1] "
        "FROM input('a.mka') f) TO 'out.mka'"
    )
    _rejects(sql, ErrorCode.UDF_ARG_TYPE, "got a stream")


def test_a_literal_where_a_stream_is_declared_is_rejected() -> None:
    sql = QUIETER + "SELECT quieter('a.mka', 0.5)"
    _rejects(sql, ErrorCode.UDF_ARG_TYPE, "'track' argument")


def test_input_is_not_an_argument() -> None:
    sql = FIRST_TRACK + "COPY (SELECT first_track(input('a.mka'))) TO 'out.mka'"
    error = _rejects(sql, ErrorCode.UDF_ARG_TYPE, "input()")
    assert error.hint is not None and "its own FROM" in error.hint


def test_an_unknown_return_type_is_rejected() -> None:
    sql = (
        "CREATE FUNCTION meta(raw text) RETURNS jsonb AS $$\n"
        "  SELECT raw\n"
        "$$ LANGUAGE sql;\n"
        "SELECT meta('x') AS m"
    )
    error = _rejects(sql, ErrorCode.UNSUPPORTED_SQL, "unknown type 'jsonb'")
    assert error.line == 1
    assert error.hint is not None and "audio_stream" in error.hint


def test_an_unknown_parameter_type_is_rejected() -> None:
    sql = (
        "CREATE FUNCTION meta(raw jsonb) RETURNS text AS $$\n"
        "  SELECT 'x'\n"
        "$$ LANGUAGE sql;\n"
        "SELECT meta('x') AS m"
    )
    _rejects(sql, ErrorCode.UNSUPPORTED_SQL, "unknown type 'jsonb'")


def test_a_map_type_is_not_nameable_in_a_signature() -> None:
    sql = (
        "CREATE FUNCTION m(t tag) RETURNS text AS $$\n"
        "  SELECT 'x'\n"
        "$$ LANGUAGE sql;\n"
        "SELECT m('x') AS m"
    )
    _rejects(sql, ErrorCode.UNSUPPORTED_SQL, "unknown type 'tag'")


def test_a_parameter_needs_a_name() -> None:
    sql = (
        "CREATE FUNCTION m(text) RETURNS text AS $$\n"
        "  SELECT 'x'\n"
        "$$ LANGUAGE sql;\n"
        "SELECT m('x') AS m"
    )
    _rejects(sql, ErrorCode.UNSUPPORTED_SQL, "parameter")


def test_a_parameter_is_a_name_and_a_type() -> None:
    for written in ("OUT a text", "INOUT a text", "VARIADIC a text[]", "a text COLLATE c"):
        sql = (
            f"CREATE FUNCTION m({written}) RETURNS text AS $$ SELECT 'x' $$ LANGUAGE sql;\n"
            "SELECT m('x') AS m"
        )
        _rejects(sql, ErrorCode.UNSUPPORTED_SQL, "not supported")


def test_a_parameter_name_is_declared_once() -> None:
    sql = (
        "CREATE FUNCTION m(a text, a text) RETURNS text AS $$\n"
        "  SELECT a\n"
        "$$ LANGUAGE sql;\n"
        "SELECT m('x', 'y') AS m"
    )
    _rejects(sql, ErrorCode.UNSUPPORTED_SQL, "'a' twice")


# ---------------------------------------------------------------------------
# DEFAULT parameters
# ---------------------------------------------------------------------------


GREET = (
    "CREATE FUNCTION greet(name text, punctuation text DEFAULT '!') RETURNS text AS $$\n"
    "  SELECT name || punctuation\n"
    "$$ LANGUAGE sql;\n"
)


def test_an_omitted_trailing_argument_takes_the_default() -> None:
    sql = GREET + "SELECT greet('hi') AS g FROM input('a.mka') f"
    assert _rows(sql) == [["hi!"]]


def test_a_null_argument_to_a_defaulted_parameter_takes_the_default() -> None:
    """The one deviation from Postgres: NULL is absence throughout the dialect
    -- an unset variable substitutes to it -- so it falls through to the
    DEFAULT the same way omitting the argument does."""
    sql = GREET + "SELECT greet('hi', NULL) AS g FROM input('a.mka') f"
    assert _rows(sql) == [["hi!"]]


def test_null_to_a_parameter_with_no_default_still_passes_through() -> None:
    """Only a DEFAULT changes what NULL means; without one it drops as ever."""
    sql = QUIETER + (
        "COPY (SELECT f.video[1], quieter(f.audio[1], NULL) FROM input('film.mkv') f) "
        "TO 'out.mkv'"
    )
    assert "[0:a:0]volume[out1]" in " ".join(_argv(sql))


def test_too_few_arguments_names_the_parameter_with_no_default() -> None:
    sql = QUIETER + "COPY (SELECT quieter(f.audio[1]) FROM input('a.mka') f) TO 'o.mka'"
    error = _rejects(sql, ErrorCode.UDF_ARG_TYPE, "parameter 'factor' has no DEFAULT")
    assert error.hint is not None and "quieter(track audio_stream, factor number)" in error.hint


def test_a_default_on_a_non_trailing_parameter_works_once_every_later_one_has_one() -> None:
    sql = (
        "CREATE FUNCTION labeled(a text, b text DEFAULT 'y', c text DEFAULT 'z') "
        "RETURNS text AS $$\n"
        "  SELECT a || b || c\n"
        "$$ LANGUAGE sql;\n"
        "SELECT labeled('x') AS g FROM input('a.mka') f"
    )
    assert _rows(sql) == [["xyz"]]


def test_a_parameter_without_a_default_after_one_with_one_is_rejected() -> None:
    sql = (
        "CREATE FUNCTION bad(a text DEFAULT 'x', b text) RETURNS text AS $$\n"
        "  SELECT a || b\n"
        "$$ LANGUAGE sql;\n"
        "SELECT bad('p', 'q') AS g FROM input('a.mka') f"
    )
    _rejects(sql, ErrorCode.UNSUPPORTED_SQL, "no DEFAULT after one that has one")


def test_default_null_makes_the_parameter_omissible() -> None:
    """DEFAULT NULL is the spelling for an optional knob: omitting the
    argument is legal and gives NULL, which drops wherever the body uses it
    as an option. Without the DEFAULT, omission is an arity error, so the
    two spellings are not redundant."""
    sql = (
        "CREATE FUNCTION quieter(track audio_stream, factor number DEFAULT NULL) "
        "RETURNS audio_stream AS $$\n"
        "  SELECT volume(track, factor)\n"
        "$$ LANGUAGE sql;\n"
        "COPY (SELECT f.video[1], quieter(f.audio[1]) FROM input('film.mkv') f) "
        "TO 'out.mkv'"
    )
    assert "[0:a:0]volume[out1]" in " ".join(_argv(sql))


# ---------------------------------------------------------------------------
# the definition statement
# ---------------------------------------------------------------------------


def test_a_value_function_may_not_be_called_in_from() -> None:
    sql = FIRST_TRACK + "COPY (SELECT t FROM first_track('a.mka') t) TO 'out.mka'"
    _rejects(sql, ErrorCode.UNSUPPORTED_SQL, "not a table")


def test_a_body_needs_a_language() -> None:
    sql = (
        "CREATE FUNCTION m(a text) RETURNS text AS $$ SELECT a $$;\n"
        "SELECT m('x') AS m"
    )
    _rejects(sql, ErrorCode.UNSUPPORTED_SQL, "declares no LANGUAGE")


def test_a_non_sql_language_is_rejected() -> None:
    sql = (
        "CREATE FUNCTION m(a text) RETURNS text AS $$ SELECT a $$ LANGUAGE plpgsql;\n"
        "SELECT m('x') AS m"
    )
    _rejects(sql, ErrorCode.UNSUPPORTED_SQL, "plpgsql")


def test_a_function_needs_a_returns_type() -> None:
    sql = (
        "CREATE FUNCTION m(a text) AS $$ SELECT a $$ LANGUAGE sql;\n"
        "SELECT m('x') AS m"
    )
    _rejects(sql, ErrorCode.UNSUPPORTED_SQL, "RETURNS")


def test_a_duplicate_function_name_is_rejected() -> None:
    sql = (
        "CREATE FUNCTION m(a text) RETURNS text AS $$ SELECT a $$ LANGUAGE sql;\n"
        "CREATE FUNCTION m(a number) RETURNS number AS $$ SELECT a $$ LANGUAGE sql;\n"
        "SELECT m('x') AS m"
    )
    error = _rejects(sql, ErrorCode.UNSUPPORTED_SQL, "defined twice")
    assert error.line == 2


def test_a_function_nothing_calls_is_rejected() -> None:
    sql = (
        "CREATE FUNCTION unused(a text) RETURNS text AS $$ SELECT a $$ LANGUAGE sql;\n"
        "COPY (SELECT f.audio[1] FROM input('a.mka') f) TO 'out.mka'"
    )
    error = _rejects(sql, ErrorCode.UNSUPPORTED_SQL, "never called")
    assert error.line == 1


def test_a_bare_definition_compiles_to_nothing() -> None:
    sql = "CREATE FUNCTION unused(a text) RETURNS text AS $$ SELECT a $$ LANGUAGE sql"
    _rejects(sql, ErrorCode.UNSUPPORTED_SQL, "never called")


def test_a_function_is_defined_before_it_is_called() -> None:
    sql = (
        "SELECT later('x') AS m;\n"
        "CREATE FUNCTION later(a text) RETURNS text AS $$ SELECT a $$ LANGUAGE sql"
    )
    _rejects(sql, ErrorCode.UNSUPPORTED_SQL, "before it is defined")


def test_a_definition_may_not_follow_a_copy() -> None:
    sql = (
        "COPY (SELECT f.audio[1] FROM input('a.mka') f) TO 'out.mka';\n"
        "CREATE FUNCTION m(a text) RETURNS text AS $$ SELECT a $$ LANGUAGE sql"
    )
    _rejects(sql, ErrorCode.UNSUPPORTED_SQL, "may not follow a COPY")


def test_create_or_replace_function_is_rejected() -> None:
    sql = (
        "CREATE OR REPLACE FUNCTION m(a text) RETURNS text AS $$ SELECT a $$ "
        "LANGUAGE sql;\nSELECT m('x') AS m"
    )
    _rejects(sql, ErrorCode.UNSUPPORTED_SQL, "OR REPLACE")


def test_a_function_option_is_rejected() -> None:
    sql = (
        "CREATE FUNCTION m(a text) RETURNS text AS $$ SELECT a $$ LANGUAGE sql "
        "IMMUTABLE;\nSELECT m('x') AS m"
    )
    _rejects(sql, ErrorCode.UNSUPPORTED_SQL, "IMMUTABLE")


def test_a_builtin_name_may_not_be_redefined() -> None:
    sql = (
        "CREATE FUNCTION coalesce(a text) RETURNS text AS $$ SELECT a $$ LANGUAGE sql;\n"
        "SELECT coalesce('x') AS m"
    )
    _rejects(sql, ErrorCode.UNSUPPORTED_SQL, "reserved")


def test_input_may_not_be_redefined() -> None:
    sql = (
        "CREATE FUNCTION input(a text) RETURNS text AS $$ SELECT a $$ LANGUAGE sql;\n"
        "SELECT input('x') AS m"
    )
    _rejects(sql, ErrorCode.UNSUPPORTED_SQL, "reserved")


# ---------------------------------------------------------------------------
# the body
# ---------------------------------------------------------------------------


def test_a_body_is_one_select() -> None:
    sql = (
        "CREATE FUNCTION m(a text) RETURNS text AS $$ SELECT a; SELECT a $$ LANGUAGE sql;\n"
        "SELECT m('x') AS m"
    )
    _rejects(sql, ErrorCode.UNSUPPORTED_SQL, "one SELECT")


def test_a_body_selects_one_column() -> None:
    sql = (
        "CREATE FUNCTION m(a text) RETURNS text AS $$ SELECT a, a $$ LANGUAGE sql;\n"
        "SELECT m('x') AS m"
    )
    _rejects(sql, ErrorCode.UNSUPPORTED_SQL, "one column")


def test_a_body_has_no_with_of_its_own() -> None:
    sql = (
        "CREATE FUNCTION m(a text) RETURNS text AS $$\n"
        "  WITH c AS (SELECT a AS x) SELECT c.x FROM c\n"
        "$$ LANGUAGE sql;\n"
        "SELECT m('x') AS m"
    )
    _rejects(sql, ErrorCode.UNSUPPORTED_SQL, "WITH")


def test_a_body_does_not_group() -> None:
    sql = (
        "CREATE FUNCTION m(a text) RETURNS text AS $$\n"
        "  SELECT g.audio[1] FROM input(a) g GROUP BY g.audio[1]\n"
        "$$ LANGUAGE sql;\n"
        "COPY (SELECT m('a.mka')) TO 'out.mka'"
    )
    error = _rejects(sql, ErrorCode.NO_STREAMING_EQUIVALENT, "GROUP BY")
    assert "body of m()" in error.message


def test_a_body_that_does_not_parse_is_anchored_on_its_definition() -> None:
    sql = (
        "CREATE FUNCTION m(a text) RETURNS text AS $$ SELECT FROM WHERE $$ LANGUAGE sql;\n"
        "SELECT m('x') AS m"
    )
    error = _rejects(sql, ErrorCode.PARSE_ERROR, "body of m()")
    assert error.line == 1


# ---------------------------------------------------------------------------
# recursion
# ---------------------------------------------------------------------------


def test_a_function_may_not_call_itself() -> None:
    sql = (
        "CREATE FUNCTION loop(a text) RETURNS text AS $$ SELECT loop(a) $$ LANGUAGE sql;\n"
        "SELECT loop('x') AS m"
    )
    error = _rejects(sql, ErrorCode.UNSUPPORTED_SQL, "recursive")
    assert "loop -> loop" in error.message


def test_two_functions_may_not_call_each_other() -> None:
    sql = (
        "CREATE FUNCTION ping(a text) RETURNS text AS $$ SELECT pong(a) $$ LANGUAGE sql;\n"
        "CREATE FUNCTION pong(a text) RETURNS text AS $$ SELECT ping(a) $$ LANGUAGE sql;\n"
        "SELECT ping('x') AS m"
    )
    error = _rejects(sql, ErrorCode.UNSUPPORTED_SQL, "recursive")
    assert "ping -> pong -> ping" in error.message


# ---------------------------------------------------------------------------
# diagnostics through two layers
# ---------------------------------------------------------------------------


def test_a_body_rejection_lands_on_the_call_site() -> None:
    """A body-only rejection anchors on the CALL, and says where in the body."""
    sql = (
        "CREATE FUNCTION rate(x number) RETURNS text AS $$\n"
        "  SELECT 'a' || 1\n"
        "$$ LANGUAGE sql;\n"
        "COPY (SELECT f.audio[1],\n"
        "             STRUCT(rate(2) AS r) AS tags\n"
        "      FROM input('a.mka') f)\n"
        "TO 'out.mka'"
    )
    error = _rejects(sql, ErrorCode.UNSUPPORTED_SQL, "body of rate()")
    assert error.line == 5, error
    assert "body line 2" in error.message, error.message


def test_a_bad_call_inside_a_body_lands_on_the_outer_call_site() -> None:
    """The inner call is body text too, so its own rejection travels the same way."""
    sql = (
        "CREATE FUNCTION inner_lang(a text) RETURNS text AS $$ SELECT a $$ LANGUAGE sql;\n"
        "CREATE FUNCTION outer_lang(a text) RETURNS text AS $$\n"
        "  SELECT inner_lang('x', 'y') || a\n"
        "$$ LANGUAGE sql;\n"
        "SELECT outer_lang(t.tags.language) AS language\n"
        "FROM input('a.mka') f, unnest(f.audio) t"
    )
    error = _rejects(sql, ErrorCode.UDF_ARG_TYPE, "got 2 arguments")
    assert error.line == 5, error
    assert "body of outer_lang()" in error.message, error.message


def test_a_rejection_after_resolve_still_lands_on_the_call_site() -> None:
    """Body positions are gone by then, so the anchor is all that is left."""
    sql = (
        "CREATE FUNCTION rate(x number) RETURNS number AS $$\n"
        "  SELECT 1 / 0 + x\n"
        "$$ LANGUAGE sql;\n"
        "COPY (SELECT f.audio[1],\n"
        "             STRUCT(rate(2) AS r) AS tags\n"
        "      FROM input('a.mka') f)\n"
        "TO 'out.mka'"
    )
    assert _rejects(sql, ErrorCode.UNSUPPORTED_SQL, "division by zero").line == 5


def test_a_rejection_over_an_argument_lands_on_the_argument() -> None:
    """The argument is the writer's own text, so it outranks the body's."""
    sql = (
        "CREATE FUNCTION lang(raw text) RETURNS text AS $$\n"
        "  SELECT CASE WHEN raw LIKE 'e%' THEN 'eng' ELSE raw END\n"
        "$$ LANGUAGE sql;\n"
        "SELECT lang(t.tags.language) AS language\n"
        "FROM input('a.mka') f, unnest(f.audio) t"
    )
    error = _rejects(sql, ErrorCode.UNSUPPORTED_SQL, "predicate")
    assert error.line == 4, error


def test_no_synthetic_line_survives_a_successful_resolve() -> None:
    """Body positions are rewritten to the call site, so lower cannot report one."""
    sql = QUIETER + (
        "COPY (SELECT quieter(f.video[1], 0.5) FROM input('a.mkv') f) TO 'out.mkv'"
    )
    error = _rejects(sql, ErrorCode.UDF_ARG_TYPE, "volume")
    assert error.line is not None and error.line <= 4, error


# ---------------------------------------------------------------------------
# table-returning functions: the row source
# ---------------------------------------------------------------------------


# The plan's table function, and the one-row shape beside it.
ENG_AUDIO = (
    "CREATE FUNCTION eng_audio(file text) RETURNS TABLE(track audio_stream) AS $$\n"
    "  SELECT a FROM input(file) f, unnest(f.audio) a WHERE a.tags.language = 'eng'\n"
    "$$ LANGUAGE sql;\n"
)
TAGGED_AUDIO = (
    "CREATE FUNCTION tagged_audio(file text, lang text)\n"
    "RETURNS TABLE(track audio_stream, language text) AS $$\n"
    "  SELECT a, lang FROM input(file) f, unnest(f.audio) a\n"
    "$$ LANGUAGE sql;\n"
)
ONE_TRACK = (
    "CREATE FUNCTION one_track(path text) RETURNS TABLE(track audio_stream) AS $$\n"
    "  SELECT g.audio[1] FROM input(path) g\n"
    "$$ LANGUAGE sql;\n"
)


def test_a_table_function_is_a_row_source_in_from() -> None:
    """The plan's target query: two eng tracks gathered beside a host video."""
    sql = ENG_AUDIO + (
        "COPY (SELECT v.video[1], array_agg(t.track)\n"
        "      FROM input('a.mp4') v, eng_audio('b.mp4') AS t\n"
        "      GROUP BY v.video[1])\n"
        "TO 'out.mkv'"
    )
    probes = {"eng_audio_1_f": _audio_probe({"language": "eng"}, {"language": "eng"})}
    assert _argv(sql, probes) == [
        "ffmpeg", "-i", "b.mp4", "-i", "a.mp4",
        "-map", "1:v:0", "-c:0", "copy",
        "-map", "0:a:0", "-c:1", "copy", "-metadata:s:1", "language=eng",
        "-map", "0:a:1", "-c:2", "copy", "-metadata:s:2", "language=eng",
        "out.mkv",
    ]


def test_a_one_row_table_function_needs_no_aggregate() -> None:
    sql = ONE_TRACK + "COPY (SELECT t.track FROM one_track('a.mka') AS t) TO 'out.mka'"
    assert _argv(sql) == [
        "ffmpeg", "-i", "a.mka", "-map", "0:a:0", "-c:0", "copy", "out.mka",
    ]


def test_a_multi_row_table_function_yields_one_row_per_body_row() -> None:
    """The call site has the BODY's cardinality: two eng tracks are two rows."""
    sql = ENG_AUDIO + "SELECT t.track FROM eng_audio('b.mp4') AS t"
    probes = {
        "eng_audio_1_f": _audio_probe(
            {"language": "eng"}, {"language": "fra"}, {"language": "eng"}
        )
    }
    assert len(_rows(sql, probes)) == 2


def test_a_table_function_cross_joins_the_host_rows() -> None:
    sql = ENG_AUDIO + (
        "SELECT u.index, t.track\n"
        "FROM input('a.mka') v, unnest(v.audio) u, eng_audio('b.mp4') AS t"
    )
    probes = {
        "v": _audio_probe({}, {}),
        "eng_audio_1_f": _audio_probe({"language": "eng"}, {"language": "eng"}),
    }
    assert len(_rows(sql, probes)) == 4


def test_the_host_where_narrows_the_cross_join() -> None:
    sql = ENG_AUDIO + (
        "SELECT u.index, t.track\n"
        "FROM input('a.mka') v, unnest(v.audio) u, eng_audio('b.mp4') AS t\n"
        "WHERE u.index = 1"
    )
    probes = {
        "v": _audio_probe({}, {}),
        "eng_audio_1_f": _audio_probe({"language": "eng"}, {"language": "eng"}),
    }
    assert [row[0] for row in _rows(sql, probes)] == [1, 1]


def test_a_grouped_call_gathers_the_calls_rows() -> None:
    sql = ENG_AUDIO + (
        "SELECT array_agg(t.track) FROM eng_audio('b.mp4') AS t"
    )
    probes = {"eng_audio_1_f": _audio_probe({"language": "eng"}, {"language": "eng"})}
    rows = _rows(sql, probes)
    assert len(rows) == 1
    assert str(rows[0][0]).count("audio") == 2


def test_an_ungrouped_multi_row_call_into_one_path_is_rejected() -> None:
    sql = ENG_AUDIO + "COPY (SELECT t.track FROM eng_audio('b.mp4') AS t) TO 'out.mka'"
    probes = {"eng_audio_1_f": _audio_probe({"language": "eng"}, {"language": "eng"})}
    with pytest.raises(FfrwdError) as caught:
        _argv(sql, probes)
    assert caught.value.code is ErrorCode.ROW_COUNT_MISMATCH, caught.value


def test_a_table_functions_columns_are_named_by_returns_table() -> None:
    """The alias exposes the declared names, mapped from the projections in
    order. A declared SCALAR column is a value the rows carry, not metadata:
    nothing of it reaches the command line unless the caller reads it."""
    sql = TAGGED_AUDIO + (
        "COPY (SELECT array_agg(t.track) FROM tagged_audio('a.mka', 'eng') AS t)\n"
        "TO 'out.mka'"
    )
    probes = {"tagged_audio_1_f": _audio_probe({}, {})}
    assert _argv(sql, probes) == [
        "ffmpeg", "-i", "a.mka",
        "-map", "0:a:0", "-c:0", "copy",
        "-map", "0:a:1", "-c:1", "copy",
        "out.mka",
    ]


def test_a_declared_value_column_names_the_files_a_fan_out_writes() -> None:
    """What the value column IS for: the caller reads it where values read."""
    sql = TAGGED_AUDIO + (
        "COPY (SELECT array_agg(t.track) FROM tagged_audio('a.mka', 'eng') AS t\n"
        "      GROUP BY t.language)\n"
        "TO (t.language || '.mka')"
    )
    probes = {"tagged_audio_1_f": _audio_probe({}, {})}
    assert _argv(sql, probes)[-1] == "eng.mka"


def test_an_undeclared_column_of_the_alias_is_rejected() -> None:
    sql = ONE_TRACK + "SELECT t.language FROM one_track('a.mka') AS t"
    error = _rejects(sql, ErrorCode.UNSUPPORTED_SQL, "unknown column 't.language'")
    assert error.hint is not None and "track" in error.hint


def test_two_calls_to_one_table_function_mint_two_inputs() -> None:
    sql = ONE_TRACK + (
        "COPY (SELECT x.track, y.track\n"
        "      FROM one_track('a.mka') AS x, one_track('b.mka') AS y)\n"
        "TO 'out.mka'"
    )
    assert _argv(sql) == [
        "ffmpeg", "-i", "a.mka", "-i", "b.mka",
        "-map", "0:a:0", "-c:0", "copy",
        "-map", "1:a:0", "-c:1", "copy",
        "out.mka",
    ]


def test_a_table_function_may_be_called_without_an_alias() -> None:
    sql = ONE_TRACK + "COPY (SELECT one_track.track FROM one_track('a.mka')) TO 'out.mka'"
    assert _argv(sql) == [
        "ffmpeg", "-i", "a.mka", "-map", "0:a:0", "-c:0", "copy", "out.mka",
    ]


def test_a_table_function_is_a_row_source_inside_a_cte() -> None:
    sql = ONE_TRACK + (
        "COPY (WITH picked AS (SELECT t.track AS track FROM one_track('a.mka') AS t)\n"
        "      SELECT picked.track FROM picked)\n"
        "TO 'out.mka'"
    )
    assert _argv(sql) == [
        "ffmpeg", "-i", "a.mka", "-map", "0:a:0", "-c:0", "copy", "out.mka",
    ]


def test_a_package_qualified_source_call_keeps_its_alias_after_adoption(
    tmp_path: Path,
) -> None:
    """A ``RETURNS source`` call adopted from a package -- ``ffrwd.moq.subscribe(...)
    s`` -- must reach lowering with its alias intact.

    Regression for `_Expander._adopt`: a FROM-position `_WasmSite`'s `node`
    is the whole `exp.Table` (alias included, `_row_source_site`), and
    `_adopt` used to hand back a bare `exp.Anonymous`, so `site.node.replace
    (replacement)` in `_expand_within` discarded the Table wrapper and the
    alias with it -- the adopted call landed bare in the FROM list, and
    `_check_wasm_calls` rejected it as "not in FROM" even though it plainly
    was. `_adopt` now rebuilds the `exp.Table` around the adopted call the
    same way `_expand_row_source` already does for a `RETURNS TABLE`
    function's FROM item, so the alias survives and lowering binds the
    catalog under it.
    """
    (tmp_path / "src").mkdir(parents=True)
    (tmp_path / "src" / "moq.sql").write_text(
        "CREATE FUNCTION subscribe(relay text, broadcast text) RETURNS source\n"
        "  AS 'moq.wasm', 'subscribe' LANGUAGE wasm;\n",
        encoding="utf-8",
    )
    (tmp_path / "ffrwd.json").write_text(
        json.dumps(
            {"name": "ffrwd/moq", "version": "1.0.0", "lib": {"subscribe": "src/moq.sql"}}
        ),
        encoding="utf-8",
    )
    packages = discover(tmp_path)
    assert packages is not None
    res = resolve(
        parse(
            "SELECT s.video[1] FROM ffrwd.moq.subscribe('relay-host', 'live') s "
            "WHERE s.height = 720"
        ),
        packages=packages,
    )
    assert list(res.wasm_sources) == ["s"]
    declared = next(iter(res.wasm.values()))

    rung = SourceRendition(name="720p", bandwidth=2_000_000, codecs=None, language=None)
    catalog = SourceCatalog(
        tracks=(
            WasmSourceTrack(
                codec="h264", time_base=(1, 90000), kind="video",
                width=1280, height=720, sample_rate=None, channels=None,
                extradata=b"", profile=None, level=None, row=0, rendition=rung,
            ),
        ),
        bounded=False,
    )
    described = Described(
        world="ffrwd:av@0.13.0",
        name="subscribe",
        params_schema={
            "properties": {
                "relay": {"type": "string"},
                "broadcast": {"type": "string"},
            }
        },
        source=True,
    )
    g = lower(
        res, {}, registry=_snapshot_registry(),
        describes={declared.module: described},
        probe_source=lambda module, params: catalog,
    )
    assert [(o.ref, o.type) for o in g.outputs] == [("src:s:v:0", "video")]
    assert g.module_sources["s"].alias == "s"


def test_a_call_joins_a_cte_the_query_already_wrote() -> None:
    sql = ONE_TRACK + (
        "COPY (WITH vid AS (SELECT v AS track FROM input('a.mkv') i, unnest(i.video) v)\n"
        "      SELECT vid.track, t.track FROM vid, one_track('b.mka') AS t)\n"
        "TO 'out.mkv'"
    )
    probes = {
        "i": ProbeResult(
            streams=[
                StreamMeta(
                    type="video", index=0, metadata={}, width=640, height=360,
                    fps="25/1", sample_rate=None, codec="h264", channels=None,
                    channel_layout=None, bitrate=None, duration=None, color_transfer=None,
                )
            ]
        )
    }
    assert _argv(sql, probes) == [
        "ffmpeg", "-i", "a.mkv", "-i", "b.mka",
        "-map", "0:v:0", "-c:0", "copy",
        "-map", "1:a:0", "-c:1", "copy",
        "out.mkv",
    ]


def test_a_table_function_body_may_call_a_value_function() -> None:
    """The call computes the declared value column; the stream rides beside it,
    keeping whatever tags it already carried."""
    sql = NORMALIZE_LANG + (
        "CREATE FUNCTION langs(path text) RETURNS TABLE(track audio_stream, language text) AS $$\n"
        "  SELECT a, normalize_lang(a.tags.language) FROM input(path) g, unnest(g.audio) a\n"
        "$$ LANGUAGE sql;\n"
        "COPY (SELECT array_agg(t.track) FROM langs('a.mka') AS t) TO 'out.mka'"
    )
    probes = {"langs_1_g": _audio_probe({"language": "english"})}
    assert _argv(sql, probes) == [
        "ffmpeg", "-i", "a.mka",
        "-map", "0:a:0", "-c:0", "copy", "-metadata:s:0", "language=english",
        "out.mka",
    ]


# ---------------------------------------------------------------------------
# a table function's body may tag its own streams
# ---------------------------------------------------------------------------


TAGGING_LANGS = (
    "CREATE FUNCTION langs(path text) RETURNS TABLE(track audio_stream, language text) AS $$\n"
    "  SELECT a, normalize_lang(a.tags.language),\n"
    "         STRUCT(normalize_lang(a.tags.language) AS language) AS tags\n"
    "  FROM input(path) g, unnest(g.audio) a\n"
    "$$ LANGUAGE sql;\n"
)


def test_a_table_function_body_tags_the_streams_it_returns() -> None:
    """The metadata map is an assertion about the body's own streams, not a
    column of its rows, so it rides to the caller's output un-renamed."""
    sql = NORMALIZE_LANG + TAGGING_LANGS + (
        "COPY (SELECT array_agg(t.track) FROM langs('a.mka') AS t) TO 'out.mka'"
    )
    probes = {"langs_1_g": _audio_probe({"language": "english"})}
    assert _argv(sql, probes) == [
        "ffmpeg", "-i", "a.mka",
        "-map", "0:a:0", "-c:0", "copy", "-metadata:s:0", "language=eng",
        "out.mka",
    ]


def test_a_body_tags_map_and_a_value_column_do_not_interfere() -> None:
    """Both at once: the map tags the streams, the declared column stays a
    value the caller reads -- here to name the file."""
    sql = NORMALIZE_LANG + TAGGING_LANGS + (
        "COPY (SELECT t.track FROM langs('a.mka') AS t) TO (t.language || '.mka')"
    )
    probes = {"langs_1_g": _audio_probe({"language": "english"})}
    args = _argv(sql, probes)
    assert args[-1] == "eng.mka"
    assert "language=eng" in args


def test_a_body_tags_map_is_not_a_column_of_the_alias() -> None:
    """It is spent on the body's streams; the caller sees only what was declared."""
    sql = NORMALIZE_LANG + TAGGING_LANGS + (
        "COPY (SELECT array_agg(t.track), t.tags AS tags FROM langs('a.mka') AS t)\n"
        "TO 'out.mka'"
    )
    probes = {"langs_1_g": _audio_probe({"language": "english"})}
    with pytest.raises(FfrwdError) as caught:
        lower(_resolved(sql), probes, registry=_snapshot_registry())
    error = caught.value
    assert error.code is ErrorCode.UNSUPPORTED_SQL
    assert "unknown column 't.tags'" in error.message
    assert error.hint is not None and "exposes" in error.hint


def test_returns_table_cannot_declare_a_column_called_tags() -> None:
    """The name belongs to the metadata map; a declared column is a stream or
    a value."""
    sql = (
        "CREATE FUNCTION bad(path text) RETURNS TABLE(track audio_stream, tags text) AS $$\n"
        "  SELECT a, 'x' FROM input(path) g, unnest(g.audio) a\n"
        "$$ LANGUAGE sql;\n"
        "COPY (SELECT array_agg(t.track) FROM bad('a.mka') AS t) TO 'out.mka'"
    )
    error = _rejects(sql, ErrorCode.UNSUPPORTED_SQL, "declares a column called 'tags'")
    assert error.hint is not None and "STRUCT('Main' AS title)" in error.hint


# ---------------------------------------------------------------------------
# table-returning functions: the rejections
# ---------------------------------------------------------------------------


def test_a_table_function_in_the_select_list_is_rejected() -> None:
    sql = ONE_TRACK + "COPY (SELECT one_track('a.mka')) TO 'out.mka'"
    error = _rejects(sql, ErrorCode.UNSUPPORTED_SQL, "returns a table, not a value")
    assert error.hint is not None and "FROM" in error.hint


def test_a_field_read_off_a_table_function_is_rejected() -> None:
    sql = ONE_TRACK + (
        "COPY (SELECT (one_track('a.mka')).track FROM input('b.mka') f) TO 'out.mka'"
    )
    error = _rejects(sql, ErrorCode.UNSUPPORTED_SQL, "returns a table, not a value")
    assert error.hint is not None and "one input per read" in error.hint


def test_a_table_function_call_with_the_wrong_arity_is_rejected() -> None:
    sql = ONE_TRACK + "COPY (SELECT t.track FROM one_track('a.mka', 'b') AS t) TO 'out.mka'"
    _rejects(sql, ErrorCode.UDF_ARG_TYPE, "got 2 arguments")


def test_a_body_column_count_must_match_returns_table() -> None:
    sql = (
        "CREATE FUNCTION pair(path text) RETURNS TABLE(track audio_stream, language text) AS $$\n"
        "  SELECT g.audio[1] FROM input(path) g\n"
        "$$ LANGUAGE sql;\n"
        "COPY (SELECT t.track FROM pair('a.mka') AS t) TO 'out.mka'"
    )
    _rejects(sql, ErrorCode.UNSUPPORTED_SQL, "selects 1 column")


def test_a_recursive_table_function_is_rejected() -> None:
    sql = (
        "CREATE FUNCTION loop_rows(path text) RETURNS TABLE(track audio_stream) AS $$\n"
        "  SELECT t.track FROM loop_rows(path) AS t\n"
        "$$ LANGUAGE sql;\n"
        "COPY (SELECT t.track FROM loop_rows('a.mka') AS t) TO 'out.mka'"
    )
    _rejects(sql, ErrorCode.UNSUPPORTED_SQL, "recursive")


def test_a_table_function_declares_at_least_one_column() -> None:
    sql = (
        "CREATE FUNCTION empty_rows(path text) RETURNS TABLE() AS $$\n"
        "  SELECT g.audio[1] FROM input(path) g\n"
        "$$ LANGUAGE sql;\n"
        "COPY (SELECT t.track FROM empty_rows('a.mka') AS t) TO 'out.mka'"
    )
    _rejects(sql, ErrorCode.UNSUPPORTED_SQL, "RETURNS TABLE")


def test_a_table_column_type_comes_from_the_vocabulary() -> None:
    sql = (
        "CREATE FUNCTION odd(path text) RETURNS TABLE(track blob) AS $$\n"
        "  SELECT g.audio[1] FROM input(path) g\n"
        "$$ LANGUAGE sql;\n"
        "COPY (SELECT t.track FROM odd('a.mka') AS t) TO 'out.mka'"
    )
    _rejects(sql, ErrorCode.UNSUPPORTED_SQL, "declares an unknown type")


def test_a_table_column_is_named_once() -> None:
    sql = (
        "CREATE FUNCTION twice(path text) RETURNS TABLE(track audio_stream, track text) AS $$\n"
        "  SELECT g.audio[1], 'x' FROM input(path) g\n"
        "$$ LANGUAGE sql;\n"
        "COPY (SELECT t.track FROM twice('a.mka') AS t) TO 'out.mka'"
    )
    _rejects(sql, ErrorCode.UNSUPPORTED_SQL, "'track' twice")


def test_a_table_function_nothing_calls_is_rejected() -> None:
    sql = ONE_TRACK + "COPY (SELECT f.audio[1] FROM input('a.mka') f) TO 'out.mka'"
    _rejects(sql, ErrorCode.UNSUPPORTED_SQL, "never called")


def test_a_rejection_inside_a_table_body_lands_on_the_call_site() -> None:
    sql = (
        "CREATE FUNCTION bad_rows(path text) RETURNS TABLE(track audio_stream) AS $$\n"
        "  SELECT g.audio[1 / 0] FROM input(path) g\n"
        "$$ LANGUAGE sql;\n"
        "COPY (SELECT t.track\n"
        "      FROM bad_rows('a.mka') AS t)\n"
        "TO 'out.mka'"
    )
    error = _rejects(sql, ErrorCode.UNSUPPORTED_SQL, "subscript")
    assert error.line == 5, error
    assert "body of bad_rows()" in error.message, error.message


# ---------------------------------------------------------------------------
# guardrail: no panics
# ---------------------------------------------------------------------------


_MALFORMED = [
    "CREATE FUNCTION",
    "CREATE FUNCTION m",
    "CREATE FUNCTION m() RETURNS text AS $$ $$ LANGUAGE sql",
    "CREATE FUNCTION m() RETURNS text AS $$ COPY (SELECT 1) TO 'x' $$ LANGUAGE sql",
    "CREATE FUNCTION m() RETURNS text AS $$ CREATE VIEW v AS SELECT 1 $$ LANGUAGE sql",
    "CREATE FUNCTION m() RETURNS text AS $$ SELECT 1 UNION ALL SELECT 2 $$ LANGUAGE sql",
    "CREATE FUNCTION m.n(a text) RETURNS text AS $$ SELECT a $$ LANGUAGE sql",
    "CREATE FUNCTION m(a text[][]) RETURNS text AS $$ SELECT 'x' $$ LANGUAGE sql",
    "CREATE FUNCTION m(a text) RETURNS text[][] AS $$ SELECT 'x' $$ LANGUAGE sql",
    'CREATE FUNCTION "M"(a text) RETURNS text AS $$ SELECT a $$ LANGUAGE sql',
    "CREATE FUNCTION m(a text) RETURNS text AS $$ SELECT a $$ LANGUAGE sql; SELECT m()",
    "CREATE FUNCTION m(a text) RETURNS text AS $$ SELECT b $$ LANGUAGE sql; SELECT m('x')",
    "CREATE FUNCTION m(a text) RETURNS text AS $$ SELECT * $$ LANGUAGE sql; SELECT m('x')",
    "CREATE FUNCTION m(a text) RETURNS text AS $$ SELECT ffmpeg.sine() $$ LANGUAGE sql;"
    " SELECT m('x')",
    "CREATE FUNCTION m(a text) RETURNS TABLE(x text) AS $$ SELECT a $$ LANGUAGE sql;"
    " SELECT m('x')",
    "CREATE FUNCTION m(a text) RETURNS TABLE(x text) AS $$ SELECT a $$ LANGUAGE sql;"
    " SELECT t.x FROM m('x') AS t, m('y') AS t",
    "CREATE FUNCTION m(a text) RETURNS TABLE(x text) AS $$ SELECT a $$ LANGUAGE sql;"
    " SELECT t.x FROM m('x') AS t (y)",
    "CREATE FUNCTION m(a text) RETURNS TABLE AS $$ SELECT a $$ LANGUAGE sql;"
    " SELECT t.x FROM m('x') AS t",
    "CREATE FUNCTION m(a text) RETURNS TABLE(x text) AS $$ SELECT a $$ LANGUAGE sql;"
    " SELECT t.x FROM unnest(m('x')) AS t",
]


def test_a_malformed_definition_is_a_rejection_not_a_crash() -> None:
    for sql in _MALFORMED:
        try:
            lower(_resolved(sql), {}, registry=_snapshot_registry())
        except FfrwdError as error:
            assert error.code is not ErrorCode.INTERNAL, error
            assert error.line is not None and 1 <= error.line <= 10, error
