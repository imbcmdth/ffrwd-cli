from __future__ import annotations

import pytest
from sqlglot import exp

from ffrwd.errors import ErrorCode, FfrwdError
from ffrwd.parser import (
    RawInputOption,
    RawSink,
    RawSource,
    Resolved,
    from_entries,
    parse,
    resolve,
    star_qualifier,
    subscript_index,
    union_branches,
)
from ffrwd.types import DISPOSITION_KEYS
from ffrwd.vars import substitute

README_SQL = """WITH pip AS (
  SELECT scale(crop(b.video[1], 1200, 50, 600, 200), 0.5) AS frame
  FROM input('game.mp4') b
)
SELECT overlay(a.video[1], pip.frame, 20, 20)
FROM input('game.mp4') a, pip
"""

# The simplest query that resolves cleanly, for wrapping in a COPY sink.
SINK_QUERY = "SELECT a.video[1] FROM input('x.mp4') a"


def _resolve(sql: str) -> Resolved:
    return resolve(parse(sql))


def _reject(sql: str) -> FfrwdError:
    with pytest.raises(FfrwdError) as excinfo:
        _resolve(sql)
    err = excinfo.value
    assert err.line is not None, "every rejection must be line-anchored"
    assert err.col is not None
    return err


def _projection(sql: str, index: int = 0) -> exp.Expr:
    """The nth top-level projection of a query that resolves cleanly."""
    select = _resolve(sql).branches[0]
    projection = select.expressions[index]
    assert isinstance(projection, exp.Expr)
    return projection


# ---------------------------------------------------------------------------
# parse
# ---------------------------------------------------------------------------


def test_parse_returns_select() -> None:
    tree = parse("SELECT a.video[1] FROM input('x.mp4') a")
    assert isinstance(tree, exp.Select)


def test_parse_returns_union_for_union_all() -> None:
    tree = parse(
        "SELECT a.video[1] FROM input('x') a UNION ALL SELECT b.video[1] FROM input('y') b"
    )
    assert isinstance(tree, exp.Union)


def test_parse_empty_is_parse_error() -> None:
    for text in ("", "   ", "\n\t "):
        with pytest.raises(FfrwdError) as excinfo:
            parse(text)
        assert excinfo.value.code is ErrorCode.PARSE_ERROR, text
        assert excinfo.value.line == 1


def test_parse_garbage_is_parse_error() -> None:
    for text in (
        "SELEC 1",
        "SELECT FROM",
        "SELECT 'unterminated",
        "(((",
        "@@@ not sql @@@",
        "SELECT a.video[1] FROM input('x') a WHERE",
    ):
        with pytest.raises(FfrwdError) as excinfo:
            parse(text)
        assert excinfo.value.code is ErrorCode.PARSE_ERROR, text
        assert excinfo.value.line is not None
        assert excinfo.value.col is not None


def test_parse_error_is_line_anchored() -> None:
    with pytest.raises(FfrwdError) as excinfo:
        parse("SELECT a.video[1]\nFROM input('x') a\nWHERE a.t BETWEEN AND 2")
    assert excinfo.value.code is ErrorCode.PARSE_ERROR
    assert excinfo.value.line == 3


# ---------------------------------------------------------------------------
# resolve — happy paths
# ---------------------------------------------------------------------------


def test_single_input() -> None:
    res = _resolve("SELECT a.video[1] FROM input('clip.mp4') a")
    assert isinstance(res.select, exp.Select)
    assert res.input_paths == ["clip.mp4"]
    assert res.sources == {"a": 0}
    assert res.ctes == {}
    assert res.branches == [res.select]


def test_readme_example_maps_one_file_to_two_inputs() -> None:
    res = _resolve(README_SQL)
    # dedup is per ALIAS, not per path: two aliases -> two -i entries
    assert res.input_paths == ["game.mp4", "game.mp4"]
    assert res.sources == {"b": 0, "a": 1}
    assert list(res.ctes) == ["pip"]
    assert isinstance(res.ctes["pip"], exp.Select)
    assert len(res.branches) == 1


def test_two_aliases_same_file_without_cte() -> None:
    res = _resolve(
        "SELECT overlay(a.video[1], b.video[1], 0, 0) "
        "FROM input('g.mp4') a, input('g.mp4') b"
    )
    assert res.input_paths == ["g.mp4", "g.mp4"]
    assert res.sources == {"a": 0, "b": 1}


def test_cte_names_are_not_inputs() -> None:
    res = _resolve(
        "WITH c AS (SELECT hflip(a.video[1]) AS frame FROM input('x.mp4') a) SELECT c.frame FROM c"
    )
    assert res.input_paths == ["x.mp4"]
    assert res.sources == {"a": 0}
    assert "c" not in res.sources
    assert list(res.ctes) == ["c"]


def test_ctes_in_definition_order() -> None:
    sql = (
        "WITH one AS (SELECT a.video[1] FROM input('a.mp4') a), "
        "two AS (SELECT hflip(one.frame) AS frame FROM one) "
        "SELECT two.frame FROM two"
    )
    res = _resolve(sql)
    assert list(res.ctes) == ["one", "two"]
    assert res.sources == {"a": 0}


def test_union_all_branches() -> None:
    sql = (
        "SELECT a.video[1] FROM input('x.mp4') a "
        "UNION ALL SELECT b.video[1] FROM input('y.mp4') b "
        "UNION ALL SELECT c.video[1] FROM input('z.mp4') c"
    )
    res = _resolve(sql)
    assert isinstance(res.select, exp.Union)
    assert res.input_paths == ["x.mp4", "y.mp4", "z.mp4"]
    assert res.sources == {"a": 0, "b": 1, "c": 2}
    assert len(res.branches) == 3
    assert [b.expressions[0].this.table for b in res.branches] == ["a", "b", "c"]


def test_union_all_inside_cte() -> None:
    sql = (
        "WITH c AS ("
        "  SELECT a.video[1] AS v FROM input('x.mp4') a "
        "  UNION ALL SELECT b.video[1] AS v FROM input('y.mp4') b"
        ") SELECT c.v FROM c"
    )
    res = _resolve(sql)
    assert list(res.ctes) == ["c"]
    assert isinstance(res.ctes["c"], exp.Union)
    assert len(union_branches(res.ctes["c"])) == 2
    assert res.sources == {"a": 0, "b": 1}


def test_where_between_is_accepted() -> None:
    res = _resolve("SELECT a.video[1] FROM input('x.mp4') a WHERE a.t BETWEEN 1 AND 2.5")
    assert res.sources == {"a": 0}
    where = res.select.args.get("where")
    assert isinstance(where, exp.Where)


def test_where_conjunction_per_alias_is_accepted() -> None:
    sql = (
        "SELECT overlay(a.video[1], b.video[1], 0, 0) FROM input('x.mp4') a, input('y.mp4') b "
        "WHERE a.t BETWEEN 0 AND 1 AND b.t BETWEEN 2 AND 3"
    )
    res = _resolve(sql)
    assert res.sources == {"a": 0, "b": 1}


# ---------------------------------------------------------------------------
# resolve — open-ended time windows
# ---------------------------------------------------------------------------


def test_where_gte_open_lower_bound_is_accepted() -> None:
    res = _resolve("SELECT a.video[1] FROM input('x.mp4') a WHERE a.t >= 5")
    assert res.sources == {"a": 0}


def test_where_lte_open_upper_bound_is_accepted() -> None:
    res = _resolve("SELECT a.video[1] FROM input('x.mp4') a WHERE a.t <= 60")
    assert res.sources == {"a": 0}


def test_where_flipped_gte_operand_order_is_accepted() -> None:
    """``120 <= a.t`` is the exact mirror of ``a.t >= 120``, same lower bound.

    VERIFIED (sqlglot 30.17, read="postgres"): operand order is NOT
    normalized at parse time, so this exercises the mirrored branch of
    ``parser._time_bounds`` for real, not the same code path as the
    unflipped form.
    """
    res = _resolve("SELECT a.video[1] FROM input('x.mp4') a WHERE 120 <= a.t")
    assert res.sources == {"a": 0}


def test_where_flipped_lte_operand_order_is_accepted() -> None:
    """``60 >= a.t`` is the exact mirror of ``a.t <= 60``, same upper bound."""
    res = _resolve("SELECT a.video[1] FROM input('x.mp4') a WHERE 60 >= a.t")
    assert res.sources == {"a": 0}


def test_where_gte_and_lte_merge_into_one_window() -> None:
    """``t >= 1 AND t <= 2`` is accepted exactly like ``t BETWEEN 1 AND 2``."""
    res = _resolve("SELECT a.video[1] FROM input('x.mp4') a WHERE a.t >= 1 AND a.t <= 2")
    assert res.sources == {"a": 0}


def test_where_open_bound_mixes_with_between_on_another_alias() -> None:
    sql = (
        "SELECT overlay(a.video[1], b.video[1], 0, 0) FROM input('x.mp4') a, input('y.mp4') b "
        "WHERE a.t >= 1 AND b.t BETWEEN 0 AND 5"
    )
    res = _resolve(sql)
    assert res.sources == {"a": 0, "b": 1}


def test_strict_inequality_is_rejected_with_dedicated_hint() -> None:
    for sql in (
        "SELECT a.video[1] FROM input('x.mp4') a WHERE a.t > 5",
        "SELECT a.video[1] FROM input('x.mp4') a WHERE 5 < a.t",
        "SELECT a.video[1] FROM input('x.mp4') a WHERE a.t < 60",
        "SELECT a.video[1] FROM input('x.mp4') a WHERE 60 > a.t",
    ):
        err = _reject(sql)
        assert err.code is ErrorCode.UNSUPPORTED_SQL, sql
        assert err.hint is not None
        assert "use >= / <=" in err.hint


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT a.video[1] FROM input('x.mp4') a WHERE a.t >= 1 AND a.t >= 2",
        "SELECT a.video[1] FROM input('x.mp4') a WHERE a.t <= 1 AND a.t <= 2",
        "SELECT a.video[1] FROM input('x.mp4') a WHERE a.t BETWEEN 1 AND 2 AND a.t >= 3",
        "SELECT a.video[1] FROM input('x.mp4') a WHERE a.t >= 1 AND a.t BETWEEN 2 AND 3",
    ],
)
def test_a_second_bound_of_the_same_kind_is_rejected(sql: str) -> None:
    """Mirrors the old one-BETWEEN rule: at most one lower and one upper bound
    per alias, whichever forms supplied them."""
    err = _reject(sql)
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert err.hint is not None


def test_non_literal_inequality_bound_is_rejected() -> None:
    sql = "SELECT a.video[1] FROM input('x.mp4') a, input('y.mp4') b WHERE a.t >= b.t"
    err = _reject(sql)
    assert err.code is ErrorCode.UNSUPPORTED_SQL


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT a.video[1] FROM input('x.mp4') a WHERE a.t >= 5 AND a.t <= 2",
        "SELECT a.video[1] FROM input('x.mp4') a WHERE a.t >= 5 AND a.t <= 5",
        "SELECT a.video[1] FROM input('x.mp4') a WHERE a.t BETWEEN 5 AND 2",
    ],
)
def test_empty_time_window_is_rejected_at_compile_time(sql: str) -> None:
    err = _reject(sql)
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert err.hint is not None


def test_alias_case_is_folded_like_postgres() -> None:
    res = _resolve("SELECT A.video[1] FROM input('x.mp4') A WHERE a.t BETWEEN 1 AND 2")
    assert res.sources == {"a": 0}


def test_parenthesized_union_branches() -> None:
    res = _resolve(
        "(SELECT a.video[1] FROM input('x.mp4') a) "
        "UNION ALL (SELECT b.video[1] FROM input('y.mp4') b)"
    )
    assert len(res.branches) == 2
    assert res.sources == {"a": 0, "b": 1}


def test_union_branches_helper_on_plain_select() -> None:
    res = _resolve("SELECT a.video[1] FROM input('x.mp4') a")
    assert union_branches(res.select) == [res.select]


# ---------------------------------------------------------------------------
# resolve — NO_STREAMING_EQUIVALENT
# ---------------------------------------------------------------------------


def test_no_streaming_equivalent() -> None:
    # LIMIT/OFFSET alone over a plain input() alias are NOT in this list: a
    # probe may resolve that alias to a rendition row table, which only
    # lower can tell, so resolve admits them on spec
    # (test_limit_and_offset_over_a_plain_input_alias_now_resolve).
    for sql in (
        "SELECT a.video[1] FROM input('x') a GROUP BY a.video[1]",
        "SELECT a.video[1] FROM input('x') a HAVING count(*) > 1",
        "SELECT a.video[1] FROM input('x') a ORDER BY a.t",
        "SELECT DISTINCT a.video[1] FROM input('x') a",
        "SELECT row_number() OVER (ORDER BY a.t) FROM input('x') a",
        "SELECT count(a.video[1]) FROM input('x') a",
        "SELECT max(a.t) FROM input('x') a",
        "SELECT a.video[1] FROM input('x') a WHERE a.t IN (SELECT b.t FROM input('y') b)",
        "SELECT a.video[1] FROM input('x') a WHERE EXISTS (SELECT 1)",
    ):
        assert _reject(sql).code is ErrorCode.NO_STREAMING_EQUIVALENT, sql


def test_union_without_all_suggests_union_all() -> None:
    err = _reject("SELECT a.video[1] FROM input('x') a UNION SELECT b.video[1] FROM input('y') b")
    assert err.code is ErrorCode.NO_STREAMING_EQUIVALENT
    assert err.hint is not None and "UNION ALL" in err.hint


# ---------------------------------------------------------------------------
# resolve — multiple projections (SELECT list = output stream list)
# ---------------------------------------------------------------------------


def test_two_output_columns_are_legal() -> None:
    res = _resolve("SELECT a.video[1], a.audio[1] FROM input('x') a")
    assert len(res.branches[0].expressions) == 2
    assert res.sources == {"a": 0}


def test_two_output_columns_in_cte_are_legal() -> None:
    res = _resolve(
        "WITH c AS (SELECT a.video[1] AS v, a.audio[1] AS aud FROM input('x') a) "
        "SELECT c.v, c.aud FROM c"
    )
    assert list(res.ctes) == ["c"]
    assert len(res.branches[0].expressions) == 2


def test_many_output_columns_are_legal() -> None:
    res = _resolve(
        "SELECT a.video[1], a.audio[1], a.audio[2], hflip(a.video[1]) FROM input('x') a"
    )
    assert len(res.branches[0].expressions) == 4


def test_multiple_projections_in_every_union_branch() -> None:
    sql = (
        "SELECT a.video[1], a.audio[1] FROM input('x') a "
        "UNION ALL SELECT b.video[1], b.audio[1] FROM input('y') b"
    )
    res = _resolve(sql)
    assert len(res.branches) == 2
    assert all(len(branch.expressions) == 2 for branch in res.branches)


def test_select_with_no_output_column_is_rejected() -> None:
    # sqlglot parses "SELECT FROM t" into a Select with an empty projection
    # list, so the no-projection guard is the one arity check that survives v2.
    err = _reject("SELECT FROM input('x') a")
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "no output column" in err.message


# ---------------------------------------------------------------------------
# resolve — SELECT * / <alias>.*
# ---------------------------------------------------------------------------


def test_star_in_projection_position_is_accepted() -> None:
    for sql in (
        "SELECT * FROM input('x') a",
        "SELECT a.* FROM input('x') a",
        "SELECT a.*, b.audio[1] FROM input('x') a, input('y') b",
        "SELECT a.video[1], b.* FROM input('x') a, input('y') b",
        "SELECT *, a.audio[1] FROM input('x') a",
        'SELECT "A".* FROM input(\'x\') "A"',
        "WITH c AS (SELECT a.video[1] AS f FROM input('x') a) SELECT c.* FROM c",
        "WITH c AS (SELECT * FROM input('x') a) SELECT * FROM c",
        "COPY (SELECT * FROM input('x.mp4') a) TO 'out.mkv'",
    ):
        resolve(parse(sql))


@pytest.mark.parametrize(
    ("sql", "qualifier"),
    [
        ("SELECT * FROM input('x') a", ""),
        ("SELECT a.* FROM input('x') a", "a"),
        # Postgres identifier folding applies to the qualifier like any other.
        ("SELECT A.* FROM input('x') a", "a"),
        ('SELECT "A".* FROM input(\'x\') "A"', "A"),
    ],
)
def test_star_qualifier_reads_both_sqlglot_shapes(sql: str, qualifier: str) -> None:
    """The two VERIFIED shapes: bare ``exp.Star`` vs ``Column(this=Star())``."""
    select = resolve(parse(sql)).branches[0]
    assert star_qualifier(select.expressions[0]) == qualifier


@pytest.mark.parametrize(
    "sql",
    [
        # not a projection: a star inside a function call
        "SELECT scale(a.*, 0.5) FROM input('x') a",
        "SELECT scale(*, 0.5) FROM input('x') a",
        # not a projection: subscripted, or aliased
        "SELECT a.*[1] FROM input('x') a",
        "SELECT * AS everything FROM input('x') a",
        "SELECT a.* AS everything FROM input('x') a",
    ],
)
def test_star_outside_projection_position_is_rejected(sql: str) -> None:
    err = _reject(sql)
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "whole SELECT column" in err.message
    assert err.hint is not None and "<alias>.*" in err.hint


def test_star_over_an_unknown_alias_is_rejected() -> None:
    err = _reject("SELECT z.* FROM input('x') a")
    assert err.code is ErrorCode.UNKNOWN_ALIAS
    assert "'z'" in err.message


def test_star_projection_is_column_anchored() -> None:
    err = _reject("SELECT\n  z.*\nFROM input('x') a")
    assert err.code is ErrorCode.UNKNOWN_ALIAS
    assert err.line == 2


def test_star_in_where_is_still_rejected() -> None:
    err = _reject("SELECT a.video[1] FROM input('x') a WHERE * BETWEEN 1 AND 2")
    assert err.code is ErrorCode.UNSUPPORTED_SQL


def test_count_star_is_still_an_aggregate_rejection() -> None:
    """``count(*)`` must not be mistaken for a star projection."""
    err = _reject("SELECT count(*) FROM input('x') a")
    assert err.code is ErrorCode.NO_STREAMING_EQUIVALENT


def test_count_star_over_track_rows_is_still_an_aggregate_rejection() -> None:
    """Table mode makes metadata columns legal SELECT
    outputs, but it does not make ffrwd a database -- ``COUNT(*)`` over a
    row table (a bare SELECT, unconditionally table-capable) is still rejected
    exactly like any other aggregate."""
    err = _reject("SELECT count(*) FROM input('f.mkv') f, unnest(f.audio) t")
    assert err.code is ErrorCode.NO_STREAMING_EQUIVALENT


# ---------------------------------------------------------------------------
# resolve — SELECT * EXCEPT(...) / REPLACE(...)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * EXCEPT(subtitle) FROM input('x') a",
        "SELECT * EXCEPT(subtitle, data) FROM input('x') a",
        "SELECT a.* EXCEPT(subtitle) FROM input('x') a",
        "SELECT * REPLACE(scale(a.video[1], 1280, -2) AS video) FROM input('x') a",
        "SELECT * REPLACE(a.video[1] AS video, a.audio[1] AS audio) FROM input('x') a",
        "SELECT * EXCEPT(subtitle) REPLACE(a.video[1] AS video) FROM input('x') a",
    ],
)
def test_star_except_and_replace_are_accepted(sql: str) -> None:
    resolve(parse(sql))


def test_star_except_empty_is_rejected() -> None:
    err = _reject("SELECT * EXCEPT() FROM input('x') a")
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "no columns" in err.message


def test_star_replace_empty_is_rejected() -> None:
    err = _reject("SELECT * REPLACE() FROM input('x') a")
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "no columns" in err.message


def test_star_except_duplicate_name_is_rejected() -> None:
    err = _reject("SELECT * EXCEPT(subtitle, subtitle) FROM input('x') a")
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "duplicate name 'subtitle'" in err.message


def test_star_except_and_replace_sharing_a_name_is_rejected() -> None:
    """A column named at most once across EXCEPT and REPLACE together."""
    err = _reject(
        "SELECT * EXCEPT(video) REPLACE(a.video[1] AS video) FROM input('x') a"
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "duplicate name 'video'" in err.message


def test_star_except_qualified_name_is_rejected() -> None:
    err = _reject("SELECT * EXCEPT(a.subtitle) FROM input('x') a")
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "bare column names" in err.message


def test_star_replace_without_as_is_rejected() -> None:
    err = _reject("SELECT * REPLACE(a.video[1]) FROM input('x') a")
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "need a name" in err.message


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * RENAME(video AS v) FROM input('x') a",
        "SELECT * ILIKE 'v%' FROM input('x') a",
    ],
)
def test_star_rename_and_ilike_are_rejected(sql: str) -> None:
    err = _reject(sql)
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "EXCEPT and REPLACE" in err.message


def test_star_replace_expression_is_checked_like_any_projection() -> None:
    """A REPLACE expression is an ordinary SELECT expression: no free pass
    from the generic checks a bare star is exempt from."""
    err = _reject("SELECT * REPLACE(count(*) AS video) FROM input('x') a")
    assert err.code is ErrorCode.NO_STREAMING_EQUIVALENT


def test_star_replace_unknown_column_is_still_checked() -> None:
    err = _reject("SELECT * REPLACE(z.video[1] AS video) FROM input('x') a")
    assert err.code is ErrorCode.UNKNOWN_ALIAS


# ---------------------------------------------------------------------------
# resolve — subtitle / data pseudo-columns
# ---------------------------------------------------------------------------


def test_subtitle_and_data_columns_are_accepted() -> None:
    for sql in (
        "SELECT a.subtitle FROM input('x') a",
        "SELECT a.subtitle[1] FROM input('x') a",
        "SELECT a.data FROM input('x') a",
        "SELECT a.data[2] FROM input('x') a",
        "SELECT a.video[1], a.audio[1], a.subtitle[1], a.data[1] FROM input('x') a",
    ):
        resolve(parse(sql))


# ---------------------------------------------------------------------------
# resolve — stream columns and subscripts
# ---------------------------------------------------------------------------


def test_stream_columns_are_accepted() -> None:
    for sql in (
        "SELECT a.video[1] FROM input('x') a",
        "SELECT a.video FROM input('x') a",
        "SELECT a.audio FROM input('x') a",
        "SELECT a.video[3] FROM input('x') a",
        "SELECT a.audio[2] FROM input('x') a",
        "SELECT a.video [ 1 ] FROM input('x') a",
        "SELECT a.video\n  [1] FROM input('x') a",
        "SELECT a.video[10] FROM input('x') a",
        "SELECT scale(a.video[1], 0.5) FROM input('x') a",
        "SELECT scale(a.video, 0.5) FROM input('x') a",
        "SELECT A.VIDEO[1] FROM input('x') A",
        "SELECT a.video[1] AS v FROM input('x') a",
    ):
        assert _resolve(sql).sources == {"a": 0}, sql


def test_cte_columns_may_have_any_name_and_be_subscripted() -> None:
    # A CTE's columns are named by AS, so the parser must not apply the input
    # pseudo-column whitelist to them — lower validates the name.
    res = _resolve(
        "WITH c AS (SELECT a.audio AS tracks FROM input('x') a) "
        "SELECT c.tracks[2], c.tracks FROM c"
    )
    assert list(res.ctes) == ["c"]
    assert res.sources == {"a": 0}


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT a.bogus FROM input('x') a",
        "SELECT a.captions[1] FROM input('x') a",
        "SELECT hflip(a.frames) FROM input('x') a",
        'SELECT a."Video" FROM input(\'x\') a',
    ],
)
def test_unknown_input_column_is_rejected(sql: str) -> None:
    err = _reject(sql)
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert err.hint is not None and "video" in err.hint


def test_unknown_input_column_is_line_anchored() -> None:
    err = _reject("SELECT\n  a.bogus\nFROM input('x') a")
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert err.line == 2


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT a.video[0] FROM input('x') a",
        "SELECT a.video[-1] FROM input('x') a",
        "SELECT a.video[-10] FROM input('x') a",
        "SELECT a.video[1.5] FROM input('x') a",
        "SELECT a.video['x'] FROM input('x') a",
        "SELECT a.video[x] FROM input('x') a",
        "SELECT a.video[a.t] FROM input('x') a",
        "SELECT a.video[null] FROM input('x') a",
        "SELECT a.video[true] FROM input('x') a",
        "SELECT a.video[1:2] FROM input('x') a",
        "SELECT a.video[1, 2] FROM input('x') a",
        "SELECT a.video[cast(1 AS INT)] FROM input('x') a",
        "SELECT scale(a.video[0], 0.5) FROM input('x') a",
        "SELECT\n  a.video[0]\nFROM input('x') a",
    ],
)
def test_bad_subscript_is_rejected(sql: str) -> None:
    err = _reject(sql)
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert err.hint is not None and "stream subscripts are 1-based" in err.hint
    # anchored on the line the subscript itself is written on
    assert err.line == sql.count("\n", 0, sql.index("[")) + 1


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT a.video[1][2] FROM input('x') a",
        "SELECT a.video[1][1][1] FROM input('x') a",
    ],
)
def test_chained_subscript_is_rejected(sql: str) -> None:
    err = _reject(sql)
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "chained" in err.message


def test_subscripting_a_function_result_is_rejected() -> None:
    err = _reject("SELECT scale(a.video[1], 0.5)[1] FROM input('x') a")
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "stream columns" in err.message


def test_subscript_on_unknown_alias_is_unknown_alias() -> None:
    assert _reject("SELECT z.video[1] FROM input('x') a").code is ErrorCode.UNKNOWN_ALIAS


# ---------------------------------------------------------------------------
# fields of a filter output: nothing probed them
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("sql", "path", "call"),
    [
        ("SELECT scale(a.video[1], 640, -2).width FROM input('x') a", "width", "scale"),
        (
            "SELECT volume(t, 0.2).tags.language "
            "FROM input('x') a, unnest(a.audio) t",
            "tags.language",
            "volume",
        ),
        (
            "SELECT volume(t, 0.2).disposition.forced "
            "FROM input('x') a, unnest(a.audio) t",
            "disposition.forced",
            "volume",
        ),
        (
            "SELECT (hflip(a.video[1])).codec FROM input('x') a",
            "codec",
            "hflip",
        ),
        (
            "SELECT ffmpeg.sine(duration => 1).codec FROM input('x') a",
            "codec",
            "ffmpeg.sine",
        ),
    ],
)
def test_a_field_of_a_filter_output_is_rejected(sql: str, path: str, call: str) -> None:
    err = _reject(sql)
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert f"'.{path}' reads a field of the output of {call}()" in err.message
    assert err.hint is not None and f"t.{path}" in err.hint


def test_a_filter_output_field_in_a_where_is_rejected_too() -> None:
    err = _reject(
        "SELECT t FROM input('x') a, unnest(a.audio) t WHERE volume(t, 0.2).codec = 'aac'"
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "was never probed" in err.message


def test_a_filter_output_field_rejection_is_line_anchored() -> None:
    err = _reject("SELECT\n  scale(a.video[1], 640, -2).width\nFROM input('x') a")
    assert err.line == 2


def test_a_bare_namespaced_call_is_not_a_field_read() -> None:
    """`ffmpeg.sine(...)` is a Dot too; only a field AFTER it is the rejection."""
    resolve(parse("SELECT ffmpeg.sine(duration => 1) FROM input('x') a"))


# ---------------------------------------------------------------------------
# subscript_index — sqlglot rebases subscripts at parse time (INDEX_OFFSET)
# ---------------------------------------------------------------------------


def test_sqlglot_rebases_the_subscript_at_parse_time() -> None:
    # Guards the assumption subscript_index is built on: read="postgres" has
    # INDEX_OFFSET = 1, so the parser stores <written index> - 1 and the
    # generator adds it back. If sqlglot ever stops doing this, this test fails
    # loudly instead of every subscript silently compiling off by one.
    for sql_index, ast_literal in {1: "0", 2: "1", 3: "2", 10: "9"}.items():
        bracket = _projection(f"SELECT a.video[{sql_index}] FROM input('x') a")
        assert isinstance(bracket, exp.Bracket)
        literal = bracket.expressions[0]
        assert isinstance(literal, exp.Literal)
        assert literal.this == ast_literal
        assert bracket.sql(dialect="postgres") == f"a.video[{sql_index}]"


def test_subscript_index_undoes_the_rebase() -> None:
    for sql_index in (1, 2, 7, 128):
        bracket = _projection(f"SELECT a.audio[{sql_index}] FROM input('x') a")
        assert isinstance(bracket, exp.Bracket)
        assert subscript_index(bracket) == sql_index


def test_subscript_index_of_a_bare_column_is_not_applicable() -> None:
    # Bare a.video is a Column, never a Bracket: callers check the node type.
    assert isinstance(_projection("SELECT a.video FROM input('x') a"), exp.Column)


def test_subscript_index_rejects_non_literals() -> None:
    bracket = parse("SELECT a.video[x] FROM input('x') a").expressions[0]
    assert isinstance(bracket, exp.Bracket)
    assert subscript_index(bracket) is None


def test_constant_folded_subscript_is_accepted_as_its_value() -> None:
    # KNOWN sqlglot behaviour: it simplifies the rebased index, so a.audio[1+1]
    # is indistinguishable from a.audio[2] by the time the parser sees it.
    # Documented rather than worked around — the result is still a valid stream.
    bracket = _projection("SELECT a.audio[1 + 1] FROM input('x') a")
    assert isinstance(bracket, exp.Bracket)
    assert subscript_index(bracket) == 2


# ---------------------------------------------------------------------------
# resolve — UNKNOWN_ALIAS
# ---------------------------------------------------------------------------


def test_unknown_alias_in_select() -> None:
    assert _reject("SELECT z.video[1] FROM input('x') a").code is ErrorCode.UNKNOWN_ALIAS


def test_unknown_alias_in_where() -> None:
    sql = "SELECT a.video[1] FROM input('x') a WHERE z.t BETWEEN 1 AND 2"
    assert _reject(sql).code is ErrorCode.UNKNOWN_ALIAS


def test_unknown_table_in_from() -> None:
    assert _reject("SELECT c.v FROM c").code is ErrorCode.UNKNOWN_ALIAS


def test_cte_cannot_forward_reference() -> None:
    sql = (
        "WITH c AS (SELECT d.v FROM d), "
        "d AS (SELECT a.video[1] AS v FROM input('x') a) SELECT c.v FROM c"
    )
    assert _reject(sql).code is ErrorCode.UNKNOWN_ALIAS


def test_unknown_alias_is_line_anchored() -> None:
    sql = "SELECT\n  z.video[1]\nFROM input('x') a"
    err = _reject(sql)
    assert err.code is ErrorCode.UNKNOWN_ALIAS
    assert err.line == 2


# ---------------------------------------------------------------------------
# resolve — UNSUPPORTED_SQL
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT a.video[1] FROM input('x') a JOIN input('y') b ON a.t = b.t",
        "SELECT a.video[1] FROM input('x') a INNER JOIN input('y') b ON a.t = b.t",
        "SELECT a.video[1] FROM input('x') a LEFT JOIN input('y') b ON a.t = b.t",
        "SELECT a.video[1] FROM input('x') a CROSS JOIN input('y') b",
    ],
)
def test_explicit_join_syntax(sql: str) -> None:
    err = _reject(sql)
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert err.hint is not None and "comma" in err.hint


def test_input_without_alias() -> None:
    err = _reject("SELECT frame FROM input('x')")
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert err.hint is not None and "alias" in err.hint


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT a.video[1] FROM input() a",
        "SELECT a.video[1] FROM input('x', 'y') a",
        "SELECT a.video[1] FROM input(1) a",
    ],
)
def test_input_requires_one_string_literal(sql: str) -> None:
    assert _reject(sql).code is ErrorCode.UNSUPPORTED_SQL


def test_unknown_table_function() -> None:
    assert _reject("SELECT a.video[1] FROM stream('x') a").code is ErrorCode.UNSUPPORTED_SQL


# ---------------------------------------------------------------------------
# input() named options
# ---------------------------------------------------------------------------


def test_input_with_no_named_options_has_no_input_options_entry() -> None:
    res = _resolve("SELECT a.video[1] FROM input('x.mp4') a")
    assert res.input_options == {}


def test_input_named_option_is_collected() -> None:
    res = _resolve("SELECT a.video[1] FROM input('x.png', loop => true) a")
    assert list(res.input_options) == ["a"]
    options = res.input_options["a"]
    assert len(options) == 1
    option = options[0]
    assert isinstance(option, RawInputOption)
    assert option.name == "loop"
    assert isinstance(option.value, exp.Boolean)


def test_input_multiple_named_options_keep_written_order() -> None:
    res = _resolve(
        "SELECT a.video[1] FROM input('x.png', loop => true, framerate => 15) a"
    )
    assert [o.name for o in res.input_options["a"]] == ["loop", "framerate"]


def test_input_option_name_is_verbatim_not_folded() -> None:
    """Unlike a sink option, input options reuse Kwarg's case-sensitive name."""
    res = _resolve("SELECT a.video[1] FROM input('x.mp4', HwAccel => 'cuda') a")
    assert res.input_options["a"][0].name == "HwAccel"


def test_two_input_aliases_keep_separate_option_sets() -> None:
    res = _resolve(
        "SELECT a.video[1], b.video[1] FROM input('x.png', loop => true) a, "
        "input('y.mp4', hwaccel => 'cuda') b"
    )
    assert [o.name for o in res.input_options["a"]] == ["loop"]
    assert [o.name for o in res.input_options["b"]] == ["hwaccel"]


def test_input_duplicate_named_option_is_rejected() -> None:
    err = _reject(
        "SELECT a.video[1] FROM input('x.png', loop => true, loop => false) a"
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL


def test_input_positional_after_named_option_is_rejected() -> None:
    err = _reject("SELECT a.video[1] FROM input('x.png', loop => true, 5) a")
    assert err.code is ErrorCode.UNSUPPORTED_SQL


def test_input_second_positional_before_any_named_option_is_rejected() -> None:
    err = _reject("SELECT a.video[1] FROM input('x.png', 5, loop => true) a")
    assert err.code is ErrorCode.UNSUPPORTED_SQL


def test_input_named_option_with_no_value_is_rejected() -> None:
    # sqlglot itself rejects a valueless `=>` -- never reaches the resolver.
    err = _reject("SELECT a.video[1] FROM input('x.png', loop =>) a")
    assert err.code is ErrorCode.PARSE_ERROR


# ---------------------------------------------------------------------------
# FROM ffmpeg.<source>(...) alias
# ---------------------------------------------------------------------------
#
# Shape only: which sources exist and which options they take is the installed
# ffmpeg's business (tests/test_lower.py, against a fixture registry).
#
# The sqlglot 30.17 shape this pass keys on -- MEASURED, and different from
# the same namespace in CALL position, which is an `exp.Dot`:
# `FROM ffmpeg.testsrc(duration => 2) t` is
# `Table(this=Anonymous(testsrc, [Kwarg]), db=Identifier(ffmpeg),
#  alias=TableAlias(t))`. `test_a_source_parses_as_a_table_with_a_db_qualifier`
# pins it so a sqlglot upgrade that moves the qualifier cannot pass silently.


def test_a_source_parses_as_a_table_with_a_db_qualifier() -> None:
    table = _resolve("SELECT t.video[1] FROM ffmpeg.testsrc(duration => 2) t").branches[
        0
    ].args["from_"].this
    assert isinstance(table, exp.Table)
    assert isinstance(table.this, exp.Anonymous)
    assert str(table.this.this) == "testsrc"
    db = table.args.get("db")
    assert isinstance(db, exp.Identifier) and db.name == "ffmpeg"
    assert not isinstance(table.this, exp.Dot)


def test_source_is_collected_with_its_raw_options() -> None:
    res = _resolve(
        "SELECT t.video[1] FROM ffmpeg.testsrc(duration => 2, size => '320x240') t"
    )
    assert list(res.source_filters) == ["t"]
    raw = res.source_filters["t"]
    assert isinstance(raw, RawSource)
    assert raw.alias == "t"
    assert raw.name == "testsrc"
    assert [o.name for o in raw.options] == ["duration", "size"]
    assert isinstance(raw.options[0].value, exp.Literal)


def test_a_source_takes_no_input_index() -> None:
    """No `-i`: a source is a zero-input filter, so it is in neither table."""
    res = _resolve("SELECT t.video[1] FROM ffmpeg.testsrc(duration => 2) t")
    assert res.input_paths == []
    assert res.sources == {}


def test_a_source_with_empty_parens_is_accepted() -> None:
    res = _resolve("SELECT t.video[1] FROM ffmpeg.testsrc() t")
    assert res.source_filters["t"].options == ()


def test_a_source_without_parens_is_rejected_with_a_hint() -> None:
    err = _reject("SELECT t.video[1] FROM ffmpeg.testsrc t")
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "'ffmpeg.testsrc' is not a table" in err.message
    assert err.hint is not None and "is a CALL" in err.hint


def test_a_source_requires_an_alias() -> None:
    err = _reject("SELECT t.video[1] FROM ffmpeg.testsrc(duration => 2)")
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "ffmpeg.testsrc() requires an alias" in err.message


def test_a_source_takes_no_positional_arguments() -> None:
    for sql in (
        "SELECT t.video[1] FROM ffmpeg.testsrc(2) t",
        "SELECT t.video[1] FROM ffmpeg.testsrc(2, duration => 1) t",
        "SELECT t.video[1] FROM ffmpeg.testsrc(duration => 1, 2) t",
    ):
        err = _reject(sql)
        assert err.code is ErrorCode.UNSUPPORTED_SQL, sql
        assert "no stream inputs" in err.message, sql


def test_a_source_rejects_a_duplicate_option() -> None:
    err = _reject("SELECT t.video[1] FROM ffmpeg.testsrc(duration => 1, duration => 2) t")
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "duplicate named argument 'duration'" in err.message


def test_a_source_option_name_is_verbatim_and_its_own_name_is_folded() -> None:
    """Function names are case-insensitive; ffmpeg AVOption names are not."""
    res = _resolve("SELECT t.video[1] FROM FFMPEG.TestSrc(Duration => 2) t")
    raw = res.source_filters["t"]
    assert raw.name == "testsrc"
    assert raw.options[0].name == "Duration"


def test_a_source_alias_may_use_as() -> None:
    res = _resolve("SELECT t.video[1] FROM ffmpeg.testsrc(duration => 2) AS t")
    assert list(res.source_filters) == ["t"]


def test_a_source_joins_an_input_with_a_comma() -> None:
    res = _resolve(
        "SELECT f.video[1], s.audio[1] FROM input('a.mp4') f, "
        "ffmpeg.anullsrc(duration => 3) s"
    )
    assert res.input_paths == ["a.mp4"]
    assert res.sources == {"f": 0}
    assert list(res.source_filters) == ["s"]


def test_a_source_is_legal_in_a_cte_body_and_a_union_branch() -> None:
    res = _resolve(
        "WITH bg AS (SELECT t.video[1] AS v FROM ffmpeg.testsrc(duration => 2) t) "
        "SELECT bg.v FROM bg"
    )
    assert list(res.source_filters) == ["t"]
    res = _resolve(
        "SELECT f.video[1] FROM input('a.mp4') f UNION ALL "
        "SELECT t.video[1] FROM ffmpeg.testsrc(duration => 2) t"
    )
    assert list(res.source_filters) == ["t"]


def test_a_source_alias_is_unique_across_the_whole_query() -> None:
    for sql in (
        "SELECT t.video[1] FROM ffmpeg.testsrc() t, ffmpeg.testsrc() t",
        "SELECT t.video[1] FROM input('x.mp4') t, ffmpeg.testsrc() t",
        "SELECT t.video[1] FROM ffmpeg.testsrc() t, input('x.mp4') t",
    ):
        err = _reject(sql)
        assert err.code is ErrorCode.UNSUPPORTED_SQL, sql
        assert "duplicate name 't'" in err.message, sql


def test_a_source_may_not_be_aliased_ffmpeg() -> None:
    """The namespace name stays reserved in FROM position too."""
    err = _reject("SELECT ffmpeg.video[1] FROM ffmpeg.testsrc(duration => 2) ffmpeg")
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "'ffmpeg' is reserved for the filter namespace" in err.message


def test_a_three_part_name_is_not_the_namespace() -> None:
    err = _reject("SELECT t.video[1] FROM x.ffmpeg.testsrc(duration => 2) t")
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "qualified table names are not supported" in err.message


def test_a_non_namespace_qualifier_is_still_rejected() -> None:
    err = _reject("SELECT a.video[1] FROM public.clips a")
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "qualified table names are not supported" in err.message


def test_a_bare_source_name_in_from_is_not_a_table_function() -> None:
    """The namespace is mandatory (`random` etc. collide bare)."""
    err = _reject("SELECT t.video[1] FROM testsrc(duration => 2) t")
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "unsupported table function testsrc()" in err.message


def test_a_source_column_is_not_whitelisted_by_the_parser() -> None:
    """Which columns a source exposes depends on its output pad TYPE, which
    only the registry knows -- so the resolver checks the alias and leaves the
    column name to lower (unlike an input's fixed pseudo-column set)."""
    res = _resolve("SELECT t.audio[1] FROM ffmpeg.testsrc(duration => 2) t")
    assert list(res.source_filters) == ["t"]


def test_a_source_rejection_is_line_anchored() -> None:
    err = _reject("SELECT t.video[1]\nFROM ffmpeg.testsrc(2) t")
    assert err.line == 2


def test_unsupported_where_forms() -> None:
    for sql in (
        "SELECT a.video[1] FROM input('x') a WHERE a.t = 1",
        "SELECT a.video[1] FROM input('x') a WHERE a.t > 1",
        "SELECT a.video[1] FROM input('x') a WHERE a.t BETWEEN 1 AND 2 OR a.t BETWEEN 3 AND 4",
        "SELECT a.video[1] FROM input('x') a WHERE NOT a.t BETWEEN 1 AND 2",
        "SELECT a.video[1] FROM input('x') a WHERE a.x BETWEEN 1 AND 2",
        "SELECT a.video[1] FROM input('x') a WHERE a.t BETWEEN 1 AND 'two'",
        "SELECT a.video[1] FROM input('x') a WHERE t BETWEEN 1 AND 2",
        "SELECT a.video[1] FROM input('x') a WHERE a.t BETWEEN 1 AND 2 AND a.t BETWEEN 3 AND 4",
    ):
        err = _reject(sql)
        assert err.code is ErrorCode.UNSUPPORTED_SQL, sql
        assert err.hint is not None


def test_duplicate_cte_name() -> None:
    sql = (
        "WITH c AS (SELECT a.video[1] FROM input('x') a), "
        "c AS (SELECT b.video[1] AS v FROM input('y') b) SELECT c.v FROM c"
    )
    assert _reject(sql).code is ErrorCode.UNSUPPORTED_SQL


def test_duplicate_alias() -> None:
    assert (
        _reject("SELECT a.video[1] FROM input('x') a, input('y') a").code
        is ErrorCode.UNSUPPORTED_SQL
    )


def test_alias_colliding_with_cte_name() -> None:
    sql = "WITH c AS (SELECT a.video[1] FROM input('x') a) SELECT c.video[1] FROM input('y') c, c"
    assert _reject(sql).code is ErrorCode.UNSUPPORTED_SQL


def test_outside_the_surface() -> None:
    for sql in (
        "INSERT INTO t VALUES (1)",
        "UPDATE t SET x = 1",
        "DELETE FROM t",
        "CREATE TABLE t (x INT)",
        "SELECT 1; SELECT 2",
        "SELECT 1",
        "SELECT a.video[1] FROM (SELECT b.video[1] FROM input('x') b) a",
        "SELECT a.video[1] FROM input('x') a EXCEPT SELECT b.video[1] FROM input('y') b",
        "SELECT a.video[1] FROM input('x') a INTERSECT SELECT b.video[1] FROM input('y') b",
        "WITH RECURSIVE c AS (SELECT a.video[1] AS v FROM input('x') a) SELECT c.v FROM c",
        "WITH c AS (WITH d AS (SELECT a.video[1] AS v FROM input('x') a) SELECT d.v FROM d) "
        "SELECT c.v FROM c",
        "SELECT a.video[1] FROM input('x') a(c)",
        "SELECT video FROM input('x') a",
        "SELECT a.video[1] FROM sch.tbl a",
    ):
        assert _reject(sql).code is ErrorCode.UNSUPPORTED_SQL, sql


def test_cte_body_must_be_a_select() -> None:
    err = _reject("WITH c AS (INSERT INTO t VALUES (1)) SELECT c.v FROM c")
    assert err.code is ErrorCode.UNSUPPORTED_SQL


# ---------------------------------------------------------------------------
# named arguments — SHAPE only
# ---------------------------------------------------------------------------
#
# Which options exist, and what they accept, is a property of the installed
# ffmpeg and belongs to lower + the registry (tests/test_lower.py). Resolve
# only knows that a named argument trails the positional ones and appears once.


def test_named_arguments_resolve() -> None:
    """The resolver accepts them structurally: `sigma` is not an alias, not a
    column, and not checked against anything here."""
    projection = _projection("SELECT gblur(a.video[1], sigma => 5) FROM input('x.mp4') a")
    assert isinstance(projection, exp.Anonymous)
    kwarg = projection.expressions[1]
    assert isinstance(kwarg, exp.Kwarg)
    assert kwarg.this.name == "sigma"


def test_named_argument_names_keep_their_case() -> None:
    """Unquoted identifiers fold lowercase in Postgres, but an option name is
    an ffmpeg AVOption, not an identifier: gblur's sigmaV must survive."""
    from ffrwd.parser import kwarg_name

    projection = _projection("SELECT gblur(a.video[1], sigmaV => 5) FROM input('x.mp4') a")
    assert isinstance(projection, exp.Anonymous)
    kwarg = projection.expressions[1]
    assert isinstance(kwarg, exp.Kwarg)
    assert kwarg_name(kwarg) == "sigmaV"


def test_named_arguments_may_be_nested_and_repeated_across_calls() -> None:
    _resolve(
        "SELECT gblur(unsharp(a.video[1], lx => 7), sigma => 2), gblur(a.video[1], sigma => 3) "
        "FROM input('x.mp4') a"
    )


def test_a_positional_argument_after_a_named_one_is_rejected() -> None:
    err = _reject("SELECT blur(a.video[1], planes => 1, 5) FROM input('x.mp4') a")
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "must come before named arguments" in err.message


def test_a_duplicate_named_argument_is_rejected() -> None:
    err = _reject("SELECT blur(a.video[1], 5, planes => 1, planes => 2) FROM input('x.mp4') a")
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "duplicate named argument 'planes'" in err.message


def test_a_duplicate_named_argument_is_anchored_on_the_second_one() -> None:
    err = _reject(
        "SELECT blur(a.video[1], 5,\n  planes => 1,\n  planes => 2)\nFROM input('x.mp4') a"
    )
    assert err.line == 3


def test_named_arguments_are_checked_inside_a_cte_and_a_union() -> None:
    for sql in (
        "WITH c AS (SELECT blur(a.video[1], planes => 1, 5) AS f FROM input('x.mp4') a) "
        "SELECT c.f FROM c",
        "SELECT a.video[1] FROM input('x.mp4') a UNION ALL "
        "SELECT blur(b.video[1], planes => 1, 5) FROM input('y.mp4') b",
    ):
        assert _reject(sql).code is ErrorCode.UNSUPPORTED_SQL


def test_a_named_argument_does_not_smuggle_in_a_column() -> None:
    """The value goes through the same column whitelist everything else does."""
    err = _reject("SELECT gblur(a.video[1], sigma => b.video[1]) FROM input('x.mp4') a")
    assert err.code is ErrorCode.UNKNOWN_ALIAS


def test_overlay_cannot_take_named_arguments() -> None:
    """Postgres has a builtin OVERLAY(x PLACING y FROM n FOR m) and sqlglot
    parses the stdlib's overlay() with that grammar, so `=>` inside it does not
    even tokenize as a named argument -- a documented dead end, not a silent
    mis-parse."""
    err = _reject(
        "SELECT overlay(a.video[1], b.video[1], 0, 0, eof_action => 'pass') "
        "FROM input('x.mp4') a, input('y.mp4') b"
    )
    assert err.code is ErrorCode.PARSE_ERROR


# ---------------------------------------------------------------------------
# COPY ... TO ... WITH (...)  — the sink wrapper
# ---------------------------------------------------------------------------


def _sink(sql: str) -> RawSink:
    """The single sink of a one-COPY statement (`sinks` is a list)."""
    sinks = _resolve(sql).sinks
    assert len(sinks) == 1
    return sinks[0]


def test_bare_select_has_no_sink() -> None:
    assert _resolve(SINK_QUERY).sinks == []


def test_copy_populates_the_sink() -> None:
    sink = _sink(f"COPY ({SINK_QUERY}) TO 'out.mkv' WITH (video_codec 'libx264', crf 20)")
    assert sink.path == "out.mkv"
    assert [option.name for option in sink.options] == ["video_codec", "crf"]
    # Values stay raw sqlglot nodes here: lower owns the option table.
    assert [option.value.sql() for option in sink.options] == ["'libx264'", "20"]


def test_copy_leaves_the_inner_query_untouched() -> None:
    """The wrapped query resolves exactly like the same query written bare."""
    bare = _resolve(SINK_QUERY)
    wrapped = _resolve(f"COPY ({SINK_QUERY}) TO 'out.mkv'")
    assert wrapped.input_paths == bare.input_paths == ["x.mp4"]
    assert wrapped.sources == bare.sources == {"a": 0}
    assert wrapped.select.sql() == bare.select.sql()


def test_copy_without_with_has_no_options() -> None:
    assert _sink(f"COPY ({SINK_QUERY}) TO 'out.mkv'").options == ()


def test_copy_with_empty_option_list_has_no_options() -> None:
    assert _sink(f"COPY ({SINK_QUERY}) TO 'out.mkv' WITH ()").options == ()


def test_copy_option_names_are_folded_lowercase() -> None:
    """sqlglot drops the quoting of an option name, so "CRF" folds like CRF."""
    for written in ("CRF 20", '"CRF" 20'):
        sink = _sink(f"COPY ({SINK_QUERY}) TO 'out.mkv' WITH ({written})")
        assert [option.name for option in sink.options] == ["crf"]


def test_copy_wraps_a_cte_query() -> None:
    sql = (
        "COPY (WITH c AS (SELECT a.video[1] AS f FROM input('x.mp4') a) "
        "SELECT c.f FROM c) TO 'out.mkv'"
    )
    res = _resolve(sql)
    assert list(res.ctes) == ["c"]
    assert [sink.path for sink in res.sinks] == ["out.mkv"]


def test_copy_wraps_a_union_all() -> None:
    res = _resolve(
        "COPY (SELECT a.video[1] FROM input('x') a UNION ALL "
        "SELECT b.video[1] FROM input('y') b) TO 'out.mkv'"
    )
    assert len(res.branches) == 2
    assert len(res.sinks) == 1


def test_a_sink_carries_its_own_validated_query() -> None:
    """One COPY is one output group, so a sink owns a whole query."""
    res = _resolve(
        "COPY (SELECT a.video[1] FROM input('x') a UNION ALL "
        "SELECT b.video[1] FROM input('y') b) TO 'out.mkv'"
    )
    sink = res.sinks[0]
    assert sink.query is res.select
    assert list(sink.branches) == res.branches


def test_copy_keeps_no_streaming_equivalent_of_the_inner_query() -> None:
    """A COPY wrapper never widens the surface: the inner error still wins."""
    err = _reject(
        "COPY (SELECT a.video[1] FROM input('x.mp4') a GROUP BY a.video[1]) "
        "TO 'out.mkv' WITH (crf 20)"
    )
    assert err.code is ErrorCode.NO_STREAMING_EQUIVALENT


@pytest.mark.parametrize(
    "sql",
    [
        "COPY (SELECT frame FROM input('x.mp4') a) TO 'out.mkv'",
        "COPY (SELECT a.video[1] FROM input('x.mp4')) TO 'out.mkv'",
        "COPY (SELECT scale(a.*, 0.5) FROM input('x.mp4') a) TO 'out.mkv'",
        # A plain input() alias's ORDER BY key is still checked column by
        # column: `a.t` is not a rendition column, so this keeps rejecting.
        "COPY (SELECT a.video[1] FROM input('x.mp4') a ORDER BY a.t LIMIT 1) TO 'out.mkv'",
    ],
)
def test_copy_does_not_relax_inner_validation(sql: str) -> None:
    assert _reject(sql).code in (
        ErrorCode.UNSUPPORTED_SQL,
        ErrorCode.NO_STREAMING_EQUIVALENT,
    )


def test_copy_admits_limit_over_a_plain_input_alias_too() -> None:
    """A COPY wrapper carries the same widened admission as a bare SELECT:
    LIMIT alone over an input() alias resolves, deferring to lower whether
    the probed input actually has renditions to limit."""
    res = _resolve("COPY (SELECT a.video[1] FROM input('x.mp4') a LIMIT 1) TO 'out.mkv'")
    assert isinstance(res.branches[0].args["limit"], exp.Limit)


def test_bad_copy_is_rejected() -> None:
    for sql in (
        # COPY FROM loads data; there is no ffmpeg equivalent.
        "COPY t FROM 'in.csv'",
        "COPY t FROM 'in.csv' WITH (format 'csv')",
        # not a parenthesized query
        "COPY t TO 'out.csv'",
        # more than one target
        f"COPY ({SINK_QUERY}) TO 'a.mkv', 'b.mkv'",
        # non-literal targets
        f"COPY ({SINK_QUERY}) TO STDOUT",
        f"COPY ({SINK_QUERY}) TO x",
        f"COPY ({SINK_QUERY}) TO PROGRAM 'cat'",
        # option with no value at all
        f"COPY ({SINK_QUERY}) TO 'o.mkv' WITH (faststart)",
        f"COPY ({SINK_QUERY}) TO 'o.mkv' WITH (faststart on)",
        # duplicate option, folded name included
        f"COPY ({SINK_QUERY}) TO 'o.mkv' WITH (crf 20, crf 21)",
        f"COPY ({SINK_QUERY}) TO 'o.mkv' WITH (crf 20, CRF 21)",
        # two COPYs re-declaring the same input alias: the flat namespace is
        # script-wide (the multi-sink rejection itself lives in lower).
        f"COPY ({SINK_QUERY}) TO 'a.mkv'; COPY ({SINK_QUERY}) TO 'b.mkv'",
    ):
        assert _reject(sql).code is ErrorCode.UNSUPPORTED_SQL, sql


def test_copy_from_names_the_supported_form() -> None:
    err = _reject("COPY t FROM 'in.csv'")
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert err.hint is not None and "COPY (<query>) TO" in err.hint


# ---------------------------------------------------------------------------
# csv sink classification: FORMAT csv, TO STDOUT
# ---------------------------------------------------------------------------


def test_format_csv_marks_the_sink_csv() -> None:
    sink = _sink(f"COPY ({SINK_QUERY}) TO 'out.csv' WITH (format 'csv')")
    assert sink.is_csv is True
    assert sink.path == "out.csv"


def test_format_csv_bare_word_also_marks_the_sink_csv() -> None:
    """``format csv`` (unquoted, a bare Var under sqlglot) folds the same way
    a quoted string does -- both spellings are stock Postgres COPY syntax."""
    sink = _sink(f"COPY ({SINK_QUERY}) TO 'out.csv' WITH (format csv)")
    assert sink.is_csv is True


def test_format_csv_is_case_insensitive_unquoted() -> None:
    sink = _sink(f"COPY ({SINK_QUERY}) TO 'out.csv' WITH (format CSV)")
    assert sink.is_csv is True


def test_media_copy_with_a_non_csv_format_is_not_csv() -> None:
    """``format`` already means "container format" for a media COPY (an
    existing SINK_OPTIONS entry) -- only the literal value 'csv' flips it."""
    sink = _sink(f"COPY ({SINK_QUERY}) TO 'out.mp4' WITH (format 'mp4')")
    assert sink.is_csv is False


def test_copy_without_format_option_is_not_csv() -> None:
    sink = _sink(f"COPY ({SINK_QUERY}) TO 'out.mkv'")
    assert sink.is_csv is False


def test_to_stdout_is_legal_for_a_csv_sink() -> None:
    sink = _sink(f"COPY ({SINK_QUERY}) TO STDOUT WITH (format 'csv')")
    assert sink.is_csv is True
    assert sink.path is None


def test_to_stdout_lowercase_is_also_accepted() -> None:
    sink = _sink(f"COPY ({SINK_QUERY}) TO stdout WITH (format 'csv')")
    assert sink.path is None


def test_to_stdout_without_format_csv_is_still_rejected() -> None:
    """A media COPY may not target STDOUT -- the TO STDOUT carve-out is
    csv-only (see test_bad_copy_is_rejected's plain "TO STDOUT" case, which
    keeps failing the exact same way)."""
    err = _reject(f"COPY ({SINK_QUERY}) TO STDOUT WITH (video_codec 'libx264')")
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "STDOUT" in (err.hint or "") or "file path" in err.message


def test_to_a_named_identifier_with_format_csv_is_still_rejected() -> None:
    """Only the literal word STDOUT is special; anything else unquoted is
    still not a path, csv or not."""
    err = _reject(f"COPY ({SINK_QUERY}) TO x WITH (format 'csv')")
    assert err.code is ErrorCode.UNSUPPORTED_SQL


def test_header_option_is_legal_on_a_csv_sink() -> None:
    sink = _sink(f"COPY ({SINK_QUERY}) TO STDOUT WITH (format 'csv', header true)")
    assert [option.name for option in sink.options] == ["format", "header"]


def test_duplicate_option_is_anchored_on_the_second_one() -> None:
    err = _reject(
        f"COPY ({SINK_QUERY})\nTO 'out.mkv' WITH (\n  crf 20,\n  crf 21\n)"
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "duplicate sink option 'crf'" in err.message
    assert err.line == 4


def test_copy_shapes_sqlglot_itself_refuses() -> None:
    for sql in (
        # sqlglot refuses a CTE in front of COPY outright ...
        f"WITH c AS ({SINK_QUERY}) COPY (SELECT c.v FROM c) TO 'o.mkv'",
        # ... and a nested COPY, and a parenthesized one.
        f"COPY (COPY ({SINK_QUERY}) TO 'a.mkv') TO 'b.mkv'",
        f"(COPY ({SINK_QUERY}) TO 'o.mkv')",
        # a negative or computed option value is not COPY syntax at all
        f"COPY ({SINK_QUERY}) TO 'o.mkv' WITH (crf -3)",
        f"COPY ({SINK_QUERY}) TO 'o.mkv' WITH (crf 20 + 1)",
    ):
        assert _reject(sql).code is ErrorCode.PARSE_ERROR, sql


# ---------------------------------------------------------------------------
# scripts + CREATE VIEW
# ---------------------------------------------------------------------------
#
# A script is `CREATE VIEW name AS <query>;`* followed by `COPY ...;`+. The
# parser accepts any number of COPYs; lowering is still single-sink, and the
# TEMPORARY rejection of a second one lives in `ffrwd.lower`.

VIEW_SCRIPT = """CREATE VIEW master AS
  SELECT scale(a.video[1], 1280, -2) AS v FROM input('film.mkv') a;

COPY (SELECT master.v FROM master) TO 'out.mp4' WITH (crf 20);
"""


def test_a_script_parses_into_a_block() -> None:
    """VERIFIED (sqlglot 30.17): parse_one wraps a multi-statement string in a
    Block; a single statement, trailing semicolon and all, is returned bare."""
    assert isinstance(parse(VIEW_SCRIPT), exp.Block)
    assert isinstance(parse(SINK_QUERY + ";"), exp.Select)
    assert isinstance(parse(SINK_QUERY + ";  \n"), exp.Select)


def test_an_extra_semicolon_is_not_a_statement() -> None:
    """`a;;` parses into a Block whose second entry is a literal None."""
    res = _resolve(SINK_QUERY + ";;")
    assert res.sinks == []
    assert isinstance(res.select, exp.Select)


def test_view_script_resolves() -> None:
    res = _resolve(VIEW_SCRIPT)
    assert list(res.views) == ["master"]
    assert [sink.path for sink in res.sinks] == ["out.mp4"]
    assert res.input_paths == ["film.mkv"]
    assert res.sources == {"a": 0}


def test_a_view_is_bound_exactly_like_a_cte() -> None:
    """`ctes` is the flat, ordered binding table lower walks; a view is one."""
    res = _resolve(VIEW_SCRIPT)
    assert list(res.ctes) == ["master"]
    assert res.ctes["master"] is res.views["master"]


def test_a_view_may_reference_an_earlier_view() -> None:
    res = _resolve(
        "CREATE VIEW one AS SELECT a.video[1] AS v FROM input('x.mp4') a;\n"
        "CREATE VIEW two AS SELECT scale(one.v, 0.5) AS v FROM one;\n"
        "COPY (SELECT two.v FROM two) TO 'out.mp4';"
    )
    assert list(res.views) == ["one", "two"]
    assert list(res.ctes) == ["one", "two"]


def test_a_view_may_not_reference_a_later_view() -> None:
    err = _reject(
        "CREATE VIEW one AS SELECT two.v AS v FROM two;\n"
        "CREATE VIEW two AS SELECT a.video[1] AS v FROM input('x.mp4') a;\n"
        "COPY (SELECT one.v FROM one) TO 'out.mp4';"
    )
    assert err.code is ErrorCode.UNKNOWN_ALIAS


def test_a_view_body_may_have_its_own_with() -> None:
    """A view BODY is a whole statement, so the nested-WITH rejection (which
    is CTE-body-only) does not apply to it. Its CTEs are hoisted into the flat
    binding table AHEAD of the view, which is the order lower needs."""
    res = _resolve(
        "CREATE VIEW v AS WITH c AS (SELECT a.video[1] AS f FROM input('x.mp4') a) "
        "SELECT c.f AS v FROM c;\n"
        "COPY (SELECT v.v FROM v) TO 'out.mp4';"
    )
    assert list(res.ctes) == ["c", "v"]
    assert list(res.views) == ["v"]


def test_a_cte_body_still_may_not_have_its_own_with() -> None:
    err = _reject(
        "CREATE VIEW v AS WITH c AS (WITH d AS (SELECT a.video[1] AS f "
        "FROM input('x.mp4') a) SELECT d.f AS f FROM d) SELECT c.f AS v FROM c;\n"
        "COPY (SELECT v.v FROM v) TO 'out.mp4';"
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "nested WITH" in err.message


def test_a_view_body_may_be_a_union_all() -> None:
    res = _resolve(
        "CREATE VIEW v AS SELECT a.video[1] AS f FROM input('x') a UNION ALL "
        "SELECT b.video[1] AS f FROM input('y') b;\n"
        "COPY (SELECT v.f FROM v) TO 'out.mp4';"
    )
    assert isinstance(res.views["v"], exp.Union)
    assert len(union_branches(res.views["v"])) == 2


def test_a_copy_may_still_carry_its_own_with_in_a_script() -> None:
    res = _resolve(
        "CREATE VIEW v AS SELECT a.video[1] AS f FROM input('x.mp4') a;\n"
        "COPY (WITH c AS (SELECT v.f AS g FROM v) SELECT c.g FROM c) TO 'out.mp4';"
    )
    assert list(res.ctes) == ["v", "c"]


def test_several_copies_resolve_cleanly() -> None:
    """Resolve is already multi-sink; only lowering is not."""
    res = _resolve(
        "CREATE VIEW m AS SELECT a.video[1] AS v FROM input('film.mkv') a;\n"
        "COPY (SELECT scale(m.v, 1280, -2) FROM m) TO '720.mp4';\n"
        "COPY (SELECT scale(m.v, 640, -2) FROM m) TO '360.mp4' WITH (crf 30);"
    )
    assert [sink.path for sink in res.sinks] == ["720.mp4", "360.mp4"]
    assert [len(sink.branches) for sink in res.sinks] == [1, 1]
    assert [option.name for option in res.sinks[1].options] == ["crf"]
    # `select`/`branches` stay the FIRST sink's.
    assert res.select is res.sinks[0].query


def test_a_view_must_precede_every_copy() -> None:
    err = _reject(
        "COPY (SELECT a.video[1] FROM input('x.mp4') a) TO 'out.mp4';\n"
        "CREATE VIEW v AS SELECT b.video[1] AS f FROM input('y.mp4') b;"
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "may not follow a COPY" in err.message
    assert err.line == 2


def test_a_bare_select_in_a_script_is_rejected() -> None:
    err = _reject(
        "CREATE VIEW v AS SELECT a.video[1] AS f FROM input('x.mp4') a;\n"
        "SELECT v.f FROM v;"
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert err.hint is not None and "COPY (<query>) TO" in err.hint


def test_a_script_with_no_copy_is_rejected() -> None:
    err = _reject("CREATE VIEW v AS SELECT a.video[1] AS f FROM input('x.mp4') a;")
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "must write its output with COPY" in err.message


def test_an_unused_view_is_rejected_at_its_create() -> None:
    err = _reject(
        "CREATE VIEW used AS SELECT a.video[1] AS f FROM input('x.mp4') a;\n"
        "CREATE VIEW spare AS SELECT b.video[1] AS f FROM input('y.mp4') b;\n"
        "COPY (SELECT used.f FROM used) TO 'out.mp4';"
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "view 'spare' is never used" in err.message
    assert err.line == 2


def test_a_view_used_only_by_another_view_counts_as_used() -> None:
    _resolve(
        "CREATE VIEW one AS SELECT a.video[1] AS v FROM input('x.mp4') a;\n"
        "CREATE VIEW two AS SELECT scale(one.v, 0.5) AS v FROM one;\n"
        "COPY (SELECT two.v FROM two) TO 'out.mp4';"
    )


def test_view_names_share_the_flat_namespace() -> None:
    for sql in (
        # view vs view
        "CREATE VIEW v AS SELECT a.video[1] AS f FROM input('x.mp4') a;\n"
        "CREATE VIEW v AS SELECT b.video[1] AS f FROM input('y.mp4') b;\n"
        "COPY (SELECT v.f FROM v) TO 'out.mp4';",
        # view vs an input alias declared inside it
        "CREATE VIEW a AS SELECT a.video[1] AS f FROM input('x.mp4') a;\n"
        "COPY (SELECT a.f FROM a) TO 'out.mp4';",
        # view vs a CTE of its own body
        "CREATE VIEW c AS WITH c AS (SELECT a.video[1] AS f FROM input('x.mp4') a) "
        "SELECT c.f AS f FROM c;\n"
        "COPY (SELECT c.f FROM c) TO 'out.mp4';",
        # view vs a CTE of a later COPY
        "CREATE VIEW v AS SELECT a.video[1] AS f FROM input('x.mp4') a;\n"
        "COPY (WITH v AS (SELECT b.video[1] AS f FROM input('y.mp4') b) "
        "SELECT v.f FROM v) TO 'out.mp4';",
        # view vs a later input alias
        "CREATE VIEW b AS SELECT a.video[1] AS f FROM input('x.mp4') a;\n"
        "COPY (SELECT b.video[1] FROM b, input('y.mp4') b) TO 'out.mp4';",
    ):
        err = _reject(sql)
        assert err.code is ErrorCode.UNSUPPORTED_SQL, sql


def test_ffmpeg_is_reserved_as_a_view_name() -> None:
    err = _reject(
        "CREATE VIEW ffmpeg AS SELECT a.video[1] AS f FROM input('x.mp4') a;\n"
        "COPY (SELECT ffmpeg.f FROM ffmpeg) TO 'out.mp4';"
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "reserved for the filter namespace" in err.message


def test_view_names_fold_the_postgres_way() -> None:
    res = _resolve(
        "CREATE VIEW Master AS SELECT a.video[1] AS f FROM input('x.mp4') a;\n"
        "COPY (SELECT MASTER.f FROM mAsTeR) TO 'out.mp4';"
    )
    assert list(res.views) == ["master"]


_VIEW_BODY = "SELECT a.video[1] AS f FROM input('x.mp4') a"
_VIEW_COPY = "COPY (SELECT v.f FROM v) TO 'out.mp4';"


def test_rejected_create_variants() -> None:
    variants = {
        "or replace": f"CREATE OR REPLACE VIEW v AS {_VIEW_BODY}",
        "temp": f"CREATE TEMP VIEW v AS {_VIEW_BODY}",
        "temporary": f"CREATE TEMPORARY VIEW v AS {_VIEW_BODY}",
        "materialized": f"CREATE MATERIALIZED VIEW v AS {_VIEW_BODY}",
        "if not exists": f"CREATE VIEW IF NOT EXISTS v AS {_VIEW_BODY}",
        "column list": f"CREATE VIEW v (c1, c2) AS {_VIEW_BODY}",
        "qualified name": f"CREATE VIEW s.v AS {_VIEW_BODY}",
        "view options": f"CREATE VIEW v WITH (security_barrier=true) AS {_VIEW_BODY}",
        # sqlglot cannot parse RECURSIVE VIEW at all and falls back to
        # exp.Command, which is not a statement ffrwd knows either.
        "recursive": f"CREATE RECURSIVE VIEW v (c) AS {_VIEW_BODY}",
        "create table": f"CREATE TABLE v AS {_VIEW_BODY}",
        "drop view": "DROP VIEW v",
        "alter view": "ALTER VIEW v RENAME TO w",
    }
    for label, create in variants.items():
        err = _reject(f"{create};\n{_VIEW_COPY}")
        assert err.code is ErrorCode.UNSUPPORTED_SQL, label
        assert err.line == 1, label


def test_a_bad_view_body_is_rejected_like_any_other_query() -> None:
    """A view never widens the surface."""
    err = _reject(
        "CREATE VIEW v AS SELECT a.video[1] AS f FROM input('x.mp4') a GROUP BY a.video[1];\n"
        f"{_VIEW_COPY}"
    )
    assert err.code is ErrorCode.NO_STREAMING_EQUIVALENT


def test_a_view_may_be_aliased_in_from() -> None:
    """`FROM master m` binds a BRANCH-LOCAL name."""
    res = _resolve(
        f"CREATE VIEW v AS {_VIEW_BODY};\nCOPY (SELECT m.f FROM v m) TO 'out.mp4';"
    )
    assert list(res.views) == ["v"]
    assert [sink.path for sink in res.sinks] == ["out.mp4"]


def test_two_sinks_may_reuse_the_same_view_alias() -> None:
    """The alias is branch-local, so it is not in the flat namespace at all."""
    res = _resolve(
        f"CREATE VIEW v AS {_VIEW_BODY};\n"
        "COPY (SELECT m.f FROM v m) TO 'a.mp4';\n"
        "COPY (SELECT m.f FROM v m) TO 'b.mp4';"
    )
    assert [sink.path for sink in res.sinks] == ["a.mp4", "b.mp4"]


def test_a_view_alias_may_not_shadow_a_flat_namespace_name() -> None:
    err = _reject(
        f"CREATE VIEW v AS {_VIEW_BODY};\n"
        "COPY (SELECT a.f FROM v a, input('y.mp4') a) TO 'out.mp4';"
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "duplicate name 'a'" in err.message


def test_a_view_alias_may_not_be_the_filter_namespace() -> None:
    err = _reject(
        f"CREATE VIEW v AS {_VIEW_BODY};\n"
        "COPY (SELECT ffmpeg.f FROM v ffmpeg) TO 'out.mp4';"
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "reserved" in err.message


def test_a_cte_may_be_aliased_too() -> None:
    """The relaxation is on the binding, not on how it was defined."""
    res = _resolve(
        "WITH c AS (SELECT a.video[1] FROM input('x') a) SELECT z.video[1] FROM c z"
    )
    assert list(res.ctes) == ["c"]


def test_an_aliased_name_may_not_collide_with_another_from_entry() -> None:
    err = _reject(
        "WITH c AS (SELECT a.video[1] FROM input('x') a), "
        "d AS (SELECT b.video[1] FROM input('y') b) "
        "SELECT z.video[1] FROM c z, d z"
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "duplicate name 'z'" in err.message


# ---------------------------------------------------------------------------
# track rows: FROM unnest(<input>.<type>) alias
# ---------------------------------------------------------------------------


def test_unnest_binds_a_track_row_table() -> None:
    res = _resolve(
        "SELECT t FROM input('f.mkv') f, unnest(f.audio) t "
        "WHERE t.tags.language = 'eng'"
    )
    assert list(res.track_rows) == ["t"]
    rows = res.track_rows["t"]
    assert (rows.alias, rows.source, rows.column) == ("t", "f", "audio")
    # A row table takes no `-i` of its own: the tracks belong to the input.
    assert res.sources == {"f": 0}
    assert res.input_paths == ["f.mkv"]


def test_every_stream_array_unnests() -> None:
    for column in ("video", "audio", "subtitle", "data"):
        res = _resolve(
            f"SELECT t FROM input('f.mkv') f, unnest(f.{column}) t"
        )
        assert res.track_rows["t"].column == column, column


def test_unnest_accepts_the_as_spelling_and_folds_the_alias() -> None:
    res = _resolve("SELECT T FROM input('f.mkv') f, unnest(f.AUDIO) AS T")
    assert list(res.track_rows) == ["t"]
    assert res.track_rows["t"].column == "audio"


def test_two_unnests_of_one_input_are_two_row_tables() -> None:
    res = _resolve(
        "SELECT a, b FROM input('f.mkv') f, "
        "unnest(f.audio) a, unnest(f.video) b"
    )
    assert [(name, rows.column) for name, rows in res.track_rows.items()] == [
        ("a", "audio"),
        ("b", "video"),
    ]


def test_unnest_binds_inside_a_cte_body_and_a_union_branch() -> None:
    res = _resolve(
        "WITH x AS (SELECT t AS a FROM input('f.mkv') f, unnest(f.audio) t) "
        "SELECT x.a FROM x"
    )
    assert res.track_rows["t"].source == "f"
    res = _resolve(
        "SELECT t FROM input('a.mkv') f, unnest(f.audio) t "
        "UNION ALL "
        "SELECT u FROM input('b.mkv') g, unnest(g.audio) u"
    )
    assert sorted(res.track_rows) == ["t", "u"]


def test_a_row_alias_shares_the_one_flat_namespace() -> None:
    err = _reject(
        "SELECT t FROM input('f.mkv') f, unnest(f.audio) t, unnest(f.video) t"
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "duplicate name 't'" in err.message
    err = _reject("SELECT t.tags.language FROM input('f.mkv') t, unnest(t.audio) t")
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "duplicate name 't'" in err.message


def test_unnest_requires_an_alias() -> None:
    err = _reject("SELECT t FROM input('f.mkv') f, unnest(f.audio)")
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "requires an alias" in err.message


def test_unnest_rejects_a_column_alias_list() -> None:
    err = _reject("SELECT t FROM input('f.mkv') f, unnest(f.audio) t(x)")
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "column aliases" in err.message


def test_unnest_rejects_with_ordinality() -> None:
    err = _reject(
        "SELECT t FROM input('f.mkv') f, unnest(f.audio) WITH ORDINALITY t"
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "ORDINALITY" in err.message
    assert err.hint is not None and "index" in err.hint


@pytest.mark.parametrize(
    "argument",
    ["f.audio[1]", "f.video[1]", "f.t", "'audio'", "f.*", "unnest(f.audio)"],
)
def test_unnest_takes_a_bare_stream_array_and_nothing_else(argument: str) -> None:
    err = _reject(f"SELECT t FROM input('f.mkv') f, unnest({argument}) t")
    assert err.code is ErrorCode.UNSUPPORTED_SQL


def test_unnest_rejects_more_than_one_argument() -> None:
    err = _reject(
        "SELECT t FROM input('f.mkv') f, unnest(f.audio, f.video) t"
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "exactly one array column" in err.message


def test_unnest_only_sees_comma_sources_written_before_it() -> None:
    # Postgres scopes an implicit-LATERAL function call to the FROM items
    # written before it, so this genuinely does not see `f`.
    err = _reject("SELECT t FROM unnest(f.audio) t, input('f.mkv') f")
    assert err.code is ErrorCode.UNKNOWN_ALIAS
    assert "unknown alias 'f'" in err.message


def test_unnest_needs_an_input_not_a_cte_or_a_generated_source() -> None:
    err = _reject(
        "WITH c AS (SELECT a.audio AS tracks FROM input('f.mkv') a) "
        "SELECT t FROM c, unnest(c.tracks) t"
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "only an input's array column can be unnested" in err.message
    err = _reject(
        "SELECT t FROM ffmpeg.anullsrc(duration => 2) s, unnest(s.audio) t"
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "only an input's array column can be unnested" in err.message


# -- JOIN between track-row tables ----------------------


def _joined(join: str = "JOIN", on: str = "ON a.tags.language = b.tags.language") -> str:
    return (
        "SELECT a FROM input('a.mkv') f, input('b.mkv') g, "
        f"unnest(f.audio) a {join} unnest(g.audio) b {on}"
    )


def test_admitted_join_forms_between_unnest_tables() -> None:
    """FULL arrives with AND without `kind='OUTER'`; both are the same join."""
    for join in (
        'JOIN',
        'INNER JOIN',
        'LEFT JOIN',
        'LEFT OUTER JOIN',
        'FULL JOIN',
        'FULL OUTER JOIN',
    ):
        assert sorted(_resolve(_joined(join)).track_rows) == ["a", "b"], join


def test_from_entries_reports_each_items_join_kind() -> None:
    select = parse(_joined("FULL OUTER JOIN"))
    assert isinstance(select, exp.Select)
    entries = from_entries(select)
    assert [None if join is None else join.kind for _, join in entries] == [
        None,  # `FROM input('a.mkv') f` is not attached by anything
        "cross",  # a comma source
        "cross",  # `, unnest(f.audio) a`
        "full",
    ]


def test_a_join_may_match_on_several_keys_and_on_a_literal() -> None:
    sql = _joined(
        on="ON a.tags.language = b.tags.language AND a.channels = b.channels "
        "AND b.codec != 'ac3'"
    )
    assert sorted(_resolve(sql).track_rows) == ["a", "b"]


def test_a_join_on_columns_of_different_types_is_rejected() -> None:
    err = _reject(_joined(on="ON a.tags.language = b.channels"))
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "so they can never match" in err.message


def test_a_join_on_may_use_in_and_not_in() -> None:
    sql = _joined(
        on="ON a.tags.language IN ('eng', 'fra') AND b.channels NOT IN (0)"
    )
    assert sorted(_resolve(sql).track_rows) == ["a", "b"]


def test_a_join_cannot_match_on_the_row_stream() -> None:
    err = _reject(_joined(on="ON a = b"))
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "is a stream, not a value to compare" in err.message


def test_a_join_with_no_on_is_the_comma_cross_join() -> None:
    """VERIFIED (sqlglot 30.17): `a JOIN b` with no ON parses to an exp.Join
    carrying nothing but `this` -- byte for byte what a comma source produces.
    The two are indistinguishable in the AST, so a bare JOIN between two row
    tables IS the bounded cross join, not a rejection we could even spell."""
    select = parse(_joined(on=""))
    assert isinstance(select, exp.Select)
    assert [None if join is None else join.kind for _, join in from_entries(select)][
        -1
    ] == "cross"
    assert sorted(_resolve(_joined(on="")).track_rows) == ["a", "b"]


def test_an_on_predicate_may_not_name_a_non_row_alias() -> None:
    err = _reject(_joined(on="ON a.tags.language = f.t"))
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "unknown column 'f.t'" in err.message
    assert "not a track-row table" in (err.hint or "")


def test_unsupported_on_shapes_are_rejected() -> None:
    err = _reject(_joined(on="ON a.tags.language LIKE b.tags.language"))
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "unsupported ON predicate" in err.message


@pytest.mark.parametrize(
    ("join", "on", "message"),
    [
        ("RIGHT JOIN", "ON a.tags.language = b.tags.language", "RIGHT JOIN is not supported"),
        ("CROSS JOIN", "", "CROSS JOIN is not supported"),
        ("NATURAL JOIN", "", "this JOIN form is not supported"),
        ("JOIN", "USING (language)", "USING is not supported"),
    ],
)
def test_join_forms_outside_the_admitted_set(join: str, on: str, message: str) -> None:
    err = _reject(_joined(join, on))
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert message in err.message


def test_join_is_still_rejected_between_stream_level_sources() -> None:
    # input-level FROM stays a comma cross-join.
    err = _reject(
        "SELECT f.video[1] FROM input('a.mkv') f JOIN input('b.mkv') g ON f.t = g.t"
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "between row tables only" in err.message


def test_join_is_rejected_when_its_left_side_is_not_a_row_table() -> None:
    err = _reject(
        "SELECT a FROM input('a.mkv') f, input('b.mkv') g "
        "JOIN unnest(g.audio) a ON a.tags.language = 'eng'"
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "between row tables only" in err.message


# -- row columns ------------------------------------------------------------


def test_every_audio_row_column_resolves() -> None:
    columns = [
        "index", "tags.language", "tags.title", "disposition.default",
        "disposition.forced", "codec", "channels",
        "channel_layout", "sample_rate", "bitrate", "duration",
    ]
    where = " AND ".join(f"t.{column} IS NOT NULL" for column in columns)
    sql = f"SELECT t FROM input('f.mkv') f, unnest(f.audio) t WHERE {where}"
    assert _resolve(sql).track_rows["t"].column == "audio"


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT t.track FROM input('f.mkv') f, unnest(f.audio) t",
        "SELECT t FROM input('f.mkv') f, unnest(f.audio) t WHERE t.track IS NOT NULL",
        "SELECT t FROM input('f.mkv') f, unnest(f.audio) t ORDER BY t.track",
    ],
)
def test_the_track_column_left_the_language(sql: str) -> None:
    err = _reject(sql)
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert err.message == "'t.track' is not a column"
    assert err.hint == "the row is the stream: use 't'"


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT f.frame FROM input('x.mp4') f",
        "COPY (SELECT t FROM input('x.mp4') f, unnest(f.audio) t) "
        "TO (f.frame || '.mka')",
    ],
)
def test_the_frame_column_left_the_language(sql: str) -> None:
    err = _reject(sql)
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert err.message == "'f.frame' is not a column"
    assert err.hint == "the first video stream is 'f.video[1]'"


def test_a_bare_row_alias_is_a_stream_not_a_where_value() -> None:
    err = _reject(
        "SELECT t FROM input('f.mkv') f, unnest(f.audio) t WHERE t IS NOT NULL"
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "'t' is a stream, not a value to compare" in err.message


def test_every_video_row_column_resolves() -> None:
    columns = ["width", "height", "fps", "color_transfer", "bitrate", "duration"]
    where = " AND ".join(f"t.{column} IS NOT NULL" for column in columns)
    sql = f"SELECT t FROM input('f.mkv') f, unnest(f.video) t WHERE {where}"
    assert _resolve(sql).track_rows["t"].column == "video"


def test_a_row_column_is_checked_against_its_stream_types_schema() -> None:
    # `channels` is an audio column; a video row has no such thing.
    err = _reject(
        "SELECT t FROM input('f.mkv') f, unnest(f.video) t WHERE t.channels = 2"
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "unknown column 't.channels'" in err.message
    assert err.hint is not None and "video track rows expose" in err.hint
    # ... and a subtitle row carries only the five common ones.
    err = _reject(
        "SELECT s FROM input('f.mkv') f, unnest(f.subtitle) s WHERE s.width = 1"
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "unknown column 's.width'" in err.message


def test_an_unknown_row_column_in_the_select_list_is_rejected() -> None:
    err = _reject("SELECT t.nope FROM input('f.mkv') f, unnest(f.audio) t")
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "unknown column 't.nope'" in err.message


# -- WHERE over row columns -------------------------------------------------


def test_row_predicates_are_admitted() -> None:
    for predicate in (
        "t.tags.language = 'eng'",
        "'eng' = t.tags.language",
        "t.tags.language != 'eng'",
        "t.channels > 2",
        "t.channels >= 2",
        "t.channels < 6",
        "t.channels <= 6",
        "t.bitrate BETWEEN 1000 AND 2000",
        "t.duration <= 2.5",
        "t.bitrate >= -1",
        "t.tags.language IS NULL",
        "t.tags.title IS NOT NULL",
        "NOT (t.tags.language = 'eng')",
        "t.tags.language = 'eng' AND t.channel_layout = 'stereo'",
        "(t.tags.language = 'eng' OR t.tags.language IS NULL) AND t.channels = 2",
        "t.tags.language IN ('eng', 'fra')",
        "t.tags.language NOT IN ('eng', 'fra')",
        "t.channels IN (2, 6)",
        "t.disposition.forced IN (true, false)",
        "t.tags.language IN ('eng') AND t.channels = 2",
    ):
        _resolve(
            f"SELECT t FROM input('f.mkv') f, unnest(f.audio) t WHERE {predicate}"
        )


@pytest.mark.parametrize(
    "predicate",
    [
        "t.tags.language LIKE 'e%'",
        "t.channels BETWEEN SYMMETRIC 2 AND 6",
        "t.channels = t.sample_rate",
        "t.tags.language IS TRUE",
    ],
)
def test_unsupported_row_predicates_are_rejected(predicate: str) -> None:
    err = _reject(
        f"SELECT t FROM input('f.mkv') f, unnest(f.audio) t WHERE {predicate}"
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL


def test_a_row_predicate_is_typed_against_the_static_column_type() -> None:
    err = _reject(
        "SELECT t FROM input('f.mkv') f, unnest(f.audio) t "
        "WHERE t.channels = 'stereo'"
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "'t.channels' is number" in err.message
    err = _reject(
        "SELECT t FROM input('f.mkv') f, unnest(f.audio) t WHERE t.tags.language = 5"
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "'t.tags.language' is text" in err.message


def test_an_in_list_types_each_element_and_anchors_on_the_offending_one() -> None:
    # `2` and `6` are valid; `'x'` is the one that breaks the list, and the
    # rejection has to point at IT, not at the list or its first element.
    sql = (
        "SELECT t FROM input('f.mkv') f, unnest(f.audio) t "
        "WHERE t.channels IN (2, 'x', 6)"
    )
    err = _reject(sql)
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "'t.channels' is number" in err.message
    assert "a string" in err.message
    assert err.line == 1
    assert err.col == sql.index("'x'") + 1


def test_in_with_an_empty_list_is_still_rejected() -> None:
    err = _reject(
        "SELECT t FROM input('f.mkv') f, unnest(f.audio) t WHERE t.tags.language IN ()"
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL


def test_in_select_over_a_row_column_still_has_no_streaming_equivalent() -> None:
    # The subquery form is still a subquery predicate wherever it is spelled --
    # desugaring only ever touches the literal-list form.
    err = _reject(
        "SELECT t FROM input('f.mkv') f, unnest(f.audio) t "
        "WHERE t.tags.language IN (SELECT 'eng')"
    )
    assert err.code is ErrorCode.NO_STREAMING_EQUIVALENT


def test_the_rows_md_worked_example_compiles() -> None:
    res = _resolve(
        "SELECT t, CASE WHEN t.tags.language IN ('en', 'english') THEN 'eng' "
        "ELSE t.tags.language END AS language "
        "FROM input('f.mkv') f, unnest(f.audio) t"
    )
    assert res.track_rows["t"].source == "f"


def test_a_row_predicate_may_not_compare_the_stream_column() -> None:
    err = _reject(
        "SELECT t FROM input('f.mkv') f, unnest(f.audio) t WHERE t = 1"
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "is a stream, not a value to compare" in err.message


def test_a_time_window_and_a_row_predicate_coexist_as_separate_conjuncts() -> None:
    res = _resolve(
        "SELECT t FROM input('f.mkv') f, unnest(f.audio) t "
        "WHERE f.t BETWEEN 1 AND 2 AND t.tags.language = 'eng'"
    )
    assert res.track_rows["t"].source == "f"


def test_a_conjunct_may_not_mix_a_row_column_with_another_alias() -> None:
    err = _reject(
        "SELECT t FROM input('f.mkv') f, unnest(f.audio) t "
        "WHERE t.tags.language = 'eng' OR f.t >= 1"
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "cannot mix track-row columns" in err.message


def test_a_conjunct_may_reference_only_one_row_table() -> None:
    err = _reject(
        "SELECT a FROM input('f.mkv') f, unnest(f.audio) a, unnest(f.video) b "
        "WHERE a.tags.language = b.tags.language"
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "only one track-row table" in err.message


def test_a_time_window_over_an_input_still_rejects_a_non_t_column() -> None:
    # The time half of the WHERE clause is untouched by the row half.
    err = _reject(
        "SELECT t FROM input('f.mkv') f, unnest(f.audio) t WHERE f.video >= 1"
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "only the time column 'f.t' can be filtered" in err.message


# -- ORDER BY: the one carve-out in the streaming rejection -----------------


def test_order_by_is_admitted_over_track_row_columns() -> None:
    res = _resolve(
        "SELECT t FROM input('f.mkv') f, unnest(f.audio) t "
        "ORDER BY t.tags.language, t.channels DESC"
    )
    order = res.branches[0].args["order"]
    assert isinstance(order, exp.Order)
    # sqlglot fills the Postgres NULL defaults in for us: ASC -> NULLS LAST,
    # DESC -> NULLS FIRST. Honoring the flags verbatim IS Postgres semantics.
    assert [bool(o.args.get("desc")) for o in order.expressions] == [False, True]
    assert [bool(o.args.get("nulls_first")) for o in order.expressions] == [False, True]


def test_order_by_and_limit_resolve_over_a_plain_input_alias() -> None:
    """FROM holds only an input() alias, no unnest -- a probe may turn it
    into a rendition row table, which only lower can confirm, so resolve
    admits ORDER BY over its rendition columns plus LIMIT on spec."""
    res = _resolve(
        "SELECT r.video[1] FROM input('ladder.m3u8') r "
        "ORDER BY r.bandwidth DESC LIMIT 3"
    )
    order = res.branches[0].args["order"]
    assert isinstance(order, exp.Order)
    assert isinstance(res.branches[0].args["limit"], exp.Limit)


def test_order_by_a_non_rendition_column_without_unnest_is_still_rejected() -> None:
    """A plain input() alias admits the ORDER BY clause itself (a probe may
    turn it into a rendition row table), but `f.t` is not one of the six
    rendition columns, so the per-key check still refuses it."""
    err = _reject("SELECT f.audio[1] FROM input('f.mkv') f ORDER BY f.t")
    assert err.code is ErrorCode.NO_STREAMING_EQUIVALENT
    assert "only track-row metadata columns can be sorted" in (err.hint or "")


def test_order_by_without_any_row_or_input_source_is_still_rejected() -> None:
    """Neither an unnest(...) nor an input() alias in FROM: the ORDER BY
    clause itself is rejected before any sort key is even inspected."""
    err = _reject("SELECT f.video FROM ffmpeg.testsrc() f ORDER BY f.video")
    assert err.code is ErrorCode.NO_STREAMING_EQUIVALENT
    assert err.hint == "remove the ORDER BY clause"


@pytest.mark.parametrize("key", ["f.t", "1", "t"])
def test_order_by_a_non_row_column_is_rejected_even_in_a_row_query(key: str) -> None:
    err = _reject(
        "SELECT t FROM input('f.mkv') f, unnest(f.audio) t "
        f"ORDER BY {key}"
    )
    assert err.code in (
        ErrorCode.NO_STREAMING_EQUIVALENT,
        ErrorCode.UNSUPPORTED_SQL,
    )


def test_the_other_streaming_rejections_are_untouched_by_the_carve_out() -> None:
    err = _reject("SELECT DISTINCT t FROM input('f.mkv') f, unnest(f.audio) t")
    assert err.code is ErrorCode.NO_STREAMING_EQUIVALENT


# -- LIMIT / OFFSET: co-legal with ORDER BY ---------------------------------


def test_limit_and_offset_are_admitted_over_a_row_table() -> None:
    res = _resolve(
        "SELECT t FROM input('f.mkv') f, unnest(f.audio) t "
        "ORDER BY t.index LIMIT 2 OFFSET 1"
    )
    assert isinstance(res.branches[0].args["limit"], exp.Limit)


def test_limit_is_admitted_in_a_cte_body() -> None:
    _resolve(
        "WITH top AS (SELECT t FROM input('f.mkv') f, unnest(f.video) t "
        "ORDER BY t.width DESC LIMIT 1) SELECT top.t FROM top"
    )


def test_limit_over_a_plain_input_alias_now_resolves() -> None:
    """A probe may turn a plain input() alias into a rendition row table --
    only lower can tell -- so resolve admits LIMIT here on spec; a
    renditionless input is lower's rejection to make, not resolve's."""
    res = _resolve("SELECT f.audio[1] FROM input('f.mkv') f LIMIT 1")
    assert isinstance(res.branches[0].args["limit"], exp.Limit)


def test_limit_without_a_row_or_input_source_keeps_the_streaming_rejection() -> None:
    err = _reject("SELECT f.video[1] FROM ffmpeg.testsrc() f LIMIT 1")
    assert err.code is ErrorCode.NO_STREAMING_EQUIVALENT
    assert err.message == "LIMIT has no streaming equivalent"
    assert "row table" in (err.hint or "")


@pytest.mark.parametrize(
    ("clause", "message"),
    [
        ("LIMIT 0", "LIMIT 0 selects no rows"),
        ("LIMIT t.index", "LIMIT must be a positive integer literal"),
        ("LIMIT 1.5", "LIMIT must be a positive integer literal"),
        ("LIMIT -1", "LIMIT must be a positive integer literal"),
        ("OFFSET t.index", "OFFSET must be a non-negative integer literal"),
        ("LIMIT NULL", "LIMIT must be a positive integer literal"),
    ],
)
def test_row_count_rejections(clause: str, message: str) -> None:
    err = _reject(
        f"SELECT t FROM input('f.mkv') f, unnest(f.audio) t {clause}"
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert err.message == message


def test_an_unset_variable_limit_names_the_variable() -> None:
    sub = substitute(
        "SELECT t FROM input('f.mkv') f, unnest(f.audio) t LIMIT :n", {}
    )
    with pytest.raises(FfrwdError) as excinfo:
        resolve(parse(sub.text, sub.unset))
    err = excinfo.value
    assert err.message == "':n' was not set"
    assert "LIMIT needs its row count" in (err.hint or "")


# -- array elements: ARRAY[<literals>][<subscript>] -------------------------


def test_an_array_element_resolves_as_a_value() -> None:
    _resolve(
        "SELECT t FROM input('f.mkv') f, unnest(f.audio) t "
        "WHERE t.channels = ARRAY[1, 2][t.index]"
    )


@pytest.mark.parametrize(
    ("expr", "message"),
    [
        ("ARRAY[1, 'a'][1]", "the subscripted array mixes number and text elements"),
        ("ARRAY[t.index][1]", "a subscripted array takes literal elements only"),
        ("ARRAY[1, 2]['a']", "an array subscript is a positive integer literal or a row column"),
        ("ARRAY[1, 2][0]", "an array subscript is a positive integer literal or a row column"),
    ],
)
def test_array_element_shape_rejections(expr: str, message: str) -> None:
    err = _reject(
        "SELECT t FROM input('f.mkv') f, unnest(f.audio) t "
        f"WHERE t.channels = {expr}"
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert err.message == message


def test_a_text_column_cannot_subscript_an_array() -> None:
    err = _reject(
        "SELECT t FROM input('f.mkv') f, unnest(f.audio) t "
        "WHERE t.channels = ARRAY[1, 2][t.codec]"
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "an array subscript is a number, but this column is text" in err.message


# ---------------------------------------------------------------------------
# subscript metadata accessors: <alias>.<type>[k].<column>
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "spelling",
    ["f.audio[1].tags.language", "(f.audio[1]).tags.language"],
)
def test_both_subscript_metadata_spellings_are_admitted(spelling: str) -> None:
    # The bracket-dot and the strictly-Postgres paren-dot forms are the same
    # shape, VERIFIED under sqlglot 30.17 (module docstring); either compiles
    # cleanly as a WHERE assertion.
    _resolve(f"SELECT f.audio[1] FROM input('f.mkv') f WHERE {spelling} = 'eng'")


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT f.audio[1].track FROM input('f.mkv') f",
        "SELECT f.audio[1] FROM input('f.mkv') f WHERE f.audio[1].track = 1",
    ],
)
def test_a_subscript_has_no_track_accessor_left(sql: str) -> None:
    err = _reject(sql)
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "'f.audio[1].track' is not a column" in err.message
    assert err.hint is not None and "use 'f.audio[1]'" in err.hint


def test_a_metadata_accessor_is_rejected_as_a_select_output_in_a_media_query() -> None:
    # this rejection is MEDIA-only -- a bare SELECT (no
    # COPY) is always at least table-capable, and metadata columns are legal
    # there (see test_subscript_metadata_output_is_legal_in_table_mode
    # below). Wrapping in a real media COPY keeps this test on the rejection
    # it means to check.
    for column in (
        "tags.language", "tags.title", "codec", "channels", "channel_layout", "index",
    ):
        err = _reject(
            f"COPY (SELECT f.audio[1].{column} FROM input('f.mkv') f) TO 'out.mp4'"
        )
        assert err.code is ErrorCode.UNSUPPORTED_SQL, column
        assert "is track metadata, not a stream" in err.message


def test_subscript_metadata_output_is_legal_in_table_mode() -> None:
    for column in ("tags.language", "tags.title", "codec", "channels", "channel_layout", "index"):
        # A bare SELECT (no COPY at all) is table-capable unconditionally.
        _resolve(f"SELECT f.audio[1].{column} FROM input('f.mkv') f")
        # So is a csv COPY.
        _resolve(
            f"COPY (SELECT f.audio[1].{column} FROM input('f.mkv') f) "
            "TO STDOUT WITH (format csv)"
        )


def test_bare_array_metadata_access_is_rejected_in_select() -> None:
    err = _reject("SELECT f.audio.language FROM input('f.mkv') f")
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "needs a subscript" in err.message
    assert err.hint is not None and "unnest" in err.hint


def test_bare_array_metadata_access_is_rejected_in_where() -> None:
    err = _reject(
        "SELECT f.audio[1] FROM input('f.mkv') f WHERE f.audio.language = 'eng'"
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "needs a subscript" in err.message


def test_subscript_metadata_predicates_are_admitted() -> None:
    for predicate in (
        "f.audio[1].tags.language = 'eng'",
        "'eng' = f.audio[1].tags.language",
        "f.audio[1].tags.language != 'eng'",
        "f.audio[1].channels > 2",
        "f.audio[1].channels >= 2",
        "f.audio[1].channels < 6",
        "f.audio[1].bitrate BETWEEN 1000 AND 2000",
        "f.audio[1].tags.language IS NULL",
        "f.audio[1].tags.title IS NOT NULL",
        "NOT (f.audio[1].tags.language = 'eng')",
        "f.audio[1].tags.language = 'eng' AND f.audio[2].tags.language = 'fra'",
        "(f.audio[1].tags.language = 'eng' OR f.audio[1].tags.language IS NULL) "
        "AND f.audio[1].channels = 2",
        "f.audio[1].tags.language IN ('eng', 'fra')",
        "f.audio[1].tags.language NOT IN ('eng', 'fra')",
        "f.audio[1].channels IN (2, 6)",
    ):
        _resolve(f"SELECT f.audio[1] FROM input('f.mkv') f WHERE {predicate}")


def test_a_subscript_predicate_is_typed_against_the_static_column_type() -> None:
    err = _reject(
        "SELECT f.audio[1] FROM input('f.mkv') f WHERE f.audio[1].channels = 'stereo'"
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "is number" in err.message
    err = _reject(
        "SELECT f.audio[1] FROM input('f.mkv') f WHERE f.audio[1].tags.language = 5"
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "is text" in err.message


def test_a_conjunct_may_not_mix_a_subscript_accessor_with_the_time_window() -> None:
    err = _reject(
        "SELECT f.audio[1] FROM input('f.mkv') f "
        "WHERE f.audio[1].tags.language = 'eng' OR f.t >= 1"
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "cannot mix subscript metadata accessors" in err.message


def test_a_time_window_and_a_subscript_predicate_coexist_as_separate_conjuncts() -> None:
    _resolve(
        "SELECT f.audio[1] FROM input('f.mkv') f "
        "WHERE f.t BETWEEN 1 AND 2 AND f.audio[1].tags.language = 'eng'"
    )


@pytest.mark.parametrize(
    ("column", "predicate"),
    [
        ("video", "f.video[1].width = 640"),
        ("subtitle", "f.subtitle[1].tags.language = 'eng'"),
        ("data", "f.data[1].index = 1"),
    ],
)
def test_video_subtitle_and_data_subscript_accessors_resolve(
    column: str, predicate: str
) -> None:
    _resolve(f"SELECT f.{column}[1] FROM input('f.mkv') f WHERE {predicate}")


def test_a_subscript_accessor_over_a_cte_is_rejected() -> None:
    err = _reject(
        "WITH c AS (SELECT a.video[1] AS v FROM input('a.mkv') a) "
        "SELECT c.v FROM c WHERE c.v[1].tags.language = 'eng'"
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "needs a row column on one side" in err.message


def test_a_subscript_accessor_over_a_generated_source_is_rejected() -> None:
    err = _reject(
        "SELECT s.video[1] FROM ffmpeg.testsrc(duration => 2) s "
        "WHERE s.video[1].tags.language = 'eng'"
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "is a generated source" in err.message


def test_a_subscript_accessor_over_a_row_alias_is_rejected() -> None:
    # In SELECT position, so the row-language WHERE grammar (which claims
    # anything mentioning a row alias first) is not what intercepts this.
    err = _reject(
        "SELECT t[1].tags.language FROM input('f.mkv') f, unnest(f.audio) t"
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "already a track-row table" in err.message


# ---------------------------------------------------------------------------
# guardrail #7: no panics on user input
# ---------------------------------------------------------------------------


def test_never_raises_anything_but_ffrwd_error() -> None:
    for sql in (
        "SELECT",
        "WITH",
        "WITH c AS () SELECT 1",
        "SELECT a.video[1] FROM input('x') a WHERE a.t BETWEEN AND 2",
        "SELECT scale(a.video[1],) FROM input('x') a",
        "SELECT a.video[1] FROM input('x') a,",
        "select 'x' from input('x')",
        "-- just a comment",
        "/* block */",
        "SELECT a.video[1] FROM input('x') a UNION ALL",
        "\x00\x01",
        "SELECT a.video[ FROM input('x') a",
        "SELECT a.video[] FROM input('x') a",
        "SELECT a.video[[1]] FROM input('x') a",
        "SELECT a.video[1 FROM input('x') a",
        "SELECT a.video[" + "1," * 200 + "1] FROM input('x') a",
        "SELECT " + "a.video[1], " * 200 + "a.audio[1] FROM input('x') a",
        "SELECT " + "f(" * 60 + "a.video[1]" + ")" * 60 + " FROM input('x') a",
    ):
        try:
            _resolve(sql)
        except FfrwdError as err:
            assert err.code is not ErrorCode.INTERNAL, sql
            assert err.line is not None


# ---------------------------------------------------------------------------
# unnest(f.chapters) in FROM, and written row tables
# ---------------------------------------------------------------------------


def test_chapters_binds_a_row_table_shaped_like_a_track_row() -> None:
    res = _resolve(
        "SELECT c.index, c.title, c.start_t, c.end_t "
        "FROM input('f.mkv') f, unnest(f.chapters) c"
    )
    assert list(res.track_rows) == ["c"]
    rows = res.track_rows["c"]
    assert (rows.alias, rows.source, rows.column) == ("c", "f", "chapters")


def test_chapters_accepts_the_as_spelling_and_folds_the_alias() -> None:
    res = _resolve("SELECT C.title FROM input('f.mkv') f, unnest(f.chapters) AS C")
    assert list(res.track_rows) == ["c"]


def test_chapters_requires_an_alias() -> None:
    err = _reject("SELECT 1 FROM input('f.mkv') f, unnest(f.chapters)")
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "requires an alias" in err.message


def test_the_chapters_table_function_is_gone() -> None:
    """`chapters(f)` was removed: it hits the unknown-table-function path,
    with a hint naming the array column that replaced it."""
    err = _reject("SELECT 1 FROM input('f.mkv') f, chapters(f) c")
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "unsupported table function chapters()" in err.message
    assert "unnest(f.chapters) c" in (err.hint or "")


def test_chapters_unnests_only_an_input_alias() -> None:
    err = _reject(f"WITH x AS ({SINK_QUERY}) SELECT 1 FROM x, unnest(x.chapters) c")
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "only an input's array column can be unnested" in err.message

    err = _reject(
        "SELECT 1 FROM input('f.mkv') f, unnest(f.audio) t, unnest(t.chapters) c"
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL

    err = _reject("SELECT 1 FROM input('f.mkv') f, unnest(nope.chapters) c")
    assert err.code is ErrorCode.UNKNOWN_ALIAS


def test_chapters_rows_combine_with_track_rows_like_any_other_array() -> None:
    """No carve-out left: two unnest tables of one input cross join."""
    res = _resolve(
        "SELECT t.tags.language, c.title "
        "FROM input('f.mkv') f, unnest(f.audio) t, unnest(f.chapters) c"
    )
    assert [rows.column for rows in res.track_rows.values()] == ["audio", "chapters"]


def test_two_inputs_chapters_are_two_row_tables() -> None:
    res = _resolve(
        "SELECT c.title, d.title FROM input('f.mkv') f, input('g.mkv') g, "
        "unnest(f.chapters) c, unnest(g.chapters) d"
    )
    assert [rows.source for rows in res.track_rows.values()] == ["f", "g"]


def test_a_chapters_subscript_is_rejected_with_an_unnest_hint() -> None:
    err = _reject("SELECT f.chapters[1].title FROM input('f.mkv') f")
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "'f.chapters' cannot be subscripted" in err.message
    assert "unnest(f.chapters) c" in (err.hint or "")


def test_a_bare_chapters_accessor_names_the_unnest() -> None:
    err = _reject("SELECT f.chapters.title FROM input('f.mkv') f")
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "'f.chapters.title' needs a row" in err.message
    assert "unnest(f.chapters) c" in (err.hint or "")


def test_a_chapters_column_in_a_value_expression_is_rejected() -> None:
    err = _reject("SELECT f.chapters::text FROM input('f.mkv') f")
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "'f.chapters' is an array of chapter records" in err.message
    assert "unnest(f.chapters) c" in (err.hint or "")


def test_chapters_column_list_is_rejected() -> None:
    err = _reject("SELECT 1 FROM input('f.mkv') f, unnest(f.chapters) c(x)")
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "table column aliases are not supported" in err.message


def test_a_chapter_row_is_not_a_stream() -> None:
    err = _reject("SELECT c FROM input('f.mkv') f, unnest(f.chapters) c")
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "'c' is a chapter row, not a stream" in err.message


def test_cues_bind_a_row_table_the_same_way_chapters_do() -> None:
    res = _resolve(
        "SELECT c.index, c.text, c.start_t, c.end_t "
        "FROM input('subs.en.vtt') v, unnest(v.cues) c"
    )
    rows = res.track_rows["c"]
    assert (rows.alias, rows.source, rows.column) == ("c", "v", "cues")


def test_a_cue_row_has_no_title_column() -> None:
    err = _reject("SELECT c.title FROM input('subs.en.vtt') v, unnest(v.cues) c")
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "unknown column 'c.title'" in err.message


def test_a_cue_row_is_not_a_stream() -> None:
    err = _reject("SELECT c FROM input('subs.en.vtt') v, unnest(v.cues) c")
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "'c' is a cue row, not a stream" in err.message


def test_a_cue_row_carries_no_tags() -> None:
    err = _reject("SELECT c.tags.title FROM input('subs.en.vtt') v, unnest(v.cues) c")
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "'c' is a cue row, and a cue carries no tags" in err.message


def test_a_cues_subscript_is_rejected_with_an_unnest_hint() -> None:
    err = _reject("SELECT v.cues[1].text FROM input('subs.en.vtt') v")
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "'v.cues' cannot be subscripted" in err.message
    assert "unnest(v.cues) c" in (err.hint or "")


def test_a_bare_cues_accessor_names_the_unnest() -> None:
    err = _reject("SELECT v.cues.text FROM input('subs.en.vtt') v")
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "'v.cues.text' needs a row" in err.message
    assert "unnest(v.cues) c" in (err.hint or "")


def test_a_lone_cue_record_is_not_a_value() -> None:
    err = _reject(
        "SELECT f.video[1], ROW('Hi', 0, 1)::cue AS x FROM input('f.mkv') f"
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "a cue record is not a value on its own" in err.message
    assert err.hint == "gather records into an array, e.g. ARRAY[STRUCT(...)::cue, ...]"


def test_a_lone_chapter_record_still_names_the_chapters_column() -> None:
    err = _reject(
        "SELECT f.video[1], ROW('Hi', 0, 1)::chapter AS x FROM input('f.mkv') f"
    )
    assert err.hint == (
        "gather records into an array, e.g. ARRAY[STRUCT(...)::chapter, ...] AS chapters"
    )


def test_a_values_cte_is_rejected() -> None:
    err = _reject(
        "COPY (WITH marks(start_t, end_t, title) AS "
        "(VALUES (0, 60, 'Intro'), (60, 300, 'Act One')) "
        f"{SINK_QUERY}) TO 'x.mkv'"
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "a VALUES row table is not supported" in err.message
    assert err.hint == (
        "a row table is written unnest(ARRAY[STRUCT(0 AS start_t, 60 AS "
        "end_t, 'Intro' AS title), STRUCT(60 AS start_t, 300 AS end_t, "
        "'Act One' AS title)]) marks"
    )


def test_a_values_row_table_in_from_is_rejected() -> None:
    err = _reject("SELECT r.w FROM (VALUES (1920, 20), (1280, 22)) AS r(w, q)")
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "a VALUES row table is not supported" in err.message
    assert err.hint == (
        "a row table is written unnest(ARRAY[STRUCT(1920 AS w, 20 AS q), "
        "STRUCT(1280 AS w, 22 AS q)]) r"
    )


def test_ordinary_cte_column_renaming_still_stays_rejected() -> None:
    err = _reject("WITH x(a, b) AS (SELECT 1, 2) SELECT * FROM x")
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "column list, which is not supported" in err.message
    assert err.hint == "name a SELECT CTE's columns with AS inside the SELECT instead"


# ---------------------------------------------------------------------------
# struct row tables: unnest(ARRAY[STRUCT(...), ...]) alias
# ---------------------------------------------------------------------------


def test_struct_row_table_binds_a_row_table_like_values_do() -> None:
    res = _resolve(
        "SELECT r.w FROM input('f.mkv') f, "
        "unnest(ARRAY[STRUCT(1920 AS w, '1080p' AS name), "
        "STRUCT(1280 AS w, '720p' AS name)]) r"
    )
    assert list(res.struct_rows) == ["r"]
    table = res.struct_rows["r"]
    assert table.columns == ("w", "name")
    assert table.types == ("number", "text")
    assert len(table.rows) == 2
    assert "r" in res.row_aliases
    assert "r" not in res.track_rows


def test_struct_row_table_field_order_is_order_free_across_rows() -> None:
    res = _resolve(
        "SELECT r.w FROM input('f.mkv') f, "
        "unnest(ARRAY[STRUCT(1920 AS w, '1080p' AS name), "
        "STRUCT('720p' AS name, 1280 AS w)]) r"
    )
    assert res.struct_rows["r"].columns == ("w", "name")


def test_struct_row_table_field_mismatch_is_rejected() -> None:
    err = _reject(
        "SELECT r.w FROM input('f.mkv') f, "
        "unnest(ARRAY[STRUCT(1920 AS w, '1080p' AS name), STRUCT(1280 AS w)]) r"
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "does not declare the same fields" in err.message
    assert "'name'" in err.message


def test_struct_row_table_empty_array_is_rejected() -> None:
    err = _reject("SELECT f.video[1] FROM input('f.mkv') f, unnest(ARRAY[]) r")
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "empty list of rows" in err.message


def test_struct_row_table_field_rejects_a_stream() -> None:
    err = _reject(
        "SELECT r.v FROM input('f.mkv') f, unnest(ARRAY[STRUCT(f.video[1] AS v)]) r"
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL


def test_struct_row_table_field_takes_arithmetic_over_a_literal() -> None:
    res = _resolve(
        "SELECT r.w FROM input('f.mkv') f, unnest(ARRAY[STRUCT(1920 / 2 AS w)]) r"
    )
    assert res.struct_rows["r"].types == ("number",)


def test_struct_row_table_field_may_not_take_a_map_columns_name() -> None:
    err = _reject(
        "SELECT r.w FROM input('f.mkv') f, unnest(ARRAY[STRUCT(1 AS w, 'x' AS tags)]) r"
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "takes a name a row's maps already use" in err.message


def test_struct_row_table_requires_an_alias() -> None:
    err = _reject("SELECT 1 FROM input('f.mkv') f, unnest(ARRAY[STRUCT(1 AS w)])")
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "requires an alias" in err.message


def test_struct_row_table_joins_like_any_other_row_table() -> None:
    resolved = _resolve(
        "SELECT t FROM input('f.mkv') f, unnest(f.audio) t "
        "JOIN unnest(ARRAY[STRUCT(1 AS w)]) r ON t.index = r.w"
    )
    assert "r" in resolved.struct_rows


# ---------------------------------------------------------------------------
# ARRAY(SELECT ...) gathers
# ---------------------------------------------------------------------------


def test_array_gather_multi_column_without_struct_is_rejected() -> None:
    err = _reject(
        "SELECT ffmpeg.concat(VARIADIC ARRAY("
        "SELECT f.video, f.audio FROM input('f.mkv') f))"
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "single-column SELECT" in err.message


def test_array_gather_needs_a_branch_with_no_row_source_of_its_own() -> None:
    err = _reject(
        "SELECT 1 FROM input('f.mkv') f, unnest(f.audio) t "
        "WHERE t.index = ANY(ARRAY(SELECT i.i FROM generate_series(1, 2) i))"
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "no row source of its own" in err.message


def test_struct_gather_needs_a_name_for_each_field() -> None:
    err = _reject(
        "SELECT ffmpeg.concat(VARIADIC ARRAY("
        "SELECT AS STRUCT c.title, c.start_t + 1 "
        "FROM unnest(ARRAY[STRUCT('Intro' AS title, 0 AS start_t)]) c))"
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "needs a name" in err.message


def test_struct_gather_rejects_a_duplicate_field_name() -> None:
    err = _reject(
        "SELECT ffmpeg.concat(VARIADIC ARRAY("
        "SELECT AS STRUCT c.title, c.start_t AS title "
        "FROM unnest(ARRAY[STRUCT('Intro' AS title, 0 AS start_t)]) c))"
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "names the field 'title' twice" in err.message


def test_array_gather_subquery_may_not_carry_its_own_with() -> None:
    err = _reject(
        "SELECT ffmpeg.concat(VARIADIC ARRAY("
        "WITH x AS (SELECT f.video FROM input('f.mkv') f) SELECT x.video FROM x))"
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "own WITH" in err.message


def test_array_gather_subquery_requires_a_from_clause() -> None:
    err = _reject("SELECT ffmpeg.concat(VARIADIC ARRAY(SELECT 1))")
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "FROM clause" in err.message


def test_array_gather_uncountable_source_is_the_ordinary_rejection() -> None:
    err = _reject(
        "SELECT ffmpeg.concat(VARIADIC ARRAY(SELECT x.video FROM nope x))"
    )
    assert err.code is ErrorCode.UNKNOWN_ALIAS


# ---------------------------------------------------------------------------
# arithmetic, casts and <input>.duration in the compile-time value grammar
# ---------------------------------------------------------------------------

_ROWS = "FROM input('x.mp4') f, unnest(f.audio) t"


def test_arithmetic_over_row_columns_types_as_a_number() -> None:
    _resolve(f"SELECT t, t.channels * 2 AS chans {_ROWS}")


def test_arithmetic_needs_numbers_on_both_sides() -> None:
    err = _reject(f"SELECT t, t.tags.language + 1 AS x {_ROWS}")
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "'+' needs numbers, but one side is text" in err.message


def test_arithmetic_result_does_not_join_text() -> None:
    err = _reject(f"SELECT t, 'n=' || t.channels + 1 AS x {_ROWS}")
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "'||' joins text" in err.message


def test_cast_to_text_bridges_a_number_into_concatenation() -> None:
    _resolve(f"SELECT t, 'n=' || t.channels::text AS title {_ROWS}")


def test_cast_function_spelling_is_the_same_cast() -> None:
    _resolve(f"SELECT t, 'n=' || CAST(t.channels AS text) AS title {_ROWS}")


def test_only_text_is_a_supported_cast_target() -> None:
    err = _reject(f"SELECT t, t.tags.language::int + 1 AS x {_ROWS}")
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "cast to int is not supported" in err.message
    assert "::text is the only cast" in (err.hint or "")


def test_a_comparison_still_needs_one_type_across_a_cast() -> None:
    err = _reject(f"SELECT t {_ROWS} WHERE t.channels::text = 2")
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "a comparison needs one type" in err.message


def test_arithmetic_is_allowed_in_a_row_between_bound() -> None:
    _resolve(f"SELECT t {_ROWS} WHERE t.channels BETWEEN 1 AND 4 * 2")


def test_a_trim_bound_may_be_an_expression_over_duration() -> None:
    _resolve("SELECT f.video[1] FROM input('x.mp4') f WHERE f.t <= f.duration - 0.5")


def test_a_bare_duration_is_a_legal_trim_bound() -> None:
    _resolve("SELECT f.video[1] FROM input('x.mp4') f WHERE f.t <= f.duration")


def test_a_parenthesized_trim_bound_still_resolves() -> None:
    """A stray paren around the arithmetic must not change the shape check."""
    _resolve(
        "SELECT f.video[1] FROM input('x.mp4') f "
        "WHERE f.t >= (f.duration - 1) AND f.t <= f.duration"
    )


def test_a_text_trim_bound_is_still_rejected() -> None:
    err = _reject("SELECT f.video[1] FROM input('x.mp4') f WHERE f.t <= 'ten'")
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "time bounds must be numeric literals" in err.message


def test_a_trim_bound_may_read_a_track_row_column() -> None:
    """One window per row is a shape resolve admits and lower decides: one
    file each under a fan-out TO, one `-i` each when the rows are gathered.
    A conjunct that mixes the two worlds any OTHER way is still rejected
    (`test_a_conjunct_may_not_mix_a_row_column_with_another_alias`)."""
    _resolve(f"SELECT t {_ROWS} WHERE f.t <= t.channels")


def test_a_row_column_the_schema_never_had_is_still_unknown() -> None:
    err = _reject(f"SELECT t, t.nope * 2 AS x {_ROWS}")
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "unknown column 't.nope'" in err.message


def test_only_duration_is_readable_off_an_input_alias() -> None:
    err = _reject("SELECT f.video[1] FROM input('x.mp4') f WHERE f.t <= f.width - 1")
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "unknown column 'f.width'" in err.message


def test_a_container_tag_types_as_text_in_the_value_grammar() -> None:
    """`f.tags.title` is text, so `||` takes it and arithmetic does not."""
    _resolve(
        "SELECT f.video[1], f.tags.title || ' (restored)' AS title "
        "FROM input('x.mp4') f"
    )
    err = _reject(
        "SELECT f.video[1], f.tags.title + 1 AS title FROM input('x.mp4') f"
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL


def test_a_container_tag_is_readable_in_a_case_condition() -> None:
    _resolve(
        "SELECT f.video[1], CASE WHEN f.tags.comment IS NULL THEN 'none' "
        "ELSE f.tags.comment END AS comment FROM input('x.mp4') f"
    )


def test_an_input_column_that_is_not_a_tag_path_is_still_unknown() -> None:
    err = _reject("SELECT f.video[1], f.mood AS mood FROM input('x.mp4') f")
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "unknown column 'f.mood'" in err.message
    assert "container tags" in (err.hint or "")


# -- rendition columns: admitted on an input alias syntactically, since only
#    a probe (which runs after resolve) can say whether one applies --------


@pytest.mark.parametrize("column", ["height", "bandwidth", "name"])
def test_a_rendition_column_resolves_on_a_plain_input_alias(column: str) -> None:
    res = _resolve(f"SELECT r.{column} AS x FROM input('ladder.m3u8') r")
    assert res.branches[0].expressions[0].alias == "x"


def test_all_six_rendition_columns_resolve_on_a_plain_input_alias() -> None:
    columns = ", ".join(
        f"r.{name}" for name in ("bandwidth", "width", "height", "codecs", "name", "language")
    )
    _resolve(f"SELECT {columns} FROM input('ladder.m3u8') r")


def test_an_unknown_column_on_an_input_alias_is_still_unknown_the_same_way() -> None:
    """Widening resolve() to admit rendition columns must not widen the
    rejection's own wording for a column that is neither structural nor a
    rendition name -- the hint still lists only INPUT_COLUMNS, because
    resolve has not probed yet and cannot say a rendition table applies."""
    err = _reject("SELECT r.mood FROM input('ladder.m3u8') r")
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "unknown column 'r.mood'" in err.message
    assert "an input exposes attachments, audio, chapters, cues, data, " \
        "duration, subtitle, t, tags, video" in (err.hint or "")


def test_a_query_nested_too_deeply_is_a_plain_rejection() -> None:
    """Deep nesting is user input, not a bug: a typed, non-internal error
    with a plain message, at whichever layer the recursion limit bites."""
    import pytest

    from ffrwd.compiler import compile_sql
    from ffrwd.errors import ErrorCode, FfrwdError

    depth = 3000
    query = "SELECT " + "(" * depth + "a.video[1]" + ")" * depth + " FROM input('x.mp4') a"
    with pytest.raises(FfrwdError) as excinfo:
        compile_sql(query)
    assert excinfo.value.code is not ErrorCode.INTERNAL
    assert "nests too deeply" in excinfo.value.message
    assert "RecursionError" not in excinfo.value.message


def test_a_removed_stream_tag_field_names_the_path_form() -> None:
    err = _reject("SELECT t.language FROM input('f.mkv') f, unnest(f.audio) t")
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "'t.language' is not a column: a tag is read by path" in err.message
    assert err.hint == "read the tag: 't.tags.language'"


def test_a_removed_container_tag_column_names_the_path_form() -> None:
    err = _reject("SELECT f.artist FROM input('f.mkv') f")
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "'f.artist' is not a column: a tag is read by path" in err.message
    assert err.hint == "read the tag: 'f.tags.artist'"


def test_a_removed_tag_field_on_a_subscript_names_the_path_form() -> None:
    err = _reject(
        "SELECT f.audio[1] FROM input('f.mkv') f WHERE f.audio[1].title IS NULL"
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "'f.audio[1].title' is not a column" in err.message
    assert err.hint == "read the tag: 'f.audio[1].tags.title'"


def test_a_chapter_row_carries_no_tags() -> None:
    err = _reject("SELECT c.tags.title FROM input('f.mkv') f, unnest(f.chapters) c")
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "'c' is a chapter row, and a chapter carries no tags" in err.message


def test_a_bare_container_tags_column_is_not_a_value() -> None:
    err = _reject(
        "SELECT f.video[1], f.tags || ' (x)' AS title FROM input('f.mkv') f"
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "'f.tags' is the whole tag map, not a single value" in err.message


def test_a_bare_row_tags_column_is_not_a_comparison_operand() -> None:
    err = _reject(
        "SELECT t FROM input('f.mkv') f, unnest(f.audio) t WHERE t.tags = 'x'"
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "'t.tags' is the whole tag map, not a single value" in err.message


def test_a_subscripted_tag_map_wants_a_key() -> None:
    err = _reject(
        "SELECT f.audio[1] FROM input('f.mkv') f WHERE f.audio[1].tags IS NULL"
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "'f.audio[1].tags' is the whole tag map, not a value" in err.message


def test_a_tag_path_types_as_text() -> None:
    _resolve(
        "SELECT t FROM input('f.mkv') f, unnest(f.audio) t "
        "WHERE t.tags.whatever = 'x'"
    )
    err = _reject(
        "SELECT t FROM input('f.mkv') f, unnest(f.audio) t WHERE t.tags.whatever = 5"
    )
    assert "'t.tags.whatever' is text" in err.message


# -- disposition flags ------------------------------------------------------


def _rows(where: str) -> str:
    return f"SELECT t FROM input('f.mkv') f, unnest(f.audio) t WHERE {where}"


def test_a_disposition_path_types_as_boolean() -> None:
    _resolve(_rows("t.disposition.forced"))
    _resolve(_rows("t.disposition.forced = true"))
    err = _reject(_rows("t.disposition.forced = 'yes'"))
    assert "'t.disposition.forced' is boolean" in err.message
    assert err.hint == (
        "compare 't.disposition.forced' against the bare word true or false"
    )


def test_an_unknown_flag_names_the_closest_one() -> None:
    err = _reject(_rows("t.disposition.forcd"))
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "'t.disposition.forcd' is not a disposition flag" in err.message
    assert err.hint == "did you mean 'forced'?"


def test_an_unrecognizable_flag_lists_the_whole_set() -> None:
    err = _reject(_rows("t.disposition.nonsense"))
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    for key in DISPOSITION_KEYS:
        assert key in (err.hint or "")


def test_a_flag_reads_off_a_subscript_too() -> None:
    _resolve("SELECT f.audio[1] FROM input('f.mkv') f WHERE f.audio[1].disposition.default")
    err = _reject(
        "SELECT f.audio[1] FROM input('f.mkv') f WHERE f.audio[1].disposition.nope"
    )
    assert "'f.audio[1].disposition.nope' is not a disposition flag" in err.message


def test_a_bare_row_disposition_column_wants_a_flag() -> None:
    err = _reject(_rows("t.disposition = 'x'"))
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "'t.disposition' is the whole flag map, not a single value" in err.message
    assert err.hint == "name the key: 't.disposition.default'"


def test_a_subscripted_flag_map_wants_a_key() -> None:
    err = _reject(
        "SELECT f.audio[1] FROM input('f.mkv') f WHERE f.audio[1].disposition IS NULL"
    )
    assert "'f.audio[1].disposition' is the whole flag map, not a value" in err.message


def test_a_container_has_no_disposition() -> None:
    for query in (
        "SELECT f.video[1] FROM input('f.mkv') f WHERE f.disposition.default",
        "SELECT f.disposition FROM input('f.mkv') f",
    ):
        err = _reject(query)
        assert err.code is ErrorCode.UNSUPPORTED_SQL
        assert "'f.disposition' is a stream field, not a container one" in err.message


def test_a_chapter_row_carries_no_disposition() -> None:
    err = _reject(
        "SELECT c.disposition.default FROM input('f.mkv') f, unnest(f.chapters) c"
    )
    assert "'c' is a chapter row, and a chapter carries no disposition" in err.message


def test_a_non_boolean_column_is_not_a_condition_on_its_own() -> None:
    err = _reject(_rows("t.tags.language"))
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "'t.tags.language' is text, not a condition" in err.message


def test_a_boolean_does_not_join_text() -> None:
    err = _reject(
        "SELECT t, 'x' || t.disposition.default AS title "
        "FROM input('f.mkv') f, unnest(f.audio) t"
    )
    assert "'||' joins text, but one side is a boolean" in err.message
