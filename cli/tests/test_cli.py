"""Tests for the CLI.

Invokes ``main([...])`` directly (no subprocess) and asserts on captured
stdout/stderr via pytest's ``capsys``, plus the process exit code returned
by ``main``.

Convention: ``compile``/``explain``/``validate``/``run`` take the
query as an inline SQL string by default; ``-f/--file`` reads it from a file
instead ('-' for stdin). Tests below use the inline positional wherever the
file itself isn't under test, and ``-f`` (with a real file, or '-' for
stdin) where file-reading behavior is the point.
"""

from __future__ import annotations

import io
import json
import subprocess
from pathlib import Path

import pytest

from ffrwd import cli, show
from ffrwd.compiler import Compiled
from ffrwd.execute import PlanResult
from ffrwd.ir import Graph, Node, Output, RowsSink, SinkUnit
from ffrwd.processes import (
    FfmpegProcess,
    ModuleShape,
    ProcessPlan,
    SidecarProcess,
    external_ids,
    partition,
)

VALID_QUERY = "SELECT scale(a.video[1], 640, 480) FROM input('x.mp4') a"
BAD_QUERY = "SELECT nope(a.video[1]) FROM input('x.mp4') a"
# `compile` refuses a bare SELECT outright (no COPY, no destination to
# invent), so any test that wants a compile happy path wraps VALID_QUERY in
# a COPY ... TO like this one.
MEDIA_QUERY = f"COPY ({VALID_QUERY}) TO 'out.mp4'"

# For tests that monkeypatch cli.compile_all to return a hand-built, already
# -sinked Graph (see _sinked_graph/_multi_sink_graph below): the REAL text
# still goes through cli.classify() first, so it must
# itself look like a media query -- a real COPY, not FORMAT csv -- for
# `run`'s new table/csv branch to stay out of the way and reach the
# monkeypatched compile_all at all.
SINKED_QUERY = f"COPY ({VALID_QUERY}) TO 'ignored.mkv'"


def _write_sql(tmp_path: Path, text: str, name: str = "query.sql") -> str:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


def _sinked_graph(path: str, options: dict[str, object] | None = None) -> Graph:
    """A hand-built, already-sinked Graph.

    Sink-path-resolution tests below monkeypatch ``cli.compile_all`` to
    return this instead of compiling real SQL.
    """
    return _multi_sink_graph((path, dict(options or {})))


def _multi_sink_graph(*sinks: tuple[str, dict[str, object]]) -> Graph:
    """One passthrough output per COPY, all reading the same input stream.

    The cross-group passthrough shape, hand-built: two groups may map
    the same source stream without a split (see `ffrwd.split._exempt_refs`).
    """
    return Graph(
        input_paths=["x.mp4"],
        sources={"a": 0},
        nodes={},
        sinks=[
            SinkUnit(
                outputs=[Output(ref="src:a:v:0", type="video", name=None, metadata={})],
                path=path,
                options=dict(options),
            )
            for path, options in sinks
        ],
    )


# ---------------------------------------------------------------------------
# compile
# ---------------------------------------------------------------------------


def test_compile_happy_path(capsys: pytest.CaptureFixture[str]) -> None:
    code = cli.main(["compile", MEDIA_QUERY])
    out = capsys.readouterr().out
    assert code == 0
    assert "-filter_complex" in out
    assert "ffmpeg" in out
    assert "out.mp4" in out


def test_quiet_changes_nothing_on_stdout(capsys: pytest.CaptureFixture[str]) -> None:
    """-q mutes narration, which lives on stderr; what a script parses is untouched."""
    assert cli.main(["compile", MEDIA_QUERY]) == 0
    plain = capsys.readouterr()
    assert cli.main(["compile", "-q", MEDIA_QUERY]) == 0
    quiet = capsys.readouterr()
    assert quiet.out == plain.out
    assert quiet.err == ""


def test_compile_graph_only(capsys: pytest.CaptureFixture[str]) -> None:
    code = cli.main(["compile", "--graph-only", VALID_QUERY])
    out = capsys.readouterr().out
    assert code == 0
    assert "-filter_complex" not in out
    assert "scale=" in out


def test_compile_bad_query_exits_1(capsys: pytest.CaptureFixture[str]) -> None:
    code = cli.main(["compile", BAD_QUERY])
    captured = capsys.readouterr()
    assert code == 1
    assert captured.out == ""
    assert captured.err.startswith("error: ")
    assert "UNKNOWN_FUNCTION" in captured.err


def test_compile_file_missing(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = cli.main(["compile", "-f", str(tmp_path / "nope.sql")])
    captured = capsys.readouterr()
    assert code == 1
    assert captured.out == ""
    assert "error:" in captured.err


def test_compile_file_happy_path(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    query = _write_sql(tmp_path, MEDIA_QUERY)
    code = cli.main(["compile", "-f", query])
    out = capsys.readouterr().out
    assert code == 0
    assert "-filter_complex" in out


def test_compile_file_dash_reads_stdin(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import io

    monkeypatch.setattr(cli.sys, "stdin", io.StringIO(MEDIA_QUERY))
    code = cli.main(["compile", "-f", "-"])
    out = capsys.readouterr().out
    assert code == 0
    assert "-filter_complex" in out


def test_compile_uses_sink_path(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        cli, "compile_all", lambda text, **kw: Compiled([_sinked_graph("sink.mkv")])
    )
    code = cli.main(["compile", VALID_QUERY])
    out = capsys.readouterr().out
    assert code == 0
    assert "sink.mkv" in out


def test_compile_prints_every_sink_path_of_a_script(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """One ffmpeg command, one output file per COPY."""
    monkeypatch.setattr(
        cli,
        "compile_all",
        lambda text, **kw: Compiled([_multi_sink_graph(("720.mp4", {}), ("360.mp4", {}))]),
    )
    code = cli.main(["compile", VALID_QUERY])
    out = capsys.readouterr().out
    assert code == 0
    assert out.count("ffmpeg") == 1
    assert "720.mp4" in out
    assert "360.mp4" in out


def test_compile_graph_only_still_works_for_a_multi_sink_script(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """--graph-only prints the filtergraph, which has no paths in it at all."""
    monkeypatch.setattr(
        cli,
        "compile_all",
        lambda text, **kw: Compiled([_multi_sink_graph(("720.mp4", {}), ("360.mp4", {}))]),
    )
    code = cli.main(["compile", "--graph-only", VALID_QUERY])
    assert code == 0
    assert capsys.readouterr().out == "\n"  # pure passthrough graph


# ---------------------------------------------------------------------------
# compile on a table/csv query
# ---------------------------------------------------------------------------


TABLE_QUERY = "SELECT t.tags.language FROM input('tests/fixtures/av2.mp4') f, unnest(f.audio) t"


@pytest.fixture
def _two_track_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    """A synthetic probe for the unnest-based CLI tests.

    The default tier must be deterministic on a bare machine (no ffprobe, no
    generated fixtures -- CI runs it before installing either), and a row
    query needs probed rows; so the rows are synthetic. The fixture PATH in
    the queries is illustrative, never read.
    """
    from ffrwd import compiler
    from ffrwd.probe import ProbeResult, StreamMeta

    def _track(index: int, language: str) -> StreamMeta:
        return StreamMeta(
            type="audio",
            index=index,
            metadata={"language": language},
            width=None,
            height=None,
            fps=None,
            sample_rate=44100,
            codec="aac",
        )

    result = ProbeResult(streams=[_track(0, "eng"), _track(1, "fra")])
    monkeypatch.setattr(compiler, "probe_path", lambda path, args=(), **kw: result)


def test_compile_on_a_table_query_is_a_typed_usage_message(
    capsys: pytest.CaptureFixture[str], _two_track_probe: None
) -> None:
    """Metadata columns have no ffmpeg command to show; `compile` says so and
    points at `run` instead of trying (and failing) to produce one."""
    code = cli.main(["compile", TABLE_QUERY])
    captured = capsys.readouterr()
    assert code == 2
    assert captured.out == ""
    assert "run" in captured.err


@pytest.fixture
def _tagged_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    """A synthetic probe carrying container tags, for the same reason
    `_two_track_probe` is synthetic: the default tier reads no file."""
    from ffrwd import compiler
    from ffrwd.probe import ProbeResult, StreamMeta

    video = StreamMeta(
        type="video",
        index=0,
        metadata={},
        width=320,
        height=240,
        fps="15/1",
        sample_rate=None,
        codec="h264",
    )
    result = ProbeResult(streams=[video], tags={"title": "Angel One"})
    monkeypatch.setattr(compiler, "probe_path", lambda path, args=(), **kw: result)


def test_compile_writes_container_tags_read_from_the_input(
    capsys: pytest.CaptureFixture[str], _tagged_probe: None
) -> None:
    """The whole path, end to end: a probed tag read, rewritten, cleared."""
    code = cli.main(
        [
            "compile",
            "COPY (SELECT f.video[1], "
            "STRUCT(f.tags.title || ' (restored)' AS title, NULL AS artist) AS tags "
            "FROM input('film.mkv') f) TO 'out.mkv'",
        ]
    )
    captured = capsys.readouterr()
    assert code == 0
    assert captured.out.strip() == (
        "ffmpeg -i film.mkv -map 0:v:0 -c:0 copy -metadata artist= "
        "-metadata 'title=Angel One (restored)' out.mkv"
    )


def test_run_prints_container_tags_as_a_table(
    capsys: pytest.CaptureFixture[str], _tagged_probe: None
) -> None:
    code = cli.main(["run", "SELECT f.tags.title, f.tags.artist FROM input('film.mkv') f"])
    captured = capsys.readouterr()
    assert code == 0
    assert "Angel One" in captured.out
    assert captured.out.startswith(" title")


def test_compile_refuses_a_bare_stream_select_even_though_it_would_compile(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A bare SELECT has no COPY ... TO, so it is a table query, always --
    compile never invents a destination for it, even one made only of stream
    columns that COULD become an ffmpeg command."""
    code = cli.main(["compile", VALID_QUERY])
    captured = capsys.readouterr()
    assert code == 2
    assert captured.out == ""
    assert "run" in captured.err


def test_validate_accepts_a_table_query(
    capsys: pytest.CaptureFixture[str], _two_track_probe: None
) -> None:
    code = cli.main(["validate", TABLE_QUERY])
    captured = capsys.readouterr()
    assert code == 0
    assert captured.out == ""
    assert captured.err == ""


def test_explain_shows_the_sinks_list(capsys: pytest.CaptureFixture[str]) -> None:
    code = cli.main(["explain", VALID_QUERY])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert [unit["path"] for unit in payload["sinks"]] == [None]


# ---------------------------------------------------------------------------
# --portable and --no-probe are GONE
# ---------------------------------------------------------------------------

# There is no portable subset any more: every function is a filter of the
# installed ffmpeg, so --portable had nothing left to select. --no-probe made
# a READABLE file compile as if unreadable, silently stripping provenance
# metadata -- a determinism switch that changed the result -- so it is gone
# too; opportunistic probing already degrades silently on
# missing/unreadable inputs, which is the whole of what it was ever for.


def test_removed_flags_are_gone_from_every_subcommand() -> None:
    for command in ("compile", "explain", "validate", "run"):
        for flag in ("--portable", "--no-probe"):
            with pytest.raises(SystemExit) as exc_info:
                cli.main([command, VALID_QUERY, flag])
            assert exc_info.value.code == 2, f"{command} {flag}"


def test_dash_o_is_no_longer_a_recognized_flag() -> None:
    """The query names its own destination with COPY ... TO now; -o is
    argparse's ordinary unrecognized-argument error, nothing bespoke."""
    for command in ("compile", "run"):
        with pytest.raises(SystemExit) as exc_info:
            cli.main([command, VALID_QUERY, "-o", "out.mp4"])
        assert exc_info.value.code == 2, command


# ---------------------------------------------------------------------------
# explain
# ---------------------------------------------------------------------------


def test_explain_round_trips_graph(capsys: pytest.CaptureFixture[str]) -> None:
    code = cli.main(["explain", VALID_QUERY])
    out = capsys.readouterr().out
    assert code == 0

    import json

    data = json.loads(out)
    graph = Graph.from_dict(data)
    assert graph.input_paths == ["x.mp4"]
    assert graph.to_dict() == data


def test_explain_shows_the_inserted_pts_reset_node(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`explain` dumps the compiled IR, so the `setpts` this compiler
    inserts after an unreset `ffmpeg.trim` call shows up in it, exactly as
    an inserted `split` node would."""
    query = "SELECT ffmpeg.trim(a.video[1], starti => 1) FROM input('x.mp4') a"
    code = cli.main(["explain", query])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert [node["filter"] for node in payload["nodes"]] == ["trim", "setpts"]
    reset_node = payload["nodes"][1]
    assert reset_node["args"] == {"expr": "PTS-STARTPTS"}
    assert reset_node["inputs"] == ["n1"]


def test_explain_bad_query(capsys: pytest.CaptureFixture[str]) -> None:
    code = cli.main(["explain", BAD_QUERY])
    captured = capsys.readouterr()
    assert code == 1
    assert captured.out == ""
    assert "error:" in captured.err


def test_explain_file_happy_path(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    query = _write_sql(tmp_path, VALID_QUERY)
    code = cli.main(["explain", "-f", query])
    out = capsys.readouterr().out
    assert code == 0

    import json

    data = json.loads(out)
    assert data["inputs"] == ["x.mp4"]


def test_explain_mermaid_prints_the_flowchart(capsys: pytest.CaptureFixture[str]) -> None:
    code = cli.main(["explain", "--mermaid", VALID_QUERY])
    out = capsys.readouterr().out
    assert code == 0
    assert out.startswith("flowchart LR")
    assert "scale" in out
    assert '"nodes"' not in out  # the flowchart replaces the JSON dump


def test_explain_mermaid_and_diagram_exclude_each_other(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as caught:
        cli.main(["explain", "--mermaid", "--diagram", VALID_QUERY])
    assert caught.value.code == 2
    assert "not allowed with" in capsys.readouterr().err


def test_explain_diagram_without_termaid_names_the_extra(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli.diagram, "termaid_available", lambda: False)
    code = cli.main(["explain", "--diagram", VALID_QUERY])
    captured = capsys.readouterr()
    assert code == 1
    assert captured.out == ""
    assert captured.err.startswith("error: ")
    assert 'pip install "ffrwd[diagram]"' in captured.err


def test_explain_diagram_renders_the_flowchart_through_termaid(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli.diagram, "termaid_available", lambda: True)
    monkeypatch.setattr(
        cli.diagram, "render_terminal", lambda text: f"boxes({text.splitlines()[0]})"
    )
    code = cli.main(["explain", "--diagram", VALID_QUERY])
    assert code == 0
    assert capsys.readouterr().out == "boxes(flowchart LR)\n"


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


def test_validate_success_is_silent(capsys: pytest.CaptureFixture[str]) -> None:
    code = cli.main(["validate", VALID_QUERY])
    captured = capsys.readouterr()
    assert code == 0
    assert captured.out == ""
    assert captured.err == ""


def test_validate_bad_query_human(capsys: pytest.CaptureFixture[str]) -> None:
    code = cli.main(["validate", BAD_QUERY])
    captured = capsys.readouterr()
    assert code == 1
    assert captured.out == ""
    assert captured.err.startswith("error: ")


def test_validate_bad_query_json_inline(capsys: pytest.CaptureFixture[str]) -> None:
    """validate --json on an inline SQL string."""
    code = cli.main(["validate", "--json", BAD_QUERY])
    captured = capsys.readouterr()
    assert code == 1
    assert captured.err == ""

    import json

    data = json.loads(captured.out)
    assert data["code"] == "UNKNOWN_FUNCTION"
    assert set(data.keys()) == {"line", "col", "code", "message", "hint"}


def test_validate_file_happy_path(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    query = _write_sql(tmp_path, VALID_QUERY)
    code = cli.main(["validate", "-f", query])
    captured = capsys.readouterr()
    assert code == 0
    assert captured.out == ""
    assert captured.err == ""


def test_validate_file_dash_reads_stdin(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import io

    monkeypatch.setattr(cli.sys, "stdin", io.StringIO(BAD_QUERY))
    code = cli.main(["validate", "--json", "-f", "-"])
    captured = capsys.readouterr()
    assert code == 1

    import json

    data = json.loads(captured.out)
    assert data["code"] == "UNKNOWN_FUNCTION"


# ---------------------------------------------------------------------------
# usage: exactly one of the positional query / -f is required
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("command", ["compile", "explain", "validate", "run"])
def test_both_query_and_file_is_a_usage_error(
    command: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    query = _write_sql(tmp_path, VALID_QUERY)
    code = cli.main([command, VALID_QUERY, "-f", query])
    captured = capsys.readouterr()
    assert code == 2
    assert captured.out == ""
    assert "not both" in captured.err


@pytest.mark.parametrize("command", ["compile", "explain", "validate", "run"])
def test_neither_query_nor_file_is_a_usage_error(
    command: str, capsys: pytest.CaptureFixture[str]
) -> None:
    code = cli.main([command])
    captured = capsys.readouterr()
    assert code == 2
    assert captured.out == ""
    assert "error:" in captured.err


# ---------------------------------------------------------------------------
# did-you-mean-f hint
# ---------------------------------------------------------------------------


def test_positional_existing_sql_file_gets_the_file_hint(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Passing an existing .sql file's path positionally (instead of -f) fails
    to parse as SQL and gets a second stderr line suggesting -f."""
    path = _write_sql(tmp_path, VALID_QUERY)
    code = cli.main(["compile", path])
    captured = capsys.readouterr()
    assert code == 1
    assert captured.err.startswith("error: ")
    assert "PARSE_ERROR" in captured.err
    assert f"hint: '{path}' looks like a file; did you mean -f '{path}'?" in captured.err


def test_positional_dot_sql_suffix_gets_the_file_hint_even_if_missing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The hint also fires on a .sql-suffixed path that does not exist -- the
    suffix alone is muscle-memory enough to suggest -f."""
    path = str(tmp_path / "does_not_exist.sql")
    code = cli.main(["compile", path])
    captured = capsys.readouterr()
    assert code == 1
    assert f"hint: '{path}' looks like a file; did you mean -f '{path}'?" in captured.err


def test_hint_does_not_fire_for_ordinary_bad_sql(capsys: pytest.CaptureFixture[str]) -> None:
    """Ordinary invalid SQL that neither exists as a file nor ends in .sql
    gets no hint -- the guard is specific to the muscle-memory case."""
    code = cli.main(["compile", "not sql at all !!"])
    captured = capsys.readouterr()
    assert code == 1
    assert "PARSE_ERROR" in captured.err
    assert "hint:" not in captured.err


def test_hint_does_not_fire_when_using_dash_f(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """-f itself never triggers the hint, even against a file that fails to
    parse as SQL -- the hint only applies to the inline-positional case."""
    bad_file = _write_sql(tmp_path, "not sql at all !!", name="bad.sql")
    code = cli.main(["compile", "-f", bad_file])
    captured = capsys.readouterr()
    assert code == 1
    assert "PARSE_ERROR" in captured.err
    assert "hint:" not in captured.err


def test_hint_does_not_fire_for_non_parse_errors(capsys: pytest.CaptureFixture[str]) -> None:
    """The hint is scoped to PARSE_ERROR only -- a non-PARSE_ERROR rejection
    (this one's error message carries the library's own unrelated "(hint:
    did you mean ...)" text) never gets the CLI's did-you-mean-f line."""
    code = cli.main(["compile", BAD_QUERY])
    captured = capsys.readouterr()
    assert code == 1
    assert "UNKNOWN_FUNCTION" in captured.err
    assert "did you mean -f" not in captured.err
    assert captured.err.count("\n") == 1


def test_hint_on_validate_json_goes_to_stderr_stdout_stays_pure_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Machine contract: --json's stdout is the library error
    verbatim, untouched by the hint; the hint (if any) only ever goes to
    stderr."""
    path = _write_sql(tmp_path, VALID_QUERY)
    code = cli.main(["validate", "--json", path])
    captured = capsys.readouterr()
    assert code == 1

    import json

    data = json.loads(captured.out)
    assert data["code"] == "PARSE_ERROR"
    assert set(data.keys()) == {"line", "col", "code", "message", "hint"}
    assert f"hint: '{path}' looks like a file; did you mean -f '{path}'?" in captured.err


# ---------------------------------------------------------------------------
# Windows note: a query containing single quotes must survive argparse
# unquoted -- the common case is input('x.mp4') itself.
# ---------------------------------------------------------------------------


def test_inline_query_with_single_quotes_compiles(capsys: pytest.CaptureFixture[str]) -> None:
    query = "COPY (SELECT scale(a.video[1], 0.5) FROM input('a clip''s name.mp4') a) TO 'out.mp4'"
    code = cli.main(["compile", query])
    out = capsys.readouterr().out
    assert code == 0
    assert "-filter_complex" in out


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def test_run_ffmpeg_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    out_path = str(tmp_path / "out.mp4")
    monkeypatch.setattr(cli.binaries, "ffmpeg_path", lambda: None)

    code = cli.main(["run", f"COPY ({VALID_QUERY}) TO :'dest'", "-v", f"dest={out_path}"])
    captured = capsys.readouterr()
    assert code == 1
    assert "ffmpeg" in captured.err
    assert "not found" in captured.err


def test_run_bad_query_never_checks_ffmpeg(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def _boom() -> str | None:
        raise AssertionError("should not check for ffmpeg before compiling")

    monkeypatch.setattr(cli.binaries, "ffmpeg_path", _boom)
    code = cli.main(["run", BAD_QUERY])
    captured = capsys.readouterr()
    assert code == 1
    assert "error:" in captured.err


def test_run_uses_sink_path(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """compile_all is monkeypatched -- see `_sinked_graph`'s docstring."""
    monkeypatch.setattr(
        cli, "compile_all", lambda text, **kw: Compiled([_sinked_graph("sink.mkv")])
    )
    monkeypatch.setattr(cli.binaries, "ffmpeg_path", lambda: None)

    code = cli.main(["run", SINKED_QUERY])
    captured = capsys.readouterr()
    assert code == 1
    assert "ffmpeg" in captured.err
    assert "not found" in captured.err


def test_run_bare_select_prints_a_table(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A sinkless SELECT is a TABLE query, always -- no COPY means run prints
    it rather than trying to invent a media destination for it."""
    code = cli.main(["run", VALID_QUERY])
    captured = capsys.readouterr()
    assert code == 0
    assert "(1 row)" in captured.out
    assert "<video" in captured.out


def test_run_csv_copy_to_stdout(
    capsys: pytest.CaptureFixture[str], _two_track_probe: None
) -> None:
    sql = (
        "COPY (SELECT t.tags.language, t.codec FROM "
        "input('tests/fixtures/av2.mp4') f, unnest(f.audio) t) "
        "TO STDOUT WITH (format 'csv', header true)"
    )
    code = cli.main(["run", sql])
    captured = capsys.readouterr()
    assert code == 0
    assert captured.out == "language,codec\neng,aac\nfra,aac\n"


def test_run_csv_copy_to_a_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], _two_track_probe: None
) -> None:
    out_path = tmp_path / "tracks.csv"
    sql = (
        "COPY (SELECT t.tags.language, t.codec FROM "
        f"input('tests/fixtures/av2.mp4') f, unnest(f.audio) t) "
        f"TO '{out_path.as_posix()}' WITH (format 'csv')"
    )
    code = cli.main(["run", sql])
    captured = capsys.readouterr()
    assert code == 0
    assert captured.out == ""  # nothing printed -- it went to the file
    assert out_path.read_text(encoding="utf-8") == "eng,aac\nfra,aac\n"


def test_run_csv_copy_defaults_header_false(
    capsys: pytest.CaptureFixture[str], _two_track_probe: None
) -> None:
    sql = (
        "COPY (SELECT t.tags.language FROM input('tests/fixtures/av2.mp4') f, "
        "unnest(f.audio) t) TO STDOUT WITH (format 'csv')"
    )
    code = cli.main(["run", sql])
    captured = capsys.readouterr()
    assert code == 0
    assert captured.out == "eng\nfra\n"  # no header row


def test_run_of_a_multi_sink_script_reaches_the_ffmpeg_check(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Every COPY's path is the query's own -- `run` needs no override at all."""
    monkeypatch.setattr(
        cli,
        "compile_all",
        lambda text, **kw: Compiled([_multi_sink_graph(("720.mp4", {}), ("360.mp4", {}))]),
    )
    monkeypatch.setattr(cli.binaries, "ffmpeg_path", lambda: None)
    code = cli.main(["run", SINKED_QUERY])
    captured = capsys.readouterr()
    assert code == 1
    assert "not found" in captured.err


def test_run_reports_no_output_path_for_a_sink_with_none(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """classify() already routes every genuine bare SELECT to the table
    path, so the only sink that can still reach this check is a hand-built
    one -- the shape a `COPY ... TO STDOUT WITH (FORMAT csv)` sink takes
    when it ends up in a media script instead of a table one."""
    graph = Graph(
        input_paths=["x.mp4"],
        sources={"a": 0},
        nodes={},
        sinks=[SinkUnit(outputs=[Output(ref="src:a:v:0", type="video", name=None, metadata={})])],
    )
    monkeypatch.setattr(cli, "compile_all", lambda text, **kw: Compiled([graph]))
    code = cli.main(["run", SINKED_QUERY])
    captured = capsys.readouterr()
    assert code == 2
    assert "no output path given" in captured.err


def test_run_checks_every_sinks_output_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = str(tmp_path / "does_not_exist" / "360.mp4")
    monkeypatch.setattr(
        cli,
        "compile_all",
        lambda text, **kw: Compiled([_multi_sink_graph(("720.mp4", {}), (missing, {}))]),
    )
    monkeypatch.setattr(cli.binaries, "ffmpeg_path", lambda: "/usr/bin/ffmpeg")
    code = cli.main(["run", SINKED_QUERY])
    captured = capsys.readouterr()
    assert code == 1
    assert "does not exist" in captured.err


def test_run_missing_output_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    out_path = str(tmp_path / "does_not_exist" / "out.mp4")
    monkeypatch.setattr(cli.binaries, "ffmpeg_path", lambda: "/usr/bin/ffmpeg")

    code = cli.main(["run", f"COPY ({VALID_QUERY}) TO :'dest'", "-v", f"dest={out_path}"])
    captured = capsys.readouterr()
    assert code == 1
    assert "does not exist" in captured.err


def test_run_file_happy_path_reaches_ffmpeg_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """-f still works end to end for run (the LLM repair-loop pipe uses -f -)."""
    query = _write_sql(tmp_path, f"COPY ({VALID_QUERY}) TO :'dest'")
    out_path = str(tmp_path / "out.mp4")
    monkeypatch.setattr(cli.binaries, "ffmpeg_path", lambda: None)

    code = cli.main(["run", "-f", query, "-v", f"dest={out_path}"])
    captured = capsys.readouterr()
    assert code == 1
    assert "ffmpeg" in captured.err
    assert "not found" in captured.err


# ---------------------------------------------------------------------------
# command sequences: a two_pass query is two chained ffmpeg commands
# ---------------------------------------------------------------------------

TWO_PASS_QUERY = (
    "COPY (SELECT a.video[1], a.audio[1] FROM input('x.mp4') a) TO 'out.mp4' "
    "WITH (video_codec 'libx264', video_bitrate '2500k', two_pass true)"
)


def test_compile_prints_a_two_pass_chain_on_one_line(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = cli.main(["compile", TWO_PASS_QUERY])
    captured = capsys.readouterr()
    assert code == 0
    line = captured.out.rstrip("\n")
    assert "\n" not in line
    first, second = line.split(" && ")
    assert first.startswith("ffmpeg ")
    assert first.endswith("-pass 1 -passlogfile out.mp4 -f null -")
    assert second.startswith("ffmpeg ")
    assert second.endswith("-pass 2 -passlogfile out.mp4 out.mp4")


def test_compile_prints_an_ordinary_query_unchained(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = cli.main(["compile", MEDIA_QUERY])
    captured = capsys.readouterr()
    assert code == 0
    assert " && " not in captured.out


def _record_runs(monkeypatch: pytest.MonkeyPatch, codes: list[int]) -> list[list[str]]:
    """Stub `subprocess.run` to return `codes` in order; collect each argv."""
    calls: list[list[str]] = []
    remaining = list(codes)

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        calls.append(list(argv))
        return subprocess.CompletedProcess(args=argv, returncode=remaining.pop(0))

    monkeypatch.setattr(cli.binaries, "ffmpeg_path", lambda: "/usr/bin/ffmpeg")
    monkeypatch.setattr(subprocess, "run", fake_run)
    return calls


def test_run_executes_both_passes_in_order(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls = _record_runs(monkeypatch, [0, 0])

    code = cli.main(["run", TWO_PASS_QUERY, "-y"])
    captured = capsys.readouterr()

    assert code == 0
    assert len(calls) == 2
    assert calls[0][:3] == ["ffmpeg", "-hide_banner", "-y"]
    assert calls[0][-3:] == ["-f", "null", "-"]
    assert calls[1][-1] == "out.mp4"
    assert captured.out.count("$ ffmpeg") == 2


def test_run_stops_at_the_first_failing_pass(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls = _record_runs(monkeypatch, [3, 0])

    code = cli.main(["run", TWO_PASS_QUERY, "-y"])
    captured = capsys.readouterr()

    assert code == 3
    assert len(calls) == 1  # pass 2 never starts, so nothing writes the dest
    assert "exited with code 3" in captured.err


def test_run_of_an_ordinary_query_still_executes_one_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _record_runs(monkeypatch, [0])

    code = cli.main(["run", f"COPY ({VALID_QUERY}) TO 'out.mp4'"])

    assert code == 0
    assert calls == [
        [
            "ffmpeg",
            "-hide_banner",
            "-n",
            "-i",
            "x.mp4",
            "-filter_complex",
            "[0:v:0]scale=width=640:height=480[out0]",
            "-map",
            "[out0]",
            "out.mp4",
        ]
    ]


# ---------------------------------------------------------------------------
# command sequences: loudnorm2 is two commands with a shell handoff between
# ---------------------------------------------------------------------------

LOUDNORM2_QUERY = (
    "COPY (SELECT ffrwd.loudnorm2(a.audio[1], I => -16) FROM input('x.mp4') a) "
    "TO 'out.m4a' WITH (audio_codec 'aac')"
)

# What ffmpeg's stderr looks like where it matters (tests/test_loudnorm.py
# carries the full capture); `run`'s stub below hands this back verbatim.
LOUDNORM_JSON = """\
[Parsed_loudnorm_0 @ 0000000000000000]
{
    "input_i" : "-21.76",
    "input_tp" : "-17.69",
    "input_lra" : "0.00",
    "input_thresh" : "-31.76",
    "target_offset" : "0.05"
}
"""


def test_compile_prints_a_loudnorm2_chain_on_one_line(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = cli.main(["compile", LOUDNORM2_QUERY])
    captured = capsys.readouterr()
    assert code == 0
    line = captured.out.rstrip("\n")
    assert "\n" not in line
    first, second = line.split(" && ")
    assert first.startswith('eval "$(ffmpeg ')
    assert first.endswith('-f null - 2>&1 | ffrwd loudnorm2env)"')
    assert second.startswith("ffmpeg ")
    assert second.endswith("-c:0 aac out.m4a")


def test_the_printed_correction_pass_leaves_the_variables_expandable(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Adjacent-quote concatenation: the filtergraph stays ONE argument, but
    the `${...}` parts sit in double quotes so the shell expands them."""
    cli.main(["compile", LOUDNORM2_QUERY])
    line = capsys.readouterr().out
    assert ":measured_I='\"${FFRWD_LN_I}\"':measured_TP='" in line
    assert "'${FFRWD_LN_I}'" not in line


def test_compile_of_a_loudnorm2_query_needs_no_ffmpeg(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The macro resolves offline; only the PRINTED line depends on
    `ffrwd loudnorm2env` being there at run time."""
    monkeypatch.setattr(cli.binaries, "ffmpeg_path", lambda: None)
    assert cli.main(["compile", LOUDNORM2_QUERY]) == 0
    assert "loudnorm2env" in capsys.readouterr().out


def _record_loudnorm_runs(
    monkeypatch: pytest.MonkeyPatch, stderr: str
) -> list[list[str]]:
    """Stub `subprocess.run` so the measuring pass hands back `stderr`."""
    calls: list[list[str]] = []

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(list(argv))
        captured = stderr if kwargs.get("stderr") is subprocess.PIPE else ""
        return subprocess.CompletedProcess(args=argv, returncode=0, stderr=captured)

    monkeypatch.setattr(cli.binaries, "ffmpeg_path", lambda: "/usr/bin/ffmpeg")
    monkeypatch.setattr(subprocess, "run", fake_run)
    return calls


def test_run_substitutes_the_measurements_into_the_second_command(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls = _record_loudnorm_runs(monkeypatch, LOUDNORM_JSON)

    code = cli.main(["run", LOUDNORM2_QUERY, "-y"])
    captured = capsys.readouterr()

    assert code == 0
    assert len(calls) == 2
    measure = calls[0][calls[0].index("-filter_complex") + 1]
    assert measure.endswith("print_format=json[out0]")
    correct = calls[1][calls[1].index("-filter_complex") + 1]
    assert "measured_I=-21.76:measured_TP=-17.69" in correct
    assert "offset=0.05:linear=true" in correct
    # No shell anywhere: nothing is ever asked to expand a `${...}`.
    assert "${" not in correct
    assert "eval" not in captured.out


def test_run_reports_stderr_with_no_loudnorm_json(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls = _record_loudnorm_runs(monkeypatch, "Conversion failed!\n")

    code = cli.main(["run", LOUDNORM2_QUERY, "-y"])
    captured = capsys.readouterr()

    assert code == 1
    assert len(calls) == 1  # the correction pass never starts
    assert "no loudnorm JSON block" in captured.err


# ---------------------------------------------------------------------------
# loudnorm2env: stdin -> export lines
# ---------------------------------------------------------------------------


def test_loudnorm2env_prints_the_exports(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO(LOUDNORM_JSON))
    code = cli.main(["loudnorm2env"])
    captured = capsys.readouterr()
    assert code == 0
    assert captured.out.splitlines() == [
        "export FFRWD_LN_I=-21.76",
        "export FFRWD_LN_TP=-17.69",
        "export FFRWD_LN_LRA=0.00",
        "export FFRWD_LN_THRESH=-31.76",
        "export FFRWD_LN_OFFSET=0.05",
    ]


def test_loudnorm2env_rejects_input_with_no_json(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO("ffmpeg version 9.0.1\n"))
    code = cli.main(["loudnorm2env"])
    captured = capsys.readouterr()
    assert code == 1
    assert captured.out == ""
    assert captured.err.splitlines() == ["error: no loudnorm JSON block in the input"]


def test_loudnorm2env_takes_no_query_argument() -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["loudnorm2env", "query.sql"])
    assert exc_info.value.code == 2


# ---------------------------------------------------------------------------
# usage
# ---------------------------------------------------------------------------


def test_no_subcommand_exits_2(capsys: pytest.CaptureFixture[str]) -> None:
    code = cli.main([])
    assert code == 2


def test_unknown_subcommand_falls_through_to_run_and_fails_as_sql(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`run` is the default subcommand, unconditionally --
    no plausibility checking. A mistyped subcommand is not special-cased; it is
    just run's SQL text, and dies as an ordinary compile error (exit 1), a
    better diagnostic than a bare usage line."""
    code = cli.main(["bogus"])
    captured = capsys.readouterr()
    assert code == 1
    assert "error:" in captured.err


# ---------------------------------------------------------------------------
# default subcommand
# ---------------------------------------------------------------------------


def test_flag_first_argv_dispatches_to_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], _two_track_probe: None
) -> None:
    """``ffrwd -f query.sql`` (no ``run`` token, flags first) is exactly
    what the cookbook recipes show and what test_examples.py exercises end
    to end; this pins the same dispatch directly against a table query."""
    query = _write_sql(
        tmp_path,
        "SELECT t.index FROM input('tests/fixtures/av2.mp4') f, unnest(f.audio) t",
    )
    code = cli.main(["-f", query])
    captured = capsys.readouterr()
    assert code == 0
    assert "(2 rows)" in captured.out


def test_bare_inline_sql_with_no_subcommand_dispatches_to_run(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = cli.main([VALID_QUERY])
    captured = capsys.readouterr()
    assert code == 0
    assert "(1 row)" in captured.out


def test_explicit_run_and_default_dispatch_agree(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    query = _write_sql(tmp_path, VALID_QUERY)
    code_explicit = cli.main(["run", "-f", query])
    explicit_out = capsys.readouterr().out
    code_default = cli.main(["-f", query])
    default_out = capsys.readouterr().out
    assert code_explicit == code_default == 0
    assert explicit_out == default_out


# ---------------------------------------------------------------------------
# -v/--set CLI variables
# ---------------------------------------------------------------------------

VAR_QUERY = "SELECT scale(a.video[1], 640, 480) FROM input(:'path') a"
MEDIA_VAR_QUERY = f"COPY ({VAR_QUERY}) TO 'out.mp4'"


def test_compile_with_v_substitutes(capsys: pytest.CaptureFixture[str]) -> None:
    code = cli.main(["compile", MEDIA_VAR_QUERY, "-v", "path=x.mp4"])
    out = capsys.readouterr().out
    assert code == 0
    assert "x.mp4" in out


def test_compile_with_set_alias(capsys: pytest.CaptureFixture[str]) -> None:
    code = cli.main(["compile", MEDIA_VAR_QUERY, "--set", "path=x.mp4"])
    out = capsys.readouterr().out
    assert code == 0
    assert "x.mp4" in out


def test_duplicate_v_last_wins(capsys: pytest.CaptureFixture[str]) -> None:
    code = cli.main(["compile", MEDIA_VAR_QUERY, "-v", "path=first.mp4", "-v", "path=second.mp4"])
    out = capsys.readouterr().out
    assert code == 0
    assert "second.mp4" in out
    assert "first.mp4" not in out


def test_malformed_v_exits_2(capsys: pytest.CaptureFixture[str]) -> None:
    for bad_pair in ("noequals", "1bad=x", "=novame"):
        code = cli.main(["compile", VALID_QUERY, "-v", bad_pair])
        captured = capsys.readouterr()
        assert code == 2, bad_pair
        assert captured.out == ""
        assert "error:" in captured.err
        assert "-v/--set" in captured.err


def test_missing_var_through_compile(capsys: pytest.CaptureFixture[str]) -> None:
    # An unset variable is NULL; input() cannot take an absent path, and the
    # rejection names the variable.
    code = cli.main(["compile", VAR_QUERY])
    captured = capsys.readouterr()
    assert code == 1
    assert captured.out == ""
    assert captured.err.startswith("error: ")
    assert "UNSUPPORTED_SQL" in captured.err
    assert "':path' was not set" in captured.err
    assert "-v path=" in captured.err


def test_missing_var_through_validate_json(capsys: pytest.CaptureFixture[str]) -> None:
    code = cli.main(["validate", "--json", VAR_QUERY])
    captured = capsys.readouterr()
    assert code == 1

    data = json.loads(captured.out)
    assert data["code"] == "UNSUPPORTED_SQL"
    assert data["message"] == "':path' was not set"


def test_naked_dispatch_accepts_v(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """`run`'s own parser must know -v -- naked dispatch (no explicit `run`
    token) reuses it, so this pins that it isn't lost along the way."""
    query = _write_sql(tmp_path, VAR_QUERY)
    code = cli.main(["-f", query, "-v", "path=x.mp4"])
    captured = capsys.readouterr()
    assert code == 0
    assert "(1 row)" in captured.out


def test_unreferenced_v_is_a_usage_error(capsys: pytest.CaptureFixture[str]) -> None:
    # The check reversed: an unset reference is legal (NULL), so a -v naming
    # a variable the text never references is the mistake worth catching.
    code = cli.main(["compile", MEDIA_QUERY, "-v", "unused=1"])
    captured = capsys.readouterr()
    assert code == 2
    assert captured.out == ""
    assert "unused=..." in captured.err
    assert "never references" in captured.err
    assert "it references no variables" in captured.err


def test_unreferenced_v_names_what_the_query_references(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = cli.main(["compile", MEDIA_VAR_QUERY, "-v", "paht=x.mp4"])
    captured = capsys.readouterr()
    assert code == 2
    assert "paht=..." in captured.err
    assert "it references :path" in captured.err


def test_an_unset_optional_variable_compiles_to_absence(
    capsys: pytest.CaptureFixture[str],
) -> None:
    query = "COPY (SELECT scale(a.video[1], :w, :h) FROM input('x.mp4') a) TO 'o.mp4'"
    code = cli.main(["compile", query, "-v", "h=480"])
    out = capsys.readouterr().out
    assert code == 0
    assert "scale=height=480" in out
    assert "width" not in out


def test_v_accepted_on_explain(capsys: pytest.CaptureFixture[str]) -> None:
    code = cli.main(["explain", VAR_QUERY, "-v", "path=x.mp4"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["inputs"] == ["x.mp4"]


def test_v_accepted_on_run(capsys: pytest.CaptureFixture[str]) -> None:
    code = cli.main(["run", VAR_QUERY, "-v", "path=x.mp4"])
    captured = capsys.readouterr()
    assert code == 0
    assert "(1 row)" in captured.out


def test_a_protocol_url_destination_skips_the_directory_check() -> None:
    """udp://host:port is not a path; ffmpeg owns protocol URLs, so the
    missing-output-directory pre-flight must not parse them as files."""
    assert cli._check_output_dir("udp://127.0.0.1:12399?pkt_size=1316") is None
    assert cli._check_output_dir("rtmp://live.example.com/app/key") is None
    assert cli._check_output_dir("no-such-dir/out.mp4") is not None


def test_version_flag_prints_the_tool_version(capsys: pytest.CaptureFixture[str]) -> None:
    """Both psql spellings; checked before the run dispatch would eat the flag."""
    from importlib import metadata

    expected = f"ffrwd {metadata.version('ffrwd')}\n"
    assert cli.main(["--version"]) == 0
    assert capsys.readouterr().out == expected
    assert cli.main(["-V"]) == 0
    assert capsys.readouterr().out == expected


def test_help_output_names_the_version(capsys: pytest.CaptureFixture[str]) -> None:
    """`ffrwd -h` lands on run's help; it must carry the version string."""
    from importlib import metadata

    version = metadata.version("ffrwd")
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["-h"])
    assert excinfo.value.code == 0
    assert f"ffrwd {version}" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# run --show / --show-only
# ---------------------------------------------------------------------------


def test_show_refuses_a_table_query(capsys: pytest.CaptureFixture[str]) -> None:
    """A bare SELECT prints a result set and runs no ffmpeg, so there is no
    picture for a window to carry."""
    code = cli.main(["run", "--show", VALID_QUERY])
    captured = capsys.readouterr()
    assert code == 1
    assert "NOTHING_TO_SHOW" in captured.err


def test_show_refuses_an_audio_only_output(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    graph = Graph(
        input_paths=["x.mp4"],
        sources={"a": 0},
        nodes={},
        sinks=[
            SinkUnit(
                outputs=[Output(ref="src:a:a:0", type="audio", name=None, metadata={})],
                path="out.m4a",
            )
        ],
    )
    monkeypatch.setattr(cli, "compile_all", lambda text, **kw: Compiled([graph]))

    code = cli.main(["run", "--show", SINKED_QUERY])
    captured = capsys.readouterr()
    assert code == 1
    assert "NOTHING_TO_SHOW" in captured.err


def test_show_refuses_without_ffplay(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """ffplay has no provisioner fallback, so an absent one is a refusal --
    and only when a window was actually asked for."""
    monkeypatch.setattr(
        cli, "compile_all", lambda text, **kw: Compiled([_sinked_graph("sink.mkv")])
    )
    monkeypatch.setattr(cli.binaries, "ffplay_path", lambda: None)

    code = cli.main(["run", "--show", SINKED_QUERY])
    captured = capsys.readouterr()
    assert code == 1
    assert "PLAYER_NOT_FOUND" in captured.err
    assert "ffplay" in captured.err


def test_a_run_without_show_never_looks_for_ffplay(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def _boom() -> str | None:
        raise AssertionError("ffplay was looked for without --show")

    monkeypatch.setattr(
        cli, "compile_all", lambda text, **kw: Compiled([_sinked_graph("sink.mkv")])
    )
    monkeypatch.setattr(cli.binaries, "ffplay_path", _boom)
    monkeypatch.setattr(cli.binaries, "ffmpeg_path", lambda: None)

    assert cli.main(["run", SINKED_QUERY]) == 1
    assert "ffmpeg" in capsys.readouterr().err


def test_show_and_show_only_are_mutually_exclusive(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """One saves and shows, the other only shows: asking for both is a
    contradiction, and argparse names the pair."""
    with pytest.raises(SystemExit):
        cli.main(["run", "--show", "--show-only", SINKED_QUERY])
    assert "--show-only" in capsys.readouterr().err


# --- a query that calls a module -------------------------------------------


def _module_plan(path: str = "sink.mkv", *, pure: bool = True) -> ProcessPlan:
    """decode -> module -> encode, the shape a wasm query partitions into."""
    g = Graph(input_paths=["x.mp4"], sources={"a": 0})
    g.nodes["m0"] = Node(
        id="m0", filter="m0.wasm", args={}, inputs=["src:a:v:0"], outputs=["video"]
    )
    g.sinks = [
        SinkUnit(
            outputs=[Output(ref="m0", type="video", name=None, metadata={})], path=path
        )
    ]
    return partition(
        g, external=external_ids("m0"), shapes={"m0.wasm": ModuleShape(pure=pure)}
    )


@pytest.fixture
def _ready_to_run(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, object]]:
    """A run that spawns nothing: the binaries are there and the plan is caught."""
    caught: list[dict[str, object]] = []

    def _capture(plan: ProcessPlan, **kw: object) -> PlanResult:
        caught.append({"plan": plan, **kw})
        return PlanResult()

    monkeypatch.setattr(cli.binaries, "ffmpeg_path", lambda: "/usr/bin/ffmpeg")
    monkeypatch.setattr(cli.binaries, "ffplay_path", lambda: "/usr/bin/ffplay")
    monkeypatch.setattr(cli, "execute_plan", _capture)
    return caught


def _terminal_sinks(caught: dict[str, object]) -> list[str | None]:
    """Where the caught plan's terminal process writes."""
    plan = caught["plan"]
    assert isinstance(plan, ProcessPlan)
    process = plan.process("ffmpeg0")
    assert isinstance(process, FfmpegProcess)
    return [unit.path for unit in process.graph.sinks]


def test_show_opens_a_window_on_a_plans_terminal_process(
    monkeypatch: pytest.MonkeyPatch, _ready_to_run: list[dict[str, object]]
) -> None:
    """The terminal ffmpeg's stdout is free, so it gains a display output and
    a player of its own, keyed by the process the window stands for."""
    monkeypatch.setattr(
        cli,
        "compile_all",
        lambda text, **kw: Compiled([_sinked_graph("sink.mkv")], plan=_module_plan()),
    )

    assert cli.main(["run", "--show", SINKED_QUERY]) == 0
    caught = _ready_to_run[0]
    assert _terminal_sinks(caught) == ["sink.mkv", show.DISPLAY_PATH]
    players = caught["players"]
    assert isinstance(players, dict)
    assert list(players) == ["ffmpeg0"]
    assert "sink.mkv" in players["ffmpeg0"]


def test_show_only_on_a_plan_writes_nothing(
    monkeypatch: pytest.MonkeyPatch, _ready_to_run: list[dict[str, object]]
) -> None:
    monkeypatch.setattr(
        cli,
        "compile_all",
        lambda text, **kw: Compiled([_sinked_graph("sink.mkv")], plan=_module_plan()),
    )

    assert cli.main(["run", "--show-only", SINKED_QUERY]) == 0
    assert _terminal_sinks(_ready_to_run[0]) == [show.DISPLAY_PATH]


def test_show_only_on_a_plan_checks_no_suppressed_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    _ready_to_run: list[dict[str, object]],
) -> None:
    """A destination nothing will write is not a destination to check."""
    missing = str(tmp_path / "does_not_exist" / "out.mkv")
    monkeypatch.setattr(
        cli,
        "compile_all",
        lambda text, **kw: Compiled([_sinked_graph(missing)], plan=_module_plan(missing)),
    )

    assert cli.main(["run", "--show-only", SINKED_QUERY]) == 0
    assert _ready_to_run  # it ran, rather than refusing the missing directory


def test_show_refuses_a_plan_that_writes_no_video(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A module whose rows are the whole output puts nothing in a window."""
    g = Graph(input_paths=["x.mp4"], sources={"a": 0})
    g.nodes["m0"] = Node(
        id="m0", filter="m0.wasm", args={}, inputs=["src:a:v:0"], outputs=["video"]
    )
    g.rows_sinks = {"m0": RowsSink(container="ndjson", path="rows.ndjson")}
    monkeypatch.setattr(
        cli,
        "compile_all",
        lambda text, **kw: Compiled(
            [_sinked_graph("sink.mkv")], plan=partition(g, external=external_ids("m0"))
        ),
    )

    code = cli.main(["run", "--show", SINKED_QUERY])
    captured = capsys.readouterr()
    assert code == 1
    assert "NOTHING_TO_SHOW" in captured.err


# --- --jobs, the parallel hosting count -------------------------------------


@pytest.mark.parametrize("command", ["compile", "run"])
@pytest.mark.parametrize("count", ["0", "-1"])
def test_jobs_below_one_is_a_usage_error(
    command: str, count: str, capsys: pytest.CaptureFixture[str]
) -> None:
    code = cli.main([command, MEDIA_QUERY, "--jobs", count])
    captured = capsys.readouterr()
    assert code == 2
    assert captured.out == ""
    assert "--jobs must be 1 or more" in captured.err


def _sidecar_argv(caught: dict[str, object]) -> list[str]:
    """The argv the caught run would have spawned its sidecar with."""
    plan = caught["plan"]
    assert isinstance(plan, ProcessPlan)
    process = plan.process("sidecar0")
    assert isinstance(process, SidecarProcess)
    render = caught["sidecar_argv"]
    assert callable(render)
    rendered = render(process)
    assert isinstance(rendered, list)
    return rendered


@pytest.fixture
def _installed_sidecar(monkeypatch: pytest.MonkeyPatch) -> None:
    """A located sidecar binary, so an argv renders without one on PATH."""
    monkeypatch.setattr(cli.binaries, "ffrwd_wasm_path", lambda: "/usr/bin/ffrwd-wasm")


def test_jobs_reaches_the_sidecar_of_a_pure_module(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    _ready_to_run: list[dict[str, object]],
    _installed_sidecar: None,
) -> None:
    monkeypatch.setattr(
        cli,
        "compile_all",
        lambda text, **kw: Compiled([_sinked_graph("sink.mkv")], plan=_module_plan()),
    )

    assert cli.main(["run", SINKED_QUERY, "--jobs", "4"]) == 0
    argv = _sidecar_argv(_ready_to_run[0])
    assert argv[argv.index("-jobs") + 1] == "4"
    assert "warning:" not in capsys.readouterr().err


def test_the_default_leaves_the_sidecar_argv_as_it_was(
    monkeypatch: pytest.MonkeyPatch,
    _ready_to_run: list[dict[str, object]],
    _installed_sidecar: None,
) -> None:
    monkeypatch.setattr(
        cli,
        "compile_all",
        lambda text, **kw: Compiled([_sinked_graph("sink.mkv")], plan=_module_plan()),
    )

    assert cli.main(["run", SINKED_QUERY]) == 0
    assert "-jobs" not in _sidecar_argv(_ready_to_run[0])


def test_an_impure_module_is_named_and_left_serial(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    _ready_to_run: list[dict[str, object]],
    _installed_sidecar: None,
) -> None:
    monkeypatch.setattr(
        cli,
        "compile_all",
        lambda text, **kw: Compiled(
            [_sinked_graph("sink.mkv")], plan=_module_plan(pure=False)
        ),
    )

    assert cli.main(["run", SINKED_QUERY, "--jobs", "4"]) == 0
    assert "-jobs" not in _sidecar_argv(_ready_to_run[0])
    err = capsys.readouterr().err
    assert "warning: --jobs 4 does not reach 'm0.wasm'" in err
    assert "carries state between calls" in err


def _network_plan() -> ProcessPlan:
    """Two modules wired to each other: one region, hands frames on in order."""
    g = Graph(input_paths=["x.mp4"], sources={"a": 0})
    g.nodes["m0"] = Node(
        id="m0", filter="m0.wasm", args={}, inputs=["src:a:v:0"], outputs=["video"]
    )
    g.nodes["m1"] = Node(
        id="m1", filter="m1.wasm", args={}, inputs=["m0"], outputs=["video"]
    )
    g.sinks = [
        SinkUnit(
            outputs=[Output(ref="m1", type="video", name=None, metadata={})],
            path="sink.mkv",
        )
    ]
    return partition(
        g,
        external=external_ids("m0", "m1"),
        shapes={"m0.wasm": ModuleShape(), "m1.wasm": ModuleShape()},
    )


def test_a_network_of_modules_says_why_it_stays_serial(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    _ready_to_run: list[dict[str, object]],
    _installed_sidecar: None,
) -> None:
    """Every module is pure; the wiring between them is what keeps one worker."""
    monkeypatch.setattr(
        cli,
        "compile_all",
        lambda text, **kw: Compiled([_sinked_graph("sink.mkv")], plan=_network_plan()),
    )

    assert cli.main(["run", SINKED_QUERY, "--jobs", "4"]) == 0
    assert "-jobs" not in _sidecar_argv(_ready_to_run[0])
    err = capsys.readouterr().err
    assert "warning: --jobs 4 does not reach" in err
    assert "hand frames on in order" in err


def test_the_default_says_nothing_about_an_impure_module(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    _ready_to_run: list[dict[str, object]],
    _installed_sidecar: None,
) -> None:
    """Nobody asked for parallel hosting, so nobody is owed the reason."""
    monkeypatch.setattr(
        cli,
        "compile_all",
        lambda text, **kw: Compiled(
            [_sinked_graph("sink.mkv")], plan=_module_plan(pure=False)
        ),
    )

    assert cli.main(["run", SINKED_QUERY]) == 0
    assert "warning:" not in capsys.readouterr().err


def test_compile_prints_the_jobs_it_would_run_with(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        cli,
        "compile_all",
        lambda text, **kw: Compiled([_sinked_graph("sink.mkv")], plan=_module_plan()),
    )

    assert cli.main(["compile", SINKED_QUERY, "--jobs", "4"]) == 0
    assert "-jobs 4" in capsys.readouterr().out
