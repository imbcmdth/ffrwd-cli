"""Tests for the MCP server: the tool bodies, the SDK wiring, and the CLI.

Three tiers, deliberately separated:

* ``ffrwd.mcp.tools`` is the whole behavior and imports no SDK, so those
  tests run everywhere -- including on a machine with no ``mcp`` package.
* the SDK wiring (which tools exist, what a call returns) needs the optional
  extra and is skipped without it.
* the CLI's ``mcp`` subcommand is tested both ways, since the message for a
  missing SDK is exactly what a user without the extra sees.

No subprocess anywhere, and every tool is asserted to leave stdout empty:
stdout is the protocol stream once the server is running.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ffrwd import cli
from ffrwd import mcp as mcp_package
from ffrwd.diagram import render_diagram
from ffrwd.errors import ErrorCode, FfrwdError
from ffrwd.ir import Graph
from ffrwd.mcp import tools

# The hand-rolled JSON Schema check that pins FfrwdError.to_dict(); reused
# here so `validate`'s payload is held to the same schema file.
from .test_docs import _load_schema, _validate

VALID_QUERY = "SELECT scale(a.video[1], 640, 480) FROM input('x.mp4') a"
MEDIA_QUERY = f"COPY ({VALID_QUERY}) TO 'out.mp4'"
BAD_QUERY = "SELECT nope(a.video[1]) FROM input('x.mp4') a"
BAD_MEDIA_QUERY = f"COPY ({BAD_QUERY}) TO 'out.mp4'"
TABLE_QUERY = "SELECT a.video[1] FROM input('x.mp4') a"


# ---------------------------------------------------------------------------
# compile
# ---------------------------------------------------------------------------


def test_compile_returns_one_argv_per_command() -> None:
    result = tools.compile_query(MEDIA_QUERY)
    assert result["commands"] == [
        [
            "ffmpeg",
            "-i",
            "x.mp4",
            "-filter_complex",
            "[0:v:0]scale=width=640:height=480[out0]",
            "-map",
            "[out0]",
            "out.mp4",
        ]
    ]
    assert result["filter_complex"] == ["[0:v:0]scale=width=640:height=480[out0]"]
    assert result["outputs"] == ["out.mp4"]
    assert result["needs_measurement"] is False


def test_compile_substitutes_variables() -> None:
    query = "COPY (SELECT scale(a.video[1], :w, 480) FROM input(:'src') a) TO :'dst'"
    result = tools.compile_query(query, {"w": "640", "src": "in.mp4", "dst": "out.mp4"})
    assert result["outputs"] == ["out.mp4"]
    assert "in.mp4" in result["commands"][0]
    assert "scale=width=640:height=480" in result["filter_complex"][0]


def test_compile_flags_a_loudnorm2_query_as_needing_measurement() -> None:
    query = (
        "COPY (SELECT ffrwd.loudnorm2(a.audio[1]) FROM input('x.mp4') a) TO 'out.m4a'"
    )
    result = tools.compile_query(query)
    assert result["needs_measurement"] is True
    assert len(result["commands"]) == 2


def test_compile_refuses_a_row_query_and_names_inspect() -> None:
    with pytest.raises(FfrwdError) as excinfo:
        tools.compile_query(TABLE_QUERY)
    assert excinfo.value.code is ErrorCode.UNSUPPORTED_SQL
    assert "inspect" in (excinfo.value.hint or "")


def test_compile_reports_the_original_error_for_a_query_that_is_no_row_query() -> None:
    with pytest.raises(FfrwdError) as excinfo:
        tools.compile_query(BAD_MEDIA_QUERY)
    assert excinfo.value.code is ErrorCode.UNKNOWN_FUNCTION


# ---------------------------------------------------------------------------
# validate -- the repair loop's structured half
# ---------------------------------------------------------------------------


def test_validate_returns_empty_for_a_query_that_compiles() -> None:
    assert tools.validate_query(MEDIA_QUERY) == {}


def test_validate_returns_empty_for_a_row_query() -> None:
    assert tools.validate_query(TABLE_QUERY) == {}


def test_validate_returns_a_schema_conformant_error_object() -> None:
    result = tools.validate_query(BAD_MEDIA_QUERY)
    _validate(result, _load_schema())
    assert result["code"] == ErrorCode.UNKNOWN_FUNCTION.value
    assert result["line"] == 1
    assert isinstance(result["message"], str) and result["message"]


def test_validate_error_object_survives_json_round_trip() -> None:
    result = tools.validate_query(BAD_MEDIA_QUERY)
    assert json.loads(json.dumps(result)) == result


def test_validate_reports_an_unset_variable_instead_of_raising() -> None:
    # Unset means NULL; input() cannot take an absent path, and the error
    # names the variable.
    result = tools.validate_query("COPY (SELECT a.video[1] FROM input(:'src') a) TO 'o.mp4'")
    _validate(result, _load_schema())
    assert result["code"] == ErrorCode.UNSUPPORTED_SQL.value
    assert result["message"] == "':src' was not set"


def test_validate_reports_an_unreferenced_vars_entry() -> None:
    result = tools.validate_query(MEDIA_QUERY, {"nope": "1"})
    _validate(result, _load_schema())
    assert result["code"] == ErrorCode.UNSUPPORTED_SQL.value
    assert "'nope'" in str(result["message"])
    assert "never references" in str(result["message"])


def test_compile_drops_an_unset_optional_variable() -> None:
    query = "COPY (SELECT scale(a.video[1], :w, 480) FROM input('in.mp4') a) TO 'o.mp4'"
    result = tools.compile_query(query)
    assert "scale=height=480" in result["filter_complex"][0]


@pytest.mark.parametrize(
    "query",
    [
        "",
        "   ",
        "DROP TABLE users",
        "SELECT",
        "\x00\x01",
        "COPY (SELECT",
        "(" * 200,
        "SELECT " + "scale(" * 60 + "a.video[1]" + ")" * 60 + " FROM input('x.mp4') a",
    ],
)
def test_validate_never_raises(query: str) -> None:
    result = tools.validate_query(query)
    assert isinstance(result, dict)
    if result:
        _validate(result, _load_schema())


# ---------------------------------------------------------------------------
# explain
# ---------------------------------------------------------------------------


def test_explain_returns_one_graph_per_command() -> None:
    result = tools.explain_query(MEDIA_QUERY)
    graphs = result["graphs"]
    assert len(graphs) == 1
    assert graphs[0]["nodes"][0]["filter"] == "scale"
    assert json.loads(json.dumps(result)) == result


def test_explain_raises_on_a_query_that_does_not_compile() -> None:
    with pytest.raises(FfrwdError):
        tools.explain_query(BAD_MEDIA_QUERY)


def test_explain_carries_the_mermaid_flowchart() -> None:
    result = tools.explain_query(MEDIA_QUERY)
    graphs = [Graph.from_dict(graph) for graph in result["graphs"]]
    assert result["mermaid"] == render_diagram(graphs)
    assert result["mermaid"].startswith("flowchart LR")


# ---------------------------------------------------------------------------
# inspect
# ---------------------------------------------------------------------------


def test_inspect_returns_rows_columns_and_printable_text() -> None:
    result = tools.inspect_query(TABLE_QUERY)
    assert len(result["results"]) == 1
    first = result["results"][0]
    assert first["columns"] == ["column"]
    assert first["rows"] == [["<video 0:v:0>"]]
    assert "<video 0:v:0>" in first["text"]
    assert first["csv"] is False
    assert first["path"] is None


def test_inspect_reports_a_csv_sinks_destination() -> None:
    query = "COPY (SELECT a.video[1] FROM input('x.mp4') a) TO 'rows.csv' WITH (format 'csv')"
    first = tools.inspect_query(query)["results"][0]
    assert first["csv"] is True
    assert first["path"] == "rows.csv"


def test_inspect_refuses_a_media_query_and_names_compile() -> None:
    with pytest.raises(FfrwdError) as excinfo:
        tools.inspect_query(MEDIA_QUERY)
    assert excinfo.value.code is ErrorCode.UNSUPPORTED_SQL
    assert "compile" in (excinfo.value.hint or "")


# ---------------------------------------------------------------------------
# filters
# ---------------------------------------------------------------------------


def test_filters_lists_the_local_ffmpegs_whole_surface() -> None:
    result = tools.list_filters()
    assert result["available"] is True
    names = {f["name"] for f in result["filters"]}
    assert {"scale", "volume", "loudnorm"} <= names
    assert "color" in {s["name"] for s in result["sources"]}
    assert "options" not in result


def test_filters_narrows_to_a_substring_of_name_or_description() -> None:
    result = tools.list_filters("loudnorm")
    assert [f["name"] for f in result["filters"]] == ["loudnorm"]
    assert result["filters"][0]["inputs"] == ["audio"]
    assert result["filters"][0]["output"] == "audio"


def test_filters_adds_the_options_when_the_pattern_is_one_filters_name() -> None:
    result = tools.list_filters("scale")
    options = {o["name"]: o for o in result["options"]}
    assert "width" in options
    assert options["width"]["type"] == "str"


def test_filters_omits_options_when_the_pattern_names_nothing() -> None:
    result = tools.list_filters("nosuchfilteranywhere")
    assert result["filters"] == []
    assert result["sources"] == []
    assert "options" not in result


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def test_run_refuses_a_row_query_before_reaching_ffmpeg() -> None:
    with pytest.raises(FfrwdError) as excinfo:
        tools.run_query(TABLE_QUERY)
    assert excinfo.value.code is ErrorCode.UNSUPPORTED_SQL
    assert "inspect" in (excinfo.value.hint or "")


def test_run_reports_the_compile_error_without_running_anything() -> None:
    with pytest.raises(FfrwdError) as excinfo:
        tools.run_query(BAD_MEDIA_QUERY)
    assert excinfo.value.code is ErrorCode.UNKNOWN_FUNCTION


def test_run_keeps_only_the_tail_of_a_long_stderr() -> None:
    text = "x" * (tools.STDERR_LIMIT + 500)
    tail = tools._tail(text)
    assert tail.endswith("x" * tools.STDERR_LIMIT)
    assert tail.startswith("[earlier output dropped]")
    assert tools._tail("short") == "short"


# ---------------------------------------------------------------------------
# stdout stays empty: it is the protocol stream
# ---------------------------------------------------------------------------


def test_no_tool_writes_to_stdout(capsys: pytest.CaptureFixture[str]) -> None:
    tools.compile_query(MEDIA_QUERY)
    tools.validate_query(MEDIA_QUERY)
    tools.validate_query(BAD_MEDIA_QUERY)
    tools.explain_query(MEDIA_QUERY)
    tools.inspect_query(TABLE_QUERY)
    tools.list_filters("scale")
    tools.dialect_prompt()
    for query in (TABLE_QUERY, BAD_MEDIA_QUERY):
        with pytest.raises(FfrwdError):
            tools.run_query(query)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_the_dialect_prompt_is_the_one_the_prompt_subcommand_prints() -> None:
    from ffrwd.prompt import build_system_prompt
    from ffrwd.registry import load_reference

    snapshot = load_reference(Path(__file__).resolve().parent / "data" / "reference_registry.json")
    assert tools.dialect_prompt() == build_system_prompt(snapshot)


# ---------------------------------------------------------------------------
# the SDK wiring
# ---------------------------------------------------------------------------

_sdk = pytest.mark.skipif(
    not mcp_package.sdk_available(), reason="the mcp extra is not installed"
)

_BASE_TOOLS = {"compile", "validate", "explain", "inspect", "filters", "search"}

# The tools that answer about something other than one query, and so take no
# query text: what the local ffmpeg has, and what the registry publishes.
_QUERYLESS_TOOLS = {"filters", "search", "install"}


def _call(server: Any, name: str, arguments: dict[str, Any]) -> Any:
    import anyio

    return anyio.run(lambda: server.call_tool(name, arguments))


@_sdk
def test_the_writing_tools_are_absent_unless_they_were_allowed() -> None:
    import anyio

    from ffrwd.mcp.server import build_server

    server = build_server()
    names = {t.name for t in anyio.run(server.list_tools)}
    assert names == _BASE_TOOLS


@_sdk
def test_run_and_install_are_registered_when_unsafe_tools_were_allowed() -> None:
    import anyio

    from ffrwd.mcp.server import build_server

    server = build_server(allow_unsafe=True)
    names = {t.name for t in anyio.run(server.list_tools)}
    assert names == _BASE_TOOLS | {"run", "install"}


@_sdk
def test_the_install_tool_says_it_downloads_code_and_writes_files() -> None:
    import anyio

    from ffrwd.mcp.server import build_server

    server = build_server(allow_unsafe=True)
    tool = next(t for t in anyio.run(server.list_tools) if t.name == "install")
    # The description is the text an MCP client shows the user when it asks.
    text = (tool.description or "").lower()
    assert "downloads code" in text and "writes files" in text
    assert "ffrwd.lock" in text and "network" in text
    assert set(tool.input_schema["properties"]) == {"package", "project"}
    assert tool.input_schema["required"] == ["package", "project"]


@_sdk
def test_every_tool_takes_the_query_and_optional_variables() -> None:
    import anyio

    from ffrwd.mcp.server import build_server

    server = build_server(allow_unsafe=True)
    for tool in anyio.run(server.list_tools):
        properties = tool.input_schema["properties"]
        assert tool.description
        if tool.name in _QUERYLESS_TOOLS:
            assert "query" not in properties
            continue
        assert {"query", "vars"} <= set(properties)
        assert tool.input_schema["required"] == ["query"]


@_sdk
def test_a_rejected_query_fails_the_call_with_the_line_anchored_message() -> None:
    from mcp.server.mcpserver.exceptions import ToolError

    from ffrwd.mcp.server import build_server

    # The SDK turns this into a failed tool result for the client; what
    # matters here is that the whole typed error rides in the message.
    with pytest.raises(ToolError) as excinfo:
        _call(build_server(), "compile", {"query": BAD_MEDIA_QUERY})
    message = str(excinfo.value)
    assert "line 1:14" in message
    assert "UNKNOWN_FUNCTION" in message
    assert "hint:" in message


@_sdk
def test_validate_comes_back_as_a_successful_call_carrying_the_error_object() -> None:
    from ffrwd.mcp.server import build_server

    result = _call(build_server(), "validate", {"query": BAD_MEDIA_QUERY})
    assert result.is_error is False
    _validate(dict(result.structured_content), _load_schema())
    assert result.structured_content["code"] == ErrorCode.UNKNOWN_FUNCTION.value


@_sdk
def test_variables_reach_the_compiler_through_the_tool_call() -> None:
    from ffrwd.mcp.server import build_server

    result = _call(
        build_server(),
        "compile",
        {
            "query": "COPY (SELECT a.video[1] FROM input(:'src') a) TO 'o.mkv'",
            "vars": {"src": "in.mp4"},
        },
    )
    assert result.is_error is False
    assert "in.mp4" in result.structured_content["commands"][0]


@_sdk
def test_the_dialect_resource_serves_the_system_prompt() -> None:
    import anyio

    from ffrwd.mcp.server import DIALECT_URI, build_server

    server = build_server()
    resources = anyio.run(server.list_resources)
    assert [str(r.uri) for r in resources] == [DIALECT_URI]

    contents = list(anyio.run(lambda: server.read_resource(DIALECT_URI)))
    assert contents[0].content == tools.dialect_prompt()


# ---------------------------------------------------------------------------
# the CLI subcommand
# ---------------------------------------------------------------------------


def test_the_mcp_subcommand_says_how_to_install_the_extra_when_it_is_missing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(mcp_package, "sdk_available", lambda: False)
    assert cli.main(["mcp"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert 'pip install "ffrwd[mcp]"' in captured.err
    assert "Traceback" not in captured.err


def test_the_mcp_subcommand_serves_and_prints_nothing_to_stdout(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    served: list[bool] = []
    monkeypatch.setattr(mcp_package, "sdk_available", lambda: True)
    monkeypatch.setattr(
        mcp_package, "serve", lambda *, allow_unsafe: served.append(allow_unsafe)
    )

    assert cli.main(["mcp"]) == 0
    assert cli.main(["mcp", "--allow-unsafe"]) == 0
    assert served == [False, True]
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_mcp_is_a_subcommand_and_not_a_query() -> None:
    assert "mcp" in cli._SUBCOMMANDS
    assert "mcp" in cli._HANDLERS


# ---------------------------------------------------------------------------
# a query that partitions into processes
# ---------------------------------------------------------------------------


def _planned_compiled() -> Any:
    from ffrwd.compiler import Compiled
    from ffrwd.ir import Graph, Node, Output, SinkUnit
    from ffrwd.processes import external_ids, partition

    graph = Graph(
        input_paths=["x.mp4"],
        sources={"a": 0},
        nodes={
            "m0": Node(
                id="m0", filter="m0.wasm", args={}, inputs=["src:a:v:0"], outputs=["video"]
            )
        },
        sinks=[
            SinkUnit(
                outputs=[Output(ref="m0", type="video", name=None, metadata={})],
                path="sink.mkv",
            )
        ],
    )
    return Compiled([graph], plan=partition(graph, external=external_ids("m0")))


def test_compile_answers_a_partitioned_query_with_the_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The compile of a wasm query is the plan, never bare ffmpeg halves."""
    monkeypatch.setattr(tools, "compile_all", lambda text, **kw: _planned_compiled())
    result = tools.compile_query("COPY (SELECT a.video[1] FROM input('x.mp4') a) TO 'sink.mkv'")
    assert "commands" not in result
    assert "plan" in result
    assert "ffrwd-wasm" in result["pipeline"]
    assert result["outputs"] == ["sink.mkv"]


def test_run_executes_a_partitioned_query_as_its_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ffrwd.execute import PlanResult

    ran: list[object] = []

    def _capture(plan: object, **kw: object) -> PlanResult:
        ran.append(plan)
        return PlanResult()

    monkeypatch.setattr(tools, "compile_all", lambda text, **kw: _planned_compiled())
    monkeypatch.setattr(tools, "execute_plan", _capture)
    monkeypatch.setattr(tools.binaries, "ffmpeg_path", lambda: "/usr/bin/ffmpeg")
    result = tools.run_query("COPY (SELECT a.video[1] FROM input('x.mp4') a) TO 'sink.mkv'")
    assert len(ran) == 1
    assert result["exit_code"] == 0
    assert result["stages"] == []
    assert "commands" not in result
    assert result["outputs"] == ["sink.mkv"]
