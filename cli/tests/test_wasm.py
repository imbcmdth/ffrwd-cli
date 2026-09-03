"""Tests for ``LANGUAGE wasm``: the declaration, the call, and the plan it becomes.

Bare-machine by rule, and this file is the one where that costs something: the
whole point of the feature is a sidecar subprocess, so nothing here may spawn
one. Every test hands lowering a synthetic :class:`~ffrwd.wasm.Described` the
same way tests/test_lower.py hands it a synthetic ``ProbeResult`` -- module
paths deliberately do not exist, and `describe` is never the real one. Running
an actual module against actual ffmpeg is the exec tier's, in
tests/exec/test_exec_wasm.py.

The filter surface is the captured snapshot, so a query mixing a module with
an ffmpeg filter resolves on a machine with no ffmpeg.
"""

from __future__ import annotations

import functools
import json
import subprocess
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import pytest
from sqlglot import exp

from ffrwd import wasm
from ffrwd.compiler import Compiled, compile_all
from ffrwd.errors import ErrorCode, FfrwdError
from ffrwd.execute import CHAIN, PIPELINE, plan_argv, render_plan
from ffrwd.functions import WasmFunction, package_modules
from ffrwd.ir import Graph
from ffrwd.lower import lower
from ffrwd.parser import ModuleExport, Resolved, parse, resolve
from ffrwd.processes import (
    PIPE,
    RAWVIDEO,
    AudioFormat,
    FfmpegProcess,
    ModuleShape,
    PadMeta,
    SidecarProcess,
    StreamEdge,
    VideoFormat,
)
from ffrwd.project import PackageSet, discover
from ffrwd.registry import Registry, load_reference
from ffrwd.split import insert_splits
from ffrwd.wasm import WORLDS, Described, DescribedFunction

SNAPSHOT_PATH = Path(__file__).resolve().parent / "data" / "reference_registry.json"

MODULE = "modules/invert.wasm"

DECLARE = (
    "CREATE FUNCTION invert(v video_stream) RETURNS video_stream\n"
    f"  AS '{MODULE}', 'invert' LANGUAGE wasm;\n"
)
QUERY = DECLARE + "COPY (SELECT invert(f.video[1]) FROM input('a.mp4') f) TO 'out.mp4'"


@functools.cache
def _snapshot_registry() -> Registry:
    return load_reference(SNAPSHOT_PATH)


def _described(
    *,
    world: str = "ffrwd:av@0.3.0",
    name: str = "invert",
    params: dict[str, object] | None = None,
    pixel_formats: tuple[str, ...] = ("rgba",),
    window: int = 1,
) -> Described:
    """A synthetic description, shaped like what ``--describe`` prints."""
    return Described(
        world=world,
        name=name,
        version="0.1.0",
        params_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": dict(params or {}),
        },
        rows_schema=None,
        pixel_formats=pixel_formats,
        window=window,
        windowed=window != 1,
    )


def _resolved(sql: str) -> Resolved:
    return resolve(parse(sql))


def _lowered(
    sql: str, described: Described | None = None, module: str = MODULE
) -> object:
    graph = lower(
        _resolved(sql),
        {},
        registry=_snapshot_registry(),
        describes={module: described or _described()},
    )
    return insert_splits(graph)


def _rejects(
    sql: str,
    code: ErrorCode,
    needle: str,
    described: Described | None = None,
    module: str = MODULE,
) -> FfrwdError:
    """Compile `sql` far enough to fail, and pin the code and the wording."""
    with pytest.raises(FfrwdError) as caught:
        _lowered(sql, described, module)
    error = caught.value
    assert error.code is code, f"{error.code} != {code}: {error}"
    assert needle in error.message, error.message
    return error


def _compiled(sql: str, described: Described | None = None) -> Compiled:
    """A whole compile, with `describe` answering from memory."""
    return compile_all(sql, describe=lambda path: described or _described())


# -- the two-part AS, at parse time ---------------------------------------


def test_the_two_part_as_parses_as_a_module_and_an_export() -> None:
    tree = parse(
        "CREATE FUNCTION f(v video_stream) RETURNS video_stream "
        "AS 'm.wasm', 'blur' LANGUAGE wasm"
    )
    assert isinstance(tree, exp.Create)
    body = tree.args.get("expression")
    assert isinstance(body, ModuleExport)
    assert body.this.this == "m.wasm"
    assert body.expression.this == "blur"


def test_both_halves_of_the_two_part_as_carry_their_position() -> None:
    tree = parse(
        "CREATE FUNCTION f(v video_stream) RETURNS video_stream "
        "AS 'm.wasm', 'blur' LANGUAGE wasm"
    )
    assert isinstance(tree, exp.Create)
    body = tree.args.get("expression")
    assert isinstance(body, ModuleExport)
    assert body.this.meta["col"] < body.expression.meta["col"]


def test_a_heredoc_body_still_parses_as_a_heredoc() -> None:
    tree = parse("CREATE FUNCTION f(a text) RETURNS text AS $$ SELECT a $$ LANGUAGE sql")
    assert isinstance(tree, exp.Create)
    assert isinstance(tree.args.get("expression"), exp.Heredoc)


def test_a_single_quoted_body_still_parses_as_a_string() -> None:
    tree = parse("CREATE FUNCTION f(a text) RETURNS text AS 'SELECT a' LANGUAGE sql")
    assert isinstance(tree, exp.Create)
    body = tree.args.get("expression")
    assert isinstance(body, exp.Literal) and body.is_string


# -- the declaration ------------------------------------------------------


def test_a_declaration_rides_out_on_the_resolved_query() -> None:
    declared = _resolved(QUERY).wasm["invert"]
    assert declared == WasmFunction(
        name="invert",
        module=MODULE,
        export="invert",
        params=declared.params,
        returns="video_stream",
        line=declared.line,
        col=declared.col,
    )
    assert [(p.name, p.type) for p in declared.params] == [("v", "video_stream")]


def test_a_declaration_anchors_on_its_own_name() -> None:
    declared = _resolved(QUERY).wasm["invert"]
    assert (declared.line, declared.col) == (1, 17)


def test_a_query_with_no_module_declares_none() -> None:
    resolved = _resolved("COPY (SELECT f.video[1] FROM input('a.mp4') f) TO 'out.mp4'")
    assert resolved.wasm == {}


def test_a_stream_parameter_on_a_value_returning_wasm_function_is_refused() -> None:
    """A scalar RETURNS is wired now; a stream parameter beside it is not."""
    sql = (
        "CREATE FUNCTION m(v video_stream) RETURNS text AS 'm.wasm', 'm' LANGUAGE wasm;\n"
        "COPY (SELECT m(f.video[1]) FROM input('a.mp4') f) TO 'out.mp4'"
    )
    error = _rejects(sql, ErrorCode.UNSUPPORTED_SQL, "declares the parameter 'v' as video_stream")
    assert error.hint is not None and "text, number or boolean" in error.hint


def test_a_table_returning_wasm_function_is_refused() -> None:
    sql = (
        "CREATE FUNCTION m(v video_stream) RETURNS TABLE(t video_stream) "
        "AS 'm.wasm', 'm' LANGUAGE wasm;\n"
        "COPY (SELECT t.t FROM m(NULL) t) TO 'out.mp4'"
    )
    error = _rejects(sql, ErrorCode.UNSUPPORTED_SQL, "returns a table")
    assert error.hint is not None and "returns one video_stream" in error.hint


def test_a_wasm_function_needs_a_stream_parameter() -> None:
    sql = (
        "CREATE FUNCTION m() RETURNS video_stream AS 'm.wasm', 'm' LANGUAGE wasm;\n"
        "COPY (SELECT m() FROM input('a.mp4') f) TO 'out.mp4'"
    )
    _rejects(sql, ErrorCode.UNSUPPORTED_SQL, "takes no stream")


def test_the_stream_parameter_comes_first() -> None:
    sql = (
        "CREATE FUNCTION m(n number, v video_stream) RETURNS video_stream "
        "AS 'm.wasm', 'm' LANGUAGE wasm;\n"
        "COPY (SELECT m(1, f.video[1]) FROM input('a.mp4') f) TO 'out.mp4'"
    )
    _rejects(sql, ErrorCode.UNSUPPORTED_SQL, "takes number as its first parameter 'n'")


def test_a_second_stream_parameter_is_a_second_input() -> None:
    sql = (
        "CREATE FUNCTION m(v video_stream, w video_stream) RETURNS video_stream "
        "AS 'm.wasm', 'm' LANGUAGE wasm;\n"
        "COPY (SELECT m(f.video[1], f.video[1]) FROM input('a.mp4') f) TO 'out.mp4'"
    )
    declared = _resolved(sql).wasm["m"]
    assert [(p.name, p.type) for p in declared.stream_params] == [
        ("v", "video_stream"),
        ("w", "video_stream"),
    ]


def test_a_signature_mixing_the_two_stream_kinds_is_refused() -> None:
    sql = (
        "CREATE FUNCTION m(a audio_stream) RETURNS video_stream "
        "AS 'm.wasm', 'm' LANGUAGE wasm;\n"
        "COPY (SELECT m(f.audio[1]) FROM input('a.mp4') f) TO 'out.mp4'"
    )
    error = _rejects(
        sql, ErrorCode.UNSUPPORTED_SQL, "takes audio_stream and returns video_stream"
    )
    assert error.hint is not None and "one kind of stream" in error.hint


def test_a_signature_mixing_the_two_kinds_the_other_way_is_refused() -> None:
    sql = (
        "CREATE FUNCTION m(v video_stream) RETURNS audio_stream "
        "AS 'm.wasm', 'm' LANGUAGE wasm;\n"
        "COPY (SELECT m(f.video[1]) FROM input('a.mp4') f) TO 'out.mp4'"
    )
    _rejects(sql, ErrorCode.UNSUPPORTED_SQL, "takes video_stream and returns audio_stream")


def test_language_wasm_without_a_module_is_refused() -> None:
    sql = (
        "CREATE FUNCTION m(v video_stream) RETURNS video_stream "
        "AS $$ SELECT v $$ LANGUAGE wasm;\n"
        "COPY (SELECT m(f.video[1]) FROM input('a.mp4') f) TO 'out.mp4'"
    )
    _rejects(sql, ErrorCode.UNSUPPORTED_SQL, "names no module and export")


def test_a_module_and_export_under_language_sql_is_refused() -> None:
    sql = (
        "CREATE FUNCTION m(a text) RETURNS text AS 'm.wasm', 'm' LANGUAGE sql;\n"
        "SELECT m('x') AS m"
    )
    _rejects(sql, ErrorCode.UNSUPPORTED_SQL, "but says LANGUAGE sql")


def test_an_empty_module_path_is_refused() -> None:
    sql = (
        "CREATE FUNCTION m(v video_stream) RETURNS video_stream "
        "AS '', 'm' LANGUAGE wasm;\n"
        "COPY (SELECT m(f.video[1]) FROM input('a.mp4') f) TO 'out.mp4'"
    )
    _rejects(sql, ErrorCode.UNSUPPORTED_SQL, "empty module path")


def test_an_empty_export_is_refused() -> None:
    sql = (
        "CREATE FUNCTION m(v video_stream) RETURNS video_stream "
        "AS 'm.wasm', '' LANGUAGE wasm;\n"
        "COPY (SELECT m(f.video[1]) FROM input('a.mp4') f) TO 'out.mp4'"
    )
    _rejects(sql, ErrorCode.UNSUPPORTED_SQL, "empty export")


def test_a_wasm_declaration_may_not_share_a_name_with_a_sql_one() -> None:
    sql = (
        "CREATE FUNCTION m(a text) RETURNS text AS $$ SELECT a $$ LANGUAGE sql;\n"
        "CREATE FUNCTION m(v video_stream) RETURNS video_stream "
        "AS 'm.wasm', 'm' LANGUAGE wasm;\n"
        "COPY (SELECT m(f.video[1]) FROM input('a.mp4') f) TO 'out.mp4'"
    )
    _rejects(sql, ErrorCode.UNSUPPORTED_SQL, "is defined twice")


def test_an_uncalled_wasm_declaration_is_refused() -> None:
    sql = DECLARE + "COPY (SELECT f.video[1] FROM input('a.mp4') f) TO 'out.mp4'"
    error = _rejects(sql, ErrorCode.UNSUPPORTED_SQL, "is never called")
    assert (error.line, error.col) == (1, 17)


def test_a_wasm_call_before_its_declaration_is_refused() -> None:
    # A view rather than a COPY: a CREATE FUNCTION after a COPY is refused by
    # the earlier rule, which would hide this one.
    sql = (
        "CREATE VIEW v AS SELECT invert(f.video[1]) AS s FROM input('a.mp4') f;\n"
        + DECLARE
        + "COPY (SELECT x.s FROM v x) TO 'out.mp4'"
    )
    _rejects(sql, ErrorCode.UNSUPPORTED_SQL, "used before it is defined")


def test_a_wasm_call_after_its_declaration_is_accepted() -> None:
    sql = (
        DECLARE
        + "CREATE VIEW v AS SELECT invert(f.video[1]) AS s FROM input('a.mp4') f;\n"
        "COPY (SELECT x.s FROM v x) TO 'out.mp4'"
    )
    assert _lowered(sql) is not None


def test_a_wasm_call_in_from_is_refused() -> None:
    sql = DECLARE + "COPY (SELECT t.v FROM invert(NULL) t) TO 'out.mp4'"
    _rejects(sql, ErrorCode.UNSUPPORTED_SQL, "returns a stream, not a table")


# -- the call -------------------------------------------------------------


def test_too_many_arguments_is_refused() -> None:
    sql = DECLARE + (
        "COPY (SELECT invert(f.video[1], 2) FROM input('a.mp4') f) TO 'out.mp4'"
    )
    _rejects(sql, ErrorCode.UDF_ARG_TYPE, "got 2 arguments, but it declares 1")


def test_a_missing_argument_with_no_default_is_refused() -> None:
    sql = (
        "CREATE FUNCTION m(v video_stream, n number) RETURNS video_stream "
        f"AS '{MODULE}', 'invert' LANGUAGE wasm;\n"
        "COPY (SELECT m(f.video[1]) FROM input('a.mp4') f) TO 'out.mp4'"
    )
    _rejects(sql, ErrorCode.UDF_ARG_TYPE, "parameter 'n' has no DEFAULT")


def test_a_value_where_the_stream_goes_is_refused() -> None:
    sql = DECLARE + "COPY (SELECT invert('x') FROM input('a.mp4') f) TO 'out.mp4'"
    _rejects(sql, ErrorCode.UDF_ARG_TYPE, "takes video_stream as its 'v' argument")


def test_input_as_an_argument_is_refused() -> None:
    sql = DECLARE + (
        "COPY (SELECT invert(input('a.mp4')) FROM input('a.mp4') f) TO 'out.mp4'"
    )
    _rejects(sql, ErrorCode.UDF_ARG_TYPE, "cannot take input()")


def test_a_named_argument_is_refused() -> None:
    sql = (
        "CREATE FUNCTION m(v video_stream, n number DEFAULT 1) RETURNS video_stream "
        f"AS '{MODULE}', 'invert' LANGUAGE wasm;\n"
        "COPY (SELECT m(f.video[1], n => 2) FROM input('a.mp4') f) TO 'out.mp4'"
    )
    _rejects(
        sql,
        ErrorCode.UNSUPPORTED_SQL,
        "does not take named arguments",
        _described(params={"n": {"type": "number"}}),
    )


def test_an_audio_stream_argument_is_refused() -> None:
    sql = DECLARE + (
        "COPY (SELECT invert(f.audio[1]) FROM input('a.mp4') f) TO 'out.mp4'"
    )
    _rejects(sql, ErrorCode.UDF_ARG_TYPE, "got (audio)")


# -- what the module said -------------------------------------------------


def test_a_world_this_ffrwd_does_not_host_is_refused() -> None:
    error = _rejects(
        QUERY,
        ErrorCode.UNSUPPORTED_SQL,
        "targets ffrwd:av@9.9.9",
        _described(world="ffrwd:av@9.9.9"),
    )
    assert " or ".join(wasm.WORLDS) in error.message


def test_every_hosted_world_is_accepted() -> None:
    for world in wasm.WORLDS:
        assert _lowered(QUERY, _described(world=world)) is not None


def test_an_export_the_module_does_not_have_is_refused() -> None:
    error = _rejects(
        QUERY, ErrorCode.UNSUPPORTED_SQL, "names the export 'invert'", _described(name="blur")
    )
    assert "exports 'blur'" in error.message


def test_a_parameter_the_module_never_declared_is_refused() -> None:
    sql = (
        "CREATE FUNCTION m(v video_stream, n number) RETURNS video_stream "
        f"AS '{MODULE}', 'invert' LANGUAGE wasm;\n"
        "COPY (SELECT m(f.video[1], 2) FROM input('a.mp4') f) TO 'out.mp4'"
    )
    error = _rejects(sql, ErrorCode.UDF_ARG_TYPE, "has no parameter 'n'")
    assert error.hint == "the module declares no parameters"


def test_a_parameter_of_the_wrong_type_is_refused() -> None:
    sql = (
        "CREATE FUNCTION m(v video_stream, n text) RETURNS video_stream "
        f"AS '{MODULE}', 'invert' LANGUAGE wasm;\n"
        "COPY (SELECT m(f.video[1], 'loud') FROM input('a.mp4') f) TO 'out.mp4'"
    )
    _rejects(
        sql,
        ErrorCode.UDF_ARG_TYPE,
        "parameter 'n' is number",
        _described(params={"n": {"type": "number"}}),
    )


def test_a_written_parameter_reaches_the_node() -> None:
    sql = (
        "CREATE FUNCTION m(v video_stream, n number) RETURNS video_stream "
        f"AS '{MODULE}', 'invert' LANGUAGE wasm;\n"
        "COPY (SELECT m(f.video[1], 2) FROM input('a.mp4') f) TO 'out.mp4'"
    )
    graph = lower(
        _resolved(sql),
        {},
        registry=_snapshot_registry(),
        describes={MODULE: _described(params={"n": {"type": "number"}})},
    )
    node = next(n for n in graph.nodes.values() if n.filter == MODULE)
    assert node.args == {"n": 2}


def test_an_omitted_defaulted_parameter_is_left_to_the_module() -> None:
    sql = (
        "CREATE FUNCTION m(v video_stream, n number DEFAULT NULL) RETURNS video_stream "
        f"AS '{MODULE}', 'invert' LANGUAGE wasm;\n"
        "COPY (SELECT m(f.video[1]) FROM input('a.mp4') f) TO 'out.mp4'"
    )
    graph = lower(
        _resolved(sql),
        {},
        registry=_snapshot_registry(),
        describes={MODULE: _described(params={"n": {"type": "number"}})},
    )
    node = next(n for n in graph.nodes.values() if n.filter == MODULE)
    assert node.args == {}


def test_a_described_module_is_a_node_named_for_its_path() -> None:
    graph = lower(
        _resolved(QUERY), {}, registry=_snapshot_registry(), describes={MODULE: _described()}
    )
    node = next(n for n in graph.nodes.values() if n.filter == MODULE)
    assert node.outputs == ["video"]
    assert len(node.inputs) == 1


def test_a_module_that_was_never_described_is_a_compiler_bug() -> None:
    with pytest.raises(FfrwdError) as caught:
        lower(_resolved(QUERY), {}, registry=_snapshot_registry(), describes={})
    assert caught.value.code is ErrorCode.UNSUPPORTED_SQL
    assert "never described" in caught.value.message


# -- pixel formats --------------------------------------------------------


def test_the_wire_format_is_the_one_the_module_and_the_wire_agree_on() -> None:
    assert wasm.wire_pix_fmt(_described(pixel_formats=("rgba",))) == "rgba"
    assert wasm.wire_pix_fmt(_described(pixel_formats=("yuv420p",))) == "yuv420p"


def test_the_modules_own_order_decides_between_two_it_both_accepts() -> None:
    assert wasm.wire_pix_fmt(_described(pixel_formats=("yuv420p", "rgba"))) == "yuv420p"


def test_a_format_the_wire_cannot_carry_is_ignored() -> None:
    assert wasm.wire_pix_fmt(_described(pixel_formats=("gbrp16le", "rgba"))) == "rgba"


def test_no_overlap_names_both_lists() -> None:
    with pytest.raises(FfrwdError) as caught:
        wasm.wire_pix_fmt(_described(pixel_formats=("gbrp16le",)))
    message = caught.value.message
    assert "gbrp16le" in message
    for spelling in wasm.WIRE_PIX_FMTS:
        assert spelling in message


def test_a_module_with_no_overlap_is_refused_at_the_declaration() -> None:
    with pytest.raises(FfrwdError) as caught:
        _compiled(QUERY, _described(pixel_formats=("gbrp16le",)))
    error = caught.value
    assert error.message.startswith("function 'invert':")
    assert (error.line, error.col) == (1, 17)


def test_both_edges_around_a_module_carry_its_format() -> None:
    plan = _compiled(QUERY).plan
    assert plan is not None
    formats = [e.format for e in plan.stream_edges]
    assert formats and all(
        isinstance(f, VideoFormat) and f.pix_fmt == "rgba" for f in formats
    )


def test_a_module_wanting_yuv_moves_both_edges_to_yuv() -> None:
    plan = _compiled(QUERY, _described(pixel_formats=("yuv420p",))).plan
    assert plan is not None
    assert {e.format.pix_fmt for e in plan.stream_edges} == {"yuv420p"}


# -- the plan -------------------------------------------------------------


def test_a_query_naming_no_module_has_no_plan() -> None:
    compiled = compile_all("COPY (SELECT f.video[1] FROM input('a.mp4') f) TO 'out.mp4'")
    assert compiled.plan is None
    assert len(compiled.graphs) == 1


def test_a_query_naming_a_module_becomes_three_processes() -> None:
    plan = _compiled(QUERY).plan
    assert plan is not None
    assert len(plan.ffmpeg) == 2
    assert [p.module for p in plan.sidecars] == [MODULE]


def test_the_whole_plan_is_one_stage() -> None:
    plan = _compiled(QUERY).plan
    assert plan is not None
    assert len(plan.stages) == 1


def test_a_describe_failure_anchors_on_the_declaration() -> None:
    def refuse(path: str) -> Described:
        raise FfrwdError(ErrorCode.UNSUPPORTED_SQL, f"no such module {path}", hint="check it")

    with pytest.raises(FfrwdError) as caught:
        compile_all(QUERY, describe=refuse)
    error = caught.value
    assert error.message == f"function 'invert': no such module {MODULE}"
    assert (error.line, error.col) == (1, 17)
    assert error.hint == "check it"


def test_one_describe_per_module_path_however_many_declarations() -> None:
    sql = (
        f"CREATE FUNCTION a(v video_stream) RETURNS video_stream AS '{MODULE}', "
        "'invert' LANGUAGE wasm;\n"
        f"CREATE FUNCTION b(v video_stream) RETURNS video_stream AS '{MODULE}', "
        "'invert' LANGUAGE wasm;\n"
        "COPY (SELECT a(b(f.video[1])) FROM input('a.mp4') f) TO 'out.mp4'"
    )
    seen: list[str] = []

    def counting(path: str) -> Described:
        seen.append(path)
        return _described()

    compile_all(sql, describe=counting)
    assert seen == [MODULE]


# -- argv -----------------------------------------------------------------


def _sidecar(args: dict[str, object] | None = None) -> SidecarProcess:
    return SidecarProcess(
        id="sidecar0", module=MODULE, node="n1", args=dict(args or {}), outputs=("video",)
    )


def test_a_printed_sidecar_names_the_program_not_a_path() -> None:
    argv = wasm.shown_argv(_sidecar())
    assert argv == [
        "ffrwd-wasm", "-f", "nut", "-i", "pipe:0", "-m", MODULE, "-f", "nut", "pipe:1",
    ]


def test_parameters_travel_as_one_json_flag() -> None:
    argv = wasm.shown_argv(_sidecar({"strength": 2, "mode": "soft"}))
    assert argv[argv.index("-params") + 1] == '{"mode": "soft", "strength": 2}'


def test_a_module_with_no_parameters_gets_no_params_flag() -> None:
    assert "-params" not in wasm.shown_argv(_sidecar())


def test_a_missing_sidecar_is_the_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("ffrwd.wasm.binaries.ffrwd_wasm_path", lambda: None)
    with pytest.raises(FfrwdError) as caught:
        wasm.sidecar_argv(_sidecar())
    assert caught.value.hint is not None
    assert "reinstall ffrwd" in caught.value.hint
    assert "FFRWD_WASM" in caught.value.hint


def test_a_missing_sidecar_refuses_the_describe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("ffrwd.wasm.binaries.ffrwd_wasm_path", lambda: None)
    with pytest.raises(FfrwdError) as caught:
        wasm.describe(MODULE)
    assert "not installed" in caught.value.message
    assert caught.value.hint is not None and "reinstall ffrwd" in caught.value.hint


def test_the_plan_renders_as_one_shell_pipeline() -> None:
    plan = _compiled(QUERY).plan
    assert plan is not None
    shown = render_plan(plan, sidecar_argv=wasm.shown_argv)
    assert shown.count(" | ") == 2
    assert "&&" not in shown
    assert f"-m {MODULE}" in shown


def test_the_sidecar_reads_stdin_and_writes_stdout() -> None:
    plan = _compiled(QUERY).plan
    assert plan is not None
    argv = plan_argv(plan, sidecar_argv=wasm.shown_argv)["sidecar0"]
    assert argv[argv.index("-i") + 1] == "pipe:0"
    assert argv[-1] == "pipe:1"


def test_the_producing_ffmpeg_writes_the_negotiated_format() -> None:
    plan = _compiled(QUERY).plan
    assert plan is not None
    feeder = next(e.source for e in plan.stream_edges if e.target == "sidecar0")
    argv = plan_argv(plan, sidecar_argv=wasm.shown_argv)[feeder]
    assert "rgba" in argv
    assert argv[argv.index("-f") + 1] == "nut"


def test_one_module_keeps_the_short_spelling() -> None:
    """No network string, no name binding: what today's sidecar already reads."""
    plan = _compiled(QUERY).plan
    assert plan is not None
    assert wasm.shown_argv(plan.sidecars[0]) == [
        "ffrwd-wasm",
        "-f",
        "nut",
        "-i",
        "pipe:0",
        "-m",
        MODULE,
        "-f",
        "nut",
        "pipe:1",
    ]


def test_a_region_of_two_modules_is_run_as_a_network() -> None:
    plan = _annotated_plan(_pair()).plan
    assert plan is not None
    assert wasm.shown_argv(plan.sidecars[0]) == [
        "ffrwd-wasm",
        "-f",
        "nut",
        "-i",
        "pipe:0",
        "-m",
        f"facebox={DETECTOR}",
        "-m",
        f"blur_boxes={BLURRER}",
        "-filter_complex",
        "[0:v]facebox[n1];[n1]blur_boxes[out0]",
        "-map",
        "[out0]",
        "-f",
        "nut",
        "pipe:1",
    ]


def test_a_networks_node_carries_its_own_parameters() -> None:
    """A module's arguments are written into the network string, not -params."""
    sql = f"""CREATE FUNCTION detect_faces(v video_stream, size number)
RETURNS STRUCT(v video_stream, faces {BOX})
  AS '{DETECTOR}', 'facebox' LANGUAGE wasm;
CREATE FUNCTION blur_boxes(v video_stream, faces {BOX}, radius number)
RETURNS video_stream AS '{BLURRER}', 'blur-boxes' LANGUAGE wasm;
COPY (SELECT blur_boxes(detect_faces(f.video[1], 60), 12)
      FROM input('a.mp4') f) TO 'out.mp4'"""
    plan = _annotated_plan(
        sql,
        detector=_annotating(params={"size": {"type": "integer"}}),
        blurrer=_consuming(params={"radius": {"type": "integer"}}),
    ).plan
    assert plan is not None
    argv = wasm.shown_argv(plan.sidecars[0])
    assert argv[argv.index("-filter_complex") + 1] == (
        "[0:v]facebox=size=60[n1];[n1]blur_boxes=radius=12[out0]"
    )
    assert "-params" not in argv


def test_the_networks_string_is_what_the_pipeline_prints() -> None:
    plan = _annotated_plan(_pair()).plan
    assert plan is not None
    shown = render_plan(plan, sidecar_argv=wasm.shown_argv)
    assert shown.count(" | ") == 2
    assert "'[0:v]facebox[n1];[n1]blur_boxes[out0]'" in shown


def test_explain_carries_the_regions_modules_and_its_latency() -> None:
    plan = _annotated_plan(_pair()).plan
    assert plan is not None
    described = plan.sidecars[0].to_dict()
    assert described["modules"] == [
        {"name": "facebox", "path": DETECTOR},
        {"name": "blur_boxes", "path": BLURRER},
    ]
    assert described["lookahead"] == 0
    graph = described["graph"]
    assert isinstance(graph, dict)
    assert [n["filter"] for n in graph["nodes"]] == ["facebox", "blur_boxes"]


def test_a_partition_rejection_is_anchored_on_the_declaration() -> None:
    """Partitioning knows a module by path; the declaration knows where it is.

    Nothing the dialect spells today reaches this -- a multi-stream module
    call has no syntax yet -- so the re-anchoring is pinned on its own.
    """
    from ffrwd.compiler import _anchored

    declared = _resolved(QUERY).wasm["invert"]
    anchored = _anchored(
        FfrwdError(
            ErrorCode.UNSUPPORTED_SQL,
            f"the module '{MODULE}' reads several streams",
            hint="feed them from one stream",
        ),
        {"invert": declared},
    )
    assert anchored.message.startswith("function 'invert': ")
    assert (anchored.line, anchored.col) == (declared.line, declared.col)
    assert anchored.hint == "feed them from one stream"


# -- what a module declares about its frame timing -------------------------


def test_a_description_with_no_shape_is_one_frame_in_one_frame_out() -> None:
    described = wasm._described(MODULE, {"world": "ffrwd:av@0.5.0", "name": "invert"})
    assert (described.window, described.stride) == (1, 1)
    assert described.pure and described.one_to_one
    assert not described.windowed


def test_a_windowed_description_reads_its_declared_shape() -> None:
    described = wasm._described(
        MODULE,
        {
            "world": "ffrwd:av@0.5.0",
            "name": "tail3",
            "window": 3,
            "stride": 3,
            "pure": True,
            "one_to_one": True,
        },
    )
    assert described.windowed
    assert described.shape == ModuleShape(window=3, stride=3, one_to_one=True)
    assert described.shape.lookahead == 2


def test_a_windowed_module_is_not_read_as_an_annotation_consumer() -> None:
    """It reports `meta` because rows ride its calls, not because it reads any."""
    described = wasm._described(
        MODULE,
        {"world": "ffrwd:av@0.5.0", "name": "double", "meta": True, "one_to_one": False},
    )
    assert described.meta
    assert not described.reads_annotations
    assert not described.shape.one_to_one


def test_a_windowed_reader_may_be_declared_without_an_annotation_column() -> None:
    """It is handed each frame's rows either way; reading them is its option."""
    described = replace(
        _described(world="ffrwd:av@0.5.0"),
        pure=False,
        windowed=True,
        reads_rows=True,
    )
    assert described.reads_annotations
    _lowered(QUERY, described)


def test_a_plain_meta_module_still_reads_annotations() -> None:
    described = wasm._described(
        MODULE, {"world": "ffrwd:av@0.4.0", "name": "blur-boxes", "meta": True}
    )
    assert described.reads_annotations


def test_a_declared_window_reaches_the_plan_as_latency() -> None:
    plan = _compiled(
        QUERY, _described(name="invert", window=4)
    ).plan
    assert plan is not None
    assert plan.sidecars[0].lookahead == 3


# -- reading a description ------------------------------------------------


def test_a_description_that_is_not_an_object_is_refused() -> None:
    with pytest.raises(FfrwdError) as caught:
        wasm._described(MODULE, ["not", "an", "object"])
    assert "not an object" in caught.value.message


def test_a_description_missing_its_world_is_refused() -> None:
    with pytest.raises(FfrwdError) as caught:
        wasm._described(MODULE, {"name": "invert"})
    assert "names no world" in caught.value.message


def test_a_description_reads_every_field_it_is_given() -> None:
    described = wasm._described(
        MODULE,
        {
            "world": "ffrwd:av@0.3.0",
            "name": "invert",
            "version": "0.1.0",
            "params_schema": {"type": "object", "properties": {}},
            "rows_schema": None,
            "pixel_formats": ["rgba"],
        },
    )
    assert described.world == "ffrwd:av@0.3.0"
    assert described.name == "invert"
    assert described.version == "0.1.0"
    assert described.rows_schema is None
    assert described.pixel_formats == ("rgba",)


def test_a_module_beside_an_ffmpeg_filter_keeps_both() -> None:
    sql = DECLARE + (
        "COPY (SELECT hflip(invert(f.video[1])) FROM input('a.mp4') f) TO 'out.mp4'"
    )
    plan = _compiled(sql).plan
    assert plan is not None
    assert [p.module for p in plan.sidecars] == [MODULE]
    filters = [
        node.filter
        for process in plan.ffmpeg
        for node in process.graph.nodes.values()
    ]
    assert "hflip" in filters
    assert MODULE not in filters


def test_sibling_legs_of_one_merge_share_a_decode() -> None:
    """A masked blur: the passthrough and blur legs are one feeder, the
    module's leg its own -- two decodes of the input, not three."""
    sql = DECLARE + (
        "COPY (SELECT maskedmerge(f.video[1], gblur(f.video[1], 12),"
        " invert(f.video[1]))\n"
        "FROM input('a.mp4') f) TO 'out.mp4'"
    )
    plan = _compiled(sql).plan
    assert plan is not None
    assert len(plan.ffmpeg) == 3
    assert len(plan.sidecars) == 1

    merge = next(
        p
        for p in plan.ffmpeg
        if any(n.filter == "maskedmerge" for n in p.graph.nodes.values())
    )
    shared = next(
        p for p in plan.ffmpeg if sum(u.path == PIPE for u in p.graph.sinks) == 2
    )
    filters = [n.filter for n in shared.graph.nodes.values()]
    assert sorted(filters) == ["gblur", "split"]
    assert {e.source for e in plan.stream_edges if e.target == merge.id} == {
        shared.id,
        "sidecar0",
    }
    # The two branches render as one argv with two nut outputs.
    argv = plan_argv(
        plan,
        sidecar_argv=wasm.shown_argv,
        pipe_path=lambda edge, side: f"pipes/{edge.source}-{edge.target}-{edge.ref}-{side}",
    )
    assert argv[shared.id].count("nut") == 2


# -- typed frame annotations ----------------------------------------------

DETECTOR = "modules/facebox.wasm"
BLURRER = "modules/blur_boxes.wasm"
BOX = "STRUCT(x number, y number, w number, h number)[]"

_BOX_ROWS: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["x", "y", "w", "h"],
    "properties": {
        "x": {"type": "integer"},
        "y": {"type": "integer"},
        "w": {"type": "integer"},
        "h": {"type": "integer"},
    },
}


def _annotating(
    rows: dict[str, object] | None = None,
    name: str = "facebox",
    params: dict[str, object] | None = None,
) -> Described:
    """A module that emits rows, as ``--describe`` reports one."""
    return Described(
        world="ffrwd:av@0.4.0",
        name=name,
        version="0.1.0",
        params_schema={"type": "object", "properties": dict(params or {})},
        rows_schema=_BOX_ROWS if rows is None else rows,
        pixel_formats=("yuv420p", "rgba"),
    )


def _consuming(
    name: str = "blur-boxes",
    *,
    meta: bool = True,
    params: dict[str, object] | None = None,
) -> Described:
    """A module that reads rows, its describe marked accordingly."""
    return Described(
        world="ffrwd:av@0.4.0",
        name=name,
        version="0.1.0",
        params_schema={"type": "object", "properties": dict(params or {})},
        rows_schema=None,
        pixel_formats=("yuv420p", "rgba"),
        meta=meta,
    )


def _pair(
    detect_returns: str = f"STRUCT(v video_stream, faces {BOX})",
    blur_takes: str = BOX,
    body: str = "blur_boxes(detect_faces(f.video[1]))",
) -> str:
    return (
        f"CREATE FUNCTION detect_faces(v video_stream) RETURNS {detect_returns}\n"
        f"  AS '{DETECTOR}', 'facebox' LANGUAGE wasm;\n"
        f"CREATE FUNCTION blur_boxes(v video_stream, faces {blur_takes})\n"
        f"RETURNS video_stream AS '{BLURRER}', 'blur-boxes' LANGUAGE wasm;\n"
        f"COPY (SELECT {body} FROM input('a.mp4') f) TO 'out.mp4'"
    )


def _annotated_plan(
    sql: str, detector: Described | None = None, blurrer: Described | None = None
) -> Compiled:
    described = {
        DETECTOR: detector or _annotating(),
        BLURRER: blurrer or _consuming(),
    }
    return compile_all(sql, describe=lambda path: described[path])


def _annotation_rejects(
    sql: str,
    code: ErrorCode,
    needle: str,
    detector: Described | None = None,
    blurrer: Described | None = None,
) -> FfrwdError:
    with pytest.raises(FfrwdError) as caught:
        _annotated_plan(sql, detector, blurrer)
    error = caught.value
    assert error.code is code, f"{error.code} != {code}: {error}"
    assert needle in error.message, error.message
    assert error.line is not None and error.hint
    return error


# the declaration


def test_a_struct_return_declares_a_stream_and_an_annotation() -> None:
    declared = _resolved(_pair()).wasm["detect_faces"]
    assert declared.returns == "video_stream"
    assert declared.stream_field == "v"
    assert declared.emits is not None
    assert [(f.name, f.type) for f in declared.emits.fields] == [
        ("x", "number"),
        ("y", "number"),
        ("w", "number"),
        ("h", "number"),
    ]
    assert declared.reads is None


def test_an_annotation_parameter_is_not_a_value_parameter() -> None:
    declared = _resolved(_pair()).wasm["blur_boxes"]
    assert declared.reads is not None and declared.reads.name == "faces"
    assert declared.value_params == ()
    assert [p.name for p in declared.written_params] == ["v"]


def test_a_signature_reads_back_with_its_annotation_column() -> None:
    declared = _resolved(_pair()).wasm["detect_faces"]
    assert declared.signature == (
        "detect_faces(v video_stream) RETURNS STRUCT(v video_stream, "
        "faces STRUCT(x number, y number, w number, h number)[])"
    )


def test_value_parameters_still_follow_an_annotation_column() -> None:
    sql = _pair(blur_takes=f"{BOX}, radius number DEFAULT 8")
    declared = _resolved(sql).wasm["blur_boxes"]
    assert [p.name for p in declared.value_params] == ["radius"]
    assert [p.name for p in declared.written_params] == ["v", "radius"]


def test_a_struct_return_of_the_wrong_width_is_refused() -> None:
    sql = _pair(detect_returns=f"STRUCT(v video_stream, a {BOX}, b {BOX})")
    _annotation_rejects(sql, ErrorCode.UNSUPPORTED_SQL, "struct of 3 fields")


def test_a_struct_return_with_no_stream_field_is_refused() -> None:
    sql = _pair(detect_returns=f"STRUCT(a number, faces {BOX})")
    error = _annotation_rejects(
        sql, ErrorCode.UNSUPPORTED_SQL, "returns the field 'a' as 'number'"
    )
    assert error.hint is not None and "one stream field" in error.hint


def test_a_struct_return_whose_second_field_is_not_an_array_is_refused() -> None:
    sql = _pair(detect_returns="STRUCT(v video_stream, faces number)")
    _annotation_rejects(sql, ErrorCode.UNSUPPORTED_SQL, "returns the field 'faces'")


def test_an_annotation_field_that_is_a_stream_is_refused() -> None:
    sql = _pair(detect_returns="STRUCT(v video_stream, faces STRUCT(s video_stream)[])")
    error = _annotation_rejects(
        sql, ErrorCode.UNSUPPORTED_SQL, "the field 's' of 'faces' as 'video_stream'"
    )
    assert error.hint is not None and "boolean, number, text" in error.hint


def test_an_annotation_column_after_a_value_parameter_is_refused() -> None:
    sql = (
        f"CREATE FUNCTION b(v video_stream, n number, faces {BOX})\n"
        f"RETURNS video_stream AS '{BLURRER}', 'blur-boxes' LANGUAGE wasm;\n"
        "COPY (SELECT b(f.video[1], 3) FROM input('a.mp4') f) TO 'out.mp4'"
    )
    _annotation_rejects(sql, ErrorCode.UNSUPPORTED_SQL, "'faces' in position 3")


# the module's own schema


def test_an_annotation_record_matches_the_rows_the_module_emits() -> None:
    assert _annotated_plan(_pair()).plan is not None


def test_an_integer_column_fits_a_number_field() -> None:
    """`number` is the dialect's one numeric type, so it covers integer."""
    rows: dict[str, object] = {
        "type": "object",
        "properties": {"n": {"type": "integer"}},
    }
    sql = _pair(
        detect_returns="STRUCT(v video_stream, hits STRUCT(n number)[])",
        blur_takes="STRUCT(n number)[]",
    )
    assert _annotated_plan(sql, detector=_annotating(rows)).plan is not None


def test_a_misspelled_annotation_field_is_refused() -> None:
    sql = _pair(
        detect_returns="STRUCT(v video_stream, faces STRUCT(left number, y number, "
        "w number, h number)[])",
        blur_takes="STRUCT(left number, y number, w number, h number)[]",
    )
    error = _annotation_rejects(sql, ErrorCode.UDF_ARG_TYPE, "declares 'faces' as")
    assert "x (integer)" in error.message


def test_an_annotation_field_of_the_wrong_type_is_refused() -> None:
    sql = _pair(
        detect_returns="STRUCT(v video_stream, faces STRUCT(x text, y number, "
        "w number, h number)[])",
        blur_takes="STRUCT(x text, y number, w number, h number)[]",
    )
    _annotation_rejects(sql, ErrorCode.UDF_ARG_TYPE, "declares 'faces' as")


def test_a_module_emitting_no_rows_cannot_declare_an_annotation_return() -> None:
    error = _annotation_rejects(
        _pair(),
        ErrorCode.UNSUPPORTED_SQL,
        "emits no rows",
        detector=_consuming(name="facebox"),
    )
    assert error.hint is not None and "RETURNS video_stream" in error.hint


# composition at the call site


def test_a_reader_over_a_plain_stream_is_refused() -> None:
    sql = (
        f"CREATE FUNCTION blur_boxes(v video_stream, faces {BOX})\n"
        f"RETURNS video_stream AS '{BLURRER}', 'blur-boxes' LANGUAGE wasm;\n"
        "COPY (SELECT blur_boxes(f.video[1]) FROM input('a.mp4') f) TO 'out.mp4'"
    )
    error = _annotation_rejects(
        sql, ErrorCode.UDF_ARG_TYPE, "its argument produces none"
    )
    assert error.hint is not None and "DEFAULT NULL" in error.hint


def test_a_plain_module_over_a_producer_is_refused() -> None:
    sql = (
        f"CREATE FUNCTION detect_faces(v video_stream)\n"
        f"RETURNS STRUCT(v video_stream, faces {BOX})\n"
        f"  AS '{DETECTOR}', 'facebox' LANGUAGE wasm;\n"
        f"CREATE FUNCTION plain(v video_stream) RETURNS video_stream\n"
        f"  AS '{BLURRER}', 'blur-boxes' LANGUAGE wasm;\n"
        "COPY (SELECT plain(detect_faces(f.video[1])) FROM input('a.mp4') f) "
        "TO 'out.mp4'"
    )
    # `plain` truly takes no annotations here (meta=False): the mismatch this
    # pins is the ARGUMENT one, not the declaration-vs-module one.
    _annotation_rejects(
        sql,
        ErrorCode.UDF_ARG_TYPE,
        "plain() takes video_stream",
        blurrer=_consuming(meta=False),
    )


def _windowed_reader(name: str = "blur-boxes") -> Described:
    """A windowed module that reads rows at its own option."""
    return replace(_consuming(name=name), windowed=True, pure=False, reads_rows=True)


def _optional_reader(body: str) -> str:
    return (
        f"CREATE FUNCTION blur_boxes(v video_stream, faces {BOX} DEFAULT NULL)\n"
        f"RETURNS video_stream AS '{BLURRER}', 'blur-boxes' LANGUAGE wasm;\n"
        f"COPY (SELECT {body} FROM input('a.mp4') f) TO 'out.mp4'"
    )


def test_an_optional_annotation_column_may_go_unfilled() -> None:
    """DEFAULT NULL on the column lets the call ride a plain stream."""
    _annotated_plan(
        _optional_reader("blur_boxes(f.video[1])"), blurrer=_windowed_reader()
    )


def test_an_optional_annotation_column_still_takes_a_producer() -> None:
    _annotated_plan(
        _pair(blur_takes=f"{BOX} DEFAULT NULL"), blurrer=_windowed_reader()
    )


def test_a_non_null_default_on_an_annotation_column_is_refused() -> None:
    sql = (
        f"CREATE FUNCTION blur_boxes(v video_stream, faces {BOX} DEFAULT ARRAY[])\n"
        f"RETURNS video_stream AS '{BLURRER}', 'blur-boxes' LANGUAGE wasm;\n"
        "COPY (SELECT blur_boxes(f.video[1]) FROM input('a.mp4') f) TO 'out.mp4'"
    )
    error = _annotation_rejects(
        sql, ErrorCode.UNSUPPORTED_SQL, "a DEFAULT", blurrer=_windowed_reader()
    )
    assert error.hint is not None and "DEFAULT NULL" in error.hint


def test_a_per_frame_consumer_cannot_default_its_annotation_column() -> None:
    """The option belongs to a windowed reader; a per-frame one always consumes."""
    _annotation_rejects(
        _optional_reader("blur_boxes(f.video[1])"),
        ErrorCode.UNSUPPORTED_SQL,
        "reads rows on every frame",
    )


def test_a_producer_nothing_reads_is_refused() -> None:
    sql = (
        f"CREATE FUNCTION detect_faces(v video_stream)\n"
        f"RETURNS STRUCT(v video_stream, faces {BOX})\n"
        f"  AS '{DETECTOR}', 'facebox' LANGUAGE wasm;\n"
        "COPY (SELECT detect_faces(f.video[1]) FROM input('a.mp4') f) TO 'out.mp4'"
    )
    error = _annotation_rejects(sql, ErrorCode.UDF_ARG_TYPE, "nothing here reads it")
    assert error.hint is not None and "a struct is not a stream" in error.hint


def test_two_annotation_records_that_disagree_are_refused() -> None:
    sql = _pair(blur_takes="STRUCT(x number, y number)[]")
    error = _annotation_rejects(
        sql, ErrorCode.UDF_ARG_TYPE, "blur_boxes() takes 'faces' as"
    )
    assert error.hint is not None and "the same fields" in error.hint


def test_field_order_between_the_two_records_does_not_matter() -> None:
    """The rows travel keyed by name, so the two spellings are one type."""
    sql = _pair(blur_takes="STRUCT(h number, w number, y number, x number)[]")
    assert _annotated_plan(sql).plan is not None


# the plan


def test_the_composed_pair_is_three_processes() -> None:
    """Adjacent modules contract: one decode, one network, one mux."""
    plan = _annotated_plan(_pair()).plan
    assert plan is not None
    assert len(plan.processes) == 3
    assert [s.module for s in plan.sidecars] == [DETECTOR]
    assert [b.path for b in plan.sidecars[0].modules] == [DETECTOR, BLURRER]


def test_the_pair_hands_its_rows_over_inside_one_process() -> None:
    plan = _annotated_plan(_pair()).plan
    assert plan is not None
    # Nothing between the two modules is an edge at all, annotated or not.
    assert not any(e.annotations for e in plan.stream_edges)
    region = plan.sidecars[0]
    assert (region.reads_rows, region.writes_rows) == (False, False)
    assert len(region.nodes) == 2


def test_the_contracted_pair_is_run_with_no_annotations_at_all() -> None:
    plan = _annotated_plan(_pair()).plan
    assert plan is not None
    assert "-annotations" not in wasm._argv("ffrwd-wasm", plan.sidecars[0])


def test_a_lone_module_is_run_with_no_annotations_at_all() -> None:
    plan = _compiled(QUERY).plan
    assert plan is not None
    assert "-annotations" not in wasm._argv("ffrwd-wasm", plan.sidecars[0])


def test_the_consuming_node_alone_is_marked_in_the_graph() -> None:
    graph = lower(
        _resolved(_pair()),
        {},
        registry=_snapshot_registry(),
        describes={DETECTOR: _annotating(), BLURRER: _consuming()},
    )
    marked = [n.filter for n in graph.nodes.values() if n.reads_annotations]
    assert marked == [BLURRER]


def test_a_marked_edge_says_so_when_written_out() -> None:
    """An edge between two processes, which annotations reach only across one."""
    edge = StreamEdge(
        source="sidecar0",
        target="sidecar1",
        ref="n0",
        format=VideoFormat(),
        annotations=True,
    )
    assert edge.to_dict()["annotations"] is True
    assert "annotations" not in replace(edge, annotations=False).to_dict()

def test_explain_carries_the_plan_beside_the_graph() -> None:
    """A query that partitions into processes explains its plan too."""
    from ffrwd.mcp.tools import explain_query

    result = explain_query(QUERY, describe=lambda path: _described())
    assert "plan" in result
    plan = result["plan"]
    assert isinstance(plan, dict)
    assert {"processes", "edges", "stages"} <= plan.keys()
    assert len(plan["processes"]) >= 2


def test_explain_on_a_plain_query_carries_no_plan() -> None:
    from ffrwd.mcp.tools import explain_query

    result = explain_query(
        "COPY (SELECT gblur(a.video[1], 5) FROM input('x.mp4') a) TO 'o.mp4'"
    )
    assert "plan" not in result


# ---------------------------------------------------------------------------
# audio modules
# ---------------------------------------------------------------------------
#
# Same wiring as a video module, one kind over: sample formats where pixel
# formats were, pcm where rawvideo was, and `0:a:0` where `0:v:0` was. What is
# new is conformance -- a module may name the rates and channel counts it
# accepts, and the producing ffmpeg is told to write the first of each.

AUDIO_MODULE = "modules/denoise.wasm"
AUDIO_DECLARE = (
    "CREATE FUNCTION denoise(a audio_stream) RETURNS audio_stream\n"
    f"  AS '{AUDIO_MODULE}', 'denoise' LANGUAGE wasm;\n"
)
AUDIO_QUERY = AUDIO_DECLARE + (
    "COPY (SELECT denoise(f.audio[1]) FROM input('a.mp4') f) TO 'out.m4a'"
)


def _audio_described(
    *,
    name: str = "denoise",
    sample_formats: tuple[str, ...] = ("f32",),
    sample_rates: tuple[int, ...] = (),
    channel_counts: tuple[int, ...] = (),
    pixel_formats: tuple[str, ...] = (),
) -> Described:
    """A synthetic description of a module that filters audio."""
    return Described(
        world="ffrwd:av@0.7.0",
        name=name,
        version="0.1.0",
        params_schema={"type": "object", "properties": {}},
        rows_schema=None,
        pixel_formats=pixel_formats,
        sample_formats=sample_formats,
        sample_rates=sample_rates,
        channel_counts=channel_counts,
    )


def _audio_plan(described: Described | None = None, sql: str = AUDIO_QUERY) -> Compiled:
    return compile_all(sql, describe=lambda path: described or _audio_described())


def _feeder_argv(compiled: Compiled) -> list[str]:
    """The argv of the ffmpeg process that writes into the sidecar."""
    plan = compiled.plan
    assert plan is not None
    feeder = next(e.source for e in plan.stream_edges if e.target == "sidecar0")
    return plan_argv(plan, sidecar_argv=wasm.shown_argv)[feeder]


# the description


def test_a_module_naming_sample_formats_filters_audio() -> None:
    assert _audio_described().kind == "audio"
    assert _described().kind == "video"


def test_a_module_naming_neither_list_filters_neither() -> None:
    assert _audio_described(sample_formats=()).kind is None


def test_the_audio_fields_are_read_off_the_describe_payload() -> None:
    described = wasm._described(
        AUDIO_MODULE,
        {
            "world": "ffrwd:av@0.7.0",
            "name": "denoise",
            "sample_formats": ["f32", "s16"],
            "sample_rates": [16000, 48000],
            "channel_counts": [1, 2],
        },
    )
    assert described.sample_formats == ("f32", "s16")
    assert described.sample_rates == (16000, 48000)
    assert described.channel_counts == (1, 2)


def test_a_describe_naming_no_audio_fields_leaves_them_empty() -> None:
    described = wasm._described(
        AUDIO_MODULE, {"world": "ffrwd:av@0.7.0", "name": "denoise"}
    )
    assert described.sample_formats == ()
    assert described.sample_rates == ()
    assert described.channel_counts == ()


def test_the_newest_world_is_hosted() -> None:
    assert "ffrwd:av@0.7.0" in wasm.WORLDS


# sample formats


def test_the_sample_format_is_the_one_the_module_and_the_wire_agree_on() -> None:
    assert wasm.wire_sample_fmt(_audio_described(sample_formats=("f32",))) == "f32"
    assert wasm.wire_sample_fmt(_audio_described(sample_formats=("s16",))) == "s16"


def test_the_modules_own_order_decides_between_two_sample_formats() -> None:
    assert wasm.wire_sample_fmt(_audio_described(sample_formats=("s16", "f32"))) == "s16"


def test_a_sample_format_the_wire_cannot_carry_is_ignored() -> None:
    assert wasm.wire_sample_fmt(_audio_described(sample_formats=("s24", "f32"))) == "f32"


def test_no_sample_format_overlap_names_both_lists() -> None:
    with pytest.raises(FfrwdError) as caught:
        wasm.wire_sample_fmt(_audio_described(sample_formats=("s24",)))
    message = caught.value.message
    assert "s24" in message
    for spelling in wasm.WIRE_SAMPLE_FMTS:
        assert spelling in message


def test_an_audio_module_with_no_overlap_is_refused_at_the_declaration() -> None:
    with pytest.raises(FfrwdError) as caught:
        _audio_plan(_audio_described(sample_formats=("s24",)))
    error = caught.value
    assert error.message.startswith("function 'denoise':")
    assert (error.line, error.col) == (1, 17)


# the kind a declaration and a module have to agree on


def test_a_module_naming_both_format_lists_is_refused() -> None:
    error = _rejects(
        AUDIO_QUERY,
        ErrorCode.UNSUPPORTED_SQL,
        f"the module '{AUDIO_MODULE}' accepts both pixel formats and sample formats",
        _audio_described(pixel_formats=("rgba",)),
        module=AUDIO_MODULE,
    )
    assert error.hint is not None and "video or audio" in error.hint


def test_an_audio_signature_over_a_video_module_is_refused() -> None:
    error = _rejects(
        AUDIO_QUERY,
        ErrorCode.UNSUPPORTED_SQL,
        "takes audio_stream, and the module "
        f"'{AUDIO_MODULE}' filters video",
        _audio_described(sample_formats=(), pixel_formats=("rgba",)),
        module=AUDIO_MODULE,
    )
    assert error.hint is not None and "video_stream" in error.hint


def test_a_video_signature_over_an_audio_module_is_refused() -> None:
    error = _rejects(
        QUERY,
        ErrorCode.UNSUPPORTED_SQL,
        f"takes video_stream, and the module '{MODULE}' filters audio",
        _audio_described(name="invert"),
    )
    assert error.hint is not None and "audio_stream" in error.hint


# the chain


def test_an_audio_chain_becomes_three_processes() -> None:
    plan = _audio_plan().plan
    assert plan is not None
    assert len(plan.ffmpeg) == 2
    assert [p.module for p in plan.sidecars] == [AUDIO_MODULE]


def test_the_feeding_ffmpeg_maps_the_audio_track_onto_a_pcm_edge() -> None:
    assert _feeder_argv(_audio_plan()) == [
        "ffmpeg",
        "-i",
        "a.mp4",
        "-map",
        "0:a:0",
        "-c:0",
        "pcm_f32le",
        "-f",
        "nut",
        "pipe:1",
    ]


def test_both_edges_around_an_audio_module_carry_its_pcm() -> None:
    plan = _audio_plan().plan
    assert plan is not None
    formats = [e.format for e in plan.stream_edges]
    assert formats and all(
        isinstance(f, AudioFormat) and f.codec == "pcm_f32le" for f in formats
    )


def test_a_module_wanting_s16_moves_both_edges_to_s16() -> None:
    plan = _audio_plan(_audio_described(sample_formats=("s16",))).plan
    assert plan is not None
    assert {e.format.codec for e in plan.stream_edges} == {"pcm_s16le"}


def test_an_s16_module_is_fed_pcm_s16le() -> None:
    argv = _feeder_argv(_audio_plan(_audio_described(sample_formats=("s16",))))
    assert argv[argv.index("-c:0") + 1] == "pcm_s16le"


# conformance


def test_a_declared_rate_and_channel_count_conform_the_feeding_stream() -> None:
    argv = _feeder_argv(
        _audio_plan(_audio_described(sample_rates=(16000,), channel_counts=(1,)))
    )
    assert argv == [
        "ffmpeg",
        "-i",
        "a.mp4",
        "-map",
        "0:a:0",
        "-ar:0",
        "16000",
        "-ac:0",
        "1",
        "-c:0",
        "pcm_f32le",
        "-f",
        "nut",
        "pipe:1",
    ]


def test_the_first_acceptable_rate_and_count_win() -> None:
    argv = _feeder_argv(
        _audio_plan(
            _audio_described(sample_rates=(48000, 16000), channel_counts=(2, 1))
        )
    )
    assert argv[argv.index("-ar:0") + 1] == "48000"
    assert argv[argv.index("-ac:0") + 1] == "2"


def test_a_module_constraining_neither_leaves_the_stream_alone() -> None:
    argv = _feeder_argv(_audio_plan())
    assert "-ar:0" not in argv
    assert "-ac:0" not in argv


def test_a_module_constraining_only_the_rate_says_nothing_about_channels() -> None:
    argv = _feeder_argv(_audio_plan(_audio_described(sample_rates=(16000,))))
    assert argv[argv.index("-ar:0") + 1] == "16000"
    assert "-ac:0" not in argv


def test_a_module_constraining_only_the_channels_says_nothing_about_the_rate() -> None:
    argv = _feeder_argv(_audio_plan(_audio_described(channel_counts=(1,))))
    assert argv[argv.index("-ac:0") + 1] == "1"
    assert "-ar:0" not in argv


def test_conformance_reaches_the_edge_the_plan_describes() -> None:
    plan = _audio_plan(
        _audio_described(sample_rates=(16000,), channel_counts=(1,))
    ).plan
    assert plan is not None
    edge = next(e for e in plan.stream_edges if e.target == "sidecar0")
    written = edge.format.to_dict()
    assert written["required_rate"] == 16000
    assert written["required_channels"] == 1


def test_an_unconstrained_edge_says_nothing_about_conformance() -> None:
    plan = _audio_plan().plan
    assert plan is not None
    edge = next(e for e in plan.stream_edges if e.target == "sidecar0")
    written = edge.format.to_dict()
    assert "required_rate" not in written
    assert "required_channels" not in written


# a region of two audio modules

TRANSCRIBER = "modules/transcribe.wasm"
DUCKER = "modules/duck_speech.wasm"
WORD = "STRUCT(start number, text text)[]"

_WORD_ROWS: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["start", "text"],
    "properties": {"start": {"type": "number"}, "text": {"type": "string"}},
}

SPEECH_PAIR = (
    f"CREATE FUNCTION transcribe(a audio_stream)\n"
    f"RETURNS STRUCT(a audio_stream, words {WORD})\n"
    f"  AS '{TRANSCRIBER}', 'transcribe' LANGUAGE wasm;\n"
    f"CREATE FUNCTION duck_speech(a audio_stream, words {WORD})\n"
    f"RETURNS audio_stream AS '{DUCKER}', 'duck-speech' LANGUAGE wasm;\n"
    "COPY (SELECT duck_speech(transcribe(f.audio[1])) FROM input('a.mp4') f)\n"
    "TO 'out.m4a'"
)


def _speech_plan(sql: str = SPEECH_PAIR) -> Compiled:
    described = {
        TRANSCRIBER: replace(
            _audio_described(name="transcribe", sample_rates=(16000,), channel_counts=(1,)),
            rows_schema=_WORD_ROWS,
        ),
        DUCKER: replace(_audio_described(name="duck-speech"), meta=True),
    }
    return compile_all(sql, describe=lambda path: described[path])


def test_an_annotating_audio_module_composes_with_its_consumer() -> None:
    declared = _resolved(SPEECH_PAIR).wasm["transcribe"]
    assert declared.returns == "audio_stream"
    assert declared.emits is not None
    assert [(f.name, f.type) for f in declared.emits.fields] == [
        ("start", "number"),
        ("text", "text"),
    ]
    assert _resolved(SPEECH_PAIR).wasm["duck_speech"].reads is not None


def test_two_audio_modules_run_as_one_network_reading_an_audio_input() -> None:
    plan = _speech_plan().plan
    assert plan is not None
    assert wasm.shown_argv(plan.sidecars[0]) == [
        "ffrwd-wasm",
        "-f",
        "nut",
        "-i",
        "pipe:0",
        "-m",
        f"transcribe={TRANSCRIBER}",
        "-m",
        f"duck_speech={DUCKER}",
        "-filter_complex",
        "[0:a]transcribe[n1];[n1]duck_speech[out0]",
        "-map",
        "[out0]",
        "-f",
        "nut",
        "pipe:1",
    ]


def test_the_entry_modules_conformance_is_what_feeds_the_region() -> None:
    argv = _feeder_argv(_speech_plan())
    assert argv[argv.index("-ar:0") + 1] == "16000"
    assert argv[argv.index("-ac:0") + 1] == "1"


def test_an_annotation_column_over_the_wrong_kind_of_stream_is_still_refused() -> None:
    """The kind rules and the annotation rules are independent."""
    sql = SPEECH_PAIR.replace(
        "CREATE FUNCTION duck_speech(a audio_stream, words",
        "CREATE FUNCTION duck_speech(a video_stream, words",
    )
    with pytest.raises(FfrwdError) as caught:
        _speech_plan(sql)
    assert caught.value.code is ErrorCode.UNSUPPORTED_SQL
    assert "takes video_stream and returns audio_stream" in caught.value.message


# ---------------------------------------------------------------------------
# value-returning wasm functions
# ---------------------------------------------------------------------------

BRAND = "brand.wasm"
BRAND_DECLARE = (
    "CREATE FUNCTION brand(title text, suffix text) RETURNS text\n"
    f"  AS '{BRAND}', 'append-brand' LANGUAGE wasm;\n"
)


def _brand_described(
    *, result_type: str = "string", required: tuple[str, ...] = ("title", "suffix")
) -> Described:
    return Described(
        world="ffrwd:av@0.4.0",
        functions=(
            DescribedFunction(
                name="append-brand",
                params_schema={
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "title": {"type": "string"},
                        "suffix": {"type": "string"},
                    },
                    "required": list(required),
                },
                result_schema={"type": result_type},
            ),
        ),
    )


def _folded(
    sql: str,
    *,
    described: Described | None = None,
    invoke: object = None,
) -> Graph:
    calls: list[tuple[str, str, dict[str, object]]] = []

    def default_invoke(module: str, function: str, args: Mapping[str, object]) -> object:
        calls.append((module, function, dict(args)))
        return f"{args.get('title')}{args.get('suffix')}"

    graph = lower(
        _resolved(sql),
        {},
        registry=_snapshot_registry(),
        describes={BRAND: described or _brand_described()},
        invoke=invoke if invoke is not None else default_invoke,
    )
    graph._calls = calls  # type: ignore[attr-defined]
    return graph


def test_a_value_function_is_declared_with_no_leading_stream() -> None:
    sql = (
        BRAND_DECLARE
        + "COPY (SELECT f.video[1], STRUCT(brand('a', 'b') AS x) AS tags "
        "FROM input('a.mp4') f) TO 'out.mp4'"
    )
    declared = _resolved(sql).wasm["brand"]
    assert declared.returns == "text"
    assert declared.is_value
    assert [(p.name, p.type) for p in declared.params] == [
        ("title", "text"),
        ("suffix", "text"),
    ]
    assert declared.value_params == declared.params
    assert declared.written_params == declared.params


def test_a_folded_value_lands_in_the_tags_struct() -> None:
    sql = (
        BRAND_DECLARE
        + "COPY (SELECT f.video[1], STRUCT(brand('Main', ' - restored') AS title) AS tags "
        "FROM input('a.mp4') f) TO 'out.mp4'"
    )
    graph = _folded(sql)
    assert graph.sinks[0].tags["title"] == "Main - restored"


def test_identical_calls_invoke_the_module_once() -> None:
    sql = (
        BRAND_DECLARE
        + "COPY (SELECT f.video[1], "
        "STRUCT(brand('Main', ' - restored') AS a, brand('Main', ' - restored') AS b) "
        "AS tags FROM input('a.mp4') f) TO 'out.mp4'"
    )
    graph = _folded(sql)
    assert len(graph._calls) == 1  # type: ignore[attr-defined]
    assert graph._calls[0] == (BRAND, "append-brand", {"title": "Main", "suffix": " - restored"})  # type: ignore[attr-defined]


def test_a_null_argument_is_omitted() -> None:
    sql = (
        BRAND_DECLARE
        + "COPY (SELECT f.video[1], STRUCT(brand(NULL, 'x') AS title) AS tags "
        "FROM input('a.mp4') f) TO 'out.mp4'"
    )
    graph = _folded(sql, described=_brand_described(required=("suffix",)))
    assert graph._calls[0][2] == {"suffix": "x"}  # type: ignore[attr-defined]


def test_the_export_must_be_in_the_modules_function_list() -> None:
    described = Described(world="ffrwd:av@0.4.0", functions=(DescribedFunction(name="other"),))
    sql = (
        BRAND_DECLARE
        + "COPY (SELECT f.video[1], STRUCT(brand('a', 'b') AS title) AS tags "
        "FROM input('a.mp4') f) TO 'out.mp4'"
    )
    with pytest.raises(FfrwdError) as caught:
        _folded(sql, described=described)
    error = caught.value
    assert error.code is ErrorCode.UNSUPPORTED_SQL
    assert "names the export 'append-brand'" in error.message
    assert error.hint is not None and "other" in error.hint


def test_a_module_with_no_functions_is_refused() -> None:
    described = Described(world="ffrwd:av@0.4.0")
    sql = (
        BRAND_DECLARE
        + "COPY (SELECT f.video[1], STRUCT(brand('a', 'b') AS title) AS tags "
        "FROM input('a.mp4') f) TO 'out.mp4'"
    )
    with pytest.raises(FfrwdError) as caught:
        _folded(sql, described=described)
    assert "declares no functions" in caught.value.message


def test_the_result_type_must_match_the_modules_result_schema() -> None:
    sql = (
        BRAND_DECLARE
        + "COPY (SELECT f.video[1], STRUCT(brand('a', 'b') AS title) AS tags "
        "FROM input('a.mp4') f) TO 'out.mp4'"
    )
    with pytest.raises(FfrwdError) as caught:
        _folded(sql, described=_brand_described(result_type="number"))
    error = caught.value
    assert error.code is ErrorCode.UDF_ARG_TYPE
    assert "declares RETURNS text" in error.message
    assert "returns number" in error.message


def test_a_wrong_typed_module_result_is_refused_per_call() -> None:
    sql = (
        BRAND_DECLARE
        + "COPY (SELECT f.video[1], STRUCT(brand('a', 'b') AS title) AS tags "
        "FROM input('a.mp4') f) TO 'out.mp4'"
    )
    with pytest.raises(FfrwdError) as caught:
        _folded(sql, invoke=lambda module, function, args: 42)
    error = caught.value
    assert error.code is ErrorCode.UDF_ARG_TYPE
    assert "returned 42" in error.message


def test_an_argument_typed_wrong_against_the_modules_schema_is_refused() -> None:
    """The module's own schema says ``suffix`` is a string; a number is not."""
    described = _brand_described()
    described = Described(
        world=described.world,
        functions=(
            DescribedFunction(
                name="append-brand",
                params_schema={
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "suffix": {"type": "number"},
                    },
                },
                result_schema={"type": "string"},
            ),
        ),
    )
    sql = (
        BRAND_DECLARE
        + "COPY (SELECT f.video[1], STRUCT(brand('a', 'b') AS title) AS tags "
        "FROM input('a.mp4') f) TO 'out.mp4'"
    )
    with pytest.raises(FfrwdError) as caught:
        _folded(sql, described=described)
    assert caught.value.code is ErrorCode.UDF_ARG_TYPE


def test_invoke_failure_is_a_typed_error_naming_the_module() -> None:
    def failing_invoke(module: str, function: str, args: Mapping[str, object]) -> object:
        raise FfrwdError(
            ErrorCode.UNSUPPORTED_SQL,
            f"the module '{module}' rejected the call to {function}(): boom",
            hint="check the arguments match what the module declares",
        )

    sql = (
        BRAND_DECLARE
        + "COPY (SELECT f.video[1], STRUCT(brand('a', 'b') AS title) AS tags "
        "FROM input('a.mp4') f) TO 'out.mp4'"
    )
    with pytest.raises(FfrwdError) as caught:
        _folded(sql, invoke=failing_invoke)
    error = caught.value
    assert error.code is ErrorCode.UNSUPPORTED_SQL
    assert "boom" in error.message
    assert error.line is not None


def test_a_default_on_a_value_functions_parameter_is_refused() -> None:
    sql = (
        "CREATE FUNCTION brand(title text, suffix text DEFAULT ' (restored)') "
        "RETURNS text AS 'brand.wasm', 'append-brand' LANGUAGE wasm;\n"
        "SELECT brand('a') AS x"
    )
    error = _rejects_resolve(
        sql, ErrorCode.UNSUPPORTED_SQL, "gives the parameter 'suffix' a DEFAULT"
    )
    assert error.hint is not None and "no default to fall back to" in error.hint


def test_a_value_function_cannot_be_called_as_a_stream() -> None:
    sql = (
        BRAND_DECLARE
        + "COPY (SELECT brand('a', 'b') FROM input('a.mp4') f) TO 'out.mp4'"
    )
    with pytest.raises(FfrwdError) as caught:
        _folded(sql)
    error = caught.value
    assert error.code is ErrorCode.UNSUPPORTED_SQL
    assert "returns text, not a stream" in error.message


def test_a_value_function_in_from_is_refused() -> None:
    sql = BRAND_DECLARE + "SELECT t.x FROM brand('a', 'b') t"
    error = _rejects_resolve(sql, ErrorCode.UNSUPPORTED_SQL, "returns text, not a table")
    assert error.hint is not None and "metadata STRUCT" in error.hint


def _rejects_resolve(sql: str, code: ErrorCode, needle: str) -> FfrwdError:
    with pytest.raises(FfrwdError) as caught:
        _resolved(sql)
    error = caught.value
    assert error.code is code, f"{error.code} != {code}: {error}"
    assert needle in error.message, error.message
    return error


# -- a value function's call, where it is one more call's argument --------
#
# The argument-classifying pass used to assume every bare call was a stream,
# which never checked a wasm declaration's own RETURNS. These pin the four
# shapes that assumption got wrong or had to keep getting right.


def test_a_value_functions_call_folds_before_the_call_wrapping_it() -> None:
    """The reproduction: an inner brand() folds before the outer one sees it.

    Nothing inlines a wasm call -- both stay written calls all the way to
    lowering -- so the inner one has to be classified by its own RETURNS,
    not assumed a stream because it is a bare call.
    """
    sql = (
        BRAND_DECLARE
        + "COPY (SELECT f.video[1], "
        "STRUCT(brand(brand('Angel One', ' [4K]'), ' [HDR]') AS title) AS tags "
        "FROM input('a.mp4') f) TO 'out.mp4'"
    )
    graph = _folded(sql)
    assert graph.sinks[0].tags["title"] == "Angel One [4K] [HDR]"
    # Two distinct calls, inner then outer -- the invoke cache keys on the
    # arguments, so the differing 'title' keeps them from colliding.
    assert graph._calls == [  # type: ignore[attr-defined]
        (BRAND, "append-brand", {"title": "Angel One", "suffix": " [4K]"}),
        (BRAND, "append-brand", {"title": "Angel One [4K]", "suffix": " [HDR]"}),
    ]


def test_a_value_functions_call_fills_a_stream_functions_value_parameter() -> None:
    """A value function's call, folded, becomes a STREAM function's own argument."""
    sql = (
        BRAND_DECLARE
        + f"CREATE FUNCTION m(v video_stream, n text) RETURNS video_stream "
        f"AS '{MODULE}', 'invert' LANGUAGE wasm;\n"
        "COPY (SELECT m(f.video[1], brand('a', 'b')) FROM input('a.mp4') f) TO 'out.mp4'"
    )
    calls: list[tuple[str, str, dict[str, object]]] = []

    def invoke(module: str, function: str, args: Mapping[str, object]) -> object:
        calls.append((module, function, dict(args)))
        return f"{args.get('title')}{args.get('suffix')}"

    graph = lower(
        _resolved(sql),
        {},
        registry=_snapshot_registry(),
        describes={
            MODULE: _described(params={"n": {"type": "string"}}),
            BRAND: _brand_described(),
        },
        invoke=invoke,
    )
    node = next(n for n in graph.nodes.values() if n.filter == MODULE)
    assert node.args == {"n": "ab"}
    assert calls == [(BRAND, "append-brand", {"title": "a", "suffix": "b"})]


def test_a_stream_functions_call_as_a_value_argument_is_still_refused() -> None:
    """A wasm function returning a stream still cannot fill a value parameter.

    Same rejection as before the fix, now earned for the right reason: the
    nested call is classified by looking up 'invert' and finding it returns
    video_stream, not by assuming every bare call does.
    """
    sql = (
        BRAND_DECLARE
        + DECLARE
        + "COPY (SELECT f.video[1], STRUCT(brand(invert(f.video[1]), 'x') AS title) AS tags "
        "FROM input('a.mp4') f) TO 'out.mp4'"
    )
    error = _rejects_resolve(
        sql, ErrorCode.UDF_ARG_TYPE, "brand() takes text as its 'title' argument, got a stream"
    )
    assert error.hint == "brand(title text, suffix text) RETURNS text"


def test_a_sql_functions_call_folds_before_a_wasm_call_around_it() -> None:
    """A LANGUAGE sql call nested in a wasm call's argument already worked.

    Expansion inlines every resolvable call across the whole tree in one
    pass, wasm declarations included or not -- a wasm call is simply never a
    site that pass resolves, so the walk continues past it into its
    arguments and finds the sql call underneath regardless.
    """
    sql = (
        "CREATE FUNCTION shout(s text) RETURNS text AS $$ SELECT s || '!' $$ LANGUAGE sql;\n"
        + BRAND_DECLARE
        + "COPY (SELECT f.video[1], STRUCT(brand(shout('Main'), ' - x') AS title) AS tags "
        "FROM input('a.mp4') f) TO 'out.mp4'"
    )
    graph = _folded(sql)
    assert graph.sinks[0].tags["title"] == "Main! - x"


def test_a_wasm_value_call_folds_before_a_sql_call_wrapping_it() -> None:
    """The reverse nesting: a wasm value call inside a sql function's argument.

    A bare wasm call is not something the inlining walk resolves, so it
    reaches the sql function's own argument check unexpanded; that check
    needed the same RETURNS lookup :func:`_check_wasm_arguments` did.
    """
    sql = (
        BRAND_DECLARE
        + "CREATE FUNCTION wrap(s text) RETURNS text AS $$ SELECT s || '!' $$ LANGUAGE sql;\n"
        "COPY (SELECT f.video[1], STRUCT(wrap(brand('Main', ' - x')) AS title) AS tags "
        "FROM input('a.mp4') f) TO 'out.mp4'"
    )
    graph = _folded(sql)
    assert graph.sinks[0].tags["title"] == "Main - x!"


# -- a declaration inside a package ---------------------------------------
#
# A package is read out of the store or a linked directory and compiled from
# whatever working directory the caller is in, so a module path written in one
# of its lib files can only mean a file the package ships. Nothing here builds
# a real module or spawns anything: the describes are synthetic and the paths
# name files that do not exist, exactly as the rest of this file does.

PACKAGE_MODULE = "modules/invert.wasm"
PACKAGE_DECLARE = (
    "CREATE FUNCTION invert(v video_stream) RETURNS video_stream\n"
    f"  AS '{PACKAGE_MODULE}', 'invert' LANGUAGE wasm;\n"
)
PACKAGE_QUERY = (
    "COPY (SELECT ffrwd.tools.invert(f.video[1]) FROM input('a.mp4') f) TO 'out.mp4'"
)


def _package(root: Path, lib: str = PACKAGE_DECLARE) -> PackageSet:
    """A one-package project at `root`, and the package set a compile sees.

    The package IS the project, so no lockfile and no store are involved --
    what is under test is where a module path is read from, not how the
    package was installed.
    """
    (root / "src").mkdir(parents=True)
    (root / "src" / "tools.sql").write_text(lib, encoding="utf-8")
    (root / "ffrwd.json").write_text(
        json.dumps(
            {
                "name": "ffrwd/tools",
                "version": "1.0.0",
                "lib": {"invert": "src/tools.sql"},
            }
        ),
        encoding="utf-8",
    )
    found = discover(root)
    assert found is not None
    return found


def test_a_packages_module_path_is_read_against_the_package_root(tmp_path: Path) -> None:
    declared = resolve(parse(PACKAGE_QUERY), packages=_package(tmp_path)).wasm
    assert list(declared) == ["ffrwd.tools.invert"]
    assert Path(declared["ffrwd.tools.invert"].module) == tmp_path / "modules" / "invert.wasm"


def test_a_packages_module_path_does_not_follow_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole point: the caller's cwd is not where a package's files are."""
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    packages = _package(tmp_path / "pkg")
    monkeypatch.chdir(elsewhere)
    declared = resolve(parse(PACKAGE_QUERY), packages=packages).wasm
    assert (
        Path(declared["ffrwd.tools.invert"].module)
        == tmp_path / "pkg" / "modules" / "invert.wasm"
    )


@pytest.mark.parametrize("written", ["../invert.wasm", "src/../../invert.wasm"])
def test_a_packages_module_path_that_leaves_the_package_is_refused(
    tmp_path: Path, written: str
) -> None:
    lib = (
        "CREATE FUNCTION invert(v video_stream) RETURNS video_stream\n"
        f"  AS '{written}', 'invert' LANGUAGE wasm;\n"
    )
    with pytest.raises(FfrwdError) as caught:
        resolve(parse(PACKAGE_QUERY), packages=_package(tmp_path, lib))
    error = caught.value
    assert error.code is ErrorCode.UNSUPPORTED_SQL
    assert f"names the module '{written}', which leaves the package directory" in error.message
    assert error.hint is not None and "a package's modules ship inside it" in error.hint


def test_a_top_level_module_path_is_still_read_exactly_as_written(tmp_path: Path) -> None:
    """A declaration the script wrote keeps today's resolution, package or no package."""
    packages = _package(tmp_path)
    sql = DECLARE + "COPY (SELECT invert(f.video[1]) FROM input('a.mp4') f) TO 'out.mp4'"
    assert resolve(parse(sql), packages=packages).wasm["invert"].module == MODULE


def test_two_calls_to_one_package_function_declare_it_once(tmp_path: Path) -> None:
    sql = (
        "COPY (SELECT ffrwd.tools.invert(f.video[1]) AS a, "
        "ffrwd.tools.invert(f.video[2]) AS b FROM input('a.mp4') f) TO 'out.mp4'"
    )
    assert list(resolve(parse(sql), packages=_package(tmp_path)).wasm) == [
        "ffrwd.tools.invert"
    ]


def test_a_package_function_compiles_to_the_same_plan_as_the_same_declaration_inline(
    tmp_path: Path,
) -> None:
    """The headline: a package's module is hosted exactly as the script's own is."""
    module = str(tmp_path / "modules" / "invert.wasm")
    inline = (
        "CREATE FUNCTION invert(v video_stream) RETURNS video_stream\n"
        f"  AS '{module}', 'invert' LANGUAGE wasm;\n"
        "COPY (SELECT invert(f.video[1]) FROM input('a.mp4') f) TO 'out.mp4'"
    )
    packaged = compile_all(
        PACKAGE_QUERY, packages=_package(tmp_path), describe=lambda path: _described()
    )
    written = compile_all(inline, describe=lambda path: _described())
    assert packaged.plan is not None and written.plan is not None
    assert [p.module for p in packaged.plan.processes if isinstance(p, SidecarProcess)] == [
        module
    ]
    assert plan_argv(packaged.plan, sidecar_argv=wasm.shown_argv) == plan_argv(
        written.plan, sidecar_argv=wasm.shown_argv
    )


def test_a_package_function_is_named_by_its_call_path(tmp_path: Path) -> None:
    """A message about the call says what the reader wrote, not an internal name."""
    sql = (
        "COPY (SELECT ffrwd.tools.invert(f.video[1], 2) FROM input('a.mp4') f) TO 'out.mp4'"
    )
    with pytest.raises(FfrwdError) as caught:
        resolve(parse(sql), packages=_package(tmp_path))
    error = caught.value
    assert error.code is ErrorCode.UDF_ARG_TYPE
    assert "ffrwd.tools.invert() got 2 arguments" in error.message


def test_a_package_function_in_from_is_not_a_table(tmp_path: Path) -> None:
    sql = "COPY (SELECT t.x FROM ffrwd.tools.invert('a') t) TO 'out.mp4'"
    with pytest.raises(FfrwdError) as caught:
        resolve(parse(sql), packages=_package(tmp_path))
    error = caught.value
    assert "wasm function 'ffrwd.tools.invert' returns a stream, not a table" in error.message


def test_a_packages_own_query_function_may_call_its_module(tmp_path: Path) -> None:
    """A bare call inside a lib body sees the package's own wasm declaration."""
    lib = PACKAGE_DECLARE + (
        "CREATE FUNCTION cleaned(v video_stream) RETURNS video_stream AS $$\n"
        "  SELECT invert(v)\n"
        "$$ LANGUAGE sql;\n"
    )
    (tmp_path / "src").mkdir(parents=True)
    (tmp_path / "src" / "tools.sql").write_text(lib, encoding="utf-8")
    (tmp_path / "ffrwd.json").write_text(
        json.dumps(
            {
                "name": "ffrwd/tools",
                "version": "1.0.0",
                "lib": {"invert": "src/tools.sql", "cleaned": "src/tools.sql"},
            }
        ),
        encoding="utf-8",
    )
    packages = discover(tmp_path)
    assert packages is not None
    sql = "COPY (SELECT ffrwd.tools.cleaned(f.video[1]) FROM input('a.mp4') f) TO 'out.mp4'"
    declared = resolve(parse(sql), packages=packages).wasm
    assert Path(declared["ffrwd.tools.invert"].module) == tmp_path / "modules" / "invert.wasm"


def test_a_packages_value_function_folds_against_the_packages_module(
    tmp_path: Path,
) -> None:
    """A value function runs at compile time, so the path it runs is the package's."""
    lib = (
        "CREATE FUNCTION brand(title text, suffix text) RETURNS text\n"
        "  AS 'modules/brand.wasm', 'append-brand' LANGUAGE wasm;\n"
    )
    (tmp_path / "src").mkdir(parents=True)
    (tmp_path / "src" / "tools.sql").write_text(lib, encoding="utf-8")
    (tmp_path / "ffrwd.json").write_text(
        json.dumps(
            {"name": "ffrwd/tools", "version": "1.0.0", "lib": {"brand": "src/tools.sql"}}
        ),
        encoding="utf-8",
    )
    packages = discover(tmp_path)
    assert packages is not None
    ran: list[str] = []

    def invoke(module: str, function: str, args: Mapping[str, object]) -> object:
        ran.append(module)
        return f"{args['title']}{args['suffix']}"

    sql = (
        "COPY (SELECT f.video[1], STRUCT(ffrwd.tools.brand('Ep 1', ' (restored)') "
        "AS title) AS tags FROM input('a.mp4') f) TO 'out.mp4'"
    )
    compiled = compile_all(
        sql,
        packages=packages,
        describe=lambda path: _brand_described(),
        invoke=invoke,
    )
    assert [Path(module) for module in ran] == [tmp_path / "modules" / "brand.wasm"]
    assert compiled.plan is None  # a value function never becomes a process


# ---------------------------------------------------------------------------
# a module's rows, projected: the track the compiler mints from them
# ---------------------------------------------------------------------------

TRANSCRIBER = "modules/transcribe.wasm"
BURNER = "modules/burn_captions.wasm"
DENOISER = "modules/denoise.wasm"
CUE_RECORD = "STRUCT(text text, start_t number, end_t number)[]"

# The shape a cue-emitting module publishes, and the second shape the same
# module offers -- one row for the whole file rather than one per cue.
_CUE_ROWS: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["start_t", "end_t", "text"],
    "properties": {
        "start_t": {"type": "number"},
        "end_t": {"type": "number"},
        "text": {"type": "string"},
    },
}
_TRANSCRIPT_ROWS: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["text"],
    "properties": {"text": {"type": "string"}},
}


def _transcriber(
    *,
    rows: dict[str, object] | None = None,
    rows_language: tuple[str, ...] = ("language_to", "language"),
) -> Described:
    """A module that transcribes audio into cue rows, as --describe reports it."""
    return Described(
        world="ffrwd:av@0.8.0",
        name="transcribe",
        version="0.1.0",
        params_schema={
            "type": "object",
            "properties": {
                "language": {"type": "string"},
                "language_to": {"type": "string"},
            },
        },
        rows_schema=rows or {"oneOf": [_CUE_ROWS, _TRANSCRIPT_ROWS]},
        rows_language=rows_language,
        sample_formats=("f32",),
        sample_rates=(16000,),
        channel_counts=(1,),
    )


def _burner() -> Described:
    """A module that CONSUMES cue rows, for the parameter side of the sugar."""
    return Described(
        world="ffrwd:av@0.8.0",
        name="burn-captions",
        version="0.1.0",
        params_schema={"type": "object", "properties": {}},
        rows_schema=None,
        sample_formats=("f32",),
        meta=True,
    )


def _transcribe_declare(
    returns: str = "cue[]", params: str = "language text, language_to text DEFAULT NULL"
) -> str:
    return (
        f"CREATE FUNCTION transcribe(a audio_stream, {params})\n"
        f"RETURNS STRUCT(a audio_stream, words {returns})\n"
        f"  AS '{TRANSCRIBER}', 'transcribe' LANGUAGE wasm;\n"
    )


def _copy(columns: str, path: str = "subbed.mp4", options: str = "") -> str:
    return (
        f"COPY (SELECT {columns}\n"
        f"  FROM input('angel-one.mp4') s) TO '{path}'{options}"
    )


RECIPE = _transcribe_declare() + _copy(
    "s.video[1], s.audio[1],\n         transcribe(s.audio[1], 'es', 'en').words"
)


def _rows_plan(
    sql: str, described: Described | None = None, burner: Described | None = None
) -> Compiled:
    modules = {TRANSCRIBER: described or _transcriber(), BURNER: burner or _burner()}
    return compile_all(sql, describe=lambda path: modules[path])


def _rows_argv(sql: str, described: Described | None = None) -> dict[str, list[str]]:
    plan = _rows_plan(sql, described).plan
    assert plan is not None
    return plan_argv(plan, sidecar_argv=wasm.shown_argv)


def _rows_rejects(
    sql: str, code: ErrorCode, needle: str, described: Described | None = None
) -> FfrwdError:
    with pytest.raises(FfrwdError) as caught:
        _rows_plan(sql, described)
    error = caught.value
    assert error.code is code, f"{error.code} != {code}: {error}"
    assert needle in error.message, error.message
    assert error.line is not None and error.hint
    return error


# the recipe, argv for argv


def test_the_recipe_compiles_to_three_processes() -> None:
    """One ffmpeg feeds the module, the sidecar reads it, one ffmpeg muxes."""
    plan = _rows_plan(RECIPE).plan
    assert plan is not None
    assert len(plan.processes) == 3
    assert [s.module for s in plan.sidecars] == [TRANSCRIBER]


def test_the_recipe_renders_every_argv() -> None:
    assert _rows_argv(RECIPE) == {
        "ffmpeg0": [
            "ffmpeg", "-i", "angel-one.mp4",
            "-f", "webvtt", "-i", "pipe:0",
            "-map", "0:v:0", "-c:0", "copy",
            "-map", "0:a:0", "-c:1", "copy",
            "-map", "1:s:0", "-metadata:s:2", "language=eng", "-c:2", "mov_text",
            "subbed.mp4",
        ],
        "ffmpeg1": [
            "ffmpeg", "-i", "angel-one.mp4",
            "-map", "0:a:0", "-ar:0", "16000", "-ac:0", "1",
            "-c:0", "pcm_f32le", "-f", "nut", "pipe:1",
        ],
        "sidecar0": [
            "ffrwd-wasm", "-f", "nut", "-i", "pipe:0",
            "-m", TRANSCRIBER,
            "-params", '{"language": "es", "language_to": "en"}',
            "-f", "webvtt", "pipe:1",
        ],
    }


def test_the_recipe_prints_as_one_shell_pipeline() -> None:
    """Nothing fans, so all three chain through stdio."""
    plan = _rows_plan(RECIPE).plan
    assert plan is not None
    printed = render_plan(plan, sidecar_argv=wasm.shown_argv)
    assert printed.count(PIPELINE) == 2
    assert CHAIN not in printed


def test_the_minted_track_is_an_edge_from_the_sidecar() -> None:
    plan = _rows_plan(RECIPE).plan
    assert plan is not None
    (edge,) = plan.rows_edges
    assert (edge.source, edge.target, edge.container) == (
        "sidecar0", "ffmpeg0", "webvtt",
    )
    rows = plan.sidecars[0].rows
    assert rows is not None and rows.alias == edge.alias


def test_the_rows_edge_puts_every_process_in_one_stage() -> None:
    plan = _rows_plan(RECIPE).plan
    assert plan is not None
    assert [stage.processes for stage in plan.stages] == [
        ("ffmpeg0", "ffmpeg1", "sidecar0")
    ]


def test_a_rows_edge_says_what_it_carries_when_written_out() -> None:
    plan = _rows_plan(RECIPE).plan
    assert plan is not None
    written = plan.to_dict()["edges"]
    assert isinstance(written, list)
    assert {
        "kind": "rows",
        "source": "sidecar0",
        "target": "ffmpeg0",
        "alias": plan.rows_edges[0].alias,
        "container": "webvtt",
    } in written


def test_the_module_output_is_not_mapped() -> None:
    """The frames the rows came off feed the track, and go nowhere else."""
    plan = _rows_plan(RECIPE).plan
    assert plan is not None
    assert plan.sidecars[0].outputs == ()
    assert not [e for e in plan.stream_edges if e.source == "sidecar0"]


def test_a_region_of_two_modules_maps_the_label_its_rows_come_off() -> None:
    """The network form names the pad, the way an `-f nut` output does."""
    denoiser = Described(
        world="ffrwd:av@0.8.0",
        name="denoise",
        version="0.1.0",
        params_schema={"type": "object", "properties": {}},
        rows_schema=None,
        sample_formats=("f32",),
    )
    modules = {TRANSCRIBER: _transcriber(), DENOISER: denoiser}
    sql = (
        f"CREATE FUNCTION denoise(a audio_stream) RETURNS audio_stream\n"
        f"  AS '{DENOISER}', 'denoise' LANGUAGE wasm;\n"
    ) + _transcribe_declare() + _copy(
        "s.video[1], transcribe(denoise(s.audio[1]), 'es').words"
    )
    plan = compile_all(sql, describe=lambda path: modules[path]).plan
    assert plan is not None
    assert plan_argv(plan, sidecar_argv=wasm.shown_argv)["sidecar0"] == [
        "ffrwd-wasm", "-f", "nut", "-i", "pipe:0",
        "-m", f"denoise={DENOISER}",
        "-m", f"transcribe={TRANSCRIBER}",
        "-filter_complex", "[0:a]denoise[n1];[n1]transcribe=language=es[out0]",
        "-map", "[out0]", "-f", "webvtt", "pipe:1",
    ]


# the language the track is tagged with


def test_the_first_named_parameter_with_a_value_wins() -> None:
    """`language_to` leads the module's list, so the translation is the tag."""
    assert "language=eng" in _rows_argv(RECIPE)["ffmpeg0"]


def test_a_default_null_parameter_is_unset_and_falls_through() -> None:
    sql = _transcribe_declare() + _copy(
        "s.video[1], s.audio[1], transcribe(s.audio[1], 'es').words"
    )
    assert "language=spa" in _rows_argv(sql)["ffmpeg0"]


def test_a_parameter_filled_from_its_default_still_counts() -> None:
    sql = _transcribe_declare(
        params="language text, language_to text DEFAULT 'en'"
    ) + _copy("s.video[1], transcribe(s.audio[1], 'es').words")
    assert "language=eng" in _rows_argv(sql)["ffmpeg0"]


def test_a_module_naming_no_rows_language_leaves_the_track_untagged() -> None:
    argv = _rows_argv(RECIPE, described=_transcriber(rows_language=()))["ffmpeg0"]
    assert not [arg for arg in argv if arg.startswith("language=")]


def test_a_language_no_container_tag_stands_for_is_refused() -> None:
    sql = _transcribe_declare() + _copy(
        "s.video[1], transcribe(s.audio[1], 'zz').words"
    )
    error = _rows_rejects(sql, ErrorCode.UDF_ARG_TYPE, "'language' as 'zz'")
    assert error.hint is not None and "two-letter code" in error.hint


def test_the_tag_is_the_three_letter_one_a_container_records() -> None:
    assert (wasm.language_tag("es"), wasm.language_tag("de")) == ("spa", "ger")


def test_a_tag_written_in_full_is_itself() -> None:
    assert wasm.language_tag("eng") == "eng"


def test_an_unknown_code_has_no_tag() -> None:
    assert wasm.language_tag("zz") is None


def test_a_track_that_is_no_language_tags_as_one_anyway() -> None:
    # A module's machine-readable rows -- decoded barcodes, telemetry -- are
    # `zxx`; speech nobody identified is `und`; several at once is `mul`.
    assert wasm.language_tag("zxx") == "zxx"
    assert wasm.language_tag("und") == "und"
    assert wasm.language_tag("mul") == "mul"


def test_the_local_use_range_is_a_modules_own_to_pick_from() -> None:
    # 639-2 promises nothing will ever be assigned in qaa-qtz.
    assert wasm.language_tag("qaa") == "qaa"
    assert wasm.language_tag("qtz") == "qtz"
    assert wasm.language_tag("QRR") == "qrr"
    # Its edges, and the two-letter spelling nobody may use, stay refused.
    assert wasm.language_tag("qua") is None
    assert wasm.language_tag("q9a") is None
    assert wasm.language_tag("qr") is None


# the container the track is written for


def test_an_mkv_carries_the_webvtt_track_as_it_stands() -> None:
    sql = _transcribe_declare() + _copy(
        "s.audio[1], transcribe(s.audio[1], 'es').words", path="subbed.mkv"
    )
    assert "mov_text" not in _rows_argv(sql)["ffmpeg0"]


def test_a_subtitle_codec_the_query_names_is_left_alone() -> None:
    sql = _transcribe_declare() + _copy(
        "s.audio[1], transcribe(s.audio[1], 'es').words",
        options=" WITH (subtitle_codec 'srt')",
    )
    argv = _rows_argv(sql)["ffmpeg0"]
    assert "srt" in argv and "mov_text" not in argv


# the rows as the output


def test_a_rows_file_is_written_by_the_sidecar_itself() -> None:
    sql = _transcribe_declare() + _copy(
        "transcribe(s.audio[1], 'es', 'en').words", path="words.ndjson"
    )
    assert _rows_argv(sql) == {
        "ffmpeg0": [
            "ffmpeg", "-i", "angel-one.mp4",
            "-map", "0:a:0", "-ar:0", "16000", "-ac:0", "1",
            "-c:0", "pcm_f32le", "-f", "nut", "pipe:1",
        ],
        "sidecar0": [
            "ffrwd-wasm", "-f", "nut", "-i", "pipe:0",
            "-m", TRANSCRIBER,
            "-params", '{"language": "es", "language_to": "en"}',
            "-f", "ndjson", "words.ndjson",
        ],
    }


def test_a_rows_file_leaves_the_graph_with_no_ffmpeg_output() -> None:
    sql = _transcribe_declare() + _copy(
        "transcribe(s.audio[1], 'es').words", path="words.ndjson"
    )
    graph = _rows_plan(sql).graphs[0]
    assert graph.sinks == []
    assert [sink.path for sink in graph.rows_sinks.values()] == ["words.ndjson"]


def test_a_rows_file_beside_a_stream_column_is_refused() -> None:
    sql = _transcribe_declare() + _copy(
        "s.video[1], transcribe(s.audio[1], 'es').words", path="words.ndjson"
    )
    error = _rows_rejects(sql, ErrorCode.UNSUPPORTED_SQL, "is a rows file")
    assert error.hint is not None and "media file of their own" in error.hint


# what may be projected


def test_the_postgres_double_paren_spelling_reads_the_same_column() -> None:
    plain = _transcribe_declare() + _copy(
        "s.video[1], transcribe(s.audio[1], 'es', 'en').words"
    )
    parens = _transcribe_declare() + _copy(
        "s.video[1], (transcribe(s.audio[1], 'es', 'en')).words"
    )
    assert _rows_argv(plain) == _rows_argv(parens)


def test_the_stream_half_of_the_struct_is_not_read_back() -> None:
    sql = _transcribe_declare() + _copy("transcribe(s.audio[1], 'es').a")
    error = _rows_rejects(sql, ErrorCode.UNSUPPORTED_SQL, "is the stream transcribe()")
    assert error.hint is not None and "select the source stream itself" in error.hint


def test_a_field_the_return_does_not_declare_is_refused() -> None:
    sql = _transcribe_declare() + _copy("transcribe(s.audio[1], 'es').lyrics")
    error = _rows_rejects(sql, ErrorCode.UNSUPPORTED_SQL, "returns no field 'lyrics'")
    assert error.hint is not None and "'words'" in error.hint


def test_a_field_read_off_a_module_returning_no_struct_is_refused() -> None:
    sql = (
        "CREATE FUNCTION plain(a audio_stream) RETURNS audio_stream\n"
        f"  AS '{TRANSCRIBER}', 'transcribe' LANGUAGE wasm;\n"
    ) + _copy("plain(s.audio[1]).words")
    error = _rows_rejects(sql, ErrorCode.UNSUPPORTED_SQL, "plain() returns audio_stream")
    assert error.hint is not None and "RETURNS STRUCT" in error.hint


def test_a_producer_neither_projected_nor_consumed_is_still_refused() -> None:
    sql = _transcribe_declare() + _copy("transcribe(s.audio[1], 'es')")
    error = _rows_rejects(sql, ErrorCode.UDF_ARG_TYPE, "nothing here reads it")
    assert error.hint is not None and "transcribe(...).words" in error.hint


# `cue[]`, the shorthand


def test_cue_sugar_declares_the_cue_records_own_fields() -> None:
    declared = _resolved(RECIPE).wasm["transcribe"]
    assert declared.emits is not None
    assert [(f.name, f.type) for f in declared.emits.fields] == [
        ("text", "text"),
        ("start_t", "number"),
        ("end_t", "number"),
    ]


def test_cue_sugar_and_the_spelled_out_record_compile_alike() -> None:
    body = _copy("s.video[1], transcribe(s.audio[1], 'es', 'en').words")
    assert _rows_argv(_transcribe_declare() + body) == _rows_argv(
        _transcribe_declare(returns=CUE_RECORD) + body
    )


def test_cue_sugar_reads_back_as_the_record_it_stands_for() -> None:
    declared = _resolved(RECIPE).wasm["transcribe"]
    assert declared.signature == (
        "transcribe(a audio_stream, language text, language_to text DEFAULT NULL) "
        f"RETURNS STRUCT(a audio_stream, words {CUE_RECORD})"
    )


def test_a_consumer_may_take_its_cue_column_as_cue_sugar() -> None:
    sql = _transcribe_declare() + (
        "CREATE FUNCTION burn(a audio_stream, words cue[]) RETURNS audio_stream\n"
        f"  AS '{BURNER}', 'burn-captions' LANGUAGE wasm;\n"
    ) + _copy("burn(transcribe(s.audio[1], 'es'))")
    plan = _rows_plan(sql).plan
    assert plan is not None
    assert [b.path for b in plan.sidecars[0].modules] == [TRANSCRIBER, BURNER]


def test_a_consumer_spelling_the_record_out_matches_the_sugar_producer() -> None:
    sql = _transcribe_declare() + (
        f"CREATE FUNCTION burn(a audio_stream, words {CUE_RECORD}) "
        "RETURNS audio_stream\n"
        f"  AS '{BURNER}', 'burn-captions' LANGUAGE wasm;\n"
    ) + _copy("burn(transcribe(s.audio[1], 'es'))")
    assert _rows_plan(sql).plan is not None


# the row schema a module publishes


def test_a_declaration_matches_whichever_arm_the_module_offers() -> None:
    """`oneOf` is several row shapes, and one of them fitting is enough."""
    assert _rows_plan(RECIPE).plan is not None


def test_the_other_arm_of_a_oneof_matches_too() -> None:
    sql = _transcribe_declare(returns="STRUCT(text text)[]") + _copy(
        "s.video[1], transcribe(s.audio[1], 'es').words"
    )
    assert _rows_plan(sql).plan is not None


def test_a_record_matching_no_arm_names_every_arm() -> None:
    sql = _transcribe_declare(returns="STRUCT(speaker text)[]") + _copy(
        "s.video[1], transcribe(s.audio[1], 'es').words"
    )
    error = _rows_rejects(sql, ErrorCode.UDF_ARG_TYPE, "declares 'words' as")
    assert " or " in error.message
    assert "start_t (number)" in error.message


def test_a_plain_row_schema_is_one_arm() -> None:
    assert wasm.rows_arms(_transcriber(rows=_CUE_ROWS)) == (
        (("end_t", "number"), ("start_t", "number"), ("text", "string")),
    )


def test_a_module_declaring_no_rows_has_no_arms() -> None:
    assert wasm.rows_arms(_burner()) is None


def test_rows_fields_answers_with_the_first_arm() -> None:
    arms = wasm.rows_arms(_transcriber())
    assert arms is not None
    assert wasm.rows_fields(_transcriber()) == arms[0]


# the describe payload


def test_the_rows_language_list_is_read_off_the_describe_payload() -> None:
    described = wasm._described(
        TRANSCRIBER,
        {
            "world": "ffrwd:av@0.8.0",
            "name": "transcribe",
            "rows_language": ["language_to", "language"],
        },
    )
    assert described.rows_language == ("language_to", "language")


def test_a_describe_naming_no_rows_language_leaves_it_empty() -> None:
    described = wasm._described(
        TRANSCRIBER, {"world": "ffrwd:av@0.8.0", "name": "transcribe"}
    )
    assert described.rows_language == ()


def test_the_world_a_projecting_module_targets_is_hosted() -> None:
    assert "ffrwd:av@0.8.0" in wasm.WORLDS


# -- several streams into one module --------------------------------------
#
# A module built against the windowed interface may read more than one stream
# at a time. The declaration names one parameter per stream, all of one kind,
# ahead of the values the module is configured with.

DEPTH = "modules/depth.wasm"
MASK = "modules/blur_mask.wasm"
DUCK = "modules/duck.wasm"
MERGE3 = "modules/merge3.wasm"
PAIRS = "modules/pair_boxes.wasm"

BOKEH_DECLARE = (
    "CREATE FUNCTION depth(v video_stream) RETURNS video_stream\n"
    f"  AS '{DEPTH}', 'depth' LANGUAGE wasm;\n"
    "CREATE FUNCTION blur_mask(v video_stream, mask video_stream,\n"
    "                          max_radius number DEFAULT 16,\n"
    "                          invert boolean DEFAULT false)\n"
    "  RETURNS video_stream\n"
    f"  AS '{MASK}', 'blur-mask' LANGUAGE wasm;\n"
)
MASK_DECLARE = (
    "CREATE FUNCTION blur_mask(v video_stream, mask video_stream,\n"
    "                          max_radius number DEFAULT 16,\n"
    "                          invert boolean DEFAULT false)\n"
    "  RETURNS video_stream\n"
    f"  AS '{MASK}', 'blur-mask' LANGUAGE wasm;\n"
)
BOKEH = BOKEH_DECLARE + (
    "COPY (SELECT blur_mask(s.video[1], depth(s.video[1]), 24, true)\n"
    "      FROM input('shot.mp4') s) TO 'bokeh.mp4'"
)

MASK_PARAMS: dict[str, object] = {
    "max_radius": {"type": "number"},
    "invert": {"type": "boolean"},
}


def _multi(
    name: str,
    *,
    inputs: int = 1,
    params: dict[str, object] | None = None,
    pixel_formats: tuple[str, ...] = ("rgba",),
    sample_formats: tuple[str, ...] = (),
    window: int = 1,
    stride: int = 1,
    windowed: bool = True,
    rows: Mapping[str, object] | None = None,
) -> Described:
    """A windowed-interface description, the shape a multi-input module has."""
    return Described(
        world="ffrwd:av@0.9.0",
        name=name,
        version="0.1.0",
        params_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": dict(params or {}),
        },
        rows_schema=rows,
        pixel_formats=pixel_formats,
        sample_formats=sample_formats,
        window=window,
        stride=stride,
        windowed=windowed,
        inputs=inputs,
    )


def _bokeh_modules(
    depth: Described | None = None, mask: Described | None = None
) -> dict[str, Described]:
    return {
        DEPTH: depth or _multi("depth"),
        MASK: mask or _multi("blur-mask", inputs=2, params=MASK_PARAMS),
    }


def _multi_plan(sql: str, modules: dict[str, Described]) -> Compiled:
    return compile_all(sql, describe=lambda path: modules[path])


def _multi_argv(sql: str, modules: dict[str, Described]) -> dict[str, list[str]]:
    plan = _multi_plan(sql, modules).plan
    assert plan is not None
    return plan_argv(plan, sidecar_argv=wasm.shown_argv)


def _multi_rejects(
    sql: str, modules: dict[str, Described], code: ErrorCode, needle: str
) -> FfrwdError:
    with pytest.raises(FfrwdError) as caught:
        _multi_plan(sql, modules)
    error = caught.value
    assert error.code is code, f"{error.code} != {code}: {error}"
    assert needle in error.message, error.message
    assert error.line is not None and error.hint
    return error


# the chain, argv for argv


def test_the_bokeh_chain_renders_every_argv() -> None:
    """One ffmpeg feeds the region, both modules run in it, one ffmpeg muxes."""
    assert _multi_argv(BOKEH, _bokeh_modules()) == {
        "ffmpeg0": [
            "ffmpeg", "-f", "nut", "-i", "pipe:0",
            "-map", "0:v:0", "-c:0", "copy",
            "bokeh.mp4",
        ],
        "ffmpeg1": [
            "ffmpeg", "-i", "shot.mp4",
            "-map", "0:v:0", "-c:0", "rawvideo", "-pix_fmt:0", "yuv420p",
            "-f", "nut", "pipe:1",
        ],
        "sidecar0": [
            "ffrwd-wasm", "-f", "nut", "-i", "pipe:0",
            "-m", f"depth={DEPTH}",
            "-m", f"blur_mask={MASK}",
            "-filter_complex",
            "[0:v]depth[n1];[0:v][n1]blur_mask=max_radius=24:invert=1[out0]",
            "-map", "[out0]", "-f", "nut", "pipe:1",
        ],
    }


def test_the_split_of_the_source_is_the_regions_one_boundary_read() -> None:
    """Both readers are inside the region, so its split dissolves into it."""
    plan = _multi_plan(BOKEH, _bokeh_modules()).plan
    assert plan is not None
    incoming = [e for e in plan.stream_edges if e.target == "sidecar0"]
    assert len(incoming) == 1
    assert [b.path for b in plan.sidecars[0].modules] == [DEPTH, MASK]


def test_an_unwritten_value_parameter_falls_back_to_its_default() -> None:
    sql = BOKEH_DECLARE + (
        "COPY (SELECT blur_mask(s.video[1], depth(s.video[1]))\n"
        "      FROM input('shot.mp4') s) TO 'bokeh.mp4'"
    )
    argv = _multi_argv(sql, _bokeh_modules())["sidecar0"]
    assert argv[argv.index("-filter_complex") + 1] == (
        "[0:v]depth[n1];[0:v][n1]blur_mask=max_radius=16:invert=0[out0]"
    )


def test_a_multi_input_modules_region_declares_no_lookahead() -> None:
    """One frame off each input, one frame back: nothing is read ahead."""
    plan = _multi_plan(BOKEH, _bokeh_modules()).plan
    assert plan is not None
    assert plan.sidecars[0].lookahead == 0


# the same stream twice, and three of them


def test_one_stream_written_twice_reaches_both_inputs() -> None:
    sql = MASK_DECLARE + (
        "COPY (SELECT blur_mask(s.video[1], s.video[1])\n"
        "      FROM input('shot.mp4') s) TO 'bokeh.mp4'"
    )
    argv = _multi_argv(sql, _bokeh_modules())["sidecar0"]
    assert argv[argv.index("-filter_complex") + 1] == (
        "[0:v][0:v]blur_mask=max_radius=16:invert=0[out0]"
    )


def test_one_module_reading_several_streams_is_run_as_a_network() -> None:
    """The short spelling says one module over one input, and cannot say this."""
    sql = MASK_DECLARE + (
        "COPY (SELECT blur_mask(s.video[1], s.video[1])\n"
        "      FROM input('shot.mp4') s) TO 'bokeh.mp4'"
    )
    plan = _multi_plan(sql, _bokeh_modules()).plan
    assert plan is not None
    assert plan.sidecars[0].network
    assert wasm.shown_argv(plan.sidecars[0]) == [
        "ffrwd-wasm", "-f", "nut", "-i", "pipe:0",
        "-m", f"blur_mask={MASK}",
        "-filter_complex", "[0:v][0:v]blur_mask=max_radius=16:invert=0[out0]",
        "-map", "[out0]", "-f", "nut", "pipe:1",
    ]


def test_three_streams_are_labelled_in_argument_order() -> None:
    sql = (
        "CREATE FUNCTION depth(v video_stream) RETURNS video_stream\n"
        f"  AS '{DEPTH}', 'depth' LANGUAGE wasm;\n"
        "CREATE FUNCTION merge3(a video_stream, b video_stream, c video_stream)\n"
        "  RETURNS video_stream\n"
        f"  AS '{MERGE3}', 'merge3' LANGUAGE wasm;\n"
        "COPY (SELECT merge3(s.video[1], depth(s.video[1]), s.video[1])\n"
        "      FROM input('shot.mp4') s) TO 'out.mp4'"
    )
    modules = {DEPTH: _multi("depth"), MERGE3: _multi("merge3", inputs=3)}
    argv = _multi_argv(sql, modules)["sidecar0"]
    assert argv[argv.index("-filter_complex") + 1] == (
        "[0:v]depth[n1];[0:v][n1][0:v]merge3[out0]"
    )


# the same rule, over audio


def test_a_module_reading_several_audio_streams_compiles() -> None:
    """The rule is kind-generic; only the demo is video."""
    sql = (
        "CREATE FUNCTION duck(a audio_stream, key audio_stream)\n"
        "  RETURNS audio_stream\n"
        f"  AS '{DUCK}', 'duck' LANGUAGE wasm;\n"
        "COPY (SELECT duck(s.audio[1], s.audio[1])\n"
        "      FROM input('a.mp4') s) TO 'out.mp4'"
    )
    modules = {
        DUCK: _multi("duck", inputs=2, pixel_formats=(), sample_formats=("f32",))
    }
    argv = _multi_argv(sql, modules)["sidecar0"]
    assert argv[argv.index("-filter_complex") + 1] == "[0:a][0:a]duck[out0]"


# what a multi-stream signature may not declare


def test_a_signature_mixing_stream_kinds_across_its_streams_is_refused() -> None:
    sql = (
        "CREATE FUNCTION duck(a audio_stream, key video_stream)\n"
        "  RETURNS audio_stream\n"
        f"  AS '{DUCK}', 'duck' LANGUAGE wasm;\n"
        "COPY (SELECT duck(s.audio[1], s.video[1])\n"
        "      FROM input('a.mp4') s) TO 'out.mp4'"
    )
    modules = {DUCK: _multi("duck", inputs=2, sample_formats=("f32",))}
    error = _multi_rejects(
        sql,
        modules,
        ErrorCode.UNSUPPORTED_SQL,
        "takes 'key' as video_stream beside audio_stream",
    )
    assert error.hint is not None and "one kind of stream" in error.hint


def test_an_annotation_column_beside_several_streams_is_refused() -> None:
    sql = (
        "CREATE FUNCTION blur_mask(v video_stream, mask video_stream,\n"
        "                          boxes STRUCT(x number)[])\n"
        "  RETURNS video_stream\n"
        f"  AS '{MASK}', 'blur-mask' LANGUAGE wasm;\n"
        "COPY (SELECT blur_mask(s.video[1], s.video[1])\n"
        "      FROM input('shot.mp4') s) TO 'out.mp4'"
    )
    modules = {MASK: _multi("blur-mask", inputs=2)}
    error = _multi_rejects(
        sql,
        modules,
        ErrorCode.UNSUPPORTED_SQL,
        "takes the annotation column 'boxes' beside several streams",
    )
    assert error.hint is not None
    assert "takes no annotation column yet" in error.hint


def test_a_multi_stream_signature_may_still_return_annotations() -> None:
    """The column leaves beside the stream however many the module read."""
    rows: Mapping[str, object] = {
        "type": "object",
        "properties": {"x": {"type": "number"}, "y": {"type": "number"}},
    }
    sql = (
        "CREATE FUNCTION pair_boxes(v video_stream, w video_stream)\n"
        "  RETURNS STRUCT(v video_stream, boxes STRUCT(x number, y number)[])\n"
        f"  AS '{PAIRS}', 'pair-boxes' LANGUAGE wasm;\n"
        "COPY (SELECT pair_boxes(s.video[1], s.video[1]).boxes\n"
        "      FROM input('a.mp4') s) TO 'boxes.ndjson'"
    )
    modules = {PAIRS: _multi("pair-boxes", inputs=2, rows=rows)}
    assert _multi_argv(sql, modules)["sidecar0"] == [
        "ffrwd-wasm", "-f", "nut", "-i", "pipe:0",
        "-m", f"pair_boxes={PAIRS}",
        "-filter_complex", "[0:v][0:v]pair_boxes[out0]",
        "-map", "[out0]", "-f", "ndjson", "boxes.ndjson",
    ]


def test_a_stream_after_a_value_parameter_is_still_a_second_stream() -> None:
    sql = (
        "CREATE FUNCTION blur_mask(v video_stream, radius number,\n"
        "                          mask video_stream)\n"
        "  RETURNS video_stream\n"
        f"  AS '{MASK}', 'blur-mask' LANGUAGE wasm;\n"
        "COPY (SELECT blur_mask(s.video[1], 4, s.video[1])\n"
        "      FROM input('shot.mp4') s) TO 'out.mp4'"
    )
    modules = {MASK: _multi("blur-mask", inputs=2)}
    error = _multi_rejects(
        sql, modules, ErrorCode.UNSUPPORTED_SQL, "takes a second stream, 'mask'"
    )
    assert error.hint is not None and "streams come first" in error.hint


# the declaration against what the module reads


def test_a_signature_short_of_the_modules_stream_count_is_refused() -> None:
    sql = (
        "CREATE FUNCTION blur_mask(v video_stream) RETURNS video_stream\n"
        f"  AS '{MASK}', 'blur-mask' LANGUAGE wasm;\n"
        "COPY (SELECT blur_mask(s.video[1]) FROM input('shot.mp4') s) TO 'out.mp4'"
    )
    modules = {MASK: _multi("blur-mask", inputs=2)}
    error = _multi_rejects(
        sql,
        modules,
        ErrorCode.UNSUPPORTED_SQL,
        f"declares 1 stream parameter, and the module '{MASK}' reads 2",
    )
    assert error.hint is not None and "2 of them" in error.hint


def test_a_signature_over_the_modules_stream_count_is_refused() -> None:
    sql = (
        "CREATE FUNCTION blur_mask(v video_stream, mask video_stream)\n"
        "  RETURNS video_stream\n"
        f"  AS '{MASK}', 'blur-mask' LANGUAGE wasm;\n"
        "COPY (SELECT blur_mask(s.video[1], s.video[1])\n"
        "      FROM input('shot.mp4') s) TO 'out.mp4'"
    )
    modules = {MASK: _multi("blur-mask")}
    error = _multi_rejects(
        sql,
        modules,
        ErrorCode.UNSUPPORTED_SQL,
        f"declares 2 stream parameters, and the module '{MASK}' reads 1",
    )
    assert error.hint is not None and "1 of them" in error.hint


def test_a_windowing_module_reading_several_streams_is_refused() -> None:
    sql = BOKEH_DECLARE + (
        "COPY (SELECT blur_mask(s.video[1], depth(s.video[1]))\n"
        "      FROM input('shot.mp4') s) TO 'bokeh.mp4'"
    )
    modules = _bokeh_modules(
        mask=_multi("blur-mask", inputs=2, params=MASK_PARAMS, window=4, stride=2)
    )
    error = _multi_rejects(
        sql,
        modules,
        ErrorCode.UNSUPPORTED_SQL,
        f"the module '{MASK}' reads 2 streams over a window of 4 every 2",
    )
    assert error.hint is not None and "window and stride of 1" in error.hint


def test_a_per_frame_module_reading_several_streams_is_refused() -> None:
    """Reading more than one stream is a windowed-interface export's."""
    sql = BOKEH_DECLARE + (
        "COPY (SELECT blur_mask(s.video[1], depth(s.video[1]))\n"
        "      FROM input('shot.mp4') s) TO 'bokeh.mp4'"
    )
    modules = _bokeh_modules(
        mask=_multi("blur-mask", inputs=2, params=MASK_PARAMS, windowed=False)
    )
    error = _multi_rejects(
        sql,
        modules,
        ErrorCode.UNSUPPORTED_SQL,
        f"the module '{MASK}' reads 2 streams, and declares the per-frame interface",
    )
    assert error.hint is not None and "windowed interface" in error.hint


# the call


def test_a_call_short_of_its_stream_arguments_names_the_parameter() -> None:
    sql = MASK_DECLARE + (
        "COPY (SELECT blur_mask(s.video[1]) FROM input('shot.mp4') s) TO 'out.mp4'"
    )
    error = _multi_rejects(
        sql,
        _bokeh_modules(),
        ErrorCode.UDF_ARG_TYPE,
        "blur_mask() got 1 argument, but its parameter 'mask' has no DEFAULT",
    )
    assert error.hint is not None and "mask video_stream" in error.hint


def test_a_stream_argument_of_the_wrong_kind_is_refused() -> None:
    sql = MASK_DECLARE + (
        "COPY (SELECT blur_mask(s.video[1], s.audio[1])\n"
        "      FROM input('shot.mp4') s) TO 'out.mp4'"
    )
    _multi_rejects(
        sql,
        _bokeh_modules(),
        ErrorCode.UDF_ARG_TYPE,
        "it takes video, video as its stream inputs, got (video, audio)",
    )


def test_two_tracks_into_one_module_are_not_in_lockstep() -> None:
    """Every input has to trace back to one point, or no frames pair up."""
    sql = MASK_DECLARE + (
        "COPY (SELECT blur_mask(s.video[1], s.video[2])\n"
        "      FROM input('shot.mp4') s) TO 'out.mp4'"
    )
    error = _multi_rejects(
        sql, _bokeh_modules(), ErrorCode.UNSUPPORTED_SQL, "do not run in lockstep"
    )
    assert error.hint is not None and "one frame out per frame in" in error.hint


# the describe payload


def test_the_input_count_is_read_off_the_describe_payload() -> None:
    described = wasm._described(
        MASK, {"world": "ffrwd:av@0.9.0", "name": "blur-mask", "inputs": 2}
    )
    assert described.inputs == 2


def test_a_describe_naming_no_input_count_reads_one() -> None:
    described = wasm._described(MASK, {"world": "ffrwd:av@0.9.0", "name": "blur-mask"})
    assert described.inputs == 1


def test_the_world_a_multi_input_module_targets_is_hosted() -> None:
    assert "ffrwd:av@0.9.0" in wasm.WORLDS


# ---------------------------------------------------------------------------
# narrowing a module's rows while it runs
# ---------------------------------------------------------------------------

SEGMENTER = "modules/segment.wasm"
SELECTOR = "modules/mask_select.wasm"
MASKER = "modules/blur_mask.wasm"
OBJECT = (
    "STRUCT(id number, class text, score number, x number, y number, "
    "w number, h number)[]"
)

_OBJECT_ROWS: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["id", "class", "score", "x", "y", "w", "h"],
    "properties": {
        "id": {"type": "integer"},
        "class": {"type": "string"},
        "score": {"type": "number"},
        "x": {"type": "integer"},
        "y": {"type": "integer"},
        "w": {"type": "integer"},
        "h": {"type": "integer"},
    },
}

# Three predicates as a filtergraph carries them: `:` escaped at both levels,
# `,` `[` `]` at the outer one, everything else through unchanged.
PERSON_PRED = '{"eq"\\\\:\\[{"field"\\\\:"class"}\\,{"lit"\\\\:"person"}\\]}'
SCORE_PRED = '{"ge"\\\\:\\[{"field"\\\\:"score"}\\,{"lit"\\\\:0.5}\\]}'
START_PRED = '{"ge"\\\\:\\[{"field"\\\\:"start_t"}\\,{"lit"\\\\:10}\\]}'


def _segmenter(*, nn: bool = False) -> Described:
    """A module returning a mask stream and the objects it found on each frame."""
    return Described(
        world="ffrwd:av@0.9.0",
        name="segment",
        version="0.1.0",
        params_schema={"type": "object", "properties": {}},
        rows_schema=_OBJECT_ROWS,
        pixel_formats=("yuv420p", "rgba"),
        nn=nn,
    )


def _selector() -> Described:
    """A module that reads those objects and narrows the mask to them."""
    return Described(
        world="ffrwd:av@0.9.0",
        name="mask_select",
        version="0.1.0",
        params_schema={"type": "object", "properties": {}},
        pixel_formats=("yuv420p", "rgba"),
        meta=True,
    )


def _masker() -> Described:
    """A module that blurs one stream where a second one masks it."""
    return Described(
        world="ffrwd:av@0.9.0",
        name="blur_mask",
        version="0.1.0",
        params_schema={"type": "object", "properties": {}},
        pixel_formats=("yuv420p", "rgba"),
        inputs=2,
        windowed=True,
    )


def _segment_paths(root: Path | None = None) -> tuple[str, str, str]:
    """The three module paths, under `root` where a test needs a real directory."""
    if root is None:
        return (SEGMENTER, SELECTOR, MASKER)
    return (
        str(root / "segment.wasm"),
        str(root / "mask_select.wasm"),
        str(root / "blur_mask.wasm"),
    )


def _segment_declare(paths: tuple[str, str, str]) -> str:
    segmenter, selector, masker = paths
    return (
        f"CREATE FUNCTION segment(v video_stream)\n"
        f"RETURNS STRUCT(map video_stream, objects {OBJECT})\n"
        f"  AS '{segmenter}', 'segment' LANGUAGE wasm;\n"
        f"CREATE FUNCTION mask_select(v video_stream, objects {OBJECT})\n"
        f"RETURNS video_stream AS '{selector}', 'mask_select' LANGUAGE wasm;\n"
        f"CREATE FUNCTION blur_mask(v video_stream, m video_stream)\n"
        f"RETURNS video_stream AS '{masker}', 'blur_mask' LANGUAGE wasm;\n"
    )


def _gather(where: str = "o.class = 'person'", stream: str = "s.video[1]") -> str:
    """One ``ARRAY(SELECT ...)`` over the objects a segmentation found."""
    return f"ARRAY(SELECT o FROM unnest(segment({stream}).objects) o WHERE {where})"


def _masked(where: str = "o.class = 'person'", stream: str = "s.video[1]") -> str:
    """The consumer call the recipe is written around."""
    return f"mask_select(segment(s.video[1]).map, {_gather(where, stream)})"


def _segment_query(body: str, paths: tuple[str, str, str] | None = None) -> str:
    """`body` as the mask of a blur, which is what these three modules are for."""
    return _segment_declare(paths or _segment_paths()) + (
        f"COPY (SELECT blur_mask(s.video[1], {body})\n"
        f"      FROM input('a.mp4') s) TO 'out.mp4'"
    )


def _segment_plan(
    sql: str, *, nn: bool = False, paths: tuple[str, str, str] | None = None
) -> Compiled:
    segmenter, selector, masker = paths or _segment_paths()
    described = {
        segmenter: _segmenter(nn=nn),
        selector: _selector(),
        masker: _masker(),
    }
    return compile_all(sql, describe=lambda path: described[path])


def _segment_argv(
    sql: str, *, nn: bool = False, paths: tuple[str, str, str] | None = None
) -> list[str]:
    plan = _segment_plan(sql, nn=nn, paths=paths).plan
    assert plan is not None
    return wasm.shown_argv(plan.sidecars[0])


def _segment_network(sql: str) -> str:
    argv = _segment_argv(sql)
    return argv[argv.index("-filter_complex") + 1]


def _segment_rejects(sql: str, code: ErrorCode, needle: str) -> FfrwdError:
    with pytest.raises(FfrwdError) as caught:
        _segment_plan(sql)
    error = caught.value
    assert error.code is code, f"{error.code} != {code}: {error}"
    assert needle in error.message, error.message
    assert error.line is not None and error.hint
    return error


def _unescaped(value: str) -> str:
    """One filtergraph-escaped option value read back, for a JSON assertion.

    Dropping every backslash is enough here: the predicate JSON carries none
    of its own, so each one in the rendered value is an escape.
    """
    return value.replace("\\", "")


def test_the_showcase_compiles_to_one_chain(tmp_path: Path) -> None:
    """Producer, row filter, consumer and blur, in one region and one argv."""
    paths = _segment_paths(tmp_path)
    (tmp_path / "segment.onnx").write_bytes(b"")
    argv = _segment_argv(
        _segment_query(_masked(), paths), nn=True, paths=paths
    )
    assert argv == [
        "ffrwd-wasm",
        "-f",
        "nut",
        "-i",
        "pipe:0",
        "-nn",
        f"segment={tmp_path / 'segment.onnx'}",
        "-m",
        f"segment={paths[0]}",
        "-m",
        f"mask_select={paths[1]}",
        "-m",
        f"blur_mask={paths[2]}",
        "-filter_complex",
        f"[0:v]segment[n1];[n1]rowfilter=pred={PERSON_PRED}[n2];"
        f"[n2]mask_select[n3];[0:v][n3]blur_mask[out0]",
        "-map",
        "[out0]",
        "-f",
        "nut",
        "pipe:1",
    ]
    assert "-nn-runtime" not in argv and "-nn-target" not in argv


def test_the_row_filter_binds_no_module() -> None:
    """A hosted node is the sidecar's own, so no -m entry names it."""
    plan = _segment_plan(_segment_query(_masked())).plan
    assert plan is not None
    assert [binding.name for binding in plan.sidecars[0].modules] == [
        "segment",
        "mask_select",
        "blur_mask",
    ]


def test_a_numeric_comparison_narrows_the_rows() -> None:
    assert _segment_network(_segment_query(_masked("o.score >= 0.5"))) == (
        f"[0:v]segment[n1];[n1]rowfilter=pred={SCORE_PRED}[n2];"
        f"[n2]mask_select[n3];[0:v][n3]blur_mask[out0]"
    )


def test_and_or_and_not_travel_as_one_predicate() -> None:
    """AND and OR spell a list, and a chain of one of them flattens into it."""
    where = "(o.class = 'person' OR o.class = 'cat') AND NOT o.score < 0.25"
    network = _segment_network(_segment_query(_masked(where)))
    predicate = network.split("pred=")[1].split("[n2];")[0]
    assert json.loads(_unescaped(predicate)) == {
        "and": [
            {
                "or": [
                    {"eq": [{"field": "class"}, {"lit": "person"}]},
                    {"eq": [{"field": "class"}, {"lit": "cat"}]},
                ]
            },
            {"not": {"lt": [{"field": "score"}, {"lit": 0.25}]}},
        ]
    }


def test_a_gather_with_no_where_mints_no_filter() -> None:
    """Narrowing nothing is the plain projection, which needs no node."""
    body = (
        "mask_select(segment(s.video[1]).map, "
        "ARRAY(SELECT o FROM unnest(segment(s.video[1]).objects) o))"
    )
    assert _segment_network(_segment_query(body)) == (
        "[0:v]segment[n1];[n1]mask_select[n2];[0:v][n2]blur_mask[out0]"
    )


def test_value_params_follow_a_written_annotation_argument() -> None:
    """A consumer's trailing values sit past the gather, not on top of it."""
    segmenter, selector, masker = _segment_paths()
    sql = (
        f"CREATE FUNCTION segment(v video_stream)\n"
        f"RETURNS STRUCT(map video_stream, objects {OBJECT})\n"
        f"  AS '{segmenter}', 'segment' LANGUAGE wasm;\n"
        f"CREATE FUNCTION mask_select(v video_stream, objects {OBJECT},\n"
        f"                            grow number DEFAULT NULL)\n"
        f"RETURNS video_stream AS '{selector}', 'mask_select' LANGUAGE wasm;\n"
        f"COPY (SELECT mask_select(segment(s.video[1]).map, {_gather()}, 8)\n"
        f"      FROM input('a.mp4') s) TO 'out.mp4'"
    )
    selector_described = Described(
        world="ffrwd:av@0.9.0",
        name="mask_select",
        version="0.1.0",
        params_schema={"type": "object", "properties": {"grow": {"type": "number"}}},
        pixel_formats=("yuv420p", "rgba"),
        meta=True,
    )
    described = {segmenter: _segmenter(), selector: selector_described}
    plan = compile_all(sql, describe=lambda path: described[path]).plan
    assert plan is not None
    argv = wasm.shown_argv(plan.sidecars[0])
    network = argv[argv.index("-filter_complex") + 1]
    assert "mask_select=grow=8" in network


def test_the_implicit_spelling_is_unchanged() -> None:
    """Writing the producer whole still hands both halves over."""
    assert _segment_network(_segment_query("mask_select(segment(s.video[1]))")) == (
        "[0:v]segment[n1];[n1]mask_select[n2];[0:v][n2]blur_mask[out0]"
    )


def test_a_field_the_record_does_not_declare_is_refused() -> None:
    error = _segment_rejects(
        _segment_query(_masked("o.clas = 'person'")),
        ErrorCode.UNSUPPORTED_SQL,
        "unknown column 'o.clas'",
    )
    assert error.hint == "did you mean 'class'?"


def test_a_literal_of_the_wrong_type_is_refused() -> None:
    error = _segment_rejects(
        _segment_query(_masked("o.class = 3")),
        ErrorCode.UDF_ARG_TYPE,
        "the field 'class' is text, and 3 is number",
    )
    assert error.hint is not None and "class text" in error.hint


def test_a_predicate_outside_the_grammar_is_refused() -> None:
    error = _segment_rejects(
        _segment_query(_masked("upper(o.class) = 'P'")),
        ErrorCode.UNSUPPORTED_SQL,
        "upper() is not part of the runtime predicate grammar",
    )
    assert error.hint is not None and "AND, OR, NOT" in error.hint


def test_between_is_outside_the_runtime_grammar() -> None:
    _segment_rejects(
        _segment_query(_masked("o.score BETWEEN 1 AND 2")),
        ErrorCode.UNSUPPORTED_SQL,
        "BETWEEN is not part of the runtime predicate grammar",
    )


def test_a_predicate_reading_past_the_alias_is_refused() -> None:
    error = _segment_rejects(
        _segment_query(_masked("z.class = 'person'")),
        ErrorCode.UNKNOWN_ALIAS,
        "'z' is not the gathered rows 'o'",
    )
    assert error.hint is not None and "o's own fields" in error.hint


def test_a_gather_selecting_one_field_is_refused() -> None:
    body = (
        "mask_select(segment(s.video[1]).map, ARRAY(SELECT o.class FROM "
        "unnest(segment(s.video[1]).objects) o WHERE o.score > 1))"
    )
    error = _segment_rejects(
        _segment_query(body), ErrorCode.UNSUPPORTED_SQL, "selects the whole row"
    )
    assert error.hint is not None and "not there yet" in error.hint


def test_rows_off_one_stream_do_not_ride_another() -> None:
    body = f"mask_select(segment(s.video[2]).map, {_gather()})"
    error = _segment_rejects(
        _segment_query(body),
        ErrorCode.UDF_ARG_TYPE,
        "reads rows off a stream segment() did not produce",
    )
    assert error.hint is not None and "the same call for both halves" in error.hint


def test_the_stream_half_is_still_unreadable_on_its_own() -> None:
    """Readable beside the rows it was made with, and nowhere else."""
    segmenter, _, masker = _segment_paths()
    sql = (
        f"CREATE FUNCTION segment(v video_stream)\n"
        f"RETURNS STRUCT(map video_stream, objects {OBJECT})\n"
        f"  AS '{segmenter}', 'segment' LANGUAGE wasm;\n"
        f"CREATE FUNCTION blur_mask(v video_stream, m video_stream)\n"
        f"RETURNS video_stream AS '{masker}', 'blur_mask' LANGUAGE wasm;\n"
        "COPY (SELECT blur_mask(s.video[1], segment(s.video[1]).map)\n"
        "      FROM input('a.mp4') s) TO 'out.mp4'"
    )
    described = {segmenter: _segmenter(), masker: _masker()}
    with pytest.raises(FfrwdError) as caught:
        compile_all(sql, describe=lambda path: described[path])
    error = caught.value
    assert error.code is ErrorCode.UNSUPPORTED_SQL
    assert "a stream is not read back off a struct" in error.message
    assert error.line is not None and error.hint
    assert "reads annotations" in error.hint


# a filtered gather stands where the unfiltered projection already stood


def _filtered_cues(where: str = "w.start_t >= 10") -> str:
    return (
        "ARRAY(SELECT w FROM unnest(transcribe(s.audio[1], 'en').words) w "
        f"WHERE {where})"
    )


def test_filtered_cue_rows_still_mint_a_subtitle_track() -> None:
    sql = _transcribe_declare() + _copy(f"s.video[1], {_filtered_cues()}")
    argv = _rows_argv(sql)["sidecar0"]
    assert argv[argv.index("-filter_complex") + 1] == (
        f"[0:a]transcribe=language=en[n1];[n1]rowfilter=pred={START_PRED}[out0]"
    )
    assert argv[-3:] == ["-f", "webvtt", "pipe:1"]


def test_filtered_rows_go_to_an_ndjson_file() -> None:
    sql = _transcribe_declare() + _copy(_filtered_cues(), path="words.ndjson")
    argv = _rows_argv(sql)["sidecar0"]
    assert argv[argv.index("-filter_complex") + 1] == (
        f"[0:a]transcribe=language=en[n1];[n1]rowfilter=pred={START_PRED}[out0]"
    )
    assert argv[-3:] == ["-f", "ndjson", "words.ndjson"]


# -- the model a module runs ----------------------------------------------


def test_a_module_running_no_model_binds_none() -> None:
    assert "-nn" not in _segment_argv(_segment_query(_masked()))


def test_a_model_that_is_not_there_is_refused(tmp_path: Path) -> None:
    paths = _segment_paths(tmp_path)
    with pytest.raises(FfrwdError) as caught:
        _segment_plan(_segment_query(_masked(), paths), nn=True, paths=paths)
    error = caught.value
    assert error.code is ErrorCode.UNSUPPORTED_SQL
    assert str(tmp_path / "segment.onnx") in error.message
    assert "is not there" in error.message
    assert error.line is not None and error.hint
    assert "fetch" in error.hint


def test_the_model_file_sits_beside_the_module() -> None:
    assert wasm.model_path("modules/segment.wasm", "segment") == "modules/segment.onnx"
    assert wasm.model_path("segment.wasm", "seg") == "seg.onnx"


def test_the_model_flag_is_read_off_the_describe_payload() -> None:
    described = wasm._described(
        SEGMENTER, {"world": "ffrwd:av@0.9.0", "name": "segment", "nn": True}
    )
    assert described.nn is True
    assert not wasm._described(SEGMENTER, {"world": "ffrwd:av@0.9.0", "name": "s"}).nn



# -- sink modules: RETURNS sink, the TO-call destination, and the plan -----

SINK_MODULE = "modules/post_rows.wasm"
STATS_MODULE = "modules/stats.wasm"

SINK_DECLARE = (
    "CREATE FUNCTION post_rows(v video_stream,\n"
    "                          boxes STRUCT(x number, y number)[],\n"
    "                          url text)\n"
    "RETURNS sink\n"
    f"  AS '{SINK_MODULE}', 'post_rows' LANGUAGE wasm;\n"
)
STATS_DECLARE = (
    "CREATE FUNCTION stats(v video_stream)\n"
    "RETURNS STRUCT(v video_stream, boxes STRUCT(x number, y number)[])\n"
    f"  AS '{STATS_MODULE}', 'stats' LANGUAGE wasm;\n"
)
SINK_RECIPE = (
    STATS_DECLARE
    + SINK_DECLARE
    + "COPY (SELECT stats(f.video[1]) FROM input('a.mp4') f)\n"
    "TO post_rows('http://127.0.0.1:9/rows')"
)

PLAIN_SINK_DECLARE = (
    "CREATE FUNCTION drain(v video_stream) RETURNS sink\n"
    f"  AS '{SINK_MODULE}', 'drain' LANGUAGE wasm;\n"
)
PLAIN_SINK_QUERY = (
    PLAIN_SINK_DECLARE + "COPY (SELECT f.video[1] FROM input('a.mp4') f) TO drain()"
)

SINK_ROWS_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {"x": {"type": "number"}, "y": {"type": "number"}},
    "required": ["x", "y"],
}


def _sink_described(
    *,
    name: str = "post_rows",
    params: dict[str, object] | None = None,
    reads_rows: bool = True,
    http: bool = False,
    udp: bool = False,
) -> Described:
    return Described(
        world="ffrwd:av@0.9.0",
        name=name,
        version="0.1.0",
        params_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": dict(
                params if params is not None else {"url": {"type": "string"}}
            ),
        },
        rows_schema=None,
        pixel_formats=("rgba",),
        window=1,
        windowed=True,
        one_to_one=False,
        reads_rows=reads_rows,
        forwards_rows=False,
        http=http,
        udp=udp,
    )


def _stats_described() -> Described:
    return Described(
        world="ffrwd:av@0.9.0",
        name="stats",
        version="0.1.0",
        params_schema={"type": "object", "additionalProperties": False, "properties": {}},
        rows_schema=SINK_ROWS_SCHEMA,
        pixel_formats=("rgba",),
        window=1,
        windowed=True,
        reads_rows=False,
        forwards_rows=False,
    )


def _sink_plan(sql: str, sink: Described | None = None) -> Compiled:
    modules = {
        SINK_MODULE: sink or _sink_described(),
        STATS_MODULE: _stats_described(),
        MODULE: _described(),
    }
    return compile_all(sql, describe=lambda path: modules[path])


def _sink_rejects(
    sql: str, code: ErrorCode, needle: str, sink: Described | None = None
) -> FfrwdError:
    with pytest.raises(FfrwdError) as caught:
        _sink_plan(sql, sink)
    error = caught.value
    assert error.code is code, f"{error.code} != {code}: {error}"
    assert needle in error.message, error.message
    assert error.hint
    return error


# the declaration


def test_a_returns_sink_declaration_rides_out_on_the_resolved_query() -> None:
    declared = _resolved(SINK_RECIPE).wasm["post_rows"]
    assert declared.returns == "sink"
    assert declared.is_sink and not declared.is_value
    assert declared.stream_kind == "video"
    assert [p.name for p in declared.value_params] == ["url"]
    assert declared.reads is not None and declared.reads.name == "boxes"


def test_a_sink_takes_the_annotation_column_like_any_consumer() -> None:
    declared = _resolved(SINK_RECIPE).wasm["post_rows"]
    assert declared.stream_arity == 1
    assert declared.signature.endswith("RETURNS sink")


def test_a_sink_declaring_no_parameters_reads_every_row_off_the_select_list() -> None:
    """No leading stream and no value either: the COPY's SELECT list is all
    there is, and the destination call itself stays empty."""
    sql = (
        "CREATE FUNCTION drain() RETURNS sink\n"
        f"  AS '{SINK_MODULE}', 'drain' LANGUAGE wasm;\n"
        + "COPY (SELECT f.video[1] FROM input('a.mp4') f) TO drain()"
    )
    declared = _resolved(sql).wasm["drain"]
    assert declared.stream_params == ()
    assert declared.reads_rows_from_select is True


def test_a_sink_opening_with_a_value_parameter_reads_rows_from_the_select_list() -> None:
    """The new form: a value parameter first means no stream parameter at
    all, so the rows come from the SELECT list rather than named arguments."""
    sql = (
        "CREATE FUNCTION drain(url text) RETURNS sink\n"
        f"  AS '{SINK_MODULE}', 'drain' LANGUAGE wasm;\n"
        + "COPY (SELECT f.video[1] FROM input('a.mp4') f) TO drain('http://x/')"
    )
    declared = _resolved(sql).wasm["drain"]
    assert declared.stream_params == ()
    assert declared.reads_rows_from_select is True
    assert [p.name for p in declared.value_params] == ["url"]
    with pytest.raises(ValueError, match="kinds come from the rows"):
        declared.stream_kind


def test_an_audio_sink_reads_audio() -> None:
    sql = (
        "CREATE FUNCTION drain(a audio_stream) RETURNS sink\n"
        f"  AS '{SINK_MODULE}', 'drain' LANGUAGE wasm;\n"
        + "COPY (SELECT f.audio[1] FROM input('a.mp4') f) TO drain()"
    )
    declared = _resolved(sql).wasm["drain"]
    assert declared.is_sink and declared.stream_kind == "audio"


# where a sink may be written


def test_a_sink_call_in_the_select_list_is_refused() -> None:
    sql = (
        PLAIN_SINK_DECLARE
        + "COPY (SELECT drain(f.video[1]) FROM input('a.mp4') f) TO 'out.mp4'"
    )
    error = _sink_rejects(
        sql, ErrorCode.UNSUPPORTED_SQL, "is not a COPY destination"
    )
    assert "COPY (SELECT <streams>) TO" in (error.hint or "")


def test_a_sink_call_in_from_is_refused() -> None:
    sql = PLAIN_SINK_DECLARE + "COPY (SELECT t.video FROM drain('x') t) TO 'out.mp4'"
    _sink_rejects(sql, ErrorCode.UNSUPPORTED_SQL, "returns sink, not a table")


def test_a_stream_function_in_to_position_is_refused() -> None:
    sql = DECLARE + "COPY (SELECT f.video[1] FROM input('a.mp4') f) TO invert()"
    with pytest.raises(FfrwdError) as caught:
        compile_all(sql, describe=lambda path: _described())
    error = caught.value
    assert error.code is ErrorCode.UNSUPPORTED_SQL
    assert "which returns video_stream" in error.message
    assert "RETURNS sink" in (error.hint or "")


def test_an_undeclared_name_in_to_position_is_refused() -> None:
    sql = "COPY (SELECT f.video[1] FROM input('a.mp4') f) TO nowhere('x')"
    with pytest.raises(FfrwdError) as caught:
        _sink_plan(sql)
    error = caught.value
    assert error.code is ErrorCode.UNSUPPORTED_SQL
    assert "no sink function declares" in error.message


def test_with_options_on_a_sink_destination_are_refused() -> None:
    # A FRAME sink: the options a packet sink would read as its encoder have
    # nothing to shape here, and the judgment needs the describe to be made.
    sql = (
        PLAIN_SINK_DECLARE
        + "COPY (SELECT f.video[1] FROM input('a.mp4') f) TO drain() "
        "WITH (video_codec 'libx264')"
    )
    _sink_rejects(
        sql,
        ErrorCode.UNSUPPORTED_SQL,
        "takes no WITH options",
        _sink_described(name="drain", params={}, reads_rows=False),
    )


def test_a_star_select_over_a_sink_destination_is_refused() -> None:
    sql = PLAIN_SINK_DECLARE + "COPY (SELECT * FROM input('a.mp4') f) TO drain()"
    _sink_rejects(sql, ErrorCode.UNSUPPORTED_SQL, "SELECT list is a star")


def test_an_uncalled_sink_declaration_is_refused() -> None:
    sql = (
        PLAIN_SINK_DECLARE
        + "COPY (SELECT f.video[1] FROM input('a.mp4') f) TO 'out.mp4'"
    )
    _sink_rejects(sql, ErrorCode.UNSUPPORTED_SQL, "is never called")


# the plan


def test_a_sink_copy_compiles_to_a_feeder_and_a_sink_process() -> None:
    plan = _sink_plan(
        PLAIN_SINK_QUERY, _sink_described(name="drain", params={}, reads_rows=False)
    ).plan
    assert plan is not None
    assert len(plan.ffmpeg) == 1 and len(plan.sidecars) == 1
    process = plan.sidecars[0]
    assert process.sink is True
    assert process.outputs == ()
    assert process.rows is None


def test_a_frame_sink_composes_with_the_module_feeding_it() -> None:
    """The always-cut rule keys on what a sink CONSUMES, not on it being a
    sink: a frame sink reads decoded payloads, so a module feeding it glues
    into the same sidecar - no encoder, no pipe between them."""
    sql = (
        DECLARE
        + PLAIN_SINK_DECLARE
        + "COPY (SELECT invert(f.video[1]) FROM input('a.mp4') f) TO drain()"
    )
    plan = _sink_plan(
        sql, _sink_described(name="drain", params={}, reads_rows=False)
    ).plan
    assert plan is not None
    assert len(plan.sidecars) == 1, "one region hosts the filter and the sink"
    region = plan.sidecars[0]
    assert [b.name for b in region.modules] == ["invert", "post_rows"]
    assert region.sink is True
    # The one stream edge is the region's boundary: decoded frames in,
    # nothing between the two modules, and no encoder anywhere.
    incoming = [e for e in plan.stream_edges if e.target == region.id]
    assert [e.ref for e in incoming] == ["src:f:v:0"]
    assert incoming[0].format.codec == RAWVIDEO


def test_a_sink_process_maps_a_null_output() -> None:
    plan = _sink_plan(
        PLAIN_SINK_QUERY, _sink_described(name="drain", params={}, reads_rows=False)
    ).plan
    assert plan is not None
    argv = wasm.shown_argv(plan.sidecars[0])
    assert argv[-3:] == ["-f", "null", "-"]
    assert argv[:6] == ["ffrwd-wasm", "-f", "nut", "-i", "pipe:0", "-m"]


def test_a_producer_and_its_sink_share_one_process() -> None:
    plan = _sink_plan(SINK_RECIPE).plan
    assert plan is not None
    assert len(plan.sidecars) == 1
    argv = wasm.shown_argv(plan.sidecars[0])
    joined = " ".join(argv)
    assert "[0:v]stats[n1];[n1]post_rows=" in joined
    assert argv[-3:] == ["-f", "null", "-"]


def test_the_destination_calls_value_arguments_become_module_params() -> None:
    plan = _sink_plan(SINK_RECIPE).plan
    assert plan is not None
    argv = wasm.shown_argv(plan.sidecars[0])
    network = argv[argv.index("-filter_complex") + 1]
    assert "post_rows=url=" in network


def test_a_value_argument_the_sink_module_never_declared_is_refused() -> None:
    with pytest.raises(FfrwdError) as caught:
        _sink_plan(SINK_RECIPE, _sink_described(params={"endpoint": {"type": "string"}}))
    error = caught.value
    assert error.code is ErrorCode.UDF_ARG_TYPE
    assert "has no parameter 'url'" in error.message


# the grants


def test_an_http_module_earns_its_grant_on_the_argv() -> None:
    plan = _sink_plan(SINK_RECIPE, _sink_described(http=True)).plan
    assert plan is not None
    argv = wasm.shown_argv(plan.sidecars[0])
    at = argv.index("-http")
    assert argv[at + 1] == SINK_MODULE
    assert "-net" not in argv


def test_a_udp_module_earns_its_grant_on_the_argv() -> None:
    plan = _sink_plan(SINK_RECIPE, _sink_described(udp=True)).plan
    assert plan is not None
    argv = wasm.shown_argv(plan.sidecars[0])
    at = argv.index("-net")
    assert argv[at + 1] == SINK_MODULE
    assert "-http" not in argv


def test_a_module_needing_no_effect_is_granted_none() -> None:
    plan = _sink_plan(SINK_RECIPE).plan
    assert plan is not None
    argv = wasm.shown_argv(plan.sidecars[0])
    assert "-http" not in argv and "-net" not in argv


def test_the_effect_flags_are_read_off_the_describe_payload() -> None:
    payload = {"world": "ffrwd:av@0.9.0", "name": "p", "http": True, "udp": True}
    described = wasm._described(SINK_MODULE, payload)
    assert described.http is True and described.udp is True
    bare = wasm._described(SINK_MODULE, {"world": "ffrwd:av@0.9.0", "name": "p"})
    assert not bare.http and not bare.udp


# the namespaced spelling


def test_a_packages_sink_is_called_in_to_position(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir(parents=True)
    (tmp_path / "src" / "tools.sql").write_text(
        "CREATE FUNCTION drain(v video_stream) RETURNS sink\n"
        "  AS 'modules/drain.wasm', 'drain' LANGUAGE wasm;\n",
        encoding="utf-8",
    )
    (tmp_path / "ffrwd.json").write_text(
        json.dumps(
            {
                "name": "ffrwd/tools",
                "version": "1.0.0",
                "lib": {"drain": "src/tools.sql"},
            }
        ),
        encoding="utf-8",
    )
    packages = discover(tmp_path)
    assert packages is not None
    sql = "COPY (SELECT f.video[1] FROM input('a.mp4') f) TO ffrwd.tools.drain()"
    resolved = resolve(parse(sql), packages=packages)
    assert list(resolved.wasm) == ["ffrwd.tools.drain"]
    assert resolved.sinks[0].module_sink == "ffrwd.tools.drain"
    assert resolved.sinks[0].path is None


# -- source modules: RETURNS source, the mirror of RETURNS sink -----------


def test_a_returns_source_declaration_is_readable_without_a_query(tmp_path: Path) -> None:
    """A source's own declaration checks the same way a sink's does: reading
    it needs no call, so `package_modules` is what proves the signature alone,
    with none of the "used"/FROM-binding questions a script would raise."""
    (tmp_path / "src").mkdir(parents=True)
    (tmp_path / "src" / "sources.sql").write_text(
        "CREATE FUNCTION subscribe(relay text, broadcast text) RETURNS source\n"
        "  AS 'modules/subscribe.wasm', 'subscribe' LANGUAGE wasm;\n",
        encoding="utf-8",
    )
    (tmp_path / "ffrwd.json").write_text(
        json.dumps(
            {
                "name": "ffrwd/moq",
                "version": "1.0.0",
                "lib": {"subscribe": "src/sources.sql"},
            }
        ),
        encoding="utf-8",
    )
    packages = discover(tmp_path)
    assert packages is not None
    package = packages.packages["ffrwd/moq"]
    modules = package_modules(package)
    assert len(modules) == 1
    declared = modules[0]
    assert declared.is_source
    assert not declared.is_value and not declared.is_sink
    assert declared.stream_params == ()
    assert [p.name for p in declared.value_params] == ["relay", "broadcast"]
    with pytest.raises(ValueError, match="kinds come from its catalog"):
        declared.stream_kind


def test_a_stream_parameter_on_a_source_is_refused_at_the_declaration() -> None:
    sql = (
        "CREATE FUNCTION subscribe(v video_stream) RETURNS source\n"
        f"  AS '{MODULE}', 'subscribe' LANGUAGE wasm;\n"
        "COPY (SELECT f.video[1] FROM input('a.mp4') f) TO 'out.mp4'"
    )
    error = _rejects(
        sql, ErrorCode.UNSUPPORTED_SQL, "declares the stream parameter 'v'"
    )
    assert error.hint is not None and "produces streams and reads none" in error.hint


def test_a_source_called_in_select_is_refused() -> None:
    """A source is a row source, not a stream function: it belongs in FROM."""
    sql = (
        "CREATE FUNCTION subscribe(relay text) RETURNS source\n"
        f"  AS '{MODULE}', 'subscribe' LANGUAGE wasm;\n"
        "COPY (SELECT subscribe('r') FROM input('a.mp4') f) TO 'out.mp4'"
    )
    error = _rejects(
        sql, ErrorCode.UNSUPPORTED_SQL, "returns source, and this call is not in FROM"
    )
    assert error.hint is not None and "FROM" in error.hint


# -- packet sinks: the encoded edge ----------------------------------------

PACKET_MODULE = "modules/packet_stats.wasm"

PACKET_DECLARE = (
    "CREATE FUNCTION packet_stats(v video_stream) RETURNS sink\n"
    f"  AS '{PACKET_MODULE}', 'packet_stats' LANGUAGE wasm;\n"
)
PACKET_QUERY = (
    PACKET_DECLARE + "COPY (SELECT f.video[1] FROM input('a.mp4') f) TO packet_stats()"
)

PACKET_ROWS_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {"packets": {"type": "integer"}},
}


def _packet_described(
    *,
    name: str = "packet_stats",
    world: str = "ffrwd:av@0.10.0",
    video_codecs: tuple[str, ...] = (),
    audio_codecs: tuple[str, ...] = (),
    video_streams: wasm.SinkArity = "one",
    audio_streams: wasm.SinkArity = "none",
    rows: bool = True,
) -> Described:
    return Described(
        world=world,
        name=name,
        version="0.1.0",
        params_schema={"type": "object", "additionalProperties": False, "properties": {}},
        rows_schema=PACKET_ROWS_SCHEMA if rows else None,
        video_codecs=video_codecs,
        audio_codecs=audio_codecs,
        video_streams=video_streams,
        audio_streams=audio_streams,
    )


def _packet_plan(sql: str, sink: Described | None = None) -> Compiled:
    modules = {
        PACKET_MODULE: sink or _packet_described(),
        MODULE: _described(),
        STATS_MODULE: _stats_described(),
    }
    return compile_all(sql, describe=lambda path: modules[path])


def _packet_rejects(
    sql: str, needle: str, sink: Described | None = None
) -> FfrwdError:
    with pytest.raises(FfrwdError) as caught:
        _packet_plan(sql, sink)
    error = caught.value
    assert error.code is ErrorCode.UNSUPPORTED_SQL, str(error)
    assert needle in error.message, error.message
    assert error.line is not None
    assert error.hint
    return error


# the describe payload


def test_the_codecs_list_is_read_off_the_describe_payload() -> None:
    payload = {"world": "ffrwd:av@0.10.0", "name": "p", "video_codecs": ["hevc", "h264"]}
    described = wasm._described(PACKET_MODULE, payload)
    assert described.video_codecs == ("hevc", "h264")
    assert described.packet_sink is True
    assert described.kind is None


def test_an_absent_codecs_key_is_no_packet_sink() -> None:
    described = wasm._described(
        PACKET_MODULE, {"world": "ffrwd:av@0.10.0", "name": "p"}
    )
    assert described.video_codecs is None
    assert described.packet_sink is False


def test_an_empty_codecs_list_still_marks_the_packet_sink() -> None:
    described = wasm._described(
        PACKET_MODULE, {"world": "ffrwd:av@0.10.0", "name": "p", "video_codecs": []}
    )
    assert described.video_codecs == ()
    assert described.packet_sink is True


def test_the_worlds_hosting_packet_sinks_start_at_the_encoded_edge() -> None:
    assert "ffrwd:av@0.10.0" in wasm.WORLDS
    assert wasm.hosts_packet_sink("ffrwd:av@0.10.0")
    assert not wasm.hosts_packet_sink("ffrwd:av@0.9.0")
    assert not wasm.hosts_packet_sink("ffrwd:av@9.9.9")


# ---------------------------------------------------------------------------
# packet sources
# ---------------------------------------------------------------------------
#
# The mirror of a packet sink: a module that PRODUCES coded packets rather
# than consuming them. `probe_source` reads its compile-time catalog off a
# faked sidecar subprocess, the way tests/test_probe.py fakes ffprobe's;
# `catalog_as_probe` bridges that catalog into the shape everything
# downstream already reads a probed FILE as.

PACKET_SOURCE_MODULE = "modules/hls_source.wasm"

# Two tracks muxed into one rendition (row 0), and a second, audio-only
# rendition (row 1) -- rows read back as [0, 0, 1].
_SOURCE_CATALOG_JSON: dict[str, object] = {
    "tracks": [
        {
            "codec": "h264",
            "time_base": [1, 90000],
            "format": {"video": {"width": 1280, "height": 720}},
            "extradata": "0102ff",
            "profile": 100,
            "level": 31,
            "row": 0,
            "rendition": {
                "name": "720p",
                "bandwidth": 2500000,
                "codecs": None,
                "language": None,
            },
        },
        {
            "codec": "aac",
            "time_base": [1, 48000],
            "format": {"audio": {"sample_rate": 48000, "channels": 2}},
            "extradata": "",
            "profile": None,
            "level": None,
            "row": 0,
            "rendition": {
                "name": "720p",
                "bandwidth": 2500000,
                "codecs": None,
                "language": None,
            },
        },
        {
            "codec": "aac",
            "time_base": [1, 44100],
            "format": {"audio": {"sample_rate": 44100, "channels": 2}},
            "extradata": "",
            "profile": None,
            "level": None,
            "row": 1,
            "rendition": {
                "name": "audio-only",
                "bandwidth": 128000,
                "codecs": None,
                "language": "en",
            },
        },
    ],
    "bounded": False,
}


def _fake_wasm_run(
    monkeypatch: pytest.MonkeyPatch, stdout: str, returncode: int = 0
) -> list[list[str]]:
    """Fakes the sidecar subprocess `probe_source` spawns, the way
    tests/test_probe.py's `_fake_run` fakes ffprobe's."""
    calls: list[list[str]] = []

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        return subprocess.CompletedProcess(argv, returncode, stdout=stdout, stderr="")

    monkeypatch.setattr(wasm.subprocess, "run", fake_run)
    monkeypatch.setattr(wasm.binaries, "ffrwd_wasm_path", lambda: "ffrwd-wasm")
    return calls


def test_probe_source_spawns_the_documented_argv(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _fake_wasm_run(monkeypatch, json.dumps(_SOURCE_CATALOG_JSON))
    wasm.probe_source(PACKET_SOURCE_MODULE, '{"url": "x"}')
    assert calls == [
        ["ffrwd-wasm", "--probe", PACKET_SOURCE_MODULE, "-params", '{"url": "x"}']
    ]


def test_probe_source_parses_a_muxed_rendition_and_an_audio_only_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_wasm_run(monkeypatch, json.dumps(_SOURCE_CATALOG_JSON))
    catalog = wasm.probe_source(PACKET_SOURCE_MODULE, "{}")
    assert catalog.bounded is False
    assert [t.row for t in catalog.tracks] == [0, 0, 1]
    video, audio0, audio1 = catalog.tracks
    assert (video.kind, video.codec, video.time_base) == ("video", "h264", (1, 90000))
    assert (video.width, video.height) == (1280, 720)
    assert (video.sample_rate, video.channels) == (None, None)
    assert (video.profile, video.level) == (100, 31)
    assert (audio0.kind, audio0.codec) == ("audio", "aac")
    assert (audio0.width, audio0.height) == (None, None)
    assert (audio0.sample_rate, audio0.channels) == (48000, 2)
    assert audio1.rendition == wasm.SourceRendition(
        name="audio-only", bandwidth=128000, codecs=None, language="en"
    )


def test_extradata_hex_is_decoded_to_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_wasm_run(monkeypatch, json.dumps(_SOURCE_CATALOG_JSON))
    catalog = wasm.probe_source(PACKET_SOURCE_MODULE, "{}")
    assert catalog.tracks[0].extradata == bytes.fromhex("0102ff")
    assert catalog.tracks[1].extradata == b""


def test_a_malformed_probe_line_is_refused_naming_the_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_wasm_run(monkeypatch, "not json")
    with pytest.raises(FfrwdError) as caught:
        wasm.probe_source(PACKET_SOURCE_MODULE, "{}")
    assert PACKET_SOURCE_MODULE in caught.value.message


def test_a_probe_track_missing_its_codec_is_refused_naming_the_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_wasm_run(
        monkeypatch,
        json.dumps({"tracks": [{"time_base": [1, 90000], "row": 0}], "bounded": True}),
    )
    with pytest.raises(FfrwdError) as caught:
        wasm.probe_source(PACKET_SOURCE_MODULE, "{}")
    assert PACKET_SOURCE_MODULE in caught.value.message
    assert "track 0" in caught.value.message


def test_a_probe_track_with_an_unknown_format_arm_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_wasm_run(
        monkeypatch,
        json.dumps(
            {
                "tracks": [
                    {
                        "codec": "h264",
                        "time_base": [1, 90000],
                        "format": {"subtitle": {}},
                        "extradata": "",
                        "profile": None,
                        "level": None,
                        "row": 0,
                        "rendition": {},
                    }
                ],
                "bounded": True,
            }
        ),
    )
    with pytest.raises(FfrwdError) as caught:
        wasm.probe_source(PACKET_SOURCE_MODULE, "{}")
    assert "neither video nor audio" in caught.value.message


def test_a_probe_with_tracks_not_a_list_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_wasm_run(monkeypatch, json.dumps({"tracks": "nope", "bounded": True}))
    with pytest.raises(FfrwdError) as caught:
        wasm.probe_source(PACKET_SOURCE_MODULE, "{}")
    assert PACKET_SOURCE_MODULE in caught.value.message


def test_a_missing_sidecar_refuses_the_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(wasm.binaries, "ffrwd_wasm_path", lambda: None)
    with pytest.raises(FfrwdError) as caught:
        wasm.probe_source(PACKET_SOURCE_MODULE, "{}")
    assert caught.value.hint is not None
    assert "reinstall ffrwd" in caught.value.hint
    assert "FFRWD_WASM" in caught.value.hint


def test_catalog_as_probe_builds_two_renditions_with_counted_indexes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_wasm_run(monkeypatch, json.dumps(_SOURCE_CATALOG_JSON))
    catalog = wasm.probe_source(PACKET_SOURCE_MODULE, "{}")
    probed = wasm.catalog_as_probe("src", catalog)
    assert probed.format_name == "packet-source"
    assert probed.live is True
    assert [(s.type, s.index) for s in probed.streams] == [
        ("video", 0),
        ("audio", 0),
        ("audio", 1),
    ]
    assert len(probed.renditions) == 2
    muxed, audio_only = probed.renditions
    assert [s.type for s in muxed.streams] == ["video", "audio"]
    assert (muxed.width, muxed.height) == (1280, 720)
    assert muxed.bandwidth == 2500000
    assert [s.type for s in audio_only.streams] == ["audio"]
    assert (audio_only.width, audio_only.height) == (None, None)
    assert (audio_only.name, audio_only.language) == ("audio-only", "en")


def test_a_bounded_catalog_is_not_live(monkeypatch: pytest.MonkeyPatch) -> None:
    bounded = {**_SOURCE_CATALOG_JSON, "bounded": True}
    _fake_wasm_run(monkeypatch, json.dumps(bounded))
    catalog = wasm.probe_source(PACKET_SOURCE_MODULE, "{}")
    assert wasm.catalog_as_probe("src", catalog).live is False


def test_the_worlds_hosting_packet_sources_start_at_0_13_0() -> None:
    assert "ffrwd:av@0.13.0" in wasm.WORLDS
    assert wasm.hosts_packet_source("ffrwd:av@0.13.0")
    assert not wasm.hosts_packet_source("ffrwd:av@0.12.0")
    assert not wasm.hosts_packet_source("ffrwd:av@9.9.9")


def test_the_source_marker_reads_off_the_describe_payload() -> None:
    described = wasm._described(
        PACKET_SOURCE_MODULE, {"world": "ffrwd:av@0.13.0", "name": "s", "source": True}
    )
    assert described.source is True


def test_an_absent_source_key_is_not_a_source() -> None:
    described = wasm._described(
        PACKET_SOURCE_MODULE, {"world": "ffrwd:av@0.13.0", "name": "s"}
    )
    assert described.source is False


# -- packet source argv, and per-pad metadata on a packet sink's reads -----


def _packet_source(outputs: tuple[str, ...] = ("video", "audio")) -> SidecarProcess:
    return SidecarProcess(
        id="sidecar0",
        module=PACKET_SOURCE_MODULE,
        node="s",
        outputs=outputs,
        packet_source=True,
    )


def test_a_packet_source_writes_one_nut_pipe_per_track_and_no_input() -> None:
    argv = wasm.shown_argv(_packet_source(), ["p0", "p1"])
    assert argv == [
        "ffrwd-wasm",
        "-m",
        PACKET_SOURCE_MODULE,
        "-f",
        "nut",
        "p0",
        "-f",
        "nut",
        "p1",
    ]
    assert "-i" not in argv


def test_a_packet_source_with_no_tracks_is_refused() -> None:
    with pytest.raises(FfrwdError) as caught:
        wasm.shown_argv(_packet_source(outputs=()), [])
    assert PACKET_SOURCE_MODULE in caught.value.message
    assert "no track to write" in caught.value.message


def _packet_sink(reads: int, pads: tuple[PadMeta | None, ...] = ()) -> SidecarProcess:
    return SidecarProcess(
        id="sidecar0",
        module=PACKET_SOURCE_MODULE,
        node="n0",
        outputs=("video",),
        inputs=tuple(f"src:a:v:{i}" for i in range(reads)),
        sink=True,
        packet_sink=True,
        pads=pads,
    )


def test_a_packet_sinks_pads_ride_right_after_their_own_input() -> None:
    metas = (
        PadMeta(row=0, name="720p", bandwidth=2_500_000),
        PadMeta(row=1, name="1080p", bandwidth=5_000_000),
    )
    argv = wasm.shown_argv(_packet_sink(2, metas), ["p0", "p1"])
    assert argv == [
        "ffrwd-wasm",
        "-f",
        "nut",
        "-i",
        "p0",
        "-pad",
        '{"row": 0, "rendition": {"name": "720p", "bandwidth": 2500000}}',
        "-f",
        "nut",
        "-i",
        "p1",
        "-pad",
        '{"row": 1, "rendition": {"name": "1080p", "bandwidth": 5000000}}',
        "-m",
        PACKET_SOURCE_MODULE,
        "-f",
        "null",
        "-",
    ]


def test_a_packet_sinks_reads_with_no_pads_render_todays_argv() -> None:
    """No `pads` at all -- the shape every sink built before this landed still
    renders, byte for byte: no `-pad` flag anywhere."""
    argv = wasm.shown_argv(_packet_sink(2), ["p0", "p1"])
    assert argv == [
        "ffrwd-wasm",
        "-f",
        "nut",
        "-i",
        "p0",
        "-f",
        "nut",
        "-i",
        "p1",
        "-m",
        PACKET_SOURCE_MODULE,
        "-f",
        "null",
        "-",
    ]
    assert "-pad" not in argv


def test_the_encoder_names_map_to_the_codecs_they_write() -> None:
    assert wasm.encoder_codec("libx264") == "h264"
    assert wasm.encoder_codec("libx265") == "hevc"
    assert wasm.encoder_codec("libsvtav1") == "av1"
    assert wasm.encoder_codec("h264_nvenc") == "h264"
    assert wasm.encoder_codec("av1_qsv") == "av1"
    assert wasm.encoder_codec("libvpx-vp9") is None
    # A bare codec name is no encoder, so a pinned command never carries one.
    assert wasm.encoder_codec("h264") is None


# the plan


def test_a_packet_sink_places_the_encoder_in_the_feeder() -> None:
    plan = _packet_plan(PACKET_QUERY).plan
    assert plan is not None
    assert len(plan.ffmpeg) == 1 and len(plan.sidecars) == 1
    argv = plan_argv(plan, sidecar_argv=wasm.shown_argv)
    assert argv[plan.ffmpeg[0].id] == [
        "ffmpeg", "-i", "a.mp4", "-map", "0:v:0",
        "-c:0", "libx264", "-pix_fmt:0", "yuv420p", "-f", "nut", "pipe:1",
    ]
    assert argv[plan.sidecars[0].id] == [
        "ffrwd-wasm", "-f", "nut", "-i", "pipe:0",
        "-m", PACKET_MODULE, "-f", "ndjson", "pipe:1",
    ]


def test_the_copys_encoder_options_shape_the_edge() -> None:
    sql = (
        PACKET_DECLARE
        + "COPY (SELECT f.video[1] FROM input('a.mp4') f) TO packet_stats() "
        "WITH (video_codec 'libx264', crf 26, preset 'veryfast', gop 48)"
    )
    plan = _packet_plan(sql).plan
    assert plan is not None
    argv = plan_argv(plan, sidecar_argv=wasm.shown_argv)[plan.ffmpeg[0].id]
    assert argv == [
        "ffmpeg", "-i", "a.mp4", "-map", "0:v:0",
        "-c:0", "libx264", "-pix_fmt:0", "yuv420p",
        "-crf:0", "26", "-g:0", "48", "-preset:0", "veryfast",
        "-f", "nut", "pipe:1",
    ]


def test_the_modules_first_codec_is_the_preference() -> None:
    plan = _packet_plan(PACKET_QUERY, _packet_described(video_codecs=("hevc", "h264"))).plan
    assert plan is not None
    argv = plan_argv(plan, sidecar_argv=wasm.shown_argv)[plan.ffmpeg[0].id]
    assert argv[argv.index("-c:0") + 1] == "libx265"


def test_a_hardware_encoder_spelling_satisfies_the_codec_list() -> None:
    sql = (
        PACKET_DECLARE
        + "COPY (SELECT f.video[1] FROM input('a.mp4') f) TO packet_stats() "
        "WITH (video_codec 'h264_nvenc')"
    )
    plan = _packet_plan(sql, _packet_described(video_codecs=("h264",))).plan
    assert plan is not None
    argv = plan_argv(plan, sidecar_argv=wasm.shown_argv)[plan.ffmpeg[0].id]
    assert argv[argv.index("-c:0") + 1] == "h264_nvenc"


def test_a_rowless_packet_sink_maps_a_null_output() -> None:
    plan = _packet_plan(PACKET_QUERY, _packet_described(rows=False)).plan
    assert plan is not None
    argv = wasm.shown_argv(plan.sidecars[0])
    assert argv[-3:] == ["-f", "null", "-"]


def test_the_sink_process_is_a_sink_with_rows_on_stdout() -> None:
    plan = _packet_plan(PACKET_QUERY).plan
    assert plan is not None
    process = plan.sidecars[0]
    assert process.sink is True
    assert process.outputs == ()
    assert process.rows is not None
    assert process.rows.container == "ndjson"
    assert process.rows.alias == "" and process.rows.path == ""


def test_each_consumers_edge_keeps_its_own_format() -> None:
    """Each consumer's edge keeps its own format: raw frames for the frame
    module, the encoder's output for the packet sink -- and the sink hosts
    alone, joining no module network."""
    sql = (
        DECLARE
        + PACKET_DECLARE
        + "COPY (SELECT invert(f.video[1]) FROM input('a.mp4') f) TO 'out.mp4';\n"
        "COPY (SELECT g.video[1] FROM input('a.mp4') g) TO packet_stats()"
    )
    plan = _packet_plan(sql).plan
    assert plan is not None
    assert len(plan.sidecars) == 2
    formats = {
        edge.target: edge.format
        for edge in plan.stream_edges
        if isinstance(edge.format, VideoFormat)
    }
    codecs = {formats[sidecar.id].codec for sidecar in plan.sidecars}
    assert codecs == {"rawvideo", "libx264"}
    hosts = {sidecar.module: len(sidecar.nodes) for sidecar in plan.sidecars}
    assert hosts[PACKET_MODULE] == 1


# the refusal sweep


def test_a_codec_outside_the_declared_list_is_refused() -> None:
    sql = (
        PACKET_DECLARE
        + "COPY (SELECT f.video[1] FROM input('a.mp4') f) TO packet_stats() "
        "WITH (video_codec 'libx264')"
    )
    error = _packet_rejects(
        sql, "'libx264' writes h264", _packet_described(video_codecs=("av1",))
    )
    assert "consumes av1" in error.message
    assert "av1" in (error.hint or "")


def test_an_encoder_for_no_wire_codec_is_refused() -> None:
    sql = (
        PACKET_DECLARE
        + "COPY (SELECT f.video[1] FROM input('a.mp4') f) TO packet_stats() "
        "WITH (video_codec 'libvpx-vp9')"
    )
    error = _packet_rejects(sql, "travels as h264, hevc or av1")
    assert "'libvpx-vp9' encodes none of them" in error.message


def test_a_preference_the_wire_cannot_carry_is_refused() -> None:
    error = _packet_rejects(
        PACKET_QUERY,
        "consumes vp9, and the stream edge carries h264, hevc or av1",
        _packet_described(video_codecs=("vp9",)),
    )
    assert PACKET_MODULE in error.message


def test_an_option_that_shapes_no_encoder_is_refused() -> None:
    sql = (
        PACKET_DECLARE
        + "COPY (SELECT f.video[1] FROM input('a.mp4') f) TO packet_stats() "
        "WITH (faststart true)"
    )
    with pytest.raises(FfrwdError) as caught:
        _packet_plan(sql)
    error = caught.value
    assert error.code is ErrorCode.UNKNOWN_SINK_OPTION
    assert "'faststart' does not shape the encoder" in error.message
    assert "video encoder options" in (error.hint or "")


AUDIO_SINK_SQL = (
    "CREATE FUNCTION packet_stats(v video_stream, a audio_stream) RETURNS sink\n"
    f"  AS '{PACKET_MODULE}', 'packet_stats' LANGUAGE wasm;\n"
    "COPY (SELECT f.video[1], f.audio[1] FROM input('a.mp4') f) TO packet_stats()"
)


def test_an_audio_stream_reaches_a_packet_sink_that_reads_one() -> None:
    """The stream edge carries encoded audio, so an audio pad has a wire to
    arrive on: the feeding ffmpeg encodes it rather than writing pcm."""
    plan = _packet_plan(
        AUDIO_SINK_SQL,
        _packet_described(audio_codecs=("aac",), audio_streams="one"),
    ).plan
    assert plan is not None
    argv = plan_argv(
        plan,
        sidecar_argv=wasm.shown_argv,
        pipe_path=lambda edge, side: f"pipes/{edge.ref}-{side}",
    )[plan.ffmpeg[0].id]
    assert "aac" in argv, argv
    assert not any(a.startswith("pcm_") for a in argv), argv


def test_a_sink_reading_any_audio_takes_a_declaration_with_none() -> None:
    """`any` is the arity that says "and works without": a query naming only
    video reaches the same module a query naming both does."""
    plan = _packet_plan(
        PACKET_QUERY,
        _packet_described(audio_codecs=("aac",), audio_streams="any"),
    ).plan
    assert plan is not None
    assert len(plan.sidecars) == 1


def test_a_sink_reading_any_audio_takes_several_of_it() -> None:
    """`any` accepts what `many` does as well: a declaration that can hand
    over SEVERAL streams of the kind."""
    sql = (
        "CREATE FUNCTION packet_stats(v video_stream, a audio_stream, b audio_stream)\n"
        "  RETURNS sink\n"
        f"  AS '{PACKET_MODULE}', 'packet_stats' LANGUAGE wasm;\n"
        "COPY (SELECT s.video[1], s.audio[1], s.audio[2] "
        "FROM input('a.mp4') s) TO packet_stats()"
    )
    plan = _packet_plan(
        sql, _packet_described(audio_codecs=("aac",), audio_streams="any")
    ).plan
    assert plan is not None


def test_a_sink_reading_one_audio_refuses_several_of_it() -> None:
    sql = (
        "CREATE FUNCTION packet_stats(v video_stream, a audio_stream, b audio_stream)\n"
        "  RETURNS sink\n"
        f"  AS '{PACKET_MODULE}', 'packet_stats' LANGUAGE wasm;\n"
        "COPY (SELECT s.video[1], s.audio[1], s.audio[2] "
        "FROM input('a.mp4') s) TO packet_stats()"
    )
    _packet_rejects(
        sql,
        "reads one audio stream",
        _packet_described(audio_codecs=("aac",), audio_streams="one"),
    )


def test_a_sink_reading_one_audio_refuses_a_declaration_with_none() -> None:
    """`one` and `many` still mean what they said: the kind must arrive."""
    _packet_rejects(
        PACKET_QUERY,
        "declares no audio_stream parameter",
        _packet_described(audio_codecs=("aac",), audio_streams="one"),
    )


def test_an_audio_codec_the_edge_cannot_carry_is_refused() -> None:
    """The refusal that remains: the edge hands through the codecs the
    sidecar's NUT reader names, and nothing else."""
    error = _packet_rejects(
        AUDIO_SINK_SQL,
        "consumes opus audio",
        _packet_described(audio_codecs=("opus",), audio_streams="one"),
    )
    assert "aac" in error.message


def test_an_audio_encoder_writing_what_the_edge_cannot_carry_is_refused() -> None:
    error = _packet_rejects(
        AUDIO_SINK_SQL.replace(
            "TO packet_stats()", "TO packet_stats() WITH (audio_codec 'libmp3lame')"
        ),
        "encodes none of them",
        _packet_described(audio_codecs=("aac",), audio_streams="one"),
    )
    assert "aac" in error.message


def test_a_packet_module_declared_as_a_frame_filter_is_refused() -> None:
    sql = (
        "CREATE FUNCTION pstats(v video_stream) RETURNS video_stream\n"
        f"  AS '{PACKET_MODULE}', 'pstats' LANGUAGE wasm;\n"
        "COPY (SELECT pstats(f.video[1]) FROM input('a.mp4') f) TO 'out.mp4'"
    )
    error = _packet_rejects(sql, "hands nothing back", _packet_described(name="pstats"))
    assert "RETURNS sink" in (error.hint or "")


def test_a_sidecar_predating_the_encoded_edge_is_refused() -> None:
    error = _packet_rejects(
        PACKET_QUERY,
        "ffrwd:av@0.9.0 cannot hand them through",
        _packet_described(world="ffrwd:av@0.9.0"),
    )
    assert "0.10.0" in (error.hint or "")


MODULE_FED_SINK_SQL = (
    DECLARE
    + PACKET_DECLARE
    + "COPY (SELECT invert(f.video[1]) FROM input('a.mp4') f) TO packet_stats()"
)


def _stage_argv(sql: str, sink: Described | None = None) -> tuple[list[str], object]:
    """The interposed encoding stage's argv, and the plan it sits in."""
    plan = _packet_plan(sql, sink).plan
    assert plan is not None
    argv = plan_argv(
        plan,
        sidecar_argv=wasm.shown_argv,
        pipe_path=lambda edge, side: f"pipes/{edge.source}-{edge.target}-{side}",
    )
    sink_region = next(s for s in plan.sidecars if s.packet_sink)
    stage_id = next(
        e.source
        for e in plan.stream_edges
        if e.target == sink_region.id and e.source.startswith("ffmpeg")
    )
    return argv[stage_id], plan


def test_a_module_fed_packet_sink_gets_an_encoding_stage() -> None:
    """A packet sink consumes the encoder's output; a module region emits
    decoded frames. The plan stands an encoding ffmpeg between them - the
    same fronting encoder the sink gets when its feed is an ffmpeg filter."""
    plan = _packet_plan(MODULE_FED_SINK_SQL).plan
    assert plan is not None
    assert len(plan.sidecars) == 2 and len(plan.ffmpeg) == 2
    module_region = next(s for s in plan.sidecars if not s.packet_sink)
    sink_region = next(s for s in plan.sidecars if s.packet_sink)
    into_stage = next(e for e in plan.stream_edges if e.source == module_region.id)
    into_sink = next(e for e in plan.stream_edges if e.target == sink_region.id)
    stage = plan.process(into_sink.source)
    assert isinstance(stage, FfmpegProcess)
    # module sidecar -> decoded NUT -> encoding ffmpeg -> encoded NUT -> sink.
    assert into_stage.target == into_sink.source
    assert into_stage.format.codec == RAWVIDEO
    assert into_sink.format.codec == "libx264"
    # The stage filters nothing: it reads the region's pipe and encodes.
    assert stage.graph.nodes == {}


def test_the_interposed_stage_takes_the_sinks_encoder_options() -> None:
    """The COPY's WITH options shape the interposed encoder exactly as they
    shape a direct feeder's."""
    argv, _ = _stage_argv(
        MODULE_FED_SINK_SQL.replace(
            "TO packet_stats()",
            "TO packet_stats() WITH (video_bitrate '2M', crf 23)",
        )
    )
    joined = " ".join(argv)
    assert "-c:0 libx264" in joined, joined
    assert "-b:0 2M" in joined, joined
    assert "-crf:0 23" in joined, joined


def test_a_module_fed_ladder_keeps_its_four_process_plan() -> None:
    """Module, then a filter fan-out into the sink: decode ffmpeg, module
    sidecar, one scale-and-encode ffmpeg, sink sidecar."""
    sql = (
        DECLARE
        + PACKET_DECLARE.replace("v video_stream", "v video_stream[]")
        + "COPY (SELECT array_agg(scale(invert(f.video[1]), ARRAY[640, 1280][i.i], -2)) "
        "FROM input('a.mp4') f, generate_series(1, 2) i) TO packet_stats()"
    )
    plan = _packet_plan(sql, _packet_described(video_streams="many")).plan
    assert plan is not None
    assert len(plan.sidecars) == 2 and len(plan.ffmpeg) == 2
    sink_region = next(s for s in plan.sidecars if s.packet_sink)
    into_sink = [e for e in plan.stream_edges if e.target == sink_region.id]
    assert len(into_sink) == 2
    # Both pads leave ONE encoding ffmpeg, already encoded.
    assert len({e.source for e in into_sink}) == 1
    assert all(e.format.codec == "libx264" for e in into_sink)
    encoder = plan.process(into_sink[0].source)
    assert isinstance(encoder, FfmpegProcess)
    assert any(n.filter == "scale" for n in encoder.graph.nodes.values())


def test_a_packet_sink_stays_out_of_select_position() -> None:
    sql = (
        PACKET_DECLARE
        + "COPY (SELECT packet_stats(f.video[1]) FROM input('a.mp4') f) TO 'out.mp4'"
    )
    _packet_rejects(sql, "not a COPY destination")


# -- -jobs, the worker-thread cap ------------------------------------------


def _impure() -> Described:
    """A module that declared it carries state between calls."""
    return replace(_described(), pure=False, windowed=True)


def test_the_default_leaves_the_sidecar_argv_as_it_was() -> None:
    """The sidecar sizes its own pool, so no cap is written for the default."""
    plan = _compiled(QUERY).plan
    assert plan is not None
    assert wasm.shown_argv(plan.sidecars[0]) == [
        "ffrwd-wasm", "-f", "nut", "-i", "pipe:0", "-m", MODULE, "-f", "nut", "pipe:1",
    ]
    assert "-jobs" not in wasm.shown_argv(plan.sidecars[0], jobs=None)


def test_jobs_one_is_written_as_the_serial_escape_hatch() -> None:
    plan = _compiled(QUERY).plan
    assert plan is not None
    argv = wasm.shown_argv(plan.sidecars[0], jobs=1)
    assert argv[argv.index("-jobs") + 1] == "1"


def test_a_pure_module_carries_the_jobs_it_was_asked_for() -> None:
    plan = _compiled(QUERY).plan
    assert plan is not None
    assert wasm.shown_argv(plan.sidecars[0], jobs=4) == [
        "ffrwd-wasm", "-f", "nut", "-i", "pipe:0", "-jobs", "4",
        "-m", MODULE, "-f", "nut", "pipe:1",
    ]


def test_an_impure_module_still_carries_the_cap() -> None:
    """The sidecar's scheduler is what keeps an impure module's lane serial;
    the cap rides through untouched."""
    plan = _compiled(QUERY, _impure()).plan
    assert plan is not None
    assert plan.sidecars[0].impure == (MODULE,)
    argv = wasm.shown_argv(plan.sidecars[0], jobs=4)
    assert argv[argv.index("-jobs") + 1] == "4"


def test_a_region_of_two_modules_carries_the_cap() -> None:
    """Each node is its own lane inside the sidecar, so the cap applies."""
    plan = _annotated_plan(_pair()).plan
    assert plan is not None
    argv = wasm.shown_argv(plan.sidecars[0], jobs=4)
    assert argv[argv.index("-jobs") + 1] == "4"


def test_one_impure_module_of_a_region_still_names_itself() -> None:
    plan = _annotated_plan(_pair(), blurrer=replace(_consuming(), pure=False)).plan
    assert plan is not None
    assert plan.sidecars[0].impure == (BLURRER,)
    argv = wasm.shown_argv(plan.sidecars[0], jobs=4)
    assert argv[argv.index("-jobs") + 1] == "4"


def test_a_packet_sink_carries_the_cap() -> None:
    """Its packets still reach one instance in decode order; the flag is a
    cap on the pool, not a promise of parallel decode."""
    plan = compile_all(
        PACKET_QUERY, describe=lambda path: _packet_described()
    ).plan
    assert plan is not None
    assert plan.sidecars[0].packet_sink
    argv = wasm.shown_argv(plan.sidecars[0], jobs=4)
    assert argv[argv.index("-jobs") + 1] == "4"


# -- a module parameter read off a row: one instance per row --


def test_a_module_parameter_read_off_a_row_makes_one_instance_per_row() -> None:
    """The same collapse a filter option had: three rows asking for three
    different parameter values are three instances, not one."""
    sql = (
        "CREATE FUNCTION blur(v video_stream, radius number DEFAULT 1) "
        f"RETURNS video_stream AS '{MODULE}', 'invert' LANGUAGE wasm;\n"
        "COPY (SELECT array_agg(blur(f.video[1], ARRAY[2, 4, 8][i.i])) "
        "FROM input('a.mp4') f, generate_series(1, 3) i) TO 'o.mkv'"
    )
    described = _described(params={"radius": {"type": "number"}})
    graph = lower(
        _resolved(sql), {}, registry=_snapshot_registry(), describes={MODULE: described}
    )
    instances = [node for node in graph.nodes.values() if node.filter == MODULE]
    assert [node.args["radius"] for node in instances] == [2, 4, 8]


def test_a_module_parameter_reading_no_row_still_makes_one_instance() -> None:
    sql = (
        "CREATE FUNCTION blur(v video_stream, radius number DEFAULT 1) "
        f"RETURNS video_stream AS '{MODULE}', 'invert' LANGUAGE wasm;\n"
        "COPY (SELECT array_agg(blur(f.video[1], 4)) "
        "FROM input('a.mp4') f, generate_series(1, 3) i) TO 'o.mkv'"
    )
    described = _described(params={"radius": {"type": "number"}})
    graph = lower(
        _resolved(sql), {}, registry=_snapshot_registry(), describes={MODULE: described}
    )
    assert len([n for n in graph.nodes.values() if n.filter == MODULE]) == 1


# -- a sink that reads SEVERAL streams --------------------------------------

LADDER_MODULE = "modules/packet_tally.wasm"

LADDER_DECLARE = (
    "CREATE FUNCTION tally(v video_stream[]) RETURNS sink\n"
    f"  AS '{LADDER_MODULE}', 'packet_tally' LANGUAGE wasm;\n"
)
LADDER_QUERY = LADDER_DECLARE + (
    "COPY (SELECT array_agg(scale(f.video[1], ARRAY[640, 1280, 1920][i.i], -2)) "
    "FROM input('a.mp4') f, generate_series(1, 3) i) TO tally()"
)


def _ladder_described(
    *,
    name: str = "packet_tally",
    video: str = "many",
    audio: str = "none",
) -> Described:
    return Described(
        world=WORLDS[-1],
        name=name,
        version="0.1.0",
        params_schema={"type": "object", "additionalProperties": False, "properties": {}},
        rows_schema=None,
        video_codecs=(),
        video_streams=video,  # type: ignore[arg-type]
        audio_streams=audio,  # type: ignore[arg-type]
    )


def _ladder_graph(sql: str, sink: Described | None = None) -> Graph:
    return lower(
        _resolved(sql),
        {},
        registry=_snapshot_registry(),
        describes={LADDER_MODULE: sink or _ladder_described()},
    )


def _ladder_rejects(
    sql: str, needle: str, sink: Described | None = None
) -> FfrwdError:
    with pytest.raises(FfrwdError) as caught:
        _ladder_graph(sql, sink)
    error = caught.value
    assert needle in error.message, error.message
    assert error.line is not None
    assert error.hint
    return error


def test_a_gathered_ladder_reaches_one_sink_instance() -> None:
    """The whole point: N renditions, ONE node. N instances would be N
    publishers with nothing shared between them."""
    graph = _ladder_graph(LADDER_QUERY)
    sinks = [node for node in graph.nodes.values() if node.filter == LADDER_MODULE]
    assert len(sinks) == 1
    assert len(sinks[0].inputs) == 3
    assert graph.module_sinks == [sinks[0].id]


def test_every_rendition_is_a_pad_of_that_one_instance() -> None:
    """In SELECT order: the pads are the streams the columns carried,
    flattened, and each is its own scale."""
    graph = _ladder_graph(LADDER_QUERY)
    sink = next(node for node in graph.nodes.values() if node.filter == LADDER_MODULE)
    widths = [graph.nodes[ref].args["width"] for ref in sink.inputs]
    assert widths == [640, 1280, 1920]


def test_a_sink_reading_one_stream_refuses_a_ladder() -> None:
    sql = LADDER_DECLARE.replace("video_stream[]", "video_stream") + (
        "COPY (SELECT array_agg(scale(f.video[1], ARRAY[640, 1280][i.i], -2)) "
        "FROM input('a.mp4') f, generate_series(1, 2) i) TO tally()"
    )
    error = _ladder_rejects(
        sql, "SELECT list carries 2 streams", _ladder_described(video="one")
    )
    assert error.code is ErrorCode.UDF_ARG_TYPE
    assert "reads 1 stream" in error.message
    assert "video_stream[]" in (error.hint or "")


def test_an_array_declaration_over_a_module_reading_one_is_refused() -> None:
    """The declaration can hand over several and the module reads one, so
    the mismatch is caught at the call rather than at run time."""
    error = _ladder_rejects(
        LADDER_QUERY, "reads one video stream", _ladder_described(video="one")
    )
    assert error.code is ErrorCode.UNSUPPORTED_SQL


def test_a_module_reading_no_video_refuses_a_video_declaration() -> None:
    error = _ladder_rejects(
        LADDER_QUERY, "reads none video stream", _ladder_described(video="none")
    )
    assert error.code is ErrorCode.UNSUPPORTED_SQL


def test_an_arrayed_parameter_is_refused_on_a_filter() -> None:
    """Only a sink's count comes from the query; a filter's pads are its
    own declaration, so an array has nothing to mean there."""
    sql = (
        "CREATE FUNCTION blur(v video_stream[]) RETURNS video_stream\n"
        f"  AS '{MODULE}', 'invert' LANGUAGE wasm;\n"
        "COPY (SELECT blur(f.video[1]) FROM input('a.mp4') f) TO 'o.mp4'"
    )
    with pytest.raises(FfrwdError) as caught:
        _lowered(sql)
    assert "takes video_stream[] and returns video_stream" in caught.value.message


def test_a_second_parameter_of_a_kind_cannot_follow_its_array() -> None:
    sql = (
        "CREATE FUNCTION tally(v video_stream[], w video_stream) RETURNS sink\n"
        f"  AS '{LADDER_MODULE}', 'packet_tally' LANGUAGE wasm;\n"
        "COPY (SELECT f.video[1], f.video[2] FROM input('a.mp4') f) TO tally()"
    )
    error = _ladder_rejects(sql, "takes 'w' after 'v'")
    assert "nothing of that kind can follow it" in (error.hint or "")


def test_each_pad_of_a_ladder_carries_its_own_encoder_options() -> None:
    """A rendition ladder differs by bitrate first: one value per pad, in
    the order the rows were gathered."""
    sql = LADDER_DECLARE + (
        "COPY (SELECT array_agg(scale(f.video[1], ARRAY[640, 1280, 1920][i.i], -2)) "
        "FROM input('a.mp4') f, generate_series(1, 3) i) TO tally() "
        "WITH (video_codec 'libx264', "
        "video_bitrate ARRAY['800k', '2000k', '5000k'][i.i])"
    )
    graph = _ladder_graph(sql)
    node = next(name for name in graph.packet_sinks)
    pads = graph.packet_sinks[node]
    assert [pad["video_bitrate"] for pad in pads] == ["800k", "2000k", "5000k"]
    assert {pad["video_codec"] for pad in pads} == {"libx264"}


def test_a_per_pad_option_over_more_rows_than_pads_is_refused() -> None:
    sql = LADDER_DECLARE + (
        "COPY (SELECT array_agg(f.video[1]) "
        "FROM input('a.mp4') f, generate_series(1, 3) i) TO tally() "
        "WITH (video_bitrate ARRAY['1k', '2k', '3k'][i.i])"
    )
    error = _ladder_rejects(sql, "read once per row over 3 rows")
    assert error.code is ErrorCode.ROW_COUNT_MISMATCH
    # The count is of the option's own kind: a video option binds video pads.
    assert "reads 1 video stream" in error.message


def test_the_ladder_reaches_one_sidecar_reading_every_rendition() -> None:
    """One process hosting one instance, its pads arriving on pipes of
    their own -- which is what a shared catalog and a common timeline
    need."""
    compiled = compile_all(
        LADDER_QUERY,
        describe=lambda path: _ladder_described(),
    )
    plan = compiled.plan
    assert plan is not None
    assert len(plan.sidecars) == 1
    sidecar = plan.sidecars[0]
    assert sidecar.packet_sink
    assert len(sidecar.inputs) == 3
    reads = [f"pipe{index}" for index in range(3)]
    argv = wasm.shown_argv(sidecar, reads)
    assert [argv[i + 1] for i, a in enumerate(argv) if a == "-i"] == reads
    assert argv.count("-m") == 1


def test_the_ladder_decodes_its_source_once() -> None:
    """One decode, N encodes: the three rungs leave ONE ffmpeg through a
    split, each on a pipe of its own. Three feeders would open the input
    three times and decode it three times over."""
    compiled = compile_all(LADDER_QUERY, describe=lambda path: _ladder_described())
    plan = compiled.plan
    assert plan is not None
    feeders = [p for p in plan.processes if isinstance(p, FfmpegProcess)]
    assert len(feeders) == 1
    assert len(feeders[0].graph.input_paths) == 1
    edges = [e for e in plan.stream_edges if e.target == plan.sidecars[0].id]
    assert len(edges) == 3
    assert {e.source for e in edges} == {feeders[0].id}
