"""Tests for the ``ffrwd.<name>`` macro namespace.

Self-contained: every helper below is a trimmed-down copy of test_lower.py's
conventions rather than an import from it.

Macros resolve against `ffrwd.macros.MACROS` alone (never the registry),
so most tests pass `registry=None` -- proving the offline claim by
construction rather than by mocking `shutil.which`. The one exception is the
ad-insert composition test, which also calls the ordinary `overlay` filter
and so needs a real filter set; it uses the captured reference snapshot
(tests/data/reference_registry.json), exactly as test_lower.py's own
offline-fallback tests do.
"""

from __future__ import annotations

import base64
import shutil
import subprocess
from pathlib import Path

import pytest

from ffrwd import loudnorm
from ffrwd.compiler import compile_sql
from ffrwd.emit import build_ffmpeg_args, build_ffmpeg_commands, emit
from ffrwd.errors import ErrorCode, FfrwdError
from ffrwd.inputs import INPUT_OPTIONS, option_spec
from ffrwd.ir import Graph, Node
from ffrwd.lower import lower, lower_table
from ffrwd.macros import INPUT_MACROS, MACROS
from ffrwd.parser import parse, resolve
from ffrwd.probe import ProbeResult, StreamMeta
from ffrwd.registry import Registry, load_reference

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"
SNAPSHOT_PATH = REPO_ROOT / "tests" / "data" / "reference_registry.json"


def _lower(
    sql: str,
    probes: dict[str, ProbeResult | None] | None = None,
    *,
    registry: Registry | None = None,
) -> Graph:
    return lower(resolve(parse(sql)), probes or {}, registry=registry)


def _reject(
    sql: str,
    probes: dict[str, ProbeResult | None] | None = None,
    *,
    registry: Registry | None = None,
) -> FfrwdError:
    with pytest.raises(FfrwdError) as excinfo:
        _lower(sql, probes, registry=registry)
    err = excinfo.value
    assert err.line is not None, "every rejection must be line-anchored"
    assert err.col is not None
    return err


def _reject_resolve(sql: str) -> FfrwdError:
    with pytest.raises(FfrwdError) as excinfo:
        resolve(parse(sql))
    err = excinfo.value
    assert err.line is not None
    assert err.col is not None
    return err


def _filters(g: Graph) -> list[str]:
    return [node.filter for node in g.nodes.values()]


def _probe_result(videos: int = 1, audios: int = 1) -> ProbeResult:
    """A synthetic ProbeResult -- no ffprobe, no fixture, no disk."""
    streams = [
        StreamMeta(
            type="video", index=i, metadata={}, width=320, height=240,
            fps="15/1", sample_rate=None, codec="h264",
        )
        for i in range(videos)
    ]
    streams += [
        StreamMeta(
            type="audio", index=i, metadata={}, width=None, height=None,
            fps=None, sample_rate=44100, codec="aac",
        )
        for i in range(audios)
    ]
    return ProbeResult(streams=streams)


# ---------------------------------------------------------------------------
# each expansion's node shape
# ---------------------------------------------------------------------------


def test_blur_regions_expands_to_crop_gblur_overlay() -> None:
    g = _lower(
        "SELECT ffrwd.blur_regions(a.video[1], 10, 20, 100, 50, 5) FROM input('x.mp4') a"
    )
    assert _filters(g) == ["crop", "gblur", "overlay"]
    crop, gblur, overlay = (g.nodes[f"n{i}"] for i in (1, 2, 3))
    # crop's arg KEYS are the registry's long names (out_w/out_h).
    assert crop.args == {"out_w": 100, "out_h": 50, "x": 10, "y": 20}
    assert crop.inputs == ["src:a:v:0"]
    assert crop.outputs == ["video"]
    assert gblur.args == {"sigma": 5}
    assert gblur.inputs == ["n1"]
    # `f` (src:a:v:0) is consumed TWICE: once by crop, once by overlay's base
    # pad -- the split pass inserts the `split` node, not lowering.
    assert overlay.args == {"x": 10, "y": 20}
    assert overlay.inputs == ["src:a:v:0", "n2"]
    assert overlay.outputs == ["video"]


def test_speed_expands_to_setpts() -> None:
    g = _lower("SELECT ffrwd.speed(a.video[1], 2) FROM input('x.mp4') a")
    assert _filters(g) == ["setpts"]
    node = g.nodes["n1"]
    assert node.args == {"expr": "PTS/2"}
    assert node.inputs == ["src:a:v:0"]
    assert node.outputs == ["video"]


def test_delay_expands_to_format_tpad() -> None:
    g = _lower("SELECT ffrwd.delay(a.video[1], 3) FROM input('x.mp4') a")
    assert _filters(g) == ["format", "tpad"]
    fmt, tpad = g.nodes["n1"], g.nodes["n2"]
    assert fmt.args == {"pix_fmts": "yuva420p"}
    assert fmt.inputs == ["src:a:v:0"]
    assert tpad.args == {"start_duration": 3, "stop": 1, "color": "black@0"}
    assert tpad.inputs == ["n1"]
    assert tpad.outputs == ["video"]


def test_delay_seconds_may_be_a_float() -> None:
    g = _lower("SELECT ffrwd.delay(a.video[1], 1.5) FROM input('x.mp4') a")
    assert g.nodes["n2"].args["start_duration"] == 1.5


# ---------------------------------------------------------------------------
# broadcasting: the machinery is type-driven, macros need no
# special casing
# ---------------------------------------------------------------------------


def test_delay_broadcasts_over_a_video_array() -> None:
    probes: dict[str, ProbeResult | None] = {"a": _probe_result(videos=2, audios=0)}
    g = _lower("SELECT ffrwd.delay(a.video, 2) FROM input('x.mp4') a", probes)
    assert _filters(g) == ["format", "tpad", "format", "tpad"]
    assert g.nodes["n1"].inputs == ["src:a:v:0"]
    assert g.nodes["n3"].inputs == ["src:a:v:1"]
    assert len(g.sinks[0].outputs) == 2


def test_speed_broadcasts_over_a_video_array() -> None:
    probes: dict[str, ProbeResult | None] = {"a": _probe_result(videos=2, audios=0)}
    g = _lower("SELECT ffrwd.speed(a.video, 2) FROM input('x.mp4') a", probes)
    assert _filters(g) == ["setpts", "setpts"]
    assert len(g.sinks[0].outputs) == 2


# ---------------------------------------------------------------------------
# ad-insert composition (needs a real filter set for the bare `overlay`)
# ---------------------------------------------------------------------------


def _snapshot_registry() -> Registry:
    return load_reference(SNAPSHOT_PATH)


def test_ad_insert_composition_compiles() -> None:
    g = _lower(
        "SELECT overlay(f.video[1], ffrwd.delay(p.video[1], 120), 20, 20) "
        "FROM input('f.mp4') f, input('p.mp4') p",
        registry=_snapshot_registry(),
    )
    assert _filters(g) == ["format", "tpad", "overlay"]
    overlay = g.nodes["n3"]
    assert overlay.inputs == ["src:f:v:0", "n2"]
    assert overlay.args == {"x": 20, "y": 20}


# ---------------------------------------------------------------------------
# signature checking (macros own their OWN positional signature)
# ---------------------------------------------------------------------------


def test_named_argument_is_rejected() -> None:
    err = _reject("SELECT ffrwd.speed(a.video[1], factor => 2) FROM input('x.mp4') a")
    assert err.code == ErrorCode.UNSUPPORTED_SQL
    assert "positional" in err.message


def test_wrong_arity_is_udf_arg_type() -> None:
    err = _reject("SELECT ffrwd.speed(a.video[1]) FROM input('x.mp4') a")
    assert err.code == ErrorCode.UDF_ARG_TYPE
    assert "ffrwd.speed(f, factor)" in (err.hint or "")


def test_wrong_arity_too_many_is_udf_arg_type() -> None:
    err = _reject("SELECT ffrwd.speed(a.video[1], 2, 3) FROM input('x.mp4') a")
    assert err.code == ErrorCode.UDF_ARG_TYPE


def test_literal_where_stream_expected_is_udf_arg_type() -> None:
    err = _reject("SELECT ffrwd.speed(5, 2) FROM input('x.mp4') a")
    assert err.code == ErrorCode.UDF_ARG_TYPE


def test_stream_where_literal_expected_is_udf_arg_type() -> None:
    err = _reject("SELECT ffrwd.speed(a.video[1], a.video[1]) FROM input('x.mp4') a")
    assert err.code == ErrorCode.UDF_ARG_TYPE


def test_delay_on_an_audio_stream_hints_bare_adelay() -> None:
    err = _reject("SELECT ffrwd.delay(a.audio[1], 2) FROM input('x.mp4') a")
    assert err.code == ErrorCode.UDF_ARG_TYPE
    assert "adelay" in (err.hint or "")


def test_unknown_macro_did_you_mean() -> None:
    err = _reject("SELECT ffrwd.spede(a.video[1], 2) FROM input('x.mp4') a")
    assert err.code == ErrorCode.UNKNOWN_FUNCTION
    assert "ffrwd.speed()" in (err.hint or "")


def test_unknown_macro_with_no_close_match_names_the_trio() -> None:
    err = _reject("SELECT ffrwd.zzz(a.video[1], 2) FROM input('x.mp4') a")
    assert err.code == ErrorCode.UNKNOWN_FUNCTION
    hint = err.hint or ""
    assert "blur_regions" in hint and "speed" in hint and "delay" in hint


# ---------------------------------------------------------------------------
# ffrwd.loudnorm2: the one macro with named options, and the one whose
# presence turns the compile into a two-command sequence
# ---------------------------------------------------------------------------

_LOUDNORM2 = (
    "SELECT ffrwd.loudnorm2(a.audio[1], I => -16, TP => -1.5, LRA => 11) "
    "FROM input('x.mp4') a"
)


def _loudnorm2_node(g: Graph) -> Node:
    (node,) = [n for n in g.nodes.values() if n.filter == loudnorm.FILTER]
    return node


def test_loudnorm2_node_carries_only_what_was_written() -> None:
    g = _lower(_LOUDNORM2)
    node = _loudnorm2_node(g)
    assert node.args == {"I": -16, "TP": -1.5, "LRA": 11}
    assert node.inputs == ["src:a:a:0"]
    assert node.outputs == ["audio"]


def test_loudnorm2_is_not_the_bare_loudnorm_filter() -> None:
    """A pseudo-filter in the IR: the phase is what decides the real arguments,
    so a bare `loudnorm(...)` call stays an ordinary one-pass node."""
    assert _filters(_lower(_LOUDNORM2)) == ["loudnorm2"]


def test_loudnorm2_options_are_all_optional() -> None:
    g = _lower("SELECT ffrwd.loudnorm2(a.audio[1]) FROM input('x.mp4') a")
    assert _loudnorm2_node(g).args == {}


def test_loudnorm2_options_render_in_the_macros_order_not_the_written_one() -> None:
    g = _lower(
        "SELECT ffrwd.loudnorm2(a.audio[1], LRA => 11, I => -16) "
        "FROM input('x.mp4') a"
    )
    assert list(_loudnorm2_node(g).args) == ["I", "LRA"]


def test_loudnorm2_rejects_an_option_it_does_not_have() -> None:
    err = _reject(
        "SELECT ffrwd.loudnorm2(a.audio[1], dual_mono => 1) FROM input('x.mp4') a"
    )
    assert err.code == ErrorCode.UDF_ARG_TYPE
    assert "no 'dual_mono' option" in err.message
    assert "I => ..." in (err.hint or "")


def test_loudnorm2_rejects_an_option_set_twice() -> None:
    """resolve's own duplicate-kwarg rule covers this, macro or filter alike."""
    err = _reject_resolve(
        "SELECT ffrwd.loudnorm2(a.audio[1], I => -16, I => -23) FROM input('x.mp4') a"
    )
    assert err.code == ErrorCode.UNSUPPORTED_SQL
    assert "duplicate named argument 'I'" in err.message


def test_loudnorm2_rejects_a_non_numeric_option_value() -> None:
    err = _reject(
        "SELECT ffrwd.loudnorm2(a.audio[1], I => 'loud') FROM input('x.mp4') a"
    )
    assert err.code == ErrorCode.UDF_ARG_TYPE
    assert "numeric literal" in err.message


def test_loudnorm2_rejects_a_positional_option() -> None:
    err = _reject("SELECT ffrwd.loudnorm2(a.audio[1], -16) FROM input('x.mp4') a")
    assert err.code == ErrorCode.UDF_ARG_TYPE
    assert "takes 1 argument" in err.message


def test_loudnorm2_rejects_a_video_stream() -> None:
    err = _reject("SELECT ffrwd.loudnorm2(a.video[1]) FROM input('x.mp4') a")
    assert err.code == ErrorCode.UDF_ARG_TYPE
    assert "audio stream" in err.message


def test_loudnorm2_signature_names_its_named_options() -> None:
    assert MACROS["loudnorm2"].signature == (
        "ffrwd.loudnorm2(stream, I => ..., TP => ..., LRA => ...)"
    )


def test_loudnorm2_compiles_with_no_registry() -> None:
    g = _lower(_LOUDNORM2, registry=None)
    assert _filters(g) == ["loudnorm2"]


# both phases' filtergraphs, off one graph


def test_loudnorm2_renders_a_second_filtergraph_for_the_measuring_pass() -> None:
    e = emit(_lower(_LOUDNORM2))
    assert e.measure_filter_complex == (
        "[0:a:0]loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json[out0]"
    )


def test_loudnorm2_correction_phase_splices_the_measurements() -> None:
    e = emit(_lower(_LOUDNORM2))
    assert e.filter_complex == (
        "[0:a:0]loudnorm=I=-16:TP=-1.5:LRA=11"
        ":measured_I=${FFRWD_LN_I}"
        ":measured_TP=${FFRWD_LN_TP}"
        ":measured_LRA=${FFRWD_LN_LRA}"
        ":measured_thresh=${FFRWD_LN_THRESH}"
        ":offset=${FFRWD_LN_OFFSET}"
        ":linear=true[out0]"
    )


def test_a_graph_without_loudnorm2_has_no_measuring_filtergraph() -> None:
    e = emit(_lower("SELECT ffrwd.speed(a.video[1], 2) FROM input('x.mp4') a"))
    assert e.measure_filter_complex == ""


def test_loudnorm2_compiles_to_two_commands() -> None:
    e = emit(_lower(_LOUDNORM2))
    measure, correct = build_ffmpeg_commands(e, "out.m4a")
    assert measure[-3:] == ["-f", "null", "-"]
    assert measure[measure.index("-filter_complex") + 1] == e.measure_filter_complex
    assert correct[-1] == "out.m4a"
    assert correct[correct.index("-filter_complex") + 1] == e.filter_complex


def test_the_measuring_pass_drops_the_sinks_options() -> None:
    """It measures and muxes nothing, so encoding anything there is work
    thrown away -- unlike two_pass, whose pass 1 must encode identically."""
    g = _lower(
        "COPY (SELECT ffrwd.loudnorm2(a.audio[1], I => -16) FROM input('x.mp4') a) "
        "TO 'out.m4a' WITH (audio_codec 'aac')"
    )
    measure, correct = build_ffmpeg_commands(emit(g))
    assert "aac" not in measure
    assert correct[-3:] == ["-c:0", "aac", "out.m4a"]


def test_the_measuring_pass_keeps_filtered_maps_and_drops_passthrough_ones() -> None:
    """A filtergraph pad with no consumer is a hard ffmpeg error, so a
    filtered map stays; a stream-copied one teaches the measurement nothing."""
    g = _lower(
        "SELECT a.video[1], ffrwd.loudnorm2(a.audio[1]) FROM input('x.mp4') a"
    )
    measure, correct = build_ffmpeg_commands(emit(g), "out.mkv")
    assert measure.count("-map") == 1
    assert measure[measure.index("-map") + 1] == "[out1]"
    assert correct.count("-map") == 2


def test_build_ffmpeg_args_refuses_a_loudnorm2_emitted() -> None:
    with pytest.raises(ValueError, match="sequence of commands"):
        build_ffmpeg_args(emit(_lower(_LOUDNORM2)), "out.m4a")


# the v1 limits


def test_two_loudnorm2_calls_are_rejected() -> None:
    err = _reject(
        "SELECT ffrwd.loudnorm2(a.audio[1]), ffrwd.loudnorm2(a.audio[2]) "
        "FROM input('x.mp4') a"
    )
    assert err.code == ErrorCode.UNSUPPORTED_SQL
    assert "one ffrwd.loudnorm2() per query, got 2" in err.message


def test_a_broadcast_loudnorm2_is_counted_as_the_several_it_is() -> None:
    probes: dict[str, ProbeResult | None] = {"a": _probe_result(videos=0, audios=2)}
    err = _reject("SELECT ffrwd.loudnorm2(a.audio) FROM input('x.mp4') a", probes)
    assert err.code == ErrorCode.UNSUPPORTED_SQL
    assert "got 2" in err.message


def test_a_single_audio_broadcast_is_still_one_call() -> None:
    probes: dict[str, ProbeResult | None] = {"a": _probe_result(videos=0, audios=1)}
    g = _lower("SELECT ffrwd.loudnorm2(a.audio) FROM input('x.mp4') a", probes)
    assert _filters(g) == ["loudnorm2"]


def test_loudnorm2_with_two_pass_is_rejected() -> None:
    err = _reject(
        "COPY (SELECT a.video[1], ffrwd.loudnorm2(a.audio[1]) FROM input('x.mp4') a) "
        "TO 'out.mp4' WITH (video_codec 'libx264', video_bitrate '2500k', two_pass true)"
    )
    assert err.code == ErrorCode.UNSUPPORTED_SQL
    assert "'two_pass' and ffrwd.loudnorm2()" in err.message


def test_loudnorm2_in_a_fanout_copy_is_rejected() -> None:
    probes: dict[str, ProbeResult | None] = {"a": _probe_result(videos=0, audios=2)}
    err = _reject(
        "COPY (SELECT ffrwd.loudnorm2(t) FROM input('x.mp4') a, "
        "unnest(a.audio) t) TO (t.index::text || '.m4a')",
        probes,
    )
    assert err.code == ErrorCode.UNSUPPORTED_SQL
    assert "fan-out TO" in err.message


def test_loudnorm2_in_a_table_query_is_rejected() -> None:
    with pytest.raises(FfrwdError) as excinfo:
        lower_table(
            resolve(parse("SELECT ffrwd.loudnorm2(a.audio[1]) FROM input('x.mp4') a")),
            {},
            registry=None,
        )
    err = excinfo.value
    assert err.code == ErrorCode.UNSUPPORTED_SQL
    assert "table query filters nothing" in err.message


# ---------------------------------------------------------------------------
# reserved name + bare-column hint (parser.py, mirrors the ffmpeg namespace)
# ---------------------------------------------------------------------------


def test_ffrwd_alias_is_reserved() -> None:
    err = _reject_resolve("SELECT frame FROM input('x.mp4') ffrwd")
    assert err.code == ErrorCode.UNSUPPORTED_SQL
    assert "reserved" in err.message


def test_ffrwd_cte_name_is_reserved() -> None:
    err = _reject_resolve(
        "WITH ffrwd AS (SELECT a.video[1] FROM input('x.mp4') a) "
        "SELECT frame FROM ffrwd"
    )
    assert err.code == ErrorCode.UNSUPPORTED_SQL
    assert "reserved" in err.message


def test_bare_ffrwd_dot_column_hints_it_is_a_call() -> None:
    err = _reject_resolve("SELECT ffrwd.speed FROM input('x.mp4') a")
    assert err.code == ErrorCode.UNKNOWN_ALIAS
    assert "is a call, not a column" in (err.hint or "")


# ---------------------------------------------------------------------------
# offline: macros need no registry at all
# ---------------------------------------------------------------------------


def test_macros_compile_with_no_registry() -> None:
    g = _lower("SELECT ffrwd.speed(a.video[1], 2) FROM input('x.mp4') a", registry=None)
    assert _filters(g) == ["setpts"]


def test_blur_regions_compiles_with_no_registry() -> None:
    g = _lower(
        "SELECT ffrwd.blur_regions(a.video[1], 0, 0, 10, 10, 2) FROM input('x.mp4') a",
        registry=None,
    )
    assert _filters(g) == ["crop", "gblur", "overlay"]


# ---------------------------------------------------------------------------
# exec: the ad-insert composition actually runs
# ---------------------------------------------------------------------------


def _sql_path(path: Path) -> str:
    return path.resolve().as_posix()


def _ffprobe_duration(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(path),
        ],
        capture_output=True, text=True, timeout=30,
    )
    return float(result.stdout.strip())


@pytest.mark.exec
def test_ad_insert_composition_execs(tmp_path: Path) -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg/ffprobe not found on PATH")
    base = FIXTURES_DIR / "av2.mp4"
    ad = FIXTURES_DIR / "testsrc.mp4"
    if not base.exists() or not ad.exists():
        pytest.skip("fixtures missing (run scripts/gen_fixtures.py first)")
    out_path = tmp_path / "ad_insert.mp4"
    query = (
        "SELECT overlay(f.video[1], ffrwd.delay(p.video[1], 1), 20, 20) "
        f"FROM input('{_sql_path(base)}') f, input('{_sql_path(ad)}') p"
    )
    graph = compile_sql(query)
    emitted = emit(graph)
    args = build_ffmpeg_args(emitted, str(out_path))
    args.insert(1, "-y")
    result = subprocess.run(args, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert out_path.exists()
    assert _ffprobe_duration(out_path) > 0


# ---------------------------------------------------------------------------
# the input-minting macro: ffrwd.empty_captions()
# ---------------------------------------------------------------------------


def test_empty_captions_is_an_input_macro_not_a_filter_one() -> None:
    """It lowers to an ``-i``, not to a node: no ffmpeg filter generates a
    subtitle stream, because a filtergraph carries no subtitle pads at all."""
    assert "empty_captions" not in MACROS
    macro = INPUT_MACROS["empty_captions"]
    assert (macro.output, macro.format) == ("subtitle", "webvtt")
    # "WEBVTT\n\n" -- a valid WebVTT file with zero cues, and nothing else.
    assert base64.b64decode(macro.path.split(",", 1)[1]) == b"WEBVTT\n\n"


def test_empty_captions_mints_an_input_with_no_registry_at_all() -> None:
    g = _lower("SELECT ffrwd.empty_captions() FROM input('x.mp4') a")
    assert not g.nodes
    assert g.input_paths[1] == INPUT_MACROS["empty_captions"].path
    assert g.input_options["ffrwd.empty_captions#2"] == {"format": "webvtt"}
    assert [(o.ref, o.type) for o in g.outputs] == [
        ("src:ffrwd.empty_captions#2:s:0", "subtitle")
    ]


def test_empty_captions_takes_no_arguments() -> None:
    err = _reject("SELECT ffrwd.empty_captions(2) FROM input('x.mp4') a")
    assert err.code == ErrorCode.UNSUPPORTED_SQL
    assert "takes no arguments" in err.message


def test_the_macro_namespace_hint_names_the_input_macro_too() -> None:
    err = _reject("SELECT ffrwd.zzz(a.video[1], 2) FROM input('x.mp4') a")
    assert "empty_captions" in (err.hint or "")


def test_format_is_both_the_minted_flag_and_a_user_facing_option() -> None:
    """`format` renders the same per-input flag either way: minted here by
    `empty_captions` (bypassing validation, built by the compiler) and, since
    plan 075, also legal for a user to write on an ordinary input()."""
    assert "format" in INPUT_OPTIONS
    assert option_spec("format") is INPUT_OPTIONS["format"]
    g = _lower("SELECT a.video[1] FROM input('x.mp4', format => 'webvtt') a")
    assert g.input_options == {"a": {"format": "webvtt"}}
