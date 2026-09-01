"""Tests for the sink option table.

Guardrail #4: SINK_OPTIONS is DATA. This file checks the table's shape
against the documented option list (name/scope/type per option) and exercises
``validate_option``'s happy/unknown/wrong-type paths.
"""

from __future__ import annotations

import pytest

from ffrwd.errors import ErrorCode, FfrwdError
from ffrwd.sink import (
    CODEC_PARAMS_FLAGS,
    CSV_OPTIONS,
    REPLACED_OPTIONS,
    SINK_OPTIONS,
    SinkOptionSpec,
    validate_csv_option,
    validate_option,
)

# name -> (scope, type), subtitle_codec included.
_EXPECTED: dict[str, tuple[str, str]] = {
    "video_codec": ("video", "str"),
    "audio_codec": ("audio", "str"),
    "crf": ("video", "int"),
    "preset": ("video", "str"),
    "pix_fmt": ("video", "str"),
    "video_bitrate": ("video", "str"),
    "audio_bitrate": ("audio", "str"),
    "sample_rate": ("audio", "int"),
    "format": ("container", "str"),
    "faststart": ("container", "bool"),
    "subtitle_codec": ("subtitle", "str"),
    "frames": ("video", "int"),
    "duration": ("container", "num"),
    "max_size": ("container", "str"),
    "shortest": ("container", "bool"),
    "maxrate": ("video", "str"),
    "bufsize": ("video", "str"),
    "gop": ("video", "int"),
    "profile": ("video", "str"),
    "level": ("video", "str"),
    "tune": ("video", "str"),
    "codec_params": ("video", "str"),
    "two_pass": ("container", "bool"),
    "movflags": ("container", "str"),
    "hls_time": ("container", "num"),
    "hls_playlist_type": ("container", "str"),
    "hls_flags": ("container", "str"),
    "hls_segment_type": ("container", "str"),
    "hls_segment_filename": ("container", "str"),
    "hls_fmp4_init_filename": ("container", "str"),
    "master_pl_name": ("container", "str"),
    "seg_duration": ("container", "num"),
    "use_template": ("container", "bool"),
    "use_timeline": ("container", "bool"),
    "init_seg_name": ("container", "str"),
    "media_seg_name": ("container", "str"),
    "single_file": ("container", "bool"),
}


def test_the_table_is_the_documented_options_with_their_scope_and_type() -> None:
    assert len(SINK_OPTIONS) == 37
    assert {name: (spec.scope, spec.type) for name, spec in SINK_OPTIONS.items()} == _EXPECTED
    assert [name for name, spec in SINK_OPTIONS.items() if spec.name != name] == []


def test_all_entries_are_spec_instances() -> None:
    for spec in SINK_OPTIONS.values():
        assert isinstance(spec, SinkOptionSpec)
        assert spec.doc  # non-empty, drives docs/prompt
        assert spec.flag.startswith("-")


def test_faststart_renders_as_movflags_bool() -> None:
    spec = SINK_OPTIONS["faststart"]
    assert spec.flag == "-movflags"
    assert spec.value_template == "+faststart"
    assert spec.per_stream is False


def test_format_is_not_per_stream() -> None:
    spec = SINK_OPTIONS["format"]
    assert spec.flag == "-f"
    assert spec.per_stream is False


def test_per_stream_options_use_colon_index_flags() -> None:
    per_stream_names = {n for n, s in SINK_OPTIONS.items() if s.per_stream}
    assert per_stream_names == {
        "video_codec",
        "audio_codec",
        "crf",
        "preset",
        "pix_fmt",
        "video_bitrate",
        "audio_bitrate",
        "sample_rate",
        "subtitle_codec",
        "frames",
        "maxrate",
        "bufsize",
        "gop",
        "profile",
        "level",
        "tune",
        "codec_params",
    }


def test_bare_options_are_container_scope_and_never_per_stream() -> None:
    bare_names = {n for n, s in SINK_OPTIONS.items() if s.bare}
    assert bare_names == {"shortest"}
    for name in bare_names:
        spec = SINK_OPTIONS[name]
        assert spec.type == "bool"
        assert spec.per_stream is False


def test_shortest_renders_as_a_bare_flag() -> None:
    spec = SINK_OPTIONS["shortest"]
    assert spec.flag == "-shortest"
    assert spec.bare is True


def test_duration_is_a_num_type_container_option() -> None:
    spec = SINK_OPTIONS["duration"]
    assert spec.type == "num"
    assert spec.scope == "container"
    assert spec.flag == "-t"
    assert spec.per_stream is False


def test_movflags_and_faststart_share_the_same_flag() -> None:
    assert SINK_OPTIONS["movflags"].flag == "-movflags"
    assert SINK_OPTIONS["faststart"].flag == "-movflags"


def test_codec_params_flag_is_a_codec_placeholder_template() -> None:
    spec = SINK_OPTIONS["codec_params"]
    assert spec.flag == "-{codec}-params"
    assert spec.scope == "video"
    assert spec.per_stream is True


def test_codec_params_flags_cover_x264_x265_svtav1() -> None:
    assert CODEC_PARAMS_FLAGS == {
        "libx264": "x264",
        "libx265": "x265",
        "libsvtav1": "svtav1",
    }


def test_the_metadata_options_were_replaced_by_the_tags_column() -> None:
    """Both are SELECT columns now; the rejection names what replaced each."""
    assert "metadata_from" not in SINK_OPTIONS
    assert "strip_metadata" not in SINK_OPTIONS
    assert "f.tags AS tags" in REPLACED_OPTIONS["metadata_from"]
    assert "STRUCT() AS tags" in REPLACED_OPTIONS["strip_metadata"]


def test_subtitle_codec_scope_and_flag() -> None:
    spec = SINK_OPTIONS["subtitle_codec"]
    assert spec.scope == "subtitle"
    assert spec.flag == "-c"
    assert spec.per_stream is True
    assert "mov_text" in spec.doc or "webvtt" in spec.doc or "srt" in spec.doc


# ---------------------------------------------------------------------------
# validate_option
# ---------------------------------------------------------------------------


def test_validate_option_returns_every_accepted_string_value_unchanged() -> None:
    accepted = {
        "video_codec": "libx264",
        "subtitle_codec": "mov_text",
        "maxrate": "2675k",
        "bufsize": "5350k",
        "profile": "baseline",
        "level": "3.1",
        "tune": "film",
        "codec_params": "keyint=48:min-keyint=48",
        "movflags": "+faststart+frag_keyframe",
        "max_size": "10M",
    }
    assert {name: validate_option(name, value) for name, value in accepted.items()} == accepted


def test_validate_option_happy_int() -> None:
    assert validate_option("crf", 20) == 20


def test_validate_option_happy_frames() -> None:
    assert validate_option("frames", 1) == 1


def test_frames_scope_and_flag() -> None:
    spec = SINK_OPTIONS["frames"]
    assert spec.scope == "video"
    assert spec.flag == "-frames"
    assert spec.per_stream is True


def test_validate_option_happy_bool_true() -> None:
    assert validate_option("faststart", True) is True


def test_validate_option_happy_bool_false() -> None:
    assert validate_option("faststart", False) is False


def test_validate_option_happy_num_int() -> None:
    assert validate_option("duration", 30) == 30


def test_validate_option_happy_num_float() -> None:
    assert validate_option("duration", 30.5) == 30.5


def test_validate_option_num_rejects_str() -> None:
    with pytest.raises(FfrwdError) as excinfo:
        validate_option("duration", "30")
    err = excinfo.value
    assert err.code == ErrorCode.SINK_OPTION_TYPE
    assert "expects a number" in err.message


def test_validate_option_num_rejects_bool() -> None:
    with pytest.raises(FfrwdError) as excinfo:
        validate_option("duration", True)
    assert excinfo.value.code == ErrorCode.SINK_OPTION_TYPE


def test_validate_option_happy_shortest() -> None:
    assert validate_option("shortest", True) is True
    assert validate_option("shortest", False) is False


def test_validate_option_happy_gop() -> None:
    assert validate_option("gop", 48) == 48


def test_validate_option_rejects_a_replaced_option_by_name() -> None:
    with pytest.raises(FfrwdError) as excinfo:
        validate_option("strip_metadata", True)
    assert excinfo.value.code is ErrorCode.UNKNOWN_SINK_OPTION
    assert "STRUCT() AS tags" in (excinfo.value.hint or "")


def test_validate_option_unknown_raises() -> None:
    with pytest.raises(FfrwdError) as excinfo:
        validate_option("bogus_option", "x")
    err = excinfo.value
    assert err.code == ErrorCode.UNKNOWN_SINK_OPTION
    assert "bogus_option" in err.message


def test_validate_option_unknown_did_you_mean_hint() -> None:
    with pytest.raises(FfrwdError) as excinfo:
        validate_option("video_codc", "libx264")
    err = excinfo.value
    assert err.code == ErrorCode.UNKNOWN_SINK_OPTION
    assert err.hint is not None
    assert "video_codec" in err.hint


def test_validate_option_unknown_no_close_match_lists_known() -> None:
    with pytest.raises(FfrwdError) as excinfo:
        validate_option("zzzzzzzzzz", "x")
    err = excinfo.value
    assert err.hint is not None
    assert "known options:" in err.hint
    for name in SINK_OPTIONS:
        assert name in err.hint


def test_validate_option_wrong_type_str_expected() -> None:
    with pytest.raises(FfrwdError) as excinfo:
        validate_option("video_codec", 5)
    err = excinfo.value
    assert err.code == ErrorCode.SINK_OPTION_TYPE
    assert "video_codec" in err.message


def test_validate_option_wrong_type_int_expected() -> None:
    with pytest.raises(FfrwdError) as excinfo:
        validate_option("crf", "20")
    assert excinfo.value.code == ErrorCode.SINK_OPTION_TYPE


def test_validate_option_int_rejects_float() -> None:
    with pytest.raises(FfrwdError) as excinfo:
        validate_option("crf", 20.5)
    assert excinfo.value.code == ErrorCode.SINK_OPTION_TYPE


def test_validate_option_int_rejects_bool() -> None:
    # bool is a subclass of int in Python; the table must not accept it
    # where an int is declared.
    with pytest.raises(FfrwdError) as excinfo:
        validate_option("crf", True)
    assert excinfo.value.code == ErrorCode.SINK_OPTION_TYPE


def test_validate_option_bool_rejects_str() -> None:
    with pytest.raises(FfrwdError) as excinfo:
        validate_option("faststart", "true")
    assert excinfo.value.code == ErrorCode.SINK_OPTION_TYPE


def test_validate_option_bool_rejects_int() -> None:
    with pytest.raises(FfrwdError) as excinfo:
        validate_option("faststart", 1)
    assert excinfo.value.code == ErrorCode.SINK_OPTION_TYPE


def test_validate_option_str_rejects_bool() -> None:
    with pytest.raises(FfrwdError) as excinfo:
        validate_option("format", True)
    assert excinfo.value.code == ErrorCode.SINK_OPTION_TYPE


def test_validate_option_preserves_line_col() -> None:
    with pytest.raises(FfrwdError) as excinfo:
        validate_option("bogus", "x", line=3, col=12)
    err = excinfo.value
    assert err.line == 3
    assert err.col == 12


# ---------------------------------------------------------------------------
# CSV_OPTIONS / validate_csv_option
# ---------------------------------------------------------------------------


def test_csv_options_has_exactly_format_and_header() -> None:
    assert set(CSV_OPTIONS) == {"format", "header"}
    assert CSV_OPTIONS["format"].type == "str"
    assert CSV_OPTIONS["header"].type == "bool"


def test_validate_csv_option_happy_format() -> None:
    assert validate_csv_option("format", "csv") == "csv"


def test_validate_csv_option_happy_header() -> None:
    assert validate_csv_option("header", True) is True
    assert validate_csv_option("header", False) is False


def test_validate_csv_option_wrong_type() -> None:
    with pytest.raises(FfrwdError) as excinfo:
        validate_csv_option("header", "true")
    assert excinfo.value.code == ErrorCode.SINK_OPTION_TYPE


def test_validate_csv_option_rejects_a_media_option() -> None:
    """A media-only option: unknown against CSV_OPTIONS, a
    SEPARATE table from SINK_OPTIONS -- not silently accepted."""
    with pytest.raises(FfrwdError) as excinfo:
        validate_csv_option("video_codec", "libx264")
    err = excinfo.value
    assert err.code == ErrorCode.UNKNOWN_SINK_OPTION
    assert "video_codec" in err.message
    assert err.hint is not None
    assert "video_codec" not in err.hint  # the hint lists CSV_OPTIONS names, not SINK_OPTIONS'
    assert "format" in err.hint or "header" in err.hint


def test_validate_option_rejects_a_csv_only_option() -> None:
    """The reverse direction already falls out of SINK_OPTIONS never having
    had `header` in it -- no code change needed, just pinning the behavior."""
    with pytest.raises(FfrwdError) as excinfo:
        validate_option("header", True)
    assert excinfo.value.code == ErrorCode.UNKNOWN_SINK_OPTION
