"""Tests for the input option table.

Guardrail #4: INPUT_OPTIONS is DATA. This file checks the table's shape
against the documented option list (name/type per option) and exercises
``validate_option``'s happy/unknown/wrong-type paths -- the input-side mirror
of ``tests/test_sink.py``.
"""

from __future__ import annotations

import pytest

from ffrwd.errors import ErrorCode, FfrwdError
from ffrwd.inputs import (
    INPUT_OPTIONS,
    InputOptionSpec,
    option_spec,
    probe_options,
    render_options,
    validate_option,
)

# name -> type.
_EXPECTED: dict[str, str] = {
    "loop": "bool",
    "stream_loop": "int",
    "framerate": "num",
    "itsoffset": "num",
    "hwaccel": "str",
    "seek_end": "num",
    "format": "str",
    "realtime": "bool",
    "sub_charenc": "str",
    "start_number": "int",
    "subtitle_decoder": "str",
    "video_size": "str",
    "pixel_format": "str",
    "sample_rate": "int",
    "channels": "int",
    "rtbufsize": "str",
    "probesize": "str",
    "analyzeduration": "str",
    "rtsp_transport": "str",
    "user_agent": "str",
}


def test_the_table_is_the_documented_options_with_their_type() -> None:
    assert {name: spec.type for name, spec in INPUT_OPTIONS.items()} == _EXPECTED
    assert [name for name, spec in INPUT_OPTIONS.items() if spec.name != name] == []


# name -> whether it belongs on the ffprobe invocation too (InputOptionSpec.probes).
# Measured against a real ffmpeg build: ffprobe rejects `-re`, `-stream_loop`,
# `-hwaccel`, `-itsoffset` and `-sseof` outright, and accepts `-loop`,
# `-framerate`, `-f` and `-video_size`. The rest are judgment calls from what
# each option actually shapes -- see the `probes=` comment next to each entry
# in ffrwd/inputs.py.
_EXPECTED_PROBES: dict[str, bool] = {
    "loop": True,
    "stream_loop": False,
    "framerate": True,
    "itsoffset": False,
    "hwaccel": False,
    "seek_end": False,
    "format": True,
    "realtime": False,
    "sub_charenc": True,
    "start_number": True,
    "subtitle_decoder": False,
    "video_size": True,
    "pixel_format": True,
    "sample_rate": True,
    "channels": True,
    "rtbufsize": False,
    "probesize": True,
    "analyzeduration": True,
    "rtsp_transport": True,
    "user_agent": True,
}


def test_the_table_pins_which_options_reach_a_probe() -> None:
    assert {name: spec.probes for name, spec in INPUT_OPTIONS.items()} == _EXPECTED_PROBES


def test_probe_options_drops_the_ffmpeg_only_flags() -> None:
    """The reported bug's exact shape: `realtime` must not reach ffprobe,
    which rejects `-re` outright; `framerate` (demuxer-shaping) survives."""
    values = {"realtime": True, "framerate": 30, "format": "dshow"}
    assert probe_options(values) == {"framerate": 30, "format": "dshow"}


def test_probe_options_of_only_ffmpeg_only_flags_is_empty() -> None:
    assert probe_options({"realtime": True, "itsoffset": 2, "hwaccel": "cuda"}) == {}


def test_probe_options_preserves_written_order() -> None:
    values = {"video_size": "960x540", "realtime": True, "framerate": 30}
    assert list(probe_options(values)) == ["video_size", "framerate"]


def test_render_options_of_probe_options_matches_the_reported_repro() -> None:
    """`input('src10.mp4', realtime => true)`: filtered to nothing, so the
    probe reads the file with no options at all -- exactly like ffmpeg's own
    decode of it minus the one flag ffprobe cannot take."""
    assert render_options(probe_options({"realtime": True})) == []


def test_all_entries_are_spec_instances() -> None:
    for spec in INPUT_OPTIONS.values():
        assert isinstance(spec, InputOptionSpec)
        assert spec.doc  # non-empty, drives docs/prompt
        assert spec.flag.startswith("-")


def test_loop_renders_as_loop_flag() -> None:
    spec = INPUT_OPTIONS["loop"]
    assert spec.flag == "-loop"
    assert spec.type == "bool"


def test_itsoffset_and_framerate_are_num() -> None:
    assert INPUT_OPTIONS["itsoffset"].type == "num"
    assert INPUT_OPTIONS["framerate"].type == "num"


def test_seek_end_is_num_and_maps_to_sseof() -> None:
    spec = INPUT_OPTIONS["seek_end"]
    assert spec.type == "num"
    assert spec.flag == "-sseof"
    assert spec.bare is False


def test_realtime_is_the_only_bare_input_option() -> None:
    bare_names = {n for n, s in INPUT_OPTIONS.items() if s.bare}
    assert bare_names == {"realtime"}
    spec = INPUT_OPTIONS["realtime"]
    assert spec.type == "bool"
    assert spec.flag == "-re"


def test_format_is_a_public_input_option() -> None:
    spec = INPUT_OPTIONS["format"]
    assert spec.type == "str"
    assert spec.flag == "-f"
    assert option_spec("format") is spec


def test_subtitle_decoder_flag_has_no_index() -> None:
    assert INPUT_OPTIONS["subtitle_decoder"].flag == "-c:s"


def test_start_number_is_an_int() -> None:
    assert INPUT_OPTIONS["start_number"].type == "int"
    assert INPUT_OPTIONS["start_number"].flag == "-start_number"


def test_sub_charenc_is_a_str() -> None:
    assert INPUT_OPTIONS["sub_charenc"].type == "str"
    assert INPUT_OPTIONS["sub_charenc"].flag == "-sub_charenc"


# ---------------------------------------------------------------------------
# validate_option
# ---------------------------------------------------------------------------


def test_validate_option_returns_every_accepted_value_unchanged() -> None:
    # A bool option answers with the bool itself, not a truthy number.
    for name in ("loop", "realtime"):
        assert validate_option(name, True) is True, name
        assert validate_option(name, False) is False, name
    # itsoffset legitimately takes a negative offset.
    accepted: dict[str, list[object]] = {
        "stream_loop": [-1],
        "start_number": [5],
        "framerate": [15, 29.97],
        "itsoffset": [-1, -1.5],
        "seek_end": [60, 12.5],
        "hwaccel": ["cuda"],
        "format": ["v4l2"],
        "sub_charenc": ["CP1250"],
        "subtitle_decoder": ["webvtt"],
    }
    assert {
        name: [validate_option(name, value) for value in values]
        for name, values in accepted.items()
    } == accepted


def test_validate_option_unknown_raises() -> None:
    with pytest.raises(FfrwdError) as excinfo:
        validate_option("bogus_option", "x")
    err = excinfo.value
    assert err.code == ErrorCode.UNKNOWN_INPUT_OPTION
    assert "bogus_option" in err.message


def test_validate_option_unknown_did_you_mean_hint() -> None:
    with pytest.raises(FfrwdError) as excinfo:
        validate_option("loob", True)
    err = excinfo.value
    assert err.code == ErrorCode.UNKNOWN_INPUT_OPTION
    assert err.hint is not None
    assert "loop" in err.hint


def test_validate_option_unknown_no_close_match_lists_known() -> None:
    with pytest.raises(FfrwdError) as excinfo:
        validate_option("zzzzzzzzzz", "x")
    err = excinfo.value
    assert err.hint is not None
    assert "known options:" in err.hint
    for name in INPUT_OPTIONS:
        assert name in err.hint


def test_validate_option_bool_rejects_str() -> None:
    with pytest.raises(FfrwdError) as excinfo:
        validate_option("loop", "true")
    assert excinfo.value.code == ErrorCode.INPUT_OPTION_TYPE


def test_validate_option_bool_rejects_int() -> None:
    with pytest.raises(FfrwdError) as excinfo:
        validate_option("loop", 1)
    assert excinfo.value.code == ErrorCode.INPUT_OPTION_TYPE


def test_validate_option_int_rejects_float() -> None:
    with pytest.raises(FfrwdError) as excinfo:
        validate_option("stream_loop", 1.5)
    assert excinfo.value.code == ErrorCode.INPUT_OPTION_TYPE


def test_validate_option_int_rejects_bool() -> None:
    # bool is a subclass of int in Python; the table must not accept it
    # where an int is declared.
    with pytest.raises(FfrwdError) as excinfo:
        validate_option("stream_loop", True)
    assert excinfo.value.code == ErrorCode.INPUT_OPTION_TYPE


def test_validate_option_num_rejects_str() -> None:
    with pytest.raises(FfrwdError) as excinfo:
        validate_option("framerate", "fast")
    err = excinfo.value
    assert err.code == ErrorCode.INPUT_OPTION_TYPE
    assert "expects a number" in err.message


def test_validate_option_num_rejects_bool() -> None:
    with pytest.raises(FfrwdError) as excinfo:
        validate_option("itsoffset", True)
    assert excinfo.value.code == ErrorCode.INPUT_OPTION_TYPE


def test_validate_option_str_rejects_bool() -> None:
    with pytest.raises(FfrwdError) as excinfo:
        validate_option("hwaccel", True)
    assert excinfo.value.code == ErrorCode.INPUT_OPTION_TYPE


def test_validate_option_str_rejects_int() -> None:
    with pytest.raises(FfrwdError) as excinfo:
        validate_option("hwaccel", 5)
    assert excinfo.value.code == ErrorCode.INPUT_OPTION_TYPE


def test_validate_option_preserves_line_col() -> None:
    with pytest.raises(FfrwdError) as excinfo:
        validate_option("bogus", "x", line=3, col=12)
    err = excinfo.value
    assert err.line == 3
    assert err.col == 12
