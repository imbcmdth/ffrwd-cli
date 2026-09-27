"""Tests for the input option table.

Guardrail #4: INPUT_OPTIONS is DATA. This file checks the table's shape
against the documented option list (name/type per option) and exercises
``validate_option``'s happy/unknown/wrong-type paths -- the input-side mirror
of ``tests/test_sink.py``.
"""

from __future__ import annotations

import pytest

from ffrwd import compiler
from ffrwd.compiler import compile_all
from ffrwd.errors import ErrorCode, FfrwdError
from ffrwd.inputs import (
    INPUT_OPTIONS,
    InputOptionSpec,
    declared_probe,
    option_spec,
    probe_options,
    render_options,
    rendered_options,
    validate_option,
)
from ffrwd.probe import ProbeResult, StreamMeta
from ffrwd.wasm import Described

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
    "rtbufsize": "size",
    "probesize": "size",
    "analyzeduration": "size",
    "rtsp_transport": "str",
    "user_agent": "str",
    "listen": "bool",
    "shape": "struct",
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
    # A probe of a listener has to listen as the run does, or it dials.
    "listen": True,
    # The compiler's own: a declared shape means there is no probe at all.
    "shape": False,
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
        # A compile-time option is rendered nowhere, so it has no flag.
        assert spec.flag.startswith("-") if not spec.compile_time else spec.flag == ""


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
        "probesize": [5000000, "32M"],
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


@pytest.mark.parametrize(
    ("name", "value", "hint"),
    [
        (
            "probesize",
            -1,
            "probesize takes a whole number, or a string with a suffix: "
            "e.g. probesize => 5000000 or probesize => '32M'",
        ),
        (
            "rtsp_transport",
            1,
            "rtsp_transport takes a single-quoted string literal, "
            "e.g. rtsp_transport => 'tcp'",
        ),
    ],
)
def test_a_wrong_value_is_hinted_with_the_options_own_example(
    name: str, value: object, hint: str
) -> None:
    with pytest.raises(FfrwdError) as excinfo:
        validate_option(name, value)
    assert excinfo.value.code == ErrorCode.INPUT_OPTION_TYPE
    assert excinfo.value.hint == hint


def test_validate_option_preserves_line_col() -> None:
    with pytest.raises(FfrwdError) as excinfo:
        validate_option("bogus", "x", line=3, col=12)
    err = excinfo.value
    assert err.line == 3
    assert err.col == 12


# --- shape => STRUCT(...): an input's streams, declared instead of probed ----

_SHAPE = {"width": 1920, "height": 1080, "fps": 30, "rate": 48000, "channels": 2}


def test_shape_is_the_compilers_own_and_never_rendered() -> None:
    spec = INPUT_OPTIONS["shape"]
    assert (spec.type, spec.compile_time, spec.probes, spec.flag) == ("struct", True, False, "")
    options = {"realtime": True, "shape": validate_option("shape", dict(_SHAPE))}
    assert render_options(options) == ["-re"]
    assert render_options(probe_options(options)) == []
    assert rendered_options(options) == {"realtime": True}


def test_a_shape_normalizes_its_rate_the_way_ffprobe_writes_one() -> None:
    for fps, text in [(30, "30/1"), (29.97, "2997/100"), ("30000/1001", "30000/1001")]:
        checked = validate_option("shape", {**_SHAPE, "fps": fps})
        assert isinstance(checked, dict)
        assert checked["fps"] == text


def test_which_streams_a_shape_declares_follows_from_its_keys() -> None:
    picture = validate_option("shape", {"width": 640, "height": 360, "fps": "25/1"})
    sound = validate_option("shape", {"rate": 44100, "channels": 1, "audio_codec": "aac"})
    assert isinstance(picture, dict) and isinstance(sound, dict)

    assert [s.type for s in declared_probe(picture).streams] == ["video"]
    assert [s.type for s in declared_probe(sound).streams] == ["audio"]
    both = declared_probe(
        {**picture, **sound, "video_codec": "h264", "pix_fmt": "yuv420p",
         "channel_layout": "mono"}
    )  # fmt: skip
    video, audio = both.streams
    assert (video.width, video.height, video.fps, video.codec, video.pix_fmt) == (
        640, 360, "25/1", "h264", "yuv420p",
    )  # fmt: skip
    assert (audio.sample_rate, audio.channels, audio.channel_layout, audio.codec) == (
        44100, 1, "mono", "aac",
    )  # fmt: skip
    assert (both.duration, video.declared, audio.declared) == (None, True, True)
    # No codec said is None, as for a stream nothing probed.
    assert declared_probe(picture).streams[0].codec is None


@pytest.mark.parametrize(
    ("value", "needle"),
    [
        ("1920x1080", "option 'shape' expects a STRUCT of the input's streams"),
        ({**_SHAPE, "widht": 1280}, "shape has no key 'widht'"),
        ({**_SHAPE, "width": "1920"}, "shape key 'width' expects a whole number"),
        ({**_SHAPE, "height": 0}, "shape key 'height' expects a whole number"),
        ({**_SHAPE, "channels": True}, "shape key 'channels' expects a whole number"),
        ({**_SHAPE, "fps": "fast"}, "shape key 'fps' expects a frame rate"),
        ({**_SHAPE, "fps": -30}, "shape key 'fps' expects a frame rate"),
        ({**_SHAPE, "video_codec": 264}, "shape key 'video_codec' expects text"),
        ({"width": 1920, "rate": 48000, "channels": 2}, "a picture (width) without its "
         "height and fps"),
        ({"pix_fmt": "yuv420p"}, "a picture (pix_fmt) without its width and height and fps"),
        ({"width": 1, "height": 1, "fps": 1, "audio_codec": "aac"},
         "a sound (audio_codec) without its rate and channels"),
        ({}, "shape declares no stream"),
    ],
)
def test_a_shape_is_refused_by_name(value: object, needle: str) -> None:
    with pytest.raises(FfrwdError) as caught:
        validate_option("shape", value, line=3, col=9)

    assert caught.value.code is ErrorCode.INPUT_OPTION_TYPE
    assert needle in caught.value.message, caught.value.message
    assert (caught.value.line, caught.value.col) == (3, 9)
    assert caught.value.hint


# A module standing in for the demo's: it reads the picture and hands one back.
_INVERT = "invert.wasm"
_DESCRIBED = {
    _INVERT: Described(
        world="ffrwd:av@0.9.0",
        name="invert",
        version="0.1.0",
        params_schema={"type": "object", "properties": {}},
        pixel_formats=("rgba",),
    )
}

_LIVE_URLS = [
    "srt://0.0.0.0:9000?mode=listener&latency=200000",
    "rtmp://0.0.0.0:1935/live/feed",
]

# How each of those waits for its sender: SRT says so in its URL, while
# ffmpeg's RTMP reads a listen query as part of the stream name and dials.
_LISTENS = {"srt": "", "rtmp": ", listen => true"}


def _live_query(url: str, shape: str | None) -> str:
    """The SMART demo's head in small: the feed conformed ahead of the split
    (its rate included), a module on one leg, the picture beside it, and the
    sound filtered into the same file."""
    declared = _LISTENS[url.partition(":")[0]]
    declared += f", shape => {shape}" if shape else ""
    return (
        "CREATE FUNCTION invert(v video_stream) RETURNS video_stream\n"
        f"  AS '{_INVERT}', 'invert' LANGUAGE wasm;\n"
        "COPY (\n"
        "  WITH feed AS (\n"
        "    SELECT scale(ffmpeg.fps(s.video[1], 30), 1280, 720) AS v,\n"
        "           aresample(s.audio[1], 48000) AS a\n"
        f"    FROM input('{url}'{declared}) s\n"
        "  )\n"
        "  SELECT ffmpeg.hstack(feed.v, invert(feed.v)), feed.a FROM feed\n"
        ") TO 'out.mkv' WITH (video_codec 'ffv1')"
    )


_STRUCT = (
    "STRUCT(1920 AS width, 1080 AS height, 30 AS fps, 'h264' AS video_codec, "
    "48000 AS rate, 2 AS channels, 'aac' AS audio_codec)"
)


def _probed_like_the_struct() -> ProbeResult:
    return ProbeResult(
        streams=[
            StreamMeta(
                type="video", index=0, metadata={}, width=1920, height=1080,
                fps="30/1", sample_rate=None, codec="h264",
            ),
            StreamMeta(
                type="audio", index=0, metadata={}, width=None, height=None,
                fps=None, sample_rate=48000, channels=2, codec="aac",
            ),
        ],
    )  # fmt: skip


@pytest.mark.parametrize("url", _LIVE_URLS)
def test_a_declared_shape_compiles_a_listener_without_probing_it(
    url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A probe of a listener takes the sender's first connection, and one that
    misses its ceiling leaves the query shapeless: with the shape declared,
    nothing is probed, and the plan is the one a probe of the same values
    makes."""

    def no_probe(*args: object, **kwargs: object) -> None:
        raise AssertionError(f"probed {args}: a declared shape must not be")

    monkeypatch.setattr(compiler, "probe_path", no_probe)
    declared = compile_all(
        _live_query(url, _STRUCT), describe=lambda path: _DESCRIBED[path]
    )
    probed_result = _probed_like_the_struct()
    monkeypatch.setattr(compiler, "probe_path", lambda path, args=(), **kw: probed_result)
    probed = compile_all(_live_query(url, None), describe=lambda path: _DESCRIBED[path])

    assert declared.plan is not None and probed.plan is not None
    assert declared.plan.to_dict() == probed.plan.to_dict()
    # The shape reaches neither ffmpeg nor the plan's input options.
    reader = next(p for p in declared.plan.ffmpeg if url in p.graph.input_paths)
    assert all("shape" not in o for o in reader.graph.input_options.values())
    # 1080p conformed to 720p ahead of the split: the pipe, not the fifo.
    bounded = [e for e in declared.plan.stream_edges if e.buffer is not None]
    assert bounded and all(e.buffer.road == "pipe" for e in bounded if e.buffer)


def test_a_shape_on_a_file_skips_its_probe_too(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_probe(*args: object, **kwargs: object) -> None:
        raise AssertionError("a declared shape must not be probed")

    monkeypatch.setattr(compiler, "probe_path", no_probe)
    graph = compile_all(
        "COPY (SELECT s.video[1] FROM input('clip.mp4', "
        "shape => STRUCT(640 AS width, 360 AS height, 25 AS fps)) s) TO 'o.mkv'"
    ).graphs[0]

    assert graph.input_paths == ["clip.mp4"]
    assert graph.input_options == {}


def test_a_shape_that_is_no_struct_is_refused_at_its_option(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(compiler, "probe_path", lambda path, args=(), **kw: None)
    with pytest.raises(FfrwdError) as caught:
        compile_all(
            "COPY (SELECT s.video[1] FROM input('srt://0.0.0.0:9000?mode=listener', "
            "shape => STRUCT(1920 AS width, 1080 AS height)) s) TO 'o.mkv'"
        )

    assert caught.value.code is ErrorCode.INPUT_OPTION_TYPE
    assert "a picture (width, height) without its fps" in caught.value.message


def test_an_rtmp_listener_renders_listen_before_its_input() -> None:
    """ffmpeg's RTMP waits for a publisher only with ``-listen 1``; a
    ``?listen=1`` in the URL is read as part of the stream name and dials."""
    graph = compile_all(
        "COPY (SELECT s.video[1] FROM input('rtmp://0.0.0.0:1935/live/feed', "
        "listen => true, shape => STRUCT(1280 AS width, 720 AS height, 30 AS fps)) s) "
        "TO 'o.mkv'"
    ).graphs[0]

    assert graph.input_options == {"s": {"listen": True}}
    assert render_options(graph.input_options["s"]) == ["-listen", "1"]
    assert render_options(probe_options({"listen": True})) == ["-listen", "1"]
