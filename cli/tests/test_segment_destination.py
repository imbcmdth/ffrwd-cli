"""Tests for a destination cut into files, ``WITH (format 'segment')``, and
for the HLS options that keep a live playlist's window and numbering.

A cut destination names every file at once, `%d` numbering them; each file
starts its clock at zero, and the options of each file's own muxer reach it
through the segment muxer. HERMETIC: ``probe_path`` is stubbed with one
640x480 30fps H.264 video and one AAC track, so no fixture is read and no
ffmpeg runs.
"""

from __future__ import annotations

import pytest

from ffrwd import compiler
from ffrwd.compiler import compile_commands
from ffrwd.emit import build_ffmpeg_args, emit
from ffrwd.errors import ErrorCode, FfrwdError
from ffrwd.probe import ProbeResult, StreamMeta

SRC = "in.mkv"


@pytest.fixture(autouse=True)
def _synthetic_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    video = StreamMeta(
        type="video",
        index=0,
        metadata={},
        width=640,
        height=480,
        fps="30/1",
        sample_rate=None,
        codec="h264",
    )
    audio = StreamMeta(
        type="audio",
        index=1,
        metadata={},
        width=None,
        height=None,
        fps=None,
        sample_rate=48000,
        codec="aac",
    )
    result = ProbeResult(streams=[video, audio], duration=10.0)
    monkeypatch.setattr(compiler, "probe_path", lambda path, args=(), **kw: result)


def _argv(sql: str) -> list[str]:
    return build_ffmpeg_args(emit(compile_commands(sql)[0]))


def _rejects(sql: str) -> FfrwdError:
    with pytest.raises(FfrwdError) as excinfo:
        compile_commands(sql)
    return excinfo.value


def _after(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


def _copy(destination: str, with_block: str) -> str:
    return f"COPY (SELECT s.* FROM input('{SRC}') s) TO '{destination}' WITH ({with_block})"


# ---------------------------------------------------------------------------
# a cut destination
# ---------------------------------------------------------------------------


def test_a_cut_copy_writes_numbered_files_from_zero() -> None:
    argv = _argv(_copy("rec/part-%06d.mp4", "format 'segment', segment_time 300"))
    assert _after(argv, "-f") == "segment"
    assert _after(argv, "-segment_time") == "300"
    assert _after(argv, "-reset_timestamps") == "1"
    assert argv[-1] == "rec/part-%06d.mp4"
    assert "-c:0" in argv and _after(argv, "-c:0") == "copy"


def test_movflags_reach_each_file_through_the_segment_muxer() -> None:
    argv = _argv(
        _copy(
            "rec/part-%06d.mp4",
            "format 'segment', segment_time 300, "
            "movflags '+frag_keyframe+empty_moov+default_base_moof'",
        )
    )
    assert "-movflags" not in argv
    assert _after(argv, "-segment_format_options") == (
        "movflags=+frag_keyframe+empty_moov+default_base_moof"
    )


def test_faststart_reaches_each_file_the_same_way() -> None:
    argv = _argv(_copy("rec/part-%d.mp4", "format 'segment', faststart true"))
    assert "-movflags" not in argv
    assert _after(argv, "-segment_format_options") == "movflags=+faststart"


def test_a_segment_list_is_written_where_the_query_names() -> None:
    argv = _argv(_copy("rec/part-%06d.mp4", "format 'segment', segment_list 'rec/list.csv'"))
    assert _after(argv, "-segment_list") == "rec/list.csv"


def test_a_cut_destination_with_one_name_is_refused() -> None:
    err = _rejects(_copy("rec/part.mp4", "format 'segment', segment_time 300"))
    assert err.code is ErrorCode.SINK_OPTION_TYPE
    assert "names one" in err.message
    assert err.hint is not None and "%d" in err.hint


@pytest.mark.parametrize("option", ["segment_time 300", "segment_list 'list.csv'"])
def test_a_segment_option_is_refused_without_the_format(option: str) -> None:
    err = _rejects(_copy("rec/part-%06d.mp4", option))
    assert err.code is ErrorCode.SINK_OPTION_TYPE
    assert "belongs to format 'segment'" in err.message


def test_strftime_names_each_file_by_when_it_opened() -> None:
    argv = _argv(
        _copy(
            "rec/%Y%m%dT%H%M%SZ.mp4",
            "format 'segment', segment_time 300, strftime true",
        )
    )
    assert _after(argv, "-strftime") == "1"
    assert argv[-1] == "rec/%Y%m%dT%H%M%SZ.mp4"


def test_a_timed_name_needs_strftime() -> None:
    err = _rejects(_copy("rec/%Y%m%dT%H%M%SZ.mp4", "format 'segment'"))
    assert err.code is ErrorCode.SINK_OPTION_TYPE
    assert err.hint is not None and "strftime true" in err.hint


def test_strftime_names_the_pictures_of_an_image_sequence() -> None:
    sql = (
        f"COPY (SELECT scale(fps(s.video[1], fps => '1/3'), width => 480, height => -2) "
        f"FROM input('{SRC}') s) TO 'thumbs/%Y%m%dT%H%M%SZ.jpg' "
        "WITH (format 'image2', video_codec 'mjpeg', strftime true)"
    )
    argv = _argv(sql)
    assert _after(argv, "-strftime") == "1"
    assert _after(argv, "-f") == "image2"


def test_strftime_is_refused_where_no_files_are_named_by_it() -> None:
    err = _rejects(_copy("out.mp4", "strftime true"))
    assert err.code is ErrorCode.SINK_OPTION_TYPE
    assert "strftime" in err.message


def test_a_file_that_is_not_cut_keeps_its_movflags() -> None:
    argv = _argv(_copy("out.mp4", "movflags '+frag_keyframe+empty_moov'"))
    assert _after(argv, "-movflags") == "+frag_keyframe+empty_moov"
    assert "-reset_timestamps" not in argv


# ---------------------------------------------------------------------------
# a live HLS playlist's window and numbering
# ---------------------------------------------------------------------------


def test_the_hls_window_and_numbering_reach_the_muxer() -> None:
    argv = _argv(
        _copy(
            "hls/master.m3u8",
            "format 'hls', hls_time 4, hls_segment_type 'fmp4', "
            "hls_list_size 150, hls_start_number_source 'epoch', "
            "master_pl_publish_rate 1",
        )
    )
    assert _after(argv, "-hls_list_size") == "150"
    assert _after(argv, "-hls_start_number_source") == "epoch"
    assert _after(argv, "-master_pl_publish_rate") == "1"


@pytest.mark.parametrize(
    "option",
    ["hls_list_size 150", "hls_start_number_source 'epoch'", "master_pl_publish_rate 1"],
)
def test_an_hls_window_option_is_refused_outside_hls(option: str) -> None:
    err = _rejects(_copy("out.mp4", option))
    assert err.code is ErrorCode.SINK_OPTION_TYPE
    assert "belongs to format 'hls'" in err.message
