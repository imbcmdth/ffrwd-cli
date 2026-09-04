"""Tests for the manifest destination: ``WITH (format 'hls')`` / ``'dash'``.

The rule under test: a manifest destination takes a MULTI-ROW relation --
each row one variant map entry, in row order, a NULL stream cell meaning
"this kind absent from this variant" -- while everywhere else the one-row
rule and the NULL refusal stand unchanged. The variant map, the keyframe
discipline and (for hls) the whole layout are the compiler's to write.

HERMETIC: ``probe_path`` is stubbed with a synthetic ``ProbeResult`` (one
640x480 30fps video, two language-tagged audio tracks), so heights, rates
and languages are fixed here rather than being properties of a file. No
fixture is read and no ffmpeg runs.
"""

from __future__ import annotations

import pytest

from ffrwd import compiler
from ffrwd.compiler import compile_commands
from ffrwd.emit import build_ffmpeg_args, emit
from ffrwd.errors import ErrorCode, FfrwdError
from ffrwd.ir import SinkUnit
from ffrwd.probe import ProbeResult, StreamMeta

SRC = "in.mkv"


def _video(width: int = 640, height: int = 480, fps: str | None = "30/1") -> StreamMeta:
    return StreamMeta(
        type="video",
        index=0,
        metadata={},
        width=width,
        height=height,
        fps=fps,
        sample_rate=None,
        codec="h264",
    )


def _audio(index: int, language: str, default: bool = False) -> StreamMeta:
    return StreamMeta(
        type="audio",
        index=index,
        metadata={"language": language},
        width=None,
        height=None,
        fps=None,
        sample_rate=48000,
        codec="aac",
        disposition={"default": default},
    )


def _stub(monkeypatch: pytest.MonkeyPatch, *streams: StreamMeta) -> None:
    result = ProbeResult(streams=list(streams), duration=10.0)
    monkeypatch.setattr(compiler, "probe_path", lambda path, args=(), **kw: result)


@pytest.fixture(autouse=True)
def _synthetic_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub(monkeypatch, _video(), _audio(0, "eng"), _audio(1, "fra"))


def _unit(sql: str) -> SinkUnit:
    graphs = compile_commands(sql)
    assert len(graphs) == 1 and len(graphs[0].sinks) == 1
    return graphs[0].sinks[0]


def _argv(sql: str) -> list[str]:
    return build_ffmpeg_args(emit(compile_commands(sql)[0]))


def _rejects(sql: str) -> FfrwdError:
    with pytest.raises(FfrwdError) as excinfo:
        compile_commands(sql)
    return excinfo.value


# The demuxed shape: two video rungs, two audio renditions, disjoint rows
# out of a FULL JOIN between two CTEs -- the JOIN lift and the row rule in
# one query.
def _demuxed(with_block: str) -> str:
    return (
        "COPY (WITH vid AS ("
        "  SELECT scale(f.video[1], ARRAY[640, 320][i.i], -2) AS v, i.i AS rung"
        f"  FROM input('{SRC}') f, generate_series(1, 2) i"
        "), aud AS ("
        "  SELECT a AS t, a.index AS pos, 2 + a.index AS rung"
        f"  FROM input('{SRC}') g, unnest(g.audio) a"
        ") SELECT vid.v, aud.t FROM vid FULL JOIN aud ON vid.rung = aud.rung) "
        f"TO 'out/master.m3u8' WITH ({with_block})"
    )


# The muxed shape: the same rungs, each row carrying BOTH cells.
def _muxed(with_block: str) -> str:
    return (
        "COPY (WITH vid AS ("
        "  SELECT scale(f.video[1], ARRAY[640, 320][i.i], -2) AS v, i.i AS rung"
        f"  FROM input('{SRC}') f, generate_series(1, 2) i"
        "), aud AS ("
        "  SELECT a AS t, a.index AS rung"
        f"  FROM input('{SRC}') g, unnest(g.audio) a"
        ") SELECT vid.v, aud.t FROM vid JOIN aud ON vid.rung = aud.rung) "
        f"TO 'out/master.m3u8' WITH ({with_block})"
    )


_HLS = "format 'hls', video_codec 'libx264', audio_codec 'aac'"


# ---------------------------------------------------------------------------
# the row rule: one row per plain path, rows per manifest
# ---------------------------------------------------------------------------


def test_a_plain_path_still_takes_one_row() -> None:
    err = _rejects(
        f"COPY (SELECT t FROM input('{SRC}') f, unnest(f.audio) t) TO 'out.mka'"
    )
    assert err.code is ErrorCode.ROW_COUNT_MISMATCH


def test_a_manifest_destination_accepts_the_rows() -> None:
    unit = _unit(
        f"COPY (SELECT t FROM input('{SRC}') f, unnest(f.audio) t) "
        "TO 'out/master.m3u8' WITH (format 'hls', audio_codec 'aac')"
    )
    assert len(unit.outputs) == 2
    assert unit.options["var_stream_map"] == (
        "a:0,agroup:aud,name:eng,language:eng,default:yes "
        "a:1,agroup:aud,name:fra,language:fra"
    )


def test_a_null_cell_is_refused_at_a_plain_path() -> None:
    # The same FULL JOIN, TO an mp4: the NULL refusal stands unchanged.
    err = _rejects(_demuxed("video_codec 'libx264'").replace(
        "TO 'out/master.m3u8'", "TO 'out.mp4'"
    ))
    assert err.code is ErrorCode.STREAM_NOT_FOUND
    assert "NULL in row" in err.message


# ---------------------------------------------------------------------------
# the transcription
# ---------------------------------------------------------------------------


def test_the_demuxed_shape_transcribes_rows_groups_and_names() -> None:
    unit = _unit(_demuxed(_HLS))
    assert unit.options["var_stream_map"] == (
        "v:0,agroup:aud,name:480p v:1,agroup:aud,name:240p "
        "a:0,agroup:aud,name:eng,language:eng,default:yes "
        "a:1,agroup:aud,name:fra,language:fra"
    )
    types = [output.type for output in unit.outputs]
    assert types == ["video", "video", "audio", "audio"]


def test_the_muxed_shape_transcribes_both_cells() -> None:
    unit = _unit(_muxed(_HLS))
    assert unit.options["var_stream_map"] == "v:0,a:0,name:480p v:1,a:1,name:240p"


def test_dash_wears_the_same_analysis() -> None:
    sql = _demuxed("format 'dash', video_codec 'libx264', audio_codec 'aac'").replace(
        "out/master.m3u8", "out/master.mpd"
    )
    unit = _unit(sql)
    assert unit.path == "out/master.mpd"
    assert unit.options["adaptation_sets"] == "id=0,streams=0,1 id=1,streams=2,3"
    # seg_duration defaults to the muxer's own 5s: 5 x 30fps = 150.
    assert unit.options["gop"] == 150
    assert "var_stream_map" not in unit.options


def test_language_fallback_is_positional_when_und_or_colliding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub(monkeypatch, _video(), _audio(0, "und"), _audio(1, "und"))
    unit = _unit(
        f"COPY (SELECT t FROM input('{SRC}') f, unnest(f.audio) t) "
        "TO 'm.m3u8' WITH (format 'hls', audio_codec 'aac')"
    )
    assert unit.options["var_stream_map"] == (
        "a:0,agroup:aud,name:a0,default:yes a:1,agroup:aud,name:a1"
    )


def test_height_fallback_is_positional_when_colliding() -> None:
    # Both rungs scale to the same width, so both names would be 480p.
    sql = _demuxed(_HLS).replace("ARRAY[640, 320]", "ARRAY[640, 640]")
    unit = _unit(sql)
    assert unit.options["var_stream_map"].startswith(
        "v:0,agroup:aud,name:v0 v:1,agroup:aud,name:v1 "
    )


def test_the_probed_default_disposition_overrides_the_first_track(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub(monkeypatch, _video(), _audio(0, "eng"), _audio(1, "fra", default=True))
    unit = _unit(
        f"COPY (SELECT t FROM input('{SRC}') f, unnest(f.audio) t) "
        "TO 'm.m3u8' WITH (format 'hls', audio_codec 'aac')"
    )
    assert "name:fra,language:fra,default:yes" in unit.options["var_stream_map"]
    assert "name:eng,language:eng,default:yes" not in unit.options["var_stream_map"]


def test_a_hand_written_map_is_refused_naming_the_compilers() -> None:
    err = _rejects(_demuxed(_HLS + ", var_stream_map 'v:0,a:0'"))
    assert err.code is ErrorCode.UNKNOWN_SINK_OPTION
    assert "compiler's to write" in err.message
    assert err.hint is not None and "v:0,agroup:aud,name:480p" in err.hint


def test_a_map_smuggled_through_codec_params_is_refused() -> None:
    err = _rejects(_demuxed(_HLS + ", codec_params 'var_stream_map=x'"))
    assert err.code is ErrorCode.SINK_OPTION_TYPE
    assert "compiler's to write" in err.message


# ---------------------------------------------------------------------------
# the vocabulary's format scoping
# ---------------------------------------------------------------------------


def test_a_manifest_option_is_refused_outside_any_manifest_format() -> None:
    err = _rejects(
        f"COPY (SELECT f.video[1] FROM input('{SRC}') f) "
        "TO 'out.mp4' WITH (hls_time 6)"
    )
    assert err.code is ErrorCode.SINK_OPTION_TYPE
    assert "belongs to format 'hls'" in err.message


def test_a_dash_option_is_refused_under_hls() -> None:
    err = _rejects(_demuxed(_HLS + ", seg_duration 6"))
    assert err.code is ErrorCode.SINK_OPTION_TYPE
    assert "belongs to format 'dash'" in err.message


def test_use_template_false_renders_a_zero() -> None:
    sql = (
        f"COPY (SELECT f.video[1] FROM input('{SRC}') f) "
        "TO 'out/m.mpd' WITH (format 'dash', video_codec 'libx264', "
        "use_template false)"
    )
    argv = _argv(sql)
    position = argv.index("-use_template")
    assert argv[position + 1] == "0"


# ---------------------------------------------------------------------------
# the derived keyframe discipline
# ---------------------------------------------------------------------------


def test_the_gop_is_derived_and_scene_cuts_disabled() -> None:
    unit = _unit(_demuxed(_HLS + ", hls_time 2"))
    assert unit.options["gop"] == 60  # 2s x 30fps
    assert unit.options["keyint_min"] == 60
    assert unit.options["sc_threshold"] == 0


def test_the_written_fps_wins_over_the_probed_one() -> None:
    sql = _demuxed(_HLS + ", hls_time 2").replace(
        "scale(f.video[1]", "scale(fps(f.video[1], 24)"
    )
    assert _unit(sql).options["gop"] == 48


def test_an_x265_sink_gets_its_private_scene_cut_param() -> None:
    with_block = (
        "format 'hls', hls_time 2, video_codec 'libx265', "
        "codec_params 'aq-mode=3', audio_codec 'aac'"
    )
    unit = _unit(_demuxed(with_block))
    assert unit.options["codec_params"] == "aq-mode=3:scenecut=0"
    assert "sc_threshold" not in unit.options


def test_an_explicit_gop_that_does_not_divide_is_refused_naming_nearest() -> None:
    err = _rejects(_demuxed(_HLS + ", hls_time 6, gop 100"))
    assert err.code is ErrorCode.SINK_OPTION_TYPE
    assert "does not divide the 180-frame segment" in err.message
    assert err.hint is not None and "90" in err.hint and "180" in err.hint


def test_an_explicit_dividing_gop_stands() -> None:
    unit = _unit(_demuxed(_HLS + ", hls_time 6, gop 90"))
    assert unit.options["gop"] == 90
    assert unit.options["keyint_min"] == 90


def test_an_unknowable_rate_is_refused_asking_for_fps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub(monkeypatch, _video(fps=None), _audio(0, "eng"), _audio(1, "fra"))
    err = _rejects(_demuxed(_HLS))
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "rate is unknown" in err.message
    assert err.hint is not None and "fps(" in err.hint


# ---------------------------------------------------------------------------
# the derived layout
# ---------------------------------------------------------------------------


def test_the_hls_layout_is_derived_beside_the_master() -> None:
    unit = _unit(_demuxed(_HLS + ", hls_segment_type 'fmp4'"))
    assert unit.path == "out/%v/index.m3u8"
    assert unit.options["master_pl_name"] == "master.m3u8"
    assert unit.options["hls_segment_filename"] == "out/%v/segment_%d.m4s"
    assert unit.options["hls_fmp4_init_filename"] == "init.mp4"


def test_mpegts_segments_take_the_ts_extension_and_no_init() -> None:
    unit = _unit(_demuxed(_HLS))
    assert unit.options["hls_segment_filename"] == "out/%v/segment_%d.ts"
    assert "hls_fmp4_init_filename" not in unit.options


def test_a_written_layout_option_pre_empts_the_derivation() -> None:
    unit = _unit(_demuxed(_HLS + ", hls_segment_filename 'seg/%v-%d.ts'"))
    assert unit.options["hls_segment_filename"] == "seg/%v-%d.ts"


# ---------------------------------------------------------------------------
# per-row options over NULL cells
# ---------------------------------------------------------------------------


def test_per_row_options_skip_the_rows_of_the_other_kind() -> None:
    with_block = (
        _HLS + ", video_bitrate ARRAY['500k', '300k'][vid.rung], "
        "audio_bitrate ARRAY['128k', '96k'][aud.pos]"
    )
    unit = _unit(_demuxed(with_block))
    assert unit.options["video_bitrate"] == ["500k", "300k"]
    assert unit.options["audio_bitrate"] == ["128k", "96k"]


def test_a_null_option_element_is_absence_for_its_track() -> None:
    # Row 2 reads a NULL element, so its rung keeps the encoder's default.
    with_block = _HLS + ", video_bitrate ARRAY['500k', NULL][vid.rung]"
    argv = _argv(_demuxed(with_block))
    assert "-b:0" in argv and argv[argv.index("-b:0") + 1] == "500k"
    assert "-b:1" not in argv


# ---------------------------------------------------------------------------
# the shapes a manifest refuses
# ---------------------------------------------------------------------------


def test_a_gathered_array_does_not_transcribe() -> None:
    err = _rejects(
        f"COPY (SELECT f.video[1], f.audio FROM input('{SRC}') f) "
        "TO 'm.m3u8' WITH (format 'hls', audio_codec 'aac')"
    )
    assert err.code is ErrorCode.ROW_COUNT_MISMATCH
    assert "one variant map entry" in err.message


def test_a_subtitle_column_is_refused_at_a_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    subtitle = StreamMeta(
        type="subtitle",
        index=0,
        metadata={"language": "eng"},
        width=None,
        height=None,
        fps=None,
        sample_rate=None,
        codec="subrip",
    )
    _stub(monkeypatch, _video(), _audio(0, "eng"), subtitle)
    err = _rejects(
        f"COPY (SELECT f.subtitle[1] FROM input('{SRC}') f) "
        "TO 'm.m3u8' WITH (format 'hls')"
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "video and audio columns" in err.message


def test_a_null_stream_cannot_be_filtered() -> None:
    sql = _demuxed(_HLS).replace(
        "SELECT vid.v, aud.t FROM", "SELECT vid.v, volume(aud.t, 0.5) FROM"
    )
    err = _rejects(sql)
    assert err.code is ErrorCode.STREAM_NOT_FOUND
    assert "NULL stream feeds" in err.message


def test_a_manifest_refuses_a_fanout_to() -> None:
    err = _rejects(
        f"COPY (SELECT t FROM input('{SRC}') f, unnest(f.audio) t) "
        "TO (t.tags.language || '.m3u8') WITH (format 'hls')"
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "one written name" in err.message


def test_a_manifest_refuses_union_all() -> None:
    err = _rejects(
        f"COPY (SELECT f.audio[1] FROM input('{SRC}') f "
        f"UNION ALL SELECT g.audio[2] FROM input('{SRC}') g) "
        "TO 'm.m3u8' WITH (format 'hls')"
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "UNION ALL" in err.message
