"""End-to-end proof for the ABR-ladder recipes (docs/examples.md 104-110,
123-130).

Builds one real HLS ladder with the compiler (the same shape recipe 104's
own query produces, run for real rather than only compiled), then executes
recipe 105's rung-picking query and recipe 107's ladder-to-ladder query
against it, reading every result back with ffprobe. A second, DEMUXED
ladder (recipe 110 needs an audio-only row to self-join against) proves 109
and 110 over DASH; a third, DEMUXED HLS ladder proves the same audio-only
row reads back from an HLS master too, video-only variants and all -- see
the notes on `_LADDER_DEMUXED_SQL` and `_LADDER_DEMUXED_HLS_SQL`. Two more,
already built by `scripts/gen_fixtures.py` rather than by this module,
prove the read side of an audio-only master and a HYBRID one (a variant
that both muxes its own audio and names an AUDIO group). Recipes 123-125
run the three shapes a stream column's cardinality decides -- a ladder's
rows carried through a CTE, one audio repeated over two rungs, and a
COALESCE over two nullable cells -- and read every master back with the
compiler's own probe. Recipes 126-127 copy-republish the demuxed DASH
ladder as HLS with `SELECT *`, and re-probe the result to prove a rung's
own name and language ride through an unmodified read (128 the same,
trimmed by WHERE; 129's exec proof is the unit tier's, a synthetic probe
being the more direct way to pin a scaled cell beside a copied one).
Recipe 130 gathers the same demuxed ladder's two CTEs with array_agg
instead of COALESCE, into one file rather than a manifest. One fixture
(av.mp4), single-digit seconds.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from ffrwd import cli
from ffrwd.probe import ProbeResult, RenditionMeta, clear_cache, probe

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = PROJECT_ROOT / "tests" / "fixtures"

# Two muxed renditions (1080p/720p), each row carrying both its own scaled
# video and the source's one audio track -- the same rung/rung join recipe
# 104 uses to keep every row a muxed variant, not a demuxed video/audio pair.
_LADDER_SQL = """\
COPY (
  WITH vid AS (
    SELECT scale(f.video[1], ARRAY[1440, 960][i.i], -2) AS v, i.i AS rung
    FROM input('{av}') f, generate_series(1, 2) i
  ),
  aud AS (
    SELECT a AS t, j.j AS rung
    FROM input('{av}') g, unnest(g.audio) a, generate_series(1, 2) j
  )
  SELECT vid.v, aud.t
  FROM vid FULL JOIN aud ON vid.rung = aud.rung
) TO 'ladder/master.m3u8'
  WITH (format 'hls', hls_time 2, hls_playlist_type 'vod',
        video_codec 'libx264', video_bitrate ARRAY['2000k', '800k'][vid.rung],
        audio_codec 'aac')
"""

_PICK_RUNG_SQL = """\
COPY (
  SELECT r.video[1], r.audio[1]
  FROM input('{ladder}') r
  WHERE r.height = 720
) TO 'rung720.mp4' WITH (video_codec 'libx264', crf 20, audio_codec 'aac')
"""

_LADDER_TO_LADDER_SQL = """\
COPY (
  SELECT r.video[1], r.audio[1]
  FROM input('{ladder}') r
) TO 'out/master.m3u8' WITH (
  format 'hls', hls_time 2, hls_playlist_type 'vod', hls_segment_type 'fmp4',
  video_codec 'libx264', audio_codec 'aac'
)
"""

_ARCHIVE_SQL = """\
COPY (
  SELECT array_agg(r.video[1])
  FROM input('{ladder}') r
  WHERE r.height >= 480
) TO 'archive.mkv' WITH (video_codec 'libx264', crf 20)
"""

# Same rung/rung shape as `_LADDER_SQL`, but keyed so the FULL JOIN never
# matches (`2 + a.index` is disjoint from the video rungs `{1, 2}`) -- every
# row comes out demuxed, video-only or audio-only. `format 'dash'`: a DASH
# MPD's `<Representation>`s stay one row apiece by construction, so its
# audio-only row reads back as one.
_LADDER_DEMUXED_SQL = """\
COPY (
  WITH vid AS (
    SELECT scale(f.video[1], ARRAY[1440, 960][i.i], -2) AS v, i.i AS rung
    FROM input('{av}') f, generate_series(1, 2) i
  ),
  aud AS (
    SELECT a AS t, 2 + a.index AS rung
    FROM input('{av}') g, unnest(g.audio) a
  )
  SELECT vid.v, aud.t
  FROM vid FULL JOIN aud ON vid.rung = aud.rung
) TO 'ladder-demuxed/master.mpd'
  WITH (format 'dash', seg_duration 2,
        video_codec 'libx264', video_bitrate ARRAY['2000k', '800k'][vid.rung],
        audio_codec 'aac')
"""

# Same demuxed shape again, `format 'hls'`: a variant naming an AUDIO group
# reads back video-only, and the group's own #EXT-X-MEDIA entry reads back
# as its own audio-only row -- ffprobe's hls demuxer attaches every member
# of a referenced AUDIO group to every variant naming it, confirmed against
# a real probe of this exact query's output.
_LADDER_DEMUXED_HLS_SQL = """\
COPY (
  WITH vid AS (
    SELECT scale(f.video[1], ARRAY[1440, 960][i.i], -2) AS v, i.i AS rung
    FROM input('{av}') f, generate_series(1, 2) i
  ),
  aud AS (
    SELECT a AS t, 2 + a.index AS rung
    FROM input('{av}') g, unnest(g.audio) a
  )
  SELECT vid.v, aud.t
  FROM vid FULL JOIN aud ON vid.rung = aud.rung
) TO 'ladder-demuxed-hls/master.m3u8'
  WITH (format 'hls', hls_time 2, hls_playlist_type 'vod',
        video_codec 'libx264', video_bitrate ARRAY['2000k', '800k'][vid.rung],
        audio_codec 'aac')
"""

_MUX_EVERY_PAIRING_SQL = """\
COPY (
  SELECT v.video[1], a.audio[1]
  FROM input('{ladder}') v, input('{ladder}') a
  WHERE v.height >= 480 AND a.height IS NULL
) TO ('out-' || v.height::text || 'p-' || a.bandwidth::text || '.mp4')
  WITH (video_codec 'libx264', crf 20, audio_codec 'aac')
"""


# Recipe 123: the muxed ladder's own rows, carried through two CTEs and
# joined on keys that never meet, so what was two muxed variants comes out as
# two video-only ones plus the audio rendition of the 720 rung.
_RE_LAY_DEMUXED_SQL = """\
COPY (
  WITH vid AS (
    SELECT r.video[1] AS v, r.height AS rung
    FROM input('{ladder}') r
  ),
  aud AS (
    SELECT s.audio[1] AS t, 1000 + s.bandwidth AS rung
    FROM input('{ladder}') s
    WHERE s.height = 720
  )
  SELECT vid.v, aud.t
  FROM vid FULL JOIN aud ON vid.rung = aud.rung
) TO 'out/master.m3u8' WITH (
  format 'hls', hls_time 2, hls_playlist_type 'vod', hls_segment_type 'fmp4',
  video_codec 'libx264', audio_codec 'aac'
)
"""

# Recipe 124: two rungs from a series, and one audio column that does not
# vary per row -- the same stream on both variants, split once.
_MUXED_FROM_ONE_FILE_SQL = """\
COPY (
  SELECT scale(f.video[1], ARRAY[320, 160][i.i], -2), f.audio[1]
  FROM input('{av}') f, generate_series(1, 2) i
) TO 'out/master.m3u8'
  WITH (format 'hls', hls_time 2, hls_playlist_type 'vod',
        video_codec 'libx264', audio_codec 'aac')
"""

# Recipe 125: a hybrid master off the demuxed DASH ladder -- muxed variants
# whose audio COALESCE took from the audio-only rendition, plus that
# rendition as its own row.
_HYBRID_SQL = """\
COPY (
  WITH vid AS (
    SELECT v.video[1] AS v, a.audio[1] AS a, v.height AS rung
    FROM input('{ladder}') v, input('{ladder}') a
    WHERE v.height IS NOT NULL AND a.height IS NULL
  ),
  aud AS (
    SELECT b.audio[1] AS t, 1000 + b.bandwidth AS rung
    FROM input('{ladder}') b
    WHERE b.height IS NULL
  )
  SELECT vid.v, COALESCE(vid.a, aud.t)
  FROM vid FULL JOIN aud ON vid.rung = aud.rung
) TO 'out/master.m3u8' WITH (
  format 'hls', hls_time 2, hls_playlist_type 'vod', hls_segment_type 'fmp4',
  video_codec 'libx264', audio_codec 'aac'
)
"""


# Recipes 126-127: a copy republish, `SELECT *` over a rendition alias --
# every cell an unmodified read, so ffmpeg needs no codec and every rung's
# own name (and, where it has one, language) rides straight into the new
# master's variant map.
_REPUBLISH_STAR_SQL = """\
COPY (SELECT * FROM input('{ladder}') r) TO 'out/master.m3u8' WITH (format 'hls')
"""


# Recipe 130: the demuxed ladder's two CTEs gathered with array_agg instead
# of COALESCE'd row by row -- every video rung plus the audio rendition, as
# separate tracks in one mkv rather than a manifest's separate variants.
_GATHER_SQL = """\
COPY (
  WITH vid AS (
    SELECT v.video[1] AS v, v.height AS rung
    FROM input('{ladder}') v
    WHERE v.height IS NOT NULL
  ),
  aud AS (
    SELECT b.audio[1] AS t, 1000 + b.bandwidth AS rung
    FROM input('{ladder}') b
    WHERE b.height IS NULL
  )
  SELECT array_agg(vid.v), array_agg(aud.t)
  FROM vid FULL JOIN aud ON vid.rung = aud.rung
) TO 'out/all.mkv'
"""


@pytest.fixture(scope="module")
def _fixtures() -> None:
    subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "gen_fixtures.py")],
        check=True,
    )


def _probe_json(path: Path) -> dict[str, object]:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-print_format", "json",
            "-show_format", "-show_streams", str(path),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    data: dict[str, object] = json.loads(result.stdout)
    return data


def _variant_count(master: Path) -> int:
    text = master.read_text(encoding="utf-8")
    return text.count("#EXT-X-STREAM-INF")


@pytest.mark.exec
def test_a_real_ladder_reads_re_encodes_and_repackages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _fixtures: None
) -> None:
    monkeypatch.chdir(tmp_path)
    av = (FIXTURES_DIR / "av.mp4").as_posix()
    (tmp_path / "ladder").mkdir()

    # 1. Build a real ABR ladder with the compiler (recipe 104's shape).
    assert cli.main(["run", _LADDER_SQL.format(av=av), "-y"]) == 0
    master = tmp_path / "ladder" / "master.m3u8"
    assert master.exists()

    # 2. Read it back: input() on the manifest sees both rungs, VOD.
    clear_cache()
    ladder_probe = probe(str(master))
    assert isinstance(ladder_probe, ProbeResult)
    assert ladder_probe.live is False
    assert len(ladder_probe.renditions) >= 2
    heights = sorted(r.height for r in ladder_probe.renditions)
    assert heights == [720, 1080]

    # 3. Recipe 105: pick the 720 rung, re-encode it alone.
    ladder = master.as_posix()
    assert cli.main(["run", _PICK_RUNG_SQL.format(ladder=ladder), "-y"]) == 0
    rung = tmp_path / "rung720.mp4"
    assert rung.exists()
    rung_streams = _probe_json(rung)["streams"]
    assert isinstance(rung_streams, list)
    video_streams = [s for s in rung_streams if s["codec_type"] == "video"]
    assert len(video_streams) == 1
    assert video_streams[0]["height"] == 720

    # 4. Recipe 107: every rendition survives, repackaged as its own ladder.
    (tmp_path / "out").mkdir()
    assert cli.main(["run", _LADDER_TO_LADDER_SQL.format(ladder=ladder), "-y"]) == 0
    out_master = tmp_path / "out" / "master.m3u8"
    assert out_master.exists()
    assert _variant_count(out_master) == _variant_count(master)


@pytest.mark.exec
def test_a_ladder_archives_into_one_file_with_a_track_per_rung(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _fixtures: None
) -> None:
    """Recipe 109: `array_agg(r.video[1])` over the muxed ladder collapses
    the surviving rung rows into one mkv, one video track per rung."""
    monkeypatch.chdir(tmp_path)
    av = (FIXTURES_DIR / "av.mp4").as_posix()
    (tmp_path / "ladder").mkdir()

    assert cli.main(["run", _LADDER_SQL.format(av=av), "-y"]) == 0
    master = tmp_path / "ladder" / "master.m3u8"
    assert master.exists()

    ladder = master.as_posix()
    assert cli.main(["run", _ARCHIVE_SQL.format(ladder=ladder), "-y"]) == 0
    archive = tmp_path / "archive.mkv"
    assert archive.exists()

    archive_streams = _probe_json(archive)["streams"]
    assert isinstance(archive_streams, list)
    video_streams = [s for s in archive_streams if s["codec_type"] == "video"]
    assert len(video_streams) == 2
    assert sorted(s["height"] for s in video_streams) == [720, 1080]


@pytest.mark.exec
def test_a_demuxed_ladder_muxes_every_rung_with_every_rendition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _fixtures: None
) -> None:
    """Recipe 110: a self-join over a DEMUXED ladder pairs each video rung
    (`v.height >= 480`) with each audio-only rendition (`a.height IS NULL`),
    one mp4 per pairing -- two here, since the ladder has two video rungs
    and one audio rendition."""
    monkeypatch.chdir(tmp_path)
    av = (FIXTURES_DIR / "av.mp4").as_posix()
    (tmp_path / "ladder-demuxed").mkdir()

    assert cli.main(["run", _LADDER_DEMUXED_SQL.format(av=av), "-y"]) == 0
    master = tmp_path / "ladder-demuxed" / "master.mpd"
    assert master.exists()

    ladder = master.as_posix()
    assert cli.main(["run", _MUX_EVERY_PAIRING_SQL.format(ladder=ladder), "-y"]) == 0

    outputs = sorted(tmp_path.glob("out-*p-*.mp4"))
    names = [p.name for p in outputs]
    assert len(names) == 2  # one file per (video rung, audio rendition) pairing
    assert any(name.startswith("out-1080p-") for name in names)
    assert any(name.startswith("out-720p-") for name in names)

    for output in outputs:
        streams = _probe_json(output)["streams"]
        assert isinstance(streams, list)
        assert len([s for s in streams if s["codec_type"] == "video"]) == 1
        assert len([s for s in streams if s["codec_type"] == "audio"]) == 1


@pytest.mark.exec
def test_a_demuxed_hls_ladder_probes_video_only_and_audio_only_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _fixtures: None
) -> None:
    """The gap this closes: an HLS master's variant that names an AUDIO
    group used to read back MUXED (ffprobe folds every member of a
    referenced group into each variant naming it, video included) --
    `input()` on a real demuxed HLS master now sees the two video rungs
    video-only and the one audio track as its own row, the same shape
    `test_a_demuxed_ladder_muxes_every_rung_with_every_rendition` already
    gets from the DASH fixture."""
    monkeypatch.chdir(tmp_path)
    av = (FIXTURES_DIR / "av.mp4").as_posix()
    (tmp_path / "ladder-demuxed-hls").mkdir()

    assert cli.main(["run", _LADDER_DEMUXED_HLS_SQL.format(av=av), "-y"]) == 0
    master = tmp_path / "ladder-demuxed-hls" / "master.m3u8"
    assert master.exists()

    clear_cache()
    ladder_probe = probe(str(master))
    assert isinstance(ladder_probe, ProbeResult)
    assert ladder_probe.live is False

    video_only = [r for r in ladder_probe.renditions if r.height is not None]
    audio_only = [r for r in ladder_probe.renditions if r.height is None]
    assert sorted(r.height for r in video_only if r.height is not None) == [720, 1080]
    assert len(audio_only) >= 1
    for rendition in audio_only:
        assert [s.type for s in rendition.streams] == ["audio"]


@pytest.mark.exec
def test_an_audio_only_hls_master_probes_as_one_row(_fixtures: None) -> None:
    """ffmpeg's hls muxer writes an audio-only destination as one variant
    naming an AUDIO group and no video: one row, the group's own."""
    master = FIXTURES_DIR / "ladder-audio-only" / "master.m3u8"
    assert master.exists()

    clear_cache()
    ladder_probe = probe(str(master))
    assert isinstance(ladder_probe, ProbeResult)
    assert len(ladder_probe.renditions) == 1

    rendition = ladder_probe.renditions[0]
    assert rendition.height is None
    assert [s.type for s in rendition.streams] == ["audio"]
    assert rendition.name == "audio_0"


@pytest.mark.exec
def test_a_hybrid_hls_master_keeps_its_own_audio_beside_the_group(
    _fixtures: None,
) -> None:
    """Muxed variants that also name an AUDIO group keep their own audio;
    the group is its own row; names come from the playlist directories."""
    master = FIXTURES_DIR / "ladder-hybrid" / "master.m3u8"
    assert master.exists()

    clear_cache()
    ladder_probe = probe(str(master))
    assert isinstance(ladder_probe, ProbeResult)
    assert ladder_probe.live is False

    video = [r for r in ladder_probe.renditions if r.height is not None]
    audio_only = [r for r in ladder_probe.renditions if r.height is None]
    assert sorted(r.height for r in video) == [720, 1080]
    for rendition in video:
        assert sorted(s.type for s in rendition.streams) == ["audio", "video"]
    assert {r.height: r.name for r in video} == {
        1080: "v1080p",
        720: "v720p",
    }

    assert len(audio_only) == 1
    assert [s.type for s in audio_only[0].streams] == ["audio"]
    assert audio_only[0].name == "audio_2"


def _rows(master: Path) -> list[tuple[int | None, list[str]]]:
    """One master read back the way a query reads it: each rendition row's
    height and the kinds it carries, in manifest order."""
    clear_cache()
    result = probe(str(master))
    assert isinstance(result, ProbeResult)
    return [(r.height, [s.type for s in r.streams]) for r in result.renditions]


@pytest.mark.exec
def test_a_muxed_ladder_re_lays_as_a_demuxed_master(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _fixtures: None
) -> None:
    """Recipe 123: a rendition table read through a CTE is one row per rung,
    so the muxed ladder's two rungs become two video-only variants and the
    720 rung's audio becomes the group's own row."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "out").mkdir()
    ladder = (FIXTURES_DIR / "ladder" / "master.m3u8").as_posix()

    assert cli.main(["run", _RE_LAY_DEMUXED_SQL.format(ladder=ladder), "-y"]) == 0
    master = tmp_path / "out" / "master.m3u8"
    assert master.exists()
    assert _rows(master) == [(1080, ["video"]), (720, ["video"]), (None, ["audio"])]


@pytest.mark.exec
def test_one_file_becomes_a_muxed_ladder_sharing_one_audio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _fixtures: None
) -> None:
    """Recipe 124: the file's one audio track is a single stream over a
    two-row relation, so both variants carry it -- muxed, not a group."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "out").mkdir()
    av = (FIXTURES_DIR / "av.mp4").as_posix()

    assert cli.main(["run", _MUXED_FROM_ONE_FILE_SQL.format(av=av), "-y"]) == 0
    master = tmp_path / "out" / "master.m3u8"
    assert master.exists()
    assert _rows(master) == [(240, ["video", "audio"]), (120, ["video", "audio"])]


@pytest.mark.exec
def test_a_hybrid_master_muxes_its_variants_and_names_a_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _fixtures: None
) -> None:
    """Recipe 125: COALESCE over two nullable stream cells gives every
    variant an audio cell, and the audio-only row keeps a row of its own --
    the hybrid shape `test_a_hybrid_hls_master_keeps_its_own_audio_beside_
    the_group` reads back from a fixture, here WRITTEN by a query."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "out").mkdir()
    ladder = (FIXTURES_DIR / "ladder-demuxed" / "master.mpd").as_posix()

    assert cli.main(["run", _HYBRID_SQL.format(ladder=ladder), "-y"]) == 0
    master = tmp_path / "out" / "master.m3u8"
    assert master.exists()
    assert _rows(master) == [
        (1080, ["video", "audio"]),
        (720, ["video", "audio"]),
        (None, ["audio"]),
    ]


@pytest.mark.exec
def test_array_agg_gathers_the_demuxed_ladder_into_one_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _fixtures: None
) -> None:
    """Recipe 130: array_agg over each side of the FULL JOIN skips the row
    the other CTE did not match, gathering both video rungs and the one
    audio rendition into a single mkv rather than a manifest's variants."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "out").mkdir()
    ladder = (FIXTURES_DIR / "ladder-demuxed" / "master.mpd").as_posix()

    assert cli.main(["run", _GATHER_SQL.format(ladder=ladder), "-y"]) == 0
    output = tmp_path / "out" / "all.mkv"
    assert output.exists()

    streams = _probe_json(output)["streams"]
    assert isinstance(streams, list)
    assert len([s for s in streams if s["codec_type"] == "video"]) == 2
    assert len([s for s in streams if s["codec_type"] == "audio"]) == 1


def _renditions_by_kind(master: Path) -> dict[tuple[str, ...], RenditionMeta]:
    """A master's renditions keyed by the sorted kinds they carry -- one
    video-only, one audio-only, matching both fixtures this proof reads."""
    clear_cache()
    result = probe(str(master))
    assert isinstance(result, ProbeResult)
    return {tuple(sorted(s.type for s in r.streams)): r for r in result.renditions}


def _assert_names_round_trip(
    before: dict[tuple[str, ...], RenditionMeta], after: dict[tuple[str, ...], RenditionMeta]
) -> None:
    """What a `SELECT *` copy-republish's own var_stream_map WROTE, checked
    against what ffmpeg's hls muxer actually put on disk (confirmed against
    a real ffmpeg run, hand-built `-var_stream_map` and all, independent of
    the compiler): a video rung's name rides in its variant playlist's own
    directory, `out/v%v/...` -- so a republish always gains one literal `v`
    over the source's own name, landing `0` in `v0/`, or a name that was
    ALREADY `v1080p` (an earlier compiler's own output) in `vv1080p/`. An
    audio-only row's `language:` rides in its #EXT-X-MEDIA LANGUAGE=
    attribute and reads back exactly -- but ffmpeg's hls muxer does not
    carry `name:` into that entry's NAME= at all: it always writes
    `audio_<output stream index>` there regardless of what var_stream_map
    said, so an audio rendition's own name does not survive an HLS
    round-trip through this attribute. The compiler still WROTE the real
    name into var_stream_map (recipes 126-127's own pinned command proves
    that); this is ffmpeg's own limitation on the read side, not the
    compiler's to work around.
    """
    assert before.keys() == after.keys()
    for kind, before_rendition in before.items():
        after_rendition = after[kind]
        assert after_rendition.language == before_rendition.language
        if kind == ("audio",):
            assert after_rendition.name == "audio_2"
        else:
            assert after_rendition.name == f"v{before_rendition.name}"


@pytest.mark.exec
def test_a_star_republish_of_the_demuxed_dash_ladder_keeps_its_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _fixtures: None
) -> None:
    """Recipes 126-127: `SELECT *` over `input(<dash mpd>) r` copy-republishes
    every rung as HLS -- see `_assert_names_round_trip` for what a re-probe
    of the result can and cannot still see."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "out").mkdir()
    source = FIXTURES_DIR / "ladder-demuxed" / "master.mpd"
    before = _renditions_by_kind(source)

    ladder = source.as_posix()
    assert cli.main(["run", _REPUBLISH_STAR_SQL.format(ladder=ladder), "-y"]) == 0
    after = _renditions_by_kind(tmp_path / "out" / "master.m3u8")
    _assert_names_round_trip(before, after)


@pytest.mark.exec
def test_a_star_republish_of_the_demuxed_hls_ladder_keeps_its_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _fixtures: None
) -> None:
    """The same proof over an HLS source whose OWN names are already
    directory-derived (`v1080p`, `v720p`, from a prior compiler run) -- a
    republish adds the same one `v`, landing in `vv1080p/`."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "out").mkdir()
    source = FIXTURES_DIR / "ladder-demuxed-hls" / "master.m3u8"
    before = _renditions_by_kind(source)

    ladder = source.as_posix()
    assert cli.main(["run", _REPUBLISH_STAR_SQL.format(ladder=ladder), "-y"]) == 0
    after = _renditions_by_kind(tmp_path / "out" / "master.m3u8")
    _assert_names_round_trip(before, after)
