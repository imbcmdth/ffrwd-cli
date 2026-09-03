"""End-to-end proof for the ABR-ladder recipes (docs/examples.md 104-110).

Builds one real HLS ladder with the compiler (the same shape recipe 104's
own query produces, run for real rather than only compiled), then executes
recipe 105's rung-picking query and recipe 107's ladder-to-ladder query
against it, reading every result back with ffprobe. A second, DEMUXED
ladder (recipe 110 needs an audio-only row to self-join against, which an
HLS master never surfaces -- see the DASH note below) proves 109 and 110.
One fixture (av.mp4), single-digit seconds.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from ffrwd import cli
from ffrwd.probe import ProbeResult, clear_cache, probe

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
# row comes out demuxed, video-only or audio-only. `format 'dash'`, not
# `'hls'`: an HLS master reads back one row per #EXT-X-STREAM-INF variant,
# with any bound audio group folded into that row rather than surfacing on
# its own, so it never has an audio-only row for recipe 110's self-join to
# find. A DASH MPD's `<Representation>`s stay one row apiece by
# construction, so its audio-only row reads back as one.
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

_MUX_EVERY_PAIRING_SQL = """\
COPY (
  SELECT v.video[1], a.audio[1]
  FROM input('{ladder}') v, input('{ladder}') a
  WHERE v.height >= 480 AND a.height IS NULL
) TO ('out-' || v.height::text || 'p-' || a.bandwidth::text || '.mp4')
  WITH (video_codec 'libx264', crf 20, audio_codec 'aac')
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
