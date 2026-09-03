"""End-to-end proof for the ABR-ladder recipes (docs/examples.md 104-107).

Builds one real HLS ladder with the compiler (the same shape recipe 104's
own query produces, run for real rather than only compiled), then executes
recipe 105's rung-picking query and recipe 107's ladder-to-ladder query
against it, reading every result back with ffprobe. One fixture (av.mp4),
single-digit seconds.
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
