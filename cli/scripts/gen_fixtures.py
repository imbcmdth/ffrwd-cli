"""Generate tiny synthetic media fixtures for exec tests.

Uses ffmpeg's ``lavfi`` test-pattern sources (``testsrc2``, ``smptebars``,
``sine``) -- nothing copyrighted, nothing externally sourced, nothing large
(guardrail #8 in ffrwd-project.md). Output goes to ``tests/fixtures/``,
which is gitignored.

The set: ``testsrc.mp4`` / ``smptebars.mp4`` (video only), ``av.mp4`` (video +
one audio track), and ``av2.mp4`` / ``av3.mp4`` (video + TWO audio tracks
tagged ``language=eng`` / ``language=fra``), which is what the broadcasting
tests expand over. av2 and av3 differ only in their sine frequencies, so a
``UNION ALL`` of the two concatenates two distinguishable multi-language
sources whose language tags agree track for track. ``stereo.mp4`` adds the
one thing none of those have: a genuinely 2-CHANNEL audio track (plan 047).
``font.ttf`` is a stub TrueType file and ``attached.mkv`` is a container
carrying it, for reading attachments back. ``ladder/master.m3u8`` (HLS) and
``ladder-demuxed/master.mpd`` (DASH) are two real ABR ladders, both built by
running the compiler itself rather than a raw ffmpeg call. The first muxes
every rung (a video and audio row share each rendition). The second keeps
video and audio apart -- an audio-only rendition alongside two video-only
rungs. A DASH MPD's ``<Representation>``s stay one row apiece regardless of
type, so its audio-only row reads back as one by construction.
``ladder-demuxed-hls/master.m3u8`` is the same demuxed shape again, but HLS:
a variant naming an AUDIO group is read back video-only, and the group's own
``#EXT-X-MEDIA`` entry reads back as its own audio-only row.

Idempotent: a fixture whose output file already exists is skipped, so this
is safe to run repeatedly, including once per CI job right before the exec
test suite.

Usage::

    python scripts/gen_fixtures.py

Stdlib only -- no third-party imports, so this script itself never needs the
``[dev]`` extra installed.
"""

from __future__ import annotations

import base64
import shutil
import subprocess
import sys
from pathlib import Path

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures"

_DURATION = 4
_SIZE = "320x240"
_RATE = 15

# name -> lavfi source description
_SOURCES: dict[str, str] = {
    "testsrc.mp4": f"testsrc2=duration={_DURATION}:size={_SIZE}:rate={_RATE}",
    "smptebars.mp4": f"smptebars=duration={_DURATION}:size={_SIZE}:rate={_RATE}",
}

_AV_NAME = "av.mp4"
_AV2_NAME = "av2.mp4"
_AV3_NAME = "av3.mp4"
_AV_ENG_NAME = "av-eng.mp4"
_STEREO_NAME = "stereo.mp4"
_SUBS_NAME = "subs.en.vtt"
_AVS_NAME = "avs.mkv"
_FRAME_PNG_NAME = "frame.png"
_AV_CHAPTERS_NAME = "av-chapters.mkv"
_TAGGED_NAME = "tagged.mp4"
_AV_2ENG_NAME = "av-2eng.mp4"
_FONT_TTF_NAME = "font.ttf"
_ATTACHED_NAME = "attached.mkv"
_LADDER_MASTER_NAME = "ladder/master.m3u8"
_LADDER_DEMUXED_MASTER_NAME = "ladder-demuxed/master.mpd"
_LADDER_DEMUXED_HLS_MASTER_NAME = "ladder-demuxed-hls/master.m3u8"

# Two muxed renditions, 1080p and 720p, built by RUNNING THE COMPILER against
# av.mp4 (never hand-typed) -- read back by `input()` on a manifest path. The
# rung/rung join is what recipe 104 uses to make each row carry both a video
# and an audio stream (a muxed variant); duplicating the one physical audio
# track across both rungs, rather than an `agroup`, keeps the fixture's shape
# the one recipes 105-107 read: `r.video[1]` and `r.audio[1]` both present on
# every rendition row, no demuxed audio group to look past.
_LADDER_SQL = """\
COPY (
  WITH vid AS (
    SELECT scale(f.video[1], ARRAY[1440, 960][i.i], -2) AS v, i.i AS rung
    FROM input('av.mp4') f, generate_series(1, 2) i
  ),
  aud AS (
    SELECT a AS t, j.j AS rung
    FROM input('av.mp4') g, unnest(g.audio) a, generate_series(1, 2) j
  )
  SELECT vid.v, aud.t
  FROM vid FULL JOIN aud ON vid.rung = aud.rung
) TO 'ladder/master.m3u8'
  WITH (format 'hls', hls_time 2, hls_playlist_type 'vod',
        video_codec 'libx264', video_bitrate ARRAY['2000k', '800k'][vid.rung],
        audio_codec 'aac')
"""

# Two video-only rungs plus one audio-only rendition, keyed the same way
# `_LADDER_SQL` is but so the FULL JOIN never matches: video rungs are 1
# and 2, the one audio track's rung is `2 + a.index` -- for av.mp4's single
# track that is `2 + 1 = 3` (`.index` is the file's own stream index, and the
# video stream takes 0), disjoint from `{1, 2}` -- the same audio-rung
# expression recipe 104 uses. `format 'dash'` rather than `'hls'`: an HLS
# master reads back one row per `#EXT-X-STREAM-INF` variant only, with any
# bound audio group folded into that row rather than surfacing on its own
# (confirmed against a real probe: every variant's program carries the
# default audio track alongside its video, HLS or not), so it never has an
# audio-only row for recipe 110's self-join to find. A DASH MPD's
# `<Representation>`s stay one row apiece by construction -- video and audio
# each get their own `AdaptationSet` regardless of how the rows that named
# them were shaped -- so this is the ladder with a genuine audio-only row.
_LADDER_DEMUXED_SQL = """\
COPY (
  WITH vid AS (
    SELECT scale(f.video[1], ARRAY[1440, 960][i.i], -2) AS v, i.i AS rung
    FROM input('av.mp4') f, generate_series(1, 2) i
  ),
  aud AS (
    SELECT a AS t, 2 + a.index AS rung
    FROM input('av.mp4') g, unnest(g.audio) a
  )
  SELECT vid.v, aud.t
  FROM vid FULL JOIN aud ON vid.rung = aud.rung
) TO 'ladder-demuxed/master.mpd'
  WITH (format 'dash', seg_duration 2,
        video_codec 'libx264', video_bitrate ARRAY['2000k', '800k'][vid.rung],
        audio_codec 'aac')
"""

# Same demuxed shape as `_LADDER_DEMUXED_SQL`, `format 'hls'` instead of
# `'dash'`: a variant naming an AUDIO group reads back video-only, and the
# group's own #EXT-X-MEDIA entry reads back as its own audio-only row (see
# `probe._with_hls_stream_inf`) -- confirmed against a real ffprobe run of
# this exact shape, ffprobe's hls demuxer attaches every member of a
# referenced AUDIO group to every variant naming it.
_LADDER_DEMUXED_HLS_SQL = """\
COPY (
  WITH vid AS (
    SELECT scale(f.video[1], ARRAY[1440, 960][i.i], -2) AS v, i.i AS rung
    FROM input('av.mp4') f, generate_series(1, 2) i
  ),
  aud AS (
    SELECT a AS t, 2 + a.index AS rung
    FROM input('av.mp4') g, unnest(g.audio) a
  )
  SELECT vid.v, aud.t
  FROM vid FULL JOIN aud ON vid.rung = aud.rung
) TO 'ladder-demuxed-hls/master.m3u8'
  WITH (format 'hls', hls_time 2, hls_playlist_type 'vod',
        video_codec 'libx264', video_bitrate ARRAY['2000k', '800k'][vid.rung],
        audio_codec 'aac')
"""

# The mimetype ffmpeg itself reports for a TrueType attachment.
_FONT_MIMETYPE = "application/x-truetype-font"

# An sfnt header with an empty table directory: the twelve bytes every
# TrueType file starts with, and nothing after them. The fixture exists to be
# attached and listed, never rendered, so a real face would only add weight.
_FONT_TTF = bytes(
    [
        0x00, 0x01, 0x00, 0x00,  # sfnt version 1.0
        0x00, 0x00,              # numTables
        0x00, 0x00,              # searchRange
        0x00, 0x00,              # entrySelector
        0x00, 0x00,              # rangeShift
    ]
)

# Intro 0-1, Chapter 1 1-2, Chapter 2 2-3, Credits 3-4 -- matches cookbook
# recipe 39's pinned table exactly.
_CHAPTERS_FFMETADATA = (
    ";FFMETADATA1\n"
    "[CHAPTER]\nTIMEBASE=1/1\nSTART=0\nEND=1\ntitle=Intro\n"
    "[CHAPTER]\nTIMEBASE=1/1\nSTART=1\nEND=2\ntitle=Chapter 1\n"
    "[CHAPTER]\nTIMEBASE=1/1\nSTART=2\nEND=3\ntitle=Chapter 2\n"
    "[CHAPTER]\nTIMEBASE=1/1\nSTART=3\nEND=4\ntitle=Credits\n"
)

_SUBS_VTT = """WEBVTT

00:00:00.000 --> 00:00:00.600
Cue one.

00:00:00.700 --> 00:00:01.300
Cue two.

00:00:01.400 --> 00:00:02.000
Cue three.
"""


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _run(out_path: Path, args: list[str]) -> None:
    """Run one ffmpeg invocation, unless `out_path` is already there."""
    if out_path.exists():
        print(f"skip (already exists): {out_path}")
        return
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"generating: {out_path}")
    result = subprocess.run(["ffmpeg", "-y", *args, str(out_path)], capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise SystemExit(f"ffmpeg failed generating {out_path}")


def _generate(name: str, lavfi: str) -> None:
    _run(FIXTURES_DIR / name, ["-f", "lavfi", "-i", lavfi, "-pix_fmt", "yuv420p"])


def _generate_av() -> None:
    """testsrc2 video + one sine audio track: the simplest A/V fixture."""
    _run(
        FIXTURES_DIR / _AV_NAME,
        [
            "-f", "lavfi", "-i", f"testsrc2=duration={_DURATION}:size={_SIZE}:rate={_RATE}",
            "-f", "lavfi", "-i", f"sine=frequency=440:duration={_DURATION}",
            "-pix_fmt", "yuv420p",
            "-shortest",
        ],
    )


def _generate_av2() -> None:
    """testsrc2 video + TWO language-tagged audio tracks (sine 440 eng, 880 fra).

    The broadcasting fixture (plan 020): a bare `a.audio` over this file is a
    2-element array, and each element carries a distinct language tag, so an
    expanded query can be checked for both its node count and its provenance.
    """
    _run(
        FIXTURES_DIR / _AV2_NAME,
        [
            "-f", "lavfi", "-i", f"testsrc2=duration={_DURATION}:size={_SIZE}:rate={_RATE}",
            "-f", "lavfi", "-i", f"sine=frequency=440:duration={_DURATION}",
            "-f", "lavfi", "-i", f"sine=frequency=880:duration={_DURATION}",
            "-map", "0:v:0", "-map", "1:a:0", "-map", "2:a:0",
            "-metadata:s:a:0", "language=eng",
            "-metadata:s:a:1", "language=fra",
            "-pix_fmt", "yuv420p",
            "-shortest",
        ],
    )


def _generate_av3() -> None:
    """A second two-language fixture (sine 550 eng, 990 fra): av2's concat partner.

    Same shape as av2.mp4 -- same size, rate, duration and language tags, so a
    ``UNION ALL`` of the two is a legal concat -- but different tones, so the
    two segments of a concatenated output are told apart by ear.
    """
    _run(
        FIXTURES_DIR / _AV3_NAME,
        [
            "-f", "lavfi", "-i", f"testsrc2=duration={_DURATION}:size={_SIZE}:rate={_RATE}",
            "-f", "lavfi", "-i", f"sine=frequency=550:duration={_DURATION}",
            "-f", "lavfi", "-i", f"sine=frequency=990:duration={_DURATION}",
            "-map", "0:v:0", "-map", "1:a:0", "-map", "2:a:0",
            "-metadata:s:a:0", "language=eng",
            "-metadata:s:a:1", "language=fra",
            "-pix_fmt", "yuv420p",
            "-shortest",
        ],
    )


def _generate_av_eng() -> None:
    """testsrc2 video + ONE language-tagged audio track (sine 660, eng).

    The track-alignment fixture (RFC-009): same video shape as av2.mp4 (so a
    UNION ALL of the two is a legal concat) but a strict SUBSET of its audio -
    English only, no French - which is exactly the mismatch the track-row
    joins exist to fill.
    """
    _run(
        FIXTURES_DIR / _AV_ENG_NAME,
        [
            "-f", "lavfi", "-i", f"testsrc2=duration={_DURATION}:size={_SIZE}:rate={_RATE}",
            "-f", "lavfi", "-i", f"sine=frequency=660:duration={_DURATION}",
            "-map", "0:v:0", "-map", "1:a:0",
            "-metadata:s:a:0", "language=eng",
            "-pix_fmt", "yuv420p",
            "-shortest",
        ],
    )


def _generate_stereo() -> None:
    """testsrc2 video + ONE two-channel audio track: 440 Hz left, 880 Hz right.

    The channelsplit fixture (plan 047). Every other audio fixture here is
    mono -- ``sine`` is a single-channel source, and two sine INPUTS are two
    mono TRACKS, not two channels of one -- so the split had nothing to split.
    ``join`` is what makes one 2-channel track out of them, inside a single
    lavfi graph: ``sine[l]; sine[r]; [l][r]join=inputs=2:channel_layout=stereo``.
    Verified with ``ffprobe``: one audio stream, ``channels=2``,
    ``channel_layout=stereo``, and the two channels carry different tones, so
    a per-channel filter's effect is audible (and measurable) per side.
    """
    _run(
        FIXTURES_DIR / _STEREO_NAME,
        [
            "-f", "lavfi", "-i", f"testsrc2=duration={_DURATION}:size={_SIZE}:rate={_RATE}",
            "-f", "lavfi", "-i",
            f"sine=frequency=440:duration={_DURATION}[l];"
            f"sine=frequency=880:duration={_DURATION}[r];"
            "[l][r]join=inputs=2:channel_layout=stereo",
            "-pix_fmt", "yuv420p",
            "-shortest",
        ],
    )


def _generate_subs_vtt() -> Path:
    """A tiny, valid WEBVTT file (3 cues over ~2s) -- plain text, no ffmpeg needed."""
    out_path = FIXTURES_DIR / _SUBS_NAME
    if out_path.exists():
        print(f"skip (already exists): {out_path}")
        return out_path
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"generating: {out_path}")
    out_path.write_text(_SUBS_VTT, encoding="utf-8")
    return out_path


def _generate_avs(subs_path: Path) -> None:
    """av.mp4's video+audio muxed with subs.en.vtt's caption track into an mkv.

    RFC-004's caption story: webvtt -> srt inside an mkv container, with a
    language tag on the subtitle stream so the fixture also exercises
    provenance-tag passthrough. `-c copy` covers video/audio; `-c:s srt`
    overrides the subtitle stream specifically (ffmpeg logs a benign
    "multiple -c options" notice for this because `-c copy` also nominally
    covers the subtitle stream before being overridden -- exit code is 0 and
    the resulting subtitle codec is srt, verified by an exec test).
    """
    _run(
        FIXTURES_DIR / _AVS_NAME,
        [
            "-i", str(FIXTURES_DIR / _AV_NAME),
            "-i", str(subs_path),
            "-map", "0:v:0", "-map", "0:a:0", "-map", "1:s:0",
            "-c", "copy",
            "-c:s", "srt",
            "-metadata:s:s:0", "language=eng",
        ],
    )


def _generate_av_chapters() -> None:
    """av2.mp4 remuxed with two chapters, via ffmpeg's own ffmetadata `data:` URI.

    The chapter-reading fixture: same mechanism the compiler itself uses to
    WRITE chapters, run once by hand to build a file to READ them back from.
    Must run after av.mp4 exists.
    """
    uri = "data:text/plain;base64," + base64.b64encode(
        _CHAPTERS_FFMETADATA.encode()
    ).decode()
    _run(
        FIXTURES_DIR / _AV_CHAPTERS_NAME,
        [
            "-i", str(FIXTURES_DIR / _AV2_NAME),
            "-f", "ffmetadata", "-i", uri,
            "-map", "0:v:0", "-map", "0:a:0", "-map", "0:a:1",
            "-map_metadata", "1",
            "-map_chapters", "1",
            "-c", "copy",
        ],
    )


def _generate_av_2eng() -> None:
    """testsrc2 video + THREE language-tagged audio tracks, two sharing a
    language (sine 440 eng, 660 eng, 880 fra): the grouped fan-out fixture -
    a GROUP BY language must put both eng tracks in one output."""
    _run(
        FIXTURES_DIR / _AV_2ENG_NAME,
        [
            "-f", "lavfi", "-i", f"testsrc2=duration={_DURATION}:size={_SIZE}:rate={_RATE}",
            "-f", "lavfi", "-i", f"sine=frequency=440:duration={_DURATION}",
            "-f", "lavfi", "-i", f"sine=frequency=660:duration={_DURATION}",
            "-f", "lavfi", "-i", f"sine=frequency=880:duration={_DURATION}",
            "-map", "0:v:0", "-map", "1:a:0", "-map", "2:a:0", "-map", "3:a:0",
            "-metadata:s:a:0", "language=eng",
            "-metadata:s:a:1", "language=eng",
            "-metadata:s:a:2", "language=fra",
            "-pix_fmt", "yuv420p",
            "-shortest",
        ],
    )


def _generate_tagged() -> None:
    """testsrc2 video + sine audio with container tags (title/artist/date):
    the container-tag read fixture. No comment tag, so a CASE fill has a
    NULL to fill. mp4 adds its own `encoder` tag; its value varies by
    ffmpeg version, so nothing may pin it."""
    _run(
        FIXTURES_DIR / _TAGGED_NAME,
        [
            "-f", "lavfi", "-i", f"testsrc2=duration={_DURATION}:size={_SIZE}:rate={_RATE}",
            "-f", "lavfi", "-i", f"sine=frequency=440:duration={_DURATION}",
            "-metadata", "title=Angel One",
            "-metadata", "artist=Docs Dept",
            "-metadata", "date=2026",
            "-pix_fmt", "yuv420p",
            "-shortest",
        ],
    )


def _generate_frame_png() -> None:
    """One still frame of testsrc.mp4 (RFC-005 SS4, plan 041's PNG title-card
    exec test): `input(frame.png, loop => true, framerate => 15)` needs a
    single-frame image input, and this is the cheapest way to make one that
    is still visually distinguishable from a blank canvas. Must run after
    testsrc.mp4 exists."""
    _run(
        FIXTURES_DIR / _FRAME_PNG_NAME,
        [
            "-i", str(FIXTURES_DIR / "testsrc.mp4"),
            "-frames:v", "1",
            "-update", "1",
        ],
    )


def _generate_font_ttf() -> Path:
    """A twelve-byte TrueType stub -- the file `attached.mkv` carries, and the
    one cookbook recipe 66 attaches. No ffmpeg needed."""
    out_path = FIXTURES_DIR / _FONT_TTF_NAME
    if out_path.exists():
        print(f"skip (already exists): {out_path}")
        return out_path
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"generating: {out_path}")
    out_path.write_bytes(_FONT_TTF)
    return out_path


def _generate_attached(font_path: Path) -> None:
    """av.mp4's video+audio remuxed into an mkv carrying font.ttf.

    The attachment-reading fixture: one attachment, tagged filename and
    mimetype, alongside one video and one audio stream -- so a read also
    shows that an attachment takes no per-type stream index. Must run after
    av.mp4 exists.
    """
    _run(
        FIXTURES_DIR / _ATTACHED_NAME,
        [
            "-i", str(FIXTURES_DIR / _AV_NAME),
            "-attach", str(font_path),
            "-map", "0:v:0", "-map", "0:a:0",
            "-c", "copy",
            "-metadata:s:2", f"mimetype={_FONT_MIMETYPE}",
            "-metadata:s:2", f"filename={_FONT_TTF_NAME}",
        ],
    )


def _generate_ladder() -> None:
    """A real two-rendition HLS ladder, run through `ffrwd run` itself
    (`python -m ffrwd`, the console entry point) rather than a hand-typed
    ffmpeg call -- this is the one fixture that IS the compiler's own
    output, since recipes 105-107 read `input()` on a manifest path and
    need `-show_programs` to report real renditions. Must run after
    av.mp4 exists.
    """
    master = FIXTURES_DIR / _LADDER_MASTER_NAME
    if master.exists():
        print(f"skip (already exists): {master}")
        return
    master.parent.mkdir(parents=True, exist_ok=True)
    print(f"generating: {master}")
    result = subprocess.run(
        [sys.executable, "-m", "ffrwd", "run", _LADDER_SQL, "-y"],
        cwd=FIXTURES_DIR,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(result.stdout, file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        raise SystemExit(f"ffrwd run failed generating {master}")


def _generate_ladder_demuxed() -> None:
    """A real DEMUXED DASH ladder -- two video-only rungs, one audio-only
    rendition -- again run through `ffrwd run` rather than hand-typed.
    Recipe 110 self-joins a ladder against itself to pick video rows from
    one side and audio rows from the other; `ladder/master.m3u8` has no
    audio-only row to find (its FULL JOIN key is chosen to mux every rung),
    and neither would an HLS reading of THIS query's demuxed rows -- see
    `_LADDER_DEMUXED_SQL`'s comment for why DASH is what makes the
    audio-only row read back as its own. Must run after av.mp4 exists.
    """
    master = FIXTURES_DIR / _LADDER_DEMUXED_MASTER_NAME
    if master.exists():
        print(f"skip (already exists): {master}")
        return
    master.parent.mkdir(parents=True, exist_ok=True)
    print(f"generating: {master}")
    result = subprocess.run(
        [sys.executable, "-m", "ffrwd", "run", _LADDER_DEMUXED_SQL, "-y"],
        cwd=FIXTURES_DIR,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(result.stdout, file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        raise SystemExit(f"ffrwd run failed generating {master}")


def _generate_ladder_demuxed_hls() -> None:
    """A real DEMUXED HLS ladder -- two video-only variants naming one
    AUDIO group, whose one member is the audio-only rendition. Same
    video/audio split as `_generate_ladder_demuxed`'s DASH one, proving
    `input()` on an HLS master reads a demuxed AUDIO group back as its own
    row too. Must run after av.mp4 exists.
    """
    master = FIXTURES_DIR / _LADDER_DEMUXED_HLS_MASTER_NAME
    if master.exists():
        print(f"skip (already exists): {master}")
        return
    master.parent.mkdir(parents=True, exist_ok=True)
    print(f"generating: {master}")
    result = subprocess.run(
        [sys.executable, "-m", "ffrwd", "run", _LADDER_DEMUXED_HLS_SQL, "-y"],
        cwd=FIXTURES_DIR,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(result.stdout, file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        raise SystemExit(f"ffrwd run failed generating {master}")


def main() -> int:
    if not _ffmpeg_available():
        print("error: ffmpeg not found on PATH", file=sys.stderr)
        return 1
    for name, lavfi in _SOURCES.items():
        _generate(name, lavfi)
    _generate_av()
    _generate_av2()
    _generate_av3()
    _generate_av_eng()
    _generate_stereo()
    subs_path = _generate_subs_vtt()
    _generate_avs(subs_path)
    _generate_av_chapters()
    _generate_av_2eng()
    _generate_tagged()
    _generate_frame_png()
    _generate_attached(_generate_font_ttf())
    _generate_ladder()
    _generate_ladder_demuxed()
    _generate_ladder_demuxed_hls()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
