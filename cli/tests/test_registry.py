"""Tests for ffrwd.registry.

Monkeypatched tests (subprocess/binaries.ffmpeg_path faked, disk cache
redirected to a tmp dir) are unmarked so the default suite (`pytest`, which
runs `-m "not exec"`) stays offline. Tests that shell out to a real ffmpeg
are marked `@pytest.mark.exec`.

Fixtures below are REAL captured output from `ffmpeg -hide_banner -filters`
and `ffmpeg -hide_banner -help filter=<name>` against
`ffmpeg version 7.1-full_build-www.gyan.dev` (captured 2026-08), trimmed to
representative subsets -- NOT retyped by hand. Exceptions are explicitly
marked CONSTRUCTED below (synthetic snippets exercising option-table shapes --
`binary`/`dictionary` types, malformed option lines -- that were not
observed in a full scan of this ffmpeg build's ~460 included filters).
"""

# ruff: noqa: E501
# The captured -filters/-help fixture text above and below is embedded
# verbatim (byte-for-byte, not retyped/rewrapped) so it stays trustworthy
# as a record of real ffmpeg output; several lines exceed the 100-col
# limit as a result. This is a whole-file exemption rather than per-line
# noqa comments because a per-line comment would corrupt the fixture text
# it would have to sit inside of.

from __future__ import annotations

import json
import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest

from ffrwd import registry as registry_mod
from ffrwd.registry import DynamicFilter, FilterOption, SourceFilter, clear_cache, load

VERSION_LINE = "ffmpeg version 7.1-full_build-www.gyan.dev Copyright (c) 2000-2024 the FFmpeg developers"

# --- captured -filters subset (real lines from a 567-line capture) ---------

FILTERS_SNIPPET = """\
Filters:
  T.. = Timeline support
  .S. = Slice threading
  ..C = Command support
  A = Audio input/output
  V = Video input/output
  N = Dynamic number and/or type of input/output
  | = Source or sink filter
 TSC aap               AA->A      Apply Affine Projection algorithm to first audio stream.
 ... abench            A->A       Benchmark part of a filtergraph.
 .S. acrossover        A->N       Split audio into per-bands streams.
 TSC gblur             V->V       Apply Gaussian Blur filter.
 TSC overlay           VV->V      Overlay a video source on top of the input.
 ..C scale             V->V       Scale the input video size and/or convert the image format.
 ... split             V->N       Pass on the input to N video outputs.
 ... subtitles         V->V       Render text subtitles onto input video using the libass library.
 ... aecho             A->A       Add echoing to the audio.
 ..C concat            N->N       Concatenate audio and video streams.
 ... testsrc           |->V       Generate test pattern.
 ... testsrc2          |->V       Generate another test pattern.
 ..C color             |->V       Provide an uniformly colored input.
 ... nullsrc           |->V       Null video source, return unprocessed video frames.
 ... anullsrc          |->A       Null audio source, return empty audio frames.
 ... sine              |->A       Generate sine wave audio signal.
 ..C avsynctest        |->AV      Generate an Audio Video Sync Test.
 ..C amovie            |->N       Read audio from a movie source.
 ... anullsink         A->|       Do absolutely nothing with the input audio.
 ... nullsink          V->|       Do absolutely nothing with the input video.
 .S. xfade             VV->V      Cross fade one video with another video.
 TSC hqdn3d            V->V       Apply a High Quality 3D Denoiser.
 T.C feedback          VV->VV     Apply feedback video filter.
 TSC colormap          VVV->V     Apply custom Color Maps to video stream.
 TSC threshold         VVVV->V    Threshold first video stream using other video streams.
 ..C a3dscope          A->V       Convert input audio to 3d scope video output.
 ..C movie             |->N       Read from a movie source.
"""

INCLUDED_NAMES = {
    "aap", "abench", "gblur", "overlay", "scale", "subtitles", "aecho",
    "xfade", "hqdn3d", "colormap", "threshold", "a3dscope",
}
# Sources: single V/A output, |->V or |->A. Excluded from `get()`/`names()`
# same as before (v1 scope check for column-function lookup), but now
# retained and reachable via `get_source()`/`source_names()`.
SOURCE_NAMES = {"testsrc", "testsrc2", "color", "nullsrc", "anullsrc", "sine"}
EXCLUDED_NAMES = {
    "acrossover",  # A->N: dynamic pads
    "split",       # V->N: dynamic pads
    "concat",      # N->N: dynamic pads
    "testsrc",     # |->V: source (now in SOURCE_NAMES, still excluded from get()/names())
    "testsrc2",    # |->V: source
    "color",       # |->V: source
    "nullsrc",     # |->V: source
    "anullsrc",    # |->A: source
    "sine",        # |->A: source
    "avsynctest",  # |->AV: multi-output source -- excluded from get_source() too
    "anullsink",   # A->|: sink
    "nullsink",    # V->|: sink
    "feedback",    # VV->VV: multi-output
    "movie",       # |->N: source + dynamic pads
    "amovie",      # |->N: source + dynamic pads -- excluded from get_source() too
}

HELP_GBLUR = """\
Filter gblur
  Apply Gaussian Blur filter.
    slice threading supported
    Inputs:
       #0: default (video)
    Outputs:
       #0: default (video)
gblur AVOptions:
   sigma             <float>      ..FV.....T. set sigma (from 0 to 1024) (default 0.5)
   steps             <int>        ..FV.....T. set number of steps (from 1 to 6) (default 1)
   planes            <int>        ..FV.....T. set planes to filter (from 0 to 15) (default 15)
   sigmaV            <float>      ..FV.....T. set vertical sigma (from -1 to 1024) (default -1)

This filter has support for timeline through the 'enable' option.
"""

HELP_XFADE = """\
Filter xfade
  Cross fade one video with another video.
    slice threading supported
    Inputs:
       #0: main (video)
       #1: xfade (video)
    Outputs:
       #0: default (video)
xfade AVOptions:
   transition        <int>        ..FV....... set cross fade transition (from -1 to 57) (default fade)
     custom          -1           ..FV....... custom transition
     fade            0            ..FV....... fade transition
     wipeleft        1            ..FV....... wipe left transition
     wiperight       2            ..FV....... wipe right transition
     wipeup          3            ..FV....... wipe up transition
     wipedown        4            ..FV....... wipe down transition
     slideleft       5            ..FV....... slide left transition
     slideright      6            ..FV....... slide right transition
     slideup         7            ..FV....... slide up transition
     slidedown       8            ..FV....... slide down transition
     circlecrop      9            ..FV....... circle crop transition
     rectcrop        10           ..FV....... rect crop transition
     distance        11           ..FV....... distance transition
     fadeblack       12           ..FV....... fadeblack transition
     fadewhite       13           ..FV....... fadewhite transition
     radial          14           ..FV....... radial transition
     smoothleft      15           ..FV....... smoothleft transition
     smoothright     16           ..FV....... smoothright transition
     smoothup        17           ..FV....... smoothup transition
     smoothdown      18           ..FV....... smoothdown transition
     circleopen      19           ..FV....... circleopen transition
     circleclose     20           ..FV....... circleclose transition
     vertopen        21           ..FV....... vert open transition
     vertclose       22           ..FV....... vert close transition
     horzopen        23           ..FV....... horz open transition
     horzclose       24           ..FV....... horz close transition
     dissolve        25           ..FV....... dissolve transition
     pixelize        26           ..FV....... pixelize transition
     diagtl          27           ..FV....... diag tl transition
     diagtr          28           ..FV....... diag tr transition
     diagbl          29           ..FV....... diag bl transition
     diagbr          30           ..FV....... diag br transition
     hlslice         31           ..FV....... hl slice transition
     hrslice         32           ..FV....... hr slice transition
     vuslice         33           ..FV....... vu slice transition
     vdslice         34           ..FV....... vd slice transition
     hblur           35           ..FV....... hblur transition
     fadegrays       36           ..FV....... fadegrays transition
     wipetl          37           ..FV....... wipe tl transition
     wipetr          38           ..FV....... wipe tr transition
     wipebl          39           ..FV....... wipe bl transition
     wipebr          40           ..FV....... wipe br transition
     squeezeh        41           ..FV....... squeeze h transition
     squeezev        42           ..FV....... squeeze v transition
     zoomin          43           ..FV....... zoom in transition
     fadefast        44           ..FV....... fast fade transition
     fadeslow        45           ..FV....... slow fade transition
     hlwind          46           ..FV....... hl wind transition
     hrwind          47           ..FV....... hr wind transition
     vuwind          48           ..FV....... vu wind transition
     vdwind          49           ..FV....... vd wind transition
     coverleft       50           ..FV....... cover left transition
     coverright      51           ..FV....... cover right transition
     coverup         52           ..FV....... cover up transition
     coverdown       53           ..FV....... cover down transition
     revealleft      54           ..FV....... reveal left transition
     revealright     55           ..FV....... reveal right transition
     revealup        56           ..FV....... reveal up transition
     revealdown      57           ..FV....... reveal down transition
   duration          <duration>   ..FV....... set cross fade duration (default 1)
   offset            <duration>   ..FV....... set cross fade start relative to first input stream (default 0)
   expr              <string>     ..FV....... set expression for custom transition

"""

HELP_OVERLAY = """\
Filter overlay
  Overlay a video source on top of the input.
    slice threading supported
    Inputs:
       #0: main (video)
       #1: overlay (video)
    Outputs:
       #0: default (video)
overlay AVOptions:
   x                 <string>     ..FV....... set the x expression (default "0")
   y                 <string>     ..FV....... set the y expression (default "0")
   eof_action        <int>        ..FV....... Action to take when encountering EOF from secondary input  (from 0 to 2) (default repeat)
     repeat          0            ..FV....... Repeat the previous frame.
     endall          1            ..FV....... End both streams.
     pass            2            ..FV....... Pass through the main input.
   eval              <int>        ..FV....... specify when to evaluate expressions (from 0 to 1) (default frame)
     init            0            ..FV....... eval expressions once during initialization
     frame           1            ..FV....... eval expressions per-frame
   shortest          <boolean>    ..FV....... force termination when the shortest input terminates (default false)
   format            <int>        ..FV....... set output format (from 0 to 8) (default yuv420)
     yuv420          0            ..FV.......
     yuv420p10       1            ..FV.......
     yuv422          2            ..FV.......
     yuv422p10       3            ..FV.......
     yuv444          4            ..FV.......
     yuv444p10       5            ..FV.......
     rgb             6            ..FV.......
     gbrp            7            ..FV.......
     auto            8            ..FV.......
   repeatlast        <boolean>    ..FV....... repeat overlay of the last overlay frame (default true)
   alpha             <int>        ..FV....... alpha format (from 0 to 1) (default straight)
     straight        0            ..FV.......
     premultiplied   1            ..FV.......

framesync AVOptions:
   eof_action        <int>        ..FV....... Action to take when encountering EOF from secondary input  (from 0 to 2) (default repeat)
     repeat          0            ..FV....... Repeat the previous frame.
     endall          1            ..FV....... End both streams.
     pass            2            ..FV....... Pass through the main input.
   shortest          <boolean>    ..FV....... force termination when the shortest input terminates (default false)
   repeatlast        <boolean>    ..FV....... extend last frame of secondary streams beyond EOF (default true)
   ts_sync_mode      <int>        ..FV....... How strictly to sync streams based on secondary input timestamps (from 0 to 1) (default default)
     default         0            ..FV....... Frame from secondary input with the nearest lower or equal timestamp to the primary input frame
     nearest         1            ..FV....... Frame from secondary input with the absolute nearest timestamp to the primary input frame

This filter has support for timeline through the 'enable' option.
"""

HELP_HQDN3D = """\
Filter hqdn3d
  Apply a High Quality 3D Denoiser.
    slice threading supported
    Inputs:
       #0: default (video)
    Outputs:
       #0: default (video)
hqdn3d AVOptions:
   luma_spatial      <double>     ..FV.....T. spatial luma strength (from 0 to DBL_MAX) (default 0)
   chroma_spatial    <double>     ..FV.....T. spatial chroma strength (from 0 to DBL_MAX) (default 0)
   luma_tmp          <double>     ..FV.....T. temporal luma strength (from 0 to DBL_MAX) (default 0)
   chroma_tmp        <double>     ..FV.....T. temporal chroma strength (from 0 to DBL_MAX) (default 0)

This filter has support for timeline through the 'enable' option.
"""

HELP_AECHO = """\
Filter aecho
  Add echoing to the audio.
    Inputs:
       #0: default (audio)
    Outputs:
       #0: default (audio)
aecho AVOptions:
   in_gain           <float>      ..F.A...... set signal input gain (from 0 to 1) (default 0.6)
   out_gain          <float>      ..F.A...... set signal output gain (from 0 to 1) (default 0.3)
   delays            <string>     ..F.A...... set list of signal delays (default "1000")
   decays            <string>     ..F.A...... set list of signal decays (default "0.5")

"""

# testsrc is a SOURCE filter (|->V) -- retained and reachable
# through the normal `Registry.options()` lazy path since testsrc IS a
# known name (self._sources), unlike sinks/multi-output/dynamic sources
# which stay fully excluded. Its short/long alias pairs (size/s, rate/r,
# duration/d, decimals/n) are the clearest real example of alias dedup;
# "Inputs: none (source filter)" (vs. "#0: ...") is never parsed by this
# module (only the AVOptions block matters), so it needs no special code.
HELP_TESTSRC = """\
Filter testsrc
  Generate test pattern.
    Inputs:
        none (source filter)
    Outputs:
       #0: default (video)
testsrc AVOptions:
   size              <image_size> ..FV....... set video size (default "320x240")
   s                 <image_size> ..FV....... set video size (default "320x240")
   rate              <video_rate> ..FV....... set video rate (default "25")
   r                 <video_rate> ..FV....... set video rate (default "25")
   duration          <duration>   ..FV....... set video duration (default -0.000001)
   d                 <duration>   ..FV....... set video duration (default -0.000001)
   sar               <rational>   ..FV....... set video sample aspect ratio (from 0 to INT_MAX) (default 1/1)
   decimals          <int>        ..FV....... set number of decimals to show (from 0 to 17) (default 0)
   n                 <int>        ..FV....... set number of decimals to show (from 0 to 17) (default 0)

"""

# anullsrc: another real short/long alias set (channel_layout/cl,
# sample_rate/r, nb_samples/n, duration/d), long name first, and a
# <channel_layout>-typed option (falls through the type map to "str").
HELP_ANULLSRC = """\
Filter anullsrc
  Null audio source, return empty audio frames.
    Inputs:
        none (source filter)
    Outputs:
       #0: default (audio)
anullsrc AVOptions:
   channel_layout    <channel_layout> ..F.A...... set channel_layout (default "stereo")
   cl                <channel_layout> ..F.A...... set channel_layout (default "stereo")
   sample_rate       <int>        ..F.A...... set sample rate (from 1 to INT_MAX) (default 44100)
   r                 <int>        ..F.A...... set sample rate (from 1 to INT_MAX) (default 44100)
   nb_samples        <int>        ..F.A...... set the number of samples per requested frame (from 1 to 65535) (default 1024)
   n                 <int>        ..F.A...... set the number of samples per requested frame (from 1 to 65535) (default 1024)
   duration          <duration>   ..F.A...... set the audio duration (default -0.000001)
   d                 <duration>   ..F.A...... set the audio duration (default -0.000001)

"""

HELP_SINE = """\
Filter sine
  Generate sine wave audio signal.
    Inputs:
        none (source filter)
    Outputs:
       #0: default (audio)
sine AVOptions:
   frequency         <double>     ..F.A...... set the sine frequency (from 0 to DBL_MAX) (default 440)
   f                 <double>     ..F.A...... set the sine frequency (from 0 to DBL_MAX) (default 440)
   beep_factor       <double>     ..F.A...... set the beep frequency factor (from 0 to DBL_MAX) (default 0)
   b                 <double>     ..F.A...... set the beep frequency factor (from 0 to DBL_MAX) (default 0)
   sample_rate       <int>        ..F.A...... set the sample rate (from 1 to INT_MAX) (default 44100)
   r                 <int>        ..F.A...... set the sample rate (from 1 to INT_MAX) (default 44100)
   duration          <duration>   ..F.A...... set the audio duration (default 0)
   d                 <duration>   ..F.A...... set the audio duration (default 0)
   samples_per_frame <string>     ..F.A...... set the number of samples per frame (default "1024")

"""

# color: the "color"/"c" option's own per-OPTION flags string ends in
# ".....T." -- but that is NOT the filter-level timeline flag (this
# module never parses per-option flags at all, see registry.py's module
# docstring). color's OWN `-filters` flag column is "..C" (no T), and its
# `-help` output correspondingly has no trailing "This filter has support
# for timeline..." sentence -- confirmed empirically, not a T-flag filter.
HELP_COLOR = """\
Filter color
  Provide an uniformly colored input.
    Inputs:
        none (source filter)
    Outputs:
       #0: default (video)
color AVOptions:
   color             <color>      ..FV.....T. set color (default "black")
   c                 <color>      ..FV.....T. set color (default "black")
   size              <image_size> ..FV....... set video size (default "320x240")
   s                 <image_size> ..FV....... set video size (default "320x240")
   rate              <video_rate> ..FV....... set video rate (default "25")
   r                 <video_rate> ..FV....... set video rate (default "25")
   duration          <duration>   ..FV....... set video duration (default -0.000001)
   d                 <duration>   ..FV....... set video duration (default -0.000001)
   sar               <rational>   ..FV....... set video sample aspect ratio (from 0 to INT_MAX) (default 1/1)

"""

# nullsrc's AVOptions header is "nullsrc/yuvtestsrc AVOptions:" -- shared
# implementation with yuvtestsrc, NOT "nullsrc AVOptions:". Same
# header-name-mismatch quirk as split/(a)split and overlay/framesync,
# now confirmed on a source too (nullsrc is |->V, single output, so it's
# a real included source, not merely excluded-and-reachable-via-private-
# -parser like the others below).
HELP_NULLSRC = """\
Filter nullsrc
  Null video source, return unprocessed video frames.
    Inputs:
        none (source filter)
    Outputs:
       #0: default (video)
nullsrc/yuvtestsrc AVOptions:
   size              <image_size> ..FV....... set video size (default "320x240")
   s                 <image_size> ..FV....... set video size (default "320x240")
   rate              <video_rate> ..FV....... set video rate (default "25")
   r                 <video_rate> ..FV....... set video rate (default "25")
   duration          <duration>   ..FV....... set video duration (default -0.000001)
   d                 <duration>   ..FV....... set video duration (default -0.000001)
   sar               <rational>   ..FV....... set video sample aspect ratio (from 0 to INT_MAX) (default 1/1)

"""

HELP_ANULLSINK = """\
Filter anullsink
  Do absolutely nothing with the input audio.
    Inputs:
       #0: default (audio)
    Outputs:
        none (sink filter)
"""

# split's AVOptions header is "(a)split AVOptions:" -- shared with asplit,
# NOT "split AVOptions:". The parser must not match on filter name.
HELP_SPLIT = """\
Filter split
  Pass on the input to N video outputs.
    Inputs:
       #0: default (video)
    Outputs:
        dynamic (depending on the options)
(a)split AVOptions:
   outputs           <int>        ..FVA...... set number of outputs (from 1 to INT_MAX) (default 2)

"""

# Trimmed real capture of `-help filter=scale`: its own AVOptions section in
# full (including the w/width, h/height alias pairs where the SHORT name
# comes first in the file -- unlike testsrc/subtitles where the long name
# comes first), plus the start of the following "SWScaler AVOptions:"
# section (2-space-indented, "-"-prefixed option names) to prove parsing
# stops at the blank line and never reaches it.
HELP_SCALE_TRIMMED = """\
Filter scale
  Scale the input video size and/or convert the image format.
    Inputs:
       #0: default (video)
        dynamic (depending on the options)
    Outputs:
       #0: default (video)
scale AVOptions:
   w                 <string>     ..FV.....T. Output video width
   width             <string>     ..FV.....T. Output video width
   h                 <string>     ..FV.....T. Output video height
   height            <string>     ..FV.....T. Output video height
   flags             <string>     ..FV....... Flags to pass to libswscale (default "")
   interl            <boolean>    ..FV....... set interlacing (default false)
   size              <string>     ..FV....... set video size
   s                 <string>     ..FV....... set video size
   in_color_matrix   <int>        ..FV....... set input YCbCr type (from -1 to 17) (default auto)
     auto            -1           ..FV.......
     bt601           5            ..FV.......
     bt470           5            ..FV.......
     smpte170m       5            ..FV.......
     bt709           1            ..FV.......
     fcc             4            ..FV.......
     smpte240m       7            ..FV.......
     bt2020          9            ..FV.......
   force_original_aspect_ratio <int>        ..FV....... decrease or increase w/h if necessary to keep the original AR (from 0 to 2) (default disable)
     disable         0            ..FV.......
     decrease        1            ..FV.......
     increase        2            ..FV.......

SWScaler AVOptions:
  -sws_flags         <flags>      E..V....... scaler flags (default bicubic)
     fast_bilinear                E..V....... fast bilinear
     bilinear                     E..V....... bilinear
"""


# --- test plumbing -----------------------------------------------------


@pytest.fixture(autouse=True)
def _isolated_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    # Redirect the disk cache into a per-test tmp dir so tests never touch
    # (or are polluted by) the real ~/.cache/ffrwd.
    monkeypatch.setattr(registry_mod, "_cache_dir", lambda: tmp_path / "cache")
    clear_cache()
    yield
    clear_cache()


def _fake_ffmpeg_present(monkeypatch: pytest.MonkeyPatch, path: str = "/usr/bin/ffmpeg") -> None:
    monkeypatch.setattr(registry_mod.binaries, "ffmpeg_path", lambda: path)


def _fake_run(
    monkeypatch: pytest.MonkeyPatch,
    filters_out: str = FILTERS_SNIPPET,
    version_out: str = VERSION_LINE,
    help_map: dict[str, str] | None = None,
    version_returncode: int = 0,
    filters_returncode: int = 0,
) -> list[list[str]]:
    calls: list[list[str]] = []
    help_map = help_map or {}

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        if "-version" in argv:
            return subprocess.CompletedProcess(argv, version_returncode, stdout=version_out, stderr="")
        if "-filters" in argv:
            return subprocess.CompletedProcess(argv, filters_returncode, stdout=filters_out, stderr="")
        for arg in argv:
            if arg.startswith("filter="):
                name = arg[len("filter=") :]
                out = help_map.get(name, "")
                return subprocess.CompletedProcess(argv, 0, stdout=out, stderr="")
        return subprocess.CompletedProcess(argv, 1, stdout="", stderr="")

    monkeypatch.setattr(registry_mod.subprocess, "run", fake_run)
    return calls


def _loaded_registry(monkeypatch: pytest.MonkeyPatch, **kwargs: object) -> registry_mod.Registry:
    _fake_ffmpeg_present(monkeypatch)
    _fake_run(monkeypatch, **kwargs)  # type: ignore[arg-type]
    return load()


# --- -filters parsing: included pad signatures --------------------------


def test_v_to_v_signature(monkeypatch: pytest.MonkeyPatch) -> None:
    reg = _loaded_registry(monkeypatch)
    assert reg.get("gblur") == DynamicFilter(
        name="gblur",
        inputs=("video",),
        output="video",
        doc="Apply Gaussian Blur filter.",
        timeline=True,  # FILTERS_SNIPPET's gblur line is "TSC"
    )


def test_aa_to_a_signature(monkeypatch: pytest.MonkeyPatch) -> None:
    reg = _loaded_registry(monkeypatch)
    f = reg.get("aap")
    assert f is not None
    assert f.inputs == ("audio", "audio")
    assert f.output == "audio"


def test_vv_to_v_signature(monkeypatch: pytest.MonkeyPatch) -> None:
    reg = _loaded_registry(monkeypatch)
    f = reg.get("overlay")
    assert f is not None
    assert f.inputs == ("video", "video")
    assert f.output == "video"


def test_vvv_and_vvvv_signatures(monkeypatch: pytest.MonkeyPatch) -> None:
    reg = _loaded_registry(monkeypatch)
    colormap = reg.get("colormap")
    threshold = reg.get("threshold")
    assert colormap is not None and colormap.inputs == ("video", "video", "video")
    assert threshold is not None and threshold.inputs == ("video",) * 4


def test_mixed_a_to_v_signature(monkeypatch: pytest.MonkeyPatch) -> None:
    reg = _loaded_registry(monkeypatch)
    f = reg.get("a3dscope")
    assert f is not None
    assert f.inputs == ("audio",)
    assert f.output == "video"


# --- -filters parsing: exclusions ---------------------------------------


def test_excluded_specs_absent_from_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    reg = _loaded_registry(monkeypatch)
    for name in EXCLUDED_NAMES:
        assert reg.get(name) is None, name
        assert name not in reg.names(), name


def test_names_returns_exactly_the_included_set(monkeypatch: pytest.MonkeyPatch) -> None:
    reg = _loaded_registry(monkeypatch)
    assert set(reg.names()) == INCLUDED_NAMES


# --- -filters parsing: sources --------------------------------------------


def test_source_names_returns_exactly_the_single_output_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reg = _loaded_registry(monkeypatch)
    assert set(reg.source_names()) == SOURCE_NAMES
    # None of the sources leak into the regular-filter set (get()/names()).
    assert SOURCE_NAMES.isdisjoint(set(reg.names()))


def test_get_source_video_output(monkeypatch: pytest.MonkeyPatch) -> None:
    reg = _loaded_registry(monkeypatch)
    f = reg.get_source("testsrc")
    assert f == SourceFilter(name="testsrc", output="video", doc="Generate test pattern.")


def test_get_source_audio_output(monkeypatch: pytest.MonkeyPatch) -> None:
    reg = _loaded_registry(monkeypatch)
    f = reg.get_source("anullsrc")
    assert f is not None
    assert f.output == "audio"
    assert f.doc == "Null audio source, return empty audio frames."


def test_get_source_unknown_name_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    reg = _loaded_registry(monkeypatch)
    assert reg.get_source("nope_not_a_filter") is None


def test_get_source_excludes_multi_output_and_dynamic(monkeypatch: pytest.MonkeyPatch) -> None:
    # avsynctest (|->AV, 2-char output) and amovie/movie (|->N, dynamic) are
    # source-shaped but excluded, same v1 scope check as regular filters.
    reg = _loaded_registry(monkeypatch)
    assert reg.get_source("avsynctest") is None
    assert reg.get_source("amovie") is None
    assert reg.get_source("movie") is None
    for name in ("avsynctest", "amovie", "movie"):
        assert name not in reg.source_names()


def test_get_source_excludes_sinks(monkeypatch: pytest.MonkeyPatch) -> None:
    reg = _loaded_registry(monkeypatch)
    assert reg.get_source("anullsink") is None
    assert reg.get_source("nullsink") is None


def test_get_still_none_for_source_names_unchanged_column_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # get() (column-function lookup) is UNCHANGED for
    # sources -- a source is not callable as a column function.
    reg = _loaded_registry(monkeypatch)
    for name in SOURCE_NAMES:
        assert reg.get(name) is None, name


# --- -filters parsing: timeline (T) flag ----------------------------------


def test_timeline_flag_true_for_t_prefixed_filters(monkeypatch: pytest.MonkeyPatch) -> None:
    reg = _loaded_registry(monkeypatch)
    # FILTERS_SNIPPET flag columns: "TSC gblur", "TSC overlay", "TSC hqdn3d",
    # "TSC colormap", "TSC threshold", "TSC aap" -- all T-prefixed.
    for name in ("gblur", "overlay", "hqdn3d", "colormap", "threshold", "aap"):
        f = reg.get(name)
        assert f is not None, name
        assert f.timeline is True, name


def test_timeline_flag_false_for_non_t_filters(monkeypatch: pytest.MonkeyPatch) -> None:
    reg = _loaded_registry(monkeypatch)
    # FILTERS_SNIPPET flag columns: "..C scale", "... subtitles",
    # "... aecho", "..C a3dscope", ".S. xfade" -- none T-prefixed. xfade in
    # particular has the S flag set but NOT T, proving the parser checks
    # only the first character, not "any flag set".
    for name in ("scale", "subtitles", "aecho", "a3dscope", "xfade"):
        f = reg.get(name)
        assert f is not None, name
        assert f.timeline is False, name


# --- -help option parsing: numeric range + default (gblur) --------------


def test_gblur_options_numeric_with_range_and_default(monkeypatch: pytest.MonkeyPatch) -> None:
    reg = _loaded_registry(monkeypatch, help_map={"gblur": HELP_GBLUR})
    opts = reg.options("gblur")
    assert opts is not None
    sigma = opts["sigma"]
    assert sigma.type == "num"
    assert sigma.minimum == 0.0
    assert sigma.maximum == 1024.0
    assert sigma.default == "0.5"
    assert sigma.constants == ()
    assert sigma.unusable is False

    sigma_v = opts["sigmaV"]
    assert sigma_v.minimum == -1.0
    assert sigma_v.maximum == 1024.0
    assert sigma_v.default == "-1"

    steps = opts["steps"]
    assert steps.type == "num"
    assert steps.minimum == 1.0
    assert steps.maximum == 6.0


# --- -help option parsing: enum constants + name-valued default (xfade) --


def test_xfade_enum_constants_and_name_valued_default(monkeypatch: pytest.MonkeyPatch) -> None:
    reg = _loaded_registry(monkeypatch, help_map={"xfade": HELP_XFADE})
    opts = reg.options("xfade")
    assert opts is not None
    transition = opts["transition"]
    assert transition.type == "str"  # enum -> str per RFC type table
    assert transition.default == "fade"  # verbatim constant name, not "0"
    assert len(transition.constants) == 59  # custom(-1) .. revealdown(57) inclusive
    assert transition.constants[0] == "custom"
    assert transition.constants[-1] == "revealdown"
    assert "fade" in transition.constants

    assert opts["duration"].type == "duration"  # <duration>
    assert opts["offset"].type == "duration"  # <duration>
    assert opts["expr"].type == "str"  # <string>


# --- -help option parsing: multi-section stop-at-blank-line (overlay) ---


def test_overlay_stops_before_framesync_section(monkeypatch: pytest.MonkeyPatch) -> None:
    reg = _loaded_registry(monkeypatch, help_map={"overlay": HELP_OVERLAY})
    opts = reg.options("overlay")
    assert opts is not None
    # Options unique to overlay's own AVOptions section.
    for name in ("x", "y", "eof_action", "eval", "shortest", "format", "repeatlast", "alpha"):
        assert name in opts, name
    # ts_sync_mode ONLY appears in the later "framesync AVOptions:" section,
    # separated by a blank line -- must NOT be parsed.
    assert "ts_sync_mode" not in opts

    fmt = opts["format"]
    assert fmt.type == "str"
    assert len(fmt.constants) == 9
    assert fmt.default == "yuv420"

    eof_action = opts["eof_action"]
    assert eof_action.constants == ("repeat", "endall", "pass")
    assert eof_action.doc == "Action to take when encountering EOF from secondary input"

    assert opts["shortest"].type == "bool"


# --- -help option parsing: double type + unparseable DBL_MAX range ------


def test_hqdn3d_double_type_and_unparseable_dbl_max_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    reg = _loaded_registry(monkeypatch, help_map={"hqdn3d": HELP_HQDN3D})
    opts = reg.options("hqdn3d")
    assert opts is not None
    luma = opts["luma_spatial"]
    assert luma.type == "num"
    assert luma.minimum == 0.0
    assert luma.maximum is None  # "DBL_MAX" does not parse as a float
    assert luma.default == "0"


# --- -help option parsing: string type for delays/decays (aecho) --------


def test_aecho_string_type_for_delays_and_decays(monkeypatch: pytest.MonkeyPatch) -> None:
    reg = _loaded_registry(monkeypatch, help_map={"aecho": HELP_AECHO})
    opts = reg.options("aecho")
    assert opts is not None
    assert opts["delays"].type == "str"
    assert opts["delays"].default == '"1000"'
    assert opts["decays"].type == "str"
    assert opts["decays"].default == '"0.5"'
    assert opts["in_gain"].type == "num"
    assert opts["in_gain"].minimum == 0.0
    assert opts["in_gain"].maximum == 1.0


# --- -help option parsing: sources through Registry.options() -----------
# Sources use the SAME lazy Registry.options() path as regular filters
# Registry.options(name) must work for sources too.


def test_testsrc_options_via_registry_alias_dedup(monkeypatch: pytest.MonkeyPatch) -> None:
    reg = _loaded_registry(monkeypatch, help_map={"testsrc": HELP_TESTSRC})
    opts = reg.options("testsrc")
    assert opts is not None
    assert "size" in opts and "s" not in opts
    assert "rate" in opts and "r" not in opts
    assert "duration" in opts and "d" not in opts
    assert opts["sar"].type == "num"


def test_anullsrc_options_via_registry_channel_layout_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reg = _loaded_registry(monkeypatch, help_map={"anullsrc": HELP_ANULLSRC})
    opts = reg.options("anullsrc")
    assert opts is not None
    assert "channel_layout" in opts and "cl" not in opts
    assert opts["channel_layout"].type == "str"  # <channel_layout> -- permissive fallback
    assert "sample_rate" in opts and "r" not in opts
    assert opts["sample_rate"].type == "num"


def test_sine_options_via_registry_double_type(monkeypatch: pytest.MonkeyPatch) -> None:
    reg = _loaded_registry(monkeypatch, help_map={"sine": HELP_SINE})
    opts = reg.options("sine")
    assert opts is not None
    assert "frequency" in opts and "f" not in opts
    assert opts["frequency"].type == "num"
    assert opts["frequency"].minimum == 0.0
    assert opts["frequency"].maximum is None  # DBL_MAX unparseable
    assert opts["samples_per_frame"].type == "str"


def test_color_options_via_registry_color_type(monkeypatch: pytest.MonkeyPatch) -> None:
    reg = _loaded_registry(monkeypatch, help_map={"color": HELP_COLOR})
    opts = reg.options("color")
    assert opts is not None
    assert "color" in opts and "c" not in opts
    assert opts["color"].type == "str"  # <color> -- permissive fallback
    assert opts["color"].default == '"black"'


def test_nullsrc_options_via_registry_header_name_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # nullsrc's AVOptions header reads "nullsrc/yuvtestsrc AVOptions:" --
    # must still parse via Registry.options(), not just the private helper.
    reg = _loaded_registry(monkeypatch, help_map={"nullsrc": HELP_NULLSRC})
    opts = reg.options("nullsrc")
    assert opts is not None
    assert "size" in opts and "s" not in opts
    assert opts["sar"].type == "num"


def test_source_options_lazy_and_memoized_like_regular_filters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _fake_run(monkeypatch, help_map={"testsrc": HELP_TESTSRC})
    _fake_ffmpeg_present(monkeypatch)
    reg = load()

    def help_calls_for(name: str) -> int:
        return sum(1 for c in calls if f"filter={name}" in c)

    assert help_calls_for("testsrc") == 0
    reg.options("testsrc")
    reg.options("testsrc")  # memoized
    assert help_calls_for("testsrc") == 1


# --- alias dedup / header quirks, tested via the private help parser ----
# split/anullsink are excluded (dynamic-pad regular filter / sink), so
# their -help output is only reachable directly, not through
# Registry.options(), which only serves known/included names. testsrc is
# ALSO exercised directly here, pinning the private parser's behavior in
# isolation --
# it is additionally covered through the public Registry.options() path
# above (test_testsrc_options_via_registry_alias_dedup).


def test_testsrc_alias_dedup_keeps_longer_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(registry_mod, "_run", lambda argv: HELP_TESTSRC)
    opts = registry_mod._parse_filter_help("ffmpeg", "testsrc")
    assert "size" in opts and "s" not in opts
    assert "rate" in opts and "r" not in opts
    assert "duration" in opts and "d" not in opts
    assert "decimals" in opts and "n" not in opts
    assert opts["sar"].type == "num"
    assert opts["sar"].minimum == 0.0
    assert opts["sar"].maximum is None  # INT_MAX unparseable
    assert opts["sar"].default == "1/1"


def test_split_header_name_mismatch_still_parses(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(registry_mod, "_run", lambda argv: HELP_SPLIT)
    opts = registry_mod._parse_filter_help("ffmpeg", "split")
    assert "outputs" in opts
    outputs = opts["outputs"]
    assert outputs.type == "num"
    assert outputs.minimum == 1.0
    assert outputs.maximum is None  # INT_MAX unparseable
    assert outputs.default == "2"


def test_anullsink_no_avoptions_section_is_empty_not_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(registry_mod, "_run", lambda argv: HELP_ANULLSINK)
    opts = registry_mod._parse_filter_help("ffmpeg", "anullsink")
    assert opts == {}


def test_scale_alias_dedup_regardless_of_file_order(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(registry_mod, "_run", lambda argv: HELP_SCALE_TRIMMED)
    opts = registry_mod._parse_filter_help("ffmpeg", "scale")
    # w/width and h/height: the SHORT name comes FIRST in the real capture,
    # yet the longer name must still be the one kept.
    assert "width" in opts and "w" not in opts
    assert "height" in opts and "h" not in opts
    # size/s: long name also comes first here -- same outcome either way.
    assert "size" in opts and "s" not in opts
    # The SWScaler section (different indent, "-"-prefixed names) must not
    # leak into scale's own options.
    assert "sws_flags" not in opts
    assert not any(name.startswith("-") for name in opts)

    in_color_matrix = opts["in_color_matrix"]
    assert in_color_matrix.type == "str"
    assert in_color_matrix.constants == (
        "auto", "bt601", "bt470", "smpte170m", "bt709", "fcc", "smpte240m", "bt2020",
    )
    assert in_color_matrix.default == "auto"  # symbolic, not "-1"
    # The kept names are in ffmpeg's own declared order, which IS its
    # positional binding order: `scale=640:480` binds w,h.
    assert list(opts)[:3] == ["width", "height", "flags"]


# --- dedup is ADJACENCY-scoped ---------------------------------
#
# ffmpeg's positional binding (`process_options`) skips a duplicate AVOption
# only when it sits IMMEDIATELY after the entry it aliases, so two same-doc
# options that are not adjacent are two REAL options that each own a
# positional slot. Modelled on cropdetect, where `reset` (deprecated alias of
# reset_count) and `reset_count` share doc text but are separated by `skip`:
# ffmpeg 7.1 binds `cropdetect=24:16:-9` to 'reset' and
# `cropdetect=24:16:0:-9' to 'skip' (measured 2026-08-16).

HELP_CROPDETECT_TRIMMED = """\
Filter cropdetect
  Auto-detect crop size.
cropdetect AVOptions:
   limit             <float>      ..FV....... Threshold below which pixels are considered black (from 0 to 65535) (default 0.0941176)
   round             <int>        ..FV....... Value by which the height/width should be divisible (from 0 to INT_MAX) (default 16)
   reset             <int>        ..FV....... Recalculate the crop area after this many frames (from 0 to INT_MAX) (default 0)
   skip              <int>        ..FV....... Number of initial frames to skip (from 0 to INT_MAX) (default 2)
   reset_count       <int>        ..FV....... Recalculate the crop area after this many frames (from 0 to INT_MAX) (default 0)
   max_outliers      <int>        ..FV....... Threshold count of outliers (from 0 to INT_MAX) (default 0)

"""

# deshake declares x,y,w,h and only THEN rx,ry -- x/rx and y/ry share doc
# text but are four lines apart. The pre-051 global grouping dropped x and y
# outright (they are shorter) and made slot 1 `w`, where ffmpeg binds `x`.
HELP_DESHAKE_TRIMMED = """\
Filter deshake
  Stabilize shaky video.
deshake AVOptions:
   x                 <int>        ..FV....... set x for the rectangular search area (from -1 to INT_MAX) (default -1)
   y                 <int>        ..FV....... set y for the rectangular search area (from -1 to INT_MAX) (default -1)
   w                 <int>        ..FV....... set width for the rectangular search area (from -1 to INT_MAX) (default -1)
   h                 <int>        ..FV....... set height for the rectangular search area (from -1 to INT_MAX) (default -1)
   rx                <int>        ..FV....... set x for the rectangular search area (from 0 to 64) (default 16)
   ry                <int>        ..FV....... set y for the rectangular search area (from 0 to 64) (default 16)

"""


def test_non_adjacent_same_doc_options_are_both_kept(monkeypatch: pytest.MonkeyPatch) -> None:
    """`reset` and `reset_count` share doc text but are not adjacent."""
    monkeypatch.setattr(registry_mod, "_run", lambda argv: HELP_CROPDETECT_TRIMMED)
    opts = registry_mod._parse_filter_help("ffmpeg", "cropdetect")
    assert "reset" in opts
    assert "reset_count" in opts
    assert list(opts) == [
        "limit", "round", "reset", "skip", "reset_count", "max_outliers",
    ]


def test_non_adjacent_same_doc_options_keep_their_positional_slots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """deshake's x/rx and y/ry are four lines apart: all four survive, in order."""
    monkeypatch.setattr(registry_mod, "_run", lambda argv: HELP_DESHAKE_TRIMMED)
    opts = registry_mod._parse_filter_help("ffmpeg", "deshake")
    assert list(opts) == ["x", "y", "w", "h", "rx", "ry"]
    # x and rx are genuinely different options, with different ranges --
    # merging them (pre-051) lost the one ffmpeg binds positionally first.
    assert opts["x"].minimum == -1.0
    assert opts["rx"].minimum == 0.0
    assert opts["rx"].maximum == 64.0


def test_adjacent_alias_run_of_three_keeps_the_longest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A run longer than two collapses to one entry at the run's position."""
    help_text = (
        "Filter x\nx AVOptions:\n"
        "   a                 <int>        ..FV....... first option (default 1)\n"
        "   aa                <int>        ..FV....... shared doc (default 1)\n"
        "   aaaa              <int>        ..FV....... shared doc (default 1)\n"
        "   aaa               <int>        ..FV....... shared doc (default 1)\n"
        "   z                 <int>        ..FV....... last option (default 1)\n"
        "\n"
    )
    monkeypatch.setattr(registry_mod, "_run", lambda argv: help_text)
    opts = registry_mod._parse_filter_help("ffmpeg", "x")
    assert list(opts) == ["a", "aaaa", "z"]


def test_adjacent_alias_run_ties_break_by_first_occurrence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    help_text = (
        "Filter x\nx AVOptions:\n"
        "   bb                <int>        ..FV....... shared doc (default 1)\n"
        "   cc                <int>        ..FV....... shared doc (default 1)\n"
        "\n"
    )
    monkeypatch.setattr(registry_mod, "_run", lambda argv: help_text)
    opts = registry_mod._parse_filter_help("ffmpeg", "x")
    assert list(opts) == ["bb"]


# --- a trailing-letter twin never shadows the documented name ---------------
#
# trim/atrim help text as ffmpeg 9.0.1 prints it (the build the reference
# registry pins), cut to the AVOptions block: `start`/`starti` and friends
# are adjacent same-doc pairs where the longer name is not the documented one.

HELP_TRIM_TRIMMED = """\
Filter trim
  Pick one continuous section from the input, drop the rest.
    Inputs:
       #0: default (video)
    Outputs:
       #0: default (video)
trim AVOptions:
   start             <duration>   ..FV....... Timestamp of the first frame that should be passed (default INT64_MAX)
   starti            <duration>   ..FV....... Timestamp of the first frame that should be passed (default INT64_MAX)
   end               <duration>   ..FV....... Timestamp of the first frame that should be dropped again (default INT64_MAX)
   endi              <duration>   ..FV....... Timestamp of the first frame that should be dropped again (default INT64_MAX)
   start_pts         <int64>      ..FV....... Timestamp of the first frame that should be  passed (from I64_MIN to I64_MAX) (default I64_MIN)
   end_pts           <int64>      ..FV....... Timestamp of the first frame that should be dropped again (from I64_MIN to I64_MAX) (default I64_MIN)
   duration          <duration>   ..FV....... Maximum duration of the output (default 0)
   durationi         <duration>   ..FV....... Maximum duration of the output (default 0)
   start_frame       <int64>      ..FV....... Number of the first frame that should be passed to the output (from -1 to I64_MAX) (default -1)
   end_frame         <int64>      ..FV....... Number of the first frame that should be dropped again (from 0 to I64_MAX) (default I64_MAX)

"""

HELP_ATRIM_TRIMMED = """\
Filter atrim
  Pick one continuous section from the input, drop the rest.
    Inputs:
       #0: default (audio)
    Outputs:
       #0: default (audio)
atrim AVOptions:
   start             <duration>   ..F.A...... Timestamp of the first frame that should be passed (default INT64_MAX)
   starti            <duration>   ..F.A...... Timestamp of the first frame that should be passed (default INT64_MAX)
   end               <duration>   ..F.A...... Timestamp of the first frame that should be dropped again (default INT64_MAX)
   endi              <duration>   ..F.A...... Timestamp of the first frame that should be dropped again (default INT64_MAX)
   start_pts         <int64>      ..F.A...... Timestamp of the first frame that should be  passed (from I64_MIN to I64_MAX) (default I64_MIN)
   end_pts           <int64>      ..F.A...... Timestamp of the first frame that should be dropped again (from I64_MIN to I64_MAX) (default I64_MIN)
   duration          <duration>   ..F.A...... Maximum duration of the output (default 0)
   durationi         <duration>   ..F.A...... Maximum duration of the output (default 0)
   start_sample      <int64>      ..F.A...... Number of the first audio sample that should be passed to the output (from -1 to I64_MAX) (default -1)
   end_sample        <int64>      ..F.A...... Number of the first audio sample that should be dropped again (from 0 to I64_MAX) (default I64_MAX)

"""


def test_trim_alias_dedup_keeps_documented_name_over_i_suffixed_twin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(registry_mod, "_run", lambda argv: HELP_TRIM_TRIMMED)
    opts = registry_mod._parse_filter_help("ffmpeg", "trim")
    assert "start" in opts and "starti" not in opts
    assert "end" in opts and "endi" not in opts
    assert "duration" in opts and "durationi" not in opts
    # start_pts/end_pts repeat end's doc text but are not adjacent to it.
    assert "start_pts" in opts
    assert "end_pts" in opts
    assert list(opts) == [
        "start", "end", "start_pts", "end_pts", "duration",
        "start_frame", "end_frame",
    ]


def test_atrim_alias_dedup_keeps_documented_name_over_i_suffixed_twin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(registry_mod, "_run", lambda argv: HELP_ATRIM_TRIMMED)
    opts = registry_mod._parse_filter_help("ffmpeg", "atrim")
    assert "start" in opts and "starti" not in opts
    assert "end" in opts and "endi" not in opts
    assert "duration" in opts and "durationi" not in opts
    assert list(opts) == [
        "start", "end", "start_pts", "end_pts", "duration",
        "start_sample", "end_sample",
    ]


def test_trailing_letter_alias_does_not_shadow_a_genuinely_longer_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`width` is `w` plus a whole word, not one trailing character -- still wins."""
    help_text = (
        "Filter x\nx AVOptions:\n"
        "   w                 <int>        ..FV....... shared doc (default 1)\n"
        "   width             <int>        ..FV....... shared doc (default 1)\n"
        "\n"
    )
    monkeypatch.setattr(registry_mod, "_run", lambda argv: help_text)
    opts = registry_mod._parse_filter_help("ffmpeg", "x")
    assert list(opts) == ["width"]


def test_empty_doc_options_are_not_merged_across_the_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """buffer's options have EMPTY doc text; only adjacent ones may collapse."""
    help_text = (
        "Filter buffer\nbuffer AVOptions:\n"
        "   width             <int>        ..FV.......\n"
        "   video_size        <image_size> ..FV.......\n"
        "   pix_fmt           <pix_fmt>    ..FV.......\n"
        "\n"
    )
    monkeypatch.setattr(registry_mod, "_run", lambda argv: help_text)
    opts = registry_mod._parse_filter_help("ffmpeg", "buffer")
    assert list(opts) == ["width", "video_size", "pix_fmt"]


# --- degradation paths ----------------------------------------------------


def test_missing_ffmpeg_registry_is_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(registry_mod.binaries, "ffmpeg_path", lambda: None)
    reg = load()
    assert reg.available() is False
    assert reg.names() == []
    assert reg.get("gblur") is None
    assert reg.options("gblur") is None


def test_garbage_filters_output_degrades_to_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    reg = _loaded_registry(monkeypatch, filters_out="not filter output\nrandom garbage\n")
    assert reg.available() is False
    assert reg.names() == []


def test_ffmpeg_version_call_failure_degrades_to_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    reg = _loaded_registry(monkeypatch, version_returncode=1)
    assert reg.available() is False


def test_filters_call_failure_degrades_to_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    reg = _loaded_registry(monkeypatch, filters_returncode=1)
    assert reg.available() is False


def test_timeout_degrades_to_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_ffmpeg_present(monkeypatch)

    def raise_timeout(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(cmd=argv, timeout=10)

    monkeypatch.setattr(registry_mod.subprocess, "run", raise_timeout)
    reg = load()
    assert reg.available() is False


def test_unparseable_option_line_degrades_to_str_type(monkeypatch: pytest.MonkeyPatch) -> None:
    # CONSTRUCTED: not a captured line -- an option-shaped (3-space indent)
    # line whose type token isn't wrapped in <angle brackets>, which real
    # ffmpeg never emits but the RFC requires we tolerate permissively.
    weird_help = (
        "Filter weirdfilter\n"
        "weirdfilter AVOptions:\n"
        "   odd_opt           some-weird-shape     ..FV....... "
        "an option that doesn't match the normal shape\n"
        "\n"
    )
    monkeypatch.setattr(registry_mod, "_run", lambda argv: weird_help)
    opts = registry_mod._parse_filter_help("ffmpeg", "weirdfilter")
    assert "odd_opt" in opts
    odd = opts["odd_opt"]
    assert odd.type == "str"
    assert "some-weird-shape" in odd.doc


def test_binary_and_dictionary_types_marked_unusable(monkeypatch: pytest.MonkeyPatch) -> None:
    # CONSTRUCTED: no real filter in a full scan of ffmpeg 7.1's included
    # filters used <binary> or <dictionary>; this exercises the RFC's
    # exclusion rule for those AVOption types regardless.
    synthetic_help = (
        "Filter syntheticfilter\n"
        "syntheticfilter AVOptions:\n"
        "   blob              <binary>     ..FV....... opaque binary blob option\n"
        "   meta              <dictionary> ..FV....... key=value dictionary option\n"
        "   ok                <int>        ..FV....... normal numeric option (from 0 to 10) (default 5)\n"
        "\n"
    )
    monkeypatch.setattr(registry_mod, "_run", lambda argv: synthetic_help)
    opts = registry_mod._parse_filter_help("ffmpeg", "syntheticfilter")
    assert opts["blob"].unusable is True
    assert opts["blob"].type == "str"
    assert opts["meta"].unusable is True
    assert opts["ok"].unusable is False
    assert opts["ok"].type == "num"


# --- type mapping table (RFC "Type mapping and validation") -------------


def test_every_ffmpeg_option_type_maps_to_its_ffrwd_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # channel_layout and pix_fmt are outside the documented table -- the
    # permissive fallback lands them on "str".
    expected = {
        "int": "num",
        "int64": "num",
        "float": "num",
        "double": "num",
        "rational": "num",
        "boolean": "bool",
        "string": "str",
        "color": "str",
        "duration": "duration",
        "image_size": "str",
        "video_rate": "str",
        "flags": "str",
        "channel_layout": "str",
        "pix_fmt": "str",
    }
    # CONSTRUCTED option block covering every type-table row at once (only a
    # subset -- int/float/double/boolean/string/duration/image_size/
    # video_rate/rational -- were captured from real filters above).
    help_text = "Filter t\nt AVOptions:\n" + "".join(
        f"   {ffmpeg_type}               <{ffmpeg_type}>      ..FV....... an option of"
        f" type {ffmpeg_type}\n"
        for ffmpeg_type in expected
    ) + "\n"
    monkeypatch.setattr(registry_mod, "_run", lambda argv: help_text)
    opts = registry_mod._parse_filter_help("ffmpeg", "t")
    assert {name: opt.type for name, opt in opts.items()} == expected


# --- caching: process memo -------------------------------------------------


def test_filters_and_version_fetched_once_per_process(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_ffmpeg_present(monkeypatch)
    calls = _fake_run(monkeypatch)
    reg = load()
    reg.available()
    reg.names()
    reg.get("gblur")
    assert sum(1 for c in calls if "-filters" in c) == 1
    assert sum(1 for c in calls if "-version" in c) == 1


def test_options_fetched_lazily_and_memoized_per_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _fake_run(monkeypatch, help_map={"gblur": HELP_GBLUR, "xfade": HELP_XFADE})
    _fake_ffmpeg_present(monkeypatch)
    reg = load()

    def help_calls_for(name: str) -> int:
        return sum(1 for c in calls if f"filter={name}" in c)

    assert help_calls_for("gblur") == 0  # not referenced yet
    reg.options("gblur")
    reg.options("gblur")  # second call: memoized, no new subprocess call
    assert help_calls_for("gblur") == 1
    assert help_calls_for("xfade") == 0  # never referenced -> never fetched

    reg.options("xfade")
    assert help_calls_for("xfade") == 1
    assert help_calls_for("gblur") == 1  # unaffected


# --- caching: disk persistence across process (Registry instance) restart -


def test_disk_cache_survives_new_registry_instance(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _fake_run(monkeypatch, help_map={"gblur": HELP_GBLUR})
    _fake_ffmpeg_present(monkeypatch)

    reg1 = load()
    reg1.options("gblur")
    assert sum(1 for c in calls if "-filters" in c) == 1
    assert sum(1 for c in calls if "filter=gblur" in " ".join(c)) == 1

    # Simulate a fresh process: drop only the in-memory singleton, keep the
    # on-disk cache file (unlike clear_cache(), which removes both).
    registry_mod._registry = None
    reg2 = load()
    assert reg2 is not reg1

    f = reg2.get("gblur")
    assert f is not None and f.inputs == ("video",)
    opts = reg2.options("gblur")
    assert opts is not None and opts["sigma"].minimum == 0.0

    # -filters and -help must NOT be called again -- only -version is
    # (needed every process start to know which cache file to key on).
    assert sum(1 for c in calls if "-filters" in c) == 1
    assert sum(1 for c in calls if "filter=gblur" in " ".join(c)) == 1
    assert sum(1 for c in calls if "-version" in c) == 2


def test_corrupt_json_cache_file_rebuilds_silently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_ffmpeg_present(monkeypatch)
    calls = _fake_run(monkeypatch)
    cache_path = registry_mod._cache_path(VERSION_LINE)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text("{not valid json", encoding="utf-8")

    reg = load()
    assert reg.available() is True
    assert sum(1 for c in calls if "-filters" in c) == 1  # rebuilt, not read


def test_wrong_shape_cache_file_rebuilds_silently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_ffmpeg_present(monkeypatch)
    calls = _fake_run(monkeypatch)
    cache_path = registry_mod._cache_path(VERSION_LINE)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({"version_line": VERSION_LINE}), encoding="utf-8")

    reg = load()
    assert reg.available() is True
    assert sum(1 for c in calls if "-filters" in c) == 1


def test_pre_040_cache_shape_rebuilds_silently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A plausible PRE-040 cache payload: valid JSON, right top-level shape,
    # but missing "format_version" (didn't exist yet) and "sources"
    # (didn't exist yet), and filter entries missing "timeline" (didn't
    # exist yet). This is the exact shape _write_disk_cache produced
    # before this plan. Must rebuild from -filters/-help, not raise and
    # not silently return stale/incomplete data.
    _fake_ffmpeg_present(monkeypatch)
    calls = _fake_run(monkeypatch, help_map={"gblur": HELP_GBLUR})
    cache_path = registry_mod._cache_path(VERSION_LINE)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    old_shape = {
        "version_line": VERSION_LINE,
        "filters": {
            "gblur": {"inputs": ["video"], "output": "video", "doc": "Apply Gaussian Blur filter."}
        },
        "options": {},
    }
    cache_path.write_text(json.dumps(old_shape), encoding="utf-8")

    reg = load()
    assert reg.available() is True
    # Rebuilt via a fresh -filters call, not read from the stale-shape file.
    assert sum(1 for c in calls if "-filters" in c) == 1
    f = reg.get("gblur")
    assert f is not None
    assert f.timeline is True  # only present because it was freshly parsed
    opts = reg.options("gblur")
    assert opts is not None and "sigma" in opts
    # The rebuilt cache file is now current-format and includes sources.
    assert set(reg.source_names()) == SOURCE_NAMES


def test_clear_cache_removes_disk_file_and_forces_refetch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_ffmpeg_present(monkeypatch)
    calls = _fake_run(monkeypatch)
    load().available()
    cache_path = registry_mod._cache_path(VERSION_LINE)
    assert cache_path.exists()

    clear_cache()
    assert not cache_path.exists()

    load().available()
    assert sum(1 for c in calls if "-filters" in c) == 2


# --- subprocess hygiene: argv shape --------------------------------------


def test_filters_argv_is_a_list_with_expected_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_ffmpeg_present(monkeypatch)
    calls = _fake_run(monkeypatch)
    load().available()
    filters_calls = [c for c in calls if "-filters" in c]
    assert len(filters_calls) == 1
    argv = filters_calls[0]
    assert isinstance(argv, list)
    assert "-hide_banner" in argv
    assert argv[0] == "/usr/bin/ffmpeg"


def test_help_argv_uses_filter_equals_name(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_ffmpeg_present(monkeypatch)
    calls = _fake_run(monkeypatch, help_map={"gblur": HELP_GBLUR})
    reg = load()
    reg.options("gblur")
    help_calls = [c for c in calls if any(a.startswith("filter=") for a in c)]
    assert len(help_calls) == 1
    argv = help_calls[0]
    assert "-hide_banner" in argv
    assert "-help" in argv
    assert "filter=gblur" in argv


# --- FilterOption / DynamicFilter dataclass shape ------------------------


def test_filter_option_is_frozen() -> None:
    opt = FilterOption(
        name="x", type="num", doc="d", minimum=0.0, maximum=1.0, default="0", constants=()
    )
    assert opt.unusable is False
    with pytest.raises(Exception):
        opt.name = "y"  # type: ignore[misc]


def test_dynamic_filter_is_frozen() -> None:
    f = DynamicFilter(name="x", inputs=("video",), output="video", doc="d", timeline=False)
    with pytest.raises(Exception):
        f.name = "y"  # type: ignore[misc]


def test_source_filter_is_frozen() -> None:
    s = SourceFilter(name="x", output="video", doc="d")
    with pytest.raises(Exception):
        s.name = "y"  # type: ignore[misc]


# --- real ffmpeg (exec) ---------------------------------------------------


@pytest.mark.exec
def test_real_registry_available() -> None:
    reg = load()
    assert reg.available() is True


@pytest.mark.exec
def test_real_names_over_300() -> None:
    reg = load()
    assert len(reg.names()) > 300


@pytest.mark.exec
def test_real_gblur_present_with_numeric_sigma() -> None:
    reg = load()
    f = reg.get("gblur")
    assert f is not None
    assert f.inputs == ("video",)
    assert f.output == "video"
    opts = reg.options("gblur")
    assert opts is not None
    assert opts["sigma"].type == "num"


@pytest.mark.exec
def test_real_overlay_two_video_in_one_video_out() -> None:
    reg = load()
    f = reg.get("overlay")
    assert f is not None
    assert f.inputs == ("video", "video")
    assert f.output == "video"


@pytest.mark.exec
def test_real_split_and_asplit_absent_dynamic_pads() -> None:
    reg = load()
    # Verified empirically: split is V->N, asplit is A->N -- both dynamic.
    assert reg.get("split") is None
    assert reg.get("asplit") is None


@pytest.mark.exec
def test_real_testsrc_absent_from_column_function_lookup() -> None:
    # get()/names() (column-function lookup) is UNCHANGED for sources --
    # testsrc is a real source (see test_real_source_names_over_20 etc.
    # below) but is not callable as a column function.
    reg = load()
    assert reg.get("testsrc") is None
    assert "testsrc" not in reg.names()


@pytest.mark.exec
def test_real_source_names_over_20() -> None:
    reg = load()
    assert len(reg.source_names()) > 20


@pytest.mark.exec
def test_real_testsrc_anullsrc_sine_color_present_with_correct_types() -> None:
    reg = load()
    testsrc = reg.get_source("testsrc")
    assert testsrc is not None and testsrc.output == "video"
    anullsrc = reg.get_source("anullsrc")
    assert anullsrc is not None and anullsrc.output == "audio"
    sine = reg.get_source("sine")
    assert sine is not None and sine.output == "audio"
    color = reg.get_source("color")
    assert color is not None and color.output == "video"


@pytest.mark.exec
def test_real_source_options_load_via_registry() -> None:
    reg = load()
    testsrc_opts = reg.options("testsrc")
    assert testsrc_opts is not None and "size" in testsrc_opts
    anullsrc_opts = reg.options("anullsrc")
    assert anullsrc_opts is not None and "sample_rate" in anullsrc_opts
    sine_opts = reg.options("sine")
    assert sine_opts is not None and "frequency" in sine_opts
    color_opts = reg.options("color")
    assert color_opts is not None and "color" in color_opts


@pytest.mark.exec
def test_real_multi_output_and_dynamic_sources_absent() -> None:
    reg = load()
    # avsynctest is |->AV (2-char output); amovie/movie are |->N (dynamic).
    assert reg.get_source("avsynctest") is None
    assert reg.get_source("amovie") is None
    assert reg.get_source("movie") is None


@pytest.mark.exec
def test_real_sinks_absent_from_sources() -> None:
    reg = load()
    assert reg.get_source("anullsink") is None
    assert reg.get_source("nullsink") is None
    assert reg.get_source("abuffersink") is None
    assert reg.get_source("buffersink") is None


@pytest.mark.exec
def test_real_timeline_flags_pinned() -> None:
    reg = load()
    gblur = reg.get("gblur")
    assert gblur is not None and gblur.timeline is True
    drawbox = reg.get("drawbox")
    assert drawbox is not None and drawbox.timeline is True
    overlay = reg.get("overlay")
    assert overlay is not None and overlay.timeline is True
    null = reg.get("null")
    assert null is not None and null.timeline is False
    fmt = reg.get("format")
    assert fmt is not None and fmt.timeline is False


@pytest.mark.exec
def test_real_options_lazy_does_not_call_help_for_unreferenced_filters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_cache()
    calls: list[list[str]] = []
    orig_run = subprocess.run

    def counting_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        return orig_run(argv, **kwargs)  # type: ignore[return-value]

    monkeypatch.setattr(registry_mod.subprocess, "run", counting_run)
    reg = load()
    reg.available()
    reg.options("gblur")
    reg.options("gblur")  # memoized -- must not add another call
    reg.options("overlay")

    help_calls = [c for c in calls if any(a.startswith("filter=") for a in c)]
    assert len(help_calls) == 2  # exactly gblur + overlay, never all ~460+
    assert sum(1 for c in calls if "-filters" in c) == 1
