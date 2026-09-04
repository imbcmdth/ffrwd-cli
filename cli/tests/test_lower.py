"""Tests for the lower pass and the compiler pipeline.

These go through the real parser: lowering is only ever handed a ``Resolved``
that ``resolve`` accepted, so hand-built inputs would test a shape that cannot
occur. ``compile_sql`` is used wherever the split pass is irrelevant or wanted;
``lower`` is called directly when a test needs the pre-split graph or a
synthetic :class:`~ffrwd.probe.ProbeResult`.

Paths in these queries deliberately do not exist, so ``compile_sql``'s
opportunistic probing degrades to symbolic lowering without shelling out
; probe-dependent behavior — which is ALL of
broadcasting, since an array's length comes from the file — is exercised
either with a hand-built ``ProbeResult`` through ``lower`` directly, or, for
the real thing, in an ``exec``-marked test against ``tests/fixtures/av.mp4``
(1 audio track) and ``tests/fixtures/av2.mp4`` (2 language-tagged tracks).

Tier-2 behavior is tested twice over: once against a `Registry`
built from the captured ffmpeg output embedded below, so the default suite
stays offline and machine-independent, and once (``exec``-marked) against the
real installed ffmpeg, where only what ffmpeg itself guarantees is asserted.
"""

# ruff: noqa: E501
# The `-filters` / `-help filter=X` fixture text below is embedded verbatim
# (byte-for-byte, not retyped or rewrapped) so it stays trustworthy as a
# record of real ffmpeg output; a few of its option lines exceed the 100-col
# limit. Whole-file exemption for the same reason tests/test_registry.py
# takes one: a per-line noqa comment would have to sit inside the fixture
# string it applies to, corrupting it.

from __future__ import annotations

import base64
import functools
import re
import shlex
import struct
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import pytest
import sqlglot

from ffrwd import compiler
from ffrwd import lower as lower_module
from ffrwd import registry as registry_module
from ffrwd.compiler import compile_sql
from ffrwd.emit import build_ffmpeg_args, emit
from ffrwd.errors import ErrorCode, FfrwdError
from ffrwd.functions import WASM_SOURCE, Parameter, WasmFunction
from ffrwd.ir import NO_CHAPTERS, NO_METADATA, Attachment, Graph, StreamType
from ffrwd.lower import lower, lower_table
from ffrwd.parser import Resolved, parse, resolve
from ffrwd.probe import (
    AttachmentMeta,
    ChapterMeta,
    CueMeta,
    ProbeFailure,
    ProbeResult,
    RenditionMeta,
    StreamMeta,
    parse_webvtt,
    probe,
)
from ffrwd.registry import Registry, load_reference
from ffrwd.split import insert_splits
from ffrwd.table import ArrayCell, RecordCell, StreamCell, render_csv, render_table
from ffrwd.types import DISPOSITION_KEYS
from ffrwd.warnings import FfrwdWarning, OnWarning, WarningCode
from ffrwd.wasm import WORLDS, Described, SourceCatalog, SourceRendition
from ffrwd.wasm import SourceTrack as WasmSourceTrack

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PROJECT_ROOT.parent
FIXTURES_DIR = PROJECT_ROOT / "tests" / "fixtures"
SNAPSHOT_PATH = PROJECT_ROOT / "tests" / "data" / "reference_registry.json"


# README ```sql blocks are dispatched by CONTENT, not by position, so moving
# an example up or down the page does not silently re-point a test. Both
# examples name files nobody has; each is compiled against the real
# two-language fixtures instead, which is exactly how its shown command was
# produced. (The union example lives in docs/examples.md; its pin is in
# tests/test_examples.py's generic harness.)
_FLAGSHIP_README_PATHS = {"film.mkv": "av2.mp4", "commentary.mkv": "av3.mp4"}


def _readme_text() -> str:
    return (REPO_ROOT / "README.md").read_text(encoding="utf-8")


def _readme_block(needle: str, *, exclude: str | None = None) -> str:
    """The one ```sql block of README.md containing `needle`, verbatim.

    `exclude`, if given, drops any block that ALSO contains that substring --
    needed for the Encoding section, whose ```sql block wraps the
    flagship query verbatim in `COPY (...)`, so it contains every needle the
    flagship's own block does (e.g. "commentary").
    """
    blocks = re.findall(r"```sql\n(.*?)```", _readme_text(), re.DOTALL)
    assert blocks, "README.md no longer contains a ```sql block"
    matching = [
        str(block)
        for block in blocks
        if needle in block and (exclude is None or exclude not in block)
    ]
    assert len(matching) == 1, f"expected exactly one README ```sql block with {needle!r}"
    return matching[0]


@functools.cache
def _snapshot_registry() -> Registry:
    """The captured ffmpeg 7.1 filter set: offline, machine-independent, complete.

    The registry is the WHOLE function surface, so even a lowering
    test that says nothing about ffmpeg needs one to resolve `gblur` or
    `volume`. `tests/data/reference_registry.json` IS that registry -- no
    subprocess, no PATH lookup, identical answers on every machine. Tests that
    want a deliberately SMALL, or absent, filter set build their own instead
    (`_registry` / `_dyn`, further down).
    """
    return load_reference(SNAPSHOT_PATH)


def _lower(
    sql: str,
    probes: dict[str, ProbeResult | None] | None = None,
    on_warning: OnWarning | None = None,
) -> Graph:
    return lower(
        resolve(parse(sql)), probes or {}, registry=_snapshot_registry(), on_warning=on_warning
    )


def _serialized_sinks(d: dict[str, object]) -> list[dict[str, object]]:
    """``Graph.to_dict()["sinks"]``, narrowed for the type checker."""
    sinks = d["sinks"]
    assert isinstance(sinks, list)
    return sinks


def _reject(sql: str) -> FfrwdError:
    with pytest.raises(FfrwdError) as excinfo:
        compile_sql(sql)
    return _anchored(excinfo.value)


def _reject_lower(sql: str, probes: dict[str, ProbeResult | None]) -> FfrwdError:
    """Like ``_reject``, for rejections that only a probed input can produce."""
    with pytest.raises(FfrwdError) as excinfo:
        _lower(sql, probes)
    return _anchored(excinfo.value)


def _anchored(err: FfrwdError) -> FfrwdError:
    assert err.line is not None, "every rejection must be line-anchored"
    assert err.col is not None
    return err


def _filters(g: Graph) -> list[str]:
    return [node.filter for node in g.nodes.values()]


def _outputs(g: Graph) -> list[tuple[str, str, str | None]]:
    return [(o.ref, o.type, o.name) for o in g.outputs]


def _probe_result(
    videos: int = 1,
    audios: int = 1,
    *,
    video_tags: dict[str, str] | None = None,
    audio_tags: dict[str, str] | None = None,
    per_audio_tags: list[dict[str, str]] | None = None,
) -> ProbeResult:
    """A synthetic ProbeResult -- no ffprobe, no fixture, no disk.

    ``audio_tags`` tags every audio stream the same; ``per_audio_tags`` tags
    them one by one (and fixes ``audios``), which is what the broadcasting
    provenance tests need.
    """
    if per_audio_tags is not None:
        audios = len(per_audio_tags)
    streams = [
        StreamMeta(
            type="video",
            index=i,
            metadata=dict(video_tags or {}),
            width=320,
            height=240,
            fps="15/1",
            sample_rate=None,
            codec="h264",
            channels=None,
            channel_layout=None,
            bitrate=None,
            duration=None,
            color_transfer=None,
        )
        for i in range(videos)
    ]
    streams += [
        StreamMeta(
            type="audio",
            index=i,
            metadata=dict(per_audio_tags[i] if per_audio_tags is not None else audio_tags or {}),
            width=None,
            height=None,
            fps=None,
            sample_rate=44100,
            codec="aac",
            channels=None,
            channel_layout=None,
            bitrate=None,
            duration=None,
            color_transfer=None,
        )
        for i in range(audios)
    ]
    return ProbeResult(streams=streams)


_LAYOUT_TYPES: dict[str, StreamType] = {
    "v": "video",
    "a": "audio",
    "s": "subtitle",
    "d": "data",
}


def _layout_probe(
    layout: str, tags: dict[int, dict[str, str]] | None = None
) -> ProbeResult:
    """A ProbeResult in FILE order, written as a compact layout.

    One character per stream -- ``v``/``a``/``s``/``d`` -- so ``"vasd"`` is the
    four-type container a star has to expand in order, and ``"vas"`` is
    avs.mkv's shape. Per-type indices are assigned in file order, exactly as
    :mod:`ffrwd.probe` does. `tags` maps a FILE position (not a per-type
    index) to that stream's metadata.
    """
    counters: dict[str, int] = {}
    streams: list[StreamMeta] = []
    for position, letter in enumerate(layout):
        index = counters.get(letter, 0)
        counters[letter] = index + 1
        streams.append(
            StreamMeta(
                type=_LAYOUT_TYPES[letter],
                index=index,
                metadata=dict((tags or {}).get(position, {})),
                width=320 if letter == "v" else None,
                height=240 if letter == "v" else None,
                fps="15/1" if letter == "v" else None,
                sample_rate=44100 if letter == "a" else None,
                codec={"v": "h264", "a": "aac", "s": "subrip", "d": "bin_data"}[letter],
                channels=None,
                channel_layout=None,
                bitrate=None,
                duration=None,
                color_transfer=None,
            )
        )
    return ProbeResult(streams=streams)


# ---------------------------------------------------------------------------
# the README flagship: PiP composite + broadcast-zip mix
# ---------------------------------------------------------------------------


def _readme_flagship_sql() -> str:
    """The headline: a CTE carrying a video column AND a whole audio array,
    re-pointed at the real fixtures (film=av2, commentary=av3).

    `c.audio` (in the CTE) and `f.audio` (in the outer `volume()` calls) are
    both bare arrays -- broadcasting them needs a real, readable file to know
    how many streams there are, same reason the union-splat example below
    needs one.
    """
    sql = _readme_block("TO ('pip.mkv')")
    for shown, fixture in _FLAGSHIP_README_PATHS.items():
        sql = sql.replace(shown, (FIXTURES_DIR / fixture).as_posix())
    return sql


@pytest.mark.exec
def test_readme_flagship_lowers_to_expected_nodes(_fixtures: None) -> None:
    """scale->overlay composites the video; volume broadcasts over each
    2-track audio array (4 nodes) and amix zips the pairs (2 nodes)."""
    g = compile_sql(_readme_flagship_sql())
    assert _filters(g) == [
        "scale", "overlay", "volume", "volume", "volume", "volume", "amix", "amix",
    ]
    # CTEs are traversed first, so the CTE's alias `c` takes input 0.
    assert g.sources == {"c": 0, "f": 1}
    assert g.nodes["n2"].inputs == ["src:f:v:0", "n1"]  # overlay(f.video[1], pip.frame)
    # amix zips f.audio[k]*0.65 with c.audio[k]*0.35, one pair per language
    assert g.nodes["n7"].inputs == ["n3", "n5"]
    assert g.nodes["n8"].inputs == ["n4", "n6"]


@pytest.mark.exec
def test_readme_flagship_selects_its_three_streams(_fixtures: None) -> None:
    """The SELECT list is the output list: one composited video column, and
    the zipped mix broadcasts into one audio column per language."""
    g = compile_sql(_readme_flagship_sql())
    assert [o.type for o in g.outputs] == ["video", "audio", "audio"]


@pytest.mark.exec
def test_readme_flagship_amix_pairs_keep_the_agreed_language(_fixtures: None) -> None:
    """A multi-stream call threads provenance when every zipped
    input agrees. Both fixtures tag audio[1]=eng, audio[2]=fra, so each mixed
    pair keeps its language; the composited video has no tag on either side
    (probed, untagged video streams) for `overlay` to agree on."""
    g = compile_sql(_readme_flagship_sql())
    assert [o.metadata for o in g.outputs] == [
        {},
        {"language": "eng"},
        {"language": "fra"},
    ]


@pytest.mark.exec
def test_readme_flagship_emits_a_filtergraph(_fixtures: None) -> None:
    e = emit(compile_sql(_readme_flagship_sql()))
    assert "scale=" in e.filter_complex
    assert "overlay=" in e.filter_complex
    assert "amix=" in e.filter_complex
    assert [m.target for m in e.maps] == ["[out0]", "[out1]", "[out2]"]
    assert [m.copy for m in e.maps] == [False, False, False]


def _readme_command_rendering(args: list[str], sink_paths: list[str]) -> str:
    """The README's display form of a compiled command: one filter chain per
    line inside the quoted graph, output groups after it wrapped at ~90
    columns, every continuation line indented two spaces. Newlines and
    leading spaces inside the graph are legal to BOTH parties: a
    single-quoted shell argument keeps them, and ffmpeg's graph parser skips
    them (verified against ffmpeg 9)."""
    graph_at = args.index("-filter_complex") + 1
    chains = args[graph_at].split(";")
    lines = [shlex.join(args[:graph_at]) + " '"]
    lines += [f"  {chain};" for chain in chains[:-1]]
    lines.append(f"  {chains[-1]}' \\")
    groups: list[list[str]] = [[]]
    for arg in args[graph_at + 1 :]:
        groups[-1].append(arg)
        if arg in sink_paths:
            groups.append([])
    groups.pop()
    tail: list[str] = []
    for group in groups:
        current = " "
        for arg in group:
            piece = shlex.join([arg])
            if len(current) + 1 + len(piece) > 88 and current.strip():
                tail.append(current)
                current = " "
            current += " " + piece
        tail.append(current)
    for k, line in enumerate(tail):
        lines.append(line + (" \\" if k < len(tail) - 1 else ""))
    return "\n".join(lines)


@pytest.mark.exec
def test_readme_flagship_command_is_the_real_compilation(_fixtures: None) -> None:
    """The command shown under the headline is what ffrwd actually prints for
    that query, with only the fixture paths written back to the shown names."""
    args = build_ffmpeg_args(emit(compile_sql(_readme_flagship_sql())))
    shown = _readme_command_rendering(args, ["pip.mkv"])
    for name, fixture in _FLAGSHIP_README_PATHS.items():
        shown = shown.replace((FIXTURES_DIR / fixture).as_posix(), name)
    assert shown in _readme_text(), shown


# ---------------------------------------------------------------------------
# the README Encoding section: the flagship wrapped in COPY
# ---------------------------------------------------------------------------


def _readme_encoding_sql() -> str:
    """The Encoding section's ```sql block: the flagship verbatim inside a
    COPY ... TO ... WITH (...), re-pointed at the real fixtures the same way.

    Needled on its WITH options (unique to this block): the PiP demo is also
    a COPY over 'pip.mkv', and the ladder script also opens with 'COPY ('.
    """
    sql = _readme_block("audio_bitrate '192k'")
    for shown, fixture in _FLAGSHIP_README_PATHS.items():
        sql = sql.replace(shown, (FIXTURES_DIR / fixture).as_posix())
    return sql


@pytest.mark.exec
def test_readme_encoding_wraps_the_flagship_query_in_a_sink(_fixtures: None) -> None:
    """Wrapping the flagship in COPY adds a sink and leaves the rest of the
    graph untouched -- same shape as `test_sink_does_not_change_the_graph_shape`."""
    plain = compile_sql(_readme_flagship_sql()).to_dict()
    wrapped = compile_sql(_readme_encoding_sql()).to_dict()
    unit = _serialized_sinks(wrapped)[0]
    assert unit["path"] == "pip.mkv"
    assert unit["options"] == {
        "video_codec": "libx264",
        "crf": 20,
        "audio_codec": "aac",
        "audio_bitrate": "192k",
    }
    # Both sections COPY to 'pip.mkv'; the WITH options are the ONLY
    # difference -- same nodes, same outputs, same destination.
    unit["options"] = {}
    assert wrapped == plain


@pytest.mark.exec
def test_readme_encoding_command_is_the_real_compilation(_fixtures: None) -> None:
    """The command shown under Encoding is what ffrwd actually prints for
    that query, with only the fixture paths written back to the shown names.

    No override: the sink's own `TO 'pip.mkv'` supplies the path, same as
    `ffrwd run query.sql` would use.
    """
    args = build_ffmpeg_args(emit(compile_sql(_readme_encoding_sql())))
    shown = _readme_command_rendering(args, ["pip.mkv"])
    for name, fixture in _FLAGSHIP_README_PATHS.items():
        shown = shown.replace((FIXTURES_DIR / fixture).as_posix(), name)
    assert shown in _readme_text(), shown


# ---------------------------------------------------------------------------
# the README "Views and multiple outputs" example
# ---------------------------------------------------------------------------
#
# Explicit subscripts only (`f.video[1]`, `f.audio[1]`), so this compiles
# fully symbolically -- no fixture substitution, no probe, no registry. Pinned
# offline, not exec-marked.


def _readme_ladder_sql() -> str:
    """The ABR-ladder script, verbatim -- 'film.mkv' need not exist."""
    return _readme_block("CREATE VIEW main")


def test_readme_ladder_example_compiles() -> None:
    g = compile_sql(_readme_ladder_sql())
    assert [unit.path for unit in g.sinks] == ["720.mp4", "360.mp4", "audio.m4a"]
    filters = [node.filter for node in g.nodes.values()]
    assert filters.count("scale") == 3  # main's own + one per video COPY
    assert filters.count("volume") == 1  # built once, split across all 3 COPYs
    assert g.input_paths == ["film.mkv"]  # one -i: decoded once


def test_readme_ladder_example_command_is_the_real_compilation() -> None:
    """The command shown under "Views and multiple outputs" is what ffrwd
    actually prints for that script, in the README's one-filter-per-line
    display form -- no fixture path to substitute back, since the query
    names no file ffrwd can (or needs to) read."""
    g = compile_sql(_readme_ladder_sql())
    args = build_ffmpeg_args(emit(g), None)
    rendering = _readme_command_rendering(args, [unit.path for unit in g.sinks])
    assert rendering in _readme_text()


def test_readme_flagship_scale_factor_is_not_a_decimal() -> None:
    """``Literal.to_py()`` yields Decimal for 0.5; the IR must carry float."""
    g = _lower("SELECT scale(a.video[1], 0.5, 0.25) FROM input('x.mp4') a")
    args = g.nodes["n1"].args
    assert args == {"width": 0.5, "height": 0.25}
    assert all(type(v) is float for v in args.values())


# ---------------------------------------------------------------------------
# typed columns, subscripts, passthrough
# ---------------------------------------------------------------------------


def test_a_video_subscript_is_the_first_video_stream() -> None:
    g = compile_sql("SELECT a.video[1] FROM input('x.mp4') a")
    assert g.nodes == {}
    assert _outputs(g) == [("src:a:v:0", "video", None)]
    assert g.sources == {"a": 0}


def test_sql_subscripts_are_one_based() -> None:
    g = compile_sql("SELECT a.video[2], a.audio[3] FROM input('x.mp4') a")
    assert _outputs(g) == [
        ("src:a:v:1", "video", None),
        ("src:a:a:2", "audio", None),
    ]


def test_remap_only_query_is_pure_passthrough() -> None:
    """`SELECT a.video[1], a.audio[2]` -> two -map/-c copy streams, no filters."""
    g = compile_sql("SELECT a.video[1], a.audio[2] FROM input('foo.mp4') a")
    assert g.nodes == {}
    assert _outputs(g) == [
        ("src:a:v:0", "video", None),
        ("src:a:a:1", "audio", None),
    ]
    e = emit(g)
    assert e.filter_complex == ""
    assert [(m.target, m.type, m.copy) for m in e.maps] == [
        ("0:v:0", "video", True),
        ("0:a:1", "audio", True),
    ]


def test_filtered_video_with_passthrough_audio() -> None:
    g = compile_sql("SELECT scale(a.video[1], 0.5), a.audio[1] FROM input('foo.mp4') a")
    assert _filters(g) == ["scale"]
    assert _outputs(g) == [("n1", "video", None), ("src:a:a:0", "audio", None)]
    e = emit(g)
    assert [(m.target, m.copy) for m in e.maps] == [("[out0]", False), ("0:a:0", True)]


def test_select_alias_becomes_the_output_name() -> None:
    g = compile_sql(
        "SELECT a.video[1] AS picture, a.audio[1] AS Sound FROM input('x.mp4') a"
    )
    assert [o.name for o in g.outputs] == ["picture", "sound"]  # unquoted -> folded


def test_quoted_select_alias_keeps_its_case() -> None:
    g = compile_sql('SELECT a.audio[1] AS "Commentary" FROM input(\'x.mp4\') a')
    assert [o.name for o in g.outputs] == ["Commentary"]


def test_output_order_is_select_order() -> None:
    g = compile_sql("SELECT a.audio[1], a.video[1] FROM input('x.mp4') a")
    assert [o.type for o in g.outputs] == ["audio", "video"]


def test_time_column_outside_where_is_rejected() -> None:
    err = _reject("SELECT a.t FROM input('x.mp4') a")
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "a.t" in err.message


def test_unknown_column_is_rejected() -> None:
    # `width` is a real column name now (a rendition one), refused by its own
    # more specific message -- see `test_a_rendition_column_on_a_renditionless_
    # input_is_refused` -- so this wants a name that is unknown outright.
    err = _reject("SELECT hflip(a.frobnicate) FROM input('x.mp4') a")
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "a.frobnicate" in err.message


def test_literal_projection_is_rejected() -> None:
    err = _reject("SELECT 1 FROM input('x.mp4') a")
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "every SELECT column must be a stream expression" in err.message


def test_arithmetic_projection_is_rejected() -> None:
    err = _reject("SELECT 1 + 2 FROM input('x.mp4') a")
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "every SELECT column must be a stream expression" in err.message


# ---------------------------------------------------------------------------
# broadcasting: bare arrays splat
# ---------------------------------------------------------------------------


def test_bare_audio_array_splats_into_one_output_per_stream() -> None:
    g = _lower("SELECT a.audio FROM input('x.mp4') a", {"a": _probe_result(audios=3)})
    assert g.nodes == {}
    assert _outputs(g) == [
        ("src:a:a:0", "audio", None),
        ("src:a:a:1", "audio", None),
        ("src:a:a:2", "audio", None),
    ]


def test_bare_video_array_splats_in_file_order() -> None:
    g = _lower(
        "SELECT a.video FROM input('x.mp4') a", {"a": _probe_result(videos=2, audios=1)}
    )
    assert _outputs(g) == [("src:a:v:0", "video", None), ("src:a:v:1", "video", None)]


def test_a_splat_keeps_its_place_in_the_select_list() -> None:
    g = _lower(
        "SELECT a.audio, a.video[1] FROM input('x.mp4') a",
        {"a": _probe_result(videos=1, audios=2)},
    )
    assert [o.type for o in g.outputs] == ["audio", "audio", "video"]


def test_as_alias_is_repeated_verbatim_on_every_splatted_element() -> None:
    """The alias names the COLUMN, so elements are NOT suffixed with an ordinal."""
    g = _lower(
        "SELECT a.audio AS Track FROM input('x.mp4') a", {"a": _probe_result(audios=2)}
    )
    assert [o.name for o in g.outputs] == ["track", "track"]


def test_an_unaliased_splat_names_nothing() -> None:
    g = _lower("SELECT a.audio FROM input('x.mp4') a", {"a": _probe_result(audios=2)})
    assert [o.name for o in g.outputs] == [None, None]


def test_a_one_stream_array_still_splats() -> None:
    g = _lower("SELECT a.audio FROM input('x.mp4') a", {"a": _probe_result(audios=1)})
    assert _outputs(g) == [("src:a:a:0", "audio", None)]


def test_unprobeable_bare_array_cannot_be_enumerated() -> None:
    """No probe -> no length -> INPUT_NOT_FOUND, the natural error."""
    err = _reject("SELECT a.audio FROM input('nope.mp4') a")
    assert err.code is ErrorCode.INPUT_NOT_FOUND
    assert "cannot enumerate the streams of 'nope.mp4'" in err.message
    assert err.line == 1 and err.col is not None


def test_unprobeable_bare_array_as_an_argument_is_the_same_error() -> None:
    err = _reject("SELECT volume(a.audio, 0.5) FROM input('nope.mp4') a")
    assert err.code is ErrorCode.INPUT_NOT_FOUND
    assert err.hint is not None and "a.audio[1]" in err.hint


def test_an_unreadable_input_also_loses_the_ability_to_enumerate() -> None:
    with pytest.raises(FfrwdError) as excinfo:
        compile_sql("SELECT a.audio FROM input('x.mp4') a")
    assert excinfo.value.code is ErrorCode.INPUT_NOT_FOUND


# ---------------------------------------------------------------------------
# probe honesty: a probe that fails for a path that DOES exist must not
# claim the file was not found
# ---------------------------------------------------------------------------


def test_a_probe_failure_for_an_existing_path_quotes_ffprobe_instead() -> None:
    """`probe_failures` says WHY the probe came back None for this alias --
    the rejection must not claim the file wasn't found, and must carry
    ffprobe's own words instead."""
    with pytest.raises(FfrwdError) as excinfo:
        lower(
            resolve(parse("SELECT a.audio FROM input('x.mp4') a")),
            {"a": None},
            registry=_snapshot_registry(),
            probe_failures={
                "a": ProbeFailure(stderr="Invalid data found when processing input")
            },
        )
    err = _anchored(excinfo.value)
    assert err.code is ErrorCode.INPUT_NOT_FOUND
    assert "file not found" not in err.message
    assert "Invalid data found when processing input" in err.message
    assert err.hint is not None
    assert "ffprobe" in err.hint


def test_a_probe_failure_with_no_stderr_still_avoids_the_missing_file_claim() -> None:
    """ffprobe can fail without saying anything on stderr (a timeout, a
    missing binary) -- the message still must not claim the file is missing."""
    with pytest.raises(FfrwdError) as excinfo:
        lower(
            resolve(parse("SELECT a.audio FROM input('x.mp4') a")),
            {"a": None},
            registry=_snapshot_registry(),
            probe_failures={"a": ProbeFailure(stderr=None)},
        )
    err = _anchored(excinfo.value)
    assert "file not found" not in err.message
    assert "the probe failed" in err.message


def test_no_recorded_failure_keeps_the_old_missing_file_message() -> None:
    """An alias `probe_failures` has nothing to say about -- unset entirely,
    or explicitly None -- keeps the familiar wording; this is the same
    "genuinely missing" story a bare `{}` probe_failures always told."""
    err = _reject_lower("SELECT a.audio FROM input('nope.mp4') a", {"a": None})
    assert "file not found or unreadable" in err.message


def test_compile_sql_is_honest_when_a_probe_fails_for_a_file_that_exists(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """End-to-end: the compiler wires `probe_failure_path` into lowering, so
    a real (if unreadable) file gets the honest message, not "not found"."""
    existing = tmp_path / "src10.mp4"
    existing.write_bytes(b"not really a video")
    monkeypatch.setattr(compiler, "probe_path", lambda path, args=(), **kw: None)
    monkeypatch.setattr(
        compiler,
        "probe_failure_path",
        lambda path, args=(), **kw: ProbeFailure(
            stderr="Invalid data found when processing input"
        ),
    )
    err = _reject(f"SELECT a.audio FROM input('{existing.as_posix()}') a")
    assert err.code is ErrorCode.INPUT_NOT_FOUND
    assert "file not found" not in err.message
    assert "Invalid data found when processing input" in err.message


# ---------------------------------------------------------------------------
# broadcasting: calls expand elementwise
# ---------------------------------------------------------------------------


def test_scalar_broadcast_makes_one_node_per_element() -> None:
    g = _lower(
        "SELECT volume(a.audio, 0.5) FROM input('x.mp4') a", {"a": _probe_result(audios=2)}
    )
    assert _filters(g) == ["volume", "volume"]
    assert g.nodes["n1"].inputs == ["src:a:a:0"]
    assert g.nodes["n2"].inputs == ["src:a:a:1"]
    assert g.nodes["n1"].args == g.nodes["n2"].args == {"volume": 0.5}
    assert _outputs(g) == [("n1", "audio", None), ("n2", "audio", None)]


def test_nested_broadcasts_compose() -> None:
    """`volume(reverb(a.audio, 0.3), 0.5)` : the inner array flows outward."""
    g = _lower(
        "SELECT volume(aecho(a.audio, 0.8, 0.9, 60, 0.3), 0.5) FROM input('x.mp4') a",
        {"a": _probe_result(audios=2)},
    )
    assert _filters(g) == ["aecho", "aecho", "volume", "volume"]
    assert g.nodes["n3"].inputs == ["n1"]
    assert g.nodes["n4"].inputs == ["n2"]


def test_two_arrays_zip_elementwise() -> None:
    g = _lower(
        "SELECT amix(a.audio, b.audio) FROM input('x.mp4') a, input('y.mp4') b",
        {"a": _probe_result(audios=2), "b": _probe_result(audios=2)},
    )
    assert _filters(g) == ["amix", "amix"]
    assert g.nodes["n1"].inputs == ["src:a:a:0", "src:b:a:0"]
    assert g.nodes["n2"].inputs == ["src:a:a:1", "src:b:a:1"]


def test_a_scalar_argument_repeats_into_every_element() -> None:
    g = _lower(
        "SELECT amix(a.audio, b.audio[1]) FROM input('x.mp4') a, input('y.mp4') b",
        {"a": _probe_result(audios=2), "b": _probe_result(audios=1)},
    )
    assert g.nodes["n1"].inputs == ["src:a:a:0", "src:b:a:0"]
    assert g.nodes["n2"].inputs == ["src:a:a:1", "src:b:a:0"]


def test_a_repeated_scalar_is_fanned_out_by_the_split_pass() -> None:
    """Broadcasting reuses one ref N times; asplit is what makes that legal."""
    g = insert_splits(
        _lower(
            "SELECT amix(a.audio, b.audio[1]) FROM input('x.mp4') a, input('y.mp4') b",
            {"a": _probe_result(audios=2), "b": _probe_result(audios=1)},
        )
    )
    assert _filters(g) == ["asplit", "amix", "amix"]
    assert g.nodes["src_b_a_0_split"].args == {"n": 2}
    assert g.nodes["n1"].inputs == ["src:a:a:0", "src_b_a_0_split:0"]
    assert g.nodes["n2"].inputs == ["src:a:a:1", "src_b_a_0_split:1"]


def test_a_fixed_arity_acrossfade_wins_over_n_input_treatment(_registry: Registry) -> None:
    """acrossfade is `AA->A` on ffmpeg 7.1 (an ordinary two-input filter, no
    `inputs` option) and `N->A` on ffmpeg 9 (n-input, `inputs` option). The
    dispatch order (`_lower_call` checks `DynamicFilter.n_input` before
    treating a name as a fixed-arity registry call) must not run n-input
    treatment against a build whose acrossfade is not n-input at all: an
    ordinary two-input call, no `inputs` option written (it has none)."""
    g = _dyn(
        "SELECT acrossfade(a.audio[1], b.audio[1], duration => 1) "
        "FROM input('x.mp4') a, input('y.mp4') b",
        _registry,
    )
    node = next(iter(g.nodes.values()))
    assert node.filter == "acrossfade"
    assert node.args == {"duration": 1}
    assert "inputs" not in node.args


@pytest.mark.exec
def test_a_variadic_acrossfade_omits_the_defaulted_count_and_writes_a_real_one() -> None:
    """On a build where acrossfade is n-input (ffmpeg 9+: `N->A`, an `inputs`
    option), `emit_default=False` -- the override this brief adds -- omits
    `inputs` only up to the default of 2, so the two-stream command stays
    valid on every ffmpeg (cookbook recipe 13's pin is version-stable)."""
    live = registry_module.load()
    dynamic = live.get("acrossfade")
    if dynamic is not None and not dynamic.n_input:
        pytest.skip(
            "this ffmpeg's acrossfade is a fixed two-input filter; the "
            "n-input path only exists on builds where it is N->A"
        )
    two = compile_sql(
        "SELECT acrossfade(a.audio[1], a.audio[2], duration => 1) "
        "FROM input('tests/fixtures/av2.mp4') a"
    )
    three = compile_sql(
        "SELECT acrossfade(a.audio[1], a.audio[2], a.audio[1], duration => 1, inputs => 3) "
        "FROM input('tests/fixtures/av2.mp4') a"
    )
    two_node = next(n for n in two.nodes.values() if n.filter == "acrossfade")
    three_node = next(n for n in three.nodes.values() if n.filter == "acrossfade")
    assert "inputs" not in two_node.args
    assert three_node.args["inputs"] == 3


def test_zip_length_mismatch_is_a_broadcast_mismatch() -> None:
    err = _reject_lower(
        "SELECT amix(a.audio, b.audio) FROM input('x.mp4') a, input('y.mp4') b",
        {"a": _probe_result(audios=3), "b": _probe_result(audios=2)},
    )
    assert err.code is ErrorCode.BROADCAST_MISMATCH
    assert "a.audio has 3 streams" in err.message
    assert "b.audio has 2 streams" in err.message


def test_broadcast_over_a_video_array() -> None:
    g = _lower(
        "SELECT hflip(a.video) FROM input('x.mp4') a", {"a": _probe_result(videos=2)}
    )
    assert _filters(g) == ["hflip", "hflip"]


def test_an_array_argument_still_type_checks_by_element() -> None:
    err = _reject_lower(
        "SELECT hflip(a.audio) FROM input('x.mp4') a", {"a": _probe_result(audios=2)}
    )
    assert err.code is ErrorCode.UDF_ARG_TYPE
    assert "it takes video as its stream input, got (audio)" in err.message


def test_broadcast_composes_with_the_input_seek() -> None:
    """One window on the -i covers every element of the broadcast array.

    The input seek: no atrim/asetpts pair per element -- the
    calls consume the raw stream refs and the whole input is seeked once.
    """
    g = _lower(
        "SELECT volume(a.audio, 0.5), aecho(a.audio, 0.8, 0.9, 60, 0.3) FROM input('x.mp4') a "
        "WHERE a.t BETWEEN 1 AND 2",
        {"a": _probe_result(audios=2)},
    )
    assert _filters(g) == ["volume", "volume", "aecho", "aecho"]
    assert g.input_trims == {"a": (1, 2)}
    assert [g.nodes[n].inputs for n in ("n1", "n2", "n3", "n4")] == [
        ["src:a:a:0"],
        ["src:a:a:1"],
        ["src:a:a:0"],
        ["src:a:a:1"],
    ]


# ---------------------------------------------------------------------------
# broadcasting: provenance
# ---------------------------------------------------------------------------


def test_broadcast_outputs_carry_their_own_source_metadata() -> None:
    g = _lower(
        "SELECT aecho(a.audio, 0.8, 0.9, 60, 0.3) FROM input('x.mp4') a",
        {
            "a": _probe_result(
                audios=2, per_audio_tags=[{"language": "eng"}, {"language": "fra"}]
            )
        },
    )
    assert [o.metadata for o in g.outputs] == [{"language": "eng"}, {"language": "fra"}]


def test_provenance_survives_a_chain_of_single_stream_calls() -> None:
    g = _lower(
        "SELECT volume(aecho(a.audio[1], 0.8, 0.9, 60, 0.3), 0.5) FROM input('x.mp4') a",
        {"a": _probe_result(audio_tags={"language": "fra", "title": "VF"})},
    )
    assert g.outputs[0].metadata == {"language": "fra", "title": "VF"}


def test_provenance_survives_the_where_trim() -> None:
    g = _lower(
        "SELECT a.audio[1] FROM input('x.mp4') a WHERE a.t BETWEEN 0 AND 1",
        {"a": _probe_result(audio_tags={"language": "fra"})},
    )
    assert g.outputs[0].metadata == {"language": "fra"}


def test_amix_keeps_provenance_both_inputs_agree_on() -> None:
    """A multi-stream call is a join like concat -- it threads the
    tag when every stream feeding it agrees, so mixing two English tracks
    yields an English track."""
    g = _lower(
        "SELECT amix(a.audio[1], a.audio[2]) FROM input('x.mp4') a",
        {
            "a": _probe_result(
                audios=2, per_audio_tags=[{"language": "eng"}, {"language": "eng"}]
            )
        },
    )
    assert _filters(g) == ["amix"]
    assert g.outputs[0].metadata == {"language": "eng"}


def test_amix_drops_provenance_its_two_inputs_disagree_on() -> None:
    """f.audio[1]=eng mixed with f.audio[2]=fra: two stream inputs that say
    different things have nothing in common to thread."""
    g = _lower(
        "SELECT amix(a.audio[1], a.audio[2]) FROM input('x.mp4') a",
        {
            "a": _probe_result(
                audios=2, per_audio_tags=[{"language": "eng"}, {"language": "fra"}]
            )
        },
    )
    assert _filters(g) == ["amix"]
    assert g.outputs[0].metadata == {}


def test_broadcast_through_amix_keeps_each_pairs_agreed_language() -> None:
    """The flagship shape: two 2-track sources tagged eng/fra alike, zipped by
    amix -- each pair keeps the language both sides of it agree on."""
    g = _lower(
        "SELECT amix(a.audio, b.audio) FROM input('x.mp4') a, input('y.mp4') b",
        {
            "a": _probe_result(
                audios=2, per_audio_tags=[{"language": "eng"}, {"language": "fra"}]
            ),
            "b": _probe_result(
                audios=2, per_audio_tags=[{"language": "eng"}, {"language": "fra"}]
            ),
        },
    )
    assert [o.metadata for o in g.outputs] == [{"language": "eng"}, {"language": "fra"}]


def test_broadcast_through_amix_drops_every_elements_provenance() -> None:
    g = _lower(
        "SELECT amix(a.audio, b.audio) FROM input('x.mp4') a, input('y.mp4') b",
        {
            "a": _probe_result(
                audios=2, per_audio_tags=[{"language": "eng"}, {"language": "fra"}]
            ),
            "b": _probe_result(audios=2, audio_tags={"language": "deu"}),
        },
    )
    assert [o.metadata for o in g.outputs] == [{}, {}]


def test_concat_keeps_provenance_every_segment_agrees_on() -> None:
    """Both segments of the pad say `language=eng`, so the pad still does."""
    g = _lower(
        "SELECT a.audio[1] FROM input('x.mp4') a "
        "UNION ALL SELECT b.audio[1] FROM input('y.mp4') b",
        {
            "a": _probe_result(audio_tags={"language": "eng"}),
            "b": _probe_result(audio_tags={"language": "eng"}),
        },
    )
    assert g.outputs[0].metadata == {"language": "eng"}


def test_concat_drops_provenance_the_segments_disagree_on() -> None:
    """An English segment concatenated with a French one is neither."""
    g = _lower(
        "SELECT a.audio[1] FROM input('x.mp4') a "
        "UNION ALL SELECT b.audio[1] FROM input('y.mp4') b",
        {
            "a": _probe_result(audio_tags={"language": "eng"}),
            "b": _probe_result(audio_tags={"language": "fra"}),
        },
    )
    assert g.outputs[0].metadata == {}


def test_concat_drops_provenance_when_one_segment_has_none() -> None:
    """Agreement is on what SURVIVES the und-filter: an untagged segment has
    nothing to say, so the tagged one does not speak for the whole output."""
    g = _lower(
        "SELECT a.audio[1] FROM input('x.mp4') a "
        "UNION ALL SELECT b.audio[1] FROM input('y.mp4') b",
        {
            "a": _probe_result(audio_tags={"language": "eng"}),
            "b": _probe_result(audio_tags={"language": "und"}),
        },
    )
    assert g.outputs[0].metadata == {}


def test_concat_of_two_untagged_segments_carries_no_metadata() -> None:
    """Two `und` streams filter down to {} apiece: they agree on nothing, and
    an empty agreement is still no metadata."""
    g = _lower(
        "SELECT a.audio[1] FROM input('x.mp4') a "
        "UNION ALL SELECT b.audio[1] FROM input('y.mp4') b",
        {
            "a": _probe_result(audio_tags={"language": "und"}),
            "b": _probe_result(audio_tags={"language": "und"}),
        },
    )
    assert g.outputs[0].metadata == {}


def test_concat_drops_provenance_when_a_segment_was_not_probed() -> None:
    g = _lower(
        "SELECT a.audio[1] FROM input('x.mp4') a "
        "UNION ALL SELECT b.audio[1] FROM input('y.mp4') b",
        {"a": _probe_result(audio_tags={"language": "eng"}), "b": None},
    )
    assert g.outputs[0].metadata == {}


def test_concat_agrees_per_pad_over_flattened_array_columns() -> None:
    """Splatted arrays pair elementwise, and each pad agrees (or not) on its
    own: track 1 matches eng/eng, track 2 is fra against deu."""
    g = _lower(
        "SELECT a.audio FROM input('x.mp4') a UNION ALL SELECT b.audio FROM input('y.mp4') b",
        {
            "a": _probe_result(
                audios=2, per_audio_tags=[{"language": "eng"}, {"language": "fra"}]
            ),
            "b": _probe_result(
                audios=2, per_audio_tags=[{"language": "eng"}, {"language": "deu"}]
            ),
        },
    )
    assert [o.metadata for o in g.outputs] == [{"language": "eng"}, {}]


def test_concat_provenance_survives_a_filtered_segment() -> None:
    """Provenance reaches the concat through each branch's own 1:1 chain."""
    g = _lower(
        "SELECT volume(a.audio[1], 0.5) FROM input('x.mp4') a "
        "UNION ALL SELECT b.audio[1] FROM input('y.mp4') b",
        {
            "a": _probe_result(audio_tags={"language": "fra", "title": "VF"}),
            "b": _probe_result(audio_tags={"language": "fra", "title": "VF"}),
        },
    )
    assert g.outputs[0].metadata == {"language": "fra", "title": "VF"}


def test_concat_after_amix_carries_no_metadata() -> None:
    """`a`'s two mixed tracks disagree (eng vs fra), so the amix segment has
    no source at all and the pad cannot agree with the other one whatever it
    says."""
    g = _lower(
        "SELECT amix(a.audio[1], a.audio[2]) FROM input('x.mp4') a "
        "UNION ALL SELECT b.audio[1] FROM input('y.mp4') b",
        {
            "a": _probe_result(
                audios=2, per_audio_tags=[{"language": "eng"}, {"language": "fra"}]
            ),
            "b": _probe_result(audio_tags={"language": "eng"}),
        },
    )
    assert g.outputs[0].metadata == {}


def test_concat_after_agreeing_amix_keeps_the_shared_language() -> None:
    """`a`'s two mixed tracks agree (eng and eng), so the amix
    segment threads that language into the concat pad, and the other segment
    agrees too -- the tag survives two joins deep."""
    g = _lower(
        "SELECT amix(a.audio[1], a.audio[2]) FROM input('x.mp4') a "
        "UNION ALL SELECT b.audio[1] FROM input('y.mp4') b",
        {
            "a": _probe_result(
                audios=2, per_audio_tags=[{"language": "eng"}, {"language": "eng"}]
            ),
            "b": _probe_result(audio_tags={"language": "eng"}),
        },
    )
    assert g.outputs[0].metadata == {"language": "eng"}


# ---------------------------------------------------------------------------
# WHERE on an INPUT alias -> input-level seek
#
# An input alias owns its own -i, so its window becomes Graph.input_trims and
# emit renders -ss/-to in front of that -i. No filter node is spliced and the
# stream refs come out of lowering UNCHANGED -- which is what makes a trimmed
# passthrough (stream copy) possible. The filter trim survives for CTE names
# only; those tests live in the CTE section below.
# ---------------------------------------------------------------------------


def test_where_between_seeks_the_input_and_leaves_the_video_ref_alone() -> None:
    g = _lower("SELECT hflip(a.video[1]) FROM input('x.mp4') a WHERE a.t BETWEEN 1 AND 2.5")
    assert g.to_dict()["nodes"] == [
        {
            "id": "n1",
            "filter": "hflip",
            "args": {},
            "inputs": ["src:a:v:0"],
            "outputs": ["video"],
        },
    ]
    assert g.input_trims == {"a": (1, 2.5)}
    assert g.to_dict()["input_trims"] == {"a": [1, 2.5]}
    assert _outputs(g) == [("n1", "video", None)]


def test_where_between_seeks_the_input_for_audio_too() -> None:
    """No atrim/asetpts pair: one seek covers every stream type of the input."""
    g = _lower("SELECT a.audio[1] FROM input('x.mp4') a WHERE a.t BETWEEN 1 AND 2")
    assert g.to_dict()["nodes"] == []
    assert g.input_trims == {"a": (1, 2)}
    assert _outputs(g) == [("src:a:a:0", "audio", None)]


def test_a_trimmed_column_that_nothing_filters_stays_a_passthrough() -> None:
    """The new capability: a WHERE no longer forces a re-encode.

    The ref is still a source ref, so emit maps it bare and stream-copies it,
    with the window carried as input options on the -i.
    """
    g = compile_sql("SELECT a.video[1] FROM input('x.mp4') a WHERE a.t BETWEEN 5 AND 60")
    assert g.nodes == {}
    assert _outputs(g) == [("src:a:v:0", "video", None)]
    emitted = emit(g)
    assert emitted.filter_complex == ""
    assert build_ffmpeg_args(emitted, "out.mp4") == [
        "ffmpeg",
        "-ss",
        "5",
        "-to",
        "60",
        "-i",
        "x.mp4",
        "-map",
        "0:v:0",
        "-c:0",
        "copy",
        "out.mp4",
    ]


def test_one_predicate_seeks_video_and_audio_in_sync() -> None:
    g = _lower(
        "SELECT a.video[1], a.audio[1] FROM input('x.mp4') a WHERE a.t BETWEEN 5 AND 10"
    )
    assert _filters(g) == []
    assert g.input_trims == {"a": (5, 10)}
    assert _outputs(g) == [("src:a:v:0", "video", None), ("src:a:a:0", "audio", None)]


def test_one_window_serves_every_consumer_of_that_input() -> None:
    g = _lower(
        "SELECT overlay(a.video[1], a.video[1], 5, 5) FROM input('x.mp4') a "
        "WHERE a.t BETWEEN 0 AND 3"
    )
    assert _filters(g) == ["overlay"]
    assert g.input_trims == {"a": (0, 3)}
    assert g.nodes["n1"].inputs == ["src:a:v:0", "src:a:v:0"]  # both arms, pre-split


def test_where_seeks_only_the_named_alias() -> None:
    g = _lower(
        "SELECT overlay(a.video[1], b.video[1], 0, 0) "
        "FROM input('x.mp4') a, input('y.mp4') b WHERE b.t BETWEEN 2 AND 4"
    )
    assert g.input_trims == {"b": (2, 4)}
    assert g.nodes["n1"].inputs == ["src:a:v:0", "src:b:v:0"]


def test_two_between_clauses_seek_both_inputs() -> None:
    g = _lower(
        "SELECT overlay(a.video[1], b.video[1], 0, 0) FROM input('x.mp4') a, input('y.mp4') b "
        "WHERE a.t BETWEEN 0 AND 1 AND b.t BETWEEN 2 AND 3"
    )
    assert _filters(g) == ["overlay"]
    assert g.input_trims == {"a": (0, 1), "b": (2, 3)}
    assert g.nodes["n1"].inputs == ["src:a:v:0", "src:b:v:0"]


def test_each_windowed_alias_gets_its_own_i_in_the_argv() -> None:
    """Two inputs, two windows: each -ss/-to sits in front of its own -i."""
    emitted = emit(
        compile_sql(
            "SELECT overlay(a.video[1], b.video[1], 0, 0) "
            "FROM input('x.mp4') a, input('y.mp4') b "
            "WHERE a.t BETWEEN 0 AND 1 AND b.t BETWEEN 2.5 AND 3"
        )
    )
    assert emitted.input_trims == [(0, 1), (2.5, 3)]
    args = build_ffmpeg_args(emitted, "out.mp4")
    assert args[:11] == [
        "ffmpeg",
        "-ss",
        "0",
        "-to",
        "1",
        "-i",
        "x.mp4",
        "-ss",
        "2.5",
        "-to",
        "3",
    ]
    assert args[11:13] == ["-i", "y.mp4"]


def test_union_all_branches_seek_their_own_inputs() -> None:
    """Per-alias windows survive a UNION ALL: one -i each, concat unchanged."""
    g = compile_sql(
        "SELECT a.video[1] FROM input('x.mp4') a WHERE a.t BETWEEN 0 AND 1 "
        "UNION ALL "
        "SELECT b.video[1] FROM input('y.mp4') b WHERE b.t BETWEEN 2 AND 3"
    )
    assert _filters(g) == ["concat"]
    assert g.nodes["n1"].inputs == ["src:a:v:0", "src:b:v:0"]
    assert g.input_trims == {"a": (0, 1), "b": (2, 3)}
    assert emit(g).input_trims == [(0, 1), (2, 3)]


def test_an_input_window_is_probe_independent() -> None:
    """The bounds are pure numbers from the SQL, not anything probing supplies."""
    query = "SELECT a.video[1] FROM input('x.mp4') a WHERE a.t BETWEEN 1.5 AND 4"
    assert compile_sql(query).input_trims == {"a": (1.5, 4)}


def test_the_seek_covers_the_whole_input_selected_or_not() -> None:
    """A seek is an input option: unselected streams are seeked too (harmlessly
    -- they are never -mapped), and no filter node is created for anything."""
    g = _lower(
        "SELECT a.video[1] FROM input('x.mp4') a WHERE a.t BETWEEN 0 AND 1",
        {"a": _probe_result(videos=1, audios=2)},
    )
    assert _filters(g) == []
    assert g.input_trims == {"a": (0, 1)}
    assert _outputs(g) == [("src:a:v:0", "video", None)]


# ---------------------------------------------------------------------------
# row-bounded windows with no fan-out TO: one -i per row, gathered
# ---------------------------------------------------------------------------


def _windowed_probe(layout: str = "a", duration: float = 12.0) -> ProbeResult:
    """`layout`'s tracks, plus a container duration for `f.duration` to read."""
    return ProbeResult(streams=_layout_probe(layout).streams, duration=duration)


def _gathered(bounds: str, count: int = 3) -> str:
    """One clip per series row, gathered into a single mix."""
    return (
        "COPY (WITH shots AS ("
        f"SELECT f.audio AS clip FROM input('x.mp4') f, generate_series(1, {count}) i "
        f"WHERE {bounds}) "
        "SELECT amix(VARIADIC array_agg(shots.clip)) FROM shots) TO 'o.mka'"
    )


def test_a_series_bounded_window_gives_every_row_its_own_input() -> None:
    """Three rows, three windows, three `-i` of the same file in one graph --
    and the streams reaching the aggregate come one from each."""
    g = _lower(
        _gathered("f.t >= (i.i - 1) * 2 AND f.t <= (i.i - 1) * 2 + 1"),
        {"f": _windowed_probe()},
    )
    assert g.input_paths == ["x.mp4"] * 3
    assert list(g.input_trims.values()) == [(0, 1), (2, 3), (4, 5)]
    assert _filters(g) == ["amix"]
    assert len(set(g.nodes["n1"].inputs)) == 3


def test_two_rows_naming_the_same_window_share_one_input() -> None:
    """One `-i` per distinct window, not per row: a repeated window is the same
    seek of the same file, so the split pass shares its decode."""
    g = _lower(
        _gathered("f.t >= 0 AND f.t <= i.i * 0 + 1"), {"f": _windowed_probe()}
    )
    assert g.input_paths == ["x.mp4"]
    assert g.input_trims == {"f": (0, 1)}


def test_a_row_bounded_window_reads_the_input_probe_through_every_copy() -> None:
    """The copies are the same file, so `f.duration` and the track layout
    answer for all of them -- only the seek differs."""
    g = _lower(
        _gathered("f.t >= f.duration - i.i AND f.t <= f.duration"),
        {"f": _windowed_probe(duration=10.0)},
    )
    assert list(g.input_trims.values()) == [(9.0, 10.0), (8.0, 10.0), (7.0, 10.0)]


def _shots_by_series(count: int, length: int) -> str:
    """The gathered form: one series row per shot, each bounding its own seek."""
    spacing = f"(f.duration - {length}) * (i.i - 1) / ({count} - 1)"
    return _gathered(f"f.t >= {spacing} AND f.t <= {spacing} + {length}", count)


def _shots_by_hand(count: int, length: int) -> str:
    """The same thing written out: one input() per shot, one literal window
    each, and the gather spelled as a positional argument list -- which has to
    say its own count, since only VARIADIC writes one from the array."""
    froms = ", ".join(f"input('x.mp4') f{k}" for k in range(count))
    args = ", ".join(f"f{k}.audio[1]" for k in range(count)) + f", inputs => {count}"
    where = " AND ".join(
        f"f{k}.t >= (f{k}.duration - {length}) * {k / (count - 1)} "
        f"AND f{k}.t <= (f{k}.duration - {length}) * {k / (count - 1)} + {length}"
        for k in range(count)
    )
    return f"COPY (SELECT amix({args}) FROM {froms} WHERE {where}) TO 'o.mka'"


def test_the_gathered_form_compiles_to_the_hand_written_one_byte_for_byte() -> None:
    """The decisive one: five series rows bounding five seeks of one file
    produce the very argv five hand-copied inputs do -- same `-i` list in the
    same order, same windows, same filtergraph."""
    probes: dict[str, ProbeResult | None] = {"f": _windowed_probe()}
    probes.update({f"f{k}": _windowed_probe() for k in range(5)})
    gathered = insert_splits(_lower(_shots_by_series(5, 2), probes))
    by_hand = insert_splits(_lower(_shots_by_hand(5, 2), probes))
    assert build_ffmpeg_args(emit(gathered), "o.mka") == build_ffmpeg_args(
        emit(by_hand), "o.mka"
    )
    assert list(gathered.input_trims.values()) == [
        (0.0, 2.0),
        (2.5, 4.5),
        (5.0, 7.0),
        (7.5, 9.5),
        (10.0, 12.0),
    ]


def _gathered_by_array(bounds: str, count: int = 3) -> str:
    """The ``ARRAY(SELECT ...)`` form of :func:`_gathered` -- no CTE, no
    ``array_agg`` written by hand."""
    return (
        "COPY (SELECT amix(VARIADIC ARRAY("
        f"SELECT f.audio FROM input('x.mp4') f, generate_series(1, {count}) i "
        f"WHERE {bounds}))) TO 'o.mka'"
    )


def test_array_gather_variadic_compiles_the_same_as_array_agg_over_a_cte() -> None:
    """``ARRAY(SELECT ...)`` inside VARIADIC is sugar for a hidden row source
    plus ``array_agg`` -- byte for byte the same argv as writing that CTE by
    hand (the recipe 75 shape)."""
    bounds = "f.t >= (i.i - 1) * 2 AND f.t <= (i.i - 1) * 2 + 1"
    probes: dict[str, ProbeResult | None] = {"f": _windowed_probe()}
    gathered = insert_splits(_lower(_gathered_by_array(bounds), probes))
    by_hand = insert_splits(_lower(_gathered(bounds), probes))
    assert build_ffmpeg_args(emit(gathered), "o.mka") == build_ffmpeg_args(
        emit(by_hand), "o.mka"
    )


def test_row_bounded_windows_reaching_one_file_name_both_ways_out() -> None:
    """Nothing gathers the rows and nothing fans them out, so three windows
    meet one destination: rejected, with the aggregate and the TO expression
    both named."""
    err = _reject_lower(
        "COPY (SELECT f.audio FROM input('x.mp4') f, generate_series(1, 3) i "
        "WHERE f.t >= i.i - 1 AND f.t <= i.i) TO 'o.mka'",
        {"f": _windowed_probe()},
    )
    assert err.code is ErrorCode.ROW_COUNT_MISMATCH
    assert "this query has 3 rows" in err.message
    assert err.hint is not None
    assert "array_agg" in err.hint and "TO expression" in err.hint


def test_a_cte_column_cannot_carry_a_row_bounded_window() -> None:
    """A CTE name is a filtergraph pad: there is no `-i` to mint one of per
    row, so the window has to name the input the streams came from."""
    err = _reject_lower(
        "COPY (WITH c AS (SELECT f.audio[1] AS v FROM input('x.mp4') f) "
        "SELECT c.v FROM c, generate_series(1, 2) i "
        "WHERE c.t >= i.i - 1 AND c.t <= i.i) TO 'o.mka'",
        {"f": _windowed_probe()},
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "has no input to seek per row" in err.message


_MOTION_THUMBNAIL = """
COPY (
  WITH shots AS (
    SELECT ffmpeg.concat(VARIADIC f.video) AS frame
    FROM input('{src}') f, generate_series(1, {count}) i
    WHERE f.t >= (f.duration - {length}) * (i.i - 1) / ({count} - 1)
      AND f.t <= (f.duration - {length}) * (i.i - 1) / ({count} - 1) + {length}
  ),
  small AS (SELECT fps(scale(shots.frame, 480, -2), 5) AS frame FROM shots)
  SELECT paletteuse(small.frame, palettegen(small.frame))
  FROM small
) TO 'out.gif'
"""


def _motion_thumbnail_by_hand(src: str, count: int, length: int) -> str:
    branches = " UNION ALL ".join(
        f"SELECT v{k} AS frame FROM input('{src}') f{k}, unnest(f{k}.video) v{k} "
        f"WHERE v{k}.index = 1 "
        f"AND f{k}.t >= (f{k}.duration - {length}) * {k / (count - 1)} "
        f"AND f{k}.t <= (f{k}.duration - {length}) * {k / (count - 1)} + {length}"
        for k in range(count)
    )
    return (
        f"COPY (WITH shots AS ({branches}), "
        "small AS (SELECT fps(scale(shots.frame, 480, -2), 5) AS frame FROM shots) "
        "SELECT paletteuse(small.frame, palettegen(small.frame)) FROM small"
        ") TO 'out.gif'"
    )


@pytest.mark.exec
def test_the_motion_thumbnail_gather_matches_its_hand_written_form() -> None:
    """The same identity over the real thing: `concat` has a variable pad count
    and only a live ffmpeg reports its options, so the shape the registry
    recipe is written in is checked here rather than against the snapshot."""
    src = (FIXTURES_DIR / "av.mp4").as_posix()
    gathered = compile_sql(_MOTION_THUMBNAIL.format(src=src, count=5, length=1))
    by_hand = compile_sql(_motion_thumbnail_by_hand(src, 5, 1))
    assert build_ffmpeg_args(emit(gathered), "out.gif") == build_ffmpeg_args(
        emit(by_hand), "out.gif"
    )
    assert len(gathered.input_paths) == 5


# ---------------------------------------------------------------------------
# open-ended input windows: >= / <=, either operand order, merging
# ---------------------------------------------------------------------------


def test_tail_only_where_seeks_with_no_upper_bound() -> None:
    g = _lower("SELECT a.video[1] FROM input('x.mp4') a WHERE a.t >= 5")
    assert g.input_trims == {"a": (5, None)}
    emitted = emit(g)
    assert emitted.input_trims == [(5, None)]
    assert build_ffmpeg_args(emitted, "out.mp4")[:4] == ["ffmpeg", "-ss", "5", "-i"]


def test_head_only_where_seeks_with_no_lower_bound() -> None:
    g = _lower("SELECT a.video[1] FROM input('x.mp4') a WHERE a.t <= 60")
    assert g.input_trims == {"a": (None, 60)}
    emitted = emit(g)
    assert emitted.input_trims == [(None, 60)]
    assert build_ffmpeg_args(emitted, "out.mp4")[:4] == ["ffmpeg", "-to", "60", "-i"]


def test_flipped_operand_order_produces_the_same_window() -> None:
    """``120 <= a.t`` is the mirror of ``a.t >= 120`` -- exact, not approximate."""
    g_unflipped = _lower("SELECT a.video[1] FROM input('x.mp4') a WHERE a.t >= 120")
    g_flipped = _lower("SELECT a.video[1] FROM input('x.mp4') a WHERE 120 <= a.t")
    assert g_unflipped.input_trims == g_flipped.input_trims == {"a": (120, None)}


def test_gte_and_lte_merge_into_the_same_window_as_between() -> None:
    g_inequalities = _lower("SELECT a.video[1] FROM input('x.mp4') a WHERE a.t >= 1 AND a.t <= 2")
    g_between = _lower("SELECT a.video[1] FROM input('x.mp4') a WHERE a.t BETWEEN 1 AND 2")
    assert g_inequalities.input_trims == g_between.input_trims == {"a": (1, 2)}


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT a.video[1] FROM input('x.mp4') a WHERE a.t >= 5 AND a.t <= 2",
        "SELECT a.video[1] FROM input('x.mp4') a WHERE a.t BETWEEN 5 AND 2",
    ],
)
def test_empty_time_window_is_rejected(sql: str) -> None:
    err = _reject(sql)
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "empty time window" in err.message


# ---------------------------------------------------------------------------
# CTEs
# ---------------------------------------------------------------------------


def test_cte_columns_are_reachable_by_their_as_names() -> None:
    g = _lower(
        "WITH c AS ("
        "  SELECT hflip(a.video[1]) AS pic, volume(a.audio[1], 0.5) AS snd "
        "  FROM input('x.mp4') a"
        ") SELECT c.snd, vflip(c.pic) FROM c"
    )
    assert _filters(g) == ["hflip", "volume", "vflip"]
    assert _outputs(g) == [("n2", "audio", None), ("n3", "video", None)]


def test_unknown_cte_column_lists_the_known_names() -> None:
    err = _reject(
        "WITH c AS (SELECT hflip(a.video[1]) AS pic, a.audio[1] AS snd "
        "FROM input('x.mp4') a) SELECT c.nope FROM c"
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "c.nope" in err.message
    assert err.hint is not None and "pic" in err.hint and "snd" in err.hint


def test_a_scalar_cte_column_cannot_be_subscripted() -> None:
    err = _reject(
        "WITH c AS (SELECT a.audio[1] AS snd FROM input('x.mp4') a) "
        "SELECT c.snd[1] FROM c"
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "'c.snd' is a single stream" in err.message


# ---------------------------------------------------------------------------
# CTE array columns: splat, broadcast again, subscript
# ---------------------------------------------------------------------------


def test_cte_array_column_splats_in_the_outer_select() -> None:
    g = _lower(
        "WITH c AS (SELECT aecho(a.audio, 0.8, 0.9, 60, 0.3) AS snd FROM input('x.mp4') a) "
        "SELECT c.snd FROM c",
        {"a": _probe_result(audios=2)},
    )
    assert _filters(g) == ["aecho", "aecho"]
    # the CTE's own AS name stays inside the CTE; the outer SELECT names the
    # output columns (here: not at all).
    assert _outputs(g) == [("n1", "audio", None), ("n2", "audio", None)]


def test_cte_array_column_broadcasts_again() -> None:
    g = _lower(
        "WITH c AS (SELECT a.audio AS snd FROM input('x.mp4') a) "
        "SELECT volume(c.snd, 0.5) FROM c",
        {"a": _probe_result(audios=2)},
    )
    assert _filters(g) == ["volume", "volume"]
    assert g.nodes["n1"].inputs == ["src:a:a:0"]
    assert g.nodes["n2"].inputs == ["src:a:a:1"]


def test_cte_array_column_is_subscriptable_one_based() -> None:
    g = _lower(
        "WITH c AS (SELECT aecho(a.audio, 0.8, 0.9, 60, 0.3) AS snd FROM input('x.mp4') a) "
        "SELECT c.snd[2] FROM c",
        {"a": _probe_result(audios=2)},
    )
    # both elements are still built (the CTE lowered as a whole); only the
    # subscripted one reaches the output list.
    assert _filters(g) == ["aecho", "aecho"]
    assert _outputs(g) == [("n2", "audio", None)]


def test_cte_array_subscript_bounds_are_static() -> None:
    """The length was recorded when the CTE lowered: no probe is consulted here."""
    err = _reject_lower(
        "WITH c AS (SELECT a.audio AS snd FROM input('x.mp4') a) "
        "SELECT c.snd[3] FROM c",
        {"a": _probe_result(audios=2)},
    )
    assert err.code is ErrorCode.STREAM_NOT_FOUND
    assert "'c.snd[3]' does not exist" in err.message
    assert "column 'c.snd' has 2 streams" in err.message


def test_cte_array_column_provenance_reaches_the_outer_output() -> None:
    g = _lower(
        "WITH c AS (SELECT aecho(a.audio, 0.8, 0.9, 60, 0.3) AS snd FROM input('x.mp4') a) "
        "SELECT c.snd FROM c",
        {
            "a": _probe_result(
                audios=2, per_audio_tags=[{"language": "eng"}, {"language": "fra"}]
            )
        },
    )
    assert [o.metadata for o in g.outputs] == [{"language": "eng"}, {"language": "fra"}]


def test_an_unnamed_cte_column_has_no_name_to_read() -> None:
    """A CTE exposes what its body named with AS, and nothing else."""
    err = _reject_lower(
        "WITH c AS (SELECT a.video FROM input('x.mp4') a) SELECT hflip(c.v) FROM c",
        {"a": _probe_result(videos=2)},
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "c.v" in err.message
    assert err.hint is not None and "no named columns" in err.hint


def test_where_trims_a_cte_column_by_its_type() -> None:
    """A CTE name is a filtergraph pad, not an -i, so its window stays a FILTER
    trim (the one surviving use of trim/atrim)."""
    g = _lower(
        "WITH c AS (SELECT a.audio[1] AS snd FROM input('x.mp4') a) "
        "SELECT c.snd FROM c WHERE c.t BETWEEN 1 AND 2"
    )
    assert _filters(g) == ["atrim", "asetpts"]
    assert g.nodes["n1"].inputs == ["src:a:a:0"]
    assert g.nodes["n1"].args == {"start": 1, "end": 2}
    assert g.input_trims == {}  # nothing is seeked: the -i is untouched


def test_cte_open_lower_trim_node_carries_only_start() -> None:
    """A CTE trim with only one bound omits the other's arg entirely."""
    g = _lower(
        "WITH c AS (SELECT hflip(a.video[1]) AS pic FROM input('x.mp4') a) "
        "SELECT c.pic FROM c WHERE c.t >= 3"
    )
    assert _filters(g) == ["hflip", "trim", "setpts"]
    assert g.nodes["n2"].filter == "trim"
    assert g.nodes["n2"].args == {"start": 3}


def test_cte_open_upper_trim_node_carries_only_end() -> None:
    g = _lower(
        "WITH c AS (SELECT a.audio[1] AS snd FROM input('x.mp4') a) "
        "SELECT c.snd FROM c WHERE c.t <= 4"
    )
    assert _filters(g) == ["atrim", "asetpts"]
    assert g.nodes["n1"].args == {"end": 4}


def test_a_where_inside_a_cte_body_still_seeks_the_input() -> None:
    """The window is on `a`, an input alias, even though it is written inside a
    CTE body -- aliases are globally unique, so the seek is a graph property."""
    g = _lower(
        "WITH c AS ("
        "  SELECT hflip(a.video[1]) AS pic FROM input('x.mp4') a WHERE a.t BETWEEN 1 AND 2"
        ") SELECT vflip(c.pic) FROM c"
    )
    assert _filters(g) == ["hflip", "vflip"]
    assert g.input_trims == {"a": (1, 2)}


def test_cte_union_all_gets_its_own_concat() -> None:
    g = _lower(
        "WITH u AS ("
        "  SELECT a.video[1] AS v FROM input('x.mp4') a"
        "  UNION ALL SELECT b.video[1] AS v FROM input('y.mp4') b"
        ") SELECT hflip(u.v) FROM u"
    )
    assert _filters(g) == ["concat", "hflip"]
    assert g.nodes["n1"].inputs == ["src:a:v:0", "src:b:v:0"]
    assert g.nodes["n1"].outputs == ["video"]
    assert g.nodes["n2"].inputs == ["n1"]


def test_a_cte_body_unnests_its_input_and_filters_the_rows() -> None:
    """A row table inside a CTE body: the WHERE picks rows at compile time and
    the survivors are the streams the outer SELECT maps."""
    g = _lower(
        "WITH tracks AS ("
        "  SELECT t AS track FROM input('x.mp4') f, unnest(f.audio) t"
        "  WHERE t.tags.language = 'fra'"
        ") SELECT tracks.track FROM tracks",
        {
            "f": _probe_result(
                per_audio_tags=[{"language": "eng"}, {"language": "fra"}]
            )
        },
    )
    assert _filters(g) == []
    assert _outputs(g) == [("src:f:a:1", "audio", None)]


# ---------------------------------------------------------------------------
# UNION ALL -> concat
# ---------------------------------------------------------------------------


def test_union_all_video_only_lowers_to_one_concat() -> None:
    g = compile_sql(
        "SELECT a.video[1] FROM input('x.mp4') a "
        "UNION ALL SELECT hflip(b.video[1]) FROM input('y.mp4') b "
        "UNION ALL SELECT c.video[1] FROM input('z.mp4') c"
    )
    concat = g.nodes["n2"]
    assert concat.filter == "concat"
    assert concat.args == {"n": 3, "v": 1, "a": 0}
    assert concat.inputs == ["src:a:v:0", "n1", "src:c:v:0"]
    assert concat.outputs == ["video"]
    assert _outputs(g) == [("n2", "video", None)]


def test_union_all_with_audio_interleaves_inputs_per_segment() -> None:
    g = compile_sql(
        "SELECT a.video[1], a.audio[1] FROM input('intro.mp4') a "
        "UNION ALL SELECT b.video[1], b.audio[1] FROM input('main.mp4') b"
    )
    concat = g.nodes["n1"]
    assert concat.args == {"n": 2, "v": 1, "a": 1}
    # per ffmpeg: [seg1 v][seg1 a][seg2 v][seg2 a]
    assert concat.inputs == ["src:a:v:0", "src:a:a:0", "src:b:v:0", "src:b:a:0"]
    assert concat.outputs == ["video", "audio"]
    assert _outputs(g) == [("n1:0", "video", None), ("n1:1", "audio", None)]


def test_concat_pads_map_back_to_the_branch_column_order() -> None:
    """Pads are videos-then-audios; the SELECT list keeps its own order."""
    g = compile_sql(
        "SELECT a.audio[1], a.video[1] FROM input('x.mp4') a "
        "UNION ALL SELECT b.audio[1], b.video[1] FROM input('y.mp4') b"
    )
    concat = g.nodes["n1"]
    assert concat.inputs == ["src:a:v:0", "src:a:a:0", "src:b:v:0", "src:b:a:0"]
    assert concat.outputs == ["video", "audio"]
    assert _outputs(g) == [("n1:1", "audio", None), ("n1:0", "video", None)]


def test_union_all_keeps_the_first_branch_column_names() -> None:
    g = compile_sql(
        "SELECT a.video[1] AS pic FROM input('x.mp4') a "
        "UNION ALL SELECT b.video[1] AS other FROM input('y.mp4') b"
    )
    assert [o.name for o in g.outputs] == ["pic"]


def test_union_all_type_mismatch_is_a_concat_mismatch() -> None:
    err = _reject(
        "SELECT a.video[1] FROM input('x.mp4') a\n"
        "UNION ALL SELECT b.audio[1] FROM input('y.mp4') b"
    )
    assert err.code is ErrorCode.CONCAT_MISMATCH
    assert err.line == 2, "the mismatching branch is the anchor"
    assert "branch 2" in err.message


def test_union_all_column_count_mismatch_is_a_concat_mismatch() -> None:
    err = _reject(
        "SELECT a.video[1], a.audio[1] FROM input('x.mp4') a\n"
        "UNION ALL SELECT b.video[1] FROM input('y.mp4') b"
    )
    assert err.code is ErrorCode.CONCAT_MISMATCH
    assert err.line == 2


def test_union_all_flattens_array_columns_into_concat_columns() -> None:
    g = _lower(
        "SELECT a.audio FROM input('x.mp4') a UNION ALL SELECT b.audio FROM input('y.mp4') b",
        {"a": _probe_result(audios=2), "b": _probe_result(audios=2)},
    )
    concat = g.nodes["n1"]
    assert concat.args == {"n": 2, "v": 0, "a": 2}
    assert concat.inputs == ["src:a:a:0", "src:a:a:1", "src:b:a:0", "src:b:a:1"]
    assert concat.outputs == ["audio", "audio"]
    assert _outputs(g) == [("n1:0", "audio", None), ("n1:1", "audio", None)]


def test_union_all_element_count_mismatch_is_a_concat_mismatch() -> None:
    """Same types, different array lengths: the flattened signatures differ."""
    err = _reject_lower(
        "SELECT a.audio FROM input('x.mp4') a\n"
        "UNION ALL SELECT b.audio FROM input('y.mp4') b",
        {"a": _probe_result(audios=2), "b": _probe_result(audios=1)},
    )
    assert err.code is ErrorCode.CONCAT_MISMATCH
    assert err.line == 2
    assert "branch 1 selects (audio[2])" in err.message
    assert "branch 2 selects (audio[1])" in err.message


def test_union_all_column_order_mismatch_is_a_concat_mismatch() -> None:
    err = _reject(
        "SELECT a.video[1], a.audio[1] FROM input('x.mp4') a\n"
        "UNION ALL SELECT b.audio[1], b.video[1] FROM input('y.mp4') b"
    )
    assert err.code is ErrorCode.CONCAT_MISMATCH


# ---------------------------------------------------------------------------
# function calls
# ---------------------------------------------------------------------------


def test_nested_calls_chain_bottom_up() -> None:
    g = _lower("SELECT gblur(hflip(vflip(a.video[1])), 4) FROM input('x.mp4') a")
    assert _filters(g) == ["vflip", "hflip", "gblur"]
    assert g.nodes["n1"].inputs == ["src:a:v:0"]
    assert g.nodes["n2"].inputs == ["n1"]
    assert g.nodes["n3"].inputs == ["n2"]
    assert _outputs(g) == [("n3", "video", None)]


def test_audio_calls_chain_bottom_up() -> None:
    g = _lower("SELECT volume(aecho(a.audio[1], 0.8, 0.9, 60, 0.3), 0.5) FROM input('x.mp4') a")
    assert _filters(g) == ["aecho", "volume"]
    assert g.nodes["n1"].inputs == ["src:a:a:0"]
    assert g.nodes["n1"].outputs == ["audio"]
    assert _outputs(g) == [("n2", "audio", None)]


def test_function_lookup_is_case_insensitive() -> None:
    g = _lower("SELECT SCALE(a.video[1], 0.5) FROM input('x.mp4') a")
    assert _filters(g) == ["scale"]


def test_negative_numeric_literals_survive() -> None:
    g = _lower("SELECT scale(a.video[1], -2, 720) FROM input('x.mp4') a")
    assert g.nodes["n1"].args == {"width": -2, "height": 720}


def test_string_literal_argument() -> None:
    g = _lower("SELECT drawbox(a.video[1], 1, 2, 3, 4, 'red') FROM input('x.mp4') a")
    assert g.nodes["n1"].args["color"] == "red"


def test_unknown_function_suggests_a_close_match() -> None:
    err = _reject("SELECT scal(a.video[1], 0.5) FROM input('x.mp4') a")
    assert err.code is ErrorCode.UNKNOWN_FUNCTION
    assert "scal()" in err.message
    assert err.hint is not None and "scale()" in err.hint


def test_unknown_function_without_a_match_names_the_filter_set() -> None:
    err = _reject("SELECT zzzz(a.video[1]) FROM input('x.mp4') a")
    assert err.code is ErrorCode.UNKNOWN_FUNCTION
    assert err.hint is not None and "filter of your installed ffmpeg" in err.hint



def test_a_call_with_no_options_at_all_is_legal() -> None:
    """Every option has an ffmpeg default, so passing none is a complete call
    -- there is no arity to satisfy beyond the pad signature."""
    g = _lower("SELECT scale(a.video[1]) FROM input('x.mp4') a")
    assert g.nodes["n1"].filter == "scale"
    assert g.nodes["n1"].args == {}


def test_argument_kind_mismatch_is_the_options_own_rejection() -> None:
    """A badly-typed POSITIONAL is an option problem,
    reported against the option it landed on -- not a signature mismatch."""
    err = _reject("SELECT gblur(a.video[1], 'lots') FROM input('x.mp4') a")
    assert err.code is ErrorCode.FILTER_OPTION_TYPE
    assert "'sigma' of filter 'gblur' expects a number, got 'lots'" in err.message


def test_audio_stream_where_video_is_expected() -> None:
    err = _reject("SELECT hflip(a.audio[1]) FROM input('x.mp4') a")
    assert err.code is ErrorCode.UDF_ARG_TYPE
    assert "hflip() is an ffmpeg filter" in err.message
    assert "it takes video as its stream input, got (audio)" in err.message


def test_video_stream_where_audio_is_expected() -> None:
    err = _reject("SELECT amix(a.video[1], a.audio[1]) FROM input('x.mp4') a")
    assert err.code is ErrorCode.UDF_ARG_TYPE
    assert "its stream inputs are all audio" in err.message
    assert "(video, audio)" in err.message


def test_video_result_where_audio_is_expected() -> None:
    """The kind of a nested call comes from its output PAD type."""
    err = _reject("SELECT volume(hflip(a.video[1]), 2) FROM input('x.mp4') a")
    assert err.code is ErrorCode.UDF_ARG_TYPE
    assert "it takes audio as its stream input, got (video)" in err.message


def test_stream_argument_where_an_option_value_is_expected() -> None:
    err = _reject("SELECT gblur(a.video[1], hflip(a.video[1])) FROM input('x.mp4') a")
    assert err.code is ErrorCode.FILTER_OPTION_TYPE
    assert "'sigma' of filter 'gblur' takes a value, got a video stream" in err.message


def test_an_unknown_nested_call_in_an_option_slot_still_names_it() -> None:
    """Classifying the argument first is what keeps the typo readable."""
    err = _reject("SELECT gblur(a.video[1], nope(a.video[1])) FROM input('x.mp4') a")
    assert err.code is ErrorCode.UNKNOWN_FUNCTION
    assert "nope()" in err.message


def test_a_null_scalar_argument_drops_the_option() -> None:
    """NULL is absence: the option is not written and ffmpeg's default stands."""
    g = _lower("SELECT gblur(a.video[1], NULL) FROM input('x.mp4') a")
    assert g.nodes["n1"].args == {}


def test_arithmetic_scalar_argument_is_folded() -> None:
    g = _lower("SELECT gblur(a.video[1], 1 + 2) FROM input('x.mp4') a")
    assert g.nodes["n1"].args["sigma"] == 3


def test_malformed_numeric_literal_is_a_typed_rejection() -> None:
    """sqlglot tokenizes `1e` as a number but ``to_py()`` raises on it."""
    err = _reject("SELECT gblur(a.video[1], 1e) FROM input('x.mp4') a")
    assert err.code is ErrorCode.FILTER_OPTION_TYPE
    assert "1e" in err.message


def test_malformed_between_bound_is_a_typed_rejection() -> None:
    err = _reject("SELECT a.video[1] FROM input('x.mp4') a WHERE a.t BETWEEN 1e AND 2")
    assert err.code is ErrorCode.UNSUPPORTED_SQL


def test_overlay_keeps_its_four_positional_arguments() -> None:
    """Postgres has a builtin OVERLAY, so sqlglot hands lower named args."""
    g = _lower(
        "SELECT overlay(a.video[1], b.video[1], 20, 30) FROM input('x.mp4') a, input('y.mp4') b"
    )
    node = g.nodes["n1"]
    assert node.filter == "overlay"
    assert node.args == {"x": 20, "y": 30}
    assert node.inputs == ["src:a:v:0", "src:b:v:0"]


def test_overlay_takes_its_options_positionally_despite_the_builtin() -> None:
    """Postgres claims the NAME `overlay`, so `=>` inside it is a parse error --
    but positional options work, and x/y are ffmpeg's first two."""
    g = _lower(
        "SELECT overlay(a.video[1], b.video[1], 20, 30) FROM input('x.mp4') a, input('y.mp4') b"
    )
    assert g.nodes["n1"].args == {"x": 20, "y": 30}


def test_a_partial_overlay_option_list_is_still_legal() -> None:
    g = _lower(
        "SELECT overlay(a.video[1], b.video[1], 20) FROM input('x.mp4') a, input('y.mp4') b"
    )
    assert g.nodes["n1"].args == {"x": 20}


def test_overlay_keeps_the_agreed_video_tag() -> None:
    """overlay is a multi-stream join exactly like amix -- when both
    probed video streams it composites agree on a tag, the composite keeps
    it. (Use the same file under two aliases, same as the README headline.)"""
    g = _lower(
        "SELECT overlay(a.video[1], b.video[1], 0, 0) FROM input('x.mp4') a, input('y.mp4') b",
        {
            "a": _probe_result(video_tags={"language": "eng"}),
            "b": _probe_result(video_tags={"language": "eng"}),
        },
    )
    assert g.outputs[0].metadata == {"language": "eng"}


def test_overlay_drops_provenance_its_two_inputs_disagree_on() -> None:
    g = _lower(
        "SELECT overlay(a.video[1], b.video[1], 0, 0) FROM input('x.mp4') a, input('y.mp4') b",
        {
            "a": _probe_result(video_tags={"language": "eng"}),
            "b": _probe_result(video_tags={"language": "fra"}),
        },
    )
    assert g.outputs[0].metadata == {}


def test_overlay_drops_provenance_when_one_side_is_unprobed() -> None:
    """One input could not be probed at all, so it has no source to agree
    with the other -- same rule an unprobed concat segment follows."""
    g = _lower(
        "SELECT overlay(a.video[1], b.video[1], 0, 0) FROM input('x.mp4') a, input('y.mp4') b",
        {"a": _probe_result(video_tags={"language": "eng"}), "b": None},
    )
    assert g.outputs[0].metadata == {}


def test_a_colliding_builtin_is_an_unknown_function() -> None:
    """`lower` is a Postgres builtin sqlglot parses into its own Func class, and
    it is neither a stdlib function nor (in any ffmpeg) a filter name."""
    err = _reject("SELECT lower(a.video[1]) FROM input('x.mp4') a")
    assert err.code is ErrorCode.UNKNOWN_FUNCTION


# ---------------------------------------------------------------------------
# probing: bounds, provenance, symbolic fallback
# ---------------------------------------------------------------------------


def test_probed_subscript_out_of_range_is_stream_not_found() -> None:
    with pytest.raises(FfrwdError) as excinfo:
        _lower(
            "SELECT a.audio[2] FROM input('x.mp4') a",
            {"a": _probe_result(videos=1, audios=1)},
        )
    err = excinfo.value
    assert err.code is ErrorCode.STREAM_NOT_FOUND
    assert err.line == 1 and err.col is not None
    assert "a.audio[2]" in err.message
    assert "x.mp4" in err.message


def test_a_probed_video_subscript_is_bounds_checked_too() -> None:
    with pytest.raises(FfrwdError) as excinfo:
        _lower("SELECT a.video[1] FROM input('x.mp4') a", {"a": _probe_result(videos=0)})
    assert excinfo.value.code is ErrorCode.STREAM_NOT_FOUND


def test_probed_subscript_in_range_lowers_normally() -> None:
    g = _lower(
        "SELECT a.audio[2] FROM input('x.mp4') a", {"a": _probe_result(audios=2)}
    )
    assert _outputs(g) == [("src:a:a:1", "audio", None)]


def test_unprobed_subscript_is_symbolic_and_out_of_range_is_fine() -> None:
    """No probe -> no bounds check; ffmpeg validates at runtime."""
    g = compile_sql("SELECT a.audio[9] FROM input('does-not-exist.mp4') a")
    assert _outputs(g) == [("src:a:a:8", "audio", None)]


def test_passthrough_output_copies_language_and_title() -> None:
    g = _lower(
        "SELECT a.audio[1] FROM input('x.mp4') a",
        {"a": _probe_result(audio_tags={"language": "fra", "title": "VF"})},
    )
    assert g.outputs[0].metadata == {"language": "fra", "title": "VF"}


def test_undefined_language_tag_is_not_copied() -> None:
    """mp4 muxers stamp language=und on untagged streams; it says nothing."""
    g = _lower(
        "SELECT a.audio[1] FROM input('x.mp4') a",
        {"a": _probe_result(audio_tags={"language": "und", "title": "Main"})},
    )
    assert g.outputs[0].metadata == {"title": "Main"}


def test_filtered_output_threads_its_sources_provenance() -> None:
    """A 1:1 filter chain keeps the tags raw ffmpeg would have lost."""
    g = _lower(
        "SELECT volume(a.audio[1], 0.5) FROM input('x.mp4') a",
        {"a": _probe_result(audio_tags={"language": "fra"})},
    )
    assert g.outputs[0].metadata == {"language": "fra"}


def test_unprobed_passthrough_has_no_metadata() -> None:
    g = compile_sql("SELECT a.audio[1] FROM input('x.mp4') a")
    assert g.outputs[0].metadata == {}


# ---------------------------------------------------------------------------
# subtitle / data columns -- same surface, passthrough-only
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("sql", "expected"),
    [
        ("SELECT a.subtitle[1] FROM input('x.mkv') a", ("src:a:s:0", "subtitle")),
        ("SELECT a.subtitle[2] FROM input('x.mkv') a", ("src:a:s:1", "subtitle")),
        ("SELECT a.data[1] FROM input('x.mkv') a", ("src:a:d:0", "data")),
    ],
)
def test_subtitle_and_data_subscripts_lower_to_s_and_d_refs(
    sql: str, expected: tuple[str, str]
) -> None:
    g = _lower(sql, {"a": _layout_probe("vassdd")})
    assert _outputs(g) == [(expected[0], expected[1], None)]
    assert g.nodes == {}  # passthrough-only: never a filtergraph node


def test_bare_subtitle_array_splats_like_any_other() -> None:
    g = _lower("SELECT a.subtitle FROM input('x.mkv') a", {"a": _layout_probe("vass")})
    assert _outputs(g) == [
        ("src:a:s:0", "subtitle", None),
        ("src:a:s:1", "subtitle", None),
    ]


def test_bare_data_array_splats_like_any_other() -> None:
    g = _lower("SELECT a.data FROM input('x.mkv') a", {"a": _layout_probe("vdd")})
    assert _outputs(g) == [
        ("src:a:d:0", "data", None),
        ("src:a:d:1", "data", None),
    ]


def test_bare_subtitle_array_over_an_unprobed_input_is_input_not_found() -> None:
    err = _reject("SELECT a.subtitle FROM input('does-not-exist.mkv') a")
    assert err.code is ErrorCode.INPUT_NOT_FOUND


def test_subtitle_subscript_is_bounds_checked_when_probed() -> None:
    err = _reject_lower(
        "SELECT a.subtitle[2] FROM input('x.mkv') a", {"a": _layout_probe("vas")}
    )
    assert err.code is ErrorCode.STREAM_NOT_FOUND
    assert "a.subtitle[2]" in err.message
    assert "1 subtitle stream" in err.message


def test_an_empty_stream_array_warns_and_contributes_nothing() -> None:
    """A column the file has no tracks for selects nothing, as unnest does."""
    said: list[FfrwdWarning] = []
    g = _lower(
        "SELECT a.video, a.subtitle FROM input('x.mkv') a",
        {"a": _layout_probe("va")},
        on_warning=said.append,
    )
    assert [o.type for o in g.outputs] == ["video"]
    assert [w.code for w in said] == [WarningCode.EMPTY_STREAM_ARRAY]
    assert "no subtitle streams" in said[0].message


def test_a_sink_left_with_no_streams_is_still_a_typed_error() -> None:
    """One empty column is a warning; every column empty is not."""
    err = _reject_lower(
        "COPY (SELECT a.subtitle FROM input('x.mkv') a) TO 'o.mkv'",
        {"a": _layout_probe("va")},
    )
    assert err.code is ErrorCode.STREAM_NOT_FOUND
    assert "would have no streams" in err.message


def test_unprobed_subtitle_subscript_stays_symbolic() -> None:
    g = compile_sql("SELECT a.subtitle[3] FROM input('does-not-exist.mkv') a")
    assert _outputs(g) == [("src:a:s:2", "subtitle", None)]


def test_subtitle_passthrough_carries_its_language_tag() -> None:
    """Provenance rides the SAME passthrough metadata path audio tags do."""
    g = _lower(
        "SELECT a.subtitle[1] FROM input('x.mkv') a",
        {"a": _layout_probe("vas", tags={2: {"language": "eng", "title": "English"}})},
    )
    assert g.outputs[0].metadata == {"language": "eng", "title": "English"}


def test_unknown_input_column_hint_lists_subtitle_and_data() -> None:
    err = _reject("SELECT a.captions FROM input('x.mkv') a")
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert err.hint is not None
    assert "subtitle" in err.hint and "data" in err.hint


# -- passthrough-only: no function ever takes one ---------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT scale(a.subtitle[1], 0.5) FROM input('x.mkv') a",
        "SELECT hflip(a.subtitle[1]) FROM input('x.mkv') a",
        "SELECT volume(a.data[1], 0.5) FROM input('x.mkv') a",
        "SELECT amix(a.audio[1], a.subtitle[1]) FROM input('x.mkv') a",
        # the array form too: broadcasting does not launder the type
        "SELECT hflip(a.subtitle) FROM input('x.mkv') a",
    ],
)
def test_stdlib_call_over_a_passthrough_stream_is_udf_arg_type(sql: str) -> None:
    err = _reject_lower(sql, {"a": _layout_probe("vasd")})
    assert err.code is ErrorCode.UDF_ARG_TYPE
    assert "cannot be filtered, only selected" in err.message


def test_dynamic_call_over_a_subtitle_stream_is_udf_arg_type_not_internal(
    _registry: Registry,
) -> None:
    """Tier 2's pad signature only ever holds video/audio, so a subtitle
    argument must produce the SAME typed rejection, never an INTERNAL."""
    err = _reject_dyn(
        "SELECT gblur(a.subtitle[1], sigma => 5) FROM input('x.mkv') a",
        _registry,
        {"a": _layout_probe("vas")},
    )
    assert err.code is ErrorCode.UDF_ARG_TYPE
    assert "cannot be filtered, only selected" in err.message


def test_a_cte_may_carry_a_subtitle_column_through_as_passthrough() -> None:
    g = _lower(
        "WITH c AS (SELECT a.video[1] AS v, a.subtitle[1] AS caps "
        "FROM input('x.mkv') a) "
        "SELECT c.v, c.caps FROM c",
        {"a": _layout_probe("vas", tags={2: {"language": "eng"}})},
    )
    assert _outputs(g) == [
        ("src:a:v:0", "video", None),
        ("src:a:s:0", "subtitle", None),
    ]
    assert g.outputs[1].metadata == {"language": "eng"}


def test_filtering_a_cte_subtitle_column_is_still_rejected() -> None:
    err = _reject_lower(
        "WITH c AS (SELECT a.subtitle[1] AS caps FROM input('x.mkv') a) "
        "SELECT hflip(c.caps) FROM c",
        {"a": _layout_probe("vas")},
    )
    assert err.code is ErrorCode.UDF_ARG_TYPE


# -- passthrough-only: an INPUT seek trims them; a CTE's filter trim cannot


def test_where_over_a_consumed_subtitle_stream_is_rejected() -> None:
    """Measured 2026-08-15: ffmpeg does NOT retime caption packets under an
    input -ss (copy or transcode), so a seeked-and-selected caption track
    would desync by the seek amount. Rejected rather than shipped broken."""
    err = _reject_lower(
        "SELECT a.subtitle[1] FROM input('x.mkv') a WHERE a.t BETWEEN 1 AND 2",
        {"a": _layout_probe("vas")},
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "out of sync" in err.message


def test_where_over_a_consumed_subtitle_stream_is_rejected_for_an_open_window() -> None:
    """The desync rejection keys on `graph.input_trims` membership,
    so an open-ended window (`t >= x`, no upper bound at all) triggers it just
    as a closed BETWEEN does -- there is no window shape that is safe."""
    err = _reject_lower(
        "SELECT a.subtitle[1] FROM input('x.mkv') a WHERE a.t >= 1",
        {"a": _layout_probe("vas")},
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "out of sync" in err.message


def test_where_over_a_consumed_data_stream_is_rejected() -> None:
    err = _reject_lower(
        "SELECT a.data FROM input('x.mkv') a WHERE a.t BETWEEN 1 AND 2",
        {"a": _layout_probe("vad")},
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "data" in err.message


def test_star_plus_where_over_a_captioned_input_is_rejected() -> None:
    """`SELECT *` consumes the caption track, so the same desync rejection
    applies; trim + star works on caption-less files (covered elsewhere)."""
    err = _reject_lower(
        "SELECT * FROM input('x.mkv') a WHERE a.t BETWEEN 1 AND 2",
        {"a": _layout_probe("vas")},
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "out of sync" in err.message


def test_where_plus_unselected_captions_still_seeks() -> None:
    """The trim stays legal when the captions are NOT selected: unmapped
    streams are seeked harmlessly."""
    g = _lower(
        "SELECT a.video[1], a.audio[1] FROM input('x.mkv') a WHERE a.t BETWEEN 1 AND 2",
        {"a": _layout_probe("vas")},
    )
    assert _filters(g) == []
    assert g.input_trims == {"a": (1, 2)}
    assert _outputs(g) == [
        ("src:a:v:0", "video", None),
        ("src:a:a:0", "audio", None),
    ]


def test_where_over_a_cte_carrying_a_subtitle_column_is_rejected() -> None:
    """Permanent: a CTE trim is a filtergraph trim."""
    err = _reject_lower(
        "WITH c AS (SELECT a.video[1] AS v, a.subtitle[1] AS caps "
        "FROM input('x.mkv') a) "
        "SELECT c.v, c.caps FROM c WHERE c.t BETWEEN 1 AND 2",
        {"a": _layout_probe("vas")},
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "a CTE's captions cannot be trimmed" in err.message
    assert err.hint is not None and "external subtitle file" in err.hint


def test_where_that_does_not_touch_the_caption_alias_still_seeks_its_own_input() -> None:
    """A window is per alias: b's caption track is mapped whole, a is seeked."""
    g = _lower(
        "SELECT a.video[1], b.subtitle[1] FROM input('x.mkv') a, input('y.mkv') b "
        "WHERE a.t BETWEEN 1 AND 2",
        {"a": _layout_probe("vas"), "b": _layout_probe("vas")},
    )
    assert _filters(g) == []
    assert g.input_trims == {"a": (1, 2)}
    assert _outputs(g) == [("src:a:v:0", "video", None), ("src:b:s:0", "subtitle", None)]


def test_a_captioned_input_may_be_trimmed_when_captions_are_not_selected() -> None:
    g = _lower(
        "SELECT a.video[1] FROM input('x.mkv') a WHERE a.t BETWEEN 1 AND 2",
        {"a": _layout_probe("vas")},
    )
    assert _filters(g) == []
    assert g.input_trims == {"a": (1, 2)}


# -- passthrough-only: concat has no s/d pads -------------------------------


def test_union_all_branch_with_a_subtitle_column_is_rejected() -> None:
    err = _reject_lower(
        "SELECT a.video[1], a.subtitle[1] FROM input('x.mkv') a "
        "UNION ALL "
        "SELECT b.video[1], b.subtitle[1] FROM input('y.mkv') b",
        {"a": _layout_probe("vas"), "b": _layout_probe("vas")},
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "video and audio only" in err.message


# ---------------------------------------------------------------------------
# SELECT * and <alias>.*
# ---------------------------------------------------------------------------


def test_bare_star_expands_every_stream_of_the_container() -> None:
    g = _lower("SELECT * FROM input('x.mkv') a", {"a": _layout_probe("vasd")})
    assert _outputs(g) == [
        ("src:a:v:0", "video", None),
        ("src:a:a:0", "audio", None),
        ("src:a:s:0", "subtitle", None),
        ("src:a:d:0", "data", None),
    ]
    assert g.nodes == {}  # a star is pure passthrough


def test_star_expands_the_stream_arrays_in_column_order() -> None:
    """The container's array columns, not file order: audio-first stays v/a/s."""
    g = _lower("SELECT * FROM input('x.mkv') a", {"a": _layout_probe("asv")})
    assert [(o.ref, o.type) for o in g.outputs] == [
        ("src:a:v:0", "video"),
        ("src:a:a:0", "audio"),
        ("src:a:s:0", "subtitle"),
    ]


def test_bare_star_covers_every_from_alias_in_from_order() -> None:
    g = _lower(
        "SELECT * FROM input('x.mkv') a, input('y.mkv') b",
        {"a": _layout_probe("va"), "b": _layout_probe("vs")},
    )
    assert _outputs(g) == [
        ("src:a:v:0", "video", None),
        ("src:a:a:0", "audio", None),
        ("src:b:v:0", "video", None),
        ("src:b:s:0", "subtitle", None),
    ]


def test_qualified_star_covers_one_alias_and_mixes_with_other_columns() -> None:
    g = _lower(
        "SELECT a.*, b.audio[1] FROM input('x.mkv') a, input('y.mkv') b",
        {"a": _layout_probe("vas"), "b": _layout_probe("vaa")},
    )
    assert _outputs(g) == [
        ("src:a:v:0", "video", None),
        ("src:a:a:0", "audio", None),
        ("src:a:s:0", "subtitle", None),
        ("src:b:a:0", "audio", None),
    ]


def test_a_star_may_follow_an_explicit_column() -> None:
    g = _lower(
        "SELECT b.video[1], a.* FROM input('x.mkv') a, input('y.mkv') b",
        {"a": _layout_probe("as"), "b": _layout_probe("v")},
    )
    assert _outputs(g) == [
        ("src:b:v:0", "video", None),
        ("src:a:a:0", "audio", None),
        ("src:a:s:0", "subtitle", None),
    ]


def test_star_carries_provenance_metadata_per_stream() -> None:
    g = _lower(
        "SELECT * FROM input('x.mkv') a",
        {
            "a": _layout_probe(
                "vas",
                tags={
                    1: {"language": "fra"},
                    2: {"language": "eng", "title": "English"},
                },
            )
        },
    )
    assert [o.metadata for o in g.outputs] == [
        {},
        {"language": "fra"},
        {"language": "eng", "title": "English"},
    ]


def test_star_over_an_unprobed_input_is_input_not_found() -> None:
    err = _reject("SELECT * FROM input('does-not-exist.mkv') a")
    assert err.code is ErrorCode.INPUT_NOT_FOUND
    assert "does-not-exist.mkv" in err.message


def test_qualified_star_over_an_unprobed_input_is_input_not_found() -> None:
    err = _reject_lower(
        "SELECT b.*, a.video[1] FROM input('x.mkv') a, input('y.mkv') b",
        {"a": _layout_probe("v"), "b": None},
    )
    assert err.code is ErrorCode.INPUT_NOT_FOUND


def test_star_over_a_stream_less_input_is_a_typed_error() -> None:
    """An attachment-only container would expand to nothing at all."""
    err = _reject_lower("SELECT * FROM input('x.mkv') a", {"a": _layout_probe("")})
    assert err.code is ErrorCode.STREAM_NOT_FOUND
    assert "selects nothing" in err.message


def test_star_over_a_cte_expands_its_columns_statically() -> None:
    """No probe is consulted: the CTE's shape was fixed when its body lowered."""
    g = _lower(
        "WITH c AS (SELECT a.video[1] AS v, a.subtitle[1] AS caps "
        "FROM input('x.mkv') a) "
        "SELECT c.* FROM c",
        {"a": _layout_probe("vas")},
    )
    assert _outputs(g) == [
        ("src:a:v:0", "video", "v"),
        ("src:a:s:0", "subtitle", "caps"),
    ]


def test_star_over_a_cte_splats_its_array_columns() -> None:
    g = _lower(
        "WITH c AS (SELECT a.audio AS tracks FROM input('x.mkv') a) "
        "SELECT c.* FROM c",
        {"a": _layout_probe("vaa")},
    )
    assert _outputs(g) == [
        ("src:a:a:0", "audio", "tracks"),
        ("src:a:a:1", "audio", "tracks"),
    ]


def test_a_star_inside_a_cte_body_expands_there() -> None:
    g = _lower(
        "WITH c AS (SELECT * FROM input('x.mkv') a) SELECT c.* FROM c",
        {"a": _layout_probe("vas")},
    )
    assert [(o.ref, o.type) for o in g.outputs] == [
        ("src:a:v:0", "video"),
        ("src:a:a:0", "audio"),
        ("src:a:s:0", "subtitle"),
    ]


def test_star_over_an_unknown_cte_or_alias_is_unknown_alias() -> None:
    err = _reject_lower(
        "SELECT nope.* FROM input('x.mkv') a", {"a": _layout_probe("v")}
    )
    assert err.code is ErrorCode.UNKNOWN_ALIAS


def test_star_output_is_still_stream_copied_end_to_end() -> None:
    """`SELECT *` is a remux: no filtergraph, one -map + -c copy per stream."""
    g = insert_splits(_lower("SELECT * FROM input('x.mkv') a", {"a": _layout_probe("vas")}))
    emitted = emit(g)
    assert emitted.filter_complex == ""
    args = build_ffmpeg_args(emitted, "out.mkv")
    assert args.count("-map") == 3
    assert [m.target for m in emitted.maps] == ["0:v:0", "0:a:0", "0:s:0"]


# ---------------------------------------------------------------------------
# SELECT * EXCEPT(...) / REPLACE(...)
# ---------------------------------------------------------------------------


def test_star_except_drops_the_named_kind() -> None:
    g = _lower(
        "SELECT * EXCEPT(subtitle) FROM input('x.mkv') a", {"a": _layout_probe("vasd")}
    )
    assert _outputs(g) == [
        ("src:a:v:0", "video", None),
        ("src:a:a:0", "audio", None),
        ("src:a:d:0", "data", None),
    ]


def test_star_except_drops_every_stream_of_a_repeated_kind() -> None:
    g = _lower(
        "SELECT * EXCEPT(audio) FROM input('x.mkv') a", {"a": _layout_probe("vaa")}
    )
    assert _outputs(g) == [("src:a:v:0", "video", None)]


def test_star_replace_substitutes_in_place_and_keeps_order() -> None:
    g = _lower(
        "SELECT * REPLACE(scale(a.video[1], 640, -2) AS video) "
        "FROM input('x.mkv') a",
        {"a": _layout_probe("vas")},
    )
    assert _filters(g) == ["scale"]
    assert [o.type for o in g.outputs] == ["video", "audio", "subtitle"]


def test_star_replace_argv_is_byte_identical_to_the_longhand_columns() -> None:
    """The whole point: `* REPLACE(...)` is shorthand, not a different
    compile path -- its argv must match writing every column out."""
    replaced_sql = (
        "SELECT * REPLACE(scale(a.video[1], 640, -2) AS video) "
        "FROM input('x.mkv') a"
    )
    longhand_sql = (
        "SELECT scale(a.video[1], 640, -2), a.audio[1], a.subtitle[1] "
        "FROM input('x.mkv') a"
    )
    probes = {"a": _layout_probe("vas")}
    replaced = emit(insert_splits(_lower(replaced_sql, probes)))
    longhand = emit(insert_splits(_lower(longhand_sql, probes)))
    assert build_ffmpeg_args(replaced, "out.mkv") == build_ffmpeg_args(longhand, "out.mkv")


def test_star_replace_composes_with_a_sink_and_its_with_options() -> None:
    """A `* REPLACE` result is an ordinary column list downstream: it feeds a
    sink, WITH options included, exactly like writing the columns out."""
    g = insert_splits(
        _lower(
            "COPY (SELECT * REPLACE(scale(a.video[1], 640, -2) AS video) "
            "FROM input('x.mkv') a) "
            "TO 'out.mp4' WITH (video_codec 'libx264', crf 20, subtitle_codec 'mov_text')",
            {"a": _layout_probe("vas")},
        )
    )
    args = build_ffmpeg_args(emit(g))
    assert "libx264" in args
    assert "mov_text" in args
    assert "20" in args


def test_star_except_on_a_file_lacking_that_kind_is_a_no_op() -> None:
    """Excepting a kind the file never had behaves exactly like the bare
    `SELECT *` a reader would otherwise have written."""
    probes = {"a": _layout_probe("va")}
    excepted = _outputs(
        _lower("SELECT * EXCEPT(subtitle) FROM input('x.mkv') a", probes)
    )
    bare = _outputs(_lower("SELECT * FROM input('x.mkv') a", probes))
    assert excepted == bare


def test_star_except_unknown_name_lists_what_the_star_holds() -> None:
    err = _reject_lower(
        "SELECT * EXCEPT(nope) FROM input('x.mkv') a", {"a": _layout_probe("va")}
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "'nope'" in err.message
    assert err.hint is not None
    assert "audio" in err.hint and "video" in err.hint


def test_star_replace_unknown_name_lists_what_the_star_holds() -> None:
    err = _reject_lower(
        "SELECT * REPLACE(a.video[1] AS nope) FROM input('x.mkv') a",
        {"a": _layout_probe("va")},
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "'nope'" in err.message


def test_star_replace_may_change_the_slot_kind() -> None:
    """No new rule invented: an ordinary aliased SELECT column already lets
    its name and its stream's real kind disagree, and REPLACE is the same
    substitution -- the position keeps the name, the value carries whatever
    kind the expression actually produces."""
    g = _lower(
        "SELECT * REPLACE(a.audio[1] AS video) FROM input('x.mkv') a",
        {"a": _layout_probe("va")},
    )
    assert [o.type for o in g.outputs] == ["audio", "audio"]


# ---------------------------------------------------------------------------
# compile_sql: probing policy
# ---------------------------------------------------------------------------


def _probe_calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, tuple[str, ...]]]:
    """Record every (path, flags) the compile probes, and probe nothing."""
    calls: list[tuple[str, tuple[str, ...]]] = []

    def counting_probe(
        path: str, args: Sequence[str] = (), *, forced_format: bool = False
    ) -> ProbeResult | None:
        calls.append((path, tuple(args)))
        return None

    monkeypatch.setattr(compiler, "probe_path", counting_probe)
    return calls


def test_compile_sql_probes_each_distinct_path_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _probe_calls(monkeypatch)
    compile_sql(
        "WITH pip AS (SELECT b.video[1] AS v FROM input('game.mp4') b) "
        "SELECT overlay(a.video[1], pip.v, 0, 0) FROM input('game.mp4') a, pip"
    )
    assert calls == [("game.mp4", ())]  # two aliases, one file, one probe


def test_compile_sql_probes_one_path_read_two_ways_twice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _probe_calls(monkeypatch)
    compile_sql(
        "SELECT a.video[1], b.video[1] "
        "FROM input('game.mp4') a, input('game.mp4', framerate => 30) b"
    )
    assert calls == [("game.mp4", ()), ("game.mp4", ("-framerate", "30"))]


def test_compile_sql_probes_identical_options_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _probe_calls(monkeypatch)
    compile_sql(
        "SELECT a.video[1], b.video[1] "
        "FROM input('game.mp4', framerate => 30) a, "
        "input('game.mp4', framerate => 30) b"
    )
    assert calls == [("game.mp4", ("-framerate", "30"))]


def test_compile_sql_probes_options_in_written_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _probe_calls(monkeypatch)
    compile_sql(
        "SELECT a.video[1] FROM input('video=cam', format => 'dshow', "
        "framerate => 30, video_size => '960x540') a"
    )
    assert calls == [
        ("video=cam", ("-f", "dshow", "-framerate", "30", "-video_size", "960x540"))
    ]


def test_compile_sql_tells_the_probe_a_demuxer_is_forced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    forced: list[bool] = []

    def recording_probe(
        path: str, args: Sequence[str] = (), *, forced_format: bool = False
    ) -> ProbeResult | None:
        forced.append(forced_format)
        return None

    monkeypatch.setattr(compiler, "probe_path", recording_probe)
    compile_sql("SELECT a.video[1] FROM input('testsrc2=size=8x8', format => 'lavfi') a")
    compile_sql("SELECT a.video[1] FROM input('game.mp4', framerate => 30) a")
    assert forced == [True, False]


def test_compile_sql_uses_probe_results_for_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(compiler, "probe_path", lambda path, args=(), **kw: _probe_result(audios=1))
    err = _reject("SELECT a.audio[4] FROM input('x.mp4') a")
    assert err.code is ErrorCode.STREAM_NOT_FOUND


def test_compile_sql_drops_ffmpeg_only_options_from_the_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reported bug: `realtime => true` used to reach ffprobe, which
    rejects `-re` outright on the measured build, so the probe silently
    became None and `unnest` misreported a readable file as unreadable.
    Every ffmpeg-only option here (`realtime`, `stream_loop`, `hwaccel`,
    `itsoffset`, `seek_end`) must be gone from what the probe sees;
    `framerate` (demuxer-shaping) is the one that survives."""
    calls = _probe_calls(monkeypatch)
    compile_sql(
        "SELECT f.video[1] FROM input('src10.mp4', realtime => true, "
        "stream_loop => 2, hwaccel => 'cuda', itsoffset => 1, "
        "seek_end => 3, framerate => 30) f"
    )
    assert calls == [("src10.mp4", ("-framerate", "30"))]


def test_realtime_input_still_unnests_once_the_probe_sees_filtered_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The exact reported repro. The fake here only answers a probe called
    with the FILTERED flags (empty, since `realtime` is the only option) --
    proving `realtime` no longer reaches ffprobe, so `unnest` sees a real
    result instead of misreporting the readable file as unreadable."""
    def probe_only_the_filtered_call(
        path: str, args: Sequence[str] = (), *, forced_format: bool = False
    ) -> ProbeResult | None:
        return _layout_probe("v") if args == () else None

    monkeypatch.setattr(compiler, "probe_path", probe_only_the_filtered_call)
    g = compile_sql(
        "COPY (SELECT v FROM input('src10.mp4', realtime => true) f, "
        "unnest(f.video) v) TO 'out.mp4'"
    )
    assert len(g.sinks) == 1


# ---------------------------------------------------------------------------
# node ids, the pipeline, the backstop
# ---------------------------------------------------------------------------


def test_node_ids_are_sequential_across_ctes_and_branches() -> None:
    g = _lower(
        "WITH c AS (SELECT hflip(a.video[1]) AS v FROM input('x.mp4') a) "
        "SELECT vflip(c.v) FROM c "
        "UNION ALL SELECT gblur(b.video[1], 2) FROM input('y.mp4') b"
    )
    assert list(g.nodes) == ["n1", "n2", "n3", "n4"]
    assert _filters(g) == ["hflip", "vflip", "gblur", "concat"]


def test_compile_sql_runs_the_split_pass() -> None:
    sql = "SELECT overlay(a.video[1], a.video[1], 5, 5) FROM input('x.mp4') a"
    assert "split" not in _filters(_lower(sql))
    assert "split" in _filters(compile_sql(sql))


def test_split_pass_picks_asplit_for_audio() -> None:
    g = compile_sql("SELECT amix(a.audio[1], a.audio[1]) FROM input('x.mp4') a")
    assert "asplit" in _filters(g)


def test_compile_sql_wraps_unexpected_exceptions_as_internal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(res: object, probes: object, **kwargs: object) -> list[Graph]:
        raise ValueError("kaboom")

    monkeypatch.setattr(compiler, "lower_commands", boom)
    with pytest.raises(FfrwdError) as excinfo:
        compile_sql("SELECT a.video[1] FROM input('x.mp4') a")
    assert excinfo.value.code is ErrorCode.INTERNAL
    assert "kaboom" in excinfo.value.message


def test_lower_wraps_unexpected_exceptions_as_internal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(*args: object, **kwargs: object) -> None:
        raise ValueError("kaboom")

    monkeypatch.setattr(lower_module._Lowerer, "run", boom)
    with pytest.raises(FfrwdError) as excinfo:
        _lower("SELECT a.video[1] FROM input('x.mp4') a")
    assert excinfo.value.code is ErrorCode.INTERNAL


@pytest.mark.exec
def test_pipeline_output_survives_a_round_trip_through_dicts(_fixtures: None) -> None:
    g = compile_sql(_readme_flagship_sql())
    assert Graph.from_dict(g.to_dict()).to_dict() == g.to_dict()


# ---------------------------------------------------------------------------
# COPY ... TO ... WITH (...) -- the sink
# ---------------------------------------------------------------------------

SINK_QUERY = "SELECT a.video[1] FROM input('x.mp4') a"


def _sink_of(options: str) -> dict[str, object]:
    """The lowered sink of `SINK_QUERY` wrapped in a COPY with `options`."""
    with_clause = f" WITH ({options})" if options else ""
    g = _lower(f"COPY ({SINK_QUERY}) TO 'out.mkv'{with_clause}")
    assert len(g.sinks) == 1
    assert g.sinks[0].path == "out.mkv"
    return g.sinks[0].options


def test_bare_select_lowers_to_one_pathless_sink() -> None:
    g = _lower(SINK_QUERY)
    assert len(g.sinks) == 1
    assert g.sinks[0].path is None
    assert g.sinks[0].options == {}
    assert _serialized_sinks(g.to_dict())[0]["path"] is None


def test_copy_lowers_to_a_sink() -> None:
    assert _sink_of("video_codec 'libx264', crf 20, faststart true") == {
        "video_codec": "libx264",
        "crf": 20,
        "faststart": True,
    }


def test_sink_options_keep_their_written_order() -> None:
    assert list(_sink_of("crf 20, preset 'slow', audio_codec 'aac'")) == [
        "crf",
        "preset",
        "audio_codec",
    ]


def test_copy_without_options_lowers_to_an_empty_sink() -> None:
    assert _sink_of("") == {}


@pytest.mark.parametrize("written, value", [("true", True), ("false", False)])
def test_faststart_takes_a_bool(written: str, value: bool) -> None:
    assert _sink_of(f"faststart {written}") == {"faststart": value}


def test_sink_option_values_are_normalized_python_scalars() -> None:
    """The IR only ever carries str/int/bool -- no sqlglot nodes, no Decimal."""
    options = _sink_of("video_codec 'libx264', sample_rate 48000, faststart true")
    assert [type(v) for v in options.values()] == [str, int, bool]


def test_sink_survives_the_split_pass_and_serializes() -> None:
    g = compile_sql(
        "COPY (SELECT a.video[1], a.video[1] FROM input('x.mp4') a) "
        "TO 'out.mp4' WITH (video_codec 'libx264', crf 18, faststart true)"
    )
    # the projection is used twice, so the split pass rebuilt the graph
    assert "split" in _filters(g)
    unit = _serialized_sinks(g.to_dict())[0]
    assert unit["path"] == "out.mp4"
    assert unit["options"] == {
        "video_codec": "libx264",
        "crf": 18,
        "faststart": True,
    }
    assert Graph.from_dict(g.to_dict()).to_dict() == g.to_dict()


def test_unknown_sink_option_is_anchored_on_its_own_line() -> None:
    err = _reject(
        f"COPY (\n  {SINK_QUERY}\n) TO 'out.mkv' WITH (\n"
        f"  crf 20,\n  bogus_option 'x'\n)"
    )
    assert err.code is ErrorCode.UNKNOWN_SINK_OPTION
    assert "unknown sink option 'bogus_option'" in err.message
    assert err.line == 5


def test_unknown_sink_option_suggests_the_near_miss() -> None:
    err = _reject(f"COPY ({SINK_QUERY}) TO 'out.mkv' WITH (crff 20)")
    assert err.code is ErrorCode.UNKNOWN_SINK_OPTION
    assert err.hint == "did you mean 'crf'?"


@pytest.mark.parametrize(
    "options, message",
    [
        ("crf 'high'", "expects an int, got 'high'"),
        ("crf 1.5", "expects an int, got 1.5"),
        ("crf true", "expects an int, got True"),
        ("faststart 1", "expects a bool, got 1"),
        ("faststart 'yes'", "expects a bool, got 'yes'"),
        # a bare word and a double-quoted word are neither a string nor a bool
        ("preset slow", "expects a str, got the bare word slow"),
        ('preset "slow"', 'expects a str, got the identifier "slow"'),
        ("video_codec 42", "expects a str, got 42"),
        ("video_codec true", "expects a str, got True"),
    ],
)
def test_bad_sink_option_value_is_a_type_error(options: str, message: str) -> None:
    err = _reject(f"COPY ({SINK_QUERY}) TO 'out.mkv' WITH ({options})")
    assert err.code is ErrorCode.SINK_OPTION_TYPE
    assert message in err.message
    assert err.hint is not None


def test_an_invalid_inner_query_beats_the_sink_options() -> None:
    """The COPY wrapper never masks (or is masked by) the query's own errors."""
    err = _reject(
        "COPY (SELECT a.video[1] FROM input('x.mp4') a GROUP BY a.video[1]) "
        "TO 'out.mkv' WITH (bogus_option 1)"
    )
    assert err.code is ErrorCode.NO_STREAMING_EQUIVALENT

    err = _reject(
        "COPY (SELECT nosuchfilter(a.video[1]) FROM input('x.mp4') a) "
        "TO 'out.mkv' WITH (bogus_option 1)"
    )
    assert err.code is ErrorCode.UNKNOWN_FUNCTION


def test_sink_does_not_change_the_graph_shape() -> None:
    """Wrapping a query in a COPY adds a sink and touches nothing else."""
    plain = compile_sql(SINK_QUERY).to_dict()
    wrapped = compile_sql(f"COPY ({SINK_QUERY}) TO 'out.mkv' WITH (crf 20)").to_dict()
    unit = _serialized_sinks(wrapped)[0]
    assert unit["path"] == "out.mkv"
    assert unit["options"] == {"crf": 20}
    unit["path"] = None
    unit["options"] = {}
    assert wrapped == plain


def test_new_output_option_batch_lowers_in_written_order() -> None:
    assert _sink_of(
        "duration 30, max_size '10M', shortest true, maxrate '2675k', "
        "bufsize '5350k', gop 48, profile 'high', level '4.0', tune 'film', "
        "movflags '+faststart'"
    ) == {
        "duration": 30,
        "max_size": "10M",
        "shortest": True,
        "maxrate": "2675k",
        "bufsize": "5350k",
        "gop": 48,
        "profile": "high",
        "level": "4.0",
        "tune": "film",
        "movflags": "+faststart",
    }


def test_duration_accepts_a_fractional_number() -> None:
    assert _sink_of("duration 30.5") == {"duration": 30.5}


def test_codec_params_with_a_matching_video_codec() -> None:
    assert _sink_of("video_codec 'libx264', codec_params 'keyint=48'") == {
        "video_codec": "libx264",
        "codec_params": "keyint=48",
    }


def test_codec_params_without_video_codec_is_rejected() -> None:
    err = _reject(f"COPY ({SINK_QUERY}) TO 'out.mkv' WITH (codec_params 'keyint=48')")
    assert err.code is ErrorCode.SINK_OPTION_TYPE
    assert "codec_params" in err.message
    assert "video_codec" in err.message
    assert err.hint is not None
    assert "libx264" in err.hint


def test_codec_params_with_an_unsupported_video_codec_is_rejected() -> None:
    err = _reject(
        f"COPY ({SINK_QUERY}) TO 'out.mkv' WITH "
        "(video_codec 'libvpx-vp9', codec_params 'keyint=48')"
    )
    assert err.code is ErrorCode.SINK_OPTION_TYPE
    assert "libvpx-vp9" in err.message


def test_two_pass_lowers_with_a_codec_and_a_bitrate() -> None:
    assert _sink_of("video_codec 'libx264', video_bitrate '2500k', two_pass true") == {
        "video_codec": "libx264",
        "video_bitrate": "2500k",
        "two_pass": True,
    }


def test_two_pass_false_needs_nothing() -> None:
    """The rules guard the two-command shape, which `false` never asks for."""
    assert _sink_of("two_pass false") == {"two_pass": False}


def test_two_pass_without_video_bitrate_is_rejected() -> None:
    err = _reject(
        f"COPY ({SINK_QUERY}) TO 'out.mkv' WITH (video_codec 'libx264', two_pass true)"
    )
    assert err.code is ErrorCode.SINK_OPTION_TYPE
    assert "video_bitrate" in err.message
    assert err.hint is not None


@pytest.mark.parametrize("codec", ["", "video_codec 'libvpx-vp9', "])
def test_two_pass_without_a_pass_capable_codec_is_rejected(codec: str) -> None:
    err = _reject(
        f"COPY ({SINK_QUERY}) TO 'out.mkv' WITH "
        f"({codec}video_bitrate '2500k', two_pass true)"
    )
    assert err.code is ErrorCode.SINK_OPTION_TYPE
    assert "video_codec" in err.message
    assert err.hint is not None
    assert "libx264" in err.hint


def test_two_pass_with_crf_is_rejected() -> None:
    err = _reject(
        f"COPY ({SINK_QUERY}) TO 'out.mkv' WITH "
        "(video_codec 'libx264', video_bitrate '2500k', crf 20, two_pass true)"
    )
    assert err.code is ErrorCode.SINK_OPTION_TYPE
    assert "crf" in err.message
    assert "two_pass" in err.message


def test_two_pass_on_an_audio_only_copy_is_rejected() -> None:
    err = _reject(
        "COPY (SELECT a.audio[1] FROM input('x.mp4') a) TO 'out.m4a' WITH "
        "(video_codec 'libx264', video_bitrate '2500k', two_pass true)"
    )
    assert err.code is ErrorCode.SINK_OPTION_TYPE
    assert "video" in err.message


def test_two_pass_in_a_multi_copy_script_is_rejected() -> None:
    err = _reject(
        f"COPY ({SINK_QUERY}) TO 'one.mkv' WITH "
        "(video_codec 'libx264', video_bitrate '2500k', two_pass true); "
        "COPY (SELECT b.video[1] FROM input('x.mp4') b) TO 'two.mkv'"
    )
    assert err.code is ErrorCode.SINK_OPTION_TYPE
    assert "two_pass" in err.message
    assert err.hint is not None


def test_two_pass_is_fine_as_the_only_copy_of_a_script() -> None:
    g = _lower(
        f"COPY ({SINK_QUERY}) TO 'one.mkv' WITH "
        "(video_codec 'libx264', video_bitrate '2500k', two_pass true)"
    )
    assert g.sinks[0].options["two_pass"] is True


def test_faststart_and_movflags_together_are_rejected() -> None:
    err = _reject(
        f"COPY ({SINK_QUERY}) TO 'out.mkv' WITH "
        "(faststart true, movflags '+faststart')"
    )
    assert err.code is ErrorCode.SINK_OPTION_TYPE
    assert "faststart" in err.message
    assert "movflags" in err.message


def test_faststart_alone_is_still_fine() -> None:
    assert _sink_of("faststart true") == {"faststart": True}


def test_movflags_alone_is_still_fine() -> None:
    assert _sink_of("movflags '+faststart'") == {"movflags": "+faststart"}


# ---------------------------------------------------------------------------
# input() named options
# ---------------------------------------------------------------------------


def test_input_with_no_options_has_no_input_options_entry() -> None:
    g = _lower(SINK_QUERY)
    assert g.input_options == {}
    assert "input_options" not in g.to_dict()


def test_input_options_lower_to_normalized_scalars() -> None:
    g = _lower(
        "SELECT p.video[1] FROM input('logo.png', loop => true, framerate => 15) p"
    )
    assert g.input_options == {"p": {"loop": True, "framerate": 15}}
    assert [type(v) for v in g.input_options["p"].values()] == [bool, int]


def test_input_options_keep_their_written_order() -> None:
    g = _lower(
        "SELECT p.video[1] FROM input("
        "'logo.png', framerate => 15, hwaccel => 'cuda', loop => true"
        ") p"
    )
    assert list(g.input_options["p"]) == ["framerate", "hwaccel", "loop"]


def test_itsoffset_accepts_a_negative_number() -> None:
    g = _lower("SELECT a.video[1] FROM input('x.mp4', itsoffset => -1.5) a")
    assert g.input_options == {"a": {"itsoffset": -1.5}}


def test_stream_loop_accepts_a_negative_int() -> None:
    g = _lower("SELECT a.video[1] FROM input('x.mp4', stream_loop => -1) a")
    assert g.input_options == {"a": {"stream_loop": -1}}


def test_two_input_aliases_get_independent_option_dicts() -> None:
    g = _lower(
        "SELECT a.video[1], b.video[1] FROM input('x.png', loop => true) a, "
        "input('y.mp4', hwaccel => 'cuda') b"
    )
    assert g.input_options == {"a": {"loop": True}, "b": {"hwaccel": "cuda"}}


def test_input_options_survive_the_split_pass_and_serialize() -> None:
    g = compile_sql(
        "SELECT p.video[1], p.video[1] FROM input('logo.png', loop => true) p"
    )
    # the projection is used twice, so the split pass rebuilt the graph
    assert "split" in _filters(g)
    assert g.to_dict()["input_options"] == {"p": {"loop": True}}
    assert Graph.from_dict(g.to_dict()).to_dict() == g.to_dict()


def test_unknown_input_option_is_anchored_on_its_value() -> None:
    err = _reject("SELECT a.video[1] FROM input('x.mp4', bogus_option => 1) a")
    assert err.code is ErrorCode.UNKNOWN_INPUT_OPTION
    assert "unknown input option 'bogus_option'" in err.message


def test_unknown_input_option_suggests_the_near_miss() -> None:
    err = _reject("SELECT a.video[1] FROM input('x.png', loob => true) a")
    assert err.code is ErrorCode.UNKNOWN_INPUT_OPTION
    assert err.hint == "did you mean 'loop'?"


@pytest.mark.parametrize(
    "option, message",
    [
        ("loop => 1", "expects a bool, got 1"),
        ("loop => 'yes'", "expects a bool, got 'yes'"),
        ("stream_loop => 1.5", "expects an int, got 1.5"),
        ("stream_loop => true", "expects an int, got True"),
        ("framerate => 'fast'", "expects a number, got 'fast'"),
        ("framerate => true", "expects a number, got True"),
        ("hwaccel => 42", "expects a str, got 42"),
        ("hwaccel => true", "expects a str, got True"),
    ],
)
def test_bad_input_option_value_is_a_type_error(option: str, message: str) -> None:
    err = _reject(f"SELECT a.video[1] FROM input('x.mp4', {option}) a")
    assert err.code is ErrorCode.INPUT_OPTION_TYPE
    assert message in err.message
    assert err.hint is not None


def test_input_option_name_is_case_sensitive() -> None:
    """Unlike a sink option (folded), an input option is Kwarg-verbatim."""
    err = _reject("SELECT a.video[1] FROM input('x.mp4', Loop => true) a")
    assert err.code is ErrorCode.UNKNOWN_INPUT_OPTION


def test_itsoffset_compiles_to_a_negative_argv_flag() -> None:
    """Compile-level, not hand-built IR: a negative itsoffset survives the
    whole pipeline (parser -> lower -> emit -> build_ffmpeg_args)."""
    graph = compile_sql("SELECT a.video[1] FROM input('x.mp4', itsoffset => -1.5) a")
    assert graph.input_options == {"a": {"itsoffset": -1.5}}
    emitted = emit(graph)
    args = build_ffmpeg_args(emitted, "out.mp4")
    assert args[:4] == ["ffmpeg", "-itsoffset", "-1.5", "-i"]


def test_new_input_option_batch_lowers_in_written_order() -> None:
    g = _lower(
        "SELECT a.video[1] FROM input("
        "'x.mp4', seek_end => 60, format => 'v4l2', realtime => true, "
        "sub_charenc => 'CP1250', start_number => 3, subtitle_decoder => 'webvtt'"
        ") a"
    )
    assert g.input_options == {
        "a": {
            "seek_end": 60,
            "format": "v4l2",
            "realtime": True,
            "sub_charenc": "CP1250",
            "start_number": 3,
            "subtitle_decoder": "webvtt",
        }
    }


def test_seek_end_compiles_to_a_negated_sseof_flag() -> None:
    graph = compile_sql("SELECT a.video[1] FROM input('x.mp4', seek_end => 60) a")
    args = build_ffmpeg_args(emit(graph), "out.mp4")
    assert args[:4] == ["ffmpeg", "-sseof", "-60", "-i"]


def test_realtime_compiles_to_a_bare_re_flag() -> None:
    graph = compile_sql("SELECT a.video[1] FROM input('x.mp4', realtime => true) a")
    args = build_ffmpeg_args(emit(graph), "out.mp4")
    assert args[:3] == ["ffmpeg", "-re", "-i"]


def test_realtime_false_emits_no_flag_at_all() -> None:
    graph = compile_sql("SELECT a.video[1] FROM input('x.mp4', realtime => false) a")
    args = build_ffmpeg_args(emit(graph), "out.mp4")
    assert "-re" not in args


def test_realtime_is_absent_by_default() -> None:
    graph = compile_sql("SELECT a.video[1] FROM input('x.mp4') a")
    assert graph.input_options == {}
    args = build_ffmpeg_args(emit(graph), "out.mp4")
    assert "-re" not in args


def test_realtime_emits_re_before_the_right_input_on_a_two_input_query() -> None:
    """`-re` sits on its OWN alias's `-i`, not the other input's."""
    graph = _lower(
        "SELECT a.video[1], b.video[1] "
        "FROM input('x.mp4') a, input('y.mp4', realtime => true) b"
    )
    args = build_ffmpeg_args(emit(graph), "out.mp4")
    b_index = args.index("y.mp4")
    assert args[b_index - 2 : b_index] == ["-re", "-i"]
    a_index = args.index("x.mp4")
    assert args[a_index - 1] == "-i"
    assert args[a_index - 2] != "-re"


def test_realtime_refuses_a_socket_input() -> None:
    """A socket already delivers at its own pace; `-re` on it is refused rather
    than silently pacing it a second time."""
    err = _reject_lower(
        "SELECT a.video[1] FROM input('srt://host/stream', realtime => true) a", {}
    )
    assert err.code is ErrorCode.INPUT_OPTION_TYPE
    assert "already live" in err.message
    assert err.hint is not None and "drop realtime" in err.hint


def test_realtime_is_not_refused_on_a_format_forced_synthetic_source() -> None:
    """`format => 'lavfi'` makes an input single-open too, but it names a
    generator, not a device -- telling the two apart needs a name list this
    table does not carry, so only a socket (`is_url`) is refused. This is the
    documented residual, and the exact combination recipe 101 in
    docs/examples.md relies on."""
    graph = compile_sql(
        "SELECT a.video[1] FROM input('testsrc2=size=8x8', "
        "format => 'lavfi', realtime => true) a"
    )
    args = build_ffmpeg_args(emit(graph), "out.mp4")
    assert args[:5] == ["ffmpeg", "-f", "lavfi", "-re", "-i"]


def test_seek_end_together_with_a_where_window_is_rejected() -> None:
    err = _reject(
        "SELECT a.video[1] FROM input('x.mp4', seek_end => 60) a WHERE a.t >= 1"
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "seek_end" in err.message
    assert "'a'" in err.message


def test_seek_end_alone_with_no_where_is_fine() -> None:
    g = _lower("SELECT a.video[1] FROM input('x.mp4', seek_end => 60) a")
    assert g.input_options == {"a": {"seek_end": 60}}
    assert g.input_trims == {}


def test_where_alone_with_no_seek_end_is_still_fine() -> None:
    g = _lower("SELECT a.video[1] FROM input('x.mp4') a WHERE a.t >= 1")
    assert g.input_trims == {"a": (1, None)}


# ---------------------------------------------------------------------------
# dynamic filters + named arguments, against an OFFLINE registry
# ---------------------------------------------------------------------------
#
# These build a real `Registry` over faked subprocess output, exactly as
# tests/test_registry.py does, so the whole tier-2 surface is exercised with
# no ffmpeg on the machine and no machine-dependent expectations. The
# `exec`-marked tests further down run the same shapes against the REAL
# installed ffmpeg, where the option tables are whatever that binary says.

_FILTERS_FIXTURE = """\
Filters:
  T.. = Timeline support
  .S. = Slice threading
  ..C = Command support
  A = Audio input/output
  V = Video input/output
  N = Dynamic number and/or type of input/output
  | = Source or sink filter
 .S. acrossover        A->N       Split audio into per-bands streams.
 ... acrossfade        AA->A      Cross fade two input audio streams.
 ... aecho             A->A       Add echoing to the audio.
 ... amerge            N->A       Merge two or more audio streams into a single multi-channel stream.
 ... join              N->A       Join multiple audio streams into multi-channel output.
 ..C amix              N->A       Audio mixing.
 .. ladspa             N->A       Apply LADSPA effect.
 .S. hstack            N->V       Stack video inputs horizontally.
 .S. vstack            N->V       Stack video inputs vertically.
 ... interleave        N->V       Temporally interleave video inputs.
 ... ainterleave       N->A       Temporally interleave audio inputs.
 ... channelsplit      A->N       Split audio into per-channel streams.
 ..C crop              V->V       Crop the input video.
 T.C feedback          VV->VV     Apply feedback video filter.
 TSC deband            V->V       Debands video.
 ... extractplanes     V->N       Extract planes as grayscale frames.
 TSC gblur             V->V       Apply Gaussian Blur filter.
 ..C scale             V->V       Scale the input video size and/or convert the image format.
 ... split             V->N       Pass on the input to N video outputs.
 ... testsrc           |->V       Generate test pattern.
 ... trim              V->V       Pick one continuous section from the input, drop the rest.
 ..C volume            A->A       Change input volume.
 TS. unsharp           V->V       Sharpen or blur the input video.
 .S. xfade             VV->V      Cross fade one video with another video.
 ... anullsrc          |->A       Null audio source, return empty audio frames.
 ... sine              |->A       Generate sine wave audio signal.
 ..C avsynctest        |->AV      Generate an Audio Video Sync Test.
 ..C movie             |->N       Read from a movie source.
 ... anullsink         A->|       Do absolutely nothing with the input audio.
"""

# Each entry is the AVOptions block of a real `ffmpeg -hide_banner -help
# filter=X` capture (ffmpeg 7.1), trimmed to the block the parser reads --
# and, for xfade, to the first 16 of its 59 transition constants, which is
# still more than a message lists before it starts counting.
_HELP_FIXTURES: dict[str, str] = {
    "volume": """volume AVOptions:
   volume            <string>     ..F.A....T. set volume adjustment expression (default "1.0")
   precision         <int>        ..F.A...... select mathematical precision (from 0 to 2) (default float)
     fixed           0            ..F.A...... select 8-bit fixed-point
     float           1            ..F.A...... select 32-bit floating-point
     double          2            ..F.A...... select 64-bit floating-point
   eval              <int>        ..F.A...... specify when to evaluate expressions (from 0 to 1) (default once)
     once            0            ..F.A...... eval volume expression once
     frame           1            ..F.A...... eval volume expression per-frame

""",
    # ffmpeg 7.1's acrossfade: a plain two-input `AA->A` filter, no `inputs`
    # option -- that arrives only with the `N->A` shape ffmpeg 9 gave it.
    "acrossfade": """acrossfade AVOptions:
   nb_samples        <int>        ..F.A...... set number of samples for cross fade duration (from 1 to 2147483647) (default 44100)
   duration          <duration>   ..F.A...... set cross fade duration (default 0)
   overlap           <boolean>    ..F.A...... overlap 1st stream end with 2nd stream start (default true)

""",
    "amix": """amix AVOptions:
   inputs            <int>        ..F.A...... Number of inputs. (from 1 to 32767) (default 2)
   duration          <int>        ..F.A...... How to determine the end-of-stream. (from 0 to 2) (default longest)
     longest         0            ..F.A...... Duration of longest input.
     shortest        1            ..F.A...... Duration of shortest input.
     first           2            ..F.A...... Duration of first input.
   dropout_transition <float>      ..F.A...... Transition time, in seconds, for volume renormalization when an input stream ends. (from 0 to INT_MAX) (default 2)
   weights           <string>     ..F.A....T. Set weight for each input. (default "1 1")
   normalize         <boolean>    ..F.A....T. Scale inputs (default true)

""",
    "hstack": """(h|v)stack AVOptions:
   inputs            <int>        ..FV....... set number of inputs (from 2 to INT_MAX) (default 2)
   shortest          <boolean>    ..FV....... force termination when the shortest input terminates (default false)

""",
    "vstack": """(h|v)stack AVOptions:
   inputs            <int>        ..FV....... set number of inputs (from 2 to INT_MAX) (default 2)
   shortest          <boolean>    ..FV....... force termination when the shortest input terminates (default false)

""",
    "amerge": """amerge AVOptions:
   inputs            <int>        ..F.A...... specify the number of inputs (from 1 to 64) (default 2)
   layout_mode       <int>        ..F.A...... method used to determine the output channel layout (from 0 to 2) (default legacy)
     legacy          0            ..F.A......
     reset           1            ..F.A......
     normal          2            ..F.A......

""",
    "join": """join AVOptions:
   inputs            <int>        ..F.A...... Number of input streams. (from 1 to INT_MAX) (default 2)
   channel_layout    <channel_layout> ..F.A...... Channel layout of the output stream. (default "stereo")
   map               <string>     ..F.A...... A comma-separated list of channels maps.

""",
    "interleave": """interleave AVOptions:
   nb_inputs         <int>        ..FV....... set number of inputs (from 1 to INT_MAX) (default 2)
   n                 <int>        ..FV....... set number of inputs (from 1 to INT_MAX) (default 2)
   duration          <int>        ..FV....... how to determine the end-of-stream (from 0 to 2) (default longest)
     longest         0            ..FV....... Duration of longest input
     shortest        1            ..FV....... Duration of shortest input
     first           2            ..FV....... Duration of first input

""",
    "ainterleave": """ainterleave AVOptions:
   nb_inputs         <int>        ..F.A...... set number of inputs (from 1 to INT_MAX) (default 2)
   n                 <int>        ..F.A...... set number of inputs (from 1 to INT_MAX) (default 2)
   duration          <int>        ..F.A...... how to determine the end-of-stream (from 0 to 2) (default longest)
     longest         0            ..F.A...... Duration of longest input
     shortest        1            ..F.A...... Duration of shortest input
     first           2            ..F.A...... Duration of first input

""",
    "gblur": """\
gblur AVOptions:
   sigma             <float>      ..FV.....T. set sigma (from 0 to 1024) (default 0.5)
   steps             <int>        ..FV.....T. set number of steps (from 1 to 6) (default 1)
   planes            <int>        ..FV.....T. set planes to filter (from 0 to 15) (default 15)
   sigmaV            <float>      ..FV.....T. set vertical sigma (from -1 to 1024) (default -1)

""",
    "unsharp": """\
unsharp AVOptions:
   luma_msize_x      <int>        ..FV....... set luma matrix horizontal size (from 3 to 23) (default 5)
   lx                <int>        ..FV....... set luma matrix horizontal size (from 3 to 23) (default 5)
   luma_amount       <float>      ..FV....... set luma effect strength (from -2 to 5) (default 1)
   la                <float>      ..FV....... set luma effect strength (from -2 to 5) (default 1)

""",
    "deband": """\
deband AVOptions:
   range             <int>        ..FV.....T. set range (from INT_MIN to INT_MAX) (default 16)
   r                 <int>        ..FV.....T. set range (from INT_MIN to INT_MAX) (default 16)
   blur              <boolean>    ..FV.....T. set blur (default true)
   b                 <boolean>    ..FV.....T. set blur (default true)
   coupling          <boolean>    ..FV.....T. set plane coupling (default false)
   c                 <boolean>    ..FV.....T. set plane coupling (default false)

""",
    "aecho": """\
aecho AVOptions:
   in_gain           <float>      ..F.A...... set signal input gain (from 0 to 1) (default 0.6)
   out_gain          <float>      ..F.A...... set signal output gain (from 0 to 1) (default 0.3)
   delays            <string>     ..F.A...... set list of signal delays (default "1000")
   decays            <string>     ..F.A...... set list of signal decays (default "0.5")

""",
    "crop": """\
crop AVOptions:
   out_w             <string>     ..FV.....T. set the width crop area expression (default "iw")
   w                 <string>     ..FV.....T. set the width crop area expression (default "iw")
   out_h             <string>     ..FV.....T. set the height crop area expression (default "ih")
   h                 <string>     ..FV.....T. set the height crop area expression (default "ih")
   x                 <string>     ..FV.....T. set the x crop area expression (default "(in_w-out_w)/2")
   y                 <string>     ..FV.....T. set the y crop area expression (default "(in_h-out_h)/2")
   keep_aspect       <boolean>    ..FV....... keep aspect ratio (default false)
   exact             <boolean>    ..FV....... do exact cropping (default false)

""",
    "scale": """\
scale AVOptions:
   w                 <string>     ..FV.....T. Output video width
   width             <string>     ..FV.....T. Output video width
   h                 <string>     ..FV.....T. Output video height
   height            <string>     ..FV.....T. Output video height
   flags             <string>     ..FV....... Flags to pass to libswscale (default "")
   interl            <boolean>    ..FV....... set interlacing (default false)

""",
    "xfade": """\
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
   duration          <duration>   ..FV....... set cross fade duration (default 1)
   offset            <duration>   ..FV....... set cross fade start relative to first input stream (default 0)
   expr              <string>     ..FV....... set expression for custom transition

""",
    # -- array-returning filters. All three are `->N` and
    # so are EXCLUDED from the registry's tables; their option blocks are
    # still reachable through `Registry.excluded_options`, which is what makes
    # them callable through the namespace. Real ffmpeg 7.1 captures.
    "channelsplit": """\
channelsplit AVOptions:
   channel_layout    <channel_layout> ..F.A...... Input channel layout. (default "stereo")
   channels          <string>     ..F.A...... Channels to extract. (default "all")

""",
    "acrossover": """\
acrossover AVOptions:
   split             <string>     ..F.A...... set split frequencies (default "500")
   order             <int>        ..F.A...... set filter order (from 0 to 9) (default 4th)
     2nd             0            ..F.A...... 2nd order (12 dB/8ve)
     4th             1            ..F.A...... 4th order (24 dB/8ve)
     6th             2            ..F.A...... 6th order (36 dB/8ve)
     8th             3            ..F.A...... 8th order (48 dB/8ve)
   level             <float>      ..F.A...... set input gain (from 0 to 1) (default 1)
   gain              <string>     ..F.A...... set output bands gain (default "1.f")

""",
    "extractplanes": """\
extractplanes AVOptions:
   planes            <flags>      ..FV....... set planes (default r)
     y                            ..FV....... set luma plane
     u                            ..FV....... set u plane
     v                            ..FV....... set v plane
     r                            ..FV....... set red plane
     g                            ..FV....... set green plane
     b                            ..FV....... set blue plane
     a                            ..FV....... set alpha plane

""",
    # -- generated sources. Same lazy `-help` path a
    # regular filter's options take; the short/long alias pairs (size/s,
    # rate/r, duration/d, frequency/f, ...) are real ffmpeg 7.1 captures.
    "testsrc": """\
testsrc AVOptions:
   size              <image_size> ..FV....... set video size (default "320x240")
   s                 <image_size> ..FV....... set video size (default "320x240")
   rate              <video_rate> ..FV....... set video rate (default "25")
   r                 <video_rate> ..FV....... set video rate (default "25")
   duration          <duration>   ..FV....... set video duration (default -0.000001)
   d                 <duration>   ..FV....... set video duration (default -0.000001)
   decimals          <int>        ..FV....... set number of decimals to show (from 0 to 17) (default 0)
   n                 <int>        ..FV....... set number of decimals to show (from 0 to 17) (default 0)

""",
    "anullsrc": """\
anullsrc AVOptions:
   channel_layout    <channel_layout> ..F.A...... set channel_layout (default "stereo")
   cl                <channel_layout> ..F.A...... set channel_layout (default "stereo")
   sample_rate       <int>        ..F.A...... set sample rate (from 1 to INT_MAX) (default 44100)
   r                 <int>        ..F.A...... set sample rate (from 1 to INT_MAX) (default 44100)
   duration          <duration>   ..F.A...... set the audio duration (default -0.000001)
   d                 <duration>   ..F.A...... set the audio duration (default -0.000001)

""",
    "sine": """\
sine AVOptions:
   frequency         <double>     ..F.A...... set the sine frequency (from 0 to DBL_MAX) (default 440)
   f                 <double>     ..F.A...... set the sine frequency (from 0 to DBL_MAX) (default 440)
   sample_rate       <int>        ..F.A...... set the sample rate (from 1 to INT_MAX) (default 44100)
   r                 <int>        ..F.A...... set the sample rate (from 1 to INT_MAX) (default 44100)
   duration          <duration>   ..F.A...... set the audio duration (default 0)
   d                 <duration>   ..F.A...... set the audio duration (default 0)

""",
    # ladspa is `N->A` with no count option at all -- its pad count comes
    # from the loaded plugin's own ports. Real ffmpeg 9.0.1 capture
    # (--enable-ladspa), not 7.1 like the rest of this fixture set: 7.1's
    # reference snapshot predates ladspa joining the N_INPUT table.
    "ladspa": """\
ladspa AVOptions:
   file              <string>     ..F.A...... set library name or full path
   f                 <string>     ..F.A...... set library name or full path
   plugin            <string>     ..F.A...... set plugin name
   p                 <string>     ..F.A...... set plugin name
   controls          <string>     ..F.A...... set plugin options
   c                 <string>     ..F.A...... set plugin options
   sample_rate       <int>        ..F.A...... set sample rate (from 1 to INT_MAX) (default 44100)
   s                 <int>        ..F.A...... set sample rate (from 1 to INT_MAX) (default 44100)
   nb_samples        <int>        ..F.A...... set the number of samples per requested frame (from 1 to INT_MAX) (default 1024)
   n                 <int>        ..F.A...... set the number of samples per requested frame (from 1 to INT_MAX) (default 1024)
   duration          <duration>   ..F.A...... set audio duration (default -0.000001)
   d                 <duration>   ..F.A...... set audio duration (default -0.000001)
   latency           <boolean>    ..F.A...... enable latency compensation (default false)
   l                 <boolean>    ..F.A...... enable latency compensation (default false)

""",
}

_FIXTURE_VERSION_LINE = (
    "ffmpeg version 7.1-full_build-www.gyan.dev Copyright (c) 2000-2024 the FFmpeg developers"
)


@pytest.fixture
def _registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Registry:
    """A real Registry over the captured fixtures above -- no ffmpeg needed.

    A fresh instance rather than ``registry.load()``: the singleton is
    process-wide, and the disk cache is redirected into tmp_path so a test
    never reads (or writes) the developer's own ~/.cache/ffrwd.
    """
    monkeypatch.setattr(registry_module, "_cache_dir", lambda: tmp_path / "cache")
    monkeypatch.setattr(registry_module.binaries, "ffmpeg_path", lambda: "/usr/bin/ffmpeg")

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if "-version" in argv:
            return subprocess.CompletedProcess(argv, 0, stdout=_FIXTURE_VERSION_LINE, stderr="")
        if "-filters" in argv:
            return subprocess.CompletedProcess(argv, 0, stdout=_FILTERS_FIXTURE, stderr="")
        for arg in argv:
            if arg.startswith("filter="):
                name = arg[len("filter=") :]
                return subprocess.CompletedProcess(
                    argv, 0, stdout=_HELP_FIXTURES.get(name, ""), stderr=""
                )
        return subprocess.CompletedProcess(argv, 1, stdout="", stderr="")

    monkeypatch.setattr(registry_module.subprocess, "run", fake_run)
    return Registry()


def _dyn(
    sql: str,
    registry: Registry | None,
    probes: dict[str, ProbeResult | None] | None = None,
) -> Graph:
    return lower(resolve(parse(sql)), probes or {}, registry=registry)


def _reject_dyn(
    sql: str,
    registry: Registry | None,
    probes: dict[str, ProbeResult | None] | None = None,
) -> FfrwdError:
    with pytest.raises(FfrwdError) as excinfo:
        _dyn(sql, registry, probes)
    return _anchored(excinfo.value)


# -- tier 2: calling a filter the registry reports --------------------------


def test_dynamic_filter_lowers_to_a_plain_node(_registry: Registry) -> None:
    g = _dyn("SELECT gblur(a.video[1], sigma => 5) FROM input('x.mp4') a", _registry)
    node = g.nodes["n1"]
    assert node.filter == "gblur"
    assert node.args == {"sigma": 5}
    assert node.inputs == ["src:a:v:0"]
    assert node.outputs == ["video"]
    assert _outputs(g) == [("n1", "video", None)]


def test_dynamic_filter_without_options_sets_no_args(_registry: Registry) -> None:
    g = _dyn("SELECT gblur(a.video[1]) FROM input('x.mp4') a", _registry)
    assert g.nodes["n1"].args == {}


def test_dynamic_options_keep_their_written_order(_registry: Registry) -> None:
    """emit renders args in insertion order, so written order is the rendered
    order -- both directions are checked here."""
    g = _dyn(
        "SELECT unsharp(a.video[1], luma_amount => 1.5, luma_msize_x => 7) "
        "FROM input('x.mp4') a",
        _registry,
    )
    assert list(g.nodes["n1"].args.items()) == [("luma_amount", 1.5), ("luma_msize_x", 7)]
    assert "unsharp=luma_amount=1.5:luma_msize_x=7" in emit(insert_splits(g)).filter_complex


def test_a_positional_after_the_pads_binds_to_the_first_option(
    _registry: Registry,
) -> None:
    """`gblur(f, 5)` IS `gblur=5`, because `sigma` is the
    option ffmpeg declares first."""
    g = _dyn("SELECT gblur(a.video[1], 5) FROM input('x.mp4') a", _registry)
    assert g.nodes["n1"].args == {"sigma": 5}


def test_positionals_bind_in_the_filters_own_option_order(
    _registry: Registry,
) -> None:
    g = _dyn("SELECT gblur(a.video[1], 5, 2, 1) FROM input('x.mp4') a", _registry)
    assert list(g.nodes["n1"].args.items()) == [("sigma", 5), ("steps", 2), ("planes", 1)]


def test_crops_positionals_are_ffmpegs_order_not_the_old_stdlibs(
    _registry: Registry,
) -> None:
    """The documented ARG-ORDER CHANGE: ffmpeg declares out_w, out_h, x, y."""
    g = _dyn("SELECT crop(a.video[1], 100, 50, 10, 20) FROM input('x.mp4') a", _registry)
    assert g.nodes["n1"].args == {"out_w": 100, "out_h": 50, "x": 10, "y": 20}


def test_a_positional_option_is_validated_as_the_option_it_lands_on(
    _registry: Registry,
) -> None:
    """Slot 2 of gblur is `steps`, an int from 1 to 6 -- so 99 is that option's
    own range rejection, not a generic arity complaint."""
    err = _reject_dyn("SELECT gblur(a.video[1], 5, 99) FROM input('x.mp4') a", _registry)
    assert err.code is ErrorCode.FILTER_OPTION_TYPE
    assert "'steps' of filter 'gblur'" in err.message
    assert "from 1 to 6" in err.message


def test_a_positional_enum_option_takes_its_constant_names(
    _registry: Registry,
) -> None:
    g = _dyn(
        "SELECT xfade(a.video[1], b.video[1], 'wipeleft', 1) "
        "FROM input('x.mp4') a, input('y.mp4') b",
        _registry,
    )
    assert g.nodes["n1"].args == {"transition": "wipeleft", "duration": 1}


def test_a_bad_positional_enum_value_is_the_options_own_rejection(
    _registry: Registry,
) -> None:
    err = _reject_dyn(
        "SELECT xfade(a.video[1], b.video[1], 'nope') FROM input('x.mp4') a, input('y.mp4') b",
        _registry,
    )
    assert err.code is ErrorCode.FILTER_OPTION_TYPE
    assert "'transition' of filter 'xfade'" in err.message


def test_more_positionals_than_the_filter_has_options(_registry: Registry) -> None:
    err = _reject_dyn(
        "SELECT gblur(a.video[1], 1, 1, 1, 1, 1) FROM input('x.mp4') a", _registry
    )
    assert err.code is ErrorCode.UDF_ARG_TYPE
    assert "got 5 positional options" in err.message
    assert "'gblur' filter has 4" in err.message
    assert err.hint is not None and "sigma" in err.hint


def test_positionals_on_a_filter_with_no_options_at_all(_registry: Registry) -> None:
    """`trim` has no -help block in this fixture, so its option table is empty
    -- every positional past its pad is one too many."""
    err = _reject_dyn("SELECT ffmpeg.trim(a.video[1], 1) FROM input('x.mp4') a", _registry)
    assert err.code is ErrorCode.UDF_ARG_TYPE
    assert "has 0" in err.message
    assert err.hint is not None and "no options ffrwd can set" in err.hint


def test_positionals_then_named_is_the_documented_mix(_registry: Registry) -> None:
    g = _dyn("SELECT gblur(a.video[1], 5, planes => 1) FROM input('x.mp4') a", _registry)
    assert g.nodes["n1"].args == {"sigma": 5, "planes": 1}


def test_a_positional_after_a_named_is_unsupported_sql(_registry: Registry) -> None:
    err = _reject_dyn(
        "SELECT gblur(a.video[1], sigma => 5, 2) FROM input('x.mp4') a", _registry
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "positional arguments must come before named arguments" in err.message


def test_a_named_option_already_bound_positionally_conflicts(
    _registry: Registry,
) -> None:
    err = _reject_dyn(
        "SELECT gblur(a.video[1], 5, sigma => 2) FROM input('x.mp4') a", _registry
    )
    assert err.code is ErrorCode.FILTER_OPTION_TYPE
    assert "'sigma' of filter 'gblur' is already set positionally" in err.message
    assert err.hint is not None and "drop one of the two spellings" in err.hint


def test_a_named_option_that_no_positional_claimed_is_fine(
    _registry: Registry,
) -> None:
    g = _dyn("SELECT gblur(a.video[1], 5, 2, sigmaV => 3) FROM input('x.mp4') a", _registry)
    assert g.nodes["n1"].args == {"sigma": 5, "steps": 2, "sigmaV": 3}


def test_enable_can_never_be_reached_positionally(_registry: Registry) -> None:
    """`enable` is framework-level and in no option table, so nothing binds to
    it by position -- gblur's four slots are sigma, steps, planes, sigmaV."""
    g = _dyn("SELECT gblur(a.video[1], 1, 1, 1, 1) FROM input('x.mp4') a", _registry)
    assert "enable" not in g.nodes["n1"].args


def test_dynamic_filter_checks_its_pad_types(_registry: Registry) -> None:
    err = _reject_dyn("SELECT gblur(a.audio[1], sigma => 1) FROM input('x.mp4') a", _registry)
    assert err.code is ErrorCode.UDF_ARG_TYPE
    assert "it takes video as its stream input, got (audio)" in err.message


def test_two_pad_dynamic_filter_needs_both_inputs(_registry: Registry) -> None:
    err = _reject_dyn("SELECT xfade(a.video[1]) FROM input('x.mp4') a", _registry)
    assert err.code is ErrorCode.UDF_ARG_TYPE
    assert "it takes video, video as its stream inputs, got (video)" in err.message


def test_two_pad_dynamic_filter_lowers_both_inputs(_registry: Registry) -> None:
    g = _dyn(
        "SELECT xfade(a.video[1], b.video[1], transition => 'wipeleft', duration => 1) "
        "FROM input('x.mp4') a, input('y.mp4') b",
        _registry,
    )
    assert g.nodes["n1"].inputs == ["src:a:v:0", "src:b:v:0"]
    assert g.nodes["n1"].args == {"transition": "wipeleft", "duration": 1}


def test_excluded_filters_are_not_callable(_registry: Registry) -> None:
    """The v1 scope check lives in the registry: dynamic pads (acrossover),
    multiple outputs (feedback) and sources (testsrc) are all in the fixture's
    -filters output but excluded, so lowering never sees them at all.

    `acrossover` is array-RETURNING and callable as
    `ffmpeg.acrossover(...)`, but that table is namespace-only: the BARE name
    resolves exactly as it always did, which is not at all."""
    for sql in (
        "SELECT acrossover(a.audio[1]) FROM input('x.mp4') a",
        "SELECT feedback(a.video[1], a.video[1]) FROM input('x.mp4') a",
        "SELECT testsrc(a.video[1]) FROM input('x.mp4') a",
    ):
        assert _reject_dyn(sql, _registry).code is ErrorCode.UNKNOWN_FUNCTION


def test_there_is_no_name_collision_left_to_win(_registry: Registry) -> None:
    """`scale` and `crop` used to be curated functions shadowing real filters.
    They are the filters now -- ffmpeg's option NAMES and ffmpeg's option
    ORDER, with no remapping layer in between."""
    g = _dyn("SELECT scale(a.video[1], 'iw/2', -2) FROM input('x.mp4') a", _registry)
    assert g.nodes["n1"].args == {"width": "iw/2", "height": -2}
    g = _dyn("SELECT crop(a.video[1], 3, 4, 1, 2) FROM input('x.mp4') a", _registry)
    assert g.nodes["n1"].args == {"out_w": 3, "out_h": 4, "x": 1, "y": 2}


def test_a_builtin_that_is_also_a_filter_still_resolves_to_the_filter(
    _registry: Registry,
) -> None:
    """sqlglot parses `trim(...)` with its own TRIM grammar, which parks the
    argument under `this` rather than in the argument list -- so the call
    resolves to ffmpeg's trim filter but arrives with NO positional args. The
    rejection is typed (and names the pad signature), not a panic."""
    err = _reject_dyn("SELECT trim(a.video[1]) FROM input('x.mp4') a", _registry)
    assert err.code is ErrorCode.UDF_ARG_TYPE
    assert "trim() is an ffmpeg filter" in err.message
    assert "got (nothing)" in err.message


def test_a_dynamic_call_nests_inside_a_stdlib_call(_registry: Registry) -> None:
    g = _dyn("SELECT scale(gblur(a.video[1], sigma => 2), 0.5) FROM input('x.mp4') a", _registry)
    assert _filters(g) == ["gblur", "scale"]
    assert g.nodes["n2"].inputs == ["n1"]


def test_a_stdlib_call_nests_inside_a_dynamic_call(_registry: Registry) -> None:
    g = _dyn("SELECT gblur(scale(a.video[1], 0.5), sigma => 2) FROM input('x.mp4') a", _registry)
    assert _filters(g) == ["scale", "gblur"]
    assert g.nodes["n2"].inputs == ["n1"]


# -- a bare name dispatches to its audio twin over audio input --------------
#
# `fade`/`afade` is a clean pair in the real registry: `fade` takes video
# only, `afade` takes audio only. `mix`/`amix` and `interleave`/`ainterleave`
# only share a stem -- both sides are N-input, with no fixed pad type -- so
# they stay out of dispatch entirely.


def test_bare_name_stays_the_video_filter_over_video_input() -> None:
    g = _lower("SELECT fade(a.video[1], type => 'in', duration => 1) FROM input('x.mp4') a")
    assert _filters(g) == ["fade"]


def test_bare_name_dispatches_to_its_audio_twin_over_audio_input() -> None:
    g = _lower("SELECT fade(a.audio[1], type => 'in', duration => 1) FROM input('x.mp4') a")
    assert _filters(g) == ["afade"]


def test_dispatched_call_compiles_byte_identical_to_writing_the_audio_filter() -> None:
    bare = compile_sql(
        "SELECT fade(a.audio[1], type => 'in', duration => 1) FROM input('x.mp4') a"
    )
    explicit = compile_sql(
        "SELECT afade(a.audio[1], type => 'in', duration => 1) FROM input('x.mp4') a"
    )
    assert build_ffmpeg_args(emit(bare), "out.mp4") == build_ffmpeg_args(
        emit(explicit), "out.mp4"
    )


def test_the_namespaced_spelling_never_dispatches() -> None:
    """`ffmpeg.fade(...)` is `fade` and nothing else -- an audio input is
    exactly the type error it was before dispatch existed."""
    err = _reject("SELECT ffmpeg.fade(a.audio[1]) FROM input('x.mp4') a")
    assert err.code is ErrorCode.UDF_ARG_TYPE
    assert "it takes video as its stream input, got (audio)" in err.message


def test_broadcast_over_an_audio_array_dispatches_once_for_the_whole_array() -> None:
    g = _lower(
        "SELECT fade(a.audio, type => 'in', duration => 1) FROM input('x.mp4') a",
        {"a": _probe_result(audios=2)},
    )
    assert _filters(g) == ["afade", "afade"]


def test_a_video_only_name_with_no_audio_twin_still_rejects_audio() -> None:
    """`crop` has no `acrop` -- the rejection is exactly what it was before."""
    err = _reject("SELECT crop(a.audio[1], 100, 100, 0, 0) FROM input('x.mp4') a")
    assert err.code is ErrorCode.UDF_ARG_TYPE
    assert "it takes video as its stream input, got (audio)" in err.message


def test_option_validation_follows_the_dispatched_target() -> None:
    """`color` is one of `fade`'s own options; `afade` has none such -- the
    dispatched call's error names the filter it actually resolved to."""
    err = _reject(
        "SELECT fade(a.audio[1], type => 'in', color => 'black') FROM input('x.mp4') a"
    )
    assert err.code is ErrorCode.UNKNOWN_FILTER_OPTION
    assert "filter 'afade' has no option 'color'" in err.message


def test_an_n_input_pair_is_not_touched_by_audio_dispatch() -> None:
    """`mix` is N-input with a fixed VIDEO pad type; a bare call over audio
    streams is rejected by the ordinary N-input rule, never quietly switched
    to `amix`."""
    err = _reject(
        "SELECT mix(a.audio[1], b.audio[1]) FROM input('x.mp4') a, input('y.mp4') b"
    )
    assert err.code is ErrorCode.UDF_ARG_TYPE
    assert "its stream inputs are all video" in err.message


# -- named option validation (both tiers, same two codes) -------------------


def test_unknown_option_suggests_a_real_one(_registry: Registry) -> None:
    err = _reject_dyn("SELECT gblur(a.video[1], sigmma => 5) FROM input('x.mp4') a", _registry)
    assert err.code is ErrorCode.UNKNOWN_FILTER_OPTION
    assert "filter 'gblur' has no option 'sigmma'" in err.message
    assert err.hint is not None and "sigma" in err.hint


def test_unknown_option_without_a_match_lists_the_real_options(_registry: Registry) -> None:
    err = _reject_dyn("SELECT gblur(a.video[1], zzzz => 5) FROM input('x.mp4') a", _registry)
    assert err.code is ErrorCode.UNKNOWN_FILTER_OPTION
    assert err.hint is not None
    assert "sigma" in err.hint and "planes" in err.hint


def test_option_names_are_case_sensitive(_registry: Registry) -> None:
    """ffmpeg AVOption names are case-sensitive, so the name is NOT folded the
    Postgres way: sigmaV is a real option, SIGMA is not."""
    g = _dyn("SELECT gblur(a.video[1], sigmaV => 3) FROM input('x.mp4') a", _registry)
    assert g.nodes["n1"].args == {"sigmaV": 3}
    err = _reject_dyn("SELECT gblur(a.video[1], SIGMA => 5) FROM input('x.mp4') a", _registry)
    assert err.code is ErrorCode.UNKNOWN_FILTER_OPTION


def test_numeric_option_rejects_a_string(_registry: Registry) -> None:
    err = _reject_dyn("SELECT gblur(a.video[1], sigma => '5') FROM input('x.mp4') a", _registry)
    assert err.code is ErrorCode.FILTER_OPTION_TYPE
    assert "expects a number" in err.message


def test_numeric_option_enforces_the_introspected_range(_registry: Registry) -> None:
    err = _reject_dyn("SELECT gblur(a.video[1], sigma => 5000) FROM input('x.mp4') a", _registry)
    assert err.code is ErrorCode.FILTER_OPTION_TYPE
    assert "from 0 to 1024" in err.message
    assert "got 5000" in err.message


def test_numeric_option_range_check_is_two_sided(_registry: Registry) -> None:
    assert (
        _reject_dyn("SELECT gblur(a.video[1], steps => 0) FROM input('x.mp4') a", _registry).code
        is ErrorCode.FILTER_OPTION_TYPE
    )
    assert _dyn(
        "SELECT gblur(a.video[1], steps => 6) FROM input('x.mp4') a", _registry
    ).nodes["n1"].args == {"steps": 6}


def test_an_unbounded_numeric_option_takes_any_number(_registry: Registry) -> None:
    """deband's `range` is `(from INT_MIN to INT_MAX)`, which does not parse as
    a float -- the registry records no bounds and no range is enforced."""
    g = _dyn("SELECT deband(a.video[1], range => -4000) FROM input('x.mp4') a", _registry)
    assert g.nodes["n1"].args == {"range": -4000}


def test_boolean_option_takes_bare_true_and_false(_registry: Registry) -> None:
    g = _dyn("SELECT deband(a.video[1], blur => false) FROM input('x.mp4') a", _registry)
    assert g.nodes["n1"].args == {"blur": False}
    # emit renders an ffmpeg boolean as 1/0
    assert "deband=blur=0" in emit(insert_splits(g)).filter_complex


def test_boolean_option_rejects_a_number(_registry: Registry) -> None:
    err = _reject_dyn("SELECT deband(a.video[1], blur => 1) FROM input('x.mp4') a", _registry)
    assert err.code is ErrorCode.FILTER_OPTION_TYPE
    assert "expects true or false" in err.message


def test_enum_option_accepts_one_of_its_constants(_registry: Registry) -> None:
    g = _dyn(
        "SELECT xfade(a.video[1], b.video[1], transition => 'circlecrop') "
        "FROM input('x.mp4') a, input('y.mp4') b",
        _registry,
    )
    assert g.nodes["n1"].args == {"transition": "circlecrop"}


def test_enum_option_rejects_anything_else(_registry: Registry) -> None:
    err = _reject_dyn(
        "SELECT xfade(a.video[1], b.video[1], transition => 'nope') "
        "FROM input('x.mp4') a, input('y.mp4') b",
        _registry,
    )
    assert err.code is ErrorCode.FILTER_OPTION_TYPE
    assert "named constants" in err.message
    assert "wipeleft" in err.message


def test_enum_option_message_stops_counting_at_a_dozen(_registry: Registry) -> None:
    """xfade's transition has 15 constants in the fixture (59 in a real
    ffmpeg): the message lists the first dozen and counts the rest."""
    err = _reject_dyn(
        "SELECT xfade(a.video[1], b.video[1], transition => 'nope') "
        "FROM input('x.mp4') a, input('y.mp4') b",
        _registry,
    )
    assert "(3 more)" in err.message


def test_enum_option_suggests_a_near_miss_constant(_registry: Registry) -> None:
    err = _reject_dyn(
        "SELECT xfade(a.video[1], b.video[1], transition => 'wipelft') "
        "FROM input('x.mp4') a, input('y.mp4') b",
        _registry,
    )
    assert err.hint is not None and "wipeleft" in err.hint


def test_enum_option_rejects_the_constants_number(_registry: Registry) -> None:
    """The registry records constant NAMES, not their values, so a bare number
    is not something ffrwd can check -- it is rejected, with the names."""
    err = _reject_dyn(
        "SELECT xfade(a.video[1], b.video[1], transition => 1) "
        "FROM input('x.mp4') a, input('y.mp4') b",
        _registry,
    )
    assert err.code is ErrorCode.FILTER_OPTION_TYPE
    assert err.hint is not None and "constant name" in err.hint


def test_a_string_option_also_takes_a_bare_number(_registry: Registry) -> None:
    """xfade's duration/offset are ffmpeg `<duration>` options, i.e. strings --
    but an option value is text on the command line either way, and writing
    `duration => '1'` for a number would be a papercut."""
    g = _dyn(
        "SELECT xfade(a.video[1], b.video[1], duration => 1, offset => 2.5) "
        "FROM input('x.mp4') a, input('y.mp4') b",
        _registry,
    )
    assert g.nodes["n1"].args == {"duration": 1, "offset": 2.5}


def test_a_string_option_rejects_a_boolean(_registry: Registry) -> None:
    err = _reject_dyn(
        "SELECT xfade(a.video[1], b.video[1], expr => true) "
        "FROM input('x.mp4') a, input('y.mp4') b",
        _registry,
    )
    assert err.code is ErrorCode.FILTER_OPTION_TYPE
    assert "expects a string" in err.message


def test_option_rejection_anchors_on_the_value(_registry: Registry) -> None:
    """A Kwarg's Var name carries no token position, so the anchor is the
    value literal -- here on line 2, where the option was written."""
    err = _reject_dyn(
        "SELECT gblur(a.video[1],\n       sigma => 5000)\nFROM input('x.mp4') a", _registry
    )
    assert err.line == 2


# -- named args on dynamic filter calls --------------------------------------


def test_a_dynamic_call_merges_named_args_after_positional(_registry: Registry) -> None:
    """`planes` is validated against gblur's options and merged AFTER the
    positionally-mapped sigma -- no stdlib/macro layer involved anymore."""
    g = _dyn("SELECT gblur(a.video[1], 5, planes => 1) FROM input('x.mp4') a", _registry)
    assert g.nodes["n1"].filter == "gblur"
    assert list(g.nodes["n1"].args.items()) == [("sigma", 5), ("planes", 1)]


def test_a_named_extra_is_validated_like_a_dynamic_one(_registry: Registry) -> None:
    err = _reject_dyn("SELECT gblur(a.video[1], 5, planes => 99) FROM input('x.mp4') a", _registry)
    assert err.code is ErrorCode.FILTER_OPTION_TYPE
    assert "from 0 to 15" in err.message

    err = _reject_dyn("SELECT gblur(a.video[1], 5, planez => 1) FROM input('x.mp4') a", _registry)
    assert err.code is ErrorCode.UNKNOWN_FILTER_OPTION
    assert "filter 'gblur'" in err.message


def test_a_named_extra_cannot_override_the_positional_signature(
    _registry: Registry,
) -> None:
    """`out_w` is what crop's positional signature maps its width onto, so
    this is a conflict, never a silent override. Message/code changed under
    the registry-driven option checker: FILTER_OPTION_TYPE, not UDF_ARG_TYPE."""
    err = _reject_dyn(
        "SELECT crop(a.video[1], 10, 10, 0, 0, out_w => 5) FROM input('x.mp4') a", _registry
    )
    assert err.code is ErrorCode.FILTER_OPTION_TYPE
    assert "already set positionally by crop()" in err.message
    assert err.hint is not None and "drop one of the two spellings" in err.hint


def test_a_named_extra_merges_with_positional_xfade_args(
    _registry: Registry,
) -> None:
    """xfade's positional signature is (transition, duration, offset); a
    positional transition plus named duration/offset merge, in written
    order, after it."""
    both = "FROM input('x.mp4') a, input('y.mp4') b"
    g = _dyn(
        f"SELECT xfade(a.video[1], b.video[1], 'wipeleft', duration => 1, offset => 8) {both}",
        _registry,
    )
    assert g.nodes["n1"].args == {"transition": "wipeleft", "duration": 1, "offset": 8}


def test_a_named_extra_conflicts_with_a_positional_xfade_transition(
    _registry: Registry,
) -> None:
    """When the positional call already sets transition, a named one on top
    of it is a genuine conflict."""
    err = _reject_dyn(
        "SELECT xfade(a.video[1], b.video[1], 'fade', transition => 'wipeleft') "
        "FROM input('x.mp4') a, input('y.mp4') b",
        _registry,
    )
    assert err.code is ErrorCode.FILTER_OPTION_TYPE
    assert "already set positionally by xfade()" in err.message


def test_a_named_extra_that_the_xfade_signature_leaves_free(_registry: Registry) -> None:
    """The 2-argument (transition-only) positional call sets no `expr`, so a
    named one merges in."""
    g = _dyn(
        "SELECT xfade(a.video[1], b.video[1], 'wipeleft', expr => 'A') "
        "FROM input('x.mp4') a, input('y.mp4') b",
        _registry,
    )
    assert g.nodes["n1"].args["expr"] == "A"


# test_a_macro_rejects_named_extras (blur_regions + named extra => UDF_ARG_TYPE)
# DEFERRED to 053b: blur_regions is ffrwd.blur_regions, not yet implemented
# by 052 at the time of this pass. blur_regions() currently raises
# UNKNOWN_FUNCTION instead of UDF_ARG_TYPE. Left out rather than left red,
# since the macro doesn't exist to call yet; re-add once 052/053b lands it.


def test_a_filter_this_ffmpeg_lacks_is_an_unknown_function(_registry: Registry) -> None:
    """`subtitles` is not in this (fixture) ffmpeg's filter set, so the NAME is
    what fails -- there is no curated entry left to resolve it first."""
    err = _reject_dyn(
        "SELECT subtitles(a.video[1], 'subs.srt', force_style => 'x') FROM input('x.mp4') a",
        _registry,
    )
    assert err.code is ErrorCode.UNKNOWN_FUNCTION


# -- broadcasting is tier-agnostic -----------------------------------------


def test_a_dynamic_call_broadcasts_over_an_array(_registry: Registry) -> None:
    g = _dyn(
        "SELECT aecho(a.audio, in_gain => 0.5) FROM input('x.mp4') a",
        _registry,
        {"a": _probe_result(per_audio_tags=[{"language": "eng"}, {"language": "fra"}])},
    )
    assert _filters(g) == ["aecho", "aecho"]
    assert g.nodes["n1"].inputs == ["src:a:a:0"]
    assert g.nodes["n2"].inputs == ["src:a:a:1"]
    assert g.nodes["n1"].args == g.nodes["n2"].args == {"in_gain": 0.5}
    # provenance threads through a single-stream-input dynamic call, exactly
    # as it does through a stdlib one
    assert [o.metadata for o in g.outputs] == [{"language": "eng"}, {"language": "fra"}]


def test_a_two_pad_dynamic_call_zips_two_arrays(_registry: Registry) -> None:
    probes = {
        "a": _probe_result(videos=2, audios=0, video_tags={"language": "eng"}),
        "b": _probe_result(videos=2, audios=0, video_tags={"language": "eng"}),
    }
    g = _dyn(
        "SELECT xfade(a.video, b.video, duration => 1) "
        "FROM input('x.mp4') a, input('y.mp4') b",
        _registry,
        probes,
    )
    assert _filters(g) == ["xfade", "xfade"]
    assert g.nodes["n1"].inputs == ["src:a:v:0", "src:b:v:0"]
    assert g.nodes["n2"].inputs == ["src:a:v:1", "src:b:v:1"]
    # both inputs of each pad agree on the language, so the join keeps it
    assert [o.metadata for o in g.outputs] == [{"language": "eng"}] * 2


def test_a_dynamic_call_reports_a_zip_mismatch(_registry: Registry) -> None:
    err = _reject_dyn(
        "SELECT xfade(a.video, b.video) FROM input('x.mp4') a, input('y.mp4') b",
        _registry,
        {
            "a": _probe_result(videos=2, audios=0),
            "b": _probe_result(videos=1, audios=0),
        },
    )
    assert err.code is ErrorCode.BROADCAST_MISMATCH


def test_a_dynamic_join_drops_a_disagreeing_tag(_registry: Registry) -> None:
    g = _dyn(
        "SELECT xfade(a.video[1], b.video[1]) FROM input('x.mp4') a, input('y.mp4') b",
        _registry,
        {
            "a": _probe_result(video_tags={"language": "eng"}),
            "b": _probe_result(video_tags={"language": "fra"}),
        },
    )
    assert g.outputs[0].metadata == {}


def test_a_dynamic_node_is_split_like_any_other(_registry: Registry) -> None:
    g = insert_splits(
        _dyn(
            "WITH c AS (SELECT gblur(a.video[1], sigma => 2) AS f FROM input('x.mp4') a) "
            "SELECT hstack(c.f, c.f) FROM c",
            _registry,
        )
    )
    assert [node.filter for node in g.nodes.values()] == ["gblur", "split", "hstack"]


# -- no registry at all: no ffmpeg ------------------------------------------
#
# There is no --portable and no tier system: there is no portable subset
# left to compile against, because every function IS a filter of the
# installed ffmpeg. A None/empty registry is now exactly one situation -- no
# ffmpeg -- and every call name in it is UNKNOWN_FUNCTION.


def test_without_a_registry_a_filter_name_is_an_unknown_function() -> None:
    err = _reject_dyn("SELECT deband(a.video[1], range => 8) FROM input('x.mp4') a", None)
    assert err.code is ErrorCode.UNKNOWN_FUNCTION
    assert err.hint is not None and "provisioner" in err.hint


def test_without_a_registry_every_name_is_unknown_including_common_ones() -> None:
    """No name is privileged any more -- there is no curated list to fall back
    on, so `scale` fails exactly the way `deband` does."""
    for name in ("scale", "crop", "gblur", "volume", "amix"):
        err = _reject_dyn(f"SELECT {name}(a.video[1]) FROM input('x.mp4') a", None)
        assert err.code is ErrorCode.UNKNOWN_FUNCTION, name


def test_without_a_registry_the_namespace_is_unknown_too() -> None:
    err = _reject_dyn("SELECT ffmpeg.gblur(a.video[1]) FROM input('x.mp4') a", None)
    assert err.code is ErrorCode.UNKNOWN_FUNCTION
    assert err.hint is not None and "provisioner" in err.hint


def test_did_you_mean_over_the_registry(_registry: Registry) -> None:
    err = _reject_dyn("SELECT gblu(a.video[1]) FROM input('x.mp4') a", _registry)
    assert err.code is ErrorCode.UNKNOWN_FUNCTION
    assert err.hint is not None and "gblur()" in err.hint


def test_did_you_mean_can_suggest_an_n_input_filter(_registry: Registry) -> None:
    """The N_INPUT names are callable, so they are candidates for the hint even
    though the registry's own tables exclude them."""
    err = _reject_dyn("SELECT amixx(a.audio[1], a.audio[2]) FROM input('x.mp4') a", _registry)
    assert err.hint is not None and "amix()" in err.hint


def test_no_match_says_the_surface_is_the_filter_set(_registry: Registry) -> None:
    err = _reject_dyn("SELECT zzzz(a.video[1]) FROM input('x.mp4') a", _registry)
    assert err.hint is not None and "filter of your installed ffmpeg" in err.hint


def test_compile_sql_always_builds_a_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """There is no flag that skips it any more: the registry IS the surface."""
    calls: list[int] = []
    real = compiler.registry_module.load

    def counted() -> Registry:
        calls.append(1)
        return real()

    monkeypatch.setattr(compiler.registry_module, "load", counted)
    compile_sql("SELECT gblur(a.video[1], 5) FROM input('x.mp4') a")
    assert calls, "compile_sql must consult the registry"


# ---------------------------------------------------------------------------
# real probing against a generated fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def _fixtures() -> None:
    subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "gen_fixtures.py")],
        check=True,
    )


@pytest.fixture(scope="module")
def _av_fixture(_fixtures: None) -> str:
    """testsrc2 + one sine track."""
    return (FIXTURES_DIR / "av.mp4").as_posix()


@pytest.fixture(scope="module")
def _av2_fixture(_fixtures: None) -> str:
    """testsrc2 + TWO sine tracks, tagged language=eng and language=fra."""
    return (FIXTURES_DIR / "av2.mp4").as_posix()


@pytest.fixture(scope="module")
def _av3_fixture(_fixtures: None) -> str:
    """av2's concat partner: same shape and tags, different sine frequencies."""
    return (FIXTURES_DIR / "av3.mp4").as_posix()


@pytest.mark.exec
def test_probed_fixture_rejects_an_out_of_range_subscript(_av_fixture: str) -> None:
    """av.mp4 has exactly one video and one audio stream."""
    err = _reject(f"SELECT a.video[2] FROM input('{_av_fixture}') a")
    assert err.code is ErrorCode.STREAM_NOT_FOUND
    assert "a.video[2]" in err.message


@pytest.mark.exec
def test_probed_fixture_accepts_its_real_streams(_av_fixture: str) -> None:
    g = compile_sql(f"SELECT a.video[1], a.audio[1] FROM input('{_av_fixture}') a")
    assert _outputs(g) == [
        ("src:a:v:0", "video", None),
        ("src:a:a:0", "audio", None),
    ]


@pytest.mark.exec
def test_an_empty_array_never_reaches_a_command_with_no_maps(_fixtures: None) -> None:
    """testsrc.mp4 is video-only, so its audio array is empty.

    Selecting it contributes nothing and warns. What must not happen is a
    COMMAND with no -map, and two separate rules see to that: a COPY left
    with no streams is a rejection, and a bare SELECT names no destination
    so it never becomes a command at all.
    """
    path = (FIXTURES_DIR / "testsrc.mp4").as_posix()
    said: list[FfrwdWarning] = []
    graph = compile_sql(
        f"COPY (SELECT a.video, a.audio FROM input('{path}') a) TO 'o.mkv'",
        on_warning=said.append,
    )
    assert [o.type for o in graph.sinks[0].outputs] == ["video"]
    assert [w.code for w in said] == [WarningCode.EMPTY_STREAM_ARRAY]

    err = _reject(f"COPY (SELECT a.audio FROM input('{path}') a) TO 'o.mkv'")
    assert err.code is ErrorCode.STREAM_NOT_FOUND
    assert "would have no streams" in err.message


# ---------------------------------------------------------------------------
# broadcasting against the real 2-audio-track fixture
# ---------------------------------------------------------------------------


@pytest.mark.exec
def test_reverb_over_every_language_track(_av2_fixture: str) -> None:
    """The headline case: one aecho per language track, each output still tagged."""
    g = compile_sql(f"SELECT aecho(a.audio, 0.8, 0.9, 60, 0.3) AS dubbed FROM input('{_av2_fixture}') a")
    assert _filters(g) == ["aecho", "aecho"]
    assert g.nodes["n1"].inputs == ["src:a:a:0"]
    assert g.nodes["n2"].inputs == ["src:a:a:1"]
    assert [(o.ref, o.name) for o in g.outputs] == [("n1", "dubbed"), ("n2", "dubbed")]
    assert [o.metadata for o in g.outputs] == [{"language": "eng"}, {"language": "fra"}]


@pytest.mark.exec
def test_probed_scalar_broadcast_makes_one_node_per_real_stream(_av2_fixture: str) -> None:
    g = compile_sql(f"SELECT volume(a.audio, 0.5) FROM input('{_av2_fixture}') a")
    assert _filters(g) == ["volume", "volume"]
    assert len(g.outputs) == 2


@pytest.mark.exec
def test_probed_splat_passes_every_track_through(_av2_fixture: str) -> None:
    g = compile_sql(f"SELECT a.video, a.audio FROM input('{_av2_fixture}') a")
    assert g.nodes == {}
    assert _outputs(g) == [
        ("src:a:v:0", "video", None),
        ("src:a:a:0", "audio", None),
        ("src:a:a:1", "audio", None),
    ]
    # the video track's mp4-stamped language=und says nothing and is dropped
    assert [o.metadata for o in g.outputs] == [
        {},
        {"language": "eng"},
        {"language": "fra"},
    ]


@pytest.mark.exec
def test_zip_mismatch_across_two_real_files(_av_fixture: str, _av2_fixture: str) -> None:
    """av2.mp4 has 2 audio tracks, av.mp4 has 1: nothing to zip."""
    err = _reject(
        f"SELECT amix(a.audio, b.audio) "
        f"FROM input('{_av2_fixture}') a, input('{_av_fixture}') b"
    )
    assert err.code is ErrorCode.BROADCAST_MISMATCH
    assert "a.audio has 2 streams" in err.message
    assert "b.audio has 1 stream" in err.message


@pytest.mark.exec
def test_probed_cte_array_subscript_against_a_real_file(_av2_fixture: str) -> None:
    err = _reject(
        f"WITH c AS (SELECT a.audio AS snd FROM input('{_av2_fixture}') a) "
        f"SELECT c.snd[3] FROM c"
    )
    assert err.code is ErrorCode.STREAM_NOT_FOUND
    assert "column 'c.snd' has 2 streams" in err.message


# ---------------------------------------------------------------------------
# UNION ALL over splatted arrays, against the two real 2-track fixtures
# ---------------------------------------------------------------------------


@pytest.mark.exec
def test_union_splat_pairs_every_language_track(_av2_fixture: str, _av3_fixture: str) -> None:
    """The flagship case: two dual-language files, `<alias>.audio` splatted in
    both branches, and the tracks pair up by position -- eng with eng, fra with
    fra -- into one concat with 1 video and 2 audio pads."""
    g = compile_sql(
        f"SELECT a.video[1], a.audio FROM input('{_av2_fixture}') a "
        f"UNION ALL SELECT b.video[1], b.audio FROM input('{_av3_fixture}') b"
    )
    assert _filters(g) == ["concat"]
    assert g.nodes["n1"].args == {"n": 2, "v": 1, "a": 2}
    assert g.nodes["n1"].inputs == [
        "src:a:v:0", "src:a:a:0", "src:a:a:1",
        "src:b:v:0", "src:b:a:0", "src:b:a:1",
    ]
    assert len(g.outputs) == 3
    assert [o.type for o in g.outputs] == ["video", "audio", "audio"]
    assert [o.metadata for o in g.outputs] == [
        {},
        {"language": "eng"},
        {"language": "fra"},
    ]


@pytest.mark.exec
def test_union_of_disagreeing_tracks_drops_the_language(
    _av2_fixture: str, _av3_fixture: str
) -> None:
    """av2's first track is eng and av3's second is fra: the concat output is
    neither, so it carries no language at all."""
    g = compile_sql(
        f"SELECT a.audio[1] FROM input('{_av2_fixture}') a "
        f"UNION ALL SELECT b.audio[2] FROM input('{_av3_fixture}') b"
    )
    assert [o.metadata for o in g.outputs] == [{}]


# ---------------------------------------------------------------------------
# crossfade of two WHERE-trimmed segments (compile-only)
# ---------------------------------------------------------------------------


@pytest.mark.exec
def test_crossfade_of_two_trimmed_segments_compiles(
    _av2_fixture: str, _av3_fixture: str
) -> None:
    """Each side is trimmed to a 2s window via WHERE, then crossfaded over 1s
    starting 1s into the first segment -- exercises xfade as a real multi-input
    call against two independently seeked inputs. The windows are input options
    (one per -i), so xfade consumes the raw refs and the graph is one node."""
    g = compile_sql(
        f"SELECT xfade(a.video[1], b.video[1], duration => 1, offset => 1) "
        f"FROM input('{_av2_fixture}') a, input('{_av3_fixture}') b "
        f"WHERE a.t BETWEEN 0 AND 2 AND b.t BETWEEN 0 AND 2"
    )
    assert _filters(g) == ["xfade"]
    assert g.input_trims == {"a": (0, 2), "b": (0, 2)}
    xfade = g.nodes["n1"]
    assert xfade.filter == "xfade"
    assert xfade.args == {"duration": 1, "offset": 1}
    assert xfade.inputs == ["src:a:v:0", "src:b:v:0"]
    assert xfade.outputs == ["video"]
    assert _outputs(g) == [("n1", "video", None)]


# ---------------------------------------------------------------------------
# ffrwd.delay -- the macro's VIDEO-only expansion
# ---------------------------------------------------------------------------
#
# The single-expansion shape (format+tpad), the arity/type-mismatch errors,
# and the named-argument rejection are all covered once, generically, by
# tests/test_macros.py -- what's left here is behavior distinct enough that
# duplicating the macro wouldn't exercise it: provenance threading, the
# audio-delay-is-now-bare-adelay ad-insert composition, and the
# subtitle-argument rejection's specific message.

_AD_INSERT = (
    "SELECT overlay(f.video[1], ffrwd.delay(scale(a.video[1], 'iw*0.33', 'ih*0.33'), 1), 20, 20), "
    "       amix(f.audio[1], volume(adelay(a.audio[1], 1000), 0.5)) "
    "FROM input('film.mp4') f, input('ad.mp4') a"
)


def test_the_ad_insert_composition_lowers_end_to_end() -> None:
    """The ad-insert driving case: a clip delayed onto a film, video via
    `ffrwd.delay`, audio via bare `adelay` in milliseconds (golden
    096-ad-insert pins the whole IR; this pins the shape)."""
    g = _lower(_AD_INSERT)
    assert _filters(g) == [
        "scale",
        "format",
        "tpad",
        "overlay",
        "adelay",
        "volume",
        "amix",
    ]
    assert g.nodes["n4"].inputs == ["src:f:v:0", "n3"]
    assert g.nodes["n7"].inputs == ["src:f:a:0", "n6"]
    assert [o.type for o in g.outputs] == ["video", "audio"]


def test_video_delay_threads_provenance_like_any_1_to_1_chain() -> None:
    g = _lower(
        "SELECT ffrwd.delay(a.video[1], 1) FROM input('x.mp4') a",
        {"a": _probe_result(video_tags={"language": "eng"})},
    )
    assert [o.metadata for o in g.outputs] == [{"language": "eng"}]


def test_delay_over_a_passthrough_stream_is_still_udf_arg_type() -> None:
    err = _reject_lower("SELECT ffrwd.delay(a.subtitle[1], 1) FROM input('x.mp4') a", {})
    assert err.code is ErrorCode.UDF_ARG_TYPE
    assert "cannot take a subtitle stream" in err.message


# ---------------------------------------------------------------------------
# the `ffmpeg.<filter>(...)` namespace
# ---------------------------------------------------------------------------
#
# One spelling of a filter name that no SQL grammar has an opinion about, and
# that resolves in the registry ALONE. The offline fixture registry above has
# `trim` in it, which is exactly the interesting case: bare `trim(a.video[1])` is
# Postgres's string TRIM and loses the argument.


def test_a_namespaced_call_lowers_to_an_ordinary_node(_registry: Registry) -> None:
    g = _dyn("SELECT ffmpeg.gblur(a.video[1], sigma => 5) FROM input('x.mp4') a", _registry)
    assert g.nodes["n1"].filter == "gblur"  # the NODE knows nothing of the namespace
    assert g.nodes["n1"].args == {"sigma": 5}
    assert g.nodes["n1"].inputs == ["src:a:v:0"]
    assert _outputs(g) == [("n1", "video", None)]


def test_the_namespace_and_the_bare_name_are_the_same_call(_registry: Registry) -> None:
    """There is no tier to resolve past. `scale` and `ffmpeg.scale`
    are one filter, one option set, one argument order -- the namespace only
    changes what Postgres's parser does with the NAME."""
    named = _dyn(
        "SELECT ffmpeg.scale(a.video[1], width => 640) FROM input('x.mp4') a", _registry
    )
    assert named.nodes["n1"].filter == "scale"
    assert named.nodes["n1"].args == {"width": 640}

    bare = _dyn("SELECT scale(a.video[1], 640, 480) FROM input('x.mp4') a", _registry)
    assert bare.nodes["n1"].filter == "scale"
    assert bare.nodes["n1"].args == {"width": 640, "height": 480}

    qualified = _dyn(
        "SELECT ffmpeg.scale(a.video[1], 640, 480) FROM input('x.mp4') a", _registry
    )
    assert qualified.nodes["n1"].args == bare.nodes["n1"].args


def test_the_namespace_reaches_a_name_postgres_claimed(_registry: Registry) -> None:
    """Bare `trim(a.video[1])` parses as Postgres's TRIM and arrives with NO
    positional arguments; the namespaced spelling keeps them."""
    err = _reject_dyn("SELECT trim(a.video[1]) FROM input('x.mp4') a", _registry)
    assert err.code is ErrorCode.UDF_ARG_TYPE
    assert "got (nothing)" in err.message

    g = _dyn("SELECT ffmpeg.trim(a.video[1]) FROM input('x.mp4') a", _registry)
    assert g.nodes["n1"].filter == "trim"
    assert g.nodes["n1"].inputs == ["src:a:v:0"]


def test_the_namespace_qualifier_folds_like_any_identifier(_registry: Registry) -> None:
    g = _dyn("SELECT FFMPEG.GBlur(a.video[1], sigma => 1) FROM input('x.mp4') a", _registry)
    assert g.nodes["n1"].filter == "gblur"


def test_a_namespaced_call_checks_its_pad_signature(_registry: Registry) -> None:
    err = _reject_dyn("SELECT ffmpeg.gblur(a.audio[1]) FROM input('x.mp4') a", _registry)
    assert err.code is ErrorCode.UDF_ARG_TYPE
    assert "it takes video as its stream input, got (audio)" in err.message


def test_a_namespaced_option_is_validated_the_ordinary_way(_registry: Registry) -> None:
    err = _reject_dyn(
        "SELECT ffmpeg.gblur(a.video[1], sigmma => 5) FROM input('x.mp4') a", _registry
    )
    assert err.code is ErrorCode.UNKNOWN_FILTER_OPTION
    # The message names the FILTER, which is what has the options.
    assert "filter 'gblur' has no option 'sigmma'" in err.message
    assert err.hint is not None and "sigma" in err.hint


def test_a_namespaced_unknown_name_suggests_the_namespaced_spelling(
    _registry: Registry,
) -> None:
    err = _reject_dyn("SELECT ffmpeg.gblurr(a.video[1]) FROM input('x.mp4') a", _registry)
    assert err.code is ErrorCode.UNKNOWN_FUNCTION
    assert "unknown function ffmpeg.gblurr()" in err.message
    assert err.hint == "did you mean ffmpeg.gblur()?"


def test_a_namespaced_did_you_mean_stays_in_the_namespace(
    _registry: Registry,
) -> None:
    """Both spellings suggest out of the same set now (there is no other set),
    but the namespaced one keeps the `ffmpeg.` prefix on its suggestion --
    which is the spelling that works whatever Postgres thinks of the name."""
    assert (
        _reject_dyn("SELECT aech(a.audio[1]) FROM input('x.mp4') a", _registry).hint
        == "did you mean aecho()?"
    )
    assert (
        _reject_dyn(
            "SELECT ffmpeg.aech(a.audio[1]) FROM input('x.mp4') a", _registry
        ).hint
        == "did you mean ffmpeg.aecho()?"
    )

    err = _reject_dyn("SELECT ffmpeg.reverb(a.audio[1]) FROM input('x.mp4') a", _registry)
    assert err.code is ErrorCode.UNKNOWN_FUNCTION
    assert err.hint is not None and "not one of them" in err.hint


def test_the_scope_check_applies_to_the_namespace_too(_registry: Registry) -> None:
    """Three `->N` names are re-admitted through this namespace (array-
    RETURNING); every `N->1` name (amix, amerge, ...) is an ordinary
    registry member, marked n-input; multi-output, source and
    `split`-shaped (`N` on the OUTPUT side) names stay excluded."""
    for sql in (
        "SELECT ffmpeg.feedback(a.video[1], a.video[1]) FROM input('x.mp4') a",
        "SELECT ffmpeg.testsrc(a.video[1]) FROM input('x.mp4') a",
        "SELECT ffmpeg.split(a.video[1]) FROM input('x.mp4') a",
    ):
        assert _reject_dyn(sql, _registry).code is ErrorCode.UNKNOWN_FUNCTION


def test_namespaced_calls_nest_in_both_directions(_registry: Registry) -> None:
    g = _dyn(
        "SELECT scale(ffmpeg.gblur(a.video[1], sigma => 2), 0.5) FROM input('x.mp4') a",
        _registry,
    )
    assert _filters(g) == ["gblur", "scale"]
    g = _dyn(
        "SELECT ffmpeg.gblur(scale(a.video[1], 0.5), sigma => 2) FROM input('x.mp4') a",
        _registry,
    )
    assert _filters(g) == ["scale", "gblur"]


def test_a_namespaced_call_broadcasts_like_any_other(_registry: Registry) -> None:
    g = _dyn(
        "SELECT ffmpeg.aecho(a.audio, in_gain => 0.5) FROM input('x.mp4') a",
        _registry,
        {"a": _probe_result(per_audio_tags=[{"language": "eng"}, {"language": "fra"}])},
    )
    assert _filters(g) == ["aecho", "aecho"]
    assert [o.metadata for o in g.outputs] == [{"language": "eng"}, {"language": "fra"}]


def test_without_a_registry_the_namespace_is_an_unknown_function() -> None:
    err = _reject_dyn("SELECT ffmpeg.gblur(a.video[1]) FROM input('x.mp4') a", None)
    assert err.code is ErrorCode.UNKNOWN_FUNCTION
    assert err.hint is not None and "provisioner" in err.hint


def test_no_ffmpeg_turns_the_namespace_off() -> None:
    err = _reject_dyn("SELECT ffmpeg.gblur(a.video[1]) FROM input('x.mp4') a", None)
    assert err.code is ErrorCode.UNKNOWN_FUNCTION
    assert err.hint is not None and "provisioner" in err.hint


def test_ffmpeg_is_reserved_as_an_input_alias() -> None:
    err = _reject_lower("SELECT ffmpeg.video[1] FROM input('x.mp4') ffmpeg", {})
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "reserved for the filter namespace" in err.message


def test_ffmpeg_is_reserved_as_a_cte_name() -> None:
    err = _reject_lower(
        "WITH ffmpeg AS (SELECT a.video[1] FROM input('x.mp4') a) "
        "SELECT ffmpeg.video[1] FROM ffmpeg",
        {},
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "reserved for the filter namespace" in err.message


def test_a_bare_ffmpeg_column_points_at_the_call_form() -> None:
    """`ffmpeg.gblur` with no parentheses is a COLUMN, and the namespace only
    exists in call position -- so the hint names the call form."""
    err = _reject_lower("SELECT ffmpeg.gblur FROM input('x.mp4') a", {})
    assert err.code is ErrorCode.UNKNOWN_ALIAS
    assert err.hint is not None and "ffmpeg.gblur(<stream>" in err.hint


def test_an_unrelated_qualified_call_is_still_rejected() -> None:
    """The namespace is exactly one name; `foo.gblur(...)` is not a call
    ffrwd knows how to read."""
    err = _reject_lower("SELECT foo.gblur(a.video[1]) FROM input('x.mp4') a", {})
    assert err.code is ErrorCode.UNSUPPORTED_SQL


# ---------------------------------------------------------------------------
# array-RETURNING filters
# ---------------------------------------------------------------------------
#
# `channelsplit`, `acrossover` and `extractplanes` are `->N` filters, excluded
# from the registry's tables, re-admitted through the `ffmpeg.` namespace by
# `lower.ARRAY_RETURNING` -- one node with N output pads, returned as an
# N-element ARRAY. The fixture registry has all three `-help` blocks (and
# `amerge`, which has none, so it stays excluded).


def test_channelsplit_defaults_to_the_stereo_layouts_two_pads(_registry: Registry) -> None:
    g = _dyn("SELECT ffmpeg.channelsplit(a.audio[1]) FROM input('x.mp4') a", _registry)
    node = g.nodes["n1"]
    assert node.filter == "channelsplit"
    assert node.args == {}
    assert node.inputs == ["src:a:a:0"]
    assert node.outputs == ["audio", "audio"]
    assert _outputs(g) == [("n1:0", "audio", None), ("n1:1", "audio", None)]


@pytest.mark.parametrize(
    ("layout", "count"),
    [("mono", 1), ("stereo", 2), ("2.1", 3), ("quad", 4), ("5.1", 6), ("7.1", 8)],
)
def test_channelsplit_counts_the_channels_of_its_layout(
    _registry: Registry, layout: str, count: int
) -> None:
    g = _dyn(
        f"SELECT ffmpeg.channelsplit(a.audio[1], channel_layout => '{layout}') "
        "FROM input('x.mp4') a",
        _registry,
    )
    assert g.nodes["n1"].outputs == ["audio"] * count
    assert len(g.outputs) == count


def test_a_one_channel_split_is_still_an_array(_registry: Registry) -> None:
    """`is_array` is not `len(streams) != 1`: mono channelsplit returns an
    array of one, so it still subscripts (and would still splat)."""
    g = _dyn(
        "WITH s AS (SELECT ffmpeg.channelsplit(a.audio[1], channel_layout => 'mono') "
        "AS ch FROM input('x.mp4') a) SELECT s.ch[1] FROM s",
        _registry,
    )
    assert g.nodes["n1"].outputs == ["audio"]
    assert _outputs(g) == [("n1:0", "audio", None)]


def test_a_custom_channel_layout_counts_its_channel_names(_registry: Registry) -> None:
    g = _dyn(
        "SELECT ffmpeg.channelsplit(a.audio[1], channel_layout => 'FL+FR+LFE') "
        "FROM input('x.mp4') a",
        _registry,
    )
    assert g.nodes["n1"].outputs == ["audio"] * 3


def test_channels_narrows_the_split_to_a_subset(_registry: Registry) -> None:
    """ffmpeg's `channels` option is the count that WINS: it names the subset
    to extract, so the pad count follows it, not the (wider) layout."""
    g = _dyn(
        "SELECT ffmpeg.channelsplit(a.audio[1], channel_layout => '5.1', "
        "channels => 'FL+FR') FROM input('x.mp4') a",
        _registry,
    )
    assert g.nodes["n1"].args == {"channel_layout": "5.1", "channels": "FL+FR"}
    assert g.nodes["n1"].outputs == ["audio", "audio"]


def test_channels_all_falls_back_to_the_layout(_registry: Registry) -> None:
    g = _dyn(
        "SELECT ffmpeg.channelsplit(a.audio[1], channel_layout => '5.1', "
        "channels => 'all') FROM input('x.mp4') a",
        _registry,
    )
    assert g.nodes["n1"].outputs == ["audio"] * 6


def test_an_unknown_channel_layout_is_a_typed_rejection(_registry: Registry) -> None:
    err = _reject_dyn(
        "SELECT ffmpeg.channelsplit(a.audio[1], channel_layout => 'nonsense') "
        "FROM input('x.mp4') a",
        _registry,
    )
    assert err.code is ErrorCode.FILTER_OPTION_TYPE
    assert "decides how many streams the call returns" in err.message
    # The layout table is longer than a message lists before it starts
    # counting (`_MAX_LISTED`), which is the same rule an enum option's
    # constants get.
    assert "one of mono, stereo, 2.1" in err.message and "more)" in err.message
    assert err.hint is not None and "ffmpeg -layouts" in err.hint and "5.1" in err.hint


def test_a_bad_layout_anchors_on_the_option_that_caused_it(_registry: Registry) -> None:
    err = _reject_dyn(
        "SELECT ffmpeg.channelsplit(\n  a.audio[1],\n  channel_layout => 'nope'\n) "
        "FROM input('x.mp4') a",
        _registry,
    )
    assert err.line == 3


def test_acrossover_returns_one_more_band_than_it_splits(_registry: Registry) -> None:
    g = _dyn("SELECT ffmpeg.acrossover(a.audio[1]) FROM input('x.mp4') a", _registry)
    assert g.nodes["n1"].outputs == ["audio", "audio"]  # default split '500'


@pytest.mark.parametrize("split", ["500|3000", "500 3000", "500||3000"])
def test_acrossover_takes_either_list_separator(_registry: Registry, split: str) -> None:
    """Spaces and `|` both separate, and a repeated separator is one
    separator -- measured against ffmpeg 7.1, which counts the same bands for
    all three spellings (a TRAILING separator likewise yields no extra band)."""
    g = _dyn(
        f"SELECT ffmpeg.acrossover(a.audio[1], split => '{split}') FROM input('x.mp4') a",
        _registry,
    )
    assert g.nodes["n1"].outputs == ["audio"] * 3


def test_a_single_numeric_split_is_two_bands(_registry: Registry) -> None:
    """A `<string>` option also takes a bare number, so the count
    rule reads the rendered text either way."""
    g = _dyn(
        "SELECT ffmpeg.acrossover(a.audio[1], split => 800) FROM input('x.mp4') a",
        _registry,
    )
    assert g.nodes["n1"].args == {"split": 800}
    assert g.nodes["n1"].outputs == ["audio", "audio"]


@pytest.mark.parametrize("split", ["zzz", "", "-500", "0", "500|zzz"])
def test_a_malformed_split_list_is_a_typed_rejection(
    _registry: Registry, split: str
) -> None:
    err = _reject_dyn(
        f"SELECT ffmpeg.acrossover(a.audio[1], split => '{split}') FROM input('x.mp4') a",
        _registry,
    )
    assert err.code is ErrorCode.FILTER_OPTION_TYPE
    assert "positive frequencies" in err.message


def test_acrossovers_other_options_validate_normally(_registry: Registry) -> None:
    g = _dyn(
        "SELECT ffmpeg.acrossover(a.audio[1], order => '6th', level => 0.5) "
        "FROM input('x.mp4') a",
        _registry,
    )
    assert g.nodes["n1"].args == {"order": "6th", "level": 0.5}
    err = _reject_dyn(
        "SELECT ffmpeg.acrossover(a.audio[1], order => 'ninth') FROM input('x.mp4') a",
        _registry,
    )
    assert err.code is ErrorCode.FILTER_OPTION_TYPE
    assert "named constants" in err.message


def test_extractplanes_returns_one_video_pad_per_plane(_registry: Registry) -> None:
    g = _dyn(
        "SELECT ffmpeg.extractplanes(a.video[1], planes => 'y') FROM input('x.mp4') a",
        _registry,
    )
    assert g.nodes["n1"].filter == "extractplanes"
    assert g.nodes["n1"].outputs == ["video"]
    assert _outputs(g) == [("n1:0", "video", None)]


def test_extractplanes_defaults_to_its_one_plane(_registry: Registry) -> None:
    """ffmpeg 7.1 prints `(default r)` for `planes`, which is one plane."""
    g = _dyn("SELECT ffmpeg.extractplanes(a.video[1]) FROM input('x.mp4') a", _registry)
    assert g.nodes["n1"].args == {}
    assert g.nodes["n1"].outputs == ["video"]


def test_a_multi_plane_value_is_rejected_before_the_count_rule(
    _registry: Registry,
) -> None:
    """The registry types an option that lists constants as an ENUM, so
    ffmpeg's own `y+u` flags spelling is FILTER_OPTION_TYPE from the ordinary
    constant check -- the count rule's `+` arithmetic is unreachable until a
    later plan teaches the registry about flags sets."""
    err = _reject_dyn(
        "SELECT ffmpeg.extractplanes(a.video[1], planes => 'y+u') FROM input('x.mp4') a",
        _registry,
    )
    assert err.code is ErrorCode.FILTER_OPTION_TYPE
    assert "named constants" in err.message


def test_an_array_call_splats_into_the_select_list(_registry: Registry) -> None:
    g = _dyn(
        "SELECT a.video[1], ffmpeg.channelsplit(a.audio[1]) FROM input('x.mp4') a",
        _registry,
    )
    assert _outputs(g) == [
        ("src:a:v:0", "video", None),
        ("n1:0", "audio", None),
        ("n1:1", "audio", None),
    ]


def test_an_array_call_subscripts_through_a_cte(_registry: Registry) -> None:
    g = _dyn(
        "WITH s AS (SELECT ffmpeg.channelsplit(a.audio[1]) AS ch FROM input('x.mp4') a) "
        "SELECT s.ch[2] FROM s",
        _registry,
    )
    assert _outputs(g) == [("n1:1", "audio", None)]


def test_an_array_call_is_bounds_checked_statically(_registry: Registry) -> None:
    err = _reject_dyn(
        "WITH s AS (SELECT ffmpeg.channelsplit(a.audio[1]) AS ch FROM input('x.mp4') a) "
        "SELECT s.ch[3] FROM s",
        _registry,
    )
    assert err.code is ErrorCode.STREAM_NOT_FOUND
    assert "has 2 streams" in err.message


def test_a_per_element_op_broadcasts_over_an_array_call(_registry: Registry) -> None:
    """The headline shape: one channelsplit, one volume per channel."""
    g = _dyn(
        "SELECT volume(ffmpeg.channelsplit(a.audio[1]), 0.5) FROM input('x.mp4') a",
        _registry,
    )
    assert _filters(g) == ["channelsplit", "volume", "volume"]
    assert g.nodes["n2"].inputs == ["n1:0"]
    assert g.nodes["n3"].inputs == ["n1:1"]


def test_an_array_call_threads_its_sources_provenance_to_every_element(
    _registry: Registry,
) -> None:
    """1:N fan: every channel of an eng track is an eng channel (this is NOT
    `_agreed_source`, which answers the N:1 question)."""
    g = _dyn(
        "SELECT ffmpeg.channelsplit(a.audio[2]) FROM input('x.mp4') a",
        _registry,
        {"a": _probe_result(per_audio_tags=[{"language": "eng"}, {"language": "fra"}])},
    )
    assert [o.metadata for o in g.outputs] == [{"language": "fra"}, {"language": "fra"}]


def test_two_readers_of_one_array_pad_get_an_asplit(_registry: Registry) -> None:
    """channelsplit pads are ordinary pads -- consume-once like any other."""
    g = insert_splits(
        _dyn(
            "WITH s AS (SELECT ffmpeg.channelsplit(a.audio[1]) AS ch "
            "FROM input('x.mp4') a) SELECT s.ch[1], volume(s.ch[1], 0.5) FROM s",
            _registry,
        )
    )
    asplit = next(node for node in g.nodes.values() if node.filter == "asplit")
    assert asplit.inputs == ["n1:0"]
    assert asplit.args == {"n": 2}


def test_an_array_call_cannot_also_broadcast_over_an_array(_registry: Registry) -> None:
    err = _reject_dyn(
        "SELECT ffmpeg.channelsplit(a.audio) FROM input('x.mp4') a",
        _registry,
        {"a": _probe_result(per_audio_tags=[{}, {}])},
    )
    assert err.code is ErrorCode.UDF_ARG_TYPE
    assert "returns an array, so it cannot also broadcast over one" in err.message
    assert err.hint is not None and "a.audio[1]" in err.hint


def test_an_array_call_checks_its_input_pad_type(_registry: Registry) -> None:
    err = _reject_dyn(
        "SELECT ffmpeg.channelsplit(a.video[1]) FROM input('x.mp4') a", _registry
    )
    assert err.code is ErrorCode.UDF_ARG_TYPE
    assert "it takes audio as its stream input, got (video)" in err.message


def test_an_array_call_takes_positional_options_like_any_other(
    _registry: Registry,
) -> None:
    """Calls are uniform: the pad comes first, then channelsplit's own options
    in ffmpeg's order (channel_layout, channels)."""
    g = _dyn(
        "SELECT ffmpeg.channelsplit(a.audio[1], '5.1') FROM input('x.mp4') a", _registry
    )
    assert g.nodes["n1"].args == {"channel_layout": "5.1"}
    assert g.nodes["n1"].outputs == ["audio"] * 6


def test_an_array_calls_positional_option_is_validated_as_that_option(
    _registry: Registry,
) -> None:
    err = _reject_dyn(
        "SELECT ffmpeg.channelsplit(a.audio[1], 2) FROM input('x.mp4') a", _registry
    )
    assert err.code is ErrorCode.FILTER_OPTION_TYPE
    assert "'channel_layout' of filter 'channelsplit'" in err.message


def test_an_array_call_rejects_an_unknown_option(_registry: Registry) -> None:
    err = _reject_dyn(
        "SELECT ffmpeg.channelsplit(a.audio[1], zzz => 1) FROM input('x.mp4') a",
        _registry,
    )
    assert err.code is ErrorCode.UNKNOWN_FILTER_OPTION
    assert err.hint is not None and "channel_layout" in err.hint


def test_an_array_filter_has_no_timeline_support(_registry: Registry) -> None:
    """None of the three is T-flagged in `ffmpeg -filters`, so `enable` is
    rejected the ordinary way rather than silently accepted."""
    err = _reject_dyn(
        "SELECT ffmpeg.channelsplit(a.audio[1], enable => 'gte(t,1)') "
        "FROM input('x.mp4') a",
        _registry,
    )
    assert err.code is ErrorCode.UNKNOWN_FILTER_OPTION
    assert "does not flag 'channelsplit' as supporting timeline editing" in err.message


def test_an_array_filter_this_ffmpeg_lacks_is_an_unknown_function(
    _registry: Registry, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The table says what SHAPE the call has, never that the filter exists:
    a build whose `-help filter=channelsplit` says nothing does not have it,
    and the namespace answers exactly as it does for any other unknown name."""
    monkeypatch.delitem(_HELP_FIXTURES, "channelsplit")
    err = _reject_dyn(
        "SELECT ffmpeg.channelsplit(a.audio[1]) FROM input('x.mp4') a", _registry
    )
    assert err.code is ErrorCode.UNKNOWN_FUNCTION
    assert err.hint is not None and "variable pad count" in err.hint


def test_an_array_call_needs_a_registry() -> None:
    err = _reject_dyn("SELECT ffmpeg.channelsplit(a.audio[1]) FROM input('x.mp4') a", None)
    assert err.code is ErrorCode.UNKNOWN_FUNCTION
    assert err.hint is not None and "provisioner" in err.hint


def test_no_ffmpeg_turns_the_array_table_off_too() -> None:
    """The table says what SHAPE the call has; the registry says whether the
    filter exists at all, and with no ffmpeg it cannot say yes."""
    err = _reject_dyn(
        "SELECT ffmpeg.channelsplit(a.audio[1]) FROM input('x.mp4') a", None
    )
    assert err.code is ErrorCode.UNKNOWN_FUNCTION
    assert err.hint is not None and "provisioner" in err.hint


def test_an_array_call_emits_one_label_per_pad(_registry: Registry) -> None:
    g = _dyn(
        "SELECT volume(ffmpeg.channelsplit(a.audio[1]), 2) FROM input('x.mp4') a",
        _registry,
    )
    assert emit(insert_splits(g)).filter_complex == (
        "[0:a:0]channelsplit[n10][n11];[n10]volume=volume=2[out0];[n11]volume=volume=2[out1]"
    )


# ---------------------------------------------------------------------------
# everything still works OFFLINE, through the captured snapshot
# ---------------------------------------------------------------------------
#
# There is no curated subset to fall back on, so "does this compile without
# ffmpeg" is answered by handing lowering a registry built from
# `tests/data/reference_registry.json` instead of from PATH. These pin that
# generated sources, the timeline `enable` option, the array-returning trio and
# broadcasting are all reachable that way, with `binaries.ffmpeg_path` stubbed
# to None (not just `shutil.which`, so this stays offline even when the
# `static-ffmpeg` provisioner is installed and cached) and `subprocess.run`
# booby-trapped so a single introspection call fails loudly rather than
# silently rescuing the test.


@pytest.fixture
def _offline(monkeypatch: pytest.MonkeyPatch) -> Registry:
    def no_ffmpeg() -> str | None:
        return None

    def boom(*args: object, **kwargs: object) -> object:
        raise AssertionError("an offline compile must never spawn a subprocess")

    monkeypatch.setattr(registry_module.binaries, "ffmpeg_path", no_ffmpeg)
    monkeypatch.setattr(registry_module.subprocess, "run", boom)
    return load_reference(SNAPSHOT_PATH)


def test_a_source_compiles_offline(_offline: Registry) -> None:
    g = lower(
        resolve(parse("SELECT t.video[1] FROM ffmpeg.testsrc(duration => 2, size => '320x240') t")),
        {},
        registry=_offline,
    )
    node = g.nodes["n1"]
    assert node.filter == "testsrc"
    assert node.inputs == []
    assert node.args == {"duration": 2, "size": "320x240"}


def test_enable_compiles_offline(_offline: Registry) -> None:
    g = lower(
        resolve(parse("SELECT gblur(a.video[1], 5, enable => 'between(t,2,5)') FROM input('x.mp4') a")),
        {},
        registry=_offline,
    )
    assert g.nodes["n1"].args == {"sigma": 5, "enable": "between(t,2,5)"}


def test_enable_still_requires_the_timeline_flag_offline(_offline: Registry) -> None:
    """scale is not T-flagged, and the snapshot carries that flag verbatim."""
    with pytest.raises(FfrwdError) as excinfo:
        lower(
            resolve(parse("SELECT scale(a.video[1], 640, enable => 'gt(t,1)') FROM input('x.mp4') a")),
            {},
            registry=_offline,
        )
    assert excinfo.value.code is ErrorCode.UNKNOWN_FILTER_OPTION


def test_the_array_returning_trio_compiles_offline(_offline: Registry) -> None:
    g = lower(
        resolve(parse("SELECT ffmpeg.channelsplit(a.audio[1]) FROM input('x.mp4') a")),
        {},
        registry=_offline,
    )
    assert g.nodes["n1"].filter == "channelsplit"
    assert g.nodes["n1"].outputs == ["audio", "audio"]


def test_the_n_input_trio_compiles_offline(_offline: Registry) -> None:
    g = lower(
        resolve(parse("SELECT amix(a.audio[1], a.audio[2]) FROM input('x.mp4') a")),
        {},
        registry=_offline,
    )
    assert g.nodes["n1"].args == {"inputs": 2}


def test_broadcasting_still_works_offline(_offline: Registry) -> None:
    g = lower(
        resolve(parse("SELECT volume(a.audio, 0.5) FROM input('x.mp4') a")),
        {"a": _probe_result(audios=2)},
        registry=_offline,
    )
    assert _filters(g) == ["volume", "volume"]
    assert g.nodes["n1"].inputs == ["src:a:a:0"]
    assert g.nodes["n2"].inputs == ["src:a:a:1"]


def test_positional_options_bind_offline_exactly_as_they_do_live(
    _offline: Registry,
) -> None:
    """The snapshot preserves ffmpeg's option ORDER, not just its content."""
    g = lower(
        resolve(parse("SELECT crop(a.video[1], 100, 50, 10, 20) FROM input('x.mp4') a")),
        {},
        registry=_offline,
    )
    assert g.nodes["n1"].args == {"out_w": 100, "out_h": 50, "x": 10, "y": 20}


# ---------------------------------------------------------------------------
# fixed-count N-INPUT filters (amix / hstack / vstack)
# ---------------------------------------------------------------------------
#
# The mirror of the array-RETURNING trio: `N->1` filters whose INPUT pad count
# is fixed by their `inputs` option, so the v1 pad scope check excludes them from the
# registry's tables even though the count is statically knowable. Reachable
# under BOTH spellings -- no Postgres grammar claims these three names.


def test_amix_is_callable_bare_with_two_streams(_registry: Registry) -> None:
    g = _dyn(
        "SELECT amix(a.audio[1], a.audio[2]) FROM input('x.mp4') a", _registry
    )
    node = g.nodes["n1"]
    assert node.filter == "amix"
    assert node.inputs == ["src:a:a:0", "src:a:a:1"]
    assert node.outputs == ["audio"]
    assert node.args == {"inputs": 2}


def test_hstack_and_vstack_are_callable_bare(_registry: Registry) -> None:
    for name in ("hstack", "vstack"):
        g = _dyn(
            f"SELECT {name}(a.video[1], b.video[1]) FROM input('x.mp4') a, input('y.mp4') b",
            _registry,
        )
        assert g.nodes["n1"].filter == name
        assert g.nodes["n1"].outputs == ["video"]
        assert g.nodes["n1"].args == {"inputs": 2}


def test_an_n_input_filter_is_reachable_through_the_namespace_too(
    _registry: Registry,
) -> None:
    g = _dyn(
        "SELECT ffmpeg.amix(a.audio[1], a.audio[2]) FROM input('x.mp4') a", _registry
    )
    assert g.nodes["n1"].filter == "amix"


def test_more_streams_need_the_count_option_to_agree(_registry: Registry) -> None:
    g = _dyn(
        "SELECT amix(a.audio[1], a.audio[2], a.audio[3], inputs => 3) "
        "FROM input('x.mp4') a",
        _registry,
    )
    node = g.nodes["n1"]
    assert node.args == {"inputs": 3}
    assert node.inputs == ["src:a:a:0", "src:a:a:1", "src:a:a:2"]


def test_the_count_option_can_be_positional_too(_registry: Registry) -> None:
    """`inputs` is amix's FIRST option, so it is also its first positional slot
    -- the same uniform rule every other filter follows."""
    g = _dyn(
        "SELECT amix(a.audio[1], a.audio[2], a.audio[3], 3) FROM input('x.mp4') a",
        _registry,
    )
    assert g.nodes["n1"].args == {"inputs": 3}


def test_a_count_that_disagrees_with_the_streams_is_udf_arg_type(
    _registry: Registry,
) -> None:
    err = _reject_dyn(
        "SELECT amix(a.audio[1], a.audio[2], a.audio[3]) FROM input('x.mp4') a",
        _registry,
    )
    assert err.code is ErrorCode.UDF_ARG_TYPE
    assert "was given 3 streams" in err.message
    assert "'inputs' option says 2" in err.message


def test_a_count_larger_than_the_streams_is_also_rejected(
    _registry: Registry,
) -> None:
    err = _reject_dyn(
        "SELECT amix(a.audio[1], a.audio[2], inputs => 5) FROM input('x.mp4') a",
        _registry,
    )
    assert err.code is ErrorCode.UDF_ARG_TYPE
    assert "was given 2 streams" in err.message
    assert "says 5" in err.message


def test_an_n_input_filter_type_checks_every_stream(_registry: Registry) -> None:
    err = _reject_dyn(
        "SELECT amix(a.audio[1], a.video[1]) FROM input('x.mp4') a", _registry
    )
    assert err.code is ErrorCode.UDF_ARG_TYPE
    assert "its stream inputs are all audio" in err.message


def test_an_n_input_filter_needs_at_least_one_stream(_registry: Registry) -> None:
    err = _reject_dyn("SELECT amix(2) FROM input('x.mp4') a", _registry)
    assert err.code is ErrorCode.UDF_ARG_TYPE
    assert "no streams" in err.message


def test_an_n_input_filter_takes_its_other_options_too(_registry: Registry) -> None:
    g = _dyn(
        "SELECT amix(a.audio[1], a.audio[2], duration => 'shortest') "
        "FROM input('x.mp4') a",
        _registry,
    )
    assert g.nodes["n1"].args == {"duration": "shortest", "inputs": 2}


def test_an_n_input_option_is_validated_like_any_other(_registry: Registry) -> None:
    err = _reject_dyn(
        "SELECT amix(a.audio[1], a.audio[2], duration => 'nope') FROM input('x.mp4') a",
        _registry,
    )
    assert err.code is ErrorCode.FILTER_OPTION_TYPE
    assert "'duration' of filter 'amix'" in err.message


def test_an_n_input_filter_broadcasts_like_any_other(_registry: Registry) -> None:
    """`a.audio` is a 2-element array, `b.audio[1]` a scalar that repeats."""
    g = _dyn(
        "SELECT amix(a.audio, b.audio[1]) FROM input('x.mp4') a, input('y.mp4') b",
        _registry,
        {"a": _probe_result(audios=2), "b": _probe_result(audios=1)},
    )
    assert _filters(g) == ["amix", "amix"]
    assert g.nodes["n1"].inputs == ["src:a:a:0", "src:b:a:0"]
    assert g.nodes["n2"].inputs == ["src:a:a:1", "src:b:a:0"]


def test_an_n_input_filter_nests(_registry: Registry) -> None:
    g = _dyn(
        "SELECT volume(amix(a.audio[1], a.audio[2]), 0.5) FROM input('x.mp4') a",
        _registry,
    )
    assert _filters(g) == ["amix", "volume"]
    assert g.nodes["n2"].inputs == ["n1"]


def test_an_n_input_filter_with_no_help_output_is_still_callable(
    _registry: Registry, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An N-input filter is an ordinary registry member now (`-filters` alone
    says its shape is real), so a build whose `-help filter=amix` fails still
    admits the call -- it just has no options to bind against, exactly like
    any other filter `-help` fails for."""
    monkeypatch.delitem(_HELP_FIXTURES, "amix")
    g = _dyn("SELECT amix(a.audio[1], a.audio[2]) FROM input('x.mp4') a", _registry)
    node = g.nodes["n1"]
    assert node.filter == "amix"
    assert node.args == {}
    err = _reject_dyn(
        "SELECT amix(a.audio[1], a.audio[2], duration => 'shortest') "
        "FROM input('x.mp4') a",
        _registry,
    )
    assert err.code is ErrorCode.UNKNOWN_FILTER_OPTION


def test_an_n_input_filter_has_no_timeline_support(_registry: Registry) -> None:
    """None of the three is T-flagged in `ffmpeg -filters`."""
    err = _reject_dyn(
        "SELECT amix(a.audio[1], a.audio[2], enable => 'gte(t,1)') "
        "FROM input('x.mp4') a",
        _registry,
    )
    assert err.code is ErrorCode.UNKNOWN_FILTER_OPTION


def test_an_n_input_node_is_split_like_any_other(_registry: Registry) -> None:
    g = insert_splits(
        _dyn(
            "WITH c AS (SELECT gblur(a.video[1], sigma => 2) AS f FROM input('x.mp4') a) "
            "SELECT hstack(c.f, c.f) FROM c",
            _registry,
        )
    )
    assert _filters(g) == ["gblur", "split", "hstack"]


# ---------------------------------------------------------------------------
# amerge / join / interleave / ainterleave: N-input filters, `nb_inputs`
# ---------------------------------------------------------------------------
#
# Same derivation as amix/hstack/vstack. amerge and join count via `inputs`;
# interleave/ainterleave count via `nb_inputs` -- VERIFIED against a real
# ffmpeg 9.0.1 (`Registry.options`): `n` is `nb_inputs`'s adjacent alias, so
# the registry's dedup rule keeps the longer name, and `nb_inputs` is what a
# positional binds too.


def test_amerge_is_callable_bare_and_counts_via_inputs(_registry: Registry) -> None:
    g = _dyn("SELECT amerge(a.audio[1], a.audio[2]) FROM input('x.mp4') a", _registry)
    node = g.nodes["n1"]
    assert node.filter == "amerge"
    assert node.inputs == ["src:a:a:0", "src:a:a:1"]
    assert node.outputs == ["audio"]
    assert node.args == {"inputs": 2}


def test_join_is_a_reserved_keyword_so_it_needs_the_namespace(
    _registry: Registry,
) -> None:
    """Unlike amerge, bare `join(...)` collides with the JOIN clause keyword
    itself -- a PARSE_ERROR from sqlglot before lowering ever runs, not an
    UNKNOWN_FUNCTION. `ffmpeg.join(...)` reaches it, same as any other
    namespace-only collision."""
    err = _reject_dyn(
        "SELECT join(a.audio[1], a.audio[2]) FROM input('x.mp4') a", _registry
    )
    assert err.code is ErrorCode.PARSE_ERROR
    g = _dyn(
        "SELECT ffmpeg.join(a.audio[1], a.audio[2]) FROM input('x.mp4') a", _registry
    )
    node = g.nodes["n1"]
    assert node.filter == "join"
    assert node.args == {"inputs": 2}


def test_interleave_and_ainterleave_count_via_nb_inputs(_registry: Registry) -> None:
    g = _dyn(
        "SELECT interleave(a.video[1], b.video[1]) FROM input('x.mp4') a, input('y.mp4') b",
        _registry,
    )
    node = g.nodes["n1"]
    assert node.filter == "interleave"
    assert node.outputs == ["video"]
    assert node.args == {"nb_inputs": 2}

    g = _dyn(
        "SELECT ainterleave(a.audio[1], a.audio[2]) FROM input('x.mp4') a", _registry
    )
    node = g.nodes["n1"]
    assert node.filter == "ainterleave"
    assert node.outputs == ["audio"]
    assert node.args == {"nb_inputs": 2}


def test_interleaves_count_option_agrees_with_the_streams_supplied(
    _registry: Registry,
) -> None:
    g = _dyn(
        "SELECT interleave(a.video[1], b.video[1], c.video[1], nb_inputs => 3) "
        "FROM input('x.mp4') a, input('y.mp4') b, input('z.mp4') c",
        _registry,
    )
    assert g.nodes["n1"].args == {"nb_inputs": 3}
    err = _reject_dyn(
        "SELECT interleave(a.video[1], b.video[1], c.video[1]) "
        "FROM input('x.mp4') a, input('y.mp4') b, input('z.mp4') c",
        _registry,
    )
    assert err.code is ErrorCode.UDF_ARG_TYPE
    assert "'nb_inputs' option says 2" in err.message


def test_amerge_rejects_a_video_stream(_registry: Registry) -> None:
    err = _reject_dyn(
        "SELECT amerge(a.audio[1], a.video[1]) FROM input('x.mp4') a", _registry
    )
    assert err.code is ErrorCode.UDF_ARG_TYPE
    assert "its stream inputs are all audio" in err.message


def test_interleave_rejects_an_audio_stream(_registry: Registry) -> None:
    err = _reject_dyn(
        "SELECT interleave(a.video[1], a.audio[1]) FROM input('x.mp4') a", _registry
    )
    assert err.code is ErrorCode.UDF_ARG_TYPE
    assert "its stream inputs are all video" in err.message


def test_amerge_join_interleave_ainterleave_reachable_through_the_namespace(
    _registry: Registry,
) -> None:
    for name in ("amerge", "join"):
        g = _dyn(
            f"SELECT ffmpeg.{name}(a.audio[1], a.audio[2]) FROM input('x.mp4') a",
            _registry,
        )
        assert g.nodes["n1"].filter == name
    g = _dyn(
        "SELECT ffmpeg.interleave(a.video[1], b.video[1]) "
        "FROM input('x.mp4') a, input('y.mp4') b",
        _registry,
    )
    assert g.nodes["n1"].filter == "interleave"


# ---------------------------------------------------------------------------
# ladspa joins the N_INPUT table -- the one entry with no count option
# ---------------------------------------------------------------------------
#
# Every other N_INPUT filter's pad count is fixed by an option (`inputs` or
# `nb_inputs`); ladspa's is fixed by the loaded LADSPA plugin's own ports, so
# there is no option to read it back from or write it to -- whatever count of
# audio streams the call supplies IS the count.


def test_ladspa_is_callable_bare_with_named_options(_registry: Registry) -> None:
    g = _dyn(
        "SELECT ladspa(a.audio[1], file => 'amp', plugin => 'amp_mono') "
        "FROM input('x.mp4') a",
        _registry,
    )
    node = g.nodes["n1"]
    assert node.filter == "ladspa"
    assert node.inputs == ["src:a:a:0"]
    assert node.outputs == ["audio"]
    assert node.args == {"file": "amp", "plugin": "amp_mono"}


def test_ladspa_takes_any_number_of_streams_and_emits_no_count_option(
    _registry: Registry,
) -> None:
    g = _dyn(
        "SELECT ladspa(a.audio[1], a.audio[2], file => 'amp') FROM input('x.mp4') a",
        _registry
    )
    node = g.nodes["n1"]
    assert node.filter == "ladspa"
    assert node.inputs == ["src:a:a:0", "src:a:a:1"]
    assert node.args == {"file": "amp"}
    assert "inputs" not in node.args


def test_ladspa_rejects_an_unknown_option(_registry: Registry) -> None:
    err = _reject_dyn(
        "SELECT ladspa(a.audio[1], file => 'amp', bogus => 1) "
        "FROM input('x.mp4') a",
        _registry,
    )
    assert err.code is ErrorCode.UNKNOWN_FILTER_OPTION


def test_ladspa_rejects_a_video_stream(_registry: Registry) -> None:
    err = _reject_dyn(
        "SELECT ladspa(a.video[1], file => 'amp') FROM input('x.mp4') a", _registry
    )
    assert err.code is ErrorCode.UDF_ARG_TYPE
    assert "its stream inputs are all audio" in err.message


def test_ladspa_reachable_through_the_namespace(_registry: Registry) -> None:
    g = _dyn(
        "SELECT ffmpeg.ladspa(a.audio[1], file => 'amp') FROM input('x.mp4') a",
        _registry,
    )
    assert g.nodes["n1"].filter == "ladspa"


# ---------------------------------------------------------------------------
# N-input filters are derived from the registry, not hand-listed
# ---------------------------------------------------------------------------
#
# registry.py includes every `N->A`/`N->V` filter as an ordinary member of
# `Registry.names()`, marked `DynamicFilter.n_input`; lower.py's
# `_n_input_spec` reads its call shape (stream/output/option/fallback) off
# that filter's own option table instead of a hand-listed table. `xstack`
# (`N->V`, `inputs`/`layout`/`grid`/`shortest`/`fill`) is a newly admitted
# filter -- it was excluded before this and reachable through no table.

# The original hand-listed table's exact values for the nine filters it
# carried, as (stream, output, option, fallback, emit_default).
_ORIGINAL_N_INPUT_TABLE: dict[str, tuple[str, str, str | None, int, bool]] = {
    "amix": ("audio", "audio", "inputs", 2, True),
    "hstack": ("video", "video", "inputs", 2, True),
    "vstack": ("video", "video", "inputs", 2, True),
    "acrossfade": ("audio", "audio", "inputs", 2, False),
    "amerge": ("audio", "audio", "inputs", 2, True),
    "join": ("audio", "audio", "inputs", 2, True),
    "interleave": ("video", "video", "nb_inputs", 2, True),
    "ainterleave": ("audio", "audio", "nb_inputs", 2, True),
    "ladspa": ("audio", "audio", None, 0, False),
}


def test_derived_n_input_spec_reproduces_the_original_table_exactly() -> None:
    """Derivation from the registry must land on exactly the values the
    hand-listed table carried for its nine filters -- not merely compile the
    same commands, the same `_NInputFilter` fields, so a later refactor of
    the derivation can diff against this."""
    registry = _snapshot_registry()
    for name, (stream, output, option, fallback, emit_default) in _ORIGINAL_N_INPUT_TABLE.items():
        dynamic = registry.get(name)
        assert dynamic is not None and dynamic.n_input, name
        options = registry.options(name)
        assert options is not None, name
        spec = lower_module._n_input_spec(name, dynamic, options)
        assert spec == lower_module._NInputFilter(
            name=name,
            stream=stream,  # type: ignore[arg-type]
            output=output,  # type: ignore[arg-type]
            option=option,
            fallback=fallback,
            emit_default=emit_default,
        ), name


def test_xstack_is_callable_bare_and_positionally() -> None:
    g = _lower(
        "SELECT xstack(a.video[1], b.video[1], grid => '2x2') "
        "FROM input('x.mp4') a, input('y.mp4') b",
    )
    node = next(iter(g.nodes.values()))
    assert node.filter == "xstack"
    assert node.outputs == ["video"]
    assert node.args == {"inputs": 2, "grid": "2x2"}


def test_xstack_is_callable_variadic() -> None:
    g = _lower(
        "SELECT ffmpeg.xstack(VARIADIC a.video, grid => '2x2') FROM input('x.mp4') a",
        {"a": _probe_result(videos=4)},
    )
    node = next(iter(g.nodes.values()))
    assert node.filter == "xstack"
    assert node.args == {"inputs": 4, "grid": "2x2"}
    assert len(node.inputs) == 4


def test_xstack_count_disagreement_is_udf_arg_type() -> None:
    """The same disagreement rejection an original N_INPUT filter gets
    (`test_a_count_that_disagrees_with_the_streams_is_udf_arg_type`), now
    exercised on a filter the registry admits rather than a curated table."""
    err = _reject_lower(
        "SELECT xstack(a.video[1], b.video[1], inputs => 3) "
        "FROM input('x.mp4') a, input('y.mp4') b",
        {},
    )
    assert err.code is ErrorCode.UDF_ARG_TYPE
    assert "was given 2 streams" in err.message
    assert "'inputs' option says 3" in err.message


def test_streamselect_and_astreamselect_stay_excluded() -> None:
    """Both are `N->N` -- dynamic on the OUTPUT side too, same as `concat` --
    so they stay excluded from the registry, unlike an ordinary `N->1`
    n-input filter."""
    registry = _snapshot_registry()
    for name in ("streamselect", "astreamselect"):
        assert registry.get(name) is None, name
        assert name not in registry.names(), name


# ---------------------------------------------------------------------------
# FROM ffmpeg.<source>(...) alias
# ---------------------------------------------------------------------------
#
# Offline, against the same fixture registry: the fixture's `-filters` block
# carries `testsrc` (|->V), `anullsrc`/`sine` (|->A), plus `avsynctest`
# (|->AV) and `movie` (|->N), which the v1 scope check excludes -- so the
# excluded half is exercised without an ffmpeg on the machine either.


def test_a_source_lowers_to_a_zero_input_node(_registry: Registry) -> None:
    g = _dyn("SELECT t.video[1] FROM ffmpeg.testsrc(duration => 2) t", _registry)
    node = g.nodes["n1"]
    assert node.filter == "testsrc"
    assert node.args == {"duration": 2}
    assert node.inputs == []
    assert node.outputs == ["video"]
    assert _outputs(g) == [("n1", "video", None)]


def test_a_source_takes_no_input_index(_registry: Registry) -> None:
    """The whole point: a source is a filter, not an `-i`. It appears in
    neither `input_paths` nor `sources`, so nothing downstream can mistake it
    for a file."""
    res = resolve(parse("SELECT t.video[1] FROM ffmpeg.testsrc(duration => 2) t"))
    assert res.input_paths == []
    assert res.sources == {}
    assert set(res.source_filters) == {"t"}
    g = _dyn("SELECT t.video[1] FROM ffmpeg.testsrc(duration => 2) t", _registry)
    assert g.input_paths == []
    assert g.sources == {}


def test_a_source_with_no_options_sets_no_args(_registry: Registry) -> None:
    g = _dyn("SELECT t.video[1] FROM ffmpeg.testsrc() t", _registry)
    assert g.nodes["n1"].args == {}


def test_a_video_source_answers_to_its_subscript(_registry: Registry) -> None:
    g = _dyn("SELECT t.video[1] FROM ffmpeg.testsrc(duration => 2) t", _registry)
    assert _outputs(g) == [("n1", "video", None)]


def test_a_source_bare_array_is_a_length_one_array(_registry: Registry) -> None:
    """`t.video` is an ARRAY of one element, not a scalar -- it splats into
    exactly one Output and broadcasts a call exactly once. No probe is
    consulted: the length is a property of the source's single output pad."""
    g = _dyn("SELECT t.video FROM ffmpeg.testsrc(duration => 2) t", _registry)
    assert _outputs(g) == [("n1", "video", None)]
    g = _dyn("SELECT gblur(t.video, sigma => 2) FROM ffmpeg.testsrc() t", _registry)
    assert _filters(g) == ["testsrc", "gblur"]
    assert _outputs(g) == [("n2", "video", None)]


def test_a_source_star_is_its_one_column(_registry: Registry) -> None:
    g = _dyn("SELECT t.* FROM ffmpeg.testsrc(duration => 2) t", _registry)
    assert _outputs(g) == [("n1", "video", None)]
    g = _dyn("SELECT * FROM ffmpeg.anullsrc(duration => 1) s", _registry)
    assert _outputs(g) == [("n1", "audio", None)]


def test_an_audio_source_answers_to_audio(_registry: Registry) -> None:
    g = _dyn("SELECT s.audio[1] FROM ffmpeg.sine(frequency => 440) s", _registry)
    assert g.nodes["n1"].outputs == ["audio"]
    assert _outputs(g) == [("n1", "audio", None)]


def test_a_source_rejects_the_other_types_column(_registry: Registry) -> None:
    """Statically typed: one output pad means the other type does not exist,
    and the message says what the source DOES produce."""
    err = _reject_dyn("SELECT t.audio[1] FROM ffmpeg.testsrc() t", _registry)
    assert err.code is ErrorCode.STREAM_NOT_FOUND
    assert "ffmpeg.testsrc produces 1 video stream" in err.message

    err = _reject_dyn("SELECT s.video[1] FROM ffmpeg.anullsrc() s", _registry)
    assert err.code is ErrorCode.STREAM_NOT_FOUND
    assert "ffmpeg.anullsrc produces 1 audio stream" in err.message

    err = _reject_dyn("SELECT t.subtitle[1] FROM ffmpeg.testsrc() t", _registry)
    assert err.code is ErrorCode.STREAM_NOT_FOUND
    assert "ffmpeg.testsrc produces 1 video stream" in err.message


def test_a_source_column_of_the_wrong_type_names_what_it_produces(
    _registry: Registry,
) -> None:
    err = _reject_dyn("SELECT s.video[1] FROM ffmpeg.anullsrc() s", _registry)
    assert err.code is ErrorCode.STREAM_NOT_FOUND
    assert "ffmpeg.anullsrc produces 1 audio stream" in err.message
    assert err.hint is not None and "s.audio" in err.hint


def test_a_source_subscript_is_bounded_statically(_registry: Registry) -> None:
    err = _reject_dyn("SELECT t.video[2] FROM ffmpeg.testsrc() t", _registry)
    assert err.code is ErrorCode.STREAM_NOT_FOUND
    assert "'t.video[2]' does not exist" in err.message
    assert "ffmpeg.testsrc produces 1 video stream" in err.message


def test_a_source_rejects_an_unknown_column(_registry: Registry) -> None:
    err = _reject_dyn("SELECT t.bogus FROM ffmpeg.testsrc() t", _registry)
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "unknown column 't.bogus'" in err.message
    assert err.hint is not None and "t.video" in err.hint


def test_a_source_node_is_minted_once_per_alias(_registry: Registry) -> None:
    """Memoized on first column access, so fan-out is the split pass's
    ordinary business -- never a second generator."""
    g = _dyn(
        "SELECT gblur(t.video[1], sigma => 2), t.video[1] FROM ffmpeg.testsrc() t", _registry
    )
    assert _filters(g) == ["testsrc", "gblur"]
    g = insert_splits(g)
    assert _filters(g) == ["testsrc", "split", "gblur"]
    assert g.nodes["n1_split"].inputs == ["n1"]


def test_an_unused_source_alias_mints_no_node(_registry: Registry) -> None:
    g = _dyn(
        "SELECT a.video[1] FROM input('x.mp4') a, ffmpeg.testsrc(duration => 2) t",
        _registry,
    )
    assert g.nodes == {}
    assert _outputs(g) == [("src:a:v:0", "video", None)]


def test_a_source_validates_its_options_against_the_registry(
    _registry: Registry,
) -> None:
    err = _reject_dyn("SELECT t.video[1] FROM ffmpeg.testsrc(durationn => 2) t", _registry)
    assert err.code is ErrorCode.UNKNOWN_FILTER_OPTION
    assert "filter 'testsrc' has no option 'durationn'" in err.message
    assert err.hint == "did you mean duration => ...?"

    err = _reject_dyn("SELECT t.video[1] FROM ffmpeg.testsrc(decimals => 'x') t", _registry)
    assert err.code is ErrorCode.FILTER_OPTION_TYPE
    assert "expects a number" in err.message


def test_a_source_option_range_is_checked(_registry: Registry) -> None:
    err = _reject_dyn("SELECT t.video[1] FROM ffmpeg.testsrc(decimals => 99) t", _registry)
    assert err.code is ErrorCode.FILTER_OPTION_TYPE
    assert "from 0 to 17" in err.message


def test_source_options_keep_their_written_order(_registry: Registry) -> None:
    g = _dyn(
        "SELECT t.video[1] FROM ffmpeg.testsrc(size => '320x240', rate => 15, "
        "duration => 2) t",
        _registry,
    )
    assert list(g.nodes["n1"].args.items()) == [
        ("size", "320x240"),
        ("rate", 15),
        ("duration", 2),
    ]
    assert (
        emit(insert_splits(g)).filter_complex
        == "testsrc=size=320x240:rate=15:duration=2[out0]"
    )


def test_an_unknown_source_suggests_a_real_one(_registry: Registry) -> None:
    err = _reject_dyn("SELECT t.video[1] FROM ffmpeg.testsrcc() t", _registry)
    assert err.code is ErrorCode.UNKNOWN_FUNCTION
    assert "unknown generated source ffmpeg.testsrcc()" in err.message
    assert err.hint == "did you mean ffmpeg.testsrc()?"


def test_an_excluded_source_gets_the_exclusion_message(_registry: Registry) -> None:
    """`avsynctest` (|->AV) and `movie` (|->N) are in the fixture's -filters
    output and excluded by the v1 scope check, so the registry never retained
    them -- they are indistinguishable from a typo here and land on the same
    rejection, whose hint states the exclusion."""
    for name in ("avsynctest", "movie", "amovie"):
        err = _reject_dyn(f"SELECT t.video[1] FROM ffmpeg.{name}() t", _registry)
        assert err.code is ErrorCode.UNKNOWN_FUNCTION, name
        assert err.hint is not None
        assert "more than one output pad (avsynctest)" in err.hint
        assert "variable pad count (movie, amovie)" in err.hint


def test_a_sink_is_not_a_source_either(_registry: Registry) -> None:
    err = _reject_dyn("SELECT t.audio[1] FROM ffmpeg.anullsink() t", _registry)
    assert err.code is ErrorCode.UNKNOWN_FUNCTION


def test_a_regular_filter_in_from_says_it_takes_inputs(_registry: Registry) -> None:
    """The one excluded case that IS positively identifiable: the name is a
    real filter of this ffmpeg, it just has input pads."""
    err = _reject_dyn("SELECT t.video[1] FROM ffmpeg.gblur(sigma => 2) t", _registry)
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "ffmpeg.gblur is an ffmpeg filter, not a source" in err.message
    assert err.hint is not None and "SELECT ffmpeg.gblur(a.video[1])" in err.hint


def test_a_source_needs_a_registry() -> None:
    err = _reject_dyn("SELECT t.video[1] FROM ffmpeg.testsrc(duration => 2) t", None)
    assert err.code is ErrorCode.UNKNOWN_FUNCTION
    assert err.hint is not None and "provisioner" in err.hint


def test_no_ffmpeg_turns_the_source_namespace_off() -> None:
    err = _reject_dyn("SELECT t.video[1] FROM ffmpeg.testsrc(duration => 2) t", None)
    assert err.code is ErrorCode.UNKNOWN_FUNCTION
    assert err.hint is not None and "provisioner" in err.hint


def test_where_on_a_source_alias_points_at_duration(_registry: Registry) -> None:
    err = _reject_dyn(
        "SELECT t.video[1] FROM ffmpeg.testsrc(duration => 2) t WHERE t.t <= 1", _registry
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "'t' is a generated source" in err.message
    assert err.hint is not None and "duration => 30" in err.hint


def test_a_source_time_column_is_not_a_stream(_registry: Registry) -> None:
    err = _reject_dyn("SELECT t.t FROM ffmpeg.testsrc() t", _registry)
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "'t.t' is a time column, not a stream" in err.message


def test_a_source_carries_no_provenance(_registry: Registry) -> None:
    """Nothing was probed, because nothing was read."""
    g = _dyn("SELECT t.video[1] FROM ffmpeg.testsrc() t", _registry)
    assert g.outputs[0].metadata == {}


def test_a_source_works_inside_a_cte(_registry: Registry) -> None:
    g = _dyn(
        "WITH bg AS (SELECT t.video[1] AS v FROM ffmpeg.testsrc(duration => 2) t) "
        "SELECT gblur(bg.v, sigma => 2) FROM bg",
        _registry,
    )
    assert _filters(g) == ["testsrc", "gblur"]
    assert _outputs(g) == [("n2", "video", None)]


def test_the_silent_audio_union_all_branch(_registry: Registry) -> None:
    """The headline: a real clip concatenated with a generated
    segment whose audio is silence, so both branches agree on (video, audio)
    and `concat` has something to join."""
    g = _dyn(
        "SELECT f.video[1], f.audio[1] FROM input('av.mp4') f "
        "UNION ALL "
        "SELECT t.video[1], s.audio[1] "
        "FROM ffmpeg.testsrc(duration => 1) t, ffmpeg.anullsrc(duration => 1) s",
        _registry,
    )
    assert _filters(g) == ["testsrc", "anullsrc", "concat"]
    concat = g.nodes["n3"]
    assert concat.args == {"n": 2, "v": 1, "a": 1}
    assert concat.inputs == ["src:f:v:0", "src:f:a:0", "n1", "n2"]
    assert concat.outputs == ["video", "audio"]
    assert _outputs(g) == [("n3:0", "video", None), ("n3:1", "audio", None)]
    assert g.input_paths == ["av.mp4"]


def test_a_union_all_branch_signature_mismatch_still_fires(
    _registry: Registry,
) -> None:
    """Without the silent-audio branch the shapes disagree, and the source
    branch is just another branch as far as the check is concerned."""
    err = _reject_dyn(
        "SELECT f.video[1], f.audio[1] FROM input('av.mp4') f "
        "UNION ALL SELECT t.video[1] FROM ffmpeg.testsrc(duration => 1) t",
        _registry,
    )
    assert err.code is ErrorCode.CONCAT_MISMATCH
    assert "branch 1 selects (video, audio), branch 2 selects (video)" in err.message


def test_a_source_in_select_position_points_at_from(_registry: Registry) -> None:
    """A source is not a column function: `SELECT ffmpeg.testsrc(...)` is
    still UNKNOWN_FUNCTION (the registry's `get` never answers for a source),
    but the hint now says where it belongs."""
    err = _reject_dyn(
        "SELECT ffmpeg.testsrc(duration => 2) FROM input('x.mp4') a", _registry
    )
    assert err.code is ErrorCode.UNKNOWN_FUNCTION
    assert err.hint is not None
    assert "is a generated source, not a function" in err.hint
    assert "FROM ffmpeg.testsrc(duration => 2) s" in err.hint


def test_an_unknown_alias_hint_lists_source_aliases(_registry: Registry) -> None:
    err = _reject_dyn(
        "SELECT nope.video[1] FROM ffmpeg.testsrc(duration => 2) t", _registry
    )
    assert err.code is ErrorCode.UNKNOWN_ALIAS
    assert err.hint == "known names: t"


# ---------------------------------------------------------------------------
# the timeline `enable` named argument
# ---------------------------------------------------------------------------
#
# Offline again, and the fixture `-filters` block is what makes it possible:
# `gblur`/`deband`/`unsharp` carry the T flag, `crop`/`scale`/`xfade`/`aecho`
# do not, and `testsrc`/`anullsrc` are sources (no flag at all). `enable` is in
# NO filter's option table -- it is framework-level -- so every one of these
# goes through the special case rather than the registry lookup.


def test_enable_is_accepted_on_a_timeline_capable_tier_two_filter(
    _registry: Registry,
) -> None:
    g = _dyn(
        "SELECT gblur(a.video[1], sigma => 5, enable => 'between(t,0.5,1.5)') "
        "FROM input('x.mp4') a",
        _registry,
    )
    assert g.nodes["n1"].args == {"sigma": 5, "enable": "between(t,0.5,1.5)"}


def test_enable_is_an_ordinary_node_arg_in_written_order(_registry: Registry) -> None:
    """It renders like any other option -- nothing downstream knows it is
    special (emit sees a plain `enable=...` in the node's args)."""
    g = _dyn(
        "SELECT gblur(a.video[1], enable => 'gt(t,1)', sigma => 2) FROM input('x.mp4') a",
        _registry,
    )
    assert list(g.nodes["n1"].args.items()) == [("enable", "gt(t,1)"), ("sigma", 2)]


def test_enable_works_through_the_namespace_spelling(_registry: Registry) -> None:
    g = _dyn(
        "SELECT ffmpeg.gblur(a.video[1], enable => 'lt(t,1)') FROM input('x.mp4') a",
        _registry,
    )
    assert g.nodes["n1"].args == {"enable": "lt(t,1)"}


def test_enable_is_rejected_on_a_filter_without_timeline_support(
    _registry: Registry,
) -> None:
    """`scale` is `..C` in the fixture AND in real ffmpeg 7.1: no T."""
    err = _reject_dyn(
        "SELECT ffmpeg.scale(a.video[1], enable => 'gt(t,1)') FROM input('x.mp4') a",
        _registry,
    )
    assert err.code is ErrorCode.UNKNOWN_FILTER_OPTION
    assert "filter 'scale' has no option 'enable'" in err.message
    assert "timeline" in err.message
    assert err.hint is not None and "T column" in err.hint


def test_enable_reaches_through_a_stdlib_call_to_its_filter(_registry: Registry) -> None:
    """Tier-1 named extra: `blur` expands to `gblur`, which has the T flag."""
    g = _dyn(
        "SELECT gblur(a.video[1], 5, enable => 'between(t,0.5,1.5)') FROM input('x.mp4') a",
        _registry,
    )
    node = g.nodes["n1"]
    assert node.filter == "gblur"
    assert node.args == {"sigma": 5, "enable": "between(t,0.5,1.5)"}


def test_enable_on_a_stdlib_call_follows_the_underlying_filters_flag(
    _registry: Registry,
) -> None:
    """The flag consulted is the TARGET filter's, not the function's name:
    `crop` and `scale` are both non-T, so neither stdlib call takes it."""
    for query, filter_name in (
        ("SELECT crop(a.video[1], 0, 0, 10, 10, enable => 'gt(t,1)')", "crop"),
        ("SELECT scale(a.video[1], 640, 360, enable => 'gt(t,1)')", "scale"),
    ):
        err = _reject_dyn(f"{query} FROM input('x.mp4') a", _registry)
        assert err.code is ErrorCode.UNKNOWN_FILTER_OPTION, filter_name
        assert f"filter '{filter_name}' has no option 'enable'" in err.message


def test_enable_on_a_macro_is_still_the_macro_rejection(_registry: Registry) -> None:
    """`enable` is a named argument like any other, and a ffrwd macro's
    arguments are positional only -- `enable` is not a way around that,
    same as any other named extra."""
    err = _reject_dyn(
        "SELECT ffrwd.blur_regions(a.video[1], 0, 0, 10, 10, 5, enable => 'gt(t,1)') "
        "FROM input('x.mp4') a",
        _registry,
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "positional" in err.message


def test_enable_on_a_generated_source_is_rejected(_registry: Registry) -> None:
    """A source MAKES frames; there is no upstream frame to switch off, and
    SourceFilter carries no timeline field at all, so this can never pass."""
    err = _reject_dyn(
        "SELECT t.video[1] FROM ffmpeg.testsrc(duration => 2, enable => 'gt(t,1)') t",
        _registry,
    )
    assert err.code is ErrorCode.UNKNOWN_FILTER_OPTION
    assert "filter 'testsrc' has no option 'enable'" in err.message


def test_enable_needs_a_string_expression(_registry: Registry) -> None:
    for value, got in (("1", "1"), ("true", "true"), ("-2.5", "-2.5")):
        err = _reject_dyn(
            f"SELECT gblur(a.video[1], enable => {value}) FROM input('x.mp4') a",
            _registry,
        )
        assert err.code is ErrorCode.FILTER_OPTION_TYPE, value
        assert "expects an ffmpeg timeline expression" in err.message
        assert got in err.message
        assert err.hint is not None and "between(t,2,5)" in err.hint


def test_enable_expression_content_is_not_validated(_registry: Registry) -> None:
    """A non-goal, stated as a test: the variable vocabulary is
    per-filter and not introspectable, so nonsense compiles and it is ffmpeg
    that rejects it at run time."""
    g = _dyn(
        "SELECT gblur(a.video[1], enable => 'wat(zzz,1)') FROM input('x.mp4') a",
        _registry,
    )
    assert g.nodes["n1"].args == {"enable": "wat(zzz,1)"}


def test_enable_is_case_sensitive_like_every_option_name(_registry: Registry) -> None:
    """`ENABLE` is not `enable`; it falls through to the ordinary lookup and
    gblur has no such option."""
    err = _reject_dyn(
        "SELECT gblur(a.video[1], ENABLE => 'gt(t,1)') FROM input('x.mp4') a", _registry
    )
    assert err.code is ErrorCode.UNKNOWN_FILTER_OPTION
    assert "'ENABLE'" in err.message


def test_enable_without_a_registry_never_gets_that_far() -> None:
    """No ffmpeg means no filter named `gblur` at all, so the call is unknown
    before anything asks about timeline support."""
    err = _reject_dyn(
        "SELECT gblur(a.video[1], 5, enable => 'gt(t,1)') FROM input('x.mp4') a", None
    )
    assert err.code is ErrorCode.UNKNOWN_FUNCTION


def test_enable_broadcasts_onto_every_element(_registry: Registry) -> None:
    """A named extra is validated once and merged into each element's node
; `enable` is no different."""
    g = _dyn(
        "SELECT gblur(a.video, enable => 'gt(t,1)') FROM input('x.mp4') a",
        _registry,
        {"a": _probe_result(videos=2)},
    )
    assert [node.args for node in g.nodes.values()] == [
        {"enable": "gt(t,1)"},
        {"enable": "gt(t,1)"},
    ]


# The "expr parameter kind" (a stdlib FUNCTIONS-table concept: which stdlib
# slots took a quoted expression vs. a bare number) is dead -- there is no
# stdlib table anymore, only the registry's
# own option types (num/str/bool/enum), covered by the dynamic-call tests
# above and by the registry faithfulness tests in tests/exec/test_exec.py.
# The section that pinned it (test_an_expr_slot_*, test_expr_slots_cover_*,
# test_a_num_slot_still_refuses_a_string, test_an_expr_argument_broadcasts_*,
# test_expr_arguments_compile_under_portable) is removed rather than
# respelled: there is no surviving concept to respell it onto.


# ---------------------------------------------------------------------------
# the same shapes against the REAL installed ffmpeg
# ---------------------------------------------------------------------------
#
# The offline tests above pin the semantics against captured fixtures; these
# pin the CONTRACT with the actual binary -- that the option tables ffrwd
# introspects really do describe filters ffmpeg then accepts, and that a
# compiled tier-2 command runs. Expectations here are therefore about what
# ffmpeg does, never about a specific option list.


def _run_compiled(query: str, out_path: Path) -> None:
    """Compile, emit and RUN a query; the exit code is the assertion."""
    args = build_ffmpeg_args(emit(compile_sql(query)), str(out_path))
    args.insert(1, "-y")
    result = subprocess.run(args, capture_output=True, text=True, timeout=60.0)
    assert result.returncode == 0, result.stderr
    assert out_path.exists()


@pytest.mark.exec
def test_a_pure_tier_two_filter_compiles_and_runs(
    _av_fixture: str, tmp_path: Path
) -> None:
    """curves is in no stdlib table: name, pad signature, `preset` and its
    constants all come from the installed ffmpeg -- and the command runs."""
    query = (
        f"SELECT curves(a.video[1], preset => 'lighter'), a.audio[1] "
        f"FROM input('{_av_fixture}') a"
    )
    g = compile_sql(query)
    assert g.nodes["n1"].filter == "curves"
    assert g.nodes["n1"].args == {"preset": "lighter"}
    assert "curves=preset=lighter" in emit(g).filter_complex
    _run_compiled(query, tmp_path / "curves.mp4")


@pytest.mark.exec
def test_two_named_options_on_a_real_filter_run(_av_fixture: str, tmp_path: Path) -> None:
    query = (
        f"SELECT unsharp(a.video[1], luma_msize_x => 7, luma_amount => 1.5) "
        f"FROM input('{_av_fixture}') a"
    )
    assert compile_sql(query).nodes["n1"].args == {"luma_msize_x": 7, "luma_amount": 1.5}
    _run_compiled(query, tmp_path / "unsharp.mp4")


@pytest.mark.exec
def test_a_real_boolean_option_renders_as_ffmpeg_wants_it(
    _av_fixture: str, tmp_path: Path
) -> None:
    """deband's `blur` is a real <boolean> AVOption; SQL writes true/false and
    emit renders 1/0, which is what ffmpeg parses."""
    query = f"SELECT deband(a.video[1], blur => false) FROM input('{_av_fixture}') a"
    assert "deband=blur=0" in emit(compile_sql(query)).filter_complex
    _run_compiled(query, tmp_path / "deband.mp4")


@pytest.mark.exec
def test_the_real_gblur_range_comes_from_ffmpeg(_av_fixture: str) -> None:
    err = _reject(f"SELECT gblur(a.video[1], sigma => 5000) FROM input('{_av_fixture}') a")
    assert err.code is ErrorCode.FILTER_OPTION_TYPE
    assert "0 to 1024" in err.message


@pytest.mark.exec
def test_the_real_xfade_transition_constants_are_enforced(
    _av2_fixture: str, _av3_fixture: str, tmp_path: Path
) -> None:
    """xfade as a DYNAMIC call (the stdlib name is crossfade): its transition
    is an ffmpeg enum, so a constant name is checked against the real list."""
    both = f"FROM input('{_av2_fixture}') a, input('{_av3_fixture}') b"
    err = _reject(f"SELECT xfade(a.video[1], b.video[1], transition => 'sideways') {both}")
    assert err.code is ErrorCode.FILTER_OPTION_TYPE
    assert "wipeleft" in err.message

    query = (
        f"SELECT xfade(a.video[1], b.video[1], transition => 'wipeleft', "
        f"duration => 1, offset => 1) {both}"
    )
    assert compile_sql(query).nodes["n1"].args == {
        "transition": "wipeleft",
        "duration": 1,
        "offset": 1,
    }
    _run_compiled(query, tmp_path / "xfade.mp4")


@pytest.mark.exec
def test_a_real_unknown_option_lists_the_real_ones(_av_fixture: str) -> None:
    err = _reject(f"SELECT gblur(a.video[1], sigmma => 5) FROM input('{_av_fixture}') a")
    assert err.code is ErrorCode.UNKNOWN_FILTER_OPTION
    assert err.hint is not None and "sigma" in err.hint


@pytest.mark.exec
def test_a_tier_one_named_extra_runs(_av_fixture: str, tmp_path: Path) -> None:
    """blur() reaches through to gblur's full option set: `planes` is not in any
    ffrwd table, it was read out of this ffmpeg."""
    query = f"SELECT gblur(a.video[1], 5, planes => 1) FROM input('{_av_fixture}') a"
    assert compile_sql(query).nodes["n1"].args == {"sigma": 5, "planes": 1}
    _run_compiled(query, tmp_path / "blur-planes.mp4")


@pytest.mark.exec
def test_a_tier_two_audio_filter_broadcasts_over_real_tracks(
    _av2_fixture: str, tmp_path: Path
) -> None:
    """The broadcast machinery is type-driven, so a dynamic audio filter
    expands over every language track exactly as reverb() does -- tags and
    all -- and the result runs."""
    query = (
        f"SELECT a.video[1], aecho(a.audio, in_gain => 0.5) "
        f"FROM input('{_av2_fixture}') a"
    )
    g = compile_sql(query)
    assert _filters(g) == ["aecho", "aecho"]
    assert g.nodes["n1"].inputs == ["src:a:a:0"]
    assert g.nodes["n2"].inputs == ["src:a:a:1"]
    assert [o.metadata for o in g.outputs] == [
        {},
        {"language": "eng"},
        {"language": "fra"},
    ]
    _run_compiled(query, tmp_path / "aecho.mkv")


@pytest.mark.exec
def test_did_you_mean_reaches_into_the_real_filter_set(_av_fixture: str) -> None:
    err = _reject(f"SELECT gblu(a.video[1]) FROM input('{_av_fixture}') a")
    assert err.code is ErrorCode.UNKNOWN_FUNCTION
    assert err.hint is not None and "gblur()" in err.hint


# ---------------------------------------------------------------------------
# POSITIONAL-BINDING FIDELITY, measured against the REAL ffmpeg
# ---------------------------------------------------------------------------
#
# The claim positional options rest on: the registry's deduped, insertion-
# ordered option list IS ffmpeg's own positional binding order. That is not
# something ffrwd can decide -- it is a property of the binary -- so it is
# measured here rather than asserted from a table.
#
# Two independent proofs per filter:
#
#  (a) BEHAVIOURAL. Compile a positional ffrwd call, and separately hand
#      ffmpeg the equivalent all-NAMED filtergraph. Decode both to `-f md5`.
#      Identical digests mean the positional slots landed on the same options.
#  (b) STRUCTURAL. Feed a value that option cannot take and check ffmpeg's own
#      diagnostic names the option the registry put in that slot.
#
# `scale` is the one to watch: the registry keeps the LONG alias (`width`,
# `height`) while ffmpeg's positional walk consumes the SHORT one (`w`, `h`).
# They are the same AVOption -- same struct offset -- which is exactly what
# (a) proves and what no amount of reading the option list could.

_FIDELITY_CASES: list[tuple[str, str, str, list[str]]] = [
    # (filter, "v"|"a", pad spelling in SQL, positional values)
    ("scale", "v", "a.video[1]", ["160", "120", "'lanczos'"]),
    ("gblur", "v", "a.video[1]", ["5", "2", "1"]),
    ("crop", "v", "a.video[1]", ["100", "50", "10", "20"]),
    ("volume", "a", "a.audio[1]", ["0.5"]),
    ("eq", "v", "a.video[1]", ["1.2", "0.1", "1.5"]),
    ("hqdn3d", "v", "a.video[1]", ["4", "3", "6"]),
]


def _md5_of(argv: list[str]) -> str:
    result = subprocess.run(argv, capture_output=True, text=True, timeout=120.0)
    assert result.returncode == 0, result.stderr
    lines = [line for line in result.stdout.splitlines() if line.startswith("MD5=")]
    assert lines, result.stdout + result.stderr
    return lines[-1]


@pytest.mark.exec
@pytest.mark.parametrize("case", _FIDELITY_CASES, ids=lambda c: str(c[0]))
def test_positional_options_bind_the_way_ffmpeg_binds_them(
    case: tuple[str, str, str, list[str]], _av_fixture: str
) -> None:
    """(a) behavioural: our positional compile == ffmpeg's named filtergraph."""
    name, kind, pad, values = case
    options = registry_module.load().options(name)
    assert options is not None
    slots = list(options)[: len(values)]

    query = f"SELECT {name}({pad}, {', '.join(values)}) FROM input('{_av_fixture}') a"
    compiled = emit(compile_sql(query)).filter_complex

    # The same options, written by NAME, straight into ffmpeg -- built from the
    # registry's slot list, which is the thing under test.
    unquoted = [v.strip("'") for v in values]
    named = f"{name}=" + ":".join(f"{o}={v}" for o, v in zip(slots, unquoted))
    stream = "0:v:0" if kind == "v" else "0:a:0"

    ours = _md5_of(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", _av_fixture,
         "-filter_complex", compiled, "-map", "[out0]", "-f", "md5", "-"]
    )
    theirs = _md5_of(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", _av_fixture,
         "-filter_complex", f"[{stream}]{named}[o]", "-map", "[o]", "-f", "md5", "-"]
    )
    assert ours == theirs, f"{compiled!r} != {named!r}"


@pytest.mark.exec
@pytest.mark.parametrize("case", _FIDELITY_CASES, ids=lambda c: str(c[0]))
def test_ffmpeg_names_the_option_the_registry_put_in_each_slot(
    case: tuple[str, str, str, list[str]], _av_fixture: str, pinned_ffmpeg: None
) -> None:
    """(b) structural: junk in slot i, and ffmpeg's diagnostic for the
    positional spelling matches its diagnostic for the named one.

    Comparing the two DIAGNOSTICS rather than grepping for a name keeps this
    honest for string/expression options, which ffmpeg accepts at set time and
    only rejects later at config time (no "Error applying option" line at all).

    Pinned-version only: other releases echo alias spellings ('w' for
    'width') in these messages, which is text drift, not a binding bug --
    the md5 fidelity test above proves the binding on every version.
    """
    name, kind, _pad, values = case
    options = registry_module.load().options(name)
    assert options is not None
    order = list(options)
    stream = "0:v:0" if kind == "v" else "0:a:0"
    junk = "@@ffrwd-not-a-value@@"

    def diagnostic(graph: str) -> str:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", _av_fixture,
             "-filter_complex", f"[{stream}]{graph}[o]", "-map", "[o]",
             "-f", "null", "-"],
            capture_output=True, text=True, timeout=120.0,
        )
        # ffmpeg echoes the whole graph text in one line, which of course
        # differs between the two spellings and carries no binding information.
        text = re.sub(r"^Failed to set value '.*' for option '.*': ", "", result.stderr, flags=re.M)
        text = re.sub(r"@ [0-9a-fA-F]+", "@ X", text)
        return re.sub(r"Parsed_\w+_\d+", "P", text).strip()

    for index in range(len(values)):
        filler = [v.strip("'") for v in values[:index]]
        positional = f"{name}=" + ":".join([*filler, junk])
        by_name = f"{name}=" + ":".join(
            [f"{order[j]}={v}" for j, v in enumerate(filler)] + [f"{order[index]}={junk}"]
        )
        assert diagnostic(positional) == diagnostic(by_name), (
            f"slot {index + 1} of {name} does not bind to {order[index]!r}"
        )


@pytest.mark.exec
def test_a_positionally_compiled_call_runs(_av_fixture: str, tmp_path: Path) -> None:
    """The whole convention, end to end: streams, positionals, then a named."""
    out = tmp_path / "positional.mp4"
    query = (
        f"SELECT gblur(crop(a.video[1], 160, 120, 10, 20), 3, 2), "
        f"volume(a.audio[1], 0.5) FROM input('{_av_fixture}') a"
    )
    _run_compiled(query, out)


@pytest.mark.exec
def test_an_n_input_call_runs(_av_fixture: str, tmp_path: Path) -> None:
    """amix is an N-input filter, and the command it builds is one ffmpeg
    accepts."""
    out = tmp_path / "amix.mp4"
    query = (
        f"SELECT amix(a.audio[1], a.audio[1]) FROM input('{_av_fixture}') a"
    )
    g = compile_sql(query)
    assert g.nodes["n1"].args["inputs"] == 2
    _run_compiled(query, out)


@pytest.mark.exec
def test_the_snapshot_agrees_with_the_installed_ffmpeg_on_option_order(
    pinned_ffmpeg: None,
) -> None:
    """The committed fixture is only a stand-in if its ORDER matches too.
    Pinned-version only: another release ordering its options differently is
    drift the registry adapts to, not a bug (`pinned_ffmpeg` skips + warns)."""
    live = registry_module.load()
    ref = load_reference(SNAPSHOT_PATH)
    for name in live.names():
        assert list(ref.options(name) or {}) == list(live.options(name) or {}), name


# ---------------------------------------------------------------------------
# the collision census, measured against the REAL filter set
# ---------------------------------------------------------------------------
#
# Which filter names Postgres parses specially is a property of sqlglot's
# grammar crossed with this ffmpeg's filter list, so it is MEASURED rather
# than reasoned about: parse `<name>(...)` for every in-scope filter, in
# several argument shapes (a collision can depend on the arity -- `overlay(a)`
# is a PARSE_ERROR while `overlay(a, b, 1, 2)` is the builtin), and collect
# every name that does not arrive as an ordinary anonymous call.
#
# The list this pins is the one docs/dynamic-filters.md publishes. It is
# allowed to grow with a new ffmpeg or a new sqlglot; what must never grow is
# the set of filters you cannot reach, which is why the second half of the
# census compiles every collided name through the namespace.

_CENSUS_ARG_FORMS = (
    "a.video[1]",
    "a.video[1], b.video[1]",
    "a.video[1], b.video[1], 1, 2",
    "a.video[1], x => 1",
)

# Measured against ffmpeg 9.0.1 (512 in-scope filters) and sqlglot 30.17.
# `join` entered when n-input filters joined the in-scope set.
_KNOWN_COLLISIONS = frozenset(
    {
        "copy",
        "join",
        "corr",
        "format",
        "median",
        "normalize",
        "null",
        "overlay",
        "pad",
        "random",
        "reverse",
        "trim",
    }
)


def _parses_as_a_plain_call(sql: str, *, namespaced: bool) -> bool:
    """Did sqlglot read this as an ordinary function call and nothing else?

    A namespaced call's ordinary shape is one level deeper --
    ``Dot(Identifier(ffmpeg), Anonymous(...))`` -- which is the whole reason
    the namespace works: the qualifier is what the parser sees first, so no
    special form ever matches.
    """
    try:
        tree = sqlglot.parse_one(sql, read="postgres")
    except Exception:
        return False
    node = tree.expressions[0]
    if not namespaced:
        return isinstance(node, sqlglot.exp.Anonymous)
    return isinstance(node, sqlglot.exp.Dot) and isinstance(
        node.args.get("expression"), sqlglot.exp.Anonymous
    )


def _census(names: list[str], *, namespaced: bool = False) -> set[str]:
    prefix = "ffmpeg." if namespaced else ""
    return {
        name
        for name in names
        for form in _CENSUS_ARG_FORMS
        if not _parses_as_a_plain_call(
            f"SELECT {prefix}{name}({form}) FROM t", namespaced=namespaced
        )
    }


@pytest.mark.exec
def test_the_collision_census_is_the_documented_set() -> None:
    """Every filter name Postgres parses specially, enumerated empirically."""
    names = registry_module.load().names()
    assert names, "no filters: this test needs a real ffmpeg"
    assert _census(names) == _KNOWN_COLLISIONS & set(names)


@pytest.mark.exec
def test_the_namespace_never_collides_with_the_grammar() -> None:
    """The other half of the census: prefixed with the namespace, EVERY filter
    name parses as an ordinary call in every argument shape."""
    names = registry_module.load().names()
    assert names, "no filters: this test needs a real ffmpeg"
    assert _census(names, namespaced=True) == set()


@pytest.mark.exec
def test_every_collided_filter_compiles_through_the_namespace(
    _av_fixture: str, _av2_fixture: str
) -> None:
    """The point of the whole feature: no in-scope filter is unreachable.

    Each collided name is called with exactly its own input pads (from the
    real pad signature), one distinct alias per pad so nothing needs a split,
    so the compile is the genuine one -- probe, registry lookup, pad type
    check and all.
    """
    registry = registry_module.load()
    collided = sorted(_KNOWN_COLLISIONS & set(registry.names()))
    assert collided, "no collisions on this ffmpeg: nothing to prove"
    files = (_av_fixture, _av2_fixture)

    for name in collided:
        dynamic = registry.get(name)
        assert dynamic is not None
        # An n-input filter reports no fixed pads; two of its own kind is a
        # legal call (the count option defaults to 2).
        kinds = list(dynamic.inputs) or [dynamic.output, dynamic.output]
        aliases = "abcd"[: len(kinds)]
        assert len(aliases) == len(kinds), name
        pads = ", ".join(
            f"{alias}.video[1]" if kind == "video" else f"{alias}.audio[1]"
            for alias, kind in zip(aliases, kinds)
        )
        sources = ", ".join(
            f"input('{files[i % len(files)]}') {alias}"
            for i, alias in enumerate(aliases)
        )
        query = f"SELECT ffmpeg.{name}({pads}) FROM {sources}"
        graph = compile_sql(query)
        # `trim` (unlike every other collided name) gets a `setpts` spliced
        # in after it: the compiler resets PTS after every trim/atrim call
        # whose result reaches a consumer, this bare one included.
        expected = [name, "setpts"] if name == "trim" else [name]
        assert [node.filter for node in graph.nodes.values()] == expected, query


@pytest.mark.exec
def test_the_real_overlay_options_are_reachable_through_the_namespace(
    _av2_fixture: str, _av3_fixture: str, tmp_path: Path
) -> None:
    """`overlay`'s named extras are unreachable in its stdlib spelling (the
    OVERLAY..PLACING grammar makes `=>` a PARSE_ERROR); namespaced, the whole
    option set is there -- and the command runs."""
    query = (
        "SELECT ffmpeg.overlay(a.video[1], b.video[1], x => 20, y => 20, "
        "eof_action => 'pass') "
        f"FROM input('{_av2_fixture}') a, input('{_av3_fixture}') b"
    )
    g = compile_sql(query)
    assert g.nodes["n1"].filter == "overlay"
    assert g.nodes["n1"].args == {"x": 20, "y": 20, "eof_action": "pass"}
    _run_compiled(query, tmp_path / "ns-overlay.mp4")

    assert _reject(
        "SELECT overlay(a.video[1], b.video[1], x => 20, y => 20) "
        f"FROM input('{_av2_fixture}') a, input('{_av3_fixture}') b"
    ).code is ErrorCode.PARSE_ERROR


@pytest.mark.exec
def test_the_real_trim_filter_runs_through_the_namespace(
    _av_fixture: str, tmp_path: Path
) -> None:
    """ffmpeg's `trim` filter, the one Postgres's TRIM grammar hides.

    The option names are ffmpeg's own as the registry reports them, aliases
    deduped to the longer spelling (`starti`, not `start`) -- read out of the
    binary, not from any table here.
    """
    query = (
        f"SELECT ffmpeg.trim(a.video[1], starti => 0.5, durationi => 1) "
        f"FROM input('{_av_fixture}') a"
    )
    assert compile_sql(query).nodes["n1"].args == {"starti": 0.5, "durationi": 1}
    _run_compiled(query, tmp_path / "ns-trim.mp4")


# ---------------------------------------------------------------------------
# an explicit trim/atrim call gets its PTS rebased -- unlike the WHERE-clause
# trim above, this one carries the source's own timestamps unless a reset is
# spliced in after it (compiler.py's insert_pts_resets, run before splits).
# ---------------------------------------------------------------------------


def test_a_trim_written_straight_to_output_gets_a_pts_reset() -> None:
    g = compile_sql(
        "SELECT ffmpeg.trim(a.video[1], starti => 1) FROM input('x.mp4') a"
    )
    assert [node.filter for node in g.nodes.values()] == ["trim", "setpts"]
    assert g.nodes["n1_pts"].args == {"expr": "PTS-STARTPTS"}
    assert g.nodes["n1_pts"].inputs == ["n1"]


def test_two_trims_concatenated_each_get_their_own_pts_reset() -> None:
    g = compile_sql(
        "SELECT ffmpeg.trim(a.video[1], starti => 0, endi => 1) FROM input('x.mp4') a "
        "UNION ALL "
        "SELECT ffmpeg.trim(b.video[1], starti => 2, endi => 3) FROM input('x.mp4') b"
    )
    filters = [node.filter for node in g.nodes.values()]
    assert filters.count("setpts") == 2
    assert filters.count("trim") == 2
    concat = next(node for node in g.nodes.values() if node.filter == "concat")
    resets = {node.id for node in g.nodes.values() if node.filter == "setpts"}
    assert set(concat.inputs) == resets


def test_a_video_and_an_audio_trim_in_one_query_get_setpts_and_asetpts() -> None:
    g = compile_sql(
        "SELECT ffmpeg.trim(a.video[1], starti => 1), ffmpeg.atrim(a.audio[1], starti => 1) "
        "FROM input('x.mp4') a"
    )
    filters_by_type = {node.filter: node.outputs[0] for node in g.nodes.values()}
    assert filters_by_type["setpts"] == "video"
    assert filters_by_type["asetpts"] == "audio"


def test_ffrwd_speed_on_a_trim_takes_over_timing_no_extra_reset() -> None:
    """`ffrwd.speed` expands to its own `setpts`; that already takes
    control of the trimmed stream's timing, so nothing extra is inserted."""
    g = compile_sql(
        "SELECT ffrwd.speed(ffmpeg.trim(a.video[1], starti => 1), 2) "
        "FROM input('x.mp4') a"
    )
    assert [node.filter for node in g.nodes.values()] == ["trim", "setpts"]
    assert g.nodes["n2"].args == {"expr": "PTS/2"}


# ---------------------------------------------------------------------------
# scripts + CREATE VIEW
# ---------------------------------------------------------------------------
#
# A view is to STATEMENTS what a CTE is to branches, and lower treats it as
# exactly that: `Resolved.ctes` is one flat, ordered binding table holding
# both, so nothing in this pass knows a view from a CTE. These tests pin that
# equivalence rather than re-testing the CTE machinery through a new syntax.

_VIEW_SCRIPT = (
    "CREATE VIEW master AS\n"
    "  SELECT scale(a.video[1], 1280, -2) AS v FROM input('film.mkv') a;\n"
    "COPY (SELECT gblur(master.v, 2) FROM master) TO 'out.mp4' WITH (crf 20);"
)

_CTE_EQUIVALENT = (
    "COPY (WITH master AS (\n"
    "  SELECT scale(a.video[1], 1280, -2) AS v FROM input('film.mkv') a\n"
    ") SELECT gblur(master.v, 2) FROM master) TO 'out.mp4' WITH (crf 20);"
)


def test_a_view_lowers_into_the_same_ir_a_cte_would() -> None:
    """The whole design claim of views-are-CTEs, as one assertion."""
    assert compile_sql(_VIEW_SCRIPT).to_dict() == compile_sql(_CTE_EQUIVALENT).to_dict()


def test_a_view_script_keeps_its_sink() -> None:
    g = compile_sql(_VIEW_SCRIPT)
    assert len(g.sinks) == 1
    assert g.sinks[0].path == "out.mp4"
    assert g.sinks[0].options == {"crf": 20}


def test_a_view_script_compiles_to_one_ffmpeg_command() -> None:
    args = build_ffmpeg_args(emit(compile_sql(_VIEW_SCRIPT)), None)
    assert args.count("-i") == 1
    assert "scale=width=1280:height=-2" in " ".join(args)
    assert args[-1] == "out.mp4"


def test_a_view_is_split_across_its_consumers() -> None:
    """Two reads of one view pad go through a split, exactly like a CTE's."""
    g = compile_sql(
        "CREATE VIEW m AS SELECT a.video[1] AS v FROM input('x.mp4') a;\n"
        "COPY (SELECT gblur(m.v, 1), gblur(m.v, 2) FROM m) TO 'out.mp4';"
    )
    assert any(node.filter == "split" for node in g.nodes.values())


def test_a_view_body_with_its_own_with_lowers() -> None:
    g = compile_sql(
        "CREATE VIEW v AS WITH c AS (SELECT a.video[1] AS f FROM input('x.mp4') a) "
        "SELECT scale(c.f, 0.5) AS v FROM c;\n"
        "COPY (SELECT v.v FROM v) TO 'out.mp4';"
    )
    assert _filters(g) == ["scale"]


def test_a_view_referencing_a_view_lowers() -> None:
    g = compile_sql(
        "CREATE VIEW one AS SELECT a.video[1] AS v FROM input('x.mp4') a;\n"
        "CREATE VIEW two AS SELECT scale(one.v, 0.5) AS v FROM one;\n"
        "COPY (SELECT gblur(two.v, 3) FROM two) TO 'out.mp4';"
    )
    assert _filters(g) == ["scale", "gblur"]


def test_a_view_column_error_still_names_the_view() -> None:
    err = _reject(
        "CREATE VIEW m AS SELECT a.video[1] AS v FROM input('x.mp4') a;\n"
        "COPY (SELECT m.nope FROM m) TO 'out.mp4';"
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert err.line == 2


# --- multiple sinks ---

_TWO_SINKS = (
    "CREATE VIEW m AS SELECT a.video[1] AS v FROM input('film.mkv') a;\n"
    "COPY (SELECT scale(m.v, 1280, -2) FROM m) TO '720.mp4';\n"
    "COPY (SELECT scale(m.v, 640, -2) FROM m) TO '360.mp4';"
)


def test_each_copy_becomes_its_own_sink_unit() -> None:
    g = compile_sql(_TWO_SINKS)
    assert [unit.path for unit in g.sinks] == ["720.mp4", "360.mp4"]
    assert [len(unit.outputs) for unit in g.sinks] == [1, 1]


def test_the_parser_and_the_ir_agree_on_the_sink_list() -> None:
    res = resolve(parse(_TWO_SINKS))
    assert [sink.path for sink in res.sinks] == ["720.mp4", "360.mp4"]
    assert [unit.path for unit in compile_sql(_TWO_SINKS).sinks] == [
        "720.mp4",
        "360.mp4",
    ]


_LADDER_SCRIPT = (
    "CREATE VIEW master AS\n"
    "  SELECT gblur(a.video[1], 2) AS v, volume(a.audio[1], 0.9) AS a\n"
    "  FROM input('film.mkv') a;\n"
    "COPY (SELECT scale(m.v, 1280, -2), m.a FROM master m) TO '720.mp4';\n"
    "COPY (SELECT scale(m.v, 640, -2), m.a FROM master m) TO '360.mp4';\n"
    "COPY (SELECT m.a FROM master m) TO 'audio.m4a';"
)


def test_the_shared_view_is_lowered_once_and_split_across_the_sinks() -> None:
    """The ladder's load-bearing property: the master view lowers ONCE.

    Three COPYs read `master`. Its two filters (`gblur`, `volume`) must appear
    exactly once each; what fans them out is the split pass, one `split` for
    the two video readers and one `asplit` for the three audio ones.
    """
    g = compile_sql(_LADDER_SCRIPT)
    filters = [node.filter for node in g.nodes.values()]
    assert filters.count("gblur") == 1
    assert filters.count("volume") == 1
    assert filters.count("scale") == 2
    assert filters.count("split") == 1
    assert filters.count("asplit") == 1
    assert g.input_paths == ["film.mkv"]  # one -i: decode once
    assert [unit.path for unit in g.sinks] == ["720.mp4", "360.mp4", "audio.m4a"]
    assert [len(unit.outputs) for unit in g.sinks] == [2, 2, 1]


def test_the_shared_views_split_pads_are_handed_out_in_sink_order() -> None:
    g = compile_sql(_LADDER_SCRIPT)
    asplit = next(node for node in g.nodes.values() if node.filter == "asplit")
    assert asplit.args == {"n": 3}
    assert [unit.outputs[-1].ref for unit in g.sinks] == [
        f"{asplit.id}:0",
        f"{asplit.id}:1",
        f"{asplit.id}:2",
    ]


@pytest.mark.exec
def test_a_view_based_query_runs(_av_fixture: str, tmp_path: Path) -> None:
    """End to end through real ffmpeg: a view + one COPY is a usable script."""
    out = tmp_path / "view.mp4"
    query = (
        "CREATE VIEW half AS\n"
        f"  SELECT scale(a.video[1], 'iw/2', -2) AS v, a.audio[1] AS a "
        f"FROM input('{_av_fixture}') a;\n"
        f"COPY (SELECT half.v, half.a FROM half) TO '{out.as_posix()}' WITH (crf 30);"
    )
    args = build_ffmpeg_args(emit(compile_sql(query)), None)
    args.insert(1, "-y")
    result = subprocess.run(args, capture_output=True, text=True, timeout=60.0)
    assert result.returncode == 0, result.stderr
    assert out.exists()


# ---------------------------------------------------------------------------
# track rows: unnest, row columns, WHERE, ORDER BY
# ---------------------------------------------------------------------------
#
# Synthetic probes throughout: what a row table holds is decided entirely by
# the StreamMeta it was built from, so a hand-built ProbeResult exercises the
# whole row model -- including the NULL columns an unprobed field produces --
# without any file on disk. The end-to-end shapes are the cookbook's business
# (recipes 23/24, exec tier).


def _track(
    stream_type: StreamType,
    index: int,
    *,
    language: str | None = None,
    title: str | None = None,
    tags: dict[str, str] | None = None,
    **fields: object,
) -> StreamMeta:
    """One synthetic probed stream, every unset field NULL except `codec`.

    Keyword `fields` name StreamMeta attributes directly (``channels=2``,
    ``codec='aac'``), so a row test names only the columns it is about and
    everything else comes back as the NULL an unprobed field yields. `codec`
    alone gets a realistic per-type default: a probed stream with NO codec is
    the ffmpeg-cannot-carry-it case, rejected for media queries, and a test
    wanting that case says ``codec=None`` explicitly.
    """
    metadata: dict[str, str] = dict(tags or {})
    if language is not None:
        metadata["language"] = language
    if title is not None:
        metadata["title"] = title
    defaults: dict[str, object] = {
        "width": None,
        "height": None,
        "fps": None,
        "sample_rate": None,
        "codec": {"video": "h264", "audio": "aac", "subtitle": "subrip", "data": "bin_data"}[
            stream_type
        ],
        "channels": None,
        "channel_layout": None,
        "bitrate": None,
        "duration": None,
        "color_transfer": None,
    }
    defaults.update(fields)
    return StreamMeta(  # type: ignore[arg-type]
        type=stream_type, index=index, metadata=metadata, **defaults
    )


_ROW_TRACKS = [
    _track("video", 0),
    _track("audio", 0, language="eng", channels=2, channel_layout="stereo", codec="aac"),
    _track("audio", 1, language="fra", channels=6, channel_layout="5.1", codec="ac3"),
    _track("audio", 2, channels=2, codec="aac"),  # no language tag at all
]


def _row_probes(*streams: StreamMeta) -> dict[str, ProbeResult | None]:
    return {"f": ProbeResult(streams=list(streams) if streams else list(_ROW_TRACKS))}


def _row_query(where: str = "", order: str = "", column: str = "audio") -> str:
    """The canonical row query: every surviving row gathered into one file.

    ``array_agg`` is what a several-row query writing one destination has to
    say, so it is part of the shape under test here rather than an extra --
    the aggregate gathers whatever rows the WHERE and the ORDER BY left, in
    their order, which is exactly what these tests are about.
    """
    return (
        f"SELECT array_agg(t) FROM input('f.mkv') f, unnest(f.{column}) t"
        + (f" WHERE {where}" if where else "")
        + (f" ORDER BY {order}" if order else "")
    )


# ---------------------------------------------------------------------------
# renditions: an ABR ladder's own row table, no unnest to ask for it
# ---------------------------------------------------------------------------
#
# `_rendition_probe` is one ladder: two muxed AV rungs (720p/1080p, distinct
# bandwidths) plus a third, audio-only rendition -- the shape that proves an
# audio-only row contributes nothing to `.video` while still counting for
# `.audio`. `RenditionMeta.streams` holds the SAME StreamMeta objects
# `ProbeResult.streams` does, exactly as probe.py's own `by_global_index`
# wires them, so a rendition's stream and the plain container's agree on
# `index` -- the fact `_bind_renditions` relies on to reuse `_source_stream`
# unchanged.


def _rendition_probe() -> ProbeResult:
    lo_v = _track("video", 0, width=1280, height=720)
    lo_a = _track("audio", 0, channels=2, codec="aac")
    hi_v = _track("video", 1, width=1920, height=1080)
    hi_a = _track("audio", 1, channels=2, codec="aac")
    solo_a = _track("audio", 2, channels=2, codec="aac", language="en")
    renditions = [
        RenditionMeta(
            streams=[lo_v, lo_a],
            bandwidth=800_000,
            width=1280,
            height=720,
            codecs="avc1.640020,mp4a.40.2",
            name="Low",
            language=None,
            program_id=1,
        ),
        RenditionMeta(
            streams=[hi_v, hi_a],
            bandwidth=3_000_000,
            width=1920,
            height=1080,
            codecs="avc1.64002a,mp4a.40.2",
            name="High",
            language=None,
            program_id=2,
        ),
        RenditionMeta(
            streams=[solo_a],
            bandwidth=128_000,
            width=None,
            height=None,
            codecs="mp4a.40.2",
            name="Audio",
            language="en",
            program_id=3,
        ),
    ]
    return ProbeResult(streams=[lo_v, lo_a, hi_v, hi_a, solo_a], renditions=renditions)


def _rendition_probes() -> dict[str, ProbeResult | None]:
    return {"r": _rendition_probe()}


def test_a_manifests_renditions_bind_as_rows_with_no_unnest() -> None:
    """``FROM input(<manifest>) r`` alone is a track-row table: one row per
    RenditionMeta, in manifest order, reached with no ``unnest`` at all.

    3 rows, but only 2 carry video and all 3 carry audio, so the ladder's
    array columns are row-FILTERED by kind, not row-COUNTED: an audio-only
    rendition is simply absent from ``.video`` while still counting for
    ``.audio``. ``array_agg`` gathers the whole row set into one file (an
    n-input filter's own job here, not `_bind_renditions`'s), which is what
    lets a 3-row ladder compile at all without a WHERE to narrow it.
    """
    g = _lower(
        "SELECT vstack(VARIADIC array_agg(r.video)) FROM input('x.m3u8') r",
        _rendition_probes(),
    )
    (node,) = g.nodes.values()
    assert node.filter == "vstack"
    assert node.inputs == ["src:r:v:0", "src:r:v:1"]  # the audio-only row: absent


def test_an_audio_only_rendition_still_counts_for_audio() -> None:
    """All 3 rows carry audio, the audio-only one included, in row order --
    its own track lands last, after the two muxed renditions'."""
    g = _lower(
        "SELECT amix(VARIADIC array_agg(r.audio)) FROM input('x.m3u8') r",
        _rendition_probes(),
    )
    (node,) = g.nodes.values()
    assert node.filter == "amix"
    assert node.inputs == ["src:r:a:0", "src:r:a:1", "src:r:a:2"]


def test_a_video_subscript_past_the_video_carrying_rows_is_rejected() -> None:
    """Only 2 rows carry video, so ``r.video[3]`` (the third ladder rung, the
    audio-only one) is out of bounds for `.video` specifically -- the bounds
    check fires before the row-count-per-file one, so no ``array_agg`` is
    needed to reach it."""
    err = _reject_lower("SELECT r.video[3] FROM input('x.m3u8') r", _rendition_probes())
    assert err.code is ErrorCode.STREAM_NOT_FOUND
    assert "r.video[3]" in err.message
    assert "2 rows carry a video track" in err.message


def test_rendition_stream_indexes_reach_the_map_flags() -> None:
    """One ladder rung, its video and audio at DELIBERATELY non-contiguous
    per-type indexes (2 other audio tracks precede this rendition's own) --
    they reach ``-map`` verbatim, proof a row's streams are the SAME
    `-map`-able refs a plain probe's `f.video[1]` would produce, not a copy
    with new indexes of their own."""
    video = _track("video", 0)
    other_a0 = _track("audio", 0)
    other_a1 = _track("audio", 1)
    own_audio = _track("audio", 2)
    probe = ProbeResult(
        streams=[video, other_a0, other_a1, own_audio],
        renditions=[
            RenditionMeta(
                streams=[video, own_audio],
                bandwidth=800_000,
                width=320,
                height=240,
                codecs=None,
                name=None,
                language=None,
                program_id=1,
            )
        ],
    )
    g = _lower(
        "SELECT r.video[1], r.audio[1] FROM input('x.m3u8') r", {"r": probe}
    )
    assert _outputs(g) == [
        ("src:r:v:0", "video", None),
        ("src:r:a:2", "audio", None),
    ]
    args = build_ffmpeg_args(emit(g), "out.mp4")
    maps = [args[i + 1] for i, arg in enumerate(args) if arg == "-map"]
    assert maps == ["0:v:0", "0:a:2"]


def test_an_input_with_no_renditions_is_untouched() -> None:
    """The whole point of the early return in `_bind_renditions`: a plain
    file (empty `renditions`) keeps meaning exactly what it always has --
    `r` stays a container alias, never a row table."""
    plain = ProbeResult(streams=[_track("video", 0), _track("audio", 0)])
    g = _lower("SELECT r.video[1], r.audio[1] FROM input('x.mp4') r", {"r": plain})
    assert _outputs(g) == [
        ("src:r:v:0", "video", None),
        ("src:r:a:0", "audio", None),
    ]
    args = build_ffmpeg_args(emit(g), "out.mp4")
    assert [args[i + 1] for i, arg in enumerate(args) if arg == "-map"] == [
        "0:v:0",
        "0:a:0",
    ]


def test_a_where_between_seek_still_works_on_a_renditionless_input() -> None:
    """A second, differently-shaped cookbook query against an empty-rendition
    probe: the WHERE trim window is untouched by `_bind_renditions`, which
    never runs for it."""
    plain = ProbeResult(streams=[_track("video", 0), _track("audio", 0)])
    g = _lower(
        "SELECT r.video[1] FROM input('x.mp4') r WHERE r.t BETWEEN 1 AND 2", {"r": plain}
    )
    assert g.to_dict()["nodes"] == []
    assert g.input_trims == {"r": (1, 2)}
    assert _outputs(g) == [("src:r:v:0", "video", None)]


def test_a_rendition_column_on_a_renditionless_input_is_refused() -> None:
    """``r.bandwidth`` (etc.) parses over ANY input alias -- resolve cannot
    know whether it will turn out to be a ladder -- so a plain file's probe
    is what actually refuses it, at bind time in lower."""
    plain = ProbeResult(streams=[_track("video", 0), _track("audio", 0)])
    err = _reject_lower("SELECT r.bandwidth FROM input('x.mp4') r", {"r": plain})
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert err.message == (
        "'r' is a single file, not a ladder: input('x.mp4') has no renditions"
    )
    assert err.hint == (
        "rendition columns (bandwidth, width, height, codecs, name, "
        "language) read from an HLS master or DASH manifest"
    )


def test_order_by_on_a_renditionless_input_keeps_the_parsers_old_refusal() -> None:
    """The parser admits ORDER BY over any `input(...)` alias now, on spec --
    only the probe can confirm there is no row table to sort. Lower raises
    the exact rejection the parser used to raise here, unchanged for a plain
    file's user experience."""
    plain = ProbeResult(streams=[_track("video", 0), _track("audio", 0)])
    err = _reject_lower(
        "SELECT r.video[1] FROM input('x.mp4') r ORDER BY r.height", {"r": plain}
    )
    assert err.code is ErrorCode.NO_STREAMING_EQUIVALENT
    assert err.message == "ORDER BY has no streaming equivalent"
    assert err.hint == "remove the ORDER BY clause"


def test_limit_and_offset_on_a_renditionless_input_keep_the_parsers_old_refusal() -> None:
    plain = ProbeResult(streams=[_track("video", 0), _track("audio", 0)])
    limit_err = _reject_lower("SELECT r.video[1] FROM input('x.mp4') r LIMIT 1", {"r": plain})
    assert limit_err.code is ErrorCode.NO_STREAMING_EQUIVALENT
    assert limit_err.message == "LIMIT has no streaming equivalent"
    offset_err = _reject_lower(
        "SELECT r.video[1] FROM input('x.mp4') r OFFSET 1", {"r": plain}
    )
    assert offset_err.code is ErrorCode.NO_STREAMING_EQUIVALENT
    assert offset_err.message == "OFFSET has no streaming equivalent"
    row_table_hint = (
        "it is legal only over a compile-time row table -- a branch whose FROM "
        "has unnest(...), generate_series(...), or a CTE or view name -- where "
        "it narrows the resolved rows, exactly like ORDER BY"
    )
    assert limit_err.hint == row_table_hint
    assert offset_err.hint == row_table_hint


def test_the_multi_row_refusal_picks_up_the_rendition_hint() -> None:
    """A ladder reaching a plain (non-manifest) one-row destination hits the
    same ROW_COUNT_MISMATCH every other multi-row relation does, but the fix
    is narrowing to one rendition, not restructuring the query into rows --
    so the hint says that instead of the generic array_agg/TO advice."""
    err = _reject_lower(
        "COPY (SELECT r.video[1] FROM input('x.m3u8') r) TO 'out.mp4'",
        _rendition_probes(),
    )
    assert err.code is ErrorCode.ROW_COUNT_MISMATCH
    assert err.message == "this query has 3 rows, and 'out.mp4' is one file"
    assert err.hint == (
        "pick a rendition: WHERE on height, bandwidth or name, or ORDER BY "
        "bandwidth DESC LIMIT 1"
    )


def test_the_one_row_manifest_refusal_picks_up_the_rendition_hint() -> None:
    """The manifest destination's OWN "too many streams for one variant"
    refusal, reached here by mixing a WHERE-narrowed ladder (cardinality 1)
    with an ordinary multi-track file's bare array column -- still picks up
    the rendition hint, because the offending relation is still `r`'s ladder,
    message body unchanged."""
    plain = ProbeResult(streams=[_track("audio", 0), _track("audio", 1)])
    err = _reject_lower(
        "COPY (SELECT r.video[1], f.audio FROM input('r.m3u8') r, "
        "input('f.mkv') f WHERE r.height = 1080) "
        "TO 'out/master.m3u8' WITH (format 'hls', video_codec 'libx264', "
        "audio_codec 'aac')",
        {**_rendition_probes(), "f": plain},
    )
    assert err.code is ErrorCode.ROW_COUNT_MISMATCH
    assert err.message == (
        "'audio' is 2 streams in one row, and each row of a format 'hls' "
        "destination is one variant map entry"
    )
    assert err.hint == (
        "pick a rendition: WHERE on height, bandwidth or name, or ORDER BY "
        "bandwidth DESC LIMIT 1"
    )


# -- array_agg(r.video[1]) / array_agg(r.audio[1]): every surviving row's
#    own stream, gathered into one file -- the whole-ladder-into-one-file
#    use case a manifest destination is not, unlike `test_a_ladder_reaches_
#    a_manifest_destination_as_n_variants` just below. --------------------


def test_array_agg_of_a_rendition_column_gathers_the_surviving_rows() -> None:
    """``WHERE r.height >= 720`` narrows the 3-rung ladder to the two muxed
    rungs; ``array_agg(r.video[1])`` gathers their own video streams, one
    per row, into one file -- both reach ``-map``, in row order."""
    g = _lower(
        "COPY (SELECT array_agg(r.video[1]) FROM input('ladder.m3u8') r "
        "WHERE r.height >= 720) TO 'all-rungs.mkv'",
        _rendition_probes(),
    )
    unit = g.sinks[0]
    assert [o.ref for o in unit.outputs] == ["src:r:v:0", "src:r:v:1"]
    args = build_ffmpeg_args(emit(g), None)
    assert [args[i + 1] for i, arg in enumerate(args) if arg == "-map"] == [
        "0:v:0",
        "0:v:1",
    ]


def test_array_agg_of_a_rendition_column_with_no_where_skips_the_audio_only_row() -> None:
    """No WHERE at all: the ladder's 2 video-carrying rows gather, and the
    third, audio-only rendition contributes nothing to ``.video`` -- the
    same kind-filtering `_rendition_kind_value` already does for a bare
    column, unchanged by going through `array_agg`."""
    g = _lower(
        "SELECT array_agg(r.video[1]) FROM input('ladder.m3u8') r",
        _rendition_probes(),
    )
    assert _outputs(g) == [
        ("src:r:v:0", "video", None),
        ("src:r:v:1", "video", None),
    ]


def test_array_agg_of_a_rendition_audio_column_gathers_every_row() -> None:
    """All 3 rows carry audio, the audio-only one included, in row order --
    the same set `test_an_audio_only_rendition_still_counts_for_audio`
    reaches through `VARIADIC`, reached here through the ordinary
    `array_agg(...)` whole-column form instead."""
    g = _lower(
        "SELECT array_agg(r.audio[1]) FROM input('ladder.m3u8') r",
        _rendition_probes(),
    )
    assert _outputs(g) == [
        ("src:r:a:0", "audio", None),
        ("src:r:a:1", "audio", None),
        ("src:r:a:2", "audio", None),
    ]


# -- array_agg(<expression over r.video[1] / r.audio[1]>): the same
#    per-surviving-row broadcast a track row's array_agg(volume(a, 0.5))
#    already gets (cookbook 115), reached by having the subscript read as
#    the array `array_agg(r.video[1])` above already gets, rather than
#    picking one rendition out of it. --------------------------------------


def test_array_agg_of_a_call_over_a_rendition_column_broadcasts_per_row() -> None:
    """``WHERE r.height >= 720`` narrows the 3-rung ladder to the two muxed
    rungs, same as `test_array_agg_of_a_rendition_column_gathers_the_surviving_
    rows`; wrapping each row's ``video[1]`` in ``scale(...)`` now lowers to
    ONE scale node per surviving row, in row order, instead of a single node
    over one picked rendition."""
    g = _lower(
        "COPY (SELECT array_agg(scale(r.video[1], 320, -2)) "
        "FROM input('ladder.m3u8') r WHERE r.height >= 720) TO 'all-rungs.mkv'",
        _rendition_probes(),
    )
    scales = [node for node in g.nodes.values() if node.filter == "scale"]
    assert len(scales) == 2
    assert [node.inputs for node in scales] == [["src:r:v:0"], ["src:r:v:1"]]
    assert all(node.args == {"width": 320, "height": -2} for node in scales)
    assert [o.ref for o in g.outputs] == [node.id for node in scales]


def test_a_where_narrows_the_broadcast_to_one_row() -> None:
    """``WHERE r.height = 720`` narrows the ladder to one row before the
    aggregate runs, so ``array_agg(scale(r.video[1], 320, -2))`` lowers to
    exactly one `scale` node -- the row model's own narrow-then-aggregate
    order, unchanged by wrapping the subscript in a call."""
    g = _lower(
        "COPY (SELECT array_agg(scale(r.video[1], 320, -2)) "
        "FROM input('ladder.m3u8') r WHERE r.height = 720) TO 'one.mkv'",
        _rendition_probes(),
    )
    (node,) = g.nodes.values()
    assert node.filter == "scale"
    assert node.inputs == ["src:r:v:0"]


def test_a_bare_subscript_under_variadic_still_gathers_the_array() -> None:
    """cookbook 114's shape (``concat(VARIADIC array_agg(s.video[1]))``) is
    reached here through `amix`, since only a live ffmpeg reports `concat`'s
    own options (see `test_the_motion_thumbnail_gather_matches_its_hand_
    written_form`): a bare ``r.audio[1]`` spread with `VARIADIC` still names
    every surviving row's own stream, all 3 rows here, unaffected by the new
    broadcast a call around the subscript now gets."""
    g = _lower(
        "SELECT amix(VARIADIC array_agg(r.audio[1])) FROM input('ladder.m3u8') r",
        _rendition_probes(),
    )
    (node,) = g.nodes.values()
    assert node.filter == "amix"
    assert node.inputs == ["src:r:a:0", "src:r:a:1", "src:r:a:2"]


def test_array_agg_of_a_rendition_column_reaches_a_two_track_destination() -> None:
    """The emitted command for the maintainer's use case: an HLS ladder
    written as one file holding every surviving rung as its own video
    track -- both rungs mapped, in row order, no fan-out and no manifest."""
    g = _lower(
        "COPY (SELECT array_agg(r.video[1]) FROM input('ladder.m3u8') r "
        "WHERE r.height >= 720) TO 'all-rungs.mkv'",
        _rendition_probes(),
    )
    args = build_ffmpeg_args(emit(g), None)
    assert args == [
        "ffmpeg",
        "-i", "ladder.m3u8",
        "-map", "0:v:0", "-c:0", "copy",
        "-map", "0:v:1", "-c:1", "copy",
        "all-rungs.mkv",
    ]


def test_a_ladder_reaches_a_manifest_destination_as_n_variants() -> None:
    """``COPY (SELECT r.video[1], r.audio[1] FROM input(<manifest>) r) TO
    ... WITH (format 'hls', ...)`` over a 3-rung ladder (two muxed rungs, one
    audio-only) lowers to 3 variant map entries: each row's own video[1]/
    audio[1] are that variant's cells, and the audio-only rung's video cell
    is NULL rather than absent -- the same gap shape a FULL JOIN leaves.
    """
    lo_v = _track("video", 0, width=1280, height=720, fps="30/1")
    lo_a = _track("audio", 0, channels=2)
    hi_v = _track("video", 1, width=1920, height=1080, fps="30/1")
    hi_a = _track("audio", 1, channels=2)
    solo_a = _track("audio", 2, channels=2, language="en")
    renditions = [
        RenditionMeta(
            streams=[lo_v, lo_a], bandwidth=800_000, width=1280, height=720,
            codecs="avc1.640020,mp4a.40.2", name="Low", language=None, program_id=1,
        ),
        RenditionMeta(
            streams=[hi_v, hi_a], bandwidth=3_000_000, width=1920, height=1080,
            codecs="avc1.64002a,mp4a.40.2", name="High", language=None, program_id=2,
        ),
        RenditionMeta(
            streams=[solo_a], bandwidth=128_000, width=None, height=None,
            codecs="mp4a.40.2", name="Audio", language="en", program_id=3,
        ),
    ]
    probe = ProbeResult(streams=[lo_v, lo_a, hi_v, hi_a, solo_a], renditions=renditions)
    g = _lower(
        "COPY (SELECT r.video[1], r.audio[1] FROM input('ladder.m3u8') r) "
        "TO 'out/master.m3u8' WITH (format 'hls', video_codec 'libx264', "
        "audio_codec 'aac')",
        {"r": probe},
    )
    unit = g.sinks[0]
    assert [o.type for o in unit.outputs] == ["video", "video", "audio", "audio", "audio"]
    assert [o.ref for o in unit.outputs] == [
        "src:r:v:0", "src:r:v:1", "src:r:a:0", "src:r:a:1", "src:r:a:2",
    ]
    var_stream_map = unit.options["var_stream_map"]
    assert isinstance(var_stream_map, str)
    variants = var_stream_map.split(" ")
    assert len(variants) == 3
    assert variants[0].startswith("v:0,a:0,")
    assert variants[1].startswith("v:1,a:1,")
    assert variants[2].startswith("a:2,")
    assert not variants[2].startswith("v:")  # the audio-only rung: NULL video
    args = build_ffmpeg_args(emit(g), None)
    maps = [args[i + 1] for i, arg in enumerate(args) if arg == "-map"]
    assert maps == ["0:v:0", "0:v:1", "0:a:0", "0:a:1", "0:a:2"]


# -- a computed TO reading rendition columns off TWO input aliases: recipe
#    110's self-join, one video-carrying ladder crossed with one audio-only
#    ladder, each alias probed separately even though the query names the
#    same literal path for both (`probes` is keyed by ALIAS, not path) ------


def _video_rung_probes() -> dict[str, ProbeResult | None]:
    lo_v = _track("video", 0, width=1280, height=720)
    hi_v = _track("video", 1, width=1920, height=1080)
    return {
        "v": ProbeResult(
            streams=[lo_v, hi_v],
            renditions=[
                RenditionMeta(
                    streams=[lo_v], bandwidth=800_000, width=1280, height=720,
                    codecs="avc1.640020", name="Low", language=None, program_id=1,
                ),
                RenditionMeta(
                    streams=[hi_v], bandwidth=3_000_000, width=1920, height=1080,
                    codecs="avc1.64002a", name="High", language=None, program_id=2,
                ),
            ],
        )
    }


def _audio_rendition_probes(*, second_language: str | None) -> dict[str, ProbeResult | None]:
    """Two audio-only renditions -- one language-tagged, one whose
    ``language`` is `second_language` (a real tag, or NULL to test the
    NULL-in-a-destination refusal)."""
    en_a = _track("audio", 0, channels=2, language="en")
    other_a = _track("audio", 1, channels=2, language=second_language)
    return {
        "a": ProbeResult(
            streams=[en_a, other_a],
            renditions=[
                RenditionMeta(
                    streams=[en_a], bandwidth=128_000, width=None, height=None,
                    codecs="mp4a.40.2", name="English", language="en", program_id=3,
                ),
                RenditionMeta(
                    streams=[other_a], bandwidth=96_000, width=None, height=None,
                    codecs="mp4a.40.2", name="Other", language=second_language,
                    program_id=4,
                ),
            ],
        )
    }


_PAIRING_SQL = (
    "COPY (SELECT v.video[1], a.audio[1] FROM input('ladder.m3u8') v, "
    "input('ladder.m3u8') a{where}) "
    "TO ('out-' || v.height::text || 'p-' || a.language || '.mp4') "
    "WITH (video_codec 'libx264', audio_codec 'aac')"
)


def test_a_computed_to_pairs_one_video_rung_with_one_audio_rendition() -> None:
    """WHERE narrows each side to its own single row; the computed TO reads
    both aliases' rendition columns, evaluated per row exactly as a
    track-row column already is in a fan-out destination."""
    probes = {**_video_rung_probes(), **_audio_rendition_probes(second_language="es")}
    g = _lower(
        _PAIRING_SQL.format(where=" WHERE v.height = 720 AND a.language = 'en'"),
        probes,
    )
    assert [unit.path for unit in g.sinks] == ["out-720p-en.mp4"]
    args = build_ffmpeg_args(emit(g), None)
    maps = [args[i + 1] for i, arg in enumerate(args) if arg == "-map"]
    assert maps == ["0:v:0", "0:a:0"]


def test_a_two_by_two_rendition_self_join_fans_out_four_destinations() -> None:
    """No WHERE at all: 2 video rungs x 2 audio renditions is 4 pairings, one
    file each, named per pairing with its own ``-map`` pair."""
    probes = {**_video_rung_probes(), **_audio_rendition_probes(second_language="es")}
    g = _lower(_PAIRING_SQL.format(where=""), probes)
    assert [unit.path for unit in g.sinks] == [
        "out-720p-en.mp4",
        "out-720p-es.mp4",
        "out-1080p-en.mp4",
        "out-1080p-es.mp4",
    ]
    args = build_ffmpeg_args(emit(g), None)
    maps = [args[i + 1] for i, arg in enumerate(args) if arg == "-map"]
    assert maps == [
        "0:v:0", "0:a:0",  # 720p-en
        "0:v:0", "0:a:1",  # 720p-es
        "0:v:1", "0:a:0",  # 1080p-en
        "0:v:1", "0:a:1",  # 1080p-es
    ]


def test_a_null_rendition_column_in_a_destination_refuses_naming_the_row() -> None:
    """The second audio rendition carries no ``language`` at all: the NULL
    reaches the destination for its row, and is refused by name rather than
    written as the literal string 'None'."""
    probes = {**_video_rung_probes(), **_audio_rendition_probes(second_language=None)}
    err = _reject_lower(
        _PAIRING_SQL.format(where=" WHERE v.height = 720 AND a.name = 'Other'"),
        probes,
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert err.message == (
        "the TO expression is NULL for this row: 'a.language' was never probed"
    )
    assert err.hint == "COALESCE the column, or filter the rows that lack it"


def test_a_plain_file_reading_a_rendition_column_in_to_is_refused_the_same_way() -> None:
    """A computed TO reading a rendition column off a plain (non-ladder)
    input alias gets the exact same refusal a SELECT column reading one
    does -- the file's own rejection, not a generic 'does not bind'."""
    plain = ProbeResult(streams=[_track("video", 0), _track("audio", 0)])
    err = _reject_lower(
        "COPY (SELECT v.video[1] FROM input('x.mp4') v) "
        "TO ('out-' || v.height::text || '.mp4')",
        {"v": plain},
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert err.message == (
        "'v' is a single file, not a ladder: input('x.mp4') has no renditions"
    )
    assert err.hint == (
        "rendition columns (bandwidth, width, height, codecs, name, "
        "language) read from an HLS master or DASH manifest"
    )


def test_a_codecless_stream_is_rejected_before_ffmpeg_can_die_on_it() -> None:
    """A probed stream ffprobe reports NO codec for (a DASH manifest's WebVTT
    AdaptationSets, measured 2026-08-18 on ffmpeg 7.1 through 9.0) cannot be
    copied or transcoded; the mux is guaranteed to fail at header-write. We
    know at compile time, so the rejection happens at compile time."""
    probes = _row_probes(_track("video", 0), _track("data", 0, language="en", codec=None))
    err = _reject_lower("SELECT f.data[1] FROM input('f.mpd') f", probes)
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "no identifiable codec" in err.message
    assert "table row" in (err.hint or "")


def test_a_codecless_stream_poisons_the_bare_array_too() -> None:
    probes = _row_probes(_track("data", 0, codec="bin_data"), _track("data", 1, codec=None))
    err = _reject_lower("SELECT f.data FROM input('f.mpd') f", probes)
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "f.data[2]" in err.message


def test_a_codecless_stream_is_named_through_select_star() -> None:
    probes = _row_probes(_track("video", 0), _track("data", 0, codec=None))
    err = _reject_lower("SELECT * FROM input('f.mpd') f", probes)
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "'f.*' includes 'f.data[1]'" in err.message


def test_a_codecless_row_track_is_rejected_for_media_queries() -> None:
    probes = _row_probes(_track("data", 0, language="en", codec=None))
    err = _reject_lower(_row_query(column="data"), probes)
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "no identifiable codec" in err.message


def test_a_codecless_row_is_still_inspectable_as_a_table() -> None:
    """The exemption that makes the rejection honest: a table query SHOWS the
    codec-less tracks (codec column NULL) -- that is how you find out."""
    probes = _row_probes(_track("data", 0, language="en", codec=None))
    sinks = lower_table(
        resolve(parse("SELECT t, t.tags.language, t.codec FROM input('f.mpd') f, unnest(f.data) t")),
        probes,
    )
    assert len(sinks) == 1
    assert len(sinks[0].result.rows) == 1


def test_an_unprobed_input_is_not_accused_of_codeclessness() -> None:
    """meta None means NOTHING is known -- nothing is knowably broken, and the
    explicit subscript keeps compiling unchecked, as it always has."""
    g = _lower("SELECT f.data[1] FROM input('f.mpd') f", {"f": None})
    assert _refs(g) == ["src:f:d:0"]


def test_unnest_yields_one_array_element_per_track_in_file_order() -> None:
    g = _lower(_row_query(), _row_probes())
    assert _outputs(g) == [
        ("src:f:a:0", "audio", None),
        ("src:f:a:1", "audio", None),
        ("src:f:a:2", "audio", None),
    ]
    # Pure passthrough: no filter node, so the streams are stream-copyable.
    assert _filters(g) == []


def test_each_row_carries_its_own_stream_meta_as_provenance() -> None:
    g = _lower(_row_query(), _row_probes())
    assert [o.metadata for o in g.outputs] == [
        {"language": "eng"},
        {"language": "fra"},
        {},
    ]


def test_a_where_over_a_row_column_keeps_only_the_matching_rows() -> None:
    g = _lower(_row_query("t.tags.language = 'eng'"), _row_probes())
    assert _outputs(g) == [("src:f:a:0", "audio", None)]


def test_a_where_that_matches_nothing_is_a_typed_rejection() -> None:
    err = _reject_lower(_row_query("t.tags.language = 'deu'"), _row_probes())
    assert err.code is ErrorCode.STREAM_NOT_FOUND
    assert "selects nothing" in err.message


def test_null_matches_nothing_including_inequality() -> None:
    # The third track has no language tag, so every comparison against it is
    # UNKNOWN -- `!= 'eng'` does not rescue it, which is the whole SQL rule.
    g = _lower(_row_query("t.tags.language != 'eng'"), _row_probes())
    assert _outputs(g) == [("src:f:a:1", "audio", None)]


def test_is_null_and_is_not_null_select_the_two_halves() -> None:
    g = _lower(_row_query("t.tags.language IS NULL"), _row_probes())
    assert _outputs(g) == [("src:f:a:2", "audio", None)]
    g = _lower(_row_query("t.tags.language IS NOT NULL"), _row_probes())
    assert _outputs(g) == [("src:f:a:0", "audio", None), ("src:f:a:1", "audio", None)]


def test_an_unprobed_field_is_null_for_every_row() -> None:
    # `bitrate` was never set on any synthetic track.
    err = _reject_lower(_row_query("t.bitrate > 0"), _row_probes())
    assert err.code is ErrorCode.STREAM_NOT_FOUND
    g = _lower(_row_query("t.bitrate IS NULL"), _row_probes())
    assert len(g.outputs) == 3


@pytest.mark.parametrize(
    ("predicate", "expected"),
    [
        ("t.channels = 2", ["src:f:a:0", "src:f:a:2"]),
        ("t.channels > 2", ["src:f:a:1"]),
        ("t.channels >= 2", ["src:f:a:0", "src:f:a:1", "src:f:a:2"]),
        ("t.channels < 6", ["src:f:a:0", "src:f:a:2"]),
        ("t.channels BETWEEN 2 AND 5", ["src:f:a:0", "src:f:a:2"]),
        ("2 = t.channels", ["src:f:a:0", "src:f:a:2"]),
        ("6 > t.channels", ["src:f:a:0", "src:f:a:2"]),
        ("t.codec = 'aac' AND t.channels = 2", ["src:f:a:0", "src:f:a:2"]),
        ("t.tags.language = 'eng' OR t.tags.language = 'fra'", ["src:f:a:0", "src:f:a:1"]),
        ("NOT (t.channels = 2)", ["src:f:a:1"]),
        ("t.index = 1", ["src:f:a:0"]),
        ("t.index = 3", ["src:f:a:2"]),
        ("t.tags.language IN ('eng', 'fra')", ["src:f:a:0", "src:f:a:1"]),
        # NULL matches nothing, same as `=`: the untagged third track survives
        # neither the IN nor the NOT IN.
        ("t.tags.language NOT IN ('eng')", ["src:f:a:1"]),
        ("t.channels IN (6, 99)", ["src:f:a:1"]),
        ("t.channels NOT IN (2)", ["src:f:a:1"]),
    ],
)
def test_the_compile_time_predicate_evaluator(
    predicate: str, expected: list[str]
) -> None:
    g = _lower(_row_query(predicate), _row_probes())
    assert [o.ref for o in g.outputs] == expected


@pytest.mark.parametrize(
    ("desugared", "written"),
    [
        (
            "t.tags.language IN ('eng', 'fra')",
            "t.tags.language = 'eng' OR t.tags.language = 'fra'",
        ),
        (
            "t.tags.language NOT IN ('eng', 'deu')",
            "t.tags.language != 'eng' AND t.tags.language != 'deu'",
        ),
        ("t.channels IN (6)", "t.channels = 6"),
        ("t.channels NOT IN (2, 99)", "t.channels != 2 AND t.channels != 99"),
    ],
)
def test_in_desugars_to_the_same_graph_as_the_hand_written_chain(
    desugared: str, written: str
) -> None:
    """`IN`/`NOT IN` are a spelling, not a second evaluator: the graph an `IN`
    query lowers to is identical -- node for node -- to the one its `=`/`OR`
    (or `!=`/`AND`) equivalent lowers to."""
    via_in = _lower(_row_query(desugared), _row_probes())
    via_chain = _lower(_row_query(written), _row_probes())
    assert via_in.to_dict() == via_chain.to_dict()


def test_not_over_an_unknown_stays_unknown() -> None:
    # NOT UNKNOWN is UNKNOWN, so the untagged track is dropped by BOTH of these.
    for predicate in ("t.tags.language = 'eng'", "NOT (t.tags.language = 'eng')"):
        g = _lower(_row_query(predicate), _row_probes())
        assert "src:f:a:2" not in [o.ref for o in g.outputs]


def test_row_index_is_one_based_and_agrees_with_the_subscript() -> None:
    rows = _lower(_row_query("t.index = 2"), _row_probes())
    plain = _lower("SELECT f.audio[2] FROM input('f.mkv') f", _row_probes())
    assert _outputs(rows) == _outputs(plain) == [("src:f:a:1", "audio", None)]


def test_order_by_resorts_the_rows_at_compile_time() -> None:
    g = _lower(_row_query(order="t.channels DESC"), _row_probes())
    # 6ch first, then the two 2ch in their original (stable) order.
    assert [o.ref for o in g.outputs] == ["src:f:a:1", "src:f:a:0", "src:f:a:2"]


def test_order_by_puts_nulls_where_postgres_puts_them() -> None:
    # ASC -> NULLS LAST, DESC -> NULLS FIRST, both filled in by sqlglot.
    g = _lower(_row_query(order="t.tags.language"), _row_probes())
    assert [o.ref for o in g.outputs] == ["src:f:a:0", "src:f:a:1", "src:f:a:2"]
    g = _lower(_row_query(order="t.tags.language DESC"), _row_probes())
    assert [o.ref for o in g.outputs] == ["src:f:a:2", "src:f:a:1", "src:f:a:0"]
    g = _lower(_row_query(order="t.tags.language ASC NULLS FIRST"), _row_probes())
    assert [o.ref for o in g.outputs] == ["src:f:a:2", "src:f:a:0", "src:f:a:1"]


def test_order_by_applies_keys_left_to_right() -> None:
    probes = _row_probes(
        _track("audio", 0, language="eng", channels=6),
        _track("audio", 1, language="fra", channels=2),
        _track("audio", 2, language="eng", channels=2),
    )
    g = _lower(_row_query(order="t.tags.language, t.channels"), probes)
    assert [o.ref for o in g.outputs] == ["src:f:a:2", "src:f:a:0", "src:f:a:1"]


def test_where_and_order_by_compose() -> None:
    g = _lower(_row_query("t.codec = 'aac'", "t.index DESC"), _row_probes())
    assert [o.ref for o in g.outputs] == ["src:f:a:2", "src:f:a:0"]


def test_row_order_is_the_files_track_order_without_an_order_by() -> None:
    probes = _row_probes(
        _track("audio", 0, language="zza"),
        _track("audio", 1, language="aaa"),
    )
    g = _lower(_row_query(), probes)
    assert [o.metadata for o in g.outputs] == [{"language": "zza"}, {"language": "aaa"}]


# -- LIMIT / OFFSET: narrowing the resolved row set -------------------------


def test_limit_caps_the_rows_the_where_and_order_by_leave() -> None:
    # WHERE keeps the two aac tracks, ORDER BY flips them, LIMIT takes the top.
    sql = (
        "SELECT array_agg(t) FROM input('f.mkv') f, unnest(f.audio) t "
        "WHERE t.codec = 'aac' ORDER BY t.index DESC LIMIT 1"
    )
    g = _lower(sql, _row_probes())
    assert _outputs(g) == [("src:f:a:2", "audio", None)]


def test_offset_skips_before_limit_takes() -> None:
    sql = (
        "SELECT array_agg(t) FROM input('f.mkv') f, unnest(f.audio) t "
        "ORDER BY t.channels DESC LIMIT 1 OFFSET 1"
    )
    g = _lower(sql, _row_probes())
    # 6ch sorts first; OFFSET 1 skips it, LIMIT 1 takes the next (stable: a:0).
    assert _outputs(g) == [("src:f:a:0", "audio", None)]


def test_order_by_limit_one_satisfies_the_one_row_rule() -> None:
    sql = (
        "SELECT t FROM input('f.mkv') f, unnest(f.audio) t "
        "ORDER BY t.channels DESC LIMIT 1"
    )
    g = _lower(sql, _row_probes())
    assert _outputs(g) == [("src:f:a:1", "audio", None)]
    err = _reject_lower(sql.replace(" LIMIT 1", ""), _row_probes())
    assert err.code is ErrorCode.ROW_COUNT_MISMATCH


def test_offset_that_skips_every_row_is_a_rejection_naming_the_count() -> None:
    err = _reject_lower(
        "SELECT array_agg(t) FROM input('f.mkv') f, unnest(f.audio) t OFFSET 3",
        _row_probes(),
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert err.message == "OFFSET 3 skips every row: this query has 3 rows"


def test_an_array_element_reads_per_row() -> None:
    sinks = lower_table(
        resolve(
            parse(
                "SELECT ARRAY[10, 20][i.i] AS w, ARRAY['a', 'b'][i.i] AS n "
                "FROM generate_series(1, 2) i"
            )
        ),
        {},
    )
    assert sinks[0].result.rows == [[10, "a"], [20, "b"]]


def test_an_array_element_past_the_end_names_the_length() -> None:
    with pytest.raises(FfrwdError) as excinfo:
        lower_table(
            resolve(
                parse(
                    "SELECT ARRAY[10, 20][i.i] AS w FROM generate_series(1, 3) i"
                )
            ),
            {},
        )
    err = excinfo.value
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert err.message == "subscript 3 is past the end: the list has 2 elements"


def test_subtitle_rows_work_the_same_and_stay_passthrough() -> None:
    probes = _row_probes(
        _track("subtitle", 0, language="eng"),
        _track("subtitle", 1, language="fra"),
    )
    g = _lower(
        "SELECT s FROM input('f.mkv') f, unnest(f.subtitle) s "
        "WHERE s.tags.language = 'eng'",
        probes,
    )
    assert _outputs(g) == [("src:f:s:0", "subtitle", None)]
    assert _filters(g) == []


def test_video_rows_carry_the_video_schema() -> None:
    probes = _row_probes(
        _track("video", 0, width=1920, height=1080, color_transfer="smpte2084"),
        _track("video", 1, width=640, height=360),
    )
    g = _lower(_row_query("t.width >= 1280", column="video"), probes)
    assert _outputs(g) == [("src:f:v:0", "video", None)]
    g = _lower(_row_query("t.color_transfer = 'smpte2084'", column="video"), probes)
    assert _outputs(g) == [("src:f:v:0", "video", None)]


def test_a_bare_row_alias_is_that_rows_stream() -> None:
    """The row IS the stream, wherever a stream is expected."""
    probes = _row_probes()
    plain = _lower("SELECT t FROM input('f.mkv') f, unnest(f.audio) t WHERE t.index = 1", probes)
    assert _outputs(plain) == [("src:f:a:0", "audio", None)]
    filtered = _lower(
        "SELECT volume(t, 0.5) FROM input('f.mkv') f, unnest(f.audio) t "
        "WHERE t.index = 1",
        probes,
    )
    assert _filters(filtered) == ["volume"]
    gathered = _lower("SELECT array_agg(t) FROM input('f.mkv') f, unnest(f.audio) t", probes)
    assert len(gathered.outputs) == 3


def test_grouping_by_the_row_groups_by_the_stream_not_its_metadata() -> None:
    """Identity is the stream itself: two tracks agreeing on every column are
    still two groups, and two groups need a file each."""
    probes = _row_probes(
        _track("video", 0, width=640, height=360),
        _track("video", 1, width=640, height=360),
    )
    sinks = lower_table(
        resolve(
            parse(
                "SELECT v, array_agg(v) FROM input('f.mkv') f, unnest(f.video) v "
                "GROUP BY v"
            )
        ),
        probes,
    )
    assert len(sinks[0].result.rows) == 2


def test_selecting_a_metadata_column_is_a_typed_rejection() -> None:
    err = _reject_lower(
        "SELECT t.tags.language FROM input('f.mkv') f, unnest(f.audio) t", _row_probes()
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "is track metadata, not a stream" in err.message
    assert err.hint is not None and "the row itself, <alias>" in err.hint


def test_unnesting_an_unprobeable_input_is_a_typed_rejection() -> None:
    err = _reject_lower(_row_query(), {"f": None})
    assert err.code is ErrorCode.INPUT_NOT_FOUND
    assert "cannot unnest 'f.audio'" in err.message


def test_unnesting_an_empty_array_selects_nothing() -> None:
    err = _reject_lower(_row_query(), _row_probes(_track("video", 0)))
    assert err.code is ErrorCode.STREAM_NOT_FOUND
    assert "selects nothing" in err.message


def test_a_star_over_a_row_table_is_not_a_media_column() -> None:
    err = _reject_lower(
        "SELECT t.* FROM input('f.mkv') f, unnest(f.audio) t", _row_probes()
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "expands their fields" in err.message
    assert err.hint is not None and "the row is the stream: select t" in err.hint


def test_star_leaves_chapters_out_of_a_media_query() -> None:
    """A chapter is not a stream, so the star expands the stream arrays only."""
    g = _lower(
        "SELECT * FROM input('x.mkv') a",
        {
            "a": ProbeResult(
                streams=list(_layout_probe("va").streams),
                chapters=[ChapterMeta(index=1, start_t=0.0, end_t=1.0, title="Intro")],
            )
        },
    )
    assert [o.type for o in g.outputs] == ["video", "audio"]


def test_star_over_a_container_is_every_array_column_in_a_table_query() -> None:
    sinks = lower_table(
        resolve(parse("SELECT * FROM input('f.mkv') f")),
        _chapter_probes(*_TWO_CHAPTERS),
    )
    assert sinks[0].result.columns == [
        "video",
        "audio",
        "subtitle",
        "data",
        "chapters",
        "attachments",
    ]
    video, audio, subtitle, data, chapters, _ = sinks[0].result.rows[0]
    assert video == ArrayCell(elements=(StreamCell(type="video", spec="0:v:0"),))
    assert audio == ArrayCell(elements=(StreamCell(type="audio", spec="0:a:0"),))
    assert subtitle == ArrayCell(elements=())
    assert data == ArrayCell(elements=())
    assert isinstance(chapters, ArrayCell) and len(chapters.elements) == 2


def test_star_over_track_rows_is_the_records_scalar_fields() -> None:
    """The metadata table: the maps stay out, and are read by name instead."""
    sinks = lower_table(
        resolve(parse("SELECT t.* FROM input('f.mkv') f, unnest(f.audio) t")),
        _row_probes(),
    )
    assert sinks[0].result.columns == [
        "index",
        "codec",
        "channels",
        "channel_layout",
        "sample_rate",
        "bitrate",
        "duration",
    ]
    assert sinks[0].result.rows[0] == [1, "aac", 2, "stereo", None, None, None]


def test_a_star_over_chapter_rows_names_no_stream_to_select() -> None:
    err = _reject_lower(
        "SELECT c.* FROM input('f.mkv') f, unnest(f.chapters) c",
        _chapter_probes(*_TWO_CHAPTERS),
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert err.hint is not None and "these rows carry no stream" in err.hint


def test_star_over_chapter_rows_is_the_chapter_fields() -> None:
    sinks = lower_table(
        resolve(parse("SELECT c.* FROM input('f.mkv') f, unnest(f.chapters) c")),
        _chapter_probes(*_TWO_CHAPTERS),
    )
    assert sinks[0].result.columns == ["index", "title", "start_t", "end_t"]
    assert sinks[0].result.rows == [[1, "Intro", 0.0, 1.0], [2, "Credits", 1.0, 2.0]]


def test_a_bare_star_covers_container_and_rows_in_a_table_query() -> None:
    sinks = lower_table(
        resolve(parse("SELECT * FROM input('f.mkv') f, unnest(f.audio) t")),
        _row_probes(_track("audio", 0, codec="aac")),
    )
    assert sinks[0].result.columns == [
        "video",
        "audio",
        "subtitle",
        "data",
        "chapters",
        "attachments",
        "index",
        "codec",
        "channels",
        "channel_layout",
        "sample_rate",
        "bitrate",
        "duration",
    ]


def test_star_over_a_cte_prints_its_columns_in_a_table_query() -> None:
    sinks = lower_table(
        resolve(parse(
            "WITH c AS (SELECT t AS track FROM input('f.mkv') f, unnest(f.audio) t) "
            "SELECT * FROM c"
        )),
        _row_probes(),
    )
    assert sinks[0].result.columns == ["track"]
    assert sinks[0].result.rows == [
        [StreamCell(type="audio", spec="0:a:0")],
        [StreamCell(type="audio", spec="0:a:1")],
        [StreamCell(type="audio", spec="0:a:2")],
    ]


def test_a_bare_star_is_not_a_group_key() -> None:
    err = _reject_lower_table(
        "SELECT *, array_agg(t) FROM input('f.mkv') f, unnest(f.audio) t "
        "GROUP BY t.tags.language",
        _row_probes(),
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "'*' is neither aggregated nor a GROUP BY key" in err.message


def test_a_row_alias_is_an_array_and_subscripts_like_one() -> None:
    g = _lower(
        "SELECT array_agg(t[2]) FROM input('f.mkv') f, unnest(f.audio) t",
        _row_probes(),
    )
    assert _outputs(g) == [("src:f:a:1", "audio", None)]
    err = _reject_lower(
        "SELECT t[9] FROM input('f.mkv') f, unnest(f.audio) t", _row_probes()
    )
    assert err.code is ErrorCode.STREAM_NOT_FOUND
    assert "'t' has 3 rows" in err.message


def test_the_row_array_broadcasts_a_call_like_any_other_array() -> None:
    g = _lower(
        "SELECT array_agg(volume(t, 0.5)) FROM input('f.mkv') f, "
        "unnest(f.audio) t WHERE t.channels = 2",
        _row_probes(),
    )
    assert _filters(g) == ["volume", "volume"]
    assert [o.metadata for o in g.outputs] == [{"language": "eng"}, {}]


def test_a_row_query_works_inside_a_cte_body() -> None:
    g = _lower(
        "WITH picked AS ("
        "  SELECT t AS a FROM input('f.mkv') f, unnest(f.audio) t "
        "  WHERE t.channel_layout = 'stereo'"
        ") SELECT picked.a FROM picked",
        _row_probes(),
    )
    # The CTE column keeps its AS name inside the body; the outer bare
    # `picked.a` reference names nothing, exactly as for any other CTE column.
    assert _outputs(g) == [("src:f:a:0", "audio", None)]


def test_a_row_query_works_in_a_union_all_branch() -> None:
    probes = _row_probes()
    probes["g"] = probes["f"]
    g = _lower(
        "SELECT f.video[1], t FROM input('f.mkv') f, unnest(f.audio) t "
        "WHERE t.tags.language = 'eng' "
        "UNION ALL "
        "SELECT g.video[1], u FROM input('g.mkv') g, unnest(g.audio) u "
        "WHERE u.tags.language = 'fra'",
        probes,
    )
    assert _filters(g) == ["concat"]


def test_a_time_window_still_reaches_the_input_of_a_row_query() -> None:
    g = _lower(_row_query("f.t BETWEEN 1 AND 2 AND t.tags.language = 'eng'"), _row_probes())
    assert g.input_trims == {"f": (1, 2)}
    assert _outputs(g) == [("src:f:a:0", "audio", None)]


def test_a_seeked_caption_row_is_still_rejected() -> None:
    # The caption-seek rejection keys off the INPUT alias, and a row table's
    # streams belong to that input -- so the rejection survives the indirection.
    probes = _row_probes(_track("subtitle", 0, language="eng"))
    err = _reject_lower(
        "SELECT s FROM input('f.mkv') f, unnest(f.subtitle) s "
        "WHERE f.t >= 1 AND s.tags.language = 'eng'",
        probes,
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "cannot trim a selected subtitle stream" in err.message


@pytest.mark.exec
def test_a_track_row_query_runs_end_to_end(tmp_path: Path) -> None:
    """Real probe, real ffmpeg: picking a track by its language actually runs.

    av2.mp4 carries an `eng` and a `fra` audio track, so this is the cookbook's
    recipe 23 shape with the compiled command actually executed -- the row set
    is decided from a real ffprobe, and what comes out is one stream-copied
    English track.
    """
    out = tmp_path / "eng.m4a"
    query = (
        f"SELECT t FROM input('{(FIXTURES_DIR / 'av2.mp4').as_posix()}') f, "
        "unnest(f.audio) t WHERE t.tags.language = 'eng'"
    )
    args = build_ffmpeg_args(emit(compile_sql(query)), str(out))
    assert "-map" in args and "0:a:0" in args
    args.insert(1, "-y")
    result = subprocess.run(args, capture_output=True, text=True, timeout=60.0)
    assert result.returncode == 0, result.stderr
    assert out.exists()


# ---------------------------------------------------------------------------
# unnest(f.chapters) read, and the `chapters` output column
# ---------------------------------------------------------------------------


def _chapter_probes(*chapters: ChapterMeta) -> dict[str, ProbeResult | None]:
    return {"f": ProbeResult(streams=[_track("video", 0), _track("audio", 0)], chapters=list(chapters))}


_TWO_CHAPTERS = [
    ChapterMeta(index=1, start_t=0.0, end_t=1.0, title="Intro"),
    ChapterMeta(index=2, start_t=1.0, end_t=2.0, title="Credits"),
]


def test_chapters_table_output_reads_the_fixed_schema() -> None:
    sinks = lower_table(
        resolve(parse(
            "SELECT c.index, c.title, c.start_t, c.end_t "
            "FROM input('f.mkv') f, unnest(f.chapters) c"
        )),
        _chapter_probes(*_TWO_CHAPTERS),
    )
    assert len(sinks) == 1
    assert sinks[0].result.columns == ["index", "title", "start_t", "end_t"]
    assert sinks[0].result.rows == [[1, "Intro", 0.0, 1.0], [2, "Credits", 1.0, 2.0]]


def test_chapters_where_and_order_by_use_the_same_row_machinery_unnest_does() -> None:
    sinks = lower_table(
        resolve(parse(
            "SELECT c.title FROM input('f.mkv') f, unnest(f.chapters) c "
            "WHERE c.start_t >= 1 ORDER BY c.title DESC"
        )),
        _chapter_probes(*_TWO_CHAPTERS),
    )
    assert sinks[0].result.rows == [["Credits"]]


def test_an_unreadable_input_rejects_chapters_like_unnest() -> None:
    err = _reject_lower(
        "SELECT c.title FROM input('f.mkv') f, unnest(f.chapters) c", {"f": None}
    )
    assert err.code is ErrorCode.INPUT_NOT_FOUND


def test_a_chapters_column_in_a_media_copy_is_a_typed_rejection_not_a_tag() -> None:
    """A row column with no alias would otherwise be checked as a tag; a
    chapters row has no stream to tag at all, so it must fall through to the
    ordinary "not an output" rejection instead."""
    err = _reject_lower(
        "COPY (SELECT c.title FROM input('f.mkv') f, unnest(f.chapters) c) TO 'out.mkv'",
        _chapter_probes(*_TWO_CHAPTERS),
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "'c.title' is track metadata, not a stream" in err.message
    assert "a chapter row has no stream column" in (err.hint or "")


def test_a_bare_chapters_column_prints_as_one_array_cell() -> None:
    """The array VALUE, not a row source: records in schema order, Postgres
    array-literal braces around them."""
    sinks = lower_table(
        resolve(parse("SELECT f.tags.title, f.chapters FROM input('f.mkv') f")),
        _chapter_probes(*_TWO_CHAPTERS),
    )
    assert sinks[0].result.columns == ["title", "chapters"]
    assert sinks[0].result.rows == [
        [
            None,
            ArrayCell(
                elements=(
                    RecordCell(fields=(1, "Intro", 0.0, 1.0)),
                    RecordCell(fields=(2, "Credits", 1.0, 2.0)),
                )
            ),
        ]
    ]
    assert "{(1,Intro,0.0,1.0),(2,Credits,1.0,2.0)}" in render_table(sinks[0].result)


def test_a_bare_chapters_column_broadcasts_over_track_rows() -> None:
    """One cell per printed row, the same broadcast a bare stream array does."""
    sinks = lower_table(
        resolve(parse(
            "SELECT t.index, f.chapters FROM input('f.mkv') f, unnest(f.audio) t"
        )),
        _chapter_probes(*_TWO_CHAPTERS),
    )
    assert len(sinks[0].result.rows) == 1
    assert isinstance(sinks[0].result.rows[0][1], ArrayCell)


def test_a_bare_chapters_column_needs_a_readable_input() -> None:
    err = _reject_lower_table("SELECT f.chapters FROM input('f.mkv') f", {"f": None})
    assert err.code is ErrorCode.INPUT_NOT_FOUND
    assert "cannot read chapters" in err.message


def test_a_bare_chapters_column_in_a_media_copy_is_a_typed_rejection() -> None:
    """Chapters are records, not streams: nothing to map into a container."""
    err = _reject_lower(
        "COPY (SELECT f.chapters FROM input('f.mkv') f) TO 'out.mkv'",
        _chapter_probes(*_TWO_CHAPTERS),
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "'f.chapters' carries no streams" in err.message
    assert "unnest(f.chapters) c" in (err.hint or "")


def test_a_chapters_subscript_is_rejected_in_a_table_query_too() -> None:
    err = _reject_lower_table(
        "SELECT f.chapters[1] FROM input('f.mkv') f", _chapter_probes(*_TWO_CHAPTERS)
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "'f.chapters' cannot be subscripted" in err.message
    assert "unnest(f.chapters) c" in (err.hint or "")


def test_chapter_rows_cross_join_track_rows() -> None:
    """No mixing carve-out left: one row per (track, chapter) pair."""
    sinks = lower_table(
        resolve(parse(
            "SELECT t.index, c.title "
            "FROM input('f.mkv') f, unnest(f.audio) t, unnest(f.chapters) c"
        )),
        _chapter_probes(*_TWO_CHAPTERS),
    )
    assert sinks[0].result.rows == [[1, "Intro"], [1, "Credits"]]


def test_a_chapters_column_mints_an_ffmetadata_input_and_sets_map_chapters() -> None:
    g = _lower(
        "COPY (SELECT f.video[1], f.audio[1], "
        "ARRAY[STRUCT('Intro' AS title, 0 AS start_t, 60 AS end_t)::chapter, "
        "STRUCT('Act One' AS title, 60 AS start_t, 300 AS end_t)::chapter] "
        "AS chapters FROM input('film.mkv') f) TO 'out.mkv'"
    )
    assert len(g.input_paths) == 2
    assert g.input_paths[1].startswith("data:text/plain;base64,")
    assert g.sinks[0].chapters == 1
    minted_alias = next(alias for alias, index in g.sources.items() if index == 1)
    assert g.input_options[minted_alias] == {"format": "ffmetadata"}


def test_a_copied_chapters_column_reuses_that_inputs_index_with_no_extra_i() -> None:
    g = _lower(
        "COPY (SELECT f.video[1], f.audio[1], f.chapters AS chapters "
        "FROM input('film.mkv') f) TO 'out.mkv'"
    )
    assert len(g.input_paths) == 1
    assert g.sinks[0].chapters == 0


def test_a_copied_chapters_column_names_the_second_inputs_index() -> None:
    g = _lower(
        "COPY (SELECT f.video[1], f.audio[1], g.chapters AS chapters "
        "FROM input('film.mkv') f, input('extra.mkv') g) TO 'out.mkv'"
    )
    assert g.sinks[0].chapters == 1


def test_a_null_chapters_column_writes_no_chapter_list() -> None:
    g = _lower(
        "COPY (SELECT f.video[1], f.audio[1], NULL AS chapters "
        "FROM input('film.mkv') f) TO 'out.mkv'"
    )
    assert len(g.input_paths) == 1
    assert g.sinks[0].chapters == NO_CHAPTERS


def test_no_chapters_column_leaves_ffmpegs_own_default_alone() -> None:
    g = _lower(
        "COPY (SELECT f.video[1], f.audio[1] FROM input('film.mkv') f) TO 'out.mkv'"
    )
    assert g.sinks[0].chapters is None


def test_the_chapters_sink_option_is_gone() -> None:
    err = _reject(
        "COPY (SELECT f.video[1] FROM input('film.mkv') f) "
        "TO 'out.mkv' WITH (chapters marks)"
    )
    assert err.code is ErrorCode.UNKNOWN_SINK_OPTION


def test_the_chapters_from_sink_option_is_gone() -> None:
    err = _reject(
        "COPY (SELECT f.video[1] FROM input('film.mkv') f) "
        "TO 'out.mkv' WITH (chapters_from f)"
    )
    assert err.code is ErrorCode.UNKNOWN_SINK_OPTION


def test_a_chapters_column_must_be_an_array_of_records() -> None:
    err = _reject(
        "COPY (SELECT f.video[1], 'Intro' AS chapters FROM input('film.mkv') f) "
        "TO 'out.mkv'"
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "takes an array of chapter records" in err.message


def test_a_positional_chapter_record_is_rejected() -> None:
    err = _reject(
        "COPY (SELECT f.video[1], ARRAY[ROW('Intro', 0, 60)::chapter] AS chapters "
        "FROM input('film.mkv') f) TO 'out.mkv'"
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert (
        "a chapter is written as STRUCT(... AS title, ... AS start_t, "
        "... AS end_t)::chapter"
    ) in err.message


def test_a_chapters_element_must_be_a_chapter_record() -> None:
    err = _reject(
        "COPY (SELECT f.video[1], ARRAY['Intro'] AS chapters "
        "FROM input('film.mkv') f) TO 'out.mkv'"
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert (
        "a chapter is written as STRUCT(... AS title, ... AS start_t, "
        "... AS end_t)::chapter"
    ) in err.message


def test_a_chapters_column_in_a_cte_body_is_rejected() -> None:
    err = _reject(
        "COPY (WITH gathered AS (SELECT f.video[1] AS v, "
        "ARRAY[STRUCT('Intro' AS title, 0 AS start_t, 60 AS end_t)::chapter] AS chapters "
        "FROM input('film.mkv') f) "
        "SELECT gathered.v FROM gathered) TO 'out.mkv'"
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "a CTE body writes no file" in err.message


def test_a_chapter_records_values_take_the_value_grammar() -> None:
    g = _lower(
        "COPY (SELECT f.video[1], "
        "ARRAY[STRUCT('Part ' || 'One' AS title, 0 AS start_t, 30 + 30 AS end_t)::chapter] AS chapters "
        "FROM input('film.mkv') f) TO 'out.mkv'"
    )
    payload = base64.b64decode(g.input_paths[1].split(",", 1)[1]).decode()
    assert "title=Part One" in payload
    assert "END=60" in payload


def test_a_chapter_records_bounds_may_read_an_inputs_duration() -> None:
    g = _lower(
        "COPY (SELECT f.video[1], ARRAY[STRUCT('All' AS title, 0 AS start_t, f.duration AS end_t)::chapter] "
        "AS chapters FROM input('film.mkv') f) TO 'out.mkv'",
        {"f": ProbeResult(streams=[_track("video", 0)], duration=12.5)},
    )
    payload = base64.b64decode(g.input_paths[1].split(",", 1)[1]).decode()
    # A fractional bound counts in milliseconds: ffmetadata's START/END are
    # integers, and 1/1 would truncate 12.5 seconds to 12.
    assert "TIMEBASE=1/1000" in payload
    assert "END=12500" in payload


def test_array_agg_over_a_values_table_builds_the_same_document() -> None:
    """The two spellings of recipes 40 and 63: one file, one set of bytes."""
    literal = _lower(
        "COPY (SELECT f.video[1], "
        "ARRAY[STRUCT('Intro' AS title, 0 AS start_t, 60 AS end_t)::chapter, "
        "STRUCT('Act One' AS title, 60 AS start_t, 300 AS end_t)::chapter] "
        "AS chapters FROM input('film.mkv') f) TO 'out.mkv'"
    )
    gathered = _lower(
        "COPY (SELECT f.video[1], "
        "array_agg(STRUCT(m.title AS title, m.start_t AS start_t, "
        "m.end_t AS end_t)::chapter) AS chapters "
        "FROM input('film.mkv') f, "
        "unnest(ARRAY[STRUCT(0 AS start_t, 60 AS end_t, 'Intro' AS title), "
        "STRUCT(60 AS start_t, 300 AS end_t, 'Act One' AS title)]) m "
        "GROUP BY f.video[1]) TO 'out.mkv'"
    )
    assert gathered.input_paths == literal.input_paths


def test_array_agg_over_chapter_rows_copies_a_list_relationally() -> None:
    g = _lower(
        "COPY (SELECT f.video[1], "
        "array_agg(STRUCT(c.title AS title, c.start_t AS start_t, c.end_t AS end_t)::chapter) AS chapters "
        "FROM input('f.mkv') f, unnest(f.chapters) c GROUP BY f.video[1]) "
        "TO 'out.mkv'",
        _chapter_probes(*_TWO_CHAPTERS),
    )
    payload = base64.b64decode(g.input_paths[1].split(",", 1)[1]).decode()
    assert payload.count("[CHAPTER]") == 2
    assert "title=Intro" in payload and "title=Credits" in payload


def test_struct_gather_feeding_chapters_compiles_the_same_as_array_agg_struct() -> None:
    """``ARRAY(SELECT AS STRUCT ...)`` feeding a ``chapters`` column is byte
    for byte the hand-written ``array_agg(STRUCT(...)::chapter)`` spelling --
    a new way to BUILD the array, the same semantics consuming it."""
    gathered = (
        "COPY (SELECT f.video[1], ARRAY(SELECT AS STRUCT c.title, c.start_t, "
        "c.end_t FROM unnest(ARRAY[STRUCT('Intro' AS title, 0 AS start_t, "
        "60 AS end_t), STRUCT('Credits' AS title, 60 AS start_t, "
        "90 AS end_t)]) c) AS chapters FROM input('f.mkv') f) TO 'out.mkv'"
    )
    by_hand = (
        "COPY (SELECT f.video[1], array_agg(STRUCT(c.title AS title, "
        "c.start_t AS start_t, c.end_t AS end_t)::chapter) AS chapters "
        "FROM input('f.mkv') f, unnest(ARRAY[STRUCT('Intro' AS title, "
        "0 AS start_t, 60 AS end_t), STRUCT('Credits' AS title, "
        "60 AS start_t, 90 AS end_t)]) c GROUP BY f.video[1]) TO 'out.mkv'"
    )
    g1 = _lower(gathered, {})
    g2 = _lower(by_hand, {})
    assert g1.input_paths == g2.input_paths
    payload = base64.b64decode(g1.input_paths[1].split(",", 1)[1]).decode()
    assert payload.count("[CHAPTER]") == 2
    assert "title=Intro" in payload and "title=Credits" in payload


def test_a_chapters_column_groups_with_the_stream_columns() -> None:
    """The chapter list is a GROUP-level value, so the rows it aggregates do
    not each become a file of their own."""
    g = _lower(
        "COPY (SELECT f.video[1], f.audio[1], "
        "array_agg(STRUCT(c.title AS title, c.start_t AS start_t, c.end_t AS end_t)::chapter) AS chapters "
        "FROM input('f.mkv') f, unnest(f.chapters) c "
        "GROUP BY f.video[1], f.audio[1]) TO 'out.mkv'",
        _chapter_probes(*_TWO_CHAPTERS),
    )
    assert len(g.sinks) == 1
    assert [o.type for o in g.sinks[0].outputs] == ["video", "audio"]


def test_a_chapters_column_rejects_a_non_numeric_start_t() -> None:
    err = _reject(
        "COPY (SELECT f.video[1], ARRAY[STRUCT('Intro' AS title, 'x' AS start_t, 60 AS end_t)::chapter] AS chapters "
        "FROM input('film.mkv') f) TO 'out.mkv'"
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "start_t' must be a number" in err.message


def test_a_chapters_column_null_title_omits_the_title_line() -> None:
    g = _lower(
        "COPY (SELECT f.video[1], ARRAY[STRUCT(NULL AS title, 0 AS start_t, 60 AS end_t)::chapter] AS chapters "
        "FROM input('film.mkv') f) TO 'out.mkv'"
    )
    payload = base64.b64decode(g.input_paths[1].split(",", 1)[1]).decode()
    assert "title=" not in payload
    assert "START=0" in payload and "END=60" in payload


def test_a_chapters_column_rejects_an_unescapable_title() -> None:
    err = _reject(
        "COPY (SELECT f.video[1], ARRAY[STRUCT('a=b' AS title, 0 AS start_t, 60 AS end_t)::chapter] AS chapters "
        "FROM input('film.mkv') f) TO 'out.mkv'"
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "cannot represent unescaped" in err.message


def test_a_chapters_column_rejects_a_chapter_that_ends_before_it_starts() -> None:
    err = _reject(
        "COPY (SELECT f.video[1], ARRAY[STRUCT('Intro' AS title, 60 AS start_t, 30 AS end_t)::chapter] AS chapters "
        "FROM input('film.mkv') f) TO 'out.mkv'"
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "ends at 30, which is not after its start 60" in err.message


def test_a_chapters_column_rejects_a_zero_length_chapter() -> None:
    err = _reject(
        "COPY (SELECT f.video[1], ARRAY[STRUCT('Intro' AS title, 0 AS start_t, 0 AS end_t)::chapter] AS chapters "
        "FROM input('film.mkv') f) TO 'out.mkv'"
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "is not after its start" in err.message


def test_a_chapters_column_rejects_records_out_of_order() -> None:
    err = _reject(
        "COPY (SELECT f.video[1], "
        "ARRAY[STRUCT('Two' AS title, 60 AS start_t, 120 AS end_t)::chapter, STRUCT('One' AS title, 0 AS start_t, 30 AS end_t)::chapter] AS chapters "
        "FROM input('film.mkv') f) TO 'out.mkv'"
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "starts at 0, before chapter 1 at 60" in err.message
    assert "ascending" in (err.hint or "")


def test_a_chapters_column_rejects_overlapping_records() -> None:
    err = _reject(
        "COPY (SELECT f.video[1], "
        "ARRAY[STRUCT('One' AS title, 0 AS start_t, 60 AS end_t)::chapter, STRUCT('Two' AS title, 30 AS start_t, 90 AS end_t)::chapter] AS chapters "
        "FROM input('film.mkv') f) TO 'out.mkv'"
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "starts at 30, inside chapter 1 which ends at 60" in err.message


def test_a_chapters_column_accepts_back_to_back_records() -> None:
    """Touching is not overlapping: one ends exactly where the next begins."""
    g = _lower(
        "COPY (SELECT f.video[1], "
        "ARRAY[STRUCT('One' AS title, 0 AS start_t, 60 AS end_t)::chapter, STRUCT('Two' AS title, 60 AS start_t, 120 AS end_t)::chapter] AS chapters "
        "FROM input('film.mkv') f) TO 'out.mkv'"
    )
    payload = base64.b64decode(g.input_paths[1].split(",", 1)[1]).decode()
    assert payload.count("[CHAPTER]") == 2


def test_a_tag_column_over_chapter_rows_tags_the_container() -> None:
    """A chapter row carries no track, so the map goes on the file itself."""
    g = _lower(
        "COPY (SELECT f.video[1], STRUCT('Ch: ' || c.title AS title) AS tags "
        "FROM input('f.mkv') f, unnest(f.chapters) c WHERE c.index = 1) TO 'o.mkv'",
        _chapter_probes(*_TWO_CHAPTERS),
    )
    assert g.sinks[0].tags == {"title": "Ch: Intro"}



# ---------------------------------------------------------------------------
# unnest(v.cues) read, and a cue[] written as a WebVTT subtitle track
# ---------------------------------------------------------------------------

_THREE_CUES = [
    CueMeta(index=1, text="Cue one.", start_t=0.0, end_t=0.6),
    CueMeta(index=2, text="Cue two.", start_t=0.7, end_t=1.3),
    CueMeta(index=3, text="Cue three.", start_t=1.4, end_t=2.0),
]


def _cue_probes() -> dict[str, ProbeResult | None]:
    """One WebVTT sidecar beside an ordinary file, the shape both recipes use."""
    return {
        "f": ProbeResult(
            streams=[_track("video", 0), _track("audio", 0)],
            format_name="matroska,webm",
        ),
        "v": ProbeResult(
            streams=[_track("subtitle", 0)],
            format_name="webvtt",
            cues=list(_THREE_CUES),
        ),
    }


_FROM_VTT = "FROM input('f.mkv') f, input('subs.en.vtt') v, unnest(v.cues) c"


def _vtt_payload(g: Graph, index: int) -> str:
    return base64.b64decode(g.input_paths[index].split(",", 1)[1]).decode()


def test_cue_rows_read_the_documents_own_order_and_fields() -> None:
    sinks = lower_table(
        resolve(parse("SELECT c.index, c.text, c.start_t, c.end_t " + _FROM_VTT)),
        _cue_probes(),
    )
    assert sinks[0].result.columns == ["index", "text", "start_t", "end_t"]
    assert sinks[0].result.rows == [
        [1, "Cue one.", 0.0, 0.6],
        [2, "Cue two.", 0.7, 1.3],
        [3, "Cue three.", 1.4, 2.0],
    ]


def test_cue_rows_filter_and_order_like_any_other_rows() -> None:
    sinks = lower_table(
        resolve(
            parse(
                "SELECT c.text " + _FROM_VTT + " WHERE c.start_t >= 0.7 "
                "ORDER BY c.start_t DESC"
            )
        ),
        _cue_probes(),
    )
    assert sinks[0].result.rows == [["Cue three."], ["Cue two."]]


def test_a_bare_cues_column_prints_as_one_array_cell() -> None:
    sinks = lower_table(
        resolve(parse("SELECT v.cues FROM input('subs.en.vtt') v")), _cue_probes()
    )
    cell = sinks[0].result.rows[0][0]
    assert isinstance(cell, ArrayCell)
    # A document is its own track and has no title, so `track` reads NULL.
    assert cell.elements[0] == RecordCell(fields=(1, None, "Cue one.", 0.0, 0.6))


def test_cues_stay_out_of_a_container_star() -> None:
    """`cues` is read-only, and a star names what a copy carries through."""
    sinks = lower_table(resolve(parse("SELECT f.* FROM input('f.mkv') f")), _cue_probes())
    assert sinks[0].result.columns == [
        "video",
        "audio",
        "subtitle",
        "data",
        "chapters",
        "attachments",
    ]


def test_a_container_with_no_caption_track_reads_no_cues() -> None:
    """A file that carries none has none to show -- zero rows, not a rejection."""
    sinks = lower_table(
        resolve(parse("SELECT c.text FROM input('f.mkv') f, unnest(f.cues) c")),
        _cue_probes(),
    )
    assert sinks[0].result.rows == []


def test_an_unreadable_input_rejects_cues_like_any_other_unnest() -> None:
    err = _reject_lower_table(
        "SELECT c.text FROM input('v.vtt') v, unnest(v.cues) c", {"v": None}
    )
    assert err.code is ErrorCode.INPUT_NOT_FOUND
    assert "cannot read cues of 'v.vtt'" in err.message


def test_a_webvtt_input_with_no_cues_reads_no_rows() -> None:
    sinks = lower_table(
        resolve(parse("SELECT c.text FROM input('v.vtt') v, unnest(v.cues) c")),
        {"v": ProbeResult(streams=[_track("subtitle", 0)], format_name="webvtt")},
    )
    assert sinks[0].result.rows == []


def test_a_cue_row_carries_no_stream_to_select() -> None:
    err = _reject_lower(
        "COPY (SELECT c FROM input('subs.en.vtt') v, unnest(v.cues) c) TO 'o.mkv'",
        _cue_probes(),
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "'c' is a cue row, not a stream" in err.message
    assert "a cue row has no stream to select" in (err.hint or "")


def test_a_cue_array_literal_mints_one_webvtt_input_and_maps_it() -> None:
    g = _lower(
        "COPY (SELECT f.video[1], ARRAY[STRUCT('Hello' AS text, 0 AS start_t, 2.5 AS end_t)::cue] "
        "FROM input('f.mkv') f) TO 'out.mkv'",
        _cue_probes(),
    )
    minted = "ffrwd.cues#2"
    assert g.sources[minted] == 1
    assert g.input_options[minted] == {"format": "webvtt"}
    assert _vtt_payload(g, 1) == (
        "WEBVTT\n\n00:00:00.000 --> 00:00:02.500\nHello\n"
    )
    assert _refs(g) == ["src:f:v:0", f"src:{minted}:s:0"]


def test_a_cue_array_is_a_subtitle_stream_and_not_a_chapters_column() -> None:
    g = _lower(
        "COPY (SELECT f.video[1], ARRAY[STRUCT('Hello' AS text, 0 AS start_t, 2.5 AS end_t)::cue] "
        "FROM input('f.mkv') f) TO 'out.mkv'",
        _cue_probes(),
    )
    assert g.sinks[0].chapters is None
    assert [output.type for output in g.outputs] == ["video", "subtitle"]


def test_array_agg_over_chapter_rows_writes_the_chapter_list_as_cues() -> None:
    g = _lower(
        "COPY (SELECT f.video[1], "
        "array_agg(STRUCT(c.title AS text, c.start_t AS start_t, c.end_t AS end_t)::cue) "
        "FROM input('f.mkv') f, unnest(f.chapters) c GROUP BY f.video[1]) "
        "TO 'out.mkv'",
        {
            "f": ProbeResult(
                streams=[_track("video", 0)],
                chapters=[
                    ChapterMeta(index=1, start_t=0.0, end_t=1.0, title="Intro"),
                    ChapterMeta(index=2, start_t=1.0, end_t=2.0, title="Credits"),
                ],
            )
        },
    )
    assert _vtt_payload(g, 1) == (
        "WEBVTT\n\n"
        "00:00:00.000 --> 00:00:01.000\nIntro\n\n"
        "00:00:01.000 --> 00:00:02.000\nCredits\n"
    )


def test_array_agg_over_cue_rows_writes_them_back_as_a_chapter_list() -> None:
    """The mirror: a .vtt file imported as chapters, milliseconds intact."""
    g = _lower(
        "COPY (SELECT f.video[1], "
        "array_agg(STRUCT(c.text AS title, c.start_t AS start_t, c.end_t AS end_t)::chapter) AS chapters "
        + _FROM_VTT
        + " GROUP BY f.video[1]) TO 'out.mkv'",
        _cue_probes(),
    )
    payload = base64.b64decode(g.input_paths[2].split(",", 1)[1]).decode()
    assert g.sinks[0].chapters == 2
    assert "TIMEBASE=1/1000" in payload
    assert "START=700\nEND=1300\ntitle=Cue two." in payload


def test_a_cue_list_may_overlap_where_a_chapter_list_may_not() -> None:
    g = _lower(
        "COPY (SELECT f.video[1], "
        "ARRAY[STRUCT('a' AS text, 0 AS start_t, 6 AS end_t)::cue, STRUCT('b' AS text, 1 AS start_t, 2 AS end_t)::cue] "
        "FROM input('f.mkv') f) TO 'out.mkv'",
        _cue_probes(),
    )
    assert _vtt_payload(g, 1).count(" --> ") == 2


def test_a_cue_list_that_goes_backwards_is_rejected() -> None:
    err = _reject_lower(
        "COPY (SELECT f.video[1], "
        "ARRAY[STRUCT('a' AS text, 5 AS start_t, 6 AS end_t)::cue, STRUCT('b' AS text, 1 AS start_t, 2 AS end_t)::cue] "
        "FROM input('f.mkv') f) TO 'out.mkv'",
        _cue_probes(),
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "cue 2 starts at 1, before cue 1 at 5" in err.message


def test_a_cue_that_ends_before_it_starts_is_rejected() -> None:
    err = _reject_lower(
        "COPY (SELECT f.video[1], ARRAY[STRUCT('a' AS text, 2.5 AS start_t, 1 AS end_t)::cue] "
        "FROM input('f.mkv') f) TO 'out.mkv'",
        _cue_probes(),
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "cue 1 ends at 1, which is not after its start 2.5" in err.message


def test_a_bare_tuple_cue_record_is_rejected() -> None:
    err = _reject_lower(
        "COPY (SELECT f.video[1], ARRAY[('a', 0, 6)::cue] "
        "FROM input('f.mkv') f) TO 'out.mkv'",
        _cue_probes(),
    )
    assert (
        "a cue is written as STRUCT(... AS text, ... AS start_t, "
        "... AS end_t)::cue"
    ) in err.message


def test_a_cues_text_may_not_be_null_or_empty() -> None:
    for written in ("NULL", "''"):
        err = _reject_lower(
            f"COPY (SELECT f.video[1], ARRAY[STRUCT({written} AS text, 0 AS start_t, 1 AS end_t)::cue] "
            "FROM input('f.mkv') f) TO 'out.mkv'",
            _cue_probes(),
        )
        assert err.code is ErrorCode.UNSUPPORTED_SQL
        assert "'cue.text'" in err.message


def test_a_cues_text_may_not_hold_an_arrow_or_a_blank_line() -> None:
    err = _reject_lower(
        "COPY (SELECT f.video[1], ARRAY[STRUCT('a --> b' AS text, 0 AS start_t, 1 AS end_t)::cue] "
        "FROM input('f.mkv') f) TO 'out.mkv'",
        _cue_probes(),
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "WebVTT cannot represent inside a cue" in err.message


def test_a_cues_text_escapes_what_webvtt_reads_as_markup() -> None:
    g = _lower(
        "COPY (SELECT f.video[1], ARRAY[STRUCT('Tom & <b>' AS text, 0 AS start_t, 1 AS end_t)::cue] "
        "FROM input('f.mkv') f) TO 'out.mkv'",
        _cue_probes(),
    )
    assert "Tom &amp; &lt;b>" in _vtt_payload(g, 1)


def test_a_cues_start_must_be_a_number() -> None:
    err = _reject_lower(
        "COPY (SELECT f.video[1], ARRAY[STRUCT('a' AS text, 'x' AS start_t, 1 AS end_t)::cue] "
        "FROM input('f.mkv') f) TO 'out.mkv'",
        _cue_probes(),
    )
    assert "'cue.start_t' must be a number" in err.message


def test_a_cues_column_alias_is_not_something_a_query_can_set() -> None:
    err = _reject_lower(
        "COPY (SELECT f.video[1], STRUCT('x' AS cues) AS tags "
        "FROM input('f.mkv') f) TO 'out.mkv'",
        _cue_probes(),
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "'cues' is a probed field of the container" in err.message


def test_a_bare_cues_column_in_a_media_copy_is_a_typed_rejection() -> None:
    err = _reject_lower(
        "COPY (SELECT v.subtitle[1], v.cues FROM input('subs.en.vtt') v) TO 'o.mkv'",
        _cue_probes(),
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "'v.cues' carries no streams" in err.message
    assert "unnest(v.cues) c" in (err.hint or "")


# ---------------------------------------------------------------------------
# titled metadata tracks: a container's own cue and vector tracks, read and
# written
# ---------------------------------------------------------------------------

# One vector per row of a 2-number track, as `_vector_payload` writes them:
# little-endian f32, base64.
_TWO_VECTORS = ((0.5, -0.25), (1.0, 0.0))


def _payload(values: tuple[float, ...]) -> str:
    return base64.b64encode(struct.pack(f"<{len(values)}f", *values)).decode()


def _described_probes() -> dict[str, ProbeResult | None]:
    """A container carrying video, a titled caption track, and a vector track."""
    return {
        "f": ProbeResult(
            streams=[
                _track("video", 0),
                _track("subtitle", 0, title="speech", codec="webvtt"),
                _track(
                    "subtitle",
                    1,
                    title="clip_vectors",
                    codec="webvtt",
                    tags={"vector_dims": "2"},
                ),
            ],
            format_name="matroska,webm",
        ),
        "g": ProbeResult(streams=[_track("video", 0)], format_name="matroska,webm"),
    }


def _extracted(
    monkeypatch: pytest.MonkeyPatch, tracks: dict[int, list[CueMeta]]
) -> None:
    """Stand in for demuxing a caption track: `tracks` maps s:<index> to cues."""
    monkeypatch.setattr(
        lower_module,
        "track_cues",
        lambda spec, index, args=(): tracks.get(index, []),
    )


def _described_tracks() -> dict[int, list[CueMeta]]:
    return {
        0: [CueMeta(index=1, text="Hello", start_t=0.0, end_t=1.5)],
        1: [
            CueMeta(index=1, text=_payload(_TWO_VECTORS[0]), start_t=0.0, end_t=1.5),
            CueMeta(index=2, text=_payload(_TWO_VECTORS[1]), start_t=1.5, end_t=3.0),
        ],
    }


def test_cue_rows_read_a_containers_own_caption_tracks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The vector track is not among them: its blocks are payloads, not text."""
    _extracted(monkeypatch, _described_tracks())
    sinks = lower_table(
        resolve(parse("SELECT c.track, c.index, c.text FROM input('d.mkv') f, unnest(f.cues) c")),
        _described_probes(),
    )
    assert sinks[0].result.rows == [["speech", 1, "Hello"]]


def test_vector_rows_read_the_numbers_their_tracks_dims_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _extracted(monkeypatch, _described_tracks())
    sinks = lower_table(
        resolve(
            parse(
                "SELECT v.track, v.start_t, v.end_t, vector_length(v.vector) "
                "FROM input('d.mkv') f, unnest(f.embeddings) v"
            )
        ),
        _described_probes(),
    )
    assert sinks[0].result.rows == [
        ["clip_vectors", 0.0, 1.5, 2],
        ["clip_vectors", 1.5, 3.0, 2],
    ]


def test_a_title_subscript_reads_one_track(monkeypatch: pytest.MonkeyPatch) -> None:
    probes = _described_probes()
    result = probes["f"]
    assert result is not None
    probes["f"] = replace(
        result,
        streams=[*result.streams, _track("subtitle", 2, title="notes", codec="webvtt")],
    )
    _extracted(
        monkeypatch,
        {**_described_tracks(), 2: [CueMeta(index=1, text="Aside", start_t=0.0, end_t=1.0)]},
    )
    sinks = lower_table(
        resolve(parse("SELECT c.text FROM input('d.mkv') f, unnest(f.cues['notes']) c")),
        probes,
    )
    assert sinks[0].result.rows == [["Aside"]]


def test_a_title_no_track_carries_is_a_typed_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _extracted(monkeypatch, _described_tracks())
    err = _reject_lower_table(
        "SELECT c.text FROM input('d.mkv') f, unnest(f.cues['nope']) c",
        _described_probes(),
    )
    assert err.code is ErrorCode.STREAM_NOT_FOUND
    assert "carries no cues track titled 'nope'" in err.message
    assert "the cues tracks it carries are titled 'speech'" in (err.hint or "")


def test_a_vector_row_that_is_not_the_tracks_dims_is_a_typed_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _extracted(
        monkeypatch,
        {1: [CueMeta(index=1, text=_payload((1.0, 2.0, 3.0)), start_t=0.0, end_t=1.0)]},
    )
    err = _reject_lower_table(
        "SELECT v.track FROM input('d.mkv') f, unnest(f.embeddings) v",
        _described_probes(),
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "track 'clip_vectors' row 1 does not hold 2 numbers" in err.message


def test_a_vector_track_whose_dims_tag_is_not_a_count_is_a_typed_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _extracted(monkeypatch, _described_tracks())
    probes: dict[str, ProbeResult | None] = {
        "f": ProbeResult(
            streams=[
                _track("subtitle", 0, title="clip_vectors", tags={"vector_dims": "many"})
            ],
            format_name="matroska,webm",
        )
    }
    err = _reject_lower_table(
        "SELECT v.track FROM input('d.mkv') f, unnest(f.embeddings) v", probes
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "track 'clip_vectors' says vector_dims='many'" in err.message


def test_an_aliased_cue_array_titles_the_track_it_mints() -> None:
    g = _lower(
        "COPY (SELECT f.video[1], "
        "ARRAY[STRUCT('Hello' AS text, 0 AS start_t, 1 AS end_t)::cue] AS speech "
        "FROM input('f.mkv') f) TO 'out.mkv'",
        _cue_probes(),
    )
    assert g.outputs[1].metadata == {"title": "speech"}


def test_a_vector_track_is_written_back_from_the_rows_it_was_read_as(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The round trip: a file's own vector rows, written into a new one."""
    _extracted(monkeypatch, _described_tracks())
    g = _lower(
        "COPY (SELECT g.video[1], array_agg(STRUCT(v.start_t AS start_t, "
        "v.end_t AS end_t, v.vector AS vector)::embedding) AS clip_vectors "
        "FROM input('o.mkv') g, input('d.mkv') f, unnest(f.embeddings) v "
        "GROUP BY g.video[1]) TO 'out.mkv'",
        _described_probes(),
    )
    assert g.outputs[1].metadata == {"title": "clip_vectors", "vector_dims": "2"}
    assert _vtt_payload(g, 2) == (
        "WEBVTT\n\n"
        f"00:00:00.000 --> 00:00:01.500\n{_payload(_TWO_VECTORS[0])}\n\n"
        f"00:00:01.500 --> 00:00:03.000\n{_payload(_TWO_VECTORS[1])}\n"
    )


def test_two_vector_lengths_in_one_track_are_refused_by_both_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _extracted(
        monkeypatch,
        {
            1: [
                CueMeta(index=1, text=_payload((1.0, 2.0)), start_t=0.0, end_t=1.0),
                CueMeta(index=2, text=_payload((1.0, 2.0)), start_t=1.0, end_t=2.0),
            ],
            2: [CueMeta(index=1, text=_payload((1.0, 2.0, 3.0)), start_t=2.0, end_t=3.0)],
        },
    )
    probes = _described_probes()
    result = probes["f"]
    assert result is not None
    probes["f"] = replace(
        result,
        streams=[
            *result.streams,
            _track("subtitle", 2, title="other", tags={"vector_dims": "3"}),
        ],
    )
    err = _reject_lower(
        "COPY (SELECT g.video[1], array_agg(STRUCT(v.start_t AS start_t, "
        "v.end_t AS end_t, v.vector AS vector)::embedding) AS all_vectors "
        "FROM input('o.mkv') g, input('d.mkv') f, unnest(f.embeddings) v "
        "GROUP BY g.video[1]) TO 'out.mkv'",
        probes,
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "embedding 3 carries 3 numbers, and embedding 1 carries 2" in err.message


def test_a_titled_track_into_a_container_that_is_not_matroska_is_refused() -> None:
    err = _reject_lower(
        "COPY (SELECT f.video[1], "
        "ARRAY[STRUCT('Hello' AS text, 0 AS start_t, 1 AS end_t)::cue] AS speech "
        "FROM input('f.mkv') f) TO 'out.mp4'",
        _cue_probes(),
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "'out.mp4' is mp4, and a titled track is Matroska's" in err.message
    assert "write the file as .mkv" in (err.hint or "")


def test_a_vector_track_into_a_container_that_is_not_matroska_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _extracted(monkeypatch, _described_tracks())
    err = _reject_lower(
        "COPY (SELECT g.video[1], array_agg(STRUCT(v.start_t AS start_t, "
        "v.end_t AS end_t, v.vector AS vector)::embedding) AS clip_vectors "
        "FROM input('o.mkv') g, input('d.mkv') f, unnest(f.embeddings) v "
        "GROUP BY g.video[1]) TO 'out.mp4'",
        _described_probes(),
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "'out.mp4' is mp4, and a vector track is Matroska's" in err.message
    assert "no other container keeps its vector_dims tag" in err.message


def test_an_untitled_caption_track_still_writes_to_any_container() -> None:
    """The refusal is about what a container LOSES, not about the track."""
    g = _lower(
        "COPY (SELECT f.video[1], "
        "ARRAY[STRUCT('Hello' AS text, 0 AS start_t, 1 AS end_t)::cue] "
        "FROM input('f.mkv') f) TO 'out.mp4'",
        _cue_probes(),
    )
    assert [output.metadata for output in g.outputs] == [{}, {}]


# ---------------------------------------------------------------------------
# a built-in function call over a row column: a STRUCT field, and WHERE
# ---------------------------------------------------------------------------


def test_a_builtin_call_over_a_row_column_folds_in_a_struct_field() -> None:
    """The reproduction: upper(c.text) inside array_agg(STRUCT(...)::cue)."""
    g = _lower(
        "COPY (SELECT f.video[1], "
        "array_agg(STRUCT(upper(c.text) AS text, c.start_t AS start_t, "
        "c.end_t AS end_t)::cue) " + _FROM_VTT + " GROUP BY f.video[1]) TO 'out.mkv'",
        _cue_probes(),
    )
    assert _vtt_payload(g, 2) == (
        "WEBVTT\n\n"
        "00:00:00.000 --> 00:00:00.600\nCUE ONE.\n\n"
        "00:00:00.700 --> 00:00:01.300\nCUE TWO.\n\n"
        "00:00:01.400 --> 00:00:02.000\nCUE THREE.\n"
    )


def test_a_builtin_call_over_a_row_column_folds_in_where() -> None:
    sinks = lower_table(
        resolve(parse("SELECT c.text " + _FROM_VTT + " WHERE upper(c.text) = 'CUE TWO.'")),
        _cue_probes(),
    )
    assert sinks[0].result.rows == [["Cue two."]]


def test_a_builtin_call_over_a_column_the_row_does_not_carry_is_rejected() -> None:
    err = _reject_lower(
        "COPY (SELECT f.video[1], "
        "array_agg(STRUCT(upper(c.nope) AS text, c.start_t AS start_t, "
        "c.end_t AS end_t)::cue) " + _FROM_VTT + " GROUP BY f.video[1]) TO 'out.mkv'",
        _cue_probes(),
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "unknown column 'c.nope'" in err.message


def test_a_builtin_call_over_the_wrong_type_refuses_by_name() -> None:
    err = _reject_lower(
        "COPY (SELECT f.video[1], "
        "array_agg(STRUCT(upper(c.start_t) AS text, c.start_t AS start_t, "
        "c.end_t AS end_t)::cue) " + _FROM_VTT + " GROUP BY f.video[1]) TO 'out.mkv'",
        _cue_probes(),
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "upper() needs text, but the argument is number" in err.message


def test_a_builtin_call_over_a_non_row_value_still_folds_once() -> None:
    """A call whose argument is a literal, not a row column, works as before."""
    g = _lower(
        "COPY (SELECT f.video[1], ARRAY[STRUCT(upper('hello') AS text, 0 AS start_t, "
        "1 AS end_t)::cue] FROM input('f.mkv') f) TO 'out.mkv'",
        _cue_probes(),
    )
    assert "HELLO" in _vtt_payload(g, 1)


# ---------------------------------------------------------------------------
# attachments: a file riding inside the container
# ---------------------------------------------------------------------------


_FONT = AttachmentMeta(index=1, filename="font.ttf", mimetype="application/x-truetype-font")
_SCRIPT = AttachmentMeta(index=2, filename="notes.txt", mimetype="text/plain")


def _attachment_probes(*attachments: AttachmentMeta) -> dict[str, ProbeResult | None]:
    return {
        "f": ProbeResult(
            streams=[_track("video", 0), _track("audio", 0)],
            attachments=list(attachments),
        )
    }


_ATTACH = (
    "ARRAY[STRUCT('font.ttf' AS filename, 'application/x-truetype-font' AS mimetype, "
    "'fonts/font.ttf' AS path)::attachment] AS attachments"
)


def test_attachment_rows_read_the_declared_schema() -> None:
    sinks = lower_table(
        resolve(parse(
            "SELECT a.index, a.filename, a.mimetype "
            "FROM input('f.mkv') f, unnest(f.attachments) a"
        )),
        _attachment_probes(_FONT, _SCRIPT),
    )
    assert sinks[0].result.columns == ["index", "filename", "mimetype"]
    assert sinks[0].result.rows == [
        [1, "font.ttf", "application/x-truetype-font"],
        [2, "notes.txt", "text/plain"],
    ]


def test_a_file_with_no_attachments_reads_no_rows() -> None:
    sinks = lower_table(
        resolve(parse("SELECT a.filename FROM input('f.mkv') f, unnest(f.attachments) a")),
        _attachment_probes(),
    )
    assert sinks[0].result.rows == []


def test_an_attachment_row_has_no_path_to_read() -> None:
    """`path` is the file a written attachment reads; a container has none."""
    err = _reject_lower_table(
        "SELECT a.path FROM input('f.mkv') f, unnest(f.attachments) a",
        _attachment_probes(_FONT),
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "an attachment row exposes filename, index, mimetype" in (err.hint or "")


def test_an_attachment_row_carries_no_stream_to_select() -> None:
    err = _reject_lower(
        "COPY (SELECT a FROM input('f.mkv') f, unnest(f.attachments) a) TO 'o.mkv'",
        _attachment_probes(_FONT),
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "'a' is an attachment row, not a stream" in err.message


def test_an_attachments_literal_writes_one_attach_per_record() -> None:
    g = _lower(
        f"COPY (SELECT f.video[1], {_ATTACH} FROM input('f.mkv') f) TO 'out.mkv'",
        _attachment_probes(),
    )
    assert g.sinks[0].attachments == [
        Attachment(
            path="fonts/font.ttf",
            filename="font.ttf",
            mimetype="application/x-truetype-font",
        )
    ]
    # It is not a stream: nothing is mapped for it and no input is minted.
    assert _refs(g) == ["src:f:v:0"]
    assert len(g.input_paths) == 1


def test_array_agg_over_attachment_rows_copies_a_list_across() -> None:
    g = _lower(
        "COPY (SELECT f.video[1], "
        "array_agg(STRUCT(a.filename AS filename, a.mimetype AS mimetype, 'fonts/' || a.filename AS path)::attachment) "
        "AS attachments FROM input('f.mkv') f, unnest(f.attachments) a "
        "GROUP BY f.video[1]) TO 'out.mkv'",
        _attachment_probes(_FONT, _SCRIPT),
    )
    assert [one.path for one in g.sinks[0].attachments] == [
        "fonts/font.ttf",
        "fonts/notes.txt",
    ]


def test_a_null_attachments_column_writes_a_file_carrying_none() -> None:
    g = _lower(
        "COPY (SELECT f.video[1], NULL AS attachments FROM input('f.mkv') f) "
        "TO 'out.mkv'",
        _attachment_probes(_FONT),
    )
    assert g.sinks[0].attachments == []


def test_an_attachments_filename_and_mimetype_may_be_null() -> None:
    """ffmpeg then names the attachment after the file and guesses the type."""
    g = _lower(
        "COPY (SELECT f.video[1], "
        "ARRAY[STRUCT(NULL AS filename, NULL AS mimetype, 'fonts/font.ttf' AS path)::attachment] AS attachments "
        "FROM input('f.mkv') f) TO 'out.mkv'",
        _attachment_probes(),
    )
    assert g.sinks[0].attachments == [Attachment(path="fonts/font.ttf")]


def test_an_attachment_without_a_path_is_rejected() -> None:
    for written in ("NULL", "''", "1"):
        err = _reject_lower(
            "COPY (SELECT f.video[1], ARRAY[STRUCT('a' AS filename, "
            f"'text/plain' AS mimetype, {written} AS path)"
            "::attachment] AS attachments FROM input('f.mkv') f) TO 'out.mkv'",
            _attachment_probes(),
        )
        assert err.code is ErrorCode.UNSUPPORTED_SQL
        assert "'attachments.path' must name a file" in err.message


def test_an_attachments_filename_must_be_text_or_null() -> None:
    err = _reject_lower(
        "COPY (SELECT f.video[1], ARRAY[STRUCT(2 AS filename, 'text/plain' AS mimetype, 'x.txt' AS path)::attachment] "
        "AS attachments FROM input('f.mkv') f) TO 'out.mkv'",
        _attachment_probes(),
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "'attachments.filename' must be a string or NULL" in err.message


def test_a_positional_attachment_record_is_rejected() -> None:
    err = _reject_lower(
        "COPY (SELECT f.video[1], "
        "ARRAY[ROW('a', 'text/plain', 'x.txt')::attachment] "
        "AS attachments FROM input('f.mkv') f) TO 'out.mkv'",
        _attachment_probes(),
    )
    assert (
        "an attachment is written as STRUCT(... AS filename, ... AS mimetype, "
        "... AS path)::attachment" in err.message
    )


def test_an_empty_attachments_list_is_rejected() -> None:
    err = _reject_lower(
        "COPY (SELECT f.video[1], ARRAY[] AS attachments FROM input('f.mkv') f) "
        "TO 'out.mkv'",
        _attachment_probes(),
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "'attachments' is an empty list" in err.message
    assert "NULL AS attachments" in (err.hint or "")


def test_a_copied_attachments_column_is_a_typed_rejection() -> None:
    """The bytes ride inside the source file; ffrwd attaches files by path."""
    err = _reject_lower(
        "COPY (SELECT f.video[1], f.attachments AS attachments "
        "FROM input('f.mkv') f) TO 'out.mkv'",
        _attachment_probes(_FONT),
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "'attachments' takes an array of attachment records" in err.message


def test_an_attachments_column_in_a_cte_body_is_a_typed_rejection() -> None:
    err = _reject_lower(
        f"COPY (WITH gathered AS (SELECT f.video[1] AS v, {_ATTACH} "
        "FROM input('f.mkv') f) SELECT gathered.v FROM gathered) TO 'out.mkv'",
        _attachment_probes(),
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "a CTE body writes no file" in err.message


def test_two_branches_may_not_write_two_attachment_lists() -> None:
    err = _reject_lower(
        f"COPY (SELECT f.video[1], {_ATTACH} FROM input('f.mkv') f "
        "UNION ALL SELECT g.video[1], "
        "ARRAY[STRUCT('other.ttf' AS filename, 'font/ttf' AS mimetype, 'o.ttf' AS path)::attachment] AS attachments "
        "FROM input('f.mkv') g) TO 'out.mkv'",
        {"f": _attachment_probes()["f"], "g": _attachment_probes()["f"]},
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "'attachments' takes two different attachment lists" in err.message
def test_a_bare_chapter_column_tags_the_container_like_any_other_value() -> None:
    """A chapter row's columns feed a tags map exactly as a track row's do, so
    the BARE column is the tag value the concatenation around it already was."""
    g = _lower(
        "COPY (SELECT f.video[1], STRUCT(c.title AS title) AS tags "
        "FROM input('f.mkv') f, unnest(f.chapters) c WHERE c.index = 1) TO 'o.mkv'",
        _chapter_probes(*_TWO_CHAPTERS),
    )
    assert g.sinks[0].tags == {"title": "Intro"}


# ---------------------------------------------------------------------------
# a VALUES CTE is rejected
# ---------------------------------------------------------------------------


def test_a_values_cte_as_a_row_source_is_rejected() -> None:
    err = _reject(
        "COPY (WITH marks(start_t, end_t, title) AS "
        "(VALUES (0, 60, 'Intro'), (60, 300, 'Act One')) "
        "SELECT m.title FROM input('f.mkv') f, marks m) TO 'o.mkv'"
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "a VALUES row table is not supported" in err.message


# ---------------------------------------------------------------------------
# struct row tables: unnest(ARRAY[STRUCT(...), ...]) alias
# ---------------------------------------------------------------------------

_RUNGS = (
    "unnest(ARRAY[STRUCT(1920 AS w, '1080p' AS name), "
    "STRUCT(1280 AS w, '720p' AS name)]) r"
)

_MARKS = (
    "unnest(ARRAY[STRUCT(0 AS start_t, 60 AS end_t, 'Intro' AS title), "
    "STRUCT(60 AS start_t, 300 AS end_t, 'Act One' AS title)]) m"
)


def test_struct_row_table_prints_its_written_rows() -> None:
    sinks = lower_table(
        resolve(parse(f"SELECT r.name, r.w FROM input('f.mkv') f, {_RUNGS}")),
        {"f": ProbeResult(streams=[_track("video", 0)])},
    )
    assert sinks[0].result.columns == ["name", "w"]
    assert sinks[0].result.rows == [["1080p", 1920], ["720p", 1280]]


def test_a_struct_row_table_star_is_its_written_columns() -> None:
    sinks = lower_table(
        resolve(parse(f"SELECT m.* FROM input('f.mkv') f, {_MARKS}")),
        {"f": ProbeResult(streams=[_track("video", 0)])},
    )
    assert sinks[0].result.columns == ["start_t", "end_t", "title"]


def test_a_struct_row_table_cross_joins_track_rows() -> None:
    sinks = lower_table(
        resolve(parse(
            "SELECT t.index, m.title "
            f"FROM input('f.mkv') f, unnest(f.audio) t, {_MARKS}"
        )),
        {"f": ProbeResult(streams=[_track("audio", 0)])},
    )
    assert sinks[0].result.rows == [[1, "Intro"], [1, "Act One"]]


def test_a_struct_row_column_filters_and_sorts_like_any_row_column() -> None:
    sinks = lower_table(
        resolve(parse(
            f"SELECT m.title FROM input('f.mkv') f, {_MARKS} "
            "WHERE m.start_t >= 60 ORDER BY m.title"
        )),
        {"f": ProbeResult(streams=[_track("video", 0)])},
    )
    assert sinks[0].result.rows == [["Act One"]]


def test_a_tag_column_over_struct_rows_tags_the_container() -> None:
    """Rows with no track have nothing to tag per stream, so the file gets it."""
    g = _lower(
        "COPY (SELECT f.video[1], STRUCT('T: ' || m.name AS title) AS tags "
        "FROM input('f.mkv') f, unnest(ARRAY[STRUCT('Doc' AS name)]) m) TO 'o.mkv'",
        {"f": ProbeResult(streams=[_track("video", 0)])},
    )
    assert g.sinks[0].tags == {"title": "T: Doc"}


def test_a_bare_struct_row_column_tags_the_container_like_any_other_value() -> None:
    """Same for a written row: the field name is the key, the column the value."""
    g = _lower(
        "COPY (SELECT f.video[1], STRUCT(m.name AS title) AS tags "
        "FROM input('f.mkv') f, unnest(ARRAY[STRUCT('Doc' AS name)]) m) TO 'o.mkv'",
        {"f": ProbeResult(streams=[_track("video", 0)])},
    )
    assert g.sinks[0].tags == {"title": "Doc"}


def test_a_struct_row_is_not_an_output_stream() -> None:
    err = _reject(f"COPY (SELECT m FROM input('f.mkv') f, {_MARKS}) TO 'o.mkv'")
    assert err.code is ErrorCode.UNSUPPORTED_SQL


def test_struct_row_table_field_reads_as_a_scale_option() -> None:
    """A row's own field is a compile-time value wherever one is wanted --
    here, a non-duration numeric filter option, not just a WHERE/TO bound."""
    g = _lower(
        f"COPY (SELECT scale(f.video[1], r.w, -2) FROM input('f.mkv') f, {_RUNGS} "
        "WHERE r.name = '1080p') TO 'o.mp4'",
        {"f": _layout_probe("v")},
    )
    assert g.nodes["n1"].args["width"] == 1920


# ---------------------------------------------------------------------------
# metadata_from / strip_metadata sink options
# ---------------------------------------------------------------------------


def test_title_is_no_longer_a_sink_option() -> None:
    """Container tags are tag columns now; the option was removed."""
    err = _reject(
        "COPY (SELECT f.video[1], f.audio[1] FROM input('film.mkv') f) "
        "TO 'out.mkv' WITH (title 'Director Cut')"
    )
    assert err.code is ErrorCode.UNKNOWN_SINK_OPTION


def test_comment_is_no_longer_a_sink_option() -> None:
    err = _reject(
        "COPY (SELECT f.video[1], f.audio[1] FROM input('film.mkv') f) "
        "TO 'out.mkv' WITH (comment 'ripped')"
    )
    assert err.code is ErrorCode.UNKNOWN_SINK_OPTION


def test_a_copied_tags_map_names_its_inputs_index() -> None:
    """`f.tags ||` is what makes the copy explicit; the index is that input's."""
    g = _lower(
        "COPY (SELECT f.video[1], f.audio[1], f.tags AS tags "
        "FROM input('film.mkv') f) TO 'out.mkv'"
    )
    assert g.sinks[0].metadata == 0


def test_a_copied_tags_map_names_the_second_input() -> None:
    g = _lower(
        "COPY (SELECT f.video[1], g.audio[1], g.tags AS tags "
        "FROM input('film.mkv') f, input('extra.mkv') g) TO 'out.mkv'"
    )
    assert g.sinks[0].metadata == 1


def test_a_copied_map_layers_the_keys_written_over_it() -> None:
    g = _lower(
        "COPY (SELECT f.video[1], f.tags || STRUCT('Cut' AS title) AS tags "
        "FROM input('film.mkv') f) TO 'out.mkv'"
    )
    assert g.sinks[0].metadata == 0
    assert g.sinks[0].tags == {"title": "Cut"}


def test_an_empty_map_writes_no_globals() -> None:
    g = _lower(
        "COPY (SELECT f.video[1], f.audio[1], STRUCT() AS tags "
        "FROM input('film.mkv') f) TO 'out.mkv'"
    )
    assert g.sinks[0].metadata == NO_METADATA


def test_a_map_that_both_copies_and_empties_is_rejected() -> None:
    err = _reject(
        "COPY (SELECT f.video[1], f.tags || STRUCT() AS tags "
        "FROM input('film.mkv') f) TO 'out.mkv'"
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "both copies and writes no tags" in err.message


def test_an_unwritten_map_leaves_ffmpegs_own_copying_alone() -> None:
    g = _lower(
        "COPY (SELECT f.video[1], f.audio[1] FROM input('film.mkv') f) TO 'out.mkv'"
    )
    assert g.sinks[0].metadata is None


def test_metadata_from_is_no_longer_a_sink_option() -> None:
    """The map is a column now, and the rejection says which spelling replaced it."""
    err = _reject(
        "COPY (SELECT f.video[1], f.audio[1] FROM input('film.mkv') f) "
        "TO 'out.mkv' WITH (metadata_from f)"
    )
    assert err.code is ErrorCode.UNKNOWN_SINK_OPTION
    assert err.hint is not None and "f.tags AS tags" in err.hint


def test_strip_metadata_is_no_longer_a_sink_option() -> None:
    err = _reject(
        "COPY (SELECT f.video[1], f.audio[1] FROM input('film.mkv') f) "
        "TO 'out.mkv' WITH (strip_metadata true)"
    )
    assert err.code is ErrorCode.UNKNOWN_SINK_OPTION
    assert err.hint is not None and "STRUCT() AS tags" in err.hint


# ---------------------------------------------------------------------------
# subscript metadata WHERE assertions
# ---------------------------------------------------------------------------
#
# Same synthetic probes `_row_probes`/`_ROW_TRACKS` already builds for the
# unnest row tests: track 1 (index 0) is `eng`/2ch/stereo/aac, track 2
# (index 1) is `fra`/6ch/5.1/ac3, track 3 (index 2) carries no language tag
# and no probed bitrate -- exactly the NULL case 3VL needs.


def _assertion_query(predicate: str) -> str:
    return f"SELECT f.audio[1] FROM input('f.mkv') f WHERE {predicate}"


def test_a_true_assertion_compiles_exactly_like_no_where_at_all() -> None:
    g = _lower(_assertion_query("f.audio[1].tags.language = 'eng'"), _row_probes())
    plain = _lower("SELECT f.audio[1] FROM input('f.mkv') f", _row_probes())
    assert _outputs(g) == _outputs(plain) == [("src:f:a:0", "audio", None)]
    assert _filters(g) == []


def test_a_false_assertion_is_a_typed_rejection() -> None:
    err = _reject_lower(
        _assertion_query("f.audio[1].tags.language = 'fra'"), _row_probes()
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "WHERE assertion failed" in err.message


def test_an_unprobed_field_makes_the_assertion_unknown_not_true() -> None:
    # `bitrate` was never set on track 1: UNKNOWN is not TRUE, so this still
    # refuses to compile -- "NULL matches nothing" applies to assertions too.
    err = _reject_lower(_assertion_query("f.audio[1].bitrate > 0"), _row_probes())
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "WHERE assertion failed" in err.message


def test_is_null_reads_the_unprobed_field_correctly() -> None:
    g = _lower(_assertion_query("f.audio[1].bitrate IS NULL"), _row_probes())
    assert _outputs(g) == [("src:f:a:0", "audio", None)]


@pytest.mark.parametrize(
    ("predicate", "expected"),
    [
        ("f.audio[1].tags.language = 'eng'", True),
        ("f.audio[2].tags.language = 'eng'", False),
        ("f.audio[1].channels = 2", True),
        ("f.audio[2].channels > 2", True),
        ("f.audio[1].index = 1", True),
        ("f.audio[1].index = 2", False),
        ("f.audio[1].tags.language = 'eng' AND f.audio[1].channels = 2", True),
        ("f.audio[1].tags.language = 'eng' AND f.audio[1].channels = 6", False),
        ("f.audio[1].tags.language = 'fra' OR f.audio[2].tags.language = 'fra'", True),
        ("NOT (f.audio[1].tags.language = 'fra')", True),
        ("NOT (f.audio[1].tags.language = 'eng')", False),
        ("f.audio[1].channels BETWEEN 1 AND 3", True),
        ("f.audio[3].tags.language IS NULL", True),
        ("f.audio[1].tags.language IS NOT NULL", True),
        ("f.audio[3].tags.language IS NOT NULL", False),
    ],
)
def test_the_subscript_assertion_evaluator(predicate: str, expected: bool) -> None:
    if expected:
        g = _lower(_assertion_query(predicate), _row_probes())
        assert _outputs(g) == [("src:f:a:0", "audio", None)]
    else:
        err = _reject_lower(_assertion_query(predicate), _row_probes())
        assert err.code is ErrorCode.UNSUPPORTED_SQL


def test_assertion_subscript_out_of_range_is_stream_not_found() -> None:
    err = _reject_lower(_assertion_query("f.audio[9].tags.language = 'eng'"), _row_probes())
    assert err.code is ErrorCode.STREAM_NOT_FOUND


def test_assertion_over_an_unprobed_input_is_input_not_found() -> None:
    err = _reject_lower(_assertion_query("f.audio[1].tags.language = 'eng'"), {"f": None})
    assert err.code is ErrorCode.INPUT_NOT_FOUND


def test_a_time_window_and_an_assertion_coexist_as_separate_conjuncts() -> None:
    g = _lower(
        _assertion_query("f.t BETWEEN 1 AND 2 AND f.audio[1].tags.language = 'eng'"),
        _row_probes(),
    )
    assert _outputs(g) == [("src:f:a:0", "audio", None)]


# ---------------------------------------------------------------------------
# track-row JOINs and COALESCE fills
# ---------------------------------------------------------------------------
#
# Synthetic probes again: a join is decided entirely by the probed columns, so
# two hand-built ProbeResults exercise every join kind, the multiplicity rule
# and every fill without a file on disk. The end-to-end shapes are the
# cookbook's (recipes 25-28, exec tier).


def _pair_probes(
    left: list[StreamMeta] | None = None,
    right: list[StreamMeta] | None = None,
) -> dict[str, ProbeResult | None]:
    """Two files: `f` with an eng and a fra track, `g` with eng only."""
    return {
        "f": ProbeResult(
            streams=list(left)
            if left is not None
            else [
                _track("audio", 0, language="eng", duration=2.0),
                _track("audio", 1, language="fra", duration=2.0),
            ]
        ),
        "g": ProbeResult(
            streams=list(right)
            if right is not None
            else [_track("audio", 0, language="eng", duration=2.0)]
        ),
    }


_GATHERED = "array_agg(a), array_agg(b)"


def _join_query(
    projection: str = "a, b",
    join: str = "JOIN",
    on: str = "ON a.tags.language = b.tags.language",
    where: str = "",
    order: str = "",
    column: str = "audio",
) -> str:
    return (
        f"SELECT {projection} FROM input('f.mkv') f, input('g.mkv') g, "
        f"unnest(f.{column}) a {join} unnest(g.{column}) b {on}"
        + (f" WHERE {where}" if where else "")
        + (f" ORDER BY {order}" if order else "")
    )


def _refs(g: Graph) -> list[str]:
    return [output.ref for output in g.outputs]


def test_an_inner_join_pairs_rows_by_their_metadata() -> None:
    g = _lower(_join_query(), _pair_probes())
    # One result row (eng/eng); fra matched nothing, so its stream is never read.
    assert _refs(g) == ["src:f:a:0", "src:g:a:0"]


def test_result_row_order_is_the_left_sides_track_order() -> None:
    probes = _pair_probes(
        right=[
            _track("audio", 0, language="fra", duration=2.0),
            _track("audio", 1, language="eng", duration=2.0),
        ]
    )
    g = _lower(_join_query(_GATHERED), probes)
    # `g` stores fra first, but the rows follow `f`: eng, then fra. (Outputs are
    # column-major -- `a`'s gather, then `b`'s -- so the PAIRING is
    # element k of one against element k of the other.)
    assert _refs(g) == ["src:f:a:0", "src:f:a:1", "src:g:a:1", "src:g:a:0"]


def test_join_multiplicity_is_real_join_semantics() -> None:
    """A row matching two rows on the other side pairs with BOTH -- real join
    multiplicity, not a duplicate-tag rejection. Two pairs, two outputs."""
    probes = _pair_probes(
        right=[
            _track("audio", 0, language="eng", channel_layout="stereo", duration=2.0),
            _track("audio", 1, language="eng", channel_layout="5.1", duration=2.0),
        ]
    )
    g = _lower(_join_query(_GATHERED), probes)
    assert _refs(g) == ["src:f:a:0", "src:f:a:0", "src:g:a:0", "src:g:a:1"]
    # ...and the fix, when that is not what was wanted, is a wider key -- which
    # here matches nothing at all (`f`'s tracks carry no layout), and an empty
    # row set selects no streams, exactly as it does without a join.
    err = _reject_lower(
        _join_query(on="ON a.tags.language = b.tags.language AND a.channel_layout = b.channel_layout"),
        probes,
    )
    assert err.code is ErrorCode.STREAM_NOT_FOUND
    assert "selects nothing" in err.message


def test_a_null_key_matches_nothing() -> None:
    probes = _pair_probes(
        left=[_track("audio", 0, duration=2.0)],  # no language tag at all
        right=[_track("audio", 0, duration=2.0)],
    )
    g = _lower(_join_query(projection="a", join="LEFT JOIN"), probes)
    err = _reject_lower(_join_query(projection="b", join="LEFT JOIN"), probes)
    assert _refs(g) == ["src:f:a:0"]  # the left row survives, unpaired
    assert err.code is ErrorCode.STREAM_NOT_FOUND


def test_a_left_join_keeps_unmatched_left_rows() -> None:
    g = _lower(
        _join_query(projection="array_agg(a)", join="LEFT JOIN"), _pair_probes()
    )
    assert _refs(g) == ["src:f:a:0", "src:f:a:1"]


def test_a_data_rows_null_row_hint_names_the_join_not_a_fill() -> None:
    """`_FILL_SPELLINGS` has no 'data' entry: the NULL-track rejection must
    still be the typed one (a KeyError here would surface as INTERNAL,
    guardrail #7) and its hint must steer to a tighter join, since nothing
    can stand in for a missing data track."""
    probes = _pair_probes(
        left=[
            _track("data", 0, language="eng"),
            _track("data", 1, language="fra"),
        ],
        right=[_track("data", 0, language="eng")],
    )
    err = _reject_lower(
        _join_query(projection="b", join="FULL OUTER JOIN", column="data"),
        probes,
    )
    assert err.code is ErrorCode.STREAM_NOT_FOUND
    assert "data tracks have no fill" in (err.hint or "")
    assert "INNER or LEFT join" in (err.hint or "")


def test_a_full_join_appends_unmatched_right_rows_in_their_own_order() -> None:
    probes = _pair_probes(
        right=[
            _track("audio", 0, language="deu", duration=2.0),
            _track("audio", 1, language="eng", duration=2.0),
        ]
    )
    # Rows are eng (matched), fra (unmatched left), then the unmatched RIGHT
    # row -- whose `a` is NULL, which is what this rejection is about.
    err = _reject_lower(
        _join_query(projection="a", join="FULL OUTER JOIN"), probes
    )
    assert err.code is ErrorCode.STREAM_NOT_FOUND
    assert "is NULL in row 3" in err.message
    assert "b.tags.language='deu'" in err.message
    assert "COALESCE(a" in (err.hint or "")
    # Filled, the whole row order shows: the matched pair, the unmatched LEFT
    # row (silence on the right), then the unmatched RIGHT row appended last.
    filled = _lower(
        _join_query(
            projection="array_agg(COALESCE(b, ffmpeg.anullsrc(duration => 1)))",
            join="FULL OUTER JOIN",
        ),
        probes,
    )
    fills = [node.id for node in filled.nodes.values()]
    assert _refs(filled) == ["src:g:a:1", fills[0], "src:g:a:0"]


def test_full_join_with_and_without_the_outer_keyword_are_one_join() -> None:
    probes = _pair_probes()
    with_kind = _lower(
        _join_query(projection="array_agg(a)", join="FULL OUTER JOIN"), probes
    )
    without = _lower(
        _join_query(projection="array_agg(a)", join="FULL JOIN"), probes
    )
    assert _refs(with_kind) == _refs(without) == ["src:f:a:0", "src:f:a:1"]


def test_a_comma_between_two_unnests_is_the_cross_join() -> None:
    g = _lower(
        "SELECT array_agg(a), array_agg(b) FROM input('f.mkv') f, "
        "input('g.mkv') g, unnest(f.audio) a, unnest(g.audio) b",
        _pair_probes(
            right=[
                _track("audio", 0, language="eng", duration=2.0),
                _track("audio", 1, language="fra", duration=2.0),
            ]
        ),
    )
    assert _refs(g) == [
        "src:f:a:0", "src:f:a:0", "src:f:a:1", "src:f:a:1",
        "src:g:a:0", "src:g:a:1", "src:g:a:0", "src:g:a:1",
    ]


def test_where_filters_the_joined_rows_not_the_tables() -> None:
    """A predicate on the nullable side runs AFTER the join, so `IS NULL` finds
    the gaps -- pushing it down would have turned the outer join into an inner
    one and dropped exactly the rows it is about."""
    g = _lower(
        _join_query(
            projection="a", join="FULL OUTER JOIN", where="b.tags.language IS NULL"
        ),
        _pair_probes(),
    )
    assert _refs(g) == ["src:f:a:1"]  # only the unpaired French row


def test_order_by_re_sorts_the_joined_rows_and_keeps_the_pairing() -> None:
    g = _lower(_join_query(_GATHERED, order="a.tags.language DESC"), _pair_probes(
        right=[
            _track("audio", 0, language="eng", duration=2.0),
            _track("audio", 1, language="fra", duration=2.0),
        ]
    ))
    assert _refs(g) == ["src:f:a:1", "src:f:a:0", "src:g:a:1", "src:g:a:0"]


def test_limit_narrows_the_joined_relation_and_keeps_the_pairing() -> None:
    # The same sorted relation as above, cut to its top PAIR: both sides of
    # tuple one survive, both sides of tuple two are never read.
    g = _lower(
        _join_query("a, b", order="a.tags.language DESC") + " LIMIT 1",
        _pair_probes(
            right=[
                _track("audio", 0, language="eng", duration=2.0),
                _track("audio", 1, language="fra", duration=2.0),
            ]
        ),
    )
    assert _refs(g) == ["src:f:a:1", "src:g:a:1"]


# -- COALESCE fills ---------------------------------------------------------


def _fill_query(fill: str, projection: str = "COALESCE(b, {fill})") -> str:
    return _join_query(
        projection=f"array_agg({projection.format(fill=fill)})", join="FULL OUTER JOIN"
    )


def test_an_audio_fill_inherits_only_the_paired_rows_duration() -> None:
    g = _lower(_fill_query("ffmpeg.anullsrc()"), _pair_probes())
    fills = [node for node in g.nodes.values() if node.filter == "anullsrc"]
    assert _refs(g) == ["src:g:a:0", fills[0].id]
    # Duration inherited from the fra row it stands beside -- and NOTHING else:
    # no sample_rate, no channel_layout the query never wrote.
    assert fills[0].args == {"duration": 2.0}
    assert fills[0].inputs == []


def test_an_explicit_fill_option_wins_over_the_inherited_one() -> None:
    g = _lower(_fill_query("ffmpeg.anullsrc(duration => 5)"), _pair_probes())
    fills = [node for node in g.nodes.values() if node.filter == "anullsrc"]
    assert fills[0].args == {"duration": 5}


def test_a_fill_with_no_duration_to_inherit_is_a_typed_rejection() -> None:
    probes = _pair_probes(
        left=[
            _track("audio", 0, language="eng"),
            _track("audio", 1, language="fra"),  # never probed for a duration
        ]
    )
    err = _reject_lower(_fill_query("ffmpeg.anullsrc()"), probes)
    assert err.code is ErrorCode.UDF_ARG_TYPE
    assert "no duration to stand in for" in err.message
    assert "duration => 2" in (err.hint or "")


def test_a_fill_takes_the_paired_rows_tags_as_its_provenance() -> None:
    g = _lower(
        _join_query(
            projection="array_agg(amix(a, COALESCE(b, ffmpeg.anullsrc())))",
            join="FULL OUTER JOIN",
        ),
        _pair_probes(),
    )
    # The silence-filled mix keeps the French tag: it came from the side that
    # existed, so `_agreed_source` still sees two streams saying `fra`.
    assert [output.metadata for output in g.outputs] == [
        {"language": "eng"},
        {"language": "fra"},
    ]


def test_a_join_with_no_gaps_mints_no_fill_at_all() -> None:
    probes = _pair_probes(
        right=[
            _track("audio", 0, language="eng", duration=2.0),
            _track("audio", 1, language="fra", duration=2.0),
        ]
    )
    g = _lower(_fill_query("ffmpeg.anullsrc()"), probes)
    assert _refs(g) == ["src:g:a:0", "src:g:a:1"]
    assert not g.nodes  # consume-once: nothing needed a stand-in


def test_a_video_fill_inherits_size_rate_and_duration() -> None:
    probes: dict[str, ProbeResult | None] = {
        "f": ProbeResult(
            streams=[
                _track("video", 0, language="eng", width=320, height=240,
                       fps="25/1", duration=2.0),
                _track("video", 1, language="fra", width=640, height=480,
                       fps="30/1", duration=3.0),
            ]
        ),
        "g": ProbeResult(
            streams=[_track("video", 0, language="eng", width=320, height=240,
                            fps="25/1", duration=2.0)]
        ),
    }
    g = _lower(
        _join_query(
            projection="array_agg(COALESCE(b, ffmpeg.color()))",
            join="FULL OUTER JOIN",
            column="video",
        ),
        probes,
    )
    fills = [node for node in g.nodes.values() if node.filter == "color"]
    assert fills[0].args == {"size": "640x480", "rate": "30/1", "duration": 3.0}


def test_a_fill_of_the_wrong_type_for_the_column_is_udf_arg_type() -> None:
    err = _reject_lower(_fill_query("ffmpeg.color()"), _pair_probes())
    assert err.code is ErrorCode.UDF_ARG_TYPE
    assert "generates a video stream, but 'b' is audio" in err.message
    assert "anullsrc" in (err.hint or "")


def test_a_caption_gap_fills_with_an_empty_captions_input() -> None:
    probes: dict[str, ProbeResult | None] = {
        "f": ProbeResult(
            streams=[
                _track("subtitle", 0, language="eng"),
                _track("subtitle", 1, language="fra"),
            ]
        ),
        "g": ProbeResult(streams=[_track("subtitle", 0, language="eng")]),
    }
    g = _lower(
        _join_query(
            projection="array_agg(COALESCE(b, ffrwd.empty_captions()))",
            join="FULL OUTER JOIN",
            column="subtitle",
        ),
        probes,
    )
    minted = "ffrwd.empty_captions#3"
    # An INPUT, not a filter node: a filtergraph carries no subtitle pads.
    assert not g.nodes
    assert g.input_paths[2] == "data:text/vtt;base64,V0VCVlRUCgo="
    assert g.sources[minted] == 2
    assert g.input_options[minted] == {"format": "webvtt"}
    assert _refs(g) == ["src:g:s:0", f"src:{minted}:s:0"]
    # ...and it takes the paired row's tag, which is the whole point of it.
    assert [output.metadata for output in g.outputs] == [
        {"language": "eng"},
        {"language": "fra"},
    ]


def test_the_empty_captions_input_renders_its_format_flag_before_the_i() -> None:
    probes: dict[str, ProbeResult | None] = {
        "f": ProbeResult(
            streams=[
                _track("subtitle", 0, language="eng"),
                _track("subtitle", 1, language="fra"),
            ]
        ),
        "g": ProbeResult(streams=[_track("subtitle", 0, language="eng")]),
    }
    g = _lower(
        _join_query(
            projection="array_agg(COALESCE(b, ffrwd.empty_captions()))",
            join="FULL OUTER JOIN",
            column="subtitle",
        ),
        probes,
    )
    args = build_ffmpeg_args(emit(g), "out.mkv")
    assert args[args.index("-f") + 1] == "webvtt"
    assert args[args.index("-f") + 2] == "-i"
    assert args[args.index("-f") + 3] == "data:text/vtt;base64,V0VCVlRUCgo="


def test_a_coalesce_fill_must_be_a_generated_stand_in() -> None:
    err = _reject_lower(_fill_query("f.audio[1]"), _pair_probes())
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "a COALESCE fill is a generated stand-in" in err.message


def test_coalesces_first_argument_is_a_row_stream() -> None:
    err = _reject_lower(
        _join_query(
            projection="COALESCE(f.audio[1], ffmpeg.anullsrc())",
            join="FULL OUTER JOIN",
        ),
        _pair_probes(),
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "first argument is a track-row stream column" in err.message


def test_coalesce_takes_exactly_two_arguments() -> None:
    err = _reject_lower(
        _join_query(
            projection="COALESCE(b, ffmpeg.anullsrc(), ffmpeg.anullsrc())",
            join="FULL OUTER JOIN",
        ),
        _pair_probes(),
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "takes a track column and one fill" in err.message


@pytest.mark.exec
def test_a_joined_track_query_runs_end_to_end(tmp_path: Path) -> None:
    """Real probes, real ffmpeg: the pairwise mix of recipe 25's shape, run.

    av2.mp4 and av3.mp4 each carry an `eng` and a `fra` track in a different
    order, so this proves the join wired eng to eng -- and that the compiled
    command is one ffmpeg actually accepts.
    """
    out = tmp_path / "mixed.mka"
    query = (
        "SELECT array_agg(amix(a, b)) FROM "
        f"input('{(FIXTURES_DIR / 'av2.mp4').as_posix()}') f, "
        f"input('{(FIXTURES_DIR / 'av3.mp4').as_posix()}') g, "
        "unnest(f.audio) a JOIN unnest(g.audio) b ON a.tags.language = b.tags.language"
    )
    args = build_ffmpeg_args(emit(compile_sql(query)), str(out))
    args.insert(1, "-y")
    result = subprocess.run(args, capture_output=True, text=True, timeout=60.0)
    assert result.returncode == 0, result.stderr
    assert out.exists()


@pytest.mark.exec
def test_an_empty_captions_fill_muxes_a_real_track(tmp_path: Path) -> None:
    """The measured claim, re-measured: `-f webvtt -i "data:..."` really does
    mux a taggable, zero-cue subtitle stream (2026-08-17, ffmpeg 7.1)."""
    out = tmp_path / "subs.mkv"
    query = (
        "SELECT s, ffrwd.empty_captions() FROM "
        f"input('{(FIXTURES_DIR / 'avs.mkv').as_posix()}') f, "
        "unnest(f.subtitle) s WHERE s.tags.language = 'eng'"
    )
    args = build_ffmpeg_args(emit(compile_sql(query)), str(out))
    args.insert(1, "-y")
    result = subprocess.run(args, capture_output=True, text=True, timeout=60.0)
    assert result.returncode == 0, result.stderr
    assert out.exists()



@pytest.mark.exec
def test_a_written_cue_track_muxes_and_reads_back(tmp_path: Path) -> None:
    """The whole round trip against a real ffmpeg: a chapter list written as a
    WebVTT track, muxed, then read back out as cues."""
    out = tmp_path / "with-cues.mkv"
    query = (
        "COPY (SELECT f.video[1], "
        "array_agg(STRUCT(c.title AS text, c.start_t AS start_t, c.end_t AS end_t)::cue) "
        f"FROM input('{(FIXTURES_DIR / 'av-chapters.mkv').as_posix()}') f, "
        "unnest(f.chapters) c GROUP BY f.video[1]) TO 'x.mkv'"
    )
    args = build_ffmpeg_args(emit(compile_sql(query)), str(out))
    args.insert(1, "-y")
    result = subprocess.run(args, capture_output=True, text=True, timeout=60.0)
    assert result.returncode == 0, result.stderr

    written = tmp_path / "cues.vtt"
    extract = build_ffmpeg_args(
        emit(compile_sql(f"SELECT f.subtitle[1] FROM input('{out.as_posix()}') f")),
        str(written),
    )
    extract.insert(1, "-y")
    assert subprocess.run(extract, capture_output=True, timeout=60.0).returncode == 0
    cues = parse_webvtt(written.read_text(encoding="utf-8"))
    assert [(c.text, c.start_t, c.end_t) for c in cues] == [
        ("Intro", 0.0, 1.0),
        ("Chapter 1", 1.0, 2.0),
        ("Chapter 2", 2.0, 3.0),
        ("Credits", 3.0, 4.0),
    ]


@pytest.mark.exec
def test_a_chapter_list_built_from_cues_keeps_its_fractional_bounds(
    tmp_path: Path,
) -> None:
    """ffmetadata counts START/END in whole units of its timebase, so a
    sub-second cue bound needs the millisecond one to survive the trip."""
    out = tmp_path / "chaptered.mkv"
    query = (
        "COPY (SELECT f.video[1], "
        "array_agg(STRUCT(c.text AS title, c.start_t AS start_t, c.end_t AS end_t)::chapter) AS chapters "
        f"FROM input('{(FIXTURES_DIR / 'av2.mp4').as_posix()}') f, "
        f"input('{(FIXTURES_DIR / 'subs.en.vtt').as_posix()}') v, "
        "unnest(v.cues) c GROUP BY f.video[1]) TO 'x.mkv'"
    )
    args = build_ffmpeg_args(emit(compile_sql(query)), str(out))
    args.insert(1, "-y")
    result = subprocess.run(args, capture_output=True, text=True, timeout=60.0)
    assert result.returncode == 0, result.stderr
    probed = probe(str(out))
    assert probed is not None
    assert [(c.title, c.start_t, c.end_t) for c in probed.chapters] == [
        ("Cue one.", 0.0, 0.6),
        ("Cue two.", 0.7, 1.3),
        ("Cue three.", 1.4, 2.0),
    ]

# ---------------------------------------------------------------------------
# metadata tag columns: CASE, ||, row-scoped overrides
# ---------------------------------------------------------------------------
#
# Synthetic probes again: a tag's value is computed from probed columns and
# lands in `Output.metadata`, so nothing here needs a file. The end-to-end
# shapes are the cookbook's (recipes 37-38, exec tier).


def _tag_query(fields: str, projection: str = "t", column: str = "audio") -> str:
    """Tag every row, then gather the rows into one file.

    `fields` is the body of the tags map, so a caller writes just the
    ``<value> AS <key>`` pairs. Two scopes, and the tags need the inner one:
    rows are tracks inside the CTE body, so the map is per stream there,
    while the outer SELECT aggregates the body's streams into the single file
    this writes. The tags ride the streams across the boundary, which is what
    these tests check.
    """
    return (
        "WITH tagged AS ("
        f"SELECT {projection} AS track, STRUCT({fields}) AS tags "
        f"FROM input('f.mkv') f, unnest(f.{column}) t"
        ") SELECT array_agg(tagged.track) FROM tagged"
    )


def test_a_tag_column_replaces_the_provenance_value() -> None:
    g = _lower(_tag_query("'zxx' AS language"), _row_probes())
    assert [o.metadata for o in g.outputs] == [{"language": "zxx"}] * 3


def test_a_tag_column_adds_a_key_provenance_never_had() -> None:
    g = _lower(_tag_query("'Main' AS title"), _row_probes())
    assert [o.metadata for o in g.outputs] == [
        {"language": "eng", "title": "Main"},
        {"language": "fra", "title": "Main"},
        {"title": "Main"},
    ]


def test_a_null_tag_clears_the_key_and_leaves_the_rest() -> None:
    g = _lower(
        _tag_query("NULL AS language, 'Main' AS title"),
        _row_probes(),
    )
    assert [o.metadata for o in g.outputs] == [{"title": "Main"}] * 3


def test_an_unselected_tag_passes_through_unchanged() -> None:
    probes = _row_probes(_track("audio", 0, language="eng", title="Commentary"))
    g = _lower(_tag_query("'zxx' AS language"), probes)
    assert [o.metadata for o in g.outputs] == [
        {"language": "zxx", "title": "Commentary"}
    ]


def test_a_tag_column_produces_no_output_stream() -> None:
    g = _lower(_tag_query("'Main' AS title"), _row_probes())
    assert _outputs(g) == [
        ("src:f:a:0", "audio", None),
        ("src:f:a:1", "audio", None),
        ("src:f:a:2", "audio", None),
    ]


def test_a_searched_case_takes_the_first_true_branch_per_row() -> None:
    g = _lower(
        _tag_query(
            "CASE WHEN t.tags.language = 'fra' THEN 'fre' "
            "WHEN t.channels = 2 THEN 'two' ELSE 'other' END AS language"
        ),
        _row_probes(),
    )
    assert [o.metadata for o in g.outputs] == [
        {"language": "two"},
        {"language": "fre"},
        {"language": "two"},
    ]


def test_a_simple_case_compares_its_operand_with_each_when() -> None:
    g = _lower(
        _tag_query(
            "CASE t.tags.language WHEN 'fra' THEN 'fre' WHEN 'eng' THEN 'en' "
            "ELSE 'und' END AS language"
        ),
        _row_probes(),
    )
    assert [o.metadata for o in g.outputs] == [
        {"language": "en"},
        {"language": "fre"},
        # The third track has no language tag: `NULL = 'fra'` is UNKNOWN, not
        # TRUE, so no WHEN matches and ELSE answers.
        {"language": "und"},
    ]


def test_an_unknown_case_condition_is_not_true() -> None:
    """3VL straight through CASE: the untagged track's comparison is UNKNOWN,
    so its branch is skipped exactly as a FALSE one is."""
    g = _lower(
        _tag_query("CASE WHEN t.tags.language != 'fra' THEN 'kept' ELSE 'else' END AS title"),
        _row_probes(),
    )
    assert [o.metadata["title"] for o in g.outputs] == ["kept", "else", "else"]


def test_a_case_with_no_else_falls_through_to_null() -> None:
    g = _lower(
        _tag_query("CASE WHEN t.tags.language = 'fra' THEN 'fre' END AS language"),
        _row_probes(),
    )
    assert [o.metadata for o in g.outputs] == [{}, {"language": "fre"}, {}]


def test_concatenation_builds_a_value_from_the_row() -> None:
    g = _lower(_tag_query("'Audio (' || t.tags.language || ')' AS title"), _row_probes())
    assert [o.metadata.get("title") for o in g.outputs] == [
        "Audio (eng)",
        "Audio (fra)",
        # NULL propagates: the untagged track's title is NULL, which CLEARS the
        # key rather than writing "Audio ()".
        None,
    ]


def test_a_tag_key_is_free_form() -> None:
    g = _lower(_tag_query("'2026' AS date"), _row_probes(_track("audio", 0)))
    assert [o.metadata for o in g.outputs] == [{"date": "2026"}]


def test_a_tag_key_folds_like_any_identifier() -> None:
    """Unquoted -> lowercase, quoted -> verbatim: Postgres's own rule, so the
    key a container ends up carrying is the one the alias spells."""
    g = _lower(
        _tag_query("""'x' AS Title, 'y' AS "Sort Name\""""),
        _row_probes(_track("audio", 0)),
    )
    assert [o.metadata for o in g.outputs] == [{"title": "x", "Sort Name": "y"}]


def test_a_tag_is_row_scoped_across_every_track_the_row_carries() -> None:
    """One result row, two tracks (a video and an audio one crossed): the tag
    the row computes lands on both."""
    probes = _row_probes(_track("video", 0), _track("audio", 0, language="eng"))
    g = _lower(
        "SELECT v, a, STRUCT('Feature' AS title) AS tags FROM input('f.mkv') f, "
        "unnest(f.video) v, unnest(f.audio) a",
        probes,
    )
    assert _refs(g) == ["src:f:v:0", "src:f:a:0"]
    assert [o.metadata for o in g.outputs] == [
        {"title": "Feature"},
        {"language": "eng", "title": "Feature"},
    ]


def test_a_joined_row_tags_one_sides_track_from_the_others_column() -> None:
    probes = _pair_probes(
        left=[_track("audio", 0, language="eng"), _track("audio", 1, language="fra")],
        right=[
            _track("audio", 0, language="eng", title="English"),
            _track("audio", 1, language="fra", title="French"),
        ],
    )
    g = _lower(
        "WITH titled AS ("
        "  SELECT a AS track, STRUCT(b.tags.title AS title) AS tags"
        "  FROM input('f.mkv') f, input('g.mkv') g,"
        "       unnest(f.audio) a JOIN unnest(g.audio) b ON a.tags.language = b.tags.language"
        ") SELECT array_agg(titled.track) FROM titled",
        probes,
    )
    assert _refs(g) == ["src:f:a:0", "src:f:a:1"]
    assert [o.metadata for o in g.outputs] == [
        {"language": "eng", "title": "English"},
        {"language": "fra", "title": "French"},
    ]


def test_one_track_cannot_take_two_values_for_the_same_tag() -> None:
    """A cross join repeats the left track once per right row; two different
    values for its tag is a rejection, not a last-one-wins guess."""
    probes = _pair_probes(
        left=[_track("audio", 0, language="eng")],
        right=[
            _track("audio", 0, language="eng"),
            _track("audio", 1, language="fra"),
        ],
    )
    err = _reject_lower(
        "SELECT a, STRUCT(b.tags.language AS language) AS tags "
        "FROM input('f.mkv') f, "
        "input('g.mkv') g, unnest(f.audio) a, unnest(g.audio) b",
        probes,
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "two different values" in err.message


def test_a_tag_survives_a_filter_that_threads_provenance() -> None:
    g = _lower(
        _tag_query("'Loud' AS title", projection="volume(t, 2.0)"),
        _row_probes(_track("audio", 0, language="eng")),
    )
    assert [o.metadata for o in g.outputs] == [{"language": "eng", "title": "Loud"}]


def test_a_query_of_nothing_but_tags_selects_no_stream() -> None:
    err = _reject_lower(
        "SELECT STRUCT('Main' AS title) AS tags "
        "FROM input('f.mkv') f, unnest(f.audio) t",
        _row_probes(),
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "selects no stream" in err.message


def test_a_media_query_with_no_row_tables_tags_the_container() -> None:
    g = _lower(
        "SELECT f.audio[1], STRUCT('Main' AS title) AS tags FROM input('f.mkv') f"
    )
    assert g.sinks[0].tags == {"title": "Main"}
    assert [o.metadata for o in g.outputs] == [{}]


def test_an_unaliased_value_column_is_still_not_a_stream() -> None:
    err = _reject(
        "SELECT f.audio[1], 'a' || 'b' FROM input('f.mkv') f"
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "every SELECT column must be a stream expression" in err.message
    assert err.hint is not None and "give it an alias" in err.hint


def test_a_case_column_over_literals_is_still_not_a_row_predicate() -> None:
    err = _reject(
        "SELECT f.audio[1], "
        "STRUCT(CASE WHEN 'a' = 'a' THEN 'x' END AS title) AS tags "
        "FROM input('f.mkv') f"
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL


# ---------------------------------------------------------------------------
# read-only fields: a tags FIELD that names one is a rejection, not a tag key
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("column", "field"),
    [
        ("STRUCT('h264' AS codec) AS tags", "codec"),
        ("STRUCT(3 AS index) AS tags", "index"),
    ],
)
def test_setting_a_read_only_track_field_is_rejected(column: str, field: str) -> None:
    err = _reject_lower(
        f"SELECT t, {column} FROM input('f.mkv') f, unnest(f.audio) t "
        "WHERE t.index = 1",
        _row_probes(),
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert f"'{field}' is a probed field of the track row" in err.message


def test_setting_a_read_only_field_of_the_rows_own_type_is_rejected() -> None:
    """The row's type decides: `width` is a video field, so a video row rejects."""
    err = _reject_lower(
        "SELECT t, STRUCT(1920 AS width) AS tags "
        "FROM input('f.mkv') f, unnest(f.video) t",
        _row_probes(),
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "'width' is a probed field of the track row" in err.message


def test_a_free_form_key_is_still_a_tag() -> None:
    g = _lower(
        "SELECT t, STRUCT('x' AS whatever) AS tags "
        "FROM input('f.mkv') f, unnest(f.audio) t "
        "WHERE t.index = 1",
        _row_probes(),
    )
    assert all(o.metadata.get("whatever") == "x" for o in g.outputs)


def test_setting_a_read_only_container_field_is_rejected() -> None:
    err = _reject(
        "SELECT f.audio[1], STRUCT(12 AS duration) AS tags FROM input('f.mkv') f"
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "'duration' is a probed field of the container" in err.message


def test_a_read_only_field_is_rejected_inside_a_cte_body_too() -> None:
    err = _reject_lower(
        "WITH c AS (SELECT t, STRUCT('h264' AS codec) AS tags "
        "FROM input('f.mkv') f, unnest(f.audio) t) "
        "SELECT array_agg(c.t) FROM c",
        _row_probes(),
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "'codec' is a probed field of the track row" in err.message


def test_disposition_is_writable_so_its_own_name_still_works() -> None:
    g = _lower(
        "SELECT t, 'forced' AS disposition "
        "FROM input('f.mkv') f, unnest(f.audio) t WHERE t.index = 1",
        _row_probes(),
    )
    assert [o.disposition for o in g.outputs] == [("forced",)]


# ---------------------------------------------------------------------------
# container tags: <input>.<key> read, aliased columns written
# ---------------------------------------------------------------------------


def _tagged_probes(**tags: str) -> dict[str, ProbeResult | None]:
    """One probed input `f` -- a video and an audio track -- carrying `tags`."""
    return {
        "f": ProbeResult(
            streams=[_track("video", 0), _track("audio", 0)], tags=dict(tags)
        )
    }


def _container_query(fields: str) -> str:
    """A container-level tags map over one input row; `fields` is its body."""
    return f"SELECT f.video[1], STRUCT({fields}) AS tags FROM input('f.mkv') f"


def test_a_container_tag_column_reads_the_probed_value() -> None:
    g = _lower(
        _container_query("f.tags.title AS title"), _tagged_probes(title="Angel One")
    )
    assert g.sinks[0].tags == {"title": "Angel One"}


def test_a_container_tag_concatenates_like_any_text_value() -> None:
    g = _lower(
        _container_query("f.tags.title || ' (restored)' AS title"),
        _tagged_probes(title="Angel One"),
    )
    assert g.sinks[0].tags == {"title": "Angel One (restored)"}


def test_an_absent_container_tag_reads_null_so_case_can_fill_it() -> None:
    g = _lower(
        _container_query(
            "CASE WHEN f.tags.comment IS NULL THEN 'no notes' ELSE f.tags.comment END AS comment"
        ),
        _tagged_probes(title="Angel One"),
    )
    assert g.sinks[0].tags == {"comment": "no notes"}


def test_a_present_container_tag_wins_the_case_fill() -> None:
    g = _lower(
        _container_query(
            "CASE WHEN f.tags.comment IS NULL THEN 'no notes' ELSE f.tags.comment END AS comment"
        ),
        _tagged_probes(comment="ripped"),
    )
    assert g.sinks[0].tags == {"comment": "ripped"}


def test_a_container_tag_on_an_unprobed_input_is_input_not_found() -> None:
    err = _reject_lower(_container_query("f.tags.title AS title"), {"f": None})
    assert err.code is ErrorCode.INPUT_NOT_FOUND
    assert "'f.tags.title' is unknown" in err.message


def test_a_container_tag_in_stream_position_is_rejected() -> None:
    err = _reject_lower(
        "SELECT f.tags.title FROM input('f.mkv') f", _tagged_probes(title="Angel One")
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "'f.tags.title' is a text tag, not a stream" in err.message
    assert "give it an alias" in (err.hint or "")


def test_an_unknown_input_column_still_names_the_structural_columns() -> None:
    err = _reject("SELECT f.bogus FROM input('f.mkv') f")
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    hint = err.hint or ""
    for column in ("video", "audio", "subtitle", "data", "t", "duration"):
        assert column in hint
    assert "container tags" in hint


def test_a_literal_container_tag_needs_no_probe_at_all() -> None:
    g = _lower(_container_query("'Remastered 2026' AS title"))
    assert g.sinks[0].tags == {"title": "Remastered 2026"}


def test_a_null_container_tag_column_clears_the_key() -> None:
    g = _lower(_container_query("NULL AS artist"))
    assert g.sinks[0].tags == {"artist": None}


def test_a_container_tag_key_is_free_form() -> None:
    g = _lower(_container_query("""'ffrwd' AS "encoded_by\""""))
    assert g.sinks[0].tags == {"encoded_by": "ffrwd"}


def test_a_number_container_tag_is_spelled_out() -> None:
    g = _lower(_container_query("2026 AS date"))
    assert g.sinks[0].tags == {"date": "2026"}


def test_a_container_has_no_disposition() -> None:
    err = _reject(
        "SELECT f.video[1], 'default' AS disposition FROM input('f.mkv') f"
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "'disposition' is a stream field, not a container one" in err.message


def test_a_map_names_each_key_once() -> None:
    """A repeated field is a mistake whatever the values: one entry per key."""
    err = _reject(_container_query("'a' AS title, 'b' AS title"))
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "names the field 'title' twice" in err.message


def test_a_repeated_key_is_rejected_even_with_the_same_value() -> None:
    err = _reject(_container_query("'a' AS title, 'a' AS title"))
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "names the field 'title' twice" in err.message


def test_two_tags_columns_cannot_disagree_on_one_container_key() -> None:
    err = _reject(
        "SELECT f.video[1], STRUCT('a' AS title) AS tags, "
        "STRUCT('b' AS title) AS tags FROM input('f.mkv') f"
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "two different values" in err.message


def test_a_row_table_branch_still_tags_per_stream() -> None:
    """The container-tag branch is the NO-row-table one; row queries are
    untouched."""
    g = _lower(
        _tag_query("'Main' AS title"), _row_probes(_track("audio", 0, language="eng"))
    )
    assert g.sinks[0].tags == {}
    assert [o.metadata for o in g.outputs] == [{"language": "eng", "title": "Main"}]


def test_a_row_table_branch_reads_container_tags_onto_its_streams() -> None:
    """`f.tags.title` is a value wherever the value grammar runs; with track rows
    it lands on each row's stream, not on the container."""
    probes = {
        "f": ProbeResult(
            streams=[_track("audio", 0, language="eng")], tags={"artist": "Docs Dept"}
        )
    }
    g = _lower(_tag_query("f.tags.artist AS artist"), probes)
    assert g.sinks[0].tags == {}
    assert [o.metadata for o in g.outputs] == [
        {"language": "eng", "artist": "Docs Dept"}
    ]


def test_two_copys_get_their_own_container_tags() -> None:
    g = _lower(
        "COPY (SELECT f.video[1], STRUCT('one' AS title) AS tags "
        "FROM input('f.mkv') f) TO 'a.mkv'; "
        "COPY (SELECT g.video[1], STRUCT('two' AS title) AS tags "
        "FROM input('g.mkv') g) TO 'b.mkv'"
    )
    assert [unit.tags for unit in g.sinks] == [{"title": "one"}, {"title": "two"}]


def test_container_tags_survive_the_ir_round_trip() -> None:
    g = _lower(_container_query("'Cut' AS title, NULL AS artist"))
    restored = Graph.from_dict(g.to_dict())
    assert restored.sinks[0].tags == {"title": "Cut", "artist": None}


def test_an_unaliased_row_metadata_column_is_still_not_a_stream() -> None:
    err = _reject_lower(
        "SELECT t, t.tags.language FROM input('f.mkv') f, unnest(f.audio) t",
        _row_probes(),
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "is track metadata, not a stream" in err.message
    assert "alias" in (err.hint or "")


def test_where_accepts_a_case_operand() -> None:
    g = _lower(
        _row_query("t.tags.language = CASE WHEN t.channels = 6 THEN 'fra' ELSE 'eng' END"),
        _row_probes(),
    )
    assert _refs(g) == ["src:f:a:0", "src:f:a:1"]


def test_where_accepts_a_concatenation_operand() -> None:
    g = _lower(_row_query("'x' || t.tags.language = 'xfra'"), _row_probes())
    assert _refs(g) == ["src:f:a:1"]


def test_an_on_predicate_accepts_a_case_operand() -> None:
    g = _lower(
        _join_query(
            on="ON a.tags.language = CASE WHEN b.channels = 2 THEN b.tags.language ELSE 'zxx' END"
        ),
        _pair_probes(right=[_track("audio", 0, language="eng", channels=2)]),
    )
    assert _refs(g) == ["src:f:a:0", "src:g:a:0"]


def test_concatenating_a_number_is_rejected_rather_than_coerced() -> None:
    err = _reject(_tag_query("'ch' || t.channels AS title"))
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "'||' joins text" in err.message
    assert "cast the number with ::text" in (err.hint or "")


def test_case_results_must_share_one_type() -> None:
    err = _reject(
        _tag_query("CASE WHEN t.tags.language = 'fra' THEN 'fre' ELSE 2 END AS language")
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "CASE results must share one type" in err.message


def test_a_simple_case_types_its_whens_against_the_operand() -> None:
    err = _reject(_tag_query("CASE t.tags.language WHEN 2 THEN 'x' END AS language"))
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "one type" in err.message


def test_a_case_condition_is_type_checked_like_any_row_predicate() -> None:
    err = _reject(_tag_query("CASE WHEN t.channels = 'two' THEN 'x' END AS title"))
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "is number" in err.message


@pytest.mark.exec
def test_a_written_tag_reads_back_out_of_the_file(tmp_path: Path) -> None:
    """The claim end to end: the compiled command really does stamp the tag,
    and ffprobe reads it back off the muxed stream."""
    out = tmp_path / "tagged.mka"
    query = (
        "SELECT t, STRUCT('Audio (' || t.tags.language || ')' AS title) AS tags "
        f"FROM input('{(FIXTURES_DIR / 'av-eng.mp4').as_posix()}') f, "
        "unnest(f.audio) t"
    )
    args = build_ffmpeg_args(emit(compile_sql(query)), str(out))
    args.insert(1, "-y")
    result = subprocess.run(args, capture_output=True, text=True, timeout=60.0)
    assert result.returncode == 0, result.stderr
    probed = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "a:0",
            "-show_entries", "stream_tags=title,language",
            "-of", "default=noprint_wrappers=1", str(out),
        ],
        capture_output=True,
        text=True,
        timeout=60.0,
    )
    assert probed.returncode == 0, probed.stderr
    assert "TAG:title=Audio (eng)" in probed.stdout
    assert "TAG:language=eng" in probed.stdout


def test_a_table_query_prints_what_a_tag_would_say() -> None:
    """The same value expression, printed instead of written: how you check a
    retag before running it."""
    sinks = lower_table(
        resolve(
            parse(
                "SELECT t.index, CASE WHEN t.tags.language = 'fra' THEN 'fre' "
                "ELSE t.tags.language END AS lang, 'A (' || t.tags.language || ')' AS title "
                "FROM input('f.mkv') f, unnest(f.audio) t"
            )
        ),
        _row_probes(),
    )
    assert sinks[0].result.rows == [
        [1, "eng", "A (eng)"],
        [2, "fre", "A (fra)"],
        [3, None, None],
    ]


def test_a_table_query_prints_container_tags_beside_the_duration() -> None:
    sinks = lower_table(
        resolve(parse("SELECT f.tags.title, f.tags.artist, f.tags.comment, f.duration "
                      "FROM input('f.mkv') f")),
        {
            "f": ProbeResult(
                streams=[_track("video", 0)],
                duration=2.0,
                tags={"title": "Angel One", "artist": "Docs Dept"},
            )
        },
    )
    assert sinks[0].result.columns == ["title", "artist", "comment", "duration"]
    assert sinks[0].result.rows == [["Angel One", "Docs Dept", None, 2.0]]


def test_a_table_query_broadcasts_a_container_tag_over_track_rows() -> None:
    sinks = lower_table(
        resolve(
            parse(
                "SELECT t.index, f.tags.title FROM input('f.mkv') f, unnest(f.audio) t"
            )
        ),
        {"f": ProbeResult(streams=list(_ROW_TRACKS), tags={"title": "Angel One"})},
    )
    assert sinks[0].result.rows == [
        [1, "Angel One"],
        [2, "Angel One"],
        [3, "Angel One"],
    ]


# ---------------------------------------------------------------------------
# arrays as table cells: a bare input array column, with and without rows
# ---------------------------------------------------------------------------
#
# `f.audio` etc. have no scalar table-cell representation, so a bare selection
# of one prints its WHOLE array as one Postgres-style array cell (braces even
# for one element), broadcasting like a subscript already does when track
# rows are also in the branch.


def test_a_rowless_bare_array_column_prints_every_element() -> None:
    """No row relation at all -- used to keep only the array's first stream."""
    sinks = lower_table(
        resolve(parse("SELECT f.audio FROM input('f.mkv') f")), _row_probes()
    )
    assert sinks[0].result.columns == ["audio"]
    assert sinks[0].result.rows == [
        [
            ArrayCell(
                elements=(
                    StreamCell(type="audio", spec="0:a:0"),
                    StreamCell(type="audio", spec="0:a:1"),
                    StreamCell(type="audio", spec="0:a:2"),
                )
            )
        ]
    ]


def test_csv_rowless_bare_array_column_matches_the_table_cell_text() -> None:
    sinks = lower_table(
        resolve(parse("SELECT f.video FROM input('f.mkv') f")), _row_probes()
    )
    assert render_csv(sinks[0].result, header=False) == "{<video 0:v:0>}\n"


def test_a_bare_array_column_broadcasts_beside_a_row_relation() -> None:
    """A bare array column alongside unnested rows used to panic on the
    cardinality mismatch; it now broadcasts one array cell over every row."""
    sinks = lower_table(
        resolve(
            parse("SELECT f.video, t.tags.language FROM input('f.mkv') f, unnest(f.audio) t")
        ),
        _row_probes(),
    )
    video_cell = ArrayCell(elements=(StreamCell(type="video", spec="0:v:0"),))
    assert sinks[0].result.rows == [
        [video_cell, "eng"],
        [video_cell, "fra"],
        [video_cell, None],
    ]


def test_csv_bare_array_column_broadcasts_beside_a_row_relation() -> None:
    sinks = lower_table(
        resolve(
            parse("SELECT f.video, t.tags.language FROM input('f.mkv') f, unnest(f.audio) t")
        ),
        _row_probes(),
    )
    assert render_csv(sinks[0].result, header=False) == (
        "{<video 0:v:0>},eng\n{<video 0:v:0>},fra\n{<video 0:v:0>},\n"
    )


def test_a_filtered_bare_array_broadcasts_beside_a_row_relation() -> None:
    """Fuzz find: the bare array through a FILTER took the row-set path, so
    its one filtered stream was indexed once per chapter row and panicked.
    A call over a bare array is not a row set -- it broadcasts, cell for cell
    with the bare column above."""
    sinks = lower_table(
        resolve(
            parse(
                "SELECT hflip(f.video), t.tags.language "
                "FROM input('f.mkv') f, unnest(f.audio) t"
            )
        ),
        _row_probes(),
        registry=_snapshot_registry(),
    )
    video_cell = ArrayCell(elements=(StreamCell(type="video", spec="n1"),))
    assert sinks[0].result.rows == [
        [video_cell, "eng"],
        [video_cell, "fra"],
        [video_cell, None],
    ]


def test_a_filtered_bare_array_keeps_every_element_in_its_one_cell() -> None:
    """The rowless case of the same column: both filtered tracks land in the
    one array cell instead of the second being dropped."""
    sinks = lower_table(
        resolve(parse("SELECT hflip(f.video) FROM input('f.mkv') f")),
        _row_probes(_track("video", 0), _track("video", 1)),
        registry=_snapshot_registry(),
    )
    assert sinks[0].result.rows == [
        [
            ArrayCell(
                elements=(
                    StreamCell(type="video", spec="n1"),
                    StreamCell(type="video", spec="n2"),
                )
            )
        ]
    ]


def test_a_subscripted_array_column_still_keeps_its_plain_cell() -> None:
    """The one array-typed shape that stays a plain stream cell: a subscript
    already picks ONE element, so there is nothing to brace."""
    sinks = lower_table(
        resolve(parse("SELECT f.audio[1] FROM input('f.mkv') f")), _row_probes()
    )
    assert sinks[0].result.rows == [[StreamCell(type="audio", spec="0:a:0")]]


# ---------------------------------------------------------------------------
# tag columns in a CTE body: per-stream tags that ride into the outer query
# ---------------------------------------------------------------------------
#
# The two levels, in one query text: inside a WITH rows are tracks, so a tag
# column tags streams; outside it the same column tags the container (the
# section above). A CTE's tags are recorded once, when its body lowers, and
# layered UNDER whatever the sink's own branch sets.


_TAGGED_CTE = (
    "WITH tagged AS ("
    "  SELECT t AS track,"
    "         STRUCT('Audio (' || t.tags.language || ')' AS title) AS tags"
    "  FROM input('f.mkv') f, unnest(f.audio) t"
    ") "
)


def _shared_probes() -> dict[str, ProbeResult | None]:
    """Two aliases over ONE file, the way ``compile_sql`` probes them: the same
    ProbeResult object, so both see the same StreamMeta identities -- which is
    what lets an outer row table name the very tracks a CTE tagged."""
    result = ProbeResult(streams=list(_ROW_TRACKS))
    return {"f": result, "g": result}


def test_a_tag_column_in_a_cte_body_tags_that_rows_streams() -> None:
    """The bare column splats, each stream carrying the tag its row computed."""
    g = _lower(_TAGGED_CTE + "SELECT array_agg(tagged.track) FROM tagged", _row_probes())
    assert [o.metadata for o in g.outputs] == [
        {"language": "eng", "title": "Audio (eng)"},
        {"language": "fra", "title": "Audio (fra)"},
        # The third track has no language: the `||` is NULL, which clears the key.
        {},
    ]


def test_a_cte_tag_survives_a_subscript_of_the_column() -> None:
    g = _lower(
        _TAGGED_CTE + "SELECT array_agg(tagged.track[2]) FROM tagged", _row_probes()
    )
    assert [o.metadata for o in g.outputs] == [
        {"language": "fra", "title": "Audio (fra)"}
    ]


def test_a_cte_tag_survives_a_filter_in_the_outer_query() -> None:
    g = _lower(
        _TAGGED_CTE + "SELECT array_agg(volume(tagged.track[1], 0.5)) FROM tagged",
        _row_probes(),
    )
    assert [o.metadata for o in g.outputs] == [
        {"language": "eng", "title": "Audio (eng)"}
    ]


def test_a_null_cte_tag_clears_the_key_as_it_does_in_a_sink() -> None:
    g = _lower(
        "WITH tagged AS ("
        "  SELECT t AS track, STRUCT(NULL AS language) AS tags"
        "  FROM input('f.mkv') f, unnest(f.audio) t"
        ") SELECT array_agg(tagged.track) FROM tagged",
        _row_probes(),
    )
    assert [o.metadata for o in g.outputs] == [{}, {}, {}]


def test_the_sinks_own_tag_wins_over_the_ctes_on_the_same_key() -> None:
    """Inner then outer is LAYERING, not the disagreement `_record_tag` rejects
    -- that check stays inside one query.

    Both sides narrow to one row, which is what keeps the outer tag a
    PER-STREAM one: an aggregate would tag the container instead.
    """
    g = _lower(
        "WITH tagged AS ("
        "  SELECT t AS track,"
        "         STRUCT('Audio (' || t.tags.language || ')' AS title) AS tags"
        "  FROM input('f.mkv') f, unnest(f.audio) t WHERE t.index = 1"
        ") SELECT tagged.track, STRUCT('Outer' AS title) AS tags "
        "FROM tagged, input('f.mkv') g, unnest(g.audio) u WHERE u.index = 1",
        _shared_probes(),
    )
    assert [o.metadata.get("title") for o in g.outputs] == ["Outer"]


def test_two_sinks_over_one_tagged_view_each_carry_its_tags() -> None:
    g = _lower(
        "CREATE VIEW tagged AS"
        "  SELECT t AS track, STRUCT('Inner' AS title) AS tags"
        "  FROM input('f.mkv') f, unnest(f.audio) t;"
        "COPY (SELECT array_agg(tagged.track) FROM tagged) TO 'a.mka';"
        "COPY (SELECT array_agg(tagged.track) FROM tagged) TO 'b.mka';",
        _row_probes(),
    )
    assert [unit.path for unit in g.sinks] == ["a.mka", "b.mka"]
    assert [[o.metadata.get("title") for o in unit.outputs] for unit in g.sinks] == [
        ["Inner"] * 3,
        ["Inner"] * 3,
    ]


def test_two_cte_bodies_cannot_tag_one_track_two_ways() -> None:
    """Unlike two COPYs, the CTE bodies of one script pour into a single
    carry-over dict, so a disagreement between them has no representation."""
    err = _reject_lower(
        "WITH one AS ("
        "  SELECT t AS track, STRUCT('One' AS title) AS tags"
        "  FROM input('f.mkv') f, unnest(f.audio) t"
        "), two AS ("
        "  SELECT u AS track, STRUCT('Two' AS title) AS tags"
        "  FROM input('g.mkv') g, unnest(g.audio) u"
        ") SELECT one.track FROM one",
        _shared_probes(),
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "two different values on the same track" in err.message


def test_a_cte_body_scalar_becomes_a_value_column_of_its_rows() -> None:
    """The body names it, the alias exposes it, a table query prints it."""
    result = lower_table(
        resolve(
            parse(
                "WITH x AS ("
                "  SELECT t AS track, t.index AS n"
                "  FROM input('f.mkv') f, unnest(f.audio) t"
                ") SELECT x.n FROM x"
            )
        ),
        _row_probes(),
    )[0]
    assert result.result.columns == ["n"]
    assert [row[0] for row in result.result.rows] == [1, 2, 3]


def test_a_value_column_selected_at_a_media_sink_is_rejected() -> None:
    """A SELECT column of a media query is an output stream; the hint says
    where a value belongs instead."""
    err = _reject_lower(
        "COPY (WITH x AS ("
        "  SELECT t AS track, t.index AS n"
        "  FROM input('f.mkv') f, unnest(f.audio) t WHERE t.index = 1"
        ") SELECT x.track, x.n FROM x) TO 'out.mka'",
        _row_probes(),
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "'x.n' is a value" in err.message
    assert err.hint is not None and "TO expression" in err.hint


def test_a_value_column_the_compiler_cannot_evaluate_names_itself() -> None:
    """Nothing probed a filter's output, so a value read off one has no value
    to carry; the rejection names the read."""
    err = _reject_lower(
        "COPY (WITH x AS ("
        "  SELECT v AS frame, scale(v, 640, -2).width AS w"
        "  FROM input('f.mkv') f, unnest(f.video) v WHERE v.index = 1"
        ") SELECT x.frame FROM x) TO ('o' || x.w::text || '.mkv')",
        _row_probes(_track("video", 0)),
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "'.width' reads a field of the output of scale()" in err.message
    assert err.hint is not None


def test_a_value_column_filters_the_rows_it_sits_on() -> None:
    g = _lower(
        "COPY (WITH x AS ("
        "  SELECT t AS track, t.tags.language AS lang"
        "  FROM input('f.mkv') f, unnest(f.audio) t"
        ") SELECT x.track FROM x WHERE x.lang = 'fra') TO 'out.mka'",
        _row_probes(),
    )
    assert _refs(g) == ["src:f:a:1"]


def test_a_value_column_reaches_a_further_cte() -> None:
    units = _lower(
        "COPY (WITH x AS ("
        "  SELECT t AS track, t.index AS n"
        "  FROM input('f.mkv') f, unnest(f.audio) t"
        "), y AS (SELECT x.track AS track, x.n AS n2 FROM x) "
        "SELECT y.track FROM y) TO ('y' || y.n2::text || '.mka')",
        _row_probes(),
    ).sinks
    assert [unit.path for unit in units] == ["y1.mka", "y2.mka", "y3.mka"]


def test_a_cte_body_with_no_track_rows_cannot_tag_a_container() -> None:
    """A CTE writes no file, so there is no container for a rowless tag column
    to name."""
    err = _reject(
        "WITH tagged AS (SELECT f.audio[1] AS snd, STRUCT('Main' AS title) AS tags "
        "FROM input('f.mkv') f) SELECT tagged.snd FROM tagged"
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "a 'tags' column in a CTE body has no track row to tag" in err.message
    assert err.hint is not None and "outer SELECT" in err.hint


def test_the_two_level_query_tags_the_streams_and_the_container() -> None:
    """Recipe 53's shape: per-stream titles from the WITH, the file's title
    from the outer SELECT, one query."""
    g = _lower(
        _TAGGED_CTE + "SELECT g.video, array_agg(tagged.track), "
        "STRUCT('Director Cut' AS title) AS tags "
        "FROM input('f.mkv') g, tagged GROUP BY g.video",
        _shared_probes(),
    )
    assert g.sinks[0].tags == {"title": "Director Cut"}
    assert [o.metadata for o in g.outputs] == [
        {},
        {"language": "eng", "title": "Audio (eng)"},
        {"language": "fra", "title": "Audio (fra)"},
        {},
    ]


def test_a_table_query_over_a_tagged_cte_prints_the_same_rows() -> None:
    """A table query has no ffmpeg output for a tag to land on: the CTE's tag
    column is accepted and changes nothing that gets printed."""
    plain = (
        "WITH tagged AS ("
        "  SELECT t AS track FROM input('f.mkv') f, unnest(f.audio) t"
        ") SELECT tagged.track FROM tagged"
    )
    tagged = lower_table(
        resolve(parse(_TAGGED_CTE + "SELECT tagged.track FROM tagged")), _row_probes()
    )[0]
    untagged = lower_table(resolve(parse(plain)), _row_probes())[0]
    assert tagged.result.columns == untagged.result.columns
    assert list(tagged.result.rows) == list(untagged.result.rows)
    # Not just mutually equal -- both sides really do carry every row: this
    # comparison used to pass by both being wrongly truncated to one.
    assert len(untagged.result.rows) == 3
    assert untagged.result.rows == [
        [StreamCell(type="audio", spec="0:a:0")],
        [StreamCell(type="audio", spec="0:a:1")],
        [StreamCell(type="audio", spec="0:a:2")],
    ]


# ---------------------------------------------------------------------------
# a CTE's own array column in a table query -- one row per element
# ---------------------------------------------------------------------------
#
# `FROM aud` binds no row relation of its own (a CTE is not `unnest`), so a
# table branch used to fall back to cardinality 1 unconditionally and a
# splat CTE column's later elements were silently dropped. The row count of
# a CTE-only FROM has to come from the CTE's own splat array column instead.


def _cte_report_query(where: str = "t.tags.language = 'eng'", *, gather: bool = False) -> str:
    column = "array_agg(aud.track)" if gather else "aud.track"
    return (
        "WITH aud AS ("
        "  SELECT t AS track FROM input('f.mkv') f, unnest(f.audio) t"
        + (f" WHERE {where}" if where else "")
        + f") SELECT {column} FROM aud"
    )


def test_a_cte_array_column_prints_one_row_per_element() -> None:
    """The maintainer's exact report shape: a WHERE-filtered CTE array column
    used to print its first row and stop."""
    sinks = lower_table(
        resolve(parse(_cte_report_query())), {"f": ProbeResult(streams=_LANG_TRACKS)}
    )
    assert sinks[0].result.columns == ["track"]
    assert sinks[0].result.rows == [
        [StreamCell(type="audio", spec="0:a:0")],
        [StreamCell(type="audio", spec="0:a:1")],
    ]


def test_csv_cte_array_column_prints_one_row_per_element() -> None:
    sinks = lower_table(
        resolve(parse(_cte_report_query())), {"f": ProbeResult(streams=_LANG_TRACKS)}
    )
    assert render_csv(sinks[0].result, header=False) == "<audio 0:a:0>\n<audio 0:a:1>\n"


def test_a_media_copy_of_the_report_query_still_maps_both_streams() -> None:
    """The media path this bug never touched, kept as a regression test."""
    g = _lower(
        _cte_report_query(gather=True), {"f": ProbeResult(streams=_LANG_TRACKS)}
    )
    assert [o.ref for o in g.outputs] == ["src:f:a:0", "src:f:a:1"]


def test_a_cte_column_broadcasts_an_input_scalar_beside_it() -> None:
    """Input-level scalars beside a CTE array column repeat per row, exactly
    as they do beside an ordinary row relation."""
    query = (
        "WITH aud AS ("
        "  SELECT t AS track FROM input('f.mkv') f, unnest(f.audio) t"
        ") SELECT aud.track, g.duration FROM aud, input('f.mkv') g"
    )
    sinks = lower_table(
        resolve(parse(query)),
        {
            "f": ProbeResult(streams=_LANG_TRACKS),
            "g": ProbeResult(streams=_LANG_TRACKS, duration=12.5),
        },
    )
    assert sinks[0].result.rows == [
        [StreamCell(type="audio", spec="0:a:0"), 12.5],
        [StreamCell(type="audio", spec="0:a:1"), 12.5],
        [StreamCell(type="audio", spec="0:a:2"), 12.5],
    ]


def test_a_subscripted_cte_column_broadcasts_over_the_ctes_rows() -> None:
    """`aud.track[1]`-style: the subscript picked one element, so the value is
    the same in every row -- but `FROM aud` still has as many rows as the body
    produced, and each of them prints it."""
    query = (
        "WITH aud AS ("
        "  SELECT t AS track FROM input('f.mkv') f, unnest(f.audio) t"
        ") SELECT aud.track[1] FROM aud"
    )
    sinks = lower_table(resolve(parse(query)), {"f": ProbeResult(streams=_LANG_TRACKS)})
    assert sinks[0].result.rows == [[StreamCell(type="audio", spec="0:a:0")]] * 3


def test_two_columns_from_the_same_cte_stay_aligned() -> None:
    """One CTE body, one row count: two of its own splat columns line up row
    for row without any extra bookkeeping."""
    query = (
        "WITH aud AS ("
        "  SELECT t AS track, v AS frame FROM input('f.mkv') f, "
        "  unnest(f.audio) t, unnest(f.video) v"
        ") SELECT aud.track, aud.frame FROM aud"
    )
    probes = {"f": ProbeResult(streams=[_track("video", 0), *_LANG_TRACKS])}
    sinks = lower_table(resolve(parse(query)), probes)
    assert sinks[0].result.rows == [
        [StreamCell(type="audio", spec="0:a:0"), StreamCell(type="video", spec="0:v:0")],
        [StreamCell(type="audio", spec="0:a:1"), StreamCell(type="video", spec="0:v:0")],
        [StreamCell(type="audio", spec="0:a:2"), StreamCell(type="video", spec="0:v:0")],
    ]


def _reject_lower_table(sql: str, probes: dict[str, ProbeResult | None]) -> FfrwdError:
    with pytest.raises(FfrwdError) as excinfo:
        lower_table(resolve(parse(sql)), probes)
    return _anchored(excinfo.value)


def test_two_ctes_cross_join_with_real_multiplicity() -> None:
    """A comma between two CTEs is a cross join: three audio rows beside two
    video rows print six, each value repeated as often as it occurs."""
    query = (
        "WITH a1 AS ("
        "  SELECT t AS track FROM input('f.mkv') f, unnest(f.audio) t"
        "), a2 AS ("
        "  SELECT v AS track FROM input('g.mkv') g, unnest(g.video) v"
        ") SELECT a1.track, a2.track FROM a1, a2"
    )
    sinks = lower_table(
        resolve(parse(query)),
        {
            "f": ProbeResult(streams=_LANG_TRACKS),
            "g": ProbeResult(streams=[_track("video", 0), _track("video", 1)]),
        },
    )
    assert sinks[0].result.rows == [
        [StreamCell(type="audio", spec=f"0:a:{a}"), StreamCell(type="video", spec=f"1:v:{v}")]
        for a in range(3)
        for v in range(2)
    ]


def test_one_cte_row_crossed_with_two_repeats_the_single_value() -> None:
    """The 1 x 2 shape the table has to print honestly: two rows, the video
    visible in both."""
    query = (
        "WITH vid AS ("
        "  SELECT v AS track FROM input('f.mkv') f, unnest(f.video) v"
        "), aud AS ("
        "  SELECT t AS track FROM input('f.mkv') f2, unnest(f2.audio) t"
        ") SELECT vid.track, aud.track FROM vid, aud"
    )
    probes = ProbeResult(streams=[_track("video", 0), *_LANG_TRACKS[:2]])
    sinks = lower_table(resolve(parse(query)), {"f": probes, "f2": probes})
    assert sinks[0].result.rows == [
        [StreamCell(type="video", spec="0:v:0"), StreamCell(type="audio", spec="0:a:0")],
        [StreamCell(type="video", spec="0:v:0"), StreamCell(type="audio", spec="0:a:1")],
    ]


def test_a_grouped_cross_join_prints_one_row_with_an_array_cell() -> None:
    """Recipe 57's table form: group by the video, gather the audio."""
    query = (
        "WITH vid AS ("
        "  SELECT v AS track FROM input('f.mkv') f, unnest(f.video) v"
        "), aud AS ("
        "  SELECT t AS track FROM input('f.mkv') f2, unnest(f2.audio) t"
        ") SELECT vid.track, array_agg(aud.track) FROM vid, aud GROUP BY vid.track"
    )
    probes = ProbeResult(streams=[_track("video", 0), *_LANG_TRACKS[:2]])
    sinks = lower_table(resolve(parse(query)), {"f": probes, "f2": probes})
    assert sinks[0].result.rows == [
        [
            StreamCell(type="video", spec="0:v:0"),
            ArrayCell(
                elements=(
                    StreamCell(type="audio", spec="0:a:0"),
                    StreamCell(type="audio", spec="0:a:1"),
                )
            ),
        ]
    ]


def test_a_grouped_cross_join_prints_one_row_per_key() -> None:
    """Two videos crossed with two audio rows: one printed row per video, each
    gathering the whole audio set."""
    query = (
        "WITH vid AS ("
        "  SELECT v AS track FROM input('f.mkv') f, unnest(f.video) v"
        "), aud AS ("
        "  SELECT t AS track FROM input('f.mkv') f2, unnest(f2.audio) t"
        ") SELECT vid.track, array_agg(aud.track) FROM vid, aud GROUP BY vid.track"
    )
    probes = ProbeResult(
        streams=[_track("video", 0), _track("video", 1), *_LANG_TRACKS[:2]]
    )
    sinks = lower_table(resolve(parse(query)), {"f": probes, "f2": probes})
    gathered = ArrayCell(
        elements=(
            StreamCell(type="audio", spec="0:a:0"),
            StreamCell(type="audio", spec="0:a:1"),
        )
    )
    assert sinks[0].result.rows == [
        [StreamCell(type="video", spec="0:v:0"), gathered],
        [StreamCell(type="video", spec="0:v:1"), gathered],
    ]


def test_a_cte_beside_an_unnest_relation_is_a_cross_join() -> None:
    """Two independent row sets used to be rejected outright; the comma
    between them is an ordinary cross join, so the table prints all six."""
    query = (
        "WITH aud AS ("
        "  SELECT t AS track FROM input('f.mkv') f, unnest(f.audio) t"
        ") SELECT aud.track, v FROM aud, input('g.mkv') g, unnest(g.video) v"
    )
    sinks = lower_table(
        resolve(parse(query)),
        {
            "f": ProbeResult(streams=_LANG_TRACKS),
            "g": ProbeResult(streams=[_track("video", 0), _track("video", 1)]),
        },
    )
    assert sinks[0].result.rows == [
        [StreamCell(type="audio", spec=f"0:a:{a}"), StreamCell(type="video", spec=f"1:v:{v}")]
        for a in range(3)
        for v in range(2)
    ]


def test_a_cte_beside_an_unnest_relation_maps_the_cross_join() -> None:
    """The same shape as a media query: every pair of the cross join is an
    output, in relation order."""
    g = _lower(
        "WITH aud AS ("
        "  SELECT t AS track FROM input('f.mkv') f, unnest(f.audio) t"
        ") SELECT array_agg(aud.track), array_agg(v) "
        "FROM aud, input('g.mkv') g, unnest(g.video) v",
        {
            "f": ProbeResult(streams=_LANG_TRACKS),
            "g": ProbeResult(streams=[_track("video", 0), _track("video", 1)]),
        },
    )
    assert [o.ref for o in g.outputs] == [
        "src:f:a:0", "src:f:a:0", "src:f:a:1",
        "src:f:a:1", "src:f:a:2", "src:f:a:2",
        "src:g:v:0", "src:g:v:1", "src:g:v:0",
        "src:g:v:1", "src:g:v:0", "src:g:v:1",
    ]


_VID_AUD_CTES = (
    "WITH vid AS ("
    "  SELECT v AS track FROM input('f.mkv') f, unnest(f.video) v"
    "), aud AS ("
    "  SELECT t AS track FROM input('f.mkv') f2, unnest(f2.audio) t"
    ") "
)


def _vid_aud_probes(videos: int = 1) -> dict[str, ProbeResult | None]:
    result = ProbeResult(
        streams=[_track("video", k) for k in range(videos)] + list(_LANG_TRACKS[:2])
    )
    return {"f": result, "f2": result}


def test_an_ungrouped_cross_join_is_rejected_against_one_file() -> None:
    """One video crossed with two audio rows is a two-row relation, and two
    rows do not fit in one file: the duplication is named, not written."""
    err = _reject_lower(
        _VID_AUD_CTES + "SELECT vid.track, aud.track FROM vid, aud", _vid_aud_probes()
    )
    assert err.code is ErrorCode.ROW_COUNT_MISMATCH
    assert "this query has 2 rows" in err.message
    assert "array_agg" in (err.hint or "")


def test_a_grouped_cross_join_maps_the_key_once() -> None:
    """Recipe 57's media form: GROUP BY the video and gather the audio, and
    the video is mapped once with both audio tracks after it."""
    g = _lower(
        _VID_AUD_CTES
        + "SELECT vid.track, array_agg(aud.track) FROM vid, aud GROUP BY vid.track",
        _vid_aud_probes(),
    )
    assert [o.ref for o in g.outputs] == [
        "src:f:v:0",
        "src:f2:a:0",
        "src:f2:a:1",
    ]


def test_a_grouped_cross_join_of_two_groups_needs_a_file_each() -> None:
    """Two videos are two groups, and a group is a file: the same rule the row
    count follows, counted over groups once the branch aggregates."""
    err = _reject_lower(
        _VID_AUD_CTES
        + "SELECT vid.track, array_agg(aud.track) FROM vid, aud GROUP BY vid.track",
        _vid_aud_probes(videos=2),
    )
    assert err.code is ErrorCode.ROW_COUNT_MISMATCH
    assert "this query has 2 groups" in err.message
    assert "TO (" in (err.hint or "")


def test_a_single_cte_source_still_gathers_its_rows_in_order() -> None:
    """The one-source shape is untouched: `FROM aud` is three rows, and an
    ungrouped COPY maps them in row order, as it always did."""
    g = _lower(
        "WITH aud AS ("
        "  SELECT t AS track FROM input('f.mkv') f, unnest(f.audio) t"
        ") SELECT array_agg(aud.track) FROM aud",
        {"f": ProbeResult(streams=_LANG_TRACKS)},
    )
    assert [o.ref for o in g.outputs] == ["src:f:a:0", "src:f:a:1", "src:f:a:2"]


def test_order_by_a_cte_column_has_no_streaming_equivalent() -> None:
    """A CTE's columns are streams, and streams have no order to sort by, so
    ORDER BY over a CTE source is rejected rather than silently ignored."""
    err = _reject(
        "WITH aud AS ("
        "  SELECT t AS track FROM input('f.mkv') f, unnest(f.audio) t"
        ") SELECT aud.track FROM aud ORDER BY aud.track"
    )
    assert err.code is ErrorCode.NO_STREAMING_EQUIVALENT
    assert "ORDER BY" in err.message


# ---------------------------------------------------------------------------
# compile-time arithmetic, ::text and <input>.duration
# ---------------------------------------------------------------------------


def _values(columns: str, probes: dict[str, ProbeResult | None] | None = None) -> list[list[object]]:
    """A table query's rows, so a computed value can be read back directly."""
    sql = f"SELECT {columns} FROM input('f.mkv') f, unnest(f.audio) t"
    sinks = lower_table(resolve(parse(sql)), probes or _row_probes())
    return [list(row) for row in sinks[0].result.rows]


def _duration_probes(duration: float | None) -> dict[str, ProbeResult | None]:
    return {"f": ProbeResult(streams=list(_ROW_TRACKS), duration=duration)}


def _trim(where: str, probes: dict[str, ProbeResult | None]) -> object:
    g = _lower(f"SELECT f.video[1] FROM input('f.mkv') f WHERE {where}", probes)
    return g.input_trims["f"]


def test_int_division_truncates_the_way_postgres_does() -> None:
    assert _values("t.channels / 4 AS q") == [[0], [1], [0]]


def test_int_division_truncates_toward_zero_not_down() -> None:
    assert _values("(0 - t.channels) / 4 AS q") == [[0], [-1], [0]]


def test_a_float_operand_makes_the_whole_result_a_float() -> None:
    assert _values("t.channels / 4.0 AS q") == [[0.5], [1.5], [0.5]]


def test_int_arithmetic_stays_an_int() -> None:
    assert _values("t.channels * 2 + 1 AS q") == [[5], [13], [5]]


def test_precedence_is_the_parsers() -> None:
    assert _values("1 + 2 * 3 AS a, (1 + 2) * 3 AS b")[0] == [7, 9]


def test_arithmetic_composes_with_a_comparison() -> None:
    g = _lower(_row_query(where="t.channels / 2 = 3"), _row_probes())
    assert _refs(g) == ["src:f:a:1"]


def test_arithmetic_composes_with_between() -> None:
    g = _lower(_row_query(where="t.channels + 1 BETWEEN 2 AND 3 + 1"), _row_probes())
    assert _refs(g) == ["src:f:a:0", "src:f:a:2"]


def test_arithmetic_on_null_propagates_null() -> None:
    assert _values("t.bitrate + 1 AS q") == [[None], [None], [None]]


def test_division_by_a_known_zero_is_a_typed_rejection() -> None:
    err = _reject_lower(_tag_query("t.channels / 0 AS title"), _row_probes())
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "division by zero" in err.message


def test_a_float_result_prints_its_shortest_roundtrip_form() -> None:
    g = _lower(_tag_query("t.channels * 1.5 AS title"), _row_probes())
    assert [o.metadata["title"] for o in g.outputs] == ["3.0", "9.0", "3.0"]


def test_cast_to_text_bridges_a_number_into_concatenation() -> None:
    g = _lower(_tag_query("'ch' || t.channels::text AS title"), _row_probes())
    assert [o.metadata["title"] for o in g.outputs] == ["ch2", "ch6", "ch2"]


def test_the_cast_function_spelling_is_the_same_cast() -> None:
    assert _values("CAST(t.channels AS text) AS q") == [["2"], ["6"], ["2"]]


def test_casting_null_stays_null() -> None:
    assert _values("t.bitrate::text AS q") == [[None], [None], [None]]


def test_duration_seeks_the_input_from_an_expression() -> None:
    assert _trim("f.t <= f.duration - 0.5", _duration_probes(12.0)) == (None, 11.5)


def test_a_bare_duration_is_a_trim_bound_of_its_own() -> None:
    assert _trim("f.t <= f.duration", _duration_probes(12.0)) == (None, 12.0)


_PAREN_TRIM = (
    "COPY (SELECT f.video[1] FROM input('a.mkv') f "
    "WHERE f.t >= (5 - 1) AND f.t <= 10) TO 'out.mp4'"
)
_BARE_TRIM = (
    "COPY (SELECT f.video[1] FROM input('a.mkv') f "
    "WHERE f.t >= 5 - 1 AND f.t <= 10) TO 'out.mp4'"
)


def test_a_parenthesized_trim_bound_compiles_to_the_same_ir_as_the_bare_one() -> None:
    assert compile_sql(_PAREN_TRIM).to_dict() == compile_sql(_BARE_TRIM).to_dict()


def test_an_unprobed_duration_is_a_rejection_naming_the_field() -> None:
    with pytest.raises(FfrwdError) as excinfo:
        _lower(
            "SELECT f.video[1] FROM input('f.mkv') f WHERE f.t <= f.duration - 0.5",
            _duration_probes(None),
        )
    err = excinfo.value
    assert err.code is ErrorCode.INPUT_NOT_FOUND
    assert "'f.duration' is unknown" in err.message


def test_duration_is_a_value_not_a_stream() -> None:
    err = _reject("SELECT f.duration FROM input('f.mkv') f")
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "'f.duration' is a number of seconds, not a stream" in err.message


def test_a_computed_filter_argument_is_evaluated_per_row() -> None:
    probes: dict[str, ProbeResult | None] = {
        "f": ProbeResult(
            streams=[
                _track("video", 0, width=320, height=240),
                _track("video", 1, width=640, height=480),
            ]
        )
    }
    g = _lower(
        "SELECT array_agg(scale(t, t.width / 2, -2)) "
        "FROM input('f.mkv') f, unnest(f.video) t",
        probes,
    )
    assert [n.args["width"] for n in g.nodes.values() if n.filter == "scale"] == [160, 320]


def test_a_computed_named_argument_is_evaluated_per_row() -> None:
    probes: dict[str, ProbeResult | None] = {
        "f": ProbeResult(
            streams=[
                _track("video", 0, width=320, height=240),
                _track("video", 1, width=640, height=480),
            ]
        )
    }
    g = _lower(
        "SELECT array_agg(scale(t, width => t.width / 4, height => -2)) "
        "FROM input('f.mkv') f, unnest(f.video) t",
        probes,
    )
    assert [n.args["width"] for n in g.nodes.values() if n.filter == "scale"] == [80, 160]


def test_a_computed_argument_still_meets_the_option_table() -> None:
    """Evaluating to a value, not to NULL, still faces the option's type."""
    err = _reject_lower(
        "SELECT gblur(t, t.tags.language || 'x') "
        "FROM input('f.mkv') f, unnest(f.video) t",
        {"f": ProbeResult(streams=[_track("video", 0, width=320, language="eng")])},
    )
    assert err.code is ErrorCode.FILTER_OPTION_TYPE
    assert "'sigma' of filter 'gblur' expects a number" in err.message


# ---------------------------------------------------------------------------
# a `duration`-typed filter option may take a computed value too
# ---------------------------------------------------------------------------


def test_a_computed_duration_option_matches_the_equivalent_seek_bound() -> None:
    """`starti` computed from `f.duration` prints through the same spelling
    rule an equivalent `-ss` bound does -- digit for digit."""
    probes = _duration_probes(9.0)
    trimmed = _lower(
        "SELECT ffmpeg.trim(f.video[1], starti => f.duration / 3) FROM input('f.mkv') f",
        probes,
    )
    seeked = _lower(
        "SELECT f.video[1] FROM input('f.mkv') f WHERE f.t >= f.duration / 3",
        probes,
    )
    assert trimmed.nodes["n1"].args["starti"] == seeked.input_trims["f"][0] == 3.0


def test_a_bare_duration_column_bounds_a_duration_option() -> None:
    """No arithmetic at all -- the probed scalar itself, unwritable before."""
    g = _lower(
        "SELECT ffmpeg.trim(f.video[1], endi => f.duration) FROM input('f.mkv') f",
        _duration_probes(9.0),
    )
    assert g.nodes["n1"].args == {"endi": 9.0}


def test_a_bare_row_column_bounds_a_duration_option_per_element() -> None:
    """A row column with no arithmetic, evaluated once per broadcast element."""
    probes: dict[str, ProbeResult | None] = {
        "f": ProbeResult(streams=[_track("video", 0, bitrate=1), _track("video", 1, bitrate=2)])
    }
    g = _lower(
        "SELECT array_agg(ffmpeg.trim(t, starti => t.bitrate)) "
        "FROM input('f.mkv') f, unnest(f.video) t",
        probes,
    )
    assert [n.args["starti"] for n in g.nodes.values() if n.filter == "trim"] == [1, 2]


def test_a_stream_argument_to_a_duration_option_is_the_typed_rejection() -> None:
    """Not compile-time-countable: a stream is not a number of seconds, and
    the rejection is the existing FILTER_OPTION_TYPE shape."""
    err = _reject_lower(
        "SELECT ffmpeg.trim(f.video[1], starti => f.video[1]) FROM input('f.mkv') f",
        _duration_probes(9.0),
    )
    assert err.code is ErrorCode.FILTER_OPTION_TYPE
    assert "'starti' of filter 'trim'" in err.message
    assert "expects a string" in err.message


# ---------------------------------------------------------------------------
# array_agg + GROUP BY: how several rows reach one file
#
# Rows are gathered only where the query says to gather them, so each shape is
# pinned twice over: the ungrouped spelling is rejected, and the aggregate one
# compiles to the exact command below it.
# ---------------------------------------------------------------------------


def _agg_argv(sql: str, probes: dict[str, ProbeResult | None]) -> list[str]:
    return build_ffmpeg_args(emit(insert_splits(_lower(sql, probes))))


def _agg_copy(select: str) -> str:
    return f"COPY ({select}) TO 'out.mkv'"


# (ungrouped spelling, aggregate spelling, the command the aggregate writes).
# The ungrouped one is rejected wherever it leaves several rows behind; where a
# WHERE narrows the relation to one row it is legal, and writes the same file.
_AGG_SHAPES = [
    (
        "SELECT t FROM input('f.mkv') f, unnest(f.audio) t",
        "SELECT array_agg(t) FROM input('f.mkv') f, unnest(f.audio) t",
        ["ffmpeg", "-i", "f.mkv",
         "-map", "0:a:0", "-c:0", "copy", "-metadata:s:0", "language=eng",
         "-map", "0:a:1", "-c:1", "copy", "-metadata:s:1", "language=fra",
         "-map", "0:a:2", "-c:2", "copy", "out.mkv"],
    ),
    (
        "SELECT f.video, t FROM input('f.mkv') f, unnest(f.audio) t",
        "SELECT f.video, array_agg(t) FROM input('f.mkv') f, "
        "unnest(f.audio) t GROUP BY f.video",
        ["ffmpeg", "-i", "f.mkv",
         "-map", "0:v:0", "-c:0", "copy",
         "-map", "0:a:0", "-c:1", "copy", "-metadata:s:1", "language=eng",
         "-map", "0:a:1", "-c:2", "copy", "-metadata:s:2", "language=fra",
         "-map", "0:a:2", "-c:3", "copy", "out.mkv"],
    ),
    (
        "SELECT volume(t, 0.5) FROM input('f.mkv') f, unnest(f.audio) t",
        "SELECT array_agg(volume(t, 0.5)) FROM input('f.mkv') f, "
        "unnest(f.audio) t",
        ["ffmpeg", "-i", "f.mkv", "-filter_complex",
         "[0:a:0]volume=volume=0.5[out0];[0:a:1]volume=volume=0.5[out1];"
         "[0:a:2]volume=volume=0.5[out2]",
         "-map", "[out0]", "-metadata:s:0", "language=eng",
         "-map", "[out1]", "-metadata:s:1", "language=fra",
         "-map", "[out2]", "out.mkv"],
    ),
    (
        "SELECT t FROM input('f.mkv') f, unnest(f.audio) t "
        "ORDER BY t.channels DESC",
        "SELECT array_agg(t) FROM input('f.mkv') f, unnest(f.audio) t "
        "ORDER BY t.channels DESC",
        ["ffmpeg", "-i", "f.mkv",
         "-map", "0:a:1", "-c:0", "copy", "-metadata:s:0", "language=fra",
         "-map", "0:a:0", "-c:1", "copy", "-metadata:s:1", "language=eng",
         "-map", "0:a:2", "-c:2", "copy", "out.mkv"],
    ),
]

_NARROWED = (
    "SELECT t FROM input('f.mkv') f, unnest(f.audio) t "
    "WHERE t.tags.language = 'fra'"
)
_NARROWED_AGGREGATE = (
    "SELECT array_agg(t) FROM input('f.mkv') f, unnest(f.audio) t "
    "WHERE t.tags.language = 'fra'"
)
_NARROWED_ARGV = [
    "ffmpeg", "-i", "f.mkv",
    "-map", "0:a:1", "-c:0", "copy", "-metadata:s:0", "language=fra", "out.mkv",
]


@pytest.mark.parametrize(
    "ungrouped,aggregate,argv", _AGG_SHAPES, ids=range(len(_AGG_SHAPES))
)
def test_the_ungrouped_shape_is_rejected(
    ungrouped: str, aggregate: str, argv: list[str]
) -> None:
    err = _reject_lower(_agg_copy(ungrouped), _row_probes())
    assert err.code is ErrorCode.ROW_COUNT_MISMATCH
    assert "3 rows" in err.message
    assert "'out.mkv' is one file" in err.message
    assert "array_agg" in (err.hint or "")
    assert "TO (" in (err.hint or "")


@pytest.mark.parametrize(
    "ungrouped,aggregate,argv", _AGG_SHAPES, ids=range(len(_AGG_SHAPES))
)
def test_the_aggregate_shape_writes_the_pinned_command(
    ungrouped: str, aggregate: str, argv: list[str]
) -> None:
    assert _agg_argv(_agg_copy(aggregate), _row_probes()) == argv


def test_a_where_that_leaves_one_row_needs_no_aggregate() -> None:
    """The count is the RESOLVED one: one surviving row is one file, written
    the same way with or without the aggregate."""
    probes = _row_probes()
    assert _agg_argv(_agg_copy(_NARROWED), probes) == _NARROWED_ARGV
    assert _agg_argv(_agg_copy(_NARROWED_AGGREGATE), probes) == _NARROWED_ARGV


def test_an_aggregate_without_a_group_by_is_one_group() -> None:
    """Postgres's own rule: an aggregate and no GROUP BY is the whole set."""
    g = _lower(
        _agg_copy("SELECT array_agg(t) FROM input('f.mkv') f, unnest(f.audio) t"),
        _row_probes(),
    )
    assert len(g.outputs) == 3


def test_a_grouped_scalar_tags_the_container_not_the_tracks() -> None:
    """A grouped branch has no per-row scope, so its scalar columns tag the
    file; ungrouped and single-row, the same column tags the track."""
    probes = _row_probes()
    grouped = _agg_copy(
        "SELECT array_agg(t), STRUCT('Set' AS album) AS tags "
        "FROM input('f.mkv') f, unnest(f.audio) t GROUP BY f.video"
    )
    per_row = _agg_copy(
        "SELECT t, STRUCT('Set' AS album) AS tags "
        "FROM input('f.mkv') f, unnest(f.audio) t WHERE t.index = 1"
    )
    assert _lower(grouped, probes).sinks[0].tags == {"album": "Set"}
    assert _lower(per_row, probes).sinks[0].tags == {}


def test_the_group_key_itself_reads_as_a_container_tag() -> None:
    probes = _row_probes(
        _track("audio", 0, language="eng", codec="aac"),
        _track("audio", 1, language="eng", codec="aac"),
    )
    g = _lower(
        "COPY (SELECT array_agg(t), STRUCT(t.tags.language AS title) AS tags "
        "FROM input('f.mkv') f, unnest(f.audio) t GROUP BY t.tags.language) "
        "TO (t.tags.language || '.mka')",
        probes,
    )
    assert g.sinks[0].tags == {"title": "eng"}
    assert g.sinks[0].path == "eng.mka"


def test_a_grouped_branch_still_has_no_container_disposition() -> None:
    err = _reject_lower(
        _agg_copy(
            "SELECT array_agg(t), 'default' AS disposition "
            "FROM input('f.mkv') f, unnest(f.audio) t"
        ),
        _row_probes(),
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "'disposition' is a stream field, not a container one" in err.message


# -- grouping validity ------------------------------------------------------


def test_an_ungrouped_row_scalar_is_rejected_beside_an_aggregate() -> None:
    err = _reject(
        _agg_copy(
            "SELECT array_agg(t), t.tags.language AS title "
            "FROM input('f.mkv') f, unnest(f.audio) t"
        )
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "'t.tags.language' is neither aggregated nor a GROUP BY key" in err.message
    assert "GROUP BY" in (err.hint or "")
    assert "CTE" in (err.hint or "")


def test_an_ungrouped_row_stream_is_rejected_beside_an_aggregate() -> None:
    err = _reject(
        _agg_copy(
            "SELECT array_agg(t), u FROM input('f.mkv') f, "
            "unnest(f.audio) t, unnest(f.video) u"
        )
    )
    assert "'u' is neither aggregated nor a GROUP BY key" in err.message
    assert "array_agg" in (err.hint or "")


def test_a_row_star_is_rejected_in_a_grouped_branch() -> None:
    err = _reject(
        _agg_copy(
            "SELECT array_agg(t), u.* FROM input('f.mkv') f, "
            "unnest(f.audio) t, unnest(f.video) u"
        )
    )
    assert "'u.*' is neither aggregated nor a GROUP BY key" in err.message


def test_a_to_expression_may_only_read_the_group_keys() -> None:
    err = _reject(
        "COPY (SELECT array_agg(t) FROM input('f.mkv') f, "
        "unnest(f.audio) t GROUP BY t.tags.language) TO (t.codec || '.mka')"
    )
    assert "'t.codec' is neither aggregated nor a GROUP BY key" in err.message


def test_grouping_by_a_row_column_needs_a_fan_out_destination() -> None:
    err = _reject(
        "COPY (SELECT array_agg(t) FROM input('f.mkv') f, "
        "unnest(f.audio) t GROUP BY t.tags.language) TO 'out.mka'"
    )
    assert "writes one file per group" in err.message
    assert "TO (t.tags.language" in (err.hint or "")


def test_a_subscript_works_as_a_group_key() -> None:
    """``GROUP BY f.video[1]``: an input-level key, so one group, and the
    keyed column is mapped once ahead of the gather."""
    g = _lower(
        "COPY (SELECT f.video[1], array_agg(t) FROM input('f.mkv') f, "
        "unnest(f.audio) t GROUP BY f.video[1]) TO 'out.mkv'",
        _row_probes(),
    )
    assert [o.ref for o in g.outputs] == [
        "src:f:v:0",
        "src:f:a:0",
        "src:f:a:1",
        "src:f:a:2",
    ]


def test_a_union_all_branch_may_aggregate_its_own_rows() -> None:
    """One concat segment is one row set, so each branch gathers its own."""
    probes = _row_probes()
    probes["g"] = ProbeResult(streams=list(_ROW_TRACKS))
    g = _lower(
        "COPY (SELECT f.video[1], array_agg(t) FROM input('f.mkv') f, "
        "unnest(f.audio) t GROUP BY f.video[1] UNION ALL "
        "SELECT g.video[1], array_agg(u) FROM input('g.mkv') g, "
        "unnest(g.audio) u GROUP BY g.video[1]) TO 'out.mkv'",
        probes,
    )
    assert [node.filter for node in g.nodes.values()] == ["concat"]
    assert g.nodes["n1"].args == {"n": 2, "v": 1, "a": 3}


# ---------------------------------------------------------------------------
# one row, one file
# ---------------------------------------------------------------------------


def test_a_tag_survives_the_gather_onto_the_output_it_rode_in_on() -> None:
    """A COALESCE silence fill inside an aggregate still carries the paired
    row's language tag: the gather moves streams, and a tag rides its own."""
    probes = _pair_probes()  # f: eng + fra, g: eng only
    g = _lower(
        "COPY (SELECT array_agg(amix(a, COALESCE(b, "
        "ffmpeg.anullsrc(duration => 1)))) "
        "FROM input('f.mkv') f, input('g.mkv') g, "
        "unnest(f.audio) a FULL OUTER JOIN unnest(g.audio) b "
        "ON a.tags.language = b.tags.language) TO 'out.mka'",
        probes,
    )
    args = build_ffmpeg_args(emit(insert_splits(g)))
    assert args[args.index("-metadata:s:1") + 1] == "language=fra"


def test_a_multi_row_view_read_in_from_is_a_row_source() -> None:
    """A view body follows the CTE rule: several rows in FROM, several rows in
    the query reading it, and one path cannot hold them."""
    err = _reject_lower(
        "CREATE VIEW aud AS SELECT t AS track "
        "FROM input('f.mkv') f, unnest(f.audio) t; "
        "COPY (SELECT aud.track FROM aud) TO 'out.mka'",
        _row_probes(),
    )
    assert err.code is ErrorCode.ROW_COUNT_MISMATCH
    assert "this query has 3 rows" in err.message
    assert "'out.mka' is one file" in err.message


def test_the_same_view_gathered_writes_the_file() -> None:
    g = _lower(
        "CREATE VIEW aud AS SELECT t AS track "
        "FROM input('f.mkv') f, unnest(f.audio) t; "
        "COPY (SELECT array_agg(aud.track) FROM aud) TO 'out.mka'",
        _row_probes(),
    )
    assert [o.ref for o in g.outputs] == ["src:f:a:0", "src:f:a:1", "src:f:a:2"]


def test_a_bare_select_compiled_to_a_file_is_anchored_on_the_query() -> None:
    """A bare SELECT lowered as media has no TO to point at, so the
    rejection anchors on the query and says what it says without naming a
    path."""
    err = _reject_lower(
        "SELECT t FROM input('f.mkv') f, unnest(f.audio) t", _row_probes()
    )
    assert err.code is ErrorCode.ROW_COUNT_MISMATCH
    assert err.message == "this query has 3 rows, and it writes one file"


def test_a_chapters_table_over_one_path_is_rejected_too() -> None:
    """Chapters are rows like any other -- the rule has no per-source carve-out."""
    probes: dict[str, ProbeResult | None] = {
        "f": ProbeResult(
            streams=[_track("video", 0)],
            chapters=[
                ChapterMeta(index=1, start_t=0.0, end_t=1.0, title="Intro"),
                ChapterMeta(index=2, start_t=1.0, end_t=2.0, title="Outro"),
            ],
        )
    }
    err = _reject_lower(
        "COPY (SELECT f.video[1] FROM input('f.mkv') f, unnest(f.chapters) c) TO 'out.mkv'",
        probes,
    )
    assert err.code is ErrorCode.ROW_COUNT_MISMATCH
    assert "this query has 2 rows" in err.message


def test_a_row_varying_cte_column_is_rejected_in_a_grouped_branch() -> None:
    """The rule resolve enforces for a track row, enforced here for the CTE
    column only lowering can measure: several body rows, one stream each, and
    neither aggregated nor a key -- so it would read the group's first tuple."""
    err = _reject_lower(
        _VID_AUD_CTES + "SELECT vid.track, aud.track, array_agg(aud.track) "
        "FROM vid, aud GROUP BY vid.track",
        _vid_aud_probes(),
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "'aud.track' is neither aggregated nor a GROUP BY key" in err.message
    assert "array_agg" in (err.hint or "")


def test_a_single_row_cte_column_stays_group_constant() -> None:
    """One body row is the same value for every tuple, so it needs no
    aggregate: the rejection above is about VARYING, not about being a CTE."""
    g = _lower(
        _VID_AUD_CTES + "SELECT vid.track, array_agg(aud.track) FROM vid, aud",
        _vid_aud_probes(),
    )
    assert [o.ref for o in g.outputs] == ["src:f:v:0", "src:f2:a:0", "src:f2:a:1"]


def test_a_row_varying_cte_column_is_rejected_in_a_grouped_table_query() -> None:
    """Table mode reads the group's first tuple the same way, so it takes the
    same rejection."""
    err = _reject_lower_table(
        _VID_AUD_CTES + "SELECT vid.track, aud.track, array_agg(aud.track) "
        "FROM vid, aud GROUP BY vid.track",
        _vid_aud_probes(),
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "'aud.track' is neither aggregated nor a GROUP BY key" in err.message


# ---------------------------------------------------------------------------
# grouped table queries: GROUP BY / array_agg print instead of write
# ---------------------------------------------------------------------------
#
# Table mode reuses 085's exact grouping validity and partition, but a table
# query has no destination -- every group is just a printed row, so the
# media-side "row-column GROUP BY needs a fan-out TO" rule does not apply.


def _reject_table(sql: str) -> FfrwdError:
    with pytest.raises(FfrwdError) as excinfo:
        resolve(parse(sql))
    return _anchored(excinfo.value)


def test_a_grouped_table_query_over_an_input_level_key_is_one_group() -> None:
    """``GROUP BY f.video`` never varies per row, so the whole relation is one
    group -- the single-file COPY's own shape, printed instead of written."""
    sinks = lower_table(
        resolve(
            parse(
                "SELECT f.video, array_agg(t) FROM input('f.mkv') f, "
                "unnest(f.audio) t GROUP BY f.video"
            )
        ),
        _row_probes(),
    )
    assert sinks[0].result.columns == ["video", "array_agg"]
    assert sinks[0].result.rows == [
        [
            ArrayCell(elements=(StreamCell(type="video", spec="0:v:0"),)),
            ArrayCell(
                elements=(
                    StreamCell(type="audio", spec="0:a:0"),
                    StreamCell(type="audio", spec="0:a:1"),
                    StreamCell(type="audio", spec="0:a:2"),
                )
            ),
        ]
    ]


_LANG_TRACKS = [
    _track("audio", 0, language="eng", codec="aac"),
    _track("audio", 1, language="eng", codec="aac"),
    _track("audio", 2, language="fra", codec="aac"),
]


def test_a_grouped_table_query_over_a_row_key_is_one_row_per_group() -> None:
    """A row-column key partitions in FIRST-APPEARANCE order -- the same
    partition a fan-out COPY would write as separate files."""
    sinks = lower_table(
        resolve(
            parse(
                "SELECT t.tags.language, array_agg(t) FROM input('f.mkv') f, "
                "unnest(f.audio) t GROUP BY t.tags.language"
            )
        ),
        {"f": ProbeResult(streams=_LANG_TRACKS)},
    )
    assert sinks[0].result.rows == [
        [
            "eng",
            ArrayCell(
                elements=(
                    StreamCell(type="audio", spec="0:a:0"),
                    StreamCell(type="audio", spec="0:a:1"),
                )
            ),
        ],
        ["fra", ArrayCell(elements=(StreamCell(type="audio", spec="0:a:2"),))],
    ]


def test_a_grouped_table_query_over_an_empty_relation_prints_no_rows() -> None:
    """Fuzz find: no key and no surviving row used to make ONE empty group,
    and printing it panicked. An empty relation has no groups -- the same
    zero rows the ungrouped branch prints."""
    sinks = lower_table(
        resolve(
            parse(
                "SELECT f.duration AS d, array_agg(f.video[1]) "
                "FROM input('f.mkv') f, unnest(f.data) t"
            )
        ),
        _row_probes(),
    )
    assert sinks[0].result.columns == ["d", "array_agg"]
    assert sinks[0].result.rows == []


def test_an_empty_row_set_still_stops_the_media_query_it_would_write() -> None:
    """The table query above prints nothing; the COPY that would WRITE those
    rows keeps its typed rejection."""
    err = _reject_lower(
        "COPY (SELECT array_agg(t) FROM input('f.mkv') f, unnest(f.data) t) "
        "TO 'out.mka'",
        _row_probes(),
    )
    assert err.code is ErrorCode.STREAM_NOT_FOUND
    assert "no data track of 'f.mkv' survived" in err.message


def test_unaliased_array_agg_column_is_named_array_agg() -> None:
    sinks = lower_table(
        resolve(
            parse(
                "SELECT f.video, array_agg(t) FROM input('f.mkv') f, "
                "unnest(f.audio) t GROUP BY f.video"
            )
        ),
        _row_probes(),
    )
    assert sinks[0].result.columns == ["video", "array_agg"]


def test_an_aliased_array_agg_column_keeps_its_alias() -> None:
    sinks = lower_table(
        resolve(
            parse(
                "SELECT f.video, array_agg(t) AS tracks FROM input('f.mkv') f, "
                "unnest(f.audio) t GROUP BY f.video"
            )
        ),
        _row_probes(),
    )
    assert sinks[0].result.columns == ["video", "tracks"]


def test_table_query_grouping_still_enforces_the_grouping_rule() -> None:
    """No GROUP BY key at all makes the whole relation one group, so an
    ungrouped row scalar beside ``array_agg`` still has nothing to match."""
    err = _reject_table(
        "SELECT t.tags.language, array_agg(t) FROM input('f.mkv') f, unnest(f.audio) t"
    )
    assert "'t.tags.language' is neither aggregated nor a GROUP BY key" in err.message


def test_table_query_still_rejects_order_by_inside_array_agg() -> None:
    err = _reject_table(
        "SELECT array_agg(t ORDER BY t.index) FROM input('f.mkv') f, unnest(f.audio) t"
    )
    assert "ORDER BY inside array_agg() is not supported" in err.message


def test_order_by_inside_array_agg_is_rejected() -> None:
    err = _reject(
        _agg_copy(
            "SELECT array_agg(t ORDER BY t.index) FROM input('f.mkv') f, "
            "unnest(f.audio) t"
        )
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "ORDER BY inside array_agg() is not supported" in err.message


def test_array_agg_outside_a_row_branch_is_rejected() -> None:
    err = _reject(_agg_copy("SELECT array_agg(f.audio) FROM input('f.mkv') f"))
    assert "array_agg" in err.message


def test_array_agg_is_only_a_whole_select_column() -> None:
    err = _reject(
        _agg_copy(
            "SELECT volume(array_agg(t), 0.5) FROM input('f.mkv') f, "
            "unnest(f.audio) t"
        )
    )
    assert "array_agg() is only supported as a whole SELECT column" in err.message


@pytest.mark.parametrize("name", ["count", "sum"])
def test_the_other_aggregates_keep_their_typed_rejection(name: str) -> None:
    err = _reject(
        _agg_copy(
            f"SELECT array_agg(t), {name}(t.index) AS n "
            "FROM input('f.mkv') f, unnest(f.audio) t"
        )
    )
    assert err.code is ErrorCode.NO_STREAMING_EQUIVALENT
    assert f"aggregate function {name}() has no streaming equivalent" in err.message
    assert "array_agg" in (err.hint or "")


_AGG_CONTEXTS = [
    (
        "a CTE body",
        "COPY (WITH c AS (SELECT array_agg(t) AS track FROM input('f.mkv') f, "
        "unnest(f.audio) t GROUP BY f.video) SELECT c.track FROM c) TO 'out.mka'",
    ),
    (
        "a view body",
        "CREATE VIEW v AS SELECT array_agg(t) AS track FROM input('f.mkv') f, "
        "unnest(f.audio) t GROUP BY f.video; COPY (SELECT v.track FROM v) TO 'out.mka'",
    ),
]


@pytest.mark.parametrize("where,sql", _AGG_CONTEXTS, ids=[c[0] for c in _AGG_CONTEXTS])
def test_aggregation_is_a_media_copys_own_select(where: str, sql: str) -> None:
    err = _reject(sql)
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert f"GROUP BY is not supported in {where}" in err.message


def test_group_by_without_track_rows_still_has_no_streaming_equivalent() -> None:
    err = _reject("COPY (SELECT f.audio[1] FROM input('f.mkv') f GROUP BY f.audio[1]) TO 'o.mka'")
    assert err.code is ErrorCode.NO_STREAMING_EQUIVALENT
    assert err.message == "GROUP BY has no streaming equivalent"


def test_a_container_tag_key_is_free_form_on_the_read_side_too() -> None:
    g = _lower(
        _container_query("f.tags.encoded_by AS encoder"),
        _tagged_probes(encoded_by="ffrwd"),
    )
    assert g.sinks[0].tags == {"encoder": "ffrwd"}


def test_a_bare_container_tags_column_prints_as_one_array_cell() -> None:
    """The whole map, key/value records in key order."""
    sinks = lower_table(
        resolve(parse("SELECT f.tags FROM input('f.mkv') f")),
        _tagged_probes(title="Angel One", artist="Docs Dept"),
    )
    assert sinks[0].result.columns == ["tags"]
    assert sinks[0].result.rows == [
        [
            ArrayCell(
                elements=(
                    RecordCell(fields=("artist", "Docs Dept")),
                    RecordCell(fields=("title", "Angel One")),
                )
            )
        ]
    ]
    assert "{(artist,Docs Dept),(title,Angel One)}" in render_table(sinks[0].result)


def test_a_bare_container_tags_column_on_an_unprobed_input_is_input_not_found() -> None:
    err = _reject_lower_table("SELECT f.tags FROM input('f.mkv') f", {"f": None})
    assert err.code is ErrorCode.INPUT_NOT_FOUND
    assert "cannot read tags of 'f.mkv'" in err.message


def test_a_bare_container_tags_column_in_a_media_query_is_a_typed_rejection() -> None:
    err = _reject_lower(
        "SELECT f.tags FROM input('f.mkv') f", _tagged_probes(title="Angel One")
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "'f.tags' carries no streams" in err.message
    assert "f.tags.title" in (err.hint or "")


def test_a_row_reads_any_tag_key_its_stream_carries() -> None:
    probes = {
        "f": _probe_result(
            audios=1, audio_tags={"language": "eng", "handler_name": "SoundHandler"}
        )
    }
    sinks = lower_table(
        resolve(
            parse(
                "SELECT t.tags.handler_name "
                "FROM input('f.mkv') f, unnest(f.audio) t"
            )
        ),
        probes,
    )
    assert sinks[0].result.columns == ["handler_name"]
    assert sinks[0].result.rows == [["SoundHandler"]]


def test_a_tag_key_the_stream_lacks_reads_null() -> None:
    sinks = lower_table(
        resolve(
            parse("SELECT t.tags.nosuchkey FROM input('f.mkv') f, unnest(f.audio) t")
        ),
        {"f": _probe_result(audios=1, audio_tags={"language": "eng"})},
    )
    assert sinks[0].result.rows == [[None]]


def test_a_bare_row_tags_column_prints_the_whole_map() -> None:
    sinks = lower_table(
        resolve(parse("SELECT t.tags FROM input('f.mkv') f, unnest(f.audio) t")),
        {"f": _probe_result(audios=1, audio_tags={"language": "eng", "title": "VO"})},
    )
    assert sinks[0].result.columns == ["tags"]
    assert sinks[0].result.rows == [
        [
            ArrayCell(
                elements=(
                    RecordCell(fields=("language", "eng")),
                    RecordCell(fields=("title", "VO")),
                )
            )
        ]
    ]


def test_a_bare_row_tags_column_is_not_a_value() -> None:
    err = _reject_lower(
        "COPY (SELECT t, STRUCT(t.tags AS x) AS tags "
        "FROM input('f.mkv') f, unnest(f.audio) t) TO 'out.mka'",
        {"f": _probe_result(audios=1, audio_tags={"language": "eng"})},
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "'t.tags' is the whole tag map, not a single value" in err.message
    assert "t.tags.language" in (err.hint or "")


def test_only_language_and_title_ride_a_filter_through() -> None:
    """A source's other tags stay put: riding them would emit -metadata
    ffmpeg does not emit today."""
    g = _lower(
        "COPY (SELECT volume(f.audio[1], 0.5) FROM input('f.mkv') f) TO 'out.mka'",
        {
            "f": _probe_result(
                audio_tags={"language": "fra", "title": "VF", "encoder": "Lavc"}
            )
        },
    )
    assert g.outputs[0].metadata == {"language": "fra", "title": "VF"}

# -- disposition flags ------------------------------------------------------


def _flag_probes() -> dict[str, ProbeResult | None]:
    return _row_probes(
        _track("audio", 0, language="eng", codec="aac", disposition={"default": True}),
        _track(
            "audio",
            1,
            language="fra",
            codec="ac3",
            disposition={"default": False, "forced": True},
        ),
    )


def _flag_copy(projection: str, where: str = "") -> str:
    return (
        f"COPY (SELECT {projection} FROM input('f.mkv') f, unnest(f.audio) t"
        + (f" WHERE {where}" if where else "")
        + ") TO 'out.mka'"
    )


def test_a_flag_reads_as_a_boolean_row_value() -> None:
    sinks = lower_table(
        resolve(parse(
            "SELECT t.index, t.disposition.default, t.disposition.karaoke "
            "FROM input('f.mkv') f, unnest(f.audio) t"
        )),
        _flag_probes(),
    )
    assert sinks[0].result.columns == ["index", "default", "karaoke"]
    # The second track reported `default` false; neither reported `karaoke`.
    assert sinks[0].result.rows == [[1, True, None], [2, False, None]]
    assert " true " in render_table(sinks[0].result)


def test_a_flag_filters_rows_at_compile_time() -> None:
    g = _lower(_flag_copy("t", where="t.disposition.default"), _flag_probes())
    assert _refs(g) == ["src:f:a:0"]
    g = _lower(_flag_copy("t", where="NOT t.disposition.default"), _flag_probes())
    assert _refs(g) == ["src:f:a:1"]
    g = _lower(_flag_copy("t", where="t.disposition.forced = true"), _flag_probes())
    assert _refs(g) == ["src:f:a:1"]


def test_a_bare_disposition_column_prints_as_one_array_cell() -> None:
    """The closed key set, in the type's order; a flag this file never reported
    reads NULL, exactly as an absent tag does."""
    sinks = lower_table(
        resolve(parse(
            "SELECT t.disposition FROM input('f.mkv') f, unnest(f.audio) t "
            "WHERE t.index = 1"
        )),
        _flag_probes(),
    )
    cell = sinks[0].result.rows[0][0]
    assert isinstance(cell, ArrayCell)
    assert [record.fields[0] for record in cell.elements] == list(DISPOSITION_KEYS)
    assert cell.elements[0] == RecordCell(fields=("default", True))
    assert cell.elements[1] == RecordCell(fields=("dub", None))
    assert "{(default,true),(dub,)" in render_table(sinks[0].result)


def test_a_written_spec_becomes_the_flags_it_names() -> None:
    g = _lower(_flag_copy("t, 'default+forced' AS disposition", "t.index = 1"),
               _flag_probes())
    assert g.sinks[0].outputs[0].disposition == ("default", "forced")
    assert build_ffmpeg_args(emit(g))[-3:-1] == ["-disposition:0", "default+forced"]


def test_the_flag_order_is_the_types_own_not_the_writers() -> None:
    g = _lower(_flag_copy("t, 'forced+default' AS disposition", "t.index = 1"),
               _flag_probes())
    assert g.sinks[0].outputs[0].disposition == ("default", "forced")


def test_a_cleared_disposition_writes_the_clearing_spec() -> None:
    for value in ("'0'", "NULL"):
        g = _lower(_flag_copy(f"t, {value} AS disposition", "t.index = 1"),
                   _flag_probes())
        assert g.sinks[0].outputs[0].disposition == ()
        assert build_ffmpeg_args(emit(g))[-3:-1] == ["-disposition:0", "0"]


def test_an_unwritten_disposition_reaches_no_argument() -> None:
    """ffmpeg copies a stream's own flags; only a written column says otherwise."""
    g = _lower(_flag_copy("t", "t.index = 1"), _flag_probes())
    assert g.sinks[0].outputs[0].disposition is None
    assert "-disposition:0" not in build_ffmpeg_args(emit(g))


def test_a_written_flag_outside_the_closed_set_is_rejected() -> None:
    err = _reject_lower(
        _flag_copy("t, 'default+forcd' AS disposition", "t.index = 1"), _flag_probes()
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "'forcd' is not a disposition flag" in err.message
    assert err.hint == "did you mean 'forced'?"


def test_a_relative_flag_spec_is_rejected() -> None:
    """ffmpeg's own `+flag`/`-flag` adjusts what the source carries; this column
    says what the whole map is, so there is nothing to adjust."""
    for value in ("'+forced'", "'-forced'", "'default+'", "''"):
        err = _reject_lower(
            _flag_copy(f"t, {value} AS disposition", "t.index = 1"), _flag_probes()
        )
        assert err.code is ErrorCode.UNSUPPORTED_SQL
        assert "is not a flag list" in err.message


def test_a_numeric_disposition_value_is_rejected() -> None:
    err = _reject_lower(_flag_copy("t, 0 AS disposition", "t.index = 1"), _flag_probes())
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "takes ffmpeg's flag spec, not a number" in err.message


def test_one_track_cannot_take_two_dispositions() -> None:
    """Row-scoped, like a tag: a track two result rows disagree about has no
    single flag map to write."""
    err = _reject_lower(
        "COPY (SELECT t, CASE WHEN u.index = 1 THEN 'default' ELSE '0' END "
        "AS disposition FROM input('f.mkv') f, unnest(f.audio) t, "
        "unnest(f.audio) u) TO 'out.mka'",
        _flag_probes(),
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "the disposition takes two different values on the same track" in err.message


def test_two_cte_bodies_cannot_flag_one_track_two_ways() -> None:
    err = _reject_lower(
        "WITH one AS ("
        "  SELECT t AS track, 'default' AS disposition"
        "  FROM input('f.mkv') f, unnest(f.audio) t"
        "), two AS ("
        "  SELECT u AS track, 'forced' AS disposition"
        "  FROM input('g.mkv') g, unnest(g.audio) u"
        ") SELECT one.track FROM one",
        _shared_probes(),
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "the disposition takes two different values on the same track" in err.message


def test_a_cte_bodys_disposition_rides_into_the_sink() -> None:
    """Recipe 41's shape: flag the rows inside the WITH, gather them outside."""
    g = _lower(
        "COPY (WITH flagged AS ("
        "SELECT t AS track, "
        "CASE WHEN t.tags.language = 'eng' THEN 'default' ELSE '0' END AS disposition "
        "FROM input('f.mkv') f, unnest(f.audio) t) "
        "SELECT array_agg(flagged.track) FROM flagged) TO 'out.mka'",
        _flag_probes(),
    )
    assert [o.disposition for o in g.sinks[0].outputs] == [("default",), ()]


def test_a_disposition_survives_a_filter_the_way_a_tag_does() -> None:
    g = _lower(
        _flag_copy("volume(t, 0.5), 'forced' AS disposition", "t.index = 2"),
        _flag_probes(),
    )
    assert g.sinks[0].outputs[0].disposition == ("forced",)


def test_a_chapter_list_may_leave_title_out() -> None:
    """title is nullable, so a record may write NULL for it - the same rule
    any nullable field follows at construction."""
    query = (
        "COPY (SELECT f.video[1], "
        "ARRAY[STRUCT(NULL AS title, 0 AS start_t, 30 AS end_t)::chapter, STRUCT(NULL AS title, 30 AS start_t, 90 AS end_t)::chapter] AS chapters "
        "FROM input('x.mp4') f) TO 'o.mkv'"
    )
    graph = compile_sql(query)
    document = base64.b64decode(graph.input_paths[1].split(",", 1)[1]).decode()
    assert "title=" not in document
    assert document.count("[CHAPTER]") == 2


def test_a_chapter_list_still_needs_its_span() -> None:
    query = (
        "COPY (SELECT f.video[1], ARRAY[STRUCT('Intro' AS title, 0 AS start_t, NULL AS end_t)::chapter] AS chapters "
        "FROM input('x.mp4') f) TO 'o.mkv'"
    )
    with pytest.raises(FfrwdError) as excinfo:
        compile_sql(query)
    assert "'chapters.end_t' must be a number, got NULL" in excinfo.value.message


# ---------------------------------------------------------------------------
# NULL is absence: unset variables drop options, required positions reject
# ---------------------------------------------------------------------------

from ffrwd.vars import substitute as _substitute  # noqa: E402


def _compile_vars(sql: str, variables: dict[str, str] | None = None) -> Graph:
    sub = _substitute(sql, variables or {})
    return compile_sql(sub.text, unset=sub.unset)


def _compile_vars_error(sql: str, variables: dict[str, str] | None = None) -> FfrwdError:
    sub = _substitute(sql, variables or {})
    with pytest.raises(FfrwdError) as excinfo:
        compile_sql(sub.text, unset=sub.unset)
    return excinfo.value


def _only_filter(g: Graph, name: str):
    found = [node for node in g.nodes.values() if node.filter == name]
    assert len(found) == 1
    return found[0]


def test_unset_positional_option_drops_and_the_next_slot_holds() -> None:
    g = _compile_vars(
        "COPY (SELECT scale(f.video[1], :w, :h) FROM input('a.mkv') f) TO 'o.mp4'",
        {"h": "480"},
    )
    assert _only_filter(g, "scale").args == {"height": 480}


def test_literal_null_positional_option_drops_identically() -> None:
    g = compile_sql(
        "COPY (SELECT scale(f.video[1], NULL, 480) FROM input('a.mkv') f) TO 'o.mp4'"
    )
    assert _only_filter(g, "scale").args == {"height": 480}


def test_unset_named_option_drops() -> None:
    g = _compile_vars(
        "COPY (SELECT scale(f.video[1], width => :w, height => 480) "
        "FROM input('a.mkv') f) TO 'o.mp4'"
    )
    assert _only_filter(g, "scale").args == {"height": 480}


def test_dropped_position_still_occupies_its_slot() -> None:
    err = _compile_vars_error(
        "COPY (SELECT scale(f.video[1], :w, width => 640) "
        "FROM input('a.mkv') f) TO 'o.mp4'"
    )
    assert err.code is ErrorCode.FILTER_OPTION_TYPE
    assert "already set positionally" in err.message


def test_unknown_named_option_still_rejects_with_a_null_value() -> None:
    err = _compile_vars_error(
        "COPY (SELECT scale(f.video[1], nope => :x) FROM input('a.mkv') f) TO 'o.mp4'"
    )
    assert err.code is ErrorCode.UNKNOWN_FILTER_OPTION


def test_null_enable_drops_like_any_option() -> None:
    g = _compile_vars(
        "COPY (SELECT gblur(f.video[1], 5, enable => :'win') "
        "FROM input('a.mkv') f) TO 'o.mp4'"
    )
    assert _only_filter(g, "gblur").args == {"sigma": 5}


def test_unset_source_filter_option_drops() -> None:
    g = _compile_vars(
        "COPY (SELECT s.audio[1] FROM ffmpeg.sine(frequency => :f, duration => 2) s) "
        "TO 'o.mp4'"
    )
    assert _only_filter(g, "sine").args == {"duration": 2}


def test_unset_input_option_drops() -> None:
    g = _compile_vars(
        "COPY (SELECT f.video[1] FROM input('a.mkv', hwaccel => :'hw') f) TO 'o.mp4'"
    )
    assert g.input_options == {}


def test_unset_sink_option_drops_and_the_set_one_stays() -> None:
    g = _compile_vars(
        "COPY (SELECT f.video[1] FROM input('a.mkv') f) TO 'o.mp4' "
        "WITH (crf :crf, preset :'preset')",
        {"preset": "fast"},
    )
    assert g.sinks[0].options == {"preset": "fast"}


def test_computed_option_evaluating_to_null_drops() -> None:
    g = _compile_vars(
        "COPY (SELECT scale(f.video[1], :w / 2, 480) FROM input('a.mkv') f) TO 'o.mp4'"
    )
    assert _only_filter(g, "scale").args == {"height": 480}


# -- required, derived from use --


def test_unset_input_path_names_the_variable() -> None:
    err = _compile_vars_error(
        "COPY (SELECT f.video[1] FROM input(:'source') f) TO 'o.mp4'"
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert err.message == "':source' was not set"
    assert err.hint is not None and "-v source=" in err.hint


def test_literal_null_input_path_rejects_plainly() -> None:
    err = _compile_vars_error("COPY (SELECT f.video[1] FROM input(NULL) f) TO 'o.mp4'")
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "input() needs a path, got NULL" in err.message


def test_unset_destination_names_the_variable() -> None:
    err = _compile_vars_error("COPY (SELECT f.video[1] FROM input('a.mkv') f) TO :'dest'")
    assert err.message == "':dest' was not set"
    assert err.hint is not None and "-v dest=" in err.hint


def test_literal_null_destination_rejects_plainly() -> None:
    err = _compile_vars_error("COPY (SELECT f.video[1] FROM input('a.mkv') f) TO NULL")
    assert "COPY needs a destination path, got NULL" in err.message


def test_unset_to_expression_names_the_variable() -> None:
    err = _compile_vars_error(
        "COPY (SELECT f.video[1] FROM input('a.mkv') f) TO (:'dest')"
    )
    assert err.message == "':dest' was not set"


def test_unset_stream_position_names_the_variable() -> None:
    err = _compile_vars_error(
        "COPY (SELECT scale(:clip, 640, 480) FROM input('a.mkv') f) TO 'o.mp4'"
    )
    assert err.code is ErrorCode.UDF_ARG_TYPE
    assert err.message == "':clip' was not set"
    assert err.hint is not None and "stream" in err.hint


def test_literal_null_stream_position_rejects_plainly() -> None:
    err = _compile_vars_error(
        "COPY (SELECT scale(NULL, 640, 480) FROM input('a.mkv') f) TO 'o.mp4'"
    )
    assert err.code is ErrorCode.UDF_ARG_TYPE
    assert "got NULL" in err.message


def test_null_macro_stream_rejects() -> None:
    err = _compile_vars_error(
        "COPY (SELECT ffrwd.delay(:clip, 2) FROM input('a.mkv') f) TO 'o.mp4'"
    )
    assert err.code is ErrorCode.UDF_ARG_TYPE
    assert err.message == "':clip' was not set"


# -- the curated required list --


def test_omitted_required_option_rejects() -> None:
    err = _compile_vars_error(
        "COPY (SELECT subtitles(f.video[1]) FROM input('a.mkv') f) TO 'o.mp4'"
    )
    assert err.code is ErrorCode.FILTER_OPTION_TYPE
    assert "requires option 'filename'" in err.message


def test_required_option_dropped_by_unset_variable_names_it() -> None:
    err = _compile_vars_error(
        "COPY (SELECT subtitles(f.video[1], filename => :'subs') "
        "FROM input('a.mkv') f) TO 'o.mp4'"
    )
    assert err.code is ErrorCode.FILTER_OPTION_TYPE
    assert err.message == "':subs' was not set"
    assert err.hint is not None and "filename" in err.hint and "-v subs=" in err.hint


def test_required_option_dropped_positionally_names_it_too() -> None:
    err = _compile_vars_error(
        "COPY (SELECT subtitles(f.video[1], :'subs') FROM input('a.mkv') f) TO 'o.mp4'"
    )
    assert err.message == "':subs' was not set"


def test_drawtext_either_of_text_and_textfile_satisfies() -> None:
    g = _compile_vars(
        "COPY (SELECT drawtext(f.video[1], text => :'text', textfile => 'x.txt') "
        "FROM input('a.mkv') f) TO 'o.mp4'"
    )
    assert _only_filter(g, "drawtext").args == {"textfile": "x.txt"}


def test_drawtext_with_neither_rejects_naming_both() -> None:
    err = _compile_vars_error(
        "COPY (SELECT drawtext(f.video[1], text => :'text') "
        "FROM input('a.mkv') f) TO 'o.mp4'"
    )
    assert err.message == "':text' was not set"


def test_xfade_requires_expr_only_for_a_custom_transition() -> None:
    base = (
        "COPY (SELECT xfade(a.video[1], b.video[1], transition => '{t}') "
        "FROM input('a.mkv') a, input('b.mkv') b) TO 'o.mp4'"
    )
    err = _compile_vars_error(base.format(t="custom"))
    assert "requires option 'expr' when transition is 'custom'" in err.message
    assert _only_filter(_compile_vars(base.format(t="fade")), "xfade").args == {
        "transition": "fade"
    }


# -- tags: NULL clears, COALESCE keeps --


def test_unset_tag_variable_clears_the_tag() -> None:
    """A cleared tag is an ABSENT key: nothing carries through to -metadata."""
    g = _compile_vars(
        "COPY (SELECT f.video[1], STRUCT(:'title' AS title) AS tags "
        "FROM input('a.mkv') f) TO 'o.mp4'"
    )
    assert g.sinks[0].tags == {"title": None}


def test_coalesce_keeps_a_tag_when_the_variable_is_unset() -> None:
    g = _compile_vars(
        "COPY (SELECT f.video[1], "
        "STRUCT(COALESCE(:'title', 'Untitled') AS title) AS tags "
        "FROM input('a.mkv') f) TO 'o.mp4'"
    )
    assert g.sinks[0].tags == {"title": "Untitled"}


def test_coalesce_prefers_the_set_variable() -> None:
    g = _compile_vars(
        "COPY (SELECT f.video[1], "
        "STRUCT(COALESCE(:'title', 'Untitled') AS title) AS tags "
        "FROM input('a.mkv') f) TO 'o.mp4'",
        {"title": "Named"},
    )
    assert g.sinks[0].tags == {"title": "Named"}


# -- a filter option read off a row: one node per row --


def test_a_filter_option_read_off_a_row_makes_one_node_per_row() -> None:
    """The gathered ladder. Rung 2 and rung 3 used to be dropped silently:
    the call expanded over its ARRAY arguments alone, and a single stream
    through an option that differs per row is not one."""
    g = _lower(
        "COPY (SELECT array_agg(scale(f.video[1], ARRAY[640, 1280, 1920][i.i], -2)) "
        "FROM input('a.mkv') f, generate_series(1, 3) i) TO 'ladder.mkv'"
    )
    scales = [node for node in g.nodes.values() if node.filter == "scale"]
    assert [node.args["width"] for node in scales] == [640, 1280, 1920]
    assert [o.type for o in g.outputs] == ["video", "video", "video"]


def test_the_rungs_of_a_gathered_ladder_are_distinct_nodes_in_a_table() -> None:
    """The same collapse, shown before any aggregate: three rows, three
    nodes."""
    sinks = lower_table(
        resolve(
            parse(
                "SELECT scale(f.video[1], ARRAY[640, 1280, 1920][i.i], -2) "
                "FROM input('a.mkv') f, generate_series(1, 3) i"
            )
        ),
        {},
        registry=_snapshot_registry(),
    )
    specs = [row[0].spec for row in sinks[0].result.rows]
    assert len(specs) == 3
    assert len(set(specs)) == 3, specs


def test_a_filter_option_reading_no_row_still_makes_one_node() -> None:
    """Only reading a row expands: rows that ask for the same thing share the
    one node they always shared."""
    g = _lower(
        "COPY (SELECT array_agg(scale(f.video[1], 640, -2)) "
        "FROM input('a.mkv') f, generate_series(1, 3) i) TO 'one.mkv'"
    )
    assert _filters(g).count("scale") == 1


def test_an_input_column_is_not_a_row_column() -> None:
    """``f.duration`` is the same for every row, so it expands nothing."""
    g = _lower(
        "COPY (SELECT array_agg(ffmpeg.trim(f.video[1], starti => f.duration / 2)) "
        "FROM input('a.mkv') f, generate_series(1, 3) i) TO 'one.mkv'",
        {"f": ProbeResult(streams=list(_probe_result().streams), duration=10.0)},
    )
    assert _filters(g).count("trim") == 1


def test_a_stream_array_and_a_per_row_option_of_different_lengths_are_refused() -> None:
    err = _reject_lower(
        "COPY (SELECT array_agg(scale(f.video, ARRAY[640, 1280, 1920][i.i], -2)) "
        "FROM input('a.mkv') f, generate_series(1, 3) i) TO 'o.mkv'",
        {"f": _probe_result(videos=2)},
    )
    assert err.code is ErrorCode.BROADCAST_MISMATCH
    assert "read once per row over" in err.message


# ---------------------------------------------------------------------------
# module sources: `FROM <source>(<values>) alias` -- a `RETURNS source`
# call, the mirror of a sink, bound through the same row machinery a
# manifest's renditions already use.
#
# resolve()'s FROM-item gate (parser.py's `_add_input`, around line 5350)
# still answers ANY function name but `input` with "unsupported table
# function <name>()" -- the chunk that widens it to admit a `RETURNS
# source` call, the way it already admits `input`, is not this one's. So
# these tests bypass `resolve()` for the FROM item and hand `lower()` the
# `Resolved` it would build once that gate opens: `parse()` alone (no
# semantic checks) for the tree, and a hand-filled `wasm` table -- exactly
# the way a `ProbeResult` stands in for `probe()` elsewhere in this file.
# ---------------------------------------------------------------------------


def _source_function(
    name: str = "subscribe",
    module: str = "modules/subscribe.wasm",
    export: str = "subscribe",
) -> WasmFunction:
    """One ``RETURNS source`` declaration: two text value parameters, the
    shape a subscription's relay/broadcast names would take."""
    return WasmFunction(
        name=name,
        module=module,
        export=export,
        params=(Parameter("relay", "text"), Parameter("broadcast", "text")),
        returns=WASM_SOURCE,
        line=1,
        col=1,
    )


def _source_described(
    declared: WasmFunction, *, source: bool = True, world: str = WORLDS[-1]
) -> Described:
    return Described(
        world=world,
        name=declared.export,
        params_schema={
            "properties": {
                "relay": {"type": "string"},
                "broadcast": {"type": "string"},
            }
        },
        source=source,
    )


def _subscribe_catalog(*, bounded: bool = False) -> SourceCatalog:
    """A muxed 720p rendition (row 0, h264+aac) beside an audio-only one
    (row 1, language 'en') -- the shape a catalog's own tracks take, mirrored
    into rendition rows exactly as :func:`ffrwd.wasm.catalog_as_probe` reads
    a manifest's `RenditionMeta` list.
    """
    rung = SourceRendition(name="720p", bandwidth=2_000_000, codecs=None, language=None)
    solo = SourceRendition(name=None, bandwidth=128_000, codecs=None, language="en")
    return SourceCatalog(
        tracks=(
            WasmSourceTrack(
                codec="h264", time_base=(1, 90000), kind="video",
                width=1280, height=720, sample_rate=None, channels=None,
                extradata=b"", profile=None, level=None, row=0, rendition=rung,
            ),
            WasmSourceTrack(
                codec="aac", time_base=(1, 48000), kind="audio",
                width=None, height=None, sample_rate=48000, channels=2,
                extradata=b"", profile=None, level=None, row=0, rendition=rung,
            ),
            WasmSourceTrack(
                codec="aac", time_base=(1, 48000), kind="audio",
                width=None, height=None, sample_rate=48000, channels=2,
                extradata=b"", profile=None, level=None, row=1, rendition=solo,
            ),
        ),
        bounded=bounded,
    )


def _module_source_resolved(sql: str, declared: WasmFunction) -> Resolved:
    """The `Resolved` `lower()` would be handed once resolve()'s FROM-item
    gate admits a `RETURNS source` call: `sql`'s tree, parsed but never
    resolved, its alias assigned no `-i` slot (`sources`/`input_paths` stay
    empty -- a module source's bytes never come from one), and `wasm`
    carrying the one declaration the call names.
    """
    tree = parse(sql)
    assert isinstance(tree, sqlglot.exp.Select)
    return Resolved(
        select=tree,
        input_paths=[],
        sources={},
        branches=[tree],
        wasm={declared.name: declared},
    )


def _lower_source(
    sql: str,
    declared: WasmFunction,
    described: Described,
    fake_probe_source: object,
) -> Graph:
    return lower(
        _module_source_resolved(sql, declared),
        {},
        registry=_snapshot_registry(),
        describes={declared.module: described},
        probe_source=fake_probe_source,  # type: ignore[arg-type]
    )


def _reject_source(
    sql: str, declared: WasmFunction, described: Described, fake_probe_source: object
) -> FfrwdError:
    with pytest.raises(FfrwdError) as excinfo:
        _lower_source(sql, declared, described, fake_probe_source)
    return _anchored(excinfo.value)


def test_a_module_source_alias_binds_two_rendition_rows() -> None:
    """`FROM subscribe(...) s` binds `s` as a row table with no `unnest`,
    same as a probed manifest: one row per catalog `row`, so `amix` over
    every row's audio reads exactly 2 inputs even though the catalog carries
    3 tracks (the muxed row's video contributes nothing to `.audio`)."""
    declared = _source_function()
    described = _source_described(declared)
    catalog = _subscribe_catalog()
    g = _lower_source(
        "SELECT amix(VARIADIC array_agg(s.audio)) FROM subscribe('relay', 'live') s",
        declared, described, lambda module, params: catalog,
    )
    (node,) = g.nodes.values()
    assert node.filter == "amix"
    assert node.inputs == ["src:s:a:0", "src:s:a:1"]


def test_module_source_rendition_columns_resolve() -> None:
    """`s.height` and `s.language` read back exactly as a manifest's own
    rendition columns do, narrowing the catalog's two rows to one apiece."""
    declared = _source_function()
    described = _source_described(declared)
    catalog = _subscribe_catalog()
    by_height = _lower_source(
        "SELECT s.video[1] FROM subscribe('relay', 'live') s WHERE s.height = 720",
        declared, described, lambda module, params: catalog,
    )
    assert _outputs(by_height) == [("src:s:v:0", "video", None)]
    by_language = _lower_source(
        "SELECT s.audio[1] FROM subscribe('relay', 'live') s WHERE s.language = 'en'",
        declared, described, lambda module, params: catalog,
    )
    # the audio-only row's own audio track: the second one the catalog counts
    assert _outputs(by_language) == [("src:s:a:1", "audio", None)]


def test_a_where_on_module_source_height_keeps_one_row() -> None:
    """`WHERE s.height = 720` narrows the two-row catalog to the muxed row
    alone, so both its video and audio reach the SELECT list."""
    declared = _source_function()
    described = _source_described(declared)
    catalog = _subscribe_catalog()
    g = _lower_source(
        "SELECT s.video[1], s.audio[1] FROM subscribe('relay', 'live') s "
        "WHERE s.height = 720",
        declared, described, lambda module, params: catalog,
    )
    assert _outputs(g) == [
        ("src:s:v:0", "video", None),
        ("src:s:a:0", "audio", None),
    ]


def test_array_agg_of_a_module_source_video_column_gathers() -> None:
    """`array_agg(s.video[1])` gathers every surviving row's own video
    stream, in row order -- here just the one row that carries video,
    exactly as a manifest's audio-only rendition contributes nothing."""
    declared = _source_function()
    described = _source_described(declared)
    catalog = _subscribe_catalog()
    g = _lower_source(
        "SELECT array_agg(s.video[1]) FROM subscribe('relay', 'live') s",
        declared, described, lambda module, params: catalog,
    )
    assert _outputs(g) == [("src:s:v:0", "video", None)]


def test_the_graph_carries_the_module_source_and_its_tracks() -> None:
    """`Graph.module_sources` records the same catalog as IR: the module
    path, the folded params JSON, one `SourceTrack` per catalog track with
    its ref/row/rendition, and `bounded` off the catalog -- the mirror of
    `Graph.packet_sinks`."""
    declared = _source_function()
    described = _source_described(declared)
    catalog = _subscribe_catalog()
    # narrowed to the muxed row, so the bare SELECT still writes one file --
    # `Graph.module_sources` records the WHOLE catalog regardless, unaffected
    # by which rows a WHERE leaves standing.
    g = _lower_source(
        "SELECT s.video[1] FROM subscribe('relay', 'live') s WHERE s.height = 720",
        declared, described, lambda module, params: catalog,
    )
    source = g.module_sources["s"]
    assert source.alias == "s"
    assert source.module == declared.module
    # value args fold into the module's own params the way a sink's do,
    # JSON-serialized the same way `-params` is (sorted keys).
    assert source.params == '{"broadcast": "live", "relay": "relay"}'
    assert [t.ref for t in source.tracks] == ["src:s:v:0", "src:s:a:0", "src:s:a:1"]
    assert [t.row for t in source.tracks] == [0, 0, 1]
    assert [t.kind for t in source.tracks] == ["video", "audio", "audio"]
    assert source.tracks[2].language == "en"
    assert source.tracks[0].codec == "h264"
    assert source.bounded is False


def test_a_bounded_module_source_reads_back_on_the_graph() -> None:
    """`bounded` on the catalog reaches `ModuleSource.bounded` unchanged --
    the processes pass (not this one's) is what turns `False` into joining
    the live set."""
    declared = _source_function()
    described = _source_described(declared)
    catalog = _subscribe_catalog(bounded=True)
    g = _lower_source(
        "SELECT s.video[1] FROM subscribe('relay', 'live') s WHERE s.height = 720",
        declared, described, lambda module, params: catalog,
    )
    assert g.module_sources["s"].bounded is True


def test_a_plain_input_alias_is_unaffected_by_the_source_binding_path() -> None:
    """`_add_table`'s Anonymous branch now checks `res.wasm` for a source
    declaration before falling back to `input()`'s own handling; a script
    naming no wasm source at all must resolve exactly as it always has."""
    g = _lower("SELECT f.video[1] FROM input('f.mp4') f", _row_probes())
    assert _outputs(g) == [("src:f:v:0", "video", None)]


def test_a_module_source_probe_failure_is_a_typed_refusal() -> None:
    """The sidecar's own probe error surfaces at the FROM item, re-anchored
    the same way an invoke() failure is (`_lower_wasm_call`)."""
    declared = _source_function()
    described = _source_described(declared)

    def failing_probe(module: str, params: str) -> SourceCatalog:
        raise FfrwdError(
            ErrorCode.UNSUPPORTED_SQL,
            "the relay refused the subscription: 403",
            hint="check the token",
        )

    err = _reject_source(
        "SELECT s.video[1] FROM subscribe('relay', 'live') s",
        declared, described, failing_probe,
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "cannot read 's'" in err.message
    assert "the relay refused the subscription: 403" in err.message
    assert err.hint == "check the token"


def test_a_where_time_window_on_a_module_source_row_is_not_a_column() -> None:
    """A module source binds through the SAME rendition-row table a
    manifest does (:meth:`_bind_renditions`), so ``s.t`` is refused the
    exact way a ladder's own rendition alias refuses it: a rendition row
    exposes its metadata columns, never a time column to seek by -- there is
    no file offset behind it either way."""
    declared = _source_function()
    described = _source_described(declared)
    catalog = _subscribe_catalog()
    err = _reject_source(
        "SELECT s.video[1] FROM subscribe('relay', 'live') s "
        "WHERE s.height = 720 AND s.t <= 1",
        declared, described, lambda module, params: catalog,
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "unknown column 's.t'" in err.message


def test_a_where_time_window_on_an_empty_module_source_is_refused() -> None:
    """The degenerate catalog with no tracks at all -- never the contracted
    shape (a real source always reports at least one row), but the one case
    where the alias stays bound as a plain input rather than a rendition row
    table, so a WHERE window would otherwise reach `Graph.input_trims` for
    an alias with no `-i` to seek. Refused by name instead of crashing."""
    declared = _source_function()
    described = _source_described(declared)
    empty_catalog = SourceCatalog(tracks=(), bounded=False)
    err = _reject_source(
        "SELECT s.video[1] FROM subscribe('relay', 'live') s WHERE s.t <= 1",
        declared, described, lambda module, params: empty_catalog,
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "'s' is a module source" in err.message
    assert "nothing to seek" in err.message
    assert err.hint is not None and "paces itself" in err.hint


def test_a_non_packet_source_module_named_in_from_is_refused() -> None:
    """The module's own describe says it is not a packet source: refused by
    name rather than left to fail at run time."""
    declared = _source_function()
    described = _source_described(declared, source=False)
    err = _reject_source(
        "SELECT s.video[1] FROM subscribe('relay', 'live') s",
        declared, described, lambda module, params: _subscribe_catalog(),
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "is not a packet source" in err.message


def test_a_module_source_on_a_pre_packet_source_world_is_refused() -> None:
    """A module built against a world before packet sources arrived (0.13.0)
    cannot be hosted as one, named by the world it targets."""
    declared = _source_function()
    described = _source_described(declared, world="ffrwd:av@0.9.0")
    err = _reject_source(
        "SELECT s.video[1] FROM subscribe('relay', 'live') s",
        declared, described, lambda module, params: _subscribe_catalog(),
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "cannot host one" in err.message


# ---------------------------------------------------------------------------
# A sink declaring no stream parameters reads the COPY's rows straight off
# the SELECT list -- the manifest destination's own reading, applied to a
# module (``ffrwd.moq.publish`` in the real package): N rows, each a video
# cell and an audio cell, either NULL, flattened into the sink's pads in
# ROW-MAJOR order. `_row_sink_graph`/`_row_sink_rejects` go through the real
# parser and resolver, the way `_lower_source` does for a module source --
# only `describes` is synthetic, exactly as `_ladder_graph` hands a packet
# sink's own describe in tests/test_wasm.py.
# ---------------------------------------------------------------------------

ROW_SINK_MODULE = "modules/publish.wasm"

ROW_SINK_DECLARE = (
    "CREATE FUNCTION publish(relay text, broadcast text) RETURNS sink\n"
    f"  AS '{ROW_SINK_MODULE}', 'publish' LANGUAGE wasm;\n"
)


def _row_sink_described(
    *,
    name: str = "publish",
    video: str = "many",
    audio: str = "many",
    video_codecs: tuple[str, ...] = ("h264",),
    audio_codecs: tuple[str, ...] = ("aac",),
) -> Described:
    return Described(
        world=WORLDS[-1],
        name=name,
        version="0.1.0",
        params_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "relay": {"type": "string"},
                "broadcast": {"type": "string"},
            },
        },
        rows_schema=None,
        video_codecs=video_codecs,
        audio_codecs=audio_codecs,
        video_streams=video,  # type: ignore[arg-type]
        audio_streams=audio,  # type: ignore[arg-type]
    )


def _row_sink_graph(
    sql: str,
    probes: dict[str, ProbeResult | None],
    described: Described | None = None,
) -> Graph:
    return lower(
        resolve(parse(ROW_SINK_DECLARE + sql)),
        probes,
        registry=_snapshot_registry(),
        describes={ROW_SINK_MODULE: described or _row_sink_described()},
    )


def _row_sink_rejects(
    sql: str,
    probes: dict[str, ProbeResult | None],
    described: Described | None = None,
) -> FfrwdError:
    with pytest.raises(FfrwdError) as excinfo:
        _row_sink_graph(sql, probes, described)
    return _anchored(excinfo.value)


def test_a_sink_call_has_no_cell_for_a_table_query_to_print() -> None:
    """A sink CONSUMES the streams its column named, so the column carries
    none -- and a table query prints cells, not destinations."""
    with pytest.raises(FfrwdError) as excinfo:
        lower_table(
            resolve(
                parse(
                    ROW_SINK_DECLARE
                    + "COPY (SELECT r.video[1] FROM input('f.mkv') r) "
                    "TO publish('relay', 'live')"
                )
            ),
            {"r": ProbeResult(streams=[_track("video", 0)])},
            registry=_snapshot_registry(),
            describes={ROW_SINK_MODULE: _row_sink_described()},
        )
    err = _anchored(excinfo.value)
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "is a sink and prints no table" in err.message


def test_a_ladder_republished_through_a_row_reading_sink_is_five_pads() -> None:
    """2 muxed rungs (row 0, row 1) beside one audio-only rendition (row 2)
    -- 5 pads, row-major: row 0's video, row 0's audio, row 1's video, row
    1's audio, row 2's audio, each remembering its own row and the
    manifest's name for it."""
    g = _row_sink_graph(
        "COPY (SELECT r.video[1], r.audio[1] FROM input('ladder.m3u8') r) "
        "TO publish('relay', 'live')",
        _rendition_probes(),
    )
    node_id = next(iter(g.packet_sinks))
    node = g.nodes[node_id]
    assert node.inputs == [
        "src:r:v:0", "src:r:a:0", "src:r:v:1", "src:r:a:1", "src:r:a:2",
    ]
    pads = g.packet_sinks[node_id]
    assert [pad["row"] for pad in pads] == [0, 0, 1, 1, 2]
    names = [pad["rendition"]["name"] for pad in pads]  # type: ignore[index]
    assert names == ["Low", "Low", "High", "High", "Audio"]
    assert pads[4]["rendition"]["language"] == "en"  # type: ignore[index]
    assert "language" not in pads[0]["rendition"]  # type: ignore[operator]


def test_a_computed_ladder_names_its_rows_by_height() -> None:
    """No rendition table behind these rows -- ``scale`` over
    ``generate_series`` -- so each pad's rendition is derived exactly as
    ``var_stream_map`` names one: height for a video row, nothing else."""
    g = _row_sink_graph(
        "COPY (SELECT scale(f.video[1], ARRAY[640, 1280][i.i], -2) "
        "FROM input('a.mp4') f, generate_series(1, 2) i) "
        "TO publish('relay', 'live')",
        _row_probes(_track("video", 0, width=1920, height=1080)),
    )
    node_id = next(iter(g.packet_sinks))
    pads = g.packet_sinks[node_id]
    assert [pad["row"] for pad in pads] == [0, 1]
    names = [pad["rendition"]["name"] for pad in pads]  # type: ignore[index]
    assert names == ["360p", "720p"]


_DEMUXED_INTO_SINK = (
    "COPY (WITH vid AS (SELECT scale(f.video[1], ARRAY[1280, 640][i.i], -2) AS v, "
    "i.i AS rung FROM input('a.mp4') f, generate_series(1, 2) i), "
    "aud AS (SELECT a AS t, 2 + a.index AS rung FROM input('a.mp4') g, unnest(g.audio) a) "
    "SELECT vid.v, aud.t FROM vid FULL JOIN aud ON vid.rung = aud.rung) "
    "TO publish('relay', 'live') WITH (video_bitrate {rates}[vid.rung], audio_bitrate '128k')"
)


def _two_alias_probes() -> dict[str, ProbeResult | None]:
    """One file with a video and an audio track, read under two aliases."""
    probe = ProbeResult(
        streams=[_track("video", 0, width=1920, height=1080), _track("audio", 0)]
    )
    return {"f": probe, "g": probe}


def test_a_per_row_option_binds_to_the_rows_that_carry_its_kind() -> None:
    """The demuxed relation is three rows -- two video rungs and one audio
    row -- and a per-row video option read over all three binds to the two
    video pads, as it does at a manifest destination."""
    g = _row_sink_graph(
        _DEMUXED_INTO_SINK.format(rates="ARRAY['800k', '300k']"),
        _two_alias_probes(),
    )
    node_id = next(iter(g.packet_sinks))
    pads = g.packet_sinks[node_id]
    assert [pad["row"] for pad in pads] == [0, 1, 2]
    assert [pad.get("video_bitrate") for pad in pads] == ["800k", "300k", None]
    assert pads[2]["audio_bitrate"] == "128k"


def test_an_unmodified_source_stream_copies_onto_the_sink() -> None:
    """A ``src:`` cell in a codec the sink already accepts (h264/aac) copies
    instead of paying for an encode nothing asked for."""
    g = _row_sink_graph(
        "COPY (SELECT r.video[1], r.audio[1] FROM input('ladder.m3u8') r "
        "WHERE r.height = 720) TO publish('relay', 'live')",
        _rendition_probes(),
    )
    node_id = next(iter(g.packet_sinks))
    pads = g.packet_sinks[node_id]
    assert pads[0]["video_codec"] == "copy"
    assert pads[1]["audio_codec"] == "copy"


def test_a_filtered_cell_still_encodes_onto_the_sink() -> None:
    """A cell reached through a filter -- ``scale`` here -- is never a bare
    ``src:`` ref, so it always encodes; the untouched audio beside it still
    copies."""
    g = _row_sink_graph(
        "COPY (SELECT scale(r.video[1], 640, -2), r.audio[1] "
        "FROM input('ladder.m3u8') r WHERE r.height = 720) "
        "TO publish('relay', 'live')",
        _rendition_probes(),
    )
    node_id = next(iter(g.packet_sinks))
    pads = g.packet_sinks[node_id]
    assert pads[0]["video_codec"] == "libx264"
    assert pads[1]["audio_codec"] == "copy"


def test_an_unaccepted_codec_still_encodes_onto_the_sink() -> None:
    """The sink's own describe names what it accepts; a source in a codec
    outside that list still pays for the encode even though the cell is
    unmodified."""
    g = _row_sink_graph(
        "COPY (SELECT r.video[1], r.audio[1] FROM input('ladder.m3u8') r "
        "WHERE r.height = 720) TO publish('relay', 'live')",
        _rendition_probes(),
        _row_sink_described(video_codecs=("hevc",)),
    )
    node_id = next(iter(g.packet_sinks))
    pads = g.packet_sinks[node_id]
    assert pads[0]["video_codec"] == "libx265"
    assert pads[1]["audio_codec"] == "copy"


def test_a_value_column_into_a_row_reading_sink_names_its_type() -> None:
    """A column that is not a video or audio stream -- a plain subtitle
    track here, a rows projection reads back the same way -- is refused
    naming what it actually is, the way a manifest destination refuses
    one."""
    err = _row_sink_rejects(
        "COPY (SELECT f.video[1], f.subtitle[1] FROM input('f.mp4') f) "
        "TO publish('relay', 'live')",
        _row_probes(_track("video", 0), _track("subtitle", 0)),
    )
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert "takes video and audio columns" in err.message
    assert "is subtitle" in err.message
    assert "'publish'" in err.message


def test_a_one_row_sink_refuses_a_multi_row_relation() -> None:
    """`video_streams`/`audio_streams` neither declaring "many" means the
    sink is file-shaped -- the one-row rule again, naming the sink instead
    of a path."""
    err = _row_sink_rejects(
        "COPY (SELECT r.video[1], r.audio[1] FROM input('ladder.m3u8') r) "
        "TO publish('relay', 'live')",
        _rendition_probes(),
        _row_sink_described(video="one", audio="one"),
    )
    assert err.code is ErrorCode.ROW_COUNT_MISMATCH
    assert err.message == "this query has 3 rows, and 'publish()' reads one"


def test_a_many_video_one_audio_sink_still_refuses_three_audio_rows() -> None:
    """Either kind declaring "many" is enough to admit several rows -- the
    check is a single arity for the whole sink, not per kind."""
    g = _row_sink_graph(
        "COPY (SELECT r.video[1], r.audio[1] FROM input('ladder.m3u8') r) "
        "TO publish('relay', 'live')",
        _rendition_probes(),
        _row_sink_described(video="many", audio="one"),
    )
    node_id = next(iter(g.packet_sinks))
    assert len(g.packet_sinks[node_id]) == 5


OLD_FORM_SINK_MODULE = "modules/tally.wasm"

OLD_FORM_SINK_DECLARE = (
    "CREATE FUNCTION tally(v video_stream) RETURNS sink\n"
    f"  AS '{OLD_FORM_SINK_MODULE}', 'tally' LANGUAGE wasm;\n"
)


def test_the_old_sink_form_is_unaffected() -> None:
    """A ``RETURNS sink`` declaring a stream parameter keeps reading its pad
    from the SELECT's own arguments, not the relation's rows: no `row` or
    `rendition` key reaches its pad dict, and a multi-row relation still
    hits the ORDINARY one-row-per-file rule rather than the sink's own
    arity check, exactly as before this change."""
    g = lower(
        resolve(parse(
            OLD_FORM_SINK_DECLARE
            + "COPY (SELECT f.video[1] FROM input('a.mp4') f) TO tally()"
        )),
        _row_probes(),
        registry=_snapshot_registry(),
        describes={OLD_FORM_SINK_MODULE: _row_sink_described(name="tally", video="one", audio="none")},
    )
    node_id = next(iter(g.packet_sinks))
    (pad,) = g.packet_sinks[node_id]
    assert "row" not in pad
    assert "rendition" not in pad
    two_videos = _row_probes(_track("video", 0), _track("video", 1))
    with pytest.raises(FfrwdError) as excinfo:
        lower(
            resolve(parse(
                OLD_FORM_SINK_DECLARE
                + "COPY (SELECT array_agg(t) FROM input('f.mkv') f, unnest(f.video) t) "
                "TO tally()"
            )),
            two_videos,
            registry=_snapshot_registry(),
            describes={OLD_FORM_SINK_MODULE: _row_sink_described(name="tally", video="one", audio="none")},
        )
    err = _anchored(excinfo.value)
    assert err.code is ErrorCode.UDF_ARG_TYPE
    assert "reads 1 stream" in err.message
