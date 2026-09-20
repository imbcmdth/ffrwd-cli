"""End-to-end exec tests for a packet filter placed by a COPY's destination.

Marked ``@pytest.mark.exec`` and excluded from the default run. Run
explicitly::

    python -m pytest -m exec tests/exec/test_exec_packet_filter.py -q

Everything is real: the query compiles against the installed ffmpeg and the
sidecar's own ``--describe``, and ``execute_plan`` spawns the whole pipeline
-- an ffmpeg that encodes (or copies), ``ffrwd-wasm`` hosting the filter, and
an ffmpeg that muxes what comes back.

The module under test is ``packet-passthrough``, whose ``process`` hands
every packet back unchanged on a one-call lag. That makes it an IDENTITY,
which is what lets every assertion here be a comparison against THE SAME
QUERY WITHOUT THE FILTER rather than against a number someone typed: same
packet times, same durations, same pictures. A difference is the placement's
fault, since the filter itself changed nothing.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from ffrwd import binaries, wasm
from ffrwd.compiler import compile_all
from ffrwd.emit import build_ffmpeg_args, emit
from ffrwd.execute import execute_plan, render_plan

pytestmark = pytest.mark.exec

_CLI_ROOT = Path(__file__).resolve().parent.parent.parent
_REPO_ROOT = _CLI_ROOT.parent
_FIXTURES_DIR = _CLI_ROOT / "tests" / "fixtures"
_SIDECAR_MODULES = _REPO_ROOT / "sidecar" / "modules"
_BUILT = _SIDECAR_MODULES / "target" / "wasm32-wasip2" / "release"
_PASSTHROUGH = _BUILT / "packet_passthrough.wasm"
_PACKET_STATS = _BUILT / "packet_stats.wasm"
_AV = _FIXTURES_DIR / "av.mp4"
_TIMEOUT = 120.0

_DECLARE = (
    "CREATE FUNCTION hand_on(v video_stream) RETURNS packets\n"
    f"  AS '{_PASSTHROUGH.as_posix()}', 'packet_passthrough' LANGUAGE wasm;\n"
)


@pytest.fixture(autouse=True)
def _require_everything() -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg/ffprobe not found on PATH")
    if not _AV.exists():
        pytest.skip(f"fixture missing: {_AV} (run scripts/gen_fixtures.py first)")
    if binaries.ffrwd_wasm_path() is None:
        pytest.skip("the ffrwd-wasm sidecar is not installed")
    if not _PASSTHROUGH.exists():
        pytest.skip(
            f"module missing: {_PASSTHROUGH} (cargo build --target "
            f"wasm32-wasip2 --release, from {_SIDECAR_MODULES})"
        )


# -- what the two files are compared on -----------------------------------


def _ffprobe(*args: str) -> dict[str, object]:
    done = subprocess.run(
        ["ffprobe", "-v", "error", "-of", "json", *args],
        capture_output=True,
        text=True,
        timeout=_TIMEOUT,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    result: dict[str, object] = json.loads(done.stdout)
    return result


def _packet_times(path: Path, kind: str) -> list[tuple[str, str, str]]:
    """Every packet of the first `kind` stream: its pts, dts and duration.

    In SECONDS, not ticks: a stream that crosses a NUT pipe is muxed from
    the time base the pipe carried, which need not be the one a direct mux
    would have picked. What has to agree is when each packet plays, not what
    integer the container spells that with.
    """
    read = _ffprobe(
        "-select_streams",
        f"{kind}:0",
        "-show_entries",
        "packet=pts_time,dts_time,duration_time",
        str(path),
    )
    packets = read["packets"]
    assert isinstance(packets, list)
    return [
        (
            str(p.get("pts_time")),
            str(p.get("dts_time")),
            str(p.get("duration_time")),
        )
        for p in packets
    ]


def _stream_facts(path: Path, kind: str) -> dict[str, object]:
    """The first `kind` stream's codec, start time and duration."""
    read = _ffprobe(
        "-select_streams",
        f"{kind}:0",
        "-show_entries",
        "stream=codec_name,start_time,duration,nb_read_packets",
        "-count_packets",
        str(path),
    )
    streams = read["streams"]
    assert isinstance(streams, list) and streams, f"{path} has no {kind} stream"
    facts: dict[str, object] = streams[0]
    return facts


def _framemd5(path: Path, kind: str) -> list[str]:
    """One md5 per decoded frame of the first `kind` stream.

    The picture-level comparison: two files whose framemd5s agree carry the
    same pictures however they were muxed.
    """
    done = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-i", str(path),
            "-map", f"0:{kind}:0", "-f", "framemd5", "-",
        ],
        capture_output=True,
        text=True,
        timeout=_TIMEOUT,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    return [line for line in done.stdout.splitlines() if line and not line.startswith("#")]


def _run(sql: str) -> None:
    """Run one query, whether it needs the sidecar or one plain ffmpeg.

    The unfiltered half of every comparison here is one ffmpeg command and
    carries no plan at all, which is the whole point of comparing to it.
    """
    compiled = compile_all(sql)
    if compiled.plan is None:
        (graph,) = compiled.graphs
        argv = build_ffmpeg_args(emit(graph))
        done = subprocess.run(
            [argv[0], "-y", *argv[1:]],
            capture_output=True,
            text=True,
            timeout=_TIMEOUT,
            check=False,
        )
        assert done.returncode == 0, done.stderr
        return
    result = execute_plan(
        compiled.plan,
        sidecar_argv=wasm.sidecar_argv,
        overwrite=True,
        timeout=_TIMEOUT,
    )
    assert result.exit_code == 0, "\n".join(
        f"{member.id} exited {member.exit_code}: {member.stderr_tail}"
        for stage in result.stages
        for member in stage.members
    )
    assert not result.timed_out


def _pair(tmp_path: Path, select: str, suffix: str, options: str = "") -> tuple[Path, Path]:
    """The same COPY written twice: once through the filter, once without.

    `select` names the video column with a ``{v}`` placeholder, so the two
    queries differ in exactly the filter call and nothing else.
    """
    filtered = tmp_path / f"filtered{suffix}"
    plain = tmp_path / f"plain{suffix}"
    for path, column in ((filtered, "hand_on(f.video[1])"), (plain, "f.video[1]")):
        # The declaration goes only with the call: a function nothing calls is
        # itself a rejection.
        _run(
            (_DECLARE if "hand_on" in column else "")
            + "COPY (\n"
            + f"  SELECT {select.format(v=column)}\n"
            + f"  FROM input('{_AV.as_posix()}') f\n"
            + f") TO '{path.as_posix()}'{options}"
        )
    return filtered, plain


# -- the tests -------------------------------------------------------------


def test_a_filter_over_a_copied_stream_writes_the_same_file(tmp_path: Path) -> None:
    """No WITH options, so the stream reaches the filter as the packets it
    already was and leaves the same way: an MKV byte-compatible with the plain
    remux, packet time for packet time, on both streams."""
    filtered, plain = _pair(tmp_path, "{v}, f.audio[1]", ".mkv")

    assert _packet_times(filtered, "v") == _packet_times(plain, "v")
    assert _packet_times(filtered, "a") == _packet_times(plain, "a")
    assert _framemd5(filtered, "v") == _framemd5(plain, "v")


def test_a_filter_over_an_encoded_stream_keeps_audio_in_sync(tmp_path: Path) -> None:
    """The encoder moves one process ahead and the muxer copies what comes
    back, while the audio is muxed straight off the source: an MP4 whose
    streams start, run and end exactly as the unfiltered query's do."""
    filtered, plain = _pair(
        tmp_path,
        "{v}, f.audio[1]",
        ".mp4",
        " WITH (video_codec 'libx264', crf 28, preset 'ultrafast', audio_codec 'aac')",
    )

    for kind in ("v", "a"):
        assert _stream_facts(filtered, kind) == _stream_facts(plain, kind), kind
        assert _packet_times(filtered, kind) == _packet_times(plain, kind), kind
    assert _framemd5(filtered, "v") == _framemd5(plain, "v")
    assert _framemd5(filtered, "a") == _framemd5(plain, "a")


def test_a_filter_over_a_video_only_copy_leaves_the_pictures_alone(
    tmp_path: Path,
) -> None:
    """The one-stream case, at an MP4 destination: nothing decodes anywhere,
    and the pictures that come out are the ones that went in."""
    filtered, plain = _pair(tmp_path, "{v}", ".mp4")

    assert _stream_facts(filtered, "v") == _stream_facts(plain, "v")
    assert _framemd5(filtered, "v") == _framemd5(plain, "v")


def test_a_filter_in_front_of_a_packet_sink_hands_it_every_packet(
    tmp_path: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    """A `RETURNS sink` destination reads the filter's output rather than the
    encoder's, and counts exactly what it would have counted without it."""
    if not _PACKET_STATS.exists():
        pytest.skip(
            f"module missing: {_PACKET_STATS} (cargo build --target "
            f"wasm32-wasip2 --release, from {_SIDECAR_MODULES})"
        )
    sink_declare = (
        "CREATE FUNCTION packet_stats(v video_stream) RETURNS sink\n"
        f"  AS '{_PACKET_STATS.as_posix()}', 'packet_stats' LANGUAGE wasm;\n"
    )
    counted: list[int] = []
    for column in ("hand_on(f.video[1])", "f.video[1]"):
        capfd.readouterr()
        _run(
            (_DECLARE if "hand_on" in column else "")
            + sink_declare
            + "COPY (\n"
            + f"  SELECT {column}\n"
            + f"  FROM input('{_AV.as_posix()}') f\n"
            + ") TO packet_stats() WITH (video_codec 'libx264', gop 5)"
        )
        rows = [
            json.loads(line)
            for line in capfd.readouterr().out.splitlines()
            if line.strip()
        ]
        assert rows, "the sink's rows never reached stdout"
        counted.append(int(rows[-1]["packets"]))

    assert counted[0] == counted[1]


def test_the_compiled_shape_is_encoder_filter_muxer() -> None:
    """Three processes, in that order, and the muxer opens the source itself
    for the streams the filter never saw."""
    compiled = compile_all(
        _DECLARE
        + "COPY (\n"
        + "  SELECT hand_on(f.video[1]), f.audio[1]\n"
        + f"  FROM input('{_AV.as_posix()}') f\n"
        + ") TO 'out.mp4' WITH (video_codec 'libx264', crf 28)"
    )
    plan = compiled.plan
    assert plan is not None
    (sidecar,) = plan.sidecars
    assert sidecar.packet_filter
    assert len(plan.ffmpeg) == 2
    encoder = next(e for e in plan.stream_edges if e.target == sidecar.id)
    muxed = next(e for e in plan.stream_edges if e.source == sidecar.id)
    assert encoder.format.codec == "libx264"
    assert muxed.format.codec == "copy"
    # The audio never travels a pipe: the muxer reads the file for it.
    muxer = next(p for p in plan.ffmpeg if p.id == muxed.target)
    assert _AV.as_posix() in [Path(p).as_posix() for p in muxer.graph.input_paths]


# -- a filter's ROWS: written in one stage, read in the next ---------------

_PACKET_SEI = _BUILT / "packet_sei.wasm"
_NOTE_ROWS = _BUILT / "note_rows.wasm"

_NOTES = (
    "CREATE FUNCTION notes(v video_stream, label text DEFAULT 'note')\n"
    "RETURNS STRUCT(v video_stream, seen STRUCT(pts number, note text)[])\n"
    f"  AS '{_NOTE_ROWS.as_posix()}', 'note_rows' LANGUAGE wasm;\n"
)
_WEAVE_ONE = (
    "CREATE FUNCTION weave(v video_stream,\n"
    "                      seen STRUCT(pts number, note text)[])\n"
    "  RETURNS packets\n"
    f"  AS '{_PACKET_SEI.as_posix()}', 'packet_sei' LANGUAGE wasm;\n"
)
_WEAVE_TWO = (
    "CREATE FUNCTION weave(v video_stream,\n"
    "                      faces STRUCT(pts number, note text)[],\n"
    "                      words STRUCT(pts number, note text)[])\n"
    "  RETURNS packets\n"
    f"  AS '{_PACKET_SEI.as_posix()}', 'packet_sei' LANGUAGE wasm;\n"
)


def _require_weaving() -> None:
    for module in (_PACKET_SEI, _NOTE_ROWS):
        if not module.exists():
            pytest.skip(
                f"module missing: {module} (cargo build --target wasm32-wasip2 "
                f"--release, from {_SIDECAR_MODULES})"
            )


def _video_bytes(path: Path) -> bytes:
    """The file's video stream, copied out, for reading what rode in it."""
    done = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-i", str(path),
            "-map", "0:v:0", "-c", "copy", "-f", "data", "-",
        ],
        capture_output=True,
        timeout=_TIMEOUT,
        check=False,
    )
    assert done.returncode == 0, done.stderr.decode(errors="replace")
    return done.stdout


def _woven(path: Path, text: str) -> int:
    """How many times `text` appears in the file's video bytes."""
    return _video_bytes(path).count(text.encode())


def test_rows_written_in_one_stage_reach_the_filter_in_the_next(tmp_path: Path) -> None:
    """The producer's rows go to a document, the document orders the stages,
    and the filter weaves what it read into packets it copied through."""
    _require_weaving()
    out = tmp_path / "woven.mp4"
    _run(
        _NOTES
        + _WEAVE_ONE
        + "COPY (\n"
        + "  SELECT weave(f.video[1], notes(f.video[1]).seen), f.audio[1]\n"
        + f"  FROM input('{_AV.as_posix()}') f\n"
        + f") TO '{out.as_posix()}'"
    )
    plain = tmp_path / "plain.mp4"
    _run(
        "COPY (\n"
        + "  SELECT f.video[1], f.audio[1]\n"
        + f"  FROM input('{_AV.as_posix()}') f\n"
        + f") TO '{plain.as_posix()}'"
    )

    assert _woven(out, "note-0") == 1, "the first note never reached the stream"
    # A note rides the packet; it does not move one. The audio the filter
    # never saw is untouched either.
    assert _packet_times(out, "a") == _packet_times(plain, "a")
    assert _framemd5(out, "v") == _framemd5(plain, "v")
    assert _framemd5(out, "a") == _framemd5(plain, "a")
    # ffmpeg decodes it without complaint, which an ill-framed NAL would not.
    assert _stream_facts(out, "v")["nb_read_packets"] == _stream_facts(plain, "v")[
        "nb_read_packets"
    ]


def test_an_encoded_weave_reaches_every_keyframe_at_an_mkv(tmp_path: Path) -> None:
    """The re-encoding half, at a second container: a short gop gives the
    filter a keyframe to put nearly every note on, and the pictures are the
    unfiltered encode's own."""
    _require_weaving()
    options = " WITH (video_codec 'libx264', gop 5, preset 'ultrafast', audio_codec 'aac')"
    out = tmp_path / "woven.mkv"
    _run(
        _NOTES
        + _WEAVE_ONE
        + "COPY (\n"
        + "  SELECT weave(f.video[1], notes(f.video[1]).seen), f.audio[1]\n"
        + f"  FROM input('{_AV.as_posix()}') f\n"
        + f") TO '{out.as_posix()}'{options}"
    )
    plain = tmp_path / "plain.mkv"
    _run(
        "COPY (\n"
        + "  SELECT f.video[1], f.audio[1]\n"
        + f"  FROM input('{_AV.as_posix()}') f\n"
        + f") TO '{plain.as_posix()}'{options}"
    )

    written = _video_bytes(out)
    assert written.count(b"note-") >= 5, "few notes reached the stream"
    assert _framemd5(out, "v") == _framemd5(plain, "v")
    assert _packet_times(out, "v") == _packet_times(plain, "v")
    assert _packet_times(out, "a") == _packet_times(plain, "a")


def test_two_rows_arguments_arrive_tagged_with_the_one_they_filled(
    tmp_path: Path,
) -> None:
    """Two producers, two documents, two `-rows-in`: the host writes the
    argument's name onto every row, and the module weaves it in, so the
    stream itself says which argument each note came from."""
    _require_weaving()
    out = tmp_path / "two.mp4"
    _run(
        _NOTES
        + _WEAVE_TWO
        + "COPY (\n"
        + "  SELECT weave(f.video[1],\n"
        + "               notes(f.video[1], 'a').seen,\n"
        + "               notes(ffmpeg.hflip(f.video[1]), 'b').seen)\n"
        + f"  FROM input('{_AV.as_posix()}') f\n"
        + f") TO '{out.as_posix()}'"
    )

    written = _video_bytes(out)
    assert b"faces:a-0" in written, "the first argument's note is not tagged with it"
    assert b"words:b-0" in written, "the second argument's note is not tagged with it"


_EMBED_NOTES = _BUILT / "embed_notes.wasm"

_EMBED = (
    "CREATE FUNCTION embed(rows STRUCT(pts number, note text)[])\n"
    "RETURNS STRUCT(pts number, note text, vector vector)[]\n"
    f"  AS '{_EMBED_NOTES.as_posix()}', 'embed_notes' LANGUAGE wasm;\n"
)


def test_a_rows_argument_may_arrive_through_a_rows_module(tmp_path: Path) -> None:
    """One argument straight off the producing module and one through a rows
    module over the same shape. The rows module adds a vector beside each row
    and marks the note, so the bytes in the stream say which of the two routes
    each note took -- and the filter reads both documents the same way."""
    _require_weaving()
    if not _EMBED_NOTES.exists():
        pytest.skip(
            f"module missing: {_EMBED_NOTES} (cargo build --target "
            f"wasm32-wasip2 --release, from {_SIDECAR_MODULES})"
        )
    out = tmp_path / "embedded.mp4"
    _run(
        _NOTES
        + _EMBED
        + _WEAVE_TWO
        + "COPY (\n"
        + "  SELECT weave(f.video[1],\n"
        + "               notes(f.video[1], 'a').seen,\n"
        + "               embed(notes(ffmpeg.hflip(f.video[1]), 'b').seen))\n"
        + f"  FROM input('{_AV.as_posix()}') f\n"
        + f") TO '{out.as_posix()}'"
    )

    written = _video_bytes(out)
    assert b"faces:a-0" in written, "the straight argument's note is not tagged"
    # `e-` is what the rows module puts in front of every note it read.
    assert b"words:e-b-0" in written, "the note never went through the rows module"


def test_rows_arguments_named_by_a_cte_reach_the_filter(tmp_path: Path) -> None:
    """The producers are written once in a WITH body and read by name in the
    COPY's SELECT. An alias of an accepted producer expression is that
    expression: the same two notes reach the stream the same way."""
    _require_weaving()
    if not _EMBED_NOTES.exists():
        pytest.skip(f"module missing: {_EMBED_NOTES}")
    out = tmp_path / "named.mp4"
    _run(
        _NOTES
        + _EMBED
        + _WEAVE_TWO
        + "COPY (\n"
        + "  WITH d AS (\n"
        + "    SELECT f.video[1] AS v,\n"
        + "           notes(f.video[1], 'a').seen AS clip,\n"
        + "           embed(notes(ffmpeg.hflip(f.video[1]), 'b').seen) AS spoken\n"
        + f"    FROM input('{_AV.as_posix()}') f\n"
        + "  )\n"
        + "  SELECT weave(d.v, d.clip, d.spoken)\n"
        + "  FROM d\n"
        + f") TO '{out.as_posix()}'"
    )

    written = _video_bytes(out)
    assert b"faces:a-0" in written
    assert b"words:e-b-0" in written


def test_a_rows_module_writes_the_document_the_filter_reads() -> None:
    """The chain is planned as what it is: the producer feeds the rows module
    over a rows edge inside one sidecar, and the document the filter reads is
    the rows module's own output."""
    _require_weaving()
    if not _EMBED_NOTES.exists():
        pytest.skip(f"module missing: {_EMBED_NOTES}")
    compiled = compile_all(
        _NOTES
        + _EMBED
        + _WEAVE_ONE
        + "COPY (\n"
        + "  SELECT weave(f.video[1], embed(notes(f.video[1]).seen))\n"
        + f"  FROM input('{_AV.as_posix()}') f\n"
        + ") TO 'out.mp4'"
    )
    plan = compiled.plan
    assert plan is not None
    shown = render_plan(plan, sidecar_argv=wasm.shown_argv)
    assert "-m " + _EMBED_NOTES.as_posix() + " -rows-from 0" in shown
    assert "-f ndjson ffrwd:rows:0" in shown
    assert "-rows-in seen=ffrwd:rows:0" in shown
    # Still two stages: a document is a file, and a file is finished before
    # whatever reads it starts.
    assert len(plan.stages) == 2


def test_a_rows_document_is_a_placeholder_until_a_run_resolves_it() -> None:
    """A compile prints the same text on every machine: the document is
    named `ffrwd:rows:<n>` at both ends, and the run is what turns it into a
    file in its own temporary directory."""
    _require_weaving()
    compiled = compile_all(
        _NOTES
        + _WEAVE_ONE
        + "COPY (\n"
        + "  SELECT weave(f.video[1], notes(f.video[1]).seen)\n"
        + f"  FROM input('{_AV.as_posix()}') f\n"
        + ") TO 'out.mp4'"
    )
    plan = compiled.plan
    assert plan is not None
    shown = render_plan(plan, sidecar_argv=wasm.shown_argv)
    assert "-f ndjson ffrwd:rows:0" in shown
    assert "-rows-in seen=ffrwd:rows:0" in shown
    # Two stages: the document is a file, and a file is finished before what
    # reads it starts.
    assert len(plan.stages) == 2
    (document,) = plan.file_edges
    assert document.format.path == "ffrwd:rows:0"
    assert document.source in plan.stages[0].processes
    assert document.target in plan.stages[1].processes


# -- the real index package, when its artefacts are here -------------------
#
# Gated on two environment variables rather than skipped by discovery: the
# `ffrwd/index` package is a repository of its own, and this suite neither
# builds it nor knows where it lives.
#
#   FFRWD_INDEX_WEAVE  the built weave.wasm, against THIS sidecar's wit
#   FFRWD_INDEX_TOOL   the built ffrwd-index binary
#
# What it proves is the placement, not the format: vectors for two spaces
# reach one filter through two rows arguments, the filter is hosted by a
# stream copy so no picture is touched, and the tool reads every record back
# out of the MP4 with the spans the query put in.

_INDEX_WEAVE = os.environ.get("FFRWD_INDEX_WEAVE", "")
_INDEX_TOOL = os.environ.get("FFRWD_INDEX_TOOL", "")


def _index_spaces_param() -> str | None:
    """The `spaces` parameter weave declares, if the dialect can fill it.

    A value parameter is a text, number, boolean or vector, so a params
    schema asking for an array of objects is one no query can supply. That
    is a property of the package, not of this suite, and reading it off the
    module's own describe is what keeps this test skipping for a reason it
    can state rather than failing for one it cannot.
    """
    described = wasm.describe(_INDEX_WEAVE)
    schema = described.params_schema or {}
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return None
    member = properties.get("spaces")
    kind = member.get("type") if isinstance(member, dict) else None
    return str(kind) if isinstance(kind, str) else None


def _require_index_package() -> None:
    for name, path in (
        ("FFRWD_INDEX_WEAVE", _INDEX_WEAVE),
        ("FFRWD_INDEX_TOOL", _INDEX_TOOL),
    ):
        if not path or not Path(path).exists():
            pytest.skip(f"{name} does not name a built artefact")
    if not _NOTE_ROWS.exists():
        pytest.skip(f"module missing: {_NOTE_ROWS}")
    kind = _index_spaces_param()
    if kind not in (None, "string", "number", "boolean"):
        pytest.skip(
            f"weave declares 'spaces' as {kind}, and a wasm function's value "
            "parameters are text, number, boolean or vector -- no query can "
            "configure it until the package takes scalars"
        )


def test_vectors_for_two_spaces_reach_the_real_weave_and_read_back(
    tmp_path: Path,
) -> None:
    """Two rows arguments into `ffrwd/index`'s own filter, hosted by a stream
    copy on an MP4, and `ffrwd-index read --mp4` gets every record back."""
    _require_index_package()
    out = tmp_path / "indexed.mp4"
    record = "STRUCT(space text, start_t number, end_t number, vector vector)[]"
    _run(
        "CREATE FUNCTION vecs(v video_stream, space text, dims number)\n"
        f"RETURNS STRUCT(v video_stream, out {record})\n"
        f"  AS '{_NOTE_ROWS.as_posix()}', 'note_rows' LANGUAGE wasm;\n"
        "CREATE FUNCTION weave(v video_stream,\n"
        f"                      clip {record},\n"
        f"                      speech {record},\n"
        "                      spaces text)\n"
        "RETURNS packets\n"
        f"  AS '{Path(_INDEX_WEAVE).as_posix()}', 'weave' LANGUAGE wasm;\n"
        "COPY (\n"
        "  SELECT weave(f.video[1],\n"
        "               vecs(f.video[1], 'clip', 8).out,\n"
        "               vecs(ffmpeg.hflip(f.video[1]), 'speech', 8).out,\n"
        "               '[{\"name\":\"clip\",\"dims\":8},"
        "{\"name\":\"speech\",\"dims\":8}]')\n"
        f"  FROM input('{_AV.as_posix()}') f\n"
        f") TO '{out.as_posix()}'"
    )

    done = subprocess.run(
        [_INDEX_TOOL, "read", "--mp4", str(out)],
        capture_output=True,
        text=True,
        timeout=_TIMEOUT,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    rows = [json.loads(line) for line in done.stdout.splitlines() if line.strip()]
    records = [row for row in rows if "vector" in row]
    assert records, f"the tool read no record back:\n{done.stdout}"
    spaces = {row.get("space") for row in rows if "space" in row}
    assert len(spaces) >= 1
    # Every record's span is the one the producer wrote: one second long.
    for row in records:
        assert row["end_t"] - row["start_t"] == pytest.approx(1.0, abs=0.1)


# -- both halves of the packet surface in one query ------------------------

_PACKET_KEYS = _BUILT / "packet_keys.wasm"
_KEYS = _FIXTURES_DIR / "keys.mkv"


def test_a_compile_time_read_and_a_packet_filter_in_one_query(tmp_path: Path) -> None:
    """The find-then-weave shape: a packet sink read in FROM picks the spans
    while the query compiles, and a packet filter weaves rows into what the
    encoder made of them.

    The two use the packet surface at opposite ends -- one before ffmpeg
    runs, one in the middle of the run -- and this is what says they compose:
    the trims are already numbers in the filtergraph, the rows document is
    its own stage, and the filter sits behind the encoder as always.
    """
    _require_weaving()
    for fixture in (_PACKET_KEYS, _KEYS):
        if not fixture.exists():
            pytest.skip(f"missing: {fixture}")
    out = tmp_path / "found.mp4"
    keys = (
        "CREATE FUNCTION packet_keys(v video_stream)\n"
        "  RETURNS STRUCT(index number, start_t number, keyframe boolean,\n"
        "                 bytes number, vector vector)[]\n"
        f"  AS '{_PACKET_KEYS.as_posix()}', 'packet_keys' LANGUAGE wasm;\n"
    )
    trim = "ffmpeg.trim(f.video[1], start => v.start_t, duration => 0.4)"
    sql = (
        keys
        + _NOTES
        + _WEAVE_ONE
        + "COPY (\n"
        + f"  SELECT weave(concat(VARIADIC array_agg({trim})),\n"
        + "               notes(f.video[1]).seen)\n"
        + f"  FROM input('{_KEYS.as_posix()}') f, packet_keys(f.video[1]) v\n"
        + "  WHERE v.start_t > 3\n"
        + ") TO '" + out.as_posix() + "' WITH (video_codec 'libx264', gop 5)"
    )

    compiled = compile_all(sql)
    plan = compiled.plan
    assert plan is not None
    # The compile-time read already happened: its rows are numbers in the
    # graph, not a process. 3.266 is the keyframe packet_keys found past 3s.
    starts = {
        value
        for process in plan.ffmpeg
        for node in process.graph.nodes.values()
        if node.filter == "trim"
        for name, value in node.args.items()
        if name == "start"
    }
    assert starts == {3.266, 3.733}, starts
    # Two stages, the rows document between them, the filter in the second.
    assert len(plan.stages) == 2
    (document,) = plan.file_edges
    assert document.format.content == "rows"

    _run(sql)
    assert _woven(out, "note-") >= 1, "no note reached the stream"
    assert int(str(_stream_facts(out, "v")["nb_read_packets"])) > 0
