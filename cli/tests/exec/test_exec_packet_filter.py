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
import shutil
import subprocess
from pathlib import Path

import pytest

from ffrwd import binaries, wasm
from ffrwd.compiler import compile_all
from ffrwd.emit import build_ffmpeg_args, emit
from ffrwd.execute import execute_plan

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
