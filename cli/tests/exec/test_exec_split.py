"""End-to-end exec tests for a plan run split across nodes on this machine.

Marked ``@pytest.mark.exec`` and excluded from the default run. Run
explicitly::

    python -m pytest -m exec tests/exec/test_exec_split.py -q

Each case is a query the exec tier already runs, run three times: on one
node, placed per module, and placed per process, each node a runner of its
own (``ffrwd run --target split-local``) with its cut edges over loopback
TCP. A split changes where the processes run and nothing they are handed,
so every packet each run writes is the same: the same stream, time, size
and bytes, packet for packet. The files themselves are not compared byte
for byte, since Matroska writes a random segment id into each.

The whole exec tier runs split as well under ``FFRWD_TEST_SPLIT`` (see
conftest.py); this file is what compares a split run with a run on one.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest

from ffrwd import binaries, cli, nodes, wasm
from ffrwd.compiler import compile_all
from ffrwd.placement import place, split

pytestmark = pytest.mark.exec

_CLI_ROOT = Path(__file__).resolve().parent.parent.parent
_FIXTURES = _CLI_ROOT / "tests" / "fixtures"
_DATA = _CLI_ROOT / "tests" / "data"
_BUILT = _CLI_ROOT.parent / "sidecar" / "modules" / "target" / "wasm32-wasip2" / "release"
_AV = _FIXTURES / "av.mp4"
_TESTSRC = _FIXTURES / "testsrc.mp4"
_TIMEOUT = 120.0


def _declare(name: str, params: str, returns: str, module: str, export: str) -> str:
    return (
        f"CREATE FUNCTION {name}({params}) RETURNS {returns} "
        f"AS '{(_BUILT / module).as_posix()}', '{export}' LANGUAGE wasm;\n"
    )


_INVERT = _declare("invert", "v video_stream", "video_stream", "invert.wasm", "invert")
_STAMP = _declare(
    "stamp", "d data_stream, node text", "data_stream", "data_stamp.wasm", "data_stamp"
)
_HAND_ON = _declare(
    "hand_on", "v video_stream", "packets", "packet_passthrough.wasm", "packet_passthrough"
)
_ENCODER = _declare("enc", "level number DEFAULT NULL", "encoder", "testcodec.wasm", "encode")
_PROBE = (
    "CREATE FUNCTION probe(v video_stream, feed video_stream DEFAULT NULL, "
    "port number DEFAULT 9000)\nRETURNS STRUCT(v video_stream, feeds "
    "STRUCT(feed_pts number, w number, h number)[])\n"
    f"AS '{(_BUILT / 'feed_probe.wasm').as_posix()}', 'feed-probe' LANGUAGE wasm;\n"
)


@dataclass(frozen=True)
class _Case:
    """One query, the modules it needs, and the files it writes (by name)."""

    modules: tuple[str, ...]
    outputs: tuple[str, ...]
    query: Callable[[Path, dict[str, Path]], str]
    fixtures: tuple[Path, ...] = (_AV,)


def _chain(where: Path, out: dict[str, Path]) -> str:
    """One region: decode, a module, encode, a stdio chain end to end."""
    return (
        _INVERT
        + f"COPY (SELECT invert(f.video[1]) FROM input('{_AV.as_posix()}') f) "
        f"TO '{out['video.mkv'].as_posix()}' WITH (video_codec 'ffv1')"
    )


def _two_regions(where: Path, out: dict[str, Path]) -> str:
    """Two regions with an ffmpeg between them, and the sound beside them:
    the muxer reads two pipes, so its ends are named pipes."""
    return (
        _INVERT
        + "COPY (SELECT invert(scale(invert(f.video[1]), 160, 120)), f.audio[1] "
        f"FROM input('{_AV.as_posix()}') f) "
        f"TO '{out['av.mkv'].as_posix()}' WITH (video_codec 'ffv1', audio_codec 'flac')"
    )


def _data_filter(where: Path, out: dict[str, Path]) -> str:
    """A data filter's messages beside the picture they came with."""
    deal = _DATA / "deal.nut"
    return (
        _STAMP
        + f"COPY (SELECT f.video[1], stamp(f.data[1], 'es') FROM input('{deal.as_posix()}') f) "
        f"TO '{out['deal.nut'].as_posix()}'"
    )


def _packet_filter(where: Path, out: dict[str, Path]) -> str:
    """An encoder, a packet filter and the muxer, the sound muxed off the source."""
    return (
        _HAND_ON
        + f"COPY (SELECT hand_on(f.video[1]), f.audio[1] FROM input('{_AV.as_posix()}') f) "
        f"TO '{out['filtered.mp4'].as_posix()}' "
        "WITH (video_codec 'libx264', crf 28, preset 'ultrafast', audio_codec 'aac')"
    )


def _codec(where: Path, out: dict[str, Path]) -> str:
    """A codec package's encoder in the sidecar, its packets muxed by ffmpeg."""
    return (
        _ENCODER
        + f"COPY (SELECT f.video[1], f.audio[1] FROM input('{_AV.as_posix()}') f) "
        f"TO '{out['coded.nut'].as_posix()}' WITH (video_codec enc(level => 3), audio_codec 'aac')"
    )


def _live(where: Path, out: dict[str, Path]) -> str:
    """A paced input read by one process that writes each consumer a pipe."""
    return (
        _INVERT
        + "COPY (SELECT invert(p.video[1]), p.audio[1] "
        f"FROM input('{_AV.as_posix()}', realtime => true) p) "
        f"TO '{out['live.mkv'].as_posix()}' WITH (video_codec 'ffv1', audio_codec 'flac')"
    )


def _feeder(where: Path, out: dict[str, Path]) -> str:
    """A module reading a second stream over its loopback port, a feeder."""
    feeder = where / "feeder.mp4"
    if not feeder.exists():
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
             "testsrc2=size=160x120:rate=15:duration=1", "-pix_fmt", "yuv420p", str(feeder)],
            check=True,
            timeout=_TIMEOUT,
        )  # fmt: skip
    return (
        _PROBE
        + "COPY (SELECT probe(p.video[1], a.video[1]).feeds FROM "
        f"input('{_TESTSRC.as_posix()}') p, input('{feeder.as_posix()}') a) "
        f"TO '{out['feeds.ndjson'].as_posix()}'"
    )


_CASES = {
    "wasm region": _Case(("invert.wasm",), ("video.mkv",), _chain),
    "two regions and a fan-in": _Case(("invert.wasm",), ("av.mkv",), _two_regions),
    "data filter": _Case(
        ("data_stamp.wasm",), ("deal.nut",), _data_filter, (_DATA / "deal.nut",)
    ),
    "packet filter": _Case(("packet_passthrough.wasm",), ("filtered.mp4",), _packet_filter),
    "codec encode": _Case(("testcodec.wasm",), ("coded.nut",), _codec),
    "live reader": _Case(("invert.wasm",), ("live.mkv",), _live),
    "feeder": _Case(("feed_probe.wasm",), ("feeds.ndjson",), _feeder, (_TESTSRC,)),
}


def _require(case: _Case) -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg/ffprobe not found on PATH")
    if binaries.ffrwd_wasm_path() is None:
        pytest.skip("the ffrwd-wasm sidecar is not installed")
    for module in case.modules:
        if not (_BUILT / module).exists():
            pytest.skip(f"module missing: {module} (cargo build --target wasm32-wasip2 --release)")
    for fixture in case.fixtures:
        if not fixture.exists():
            pytest.skip(f"fixture missing: {fixture} (run scripts/gen_fixtures.py first)")


def _written(path: Path) -> list[str]:
    """What a run wrote: each packet's stream, times, size and hash, or a
    rows file's lines."""
    if path.suffix == ".ndjson":
        return path.read_text(encoding="utf-8").splitlines()
    done = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-map", "0", "-c", "copy",
         "-f", "framemd5", "-"],
        capture_output=True,
        text=True,
        timeout=_TIMEOUT,
        check=True,
    )  # fmt: skip
    return [line for line in done.stdout.splitlines() if not line.startswith("#")]


# A data edge's heartbeat, a one-byte packet saying only that time has moved
# on, as `_written` lists it: its size and its hash.
_HEARTBEAT = f"1, {hashlib.md5(b' ').hexdigest()}"


def _messages(lines: list[str]) -> list[str]:
    """`_written`'s lines without the heartbeats. Each writer of a data edge
    puts one out whenever it sees time move on with nothing written, so how
    many there are, and where, is the run's timing; the messages are not."""
    return [
        line for line in lines if not line.replace(" ", "").endswith(_HEARTBEAT.replace(" ", ""))
    ]


def _run(case: _Case, where: Path, *target: str) -> dict[str, list[str]]:
    run_as = where / (target[-1] if target else "one")
    run_as.mkdir(exist_ok=True)
    out = {name: run_as / name for name in case.outputs}
    sql = case.query(where, out)
    assert cli.main(["run", sql, "-y", "-q", *target]) == 0
    return {name: _written(path) for name, path in out.items()}


@pytest.mark.parametrize("name", sorted(_CASES))
def test_a_split_run_writes_every_packet_a_run_on_one_node_writes(
    name: str, tmp_path: Path
) -> None:
    case = _CASES[name]
    _require(case)
    plan = compile_all(case.query(tmp_path, {n: tmp_path / n for n in case.outputs})).plan
    assert plan is not None
    cuts = {
        how: sum(len(node.dials) for node in split(plan, place(plan, how)))
        for how in ("per-module", "per-process")
    }
    assert cuts["per-process"] > 0, "a case whose plan cannot be cut tests nothing here"

    one = _run(case, tmp_path)
    assert all(one.values()), f"the run on one node wrote nothing: {one}"
    for how in ("per-module", "per-process"):
        placed = _run(case, tmp_path, "--target", "split-local", "--placement", how)
        assert placed == one, f"{how} ({cuts[how]} cut edges) wrote something else"


def test_a_lateral_and_its_feeder_run_on_one_node_of_three(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The SMART head's shape with the fleet's modules: a paced programme into
    a module that reads a feeder, the feeder written per message by a
    run-time lateral's instances, and the messages stamped by a data filter
    on their way to the lateral and to the file. Placed per process, the
    lateral's writer and the module it feeds share a node; every other edge
    is cut. The rows the instances end with and every packet written are
    the ones a run on one node gives."""
    for module in ("feed_probe.wasm", "data_stamp.wasm"):
        if not (_BUILT / module).exists():
            pytest.skip(f"module missing: {module}")
    if binaries.ffrwd_wasm_path() is None:
        pytest.skip("the ffrwd-wasm sidecar is not installed")
    ffmpeg = ["ffmpeg", "-v", "error", "-y"]
    subprocess.run(
        [*ffmpeg, "-f", "lavfi", "-i", "testsrc2=size=160x120:rate=15:duration=1",
         "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=1",
         "-c:v", "ffv1", "-c:a", "pcm_s16le", str(tmp_path / "ad.mkv")],
        check=True, timeout=_TIMEOUT,
    )  # fmt: skip
    programme = tmp_path / "programme.mkv"
    subprocess.run(
        [*ffmpeg, "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=15:duration=6",
         "-c:v", "ffv1", str(programme)],
        check=True, timeout=_TIMEOUT,
    )  # fmt: skip
    monkeypatch.chdir(tmp_path)
    play = """CREATE FUNCTION play(launch data_stream, url text, start_pts number,
                     duration number, width number, height number, fps number,
                     pix_fmt text)
RETURNS TABLE(video video_stream) AS $$
  SELECT setpts(ffmpeg.format(fps(scale(m.video[1], width, height), fps),
                              pix_fmts => pix_fmt),
                'PTS+' || start_pts::text || '/TB') AS video
  FROM input(url) m
$$ LANGUAGE sql;
"""

    def query(out: Path) -> str:
        return (
            "CREATE FUNCTION probe(v video_stream, feed video_stream DEFAULT NULL, "
            "port number DEFAULT 9000) RETURNS video_stream "
            f"AS '{(_BUILT / 'feed_probe.wasm').as_posix()}', 'feed-probe' LANGUAGE wasm;\n"
            + _STAMP
            + play
            + "COPY (WITH w AS (SELECT stamp(f.data[1], 'leaf') AS s "
            f"FROM input('{(_DATA / 'launch.nut').as_posix()}') f), "
            "ads AS (SELECT ad.video FROM w, LATERAL play(w.s) ad) "
            "SELECT probe(p.video[1], ads.video), w.s "
            f"FROM input('{programme.as_posix()}', realtime => true) p, ads, w) "
            f"TO '{out.as_posix()}' WITH (video_codec 'ffv1')"
        )

    plan = compile_all(query(tmp_path / "x.nut")).plan
    assert plan is not None and plan.laterals
    placement = place(plan, "per-process")
    writer = plan.laterals[0].writer
    fed = {edge.target for edge in plan.feeder_edges}
    assert {placement.node(pid) for pid in fed} == {placement.node(writer)}
    assert placement.count >= 3

    written: dict[str, tuple[list[str], list[str]]] = {}
    for how in ("one", "per-process"):
        out = tmp_path / f"{how}.nut"
        target = [] if how == "one" else ["--target", "split-local", "--placement", how]
        capsys.readouterr()
        assert cli.main(["run", query(out), "-y", "-q", *target]) == 0
        rows = sorted(
            capsys.readouterr().out.splitlines(), key=lambda line: json.loads(line)["row"]
        )
        written[how] = (rows, _messages(_written(out)))
    assert written["per-process"] == written["one"]
    assert len(written["one"][0]) == 4


def test_a_runner_killed_mid_run_stops_every_node_and_is_named(tmp_path: Path) -> None:
    """A paced run on three nodes loses one node's runner: the others are
    stopped at once and the run names the node and what it ran."""
    case = _CASES["live reader"]
    _require(case)
    out = {"live.mkv": tmp_path / "live.mkv"}
    plan = compile_all(case.query(tmp_path, out)).plan
    assert plan is not None
    placement = place(plan, "per-process")
    assert placement.count >= 3
    runners: dict[int, subprocess.Popen[bytes]] = {}
    victim = placement.node(plan.sidecars[0].id)

    def kill() -> None:
        deadline = time.monotonic() + 30
        while victim not in runners and time.monotonic() < deadline:
            time.sleep(0.05)
        time.sleep(2.0)
        runners[victim].kill()

    killer = threading.Thread(target=kill)
    killer.start()
    started = time.monotonic()
    result = nodes.execute_split(
        plan,
        placement,
        sidecar_argv=wasm.sidecar_argv,
        timeout=_TIMEOUT,
        overwrite=True,
        started=lambda node, process: runners.__setitem__(node, process),
    )
    killer.join()
    assert time.monotonic() - started < 20
    assert result.exit_code != 0
    assert result.failure is not None
    assert result.failure.lost and result.failure.node == victim
    assert result.failure.id == plan.sidecars[0].id
    assert all(runner.poll() is not None for runner in runners.values())
