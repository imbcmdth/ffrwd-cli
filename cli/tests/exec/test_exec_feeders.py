"""End-to-end exec tests for a feeder argument.

Marked ``@pytest.mark.exec`` and excluded from the default run. Run
explicitly::

    python -m pytest -m exec tests/exec/test_exec_feeders.py -q

The module under test is ``feed-probe``: it passes its own stream through
untouched, listens on 127.0.0.1 at its `port` param, reads the NUT a feeder
writes there, and writes one row per picture it reads, ``{"feed_pts",
"w", "h"}``. Its describe names one feeder, the call's second stream, with
its port in `port`. The programme is ``tests/fixtures/testsrc.mp4``
(``scripts/gen_fixtures.py``), 320x240 at 15 frames a second; the feeder is
a second file written here at another size and the same rate, so each of
its frames is one row, at the programme's size.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from ffrwd import binaries, cli, wasm
from ffrwd.compiler import compile_all
from ffrwd.execute import render_plan

pytestmark = pytest.mark.exec

_CLI_ROOT = Path(__file__).resolve().parent.parent.parent
_REPO_ROOT = _CLI_ROOT.parent
_SIDECAR_MODULES = _REPO_ROOT / "sidecar" / "modules"
_PROBE = _SIDECAR_MODULES / "target" / "wasm32-wasip2" / "release" / "feed_probe.wasm"
_PROGRAMME = _CLI_ROOT / "tests" / "fixtures" / "testsrc.mp4"
_TIMEOUT = 120.0
# The feeder: a second at the programme's rate, at half its size.
_FEEDER_SPEC = "testsrc2=size=160x120:rate=15:duration=1"

_DECLARE = (
    "CREATE FUNCTION probe(v video_stream, feed video_stream DEFAULT NULL, "
    "port number DEFAULT 9000)\n"
    "RETURNS STRUCT(v video_stream, feeds STRUCT(feed_pts number, w number, h number)[])\n"
    "AS '{module}', 'feed-probe' LANGUAGE wasm;\n"
)


@pytest.fixture(autouse=True)
def _require_everything() -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg/ffprobe not found on PATH")
    if binaries.ffrwd_wasm_path() is None:
        pytest.skip("the ffrwd-wasm sidecar is not installed")
    if not _PROBE.exists():
        pytest.skip(
            f"module missing: {_PROBE} (cargo build --target wasm32-wasip2 "
            f"--release, from {_SIDECAR_MODULES})"
        )
    if not _PROGRAMME.exists():
        pytest.skip(f"fixture missing: {_PROGRAMME} (run scripts/gen_fixtures.py first)")


def _sql(call: str, out: str, *, inputs: str = "") -> str:
    return _DECLARE.format(module=_PROBE.as_posix()) + (
        f"COPY (SELECT {call} FROM input('{_PROGRAMME.as_posix()}') p{inputs}) TO '{out}'"
    )


def _stream(path: Path) -> dict[str, object]:
    done = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
            "-show_entries", "stream=width,height,nb_read_frames", "-of", "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=_TIMEOUT,
        check=True,
    )
    streams = json.loads(done.stdout)["streams"]
    assert len(streams) == 1
    return dict(streams[0])


def test_a_stream_feeder_reaches_the_module_at_the_programmes_size(tmp_path: Path) -> None:
    feeder = tmp_path / "feeder.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", _FEEDER_SPEC,
         "-pix_fmt", "yuv420p", str(feeder)],
        check=True,
        timeout=_TIMEOUT,
    )
    fed = _stream(feeder)
    programme = _stream(_PROGRAMME)
    rows_path = tmp_path / "rows.ndjson"
    sql = _sql(
        "probe(p.video[1], a.video[1]).feeds",
        rows_path.as_posix(),
        inputs=f", input('{feeder.as_posix()}') a",
    )
    assert cli.main(["run", sql, "-y", "-q"]) == 0
    rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line]
    assert len(rows) == int(str(fed["nb_read_frames"]))
    assert {(row["w"], row["h"]) for row in rows} == {
        (programme["width"], programme["height"])
    }
    pts = [row["feed_pts"] for row in rows]
    assert pts == sorted(pts) and len(set(pts)) == len(pts)


def _shown(call: str) -> str:
    compiled = compile_all(_sql(call, "rows.ndjson"))
    assert compiled.plan is not None
    assert not compiled.plan.feeder_edges
    return render_plan(compiled.plan, sidecar_argv=wasm.shown_argv)


def test_the_port_spelling_wires_nothing_and_compiles_as_the_port_by_name() -> None:
    """A number in the feeder's place is the port, as it was before feeders."""
    shown = _shown("probe(p.video[1], 9100).feeds")
    assert shown == _shown("probe(p.video[1], port => 9100).feeds")
    assert "tcp://" not in shown
    assert """-params '{"port": 9100}'""" in shown


def test_a_feeder_left_out_takes_the_ports_default() -> None:
    shown = _shown("probe(p.video[1]).feeds")
    assert """-params '{"port": 9000}'""" in shown
    assert "tcp://" not in shown


_HEAR = _SIDECAR_MODULES / "target" / "wasm32-wasip2" / "release" / "feed_probe_audio.wasm"
_HEAR_DECLARE = (
    "CREATE FUNCTION hear(a audio_stream, feed audio_stream DEFAULT NULL, "
    "port number DEFAULT 9000)\n"
    "RETURNS STRUCT(a audio_stream, feeds STRUCT(feed_pts number, rate number, "
    "channels number, sample_fmt text, samples number)[])\n"
    "AS '{module}', 'feed-probe-audio' LANGUAGE wasm;\n"
)


def test_a_sound_feeder_reaches_the_module_in_the_format_it_reads(tmp_path: Path) -> None:
    """``feed-probe-audio`` reads f32 at 48 kHz in stereo, and its feeder is a
    second of mono sound at that rate: every sample of it arrives, doubled
    into two channels, in the module's own format."""
    if not _HEAR.exists():
        pytest.skip(f"module missing: {_HEAR}")
    programme = _CLI_ROOT / "tests" / "fixtures" / "av.mp4"
    feeder = tmp_path / "feeder.wav"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
         "sine=frequency=440:sample_rate=48000:duration=1", str(feeder)],
        check=True,
        timeout=_TIMEOUT,
    )
    rows_path = tmp_path / "rows.ndjson"
    sql = _HEAR_DECLARE.format(module=_HEAR.as_posix()) + (
        f"COPY (SELECT hear(p.audio[1], a.audio[1]).feeds FROM "
        f"input('{programme.as_posix()}') p, input('{feeder.as_posix()}') a) "
        f"TO '{rows_path.as_posix()}'"
    )
    assert cli.main(["run", sql, "-y", "-q"]) == 0
    rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line]
    assert {(row["rate"], row["channels"], row["sample_fmt"]) for row in rows} == {
        (48000, 2, "f32")
    }
    assert sum(int(row["samples"]) for row in rows) == 48000
