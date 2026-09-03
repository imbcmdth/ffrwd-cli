"""A `RETURNS source` wasm function in FROM, both of its module shapes.

The URL-source half is unit tier and runs on a bare machine: a values
module, a fake `invoke` for its answer and a fake path probe for the rows
it names, so nothing is spawned and no file is read. The packet-source
half is the exec proof below.

`source-replay` (the sidecar's own fleet module,
`sidecar/modules/source-replay`) is a packet source with no filesystem
input: it compiles in a fixed seven-packet h264-shaped stream (one
video track, 32x24, decode order I0 P3 B1 B2 P6 B4 B5, a reorder depth
of 2) and hands it out through the same `packet-source` wit interface
any packet source implements. `probe`/`open` both publish a one-row,
one-track, bounded catalog, so `FROM replay() s` reads it exactly like
a probed manifest's rendition rows -- `s.height`, `s.video[1]`, WHERE,
the one-row rule.

Real throughout: the sidecar runs the module for real (`ffrwd-wasm -m
... -f nut pipe:1`, no `-i` before it -- a packet source rides alone,
reading nothing), and the far ffmpeg decodes and re-encodes what it
wrote. Requires ffmpeg/ffprobe on PATH, the `ffrwd-wasm` sidecar (`uv
sync --extra wasm`), and `source-replay` built for `wasm32-wasip2`.
Tests skip cleanly when any of those is missing.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from ffrwd import binaries, cli
from ffrwd.compiler import compile_sql
from ffrwd.emit import build_ffmpeg_args, emit
from ffrwd.errors import ErrorCode, FfrwdError
from ffrwd.ir import Graph
from ffrwd.lower import lower, lower_table
from ffrwd.parser import parse, resolve
from ffrwd.probe import ProbeResult, StreamMeta
from ffrwd.processes import is_live_probe
from ffrwd.registry import load_reference
from ffrwd.split import insert_splits
from ffrwd.wasm import WORLDS, Described, DescribedFunction

_CLI_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = _CLI_ROOT.parent
_SIDECAR_MODULES = _REPO_ROOT / "sidecar" / "modules"
_MODULE = _SIDECAR_MODULES / "target" / "wasm32-wasip2" / "release" / "source_replay.wasm"
_SNAPSHOT_PATH = _CLI_ROOT / "tests" / "data" / "reference_registry.json"

# The catalog `source-replay` publishes: one video track, 32x24, 7 coded
# packets -- see sidecar/modules/source-replay/src/lib.rs.
_HEIGHT = 24
_PACKET_COUNT = 7
_SUBPROCESS_TIMEOUT = 120.0


@pytest.fixture(autouse=True)
def _require_everything(request: pytest.FixtureRequest) -> None:
    """The exec tier's preconditions, and only the exec tier's -- the URL
    source tests below run ffmpeg, ffprobe and the sidecar not at all."""
    if request.node.get_closest_marker("exec") is None:
        return
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg/ffprobe not found on PATH")
    if binaries.ffrwd_wasm_path() is None:
        pytest.skip("ffrwd-wasm not found (uv sync --extra wasm)")
    if not _MODULE.exists():
        pytest.skip(
            f"module missing: {_MODULE} (cargo build --target wasm32-wasip2 "
            f"--release, from {_SIDECAR_MODULES})"
        )


def _query(out_path: Path, *, where: str = "") -> str:
    return (
        "CREATE FUNCTION replay() RETURNS source\n"
        f"  AS '{_MODULE.as_posix()}', 'source_replay' LANGUAGE wasm;\n"
        "COPY (\n"
        "  SELECT s.video[1]\n"
        f"  FROM replay() s{where}\n"
        f") TO '{out_path.as_posix()}' WITH (video_codec 'libx264', crf 20)"
    )


def _video_streams(path: Path) -> list[dict[str, object]]:
    """Every video stream `path` carries, frame-counted."""
    done = subprocess.run(
        [
            "ffprobe", "-v", "error", "-count_frames",
            "-show_entries", "stream=codec_type,nb_read_frames,width,height",
            "-of", "json", str(path),
        ],
        capture_output=True,
        text=True,
        timeout=_SUBPROCESS_TIMEOUT,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    streams: list[dict[str, object]] = json.loads(done.stdout)["streams"]
    return [s for s in streams if s["codec_type"] == "video"]


@pytest.mark.exec
def test_a_replayed_source_decodes_and_reencodes_all_seven_packets(tmp_path: Path) -> None:
    """`FROM replay() s` reads the module's compiled-in stream like any
    input: the sidecar rides alone (no `-i` feeds it) and writes the seven
    packets as one NUT pipe, the far ffmpeg decodes and re-encodes them --
    one video stream, seven frames, none dropped or duplicated by the
    two-deep reorder the decode order needs."""
    out = tmp_path / "out.mp4"
    assert cli.main(["run", _query(out), "-y"]) == 0
    assert out.exists()

    video_streams = _video_streams(out)
    assert len(video_streams) == 1
    assert int(video_streams[0]["nb_read_frames"]) == _PACKET_COUNT
    assert int(video_streams[0]["height"]) == _HEIGHT


@pytest.mark.exec
def test_where_on_source_height_keeps_the_row_and_a_wrong_height_refuses(
    tmp_path: Path,
) -> None:
    """`WHERE s.height = 24` narrows the catalog's one row to itself, so
    `s.video[1]` still reads -- the same one-row rule a probed manifest's
    rendition rows already obey. A height nothing in the catalog carries
    narrows the row set to zero, and reading a stream off zero rows is a
    compile-time refusal, not a runtime one."""
    out = tmp_path / "out.mp4"
    assert cli.main(["run", _query(out, where=f" WHERE s.height = {_HEIGHT}"), "-y"]) == 0
    assert out.exists()
    video_streams = _video_streams(out)
    assert len(video_streams) == 1
    assert int(video_streams[0]["height"]) == _HEIGHT

    wrong_height = _HEIGHT + 1
    with pytest.raises(FfrwdError) as excinfo:
        compile_sql(_query(tmp_path / "wrong.mp4", where=f" WHERE s.height = {wrong_height}"))
    err = excinfo.value
    assert err.code is ErrorCode.STREAM_NOT_FOUND
    assert err.message == "'s.video[1]' does not exist: 0 rows carry a video track"


@pytest.mark.exec
def test_compile_prints_the_source_riding_alone(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """A packet source reads nothing, so its sidecar is fed no `-i` at
    all -- `compile`'s printed pipeline puts `-f nut` on the sidecar's own
    output and no `-i` anywhere before its `-m`, the mirror of every other
    module-bearing recipe where the sidecar instead reads a decoded pipe."""
    out = tmp_path / "out.mp4"
    code = cli.main(["compile", _query(out)])
    printed = capsys.readouterr().out
    assert code == 0

    sidecar_segment = printed.split(" | ", 1)[0]
    assert "-m " in sidecar_segment
    assert "-f nut" in sidecar_segment
    assert "-i" not in sidecar_segment


# ---------------------------------------------------------------------------
# A URL source: `RETURNS source` over a VALUES module. The module is invoked
# once at compile time and answers with rows naming files; each row mints one
# hidden `-i`, is probed like any `input()` path, and binds as a rendition row
# carrying the module's own columns beside the probed ones.
#
# Unit tier throughout: `invoke` and the path probe are both fakes, so no
# sidecar runs, no ffprobe runs, and no file has to exist.
# ---------------------------------------------------------------------------


_URL_MODULE = "modules/source_files.wasm"
_URL_DECLARE = (
    "CREATE FUNCTION files(paths text) RETURNS source\n"
    f"  AS '{_URL_MODULE}', 'files' LANGUAGE wasm;\n"
)
_URL_ANSWER: dict[str, object] = {
    "rows": [
        {"url": "a.mp4", "sequence": 1, "ad": "a", "bandwidth": 4500000},
        {"url": "b.mp4", "sequence": 2, "ad": "b", "bandwidth": 900000},
    ],
    "document": "a.mp4,b.mp4",
}


def _url_described(export: str = "files") -> Described:
    """A values module offering one function: no single export and no
    packets -- the shape a module of value functions describes as."""
    return Described(
        world=WORLDS[-1],
        functions=(
            DescribedFunction(
                name=export,
                params_schema={"properties": {"paths": {"type": "string"}}},
                result_schema={"type": "object"},
            ),
        ),
    )


def _probe_of(height: int) -> ProbeResult:
    """One synthetic probed file: a 16:9 video track and a stereo audio one."""
    return ProbeResult(
        streams=[
            StreamMeta(
                type="video", index=0, metadata={}, width=height * 16 // 9,
                height=height, fps="25/1", sample_rate=None, codec="h264",
            ),
            StreamMeta(
                type="audio", index=0, metadata={}, width=None, height=None,
                fps=None, sample_rate=48000, codec="aac", channels=2,
            ),
        ]
    )


_URL_PROBES = {"a.mp4": _probe_of(360), "b.mp4": _probe_of(720)}


class _Calls:
    """A fake `invoke` that counts how often the module was actually run."""

    def __init__(self, answer: object) -> None:
        self.answer = answer
        self.count = 0

    def __call__(self, module: str, export: str, args: dict[str, object]) -> object:
        self.count += 1
        return self.answer


def _url_lowered(sql: str, *, answer: object = None) -> Graph:
    return lower(
        resolve(parse(_URL_DECLARE + sql)),
        {},
        registry=load_reference(_SNAPSHOT_PATH),
        describes={_URL_MODULE: _url_described()},
        invoke=_Calls(_URL_ANSWER if answer is None else answer),
        probe_path=_URL_PROBES.get,
    )


def _url_rows(sql: str) -> list[list[object]]:
    sinks = lower_table(
        resolve(parse(_URL_DECLARE + sql)),
        {},
        registry=load_reference(_SNAPSHOT_PATH),
        describes={_URL_MODULE: _url_described()},
        invoke=_Calls(_URL_ANSWER),
        probe_path=_URL_PROBES.get,
    )
    return sinks[0].result.rows


def _url_refuses(sql: str, *, answer: object = None) -> FfrwdError:
    with pytest.raises(FfrwdError) as excinfo:
        _url_lowered(sql, answer=answer)
    return excinfo.value


def test_a_url_sources_rows_carry_the_modules_column_beside_the_probed_one() -> None:
    """Two rows bind as two rendition rows: `s.sequence` is the module's own
    column and `s.height` the probe's, read off the file THAT row named --
    which is what makes one row table span two different inputs."""
    assert _url_rows(
        "SELECT s.sequence, s.height, s.ad FROM files('a.mp4,b.mp4') s"
    ) == [[1, 360, "a"], [2, 720, "b"]]


def test_a_where_on_the_modules_column_maps_that_rows_own_input() -> None:
    """`WHERE s.sequence = 2` leaves one row, so the one-row rule is
    satisfied and `s.video[1]` is the SECOND minted input's video -- but
    ffmpeg only ever opens the row that survived: emit drops the unmapped
    first `-i` and renumbers the map onto the one that is left."""
    g = _url_lowered(
        "COPY (SELECT s.video[1] FROM files('a.mp4,b.mp4') s WHERE s.sequence = 2)\n"
        "TO 'o.mp4'"
    )
    assert build_ffmpeg_args(emit(insert_splits(g))) == [
        "ffmpeg", "-i", "b.mp4",
        "-map", "0:v:0", "-c:0", "copy", "o.mp4",
    ]


def test_an_aggregate_over_a_url_sources_rows_gathers_in_row_order() -> None:
    """`array_agg(s.video[1])` gathers one stream per row, in the order the
    module wrote them, and each row's stream comes from its OWN minted
    input -- which is what a `concat(VARIADIC ...)` of the same array then
    stitches (cookbook 114, exec: only a live ffmpeg reports `concat`'s
    options, so the gather itself is what this tier can pin)."""
    g = _url_lowered(
        "COPY (SELECT array_agg(s.video[1]) FROM files('a.mp4,b.mp4') s) TO 'o.mkv'"
    )
    assert g.input_paths == ["a.mp4", "b.mp4"]
    assert [output.ref for output in g.outputs] == [
        "src:ffrwd.s#1:v:0",
        "src:ffrwd.s#2:v:0",
    ]
    assert build_ffmpeg_args(emit(insert_splits(g))) == [
        "ffmpeg", "-i", "a.mp4", "-i", "b.mp4",
        "-map", "0:v:0", "-c:0", "copy",
        "-map", "1:v:0", "-c:1", "copy", "o.mkv",
    ]


def test_the_graph_records_the_url_source_and_it_survives_a_round_trip() -> None:
    """`Graph.url_sources` is the record of what the module answered: the
    folded params, the document it wrote beside its rows, and one row per url
    with the input index it was minted at and its own columns."""
    g = _url_lowered(
        "COPY (SELECT s.video[1] FROM files('a.mp4,b.mp4') s WHERE s.sequence = 1)\n"
        "TO 'o.mp4'"
    )
    source = g.url_sources["s"]
    assert source.module == _URL_MODULE
    assert source.params == '{"paths": "a.mp4,b.mp4"}'
    assert source.document == "a.mp4,b.mp4"
    assert [(row.url, row.input) for row in source.rows] == [("a.mp4", 0), ("b.mp4", 1)]
    assert source.rows[1].columns == {"sequence": 2, "ad": "b"}
    assert Graph.from_dict(g.to_dict()).url_sources == g.url_sources


def test_two_reads_of_one_call_run_the_module_once() -> None:
    """The value call's own cache, keyed the same way: two COPYs naming the
    same call with the same arguments cost one run of the module."""
    invoke = _Calls(_URL_ANSWER)
    lower(
        resolve(
            parse(
                _URL_DECLARE
                + "COPY (SELECT s.video[1] FROM files('a.mp4,b.mp4') s "
                "WHERE s.sequence = 1) TO 'one.mp4';\n"
                "COPY (SELECT t.video[1] FROM files('a.mp4,b.mp4') t "
                "WHERE t.sequence = 2) TO 'two.mp4'"
            )
        ),
        {},
        registry=load_reference(_SNAPSHOT_PATH),
        describes={_URL_MODULE: _url_described()},
        invoke=invoke,
        probe_path=_URL_PROBES.get,
    )
    assert invoke.count == 1


def test_a_url_source_row_that_cannot_be_probed_is_refused() -> None:
    """A row names an input, and an input that cannot be read has no streams
    to bind -- the policy `input()` already has, said against the url the
    module produced."""
    err = _url_refuses(
        "SELECT s.video[1] FROM files('a.mp4,b.mp4') s",
        answer={"rows": [{"url": "a.mp4"}, {"url": "gone.mp4"}]},
    )
    assert err.code is ErrorCode.INPUT_NOT_FOUND
    assert err.message == "cannot read row 2 of 's': 'gone.mp4' could not be probed"


@pytest.mark.parametrize(
    ("answer", "message"),
    [
        pytest.param({"rows": []}, "'s' produced no rows", id="empty"),
        pytest.param(
            {"rows": [{"sequence": 1}]}, "row 1 of 's' names no url", id="no-url"
        ),
        pytest.param(
            {"rows": [{"url": "a.mp4", "height": 720}]},
            "row 1 of 's' names 'height'",
            id="probed-column",
        ),
        pytest.param(
            {"rows": [{"url": "a.mp4", "video": 1}]},
            "row 1 of 's' names 'video'",
            id="stream-column",
        ),
        pytest.param(
            {"rows": [{"url": "a.mp4", "n": 1}, {"url": "b.mp4", "m": 2}]},
            "row 2 of 's' does not name the same columns row 1 does (n): "
            "'m' is unexpected",
            id="odd-column",
        ),
        pytest.param(
            {"rows": [{"url": "a.mp4", "n": 1}, {"url": "b.mp4", "n": "two"}]},
            "column 's.n' holds both number and text",
            id="mixed-type",
        ),
        pytest.param(
            {"rows": [{"url": "a.mp4", "Seq": 1}]},
            "row 1 of 's' names the column 'Seq'",
            id="odd-name",
        ),
        pytest.param(
            {"rows": [{"url": "a.mp4", "bandwidth": "fast"}]},
            "row 1 of 's' gives 'bandwidth' 'fast'",
            id="bad-bandwidth",
        ),
        pytest.param({"rows": {}}, "'files()' returned no 'rows' list", id="no-rows"),
    ],
)
def test_a_malformed_url_source_answer_is_refused(answer: object, message: str) -> None:
    """Each shape rule, pinned by the message it produces: a source that
    answers nothing usable is refused at compile time, never at run time."""
    err = _url_refuses("SELECT s.video[1] FROM files('a.mp4,b.mp4') s", answer=answer)
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert err.message == message
    assert err.hint is not None


def test_a_column_no_row_named_is_an_unknown_column() -> None:
    """The module says what columns the alias has, so reading one it never
    named is refused here -- resolve cannot know them, and defers."""
    err = _url_refuses("SELECT s.rank FROM files('a.mp4,b.mp4') s")
    assert err.code is ErrorCode.UNSUPPORTED_SQL
    assert err.message == "unknown column 's.rank'"
    assert err.hint is not None
    assert "'s' exposes" in err.hint and "sequence" in err.hint


@pytest.mark.parametrize("bounded", [True, False])
def test_bounded_says_whether_a_url_sources_inputs_read_live(bounded: bool) -> None:
    """`bounded: false` marks the probes lowering registered -- the alias's
    own and every minted input's -- live, which is the same thing a manifest
    with no end tells the partitioner. `bounded` defaults to true, so an
    answer that leaves it out reads as an ordinary re-readable file."""
    probes: dict[str, ProbeResult | None] = {}
    answer: dict[str, object] = {"rows": [{"url": "a.mp4"}]}
    if not bounded:
        answer["bounded"] = False
    lower(
        resolve(
            parse(
                _URL_DECLARE
                + "COPY (SELECT s.video[1] FROM files('a.mp4') s) TO 'o.mp4'"
            )
        ),
        probes,
        registry=load_reference(_SNAPSHOT_PATH),
        describes={_URL_MODULE: _url_described()},
        invoke=_Calls(answer),
        probe_path=_URL_PROBES.get,
    )
    assert sorted(probes) == ["ffrwd.s#1", "s"]
    assert [is_live_probe(result) for result in probes.values()] == [not bounded] * 2
