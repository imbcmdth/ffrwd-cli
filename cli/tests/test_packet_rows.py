"""A packet sink read while compiling: `FROM input('f.mp4') f, keys(f.video[1]) v`.

The unit half runs on a bare machine. `read_packets` is the seam -- the one
call that would spawn ffmpeg and the sidecar -- so every test here hands
lowering rows of its own and nothing is spawned, no module is described for
real and no file has to exist. It covers the declaration form, the position
rule, the refusals, the memo's key, and rows shaping a real graph.

The exec half below is the same thing against real ffmpeg, the real sidecar
and the `packet-keys` / `packet-head` fixture modules
(``sidecar/modules/``), on the two containers the fixture pair carries.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from ffrwd import binaries
from ffrwd.compiler import compile_table_sql
from ffrwd.emit import build_ffmpeg_args, emit
from ffrwd.errors import ErrorCode, FfrwdError
from ffrwd.ir import Graph
from ffrwd.lower import lower, lower_table
from ffrwd.parser import parse, resolve
from ffrwd.probe import ProbeResult, StreamMeta, clear_cache
from ffrwd.registry import load_reference
from ffrwd.split import insert_splits
from ffrwd.wasm import (
    WORLDS,
    Described,
    DescribedFunction,
    PacketRead,
    SinkWants,
    copy_argv,
)

_CLI_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = _CLI_ROOT.parent
_SIDECAR_MODULES = _REPO_ROOT / "sidecar" / "modules"
_BUILT = _SIDECAR_MODULES / "target" / "wasm32-wasip2" / "release"
_KEYS_MODULE = _BUILT / "packet_keys.wasm"
_HEAD_MODULE = _BUILT / "packet_head.wasm"
_FIXTURES = _CLI_ROOT / "tests" / "fixtures"
_SNAPSHOT_PATH = _CLI_ROOT / "tests" / "data" / "reference_registry.json"
_SUBPROCESS_TIMEOUT = 120.0

# ---------------------------------------------------------------------------
# unit tier: the rows are injected, so nothing runs
# ---------------------------------------------------------------------------

_MODULE = "modules/keys.wasm"
_DECLARE = (
    "CREATE FUNCTION keys(v video_stream)\n"
    "RETURNS STRUCT(index number, start_t number, keyframe boolean, vector vector)[]\n"
    f"  AS '{_MODULE}', 'keys' LANGUAGE wasm;\n"
)
# The same module, declared the other way: a COPY destination, which is what
# it has always been and what this feature does not touch.
_SINK_DECLARE = (
    "CREATE FUNCTION keys(v video_stream) RETURNS sink\n"
    f"  AS '{_MODULE}', 'keys' LANGUAGE wasm;\n"
)

_ROWS_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "index": {"type": "integer"},
        "start_t": {"type": "number"},
        "keyframe": {"type": "boolean"},
        "bytes": {"type": "integer"},
        "vector": {"type": "array", "items": {"type": "number"}},
    },
}

_ANSWER: tuple[dict[str, object], ...] = (
    {"index": 1, "start_t": 0.0, "keyframe": True, "bytes": 10, "vector": [1.0, 0.0]},
    {"index": 2, "start_t": 1.5, "keyframe": True, "bytes": 20, "vector": [0.0, 1.0]},
    {"index": 3, "start_t": 3.0, "keyframe": True, "bytes": 30, "vector": [1.0, 1.0]},
)


def _described(
    *,
    wants: SinkWants = "keyframes",
    export: str = "keys",
    rows_schema: dict[str, object] | None = None,
    packet_filter: bool = False,
    video_streams: str = "one",
) -> Described:
    """A packet sink's description: a filled video codec list is what says so."""
    return Described(
        world=WORLDS[-1],
        name=export,
        rows_schema=_ROWS_SCHEMA if rows_schema is None else rows_schema,
        video_codecs=(),
        audio_codecs=(),
        video_streams=video_streams,  # type: ignore[arg-type]
        audio_streams="none",
        wants=wants,
        packet_filter=packet_filter,
    )


class _Reads:
    """A fake `read_packets` that records every read it was asked for."""

    def __init__(self, answer: tuple[dict[str, object], ...] = _ANSWER) -> None:
        self.answer = answer
        self.reads: list[PacketRead] = []

    def __call__(
        self, read: PacketRead, *, described: Described | None = None
    ) -> tuple[dict[str, object], ...]:
        self.reads.append(read)
        return self.answer


_PROMPT_MODULE = "modules/prompt.wasm"
# The only way a vector enters a query: a value function that answers one.
_PROMPT_DECLARE = (
    "CREATE FUNCTION prompt(words text) RETURNS vector\n"
    f"  AS '{_PROMPT_MODULE}', 'prompt' LANGUAGE wasm;\n"
)


def _prompt_described() -> Described:
    return Described(
        world=WORLDS[-1],
        functions=(
            DescribedFunction(
                name="prompt",
                params_schema={"properties": {"words": {"type": "string"}}},
                result_schema={"type": "array", "items": {"type": "number"}},
            ),
        ),
    )


class _Prompt:
    """A fake `invoke` answering one fixed vector, whatever it is asked."""

    def __init__(self, vector: list[float]) -> None:
        self.vector = vector

    def __call__(
        self,
        module: str,
        export: str,
        args: dict[str, object],
        described: Described | None = None,
    ) -> object:
        return self.vector


class _Fails:
    """A fake `read_packets` that fails the way the real one does."""

    def __init__(self, error: FfrwdError) -> None:
        self.error = error

    def __call__(
        self, read: PacketRead, *, described: Described | None = None
    ) -> tuple[dict[str, object], ...]:
        raise self.error


def _probe(*, live: bool = False, audio: bool = True) -> ProbeResult:
    streams = [
        StreamMeta(
            type="video", index=0, metadata={}, width=320, height=240,
            fps="15/1", sample_rate=None, codec="h264",
        )
    ]
    if audio:
        streams.append(
            StreamMeta(
                type="audio", index=0, metadata={}, width=None, height=None,
                fps=None, sample_rate=48000, codec="aac", channels=2,
            )
        )
    return ProbeResult(streams=streams, live=live)


@pytest.fixture(autouse=True)
def _fresh_memo() -> None:
    """The read is memoized process-wide; each test starts with nothing in it."""
    clear_cache()


def _describes(described: Described | None) -> dict[str, Described]:
    return {
        _MODULE: _described() if described is None else described,
        _PROMPT_MODULE: _prompt_described(),
    }


def _lowered(
    sql: str,
    *,
    declare: str = _DECLARE,
    reads: object = None,
    described: Described | None = None,
    probes: dict[str, ProbeResult | None] | None = None,
    vector: list[float] | None = None,
) -> Graph:
    return lower(
        resolve(parse(declare + sql)),
        {"f": _probe()} if probes is None else probes,
        registry=load_reference(_SNAPSHOT_PATH),
        describes=_describes(described),
        invoke=_Prompt(vector or [1.0, 0.0]),
        read_packets=reads or _Reads(),  # type: ignore[arg-type]
    )


def _rows(
    sql: str,
    *,
    declare: str = _DECLARE,
    reads: object = None,
    described: Described | None = None,
    probes: dict[str, ProbeResult | None] | None = None,
    vector: list[float] | None = None,
) -> list[list[object]]:
    sinks = lower_table(
        resolve(parse(declare + sql)),
        {"f": _probe()} if probes is None else probes,
        registry=load_reference(_SNAPSHOT_PATH),
        describes=_describes(described),
        invoke=_Prompt(vector or [1.0, 0.0]),
        read_packets=reads or _Reads(),  # type: ignore[arg-type]
    )
    return sinks[0].result.rows


def _refuses(sql: str, **kwargs: object) -> FfrwdError:
    with pytest.raises(FfrwdError) as caught:
        _lowered(sql, **kwargs)  # type: ignore[arg-type]
    return caught.value


def test_the_declared_columns_are_the_aliass_columns_vector_included() -> None:
    """The RETURNS names the rows, so the alias exposes exactly those columns
    with exactly those types -- and a `vector` column reads through the vector
    builtins, which is the whole reason the type is in the record vocabulary."""
    assert _rows(
        "SELECT v.index, v.start_t, v.keyframe, vector_length(v.vector) "
        "FROM input('f.mp4') f, keys(f.video[1]) v"
    ) == [[1, 0.0, True, 2], [2, 1.5, True, 2], [3, 3.0, True, 2]]


def test_a_vector_with_no_declared_length_compares_on_the_values() -> None:
    """The module's schema fixes no length -- a record's dims live in the
    stream, not in the schema -- so what `cos_similarity` checks is the two
    vectors it is actually handed, and a mismatch is refused there."""
    assert _rows(
        "SELECT round(cos_similarity(v.vector, prompt('a cat')), 3) "
        "FROM input('f.mp4') f, keys(f.video[1]) v",
        declare=_DECLARE + _PROMPT_DECLARE,
    ) == [[1.0], [0.0], [0.707]]

    with pytest.raises(FfrwdError) as caught:
        _rows(
            "SELECT cos_similarity(v.vector, prompt('a cat')) "
            "FROM input('f.mp4') f, keys(f.video[1]) v",
            declare=_DECLARE + _PROMPT_DECLARE,
            vector=[1.0, 0.0, 0.0],
        )
    assert "2" in caught.value.message and "3" in caught.value.message


def test_where_over_the_rows_shapes_the_graph() -> None:
    """The rows are a compile-time relation like any other: a WHERE over them
    decides how many trims exist and at what times, which is the whole point
    of reading them while compiling rather than at run time."""
    g = _lowered(
        "COPY (\n"
        "  SELECT array_agg("
        "ffmpeg.trim(f.video[1], start => v.start_t, end => v.start_t + 0.5))\n"
        "  FROM input('f.mp4') f, keys(f.video[1]) v\n"
        "  WHERE v.start_t > 0.5\n"
        ") TO 'o.mkv'"
    )
    # One trim per surviving row, at that row's own time: the rows decided how
    # many nodes the graph has, which a run-time read could not have done.
    assert [(node.filter, node.args) for node in g.nodes.values()] == [
        ("trim", {"start": 1.5, "end": 2.0}),
        ("trim", {"start": 3.0, "end": 3.5}),
    ]
    printed = " ".join(build_ffmpeg_args(emit(insert_splits(g))))
    assert "trim=start=1.5:end=2.0" in printed
    assert "trim=start=3.0:end=3.5" in printed
    assert "trim=start=0.0" not in printed


def test_the_read_names_the_stream_and_what_the_module_asked_for() -> None:
    """What reaches the seam is the file as the query wrote it, the stream the
    call named, the module's own parameters, and the `wants` off its describe
    -- nothing about the query beyond that."""
    reads = _Reads()
    _rows("SELECT v.index FROM input('f.mp4') f, keys(f.video[1]) v", reads=reads)
    (read,) = reads.reads
    assert read.spec == "f.mp4"
    assert (read.kind, read.index) == ("video", 0)
    assert read.module == _MODULE
    assert read.wants == "keyframes"
    assert read.params == ""


def test_the_calls_arguments_become_the_modules_own_parameters() -> None:
    """Past the stream the arguments are positional, like every other wasm
    call, and reach the sidecar as the named object a run-time sink's do --
    checked against the schema the module declares, by name and by type."""
    declare = (
        "CREATE FUNCTION keys(v video_stream, space number, label text)\n"
        "RETURNS STRUCT(index number, start_t number, keyframe boolean, vector vector)[]\n"
        f"  AS '{_MODULE}', 'keys' LANGUAGE wasm;\n"
    )
    configured = _described()
    configured = replace(
        configured,
        params_schema={
            "properties": {"space": {"type": "number"}, "label": {"type": "string"}}
        },
    )
    reads = _Reads()
    _rows(
        "SELECT v.index FROM input('f.mp4') f, keys(f.video[1], 3, 'clip') v",
        declare=declare,
        reads=reads,
        described=configured,
    )
    assert reads.reads[0].params == '{"label": "clip", "space": 3}'

    with pytest.raises(FfrwdError) as caught:
        _rows(
            "SELECT v.index FROM input('f.mp4') f, keys(f.video[1], 'three', 'clip') v",
            declare=declare,
            described=configured,
        )
    assert "space" in caught.value.message


def test_one_read_answers_every_column_and_a_second_alias_of_the_same_stream() -> None:
    """Memoized per (file, stream, module, params, wants): naming several of
    the alias's columns reads once, and so does a second alias over the same
    stream through the same module."""
    reads = _Reads()
    rows = _rows(
        "SELECT v.index, v.start_t, w.index "
        "FROM input('f.mp4') f, keys(f.video[1]) v "
        "JOIN keys(f.video[1]) w ON v.index = w.index",
        reads=reads,
    )
    assert rows == [[1, 0.0, 1], [2, 1.5, 2], [3, 3.0, 3]]
    assert len(reads.reads) == 1


def test_a_different_stream_or_a_different_want_is_a_different_read() -> None:
    """The key is the whole question asked, so a read of another stream of the
    same file is its own, and so is the same stream through a module asking
    for a different amount of it."""
    reads = _Reads()
    _rows(
        "SELECT v.index, w.index "
        "FROM input('f.mp4') f, input('g.mp4') g, keys(f.video[1]) v "
        "JOIN keys(g.video[1]) w ON v.index = w.index",
        reads=reads,
        probes={"f": _probe(), "g": _probe()},
    )
    assert [read.spec for read in reads.reads] == ["f.mp4", "g.mp4"]

    clear_cache()
    first = _Reads()
    _rows("SELECT v.index FROM input('f.mp4') f, keys(f.video[1]) v", reads=first)
    again = _Reads()
    _rows(
        "SELECT v.index FROM input('f.mp4') f, keys(f.video[1]) v",
        reads=again,
        described=_described(wants="first"),
    )
    assert len(first.reads) == 1 and len(again.reads) == 1


def test_a_second_compile_of_the_same_question_runs_nothing() -> None:
    """The memo outlives one compile, the way a probe's does: the same file
    read the same way through the same module answers from what it said."""
    first = _Reads()
    _rows("SELECT v.index FROM input('f.mp4') f, keys(f.video[1]) v", reads=first)
    warm = _Reads()
    assert _rows(
        "SELECT v.index FROM input('f.mp4') f, keys(f.video[1]) v", reads=warm
    ) == [[1], [2], [3]]
    assert len(first.reads) == 1 and warm.reads == []


# -- the position rule ------------------------------------------------------


def test_the_same_module_after_to_is_still_a_run_time_destination() -> None:
    """Position is the discriminator: declared `RETURNS sink` and written
    after TO, the module is the destination it has always been, and no read
    happens while compiling."""
    reads = _Reads()
    g = _lowered(
        "COPY (SELECT f.video[1] FROM input('f.mp4') f) TO keys()",
        declare=_SINK_DECLARE,
        reads=reads,
    )
    assert reads.reads == []
    assert [node.filter for node in g.nodes.values()] == [_MODULE]
    assert list(g.packet_sinks) == list(g.nodes)


def test_a_packet_rows_call_outside_from_is_refused() -> None:
    error = _refuses("COPY (SELECT keys(f.video[1]) FROM input('f.mp4') f) TO 'o.mp4'")
    assert error.code is ErrorCode.UNSUPPORTED_SQL
    assert "this call is not in FROM" in error.message
    assert "FROM input('<path>') f" in (error.hint or "")


# -- the refusals -----------------------------------------------------------


def test_a_filtered_stream_is_refused_naming_what_a_read_needs() -> None:
    error = _refuses(
        "COPY (SELECT f.video[1] FROM input('f.mp4') f, "
        "keys(scale(f.video[1], 640, -2)) v) TO 'o.mp4'"
    )
    assert error.code is ErrorCode.UNSUPPORTED_SQL
    assert error.message == (
        "keys() reads a file's own packets, and this is not a stream of one"
    )
    assert "as it is on disk" in (error.hint or "")


def test_a_stream_from_another_query_stage_is_refused() -> None:
    error = _refuses(
        "COPY (SELECT t FROM input('f.mp4') f, unnest(f.video) t, keys(t.stream) v) TO 'o.mp4'"
    )
    assert error.code is ErrorCode.UNSUPPORTED_SQL
    assert "a compile-time read needs a stream of a file as it is on disk" in error.message


def test_a_live_input_is_refused_saying_where_it_is_read_instead() -> None:
    error = _refuses(
        "COPY (SELECT f.video[1] FROM input('f.m3u8') f, keys(f.video[1]) v) TO 'o.mp4'",
        probes={"f": _probe(live=True)},
    )
    assert error.code is ErrorCode.UNBOUNDED_LIVE_INPUT
    assert "never ends" in error.message
    assert "read at run time" in (error.hint or "")


def test_a_module_that_is_not_a_packet_sink_is_refused_by_name() -> None:
    error = _refuses(
        "SELECT v.index FROM input('f.mp4') f, keys(f.video[1]) v",
        described=_described(packet_filter=True),
    )
    assert error.code is ErrorCode.UNSUPPORTED_SQL
    assert "is not a packet sink" in error.message


def test_a_declared_column_the_module_never_writes_is_refused() -> None:
    error = _refuses(
        "SELECT v.index FROM input('f.mp4') f, keys(f.video[1]) v",
        described=_described(
            rows_schema={"type": "object", "properties": {"index": {"type": "integer"}}}
        ),
    )
    assert error.code is ErrorCode.UDF_ARG_TYPE
    assert "writes rows of index (integer)" in error.message


def test_a_row_that_does_not_match_the_declared_columns_is_refused() -> None:
    error = _refuses(
        "SELECT v.index FROM input('f.mp4') f, keys(f.video[1]) v",
        reads=_Reads(
            (
                {"index": 1, "start_t": 0.0, "keyframe": True, "vector": [1.0]},
                {"index": 2, "start_t": "late", "keyframe": True, "vector": [1.0]},
            )
        ),
    )
    assert error.code is ErrorCode.UDF_ARG_TYPE
    assert error.message == "row 2 of 'v' holds text in 'start_t', which is declared number"


def test_a_column_a_row_leaves_out_reads_null() -> None:
    """A schema may make a column optional -- a reading that could not be
    taken -- so a row without it is NULL, not a refusal."""
    assert _rows(
        "SELECT v.index, v.start_t FROM input('f.mp4') f, keys(f.video[1]) v",
        reads=_Reads(({"index": 1, "keyframe": True, "vector": [1.0]},)),
    ) == [[1, None]]


def test_a_failed_read_is_reported_against_the_stream_it_was_asked_for() -> None:
    error = _refuses(
        "SELECT v.index FROM input('f.mp4') f, keys(f.video[1]) v",
        reads=_Fails(
            FfrwdError(
                ErrorCode.UNSUPPORTED_SQL,
                "the module 'modules/keys.wasm' rejected the stream: no such codec",
                hint="check the module reads the codec the stream carries",
            )
        ),
    )
    assert error.code is ErrorCode.UNSUPPORTED_SQL
    assert error.message.startswith("cannot read 'v': ")
    assert "no such codec" in error.message
    assert error.hint == "check the module reads the codec the stream carries"


def test_a_declaration_reads_back_as_a_stream_in_rows_out() -> None:
    declared = resolve(
        parse(_DECLARE + "SELECT v.index FROM input('f.mp4') f, keys(f.video[1]) v")
    ).wasm["keys"]
    assert declared.is_packet_rows
    assert not declared.is_rows and not declared.is_value and not declared.is_sink
    assert declared.stream_kind == "video"
    assert [p.name for p in declared.stream_params] == ["v"]
    assert declared.reads is None


# -- the command the read builds --------------------------------------------


@pytest.mark.parametrize(
    ("wants", "expected"),
    [
        ("all", ["-c", "copy", "-bsf:0", "setts=dts=DTS-STARTDTS"]),
        (
            "keyframes",
            ["-c", "copy", "-bsf:0", "noise=drop=not(key),setts=dts=DTS-STARTDTS"],
        ),
        (
            "first",
            ["-frames:0", "1", "-c", "copy", "-bsf:0", "setts=dts=DTS-STARTDTS"],
        ),
    ],
)
def test_what_each_want_copies(wants: str, expected: list[str]) -> None:
    """`wants` shapes the copy and nothing else. Every case keeps the clock
    filter: the copy has to land on the timestamps the file presents, whatever
    the module asked to be handed."""
    read = PacketRead(
        spec="f.mp4", input_args=("-r", "30"), kind="video", index=1,
        module=_MODULE, params="", wants=wants,  # type: ignore[arg-type]
    )
    argv = copy_argv("ffmpeg", read)
    assert argv[:8] == ["ffmpeg", "-v", "error", "-r", "30", "-i", "f.mp4", "-map"]
    assert argv[8] == "0:v:1"
    assert argv[9:-3] == expected
    assert argv[-3:] == ["-f", "nut", "pipe:1"]


def test_a_refused_copy_widens_rather_than_narrows() -> None:
    """An ffmpeg that cannot take one of the filters is retried with less of
    the chain, never with more of it: every retry hands the sink MORE of the
    stream, which `wants` allows, and handing over less is what it forbids."""
    read = PacketRead(
        spec="f.mp4", input_args=(), kind="video", index=0,
        module=_MODULE, params="", wants="keyframes",
    )
    chains = [
        [token for token in copy_argv("ffmpeg", read, attempt) if "=" in token]
        for attempt in range(3)
    ]
    assert chains == [
        ["noise=drop=not(key),setts=dts=DTS-STARTDTS"],
        ["setts=dts=DTS-STARTDTS"],
        [],
    ]


# ---------------------------------------------------------------------------
# exec tier: real ffmpeg, the real sidecar, the fixture modules
# ---------------------------------------------------------------------------

_EXEC_DECLARE = (
    "CREATE FUNCTION packet_keys(v video_stream)\n"
    "RETURNS STRUCT(index number, start_t number, keyframe boolean, bytes number,\n"
    "               vector vector)[]\n"
    f"  AS '{_KEYS_MODULE.as_posix()}', 'packet_keys' LANGUAGE wasm;\n"
)
_EXEC_HEAD_DECLARE = (
    "CREATE FUNCTION packet_head(v video_stream)\n"
    "RETURNS STRUCT(codec text, start_t number, extradata number)[]\n"
    f"  AS '{_HEAD_MODULE.as_posix()}', 'packet_head' LANGUAGE wasm;\n"
)


@pytest.fixture
def _require_everything() -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg/ffprobe not found on PATH")
    if binaries.ffrwd_wasm_path() is None:
        pytest.skip("ffrwd-wasm not found (uv sync --extra wasm)")
    for module in (_KEYS_MODULE, _HEAD_MODULE):
        if not module.exists():
            pytest.skip(
                f"module missing: {module} (cargo build --target wasm32-wasip2 "
                f"--release, from {_SIDECAR_MODULES})"
            )
    for fixture in (_FIXTURES / "keys.mp4", _FIXTURES / "keys.mkv"):
        if not fixture.exists():
            pytest.skip(f"fixture missing: {fixture} (run scripts/gen_fixtures.py first)")


def _table(sql: str) -> list[list[object]]:
    """Compile a bare SELECT for real: real describe, real read."""
    return compile_table_sql(sql)[0].result.rows


def _keyframe_times(path: Path) -> list[float]:
    """The presentation time of every keyframe, as ffmpeg's own decode reports
    it -- the clock a `trim` in a query is written against."""
    done = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-skip_frame", "nokey",
            "-show_entries", "frame=pts_time", "-of", "json", str(path),
        ],
        capture_output=True, text=True, timeout=_SUBPROCESS_TIMEOUT, check=False,
    )
    assert done.returncode == 0, done.stderr
    return [float(frame["pts_time"]) for frame in json.loads(done.stdout)["frames"]]


def _framemd5(args: list[str]) -> list[str]:
    done = subprocess.run(
        ["ffmpeg", "-v", "error", *args, "-f", "framemd5", "-"],
        capture_output=True, text=True, timeout=_SUBPROCESS_TIMEOUT, check=False,
    )
    assert done.returncode == 0, done.stderr
    return [
        line.split(",")[-1].strip()
        for line in done.stdout.splitlines()
        if line and not line.startswith("#")
    ]


@pytest.mark.exec
@pytest.mark.parametrize("container", ["mp4", "mkv"])
def test_keyframes_reads_the_keyframes_of_either_container(
    _require_everything: None, container: str
) -> None:
    """`wants: keyframes` copies what decoding can start at. The fixture pair
    is one encode in two containers, so the rows are the same rows -- which is
    what says the answer does not depend on which demuxer read it."""
    clear_cache()
    source = _FIXTURES / f"keys.{container}"
    rows = _table(
        _EXEC_DECLARE
        + "SELECT v.index, v.start_t, v.keyframe\n"
        f"FROM input('{source.as_posix()}') f, packet_keys(f.video[1]) v"
    )
    assert rows, "the fixture carries keyframes"
    assert all(row[2] is True for row in rows), rows
    read = [float(row[1]) for row in rows]  # type: ignore[arg-type]
    truth = _keyframe_times(source)
    # A host may hand a sink more than it asked for and never less, so what is
    # pinned is that every keyframe arrived, not that nothing else did.
    for expected in truth:
        assert any(abs(expected - seen) < 0.002 for seen in read), (expected, read)


@pytest.mark.exec
def test_first_reads_one_packet(_require_everything: None) -> None:
    """`wants: first` stops after one packet, so a sink reading only what the
    stream declares about itself pays for one packet of the file."""
    clear_cache()
    source = _FIXTURES / "keys.mp4"
    rows = _table(
        _EXEC_HEAD_DECLARE
        + "SELECT v.codec, v.start_t\n"
        f"FROM input('{source.as_posix()}') f, packet_head(f.video[1]) v"
    )
    assert rows == [["h264", 0.0]]


@pytest.mark.exec
@pytest.mark.parametrize("container", ["mp4", "mkv"])
def test_a_reported_time_is_the_time_a_trim_means(
    _require_everything: None, container: str
) -> None:
    """The clock rule. The fixture opens on a negative dts, which is what a
    plain copy's timestamps get shifted by; a trim written from a time this
    read reported lands on THAT frame of the source, byte for byte."""
    clear_cache()
    source = _FIXTURES / f"keys.{container}"
    rows = _table(
        _EXEC_DECLARE
        + "SELECT v.start_t\n"
        f"FROM input('{source.as_posix()}') f, packet_keys(f.video[1]) v\n"
        "WHERE v.index = 4"
    )
    ((reported,),) = rows
    start = float(reported)  # type: ignore[arg-type]

    # What the query would do with that time, and what the source holds at it.
    trimmed = _framemd5(
        ["-i", str(source), "-map", "0:v:0",
         "-vf", f"trim=start={start}:end={start + 0.01},setpts=PTS-STARTPTS"]
    )
    whole = _framemd5(["-i", str(source), "-map", "0:v:0"])
    times = _keyframe_times(source)
    assert trimmed, "the trim keeps at least one frame"
    # The reported time is a keyframe of the source, and the frame a trim from
    # it opens on is that keyframe -- not the one a shifted clock would name.
    at = min(range(len(times)), key=lambda i: abs(times[i] - start))
    assert abs(times[at] - start) < 0.002, (start, times)
    assert trimmed[0] == whole[round(start * 15)]


@pytest.mark.exec
def test_a_cold_and_a_warm_cache_answer_the_same(_require_everything: None) -> None:
    clear_cache()
    source = (_FIXTURES / "keys.mp4").as_posix()
    query = (
        _EXEC_DECLARE
        + f"SELECT v.index, v.start_t, v.bytes FROM input('{source}') f, "
        "packet_keys(f.video[1]) v"
    )
    cold = _table(query)
    warm = _table(query)
    assert cold == warm and cold


@pytest.mark.exec
def test_a_module_that_reads_nothing_of_the_stream_is_refused_for_real(
    _require_everything: None,
) -> None:
    """A real describe behind the refusals: `packet_head` is a packet sink
    over video, so declaring it over audio is refused at the call rather than
    by the module failing once the copy is already running."""
    clear_cache()
    source = _FIXTURES / "keys.mp4"
    with pytest.raises(FfrwdError) as caught:
        _table(
            "CREATE FUNCTION packet_head(a audio_stream)\n"
            "RETURNS STRUCT(codec text, start_t number, extradata number)[]\n"
            f"  AS '{_HEAD_MODULE.as_posix()}', 'packet_head' LANGUAGE wasm;\n"
            f"SELECT v.codec FROM input('{source.as_posix()}') f, packet_head(f.audio[1]) v"
        )
    assert caught.value.code is ErrorCode.UNSUPPORTED_SQL
    assert "reads none" in caught.value.message


# The index package's own `records` sink, over a file it wove: the read this
# whole feature exists for, end to end. It runs only when both environment
# variables point at one, since the package is not in this repo and CI must
# not depend on it.
_INDEX_RECORDS = "FFRWD_INDEX_RECORDS_WASM"
_INDEX_FIXTURE = "FFRWD_INDEX_WOVEN"


@pytest.mark.exec
def test_a_woven_file_answers_a_find(_require_everything: None) -> None:
    """`FROM input(:'src') f, records(f.video[1]) v` on a file with vectors in
    its packets: the rows come back typed, the vector column reads through the
    vector builtins, and a trim per surviving row is what the graph becomes."""
    module = os.environ.get(_INDEX_RECORDS)
    woven = os.environ.get(_INDEX_FIXTURE)
    if not module or not woven:
        pytest.skip(f"set {_INDEX_RECORDS} and {_INDEX_FIXTURE} to run the index read")
    clear_cache()
    rows = _table(
        "CREATE FUNCTION records(v video_stream)\n"
        "RETURNS STRUCT(index number, space number, start_t number, end_t number,\n"
        "               vector vector)[]\n"
        f"  AS '{Path(module).as_posix()}', 'records' LANGUAGE wasm;\n"
        "SELECT v.index, v.space, v.start_t, v.end_t, vector_length(v.vector)\n"
        f"FROM input('{Path(woven).as_posix()}') f, records(f.video[1]) v"
    )
    assert rows, "the woven fixture carries records"
    assert all(len(row) == 5 for row in rows)
    assert all(isinstance(row[4], int) and row[4] > 0 for row in rows)
