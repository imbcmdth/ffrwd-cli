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
from typing import cast

import pytest

from ffrwd import binaries
from ffrwd.compiler import compile_sql, compile_table_sql
from ffrwd.emit import build_ffmpeg_args, emit
from ffrwd.errors import ErrorCode, FfrwdError
from ffrwd.ir import Graph
from ffrwd.lower import lower, lower_table
from ffrwd.parser import parse, resolve
from ffrwd.probe import ProbeResult, StreamMeta, clear_cache
from ffrwd.registry import load_reference
from ffrwd.split import insert_splits
from ffrwd.vars import substitute
from ffrwd.wasm import (
    TIMEOUT_ENV,
    WORLDS,
    Described,
    DescribedFunction,
    PacketRead,
    SinkWants,
    copy_argv,
    read_packet_rows,
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
        functions=tuple(
            DescribedFunction(
                name=name,
                params_schema={"properties": {"words": {"type": "string"}}},
                result_schema={"type": "array", "items": {"type": "number"}},
            )
            for name in ("prompt", "embed_text", "embed_clip_text")
        ),
    )


class _Prompt:
    """A fake `invoke`: one vector per export, so two spaces are two answers."""

    def __init__(
        self, vector: list[float], per_export: dict[str, list[float]] | None = None
    ) -> None:
        self.vector = vector
        self.per_export = per_export or {}

    def __call__(
        self,
        module: str,
        export: str,
        args: dict[str, object],
        described: Described | None = None,
    ) -> object:
        return self.per_export.get(export, self.vector)


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


# The shape `ffrwd/describe`'s find recipe writes: two spaces kept apart in two
# UNION ALL branches, each narrowed to its own space by an AND before its own
# `cos_similarity`, so neither prompt is ever scored against the other's rows.
_FIND_DECLARE = (
    "CREATE FUNCTION keys(v video_stream)\n"
    "RETURNS STRUCT(space number, label text, start_t number, end_t number,\n"
    "               vector vector)[]\n"
    f"  AS '{_MODULE}', 'keys' LANGUAGE wasm;\n"
    "CREATE FUNCTION embed_text(words text) RETURNS vector\n"
    f"  AS '{_PROMPT_MODULE}', 'embed_text' LANGUAGE wasm;\n"
    "CREATE FUNCTION embed_clip_text(words text) RETURNS vector\n"
    f"  AS '{_PROMPT_MODULE}', 'embed_clip_text' LANGUAGE wasm;\n"
)

_FIND = """COPY (
  SELECT array_agg(ffmpeg.trim(f.video[1],  start => v.start_t, end => v.end_t)),
         array_agg(ffmpeg.atrim(f.audio[1], start => v.start_t, end => v.end_t))
  FROM input(:'src') f, keys(f.video[1]) v
  WHERE v.label <> 'clip'
    AND cos_similarity(v.vector, embed_text(:'prompt')) > :threshold
  UNION ALL
  SELECT array_agg(ffmpeg.trim(g.video[1],  start => w.start_t, end => w.end_t)),
         array_agg(ffmpeg.atrim(g.audio[1], start => w.start_t, end => w.end_t))
  FROM input(:'src') g, keys(g.video[1]) w
  WHERE w.space = 1
    AND cos_similarity(w.vector, embed_clip_text(:'prompt')) > :threshold
) TO :'dest'"""

# Three rows over two spaces. Against the two prompt vectors below, exactly
# one row of each branch scores above the threshold: the speech row pointing
# the way embed_text does, and the clip row pointing the way the other does.
_FIND_ROWS: tuple[dict[str, object], ...] = (
    {"space": 0, "label": "speech", "start_t": 0.0, "end_t": 1.0, "vector": [1.0, 0.0]},
    {"space": 1, "label": "clip", "start_t": 1.0, "end_t": 2.0, "vector": [0.0, 1.0]},
    {"space": 0, "label": "speech", "start_t": 2.0, "end_t": 3.0, "vector": [0.0, 1.0]},
)

_FIND_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "space": {"type": "integer"},
        "label": {"type": "string"},
        "start_t": {"type": "number"},
        "end_t": {"type": "number"},
        "vector": {"type": "array", "items": {"type": "number"}},
    },
}


def test_the_find_recipes_shape_works_over_packet_rows() -> None:
    """The whole point of the feature, in the shape it exists for. Packet rows
    go through the same predicate evaluator an `unnest` row does,
    so all of it holds at once: text `<>` and number `=` on a row column,
    ANDed with a `cos_similarity` against a compile-time module value, against
    a threshold that came in as a variable, in two UNION ALL branches, with
    video AND audio trimmed by the same surviving rows."""
    text = substitute(
        _FIND_DECLARE + _FIND,
        {"src": "f.mp4", "prompt": "a cat", "threshold": "0.2", "dest": "out.mkv"},
    )
    g = lower(
        resolve(parse(text.text, text.unset)),
        {"f": _probe(), "g": _probe()},
        registry=load_reference(_SNAPSHOT_PATH),
        describes={
            _MODULE: replace(_described(), rows_schema=_FIND_SCHEMA),
            _PROMPT_MODULE: _prompt_described(),
        },
        invoke=_Prompt(
            [1.0, 0.0],
            {"embed_text": [1.0, 0.0], "embed_clip_text": [0.0, 1.0]},
        ),
        read_packets=_Reads(_FIND_ROWS),
    )
    assert [(node.filter, node.args) for node in g.nodes.values()] == [
        ("trim", {"start": 0.0, "end": 1.0}),
        ("atrim", {"start": 0.0, "end": 1.0}),
        ("trim", {"start": 1.0, "end": 2.0}),
        ("atrim", {"start": 1.0, "end": 2.0}),
        ("concat", {"n": 2, "v": 1, "a": 1}),
    ]


def test_a_threshold_that_keeps_nothing_keeps_nothing() -> None:
    """The same query with the cutoff raised past every row's score: each
    branch's WHERE is evaluated on its own rows, so both go empty and the
    refusal is about the aggregate, not about the read."""
    text = substitute(
        _FIND_DECLARE + _FIND,
        {"src": "f.mp4", "prompt": "a cat", "threshold": "1.5", "dest": "out.mkv"},
    )
    with pytest.raises(FfrwdError) as caught:
        lower(
            resolve(parse(text.text, text.unset)),
            {"f": _probe(), "g": _probe()},
            registry=load_reference(_SNAPSHOT_PATH),
            describes={
                _MODULE: replace(_described(), rows_schema=_FIND_SCHEMA),
                _PROMPT_MODULE: _prompt_described(),
            },
            invoke=_Prompt([1.0, 0.0]),
            read_packets=_Reads(_FIND_ROWS),
        )
    assert caught.value.code is ErrorCode.STREAM_NOT_FOUND
    assert caught.value.message.startswith("this COPY has nothing to write: no row matched")
    # Both branches' own predicates are named, each with the threshold the
    # variable carried into it.
    assert "v.label <> 'clip'" in caught.value.message
    assert "w.space = 1" in caught.value.message
    assert caught.value.message.count("1.5") == 2


@pytest.mark.parametrize(
    ("where", "kept"),
    [
        ("v.start_t BETWEEN 1 AND 2", [2]),
        ("NOT (v.index = 1 OR v.index = 3)", [2]),
        ("v.keyframe", [1, 2, 3]),
        ("v.start_t IS NOT NULL AND v.index <> 2", [1, 3]),
        ("round(v.start_t, 0) = 2", [2]),
    ],
)
def test_the_row_predicate_grammar_reaches_packet_rows_whole(
    where: str, kept: list[int]
) -> None:
    """Not a narrower grammar than any other row table's: the same
    conjunctions, negations, ranges, null tests, bare boolean columns and
    computed operands, over the columns the declaration named."""
    assert _rows(
        f"SELECT v.index FROM input('f.mp4') f, keys(f.video[1]) v WHERE {where}"
    ) == [[n] for n in kept]


def test_order_by_and_limit_rank_packet_rows() -> None:
    """Ranking by a score rather than filtering by one: the shape a search
    takes when it wants the best n rather than everything above a cutoff."""
    assert _rows(
        "SELECT v.index FROM input('f.mp4') f, keys(f.video[1]) v "
        "ORDER BY cos_similarity(v.vector, prompt('a cat')) DESC, v.index LIMIT 2",
        declare=_DECLARE + _PROMPT_DECLARE,
    ) == [[1], [3]]


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


def test_a_read_that_ran_out_of_time_is_not_what_the_memo_remembers() -> None:
    """Only an answer is remembered. A read the budget cut short is a
    refusal, and the next compile of the same query runs it again rather than
    handing the failure back -- which is what makes raising the budget and
    compiling again work in a process that outlives one compile."""
    sql = "SELECT v.index FROM input('f.mp4') f, keys(f.video[1]) v"
    timed_out = FfrwdError(
        ErrorCode.UNSUPPORTED_SQL,
        "reading 'f.mp4' through 'modules/keys.wasm' did not finish within 120s",
        hint=f"raise {TIMEOUT_ENV}",
    )
    with pytest.raises(FfrwdError):
        _rows(sql, reads=_Fails(timed_out))

    again = _Reads()
    assert _rows(sql, reads=again) == [[1], [2], [3]]
    assert len(again.reads) == 1


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


_CLAMP = r"setts=pts=max(PTS\,0):dts=max(DTS\,0)"


@pytest.mark.parametrize(
    ("wants", "expected"),
    [
        ("all", ["-c", "copy", "-bsf:0", _CLAMP]),
        ("keyframes", ["-c", "copy", "-bsf:0", f"noise=drop=not(key),{_CLAMP}"]),
        ("first", ["-frames:0", "1", "-c", "copy", "-bsf:0", _CLAMP]),
    ],
)
def test_what_each_want_copies(wants: str, expected: list[str]) -> None:
    """`wants` shapes the copy and nothing else. Every case keeps the clock
    filter: the copy has to land on the timestamps a run-time graph over the
    same stream is handed, whatever the module asked to be handed.

    No `-copyts`, which is what puts it there: ffmpeg subtracts the input's
    start on the way in, at run time and here alike, so the demuxer does the
    conversion and the read needs to know nothing about where the file starts.
    """
    read = PacketRead(
        spec="f.mp4", input_args=("-r", "30"), kind="video", index=1,
        module=_MODULE, params="", wants=wants,  # type: ignore[arg-type]
    )
    argv = copy_argv("ffmpeg", read)
    assert argv[:8] == ["ffmpeg", "-v", "error", "-r", "30", "-i", "f.mp4", "-map"]
    assert argv[8] == "0:v:1"
    assert argv[9:-5] == expected
    assert argv[-5:] == ["-avoid_negative_ts", "disabled", "-f", "nut", "pipe:1"]
    assert "-copyts" not in argv


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
        [f"noise=drop=not(key),{_CLAMP}"],
        [_CLAMP],
        [],
    ]


def test_the_last_attempt_drops_the_whole_clock_arrangement() -> None:
    """The clamp and the muxer flag are one thing. Keeping the flag after the
    clamp is gone would hand the muxer a timestamp it cannot write and the
    sidecar's reader one it cannot read, so an ffmpeg too old for `setts` gets
    a plain copy on whatever clock the muxer picks."""
    read = PacketRead(
        spec="f.mp4", input_args=(), kind="video", index=0,
        module=_MODULE, params="", wants="all",
    )
    assert copy_argv("ffmpeg", read, 2) == [
        "ffmpeg", "-v", "error", "-i", "f.mp4", "-map", "0:v:0",
        "-c", "copy", "-f", "nut", "pipe:1",
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
# `fauxlate`'s third export: a fake embedder, and the one thing in this repo
# that answers a `RETURNS vector`. It writes eight components, which is what
# `packet_keys` writes too, so a row read out of a stream scores against a
# prompt without either side being told the other's length.
_FAUXLATE = _BUILT / "fauxlate.wasm"
_EXEC_EMBED_DECLARE = (
    "CREATE FUNCTION embed_text(prompt text) RETURNS vector\n"
    f"  AS '{_FAUXLATE.as_posix()}', 'embed_text' LANGUAGE wasm;\n"
)
_EXEC_PROMPT = "a cat sat on the mat"

# The same encode as `keys.mp4`, with its clock moved three ways a real file's
# is: the picture behind the sound in a file that still starts at zero, a whole
# clock carried forward, and a copy cut that opens on a packet presented before
# zero. See `scripts/gen_fixtures.py`.
_OFFSET_FIXTURES = ("keys-late.mkv", "keys-offset.mp4", "keys-cut.mp4", "keys.ts")


@pytest.fixture
def _require_everything() -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg/ffprobe not found on PATH")
    if binaries.ffrwd_wasm_path() is None:
        pytest.skip("ffrwd-wasm not found (uv sync --extra wasm)")
    for module in (_KEYS_MODULE, _HEAD_MODULE, _FAUXLATE):
        if not module.exists():
            pytest.skip(
                f"module missing: {module} (cargo build --target wasm32-wasip2 "
                f"--release, from {_SIDECAR_MODULES})"
            )
    for name in ("keys.mp4", "keys.mkv", *_OFFSET_FIXTURES):
        if not (_FIXTURES / name).exists():
            pytest.skip(
                f"fixture missing: {_FIXTURES / name} (run scripts/gen_fixtures.py first)"
            )


def _table(sql: str) -> list[list[object]]:
    """Compile a bare SELECT for real: real describe, real read."""
    return compile_table_sql(sql)[0].result.rows


def _file_times(path: Path, keyframes_only: bool = False) -> list[float]:
    """The presentation time of every frame as the FILE states it, in shown order.

    The container's own clock, which is what ffprobe reports and what a row
    time is NOT: ffmpeg takes the input's start off before any filter runs.
    Here to be printed beside the graph's times when an assertion fails, so a
    miss says which of the two clocks moved.
    """
    done = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            *(["-skip_frame", "nokey"] if keyframes_only else []),
            "-show_entries", "frame=pts_time", "-of", "json", str(path),
        ],
        capture_output=True, text=True, timeout=_SUBPROCESS_TIMEOUT, check=False,
    )
    assert done.returncode == 0, done.stderr
    return [float(frame["pts_time"]) for frame in json.loads(done.stdout)["frames"]]


def _graph_times(path: Path, keyframes_only: bool = False) -> list[float]:
    """The pts a FILTERGRAPH is handed, which is what `trim` compares against.

    Not the file's own times: ffmpeg re-bases an input whose container starts
    away from zero, and that difference is what a trim written from a row's
    time would otherwise land on the wrong side of. Read off the same
    one-video-stream mapping the trim below uses, because for a container
    that carries discontinuities the start ffmpeg subtracts is the smallest
    over the streams a command actually reads.
    """
    done = subprocess.run(
        ["ffmpeg", "-v", "info", "-i", str(path), "-map", "0:v:0",
         "-vf", "showinfo", "-f", "null", "-"],
        capture_output=True, text=True, timeout=_SUBPROCESS_TIMEOUT, check=False,
    )
    assert done.returncode == 0, done.stderr
    seen = []
    for line in done.stderr.splitlines():
        if "pts_time:" not in line:
            continue
        if keyframes_only and "iskey:1" not in line:
            continue
        seen.append(float(line.split("pts_time:")[1].split()[0]))
    return seen


def _frame_at(path: Path, time: float) -> int:
    """Which frame the graph shows at `time`, counted from the first it is handed.

    Counted rather than divided by a frame rate: a file whose pictures do not
    open at zero would otherwise be off by wherever it does open.
    """
    times = _graph_times(path)
    return min(range(len(times)), key=lambda i: abs(times[i] - time))


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
    truth = _graph_times(source, keyframes_only=True)
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
def test_a_search_over_the_vectors_a_read_returned(_require_everything: None) -> None:
    """The find shape, nothing faked: real ffmpeg copies the keyframes, the
    real sidecar runs the module, the vectors come back off the stream, and a
    real prompt vector scores them. The WHERE that survives decides how many
    trims exist, and video and audio are cut by the same rows."""
    clear_cache()
    source = (_FIXTURES / "keys.mkv").as_posix()
    declare = _EXEC_DECLARE + _EXEC_EMBED_DECLARE
    scored = _table(
        declare
        + "SELECT v.start_t, "
        f"cos_similarity(v.vector, embed_text('{_EXEC_PROMPT}')) AS score\n"
        f"FROM input('{source}') f, packet_keys(f.video[1]) v"
    )
    assert len(scored) > 2, scored
    cutoff = 0.5
    wanted = sorted(float(row[0]) for row in scored if float(row[1]) > cutoff)  # type: ignore[arg-type]
    assert 0 < len(wanted) < len(scored), (wanted, scored)

    g = compile_sql(
        declare
        + "COPY (\n"
        "  SELECT array_agg(ffmpeg.trim(f.video[1],  start => v.start_t, "
        "end => v.start_t + 0.2)),\n"
        "         array_agg(ffmpeg.atrim(f.audio[1], start => v.start_t, "
        "end => v.start_t + 0.2))\n"
        f"  FROM input('{source}') f, packet_keys(f.video[1]) v\n"
        "  WHERE v.keyframe\n"
        f"    AND cos_similarity(v.vector, embed_text('{_EXEC_PROMPT}')) > {cutoff}\n"
        ") TO 'clips.mkv'"
    )
    cut = {
        kind: sorted(float(node.args["start"]) for node in g.nodes.values()
                     if node.filter == kind)
        for kind in ("trim", "atrim")
    }
    assert cut["trim"] == wanted
    assert cut["atrim"] == wanted


@pytest.mark.exec
@pytest.mark.parametrize("name", ["keys.mp4", "keys.mkv", *_OFFSET_FIXTURES])
def test_a_reported_time_is_the_time_a_trim_means(
    _require_everything: None, name: str
) -> None:
    """The clock rule, over every shape whose clock is moved.

    A trim written from a time this read reported lands on THAT keyframe of
    the graph, byte for byte. The fixtures put the offset somewhere different
    each time -- a picture behind its sound, a packager's whole-clock offset,
    a copy cut into a packet presented before zero, an MPEG-TS remux whose
    start the muxer chose -- and a read that re-based on any one of them
    lands somewhere else.
    """
    clear_cache()
    source = _FIXTURES / name
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
    times = _graph_times(source, keyframes_only=True)
    # An empty trim means the reported time is not a time the graph has a
    # frame at, and the two clocks below say which of them moved.
    assert trimmed, (
        f"the trim keeps at least one frame\n"
        f"  reported  : {start}\n"
        f"  keyframes : {times[:6]}\n"
        f"  the file  : {_file_times(source)[:6]}\n"
        f"  the graph : {_graph_times(source)[:6]}"
    )
    # The reported time is a keyframe the graph has, and the frame a trim from
    # it opens on is that keyframe -- not the one a shifted clock would name.
    at = min(range(len(times)), key=lambda i: abs(times[i] - start))
    assert abs(times[at] - start) < 0.002, (start, times)
    assert trimmed[0] == whole[_frame_at(source, start)]


@pytest.mark.exec
@pytest.mark.parametrize("name", _OFFSET_FIXTURES)
@pytest.mark.parametrize("wants", ["all", "keyframes", "first"])
def test_a_read_stays_on_the_clock_the_graph_is_handed(
    _require_everything: None, name: str, wants: str
) -> None:
    """A stream that does not open at zero is read on the times a run-time
    graph over it is handed, not on times counted from its own first packet
    and not on the times its container states.

    Every shape here is the same encode as `keys.mp4` with its clock moved,
    and every one of them is ordinary: a picture behind its sound, a packager's
    timestamp offset, a copy cut, an MPEG-TS remux. What the copy hands the
    module has to be what a filter over that stream would see, whichever of
    the three `wants` shaped it -- the amount of the stream a module asked for
    is not a clock.

    A copy cut opens on a packet presented BEFORE zero, which is the one time
    a NUT pipe cannot carry; that one packet arrives at zero instead, which is
    where the graph itself already starts.
    """
    clear_cache()
    source = _FIXTURES / name
    read = read_packet_rows(
        PacketRead(
            spec=source.as_posix(), input_args=(), kind="video", index=0,
            module=str(_KEYS_MODULE), params="",
            wants=cast(SinkWants, wants),
        )
    )
    seen = [float(cast(float, row["start_t"])) for row in read]
    assert seen, (name, wants)
    if wants == "first":
        assert abs(seen[0] - _graph_times(source)[0]) < 0.002, (seen, name)
        return
    for expected in _graph_times(source, keyframes_only=True):
        assert any(abs(expected - time) < 0.002 for time in seen), (expected, seen)


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
