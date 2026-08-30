"""How long a run may take: the compile's budget, and what the CLI applies.

Unit tier: probes are synthetic and nothing is spawned. A duration is
something ffprobe reported, so a ``ProbeResult`` carrying one is all the
budget needs -- no file, and no ffmpeg.
"""

from __future__ import annotations

import argparse

import pytest

from ffrwd import cli, compiler
from ffrwd.compiler import Compiled, compile_all
from ffrwd.execute import DEFAULT_TIMEOUT, ExecutionResult
from ffrwd.ir import Graph, Output, SinkUnit
from ffrwd.probe import ProbeResult, StreamMeta

ONE_INPUT = "COPY (SELECT a.video[1] FROM input('x.mp4') a) TO 'out.mp4'"
TWO_INPUTS = (
    "COPY (SELECT a.video[1], b.audio[1] FROM input('x.mp4') a, input('y.mp4') b) "
    "TO 'out.mkv'"
)


def _probe(duration: float | None) -> ProbeResult:
    """A file of `duration` seconds carrying one video and one audio track."""
    video = StreamMeta(
        type="video",
        index=0,
        metadata={},
        width=1920,
        height=1080,
        fps="30/1",
        sample_rate=None,
        codec="h264",
        duration=duration,
    )
    audio = StreamMeta(
        type="audio",
        index=0,
        metadata={},
        width=None,
        height=None,
        fps=None,
        sample_rate=48000,
        codec="aac",
        channels=2,
        duration=duration,
    )
    return ProbeResult(streams=[video, audio], duration=duration)


def _probes(
    monkeypatch: pytest.MonkeyPatch, by_path: dict[str, ProbeResult | None]
) -> None:
    monkeypatch.setattr(
        compiler, "probe_path", lambda path, args=(), **kw: by_path[path]
    )


# --- the compile's own budget ----------------------------------------------


def test_the_budget_is_ten_times_the_longest_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 4K encode renders at a fraction of realtime, so a run legitimately
    takes many multiples of its material's length."""
    _probes(monkeypatch, {"x.mp4": _probe(300.0), "y.mp4": _probe(120.0)})

    assert compile_all(TWO_INPUTS).default_timeout == 3000.0


def test_a_short_input_still_gets_the_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ten times a 20-second clip is less than the flat default ever was."""
    _probes(monkeypatch, {"x.mp4": _probe(20.0)})

    assert compile_all(ONE_INPUT).default_timeout == float(DEFAULT_TIMEOUT)


def test_an_input_with_no_duration_has_no_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A camera and a live URL never end, so nothing scales to them."""
    _probes(monkeypatch, {"x.mp4": _probe(None)})

    assert compile_all(ONE_INPUT).default_timeout is None


def test_one_unbounded_input_unbounds_the_whole_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A run is as long as its longest input, and one of these has no length."""
    _probes(monkeypatch, {"x.mp4": _probe(300.0), "y.mp4": _probe(None)})

    assert compile_all(TWO_INPUTS).default_timeout is None


def test_an_input_that_did_not_probe_has_no_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Probing is opportunistic; an input it says nothing about is not a
    short one."""
    _probes(monkeypatch, {"x.mp4": None})

    assert compile_all(ONE_INPUT).default_timeout is None


# --- what the CLI applies ---------------------------------------------------


def _graph(path: str = "sink.mkv") -> Graph:
    return Graph(
        input_paths=["x.mp4"],
        sources={"a": 0},
        nodes={},
        sinks=[
            SinkUnit(
                outputs=[Output(ref="src:a:v:0", type="video", name=None, metadata={})],
                path=path,
            )
        ],
    )


@pytest.fixture
def _applied(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    """The timeout `run` hands `execute`, captured instead of running ffmpeg."""
    seen: list[object] = []

    def _capture(emitted: object, **kw: object) -> ExecutionResult:
        seen.append(kw["timeout"])
        return ExecutionResult()

    monkeypatch.setattr(cli.binaries, "ffmpeg_path", lambda: "/usr/bin/ffmpeg")
    monkeypatch.setattr(cli, "execute", _capture)
    return seen


def _compiles(monkeypatch: pytest.MonkeyPatch, budget: float | None) -> None:
    monkeypatch.setattr(
        cli, "compile_all", lambda text, **kw: Compiled([_graph()], default_timeout=budget)
    )


QUERY = "COPY (SELECT a.video[1] FROM input('x.mp4') a) TO 'ignored.mkv'"


def test_an_unset_timeout_takes_the_compiles_budget(
    monkeypatch: pytest.MonkeyPatch, _applied: list[object]
) -> None:
    _compiles(monkeypatch, 3000.0)

    assert cli.main(["run", QUERY]) == 0
    assert _applied == [3000.0]


def test_an_explicit_timeout_wins_over_the_budget(
    monkeypatch: pytest.MonkeyPatch, _applied: list[object]
) -> None:
    _compiles(monkeypatch, 3000.0)

    assert cli.main(["run", "--timeout", "45", QUERY]) == 0
    assert _applied == [45.0]


def test_an_explicit_timeout_wins_where_there_is_no_budget(
    monkeypatch: pytest.MonkeyPatch, _applied: list[object]
) -> None:
    """A live run is unbounded by default and boundable on demand."""
    _compiles(monkeypatch, None)

    assert cli.main(["run", "--timeout", "45", QUERY]) == 0
    assert _applied == [45.0]


def test_a_run_with_no_budget_runs_with_no_timeout(
    monkeypatch: pytest.MonkeyPatch, _applied: list[object]
) -> None:
    _compiles(monkeypatch, None)

    assert cli.main(["run", QUERY]) == 0
    assert _applied == [None]


def test_the_timeout_resolution_reads_the_flag_first() -> None:
    """The precedence, without a run around it."""
    given = argparse.Namespace(timeout=12.0)
    unset = argparse.Namespace(timeout=None)

    assert cli._timeout(given, Compiled([], default_timeout=900.0)) == 12.0
    assert cli._timeout(unset, Compiled([], default_timeout=900.0)) == 900.0
    assert cli._timeout(unset, Compiled([], default_timeout=None)) is None
