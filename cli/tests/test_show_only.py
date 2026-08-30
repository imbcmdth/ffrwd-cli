"""Closing the windows of a ``--show-only`` run ends it.

Unit tier: the waiting and the watching are handed process stand-ins, so
what decides to stop is exercised without spawning ffmpeg or a player.
Nothing here reads a file or opens a window; the live case -- a webcam, a
real ffplay, a real close -- is verified by hand.
"""

from __future__ import annotations

import importlib
import subprocess

import pytest

from ffrwd import cli
from ffrwd.compiler import Compiled
from ffrwd.execute import ExecutionResult, PlanResult, _await_ffmpeg, _Member, _watch
from ffrwd.ir import Graph, Node, Output, SinkUnit
from ffrwd.processes import external_ids, partition

# The package exports `execute` the function under the name its module has,
# so the module itself is fetched rather than imported.
execute_module = importlib.import_module("ffrwd.execute")


class _Stub:
    """A process stand-in that exits with `code` after `after` polls.

    `code` of None never exits, which is what a live capture does until
    something stops it.
    """

    def __init__(self, code: int | None = None, after: int = 0) -> None:
        self.args = ["ffmpeg"]
        self.returncode: int | None = None
        self.waited = False
        self._code = code
        self._after = after
        self._polls = 0

    def poll(self) -> int | None:
        self._polls += 1
        if self._code is not None and self._polls > self._after:
            self.returncode = self._code
        return self.returncode

    def wait(self, timeout: float | None = None) -> int | None:
        self.waited = True
        self.returncode = 0 if self._code is None else self._code
        return self.returncode


@pytest.fixture
def _no_kill(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    """`_end_tree`, caught rather than run: the stubs have no real pids."""
    ended: list[object] = []
    monkeypatch.setattr(execute_module, "_end_tree", ended.append)
    return ended


# --- one command, one window ------------------------------------------------


def test_a_closed_window_ends_the_command_cleanly(_no_kill: list[object]) -> None:
    """Nothing observes the run and nothing is written, so it is over -- and
    over is not a failure."""
    ffmpeg = _Stub()
    window = _Stub(code=0)

    assert _await_ffmpeg(ffmpeg, timeout=None, watching=window) == 0
    assert _no_kill == [ffmpeg]


def test_ffmpegs_own_exit_still_wins(_no_kill: list[object]) -> None:
    """A command that failed reports its code, window or no window."""
    ffmpeg = _Stub(code=3)
    window = _Stub(code=0)

    assert _await_ffmpeg(ffmpeg, timeout=None, watching=window) == 3
    assert _no_kill == []


def test_a_closed_window_ends_nothing_without_show_only() -> None:
    """`watching` of None is a plain wait: the files are still being written,
    so a closed window is the window's business alone."""
    ffmpeg = _Stub(code=0)

    assert _await_ffmpeg(ffmpeg, timeout=None, watching=None) == 0
    assert ffmpeg.waited


def test_the_timeout_still_strikes_while_watching() -> None:
    """An explicit timeout bounds a shown run exactly as it bounds any other."""
    with pytest.raises(subprocess.TimeoutExpired):
        _await_ffmpeg(_Stub(), timeout=0.0, watching=_Stub())


# --- a plan's stage ---------------------------------------------------------


def _member(name: str, proc: _Stub) -> _Member:
    return _Member(id=name, argv=["ffmpeg"], proc=proc)


def test_a_stage_stops_when_its_last_window_closes() -> None:
    """No member ended it, so nothing failed: the caller stops what is still
    running and reports the stage as done."""
    running = _member("ffmpeg0", _Stub())
    closed = [_Stub(code=0), _Stub(code=0)]

    assert _watch([running], float("inf"), closed) == (None, False, None)


def test_a_stage_runs_on_while_one_window_is_open() -> None:
    """Two windows, one still watching: the run is still being watched."""
    ending = _member("ffmpeg0", _Stub(code=0, after=1))
    windows = [_Stub(code=0), _Stub()]

    assert _watch([ending], float("inf"), windows) == (None, False, None)
    assert ending.proc.returncode == 0  # the member ended it, not the windows


def test_a_failing_member_wins_over_closed_windows() -> None:
    """The viewer closing a window as a member dies must not report success."""
    failing = _member("sidecar0", _Stub(code=2))

    assert _watch([failing], float("inf"), [_Stub(code=0)]) == ("sidecar0", False, None)


# --- the flag reaches the runner --------------------------------------------


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


QUERY = "COPY (SELECT a.video[1] FROM input('x.mp4') a) TO 'ignored.mkv'"


@pytest.fixture
def _asked(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    """Whether `run` told `execute` the windows are all it feeds."""
    seen: list[object] = []

    def _capture(emitted: object, **kw: object) -> ExecutionResult:
        seen.append(kw["show_only"])
        return ExecutionResult()

    monkeypatch.setattr(cli.binaries, "ffmpeg_path", lambda: "/usr/bin/ffmpeg")
    monkeypatch.setattr(cli.binaries, "ffplay_path", lambda: "/usr/bin/ffplay")
    monkeypatch.setattr(cli, "execute", _capture)
    monkeypatch.setattr(cli, "compile_all", lambda text, **kw: Compiled([_graph()]))
    return seen


def test_show_only_says_the_windows_are_the_run(_asked: list[object]) -> None:
    assert cli.main(["run", "--show-only", QUERY]) == 0
    assert _asked == [True]


def test_show_keeps_writing_after_a_window_closes(_asked: list[object]) -> None:
    """`--show` writes files, so its windows are not what the run is for."""
    assert cli.main(["run", "--show", QUERY]) == 0
    assert _asked == [False]


def test_a_run_with_no_window_never_asks(_asked: list[object]) -> None:
    assert cli.main(["run", QUERY]) == 0
    assert _asked == [False]


def test_a_plan_is_told_the_same_thing(monkeypatch: pytest.MonkeyPatch) -> None:
    """A query reaching a module runs as a plan, and its windows are the run
    under `--show-only` exactly as one command's are."""
    seen: list[object] = []

    def _capture(plan: object, **kw: object) -> PlanResult:
        seen.append(kw["show_only"])
        return PlanResult()

    graph = _graph()
    graph.nodes["m0"] = Node(
        id="m0", filter="m0.wasm", args={}, inputs=["src:a:v:0"], outputs=["video"]
    )
    graph.sinks = [
        SinkUnit(
            outputs=[Output(ref="m0", type="video", name=None, metadata={})],
            path="sink.mkv",
        )
    ]
    monkeypatch.setattr(cli.binaries, "ffmpeg_path", lambda: "/usr/bin/ffmpeg")
    monkeypatch.setattr(cli.binaries, "ffplay_path", lambda: "/usr/bin/ffplay")
    monkeypatch.setattr(cli, "execute_plan", _capture)
    monkeypatch.setattr(
        cli,
        "compile_all",
        lambda text, **kw: Compiled(
            [_graph()], plan=partition(graph, external=external_ids("m0"))
        ),
    )

    assert cli.main(["run", "--show-only", QUERY]) == 0
    assert seen == [True]
