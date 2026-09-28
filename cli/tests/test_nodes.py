"""A plan placed on several nodes: the cut edges' TCP connections, their
opening exchange, and the coordinator over one runner per node.

Unit tier: no ffmpeg. The runs here are of plans whose every process is a
region the sidecar hook renders, and the hook renders each as this
interpreter running a few lines, so a run crosses real runners, real
sockets and real pipes and nothing else. Real plans run split in the exec
tier (tests/exec/test_exec_split.py).
"""

from __future__ import annotations

import io
import math
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import IO

import pytest

from ffrwd.errors import ErrorCode, FfrwdError
from ffrwd.execute import _End, _pump, _SocketEnd
from ffrwd.nodes import CUT_HELLO, Listener, dial, execute_split, new_secret
from ffrwd.placement import Placement, place
from ffrwd.processes import ProcessPlan, SidecarProcess, StreamEdge, VideoFormat

_PAYLOAD = bytes(range(256)) * 4096


class _Bytes(_End):
    """An end over an in-memory stream."""

    def __init__(self, stream: IO[bytes]) -> None:
        self.stream = stream

    def open(self, deadline: float) -> IO[bytes]:
        return self.stream

    def close(self) -> None:
        pass


class _Kept(io.BytesIO):
    """A BytesIO whose close keeps what was written, for reading after."""

    def close(self) -> None:
        pass


def _listening(secret: str, *keys: str) -> Listener:
    return Listener("127.0.0.1", secret, keys)


def _refusal(listener: Listener) -> str:
    deadline = time.monotonic() + 5
    while not listener.refused and time.monotonic() < deadline:
        time.sleep(0.01)
    return listener.refused[0]


def test_a_cut_edge_carries_the_bytes_it_was_given_end_to_end() -> None:
    secret = new_secret()
    listener = _listening(secret, "e0")
    try:
        closed = threading.Event()
        taken = _SocketEnd(lambda d, c: listener.take("e0", d, c), "rb")
        dialed = _SocketEnd(lambda d, c: dial(listener.address, secret, "e0", d, c), "wb")
        received = _Kept()
        consumer = threading.Thread(target=_pump, args=(taken, _Bytes(received), math.inf))
        consumer.start()
        _pump(_Bytes(io.BytesIO(_PAYLOAD)), dialed, time.monotonic() + 10)
        consumer.join(10)
        assert received.getvalue() == _PAYLOAD
        assert listener.refused == []
        assert not closed.is_set()
    finally:
        listener.close()


def test_a_connection_with_the_wrong_secret_is_closed_before_its_data_is_read() -> None:
    secret = new_secret()
    listener = _listening(secret, "e0")
    try:
        with pytest.raises(ConnectionRefusedError):
            dial(listener.address, new_secret(), "e0", time.monotonic() + 5, threading.Event())
        assert _refusal(listener) == "the wrong secret"
        with pytest.raises(TimeoutError):
            listener.take("e0", time.monotonic() + 0.2, threading.Event())
    finally:
        listener.close()


def test_a_connection_naming_an_edge_this_node_does_not_listen_for_is_closed() -> None:
    secret = new_secret()
    listener = _listening(secret, "e0")
    try:
        with pytest.raises(ConnectionRefusedError):
            dial(listener.address, secret, "e7", time.monotonic() + 5, threading.Event())
        assert _refusal(listener) == "an edge this node does not listen for: e7"
    finally:
        listener.close()


def test_a_second_connection_for_one_edge_is_closed() -> None:
    secret = new_secret()
    listener = _listening(secret, "e0")
    try:
        first = dial(listener.address, secret, "e0", time.monotonic() + 5, threading.Event())
        with pytest.raises(ConnectionRefusedError):
            dial(listener.address, secret, "e0", time.monotonic() + 5, threading.Event())
        assert _refusal(listener) == "a second connection for e0"
        first.close()
    finally:
        listener.close()


def test_a_connection_that_sends_data_first_is_closed_unread() -> None:
    """Bytes that are not an opening line are never handed to an edge."""
    secret = new_secret()
    listener = _listening(secret, "e0")
    try:
        with socket.create_connection(listener.address) as raw:
            raw.sendall(b"\x00" * 64 + b"\n" + _PAYLOAD[:1024])
            raw.settimeout(5)
            # Closed with the rest unread: an end of stream, or a reset.
            try:
                assert raw.recv(16) == b""
            except ConnectionResetError:
                pass
        assert _refusal(listener) == "not a cut edge's opening line"
        with pytest.raises(TimeoutError):
            listener.take("e0", time.monotonic() + 0.2, threading.Event())
    finally:
        listener.close()


def test_the_opening_line_names_its_version() -> None:
    assert CUT_HELLO == "FFRWD-CUT 1"


# -- runs


_PRODUCE = "import sys; sys.stdout.buffer.write(bytes(range(256)) * 4096); sys.stdout.flush()"


def _consume(path: Path) -> str:
    return (
        "import sys; data = sys.stdin.buffer.read(); "
        f"open({str(path)!r}, 'wb').write(data)"
    )


def _python(code: str) -> list[str]:
    return [sys.executable, "-c", code]


def _pair(tmp_path: Path, consumer: str | None = None) -> tuple[ProcessPlan, Path]:
    out = tmp_path / "received.bin"
    plan = ProcessPlan(
        processes=(
            SidecarProcess(id="sidecar0", module=_PRODUCE, node="p", outputs=("video",)),
            SidecarProcess(
                id="sidecar1", module=consumer or _consume(out), node="c", inputs=("n0",)
            ),
        ),
        edges=(
            StreamEdge(source="sidecar0", target="sidecar1", ref="n0", format=VideoFormat()),
        ),
    )
    return plan, out


def _hook(process: SidecarProcess, reads: Sequence[str], writes: Sequence[str]) -> list[str]:
    return _python(process.module)


def test_a_plan_split_on_two_nodes_hands_its_bytes_across(tmp_path: Path) -> None:
    plan, out = _pair(tmp_path)
    placement = place(plan, "per-process")
    assert placement.count == 2
    result = execute_split(plan, placement, sidecar_argv=_hook, timeout=60)
    assert result.exit_code == 0, [m.stderr for s in result.stages for m in s.members]
    assert out.read_bytes() == _PAYLOAD
    assert [(m.id, m.node, m.exit_code) for m in result.stages[0].members] == [
        ("sidecar0", 0, 0),
        ("sidecar1", 1, 0),
    ]


def test_a_member_failing_on_one_node_ends_the_run_and_is_named_with_its_node(
    tmp_path: Path,
) -> None:
    plan, _ = _pair(tmp_path, consumer="import sys; sys.stdin.buffer.read(10); sys.exit(3)")
    result = execute_split(plan, place(plan, "per-process"), sidecar_argv=_hook, timeout=60)
    assert result.exit_code == 3
    assert result.failure is not None
    assert (result.failure.id, result.failure.node) == ("sidecar1", 1)


def test_a_runner_killed_mid_run_stops_the_rest_and_is_named(tmp_path: Path) -> None:
    """The consumer's node's runner is killed while the consumer reads: the
    coordinator reports that member lost, with its node, and the producer's
    node is stopped."""
    plan, _ = _pair(tmp_path, consumer="import sys, time; sys.stdin.buffer.read(1); time.sleep(60)")
    runners: dict[int, subprocess.Popen[bytes]] = {}

    def kill_the_consumers_runner() -> None:
        deadline = time.monotonic() + 30
        while 1 not in runners and time.monotonic() < deadline:
            time.sleep(0.05)
        time.sleep(2.0)
        runners[1].kill()

    killer = threading.Thread(target=kill_the_consumers_runner)
    killer.start()
    started = time.monotonic()
    result = execute_split(
        plan,
        place(plan, "per-process"),
        sidecar_argv=_hook,
        timeout=60,
        started=lambda node, process: runners.__setitem__(node, process),
    )
    killer.join()
    assert time.monotonic() - started < 30
    assert result.exit_code != 0
    assert result.failure is not None
    assert (result.failure.id, result.failure.node, result.failure.lost) == (
        "sidecar1",
        1,
        True,
    )
    assert all(runner.poll() is not None for runner in runners.values())


def test_a_placement_the_runner_cannot_carry_out_starts_nothing(tmp_path: Path) -> None:
    plan, _ = _pair(tmp_path)
    started: list[int] = []
    with pytest.raises(FfrwdError) as caught:
        execute_split(
            plan,
            Placement(nodes={"sidecar0": 0}),
            sidecar_argv=_hook,
            started=lambda node, process: started.append(node),
        )
    assert caught.value.code is ErrorCode.PLACEMENT_REFUSED
    assert started == []


def test_a_remote_run_places_itself_and_takes_no_target(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from ffrwd import cli

    assert cli.main(["run", "SELECT 1", "--remote", "--target", "split-local"]) == 2
    assert "--target is for a run on this machine" in capsys.readouterr().err

# -- a runner on a machine of its own


class _Said:
    """A control channel that keeps what the runner sends and hears nothing."""

    closed = False

    def __init__(self) -> None:
        self.sent: list[dict[str, object]] = []

    def send(self, message: dict[str, object]) -> None:
        self.sent.append(message)

    def receive(self) -> None:
        return None

    def close(self) -> None:
        pass


def _agent(plan: ProcessPlan, argv: dict[str, list[str]], sidecar: str | None) -> tuple:
    from ffrwd.nodes import _Agent
    from ffrwd.placement import split

    part = split(plan, place(plan, "one"))[0]
    said = _Said()
    job = {
        "type": "job",
        "plan": plan.to_dict(),
        "node": part.to_dict(),
        "argv": argv,
        "sidecar": sidecar,
    }
    return _Agent(said, job, new_secret(), "127.0.0.1"), said  # type: ignore[arg-type]


def _modelled(tmp_path: Path) -> ProcessPlan:
    from ffrwd.processes import ModelBinding

    module = tmp_path / "m.wasm"
    model = tmp_path / "m.onnx"
    module.write_bytes(b"")
    model.write_bytes(b"")
    return ProcessPlan(
        processes=(
            SidecarProcess(
                id="sidecar0",
                module=str(module),
                node="m",
                outputs=("video",),
                models=(ModelBinding(name="m", path=str(model)),),
            ),
        ),
    )


def test_a_runner_names_its_own_sidecar_and_onnx_runtime_in_place_of_the_coordinators(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ffrwd import binaries, nn

    monkeypatch.setattr(binaries, "ffrwd_wasm_path", lambda: "/here/ffrwd-wasm")
    monkeypatch.setattr(
        nn, "spawn_args", lambda: ["-nn-runtime", "/here/ort", "-nn-target", "cuda"]
    )
    plan = _modelled(tmp_path)
    rendered = [
        "/there/ffrwd-wasm",
        "-m",
        "x.wasm",
        "-nn-runtime",
        "/there/ort",
        "-nn-target",
        "cpu",
        "-nn-exclude",
        "dml",
        "-nn",
        "m=/p/m.onnx",
    ]
    agent, _ = _agent(plan, {"sidecar0": rendered}, "/there/ffrwd-wasm")
    try:
        assert agent.argv["sidecar0"] == [
            "/here/ffrwd-wasm",
            "-m",
            "x.wasm",
            "-nn-runtime",
            "/here/ort",
            "-nn-target",
            "cuda",
            "-nn-exclude",
            "dml",
            "-nn",
            "m=/p/m.onnx",
        ]
    finally:
        agent.close()


def test_a_runner_given_a_hooks_argv_leaves_it_as_it_is(tmp_path: Path) -> None:
    plan = _modelled(tmp_path)
    rendered = ["python", "-c", "pass", "-nn-runtime", "/there/ort"]
    agent, _ = _agent(plan, {"sidecar0": rendered}, None)
    try:
        assert agent.argv["sidecar0"] == rendered
    finally:
        agent.close()


def test_a_runner_missing_a_model_file_refuses_before_it_says_ready(tmp_path: Path) -> None:
    plan = _modelled(tmp_path)
    missing = tmp_path / "m.onnx"
    missing.unlink()
    agent, said = _agent(plan, {"sidecar0": ["/there/ffrwd-wasm"]}, "/there/ffrwd-wasm")
    try:
        assert agent.serve() == 1
    finally:
        agent.close()
    assert [message["type"] for message in said.sent] == ["refused"]
    error = said.sent[0]["error"]
    assert isinstance(error, dict)
    assert error["message"] == f"node 0 does not have {missing}"


def test_a_host_node_is_named_not_loaded_and_is_never_missing(tmp_path: Path) -> None:
    """A leaky or a row filter is the sidecar's own node: its module is a
    name, not a file, and a runner does not look for it on disk."""
    plan = ProcessPlan(
        processes=(SidecarProcess(id="sidecar0", module="leaky", node="l", outputs=("video",)),)
    )
    agent, said = _agent(plan, {"sidecar0": ["/there/ffrwd-wasm"]}, "/there/ffrwd-wasm")
    try:
        assert agent.missing() == []
    finally:
        agent.close()


def test_a_runner_that_refuses_is_the_error_not_its_dropped_connection(tmp_path: Path) -> None:
    """The coordinator raises the runner's own reason, whatever order its
    refusal and its closing connection arrive in."""
    from ffrwd import binaries, wasm

    if binaries.ffrwd_wasm_path() is None:
        pytest.skip("the ffrwd-wasm sidecar is not installed")
    missing = tmp_path / "gone.wasm"
    plan = ProcessPlan(
        processes=(
            SidecarProcess(id="sidecar0", module=str(missing), node="m", outputs=("video",)),
        )
    )
    with pytest.raises(FfrwdError) as caught:
        execute_split(
            plan, place(plan, "one"), sidecar_argv=wasm.sidecar_argv, timeout=30, startup=30
        )
    assert caught.value.message == f"node 0 does not have {missing}"
