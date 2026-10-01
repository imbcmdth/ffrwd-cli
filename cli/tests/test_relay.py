"""The relay, driven the way a run drives it (ffrwd.relay).

The processes on either end are this one, opening the pipe paths as files,
which is exactly what ffmpeg and the sidecar do. Skipped where the ffrwd-wasm
sidecar, which the relay is a mode of, is not installed.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import pytest

from ffrwd import binaries, pipes
from ffrwd.errors import FfrwdError
from ffrwd.execute import Flow, _heard
from ffrwd.processes import StreamEdge, VideoFormat
from ffrwd.relay import Relay, RelayEdge

pytestmark = pytest.mark.skipif(
    binaries.ffrwd_wasm_path() is None, reason="the ffrwd-wasm sidecar is not installed"
)

_SOON = 10.0


@pytest.fixture
def relay() -> object:
    running = Relay.start()
    yield running
    running.close()


def _edge(tmp_path: Path, name: str, *, depth: int = 1 << 16, spool: bool = False) -> RelayEdge:
    return RelayEdge(
        id=name,
        source=pipes.path(tmp_path, f"{name}-in"),
        dest=pipes.path(tmp_path, f"{name}-out"),
        depth=depth,
        buffer=1 << 16,
        spool=spool,
    )


def _flow() -> Flow:
    return Flow(
        edge=StreamEdge(source="a", target="b", ref="v", format=VideoFormat()),
        at=time.monotonic(),
    )


def _open(relay: Relay, edge: RelayEdge) -> Flow:
    flow = _flow()
    relay.open([edge], {edge.id: lambda row: _heard(flow, row)}, time.monotonic() + _SOON)
    return flow


def _until(check: object, seconds: float = _SOON) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if check():  # type: ignore[operator]
            return True
        time.sleep(0.02)
    return False


def _read_all(path: str, into: dict[str, object]) -> None:
    try:
        with open(path, "rb", buffering=0) as stream:
            got = bytearray()
            while chunk := stream.read(1 << 16):
                got += chunk
            into["bytes"] = bytes(got)
    except Exception as err:  # noqa: BLE001 -- the assertion reports it whole
        into["error"] = repr(err)


def test_every_byte_crosses_in_order_and_the_flow_counts_it(
    relay: Relay, tmp_path: Path
) -> None:
    """A writer sending in bursts, the pipe empty between them, past what
    either pipe holds: everything arrives, in order, and the flow says so."""
    edge = _edge(tmp_path, "slow")
    flow = _open(relay, edge)
    sent = bytes(range(256)) * 4096  # 1 MiB, no two neighbours alike
    received: dict[str, object] = {}
    consumer = threading.Thread(target=_read_all, args=(edge.dest, received), daemon=True)
    consumer.start()
    with open(str(edge.source), "wb", buffering=0) as producer:
        for at in range(0, len(sent), 1 << 18):
            producer.write(sent[at : at + (1 << 18)])
            time.sleep(0.05)
    consumer.join(_SOON)
    assert received == {"bytes": sent}
    assert _until(lambda: flow.moved == len(sent))
    assert flow.began is not None


def test_a_read_hands_back_what_has_arrived_rather_than_a_full_chunk(
    relay: Relay, tmp_path: Path
) -> None:
    """Nothing is held back: the producer writes far less than a read asks
    for and stops with its pipe still open, and the consumer has those bytes
    at once. Waiting for the rest would wedge a plan whose writer is itself
    waiting on the frame those bytes finish."""
    edge = _edge(tmp_path, "partial")
    _open(relay, edge)
    with open(str(edge.source), "wb", buffering=0) as producer:
        producer.write(b"c" * 300)
        with open(str(edge.dest), "rb", buffering=0) as consumer:
            assert consumer.read(1 << 16) == b"c" * 300


@pytest.mark.skipif(
    sys.platform != "win32",
    reason="a FIFO writer waits for its reader, so the producer cannot leave first",
)
def test_a_producer_that_came_and_went_still_hands_over_its_bytes(
    relay: Relay, tmp_path: Path
) -> None:
    """The producer opens the pipe the moment it exists, writes, and closes
    before the consumer has opened anything: what it wrote arrives, then the
    end of the stream."""
    edge = _edge(tmp_path, "fast")
    _open(relay, edge)
    with open(str(edge.source), "wb", buffering=0) as producer:
        producer.write(b"b" * 1000)
    received: dict[str, object] = {}
    _read_all(str(edge.dest), received)
    assert received == {"bytes": b"b" * 1000}


def test_a_deep_edge_reads_ahead_of_a_consumer_that_has_not_opened(
    relay: Relay, tmp_path: Path
) -> None:
    """The depth is read before the consumer is there at all: a producer
    whose first output nobody takes yet still gets that far, and the flow
    shows the copy waiting on the consumer's end."""
    depth = 4 << 20
    edge = _edge(tmp_path, "deep", depth=depth)
    flow = _open(relay, edge)
    wrote = threading.Event()

    def produce() -> None:
        with open(str(edge.source), "wb", buffering=0) as producer:
            producer.write(b"d" * (2 << 20))
            wrote.set()
            producer.write(b"d" * (2 << 20))

    threading.Thread(target=produce, daemon=True).start()
    assert wrote.wait(_SOON), "the producer was held before the depth was read"
    assert _until(lambda: flow.opening)
    received: dict[str, object] = {}
    _read_all(str(edge.dest), received)
    assert received == {"bytes": b"d" * (4 << 20)}


def test_a_spool_never_holds_its_producer(relay: Relay, tmp_path: Path) -> None:
    """A rows document is read whole when its consumer opens it, so its
    producer must never be the one waiting, however much it writes first."""
    edge = _edge(tmp_path, "rows", spool=True)
    _open(relay, edge)
    rows = b'{"n": 1}\n' * 800_000  # 7 MiB, past what is held in memory
    with open(str(edge.source), "wb", buffering=0) as producer:
        producer.write(rows)
    received: dict[str, object] = {}
    _read_all(str(edge.dest), received)
    assert received == {"bytes": rows}


def test_a_stop_ends_a_copy_whose_consumer_never_came(relay: Relay, tmp_path: Path) -> None:
    """A stage that ends with a consumer that never opened its end: the stop
    ends the copy, and the producer's next write fails rather than hanging."""
    edge = _edge(tmp_path, "abandoned")
    flow = _open(relay, edge)
    failed: dict[str, object] = {}

    def produce() -> None:
        try:
            with open(str(edge.source), "wb", buffering=0) as producer:
                while True:
                    producer.write(b"e" * (1 << 16))
        except OSError as err:
            failed["error"] = repr(err)

    producing = threading.Thread(target=produce, daemon=True)
    producing.start()
    assert _until(lambda: flow.opening)
    relay.stop([edge.id])
    producing.join(_SOON)
    assert not producing.is_alive()
    assert "error" in failed


def test_a_batch_the_relay_cannot_make_is_refused(relay: Relay, tmp_path: Path) -> None:
    """A pipe that cannot be made fails its batch with the relay's reason,
    and the relay carries on with the next."""
    bad = RelayEdge(
        id="bad",
        source=str(tmp_path / "no-such-dir" / "x") if sys.platform != "win32" else "C:\\nope",
        dest=pipes.path(tmp_path, "bad-out"),
        depth=1 << 16,
        buffer=1 << 16,
    )
    with pytest.raises(FfrwdError):
        relay.open([bad], {}, time.monotonic() + _SOON)
    good = _edge(tmp_path, "after")
    _open(relay, good)
    with open(str(good.source), "wb", buffering=0) as producer:
        producer.write(b"f" * 10)
    received: dict[str, object] = {}
    _read_all(str(good.dest), received)
    assert received == {"bytes": b"f" * 10}


def test_a_run_without_the_sidecar_is_refused_by_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(binaries, "ffrwd_wasm_path", lambda: None)
    with pytest.raises(FfrwdError) as caught:
        Relay.start()
    assert "ffrwd-wasm" in caught.value.message
