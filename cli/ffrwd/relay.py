"""The relay: the process that carries a run's pipe edges, so that no byte of
a stream crosses this one.

An edge that is not one process's stdout handed to the next one's stdin is a
named pipe at each end, and between them is ``ffrwd-wasm relay``: one per run
on each machine, the sidecar binary in a mode of its own. This module starts
it, hands it a stage's edges at a time, and listens to what it says.

The relay takes its orders as one JSON object per line on its stdin, and
answers on its stderr with rows (:data:`~ffrwd.ir.STDERR_ROW`):

- ``{"edges": [...], "batch": N}`` makes every pipe of the batch and starts
  its copies; ``{"kind": "relay", "ready": N}`` says the pipes are there and
  listening, and only then is a process told to open one. A batch it could
  not set up is ``{"kind": "relay", "batch": N, "error": "..."}``.
- ``{"stop": [ids]}`` ends those copies at once.
- ``{"listen": {"host": HOST, "keys": [...]}}`` opens the node's one data
  port for the cut edges arriving from other nodes (:mod:`ffrwd.nodes`);
  ``{"kind": "relay", "listening": [HOST, PORT]}`` says where it is. A
  connection is taken only with the job's secret, which the relay reads from
  :data:`SECRET_ENV` and never from argv, and a key it listens for, once;
  each one refused is ``{"kind": "relay", "refused": "..."}``.
- Its stdin closing ends the rest, and the relay with them.
- ``{"kind": "flow", "edge": ID, ...}`` is one copy's counters, on every
  change and at most four times a second while bytes move: what
  :class:`~ffrwd.execute.Flow` holds, which is how a stalled or overflowing
  edge is still seen from here.

A relay that dies takes its edges with it, and what it said last is the
failure reported.
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import IO

from . import binaries
from .errors import ErrorCode, FfrwdError
from .ir import STDERR_ROW

__all__ = ["SECRET_ENV", "FlowHeard", "Relay", "RelayEdge"]

# The environment variable a node's job secret travels in, to the runner and
# from it to its relay: never argv, which other users of the machine can list.
SECRET_ENV = "FFRWD_NODE_SECRET"

# How long a relay whose stdin has closed is given to end its copies and exit.
_CLOSE = 5.0
# How many of the relay's own log lines are kept for a failure report.
_TAIL = 40

_ROW = STDERR_ROW.encode()

# Hears one edge's flow row, as the relay wrote it.
FlowHeard = Callable[[Mapping[str, object]], None]


@dataclass(frozen=True)
class RelayEdge:
    """One edge the relay carries.

    `source` is the pipe it makes and reads, which the producing process opens
    and writes; `dest` the one it makes and writes, which the consuming
    process opens and reads. Either may instead be the far side of a cut
    edge, as the relay spells one: ``{"listen": KEY}`` for the bytes another
    node sends here, ``{"dial": [HOST, PORT], "key": KEY}`` for those this
    node sends on. `depth` is how far ahead of the consumer it
    reads, `buffer` how much each pipe holds, and `spool` marks a rows edge,
    whose reader never waits.
    """

    id: str
    source: object
    dest: object
    depth: int
    buffer: int
    spool: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "from": self.source,
            "to": self.dest,
            "depth": self.depth,
            "buffer": self.buffer,
            "spool": self.spool,
        }


class Relay:
    """One running ``ffrwd-wasm relay`` and what it has said."""

    def __init__(self, binary: str, secret: str | None = None) -> None:
        flags = 0
        if sys.platform == "win32":
            flags = subprocess.CREATE_NO_WINDOW
        env = None
        if secret is not None:
            env = {**os.environ, SECRET_ENV: secret}
        self._proc = subprocess.Popen(
            [binary, "relay"],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            bufsize=0,
            creationflags=flags,
            env=env,
        )
        self._change = threading.Condition()
        self._heard: dict[str, FlowHeard] = {}
        self._answers: dict[int, str | None] = {}
        self._fault: str | None = None
        self._listening: tuple[str, int] | None = None
        self.refused: list[str] = []
        self._ended = False
        self._tail: deque[str] = deque(maxlen=_TAIL)
        self._batch = 0
        self._order = threading.Lock()
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()

    @classmethod
    def start(cls, secret: str | None = None) -> Relay:
        """A relay from the installed sidecar, or the refusal saying there is
        none. `secret` is the job's, for a node's relay taking cut edges."""
        binary = binaries.ffrwd_wasm_path()
        if binary is None:
            raise FfrwdError(
                ErrorCode.UNSUPPORTED_SQL,
                "this plan joins processes by named pipes, which the ffrwd-wasm "
                "relay carries, and the ffrwd-wasm sidecar is not installed",
                hint="reinstall ffrwd (the sidecar comes with it on supported "
                f"platforms), or point {binaries.FFRWD_WASM_ENV} at an ffrwd-wasm binary",
            )
        return cls(binary, secret)

    def open(
        self, edges: Sequence[RelayEdge], heard: Mapping[str, FlowHeard], deadline: float
    ) -> None:
        """Have the relay make `edges`' pipes and start copying them.

        Returns once every pipe is there, so a process may open one; `heard`
        hears each edge's flow rows from then on. Raises :class:`FfrwdError`
        when the relay refuses the batch, dies, or says nothing by `deadline`.
        """
        if not edges:
            return
        with self._change:
            self._heard.update(heard)
            self._batch += 1
            batch = self._batch
        self._send({"edges": [edge.to_dict() for edge in edges], "batch": batch})
        with self._change:
            while batch not in self._answers and not self._ended:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise self._failure("it did not make the pipes in time")
                self._change.wait(min(remaining, 1.0))
            if batch not in self._answers:
                raise self._failure("it ended")
            error = self._answers.pop(batch)
        if error is not None:
            raise self._failure(error)

    def listen(self, host: str, keys: Sequence[str], deadline: float) -> tuple[str, int]:
        """Open this node's data port on `host` for the cut edges `keys`;
        the address it is reached at."""
        self._send({"listen": {"host": host, "keys": list(keys)}})
        with self._change:
            while self._listening is None and self._fault is None and not self._ended:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise self._failure("it opened no data port in time")
                self._change.wait(min(remaining, 1.0))
            if self._listening is None:
                raise self._failure("it opened no data port")
            return self._listening

    def stop(self, ids: Sequence[str]) -> None:
        """End those copies at once, whatever they still hold."""
        if ids:
            with contextlib.suppress(OSError, ValueError):
                self._send({"stop": list(ids)})

    def close(self) -> None:
        """End every copy and the relay with them."""
        stdin = self._proc.stdin
        if stdin is not None:
            with contextlib.suppress(OSError, ValueError):
                stdin.close()
        try:
            self._proc.wait(timeout=_CLOSE)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(OSError):
                self._proc.kill()
            with contextlib.suppress(subprocess.TimeoutExpired):
                self._proc.wait(timeout=_CLOSE)
        self._reader.join(_CLOSE)

    @property
    def alive(self) -> bool:
        """Whether the relay is still running its copies."""
        return self._proc.poll() is None

    def _send(self, order: Mapping[str, object]) -> None:
        stdin = self._proc.stdin
        assert stdin is not None  # asked for as a pipe
        line = (json.dumps(order) + "\n").encode()
        with self._order:
            stdin.write(line)
            stdin.flush()

    def _read(self) -> None:
        stderr = self._proc.stderr
        assert stderr is not None  # asked for as a pipe
        try:
            for line in _lines(stderr):
                if line.startswith(_ROW):
                    with contextlib.suppress(ValueError):
                        row = json.loads(line[len(_ROW) :])
                        if isinstance(row, dict):
                            self._row(row)
                            continue
                self._tail.append(line.decode("utf-8", "replace").rstrip())
        finally:
            with self._change:
                self._ended = True
                self._change.notify_all()

    def _row(self, row: Mapping[str, object]) -> None:
        if row.get("kind") == "flow":
            heard = self._heard.get(str(row.get("edge")))
            if heard is not None:
                heard(row)
            return
        if row.get("kind") != "relay":
            return
        with self._change:
            ready = row.get("ready")
            batch = row.get("batch")
            error = row.get("error")
            listening = row.get("listening")
            refused = row.get("refused")
            if isinstance(listening, list) and len(listening) == 2:
                self._listening = (str(listening[0]), int(listening[1]))
            elif isinstance(refused, str):
                self.refused.append(refused)
            elif isinstance(ready, int):
                self._answers[ready] = None
            elif isinstance(batch, int):
                self._answers[batch] = str(error)
            elif "edge" in row:
                # One edge's end would not open: that edge has ended, with
                # its done row to follow, and the relay carries on. What it
                # said is kept for a failure report, as its log lines are.
                self._tail.append(f"edge {row.get('edge')}: {error}")
            elif error is not None:
                self._fault = str(error)
            self._change.notify_all()

    def _failure(self, what: str) -> FfrwdError:
        said = self._fault or "\n".join(self._tail)
        return FfrwdError(
            ErrorCode.INTERNAL,
            f"the relay carrying this run's pipes failed: {what}"
            + (f"\n{said}" if said else ""),
            hint="the relay is ffrwd-wasm's own; FFRWD_DUMP_STDERR keeps every "
            "member's log, and the relay's last words are above",
        )


def _lines(stream: IO[bytes]) -> Iterator[bytes]:
    """Each line `stream` writes, as it arrives."""
    rest = b""
    while chunk := stream.read(1 << 16):
        lines = (rest + chunk).split(b"\n")
        rest = lines.pop()
        yield from lines
    if rest:
        yield rest
