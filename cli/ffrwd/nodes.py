"""Running one plan on several nodes.

A placed plan (:mod:`ffrwd.placement`) runs as one RUNNER per node and a
COORDINATOR over them. A runner is ``ffrwd node``: the plan runner of
:func:`~ffrwd.execute.execute_plan` over its node's processes. It spawns
them, serves their pipes and copies their wires exactly as a run on one
node does. A wire whose other end is on another node is copied into or out
of a TCP connection instead of a pipe (:class:`~ffrwd.execute._SocketEnd`),
so a cut edge is pipe, TCP, pipe, carrying the same NUT bytes. No argv
changes: the coordinator renders every member's argv once, as a run on one
node would, and each runner only names its own pipes in it.

The coordinator applies the run's rules to what the runners report: a stage
ends once every member has, the member that ended it is found from the exit
times with the one-second cascade, stalls and overflows are read off the
copies' counters, and rows are printed as they arrive. It stops everything
on Ctrl-C, on a failure, or at the end.

Control
-------
Each runner dials the coordinator and keeps that one connection for the
run: NDJSON, one message per line, each an object with a ``type``. The
runner's first line is ``hello`` with its node and the job's secret; the
coordinator closes a connection whose secret is wrong before reading more.

From the coordinator: ``job`` (the plan, this node's part of it, the
rendered argv and the run's options), ``peers`` (every node's data
address), ``stage`` (run a stage), ``live`` (a live input's programme has
started flowing on some node), ``stop`` (end the running stage),
``compiled`` (the answer to a ``compile``), ``exit``.

From a runner: ``hello``, ``ready`` (its data address), ``started`` and
``exited`` per member, ``flows`` (every copy's counters and each running
member's CPU time), ``work`` (the progress of the member writing the
destinations), ``row``, ``unheard`` (a feeder port nothing listened on),
``windows-closed``, ``compile`` (a run-time lateral's instance to compile),
``error``, and ``stage-end`` with each member's exit code, argv and stderr.

Cut edges
---------
The consumer's node listens on one data port for all its cut edges; the
producer's node dials it. A connection opens with one line, ``FFRWD-CUT 1
<secret> <edge>``, and the listener closes one that does not name the
job's secret and an edge this node listens for, before reading anything
else; it answers ``OK`` to one that does, and only then does the edge's
first byte cross.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import hmac
import json
import math
import os
import secrets
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import cast

from . import pipes, wasm
from .console import Work, WorkProgress
from .errors import ErrorCode, FfrwdError
from .execute import (
    _FAILED,
    _GRACE,
    _JOIN,
    _POLL,
    DEFAULT_STALL,
    DEFAULT_TIMEOUT,
    CompileInstance,
    Flow,
    PipeEdge,
    PlanResult,
    ProcessResult,
    RemoteProcess,
    RowSink,
    Side,
    SidecarArgv,
    StageResult,
    Wire,
    _cpu_seconds,
    _End,
    _is_live,
    _Laterals,
    _Member,
    _pipe_buffer,
    _print_row,
    _resolve_rows_documents,
    _SocketEnd,
    _StageRun,
    _watch,
    plan_argv,
    stage_result,
    stage_wires,
    terminal_member,
    wires,
)
from .pipes import NamedPipe
from .placement import NodePlan, Placement, check_placement, cut_key, pipe_edges, split
from .processes import ProcessPlan, Stage

__all__ = [
    "CUT_HELLO",
    "SECRET_ENV",
    "Channel",
    "Listener",
    "agent_main",
    "dial",
    "execute_split",
    "new_secret",
]

# The environment variable a runner reads the job's secret from: never argv,
# which other users of the machine can list.
SECRET_ENV = "FFRWD_NODE_SECRET"

# The first word of a cut edge's opening line, and its version.
CUT_HELLO = "FFRWD-CUT 1"
# What a listener answers an opening it accepts.
_ACK = b"OK\n"
# The longest opening line a listener reads before closing the connection.
_HELLO_LIMIT = 256
# How long a connection is given to send its opening line.
_HELLO_WAIT = 5.0
# How long a dial that was refused waits before trying again.
_DIAL_RETRY = 0.05
# How often a runner reports its copies' counters and its members' CPU time.
_REPORT_EVERY = 0.2
# How long the coordinator waits for every runner to dial it and say ready.
_STARTUP = 60.0
# How long the coordinator waits for a runner to report a stopped stage.
_STAGE_END_WAIT = 2 * _GRACE + 2 * _JOIN + 5.0

# How a pipe end is named in the argv the coordinator renders, for the runner
# that makes the pipe to name it for real.
_PIPE_TOKEN = "<ffrwd-pipe {index} {side}>"


def new_secret() -> str:
    """A fresh secret for one job: every connection of the run presents it."""
    return secrets.token_hex(32)


def pipe_token(index: int, side: Side) -> str:
    """The stand-in for the pipe at `side` of the plan's pipe edge `index`."""
    return _PIPE_TOKEN.format(index=index, side=side)


def _address(text: str) -> tuple[str, int]:
    host, _, port = text.rpartition(":")
    return host.strip("[]"), int(port)


def _server(host: str, port: int = 0) -> socket.socket:
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    return socket.create_server((host, port), family=family)


# -- the control connection


class Channel:
    """One NDJSON connection, a message per line. Sending is thread-safe."""

    def __init__(self, connection: socket.socket) -> None:
        connection.settimeout(None)
        self._socket = connection
        self._reader = connection.makefile("rb")
        self._lock = threading.Lock()
        self.closed = False

    def send(self, message: Mapping[str, object]) -> bool:
        """Send one message; False once the far end has gone."""
        data = (json.dumps(message) + "\n").encode()
        with self._lock:
            if self.closed:
                return False
            try:
                self._socket.sendall(data)
            except OSError:
                self.closed = True
                return False
        return True

    def receive(self) -> dict[str, object] | None:
        """The next message, or None once the far end has gone."""
        try:
            line = self._reader.readline()
        except (OSError, ValueError):
            return None
        if not line:
            return None
        try:
            message = json.loads(line)
        except ValueError:
            return None
        return message if isinstance(message, dict) else None

    def close(self) -> None:
        with self._lock:
            self.closed = True
        with contextlib.suppress(OSError):
            self._socket.shutdown(socket.SHUT_RDWR)
        with contextlib.suppress(OSError):
            self._reader.close()
        with contextlib.suppress(OSError):
            self._socket.close()


# -- cut edges


def _opening(secret: str, key: str) -> bytes:
    return f"{CUT_HELLO} {secret} {key}\n".encode()


class Listener:
    """A node's one data port: every cut edge into this node arrives on it.

    Each connection must open with the job's secret and the name of an edge
    this node listens for (`keys`), each edge once; anything else is closed
    before a byte past the opening line is read. `refused` says why each
    closed connection was, in arrival order.
    """

    def __init__(self, host: str, secret: str, keys: Iterable[str], port: int = 0) -> None:
        self._secret = secret.encode()
        self._keys = set(keys)
        self._server = _server(host, port)
        self._server.settimeout(_POLL * 5)
        self.address: tuple[str, int] = self._server.getsockname()[:2]
        self._arrived: dict[str, socket.socket] = {}
        self._taken: set[str] = set()
        self._change = threading.Condition()
        self._closed = False
        self.refused: list[str] = []
        self._thread = threading.Thread(target=self._accept, daemon=True)
        self._thread.start()

    def _accept(self) -> None:
        while not self._closed:
            try:
                connection, _ = self._server.accept()
            except TimeoutError:
                continue
            except OSError:
                return
            threading.Thread(target=self._greet, args=(connection,), daemon=True).start()

    def _greet(self, connection: socket.socket) -> None:
        reason = self._check(connection)
        if reason is not None:
            with self._change:
                self.refused.append(reason)
                self._change.notify_all()
            with contextlib.suppress(OSError):
                connection.close()

    def _check(self, connection: socket.socket) -> str | None:
        """Read the opening line and accept or refuse; why, when refused."""
        connection.settimeout(_HELLO_WAIT)
        line = b""
        try:
            # A byte at a time, so nothing past the opening line is read: a
            # dialer sends nothing more until it is answered, and whatever
            # anything else sends after it is never taken off the socket.
            while not line.endswith(b"\n"):
                if len(line) >= _HELLO_LIMIT:
                    return "an opening line too long"
                byte = connection.recv(1)
                if not byte:
                    return "closed before its opening line"
                line += byte
        except OSError:
            return "no opening line in time"
        words = line.decode("ascii", "replace").split()
        if len(words) != 4 or " ".join(words[:2]) != CUT_HELLO:
            return "not a cut edge's opening line"
        if not hmac.compare_digest(words[2].encode(), self._secret):
            return "the wrong secret"
        key = words[3]
        with self._change:
            if key not in self._keys:
                return f"an edge this node does not listen for: {key}"
            if key in self._arrived or key in self._taken:
                return f"a second connection for {key}"
            try:
                connection.sendall(_ACK)
            except OSError:
                return "gone before it was answered"
            connection.settimeout(None)
            self._arrived[key] = connection
            self._change.notify_all()
        return None

    def take(self, key: str, deadline: float, closed: threading.Event) -> socket.socket:
        """The connection for edge `key`, once it has arrived."""
        with self._change:
            while key not in self._arrived:
                if closed.is_set() or self._closed:
                    raise OSError(f"stopped waiting for {key}")
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"no connection for {key} in time")
                self._change.wait(_POLL * 5)
            self._taken.add(key)
            return self._arrived.pop(key)

    def close(self) -> None:
        with self._change:
            self._closed = True
            left = list(self._arrived.values())
            self._arrived.clear()
            self._change.notify_all()
        with contextlib.suppress(OSError):
            self._server.close()
        for connection in left:
            with contextlib.suppress(OSError):
                connection.close()
        self._thread.join(_JOIN)


def dial(
    address: tuple[str, int],
    secret: str,
    key: str,
    deadline: float,
    closed: threading.Event,
) -> socket.socket:
    """A connection to `address` for edge `key`, opened and answered.

    A refused connection is tried again until `deadline`, since the far
    node may not be listening yet; an opening the listener does not answer
    raises, since a wrong secret does not become right by waiting.
    """
    while True:
        if closed.is_set():
            raise OSError(f"stopped dialing for {key}")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"no connection to {address[0]}:{address[1]} for {key} in time")
        try:
            connection = socket.create_connection(address, timeout=_HELLO_WAIT)
        except OSError:
            time.sleep(_DIAL_RETRY)
            continue
        try:
            connection.sendall(_opening(secret, key))
            answer = b""
            while len(answer) < len(_ACK):
                chunk = connection.recv(len(_ACK) - len(answer))
                if not chunk:
                    break
                answer += chunk
        except OSError:
            connection.close()
            raise
        if answer != _ACK:
            connection.close()
            raise ConnectionRefusedError(
                f"{address[0]}:{address[1]} refused the connection for {key}"
            )
        connection.settimeout(None)
        return connection


# -- the runner


def agent_main(argv: Sequence[str]) -> int:
    """``ffrwd node``: run one node's part of a placed plan for a coordinator.

    Hidden: the coordinator starts it with ``--control`` (where to dial it),
    ``--node`` (which node this is) and ``--listen`` (the address the data
    port binds, never every interface unless told), and the job's secret in
    :data:`SECRET_ENV`.
    """
    parser = argparse.ArgumentParser(prog="ffrwd node")
    parser.add_argument("--control", required=True)
    parser.add_argument("--node", type=int, required=True)
    parser.add_argument("--listen", default="127.0.0.1")
    args = parser.parse_args(list(argv))
    secret = os.environ.get(SECRET_ENV)
    if not secret:
        print(f"error: node: {SECRET_ENV} is not set", file=sys.stderr)
        return 2
    # The coordinator decides when a run stops: a Ctrl-C reaches this
    # runner's console as well, and its members hear it for themselves.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if sys.platform == "win32":
        signal.signal(signal.SIGBREAK, signal.SIG_IGN)
    _end_with_this_process()
    try:
        connection = socket.create_connection(_address(args.control), timeout=_STARTUP)
    except OSError as err:
        print(f"error: node {args.node}: cannot reach the coordinator: {err}", file=sys.stderr)
        return 1
    channel = Channel(connection)
    channel.send({"type": "hello", "node": args.node, "secret": secret})
    job = channel.receive()
    if job is None or job.get("type") != "job":
        channel.close()
        return 1
    agent = _Agent(channel, job, secret, args.listen)
    try:
        return agent.serve()
    finally:
        agent.close()
        channel.close()


def _end_with_this_process() -> None:
    """Put this runner and everything it starts in a job object that ends
    with it, on Windows: a runner that is killed takes its members along
    rather than leaving them to run for nobody."""
    if sys.platform != "win32":
        return
    kernel32 = ctypes.windll.kernel32
    kernel32.CreateJobObjectW.restype = ctypes.c_void_p
    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        return

    class _Limits(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", ctypes.c_uint32),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", ctypes.c_uint32),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", ctypes.c_uint32),
            ("SchedulingClass", ctypes.c_uint32),
        ]

    class _Counters(ctypes.Structure):
        _fields_ = [(name, ctypes.c_uint64) for name in ("r", "w", "o", "rb", "wb", "ob")]

    class _Extended(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _Limits),
            ("IoInfo", _Counters),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    kill_on_close = 0x2000
    extended_limits = 9
    info = _Extended()
    info.BasicLimitInformation.LimitFlags = kill_on_close
    kernel32.SetInformationJobObject.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_uint32,
    ]
    kernel32.AssignProcessToJobObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    if kernel32.SetInformationJobObject(
        job, extended_limits, ctypes.byref(info), ctypes.sizeof(info)
    ):
        kernel32.AssignProcessToJobObject(job, kernel32.GetCurrentProcess())
    # The handle is kept open for as long as this process lives, on purpose:
    # closing it is what ends the job's processes.


@dataclass
class _Compile:
    """One instance compile the runner is waiting on."""

    done: threading.Event = field(default_factory=threading.Event)
    answer: dict[str, object] = field(default_factory=dict)


class _Agent:
    """One node's runner, from the job to the last stage."""

    def __init__(
        self, channel: Channel, job: Mapping[str, object], secret: str, host: str
    ) -> None:
        self._channel = channel
        self._secret = secret
        raw_plan, raw_node = job.get("plan"), job.get("node")
        if not isinstance(raw_plan, dict) or not isinstance(raw_node, dict):
            raise ValueError("a job carries a plan and its node's part of it")
        self.plan = ProcessPlan.from_dict(raw_plan)
        self.part = NodePlan.from_dict(raw_node)
        self.mine = set(self.part.processes)
        self._edges = pipe_edges(self.plan)
        self._index = {edge: index for index, edge in enumerate(self._edges)}
        self._stages = self.plan.stages
        self._assigned = wires(self.plan)
        timeout = job.get("timeout")
        self._timeout = float(timeout) if isinstance(timeout, int | float) else None
        self._overwrite = job.get("overwrite") is True
        raw_players = job.get("players")
        self._players = {
            str(pid): [str(word) for word in words]
            for pid, words in (raw_players.items() if isinstance(raw_players, dict) else ())
            if isinstance(words, list)
        }
        terminal = job.get("terminal")
        self._terminal = terminal if isinstance(terminal, str) else None
        jobs = job.get("jobs")
        self._jobs = jobs if isinstance(jobs, int) and not isinstance(jobs, bool) else None
        dump = job.get("dump")
        self._dump = Path(dump) if isinstance(dump, str) and dump else None
        self._stack = contextlib.ExitStack()
        self._home: Path | None = None
        self._served: dict[tuple[PipeEdge, Side], NamedPipe] = {}
        raw_argv = job.get("argv")
        rendered = {
            str(pid): [str(word) for word in words]
            for pid, words in (raw_argv.items() if isinstance(raw_argv, dict) else ())
            if isinstance(words, list) and pid in self.mine
        }
        self.argv = self._name_pipes(rendered)
        self.listener = Listener(host, secret, [cut.key for cut in self.part.listens])
        self._peers: dict[int, tuple[str, int]] = {}
        self._stop = threading.Event()
        self._stage_stop: threading.Event | None = None
        self._stage_thread: threading.Thread | None = None
        self._live = Flow(edge=self._edges[0], at=0.0) if self._edges else None
        self._compiles: dict[int, _Compile] = {}
        self._compile_count = 0
        self._compile_lock = threading.Lock()

    # -- argv

    def _workspace(self) -> Path:
        if self._home is None:
            self._home = Path(
                self._stack.enter_context(tempfile.TemporaryDirectory(prefix="ffrwd-node-"))
            )
        return self._home

    def _name_pipes(self, rendered: Mapping[str, list[str]]) -> dict[str, list[str]]:
        """Each member's argv with its pipes made and named, and its rows
        documents given files in this node's own directory."""
        tokens: dict[str, str] = {}

        def make(index: int, side: Side) -> str:
            edge = self._edges[index]
            pipe = pipes.create(
                self._workspace(),
                str(len(self._served)),
                writing=side == "read",
                buffer=_pipe_buffer(edge),
            )
            self._stack.callback(pipe.close)
            self._served[(edge, side)] = pipe
            return pipe.path

        def resolve(word: str) -> str:
            arg, sep, rest = word.partition("=")
            if sep and rest in tokens:
                return f"{arg}={tokens[rest]}"
            return tokens.get(word, word)

        sides: tuple[Side, Side] = ("read", "write")
        for index, edge in enumerate(self._edges):
            for side in sides:
                token = pipe_token(index, side)
                owner = edge.target if side == "read" else edge.source
                if owner in self.mine and any(token in words for words in rendered.values()):
                    tokens[token] = make(index, side)

        def rows_path(placeholder: str) -> str:
            name = placeholder.rpartition(":")[2] or "0"
            return str(self._workspace() / f"rows-{name}.ndjson")

        named = {pid: [resolve(word) for word in words] for pid, words in rendered.items()}
        return _resolve_rows_documents(named, rows_path)

    # -- the conversation

    def serve(self) -> int:
        self._channel.send(
            {"type": "ready", "address": [self.listener.address[0], self.listener.address[1]]}
        )
        while True:
            message = self._channel.receive()
            if message is None:
                # The coordinator has gone: nobody is left to report to.
                self._end_stage()
                return 1
            kind = message.get("type")
            if kind == "peers":
                self._take_peers(message.get("addresses"))
            elif kind == "stage":
                index = message.get("index")
                if isinstance(index, int):
                    self._begin_stage(index)
            elif kind == "stop":
                self._end_stage()
            elif kind == "live":
                if self._live is not None and self._live.began is None:
                    self._live.began = time.monotonic()
            elif kind == "compiled":
                self._answered(message)
            elif kind == "exit":
                self._end_stage()
                return 0

    def close(self) -> None:
        self._stop.set()
        self._end_stage()
        for pending in self._compiles.values():
            pending.done.set()
        self.listener.close()
        self._stack.close()

    def _take_peers(self, addresses: object) -> None:
        if not isinstance(addresses, dict):
            return
        for node, address in addresses.items():
            if isinstance(address, list) and len(address) == 2:
                self._peers[int(node)] = (str(address[0]), int(address[1]))

    def _begin_stage(self, index: int) -> None:
        self._end_stage()
        stop = threading.Event()
        self._stage_stop = stop
        if self._live is not None:
            self._live.began = None
        self._stage_thread = threading.Thread(
            target=self._run_stage, args=(self._stages[index], stop), daemon=True
        )
        self._stage_thread.start()

    def _end_stage(self) -> None:
        if self._stage_stop is not None:
            self._stage_stop.set()
        if self._stage_thread is not None:
            self._stage_thread.join()
        self._stage_stop = None
        self._stage_thread = None

    # -- a stage

    def _run_stage(self, stage: Stage, stop: threading.Event) -> None:
        send = self._channel.send
        laterals = _Laterals(
            self._compile if self.plan.laterals else None,
            _instance_sidecar_argv(self._jobs),
            self._row,
            self._dump,
        )
        run = _StageRun(
            self.plan,
            stage,
            self.argv,
            self._served,
            self._assigned,
            self._timeout,
            self._overwrite,
            None,
            self._players,
            work=self._work if self._terminal in self.mine else None,
            terminal=self._terminal,
            laterals=laterals,
            local=self.mine,
            remote=self._cut_end,
            spawned=self._spawned,
        )
        live_here = any(_is_live(wire.edge) for wire in run.stage_wires)
        elsewhere = [self._live] if live_here and self._live is not None else []
        watching = threading.Thread(target=self._report, args=(run, stop), daemon=True)
        try:
            run.start()
            watching.start()
            unheard = run.feed_writers(stop, elsewhere)
            if unheard is not None:
                member, _, error = unheard
                send({"type": "unheard", "process": member, "error": error.to_dict()})
            stop.wait()
        except FfrwdError as err:
            send({"type": "error", "error": err.to_dict()})
            stop.wait()
        except OSError as err:
            send({"type": "error", "error": {"code": "INTERNAL", "message": str(err)}})
            stop.wait()
        finally:
            run.end()
            if watching.is_alive():
                watching.join(_JOIN)
            send(
                {
                    "type": "stage-end",
                    "index": stage.index,
                    "members": [
                        {
                            "process": result.id,
                            "argv": result.argv,
                            "exit_code": result.exit_code,
                            "stderr": result.stderr,
                            "terminated": result.terminated,
                        }
                        for result in run.results()
                    ],
                }
            )

    def _spawned(self, member: _Member) -> None:
        self._channel.send({"type": "started", "process": member.id, "argv": member.argv})

    def _report(self, run: _StageRun, stop: threading.Event) -> None:
        """Each member's exit as it is seen, and every copy's counters and
        each member's CPU time a few times a second, until the stage stops."""
        exited: set[str] = set()
        closed = False
        last = 0.0
        sent: list[object] = []
        while not stop.wait(_POLL):
            for member in list(run.members.values()):
                code = member.proc.poll()
                if code is not None and member.id not in exited:
                    exited.add(member.id)
                    self._channel.send({"type": "exited", "process": member.id, "code": code})
            if run.watching and not closed and all(
                window.poll() is not None for window in run.watching.values()
            ):
                closed = True
                self._channel.send({"type": "windows-closed"})
            now = time.monotonic()
            if now - last < _REPORT_EVERY:
                continue
            last = now
            flows = [self._counters(flow) for flow in list(run.flows)]
            cpu = {
                member.id: _cpu_seconds(member.proc)
                for member in list(run.members.values())
                if member.proc.poll() is None
            }
            if flows != sent or cpu:
                sent = list(flows)
                self._channel.send({"type": "flows", "flows": flows, "cpu": cpu})

    def _counters(self, flow: Flow) -> dict[str, object]:
        edge = flow.edge
        if edge.source in self.mine and edge.target in self.mine:
            side = "local"
        else:
            side = "in" if edge.target in self.mine else "out"
        return {
            "edge": self._index[edge],
            "side": side,
            "moved": flow.moved,
            "began": flow.began is not None,
            "writing": flow.writing,
            "opening": flow.opening,
        }

    def _cut_end(self, wire: Wire, side: Side) -> _End:
        """The TCP end of a wire whose other member is on another node."""
        index = self._index[wire.edge]
        key = cut_key(index)
        if side == "write":
            # The producer is elsewhere: its bytes arrive on this node's port.
            def arrive(deadline: float, closed: threading.Event) -> socket.socket:
                return self.listener.take(key, deadline, closed)

            return _SocketEnd(arrive, "rb")
        consumer = next(cut.consumer for cut in self.part.dials if cut.key == key)
        address = self._peers[consumer]

        def depart(deadline: float, closed: threading.Event) -> socket.socket:
            return dial(address, self._secret, key, deadline, closed)

        return _SocketEnd(depart, "wb")

    def _work(self, reading: Work) -> None:
        self._channel.send({"type": "work", **asdict(reading)})

    def _row(self, row: Mapping[str, object]) -> None:
        self._channel.send({"type": "row", "row": dict(row)})

    def _compile(self, text: str, unset: Mapping[tuple[int, int], str]) -> ProcessPlan:
        """One run-time lateral's instance, compiled by the coordinator,
        which holds the query's packages and probes."""
        with self._compile_lock:
            self._compile_count += 1
            number = self._compile_count
            pending = self._compiles[number] = _Compile()
        self._channel.send(
            {
                "type": "compile",
                "id": number,
                "text": text,
                "unset": [[line, col, name] for (line, col), name in unset.items()],
            }
        )
        while not pending.done.wait(_POLL * 5):
            if self._stop.is_set() or self._channel.closed:
                break
        self._compiles.pop(number, None)
        plan, error = pending.answer.get("plan"), pending.answer.get("error")
        if isinstance(plan, dict):
            return ProcessPlan.from_dict(plan)
        raise _error_from(error if isinstance(error, dict) else {})

    def _answered(self, message: Mapping[str, object]) -> None:
        number = message.get("id")
        pending = self._compiles.get(number) if isinstance(number, int) else None
        if pending is not None:
            pending.answer = dict(message)
            pending.done.set()


def _instance_sidecar_argv(jobs: int | None) -> SidecarArgv:
    """How a runner renders a sidecar for a run-time lateral's instance:
    as the run would, with its ``--jobs``."""

    def render(process: object, reads: Sequence[str], writes: Sequence[str]) -> list[str]:
        from .processes import SidecarProcess

        assert isinstance(process, SidecarProcess)
        return wasm.sidecar_argv(process, reads, writes, jobs=jobs)

    return render


def _error_from(written: Mapping[str, object]) -> FfrwdError:
    """An error as :meth:`FfrwdError.to_dict` wrote it."""
    try:
        code = ErrorCode(str(written.get("code")))
    except ValueError:
        code = ErrorCode.INTERNAL
    hint = written.get("hint")
    line, col = written.get("line"), written.get("col")
    return FfrwdError(
        code,
        str(written.get("message", "a node's runner failed")),
        line=line if isinstance(line, int) else None,
        col=col if isinstance(col, int) else None,
        hint=hint if isinstance(hint, str) else None,
    )


# -- the coordinator

# Starts one node's runner: given the node, the control address to dial and
# the address its data port binds, the process running it.
StartRunner = Callable[[int, str, str, Mapping[str, str]], subprocess.Popen[bytes]]


def start_local_runner(
    node: int, control: str, listen: str, env: Mapping[str, str]
) -> subprocess.Popen[bytes]:
    """A runner as a subprocess of this one, with this interpreter."""
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "ffrwd",
            "node",
            "--control",
            control,
            "--node",
            str(node),
            "--listen",
            listen,
        ],
        env=dict(env),
        stdin=subprocess.DEVNULL,
    )


@dataclass
class _Node:
    """One node's runner as the coordinator knows it."""

    index: int
    part: NodePlan
    process: subprocess.Popen[bytes] | None = None
    channel: Channel | None = None
    address: tuple[str, int] | None = None
    ready: threading.Event = field(default_factory=threading.Event)
    lost: bool = False
    ended: dict[int, list[dict[str, object]]] = field(default_factory=dict)
    change: threading.Condition = field(default_factory=threading.Condition)


class _Stage:
    """The stage the coordinator is watching, as the runners report it."""

    def __init__(self, stage: Stage, flows: Mapping[tuple[int, str], Flow]) -> None:
        self.stage = stage
        self.procs = {pid: RemoteProcess() for pid in stage.processes}
        self.members = {
            pid: _Member(id=pid, argv=[], proc=cast("subprocess.Popen[bytes]", proc))
            for pid, proc in self.procs.items()
        }
        self.started: set[str] = set()
        self.lost: set[str] = set()
        self.flows = flows
        self.windows = RemoteProcess()
        self.abort = threading.Event()
        self.unheard: tuple[str, FfrwdError] | None = None
        self.error: FfrwdError | None = None
        self.live_sent = False


class _Coordinator:
    def __init__(
        self,
        plan: ProcessPlan,
        placement: Placement,
        *,
        argv: Mapping[str, list[str]],
        options: Mapping[str, object],
        echo: Callable[[str, list[str]], None] | None,
        work: WorkProgress | None,
        rows: RowSink,
        compile_instance: CompileInstance | None,
        host: str,
        start: StartRunner,
        started: Callable[[int, subprocess.Popen[bytes]], None] | None,
    ) -> None:
        self.plan = plan
        self.placement = placement
        self._argv = argv
        self._options = options
        self._echo = echo
        self._work = work
        self._rows = rows
        self._compile = compile_instance
        self._host = host
        self._start = start
        self._started = started
        self._secret = new_secret()
        self.nodes = [_Node(index=part.node, part=part) for part in split(plan, placement)]
        self._edges = pipe_edges(plan)
        self._index = {edge: index for index, edge in enumerate(self._edges)}
        self.stage: _Stage | None = None
        self._lock = threading.Lock()
        self._server: socket.socket | None = None
        self._threads: list[threading.Thread] = []

    # -- starting and ending the runners

    def open(self) -> None:
        server = self._server = _server(self._host)
        server.settimeout(_POLL * 5)
        host, port = server.getsockname()[:2]
        control = f"[{host}]:{port}" if ":" in host else f"{host}:{port}"
        env = dict(os.environ)
        env[SECRET_ENV] = self._secret
        for node in self.nodes:
            node.process = self._start(node.index, control, self._host, env)
            if self._started is not None:
                self._started(node.index, node.process)
        waiting = {node.index: node for node in self.nodes}
        deadline = time.monotonic() + _STARTUP
        while waiting:
            if time.monotonic() >= deadline:
                raise FfrwdError(
                    ErrorCode.INTERNAL,
                    f"the runner of node {min(waiting)} never reached the coordinator",
                    hint="check that `python -m ffrwd node` starts on this machine",
                )
            gone = [n for n in waiting.values() if n.process and n.process.poll() is not None]
            if gone:
                raise FfrwdError(
                    ErrorCode.INTERNAL,
                    f"the runner of node {gone[0].index} ended before it reached the "
                    f"coordinator (exit {gone[0].process.poll() if gone[0].process else None})",
                    hint="check that `python -m ffrwd node` starts on this machine",
                )
            try:
                connection, _ = server.accept()
            except TimeoutError:
                continue
            channel = Channel(connection)
            hello = channel.receive() if self._greeted(connection) else None
            index = hello.get("node") if hello else None
            secret = hello.get("secret") if hello else None
            if (
                hello is None
                or hello.get("type") != "hello"
                or not isinstance(index, int)
                or index not in waiting
                or not isinstance(secret, str)
                or not hmac.compare_digest(secret.encode(), self._secret.encode())
            ):
                channel.close()
                continue
            node = waiting.pop(index)
            node.channel = channel
            channel.send(self._job(node))
            thread = threading.Thread(target=self._listen, args=(node,), daemon=True)
            thread.start()
            self._threads.append(thread)
        for node in self.nodes:
            if not node.ready.wait(max(deadline - time.monotonic(), 0.1)) or node.lost:
                raise FfrwdError(
                    ErrorCode.INTERNAL,
                    f"the runner of node {node.index} never said it was ready",
                    hint="check that `python -m ffrwd node` starts on this machine",
                )
        addresses = {
            str(node.index): list(node.address) for node in self.nodes if node.address
        }
        for node in self.nodes:
            self._send(node, {"type": "peers", "addresses": addresses})

    @staticmethod
    def _greeted(connection: socket.socket) -> bool:
        """Whether a hello line arrives in time; a connection that sends none is closed."""
        connection.settimeout(_HELLO_WAIT)
        try:
            ready = connection.recv(1, socket.MSG_PEEK)
        except OSError:
            return False
        finally:
            connection.settimeout(None)
        return bool(ready)

    def _job(self, node: _Node) -> dict[str, object]:
        mine = set(node.part.processes)
        return {
            "type": "job",
            "plan": self.plan.to_dict(),
            "node": node.part.to_dict(),
            "argv": {pid: words for pid, words in self._argv.items() if pid in mine},
            **{
                key: value
                for key, value in self._options.items()
                if key != "players" or node.index == 0
            },
        }

    def close(self) -> None:
        for node in self.nodes:
            self._send(node, {"type": "exit"})
        for node in self.nodes:
            if node.process is None:
                continue
            try:
                node.process.wait(_GRACE * 2)
            except subprocess.TimeoutExpired:
                # A runner is this process's own child: end it and what it runs.
                with contextlib.suppress(OSError):
                    node.process.kill()
                with contextlib.suppress(subprocess.TimeoutExpired):
                    node.process.wait(_GRACE)
        for node in self.nodes:
            if node.channel is not None:
                node.channel.close()
        if self._server is not None:
            with contextlib.suppress(OSError):
                self._server.close()
        for thread in self._threads:
            thread.join(_JOIN)

    def _send(self, node: _Node, message: Mapping[str, object]) -> None:
        if node.channel is not None and not node.lost:
            node.channel.send(message)

    # -- what the runners say

    def _listen(self, node: _Node) -> None:
        assert node.channel is not None
        while True:
            message = node.channel.receive()
            if message is None:
                self._lose(node)
                return
            self._heard(node, message)

    def _lose(self, node: _Node) -> None:
        """A runner's connection closed: every member it ran is lost with it."""
        with node.change:
            node.lost = True
            node.ready.set()
            node.change.notify_all()
        stage = self.stage
        if stage is None:
            return
        for pid in node.part.processes:
            proc = stage.procs.get(pid)
            if proc is not None and proc.code is None:
                stage.lost.add(pid)
                proc.code = _FAILED

    def _heard(self, node: _Node, message: Mapping[str, object]) -> None:
        kind = message.get("type")
        stage = self.stage
        if kind == "ready":
            address = message.get("address")
            if isinstance(address, list) and len(address) == 2:
                node.address = (str(address[0]), int(address[1]))
            node.ready.set()
        elif kind == "stage-end":
            index, members = message.get("index"), message.get("members")
            if isinstance(index, int) and isinstance(members, list):
                with node.change:
                    node.ended[index] = [m for m in members if isinstance(m, dict)]
                    node.change.notify_all()
        elif kind == "compile":
            threading.Thread(target=self._compile_for, args=(node, message), daemon=True).start()
        elif kind == "row":
            row = message.get("row")
            if isinstance(row, dict):
                self._rows(row)
        elif kind == "work":
            if self._work is not None:
                fields = {k: v for k, v in message.items() if k != "type"}
                with contextlib.suppress(TypeError):
                    self._work(Work(**fields))  # type: ignore[arg-type]
        elif stage is not None:
            self._heard_in_stage(node, stage, kind, message)

    def _heard_in_stage(
        self, node: _Node, stage: _Stage, kind: object, message: Mapping[str, object]
    ) -> None:
        pid = message.get("process")
        if kind == "started" and isinstance(pid, str) and pid in stage.members:
            argv = message.get("argv")
            words = [str(w) for w in argv] if isinstance(argv, list) else []
            stage.members[pid].argv = words
            stage.started.add(pid)
            if self._echo is not None:
                self._echo(pid, words)
        elif kind == "exited" and isinstance(pid, str) and pid in stage.procs:
            code = message.get("code")
            if isinstance(code, int) and stage.procs[pid].code is None:
                stage.procs[pid].code = code
        elif kind == "flows":
            self._counted(stage, message)
        elif kind == "unheard" and isinstance(pid, str):
            error = message.get("error")
            stage.unheard = (pid, _error_from(error if isinstance(error, dict) else {}))
            stage.abort.set()
        elif kind == "error":
            error = message.get("error")
            stage.error = _error_from(error if isinstance(error, dict) else {})
            stage.abort.set()
        elif kind == "windows-closed":
            stage.windows.code = 0

    def _counted(self, stage: _Stage, message: Mapping[str, object]) -> None:
        now = time.monotonic()
        cpu = message.get("cpu")
        if isinstance(cpu, dict):
            for pid, seconds in cpu.items():
                proc = stage.procs.get(str(pid))
                if proc is not None and isinstance(seconds, int | float):
                    proc.cpu = float(seconds)
        flows = message.get("flows")
        for counters in flows if isinstance(flows, list) else ():
            if not isinstance(counters, dict):
                continue
            key = (counters.get("edge"), counters.get("side"))
            flow = stage.flows.get(key)  # type: ignore[arg-type]
            if flow is None:
                continue
            moved = counters.get("moved")
            if isinstance(moved, int) and moved > flow.moved:
                flow.moved = moved
                flow.at = now
            if counters.get("began") and flow.began is None:
                flow.began = now
            flow.writing = counters.get("writing") is True
            flow.opening = counters.get("opening") is True
            if flow.began is not None and _is_live(flow.edge) and not stage.live_sent:
                stage.live_sent = True
                for node in self.nodes:
                    self._send(node, {"type": "live"})

    def _compile_for(self, node: _Node, message: Mapping[str, object]) -> None:
        number = message.get("id")
        text = message.get("text")
        raw_unset = message.get("unset")
        unset: dict[tuple[int, int], str] = {}
        for entry in raw_unset if isinstance(raw_unset, list) else ():
            if isinstance(entry, list) and len(entry) == 3:
                unset[(int(entry[0]), int(entry[1]))] = str(entry[2])
        answer: dict[str, object] = {"type": "compiled", "id": number}
        try:
            if self._compile is None or not isinstance(text, str):
                raise FfrwdError(
                    ErrorCode.INTERNAL,
                    "a runner asked for an instance to be compiled, and nothing compiles one",
                )
            answer["plan"] = self._compile(text, unset).to_dict()
        except FfrwdError as err:
            answer["error"] = err.to_dict()
        self._send(node, answer)

    # -- a stage

    def run_stage(
        self,
        stage: Stage,
        *,
        timeout: float | None,
        stall: float | None,
        show_only: bool,
        players: Mapping[str, list[str]],
        stop: threading.Event | None,
    ) -> StageResult:
        assigned = wires(self.plan)
        node_of = self.placement.nodes
        found, writers = stage_wires(
            self.plan,
            stage,
            assigned,
            lambda wire: node_of[wire.edge.source] != node_of[wire.edge.target],
        )
        feeds = [(wire.edge.source, wire.edge.target) for wire in found]
        feeding = {pid: readers for pid, (_, readers) in writers.items()}
        start = time.monotonic()
        flows: dict[tuple[int, str], Flow] = {}
        for wire in found:
            if wire.chained:
                continue
            apart = node_of[wire.edge.source] != node_of[wire.edge.target]
            flows[(self._index[wire.edge], "in" if apart else "local")] = Flow(
                edge=wire.edge, at=start
            )
        current = self.stage = _Stage(stage, flows)
        involved = [n for n in self.nodes if set(n.part.processes) & set(stage.processes)]
        for node in involved:
            if node.lost:
                self._lose(node)
            self._send(node, {"type": "stage", "index": stage.index})
        deadline = math.inf if timeout is None else start + timeout
        if stop is not None:
            _chain(stop, current.abort)
        failed: str | None = None
        timed_out = False
        wedge: FfrwdError | None = None
        interrupted = False
        try:
            failed, timed_out, wedge = _watch(
                current.members.values(),
                deadline,
                [cast("subprocess.Popen[bytes]", current.windows)]
                if show_only and any(pid in players for pid in stage.processes)
                else None,
                list(flows.values()),
                stall,
                feeds,
                feeding,
                current.abort,
            )
        except KeyboardInterrupt:
            interrupted = True
        if current.unheard is not None and not interrupted:
            failed, timed_out, wedge = current.unheard[0], True, current.unheard[1]
        for node in involved:
            self._send(node, {"type": "stop"})
        results = self._collect(stage, current, involved)
        self.stage = None
        if current.error is not None and not interrupted:
            raise current.error
        ended = {
            member.id: member.ended_at
            for member in current.members.values()
            if member.ended_at is not None
        }
        return stage_result(
            stage.index,
            results,
            ended,
            feeds,
            failed=failed,
            timed_out=timed_out,
            wedge=wedge,
            interrupted=interrupted,
        )

    def _collect(
        self, stage: Stage, current: _Stage, involved: Sequence[_Node]
    ) -> list[ProcessResult]:
        """Every member each runner spawned, as it reports them once stopped;
        a lost runner's members as the watch last saw them."""
        reported: dict[str, ProcessResult] = {}
        deadline = time.monotonic() + _STAGE_END_WAIT
        for node in involved:
            with node.change:
                while stage.index not in node.ended and not node.lost:
                    left = deadline - time.monotonic()
                    if left <= 0:
                        break
                    node.change.wait(min(left, _POLL * 5))
                members = node.ended.pop(stage.index, [])
            for written in members:
                pid = str(written.get("process"))
                argv, code = written.get("argv"), written.get("exit_code")
                reported[pid] = ProcessResult(
                    id=pid,
                    argv=[str(w) for w in argv] if isinstance(argv, list) else [],
                    exit_code=code if isinstance(code, int) else _FAILED,
                    stderr=str(written.get("stderr", "")),
                    terminated=written.get("terminated") is True,
                    node=node.index,
                )
        results: list[ProcessResult] = []
        for pid in stage.processes:
            where = self.placement.node(pid)
            if pid in current.lost or (pid not in reported and pid in current.started):
                results.append(
                    ProcessResult(
                        id=pid,
                        argv=current.members[pid].argv,
                        exit_code=_FAILED,
                        stderr=f"the runner of node {where} ended while {pid} was running\n",
                        node=where,
                        lost=True,
                    )
                )
            elif pid in reported:
                results.append(reported[pid])
        return results


def _chain(outer: threading.Event, inner: threading.Event) -> None:
    """Set `inner` once `outer` is set, from a thread of its own."""

    def wait() -> None:
        while not inner.is_set():
            if outer.wait(_POLL * 5):
                inner.set()
                return

    threading.Thread(target=wait, daemon=True).start()


def execute_split(
    plan: ProcessPlan,
    placement: Placement,
    *,
    sidecar_argv: SidecarArgv | None = None,
    timeout: float | None = DEFAULT_TIMEOUT,
    overwrite: bool = False,
    echo: Callable[[str, list[str]], None] | None = None,
    players: Mapping[str, list[str]] | None = None,
    show_only: bool = False,
    stall: float | None = DEFAULT_STALL,
    work: WorkProgress | None = None,
    compile_instance: CompileInstance | None = None,
    rows: RowSink | None = None,
    dump: Path | None = None,
    stop: threading.Event | None = None,
    jobs: int | None = None,
    host: str = "127.0.0.1",
    start: StartRunner = start_local_runner,
    started: Callable[[int, subprocess.Popen[bytes]], None] | None = None,
) -> PlanResult:
    """Run `plan` placed on the nodes of `placement`, one runner per node.

    Takes what :func:`~ffrwd.execute.execute_plan` takes and returns what it
    returns, each member's result naming its node. `host` is the address
    every runner's data port binds and the coordinator listens on: loopback
    for a split on this machine. `start` starts one node's runner, a
    subprocess of this one by default, and `started` hears each as it starts.
    `jobs` is what a runner renders a run-time lateral's sidecars with.

    Refused before anything starts: a placement :func:`check_placement`
    refuses, and a plan whose argv does not render.
    """
    shown = dict(players or {})
    check_placement(plan, placement, shown=list(shown))
    if plan.laterals and compile_instance is None:
        raise FfrwdError(
            ErrorCode.INTERNAL,
            f"{plan.laterals[0].call} is started once per message, and "
            "nothing was given to compile its instances",
            hint="pass compile_instance, which compiles one instance into a plan",
        )
    index = {edge: position for position, edge in enumerate(pipe_edges(plan))}
    argv = plan_argv(
        plan,
        sidecar_argv=sidecar_argv,
        pipe_path=lambda edge, side: pipe_token(index[edge], side),
    )
    options: dict[str, object] = {
        "timeout": timeout,
        "overwrite": overwrite,
        "players": shown,
        "terminal": terminal_member(plan) if work is not None else None,
        "jobs": jobs,
        "dump": str(dump) if dump is not None else None,
    }
    coordinator = _Coordinator(
        plan,
        placement,
        argv=argv,
        options=options,
        echo=echo,
        work=work,
        rows=rows or _print_row,
        compile_instance=compile_instance,
        host=host,
        start=start,
        started=started,
    )
    stages: list[StageResult] = []
    try:
        coordinator.open()
        for stage in plan.stages:
            if stop is not None and stop.is_set():
                break
            result = coordinator.run_stage(
                stage,
                timeout=timeout,
                stall=stall,
                show_only=show_only,
                players=shown,
                stop=stop,
            )
            stages.append(result)
            if result.interrupted:
                return PlanResult(stages, interrupted=True)
            if result.exit_code != 0:
                return PlanResult(
                    stages,
                    result.exit_code,
                    result.timed_out,
                    result.failure,
                    result.failures,
                    result.overflow,
                    result.consequences,
                )
        return PlanResult(stages)
    except KeyboardInterrupt:
        return PlanResult(stages, interrupted=True)
    finally:
        coordinator.close()

