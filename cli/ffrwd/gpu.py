"""Which GPU adapter a sidecar opens its device on.

The sidecar opens one device per process, and the modules granted ``-gpu``,
its decoder, its encoders and its filters all work on it. ``--gpu-adapter``
on ``run`` and ``compile``, or `FFRWD_GPU_ADAPTER` where the flag is absent,
names the adapter: a zero-based index into the list ``ffrwd-wasm
--gpu-info`` prints, or a case-insensitive part of its name. Neither set,
the sidecar opens the adapter it always has and nothing is spelled.

:func:`spawn_args` is the ``-gpu-adapter`` a sidecar holding frames on the
GPU is given, :func:`info` what the sidecar lists, and :func:`check` the
refusal a run gets before anything is spawned when the spec names no one
adapter.
"""

from __future__ import annotations

import functools
import json
import os
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from . import binaries
from .errors import ErrorCode, FfrwdError
from .table import CellValue, TableResult, render_table

__all__ = [
    "ADAPTER_FLAG",
    "ADAPTER_VAR",
    "SETUP_COMMAND",
    "Adapter",
    "Info",
    "check",
    "chosen",
    "describe",
    "info",
    "spawn_args",
    "spec",
]

# The environment variable naming the adapter where the command line does not.
ADAPTER_VAR = "FFRWD_GPU_ADAPTER"

# The sidecar's flag naming it.
ADAPTER_FLAG = "-gpu-adapter"

SETUP_COMMAND = "ffrwd setup gpu"

# What the sidecar's own refusal ends with; the CLI's hint names the CLI's
# command instead.
_SIDECAR_HINT = "; `ffrwd-wasm --gpu-info` lists them"

# How long the sidecar gets to answer --gpu-info: it opens no device, but
# asks every backend for its adapters.
_INFO_TIMEOUT = 30.0

# ``--gpu-adapter``'s value while a command runs, None outside one or where
# it was not given.
_flag: str | None = None


@contextmanager
def chosen(flag: str | None) -> Iterator[None]:
    """``--gpu-adapter``'s value for what runs inside, over the environment's.

    The variable is set to it too, so a process this one starts, a sidecar
    or a node's runner, reads the same. None leaves the environment in
    charge.
    """
    global _flag
    if flag is None:
        yield
        return
    before, named = _flag, os.environ.get(ADAPTER_VAR)
    _flag = flag
    os.environ[ADAPTER_VAR] = flag
    try:
        yield
    finally:
        _flag = before
        if named is None:
            os.environ.pop(ADAPTER_VAR, None)
        else:
            os.environ[ADAPTER_VAR] = named


def spec() -> str | None:
    """The adapter this run names: the flag's, else `FFRWD_GPU_ADAPTER`'s,
    where either is set and not blank; else None."""
    given = _flag if _flag is not None else os.environ.get(ADAPTER_VAR, "")
    return given if given.strip() else None


def _label(given: str) -> str:
    """How the CLI names the spec in a refusal."""
    if _flag is not None:
        return f"--gpu-adapter {given}"
    return f"{ADAPTER_VAR}={given}"


def spawn_args() -> list[str]:
    """The ``-gpu-adapter`` a sidecar holding frames on the GPU is given:
    empty where nothing names an adapter, so the command is what it always
    was."""
    given = spec()
    return [ADAPTER_FLAG, given] if given is not None else []


def _reject(
    message: str, hint: str, *, code: ErrorCode = ErrorCode.UNSUPPORTED_SQL
) -> FfrwdError:
    return FfrwdError(code, message, hint=hint)


@dataclass(frozen=True)
class Adapter:
    """One adapter ``ffrwd-wasm --gpu-info`` lists."""

    index: int
    name: str
    # wgpu's name for the backend: dx12, vulkan, metal, gl.
    backend: str
    vendor: int
    device: int
    # discrete, integrated, virtual, software or other.
    kind: str
    # Whether the sidecar opens it when nothing names an adapter.
    default: bool


@dataclass(frozen=True)
class Info:
    """What ``ffrwd-wasm --gpu-info`` answers, and what it made of a spec.

    `chosen` is the index the spec names, and `refused` the sidecar's
    reason where it names no one adapter; both None where no spec was
    asked about.
    """

    adapters: tuple[Adapter, ...]
    chosen: int | None = None
    refused: str | None = None


@functools.cache
def info(given: str | None = None) -> Info:
    """Ask the sidecar for its adapters, and which one `given` names, once
    per process for each spec.

    Raises ``FfrwdError`` -- and nothing else -- when the sidecar is not
    installed, will not run, or answers with something that is not an
    adapter list.
    """
    sidecar = binaries.ffrwd_wasm_path()
    if sidecar is None:
        raise _reject(
            "the ffrwd-wasm sidecar is not installed, and it is what lists the GPU adapters",
            hint=binaries.INSTALL_HINT,
        )
    argv = [sidecar, "--gpu-info"]
    if given is not None:
        argv += [ADAPTER_FLAG, given]
    # The spec asked about is the one on the argv, never the variable.
    env = {k: v for k, v in os.environ.items() if k != ADAPTER_VAR}
    try:
        done = subprocess.run(
            argv,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=_INFO_TIMEOUT,
            check=False,
            env=env,
        )
    except OSError as err:
        raise _reject(
            f"could not run the ffrwd-wasm sidecar at {sidecar}: {err.strerror or err}",
            hint=binaries.INSTALL_HINT,
        ) from err
    except subprocess.TimeoutExpired as err:
        raise _reject(
            f"the ffrwd-wasm sidecar did not answer --gpu-info within {_INFO_TIMEOUT:.0f}s",
            hint=binaries.INSTALL_HINT,
        ) from err
    if done.returncode != 0:
        raise _reject(
            f"the ffrwd-wasm sidecar at {sidecar} does not answer --gpu-info",
            hint="it is older than this ffrwd; install the two together",
        )
    try:
        payload = json.loads(done.stdout)
    except ValueError as err:
        raise _reject(
            "the ffrwd-wasm sidecar's answer to --gpu-info is not JSON",
            hint="the sidecar found may be a different version than this ffrwd",
        ) from err
    return _info_document(payload)


def _info_document(payload: object) -> Info:
    """`payload` as an :class:`Info`, refused where it is not one."""
    wrong = _reject(
        "the ffrwd-wasm sidecar's answer to --gpu-info is not an adapter list",
        hint="the sidecar found may be a different version than this ffrwd",
    )
    if not isinstance(payload, dict) or not isinstance(payload.get("adapters"), list):
        raise wrong
    adapters = []
    for one in payload["adapters"]:
        if not isinstance(one, dict):
            raise wrong
        try:
            adapters.append(
                Adapter(
                    index=int(one["index"]),
                    name=str(one["name"]),
                    backend=str(one["backend"]),
                    vendor=int(one["vendor"]),
                    device=int(one["device"]),
                    kind=str(one["type"]),
                    default=bool(one["default"]),
                )
            )
        except (KeyError, TypeError, ValueError) as err:
            raise wrong from err
    picked = payload.get("chosen")
    refused = payload.get("refused")
    return Info(
        adapters=tuple(adapters),
        chosen=picked if isinstance(picked, int) else None,
        refused=refused if isinstance(refused, str) else None,
    )


def check() -> None:
    """Refuse a spec that names no one adapter, before a run spawns a sidecar
    that would refuse it itself. Nothing to do where no spec is set.

    Raises ``FfrwdError`` with the sidecar's reason, the adapters there are,
    and the setup command that lists them.
    """
    given = spec()
    if given is None:
        return
    found = info(given)
    if found.refused is None:
        return
    reason = found.refused
    prefix = f"{ADAPTER_FLAG} {given}: "
    if reason.startswith(prefix):
        reason = f"{_label(given)}: {reason[len(prefix):]}"
    reason = reason.removesuffix(_SIDECAR_HINT)
    raise _reject(
        reason,
        hint=f"`{SETUP_COMMAND}` lists the adapters this machine has",
        code=ErrorCode.RUNTIME_NOT_FOUND,
    )


def _hex(value: int) -> str:
    return f"0x{value:04x}"


def describe(found: Info) -> str:
    """The table ``ffrwd setup gpu`` prints: every adapter, with its index."""
    rows: list[list[CellValue]] = [
        [
            one.index,
            one.name,
            one.backend,
            one.kind,
            _hex(one.vendor),
            _hex(one.device),
            "yes" if one.default else "",
        ]
        for one in found.adapters
    ]
    return render_table(
        TableResult(
            columns=["index", "name", "backend", "type", "vendor", "device", "default"],
            rows=rows,
        )
    )
