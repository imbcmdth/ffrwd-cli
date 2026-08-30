"""``ffrwd run --remote`` and ``ffrwd jobs``: thin clients of the hosted job API.

A remote run is the same query, executed by the registry's runner instead of
this machine's ffmpeg. :func:`submit_run` builds the submit payload -- the
substituted SQL plus the raw ``-v`` pairs it was substituted from, the recipe
name and owning package when one was run by name, the effective lock (the
project's entries over the machine-wide lockfile's, one document), every
``input()`` path (local files hashed and uploaded; any "://" spec passed
through untouched, the runner's to open), every ``COPY ... TO`` destination,
and ``--timeout`` -- posts it, uploads the file inputs, and starts the job.
:func:`jobs_command` is the other half: list, watch, cancel, and fetch a
succeeded job's outputs.

Every refusal this module makes is decided BEFORE anything reaches the
network, and every one is a typed :class:`~ffrwd.errors.FfrwdError` with a
hint; the server's own refusals (``{error, hint}`` bodies with honest
statuses) pass through as themselves. All HTTP goes through
``ffrwd.packages``' seams, so the unit tier fakes the server the way
``test_publish`` does.

What does NOT ride along: the project's own source files. A query resolving a
call through the project's OWN manifest package compiles here and fails
remotely with the compiler's ordinary unknown-function error -- published
dependencies are the runner's to install from the lockfile, and installed
packages travel by name, not by upload.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from sqlglot import exp

from ffrwd import __version__, credentials, store
from ffrwd import packages as packages_module
from ffrwd.console import Announce, written_size
from ffrwd.errors import ErrorCode, FfrwdError
from ffrwd.parser import copy_destinations, input_specs, parse
from ffrwd.probe import is_url
from ffrwd.project import (
    LockEntry,
    Lockfile,
    Package,
    PackageSet,
    RegistryEntry,
    find_lockfile,
    lockfile_text,
    read_lockfile,
)
from ffrwd.table import CellValue, TableResult, render_table

__all__ = ["RunQuery", "jobs_command", "submit_run"]

# The submit payload format this client writes. The endpoint refuses any
# other with a 426 naming the upgrade.
JOB_FORMAT_VERSION = 1

# What one API answer may weigh. Job listings and signed-URL lists are small
# JSON; output DOWNLOADS are unbounded and streamed, not read through this.
_MAX_RESPONSE_BYTES = 4 * 1024 * 1024

# How often --watch redraws.
WATCH_SECONDS = 3.0

_CHUNK_BYTES = 1 << 20

_ACTIVE_STATES = frozenset({"submitted", "queued", "running"})

_TOKEN_HINT = (
    "mint a token with the run scope and save it with `ffrwd login --token <token>`"
)

# A seam for tests: --watch's pacing, without the waiting.
_sleep = time.sleep


def _utcnow() -> datetime:
    """The clock the listing reads. A test pins it."""
    return datetime.now(timezone.utc)


class RunQuery(Protocol):
    """What `submit_run` reads off the resolved query (the CLI's own record)."""

    @property
    def text(self) -> str: ...

    @property
    def recipe(self) -> str | None: ...

    @property
    def owner(self) -> tuple[str, str] | None: ...

    @property
    def variables(self) -> Mapping[str, str]: ...


def _reject(
    message: str, hint: str, *, line: int | None = None, col: int | None = None
) -> FfrwdError:
    return FfrwdError(ErrorCode.UNSUPPORTED_SQL, message, line=line, col=col, hint=hint)


def _jobs_url() -> str:
    return f"{packages_module.api_url().rstrip('/')}/functions/v1/jobs"


def _token() -> str:
    token = credentials.load()
    if token is None:
        raise _reject(
            "running remotely needs an ffrwd token, and this machine holds none",
            _TOKEN_HINT,
        )
    return token


def _bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


# --------------------------------------------------------------------------
# talking to the API
# --------------------------------------------------------------------------


def _server_refusal(status: int, body: bytes) -> FfrwdError:
    """The API's own refusal, rendered the way every other rejection reads."""
    message = ""
    hint = ""
    try:
        data = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        data = None
    if isinstance(data, dict):
        said, hinted = data.get("error"), data.get("hint")
        message = said if isinstance(said, str) else ""
        hint = hinted if isinstance(hinted, str) else ""
    if not message:
        message = f"the job service refused the request with HTTP {status}"
    if not hint:
        hint = "the job service said no more than that; try again, or report it"
    return _reject(message, hint)


# Data-plane calls wait out a cold runner: the first request to a Modal
# endpoint holds until its container is up, and an upload's answer follows
# the whole body.
_DATA_PLANE_TIMEOUT = 600.0


def _unreachable(url: str, err: Exception) -> FfrwdError:
    reason = getattr(err, "reason", None) or getattr(err, "__cause__", None) or err
    return _reject(
        f"the job service could not be reached at {url}: {reason}",
        "check the network connection, or try again -- a cold runner can take a minute",
    )


def _call(
    url: str,
    *,
    headers: dict[str, str],
    data: bytes | None = None,
    limit: int = _MAX_RESPONSE_BYTES,
    timeout: float = packages_module.TIMEOUT,
) -> bytes:
    """One request: POST when `data` is given, GET otherwise. The body, or a refusal.

    POSTs ride ``packages.exchange``; a GET is the same seam one layer down
    (``packages._urlopen``), which exchange does not offer. Either way a
    status >= 400 becomes the server's own {error, hint}.
    """
    if data is not None:
        try:
            status, body = packages_module.exchange(
                url, headers=headers, data=data, limit=limit, timeout=timeout
            )
        except FfrwdError as err:
            raise _unreachable(url, err) from err
        if status >= 400:
            raise _server_refusal(status, body)
        return body
    request = packages_module._request(url, headers=headers)
    try:
        with packages_module._urlopen(request, timeout=timeout) as response:
            body = bytes(response.read(limit + 1))
    except urllib.error.HTTPError as err:
        try:
            said = bytes(err.read(limit + 1))
        except OSError:
            said = b""
        raise _server_refusal(err.code, said) from err
    except (OSError, ValueError, urllib.error.URLError) as err:
        raise _unreachable(url, err) from err
    if len(body) > limit:
        raise _reject(
            f"the job service served more than {limit} bytes at {url}",
            "that is not an answer this client reads; check the API base URL",
        )
    return body


def _malformed(where: str) -> FfrwdError:
    return _reject(
        f"{where} answered with something this client does not read",
        "the job API may be newer than this ffrwd; upgrade it: "
        "`uv tool upgrade ffrwd` or `pip install -U ffrwd`",
    )


def _json_object(body: bytes, where: str) -> dict[str, object]:
    try:
        data = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError, RecursionError) as err:
        raise _malformed(where) from err
    if not isinstance(data, dict):
        raise _malformed(where)
    return {str(key): value for key, value in data.items()}


# --------------------------------------------------------------------------
# submit
# --------------------------------------------------------------------------


def submit_run(
    query: RunQuery,
    packages: PackageSet | None,
    args: argparse.Namespace,
    *,
    announce: Announce | None = None,
) -> str:
    """Submit `query` as a hosted job: post the spec, upload the inputs, start it.

    Called from ``run``'s fork after the query classified as a media one.
    Raises :class:`FfrwdError` for every refusal -- this machine's own all
    before the first request, the server's passed through -- and returns the
    job id the caller reports. `announce` hears one line per step: the
    submit, each upload with its size, and the start.
    """
    if args.show or args.show_only:
        raise _reject(
            "--show opens a window on this machine, and a remote run has none",
            "drop --show/--show-only, or run without --remote",
        )
    token = _token()
    _refuse_linked(packages)
    text = query.text
    _refuse_computed_destinations(text)
    inputs, uploads = _inputs(text)

    spec: dict[str, object] = {
        "format_version": JOB_FORMAT_VERSION,
        "query": text,
        # The stored query has the -v pairs substituted in; they travel raw
        # as well, so the job's record holds what was asked, not only the
        # text it became.
        "variables": dict(query.variables),
        "recipe": query.recipe,
        "owner": list(query.owner) if query.owner is not None else None,
        "lock": _lock_text(packages),
        "inputs": inputs,
        "outputs": copy_destinations(text),
        "timeout_s": args.timeout,
        "client_version": __version__,
    }
    if announce is not None:
        announce("submitting the job")
    where = _jobs_url()
    body = _call(where, headers=_bearer(token), data=json.dumps(spec).encode("utf-8"))
    answer = _json_object(body, where)
    job_id = answer.get("job_id")
    job_token = answer.get("job_token")
    upload_url = answer.get("upload_url")
    start_url = answer.get("start_url")
    if not (
        isinstance(job_id, str)
        and isinstance(job_token, str)
        and isinstance(upload_url, str)
        and isinstance(start_url, str)
    ):
        raise _malformed(where)

    for path, digest in uploads:
        _upload(upload_url, job_token, path, digest, announce)
    if announce is not None:
        announce("starting the job")
    _call(
        start_url,
        headers={"x-job-token": job_token},
        data=b"{}",
        timeout=_DATA_PLANE_TIMEOUT,
    )
    return job_id


def _refuse_linked(packages: PackageSet | None) -> None:
    """A linked directory has no digest, so no lockfile reproduces it remotely."""
    for package in _every_package(packages):
        if package.linked:
            raise _reject(
                f"package '{package.name}' is linked to {package.root}, and no "
                "digest reproduces a linked directory remotely",
                "install a published version of it, or run locally",
            )


def _every_package(packages: PackageSet | None) -> list[Package]:
    if packages is None:
        return []
    found = list(packages.packages.values())
    for versions in packages.versions.values():
        found.extend(versions.values())
    return found


def _refuse_computed_destinations(text: str) -> None:
    """A ``TO (<expression>)`` computes its paths, so a submit cannot declare them."""
    for node in parse(text).walk():
        if not isinstance(node, exp.Copy) or node.args.get("kind"):
            continue
        for target in node.args.get("files") or []:
            if isinstance(target, exp.Paren):
                raise _reject(
                    "a TO (<expression>) destination computes its paths per row, "
                    "so a submit cannot declare them",
                    "name the outputs with quoted paths, or run locally",
                )


def _inputs(text: str) -> tuple[list[dict[str, object]], list[tuple[str, str]]]:
    """The payload's inputs, and the (path, sha256) pairs to upload.

    One entry per distinct path, in first-written order -- two aliases over
    one file are one declaration and one upload. Any "://" spec passes
    through untouched: a live rtmp/udp/srt input is the runner's to open,
    never existence-checked here.
    """
    entries: list[dict[str, object]] = []
    uploads: list[tuple[str, str]] = []
    staged: set[str] = set()
    seen: set[str] = set()
    for spec in input_specs(text):
        if spec.path in seen:
            continue
        seen.add(spec.path)
        if is_url(spec.path):
            entries.append({"path": spec.path, "kind": "url"})
            continue
        if not os.path.isfile(spec.path):
            raise _reject(
                f"input '{spec.path}' does not exist",
                "a file input is uploaded from this machine, so the file has to be here",
                line=spec.line,
                col=spec.col,
            )
        digest, size = _digest(spec.path, line=spec.line, col=spec.col)
        entries.append(
            {"path": spec.path, "kind": "file", "sha256": digest, "bytes": size}
        )
        if digest not in staged:
            staged.add(digest)
            uploads.append((spec.path, digest))
    return entries, uploads


def _digest(path: str, *, line: int, col: int) -> tuple[str, int]:
    hasher = hashlib.sha256()
    size = 0
    try:
        with open(path, "rb") as handle:
            while True:
                chunk = handle.read(_CHUNK_BYTES)
                if not chunk:
                    break
                hasher.update(chunk)
                size += len(chunk)
    except OSError as err:
        raise _reject(
            f"input '{path}' could not be read: {err.strerror or err}",
            "a file input is uploaded from this machine, so it has to be readable",
            line=line,
            col=col,
        ) from err
    return hasher.hexdigest(), size


def _lock_text(packages: PackageSet | None) -> str | None:
    """The effective lock: the project's entries over the machine-wide lockfile's.

    Local resolution layers the global lockfile under the project's, so a
    query compiling here against globally installed packages must submit a
    lock the runner can honor too. One document is synthesized -- the project
    lockfile's entries, then every global entry whose name the project does
    not pin -- in the same format the runner reads. With nothing installed
    globally the project's file travels verbatim, as it always did; with no
    lockfile anywhere there is nothing to send.
    """
    start = packages.root if packages is not None else Path.cwd()
    found = find_lockfile(start)
    machine_wide = _global_lock(found)
    if machine_wide is None:
        if found is None:
            return None
        try:
            return found.read_text(encoding="utf-8")
        except OSError as err:
            raise _reject(
                f"{found} could not be read: {err.strerror or err}",
                "the lockfile rides along verbatim so the runner installs what "
                "this project pins",
            ) from err
    local = read_lockfile(found) if found is not None else None
    entries: list[LockEntry] = list(local.entries) if local is not None else []
    pinned = {entry.name for entry in entries if isinstance(entry, RegistryEntry)}
    for entry in machine_wide.entries:
        if isinstance(entry, RegistryEntry) and entry.name in pinned:
            continue
        entries.append(entry)
    held = local if local is not None else machine_wide
    return lockfile_text(entries, dependencies=held.dependencies)


def _global_lock(local: Path | None) -> Lockfile | None:
    """The machine-wide lockfile, or None when nothing was installed globally."""
    path = store.global_lock_path()
    try:
        if not path.is_file() or (local is not None and path == local):
            return None
    except (OSError, ValueError):
        return None
    return read_lockfile(path)


def _upload(
    upload_url: str,
    job_token: str,
    path: str,
    digest: str,
    announce: Announce | None = None,
) -> None:
    """POST one file input's bytes. An ``already: true`` answer is the dedupe
    hit -- the content is staged, nothing more to send for it."""
    try:
        content = Path(path).read_bytes()
    except OSError as err:
        raise _reject(
            f"input '{path}' could not be read: {err.strerror or err}",
            "a file input is uploaded from this machine, so it has to be readable",
        ) from err
    if announce is not None:
        announce(f"uploading {path} ({written_size(len(content))})")
    _call(
        f"{upload_url}?sha256={digest}",
        headers={"x-job-token": job_token, "Content-Type": "application/octet-stream"},
        data=content,
        timeout=_DATA_PLANE_TIMEOUT,
    )


# --------------------------------------------------------------------------
# jobs
# --------------------------------------------------------------------------


def jobs_command(args: argparse.Namespace, *, announce: Announce | None = None) -> int:
    """``ffrwd jobs``: list (the default), --json, --watch, --cancel, --fetch.

    `announce` hears one line per output ``--fetch`` downloads; the other
    modes print their answer and narrate nothing.
    """
    token = _token()
    if args.cancel is not None:
        return _cancel(token, str(args.cancel))
    if args.fetch is not None:
        return _fetch(
            token, str(args.fetch), overwrite=bool(args.overwrite), announce=announce
        )
    if args.watch:
        return _watch(token)
    rows = _rows(token)
    if args.as_json:
        print(json.dumps(rows, indent=2))
        return 0
    _print_listing(rows, _utcnow())
    return 0


def _rows(token: str) -> list[dict[str, object]]:
    where = _jobs_url()
    data = _json_object(_call(where, headers=_bearer(token)), where)
    rows = data.get("jobs")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise _malformed(where)
    return [{str(key): value for key, value in row.items()} for row in rows]


def _text(row: dict[str, object], key: str) -> str:
    value = row.get(key)
    return value if isinstance(value, str) else ""


def _seconds(row: dict[str, object], key: str) -> float:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return float(value)


def _duration(seconds: float) -> str:
    total = int(round(seconds))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _runtime(cpu: float, gpu: float) -> str:
    """The runtime cell: cpu seconds, plus the gpu share when there is one."""
    if cpu == 0 and gpu == 0:
        return ""
    if gpu:
        return f"{_duration(cpu)}+{_duration(gpu)} gpu"
    return _duration(cpu)


def _parse_when(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        when = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return when if when.tzinfo is not None else when.replace(tzinfo=timezone.utc)


def _ago(when: datetime | None, now: datetime) -> str:
    if when is None:
        return ""
    seconds = max(0, int((now - when).total_seconds()))
    if seconds < 60:
        return "just now"
    if seconds < 3600:
        return f"{seconds // 60}m ago"
    if seconds < 86400:
        return f"{seconds // 3600}h ago"
    return f"{seconds // 86400}d ago"


def _listing_table(rows: list[dict[str, object]], now: datetime) -> TableResult:
    shown: list[list[CellValue]] = []
    for row in rows:
        shown.append(
            [
                _text(row, "id")[:8],
                _text(row, "state"),
                _text(row, "recipe") or "sql",
                bool(row.get("gpu")),
                _runtime(_seconds(row, "duration_cpu_s"), _seconds(row, "duration_gpu_s")),
                _ago(_parse_when(row.get("created_at")), now),
            ]
        )
    return TableResult(
        columns=["id", "state", "recipe", "gpu", "runtime", "created"], rows=shown
    )


def _month_footer(rows: list[dict[str, object]], now: datetime) -> str:
    """This calendar month's totals (UTC), over the same rows the listing shows."""
    cpu = gpu = 0.0
    count = 0
    for row in rows:
        when = _parse_when(row.get("created_at"))
        if when is None:
            continue
        stamped = when.astimezone(timezone.utc)
        if (stamped.year, stamped.month) != (now.year, now.month):
            continue
        count += 1
        cpu += _seconds(row, "duration_cpu_s")
        gpu += _seconds(row, "duration_gpu_s")
    return (
        f"this month: {int(cpu // 60)} min cpu · {int(gpu // 60)} min gpu · {count} jobs"
    )


def _print_listing(rows: list[dict[str, object]], now: datetime) -> None:
    print(render_table(_listing_table(rows, now)))
    print(_month_footer(rows, now))


def _watch(token: str) -> int:
    """Redraw the listing until nothing is submitted, queued or running."""
    try:
        while True:
            rows = _rows(token)
            if sys.stdout.isatty():
                print("\x1b[2J\x1b[H", end="")
            _print_listing(rows, _utcnow())
            if not any(_text(row, "state") in _ACTIVE_STATES for row in rows):
                return 0
            _sleep(WATCH_SECONDS)
    except KeyboardInterrupt:
        return 0


def _resolve_id(token: str, written: str) -> str:
    """The one job id `written` names, resolved as a prefix of the caller's own."""
    wanted = written.lower()
    ids = [job_id for row in _rows(token) if (job_id := _text(row, "id"))]
    if wanted in ids:
        return wanted
    matches = [one for one in ids if one.startswith(wanted)]
    if not matches:
        raise _reject(
            f"'{written}' matches none of your jobs", "run `ffrwd jobs` to list them"
        )
    if len(matches) > 1:
        raise _reject(
            f"'{written}' matches more than one job: {', '.join(matches)}",
            "give more of the id",
        )
    return matches[0]


def _cancel(token: str, written: str) -> int:
    job_id = _resolve_id(token, written)
    where = f"{_jobs_url()}/{job_id}/cancel"
    row = _json_object(_call(where, headers=_bearer(token), data=b"{}"), where)
    state = _text(row, "state")
    if state == "cancelled":
        print(f"cancelled {job_id[:8]}")
    else:
        print(
            f"asked to cancel {job_id[:8]}; it is {state} and stops at the "
            "runner's next heartbeat"
        )
    return 0


@dataclass(frozen=True)
class _Output:
    """One fetchable output: where it goes, what it must hash to, where it is.

    `size` is the byte count the job recorded, None when the answer omits it
    -- it only feeds the narration line.
    """

    path: str
    sha256: str
    url: str
    size: int | None = None


def _fetch(
    token: str, written: str, *, overwrite: bool, announce: Announce | None = None
) -> int:
    job_id = _resolve_id(token, written)
    where = f"{_jobs_url()}/{job_id}/fetch"
    answer = _json_object(_call(where, headers=_bearer(token), data=b"{}"), where)
    outputs = _outputs(answer, where)
    # Refusing before the first byte: a partial fetch over one collision
    # helps nobody.
    for output in outputs:
        if Path(output.path).exists() and not overwrite:
            raise _reject(f"'{output.path}' already exists", "pass -y to overwrite it")
    for output in outputs:
        if announce is not None:
            written_bytes = (
                f" ({written_size(output.size)})" if output.size is not None else ""
            )
            announce(f"downloading {output.path}{written_bytes}")
        _download(output.url, Path(output.path), output.sha256)
        print(f"wrote {output.path}")
    return 0


def _outputs(answer: dict[str, object], where: str) -> list[_Output]:
    written = answer.get("outputs")
    if not isinstance(written, list):
        raise _malformed(where)
    outputs: list[_Output] = []
    for one in written:
        if not isinstance(one, dict):
            raise _malformed(where)
        path, sha256, url = one.get("path"), one.get("sha256"), one.get("url")
        if not (isinstance(path, str) and isinstance(sha256, str) and isinstance(url, str)):
            raise _malformed(where)
        count = one.get("bytes")
        size = count if isinstance(count, int) and not isinstance(count, bool) else None
        outputs.append(_Output(path=path, sha256=sha256, url=url, size=size))
    return outputs


def _download(url: str, path: Path, sha256: str) -> None:
    """Stream `url` to `path`, verifying the digest before the file lands.

    Written beside the destination and moved onto it, so an interrupted or
    corrupt download never leaves a half-file under the real name.
    """
    parent = path.parent
    try:
        if str(parent):
            parent.mkdir(parents=True, exist_ok=True)
    except OSError as err:
        raise _reject(
            f"'{path}' could not be written: {err.strerror or err}",
            "check that the destination is writable",
        ) from err
    part = path.with_name(f"{path.name}.part")
    hasher = hashlib.sha256()
    request = packages_module._request(url)
    try:
        with packages_module._urlopen(request, timeout=_DATA_PLANE_TIMEOUT) as response:
            with open(part, "wb") as handle:
                while True:
                    chunk = bytes(response.read(_CHUNK_BYTES))
                    if not chunk:
                        break
                    hasher.update(chunk)
                    handle.write(chunk)
    except urllib.error.HTTPError as err:
        part.unlink(missing_ok=True)
        try:
            said = bytes(err.read(_MAX_RESPONSE_BYTES))
        except OSError:
            said = b""
        raise _server_refusal(err.code, said) from err
    except (OSError, ValueError, urllib.error.URLError) as err:
        part.unlink(missing_ok=True)
        raise _reject(
            f"'{path}' could not be downloaded: {err}", "try the fetch again"
        ) from err
    if hasher.hexdigest() != sha256.lower():
        part.unlink(missing_ok=True)
        raise _reject(
            f"'{path}' downloaded as {hasher.hexdigest()}, not the {sha256} the "
            "job recorded",
            "try the fetch again",
        )
    os.replace(part, path)
