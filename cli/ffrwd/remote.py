"""``ffrwd run --remote`` and ``ffrwd jobs``: thin clients of the hosted job API.

A remote run is the same query, executed by the registry's runner instead of
this machine's ffmpeg. :func:`submit_run` builds the submit payload -- the
substituted SQL plus the raw ``-v`` pairs it was substituted from, the recipe
name and owning package when one was run by name, the effective lock (the
project's entries over the machine-wide lockfile's, one document), every
``input()`` path (local files hashed and uploaded; any "://" spec passed
through untouched, the runner's to open), every ``COPY ... TO`` destination,
and ``--timeout`` -- posts it, uploads the file inputs and the packed linked
packages, and starts the job.
:func:`jobs_command` is the other half: list, watch, cancel, fetch a
succeeded job's outputs, or show one job's own detail -- the fields a
listing row omits, its log tail included. A failed or cancelled job's tail
also prints on its own, right under the error: from ``--wait`` when the job
it is following lands there, and from ``--watch`` once its loop stops.

Every refusal this module makes is decided BEFORE anything reaches the
network, and every one is a typed :class:`~ffrwd.errors.FfrwdError` with a
hint; the server's own refusals (``{error, hint}`` bodies with honest
statuses) pass through as themselves. All HTTP goes through
``ffrwd.packages``' seams, so the unit tier fakes the server the way
``test_publish`` does.

A LINKED package does ride along. It has no published digest, so it is packed
at submit time -- the same :func:`ffrwd.store.pack` ``publish`` uses, so what
travels is the manifest's closure and what the ignore files allow, models
excluded -- uploaded to the content-addressed endpoint the file inputs use, and
spelled in the submitted lock as a pin against that archive's digest. The
lockfile ON DISK is untouched: a link stays a link for local development, and
the rewrite exists only on the wire.

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
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from sqlglot import exp

from ffrwd import __version__, credentials, store
from ffrwd import packages as packages_module
from ffrwd.console import Announce, Console, written_size
from ffrwd.errors import ErrorCode, FfrwdError
from ffrwd.parser import copy_destinations, input_specs, parse
from ffrwd.probe import is_url
from ffrwd.project import (
    LOCKFILE_NAME,
    MANIFEST_NAME,
    LinkEntry,
    LockEntry,
    Lockfile,
    Package,
    PackageSet,
    RegistryEntry,
    entry_root,
    find_lockfile,
    lockfile_text,
    read_lockfile,
    read_manifest,
)
from ffrwd.table import CellValue, TableResult, render_table

__all__ = [
    "RunQuery",
    "Submitted",
    "free_footer",
    "jobs_command",
    "submit_run",
    "wait_for_run",
]

# The submit payload format this client writes. The endpoint refuses any
# other with a 426 naming the upgrade.
JOB_FORMAT_VERSION = 1

# What one API answer may weigh. Job listings and signed-URL lists are small
# JSON; output DOWNLOADS are unbounded and streamed, not read through this.
_MAX_RESPONSE_BYTES = 4 * 1024 * 1024

# What one packed linked package may weigh on the wire. A package is SQL plus
# at most a few wasm modules -- the largest real one packs to a few hundred
# kilobytes, and its models are excluded and fetched by the runner -- so this
# sits a hundredfold above anything legitimate. It is also under the store's
# own unpacked cap, so an archive that passes here is one the runner can still
# unpack.
MAX_PACKAGE_BYTES = 32 * 1024 * 1024

# How often --watch redraws.
WATCH_SECONDS = 3.0

_CHUNK_BYTES = 1 << 20

_ACTIVE_STATES = frozenset({"submitted", "queued", "running"})

# --wait's terminal states: whatever a job lands on once it stops polling.
_WAIT_TERMINAL_STATES = frozenset({"succeeded", "failed", "cancelled"})

# --wait's bar: a fixed character count, filled by the row's progress_pct.
_BAR_WIDTH = 24

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


def _remaining(data: dict[str, object]) -> dict[str, object] | None:
    """The optional ``remaining`` object a submit or a listing may carry.

    None on a server that predates the free allotment, or that sent
    something other than an object under the key -- either way there is
    nothing to render.
    """
    value = data.get("remaining")
    if not isinstance(value, dict):
        return None
    return {str(key): item for key, item in value.items()}


# --------------------------------------------------------------------------
# submit
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Submitted:
    """What a submit hands back: the job id, and the account's free balance.

    `remaining` is the server's own object, verbatim -- None on a server that
    predates the free allotment, in which case the caller prints nothing
    about it.
    """

    job_id: str
    remaining: dict[str, object] | None


def submit_run(
    query: RunQuery,
    packages: PackageSet | None,
    args: argparse.Namespace,
    *,
    announce: Announce | None = None,
) -> Submitted:
    """Submit `query` as a hosted job: post the spec, upload the inputs, start it.

    Called from ``run``'s fork after the query classified as a media one.
    Raises :class:`FfrwdError` for every refusal -- this machine's own all
    before the first request, the server's passed through -- and returns the
    job id the caller reports, alongside what the account has left this
    month. `announce` hears one line per step: the submit, each upload with
    its size, and the start.
    """
    if args.show or args.show_only:
        raise _reject(
            "--show opens a window on this machine, and a remote run has none",
            "drop --show/--show-only, or run without --remote",
        )
    token = _token()
    text = query.text
    _refuse_computed_destinations(text)
    inputs, uploads = _inputs(text)
    lock, archives = _lock(packages)

    spec: dict[str, object] = {
        "format_version": JOB_FORMAT_VERSION,
        "query": text,
        # The stored query has the -v pairs substituted in; they travel raw
        # as well, so the job's record holds what was asked, not only the
        # text it became.
        "variables": dict(query.variables),
        "recipe": query.recipe,
        "owner": list(query.owner) if query.owner is not None else None,
        "lock": lock,
        # The digests this submit uploads rather than the registry publishes:
        # what the lock pins for a package that was linked here.
        "packages": [archive.entry.sha256 for archive in archives],
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

    for archive in archives:
        if announce is not None:
            announce(
                f"uploading package {archive.entry.name} "
                f"({written_size(len(archive.content))})"
            )
        _upload_bytes(upload_url, job_token, archive.content, archive.entry.sha256)
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
    return Submitted(job_id=job_id, remaining=_remaining(answer))


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


@dataclass(frozen=True)
class _Archive:
    """One linked package packed for the wire: the pin it becomes, and the bytes."""

    entry: RegistryEntry
    content: bytes


def _lock(packages: PackageSet | None) -> tuple[str | None, tuple[_Archive, ...]]:
    """The submitted lock, and the packed archives its pins name.

    The effective lock is the project's entries over the machine-wide
    lockfile's: local resolution layers the global lockfile under the
    project's, so a query compiling here against globally installed packages
    must submit a lock the runner can honor too. One document is synthesized
    -- the project lockfile's entries, then every global entry whose name the
    project does not pin -- in the same format the runner reads.

    Every link in that document is then packed and replaced by a pin against
    its archive's digest, transitively: a linked package's own lockfile may
    link others, and those travel too. The document on the wire therefore
    holds registry pins only. Nothing is written to either lockfile on disk.

    With nothing installed globally and nothing linked, the project's file
    travels verbatim, as it always did; with no lockfile anywhere there is
    nothing to send.
    """
    start = packages.root if packages is not None else Path.cwd()
    found = find_lockfile(start)
    machine_wide = _global_lock(found)
    local = read_lockfile(found) if found is not None else None
    if local is None and machine_wide is None:
        return None, ()

    # Each entry with the lockfile it came from: a link's path is written
    # relative to the file holding it, and the merge below loses that.
    sourced: list[tuple[LockEntry, Path]] = []
    if local is not None:
        sourced = [(entry, local.path) for entry in local.entries]
    pinned = {
        entry.name for entry, _ in sourced if isinstance(entry, RegistryEntry)
    }
    if machine_wide is not None:
        for entry in machine_wide.entries:
            if isinstance(entry, RegistryEntry) and entry.name in pinned:
                continue
            sourced.append((entry, machine_wide.path))

    linked = any(isinstance(entry, LinkEntry) for entry, _ in sourced)
    if machine_wide is None and not linked and found is not None:
        try:
            return found.read_text(encoding="utf-8"), ()
        except OSError as err:
            raise _reject(
                f"{found} could not be read: {err.strerror or err}",
                "the lockfile rides along verbatim so the runner installs what "
                "this project pins",
            ) from err

    packed: dict[Path, RegistryEntry] = {}
    archives: list[_Archive] = []
    entries = _resolved(sourced, [], packed, archives)
    held = local if local is not None else machine_wide
    assert held is not None  # both None returned above
    return lockfile_text(entries, dependencies=held.dependencies), tuple(archives)


def _resolved(
    sourced: Sequence[tuple[LockEntry, Path]],
    chain: list[tuple[Path, str]],
    packed: dict[Path, RegistryEntry],
    archives: list[_Archive],
) -> list[LockEntry]:
    """`sourced`'s entries with every link replaced by the pin its archive earns.

    Depth first, so a linked package's own links are packed before the pin
    that names it. `packed` is what has already been packed, keyed by the
    directory -- one link, one archive, however many lockfiles name it.
    `chain` is the links currently being walked, so a loop is caught rather
    than followed.
    """
    entries: list[LockEntry] = []
    for entry, lock_path in sourced:
        if not isinstance(entry, LinkEntry):
            entries.append(entry)
            continue
        root = _link_root(entry, lock_path)
        held = packed.get(root)
        if held is None:
            held = _pack_link(entry, root, chain, packed, archives)
        if not any(
            isinstance(one, RegistryEntry)
            and one.name == held.name
            and one.version == held.version
            for one in entries
        ):
            entries.append(held)
    return entries


def _link_root(entry: LinkEntry, lock_path: Path) -> Path:
    """The directory `entry` links, or a refusal naming the link."""
    try:
        return entry_root(entry, lock_path)
    except FfrwdError as err:
        raise _reject(
            f"the link to {entry.path} could not be read: {err.message}",
            "a linked package travels as bytes, so its directory has to be "
            "readable; unlink it, or run without --remote",
        ) from err


def _pack_link(
    entry: LinkEntry,
    root: Path,
    chain: list[tuple[Path, str]],
    packed: dict[Path, RegistryEntry],
    archives: list[_Archive],
) -> RegistryEntry:
    """Pack the package at `root` -- and everything it links -- into `archives`.

    Returns the pin the submitted lock records for it: the manifest's own name
    and version, the archive's digest, and the store path that digest lives at
    on the runner.
    """
    package = _link_manifest(entry, root)
    for at, (walked, _named) in enumerate(chain):
        if walked == root:
            loop = " -> ".join([*(name for _, name in chain[at:]), package.name])
            raise _reject(
                f"link cycle: {loop}",
                "there is no resolver here to break it; one of these packages has "
                "to stop linking another in the loop",
            )
    chain.append((root, package.name))
    nested = root / LOCKFILE_NAME
    if nested.is_file():
        held = read_lockfile(nested)
        _resolved([(one, nested) for one in held.entries], chain, packed, archives)
    chain.pop()

    content = _pack(package.name, root)
    if len(content) > MAX_PACKAGE_BYTES:
        raise _reject(
            f"package '{package.name}' packs to {written_size(len(content))}, and a "
            f"submit carries at most {written_size(MAX_PACKAGE_BYTES)} per package",
            f"exclude what the run does not need in {store.IGNORE_NAME}; a build "
            "directory or test media in the package tree is the usual cause",
        )
    digest = hashlib.sha256(content).hexdigest()
    pin = RegistryEntry(
        name=package.name,
        version=package.version,
        sha256=digest,
        store=store.entry_path(digest),
    )
    packed[root] = pin
    archives.append(_Archive(entry=pin, content=content))
    return pin


def _link_manifest(entry: LinkEntry, root: Path) -> Package:
    try:
        return read_manifest(root / MANIFEST_NAME)
    except FfrwdError as err:
        raise _reject(
            f"the link to {entry.path} could not be read: {err.message}",
            "a linked package travels as bytes, so its manifest has to be "
            "readable; unlink it, or run without --remote",
        ) from err


def _pack(name: str, root: Path) -> bytes:
    try:
        return store.pack(root)
    except FfrwdError as err:
        raise _reject(
            f"package '{name}' could not be packed: {err.message}",
            err.hint or "fix the package directory, or run without --remote",
        ) from err
    except OSError as err:
        raise _reject(
            f"package '{name}' at {root} could not be read: {err.strerror or err}",
            "a linked package travels as bytes, so its directory has to be readable",
        ) from err


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
    _upload_bytes(upload_url, job_token, content, digest)


def _upload_bytes(upload_url: str, job_token: str, content: bytes, digest: str) -> None:
    """POST `content` to the content-addressed endpoint under its digest.

    Shared by the file inputs and the packed linked packages: both are bytes
    the runner reads back by digest, and the endpoint answers ``already: true``
    for content it already holds.
    """
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
    """``ffrwd jobs``: list (the default), an ID, --json, --watch, --cancel, --fetch.

    `announce` hears one line per output ``--fetch`` downloads; the other
    modes print their answer and narrate nothing.
    """
    token = _token()
    if args.cancel is not None:
        return _cancel(token, str(args.cancel))
    if args.fetch is not None:
        _fetch(token, str(args.fetch), overwrite=bool(args.overwrite), announce=announce)
        return 0
    written_id = getattr(args, "id", None)
    if written_id is not None:
        return _show_job(token, str(written_id), as_json=bool(args.as_json))
    if args.watch:
        return _watch(token)
    rows, remaining = _listing(token)
    if args.as_json:
        print(json.dumps({"jobs": rows, "remaining": remaining}, indent=2))
        return 0
    _print_listing(rows, _utcnow(), remaining)
    return 0


def _listing(token: str) -> tuple[list[dict[str, object]], dict[str, object] | None]:
    where = _jobs_url()
    data = _json_object(_call(where, headers=_bearer(token)), where)
    rows = data.get("jobs")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise _malformed(where)
    return [{str(key): value for key, value in row.items()} for row in rows], _remaining(data)


def _rows(token: str) -> list[dict[str, object]]:
    """The listing's rows alone, for the callers that never render a footer."""
    rows, _ = _listing(token)
    return rows


def _job_detail(token: str, job_id: str) -> dict[str, object]:
    """GET /jobs/<uuid>: the one row this caller owns, ``log_tail`` and
    ``outputs`` included -- the fields the listing route omits."""
    where = f"{_jobs_url()}/{job_id}"
    return _json_object(_call(where, headers=_bearer(token)), where)


def _try_job_detail(token: str, job_id: str) -> dict[str, object] | None:
    """`_job_detail`, best-effort: None on any failure, never raises.

    Used where the detail only feeds a log tail printed alongside a row
    error that already came from the listing -- a secondary failure here
    (a network hiccup, a token that expired between polls) must never mask
    that error, so this swallows it and shows nothing more.
    """
    try:
        return _job_detail(token, job_id)
    except FfrwdError:
        return None


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


def free_footer(remaining: dict[str, object]) -> str:
    """The free allotment left this month, over one `remaining` object.

    Sibling of `_month_footer`: minutes are whole, floored from the seconds
    the server holds authoritative, never below zero. Shared with `submit_run`'s
    caller, which prints the same line right after a submit.
    """
    cpu = max(0, int(_seconds(remaining, "cpu_seconds") // 60))
    gpu = max(0, int(_seconds(remaining, "gpu_seconds") // 60))
    return f"free this month: {cpu}m CPU + {gpu}m GPU remaining"


def _print_listing(
    rows: list[dict[str, object]], now: datetime, remaining: dict[str, object] | None
) -> None:
    print(render_table(_listing_table(rows, now)))
    print(_month_footer(rows, now))
    if remaining is not None:
        print(free_footer(remaining))


# How many of the log tail's own lines print. The server already bounds what
# `log_tail` stores; this bounds what a terminal shows of it.
_LOG_TAIL_LINES = 20


def _tail_block(detail: dict[str, object]) -> str | None:
    """The bounded ``log tail:`` block for one job's detail, or None when
    the job carries nothing stored to show (a null or empty ``log_tail``)."""
    tail = detail.get("log_tail")
    if not isinstance(tail, str):
        return None
    lines = tail.splitlines()
    if not lines:
        return None
    shown = lines[-_LOG_TAIL_LINES:]
    cut = len(lines) - len(shown)
    header = (
        "log tail:"
        if cut == 0
        else f"log tail (last {_LOG_TAIL_LINES} of {len(lines)} lines, {cut} cut):"
    )
    return "\n".join([header, *(f"  {line}" for line in shown)])


def _show_job(token: str, written: str, *, as_json: bool) -> int:
    """``ffrwd jobs <id>``: one job's detail, the log tail included."""
    job_id = _resolve_id(token, written)
    detail = _job_detail(token, job_id)
    if as_json:
        print(json.dumps(detail, indent=2))
        return 0
    _print_job_detail(detail, _utcnow())
    return 0


def _print_job_detail(detail: dict[str, object], now: datetime) -> None:
    """The listing's own row rendering for one job, plus what a listing
    never carries: its error/hint pair, its outputs, and its log tail."""
    print(render_table(_listing_table([detail], now)))
    if _text(detail, "state") in ("failed", "cancelled"):
        error = _text(detail, "error")
        hint = _text(detail, "hint")
        if error:
            print(f"error: {error}")
        if hint:
            print(f"hint: {hint}")
    outputs = detail.get("outputs")
    if isinstance(outputs, list):
        for one in outputs:
            if isinstance(one, dict) and isinstance(one.get("path"), str):
                print(f"output: {one['path']}")
    block = _tail_block(detail)
    if block is not None:
        print(block)


def _print_row_failure(row: dict[str, object], detail: dict[str, object] | None) -> None:
    """A terminal failed/cancelled row's error, and its tail beneath it --
    shared by ``--wait`` and ``--watch`` so both read the same way."""
    print(f"error: {_row_error(row)}", file=sys.stderr)
    if detail is not None:
        block = _tail_block(detail)
        if block is not None:
            print(block, file=sys.stderr)


def _watch(token: str) -> int:
    """Redraw the listing until nothing is submitted, queued or running.

    The final redraw, the one that ends the loop, also prints any failed or
    cancelled row's error and log tail -- the same shape ``--wait`` prints
    for the one job it follows.
    """
    try:
        while True:
            rows, remaining = _listing(token)
            if sys.stdout.isatty():
                print("\x1b[2J\x1b[H", end="")
            _print_listing(rows, _utcnow(), remaining)
            if not any(_text(row, "state") in _ACTIVE_STATES for row in rows):
                for row in rows:
                    if _text(row, "state") in ("failed", "cancelled"):
                        job_id = _text(row, "id")
                        _print_row_failure(row, _try_job_detail(token, job_id))
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
    token: str,
    written: str,
    *,
    overwrite: bool,
    announce: Announce | None = None,
    quiet: bool = False,
) -> list[str]:
    """Download a job's outputs to their as-written paths. Returns what it wrote.

    Shared by ``jobs --fetch`` (which only cares that this returns at all --
    a raise is the failure) and ``--wait`` (which reports the paths itself,
    as JSON, when `quiet` drops the plain ``wrote <path>`` lines).
    """
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
        if not quiet:
            print(f"wrote {output.path}")
    return [output.path for output in outputs]


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


# --------------------------------------------------------------------------
# --wait
# --------------------------------------------------------------------------


def _percent(row: dict[str, object]) -> float:
    value = row.get("progress_pct")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return max(0.0, min(100.0, float(value)))


def _bar_line(row: dict[str, object]) -> str:
    """--wait's one redrawn line: a bar from `progress_pct`, then the state."""
    pct = _percent(row)
    filled = int(round(_BAR_WIDTH * pct / 100))
    bar = "#" * filled + "-" * (_BAR_WIDTH - filled)
    return f"[{bar}] {int(pct):3d}% {_text(row, 'state')}"


def _poll_to_terminal(
    token: str, job_id: str, *, draw: Callable[[str], None] | None
) -> tuple[dict[str, object], dict[str, object] | None]:
    """Redraw the bar off the same listing --watch reads, until `job_id` lands.

    `draw` is None under --json, which draws nothing. Returns the terminal
    row alongside that poll's `remaining`, for the budget note.
    """
    while True:
        rows, remaining = _listing(token)
        row = next((one for one in rows if _text(one, "id") == job_id), None)
        if row is not None:
            if draw is not None:
                draw(_bar_line(row))
            if _text(row, "state") in _WAIT_TERMINAL_STATES:
                return row, remaining
        _sleep(WATCH_SECONDS)


def _row_error(row: dict[str, object]) -> FfrwdError:
    """A failed or cancelled row's own report, rendered the way every other
    rejection from this module is -- str()'d for `_print_row_failure`, never
    raised: --wait and --watch are its only callers, and both print their own
    output rather than unwind to the CLI's generic error handler."""
    message = _text(row, "error") or f"the job {_text(row, 'state')}"
    hint = _text(row, "hint") or "run `ffrwd jobs` to see the full listing"
    return _reject(message, hint)


def _budget_note(remaining: dict[str, object] | None) -> str:
    """The one line after a budget-stopped fetch: the stop, and the reset.

    A budget stop is a SUCCESS whose outputs are whole up to where it
    stopped -- the row's state is succeeded and `budget_exhausted` marks it.
    """
    reset = _text(remaining, "resets_on") if remaining is not None else ""
    if reset:
        return f"the free allotment stopped this job early; it resets on {reset}"
    return "the free allotment stopped this job early"


def _detach_message(job_id: str) -> str:
    short = job_id[:8]
    return (
        f"detached: {job_id} keeps running -- "
        f"`ffrwd jobs --watch {short}` to follow it, "
        f"`ffrwd jobs --fetch {short}` to take its outputs later"
    )


def wait_for_run(job_id: str, args: argparse.Namespace, *, console: Console) -> int:
    """``run --remote --wait``: poll `job_id` to a terminal state, then fetch.

    Shares `_listing` (--watch's own endpoint and interval) for the poll, and
    `_fetch` (--fetch's own machinery) for the download. A Ctrl-C during the
    wait never cancels the job -- it prints where to pick it back up and
    exits 130. `args.as_json` drops the bar and emits the terminal row (plus
    any written paths) as one JSON object instead of the plain narration. A
    failed or cancelled job also fetches its own detail here, to print (or,
    under --json, to carry) its log tail -- the thing the row's own error
    tells you to look at.
    """
    token = _token()
    draw = None if args.as_json else console.transient
    try:
        row, remaining = _poll_to_terminal(token, job_id, draw=draw)
    except KeyboardInterrupt:
        console.end_transient()
        print(_detach_message(job_id), file=sys.stderr)
        return 130
    console.end_transient()
    state = _text(row, "state")

    if state in ("failed", "cancelled"):
        detail = _try_job_detail(token, job_id)
        if args.as_json:
            payload = dict(row)
            payload["log_tail"] = detail.get("log_tail") if detail is not None else None
            print(json.dumps({"job": payload}, indent=2))
            return 1
        _print_row_failure(row, detail)
        return 1

    # succeeded, budget-stopped or not: fetched exactly as `jobs --fetch` does.
    announce = None if args.as_json else console.say
    paths = _fetch(
        token,
        job_id,
        overwrite=bool(args.overwrite),
        announce=announce,
        quiet=bool(args.as_json),
    )
    if args.as_json:
        print(json.dumps({"job": row, "written": paths}, indent=2))
        return 0
    if row.get("budget_exhausted"):
        print(_budget_note(remaining))
    return 0
