"""The PACKAGE registry client: finding a package, and getting it onto this machine.

Not :mod:`ffrwd.registry`, which is the FILTER registry -- what the local
ffmpeg binary supports. Two registries in one package would be a lasting
confusion, so the filter one keeps its name and this one is named for what it
serves: installable packages.

Two hosts answer, and they are two settings. :data:`DEFAULT_REGISTRY` is where
the detail documents are served from -- ``p/<namespace>/<package>.json``, one
per package, holding every published version and per version the archive's
sha256 and size. :data:`DEFAULT_API` is where the endpoints live that a static
file cannot be: the search function, the endpoint that signs an archive URL,
and the one that serves a private package's detail. ``FFRWD_REGISTRY`` and
``FFRWD_API`` override them.

There is no catalogue file and no index step. A package is resolved by
fetching its own detail document; a 404 means the registry has no such
package, and the suggestion in that rejection comes from the search function,
best effort -- a search that cannot be reached costs the suggestion and
nothing else.

``FFRWD_REGISTRY`` may also name a local DIRECTORY holding the same layout --
the detail documents under ``p/`` and the archives under ``archives/``. That
is the offline story and the one every check in the test suite uses: no
signing, no search function, the files read straight off the filesystem.

EVERY archive over HTTP is fetched through a signed URL. The client asks the
archive endpoint for one, sending its token when it has one, and then makes a
plain GET on the URL it was handed. A public version's archive signs without a
token; a private one needs it, and a refusal with no token on this machine
says to log in.

Nothing here writes to stdout or stderr, and nothing raises anything but
``FfrwdError``: a registry that is unreachable, unparseable, or serving an
archive that does not match its digest is a typed rejection naming the URL,
never a traceback.

Ordering matters at install: the archive's bytes are verified against the
digest the registry recorded before anything opens them (:func:`ffrwd.store.unpack`
does that), the models the manifest pins are fetched and verified next, and
the lockfile is written only after all of it is on disk. An install that
cannot verify a hash leaves both untouched.

Installing one package installs what it depends on too, recursively: after a
package is stored, its own manifest's ``dependencies`` are walked and each is
installed at its highest published version, exactly as a direct install
resolves one -- there is no resolver here, and this module never picks among
versions of a name. A dependency already pinned at the exact version wanted
is left alone, not refetched and not walked again; a different version of the
same name already pinned is not a conflict, since a package is content
addressed by its own (name, version) and two versions simply coexist -- what
changes is which one a given DEPENDENT resolves against, which is recorded on
its own lockfile entry (see :class:`~ffrwd.project.RegistryEntry`). A cycle
in that walk is the one thing this module refuses outright, naming the loop.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO

from . import credentials, nn, store, wasm
from .console import Announce, written_size
from .errors import ErrorCode, FfrwdError
from .functions import package_modules
from .project import (
    MANIFEST_NAME,
    LinkEntry,
    LockEntry,
    ModelPin,
    Package,
    RegistryEntry,
    add_dependency,
    held_links,
    is_package_name,
    link_target,
    links_path,
    name_refusal,
    read_linksfile,
    read_lockfile,
    read_manifest,
    stored_name,
    write_linksfile,
    write_lockfile,
)

__all__ = [
    "ANON_KEY",
    "API_ENV",
    "DEFAULT_API",
    "DEFAULT_REGISTRY",
    "REGISTRY_ENV",
    "Installed",
    "Listing",
    "Release",
    "api_url",
    "base_url",
    "exchange",
    "install",
    "is_version",
    "resolve",
    "search",
    "version_key",
]

# The hosted registry's project. The storage base serves the detail
# documents; the API base serves the search function, the archive-signing
# endpoint and the private detail endpoint; the anon key is public by
# design -- it identifies the project and authorizes nothing the row
# policies do not already allow.
_PROJECT_REF = "oanxpjiidhkijziqdhuv"
DEFAULT_REGISTRY = (
    f"https://{_PROJECT_REF}.supabase.co/storage/v1/object/public/packages"
)
DEFAULT_API = f"https://{_PROJECT_REF}.supabase.co"
ANON_KEY = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im9h"
    "bnhwamlpZGhraWp6aXFkaHV2Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODc5MTk0NzYsImV4"
    "cCI6MjEwMzQ5NTQ3Nn0.YOoUg2ES30OQfweGZOKilsNvf-ClwupcF-MfxFV1KYc"
)

REGISTRY_ENV = "FFRWD_REGISTRY"
API_ENV = "FFRWD_API"

# Every request gets one. A host that accepts a connection and then says
# nothing must not hang a command forever.
TIMEOUT = 30.0

# The shape of the documents this client reads. A registry publishing another
# version is refused rather than guessed at -- the alternative is installing
# content off a file whose keys mean something else.
FORMAT_VERSION = 2

# What one package's detail may weigh. It is JSON the client parses whole, and
# has no business being large.
_MAX_DETAIL_BYTES = 16 * 1024 * 1024
# The archive cap the store's own unpacked-size cap is the other half of.
_MAX_ARCHIVE_BYTES = 64 * 1024 * 1024
# What the search function may answer with.
_MAX_SEARCH_BYTES = 4 * 1024 * 1024
# A backstop on what one pinned file may weigh, not a judgment about model
# sizes: no honest export comes near it, and a stream that runs past it is a
# hostile answer rather than a large model.
_MAX_MODEL_BYTES = 256 * 1024 * 1024 * 1024
# What the hub's listing of pinned paths may answer with.
_MAX_PATHS_INFO_BYTES = 4 * 1024 * 1024

# Where Hugging Face serves a pinned file from.
HUGGINGFACE = "https://huggingface.co"

# A package name's SHAPE is the project reader's rule (`is_package_name`):
# `<namespace>/<package>`, each half a lowercase plain identifier. Checked
# before the name is ever part of a URL or a path, so a name off the network
# cannot name a directory above the one it belongs in.
_VERSION_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9.+-]*")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")

_NAME_HINT = (
    "a package name is <namespace>/<package>, each half a lowercase plain identifier"
)
_REGISTRY_HINT = (
    f"set {REGISTRY_ENV} to another registry, or check the network connection"
)
_PUBLISHED_HINT = "run `ffrwd search` to see what is published"
_LOGIN_HINT = "run `ffrwd login --token <token>` and try again"


def _reject(message: str, hint: str) -> FfrwdError:
    return FfrwdError(ErrorCode.UNSUPPORTED_SQL, message, hint=hint)


# --------------------------------------------------------------------------
# what the registry describes
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Listing:
    """One package as a search answers for it: what `search` prints and filters.

    `installs_week` is how many times its archive was fetched over the
    trailing week -- the number the registry counts at the signing endpoint,
    and 0 for a local directory, which counts nothing.
    """

    name: str
    version: str
    description: str
    functions: tuple[str, ...]
    recipes: tuple[str, ...]
    installs_week: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "functions": list(self.functions),
            "recipes": list(self.recipes),
            "installs_week": self.installs_week,
        }


@dataclass(frozen=True)
class Release:
    """One published version: what to fetch, and what its bytes must hash to."""

    name: str
    version: str
    sha256: str
    size: int


@dataclass(frozen=True)
class Installed:
    """What one install did, for the caller that reports it.

    `brought` is every OTHER package this install pulled in transitively, in
    the order they were resolved -- empty when `release` depends on nothing,
    or everything it depends on was already pinned. `replaced` is the entry
    the project directly pointed `release.name` at before this install, if
    this changes it -- not removed from the lockfile, since another
    package's own dependency may still need it; only what THIS project's
    manifest and this lockfile's own top-level `dependencies` name for
    `release.name` changes.
    """

    release: Release
    brought: tuple[Release, ...]
    replaced: LockEntry | None
    root: Path
    lock: Path
    manifest: Path | None
    downloaded: bool


@dataclass(frozen=True)
class ProjectInstalled:
    """What a bare install did for the package standing in the working tree.

    `brought` is every package fetched into the lockfile, in the order it
    was resolved -- empty when everything the manifest pins was already
    pinned and stored.
    """

    package: Package
    brought: tuple[Release, ...]
    lock: Path


# --------------------------------------------------------------------------
# where to read
# --------------------------------------------------------------------------


def _setting(name: str, fallback: str) -> str:
    written = os.environ.get(name)
    return written.strip() if written and written.strip() else fallback


def base_url() -> str:
    """The registry to read the detail documents from, the environment's if it set one."""
    return _setting(REGISTRY_ENV, DEFAULT_REGISTRY)


def api_url() -> str:
    """The host serving search, archive signing and private detail."""
    return _setting(API_ENV, DEFAULT_API)


def _local_root(base: str) -> Path | None:
    """The directory `base` names, or None when it is an HTTP registry.

    Anything that is not http(s) is read off the filesystem -- a ``file://``
    URL or a plain directory path. That is also the rule that keeps urllib
    from being handed a protocol nobody asked for: a registry is a set of
    files over HTTP or a directory of them, and there is no third kind.
    """
    if base.startswith(("http://", "https://")):
        return None
    if base.startswith("file://"):
        parsed = urllib.parse.urlparse(base)
        return Path(urllib.request.url2pathname(parsed.netloc + parsed.path))
    return Path(base)


def _where(relative: str) -> str:
    """The URL or path `relative` names under the current registry, for a message."""
    base = base_url()
    root = _local_root(base)
    if root is not None:
        return str(root.joinpath(*relative.split("/")))
    return f"{base.rstrip('/')}/{relative}"


def _unreachable(where: str, reason: str) -> FfrwdError:
    return _reject(f"the registry could not be read at {where}: {reason}", _REGISTRY_HINT)


def _too_large(where: str, limit: int) -> FfrwdError:
    return _reject(
        f"the registry served more than {limit} bytes at {where}",
        "that is not a file this registry publishes; check the base URL",
    )


# --------------------------------------------------------------------------
# reading over the network
# --------------------------------------------------------------------------

# Every request this module makes goes through here. It is the seam the unit
# tier replaces: no check in it opens a socket, and what it replaces behaves
# the way urllib does -- a response with `status` and `read`, and an
# `HTTPError` for anything but success.
_urlopen = urllib.request.urlopen


class _Status(Exception):
    """An HTTP status the caller judges rather than the fetcher.

    Raised only inside this module and never allowed out of it: a caller with
    nothing to say about a status turns it into a rejection naming the URL.
    `body` is what came back with it, which is the only way to tell some
    statuses apart -- see :func:`_absent`.
    """

    def __init__(self, code: int, body: bytes = b"") -> None:
        super().__init__(code)
        self.code = code
        self.body = body


# What object storage puts in the body when the object is not there. It
# answers 400 rather than 404 for a missing PUBLIC object, so the status alone
# does not say "no such package" and the body has to.
_ABSENT_MARKERS = ("404", "not_found", "NoSuchKey")


def _absent(status: _Status) -> bool:
    """True when `status` means the thing asked for is not there.

    A plain 404, or the 400 object storage answers a missing public object
    with -- whose body names the 404 the status does not.
    """
    if status.code == 404:
        return True
    if status.code != 400:
        return False
    try:
        data = json.loads(status.body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError, RecursionError):
        return False
    if not isinstance(data, dict):
        return False
    return any(
        str(data.get(key)) in _ABSENT_MARKERS for key in ("statusCode", "error", "code")
    )


def _request(
    url: str, *, headers: Mapping[str, str] | None = None, data: bytes | None = None
) -> urllib.request.Request:
    written = {"Accept": "*/*", **(dict(headers) if headers else {})}
    if data is not None:
        written.setdefault("Content-Type", "application/json")
    return urllib.request.Request(url, data=data, headers=written)


def _read(
    url: str, limit: int, request: urllib.request.Request, timeout: float = TIMEOUT
) -> bytes:
    """The bytes `url` serves, bounded by `limit`, or a rejection naming it.

    Raises :class:`_Status` for an HTTP status, which the caller decides about.
    """
    try:
        with _urlopen(request, timeout=timeout) as response:
            content = bytes(response.read(limit + 1))
    except urllib.error.HTTPError as err:
        try:
            said = bytes(err.read(limit + 1))
        except OSError:  # a refusal with no body to read
            said = b""
        raise _Status(err.code, said) from err
    except (OSError, ValueError, urllib.error.URLError) as err:
        reason = getattr(err, "reason", None)
        raise _unreachable(url, str(reason or err)) from err
    if len(content) > limit:
        raise _too_large(url, limit)
    return content


def _fetched(url: str, limit: int, request: urllib.request.Request) -> bytes:
    """:func:`_read` for a caller with nothing to say about a status."""
    try:
        return _read(url, limit, request)
    except _Status as status:
        raise _unreachable(url, f"HTTP {status.code}") from status


def _local_dir(base: str) -> Path | None:
    """:func:`_local_root`, checked to be there.

    A registry that is a directory path is only a registry when the directory
    exists: a typo would otherwise read as a registry publishing nothing.
    """
    root = _local_root(base)
    if root is None:
        return None
    try:
        present = root.is_dir()
    except OSError as err:  # pragma: no cover -- a path the OS refuses to stat
        raise _unreachable(str(root), err.strerror or str(err)) from err
    if not present:
        raise _unreachable(str(root), "there is no such directory")
    return root


def exchange(
    url: str,
    *,
    headers: Mapping[str, str],
    data: bytes,
    limit: int,
    timeout: float = TIMEOUT,
) -> tuple[int, bytes]:
    """One POST through this module's HTTP seam: the status, and the body.

    Unlike everything else here the STATUS comes back rather than becoming a
    rejection. Publishing is the one exchange whose refusals are the
    registry's own -- a name it will not take, a version it already has -- and
    they carry their own message and hint for the caller to render. Any
    success reads as 200; only whether it succeeded is a caller's question.
    """
    request = _request(url, headers=headers, data=data)
    try:
        return 200, _read(url, limit, request, timeout)
    except _Status as status:
        return status.code, status.body


def _read_file(path: Path, limit: int) -> bytes | None:
    """`path`'s bytes, or None when there is no such file."""
    try:
        with open(path, "rb") as handle:
            content = handle.read(limit + 1)
    except FileNotFoundError:
        return None
    except OSError as err:
        raise _unreachable(str(path), err.strerror or str(err)) from err
    if len(content) > limit:
        raise _too_large(str(path), limit)
    return content


def _token_headers() -> dict[str, str]:
    """The bearer this machine sends, or nothing when it holds no token."""
    token = credentials.load()
    return {} if token is None else {"Authorization": f"Bearer {token}"}


def _anon_headers() -> dict[str, str]:
    """What the search function takes: the project's public key, twice over."""
    return {"apikey": ANON_KEY, "Authorization": f"Bearer {ANON_KEY}"}


# --------------------------------------------------------------------------
# parsing what came back
# --------------------------------------------------------------------------


def _malformed(where: str, detail: str) -> FfrwdError:
    return _reject(
        f"{where} is not a registry file this ffrwd reads: {detail}",
        "the registry is serving something else, or a newer format; check the base URL",
    )


def _document(raw: bytes, where: str) -> dict[str, object]:
    """`raw` as the one JSON object a registry file is, format version checked."""
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError, RecursionError) as err:
        raise _malformed(where, "it is not JSON") from err
    if not isinstance(data, dict):
        raise _malformed(where, "it is not a JSON object")
    if data.get("format_version") != FORMAT_VERSION:
        raise _malformed(
            where,
            f"it is format version {data.get('format_version')!r} and this ffrwd "
            f"reads {FORMAT_VERSION}",
        )
    return data


def _text(data: Mapping[str, object], key: str, where: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise _malformed(where, f"{key!r} is not a non-empty string")
    return value


def _optional_text(data: Mapping[str, object], key: str, where: str) -> str:
    value = data.get(key, "")
    if not isinstance(value, str):
        raise _malformed(where, f"{key!r} is not a string")
    return value


def _names(data: Mapping[str, object], key: str, where: str) -> tuple[str, ...]:
    """The strings `key` holds -- a list of names, or a list of objects carrying one."""
    value = data.get(key, [])
    if not isinstance(value, list):
        raise _malformed(where, f"{key!r} is not a list")
    found: list[str] = []
    for item in value:
        if isinstance(item, str):
            found.append(item)
        elif isinstance(item, dict) and isinstance(item.get("name"), str):
            found.append(str(item["name"]))
        else:
            raise _malformed(where, f"{key!r} holds something that names nothing")
    return tuple(found)


def _count(data: Mapping[str, object], key: str, where: str) -> int:
    value = data.get(key, 0)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise _malformed(where, f"{key!r} is not a count")
    return value


def _objects(data: Mapping[str, object], key: str, where: str) -> list[dict[str, object]]:
    value = data.get(key)
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise _malformed(where, f"{key!r} is not a list of objects")
    return [item for item in value if isinstance(item, dict)]


def _checked_name(name: str, where: str) -> str:
    if not is_package_name(name):
        raise _malformed(where, f"{name!r} is not a package name")
    refused = name_refusal(name)
    if refused is not None:
        raise _malformed(where, f"{name!r}: {refused[0]}")
    return name


def _listing(raw: Mapping[str, object], where: str) -> Listing:
    return Listing(
        name=_checked_name(_text(raw, "name", where), where),
        version=_text(raw, "version", where),
        description=_optional_text(raw, "description", where),
        functions=_names(raw, "functions", where),
        recipes=_names(raw, "recipes", where),
        installs_week=_count(raw, "installs_week", where),
    )


# --------------------------------------------------------------------------
# search
# --------------------------------------------------------------------------

_SEARCH_PATH = "rest/v1/rpc/search_packages"


def _search_url() -> str:
    return f"{api_url().rstrip('/')}/{_SEARCH_PATH}"


def search(term: str | None = None) -> tuple[Listing, ...]:
    """The packages matching `term`, most relevant first.

    The registry ranks them: an exact name, then a near name, then the text of
    the description, the keywords and the names a package exports. No term
    browses everything; a term matching nothing is an empty result, which is
    an answer and not a rejection. A local directory has no ranking to offer
    and answers by reading the detail documents it holds.
    """
    written = (term or "").strip()
    root = _local_dir(base_url())
    if root is not None:
        return _local_search(root, written)
    url = _search_url()
    payload = json.dumps({"q": written}).encode("utf-8")
    raw = _fetched(
        url, _MAX_SEARCH_BYTES, _request(url, headers=_anon_headers(), data=payload)
    )
    return _search_rows(raw, url)


def _search_rows(raw: bytes, where: str) -> tuple[Listing, ...]:
    """The rows the search function answered with."""
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError, RecursionError) as err:
        raise _malformed(where, "it is not JSON") from err
    if not isinstance(data, list):
        raise _malformed(where, "it is not a list of packages")
    return tuple(
        _listing(row, where) for row in data if isinstance(row, dict)
    )


def _suggestions(term: str) -> tuple[str, ...]:
    """The names a search would put first for `term`, or nothing.

    Best effort by contract: this only decorates the rejection for a name the
    registry does not have, so a search that cannot be reached costs the
    suggestion and nothing else.
    """
    try:
        return tuple(listing.name for listing in search(term))
    except FfrwdError:
        return ()


def _local_search(root: Path, term: str) -> tuple[Listing, ...]:
    """What a local directory holds, filtered the way the fields read.

    Matched case-insensitively against the name, the description and the names
    of the functions each package exports -- the same fields the hosted search
    ranks over, without the ranking.
    """
    try:
        paths = sorted(root.glob("p/*/*.json"))
    except OSError as err:  # pragma: no cover -- a directory the OS refuses to walk
        raise _unreachable(str(root), err.strerror or str(err)) from err
    found: list[Listing] = []
    for path in paths:
        raw = _read_file(path, _MAX_DETAIL_BYTES)
        if raw is None:  # pragma: no cover -- removed between the glob and the read
            continue
        listing = _latest_listing(_document(raw, str(path)), str(path))
        if listing is not None and _matches(listing, term.lower()):
            found.append(listing)
    return tuple(found)


def _latest_listing(data: Mapping[str, object], where: str) -> Listing | None:
    """One detail document's highest version, as a listing, or None when it has none."""
    versions = _objects(data, "versions", where)
    if not versions:
        return None
    latest = max(versions, key=lambda raw: version_key(_text(raw, "version", where)))
    return _listing({**latest, "name": _text(data, "name", where)}, where)


def _matches(listing: Listing, needle: str) -> bool:
    if not needle:
        return True
    fields = (listing.name, listing.description, *listing.functions)
    return any(needle in field.lower() for field in fields)


# --------------------------------------------------------------------------
# resolving one package to one archive
# --------------------------------------------------------------------------


def version_key(version: str) -> tuple[tuple[int, int, str], ...]:
    """Sort key for a version: dot-separated parts, numeric ones compared as numbers.

    Enough for the exact-pin world v0 lives in -- it orders 1.10.0 above 1.9.0,
    which string order does not -- and it never has to decide what a range
    means, because nothing here solves one.
    """
    parts: list[tuple[int, int, str]] = []
    for piece in version.split("."):
        if piece.isdigit():
            parts.append((0, int(piece), ""))
        else:
            parts.append((1, 0, piece))
    return tuple(parts)


def is_version(text: str) -> bool:
    """True when `text` is spelled like a version -- what may follow '@' in a spec."""
    return _VERSION_RE.fullmatch(text) is not None


def _requested(request: str) -> tuple[str, str | None]:
    """`<name>` or `<name>@<version>` split, both halves checked for shape."""
    name, separator, version = request.partition("@")
    if not is_package_name(name):
        raise _reject(f"{request!r} does not name a package", _NAME_HINT)
    refused = name_refusal(name)
    if refused is not None:
        raise _reject(refused[0], refused[1])
    if not separator:
        return name, None
    if _VERSION_RE.fullmatch(version) is None:
        raise _reject(
            f"{request!r} does not name a version",
            "a version is written after '@', e.g. broadcast/tracks@1.2.0",
        )
    return name, version


def _detail_path(name: str) -> str:
    return f"p/{name}.json"


def _private_detail_url(name: str) -> str:
    return f"{api_url().rstrip('/')}/functions/v1/private/p/{name}"


def _release(raw: Mapping[str, object], name: str, where: str) -> Release:
    sha256 = _text(raw, "sha256", where)
    if _SHA256_RE.fullmatch(sha256) is None:
        raise _malformed(where, f"{sha256!r} is not a sha256 digest")
    size = raw.get("size")
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        raise _malformed(where, "'size' is not a byte count")
    return Release(
        name=name, version=_text(raw, "version", where), sha256=sha256, size=size
    )


def _did_you_mean(name: str) -> str:
    """What a search puts first for a name the registry does not have."""
    stem = name.partition("/")[2] or name
    close = [held for held in _suggestions(stem) if held != name][:3]
    return f"did you mean {', '.join(close)}?" if close else _PUBLISHED_HINT


def _no_such_package(name: str) -> FfrwdError:
    return _reject(f"the registry has no package '{name}'", _did_you_mean(name))


def _detail(name: str) -> tuple[dict[str, object], str]:
    """`name`'s detail document, and where it was read from.

    Public first. A 404 with credentials on this machine is retried against
    the endpoint that answers for private packages, since a package this user
    may see is not one the public documents list.
    """
    relative = _detail_path(name)
    root = _local_dir(base_url())
    if root is not None:
        path = root.joinpath(*relative.split("/"))
        raw = _read_file(path, _MAX_DETAIL_BYTES)
        if raw is None:
            raise _no_such_package(name)
        return _document(raw, str(path)), str(path)

    url = _where(relative)
    try:
        return _document(_read(url, _MAX_DETAIL_BYTES, _request(url)), url), url
    except _Status as status:
        if not _absent(status):
            raise _unreachable(url, f"HTTP {status.code}") from status
    headers = _token_headers()
    if not headers:
        raise _no_such_package(name)
    private = _private_detail_url(name)
    try:
        raw = _read(private, _MAX_DETAIL_BYTES, _request(private, headers=headers))
    except _Status as status:
        if status.code in (401, 403):
            raise _reject(
                f"this token does not authorize reading '{name}'",
                "the token's account is not a member of that namespace; mint one that is",
            ) from status
        if _absent(status):
            raise _no_such_package(name) from status
        raise _unreachable(private, f"HTTP {status.code}") from status
    return _document(raw, private), private


def resolve(request: str) -> Release:
    """The published version `request` names: the highest one when it names none.

    One fetch: the package's own detail document says which versions exist and
    what each one's archive must hash to. Exact pins only -- there is nothing
    to solve here, and nothing that does.
    """
    name, wanted = _requested(request)
    data, where = _detail(name)
    if _checked_name(_text(data, "name", where), where) != name:
        raise _malformed(where, f"it describes another package than '{name}'")
    releases = [_release(raw, name, where) for raw in _objects(data, "versions", where)]
    if not releases:
        raise _reject(f"the registry publishes no version of '{name}'", _PUBLISHED_HINT)
    if wanted is None:
        return max(releases, key=lambda release: version_key(release.version))
    for release in releases:
        if release.version == wanted:
            return release
    published = ", ".join(
        release.version
        for release in sorted(releases, key=lambda release: version_key(release.version))
    )
    raise _reject(
        f"the registry has no version {wanted} of '{name}'", f"published: {published}"
    )


# --------------------------------------------------------------------------
# fetching one archive
# --------------------------------------------------------------------------


def _archive_url(sha256: str) -> str:
    return f"{api_url().rstrip('/')}/functions/v1/archive/{sha256}"


def sign_url(release: Release) -> str:
    """Ask the registry for a URL this archive can be downloaded from.

    The signing request is where an install is counted, so every archive goes
    through it -- public and private alike. This machine's token is sent when
    it has one; a refusal without one says to log in, since a private version
    is exactly what looks like this.
    """
    url = _archive_url(release.sha256)
    try:
        raw = _read(url, _MAX_DETAIL_BYTES, _request(url, headers=_token_headers()))
    except _Status as status:
        if status.code in (401, 403):
            if credentials.load() is None:
                raise _reject(
                    f"package '{release.name}' {release.version} is not public, and "
                    "this machine holds no ffrwd token",
                    _LOGIN_HINT,
                ) from status
            raise _reject(
                f"this token does not authorize downloading '{release.name}' "
                f"{release.version}",
                "the token's account is not a member of that namespace; mint one that is",
            ) from status
        raise _unreachable(url, f"HTTP {status.code}") from status
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError, RecursionError) as err:
        raise _malformed(url, "it is not JSON") from err
    if not isinstance(data, dict):
        raise _malformed(url, "it is not a JSON object")
    signed = data.get("url")
    if not isinstance(signed, str) or not signed.startswith(("http://", "https://")):
        raise _malformed(url, "'url' is not an http URL to download from")
    return signed


def _archive_bytes(release: Release) -> bytes:
    """`release`'s archive: off the filesystem, or through a signed URL."""
    relative = f"archives/{release.sha256}"
    root = _local_dir(base_url())
    if root is not None:
        raw = _read_file(root.joinpath(*relative.split("/")), _MAX_ARCHIVE_BYTES)
        if raw is None:
            raise _unreachable(_where(relative), "no such file")
        return raw
    signed = sign_url(release)
    return _fetched(signed, _MAX_ARCHIVE_BYTES, _request(signed))


def fetch(release: Release) -> Path:
    """`release`'s content in the store, downloading it only when it is not there.

    The bytes are verified against the digest the registry recorded before
    anything opens them, which is :func:`ffrwd.store.unpack`'s contract: a
    download that does not match is discarded unopened and nothing is written.
    """
    archive = _archive_bytes(release)
    if len(archive) != release.size:
        raise _reject(
            f"package '{release.name}': the registry served {len(archive)} bytes for "
            f"archive {release.sha256} and recorded {release.size}",
            "the download is not what was published; nothing was written",
        )
    return store.unpack(release.name, archive, release.sha256)


def stored(release: Release) -> Path | None:
    """Where `release` already sits in the store, or None when nothing does."""
    directory = store.store_dir() / store.entry_path(release.sha256)
    try:
        return directory if directory.is_dir() else None
    except OSError:  # pragma: no cover -- a path the OS refuses to stat
        return None


# --------------------------------------------------------------------------
# the models a package pins
# --------------------------------------------------------------------------


def model_url(pin: ModelPin) -> str:
    """Where Hugging Face serves the exact file `pin` names."""
    return f"{HUGGINGFACE}/{pin.repo}/resolve/{pin.revision}/{pin.file}"


def _model_refusal(package: str, export: str, pin: ModelPin, detail: str) -> FfrwdError:
    return _reject(
        f"package '{package}': the model for '{export}' "
        f"({pin.repo}@{pin.revision} {pin.file}) {detail}",
        "the package pins it in its manifest's models; nothing was installed",
    )


def _digest_of(path: Path) -> str | None:
    """`path`'s sha256, or None when there is no file there."""
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError:
        return None
    return digest.hexdigest()


def _download_model(
    package: str,
    export: str,
    pin: ModelPin,
    destination: Path,
    announce: Announce | None = None,
) -> None:
    """Fetch, verify and place one model file. Nothing is written until it verifies.

    Streamed into a temporary file beside its destination and moved onto it,
    so a model is complete or absent and never half of either -- and hashed on
    the way through, since it is far too large to hold.
    """
    url = model_url(pin)
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(
            dir=destination.parent, prefix=f"{destination.name}-", suffix=".tmp"
        )
    except OSError as err:
        raise _model_refusal(
            package, export, pin, f"could not be written: {err.strerror or err}"
        ) from err
    try:
        with os.fdopen(handle, "wb") as file:
            found, arrived = _stream(url, file, package, export, pin, announce)
        if arrived > _MAX_MODEL_BYTES:
            raise _model_refusal(
                package,
                export,
                pin,
                f"ran past the {written_size(_MAX_MODEL_BYTES)} one file may weigh, "
                f"and was abandoned at {written_size(arrived)}",
            )
        if found != pin.sha256:
            raise _model_refusal(
                package, export, pin, f"hashes to {found}, and {pin.sha256} was expected"
            )
        try:
            os.replace(temporary, destination)
        except OSError as err:
            raise _model_refusal(
                package, export, pin, f"could not be written: {err.strerror or err}"
            ) from err
    finally:
        try:
            os.unlink(temporary)
        except OSError:  # already moved onto the destination, or already gone
            pass


def _model_notice(pin: ModelPin, size: int | None) -> str:
    """The one line a model fetch announces: the file, its size, and the repo."""
    if size is None:
        return f"model {pin.file} from {pin.repo}"
    return f"model {pin.file} ({written_size(size)}) from {pin.repo}"


def _content_length(response: object) -> int | None:
    """What the answer says it weighs, or None when it does not say."""
    headers = getattr(response, "headers", None)
    written = headers.get("Content-Length") if headers is not None else None
    try:
        return int(written) if written else None
    except (TypeError, ValueError):
        return None


def _stream(
    url: str,
    file: IO[bytes],
    package: str,
    export: str,
    pin: ModelPin,
    announce: Announce | None = None,
) -> tuple[str, int]:
    """Copy `url` into `file`, hashing it: the digest, and how many bytes arrived.

    Abandoned at the block that crosses the backstop rather than after the
    whole thing has been written, so a hostile answer costs one block over the
    bound -- and a count past the backstop is what says it was abandoned.
    `announce` hears the one line naming the model, once the answer is open
    and its size is known.
    """
    digest = hashlib.sha256()
    written = 0
    try:
        with _urlopen(_request(url), timeout=TIMEOUT) as response:
            if announce is not None:
                announce(_model_notice(pin, _content_length(response)))
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    return digest.hexdigest(), written
                written += len(block)
                if written > _MAX_MODEL_BYTES:
                    return digest.hexdigest(), written
                digest.update(block)
                file.write(block)
    except urllib.error.HTTPError as err:
        raise _model_refusal(package, export, pin, f"answered HTTP {err.code}") from err
    except (OSError, ValueError, urllib.error.URLError) as err:
        reason = getattr(err, "reason", None)
        raise _model_refusal(
            package, export, pin, f"could not be read at {url}: {reason or err}"
        ) from err


def _install_models(package: Package, announce: Announce | None = None) -> None:
    """Put every file `package` pins beside the module whose export loads it.

    The compiler looks for ``<export>.onnx`` beside the module's wasm file, so
    that is where each export's FIRST pin lands. A model of several files
    lands the rest beside it under their own names, which is what the graph
    refers to them by. A file already there and hashing to its pin is left
    alone; one that fails to arrive fails the install, naming the model and
    the pin.
    """
    if not package.models:
        return
    modules = {declared.export: declared.module for declared in package_modules(package)}
    for export, pins in package.models.items():
        module = modules.get(export)
        if module is None:
            raise _model_refusal(
                package.name, export, pins[0], "names an export no module in it declares"
            )
        graph = Path(wasm.model_path(module, export))
        for number, pin in enumerate(pins):
            destination = graph if number == 0 else graph.parent / pin.filename
            if _digest_of(destination) == pin.sha256:
                continue
            _download_model(package.name, export, pin, destination, announce)


def model_sizes(pins: Sequence[ModelPin]) -> dict[ModelPin, int]:
    """What each pinned file weighs on the hub, for the pins the hub answers for.

    One request per repository and revision, asking about only the files
    pinned there. A pin the hub says nothing about is simply absent from the
    answer: a size is a courtesy, never a reason to refuse anything.
    """
    by_revision: dict[tuple[str, str], list[ModelPin]] = {}
    for pin in pins:
        by_revision.setdefault((pin.repo, pin.revision), []).append(pin)
    found: dict[ModelPin, int] = {}
    for (repo, revision), pinned in by_revision.items():
        weighed = _hub_sizes(repo, revision, [pin.file for pin in pinned])
        for pin in pinned:
            size = weighed.get(pin.file)
            if size is not None:
                found[pin] = size
    return found


def _hub_sizes(repo: str, revision: str, files: Sequence[str]) -> dict[str, int]:
    """What the hub says each of `files` weighs in one revision, keyed by path.

    Empty when the answer does not arrive or cannot be read. An LFS file's
    real size is the one under "lfs"; the entry's own is the pointer's.
    """
    url = f"{HUGGINGFACE}/api/models/{repo}/paths-info/{revision}"
    body = json.dumps({"paths": list(files)}).encode("utf-8")
    try:
        answered = json.loads(_read(url, _MAX_PATHS_INFO_BYTES, _request(url, data=body)))
    except (_Status, FfrwdError, OSError, ValueError):
        return {}
    if not isinstance(answered, list):
        return {}
    sizes: dict[str, int] = {}
    for entry in answered:
        if not isinstance(entry, dict):
            continue
        lfs = entry.get("lfs")
        size = lfs.get("size") if isinstance(lfs, dict) else entry.get("size")
        written = entry.get("path")
        if isinstance(written, str) and isinstance(size, int) and not isinstance(size, bool):
            sizes[written] = size
    return sizes


def module_capabilities(package: Package) -> dict[str, tuple[str, ...]]:
    """What each of `package`'s modules needs granted, keyed by the path its lib names.

    Read from what the sidecar says each module declares, not from anything
    the manifest claims: a module that runs a model is what makes the package
    one that needs one. Keyed by the written path so a caller can name the
    module a capability came from.
    """
    found: dict[str, tuple[str, ...]] = {}
    for declared in package_modules(package):
        try:
            written = Path(declared.module).relative_to(package.root).as_posix()
        except ValueError:  # a module path the package does not hold
            written = declared.module
        described = wasm.describe(declared.module)
        found[written] = tuple(
            name
            for name, needed in (
                ("http", described.http),
                ("nn", described.nn),
                ("udp", described.udp),
            )
            if needed
        )
    return found


def capabilities(package: Package) -> tuple[str, ...]:
    """What this package's modules do beyond filtering frames, in name order."""
    return tuple(
        sorted({name for needed in module_capabilities(package).values() for name in needed})
    )


def _install_runtime(package: Package, announce: Announce | None = None) -> None:
    """Fetch the ONNX Runtime a package that runs models will need to run it.

    Here rather than at the first query, so the whole cost of installing such
    a package is paid in one place with a network connection already assumed.
    A package whose modules cannot be described is left to the run, which
    provisions the same way: reading a module is the sidecar's job, and a
    missing sidecar is that command's rejection rather than this one's.
    """
    try:
        needed = "nn" in capabilities(package)
    except FfrwdError:
        return
    if needed:
        nn.ensure(announce=announce)


# --------------------------------------------------------------------------
# installing
# --------------------------------------------------------------------------


def _agrees(release: Release, root: Path) -> None:
    """The installed package's name and version, checked against the registry's.

    The registry said what it was publishing and the package says what it is;
    a disagreement is caught here rather than at the next compile, where the
    lockfile reader would refuse an entry this command had just written.
    """
    package = read_manifest(root / MANIFEST_NAME)
    for recorded, found, field in (
        (release.name, package.name, "name"),
        (release.version, package.version, "version"),
    ):
        if recorded != found:
            raise _reject(
                f"package '{release.name}': the registry records {field} {recorded!r} and "
                f"the package says {found!r}",
                "the registry is serving an archive for another package; report it",
            )


def _entry_for(name: str, version: str, entries: Sequence[LockEntry]) -> RegistryEntry | None:
    """The registry entry pinning `name` at exactly `version`, or None."""
    for entry in entries:
        if isinstance(entry, RegistryEntry) and entry.name == name and entry.version == version:
            return entry
    return None


def _ensure(
    release: Release,
    entries: list[LockEntry],
    chain: list[str],
    brought: list[Release],
    announce: Announce | None = None,
) -> None:
    """Make sure `release`, and everything its manifest depends on, sits in `entries`.

    Post-order: a package's own entry is appended only once every dependency
    it names has been resolved, so the `dependencies` map recorded on it is
    complete the moment it is written. `chain` is the names currently being
    walked -- an ancestor reappearing is a cycle, checked before anything
    else, since it is what stops an infinite walk. An exact (name, version)
    already in `entries` is left alone -- not walked again, and not refetched
    unless its content has gone missing from the store, in which case the
    content is brought back and the entry stands as written. A
    different version of the same name is not a conflict -- it is simply
    added beside the one already there; nothing here picks between them.
    """
    if release.name in chain:
        loop = " -> ".join([*chain[chain.index(release.name) :], release.name])
        raise _reject(
            f"dependency cycle: {loop}",
            "there is no resolver here to break it; one of these packages has to stop "
            "depending on another in the loop",
        )
    pinned = _entry_for(release.name, release.version, entries)
    already = stored(release)
    if pinned is not None and already is not None:
        return
    if already is None and announce is not None:
        announce(
            f"fetching {release.name} {release.version} ({written_size(release.size)})"
        )
    root = already if already is not None else fetch(release)
    _agrees(release, root)
    package = read_manifest(root / MANIFEST_NAME)
    _install_models(package, announce)
    _install_runtime(package, announce)

    if pinned is not None:
        # The entry already says what this version resolved each dependency
        # to; bring back any of that content the store lost, at those exact
        # versions, and leave the lockfile as it was.
        chain.append(release.name)
        for name, version in pinned.dependencies.items():
            if announce is not None:
                announce(f"resolving {name}")
            _ensure(resolve(f"{name}@{version}"), entries, chain, brought, announce)
        chain.pop()
        brought.append(release)
        return

    chain.append(release.name)
    resolved: dict[str, str] = {}
    for name in package.dependencies:
        if announce is not None:
            announce(f"resolving {name}")
        dependency = resolve(name)
        _ensure(dependency, entries, chain, brought, announce)
        resolved[name] = dependency.version
    chain.pop()

    entries.append(
        RegistryEntry(
            name=release.name,
            version=release.version,
            sha256=release.sha256,
            store=store.entry_path(release.sha256),
            dependencies=resolved,
        )
    )
    brought.append(release)


def _write_lockfile_migrating(
    lock: Path, entries: Sequence[LockEntry], wanted: Mapping[str, str]
) -> None:
    """Write `lock`, moving any link entry an older ffrwd left in it to the links file.

    The links file is written first: a failure between the two writes leaves a
    link recorded in both places, which reads as one link and is cleaned by
    the next write, rather than in neither.
    """
    carried = [entry for entry in entries if isinstance(entry, LinkEntry)]
    if carried:
        beside = links_path(lock)
        held = list(read_linksfile(beside))
        known = {link_target(entry, beside) or entry.path for entry in held}
        for entry in carried:
            if (link_target(entry, lock) or entry.path) not in known:
                held.append(entry)
        write_linksfile(beside, held)
    write_lockfile(
        lock,
        [entry for entry in entries if not isinstance(entry, LinkEntry)],
        dependencies=wanted,
    )


def install(
    request: str,
    *,
    lock: Path,
    manifest: Path | None = None,
    announce: Announce | None = None,
) -> Installed:
    """Install `request` into the lockfile `lock`, recording it in `manifest`.

    Fetches what it depends on too, recursively, each at its highest
    published version. In order: resolve the version, put its content in the
    store, fetch the models it pins, walk its manifest's own dependencies the
    same way, then write the lockfile and the manifest. Nothing is recorded
    before the content is there, so an install that fails anywhere leaves a
    project pinning only what it had.

    Only `release.name` is recorded in `manifest`'s own dependencies -- what
    it pulled in transitively is the lockfile's business, not the project's.
    With no `manifest` -- a global install -- nothing is written there either,
    but the lockfile's own top-level `dependencies` still records what was
    directly asked for, since that is what lets a call written in THIS
    lockfile's own script resolve at the right version.

    `announce` hears one line per step -- resolving, each archive fetched,
    each model downloaded, the runtime -- and nothing when a step costs
    nothing.
    """
    if announce is not None:
        announce(f"resolving {request}")
    release = resolve(request)
    was_stored = stored(release) is not None

    current = read_lockfile(lock) if lock.is_file() else None
    entries: list[LockEntry] = list(current.entries) if current is not None else []
    wanted: dict[str, str] = dict(current.dependencies) if current is not None else {}
    previous_version = wanted.get(release.name)
    previous_entry = (
        _entry_for(release.name, previous_version, entries)
        if previous_version is not None
        else None
    )

    brought: list[Release] = []
    _ensure(release, entries, [], brought, announce)
    # `release` itself was brought along too, by the same walk; it is not one
    # of its OWN dependencies.
    brought = [
        one for one in brought if not (one.name == release.name and one.version == release.version)
    ]

    wanted[release.name] = release.version
    _write_lockfile_migrating(lock, entries, wanted)
    if manifest is not None:
        add_dependency(manifest, release.name, release.version)

    root = stored(release)
    if root is None:  # `_ensure` verified or fetched it, so something removed it since
        raise _reject(
            f"package '{release.name}' {release.version} was fetched but its "
            "content is not in the store",
            "something is removing store entries while install runs; "
            "try the install again",
        )
    return Installed(
        release=release,
        brought=tuple(brought),
        replaced=previous_entry,
        root=root,
        lock=lock,
        manifest=manifest,
        downloaded=not was_stored,
    )


def install_project(
    manifest: Path,
    *,
    lock: Path,
    announce: Announce | None = None,
) -> ProjectInstalled:
    """Install what the package at `manifest` needs to build and publish.

    Everything installing this package FROM the registry would have
    fetched, done for the working tree itself: each dependency the
    manifest pins, recursively; the models it pins, placed beside its own
    modules; and the runtime those modules load. A fresh clone is ready
    after this. Dependencies resolve at the manifest's written version --
    the manifest of the tree standing here is the pin, not a request for
    the highest.

    A dependency this project links to a working directory is left as the
    link it is: that is what a link is for, and fetching the registry's copy
    beside it would shadow the link and refuse the whole install for a
    package that is not published yet.
    """
    package = read_manifest(manifest)
    current = read_lockfile(lock) if lock.is_file() else None
    entries: list[LockEntry] = list(current.entries) if current is not None else []
    wanted: dict[str, str] = dict(current.dependencies) if current is not None else {}
    linked = {stored_name(entry, source) for entry, source in held_links(lock)}

    brought: list[Release] = []
    for name, version in package.dependencies.items():
        if name in linked:
            if announce is not None:
                announce(f"{name} is linked to a working directory")
            continue
        if announce is not None:
            announce(f"resolving {name} {version}")
        release = resolve(f"{name}@{version}")
        _ensure(release, entries, [], brought, announce)
        wanted[name] = release.version
    _write_lockfile_migrating(lock, entries, wanted)

    _install_models(package, announce)
    _install_runtime(package, announce)
    return ProjectInstalled(package=package, brought=tuple(brought), lock=lock)
