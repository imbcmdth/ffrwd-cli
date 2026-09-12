"""The package store: installed package content, addressed by its archive's sha256.

A package is not only SQL -- a wasm filter ships a binary -- so what travels
between a registry and a client is one compressed archive, and the digest a
lockfile pins is over that archive's bytes. Hashing what travels is what lets
a bad download be thrown away without being opened: bytes in, hash, compare to
the pin, discard. Nothing unverified reaches the extractor.

Four things live here. :func:`pack` builds a package directory into a
deterministic gzipped tar, so the same content produces the same bytes and so
the same digest on any machine. What it puts in is the manifest's closure --
the files a package cannot be read without -- plus whatever the manifest's
``files`` names, and nothing else: the built module ships and the tree it was
built from does not. :func:`unpack` verifies bytes against a digest and writes
what they hold into the store. :func:`load` turns a lockfile entry back into a
directory to read. The fourth is the MODEL cache at the foot of the file,
which is a different thing under the same roof: the large files a manifest
pins on the hub, kept once by their own digest and linked into each package
entry that pins them -- by hard link, symlink or copy, whichever the
filesystem allows.

Layout follows the registry's disk cache (`registry.py`): everything under
``~/.cache/ffrwd/``, :func:`_cache_dir` the only place that names the home
directory (and the seam a test redirects), and a format version -- here the
first component of every store path, so content written by a future layout
cannot be read as this one's.

One discipline differs, and the difference matters. The registry cache is an
OPTIMIZATION: every failure there is swallowed and the answer rebuilt from
ffmpeg. A store entry is the only copy of what a lockfile pinned, so an entry
that is missing, unreadable or written by another layout is a rejection naming
the package -- never a fall back to some other content.

Reads out of the store do not hash anything. The digest did its work at the
boundary where the bytes were untrusted; a store entry is the user's own cache
under their own home, on the near side of that boundary.

Verified is not the same as safe to extract: a digest proves an archive is the
one that was published, not that its members behave. The extractor takes
regular files and directories under the destination and nothing else -- no
absolute paths, no ``..``, no links, no devices -- with a member count and an
uncompressed size cap.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import os
import re
import secrets
import shutil
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from .errors import ErrorCode, FfrwdError
from .warnings import FfrwdWarning, OnWarning, WarningCode

__all__ = [
    "GITIGNORE_NAME",
    "IGNORE_NAME",
    "LICENSE_NAMES",
    "MODEL_FORMAT",
    "STORE_FORMAT",
    "SUBSET_HINT",
    "cache_model",
    "cached_model",
    "keep_model",
    "entry_path",
    "global_lock_path",
    "load",
    "model_entry_path",
    "model_staging",
    "models_dir",
    "pack",
    "place_model",
    "store_dir",
    "unpack",
    "unreadable_pattern",
]

# The store layout's version, and the first component of every store path.
# Bump it on any change to what a store directory holds: an entry written by
# another version is rejected, never guessed at.
STORE_FORMAT = "v1"

_SHA256_RE = re.compile(r"[0-9a-f]{64}")

_REINSTALL_HINT = "install the package again to put its content back in the store"

_ARCHIVE_HINT = "the archive is not a package this ffrwd will unpack; report it to its author"

# Fixed modes and ownership, so the same content packs to the same bytes
# whatever the umask and whoever is packing.
_FILE_MODE = 0o644
_DIR_MODE = 0o755

# What an archive may hold. A package is SQL plus at most a filter binary, so
# both sit far above anything legitimate and still bound what a hostile
# archive costs: extraction stops at the member that crosses one.
_MAX_MEMBERS = 4096
_MAX_UNPACKED_BYTES = 64 * 1024 * 1024


def _cache_dir() -> Path:
    try:
        return Path.home() / ".cache" / "ffrwd"
    except RuntimeError:  # pragma: no cover -- no resolvable home directory
        return Path(tempfile.gettempdir()) / "ffrwd-cache"


def store_dir() -> Path:
    """The root every stored package sits under."""
    return _cache_dir() / "packages"


def global_lock_path() -> Path:
    """The machine-wide lockfile, written by a global install.

    Beside the store rather than in a config directory: the two are written
    together, they are recovered together by reinstalling, and one home-
    directory seam covers both.
    """
    return _cache_dir() / "ffrwd.lock"


def entry_path(sha256: str) -> str:
    """Where content of this digest belongs, relative to :func:`store_dir`."""
    return f"{STORE_FORMAT}/{sha256[:2]}/{sha256}"


def _reject(message: str, hint: str) -> FfrwdError:
    return FfrwdError(ErrorCode.UNSUPPORTED_SQL, message, hint=hint)


def _require_digest(package: str, sha256: str) -> None:
    """Reject a digest that is not one, before it names a path or a comparison."""
    if _SHA256_RE.fullmatch(sha256) is None:
        raise _reject(
            f"package '{package}': {sha256!r} is not a sha256 digest",
            "a digest is 64 lowercase hex characters",
        )


# --------------------------------------------------------------------------
# packing
# --------------------------------------------------------------------------


# The ignore files read at the package root, in the order their patterns
# merge. Neither replaces the other: with both there the exclusions are the
# union of what the two declare, which is safe because the manifest's closure
# ships whatever they say.
IGNORE_NAME = ".ffrwdignore"
GITIGNORE_NAME = ".gitignore"

# The licence spellings that ship with the closure, at the package root. A
# package arriving without its terms is worse than one carrying a spare
# kilobyte, so the file travels like README.md rather than waiting to be
# named. Separate from the manifest's "license", which is the identifier:
# the field says which licence, the file carries its text.
LICENSE_NAMES = (
    "LICENSE",
    "LICENSE.md",
    "LICENSE.txt",
    "LICENCE",
    "LICENCE.md",
    "LICENCE.txt",
)

# Pattern characters this subset does not read: character classes, the
# single-character wildcard, and escapes.
_UNREADABLE_CHARACTERS = frozenset("?[]\\")

SUBSET_HINT = (
    "the patterns read here are a name, 'dir/', '*' within a path segment and "
    "'**' across them"
)

_NEGATION_HINT = (
    "there is nothing to negate: every file the manifest names ships whatever "
    "the ignore rules say"
)


@dataclass(frozen=True)
class _Pattern:
    """One exclusion an ignore file declares."""

    regex: re.Pattern[str]
    # Written with a slash in it, so it matches the path from the package root
    # rather than a name at any depth.
    anchored: bool
    # Written `dir/`, so it matches directories only.
    directory: bool


def _translate(glob: str) -> str:
    """`glob` as a regex source: ``*`` matches within a segment, ``**`` across them."""
    source: list[str] = []
    index = 0
    while index < len(glob):
        if glob.startswith("**/", index):
            source.append("(?:.*/)?")
            index += 3
        elif glob.startswith("**", index):
            source.append(".*")
            index += 2
        elif glob[index] == "*":
            source.append("[^/]*")
            index += 1
        else:
            source.append(re.escape(glob[index]))
            index += 1
    return "".join(source)


def _pattern(written: str) -> _Pattern:
    """One ignore line as the exclusion it declares."""
    directory = written.endswith("/")
    glob = written.rstrip("/")
    if glob.startswith("/"):
        anchored = True
        glob = glob.lstrip("/")
    else:
        anchored = "/" in glob
    return _Pattern(regex=re.compile(_translate(glob)), anchored=anchored, directory=directory)


def unreadable_pattern(written: str) -> str | None:
    """What is wrong with this pattern, or None when the subset reads it.

    One grammar, wherever a pattern is written: an ignore file's lines and the
    manifest's ``files``.
    """
    if written.startswith("!"):
        return "negates a pattern"
    if _UNREADABLE_CHARACTERS.intersection(written):
        return "is written with a pattern character this grammar does not read"
    return None


def _patterns(root: Path, package: str, on_warning: OnWarning | None) -> tuple[_Pattern, ...]:
    """The exclusions the root's ignore files declare, merged.

    ``.ffrwdignore`` is ffrwd's own, so its grammar is enforced: a line outside
    the subset is a rejection naming it. ``.gitignore`` was written for another
    tool and is only borrowed, so a line outside the subset is skipped and
    warned about instead. Only the package root is read; ignore files in
    subdirectories are not.
    """
    found: list[_Pattern] = []
    for name, own in ((IGNORE_NAME, True), (GITIGNORE_NAME, False)):
        path = root / name
        if not path.is_file():
            continue
        for number, line in enumerate(
            path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
        ):
            written = line.strip()
            if not written or written.startswith("#"):
                continue
            wrong = unreadable_pattern(written)
            if wrong is None:
                found.append(_pattern(written))
                continue
            hint = _NEGATION_HINT if written.startswith("!") else SUBSET_HINT
            if own:
                raise _reject(f"{path}: line {number}: {written!r} {wrong}", hint)
            if on_warning is not None:
                on_warning(
                    FfrwdWarning(
                        code=WarningCode.IGNORE_PATTERN,
                        package=package,
                        message=f"{name} line {number}: {written!r} {wrong}, and was skipped",
                        line=number,
                        hint=hint,
                    )
                )
    return tuple(found)


def _matched(patterns: tuple[_Pattern, ...], relative: str, directory: bool) -> bool:
    """True when `patterns` match `relative` or a directory above it."""
    segments = relative.split("/")
    for depth in range(1, len(segments) + 1):
        prefix = "/".join(segments[:depth])
        # Every prefix short of the whole path names a directory; the whole
        # path names one only when the entry itself is a directory.
        is_directory = directory or depth < len(segments)
        for pattern in patterns:
            if pattern.directory and not is_directory:
                continue
            if pattern.regex.fullmatch(prefix if pattern.anchored else segments[depth - 1]):
                return True
    return False


@dataclass(frozen=True)
class _Declared:
    """What the package at the root says its archive holds.

    `closure` is what the package cannot be read without; `files` is what the
    manifest's own ``files`` names on top of it. `manifested` is False for a
    directory holding no manifest, which has nothing to declare an archive
    with and packs the way it always did.
    """

    closure: frozenset[str]
    files: tuple[_Pattern, ...]
    package: str
    manifested: bool = True


def _declared(root: Path) -> _Declared:
    """What the manifest at `root` puts in the archive, and the name to call it by.

    The closure is the manifest itself, every lib and bin file it names, every
    module its lib SQL declares, README.md and the licence -- as relative posix
    paths. These ship whatever the ignore rules say, which is what lets a build
    directory be excluded while the built module inside it still travels.
    ``files`` is read as patterns of the same grammar the ignore files are
    written in.

    A manifest that does not read, or that names a lib or bin file that is not
    there, raises where it always did.
    """
    # Deferred: project.py and functions.py both import this module.
    from .functions import package_modules
    from .project import MANIFEST_NAME, README_NAME, read_manifest

    manifest = root / MANIFEST_NAME
    if not manifest.is_file():
        return _Declared(closure=frozenset(), files=(), package=root.name, manifested=False)
    package = read_manifest(manifest)
    found = {MANIFEST_NAME}
    for name in (README_NAME, *LICENSE_NAMES):
        if (root / name).is_file():
            found.add(name)
    named = [*package.exports.values(), *package.recipes.values()]
    named.extend(Path(declared.module) for declared in package_modules(package))
    for path in named:
        try:
            found.add(path.relative_to(root).as_posix())
        except ValueError:  # a path the package does not hold
            continue
    return _Declared(
        closure=frozenset(found),
        files=tuple(_pattern(written) for written in package.files),
        package=package.name,
    )


def _ships(relative: str, declared: _Declared, patterns: tuple[_Pattern, ...]) -> bool:
    """True when this file belongs in the archive.

    The closure first and unconditionally. Everything else ships only because
    the manifest's ``files`` names it, and not then if it is a dot-entry or the
    ignore rules exclude it: the archive is what the package declares, not the
    directory minus what someone remembered to exclude. A directory holding no
    manifest has neither, and keeps the older rule -- everything the ignore
    rules leave.
    """
    if relative in declared.closure:
        return True
    if relative.rpartition("/")[2].startswith("."):
        return False
    if _matched(patterns, relative, False):
        return False
    if not declared.manifested:
        return True
    return _matched(declared.files, relative, False)


def _descends(relative: str, declared: _Declared, patterns: tuple[_Pattern, ...]) -> bool:
    """True when the archive may hold something under this directory.

    The closure reaching inside is what pulls a built module out of an excluded
    build directory. Past that a walk is worth it only where something could
    ship: what ``files`` names, or -- with no manifest to name anything -- the
    whole tree.
    """
    if any(path.startswith(f"{relative}/") for path in declared.closure):
        return True
    if relative.rpartition("/")[2].startswith("."):
        return False
    if _matched(patterns, relative, True):
        return False
    return bool(declared.files) or not declared.manifested


def _refuse_irregular(path: Path, link: bool) -> None:
    """Refuse an entry that would ship and is not a regular file.

    A link the package declares as content is a rejection; one lying about the
    tree unnamed is simply not packed, like anything else nothing names.
    """
    if link or not (path.is_dir() or path.is_file()):
        raise _reject(
            f"{path} cannot be packed: a package holds regular files and directories only",
            "remove the link or special file from the package directory",
        )


def _walk(
    directory: Path,
    prefix: str,
    declared: _Declared,
    patterns: tuple[_Pattern, ...],
    found: list[tuple[str, Path]],
) -> bool:
    """Collect what ships under `directory`, whose path from the root is `prefix`.

    True when anything under it ships, which is what makes the directory itself
    a member: an archive carries no directory it puts nothing in.
    """
    holds_shipped = False
    for path in directory.iterdir():
        relative = f"{prefix}{path.name}"
        link = path.is_symlink()
        holds = path.is_dir() and not link
        if holds:
            if not _descends(relative, declared, patterns):
                continue
            if _walk(path, f"{relative}/", declared, patterns, found):
                found.append((relative, path))
                holds_shipped = True
            continue
        if not _ships(relative, declared, patterns):
            continue
        _refuse_irregular(path, link)
        found.append((relative, path))
        holds_shipped = True
    return holds_shipped


def _entries(
    root: Path, declared: _Declared, patterns: tuple[_Pattern, ...]
) -> list[tuple[str, Path]]:
    """Every directory and file under `root` that ships, as (relative posix path, path).

    Sorted, so a parent comes before what it holds and the member order is the
    tree's own order rather than the filesystem's. A directory nothing ships
    out of is not descended into, so a build directory costs one decision
    rather than a walk.
    """
    found: list[tuple[str, Path]] = []
    _walk(root, "", declared, patterns, found)
    return sorted(found)


# How many left-out names one warning prints before it counts the rest.
_LEFT_OUT_SHOWN = 8


def _left_out(
    root: Path,
    entries: list[tuple[str, Path]],
    declared: _Declared,
    patterns: tuple[_Pattern, ...],
) -> list[str]:
    """What sits at the package root and put nothing in the archive, unasked.

    An entry nobody excluded and nobody named is one whose author probably
    expected it to travel, and saying nothing is what let a package ship a
    megabyte of build input for a year. Anything an ignore file names is
    silent -- naming it there is the author saying they know -- and so is a
    dot-entry, which never ships, and the lockfile, which is this project's
    own record rather than anything a consumer of the package resolves
    against.
    """
    from .project import LOCKFILE_NAME

    if not declared.manifested:
        return []
    shipped = {relative.partition("/")[0] for relative, _ in entries}
    missing: list[str] = []
    for path in sorted(root.iterdir()):
        name = path.name
        holds = path.is_dir() and not path.is_symlink()
        if name in shipped or name.startswith(".") or name == LOCKFILE_NAME:
            continue
        if _matched(patterns, name, holds):
            continue
        missing.append(f"{name}/" if holds else name)
    return missing


def _left_out_warning(package: str, missing: list[str]) -> FfrwdWarning:
    shown = ", ".join(missing[:_LEFT_OUT_SHOWN])
    rest = len(missing) - _LEFT_OUT_SHOWN
    written = shown if rest <= 0 else f"{shown} and {rest} more"
    return FfrwdWarning(
        code=WarningCode.NOT_SHIPPED,
        package=package,
        message=f"package '{package}' leaves {written} out of the archive",
        hint=f'name what should ship in "files"; name what should not in '
        f"{IGNORE_NAME}, which is also how this stops being said",
    )


def pack(root: Path, *, on_warning: OnWarning | None = None) -> bytes:
    """The package directory `root` as one gzipped tar, byte-for-byte reproducible.

    What ships is the manifest's closure -- the manifest, its lib and bin
    files, the modules its lib SQL declares, README.md and the licence -- plus
    whatever the manifest's ``files`` names. Nothing else: a package is what it
    declares, so a directory nobody thought about stays out of the archive
    rather than travelling because nobody remembered to exclude it. A
    dot-entry never ships, and ``.ffrwdignore`` and ``.gitignore`` at the root
    take back what ``files`` named; the closure ships regardless of either. A
    directory holding no manifest has nothing to declare an archive with, and
    packs the way it always did: everything the ignore rules leave.

    Sorted member order, zeroed mtimes, uid/gid 0 with empty owner names, fixed
    modes and a gzip header with no timestamp: the same content packs to the
    same bytes on any machine, which is what keeps a published version's digest
    stable.

    `on_warning` hears about a ``.gitignore`` line this subset does not read,
    and about what sits at the package root and ships nothing.

    Raises ``FfrwdError`` for a declared entry that is not a regular file or
    directory, for a manifest that does not read, and for an ``.ffrwdignore``
    line outside the grammar; OSError if the tree cannot be read.
    """
    declared = _declared(root)
    patterns = _patterns(root, declared.package, on_warning)
    entries = _entries(root, declared, patterns)
    if on_warning is not None:
        missing = _left_out(root, entries, declared, patterns)
        if missing:
            on_warning(_left_out_warning(declared.package, missing))
    raw = io.BytesIO()
    with gzip.GzipFile(
        filename="", mode="wb", compresslevel=9, fileobj=raw, mtime=0
    ) as compressed:
        with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive:
            for relative, path in entries:
                info = tarfile.TarInfo(relative)
                info.mtime = 0
                info.uid = 0
                info.gid = 0
                info.uname = ""
                info.gname = ""
                if path.is_dir():
                    info.type = tarfile.DIRTYPE
                    info.mode = _DIR_MODE
                    archive.addfile(info)
                    continue
                data = path.read_bytes()
                info.type = tarfile.REGTYPE
                info.mode = _FILE_MODE
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
    return raw.getvalue()


# --------------------------------------------------------------------------
# unpacking
# --------------------------------------------------------------------------


def _target(package: str, name: str, destination: Path) -> Path:
    """Where the member `name` writes to under `destination`, or a rejection.

    Absolute names, ``..`` anywhere, and separators the archive has no business
    using are refused by their shape; what survives is checked again against
    where it actually resolves.
    """
    written = PurePosixPath(name)
    shaped = bool(name) and not (
        name.startswith("/")
        or "\\" in name
        or ":" in name
        or ".." in written.parts
        or written.is_absolute()
    )
    if shaped:
        target = destination.joinpath(*written.parts)
        if _under(destination, target):
            return target
    raise _reject(
        f"package '{package}': its archive holds the member path {name!r}, which leaves "
        "the directory it would be extracted into",
        _ARCHIVE_HINT,
    )


def _under(destination: Path, target: Path) -> bool:
    """True when `target` resolves inside `destination`."""
    root = Path(os.path.realpath(destination))
    resolved = Path(os.path.realpath(target))
    return resolved != root and root in resolved.parents


def _extract(package: str, archive: bytes, destination: Path) -> None:
    """Write the members of `archive` under `destination`, refusing every other kind.

    One pass, checking as it goes: a hostile archive is abandoned at the member
    that crosses a cap rather than after it has been decompressed whole.
    """
    members = 0
    unpacked = 0
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as opened:
        for member in opened:
            members += 1
            if members > _MAX_MEMBERS:
                raise _reject(
                    f"package '{package}': its archive holds more than {_MAX_MEMBERS} members",
                    _ARCHIVE_HINT,
                )
            if not (member.isreg() or member.isdir()):
                raise _reject(
                    f"package '{package}': its archive holds {member.name!r}, which is neither "
                    "a regular file nor a directory",
                    _ARCHIVE_HINT,
                )
            target = _target(package, member.name, destination)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            unpacked += member.size
            if unpacked > _MAX_UNPACKED_BYTES:
                raise _reject(
                    f"package '{package}': its archive unpacks to more than "
                    f"{_MAX_UNPACKED_BYTES} bytes",
                    _ARCHIVE_HINT,
                )
            content = opened.extractfile(member)
            if content is None:  # pragma: no cover -- isreg() already ruled this out
                raise _reject(
                    f"package '{package}': its archive holds {member.name!r} with no content",
                    _ARCHIVE_HINT,
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, "wb") as handle:
                shutil.copyfileobj(content, handle)


def _staging_dir(parent: Path, prefix: str) -> Path:
    """A fresh directory under `parent`, holding the parent's own permissions.

    A plain mkdir rather than ``tempfile.mkdtemp``: on Windows mkdtemp writes
    its own access list, and one made from an elevated shell is owned by the
    Administrators group alone, so the user who ran the install cannot read
    the entry afterwards.
    """
    for _ in range(100):
        candidate = parent / f"{prefix}{secrets.token_hex(4)}"
        try:
            candidate.mkdir()
        except FileExistsError:
            continue
        return candidate
    raise FileExistsError(f"no free staging name under {parent}")


def unpack(package: str, archive: bytes, sha256: str) -> Path:
    """Verify `archive` against `sha256`, then write what it holds into the store.

    The bytes are hashed before any tar API sees them, so an archive that is
    not the one pinned is discarded unopened. What survives is extracted beside
    the destination and moved onto it, so an entry is complete or absent and
    never half of either; content already stored under that digest is that same
    content, so it is left alone.

    Returns the store directory the content now sits in. :func:`entry_path`
    names the same place for a lockfile entry.
    """
    _require_digest(package, sha256)
    found = hashlib.sha256(archive).hexdigest()
    if found != sha256:
        raise _reject(
            f"package '{package}': its archive hashes to {found}, and {sha256} was expected",
            "the download is not what was published; nothing was written",
        )
    destination = store_dir() / entry_path(sha256)
    if destination.is_dir():
        return destination
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = _staging_dir(destination.parent, f"{sha256[:16]}-")
    except OSError as err:
        raise _reject(
            f"package '{package}': the store could not be written: {err.strerror or err}",
            "check that the cache directory exists and is writable",
        ) from err
    try:
        try:
            _extract(package, archive, staging)
        except (EOFError, OSError, tarfile.TarError) as err:
            raise _reject(
                f"package '{package}': its archive could not be unpacked: {err}",
                _ARCHIVE_HINT,
            ) from err
        try:
            os.replace(staging, destination)
        except OSError as err:
            # Another process storing the same digest got there first, which is
            # the same content by definition; anything else is a rejection.
            if not destination.is_dir():
                raise _reject(
                    f"package '{package}': the store could not be written: {err.strerror or err}",
                    "check that the cache directory exists and is writable",
                ) from err
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return destination


# --------------------------------------------------------------------------
# reading
# --------------------------------------------------------------------------


def _directory(package: str, stored: str) -> Path:
    """The store path `stored` names, checked for shape and format version."""
    written = PurePosixPath(stored.replace("\\", "/"))
    parts = written.parts
    if written.is_absolute() or ".." in parts or not parts:
        raise _reject(
            f"package '{package}': the store path {stored!r} leaves the store",
            "a store path is relative to the store directory and stays under it",
        )
    if parts[0] != STORE_FORMAT:
        raise _reject(
            f"package '{package}': the store path {stored!r} was written by store "
            f"format {parts[0]!r}, and this ffrwd reads {STORE_FORMAT!r}",
            _REINSTALL_HINT,
        )
    return store_dir().joinpath(*parts)


def load(package: str, stored: str, sha256: str) -> Path:
    """The store directory for `stored`, checked to be there.

    Nothing is hashed here. `sha256` is what the archive was verified against
    when it was installed, and the tree it unpacked to is read as it stands.
    The two are still checked against each other: an entry naming one digest
    while pointing at another's content would read content nothing pinned.

    `package` names the package in every rejection: the reader of the message
    has a lockfile in front of them, not a path under a cache directory.
    """
    _require_digest(package, sha256)
    directory = _directory(package, stored)
    if stored != entry_path(sha256):
        raise _reject(
            f"package '{package}': the store path {stored!r} is not where content of "
            f"digest {sha256} lives",
            _REINSTALL_HINT,
        )
    try:
        present = directory.is_dir()
    except OSError as err:  # pragma: no cover -- a path the OS refuses to stat
        raise _reject(
            f"package '{package}': its store directory could not be read: {err.strerror or err}",
            _REINSTALL_HINT,
        ) from err
    if not present:
        raise _reject(
            f"package '{package}': its content is not in the store at {directory}",
            _REINSTALL_HINT,
        )
    return directory


# --------------------------------------------------------------------------
# the model cache
# --------------------------------------------------------------------------

# A pinned model is fetched from the hub and placed beside the module whose
# export loads it -- inside a package's own store entry, which is addressed by
# its ARCHIVE's digest. So a package that republishes without touching a model
# pin lands in a new entry, and the same file used to be pulled over the
# network again for it. Here the file is kept once, addressed by its OWN
# digest, and what sits beside each module is a link to it.
#
# Unlike a store entry, the cache is an OPTIMIZATION: a file kept here can
# always be fetched again, so a cache that cannot be read is a miss rather
# than a rejection. What it must never hold is a torn or unverified file under
# a final name. Everything written here is written to a temporary in the same
# directory, verified, and renamed onto its name -- the one step a reader can
# observe -- and a name already taken is success, since a digest names one
# content. Several processes, or several containers sharing one volume, may
# fetch the same model at once; each pays for its download and the cache ends
# holding one whole file either way.
#
# NOTHING EVICTS. There is no safe rule to evict by. An entry is addressed by
# a digest some manifest on this machine pins, and the lockfiles that pin it
# live in projects all over the disk -- there is no reference count to read,
# and a last-used time cannot tell "stale" from "a project not opened this
# month", whose model would then cost a gigabyte to fetch again. A cap would
# evict by size, which is exactly backwards: the largest entry is the one
# most worth keeping. Clearing it is the user's own removal of the directory,
# which costs a refetch -- and, where a module's model is a symlink into it,
# an install of that package to put the file back.

# The model cache's layout version, and the first component of every path in
# it. Separate from STORE_FORMAT -- the two hold different things, and a
# change to one has no business invalidating the other.
MODEL_FORMAT = "v1"

# Whether a symlink is worth trying where a hard link was refused. POSIX only:
# Windows needs a privilege for one that nothing here can assume. A seam, so
# a check can take either road on any machine.
_SYMLINKS = os.name == "posix"


def models_dir() -> Path:
    """The root every cached model file sits under."""
    return _cache_dir() / "models"


def model_entry_path(sha256: str) -> str:
    """Where a model file of this digest belongs, relative to :func:`models_dir`."""
    return f"{MODEL_FORMAT}/{sha256[:2]}/{sha256}"


def cached_model(sha256: str) -> Path | None:
    """The cached file of this digest, or None when nothing is cached under it.

    Nothing is hashed here, for the reason nothing is hashed reading the
    package store: the digest did its work before the file was given its
    name, and what carries that name is the user's own cache.
    """
    if _SHA256_RE.fullmatch(sha256) is None:
        return None
    path = models_dir() / model_entry_path(sha256)
    try:
        return path if path.is_file() else None
    except OSError:  # pragma: no cover -- a path the OS refuses to stat
        return None


def model_staging(sha256: str) -> Path | None:
    """The directory to write a model of this digest into before it is kept.

    The directory its final name is in, so keeping it is a rename and never a
    copy. None when the directory cannot be made, which leaves the caller to
    do without the cache.
    """
    if _SHA256_RE.fullmatch(sha256) is None:
        return None
    directory = (models_dir() / model_entry_path(sha256)).parent
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    return directory


def keep_model(temporary: Path, sha256: str) -> Path | None:
    """Give a VERIFIED temporary in :func:`model_staging` its final name in the cache.

    The caller hashed `temporary` as it wrote it; this does not read it
    again. A name already there is success -- it holds this digest's content
    by definition -- and `temporary` is discarded rather than renamed over it.
    Returns the cached path, or None when the rename failed and nothing holds
    the name.
    """
    final = models_dir() / model_entry_path(sha256)
    try:
        if final.is_file():
            _discard(temporary)
            return final
        os.replace(temporary, final)
    except OSError:
        _discard(temporary)
        # Another writer can take the name between the check and the rename;
        # on a platform that refuses to rename over it, that is still success.
        return final if final.is_file() else None
    return final


def cache_model(source: Path, sha256: str) -> Path | None:
    """Keep a file already verified where it sits, leaving `source` in place.

    For a model an earlier install put beside its module before there was a
    cache, hashed by the caller just now. A hard link costs nothing and needs
    no second look. A copy is new bytes on what may be a network volume, so
    it is hashed as it is written and kept only when it matches. Returns the
    cached path, or None when nothing could be kept.
    """
    existing = cached_model(sha256)
    if existing is not None:
        return existing
    staging = model_staging(sha256)
    if staging is None:
        return None
    temporary = _fresh(staging, f"{sha256[:16]}-")
    if temporary is None:
        return None
    try:
        try:
            os.link(source, temporary)
        except OSError:
            if _copied_digest(source, temporary) != sha256:
                _discard(temporary)
                return None
    except OSError:
        _discard(temporary)
        return None
    return keep_model(temporary, sha256)


def place_model(sha256: str, destination: Path, *, copy: bool = True) -> str | None:
    """Put the cached file of this digest at `destination`, saying how.

    Tried in order: a hard link, which costs nothing and survives the cache
    being cleared; on POSIX, a RELATIVE symlink, which costs nothing and
    survives the volume holding both being mounted somewhere else; a copy.
    Returns ``"hard link"``, ``"symlink"`` or ``"copy"``, or None when there is
    nothing cached or nothing could be written. Made beside `destination` and
    renamed onto it, so the file there is whole or absent -- a symlink is
    renamed like any other file, never followed.

    `copy=False` stops short of the copy and answers None instead. That is for
    replacing a file that is already whole with a link to the cached one: a
    link frees the duplicate's disk, and a copy would write a gigabyte to end
    exactly where it started.
    """
    source = cached_model(sha256)
    if source is None:
        return None
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    temporary = _fresh(destination.parent, f"{destination.name}-")
    if temporary is None:
        return None
    try:
        how = _made(source, temporary, copy=copy)
        if how is None:
            return None
        os.replace(temporary, destination)
    except OSError:
        _discard(temporary)
        return None
    return how


def _made(source: Path, temporary: Path, *, copy: bool = True) -> str | None:
    """Make `temporary` stand for `source` by the cheapest road the filesystem allows.

    None, with nothing written, when only a copy is left and `copy` is False.
    """
    try:
        os.link(source, temporary)
        return "hard link"
    except OSError:
        pass
    if _SYMLINKS:
        try:
            os.symlink(os.path.relpath(source, temporary.parent), temporary)
            return "symlink"
        except OSError:
            pass
    if not copy:
        return None
    shutil.copyfile(source, temporary)
    return "copy"


def _fresh(directory: Path, prefix: str) -> Path | None:
    """An unused temporary name in `directory`, with nothing at it yet."""
    try:
        handle, written = tempfile.mkstemp(dir=directory, prefix=prefix, suffix=".tmp")
    except OSError:
        return None
    os.close(handle)
    path = Path(written)
    _discard(path)
    return path


def _copied_digest(source: Path, target: Path) -> str:
    """Copy `source` to `target` block by block: the sha256 of what was written."""
    digest = hashlib.sha256()
    with open(source, "rb") as reading, open(target, "xb") as writing:
        for block in iter(lambda: reading.read(1024 * 1024), b""):
            digest.update(block)
            writing.write(block)
    return digest.hexdigest()


def _discard(path: Path) -> None:
    """Remove a temporary, whatever state it is in."""
    try:
        os.unlink(path)
    except OSError:
        pass
