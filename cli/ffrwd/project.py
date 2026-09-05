"""The project files -- ``ffrwd.json`` and ``ffrwd.lock`` -- and what a query may call.

A directory holding a ``ffrwd.json`` is a PROJECT, and a project is itself a
PACKAGE. A package is named ``<namespace>/<package>``, and that name is the
path a call writes: ``imbcmdth/audio`` is called as ``imbcmdth.audio``. The
manifest says what the package provides under that name::

    { "name": "imbcmdth/audio", "version": "1.0.0",
      "bin": { "volume": "recipes/volume.sql", "duck": "recipes/duck.sql" },
      "lib": "src/audio.sql",
      "dependencies": { "broadcast/tracks": "^1.2.0" } }

``lib`` and ``bin`` each accept a string, or a map, never both at once. A
string names one file, its member named for the package segment --
``imbcmdth/deband``'s ``lib`` calls as ``imbcmdth.deband(...)``,
``imbcmdth/clip``'s ``bin`` runs as ``imbcmdth.clip``. A map names several
members, one file per name, and there is no root-callable one:
``imbcmdth.audio.volume``, never ``imbcmdth.audio(...)``. ``lib`` is the
exports: each key an exported function name, its value the file defining it.
``bin`` is the recipes: whole runnable queries, one file per name. The two
are read by ROLE -- a lib file holds definitions and nothing else, a
recipe's file is a query -- and a manifest declaring neither is a consumer
project that only holds dependencies.

``dependencies`` keys are package NAMES -- ``<namespace>/<package>`` -- and the
values are the version range recorded for each; the range is recorded, never
solved. A call across packages is always written in full,
``<namespace>.<package>.<member>``, so a manifest's ``dependencies`` needs no
alias to bind.

Beside it, ``ffrwd.lock`` records what the project INSTALLED: one entry per
package VERSION -- several may share a name, since installing never removes
one version to make room for another -- each a registry entry pinning a
version and the sha256 of the archive its content was installed from. A
registry entry also carries its own ``dependencies``, package name to the
exact version ITS OWN install resolved, so a call written inside that package
binds to what it depends on, not what the project directly does. The
lockfile's own top-level ``dependencies`` is the same shape, one level up:
what the project itself directly installed. It is machine-owned -- installing
writes it, nobody hand-edits it.

Third, ``ffrwd.links`` -- machine-local, never committed. The machine-wide
one, beside the machine-wide lockfile, is where `ffrwd link` run in a
package's directory records name -> directory; the one beside a project's
lockfile records the names that project reads live, each resolved through
the machine-wide file. A linked package resolves its own calls through its
own lockfile, and the project around it never sees those pins. Older ffrwds
wrote directory paths -- into the lockfile itself, or into a project's links
file; those are still read, and the next write migrates them.

:func:`discover` builds the set a compile resolves in, from three layers, the
first claim on a name winning:

1. the local manifest's own package -- the project is a package,
2. the local links file, then the local lockfile -- what this project reads
   live, over what it installed,
3. the global links file, then the global lockfile -- the machine-wide pair.

All of it is OPTIONAL: none of the three found means no packages, and a query
compiles exactly as it did before this file existed.

This module reads and validates those two files and resolves a package to the
directory its files live in -- the project's own, a linked one, or one in
the content-addressed store (:mod:`ffrwd.store`). It does not parse SQL:
what those files DEFINE is :mod:`ffrwd.functions`' business, and keeping the
split that way is what lets ``functions.py`` import this module without a
cycle. In particular, "the named function exists in the named file" is checked
where the file is parsed, not here.

It also WRITES the two, since the reader owns what a valid one looks like:
:func:`write_manifest` for ``init``, :func:`write_lockfile` for everything
that records a package. Both replace the file in one step and pin LF endings,
and the lockfile writer decides the ``reproducible`` claim from the entries
themselves rather than trusting a caller to keep the two in step.

Every rejection is a `FfrwdError`, anchored on the line where the offending
key is written when there is one to point at.
"""

from __future__ import annotations

import difflib
import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath
from typing import Literal

from . import store
from .errors import ErrorCode, FfrwdError
from .macros import macro_names
from .parser import FILTER_NAMESPACE, MACRO_NAMESPACE
from .wasm import MODEL_SUFFIX

__all__ = [
    "CAPABILITIES",
    "LINKSFILE_NAME",
    "LOCKFILE_NAME",
    "MANIFEST_NAME",
    "OFFICIAL_NAMESPACE",
    "README_NAME",
    "RESERVED_NAMESPACES",
    "STATEMENT_KEYWORDS",
    "LinkEntry",
    "Lockfile",
    "ModelPin",
    "Package",
    "PackageSet",
    "RegistryEntry",
    "add_dependency",
    "discover",
    "entry_root",
    "find_lockfile",
    "find_manifest",
    "held_entry",
    "held_links",
    "is_namespace",
    "is_package_name",
    "is_recipe_name",
    "leaves_package",
    "link_refusal",
    "link_target",
    "link_written",
    "links_path",
    "lockfile_text",
    "name_refusal",
    "read_linksfile",
    "read_lockfile",
    "read_manifest",
    "registered_root",
    "stored_name",
    "stored_version",
    "under_package",
    "with_entry",
    "without_entry",
    "write_linksfile",
    "write_lockfile",
    "write_manifest",
]

MANIFEST_NAME = "ffrwd.json"
LOCKFILE_NAME = "ffrwd.lock"
# Beside the lockfile, machine-local: where `ffrwd link` records what this
# machine reads live -- name -> directory beside the machine-wide lockfile,
# names alone beside a project's. Local state on one machine means nothing on
# another, so it is a separate file, not for version control.
LINKSFILE_NAME = "ffrwd.links"

# A package's own prose, beside its manifest. It ships in the archive and is
# what the registry shows on the version's page.
README_NAME = "README.md"

# A namespace becomes a call qualifier, so it may not be one the dialect
# already answers for: `ffmpeg.<filter>` is resolved by lower, and `wasm` is
# held for the frei0r/wasm bridge.
WASM_NAMESPACE = "wasm"
RESERVED_NAMESPACES = frozenset({FILTER_NAMESPACE, WASM_NAMESPACE})

# ffrwd's own namespace, which packages MAY claim: it is where the official
# packages are published. It doubles as the macro qualifier, and the two never
# overlap -- `ffrwd.speed(...)` is two segments and always the macro,
# `ffrwd.tools.frames(...)` is three and always a package. A package named for
# a macro is refused, since its two-segment call form is the macro's.
OFFICIAL_NAMESPACE = MACRO_NAMESPACE

# Unquoted identifiers fold to lowercase, so a name a query can write without
# quoting is a lowercase plain identifier. Both halves of a package name are
# one, and so is an exported function name.
_IDENTIFIER_RE = re.compile(r"[a-z_][a-z0-9_]*")

# A package name: `<namespace>/<package>`, each half a plain identifier.
_NAME_RE = re.compile(rf"{_IDENTIFIER_RE.pattern}/{_IDENTIFIER_RE.pattern}")

# A recipe is typed as a command word, so its name is spelled like one: a
# lowercase letter, then lowercase letters, digits, '-' or '_'.
_RECIPE_NAME_RE = re.compile(r"[a-z][a-z0-9_-]*")

# The four words a statement can begin with. Text starting with one is read as
# SQL, always; anything else typed where a query goes may name a recipe
# instead. So a recipe named for one of them could never be run by name.
STATEMENT_KEYWORDS = ("select", "copy", "create", "with")

# Characters that make a written path a pattern rather than a file.
_GLOB_CHARACTERS = "*?["

_REQUIRED = ("name", "version")
_KNOWN = frozenset(
    {
        *_REQUIRED,
        "description",
        "lib",
        "bin",
        "dependencies",
        "keywords",
        "license",
        "homepage",
        "ffrwd",
        "models",
        "capabilities",
        "private",
        "test",
    }
)

# What a package may declare it needs the host to grant. Not a list of wasi
# imports: clocks, random and the rest are ambient, and a module having them
# says nothing. "http" is outbound HTTP requests, "nn" is model inference,
# "udp" is UDP sockets.
CAPABILITIES = ("http", "nn", "udp")

# A keyword is a short label the registry indexes. Both bounds keep a document
# out of the place a list of labels belongs.
_MAX_KEYWORDS = 16
_MAX_KEYWORD_LENGTH = 32

# "ffrwd" is the range of compiler versions the package declares it runs on.
# Recorded and shown, never solved -- so only its shape is checked, the way a
# dependency range's is: comparator terms, each a version after an optional
# operator, separated by spaces, commas or '||'.
_ENGINES_TERM = r"[<>=^~]{0,2}[0-9*][0-9A-Za-z.*+-]*"
_ENGINES_RE = re.compile(rf"{_ENGINES_TERM}(\s*(,|\|\|)?\s*{_ENGINES_TERM})*")
_MAX_ENGINES_LENGTH = 64

# "license" is what the package is published under. Shape only: SPDX's list is
# long and its expressions compose, so an identifier and an expression over
# identifiers are the same thing here -- one short line of printable text, with
# no allowlist behind it.
_LICENSE_RE = re.compile(r"[^\x00-\x1f\x7f]+")
_MAX_LICENSE_LENGTH = 64

# "homepage" is a project page for the package, shown on its listing. http(s)
# only; trimmed, and an empty result is the same as the key never having been
# written at all.
_MAX_HOMEPAGE_LENGTH = 300

# "test" is the command `ffrwd publish` runs instead of its own recipe-compile
# check, when the manifest declares one. One line, trimmed; empty is the same
# as the key never having been written at all -- the same as "homepage".
_MAX_TEST_LENGTH = 500

# One model file on the hub: a `<owner>/<name>` repository, one revision --
# a branch, a tag or a commit -- and one file inside it, each half checked
# before it is ever part of a URL.
_REPO_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*")
_REVISION_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_MODEL_DIGEST_RE = re.compile(r"[0-9a-f]{64}")
_MODEL_KEYS = ("repo", "revision", "file", "sha256")

# "not_on" is optional, so it sits apart from the four every pin must give.
_MODEL_OPTIONAL_KEYS = ("not_on",)

# The execution providers a model's pin may deny itself, spelled the way the
# sidecar's own ``-nn-exclude`` takes them. "cpu" is not among them: every
# model runs somewhere, and CPU is the one provider ONNX Runtime always
# offers.
EXCLUDABLE_PROVIDERS = ("coreml", "cuda", "directml")

# Keys older manifests wrote, each with the hint that says what replaced it.
_RETIRED = {
    "namespace": 'the name carries the namespace: "name" is "<namespace>/<package>"',
    "exports": '"lib" replaced it: a string or a map of exported function name to file',
    "libs": '"lib" replaced it: a string names one file, a map names exported function '
    "name to file",
    "bins": '"bin" replaced it: a string names one file, a map names recipe name to file',
}

_NAMESPACE_HINT = (
    "a namespace is a lowercase plain identifier: a letter or underscore, then "
    "letters, digits or underscores"
)
_NAME_HINT = (
    'a package name is "<namespace>/<package>", each half a lowercase plain identifier'
)
_MANIFEST_HINT = (
    'a manifest is one JSON object with "name" and "version"; "lib" and "bin" -- each a '
    "string or a map -- are what the package provides, both optional"
)
_LIB_HINT = (
    '"lib" is a string naming one file, or a map of exported function name to file, '
    'relative to the manifest, e.g. {"quieter": "src/audio.sql"}'
)
_BIN_HINT = (
    '"bin" is a string naming one file, or a map of recipe name to one query file, '
    'relative to the manifest, e.g. {"duck": "recipes/duck.sql"}'
)
_DEPENDENCIES_HINT = (
    'dependencies maps "<namespace>/<package>" to a version range, e.g. '
    '{"broadcast/tracks": "^1.2.0"}'
)
_RECIPE_NAME_HINT = (
    "a recipe name is a command word: a lowercase letter, then lowercase "
    "letters, digits, '-' or '_'"
)
_KEYWORDS_HINT = (
    f'"keywords" is a list of at most {_MAX_KEYWORDS} short labels, each a '
    f'non-empty string of at most {_MAX_KEYWORD_LENGTH} characters, e.g. '
    '["audio", "loudness"]'
)
_ENGINES_HINT = (
    '"ffrwd" is the range of ffrwd versions this package runs on, written as a '
    'string, e.g. ">=0.9"'
)
_LICENSE_HINT = (
    '"license" is what this package is published under, written as an SPDX '
    'identifier or an expression over them, e.g. "MIT" or "MIT OR Apache-2.0"'
)
_HOMEPAGE_HINT = (
    f'"homepage" is a project page, an http:// or https:// URL of at most '
    f"{_MAX_HOMEPAGE_LENGTH} characters"
)
_CAPABILITIES_HINT = (
    '"capabilities" is a list of what the host must grant this package\'s '
    f"modules, each one of {', '.join(CAPABILITIES)}, e.g. [\"nn\"]"
)
_PRIVATE_HINT = '"private" is true or false; true publishes each version private'
_MODELS_HINT = (
    '"models" maps an exported name to the model file it loads: '
    '{"depth": {"repo": "<owner>/<name>", "revision": "<branch, tag or commit>", '
    '"file": "model.onnx", "sha256": "<64 hex>"}}; a model of several files is a '
    "list of those, the graph first"
)
_NOT_ON_HINT = (
    '"not_on" is a list of execution providers this model refuses, from '
    f"{', '.join(EXCLUDABLE_PROVIDERS)}"
)
_TEST_HINT = (
    '"test" is one line: the command `ffrwd publish` runs instead of compiling the recipes'
)


def is_namespace(text: str) -> bool:
    """True when `text` is spelled like a namespace -- a lowercase plain identifier.

    Shape only: whether the namespace is one ffrwd keeps for itself is
    `RESERVED_NAMESPACES`' answer, and a caller deriving a namespace wants the
    two apart to say which of them went wrong.
    """
    return _IDENTIFIER_RE.fullmatch(text) is not None


def is_package_name(text: str) -> bool:
    """True when `text` is spelled like a package name -- ``<namespace>/<package>``.

    Shape only, like :func:`is_namespace`: whether the name is one ffrwd keeps
    for itself is :func:`name_refusal`'s question, with its own message.
    """
    return _NAME_RE.fullmatch(text) is not None


def is_recipe_name(text: str) -> bool:
    """True when `text` is spelled like a recipe name -- a command word."""
    return _RECIPE_NAME_RE.fullmatch(text) is not None


def name_refusal(name: str) -> tuple[str, str] | None:
    """Why no package may be called `name`, as ``(message, hint)``, or None if one may.

    Two refusals, and the caller prefixes whatever context it has. A reserved
    namespace is one the dialect answers for itself. A package under the
    official namespace named for a macro is refused because the two-segment
    call the name earns -- ``ffrwd.speed(...)`` -- is always the macro, so no
    caller could ever reach the package that way.
    """
    namespace, _, segment = name.partition("/")
    if namespace in RESERVED_NAMESPACES:
        reserved = ", ".join(sorted(RESERVED_NAMESPACES))
        return (
            f"namespace {namespace!r} is reserved",
            f"{reserved} belong to ffrwd itself; pick another namespace",
        )
    if namespace == OFFICIAL_NAMESPACE and segment in macro_names():
        return (
            f"package {name!r} is named for the macro "
            f"{OFFICIAL_NAMESPACE}.{segment}()",
            f"{OFFICIAL_NAMESPACE}.{segment}(...) is two segments, which is always "
            "the macro; rename the package",
        )
    return None


# Which of the three layers a package was found in. Factual, not a judgment:
# what to say about landing on "global" is the compiler's call, since only it
# knows the call site.
Layer = Literal["project", "local", "global"]


@dataclass(frozen=True)
class ModelPin:
    """One model file an exported wasm function loads, pinned exactly.

    `repo` and `revision` name it on the hub, `file` names it inside that
    revision, and `sha256` is what the bytes must hash to. Install fetches it
    beside the module the export belongs to; the model is not in the archive,
    which is what keeps a package that runs one down to kilobytes.

    `not_on` is the execution providers this model's own graph cannot run on
    -- measured, not guessed, and carried on the graph's own pin since the
    constraint belongs to the model, not the module hosting it. Empty for a
    model that runs anywhere. A model pinned as several files carries it on
    the first entry only, the graph itself; the files after it are inputs
    that graph reads, not their own model.
    """

    repo: str
    revision: str
    file: str
    sha256: str
    not_on: tuple[str, ...] = ()

    @property
    def filename(self) -> str:
        """The last segment of `file`: the name a fetch of it lands under.

        A path inside a revision is written POSIX-style, so only "/" divides it.
        """
        return self.file.rpartition("/")[2]


@dataclass(frozen=True)
class Package:
    """One package: its name, and what it provides under it.

    `name` is ``<namespace>/<package>``; the two halves are derived, never
    stored twice. `exports` maps each exported function name to the file
    defining it: a string ``lib``'s one entry, named for the package segment,
    or a map's several. `recipes` does the same for the runnable queries,
    from ``bin``. Several names may share one file. `dependencies` is the
    manifest's own ``dependencies``: each key the depended-on package's name,
    each value the version range as written -- recorded and shown, never
    solved.

    `keywords` are the labels the registry indexes it under, `license` what
    the package is published under (None when it says nothing), `homepage` a
    project page for it (None the same way), `capabilities` what the manifest
    declares its modules need the host to grant, `engines` the range of ffrwd
    versions the manifest's ``ffrwd`` key declares (None when it declares
    none), `models` the files each named export's model is, keyed by export
    name and led by the graph itself, and `private` whether a publish stamps
    the version private. All seven are the registry's business rather than a
    compile's: they are read, validated and carried, and nothing here acts on
    them.

    `test` is the command the manifest's ``test`` key declares, empty when it
    declares none: what ``ffrwd publish`` runs instead of its own
    recipe-compile check, on `Package`'s own author's word, unrun here.

    `linked` marks a package read straight out of a working directory rather
    than out of the store. Its files are whatever they are right now, so no
    digest pins them and no lockfile makes a build using it reproducible.
    """

    name: str
    version: str
    root: Path
    manifest: Path
    exports: Mapping[str, Path] = field(default_factory=dict)
    recipes: Mapping[str, Path] = field(default_factory=dict)
    dependencies: Mapping[str, str] = field(default_factory=dict)
    keywords: tuple[str, ...] = ()
    license: str | None = None
    homepage: str | None = None
    capabilities: tuple[str, ...] = ()
    engines: str | None = None
    models: Mapping[str, tuple[ModelPin, ...]] = field(default_factory=dict)
    private: bool = False
    test: str = ""
    # Set only where the manifest wrote a plain string: the one member that
    # answers to the package's own name.
    root_export: str | None = None
    root_recipe: str | None = None
    layer: Layer = "project"
    linked: bool = False

    @property
    def namespace(self) -> str:
        """The first segment of the name: what qualifies a call."""
        return self.name.partition("/")[0]

    @property
    def package(self) -> str:
        """The second segment of the name: the default export's and recipe's name."""
        return self.name.partition("/")[2]

    def export(self, member: str | None = None) -> Path | None:
        """The file exporting `member`, or the root-callable export's for None.

        Only the string form of ``lib`` has a root-callable export, and
        `root_export` is set when it does. A map entry that happens to be
        named for the package is reached at its own name like any other.
        """
        if member is None:
            member = self.root_export
        return None if member is None else self.exports.get(member)

    def recipe(self, member: str | None = None) -> Path | None:
        """The file of recipe `member`, or the root-callable recipe's for None."""
        if member is None:
            member = self.root_recipe
        return None if member is None else self.recipes.get(member)


@dataclass(frozen=True)
class PackageSet:
    """The packages a compile may resolve a call in, keyed by name.

    `packages` is one CANONICAL `Package` per name -- the first layer that
    claimed it -- for listing and for a fallback resolution. `versions` is
    every installed version of every name, since install never makes two
    versions of one package fight over the name: a project depending on two
    packages that each depend on a different version of a third package pins
    both, and each keeps calling into its own.

    `wants` answers the question `versions` alone cannot: which of those
    versions a given DEPENDENT package itself resolved to, when its own
    dependencies were installed. Keyed by the dependent's own name --
    `project` for the top-level script and the project's own `ffrwd.json`
    -- to a map of dependency name to the exact version that dependent binds
    it to. See :meth:`resolve`.

    `in_project` is True when the query sits inside a project -- a manifest or
    a lockfile was found above it. It is what makes landing on the global
    layer worth warning about: outside a project, a globally installed package
    is the only thing there is to resolve against.

    `manifest` is the ``ffrwd.json`` the walk found, or None when it found
    none; `start` is the directory the walk began from. A rejection that
    needs to say what discovery consulted reads both rather than guessing
    from `root`, which is populated from a lockfile or the start directory
    itself when there is no manifest.
    """

    root: Path
    packages: dict[str, Package] = field(default_factory=dict)
    versions: dict[str, dict[str, Package]] = field(default_factory=dict)
    wants: dict[str, dict[str, str]] = field(default_factory=dict)
    project: str | None = None
    in_project: bool = True
    manifest: Path | None = None
    start: Path | None = None

    def get(self, name: str) -> Package | None:
        """The canonical package `name` names, by its full ``<namespace>/<package>``."""
        return self.packages.get(name)

    def find(self, namespace: str, package: str) -> Package | None:
        """The canonical package the two segments name, or None."""
        return self.packages.get(f"{namespace}/{package}")

    def resolve(self, dependent: str | None, name: str) -> Package | None:
        """The package `name` names, at the version `dependent` itself depends on it at.

        `dependent` is a package's own name -- whose source a call is written
        in -- or None/empty for the top-level script, which resolves as the
        project itself. Falls back to the canonical entry when nothing
        recorded a binding: a linked package (not walked by `install`), or a
        lockfile with no project above it to hold one.
        """
        who = dependent or self.project
        version = self.wants.get(who, {}).get(name) if who is not None else None
        if version is not None:
            found = self.versions.get(name, {}).get(version)
            if found is not None:
                return found
        return self.get(name)

    def in_namespace(self, namespace: str) -> tuple[Package, ...]:
        """Every canonical package under `namespace`, in name order."""
        return tuple(
            self.packages[name]
            for name in sorted(self.packages)
            if self.packages[name].namespace == namespace
        )

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self.packages))

    def namespaces(self) -> tuple[str, ...]:
        return tuple(sorted({package.namespace for package in self.packages.values()}))


# -- rejections ------------------------------------------------------------


def _reject(
    path: Path,
    message: str,
    *,
    line: int | None = None,
    col: int | None = None,
    hint: str | None = None,
) -> FfrwdError:
    """A manifest rejection, naming the file and the line when there is one."""
    return FfrwdError(
        ErrorCode.UNSUPPORTED_SQL,
        f"{path}: {message}",
        line=line,
        col=col if line is not None else None,
        hint=hint,
    )


def _key_line(text: str, key: str) -> int | None:
    """The line `key` is written on, so a rejection about it can point there."""
    needle = f'"{key}"'
    for number, line in enumerate(text.splitlines(), start=1):
        if needle in line:
            return number
    return None


def _did_you_mean(name: str, candidates: list[str]) -> str | None:
    matches = difflib.get_close_matches(name, candidates, n=1, cutoff=0.6)
    return f"did you mean {matches[0]!r}?" if matches else None


# -- reading one manifest --------------------------------------------------


class _Object(dict[str, object]):
    """A parsed JSON object that remembers the keys its text wrote twice.

    ``json`` keeps the last of two same-named keys and says nothing, which
    would silently drop one of two recipes written under one name.
    """

    repeated: tuple[str, ...] = ()


def _as_object(pairs: list[tuple[str, object]]) -> _Object:
    """The object-parsing hook: build the dict, and note every repeated key."""
    seen: set[str] = set()
    repeated: list[str] = []
    for key, _value in pairs:
        if key in seen and key not in repeated:
            repeated.append(key)
        seen.add(key)
    parsed = _Object(pairs)
    parsed.repeated = tuple(repeated)
    return parsed


def _text_field(
    data: dict[str, object], key: str, path: Path, text: str, *, hint: str = _MANIFEST_HINT
) -> str:
    value = data[key]
    if not isinstance(value, str) or not value.strip():
        raise _reject(
            path,
            f'"{key}" must be a non-empty string',
            line=_key_line(text, key),
            hint=hint,
        )
    return value


def _package_name(data: dict[str, object], path: Path, text: str) -> str:
    """The name `data` declares, validated: the shape, and what ffrwd keeps for itself."""
    line = _key_line(text, "name")
    name = _text_field(data, "name", path, text)
    if not is_package_name(name):
        raise _reject(
            path, f"name {name!r} is not a package name", line=line, hint=_NAME_HINT
        )
    refused = name_refusal(name)
    if refused is not None:
        raise _reject(path, refused[0], line=line, hint=refused[1])
    return name


def leaves_package(written: str) -> bool:
    """True for a written path that names something outside the manifest's directory.

    Absolute, drive-qualified, or reaching up through ``..``: a package's own
    files -- lib, bin and the modules a lib names -- ship inside it.
    """
    relative = PurePosixPath(written.replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts:
        return True
    return re.match(r"[A-Za-z]:", written) is not None


def under_package(root: Path, written: str) -> Path:
    """The file `written` names, read against a package `root`.

    A manifest and a lib file both write their paths POSIX-style, whatever
    platform reads them, so the separators are normalized before the path is
    joined. Says nothing about whether the result stays under `root` --
    :func:`leaves_package` is that question -- or whether it is there.
    """
    return root / Path(*PurePosixPath(written.replace("\\", "/")).parts)


def _file_value(
    written: object, owner: str, root: Path, path: Path, line: int | None, hint: str
) -> Path:
    """The one file `owner` names, validated: one path, under the project, and there."""
    if not isinstance(written, str) or not written.strip():
        raise _reject(path, f"{owner} must name one file", line=line, hint=hint)
    if leaves_package(written):
        raise _reject(
            path,
            f"{owner} points at {written!r}, which leaves the project directory",
            line=line,
            hint="the file is relative to the manifest and stays under it",
        )
    if any(character in written for character in _GLOB_CHARACTERS):
        raise _reject(
            path,
            f"{owner} names the pattern {written!r}, not a file",
            line=line,
            hint="one name, one file",
        )
    file = under_package(root, written)
    try:
        present = file.is_file()
    except (OSError, ValueError) as err:
        raise _reject(
            path,
            f"{owner}: {written!r} could not be read: {err}",
            line=line,
            hint=hint,
        ) from err
    if not present:
        raise _reject(
            path,
            f"{owner} names no file: {written!r}",
            line=line,
            hint=f"relative to {root}; check the path and the extension",
        )
    return file


def _map_of(data: dict[str, object], key: str, path: Path, text: str, hint: str) -> _Object:
    """The JSON object `key` holds, its repeated keys rejected."""
    at = _key_line(text, key)
    value = data[key]
    if not isinstance(value, dict):
        raise _reject(path, f'"{key}" must be a JSON object', line=at, hint=hint)
    if isinstance(value, _Object) and value.repeated:
        repeated = value.repeated[0]
        raise _reject(
            path,
            f"{key} declares {repeated!r} twice",
            line=_key_line(text, repeated) or at,
            hint="one name, one entry; keep the one you meant",
        )
    result = _Object(value)
    return result


def _string_or_map(data: dict[str, object], key: str, path: Path, text: str, hint: str) -> _Object:
    """The JSON object `key` holds, once its string form is ruled out; repeated keys rejected."""
    at = _key_line(text, key)
    value = data[key]
    if not isinstance(value, dict):
        raise _reject(path, f'"{key}" must be a string or a JSON object', line=at, hint=hint)
    if isinstance(value, _Object) and value.repeated:
        repeated = value.repeated[0]
        raise _reject(
            path,
            f"{key} declares {repeated!r} twice",
            line=_key_line(text, repeated) or at,
            hint="one name, one entry; keep the one you meant",
        )
    return _Object(value)


def _exports(
    data: dict[str, object], segment: str, root: Path, path: Path, text: str
) -> dict[str, Path]:
    """The export map ``lib`` declares: name to file. A string is one export named for `segment`."""
    if "lib" not in data:
        return {}
    at = _key_line(text, "lib")
    if isinstance(data["lib"], str):
        return {segment: _file_value(data["lib"], '"lib"', root, path, at, _MANIFEST_HINT)}
    exports: dict[str, Path] = {}
    for name, written in _string_or_map(data, "lib", path, text, _LIB_HINT).items():
        line = _key_line(text, name) or at
        if _IDENTIFIER_RE.fullmatch(name) is None:
            raise _reject(
                path,
                f"export name {name!r} is not a plain identifier",
                line=line,
                hint="an exported function name is a lowercase plain identifier",
            )
        exports[name] = _file_value(written, f"export '{name}'", root, path, line, _LIB_HINT)
    return exports


def _recipes(
    data: dict[str, object], segment: str, root: Path, path: Path, text: str
) -> dict[str, Path]:
    """The recipe map ``bin`` declares: name to file. A string is one, named for `segment`."""
    if "bin" not in data:
        return {}
    at = _key_line(text, "bin")
    if isinstance(data["bin"], str):
        return {segment: _file_value(data["bin"], '"bin"', root, path, at, _MANIFEST_HINT)}
    recipes: dict[str, Path] = {}
    for name, written in _string_or_map(data, "bin", path, text, _BIN_HINT).items():
        line = _key_line(text, name) or at
        if _RECIPE_NAME_RE.fullmatch(name) is None:
            raise _reject(
                path,
                f"recipe name {name!r} is not a command name",
                line=line,
                hint=_RECIPE_NAME_HINT,
            )
        if name in STATEMENT_KEYWORDS:
            raise _reject(
                path,
                f"recipe name {name!r} is a word a query begins with",
                line=line,
                hint=f"text beginning with {', '.join(STATEMENT_KEYWORDS)} is read as SQL, "
                "so a recipe of that name could never be run by name; rename it",
            )
        recipes[name] = _file_value(written, f"recipe '{name}'", root, path, line, _BIN_HINT)
    return recipes


def _dependencies(data: dict[str, object], path: Path, text: str) -> dict[str, str]:
    """The dependency map: package name to version range, empty when there is none."""
    if "dependencies" not in data:
        return {}
    at = _key_line(text, "dependencies")
    dependencies: dict[str, str] = {}
    for name, written in _map_of(data, "dependencies", path, text, _DEPENDENCIES_HINT).items():
        line = _key_line(text, name) or at
        if not is_package_name(name):
            raise _reject(
                path,
                f"dependency key {name!r} is not a package name",
                line=line,
                hint=_NAME_HINT,
            )
        refused = name_refusal(name)
        if refused is not None:
            raise _reject(path, f"dependency {name!r}: {refused[0]}", line=line, hint=refused[1])
        if not isinstance(written, str) or not written.strip():
            raise _reject(
                path,
                f"dependency {name!r} must be a string",
                line=line,
                hint=_DEPENDENCIES_HINT,
            )
        dependencies[name] = written
    return dependencies


def _keywords(data: dict[str, object], path: Path, text: str) -> tuple[str, ...]:
    """The labels ``keywords`` declares, empty when it declares none."""
    if "keywords" not in data:
        return ()
    at = _key_line(text, "keywords")
    written = data["keywords"]
    if not isinstance(written, list):
        raise _reject(path, '"keywords" must be a list', line=at, hint=_KEYWORDS_HINT)
    if len(written) > _MAX_KEYWORDS:
        raise _reject(
            path,
            f'"keywords" declares {len(written)}, and at most {_MAX_KEYWORDS} are read',
            line=at,
            hint=_KEYWORDS_HINT,
        )
    found: list[str] = []
    for keyword in written:
        if not isinstance(keyword, str) or not keyword.strip():
            raise _reject(
                path, f"keyword {keyword!r} is not a label", line=at, hint=_KEYWORDS_HINT
            )
        if len(keyword) > _MAX_KEYWORD_LENGTH:
            raise _reject(
                path,
                f"keyword {keyword!r} is longer than {_MAX_KEYWORD_LENGTH} characters",
                line=at,
                hint=_KEYWORDS_HINT,
            )
        found.append(keyword)
    return tuple(found)


def _capabilities(data: dict[str, object], path: Path, text: str) -> tuple[str, ...]:
    """What ``capabilities`` declares, in name order; empty when it declares none.

    Absent and empty are the same claim: this package's modules need nothing
    granted. A name outside the vocabulary is a rejection rather than a
    capability the host silently never grants.
    """
    if "capabilities" not in data:
        return ()
    at = _key_line(text, "capabilities")
    written = data["capabilities"]
    if not isinstance(written, list):
        raise _reject(path, '"capabilities" must be a list', line=at, hint=_CAPABILITIES_HINT)
    found: set[str] = set()
    for capability in written:
        if not isinstance(capability, str) or capability not in CAPABILITIES:
            raise _reject(
                path,
                f"capability {capability!r} is not one this ffrwd grants",
                line=at,
                hint=f"the capabilities are {', '.join(CAPABILITIES)}",
            )
        found.add(capability)
    return tuple(sorted(found))


def _private(data: dict[str, object], path: Path, text: str) -> bool:
    """Whether ``private`` declares the package's versions publish private."""
    if "private" not in data:
        return False
    written = data["private"]
    if not isinstance(written, bool):
        raise _reject(
            path,
            '"private" must be true or false',
            line=_key_line(text, "private"),
            hint=_PRIVATE_HINT,
        )
    return written


def _license(data: dict[str, object], path: Path, text: str) -> str | None:
    """What ``license`` declares. Shape only: no list of licenses is consulted."""
    if "license" not in data:
        return None
    at = _key_line(text, "license")
    written = data["license"]
    if not isinstance(written, str) or not written.strip():
        raise _reject(path, '"license" must be a non-empty string', line=at, hint=_LICENSE_HINT)
    if len(written) > _MAX_LICENSE_LENGTH:
        raise _reject(
            path,
            f'"license" is longer than {_MAX_LICENSE_LENGTH} characters',
            line=at,
            hint=_LICENSE_HINT,
        )
    if _LICENSE_RE.fullmatch(written) is None:
        raise _reject(
            path, f"license {written!r} is not one line of text", line=at, hint=_LICENSE_HINT
        )
    return written


def _homepage(data: dict[str, object], path: Path, text: str) -> str | None:
    """The project page ``homepage`` names. Trimmed; empty after trimming
    reads as though the key were never written."""
    if "homepage" not in data:
        return None
    at = _key_line(text, "homepage")
    written = data["homepage"]
    if not isinstance(written, str):
        raise _reject(path, '"homepage" must be a string', line=at, hint=_HOMEPAGE_HINT)
    trimmed = written.strip()
    if not trimmed:
        return None
    if len(trimmed) > _MAX_HOMEPAGE_LENGTH:
        raise _reject(
            path,
            f'"homepage" is longer than {_MAX_HOMEPAGE_LENGTH} characters',
            line=at,
            hint=_HOMEPAGE_HINT,
        )
    if not trimmed.startswith(("http://", "https://")):
        raise _reject(
            path,
            f'"homepage" {trimmed!r} must start with http:// or https://',
            line=at,
            hint=_HOMEPAGE_HINT,
        )
    return trimmed


def _engines(data: dict[str, object], path: Path, text: str) -> str | None:
    """The ffrwd version range ``ffrwd`` declares. Shape only; nothing solves it."""
    if "ffrwd" not in data:
        return None
    at = _key_line(text, "ffrwd")
    written = data["ffrwd"]
    if not isinstance(written, str) or not written.strip():
        raise _reject(path, '"ffrwd" must be a non-empty string', line=at, hint=_ENGINES_HINT)
    if len(written) > _MAX_ENGINES_LENGTH:
        raise _reject(
            path,
            f'"ffrwd" is longer than {_MAX_ENGINES_LENGTH} characters',
            line=at,
            hint=_ENGINES_HINT,
        )
    if _ENGINES_RE.fullmatch(written) is None:
        raise _reject(
            path, f"ffrwd {written!r} is not a version range", line=at, hint=_ENGINES_HINT
        )
    return written


def _not_on(
    written: dict[str, object], named: str, path: Path, line: int | None
) -> tuple[str, ...]:
    """The providers ``not_on`` denies, validated against what the sidecar knows."""
    if "not_on" not in written:
        return ()
    value = written["not_on"]
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise _reject(
            path,
            f'{named} must give "not_on" as a list of provider names',
            line=line,
            hint=_NOT_ON_HINT,
        )
    seen: list[str] = []
    for provider in value:
        if provider not in EXCLUDABLE_PROVIDERS:
            raise _reject(
                path,
                f'{named} names an unknown provider {provider!r} in "not_on"',
                line=line,
                hint=_did_you_mean(provider, list(EXCLUDABLE_PROVIDERS)) or _NOT_ON_HINT,
            )
        if provider not in seen:
            seen.append(provider)
    return tuple(seen)


def _model(written: object, named: str, path: Path, line: int | None) -> ModelPin:
    """One pin, every field checked before it is part of a URL."""
    if not isinstance(written, dict):
        raise _reject(path, f"{named} must be a JSON object", line=line, hint=_MODELS_HINT)
    for key in sorted(written):
        if key not in _MODEL_KEYS and key not in _MODEL_OPTIONAL_KEYS:
            raise _reject(
                path,
                f"unknown key {key!r} in {named}",
                line=line,
                hint=_did_you_mean(key, list(_MODEL_KEYS) + list(_MODEL_OPTIONAL_KEYS))
                or f"a model holds: {', '.join(_MODEL_KEYS)}",
            )
    for key in _MODEL_KEYS:
        value = written.get(key)
        if not isinstance(value, str) or not value.strip():
            raise _reject(
                path,
                f'{named} must name a non-empty "{key}"',
                line=line,
                hint=_MODELS_HINT,
            )
    repo, revision = str(written["repo"]), str(written["revision"])
    file, sha256 = str(written["file"]), str(written["sha256"])
    if _REPO_RE.fullmatch(repo) is None:
        raise _reject(
            path,
            f"{named} names the repository {repo!r}, which is not <owner>/<name>",
            line=line,
            hint=_MODELS_HINT,
        )
    if _REVISION_RE.fullmatch(revision) is None:
        raise _reject(
            path,
            f"{named} names the revision {revision!r}, which is not a branch, tag or commit",
            line=line,
            hint=_MODELS_HINT,
        )
    if leaves_package(file) or any(character in file for character in _GLOB_CHARACTERS):
        raise _reject(
            path,
            f"{named} names the file {file!r}, which is not a file inside the revision",
            line=line,
            hint="the file is written relative to the repository root, e.g. "
            '"onnx/model.onnx"',
        )
    if _MODEL_DIGEST_RE.fullmatch(sha256) is None:
        raise _reject(
            path,
            f"{named}: {sha256!r} is not a sha256 digest",
            line=line,
            hint="a digest is 64 lowercase hex characters",
        )
    not_on = _not_on(written, named, path, line)
    return ModelPin(repo=repo, revision=revision, file=file, sha256=sha256, not_on=not_on)


def _landings(
    pins: tuple[ModelPin, ...], export: str, named: str, path: Path, line: int | None
) -> None:
    """Check where the pins after the first land: install writes each under its own
    file name, beside the graph the first one becomes."""
    graph = f"{export}{MODEL_SUFFIX}"
    seen: dict[str, int] = {}
    for number, pin in enumerate(pins[1:], start=2):
        if pin.not_on:
            raise _reject(
                path,
                f'{named} entry {number} names "not_on"',
                line=line,
                hint="a provider constraint belongs to the model, not one of its "
                "files; put it on the first entry, the graph the export loads",
            )
        landing = pin.filename
        if not landing or landing in (".", "..") or "\\" in landing:
            raise _reject(
                path,
                f"{named} entry {number} names {pin.file!r}, which ends in no plain "
                "filename",
                line=line,
                hint="every entry after the first lands under the last segment of what "
                "it names, so that segment is a plain filename",
            )
        if landing == graph:
            raise _reject(
                path,
                f"{named} entry {number} would land under {graph!r}, where the first "
                "entry lands",
                line=line,
                hint=f"install writes the first entry as {graph!r} and every later one "
                "under its own name; rename it, or put it first",
            )
        if landing in seen:
            raise _reject(
                path,
                f"{named} entries {seen[landing]} and {number} would both land under "
                f"{landing!r}",
                line=line,
                hint="each entry after the first lands under its own file name, so two "
                "of a name would overwrite each other",
            )
        seen[landing] = number


def _model_pins(
    written: object, export: str, path: Path, line: int | None
) -> tuple[ModelPin, ...]:
    """The pins one ``models`` entry holds: one file, or a list of them.

    The first is the graph the export loads, whatever it is named on the hub;
    every later one is a file that graph refers to by name, so it lands under
    its own.
    """
    named = f"model '{export}'"
    if not isinstance(written, list):
        return (_model(written, named, path, line),)
    if not written:
        raise _reject(path, f"{named} lists no file", line=line, hint=_MODELS_HINT)
    pins = tuple(
        _model(entry, f"{named} entry {number}", path, line)
        for number, entry in enumerate(written, start=1)
    )
    _landings(pins, export, named, path, line)
    return pins


def _models(data: dict[str, object], path: Path, text: str) -> dict[str, tuple[ModelPin, ...]]:
    """The files each named export loads, empty when the manifest names none.

    The key is the export name the module's file is named for, which is what
    install writes the first of them as beside the module. Whether the package
    actually declares that export is checked where its lib files are read.
    """
    if "models" not in data:
        return {}
    at = _key_line(text, "models")
    models: dict[str, tuple[ModelPin, ...]] = {}
    for export, written in _map_of(data, "models", path, text, _MODELS_HINT).items():
        line = _key_line(text, export) or at
        if _IDENTIFIER_RE.fullmatch(export) is None:
            raise _reject(
                path,
                f"model name {export!r} is not a plain identifier",
                line=line,
                hint="a model is keyed by the exported function name that loads it",
            )
        models[export] = _model_pins(written, export, path, line)
    return models


def _test(data: dict[str, object], path: Path, text: str) -> str:
    """The command ``test`` declares, empty when the manifest declares none.

    Trimmed; empty after trimming reads as though the key were never written,
    the same as ``homepage``. One line: a value spanning several is refused,
    since a shell run against it would take only the first.
    """
    if "test" not in data:
        return ""
    at = _key_line(text, "test")
    written = data["test"]
    if not isinstance(written, str):
        raise _reject(path, '"test" must be a string', line=at, hint=_TEST_HINT)
    trimmed = written.strip()
    if not trimmed:
        return ""
    if len(trimmed) > _MAX_TEST_LENGTH:
        raise _reject(
            path,
            f'"test" is longer than {_MAX_TEST_LENGTH} characters',
            line=at,
            hint=_TEST_HINT,
        )
    if "\n" in trimmed or "\r" in trimmed:
        raise _reject(
            path,
            f'"test" {trimmed!r} must be one line: a shell would run only the first',
            line=at,
            hint=_TEST_HINT,
        )
    return trimmed


def read_manifest(path: Path) -> Package:
    """Parse and validate one ``ffrwd.json`` into the package it declares.

    Raises ``FfrwdError`` -- and nothing else -- on every rejection: an
    unreadable file, text that is not JSON, a missing or malformed key, a name
    ffrwd keeps for itself (:func:`name_refusal`), a lib or bin that names no
    file, a malformed dependency. Whether a named file DEFINES what the manifest says
    it does is checked where the file is parsed, not here.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as err:
        raise _reject(path, f"could not be read: {err.strerror or err}") from err
    try:
        data = json.loads(text, object_pairs_hook=_as_object)
    except json.JSONDecodeError as err:
        raise _reject(
            path,
            f"is not valid JSON: {err.msg}",
            line=err.lineno,
            col=err.colno,
            hint=_MANIFEST_HINT,
        ) from err
    except (ValueError, RecursionError) as err:  # backstop: never a traceback
        raise _reject(path, "is not valid JSON", line=1, col=1, hint=_MANIFEST_HINT) from err
    if not isinstance(data, dict):
        raise _reject(path, "is not a JSON object", line=1, col=1, hint=_MANIFEST_HINT)

    for key in sorted(data):
        if key in _KNOWN:
            continue
        hint = (
            _RETIRED.get(key)
            or _did_you_mean(key, sorted(_KNOWN))
            or f"known keys: {', '.join(sorted(_KNOWN))}"
        )
        raise _reject(path, f"unknown key {key!r}", line=_key_line(text, key), hint=hint)
    for key in _REQUIRED:
        if key not in data:
            raise _reject(path, f'is missing "{key}"', line=1, col=1, hint=_MANIFEST_HINT)

    name = _package_name(data, path, text)
    segment = name.partition("/")[2]
    return Package(
        name=name,
        version=_text_field(data, "version", path, text),
        root=path.parent,
        manifest=path,
        exports=_exports(data, segment, path.parent, path, text),
        recipes=_recipes(data, segment, path.parent, path, text),
        root_export=segment if isinstance(data.get("lib"), str) else None,
        root_recipe=segment if isinstance(data.get("bin"), str) else None,
        dependencies=_dependencies(data, path, text),
        keywords=_keywords(data, path, text),
        license=_license(data, path, text),
        homepage=_homepage(data, path, text),
        capabilities=_capabilities(data, path, text),
        engines=_engines(data, path, text),
        models=_models(data, path, text),
        private=_private(data, path, text),
        test=_test(data, path, text),
    )


# -- reading one lockfile --------------------------------------------------

# Bump on any change to the lockfile's shape. Another version's file is
# rejected rather than read optimistically: installing rewrites the lockfile,
# and guessing at a shape would resolve a call against content nobody pinned.
LOCK_FORMAT_VERSION = 3

_LOCK_HINT = 'a lockfile is one JSON object with "format_version", "reproducible" and "packages"'

_LOCK_REQUIRED = ("format_version", "reproducible", "packages")
_LOCK_KNOWN = frozenset({*_LOCK_REQUIRED, "not_reproducible_because", "dependencies"})

_REGISTRY_KEYS = ("kind", "name", "version", "sha256", "store", "dependencies")
_LINK_KEYS = ("kind", "path")
_KINDS = ("link", "registry")

_ENTRY_DEPENDENCIES_HINT = (
    'a registry entry\'s "dependencies" is an object of package name to the exact '
    "version it was resolved to"
)


@dataclass(frozen=True)
class RegistryEntry:
    """A package installed from the registry: a version, and the archive digest that pinned it.

    `dependencies` is what THIS version's own manifest resolved its own
    dependencies to when it was installed -- package name to the exact
    version, never a range. Several entries may share one `name` at
    different `version`s: install never removes one version to make room for
    another, so a project depending on two packages that each depend on a
    different version of a third package pins both.
    """

    name: str
    version: str
    sha256: str
    store: str
    dependencies: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class LinkEntry:
    """A package read live out of a working directory: no version, no digest.

    That is not an omission. A link exists so edits to that directory land in
    the next compile, which is exactly what a digest cannot survive.

    One of the two fields is set, sometimes both. The machine-wide links file
    records both: `ffrwd link` run in the package's directory wrote name ->
    directory there. A project's links file records the `name` alone, and the
    machine-wide file says where it lives -- re-linking from a new directory
    re-points every project at once. A bare `path` is what older ffrwds
    wrote, into a project's links file or the lockfile itself; still read,
    and migrated by the next write.
    """

    path: str | None = None
    name: str | None = None


LockEntry = RegistryEntry | LinkEntry


@dataclass(frozen=True)
class Lockfile:
    """One ``ffrwd.lock``: what it pins, and whether it pins all of it.

    `reproducible` is false when some entry is a link, and the file says so in
    its own text -- both the flag and a sentence naming why -- so a human
    reading it is not left to infer it from the entry kinds.

    `dependencies` is what THIS lockfile's own project directly installed --
    package name to the exact version -- the same shape a `RegistryEntry`
    carries for its own dependencies, one level up. Empty for a lockfile with
    no project above it, or one nothing has been directly installed into yet.
    """

    path: Path
    reproducible: bool
    entries: tuple[LockEntry, ...]
    dependencies: Mapping[str, str] = field(default_factory=dict)

    def links(self) -> tuple[LinkEntry, ...]:
        return tuple(entry for entry in self.entries if isinstance(entry, LinkEntry))


def _value_line(text: str, value: object) -> int | None:
    """The line a string VALUE is written on, for a rejection about one entry.

    Entries repeat their keys, so a rejection anchored on ``"name"`` would
    point at the first entry whatever entry it is about; the name's own text
    is what tells them apart.
    """
    return _key_line(text, value) if isinstance(value, str) else None


def _entry_dict(raw: object, path: Path, text: str, index: int) -> dict[str, object]:
    if not isinstance(raw, dict):
        raise _reject(
            path,
            f"package entry {index} is not a JSON object",
            line=_key_line(text, "packages"),
            hint=_LOCK_HINT,
        )
    return raw


def _entry_kind(data: dict[str, object], path: Path, line: int | None) -> str:
    kind = data.get("kind")
    if kind in _KINDS:
        assert isinstance(kind, str)
        return kind
    written = f"kind {kind!r}" if isinstance(kind, str) else 'no "kind"'
    hint = (_did_you_mean(kind, list(_KINDS)) if isinstance(kind, str) else None) or (
        "a package entry is a 'registry' one, pinning a version and a digest, "
        "or a 'link' one, naming a directory"
    )
    raise _reject(path, f"a package entry has {written}", line=line, hint=hint)


def _string_map(value: object, path: Path, line: int | None, hint: str) -> dict[str, str]:
    """A JSON object of string to string, or a typed rejection."""
    if not isinstance(value, dict) or any(
        not isinstance(k, str) or not isinstance(v, str) for k, v in value.items()
    ):
        raise _reject(path, "must be an object of name to version", line=line, hint=hint)
    return {str(k): str(v) for k, v in value.items()}


def _entry(raw: object, path: Path, text: str, index: int) -> LockEntry:
    """One ``packages`` element, validated into the entry it declares."""
    data = _entry_dict(raw, path, text, index)
    line = _value_line(text, data.get("name")) or _value_line(text, data.get("path"))
    kind = _entry_kind(data, path, line)
    keys = _REGISTRY_KEYS if kind == "registry" else _LINK_KEYS
    for key in sorted(data):
        if key in keys:
            continue
        hint = _did_you_mean(key, list(keys)) or f"a {kind} entry holds: {', '.join(keys)}"
        raise _reject(path, f"unknown key {key!r} in a {kind} entry", line=line, hint=hint)
    for key in keys:
        if key == "dependencies":  # the one optional key: absent means none
            continue
        if key not in data:
            raise _reject(
                path,
                f'a {kind} entry is missing "{key}"',
                line=line,
                hint=f"a {kind} entry holds: {', '.join(keys)}",
            )
    if kind == "link":
        return LinkEntry(path=_text_field(data, "path", path, text, hint=_LOCK_HINT))
    name = _text_field(data, "name", path, text, hint=_LOCK_HINT)
    if not is_package_name(name):
        raise _reject(
            path, f"name {name!r} is not a package name", line=line, hint=_NAME_HINT
        )
    refused = name_refusal(name)
    if refused is not None:
        raise _reject(
            path,
            f"package '{name}': {refused[0]}",
            line=line,
            hint=f"{refused[1]}; no such package exists, so drop the entry",
        )
    return RegistryEntry(
        name=name,
        version=_text_field(data, "version", path, text, hint=_LOCK_HINT),
        sha256=_text_field(data, "sha256", path, text, hint=_LOCK_HINT),
        store=_text_field(data, "store", path, text, hint=_LOCK_HINT),
        dependencies=_string_map(
            data.get("dependencies", {}), path, line, _ENTRY_DEPENDENCIES_HINT
        ),
    )


def _entries(data: dict[str, object], path: Path, text: str) -> tuple[LockEntry, ...]:
    line = _key_line(text, "packages")
    raw = data["packages"]
    if not isinstance(raw, list):
        raise _reject(path, '"packages" must be a list', line=line, hint=_LOCK_HINT)
    entries: list[LockEntry] = []
    named: set[tuple[str, str]] = set()
    linked: set[str] = set()
    for index, element in enumerate(raw):
        entry = _entry(element, path, text, index)
        if isinstance(entry, RegistryEntry):
            identity = (entry.name, entry.version)
            if identity in named:
                raise _reject(
                    path,
                    f"two entries pin package '{entry.name}' at version '{entry.version}'",
                    line=_value_line(text, entry.name),
                    hint="one package at one version, one entry",
                )
            named.add(identity)
        else:
            written = entry.path or ""
            if written in linked:
                raise _reject(
                    path,
                    f"two entries link {entry.path!r}",
                    line=_value_line(text, entry.path),
                    hint="one directory, one link",
                )
            linked.add(written)
        entries.append(entry)
    return tuple(entries)


def read_lockfile(path: Path) -> Lockfile:
    """Parse and validate one ``ffrwd.lock`` into the packages it pins.

    Raises ``FfrwdError`` -- and nothing else -- on every rejection: an
    unreadable file, text that is not JSON, another format version, a
    malformed entry, two entries naming one package, or a file that claims
    to be reproducible while linking a directory.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as err:
        raise _reject(path, f"could not be read: {err.strerror or err}") from err
    try:
        data = json.loads(text)
    except json.JSONDecodeError as err:
        raise _reject(
            path, f"is not valid JSON: {err.msg}", line=err.lineno, col=err.colno, hint=_LOCK_HINT
        ) from err
    except (ValueError, RecursionError) as err:  # backstop: never a traceback
        raise _reject(path, "is not valid JSON", line=1, col=1, hint=_LOCK_HINT) from err
    if not isinstance(data, dict):
        raise _reject(path, "is not a JSON object", line=1, col=1, hint=_LOCK_HINT)

    for key in sorted(data):
        if key in _LOCK_KNOWN:
            continue
        hint = (
            _did_you_mean(key, sorted(_LOCK_KNOWN))
            or f"known keys: {', '.join(sorted(_LOCK_KNOWN))}"
        )
        raise _reject(path, f"unknown key {key!r}", line=_key_line(text, key), hint=hint)
    for key in _LOCK_REQUIRED:
        if key not in data:
            raise _reject(path, f'is missing "{key}"', line=1, col=1, hint=_LOCK_HINT)
    if data["format_version"] != LOCK_FORMAT_VERSION:
        raise _reject(
            path,
            f"was written in lockfile format {data['format_version']!r}, and this "
            f"ffrwd reads {LOCK_FORMAT_VERSION}",
            line=_key_line(text, "format_version"),
            hint="install the project's packages again to rewrite it",
        )
    reproducible = data["reproducible"]
    if not isinstance(reproducible, bool):
        raise _reject(
            path,
            '"reproducible" must be true or false',
            line=_key_line(text, "reproducible"),
            hint=_LOCK_HINT,
        )
    because = data.get("not_reproducible_because")
    if because is not None and not isinstance(because, str):
        raise _reject(
            path,
            '"not_reproducible_because" must be a string',
            line=_key_line(text, "not_reproducible_because"),
            hint="it is the sentence a reader of the file sees; leave it out when there "
            "is nothing to say",
        )

    dependencies = _string_map(
        data.get("dependencies", {}),
        path,
        _key_line(text, "dependencies"),
        'a lockfile\'s own "dependencies" is an object of package name to the exact '
        "version this project directly installed",
    )
    lockfile = Lockfile(
        path=path,
        reproducible=reproducible,
        entries=_entries(data, path, text),
        dependencies=dependencies,
    )
    linked = lockfile.links()
    if reproducible and linked:
        raise _reject(
            path,
            f"claims to be reproducible while linking {linked[0].path!r}",
            line=_key_line(text, "reproducible"),
            hint="a linked directory is edited in place, so nothing here pins it: a "
            "lockfile holding a link is not reproducible",
        )
    return lockfile


# -- writing the two files -------------------------------------------------

# The sentence a lockfile holding a link carries in its own text.
_LINKED_BECAUSE = (
    "a package is linked to a working directory, so its files are not pinned here"
)

_WRITE_HINT = "check that the directory exists and is writable"


def _unwritable(path: Path, err: OSError) -> FfrwdError:
    return _reject(path, f"could not be written: {err.strerror or err}", hint=_WRITE_HINT)


def _write_atomically(path: Path, text: str) -> None:
    """Replace `path` with `text` in one step, LF-terminated on every platform.

    Written beside the destination and moved onto it, so a reader sees the old
    file or the new one and never half of either. Unlike the registry's disk
    cache, a failure here is a rejection: this file is the project's, and
    losing it silently is not an option the caller has.
    """
    directory = path.parent
    try:
        directory.mkdir(parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(dir=directory, prefix=f"{path.name}-", suffix=".tmp")
    except OSError as err:
        raise _unwritable(path, err) from err
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as file:
            file.write(text)
        os.replace(temporary, path)
    except OSError as err:
        try:
            os.unlink(temporary)
        except OSError:  # already gone, or the directory refuses us twice
            pass
        raise _unwritable(path, err) from err


def _rendered(payload: dict[str, object]) -> str:
    """`payload` as the text of one project file: written order, 2-space indent, LF.

    Insertion order, never sorted: the order keys and entries were handed over
    in is the order the file holds them, so writing the same thing twice
    produces the same bytes and a rewrite shows only what changed.
    """
    return json.dumps(payload, indent=2) + "\n"


def write_manifest(
    path: Path,
    *,
    name: str,
    version: str,
    description: str | None = None,
    lib: str | Mapping[str, str] | None = None,
    bin: str | Mapping[str, str] | None = None,
    dependencies: Mapping[str, str] | None = None,
    keywords: Sequence[str] | None = None,
    capabilities: Sequence[str] | None = None,
) -> None:
    """Write the ``ffrwd.json`` declaring this package.

    `lib` and `bin` each take a string (one file, named for the package
    segment) or a map (several, named for their keys). Only the two required
    keys are always written; an optional one is left out entirely rather than
    written empty.

    `keywords` and `capabilities` are the exception: given, each is written as
    given, empty list included. A scaffold declares both empty for its author
    to fill in, and absent is what a caller passing None asks for.
    """
    payload: dict[str, object] = {"name": name, "version": version}
    if description:
        payload["description"] = description
    if lib:
        payload["lib"] = lib if isinstance(lib, str) else dict(lib)
    if bin:
        payload["bin"] = bin if isinstance(bin, str) else dict(bin)
    if dependencies:
        payload["dependencies"] = dict(dependencies)
    if keywords is not None:
        payload["keywords"] = list(keywords)
    if capabilities is not None:
        payload["capabilities"] = list(capabilities)
    _write_atomically(path, _rendered(payload))


def add_dependency(path: Path, name: str, version: str) -> None:
    """Record `name` at `version` in the manifest's own ``dependencies``, keyed by name.

    The file is rewritten from its own text rather than from a parsed
    `Package`, so what the author wrote stays as written and a rewrite shows
    only the dependency that changed.

    The version is written exact. A range is a thing a manifest may hold and
    a reader may show; nothing here solves one.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as err:
        raise _reject(path, f"could not be read: {err.strerror or err}") from err
    try:
        data = json.loads(text, object_pairs_hook=_as_object)
    except (ValueError, RecursionError) as err:
        raise _reject(path, "is not valid JSON", line=1, col=1, hint=_MANIFEST_HINT) from err
    if not isinstance(data, dict):
        raise _reject(path, "is not a JSON object", line=1, col=1, hint=_MANIFEST_HINT)
    held = data.get("dependencies", {})
    if not isinstance(held, dict) or any(not isinstance(value, str) for value in held.values()):
        raise _reject(
            path,
            '"dependencies" is not an object of "<namespace>/<package>" to version',
            line=_key_line(text, "dependencies"),
            hint=_DEPENDENCIES_HINT,
        )
    kept = dict(held)
    kept[name] = version
    data["dependencies"] = kept
    _write_atomically(path, _rendered(dict(data)))


def _entry_payload(entry: LockEntry) -> dict[str, object]:
    """One entry as the lockfile writes it, keys in the order the reader lists them."""
    if isinstance(entry, LinkEntry):
        return {"kind": "link", "path": entry.path}
    payload: dict[str, object] = {
        "kind": "registry",
        "name": entry.name,
        "version": entry.version,
        "sha256": entry.sha256,
        "store": entry.store,
    }
    if entry.dependencies:
        payload["dependencies"] = dict(entry.dependencies)
    return payload


def lockfile_text(
    entries: Sequence[LockEntry], *, dependencies: Mapping[str, str] | None = None
) -> str:
    """The text of an ``ffrwd.lock`` pinning `entries`, in the order given.

    What :func:`write_lockfile` puts on disk, as a string: a remote submit
    sends a lock document that lives nowhere as a file.

    The reproducibility claim is this function's, not the caller's: a link is
    read live and no digest survives that, so any link among `entries` makes
    the text say it is not reproducible and say why. `read_lockfile` refuses a
    file claiming otherwise, so a caller allowed to set the flag could write
    one ffrwd would not read back.
    """
    linked = any(isinstance(entry, LinkEntry) for entry in entries)
    payload: dict[str, object] = {
        "format_version": LOCK_FORMAT_VERSION,
        "reproducible": not linked,
    }
    if linked:
        payload["not_reproducible_because"] = _LINKED_BECAUSE
    if dependencies:
        payload["dependencies"] = dict(dependencies)
    payload["packages"] = [_entry_payload(entry) for entry in entries]
    return _rendered(payload)


def write_lockfile(
    path: Path, entries: Sequence[LockEntry], *, dependencies: Mapping[str, str] | None = None
) -> None:
    """Write the ``ffrwd.lock`` pinning `entries`, in the order given.

    `dependencies` is what this lockfile's own project directly installed --
    package name to the exact version -- carried over unchanged by a caller
    that is not itself recording a direct install (``link``, ``unlink``).
    """
    _write_atomically(path, lockfile_text(entries, dependencies=dependencies))


def stored_name(entry: LockEntry, lock_path: Path) -> str | None:
    """The package name `entry` pins: a registry entry's own, a link's manifest's.

    None for a link whose directory holds no readable manifest -- a dead link
    still has to be listable and removable, so this never raises.
    """
    if isinstance(entry, RegistryEntry):
        return entry.name
    try:
        return read_manifest(_linked_root(entry, lock_path) / MANIFEST_NAME).name
    except FfrwdError:
        return None


def stored_version(entry: LockEntry, lock_path: Path) -> str | None:
    """The version `entry` pins, read the same way :func:`stored_name` reads the name."""
    if isinstance(entry, RegistryEntry):
        return entry.version
    try:
        return read_manifest(_linked_root(entry, lock_path) / MANIFEST_NAME).version
    except FfrwdError:
        return None


def entry_root(entry: LockEntry, lock_path: Path) -> Path:
    """The directory an entry's content is read out of.

    The store directory for an installed package, the linked directory for a
    link. Raises ``FfrwdError`` when the store holds no such content.
    """
    if isinstance(entry, LinkEntry):
        return _linked_root(entry, lock_path)
    return store.load(entry.name, entry.store, entry.sha256)


def held_entry(entries: Sequence[LockEntry], name: str, lock_path: Path) -> LockEntry | None:
    """The entry pinning package `name`, whichever kind it is, or None."""
    for entry in entries:
        if stored_name(entry, lock_path) == name:
            return entry
    return None


def with_entry(
    entries: Sequence[LockEntry], entry: LockEntry, replaced: LockEntry | None = None
) -> tuple[LockEntry, ...]:
    """`entries` with `entry` in place of `replaced`, or appended when there is none.

    One package, one entry: the caller finds what an install or a link
    replaces with :func:`held_entry`, since only it knows the lockfile the
    entries came from.
    """
    if replaced is None:
        return (*entries, entry)
    return tuple(entry if held == replaced else held for held in entries)


def without_entry(entries: Sequence[LockEntry], entry: LockEntry) -> tuple[LockEntry, ...]:
    """`entries` without `entry`, the rest in order."""
    return tuple(held for held in entries if held != entry)


# -- the links file --------------------------------------------------------

LINKS_FORMAT_VERSION = 2

# Version 1 held directory paths only; its files still read.
_LINKS_READABLE = frozenset({1, LINKS_FORMAT_VERSION})

# The sentence the file carries about itself, under "machine_local".
_LINKS_PURPOSE = (
    "the packages this machine reads live out of working directories; not for version control"
)

_LINKS_HINT = (
    "`ffrwd link` run in a package's directory records it for this machine; "
    "`ffrwd link <name>` records it in a project; `ffrwd unlink <name>` removes one"
)

_LINKS_KNOWN = frozenset({"format_version", "machine_local", "links"})

_LINK_SHAPE_HINT = (
    'each link is {"name": "<namespace>/<package>"}, {"path": "<directory>"}, or both'
)

_RELINK_HINT = (
    "a linked package resolves through its own lockfile; run `ffrwd link` "
    "in its directory again"
)

_GONE_HINT = "the link is gone; run `ffrwd link` in the package's directory again"


def links_path(lock: Path) -> Path:
    """The links file that accompanies the lockfile at `lock`, present or not."""
    return lock.with_name(LINKSFILE_NAME)


def read_linksfile(path: Path) -> tuple[LinkEntry, ...]:
    """Parse the links file at `path`; no file is simply no links.

    Raises ``FfrwdError`` -- and nothing else -- for a file that is there and
    wrong: unreadable, not JSON, another format version, a record that names
    no directory.
    """
    try:
        if not path.is_file():
            return ()
        text = path.read_text(encoding="utf-8")
    except OSError as err:
        raise _reject(path, f"could not be read: {err.strerror or err}") from err
    try:
        data = json.loads(text)
    except json.JSONDecodeError as err:
        raise _reject(
            path, f"is not valid JSON: {err.msg}", line=err.lineno, col=err.colno, hint=_LINKS_HINT
        ) from err
    except (ValueError, RecursionError) as err:  # backstop: never a traceback
        raise _reject(path, "is not valid JSON", line=1, col=1, hint=_LINKS_HINT) from err
    if not isinstance(data, dict):
        raise _reject(path, "is not a JSON object", line=1, col=1, hint=_LINKS_HINT)
    for key in sorted(data):
        if key in _LINKS_KNOWN:
            continue
        hint = (
            _did_you_mean(key, sorted(_LINKS_KNOWN))
            or f"known keys: {', '.join(sorted(_LINKS_KNOWN))}"
        )
        raise _reject(path, f"unknown key {key!r}", line=_key_line(text, key), hint=hint)
    if data.get("format_version") not in _LINKS_READABLE:
        raise _reject(
            path,
            f"was written in links format {data.get('format_version')!r}, and this "
            f"ffrwd reads {LINKS_FORMAT_VERSION}",
            line=_key_line(text, "format_version"),
            hint="link the packages again to rewrite it",
        )
    note = data.get("machine_local")
    if note is not None and not isinstance(note, str):
        raise _reject(
            path,
            '"machine_local" must be a string',
            line=_key_line(text, "machine_local"),
            hint=_LINKS_HINT,
        )
    written = data.get("links", [])
    if not isinstance(written, list):
        raise _reject(
            path, '"links" must be a list', line=_key_line(text, "links"), hint=_LINKS_HINT
        )
    links: list[LinkEntry] = []
    for index, raw in enumerate(written):
        record = raw if isinstance(raw, dict) else {}
        named = record.get("name")
        directory = record.get("path")
        if named is not None and (not isinstance(named, str) or not is_package_name(named)):
            raise _reject(
                path,
                f"link {index + 1} does not name a package",
                line=_key_line(text, "links"),
                hint=_LINK_SHAPE_HINT,
            )
        if directory is not None and (not isinstance(directory, str) or not directory):
            raise _reject(
                path,
                f"link {index + 1} does not name a directory",
                line=_key_line(text, "links"),
                hint=_LINK_SHAPE_HINT,
            )
        if named is None and directory is None:
            raise _reject(
                path,
                f"link {index + 1} names neither a package nor a directory",
                line=_key_line(text, "links"),
                hint=_LINK_SHAPE_HINT,
            )
        links.append(LinkEntry(path=directory, name=named))
    return tuple(links)


def write_linksfile(path: Path, links: Sequence[LinkEntry]) -> None:
    """Write the links file naming `links`, in the order given; none removes the file.

    Every write migrates what an older ffrwd recorded (:func:`_migrated`).
    The sentence under ``machine_local`` is for whoever finds the file: it
    says why this one is not committed the way the lockfile is.
    """
    if not links:
        try:
            path.unlink(missing_ok=True)
        except OSError as err:
            raise _unwritable(path, err) from err
        return
    written: list[dict[str, str]] = []
    for entry in _migrated(path, links):
        record: dict[str, str] = {}
        if entry.name is not None:
            record["name"] = entry.name
        if entry.path is not None:
            record["path"] = entry.path
        written.append(record)
    payload: dict[str, object] = {
        "format_version": LINKS_FORMAT_VERSION,
        "machine_local": _LINKS_PURPOSE,
        "links": written,
    }
    _write_atomically(path, _rendered(payload))


def _migrated(path: Path, links: Sequence[LinkEntry]) -> tuple[LinkEntry, ...]:
    """`links` with each bare directory path re-recorded the current way.

    In the machine-wide file a path gains the name its manifest declares. In
    a project's file a path whose package the machine-wide file links becomes
    the name alone; one it does not links stays a path, resolving as it
    always did. A path whose manifest cannot be read stays as written --
    migration never invents or drops a record.
    """
    machine = links_path(store.global_lock_path())
    registered: tuple[LinkEntry, ...] = ()
    if path != machine:
        try:
            registered = read_linksfile(machine)
        except FfrwdError:
            registered = ()
    migrated: list[LinkEntry] = []
    seen: set[tuple[str | None, str | None]] = set()
    for entry in links:
        moved = entry
        if entry.path is not None and entry.name is None:
            try:
                named = read_manifest(
                    (path.parent / Path(entry.path)).resolve() / MANIFEST_NAME
                ).name
            except (FfrwdError, OSError, ValueError):
                named = None
            if named is not None and path == machine:
                moved = LinkEntry(path=entry.path, name=named)
            elif named is not None and any(
                one.name == named and one.path is not None for one in registered
            ):
                moved = LinkEntry(name=named)
        key = (moved.name, moved.path)
        if key in seen:
            continue
        seen.add(key)
        migrated.append(moved)
    return tuple(migrated)


def link_target(entry: LinkEntry, source: Path) -> Path | None:
    """The directory `entry` reads from, or None.

    A recorded path resolves against the file recording it; a name resolves
    through the machine-wide links file, or nowhere when this machine no
    longer links it.
    """
    if entry.path is not None:
        try:
            return (source.parent / Path(entry.path)).resolve()
        except (OSError, ValueError):
            return None
    if entry.name is None:
        return None
    return registered_root(entry.name)


def link_written(entry: LinkEntry) -> str:
    """The identifier a link was recorded as: its package name, or its path."""
    if entry.name is not None:
        return entry.name
    return entry.path or ""


def _link_label(entry: LinkEntry) -> str:
    """How a message points at one link record."""
    if entry.name is not None:
        return f"link '{entry.name}'"
    return f"link {entry.path!r}"


def registered_root(name: str) -> Path | None:
    """The directory the machine-wide links file records for package `name`, or None.

    Where a project's name-only link entry points: `ffrwd link` run in the
    package's directory wrote the record, and run from somewhere new it
    re-points every project reading the name.
    """
    try:
        return _registered_root(name, links_path(store.global_lock_path()))
    except FfrwdError:
        return None


def _registered_root(name: str, at: Path) -> Path:
    """:func:`registered_root`, or the rejection a resolving reader raises.

    `at` is the file whose record names the package, so the rejection lands
    on something the reader has open.
    """
    machine = links_path(store.global_lock_path())
    for entry in read_linksfile(machine):
        if entry.name != name or entry.path is None:
            continue
        try:
            return (machine.parent / Path(entry.path)).resolve()
        except (OSError, ValueError) as err:
            raise _reject(
                machine,
                f"link '{name}': {entry.path!r} is not a directory path",
                hint="a link names the directory the package is developed in",
            ) from err
    raise _reject(at, f"link '{name}': nothing on this machine links it", hint=_GONE_HINT)


def held_links(lock: Path) -> tuple[tuple[LinkEntry, Path], ...]:
    """Every link the project at `lock` holds, each with the file recording it.

    The links file first, then any link entry still sitting in the lockfile
    itself -- older ffrwds recorded them there, and they keep working until a
    write moves them over. Two records naming one directory are one link.
    """
    pairs: list[tuple[LinkEntry, Path]] = [
        (entry, links_path(lock)) for entry in read_linksfile(links_path(lock))
    ]
    try:
        present = lock.is_file()
    except (OSError, ValueError):
        present = False
    if present:
        pairs.extend(
            (entry, lock)
            for entry in read_lockfile(lock).entries
            if isinstance(entry, LinkEntry)
        )
    seen: set[Path | str] = set()
    kept: list[tuple[LinkEntry, Path]] = []
    for entry, source in pairs:
        root = link_target(entry, source)
        key: Path | str = root if root is not None else link_written(entry)
        if key in seen:
            continue
        seen.add(key)
        kept.append((entry, source))
    return tuple(kept)


def link_refusal(package: Package, root: Path) -> tuple[str, str] | None:
    """Why the linked package at `root` cannot resolve -- message and hint -- or None.

    A linked package resolves its calls through its OWN lockfile: every
    dependency its manifest pins must be pinned there at the written version,
    or itself linked in that directory. A package with no dependencies needs
    neither. `ffrwd link` run in the directory installs exactly this, so a
    refusal means the tree drifted since -- raised at the first resolution
    that reads it.
    """
    if not package.dependencies:
        return None
    lock = root / LOCKFILE_NAME
    try:
        present = lock.is_file()
    except (OSError, ValueError):
        present = False
    entries = read_lockfile(lock).entries if present else ()
    linked = {stored_name(entry, source) for entry, source in held_links(lock)}
    for name, version in package.dependencies.items():
        if name in linked:
            continue
        if any(
            isinstance(entry, RegistryEntry) and entry.name == name and entry.version == version
            for entry in entries
        ):
            continue
        where = (
            f"its own {LOCKFILE_NAME} does not pin it"
            if present
            else f"it has no {LOCKFILE_NAME}"
        )
        return (
            f"package '{package.name}' at {root} depends on '{name}@{version}' and {where}",
            _RELINK_HINT,
        )
    return None


# -- discovery -------------------------------------------------------------


def _start_directory(start: Path) -> Path | None:
    """Where an upward walk from `start` begins, or None for a path we cannot read."""
    try:
        current = Path(start).resolve()
        if current.is_file():
            current = current.parent
    except (OSError, ValueError):  # an unreadable or malformed path is not a project
        return None
    return current


def _walk_up(start: Path, name: str) -> Path | None:
    """The nearest `name` at or above `start`, or None at the filesystem root."""
    current = _start_directory(start)
    if current is None:
        return None
    for directory in (current, *current.parents):
        candidate = directory / name
        try:
            if candidate.is_file():
                return candidate
        except (OSError, ValueError):
            continue
    return None


def find_manifest(start: Path) -> Path | None:
    """The nearest ``ffrwd.json`` at or above `start`, or None at the root.

    The walk stops at the filesystem root; a project is optional, and finding
    none is the ordinary case, not a rejection.
    """
    return _walk_up(start, MANIFEST_NAME)


def find_lockfile(start: Path) -> Path | None:
    """The lockfile that belongs to the project above `start`, or None.

    Beside the manifest when there is one -- installing writes the two
    together, and a lockfile further up belongs to the project further up, not
    to this one. With no manifest anywhere, the nearest lockfile above `start`
    is the project.
    """
    manifest = find_manifest(start)
    if manifest is None:
        return _walk_up(start, LOCKFILE_NAME)
    beside = manifest.parent / LOCKFILE_NAME
    try:
        return beside if beside.is_file() else None
    except (OSError, ValueError):
        return None


def _linked_root(entry: LinkEntry, lock_path: Path) -> Path:
    """The directory a link entry reads from: its own path, or its name's record.

    A path resolves against the file holding the entry; a name resolves
    through the machine-wide links file, with the record gone a rejection.
    """
    if entry.path is None:
        if entry.name is None:
            raise _reject(
                lock_path,
                "a link names neither a package nor a directory",
                hint=_LINK_SHAPE_HINT,
            )
        return _registered_root(entry.name, lock_path)
    try:
        return (lock_path.parent / Path(entry.path)).resolve()
    except (OSError, ValueError) as err:
        raise _reject(
            lock_path,
            f"link {entry.path!r} is not a directory path",
            hint="a link names the directory the package is developed in",
        ) from err


def _manifest_of(root: Path, named: str, at: Path, missing: str) -> Path:
    """The manifest a recorded package is read through, or a rejection naming `missing`.

    `at` is the file whose record sent the reader here -- a lockfile, or the
    links file -- so the rejection lands on something the reader has open.
    """
    manifest = root / MANIFEST_NAME
    try:
        present = manifest.is_file()
    except (OSError, ValueError):
        present = False
    if not present:
        raise _reject(
            at,
            f"{named}: {root} holds no {MANIFEST_NAME}",
            hint=missing,
        )
    return manifest


def _same(entry_value: str, found: str, field_name: str, name: str, lock: Lockfile) -> None:
    """Reject a lockfile entry the package it points at disagrees with."""
    if entry_value == found:
        return
    raise _reject(
        lock.path,
        f"package '{name}': the lockfile records {field_name} {entry_value!r} and the "
        f"package says {found!r}",
        hint="the package changed since it was installed; install it again",
    )


def _linked_package(entry: LinkEntry, lock: Lockfile, layer: Layer) -> Package:
    """A link resolved: the local layer again, rooted somewhere else.

    Same `read_manifest`, so a linked package is validated exactly as the
    project's own is -- and its files are read on every compile, which is
    what makes an edit show up without reinstalling. Its name is the
    manifest's, recorded nowhere else.
    """
    root = _linked_root(entry, lock.path)
    manifest = _manifest_of(
        root,
        _link_label(entry),
        lock.path,
        "run `ffrwd link` in that directory again, or restore its manifest",
    )
    package = read_manifest(manifest)
    return replace(package, layer=layer, linked=True)


def _stored_package(entry: RegistryEntry, lock: Lockfile, layer: Layer) -> Package:
    """A registry entry resolved: the store directory its digest names."""
    try:
        root = store.load(entry.name, entry.store, entry.sha256)
    except FfrwdError as err:
        # Renamed onto the lockfile: that is the file the reader has open, not
        # a path under a cache directory they never chose.
        raise _reject(lock.path, err.message, hint=err.hint) from err
    manifest = _manifest_of(
        root,
        f"package '{entry.name}'",
        lock.path,
        "the stored content is not a package; install it again",
    )
    package = read_manifest(manifest)
    _same(entry.name, package.name, "name", entry.name, lock)
    _same(entry.version, package.version, "version", entry.name, lock)
    return replace(package, layer=layer)


def _add_layer(
    packages: dict[str, Package],
    versions: dict[str, dict[str, Package]],
    wants: dict[str, dict[str, str]],
    lock: Lockfile | None,
    layer: Layer,
) -> None:
    """Add `lock`'s packages under the names/versions no earlier layer claimed."""
    if lock is None:
        return
    resolved: set[tuple[str, str]] = set()
    for entry in lock.entries:
        if isinstance(entry, LinkEntry):
            package = _linked_package(entry, lock, layer)
        else:
            package = _stored_package(entry, lock, layer)
            wants.setdefault(package.name, dict(entry.dependencies))
        identity = (package.name, package.version)
        if identity in resolved:
            # The reader catches two same-kind claims; a registry entry and a
            # link resolving to one package AT ONE VERSION is only knowable
            # here. Two different versions of one name are not a collision --
            # that is the whole point of carrying several.
            raise _reject(
                lock.path,
                f"two entries name package '{package.name}' {package.version}",
                hint="one package at one version, one entry; install or link the one "
                "you meant to keep",
            )
        resolved.add(identity)
        versions.setdefault(package.name, {}).setdefault(package.version, package)
        if package.name in packages:  # first claim wins, layer by layer
            continue
        packages[package.name] = package


def _add_links(
    packages: dict[str, Package],
    versions: dict[str, dict[str, Package]],
    wants: dict[str, dict[str, str]],
    links: Sequence[tuple[LinkEntry, Path]],
    layer: Layer,
) -> dict[str, str]:
    """Add the linked directories under the names no earlier claim took.

    Returns what was claimed -- linked package name to its manifest's version
    -- for the caller to bind the project's own calls to: a link answers its
    name over anything installed under it.
    """
    claimed: dict[str, str] = {}
    chain: list[tuple[Path, str]] = []
    for entry, source in links:
        package = _add_link(packages, versions, wants, entry, source, layer, chain, claims=True)
        if packages.get(package.name) is package:
            claimed[package.name] = package.version
    return claimed


def _add_link(
    packages: dict[str, Package],
    versions: dict[str, dict[str, Package]],
    wants: dict[str, dict[str, str]],
    entry: LinkEntry,
    source: Path,
    layer: Layer,
    chain: list[tuple[Path, str]],
    *,
    claims: bool,
) -> Package:
    """One linked directory: the live package, resolving through its own lockfile.

    The linked tree's lockfile is where ITS calls bind: its pins load into
    `versions` and `wants` for the calls it owns, and nowhere else -- the
    project around the link never sees them. `claims` is True for a link this
    layer's own project made, which answers the package's name everywhere; a
    link found inside another linked package claims no name. `chain` is the
    directories the walk is inside, so a link leading back into one is refused
    as the cycle it is.
    """
    root = _linked_root(entry, source)
    manifest = _manifest_of(
        root,
        _link_label(entry),
        source,
        "run `ffrwd link` in that directory again, or restore its manifest",
    )
    package = replace(read_manifest(manifest), layer=layer, linked=True)
    for at, (walked, _named) in enumerate(chain):
        if walked == root:
            loop = " -> ".join([*(name for _, name in chain[at:]), package.name])
            raise _reject(
                source,
                f"link cycle: {loop}",
                hint="one of these packages has to stop linking another in the loop",
            )
    refused = link_refusal(package, root)
    if refused is not None:
        raise _reject(source, refused[0], hint=refused[1])
    versions.setdefault(package.name, {}).setdefault(package.version, package)
    if claims:
        packages.setdefault(package.name, package)

    own_path = root / LOCKFILE_NAME
    try:
        own = read_lockfile(own_path) if own_path.is_file() else None
    except (OSError, ValueError):
        own = None
    binding: dict[str, str] = dict(own.dependencies) if own is not None else {}
    chain.append((root, package.name))
    for nested_entry, nested_source in held_links(own_path):
        nested = _add_link(
            packages, versions, wants, nested_entry, nested_source, layer, chain, claims=False
        )
        binding[nested.name] = nested.version
    chain.pop()
    wants.setdefault(package.name, binding)
    if own is not None:
        for held in own.entries:
            if isinstance(held, LinkEntry):
                continue
            stored = _stored_package(held, own, layer)
            versions.setdefault(stored.name, {}).setdefault(stored.version, stored)
            wants.setdefault(stored.name, dict(held.dependencies))
    return package


def _global_lockfile(local: Path | None) -> Lockfile | None:
    """The machine-wide lockfile, or None when nothing was installed globally."""
    path = store.global_lock_path()
    try:
        if not path.is_file() or (local is not None and path == local):
            return None
    except (OSError, ValueError):
        return None
    return read_lockfile(path)


def discover(start: Path | str | None = None) -> PackageSet | None:
    """The package set for a query written in `start`, or None with nothing to resolve in.

    `start` is a directory or a query file's path; None means the working
    directory, which is the CLI's answer for a query typed on the command
    line.

    Three layers, the first claim on a name winning: the project's own
    manifest, then its lockfile, then the machine-wide one -- each lockfile
    preceded by the links file beside it, so a linked directory answers its
    name over anything installed under it. The layering lives here and
    nowhere else -- what the compiler gets is one name to one canonical
    package per layer, plus every version install ever pinned, with no idea
    which layer answered.

    Raises ``FfrwdError`` for a manifest, lockfile or links file that is
    found but malformed, for a recorded package the store or the linked
    directory cannot produce, or for a linked package whose own lockfile does
    not cover its manifest's dependencies.
    """
    base = Path(start) if start is not None else Path.cwd()
    manifest = find_manifest(base)
    local = find_lockfile(base)
    packages: dict[str, Package] = {}
    versions: dict[str, dict[str, Package]] = {}
    wants: dict[str, dict[str, str]] = {}
    project: Package | None = None
    if manifest is not None:
        project = read_manifest(manifest)
        packages[project.name] = project
        versions.setdefault(project.name, {})[project.version] = project

    # Links before pins, layer by layer: a linked directory answers its name
    # over anything installed under it.
    claimed: dict[str, str] = {}
    local_lock = read_lockfile(local) if local is not None else None
    if local is not None:
        pairs = [(entry, links_path(local)) for entry in read_linksfile(links_path(local))]
        claimed.update(_add_links(packages, versions, wants, pairs, "local"))
    _add_layer(packages, versions, wants, local_lock, "local")
    global_lock = store.global_lock_path()
    if local is None or global_lock != local:
        pairs = [
            (entry, links_path(global_lock)) for entry in read_linksfile(links_path(global_lock))
        ]
        claimed.update(_add_links(packages, versions, wants, pairs, "global"))
    _add_layer(packages, versions, wants, _global_lockfile(local), "global")

    if project is not None:
        wants[project.name] = dict(local_lock.dependencies) if local_lock is not None else {}
        wants[project.name].update(claimed)

    in_project = manifest is not None or local is not None
    if not in_project and not packages:
        return None
    root = manifest.parent if manifest is not None else local.parent if local is not None else base
    return PackageSet(
        root=root,
        packages=packages,
        versions=versions,
        wants=wants,
        project=project.name if project is not None else None,
        in_project=in_project,
        manifest=manifest,
        start=base,
    )
