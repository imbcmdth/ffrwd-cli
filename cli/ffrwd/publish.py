"""Publishing a package: everything checked here, then one upload.

The publisher has the whole toolchain -- the compiler, ffmpeg and the sidecar
are what a package is built with -- so the validation lives on this side.
``ffrwd publish`` reads the manifest, parses every export, resolves every
dependency against the registry, asks the sidecar what each module declares,
runs the package's own test command when its manifest declares one, packs the
directory and writes the documents the registry stores. Only then does
anything leave the machine, and what leaves is one request.

A package ships queries, not media, so there is no file to probe: every
recipe is still compiled against :data:`SYNTHETIC_PROBE`, one synthetic file
with a video, an audio and a subtitle stream plus chapters, with the values
the recipe's own ``-- example:`` line names substituted in -- not to validate
the recipe, which is the package's own ``test`` command's job, but to read
off each variable's required or optional list for the version document. A
recipe with no example line compiles with its variables unset, which is what
running it that way would do. A recipe that will not even compile that way
never blocks the publish: its entry gets ``"compiles": false`` and empty
``required``/``optional`` lists instead of a claim nobody checked, and one
warning names the recipe and the compiler's own message.

What the registry then checks is only what it alone can: the token and its
scope, that the namespace is the publisher's, that a version already
published is not being changed under the same name, and that the bytes hash
to the claimed digest. Its refusals come back as a message and a hint, and
are printed exactly as any other rejection here is.

Nothing in this module writes to stdout or stderr; every rejection is a
``FfrwdError``.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import shlex
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from markdown_it import MarkdownIt

from . import compiler, credentials, store
from . import packages as packages_module
from .console import Announce, written_size
from .errors import ErrorCode, FfrwdError
from .functions import package_modules, package_signatures
from .probe import ChapterMeta, ProbeResult, StreamMeta
from .project import (
    README_NAME,
    ModelPin,
    Package,
    PackageSet,
    is_package_name,
    name_refusal,
    read_manifest,
)
from .vars import Variable, declared_variables, substitute, unset_variable
from .warnings import FfrwdWarning, OnWarning, WarningCode

__all__ = [
    "PRIVATE",
    "PUBLIC",
    "SYNTHETIC_PROBE",
    "Prepared",
    "Published",
    "RecipeVariables",
    "prepare",
    "publish",
    "recipe_variables",
    "required_variables",
]

Document = dict[str, Any]

# What the registry's documents are shaped as. The same number the client
# reads back.
FORMAT_VERSION = packages_module.FORMAT_VERSION

# What one publish may weigh. Models live on the hub, not in the archive,
# which is what keeps a package to kilobytes.
_MAX_ARCHIVE_BYTES = 16 * 1024 * 1024

# What the registry may answer a publish with.
_MAX_RESPONSE_BYTES = 1024 * 1024

_LOGIN_HINT = "run `ffrwd login --token <token>` first"

# What a version is stamped at publish, and where the stamp comes from: the
# command, not the manifest. The registry reads it only when the package is
# new -- afterwards the package's own default, edited in the dashboard, is
# what a publish is stamped with.
PUBLIC = "public"
PRIVATE = "private"


def _reject(message: str, hint: str) -> FfrwdError:
    return FfrwdError(ErrorCode.UNSUPPORTED_SQL, message, hint=hint)


# --------------------------------------------------------------------------
# compiling a recipe without a media file
# --------------------------------------------------------------------------

SYNTHETIC_PROBE = ProbeResult(
    streams=[
        StreamMeta(
            type="video",
            index=0,
            metadata={},
            width=1920,
            height=1080,
            fps="30/1",
            sample_rate=None,
            codec="h264",
        ),
        StreamMeta(
            type="audio",
            index=0,
            metadata={"language": "eng"},
            width=None,
            height=None,
            fps=None,
            sample_rate=48000,
            codec="aac",
            channels=2,
            channel_layout="stereo",
        ),
        StreamMeta(
            type="subtitle",
            index=0,
            metadata={"language": "eng"},
            width=None,
            height=None,
            fps=None,
            sample_rate=None,
            codec="srt",
        ),
    ],
    chapters=[
        ChapterMeta(index=1, start_t=0.0, end_t=30.0, title="Intro"),
        ChapterMeta(index=2, start_t=30.0, end_t=90.0, title="Credits"),
    ],
    # Real files report one, and a recipe may compute from it -- a trim bound
    # relative to the end, say. Without it such a recipe cannot be validated.
    duration=90.0,
    # Container tags, for the same reason: a recipe that rewrites a title from
    # the file's own has nothing to read without them.
    tags={"title": "Synthetic", "artist": "ffrwd"},
)


class _synthetic_probe:
    """Answer every probe with the synthetic file, for as long as this is held.

    The compiler's probe seam, redirected the way ffrwd's own checks redirect
    it, and put back afterwards: a publish runs in the same process as
    whatever asked for it.
    """

    def __enter__(self) -> None:
        self._held = compiler.probe_path
        compiler.probe_path = _synthetic

    def __exit__(self, *_exception: object) -> None:
        compiler.probe_path = self._held


def _synthetic(path: str, *_args: object, **_named: object) -> ProbeResult:
    """Every probe's answer while a publish runs: the one synthetic file."""
    return SYNTHETIC_PROBE


# The `-- example:` line a runnable query carries names values for its own
# variables, which is where the placeholders a compile check substitutes come
# from: a package supplies its own, and nothing here holds a table of names.
_EXAMPLE_RE = re.compile(r"^--\s*example:(?P<body>.*)$", re.MULTILINE)
_SET_RE = re.compile(r"-v\s+([A-Za-z_][A-Za-z0-9_]*)=(\S+)")


def _placeholders(text: str) -> dict[str, str]:
    """The example line's own values -- nothing filled in for a name it leaves out.

    A declared variable the example does not set stays unset, which substitutes
    to NULL: a required one left out is a real rejection (the example should
    run), and an optional one left out simply drops -- the check exercises
    exactly what running the example does, no invented value standing in.
    """
    example = _EXAMPLE_RE.search(text)
    return dict(_SET_RE.findall(example.group("body"))) if example is not None else {}


def _example_pairs(text: str) -> list[tuple[str, str]]:
    """The `-v name=value` pairs `text`'s own `-- example:` line sets, in its own order.

    Shell-quoted the way a value with a space in it (``-v title='My Film'``)
    is written -- ``shlex`` undoes the quoting so the value is the whole
    ``My Film``, not the fragment before the space a plain regex would stop
    at. A recipe with no example line sets none.
    """
    example = _EXAMPLE_RE.search(text)
    if example is None:
        return []
    try:
        tokens = shlex.split(example.group("body"))
    except ValueError:
        return []
    pairs: list[tuple[str, str]] = []
    words = iter(tokens)
    for token in words:
        if token != "-v":
            continue
        setting = next(words, None)
        if setting is not None and "=" in setting:
            name, _, value = setting.partition("=")
            pairs.append((name, value))
    return pairs


# A recipe's first `--` line, before its `-- variables:` / `-- example:`
# lines -- what a recipe says about itself in one sentence.
_FIRST_LINE_RE = re.compile(r"\A--[ \t]*(?P<body>[^\r\n]*)")
_HEADER_LINE_RE = re.compile(r"^(variables|example):", re.IGNORECASE)


def _recipe_description(text: str) -> str:
    """`text`'s own opening line, its `--` marker and surrounding space stripped.

    Empty when the header opens straight into ``-- variables:`` or
    ``-- example:`` rather than a description, or carries no ``--`` line at
    all.
    """
    match = _FIRST_LINE_RE.match(text)
    if match is None:
        return ""
    body = match.group("body").strip()
    return "" if _HEADER_LINE_RE.match(body) else body


def _qualified_recipe(package: Package, name: str) -> str:
    """`name` addressed the way ``ffrwd run`` reaches it: the package name
    alone for its default recipe, ``:<name>`` appended for any other."""
    if name == package.package:
        return package.name
    return f"{package.name}:{name}"


def _usage(package: Package, name: str, text: str) -> str:
    """A copy-pasteable ``ffrwd run`` command for recipe `name`.

    Synthesized rather than lifted from the example line's own command text,
    which names a path in the author's own tree that means nothing once the
    package is published -- only its ``-v name=value`` pairs travel, in the
    order the example itself writes them.
    """
    address = _qualified_recipe(package, name)
    settings = " ".join(f"-v {shlex.quote(f'{n}={v}')}" for n, v in _example_pairs(text))
    return f"ffrwd run {address}" + (f" {settings}" if settings else "")


def _compiles_as_table(
    text: str, packages: PackageSet, unset: Mapping[tuple[int, int], str] | None = None
) -> bool:
    """True when `text` compiles through the table/csv pipeline.

    A metadata or CSV query has no ffmpeg command to build and is refused by
    the media compiler; it is still a query that compiles, which is what
    ``ffrwd validate`` means by valid.
    """
    try:
        is_table_capable, _has_copy = compiler.classify(text, packages=packages, unset=unset)
        if not is_table_capable:
            return False
        compiler.compile_table_sql(text, packages=packages, unset=unset)
    except FfrwdError:
        return False
    return True


def _try_compile(
    text: str, packages: PackageSet | None, unset: dict[tuple[int, int], str]
) -> FfrwdError | None:
    """None on a clean compile, else the rejection -- streaming first, then table.

    A query with no streaming representation (metadata columns, an
    un-COALESCEd join gap) is retried as a table query before giving up,
    exactly as `compile`/`validate` do, so a recipe of either shape gets the
    same required/optional derivation.
    """
    try:
        compiler.compile_commands(text, packages=packages, unset=unset)
        return None
    except FfrwdError as err:
        stream_err = err
    try:
        is_table_capable, _has_copy = compiler.classify(text, packages=packages, unset=unset)
    except FfrwdError as err:
        return err
    if not is_table_capable:
        return stream_err
    try:
        compiler.compile_table_sql(text, packages=packages, unset=unset)
        return None
    except FfrwdError as err:
        return err


def required_variables(
    text: str, names: frozenset[str], packages: PackageSet | None
) -> frozenset[str]:
    """Every declared variable a compile rejects as required when it alone is unset.

    One compile per name: every OTHER declared variable gets a placeholder,
    leaving only this one NULL, so a rejection this round can only be about
    it -- and counts only when it IS (:func:`ffrwd.vars.unset_variable`, the
    shape every required NULL takes: `input(NULL)`, `TO NULL`, a NULL stream
    position, or a NULL on the curated required-options list). Testing one
    name at a time, rather than accumulating placeholders across rounds,
    survives a query where an earlier position's own NULL would otherwise
    block ever reaching a later one -- a UNION branch's trim bound ahead of
    its own `input()`, say -- so every declared name gets an independent
    read regardless of where in the query it sits.
    """
    required: set[str] = set()
    for name in names:
        values = {other: "1" for other in names if other != name}
        try:
            sub = substitute(text, values)
        except FfrwdError:
            # A "1" placeholder can trip a list subscript on ANOTHER
            # variable, which says nothing about this name's own read.
            continue
        err = _try_compile(sub.text, packages, sub.unset)
        if err is not None and unset_variable(err) == name:
            required.add(name)
    return frozenset(required)


@dataclass(frozen=True)
class RecipeVariables:
    """One recipe's declared variables, split when the split is knowable.

    A single probe compile decides it -- `text`'s own ``-- example:`` values
    substituted in, exactly as running the example would set them, tried once
    through :func:`_try_compile`. A recipe that compiles that way gets its
    usual split, one further compile per declared variable
    (:func:`required_variables`). A recipe that does not is never swept: a
    rejection off a recipe that fails outright says nothing about any one
    variable, and `failure` carries the compiler's own message instead, with
    both lists left empty -- N further failing compiles would only repeat
    what this one rejection already said.
    """

    required: tuple[Variable, ...]
    optional: tuple[Variable, ...]
    failure: str | None = None

    @property
    def compiles(self) -> bool:
        return self.failure is None


def recipe_variables(text: str, packages: PackageSet | None) -> RecipeVariables:
    """`text`'s declared variables, required split from optional -- or why that
    split cannot be known.

    Shared by ``ffrwd publish`` (the version document's per-recipe
    ``required``/``optional``) and ``ffrwd list`` (the same split, read
    without a network round trip): both want the identical rule -- a recipe
    that cannot compile even once must not claim a split nobody checked.

    Runs under :func:`_synthetic_probe` regardless of the caller: a recipe's
    own ``-- example:`` values are real-looking paths (``in.mkv``), not files
    on this machine, and neither reader is validating media -- only reading
    off what a recipe declares. Nested safely inside ``prepare``'s own
    synthetic-probe block, since the seam only ever redirects to the same
    synthetic answer.
    """
    declared = declared_variables(text)
    names = frozenset(variable.name for variable in declared)
    with _synthetic_probe():
        try:
            probe = substitute(text, _placeholders(text))
        except FfrwdError as err:
            return RecipeVariables(required=(), optional=(), failure=err.message)
        rejection = _try_compile(probe.text, packages, probe.unset)
        if rejection is not None:
            return RecipeVariables(required=(), optional=(), failure=rejection.message)
        required = required_variables(text, names, packages)
    return RecipeVariables(
        required=tuple(variable for variable in declared if variable.name in required),
        optional=tuple(variable for variable in declared if variable.name not in required),
    )


# --------------------------------------------------------------------------
# validating one package
# --------------------------------------------------------------------------


def _relative(path: Path, root: Path) -> str:
    """`path` under `root` as one written path, for a document or a message."""
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _description(manifest: Path) -> str:
    """What the manifest says the package is; empty when it says nothing.

    Read off the file rather than off the parsed package: a description is
    documentation, and the manifest reader keeps only what a compile needs.
    """
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError):  # the reader already refused anything unreadable
        return ""
    described = data.get("description", "") if isinstance(data, dict) else ""
    return described if isinstance(described, str) else ""


def _checked_name(package: Package) -> None:
    """The name rules, in publishing's own voice.

    The manifest reader applies the same two, so this cannot normally fire --
    and it is written out here anyway, because publishing is where a name
    stops being this machine's business and becomes everyone's.
    """
    if not is_package_name(package.name):
        raise _reject(
            f"{package.name!r} is not a package name",
            "a package name is <namespace>/<package>, each half a lowercase plain "
            "identifier",
        )
    refused = name_refusal(package.name)
    if refused is not None:
        raise _reject(refused[0], refused[1])


def _resolvable(package: Package) -> None:
    """Every dependency the manifest names, looked up in the registry.

    A published package's dependencies have to come from somewhere a consumer
    can reach: a name the registry does not have would install as a package
    that simply is not there.
    """
    for name in package.dependencies:
        try:
            packages_module.resolve(name)
        except FfrwdError as err:
            raise _reject(
                f"package '{package.name}' depends on '{name}', which the registry "
                f"could not resolve: {err.message}",
                "publish what it depends on first, or drop the dependency",
            ) from err


def _capabilities(package: Package) -> tuple[str, ...]:
    """What this package's modules need granted, checked against what it declares.

    The manifest's list and what the modules turn out to need are the same set
    or the publish is refused, in either direction: a module needing something
    undeclared would install without the grant it runs on, and a declared
    capability nothing uses asks a consumer to allow more than the package
    does.

    What is returned is the DERIVED set -- the same answer an install reads, so
    what the registry publishes and what a client provisions for cannot
    disagree.
    """
    by_module = packages_module.module_capabilities(package)
    declared = set(package.capabilities)
    for module, needed in sorted(by_module.items()):
        for capability in sorted(set(needed) - declared):
            raise _reject(
                f"package '{package.name}': the module '{module}' needs the "
                f"'{capability}' capability, which the manifest does not declare",
                f'declare it in the manifest\'s "capabilities": ["{capability}"]',
            )
    derived = {name for needed in by_module.values() for name in needed}
    for capability in sorted(declared - derived):
        raise _reject(
            f"package '{package.name}': the manifest declares the '{capability}' capability",
            'no module in this package needs it; drop it from "capabilities"',
        )
    return tuple(sorted(derived))


def _checked_models(package: Package) -> None:
    """Every model the manifest pins names an export one of its modules declares."""
    if not package.models:
        return
    exported = {declared.export for declared in package_modules(package)}
    for export in package.models:
        if export not in exported:
            raise _reject(
                f"package '{package.name}': the model for '{export}' names an export "
                "no module in this package declares",
                "a model is keyed by the export whose module loads it; "
                f"this package declares {', '.join(sorted(exported)) or 'none'}",
            )


def _model_sizes(package: Package) -> dict[str, tuple[int | None, ...]]:
    """What each pinned file weighs on the hub, in the order its export pins them.

    One request per repository and revision. A size the hub does not answer
    for is None and the publish carries on without it.
    """
    if not package.models:
        return {}
    weighed = packages_module.model_sizes(
        [pin for pins in package.models.values() for pin in pins]
    )
    return {
        export: tuple(weighed.get(pin) for pin in pins)
        for export, pins in package.models.items()
    }


def _pin_document(pin: ModelPin, size: int | None) -> Document:
    """One pin as the registry stores it: what it names, and what it weighs.

    A size the hub did not answer for is left out rather than written null.
    """
    written: Document = {
        "repo": pin.repo,
        "revision": pin.revision,
        "file": pin.file,
        "sha256": pin.sha256,
    }
    if size is not None:
        written["size"] = size
    return written


def _tested(package: Package, announce: Announce | None) -> None:
    """Run the manifest's own ``test`` command in place of compiling recipes as a check.

    Runs through the platform's own shell (``&&`` and a pipe mean what the
    author typed), inheriting the environment, with the package's root as its
    working directory. Its own stdout and stderr are left alone -- an
    author's test suite writes to the terminal a publish is run from, not
    somewhere this captures. Exit 0 continues; anything else is a rejection,
    and so is a shell that cannot even start.

    A manifest declaring no ``test`` is not checked, and nothing is compiled
    in its place.
    """
    if not package.test:
        return
    if announce is not None:
        announce(f"running {package.test}")
    try:
        result = subprocess.run(package.test, shell=True, cwd=package.root)
    except OSError as err:
        raise _reject(
            f"package '{package.name}': its test command {package.test!r} could not "
            f"be run: {err.strerror or err}",
            "\"test\" is the package's own check, declared in its manifest; fix the "
            "command and publish again",
        ) from err
    if result.returncode != 0:
        raise _reject(
            f"package '{package.name}': its test command {package.test!r} exited "
            f"{result.returncode}",
            "\"test\" is the package's own check, declared in its manifest; the "
            "publish stops until it passes",
        )


# --------------------------------------------------------------------------
# what a published version says about itself
# --------------------------------------------------------------------------


def _readme_html(package: Package) -> str | None:
    """The package's ``README.md`` rendered from CommonMark, or None when it has none.

    Plain CommonMark, raw HTML and all: the site runs DOMPurify over what it
    inserts, and that is the boundary. A publish payload is written by the
    client anyway, so sanitizing on this side would protect nobody -- what it
    would do is silently drop the markup an author wrote on purpose.
    """
    path = package.root / README_NAME
    if not path.is_file():
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as err:
        raise _reject(
            f"package '{package.name}': {README_NAME} could not be read: {err.strerror or err}",
            f"its file is {path}",
        ) from err
    return str(MarkdownIt("commonmark").render(text))


def _detail_document(
    package: Package,
    description: str,
    packages: PackageSet,
    sha256: str,
    size: int,
    readme_html: str | None,
    on_warning: OnWarning | None,
) -> Document:
    """This version, as the registry stores and serves it."""
    functions = [
        {
            "name": signature.name,
            "params": [{"name": param.name, "type": param.type} for param in signature.params],
            "returns": signature.returns,
            "written": signature.written,
            "export": _relative(signature.export, package.root),
        }
        for signature in package_signatures(package)
    ]
    recipes = []
    for name, path in package.recipes.items():
        text = path.read_text(encoding="utf-8")
        split = recipe_variables(text, packages)
        if not split.compiles and on_warning is not None:
            on_warning(
                FfrwdWarning(
                    code=WarningCode.RECIPE_DOES_NOT_COMPILE,
                    package=_qualified_recipe(package, name),
                    message=f"package '{package.name}': recipe '{name}' does not compile "
                    f"({split.failure}), so its variables are listed as not known",
                    hint="fix the recipe so it compiles, or drop it from the package",
                )
            )
        recipes.append(
            {
                "name": name,
                "file": _relative(path, package.root),
                "description": _recipe_description(text),
                "usage": _usage(package, name, text),
                "compiles": split.compiles,
                "required": [
                    {"name": variable.name, "description": variable.description}
                    for variable in split.required
                ],
                "optional": [
                    {"name": variable.name, "description": variable.description}
                    for variable in split.optional
                ],
            }
        )
    document: Document = {
        "version": package.version,
        "sha256": sha256,
        "size": size,
        "namespace": package.namespace,
        "description": description,
        "license": package.license,
        "functions": functions,
        "recipes": recipes,
    }
    if readme_html is not None:
        document["readme_html"] = readme_html
    return document


def _sources(package: Package) -> dict[str, str]:
    """A package's recipes, by name, as their own SQL text -- for the site to show."""
    return {name: path.read_text(encoding="utf-8") for name, path in package.recipes.items()}


# --------------------------------------------------------------------------
# the preflight
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Prepared:
    """One package, validated and packed: everything the upload sends.

    `detail` is what a consumer reads back off the registry for this version,
    `sources` the recipe text the site shows, and `capabilities` what the
    sidecar said this package's modules do.
    """

    package: Package
    archive: bytes
    sha256: str
    detail: Document
    sources: dict[str, str]
    capabilities: tuple[str, ...]
    visibility: str = PUBLIC
    model_sizes: Mapping[str, tuple[int | None, ...]] = field(default_factory=dict)

    @property
    def size(self) -> int:
        return len(self.archive)

    def metadata(self) -> Document:
        """The JSON the upload carries beside the archive."""
        return {
            "name": self.package.name,
            "version": self.package.version,
            "sha256": self.sha256,
            "visibility": self.visibility,
            "detail": self.detail,
            "sources": self.sources,
            "capabilities": list(self.capabilities),
            "engines": self.package.engines,
            "keywords": list(self.package.keywords),
            "license": self.package.license,
            "homepage": self.package.homepage,
            "models": {
                export: self._pinned(export, pins)
                for export, pins in self.package.models.items()
            },
        }

    def _pinned(self, export: str, pins: tuple[ModelPin, ...]) -> Document | list[Document]:
        """One export's pins, array-shaped when it names more than one file."""
        sizes = self.model_sizes.get(export, ())
        written = [
            _pin_document(pin, sizes[number] if number < len(sizes) else None)
            for number, pin in enumerate(pins)
        ]
        return written[0] if len(written) == 1 else written


def prepare(
    manifest: Path,
    packages: PackageSet | None = None,
    *,
    on_warning: OnWarning | None = None,
    announce: Announce | None = None,
) -> Prepared:
    """Validate the package `manifest` declares, and pack it. Reaches the registry
    to resolve what the package depends on, and the hub for what each pinned
    model file weighs.

    In order: the manifest reads and its name is one a package may have; every
    dependency resolves; every export parses and defines what the manifest
    says; every module describes, and every pinned model names one of their
    exports; the package's own test command runs, when its manifest declares
    one. The hub is asked what each pinned file weighs, and a size that does
    not arrive is left out rather than refused over. Then the directory is
    packed, and the documents the registry stores are built out of the same
    package the checks just ran over.

    `packages` is what a recipe's calls resolve against -- the project's own
    installed set. Defaults to the package alone, which is enough for one that
    depends on nothing. The version's visibility is the manifest's
    ``private`` key: true publishes it private, absent or false public.
    `on_warning` hears what does not stop a publish: a missing README.md, a
    manifest declaring no license, and an ignore line the packer skipped.
    `announce` hears one line per step: the validation, and the packing.
    """
    package = read_manifest(manifest)
    _checked_name(package)
    if announce is not None:
        announce(f"validating {package.name} {package.version}")
    _resolvable(package)
    resolvable = packages if packages is not None else PackageSet(
        root=package.root, packages={package.name: package}
    )
    with _synthetic_probe():
        package_signatures(package)
        capabilities = _capabilities(package)
        _checked_models(package)
        sizes = _model_sizes(package)
        _tested(package, announce)

        readme_html = _readme_html(package)
        if readme_html is None and on_warning is not None:
            on_warning(
                FfrwdWarning(
                    code=WarningCode.MISSING_README,
                    package=package.name,
                    message=f"package '{package.name}' carries no {README_NAME}, so its "
                    "page on the registry has nothing to show",
                    hint="write one beside the manifest; it ships in the archive",
                )
            )
        if package.license is None and on_warning is not None:
            on_warning(
                FfrwdWarning(
                    code=WarningCode.MISSING_LICENSE,
                    package=package.name,
                    message=f'package \'{package.name}\' declares no "license", so its '
                    "page on the registry cannot say how it may be used",
                    hint='add "license" to the manifest, e.g. "MIT"',
                )
            )
        if announce is not None:
            announce(f"packing {package.root}")
        try:
            archive = store.pack(package.root, on_warning=on_warning)
        except OSError as err:
            raise _reject(
                f"package '{package.name}' could not be packed: {err.strerror or err}",
                f"check that everything under {package.root} is readable",
            ) from err
        if len(archive) > _MAX_ARCHIVE_BYTES:
            raise _reject(
                f"package '{package.name}' packs to {len(archive)} bytes, and at most "
                f"{_MAX_ARCHIVE_BYTES} are published",
                'a model belongs in the manifest\'s "models", not in the package '
                "directory; install fetches it from the hub",
            )
        sha256 = hashlib.sha256(archive).hexdigest()
        detail = _detail_document(
            package,
            _description(manifest),
            resolvable,
            sha256,
            len(archive),
            readme_html,
            on_warning,
        )
        sources = _sources(package)
    return Prepared(
        package=package,
        archive=archive,
        sha256=sha256,
        detail=detail,
        sources=sources,
        capabilities=capabilities,
        visibility=PRIVATE if package.private else PUBLIC,
        model_sizes=sizes,
    )


# --------------------------------------------------------------------------
# the upload
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Published:
    """What the registry said it did with one publish."""

    name: str
    version: str
    sha256: str
    size: int
    visibility: str


_METADATA_PART = "metadata"
_ARCHIVE_PART = "archive"


def _multipart(metadata: Document, archive: bytes, filename: str) -> tuple[str, bytes]:
    """The two parts as one body: the content type, and the bytes.

    The boundary is random, and both parts are checked against it before
    anything is joined -- a body whose own content could open a part is not a
    body this sends.
    """
    boundary = f"ffrwd{os.urandom(16).hex()}"
    # Compressed, like the archive beside it: the detail document is full of
    # SQL and HTML -- ordinary package documentation that proxies between
    # here and the registry misread in the plain. mtime=0 keeps the bytes
    # deterministic. The registry reads plain JSON too.
    written = gzip.compress(
        json.dumps(metadata, ensure_ascii=False).encode("utf-8"), mtime=0
    )
    marker = f"--{boundary}".encode()
    if marker in written or marker in archive:  # pragma: no cover -- 16 random bytes
        raise _reject(
            "the upload could not be framed",
            "run the command again; the framing marker collided with the content",
        )
    body = b"".join(
        [
            marker,
            b'\r\nContent-Disposition: form-data; name="',
            _METADATA_PART.encode(),
            b'"; filename="metadata.json.gz"'
            b"\r\nContent-Type: application/gzip\r\n\r\n",
            written,
            b"\r\n",
            marker,
            b'\r\nContent-Disposition: form-data; name="',
            _ARCHIVE_PART.encode(),
            b'"; filename="',
            filename.encode(),
            b'"\r\nContent-Type: application/octet-stream\r\n\r\n',
            archive,
            b"\r\n",
            marker,
            b"--\r\n",
        ]
    )
    return f"multipart/form-data; boundary={boundary}", body


def _refusal(status: int, body: bytes) -> FfrwdError:
    """The registry's own refusal, rendered the way every other rejection reads."""
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
        message = f"the registry refused the publish with HTTP {status}"
    if not hint:
        hint = "the registry said no more than that; try again, or report it"
    return _reject(message, hint)


def _published(prepared: Prepared, body: bytes, where: str) -> Published:
    """What the registry answered with, or a rejection naming what it sent instead."""
    try:
        data = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError, RecursionError) as err:
        raise _reject(
            f"{where} answered with something that is not JSON",
            "the package may or may not have been published; run `ffrwd search` to see",
        ) from err
    if not isinstance(data, dict):
        raise _reject(
            f"{where} answered with something that is not a JSON object",
            "the package may or may not have been published; run `ffrwd search` to see",
        )
    visibility = data.get("visibility")
    return Published(
        name=prepared.package.name,
        version=prepared.package.version,
        sha256=prepared.sha256,
        size=prepared.size,
        # The registry's answer, not the request's: a package it already holds
        # is stamped with its OWN default, whatever this publish asked for.
        visibility=visibility if isinstance(visibility, str) else prepared.visibility,
    )


def publish(prepared: Prepared, *, announce: Announce | None = None) -> Published:
    """Upload `prepared` to the registry. One request, and the token has to be there.

    A refusal is the registry's own message and hint. A version already
    published under these exact bytes is not one: republishing what is already
    there succeeds and changes nothing. `announce` hears the one line naming
    the upload.
    """
    token = credentials.load()
    if token is None:
        raise _reject(
            "publishing needs an ffrwd token, and this machine holds none", _LOGIN_HINT
        )
    if announce is not None:
        announce(
            f"uploading {prepared.package.name} {prepared.package.version} "
            f"({written_size(prepared.size)})"
        )
    url = f"{packages_module.api_url().rstrip('/')}/functions/v1/publish"
    filename = f"{prepared.package.name.replace('/', '-')}-{prepared.package.version}.tgz"
    content_type, body = _multipart(prepared.metadata(), prepared.archive, filename)
    status, answered = packages_module.exchange(
        url,
        headers={"Authorization": f"Bearer {token}", "Content-Type": content_type},
        data=body,
        limit=_MAX_RESPONSE_BYTES,
    )
    if status >= 400:
        raise _refusal(status, answered)
    return _published(prepared, answered, url)
