"""The ONNX Runtime a module's inference needs, fetched on first use.

A module that runs a model needs a native ONNX Runtime beside it, and the
sidecar has no link-time dependency on one: it loads whatever the directory
named by ``-nn-runtime`` holds. That directory is what this module fills.

Three things live here. :func:`info` asks the sidecar which runtime IT
demands, so nothing has to be written down twice. :func:`ensure` puts that
runtime under the cache directory, fetching only what is missing.
:func:`spawn_args` is the ``-nn-runtime``/``-nn-target`` pair a spawned
sidecar is told.

The version comes from the binary, never from a constant here. ``ort`` refuses
a library whose version does not match what it was built against, naming both,
so a bootstrap that guessed would produce a refusal rather than a mismatch --
and ``ffrwd-wasm --nn-info`` removes the guess.

Layout, mirroring what the sidecar resolves::

    ~/.cache/ffrwd/nn-runtime/1.22.0/win-x64/onnxruntime.dll
    ~/.cache/ffrwd/nn-runtime/1.22.0/win-x64/onnxruntime_providers_cuda.dll
    ~/.cache/ffrwd/nn-runtime/1.22.0/win-x64/directml/onnxruntime.dll
    ~/.cache/ffrwd/nn-runtime/1.22.0/win-x64/directml/DirectML.dll

Version-scoped, because a sidecar upgrade must not read the set fetched for
the version before it; and under the cache, because every byte here is
redownloadable, unlike a token or a lockfile.

What is fetched is per platform, and the point is that a run never has to ask
for it. Windows takes the CPU library and the DirectML pair -- DirectML runs
on any Direct3D 12 adapter and asks for nothing installed. The CUDA tier is
five times the size and needs a CUDA 12 runtime and cuDNN 9 on the machine, so
on Windows it is only ever fetched by ``ffrwd setup nn --cuda``. Linux takes
the CPU library, and the CUDA provider beside it when an NVIDIA driver is
there -- CUDA is the only accelerator Linux has. macOS takes the stock
archive, which carries CoreML inside it.

Every artifact is pinned: a URL, a sha256, and its exact byte count, per
runtime version and platform. The download stops at the first block past that
count, the archive is verified whole before anything is opened, each library
is written through a temporary file and moved onto its name, and a tier lands
completely or not at all.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO

from . import binaries
from .errors import ErrorCode, FfrwdError

# What a fetch tells the caller, so nothing here decides where a line prints.
Announce = Callable[[str], None]

__all__ = [
    "Info",
    "SETUP_HINT",
    "ensure",
    "info",
    "provision",
    "runtime_dir",
    "spawn_args",
    "wanted_tiers",
]

# The environment variable the sidecar reads for the runtime directory. Set,
# it is the developer's own directory and nothing here provisions or spells
# one.
RUNTIME_DIR_VAR = "FFRWD_NN_RUNTIME"

# The environment variable the sidecar reads for the target. Set, it is the
# answer and no default is spelled.
TARGET_VAR = "FFRWD_NN_TARGET"

_NN_RUNTIME_FLAG = "-nn-runtime"
_NN_TARGET_FLAG = "-nn-target"

SETUP_HINT = "run `ffrwd setup nn` once with a network connection"

_TIMEOUT = 60.0

# How long the sidecar gets to answer --nn-info. It reads no model and opens
# no runtime, so this is process startup and nothing else.
_INFO_TIMEOUT = 30.0


def _reject(message: str, hint: str) -> FfrwdError:
    return FfrwdError(ErrorCode.UNSUPPORTED_SQL, message, hint=hint)


# --------------------------------------------------------------------------
# what the sidecar demands
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Info:
    """What ``ffrwd-wasm --nn-info`` answers.

    `ort_version` is the ONNX Runtime the sidecar's own crate demands,
    `providers` the order ``-nn-target gpu`` walks on this platform, and
    `platform` the operating system and architecture key a fetched runtime is
    filed under.
    """

    ort_version: str
    providers: tuple[str, ...]
    platform: str


_INFO: Info | None = None


def info() -> Info:
    """Ask the sidecar what runtime it demands, once per process.

    Raises ``FfrwdError`` -- and nothing else -- when the sidecar is not
    installed, will not run, or answers with something that is not an info
    document.
    """
    global _INFO
    if _INFO is None:
        _INFO = _read_info()
    return _INFO


def _read_info() -> Info:
    sidecar = binaries.ffrwd_wasm_path()
    if sidecar is None:
        raise _reject(
            "the ffrwd-wasm sidecar is not installed, and a query that runs a "
            "model needs it to say which ONNX Runtime to fetch",
            hint=binaries.INSTALL_HINT,
        )
    try:
        done = subprocess.run(
            [sidecar, "--nn-info"],
            capture_output=True,
            text=True,
            timeout=_INFO_TIMEOUT,
            check=False,
        )
    except OSError as err:
        raise _reject(
            f"could not run the ffrwd-wasm sidecar at {sidecar}: {err.strerror or err}",
            hint=binaries.INSTALL_HINT,
        ) from err
    except subprocess.TimeoutExpired as err:
        raise _reject(
            f"the ffrwd-wasm sidecar did not answer --nn-info within {_INFO_TIMEOUT:.0f}s",
            hint=binaries.INSTALL_HINT,
        ) from err
    if done.returncode != 0:
        raise _reject(
            f"the ffrwd-wasm sidecar at {sidecar} does not answer --nn-info",
            hint="it is older than this ffrwd; install the two together",
        )
    try:
        payload = json.loads(done.stdout)
    except ValueError as err:
        raise _reject(
            "the ffrwd-wasm sidecar's answer to --nn-info is not JSON",
            hint="the sidecar found may be a different version than this ffrwd",
        ) from err
    return _info_document(payload)


def _info_document(payload: object) -> Info:
    if not isinstance(payload, dict):
        raise _reject(
            "the ffrwd-wasm sidecar's answer to --nn-info is not an object",
            hint="the sidecar found may be a different version than this ffrwd",
        )
    version = payload.get("ort_version")
    platform = payload.get("platform")
    providers = payload.get("providers")
    if not isinstance(version, str) or not version:
        raise _reject(
            "the ffrwd-wasm sidecar's --nn-info names no ort_version",
            hint="the sidecar found may be a different version than this ffrwd",
        )
    if not isinstance(platform, str) or not platform:
        raise _reject(
            "the ffrwd-wasm sidecar's --nn-info names no platform",
            hint="the sidecar found may be a different version than this ffrwd",
        )
    named = (
        tuple(p for p in providers if isinstance(p, str))
        if isinstance(providers, list)
        else ()
    )
    return Info(ort_version=version, providers=named, platform=platform)


# --------------------------------------------------------------------------
# where it goes
# --------------------------------------------------------------------------


def _cache_dir() -> Path:
    try:
        return Path.home() / ".cache" / "ffrwd"
    except RuntimeError:  # pragma: no cover -- no resolvable home directory
        return Path(tempfile.gettempdir()) / "ffrwd-cache"


def runtime_root() -> Path:
    """The directory every fetched runtime sits under."""
    return _cache_dir() / "nn-runtime"


def runtime_dir(found: Info | None = None) -> Path:
    """Where THIS sidecar's runtime belongs: version, then platform."""
    known = found if found is not None else info()
    return runtime_root() / known.ort_version / known.platform


# --------------------------------------------------------------------------
# what is pinned
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Member:
    """One library taken out of an archive.

    `entry` is its full path inside the archive -- full rather than a leaf
    because the nuget packages carry the same leaf for every architecture.
    `name` is what it is called under the runtime directory and `subdir`
    where, empty for the directory itself. `aliases` are further names the
    same library answers to: the sonames a Linux loader looks for, written as
    links to `name` where the platform has them and as copies where it does
    not.
    """

    entry: str
    name: str
    subdir: str = ""
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class Artifact:
    """One archive: where it comes from, what it must hash to, and what it holds.

    `size` is its exact published byte count. A download is abandoned at the
    first block past it, so an answer that is not this artifact costs one
    block over the pin rather than a disk.
    """

    url: str
    sha256: str
    size: int
    members: tuple[Member, ...] = field(default_factory=tuple)


_ORT_RELEASES = "https://github.com/microsoft/onnxruntime/releases/download"
_NUGET = "https://api.nuget.org/v3-flatcontainer"
_NVIDIA = "https://developer.download.nvidia.com/compute"

# What DirectML the sidecar's ONNX Runtime was built against. Windows ships a
# DirectML of its own and a recent one answers, but the version the runtime
# names is the one that is certain to.
_DML_VERSION = "1.15.4"

# Every artifact, by (ONNX Runtime version, platform) and then by tier. A
# version the sidecar demands and this table does not carry is a refusal
# naming both -- never a fetch of the nearest thing.
_PINS: Mapping[tuple[str, str], Mapping[str, tuple[Artifact, ...]]] = {
    ("1.22.0", "win-x64"): {
        "cpu": (
            Artifact(
                url=f"{_ORT_RELEASES}/v1.22.0/onnxruntime-win-x64-1.22.0.zip",
                sha256="174c616efc0271194488642a72f1a514e01487da4dfe84c49296d66e40ebe0da",
                size=72368545,
                members=(
                    Member(
                        entry="onnxruntime-win-x64-1.22.0/lib/onnxruntime.dll",
                        name="onnxruntime.dll",
                    ),
                ),
            ),
        ),
        # The DirectML build of onnxruntime.dll is a different binary carrying
        # the same name as the CPU and CUDA one, so it gets a directory of its
        # own rather than overwriting them.
        "directml": (
            Artifact(
                url=f"{_NUGET}/microsoft.ml.onnxruntime.directml/1.22.0"
                "/microsoft.ml.onnxruntime.directml.1.22.0.nupkg",
                sha256="29f9872d786236b79aa83f94482f3a17c14297e4833768d6d0ed4883ee732e60",
                size=17898472,
                members=(
                    Member(
                        entry="runtimes/win-x64/native/onnxruntime.dll",
                        name="onnxruntime.dll",
                        subdir="directml",
                    ),
                ),
            ),
            Artifact(
                url=f"{_NUGET}/microsoft.ai.directml/{_DML_VERSION}"
                f"/microsoft.ai.directml.{_DML_VERSION}.nupkg",
                sha256="4e7cb7ddce8cf837a7a75dc029209b520ca0101470fcdf275c1f49736a3615b9",
                size=202292617,
                members=(
                    Member(
                        entry="bin/x64-win/DirectML.dll",
                        name="DirectML.dll",
                        subdir="directml",
                    ),
                ),
            ),
        ),
        # The CUDA build's own onnxruntime.dll comes with it: the provider is
        # built against that binary, not against the CPU one it replaces.
        "cuda": (
            Artifact(
                url=f"{_ORT_RELEASES}/v1.22.0/onnxruntime-win-x64-gpu-1.22.0.zip",
                sha256="5b5241716b2628c1ab5e79ee620be767531021149ee68f30fc46c16263fb94dd",
                size=312740706,
                members=(
                    Member(
                        entry="onnxruntime-win-x64-gpu-1.22.0/lib/onnxruntime.dll",
                        name="onnxruntime.dll",
                    ),
                    Member(
                        entry="onnxruntime-win-x64-gpu-1.22.0/lib"
                        "/onnxruntime_providers_shared.dll",
                        name="onnxruntime_providers_shared.dll",
                    ),
                    Member(
                        entry="onnxruntime-win-x64-gpu-1.22.0/lib"
                        "/onnxruntime_providers_cuda.dll",
                        name="onnxruntime_providers_cuda.dll",
                    ),
                ),
            ),
        ),
        # The CUDA 12 runtime and cuDNN 9 belong to the machine, like the
        # driver. This is the escape hatch for one that has neither.
        "full": (
            Artifact(
                url=f"{_NVIDIA}/cuda/redist/cuda_cudart/windows-x86_64"
                "/cuda_cudart-windows-x86_64-12.9.79-archive.zip",
                sha256="179e9c43b0735ffe67207b3da556eb5a0c50f3047961882b7657d3b822d34ef8",
                size=3521238,
                members=(
                    Member(
                        entry="cuda_cudart-windows-x86_64-12.9.79-archive/bin/cudart64_12.dll",
                        name="cudart64_12.dll",
                    ),
                ),
            ),
            Artifact(
                url=f"{_NVIDIA}/cuda/redist/libcublas/windows-x86_64"
                "/libcublas-windows-x86_64-12.9.1.4-archive.zip",
                sha256="d534d98b0b453a98914dbf3adf47d7e84b55037abf02f87466439e1dcef581ed",
                size=549755186,
                members=(
                    Member(
                        entry="libcublas-windows-x86_64-12.9.1.4-archive/bin/cublas64_12.dll",
                        name="cublas64_12.dll",
                    ),
                    Member(
                        entry="libcublas-windows-x86_64-12.9.1.4-archive/bin/cublasLt64_12.dll",
                        name="cublasLt64_12.dll",
                    ),
                ),
            ),
            Artifact(
                url=f"{_NVIDIA}/cuda/redist/libcufft/windows-x86_64"
                "/libcufft-windows-x86_64-11.4.1.4-archive.zip",
                sha256="f26f80bb9abff3269c548e1559e8c2b4ba58ccb8acc6095bbc6404fc962d4b80",
                size=198361265,
                members=(
                    Member(
                        entry="libcufft-windows-x86_64-11.4.1.4-archive/bin/cufft64_11.dll",
                        name="cufft64_11.dll",
                    ),
                ),
            ),
            # cudnn64_9 loads the rest itself, by name, out of this directory.
            # The graph, ops and engine libraries are what an ONNX session
            # reaches; the convolution and attention ones it does not.
            Artifact(
                url=f"{_NVIDIA}/cudnn/redist/cudnn/windows-x86_64"
                "/cudnn-windows-x86_64-9.10.2.21_cuda12-archive.zip",
                sha256="c1a4567d822ebda7373fa1f19255dff4942302de741f830160b6c7d1fb31af23",
                size=683336095,
                members=tuple(
                    Member(
                        entry=f"cudnn-windows-x86_64-9.10.2.21_cuda12-archive/bin/{lib}",
                        name=lib,
                    )
                    for lib in (
                        "cudnn64_9.dll",
                        "cudnn_graph64_9.dll",
                        "cudnn_ops64_9.dll",
                        "cudnn_heuristic64_9.dll",
                        "cudnn_engines_precompiled64_9.dll",
                        "cudnn_engines_runtime_compiled64_9.dll",
                    )
                ),
            ),
        ),
    },
    # The tgz members are the real files; the unversioned names the archive
    # carries are symlinks to them, and so are the ones written here.
    ("1.22.0", "linux-x64"): {
        "cpu": (
            Artifact(
                url=f"{_ORT_RELEASES}/v1.22.0/onnxruntime-linux-x64-1.22.0.tgz",
                sha256="8344d55f93d5bc5021ce342db50f62079daf39aaafb5d311a451846228be49b3",
                size=7798730,
                members=(
                    Member(
                        entry="onnxruntime-linux-x64-1.22.0/lib/libonnxruntime.so.1.22.0",
                        name="libonnxruntime.so",
                        aliases=("libonnxruntime.so.1", "libonnxruntime.so.1.22.0"),
                    ),
                ),
            ),
        ),
        "cuda": (
            Artifact(
                url=f"{_ORT_RELEASES}/v1.22.0/onnxruntime-linux-x64-gpu-1.22.0.tgz",
                sha256="2a19dbfa403672ec27378c3d40a68f793ac7a6327712cd0e8240a86be2b10c55",
                size=227317516,
                members=(
                    Member(
                        entry="onnxruntime-linux-x64-gpu-1.22.0/lib/libonnxruntime.so.1.22.0",
                        name="libonnxruntime.so",
                        aliases=("libonnxruntime.so.1", "libonnxruntime.so.1.22.0"),
                    ),
                    Member(
                        entry="onnxruntime-linux-x64-gpu-1.22.0/lib"
                        "/libonnxruntime_providers_shared.so",
                        name="libonnxruntime_providers_shared.so",
                    ),
                    Member(
                        entry="onnxruntime-linux-x64-gpu-1.22.0/lib"
                        "/libonnxruntime_providers_cuda.so",
                        name="libonnxruntime_providers_cuda.so",
                    ),
                ),
            ),
        ),
    },
    ("1.22.0", "osx-arm64"): {
        "cpu": (
            Artifact(
                url=f"{_ORT_RELEASES}/v1.22.0/onnxruntime-osx-arm64-1.22.0.tgz",
                sha256="cab6dcbd77e7ec775390e7b73a8939d45fec3379b017c7cb74f5b204c1a1cc07",
                size=25943843,
                members=(
                    Member(
                        entry="onnxruntime-osx-arm64-1.22.0/lib/libonnxruntime.1.22.0.dylib",
                        name="libonnxruntime.dylib",
                        aliases=("libonnxruntime.1.22.0.dylib",),
                    ),
                ),
            ),
        ),
    },
    ("1.22.0", "osx-x86_64"): {
        "cpu": (
            Artifact(
                url=f"{_ORT_RELEASES}/v1.22.0/onnxruntime-osx-x86_64-1.22.0.tgz",
                sha256="e4ec94a7696de74fb1b12846569aa94e499958af6ffa186022cfde16c9d617f0",
                size=27889590,
                members=(
                    Member(
                        entry="onnxruntime-osx-x86_64-1.22.0/lib/libonnxruntime.1.22.0.dylib",
                        name="libonnxruntime.dylib",
                        aliases=("libonnxruntime.1.22.0.dylib",),
                    ),
                ),
            ),
        ),
    },
}

# The order tiers are applied in. The CUDA build supplies its own
# onnxruntime.dll, so it goes after the CPU tier that would otherwise be
# overwritten by it rather than the other way round.
_TIER_ORDER = ("cpu", "directml", "cuda", "full")

# What each tier is called in the one line a fetch prints.
_TIER_NAMES = {
    "cpu": "the CPU provider",
    "directml": "the DirectML provider",
    "cuda": "the CUDA provider",
    "full": "the CUDA 12 and cuDNN 9 libraries",
}


def _table(found: Info) -> Mapping[str, tuple[Artifact, ...]]:
    """The artifacts pinned for this sidecar's runtime on this platform."""
    pinned = _PINS.get((found.ort_version, found.platform))
    if pinned is None:
        raise _reject(
            f"no ONNX Runtime {found.ort_version} is pinned for {found.platform}, "
            "and the sidecar demands exactly that one",
            hint="this ffrwd is older than the sidecar it found; install the two "
            "together, or point FFRWD_NN_RUNTIME at a runtime of your own",
        )
    return pinned


# --------------------------------------------------------------------------
# which tiers this machine wants
# --------------------------------------------------------------------------


def nvidia_driver() -> bool:
    """Whether this machine looks like it has an NVIDIA driver.

    Deliberately shallow: the kernel module's own directory under /proc, or
    the tool the driver installs on PATH. Either is enough to decide whether
    downloading a CUDA provider is worth the bytes, and a wrong answer costs
    a fetch or a fall back to the CPU, never a wrong result.
    """
    if Path("/proc/driver/nvidia").exists():
        return True
    return shutil.which("nvidia-smi") is not None


def wanted_tiers(found: Info | None = None) -> tuple[str, ...]:
    """What a run that reaches a model fetches on this platform, unasked.

    Windows takes DirectML, which runs on any Direct3D 12 adapter; its CUDA
    tier is an explicit `ffrwd setup nn --cuda` and never lands here. Linux
    takes CUDA when a driver is there, since it is the only accelerator it
    has. macOS takes the stock archive alone -- CoreML is inside it.
    """
    known = found if found is not None else info()
    if known.platform.startswith("win-"):
        return ("cpu", "directml")
    if known.platform.startswith("linux-") and nvidia_driver():
        return ("cpu", "cuda")
    return ("cpu",)


def default_target(found: Info | None = None) -> str:
    """The ``-nn-target`` a spawned sidecar is given when nothing else says.

    ``gpu`` wherever an accelerator tier is provisioned for: a Direct3D 12
    adapter on Windows, which every recent machine has; an NVIDIA driver on
    Linux; CoreML on macOS, which is part of the operating system. The
    sidecar names every provider it passes over on the way, and falls back to
    the CPU rather than refusing, so this is a preference and not a claim.
    """
    known = found if found is not None else info()
    if known.platform.startswith("linux-"):
        return "gpu" if nvidia_driver() else "cpu"
    return "gpu"


def _present(directory: Path, artifacts: Sequence[Artifact]) -> bool:
    """Whether every library a tier's artifacts hold is already on disk."""
    return all(
        _placed(directory, member).is_file()
        for artifact in artifacts
        for member in artifact.members
    )


def _placed(directory: Path, member: Member) -> Path:
    return directory / member.subdir / member.name if member.subdir else directory / member.name


def missing_tiers(tiers: Sequence[str], found: Info | None = None) -> tuple[str, ...]:
    """Which of `tiers` this machine does not already hold, in the order they apply."""
    known = found if found is not None else info()
    pinned = _table(known)
    directory = runtime_dir(known)
    return tuple(
        tier
        for tier in _TIER_ORDER
        if tier in tiers and not _present(directory, pinned.get(tier, ()))
    )


# --------------------------------------------------------------------------
# fetching
# --------------------------------------------------------------------------

# Every request this module makes goes through here. It is the seam the unit
# tier replaces: no check in it opens a socket, and what it replaces behaves
# the way urllib does.
_urlopen = urllib.request.urlopen


def _unreachable(url: str, reason: str) -> FfrwdError:
    return _reject(
        f"the ONNX Runtime could not be downloaded from {url}: {reason}",
        "check the network connection, or point FFRWD_NN_RUNTIME at a runtime "
        "already on this machine",
    )


def _mismatched(url: str, found: str, expected: str) -> FfrwdError:
    return _reject(
        f"{url} hashes to {found}, and {expected} was expected; nothing was written",
        "the download was not the artifact this ffrwd pins; try again, and report "
        "it if it keeps happening",
    )


def _download(artifact: Artifact, into: Path) -> Path:
    """Fetch and verify one archive into `into`, returning where it landed.

    Streamed to disk rather than held: the largest pinned artifact is most of
    a gigabyte. Hashed on the way through and thrown away unopened when the
    digest disagrees, so nothing unverified reaches an extractor.
    """
    handle, temporary = tempfile.mkstemp(dir=into, prefix="archive-", suffix=".tmp")
    try:
        with os.fdopen(handle, "wb") as file:
            found, oversized = _stream(artifact, file)
        if oversized:
            raise _unreachable(
                artifact.url, f"it is longer than the {artifact.size} bytes pinned for it"
            )
        if found != artifact.sha256:
            raise _mismatched(artifact.url, found, artifact.sha256)
    except BaseException:
        _discard(Path(temporary))
        raise
    return Path(temporary)


def _stream(artifact: Artifact, file: IO[bytes]) -> tuple[str, bool]:
    """Copy the artifact into `file`, hashing it: the digest, and whether it ran long.

    Abandoned at the block that crosses the pinned size rather than after the
    whole answer has been written.
    """
    digest = hashlib.sha256()
    written = 0
    try:
        with _urlopen(urllib.request.Request(artifact.url), timeout=_TIMEOUT) as response:
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    return digest.hexdigest(), False
                written += len(block)
                if written > artifact.size:
                    return digest.hexdigest(), True
                digest.update(block)
                file.write(block)
    except urllib.error.HTTPError as err:
        raise _unreachable(artifact.url, f"HTTP {err.code}") from err
    except (OSError, ValueError, urllib.error.URLError) as err:
        reason = getattr(err, "reason", None)
        raise _unreachable(artifact.url, str(reason or err)) from err


def _discard(path: Path) -> None:
    try:
        path.unlink()
    except OSError:  # already moved onto its name, or already gone
        pass


@contextmanager
def _opened(archive: Path) -> Iterator[Callable[[Member], IO[bytes]]]:
    """`archive` open once, with a way to read any member out of it by full path.

    Full path inside the archive, never a leaf: the nuget packages carry one
    library per architecture under the same name. Opened once because a tgz
    is decompressed from the front for every member read out of it.
    """
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zipped:

            def from_zip(member: Member) -> IO[bytes]:
                try:
                    return zipped.open(zipped.getinfo(member.entry))
                except KeyError as err:
                    raise _incomplete(member) from err

            yield from_zip
        return
    with tarfile.open(archive, "r:*") as tarred:

        def from_tar(member: Member) -> IO[bytes]:
            try:
                inside = tarred.extractfile(tarred.getmember(member.entry))
            except KeyError as err:
                raise _incomplete(member) from err
            if inside is None:
                raise _incomplete(member)
            return inside

        yield from_tar


def _incomplete(member: Member) -> FfrwdError:
    return _reject(
        f"the archive holds no {member.entry}, and the runtime needs it",
        "the artifact this ffrwd pins has changed shape; install a newer ffrwd",
    )


def _extract(
    archive: Path, artifact: Artifact, directory: Path, placed: list[tuple[Path, Path]]
) -> None:
    """Write every library `artifact` holds beside its destination, unnamed yet.

    Each is appended to `placed` as a (temporary, destination) pair, the
    caller's to move or to throw away. Nothing is moved onto a name here: a
    tier is placed only once every one of its libraries has come out, so a
    failure part way through cannot leave a half-set that later dlopens.
    """
    with _opened(archive) as reader:
        for member in artifact.members:
            destination = _placed(directory, member)
            destination.parent.mkdir(parents=True, exist_ok=True)
            handle, temporary = tempfile.mkstemp(
                dir=destination.parent, prefix=f"{destination.name}-", suffix=".tmp"
            )
            placed.append((Path(temporary), destination))
            with os.fdopen(handle, "wb") as file, reader(member) as inside:
                for block in iter(lambda: inside.read(1024 * 1024), b""):
                    file.write(block)


def _place(pairs: Sequence[tuple[Path, Path]]) -> None:
    """Move each extracted library onto its name."""
    for temporary, destination in pairs:
        os.replace(temporary, destination)


def _link_aliases(directory: Path, artifacts: Sequence[Artifact]) -> None:
    """Give each library the further names its loader may look for.

    A symlink where the platform has them, which is the layout the archive
    itself ships; a copy where it does not, so the name resolves either way.
    """
    for artifact in artifacts:
        for member in artifact.members:
            target = _placed(directory, member)
            for alias in member.aliases:
                beside = target.parent / alias
                _discard(beside)
                try:
                    beside.symlink_to(target.name)
                except (OSError, NotImplementedError):
                    shutil.copyfile(target, beside)


def _fetch_tier(directory: Path, tier: str, artifacts: Sequence[Artifact]) -> None:
    """Put one tier on disk, completely or not at all."""
    directory.mkdir(parents=True, exist_ok=True)
    pending: list[tuple[Path, Path]] = []
    archives: list[Path] = []
    try:
        for artifact in artifacts:
            archive = _download(artifact, directory)
            archives.append(archive)
            _extract(archive, artifact, directory, pending)
    except BaseException:
        for temporary, _ in pending:
            _discard(temporary)
        raise
    finally:
        for archive in archives:
            _discard(archive)
    _place(pending)
    _link_aliases(directory, artifacts)


def _size_of(artifacts: Sequence[Artifact]) -> int:
    return sum(artifact.size for artifact in artifacts)


def _megabytes(size: int) -> str:
    return f"{size / (1024 * 1024):.0f} MB"


def _notice(found: Info, tiers: Sequence[str], size: int) -> str:
    """The one line a fetch prints: what, how big, and that it happens once."""
    what = " and ".join(_TIER_NAMES.get(tier, tier) for tier in tiers)
    return (
        f"[nn] fetching ONNX Runtime {found.ort_version} for {found.platform} "
        f"({what}, {_megabytes(size)}) -- once, into {runtime_dir(found)}"
    )


def provision(
    tiers: Sequence[str],
    *,
    announce: Announce | None = None,
    found: Info | None = None,
) -> Path:
    """Put every tier of `tiers` under the runtime directory, and return it.

    A tier already on disk is left alone and costs nothing. `announce` is
    called with the one line a fetch prints, and not called at all when
    nothing had to be fetched.
    """
    known = found if found is not None else info()
    pinned = _table(known)
    for tier in tiers:
        if tier not in pinned:
            raise _reject(
                f"there is no {tier} ONNX Runtime for {known.platform}",
                f"{known.platform} has "
                + ", ".join(name for name in _TIER_ORDER if name in pinned),
            )
    directory = runtime_dir(known)
    absent = missing_tiers(tiers, known)
    if absent and announce is not None:
        announce(_notice(known, absent, sum(_size_of(pinned[tier]) for tier in absent)))
    for tier in absent:
        _fetch_tier(directory, tier, pinned[tier])
    return directory


def ensure(*, announce: Announce | None = None) -> Path | None:
    """Provision what this platform wants, unless the environment names its own.

    `FFRWD_NN_RUNTIME` set is a developer pointing the sidecar at a directory
    of their own: nothing is fetched and nothing is spelled, and the sidecar
    reads the variable itself. Returns the directory a spawn should name, or
    None when the environment already answers.
    """
    if _named_runtime() is not None:
        return None
    known = info()
    return provision(wanted_tiers(known), announce=announce, found=known)


def _named_runtime() -> str | None:
    """The runtime directory the environment names, or None for a blank one."""
    named = os.environ.get(RUNTIME_DIR_VAR)
    return named.strip() if named and named.strip() else None


# --------------------------------------------------------------------------
# what a spawned sidecar is told
# --------------------------------------------------------------------------


def spawn_args() -> list[str]:
    """The ``-nn-runtime`` and ``-nn-target`` a spawned sidecar is given.

    Empty when the environment already names a runtime directory, which is
    what keeps a developer's own `FFRWD_NN_RUNTIME` in charge; empty too when
    nothing has been provisioned, so a sidecar that would have run anyway
    still gets its own refusal rather than a directory that is not there.

    ``-nn-target`` is spelled only when `FFRWD_NN_TARGET` does not, since the
    variable is how a run asks for something other than the default.
    """
    if _named_runtime() is not None:
        return []
    try:
        known = info()
    except FfrwdError:
        return []
    directory = runtime_dir(known)
    if not directory.is_dir():
        return []
    args = [_NN_RUNTIME_FLAG, str(directory)]
    if not os.environ.get(TARGET_VAR):
        args += [_NN_TARGET_FLAG, default_target(known)]
    return args
