"""ffmpeg/ffprobe/ffrwd-wasm binary discovery for ffrwd.

ffrwd requires BOTH ffmpeg and ffprobe. Discovery is PATH-first, so a system
install always wins; only when nothing is on PATH does it fall back to the
``static-ffmpeg`` provisioning package (a default dependency, chosen because
it is the only candidate that ships ffprobe as well as ffmpeg, fetching a
static prebuilt pair on first use and caching it under its own package dir).

``ffmpeg_path()`` / ``ffprobe_path()`` / ``ffrwd_wasm_path()`` are the ONLY
entry points other modules may use to locate these binaries --
:mod:`ffrwd.registry`, :mod:`ffrwd.probe` and :mod:`ffrwd.cli`'s ``run``
route through here rather than ``shutil.which``, so one place knows about the
provider fallback.

All three NEVER raise; they return ``None`` when a binary is on neither PATH
nor delivered by its provider (a broken install, unwritable cache dir, no
network on first use). ``INSTALL_HINT`` is the user-facing wording for the
ffmpeg/ffprobe case.
"""

from __future__ import annotations

import functools
import importlib.metadata
import os
import re
import shutil
import subprocess
import sysconfig

INSTALL_HINT = (
    "the static-ffmpeg provisioner should have supplied ffmpeg/ffprobe "
    "automatically; check it installed correctly (pip show static-ffmpeg), "
    "or put a system ffmpeg/ffprobe on PATH yourself"
)

FFPLAY_HINT = (
    "ffplay ships with ffmpeg but the static-ffmpeg provisioner does not "
    "supply it; install a full ffmpeg build and put ffplay on PATH, or drop "
    "the flag and let the run write its files"
)

FFRWD_WASM_ENV = "FFRWD_WASM"
_SIDECAR_DISTRIBUTION = "ffrwd-wasm"
# The sidecar's program name: what a printed command line names, and what
# PATH is searched for.
SIDECAR_EXECUTABLE = "ffrwd-wasm"


def _provider_paths() -> tuple[str, str] | None:
    """``(ffmpeg, ffprobe)`` from the ``static-ffmpeg`` provisioner, or None.

    Imported lazily: at module load it would cost a ~95MB first-use download
    even on the common path where PATH already has both binaries. Never
    raises -- an absent or broken provider, a failed download, or any other
    provisioning error degrades to None, exactly like a PATH miss.
    """
    try:
        from static_ffmpeg.run import get_or_fetch_platform_executables_else_raise
    except ImportError:
        return None
    try:
        ffmpeg, ffprobe = get_or_fetch_platform_executables_else_raise()
    except Exception:
        return None
    return ffmpeg, ffprobe


def ffmpeg_path() -> str | None:
    """The ffmpeg binary to use: PATH first, provider fallback, else None."""
    found = shutil.which("ffmpeg")
    if found is not None:
        return found
    provided = _provider_paths()
    return provided[0] if provided is not None else None


def ffprobe_path() -> str | None:
    """The ffprobe binary to use: PATH first, provider fallback, else None."""
    found = shutil.which("ffprobe")
    if found is not None:
        return found
    provided = _provider_paths()
    return provided[1] if provided is not None else None


def ffplay_path() -> str | None:
    """The ffplay binary to use: PATH only, else None.

    No provider fallback: the ``static-ffmpeg`` provisioner ships ffmpeg and
    ffprobe and no player, so an ffplay that is not on PATH is not anywhere.
    """
    return shutil.which("ffplay")


def _sidecar_distribution_installed() -> bool:
    try:
        importlib.metadata.distribution(_SIDECAR_DISTRIBUTION)
    except importlib.metadata.PackageNotFoundError:
        return False
    return True


def _sidecar_scripts_path() -> str | None:
    """The ``ffrwd-wasm`` executable in this environment's scripts dir, or None.

    A maturin ``bindings = "bin"`` wheel installs its executable the same way
    a console-script wrapper lands: under the wheel's ``.data/scripts/``,
    which pip unpacks into the environment's scripts directory
    (``sysconfig``, not PATH, since a venv need not be activated). Guarded on
    the distribution actually being installed, so a stray same-named file
    left over from something else is never picked up.
    """
    if not _sidecar_distribution_installed():
        return None
    suffix = sysconfig.get_config_var("EXE") or ""
    candidate = os.path.join(sysconfig.get_path("scripts"), SIDECAR_EXECUTABLE + suffix)
    return candidate if os.path.isfile(candidate) else None


def ffrwd_wasm_path() -> str | None:
    """The ffrwd-wasm sidecar to use: env override, installed wheel, PATH, else None.

    ``FFRWD_WASM`` wins outright when set to a non-blank value. Otherwise the
    ``ffrwd-wasm`` distribution's own executable is tried, then plain PATH
    for a sidecar installed by other means.
    """
    override = os.environ.get(FFRWD_WASM_ENV)
    if override and override.strip():
        return override.strip()
    from_wheel = _sidecar_scripts_path()
    if from_wheel is not None:
        return from_wheel
    return shutil.which(SIDECAR_EXECUTABLE)


# `ffmpeg version 9.0.1-full_build-...`, `ffmpeg version n8.0.1`,
# `ffmpeg version 2026-01-04-git-abc123`: the major number when the build
# states one, and nothing for a dated git build, which states none.
_VERSION_RE = re.compile(r"^ffmpeg version n?(\d+)\.")

# The first ffmpeg release that leaves an input starting below zero alone.
# Before it, such an input is re-based like any other and every frame moves
# later by the negative start.
_FIRST_KEEPING_A_NEGATIVE_START = 9

# How long `ffmpeg -version` may take. It prints a banner and exits; a
# binary that cannot manage that in this long is not one to wait on.
_VERSION_TIMEOUT_SECONDS = 15.0


@functools.cache
def ffmpeg_major_version() -> int | None:
    """The installed ffmpeg's major version, or None when it does not say.

    Never raises, like everything else here: a missing binary, a spawn that
    fails, a timeout, and a banner in a shape this does not know all read as
    None. Cached, and asked for only where a version actually decides
    something, so the ordinary compile never spawns this at all.
    """
    ffmpeg = ffmpeg_path()
    if ffmpeg is None:
        return None
    try:
        done = subprocess.run(
            [ffmpeg, "-version"],
            capture_output=True,
            text=True,
            timeout=_VERSION_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode != 0:
        return None
    first = done.stdout.splitlines()[0] if done.stdout.splitlines() else ""
    found = _VERSION_RE.match(first.strip())
    return int(found.group(1)) if found is not None else None


def ffmpeg_rebases_a_negative_start() -> bool:
    """Whether the installed ffmpeg subtracts an input start BELOW zero too.

    n8 and earlier do, moving every frame of such an input later by the
    negative start; 9.0 leaves it where it is. A build that does not state a
    major version is read as a recent one, which is what a git build off the
    development branch is.
    """
    major = ffmpeg_major_version()
    return major is not None and major < _FIRST_KEEPING_A_NEGATIVE_START
