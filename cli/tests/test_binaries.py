"""Tests for ffrwd.binaries.

All monkeypatched -- PATH via ``shutil.which``, the provider via
``sys.modules["static_ffmpeg.run"]`` (the lazy-import seam) -- so these never
touch a real ffmpeg, never import the real ``static_ffmpeg`` package, and
never risk its first-use download. Unmarked so they stay in the default
suite.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import os
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path, PurePosixPath
from types import ModuleType

import pytest

from ffrwd import binaries


def _fake_version(
    monkeypatch: pytest.MonkeyPatch, banner: str, returncode: int = 0
) -> None:
    """Make ``ffmpeg -version`` print `banner`, and clear what was cached.

    The answer is memoized for the process, so a test that changes it has to
    drop the memo on the way in and the next one has to drop it again.
    """
    monkeypatch.setattr(binaries, "ffmpeg_path", lambda: "ffmpeg")

    def fake_run(argv: list[str], **kwargs: object) -> object:
        return binaries.subprocess.CompletedProcess(
            argv, returncode, stdout=banner + "\nbuilt with gcc 16.1.0\n", stderr=""
        )

    monkeypatch.setattr(binaries.subprocess, "run", fake_run)
    binaries.ffmpeg_major_version.cache_clear()


@pytest.fixture(autouse=True)
def _forget_the_version() -> Iterator[None]:
    """No test leaves a faked ffmpeg version memoized for the next one."""
    binaries.ffmpeg_major_version.cache_clear()
    yield
    binaries.ffmpeg_major_version.cache_clear()


def _install_fake_provider(
    monkeypatch: pytest.MonkeyPatch,
    *,
    ffmpeg: str = "/provider/ffmpeg",
    ffprobe: str = "/provider/ffprobe",
) -> list[int]:
    """Fake ``static_ffmpeg.run`` module; returns a call counter list."""
    calls: list[int] = []
    fake_module = ModuleType("static_ffmpeg.run")

    def fake_get() -> tuple[str, str]:
        calls.append(1)
        return ffmpeg, ffprobe

    fake_module.get_or_fetch_platform_executables_else_raise = fake_get  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "static_ffmpeg.run", fake_module)
    monkeypatch.setitem(sys.modules, "static_ffmpeg", ModuleType("static_ffmpeg"))
    return calls


def _remove_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``from static_ffmpeg.run import ...`` raise ImportError.

    Setting a ``sys.modules`` entry to ``None`` is the documented way to
    force an ``ImportError`` on the next import of that name (PEP 328),
    regardless of whether the real package happens to be installed.
    """
    monkeypatch.setitem(sys.modules, "static_ffmpeg.run", None)
    monkeypatch.setitem(sys.modules, "static_ffmpeg", None)


# ---------------------------------------------------------------------------
# PATH wins
# ---------------------------------------------------------------------------


def test_ffmpeg_path_prefers_path_over_the_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(binaries.shutil, "which", lambda name: f"/usr/bin/{name}")
    calls = _install_fake_provider(monkeypatch)
    assert binaries.ffmpeg_path() == "/usr/bin/ffmpeg"
    assert calls == []  # provider never consulted -- PATH already answered


def test_ffprobe_path_prefers_path_over_the_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(binaries.shutil, "which", lambda name: f"/usr/bin/{name}")
    calls = _install_fake_provider(monkeypatch)
    assert binaries.ffprobe_path() == "/usr/bin/ffprobe"
    assert calls == []


# ---------------------------------------------------------------------------
# fallback consulted when PATH misses
# ---------------------------------------------------------------------------


def test_ffmpeg_path_falls_back_to_the_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(binaries.shutil, "which", lambda name: None)
    _install_fake_provider(monkeypatch, ffmpeg="/cache/ffmpeg", ffprobe="/cache/ffprobe")
    assert binaries.ffmpeg_path() == "/cache/ffmpeg"


def test_ffprobe_path_falls_back_to_the_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(binaries.shutil, "which", lambda name: None)
    _install_fake_provider(monkeypatch, ffmpeg="/cache/ffmpeg", ffprobe="/cache/ffprobe")
    assert binaries.ffprobe_path() == "/cache/ffprobe"


def test_provider_is_consulted_exactly_once_per_call(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(binaries.shutil, "which", lambda name: None)
    calls = _install_fake_provider(monkeypatch)
    binaries.ffmpeg_path()
    assert calls == [1]


# ---------------------------------------------------------------------------
# both absent: never raises, returns None, INSTALL_HINT exists
# ---------------------------------------------------------------------------


def test_ffmpeg_path_is_none_when_path_and_provider_both_miss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(binaries.shutil, "which", lambda name: None)
    _remove_provider(monkeypatch)
    assert binaries.ffmpeg_path() is None


def test_ffprobe_path_is_none_when_path_and_provider_both_miss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(binaries.shutil, "which", lambda name: None)
    _remove_provider(monkeypatch)
    assert binaries.ffprobe_path() is None


def test_a_broken_provider_degrades_to_none_rather_than_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A download failure, a locked cache dir, ... -- anything the provider
    package might raise -- must never propagate out of ffrwd."""
    monkeypatch.setattr(binaries.shutil, "which", lambda name: None)
    fake_module = ModuleType("static_ffmpeg.run")

    def _boom() -> tuple[str, str]:
        raise RuntimeError("network unreachable")

    fake_module.get_or_fetch_platform_executables_else_raise = _boom  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "static_ffmpeg.run", fake_module)
    monkeypatch.setitem(sys.modules, "static_ffmpeg", ModuleType("static_ffmpeg"))

    assert binaries.ffmpeg_path() is None
    assert binaries.ffprobe_path() is None


def test_install_hint_is_a_nonempty_string() -> None:
    assert isinstance(binaries.INSTALL_HINT, str)
    assert binaries.INSTALL_HINT.strip() != ""
    assert "static-ffmpeg" in binaries.INSTALL_HINT


# ---------------------------------------------------------------------------
# ffrwd_wasm_path: env override, installed wheel, PATH, then None
# ---------------------------------------------------------------------------


def test_ffrwd_wasm_path_prefers_env_override_over_everything(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(binaries.FFRWD_WASM_ENV, "/override/ffrwd-wasm")
    monkeypatch.setattr(binaries, "_sidecar_scripts_path", lambda: "/wheel/ffrwd-wasm")
    monkeypatch.setattr(binaries.shutil, "which", lambda name: "/usr/bin/ffrwd-wasm")
    assert binaries.ffrwd_wasm_path() == "/override/ffrwd-wasm"


def test_ffrwd_wasm_path_ignores_a_blank_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(binaries.FFRWD_WASM_ENV, "   ")
    monkeypatch.setattr(binaries, "_sidecar_scripts_path", lambda: None)
    monkeypatch.setattr(binaries.shutil, "which", lambda name: "/usr/bin/ffrwd-wasm")
    assert binaries.ffrwd_wasm_path() == "/usr/bin/ffrwd-wasm"


def test_ffrwd_wasm_path_uses_the_installed_wheels_executable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(binaries.FFRWD_WASM_ENV, raising=False)
    monkeypatch.setattr(binaries, "_sidecar_scripts_path", lambda: "/venv/Scripts/ffrwd-wasm.exe")
    monkeypatch.setattr(binaries.shutil, "which", lambda name: "/usr/bin/ffrwd-wasm")
    assert binaries.ffrwd_wasm_path() == "/venv/Scripts/ffrwd-wasm.exe"


class _FakeDist:
    """A distribution whose RECORD lists `files`, each located under `root`."""

    def __init__(self, root: Path, files: list[str]) -> None:
        self.root = root
        self.files = [PurePosixPath(name) for name in files]

    def locate_file(self, path: PurePosixPath) -> Path:
        return self.root / path


def _installed(monkeypatch: pytest.MonkeyPatch, dist: _FakeDist | None) -> None:
    def distribution(name: str) -> _FakeDist:
        if dist is None:
            raise importlib.metadata.PackageNotFoundError(name)
        return dist

    monkeypatch.setattr(binaries.importlib.metadata, "distribution", distribution)
    monkeypatch.setattr(binaries.sysconfig, "get_config_var", lambda name: ".exe")


def test_sidecar_scripts_path_is_the_executable_the_distribution_installed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An upgrade pip put in the USER scheme, beside a system install it could
    not write to, is found where it went, not in the interpreter's default
    scripts dir where the old sidecar still sits."""
    user = tmp_path / "user" / "site-packages"
    user_scripts = tmp_path / "user" / "Scripts"
    user_scripts.mkdir(parents=True)
    (user_scripts / "ffrwd-wasm.exe").write_text("new")
    system_scripts = tmp_path / "system" / "Scripts"
    system_scripts.mkdir(parents=True)
    (system_scripts / "ffrwd-wasm.exe").write_text("old")
    listed = ["../Scripts/ffrwd-wasm.exe", "ffrwd_wasm/__init__.py"]
    _installed(monkeypatch, _FakeDist(user, listed))
    monkeypatch.setattr(binaries.sysconfig, "get_path", lambda name: str(system_scripts))
    found = binaries._sidecar_scripts_path()
    assert found is not None and Path(found) == user_scripts / "ffrwd-wasm.exe"


def test_sidecar_scripts_path_falls_back_to_the_default_scripts_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A distribution whose record names no executable still finds one in
    the environment's scripts dir."""
    exe = tmp_path / "ffrwd-wasm.exe"
    exe.write_text("")
    _installed(monkeypatch, _FakeDist(tmp_path / "site", []))
    monkeypatch.setattr(binaries.sysconfig, "get_path", lambda name: str(tmp_path))
    assert binaries._sidecar_scripts_path() == str(exe)


def test_sidecar_scripts_path_is_none_when_distribution_is_not_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _installed(monkeypatch, None)
    assert binaries._sidecar_scripts_path() is None


def test_ffrwd_wasm_path_falls_back_to_path_when_wheel_is_not_installed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """PATH fallback found via a stub executable in a tmp dir on a patched PATH.

    Uses the real ``shutil.which`` against a PATH pointed only at ``tmp_path``,
    so this proves the fallback actually walks PATH rather than trusting a
    mocked lookup.
    """
    monkeypatch.delenv(binaries.FFRWD_WASM_ENV, raising=False)
    monkeypatch.setattr(binaries, "_sidecar_scripts_path", lambda: None)
    suffix = ".exe" if sys.platform == "win32" else ""
    stub = tmp_path / f"ffrwd-wasm{suffix}"
    stub.write_text("")
    stub.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))
    # normcase: on Windows, shutil.which resolves the extension against
    # PATHEXT and can hand back ``.EXE`` for a stub written as ``.exe``.
    found = binaries.ffrwd_wasm_path()
    assert found is not None
    assert os.path.normcase(found) == os.path.normcase(str(stub))


def test_ffrwd_wasm_path_is_none_when_absent_everywhere(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(binaries.FFRWD_WASM_ENV, raising=False)
    monkeypatch.setattr(binaries, "_sidecar_scripts_path", lambda: None)
    monkeypatch.setattr(binaries.shutil, "which", lambda name: None)
    assert binaries.ffrwd_wasm_path() is None


# --- which ffmpeg is installed ---------------------------------------------


@pytest.mark.parametrize(
    ("banner", "major"),
    [
        ("ffmpeg version 9.0.1-full_build-www.gyan.dev Copyright (c) 2000", 9),
        ("ffmpeg version n8.0.1 Copyright (c) 2000-2025", 8),
        ("ffmpeg version 7.1-full_build-www.gyan.dev Copyright (c)", 7),
        # A dated git build states no release number.
        ("ffmpeg version 2026-01-04-git-abc1234 Copyright (c)", None),
        ("something else entirely", None),
        ("", None),
    ],
)
def test_the_major_version_is_read_off_the_banner(
    monkeypatch: pytest.MonkeyPatch, banner: str, major: int | None
) -> None:
    """The first line of ``ffmpeg -version``, in the shapes builds write it.
    A release states a number, with or without the ``n`` a tag carries; a git
    build states a date, which is no version at all."""
    _fake_version(monkeypatch, banner)
    assert binaries.ffmpeg_major_version() == major


@pytest.mark.parametrize(
    ("banner", "rebases"),
    [
        ("ffmpeg version n8.0.1 Copyright (c)", True),
        ("ffmpeg version 9.0.1-full_build Copyright (c)", False),
        ("ffmpeg version 2026-01-04-git-abc1234 Copyright (c)", False),
    ],
)
def test_which_builds_rebase_a_start_below_zero(
    monkeypatch: pytest.MonkeyPatch, banner: str, rebases: bool
) -> None:
    """n8 subtracts a negative input start like any other; 9.0 leaves it. A
    build that states no version is read as a recent one, which is what a git
    build off the development branch is."""
    _fake_version(monkeypatch, banner)
    assert binaries.ffmpeg_rebases_a_negative_start() is rebases


def test_a_version_that_cannot_be_read_is_no_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Never raises, like every other lookup here: no ffmpeg, a spawn that
    fails, and a nonzero exit all read as None."""
    monkeypatch.setattr(binaries, "ffmpeg_path", lambda: None)
    binaries.ffmpeg_major_version.cache_clear()
    assert binaries.ffmpeg_major_version() is None

    _fake_version(monkeypatch, "ffmpeg version 9.0.1 Copyright", returncode=1)
    assert binaries.ffmpeg_major_version() is None

    monkeypatch.setattr(binaries, "ffmpeg_path", lambda: "ffmpeg")

    def explode(argv: list[str], **kwargs: object) -> object:
        raise OSError("no such binary")

    monkeypatch.setattr(binaries.subprocess, "run", explode)
    binaries.ffmpeg_major_version.cache_clear()
    assert binaries.ffmpeg_major_version() is None


# --- a timed-out launcher ends with everything it started -------------------

# The real binary: it holds the pipes it inherited and beats into a file until
# it is ended, or for 30 s at the most, so a failing test leaves nothing
# running for long.
_GRANDCHILD = """
import sys, time
end = time.monotonic() + 30
while time.monotonic() < end:
    with open(sys.argv[1], "a") as beat:
        beat.write(".")
    print("still here", flush=True)
    time.sleep(0.05)
"""

# The launcher: starts the real one as its child, sharing its own stdout and
# stderr, and waits on it, as chocolatey's ffprobe.exe shim does.
_LAUNCHER = """
import subprocess, sys
subprocess.run([sys.executable, sys.argv[1], sys.argv[2]])
"""


def _tree(tmp_path: Path) -> tuple[list[str], Path]:
    """A launcher and its child, and the file the child beats into."""
    grandchild = tmp_path / "grandchild.py"
    grandchild.write_text(_GRANDCHILD)
    launcher = tmp_path / "launcher.py"
    launcher.write_text(_LAUNCHER)
    beat = tmp_path / "beat.txt"
    return [sys.executable, str(launcher), str(grandchild), str(beat)], beat


def _stopped(beat: Path) -> bool:
    """True once `beat` stops growing: its writer is gone."""
    before = beat.stat().st_size if beat.exists() else 0
    time.sleep(0.5)
    return (beat.stat().st_size if beat.exists() else 0) == before


def _alive(beat: Path) -> bool:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if beat.exists() and beat.stat().st_size > 0:
            return True
        time.sleep(0.05)
    return False


def test_a_timeout_ends_the_launcher_and_the_binary_it_started(tmp_path: Path) -> None:
    """``subprocess.run`` would end the launcher and then wait, unbounded, on
    pipes its child still holds; this ends the whole tree and comes back."""
    argv, beat = _tree(tmp_path)
    started = time.monotonic()

    with pytest.raises(subprocess.TimeoutExpired):
        binaries.run_to_ceiling(argv, 3.0)

    assert time.monotonic() - started < 3.0 + 2 * binaries._REAP_SECONDS
    assert _alive(beat), "the launcher never started its child"
    assert _stopped(beat), "the launcher's child outlived the timeout"


def test_a_run_that_finishes_in_time_hands_back_its_output(tmp_path: Path) -> None:
    done = binaries.run_to_ceiling(
        [sys.executable, "-c", "import sys; print('out'); print('err', file=sys.stderr)"],
        30.0,
    )

    assert (done.returncode, done.stdout.strip(), done.stderr.strip()) == (0, "out", "err")


def test_a_probe_that_runs_out_of_time_ends_the_real_ffprobe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The listener case: ffprobe on PATH is a launcher, the URL's sender never
    comes, and the probe's ceiling passes. The probe answers None in bounded
    time and the binary behind the launcher is gone, port and all."""
    probe = importlib.import_module("ffrwd.probe")  # the package exports a function by that name
    argv, beat = _tree(tmp_path)
    if sys.platform == "win32":
        shim = tmp_path / "ffprobe.cmd"
        shim.write_text("@" + subprocess.list2cmdline(argv) + "\r\n")
    else:
        shim = tmp_path / "ffprobe"
        shim.write_text("#!/bin/sh\n" + " ".join(f"'{word}'" for word in argv) + "\n")
        shim.chmod(0o755)
    monkeypatch.setattr(binaries, "ffprobe_path", lambda: str(shim))
    monkeypatch.setattr(probe, "_REMOTE_TIMEOUT_SECONDS", 3.0)
    probe.clear_cache()
    started = time.monotonic()

    assert probe.probe("srt://127.0.0.1:9?mode=listener") is None

    assert time.monotonic() - started < 3.0 + 2 * binaries._REAP_SECONDS
    assert _alive(beat), "the shim never started its child"
    assert _stopped(beat), "the launcher's child outlived the probe"
    probe.clear_cache()
