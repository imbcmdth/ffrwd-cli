"""Tests for ffrwd.binaries.

All monkeypatched -- PATH via ``shutil.which``, the provider via
``sys.modules["static_ffmpeg.run"]`` (the lazy-import seam) -- so these never
touch a real ffmpeg, never import the real ``static_ffmpeg`` package, and
never risk its first-use download. Unmarked so they stay in the default
suite.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import ModuleType

import pytest

from ffrwd import binaries


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


def test_sidecar_scripts_path_resolves_via_sysconfig_when_distribution_installed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Exercises the real ``_sidecar_scripts_path`` body: a stub executable
    sits in a fake scripts dir, the distribution is faked as installed, and
    the suffix is faked to match, all independent of any real ffrwd-wasm."""
    scripts_dir = tmp_path
    exe = scripts_dir / "ffrwd-wasm.exe"
    exe.write_text("")
    monkeypatch.setattr(binaries, "_sidecar_distribution_installed", lambda: True)
    monkeypatch.setattr(binaries.sysconfig, "get_path", lambda name: str(scripts_dir))
    monkeypatch.setattr(binaries.sysconfig, "get_config_var", lambda name: ".exe")
    assert binaries._sidecar_scripts_path() == str(exe)


def test_sidecar_scripts_path_is_none_when_distribution_is_not_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(binaries, "_sidecar_distribution_installed", lambda: False)
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
