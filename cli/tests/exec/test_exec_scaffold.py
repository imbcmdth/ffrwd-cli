"""``ffrwd init --rust``: the scaffold is checked by building it.

Marked ``@pytest.mark.exec`` and excluded from the default run, same as the
other files here. Run explicitly::

    python -m pytest -m exec tests/exec/test_exec_scaffold.py -q

Nothing here is a fixture the repo holds: ``init --rust`` writes a package
into a temporary directory, cargo builds it for ``wasm32-wasip2``, the sidecar
describes what came out, and the compiler compiles the recipe the scaffold
shipped against that module. What the scaffold claims -- build, then publish --
is the test.

``FFRWD_WIT_DIR`` is pointed at the sidecar's own wit, the route each in-repo
package workspace takes; the other route the scaffold's ``build.rs`` knows
asks ``ffrwd path ffrwd/wasm``, which needs the package installed. Requires
cargo with the ``wasm32-wasip2`` target, ffmpeg on PATH, and the ``ffrwd-wasm``
sidecar; skips cleanly without any of them.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from ffrwd import binaries, cli, wasm

pytestmark = pytest.mark.exec

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_SIDECAR_WIT = _REPO_ROOT / "sidecar" / "wit"

_NAMESPACE = "me"
_SEGMENT = "scaffolded"
_TARGET = "wasm32-wasip2"
_BUILD_TIMEOUT = 600.0


@pytest.fixture(autouse=True)
def _require_a_toolchain() -> None:
    if shutil.which("cargo") is None:
        pytest.skip("cargo not found on PATH")
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg/ffprobe not found on PATH")
    if binaries.ffrwd_wasm_path() is None:
        pytest.skip("ffrwd-wasm not found (uv sync --extra wasm, or set FFRWD_WASM)")
    if not (_SIDECAR_WIT / "av.wit").is_file():
        pytest.skip(f"the sidecar's wit is not there: {_SIDECAR_WIT}")


@pytest.fixture(scope="module")
def scaffold(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A scaffolded package with its module built. Once per module: cargo is slow."""
    root = tmp_path_factory.mktemp("scaffold") / _SEGMENT
    root.mkdir()
    here = Path.cwd()
    try:
        os.chdir(root)
        assert cli.main(["init", "--rust", "--namespace", _NAMESPACE]) == 0
    finally:
        os.chdir(here)

    environment = dict(os.environ, FFRWD_WIT_DIR=str(_SIDECAR_WIT))
    done = subprocess.run(
        ["cargo", "build", "--target", _TARGET, "--release"],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=_BUILD_TIMEOUT,
    )
    assert done.returncode == 0, done.stderr
    return root


def _module(scaffold: Path) -> Path:
    return scaffold / "target" / _TARGET / "release" / f"{_SEGMENT}.wasm"


def test_the_scaffolded_crate_builds_a_module(scaffold: Path) -> None:
    assert _module(scaffold).is_file()
    # build.rs put the wit where the bindings macro reads it.
    assert (scaffold / "wit" / "av.wit").read_bytes() == (_SIDECAR_WIT / "av.wit").read_bytes()


def test_the_sidecar_describes_it_as_the_video_module_it_declares(scaffold: Path) -> None:
    described = wasm.describe(str(_module(scaffold)))
    assert described.world == f"ffrwd:av@{wasm.WORLD_VERSION}"
    assert described.name == "invert"
    assert wasm.wire_pix_fmt(described) in wasm.WIRE_PIX_FMTS


def test_the_recipe_the_scaffold_ships_compiles_against_that_module(
    scaffold: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(scaffold)
    code = cli.main(
        ["compile", "-f", "recipes/invert.sql", "-v", "source=in.mp4", "-v", "dest=out.mp4"]
    )
    printed = capsys.readouterr().out
    assert code == 0, printed
    # The module is hosted in a sidecar between two ffmpegs, which is what
    # calling one costs and what the printed line has to show.
    assert printed.count("ffmpeg ") == 2
    assert str(_module(scaffold)) in printed
