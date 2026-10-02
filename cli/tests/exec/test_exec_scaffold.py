"""``ffrwd init --rust``: the scaffold is checked by building it.

Marked ``@pytest.mark.exec`` and excluded from the default run, same as the
other files here. Run explicitly::

    python -m pytest -m exec tests/exec/test_exec_scaffold.py -q

Nothing here is a fixture the repo holds: ``init --rust`` writes a package
into a temporary directory, cargo builds it for ``wasm32-wasip2``, the sidecar
describes what came out, and the compiler compiles and runs the recipe the
scaffold shipped against that module. What the scaffold claims -- build, then
publish -- is the test.

The crate takes ``ffrwd-node`` and ``ffrwd-frame`` from git, so building needs
the network or a warm cargo cache. Requires cargo with the ``wasm32-wasip2``
target, ffmpeg on PATH, and the ``ffrwd-wasm`` sidecar; skips cleanly without
any of them.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from ffrwd import binaries, cli, wasm

pytestmark = pytest.mark.exec

_SOURCE = Path(__file__).resolve().parent.parent / "fixtures" / "av.mp4"

_NAMESPACE = "me"
_SEGMENT = "scaffolded"
_TARGET = "wasm32-wasip2"
_BUILD_TIMEOUT = 600.0
_PROBE_TIMEOUT = 60.0


@pytest.fixture(autouse=True)
def _require_a_toolchain() -> None:
    if shutil.which("cargo") is None:
        pytest.skip("cargo not found on PATH")
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg/ffprobe not found on PATH")
    if binaries.ffrwd_wasm_path() is None:
        pytest.skip("ffrwd-wasm not found (uv sync --extra wasm, or set FFRWD_WASM)")
    if not _SOURCE.is_file():
        pytest.skip(f"fixture missing: {_SOURCE} (run scripts/gen_fixtures.py first)")


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

    done = subprocess.run(
        ["cargo", "build", "--release", "--target", _TARGET],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=_BUILD_TIMEOUT,
    )
    assert done.returncode == 0, done.stderr
    return root


def _module(scaffold: Path) -> Path:
    return scaffold / "target" / _TARGET / "release" / f"{_SEGMENT}.wasm"


def _video(path: Path) -> dict[str, int]:
    done = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
            "-show_entries", "stream=width,height,nb_read_frames", "-of", "json", str(path),
        ],
        capture_output=True,
        text=True,
        timeout=_PROBE_TIMEOUT,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    stream = json.loads(done.stdout)["streams"][0]
    return {name: int(value) for name, value in stream.items()}


def test_the_scaffolded_crate_builds_a_module(scaffold: Path) -> None:
    assert _module(scaffold).is_file()
    # The SDK carries the wit: the crate neither writes one nor needs a script to.
    assert not (scaffold / "build.rs").exists() and not (scaffold / "wit").exists()


def test_the_sidecar_describes_it_as_the_node_it_declares(scaffold: Path) -> None:
    sidecar = binaries.ffrwd_wasm_path()
    assert sidecar is not None
    done = subprocess.run(
        [str(sidecar), "--describe", str(_module(scaffold))],
        capture_output=True,
        text=True,
        timeout=_PROBE_TIMEOUT,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout)["world"] == wasm.NODE_WORLD

    described = wasm.describe(str(_module(scaffold)))
    assert described.node and described.name == "passthrough"


def test_the_recipe_the_scaffold_ships_compiles_against_that_module(
    scaffold: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(scaffold)
    code = cli.main(
        ["compile", "-f", "recipes/passthrough.sql", "-v", "source=in.mp4", "-v", "dest=out.mp4"]
    )
    printed = capsys.readouterr().out
    assert code == 0, printed
    # The module is hosted in a sidecar between two ffmpegs, which is what
    # calling one costs and what the printed line has to show.
    assert printed.count("ffmpeg ") == 2
    assert str(_module(scaffold)) in printed


def test_the_recipe_the_scaffold_ships_runs_and_hands_the_picture_back(
    scaffold: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(scaffold)
    dest = scaffold / "out.mkv"
    code = cli.main(
        [
            "run", "passthrough", "-y",
            "-v", f"source={_SOURCE.as_posix()}", "-v", f"dest={dest.as_posix()}",
        ]
    )
    assert code == 0, capsys.readouterr().err
    assert _video(dest) == _video(_SOURCE)
