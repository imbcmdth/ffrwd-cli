"""The ONNX Runtime bootstrap: what it fetches, where it puts it, and what it
tells a spawned sidecar.

Nothing here opens a socket. The module's HTTP seam, ``nn._urlopen``, is
replaced by one serving archives built in memory, and the sidecar's
``--nn-info`` answer is set directly rather than asked of a binary -- so the
whole tier runs on a machine with neither a sidecar nor a network, which is
what CI is.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import subprocess
import tarfile
import zipfile
from collections.abc import Iterator, Sequence
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ffrwd import nn, wasm
from ffrwd.errors import FfrwdError
from ffrwd.processes import ModelBinding, SidecarProcess

WINDOWS = nn.Info(
    ort_version="9.9.9", providers=("directml", "cuda", "cpu"), platform="win-x64"
)
LINUX = nn.Info(ort_version="9.9.9", providers=("cuda", "cpu"), platform="linux-x64")
MACOS = nn.Info(ort_version="9.9.9", providers=("coreml", "cpu"), platform="osx-arm64")


# --------------------------------------------------------------------------
# archives, built here rather than downloaded
# --------------------------------------------------------------------------


def _zip_bytes(entries: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, payload in entries.items():
            archive.writestr(name, payload)
    return buffer.getvalue()


def _tgz_bytes(entries: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, payload in entries.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return buffer.getvalue()


class _Answer:
    """What the seam hands back, shaped the way urlopen's answer is."""

    def __init__(self, payload: bytes) -> None:
        self._stream = io.BytesIO(payload)

    def read(self, size: int = -1) -> bytes:
        return self._stream.read(size)

    def __enter__(self) -> _Answer:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def _serving(
    monkeypatch: pytest.MonkeyPatch, bodies: dict[str, bytes], seen: list[str]
) -> None:
    """Point the module's HTTP seam at `bodies`, recording every URL asked for."""

    def fake(request: Any, timeout: float | None = None) -> _Answer:
        url = request.full_url if hasattr(request, "full_url") else str(request)
        seen.append(url)
        if url not in bodies:
            raise OSError("no route to host")
        return _Answer(bodies[url])

    monkeypatch.setattr(nn, "_urlopen", fake)


def _artifact(url: str, payload: bytes, members: Sequence[nn.Member]) -> nn.Artifact:
    return nn.Artifact(
        url=url,
        sha256=hashlib.sha256(payload).hexdigest(),
        size=len(payload),
        members=tuple(members),
    )


@pytest.fixture(autouse=True)
def _own_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A cache directory per check, unlike the suite's shared one.

    What is on disk is half of what these assert -- that a tier already there
    costs nothing, that a failure leaves nothing -- so one check's fetch must
    not be another's starting state.
    """
    monkeypatch.setattr(nn, "_cache_dir", lambda: tmp_path / "cache")


@pytest.fixture
def pinned(monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, bytes]]:
    """A two-tier table over archives built here, with the seam serving them.

    The real table names artifacts nobody may download in this tier; what it
    exercises is the machinery around one, so the machinery gets a table of
    its own and the real one gets a shape check.
    """
    cpu = _zip_bytes({"build/lib/runtime.so": b"cpu runtime"})
    gpu = _tgz_bytes(
        {
            "build/lib/runtime.so.9.9.9": b"gpu runtime",
            "build/lib/provider.so": b"the provider",
        }
    )
    bodies = {"https://example.invalid/cpu.zip": cpu, "https://example.invalid/gpu.tgz": gpu}
    table = {
        ("9.9.9", "win-x64"): {
            "cpu": (
                _artifact(
                    "https://example.invalid/cpu.zip",
                    cpu,
                    [nn.Member(entry="build/lib/runtime.so", name="runtime.so")],
                ),
            ),
            "directml": (
                _artifact(
                    "https://example.invalid/gpu.tgz",
                    gpu,
                    [
                        nn.Member(
                            entry="build/lib/runtime.so.9.9.9",
                            name="runtime.so",
                            subdir="directml",
                            aliases=("runtime.so.9",),
                        ),
                        nn.Member(
                            entry="build/lib/provider.so",
                            name="provider.so",
                            subdir="directml",
                        ),
                    ],
                ),
            ),
        }
    }
    monkeypatch.setattr(nn, "_PINS", table)
    yield bodies


# --------------------------------------------------------------------------
# what the sidecar demands
# --------------------------------------------------------------------------


def _answering(monkeypatch: pytest.MonkeyPatch, stdout: str, code: int = 0) -> list[list[str]]:
    """Make the sidecar answer `stdout`, and record every argv it was run with."""
    calls: list[list[str]] = []

    def fake(argv: list[str], **kwargs: object) -> object:
        calls.append(argv)
        return SimpleNamespace(returncode=code, stdout=stdout, stderr="")

    monkeypatch.setattr(nn.binaries, "ffrwd_wasm_path", lambda: "ffrwd-wasm")
    monkeypatch.setattr(subprocess, "run", fake)
    return calls


def test_the_runtime_version_is_read_off_the_binary_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _answering(
        monkeypatch,
        json.dumps(
            {
                "ort_version": "1.22.0",
                "providers": ["directml", "cuda", "cpu"],
                "platform": "win-x64",
            }
        ),
    )
    found = nn.info()
    assert found == nn.Info(
        ort_version="1.22.0", providers=("directml", "cuda", "cpu"), platform="win-x64"
    )
    assert calls == [["ffrwd-wasm", "--nn-info"]]

    # Asked once per process: the answer cannot change under a running compile.
    assert nn.info() is found
    assert len(calls) == 1


def test_a_sidecar_that_cannot_answer_is_a_hinted_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _answering(monkeypatch, "", code=2)
    with pytest.raises(FfrwdError) as caught:
        nn.info()
    assert "--nn-info" in caught.value.message
    assert caught.value.hint is not None


@pytest.mark.parametrize(
    "payload",
    ["not json at all", "[]", '{"platform": "win-x64"}', '{"ort_version": "1.22.0"}'],
)
def test_an_answer_that_is_not_an_info_document_is_refused(
    monkeypatch: pytest.MonkeyPatch, payload: str
) -> None:
    _answering(monkeypatch, payload)
    with pytest.raises(FfrwdError) as caught:
        nn.info()
    assert caught.value.hint is not None


def test_a_missing_sidecar_names_what_installs_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nn.binaries, "ffrwd_wasm_path", lambda: None)
    with pytest.raises(FfrwdError) as caught:
        nn.info()
    assert caught.value.hint == nn.binaries.INSTALL_HINT


# --------------------------------------------------------------------------
# the layout
# --------------------------------------------------------------------------


def test_the_layout_keys_on_the_version_and_the_platform() -> None:
    windows = nn.runtime_dir(WINDOWS)
    assert windows.parent.name == "9.9.9"
    assert windows.name == "win-x64"
    assert windows.parent.parent == nn.runtime_root()

    # A sidecar demanding another version reads a directory of its own, so an
    # upgrade cannot load the set fetched for the version before it.
    other = nn.runtime_dir(replace(WINDOWS, ort_version="1.23.0"))
    assert other != windows
    assert other.parent.parent == windows.parent.parent


def test_a_version_this_ffrwd_does_not_pin_is_refused_by_name() -> None:
    with pytest.raises(FfrwdError) as caught:
        nn.provision(["cpu"], found=replace(WINDOWS, ort_version="99.0.0"))
    assert "99.0.0" in caught.value.message
    assert "win-x64" in caught.value.message
    assert caught.value.hint is not None


# --------------------------------------------------------------------------
# what each platform takes
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("found", "driver", "tiers", "target"),
    [
        # Windows never auto-fetches CUDA: DirectML runs on any Direct3D 12
        # adapter, and the CUDA tier is five times the size.
        (WINDOWS, False, ("cpu", "directml"), "gpu"),
        (WINDOWS, True, ("cpu", "directml"), "gpu"),
        # CUDA is Linux' only accelerator, and only worth the bytes with a driver.
        (LINUX, True, ("cpu", "cuda"), "gpu"),
        (LINUX, False, ("cpu",), "cpu"),
        # CoreML rides inside the stock archive and is part of the system.
        (MACOS, False, ("cpu",), "gpu"),
    ],
)
def test_what_a_platform_fetches_and_what_it_runs_on(
    monkeypatch: pytest.MonkeyPatch,
    found: nn.Info,
    driver: bool,
    tiers: tuple[str, ...],
    target: str,
) -> None:
    monkeypatch.setattr(nn, "nvidia_driver", lambda: driver)
    assert nn.wanted_tiers(found) == tiers
    assert nn.default_target(found) == target


def test_the_driver_probe_reads_the_two_places_a_driver_shows_up(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(nn.shutil, "which", lambda name: None)
    monkeypatch.setattr(nn, "Path", lambda p: tmp_path / "absent")
    assert nn.nvidia_driver() is False

    monkeypatch.setattr(nn.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    assert nn.nvidia_driver() is True


# --------------------------------------------------------------------------
# fetching
# --------------------------------------------------------------------------


def test_a_tier_lands_where_the_host_looks_for_it(
    monkeypatch: pytest.MonkeyPatch, pinned: dict[str, bytes]
) -> None:
    seen: list[str] = []
    _serving(monkeypatch, pinned, seen)
    said: list[str] = []

    directory = nn.provision(["cpu", "directml"], announce=said.append, found=WINDOWS)

    assert (directory / "runtime.so").read_bytes() == b"cpu runtime"
    # The second build carries the same library name and cannot share a
    # directory with the first, so it gets one of its own.
    assert (directory / "directml" / "runtime.so").read_bytes() == b"gpu runtime"
    assert (directory / "directml" / "provider.so").read_bytes() == b"the provider"
    # The further name a loader may look the library up under.
    assert (directory / "directml" / "runtime.so.9").read_bytes() == b"gpu runtime"
    # Nothing is left behind: the archives are deleted once emptied.
    assert sorted(p.name for p in directory.iterdir()) == ["directml", "runtime.so"]

    assert len(said) == 1
    assert "9.9.9" in said[0] and "win-x64" in said[0]
    assert len(seen) == 2


def test_a_tier_already_on_disk_costs_nothing(
    monkeypatch: pytest.MonkeyPatch, pinned: dict[str, bytes]
) -> None:
    seen: list[str] = []
    _serving(monkeypatch, pinned, seen)
    nn.provision(["cpu"], found=WINDOWS)
    assert len(seen) == 1

    said: list[str] = []
    nn.provision(["cpu"], announce=said.append, found=WINDOWS)
    assert seen == seen[:1]
    # Silence is the whole point: a run that needs nothing says nothing.
    assert said == []


def test_a_hash_that_disagrees_writes_nothing(
    monkeypatch: pytest.MonkeyPatch, pinned: dict[str, bytes]
) -> None:
    seen: list[str] = []
    _serving(monkeypatch, {"https://example.invalid/cpu.zip": b"something else"}, seen)

    with pytest.raises(FfrwdError) as caught:
        nn.provision(["cpu"], found=WINDOWS)
    assert "hashes to" in caught.value.message
    assert caught.value.hint is not None
    assert not nn.runtime_dir(WINDOWS).joinpath("runtime.so").exists()
    assert list(nn.runtime_dir(WINDOWS).iterdir()) == []


def test_an_answer_longer_than_the_pin_is_abandoned(
    monkeypatch: pytest.MonkeyPatch, pinned: dict[str, bytes]
) -> None:
    seen: list[str] = []
    body = pinned["https://example.invalid/cpu.zip"]
    _serving(monkeypatch, {"https://example.invalid/cpu.zip": body + b"x" * 4096}, seen)

    with pytest.raises(FfrwdError) as caught:
        nn.provision(["cpu"], found=WINDOWS)
    assert "longer than" in caught.value.message
    assert list(nn.runtime_dir(WINDOWS).iterdir()) == []


def test_a_fetch_with_nowhere_to_fetch_from_is_a_hinted_rejection(
    monkeypatch: pytest.MonkeyPatch, pinned: dict[str, bytes]
) -> None:
    seen: list[str] = []
    _serving(monkeypatch, {}, seen)

    with pytest.raises(FfrwdError) as caught:
        nn.provision(["cpu"], found=WINDOWS)
    assert "https://example.invalid/cpu.zip" in caught.value.message
    assert caught.value.hint is not None
    assert nn.RUNTIME_DIR_VAR in caught.value.hint


def test_a_tier_whose_archive_is_short_a_library_lands_not_at_all(
    monkeypatch: pytest.MonkeyPatch, pinned: dict[str, bytes]
) -> None:
    # The DirectML tier's one archive holds two libraries. An archive holding
    # only the first must leave neither: half a set is what later dlopens.
    short = _tgz_bytes({"build/lib/runtime.so.9.9.9": b"gpu runtime"})
    table = dict(nn._PINS[("9.9.9", "win-x64")])
    original = table["directml"][0]
    table["directml"] = (
        replace(original, sha256=hashlib.sha256(short).hexdigest(), size=len(short)),
    )
    monkeypatch.setattr(nn, "_PINS", {("9.9.9", "win-x64"): table})
    _serving(monkeypatch, {"https://example.invalid/gpu.tgz": short}, [])

    with pytest.raises(FfrwdError) as caught:
        nn.provision(["directml"], found=WINDOWS)
    assert "build/lib/provider.so" in caught.value.message
    assert list(nn.runtime_dir(WINDOWS).joinpath("directml").iterdir()) == []


def test_a_tier_this_platform_has_none_of_is_refused_with_what_it_has(
    pinned: dict[str, bytes],
) -> None:
    with pytest.raises(FfrwdError) as caught:
        nn.provision(["cuda"], found=WINDOWS)
    assert "cuda" in caught.value.message
    assert caught.value.hint is not None
    assert "directml" in caught.value.hint


def test_the_environment_naming_a_runtime_stops_the_bootstrap(
    monkeypatch: pytest.MonkeyPatch, pinned: dict[str, bytes]
) -> None:
    seen: list[str] = []
    _serving(monkeypatch, pinned, seen)
    monkeypatch.setenv(nn.RUNTIME_DIR_VAR, "/somewhere/of/my/own")

    assert nn.ensure() is None
    assert seen == []


# --------------------------------------------------------------------------
# the real table
# --------------------------------------------------------------------------


def test_every_pinned_artifact_is_named_hashed_and_sized() -> None:
    for (version, platform), tiers in nn._PINS.items():
        assert re.fullmatch(r"\d+\.\d+\.\d+", version)
        assert "cpu" in tiers, f"{platform} has no CPU tier to fall back to"
        for tier, artifacts in tiers.items():
            assert tier in nn._TIER_ORDER
            assert artifacts, f"{platform} {tier} pins no artifact"
            for artifact in artifacts:
                assert artifact.url.startswith("https://")
                assert re.fullmatch(r"[0-9a-f]{64}", artifact.sha256)
                assert artifact.size > 0
                assert artifact.members
                for member in artifact.members:
                    assert "/" in member.entry, "a member is read by full path"
                    assert "/" not in member.name


def test_the_pinned_windows_layout_is_the_one_the_host_resolves() -> None:
    # The host loads onnxruntime.dll out of the runtime directory, and the
    # DirectML build -- a different binary of the same name -- out of
    # directml/ beside it.
    tiers = nn._PINS[("1.22.0", "win-x64")]
    placed = {
        (member.subdir, member.name)
        for tier in ("cpu", "directml", "cuda")
        for artifact in tiers[tier]
        for member in artifact.members
    }
    assert ("", "onnxruntime.dll") in placed
    assert ("directml", "onnxruntime.dll") in placed
    assert ("directml", "DirectML.dll") in placed
    assert ("", "onnxruntime_providers_cuda.dll") in placed
    assert ("", "onnxruntime_providers_shared.dll") in placed


# The sublibraries cudnn64_9.dll loads by bare name from its own directory:
# the sidecar's CUDNN_SUBLIBS Windows list, pinned here rather than read
# across the repo.
_CUDNN_SUBLIBS_WINDOWS = (
    "cudnn_adv64_9.dll",
    "cudnn_graph64_9.dll",
    "cudnn_ops64_9.dll",
    "cudnn_heuristic64_9.dll",
    "cudnn_engines_precompiled64_9.dll",
    "cudnn_engines_runtime_compiled64_9.dll",
)


def test_the_full_tier_carries_every_cudnn_sublibrary_the_sidecar_loads() -> None:
    full = nn._PINS[("1.22.0", "win-x64")]["full"]
    names = {member.name for artifact in full for member in artifact.members}
    for lib in _CUDNN_SUBLIBS_WINDOWS:
        assert lib in names, f"{lib} is missing from the full tier's cudnn artifact"


# --------------------------------------------------------------------------
# what a spawned sidecar is told
# --------------------------------------------------------------------------

MODULE = "modules/depth.wasm"


def _with_model() -> SidecarProcess:
    return SidecarProcess(
        id="sidecar0",
        module=MODULE,
        node="n1",
        args={},
        outputs=("video",),
        models=(ModelBinding(name="depth", path="modules/depth.onnx"),),
    )


def _without_model() -> SidecarProcess:
    return SidecarProcess(
        id="sidecar0", module=MODULE, node="n1", args={}, outputs=("video",)
    )


@pytest.fixture
def provisioned(monkeypatch: pytest.MonkeyPatch, pinned: dict[str, bytes]) -> Path:
    """A runtime on disk for `WINDOWS`, with the sidecar's answer already known."""
    seen: list[str] = []
    _serving(monkeypatch, pinned, seen)
    monkeypatch.setattr(nn, "_INFO", WINDOWS)
    monkeypatch.setattr(nn.binaries, "ffrwd_wasm_path", lambda: "ffrwd-wasm")
    return nn.provision(["cpu"], found=WINDOWS)


def test_a_process_that_binds_a_model_is_told_where_the_runtime_is(
    provisioned: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(nn.TARGET_VAR, raising=False)
    argv = wasm.sidecar_argv(_with_model())
    assert argv[argv.index("-nn-runtime") + 1] == str(provisioned)
    assert argv[argv.index("-nn-target") + 1] == "gpu"
    # Ahead of the bindings, which is where the sidecar reads them.
    assert argv.index("-nn-runtime") < argv.index("-nn")


def test_a_process_that_binds_none_is_told_nothing_about_a_runtime(
    provisioned: Path,
) -> None:
    argv = wasm.sidecar_argv(_without_model())
    assert "-nn-runtime" not in argv
    assert "-nn-target" not in argv


def test_a_printed_command_spells_no_runtime_directory(provisioned: Path) -> None:
    # It is under THIS machine's cache; a reader has their own.
    argv = wasm.shown_argv(_with_model())
    assert "-nn-runtime" not in argv
    assert "-nn-target" not in argv
    assert argv[argv.index("-nn") + 1] == "depth=modules/depth.onnx"


def test_the_target_variable_beats_the_default(
    provisioned: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The sidecar reads the variable itself, so the way to defer to it is to
    # spell no target at all.
    monkeypatch.setenv(nn.TARGET_VAR, "cuda")
    argv = wasm.sidecar_argv(_with_model())
    assert "-nn-target" not in argv
    assert "-nn-runtime" in argv


def test_an_unprovisioned_machine_leaves_the_sidecar_its_own_refusal(
    monkeypatch: pytest.MonkeyPatch, pinned: dict[str, bytes]
) -> None:
    monkeypatch.setattr(nn, "_INFO", WINDOWS)
    monkeypatch.setattr(nn.binaries, "ffrwd_wasm_path", lambda: "ffrwd-wasm")
    argv = wasm.sidecar_argv(_with_model())
    assert "-nn-runtime" not in argv


def test_a_developers_own_runtime_directory_is_left_alone(
    provisioned: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(nn.RUNTIME_DIR_VAR, "/somewhere/of/my/own")
    argv = wasm.sidecar_argv(_with_model())
    assert "-nn-runtime" not in argv
    assert "-nn-target" not in argv
