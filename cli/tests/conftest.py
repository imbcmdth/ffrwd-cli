"""Suite-wide default: ``compile_sql`` resolves against the captured snapshot.

The installed ffmpeg's filter set is THE function surface, which would
otherwise make every ``compile_sql``-based test depend on whichever ffmpeg the
machine happens to have -- and fail on a machine with none. CI runs the default
suite BEFORE installing ffmpeg, deliberately: the non-exec tier must be
deterministic on a bare machine, and ``tests/data/reference_registry.json``
exists for exactly that -- it serves golden tests, fuzzing and offline CI.

The seam is the compiler module's ``registry_module`` reference: swapping it
for a shim redirects ``compile_sql`` (and everything above it: the CLI, the
cookbook harness, the prompt example checks) without touching
``ffrwd.registry`` itself, so registry-introspection tests, direct
``lower()`` calls with hand-built registries, and ``shutil.which``
simulations all behave exactly as they would in production.

Exec-marked tests are exempt: they run their compiled commands through the
real ffmpeg, so they must compile against the real ffmpeg's own registry.
"""

from __future__ import annotations

import functools
import warnings
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest

from ffrwd import registry as registry_module
from ffrwd.registry import Registry, load_reference

SNAPSHOT_PATH = Path(__file__).resolve().parent / "data" / "reference_registry.json"


@functools.cache
def _reference_registry() -> Registry:
    return load_reference(SNAPSHOT_PATH)


@pytest.fixture
def pinned_ffmpeg() -> None:
    """Skip -- loudly, never fail -- unless the installed ffmpeg IS the
    snapshot's pinned build.

    A handful of exec tests compare the committed snapshot against the live
    binary (freshness, option order, diagnostic text). Those comparisons are
    only meaningful on the exact ffmpeg the snapshot was generated from;
    on any other version the differences are ordinary release drift, which
    is the situation ffrwd exists to handle, not a defect. Behavioral
    guarantees stay tested on EVERY version (the md5 positional-fidelity
    exec test carries them); what skips here is only the byte-level
    version-pinned redundancy. The skip also warns, so a CI run against a
    different ffmpeg reports the drift in the warnings summary without
    breaking the build.
    """
    live = registry_module.load()
    live.available()
    pinned = _reference_registry().snapshot_of
    installed = live._version_line
    if installed != pinned:
        message = (
            f"installed ffmpeg is not the snapshot's pinned build; "
            f"version-pinned snapshot checks skipped (informational, not a "
            f"failure). installed: {installed!r}, snapshot: {pinned!r}. "
            f"To refresh the pin, run scripts/gen_snapshot.py on the new "
            f"version and review the diff."
        )
        warnings.warn(message, stacklevel=1)
        pytest.skip(message)


@pytest.fixture(scope="session")
def _store_home(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("store-home")


@pytest.fixture(autouse=True)
def _isolated_store(
    _store_home: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[None]:
    """No test reads or writes the real ``~/.cache/ffrwd``.

    `discover` consults the machine-wide lockfile, so without this a developer
    who has installed a package globally would compile against it here and get
    results nobody else gets. Tests that want a global lockfile point the same
    seam at a directory of their own.

    The fetched ONNX Runtime lives under the same cache and has a seam of its
    own, redirected here too: a developer who has run `ffrwd setup nn` must
    not have a spawned sidecar told about their runtime directory in a tier
    that has to behave the same on a machine with none. What the sidecar
    demands is cached per process, so that answer is cleared as well.
    """
    from ffrwd import nn, store

    monkeypatch.setattr(store, "_cache_dir", lambda: _store_home)
    monkeypatch.setattr(nn, "_cache_dir", lambda: _store_home)
    monkeypatch.setattr(nn, "_INFO", None)
    yield
    # The directory is shared for speed - a per-test one costs four times the
    # suite's runtime. Only the machine-wide lockfile leaks between tests, so
    # only it is cleared: what a project can SEE is what several tests assert.
    lock = store.global_lock_path()
    if lock.exists():
        lock.unlink()


@pytest.fixture(autouse=True)
def _offline_registry(
    _store_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No test reaches the published registry, and none is logged in.

    A check that forgot to point `FFRWD_REGISTRY` somewhere of its own would
    otherwise fetch over the network in CI. This aims it at a directory that
    does not exist, so such a check fails as unreachable rather than quietly
    depending on what is published, and aims the API host at a name that
    cannot resolve. The credentials file follows the same rule and gets a
    directory PER TEST, unlike the store: a developer who has logged in must
    not have their token read here, and a check that saves one must not leave
    the next check logged in.
    """
    from ffrwd import credentials

    monkeypatch.setenv("FFRWD_REGISTRY", str(_store_home / "no-registry-here"))
    monkeypatch.setenv("FFRWD_API", "http://api.invalid")
    monkeypatch.delenv(credentials.TOKEN_ENV, raising=False)
    monkeypatch.setattr(credentials, "_config_dir", lambda: tmp_path / "config")


@pytest.fixture(autouse=True)
def _snapshot_function_surface(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    if request.node.get_closest_marker("exec") is not None:
        return
    from ffrwd import cli, compiler
    from ffrwd.mcp import tools as mcp_tools

    monkeypatch.setattr(
        compiler, "registry_module", SimpleNamespace(load=_reference_registry)
    )
    # cli.py holds its own reference for the `prompt` subcommand's registry
    # -- same shim, same reason: the default tier is deterministic on a bare machine.
    monkeypatch.setattr(
        cli, "registry_module", SimpleNamespace(load=_reference_registry)
    )
    # And the MCP tools, whose `filters` tool and dialect resource read it.
    monkeypatch.setattr(
        mcp_tools, "registry_module", SimpleNamespace(load=_reference_registry)
    )
