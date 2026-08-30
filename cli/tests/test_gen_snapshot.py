"""Tests for scripts/gen_snapshot.py and ffrwd.registry.load_reference().

The snapshot is a committed TEST FIXTURE, `tests/data/reference_registry.json`
: it is NOT package data, nothing in the installed package reads it,
and `load_reference(_SNAPSHOT_PATH)` takes the path explicitly. It exists so the suite can
build a fully-populated Registry with no ffmpeg and no subprocess.

Split like tests/test_registry.py: monkeypatched / no-subprocess tests are
unmarked so the default suite (`pytest`, `-m "not exec"`) stays offline;
tests that shell out to a real ffmpeg (regenerating the snapshot, or
comparing it against a freshly-introspected live Registry) are marked
`@pytest.mark.exec`.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path
from types import ModuleType

import pytest

from ffrwd import registry as registry_mod
from ffrwd.registry import Registry, clear_cache, load_reference

_REPO_ROOT = Path(__file__).resolve().parent.parent
_GEN_SNAPSHOT_PATH = _REPO_ROOT / "scripts" / "gen_snapshot.py"
_SNAPSHOT_PATH = _REPO_ROOT / "tests" / "data" / "reference_registry.json"


def _load_gen_snapshot() -> ModuleType:
    """Load scripts/gen_snapshot.py as a module without scripts/ being a package."""
    spec = importlib.util.spec_from_file_location("gen_snapshot", _GEN_SNAPSHOT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def _forced_registry(tmp_path_factory: pytest.TempPathFactory) -> Registry:
    """A single force-loaded `Registry` (`build_registry()`), built ONCE for
    the whole module against a dedicated, session-local cache dir.

    `build_registry()` does a full `-help filter=X` pass over every included
    name (~500), which costs tens of seconds cold. Sharing one already-built
    `Registry` object across every `render()`-based test below (`render()`
    takes an injectable `registry=` for exactly this reason) means that cost
    is paid once for the module, not once per test -- and never touches the
    real `~/.cache/ffrwd` (a dedicated tmp dir, isolated via
    `pytest.MonkeyPatch.context()`, which reverts on exit).
    """
    cache_dir = tmp_path_factory.mktemp("gen_snapshot_cache")
    gen_snapshot = _load_gen_snapshot()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(registry_mod, "_cache_dir", lambda: cache_dir)
        clear_cache()
        registry = gen_snapshot.build_registry()
    clear_cache()  # reset the process-wide load() singleton pointer only
    return registry


# ---------------------------------------------------------------------------
# committed file is up to date / deterministic (requires real ffmpeg)
# ---------------------------------------------------------------------------


@pytest.mark.exec
def test_snapshot_is_up_to_date(_forced_registry: Registry, pinned_ffmpeg: None) -> None:
    """Regenerating with the committed file's own `generated` stamp reproduces
    it byte-for-byte -- the file is generated, not hand-written.

    Only meaningful on the snapshot's own ffmpeg build; any other version
    skips with a drift warning (`pinned_ffmpeg`) instead of failing."""
    gen_snapshot = _load_gen_snapshot()
    with _SNAPSHOT_PATH.open(encoding="utf-8", newline="") as fh:
        committed = fh.read()
    stamp = json.loads(committed)["generated"]
    regenerated = gen_snapshot.render(stamp, registry=_forced_registry)
    assert regenerated == committed, (
        "tests/data/reference_registry.json is stale -- run "
        "`python scripts/gen_snapshot.py --stamp <value>` to regenerate"
    )


@pytest.mark.exec
def test_generation_is_deterministic_across_two_runs(_forced_registry: Registry) -> None:
    gen_snapshot = _load_gen_snapshot()
    first = gen_snapshot.render("2026-01-01", registry=_forced_registry)
    second = gen_snapshot.render("2026-01-01", registry=_forced_registry)
    assert first == second


@pytest.mark.exec
def test_generation_is_deterministic_across_two_independent_builds(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """Two SEPARATE `build_registry()` calls -- independent `Registry` objects,
    independent `-filters`/`-help` subprocess sequences, sharing only a
    (dedicated tmp) disk cache dir once the first warms it -- produce
    byte-identical JSON. A stronger check than calling `render()` twice on
    the SAME already-built object (trivially pure given `sort_keys=True`).
    Does not touch the real `~/.cache/ffrwd` or the committed snapshot
    file -- `render()` returns text, `main()`/the CLI is what writes."""
    cache_dir = tmp_path_factory.mktemp("gen_snapshot_cache_determinism")
    gen_snapshot = _load_gen_snapshot()
    texts: list[str] = []
    for _ in range(2):
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(registry_mod, "_cache_dir", lambda: cache_dir)
            clear_cache()
            registry = gen_snapshot.build_registry()
            texts.append(gen_snapshot.render("2026-01-01", registry=registry))
    clear_cache()
    assert texts[0] == texts[1]


@pytest.mark.exec
def test_generation_output_is_lf_only(_forced_registry: Registry) -> None:
    gen_snapshot = _load_gen_snapshot()
    text = gen_snapshot.render("2026-01-01", registry=_forced_registry)
    assert "\r" not in text


@pytest.mark.exec
def test_build_registry_raises_without_ffmpeg(monkeypatch: pytest.MonkeyPatch) -> None:
    gen_snapshot = _load_gen_snapshot()
    monkeypatch.setattr(registry_mod.binaries, "ffmpeg_path", lambda: None)
    with pytest.raises(RuntimeError):
        gen_snapshot.build_registry()


# ---------------------------------------------------------------------------
# round-trip: load_reference() answers exactly what a live Registry did,
# for a spot-check set (requires real ffmpeg to build the live side)
# ---------------------------------------------------------------------------


@pytest.mark.exec
def test_round_trip_matches_live_registry_spot_check(pinned_ffmpeg: None) -> None:
    live = registry_mod.load()
    assert live.available() is True  # force _ensure_loaded() before anything else
    ref = load_reference(_SNAPSHOT_PATH)

    assert ref.source == "reference"
    assert ref.snapshot_of is not None and ref.snapshot_of.startswith("ffmpeg version")
    assert ref.generated is not None

    # gblur options (numeric, range, default, timeline).
    assert ref.options("gblur") == live.options("gblur")
    gblur_live = live.get("gblur")
    gblur_ref = ref.get("gblur")
    assert gblur_live is not None and gblur_ref is not None
    assert gblur_ref.timeline is True
    assert gblur_ref == gblur_live

    # xfade enum constants.
    assert ref.options("xfade") == live.options("xfade")

    # testsrc source.
    assert ref.get_source("testsrc") == live.get_source("testsrc")
    assert ref.options("testsrc") == live.options("testsrc")

    # timeline flags (pinned against the same real names test_registry.py uses).
    for name, expected in (("gblur", True), ("drawbox", True), ("overlay", True), ("null", False)):
        r = ref.get(name)
        v = live.get(name)
        assert r is not None and v is not None
        assert r.timeline == expected
        assert v.timeline == expected

    # excluded array-returning trio.
    for name in ("channelsplit", "acrossover", "extractplanes"):
        assert ref.excluded_options(name) == live.excluded_options(name)
        assert ref.get(name) is None  # still excluded from the filters table
        assert live.get(name) is None

    # names()/source_names() are the same sets.
    assert set(ref.names()) == set(live.names())
    assert set(ref.source_names()) == set(live.source_names())


# ---------------------------------------------------------------------------
# option ORDER is data, not presentation
# ---------------------------------------------------------------------------


def test_snapshot_preserves_ffmpeg_option_order() -> None:
    """The snapshot must NOT be alphabetised.

    An option table's order is ffmpeg's AVOption declaration order, which is
    what a positional filter argument binds against (`crop(f, 100, 50, 10,
    20)` -> out_w, out_h, x, y). `sort_keys=True` -- harmless while the
    snapshot was read only for its CONTENT -- silently rewrites every
    positional call compiled against it, so this pins a few hand-checked
    orders that alphabetising would visibly break.
    """
    ref = load_reference(_SNAPSHOT_PATH)
    expected = {
        "crop": ["out_w", "out_h", "x", "y"],
        "gblur": ["sigma", "steps", "planes", "sigmaV"],
        "scale": ["width", "height", "flags"],
        "xfade": ["transition", "duration", "offset"],
        "eq": ["contrast", "brightness", "saturation"],
    }
    for name, head in expected.items():
        options = ref.options(name)
        assert options is not None, name
        assert list(options)[: len(head)] == head, name


def test_snapshot_carries_the_options_lowering_needs() -> None:
    """The array-returning trio's options are in the payload, or they are
    uncallable offline -- they stay excluded from `Registry.names()` (dynamic
    on the OUTPUT side), so `excluded_options()` is their only door. An
    N-input filter (amix, hstack, ...) is an ordinary registry member now, so
    its options are already there via `options()`; `excluded_options()` reads
    the same cache and answers just as well."""
    ref = load_reference(_SNAPSHOT_PATH)
    for name in ("channelsplit", "acrossover", "extractplanes"):
        assert ref.excluded_options(name), name
    for name in ("amix", "hstack", "vstack"):
        options = ref.options(name)
        assert options is not None and "inputs" in options, name
        assert options["inputs"].default == "2", name


# ---------------------------------------------------------------------------
# the fixture exists, and is NOT shipped inside the package
# ---------------------------------------------------------------------------


def test_snapshot_fixture_is_present_and_well_shaped() -> None:
    data = json.loads(_SNAPSHOT_PATH.read_text(encoding="utf-8"))
    assert data["format_version"] == 3
    assert isinstance(data["filters"], dict) and len(data["filters"]) > 300
    assert isinstance(data["sources"], dict) and len(data["sources"]) > 20
    assert isinstance(data["options"], dict) and len(data["options"]) > 300


def test_snapshot_is_not_package_data() -> None:
    """It lives under tests/, never under ffrwd/ -- the wheel must not carry
    a 800 KB fixture no installed code path reads."""
    assert not (_REPO_ROOT / "ffrwd" / "reference_registry.json").exists()
    assert not list((_REPO_ROOT / "ffrwd").glob("*.json"))


# ---------------------------------------------------------------------------
# load_reference(): zero subprocess calls, ever -- with or without ffmpeg
# ---------------------------------------------------------------------------


def test_load_reference_never_spawns_a_subprocess_with_no_ffmpeg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []

    def fake_ffmpeg_path() -> str | None:
        calls.append(("ffmpeg_path",))
        return None

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(("run", args, kwargs))
        raise AssertionError("subprocess.run must never be called by load_reference()")

    monkeypatch.setattr(registry_mod.binaries, "ffmpeg_path", fake_ffmpeg_path)
    monkeypatch.setattr(registry_mod.subprocess, "run", fake_run)

    ref = load_reference(_SNAPSHOT_PATH)
    assert ref.source == "reference"
    assert ref.available() is True
    assert ref.get("gblur") is not None
    assert ref.options("gblur") is not None
    assert ref.get_source("testsrc") is not None
    assert ref.options("testsrc") is not None
    assert ref.excluded_options("channelsplit") is not None
    assert len(ref.names()) > 300
    assert len(ref.source_names()) > 20

    assert calls == [], f"expected zero subprocess/ffmpeg_path calls, got {len(calls)}: {calls}"


def test_load_reference_never_spawns_a_subprocess_with_ffmpeg_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same proof, but with a (faked) ffmpeg ON PATH -- the reference
    Registry must not shell out even when it COULD."""
    calls: list[object] = []

    def fake_ffmpeg_path() -> str | None:
        calls.append(("ffmpeg_path",))
        return "C:/fake/ffmpeg.exe"

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(("run", args, kwargs))
        raise AssertionError("subprocess.run must never be called by load_reference()")

    monkeypatch.setattr(registry_mod.binaries, "ffmpeg_path", fake_ffmpeg_path)
    monkeypatch.setattr(registry_mod.subprocess, "run", fake_run)

    ref = load_reference(_SNAPSHOT_PATH)
    assert ref.available() is True
    assert ref.get("gblur") is not None
    assert calls == []


# ---------------------------------------------------------------------------
# markers: source / snapshot_of / generated
# ---------------------------------------------------------------------------


def test_default_registry_source_is_live() -> None:
    assert Registry().source == "live"
    assert Registry().snapshot_of is None
    assert Registry().generated is None


def test_load_reference_source_and_stamps_are_set() -> None:
    ref = load_reference(_SNAPSHOT_PATH)
    assert ref.source == "reference"
    assert ref.snapshot_of is not None and ref.snapshot_of.startswith("ffmpeg version")
    assert ref.generated is not None


# ---------------------------------------------------------------------------
# permissive degradation (NEVER raises, even for a broken snapshot)
# ---------------------------------------------------------------------------


def test_load_reference_missing_file_degrades_permissively(tmp_path: Path) -> None:
    ref = load_reference(tmp_path / "nope.json")
    assert ref.source == "reference"
    assert ref.available() is False
    assert ref.get("gblur") is None
    assert ref.names() == []


def test_load_reference_corrupt_json_degrades_permissively(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{not valid json", encoding="utf-8")
    ref = load_reference(path)
    assert ref.available() is False


def test_load_reference_wrong_format_version_degrades_permissively(tmp_path: Path) -> None:
    path = tmp_path / "stale.json"
    path.write_text(
        json.dumps({"format_version": 999, "filters": {}, "sources": {}, "options": {}}),
        encoding="utf-8",
    )
    ref = load_reference(path)
    assert ref.available() is False


def test_load_reference_bad_shape_degrades_permissively(tmp_path: Path) -> None:
    path = tmp_path / "shape.json"
    path.write_text(
        json.dumps({"format_version": 3, "filters": "not-a-dict", "sources": {}, "options": {}}),
        encoding="utf-8",
    )
    ref = load_reference(path)
    assert ref.available() is False


def test_load_reference_accepts_a_string_path() -> None:
    ref = load_reference(str(_SNAPSHOT_PATH))
    assert ref.available() is True
    assert ref.source == "reference"
