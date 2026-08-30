"""The world's version, agreed on by everything that states it.

A world bump moves three things at once: sidecar/wit/av.wit, the
version in sidecar/wasm/ffrwd.json (the `ffrwd/wasm` package the
sidecar's release publishes), and WORLD_VERSION here in the cli.
These tests hold them together, and hold the superseded worlds
frozen under sidecar/worlds/ exactly as their bumps left them.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from ffrwd.wasm import WORLD_VERSION, WORLDS

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
WIT_PATH = REPO_ROOT / "sidecar" / "wit" / "av.wit"
PACKAGE_MANIFEST = REPO_ROOT / "sidecar" / "wasm" / "ffrwd.json"
WORLDS_DIR = REPO_ROOT / "sidecar" / "worlds"

_PACKAGE_LINE = re.compile(r"^package ffrwd:av@(?P<version>[0-9.]+);", re.MULTILINE)


def _declared_version(wit: Path) -> str:
    match = _PACKAGE_LINE.search(wit.read_text(encoding="utf-8"))
    assert match is not None, f"{wit} declares no `package ffrwd:av@...;` line"
    return match.group("version")


def test_the_wit_names_the_world_this_compiler_hosts() -> None:
    assert _declared_version(WIT_PATH) == WORLD_VERSION, (
        f"sidecar/wit/av.wit declares ffrwd:av@{_declared_version(WIT_PATH)} and the "
        f"compiler's WORLD_VERSION is {WORLD_VERSION}; a world bump moves both at once"
    )


def test_the_package_version_is_the_world_version() -> None:
    manifest = json.loads(PACKAGE_MANIFEST.read_text(encoding="utf-8"))
    assert manifest["name"] == "ffrwd/wasm"
    assert manifest["version"] == WORLD_VERSION, (
        f"sidecar/wasm is version {manifest['version']} and the world is "
        f"{WORLD_VERSION}; a world bump moves three things at once: sidecar/wit/av.wit, "
        "the version in sidecar/wasm/ffrwd.json, and WORLD_VERSION in cli/ffrwd/wasm.py"
    )


def test_every_superseded_world_is_frozen_under_its_own_version() -> None:
    superseded = [world.partition("@")[2] for world in WORLDS[:-1]]
    held = sorted(entry.name for entry in WORLDS_DIR.iterdir() if entry.is_dir())
    assert held == sorted(superseded), (
        f"sidecar/worlds holds {held} and the superseded worlds are "
        f"{sorted(superseded)}; a bump freezes the world it replaces and nothing else"
    )
    for version in superseded:
        frozen = WORLDS_DIR / version / "av.wit"
        assert _declared_version(frozen) == version, (
            f"{frozen} declares ffrwd:av@{_declared_version(frozen)}, and a frozen "
            "world is the one its directory names"
        )
