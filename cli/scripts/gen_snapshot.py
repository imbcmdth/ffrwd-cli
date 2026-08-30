"""Generate ``tests/data/reference_registry.json`` from the ffmpeg on PATH.

The reference snapshot is a committed TEST FIXTURE (plan 051): it lets the
test suite build a fully populated :class:`~ffrwd.registry.Registry` via
``ffrwd.registry.load_reference(path)`` with NO ffmpeg on PATH and NO
subprocess call, ever -- the same parsed ``-filters``/``-help`` shapes the
live disk cache uses, just captured ahead of time. It is NOT package data
and is not shipped in the wheel; nothing in the installed package reads it.

Unlike ``ffrwd.registry.load()``, which parses ``-help filter=X`` lazily
per filter on first reference, this script force-loads EVERY name's
options up front:

  - every in-scope filter's options (``Registry.options()`` over
    ``Registry.names()``)
  - every source's options (``Registry.options()`` over
    ``Registry.source_names()``)
  - the array-returning trio's options (``Registry.excluded_options()`` --
    ``channelsplit``/``acrossover``/``extractplanes`` are excluded from the
    filters/sources tables by the v1 pad scope check, so ``options()`` cannot
    reach them; ``excluded_options()`` is the only door, see
    ``ffrwd/lower.py``'s ``ARRAY_RETURNING``)
  - every N-input filter's (``amix``/``hstack``/``xstack``/...,
    ``DynamicFilter.n_input``) options too, for free: registry.py includes
    them in ``Registry.names()`` now (a dynamic INPUT pad count is no longer
    excluded), so the ``options()`` pass above already force-loads them
  - the eleven "collision census" names (``Registry.excluded_options()`` too
    -- they are ordinary in-scope filters already covered by the
    ``options()`` pass above, so this adds no new data, but it exercises
    ``excluded_options()``'s cache-hit branch against a well-known name set;
    see docs/dynamic-filters.md "The collision census" and
    tests/test_lower.py's ``_KNOWN_COLLISIONS``)

Output is deterministic given the same ffmpeg binary and ``--stamp`` value:
every dict is written in INSERTION order (the only wall-clock-ish value is
whatever ``--stamp`` the caller passes, never ``datetime.now()``), and the
file is written LF-only (``tests/data/*.json`` is pinned in .gitattributes) so
regenerating on any platform reproduces byte-identical output.

Insertion order, NOT ``sort_keys=True`` (plan 051): an option table's ORDER is
load-bearing data, not presentation. It is ffmpeg's AVOption declaration
order, which is what positional filter arguments bind against
(``crop(f, 100, 50, 10, 20)``), so alphabetising it -- as this script did when
the snapshot was only ever read for its CONTENT -- silently rewrites every
positional call compiled against the snapshot. Insertion order is just as
deterministic here: it comes from parsing the same ffmpeg's ``-filters`` /
``-help`` output, in the order that output prints.

Usage::

    python scripts/gen_snapshot.py --stamp 2026-08-16

The generated file is COMMITTED (it is a checked-in fixture, not a build
artifact) -- rerun this after any ffmpeg upgrade this project pins against,
or after any change to ``ffrwd/registry.py``'s parsing rules that would
change the captured shapes. ``tests/test_gen_snapshot.py`` asserts (under
``@pytest.mark.exec``, since it requires the real binary) that regenerating
with the committed file's own ``generated`` stamp reproduces it exactly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ffrwd.lower import ARRAY_RETURNING
from ffrwd.registry import Registry

_REPO_ROOT = Path(__file__).resolve().parent.parent
_OUTPUT_PATH = _REPO_ROOT / "tests" / "data" / "reference_registry.json"

# The eleven names docs/dynamic-filters.md "The collision census" and
# tests/test_lower.py's `_KNOWN_COLLISIONS` document as colliding with
# Postgres/stdlib grammar under a bare call -- kept as a small literal copy
# here (not imported from the test module) since this is generation-time
# coverage, not a correctness dependency: every name below is an ordinary
# in-scope filter already force-loaded via `options()`, so a stale copy
# only under- or over-exercises `excluded_options()`, it cannot desync the
# snapshot's actual content.
_CENSUS_NAMES = (
    "copy",
    "corr",
    "format",
    "median",
    "normalize",
    "null",
    "overlay",
    "pad",
    "random",
    "reverse",
    "trim",
)


def build_registry() -> Registry:
    """A live `Registry`, force-loaded: every reachable name's options present.

    Raises `RuntimeError` if no ffmpeg is on PATH -- there is nothing to
    snapshot. (This is the one place in the whole project a `Registry`
    failure is NOT swallowed permissively: `ffrwd/registry.py`'s
    NEVER-raises contract is about compiling ffrwd queries without a
    working ffmpeg, not about generating this file, which has no reason to
    run without one.)
    """
    registry = Registry()
    if not registry.available():
        raise RuntimeError(
            "no ffmpeg on PATH -- scripts/gen_snapshot.py needs a real ffmpeg to "
            "introspect (this is a generation-time tool, not a compile-time one)"
        )
    for name in registry.names():
        registry.options(name)
    for name in registry.source_names():
        registry.options(name)
    for name in ARRAY_RETURNING:
        registry.excluded_options(name)
    for name in _CENSUS_NAMES:
        registry.excluded_options(name)
    return registry


def render(stamp: str, registry: Registry | None = None) -> str:
    """The snapshot file's exact text: sorted keys, 2-space indent, LF, trailing newline.

    `registry` is injectable for tests that want to force-load a specific
    (e.g. monkeypatched) `Registry` rather than the real ffmpeg on PATH;
    omitted, a fresh one is built and force-loaded via `build_registry()`.
    """
    if registry is None:
        registry = build_registry()
    payload = registry.to_snapshot_payload()
    if payload is None:
        raise RuntimeError(
            "registry has no version line after build_registry() succeeded -- unreachable"
        )
    payload = dict(payload)
    payload["snapshot_of"] = payload["version_line"]
    payload["generated"] = stamp
    return json.dumps(payload, indent=2) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate tests/data/reference_registry.json from the ffmpeg on PATH."
    )
    parser.add_argument(
        "--stamp",
        required=True,
        help="deterministic 'generated' value stamped into the snapshot "
        "(e.g. an ISO date) -- never wall-clock time, so regeneration with "
        "the same stamp is byte-identical",
    )
    args = parser.parse_args()
    text = render(args.stamp)
    with open(_OUTPUT_PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print(f"wrote {_OUTPUT_PATH} ({len(text)} bytes)")


if __name__ == "__main__":
    main()
