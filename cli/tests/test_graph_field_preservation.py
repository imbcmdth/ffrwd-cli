"""Every pass that rebuilds `ir.Graph` from an existing one must let a field
it does not touch ride through unchanged. `Graph.dropped_aliases` was reset
to empty by two passes that listed every field by hand until patched by
hand (commit def7765) -- this file is the regression guard: it fails the
day a THIRD field goes missing the same way, without anyone adding a new
assertion for it.

Each pass/helper below is exercised once, against one graph carrying a
distinct sentinel in every field. `dataclasses.fields(Graph)` drives the
comparison, so a field added to `Graph` later is swept in automatically --
nothing here needs editing for that; only a pass's own `_CHANGES` set needs
updating if that pass is meant to start touching the new field.
"""

from __future__ import annotations

import dataclasses

from ffrwd.emit import _drop_input_slots
from ffrwd.ir import (
    Graph,
    ModuleSource,
    Node,
    Output,
    RowsSink,
    SinkUnit,
    SourceTrack,
    UrlSource,
    UrlSourceRow,
    dedup_inputs,
)
from ffrwd.pts import insert_pts_resets
from ffrwd.split import insert_splits


def _sentinel_graph() -> Graph:
    """A `Graph` with a distinct, recognizable value in every field.

    `input_paths`/`sources` carry a duplicate ("dup.mp4" twice, untrimmed,
    same options) so `dedup_inputs` has something to fold; `n0` reads the
    third alias so `_drop_input_slots` has a real slot to drop.
    """
    g = Graph(
        input_paths=["dup.mp4", "dup.mp4", "solo.mp4"],
        sources={"a": 0, "b": 1, "c": 2},
    )
    g.nodes = {
        "n0": Node(
            id="n0", filter="scale", args={"w": 100}, inputs=["src:c:v:0"], outputs=["video"]
        )
    }
    g.sinks = [SinkUnit(outputs=[Output(ref="n0", type="video", name=None, metadata={})])]
    g.input_trims = {"c": (1.0, 2.0)}
    g.input_options = {"c": {"loop": True}}
    g.rows_sinks = {"m": RowsSink(container="subtitle", alias="rows_in")}
    g.module_sinks = ["m"]
    g.packet_sinks = {"m": [{"video_codec": "h264"}]}
    g.module_sources = {
        "s": ModuleSource(
            alias="s",
            module="replay.wasm",
            params="{}",
            tracks=(
                SourceTrack(
                    ref="src:s:v:0", kind="video", codec="h264", time_base=(1, 90000), row=0
                ),
            ),
            bounded=True,
        )
    }
    g.url_sources = {
        "u": UrlSource(
            alias="u",
            module="list.wasm",
            params="{}",
            document=None,
            rows=(UrlSourceRow(url="http://example/x", input=2),),
        )
    }
    g.dropped_aliases = {"c"}
    return g


# The fields each pass/helper is documented to change, via its own
# `dataclasses.replace(g, ...)` call -- everything else in
# `dataclasses.fields(Graph)` must come through untouched.
_CASES: list[tuple[str, object, frozenset[str]]] = [
    ("insert_splits", lambda g: insert_splits(g), frozenset({"nodes", "sinks"})),
    ("insert_pts_resets", lambda g: insert_pts_resets(g), frozenset({"nodes", "sinks"})),
    ("dedup_inputs", lambda g: dedup_inputs(g), frozenset({"input_paths", "sources"})),
    (
        "_drop_input_slots",
        lambda g: _drop_input_slots(g, {2}),
        frozenset(
            {
                "input_paths",
                "sources",
                "input_trims",
                "input_options",
                "url_sources",
                "sinks",
                "dropped_aliases",
            }
        ),
    ),
]


def test_every_field_not_touched_by_a_pass_survives_it() -> None:
    field_names = [f.name for f in dataclasses.fields(Graph)]
    for name, run, changes in _CASES:
        g = _sentinel_graph()
        before = {field: getattr(g, field) for field in field_names}
        out = run(g)
        for field in field_names:
            if field in changes:
                continue
            assert getattr(out, field) == before[field], (
                f"{name} changed Graph.{field}, which it does not list in "
                f"its own dataclasses.replace(...) call"
            )
        # every field the pass claims to change is actually a real field --
        # a stale entry in `changes` would hide a field this test should
        # have covered.
        assert changes <= set(field_names)
