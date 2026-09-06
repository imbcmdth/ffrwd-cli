"""The merge table: what `merge_cues` does to a set of rows.

One table, one assertion. The sidecar's `rowmerge` node is pinned by the
same cases (`sidecar/ffrwd-wasm/src/rowmerge.rs`), so the two halves of the
feature are held to one answer.
"""

from __future__ import annotations

from ffrwd.merge import merge_rows

# (what it shows, rows in, max_distance, rows out)
_TABLE: list[tuple[str, list[dict[str, object]], float, list[dict[str, object]]]] = [
    ("an empty array stays empty", [], 0, []),
    (
        "one row is its own run",
        [{"start_t": 1.0, "end_t": 2.0}],
        0,
        [{"start_t": 1.0, "end_t": 2.0}],
    ),
    (
        "rows that touch merge at distance 0",
        [{"start_t": 0.0, "end_t": 1.0}, {"start_t": 1.0, "end_t": 2.0}],
        0,
        [{"start_t": 0.0, "end_t": 2.0}],
    ),
    (
        "a gap equal to max_distance merges",
        [{"start_t": 0.0, "end_t": 1.0}, {"start_t": 2.0, "end_t": 3.0}],
        1,
        [{"start_t": 0.0, "end_t": 3.0}],
    ),
    (
        "a gap past max_distance does not",
        [{"start_t": 0.0, "end_t": 1.0}, {"start_t": 2.0, "end_t": 3.0}],
        0.9,
        [{"start_t": 0.0, "end_t": 1.0}, {"start_t": 2.0, "end_t": 3.0}],
    ),
    (
        "overlapping rows merge, and the run keeps the furthest end",
        [{"start_t": 0.0, "end_t": 5.0}, {"start_t": 1.0, "end_t": 2.0}],
        0,
        [{"start_t": 0.0, "end_t": 5.0}],
    ),
    (
        "rows out of order are taken in start_t order",
        [{"start_t": 4.0, "end_t": 5.0}, {"start_t": 0.0, "end_t": 1.0}],
        0,
        [{"start_t": 0.0, "end_t": 1.0}, {"start_t": 4.0, "end_t": 5.0}],
    ),
    (
        "zero-length rows merge like any other",
        [
            {"start_t": 1.0, "end_t": 1.0},
            {"start_t": 1.5, "end_t": 1.5},
            {"start_t": 3.0, "end_t": 3.0},
        ],
        0.5,
        [{"start_t": 1.0, "end_t": 1.5}, {"start_t": 3.0, "end_t": 3.0}],
    ),
    (
        "text joins with one space and every other field is the first row's",
        [
            {"index": 1, "track": "speech", "text": "hello", "start_t": 0.0, "end_t": 1.0},
            {"index": 2, "track": "other", "text": "there", "start_t": 1.0, "end_t": 2.0},
        ],
        0,
        [{"index": 1, "track": "speech", "text": "hello there", "start_t": 0.0, "end_t": 2.0}],
    ),
]


def test_the_merge_table() -> None:
    assert [merge_rows(rows, distance) for _, rows, distance, _ in _TABLE] == [
        expected for *_, expected in _TABLE
    ], [name for name, rows, distance, expected in _TABLE if merge_rows(rows, distance) != expected]
