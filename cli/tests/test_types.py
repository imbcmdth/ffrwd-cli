"""Tests for the declared type system.

Two jobs. First, the registry is internally consistent: every field's type
reference resolves, every array column's elements are a declared record, and
every field carries a W/RO mark and a declared type reference — pinned here
per type, so an accidental edit to a declaration fails loudly.

Second, and the load-bearing half: every derived view still equals the exact
value it had when it was typed out by hand in parser.py. The expected
constants below are those literals, pasted. A change to a declaration that
silently moves a column set, a column ORDER, or a hint's wording has to break
one of these before it can reach a golden test.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from ffrwd import lower, parser, types
from ffrwd.types import COLUMN_TYPES, TYPES, Field, element_type, is_array, resolve

# Every declared field, per type: name -> (type reference, W or RO). The
# reference tables for stream records, non-stream records and the container,
# restated independently of the declarations they check.
EXPECTED_FIELDS: dict[str, dict[str, tuple[str, str]]] = {
    "text": {},
    "number": {},
    "boolean": {},
    "vector": {},
    "stream": {},
    "seek": {},
    "tag": {"key": ("text", "W"), "value": ("text", "W")},
    "flag": {"key": ("text", "W"), "set": ("boolean", "W")},
    "video_stream": {
        "index": ("number", "RO"),
        "tags": ("tag[]", "W"),
        "disposition": ("flag[]", "W"),
        "codec": ("text", "RO"),
        "width": ("number", "RO"),
        "height": ("number", "RO"),
        "fps": ("text", "RO"),
        "bitrate": ("number", "RO"),
        "duration": ("number", "RO"),
        "color_transfer": ("text", "RO"),
    },
    "audio_stream": {
        "index": ("number", "RO"),
        "tags": ("tag[]", "W"),
        "disposition": ("flag[]", "W"),
        "codec": ("text", "RO"),
        "channels": ("number", "RO"),
        "channel_layout": ("text", "RO"),
        "sample_rate": ("number", "RO"),
        "bitrate": ("number", "RO"),
        "duration": ("number", "RO"),
    },
    "subtitle_stream": {
        "index": ("number", "RO"),
        "tags": ("tag[]", "W"),
        "disposition": ("flag[]", "W"),
        "codec": ("text", "RO"),
    },
    "data_stream": {
        "index": ("number", "RO"),
        "tags": ("tag[]", "W"),
        "disposition": ("flag[]", "W"),
        "codec": ("text", "RO"),
    },
    "chapter": {
        "index": ("number", "RO"),
        "title": ("text", "W"),
        "start_t": ("number", "W"),
        "end_t": ("number", "W"),
    },
    "attachment": {
        "index": ("number", "RO"),
        "filename": ("text", "W"),
        "mimetype": ("text", "W"),
        "path": ("text", "W"),
    },
    "cue": {
        "index": ("number", "RO"),
        "track": ("text", "RO"),
        "text": ("text", "W"),
        "start_t": ("number", "W"),
        "end_t": ("number", "W"),
    },
    "embedding": {
        "index": ("number", "RO"),
        "track": ("text", "RO"),
        "start_t": ("number", "W"),
        "end_t": ("number", "W"),
        "vector": ("vector", "W"),
    },
    "container": {
        "video": ("video_stream[]", "W"),
        "audio": ("audio_stream[]", "W"),
        "subtitle": ("subtitle_stream[]", "W"),
        "data": ("data_stream[]", "W"),
        "chapters": ("chapter[]", "W"),
        "cues": ("cue[]", "RO"),
        "embeddings": ("embedding[]", "RO"),
        "attachments": ("attachment[]", "W"),
        "t": ("seek", "RO"),
        "duration": ("number", "RO"),
        "tags": ("tag[]", "W"),
    },
}

# The fields declared but not surfaced by the compiler today, as
# "<type>.<field>". Nothing else may carry `exposed=False`.
EXPECTED_UNEXPOSED: set[str] = set()

# The WRITE-ONLY fields, as "<type>.<field>": written in a record literal,
# never a column of the row a query reads back. Nothing else may carry
# `readable=False`.
EXPECTED_UNREADABLE = {"attachment.path"}

EXPECTED_KINDS = {
    "text": "scalar",
    "number": "scalar",
    "boolean": "scalar",
    "vector": "scalar",
    "stream": "handle",
    "seek": "handle",
    "tag": "map",
    "flag": "map",
    "video_stream": "stream",
    "audio_stream": "stream",
    "subtitle_stream": "stream",
    "data_stream": "stream",
    "chapter": "record",
    "attachment": "record",
    "cue": "record",
    "embedding": "record",
    "container": "container",
}

# --- The views, as parser.py spelled them out before they were derived. ---

EXPECTED_ROW_COMMON = {
    "index": "number",
    "tags": "tag[]",
    "disposition": "flag[]",
    "codec": "text",
}

EXPECTED_ROW_SCHEMAS = {
    "audio": EXPECTED_ROW_COMMON
    | {
        "channels": "number",
        "channel_layout": "text",
        "sample_rate": "number",
        "bitrate": "number",
        "duration": "number",
    },
    "video": EXPECTED_ROW_COMMON
    | {
        "width": "number",
        "height": "number",
        "fps": "text",
        "bitrate": "number",
        "duration": "number",
        "color_transfer": "text",
    },
    "subtitle": dict(EXPECTED_ROW_COMMON),
    "data": dict(EXPECTED_ROW_COMMON),
    "chapters": {
        "index": "number",
        "title": "text",
        "start_t": "number",
        "end_t": "number",
    },
    "cues": {
        "index": "number",
        "track": "text",
        "text": "text",
        "start_t": "number",
        "end_t": "number",
    },
    "embeddings": {
        "index": "number",
        "track": "text",
        "start_t": "number",
        "end_t": "number",
        "vector": "vector",
    },
    "attachments": {
        "index": "number",
        "filename": "text",
        "mimetype": "text",
    },
}

EXPECTED_INPUT_COLUMNS = frozenset(
    {
        "video",
        "audio",
        "subtitle",
        "data",
        "t",
        "duration",
        "chapters",
        "cues",
        "embeddings",
        "attachments",
        "tags",
    }
)

EXPECTED_STREAM_ARRAY_COLUMNS = frozenset({"video", "audio", "subtitle", "data"})

EXPECTED_UNNEST_COLUMNS = EXPECTED_STREAM_ARRAY_COLUMNS | {
    "chapters",
    "cues",
    "embeddings",
    "attachments",
}

EXPECTED_STAR_COLUMNS = (
    "video",
    "audio",
    "subtitle",
    "data",
    "chapters",
    "attachments",
)

EXPECTED_ROW_STAR_COLUMNS = {
    "audio": ("index", "codec", "channels", "channel_layout", "sample_rate",
              "bitrate", "duration"),
    "video": ("index", "codec", "width", "height", "fps", "bitrate", "duration",
              "color_transfer"),
    "subtitle": ("index", "codec"),
    "data": ("index", "codec"),
    "chapters": ("index", "title", "start_t", "end_t"),
    "cues": ("index", "track", "text", "start_t", "end_t"),
    "embeddings": ("index", "track", "start_t", "end_t", "vector"),
    "attachments": ("index", "filename", "mimetype"),
}

EXPECTED_ROW_READONLY_FIELDS = {
    "audio": frozenset({"index", "codec", "channels", "channel_layout",
                        "sample_rate", "bitrate", "duration"}),
    "video": frozenset({"index", "codec", "width", "height", "fps", "bitrate",
                        "duration", "color_transfer"}),
    "subtitle": frozenset({"index", "codec"}),
    "data": frozenset({"index", "codec"}),
    "chapters": frozenset({"index"}),
    "cues": frozenset({"index", "track"}),
    "embeddings": frozenset({"index", "track"}),
    "attachments": frozenset({"index"}),
}


# The map columns and the record each holds.
EXPECTED_MAP_ELEMENTS = {"tags": "tag", "disposition": "flag"}

# Every key `ffprobe -show_entries stream_disposition` prints, in its order.
EXPECTED_DISPOSITION_KEYS = (
    "default",
    "dub",
    "original",
    "comment",
    "lyrics",
    "karaoke",
    "forced",
    "hearing_impaired",
    "visual_impaired",
    "clean_effects",
    "attached_pic",
    "timed_thumbnails",
    "non_diegetic",
    "captions",
    "descriptions",
    "metadata",
    "dependent",
    "still_image",
    "multilayer",
)


def _all_fields() -> list[tuple[str, Field]]:
    return [(declared.name, entry) for declared in TYPES.values() for entry in declared.fields]


def test_every_declared_type_is_keyed_by_its_own_name() -> None:
    for name, declared in TYPES.items():
        assert declared.name == name


def test_kinds_are_the_declared_ones() -> None:
    assert {name: declared.kind for name, declared in TYPES.items()} == EXPECTED_KINDS


def test_every_field_type_reference_resolves() -> None:
    for owner, entry in _all_fields():
        assert element_type(entry.type) in TYPES, f"{owner}.{entry.name}"
        assert resolve(entry.type) is TYPES[element_type(entry.type)]


def test_fields_match_the_reference_tables() -> None:
    """Name, type reference and W/RO mark, per type, in declaration order."""
    declared = {
        name: {f.name: (f.type, "W" if f.writable else "RO") for f in type_.fields}
        for name, type_ in TYPES.items()
    }
    assert declared == EXPECTED_FIELDS
    for name, type_ in TYPES.items():
        assert [f.name for f in type_.fields] == list(EXPECTED_FIELDS[name])


def test_field_names_are_unique_within_a_type() -> None:
    for declared in TYPES.values():
        names = [entry.name for entry in declared.fields]
        assert len(names) == len(set(names)), declared.name


def test_exposed_marks_are_the_declared_ones() -> None:
    unexposed = {f"{owner}.{f.name}" for owner, f in _all_fields() if not f.exposed}
    assert unexposed == EXPECTED_UNEXPOSED


def test_readable_marks_are_the_declared_ones() -> None:
    unreadable = {f"{owner}.{f.name}" for owner, f in _all_fields() if not f.readable}
    assert unreadable == EXPECTED_UNREADABLE
    # A write-only field is still one a record literal supplies.
    written = [f.name for f in types.RECORD_FIELDS["attachment"]]
    assert "path" in written and "path" not in types.ROW_SCHEMAS["attachments"]


def test_a_map_type_is_a_key_and_one_more_field() -> None:
    for declared in TYPES.values():
        if declared.kind == "map":
            assert [f.name for f in declared.fields][0] == "key"
            assert len(declared.fields) == 2


def test_every_array_field_elements_a_declared_record() -> None:
    for owner, entry in _all_fields():
        if is_array(entry.type):
            assert resolve(entry.type).kind in {"stream", "record", "map"}, (
                f"{owner}.{entry.name}"
            )


def test_scalars_and_handles_have_no_fields() -> None:
    for declared in TYPES.values():
        if declared.kind in {"scalar", "handle"}:
            assert declared.fields == ()


def test_column_types_are_declared_types() -> None:
    for name in COLUMN_TYPES:
        assert name in TYPES


def test_field_lookup() -> None:
    audio = TYPES["audio_stream"]
    assert audio.field("channels") == Field("channels", "number", writable=False)
    assert audio.field("width") is None


def test_resolve_rejects_an_undeclared_name() -> None:
    with pytest.raises(ValueError, match="undeclared type 'nope'"):
        resolve("nope[]")


def test_element_type_and_is_array() -> None:
    assert is_array("tag[]") and not is_array("tag")
    assert element_type("tag[]") == "tag"
    assert element_type("tag") == "tag"


def test_declarations_are_frozen() -> None:
    with pytest.raises(FrozenInstanceError):
        TYPES["chapter"].fields[0].name = "other"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        TYPES["chapter"].name = "other"  # type: ignore[misc]


def test_row_schemas_view() -> None:
    assert types.ROW_SCHEMAS == EXPECTED_ROW_SCHEMAS
    for column, schema in EXPECTED_ROW_SCHEMAS.items():
        assert list(types.ROW_SCHEMAS[column]) == list(schema), column


def test_row_common_view() -> None:
    assert types.ROW_COMMON == EXPECTED_ROW_COMMON


def test_map_element_view() -> None:
    assert types.MAP_ELEMENTS == EXPECTED_MAP_ELEMENTS
    assert types.TAGS_COLUMN in types.MAP_ELEMENTS
    assert types.DISPOSITION_COLUMN in types.MAP_ELEMENTS


def test_disposition_keys_are_the_closed_set() -> None:
    """The key set ffmpeg itself reports, in ffprobe's own order."""
    assert types.DISPOSITION_KEYS == EXPECTED_DISPOSITION_KEYS
    assert list(types.ROW_COMMON) == list(EXPECTED_ROW_COMMON)


def test_column_set_views() -> None:
    assert types.INPUT_COLUMNS == EXPECTED_INPUT_COLUMNS
    assert types.UNNEST_COLUMNS == EXPECTED_UNNEST_COLUMNS
    assert types.STREAM_ARRAY_COLUMNS == EXPECTED_STREAM_ARRAY_COLUMNS


def test_star_column_views() -> None:
    assert types.STAR_COLUMNS == EXPECTED_STAR_COLUMNS
    assert types.ROW_STAR_COLUMNS == EXPECTED_ROW_STAR_COLUMNS
    # A star over a row never carries a map column: those are read by name.
    for column, names in types.ROW_STAR_COLUMNS.items():
        schema = types.ROW_SCHEMAS[column]
        assert names == tuple(n for n in schema if n not in ("tags", "disposition"))


def test_read_only_field_views() -> None:
    assert types.ROW_READONLY_FIELDS == EXPECTED_ROW_READONLY_FIELDS
    assert types.CONTAINER_READONLY_FIELDS == frozenset(
        {"t", "duration", "cues", "embeddings"}
    )
    # On a STREAM row the writable half is exactly the maps, which have
    # spellings of their own; a chapter's title and bounds are writable too.
    for column in types.STREAM_ARRAY_COLUMNS:
        writable = set(types.ROW_SCHEMAS[column]) - types.ROW_READONLY_FIELDS[column]
        assert writable == {"tags", "disposition"}


def test_named_column_views() -> None:
    assert types.STREAM_TAG_COLUMNS == ("language", "title")
    assert types.TAGS_COLUMN == "tags"
    assert types.CHAPTERS_COLUMN == "chapters"
    assert types.ATTACHMENTS_COLUMN == "attachments"
    assert types.ATTACHMENT_TYPE == "attachment"
    assert types.EMBEDDINGS_COLUMN == "embeddings"
    assert types.EMBEDDING_TYPE == "embedding"
    assert types.TRACK_RECORD_COLUMNS == frozenset({"cues", "embeddings"})
    assert types.INPUT_DURATION_COLUMN == "duration"
    assert types.TIME_COLUMN == "t"


def test_parser_reexports_are_the_view_objects() -> None:
    assert parser.ROW_SCHEMAS is types.ROW_SCHEMAS
    assert parser.TAGS_COLUMN == types.TAGS_COLUMN
    assert parser.INPUT_COLUMNS is types.INPUT_COLUMNS
    assert parser.UNNEST_COLUMNS is types.UNNEST_COLUMNS
    assert parser.STREAM_ARRAY_COLUMNS is types.STREAM_ARRAY_COLUMNS
    assert parser.CHAPTERS_COLUMN == types.CHAPTERS_COLUMN
    assert parser.INPUT_DURATION_COLUMN == types.INPUT_DURATION_COLUMN


def test_lower_stream_array_map_covers_the_stream_columns() -> None:
    """lower keeps its own column -> IR StreamType map; the keys must agree."""
    assert set(lower._ARRAY_COLUMNS) == types.STREAM_ARRAY_COLUMNS
    assert set(lower._PASSTHROUGH_ONLY) <= types.STREAM_ARRAY_COLUMNS
