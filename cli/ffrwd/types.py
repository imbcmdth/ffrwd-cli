"""The built-in types, declared as data.

Every shape the compiler knows is declared once here — the scalars, the four
stream records, the non-stream records, and the `container` an `input()`
alias is one row of. The column sets parser and lower work with
(:data:`ROW_SCHEMAS`, :data:`INPUT_COLUMNS`, :data:`UNNEST_COLUMNS`, ...) are
VIEWS over these declarations, computed at the bottom of this module. Nothing
else may restate them.

A :class:`Type` is a name, a kind, and an ordered tuple of :class:`Field`. A
field's ``type`` is another type's name, ``[]``-suffixed for an array.

Kinds:

    scalar     ``text``, ``number``, ``boolean`` — Postgres typing; boolean
               is what a `flag` entry holds and what a predicate answers,
               never a column of its own.
    handle     internal, with no name in the language: ``stream`` is the
               graph node behind a stream record — no field carries it,
               since the record IS the stream — and ``seek`` is
               ``<alias>.t``.
    stream     a record ABOUT a stream: ``video_stream``, ``audio_stream``,
               ``subtitle_stream``, ``data_stream``. The record IS the
               stream; identity is nominal, never field-by-field.
    record     a record that is not a stream and is a set of rows:
               ``chapter``, ``attachment``, ``cue``. An array of one unnests.
    map        a key/value record read by path off the column that holds it:
               ``tag``, ``flag``. ``f.tags.title`` names one entry; the array
               is never a set of rows, so it never unnests.
    container  the type of an `input()` row.

Field flags:

    writable   W: an assertion a query may make about the value, emitted as
               that stream's or container's metadata. Its opposite covers
               both a probed fact (RO — setting it is a rejection) and a
               handle, which is not a value a query can assert either way.
    exposed    the compiler surfaces this field TODAY. A field declared here
               but not yet wired up carries ``exposed=False`` and stays out
               of every view.
    readable   the field is a COLUMN of the row a query reads. Its opposite is
               write-only: ``attachment.path`` is the file to read when one is
               written, and a container carries the bytes rather than where
               they came from, so there is nothing to read it back from.

Field ORDER is column order: a row prints its fields in declaration order,
so every declaration below is written in the order that row already prints.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

__all__ = [
    "ATTACHMENTS_COLUMN",
    "ATTACHMENT_TYPE",
    "CHAPTERS_COLUMN",
    "CHAPTER_TYPE",
    "COLUMN_TYPES",
    "CONTAINER_READONLY_FIELDS",
    "CUES_COLUMN",
    "CUE_TYPE",
    "DISPOSITION_COLUMN",
    "DISPOSITION_KEYS",
    "INPUT_COLUMNS",
    "INPUT_DURATION_COLUMN",
    "MAP_ELEMENTS",
    "RECORD_ARRAY_COLUMNS",
    "RECORD_COLUMNS",
    "RECORD_ELEMENTS",
    "RECORD_FIELDS",
    "ROW_COMMON",
    "ROW_READONLY_FIELDS",
    "ROW_SCHEMAS",
    "ROW_STAR_COLUMNS",
    "STAR_COLUMNS",
    "STREAM_ARRAY_COLUMNS",
    "STREAM_TAG_COLUMNS",
    "TAGS_COLUMN",
    "TIME_COLUMN",
    "TYPES",
    "UNNEST_COLUMNS",
    "Field",
    "RowColumnType",
    "Type",
    "TypeKind",
    "element_type",
    "is_array",
    "resolve",
]

TypeKind = Literal["scalar", "handle", "stream", "record", "map", "container"]

# The compile-time type of a row column. `text` and `number` are probed
# metadata, comparable against literals of the matching kind and nothing else;
# `stream` is what a bare row alias types as — the row itself, the only thing
# that can BE an output; `tag[]` and `flag[]` are map columns, read one key at
# a time by path and never comparable whole.
RowColumnType = str  # "stream" | "text" | "number" | "tag[]" | "flag[]"

COLUMN_TYPES = frozenset({"stream", "text", "number"})

_ARRAY_SUFFIX = "[]"


@dataclass(frozen=True)
class Field:
    """One field of a type: a name, a type reference, and its flags."""

    name: str
    type: str
    writable: bool
    exposed: bool = True
    readable: bool = True


@dataclass(frozen=True)
class Type:
    """One declared type: a name, a kind, and its fields in column order."""

    name: str
    kind: TypeKind
    fields: tuple[Field, ...] = ()

    def field(self, name: str) -> Field | None:
        """The field called `name`, or None."""
        for entry in self.fields:
            if entry.name == name:
                return entry
        return None

    def columns(self) -> dict[str, RowColumnType]:
        """The exposed fields that are row columns, in declaration order."""
        return {f.name: f.type for f in self.fields if f.exposed and _is_column(f)}


def is_array(ref: str) -> bool:
    """True if a field's type reference is an array."""
    return ref.endswith(_ARRAY_SUFFIX)


def element_type(ref: str) -> str:
    """The element type name of a reference; the reference itself if scalar."""
    return ref[: -len(_ARRAY_SUFFIX)] if is_array(ref) else ref


def resolve(ref: str) -> Type:
    """The declared type a field's type reference names.

    Raises ValueError for an undeclared name — a programmer error, since
    every reference here is written in this module.
    """
    name = element_type(ref)
    declared = TYPES.get(name)
    if declared is None:
        raise ValueError(f"undeclared type '{name}'")
    return declared


def _is_column(f: Field) -> bool:
    """True if a field is a row column: a scalar one, or a map array."""
    if not f.readable:
        return False
    return f.type in COLUMN_TYPES or (is_array(f.type) and resolve(f.type).kind == "map")


def _ro(name: str, type_: str, *, exposed: bool = True) -> Field:
    """A probed fact or a handle: not an assertion the query can make."""
    return Field(name, type_, writable=False, exposed=exposed)


def _w(name: str, type_: str, *, exposed: bool = True, readable: bool = True) -> Field:
    """A writable field: an assertion the query may make. NULL clears it."""
    return Field(name, type_, writable=True, exposed=exposed, readable=readable)


def _stream_type(name: str, *probed: Field) -> Type:
    """A stream record: the shared head, this kind's probed fields, the maps.

    `index` is 1-BASED, deliberately: it is the same number
    ``<alias>.<array>[k]`` takes, so ``WHERE t.index = 1`` and ``f.audio[1]``
    name the same stream. (probe's `StreamMeta.index` is 0-based; lower does
    the +1.) `language` and `title` are entries of `tags`, read by path.
    """
    return Type(
        name,
        "stream",
        (
            _ro("index", "number"),
            _w("tags", "tag[]"),
            _w("disposition", "flag[]"),
            _ro("codec", "text"),
            *probed,
        ),
    )


# The tag keys that RIDE a filter through to its output stream, so a filtered
# track keeps saying what its source said. Deliberately not every key the
# source carries: an `encoder` or `handler_name` tag riding through would emit
# `-metadata` ffmpeg does not emit today and move bytes nobody asked to move.
STREAM_TAG_COLUMNS = ("language", "title")

# Every disposition key ffmpeg knows, in the order ffprobe prints them. The set
# is CLOSED: `a.disposition.<key>` reads one of these and nothing else, and a
# written spec names them. Taken from `ffprobe -show_entries stream_disposition`.
DISPOSITION_KEYS = (
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

_DECLARED: tuple[Type, ...] = (
    Type("text", "scalar"),
    Type("number", "scalar"),
    Type("boolean", "scalar"),
    # Internal, unnameable in the language: the graph node a stream record
    # stands for, and the seek handle `<alias>.t`.
    Type("stream", "handle"),
    Type("seek", "handle"),
    # Free-form keys: `f.tags.title` reads one entry, `'eng' AS language`
    # writes one, and a key the file does not carry reads NULL.
    Type("tag", "map", (_w("key", "text"), _w("value", "text"))),
    # The closed key set is :data:`DISPOSITION_KEYS`; a key outside it is a
    # typed rejection.
    Type("flag", "map", (_w("key", "text"), _w("set", "boolean"))),
    _stream_type(
        "video_stream",
        _ro("width", "number"),
        _ro("height", "number"),
        _ro("fps", "text"),  # verbatim "30000/1001", exactly as ffprobe prints it
        _ro("bitrate", "number"),
        _ro("duration", "number"),
        _ro("color_transfer", "text"),
    ),
    _stream_type(
        "audio_stream",
        _ro("channels", "number"),
        _ro("channel_layout", "text"),
        _ro("sample_rate", "number"),
        _ro("bitrate", "number"),
        _ro("duration", "number"),
    ),
    # Subtitle and data streams carry the common head only: a caption track
    # has no dimensions, no channels and no frame rate, and columns a probe
    # never fills would just be NULL on every row. `data` is the bucket for
    # what ffprobe cannot classify — never attachments, never cues.
    _stream_type("subtitle_stream"),
    _stream_type("data_stream"),
    # A chapter is not a stream: nothing to select as output, only metadata
    # to read or filter on. `index` is ffprobe's own chapter order, 1-based.
    Type(
        "chapter",
        "record",
        (
            _ro("index", "number"),
            _w("title", "text"),
            _w("start_t", "number"),  # seconds
            _w("end_t", "number"),  # seconds
        ),
    ),
    # A file riding inside the container: a subtitle font, cover art, a
    # script. Not a stream ffrwd maps -- ffmpeg attaches it by path.
    # `index` is the attachment's order in the file, 1-based.
    Type(
        "attachment",
        "record",
        (
            _ro("index", "number"),
            _w("filename", "text"),
            _w("mimetype", "text"),
            # The file to read when one is written. A container carries the
            # bytes, not where they came from, so it is write-only.
            _w("path", "text", readable=False),
        ),
    ),
    # A cue is a chapter's twin: a caption over a time span. Same field order,
    # so the two convert into each other by swapping the cast. `index` is the
    # cue's place in the WebVTT document, 1-based.
    Type(
        "cue",
        "record",
        (
            _ro("index", "number"),
            _w("text", "text"),
            _w("start_t", "number"),  # seconds
            _w("end_t", "number"),  # seconds
        ),
    ),
    Type(
        "container",
        "container",
        (
            _w("video", "video_stream[]"),
            _w("audio", "audio_stream[]"),
            # Same array/subscript/splat surface as video/audio, but
            # passthrough-only downstream (lower enforces that half).
            _w("subtitle", "subtitle_stream[]"),
            _w("data", "data_stream[]"),
            _w("chapters", "chapter[]"),
            # Read-only: a WebVTT document's cues, parsed by ffrwd itself.
            # A cue array is WRITTEN in a stream position (it IS a subtitle
            # track), never as a column named `cues`.
            _ro("cues", "cue[]"),
            _w("attachments", "attachment[]"),
            # The seek handle: legal only in a WHERE trim window, never a
            # value and never part of SELECT *.
            _ro("t", "seek"),
            # The probed container length in seconds. A value, never a
            # stream, so it is only legal inside a compile-time expression;
            # lower reads it off the probe.
            _ro("duration", "number"),
            # The container's own tags, free-form keys, read by path off the
            # probe's `format.tags`.
            _w("tags", "tag[]"),
        ),
    ),
)

TYPES: dict[str, Type] = {declared.name: declared for declared in _DECLARED}

_CONTAINER = TYPES["container"]


def _sole(names: tuple[str, ...], what: str) -> str:
    """The one name a singleton view derives.

    Raises ValueError when the declarations above stopped agreeing with the
    view — a programmer error, and what test_types.py guards.
    """
    if len(names) != 1:
        raise ValueError(f"{what}: expected one field, found {list(names)}")
    return names[0]


def _container_fields() -> tuple[Field, ...]:
    """The exposed container fields, in declaration order."""
    return tuple(f for f in _CONTAINER.fields if f.exposed)


def _array_columns(*kinds: TypeKind, writable: bool | None = None) -> tuple[str, ...]:
    """The exposed container array columns whose elements have those kinds.

    `writable` narrows to the columns a query may assert (or, False, to the
    ones it may not); None takes both.
    """
    return tuple(
        f.name
        for f in _container_fields()
        if is_array(f.type)
        and resolve(f.type).kind in kinds
        and (writable is None or f.writable == writable)
    )


# --- Views. Everything below is derived; nothing below declares. ---

# The record-array column an input's chapter list lives in. An array VALUE
# like the stream arrays, unnested the same way, but its records are not
# streams — no subscript, nothing to select as output.
CHAPTERS_COLUMN = _sole(
    tuple(f.name for f in _CONTAINER.fields if element_type(f.type) == "chapter"),
    "the chapter array column",
)

# Each record-array column of a container and the record it holds: `chapters`
# holds chapters, `cues` holds cues. Unnesting one gives STREAMLESS rows -- a
# record is metadata the container carries, never a track.
RECORD_ELEMENTS: dict[str, str] = {
    f.name: element_type(f.type)
    for f in _container_fields()
    if is_array(f.type) and resolve(f.type).kind == "record"
}

RECORD_ARRAY_COLUMNS = frozenset(RECORD_ELEMENTS)

# The same pairing read the other way: which column each record type lives in.
RECORD_COLUMNS: dict[str, str] = {
    record: column for column, record in RECORD_ELEMENTS.items()
}

# The record type the chapter column holds, and the one a written chapter
# casts to.
CHAPTER_TYPE = RECORD_ELEMENTS[CHAPTERS_COLUMN]

# The container column an input's cue list lives in, and the record it holds.
CUES_COLUMN = _sole(
    tuple(name for name, record in RECORD_ELEMENTS.items() if record == "cue"),
    "the cue array column",
)
CUE_TYPE = RECORD_ELEMENTS[CUES_COLUMN]

# The container column an input's attachment list lives in, and the record it
# holds.
ATTACHMENTS_COLUMN = _sole(
    tuple(name for name, record in RECORD_ELEMENTS.items() if record == "attachment"),
    "the attachment array column",
)
ATTACHMENT_TYPE = RECORD_ELEMENTS[ATTACHMENTS_COLUMN]

# The scalar pseudo-column every INPUT alias carries.
INPUT_DURATION_COLUMN = _sole(
    tuple(f.name for f in _container_fields() if f.type == "number"),
    "the container's scalar column",
)

# The seek-handle column, legal only in a WHERE trim window.
TIME_COLUMN = _sole(
    tuple(f.name for f in _container_fields() if f.type == "seek"),
    "the container's seek column",
)

# The structural column names an INPUT alias exposes. A CTE alias exposes
# whatever its body named with AS, so the whitelist does not apply there (lower
# checks those).
INPUT_COLUMNS = frozenset(f.name for f in _container_fields())

# The map column the container and every stream record carry their tags in.
# Read by path (`f.tags.title`), written as a tag column (`'eng' AS language`).
TAGS_COLUMN = _sole(
    tuple(f.name for f in _CONTAINER.fields if element_type(f.type) == "tag"),
    "the container's tag map column",
)

# The map column every stream record carries its disposition flags in. Read by
# path (`a.disposition.forced`), written as a flag column
# (`'default+forced' AS disposition`).
DISPOSITION_COLUMN = _sole(
    tuple(
        dict.fromkeys(
            f.name
            for declared in _DECLARED
            if declared.kind == "stream"
            for f in declared.fields
            if element_type(f.type) == "flag"
        )
    ),
    "the stream record's flag map column",
)

# Each map column and the record it holds: `tags` holds tags, `disposition`
# holds flags. A rejection names the record, not the column.
MAP_ELEMENTS: dict[str, str] = {
    f.name: element_type(f.type)
    for declared in _DECLARED
    if declared.kind in {"stream", "container"}
    for f in declared.fields
    if f.exposed and is_array(f.type) and resolve(f.type).kind == "map"
}

# The fields a record LITERAL names, per record type: the writable ones, in
# declaration order. A read-only field is ffprobe's own (`chapter.index` is
# the order the container already lists), so a query never supplies it. A
# record type with nothing exposed yet has no entry at all.
RECORD_FIELDS: dict[str, tuple[Field, ...]] = {
    declared.name: written
    for declared in _DECLARED
    if declared.kind == "record"
    and (written := tuple(f for f in declared.fields if f.exposed and f.writable))
}

# The array columns whose elements are STREAMS: the only ones a subscript or
# a `[k].<column>` metadata accessor reaches.
_STREAM_ARRAYS = _array_columns("stream")
STREAM_ARRAY_COLUMNS = frozenset(_STREAM_ARRAYS)

# The array columns `unnest(<alias>.<name>)` accepts: every array of records.
# `t` is a timeline and `duration` is a scalar, and neither is a set of rows.
UNNEST_COLUMNS = frozenset(_array_columns("stream", "record"))

# What `SELECT *` expands a container into: the array columns a copy carries
# through, in declaration order -- the four stream arrays, then the record
# arrays. Never the scalars (`duration` is a value, `t` a seek handle, `tags`
# a map read by key), and never a READ-ONLY array: `cues` are read out of a
# WebVTT document by name, and a file that is not one has none to show.
STAR_COLUMNS: tuple[str, ...] = _array_columns("stream", "record", writable=True)

# The container fields a query may not assert: probed facts and handles. The
# reserved set an aliased column is checked against before it counts as a
# free-form tag key.
CONTAINER_READONLY_FIELDS = frozenset(f.name for f in _container_fields() if not f.writable)

# The columns each unnest row table exposes, per container array column. The
# row's own stream is not among them: the row IS the stream.
ROW_SCHEMAS: dict[str, dict[str, RowColumnType]] = {
    f.name: resolve(f.type).columns() for f in _container_fields() if f.name in UNNEST_COLUMNS
}

# What `SELECT *` expands a row into, per container array column: the record's
# scalar fields, in declaration order. The map columns stay out -- one
# disposition cell is every flag ffmpeg knows -- and are read by name instead.
ROW_STAR_COLUMNS: dict[str, tuple[str, ...]] = {
    column: tuple(name for name, type_ in schema.items() if type_ in COLUMN_TYPES)
    for column, schema in ROW_SCHEMAS.items()
}

# The fields of each row type a query may not assert, per container array
# column: probed facts, not assertions.
ROW_READONLY_FIELDS: dict[str, frozenset[str]] = {
    f.name: frozenset(
        entry.name for entry in resolve(f.type).fields if entry.exposed and not entry.writable
    )
    for f in _container_fields()
    if f.name in UNNEST_COLUMNS
}

# What every stream row carries, whatever its type.
ROW_COMMON: dict[str, RowColumnType] = {
    name: type_
    for name, type_ in ROW_SCHEMAS[_STREAM_ARRAYS[0]].items()
    if all(name in ROW_SCHEMAS[column] for column in _STREAM_ARRAYS)
}
