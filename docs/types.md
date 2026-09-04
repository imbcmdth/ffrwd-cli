# Types

Every column, value and stream in the dialect has a declared type, and
error messages name them. This page is the vocabulary; the per-shape
column tables are in [rows.md](rows.md).

## The kinds

| kind | types | what it is |
| --- | --- | --- |
| scalar | `text`, `number`, `boolean`, `vector` | a compile-time value. `number` follows Postgres typing (int/int truncates); `boolean` comes from a flag map and stands alone as a predicate |
| stream record | `video_stream`, `audio_stream`, `subtitle_stream`, `data_stream` | one track: **the record IS the stream**, plus the metadata about it |
| record | `chapter`, `cue`, `attachment`, `embedding` | data the container carries that is not a stream |
| map | `tag`, `flag` | key/value pairs read by path, never unnested |
| container | `container` | one input file: its stream arrays, its chapter list, its scalars |
| array | `T[]` | `unnest` turns it into rows of `T` |

`input('film.mkv') f` is a table of ONE `container` row.
`unnest(f.audio) a` is rows of `audio_stream`.

`vector` is a scalar with no literal: a list of numbers, which reaches a
query as a `LANGUAGE wasm` value function's own domain (a parameter, a
`RETURNS`, an annotation field) or as the `vector` field of an
`embedding` row read out of a file's vector track. A vector reads and
prints (capped, as a cell), but does not compare, concatenate, or cast to
text; `cos_similarity(vector, vector)` and `vector_length(vector)` are the
two functions that read one, each returning `number`, and a tag may not
carry one at all.

`embedding` is the record over it: `index`, `track`, `start_t`, `end_t`,
`vector`, the same read/write split a `cue` has. An array of them in a
stream position IS a vector track, the way an array of cues is a subtitle
track ([rows.md](rows.md#embedding-rows---unnestfembeddings-v)).

## The record is the stream

A stream record is what filters take and return, what `-map` maps, and
what `SELECT a` selects. The graph node behind it has no name in the
language. Two records are the same stream or they are not: identity is
which stream, never a field-by-field comparison, so `GROUP BY a` keeps
two tracks apart even when every probed column agrees.

A filter's output is a stream whose fields were never probed, so it has
none to read: `scale(v, 640, -2).width` is a typed rejection, not a
NULL. Read the field on what goes in.

## Writable and read-only fields

Every field is one or the other, and the distinction is enforced:

- **Writable** — an assertion your query may make: a stream's `tags`
  and `disposition`, a container's `tags`, a chapter's, cue's,
  embedding's or attachment's own fields. A `tags` map is written with a
  `tags` column (`STRUCT('eng' AS language) AS tags`, a `NULL` field
  clears); a record's fields are written by name in a literal,
  `STRUCT('Intro' AS title, 0 AS start_t, 60 AS end_t)::chapter`.
- **Write-only** — one field, `attachment.path`: it names the file to
  attach when constructing a record and has nothing to report back, so
  reading it is a rejection.
- **Read-only** — a probed fact: `index`, `track`, `codec`, `width`,
  `height`, `fps`, `channels`, `sample_rate`, `channel_layout`,
  `bitrate`, `duration`, `color_transfer`. Setting one is a typed rejection
  naming the field as probed.

The reserved keys are the read-only fields **of the record the column
sits over**, so `STRUCT(1920 AS width) AS tags` is rejected over a video
row and is an ordinary tag over an audio row, which has no `width`.
`disposition` is reserved everywhere - it is the row's own field, not
metadata. Every other field name is a free-form tag key.

## Maps: tags and disposition

`tags` (free-form keys) and `disposition` (ffmpeg's closed flag set)
are maps, read by path:

```sql
SELECT t.tags.language, t.tags.title, t.disposition.forced
FROM input('film.mkv') f, unnest(f.audio) t
```

An absent key reads NULL. A disposition key outside ffmpeg's set is a
typed rejection with a did-you-mean. Naming a map without a key
(`t.tags`) is a value-position rejection, though it prints as one cell
of `(key,value)` records in a table query - handy for seeing every tag
a file carries.

Writing: `STRUCT('eng' AS language) AS tags` sets that entry and
leaves the rest of the map alone, and a `NULL` field clears its key.
`||` merges two maps with the right side winning, so
`t.tags || STRUCT(...) AS tags` is copy-and-override.
`'default+forced' AS disposition` stays its own column and sets the
whole flag map (a relative spec like `'+forced'` is rejected - the
column sets the map, so there is nothing to adjust); `NULL` clears it.

Tags that ride: only `language` and `title` follow a stream through a
filter to the output. The rest describe the source.

## `SELECT *`

- Over a container: its stream arrays, video/audio/subtitle/data - the
  remux shape. In a table query its `chapters` and `attachments` join
  them; `cues` and `embeddings` are read-only and stay out.
- Over rows: the record's scalar fields, the metadata table. Map
  columns are excluded (a disposition cell is 250 characters wide);
  name them when you want them.
- Over rows in a **media** query: a typed rejection. A row's star means
  its fields, and fields are not output streams - select the row
  itself.
