# Row shapes

Every FROM item is a compile-time table with a fixed shape. This page lists each shape, its columns, and what each column is for. Column values are ffprobe results, so inputs must be probeable wherever a metadata column is read (typed rejection otherwise); everything here evaluates at compile time - no filter, join, or sort reaches the ffmpeg command, only the wiring they decided.

The type vocabulary - what a stream record is, which fields a query may set, how the tag and disposition maps work - is [types.md](types.md). Here: the shapes and what you do with them. Unreported values are NULL with SQL semantics: `=` and `!=` both fail against NULL; use `IS [NOT] NULL`.

## Input rows - `input('film.mkv') f`

One row per input: the shape of a container file. Arrays of streams plus the container's own scalars.

| column | type | notes |
| --- | --- | --- |
| `video`, `audio`, `subtitle`, `data` | stream array | splat (`f.audio` = every track), subscript (`f.audio[1]`, 1-based), or `unnest` into track rows. `subtitle`/`data` are passthrough-only |
| `chapters` | record array | `unnest` into chapter rows; no splat, no subscript. Bare, it prints as one array cell |
| `attachments` | record array | files riding inside the container - `unnest` into attachment rows |
| `cues` | record array | a WebVTT file's caption cues, or every caption track's, read-only - `unnest` into cue rows |
| `embeddings` | record array | every vector track's rows, read-only - `unnest` into embedding rows |
| `t` | timeline | only in `WHERE` trim windows: `f.t BETWEEN 5 AND 60`, either bound alone, or against chapter bounds |
| `duration` | number | probed container duration in seconds |
| `tags` | tag map | the container's own tags, read by path: `f.tags.title`, `f.tags.artist`, any key. NULL when the file doesn't carry it. Bare, it prints as one array cell of `(key,value)` records |

Subscripts reach track-row columns without unnest: `f.audio[1].tags.language` (strict-Postgres `(f.audio[1]).tags.language` also parses). In a `WHERE` this is an **assertion** - the subscript names one track, so a false predicate refuses to compile ([recipe 29](examples.md#29-assert-what-youre-shipping)).

Only an input-side read has facts to report. A field read off a FILTER OUTPUT - `scale(f.video[1], 640, -2).width`, `volume(t, 0.2).tags.language` - is a typed rejection: nothing probed that stream, and the hint names the input-side read to write instead.

`SELECT *` over an input alias is its ARRAY columns, never the scalars. In a media query that is the four stream arrays in `video`, `audio`, `subtitle`, `data` order, each a passthrough - the remux shape; chapters ride through as ffmpeg's own default. In a table/CSV query it is every writable array column - the four stream arrays plus `chapters` and `attachments` - one cell each; `cues` and `embeddings` are read-only and stay out. `f.*` does one alias, a bare `*` every alias in `FROM` order.

## Rendition rows - `input('ladder.m3u8') r`

`input()` on an HLS master playlist or a DASH MPD reads one row per ABR rendition instead of one row for the file - a manifest input IS a row table, the way `unnest()` makes one.

| column | type | notes |
| --- | --- | --- |
| `video`, `audio` | stream array | this rendition's own streams; subscript (`r.video[1]`) or unnest, same as an input row's |
| `bandwidth` | number | HLS BANDWIDTH / MPD @bandwidth |
| `width`, `height` | number | from the rendition's video stream |
| `codecs` | text | HLS CODECS / MPD @codecs, verbatim |
| `name` | text | HLS NAME, else the variant playlist's directory; a DASH Representation's id |
| `language` | text | HLS LANGUAGE / MPD @lang, else NULL |

`WHERE`, `ORDER BY` and `LIMIT` over these columns filter and rank renditions the same as track rows do below ([recipes 105-106](examples.md#105-pick-a-rung-from-an-abr-ladder)); a `format 'hls'`/`format 'dash'` destination fed every surviving rendition writes a new ladder, rung for rung ([recipe 107](examples.md#107-re-encode-an-abr-ladder-through-to-another-abr-ladder)). Carried through a CTE they stay rows: `r.video[1]` in a body is THAT row's video, one cell per surviving rendition, NULL where the rung carries no track of the kind ([recipe 123](examples.md#123-re-lay-a-muxed-ladder-as-a-demuxed-one)).

## Source rows - `ffrwd.moq.subscribe(:'relay', :'broadcast') s`

A `RETURNS source` wasm function in FROM is probed at compile time for
its catalog and reads exactly like a manifest input: one row per
rendition, the same columns as rendition rows above, `WHERE`, `ORDER
BY` and `LIMIT` over them. A source that reports itself unbounded is
a live input. Its arguments are values only - a source reads no
streams.

A source may also be a values-world module: invoked once at compile
time rather than probed, each row it returns names an input ffmpeg
opens with `-i` itself, and the alias reads as rendition rows with
the module's own columns beside them ([recipe
114](examples.md#114-a-source-whose-rows-are-files-ffmpeg-opens-itself)).
Each row names a `url` ffmpeg opens and may carry `bandwidth`,
`codecs`, `name` or `language`; any other key is a value column typed
by its JSON value, while `width`/`height` still come from probing the
named url, same as any other rendition row. A row the query drops
leaves the command entirely.

## Track rows - `unnest(f.audio) t`

One row per track. The argument is an array column of an input declared earlier in the same FROM list; alias mandatory. All eight array columns unnest - the four stream arrays here, and `chapters`, `cues`, `embeddings` and `attachments` below. The schema varies by stream type:

The row IS the stream: a bare `t` where a stream is expected selects it, filters it, or gathers it. The columns below are the metadata ABOUT it.

| column | type | audio | video | subtitle | data |
| --- | --- | --- | --- | --- | --- |
| `index` | number | yes | yes | yes | yes |
| `tags` | tag map | yes | yes | yes | yes |
| `disposition` | flag map | yes | yes | yes | yes |
| `codec` | text | yes | yes | yes | yes |
| `channels`, `sample_rate` | number | yes | - | - | - |
| `channel_layout` | text | yes | - | - | - |
| `width`, `height` | number | - | yes | - | - |
| `fps` | text | - | verbatim, e.g. `30000/1001` | - | - |
| `color_transfer` | text | - | yes | - | - |
| `bitrate`, `duration` | number | yes | yes | - | - |

`index` is 1-based and agrees with the subscript: `WHERE t.index = 1` and `f.audio[1]` name the same track.

`SELECT t.*` is the row's scalar fields, in the order above - the metadata table. The map columns stay out (one `disposition` cell is every flag ffmpeg knows, which no table prints readably); name them to print them. In a media query a star over rows is a typed rejection: fields are not output streams, and the stream is the bare `t`.

`tags` is a map read by path - `t.tags.language`, `t.tags.title`, any key the file carries; absent reads NULL. There is no bare `t.language` spelling. Bare, `t.tags` prints the whole map as one array cell of `(key,value)` records.

`disposition` is the same shape over a CLOSED key set - the flags ffmpeg itself reports: `default`, `dub`, `original`, `comment`, `lyrics`, `karaoke`, `forced`, `hearing_impaired`, `visual_impaired`, `clean_effects`, `attached_pic`, `timed_thumbnails`, `non_diegetic`, `captions`, `descriptions`, `metadata`, `dependent`, `still_image`, `multilayer`. `t.disposition.forced` is a boolean; a key outside the set is a typed rejection. Bare, `t.disposition` prints as one array cell of `(key,set)` records. Writing it is under Tags below.

`WHERE` over row columns filters tracks; `ORDER BY` re-sorts them (multi-key, Postgres NULL placement) - without it, rows keep file order, which is player-visible and never changed implicitly. Both take the compile-time predicate grammar: `=`, `!=`, `<`, `<=`, `>`, `>=`, `BETWEEN`, `IS [NOT] NULL`, `AND`/`OR`/`NOT`, statically type-checked.

`LIMIT` and `OFFSET` narrow the row set after `WHERE` and `ORDER BY` and before grouping, the fan-out pin, and the one-row rule - `ORDER BY t.width DESC LIMIT 1` is the top row, no aggregate. They are legal exactly where `ORDER BY` is (any row-table branch, CTE bodies included). Counts are integer literals after `-v` substitution; `LIMIT 0` and an `OFFSET` that skips every row are rejections - a query that selects nothing is a mistake worth naming.

## Chapter rows - `unnest(f.chapters) c`

The same shape as track rows, over the container's chapter list. A chapter is not a stream, so a bare `c` selects nothing and an unaliased column of it in a media query is a typed rejection; the columns feed trim windows (`WHERE f.t BETWEEN c.start_t AND c.end_t`), fan-out destinations, `tags` fields, and table/CSV output. Chapter rows cross join with track rows like any other pair of sources.

| column | type | notes |
| --- | --- | --- |
| `index` | number | ffprobe's chapter order, 1-based |
| `title` | text | |
| `start_t`, `end_t` | number | seconds |

Writing chapters is the mirror: an aliased `chapters` column holding `chapter` records IS the output's chapter list - a literal `ARRAY[STRUCT('Intro' AS title, 0 AS start_t, 60 AS end_t)::chapter, ...]`, another input's list copied whole (`g.chapters AS chapters`), or `array_agg(STRUCT(...)::chapter)` gathered over any row source. `NULL AS chapters` writes none; omitting the column leaves ffmpeg's own default alone. It compiles to one extra self-contained input carrying the list. Recipes [39-40](examples.md#39-list-a-files-chapters) and [63](examples.md#63-copy-or-rebuild-a-chapter-list).

Written chapters are checked at compile time: `start_t` and `end_t` must be numbers (`title` may be `NULL`), each chapter must end after it starts, and the rows must run in ascending order without overlapping. Back-to-back is fine - one may end exactly where the next begins.

## Cue rows - `unnest(v.cues) c`

Caption cues, the same shape as chapters: a WebVTT file's, or every caption track a container carries. ffprobe does not enumerate cues, so ffrwd reads the document itself - extracting an embedded track once, cached like a probe. A file carrying no caption track reads zero rows.

| column | type | notes |
| --- | --- | --- |
| `index` | number | document order within its own track, 1-based |
| `track` | text | the title of the track it came from; NULL for a `.vtt` document and for an untitled track |
| `text` | text | the cue payload; multiple lines joined with a newline |
| `start_t`, `end_t` | number | seconds |

`unnest(f.cues['speech']) c` reads ONE track, the one titled `speech`. The subscript is an assertion, like a stream subscript: a title the file does not carry is a typed rejection listing the ones it does. Bounds are milliseconds - what WebVTT writes and what reads back.

Writing is the mirror: an array of `cue` records **in a stream position** IS a WebVTT subtitle track, minted as one self-contained input. Cues must each end after they start and run in ascending order; unlike chapters they may overlap, which WebVTT allows. Because a cue and a chapter are the same shape, converting either way is one `array_agg` - [recipe 65](examples.md#65-turn-chapters-into-a-subtitle-track-and-back).

An ALIAS on that column is the track's TITLE - `ARRAY[...] AS speech` emits `-metadata:s:<n> title=speech` - and it is the name `f.cues['speech']` finds it by afterwards. Several aliased columns are several tracks. A titled track is written to Matroska only: every other container rewrites or drops the title, so a destination that is not `.mkv` is refused by name. Untitled, the column writes wherever captions do.

## Embedding rows - `unnest(f.embeddings) v`

Every vector track's rows: a vector over a time span, and what it says about that stretch of the file.

| column | type | notes |
| --- | --- | --- |
| `index` | number | order within its own track, 1-based |
| `track` | text | the title of the track it came from |
| `start_t`, `end_t` | number | seconds |
| `vector` | vector | the numbers themselves; read by `cos_similarity`/`vector_length`, never compared |

A vector track rides in the same WebVTT document a caption track does, each block's text the row's numbers as little-endian f32 in base64, and the track's `vector_dims` tag says how many. That tag is what tells the two apart: a caption track has none, and a track that has one never shows up in `cues`. `unnest(f.embeddings['clip_vectors']) v` reads one track by title, the same as cues.

Writing is the mirror again: an array of `embedding` records **in a stream position** IS a vector track, `STRUCT(<start_t>, <end_t>, <vector>)::embedding`. There is no vector literal, so the `vector` field comes from a `RETURNS vector` wasm function or from another file's own rows. Every row of one track carries the same number of values - two lengths in one column is a rejection naming both - and the track's title is the column's alias. Matroska only, since no other container keeps `vector_dims`. [Recipes 115-116](examples.md#118-write-rows-as-titled-tracks).

## Attachment rows - `unnest(f.attachments) a`

Files riding inside the container: subtitle fonts, cover art, scripts.

| column | type | notes |
| --- | --- | --- |
| `index` | number | 1-based |
| `filename` | text | as stored |
| `mimetype` | text | as stored |

Writing takes a third field, `path` - the file to read - which is **write-only**: it names a source at construction time and has nothing to report back, so reading `a.path` is a rejection. `ARRAY[STRUCT('font.ttf' AS filename, 'application/x-truetype-font' AS mimetype, :'font' AS path)::attachment] AS attachments` attaches one; `filename` and `mimetype` may be `NULL` to take ffmpeg's defaults, `path` may not. [Recipe 66](examples.md#66-attach-a-font-and-list-what-a-file-carries).

## CTE rows - `WITH x AS (...)`

A CTE exposes whatever its body named with `AS`, and referencing it in FROM contributes its body's ROWS - a two-row CTE is a two-row source, and comma between sources is a cross join with real multiplicity, exactly as SQL says. A `tags` column in the body rides on its streams (see Tags below).

A body column that is a compile-time VALUE rather than a stream - a series value, a probed scalar, a row column, arithmetic over those - is a **value column** of the CTE's rows, readable wherever row columns are: `WHERE`, a fan-out `TO` expression, `GROUP BY`, a further CTE, and table/CSV output. `SELECT v AS frame, i.i AS n ...` gives the rows an `n` that names one file per row ([recipe 81](examples.md#81-grab-n-evenly-spaced-frames-one-file-each)). Selecting one as a column of a MEDIA query is a rejection: a SELECT column there is an output stream. A value the compiler cannot evaluate is a typed rejection naming the column. A `tags` column is not one of them either: it is spent on the body's own streams, so reading `<cte>.tags` is an unknown column. No other columns exist on a CTE alias; there is no natural naming from a bare `a`. Views referenced in FROM follow the same rules.

## Series rows - `generate_series(1, 5) i`

A count rather than a file: one row per integer in the range, computed at compile time from `start`, `stop`, and an optional `step` - a struct row table with its cells computed instead of written. The alias is mandatory, like every other call-shaped FROM item (`input()`, `unnest()`, `ffmpeg.<source>()`), and it names both the row table and its one column: `generate_series(1, 5) i` reads its value back as `i.i`, the same dot-qualified spelling any row column takes - there is no bare `i` for the value, and no other column.

`start`, `stop`, and `step` must be integer literals by the time this pass runs, which is after `-v` substitution - `generate_series(1, :count)` is fine, a column reference or any other computed expression is a typed rejection, because that is what keeps the row count (`stop - start` over `step`, inclusive) known before anything runs. A `0` step is rejected, and so is a range that would produce no rows (descending bounds under the default ascending step, or the reverse under a negative one): a series that silently produces nothing is a mistake worth naming, not a valid empty table.

The rows are streamless - no track, no `-i` - and behave exactly like a struct row: cross join them against an input or another row source with a comma, narrow them in `WHERE`, `array_agg` them, read `i.i` in a SELECT expression, key a fan-out `TO (expression)`, bound a trim window ([trimming.md](trimming.md#row-bounded-windows-one-seek-per-row)). [Recipes 74-75](examples.md#74-cut-a-file-into-n-clips-one-file-each) drive N files and one gathered file from the same count.

## Struct row tables - `unnest(ARRAY[STRUCT(...), ...]) r`

An inline written row table, its columns named by the STRUCT fields instead of a column list:

```sql
FROM input(:'source') f,
     unnest(ARRAY[STRUCT(1920 AS w, '1080p' AS name), STRUCT(1280 AS w, '720p' AS name)]) r
```

gives rows readable as `r.w`, `r.name`. Every STRUCT in the array declares the same field set, order-free - a mismatch is a typed rejection naming the odd field. A field's value takes the same compile-time value grammar any other value position does: a literal, or an expression over one (arithmetic, `CASE`, `||`, `::text`, an earlier alias's probed scalar); a stream inside is a typed rejection, since these are value rows, never streams. An empty array is a typed rejection too - the same posture `generate_series`'s empty range takes, not a silently empty table.

Otherwise it behaves like any other written row: cross join it with a comma, narrow it in `WHERE`, sort it, `array_agg` it, key a fan-out `TO (expression)` - [recipe 85](examples.md#85-key-an-encode-ladder-from-written-rows) keys an encode ladder off one, reading a rung's own width straight into a filter call's option. It joins `JOIN ... ON` with nothing, including itself: explicit JOIN stays reserved for `unnest` track rows.

## Joins

```sql
SELECT array_agg(amix(a, b))
FROM input('film.mkv') f, input('commentary.mkv') g,
     unnest(f.audio) a JOIN unnest(g.audio) b ON a.tags.language = b.tags.language
```

- `INNER`, `LEFT`, `FULL OUTER` between row tables - unnest tables, CTEs and views, struct row tables, `generate_series`; comma between them is a cross join. Joins at input level stay rejected.
- Result order: the left side's track order; a FULL join appends unmatched right rows after, in their order.
- Real join multiplicity: one row matching two pairs with both. To pair a 5.1 and a stereo English track separately, widen the key: `ON a.tags.language = b.tags.language AND a.channel_layout = b.channel_layout`.
- `ON` takes the same grammar as `WHERE`, column vs column or literal. A bare row alias is a stream, not a value to compare, so it is not usable inside `ON`.

A stream column's cardinality follows its relation: one cell per surviving row, NULL where the row carries no track of that kind, and a single stream on every row of the relation it is read beside - so one audio track is the audio of every rung ([recipe 124](examples.md#124-a-muxed-ladder-from-one-file)), and a one-row CTE joined into three rows is present on the row it matched and NULL on the others. A gathered array (`array_agg`, a bare input array) is one unit instead, and one whose length is not the row count is a typed rejection.

An outer join's gap side is a NULL row. Selecting it bare is a typed rejection - except at a manifest destination (`WITH (format 'hls')` / `'dash'`), where the gap is the variant map's own vocabulary for "this kind absent" ([dialect.md](dialect.md#destinations-and-options)). Elsewhere, fill it with `COALESCE`: its first argument is any nullable stream cell - a track-row alias, a CTE's or subquery's stream column, a rendition array subscript - and what follows is any stream of the same kind, another nullable cell included; per row the value is the first non-NULL, and a row where every argument is NULL is NULL ([recipe 125](examples.md#125-a-hybrid-master-muxed-variants-and-an-audio-group)). The generated stand-ins, by stream type: **audio** `ffmpeg.anullsrc(...)` (silence; `duration` inherits from the paired track when omitted, and no duration anywhere is a rejection), **video** `ffmpeg.color(...)` (black by default; `size`/`rate`/`duration` inherit), **captions** `ffrwd.empty_captions()` (a zero-cue subtitle track as one extra `data:`-URI input). Fills carry the paired row's tags, so a silence-filled French slot still emits `-metadata:s:N language=fra`. The pattern is [recipe 27](examples.md#27-concatenate-files-with-different-track-counts).

## Tags

Metadata is written by ONE column, named `tags`, holding a map. `STRUCT(<value> AS <key>, ...)` is the map literal: each field name is a key (free-form; quoted identifiers for unusual keys), each value any compile-time expression. A `tags` column **sets the keys it names** and leaves every other key alone; a `NULL` field clears exactly its key. No other SELECT column writes metadata - an aliased scalar anywhere else is a value column (see CTE rows above), and at a media sink it is a typed rejection naming this spelling.

`||` merges two maps, right side winning: `f.tags || STRUCT('Cut' AS title) AS tags` copies an input's own map and overrides one key of it. The scope is the row shape the column sits over:

- **Over track rows**: the keys land on that row's stream(s) - `-metadata:s:N`. `STRUCT(CASE WHEN t.tags.language IN ('en', 'english') THEN 'eng' ELSE t.tags.language END AS language) AS tags` retags a library in one expression. Keys the column does not name pass through unchanged. Recipes [37-38](examples.md#37-retitle-tracks-from-their-own-metadata).
- **Over input rows only** (no track rows in the branch): the keys are the container's - `-metadata`. The input's own tags feed the expressions, so `STRUCT(CASE WHEN f.tags.title IS NULL THEN 'Untitled' ELSE f.tags.title END AS title) AS tags` fills a missing title. [Recipe 52](examples.md#52-read-the-containers-tags-rewrite-them-with-case).
- **Both in one query**: layer with a CTE - a `tags` column in the body is per-stream, in the outer SELECT container-level, and the outer SELECT gathers the CTE's rows (`array_agg` + `GROUP BY`, see Combining rows); the outer value wins on a shared key. [Recipe 53](examples.md#53-tag-the-tracks-and-the-container-in-one-query).

At container level the map's own SOURCE is written too: `f.tags || STRUCT(...) AS tags` emits `-map_metadata` for that input, and a bare `STRUCT() AS tags` emits `-map_metadata -1`, writing no globals at all. Over track rows a map that names no key is a rejection - there is nothing to set. [Recipe 82](examples.md#82-keep-a-files-tags-and-change-one).

The reserved keys are the read-only fields of whatever the column sits over - a track row's `codec`, `index`, `width`, ... or the container's `duration` and `t`. Those are probed facts, so `STRUCT('h264' AS codec) AS tags` is a typed rejection rather than a tag called `codec`; every other name is a free-form key. `disposition` is reserved too, because it is the row's own field rather than metadata.

`disposition` is not a tag but the row's own field: its value is ffmpeg's disposition spec (`'default'`, `'forced'`, `'default+forced'`, `'0'` clears), it says what the whole flag map is, and it emits `-disposition:N` - [recipe 41](examples.md#41-flag-the-default-track). A container has no disposition, so it needs a track row. The same columns in a table/CSV query print as plain data, which previews what a retag will write.

## Combining rows

Four rules, no exceptions:

1. A query produces a relation. A bare SELECT prints it; COPY serializes it. Same relation.
2. A single destination takes exactly ONE row - any container, manifests included. Rows combine only when written: `array_agg` gathers a column's streams in row order, `GROUP BY` names what stays constant (an aggregate with no `GROUP BY` is one group, Postgres's own rule).
3. `TO (expression over row columns)` writes one file per row - rule 2, applied N times.
4. A multi-row relation into a single path is a compile error (`ROW_COUNT_MISMATCH`) naming the row count, the destination, and both ways out.

```sql
COPY (
  SELECT f.video, array_agg(a)
  FROM input('film.mkv') f, unnest(f.audio) a
  GROUP BY f.video
) TO 'out.mp4'
```

The row count is the RESOLVED count against the actual file: a `WHERE` that narrows a row table to one row needs no aggregate, and neither does an `ORDER BY ... LIMIT 1`. Queries with only input aliases in FROM are one row - arrays are values inside it, so splats, subscripts, and `SELECT *` never need gathering.

`array_agg` takes any per-row stream expression (`array_agg(volume(a, 0.5))`) and must be a whole SELECT column, or the sole argument of `VARIADIC` (`concat(VARIADIC array_agg(a))`, [examples.md#70](examples.md#70-join-however-many-tracks-a-file-has-with-concat)); row order is the aggregation order (`ORDER BY` before the aggregate reorders it; `ORDER BY` inside `array_agg` is rejected). Postgres's grouping rule is enforced: outside an aggregate, a row-varying expression must match a `GROUP BY` key. Group keys may be streams (`GROUP BY vid`, `GROUP BY f.video[1]`).

`GROUP BY` a row column partitions the rows, one output file per group - this requires a fan-out `TO (expression over the group keys)` (N groups are N rows; rule 2). Group keys are group-constants, so a `tags` column may read them. [Recipe 55](examples.md#55-one-file-per-language-all-its-tracks-inside) writes one file per language with all of that language's tracks inside, titled by its key; [recipe 57](examples.md#57-combine-tracks-selected-by-separate-ctes) gathers across CTE boundaries.

`ARRAY(<select>)` is `array_agg`'s converse: an expression-position gather of a countable subquery, without the CTE + `array_agg` + `GROUP BY` ceremony - [recipe 86](examples.md#86-gather-clips-into-one-file-without-the-cte) is recipe 75's contact sheet written as one expression, byte for byte the same command. It stands wherever an `array_agg` result already does (a whole SELECT column, `VARIADIC`'s argument); a multi-column subquery needs `SELECT AS STRUCT <cols>` to gather an array of structs instead, feeding `chapters` / `attachments` / a cue array the way `array_agg(STRUCT(...)::<record>)` does by hand ([recipe 85](examples.md#85-key-an-encode-ladder-from-written-rows) uses the struct row table above the same way `array_agg` uses any other row source). `array_agg` itself is unchanged - it stays the aggregate a `GROUP BY` partitions; `ARRAY(...)` is the ungrouped, no-partition case, and the two overlapping there is expected, not a duplication to resolve.

## Rows between modules

Rows exist at two times, and each has its own way of being read by a function.

Rows that exist before the run - cues, chapters, a series - are compile-time rows, and a function over one of their columns runs once per row: the built-in text and number functions (`upper(c.text)`), and a value `LANGUAGE wasm` function, memoized on its arguments so two rows saying the same thing cost one call. The result stands wherever a row value does - a `STRUCT(...)::cue` field rebuilding a track, a computed `TO`, a `WHERE` ([recipe 112](examples.md#112-a-function-over-a-caption-files-cues)).

Rows a module writes during the run are never counted, and a function over them is a **rows function**: `CREATE FUNCTION fauxlate(cues cue[]) RETURNS cue[] AS '<module>', 'fauxlate' LANGUAGE wasm`, one annotation parameter, an annotation return, no stream. Its argument is another module's annotation column (`fauxlate(captions(f.video[1]).cues)`), or a CTE column bound to one - resolved back to the same producer, so a query that needs the column twice (once on its own, once as the argument) still runs the module once; it runs in that module's sidecar process, fed each row as it is written, and its result is a row column of the declared type - a subtitle track when projected, the rows at a `.ndjson` destination. The module says what rows it reads and what it writes; the declaration is checked against both, and the producer's record has to be the one the function reads. A rows function over a file's cues is refused, pointing at the value form ([recipe 113](examples.md#113-translate-captions-as-they-are-produced)).

## Inspecting

Any of these shapes prints as a table with a bare SELECT (no COPY), or as CSV with `COPY ... TO STDOUT WITH (format 'csv')` - [recipes 30-32](examples.md#30-look-at-a-files-tracks-as-a-table). A bare input array column (`f.audio`, not subscripted) prints as one cell, Postgres array-literal style - `{<audio 0:a:0>,<audio 0:a:1>}`, braces even for one element; a subscript (`f.audio[1]`) or a bare track row (`t`) still prints its plain `<audio 0:a:0>` placeholder. `f.chapters` prints the same way, its records parenthesized in schema order: `{(1,Intro,0.0,1.0),(2,Credits,1.0,2.0)}`.

`GROUP BY` and `array_agg` are legal here too - table mode has no destination to fan out over, so every group just prints as one row, in first-appearance order, `array_agg` an array cell of the group's tracks. It is how you preview a fan-out COPY's partitions before writing any file - [recipe 56](examples.md#56-preview-a-grouped-shape-as-a-table).
