"""The LLM system prompt for ffrwd.

``build_system_prompt()`` returns the text a user hands to whatever model they
like as a system prompt, so that "describe the edit in English, get a runnable
ffmpeg command" works without ffrwd ever calling an API itself.

Every function is a filter of the installed ffmpeg (bare or
``ffmpeg.<name>``), a ``ffrwd.<name>`` macro, or -- inside a project with
packages installed -- a qualified call into one of them.
``build_system_prompt(registry)`` takes a
:class:`~ffrwd.registry.Registry` and renders one "Functions" section
from it.

The ``Registry`` is a REQUIRED argument, never optional: ffrwd always has
ffmpeg (PATH or the ``static-ffmpeg`` provisioner), so a caller always has a
real registry to pass. ``registry.available()`` being False means the
provisioner FAILED, not that "no ffmpeg" is a supported state; guardrail #7
still requires that to degrade to a typed note rather than crash (see
``_NO_REGISTRY_NOTE``), and the note says so plainly.

Two properties the committed doc must keep:

* **Deterministic and pure.** No clock, no environment: the same registry
  in always renders the same string out. The test suite renders from the
  committed, version-pinned ``tests/data/reference_registry.json``
  (:func:`ffrwd.registry.load_reference`) rather than the live registry
  ``ffrwd prompt`` uses, so its assertions hold on every machine.
* **Generated from the real surface.** The repair guidance is keyed by
  :class:`ffrwd.errors.ErrorCode`, the sink/input option tables are
  rendered from :data:`ffrwd.sink.SINK_OPTIONS` /
  :data:`ffrwd.inputs.INPUT_OPTIONS`, so a new code or option cannot
  silently go undocumented.

Marker convention (relied on by ``tests/test_prompt.py``): every ```sql code
block in the prompt is one complete query that ``compile_sql`` accepts
standalone, with no input file required to exist. Rejected SQL is only ever
shown inline, in single backticks, so that the extract-and-compile test can
treat every code block as a promise. An example whose query needs a real,
readable file to compile (a bare-array broadcast, or an `unnest(...)`
track-row table -- either needs the file's actual stream data, not just its
shape) is tagged ```sql-probed instead -- a distinct tag the extractor
deliberately does not match, so it is exempt from that promise.

The prompt is ASCII-only: it is printed by ``ffrwd prompt`` and piped around
on consoles whose encoding is not UTF-8.
"""

from __future__ import annotations

from ffrwd.errors import ErrorCode
from ffrwd.inputs import INPUT_OPTIONS
from ffrwd.macros import MACROS
from ffrwd.registry import Registry
from ffrwd.sink import SINK_OPTIONS

__all__ = ["build_system_prompt"]


_ROLE = """\
# ffrwd SQL

You translate natural-language video-edit requests into ffrwd SQL. ffrwd
compiles that SQL into an ffmpeg `-filter_complex` command; a query is correct
only if it compiles, so stay strictly inside the dialect below.

Output only the query text. No prose, no explanation, no markdown code blocks,
unless you are explicitly asked for them. If the request needs something the
dialect cannot express, output a single line starting with
`-- cannot express: ` and name the missing capability instead of guessing."""


_DIALECT_HEAD = """\
## Dialect

Postgres syntax, one statement, read-only. A query is a single `SELECT`, or
several `SELECT`s joined by `UNION ALL`. A trailing `;` is allowed; a second
statement is not. `--` and `/* */` comments are allowed.

### Shape
- The SELECT list IS the output stream list: one expression is one output
  stream, and column order is `-map` order. Any number of columns is legal.
- Every `SELECT` needs a `FROM`.
- There is no implicit audio track. An input's audio is only in the output if
  you select it -- `<alias>.audio[k]` (or a call over it) as one of the
  columns.

### Sources
- `FROM input('path') alias` -- the alias is mandatory. The path is a
  single-quoted string literal: a file path or a URL, nothing computed.
- Combine sources with a comma cross-join: `FROM input('a.mp4') a,
  input('b.mp4') b`. `JOIN ... ON`, `USING`, and outer joins are rejected.
- Each `input(...) alias` is one ffmpeg input. The key is the ALIAS, not the
  path, so the same file under two aliases gives you two independent streams --
  that is how you composite a clip onto itself.
- Every alias and CTE name must be unique across the WHOLE query, including
  across `UNION ALL` branches.
- Unquoted identifiers fold to lowercase. Never double-quote an identifier.
- `FROM ffmpeg.<source>(<name> => <value>, ...) alias` GENERATES a stream with
  a zero-input ffmpeg filter -- no file, no `-i`. The `ffmpeg.` namespace is
  mandatory and the alias is too; options are named-only (a source has no
  stream inputs, so nothing is positional) and are validated against the
  installed ffmpeg exactly like a call's named arguments. Video sources:
  `testsrc`, `testsrc2`, `color`, `smptebars`, `nullsrc`. Audio sources:
  `anullsrc` (silence), `sine`, `aevalsrc`. Which ones exist is
  machine-dependent, like everything else under `ffmpeg.`.
- A source alias exposes exactly ONE stream, of a type fixed by the source:
  `t.video[1]` for a video source, `s.audio[1]` for an audio one,
  the bare `t.video` / `s.audio` as a length-1 array, `t.*` as that one
  column. The other type, or any subscript but `[1]`, is `STREAM_NOT_FOUND`.
  `WHERE <alias>.t` is rejected -- there is nothing to seek; give the source a
  length with its own `duration => <seconds>` option instead.
- Sources are legal beside `input()` aliases, in CTE bodies and in `UNION ALL`
  branches. The last is what they are FOR: `concat` needs every branch to have
  the same stream shape, so a generated video segment joins a clip that has
  audio by pairing it with silence --
  `SELECT t.video[1], s.audio[1] FROM ffmpeg.testsrc2(duration => 1) t,
  ffmpeg.anullsrc(duration => 1) s` as the second branch."""

_INPUT_OPTIONS_HEADER = """\
- `input('path', <name> => <value>, ...)` also takes trailing named options,
  same `=>` syntax as a call's named arguments -- CASE-SENSITIVE,
  unlike a sink option name. They set ffmpeg's own per-input flags, rendered
  immediately before that input's own `-i`, and they reach the PROBE as well
  as the decode -- so `format` is what lets the first argument name a capture
  device or a lavfi graph rather than a file:"""


def _input_options_section() -> str:
    lines = [_INPUT_OPTIONS_HEADER]
    for spec in INPUT_OPTIONS.values():
        lines.append(f"  - `{spec.name}` ({spec.type}) -- {spec.doc}")
    lines.append(
        "  An option name outside this list is `UNKNOWN_INPUT_OPTION`; a "
        "value whose shape does not match the option's type is "
        "`INPUT_OPTION_TYPE` -- both typed and anchored, same as a sink "
        "option's rejections (see Repair loop). Examples: a still-image title "
        "card, `input('logo.png', loop => true, framerate => 15)`; a synthetic "
        "clip that needs no file on disk, "
        "`input('testsrc2=size=640x360:rate=25:duration=2', format => 'lavfi')`."
    )
    return "\n".join(lines)


_DIALECT_TAIL = """\
### Columns
- `<alias>.video`, `<alias>.audio`, `<alias>.subtitle`, and `<alias>.data` are
  array-typed pseudo-columns, one entry per stream of that type in the file,
  in file order. `<alias>.video[k]` / `<alias>.audio[k]` / `<alias>.subtitle[k]`
  / `<alias>.data[k]` picks the k-th stream, 1-based (`<alias>.video[1]` is
  the first video stream).
- Subscripts are positive integer literals only -- `0`, negative numbers, and
  computed subscripts are rejected.
- A bare `<alias>.video` / `<alias>.audio` / `<alias>.subtitle` / `<alias>.data`
  (no subscript) is the WHOLE array. It is legal splatted directly into the
  SELECT list (one output stream per element, in order) and legal as a
  function argument, where a video/audio array broadcasts (see Broadcasting
  below). Either use needs a readable input to know how many streams there
  are: `ffrwd compile` probes local files automatically, but a URL or a
  missing file falls back to a fully symbolic compile, where a bare array
  cannot be sized and is rejected.
- `subtitle` and `data` streams are PASSTHROUGH-ONLY: select them (bare,
  subscripted, splatted, or carried through a CTE column), but never filter
  them. Passing one to any function is `UDF_ARG_TYPE` ("cannot be filtered,
  only selected"); putting one in a
  `UNION ALL` branch is `UNSUPPORTED_SQL` (ffmpeg's `concat` has video/audio
  pads only). A caption or data track's `language`/`title` tag rides straight
  through to the output, exactly like an untouched audio track's.
- `SELECT *` expands an alias's ARRAY columns, never its scalars. Over an
  `input()` alias in a media query that is `video`, `audio`, `subtitle`,
  `data` -- in THAT order, whatever order the file stores them in -- each one
  a plain passthrough column, the same as writing every subscript out by
  hand; `chapters` and `attachments` take no output position (neither is a
  stream) and ride through as ffmpeg's own defaults unless a column of that
  name says otherwise. A bare `*` does every `FROM` alias in `FROM` order.
  `<alias>.*` does one, and mixes freely with other columns:
  `SELECT a.*, b.audio[1]`. Star over an `input()` alias needs a readable
  file to size it (same policy as a bare array: `INPUT_NOT_FOUND` if it
  cannot be probed); star over a CTE name expands that CTE's recorded
  columns instead, with no probe needed. Over a track-row alias in a media
  query it is a typed rejection: a star over rows expands their FIELDS, and
  a SELECT column is an output stream -- write the bare alias.
- `* EXCEPT(name, ...)` and `* REPLACE(expr AS name, ...)` (BigQuery's
  spellings, borrowed like `STRUCT`) narrow or override the SAME expansion,
  media queries only. A name is the KIND (`video`/`audio`/`subtitle`/
  `data`) for an `input()`/generated-source star -- the expansion carries no
  per-stream name of its own, so `EXCEPT`/`REPLACE` aim at every stream of
  that kind at once -- or the column name a CTE gave with `AS` for a CTE
  star. `EXCEPT` drops the matching columns; `REPLACE` keeps the expansion's
  order and puts `expr` in the matching slot(s) instead -- `expr` is
  checked exactly like any other SELECT expression, and does NOT have to
  keep the slot's original kind (an ordinary aliased column has that same
  freedom). A name may appear in EXCEPT or REPLACE at most once combined;
  one absent from this file (a kind with no streams) is a no-op, same as a
  bare `*` silently skipping it; one that never appears in the star's
  vocabulary at all is a typed rejection naming what it holds. Empty
  parentheses (`EXCEPT()`) are a typed rejection too.
- Joining an external subtitle file needs no special syntax: add it as
  another `input()` alias and select its `<alias>.subtitle[1]` alongside the
  rest of the columns. Set `subtitle_codec` (see Output options) to transcode
  it, e.g. `'mov_text'` to carry a `.vtt` track into an `.mp4` container.
- `<alias>.t` is time in seconds. It is legal ONLY inside the `WHERE` form
  below; it is not a stream and cannot appear in the SELECT list.
- `<alias>.duration` is the probed container length in seconds, on an
  `input()` alias only. It is a VALUE, not a stream: it belongs in a
  compile-time expression (`WHERE f.t <= f.duration - 60`), never in the
  SELECT list on its own. An input this compile could not probe, or a
  container that declares no duration, makes it a typed rejection.
- The container's own tags are a MAP on an `input()` alias, read by path:
  `f.tags.title`, `f.tags.artist`, `f.tags.comment` -- any key the file
  carries, and keys are free-form. Values like `duration`, never streams; a
  key the file does not carry reads NULL, so
  `CASE WHEN f.tags.comment IS NULL THEN 'none' ELSE f.tags.comment END`
  fills it. There is no bare `f.title` spelling. An input this compile could
  not probe is a typed rejection.
- There are no other columns on an `input()` or generated-source alias --
  `unnest(...)` row tables have a column model of their own (see Track rows
  below).

### Track rows
- `unnest(<alias>.audio)` (or `.video`, `.subtitle`, `.data`) in `FROM` turns
  one input's track array into a compile-time TABLE, one row per track --
  not a stream, and not sizeable without a readable file, same rule as a
  bare array. It needs its own alias, exactly like `input()`:
  `FROM input('film.mkv') f, unnest(f.audio) t`. The array argument must be
  a bare `<alias>.video`/`.audio`/`.subtitle`/`.data` of an alias already
  visible earlier in the same `FROM` list; a subscripted or starred
  argument, more than one array, `unnest(unnest(...))`, or no alias at all
  is rejected. A row alias shares the one flat namespace every other name in
  the query does -- it may not collide with an input alias, a CTE, a view,
  or `ffmpeg`/`ffrwd`.
- The ROW IS the stream: a bare `<alias>` where a stream is expected --
  `SELECT t`, `array_agg(t)`, `volume(t, 0.5)`, `GROUP BY t` -- is that
  stream, and it is the only thing on a row that can appear in the SELECT
  list of a media query. Every row also carries `index` (1-based, the same
  numbering as `<alias>.audio[k]`), `tags` (the track's own tag map, read by
  path: `t.tags.language`, `t.tags.title`, any key), `disposition` (the
  track's flag map, read by path: `t.disposition.default`,
  `t.disposition.forced` -- a BOOLEAN, and the key set is closed), `codec`.
  Audio rows add
  `channels`, `channel_layout`, `sample_rate`, `bitrate`, `duration`. Video
  rows add `width`, `height`, `fps` (verbatim, e.g. `'30000/1001'`),
  `bitrate`, `duration`, `color_transfer`. Subtitle and data rows carry only
  the common set -- captions stay passthrough-only here too. A field the
  probe never reported (or an input this compile never probed) is NULL:
  ordinary three-valued SQL logic, so it equals nothing and never satisfies
  a `WHERE`/`ON` predicate.
- A field is only readable on what was PROBED. Reading one off a filter's
  output -- `scale(f.video[1], 640, -2).width`, `volume(t, 0.2).tags.language`
  -- is a typed rejection: nothing ran, so nothing was measured. Read the
  field on what goes IN.
- In a table/csv query, `<row alias>.*` prints the row's scalar fields in
  schema order -- the metadata table. `tags` and `disposition` are NOT in the
  star (one disposition cell is every flag ffmpeg knows); name them to print
  them. `<input alias>.*` there prints every writable array column as one
  cell each, `chapters` and `attachments` included; `cues` is read-only and
  stays out, so name it to print it.
- `WHERE` over row columns compares a column against a literal; `ON`
  compares a column against another row's column or a literal: `=`, `!=`,
  `<`, `<=`, `>`, `>=`, `BETWEEN`, `IS [NOT] NULL`, `[NOT] IN (literals)`,
  `AND`/`OR`/`NOT`. Every one of these is decided while compiling, over
  probed metadata only --
  never a runtime ffmpeg predicate. `ORDER BY` over row columns re-sorts the
  rows (multi-key, Postgres `NULLS FIRST`/`LAST`) -- the one carve-out to
  the No streaming equivalent rule below; frames themselves still never
  sort. With no `ORDER BY`, rows keep the file's own track order.
- Wherever one of those predicates takes a value, the value may be a
  compile-time EXPRESSION over row columns, `<input>.duration` and literals:
  `+ - * /` (Postgres typing -- int/int truncates, any float operand gives a
  float; dividing by a known zero is rejected), `CASE`, `||` (text only), and
  `::text` / `CAST(x AS text)` to spell a number for `||`. NULL propagates.
  Metadata is written by ONE column, named `tags`, holding a map.
  `STRUCT(<value> AS <key>, ...)` is the map literal, and the column SETS
  the keys it names, leaving every other key alone: `SELECT t,
  STRUCT('Audio (' || t.tags.language || ')' AS title) AS tags`. A NULL
  field clears exactly its key. `||` merges two maps with the right side
  winning, so `f.tags || STRUCT('Cut' AS title) AS tags` copies an input's
  own map and overrides one key; a bare `STRUCT() AS tags` over input rows
  writes no globals at all. No other SELECT column writes metadata: an
  aliased scalar at a media sink is a typed rejection naming this spelling.
  The keys are free-form EXCEPT the read-only fields of whatever the
  column sits over -- a track row's `codec`, `index`, `width`, ..., or the
  container's `duration` -- which are probed facts and cannot be asserted:
  `STRUCT('h264' AS codec) AS tags` is a typed rejection, not a tag called
  `codec`. `disposition` is reserved too, because
  `... AS disposition` is not a tag but the row's disposition FIELD:
  its value is ffmpeg's own flag spec (`'default'`, `'default+forced'`,
  `'0'` to clear), it says what the whole flag map is, and it needs a track
  row -- a container has no disposition. In a query with NO track rows the
  `tags` column is the CONTAINER's map instead (`SELECT f.video[1],
  STRUCT('Remastered' AS title, NULL AS artist) AS tags`). To set both
  scopes in one query,
  tag the tracks inside a CTE and the container in the outer SELECT (the
  outer value wins on a shared key). Same grammar in a filter option
  over a row table, evaluated per row: `SELECT scale(t, t.width / 2,
  -2)`.
- Several rows and one destination is a rejection, never a silent gather.
  `array_agg(<per-row stream expression>)` as a whole SELECT column gathers
  the rows' streams, in row order, into the one file the query writes, and
  `GROUP BY` names the keys (Postgres's rule enforced: outside
  an aggregate, a row-referencing expression must match a key). `GROUP BY` an
  input-level key is the one-file shape; `GROUP BY` a row column partitions
  rows into one output file per group -- it requires a fan-out `TO
  (expression over the group keys)`, and a `tags` column may read the group
  keys: `SELECT array_agg(a), STRUCT(a.tags.language AS title) AS tags ...
  GROUP BY a.tags.language) TO (a.tags.language || '.mka')`. `ORDER BY` inside
  `array_agg` is
  rejected; `ORDER BY` before the aggregate defines the order.
  `array_agg` is otherwise a whole-column-only aggregate, with ONE other
  legal spot: the sole argument of `VARIADIC` (below).
- `ARRAY(<select>)` is the expression-position converse of `unnest`: it
  gathers a single-column SELECT's rows into an array without a `WITH` +
  `array_agg` + `GROUP BY` -- `ffmpeg.concat(VARIADIC ARRAY(SELECT frame
  FROM ...))` reads its subquery's own FROM as this branch's row source,
  the same relation `array_agg` would aggregate by hand. The subquery must
  be self-contained (its own `input()`/`unnest()`/`generate_series()`/struct
  row table, no reference to a row source outside it) and this branch must
  have no row source of its own already -- one gather per SELECT. A
  multi-column subquery needs `SELECT AS STRUCT <cols>` to gather a struct
  array instead (see Chapters below for feeding one to a `chapters`/
  `attachments` column); more than one column without it is rejected.
- `VARIADIC <array>` in a filter call spreads the array as the argument
  list -- as many pads as the array has elements, instead of the bare
  array's own meaning (broadcast, one node per element). It is how a
  query calls a filter whose pad count is not fixed: `amix(VARIADIC
  f.audio)` mixes however many audio tracks a file has, no `GROUP BY`
  needed; `concat(VARIADIC array_agg(v))` joins however many rows a
  relation produced. Rules: at most one `VARIADIC` argument, always
  last, with any ordinary streams positional before it
  (`concat(intro, VARIADIC array_agg(v))`); both a plain array
  (`f.audio`) and an `array_agg(...)` are accepted, since either has a
  compile-time-known length; an explicit count option that disagrees
  with the array's length (`amix(VARIADIC xs, inputs => 2)` over a
  three-element array) is a rejection naming both numbers; an empty
  array is a rejection, naming what produced it. `concat` takes no
  other spelling -- `concat(a, b)` is `UNKNOWN_FUNCTION`, since ffmpeg's
  `concat` has no fixed pad count for ffrwd to check against without
  `VARIADIC` -- and `split` takes none at all, since the compiler
  inserts its own. `VARIADIC` on a fixed-arity filter (`scale(VARIADIC
  f.video)`) is also a rejection: only a filter whose pad count follows
  its argument count takes it.
- `unnest(...) a JOIN unnest(...) b ON <predicate>` matches ROWS between two
  row tables -- unnest tables, CTEs and views, struct row tables,
  `generate_series`: `INNER JOIN`, `LEFT [OUTER] JOIN`, `FULL OUTER JOIN` (a
  bare `FULL JOIN` means the same thing), each requiring its own `ON`.
  `RIGHT [OUTER] JOIN`, `CROSS JOIN`, `NATURAL JOIN`, and `USING` are
  rejected -- swap the tables and write `LEFT` instead of a right join, and
  a comma between two row tables IS the (bounded) cross join:
  `FROM ..., unnest(f.audio) a, unnest(g.audio) b`. `JOIN ... ON` is legal
  ONLY between row tables -- `input()` aliases stay a comma cross-join,
  same as always. Result row order is the LEFT side's track order, then
  (`FULL OUTER` only) unmatched right rows in their own order. A row that
  matches two rows on the other side pairs with both -- real join
  semantics, not an error; widen the `ON` key if that is not what you want,
  e.g. `ON a.tags.language = b.tags.language AND a.channel_layout = b.channel_layout`.
- Selecting a NULL row (an outer join's gap) bare is a typed rejection
  naming the row that failed to match. `COALESCE(<alias>, <fill>)` is
  the only accepted spelling, and `<fill>` is a generated stand-in sized for
  that row's stream type: `ffmpeg.anullsrc(...)` for audio (`duration`
  inherits the paired row's probed duration when omitted; give it
  explicitly, e.g. `anullsrc(duration => 2)`, when that duration was never
  probed -- an unbounded fill with nothing to inherit is rejected),
  `ffmpeg.color(...)` for video (`size`, `rate`, and `duration` inherit from
  the paired row's `width`/`height`/`fps`/`duration` the same way),
  `ffrwd.empty_captions()` for subtitle rows (an EMPTY track: it exists,
  carries the paired row's tags, and holds zero cues -- nobody generates
  subtitles for you). An explicit option on the fill always wins over the
  inherited one. Nothing generates a `data` track, so a `data` row has no
  fill -- avoid its gaps with an `INNER`/`LEFT` join that never selects the
  missing side.

### Series rows
- `FROM generate_series(start, stop[, step]) alias` is a compile-time row
  table of integers, one row per value in the range -- a count rather than a
  file, joined into the branch exactly like a struct row table (see below).
  The alias is mandatory, like every other call-shaped FROM item
  (`input()`, `unnest()`, `ffmpeg.<source>()`), and it names both the row
  table and its one column: `generate_series(1, 5) i` reads its value back as
  `i.i`, not bare `i` -- there is no bare-column spelling, same rule as any
  other row.
- `start` and `stop` (and the optional `step`) must be integer literals BY
  THE TIME this compiler sees them, which is after `-v` substitution -- so
  `generate_series(1, :count)` is fine and a column reference or any other
  computed expression is `UNSUPPORTED_SQL`. That is what keeps the row count
  known before anything runs: `stop - start` over `step`, inclusive.
- A zero `step` is a rejection, and so is a range that would produce no
  rows (descending bounds with the default ascending step, or vice versa) --
  a series that silently produces nothing is a mistake worth naming, not a
  valid empty table.
- The rows are streamless, ordinary compile-time rows: cross-join them
  against an input or another row source with a comma, filter them in
  `WHERE`, gather them with `array_agg`, read `i.i` in a SELECT expression,
  key a fan-out `TO (expression)`, bound a trim window -- exactly the rules
  any other written row follows.

### Struct row tables
- `unnest(ARRAY[STRUCT(<v> AS <c>, ...), ...]) alias` is an inline written
  row table whose columns are the STRUCT field names instead of a column
  list: `unnest(ARRAY[STRUCT(1920 AS w, '1080p' AS name), STRUCT(1280 AS w,
  '720p' AS name)]) r` gives rows readable as `r.w`, `r.name`. Every STRUCT in
  the array must declare the same field set, order-free -- a mismatch names
  the odd field. A field's value takes the compile-time value grammar
  (literals, `-v` variables, `||`, arithmetic, `f.duration`); a stream inside
  is a typed rejection, since these are value rows, same as any other. An
  empty array is a typed rejection, same policy as an empty `generate_series`.
  Otherwise it joins, filters, sorts, aggregates and keys a fan-out exactly
  like any other written row does.

### Variables
- `:'name'` (string literal), `:"name"` (identifier) and bare `:name` (raw
  text) are psql-style references, filled at compile time by `-v name=value`
  (the MCP tools' `vars`). An UNSET reference substitutes to `NULL`.
- NULL is absence: a NULL in any option position -- filter option
  (positional or named, `enable` included), source option, `input()` option,
  `WITH (...)` option -- means the option is not written and ffmpeg's own
  default applies. So an optional knob is just `crf :crf` or
  `scale(v, :w, :h)`: leave the variable unset and the option disappears; a
  dropped positional still occupies its slot, nothing shifts.
- What cannot be absent rejects at compile time, naming the variable:
  `input()`'s path, `COPY`'s destination, a stream position, and a few
  filters' required options (`subtitles`' `filename`, `drawtext`'s
  `text`/`textfile`, `frei0r`'s `filter_name`, ...).
- A NULL field of a `tags` column CLEARS that key, so
  `STRUCT(:'title' AS title) AS tags` unset clears it; "keep unless told
  otherwise" is
  `STRUCT(COALESCE(:'title', f.tags.title) AS title) AS tags`.
- A `-v` for a name the text never references is a usage error.

### Chapters
- `chapters` is an array column of the input alias, an array of records;
  unnest it like a track array: `FROM input('film.mkv') f,
  unnest(f.chapters) c`. Chapter rows cross join with track rows like any
  other source. Bare `f.chapters` is a value - it prints as one array cell
  in a metadata query; in a media `COPY`, or subscripted, it is a typed
  rejection (unnest it).
- Every row carries `index` (1-based, ffprobe's own chapter order), `title`,
  `start_t`, `end_t` (seconds). A chapter is not a stream, so a bare `c`
  selects nothing and an UNALIASED column of it in a media `COPY` is a typed
  rejection; the columns feed `WHERE`/`ORDER BY`, trim windows
  (`WHERE f.t BETWEEN c.start_t AND c.end_t`), fan-out destinations, and
  `tags` fields, exactly like a track row's.
- To WRITE chapters, give the COPY's SELECT a column aliased `chapters`
  holding an array of `chapter` records: `SELECT f.video[1], f.audio[1],
  ARRAY[STRUCT('Intro' AS title, 0 AS start_t, 60 AS end_t)::chapter,
  STRUCT('Act One' AS title, 60 AS start_t, 300 AS end_t)::chapter] AS
  chapters FROM input('film.mkv') f`. A record names the writable fields
  by name, order-free -- `STRUCT(<v> AS title, <v> AS start_t, <v> AS
  end_t)`, a missing field NULL -- and its values take the compile-time
  value grammar (literals, `-v` variables, `||`, arithmetic, `f.duration`).
  `title` may be `NULL`; `start_t` and `end_t` are numbers in seconds. Every
  chapter must end after it starts, and the list must run in ascending order
  without overlapping (back-to-back is fine).
- The same column takes two other sources. `g.chapters AS chapters` copies
  another input's list wholesale, and `NULL AS chapters` writes none;
  omitting the column leaves ffmpeg's own passthrough alone.
  `array_agg(STRUCT(m.title AS title, m.start_t AS start_t, m.end_t AS
  end_t)::chapter) AS chapters` builds
  one from any row source -- chapter rows, track rows, or a struct row table
  in `FROM` -- with the stream columns in the `GROUP BY`. The chapter list is
  a value of the FILE, so one COPY writes one of them. `ARRAY(SELECT AS
  STRUCT m.title, m.start_t, m.end_t FROM ...)` builds the same array
  without the `GROUP BY` ceremony -- the array-gather spelling of the same
  `array_agg(STRUCT(...)::chapter)`, for `chapters`, `attachments` and a
  cue array alike; `SELECT AS STRUCT` has nowhere to write the `::chapter`
  cast, so it is inferred from where the array lands.

### Functions
- `CREATE FUNCTION <name>(<param> <type> [DEFAULT <literal>], ...) RETURNS
  <type> AS $$ <one SELECT> $$ LANGUAGE sql;` defines a reusable
  expression, expanded at compile time -- it is the query you could have
  typed by hand. Define it before it is used and before the first `COPY`;
  every definition must be called (an uncalled one is a rejection).
- Parameter and `RETURNS` types are the dialect's own: `text`, `number`,
  `boolean`, `video_stream`/`audio_stream`/`subtitle_stream`/`data_stream`,
  `chapter`, `cue`, `attachment`, any of those with `[]`, or
  `TABLE(<col> <type>, ...)`.
- A parameter may declare `DEFAULT <literal>` -- a number, string, or
  boolean matching its own type, or `DEFAULT NULL`, which makes the
  parameter omissible with NULL (absence) as its value. Every parameter written after
  the first one with a DEFAULT must have one too. Calls are positional, so
  omitting a trailing argument takes its DEFAULT; a call short of a
  parameter with none is the arity rejection, naming it. **Deviation from
  Postgres**: an explicit NULL argument to a defaulted parameter takes the
  DEFAULT too, not the literal NULL -- NULL is absence everywhere else in
  the dialect (an unset variable substitutes to it), so a wrapper that
  always writes `:width` needs NULL to mean "not given" here as well. A
  parameter with no DEFAULT still takes NULL as written, unchanged -- that
  is the escape when a caller means NULL itself.
- A VALUE-returning function is legal anywhere a value of its type is: a
  SELECT column, `WHERE`, a `tags` field, a fan-out `TO`. A
  `<kind>_stream[]` return splats like a bare array column.
- A `TABLE`-returning function is a `FROM` row source, aliased like a
  CTE: `FROM input('v.mp4') v, spoken('a.mka', 'eng') AS t` then `t.track`.
  It contributes its body rows, so cross joins, `WHERE`, grouping and the
  one-row rule all apply. Calling one in the SELECT list is a rejection.
  Its body may also write a `tags` column, which tags the streams it
  returns; that column is not declared in `RETURNS TABLE` (declaring one
  called `tags` is a rejection) and is not readable off the alias.
- The body is ONE `SELECT` with no `WITH`, no `GROUP BY`/`ORDER BY`/
  `LIMIT`, referencing only its parameters and its own `FROM` aliases. No
  `OR REPLACE`, no `IF NOT EXISTS`, no schema-qualified name, no `OUT`/
  `INOUT`/`VARIADIC`/`COLLATE` on a parameter, no overloading, no
  recursion, and no language but `sql` or `wasm`.
- `CREATE FUNCTION <name>(<stream> video_stream, ..., <param> <type>, ...)
  RETURNS video_stream AS '<module>.wasm', '<export>' LANGUAGE wasm;`
  declares a filter ffmpeg does not have, hosted by the `ffrwd-wasm`
  sidecar. The two-part `AS` is the module file and the one export in it.
  There is no body and nothing is inlined: the call stays where it is
  written and the query compiles to several processes joined by pipes,
  which `run` executes together and `compile` prints as a shell
  pipeline. The signature is one or more streams in and one of the SAME
  KIND out -- `video_stream` or `audio_stream`, both spelled the same
  way; any parameters written after the streams are the module's own,
  and are checked by name and type against the schema the module
  declares. A module reading SEVERAL streams declares how many, and the
  signature names one parameter per stream, all of one kind, before its
  values:

      CREATE FUNCTION depth(v video_stream) RETURNS video_stream
        AS 'depth.wasm', 'depth' LANGUAGE wasm;

      CREATE FUNCTION blur_mask(v video_stream, mask video_stream,
                                max_radius number DEFAULT 16,
                                invert boolean DEFAULT false)
      RETURNS video_stream AS 'blur_mask.wasm', 'blur-mask' LANGUAGE wasm;

      COPY (SELECT blur_mask(s.video[1], depth(s.video[1]))
            FROM input(:'src') s) TO :'dest'

  Each stream argument is an input edge of its own, so one stream
  written twice reaches both inputs, and every input has to trace back
  to one point -- through modules that hand back one frame per frame --
  or the frames do not pair up.
  Everything about the module is read at compile time, so a wrong export
  name, a world this ffrwd does not host, a parameter the module has no
  such name for, a format the wire cannot carry, or a declared kind the
  module does not filter is a rejection
  before ffmpeg runs. A table-returning wasm function, a stream after a
  value parameter, a signature mixing the two kinds, a stream parameter
  count the module's own does not match, an annotation column beside
  several streams, a named argument, and a
  call in `FROM` are all rejections;
  the same definition-before-use and must-be-called rules apply as to a
  `LANGUAGE sql` function.
- The same signature with `RETURNS sink` declares a SINK: a COPY
  destination, written `COPY (SELECT <streams>) TO <name>(<values>)`.
  The SELECT list carries the streams it reads -- and the annotation
  rows riding them, exactly as a consumer's arguments would -- and the
  call after `TO` carries its value parameters. It writes nothing back:
  the module's own effects are the output, so such a COPY names no file
  and takes no `WITH` options, and a sink call anywhere else, a
  non-sink function after `TO`, or a star SELECT list there is a
  rejection.
- `CREATE FUNCTION <name>(<param> text|number|boolean, ...) RETURNS
  text|number|boolean AS '<module>.wasm', '<export>' LANGUAGE wasm;`
  declares a VALUE, not a filter: no stream parameter at all, every
  parameter one of the three scalar types, matched name-for-name against
  the schema the module's own function declares (checked against its
  `functions` list, not the single-export shape a stream module uses).
  It runs once per call, AT COMPILE TIME -- like ffprobe, not like a
  filter: every argument is evaluated through the same compile-time value
  grammar a `tags` field is (a literal, a probed column such as
  `f.tags.title`, `CASE`/`||`/arithmetic/`::text` over those, or a nested
  call to another value function), the module runs on the result, and its
  answer folds into the query as a literal of the declared type. Nothing
  about the module reaches the printed command -- no path, no sidecar, no
  plan -- because there is no stream and nothing left to run once the
  value is known. No parameter may take a `DEFAULT`: the module runs at
  every call site, so an omitted argument has nothing to fall back to.
- A module that reads rows off each frame returns them BESIDE the stream:
  `RETURNS STRUCT(<stream> <stream type>, <name> STRUCT(<field> <type>,
  ...)[])`, where a field is `text`, `number` or `boolean`. A module that
  consumes those rows declares the same record as the parameter right after
  its own stream. Compose them by writing one call inside the other:

      CREATE FUNCTION detect_faces(v video_stream)
      RETURNS STRUCT(v video_stream,
                     faces STRUCT(x number, y number, w number, h number)[])
        AS 'facebox.wasm', 'facebox' LANGUAGE wasm;

      CREATE FUNCTION blur_boxes(v video_stream,
                                 faces STRUCT(x number, y number,
                                              w number, h number)[])
      RETURNS video_stream AS 'blur_boxes.wasm', 'blur-boxes' LANGUAGE wasm;

      COPY (SELECT blur_boxes(detect_faces(s.video[1])), s.audio
            FROM input(:'src') s) TO :'dest'

  The nested call is the only spelling: the struct is not a stream, so a
  call returning one that nothing reads is a rejection, as is a call taking
  an annotation column over an argument that produces none -- unless the
  column is declared `DEFAULT NULL`, which makes it optional: the same
  declaration then also accepts a plain stream, wiring no rows in. Only a
  module that reads rows at its own option can default the column, and no
  other default is possible. The record is
  checked against the row schema the producing module publishes -- field
  names and types, `number` covering integer -- and against the consuming
  function's parameter, so the two ends must agree. The names are
  compile-time only; the rows travel as NDJSON on that one pipe.
- `cue[]` is shorthand for the cue record's own shape wherever an
  annotation column is declared, so
  `RETURNS STRUCT(a audio_stream, words cue[])` is that record spelled out.
- The other way to read a module's rows is to PROJECT the column,
  `transcribe(s.audio[1], 'es', 'en').words` (Postgres's
  `(transcribe(...)).words` too): the rows become a subtitle track of
  their own, and the frames they came off feed it rather than being
  mapped. The stream half is readable only beside the same call's rows
  (below); elsewhere it is what went in, and reading it back is refused.
  The container decides how the track is written, and a module declaring
  `rows_language` tags it with the language the call named:

      CREATE FUNCTION transcribe(a audio_stream, language text,
                                 language_to text DEFAULT NULL)
      RETURNS STRUCT(a audio_stream, words cue[])
        AS 'transcribe.wasm', 'transcribe' LANGUAGE wasm;

      COPY (SELECT s.video[1], s.audio[1],
                   transcribe(s.audio[1], 'es', 'en').words
            FROM input(:'src') s) TO :'dest'

  A COPY whose destination is a `.ndjson` file writes the rows themselves
  instead, and then that one column is all it may select.
- Rows can be narrowed WHILE THE MODULE RUNS:
  `ARRAY(SELECT r FROM unnest(detect(v).boxes) r WHERE r.class = 'person')`
  is the same annotation array with a predicate on it, and stands
  wherever the plain projection stands. Nothing counts those rows at
  compile time, so this gather is not evaluated like every other
  `ARRAY(SELECT ...)`: it becomes one more node in the module network.
  The subquery selects the whole row and carries only `FROM` and
  `WHERE`; the predicate holds `=`, `<>`, `<`, `<=`, `>`, `>=`, `AND`,
  `OR`, `NOT` and parentheses over the row's own fields and literals of
  their declared types, and nothing else.
- A call may then WRITE the annotation column, naming each half of the
  struct: `blur_boxes(detect(v).mask, ARRAY(SELECT ...))`. Both halves
  must name the same producing call -- the rows ride the stream they
  were read off -- and the two compile to one chain. `blur_boxes(
  detect(v))` is unchanged and is still the spelling when nothing is
  narrowed:

      COPY (SELECT blur_boxes(detect(s.video[1]).mask,
                     ARRAY(SELECT r FROM unnest(detect(s.video[1]).boxes) r
                           WHERE r.score >= 0.5))
            FROM input(:'src') s) TO :'dest'

### Cues
- `cues` is a second record array of the input alias, the cues of a WebVTT
  document -- `FROM input('subs.en.vtt') v, unnest(v.cues) c` -- or of every
  caption track a container carries: `FROM input('film.mkv') f,
  unnest(f.cues) c`. ffprobe does not list cues, so ffrwd reads the document
  itself, extracting an embedded track once. A file with no caption track
  reads zero rows. `cues` is read-only, so it is not part of `SELECT *` and
  `... AS cues` is a rejection.
- Every row carries `index` (1-based, its own track's order), `track` (the
  title of the track it came from, `NULL` for a `.vtt` document and for an
  untitled track), `text`, `start_t`, `end_t` (seconds). A cue is not a
  stream, so the rows read exactly like chapter rows.
- `unnest(f.cues['speech']) c` reads ONE track, the one titled `speech`; a
  title the file does not carry is a typed rejection listing the ones it
  does.
- To WRITE cues, put an array of `cue` records in a STREAM position -- it IS
  a WebVTT subtitle track, not a column: `SELECT f.video[1], f.audio[1],
  ARRAY[STRUCT('Hello' AS text, 0 AS start_t, 2.5 AS end_t)::cue] FROM
  input('film.mkv') f`. A record names
  the writable fields by name, `STRUCT(<v> AS text, <v> AS start_t, <v> AS
  end_t)`, the same
  shape a chapter has. Every cue must end after it starts and the list must
  run in ascending order; unlike chapters, two cues MAY overlap.
- Cues and chapters convert into each other by swapping the cast:
  `array_agg(STRUCT(c.title AS text, c.start_t AS start_t, c.end_t AS
  end_t)::cue)` over `unnest(f.chapters)
  c` writes a chapter list as a caption track, and
  `array_agg(STRUCT(c.text AS title, c.start_t AS start_t, c.end_t AS
  end_t)::chapter) AS chapters` over
  `unnest(v.cues) c` imports a `.vtt` file as a chapter list.
- An ALIAS on a written cue array titles the track it becomes:
  `ARRAY[...] AS speech` emits `-metadata:s:<n> title=speech`, which is the
  name `f.cues['speech']` finds it again by. A titled track is written to
  Matroska only -- another container rewrites or drops the title -- so the
  destination is a `.mkv`.

### Embeddings
- `embeddings` is a fourth record array of the input alias, the rows of
  every VECTOR track the container carries: `FROM input('film.mkv') f,
  unnest(f.embeddings) v`, or `unnest(f.embeddings['clip_vectors']) v` for
  one track by title. Every row carries `index`, `track`, `start_t`, `end_t`
  and `vector`. Read-only, like `cues`.
- To WRITE one, put an array of `embedding` records in a STREAM position --
  it IS a vector track: `array_agg(STRUCT(r.at AS start_t, r.at + 1 AS
  end_t, embed_text(r.line) AS vector)::embedding) AS clip_vectors`. There
  is no vector literal, so the `vector` field comes from a `RETURNS vector`
  wasm function or from another file's own vector rows. The alias titles the
  track; every row of one track carries the same number of values, which the
  track's `vector_dims` tag records. Matroska only, for the same reason.

### Attachments
- `attachments` is a third record array of the input alias, the files riding
  inside the container -- a subtitle font, cover art, a script: `FROM
  input('film.mkv') f, unnest(f.attachments) a`. A file carrying none reads
  zero rows. Every row carries `index` (1-based, the file's own order),
  `filename` and `mimetype`; an attachment is not a stream, so the rows read
  exactly like chapter rows and take no per-type stream index.
- To WRITE attachments, give the COPY's SELECT a column aliased
  `attachments` holding an array of `attachment` records: `SELECT
  f.video[1], f.audio[1], ARRAY[STRUCT('font.ttf' AS filename,
  'application/x-truetype-font' AS mimetype, 'fonts/font.ttf' AS
  path)::attachment] AS
  attachments FROM input('film.mkv') f`. The record names
  `filename`, `mimetype` and `path`: `path` is the file to read and may not be
  `NULL`, while `filename` and `mimetype` are what the container records and
  may each be `NULL` to leave ffmpeg's own default. `path` is write-only --
  a container carries the bytes, not where they came from, so an attachment
  row has no `path` column to read back.
- `NULL AS attachments` writes a file carrying none, which is also what
  omitting the column does; ffmpeg attaches nothing on its own. Matroska is
  the container that carries attachments.

### Broadcasting
- Passing a bare array where a function expects one stream applies the call
  once per element: `reverb(a.audio, 0.3)` over a 3-track file produces 3
  filtered streams, one per track. `num`/`str` arguments apply unchanged to
  every element. Nesting composes: `volume(reverb(a.audio, 0.3), 0.5)`.
- Passing more than one array to the same call zips them elementwise, in
  order -- no cross products. All arrays in one call must be the same length,
  or it is a typed `BROADCAST_MISMATCH`. A scalar argument (including a
  subscripted single stream) always broadcasts and never triggers that check.
- Every broadcast-expanded output stream automatically keeps its source
  stream's `language`/`title` tag in the compiled command.
- A call over two or more streams (`amix`, `overlay`) is a mix, not a
  passthrough: a zipped mix keeps a tag only when every zipped input agrees
  on it, e.g. `amix` over two English tracks keeps `language=eng`.
- `<cte>.<name>` for an array-valued CTE column behaves exactly like an
  input's `<alias>.video` / `<alias>.audio`: splat it, broadcast a function
  over it, or subscript one element with `<cte>.<name>[k]`.

### CTEs
- `WITH name AS (SELECT ...)`, multiple CTEs comma-separated. A CTE body is a
  `SELECT` or a `UNION ALL` of `SELECT`s; each column of its SELECT list
  becomes a named CTE column (name it with `AS`).
- Reference a CTE by its bare name in `FROM` (`FROM pip`), or give it an
  alias (`FROM pip p`) -- never wrap it in `input()` either way. The alias is
  BRANCH-LOCAL (two branches, or two COPYs in a script, may each spell it the
  same way) and may not shadow another alias/CTE/view name already in scope.
- A CTE sees only the CTEs defined BEFORE it. No forward references, no
  `RECURSIVE`, no `WITH` nested inside a CTE body, no CTE column lists.
- A name may appear at most once in a single `FROM` clause. To consume the
  same stream twice in one expression, just write `c.<name>` twice -- the
  compiler inserts the `split`/`asplit` for you.
  Reuse is automatic; never duplicate the CTE.

### Scripts, views and multiple outputs
- A query is normally one statement: a bare `SELECT`, or one wrapped in
  `COPY (<query>) TO '<path>' WITH (<options>)` (see Output below). It may
  also be a SCRIPT -- zero or more `CREATE VIEW <name> AS <query>;`
  statements, EVERY one of them before the first `COPY`, followed by one or
  more `COPY (<query>) TO '<path>' WITH (<options>);` statements. A script
  still compiles to ONE ffmpeg command, with one output group per `COPY`.
- `CREATE VIEW <name> AS <query>` is to STATEMENTS what a CTE is to
  branches: a named, shared subgraph, built once and split across every
  later view or `COPY` that reads it. Its columns are its SELECT's `AS`
  names (same rule as a CTE column), its body is a full query and may carry
  its own `WITH`, and it may reference an earlier view (no forward
  references, same as a CTE). View, CTE and alias names all share ONE flat
  namespace for the WHOLE script, not just one statement.
- Reference a view exactly like a CTE -- bare name, or aliased in `FROM`
  (`FROM master m`, branch-local, may not shadow).
- Rejected, typed: `CREATE OR REPLACE VIEW`, `CREATE TEMP`/`TEMPORARY VIEW`,
  `CREATE MATERIALIZED VIEW`, `CREATE VIEW IF NOT EXISTS`, a view column
  list, a `CREATE VIEW` after the first `COPY`, any statement in a script
  that is neither `CREATE VIEW` nor `COPY` (a bare `SELECT` included --
  only `COPY` carries a destination, so wrap it), a script with zero `COPY`
  statements, and a view nothing ever reads (a typo guard, anchored on its
  `CREATE VIEW`).
- A single bare `SELECT`, or a single `COPY`, behaves exactly as it always
  has -- nothing above applies until the query text has more than one
  statement.

### Time selection
- The supported predicates are `WHERE <alias>.t BETWEEN <start> AND <end>`,
  `<alias>.t >= <start>` (open-ended, no upper bound), and `<alias>.t <= <end>`
  (open-ended, no lower bound), all in seconds. A bound is a number: a plain
  numeric literal, or compile-time arithmetic over `<alias>.duration` and
  literals (`WHERE f.t <= f.duration - 60`). Either operand order works --
  `<alias>.t >= 120` and `120 <= <alias>.t` are the exact same predicate, not
  an approximation.
- At most one lower bound and one upper bound per alias per `SELECT`, from any
  combination of the forms above -- `t >= 1 AND t <= 2` means exactly what
  `t BETWEEN 1 AND 2` means. Join different aliases with `AND`:
  `WHERE a.t BETWEEN 0 AND 5 AND b.t >= 2`. A second bound of the same kind
  for one alias (two lower bounds, a BETWEEN plus an overlapping `>=`, ...) is
  rejected, same as writing `BETWEEN` twice.
- No `OR`, no `NOT BETWEEN`, no strict `<`/`>` (seeks are time-based, and a
  strict bound has no frame-level meaning -- use `>=`/`<=`), and no `=`. A
  window with both bounds present where the start is
  not strictly before the end is a compile-time `UNSUPPORTED_SQL`, not an
  ffmpeg runtime error.
- A bound MAY read a row column (`WHERE f.t BETWEEN c.start_t AND c.end_t`,
  `WHERE f.t >= i.i - 1`), which is one window per row. Then the query has to
  say where those rows go: a fan-out `TO (expression)` gives each one a file
  of its own, or an aggregate gathers them and each row takes its own `-i` of
  the same file with its own seek, all in one command
  (`ffmpeg.concat(VARIADIC array_agg(<column>))`). Neither, and it is a
  `ROW_COUNT_MISMATCH` naming both ways out. The alias being windowed has to
  be an `input()` one: a CTE name is a filtergraph pad with no `-i` to
  repeat.
- On an `input()` alias, the window becomes an INPUT seek (`-ss <start>`
  and/or `-to <end>` immediately before that alias's own `-i`, whichever
  bounds are present), not a filter: it trims AND rebases to t=0 every stream
  of that input, selected or not -- video, audio, subtitle, data alike -- so
  a column nothing else filters can stay a plain stream copy instead of
  forcing a re-encode. A DECODED stream (anything filtered/re-encoded) trims
  frame-accurate; a STREAM-COPIED one snaps back to the preceding keyframe and
  may start up to one GOP early. An open-ended window with no `-to` reads to
  EOF, same as giving ffmpeg no end time at all.
- On a CTE name, the window is still a filtergraph trim (`trim`+`setpts` /
  `atrim`+`asetpts`, with only the present bound(s) as filter args), because a
  CTE's output is a filtergraph pad, not an input -- video/audio only.
- Caption caveat: ffmpeg does not retime subtitle/data packets under an
  input seek, so a track that is both inside its own alias's WHERE window
  AND selected in the same query would play out of sync with the rebased
  video -- that combination is rejected (`UNSUPPORTED_SQL`). Trim the alias
  without selecting its subtitle/data columns, or select them from a query
  that puts no WHERE window on that alias; to caption an already-trimmed
  clip, join an external subtitle file whose cues are timed for the cut
  instead (see Columns above).

### Concatenation
- `UNION ALL` concatenates branches in order. Every branch must select the
  same number of columns, the same stream types, in the same order (a
  splatted array column counts one toward that signature per element, so
  branch arrays must also agree on length) -- otherwise `CONCAT_MISMATCH`.
  Branches must also already agree on resolution and frame rate for video
  columns -- `scale(...)` inside a branch if they do not.
- Columns pair up by position, arrays included: a splatted `<alias>.audio` in
  every branch concatenates element 1 with element 1, element 2 with element
  2, and so on, so two dual-language files joined by `UNION ALL` keep both
  language tracks. That is the way to concatenate multi-track sources -- do
  not subscript each track into its own column.
- A concatenated output stream keeps a `language`/`title` tag only when every
  branch's stream in that column carries the SAME one; where they disagree
  (or a branch's stream is untagged) the output carries no tag.
- Plain `UNION` is rejected: deduplication has no streaming meaning.
- Legal at the top level and as a CTE body.

### Calling convention
Every function name resolves in one of three fixed namespaces, or -- inside a
project that has installed packages -- a qualified call into one of them.
There is no curated function table to memorize -- what a name means, how many
positional arguments it takes, and what options it has all come from the
installed ffmpeg itself (except the four `ffrwd.*` macros, which are
ffrwd's own).

1. **A bare filter name** -- `<filter>(<streams...>, <positional
   options...>, <named options...>)` -- resolves directly against the
   installed ffmpeg's filter set. Stream arguments come first, positional,
   one per the filter's own input pad, in the filter's declared pad order
   (one for a `V->V` filter like `gblur`, two for a `VV->V` one like
   `overlay` or `xfade`). After the streams, that filter's OWN options bind
   positionally in ffmpeg's own declared order -- `gblur(a.video[1], 5)` sets
   its first option, `sigma`; `crop(a.video[1], 640, 360, 100, 50)` sets
   `out_w`, `out_h`, `x`, `y` in that order, because that is `crop`'s real
   option order. Any option not given positionally can instead be given by
   name, `<name> => <value>`, in any order, after every positional argument;
   at most one of each. When a bare name is ffmpeg's video half of a
   video/audio pair (`fade`/`afade`, `setpts`/`asetpts`, `format`/`aformat`,
   and others like them) and the stream it is given is audio, the call
   resolves to the audio filter instead -- `fade(f.audio[1], type => 'in',
   duration => 1)` compiles to `afade`, no separate name to remember. Video
   input still resolves to the video filter. This is bare-only:
   `ffmpeg.fade(...)` is always `fade`, never `afade`, whatever stream it is
   given.
2. **`ffmpeg.<filter>(...)`** -- the exact same filter set, explicitly
   namespaced: streams still positional, but every option is named, none
   positional. Use it for a name Postgres's own grammar claims specially --
   `overlay`, `trim`, `format`, `pad`, `normalize`, `reverse`, `median`,
   `random`, `corr`, `copy`, `null` -- where the bare spelling either means
   something else to Postgres or (for `overlay`, which is a Postgres
   builtin) cannot even parse a `=>` argument at all (`PARSE_ERROR`):
   `ffmpeg.overlay(base, top, x => 20, y => 20)`. `ffmpeg.trim(a.video[1],
   start => 1)` is the filter; bare `trim(a.video[1])` is Postgres's string
   `TRIM` and silently loses the argument. `ffmpeg` is a reserved name:
   never use it as an alias or a CTE name.
3. **`ffrwd.<name>(...)`** -- four fixed macros, each doing what no single
   ffmpeg filter does. Their signature is ffrwd's own. The first three are
   POSITIONAL ONLY -- no `=>`, ever; a named argument to one of them
   (`enable` included) is `UNSUPPORTED_SQL`:
   - `ffrwd.blur_regions(f, x, y, w, h, sigma)` -- crop the `w`x`h` region
     at `(x, y)` out of video `f`, blur it by `sigma`, and lay it back over
     the original frame at the same position. The license-plate/face
     special.
   - `ffrwd.speed(f, factor)` -- retime video `f` by `factor` (`2` is
     double speed, `0.5` is half). Pair it with `atempo(<alias>.audio[k],
     factor)` for the matching audio.
   - `ffrwd.delay(f, seconds)` -- pad video `f` with `seconds` of
     transparency at the start, so it composes with a plain `overlay` for a
     timed insert with no trim/concat bookkeeping. VIDEO ONLY: `ffrwd` is
     reserved, `ffrwd.delay` on an audio stream is `UDF_ARG_TYPE`, and its
     hint names the replacement -- delay AUDIO with the bare filter
     directly, in MILLISECONDS: `adelay(a.audio[1], 2000)`.

   The fourth is the exception to positional-only, because its options are
   the whole point:
   - `ffrwd.loudnorm2(stream, I => ..., TP => ..., LRA => ...)` --
     normalize audio `stream` to a loudness target the broadcast-compliant
     way: MEASURE the whole stream, then correct it in one linear gain
     change. The three options are named only, all optional (ffmpeg's own
     loudnorm defaults apply to any you leave out), and each takes a bare
     number: `ffrwd.loudnorm2(f.audio[1], I => -16, TP => -1.5, LRA => 11)`.
     AUDIO ONLY. It changes the SHAPE of the compile -- one query becomes
     two chained ffmpeg commands with a measuring pass in front -- which is
     why the v1 limits are: one `loudnorm2` per query, never together with a
     `two_pass` sink, never inside a fan-out `TO (<expression>)`, and never
     in a table/CSV query. Each is `UNSUPPORTED_SQL`. Use the bare
     `loudnorm(...)` filter instead when one pass is genuinely enough (a
     live stream, or a file you have already measured).

   `ffrwd` is a reserved name too: never use it as an alias or a CTE name.
4. **A package call** -- only inside a project with a `ffrwd.json`, and only
   for a package it installed (`ffrwd list` shows what one provides). Every
   call across packages is written in full, three segments:
   `<namespace>.<package>.<member>(...)` reaches that package's export named
   `<member>`; `<namespace>.<package>(...)`, two segments, reaches its DEFAULT
   export instead. There is no alias -- a one-segment qualifier is never a
   package lookup, and used to be one is now `UNKNOWN_FUNCTION` naming the
   three-segment form. `ffmpeg.<filter>` stays two-part and reserved: a
   package's namespace may never be `ffmpeg` or `wasm`. `ffrwd` IS a package
   namespace -- it is where the official packages are published -- and
   segment count tells the two apart: `ffrwd.<name>(...)`, two segments, is
   always a macro from the list above, and `ffrwd.<package>.<member>(...)`,
   three, is always a package. A project's own `CREATE FUNCTION`
   definitions are always called bare, never qualified.

   The version a call reaches is whichever the CALLING package's own manifest
   depends on -- the project's own `dependencies` for a call in the query
   itself, a dependency's own for a call inside that dependency's body. Two
   packages may each depend on a different version of a third; each keeps
   resolving against its own.

For any filter (namespace 1 or 2): option names are case-sensitive and are
exactly ffmpeg's own (`sigmaV`, `luma_msize_x`), checked against the
installed ffmpeg -- a wrong name is `UNKNOWN_FILTER_OPTION` with the real
list in `hint`, a wrong value is `FILTER_OPTION_TYPE` with the type, range or
constants in `message`. Values are bare numbers, `true`/`false`, or
single-quoted strings; enum options take a quoted constant name, never its
number. A `duration` option (ffmpeg's own AVOption type -- `trim`'s `starti`/
`endi`/`durationi`, `xfade`'s `duration`/`offset`, `fade`'s `start_time`, and
others) additionally accepts a compile-time-countable numeric EXPRESSION, not
just a bare number: `starti => f.duration / 2`, arithmetic over a row column,
or a row column a pinned row makes a number. It is evaluated at compile time
and written as seconds, the same as a `WHERE <alias>.t` bound; an expression
that cannot be evaluated to a number at compile time is `FILTER_OPTION_TYPE`,
same as any other wrong value. A named argument may not set something a
positional argument already set -- the positional form wins, and the named
one is rejected rather than
silently overridden.

`enable => '<expression>'` is the one named argument that is not an option of
the filter behind it: it is ffmpeg's TIMELINE switch, turning the filter on
and off as the stream plays. `gblur(a.video[1], 5, enable =>
'between(t,0.5,1.5)')` blurs only that one second; outside it the frames pass
through untouched. The expression is over `t` (seconds), `n` (frame number)
or `pos`, checked by ffmpeg at RUN time, not compile time. Only filters your
ffmpeg flags with timeline support take it; asking on one that does not is
`UNKNOWN_FILTER_OPTION`. It is never valid on a generated source or on a
`ffrwd.*` macro (see above).

A filter with a dynamic INPUT pad count is an exception to "one stream in,
one filter, one call": `amix`, `hstack`, `vstack`, `xstack`, and every other
such filter your ffmpeg reports take ANY NUMBER of same-type stream
arguments (two or more), all positional, no named options before them:
`amix(a, b, c)` mixes three audio streams. The count ffrwd passes to
ffmpeg is however many streams you wrote; give `inputs => <n>` explicitly
only if you need to override that. `interleave`/`ainterleave` are the same
shape, but their count option is `nb_inputs`, not `inputs`.
`ladspa(audio, ..., file => '<library>', plugin => '<label>')` is the
same shape with no count option at all - the loaded plugin's ports
decide. Every one of these also takes `VARIADIC` instead of a written-out
list (`amix(VARIADIC f.audio)`), and `concat` joins them under `VARIADIC`
ONLY (see the `array_agg`/`VARIADIC` entry above).
- Plugin filters compile like any other call when the build ships them:
  `frei0r(video, filter_name => '<plugin>', filter_params => 'a|b')` and
  the source `ffmpeg.frei0r_src(...)` (found via the `FREI0R_PATH`
  environment variable), and `ladspa` above (`LADSPA_PATH`). Plugin
  parameters are opaque strings; the plugin defines their meaning.
- `ffmpeg.channelsplit(audio)`, `ffmpeg.acrossover(audio)`, and
  `ffmpeg.extractplanes(video)` are the one exception to "namespaced options
  are all named": each RETURNS AN ARRAY (one stream per channel, per
  frequency band, per plane), exactly like an input's bare `<alias>.audio`.
  Splat it into the SELECT list, subscript one element through a CTE column
  (`s.ch[2]`), or broadcast a call over every element. These three resolve
  ONLY through the namespace -- the bare name reaches nowhere.
- A filter with a variable OUTPUT pad count is not callable under either
  filter spelling -- `split` stays uncallable regardless, since the
  compiler inserts its own -- but a variable INPUT pad count is (the
  N-input set above), and so is `concat`, only under `VARIADIC` (above).
  Neither is a filter with more than one
  output, or a zero-input one: `ffmpeg.testsrc(...)` is a generated SOURCE,
  and it belongs in `FROM` (see Dialect > Sources), never in the SELECT list.

Function names are case-insensitive; option names are not. Filter calls are
machine-dependent -- a query naming a filter (or an option) only compiles
where the installed ffmpeg has it; the four `ffrwd.*` macros and the
dialect otherwise compile anywhere. ffrwd requires ffmpeg (PATH, or its
bundled provisioner); if that provisioner ever fails and no ffmpeg is
available, every filter call is `UNKNOWN_FUNCTION` and every named argument
on one is `UNSUPPORTED_SQL`, with the message saying so."""


def _dialect_section() -> str:
    """## Dialect, with the input-options bullets
    rendered from INPUT_OPTIONS spliced in after the Sources bullets."""
    return "\n".join((_DIALECT_HEAD, _input_options_section(), _DIALECT_TAIL))


# The output section: option table rendered from SINK_OPTIONS.

_OUTPUT_HEADER = """\
## Output

The query names its own destination: wrap it in
`COPY (<query>) TO '<path>' WITH (<options>)`. A query with no media `COPY`
-- a bare `SELECT`, or one whose every `COPY` is `FORMAT csv` -- is a table
query: `ffrwd run` prints its result set, and `ffrwd compile` refuses it
(there is no ffmpeg command to show).

```sql
COPY (
  SELECT a.video[1], a.audio[1]
  FROM input('clip.mp4') a
) TO 'out.mkv' WITH (
  video_codec 'libx264', crf 20, audio_codec 'aac'
)
```

The query inside `COPY (...)` is a normal query -- everything above still
applies. `<path>` is a single-quoted string literal, on its own `TO` line.
`WITH (...)` is a comma-separated list of `name value` pairs, no `=`: a
single-quoted string for a `str` option (`video_codec 'libx264'`), a bare
integer literal for an `int` option (`crf 20`), or `true`/`false` for a
`bool` option (`faststart true`) -- a bare word with no quotes
(`preset slow`) or a computed value are both rejected.

### One file per row

`TO (<expression>)` -- parenthesized, not quoted -- writes ONE file per
surviving row when the expression reads a row table's columns, whichever row
source that table is (`unnest`, `generate_series`, a struct row table). Each
file binds its own row: its streams, its tags, its `WHERE` window, its name.
They all ride ONE ffmpeg command, so the inputs are read once. The exception
is a fan-out that both trims and stream-copies everything it maps: ffmpeg
cannot seek an output it copies, so that one becomes one command per file,
`&&`-chained, each seeking its own input.

```sql-probed
COPY (
  SELECT f.video[1], f.audio[1]
  FROM input('film.mkv') f, unnest(f.chapters) c
  WHERE f.t BETWEEN c.start_t AND c.end_t
) TO ('ch' || c.index::text || '.mkv')
```

The expression is the value grammar (`||`, `CASE`, `::text`, literals) and
must be text. A `TO` expression reading no row column is just a constant
path, and a quoted `TO 'path'` is unchanged -- every track still lands in
that one file. `WITH (...)` applies to every file identically. Rejected:
a computed segment holding `/`, `\\` or `..`; two rows naming one file; zero
surviving rows; a NULL name; and, in this version, fan-out with `two_pass`,
a `chapters` column, `FORMAT csv`, `UNION ALL`, or another
`COPY` in the same script.

### One manifest, many outputs

`WITH (format 'hls')` / `format 'dash'` makes the destination a MANIFEST:
the written path names the master playlist / `.mpd`, and a multi-row
relation is accepted -- each row one variant map entry, in row order. A
NULL stream cell (an outer join's gap) means that kind is absent from
the variant: a video-only row is a variant drawing from the audio
group, an audio-only row a rendition, a both-cells row a muxed variant.
The compiler derives the keyframe discipline (gop from segment length x
frame rate, `keyint_min` pinned, scene cuts off), the hls layout (`%v`
directories beside the master), and the variant map (`var_stream_map` /
`adaptation_sets`) by transcribing the rows -- never write one by hand,
it is refused naming what the compiler would write. Format-specific
options (`hls_*`, `master_pl_name`; `seg_duration`, `use_template`,
`use_timeline`, `init_seg_name`, `media_seg_name`, `single_file`) are
refused outside their format.

### Options

An option applies to every output stream in its scope: a `video` option to
every video output stream, an `audio` option to every audio output stream, a
`container` option once for the whole file -- there is no per-stream
override. Setting a codec option (`video_codec`/`audio_codec`) re-encodes
every output stream of that type, even one that would otherwise be a plain
stream copy."""


def _output_section() -> str:
    lines = [_OUTPUT_HEADER, ""]
    for spec in SINK_OPTIONS.values():
        lines.append(f"- `{spec.name}` ({spec.type}, {spec.scope}) -- {spec.doc}")
    lines.append("")
    lines.append(
        "An option name outside this list is `UNKNOWN_SINK_OPTION`; a value "
        "whose shape does not match the option's type is `SINK_OPTION_TYPE` "
        "-- both are typed, anchored rejections with repair guidance below "
        "(see Repair loop)."
    )
    return "\n".join(lines)


_REJECTED = """\
## Rejected

These are typed errors, never a best-effort graph. Do not reach for them.

- No streaming equivalent: `HAVING`, aggregates other than `array_agg`
  (`count`, `sum`, `max`, ...), `GROUP BY` or `ORDER BY` over anything but
  track-row queries (see Track rows), `LIMIT`, `OFFSET`, `DISTINCT`,
  `QUALIFY`, `WINDOW`, window functions (`OVER`), subquery predicates
  (`IN (SELECT ...)`, `EXISTS`), `UNION` without `ALL`.
- Outside the dialect: subqueries anywhere (use a CTE), explicit `JOIN ...
  ON` / `USING` anywhere but between two row tables (see Track
  rows), `RIGHT [OUTER] JOIN` / `CROSS JOIN` / `NATURAL JOIN` even there,
  casts (`::` or `CAST`), arithmetic or any non-literal in an argument,
  unqualified columns, schema-qualified tables, an alias that shadows
  another alias/CTE/view name already in scope, `WITH RECURSIVE`, nested
  `WITH`, CTE column lists, table functions other than `input()`, `unnest()`
  and `generate_series()`, a statement that is neither a `SELECT` nor (in a script)
  `CREATE VIEW` / `COPY`, a zero/negative/computed array subscript.
- `generate_series(...)`: a bound or step that is not an integer literal
  after `-v` substitution (a column reference included), a `0` step, a
  descending or empty range, and an unaliased call -- the alias is
  mandatory, same as `input()`/`unnest()`/`ffmpeg.<source>()`.
- Any name that is neither one of the four `ffrwd.*` macros nor a filter
  the installed ffmpeg provides (see Calling convention), including
  transitions between concatenated branches, motion tracking,
  subtitle/data-stream filtering, and anything requiring more than one pass
  over the stream.
- A field read off a filter's OUTPUT (`scale(v, 640, -2).width`,
  `volume(a, 0.2).tags.language`): nothing probed it. Setting a read-only
  field (`STRUCT('h264' AS codec) AS tags`, `STRUCT(3 AS index) AS tags`):
  it is a probed fact, not an assertion. An aliased scalar column that is
  not `tags`/`chapters`/`attachments`/`disposition` at a media sink: a
  SELECT column there is an output stream, and metadata is written by a
  `tags` column. `SELECT *` over a track-row alias in a media
  query: a star expands FIELDS, and a SELECT column is an output stream.
- A written chapter list that goes backwards: each chapter must end after it
  starts, and the rows must ascend without overlapping.
- A `WHERE` window on an alias whose subtitle/data column is ALSO selected in
  the same query (ffmpeg cannot retime captions under an input seek -- see
  Time selection); a subtitle/data column inside a CTE that also carries a
  `WHERE` window (a CTE trim is a filtergraph trim, which cannot carry
  captions at all)."""


# The function reference, rendered from the live Registry -- machine-dependent
# by design, since what a bare or `ffmpeg.<name>` call resolves to is exactly
# what THIS installed ffmpeg reports. Name, pad signature, one-line doc, sorted
# alphabetically; no per-filter option dump (~460 filters' worth of option
# tables would be enormous, and `validate --json`'s UNKNOWN_FILTER_OPTION /
# FILTER_OPTION_TYPE cover that long tail instead).
#
# `registry.available()` being False degrades to one explanatory note
# (guardrail #7: a broken provisioner must fail typed, not crash). That note is
# the only part of `build_system_prompt` that can vary given the same registry.

_NO_REGISTRY_NOTE = (
    "This registry is unavailable, which means the ffmpeg provisioner failed "
    "to supply a working ffmpeg -- there is no per-machine filter list to "
    "render here. Calling convention above still describes the mechanism -- "
    "bare and `ffmpeg.<name>` calls both resolve against the installed "
    "ffmpeg's filter set, whatever it turns out to be, once the provisioner "
    "is fixed. `ffrwd.blur_regions` / `ffrwd.speed` / `ffrwd.delay` / "
    "`ffrwd.loudnorm2` need no registry at all and always compile."
)


def _filter_line(name: str, registry: Registry) -> str:
    f = registry.get(name)
    assert f is not None  # name came from registry.names() itself
    signature = ", ".join(f.inputs)
    return f"- `{name}({signature}) -> {f.output}` -- {f.doc}"


def _function_reference(registry: Registry) -> str:
    if not registry.available():
        return f"## Functions\n\n{_NO_REGISTRY_NOTE}"
    names = sorted(registry.names())
    lines = [
        f"## Functions ({len(names)} installed filters, plus the "
        f"{len(MACROS)} ffrwd.* macros)",
        "",
        "Every filter this machine's installed ffmpeg reports -- name, pad "
        "signature, one-line description, sorted alphabetically -- callable "
        "bare or as `ffmpeg.<name>` (see Calling convention). This list is "
        "machine-dependent: it is only what THIS ffmpeg reported when this "
        "prompt was generated. Options are never dumped here -- pass them by "
        "name (`<name> => <value>`) as usual and let `ffrwd validate "
        "--json` report the real option set on a mistake "
        "(`UNKNOWN_FILTER_OPTION` / `FILTER_OPTION_TYPE`); the repair loop "
        "below covers the rest. The four `ffrwd.*` macros "
        "(`blur_regions`, `speed`, `delay`, `loudnorm2`) are not in this "
        "list -- their signatures are fixed and given in full under Calling "
        "convention.",
        "",
    ]
    lines.extend(_filter_line(name, registry) for name in names)
    return "\n".join(lines)


# (natural-language request, query). Every query here is compiled by
# tests/test_prompt.py -- an example that does not compile fails the build.
_EXAMPLES: tuple[tuple[str, str], ...] = (
    (
        "Make clip.mp4 half size.",
        "SELECT scale(a.video[1], 'iw/2', 'ih/2')\nFROM input('clip.mp4') a",
    ),
    (
        "Crop the 640x360 region at (100, 50) out of clip.mp4 and double it.",
        "SELECT scale(crop(a.video[1], 640, 360, 100, 50), 'iw*2', 'ih*2')\n"
        "FROM input('clip.mp4') a",
    ),
    (
        "Keep only the part of clip.mp4 from 5s to 12.5s.",
        "SELECT a.video[1]\nFROM input('clip.mp4') a\nWHERE a.t BETWEEN 5 AND 12.5",
    ),
    (
        "Drop everything before the 90 second mark in clip.mp4; keep the rest.",
        "SELECT a.video[1]\nFROM input('clip.mp4') a\nWHERE a.t >= 90",
    ),
    (
        "Put a half-size copy of the scoreboard (the 600x200 box at 1200,50) "
        "in the top-left corner of game.mp4, and mix its audio under the main "
        "feed's at 65/35.",
        "WITH pip AS (\n"
        "  SELECT scale(crop(b.video[1], 600, 200, 1200, 50), 'iw/2', 'ih/2') AS frame,\n"
        "         b.audio[1] AS sound\n"
        "  FROM input('game.mp4') b\n"
        ")\n"
        "SELECT overlay(a.video[1], pip.frame, 20, 20),\n"
        "       amix(volume(a.audio[1], 0.65), volume(pip.sound, 0.35))\n"
        "FROM input('game.mp4') a, pip",
    ),
    (
        "Keep only the first video stream and the second audio stream of "
        "foo.mp4, untouched.",
        "SELECT a.video[1], a.audio[2]\nFROM input('foo.mp4') a",
    ),
    (
        "Halve the size of clip.mp4's video and keep its first audio track "
        "as-is.",
        "SELECT scale(a.video[1], 'iw/2', 'ih/2'), a.audio[1]\nFROM input('clip.mp4') a",
    ),
    (
        "Blur the 160x120 area at (220, 90) in clip.mp4.",
        "SELECT ffrwd.blur_regions(a.video[1], 220, 90, 160, 120, 20)\n"
        "FROM input('clip.mp4') a",
    ),
    (
        "Outline a red 300x120 box at (40, 40) on clip.mp4 and label it "
        "TAKE 3 at (60, 70) in 36pt.",
        "SELECT drawtext(drawbox(a.video[1], 40, 40, 300, 120, 'red'), "
        "text => 'TAKE 3', x => 60, y => 70, fontsize => 36)\n"
        "FROM input('clip.mp4') a",
    ),
    (
        "Center watermark.png over film.mp4, whatever size either of them is.",
        # An expr argument: ffmpeg evaluates (W-w)/2 against the real frame
        # sizes, so this needs no probe and works for any pair of inputs.
        "SELECT overlay(f.video[1], logo.video[1], '(W-w)/2', '(H-h)/2')\n"
        "FROM input('film.mp4') f, input('watermark.png') logo",
    ),
    (
        "Play intro.mp4 and then main.mp4 as one video.",
        "SELECT a.video[1] FROM input('intro.mp4') a\n"
        "UNION ALL\n"
        "SELECT b.video[1] FROM input('main.mp4') b",
    ),
    (
        "Double-speed the 20-second clip.mp4 with a 1 second fade in and a "
        "1.5 second fade out at the end.",
        # 20s source at 2x -> 10s output; the fade-out window starts at 10 - 1.5.
        "SELECT fade(fade(ffrwd.speed(a.video[1], 2), 'in', duration => 1), "
        "'out', start_time => 8.5, duration => 1.5)\n"
        "FROM input('clip.mp4') a",
    ),
    (
        "Pull the first subtitle track out of film.mkv into its own file.",
        "COPY (SELECT a.subtitle[1] FROM input('film.mkv') a) TO 'subs.en.srt'",
    ),
    (
        "Join subs.en.vtt onto clip.mp4 as a proper subtitle track, mov_text "
        "coded for an mp4 output, video and audio untouched.",
        "COPY (\n"
        "  SELECT a.video[1], a.audio[1], s.subtitle[1]\n"
        "  FROM input('clip.mp4') a, input('subs.en.vtt') s\n"
        ") TO 'out.mp4' WITH (subtitle_codec 'mov_text')",
    ),
    (
        "Encode film.mkv to a 720p and a 360p mp4 plus a standalone AAC "
        "audio file, decoding and filtering the source only once.",
        "CREATE VIEW master AS\n"
        "  SELECT scale(f.video[1], 1920, -2) AS v, volume(f.audio[1], 0.9) AS a\n"
        "  FROM input('film.mkv') f;\n"
        "COPY (SELECT scale(m.v, 1280, -2) AS v, m.a FROM master m)\n"
        "TO '720.mp4' WITH (video_codec 'libx264', crf 21, audio_codec 'aac');\n"
        "COPY (SELECT scale(m.v, 640, -2) AS v, m.a FROM master m)\n"
        "TO '360.mp4' WITH (video_codec 'libx264', crf 26, audio_codec 'aac');\n"
        "COPY (SELECT m.a FROM master m)\n"
        "TO 'audio.m4a' WITH (audio_codec 'aac', audio_bitrate '128k')",
    ),
    (
        "Give film.mkv four evenly spaced chapter markers, ten seconds each.",
        "COPY (\n"
        "  SELECT f.video[1], f.audio[1],\n"
        "         array_agg(STRUCT('Chapter ' || i.i::text AS title,\n"
        "                          (i.i - 1) * 10 AS start_t,\n"
        "                          i.i * 10 AS end_t)::chapter) AS chapters\n"
        "  FROM input('film.mkv') f, generate_series(1, 4) i\n"
        ") TO 'chaptered.mkv'",
    ),
)

# Examples that broadcast a bare array need a real, readable file to know how
# many streams to expand over. tests/test_prompt.py compiles every ```sql code
# block as a promise it succeeds standalone, so these are tagged ```sql-probed:
# a distinct tag the extractor does not match.
_PROBED_EXAMPLES: tuple[tuple[str, str], ...] = (
    (
        "Add echo/reverb to every audio track in film.mkv, whatever language "
        "each one is in.",
        "SELECT v.video[1], aecho(v.audio, 0.8, 0.9, 60, 0.3)\nFROM input('film.mkv') v",
    ),
    (
        "Play episode1.mkv then episode2.mkv as one file, keeping every "
        "language track of both.",
        "SELECT a.video[1], a.audio FROM input('episode1.mkv') a\n"
        "UNION ALL\n"
        "SELECT b.video[1], b.audio FROM input('episode2.mkv') b",
    ),
    (
        "Put a quarter-size picture-in-picture copy of commentary.mkv in the "
        "corner of film.mkv, and mix every language track of both under it, "
        "the film at 65% and the commentary at 35%.",
        "WITH pip AS (\n"
        "  SELECT scale(c.video[1], 'iw/4', 'ih/4') AS frame, c.audio AS sound\n"
        "  FROM input('commentary.mkv') c\n"
        ")\n"
        "SELECT overlay(f.video[1], pip.frame, 20, 20),\n"
        "       amix(volume(f.audio, 0.65), volume(pip.sound, 0.35))\n"
        "FROM input('film.mkv') f, pip",
    ),
    (
        "Keep absolutely everything in film.mkv untouched -- every video, "
        "audio, subtitle and data stream it has.",
        "SELECT * FROM input('film.mkv') a",
    ),
    (
        "Pull every English audio track out of film.mkv, whatever subscript "
        "each one happens to sit at.",
        "SELECT array_agg(t)\nFROM input('film.mkv') f, unnest(f.audio) t\n"
        "WHERE t.tags.language = 'eng'",
    ),
    (
        "Mix film.mkv's audio with commentary.mkv's, matched by language; "
        "where commentary.mkv has no matching language, fill with 2 "
        "seconds of silence instead.",
        "SELECT array_agg(\n"
        "         amix(a, COALESCE(b, ffmpeg.anullsrc(duration => 2)))\n"
        "       )\n"
        "FROM input('film.mkv') f, input('commentary.mkv') g,\n"
        "     unnest(f.audio) a FULL OUTER JOIN unnest(g.audio) b\n"
        "       ON a.tags.language = b.tags.language",
    ),
)

_EXAMPLES_HEADER = """\
## Examples"""

_PROBED_EXAMPLES_NOTE = (
    "The next examples use a bare array (`v.audio`, `a.audio`) -- broadcast "
    "over, or splatted into the SELECT list -- or an `unnest(...)` track-row "
    "table. Either way they only compile against a real, readable file, "
    "since sizing an array or building a row table's metadata needs the "
    "file's actual stream data (see Broadcasting and Track rows above)."
)


def _examples() -> str:
    lines = [_EXAMPLES_HEADER, ""]
    for task, query in _EXAMPLES:
        lines.append(task)
        lines.append("")
        lines.append("```sql")
        lines.append(query)
        lines.append("```")
        lines.append("")
    lines.append(_PROBED_EXAMPLES_NOTE)
    lines.append("")
    for task, query in _PROBED_EXAMPLES:
        lines.append(task)
        lines.append("")
        lines.append("```sql-probed")
        lines.append(query)
        lines.append("```")
        lines.append("")
    return "\n".join(lines).rstrip()


# The repair loop, keyed by ErrorCode: every code must have guidance.

_REPAIR_HEADER = """\
## Repair loop

Validate every query you produce:

    ffrwd validate --json "<your query>"

(or `ffrwd validate --json -f query.sql` if it is in a file). Exit 0 with
no output means it compiles. Exit 1 prints exactly one JSON object:

```json
{"line": 1, "col": 8, "code": "UDF_ARG_TYPE", "message": "scale() expects \
scale(video, num) | scale(video, num, num), got scale(video, video)", "hint": \
"arguments are stream expressions or literals, in the order shown"}
```

`line` and `col` are 1-based and point at the offending token. `hint` may be
null; when it is present it usually names the exact replacement. Change only
what the error names, then re-validate. Do not rewrite the whole query, and do
not repeat a fix that already failed -- if a construct is rejected twice, it is
not in the dialect."""

# code -> what to change in the SQL. Guidance, not a restatement of the error.
_REPAIR: dict[ErrorCode, str] = {
    ErrorCode.PARSE_ERROR: (
        "Not valid Postgres SQL. Check for unbalanced parentheses, a missing "
        "comma between FROM items, double quotes used for a string, or text "
        "after the statement. One more cause: a `=>` argument inside a call "
        "whose name Postgres parses specially (`overlay`, `trim`, `format`, "
        "...) -- write that call as `ffmpeg.<name>(...)` instead. Re-emit one "
        "well-formed SELECT."
    ),
    ErrorCode.UNKNOWN_FUNCTION: (
        "That name is neither one of the 4 `ffrwd.*` macros nor a filter "
        "the installed ffmpeg has. Take the did-you-mean from `hint` if "
        "there is one -- it comes back in the spelling that works, so "
        "`ffmpeg.<name>()` in the hint means write it with the namespace. "
        "Otherwise rebuild the effect from the Functions list, or declare "
        "it inexpressible. Do not invent ffmpeg filter names, and do not "
        "retry a bare name as `ffmpeg.<name>` unless the filter really "
        "exists: the namespace changes how the name resolves, not which "
        "filters exist."
    ),
    ErrorCode.UNKNOWN_ALIAS: (
        "The qualifier before the dot is not in this SELECT's FROM. Add "
        "`input('path') <alias>` or the CTE name to that FROM clause; `hint` "
        "lists what is in scope. Names do not cross UNION ALL branches, and a "
        "CTE is in scope only in the branches whose FROM names it."
    ),
    ErrorCode.UDF_ARG_TYPE: (
        "Wrong arity or wrong argument kind. Count the arguments against one "
        "signature in `message` and fix the mismatched slot: `video` needs "
        "`<alias>.video[k]`, `<alias>.frame`, or a nested video-returning call; "
        "`audio` needs `<alias>.audio[k]` or a nested audio-returning call; "
        "`num` needs a bare number; `str` needs a single-quoted literal. "
        "`video`/`audio` in the got-list where the other was expected usually "
        "means you mixed up video and audio; a stream kind where `num` was "
        "expected usually means you passed a stream or `<alias>.t` instead of "
        "a number. A got-list that is EMPTY (`trim(), got trim()`) for a call "
        "you did pass arguments to means Postgres claimed the name: rewrite it "
        "as `ffmpeg.<name>(...)`."
    ),
    ErrorCode.SINGLE_OUTPUT_ONLY: (
        "Reserved; ffrwd never raises this code -- a multi-column SELECT is "
        "ordinary usage, one column per output stream. If you see it anyway, "
        "treat it as INTERNAL: re-emit the simplest form of the query."
    ),
    ErrorCode.NO_STREAMING_EQUIVALENT: (
        "Delete the clause named in `message`; there is no filtergraph form for "
        "it. If you used it to pick a time range, use "
        "`WHERE <alias>.t BETWEEN a AND b`, `<alias>.t >= a`, or `<alias>.t <= "
        "b`. If you used it to pick one branch, just write that branch."
    ),
    ErrorCode.CONCAT_MISMATCH: (
        "UNION ALL branches disagree. `message` says how: if it names stream "
        "types, column counts, or order, make every branch select the same "
        "columns, the same types, in the same order (a splatted array counts "
        "one column per element, so array lengths must match too); if it is a "
        "resolution/frame-rate mismatch, wrap each branch's video column in "
        "`scale(<expr>, width, height)` with the same width and height "
        "everywhere."
    ),
    ErrorCode.STREAM_NOT_FOUND: (
        "The subscript is out of range for the actual stream count named in "
        "`message` (from the probed file, or a CTE array column's recorded "
        "length). Subscripts are 1-based -- lower the number into range, or "
        "select a different alias/column/element."
    ),
    ErrorCode.INPUT_NOT_FOUND: (
        "A bare array (`<alias>.video` / `<alias>.audio` with no subscript, "
        "splatted or handed to a function) needs to read the file to know how "
        "many streams it has, and this input cannot be read (missing path or "
        "a URL). Subscript one specific stream instead -- `hint` names one -- "
        "or point at a path you know is readable."
    ),
    ErrorCode.BROADCAST_MISMATCH: (
        "Two array arguments to the same call have different lengths (both "
        "named in `message`) and cannot zip elementwise. Subscript one of "
        "them down to a single stream, e.g. `<alias>.audio[1]`, so it "
        "broadcasts as a scalar instead, or make both arrays the same length."
    ),
    ErrorCode.ROW_COUNT_MISMATCH: (
        "The query's relation has more rows (or more groups) than the one file "
        "it writes, and rows are never combined silently. Two fixes, and "
        "`message` says which count is involved: wrap the row column in "
        "`array_agg(...)` so the rows land in that one file as consecutive "
        "streams -- add `GROUP BY <the column they share>` when another column "
        "has to stay unaggregated -- or write one file per row by replacing the "
        "quoted `TO 'path'` with `TO (<expression over the row's columns>)`, "
        "e.g. `TO (t.tags.language || '.mka')`. Narrowing the `WHERE` until one row "
        "survives also works when a single track is what you meant."
    ),
    ErrorCode.UNSUPPORTED_SQL: (
        "The construct is outside the dialect. Read `hint` -- it names the "
        "replacement: a CTE instead of a subquery, a comma instead of "
        "`JOIN ... ON`, a bare CTE name instead of an aliased one, one lower "
        "and one upper time bound per alias, `<alias>.video`/`<alias>.audio` "
        "instead of `*` or an unqualified column, a positive integer literal "
        "subscript instead of "
        "zero, a negative number, or a computed index."
    ),
    ErrorCode.UNKNOWN_SINK_OPTION: (
        "The name inside `COPY ... WITH (...)` is not a known sink option. "
        "Take the did-you-mean from `hint` if there is one, otherwise pick "
        "from the exact set of option names ffrwd supports; do not invent "
        "an ffmpeg flag name or option that isn't in that set."
    ),
    ErrorCode.SINK_OPTION_TYPE: (
        "The value given for that `COPY ... WITH (...)` option does not "
        "match its expected type, named in `message`. A `str` option needs a "
        "single-quoted literal (e.g. `video_codec 'libx264'`), an `int` "
        "option needs a bare integer literal with no quotes and no decimal "
        "point (e.g. `crf 20`), and a `bool` option needs exactly `true` or "
        "`false` with no quotes."
    ),
    ErrorCode.UNKNOWN_INPUT_OPTION: (
        "The name inside `input('path', ...)`'s trailing `name => value` "
        "arguments is not a known input option. Take the did-you-mean from "
        "`hint` if there is one, otherwise pick from the exact set of option "
        "names ffrwd supports (see Dialect > Sources); do not invent an "
        "ffmpeg flag name or option that isn't in that set. Unlike a sink "
        "option name, an input option name is CASE-SENSITIVE."
    ),
    ErrorCode.INPUT_OPTION_TYPE: (
        "The value given for that `input(...)` option does not match its "
        "expected type, named in `message`. A `str` option needs a "
        "single-quoted literal, an `int` option needs a bare integer literal "
        "with no quotes and no decimal point, a `num` option needs a bare "
        "numeric literal (a leading `-` is fine, e.g. `itsoffset => -1`), and "
        "a `bool` option needs exactly `true` or `false` with no quotes."
    ),
    ErrorCode.UNKNOWN_FILTER_OPTION: (
        "A `<name> => <value>` argument names an option the ffmpeg filter does "
        "not have. Take the did-you-mean from `hint` if there is one; otherwise "
        "pick from the option names `hint` lists -- they were read out of the "
        "installed ffmpeg, so they are the complete set for that filter. Never "
        "invent an option name, and never move the value to a positional "
        "argument: a dynamic filter takes only its stream inputs positionally."
    ),
    ErrorCode.FILTER_OPTION_TYPE: (
        "The value of a `<name> => <value>` argument does not match what the "
        "option accepts, and `message` says exactly what it does accept: a bare "
        "number (with the allowed range, if ffmpeg declares one), the bare word "
        "`true`/`false`, or one of a listed set of named constants -- those go "
        "in single quotes, e.g. `transition => 'wipeleft'`. Change the value "
        "only; the option name was accepted."
    ),
    ErrorCode.UNBOUNDED_LIVE_INPUT: (
        "The input named at `line`/`col` can only be opened once -- a protocol "
        "URL, or an input whose `format =>` names a device -- so one process "
        "reads it and hands every other one a pipe. Two of those pipes come "
        "back together later in the query, and `message` names the stage "
        "between them whose frame count ffrwd cannot follow, so it cannot say "
        "how deep the buffers between them have to be. Take that stage out of "
        "the part of the query that runs between the input and the point the "
        "two paths meet -- apply it after they meet, or drop it -- or record "
        "the input to a file first and write the query over the file."
    ),
    ErrorCode.STARTUP_DEADLOCK: (
        "The query splits one input's streams apart and brings them back "
        "together in a way whose processes would each be waiting on the next "
        "before any of them could read or write; `message` names them in the "
        "order they wait. Rewrite the query with one fewer place where the "
        "streams off a single input split and rejoin -- most often by taking "
        "a module out of the middle of one leg -- or write it over a recorded "
        "file rather than the live input."
    ),
    ErrorCode.INTERNAL: (
        "A compiler bug, not your SQL. Re-emit the simplest query that still "
        "expresses the request and report the original as a bug."
    ),
}


# Codes no compile can raise: they refuse a `run` flag or a command-line
# name, or say how a run ended, rather than refusing a query -- so no
# rewriting of the SQL answers one and the repair loop never sees it.
NOT_REPAIRABLE: frozenset[ErrorCode] = frozenset(
    {
        ErrorCode.BUFFER_OVERFLOW,
        ErrorCode.NOTHING_TO_SHOW,
        ErrorCode.PLAYER_NOT_FOUND,
        ErrorCode.UNKNOWN_RECIPE,
    }
)


def _repair_section() -> str:
    lines = [_REPAIR_HEADER, ""]
    for code in ErrorCode:
        if code in NOT_REPAIRABLE:
            continue
        lines.append(f"- `{code.value}` -- {_REPAIR[code]}")
    return "\n".join(lines)


def build_system_prompt(registry: Registry) -> str:
    """The ffrwd system prompt: ASCII, no trailing newline.

    `registry` is REQUIRED (ffmpeg is always there, PATH or the
    provisioner, so there is always a real registry to pass) and is the only
    thing that can vary the output -- everything else here is pure, with no
    I/O, clock, or environment of its own. `ffrwd prompt` passes the live
    registry (`registry.load()`), machine-dependent by nature; the test
    suite passes the committed reference snapshot
    (`registry.load_reference(...)`) instead, so its assertions hold on
    every machine. `registry.available()` being False (a
    failed provisioner, guardrail #7) degrades the Functions section to one
    explanatory note rather than crashing; that is the only other thing that
    can vary given two different registries.
    """
    sections: tuple[str, ...] = (
        _ROLE,
        _dialect_section(),
        _output_section(),
        _REJECTED,
        _function_reference(registry),
        _examples(),
        _repair_section(),
    )
    return "\n\n".join(sections)
