# ffrwd error contract

Every rejection is a `FfrwdError`: a typed `code`, a `message`, an optional `line`/`col` anchor, and an optional `hint`. `code` is an `ErrorCode` value from `ffrwd/errors.py`; the JSON shape is `docs/error-schema.json`.

Structured form: `ffrwd validate --json "<query>"` (or `-f query.sql`). Success prints nothing, exit 0; rejection prints one JSON object to stdout, exit 1. Every example below is real captured output; probe-dependent ones were captured against `tests/fixtures/` (`av.mp4`: 1 video + 1 audio; `av2.mp4`: 1 video + 2 audio tagged eng/fra; `avs.mkv`: video + audio + eng subtitle).

## PARSE_ERROR

**Meaning:** The query text is not valid SQL under sqlglot's `postgres` dialect (guardrail #2: ffrwd always parses Postgres dialect). This fires before any ffrwd-specific validation runs. It is whatever sqlglot itself rejects, plus ffrwd's own "empty query" and "no statement found" checks for degenerate input.

**Fires when:** the text fails to tokenize or parse. Missing parens, garbled keywords, truncated statements, an empty or whitespace-only query.

**Example query:**

```sql
SELECT a.video[1] FROM input('x.mp4' a
```

(note the missing closing paren after `'x.mp4'`)

**Error JSON:**

```json
{"line": 1, "col": 38, "code": "PARSE_ERROR", "message": "Expecting )", "hint": null}
```

## UNKNOWN_FUNCTION

**Meaning:** A call names a function that is not a filter the installed ffmpeg reports (`ffrwd/registry.py`, see [docs/filters.md](filters.md)) and not one of the three `ffrwd.<name>` macros. Checked for the outer call and for nested calls used as arguments.

**Fires when:** the name resolves nowhere - a typo, or a filter your ffmpeg build doesn't ship. (An empty registry - possible only if the ffmpeg provisioner failed - makes every filter name unknown, and the hint says exactly that.)

**Example query:**

```sql
SELECT gblu(a.video[1])
FROM input('x.mp4') a
```

**Error JSON** (against ffmpeg 7.1; the candidate list is that binary's):

```json
{"line": 1, "col": 8, "code": "UNKNOWN_FUNCTION", "message": "unknown function gblu()", "hint": "did you mean gblur()?"}
```

The hint is a did-you-mean match against every filter name the installed ffmpeg reports. A `ffrwd.<name>` call matches against the macro names the same way (`ffrwd.dela()` suggests `ffrwd.delay()`). Should the registry come up empty - which since ffmpeg became a managed requirement means the provisioner failed - the hint states that real problem instead of guessing: check `static-ffmpeg` installed correctly, or put a system ffmpeg on `PATH`.

`concat` is the one name this fires for even when spelled exactly right: it has a variable pad count, so it is unknown UNLESS the call carries `VARIADIC` (see `UDF_ARG_TYPE` and `UNSUPPORTED_SQL` below), and the hint says so instead of guessing a near miss:

```json
{"line": 1, "col": 8, "code": "UNKNOWN_FUNCTION", "message": "unknown function concat()", "hint": "concat has a variable pad count: call it with VARIADIC, e.g. concat(VARIADIC array_agg(v))"}
```

## UNKNOWN_ALIAS

**Meaning:** A `<alias>.video`/`<alias>.audio` reference (or a bare table reference in `FROM`) names an alias or CTE this query never introduced.

**Fires when:** you reference `b.video[1]` but only declared `FROM input('x.mp4') a`, or reference a CTE that hasn't been defined or isn't visible yet (CTEs see only earlier CTEs in the same `WITH`; no forward references).

**Example query:**

```sql
SELECT b.video[1] FROM input('x.mp4') a
```

**Error JSON:**

```json
{"line": 1, "col": 8, "code": "UNKNOWN_ALIAS", "message": "unknown alias 'b'", "hint": "known names: a"}
```

## UDF_ARG_TYPE

**Meaning:** A call's *stream* arguments don't match. For a filter, that is the pad signature: `gblur` is `V->V`, so exactly one video in; `xfade` is `VV->V`, so two. For a `ffrwd.<name>` macro it is the macro's own signature. Option problems are never this code - a positional option validates as the option it binds to, so those are `UNKNOWN_FILTER_OPTION`/`FILTER_OPTION_TYPE` below. The exceptions: more positional options than the filter has options at all; an N-input filter's count option (`inputs`, `n`, ...) disagreeing with the stream count you actually passed, written out or spread with `VARIADIC`; a `VARIADIC` argument that is not an array, or is an empty one (naming what produced it) - all arity statements, all here.

**Fires when:** a stream is missing, one too many, or the wrong type - `gblur(a.audio[1], 5)` hands audio to a video filter, that sort of thing.

**Example query:**

```sql
SELECT gblur(a.audio[1], 5)
FROM input('x.mp4') a
```

**Error JSON:**

```json
{"line": 1, "col": 8, "code": "UDF_ARG_TYPE", "message": "gblur() is an ffmpeg filter: it takes video as its stream input, got (audio)", "hint": "stream inputs come first, then options in the filter's own order, then named options: gblur(video, <option>, <option> => <value>)"}
```

The arity flavor, captured too - the hint lists the filter's options in the order positionals bind, which is usually what you were trying to remember:

```json
{"line": 1, "col": 30, "code": "UDF_ARG_TYPE", "message": "setsar() got 3 positional options, but the 'setsar' filter has 2", "hint": "its options, in the order they bind: ratio, max"}
```

The macro flavor:

```json
{"line": 1, "col": 22, "code": "UDF_ARG_TYPE", "message": "ffrwd.delay() takes a video stream as its 'f' argument, got audio", "hint": "ffrwd.delay() is the video (transparent-canvas) macro; delay an audio stream with the bare filter directly, in milliseconds, e.g. adelay(a.audio[1], delays => '2000')"}
```

The NULL flavor: a stream position cannot be absent, so a NULL there — an unset variable's, or a literal one — is this code, naming the variable when there is one:

```json
{"line": 1, "col": 20, "code": "UDF_ARG_TYPE", "message": "':clip' was not set", "hint": "scale() needs a stream in this position; set it with -v clip=<value>"}
```

## SINGLE_OUTPUT_ONLY

**Reserved, not currently raised.** The SELECT list is the output stream list (every column is its own `-map`, in order), so a multi-column SELECT is ordinary usage rather than an error. The code stays in the enum, and in `docs/error-schema.json`'s `code` enum, purely for wire-format stability. No code path in `ffrwd/*.py` raises it, and no example JSON exists because none can.

## NO_STREAMING_EQUIVALENT

**Meaning:** The construct is valid SQL with perfectly good relational semantics that simply cannot exist in a single-pass streaming filtergraph. ffmpeg processes frames as they arrive; it cannot look ahead, sort, or deduplicate across an input.

**Fires when:** the query uses `HAVING`, `SORT BY`, `CLUSTER BY`, `DISTRIBUTE BY`, `DISTINCT`, `QUALIFY`, `WINDOW`, `CONNECT BY`, a window function, a subquery predicate (`IN (SELECT ...)`, `EXISTS`, etc.), or `UNION` without `ALL` (plain `UNION` requires deduplication, and there is no meaningful way to deduplicate a stream of frames). `GROUP BY`, `ORDER BY`, `LIMIT`, `OFFSET`, and `array_agg` are legal over track rows, media query or table query alike - they operate on the compile-time row table, not the streams; outside track rows they fire this error, and other aggregate functions (`count`, `sum`, ...) always do.

**Example query:**

```sql
SELECT a.video[1]
FROM input('x.mp4') a
GROUP BY a.video[1]
```

**Error JSON:**

```json
{"line": 3, "col": 10, "code": "NO_STREAMING_EQUIVALENT", "message": "GROUP BY has no streaming equivalent", "hint": "remove the GROUP BY clause"}
```

## CONCAT_MISMATCH

**Meaning:** `UNION ALL` lowers to ffmpeg's `concat` filter, which demands every branch produce the identical column signature: same count, same stream types, same order. Array columns count per element, so `audio[2]` against `audio[1]` mismatches too. When every input is probeable this also catches real fps/resolution/sample-rate disagreements at the filter level; the type/count/order check needs no probing and fires from the SQL shape alone.

**Fires when:** two `UNION ALL` branches select a different number of columns, different stream types, a different order, or differently-sized arrays.

**Example query:**

```sql
SELECT a.video[1]
FROM input('x.mp4') a
UNION ALL
SELECT b.audio[1]
FROM input('y.mp4') b
```

**Error JSON:**

```json
{"line": 4, "col": 8, "code": "CONCAT_MISMATCH", "message": "UNION ALL branches must select the same stream types in the same order: branch 1 selects (video), branch 2 selects (audio)", "hint": "ffmpeg concat needs identical segments; reorder or add columns"}
```

## UNSUPPORTED_SQL

**Meaning:** The catch-all for syntactically valid SQL outside the dialect that isn't one of the more specific codes above. No streaming-vs-batch philosophy involved; the surface just doesn't include it. This is the most common code in practice. Most of `ffrwd/parser.py`'s rejections use it: multiple statements, unsupported clause keys, explicit `JOIN` syntax (comma cross-joins only), aliased or nested subqueries, `WITH RECURSIVE`, malformed or duplicate CTE/alias names, an empty `WHERE`, a non-positive or non-literal array subscript, or a top-level statement that isn't a `SELECT`/`UNION ALL`. (`SELECT *` and `<alias>.*` compile now, so they are off this list; see [docs/trimming.md](trimming.md) for the caption rejections below.)

The unset-variable rejections land here when the NULL sits where a value is REQUIRED. An unset `:'variable'` (psql-style, filled by `-v name=value`) substitutes to `NULL` — absence, which drops an option cleanly — but `input()` needs a path, `COPY` needs a destination, and a `TO` expression must come out text, so a NULL there is rejected at its point of use, naming the variable when the NULL came from one:

```json
{"line": 1, "col": 30, "code": "UNSUPPORTED_SQL", "message": "':source' was not set", "hint": "input() needs a path; set it with -v source=<value>"}
```

A literal NULL in the same positions is the same rejection without the name (`"input() needs a path, got NULL"`). The reverse check — `-v name=value` for a name the text never references — is a CLI usage error (exit 2), not a compile error, and carries no code.

One probed-reality rejection lands here: a stream ffprobe reports NO codec for (some DASH manifests' WebVTT tracks arrive this way - ffmpeg's demuxer sees them but cannot name them, an open ffmpeg limitation measured through 9.0) selected into a media sink. Such a stream can be neither copied (no tag to write) nor transcoded (no decoder to invoke), so the run is guaranteed to die at header-write; ffrwd knows at compile time and says so at compile time. Table queries are exempt on purpose - a codec-less track shows up as a row with a NULL codec column, which is how you discover it:

```json
{"line": 1, "col": 26, "code": "UNSUPPORTED_SQL", "message": "'s' (row 1) has no identifiable codec: ffmpeg's demuxer reports none, so the stream can be neither copied nor transcoded and no container can carry it", "hint": "drop it from the SELECT (a query with no COPY can still inspect it as a table row, codec column NULL); if it is a subtitle track, extract it with a tool that can read it and mux the resulting file as its own input() instead"}
```

Two argument-shape rejections land here: a named argument written out of place (a positional after a named one, or the same name twice - standard Postgres rules), and a named argument on a `ffrwd.<name>` macro, whose signature is positional only:

```json
{"line": 1, "col": 42, "code": "UNSUPPORTED_SQL", "message": "ffrwd.delay() is a ffrwd macro: its arguments are positional only, in the documented order", "hint": "its signature is ffrwd.delay(f, seconds)"}
```

Three caption rejections land here too, all consequences of one measured fact: ffmpeg does not retime subtitle/data packets under an input seek (the receipts are in [docs/trimming.md](trimming.md)):

- a `WHERE <alias>.t BETWEEN ...` window on an input alias whose subtitle/data column is also selected in the same query. The seeked track would play out of sync with the rebased video, so it is rejected instead of shipped broken (captured example below);
- a `WHERE` window on a CTE whose columns include a subtitle/data column. A CTE trim is a filtergraph trim (`trim`/`atrim`), and a filtergraph cannot carry captions at all, so this rejects unconditionally, selected or not;
- a subtitle/data column inside a `UNION ALL` branch. `concat` has video and audio pads, full stop.

Three script rejections (`CREATE VIEW name AS <query>;`* followed by `COPY ...;`+, see [the README](../README.md#views-and-multiple-outputs)) land here too, all typo/shape guards rather than anything semantic:

- a view nobody ever reads, anchored on its `CREATE VIEW` (captured example below) -- a script's whole point is that its views feed later views or COPYs, so one that feeds nothing is almost always a misspelled name somewhere else;
- a bare `SELECT` sitting among other statements. Only `COPY` carries a destination, so a lone `SELECT` in a multi-statement script has nowhere to send its streams -- wrap it in `COPY (<query>) TO '<path>'`;
- a `CREATE VIEW` written after the first `COPY`. Every view must precede every `COPY`, so the whole script can resolve names left-to-right in one pass.

The output fan-out rejections land here too. `COPY (...) TO (<expression>)` writes one file per surviving row when the expression reads a row table's columns, and everything about that shape that cannot be answered is typed rather than guessed at:

- a computed path segment holding a path separator or `..` -- a directory chosen by file metadata, named with the offending value;
- two rows naming the same file, naming both rows and the collision;
- zero surviving rows, and a `TO` expression that comes out NULL for a row (naming the column that was never probed);
- a `TO` expression that is not text;
- and, in this version, fan-out combined with `two_pass`, a `chapters` column, `FORMAT csv`, `UNION ALL`, or another `COPY` in the same script.

**Fires when:** (one example among many) the query selects a subtitle stream that sits inside its own alias's `WHERE` time window.

**Example query** (`tests/fixtures/avs.mkv` has one subtitle stream):

```sql
SELECT a.subtitle[1]
FROM input('tests/fixtures/avs.mkv') a
WHERE a.t BETWEEN 1 AND 2
```

**Error JSON:**

```json
{"line": 1, "col": 8, "code": "UNSUPPORTED_SQL", "message": "'WHERE a.t' cannot trim a selected subtitle stream: ffmpeg does not retime caption packets under an input seek, so they would play out of sync with the trimmed video", "hint": "trim the video/audio without selecting the subtitle/data columns, or select them in a query without a WHERE time range; to caption a trimmed clip, join an external subtitle file whose cues are timed for the cut"}
```

**Example query** (a script: the view is defined but nothing reads it):

```sql
CREATE VIEW unused AS SELECT a.video[1] AS v FROM input('x.mp4') a;
COPY (SELECT b.video[1] FROM input('x.mp4') b) TO 'out.mp4';
```

**Error JSON:**

```json
{"line": 1, "col": 13, "code": "UNSUPPORTED_SQL", "message": "view 'unused' is never used", "hint": "every view must be read by a later view or COPY; check the spelling of the name in its FROM clauses"}
```

## STREAM_NOT_FOUND

**Meaning:** A subscript (`<alias>.video[k]` / `<alias>.audio[k]`, or the recorded bound of a CTE array column) is out of range for the streams actually present. Only reachable against a probed input (local, readable) or against a CTE array column whose length was recorded when it lowered. An explicit subscript against an unprobed input compiles unchecked and lets ffmpeg deliver the bad news at run time instead.

**Fires when:** the subscript is positive and 1-based but exceeds the probed file's per-type stream count, exceeds a CTE array column's recorded length, or the whole array is empty (splatting `.audio` on a video-only file would select nothing, which is never what was meant).

**Example query** (`tests/fixtures/av.mp4` has exactly one video stream):

```sql
SELECT a.video[2]
FROM input('tests/fixtures/av.mp4') a
```

**Error JSON:**

```json
{"line": 1, "col": 8, "code": "STREAM_NOT_FOUND", "message": "'a.video[2]' does not exist: 'tests/fixtures/av.mp4' has 1 video stream", "hint": "stream subscripts are 1-based: a.video[1] is the first video stream"}
```

## INPUT_NOT_FOUND

**Meaning:** A bare array (`<alias>.video` / `<alias>.audio`, splatted into the SELECT list or broadcast through a function) needs the file's actual stream count to expand, and the input could not be probed: missing, unreadable, or a remote spec ffprobe couldn't fetch. (URLs DO probe - ffprobe is the authority on its own protocols, so an `https://` manifest's tracks are as unnestable as a local file's; only an unreachable or unsupported one lands here.) "Cannot enumerate the streams of a file I cannot read" gets its own honest error rather than hiding inside a generic probing-failure code. A path that exists but whose probe still failed gets a different message: "the probe failed (...)", quoting ffprobe's own last diagnostic line when there is one, with a hint pointing at the probe rather than at the file's existence.

**Fires when:** a bare array appears over an input with no probe result.

**Example query:**

```sql
SELECT a.audio
FROM input('nope.mp4') a
```

**Error JSON:**

```json
{"line": 1, "col": 8, "code": "INPUT_NOT_FOUND", "message": "cannot enumerate the streams of 'nope.mp4': file not found or unreadable", "hint": "'a.audio' is the whole stream array, and only a readable input can size it; subscript one stream, e.g. a.audio[1]"}
```

## BROADCAST_MISMATCH

**Meaning:** Broadcasting a call over more than one array argument zips them elementwise, no cross products, so every array argument must be the same length. A scalar argument (including a single subscripted stream) broadcasts freely and never triggers this.

**Fires when:** two or more array arguments to one call have different lengths.

**Example query** (`tests/fixtures/av2.mp4` has 2 audio streams,
`tests/fixtures/av.mp4` has 1):

```sql
SELECT amix(a.audio, b.audio)
FROM input('tests/fixtures/av2.mp4') a, input('tests/fixtures/av.mp4') b
```

**Error JSON:**

```json
{"line": 1, "col": 8, "code": "BROADCAST_MISMATCH", "message": "amix() cannot broadcast over arrays of different lengths: a.audio has 2 streams, b.audio has 1 stream", "hint": "broadcast arrays zip elementwise, one output per element; subscript one of them to pair a single stream with the other, e.g. a.audio[1]"}
```

## ROW_COUNT_MISMATCH

**Meaning:** One row is one file. A query whose relation resolves to more than one row (or, once it aggregates, more than one group) names a single destination, and rows are never combined behind your back. The two ways to combine them are both in the SQL: `array_agg(...)` gathers a branch's rows into the one file it writes (with `GROUP BY` when another column has to stay unaggregated), and `TO (<expression>)` gives every row a destination of its own.

**Fires when:** an `unnest(...)` table, a join or a multi-row CTE reference leaves several rows in a media `COPY` (or a bare `SELECT` with no destination) that writes one path. The count is the RESOLVED one, after the `WHERE` and the joins: a predicate that narrows the rows to one on the actual file compiles as it always did.

**Anchor:** the `TO` the file is named at, or the query itself for a bare `SELECT`.

**Example query** (`tests/fixtures/av2.mp4` has 2 audio streams):

```sql
COPY (
  SELECT t
  FROM input('tests/fixtures/av2.mp4') f, unnest(f.audio) t
) TO 'out.mka'
```

**Error JSON:**

```json
{"line": 4, "col": 6, "code": "ROW_COUNT_MISMATCH", "message": "this query has 2 rows, and 'out.mka' is one file", "hint": "gather the rows into that one file with array_agg(...), adding GROUP BY the column they share when they share one; or give each row a file of its own with a TO expression, e.g. TO (t.tags.language || '.mka')"}
```

Both exits compile: `SELECT array_agg(t)` writes both tracks into `out.mka`, and `TO (t.tags.language || '.mka')` writes one file per track instead.

## UNKNOWN_SINK_OPTION

**Meaning:** A `COPY (query) TO 'path' WITH (...)` option name is not one of the entries in `ffrwd.sink.SINK_OPTIONS`.

**Fires when:** an option in the `WITH (...)` list is misspelled or doesn't exist. `video_codc 'libx264'` instead of `video_codec 'libx264'`, or an option outside the v1 table entirely.

**Example query:**

```sql
COPY (
  SELECT a.video[1]
  FROM input('x.mp4') a
) TO 'out.mp4' WITH (
  video_codc 'libx264'
)
```

**Error JSON:**

```json
{"line": 5, "col": 14, "code": "UNKNOWN_SINK_OPTION", "message": "unknown sink option 'video_codc'", "hint": "did you mean 'video_codec'?"}
```

The anchor lands on the option's VALUE, not its name: sqlglot records no token position on a bare `WITH (...)` option name, only on a string or number literal, so `line`/`col` here point at the `'libx264'` literal, one line below the misspelled name. Imperfect, but deterministic and documented.

## SINK_OPTION_TYPE

**Meaning:** A `COPY (query) TO 'path' WITH (...)` option's value doesn't match the type declared for it in `ffrwd.sink.SINK_OPTIONS` (`str` / `int` / `bool`).

**Fires when:** a `str`-typed option gets a number or bool, an `int`-typed option gets a string or float, or a `bool`-typed option gets anything but `true`/`false`. `crf 'high'` (a string where `crf` wants an int) or `faststart 1` (an int where `faststart` wants a bool).

**Example query:**

```sql
COPY (
  SELECT a.video[1]
  FROM input('x.mp4') a
) TO 'out.mp4' WITH (
  crf 'high'
)
```

**Error JSON:**

```json
{"line": 5, "col": 7, "code": "SINK_OPTION_TYPE", "message": "option 'crf' expects an int, got 'high'", "hint": "crf takes a bare integer literal, e.g. crf 20"}
```

## UNKNOWN_FILTER_OPTION

**Meaning:** A named argument (`<name> => <value>`) names an option the targeted ffmpeg filter doesn't have. The option set is read out of the installed ffmpeg (`ffmpeg -help filter=<name>`, see `ffrwd/registry.py`), so it is exactly what that binary supports, not a table somebody in this repo has to keep current.

**Fires when:** the option name is misspelled or belongs to a different filter: `gblur(a.video[1], sigmma => 5)`. A *positional* option can't reach this code (it binds by position, so there is no name to get wrong); its failure modes are `FILTER_OPTION_TYPE` for a bad value and `UDF_ARG_TYPE` for one option too many.

**Example query:**

```sql
SELECT gblur(a.video[1], sigmma => 5)
FROM input('x.mp4') a
```

**Error JSON** (against ffmpeg 7.1; the option list is that binary's):

```json
{"line": 1, "col": 33, "code": "UNKNOWN_FILTER_OPTION", "message": "filter 'gblur' has no option 'sigmma'", "hint": "did you mean sigma => ...?"}
```

The anchor lands on the option's VALUE: sqlglot records no token position on the `exp.Var` holding a named argument's name (the same gap `COPY ... WITH` option names have), so `line`/`col` point at the `5`.

A query using this code compiles only where that ffmpeg does. With no working ffmpeg at all (a failed provisioner) the failure comes earlier, as `UNKNOWN_FUNCTION` on the filter name.

**Also fires for `enable`:** `enable` is never a real option of any filter (it is framework-level, see [docs/filters.md](filters.md)), so the validator special-cases the name instead of looking it up — but it names this same code, worded to say so, when the target filter isn't one your ffmpeg flags as timeline-capable (the `T` column of `ffmpeg -filters`):

```sql
SELECT scale(a.video[1], 640, 360, enable => 'gt(t,1)')
FROM input('x.mp4') a
```

```json
{"line": 1, "col": 43, "code": "UNKNOWN_FILTER_OPTION", "message": "filter 'scale' has no option 'enable': your ffmpeg does not flag 'scale' as supporting timeline editing", "hint": "enable is only accepted by filters your ffmpeg flags with timeline support (the T column of `ffmpeg -filters`: gblur has it, scale does not); drop it, or express the timing with a WHERE window over the input"}
```

## FILTER_OPTION_TYPE

**Meaning:** An option's value doesn't match its introspected type, declared range, or set of named constants. Positional or named makes no difference: a positional binds to the option its slot lands on and validates as that option, so `gblur(a.video[1], 5000)` and `gblur(a.video[1], sigma => 5000)` fail identically.

**Fires when:** a numeric option gets a string or a value outside its `(from A to B)` range, a boolean option gets anything but `true`/`false`, an enum option gets something that isn't one of its constants (or a bare number instead of a quoted constant name), or the option's ffmpeg type is one ffrwd cannot set at all (`binary`, `dictionary`).

**Example query:**

```sql
SELECT gblur(a.video[1], sigma => 5000)
FROM input('x.mp4') a
```

**Error JSON** (against ffmpeg 7.1; the range is that binary's):

```json
{"line": 1, "col": 32, "code": "FILTER_OPTION_TYPE", "message": "option 'sigma' of filter 'gblur' accepts a number from 0 to 1024, got 5000", "hint": "pick a value from 0 to 1024"}
```

Enum options quote their constant name (`transition => 'wipeleft'`), and the message lists the constants, truncated with a count when there are many (`xfade`'s `transition` alone has 59). Anchoring matches `UNKNOWN_FILTER_OPTION`: the value, not the name.

**Also fires for the positional/named collision:** a named argument naming an option a positional already bound is this code, never a silent override:

```json
{"line": 1, "col": 35, "code": "FILTER_OPTION_TYPE", "message": "option 'sigma' of filter 'gblur' is already set positionally by gblur()", "hint": "a named argument never overrides what the call itself set; drop one of the two spellings"}
```

**Also fires for `enable`:** on a filter that does accept it, `enable`'s value must still be a single-quoted string (an ffmpeg timeline expression) — anything else is this code, not `UNKNOWN_FILTER_OPTION`, since the name itself was fine:

```sql
SELECT gblur(a.video[1], 5, enable => 5)
FROM input('x.mp4') a
```

```json
{"line": 1, "col": 35, "code": "FILTER_OPTION_TYPE", "message": "option 'enable' of filter 'gblur' expects an ffmpeg timeline expression, got 5", "hint": "enable takes a single-quoted ffmpeg timeline expression over t (seconds), n (frame number) or pos, e.g. enable => 'between(t,2,5)'"}
```

**Also fires for a required option** (the hand-kept list in [docs/dialect.md](dialect.md#variables): `subtitles`' `filename`, `frei0r`'s `filter_name`, ...). A NULL value drops an option before validation — absence, ffmpeg's default applies — so a filter that cannot run without one rejects at compile time, whether the option was dropped by an unset variable (the message names it: `"':subs' was not set"`) or never written at all:

```json
{"line": 1, "col": 14, "code": "FILTER_OPTION_TYPE", "message": "filter 'subtitles' requires option 'filename'", "hint": "ffmpeg would refuse the filter at run time; write filename => <value>"}
```

The expression's own *content* is never checked here (or anywhere at compile time) — see [docs/filters.md](filters.md). The same goes for expressions in ordinary string-typed options (`scale(a.video[1], 'iw/2', -2)`, `overlay(a.video[1], b.video[1], '(W-w)/2', '(H-h)/2')`): the string is accepted as the option's value, and a typo inside the quotes surfaces when the command runs.

## UNKNOWN_INPUT_OPTION

**Meaning:** An `input('path', <name> => <value>, ...)` trailing named option names something outside `ffrwd.inputs.INPUT_OPTIONS`. This table is curated and fixed, exactly like `SINK_OPTIONS` -- there is no escape hatch to an arbitrary ffmpeg input flag.

**Fires when:** the option name is misspelled or simply isn't one of `loop`, `stream_loop`, `framerate`, `itsoffset`, `hwaccel`. Unlike a sink option name (folded lowercase from `WITH (name value)`), an input option name is the same `=>` named-argument syntax every call uses, so it is checked CASE-SENSITIVELY.

**Example query:**

```sql
SELECT p.video[1] FROM input('logo.png', loob => true) p
```

**Error JSON:**

```json
{"line": 1, "col": 27, "code": "UNKNOWN_INPUT_OPTION", "message": "unknown input option 'loob'", "hint": "did you mean 'loop'?"}
```

Anchoring: like a named argument's `exp.Var` name (`UNKNOWN_FILTER_OPTION`) and a `WITH (...)` option name (`UNKNOWN_SINK_OPTION`), sqlglot records no token position on the `Var` holding an `=>` name, so the anchor falls back to the option's VALUE -- except here the value is `true`, an `exp.Boolean`, which ALSO carries none, so it falls back one step further, to the `input()`'s own path string literal (`'logo.png'`), which is why `line`/`col` land there instead.

## INPUT_OPTION_TYPE

**Meaning:** An `input('path', <name> => <value>, ...)` option's value doesn't match the type declared for it in `ffrwd.inputs.INPUT_OPTIONS` (`str` / `int` / `bool` / `num`). `num` is new relative to the sink table's vocabulary: it accepts an `int` OR a `float`, and -- for `itsoffset` specifically -- a negative one (ffmpeg legitimately shifts a stream's timestamps earlier).

**Fires when:** a `bool` option (`loop`) gets anything but `true`/`false`; an `int` option (`stream_loop`) gets a float, string, or bool; a `num` option (`framerate`, `itsoffset`) gets a string or bool; a `str` option (`hwaccel`) gets anything but a single-quoted literal.

**Example query:**

```sql
SELECT p.video[1] FROM input('logo.png', framerate => 'fast') p
```

**Error JSON:**

```json
{"line": 1, "col": 55, "code": "INPUT_OPTION_TYPE", "message": "option 'framerate' expects a number, got 'fast'", "hint": "framerate takes a bare numeric literal, e.g. framerate => 15"}
```

## NOTHING_TO_SHOW

**Meaning:** `ffrwd run --show` or `--show-only` was asked for and the query has no video output file to play. A window shows a `COPY` that writes video; a bare `SELECT`, a `FORMAT csv` COPY, an audio-only output, a subtitle document and a module's rows have nothing to put in one. A query calling a `LANGUAGE wasm` module shows like any other: it runs as several processes, and the one writing the video file is the one the window reads.

**Fires when:** the flag is given for a table query; or for a media query whose every output file is audio-only, subtitles or rows.

This code and `PLAYER_NOT_FOUND` are the two no compile can raise -- they refuse a `run` flag, not a query -- so `ffrwd prompt` lists neither and the repair loop never sees one.

**Error JSON:**

```json
{"line": 1, "col": 1, "code": "NOTHING_TO_SHOW", "message": "nothing to show: the query writes no video output file", "hint": "a window plays a COPY that writes video; drop the flag to run the query as it stands"}
```

## PLAYER_NOT_FOUND

**Meaning:** `--show`/`--show-only` needs `ffplay` and it is not on PATH. Unlike ffmpeg and ffprobe there is no fallback: the `static-ffmpeg` provisioner ships that pair and no player, so an ffplay that is not on PATH is not anywhere. Raised only when a window was actually asked for -- a run without the flag never looks for one.

**Error JSON:**

```json
{"line": 1, "col": 1, "code": "PLAYER_NOT_FOUND", "message": "ffplay not found", "hint": "ffplay ships with ffmpeg but the static-ffmpeg provisioner does not supply it; install a full ffmpeg build and put ffplay on PATH, or drop the flag and let the run write its files"}
```

## UNBOUNDED_LIVE_INPUT

**Meaning:** An input that can only be opened once - a `scheme://` URL, or one whose `format =>` names a capture device or a graph - feeds a query that splits across processes, and the compiler cannot say how deep the buffers between those processes have to be. One process reads such an input and hands every other one a pipe (cookbook recipe 101). Where two of those pipes come back together, the pipe whose path is quicker holds frames until the slower one arrives, and the compiler counts how many. A stage between them that hands on a different number of frames than it reads makes that count meaningless: `fps`, `framerate`, `select`, `decimate`, `trim`, `reverse` and their audio counterparts, or a module whose describe says it is not one frame out per frame in.

**Fires when:** a one-open input feeds two paths that meet again, with such a stage on one of them. Never for a plain file: a file opens as many times as the query needs, so each path decodes it for itself and there is no buffer to size. Never for a one-open input read by a single process either - there is nothing to run ahead of.

**Example query:**

```sql
CREATE FUNCTION invert(v video_stream) RETURNS video_stream
  AS '../sidecar/modules/target/wasm32-wasip2/release/invert.wasm', 'invert'
  LANGUAGE wasm;

COPY (
  SELECT ffmpeg.overlay(ffmpeg.fps(a.video[1], fps => 15), invert(a.video[1]))
  FROM input('testsrc2=size=640x360:rate=30', format => 'lavfi', realtime => true) a
) TO 'out.mp4'
```

**Error JSON:**

```json
{"line": 7, "col": 14, "code": "UNBOUNDED_LIVE_INPUT", "message": "'testsrc2=size=640x360:rate=30' can only be opened once, so one process reads it and hands every other one a pipe -- and two of those paths come back together, with 'fps' between them handing on a different number of frames than it reads", "hint": "the buffer between two such paths is sized from how far one runs ahead of the other, and 'fps' makes that distance uncountable: apply it after the two paths come together rather than before, or record the input to a file and run the query over the file"}
```

The anchor is the `input()` path itself, since the input is what constrains the shape. The same code, with a different message, refuses a one-open input whose subtitle or data track a second process would have to read: a pipe between processes carries pictures and sound and nothing else.

## BUFFER_OVERFLOW

**Meaning:** Not a compile rejection - the one code a RUN produces. The buffers a plan sized from its bounds were not deep enough, and the pipeline wedged: nothing crossed any pipe of the stage while every process was still alive and one of them was still waiting to hand its bytes over. `ffrwd run` reports it instead of letting the stage sit until the timeout, so the message names the edge, the depth it was given, and how long nothing moved - never a bare "timed out", and never a silently dropped frame.

**Fires when:** the paths out of a one-open input's single reader drift further apart at run time than the compiler counted them - a filter slower than its declared shape, a consumer that cannot keep up with a live source, or a delay in a stage the compiler bounded at zero because it had nothing better to go on.

Like `NOTHING_TO_SHOW` and `PLAYER_NOT_FOUND` this is no compile's, so `ffrwd prompt` does not list it and the repair loop never sees one: no rewriting of the SQL is implied, though recording the input to a file and running the query over the file removes the pacing that caused it.

**Error text** (printed to stderr by `ffrwd run`, not as JSON):

```
error: BUFFER_OVERFLOW: the pipe buffer carrying 'src_a_v_0_split:1' from ffmpeg1 to ffmpeg0 overflowed: it was sized for the 2 frames the compiler bounded it at, and with every process still running nothing has crossed any pipe of this stage for 30s (hint: the paths out of the one process reading the input drifted further apart than the compiler counted them: record the input to a file and run this query over the file, or take the slower path's work out of the pipeline)
```

## STARTUP_DEADLOCK

**Meaning:** The plan's processes would each wait on the next before any of them could read or write, so the run could never reach its first frame. ffmpeg opens its inputs one at a time and writes its outputs interleaved, so a process blocked opening its first pipe is not draining the pipes it already opened - and the producer filling one of those stops before it reaches the output the blocked open is waiting for. The compiler chooses the order of every pipe to avoid that; this is the plan no order avoids it for.

**Fires when:** the compiler could find no order of a plan's pipes that starts. It is the check on the pass that picks that order, not a rejection any query is known to reach: for every plan shape ffrwd builds today an order exists and is found. What it guards against is a plan whose order is not the compiler's to pick - a track pinned to the head of an `-i` list, a module process whose own pipes are spelled for it - ending up in a cycle nothing would otherwise report until the run had stood still for thirty seconds.

**Error JSON:**

```json
{"line": 1, "col": 17, "code": "STARTUP_DEADLOCK", "message": "these processes cannot start, each waiting on the next: ffmpeg1 waits to write output 1, ffmpeg2 waits for every input it reads, ffmpeg2 waits to open input 1, and ffmpeg1 again", "hint": "each of those processes is waiting for the one after it before it can read or write anything: give the query one fewer place where the streams off a single input split apart and come back together, or record the input to a file and run the query over the file"}
```

The message walks the cycle in the order the processes wait, so the way out is visible from it: break the query where two of the named processes meet. The anchor is the module declaration that put a second process in the plan.

## INTERNAL

**Bug backstop, not a user-input error.** Every compiler pass (`parse`, `lower`, `insert_splits`, via `compile_sql`) wraps its body in a catch-all that converts any unexpected exception (a sqlglot internal, a `RecursionError` on a pathologically nested query, or an actual bug in ffrwd) into `ErrorCode.INTERNAL` rather than letting a raw traceback escape (guardrail #7: no panics on user input, ever). The fuzz corpus in `tests/test_fuzz.py` asserts this code never fires across its mutated queries. If you see `INTERNAL` in the wild, ffrwd has a bug, and we would genuinely like the query that triggered it. No example JSON here, because no known SQL input reaches this path, and we intend to keep it that way.
