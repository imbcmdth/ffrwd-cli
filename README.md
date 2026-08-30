<p align="center">
  <img src="ffrwd.png" alt="ffrwd" width="500">
</p>

> [ffrwd.video](https://ffrwd.video) — /frood/
> 1. Welsh for stream
> 2. Someone who really knows where their towel is

**SQL in; ffmpeg out**

You write a `SELECT` statement, ffrwd compiles it into a `-filter_complex` invocation, and ffmpeg does the actual pixel-pushing. This tool never decodes a single frame: it's a compiler, and ffmpeg is the executor.

## Why does this exist?

Look, ffmpeg is a marvel - one that I've used for over a decade - but I still need to lookup the syntax *every single time* I want to do something non-trivial. And AI barely helps - it often just gives you the same wrong answer from Reddit that you could have Googled yourself.

 SQL, meanwhile, has been describing dataflow DAGs for fifty years, and it's the language every developer (and every LLM) already speaks.

 This project connects the two so you can bring your preexisting knowledge and create **declarative**, **composable**, and **generalizable** ffmpeg formulae.

 The dialect is deliberately two-tongued: the query surface is Postgres - `COPY`, dollar-quoted functions, psql variables - while the data model is BigQuery's structs and arrays, because a media file *is* an array of structured things. If you speak either, you already speak most of it.

## Install

Python 3.10+.

```bash
pip install ffrwd
```

Or run it without installing anything:
```bash
uvx ffrwd
```

`ffmpeg` and `ffprobe` are required and, optionally, handled for you. A preexisting ffmpeg install on `PATH` always wins. On a machine without one, the bundled provisioner (`static-ffmpeg`) fetches both binaries on first use.

## Ask before you act

`run` is the default subcommand, so a query is the whole invocation - and a query with no `COPY ... TO` is a **metadata query**: the answer is probed metadata, fully known the moment compilation ends, so ffrwd prints it as a table and never runs `ffmpeg` at all. This works on anything `ffprobe` can read, remote manifests included:

```bash
$ ffrwd "SELECT t.index, t.tags.language, t.codec \
FROM input(:'src') f, unnest(f.audio) t \
WHERE t.codec = 'aac'" \
-v src=https://storage.googleapis.com/shaka-demo-assets/angel-one/dash.mpd
 index | language | codec
-------+----------+-------
 1     | es       | aac
 2     | de       | aac
 3     | en       | aac
 7     | fr       | aac
 10    | it       | aac
(5 rows)
```

`COPY ... TO STDOUT WITH (format 'csv')` is the scriptable spelling of the same thing - stock Postgres COPY, `header true` optional, or `TO 'tracks.csv'` to write a file:

```bash
$ ffrwd "COPY ( \
  SELECT t.tags.language, t.codec \
  FROM input(:'src') f, unnest(f.audio) t \
  WHERE t.codec = 'aac' \
)TO STDOUT WITH (format 'csv', header true)" \
-v src=https://storage.googleapis.com/shaka-demo-assets/angel-one/dash.mpd
language,codec
es,aac
de,aac
en,aac
fr,aac
it,aac
```

Media only moves when you ask for a file: using the same `COPY ... TO 'out.mkv'` inside the query.

## PiP demo

Imagine you wanted to shrink `commentary.mkv` into the corner of `film.mkv`, and duck the commentary under the main mix. And both files carry two audio tracks - an English and a French language.

```sql
COPY(
WITH pip AS (
  SELECT scale(c.video[1], 'iw/4', -2) AS frame, c.audio AS sound
  FROM input('commentary.mkv') c
)
SELECT overlay(f.video[1], pip.frame, 20, 20),
       amix(volume(f.audio, 0.65), volume(pip.sound, 0.35))
FROM input('film.mkv') f, pip
) TO ('pip.mkv')
```

```bash
$ ffrwd compile -f query.sql
ffmpeg -i commentary.mkv -i film.mkv -filter_complex '
  [0:v:0]scale=width=iw/4:height=-2[n1];
  [1:v:0][n1]overlay=x=20:y=20[out0];
  [1:a:0]volume=volume=0.65[n3];
  [1:a:1]volume=volume=0.65[n4];
  [0:a:0]volume=volume=0.35[n5];
  [0:a:1]volume=volume=0.35[n6];
  [n3][n5]amix=inputs=2[out1];
  [n4][n6]amix=inputs=2[out2]' \
  -map '[out0]' -map '[out1]' -metadata:s:1 language=eng -map '[out2]' -metadata:s:2 \
  language=fra pip.mkv
```

Check out all that work you didn't need to do! No pad labels or bookkeeping. You never even said how many audio tracks there were: `c.audio` is the whole array, `volume` broadcasts over it (one node per language), and `amix` zips the two arrays elementwise, English with English, French with French. Each mixed track keeps its language tag, because both parents agreed. (`compile` shows the command; drop it - `ffrwd -f query.sql` - and the default `run` executes it instead.)

## Encoding

The query above describes the edit and says nothing about codecs, so ffmpeg picks its defaults. When you care about the encode, wrap the query in `COPY ... TO ... WITH (...)` - stock Postgres syntax - and the destination and codec settings ride along inside the query:

```sql
COPY (
  WITH pip AS (
    SELECT scale(c.video[1], 'iw/4', -2) AS frame, c.audio AS sound
    FROM input('commentary.mkv') c
  )
  SELECT overlay(f.video[1], pip.frame, 20, 20),
         amix(volume(f.audio, 0.65), volume(pip.sound, 0.35))
  FROM input('film.mkv') f, pip
) TO 'pip.mkv' WITH (
  video_codec 'libx264', crf 20, audio_codec 'aac', audio_bitrate '192k'
)
```

```bash
$ ffrwd compile -f query.sql
ffmpeg -i commentary.mkv -i film.mkv -filter_complex '
  [0:v:0]scale=width=iw/4:height=-2[n1];
  [1:v:0][n1]overlay=x=20:y=20[out0];
  [1:a:0]volume=volume=0.65[n3];
  [1:a:1]volume=volume=0.65[n4];
  [0:a:0]volume=volume=0.35[n5];
  [0:a:1]volume=volume=0.35[n6];
  [n3][n5]amix=inputs=2[out1];
  [n4][n6]amix=inputs=2[out2]' \
  -map '[out0]' -map '[out1]' -metadata:s:1 language=eng -map '[out2]' -metadata:s:2 \
  language=fra -c:0 libx264 -crf:0 20 -c:1 aac -c:2 aac -b:1 192k -b:2 192k pip.mkv
```

## Views and multiple outputs

A `CREATE VIEW name AS <query>;` followed by one or more `COPY (...) TO '<path>' WITH (...);` is a script - the ABR-ladder shape, one decode feeding several encodes. It still compiles to ONE ffmpeg invocation, one output group per COPY:

```sql
CREATE VIEW main AS
  SELECT scale(f.video[1], 1920, -2) AS v, volume(f.audio[1], 0.9) AS a
  FROM input('film.mkv') f;

COPY (SELECT scale(m.v, 1280, -2) AS v, m.a FROM main m) TO '720.mp4'
WITH (video_codec 'libx264', crf 21, audio_codec 'aac');

COPY (SELECT scale(m.v, 640, -2) AS v, m.a FROM main m) TO '360.mp4'
WITH (video_codec 'libx264', crf 26, audio_codec 'aac');

COPY (SELECT m.a FROM main m) TO 'audio.m4a'
WITH (audio_codec 'aac', audio_bitrate '128k')
```

```bash
$ ffrwd compile -f query.sql
ffmpeg -i film.mkv -filter_complex '
  [0:v:0]scale=width=1920:height=-2[n1];
  [0:a:0]volume=volume=0.9[n2];
  [n1]split=2[n1_split0][n1_split1];
  [n1_split0]scale=width=1280:height=-2[out0];
  [n1_split1]scale=width=640:height=-2[out2];
  [n2]asplit=3[out1][out3][out4]' \
  -map '[out0]' -map '[out1]' -c:0 libx264 -crf:0 21 -c:1 aac 720.mp4 \
  -map '[out2]' -map '[out3]' -c:0 libx264 -crf:0 26 -c:1 aac 360.mp4 \
  -map '[out4]' -c:0 aac -b:0 128k audio.m4a
```

A view is to statements what a CTE is to branches: `main` decodes and filters `film.mkv` exactly once - `scale` and `volume` each appear a single time in the graph above - and the split pass hands out however many pads its readers need (`split=2` for the two video consumers, `asplit=3` for the three audio ones).

There's much more - watermarks, GIFs, subtitle muxing, multiband compression, generated test media - and it all lives in the **[cookbook](docs/examples.md)**: nearly sixty real tasks, simple to complex, every shown output rerun and byte-checked by the test suite - and most of them parameterized with `-v` variables, so they run against your files as-is.

## CLI reference

```
ffrwd <command> [-h] [-f FILE] [-v NAME=VALUE] [--timeout TIMEOUT] [-y] [query]
```

**Watching a run.** `run --show` writes the files as usual and, for each one
that carries video, opens an ffplay window on it: the terminal ffmpeg gains a
second output carrying the same streams as raw NUT on its stdout, and that
pipe feeds one player per shown file. `--show-only` shows the same windows and
writes nothing at all - the file outputs are suppressed, so no encoder runs
and nothing is overwritten. The query itself is unchanged either way: `COPY`
still spells its destination, and the flag decides at run time whether that
destination is written, which is what lets one query serve both testing and
production. The two flags are mutually exclusive. Under `--show`, closing a
window does not end the run - the remaining work finishes and the files still
land. Under `--show-only` the windows are all there is, so closing the last
one ends the run, which is how you stop watching a camera. A query
with no video output file is refused
rather than shown; `--show` needs `ffplay` on PATH, which the bundled ffmpeg
provisioner does not supply.

`run` is the default subcommand: any invocation that doesn't start with a subcommand name is `run`'s, so `ffrwd "SELECT ..."` and `ffrwd -f query.sql` just work. All four query commands take the SQL as text right on the command line, or from a file with `-f query.sql` (`-f -` reads stdin). Exactly one of the two. All four also take `-v name=value` (repeatable - psql's flag, psql's syntax): `:'name'` in the query becomes the value as an escaped string literal, bare `:name` becomes it raw, and a variable you leave unset is `NULL`, which means absence: an option it fills is simply not written, and where a value is required the error names the variable. A query file plus `-v` is a reusable recipe; the registry at ffrwd.video collects ready-made ones as installable packages.

| command | what it does | flags |
|---|---|---|
| `run` | **the default** (`ffrwd "SELECT ..."` is enough): a query with a media destination compiles and executes ffmpeg; one without prints its result set as a table, psql-style, executing nothing | `--timeout SECS` (default: ten times the longest input's duration, at least 600, and no timeout at all when an input is live) · `-y` (overwrite) · `--show` · `--show-only` |
| `compile` | print the full ffmpeg command | `--graph-only` (just the filtergraph string) |
| `explain` | dump the compiled IR graph as JSON | |
| `validate` | exit 0 if the query compiles, else a line-anchored error | `--json` (machine-readable error object on stdout) |
| `prompt` | print the LLM system prompt | |
| `mcp` | serve the compiler to an editor or agent over MCP (stdio) | `--allow-unsafe` (also expose the tools that do more than answer about a query) |

## The ideas, briefly

- **Streams are columns.** Every input exposes `<alias>.video`, `<alias>.audio`, `<alias>.subtitle`, `<alias>.data` (1-based subscripts), and **the SELECT list is the result set** - in a media query (one with a `COPY` destination), one column is one `-map`, in order, nothing implicit. A bare subscript no function touches stays a stream copy. `input()` takes per-input options (`loop => true` keeps a still image alive). `SELECT *` keeps everything.
- **Bare arrays broadcast.** `atempo(v.audio, 1.25)` fans out one node per track, each output keeping its language tag. Two arrays in one call zip elementwise.
- **Tracks are rows when you need them.** `unnest(f.audio)` turns a track array into a compile-time table whose columns are the probed metadata, so picking a track is `WHERE t.tags.language = 'eng'` and aligning two files' tracks is a real SQL `JOIN` - inner, left, or full outer, with generated silence (or an empty caption track) standing in for what a file lacks. Selecting a `tags` map next to a track *edits its tags* (`STRUCT(CASE ... END AS language) AS tags` retags a whole library in one expression), and `chapters` is just another array column - `unnest(f.chapters)` reads them, an `ARRAY[STRUCT(...)::chapter, ...]` literal writes them. Rows combine only when written: `array_agg` gathers them into one file, a `TO (expression)` writes one file per row, and a multi-row query into a single path is a compile error - what the table preview shows is exactly what a COPY serializes. Every join is decided at compile time; ffmpeg only sees the wiring. [docs/rows.md](docs/rows.md) has the whole story.
- **A SELECT with no COPY prints a table.** The result set was fully known at compile time, so `ffrwd "SELECT t.* FROM input('film.mkv') f, unnest(f.audio) t"` prints the tracks as rows - ffprobe you can read, joins included - and `COPY (...) TO STDOUT WITH (FORMAT csv)` makes it scriptable. ffmpeg only runs when a `COPY` names a media destination.
- **Trims are seeks.** `WHERE a.t BETWEEN 5 AND 60` (or either bound alone, open-ended) becomes `-ss`/`-to` on that alias's `-i`: fast, all stream types at once, stream-copy still possible. Decoded streams cut frame-accurate; copied ones snap to a keyframe. The measurements, and the caption caveat, are in [docs/trimming.md](docs/trimming.md).
- **Every filter, one convention.** All ~450 filters in your ffmpeg build are callable: streams first, then options - positionally in the exact order `ffmpeg -help filter=<name>` prints them, by name (`unsharp(a.video[1], luma_amount => 1.5)`), or both. Every option is type-checked against what the binary reports. `ffmpeg.<name>(...)` always means the raw filter, including the eleven names Postgres grammar would otherwise eat; `ffrwd.<name>(...)` holds exactly four macros for jobs no single filter does (`delay`, `speed`, `blur_regions`, and `loudnorm2`, which measures a stream's loudness and corrects it in a second pass). A few multi-output filters (`channelsplit`, `acrossover`, `extractplanes`) return arrays. [docs/filters.md](docs/filters.md) has the whole story.
- **Generated sources live in FROM.** `ffmpeg.sine(frequency => 440, duration => 1) s` is a table function, not a file - the compiled command has no `-i` at all.
- **`enable` and expressions.** `gblur(a.video[1], 12, enable => 'between(t,10,20)')` windows an effect in time; expression strings like `'(W-w)/2'` do per-frame geometry in any string-typed option.
- **Captions ride along, untouched.** Subtitle and data streams select, extract and mux like anything else, but they're passthrough-only - a filtergraph has no subtitle pads.
- **Errors are a feature.** Every rejection is a typed, line-anchored JSON object with a hint, documented with captured examples in [docs/errors.md](docs/errors.md).

## Use with an AI

ffrwd ships the system prompt. Bring whatever model you like.

```bash
$ ffrwd prompt > system.txt      # the dialect, the calling convention, your filters
```

Pipe that in as the system prompt, ask for the edit in English, and put the reply through the validator:

```bash
$ ffrwd validate --json -f query.sql
{"line": 1, "col": 8, "code": "UDF_ARG_TYPE", "message": "...", "hint": "..."}
```

The prompt's filter reference is rendered from the same registry the compiler resolves against - your installed ffmpeg - so it cannot drift, and the model works with your actual machine rather than a platonic ideal of one.

An editor or agent that speaks MCP can have the same loop without the pipes:

```bash
$ pip install "ffrwd[mcp]"
$ ffrwd mcp                      # a stdio MCP server; add --allow-unsafe to let it run ffmpeg
```

It serves the prompt as a resource and five tools: `compile`, `validate` (empty when the query is good, the typed error object when it isn't - the repair loop), `explain`, `inspect` for a file's tracks and chapters, and `filters` for what your ffmpeg actually has. `run` is the sixth, off unless you pass `--allow-unsafe`: everything else returns text about a query, and that one writes files.

## Layout

- `cli/` - the compiler and the `ffrwd` command, a Python package.
- `sidecar/` - the wasm host, a Rust workspace.
- `docs/` - the reference and the cookbook, shared by both.

---

More docs: [dialect](docs/dialect.md) · [types](docs/types.md) · [cookbook](docs/examples.md) · [filters](docs/filters.md) · [row shapes](docs/rows.md) · [trimming](docs/trimming.md) · [error contract](docs/errors.md) · [known gaps](docs/known_gaps.md)
