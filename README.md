<p align="center">
  <img src="https://raw.githubusercontent.com/imbcmdth/ffrwd-cli/main/ffrwd-logo.svg" alt="ffrwd" width="468">
</p>

> [ffrwd.video](https://ffrwd.video) — /frood/
> 1. Welsh for stream
> 2. Someone who really knows where their towel is

**SQL in; ffmpeg out**

You write a `SELECT` statement, ffrwd compiles it into a `-filter_complex` invocation, and ffmpeg does the actual pixel-pushing. The compiler never decodes a frame itself. When a query calls something ffmpeg has no filter for - a detector, a transcriber, a live relay - a small Rust sidecar hosts that as a wasm module beside ffmpeg, and the query reads the same either way.

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

`ffmpeg` and `ffprobe` are required and, optionally, handled for you. A preexisting ffmpeg install on `PATH` always wins. On a machine without one, the bundled provisioner (`static-ffmpeg`) fetches both binaries on first use. The sidecar ships as a wheel pinned to the same version, so a query that calls a module needs nothing more; a module that runs a model needs an ONNX Runtime, which `ffrwd setup nn` downloads once (`--cuda` for the GPU provider).

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

A rendition ladder is rows, too: an HLS or DASH manifest reads as one row per rung with its `height`, `bandwidth` and `codecs`, `WHERE` picks the rungs, and a `format 'hls'` destination fed every surviving row writes a new ladder, rung for rung.

## Packages

A query can call more than ffmpeg. A package is a directory with an `ffrwd.json`: SQL functions, wasm modules, the models they load, and recipes - query files with a `-- variables:` header. Install one and call it by namespace:

```bash
$ ffrwd install ffrwd/describe
$ ffrwd run ffrwd/describe:describe -v src=film.mkv -v dest=film.described.mkv
```

The first line pulls the package and what it depends on - here the shot detector, the voice detector and whisper - and their pinned weights; the second runs a recipe the package ships, by name, with its variables filled from `-v`. Inside a query the same package is `ffrwd.describe.<function>(...)`, three segments, and its functions compose with everything else: a detector's boxes drawn over the picture is one `overlay` with a node the sidecar hosts.

What a module writes beside the picture - cues, embeddings, boxes - lands in the file as tracks, and reads back as rows: `unnest(f.cues['speech']) c`, `unnest(f.embeddings) v`. Rank them with `cos_similarity`, keep the top few with `ORDER BY ... LIMIT`, collapse touching spans with `merge_cues`, and cut the film to what matched - the whole search is compile-time SQL, and ffmpeg only ever sees the trims. [docs/rows.md](https://github.com/imbcmdth/ffrwd-cli/blob/main/docs/rows.md) has the row shapes; `ffrwd search` and [ffrwd.video](https://ffrwd.video) have the packages.

Writing one is `ffrwd init`, `CREATE FUNCTION ... LANGUAGE wasm` pointing at a module built against the published `ffrwd/wasm` world, and `ffrwd publish`. The [dialect reference](https://github.com/imbcmdth/ffrwd-cli/blob/main/docs/dialect.md#projects-and-packages) covers the manifest, the lockfile, linking a checkout during development, and the registry.

There's much more - watermarks, GIFs, subtitle muxing, multiband compression, generated test media, live sources - and it all lives in the **[cookbook](https://github.com/imbcmdth/ffrwd-cli/blob/main/docs/examples.md)**: thirty real tasks, simple to complex, with [over a hundred more](https://github.com/imbcmdth/ffrwd-cli/blob/main/docs/corpus.md) beside them. Every shown output is rerun and byte-checked by the test suite, and most are parameterized with `-v` variables, so they run against your files as-is.

## CLI reference

```
ffrwd [<command>] [-h] [-f FILE] [-v NAME=VALUE] [-q] [query | recipe]
```

`run` is the default subcommand: any invocation that doesn't start with a subcommand name is `run`'s, so `ffrwd "SELECT ..."` and `ffrwd -f query.sql` just work. The four query commands take the SQL as text right on the command line, or from a file with `-f query.sql` (`-f -` reads stdin), or the name of a recipe an installed package ships - bare when it is unambiguous, `ns/pkg:recipe` when it is not. They all take `-v name=value` (repeatable - psql's flag, psql's syntax): `:'name'` in the query becomes the value as an escaped string literal, bare `:name` becomes it raw, and a variable you leave unset is `NULL`, which means absence: an option it fills is simply not written, and where a value is required the error names the variable. `-q` prints only the result - no narration, no spinner, no progress bar.

| command | what it does | flags |
|---|---|---|
| `run` | **the default**: a query with a media destination compiles and executes it; one without prints its result set as a table, psql-style, executing nothing | `--timeout SECS` (default: ten times the longest input's duration, at least 600, none when an input is live) · `-y` (overwrite) · `--jobs N` (cap the sidecar's threads) · `--show` / `--show-only` (below) · `--remote` (below) |
| `compile` | print the full ffmpeg command | `--graph-only` (just the filtergraph string) |
| `explain` | dump the compiled graph as JSON | `--mermaid` (as a flowchart) · `--diagram` (rendered in the terminal; needs `ffrwd[diagram]`) |
| `validate` | exit 0 if the query compiles, else a line-anchored error | `--json` (machine-readable error object on stdout) |
| `list` | every package this project and the machine hold, with each one's functions and recipes | `--json` |
| `search` | ask the registry for packages matching a term | |
| `install` | install a package into this project, or with `-g` machine-wide; with no argument, install what the project's `ffrwd.json` declares | `-g` |
| `init` | start a package here: `ffrwd.json`, an empty lockfile, a starter recipe | `--name` · `--namespace` |
| `path` | where an installed package sits on disk | |
| `link` / `unlink` | develop a package from its checkout: the project resolves it from there until unlinked | |
| `login` / `logout` | hold, or forget, a registry token on this machine | |
| `publish` | run the package's tests, pack it, and publish it to the registry | |
| `setup nn` | download the ONNX Runtime the model-running modules need | `--cuda` · `--full` |
| `jobs` | list, watch, cancel or fetch your remote runs | `--watch` · `--cancel ID` · `--fetch ID` · `-y` · `--json` |
| `prompt` | print the LLM system prompt | |
| `mcp` | serve the compiler to an editor or agent over MCP (stdio) | `--allow-unsafe` (also expose the tools that do more than answer about a query) |

**Watching a run.** `run --show` writes the files as usual and, for each one that carries video, opens an ffplay window on it: the terminal ffmpeg gains a second output carrying the same streams as raw NUT on its stdout, and that pipe feeds one player per shown file. `--show-only` shows the same windows and writes nothing at all - the file outputs are suppressed, so no encoder runs and nothing is overwritten. The query itself is unchanged either way: `COPY` still spells its destination, and the flag decides at run time whether that destination is written, which is what lets one query serve both testing and production. Under `--show`, closing a window does not end the run; under `--show-only` closing the last window ends it, which is how you stop watching a camera. `--show` needs `ffplay` on PATH, which the bundled provisioner does not supply.

**Running elsewhere.** `run --remote` submits the query to the hosted runner instead of this machine: local inputs upload with a progress bar, the job runs on a machine with the models and a GPU, and `ffrwd jobs --fetch <id>` brings the outputs back to the paths the query named. `--wait` blocks until it finishes; `--json` with it prints the finished job as JSON. The runner takes only sources its probe can bound - files, URLs to files, manifests that have ended - so a live source is refused up front, and it needs a run token from [ffrwd.video](https://ffrwd.video).

## The ideas, briefly

- **Streams are columns.** Every input exposes `<alias>.video`, `<alias>.audio`, `<alias>.subtitle`, `<alias>.data` (1-based subscripts), and **the SELECT list is the result set** - in a media query (one with a `COPY` destination), one column is one `-map`, in order, nothing implicit. A bare subscript no function touches stays a stream copy. `input()` takes per-input options (`loop => true` keeps a still image alive, `realtime => true` paces a file like a feed). `SELECT *` keeps everything.
- **Bare arrays broadcast.** `atempo(v.audio, 1.25)` fans out one node per track, each output keeping its language tag. Two arrays in one call zip elementwise.
- **Tracks are rows when you need them.** `unnest(f.audio)` turns a track array into a compile-time table whose columns are the probed metadata, so picking a track is `WHERE t.tags.language = 'eng'` and aligning two files' tracks is a real SQL `JOIN` - inner, left, or full outer, with generated silence (or an empty caption track) standing in for what a file lacks. Selecting a `tags` map next to a track *edits its tags*, and `chapters` is just another array column - `unnest(f.chapters)` reads them, an `ARRAY[STRUCT(...)::chapter, ...]` literal writes them. Rows combine only when written: `array_agg` gathers them into one file, `UNION ALL` stitches branches back to back and a branch that matched nothing contributes nothing, a `TO (expression)` writes one file per row, and a multi-row query into a single path is a compile error. A CTE's rows keep their value columns, so a query can rank inside one and re-sort outside it. Every join is decided at compile time; ffmpeg only sees the wiring. [docs/rows.md](https://github.com/imbcmdth/ffrwd-cli/blob/main/docs/rows.md) has the whole story.
- **A SELECT with no COPY prints a table.** The result set was fully known at compile time, so `ffrwd "SELECT t.* FROM input('film.mkv') f, unnest(f.audio) t"` prints the tracks as rows - ffprobe you can read, joins included - and `COPY (...) TO STDOUT WITH (FORMAT csv)` makes it scriptable. ffmpeg only runs when a `COPY` names a media destination.
- **Trims are seeks.** `WHERE a.t BETWEEN 5 AND 60` (or either bound alone, open-ended) becomes `-ss`/`-to` on that alias's `-i`: fast, all stream types at once, stream-copy still possible. Decoded streams cut frame-accurate; copied ones snap to a keyframe. The measurements, and the caption caveat, are in [docs/trimming.md](https://github.com/imbcmdth/ffrwd-cli/blob/main/docs/trimming.md).
- **Every filter, one convention.** All ~450 filters in your ffmpeg build are callable: streams first, then options - positionally in the exact order `ffmpeg -help filter=<name>` prints them, by name (`unsharp(a.video[1], luma_amount => 1.5)`), or both. Every option is type-checked against what the binary reports. `ffmpeg.<name>(...)` always means the raw filter, including the eleven names Postgres grammar would otherwise eat; `ffrwd.<name>(...)` holds exactly four macros for jobs no single filter does (`delay`, `speed`, `blur_regions`, and `loudnorm2`, which measures a stream's loudness and corrects it in a second pass); three segments, `ffrwd.<package>.<name>(...)`, is a package. A few multi-output filters (`channelsplit`, `acrossover`, `extractplanes`) return arrays. [docs/filters.md](https://github.com/imbcmdth/ffrwd-cli/blob/main/docs/filters.md) has the whole story.
- **Modules are functions.** `CREATE FUNCTION detect(v video_stream) RETURNS STRUCT(v video_stream, boxes STRUCT(...)[]) AS 'detect.wasm', 'detect' LANGUAGE wasm` declares a wasm module the sidecar runs on the frames, returning the picture and a column of rows beside it. A module can return a `source` (a FROM relation - a MoQ subscription, say) or a `sink` (a COPY destination - a MoQ publish), and the query treats both like a file.
- **Live is not special.** An `rtmp://`, `srt://` or `udp://` input, a device, a manifest still being written or a module source is a live relation: the query is the same, the compiler counts what it can and paces the rest, and a filter with no bound on its input runs for as long as the feed does. The [FROM items](https://github.com/imbcmdth/ffrwd-cli/blob/main/docs/dialect.md#from-items) section says what changes.
- **Generated sources live in FROM.** `ffmpeg.sine(frequency => 440, duration => 1) s` is a table function, not a file - the compiled command has no `-i` at all.
- **`enable` and expressions.** `gblur(a.video[1], 12, enable => 'between(t,10,20)')` windows an effect in time; expression strings like `'(W-w)/2'` do per-frame geometry in any string-typed option.
- **Captions ride along.** Subtitle and data streams select, extract and mux like anything else; a filtergraph has no subtitle pads, so a module that writes cues is how captions get made.
- **Errors are a feature.** Every rejection is a typed, line-anchored JSON object with a hint, documented with captured examples in [docs/errors.md](https://github.com/imbcmdth/ffrwd-cli/blob/main/docs/errors.md). A run that wedges is named too - which pipe, which process, what it waited for - never a bare timeout.

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
- `sidecar/` - the wasm host, a Rust workspace, published as the `ffrwd-wasm` wheel.
- `docs/` - the reference and the cookbook, shared by both.

---

More docs: [dialect](https://github.com/imbcmdth/ffrwd-cli/blob/main/docs/dialect.md) · [types](https://github.com/imbcmdth/ffrwd-cli/blob/main/docs/types.md) · [cookbook](https://github.com/imbcmdth/ffrwd-cli/blob/main/docs/examples.md) · [filters](https://github.com/imbcmdth/ffrwd-cli/blob/main/docs/filters.md) · [row shapes](https://github.com/imbcmdth/ffrwd-cli/blob/main/docs/rows.md) · [trimming](https://github.com/imbcmdth/ffrwd-cli/blob/main/docs/trimming.md) · [error contract](https://github.com/imbcmdth/ffrwd-cli/blob/main/docs/errors.md) · [known gaps](https://github.com/imbcmdth/ffrwd-cli/blob/main/docs/known_gaps.md)
