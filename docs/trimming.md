# Trimming

`WHERE <alias>.t BETWEEN <start> AND <end>` on an `input()` alias lowers to an input seek - `-ss <start> -to <end>` before that alias's `-i`. No filtergraph node:

```sql
SELECT a.video[1]
FROM input('clip.mp4') a
WHERE a.t BETWEEN 5 AND 60
```

```
$ ffrwd compile "SELECT a.video[1] FROM input('clip.mp4') a WHERE a.t BETWEEN 5 AND 60"
ffmpeg -ss 5 -to 60 -i clip.mp4 -map 0:v:0 -c:0 copy out.mp4
```

One window per alias (one lower bound, one upper, in any spelling). The seek trims and rebases every stream of the input at once, the demuxer skips the front of the file, and unfiltered columns stay stream-copies.

A window on a **CTE** lowers to `trim`+`setpts` / `atrim`+`asetpts` instead (a CTE's output is a pad, not an `-i`). Video and audio only.

## Open-ended windows

`<alias>.t >= <start>` seeks with no `-to`; `<alias>.t <= <end>` stops with no `-ss`. Operand order is free (`120 <= a.t` ≡ `a.t >= 120`). A `>=` and a `<=` on the same alias merge into one seek, identical to `BETWEEN`:

```sql
SELECT a.video[1]
FROM input('clip.mp4') a
WHERE a.t >= 120
```

```
$ ffrwd compile "SELECT a.video[1] FROM input('clip.mp4') a WHERE a.t >= 120"
ffmpeg -ss 120 -i clip.mp4 -map 0:v:0 -c:0 copy out.mp4
```

Rejected:

- **Strict `<` / `>`** - no frame-level meaning at seek granularity; `UNSUPPORTED_SQL` with a hint pointing at `>=`/`<=`.
- **Empty or doubled windows** - start not strictly before end, or a second bound of the same kind for one alias: compile-time `UNSUPPORTED_SQL`.

On a CTE, open windows pass only the bounds they have (`trim=start=3` with no `end=`).

## Row-bounded windows: one seek per row

A bound may read a row column - `WHERE f.t BETWEEN c.start_t AND c.end_t`, `WHERE f.t >= i.i - 1` - which makes the window one seek per surviving row. Where those seeks go is the query's business, and there are two answers:

- **A fan-out `TO (expression)`** gives each row a file of its own; the windows are per-FILE, described below.
- **An aggregate** gathers the rows into one file. Each row then mints its own `-i` of the same path with its own `-ss`/`-to`, all in ONE graph, and the per-row streams flow into the aggregate: `ffmpeg.concat(VARIADIC array_agg(shots.frame))` over N series rows is N seeks of one file joined end to end. Two rows naming the same window share one `-i`. [Recipe 75](examples.md#75-gather-n-clips-from-one-file-into-one).

Neither, and N windows reach a single destination: a `ROW_COUNT_MISMATCH` naming both ways out. The alias being windowed has to be an `input()` one - a CTE name is a filtergraph pad, with no `-i` to repeat.

Each window is its own input, so its timestamps start at zero; nothing is re-based afterwards.

## Fan-out windows: one seek per output file

A fan-out `TO (expression)` whose rows carry a window writes one file per row, and where it can, all of them in one ffmpeg command: each output takes its own `-ss <start> -to <end>` ahead of that output's `-map` list, so the input is read and decoded once no matter how many pieces come out. The cuts are frame-accurate.

Seeking an output re-encodes it. A stream that would have been a plain copy therefore takes whatever codec the sink names, or ffmpeg's default encoder for the container when it names none.

When every mapped stream in every output is a stream copy, that form is unavailable - ffmpeg writes corrupt files from an output seek plus `-c copy`. Such a fan-out compiles to one command per file instead, `&&`-chained, each seeking its own `-i`: fast, nothing decodes, cuts snapping to keyframes. [Recipe 47](examples.md#47-split-a-file-by-its-chapters) shows both forms of one query.

## Accuracy: decoded vs. stream-copied

- **Decoded** (filtered or re-encoded): frame-accurate - ffmpeg decodes from the previous keyframe and discards up to the requested point.
- **Stream-copied**: the cut snaps back to the previous keyframe (a copied stream can only begin at one). The output may start up to one GOP early.

Measured against `tests/fixtures/testsrc.mp4` (30 frames, 15 fps, 2.000s, one keyframe at t=0 - the worst case), `WHERE a.t BETWEEN 0.5 AND 1.5`:

| path | `ffmpeg` | measured output duration |
|---|---|---|
| stream-copied (`SELECT a.video[1]`, nothing filters it) | `-ss 0.5 -to 1.5 -i ... -c copy` | **1.367s** - snapped to the file's only keyframe at t=0 |
| decoded (wrapped in any filter, e.g. `hflip(a.video[1])`) | `-ss 0.5 -to 1.5 -i ... -c:v libx264 ...` | **1.000s** exactly |

If the exact cut point matters, wrap the column in a filter and accept the re-encode. Fast, frame-accurate copy-trims do not exist in any tool without a re-encode somewhere.

### mkv durations: check packets, not the summary

After a copy-trim to Matroska, container and per-track durations are muxer bookkeeping, not recomputed. Measured on a copy-trim of `tests/fixtures/avs.mkv`, nominal 0.6s window (`WHERE a.t BETWEEN 0.3 AND 0.9`):

```
$ ffprobe -show_entries format=duration output.mkv
duration=1.323000
$ ffprobe -show_entries stream_tags=DURATION output.mkv
video:    DURATION=00:00:01.156000000
audio:    DURATION=00:00:00.905000000
subtitle: DURATION=00:00:01.323000000
```

`format=duration` is just the largest track tag. For the real extent, read packet timestamps: `ffprobe -show_entries packet=pts_time`.

## Captions: trim + selected subtitles is rejected

ffmpeg does not retime caption packets under an input `-ss`, on either the copy or the transcode path. Measured on `tests/fixtures/avs.mkv` (cues at 0.023 / 0.723 / 1.423s), seeking `WHERE a.t BETWEEN 0.5 AND 1.5`:

| path | subtitle packet times in the output |
|---|---|
| stream-copied (everything `-c copy`) | 0.023 / 0.723 / 1.423 - unchanged |
| transcoded (video/audio re-encoded) | 0.000 / 0.700 / 1.400 - original spacing, not shifted |

Every cue plays early by the seek offset, so trimming an alias and selecting its subtitle/data column in the same query is a typed rejection. Any window shape triggers it - a tail-only `>=` desyncs identically:

```sql
SELECT a.subtitle[1]
FROM input('tests/fixtures/avs.mkv') a
WHERE a.t BETWEEN 1 AND 2
```

```
$ ffrwd validate --json "SELECT a.subtitle[1] FROM input('tests/fixtures/avs.mkv') a WHERE a.t BETWEEN 1 AND 2"
{"line": 1, "col": 8, "code": "UNSUPPORTED_SQL", "message": "'WHERE a.t' cannot trim a selected subtitle stream: ffmpeg does not retime caption packets under an input seek, so they would play out of sync with the trimmed video", "hint": "trim the video/audio without selecting the subtitle/data columns, or select them in a query without a WHERE time range; to caption a trimmed clip, join an external subtitle file whose cues are timed for the cut"}
```

Still legal: trimming an input whose captions are NOT selected (the unmapped stream never reaches the output), and selecting captions from an alias with no window (plain passthrough).

To caption a trimmed clip, join an external subtitle file timed for the cut as a second `input()` alias ([recipe 10](examples.md#10-mux-external-subtitles-in-or-pull-them-back-out)).

A window on a CTE carrying a subtitle/data column is rejected unconditionally (a filtergraph cannot carry captions); the same constraint bars subtitle columns from `UNION ALL` branches (`concat` has video and audio pads only).
