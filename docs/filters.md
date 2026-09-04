# Filters

Every filter the installed ffmpeg reports is callable. The surface is introspected (`ffmpeg -filters`, then `ffmpeg -help filter=<name>` lazily), not hand-maintained.

One calling convention:

```
<name>(<stream inputs...>, <positional options...>, <named options...>)
```

- **Stream inputs first.** Count and types come from the pad signature (`unsharp` is `V->V`: one video in; `xfade` is `VV->V`: two). Mismatch: `UDF_ARG_TYPE`.
- **One name for both kinds.** A video filter whose audio twin ffmpeg also ships (`realtime`/`arealtime`, `setpts`/`asetpts`, `settb`/`asettb`, ...) is called by the bare video name over either column; the compiler picks the twin from the column's type. Only a true pair qualifies - a video-only filter and an audio-only `a<name>` - so `mix`/`amix` and `interleave`/`ainterleave`, which merely share a stem, stay two names. `ffmpeg.<name>(...)` is exact and never switches.
- **Positional options bind in declared order** - the order `ffmpeg -help filter=<name>` prints, identical to how `gblur=5:1` binds in a filtergraph. `gblur(a.video[1], 5, 1)` is `sigma=5:steps=1`; `crop(a.video[1], 640, 360, 0, 0)` is `out_w=640:out_h=360:x=0:y=0`.
- **Named options follow**, Postgres `name => value`: `unsharp(a.video[1], luma_amount => 1.5)`, or mixed after positionals. A positional after a named argument is rejected; a named argument naming an option already bound positionally is `FILTER_OPTION_TYPE` ("already set"), never a silent override.

A positional validates as the option it lands on. Option errors are `UNKNOWN_FILTER_OPTION` (with did-you-mean) and `FILTER_OPTION_TYPE` (with type, range, or constants) - captured examples in [errors.md](errors.md).

## Values

Bare numbers, `true`/`false`, single-quoted strings. Enum options take the quoted constant name (`transition => 'wipeleft'`), not the integer. String-typed options accept ffmpeg expressions (`scale(a.video[1], 'iw/2', -2)`, `overlay(a.video[1], b.video[1], '(W-w)/2', '(H-h)/2')`), evaluated per frame by ffmpeg. Expression content is not validated at compile time; a typo inside the quotes surfaces at run time.

## `enable`

`enable` is ffmpeg's timeline switch, a framework option rather than a filter option: an expression evaluated per frame; while false, frames pass through untouched.

```sql
SELECT gblur(a.video[1], 12, enable => 'between(t,0.5,1.5)')
FROM input('clip.mp4') a
```

Accepted only on filters whose `ffmpeg -filters` line carries the `T` flag; elsewhere it is `UNKNOWN_FILTER_OPTION`.

## The `ffmpeg.` namespace

Some filter names are Postgres grammar (`overlay`, `trim`, `format`, ...) and parse as builtins before ffrwd sees them. `ffmpeg.<name>(...)` always means the raw filter and never collides:

```sql
SELECT ffmpeg.trim(a.video[1], start => 1, end => 4),
       ffmpeg.overlay(a.video[1], b.video[1], x => 20, y => 20, eof_action => 'pass')
FROM input('a.mp4') a, input('b.mp4') b
```

Twelve bare names need it (measured against ffmpeg 9.0.1 / sqlglot 30.17): `copy`, `corr`, `format`, `join`, `median`, `normalize`, `null`, `overlay`, `pad`, `random`, `reverse`, `trim`. Whether a collision errors or mangles depends on arity; when in doubt, or in generated code, use the namespace.

`ffmpeg` is reserved: an alias, CTE, or view by that name is `UNSUPPORTED_SQL`.

## The `ffrwd.` namespace

Four macros that do what no single filter does. `ffrwd` is reserved as an alias, CTE or view name, like `ffmpeg`. Two segments under it - `ffrwd.<name>(...)` - is always a macro; packages published under `ffrwd/*` are called with three ([dialect.md](dialect.md#projects-and-packages)). The first three expand to a subgraph and take positional arguments only, in the documented order; named arguments are rejected.

- **`ffrwd.delay(f, seconds)`** - delay a video stream on a transparent canvas (`format=pix_fmts=yuva420p` + `tpad=start_duration=<s>:stop=1:color=black@0`). Use for timed overlays: the delayed stream is invisible until its start time.

  ```sql
  SELECT overlay(f.video[1], ffrwd.delay(p.video[1], 120), 20, 20)
  FROM input('film.mkv') f, input('ad.mp4') p
  ```

  ```
  $ ffrwd compile -f query.sql
  ffmpeg -i film.mkv -i ad.mp4 -filter_complex '[1:v:0]format=pix_fmts=yuva420p,tpad=start_duration=120:stop=1:color=black@0[n2];[0:v:0][n2]overlay=x=20:y=20[out0]' -map '[out0]' out.mp4
  ```

  For audio, use the bare filter in milliseconds: `adelay(a.audio[1], delays => '120000')`.

- **`ffrwd.speed(f, factor)`** - `setpts=PTS/<factor>`; 2 is double speed, 0.5 half.

- **`ffrwd.blur_regions(f, x, y, w, h, sigma)`** - crop the rectangle, `gblur` it, overlay it back in place.

The fourth expands in TIME rather than in the graph, and is the one macro with named options:

- **`ffrwd.loudnorm2(stream, I => ..., TP => ..., LRA => ...)`** - normalize audio to a loudness target the broadcast-compliant way: measure the whole stream, then correct it with one linear gain change. The three options are named only and all optional (ffmpeg's own `loudnorm` defaults cover any you omit); only what you write is rendered. Audio only.

  It compiles to TWO chained commands, not one. Pass 1 runs the same stream through `loudnorm=<your options>:print_format=json` into `-f null -`; `ffrwd loudnorm2env` turns the JSON block on its stderr into `export FFRWD_LN_*=` lines; pass 2 splices those variables into its filtergraph alongside `linear=true`. Recipe 49 in [corpus.md](corpus.md) shows the whole line.

  Two costs come with that shape:

  - the printed command is the one compiled output that is not pure ffmpeg - it needs `ffrwd loudnorm2env` on `PATH` when you run it. If you would rather not depend on that, run pass 1 by hand and paste the five numbers into a bare `loudnorm(...)` call yourself.
  - the printed command is POSIX-shell only (`eval`, `$( )`, `${ }`). On cmd.exe or PowerShell, use `ffrwd run`, which does the measure/substitute/encode handoff in process and needs no shell anywhere.

  Current limits, each rejected as `UNSUPPORTED_SQL`: one `loudnorm2` per query; never together with a `two_pass` sink; never inside a fan-out `TO (<expression>)`; never in a table/CSV query. Use the bare `loudnorm(...)` filter when one pass is genuinely enough.

## Generated sources in `FROM`

Zero-input filters (`testsrc`, `color`, `anullsrc`, `sine`, ...) live in `FROM` with a mandatory alias, namespace required:

```sql
SELECT f.video[1], f.audio[1] FROM input('clip.mp4') f
UNION ALL
SELECT t.video[1], s.audio[1]
FROM ffmpeg.testsrc2(duration => 1, size => '320x240', rate => 15) t,
     ffmpeg.anullsrc(duration => 1) s
```

Sources take options by name only (no input pads means no positional slots). A source alias exposes exactly one stream of one known type (`t.video[1]` or `s.audio[1]`), carries no provenance, cannot be `WHERE`-trimmed (length comes from its own `duration =>`), and adds no `-i` to the command.

## Array-returning and N-input filters

Three filters return an **array** of streams, sized statically by an option: `ffmpeg.channelsplit` (per channel of `channel_layout`), `ffmpeg.acrossover` (per `split` frequency, plus one), `ffmpeg.extractplanes` (per plane). The result splats, subscripts, and broadcasts like any array:

```sql
WITH ch AS (
  SELECT ffmpeg.channelsplit(a.audio[1]) AS lr FROM input('stereo.mp4') a
)
SELECT amix(volume(ch.lr[1], 0.5), volume(ch.lr[2], 2.0))
FROM ch
```

```
$ ffrwd compile -f query.sql
ffmpeg -i stereo.mp4 -filter_complex '[0:a:0]channelsplit[n10][n11];[n10]volume=volume=0.5[n2];[n11]volume=volume=2.0[n3];[n2][n3]amix=inputs=2[out0]' -map '[out0]' out.mp4
```

An N-input filter takes however many streams you pass: any filter your ffmpeg reports with a dynamic *input* pad count and one output (`amix`, `hstack`, `xstack`, `ffmpeg.join` - bare `join` is the SQL keyword - and dozens more, whatever your build has). Where the filter has an `inputs`/`nb_inputs` option it is set from the stream count, and a written value that disagrees is a typed error; `ladspa` has none - its pad count comes from the loaded plugin. A dynamic *output* pad count (`split`, the array-returning trio's mirror) is a different shape entirely, and stays out of reach: the compiler inserts its own splits.

To choose which tracks to pass in the first place (by language, codec, resolution), use track rows: [rows.md](rows.md).

## Plugins

Two filters load code at runtime, and both compile like any other call:

- `frei0r(video, filter_name => '<plugin>', filter_params => 'a|b|c')` and the source `ffmpeg.frei0r_src(...)`: video effect plugins found through the `FREI0R_PATH` environment variable. [Recipe 62](corpus.md#62-use-a-plugin-filter).
- `ladspa(audio, ..., file => '<library>', plugin => '<label>', controls => 'c0|c1')`: audio plugins found through `LADSPA_PATH`; pass as many streams as the plugin has input ports.

Whether either exists is a property of the installed ffmpeg build, exactly like every other filter - `frei0r` and `ladspa` both appear in common full builds. Plugin parameters are opaque strings to the compiler; their meaning belongs to the plugin.

## Not supported

The canonical list of what cannot be called or expressed is [known_gaps.md](known_gaps.md).

## Introspection and binaries

`-filters` is parsed once per process; each `-help filter=<name>` once, on first reference. Results are cached on disk under `~/.cache/ffrwd/`, keyed by a hash of `ffmpeg -version`'s first line - upgrading ffmpeg invalidates the cache automatically. `ffrwd.registry.clear_cache()` wipes both layers.

ffmpeg and ffprobe are required. A system install on `PATH` wins; otherwise the bundled `static-ffmpeg` provisioner fetches both on first use. `ffrwd explain` shows what the registry resolved.
