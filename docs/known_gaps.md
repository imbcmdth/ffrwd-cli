# Known gaps

What ffrwd cannot express today, and the sharp edges of what it can.
If one of these blocks you, run ffmpeg directly for that step — ffrwd
output is plain ffmpeg, so the two mix freely in a script.

## Not expressible

| gap | ffmpeg surface | notes |
| --- | --- | --- |
| HLS / DASH packaging | `hls_*`, segment muxer options | Format-specific muxer option families are not modeled. Writing to an `.m3u8` path may work for defaults, but segment length, playlist type, and encryption options have no spelling. |
| Protocol options | `-headers`, `-user_agent`, `-rtsp_transport`, `-timeout` | Network inputs and outputs are passed to ffmpeg verbatim; per-protocol tuning options have no input/sink spelling. Authenticated URLs work only if the credential fits in the URL itself. |
| Lossless concat | concat demuxer (`-f concat -i list.txt -c copy`) | Joining files without re-encoding needs the demuxer's list-file protocol. `concat` in ffrwd is the filter, which re-encodes. |

## Not callable

- Variable-OUTPUT-pad filters and multi-output filters (`scale2ref`,
  `feedback`): `UNSUPPORTED_SQL`. `split` stays rejected regardless —
  the compiler inserts its own. A variable-INPUT-pad filter (`amix`,
  `hstack`, `xstack`, and every other filter your ffmpeg reports that
  way) is callable, and so is the array-returning trio (`channelsplit`,
  `acrossover`, `extractplanes`); `concat` joins them under `VARIADIC`
  only — `concat(a, b)` is still `UNSUPPORTED_SQL`,
  `concat(VARIADIC array_agg(v))` is a call. `UNION ALL` is concat too;
  that spelling never needs `VARIADIC`.
- Sources with more than one output pad (`avsynctest`, `movie`); all
  sinks.
- Options typed `binary` or `dictionary`: setting one is
  `FILTER_OPTION_TYPE`; the filter's other options work.
- Runtime filter commands (`sendcmd`, `zmq`).
- A wasm module exporting `packet-filter` that READS ROWS. The
  placement is there -- a `RETURNS packets` call in a COPY's SELECT
  compiles to encoder, filter, muxer -- but the rows such a filter reads
  have no spelling: they reach the sidecar on `-rows-in`, which needs a
  rows file the plan writes before the filter's stage starts, and
  nothing emits one. A module declaring `reads-rows`, and a declaration
  with an annotation parameter, are `UNSUPPORTED_SQL` at the call. Rows
  arriving on a pipe while the packets flow is further out still.
- A packet filter at a MANIFEST destination. One filter instance per
  rendition row is the shape it would take, and the planner treats each
  pad as its own encoded stream already; what is missing is a manifest
  destination that accepts a filtered column. A packets cell at one is
  `UNSUPPORTED_SQL`. A filter in front of a `RETURNS sink` packet sink
  works, ladder or not.

## Sharp edges

- **Stream-copied splits snap to keyframes.** An output fan-out that
  splits by chapter (or any time window) with stream copy starts each
  piece at the nearest preceding keyframe, exactly as ffmpeg does.
  Re-encode the video for frame-accurate cuts.
- **The printed `loudnorm2` chain is POSIX-shell only.** It uses
  `eval`, `$()`, and environment splices, and calls `ffrwd
  loudnorm2env` at run time. On cmd.exe or PowerShell, use `ffrwd
  run`, which performs the substitution in-process.
- **A printed process-plan pipeline is POSIX-shell only.** A plan
  routed through a sidecar module without any fan-in prints as one
  `|` chain, which cmd.exe and PowerShell do not run the same way
  POSIX shells do. Use `ffrwd run` there too. A plan with fan-in
  cannot be a pipeline on any shell -- it prints as a numbered,
  run-only listing instead, and says so.
- **A compile-time packet read copies the stream through ffmpeg.** A
  packet sink read in FROM
  ([rows.md](rows.md#packet-rows---ffrwdindexrecordsfvideo1-v)) runs
  ffmpeg and the sidecar while the query compiles, so a compile of such
  a query needs both installed and costs one stream copy of the file.
  `wants: keyframes` still demuxes the whole track - the bitstream
  filter that drops the rest runs after the demuxer - so it saves the
  work after the demux, not the read. A demuxer-level skip
  (`-discard nokey`) would save the read too, but on mov it reports the
  wrong presentation times for a stream that reorders frames, which
  would put the rows on a different clock from the rest of the query.
- **A sidecar process reads one stream and writes one.** A region of
  modules can fan out and fan in as much as it likes inside itself,
  but its BOUNDARY is one pipe each way: only stdin and stdout are
  wired to it, so a region reading or writing two streams at its edge
  is refused. Nothing the dialect can spell today produces one.
- **A module reading several streams needs them in lockstep.** They
  have to reach it from one point through modules that declare one
  frame out per frame in; a `split` counts, and an ffmpeg filter does
  not - it declares nothing about its frame timing, and ffrwd will not
  assume. Anything else is `UNSUPPORTED_SQL` at the declaration.
- **`drawtext` needs a font on some builds.** The filter works out of
  the box; pass `fontfile` like any other option. Omitting it falls
  back to fontconfig, which depends on how the local ffmpeg was built —
  some Windows builds crash instead of picking a default. When in
  doubt, name the font.
- **A fan-out over a CTE needs a value column to name its files.** A
  fan-out `TO (expression)` builds its filename from row columns, and a
  CTE exposes only what its body selected. Select the value you want to
  name files with - `SELECT v AS frame, i.i AS n ...` - and the outer
  `TO` reads `x.n` like any other row column; the same goes for a
  `GROUP BY` over a CTE column. A body that selects streams alone still
  has nothing to name files with, and the `ROW_COUNT_MISMATCH` says so.
  A table-returning function is a CTE by the time lowering sees it, so
  its `RETURNS TABLE(n number, ...)` column works the same way.
- **Filter outputs carry no facts.** Metadata columns describe probed
  input streams only; a filter's output is a stream with no readable
  `channel_layout`, `width`, `codec` and so on, even where ffmpeg
  itself derives them (`channelsplit` emits one-channel `FL`/`FR`
  layouts, which the AAC encoder then rejects as non-`mono` - add
  `aformat(..., channel_layouts => 'mono')` per leg). Reading a field
  off a filter output is a typed rejection. Deriving facts through
  filters that change them is future work.
- **Streams ffmpeg cannot identify are rejected at compile time.**
  Some sources carry streams with no detectable codec (certain DASH
  text tracks, for example). Selecting one is a compile error; table
  queries over the same source still work.
- **`merge_cues` reads a column or a producer's call, not a name for
  one.** Over rows a file carries it takes a record array column (with
  or without a track subscript) or an `ARRAY(SELECT ... WHERE ...)`
  gather over one; over rows a module writes it takes that module's own
  annotation column. A CTE column bound to either, and a rows
  function's result, are refused - the rejection names what it does
  read. Merge the rows where they are produced instead.
