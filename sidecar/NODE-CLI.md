# Node calls on the sidecar's command line

How a compiled query hands the sidecar a network that holds node modules
(`ffrwd:av@0.19.0`). Everything an older module takes stays as it is: the
same `-m`, the same positional pads, the same outputs. A node module is
told apart by its export, and its pads say which port they bind.

What a node's ports are named, which are required, and what kind each
carries come from `ffrwd-wasm --shape` (`NODE-SHAPE.md`). Port names in
the examples below are the stand-in modules' own.

## Inputs

    -f nut -i <path>

One `-i` is one edge: one NUT, carrying every stream that crosses it,
video, audio, coded packets and data (a JSON stream, codec `json`) in any
mix, interleaved by time. A producer writes all of an edge's streams into
that one pipe, so no stream of it can wait on another pipe.

A pad names a stream of an input by class and position, as ffmpeg's
stream specifiers do:

| Pad | Stream |
|---|---|
| `[0:v]`, `[0:v:1]` | input 0's first, second video stream, raw or coded |
| `[0:a]`, `[0:a:1]` | its audio streams, raw or coded |
| `[0:d]`, `[0:d:1]` | its data streams |

`[N:v]` is `[N:v:0]`. A raw stream binds a video or audio port, a coded
one a packets port, a data stream a data port.

## Nodes

    -m <name>=<path>
    -filter_complex '<chain>;<chain>;...'
    -params-from <name>=<file>

A chain is input pads, the node, output pads, as for any module. For a
node module every pad carries the port it binds before an `=`:

    [<port>=<pad>]...<name>=<k>=<v>:<k>=<v>[<port>=<label>]...

- **Inputs by name.** `[v=0:v]` binds input 0's video to port `v`;
  `[spots=s]` binds the stream labelled `s`. Order does not matter
  between ports. A pad with no `<port>=` on a node is refused.
- **Many-ports.** Several pads naming one port bind all of them, in the
  order written: `[v=0:v][v=1:v][v=2:v]`.
- **Optional ports.** A port the chain names no pad for is unbound and
  absent from the shape the module is asked for. A required port left
  unbound is refused naming it.
- **Outputs by name.** `[out=o0]` labels what port `out` writes. A port
  no pad names is not latched: the node is told the query does not read
  it, and may leave it unmade.
- **Rows.** `[@rows=r0]` labels the rows the node emits beside its ports
  (`emitted.rows`), a data stream like any other.
- **Data pads.** A label's kind is its port's: a data output's label is a
  data edge, and binds only a data port.
- **Params.** Options are typed by the params schema, as for any module.
  `-params-from <name>=<file>` reads one node's params whole, as JSON,
  out of a file, for params too long for a command line; it replaces that
  node's options.
- **Hold inputs given by port.** A hold port with a `port_param`, left
  unbound, is listened for on the port its param names: `inset=port=9100`
  binds no pad for `feed`. This host does not listen yet, and runs the
  node with the port absent.
- **One call, many readers.** A label may be read by any number of pads
  and `-map`s; the host splits it.

A frame module of an older world keeps its positional pads in the same
`-filter_complex`, and runs as the node its adapter makes of it. The host's
`rowfilter` and `rowmerge` read one data edge by position: `[n1]rowfilter`.

## Outputs

    -map '[<label>]' [-map '[<label>]' ...] -f nut <path>
    -map '[<label>]' -f ndjson <path>

Every `-map` before one output writes into it, so one output is one NUT
carrying those streams, in that order: one edge. The streams are written
in time order across them, each item as soon as no other stream can still
bring one earlier, so an output is the same bytes at every `-jobs`. A data
label written `-f nut` is a JSON data stream with the host's progress
marks on it (see `ffrwd-wasm/src/heartbeat.rs`); written `-f ndjson` it is
one message per line, progress dropped. `-f srt` and `-f webvtt` take one
data label of cues each and write the document whole once it ends; `-f
null` takes any.

A network with no `-i` at all is a source: it runs until its nodes finish
or every output's reader has closed, and a reader closing is a clean end.

## How a run goes

- **Inputs.** Every `-i`'s headers are read first, all at once; then a
  thread per input hands each frame to whatever reads its stream, waiting
  while a reader's queue is full. A stream nothing reads is skipped.
- **Ticks.** Each node's ticks are cut once, centrally, by its shape: a
  clock ticks when the frame or packet after its tick has arrived (or its
  frame says how long it lasts), so every tick's interval is settled
  before it runs, and the progress the node sends after it is the end of
  that interval less the port's latency.
- **Lanes.** A pure node runs on as many workers as `-jobs` allows, its
  results put back in tick order before they leave. A pure node with a
  state input opens every instance before its first tick, and each
  instance is handed the rows of the ticks it did not process. Any other
  node runs one tick at a time.
- **Finishing.** A node that says it is finished gets its last call on the
  instance that said so, and its outputs end; a rate clock with inputs
  makes its last call once every input has ended and its ticks have
  passed the last frame.
- **`rowmerge=max_span=<s>`** is the span reducer: rows sharing a
  `start_t` are one span, which keeps the last row's fields and ends at the
  end of the last tick that carried one (a tick with none is a gap inside
  it). A span leaves once its producer's progress is `max_span` past its
  start, cut there if it is still going and carried on from the cut, or
  when the input ends. `max_span` is its output's latency.
- **A rate of an input.** A node clocked at an input's rate ticks at the
  rate the first stream bound to that port states in its header, or at
  its time base's inverse where it states none.

## The recipes

The sidecar half of each, as `ffrwd compile` prints it, with the ffmpeg
producers and consumers around it left out: `pipe:0` is fed by an ffmpeg
decoding the named file, `pipe:1` read by the one encoding the result, and
any other path is a named pipe the compiler makes. Module paths are
shortened.

**145.** `ring(f.video[1], spot(f.video[1]))`. `spot`'s rows are a data
edge; the picture reaches `ring` from the source.

    ffrwd-wasm -f nut -i pipe:0 -m spot=spot.wasm -m ring=ring.wasm -filter_complex \
      '[v=0:v]spot=every=30[spots=n1];[v=0:v][spots=n1]ring[v=out0]' -map '[out0]' -f nut \
      pipe:1

**146.** `dim` over the rows the WHERE keeps. The predicate runs on the
data edge, before `dim` reads it.

    ffrwd-wasm -f nut -i pipe:0 -m spot=spot.wasm -m dim=dim.wasm -filter_complex \
      '[v=0:v]spot=every=30[spots=n1];'\
    '[n1]rowfilter=pred={"ge"\\:\[{"field"\\:"w"}\,{"lit"\\:20}\]}[n2];'\
    '[v=0:v][boxes=n2]dim=amount=0.5[v=out0]' -map '[out0]' -f nut pipe:1

**147.** One `hear`, read twice: by `burn` and as a caption track. Input 0
carries the sound and the picture in one NUT.

    ffrwd-wasm -f nut -i pipe:0 -m hear=hear.wasm -m burn=burn.wasm -filter_complex \
      '[a=0:a]hear[cues=out1];[v=0:v][words=out1]burn[v=out0]' -map '[out0]' -f nut \
      '<named pipe n2>' -map '[out1]' -f webvtt '<named pipe cues>'

**148.** Mixed kinds in one call. The input carries a sound, the picture
and the same sound again; `[a=0:a:1]` is the second.

    ffrwd-wasm -f nut -i pipe:0 -m hear=hear.wasm -m burn=burn.wasm -filter_complex \
      '[a=0:a]hear[cues=n1];[v=0:v][a=0:a:1][words=n1]burn[v=out0]' -map '[out0]' -f nut \
      pipe:1

**149.** `inset(burn(v, a), port => 9100)`: `burn` without words, `inset`
with its feed unbound.

    ffrwd-wasm -f nut -i pipe:0 -m burn=burn.wasm -m inset=inset.wasm -filter_complex \
      '[v=0:v][a=0:a]burn[v=n1];[v=n1]inset=port=9100:lead=0.5[v=out0]' -map '[out0]' -f nut \
      pipe:1

**150.** Three pictures on one many-port, from three inputs.

    ffrwd-wasm -f nut -i '<named pipe a>' -f nut -i '<named pipe b>' -f nut -i '<named pipe c>' \
      -m tile=tile.wasm -filter_complex '[v=0:v][v=1:v][v=2:v]tile=columns=3[v=out0]' \
      -map '[out0]' -f nut pipe:1

**151.** Two outputs of one call, the matte written beside the picture
`dim` makes from the rows: one NUT of two streams.

    ffrwd-wasm -f nut -i pipe:0 -m matte=matte.wasm -m dim=dim.wasm -filter_complex \
      '[v=0:v]matte=every=30[mask=out1][spots=n11];[v=0:v][boxes=n11]dim=amount=0.5[v=out0]' \
      -map '[out0]' -map '[out1]' -f nut pipe:1

**152.** The host's reducer on a data edge, written as rows.

    ffrwd-wasm -f nut -i pipe:0 -m spot=spot.wasm -filter_complex \
      '[v=0:v]spot=every=30[spots=n1];[n1]rowmerge=max_span=10[out0]' -map '[out0]' -f \
      ndjson spots.ndjson

**153.** `explain` reads shapes and runs nothing; the run is 148's.

**154.** A source: no `-i`, one chain with no input pads.

    ffrwd-wasm -m ticker=ticker.wasm -filter_complex \
      'ticker=text=Nothing\ to\ see\ here:width=1280:height=720:fps=30[video=out0]' -map \
      '[out0]' -f nut pipe:1

`WHERE s.t < 10` is the consumer's: it reads ten seconds and closes the
pipe, and the sidecar ends.
