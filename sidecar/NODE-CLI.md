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
  binds no pad for `feed`.
- **One call, many readers.** A label may be read by any number of pads
  and `-map`s; the host splits it.

Older modules and the host's own nodes (`rowfilter`, `rowmerge`,
`leaky`) keep their positional pads in the same `-filter_complex`.

## Outputs

    -map '[<label>]' [-map '[<label>]' ...] -f nut <path>
    -map '[<label>]' -f ndjson <path>

Every `-map` before one output writes into it, so one output is one NUT
carrying those streams, in that order: one edge. A data label written
`-f nut` is a JSON data stream with the host's progress marks on it (see
`ffrwd-wasm/src/heartbeat.rs`); written `-f ndjson` it is one message per
line, progress dropped. `-f srt`, `-f webvtt` and `-f null` take one data
label each, as they take rows today.

A network with no `-i` at all is a source: it runs until its nodes
finish or its outputs' readers close.

## The recipes

The sidecar half of each, with the ffmpeg producers and consumers around
it left out: `pipe:0` is fed by an ffmpeg decoding the named file, `pipe:1`
read by the one encoding the result, and any other path is a named pipe
the compiler makes.

**145.** `ring(f.video[1], spot(f.video[1]))`. `spot`'s rows are a data
edge; the picture reaches `ring` from the source.

    ffrwd-wasm -f nut -i pipe:0 -m spot=spot.wasm -m ring=ring.wasm \
      -filter_complex '[v=0:v]spot[spots=s];[v=0:v][spots=s]ring[out=o]' \
      -map '[o]' -f nut pipe:1

**146.** `dim` over the rows the WHERE keeps. The predicate runs on the
data edge, before `dim` reads it.

    ffrwd-wasm -f nut -i pipe:0 -m spot=spot.wasm -m dim=dim.wasm \
      -filter_complex '[v=0:v]spot[spots=s];[s]rowfilter=pred=<json>[k];'\
    '[v=0:v][boxes=k]dim[out=o]' \
      -map '[o]' -f nut pipe:1

**147.** One `hear`, read twice: by `burn` and as a caption track.

    ffrwd-wasm -f nut -i pipe:0 -m hear=hear.wasm -m burn=burn.wasm \
      -filter_complex '[a=0:a]hear[cues=h];[v=0:v][words=h]burn[out=o]' \
      -map '[o]' -f nut pipe:1 \
      -map '[h]' -f webvtt captions.vtt

Input 0 carries the picture and the sound in one NUT.

**148.** Mixed kinds in one call.

    ffrwd-wasm -f nut -i pipe:0 -m hear=hear.wasm -m burn=burn.wasm \
      -filter_complex '[a=0:a]hear[cues=h];[v=0:v][a=0:a][words=h]burn[out=o]' \
      -map '[o]' -f nut pipe:1

**149.** `inset(burn(v, a), port => 9100)`: `burn` without words, `inset`
with its feed unbound.

    ffrwd-wasm -f nut -i pipe:0 -m burn=burn.wasm -m inset=inset.wasm \
      -filter_complex '[v=0:v][a=0:a]burn[out=b];[v=b]inset=port=9100:lead=0.5[out=o]' \
      -map '[o]' -f nut pipe:1

**150.** Three pictures on one many-port, from three inputs.

    ffrwd-wasm -f nut -i pipe:0 -f nut -i av2.nut -f nut -i testsrc.nut -m tile=tile.wasm \
      -filter_complex '[v=0:v][v=1:v][v=2:v]tile=columns=3[out=o]' \
      -map '[o]' -f nut pipe:1

**151.** Two outputs of one call, both read.

    ffrwd-wasm -f nut -i pipe:0 -m matte=matte.wasm -m dim=dim.wasm \
      -filter_complex '[v=0:v]matte[mask=m][spots=s];[v=m][boxes=s]dim[out=o]' \
      -map '[o]' -map '[s]' -f nut pipe:1

The output is one NUT with the picture and the rows' data stream.

**152.** The host's reducer on a data edge, written as rows.

    ffrwd-wasm -f nut -i pipe:0 -m spot=spot.wasm \
      -filter_complex '[v=0:v]spot[spots=s];[s]rowmerge=max_span=10[m]' \
      -map '[m]' -f ndjson spots.ndjson

`max_span` is the option `rowmerge` gains with the per-tick shape: its
latency, and where a span still open is cut.

**153.** `explain` reads shapes and runs nothing; the run is 148's.

**154.** A source: no `-i`, one chain with no input pads.

    ffrwd-wasm -m ticker=ticker.wasm \
      -filter_complex 'ticker=text=Nothing to see here:width=1280:height=720:fps=30[out=o]' \
      -map '[o]' -f nut pipe:1

`WHERE s.t < 10` is the consumer's: it reads ten seconds and closes the
pipe, and the sidecar ends.
