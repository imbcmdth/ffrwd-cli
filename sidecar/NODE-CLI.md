# Node calls on the sidecar's command line

How a compiled query hands the sidecar a network that holds node modules
(`ffrwd:av@0.19.1`). Everything an older module takes stays as it is: the
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

    -pad '{"color": {"range": "tv", "primaries": "bt709", "trc": "bt709", "space": "bt709"}, "tags": {"smart_timed": "1"}}'

after an `-i` says what its NUT does not carry: a raw picture's colour, in
ffmpeg's names, and tags for its streams beside their own, which reach a
node in `stream-info.tags` (a hold input anchored `tagged` reads them).

    -pad '{"rate": {"num": 30000, "den": 1001}}'
    -pad '{"rate": {"v": {"num": 25, "den": 1}, "a": {"num": 48000, "den": 1}, "a:1": {"num": 16000, "den": 1}}}'

`rate` is what the compiler knew of the input's streams when it asked a
node for its shape: a video stream's frame rate, an audio stream's sample
rate over 1 (the WIT's `stream-hint`). One rational is the rate of every
stream of that input, for an edge carrying one stream; an object of them
is one per stream, keyed by the stream as a pad names it after the input's
number (`v` is `v:0`, then `v:1`, `a`, `a:1`, `d`, ...), and a stream it
leaves out has none. Both numbers are positive. Write it wherever the rate
was known when the shape was asked, and only then: the host builds the
same `binding` list from it (see "Asking the shape again"), and hands each
bound stream the same hint at `init` (`bound-stream.hint`), so a module
that derives its shape again there derives the plan's.

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
  unbound, is served by a loopback listener of the host's own on the port
  its param names: `inset=port=9100` binds no pad for `feed`, and the host
  listens on 127.0.0.1:9100 (see "Feeds by port"). A hold port with a
  `port_param` that a pad binds gets the port the host picked written into
  that param.
- **One call, many readers.** A label may be read by any number of pads
  and `-map`s; the host splits it.

### Asking the shape again

The host asks every node module for its shape at the start of the run, and
it must get the shape the plan was made from, so it asks with the list the
compiler asked `--shape --bound` with (`NODE-SHAPE.md`):

- one `binding` per port the chain binds, in the order its pads first name
  each port;
- in each, one stream per pad naming that port, in the order written;
- each stream's `rate` is the one `-pad` gives that stream of its input
  (above); a stream bound by a label, which another chain writes, has none.

A hold input served by a port is not in the list, as it is not among the
pads. Ask `--shape` with exactly this list, as JSON wherever a rate is
known: `[v=0:v][a=0:a][words=n1]` over `-pad '{"rate": {"v": {"num": 30,
"den": 1}}}'` is `--bound
'[{"input":"v","streams":[{"rate":{"num":30,"den":1}}]},{"input":"a","streams":[{"rate":null}]},{"input":"words","streams":[{"rate":null}]}]'`.

A frame module of an older world keeps its positional pads in the same
`-filter_complex`, and runs as the node its adapter makes of it. The host's
`rowfilter` and `rowmerge` read one data edge by position: `[n1]rowfilter`,
and `leaky` one picture stream, decoded or coded: `[v=0:v]leaky=max_lateness=0.5[live]`.
Over packets it drops whole groups, from a late packet to the next keyframe
in time.

## Outputs

    -map '[<label>]' [-map '[<label>]' ...] -f nut <path>
    -map '[<label>]' -f ndjson <path>

Every `-map` before one output writes into it, so one output is one NUT
carrying those streams, in that order: one edge. The streams are written
in time order across them, each item as soon as no other stream can still
bring one earlier, so an output is the same bytes at every `-jobs`. A data
label written `-f nut` to a pipe is a JSON data stream with the host's
progress marks on it (see `ffrwd-wasm/src/heartbeat.rs`), which its reader
needs as the bytes arrive; to a file it carries the messages alone. Written
`-f ndjson` it is one message per line, each stamped with its `pts` and its
`time` in seconds where the row does not name them itself, progress
dropped; a self-clocked node with inputs ticks on the host's clock, so its
rows are left as it wrote them. `-f srt` and `-f webvtt` take one
data label of cues each and write the document whole once it ends; `-f
null` takes any.

A network with no `-i` at all is a source: it runs until its nodes finish
or every output's reader has closed, and a reader closing is a clean end.

## Feeds by port

A hold input the call gives a port rather than a stream is bound a stream
of the host's own at `init`, in the port's format: the size and colour of
the input it follows (`accepts.like`, or the clock input), in the first
pixel or sample format the port accepts. The host listens on the port
from the moment the run starts, and whatever connects and writes a NUT of
raw video and PCM is the input's source for as long as it stays: its
picture is conformed to that format on the way in (rgba and yuv420p, at
any size), its sound to the port's sample format (s16 or f32; another rate
or channel count is refused by name and the pictures are kept). A group of
hold inputs (`hold.group`) shares one listener and one connection, its
picture to the video port and its sound to the audio port.

One connection at a time: a second while one is on is accepted and closed
at once, and a connection that has sent nothing is dropped for one that
knocks. Every connection asks for a 16 MiB receive buffer, and is read
only while the input has room, so a source that outruns the clock waits on
its socket. Each connection is a feed (`tick.feed`), with the source's own
tags and time base from its start on; a source whose pts jump backwards or
more than a second forwards is a new feed on the same connection. A
connection may carry any of a group's members, and streams beyond them are
left on the wire: a feeder of sound alone feeds the sound. Every member of
a group is told the lead's offset, the picture's where the connection
brings one and otherwise its first member's: its `feed-start.at` is the
lead's, and its `first-pts` the point of its own stream that offset puts
there, so `(at, first-pts)` maps any member's pts onto the clock. A member
the connection does not bring (not sent, or its sound refused) gets no feed
at all on any tick, where one not arrived yet gets a feed and no frames.

A data input whose `interval` names the group (`interval.group`) arrives
on the same connection, unbound by any pad, as the hold inputs do. What a
feeder writes so its rows land there, in one NUT:

- the picture (raw video, rgba or yuv420p) and the sound (PCM) as above,
  and a data stream: stream class 3, codec tag `JSON`, one packet per
  message whose payload is one JSON object (what ffrwd writes for any data
  edge);
- the data stream's pts counted on the same origin as the picture's, in a
  time base of its own: the host places them with the group's one offset,
  fixed by the first picture, so a row stamped at a picture's pts lands on
  the tick that picture shows on;
- data streams go to the group's data inputs in the order of the shape's
  inputs, the first data stream on the wire to the first such input; one
  beyond them is left on the wire, and a progress mark on it is ignored;
- tags as for any feed: the connection's and the picture's reach every
  member, so `smart_timed` on the picture times the rows too.

A message is handed on the tick whose interval holds its time on the
clock (`ahead` included), and one whose time the clock has passed, or that
arrived before the feed's start was fixed, on the first tick that can take
it. A connection's messages never outlive its feed: those queued when it
ends are dropped with it. A group data input bound by a pad is refused;
with its group's hold inputs bound to streams rather than a port, it is
bound to a stream like any data input and placed with that group's offset.

The host says what it did on stderr, as lines and as rows behind
`ffrwd:row `: `{"kind":"listen",...}` once the port is bound,
`{"kind":"feed","event":"start",...}` with the anchor and the mapping when
a feed comes up (and `"absent"`, the members it does not bring) and `{"kind":"feed","event":"end",...}` with how many
frames it showed, repeated and skipped when it ends, and
`{"kind":"late",...}` when an interval input's messages arrived after the
tick that held them.

## How a run goes

- **Inputs.** Every `-i`'s headers are read first, all at once; then a
  thread per input hands each frame to whatever reads its stream, waiting
  while a reader's queue is full. A stream nothing reads is skipped.
- **Ticks.** Each node's ticks are cut once, centrally, by its shape: a
  clock ticks when the frame or packet after its tick has arrived (or its
  frame says how long it lasts), so every tick's interval is settled
  before it runs, and the progress the node sends after it is the end of
  that interval less the port's latency.
- **Hold inputs.** A hold input bound to a stream holds the tick until its
  source has reached the tick (a frame past it, or its end), for no longer
  than its `timeout` of clock time, after which the source is behind; a
  hold input served by a port never holds a tick. Either way a video input
  hands the newest frame at or before the tick, repeating the last one
  while the source is behind and skipping forward when it catches up, and
  an audio input hands the tick's samples re-cut from what arrived, or
  none while the source is behind. A `first-frame` source is primed until
  `lead` seconds of it are held, or its end, and scheduled `lead` ahead of
  the clock, on the clock's grid; a `shared-clock` source's first frame
  waits for the clock to reach its pts, and one the clock has passed shows
  at once from where the clock is. The last frame shows on its own turn
  and the feed ends after it, unless `linger` keeps it; a clock that jumps
  (backwards, or more than a second forwards) ends every feed. Where the
  source has ended the host foretells the last tick the feed shows on
  (`feed.ends`), counted on the clock's grid: exact for a rate clock,
  learned from the frames of an input clock.
- **Interval inputs.** The clock is held until every interval input's
  producer has progressed past the tick's interval (and `ahead`), for no
  longer than the input's `latency` past it, counted on the clock input's
  own arrival, or on a rate clock on the newest time any input has
  reached. A message that arrives after its tick was cut is delivered at
  the next tick and reported.
- **Re-stamping.** An interval input anchored `first-frame` (or `tagged`
  where the stream's tags do not carry the name set to `1`) counts on an
  origin of its own. Its first message stands at the first tick the host
  cuts once that message has arrived: that fixes an offset for the
  stream's life, and every message, and every progress mark, is restamped
  onto the clock's time base with it (`pts` rescaled, then the offset
  added). The node is told the clock's time base for that stream at
  `init` and on every tick. Until its first message the input holds no
  tick, so a stream on another origin never waits behind the programme.
  `shared-clock` is the stream's own pts, rescaled, as before.
- **Timing inputs.** An input whose `accepts.wants` is `timing` is handed
  each frame's `pts` and `duration` and the stream's info, and no bytes:
  the host copies and converts nothing for it, a port feed's picture is
  not conformed, and `fetch` (or `same`) on its frames is a fault. Hand
  it the stream in whatever raw format the source has cheapest; its
  `accepts` formats are not checked. An audio timing input is re-cut by
  sample count as any audio input is.
- **Ordinal.** `tick.ordinal` is the tick's number in the run, from 0,
  counted over every instance: on every worker the same tick has the same
  number, so a node that numbers things by frame can run on many.
- **Ended feeds.** `tick.ended-feeds(id)` on a hold input lists the feeds
  that ended since that instance's previous call, oldest first, every one
  that ended before on its first call: whatever ended them (the source,
  `timeout`, a jump in the clock, a connection closing with nothing
  queued), foretold or not. Each has `ends` set to the last tick it
  showed on, and its `start` as `tick.feed` gave it.
- **Known.** `feed-start.known` is the clock time of the tick the feed's
  start was fixed on: for a `first-frame` feed the tick its priming
  finished, `lead` before `at` on the clock's grid; for a `shared-clock`
  feed the tick its first frame arrived by, which for a timed feeder is
  seconds before `at`; equal to `at` where the clock had already passed
  the first frame, which then shows at once.
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
