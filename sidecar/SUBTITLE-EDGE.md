# The subtitle edge

What the sidecar accepts on the command line for subtitle output, and what
it writes. The compiler builds this argv; ffmpeg reads what comes out.

## NUT carries no cue durations

ffmpeg's NUT muxer defines a codec tag for two subtitle codecs only. `srt`,
`subrip`, `webvtt` and `mov_text` are all refused at header time with
`No codec tag defined for stream 0`; `text` and `ass`/`ssa` mux. Neither of
those two keeps the packet duration, which is where a cue's end time lives:
the same encoder writing Matroska keeps `duration=2000`, and writing NUT
gives `duration=N/A`. Round-tripped back out, every cue ends where it
starts.

So subtitles do not ride the NUT edge. They go to an output of their own,
in one of two forms. A document (`srt`, `webvtt`) gathers the cue rows and
is written whole when the stream ends. ffmpeg reads such a file to its end
before it opens the inputs after it, so a stream that never ends needs the
other form: `webm`, WebVTT in WebM, each cue written the moment its row
arrives, its end time in the block, which ffmpeg reads as it goes.

## The argv

An output is `-f <format> <path>`, and in a network a `-map` names the label
it writes, exactly as an `-f nut` or `-f ndjson` output does. `-` and
`pipe:1` are stdout; anything else is an ordinary path.

    -map [out0] -f srt    subs.srt
    -map [out0] -f webvtt subs.vtt
    -map [out0] -f webm   pipe:3

A `webm` output writes its header when it opens, before any cue, so its
reader can open it while the cues are still to come. The header names one
track, `D_WEBVTT/SUBTITLES`, in a segment of unknown size, and ends with an
empty cluster: ffmpeg's matroska demuxer reads a header up to the first
cluster, and would otherwise wait for the first cue before it opened the
inputs after this one, which may be the picture the cues are said over.
Each cue is a cluster of its own, its block's data the cue's empty identifier
and settings lines and then its text, its duration the cue's, in
milliseconds.

Between cues a `webm` output writes heartbeats: a block lasting no time, its
text a single space (ffmpeg's matroska demuxer refuses an empty WebVTT
block), each time the node writing the cues moves on. ffmpeg reads no input
more than a tenth of a second ahead of an output's furthest-behind stream,
so a cue stream with nothing to say would stop it reading the picture beside
it, and the picture's pipe would fill. The heartbeats keep the cue stream
level with the picture. Every cue lasts, so the reading ffmpeg drops the
heartbeats by their duration before its muxer:

    -c:s copy -bsf:s noise=drop=not(duration)

A heartbeat is at the newest time the node has handed on any of its outputs,
its picture's where it writes one, as well as at the cue label's own
progress. A text track can run seconds behind the picture beside it, as one
ffrwd/moq's `subscribe` reads runs three seconds behind, held there by its
publisher, and a cue stream beating on the text track's time would hold the
picture back by as much. Only the cue stream hears these beats: a node
reading the same cue label still waits on the label's own progress.

The rows output that already exists is spelled the same way, and writes one
NDJSON line per row as the rows arrive:

    -map [out0] -f ndjson rows.ndjson

At most one output of each format. A subtitle output may name the same label
an `-f nut` output names, so the frames and the subtitles come off one run:

    ffrwd-wasm -f nut -i - \
      -m transcribe=transcribe.wasm \
      -filter_complex "[0:a]transcribe=language=es:language_to=en[out0]" \
      -map [out0] -f nut - \
      -map [out0] -f srt subs.srt

Rows are the exception to that one-per-format rule: a line writes one
document per set of rows it has, and `-rows <index>` says whose rows each
holds - the 0-based position, among the line's `-m` flags, of the module
that produced them, whether it read them off frames or is a rows module
chained after one. It precedes the output it names, the way `-map` does,
and an output carrying frames is refused one.

    ffrwd-wasm -f nut -i - \
      -m captions.wasm -m fauxlate.wasm -rows-from 0 \
      -rows 0 -f webvtt cues.vtt \
      -rows 1 -f webvtt translated.vtt

A line writing ONE rows document needs no `-rows`: it is the only rows there
are, and it takes whatever chain of rows modules the line carries.

## The rows a subtitle output reads

A cue row is a JSON object with all three of:

    start_t  number, seconds from the start of the stream
    end_t    number, seconds from the start of the stream
    text     string

and two it may carry:

    final    boolean, false for an utterance still growing
    voice    string, who says the cue

Only a finished cue is written: a row whose `final` is false is sent again
as its utterance grows, and its earlier forms would be cues that overlap. A
row with no `final` is a finished cue. A row's `voice` marks the cue in
WebVTT as `<v Name>`; SRT has no such mark and writes the text alone. In
WebVTT a cue's `&`, `<` and `>` are written as character references, and a
blank line inside it, which would end the cue, is dropped.

A document's cues are written in the order the rows arrive, numbered from 1.
A `webm` stream's go out in the order they start, since its reader takes
each as the next packet: a cue whose row arrives after a later cue's starts
where that one did, and one that ended by then is dropped. Heartbeats take no
part in that order: a cue whose row arrives late keeps its own start however
far the heartbeats have gone. A row object
carrying none of the three names is skipped: that is the other arm of a
module's `oneOf` rows schema, such as transcribe's trailing
`{"transcript": ...}`. A row carrying some of them but not all, or one whose
values are the wrong type or not a time, is refused naming the output and the
field. A row that is not a JSON object at all is refused naming the output.

## What ffmpeg does with it

The document is an ordinary subtitle file, so the terminal ffmpeg reads it as
an input and encodes it into the container:

    ffmpeg -i video.mp4 -i subs.srt -c:v copy -c:a copy \
      -c:s mov_text -metadata:s:s:0 language=eng out.mp4

A `webm` stream is an input too, `-f webm -i <pipe>`, and the one the compiler
hands a module's cue track over: a file's muxer copies or encodes its cues,
and an hls destination copies them into the WebVTT segments it cuts.

The language tag is the compiler's to write; the sidecar does not put one in
the document.

`mov_text` shows one cue at a time: where two cues in the document overlap,
ffmpeg clips the earlier one at the later one's start. Cues that do not
overlap keep their spans exactly.

A stream that produced no cues leaves an empty document. An empty `.srt` is
zero bytes and ffmpeg refuses to open it (`Invalid data found`); an empty
`.vtt` is its `WEBVTT` header alone and ffmpeg reads it as a stream with no
cues.
