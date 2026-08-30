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

So subtitles do not ride the NUT edge. The sidecar gathers the cue rows -
they are all known by the end of the stream - and writes a finished subtitle
document to an output of its own.

## The argv

An output is `-f <format> <path>`, and in a network a `-map` names the label
it writes, exactly as an `-f nut` or `-f ndjson` output does. `-` and
`pipe:1` are stdout; anything else is an ordinary path.

    -map [out0] -f srt    subs.srt
    -map [out0] -f webvtt subs.vtt

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

## The rows a subtitle output reads

A cue row is a JSON object with all three of:

    start_t  number, seconds from the start of the stream
    end_t    number, seconds from the start of the stream
    text     string

Cues are written in the order the rows arrive, numbered from 1. A row object
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

The language tag is the compiler's to write; the sidecar does not put one in
the document.

`mov_text` shows one cue at a time: where two cues in the document overlap,
ffmpeg clips the earlier one at the later one's start. Cues that do not
overlap keep their spans exactly.

A stream that produced no cues leaves an empty document. An empty `.srt` is
zero bytes and ffmpeg refuses to open it (`Invalid data found`); an empty
`.vtt` is its `WEBVTT` header alone and ffmpeg reads it as a stream with no
cues.
