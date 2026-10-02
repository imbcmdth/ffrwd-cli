# Node shapes on the command line

A module built against `ffrwd:av@0.19.1` exports `node`. Its ports and
clock depend on its params and on which inputs a call binds, so the
sidecar asks the module and prints the answer as JSON.

## Describe

`ffrwd-wasm --describe <module>` on a node module prints the usual
description with two differences:

- `"world": "node-module"`. Every module of an older world still reports
  `"world": "ffrwd:av@0.18.0"`, the newest world of the interfaces the
  node replaces.
- `"node": true`. The key is absent for every other module.

`params_schema`, `rows_schema`, `rows_language`, `name` and `version`
are the node's `describe()`. `pixel_formats` and `sample_formats` are
empty: a node's ports say what they accept. `inputs` and the window
fields mean nothing for a node; read the shape.

A node module may also export `values`, `encoder` and `decoder`; those
fields are filled in as for any module. A node beside a frame, packet,
rows or data-filter interface is refused.

## Shape

    ffrwd-wasm --shape <module> [--params <json> | --params-from <file>] [--bound <port,port,...> | --bound <json>]

- `--params`: the call's static params, one JSON object. Absent is the
  empty string, which is what a module with no params is handed.
- `--bound`: the inputs the call binds, as the WIT's `list<binding>`:
  a JSON array of `{"input": "<name>", "streams": [{"rate": {"num": n,
  "den": d}}, ...]}`, one `binding` per input and one stream hint per
  stream bound to it, in the order the call names them. `rate` is a video
  stream's frame rate or an audio stream's sample rate over 1, `null` (or
  absent) where nothing settles it before the run; `streams` absent is
  one stream with no rate. Or the comma list of the names alone, each one
  stream with no rate, a name written again one stream more (`v,v,v`
  binds three to `v`). Absent or empty binds none. A name the shape does
  not declare is refused, and so is an input named twice in the JSON, an
  empty `streams`, a rate that is not positive and a key not listed here.
  The run is handed the list each call was asked with as `-bound`
  (`NODE-CLI.md`, "The shape a call was planned with"), and asks the
  shape with it again.
- `-http`, `-udp`, `-tcp` and `-gpu` grant effects the way a run's argv
  does, for a source that reads the network to answer.

Exit 0 prints one JSON line on stdout. Exit 1 prints `ffrwd-wasm: ...` on
stderr: the module's own refusal (`<name> refused the shape: <its
message>`), or the host's, naming the module and the port. The host
refuses what `shape`'s doc in `wit/av.wit` lists:

- `clock` names no input, or one that is optional, many or not lockstep;
- a lockstep input on a node with no input clock;
- an input paired other than `arrival` on a self-clocked node;
- `hold` on a data or packets input, `interval` on a video or audio one;
- a data input whose `rows` is `ignore`;
- `like` (on an output, or in `accepts`) naming an input that is not
  declared, takes many streams, is not bound, or carries no frames;
- a stride of 0 or above its window;
- `wants` `timing` on a data or packets input;
- an `interval` naming a group no hold input declares, or naming one with
  an anchor other than `shared_clock`;

and, beyond that list: a window other than 1/1 on an input that is not
the clock, a port name declared twice, a rate clock that is no rate.

## Encoding

The JSON is the WIT's `node-shape` record:

- Field names are the WIT's, in snake_case (`one-to-one` is `one_to_one`,
  `port-param` is `port_param`).
- An enum is its case as a string, snake_case: `kind` is `"video"`,
  `"audio"`, `"data"` or `"packets"`; `rows` is `"ignore"`,
  `"per_frame"` or `"state"`; `wants` is `"all"`, `"keyframes"`,
  `"first"` or `"timing"`.
- A variant is an object whose `kind` names the case. A case carrying a
  record has that record's fields beside `kind`; a case carrying anything
  else has it under the name below.
- `option<T>` is the value or `null`. `list<string>` is an array.
- `schema` and `relation` stay strings, as the WIT has them: a schema is
  JSON Schema text, a relation row is JSON object text.
- A rational is `{"num": n, "den": d}`.
- Object keys arrive in no particular order.

| Variant | Cases |
|---|---|
| `pairing` | `{"kind":"lockstep"}`, `{"kind":"hold", "anchor", "lead", "linger", "timeout", "group", "port_param"}`, `{"kind":"interval", "latency", "ahead", "anchor", "group"}`, `{"kind":"arrival"}` |
| `anchor` | `{"kind":"shared_clock"}`, `{"kind":"first_frame"}`, `{"kind":"tagged", "tag": "<name>"}` |
| `clock` | `{"kind":"input", "port": "<name>"}`, `{"kind":"rate", "num", "den"}`, `{"kind":"rate_of", "port": "<name>"}`, `{"kind":"self_clocked"}` |
| `output-format` | `{"kind":"video", "width", "height", "pix_fmt", "color"}`, `{"kind":"audio", "sample_rate", "channels", "sample_fmt", "channel_layout"}`, `{"kind":"data", "codec": "json"}`, `{"kind":"packets", ...coded-stream}`, `{"kind":"like", "port", "pixel_format", "sample_format"}` |
| `coded-format` (in a coded stream) | `{"kind":"video", "width", "height", "sample_aspect_ratio", "color"}`, `{"kind":"audio", "sample_rate", "channels", "channel_layout"}`, `{"kind":"data"}` |

`color` is `{"range", "primaries", "trc", "space"}` or `null`. A coded
stream is `{"codec", "time_base", "format", "extradata", "profile",
"level"}` with `extradata` in lowercase hex, as `--probe` prints it.

An output's `format` of `null` is the clock input's format, and a
`time_base` of `null` the clock input's time base; the compiler resolves
both, and `like`, against the streams it binds.

## Example

`modules/shape-probe`, a test module whose shape touches every field:

    ffrwd-wasm --shape shape_probe.wasm --params '{"canvas":{"width":640,"height":360}}'       --bound '[{"input":"v","streams":[{"rate":{"num":25,"den":1}}]},{"input":"feed"},{"input":"words"},{"input":"a"}]'

Printed on one line; here indented, keys in the WIT's order:

```json
{
  "inputs": [
    {
      "name": "v",
      "kind": "video",
      "required": true,
      "many": false,
      "pairing": {"kind": "lockstep"},
      "rows": "per_frame",
      "window": 1,
      "stride": 1,
      "accepts": {
        "pixel_formats": ["yuv420p", "rgba"],
        "sample_formats": [],
        "sample_rates": [],
        "channel_counts": [],
        "codecs": [],
        "wants": "all",
        "like": null
      },
      "schema": null
    },
    {
      "name": "feed",
      "kind": "video",
      "required": false,
      "many": false,
      "pairing": {
        "kind": "hold",
        "anchor": {"kind": "tagged", "tag": "smart_timed"},
        "lead": 0.5,
        "linger": 1.0,
        "timeout": null,
        "group": "feeder",
        "port_param": "port"
      },
      "rows": "ignore",
      "window": 1,
      "stride": 1,
      "accepts": {
        "pixel_formats": [],
        "sample_formats": [],
        "sample_rates": [],
        "channel_counts": [],
        "codecs": [],
        "wants": "all",
        "like": null
      },
      "schema": null
    },
    {
      "name": "words",
      "kind": "data",
      "required": false,
      "many": true,
      "pairing": {
        "kind": "interval",
        "latency": 2.0,
        "ahead": 0.25,
        "anchor": {"kind": "first_frame"},
        "group": null
      },
      "rows": "state",
      "window": 1,
      "stride": 1,
      "accepts": {
        "pixel_formats": [],
        "sample_formats": [],
        "sample_rates": [],
        "channel_counts": [],
        "codecs": [],
        "wants": "all",
        "like": null
      },
      "schema": "{\"type\":\"object\",\"properties\":{\"text\":{\"type\":\"string\"}}}"
    },
    {
      "name": "a",
      "kind": "audio",
      "required": false,
      "many": false,
      "pairing": {"kind": "lockstep"},
      "rows": "ignore",
      "window": 1,
      "stride": 1,
      "accepts": {
        "pixel_formats": [],
        "sample_formats": ["f32"],
        "sample_rates": [],
        "channel_counts": [],
        "codecs": [],
        "wants": "all",
        "like": null
      },
      "schema": null
    },
    {
      "name": "cues",
      "kind": "data",
      "required": false,
      "many": false,
      "pairing": {
        "kind": "interval",
        "latency": null,
        "ahead": 0.0,
        "anchor": {"kind": "shared_clock"},
        "group": "feeder"
      },
      "rows": "per_frame",
      "window": 1,
      "stride": 1,
      "accepts": {
        "pixel_formats": [],
        "sample_formats": [],
        "sample_rates": [],
        "channel_counts": [],
        "codecs": [],
        "wants": "all",
        "like": null
      },
      "schema": null
    },
    {
      "name": "size",
      "kind": "video",
      "required": false,
      "many": false,
      "pairing": {"kind": "lockstep"},
      "rows": "ignore",
      "window": 1,
      "stride": 1,
      "accepts": {
        "pixel_formats": [],
        "sample_formats": [],
        "sample_rates": [],
        "channel_counts": [],
        "codecs": [],
        "wants": "timing",
        "like": null
      },
      "schema": null
    }
  ],
  "outputs": [
    {
      "name": "mask",
      "kind": "video",
      "format": {"kind": "like", "port": "v", "pixel_format": "gray", "sample_format": null},
      "time_base": null,
      "latency": 0.0,
      "schema": null,
      "row": null
    },
    {
      "name": "copy",
      "kind": "video",
      "format": {"kind": "like", "port": "v", "pixel_format": null, "sample_format": null},
      "time_base": null,
      "latency": 0.0,
      "schema": null,
      "row": null
    },
    {
      "name": "canvas",
      "kind": "video",
      "format": {"kind": "video", "width": 640, "height": 360, "pix_fmt": "rgba", "color": null},
      "time_base": null,
      "latency": 0.0,
      "schema": null,
      "row": null
    },
    {
      "name": "spots",
      "kind": "data",
      "format": {"kind": "data", "codec": "json"},
      "time_base": {"num": 1, "den": 1000000},
      "latency": 0.48,
      "schema": "{\"type\":\"object\",\"properties\":{\"start_t\":{\"type\":\"number\"}}}",
      "row": null
    }
  ],
  "clock": {"kind": "input", "port": "v"},
  "pure": true,
  "one_to_one": false,
  "bounded": true,
  "relation": []
}
```

`spots` is as late as twelve of `v`'s frames, 0.48 s at the 25/1 the
binding gives `v`; with `--bound v,feed,words,a`, which says no rate, it
is 0.5. The same module with `--params '{"rate":25}'` and nothing bound
has `"clock": {"kind": "rate", "num": 25, "den": 1}`, holds `v` and `a`,
and drops `mask` and `copy`, since an output `like` an unbound input is
left out.

## Modules of older worlds

`--shape` on a module of an older world prints the shape the host runs
it as, in the same encoding, so every module reads through one record. A
0.18 detector's rows reach a node over a data edge, and a node's rows
reach a 0.18 reader the same way. `--describe` still reports what the
module itself declares.

| Old world | Adapted shape |
|---|---|
| filter, meta-filter, window-filter | Inputs `in0`, `in1`, ... in the order of the module's stream arguments. The first that is not a feeder is the clock, with the module's window and stride; the others are lockstep with it. A feeder is a hold input: `anchor` `first_frame`, `port_param` the param its port is written into, `group` its group. Where the module reads rows, `rows`: data, optional, lockstep, `per_frame`. Outputs `out`, in the clock's format, and, where the module declares a rows schema, `rows`: data, one message per row at its frame's pts, in the clock's time base. `pure` and `one_to_one` are the module's; a per-frame filter says whether it is pure only once opened, so it is `false` here. |
| packet sink | `video` and `audio` (packets) and `data` (data), each where the sink's arity has one: a many-port, required where it reads one or many, by arrival, `accepts.codecs` and `wants` the sink's. Clock `rate` 50/1: called at least every fiftieth of a second. No outputs: its rows are the node's own (`@rows`). |
| packet filter | The sink's inputs, and `rows` (data, many, by arrival, required where the filter reads rows). Self-clocked. An output named after each coded input port, with no format: every stream bound to that input leaves on it, in the order bound, its header the input's with the extradata the filter answered. |
| packet source | No inputs; self-clocked. An output `track0`, `track1`, ... per catalog track `probe` answers for `--params`, `packets` with the track's coded stream or `data`; `row` indexes `relation`, which holds one object per catalog row (`row`, `name`, `bandwidth`, `codecs`, `language`); `bounded` is the catalog's. |
| data filter | `data` (data, many). With `clock` bound: `clock` (video, lockstep) is the clock, and `data` pairs by interval with `latency` 0. Without: self-clocked, `data` by arrival. Outputs `out0`, `out1`, ... per stream it writes, in its declared time base. |
| rows module | `rows` (data, required, the clock), its declared input schema; `out` (data), its rows schema. |

Where the node world says it differently, the host keeps what each
world's host did: rows ride their frames between older modules, a data
filter is handed every message at or before its clock rather than those
from its tick on, a packet filter writes one stream per stream it reads,
and a data filter's clock pad may be video, audio or coded although the
port says video.

`modules/facebox`, a 0.10 window-filter that reads rows and writes them:

    ffrwd-wasm --shape facebox.wasm --bound in0,rows

```json
{
  "inputs": [
    {
      "name": "in0",
      "kind": "video",
      "required": true,
      "many": false,
      "pairing": {"kind": "lockstep"},
      "rows": "ignore",
      "window": 1,
      "stride": 1,
      "accepts": {
        "pixel_formats": ["yuv420p", "rgba"],
        "sample_formats": [],
        "sample_rates": [],
        "channel_counts": [],
        "codecs": [],
        "wants": "all",
        "like": null
      },
      "schema": null
    },
    {
      "name": "rows",
      "kind": "data",
      "required": false,
      "many": false,
      "pairing": {"kind": "lockstep"},
      "rows": "per_frame",
      "window": 1,
      "stride": 1,
      "accepts": {
        "pixel_formats": [],
        "sample_formats": [],
        "sample_rates": [],
        "channel_counts": [],
        "codecs": [],
        "wants": "all",
        "like": null
      },
      "schema": null
    }
  ],
  "outputs": [
    {
      "name": "out",
      "kind": "video",
      "format": null,
      "time_base": null,
      "latency": 0.0,
      "schema": null,
      "row": null
    },
    {
      "name": "rows",
      "kind": "data",
      "format": {"kind": "data", "codec": "json"},
      "time_base": null,
      "latency": 0.0,
      "schema": "{\"type\":\"object\",\"properties\":{\"x\":{\"type\":\"integer\"},\"y\":{\"type\":\"integer\"},\"w\":{\"type\":\"integer\"},\"h\":{\"type\":\"integer\"}},\"required\":[\"x\",\"y\",\"w\",\"h\"],\"additionalProperties\":false}",
      "row": null
    }
  ],
  "clock": {"kind": "input", "port": "in0"},
  "pure": false,
  "one_to_one": true,
  "bounded": true,
  "relation": []
}
```

A module that exports no stream interface (values or a codec alone) has
no shape, and `--shape` says so.
