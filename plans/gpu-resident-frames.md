# GPU-resident frames

Decode, compose, infer and encode on one GPU inside a sidecar region, with
raw frames never crossing a pipe and never entering a module unless it asks
for the bytes. Target: 4K60 in real time through ffrwd/blitz's `compose`
over several held inputs, and through a model on the GPU, on the hosted
runner's L4.

ffmpeg keeps containers, protocols, probing, software codecs and the whole
CPU filtergraph. Only a region holding a module that reads frames on the GPU
grows to include decode and encode, and only the packets cross its edges.

This file is the working plan. It names files and functions as they stand
today so each wave can be dispatched on its own; where a design detail is
still a sketch it says so. The experiment of section 3 has run
(`plans/experiment-compose-4k60-results.md`): the design's assumption
holds, and this revision folds in what it measured and what a survey of
the hardware the sidecar will meet changed.

## 1. Where the time goes today

A raw video frame from ffmpeg's decoder to ffmpeg's encoder, through one
module that uses the GPU, is copied in full at every row below.

| stage | copies | where |
|---|---|---|
| ffmpeg packs rawvideo into NUT and writes the pipe | 2 to 3 | ffmpeg |
| sidecar reads the pipe through a 1 MiB reader into 64 KiB chunks | 2 | `sidecar/ffrwd-wasm/src/edges.rs:59`, `:99` |
| NUT demuxer assembles the payload in its own buffer | 1 | ffrwd-nut |
| payload becomes the held frame | 1 | `sidecar/ffrwd-wasm/src/node_graph.rs:355` |
| `fetch` clones the Vec, then the canonical ABI copies it into wasm memory | 2 | `sidecar/runtime/src/runtime/node_world.rs:172` |
| `write-buffer-with-copy` lifts the bytes out of wasm, wgpu stages them | 2 + DMA | `sidecar/vendor/wasi-webgpu-wasmtime/src/trait_impls.rs:1120` |
| `map-async`, `get-mapped-range-get-with-copy` to a Vec, then into wasm | 2 | `trait_impls.rs:2657` |
| the emitted frame is lifted out of wasm | 1 | `node_world.rs:550` |
| NUT muxer writes through a buffered writer into the pipe | 2 | `edges.rs:417` |
| ffmpeg reads the pipe and decodes rawvideo | 2 to 3 | ffmpeg |

Eighteen to twenty copies per frame, plus two more per edge the relay
carries (`sidecar/ffrwd-wasm/src/relay.rs`). The `wasi:nn` path has the same
shape: the tensor is lifted from wasm, cloned at
`sidecar/vendor/wasmtime-wasi-nn/src/backend/onnx.rs:291`, copied into an
ONNX Runtime tensor, uploaded to CUDA, and comes back the same way. No I/O
binding.

| 4K frame | bytes | at 60 fps, one copy | at 60 fps, 18 copies |
|---|---|---|---|
| yuv420p | 12.4 MB | 0.75 GB/s | 13 GB/s |
| rgba | 33.2 MB | 2.0 GB/s | 36 GB/s |
| h264 at 40 Mbit/s | 0.08 MB | 5 MB/s | n/a |

The compiler puts a module's first accepted pixel format on the wire
(`cli/ffrwd/processes.py:4833`), and most modules list rgba first. Each
worker opens a wgpu device of its own, which is why GPU regions are capped
at two workers (`cli/ffrwd/wasm.py:420`).

Compositing is worse, because it scales with inputs. ffrwd/blitz's
`compose` holds its `inputs` on loopback ports. A lateral is its own ffmpeg
writing rawvideo over TCP (`cli/ffrwd/lower.py:1836`); the listener copies
each frame, converts yuv420p to rgba, resizes, and converts back
(`sidecar/ffrwd-wasm/src/feeds.rs:98`), to the clock input's size
(`node_graph.rs:1744`), so a 1080p lower third is upscaled to 4K on the CPU
before the document shrinks it. A held frame that repeats across ticks is
fetched again every tick. blitz then rasterizes the whole 4K canvas on the
CPU with Vello: its README measures a rendered 1080p frame at 10.6 to
18.0 ms on one worker, so a 4K frame is 40 to 70 ms, and 4K60 needs three
to five workers each fetching every drawn input in full.

Programme plus four feeds plus output at 4K60 rgba, about eight copies
each, is 95 GB/s of memory traffic. That exceeds the machine. It is not a
tuning problem.

The experiment measured the two halves of that on an integrated Adreno,
through the real module path (vello_hybrid in a module, wasi:webgpu
underneath), at N of 8 inputs, 4K:

| path | CPU a frame | GPU a frame | across the wit | fps |
|---|---|---|---|---|
| pictures resident on the GPU, render only | 1.6 ms | 4.9 ms | 1.4 MB | 126 |
| pictures uploaded every frame (today's input wire) | 53 ms | 11.7 ms | 92 MB | 13 |
| resident, plus a 4K readback each tick (today's output path) | 1.9 ms + 10.4 ms copy | 11.6 ms | 35 MB | 37 |

The upload is 25x the resident variant's CPU; the readback caps the run
at 37 fps whatever N is, and costs more than the rendering. Today's output
path is the bound, so handing the frame back as a texture (wave 3) is
worth more than trimming the raw path (wave 0).

## 2. Design

### 2.1 The shape of a run

    ffmpeg (demux, -c:v copy) --packets over NUT--> sidecar region:
        decode (host, libavcodec, NVDEC) -> GPU frame
        -> module reads it by handle (wasi:webgpu texture), draws or infers
        -> emits a GPU frame by handle, or passes one through (`same`)
        -> encode (host, libavcodec, NVENC) -> packets
    --packets over NUT--> ffmpeg (mux, -c:v copy)

A module that wants the bytes calls `fetch`, as today; that is the one
download. A region whose exit feeds something only ffmpeg can do (a software
encoder, a CPU filter) downloads once at its boundary and writes the raw wire
as today, which is still a fraction of the current cost. A region whose entry
is fed by something that cannot stream-copy (a filtered picture, a lateral
that renders) takes the raw wire in and uploads once.

Ownership: one wgpu device per sidecar process, created by the host on the
platform's native API (D3D12 on Windows, Vulkan on Linux and Android, Metal
on macOS), handed to every module's `wasi:webgpu` and to every worker; the
decoder, the encoder and the inference provider on the same physical device
through that API's own sharing; frames in a pool the host owns, referenced
by handle, released by credit.

Zero transfer, not zero copy. Every hardware decoder emits NV12 or P010
and pyrowave emits YUV, so one device-local pass per input, in the
compositor's sampling or in a conversion, is the floor; what the design
removes is every trip through system memory and every byte across a pipe
or the wit. On unified-memory parts the readback the experiment measured
is untiling into uncached memory, not a bus transfer, so zero transfer
there means staying in tiled, cached allocations.

### 2.2 The world: `ffrwd:av@0.20.0`

Additive on 0.19.1; the sidecar keeps hosting every older world with host
memory, as it does today. A sketch, to be settled in wave 3:

- `node-types.accepts` gains `memory: frame-memory`, `enum frame-memory
  { host, gpu }`, default `host`. A `gpu` input port's `pixel-formats`
  name what the frames are handed as: `nv12` and `p010`, the decoder's own
  planes as two textures (r8 and rg8, or r16 and rg16) with the stream's
  colour, which a compositor samples directly and which costs no pass; or
  `rgba`, which the host converts once per source frame for a module that
  asks. `output-port.format` gains the same field for an output the module
  writes on the GPU; an output written as `rgba` is converted ahead of an
  encoder that takes NV12 only.
- A new interface, `gpu-frames`, imported only by modules that read frames
  on the GPU, so `node-tick` and `node` stay free of the webgpu dependency:

      interface gpu-frames {
        use wasi:webgpu/webgpu@0.0.1.{gpu-texture};
        use node-tick.{tick};
        /// Frame `index` of stream `id` this tick, as a texture of the
        /// guest's device: the host's memory, valid for the call.
        texture: func(tick: borrow<tick>, id: u32, index: u32) -> result<gpu-texture, string>;
        /// Hands a texture the module rendered to the host, which takes it;
        /// the token names it in `payload`.
        emit-texture: func(texture: gpu-texture) -> u32;
      }

  and `node.payload` gains `gpu-frame(record { pts, duration, token: u32 })`.
  `same-frame` is unchanged and works on GPU frames. `fetch` on a `gpu`
  port downloads, and is allowed.
- `wasi:webgpu`'s `request-adapter` and `request-device` answer the host's
  device, so a module's textures and the host's frames live on one device.
  A module's `required-features` and limits are checked against it.
- `ffrwd/wasm@0.20.0` published; `ffrwd-node` SDK gains the `gpu` port
  builder and a `Tick::texture` helper.
- The `wasi:webgpu` the sidecar hosts is no longer upstream's 0.0.1 as
  published: the experiment added `gpu-command-encoder.write-timestamp`
  and `gpu-queue.get-timestamp-period`, and filled `get-compilation-info`
  and the texture getters. It is versioned as ffrwd's own package, a
  superset of 0.0.1, and a guest binds the sidecar's vendored copy by
  path so guest and host cannot drift.

### 2.3 Host nodes

Three nodes the host answers for, following `sidecar/ffrwd-wasm/src/host_nodes.rs`:

- `decode`: one `packets` input, one video output in `gpu` memory (or
  `host` with no hardware device). Options: `hw` (cuda, d3d12va,
  videotoolbox, vaapi, none; default: what the machine has), `format`
  (what the reader asked for). Output latency is the stream's
  `decode-delay`. Its `coded-stream` comes from the NUT header, as a
  packet filter's does.
- `encode`: one video input (`gpu` or `host`), one `packets` output with
  `coded-stream` and extradata for the NUT header, written the way a codec
  package's packets are (`sidecar/ffrwd-wasm/src/codec.rs`). Options are
  the COPY's codec options as the compiler lowers them (`video_codec`,
  `crf`, `preset`, `gop`, `profile`, `level`, `video_bitrate`, `maxrate`,
  `bufsize`, `keyint_min`, `codec_params`), mapped to libavcodec's
  AVOptions by the same names ffmpeg's CLI uses.
- `convert`: GPU colour and size conversion the host runs for itself,
  never named in a query: NV12 or P010 to rgba8 when a `gpu` port asks for
  rgba, rgba8 to NV12 ahead of an encoder that takes nothing else, resize
  for `accepts.like`, and the unpack of a capture card's packed 4:2:2.
  WGSL on the host device, one pass per source frame, cached on the frame
  so every reader shares it. A compositor that samples the decoder's
  planes itself needs none of it, which is the common case.

### 2.4 libavcodec in the sidecar

Loaded at run time, never linked, the way ONNX Runtime is
(`sidecar/runtime/src/nn.rs`, `nn/dlls.rs`): `libavcodec` and `libavutil`
only. The sidecar has NUT already and ffmpeg keeps every container, so no
`libavformat`. A dlopen shim over the forty-odd functions needed (codec
lookup, context, send/receive, hw device and frames contexts, bitstream
filters, option setting), written against the pinned version's headers.

- No setup step. The libraries ship inside each platform's sidecar wheel
  beside the binary, and the sidecar finds them by its own path, so
  `pip install ffrwd` stays the whole install. Argv `-av-runtime <dir>`,
  else `FFRWD_AV_RUNTIME`, else the wheel's own directory, so a checkout
  can point at a build of its own. `ffrwd-wasm --av-info` answers the
  version the binary demands and where it found it, as `--nn-info` does.
  A wheel cannot run code at install time, so bundling is what makes
  this seamless; ONNX Runtime, at 72 MB to 1.4 GB and chosen by the
  machine's driver, stays a first-use fetch (2.9).
- The builds are ours and LGPL only: `--disable-everything`, the decoders
  and parsers for h264, hevc, av1, vp9, mpeg2, prores, the hardware
  codecs (`--enable-ffnvcodec` for nvdec and nvenc, which dlopen the
  driver's own libraries; `d3d12va`, `d3d11va`, `mediafoundation` and
  `amf` on Windows; `videotoolbox` on macOS; `vulkan` and `vaapi` on
  Linux; `mediacodec` on Android), no `--enable-gpl`, no
  `--enable-nonfree`, no external encoders. Dynamic linking keeps the
  LGPL obligation to shipping the library and its licence.
- The hardware device is the sidecar's own, handed to libavcodec through
  `av_hwdevice_ctx` on each platform: the D3D12 device wgpu created on
  Windows (`d3d12va`, with `d3d11va` and Media Foundation on the adapters
  that have no D3D12 decode, the Snapdragon among them), the Vulkan device
  on Linux (`vulkan` decode on NVIDIA, which writes straight into
  VkImages; `vaapi` elsewhere, imported through DMA-BUF), VideoToolbox on
  macOS. A CUDA context is created beside the Vulkan device only where an
  inference provider needs one, matched by UUID.
- Decoder and encoder sessions are a capacity, not a count of engines:
  integrated parts cap concurrent sessions (the experiment's Snapdragon
  decodes 4K120 and encodes 4K60, or one 4K60 decode beside two 4K30
  encodes), and the compiler's placement reads the cap per machine the
  way it reads NVDEC and NVENC engines today.

### 2.5 The frame pool and interop, per platform

- `TickFrame.data` and `OutFrame.data` (`sidecar/runtime/src/node.rs:362`,
  `:423`) become an enum: `Host(Arc<Vec<u8>>)` or `Gpu(Arc<GpuFrame>)`. A
  `GpuFrame` is one slot of a pool: a native image the host allocated on
  the wgpu device, its planes, format and colour, the handle the decoder
  or encoder wrote or reads it through, and a refcount the lanes' credit
  reads (`sidecar/ffrwd-wasm/src/lanes.rs`).
- The backend is chosen at the start, per platform, and the interop is
  that API's own; the matrix replaces the earlier Vulkan-and-CUDA sketch:

  | platform | device | decode into the pool | encode from it | the guest's texture |
  |---|---|---|---|---|
  | Windows | D3D12 | `d3d12va` and `d3d11va` textures, shared handles across the two APIs; Media Foundation where libavcodec has no hwaccel for the part | NVENC, AMF, QSV and Media Foundation from D3D12 or D3D11 textures | wgpu-hal dx12 wraps the resource |
  | Linux | Vulkan | `vulkan` decode writes VkImages directly on NVIDIA; `vaapi` surfaces imported as DMA-BUF | NVENC from a CUDA import of the exported image where Vulkan encode is absent; VAAPI from the DMA-BUF | wgpu-hal vulkan wraps the image |
  | Android | Vulkan | AHardwareBuffer import | MediaCodec from the buffer | the same |
  | macOS | Metal | VideoToolbox IOSurfaces | VideoToolbox from the IOSurface | wgpu-hal metal wraps the texture |

  `runtime/src/gpu.rs` tries Vulkan before D3D12 today; on Windows the
  order flips, since every decoder there is a D3D11 or D3D12 object.
- Synchronisation is the API's own: its signals and shared semaphores where the
  decoder and the device are two objects, a stream sync on the CUDA
  import in the first cut, timeline semaphores later. A device-to-device
  copy into a pool slot stays as the fallback for a decoder that cannot
  write a shared image (about 0.05 ms for 12 MB on an L4).
- A texture for the guest: `wgpu_core`'s `create_texture_from_hal` over
  the slot's image, one view per plane, pushed into the module's
  wasi:webgpu resource table for the tick, the way `trait_impls.rs` holds
  every texture by its core id. Handed back through `emit-texture`, the
  resource's core id names the slot, and the host takes it.
- Bounds: a pool is counted in frames, not bytes, eight to sixteen per
  stream, with the TCP and credit backpressure that `hold.rs`'s
  `MAX_HELD_FRAMES` and `MAX_HELD_BYTES` give today; a port feed that
  outruns its reader waits on its socket as now. A 4K slot is 12 MB as
  NV12, so an L4's 24 GB holds hundreds, but a `page` rendering ahead of
  the `compose` holding it must be held back by credit, which it is not
  today (blitz README, limitations).
- Warm-up: the first frame costs 100 ms, 850 ms with eight 4K inputs, in
  pipeline compilation and first uploads. A run absorbs it before its
  clock starts: the host ticks a GPU region once with its first frames
  held and starts the clock on the second.

### 2.6 The guest: wgpu over wasi:webgpu

This is code. `wgpu-wasi-webgpu`, on the blitz repository's
`bench-compose-4k60` branch under `backend/`, implements wgpu 29's
`custom` dispatch traits over wit-bindgen bindings to the sidecar's
vendored wit, scoped to what vello_hybrid calls: about 1,200 lines with
the conversions; render bundles, surfaces, acceleration structures,
binding arrays, pipeline caches, immediates, multi-draw and timestamps
inside passes are `todo!()`. The 0.0.1 wit's copy-based mapping maps onto
wgpu's mapped ranges directly: a read mapping is the Vec
`get-mapped-range-get-with-copy` returns, a write mapping goes back
through `get-mapped-range-set-with-copy` when wgpu drops the range. Every
list crossing the wit is counted, which is where the experiment's `wit`
column came from. Any wgpu-based crate runs in a module on it unchanged;
what remains is the rest of the traits, tests, and a repository of its own.

Two things a module on it needs: wgpu without its `webgpu` default
feature on a wasm32 target (that feature compiles the browser backend,
whose wasm-bindgen imports a wasip2 component cannot carry; the blitz
branch vendors wgpu with that one change until the crate takes a feature
for it), and the guest's wgpu pinned to what its renderer pins, with the
host's wgpu-core version irrelevant behind the wit.

blitz on it: `anyrender_vello_hybrid` 0.10.0 matches blitz's pins exactly,
so nothing in blitz changed for the renderer; two methods were added to
blitz core so a picture can be handed back under the same blob id, which
is what the hybrid renderer caches an upload by. The product version of
`compose` renders to a texture, draws each held input from
`gpu-frames.texture` as an external texture, and hands the output back
through `emit-texture`; the probe, bypass and everything above the
rasterizer stay as they are. Measured per-frame traffic across the wit:
1.4 MB at N of 8.

### 2.7 The compiler

- `--shape` carries `memory` on `accepts` and on an output's `format`;
  `cli/ffrwd/shapes.py` parses it (`Accepts`, `OutputFormat`).
- Partition (`cli/ffrwd/processes.py`): a region whose entry port is `gpu`
  and whose feeding stream is a source's own coded stream takes the edge
  coded: the `VideoFormat` on the edge keeps the source's `codec` and the
  producing ffmpeg writes `-c:v copy`, the way an edge into a packet sink
  already carries a codec (`VideoFormat.codec`, `processes.py:523`), and
  the region's filtergraph gains `decode` ahead of the module. A region
  whose exit feeds a COPY whose `video_codec` the sidecar can host on the
  machine (an `_nvenc` codec today) gains `encode` and the consuming ffmpeg
  writes `-c:v copy`; otherwise the region's output edge is raw, as now.
  `_ENCODER_SHAPING` (`cli/ffrwd/lower.py`) is the option set the encode
  node is handed.
- Laterals (`lower.py:1836`): the feeder COPY writes `video_codec 'copy'`
  where the source is a file or a stream-copyable input, a hardware encoder
  where the lateral renders and the machine has one, and rawvideo at the
  source's own size otherwise. The listener (`feeds.rs:328`) accepts coded
  video and routes it through `decode`. A `gpu` hold port is never conformed.
- Placement (`cli/ffrwd/placement_cost.py`): a region holding `decode` and
  `encode` counts one decode and one encode session each against a cap
  that is per machine and per part, the NVDEC and NVENC engines it models
  today being one row of it; `gpu_processes` in `placement.py` treats such
  a region as on a GPU.
- Grants: a module with a `gpu` port imports `wasi:webgpu` and needs `-gpu`
  as today; a region with `decode` or `encode` is spawned with
  `-av-runtime`. `GPU_JOBS` (`wasm.py:420`) stops meaning anything once the
  device is the host's; measured, then removed.
- Recipes before implementation: each wave starts as entries in
  `docs/examples.md`, red, and `docs/dialect.md` and `docs/known_gaps.md`
  say what changed.

### 2.8 Determinism and tests

Outputs stay the same bytes at every `-jobs`: the scene a worker renders
is the same commands, and the decode and encode are one instance each.
Hardware decode differs from software decode by rounding, so the cookbook's
byte-checked recipes keep the software path, and GPU recipes are checked on
the GPU runner with a per-path pin. CI stays on software: the unit tier and
the exec tier run as today; the GPU tier is nightly on a machine with a card.

### 2.9 Install and first use

One step: `pip install ffrwd`, or `uvx ffrwd`. Both install wheels, and a
wheel cannot run code at install time; an sdist with a `setup.py` hook
could, but pip, uv and pipx build it in isolation with no promise of
network, no progress display and a cached result, so a hook there fails
silently and once. The seamless moment is therefore first use, which is
how the project already works: static-ffmpeg fetches ffmpeg on the first
call that needs it (`cli/ffrwd/binaries.py:125`), and the ONNX Runtime
tiers are fetched by the first run that reaches a model (`cli/ffrwd/cli.py:1861`)
and by `ffrwd install` of a package whose modules run one
(`cli/ffrwd/packages.py:1564`), each saying what it fetches and how big.

What changes:

- libavcodec needs no fetch at all, since it ships in the sidecar wheel.
- `ffrwd setup` with no argument fetches everything detection wants on
  this machine, ffmpeg and the ONNX tiers, for a container image or a
  machine about to lose its network; `setup nn` stays as the narrower
  spelling. `FFRWD_OFFLINE=1` refuses any fetch with a message naming it.
- A run that will fetch starts the download while it probes its inputs
  rather than after compiling, so the first query's wait overlaps work
  it was doing anyway.

## 3. The experiment: blitz at 4K60 on the GPU

The design rested on one assumption nothing had measured: that
vello_hybrid, driven from a module through wasi:webgpu, composites a 4K60
document over several video textures inside the frame budget, with the
guest's single-threaded CPU share small enough to leave room for the rest
of the tick. `plans/experiment-compose-4k60.md` is the brief;
`plans/experiment-compose-4k60-results.md` is the run, on an integrated
Adreno X1-85 under Windows on aarch64, 2026-10-10, native and then in a
module under `ffrwd-wasm` with its `-gpu` grant.

The pass line was written for a 4090; on the integrated part it reads:

| pass line, N of 8, render only | measured, wasm | |
|---|---|---|
| guest CPU under 8 ms a frame | 1.6 ms (p99 3.5) | holds, 5x margin |
| GPU under 8 ms a frame | 4.9 ms median, p99 11.7 | holds at the median |
| under 16 MB a frame across the wit | 1.4 MB | holds, 11x margin |
| 60 fps held over 600 frames | 126 fps, one miss (frame 0) | holds |
| 60 fps with a 4K readback each tick | 37 fps | does not hold here |

So the assumption stands on the hardware weakest for it, and neither
failure branch is taken: the GPU is not over budget, so vello_hybrid stays
the compositor for the video layers and the picture count barely moves
the cost (N of 2 to 8 adds 0.4 ms of CPU and 0.5 ms of GPU); the guest CPU
is not over budget, so strip generation stays in the guest. The wasm run
costs 1.35x the native one per frame, 0.7 ms of it the strip and alpha
lists crossing the wit and 1.1 ms the host's own submit path. The picture
is right: a frame from the module differs from the native dump in 3 of
8.3 million pixels, by one level, rasterisation rounding.

The line that fails is the one the design exists to retire: the readback
of the 4K frame each tick costs more than rendering it, 11.6 ms of GPU and
10.4 ms of guest copy with 33 MB across the wit, and today's output path
sits at 37 fps whatever N is. The upload variant says the same of the
input side at 53 ms of CPU and 92 MB a frame. GPU-resident frames are the
difference between 13 and 126 fps on this machine.

What carries forward is in the results file's last section and folded in
above: the per-platform matrix (2.5), the host's wit additions and the
backend as code (2.2, 2.6), YUV planes as the frames a compositor samples
(2.2, 2.3), the warm-up (2.5), session caps as a capacity (2.4, 2.7), the
inference matrix (wave 4), and two candidates to measure (wave 5). The
same harness runs unchanged on the 4090 and the L4, where the GPU and
readback columns should fall by several times and the CPU columns stay.

## 4. Waves

Each wave is dispatchable on its own and leaves the tree green. Numbers are
estimates of the sidecar-side diff; the compiler side is listed with it.

### Wave 0: measure, then trim the raw path

No interface change. Everything here is worth shipping alone, and none of
it is the bound: the experiment put today's limit on the output copy,
which only wave 3 removes, so this wave is scheduled behind waves 1 to 3
and done where hands are free.

1. **Profile rows.** Per stage, bytes copied and time: pipe read, demux,
   hold, fetch, process, lift, mux write. Rows behind `ffrwd:row` as the
   relay's `flow` rows are, summed at the end of the run, on by
   `--profile`. Files: `edges.rs`, `node_graph.rs`, `node_world.rs`,
   `lanes.rs`, `heartbeat.rs`. The 4K60 budget becomes visible.
2. **`fetch` without the clone.** Replace the generated binding at
   `node_world.rs:151` with a `func_wrap` that lowers the held `&[u8]`
   straight into guest memory. One copy instead of two.
3. **Read the pipe into the demuxer.** Drop the 1 MiB `BufReader` at
   `edges.rs:59` and feed the demuxer from reads of its own, in chunks the
   size of a frame's remainder; have ffrwd-nut hand over the payload
   (`take_payload`) instead of lending it so `node_graph.rs:355` holds it
   without copying. Two copies fewer.
4. **Bigger pipes.** Size stdio pipes the way the relay sizes named ones:
   `F_SETPIPE_SZ` up to `PIPE_BUFFER_LIMIT` on Linux. Fewer syscalls per
   frame, no copy change.
5. **One device.** `runtime/src/gpu.rs`: the shared instance becomes a
   shared adapter and device; the vendored host's `request-adapter` and
   `request-device` answer it. Measure pyrowave at `-jobs 32` again; lift
   `GPU_JOBS` if the contention is gone.
6. **blitz on its own.** Reuse the output buffer instead of allocating per
   frame (`core/src/session.rs`); accept yuv420p for `v` and convert in
   the guest, halving the programme's wire.
7. **ffmpeg's own GPU path, chosen by the compiler.** No sidecar in it.
   On the pinned build 70 of 512 filters take hardware frames, each tied
   to one API (11 `_cuda`, 17 `_vulkan`, 19 `_opencl`, 13 `_vaapi`, 7
   `_qsv`, 3 `_amf`, and `libplacebo`); the other 440 are CPU and
   `hwdownload` stands between. When an input is decoded with a hwaccel
   and the COPY's encoder is a hardware one, the compiler substitutes the
   hardware variant for the common filters between them (`scale`,
   `overlay`, `pad`, `transpose`, the flips, `yadif`, `bwdif`) and sets
   `-hwaccel_output_format`, so an ffmpeg-only query stays on the card
   from decode to encode; a filter with no variant gets one `hwdownload`
   and one `hwupload` around the whole CPU segment, never around each
   filter. Placement already classes such a process as a GPU job.

Acceptance: at most ten copies per frame on the raw path at
`-jobs 1`, measured by the profile rows; 4K30 rgba passthrough through an
`invert` module at real time on the owner's machine.

**Shipped (2026-10-12, on this branch, from the cloud session):** items
1 to 5 on the node-world path.

- `-profile` (`runtime/src/profile.rs`, `NODE-CLI.md`): one row at the
  end of the run with bytes, calls and seconds per stage: `read`, `held`,
  `fetch`, `process`, `emit`, `upload`, `download`, `write`. The counters
  are process-wide atomics and always on; the row is written only when
  asked. `tests/gpu_frames.rs` pins the stages a frame crosses in host
  memory.
- The pipe is read into the demuxer (`edges.rs`): no `BufReader`; where
  the demuxer is waiting on a frame's payload with at least a chunk still
  wanted, the pipe is read straight into that payload
  (`PushDemuxer::payload_space`), else a 64 KiB chunk is fed; and the
  payload is taken out of the demuxer (`take_payload`) rather than
  copied, so a frame off the wire becomes the held frame with no copy
  of its own.
- `fetch` lowers the held bytes straight into the module's memory
  (`node_world.rs`, `Held`, `shadow_fetch`): the generated binding's
  `Vec<u8>` return copied the frame once before the canonical ABI copied
  it again; the shadow is a `func_wrap` over the same interface name that
  lowers the `Arc<Vec<u8>>` as a slice. One copy where there were two.
- Stdio and fifo pipes are asked for 1 MiB on Linux (`main.rs`,
  `widen_pipe`), clipped to `/proc/sys/fs/pipe-max-size`, the way the
  relay sizes its own.
- Item 5 is wave 3's shared device; `GPU_JOBS` stays until the local
  session's measurement.

On the node-world raw path at `-jobs 1` the sidecar's own copies are now:
pipe read into the demuxer's payload (1), `fetch` into wasm memory (1),
the emitted frame lifted out of wasm (1), the mux write through the 1 MiB
`BufWriter` into the pipe (2). The older-world path (`main.rs`,
`older.rs`, `adapters.rs`) and the codec runner now read the same way
(2026-10-12): ffrwd-nut's blocking `Demuxer` already read each payload in
place and handed it over, so the extra copies there were the 1 MiB
`BufReader` in front of it, now gone; one input type counts `read` for
every path, and a test holds the older path's `read` and `held` bytes
equal to the node path's on the same input. The older path keeps the
blocking demuxer for what the push one does not do: ADTS AAC (deriving
the AudioSpecificConfig and stripping the headers) and the annotation
rows. The node-network edges have no ADTS handling, which a coded AAC
input to a node network would need. Item 6 is blitz's.

Checked here: the unit tier, `gpu_frames`, `node_shapes`, `ffrwd_wasm`,
`av`, `data_streams`, `packets`, `rows_module` and `relay`. The `live`
suite fails in this container on the tree before this wave and before
wave 3's slice alike (the feed shows for 35 frames where 50 to 62 are
expected), so it is the container's timing, not the change; it is the
local session's to run. The `-profile` row is the measurement item 1
asked for; the copy counts above are read off the code, not yet off the
row on a 4K run.

**Shipped (2026-10-12, item 7):** `cli/ffrwd/hardware.py`, a pass after
the split: an input with `hwaccel => 'cuda'` whose picture reaches an
`_nvenc` destination is told `hwaccel_output_format cuda`; `scale`,
`overlay`, `pad`, `transpose`, `yadif` and `bwdif` become their `_cuda`
variants where the call names only options the variant takes (`scale`'s
`width`/`height` spelled `w`/`h`); `split` and the timing filters ride on
the card; every other filter runs on the CPU as one segment with one
`hwdownload` ahead and one `hwupload_cuda` behind where card work follows;
a software encoder, a module or a second file reading a card frame gets
one download. Where the compile knows its ffmpeg, the pass runs only if
that build has the variants; where it knows none, the command is written
for the reader's. Recipe 156, `docs/dialect.md` (input options) and
`docs/known_gaps.md` say so. The Vulkan family followed (`hwaccel
vulkan` to a `_vulkan` encoder, the `_vulkan` filters, `hwupload`; the
encoder reads only the card's frames, so a CPU segment ahead of it ends
with an upload); the families never mix. Neither has run on a card from
here: the pinned commands are what ffmpeg is handed, and a run on the
4090 (CUDA) and the L4 (Vulkan) is what proves them.

**Measured on the owner's machine (9950X, RTX 4090, Windows,
2026-10-12):** the `-profile` row, 1080p60 h264 in, 600 frames, through
`ffrwd-wasm` alone. `held`, `fetch` and `emit` count frames handed on,
so a 1080p yuv420p frame is 3.1 MB in each; `upload` and `download` are
the host's own transfers.

| chain | fps | held | fetch | emit | upload | download |
|---|---|---|---|---|---|---|
| software decode, `gpu-box` on bytes, `h264_nvenc` | 249 | 1,200 | 600 | 600 | 0 | 0 |
| d3d12va decode into the pool, `gpu-box` reading planes, emitting bytes, `h264_nvenc` | 182 | 1,200 | 0 | 600 | 0 | 0 |
| d3d12va decode into the pool, `h264_d3d12va` from the pool | 235 | 600 | 0 | 0 | 0 | 0 |

The device-to-device chain moves no frame byte through the host at
all. At 1080p on this CPU it is no faster than the host chain, which
the CPU keeps up with; the encoder bounds it. The middle row is the test
module reading its planes back through its own `wasi:webgpu` mapping
(which the host's `download` does not count) to draw on bytes, so it
measures the test module, not the path. The 4K and many-input runs, where
section 1 says the host path cannot keep up, are the numbers still to
take, with blitz's texture path.

Recipe 155 (320x240, 60 frames, through `ffrwd run --profile`): `held`
120 calls, `fetch` 60 and `emit` 60 at 307,200 bytes a frame, the edges'
`read` and `write` the coded packets only: four full-frame stages on the
coded path, as wave 2's acceptance counted.

Item 7 on the card (CUDA, 1080p and 4K, `h264_nvenc`): the commands the
pass wrote failed in ffmpeg 9.0.1 wherever a CPU segment needed a
download ("Invalid output format yuv420p for hwframe download"):
`hwdownload` must be followed by a `format` naming the frames' software
format, and a list does not work, since ffmpeg negotiates before the
frames context exists. The pass now names the stream's own (`nv12` for
8-bit 4:2:0, `p010le` for 10-bit) and leaves an input it cannot name off
the card. Timed against the same query without the pass: a scale alone
on the card and on the CPU take the same time at 1080p (0.98 s against
0.99 s, 600 frames) and at 4K (1.59 s against 1.64 s, 300 frames); a CPU
segment costs the pass about 10% for its upload (1.18 s against 1.01 s at
1080p). Decode and encode bound both here, so the pass pays where the CPU
filters are heavy or the CPU is weak, not on this machine.

### Wave 1: wgpu over wasi:webgpu, finished

The experiment's backend made a product. Independent of the sidecar
changes; can run in parallel with wave 2.

1. **The crate's own repository** beside `ffrwd-node`, moved from
   `backend/` on the blitz branch, pinned per wgpu release with the
   renderer that uses it; wgpu's `webgpu` default feature handled by a
   feature of the crate rather than a vendored wgpu.
2. **The rest of the traits.** Compute passes, index buffers, render
   bundles, query sets inside passes and the remaining `todo!()`s, so a
   module that is not vello_hybrid runs too.
3. **The host's wit as ffrwd's package.** The additions the experiment
   made in `sidecar/vendor/wasi-webgpu-wasmtime` (`write-timestamp`,
   `get-timestamp-period`, compilation info, texture getters) versioned as
   a superset of 0.0.1 and published with `ffrwd/wasm`, so a guest binds a
   package and not a path.
4. **Tests.** wgpu's `hello-compute` and `hello-triangle` rendering to a
   texture, built for `wasm32-wasip2`, run as value modules through
   `ffrwd-wasm --invoke` on a `-gpu` grant; vello_hybrid's test scenes
   rendered to a texture and compared against `vello_cpu` within a
   tolerance; the experiment's bench kept as the GPU tier's benchmark.
5. **blitz on `anyrender_vello_hybrid` as a release.** `compose` renders
   to a texture and downloads it (`map-async` on a readback buffer) until
   wave 3 hands textures back; the CPU path stays when no `-gpu` grant is
   given. The two blitz core methods from the experiment land in main.

Acceptance: the experiment's harness runs on the GPU tier from the
published crates, and the lbar recipe renders on the GPU path with frames
within tolerance of the CPU path.

**Shipped (2026-10-11):** `wgpu-wasi-webgpu` is a repository of its own
(github.com/imbcmdth/wgpu-wasi-webgpu, version 29.0.0 at tag `v29.0.0`,
pinned to wgpu 29, with its own copy of the wit under `wit/deps` and a
script that checks it against ffrwd-cli's), moved out of blitz's
`bench-compose-4k60` branch, which keeps the bench. Every `todo!()` is gone: render bundles, indirect draws, the
compressed texture formats and the WGSL language features are implemented
over the wit, and what the 0.0.1 interface cannot express (surfaces,
acceleration structures, mesh shaders, binding arrays, pipeline caches,
immediates, pipeline statistics, `clear-texture`) refuses by name. The
host's wit additions ship in `ffrwd/wasm` under `wit/deps` under the
upstream package name. Tests on the GPU: wgpu's hello-compute and
hello-triangle as value modules through `ffrwd-wasm --invoke` with a `-gpu`
grant, and a vello_hybrid scene against vello_cpu, all passing on the
owner's RTX 4090 (0 of 262,144 pixels differ by more than 2 levels; the
largest difference is 1).

What was learned:

- No crate feature can replace the vendored wgpu: wgpu 29's build script
  turns on its browser backend for every wasm32 target except Emscripten
  whenever its `webgpu` feature is set, and `anyrender_vello_hybrid` 0.10
  depends on wgpu with its defaults, which Cargo unifies into the whole
  build. The backend crate takes wgpu with defaults off and needs no patch;
  a module that uses that renderer keeps a `[patch.crates-io]` wgpu with
  the one line removed until wgpu excludes `target_os = "wasi"` from that
  cfg upstream. `vello_hybrid` 0.1 alone keeps the defaults off behind its
  `wgpu_default` feature, so the backend's own vello test needs no patch.
- Render bundles are implemented but unreachable: wgpu 29's own
  `RenderBundleEncoder::finish` is unimplemented for every custom backend.
- A host with no adapter now returns `RequestAdapterError` rather than
  panicking, which is what a module's CPU fallback keys on. The grant is
  not the fallback's signal: the compiler emits `-gpu` for every module
  whose describe imports `wasi:webgpu`, so "no GPU" means no adapter; a
  sidecar built without the `gpu` feature refuses such a module outright.
- A write mapping starts zeroed and is written back whole on drop; a
  program that reads a mapped buffer back before writing it sees zeros.

blitz 0.3.0 is on branch `gpu-compose` of ffrwd-package-blitz, unpublished
until the ffrwd release after 0.29.5 hosts the host's wit additions:
`compose` and `page` render through `anyrender_vello_hybrid` into a
texture and read it back when `renderer` is `gpu`, or `auto` with an
adapter; the two core methods are in. The GPU and CPU frames agree within
the tolerance on every frame but the ones where the programme is being
scaled, where the CPU path's `Extend::Repeat` seam (the last row and
column taking colour from the first) differs from the GPU's, so the test
holds those frames to a looser limit; the GPU frame is the one without
the seam. Measured on the owner's 9950X and 4090, the lbar example
through `ffrwd-wasm`, 360 frames: with every core the CPU path still wins
(1080p 115 fps against 95, 4K 35 against 29) because each frame comes
back by readback and each picture goes up by copy, which is what wave 3
removes; on one worker the GPU path wins (1080p 74 against 71, 4K 19
against 16). So 0.3.0's default `renderer` is `cpu`, and the default
moves to `auto` with wave 3. Two costs to carry into wave 0 and wave 3:
the compiler grants `-gpu` to this module whatever `renderer` says and
caps the sidecar at two workers (`GPU_JOBS`), which leaves the CPU path a
third slower at 1080p and 40% slower at 4K than a run on every core, so
lifting the cap (wave 0, item 5) matters before the release. The crate
is github.com/imbcmdth/wgpu-wasi-webgpu, and blitz pins it by tag
(`v29.0.0`).

### Wave 2: libavcodec in the sidecar, packets on the wire

Software frames first: this wave removes the rawvideo pipe and ffmpeg's
rawvideo copies on both sides before any GPU frame exists.

1. **The shim.** `sidecar/runtime/src/av/`: dlopen of `libavcodec` and
   `libavutil` by full path from `-av-runtime`, version check against the
   pinned major, the function table, `--av-info`. Refusals name the
   directory looked in and the reinstall hint `cli/ffrwd/wasm.py` already
   carries for a missing sidecar, as `nn.rs` names `ffrwd setup nn`.
2. **The builds, in the wheel.** A workflow in a new repository produces
   the LGPL-only shared libraries for the release matrix
   (`.github/workflows/release.yml`: manylinux x86_64 and aarch64, musl,
   macOS, Windows), each release pinned by URL, sha256 and byte count.
   The sidecar's release workflow fetches the pinned set for its target
   before `maturin build` and packages it beside the binary
   (`sidecar/ffrwd-wasm/pyproject.toml`, maturin's `include`), with the
   LGPL text. `cli/ffrwd/binaries.py` learns where the wheel put it the
   way it finds the sidecar binary. No setup command for it.
3. **`decode` and `encode` host nodes** with host-memory frames, in
   `host_nodes.rs` or files beside it, software codecs through the shim.
   Both are one instance, in order, as `codec.rs` runs a codec package.
   The hwaccel per platform from 2.4 is wired here even though frames stay
   in host memory until wave 3, so the session caps (2.4) are learned and
   the compiler's capacity model has numbers.
4. **The wire.** `processes.py` partition takes coded edges into and out
   of a region that declares it (a module option in this wave, the
   `memory` field in the next); `-c:v copy` on both ffmpegs. `feeds.rs`
   accepts coded video; the lateral template writes `copy`.
5. **Docs.** `sidecar/AV-EDGE.md` for the argv and the refusals;
   `NODE-CLI.md` for the two nodes; `docs/dialect.md`.
6. **Tests.** `sidecar/ffrwd-wasm/tests/av.rs`: h264 packets in,
   frames out byte-equal to ffmpeg's own decode of the same file (same
   libavcodec build, same bytes); `encode` packets muxed by ffmpeg and
   decoded back; a cookbook recipe whose compiled command shows the copy
   on both sides.

Acceptance: the raw pipe is gone from a region that declares coded edges;
copies per frame on that path at most four (decode output, fetch, lift,
encode input).

**Shipped (2026-10-11, on this branch):** the shim
(`sidecar/runtime/src/av/`, `-av-runtime`, `--av-info`), the `decode` and
`encode` host nodes with a port feed that arrives coded
(`sidecar/ffrwd-wasm/src/av_nodes.rs`, `feeds.rs`), the compiler's
partition over `SET ffrwd.edges = 'coded'` (`cli/ffrwd/av.py`,
`processes.py`, `functions.py`, `lower.py`), the wheel packaging
(`sidecar/ffrwd-wasm/fetch_av.py`, `av-pins.json`, the release workflow),
the builds repository github.com/imbcmdth/ffrwd-av, and the docs
(`sidecar/AV-EDGE.md`, `NODE-CLI.md`, `docs/dialect.md`, recipe 155).
Recipe 155 ran on the owner's x86_64 machine: 60 h264 frames copied in,
decoded in the sidecar, inverted, encoded by NVENC, copied into an mp4,
timestamps intact.

What was learned, for the waves after:

- The pin is FFmpeg 9.0 (avutil 61, swresample 7, avcodec 63, swscale 10),
  and libswscale is bundled: the decoder's native format and the wire's
  differ for everything but yuv420p, and the encoder's input likewise.
  The shim converts straight between the wire's packed buffer and the
  codec's frame, so a format change costs no pass of its own.
- The builds carry no VA-API: libavutil links libva at load time when it
  is enabled, and a wheel cannot assume libva on the machine. Vulkan is
  opened at run time and covers NVIDIA on Linux; the native av1 decoder
  decodes only through a device, so there is no software AV1. The
  musl and macOS archives carry software codecs and VideoToolbox.
- The wire's own table names h264, hevc and av1; every other codec
  travels under ffmpeg's riff tag and the shim maps the tag
  (`av/fourcc.rs`). `encode` codes in `pix_fmt` else the encoder's first
  format, so lossless ffv1 over rgba wants `pix_fmt=bgra`.
- The declaration is per function (`SET ffrwd.edges = 'coded'`), and the
  partition is a pass over the region's edges after the regions are built
  (`_code_edges`), for old-world and node-world regions alike; the coded
  input of a node region carries no colour pad, since the decoder has the
  stream's. Wave 3's `memory` field replaces the clause.
- The consuming ffmpeg loses `-copyts` on a coded exit: the packets keep
  the clock their input had, and the region's decoder reads them as they
  arrived. The hosted encoder's delay counts in the region's lookahead, but
  `encoder_delay`'s table has no entry for a hosted encoder yet, so today
  it counts zero.
- The decode node's output header is settled before the first frame, so
  its colour is what NUT carries (range and matrix); primaries and
  transfer are unknown until `convert` reads them off the codec in wave 3.
- `hw=auto` engaged d3d11va on the owner's machine with bytes identical to
  the software decode; session caps were not measured, only the engaged
  device is reported, so the compiler's capacity model still has no
  numbers from this wave.
- `av-pins.json` pins ffrwd-av v1.1.0 on all seven wheel platforms (v1.0.0
  until the d3d12va encoders were added); the
  Windows x86_64 archive was fetched through `fetch_av.py` and loaded by
  the shim here. A platform without a pin would ship without codecs, and
  the sidecar then refuses `decode` and `encode` naming the directory it
  looked in.

### Wave 3: GPU frames

Unblocked by the experiment; the first thing it buys is the output path,
which is today's bound.

1. **World 0.20.0** as in 2.2; `ffrwd/wasm@0.20.0`; `ffrwd-node` builders;
   the host adapts every older world as `host`.
2. **The pool.** `node.rs` enum; `GpuFrame` with planes; pool per stream
   with the count bound; credit in `lanes.rs`; `hold.rs` holds handles.
   Source nodes held by a hold input are held back by credit (the `page`
   into `compose` case). The warm-up tick before the clock starts.
3. **The backend per platform** (2.5): D3D12 first on Windows in
   `runtime/src/gpu.rs`, Vulkan on Linux, Metal on macOS; the decoder's
   images wrapped as the pool's slots on each, with the device-to-device
   copy as the fallback. Linux on the L4 and Windows on the owner's
   machines are the two done first; macOS follows on the same shape.
4. **Textures for the guest.** `gpu-frames.texture` over
   `create_texture_from_hal`, one view per plane; `emit-texture`;
   `payload::gpu-frame`; `same` over GPU frames.
5. **`decode` and `encode` on hardware.** `hw` option; the pool's slots as
   the decoder's output and the encoder's input through each platform's
   frames context; fallback to wave 2's software path when the machine has
   no device, which is CI.
6. **`convert`** for the cases 2.3 lists; the compositor samples NV12 and
   P010 planes itself, so a composite of decoded video runs no conversion.
6a. **`filter`, a fourth host node: libavfilter's hardware subset on the
   pool.** An ffmpeg GPU filter chain next to a region cannot share the
   region's memory, so the filters a query puts beside a GPU module run
   inside it. On Linux libavfilter's Vulkan device context is built over
   the pool's own instance, device and queues, so the `_vulkan` filters
   and libplacebo run on the pool's images with no interop; the `_cuda`
   filters reach the pool through the same external-memory import the
   inference providers use, on Linux and on the Windows D3D12 pool, where
   no `_d3d12` filters exist; this build has no VideoToolbox filters. The
   compiler maps a query's `ffmpeg.<name>` to the variant the pool has,
   the node reports what engaged, and a filter with no variant on the
   machine means a download at that filter's input, never the region.
7. **The compiler.** `shapes.py` reads `memory`; partition routes by it;
   `placement_cost.py` counts sessions against the per-machine capacity;
   the lateral template picks a hardware encoder where one is; `GPU_JOBS`
   removed.
8. **blitz.** Held inputs from `gpu-frames.texture` as YUV planes drawn by
   vello_hybrid's external-texture path with the matrix in the sampling,
   output through `emit-texture`, bypass by `same`. The experiment's
   document at N of 8 is the benchmark, with real decoded inputs.
9. **Tests.** A test module beside `modules/gpu-probe` that reads a GPU
   frame, draws on it, hands it back; byte-checked on the GPU tier against
   a pinned output per platform; the profile rows showing zero host-side
   frame copies on the path.

Acceptance: the benchmark recipe runs at real time on the L4 and on the
owner's Windows machine with the encoder as the only thing above 50% of
its engine; no raw frame bytes in system memory on the path, by the
profile rows; the readback and upload rows of the experiment's table gone.

**Shipped, first slice (2026-10-12, on this branch, from the cloud
session; not yet run on a machine with a GPU):** items 1, 2 and 4 in a
form that keeps the node world at 0.19.1, and the compiler half of 7.

- One device per process. `runtime/src/gpu.rs` opens one wgpu device the
  first time anything asks (`shared_device()`: the platform's preferred
  backends in order, the features the pool and the filters want), and the
  vendored host answers every store's `request-adapter` and
  `request-device` with it, refusing a request whose features or limits
  exceed the device's. Dropping a module's device handle drops nothing.
  A machine with no adapter answers `request-adapter` with none, as
  before, and `--shape` and the host-memory path still run.
- The pool. `runtime/src/gpu/frames.rs`: `GpuFrame`, one texture per
  plane in the formats the wire carries (rgba as `rgba8unorm`; yuv420p,
  yuv422p, yuv444p as `r8unorm` planes at their own size), uploaded from
  host bytes or adopted from a texture a module made. `bytes()` downloads
  once and keeps the copy, so a frame leaving on a raw edge, or fetched
  by a module as bytes, costs one readback and no more.
- Frames carry where they are. `FrameData::{Host, Gpu}` on tick frames
  and output frames (`runtime/src/node.rs`); every reader of frame bytes
  in `ffrwd-wasm/src` goes through `bytes()`, so older worlds, raw
  edges, holds and `same` work unchanged over a GPU frame. The shape
  carries `memory` on each input's `accepts` and on each output
  (`host` unless said otherwise), in the JSON and in
  `cli/ffrwd/shapes.py`.
- The companion package, not a world bump. `sidecar/wit-gpu/gpu.wit` is
  `ffrwd:gpu@0.1.0`: `ports.memory(params)` the module exports beside
  `shape`, and `frames.planes`/`frames.emit` it imports. A module
  includes `ffrwd:gpu/gpu-module` beside `ffrwd:av/node-module@0.19.1`;
  the host binds the package only on a module that exports `ports`
  (`runtime/src/runtime.rs`, `node_world.rs`). `ffrwd/gpu@0.1.0` is the
  registry package (`sidecar/wasm-gpu`), staged by the release workflow
  the way `ffrwd/wasm` is. Section 2.2's `0.20.0` is not needed for
  this; it stays for the day the node world itself changes.
- Lane intake uploads. A stream bound to an input held on the GPU is
  uploaded as it arrives at the lane (`lanes.rs`, `Lane::resident`),
  so a module sees textures today over a raw edge, with the decoder's
  interop (items 3 and 5) to replace that upload with the decoder's own
  images. A module emitting a texture hands it to the host, which
  adopts it as the frame and lets it leave after the items `process`
  returned on that port.
- Tests. `modules/gpu-box` (node world plus `ffrwd:gpu`, `memory=gpu`
  or `host`) and `ffrwd-wasm/tests/gpu_frames.rs`: the shape says which
  ports are held where, the host path draws the box, and the GPU path
  is checked against it byte for byte where an adapter exists, skipping
  with a note where none does. Here there is none, so the GPU case is
  unrun; it is the local session's to run first.

Of item 7, the partition now routes a region by its entry port's memory
(`cli/ffrwd/processes.py`, `_holds_on_the_gpu`; the lateral's copy in
`lower.py` the same way): a node module whose shape holds an input on the
GPU takes coded edges as one declared `SET ffrwd.edges = 'coded'` does,
with no clause, since the frames go onto the device either way (recipe
157, `docs/dialect.md`). Shipped once item 5 had the decoder's frames in
the pool on Windows. Items 6, 6a and 8 are open. The pool holds no per-frame colour:
a frame's colour is its stream's, as the NUT header has it, and the
decode node settles that header before the first frame.

**Run on the owner's machine (RTX 4090, Windows, 2026-10-12):**
`gpu_frames` passes all three cases, the GPU path byte-equal to the host
path; `node_shapes` stays green. Two things the first slice did not see
without a GPU:

- The shared device opens on D3D12 first, and wgpu's D3D12 backend
  reports no `subgroups` without a DXC it can load, so pyrowave's compute
  module refused to open on it ("The adapter does not support the
  subgroups feature"); the host used to try Vulkan first for exactly
  that. The fix is the static DXC compiled into the vendored host
  (wgpu-core's `static-dxc`, `Dx12Compiler::StaticDxc` in
  `shared_device`), in flight with items 3 and 5.
- `GPU_JOBS`, measured with the pyrowave round trip (300 frames of 1080p
  yuv420p, `bytes_per_frame` 60000). With a device per worker (the binary
  before this slice): 227 fps at one worker, 252 at two, 190 at eight,
  240 at the default. With the shared device, on Vulkan (`WGPU_BACKEND`,
  until D3D12 has subgroups): 202 to 212 at one, 249 to 272 at two, 223
  to 282 at eight, 287 at the default, output byte-identical. So the
  shared device removes the loss past two workers but the GPU is the
  bound and nothing scales past two; the machine was under load from
  three builds during the runs. The quiet run on D3D12 with DXC
  (`WGPU_DX12_COMPILER=dxc`, 1200 frames, two rounds): 161 fps at one
  worker, 237 to 247 at two, 154 to 228 at four, 223 to 233 at eight,
  208 to 257 at the default. The contention the cap existed for is gone,
  so `GPU_JOBS` is removed: a GPU module takes the run's workers like any
  other, which also gives blitz's CPU path its cores back.

**Items 3 and 5, Windows (2026-10-12, owner's machine):** `decode` opens
`d3d12va` around the shared device's own `ID3D12Device` when a reader
holds its frames on the GPU (`hw` auto or `d3d12va`, `format=yuv420p`),
and the pool takes the decoder's NV12 resource as a wgpu texture through
`create_texture_from_hal` with no copy (`runtime/src/gpu/d3d12.rs`,
`av::Decoder::open_shared`, `receive_any`, `DeviceFrame`). The shared
queue waits on the decoder's sync object before any read and signals it
back when the frame is released; a texture array slice, a device without
`TEXTURE_FORMAT_NV12` or a refused wrap falls back to a device-to-device
copy into pool textures, which was exercised by forcing it and is
byte-equal. Modules still read three `r8unorm` planes: a small device
pass splits the chroma once per frame and caches it
(`gpu/convert.rs`), so the `ffrwd:gpu` contract did not change. On the
4090 every stream tried (the 64x48 fixture, an odd 318x238 one, 1080p
with B-frames) wrapped, and the device decode is byte-identical to the
software decode. `encode` reads a GPU frame back once: this build's
`h264_nvenc` takes `cuda` and `d3d11` frames and no `d3d12`, while the
build does carry `h264_d3d12va`, `hevc_d3d12va` and `av1_d3d12va`
encoders, which are the next step for an encode from the pool with no
readback.

What was learned:

- The decoder never runs out of frames when the host holds them: the
  d3d12va frame pool grows on demand, and array mode only when asked.
- P010 is not held on the GPU (the wire has no 10-bit format), and
  `decode=format=rgba` into a GPU reader decodes as before and uploads at
  the lane, since NV12 to rgba is `convert`'s pass and would not match
  swscale byte for byte.
- A wrapped frame can be read by the host on the direct queue while the
  decoder reads it as a reference on its video queue; both only read,
  but in different states, which the D3D12 debug layer may flag. Not run
  under the debug layer yet.
- Linux needs: the shared Vulkan device opened with the video-decode
  extensions and a decode queue family through wgpu-hal's raw open, the
  `AVVulkanDeviceContext` filled from it, `AVVkFrame`'s image, layout and
  timeline semaphore read, a wait on the semaphore through a raw submit
  (wgpu-core's submit has no semaphore waits), the `VkImage` wrapped as
  `G8_B8R8_2PLANE_420`; VAAPI through a DMA-BUF import. The structure is
  split by `cfg` so a `gpu/vulkan.rs` and an `av::Shared::Vulkan` slot
  in beside the D3D12 ones.
- DXC: wgpu-hal links the static DXC only for x86_64 MSVC, so the
  feature is on for that target alone (a build with it on elsewhere
  panics at instance creation). The prebuilt library needs ATL to link,
  which this machine's build tools lack, so the MSVC binary's size
  change is unmeasured (an estimate is 15 to 20 MB); CI's first Windows
  wheel run proves it. A gnullvm or aarch64 Windows build keeps D3D12
  first and needs `dxcompiler.dll` on the path for subgroups.

**A GPU in CI (2026-10-12):** the sidecar workflow's Linux job installs
mesa's lavapipe, a Vulkan driver that runs on the CPU, and runs
`gpu_frames` with `FFRWD_GPU_REQUIRED` set, so a run that finds no
adapter fails instead of skipping. The first run passed with the GPU case
taken: the shared device, the frame pool, the planes a module reads and
the texture it hands back go through wgpu's real code on every push. It
does not prove speed, any vendor's driver or its limits, or the hardware
decoder and encoder: the five D3D12 cases skip on Linux, and the Vulkan
decoder has no test until it exists. The workflow can be dispatched on a
branch by hand.

**Items 5 and 6 for the encoder (2026-10-12, owner's machine):**
`h264_d3d12va` and `hevc_d3d12va` take their frames from the pool: the
encoder is opened on the shared device with an NV12 frames context, each
pool frame is copied once on the device into the encoder's frame (one
`CopyResource` where the decoder's resource matches, else a copy per
plane), the queue waits on the frame's sync object and signals the
encoder's, and nothing reaches the host. A module's rendered rgba frame
is made NV12 first by a WGSL pass (`gpu/convert.rs`: BT.709 or BT.601,
limited or full, from the stream's colour; ffmpeg's chroma filter, a
horizontal pair average and an 8-row bicubic with a = -0.6, measured off
an impulse), within one level of ffmpeg's own conversion at every pixel.
Every other encoder reads a GPU frame back once (`h264_nvenc` in this
build takes `cuda` and `d3d11` frames, no `d3d12`). The compiler lists
the two encoders as hosted and as hardware sessions.

What was learned:

- These encoders in FFmpeg 9.0.1 on NVIDIA's driver fail on the first
  B-frame (ffmpeg's own command line too), so `bf=0` is set unless asked;
  and they write no extradata, so a second encoder codes one picture
  ahead of the stream to take the parameter sets for the header.
  `av1_d3d12va` does not open on this card at all ("Codec sequence
  initialisation failed: -22"), and the encoder refuses 64x48.
- The pinned LGPL builds did not carry the d3d12va encoders; ffrwd-av
  v1.1.0 adds `h264_d3d12va` and `hevc_d3d12va` to both Windows builds.
- 1080p, decode into the pool into `h264_d3d12va`: 235 fps with zero
  upload and download (wave 0's table).

**Item 6, `convert` (2026-10-12, owner's machine):** one conversion
path on the device, `runtime/src/gpu/convert.rs`, and one entry point on
the frame, `GpuFrame::converted(Target)`. A target is a layout (rgba,
yuv420p, yuv422p, yuv444p, NV12), a size, and the matrix and range of a
conversion between rgba and YUV (BT.709 or BT.601, limited or full). The
result is a frame of its own, kept on the source frame under its target,
so every reader asking for the same thing shares one conversion.
`wire_planes` and the encoder's NV12 are callers of it, and the NV12
chroma split, the rgba-to-NV12 pass and the yuv420p interleave are cases
of it rather than functions of their own. It is one WGSL module with one
bind group layout (a uniform holding the sizes and the colour, and three
textures), its pipelines made per entry point the first time each is
asked for. Each distinct conversion says so once on stderr (`[gpu]
convert nv12 1920x1080 to rgba 1920x1080`), and the profile row has a
`convert` stage.

Each pass was measured against FFmpeg 9.0.1's `scale` on a 64x48 picture
with ramps, hard edges, a red block and a transparent corner, and the
numbers are pinned in the tests in `gpu/frames.rs`:

- YUV (NV12, yuv420p, yuv422p, yuv444p) to rgba, at the picture's size,
  smaller and larger, in BT.709 limited and BT.601 full: at most 1 level
  apart, mean at most 0.026, against `flags=bilinear+full_chroma_int+accurate_rnd`.
  ffmpeg's default conversion is not the reference: at the same size it
  takes its unscaled converter, which repeats each chroma sample across
  two pixels and lands 60 to 100 levels from an interpolated result at a
  hard chroma edge.
- rgba to yuv420p at its own size: at most 1, mean 0.051 (BT.709 limited)
  and 0.196 (BT.601 full), against ffmpeg's default bicubic, whose chroma
  filter the NV12 pass already follows. At another size each plane is
  resampled from the rgba directly: at most 1, mean at most 0.178, against
  `bilinear+full_chroma_inp+accurate_rnd`. Resizing first and converting
  after was 10 to 15 levels off.
- rgba to NV12, the existing pass: at most 1 (it was pinned at 2), means
  0.035 for luma and 0.488 for full-range chroma, where ffmpeg rounds
  half a level apart.
- Resizing rgba, yuv420p, NV12 and yuv444p to 40x30, 32x24, 96x72 and
  17x13: at most 1, mean at most 0.096, against
  `bilinear+full_chroma_inp+full_chroma_int`. Without the chroma flags
  ffmpeg halves rgba's chroma across before it scales, 26 levels off at an
  exact half. NV12 and yuv420p made into one another are exact.

The routes, each engaged in `ffrwd-wasm/tests/gpu_frames.rs`:

- `decode=format=rgba` read by an input held on the GPU now decodes on
  the shared device too (`av::ON_DEVICE`). The frame is held as the
  decoder's NV12 and read as rgba, converted once per picture in the
  stream's matrix and range (`Yuv::read_as`: BT.601 where the stream says
  nothing, as libswscale reads it). In
  `pictures_decoded_on_the_device_reach_an_rgba_reader_converted_there`
  stderr says `[gpu] convert nv12 64x48 to rgba 64x48`, the `convert`
  row has one call per picture, `upload` is 0 and `download` is the
  output alone; the pictures are within 1 of ffmpeg's, mean 0.018. A
  stream in another matrix keeps the old path, read back and uploaded.
- An input held on the GPU with `accepts.like` is told the followed
  input's size at `init` and resized at the lane (`Lane::resident`). In
  `an_input_that_follows_another_is_resized_on_the_device` stderr says
  `[gpu] gpu_box: resizing the frames of stream 1 on the GPU to 32x24`,
  within 1 of ffmpeg, mean 0.047. A bound stream in host memory is still
  handed over at its own size, since the host never conformed one.
- A port feed into an input held on the GPU uploads the feeder's picture
  as it came and conforms it on the device (`feeds.rs`, `Picture::frame`);
  the CPU conform stays for host-memory ports. In
  `a_feed_held_on_the_gpu_is_conformed_on_the_device` stderr says
  `feed: gpu_box 'w': the feeder's picture is 64x48 rgba, conformed to
  32x24 rgba on the GPU`, within 1 of ffmpeg, mean 0.047.
- The encoder's NV12 comes from `converted` (`EncoderFeed::feed`); the
  three `h264_d3d12va` tests pass as they were.

`modules/gpu-box` gained the second input these use: `w=pad` binds `w`
to a stream, `w=port` feeds it on a port, and either way it follows `v`.

What was learned:

- Chroma is sited at the centre of the luma it covers, as libswscale
  assumes when a frame does not say. The h264 fixture reports
  `chroma_location=left`, yet ffmpeg's `scale` with its default
  `in_chroma_loc` matched the device within 1; forcing `left` moves
  ffmpeg up to 116 levels at a chroma edge. Honouring a stream's siting
  would be a quarter-texel offset in the shader's resampling, if a source
  needs it.
- A frame now carries the colour its YUV is read in, set by the decoder
  for a frame held as NV12 and read as rgba. The first slice's "the pool
  holds no per-frame colour" still holds for frames held in their wire
  layout.
- Each conversion makes new textures, views and a bind group. In a debug
  build that is about 5 ms of host time per 64x48 frame. Reusing the
  targets and bind groups per frame shape is the next step when a profile
  shows it; nothing was measured at 1080p in release.
- The packed 4:2:2 unpack (UYVY, v210) waits on a capture card. Nothing in
  the tree feeds the sidecar a packed format: a capture device is opened
  by ffmpeg, which converts before the wire, and the wire carries no
  packed 4:2:2.

**Item 6a, `filter`: not built (2026-10-12).** Two findings change the
item. FFmpeg 9.0 has D3D12 filters: `scale_d3d12` (size and format),
`deinterlace_d3d12` and `mestimate_d3d12`. `scale_d3d12` runs on the
4090 from ffmpeg's own command line (`-hwaccel d3d12va
-hwaccel_output_format d3d12`, `-vf scale_d3d12=w=640:h=360`, into
`h264_d3d12va`), so on Windows the pool's frames reach those three with
no interop, the way the d3d12va encoder already reads them, and CUDA is
needed only for the `_cuda` filters. Second, libavfilter is not in our
libav (`--disable-avfilter` in ffrwd-av), so there is nothing to build
until the builds change. The LGPL build from BtbN on this machine has
`avfilter-12.dll` with the `_d3d12`, `_cuda`, `_vulkan`, `_opencl`,
`_qsv` and `_amf` filters and libplacebo, but that is a development
convenience, not what the wheel ships. What 6a needs, to be dispatched
on its own:

1. ffrwd-av: `--enable-avfilter`, with `--disable-filters` and then only
   `buffer`, `buffersink`, `format`, `hwupload`, `hwdownload` and
   `hwmap`, and per platform: on Windows `scale_d3d12`,
   `deinterlace_d3d12` and `mestimate_d3d12`; with `--enable-cuda-llvm`
   (LGPL; `--enable-cuda-nvcc` is non-free), on Windows and Linux,
   `scale_cuda`, `overlay_cuda`, `pad_cuda`, `transpose_cuda`,
   `bwdif_cuda`, `yadif_cuda`, `chromakey_cuda`, `colorspace_cuda`,
   `bilateral_cuda` and `thumbnail_cuda`; on Linux the Vulkan filters
   (`scale_vulkan`, `overlay_vulkan`, `blend_vulkan`, `gblur_vulkan`,
   `avgblur_vulkan`, `bwdif_vulkan`, `nlmeans_vulkan`, `transpose_vulkan`,
   `flip_vulkan`, `hflip_vulkan`, `vflip_vulkan`, `xfade_vulkan`,
   `chromaber_vulkan`) and `libplacebo`, which brings glslang or shaderc
   into the build. `avfilter-12` ships beside `avcodec-63` in each wheel,
   and `--av-info` lists the filters.
2. The shim (`runtime/src/av/ffi.rs`): `avfilter_graph_alloc`,
   `avfilter_get_by_name`, `avfilter_graph_alloc_filter`,
   `avfilter_init_str`, `avfilter_link`, `avfilter_graph_config`,
   `avfilter_graph_free`, `av_buffersrc_parameters_alloc` and
   `av_buffersrc_parameters_set` (to hand the source its
   `hw_frames_ctx`), `av_buffersrc_add_frame_flags`,
   `av_buffersink_get_frame`, `av_buffersink_get_hw_frames_ctx` and
   `av_buffersink_get_w`, `_h` and `_format`, and the filter context's
   `hw_device_ctx`, a field set before `avfilter_init_str` whose offset
   the shim reads off the 9.0 headers as it does the codec context's.
3. Windows, D3D12: a D3D12VA frames context over the shared device, as
   `encode.rs` sets one up for the encoder (`attach_d3d12_frames`), each
   frame an `AVD3D12VAFrame` with its texture and sync object. The
   pool's frame is copied once on the device into a frame of that
   context, as `EncoderFeed` does for the encoder, and handed to
   `buffer`. The filter's output frames come from its own pool on the
   same device and are wrapped as frames with `create_texture_from_hal`,
   as the decoder's are (`gpu/d3d12.rs`), the queue waiting on each
   frame's sync object.
4. Windows, CUDA, for the `_cuda` filters: `cuImportExternalMemory` of
   the D3D12 resource through an NT handle from
   `ID3D12Device::CreateSharedHandle`, which needs the frame's texture
   allocated with `D3D12_HEAP_FLAG_SHARED` (wgpu-hal does not set it, so
   such a frame is an allocation of our own wrapped through
   `create_texture_from_hal`); `cuExternalMemoryGetMappedMipmappedArray`
   per plane; `cuImportExternalSemaphore` of the frame's sync object,
   with `cuWaitExternalSemaphoresAsync` and
   `cuSignalExternalSemaphoresAsync` around the work; and an
   `AVCUDADeviceContext` on the primary context of the device matched by
   LUID. libavfilter's CUDA frames are linear device memory, not arrays,
   so each frame is one `cuMemcpy2DAsync` in and one out, both on the
   device. On Linux the same import comes from a Vulkan export
   (`VK_KHR_external_memory_fd`) on NVIDIA.
5. Linux, Vulkan: an `AVVulkanDeviceContext` filled from the pool's
   instance, device and queue families (the work item 3 lists for the
   decoder), so the `_vulkan` filters and libplacebo run on the pool's
   images with no copy.
6. The compiler: where a query's `ffmpeg.<name>` call sits next to a
   GPU region (the partition in `cli/ffrwd/processes.py` that
   `_holds_on_the_gpu` routes, and its copy in `lower.py`), map the name
   to the variant the machine's pool has (`scale` to `scale_d3d12`,
   `scale_cuda` or `scale_vulkan`), read from the sidecar's `--av-info`
   filter list, and lower it as a `filter` host node in the region; a
   filter with no variant stays in ffmpeg with a download at its input.

What blocks it is (1), the libav build, which belongs to the builds
repository. After that, (2) and (3) are the shortest route: three D3D12
filters and no interop.

**Item 6a, `filter` (2026-10-12, owner's machine):** the host's fourth
node runs a libavfilter graph on the D3D12 device, and the compiler moves a
query's `scale` into it where the picture is on the device on both sides.

- ffrwd-av (commit on `main`, not pushed or released): libavfilter 12 is
  built, with `buffer`, `buffersink`, `format`, `hwupload`, `hwdownload`
  and `null` everywhere, and per platform `scale_d3d12` and
  `deinterlace_d3d12` on Windows; on Linux glibc the `_cuda` filters
  (`scale`, `overlay`, `pad`, `transpose`, `yadif`, `bwdif`, `chromakey`,
  `colorspace`) through `--enable-cuda-llvm` (clang's NVPTX target; the
  nvcc route is non-free) and thirteen `_vulkan` ones; none on musl or
  macOS. In 9.0 the Vulkan filters' shaders are compiled to SPIR-V at
  build time by a `glslc` or `glslang` program, so the archive gains only
  libavfilter; there is no run-time shader compiler option left to bundle.
  `libplacebo` is not built (a library of its own). clang missing or unable
  to target NVPTX drops `cuda_llvm` and the `_cuda` filters the way missing
  vulkan headers drop Vulkan; a hardware filter configure turns off is
  recorded in `CONFIGURE`. The CI run is the owner's to dispatch; nothing
  here ran the builds.
- The wheel: `fetch_av.py` takes `avfilter-12` where the archive has it,
  so the v1.1.0 pins still build; `av-pins.json` names the major. The pins
  still point at v1.1.0, which has no libavfilter: they move with the next
  ffrwd-av release. Until then a compile whose sidecar loads v1.1.0 keeps
  `scale` in ffmpeg, so recipe 158's exec test fails there (the release
  workflow's exec tier included); it passes where `FFRWD_AV_RUNTIME` names
  a libav with `scale_d3d12`, and where the sidecar loads no libav (CI's
  exec tier).
- The shim (`runtime/src/av/ffi.rs`, `sys.rs`, `filter.rs`): libavfilter
  loads after the four, optional; without it everything else runs and
  `filter` is refused with the reinstall hint. Twenty-one functions;
  `AVFilterContext`, `AVFilterGraph`, `AVFilterInOut` and
  `AVBufferSrcParameters` transcribed with compile-time offsets, and checked
  at load against the filter context's and graph's own option tables, a
  filter allocated into a graph, and the source parameters' defaults.
  `--av-info` adds `avfilter`, `avfilter_error` and `filters`.
- The node (`ffrwd-wasm/src/av_nodes.rs`, `AvFilter`):
  `filter=graph=<graph>[:format=yuv420p|rgba]`. One graph per instance,
  configured at open between a `buffer` source whose `hw_frames_ctx` is a
  d3d12va NV12 frames context and a `buffersink`, so the output's size is
  known before the first frame. On the shared device (a reader or writer of
  its frames is on the GPU) a frame there is copied once on the device into
  a picture of the source, as for the encoder, and the sink's pictures are
  wrapped as frames with no copy. A frame in host memory is made NV12 by
  libswscale and uploaded; with no shared device the graph runs on a device
  of its own and its pictures are read back. `[av] filter scale_d3d12: on
  the host's device` at open.
- What scale_d3d12 needed of the protocol: it signals one sync object of
  its own for every picture it writes, never waits on an output picture
  before writing its texture again, and never signals the input picture
  after reading it. So the host keeps each picture it took from a graph
  until its own sync object says the queue has read it, rather than
  signalling the filter's object (which would let the next picture's
  reader through early), and copies into the source only after waiting on
  what the filter signalled for its last output.
- The compiler (`cli/ffrwd/device_filters.py`): a `scale` with only its
  width and height, whose picture comes from the device (a node's output
  held on the GPU, another such filter, or an input's own picture) and is
  read only there (a node's input held on the GPU, another such filter, or
  an `_d3d12va` encoder), and whose run touches a node holding its picture
  on the GPU, becomes a `filter` node; a run of them is one graph. Which
  variants the run has is what the compile's sidecar lists under
  `filters`, asked once per process the way `--describe` is, and only
  where a candidate exists. Where that sidecar loads no libav at all the
  command is written for a reader whose sidecar has the bundled
  libavfilter, as item 7 writes for a reader's ffmpeg; where it loads a
  libav without libavfilter, nothing moves. Recipe 158; `docs/dialect.md`.
- Found on the way: a compiled region holding its pictures on the GPU
  decoded in software, since `decode` defaults to `hw=none` and the
  compiler wrote no `hw`; recipe 157 was uploading every frame. Such a
  region's `decode` is now told `hw=auto` (157's pin changed), and recipe
  158 runs with `upload` and `download` at 0 bytes.

Tests: `gpu_frames` `scale_d3d12_runs_between_a_decoder_and_an_encoder_on_the_device`
(h264 320x240, 60 frames, decode on the device, `scale_d3d12` to
160x120, `h264_d3d12va`: 60 pictures at 160x120, profile `upload` and
`download` 0 bytes, 0.238 levels from ffmpeg's own `scale_d3d12` on
average, plane means within 0.074) and
`scale_d3d12_takes_pictures_in_host_memory_uploaded_and_reads_them_back`
(byte-identical to ffmpeg's `hwupload,scale_d3d12,hwdownload`); `av`
gains the refusal without libavfilter, the av-info fields and three
option refusals; `test_node_world` the compiler's table of cases.

What was learned:

- `deinterlace_d3d12` does not run on the 4090: `CreateVideoProcessor`
  returns 0x887A0005 (DXGI_ERROR_DEVICE_REMOVED), "Failed to create video
  processor", from ffmpeg 9.0.1's own command line and from the node over
  BtbN's 9.0 libraries, on progressive and field-flagged input alike. So `yadif` and `bwdif` are not
  mapped; the node still runs it by hand where a driver allows.
- ffmpeg's own `-hwaccel d3d12va -hwaccel_output_format d3d12` into
  `scale_d3d12` made one picture 9 levels off the software decode's scale
  in one of seven runs (frame 11 of 60); the node's runs matched every
  time, and the test's reference decodes in software and uploads.
- One `ffrwd run` of recipe 158 printed device-removed errors when the
  host first made its sync object for kept pictures, with the output
  complete; three reruns and every direct run were clean, and other GPU
  work was running on the machine at the time. Not explained.

Linux beyond the build: the `_cuda` filters need the pool's Vulkan images
exported (`VK_KHR_external_memory_fd`) and imported into CUDA
(`cuImportExternalMemory`, `cuExternalMemoryGetMappedMipmappedArray`), the
images' timeline semaphores imported (`cuImportExternalSemaphore`) and
waited and signalled around the work, an `AVCUDADeviceContext` on the
primary context of the device matched by UUID, and one device-side
`cuMemcpy2DAsync` in and one out, since libavfilter's CUDA frames are
linear memory; that is the inference providers' import, to be written
once for both. The `_vulkan` filters need the `AVVulkanDeviceContext`
filled from the shared device (instance, physical device, device, queue
families, the enabled extensions and features, which wgpu's open must
include), each pool image handed over as an `AVVkFrame` with its layout,
access and timeline semaphore, and raw submits that wait on those
semaphores, which wgpu-core's submit does not do; the decoder's Vulkan
path needs the same.

Measured on the owner's machine (release build, h264 sources from
`testsrc2` with B-frames, 600 frames at 1080p60 and 300 at 4K60, the
sidecar alone against ffmpeg 9.0.1 alone, both decoding with d3d12va and
encoding with `h264_d3d12va`; ffmpeg needs `-bf 0` for that encoder on
this driver, which the node sets itself; median of three runs after a
warm one):

| chain | node ms/frame | ffmpeg ms/frame |
|---|---|---|
| 1080p, no filter | 5.24 | 2.28 |
| 1080p, `filter=graph=null` | 5.02 | - |
| 1080p, `scale_d3d12` to 1920x1080 | 5.13 | 2.39 |
| 1080p, `scale_d3d12` to 1280x720 | 5.14 | 2.20 |
| 4K, no filter | 13.41 | 6.99 |
| 4K, `filter=graph=null` | 14.20 | - |
| 4K, `scale_d3d12` to 3840x2160 | 14.15 | 7.35 |
| 4K, `scale_d3d12` to 1920x1080 | 9.94 | 5.91 |

The scale itself costs ffmpeg 0.1 ms a frame at 1080p and 0.4 ms at 4K,
and the node about the same: at 1080p its runs with and without a filter
are within run-to-run noise, at 4K the node adds 0.7 to 0.8 ms (the copy
into the source's picture and the pass-through), and `scale_d3d12` over
`null` adds nothing measurable. The node's chain is about twice ffmpeg's
time with no filter at all, so the gap is the decode-and-encode path
through the network, which wave 0's table already showed at 235 fps, not
the filter.

### Wave 4: inference on the GPU

1. **The provider is per model.** `NN-EDGE.md` has every model a process
   binds share one `-nn-target` walk, with the compiler passing the union
   of the pins' `-nn-exclude`s. That changes: each model's session
   resolves its own provider from its own pin and the machine, since
   sessions are independent and one process holds a DirectML session
   beside a CUDA one. A model that works on CUDA and not DirectML and one
   that works the other way round run side by side. The engaged provider
   is read per session before anything is bound, as the host reports it
   today, because a session that fell back to the CPU cannot bind device
   memory.
2. **Switching providers never moves the frame.** The pool lives on the
   compositor's device, D3D12 on Windows and Vulkan on Linux, and every
   provider reaches that memory by its own route to the same allocation,
   imported once per pool slot and cached:

   | provider | route to the pool's memory on the same card |
   |---|---|
   | DirectML | binds the D3D12 buffer directly |
   | CUDA, TensorRT, ROCm | external-memory import of the D3D12 resource's shared handle, or of the Vulkan image's exported memory |
   | OpenVINO | remote tensors over D3D11 textures, VA surfaces and DMA-BUFs |
   | WebGPU provider | runs on the compositor's own device (Mali, Adreno on Linux) |
   | CPU, NPUs, CoreML through ONNX Runtime | a download, for that model's input only; zero transfer on Apple is CoreML directly with an IOSurface-backed pixel buffer |

   DirectML and CUDA on one NVIDIA card are two mappings of one
   allocation, so a query whose models pin different providers costs
   nothing in transfers; the one model that can share nothing costs a
   download at its own input, never the region. This is why the pool is
   D3D12 and not Vulkan on Windows: DirectML has no route to a Vulkan
   image.
3. An `ffrwd:av/gpu-nn` extension beside standard `wasi:nn`: a tensor
   from a `gpu-buffer`, an output bound to one, implemented per provider
   in the vendored backend (`vendor/wasmtime-wasi-nn`) over the routes
   above. The open check stands: whether the pinned `ort` rc.10 exposes
   I/O binding to foreign device memory; if not, wave 4 waits on a
   wasmtime-wasi-nn that moves `ort` forward.
4. Preprocessing is a device pass whose buffer is what gets bound:
   `convert` gains planar fp32 at a model's input size, written in the
   pool's API and imported per provider like a frame.
5. Image-sized outputs are bound to device buffers and read as textures;
   boxes and scalars come back by `fetch`. The depth and segment modules
   read a GPU frame, run the model from a buffer, and write their rows.

**Shipped (2026-10-12):** item 1, and section 2.9's first use.
`-nn-exclude <name>=<provider>` keeps one model off a provider (the bare
spelling keeps every model off it); the compiler writes one pair per
provider in a model's `not_on`; each model's session asks for the
process's provider or the CPU; a provider is registered so that a session
it cannot engage for fails to build rather than falling back in silence,
and the host builds that session again on the CPU and says so, one
verdict line per model (`sidecar/NN-EDGE.md`). `ffrwd setup` with no
argument fetches ffmpeg and the runtime tiers detection wants;
`FFRWD_OFFLINE` refuses every fetch naming it; a run that will fetch
starts the download once the script is resolved and joins it before the
sidecar spawns.

What was learned:

- One process loads one ONNX Runtime binary, and DirectML's is a
  different binary from the CPU and CUDA one, so "a DirectML session
  beside a CUDA one" does not hold with the pinned artifacts. The walk
  still settles one provider for the process, the first whose preflight
  is clean and that some model does not exclude; a model that excludes
  it runs on the CPU. Two GPU providers in one process would need one
  runtime build carrying both, which the pinned packages do not.
- The engaged provider is read from whether the session built with the
  provider registered to fail loudly, not from the process's loaded
  libraries; a provider that registers can still run a node it cannot
  place on the CPU (the cuDNN case in `NN-EDGE.md`), which no verdict
  catches.
- The `ort` pin, checked in rc.10's source: I/O binding exists
  (`Session::create_binding`, `IoBinding::bind_input(&Value)`,
  `bind_output(Value)` owned, `bind_output_to_device(&MemoryInfo)`,
  `run_binding`). A foreign CUDA pointer binds as an input through
  `TensorRefMut::from_raw(MemoryInfo, ptr, shape)` with no copy; a
  caller-owned output needs an owned `Value`, built through
  `ort::api().CreateTensorWithDataAsOrtValue` and `Value::from_ptr`, one
  unsafe step; or ort allocates on the device with
  `bind_output_to_device`. A device output cannot be extracted to the
  CPU. DirectML cannot bind foreign memory with rc.10: ort-sys carries
  none of the DML interop API (`OrtDmlApi`,
  `CreateGPUAllocationFromD3DResource`,
  `GetD3D12ResourceFromAllocation`) and the DML provider takes only a
  device id, never the caller's D3D12 device and queue; both would be
  hand-written FFI over `GetExecutionProviderApi("DML", ...)`. The
  vendored backend copies every input into a CPU tensor and every output
  back to bytes on each `compute`; an `IoBinding` on its execution
  context, with a side channel for device values since the wasi-nn
  tensor is bytes only, is where item 3 plugs in. So item 3 on CUDA
  waits on nothing in `ort`; on DirectML it waits on that FFI or on an
  `ort` that carries it.
- A script that declares a model-running module and never calls it
  still starts the background fetch, since resolve keeps uncalled
  declarations; the download is then joined at the end of the run.

**Items 2 to 4 on CUDA, Windows (2026-10-12, owner's machine):** a
module asks the host for a frame as its model's input, and on a CUDA
session the tensor is the pool's memory, bound in place.

- The interface is `ffrwd:gpu/nn` rather than an `ffrwd:av/gpu-nn`:
  `frame-tensor(id, index, options)` addresses a frame by the tick's
  numbering, as `frames.planes` does, and means something only beside a
  port `ports.memory` holds, so it belongs with them, and the node world
  stays unchanged, which is what the companion package is for. A module
  includes the new world `gpu-nn-module` and imports the `wasi:nn`
  interfaces it calls; the options are the tensor's size, NCHW or NHWC,
  RGB or BGR, stretch or letterbox, and `(value * scale - mean) / std`.
  The answer is an ordinary `wasi:nn` tensor. `ffrwd:gpu` stays 0.1.0,
  since it has not been released; its staged deps now carry `wasi:nn`.
- Preprocessing is one compute pass on the shared device
  (`runtime/src/gpu/tensor.rs`): bilinear from pixel centres, the
  stream's matrix and range, fp32 out, reading the planes as held (a
  decoder's NV12 included, so no repack runs first), cached on the frame
  per spec like the other conversions. The same arithmetic on the CPU
  makes the tensor of a frame in host memory; the two agree to within
  1e-4 (a unit test compares them on rgba and yuv420p, both layouts,
  both fits).
- What CUDA imports is the pass's output buffer, not the decoder's
  texture: the model reads only the pass's output, so only that buffer is
  made on a shared heap (`D3D12_HEAP_FLAG_SHARED`, handed to wgpu with
  `create_buffer_from_hal`), and the decoder's pictures stay as libav
  allocates them. It is imported as a dedicated D3D12 resource
  (`cudaImportExternalMemory`, `cudaExternalMemoryGetMappedBuffer`); no
  mipmapped array was needed. The buffers come from a pool and each is
  imported once, the first time a CUDA session binds it, and kept. The
  queue signals one shared D3D12 sync object after each pass; CUDA
  imports it once as an external semaphore, and before the session runs
  the wait is queued on a stream of its own and that stream is waited
  for, so the CPU waits for the pass. The CUDA runtime is loaded at run
  time from where the CUDA provider's preflight found `cudart64_12.dll`
  (`runtime/src/nn/cuda.rs`); the CUDA device is the one whose LUID is the
  D3D12 adapter's, and has to be device 0, the CUDA provider's.
- The vendored backend: a `Tensor` may carry a handle to device memory
  (`DeviceInput`: a CUDA pointer, a wait for its writer, a read back),
  so the wasi-nn tensor itself stays bytes or nothing. When any input
  carries one, `compute` binds every input (that one through
  `TensorRefMut::from_raw`, the rest as CPU tensors), binds every output
  to host memory, and runs the binding; otherwise it runs as before.
- The fallbacks: a DirectML, CPU or CoreML session gets the pass's
  output read back once, a frame in host memory is made on the CPU, and
  one line per model says which: `[nn] <model>: input from the pool,
  bound on CUDA`, `input downloaded, DIRECTML takes host memory (the
  pinned ort cannot hand a D3D12 buffer to a DirectML session)`, `input
  downloaded, CPU takes host memory`, `input in host memory, made on the
  CPU`. A CUDA session whose import fails takes the downloaded route and
  says why. `FFRWD_NN_BIND=off` forces the downloaded route, to compare
  the two on one session.
- Tests: `modules/nn-frames`, a sink running a graph of each channel's
  mean and largest value (`model/channel-means.onnx` and its NHWC twin,
  made by `model/make-channel-means.py`), and two cases in
  `ffrwd-wasm/tests/nn.rs`: the CPU, downloaded and bound routes give the
  same rows within 1e-4, the bound route counts no `download` bytes and
  the downloaded one exactly the tensors' bytes, each line names its
  route, over uploaded frames and over pictures `d3d12va` decoded into
  the pool. Both pass here, with nothing skipped.

Measured on the 4090, release build, h264 decoded into the pool, the
tensor 518x518 letterboxed, 240 frames, median per frame of the time
inside the module for `frame-tensor` and `compute` together (the pass
and the bind, or the pass and the read back; then the run):

| model | input | bound on CUDA | downloaded, CUDA session |
|---|---|---|---|
| channel-means | 1080p | 1.33 ms (0.10 + 1.19) | 6.74 ms (0.74 + 6.00) |
| channel-means | 4K | 1.41 ms (0.08 + 1.32) | 7.41 ms (0.76 + 6.60) |
| Depth Anything V2 small | 1080p | 29.4 ms, again 29.5 | 34.2 ms, again 38.4 |
| Depth Anything V2 small | 4K | 30.7 ms | 40.3 ms |

The input's size barely moves the bound route, since the pass reads only
the pixels it samples; the downloaded route pays about 5 ms a frame for a
3.2 MB tensor (the read back, the conversion into a CPU tensor, and
ONNX Runtime's copy to the device). The depth model's outputs differ
between routes by up to 0.002 on values of 1.4 to 6.3, as much as two
bound runs differ from each other, so that is the CUDA session's own
run-to-run variation. A DirectML session over the same 1080p input, on
the downloaded route: 6.8 ms for channel-means.

What was learned:

- `cudaExternalSemaphoreWaitParams`'s keyed-mutex member is a u64 and a
  u32, so it carries four bytes of padding; a layout without them put
  every later field four bytes early, and the runtime, reading a reserved
  word past the end of the struct, refused the wait with "invalid
  argument" in release builds only. The struct sizes are now asserted.
- In a debug build with `-nn-target directml`, the shared device opened
  on the Microsoft Basic Render Driver and the DirectML session's run
  then failed; a release build opens the 4090 and runs. A debug build's
  wgpu turns the D3D12 debug layer on, after the DirectML session has
  made its device, which is the likely cause; not confirmed.
- The CPU waits for the pass before the run because the CUDA provider's
  stream is its own; rc.10's `CUDAExecutionProvider::with_compute_stream`
  would put the session on the stream the wait is queued in, given the
  stream at session build.

What waits:

- DirectML binding the D3D12 buffer: rc.10 has no `OrtDmlApi` and its
  DirectML provider takes only a device id, so it waits on hand-written
  FFI over `GetExecutionProviderApi("DML", ...)` (with the session
  created on the shared device and queue) or on an `ort` that carries it.
- Item 5, outputs bound to device buffers and read as textures, and the
  depth and segment modules moved onto `frame-tensor`: outputs still come
  back to host memory through `compute`.
- Linux: the pool is on Vulkan there, and the import of its memory and
  timeline semaphore into CUDA is not written; a CUDA session there takes
  the downloaded route and says so.

### Wave 5: two candidates to measure

Neither is in the design yet; each is a measurement on the GPU tier that
decides whether it joins.

- **Pyrowave as the wire between GPU regions.** Intra-only, YUV 4:2:0 or
  4:4:4, 5 to 50x smaller than raw, decoding to a texture on the
  consumer's device. Textures inside a region, pyrowave packets across an
  edge between two GPU regions, raw only at an edge that ends in CPU code.
  To measure on the round-trip module: per-hop GPU time at 4K, generation
  loss over three hops, and alpha, which it lacks.
- **Capture and playout cards as pool producers and consumers.** DeckLink
  GPUDirect and DirectGMA on x86 pro cards, Rivermax for SMPTE 2110:
  packed 4:2:2 buffers by DMA with no CPU, plus `convert`'s unpack pass.
  To measure: latency from the card's frame to the compositor's tick, and
  whether the pool's count bound holds under a card's own pacing.

## 5. Risks and open questions

- **wgpu's dispatch traits are not semver-stable.** The unreleased
  changelog already notes a breaking change for custom backend
  implementers. The backend crate is pinned per wgpu release with the
  renderer that uses it.
- **`wasm32-wasip2` has no threads.** `vello_hybrid`'s CPU preprocessing
  is single-threaded in the guest. Measured: 1.6 ms a frame at 4K with
  eight inputs, 1.7x the native figure, and it scales with the page's
  geometry rather than its pixels. Closed for the compositor; open for a
  module whose CPU side is heavier.
- **Timestamp queries bracket the whole submission on some drivers.** On
  the experiment's Adreno `gpu` equalled `submit+wait` even with the
  copies outside the timestamps. Read GPU time beside the wall clock on
  every new GPU before trusting either column.
- **First-frame cost.** 100 ms, 850 ms with eight 4K inputs, in pipeline
  compilation and first uploads; the warm-up tick in 2.5 absorbs it, and a
  live run that cannot wait shows its first frames late without it.
- **The vendored host is on 0.0.1 and synchronous.** Upstream wasi:webgpu
  is 0.3 and async-only, which the sidecar's synchronous stores cannot
  host. The 0.0.1 wit is frozen in the vendor directory, so the guest
  backend has a fixed target; when the stores go async, both move.
- **One device, several APIs.** The compositor's device, the decoder,
  the encoder and the inference provider must land on one card, matched
  by UUID or LUID, through that platform's sharing. A machine with two
  cards is a configuration, not a default; a part that shares nothing
  (an NPU, a decoder with no shared-image path) gets the device-to-device
  copy or the boundary download, never a silent fallback to host memory.
- **Failure domain.** A decoder crash today takes one ffmpeg process; in
  the sidecar it takes the region. Decode on its own thread; keep the
  ffmpeg path as the fallback when no hardware device is present.
- **LGPL.** Dynamic linking and the licence text beside the libraries in
  the wheel are the obligation; the builds must never carry
  `--enable-gpl`.
- **Wheel size.** libavcodec and libavutil with the hardware codecs and a
  modest decoder set add an estimated 10 to 20 MB per platform wheel,
  well under PyPI's 100 MB file limit. The musl wheels load no NVIDIA
  driver library, so they carry software codecs only, as they do for ONNX
  Runtime today.
- **The `ort` pin.** I/O binding in rc.10 is unverified. If absent, wave
  4 waits on a wasmtime-wasi-nn that moves `ort` forward.
- **Pixel-exact tests.** GPU and CPU rasterization differ, hardware and
  software decode differ by rounding, and vello_hybrid picks its SIMD
  level per target (3 pixels in 8.3 million between the native and wasm
  runs); every byte-checked recipe keeps its software path and GPU recipes
  pin their own output per platform on the GPU tier.
- **Providers differ per model, not per machine.** The same card runs
  one model on DirectML and another on CUDA, and a provider that engaged
  for one model falls back to the CPU for another. Wave 4 resolves the
  provider per session and shares the pool's memory across providers by
  import, so the choice is never a transfer; what it cannot do is bind
  device memory into a session that silently fell back, which is why the
  engaged provider is read before binding.
- **Pyrowave has no alpha.** As a wire between regions it carries pictures
  and not mattes; a region whose output is a matte stays raw or on the
  device.
- **Colour.** Decoded frames are NV12 in the stream's matrix and range;
  blitz renders premultiplied rgba and encodes transparent as black. The
  convert passes own both conversions; nothing un-premultiplies.
- **Port feeds and rgba programmes.** The refusal in blitz's limitations
  ("yuv in the gbr matrix is not converted here") goes away with `gpu`
  hold ports, which are never conformed; host-memory ports keep the
  conform step and the refusal until `convert` serves them too.

## 6. Measurements to take first

Before wave 0's code, on the owner's machine and the L4, with today's
release:

1. `ffrwd run` of a 4K60 rgba passthrough through `invert`, `-jobs 1`,
   `--verbose`: frames a second, and `perf stat` or the equivalent for
   memcpy share and syscalls per frame.
2. The same at yuv420p, and with the relay on the edge (a named-pipe
   plan) and off (a stdio chain).
3. Done: the experiment's table (section 3) stands in for blitz's
   per-frame columns; the same harness on the 4090 and the L4 is the
   number still to take.
4. The depth module at 4K through `-nn-target cuda`: time in the module
   against time in `set_input` and `compute`, from the profile rows once
   wave 0 adds them, or from `strace -c` and ONNX Runtime's profiler
   until then.

These four numbers are what every acceptance line above is measured
against.
