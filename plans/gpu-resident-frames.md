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
- Synchronisation is the API's: fences and shared semaphores where the
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
(`E:/wgpu-wasi-webgpu` on the owner's machine, version 29.0.0, pinned to
wgpu 29, with its own copy of the wit under `wit/deps` and a script that
checks it against ffrwd-cli's), moved out of blitz's `bench-compose-4k60`
branch, which keeps the bench and depends on the crate by path until it is
published. Every `todo!()` is gone: render bundles, indirect draws, the
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
lifting the cap (wave 0, item 5) matters before the release; and the
module depends on `wgpu-wasi-webgpu` by path until that crate is on
GitHub and pinned by tag, so blitz's CI cannot build it until then.

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
- `av-pins.json` pins ffrwd-av v1.0.0 on all seven wheel platforms; the
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
