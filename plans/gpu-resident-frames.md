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
still a sketch it says so.

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

Ownership: one wgpu device per sidecar process, created by the host, handed
to every module's `wasi:webgpu` and to every worker; one CUDA context on the
same physical device for libavcodec and ONNX Runtime; frames in a pool the
host owns, referenced by handle, released by credit.

### 2.2 The world: `ffrwd:av@0.20.0`

Additive on 0.19.1; the sidecar keeps hosting every older world with host
memory, as it does today. A sketch, to be settled in wave 3:

- `node-types.accepts` gains `memory: frame-memory`, `enum frame-memory
  { host, gpu }`, default `host`. A `gpu` input port's `pixel-formats`
  name the texture format the frames are handed in, `rgba` (rgba8unorm) in
  the first cut, `nv12` later as two planes. `output-port.format` gains the
  same field for an output the module writes on the GPU.
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
  never named in a query: NV12 to rgba8 when a `gpu` port asks for rgba,
  rgba8 to NV12 ahead of NVENC, resize for `accepts.like`. WGSL compute
  on the host device, one pass per source frame, cached on the frame so
  every reader of that frame shares it.

### 2.4 libavcodec in the sidecar

Loaded at run time, never linked, the way ONNX Runtime is
(`sidecar/runtime/src/nn.rs`, `nn/dlls.rs`): `libavcodec` and `libavutil`
only. The sidecar has NUT already and ffmpeg keeps every container, so no
`libavformat`. A dlopen shim over the forty-odd functions needed (codec
lookup, context, send/receive, hw device and frames contexts, bitstream
filters, option setting), written against the pinned version's headers.

- Provisioning mirrors `cli/ffrwd/nn.py`: `ffrwd setup av`, a query that
  reaches a `decode` or `encode` node fetches on its way, pinned URL,
  sha256 and byte count per platform, under
  `~/.cache/ffrwd/av-runtime/<version>/<platform>/`. `ffrwd-wasm --av-info`
  answers the version the binary demands, as `--nn-info` does. Argv
  `-av-runtime <dir>`, else `FFRWD_AV_RUNTIME`.
- The builds are ours and LGPL only: `--disable-everything`, the decoders
  and parsers for h264, hevc, av1, vp9, mpeg2, prores, the hardware
  codecs (`--enable-ffnvcodec` for nvdec and nvenc, which dlopen the
  driver's own libraries; `d3d12va`, `d3d11va` and `amf` on Windows;
  `videotoolbox` on macOS; `vaapi` on Linux), no `--enable-gpl`, no
  `--enable-nonfree`, no external encoders. Dynamic linking keeps the
  LGPL obligation to shipping the library and its licence.
- The hardware frame pool and device: `av_hwdevice_ctx` created from the
  sidecar's own CUDA context (`AVCUDADeviceContext.cuda_ctx`), so
  libavcodec, ONNX Runtime and the Vulkan interop share one context on
  one physical device, chosen by UUID to match the wgpu adapter.

### 2.5 The frame pool and interop

- `TickFrame.data` and `OutFrame.data` (`sidecar/runtime/src/node.rs:362`,
  `:423`) become an enum: `Host(Arc<Vec<u8>>)` or `Gpu(Arc<GpuFrame>)`. A
  `GpuFrame` is one slot of a pool: a Vulkan image the host allocated on
  the wgpu device with export flags, its CUDA import, the format, the
  colour, and a refcount the lanes' credit reads
  (`sidecar/ffrwd-wasm/src/lanes.rs`).
- Decode on CUDA: NVDEC writes NV12 into libavcodec's own frames; the
  host copies device to device into a pool slot (about 0.05 ms for 12 MB
  on an L4), converts to the format the reader asked for, and synchronises
  with a stream sync in the first cut, CUDA external semaphores to Vulkan
  timeline semaphores later. Decoding straight into Vulkan images is not
  attempted.
- Encode on CUDA: the pool slot is wrapped as an `AV_PIX_FMT_CUDA` frame
  on a frames context over the same memory; NVENC takes NV12 or rgba
  directly. The convert pass to NV12 runs first so the matrix is ours.
- A texture for the guest: `wgpu_core`'s `create_texture_from_hal` over
  the pool slot's image, pushed into the module's wasi:webgpu resource
  table for the tick, the way `trait_impls.rs` holds every texture by its
  core id. Handed back through `emit-texture`, the resource's core id
  names the slot, and the host takes it.
- Bounds: a pool is counted in frames, not bytes, eight to sixteen per
  stream, with the TCP and credit backpressure that `hold.rs`'s
  `MAX_HELD_FRAMES` and `MAX_HELD_BYTES` give today; a port feed that
  outruns its reader waits on its socket as now. A 4K pool slot is 12 MB
  as NV12 and 33 MB as rgba8, so an L4's 24 GB holds hundreds, but a
  `page` rendering ahead of the `compose` holding it must be held back by
  credit, which it is not today (blitz README, limitations).

### 2.6 The guest: wgpu over wasi:webgpu

A wgpu custom backend crate over the `wasi:webgpu@0.0.1` wit the sidecar
hosts, so any wgpu-based crate runs in a module unchanged. wgpu routes every
call through object-safe dispatch traits and its `custom` cargo feature lets
a crate outside wgpu implement them; `Dispatch*::custom(t)` and `as_custom`
wrap a backend object into a public wgpu type.

| piece | size |
|---|---|
| dispatch traits to implement | 29 traits, about 160 required methods, many empty |
| wgpu's browser backend, the template | about 4,000 lines |
| wasi:webgpu 0.0.1 as the sidecar hosts it | 38 resources, 218 functions |

The browser backend already copes with every deviation 0.0.1 makes from
the IDL: `write-buffer-with-copy` takes a byte slice as the browser call
does; `get-mapped-range-get-with-copy` returns a Vec, the browser backend's
own copy-to-Vec pattern; `map-async` blocks on this host so `poll` is a
no-op as on the web. wasi-gfx's own wgpu fork, "a wgpu fork with
wasi-webgpu backend", is the starting point, retargeted to the `custom`
hook and the 0.0.1 wit. The guest's wgpu version is whatever its renderer
pins; the host stays on wgpu-core 29 behind the wit. Nothing aligns.

blitz then swaps `anyrender_vello_cpu` for `anyrender_vello_hybrid`
(`vello_gpu`: CPU preprocessing, GPU rasterization, external textures with
atlas support), renders to a texture instead of a Vec, and wraps each held
input's texture from `gpu-frames.texture` with `DispatchTexture::custom`.
The probe, bypass and everything above the rasterizer stay as they are.
Per-frame traffic across the wit becomes the strip and alpha buffers, a few
megabytes at most.

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
  `encode` counts one NVDEC and one NVENC session each against the caps it
  already models; `gpu_processes` in `placement.py` treats such a region as
  on a GPU.
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

## 3. Waves

Each wave is dispatchable on its own and leaves the tree green. Numbers are
estimates of the sidecar-side diff; the compiler side is listed with it.

### Wave 0: measure, then trim the raw path

No interface change. Everything here is worth shipping alone.

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

Acceptance: at most ten copies per frame on the raw path at
`-jobs 1`, measured by the profile rows; 4K30 rgba passthrough through an
`invert` module at real time on the owner's machine.

### Wave 1: wgpu over wasi:webgpu

Independent of the sidecar changes; can run in parallel with wave 2.

1. **The backend crate.** A new repository beside `ffrwd-node`: wgpu's
   `custom` dispatch traits over wit-bindgen bindings to the 0.0.1 wit in
   `sidecar/vendor/wasi-webgpu-wasmtime/wit/deps/`. Start from wasi-gfx's
   wgpu fork's backend file. Pinned to the wgpu version `vello_gpu` pins.
2. **Host gaps.** `get-compilation-info` implemented so WGSL errors reach
   the guest (`trait_impls.rs:914`); texture getters implemented
   (`:789` to `:838`) although wgpu caches descriptors; `write-timestamp`
   only if a profiler needs it.
3. **Tests.** wgpu's `hello-compute` and `hello-triangle` rendering to a
   texture, built for `wasm32-wasip2`, run as value modules through
   `ffrwd-wasm --invoke` on a `-gpu` grant; `vello_gpu`'s test scenes
   rendered to a texture and compared against `vello_cpu` within a
   tolerance. These run on the GPU tier.
4. **blitz on `anyrender_vello_hybrid`.** Build the module with the
   backend, render to a texture, download it for now (`map-async` on a
   readback buffer) and emit as today. The measurement that matters: a
   rendered 4K frame's CPU time in the guest, since `wasm32-wasip2` has
   no threads and `vello_gpu`'s preprocessing is single-threaded there.

Acceptance: the lbar recipe renders on the GPU path with frames within
tolerance of the CPU path; the guest-side CPU time per rendered 4K frame
is under 8 ms on the owner's machine.

### Wave 2: libavcodec in the sidecar, packets on the wire

Software frames first: this wave removes the rawvideo pipe and ffmpeg's
rawvideo copies on both sides before any GPU frame exists.

1. **The shim.** `sidecar/runtime/src/av/`: dlopen of `libavcodec` and
   `libavutil` by full path from `-av-runtime`, version check against the
   pinned major, the function table, `--av-info`. Refusals name the
   directory and `ffrwd setup av`, as `nn.rs` does.
2. **Provisioning.** `cli/ffrwd/av.py` beside `nn.py`: pinned artifacts
   per platform, `ffrwd setup av`, fetch on first run. The builds:
   a workflow in a new repository producing the LGPL-only shared
   libraries for the release matrix (`.github/workflows/release.yml`:
   manylinux x86_64 and aarch64, musl, macOS, Windows).
3. **`decode` and `encode` host nodes** with host-memory frames, in
   `host_nodes.rs` or files beside it, software codecs through the shim.
   Both are one instance, in order, as `codec.rs` runs a codec package.
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

### Wave 3: GPU frames

1. **World 0.20.0** as in 2.2; `ffrwd/wasm@0.20.0`; `ffrwd-node` builders;
   the host adapts every older world as `host`.
2. **The pool.** `node.rs` enum; `GpuFrame`; pool per stream with the
   count bound; credit in `lanes.rs`; `hold.rs` holds handles. Source
   nodes held by a hold input are held back by credit (the `page` into
   `compose` case).
3. **CUDA and Vulkan.** Device selection by UUID across wgpu, CUDA and
   libavcodec; the exported Vulkan image per slot; CUDA import; the
   device-to-device copy from NVDEC's frame; stream sync.
4. **`convert`.** WGSL kernels on the host device: NV12 to rgba8 with the
   stream's matrix and range, rgba8 to NV12, resize. Cached per frame.
5. **`decode` and `encode` on hardware.** `hw` option; `AV_PIX_FMT_CUDA`
   frames over the pool; NVENC from a slot; fallback to wave 2's software
   path when the machine has no device, which is CI.
6. **Textures for the guest.** `gpu-frames.texture` over
   `create_texture_from_hal`; `emit-texture`; `payload::gpu-frame`;
   `same` over GPU frames.
7. **The compiler.** `shapes.py` reads `memory`; partition routes by it;
   `placement_cost.py` counts the sessions; the lateral template picks a
   hardware encoder where one is; `GPU_JOBS` removed.
8. **blitz.** Held inputs from `gpu-frames.texture`, output through
   `emit-texture`, bypass by `same`. The lbar recipe at 4K60 with a 4K
   programme and two 1080p feeds is the benchmark.
9. **Tests.** A test module beside `modules/gpu-probe` that reads a GPU
   frame, draws on it, hands it back; byte-checked on the GPU tier against
   a pinned output; the profile rows showing zero host-side frame copies
   on the path.

Acceptance: the benchmark recipe runs at real time on the L4 with the
encoder as the only thing above 50% of its engine; no raw frame bytes in
system memory on the path, by the profile rows.

### Wave 4: inference on the GPU

1. An `ffrwd:av/gpu-nn` extension beside standard `wasi:nn`: a tensor
   from a `gpu-buffer`, an output bound to one. ONNX Runtime I/O binding
   on CUDA in the vendored backend (`vendor/wasmtime-wasi-nn`); check
   first whether the pinned `ort` rc.10 exposes it, since wasmtime-wasi-nn
   48 holds that pin.
2. `convert` gains planar fp32 at a model's input size, so preprocessing
   never touches the CPU.
3. The depth and segment modules read a GPU frame, run the model from a
   buffer, and write their rows; outputs the size of a mask come back by
   handle, boxes by `fetch`.

### Wave 5: Windows and macOS

- Windows: `d3d12va` decode, DirectML through ONNX Runtime on the same
  D3D12 device, wgpu on D3D12 with shared resources, NVENC from D3D12
  textures. Avoids the 1.4 GB CUDA tier entirely.
- macOS: VideoToolbox decode to IOSurface, Metal textures through
  wgpu-hal, VideoToolbox encode. CoreML copies through ONNX Runtime
  regardless; inference stays wave 4's CUDA path or the CPU.

## 4. Risks and open questions

- **wgpu's dispatch traits are not semver-stable.** The unreleased
  changelog already notes a breaking change for custom backend
  implementers. The backend crate is pinned per wgpu release with the
  renderer that uses it.
- **`wasm32-wasip2` has no threads.** `vello_gpu`'s CPU preprocessing is
  single-threaded in the guest. It scales with the page's geometry, not
  its pixels; wave 1 measures it at 4K before anything depends on it.
- **The vendored host is on 0.0.1 and synchronous.** Upstream wasi:webgpu
  is 0.3 and async-only, which the sidecar's synchronous stores cannot
  host. The 0.0.1 wit is frozen in the vendor directory, so the guest
  backend has a fixed target; when the stores go async, both move.
- **One device, three APIs.** wgpu's Vulkan device, libavcodec's CUDA
  context and ONNX Runtime's CUDA provider must land on one card, by
  UUID. A machine with two cards is a configuration, not a default.
- **Failure domain.** A decoder crash today takes one ffmpeg process; in
  the sidecar it takes the region. Decode on its own thread; keep the
  ffmpeg path as the fallback when no hardware device is present.
- **LGPL.** Dynamic linking and a licence file in the fetched set are the
  obligation; the builds must never carry `--enable-gpl`.
- **The `ort` pin.** I/O binding in rc.10 is unverified. If absent, wave
  4 waits on a wasmtime-wasi-nn that moves `ort` forward.
- **Pixel-exact tests.** GPU and CPU rasterization differ, hardware and
  software decode differ by rounding; every byte-checked recipe keeps its
  software path and GPU recipes pin their own output on the GPU tier.
- **Colour.** Decoded frames are NV12 in the stream's matrix and range;
  blitz renders premultiplied rgba and encodes transparent as black. The
  convert passes own both conversions; nothing un-premultiplies.
- **Port feeds and rgba programmes.** The refusal in blitz's limitations
  ("yuv in the gbr matrix is not converted here") goes away with `gpu`
  hold ports, which are never conformed; host-memory ports keep the
  conform step and the refusal until `convert` serves them too.

## 5. Measurements to take first

Before wave 0's code, on the owner's machine and the L4, with today's
release:

1. `ffrwd run` of a 4K60 rgba passthrough through `invert`, `-jobs 1`,
   `--verbose`: frames a second, and `perf stat` or the equivalent for
   memcpy share and syscalls per frame.
2. The same at yuv420p, and with the relay on the edge (a named-pipe
   plan) and off (a stdio chain).
3. blitz's lbar recipe at 1080p30 and 4K30 with `log => 'frame'`: the
   `fetch`, `paint_cmds` and `raster` columns, `-jobs 1` and `-jobs 4`.
4. The depth module at 4K through `-nn-target cuda`: time in the module
   against time in `set_input` and `compute`, from the profile rows once
   wave 0 adds them, or from `strace -c` and ONNX Runtime's profiler
   until then.

These four numbers are what every acceptance line above is measured
against.
