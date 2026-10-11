# Results: blitz compositing at 4K60 on the GPU

The run of `plans/experiment-compose-4k60.md` on the owner's machine,
2026-10-10. Integrated graphics: the numbers are a floor, not the pass
line, and the same harness runs on the 4090 and the L4 unchanged.

## The machine

| | |
|---|---|
| OS | Windows 11 Pro 10.0.26100, aarch64 |
| CPU | Snapdragon X Elite X1E78100 (12 Oryon cores), 32 GB |
| adapter | Qualcomm Adreno X1-85 (integrated) |
| backend | Vulkan, Qualcomm Adreno Vulkan driver (build 1698055923, 2023-10-23, compiler E031.50.17.00); Windows driver 31.0.124.0 |
| GPU time | wgpu timestamp queries (`TIMESTAMP_QUERY_INSIDE_ENCODERS` is available) |
| toolchain | rustc 1.99.0 (blitz needs 1.91+), wasm32-wasip2, Python 3.12 for stylo's build, LLVM 23 clang for `ring` |

The machine had a standalone Rust 1.89 and no Python or clang; the
experiment installed rustup, Python and LLVM (notes at the end).

## Version decisions

- **`anyrender_vello_hybrid` 0.10.0, no bump.** It depends on anyrender
  0.13, vello_common 0.1, vello_hybrid 0.1 (vello_gpu before the rename)
  and wgpu 29: exactly blitz's own pins at rev `c20690f3`, which lists
  the same 0.10.0 in its workspace. The 0.11.0 on crates.io (anyrender
  0.14, vello_gpu 0.3, wgpu 30) would have meant bumping blitz-paint and
  anyrender together. Nothing in blitz changed for the renderer.
- **wgpu 29, as vello_hybrid pins it,** for the backend crate too. The
  host is wgpu-core 29 behind the wit; nothing had to align.
- **wgpu vendored for the wasm build** (`vendor/wgpu` in the blitz repo,
  one change: `webgpu` off its default features). vello_hybrid and
  anyrender_vello_hybrid take wgpu with its defaults, and on a wasm32
  target that feature compiles wgpu's browser backend, whose wasm-bindgen
  imports a wasip2 component cannot carry. Native builds are unchanged.
- **Two small changes to blitz core** so the bench can drive it:
  `Compositor::paint_into` paints the document into any anyrender scene,
  and `set_frame_image` hands a picture back with the same blob id, which
  is what the hybrid renderer caches an upload by (`set_frame` makes a
  new id each call, so "the same blob every frame" needed it).

## How the numbers were taken

Per frame, by `Instant`: `resolve`, `plan`, `paint_cmds` (painting the
document into the vello_hybrid scene, which includes converting and
uploading any picture the renderer has not got), `encode` (vello_hybrid's
preparation and command encoding). `cpu` is their sum: the guest's share
of a frame. GPU time is a timestamp pair around the frame's encoder,
resolved and read back each frame; `submit+wait` is the wall clock from
submit to the queue idle. `readback` is mapping the 4K frame and copying
it into a Vec. `frame` is the whole tick, with the input copies of
variant 2 (`synth`, not the compositor's cost) in it. 600 frames at
t = n / 60, no warm-up: pipeline compilation and the first uploads land in
frame 0 and show in the p99 column.

Two things to know when reading the table:

- **On this driver the timestamp pair brackets the whole submission.**
  In the variants that upload or read back, `gpu` equals `submit+wait`
  though the copies are outside the timestamps; in the render-only
  variants the two agree too. So `gpu` here is "GPU busy for the frame",
  which is what the pass line asks anyway. The `submit+wait` column is
  there so a discrete GPU, where the two will differ, can be read.
- **Variant 2 frees last frame's atlas regions in a submission of its
  own.** A first version recorded the clears in the frame's encoder, and
  since `write_texture` is staged ahead of the command buffers of the
  submit it lands in, the clear wiped the upload into the same region
  and the frame drew no video at all. A frame dump caught it; every
  number below is from the fixed version, and a dump of frame 123 shows
  the counter at 0123 on every input.

The document: input 0 over the whole frame with the L-bar cycle (8 s,
scale 1 to 0.75 and back), inputs 1 to N-1 as 288x162 CSS px pictures in
a grid with a drift animation each, the lower third sliding in and out
(6 s), the logo pulsing (1.2 s), at 1280 CSS px, so Blitz paints at a
device scale of 3. Inputs are gradients with the frame counter burned in,
input 0 at 3840x2160 and the rest at 1920x1080 unless `--all-4k`.

## Step A: native wgpu

Times in ms, median / p99. `miss` is frames over 16.67 ms. `fps` is
frames over the run's wall time. `upload` is MB of pictures uploaded a
frame. Peak is the process's working set, which includes the CPU
renderer blitz core still constructs and the inputs themselves (3.2 GB
with eight 4K inputs is mostly the pictures and their copies).

| N | variant | inputs | resolve | plan | paint_cmds | encode | cpu | gpu | submit+wait | readback | frame | fps | miss | upload | peak MB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 1 resident | 4K + 1080p | 0.06 / 0.19 | 0.01 / 0.04 | 0.39 / 0.88 | 0.09 / 0.21 | 0.55 / 1.25 | 3.36 / 11.40 | 3.87 / 12.14 | 0 | 4.48 / 12.84 | 194.0 | 1 | 0.1 | 705 |
| 2 | 2 upload | 4K + 1080p | 1.32 / 1.82 | 0.04 / 0.06 | 14.75 / 18.09 | 0.13 / 0.21 | 16.25 / 19.58 | 11.28 / 12.88 | 13.58 / 23.15 | 0 | 37.06 / 49.70 | 26.7 | 600 | 41.5 | 1033 |
| 2 | 3 readback | 4K + 1080p | 0.14 / 0.26 | 0.03 / 0.06 | 0.43 / 0.83 | 0.16 / 0.34 | 0.77 / 1.22 | 11.11 / 12.91 | 13.86 / 15.77 | 6.33 / 8.51 | 22.24 / 27.31 | 45.0 | 600 | 0.1 | 736 |
| 2 | 4 idle | 4K + 1080p | 0.01 / 0.05 | 0.02 / 0.07 | 0.35 / 0.96 | 0.09 / 0.32 | 0.48 / 1.39 | 3.39 / 11.65 | 4.00 / 12.79 | 0 | 4.56 / 13.96 | 186.4 | 1 | 0.1 | 704 |
| 4 | 1 resident | 4K + 1080p | 0.06 / 0.20 | 0.02 / 0.04 | 0.49 / 0.67 | 0.10 / 0.29 | 0.67 / 1.07 | 3.44 / 6.54 | 4.07 / 7.32 | 0 | 4.81 / 8.12 | 193.5 | 2 | 0.1 | 720 |
| 4 | 2 upload | 4K + 1080p | 1.36 / 2.46 | 0.04 / 0.06 | 17.34 / 23.00 | 0.16 / 0.35 | 18.94 / 25.26 | 11.43 / 12.81 | 14.25 / 19.52 | 0 | 41.65 / 51.03 | 23.7 | 600 | 58.1 | 960 |
| 4 | 3 readback | 4K + 1080p | 0.15 / 0.27 | 0.03 / 0.05 | 0.53 / 0.72 | 0.18 / 0.32 | 0.89 / 1.32 | 11.31 / 12.48 | 14.08 / 16.96 | 6.34 / 7.60 | 22.48 / 28.32 | 44.4 | 600 | 0.1 | 752 |
| 4 | 4 idle | 4K + 1080p | 0.01 / 0.03 | 0.01 / 0.05 | 0.43 / 1.02 | 0.09 / 0.28 | 0.55 / 1.27 | 3.10 / 11.46 | 3.77 / 12.44 | 0 | 4.48 / 13.14 | 198.2 | 1 | 0.1 | 720 |
| 8 | 1 resident | 4K + 1080p | 0.10 / 0.36 | 0.02 / 0.09 | 0.69 / 0.92 | 0.13 / 0.28 | 0.95 / 1.39 | 3.82 / 6.85 | 4.38 / 7.84 | 0 | 5.38 / 8.95 | 167.5 | 2 | 0.2 | 752 |
| 8 | 2 upload | 4K + 1080p | 1.95 / 2.70 | 0.04 / 0.07 | 24.38 / 27.87 | 0.18 / 0.29 | 26.61 / 30.55 | 11.65 / 13.00 | 16.43 / 24.58 | 0 | 56.19 / 65.85 | 17.7 | 600 | 91.2 | 1169 |
| 8 | 3 readback | 4K + 1080p | 0.18 / 0.35 | 0.04 / 0.09 | 0.74 / 1.03 | 0.21 / 0.33 | 1.16 / 1.62 | 11.52 / 12.82 | 14.30 / 15.88 | 6.41 / 7.65 | 23.08 / 27.44 | 43.1 | 600 | 0.2 | 784 |
| 8 | 4 idle | 4K + 1080p | 0.01 / 0.05 | 0.02 / 0.10 | 0.63 / 1.98 | 0.15 / 0.50 | 0.82 / 2.49 | 4.20 / 13.01 | 4.82 / 14.32 | 0 | 5.74 / 16.48 | 146.4 | 4 | 0.2 | 752 |
| 8 | 1 resident | all 4K | 0.10 / 0.36 | 0.02 / 0.06 | 0.72 / 1.41 | 0.14 / 0.42 | 1.01 / 2.10 | 4.48 / 13.13 | 5.05 / 13.90 | 0 | 6.09 / 15.47 | 129.8 | 1 | 0.4 | 3233 |
| 8 | 2 upload | all 4K | 9.21 / 11.12 | 0.05 / 0.07 | 96.10 / 106.01 | 0.21 / 0.31 | 105.69 / 115.44 | 13.35 / 14.54 | 43.60 / 53.78 | 0 | 203.31 / 217.55 | 5.1 | 600 | 265.4 | 3486 |
| 8 | 3 readback | all 4K | 0.17 / 0.35 | 0.03 / 0.06 | 0.73 / 1.49 | 0.19 / 0.46 | 1.13 / 2.16 | 10.34 / 13.80 | 12.71 / 19.56 | 6.44 / 10.46 | 21.50 / 30.10 | 43.5 | 600 | 0.4 | 3234 |

What it says, on this integrated GPU:

- **Render-only at N of 8 is cheap.** About 1 ms of CPU and 4 ms of GPU
  a frame, 167 fps, and the picture count barely moves it: N of 2 to 8
  adds 0.4 ms of CPU and 0.5 ms of GPU. All eight inputs at 4K add
  another 0.7 ms of GPU. The document's own cost (idle variant) is the
  same as the animated one: Blitz's resolve is 0.01 to 0.1 ms, so
  animation is free and the cost is painting.
- **Uploading the pictures every frame is the whole cost of today's
  wire.** At N of 8 it is 24 ms of CPU a frame (converting 91 MB to
  premultiplied and `write_texture`, 23 ms; freeing last frame's Vecs,
  2 ms) and 12 ms of GPU, and the run drops from 167 to 18 fps. With
  every input at 4K it is 96 ms and 265 MB a frame. The ratio to the
  resident variant, 25x on CPU and 3x on GPU, is the case for GPU-resident
  frames in one row.
- **Reading the frame back costs more than rendering it.** The 4K
  readback adds 6.4 ms of guest copy and takes the GPU from 4 to 11 ms a
  frame (the copy out of the render target, on a unified-memory part),
  and the run sits at 43 to 45 fps whatever N is. That is today's output
  path, until wave 3 hands textures back; on this GPU alone it breaks 60.

## Step B: wgpu over wasi:webgpu

`backend/` in the blitz repo (`wgpu-wasi-webgpu`): wgpu 29's `custom`
dispatch traits over wit-bindgen bindings to the sidecar's vendored
wit, scoped to what vello_hybrid calls (its wgpu surface is small: no
compute passes, no index buffers, `create_buffer_init`, `write_buffer`,
`write_texture`, render passes, two texture copies). About 1,200 lines
with the conversions; render bundles, surfaces, acceleration structures,
binding arrays, pipeline caches, immediates, multi-draw and timestamps
inside passes are `todo!()`. The 0.0.1 wit's copy-based mapping maps
onto wgpu's mapped ranges directly: a read mapping is the Vec
`get-mapped-range-get-with-copy` returns, a write mapping is a Vec that
goes back through `get-mapped-range-set-with-copy` when wgpu drops the
range. Every list crossing the wit is counted; the module adds the sum
to each row.

Host gaps filled in `sidecar/vendor/wasi-webgpu-wasmtime`:

- the wit gained `gpu-command-encoder.write-timestamp` (wgpu's
  `TIMESTAMP_QUERY_INSIDE_ENCODERS`; the host asks the device for it
  whenever `timestamp-query` is requested and the adapter has it) and
  `gpu-queue.get-timestamp-period`, without which resolved timestamps
  are raw ticks in unknown units (52 ns each on this Adreno);
- `get-compilation-info` answers with naga's message and its line
  instead of trapping (the shader module keeps what wgpu-core said);
- the eight texture getters answer from the descriptor the texture was
  created with instead of trapping.

The wit the guest binds is ffrwd-cli's vendored copy, by path, so guest
and host cannot drift.

## Step C: the wasm run

The same harness as a source node (`bench/wasm` in the blitz repo, built
for wasm32-wasip2 on the backend), run under `ffrwd-wasm` with its
`-gpu` grant, rows to `-f ndjson`, the picture unmapped except in the
readback variant, where it goes to `-f null` so the frame leaves the
module as it would in a query. One instance (the node is not declared
pure, so the host does not spread it over workers), the tick number as
the frame. `wit` is MB a frame across the wit: strips, alpha, the
timestamp readback, and in the upload and readback variants the
pictures and the frame. The host's wgpu is the same Vulkan device as
step A's. Peak memory is not read on wasm.

| N | variant | inputs | resolve | plan | paint_cmds | encode | cpu | gpu | submit+wait | readback | frame | fps | miss | upload | wit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 1 resident | 4K + 1080p | 0.11 / 0.44 | 0.03 / 0.09 | 0.59 / 1.53 | 0.20 / 0.61 | 0.93 / 2.28 | 3.91 / 11.35 | 4.45 / 12.91 | 0 | 5.52 / 15.02 | 165.6 | 1 | 0.1 | 0.8 |
| 2 | 2 upload | 4K + 1080p | 0.21 / 0.41 | 0.05 / 0.10 | 28.23 / 34.31 | 0.28 / 0.55 | 28.79 / 34.82 | 11.31 / 13.12 | 13.76 / 16.97 | 0 | 45.00 / 54.72 | 22.1 | 600 | 41.5 | 42.2 |
| 2 | 3 readback | 4K + 1080p | 0.23 / 0.55 | 0.05 / 0.08 | 0.66 / 1.03 | 0.35 / 0.66 | 1.30 / 1.97 | 11.18 / 12.55 | 14.06 / 17.27 | 10.37 / 13.50 | 25.85 / 30.70 | 38.6 | 600 | 0.1 | 34.0 |
| 2 | 4 idle | 4K + 1080p | 0.02 / 0.11 | 0.04 / 0.10 | 0.54 / 1.36 | 0.23 / 0.55 | 0.84 / 1.96 | 4.49 / 11.33 | 5.18 / 12.44 | 0 | 6.08 / 13.36 | 152.1 | 2 | 0.1 | 0.9 |
| 4 | 1 resident | 4K + 1080p | 0.15 / 0.44 | 0.03 / 0.11 | 0.76 / 1.99 | 0.27 / 0.58 | 1.22 / 3.14 | 4.65 / 11.23 | 5.46 / 11.96 | 0 | 6.88 / 13.18 | 134.4 | 1 | 0.1 | 1.2 |
| 4 | 2 upload | 4K + 1080p | 0.22 / 0.41 | 0.05 / 0.13 | 36.12 / 41.77 | 0.34 / 0.59 | 36.76 / 42.34 | 11.43 / 13.37 | 14.51 / 22.57 | 0 | 54.73 / 66.04 | 18.2 | 600 | 58.1 | 59.1 |
| 4 | 3 readback | 4K + 1080p | 0.24 / 0.50 | 0.06 / 0.07 | 0.82 / 1.13 | 0.40 / 0.83 | 1.52 / 2.49 | 11.34 / 13.02 | 14.25 / 16.55 | 10.37 / 13.20 | 26.20 / 30.81 | 38.0 | 600 | 0.1 | 34.3 |
| 4 | 4 idle | 4K + 1080p | 0.02 / 0.08 | 0.04 / 0.12 | 0.68 / 1.77 | 0.23 / 0.65 | 0.99 / 2.44 | 4.08 / 11.81 | 4.78 / 13.06 | 0 | 5.85 / 15.38 | 146.6 | 1 | 0.1 | 1.0 |
| 8 | 1 resident | 4K + 1080p | 0.18 / 0.39 | 0.04 / 0.08 | 1.07 / 2.46 | 0.31 / 0.65 | 1.61 / 3.48 | 4.93 / 12.22 | 5.65 / 13.08 | 0 | 7.29 / 14.94 | 126.5 | 1 | 0.2 | 1.4 |
| 8 | 2 upload | 4K + 1080p | 0.26 / 0.41 | 0.05 / 0.08 | 52.42 / 62.73 | 0.39 / 0.77 | 53.13 / 63.43 | 11.66 / 13.00 | 16.68 / 19.23 | 0 | 75.74 / 87.39 | 13.2 | 600 | 91.2 | 92.5 |
| 8 | 3 readback | 4K + 1080p | 0.27 / 0.59 | 0.06 / 0.19 | 1.14 / 1.75 | 0.44 / 0.70 | 1.91 / 2.68 | 11.64 / 12.82 | 14.44 / 23.78 | 10.36 / 12.67 | 26.79 / 37.99 | 36.9 | 600 | 0.2 | 34.6 |
| 8 | 4 idle | 4K + 1080p | 0.02 / 0.07 | 0.04 / 0.11 | 0.99 / 2.34 | 0.28 / 0.75 | 1.35 / 3.08 | 4.27 / 12.14 | 5.18 / 13.66 | 0 | 6.72 / 15.24 | 133.6 | 4 | 0.2 | 1.4 |
| 8 | 1 resident | all 4K | 0.17 / 0.48 | 0.04 / 0.12 | 1.08 / 2.70 | 0.31 / 0.80 | 1.63 / 3.77 | 5.67 / 13.77 | 6.33 / 14.92 | 0 | 8.07 / 16.83 | 100.3 | 11 | 0.4 | 1.7 |
| 8 | 2 upload | all 4K | 0.26 / 0.59 | 0.06 / 0.13 | 182.33 / 200.41 | 0.44 / 0.86 | 183.13 / 201.41 | 13.54 / 16.65 | 44.75 / 58.09 | 0 | 240.10 / 264.02 | 4.2 | 600 | 265.4 | 266.7 |
| 8 | 3 readback | all 4K | 0.27 / 0.74 | 0.06 / 0.11 | 1.14 / 1.65 | 0.43 / 0.75 | 1.91 / 2.70 | 13.26 / 14.51 | 16.20 / 18.83 | 10.32 / 13.46 | 28.52 / 33.51 | 33.7 | 600 | 0.4 | 34.9 |

The misses in the render-only variants are frame 0 (pipeline
compilation and the first uploads: 100 ms on wasm, 60 ms native) and a
few frames in the first forty while the atlas and the driver settle;
with eight 4K inputs frame 0 is 850 ms and ten frames of the first forty
sit at 16.7 to 17.3 ms. After that every render-only run holds 60 fps.

**The picture is right.** Frame 4 of the N of 2 run, written by the
module to a NUT file and pulled out with `nutframe`, differs from the
native bench's dump of the same frame in 3 of 8,294,400 pixels, each by
one level: rasterisation rounding (vello_hybrid picks its SIMD level per
target). The frame-123 hashes in the two tables differ for the same
reason and are not a signal on their own.

**What wasi adds**, at N of 8 against step A:

| | native | wasm | ratio |
|---|---|---|---|
| render only, CPU a frame | 0.95 ms | 1.61 ms | 1.7x |
| render only, GPU a frame | 3.82 ms | 4.93 ms | 1.3x |
| render only, frame | 5.38 ms, 167 fps | 7.29 ms, 126 fps | 1.35x |
| render only, across the wit | | 1.4 MB | |
| upload every frame, CPU | 26.6 ms | 53.1 ms | 2.0x |
| upload every frame, across the wit | | 92.5 MB | |
| readback, the copy | 6.4 ms | 10.4 ms | 1.6x |
| readback, frame | 22.0 ms, 45 fps | 26.8 ms, 37 fps | 1.2x |

The CPU share grows by 0.7 ms: the strip and alpha uploads crossing the
wit as lists (1.4 MB a frame, lifted once by the canonical ABI and once
by wgpu's staging) and the timestamp readback's map-async, which blocks
on the host. The GPU share grows by 1.1 ms, which is the host's own
submit path (wgpu-core behind the wit, polled to idle for the timestamp
map) rather than more work: the idle variant shows the same 4.3 ms. The
upload variant doubles its CPU because 91 MB a frame now crosses the wit
as `list<u8>` on top of the premultiply conversion; the readback adds 4
ms for the 33 MB frame coming back the same way. Guest memory was not
read; the module's heap holds the inputs (8 pictures, 91 MB) and vello's
scratch and nothing else of size.

## Verdict

The pass line is for a 4090, and this is an integrated Adreno, so the
reading is the one the brief asks for: the numbers and the ratios, not
a miss read as a verdict. Against the line itself, on this GPU:

| pass line, N of 8, render only | measured here, wasm | |
|---|---|---|
| guest CPU under 8 ms a frame | 1.6 ms (p99 3.5) | holds, 5x margin |
| GPU under 8 ms a frame | 4.9 ms median; p99 11.7 (a few frames in each hundred, native shows the same shape at 6.8) | holds at the median, not at p99 on this GPU |
| under 16 MB a frame across the wit | 1.4 MB | holds, 11x margin |
| 60 fps held over 600 frames | 126 fps, one miss (frame 0) | holds |
| 60 fps held with the readback variant too | 37 fps | does not hold here |

So the design's assumption stands on the hardware that is weakest for
it: vello_hybrid driven from a module through wasi:webgpu composites
the 4K60 document over eight video textures in a third of the frame
budget, with the guest's single-threaded share at 1.6 ms, and the bytes
crossing the wit at a hundredth of today's wire. Neither of the brief's
failure branches is taken: the GPU is not over budget, so vello_hybrid
stays the compositor for the video layers and no separate WGSL
compositing pass is needed; the guest CPU is not over budget, so strip
generation stays in the guest. On a 4090 the GPU and readback columns
should fall by several times and the CPU columns stay as they are (the
guest's work does not depend on the GPU), which is where the 4x margin
the L4 needs comes from; this harness runs there unchanged.

The one line that does not hold is the readback variant, and it is the
line the design exists to retire: reading the 4K frame back each tick
costs more than rendering it (11.6 ms of GPU for the copy out of the
render target on this unified-memory part, 10.4 ms of guest copy with
33 MB crossing the wit), and today's output path sits at 37 fps whatever
N is. Wave 3's hand-back of the frame as a texture removes it entirely;
until then a 4K60 compose through the sidecar is bound by its output
copy, not its rendering. The upload variant says the same of the input
side, 25x the resident variant's CPU and 92 MB a frame across the wit:
GPU-resident frames are the difference between 13 and 126 fps on this
machine.

Two measurement notes carry forward: the first frame costs 100 ms (850
ms with eight 4K inputs) for pipeline compilation and the first atlas
uploads, which a run should absorb before its clock starts; and this
driver's timestamps bracket the submission, so on a discrete GPU the
`gpu` and `submit+wait` columns should be read side by side.

## Where the code is

- ffrwd-cli, branch `claude/sidecar-gpu-memory-perf-crkey1`: this file,
  and the host gaps filled in `sidecar/vendor/wasi-webgpu-wasmtime`.
- ffrwd-package-blitz, branch `bench-compose-4k60`: `bench/` (the native
  bench, the summariser, `nutframe`, the run scripts and their README),
  `bench/doc` (the document generator, the inputs, the rows),
  `bench/harness` (the frame loop both steps share), `bench/wasm` (the
  step C node), `backend/` (wgpu over wasi:webgpu, with its README),
  `vendor/wgpu` (the one-line patch), and the two methods added to
  blitz core. The rows of every run above are in `bench/out`, not
  committed; the scripts regenerate them.

## Appendix: what the experiment installed on the machine

rustup with stable 1.99.0 and the wasm32-wasip2 target (`~/.cargo/bin`;
the standalone 1.89 install is still first on PATH), Python 3.12
(user scope, winget, for stylo's build script), LLVM 23 (machine scope,
winget, for `ring`'s C on aarch64; clang needs `INCLUDE` set to the
MSVC and Windows SDK include directories outside a developer prompt).

## What this changes in the plan

For the revision of `plans/gpu-resident-frames.md`, from the experiment
and from a survey of the hardware the sidecar will meet (decoders,
compositor backends, inference providers, capture cards):

1. **Wave 3 is unblocked; section 3's failure branches are not taken.**
   vello_hybrid stays the compositor, strip generation stays in the guest.
2. **Section 1's estimates become measurements.** Per-frame uploads are
   25x the resident variant's CPU and 92 MB a frame across the wit; the
   readback caps the run at 37 fps whatever N is. Today's output path is
   the bound, so wave 3's texture hand-back is worth more than wave 0.
3. **A run needs a warm-up.** Frame 0 is 100 ms (850 ms with eight 4K
   inputs); the clock should start after it.
4. **Section 2.5 needs a per-platform matrix, not one Vulkan sketch.**
   Windows is D3D12 (the Snapdragon decodes through D3D11VA and Media
   Foundation with no Vulkan Video; DirectML binds D3D12 buffers directly;
   NVIDIA, AMD and Intel decode through D3D11VA/D3D12VA there too). Linux
   and Android are Vulkan with DMA-BUF or AHardwareBuffer import, and
   NVIDIA decodes straight into VkImages with Vulkan Video, the cleanest
   case for the L4. macOS is Metal through VideoToolbox's IOSurfaces.
   wgpu wraps native images on all three. Wave 5 therefore becomes the
   backend choice at the start rather than a port at the end, and the
   host should prefer D3D12 over Vulkan on Windows (`runtime/src/gpu.rs`
   tries Vulkan first today).
5. **"Zero transfer", not "zero copy".** Every hardware decoder emits NV12
   or P010 and pyrowave emits YUV, so one device-local conversion pass per
   input is the floor. The video sampling pass from section 3's failure
   branch belongs in wave 3 anyway, for format rather than speed: the
   compositor reads YUV planes from the decoder's image and no RGBA copy
   exists. On unified-memory parts the readback cost measured here is
   untiling into uncached memory, not a bus transfer; zero transfer there
   means staying in tiled, cached allocations.
6. **Wave 2 inherits the decoder matrix and the encode caps.** Hwaccel
   per platform; integrated parts cap concurrent sessions (the
   Snapdragon's VPU: 4K120 decode, 4K60 encode, one 4K60 decode beside
   two 4K30 encodes), which the compiler should treat as a capacity.
7. **Wave 4 inherits an inference matrix.** DirectML is the one ONNX
   Runtime provider that binds the compositor's memory directly; CUDA,
   TensorRT and ROCm follow through external-memory import; OpenVINO
   wraps VA surfaces, dma-bufs and D3D11 textures as remote tensors;
   ort's CoreML provider copies, so zero transfer on Apple means CoreML
   directly with an IOSurface-backed pixel buffer; the WebGPU provider is
   the universal fallback (Mali, Adreno on Linux); NPUs take host memory.
   Preprocessing is a device pass whose buffer is what gets bound;
   image-sized outputs are bound to device buffers and read as textures.
   The open check stands: whether the pinned ort exposes I/O binding to
   foreign device memory.
8. **Two new candidates.** Pyrowave as the wire between GPU regions:
   intra-only, YUV 4:2:0 or 4:4:4, 5 to 50x smaller than raw, decoding to
   a texture on the consumer's device; textures inside a region, pyrowave
   packets across edges between GPU regions, raw only at edges that end
   in CPU code; to measure on the round-trip module: per-hop GPU time at
   4K, generation loss over three hops, and alpha, which it lacks. And
   capture and playout cards as pool producers and consumers (DeckLink
   GPUDirect and DirectGMA on x86 pro cards, Rivermax for SMPTE 2110):
   packed 4:2:2 buffers by DMA with no CPU, plus an unpack pass.
9. **Section 2.6 is code.** `wgpu-wasi-webgpu` exists; the host's wit
   gained `write-timestamp` and `get-timestamp-period` and no longer
   traps on compilation info or texture getters, so the sidecar's wit is
   ffrwd's own superset of wasi:webgpu 0.0.1 and should be versioned as
   such. A wgpu-based module on wasip2 needs wgpu without its `webgpu`
   default feature.
10. **Risks.** Timestamp queries on this driver bracket the whole
    submission; read GPU time beside the wall clock on every new GPU.
