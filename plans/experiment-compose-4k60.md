# Experiment: blitz compositing at 4K60 on the GPU

The operational brief for section 3 of `plans/gpu-resident-frames.md`. It
runs on a developer machine with a GPU; the cloud container has none. The
machine at hand has integrated graphics, so its numbers are a floor, not
the pass line: record them with the adapter's name and keep the same
harness for the 4090 and the L4.

## Question

For a document showing N video inputs, N in 2, 4 and 8, at 3840x2160,
with an L-bar cycle, a lower third and a pulsing logo animating every
frame: what does one rendered frame cost in guest CPU time, in GPU time
and in bytes crossing the wit, and does it hold 60 frames a second over
600 frames?

## Repositories

- `ffrwd-cli`, branch `claude/sidecar-gpu-memory-perf-crkey1`: the plan,
  this brief, and the sidecar (`sidecar/`), whose `ffrwd-wasm` hosts
  `wasi:webgpu@0.0.1` through `sidecar/vendor/wasi-webgpu-wasmtime`.
- `ffrwd-package-blitz` (github.com/imbcmdth/ffrwd-package-blitz),
  checked out beside it: the compositor. `core/` is the document engine,
  testable natively; `compose/` is the wasm module; `examples/lbar.sql`
  is the document the bench generalises.

## Versions found so far

- blitz pins `anyrender` 0.13.0 and `anyrender_vello_cpu` 0.17.0, with
  blitz-paint by git rev `c20690f3`.
- `anyrender_vello_hybrid` 0.11.0 on crates.io pulls `anyrender` 0.14.0,
  `vello_gpu` 0.3.0 and `vello_common` 0.3.0. First task: find the
  `anyrender_vello_hybrid` release that matches blitz's anyrender, or
  bump blitz's anyrender and blitz-paint together; take whichever is less
  work and note it.
- `vello_gpu` is the renamed `vello_hybrid`: CPU preprocessing, GPU
  rasterization, external textures with atlas support.
- wasi-gfx's wgpu fork (github.com/wasi-gfx/wgpu, `wgpu/src/backend/
  wasi_webgpu.rs`) targets this same `wasi:webgpu@0.0.1` wit but is wgpu
  0.19 on the old monolithic `Context` trait with about sixty-six
  `todo!()`s: a source of type conversions and the copy-based mapping
  pattern, not a drop-in.
- wgpu's `custom` cargo feature exposes the dispatch traits (29 traits,
  about 160 required methods) with `Dispatch*::custom(t)` constructors;
  `wgpu/src/backend/webgpu.rs` upstream is the structural template.

## Step A: native baseline

A bench binary in the blitz workspace, `bench/` as a new member, using
`anyrender_vello_hybrid` on native wgpu. No wasm, no sidecar.

- Documents: `examples/lbar.sql`'s document generalised by a generator:
  `<img src="ffrwd:0">` over the whole frame, inputs 1 to N-1 in a
  picture-in-picture stack or grid with `transform` animations running,
  the lower third sliding, the logo pulsing, `css_width` 1280. One
  generator, used by steps A and C.
- Inputs: input 0 at 3840x2160, the rest at 1920x1080, synthetic rgba
  (a gradient with a frame counter burned in so a wrong frame is visible).
  A flag makes every input 4K for the worst case.
- Variants, each at N of 2, 4 and 8:
  1. textures uploaded once (the same image blob every frame), render only;
  2. a new image blob every frame (today's wire), render only;
  3. variant 1 plus a readback of the 4K frame each tick;
  4. variant 1 with the animations idle.
- Timing per frame: `resolve`, `plan`, `paint_cmds` and the renderer's
  prepare and encode, by `Instant`; GPU time by wgpu timestamp queries
  where the adapter has `TIMESTAMP_QUERY`, else wall clock around submit
  and poll, and say which; readback time; bytes uploaded.
- 600 frames at t = n / 60; one JSON row a frame on stdout; a short
  script in `bench/` summarises rows into a table: median and p99 per
  phase, frames a second, peak memory, adapter name and backend.

## Step B: wgpu over wasi:webgpu, just enough

A crate implementing wgpu's `custom` dispatch traits over wit-bindgen
bindings to `sidecar/vendor/wasi-webgpu-wasmtime/wit/deps/webgpu` and
`graphics-context` and `io`, scoped to what `vello_gpu` calls: instance,
adapter, device, queue, buffers, textures and views, samplers, bind groups,
pipeline layouts, shader modules, compute and render pipelines, command
encoders, compute and render passes, copies, `write-buffer`,
`write-texture`, `map-async` with the copy-based mapped range, submit,
`on-submitted-work-done`, query sets. Render bundles, surfaces and
acceleration structures stay `todo!()`. Pinned to the wgpu `vello_gpu`
pins. Start it inside the blitz workspace for iteration speed; it moves
to its own repository once it works.

Host gaps to fill in `sidecar/vendor/wasi-webgpu-wasmtime/src/trait_impls.rs`:
`write-timestamp` is absent, so GPU time cannot be read; `get-compilation-info`
(line 914) traps, so a WGSL error is a trap and not a message; the texture
getters (lines 789 to 838) trap. Build the sidecar with
`cd sidecar && cargo build --release` (the `gpu` feature is on by default).

## Step C: the wasm run

blitz's module built with the backend, as a source node in the shape of
`page`: no inputs, N textures the module makes itself, `fps` 60,
3840x2160, params for the variant. Build with
`cargo build --target wasm32-wasip2 --release`. Run under `ffrwd-wasm`
with a `-gpu` grant for the module's name, the picture to `-f null` or to
a NUT file for a verification run, the rows (`[@rows=...]`) to
`-f ndjson`; `sidecar/NODE-CLI.md` has the argv shapes and the `-gpu`
grant is taken as `-gpu <name>` in `sidecar/ffrwd-wasm/src/main.rs`
(`take_grant_args`). The module ends itself after 600 frames. Same four
variants, same rows, plus bytes crossing the wit counted in the backend
(`write-buffer`, `write-texture`, `get-mapped-range`).

## Pass line

At N of 8, render only, on a 4090: guest CPU under 8 ms a frame, GPU
under 8 ms a frame, under 16 MB a frame across the wit, 60 frames a
second held with the readback variant too. The L4 is about a quarter of a
4090. On integrated graphics, report the numbers and the ratio between
variants; do not read a miss there as a verdict.

## If it fails

- GPU time over budget at N of 8: `vello_gpu` is not the compositor for
  the video layers. A small WGSL pass composites the video rectangles and
  `vello_gpu` draws the graphics layer over them, the document split at
  the video draws, which `core/src/probe.rs` already finds.
- Guest CPU over budget: the single-threaded preparation on `wasm32-wasip2`
  is the problem; the strip generation moves to the host or waits for
  threads.
- Either rewrites wave 3 of the plan before it starts.

## Deliverable

`plans/experiment-compose-4k60-results.md` on the plan's branch: the
machine (OS, adapter, backend, driver), the version decisions, the table
per step and variant, and the verdict against the pass line. Bench code
on a branch of the blitz repository, the backend crate beside it.
