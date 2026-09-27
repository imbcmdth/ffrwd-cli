# wasi-webgpu-wasmtime (vendored)

The wasmtime host for `wasi:webgpu@0.0.1` over wgpu-core 29, which the sidecar
links for a module granted `-gpu` (see `runtime/src/gpu.rs`).

Taken from [wasi-gfx-runtime](https://github.com/wasi-gfx/wasi-gfx-runtime)'s
`crates/wasi-webgpu-wasmtime` at 301716b (2026-06-22), the last revision
before upstream moved to `wasi:webgpu@0.3.0-rc.2`. Apache-2.0 WITH
LLVM-exception, as upstream (`LICENSE`).

## Why this revision, and not upstream's current crate

Upstream's 0.3.0 crate is already on wasmtime 48, but it implements the WASI
0.3 version of the interface: `request-adapter`, `map-async` and the other
operations a browser runs asynchronously are async functions there, hosted as
wasmtime concurrent functions. Linking one makes every store it is linked into
async-only (wasmtime refuses `instantiate` and `call` on such a store), and
the sidecar hosts every module, of every world, on synchronous stores. The
0.0.1 interface is the WASI 0.2 one: the same operations are blocking
functions, which a synchronous store can host. When the sidecar's stores go
async, this copy should give way to upstream's crate and the 0.3 interface,
and C guests to [wasi-webgpu-headers](https://github.com/wasi-gfx/wasi-webgpu-headers).

## What changed from upstream

* wasmtime 46 to 48; the context is a borrowed view (`WasiWebGpuCtx`: the wgpu
  instance and the resource table), the shape upstream's later crate has.
* `map-async` is synchronous: it maps and polls the devices until the map
  lands. Upstream's was a wasmtime async function, which needs an async store.
* `on-submitted-work-done` is implemented, the same way.
* Compute only: surfaces, canvases and window handles are gone (no
  raw-window-handle, no UI thread). `wasi:graphics-context`, which webgpu's
  types name, is linked with every function refusing.
* `subgroups` maps to wgpu's `FeaturesWGPU::SUBGROUP`, which is how wgpu 29
  offers the WGSL subgroup builtins (it has no standard `subgroups` yet).
* Adapter requests try Vulkan, then Metal, then D3D12, then GL: on Windows
  the same GPU shows up under Vulkan and D3D12, and wgpu's D3D12 reports no
  subgroups unless it can load DXC. `WGPU_BACKEND` narrows the instance's
  backends as wgpu reads it.
* Implemented where upstream had `todo!()` and PyroWave needs them: the
  pipeline constant and required limit records (ported from upstream's later
  crate), storage texture binding layouts, the WGSL language features, and
  getting a mapped range of a buffer that is not mapped (an error, not a
  panic). Unknown `required-limits` keys and features the adapter lacks reject
  the device request as WebGPU says.
* No host panics a guest can reach through a call that is not implemented:
  every remaining `todo!()` is a trap naming the function, and passes used
  after `end` trap instead of unwrapping.
* Uncaptured errors are printed to stderr by the host: 0.0.1's subscription
  is a bare pollable and cannot carry the error to the guest.
* `wgpu-core` features per platform: Vulkan on Linux and Android; Vulkan and
  D3D12 on Windows; Metal on macOS and iOS; GLES elsewhere. No `noop` backend.

## WIT

`wit/deps/` holds `wasi:webgpu@0.0.1` and `wasi:graphics-context@0.0.1` from
the wasi-gfx proposal's v0.0.1 branch (the copies wasi-gfx-runtime 301716b and
wasi-webgpu-headers 1bb9092 both carried, identical), and `wasi:io@0.2.0`.
`wit/world.wit` is the world the bindings are generated for.
