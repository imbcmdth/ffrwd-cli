//! Monocular depth: every frame leaves as a greyscale map of how far away it
//! is, near bright and far dark.
//!
//! The graph is Depth Anything V2 (small), run through `wasi:nn`. The module
//! never opens a file - the host binds the graph to a name with
//! `-nn depth=<path>` and this module asks for that name and nothing else.
//!
//! One source, two builds. `depth.wasm` is a window filter that makes the
//! graph's input itself, on the CPU, and imports nothing of the GPU's, so any
//! sidecar runs it. `depth_gpu.wasm` (the `gpu` feature, built by the
//! `depth-gpu` crate) is a node beside `ffrwd:gpu`: the host makes the
//! graph's input, and on `memory=gpu` the frames stay on the device. It
//! imports `wasi:webgpu`, so it runs only where the sidecar is granted the
//! device with `-gpu`.

mod common;

#[cfg(not(feature = "gpu"))]
mod host;

#[cfg(feature = "gpu")]
mod gpu;
