//! Instance segmentation: every frame leaves as an index map naming which
//! object owns each pixel, and one row per object beside it.
//!
//! The graph is YOLOv8n-seg, run through `wasi:nn`. The module never opens a
//! file - the host binds the graph to a name with `-nn segment=<path>` and this
//! module asks for that name and nothing else.
//!
//! # The id-as-luma encoding
//!
//! The frame that leaves is greyscale, the same geometry and pixel format the
//! instance was opened for, and the id of the detection owning a pixel goes
//! straight into that pixel's luma. 0 is background, 1 is the first detection,
//! 2 the second, and so on: id 1 is luma 1, id 3 is luma 3. The picture is
//! therefore almost black to look at - it is a lookup table, not a rendering,
//! and a downstream module reads a pixel's value as an index into the rows.
//! In yuv420p the ids are the Y plane and the chroma is neutral; in rgba the
//! id is in red, green and blue alike, opaque. Ids run in descending
//! confidence, so id 1 is the frame's surest object, and where two detections
//! claim a pixel the surer one keeps it.
//!
//! At most 255 detections fit in a byte, and a frame with more is refused
//! rather than wrapped or truncated.
//!
//! The rows carry `id`, `class`, `score` and the box as `x`, `y`, `w`, `h` in
//! the frame's own pixels, so an id read out of the map has somewhere to go.
//!
//! One source, two builds. `segment.wasm` is a window filter that makes the
//! graph's input itself, on the CPU, and imports nothing of the GPU's, so any
//! sidecar runs it; its rows ride the map's frames. `segment_gpu.wasm` (the
//! `gpu` feature, built by the `segment-gpu` crate) is a node beside
//! `ffrwd:gpu` whose map and rows leave on ports of their own: the host makes
//! the graph's input, and on `memory=gpu` the frames stay on the device. It
//! imports `wasi:webgpu`, so it runs only where the sidecar is granted the
//! device with `-gpu`.

mod common;

#[cfg(not(feature = "gpu"))]
mod host;

#[cfg(feature = "gpu")]
mod gpu;
