//! Monocular depth: every frame leaves as a greyscale map of how far away it
//! is, near bright and far dark.
//!
//! The graph is Depth Anything V2 (small), run through `wasi:nn`. The module
//! never opens a file - the host binds the graph to a name with
//! `-nn depth=<path>` and this module asks for that name and nothing else.
//!
//! One frame in, one out, at the same geometry, so the map can be fed
//! straight back into a module that reads the frame beside it. The host
//! makes each frame into the graph's input (`ffrwd:gpu/nn.frame-tensor`):
//! letterboxed into the square the graph takes and normalized the way the
//! model was trained. The graph's map is then min-max normalized over the
//! frame's own range and sampled back out to the frame's own size.
//!
//! `memory` says where the frames are held. On `host`, the default, the
//! tensor is made on the CPU, the map comes back as bytes, and the frame
//! leaves in the input's own pixel format. On `gpu`, the input is held on
//! the device, the tensor is made there, the map is bound to a texture
//! (`ffrwd:gpu/nn.output-texture`), and two passes on the device find its
//! range and paint the frame, which leaves as an rgba texture: nothing
//! crosses to host memory where the model's session can read and write the
//! device's.

// `generate_all`: the world's interfaces come from other packages -
// ffrwd:av, ffrwd:gpu, wasi:nn and wasi:webgpu - and without it bindgen
// expects them to have been generated somewhere else.
wit_bindgen::generate!({
    path: ["../../wit", "../../wit-gpu", "wit"],
    // Fully qualified: several packages are in scope, and each has worlds.
    world: "ffrwd:depth/depth",
    generate_all,
});

use std::cell::RefCell;

use exports::ffrwd::av::node::{Emission, Emitted, Guest, Payload};
use exports::ffrwd::gpu::ports::Guest as PortsGuest;
use ffrwd::av::node_tick::Tick;
use ffrwd::av::node_types::{
    Accepts, Binding, BoundStream, Clock, InputPort, LikeInput, NodeShape, OutputFormat,
    OutputPort, Pairing, PortKind, RowsUse,
};
use ffrwd::av::types::{Meta, RawFrame, Wants};
use ffrwd::gpu::frames::emit;
use ffrwd::gpu::nn::{
    frame_tensor, output_texture, ChannelOrder, Fit, FrameTensorOptions, OutputTextureOptions,
    TensorLayout,
};
use ffrwd::gpu::types::{FrameMemory, PortMemory};
use serde::Deserialize;
use wasi::nn::graph::{load_by_name, Graph};
use wasi::nn::inference::GraphExecutionContext;
use wasi::webgpu::webgpu::{
    get_gpu, GpuBindGroupDescriptor, GpuBindGroupEntry, GpuBindGroupLayout,
    GpuBindGroupLayoutDescriptor, GpuBindGroupLayoutEntry, GpuBindingResource, GpuBuffer,
    GpuBufferBinding, GpuBufferBindingLayout, GpuBufferBindingType, GpuBufferDescriptor,
    GpuBufferUsage, GpuComputePipeline, GpuComputePipelineDescriptor, GpuDevice, GpuExtent3D,
    GpuLayoutMode, GpuPipelineLayoutDescriptor, GpuProgrammableStage, GpuShaderModuleDescriptor,
    GpuShaderStage, GpuStorageTextureAccess, GpuStorageTextureBindingLayout, GpuTexture,
    GpuTextureBindingLayout, GpuTextureDescriptor, GpuTextureFormat, GpuTextureSampleType,
    GpuTextureUsage, GpuTextureView, GpuTextureViewDimension,
};

/// The name the host binds the graph to. `-nn depth=<path>`.
const MODEL: &str = "depth";

/// The square the graph is run at. Depth Anything works in patches of 14, and
/// 518 is 37 of them - the size its own preprocessing uses.
const SIDE: usize = 518;

/// The graph's own names for the tensors it takes and returns.
const INPUT_NAME: &str = "pixel_values";
const OUTPUT_NAME: &str = "predicted_depth";

/// What the model was trained on: each channel rescaled to 0..1, then
/// standardized by these.
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

const SCHEMA: &str = r#"{"type":"object","properties":{"memory":{"type":"string","enum":["host","gpu"]}},"additionalProperties":false}"#;

/// The passes that find the map's range over the frame and paint the frame:
/// `extent` samples the map at each frame pixel and keeps the least and
/// greatest, `paint` samples it again and writes the grey. Each sample is
/// the host path's ([`to_grey`]), in its order of operations.
const PASSES: &str = r#"
struct Params {
    frame_w: u32,
    frame_h: u32,
    grid_w: u32,
    grid_h: u32,
    scale: f32,
    offset_x: f32,
    offset_y: f32,
    side: f32,
}

@group(0) @binding(0) var map: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> range: array<atomic<u32>, 2>;
@group(0) @binding(2) var<uniform> p: Params;
@group(0) @binding(3) var frame: texture_storage_2d<rgba8unorm, write>;

fn at(x: u32, y: u32) -> f32 {
    return textureLoad(map, vec2<i32>(i32(x), i32(y)), 0).r;
}

// The map's value at frame pixel (x, y): where the pixel lands in the
// square, carried onto the map by ratio, read bilinearly.
fn sample(x: u32, y: u32) -> f32 {
    let gx = (p.offset_x + (f32(x) + 0.5) * p.scale - 0.5) * (f32(p.grid_w) / p.side);
    let gy = (p.offset_y + (f32(y) + 0.5) * p.scale - 0.5) * (f32(p.grid_h) / p.side);
    let cx = clamp(gx, 0.0, f32(p.grid_w - 1u));
    let cy = clamp(gy, 0.0, f32(p.grid_h - 1u));
    let x0 = u32(floor(cx));
    let y0 = u32(floor(cy));
    let x1 = min(x0 + 1u, p.grid_w - 1u);
    let y1 = min(y0 + 1u, p.grid_h - 1u);
    let tx = cx - floor(cx);
    let ty = cy - floor(cy);
    let top = at(x0, y0) + (at(x1, y0) - at(x0, y0)) * tx;
    let bottom = at(x0, y1) + (at(x1, y1) - at(x0, y1)) * tx;
    return top + (bottom - top) * ty;
}

// A float as a u32 that orders as the float does.
fn key(v: f32) -> u32 {
    let b = bitcast<u32>(v);
    if ((b & 0x80000000u) != 0u) {
        return ~b;
    }
    return b | 0x80000000u;
}

fn unkey(k: u32) -> f32 {
    if ((k & 0x80000000u) != 0u) {
        return bitcast<f32>(k & 0x7fffffffu);
    }
    return bitcast<f32>(~k);
}

@compute @workgroup_size(8, 8)
fn extent(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= p.frame_w || id.y >= p.frame_h) {
        return;
    }
    let k = key(sample(id.x, id.y));
    atomicMin(&range[0], k);
    atomicMax(&range[1], k);
}

@compute @workgroup_size(8, 8)
fn paint(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= p.frame_w || id.y >= p.frame_h) {
        return;
    }
    let low = unkey(atomicLoad(&range[0]));
    let high = unkey(atomicLoad(&range[1]));
    let span = high - low;
    var grey = 128.0;
    if (span > 0.0) {
        grey = clamp(floor((sample(id.x, id.y) - low) / span * 255.0 + 0.5), 0.0, 255.0);
    }
    let c = grey / 255.0;
    textureStore(frame, vec2<i32>(i32(id.x), i32(id.y)), vec4<f32>(c, c, c, 1.0));
}
"#;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Params {
    #[serde(default = "host")]
    memory: String,
}

fn host() -> String {
    "host".to_string()
}

fn parse(text: &str) -> Result<Params, String> {
    let params: Params = if text.trim().is_empty() {
        Params { memory: host() }
    } else {
        serde_json::from_str(text).map_err(|e| format!("depth: params: {e}"))?
    };
    if params.memory != "gpu" && params.memory != "host" {
        return Err(format!(
            "depth: memory is host or gpu, not '{}'",
            params.memory
        ));
    }
    Ok(params)
}

/// The pixel format of the instance's input, for the host path's frames.
#[derive(Clone, Copy, PartialEq)]
enum PixFmt {
    Yuv420p,
    Rgba,
}

/// Where the frame sits inside the square the graph is run at, once it has
/// been scaled to fit with its shape kept: the host's letterbox.
#[derive(Clone, Copy)]
struct Letterbox {
    /// Square pixels per frame pixel.
    scale: f32,
    /// Where the scaled frame starts inside the square.
    offset_x: f32,
    offset_y: f32,
}

impl Letterbox {
    fn new(width: usize, height: usize) -> Letterbox {
        let scale = (SIDE as f32 / width as f32).min(SIDE as f32 / height as f32);
        let scaled_w = ((width as f32 * scale).round() as usize).clamp(1, SIDE);
        let scaled_h = ((height as f32 * scale).round() as usize).clamp(1, SIDE);
        Letterbox {
            scale,
            offset_x: ((SIDE - scaled_w) / 2) as f32,
            offset_y: ((SIDE - scaled_h) / 2) as f32,
        }
    }

    /// Where a frame pixel lands in the square, in square coordinates.
    fn to_square(self, x: usize, y: usize) -> (f32, f32) {
        (
            self.offset_x + (x as f32 + 0.5) * self.scale - 0.5,
            self.offset_y + (y as f32 + 0.5) * self.scale - 0.5,
        )
    }
}

/// The device's half: the map's texture, the passes and what they bind.
struct Device {
    device: GpuDevice,
    map: GpuTexture,
    map_view: GpuTextureView,
    layout: GpuBindGroupLayout,
    extent: GpuComputePipeline,
    paint: GpuComputePipeline,
    range: GpuBuffer,
    params: GpuBuffer,
}

/// What `init` settled, plus the graph it loaded.
struct Opened {
    v: u32,
    width: usize,
    height: usize,
    pix_fmt: PixFmt,
    letterbox: Letterbox,
    options: FrameTensorOptions,
    /// Held for the life of the instance: building it once is what keeps a
    /// provider's kernels from being chosen again per frame.
    context: GraphExecutionContext,
    /// Kept alive because the context is only valid while its graph is.
    _graph: Graph,
    /// On `memory=gpu`.
    gpu: Option<Device>,
}

thread_local! {
    static OPENED: RefCell<Option<Opened>> = const { RefCell::new(None) };
}

/// The spec's spelling of an error code, so a message says what actually
/// went wrong rather than how this module happens to format things.
fn failed(what: &str, error: &wasi::nn::errors::Error) -> String {
    use wasi::nn::errors::ErrorCode;
    let code = match error.code() {
        ErrorCode::InvalidArgument => "invalid-argument",
        ErrorCode::InvalidEncoding => "invalid-encoding",
        ErrorCode::Timeout => "timeout",
        ErrorCode::RuntimeError => "runtime-error",
        ErrorCode::UnsupportedOperation => "unsupported-operation",
        ErrorCode::TooLarge => "too-large",
        ErrorCode::NotFound => "not-found",
        ErrorCode::Security => "security",
        ErrorCode::Unknown => "unknown",
    };
    format!("depth: {what}: {code} ({})", error.data())
}

/// The graph's output as a plain grid of floats, whichever rank it came back
/// with: [1, h, w] as this graph spells it, or [1, 1, h, w] as an export that
/// keeps the channel does.
fn depth_grid(dimensions: &[u32], data: &[u8]) -> Result<(Vec<f32>, usize, usize), String> {
    let (h, w) = match dimensions {
        [_, h, w] => (*h as usize, *w as usize),
        [_, _, h, w] => (*h as usize, *w as usize),
        other => {
            return Err(format!(
                "depth: the graph returned a depth map of {} dimension(s), expected 3 or 4",
                other.len()
            ))
        }
    };
    let (whole, _) = data.as_chunks::<4>();
    let values: Vec<f32> = whole.iter().copied().map(f32::from_le_bytes).collect();
    if values.len() < h * w {
        return Err(format!(
            "depth: the graph returned {} value(s) for a {h}x{w} depth map",
            values.len()
        ));
    }
    Ok((values, h, w))
}

/// The depth map sampled back out to the frame's own size and scaled to fill
/// a byte. The range is the frame's own: the nearest thing in it is 255 and
/// the furthest 0, so a flat scene still uses the whole scale.
fn to_grey(
    grid: &[f32],
    grid_w: usize,
    grid_h: usize,
    width: usize,
    height: usize,
    box_: Letterbox,
) -> Vec<u8> {
    // The graph's grid may not be the square it was given, so square
    // coordinates are carried onto it by ratio.
    let gx_per_square = grid_w as f32 / SIDE as f32;
    let gy_per_square = grid_h as f32 / SIDE as f32;

    // Where each frame column and row reads from on the grid. Neither depends
    // on the other axis, so both are built once and the per-pixel work is
    // what is left: two rows of the grid mixed along their length.
    let place = |g: f32, extent: usize| -> (usize, usize, f32) {
        let g = g.clamp(0.0, (extent - 1) as f32);
        let base = g.floor() as usize;
        (base, (base + 1).min(extent - 1), g - base as f32)
    };
    let columns: Vec<(usize, usize, f32)> = (0..width)
        .map(|x| place(box_.to_square(x, 0).0 * gx_per_square, grid_w))
        .collect();

    // The range is taken over what the frame covers, so the black bars never
    // stretch it.
    let mut raw = vec![0f32; width * height];
    let (mut low, mut high) = (f32::INFINITY, f32::NEG_INFINITY);
    for y in 0..height {
        let (y0, y1, ty) = place(box_.to_square(0, y).1 * gy_per_square, grid_h);
        let top_row = &grid[y0 * grid_w..(y0 + 1) * grid_w];
        let bottom_row = &grid[y1 * grid_w..(y1 + 1) * grid_w];
        let target = &mut raw[y * width..(y + 1) * width];
        for (sample, (x0, x1, tx)) in target.iter_mut().zip(&columns) {
            let top = top_row[*x0] + (top_row[*x1] - top_row[*x0]) * tx;
            let bottom = bottom_row[*x0] + (bottom_row[*x1] - bottom_row[*x0]) * tx;
            let value = top + (bottom - top) * ty;
            *sample = value;
            low = low.min(value);
            high = high.max(value);
        }
    }

    let span = high - low;
    let mut out = vec![0u8; width * height];
    if span <= 0.0 {
        // A frame the model read as one flat distance. Mid grey says so,
        // rather than a scale that divides by nothing.
        out.fill(128);
        return out;
    }
    for (sample, value) in out.iter_mut().zip(&raw) {
        *sample = (((value - low) / span) * 255.0).round().clamp(0.0, 255.0) as u8;
    }
    out
}

/// A greyscale map written as a frame of the instance's own format: the luma
/// plane with neutral chroma, or the same value in red, green and blue.
fn to_frame(grey: &[u8], pix_fmt: PixFmt, width: usize, height: usize) -> Vec<u8> {
    match pix_fmt {
        PixFmt::Yuv420p => {
            let chroma = width.div_ceil(2) * height.div_ceil(2);
            let mut out = vec![128u8; width * height + 2 * chroma];
            out[..width * height].copy_from_slice(grey);
            out
        }
        PixFmt::Rgba => {
            let mut out = vec![0u8; width * height * 4];
            for (index, value) in grey.iter().enumerate() {
                let base = index * 4;
                out[base] = *value;
                out[base + 1] = *value;
                out[base + 2] = *value;
                out[base + 3] = 255;
            }
            out
        }
    }
}

/// The square's letterbox spec, as the host makes the tensor.
fn tensor_options() -> FrameTensorOptions {
    FrameTensorOptions {
        model: MODEL.to_string(),
        width: SIDE as u32,
        height: SIDE as u32,
        layout: TensorLayout::Nchw,
        order: ChannelOrder::Rgb,
        fit: Fit::Letterbox,
        pad: 0.0,
        scale: 1.0,
        mean: (MEAN[0], MEAN[1], MEAN[2]),
        std: (STD[0], STD[1], STD[2]),
    }
}

/// Opens the device's half: the map bound to a texture, the passes, and
/// their buffers.
fn open_device(
    context: &GraphExecutionContext,
    opened_box: Letterbox,
    width: u32,
    height: u32,
) -> Result<Device, String> {
    let device = get_gpu()
        .request_adapter(None)
        .ok_or("depth: no GPU adapter; memory=host runs without one")?
        .request_device(None)
        .map_err(|_| "depth: the adapter gave no device".to_string())?;
    let map = output_texture(
        context,
        &OutputTextureOptions {
            output: OUTPUT_NAME.to_string(),
            width: SIDE as u32,
            height: SIDE as u32,
        },
    )
    .map_err(|e| format!("depth: {e}"))?;
    let map_view = map.create_view(None);
    let compute = GpuShaderStage::compute();
    let entry = |binding| GpuBindGroupLayoutEntry {
        binding,
        visibility: compute,
        buffer: None,
        sampler: None,
        texture: None,
        storage_texture: None,
    };
    let layout = device.create_bind_group_layout(&GpuBindGroupLayoutDescriptor {
        entries: vec![
            GpuBindGroupLayoutEntry {
                texture: Some(GpuTextureBindingLayout {
                    sample_type: Some(GpuTextureSampleType::UnfilterableFloat),
                    view_dimension: Some(GpuTextureViewDimension::D2),
                    multisampled: Some(false),
                }),
                ..entry(0)
            },
            GpuBindGroupLayoutEntry {
                buffer: Some(GpuBufferBindingLayout {
                    type_: Some(GpuBufferBindingType::Storage),
                    has_dynamic_offset: None,
                    min_binding_size: None,
                }),
                ..entry(1)
            },
            GpuBindGroupLayoutEntry {
                buffer: Some(GpuBufferBindingLayout {
                    type_: Some(GpuBufferBindingType::Uniform),
                    has_dynamic_offset: None,
                    min_binding_size: None,
                }),
                ..entry(2)
            },
            GpuBindGroupLayoutEntry {
                storage_texture: Some(GpuStorageTextureBindingLayout {
                    access: Some(GpuStorageTextureAccess::WriteOnly),
                    format: GpuTextureFormat::Rgba8unorm,
                    view_dimension: Some(GpuTextureViewDimension::D2),
                }),
                ..entry(3)
            },
        ],
        label: None,
    });
    let pipeline_layout = device.create_pipeline_layout(&GpuPipelineLayoutDescriptor {
        bind_group_layouts: vec![Some(&layout)],
        label: None,
    });
    let module = device.create_shader_module(&GpuShaderModuleDescriptor {
        code: PASSES.to_string(),
        compilation_hints: None,
        label: Some("depth passes".to_string()),
    });
    let pipeline = |entry_point: &str| {
        device.create_compute_pipeline(GpuComputePipelineDescriptor {
            compute: GpuProgrammableStage {
                module: &module,
                entry_point: Some(entry_point.to_string()),
                constants: None,
            },
            layout: GpuLayoutMode::Specific(&pipeline_layout),
            label: None,
        })
    };
    let extent = pipeline("extent");
    let paint = pipeline("paint");
    let range = device.create_buffer(&GpuBufferDescriptor {
        size: 8,
        usage: GpuBufferUsage::storage() | GpuBufferUsage::copy_dst(),
        mapped_at_creation: None,
        label: None,
    });
    let params = device.create_buffer(&GpuBufferDescriptor {
        size: 32,
        usage: GpuBufferUsage::uniform() | GpuBufferUsage::copy_dst(),
        mapped_at_creation: None,
        label: None,
    });
    let mut words = Vec::with_capacity(32);
    for word in [width, height, SIDE as u32, SIDE as u32] {
        words.extend_from_slice(&word.to_le_bytes());
    }
    for value in [
        opened_box.scale,
        opened_box.offset_x,
        opened_box.offset_y,
        SIDE as f32,
    ] {
        words.extend_from_slice(&value.to_le_bytes());
    }
    device
        .queue()
        .write_buffer_with_copy(&params, 0, &words, None, None)
        .map_err(|_| "depth: writing the passes' parameters".to_string())?;
    Ok(Device {
        device,
        map,
        map_view,
        layout,
        extent,
        paint,
        range,
        params,
    })
}

/// One frame through the graph on the host path: its bytes out.
fn run_host(opened: &Opened, index: u32) -> Result<Vec<u8>, String> {
    let tensor =
        frame_tensor(opened.v, index, &opened.options).map_err(|e| format!("depth: {e}"))?;
    let outputs = opened
        .context
        .compute(vec![(INPUT_NAME.to_string(), tensor)])
        .map_err(|e| failed("compute", &e))?;
    let out = outputs
        .into_iter()
        .find(|(name, _)| name == OUTPUT_NAME)
        .map(|(_, tensor)| tensor)
        .ok_or_else(|| format!("depth: the graph returned no tensor named {OUTPUT_NAME}"))?;
    let (grid, grid_h, grid_w) = depth_grid(&out.dimensions(), &out.data())?;
    let grey = to_grey(
        &grid,
        grid_w,
        grid_h,
        opened.width,
        opened.height,
        opened.letterbox,
    );
    Ok(to_frame(&grey, opened.pix_fmt, opened.width, opened.height))
}

/// One frame through the graph on the device: the map's texture painted
/// into a new rgba texture, which is the frame that leaves.
fn run_gpu(opened: &Opened, gpu: &Device, index: u32) -> Result<GpuTexture, String> {
    let tensor =
        frame_tensor(opened.v, index, &opened.options).map_err(|e| format!("depth: {e}"))?;
    // The map is in the texture once this returns; the listed output is
    // not read, so nothing comes back to host memory.
    opened
        .context
        .compute(vec![(INPUT_NAME.to_string(), tensor)])
        .map_err(|e| failed("compute", &e))?;
    let (width, height) = (opened.width as u32, opened.height as u32);
    let frame = gpu.device.create_texture(&GpuTextureDescriptor {
        size: GpuExtent3D {
            width,
            height: Some(height),
            depth_or_array_layers: None,
        },
        mip_level_count: None,
        sample_count: None,
        dimension: None,
        format: GpuTextureFormat::Rgba8unorm,
        usage: GpuTextureUsage::storage_binding()
            | GpuTextureUsage::texture_binding()
            | GpuTextureUsage::copy_src(),
        view_formats: None,
        label: None,
    });
    let frame_view = frame.create_view(None);
    let group = gpu.device.create_bind_group(&GpuBindGroupDescriptor {
        layout: &gpu.layout,
        entries: vec![
            GpuBindGroupEntry {
                binding: 0,
                resource: GpuBindingResource::GpuTextureView(&gpu.map_view),
            },
            GpuBindGroupEntry {
                binding: 1,
                resource: GpuBindingResource::GpuBufferBinding(GpuBufferBinding {
                    buffer: &gpu.range,
                    offset: None,
                    size: None,
                }),
            },
            GpuBindGroupEntry {
                binding: 2,
                resource: GpuBindingResource::GpuBufferBinding(GpuBufferBinding {
                    buffer: &gpu.params,
                    offset: None,
                    size: None,
                }),
            },
            GpuBindGroupEntry {
                binding: 3,
                resource: GpuBindingResource::GpuTextureView(&frame_view),
            },
        ],
        label: None,
    });
    let queue = gpu.device.queue();
    // The least as high as a key goes, the greatest as low.
    let mut reset = Vec::with_capacity(8);
    reset.extend_from_slice(&u32::MAX.to_le_bytes());
    reset.extend_from_slice(&0u32.to_le_bytes());
    queue
        .write_buffer_with_copy(&gpu.range, 0, &reset, None, None)
        .map_err(|_| "depth: resetting the range".to_string())?;
    let encoder = gpu.device.create_command_encoder(None);
    for pipeline in [&gpu.extent, &gpu.paint] {
        let pass = encoder.begin_compute_pass(None);
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, Some(&group), None, None, None)
            .map_err(|_| "depth: binding the passes' resources".to_string())?;
        pass.dispatch_workgroups(width.div_ceil(8), Some(height.div_ceil(8)), None);
        pass.end();
    }
    queue.submit(&[&encoder.finish(None)]);
    let _ = &gpu.map;
    Ok(frame)
}

struct Depth;

impl Guest for Depth {
    fn describe() -> Meta {
        Meta {
            name: "depth".to_string(),
            version: "0.2.0".to_string(),
            params_schema: SCHEMA.to_string(),
            rows_schema: String::new(),
            pixel_formats: Vec::new(),
            sample_formats: Vec::new(),
            sample_rates: Vec::new(),
            channel_counts: Vec::new(),
            rows_language: Vec::new(),
        }
    }

    fn shape(params: String, _bound: Vec<Binding>) -> Result<NodeShape, String> {
        let on_gpu = parse(&params)?.memory == "gpu";
        // On the device the host makes the tensor from any format it holds
        // there; on the host this module writes the map in the input's.
        let pixel_formats: &[&str] = if on_gpu {
            &["rgba", "yuv420p", "yuv422p", "yuv444p"]
        } else {
            &["yuv420p", "rgba"]
        };
        let v = InputPort {
            name: "v".to_string(),
            kind: PortKind::Video,
            required: true,
            many: false,
            pairing: Pairing::Lockstep,
            rows: RowsUse::Ignore,
            window: 1,
            stride: 1,
            accepts: Accepts {
                pixel_formats: pixel_formats.iter().map(|f| f.to_string()).collect(),
                sample_formats: Vec::new(),
                sample_rates: Vec::new(),
                channel_counts: Vec::new(),
                codecs: Vec::new(),
                wants: Wants::All,
                like: None,
            },
            schema: None,
        };
        let out = OutputPort {
            name: "out".to_string(),
            kind: PortKind::Video,
            // A texture handed back is rgba.
            format: Some(OutputFormat::Like(LikeInput {
                port: "v".to_string(),
                pixel_format: on_gpu.then(|| "rgba".to_string()),
                sample_format: None,
            })),
            time_base: None,
            latency: 0.0,
            schema: None,
            row: None,
        };
        Ok(NodeShape {
            inputs: vec![v],
            outputs: vec![out],
            clock: Clock::Input("v".to_string()),
            pure: true,
            one_to_one: true,
            bounded: true,
            relation: Vec::new(),
        })
    }

    fn init(bound: Vec<BoundStream>, _latched: Vec<String>, params: String) -> Result<(), String> {
        let params = parse(&params)?;
        let v = bound
            .iter()
            .find(|b| b.port == "v")
            .ok_or("depth: v is bound")?;
        let Some(OutputFormat::Video(video)) = &v.format else {
            return Err("depth reads frames, and this stream is not video".to_string());
        };
        let on_gpu = params.memory == "gpu";
        let pix_fmt = match video.pix_fmt.as_str() {
            "yuv420p" => PixFmt::Yuv420p,
            "rgba" => PixFmt::Rgba,
            _ if on_gpu => PixFmt::Rgba,
            other => return Err(format!("depth does not accept pixel format {other}")),
        };
        let (width, height) = (video.width as usize, video.height as usize);
        let letterbox = Letterbox::new(width, height);

        // The graph is loaded once per instance, and the session built once:
        // the first frame is what a provider picks its kernels on, and every
        // frame after it reuses them.
        let graph =
            load_by_name(MODEL).map_err(|e| failed(&format!("load-by-name({MODEL:?})"), &e))?;
        let context = graph
            .init_execution_context()
            .map_err(|e| failed("init-execution-context", &e))?;
        let gpu = if on_gpu {
            Some(open_device(&context, letterbox, video.width, video.height)?)
        } else {
            None
        };

        OPENED.with(|o| {
            *o.borrow_mut() = Some(Opened {
                v: v.id,
                width,
                height,
                pix_fmt,
                letterbox,
                options: tensor_options(),
                context,
                _graph: graph,
                gpu,
            });
        });
        Ok(())
    }

    fn set_params(params: String) -> Result<(), String> {
        parse(&params).map(|_| ())
    }

    fn process(tick: &Tick) -> Result<Emitted, String> {
        OPENED.with(|opened| {
            let borrowed = opened.borrow();
            let opened = borrowed.as_ref().ok_or("depth: process before init")?;
            let mut items = Vec::new();
            for frame in tick.frames(opened.v) {
                match &opened.gpu {
                    Some(gpu) => {
                        let texture = run_gpu(opened, gpu, frame.index)?;
                        emit("out", frame.pts, frame.duration, texture)?;
                    }
                    None => items.push(Emission {
                        port: "out".to_string(),
                        payload: Payload::Frame(RawFrame {
                            pts: frame.pts,
                            duration: frame.duration,
                            data: run_host(opened, frame.index)?,
                        }),
                    }),
                }
            }
            Ok(Emitted {
                items,
                rows: Vec::new(),
                finished: false,
            })
        })
    }
}

impl PortsGuest for Depth {
    fn memory(params: String) -> Vec<PortMemory> {
        match parse(&params) {
            Ok(params) if params.memory == "gpu" => ["v", "out"]
                .into_iter()
                .map(|port| PortMemory {
                    port: port.to_string(),
                    memory: FrameMemory::Gpu,
                })
                .collect(),
            _ => Vec::new(),
        }
    }
}

export!(Depth);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_wide_frame_is_letterboxed_with_bars_above_and_below() {
        let box_ = Letterbox::new(1024, 512);
        assert_eq!(box_.offset_x, 0.0, "the wide side fills the square");
        // 512 frame pixels at 518/1024 to a pixel are 259, centred.
        assert_eq!(box_.offset_y, 129.0, "the bars are above and below");
    }

    #[test]
    fn a_square_frame_fills_the_square() {
        let box_ = Letterbox::new(256, 256);
        assert_eq!((box_.offset_x, box_.offset_y), (0.0, 0.0));
    }

    #[test]
    fn a_depth_map_comes_back_from_three_dimensions_or_four() {
        let data: Vec<u8> = (0..6).flat_map(|v| (v as f32).to_le_bytes()).collect();
        let (three, h, w) = depth_grid(&[1, 2, 3], &data).expect("rank 3");
        assert_eq!((h, w), (2, 3));
        assert_eq!(three.len(), 6);
        let (four, h, w) = depth_grid(&[1, 1, 2, 3], &data).expect("rank 4");
        assert_eq!((h, w), (2, 3));
        assert_eq!(four, three, "the same grid, however it was shaped");
    }

    #[test]
    fn a_map_with_no_range_at_all_comes_out_mid_grey() {
        let flat = vec![7.0f32; SIDE * SIDE];
        let grey = to_grey(&flat, SIDE, SIDE, 4, 4, Letterbox::new(4, 4));
        assert!(
            grey.iter().all(|v| *v == 128),
            "one flat distance is no scale to normalize onto"
        );
    }

    #[test]
    fn the_range_is_stretched_to_fill_a_byte() {
        // A ramp across the square: whatever its raw values, the frame it
        // becomes runs from 0 to 255.
        let mut grid = vec![0f32; SIDE * SIDE];
        for y in 0..SIDE {
            for x in 0..SIDE {
                grid[y * SIDE + x] = 100.0 + x as f32 * 0.5;
            }
        }
        let grey = to_grey(&grid, SIDE, SIDE, 64, 64, Letterbox::new(64, 64));
        assert_eq!(*grey.iter().min().expect("pixels"), 0);
        assert_eq!(*grey.iter().max().expect("pixels"), 255);
    }

    #[test]
    fn a_grey_map_writes_neutral_chroma_and_opaque_alpha() {
        let grey = vec![40u8; 4 * 4];
        let yuv = to_frame(&grey, PixFmt::Yuv420p, 4, 4);
        assert_eq!(yuv.len(), 4 * 4 + 2 * 2 * 2);
        assert!(
            yuv[..16].iter().all(|v| *v == 40),
            "the luma plane is the map"
        );
        assert!(
            yuv[16..].iter().all(|v| *v == 128),
            "and the chroma is neutral"
        );

        let rgba = to_frame(&grey, PixFmt::Rgba, 4, 4);
        let (pixels, _) = rgba.as_chunks::<4>();
        for pixel in pixels {
            assert_eq!(
                *pixel,
                [40, 40, 40, 255],
                "equal in every channel, and opaque"
            );
        }
    }
}
