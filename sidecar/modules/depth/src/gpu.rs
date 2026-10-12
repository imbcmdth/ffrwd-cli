//! The GPU build: a node of the node world beside `ffrwd:gpu`. The host
//! makes each frame into the graph's input (`ffrwd:gpu/nn.frame-tensor`):
//! letterboxed into the square the graph takes and normalized the way the
//! model was trained.
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
// expects them to have been generated somewhere else. The paths are the
// `depth` crate's, so the `depth-gpu` crate reaching this file finds them
// too.
wit_bindgen::generate!({
    path: ["../../wit", "../../wit-gpu", "../depth/wit-gpu"],
    // Fully qualified: several packages are in scope, and each has worlds.
    world: "ffrwd:depth/depth-gpu",
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

use crate::common::{
    depth_grid, to_frame, to_grey, Letterbox, PixFmt, INPUT_NAME, MEAN, MODEL, OUTPUT_NAME, SIDE,
    STD,
};

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
