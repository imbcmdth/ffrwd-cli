//! The GPU build: a node of the node world beside `ffrwd:gpu`. The map leaves
//! on `map` and the rows on `objects`, a data port of their own, one message
//! per object at its frame's pts. The host makes each frame into the graph's
//! input (`ffrwd:gpu/nn.frame-tensor`), letterboxed onto the grey the model
//! was trained on.
//!
//! `memory` says where the frames are held. On `host`, the default, the
//! tensor is made on the CPU, both outputs come back as bytes, and the map
//! leaves in the input's own pixel format. On `gpu`, the input is held on the
//! device, the tensor is made there, and the mask prototypes are bound to a
//! texture (`ffrwd:gpu/nn.output-texture`): a pass on the device spends each
//! detection's coefficients on them and paints the map, which leaves as an
//! rgba texture. The predictions still come back to host memory, since the
//! detections are decoded and thinned there; the prototypes, the larger
//! output, never do where the model's session writes the device's memory.

// `generate_all`: the world's interfaces come from other packages -
// ffrwd:av, ffrwd:gpu, wasi:nn and wasi:webgpu - and without it bindgen
// expects them to have been generated somewhere else. The paths are the
// `segment` crate's, so the `segment-gpu` crate reaching this file finds them
// too.
wit_bindgen::generate!({
    path: ["../../wit", "../../wit-gpu", "../segment/wit-gpu"],
    // Fully qualified: several packages are in scope, and each has worlds.
    world: "ffrwd:segment/segment-gpu",
    generate_all,
});

use std::cell::{Cell, RefCell};

use exports::ffrwd::av::node::{Emission, Emitted, Guest, Payload};
use exports::ffrwd::gpu::ports::Guest as PortsGuest;
use ffrwd::av::node_tick::Tick;
use ffrwd::av::node_types::{
    Accepts, Binding, BoundStream, Clock, InputPort, LikeInput, Message, NodeShape, OutputFormat,
    OutputPort, Pairing, PortKind, RowsUse,
};
use ffrwd::av::types::{Meta, RawFrame, Wants};
use ffrwd::gpu::frames::emit;
use ffrwd::gpu::nn::{
    frame_tensor, output_texture, ChannelOrder, Fit, FrameTensorOptions, OutputTextureOptions,
    ResizeFilter, TensorLayout,
};
use ffrwd::gpu::types::{FrameMemory, PortMemory};
use wasi::nn::graph::{load_by_name, Graph};
use wasi::nn::inference::GraphExecutionContext;
use wasi::nn::tensor::{Tensor, TensorType};
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
    decode, floats_of, frame_box, index_map, outputs, parse_params, suppress, to_frame, to_row,
    Detection, Letterbox, Params, PixFmt, COEFFICIENTS, INPUT_INDEX, INPUT_NAME, MAX_DETECTIONS,
    MODEL, PAD, ROWS_SCHEMA, SIDE,
};

const SCHEMA: &str = r#"{"type":"object","properties":{"confidence":{"type":"number","default":0.25},"iou":{"type":"number","default":0.45},"memory":{"type":"string","enum":["host","gpu"],"default":"host"}},"additionalProperties":false}"#;

/// Floats per detection in the buffer the pass reads: its box on the frame,
/// left, top, right and bottom, then its coefficients.
const STRIDE: usize = 4 + COEFFICIENTS;

/// The pass that paints the index map. Each frame pixel takes the id of the
/// first detection, in descending score, whose box holds it and whose mask
/// is positive there; the mask is sampled as the host path's
/// ([`index_map`]) samples it, in its order of operations.
const PASS: &str = r#"
struct Params {
    frame_w: u32,
    frame_h: u32,
    proto_w: u32,
    proto_h: u32,
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
    scale: f32,
    offset_x: f32,
    offset_y: f32,
    side: f32,
}

@group(0) @binding(0) var protos: texture_2d<f32>;
@group(0) @binding(1) var<storage, read> found: array<f32>;
@group(0) @binding(2) var<uniform> p: Params;
@group(0) @binding(3) var map: texture_storage_2d<rgba8unorm, write>;

const STRIDE: u32 = 36u;
const COEFFICIENTS: u32 = 32u;

// Plane k's value at (x, y): the planes are stacked down the texture.
fn proto(k: u32, x: u32, y: u32) -> f32 {
    return textureLoad(protos, vec2<i32>(i32(x), i32(k * p.proto_h + y)), 0).r;
}

// The prototypes at (x, y) weighted by detection `base`'s coefficients.
fn weighed(base: u32, x: u32, y: u32) -> f32 {
    var sum = 0.0;
    for (var k = 0u; k < COEFFICIENTS; k = k + 1u) {
        sum = sum + found[base + 4u + k] * proto(k, x, y);
    }
    return sum;
}

@compute @workgroup_size(8, 8)
fn paint(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= p.frame_w || id.y >= p.frame_h) {
        return;
    }
    let fx = f32(id.x);
    let fy = f32(id.y);
    let per_x = f32(p.proto_w) / p.side;
    let per_y = f32(p.proto_h) / p.side;
    var owner = 0u;
    for (var i = 0u; i < p.count; i = i + 1u) {
        let base = i * STRIDE;
        if (fx < found[base] || fx >= found[base + 2u] || fy < found[base + 1u] || fy >= found[base + 3u]) {
            continue;
        }
        let gx = clamp((p.offset_x + fx * p.scale + 0.5) * per_x - 0.5, 0.0, f32(p.proto_w - 1u));
        let gy = clamp((p.offset_y + fy * p.scale + 0.5) * per_y - 0.5, 0.0, f32(p.proto_h - 1u));
        let x0 = u32(floor(gx));
        let y0 = u32(floor(gy));
        let x1 = min(x0 + 1u, p.proto_w - 1u);
        let y1 = min(y0 + 1u, p.proto_h - 1u);
        let tx = gx - floor(gx);
        let ty = gy - floor(gy);
        let a = weighed(base, x0, y0);
        let b = weighed(base, x1, y0);
        let c = weighed(base, x0, y1);
        let d = weighed(base, x1, y1);
        let top = a + (b - a) * tx;
        let bottom = c + (d - c) * tx;
        if (top + (bottom - top) * ty > 0.0) {
            owner = i + 1u;
            break;
        }
    }
    let v = f32(owner) / 255.0;
    textureStore(map, vec2<i32>(i32(id.x), i32(id.y)), vec4<f32>(v, v, v, 1.0));
}
"#;

/// The params, `memory` beside the thresholds both builds take.
struct Settings {
    params: Params,
    on_gpu: bool,
}

fn parse(text: &str) -> Result<Settings, String> {
    let mut fields: serde_json::Map<String, serde_json::Value> = if text.trim().is_empty() {
        serde_json::Map::new()
    } else {
        serde_json::from_str(text).map_err(|e| format!("segment cannot read its params: {e}"))?
    };
    let on_gpu = match fields.remove("memory") {
        None => false,
        Some(serde_json::Value::String(memory)) if memory == "host" => false,
        Some(serde_json::Value::String(memory)) if memory == "gpu" => true,
        Some(other) => return Err(format!("segment: memory is host or gpu, not {other}")),
    };
    let params = parse_params(&serde_json::Value::Object(fields).to_string())?;
    Ok(Settings { params, on_gpu })
}

/// The device's half: the prototypes' texture, the pass and what it binds.
struct Device {
    device: GpuDevice,
    protos: GpuTexture,
    protos_view: GpuTextureView,
    layout: GpuBindGroupLayout,
    paint: GpuComputePipeline,
    found: GpuBuffer,
    params: GpuBuffer,
    /// The prototype grid's width and height.
    proto: (usize, usize),
}

/// What `init` settled, plus the graph it loaded.
struct Opened {
    v: u32,
    width: usize,
    height: usize,
    pix_fmt: PixFmt,
    letterbox: Letterbox,
    params: Params,
    options: FrameTensorOptions,
    /// What the graph calls its input, settled by the first call that works.
    input_name: Cell<&'static str>,
    /// Whether the query reads the map, and the rows.
    paints: bool,
    tells: bool,
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
    format!("segment: {what}: {code} ({})", error.data())
}

/// The square's letterbox spec, as the host makes the tensor: 0 to 1, on
/// the model's grey.
fn tensor_options() -> FrameTensorOptions {
    FrameTensorOptions {
        model: MODEL.to_string(),
        width: SIDE as u32,
        height: SIDE as u32,
        layout: TensorLayout::Nchw,
        order: ChannelOrder::Rgb,
        fit: Fit::Letterbox,
        pad: PAD,
        scale: 1.0,
        mean: (0.0, 0.0, 0.0),
        std: (1.0, 1.0, 1.0),
        filter: ResizeFilter::Bilinear,
        antialias: false,
    }
}

/// One run of the graph, however it names its input: a tensor is made
/// again for the retry, since `compute` takes it.
fn compute(
    context: &GraphExecutionContext,
    input_name: &Cell<&'static str>,
    tensor: impl Fn() -> Result<Tensor, String>,
) -> Result<Vec<(String, Tensor)>, String> {
    let name = input_name.get();
    match context.compute(vec![(name.to_string(), tensor()?)]) {
        Ok(returned) => Ok(returned),
        // An export whose input is not called what ultralytics calls it. The
        // host takes a position where it takes a name, so the retry names none,
        // and the name that worked is kept for every run after this one.
        Err(_) if name == INPUT_NAME => {
            input_name.set(INPUT_INDEX);
            context
                .compute(vec![(INPUT_INDEX.to_string(), tensor()?)])
                .map_err(|e| failed("compute", &e))
        }
        Err(e) => Err(failed("compute", &e)),
    }
}

/// The two outputs' names and shapes, learned from one run over a grey
/// square: which is the prototypes is known only from what comes back.
fn learn_outputs(
    context: &GraphExecutionContext,
    input_name: &Cell<&'static str>,
) -> Result<(String, Vec<u32>), String> {
    let grey = vec![PAD; 3 * SIDE * SIDE];
    let bytes: Vec<u8> = grey.iter().flat_map(|v| v.to_le_bytes()).collect();
    let dimensions = [1, 3, SIDE as u32, SIDE as u32];
    let returned = compute(context, input_name, || {
        Ok(Tensor::new(&dimensions, TensorType::Fp32, &bytes))
    })?;
    let shapes: Vec<Vec<u32>> = returned.iter().map(|(_, t)| t.dimensions()).collect();
    let (_, prototypes) = outputs(&shapes)?;
    Ok((returned[prototypes].0.clone(), shapes[prototypes].clone()))
}

/// Opens the device's half: the prototypes bound to a texture, the pass,
/// and its buffers.
fn open_device(
    context: &GraphExecutionContext,
    input_name: &Cell<&'static str>,
) -> Result<Device, String> {
    let device = get_gpu()
        .request_adapter(None)
        .ok_or("segment: no GPU adapter; memory=host runs without one")?
        .request_device(None)
        .map_err(|_| "segment: the adapter gave no device".to_string())?;
    let (name, shape) = learn_outputs(context, input_name)?;
    let [_, planes, proto_h, proto_w] = shape[..] else {
        return Err(format!("segment: the prototypes are {shape:?}"));
    };
    let protos = output_texture(
        context,
        &OutputTextureOptions {
            output: name,
            width: proto_w,
            height: planes * proto_h,
        },
    )
    .map_err(|e| format!("segment: {e}"))?;
    let protos_view = protos.create_view(None);
    let compute_stage = GpuShaderStage::compute();
    let entry = |binding| GpuBindGroupLayoutEntry {
        binding,
        visibility: compute_stage,
        buffer: None,
        sampler: None,
        texture: None,
        storage_texture: None,
    };
    let buffer_entry = |binding, type_| GpuBindGroupLayoutEntry {
        buffer: Some(GpuBufferBindingLayout {
            type_: Some(type_),
            has_dynamic_offset: None,
            min_binding_size: None,
        }),
        ..entry(binding)
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
            buffer_entry(1, GpuBufferBindingType::ReadOnlyStorage),
            buffer_entry(2, GpuBufferBindingType::Uniform),
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
        code: PASS.to_string(),
        compilation_hints: None,
        label: Some("segment pass".to_string()),
    });
    let paint = device.create_compute_pipeline(GpuComputePipelineDescriptor {
        compute: GpuProgrammableStage {
            module: &module,
            entry_point: Some("paint".to_string()),
            constants: None,
        },
        layout: GpuLayoutMode::Specific(&pipeline_layout),
        label: None,
    });
    let found = device.create_buffer(&GpuBufferDescriptor {
        size: (MAX_DETECTIONS * STRIDE * 4) as u64,
        usage: GpuBufferUsage::storage() | GpuBufferUsage::copy_dst(),
        mapped_at_creation: None,
        label: None,
    });
    let params = device.create_buffer(&GpuBufferDescriptor {
        size: 48,
        usage: GpuBufferUsage::uniform() | GpuBufferUsage::copy_dst(),
        mapped_at_creation: None,
        label: None,
    });
    Ok(Device {
        device,
        protos,
        protos_view,
        layout,
        paint,
        found,
        params,
        proto: (proto_w as usize, proto_h as usize),
    })
}

/// The detections a run found, thinned, in descending score.
fn detections(opened: &Opened, predictions: &Tensor) -> Result<Vec<Detection>, String> {
    let shape = predictions.dimensions();
    let [_, channels, anchors] = shape[..] else {
        return Err(format!("segment: the predictions are {shape:?}"));
    };
    let found = decode(
        &floats_of(&predictions.data()),
        channels as usize,
        anchors as usize,
        opened.params.confidence as f32,
    );
    let kept = suppress(found, opened.params.iou as f32);
    if kept.len() > MAX_DETECTIONS {
        return Err(format!(
            "segment: {} detections in one frame, and an index map names at most \
             {MAX_DETECTIONS}; raise confidence",
            kept.len()
        ));
    }
    Ok(kept)
}

/// One frame through the graph: its detections, and the map, as bytes on
/// the host path or as a texture on the device.
fn run(opened: &Opened, index: u32) -> Result<(Vec<Detection>, Option<Map>), String> {
    let returned = compute(&opened.context, &opened.input_name, || {
        frame_tensor(opened.v, index, &opened.options).map_err(|e| format!("segment: {e}"))
    })?;
    let shapes: Vec<Vec<u32>> = returned.iter().map(|(_, t)| t.dimensions()).collect();
    let (predictions, prototypes) = outputs(&shapes)?;
    let kept = detections(opened, &returned[predictions].1)?;
    if !opened.paints {
        return Ok((kept, None));
    }
    let map = match &opened.gpu {
        Some(gpu) => Map::Texture(paint(opened, gpu, &kept)?),
        None => {
            let proto = &shapes[prototypes];
            let map = index_map(
                &kept,
                &floats_of(&returned[prototypes].1.data()),
                (proto[3] as usize, proto[2] as usize),
                (opened.width, opened.height),
                opened.letterbox,
            );
            let len = match opened.pix_fmt {
                PixFmt::Yuv420p => {
                    opened.width * opened.height
                        + 2 * opened.width.div_ceil(2) * opened.height.div_ceil(2)
                }
                PixFmt::Rgba => opened.width * opened.height * 4,
            };
            Map::Bytes(to_frame(
                &map,
                opened.pix_fmt,
                opened.width,
                opened.height,
                len,
            ))
        }
    };
    Ok((kept, Some(map)))
}

/// The map a frame leaves as.
enum Map {
    Bytes(Vec<u8>),
    Texture(GpuTexture),
}

/// The map painted on the device from the prototypes' texture: the frame
/// that leaves.
fn paint(opened: &Opened, gpu: &Device, kept: &[Detection]) -> Result<GpuTexture, String> {
    let (width, height) = (opened.width as u32, opened.height as u32);
    let mut found = Vec::with_capacity(kept.len().max(1) * STRIDE * 4);
    for detection in kept {
        let (x0, y0, x1, y1) = frame_box(detection, opened.letterbox, opened.width, opened.height);
        for value in [x0, y0, x1, y1] {
            found.extend_from_slice(&(value as f32).to_le_bytes());
        }
        for coefficient in &detection.coefficients {
            found.extend_from_slice(&coefficient.to_le_bytes());
        }
    }
    let queue = gpu.device.queue();
    if !found.is_empty() {
        queue
            .write_buffer_with_copy(&gpu.found, 0, &found, None, None)
            .map_err(|_| "segment: writing the detections".to_string())?;
    }
    let mut words = Vec::with_capacity(48);
    for word in [
        width,
        height,
        gpu.proto.0 as u32,
        gpu.proto.1 as u32,
        kept.len() as u32,
        0,
        0,
        0,
    ] {
        words.extend_from_slice(&word.to_le_bytes());
    }
    for value in [
        opened.letterbox.scale,
        opened.letterbox.offset_x as f32,
        opened.letterbox.offset_y as f32,
        SIDE as f32,
    ] {
        words.extend_from_slice(&value.to_le_bytes());
    }
    queue
        .write_buffer_with_copy(&gpu.params, 0, &words, None, None)
        .map_err(|_| "segment: writing the pass's parameters".to_string())?;
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
    let buffer = |buffer| {
        GpuBindingResource::GpuBufferBinding(GpuBufferBinding {
            buffer,
            offset: None,
            size: None,
        })
    };
    let group = gpu.device.create_bind_group(&GpuBindGroupDescriptor {
        layout: &gpu.layout,
        entries: vec![
            GpuBindGroupEntry {
                binding: 0,
                resource: GpuBindingResource::GpuTextureView(&gpu.protos_view),
            },
            GpuBindGroupEntry {
                binding: 1,
                resource: buffer(&gpu.found),
            },
            GpuBindGroupEntry {
                binding: 2,
                resource: buffer(&gpu.params),
            },
            GpuBindGroupEntry {
                binding: 3,
                resource: GpuBindingResource::GpuTextureView(&frame_view),
            },
        ],
        label: None,
    });
    let encoder = gpu.device.create_command_encoder(None);
    let pass = encoder.begin_compute_pass(None);
    pass.set_pipeline(&gpu.paint);
    pass.set_bind_group(0, Some(&group), None, None, None)
        .map_err(|_| "segment: binding the pass's resources".to_string())?;
    pass.dispatch_workgroups(width.div_ceil(8), Some(height.div_ceil(8)), None);
    pass.end();
    queue.submit(&[&encoder.finish(None)]);
    let _ = &gpu.protos;
    Ok(frame)
}

struct Segment;

impl Guest for Segment {
    fn describe() -> Meta {
        Meta {
            name: "segment".to_string(),
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
        let on_gpu = parse(&params)?.on_gpu;
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
        let map = OutputPort {
            name: "map".to_string(),
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
        let objects = OutputPort {
            name: "objects".to_string(),
            kind: PortKind::Data,
            format: Some(OutputFormat::Data("json".to_string())),
            time_base: None,
            latency: 0.0,
            schema: Some(ROWS_SCHEMA.to_string()),
            row: None,
        };
        Ok(NodeShape {
            inputs: vec![v],
            outputs: vec![map, objects],
            clock: Clock::Input("v".to_string()),
            pure: true,
            one_to_one: true,
            bounded: true,
            relation: Vec::new(),
        })
    }

    fn init(bound: Vec<BoundStream>, latched: Vec<String>, params: String) -> Result<(), String> {
        let settings = parse(&params)?;
        let v = bound
            .iter()
            .find(|b| b.port == "v")
            .ok_or("segment: v is bound")?;
        let Some(OutputFormat::Video(video)) = &v.format else {
            return Err("segment reads frames, and this stream is not video".to_string());
        };
        let pix_fmt = match video.pix_fmt.as_str() {
            "yuv420p" => PixFmt::Yuv420p,
            "rgba" => PixFmt::Rgba,
            _ if settings.on_gpu => PixFmt::Rgba,
            other => return Err(format!("segment does not accept pixel format {other}")),
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
        let input_name = Cell::new(INPUT_NAME);
        let gpu = if settings.on_gpu {
            Some(open_device(&context, &input_name)?)
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
                params: settings.params,
                options: tensor_options(),
                input_name,
                paints: latched.iter().any(|port| port == "map"),
                tells: latched.iter().any(|port| port == "objects"),
                context,
                _graph: graph,
                gpu,
            });
        });
        Ok(())
    }

    fn set_params(params: String) -> Result<(), String> {
        let settings = parse(&params)?;
        OPENED.with(|o| {
            if let Some(opened) = o.borrow_mut().as_mut() {
                opened.params = settings.params;
            }
        });
        Ok(())
    }

    fn process(tick: &Tick) -> Result<Emitted, String> {
        OPENED.with(|opened| {
            let borrowed = opened.borrow();
            let opened = borrowed.as_ref().ok_or("segment: process before init")?;
            let mut items = Vec::new();
            for frame in tick.frames(opened.v) {
                let (kept, map) = run(opened, frame.index)?;
                match map {
                    Some(Map::Texture(texture)) => {
                        emit("map", frame.pts, frame.duration, texture)?;
                    }
                    Some(Map::Bytes(data)) => items.push(Emission {
                        port: "map".to_string(),
                        payload: Payload::Frame(RawFrame {
                            pts: frame.pts,
                            duration: frame.duration,
                            data,
                        }),
                    }),
                    None => {}
                }
                if !opened.tells {
                    continue;
                }
                for (index, detection) in kept.iter().enumerate() {
                    let row = to_row(
                        detection,
                        (index + 1) as u32,
                        opened.letterbox,
                        opened.width,
                        opened.height,
                    );
                    items.push(Emission {
                        port: "objects".to_string(),
                        payload: Payload::Message(Message {
                            pts: frame.pts,
                            data: row.into_bytes(),
                        }),
                    });
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

impl PortsGuest for Segment {
    fn memory(params: String) -> Vec<PortMemory> {
        match parse(&params) {
            Ok(settings) if settings.on_gpu => ["v", "map"]
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

export!(Segment);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_is_read_beside_the_thresholds() {
        let settings = parse(r#"{"memory":"gpu","confidence":0.5}"#).expect("both");
        assert!(settings.on_gpu);
        assert_eq!(settings.params.confidence, 0.5);
        assert!(!parse("").expect("the defaults").on_gpu, "host by default");
        assert!(parse(r#"{"memory":"disk"}"#).is_err());
        assert!(
            parse(r#"{"radius":3}"#).is_err(),
            "and a param neither build has is refused"
        );
    }
}
