//! Runs a model on every frame of its one video input and writes what the
//! model returned as rows on its one output, `rows`: it emits no frames.
//!
//! The module never reads a frame's bytes. It asks the host for the frame
//! as the model's input tensor (`ffrwd:gpu/nn.frame-tensor`), at the size,
//! layout, resampling and normalisation its params give, and passes that
//! tensor to `compute` like any other. With `crops`, a list of `[x, y,
//! width, height]` (as JSON, or text holding it), it asks for those
//! rectangles of the frame instead, one picture each along the tensor's
//! first dimension (`ffrwd:gpu/nn.crops-tensor`). Where the frame is held
//! is `memory`: on the GPU, the host makes the tensor on the device, and
//! binds it to a CUDA session in place where it can; in host memory, the
//! host makes it on the CPU. Either way the rows are the same, which is what the tests check.
//!
//! Each row is one frame: its pts, each output by name (its values when it
//! has 16 or fewer, else their count, mean, least and greatest), and the
//! milliseconds the tensor and the compute took inside the module.
//!
//! `texture` names an output to bind to a texture
//! (`ffrwd:gpu/nn.output-texture`) of `texture_width` by `texture_height`;
//! that output's values are then read from the texture, a copy on the
//! device and one read back, and the row adds the milliseconds that took as
//! `output_ms`.
//!
//! `model/channel-means.onnx` (NCHW) and `model/channel-means-nhwc.onnx`
//! (NHWC) are the graphs the tests run: each channel's mean and largest
//! value, six numbers a test can work out from the frame on its own.
//! `model/channel-means-batch.onnx` takes a batch, crops among them, and
//! gives the six numbers of each picture.

wit_bindgen::generate!({
    path: ["../../wit", "../../wit-gpu", "wit"],
    // Fully qualified: several packages are in scope, and each has worlds.
    world: "ffrwd:nn-frames/nn-frames",
    generate_all,
});

use std::cell::RefCell;
use std::time::Instant;

use exports::ffrwd::av::node::{Emission, Emitted, Guest, Payload};
use exports::ffrwd::gpu::ports::Guest as PortsGuest;
use ffrwd::av::node_tick::Tick;
use ffrwd::av::node_types::{
    Accepts, Binding, BoundStream, Clock, InputPort, Message, NodeShape, OutputFormat, OutputPort,
    Pairing, PortKind, RowsUse,
};
use ffrwd::av::types::{Meta, Wants};
use ffrwd::gpu::nn::{
    crops_tensor, frame_tensor, output_texture, ChannelOrder, Fit, FrameTensorOptions,
    OutputTextureOptions, Rect, ResizeFilter, TensorLayout,
};
use ffrwd::gpu::types::{FrameMemory, PortMemory};
use serde::Deserialize;
use serde_json::{json, Map, Value};
use wasi::nn::graph::load_by_name;
use wasi::nn::inference::GraphExecutionContext;
use wasi::webgpu::webgpu::{
    get_gpu, GpuBufferDescriptor, GpuBufferUsage, GpuDevice, GpuExtent3D, GpuMapMode,
    GpuTexelCopyBufferInfo, GpuTexelCopyTextureInfo, GpuTexture,
};

const SCHEMA: &str = r#"{"type":"object","properties":{"memory":{"type":"string","enum":["gpu","host"]},"model":{"type":"string"},"input":{"type":"string"},"width":{"type":"integer","minimum":1},"height":{"type":"integer","minimum":1},"layout":{"type":"string","enum":["nchw","nhwc"]},"order":{"type":"string","enum":["rgb","bgr"]},"fit":{"type":"string","enum":["stretch","letterbox"]},"pad":{"type":"number"},"scale":{"type":"number"},"mean":{"type":"array","items":{"type":"number"},"minItems":3,"maxItems":3},"std":{"type":"array","items":{"type":"number"},"minItems":3,"maxItems":3},"texture":{"type":"string"},"texture_width":{"type":"integer","minimum":1},"texture_height":{"type":"integer","minimum":1},"filter":{"type":"string","enum":["bilinear","bicubic"]},"antialias":{"type":"boolean"},"crops":{"type":["string","array"]}},"required":["model"],"additionalProperties":false}"#;

const ROWS_SCHEMA: &str = r#"{"type":"object","properties":{"pts":{"type":"integer"},"outputs":{"type":"object"},"tensor_ms":{"type":"number"},"compute_ms":{"type":"number"},"output_ms":{"type":"number"}}}"#;

/// An output with this many values or fewer is written whole.
const WHOLE: usize = 16;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Params {
    #[serde(default = "gpu")]
    memory: String,
    model: String,
    /// The graph's name for its input.
    #[serde(default = "image")]
    input: String,
    #[serde(default = "side")]
    width: u32,
    #[serde(default = "side")]
    height: u32,
    #[serde(default = "nchw")]
    layout: String,
    #[serde(default = "rgb")]
    order: String,
    #[serde(default = "stretch")]
    fit: String,
    #[serde(default)]
    pad: f32,
    #[serde(default = "one")]
    scale: f32,
    #[serde(default)]
    mean: [f32; 3],
    #[serde(default = "ones")]
    std: [f32; 3],
    /// An output bound to a texture, and the texture's size.
    #[serde(default)]
    texture: Option<String>,
    #[serde(default)]
    texture_width: Option<u32>,
    #[serde(default)]
    texture_height: Option<u32>,
    #[serde(default = "bilinear")]
    filter: String,
    #[serde(default)]
    antialias: bool,
    /// Rectangles of the frame to make the tensor of, each `[x, y, width,
    /// height]`: a JSON list, or text holding one.
    #[serde(default)]
    crops: Option<Value>,
}

fn gpu() -> String {
    "gpu".to_string()
}
fn image() -> String {
    "image".to_string()
}
fn side() -> u32 {
    64
}
fn nchw() -> String {
    "nchw".to_string()
}
fn rgb() -> String {
    "rgb".to_string()
}
fn stretch() -> String {
    "stretch".to_string()
}
fn bilinear() -> String {
    "bilinear".to_string()
}
fn one() -> f32 {
    1.0
}
fn ones() -> [f32; 3] {
    [1.0; 3]
}

fn parse(text: &str) -> Result<Params, String> {
    let params: Params = serde_json::from_str(text).map_err(|e| format!("params: {e}"))?;
    if params.memory != "gpu" && params.memory != "host" {
        return Err(format!("memory is gpu or host, not '{}'", params.memory));
    }
    if params.texture.is_some() != params.texture_width.is_some()
        || params.texture.is_some() != params.texture_height.is_some()
    {
        return Err("texture takes texture_width and texture_height".to_string());
    }
    params.crops()?;
    Ok(params)
}

impl Params {
    fn options(&self) -> Result<FrameTensorOptions, String> {
        Ok(FrameTensorOptions {
            model: self.model.clone(),
            width: self.width,
            height: self.height,
            layout: match self.layout.as_str() {
                "nchw" => TensorLayout::Nchw,
                "nhwc" => TensorLayout::Nhwc,
                other => return Err(format!("layout is nchw or nhwc, not '{other}'")),
            },
            order: match self.order.as_str() {
                "rgb" => ChannelOrder::Rgb,
                "bgr" => ChannelOrder::Bgr,
                other => return Err(format!("order is rgb or bgr, not '{other}'")),
            },
            fit: match self.fit.as_str() {
                "stretch" => Fit::Stretch,
                "letterbox" => Fit::Letterbox,
                other => return Err(format!("fit is stretch or letterbox, not '{other}'")),
            },
            pad: self.pad,
            scale: self.scale,
            mean: (self.mean[0], self.mean[1], self.mean[2]),
            std: (self.std[0], self.std[1], self.std[2]),
            filter: match self.filter.as_str() {
                "bilinear" => ResizeFilter::Bilinear,
                "bicubic" => ResizeFilter::Bicubic,
                other => return Err(format!("filter is bilinear or bicubic, not '{other}'")),
            },
            antialias: self.antialias,
        })
    }

    fn crops(&self) -> Result<Option<Vec<Rect>>, String> {
        let crops = match &self.crops {
            None => return Ok(None),
            Some(Value::String(text)) => {
                serde_json::from_str(text).map_err(|e| format!("crops: {e}"))?
            }
            Some(list) => list.clone(),
        };
        let quads: Vec<[u32; 4]> = serde_json::from_value(crops)
            .map_err(|e| format!("crops is a list of [x, y, width, height]: {e}"))?;
        Ok(Some(
            quads
                .into_iter()
                .map(|[x, y, width, height]| Rect {
                    x,
                    y,
                    width,
                    height,
                })
                .collect(),
        ))
    }
}

struct State {
    v: u32,
    input: String,
    options: FrameTensorOptions,
    /// The rectangles to make the tensor of, or the whole frame.
    crops: Option<Vec<Rect>>,
    context: GraphExecutionContext,
    /// The output bound to a texture, the texture, and the device to read
    /// it back with.
    texture: Option<(String, GpuTexture, GpuDevice)>,
}

/// `texture`'s `width` x `height` fp32 texels, read back: a copy into a
/// buffer, then a map.
fn read_texture(
    device: &GpuDevice,
    texture: &GpuTexture,
    width: u32,
    height: u32,
) -> Result<Vec<u8>, String> {
    let row = width * 4;
    let padded = row.div_ceil(256) * 256;
    let buffer = device.create_buffer(&GpuBufferDescriptor {
        size: u64::from(padded * height),
        usage: GpuBufferUsage::copy_dst() | GpuBufferUsage::map_read(),
        mapped_at_creation: None,
        label: None,
    });
    let encoder = device.create_command_encoder(None);
    encoder.copy_texture_to_buffer(
        &GpuTexelCopyTextureInfo {
            texture,
            mip_level: None,
            origin: None,
            aspect: None,
        },
        &GpuTexelCopyBufferInfo {
            buffer: &buffer,
            offset: None,
            bytes_per_row: Some(padded),
            rows_per_image: Some(height),
        },
        GpuExtent3D {
            width,
            height: Some(height),
            depth_or_array_layers: None,
        },
    );
    device.queue().submit(&[&encoder.finish(None)]);
    buffer
        .map_async(GpuMapMode::read(), None, None)
        .map_err(|_| "mapping the texture's readback".to_string())?;
    let mapped = buffer
        .get_mapped_range_get_with_copy(None, None)
        .map_err(|_| "reading the texture's readback".to_string())?;
    let mut out = Vec::with_capacity((row * height) as usize);
    for y in 0..height as usize {
        let at = y * padded as usize;
        out.extend_from_slice(&mapped[at..at + row as usize]);
    }
    buffer
        .unmap()
        .map_err(|_| "unmapping the texture's readback".to_string())?;
    Ok(out)
}

thread_local! {
    static STATE: RefCell<Option<State>> = const { RefCell::new(None) };
}

/// One output as a row's value: its values when there are few, else a
/// summary.
fn written(data: &[u8]) -> Value {
    let (whole, _) = data.as_chunks::<4>();
    let values: Vec<f32> = whole.iter().copied().map(f32::from_le_bytes).collect();
    if values.len() <= WHOLE {
        return json!(values);
    }
    let sum: f64 = values.iter().map(|&v| f64::from(v)).sum();
    let least = values.iter().copied().fold(f32::INFINITY, f32::min);
    let greatest = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    json!({
        "count": values.len(),
        "mean": sum / values.len() as f64,
        "min": least,
        "max": greatest,
    })
}

fn ms(since: Instant) -> f64 {
    since.elapsed().as_secs_f64() * 1000.0
}

struct Node;

impl Guest for Node {
    fn describe() -> Meta {
        Meta {
            name: "nn_frames".to_string(),
            version: "0.1.0".to_string(),
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
        parse(&params)?;
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
                pixel_formats: vec![
                    "yuv420p".to_string(),
                    "rgba".to_string(),
                    "yuv422p".to_string(),
                    "yuv444p".to_string(),
                ],
                sample_formats: Vec::new(),
                sample_rates: Vec::new(),
                channel_counts: Vec::new(),
                codecs: Vec::new(),
                wants: Wants::All,
                like: None,
            },
            schema: None,
        };
        let rows = OutputPort {
            name: "rows".to_string(),
            kind: PortKind::Data,
            format: Some(OutputFormat::Data("json".to_string())),
            time_base: None,
            latency: 0.0,
            schema: Some(ROWS_SCHEMA.to_string()),
            row: None,
        };
        Ok(NodeShape {
            inputs: vec![v],
            outputs: vec![rows],
            clock: Clock::Input("v".to_string()),
            pure: true,
            one_to_one: false,
            bounded: true,
            relation: Vec::new(),
        })
    }

    fn init(bound: Vec<BoundStream>, _latched: Vec<String>, params: String) -> Result<(), String> {
        let params = parse(&params)?;
        let v = bound.iter().find(|b| b.port == "v").ok_or("v is bound")?;
        if !matches!(&v.format, Some(OutputFormat::Video(_))) {
            return Err("v is video".to_string());
        }
        let graph = load_by_name(&params.model)
            .map_err(|e| format!("load-by-name({:?}): {}", params.model, e.data()))?;
        let context = graph
            .init_execution_context()
            .map_err(|e| format!("init-execution-context: {}", e.data()))?;
        let options = params.options()?;
        let crops = params.crops()?;
        let texture = match (&params.texture, params.texture_width, params.texture_height) {
            (Some(name), Some(width), Some(height)) => {
                let texture = output_texture(
                    &context,
                    &OutputTextureOptions {
                        output: name.clone(),
                        width,
                        height,
                    },
                )?;
                let device = get_gpu()
                    .request_adapter(None)
                    .ok_or("no GPU adapter to read the texture with")?
                    .request_device(None)
                    .map_err(|_| "the adapter gave no device".to_string())?;
                Some((name.clone(), texture, device))
            }
            _ => None,
        };
        STATE.with(|s| {
            *s.borrow_mut() = Some(State {
                v: v.id,
                input: params.input.clone(),
                options,
                crops,
                context,
                texture,
            })
        });
        Ok(())
    }

    fn set_params(_params: String) -> Result<(), String> {
        Ok(())
    }

    fn process(tick: &Tick) -> Result<Emitted, String> {
        STATE.with(|s| {
            let state = s.borrow();
            let state = state.as_ref().ok_or("process before init")?;
            let mut rows = Vec::new();
            for frame in tick.frames(state.v) {
                let started = Instant::now();
                let tensor = match &state.crops {
                    Some(crops) => crops_tensor(state.v, frame.index, crops, &state.options)?,
                    None => frame_tensor(state.v, frame.index, &state.options)?,
                };
                let tensor_ms = ms(started);
                let started = Instant::now();
                let outputs = state
                    .context
                    .compute(vec![(state.input.clone(), tensor)])
                    .map_err(|e| format!("compute: {}", e.data()))?;
                let compute_ms = ms(started);
                let mut written_outputs = Map::new();
                let mut output_ms = 0.0;
                for (name, out) in &outputs {
                    match &state.texture {
                        Some((bound, texture, device)) if bound == name => {
                            let started = Instant::now();
                            let bytes =
                                read_texture(device, texture, texture.width(), texture.height())?;
                            output_ms = ms(started);
                            written_outputs.insert(name.clone(), written(&bytes));
                        }
                        _ => {
                            written_outputs.insert(name.clone(), written(&out.data()));
                        }
                    }
                }
                let row = json!({
                    "pts": frame.pts,
                    "outputs": written_outputs,
                    "tensor_ms": tensor_ms,
                    "compute_ms": compute_ms,
                    "output_ms": output_ms,
                });
                rows.push(Emission {
                    port: "rows".to_string(),
                    payload: Payload::Message(Message {
                        pts: frame.pts,
                        data: row.to_string().into_bytes(),
                    }),
                });
            }
            Ok(Emitted {
                items: rows,
                rows: Vec::new(),
                finished: false,
            })
        })
    }
}

impl PortsGuest for Node {
    fn memory(params: String) -> Vec<PortMemory> {
        match parse(&params) {
            Ok(params) if params.memory == "gpu" => vec![PortMemory {
                port: "v".to_string(),
                memory: FrameMemory::Gpu,
            }],
            _ => Vec::new(),
        }
    }
}

export!(Node);
