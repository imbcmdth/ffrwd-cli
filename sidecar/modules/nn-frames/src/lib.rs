//! Runs a model on every frame of its one video input and writes what the
//! model returned as rows: a sink, so it emits no frames.
//!
//! The module never reads a frame's bytes. It asks the host for the frame
//! as the model's input tensor (`ffrwd:gpu/nn.frame-tensor`), at the size,
//! layout and normalisation its params give, and passes that tensor to
//! `compute` like any other. Where the frame is held is `memory`: on the
//! GPU, the host makes the tensor on the device, and binds it to a CUDA
//! session in place where it can; in host memory, the host makes it on the
//! CPU. Either way the rows are the same, which is what the tests check.
//!
//! Each row is one frame: its pts, each output by name (its values when it
//! has 16 or fewer, else their count, mean, least and greatest), and the
//! milliseconds the tensor and the compute took inside the module.
//!
//! `model/channel-means.onnx` (NCHW) and `model/channel-means-nhwc.onnx`
//! (NHWC) are the graphs the tests run: each channel's mean and largest
//! value, six numbers a test can work out from the frame on its own.

wit_bindgen::generate!({
    path: ["../../wit", "../../wit-gpu", "wit"],
    // Fully qualified: several packages are in scope, and each has worlds.
    world: "ffrwd:nn-frames/nn-frames",
    generate_all,
});

use std::cell::RefCell;
use std::time::Instant;

use exports::ffrwd::av::node::{Emitted, Guest};
use exports::ffrwd::gpu::ports::Guest as PortsGuest;
use ffrwd::av::node_tick::Tick;
use ffrwd::av::node_types::{
    Accepts, Binding, BoundStream, Clock, InputPort, NodeShape, OutputFormat, Pairing, PortKind,
    RowsUse,
};
use ffrwd::av::types::{Meta, Wants};
use ffrwd::gpu::nn::{frame_tensor, ChannelOrder, Fit, FrameTensorOptions, TensorLayout};
use ffrwd::gpu::types::{FrameMemory, PortMemory};
use serde::Deserialize;
use serde_json::{json, Map, Value};
use wasi::nn::graph::load_by_name;
use wasi::nn::inference::GraphExecutionContext;

const SCHEMA: &str = r#"{"type":"object","properties":{"memory":{"type":"string","enum":["gpu","host"]},"model":{"type":"string"},"input":{"type":"string"},"width":{"type":"integer","minimum":1},"height":{"type":"integer","minimum":1},"layout":{"type":"string","enum":["nchw","nhwc"]},"order":{"type":"string","enum":["rgb","bgr"]},"fit":{"type":"string","enum":["stretch","letterbox"]},"scale":{"type":"number"},"mean":{"type":"array","items":{"type":"number"},"minItems":3,"maxItems":3},"std":{"type":"array","items":{"type":"number"},"minItems":3,"maxItems":3}},"required":["model"],"additionalProperties":false}"#;

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
    #[serde(default = "one")]
    scale: f32,
    #[serde(default)]
    mean: [f32; 3],
    #[serde(default = "ones")]
    std: [f32; 3],
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
            scale: self.scale,
            mean: (self.mean[0], self.mean[1], self.mean[2]),
            std: (self.std[0], self.std[1], self.std[2]),
        })
    }
}

struct State {
    v: u32,
    input: String,
    options: FrameTensorOptions,
    context: GraphExecutionContext,
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
        Ok(NodeShape {
            inputs: vec![v],
            outputs: Vec::new(),
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
        STATE.with(|s| {
            *s.borrow_mut() = Some(State {
                v: v.id,
                input: params.input.clone(),
                options,
                context,
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
                let tensor = frame_tensor(state.v, frame.index, &state.options)?;
                let tensor_ms = ms(started);
                let started = Instant::now();
                let outputs = state
                    .context
                    .compute(vec![(state.input.clone(), tensor)])
                    .map_err(|e| format!("compute: {}", e.data()))?;
                let compute_ms = ms(started);
                let mut written_outputs = Map::new();
                for (name, out) in &outputs {
                    written_outputs.insert(name.clone(), written(&out.data()));
                }
                rows.push(
                    json!({
                        "pts": frame.pts,
                        "outputs": written_outputs,
                        "tensor_ms": tensor_ms,
                        "compute_ms": compute_ms,
                    })
                    .to_string(),
                );
            }
            Ok(Emitted {
                items: Vec::new(),
                rows,
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
