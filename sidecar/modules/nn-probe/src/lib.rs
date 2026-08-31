//! Inference through `wasi:nn`, small enough to assert on by hand.
//!
//! The graph is `model/tiny.onnx`: `y = a @ w + b` over fp32, with `a` of
//! shape [1, 4] and `y` of shape [1, 2]. The host registers it under a name
//! with `-nn`; this module names it and never opens a file.
//!
//! `sandbox` is the other half of the story. A module doing native inference
//! is still a sandboxed module: it has no preopened directory, so the model
//! file it just ran is unreachable to it, and so is everything else.

// `generate_all`: the world's interfaces come from two other packages -
// ffrwd:av and wasi:nn - and without it bindgen expects them to have been
// generated somewhere else.
wit_bindgen::generate!({
    path: ["../../worlds/0.10.0", "wit"],
    // Fully qualified: three packages are in scope, and each has worlds.
    world: "ffrwd:nn-probe/nn-probe",
    generate_all,
});

use exports::ffrwd::av::values::{FunctionMeta, Guest};
use serde::Deserialize;
use serde_json::json;
use wasi::nn::errors::ErrorCode;
use wasi::nn::graph::load_by_name;
use wasi::nn::tensor::{Tensor, TensorType};

/// Elements the graph's input takes.
const INPUT_LEN: usize = 4;

/// The graph's own names for its input and output tensors.
const INPUT_NAME: &str = "a";

const RUN_PARAMS: &str = r#"{"type":"object","properties":{"model":{"type":"string"},"input":{"type":"array","items":{"type":"number"},"minItems":4,"maxItems":4}},"required":["model","input"],"additionalProperties":false}"#;
const RUN_RESULT: &str = r#"{"type":"object","properties":{"dimensions":{"type":"array","items":{"type":"integer"}},"output":{"type":"array","items":{"type":"number"}}},"required":["dimensions","output"]}"#;
const SANDBOX_PARAMS: &str = r#"{"type":"object","properties":{},"additionalProperties":false}"#;
const SANDBOX_RESULT: &str = r#"{"type":"object","properties":{"reachable":{"type":"array","items":{"type":"string"}},"denied":{"type":"array","items":{"type":"string"}}},"required":["reachable","denied"]}"#;
const DETECT_PARAMS: &str = r#"{"type":"object","properties":{"model":{"type":"string"},"rgb":{"type":"string"},"runs":{"type":"integer","default":1}},"required":["model","rgb"],"additionalProperties":false}"#;
const DETECT_RESULT: &str = r#"{"type":"object","properties":{"detections":{"type":"array","items":{"type":"object"}},"first_ms":{"type":"number"},"mean_ms":{"type":"number"}},"required":["detections","first_ms","mean_ms"]}"#;

/// The spec's spelling of an error code, so a caller can match on it without
/// depending on how this module happens to format things.
fn code_name(code: ErrorCode) -> &'static str {
    match code {
        ErrorCode::InvalidArgument => "invalid-argument",
        ErrorCode::InvalidEncoding => "invalid-encoding",
        ErrorCode::Timeout => "timeout",
        ErrorCode::RuntimeError => "runtime-error",
        ErrorCode::UnsupportedOperation => "unsupported-operation",
        ErrorCode::TooLarge => "too-large",
        ErrorCode::NotFound => "not-found",
        ErrorCode::Security => "security",
        ErrorCode::Unknown => "unknown",
    }
}

fn failed(what: &str, error: &wasi::nn::errors::Error) -> String {
    format!("{what}: {} ({})", code_name(error.code()), error.data())
}

#[derive(Deserialize)]
struct RunArgs {
    model: String,
    input: Vec<f32>,
}

/// Names the guest tries to read, to show it cannot. The model path is the
/// one the host was given on the command line for the exec tier's own model,
/// so a breach would be visible rather than theoretical.
const UNREACHABLE: [&str; 4] = [
    "model/tiny.onnx",
    "/",
    ".",
    "E:/projects/ffrwd/sidecar/modules/nn-probe/model/tiny.onnx",
];

fn run(args: &str) -> Result<String, String> {
    let args: RunArgs = serde_json::from_str(args).map_err(|e| format!("invalid args: {e}"))?;
    if args.input.len() != INPUT_LEN {
        return Err(format!(
            "the graph takes {INPUT_LEN} numbers, got {}",
            args.input.len()
        ));
    }

    // The guest names the model. It never opens a file; it has no preopens.
    let graph = load_by_name(&args.model)
        .map_err(|e| failed(&format!("load-by-name({:?})", args.model), &e))?;
    let context = graph
        .init_execution_context()
        .map_err(|e| failed("init-execution-context", &e))?;

    let bytes: Vec<u8> = args.input.iter().flat_map(|v| v.to_le_bytes()).collect();
    let tensor = Tensor::new(&[1, INPUT_LEN as u32], TensorType::Fp32, &bytes);
    let outputs = context
        .compute(vec![(INPUT_NAME.to_string(), tensor)])
        .map_err(|e| failed("compute", &e))?;

    let (_, out) = outputs
        .into_iter()
        .next()
        .ok_or_else(|| "the graph returned no output tensor".to_string())?;
    let data = out.data();
    let (whole, _) = data.as_chunks::<4>();
    let values: Vec<f32> = whole.iter().copied().map(f32::from_le_bytes).collect();

    Ok(json!({ "dimensions": out.dimensions(), "output": values }).to_string())
}

/// The side a detection graph's square input takes.
const SIDE: usize = 640;
/// What ultralytics calls a detector's input.
const DETECT_INPUT_NAME: &str = "images";
/// The first input by position, for an export that calls it something else.
const DETECT_INPUT_INDEX: &str = "0";
/// Classes the COCO-trained detector scores each candidate against.
const CLASSES: usize = 80;
/// Below this score a candidate is not a detection.
const CONF: f32 = 0.25;
/// Above this overlap, two boxes of one class are the same object.
const IOU: f32 = 0.45;

#[derive(Deserialize)]
struct DetectArgs {
    model: String,
    /// A letterboxed 640x640 RGB image, two hex digits per byte. Hex rather
    /// than anything denser because the decoder is five lines and this is a
    /// fixture, not a wire.
    rgb: String,
    #[serde(default = "one")]
    runs: usize,
}

fn one() -> usize {
    1
}

fn unhex(text: &str) -> Result<Vec<u8>, String> {
    if !text.len().is_multiple_of(2) {
        return Err("rgb has an odd number of hex digits".to_string());
    }
    (0..text.len() / 2)
        .map(|i| {
            u8::from_str_radix(&text[i * 2..i * 2 + 2], 16)
                .map_err(|_| format!("rgb is not hex at byte {i}"))
        })
        .collect()
}

struct Detection {
    class: usize,
    score: f32,
    /// Centre x, centre y, width, height, in letterboxed pixels.
    box_: [f32; 4],
}

fn iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let corners = |v: &[f32; 4]| {
        (
            v[0] - v[2] / 2.0,
            v[1] - v[3] / 2.0,
            v[0] + v[2] / 2.0,
            v[1] + v[3] / 2.0,
        )
    };
    let (ax0, ay0, ax1, ay1) = corners(a);
    let (bx0, by0, bx1, by1) = corners(b);
    let w = (ax1.min(bx1) - ax0.max(bx0)).max(0.0);
    let h = (ay1.min(by1) - ay0.max(by0)).max(0.0);
    let intersection = w * h;
    let union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - intersection;
    if union <= 0.0 {
        0.0
    } else {
        intersection / union
    }
}

/// Interleaved RGB bytes as the planar fp32 tensor a detector takes.
fn to_nchw(rgb: &[u8]) -> Vec<u8> {
    let pixels = SIDE * SIDE;
    let mut plane = vec![0f32; pixels * 3];
    for i in 0..pixels {
        plane[i] = f32::from(rgb[i * 3]) / 255.0;
        plane[pixels + i] = f32::from(rgb[i * 3 + 1]) / 255.0;
        plane[2 * pixels + i] = f32::from(rgb[i * 3 + 2]) / 255.0;
    }
    plane.into_iter().flat_map(f32::to_le_bytes).collect()
}

/// Runs a COCO detector over one letterboxed image and returns what it found.
/// The whole decode - candidates over the score, then non-maximum suppression
/// - happens here in the guest; the host only lent it a graph.
fn detect(args: &str) -> Result<String, String> {
    let args: DetectArgs = serde_json::from_str(args).map_err(|e| format!("invalid args: {e}"))?;
    let rgb = unhex(&args.rgb)?;
    if rgb.len() != SIDE * SIDE * 3 {
        return Err(format!(
            "a letterboxed {SIDE}x{SIDE} RGB image is {} bytes, got {}",
            SIDE * SIDE * 3,
            rgb.len()
        ));
    }
    if args.runs == 0 {
        return Err("runs must be at least 1".to_string());
    }

    let graph = load_by_name(&args.model)
        .map_err(|e| failed(&format!("load-by-name({:?})", args.model), &e))?;
    let context = graph
        .init_execution_context()
        .map_err(|e| failed("init-execution-context", &e))?;

    let input = to_nchw(&rgb);
    // Whatever this export calls its input. One that does not use ultralytics'
    // name is reached by position instead, which the host takes where it takes
    // a name; the spelling that worked is kept for the runs after it.
    let mut name = DETECT_INPUT_NAME;
    // The first call is not like the others: a GPU provider picks kernels and
    // builds engines on it, which is seconds where the calls after it are
    // milliseconds. Reported apart, so neither hides the other.
    let mut first = 0.0f64;
    let mut rest = std::time::Duration::ZERO;
    let mut last = None;
    for run in 0..args.runs {
        let dimensions = [1, 3, SIDE as u32, SIDE as u32];
        let tensor = Tensor::new(&dimensions, TensorType::Fp32, &input);
        let started = std::time::Instant::now();
        let outputs = match context.compute(vec![(name.to_string(), tensor)]) {
            Ok(outputs) => outputs,
            Err(_) if name == DETECT_INPUT_NAME => {
                name = DETECT_INPUT_INDEX;
                let tensor = Tensor::new(&dimensions, TensorType::Fp32, &input);
                context
                    .compute(vec![(name.to_string(), tensor)])
                    .map_err(|e| failed("compute", &e))?
            }
            Err(e) => return Err(failed("compute", &e)),
        };
        let elapsed = started.elapsed();
        if run == 0 {
            first = elapsed.as_secs_f64() * 1000.0;
        } else {
            rest += elapsed;
        }
        last = outputs.into_iter().next();
    }
    let mean_ms = if args.runs > 1 {
        rest.as_secs_f64() * 1000.0 / (args.runs - 1) as f64
    } else {
        first
    };

    let (_, out) = last.ok_or_else(|| "the graph returned no output tensor".to_string())?;
    let dims = out.dimensions();
    if dims.len() != 3 {
        return Err(format!("expected a rank-3 output, got dimensions {dims:?}"));
    }
    // The detector's head is [1, 4 + classes, candidates]: the box parameters
    // first, then one score per class.
    let rows = dims[1] as usize;
    let candidates = dims[2] as usize;
    let data = out.data();
    let (whole, _) = data.as_chunks::<4>();
    let values: Vec<f32> = whole.iter().copied().map(f32::from_le_bytes).collect();
    let at = |row: usize, i: usize| values[row * candidates + i];

    let mut found: Vec<Detection> = Vec::new();
    for i in 0..candidates {
        let mut best = 0usize;
        let mut score = 0.0f32;
        for c in 0..CLASSES.min(rows.saturating_sub(4)) {
            let s = at(4 + c, i);
            if s > score {
                score = s;
                best = c;
            }
        }
        if score >= CONF {
            found.push(Detection {
                class: best,
                score,
                box_: [at(0, i), at(1, i), at(2, i), at(3, i)],
            });
        }
    }

    found.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut kept: Vec<Detection> = Vec::new();
    for candidate in found {
        if kept
            .iter()
            .all(|k| k.class != candidate.class || iou(&k.box_, &candidate.box_) < IOU)
        {
            kept.push(candidate);
        }
    }

    let detections: Vec<_> = kept
        .iter()
        .map(|d| {
            let [cx, cy, w, h] = d.box_;
            json!({
                "class": COCO.get(d.class).copied().unwrap_or("?"),
                "score": d.score,
                "box": [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0],
            })
        })
        .collect();

    Ok(json!({
        "detections": detections,
        "first_ms": first,
        "mean_ms": mean_ms,
        "runs": args.runs,
    })
    .to_string())
}

/// The COCO class names, in the order a detector's head scores them.
const COCO: [&str; CLASSES] = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
];

fn sandbox() -> String {
    let mut reachable = Vec::new();
    let mut denied = Vec::new();
    for path in UNREACHABLE {
        match std::fs::read(path) {
            Ok(_) => reachable.push(path),
            Err(_) => denied.push(path),
        }
    }
    json!({ "reachable": reachable, "denied": denied }).to_string()
}

struct NnProbe;

impl Guest for NnProbe {
    fn list_functions() -> Vec<FunctionMeta> {
        vec![
            FunctionMeta {
                name: "run".to_string(),
                params_schema: RUN_PARAMS.to_string(),
                result_schema: RUN_RESULT.to_string(),
            },
            FunctionMeta {
                name: "sandbox".to_string(),
                params_schema: SANDBOX_PARAMS.to_string(),
                result_schema: SANDBOX_RESULT.to_string(),
            },
            FunctionMeta {
                name: "detect".to_string(),
                params_schema: DETECT_PARAMS.to_string(),
                result_schema: DETECT_RESULT.to_string(),
            },
        ]
    }

    fn invoke(name: String, args: String) -> Result<String, String> {
        match name.as_str() {
            "run" => run(&args),
            "sandbox" => Ok(sandbox()),
            "detect" => detect(&args),
            other => Err(format!("nn-probe has no function {other:?}")),
        }
    }
}

export!(NnProbe);
