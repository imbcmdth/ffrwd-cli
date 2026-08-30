//! A real object detector through the runtime, no ffmpeg and no argv.
//!
//! An image is millions of bytes and a command line holds tens of thousands,
//! so this drives `runtime::invoke` directly the way the transcribe tests
//! drive `Filter`. The argv that reaches the same place is `ffrwd-wasm`'s own
//! `tests/nn.rs`.
//!
//! Neither the weights nor the image are in git, so this skips - loudly -
//! unless the environment names both:
//!
//!   FFRWD_NN_YOLO_MODEL   a yolov8n.onnx
//!   FFRWD_NN_YOLO_INPUT   that image letterboxed to 640x640, raw interleaved
//!                         RGB, 1228800 bytes
//!   FFRWD_NN_YOLO_TARGET  any -nn-target spelling; cpu is the default
//!   FFRWD_NN_RUNTIME      where ONNX Runtime is, else the cache

use std::path::PathBuf;
use std::process::Command;

use ffrwd_wasm_runtime::nn;
use ffrwd_wasm_runtime::runtime;

/// The side the detector's input takes, and the bytes of one letterboxed
/// image at that size.
const SIDE: usize = 640;
const IMAGE_LEN: usize = SIDE * SIDE * 3;

/// How many times the graph runs, so the reported time is not one cold call.
const RUNS: usize = 10;

fn sidecar_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("runtime/ has a parent directory")
        .to_path_buf()
}

fn probe_module() -> PathBuf {
    let output = Command::new("cargo")
        .args([
            "build",
            "--release",
            "--target",
            "wasm32-wasip2",
            "-p",
            "nn-probe",
        ])
        .current_dir(sidecar_root().join("modules"))
        .output()
        .expect("spawn cargo build for nn-probe");
    assert!(
        output.status.success(),
        "building nn-probe failed (status {:?}):\n{}",
        output.status.code(),
        String::from_utf8_lossy(&output.stderr)
    );
    sidecar_root().join("modules/target/wasm32-wasip2/release/nn_probe.wasm")
}
/// Where `ffrwd setup nn` puts a runtime: the user's cache, keyed by the
/// version this build demands and the platform it runs on.
fn cached_runtime_dir() -> Option<PathBuf> {
    let home = std::env::var_os(if cfg!(windows) { "USERPROFILE" } else { "HOME" })?;
    Some(
        PathBuf::from(home)
            .join(".cache/ffrwd/nn-runtime")
            .join(nn::ort_version())
            .join(nn::platform()),
    )
}

fn runtime_dir() -> Option<PathBuf> {
    let named = std::env::var_os(nn::RUNTIME_DIR_VAR).map(PathBuf::from);
    let candidates = [named, cached_runtime_dir()];
    candidates.into_iter().flatten().find(|dir| {
        [
            "onnxruntime.dll",
            "libonnxruntime.so",
            "libonnxruntime.dylib",
        ]
        .iter()
        .any(|lib| dir.join(lib).is_file())
    })
}

#[test]
fn a_real_detector_runs_through_the_sidecar() {
    let (Some(model), Some(input)) = (
        std::env::var_os("FFRWD_NN_YOLO_MODEL").map(PathBuf::from),
        std::env::var_os("FFRWD_NN_YOLO_INPUT").map(PathBuf::from),
    ) else {
        eprintln!(
            "SKIPPED: set FFRWD_NN_YOLO_MODEL and FFRWD_NN_YOLO_INPUT to run a real \
             detector through the sidecar."
        );
        return;
    };
    if !model.is_file() || !input.is_file() {
        eprintln!(
            "SKIPPED: {} or {} is not there.",
            model.display(),
            input.display()
        );
        return;
    }
    let Some(dir) = runtime_dir() else {
        eprintln!(
            "SKIPPED: a detector needs an ONNX Runtime. Run \
             `ffrwd setup nn` to download one."
        );
        return;
    };

    let target = match std::env::var("FFRWD_NN_YOLO_TARGET") {
        Ok(raw) => nn::Target::parse(&raw).expect("FFRWD_NN_YOLO_TARGET names a target"),
        Err(_) => nn::Target::Cpu,
    };
    nn::configure(&nn::Config {
        models: vec![nn::ModelSpec {
            name: "yolo".to_string(),
            path: model,
        }],
        runtime_dir: Some(dir),
        target,
    })
    .expect("loading the detector");

    let rgb = std::fs::read(&input).expect("reading the letterboxed image");
    assert_eq!(
        rgb.len(),
        IMAGE_LEN,
        "the input is a letterboxed {SIDE}x{SIDE} RGB image"
    );
    let hex: String = rgb.iter().map(|b| format!("{b:02x}")).collect();
    let args = serde_json::json!({ "model": "yolo", "rgb": hex, "runs": RUNS }).to_string();

    let module = probe_module();
    let result = runtime::invoke(&module.to_string_lossy(), "detect", &args)
        .expect("the host drove the module")
        .expect("the module ran the detector");
    let result: serde_json::Value = serde_json::from_str(&result).expect("the result is JSON");

    let detections = result["detections"]
        .as_array()
        .expect("detections is an array");
    assert!(
        !detections.is_empty(),
        "a detector over a photograph finds something: {result}"
    );

    eprintln!(
        "{} detections on {}: first call {:.1} ms, then {:.1} ms/run over {} more",
        detections.len(),
        target.label(),
        result["first_ms"].as_f64().unwrap_or_default(),
        result["mean_ms"].as_f64().unwrap_or_default(),
        RUNS - 1
    );
    for detection in detections {
        eprintln!(
            "  {:<14} {:.3}  {}",
            detection["class"].as_str().unwrap_or("?"),
            detection["score"].as_f64().unwrap_or_default(),
            detection["box"]
        );
    }
}
