//! A frame filter that asks `wasi:webgpu` for an adapter and reports what it
//! got, small enough to assert on by hand.
//!
//! Frames pass through untouched. The first frame carries one row saying
//! whether an adapter was found, whether it has subgroups, and its name.
//! Without a `-gpu` grant the host refuses to run it at all; with one on a
//! machine without a GPU the row says `"adapter": false`.
//!
//! With `{"subgroups": true}` it also asks the adapter for a device with
//! subgroups, and the row's `device` is the error `request-device` returned,
//! or empty when the device opened.

wit_bindgen::generate!({
    path: [
        "../../worlds/0.18.0",
        "../../vendor/wasi-webgpu-wasmtime/wit/deps/io",
        "../../vendor/wasi-webgpu-wasmtime/wit/deps/graphics-context",
        "../../vendor/wasi-webgpu-wasmtime/wit/deps/webgpu",
        "wit",
    ],
    // Fully qualified: several packages are in scope, and each has worlds.
    world: "ffrwd:gpu-probe/gpu-probe",
    generate_all,
});

use std::cell::RefCell;

use exports::ffrwd::av::filter::{FrameInfo, Guest, Meta, Outcome, Output, StreamInfo};
use wasi::webgpu::webgpu::{get_gpu, GpuDeviceDescriptor, GpuFeatureName};

const PARAMS_SCHEMA: &str = r#"{"type":"object","properties":{"subgroups":{"type":"boolean"}},"additionalProperties":false}"#;
const ROWS_SCHEMA: &str = r#"{"type":"object","properties":{"adapter":{"type":"boolean"},"subgroups":{"type":"boolean"},"name":{"type":"string"},"device":{"type":"string"}},"required":["adapter","subgroups","name"]}"#;

thread_local! {
    /// The row the first frame carries, taken when it is emitted.
    static FOUND: RefCell<Option<String>> = const { RefCell::new(None) };
}

/// What the host's GPU looks like to this module, as one row; with
/// `request_subgroups`, also what asking for a device with subgroups got.
fn probe(request_subgroups: bool) -> String {
    let adapter = get_gpu().request_adapter(None);
    let (found, subgroups, name) = match &adapter {
        Some(adapter) => (
            true,
            adapter.features().has("subgroups"),
            adapter.info().device(),
        ),
        None => (false, false, String::new()),
    };
    let device = match (&adapter, request_subgroups) {
        (Some(adapter), true) => {
            let descriptor = GpuDeviceDescriptor {
                required_features: Some(vec![GpuFeatureName::Subgroups]),
                required_limits: None,
                default_queue: None,
                label: None,
            };
            let error = match adapter.request_device(Some(descriptor)) {
                Ok(_) => String::new(),
                Err(error) => error.message,
            };
            format!(r#","device":{}"#, json_string(&error))
        }
        _ => String::new(),
    };
    format!(
        r#"{{"adapter":{found},"subgroups":{subgroups},"name":{}{device}}}"#,
        json_string(&name)
    )
}

fn json_string(text: &str) -> String {
    let mut out = String::from("\"");
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// Whether the params ask for a device with subgroups.
fn parse_params(params: &str) -> Result<bool, String> {
    if params.trim().is_empty() {
        return Ok(false);
    }
    let value: serde_json::Value =
        serde_json::from_str(params).map_err(|e| format!("gpu-probe params: {e}"))?;
    let Some(object) = value.as_object() else {
        return Err(format!("gpu-probe params are an object, got: {params}"));
    };
    if let Some(key) = object.keys().find(|key| *key != "subgroups") {
        return Err(format!("gpu-probe takes only `subgroups`, got: {key}"));
    }
    match object.get("subgroups") {
        None => Ok(false),
        Some(serde_json::Value::Bool(b)) => Ok(*b),
        Some(other) => Err(format!(
            "gpu-probe's `subgroups` is a boolean, got: {other}"
        )),
    }
}

struct GpuProbe;

impl Guest for GpuProbe {
    fn describe() -> Meta {
        Meta {
            name: "gpu-probe".to_string(),
            version: "0.1.0".to_string(),
            params_schema: PARAMS_SCHEMA.to_string(),
            rows_schema: ROWS_SCHEMA.to_string(),
            pixel_formats: vec!["rgba".to_string(), "yuv420p".to_string()],
            sample_formats: vec![],
            sample_rates: vec![],
            channel_counts: vec![],
            rows_language: vec![],
        }
    }

    fn init(
        _width: u32,
        _height: u32,
        _pix_fmt: String,
        _stream_info: StreamInfo,
        params: String,
    ) -> Result<(), String> {
        let request_subgroups = parse_params(&params)?;
        FOUND.with(|found| *found.borrow_mut() = Some(probe(request_subgroups)));
        Ok(())
    }

    fn set_params(params: String) -> Result<(), String> {
        parse_params(&params).map(|_| ())
    }

    fn frame_independent() -> bool {
        // One row for the whole stream, on whichever frame comes first.
        false
    }

    fn process(_info: FrameInfo, _frame: Vec<u8>) -> Outcome {
        Outcome {
            output: Output::Passthrough,
            rows: FOUND.with(|found| found.borrow_mut().take().into_iter().collect()),
        }
    }
}

export!(GpuProbe);
