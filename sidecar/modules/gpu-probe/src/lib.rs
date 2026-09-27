//! A frame filter that asks `wasi:webgpu` for an adapter and reports what it
//! got, small enough to assert on by hand.
//!
//! Frames pass through untouched. The first frame carries one row saying
//! whether an adapter was found, whether it has subgroups, and its name.
//! Without a `-gpu` grant the host refuses to run it at all; with one on a
//! machine without a GPU the row says `"adapter": false`.

wit_bindgen::generate!({
    path: [
        "../../wit",
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
use wasi::webgpu::webgpu::get_gpu;

const PARAMS_SCHEMA: &str = r#"{"type":"object","properties":{},"additionalProperties":false}"#;
const ROWS_SCHEMA: &str = r#"{"type":"object","properties":{"adapter":{"type":"boolean"},"subgroups":{"type":"boolean"},"name":{"type":"string"}},"required":["adapter","subgroups","name"]}"#;

thread_local! {
    /// The row the first frame carries, taken when it is emitted.
    static FOUND: RefCell<Option<String>> = const { RefCell::new(None) };
}

/// What the host's GPU looks like to this module, as one row.
fn probe() -> String {
    let adapter = get_gpu().request_adapter(None);
    let (found, subgroups, name) = match &adapter {
        Some(adapter) => (
            true,
            adapter.features().has("subgroups"),
            adapter.info().device(),
        ),
        None => (false, false, String::new()),
    };
    format!(
        r#"{{"adapter":{found},"subgroups":{subgroups},"name":{}}}"#,
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

fn validate_params(params: &str) -> Result<(), String> {
    match params.trim() {
        "" | "{}" => Ok(()),
        other => Err(format!("gpu-probe takes no params, got: {other}")),
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
        validate_params(&params)?;
        FOUND.with(|found| *found.borrow_mut() = Some(probe()));
        Ok(())
    }

    fn set_params(params: String) -> Result<(), String> {
        validate_params(&params)
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
