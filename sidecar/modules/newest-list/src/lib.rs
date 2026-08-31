//! Reads only the newest frame of a fifteen-frame window and passes it
//! through, with a row summing that frame's bytes. Half of a measured pair:
//! this one is built against the world whose `process` receives the whole
//! window as a list, so every call pays the copy of fifteen frames to read
//! one. `newest-borrow` is the same filter against the world that borrows
//! the window instead; the two produce identical output.
//!
//! A call short of a full window - the ramp at the end of the stream -
//! emits nothing, so every frame leaves exactly once, when it is newest.

wit_bindgen::generate!({
    path: "../../worlds/0.10.0",
    world: "window-module",
});

use exports::ffrwd::av::window_filter::{
    Format, FramePayload, Guest, InFrame, Meta, OutFrame, Processed, StreamInfo, WindowMeta,
};
use serde::Serialize;

const WINDOW: u32 = 15;

const PARAMS_SCHEMA: &str = r#"{"type":"object","properties":{},"additionalProperties":false}"#;
const ROWS_SCHEMA: &str =
    r#"{"type":"object","properties":{"sum":{"type":"integer"}},"additionalProperties":false}"#;

#[derive(Serialize)]
struct Row {
    sum: u64,
}

/// Validates that `params` is empty or `{}`; this module takes none.
fn validate_params(params: &str) -> Result<(), String> {
    match params.trim() {
        "" | "{}" => Ok(()),
        other => Err(format!("newest-list takes no params, got: {other}")),
    }
}

struct NewestList;

impl Guest for NewestList {
    fn describe() -> WindowMeta {
        WindowMeta {
            meta: Meta {
                name: "newest-list".to_string(),
                version: "0.1.0".to_string(),
                params_schema: PARAMS_SCHEMA.to_string(),
                rows_schema: ROWS_SCHEMA.to_string(),
                pixel_formats: vec!["rgba".to_string(), "yuv420p".to_string()],
                sample_formats: vec![],
                sample_rates: vec![],
                channel_counts: vec![],
                rows_language: vec![],
            },
            window: WINDOW,
            stride: 1,
            pure: true,
            // A full window emits one frame at the newest pts, not one per
            // frame consumed.
            one_to_one: false,
            reads_rows: false,
            forwards_rows: false,
            inputs: 1,
        }
    }

    fn init(_format: Format, _stream_info: StreamInfo, params: String) -> Result<(), String> {
        validate_params(&params)
    }

    fn set_params(params: String) -> Result<(), String> {
        validate_params(&params)
    }

    fn process(frames: Vec<InFrame>, trailing: Vec<String>, _last: bool) -> Processed {
        let mut out = Vec::new();
        if frames.len() == WINDOW as usize {
            let newest = frames.last().expect("a full window has a newest frame");
            let row = Row {
                sum: newest.frame.iter().map(|b| u64::from(*b)).sum(),
            };
            out.push(OutFrame {
                pts: newest.pts,
                frame: FramePayload::Same,
                rows: vec![serde_json::to_string(&row).expect("row serializes")],
            });
        }
        Processed {
            frames: out,
            trailing,
        }
    }
}

export!(NewestList);
