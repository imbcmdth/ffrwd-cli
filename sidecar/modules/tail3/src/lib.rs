//! Three frames at a time, passed through untouched, with a row per call
//! recording how many frames that call carried and whether it was the last.
//! What it exists to show is the tail: a stream whose length is not a
//! multiple of three ends with a short call, and that call is the one marked
//! `last`.
//!
//! The row rides on the call's first frame, so a final call carrying no
//! frames - which is how a stream of exactly three, six or nine ends - has
//! nowhere to put a row and emits none.

wit_bindgen::generate!({
    path: "../../worlds/0.10.0",
    world: "window-module",
});

use exports::ffrwd::av::window_filter::{
    Format, FramePayload, Guest, InFrame, Meta, OutFrame, Processed, StreamInfo, WindowMeta,
};
use serde::Serialize;

const PARAMS_SCHEMA: &str = r#"{"type":"object","properties":{},"additionalProperties":false}"#;
const ROWS_SCHEMA: &str = r#"{"type":"object","properties":{"frames":{"type":"integer"},"last":{"type":"boolean"}},"additionalProperties":false}"#;

#[derive(Serialize)]
struct Row {
    frames: usize,
    last: bool,
}

/// Validates that `params` is empty or `{}`; tail3 takes no parameters.
fn validate_params(params: &str) -> Result<(), String> {
    match params.trim() {
        "" | "{}" => Ok(()),
        other => Err(format!("tail3 takes no params, got: {other}")),
    }
}

struct Tail3;

impl Guest for Tail3 {
    fn describe() -> WindowMeta {
        WindowMeta {
            meta: Meta {
                name: "tail3".to_string(),
                version: "0.1.0".to_string(),
                params_schema: PARAMS_SCHEMA.to_string(),
                rows_schema: ROWS_SCHEMA.to_string(),
                pixel_formats: vec!["rgba".to_string(), "yuv420p".to_string()],
                // Not an audio module, so it names no sample formats.
                sample_formats: vec![],
                sample_rates: vec![],
                channel_counts: vec![],
                rows_language: vec![],
            },
            window: 3,
            stride: 3,
            // Nothing carries over between calls.
            pure: true,
            one_to_one: true,
            // Incoming rows are passed on beside this module's own, not read.
            reads_rows: false,
            forwards_rows: true,
            // One stream in, which is every module here.
            inputs: 1,
        }
    }

    fn init(_format: Format, _stream_info: StreamInfo, params: String) -> Result<(), String> {
        validate_params(&params)
    }

    fn set_params(params: String) -> Result<(), String> {
        validate_params(&params)
    }

    fn process(frames: Vec<InFrame>, trailing: Vec<String>, last: bool) -> Processed {
        let row = Row {
            frames: frames.len(),
            last,
        };
        let mut call_rows = vec![serde_json::to_string(&row).expect("row serializes")];

        let out = frames
            .into_iter()
            .map(|frame| {
                let mut rows = std::mem::take(&mut call_rows);
                rows.extend(frame.rows);
                OutFrame {
                    pts: frame.pts,
                    frame: FramePayload::Same,
                    rows,
                }
            })
            .collect();

        // Trailing rows are carried on the same way rows riding a frame are.
        Processed {
            frames: out,
            trailing,
        }
    }
}

export!(Tail3);
