//! Two streams in, pad 0 out, and a row saying what rows rode each pad.
//!
//! The fixture that shows which pad a module reading several streams is
//! handed rows on. Every call carries one frame per pad at one timestamp;
//! this module keeps pad 0's frame and writes one row of its own: how many
//! rows arrived on pad 0, how many on every other pad together, and the
//! `note` of each pad 0 row. A test feeds a producer into each pad and reads
//! the answer off that row rather than inferring it from pictures.

wit_bindgen::generate!({
    path: "../../worlds/0.18.0",
    world: "window-module",
});

use exports::ffrwd::av::window_filter::{
    Format, FramePayload, Guest, InWindow, Meta, OutFrame, Processed, StreamInfo, WindowMeta,
};
use serde::{Deserialize, Serialize};

const PARAMS_SCHEMA: &str = r#"{"type":"object","properties":{},"additionalProperties":false}"#;
const ROWS_SCHEMA: &str = r#"{"type":"object","properties":{"pts":{"type":"integer"},"pad0":{"type":"integer"},"others":{"type":"integer"},"notes":{"type":"string"}},"required":["pts","pad0","others","notes"],"additionalProperties":false}"#;

/// The streams this module reads: the one whose rows it is handed, and one
/// whose rows never reach it.
const INPUTS: u32 = 2;

/// The one field this module reads of a row handed to it.
#[derive(Deserialize)]
struct Note {
    #[serde(default)]
    note: String,
}

/// What one call saw.
#[derive(Serialize)]
struct Seen {
    pts: i64,
    pad0: usize,
    others: usize,
    notes: String,
}

struct PadRows;

fn validate_params(params: &str) -> Result<(), String> {
    match params.trim() {
        "" | "{}" => Ok(()),
        other => Err(format!("pad_rows takes no params, got: {other}")),
    }
}

impl Guest for PadRows {
    fn describe() -> WindowMeta {
        WindowMeta {
            meta: Meta {
                name: "pad_rows".to_string(),
                version: "0.1.0".to_string(),
                params_schema: PARAMS_SCHEMA.to_string(),
                rows_schema: ROWS_SCHEMA.to_string(),
                pixel_formats: vec!["rgba".to_string()],
                sample_formats: vec![],
                sample_rates: vec![],
                channel_counts: vec![],
                rows_language: vec![],
            },
            window: 1,
            stride: 1,
            pure: true,
            one_to_one: true,
            reads_rows: true,
            // The rows leaving are this module's own account of the ones
            // that arrived.
            forwards_rows: false,
            inputs: INPUTS,
            feeders: vec![],
        }
    }

    fn init(format: Format, _stream_info: StreamInfo, params: String) -> Result<(), String> {
        let Format::Video(_) = format else {
            return Err("pad_rows reads pictures, and this stream is audio".to_string());
        };
        validate_params(&params)
    }

    fn set_params(params: String) -> Result<(), String> {
        validate_params(&params)
    }

    fn process(window: &InWindow, _trailing: Vec<String>, _last: bool) -> Processed {
        // The final call carries nothing: window and stride are 1, so no
        // frame is ever left over.
        if window.len() == 0 {
            return Processed {
                frames: vec![],
                trailing: vec![],
            };
        }
        let first = window.rows(0);
        let others = (1..window.len()).map(|pad| window.rows(pad).len()).sum();
        let notes: Vec<String> = first
            .iter()
            .map(|row| {
                serde_json::from_str::<Note>(row)
                    .map(|read| read.note)
                    .unwrap_or_default()
            })
            .collect();
        let seen = Seen {
            pts: window.pts(0),
            pad0: first.len(),
            others,
            notes: notes.join("+"),
        };
        Processed {
            frames: vec![OutFrame {
                pts: window.pts(0),
                // Pad 0's bytes, which this module never copied out.
                frame: FramePayload::Same,
                rows: vec![serde_json::to_string(&seen).expect("a row serializes")],
            }],
            trailing: vec![],
        }
    }
}

export!(PadRows);
