//! Two streams in, pad 0 out: the fixture that shows lockstep delivery
//! without a model or any arithmetic in the way.
//!
//! Every call is handed one frame per pad at one timestamp. This module keeps
//! pad 0's and drops the rest, so an output byte-identical to pad 0's input
//! says the pads arrived paired and in order. It emits one row per call
//! naming how many pads it received and what timestamp they shared, which is
//! how a test reads the pairing rather than infers it.

wit_bindgen::generate!({
    path: "../../worlds/0.10.0",
    world: "window-module",
});

use exports::ffrwd::av::window_filter::{
    Format, FramePayload, Guest, InFrame, Meta, OutFrame, Processed, StreamInfo, WindowMeta,
};

const PARAMS_SCHEMA: &str = r#"{"type":"object","properties":{},"additionalProperties":false}"#;
const ROWS_SCHEMA: &str = r#"{"type":"object","properties":{"pads":{"type":"integer"},"pts":{"type":"integer"},"rows-in":{"type":"integer"}},"additionalProperties":false}"#;

/// The streams this module reads: the one it keeps, and the one it drops.
const INPUTS: u32 = 2;

struct PickFirst;

fn validate_params(params: &str) -> Result<(), String> {
    match params.trim() {
        "" | "{}" => Ok(()),
        other => Err(format!("pick_first takes no params, got: {other}")),
    }
}

impl Guest for PickFirst {
    fn describe() -> WindowMeta {
        WindowMeta {
            meta: Meta {
                name: "pick_first".to_string(),
                version: "0.1.0".to_string(),
                params_schema: PARAMS_SCHEMA.to_string(),
                rows_schema: ROWS_SCHEMA.to_string(),
                pixel_formats: vec!["rgba".to_string(), "yuv420p".to_string()],
                sample_formats: vec![],
                sample_rates: vec![],
                channel_counts: vec![],
                rows_language: vec![],
            },
            window: 1,
            stride: 1,
            pure: true,
            one_to_one: true,
            // It counts the rows it was handed into a row of its own, which
            // is acting on them rather than only carrying them through.
            reads_rows: true,
            forwards_rows: true,
            inputs: INPUTS,
        }
    }

    fn init(format: Format, _stream_info: StreamInfo, params: String) -> Result<(), String> {
        let Format::Video(_) = format else {
            return Err("pick_first reads frames, and this stream is audio".to_string());
        };
        validate_params(&params)
    }

    fn set_params(params: String) -> Result<(), String> {
        validate_params(&params)
    }

    fn process(frames: Vec<InFrame>, _trailing: Vec<String>, _last: bool) -> Processed {
        // The final call carries nothing: window and stride are 1, so no
        // frame is ever left over.
        let Some(first) = frames.first() else {
            return Processed {
                frames: vec![],
                trailing: vec![],
            };
        };
        let mut rows = vec![format!(
            r#"{{"pads":{},"pts":{},"rows-in":{}}}"#,
            frames.len(),
            first.pts,
            first.rows.len()
        )];
        rows.extend(first.rows.iter().cloned());
        Processed {
            frames: vec![OutFrame {
                pts: first.pts,
                // Pad 0's bytes, which this module never copied out.
                frame: FramePayload::Same,
                rows,
            }],
            trailing: vec![],
        }
    }
}

export!(PickFirst);
