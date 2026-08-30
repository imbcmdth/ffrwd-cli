//! A windowed module of the world before the packet sink existed, so the
//! adapter carrying that world is exercised by a module actually shaped
//! that way.
//!
//! Frames pass through one at a time, each carrying away one row naming its
//! timestamp. That world already said how many streams a module reads, so
//! the declaration crosses the adapter rather than being answered for it.

wit_bindgen::generate!({
    path: "../../worlds/0.9.0",
    world: "window-module",
});

use exports::ffrwd::av::window_filter::{
    Format, FramePayload, Guest, InFrame, Meta, OutFrame, Processed, StreamInfo, WindowMeta,
};

const PARAMS_SCHEMA: &str = r#"{"type":"object","properties":{},"additionalProperties":false}"#;
const ROWS_SCHEMA: &str =
    r#"{"type":"object","properties":{"pts":{"type":"integer"}},"additionalProperties":false}"#;

struct Adapted090;

impl Guest for Adapted090 {
    fn describe() -> WindowMeta {
        WindowMeta {
            meta: Meta {
                name: "adapted-090".to_string(),
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
            reads_rows: false,
            forwards_rows: false,
            // What that world added: the module says itself how many streams
            // it reads.
            inputs: 1,
        }
    }

    fn init(format: Format, _stream_info: StreamInfo, params: String) -> Result<(), String> {
        let Format::Video(_) = format else {
            return Err("adapted-090 reads frames, and this stream is audio".to_string());
        };
        match params.trim() {
            "" | "{}" => Ok(()),
            other => Err(format!("adapted-090 takes no params, got: {other}")),
        }
    }

    fn set_params(params: String) -> Result<(), String> {
        match params.trim() {
            "" | "{}" => Ok(()),
            other => Err(format!("adapted-090 takes no params, got: {other}")),
        }
    }

    fn process(frames: Vec<InFrame>, _trailing: Vec<String>, _last: bool) -> Processed {
        let out: Vec<OutFrame> = frames
            .into_iter()
            .map(|frame| OutFrame {
                pts: frame.pts,
                frame: FramePayload::Same,
                rows: vec![format!(r#"{{"pts":{}}}"#, frame.pts)],
            })
            .collect();
        Processed {
            frames: out,
            trailing: Vec::new(),
        }
    }
}

export!(Adapted090);
