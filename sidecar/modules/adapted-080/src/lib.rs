//! A windowed module of the world before a module said how many streams it
//! reads, so the adapter carrying that world is exercised.
//!
//! Frames pass through one at a time. Each carries away the language its
//! rows are in - which is what that world added, a description naming the
//! params that settle it - and how many rows arrived with the frame. Rows
//! that arrived travel on beside it, and the trailing rows the stream ends
//! with are counted into one row of this module's own.
//!
//! The description of that world has no place to say how many streams a
//! module reads, so the host answers one on its behalf.

wit_bindgen::generate!({
    path: "../../worlds/0.8.0",
    world: "window-module",
});

use std::cell::RefCell;

use exports::ffrwd::av::window_filter::{
    Format, FramePayload, Guest, InFrame, Meta, OutFrame, Processed, StreamInfo, WindowMeta,
};

const PARAMS_SCHEMA: &str =
    r#"{"type":"object","properties":{"language":{"type":"string"}},"additionalProperties":false}"#;
const ROWS_SCHEMA: &str = r#"{"type":"object","properties":{"language":{"type":"string"},"rows-in":{"type":"integer"},"trailing-in":{"type":"integer"}},"additionalProperties":false}"#;

thread_local! {
    static LANGUAGE: RefCell<String> = const { RefCell::new(String::new()) };
}

struct Adapted080;

/// The `language` param out of a params object, or empty when it is unset.
/// Small enough to read by hand: this module carries no JSON library.
fn language_of(params: &str) -> String {
    let Some(at) = params.find("\"language\"") else {
        return String::new();
    };
    let rest = &params[at + "\"language\"".len()..];
    let Some(colon) = rest.find(':') else {
        return String::new();
    };
    let value = rest[colon + 1..].trim_start();
    let Some(stripped) = value.strip_prefix('"') else {
        return String::new();
    };
    match stripped.find('"') {
        Some(end) => stripped[..end].to_string(),
        None => String::new(),
    }
}

impl Guest for Adapted080 {
    fn describe() -> WindowMeta {
        WindowMeta {
            meta: Meta {
                name: "adapted-080".to_string(),
                version: "0.1.0".to_string(),
                params_schema: PARAMS_SCHEMA.to_string(),
                rows_schema: ROWS_SCHEMA.to_string(),
                pixel_formats: vec!["rgba".to_string()],
                sample_formats: vec![],
                sample_rates: vec![],
                channel_counts: vec![],
                // What that world added: the param whose value says which
                // language this module's rows are in.
                rows_language: vec!["language".to_string()],
            },
            window: 1,
            stride: 1,
            pure: true,
            one_to_one: true,
            reads_rows: true,
            forwards_rows: true,
        }
    }

    fn init(format: Format, _stream_info: StreamInfo, params: String) -> Result<(), String> {
        let Format::Video(_) = format else {
            return Err("adapted-080 reads frames, and this stream is audio".to_string());
        };
        LANGUAGE.with(|l| *l.borrow_mut() = language_of(&params));
        Ok(())
    }

    fn set_params(params: String) -> Result<(), String> {
        LANGUAGE.with(|l| *l.borrow_mut() = language_of(&params));
        Ok(())
    }

    fn process(frames: Vec<InFrame>, trailing: Vec<String>, last: bool) -> Processed {
        let language = LANGUAGE.with(|l| l.borrow().clone());

        let out: Vec<OutFrame> = frames
            .into_iter()
            .map(|frame| {
                let mut rows = vec![format!(
                    r#"{{"language":"{language}","rows-in":{}}}"#,
                    frame.rows.len()
                )];
                rows.extend(frame.rows);
                OutFrame {
                    pts: frame.pts,
                    frame: FramePayload::Same,
                    rows,
                }
            })
            .collect();

        // The trailing rows have no frame to ride, so what they amount to
        // leaves as one trailing row of this module's own.
        let trailing_out = if last {
            vec![format!(r#"{{"trailing-in":{}}}"#, trailing.len())]
        } else {
            Vec::new()
        };
        Processed {
            frames: out,
            trailing: trailing_out,
        }
    }
}

export!(Adapted080);
