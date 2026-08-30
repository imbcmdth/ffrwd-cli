//! An audio module that breaks the two rules the host holds an audio module
//! to, so that the refusals are exercised by something actually shaped that
//! way rather than only described.
//!
//! Its windows overlap - a stride of half its window - and it declares itself
//! one-to-one, which no honest module could be at that shape. What it does
//! with each window is the `wrong` parameter:
//!
//! - `same` passes the window through as the samples it was handed. Every
//!   sample would leave twice over, so the host refuses it.
//! - `short` returns half the window, which is its stride, at the window's own
//!   timestamp. Each call on its own looks continuous, and the stream still
//!   ends shorter than it began: what catches this is the count over the
//!   instance's life.
//! - `gap` returns a quarter of the window. The second call then starts past
//!   where the first one left off, which is a hole in the output, and the
//!   host refuses it there rather than waiting for the count.
//!
//! Nothing wires this into a pipeline; it exists for the host's tests.

wit_bindgen::generate!({
    path: "../../wit",
    world: "window-module",
});

use std::cell::RefCell;

use exports::ffrwd::av::window_filter::{
    Format, FramePayload, Guest, InFrame, Meta, OutFrame, Processed, StreamInfo, WindowMeta,
};
use serde::Deserialize;

const PARAMS_SCHEMA: &str = r#"{"type":"object","properties":{"wrong":{"type":"string","enum":["same","short","gap"],"default":"same"}},"additionalProperties":false}"#;

const WINDOW: u32 = 2048;
const STRIDE: u32 = 1024;

fn default_wrong() -> String {
    "same".to_string()
}

#[derive(Deserialize)]
struct Params {
    #[serde(default = "default_wrong")]
    wrong: String,
}

/// Which rule this instance breaks.
#[derive(Clone, Copy)]
enum Wrong {
    Same,
    Short,
    Gap,
}

thread_local! {
    static WRONG: RefCell<Option<Wrong>> = const { RefCell::new(None) };
}

fn parse_params(params: &str) -> Result<Wrong, String> {
    let trimmed = params.trim();
    let parsed: Params = if trimmed.is_empty() {
        Params {
            wrong: default_wrong(),
        }
    } else {
        serde_json::from_str(trimmed).map_err(|e| format!("invalid params: {e}"))?
    };
    match parsed.wrong.as_str() {
        "same" => Ok(Wrong::Same),
        "short" => Ok(Wrong::Short),
        "gap" => Ok(Wrong::Gap),
        other => Err(format!("wrong must be same, short or gap, got {other:?}")),
    }
}

/// The first `1 / divisor` of a window, as new samples.
fn keep(window: &[u8], divisor: usize) -> FramePayload {
    let samples = window.len() / (4 * divisor);
    FramePayload::New(window[..samples * 4].to_vec())
}

struct BrokenAudio;

impl Guest for BrokenAudio {
    fn describe() -> WindowMeta {
        WindowMeta {
            meta: Meta {
                name: "broken-audio".to_string(),
                version: "0.1.0".to_string(),
                params_schema: PARAMS_SCHEMA.to_string(),
                rows_schema: String::new(),
                pixel_formats: vec![],
                sample_formats: vec!["f32".to_string()],
                sample_rates: vec![],
                channel_counts: vec![],
                rows_language: vec![],
            },
            window: WINDOW,
            // Half the window, so the windows overlap.
            stride: STRIDE,
            pure: true,
            // The claim the host holds it to, and the claim it breaks.
            one_to_one: true,
            reads_rows: false,
            forwards_rows: false,
            // One stream in, which is every module here.
            inputs: 1,
        }
    }

    fn init(format: Format, _stream_info: StreamInfo, params: String) -> Result<(), String> {
        let wrong = parse_params(&params)?;
        let Format::Audio(_) = format else {
            return Err("broken-audio reads samples, and this stream is video".to_string());
        };
        WRONG.with(|w| *w.borrow_mut() = Some(wrong));
        Ok(())
    }

    fn set_params(params: String) -> Result<(), String> {
        let wrong = parse_params(&params)?;
        WRONG.with(|w| *w.borrow_mut() = Some(wrong));
        Ok(())
    }

    fn process(frames: Vec<InFrame>, _trailing: Vec<String>, _last: bool) -> Processed {
        let wrong = WRONG.with(|w| w.borrow().expect("process called before init"));
        let out = frames
            .into_iter()
            .map(|frame| OutFrame {
                pts: frame.pts,
                frame: match wrong {
                    Wrong::Same => FramePayload::Same,
                    Wrong::Short => keep(&frame.frame, 2),
                    Wrong::Gap => keep(&frame.frame, 4),
                },
                rows: vec![],
            })
            .collect();
        Processed {
            frames: out,
            trailing: Vec::new(),
        }
    }
}

export!(BrokenAudio);
