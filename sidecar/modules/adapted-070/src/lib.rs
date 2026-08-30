//! A windowed module of the world before the description named a language for
//! its rows, so the adapter carrying that world is exercised.
//!
//! An audio module, because audio is what that world added: its `init` is
//! handed a kind-bearing format rather than a frame size, and its windows and
//! strides count samples. Samples pass through as the ones that arrived, 1024
//! at a time. Each window carries away what the format arm said - the rate,
//! the channel count and the sample format - and how many rows arrived with
//! it; rows that arrived travel on beside it, and the trailing rows the stream
//! ends with are counted into one row of this module's own.
//!
//! The description of that world has no place to name a language for its rows,
//! so the host answers an empty list on its behalf.

wit_bindgen::generate!({
    path: "../../worlds/0.7.0",
    world: "window-module",
});

use std::cell::RefCell;

use exports::ffrwd::av::window_filter::{
    Format, FramePayload, Guest, InFrame, Meta, OutFrame, Processed, StreamInfo, WindowMeta,
};

const PARAMS_SCHEMA: &str = r#"{"type":"object","properties":{},"additionalProperties":false}"#;
const ROWS_SCHEMA: &str = r#"{"type":"object","properties":{"opened-as":{"type":"string"},"rows-in":{"type":"integer"},"trailing-in":{"type":"integer"}},"additionalProperties":false}"#;

/// Samples one call works in, disjoint from the next so the samples that
/// arrive are the ones that leave.
const WINDOW: u32 = 1024;

thread_local! {
    static OPENED_AS: RefCell<String> = const { RefCell::new(String::new()) };
}

struct Adapted070;

impl Guest for Adapted070 {
    fn describe() -> WindowMeta {
        WindowMeta {
            meta: Meta {
                name: "adapted-070".to_string(),
                version: "0.1.0".to_string(),
                params_schema: PARAMS_SCHEMA.to_string(),
                rows_schema: ROWS_SCHEMA.to_string(),
                // An audio module, so it names no pixel formats.
                pixel_formats: vec![],
                sample_formats: vec!["f32".to_string()],
                // Nothing here depends on the rate or the channel count.
                sample_rates: vec![],
                channel_counts: vec![],
            },
            window: WINDOW,
            stride: WINDOW,
            pure: true,
            one_to_one: true,
            reads_rows: true,
            forwards_rows: true,
        }
    }

    fn init(format: Format, _stream_info: StreamInfo, _params: String) -> Result<(), String> {
        let Format::Audio(audio) = format else {
            return Err("adapted-070 reads samples, and this stream is video".to_string());
        };
        OPENED_AS.with(|o| {
            *o.borrow_mut() = format!(
                "{} Hz, {} channel(s), {}",
                audio.sample_rate, audio.channels, audio.sample_fmt
            );
        });
        Ok(())
    }

    fn set_params(_params: String) -> Result<(), String> {
        Ok(())
    }

    fn process(frames: Vec<InFrame>, trailing: Vec<String>, last: bool) -> Processed {
        let opened_as = OPENED_AS.with(|o| o.borrow().clone());
        let arrived: usize = frames.iter().map(|frame| frame.rows.len()).sum();

        let out: Vec<OutFrame> = frames
            .into_iter()
            .map(|frame| {
                let mut rows = vec![format!(
                    r#"{{"opened-as":"{opened_as}","rows-in":{arrived}}}"#
                )];
                rows.extend(frame.rows);
                OutFrame {
                    pts: frame.pts,
                    frame: FramePayload::Same,
                    rows,
                }
            })
            .collect();

        // The trailing rows have no samples to ride, so what they amount to
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

export!(Adapted070);
