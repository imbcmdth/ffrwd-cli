//! A module of the oldest world the host still loads, so the adapter that
//! carries it is exercised by something built the way a module of that world
//! actually was.
//!
//! Frames pass through. Each carries away the time base the host stamped onto
//! the stream tags and the seconds the host converted its timestamp to, which
//! is what that world's per-frame interface is handed in place of a timestamp.

wit_bindgen::generate!({
    path: "../../worlds/0.2.0",
    world: "video-module",
});

use std::cell::RefCell;

use exports::ffrwd::av::filter::{FrameInfo, Guest, Meta, Outcome, Output, StreamInfo};

const PARAMS_SCHEMA: &str = r#"{"type":"object","properties":{},"additionalProperties":false}"#;
const ROWS_SCHEMA: &str = r#"{"type":"object","properties":{"time-base":{"type":"string"},"time":{"type":"number"}},"required":["time-base","time"],"additionalProperties":false}"#;

/// Stream tag naming the unit timestamps are counted in. A world this old has
/// no field for it, so the host stamps this instead.
const TIME_BASE_TAG: &str = "time-base";

thread_local! {
    static TIME_BASE: RefCell<String> = const { RefCell::new(String::new()) };
}

struct Adapted020;

impl Guest for Adapted020 {
    fn describe() -> Meta {
        Meta {
            name: "adapted-020".to_string(),
            version: "0.1.0".to_string(),
            params_schema: PARAMS_SCHEMA.to_string(),
            rows_schema: ROWS_SCHEMA.to_string(),
            pixel_formats: vec!["rgba".to_string()],
        }
    }

    fn init(
        _width: u32,
        _height: u32,
        _pix_fmt: String,
        stream_info: StreamInfo,
        _params: String,
    ) -> Result<(), String> {
        let tag = stream_info
            .tags
            .iter()
            .find(|(key, _)| key == TIME_BASE_TAG)
            .map(|(_, value)| value.clone())
            .ok_or("the stream info carries no time-base tag")?;
        TIME_BASE.with(|t| *t.borrow_mut() = tag);
        Ok(())
    }

    fn set_params(_params: String) -> Result<(), String> {
        Ok(())
    }

    fn frame_independent() -> bool {
        true
    }

    fn process(info: FrameInfo, _frame: Vec<u8>) -> Outcome {
        let time_base = TIME_BASE.with(|t| t.borrow().clone());
        Outcome {
            output: Output::Passthrough,
            rows: vec![format!(
                r#"{{"time-base":"{time_base}","time":{}}}"#,
                info.time
            )],
        }
    }
}

export!(Adapted020);
