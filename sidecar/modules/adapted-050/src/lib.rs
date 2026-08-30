//! A windowed module of the world before trailing rows and forwarding
//! declarations, so the adapter carrying that world is exercised.
//!
//! Frames pass through, two at a time. Each window's first frame carries away
//! the time base the host stamped onto the stream tags - which is how a
//! module of that world learns what its timestamps are counted in - and the
//! rows that arrived travel on beside it, which is what the host assumed of
//! every windowed module then.

wit_bindgen::generate!({
    path: "../../worlds/0.5.0",
    world: "window-module",
});

use std::cell::RefCell;

use exports::ffrwd::av::window_filter::{
    FramePayload, Guest, InFrame, Meta, OutFrame, StreamInfo, WindowMeta,
};

const PARAMS_SCHEMA: &str = r#"{"type":"object","properties":{},"additionalProperties":false}"#;
const ROWS_SCHEMA: &str = r#"{"type":"object","properties":{"time-base":{"type":"string"},"rows-in":{"type":"integer"}},"required":["time-base","rows-in"],"additionalProperties":false}"#;

/// Stream tag naming the unit timestamps are counted in. This world has no
/// field for it, so the host stamps this instead.
const TIME_BASE_TAG: &str = "time-base";

thread_local! {
    static TIME_BASE: RefCell<String> = const { RefCell::new(String::new()) };
}

struct Adapted050;

impl Guest for Adapted050 {
    fn describe() -> WindowMeta {
        WindowMeta {
            meta: Meta {
                name: "adapted-050".to_string(),
                version: "0.1.0".to_string(),
                params_schema: PARAMS_SCHEMA.to_string(),
                rows_schema: ROWS_SCHEMA.to_string(),
                pixel_formats: vec!["rgba".to_string()],
            },
            window: 2,
            stride: 2,
            pure: true,
            one_to_one: true,
            reads_rows: true,
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

    fn process(frames: Vec<InFrame>, _last: bool) -> Vec<OutFrame> {
        let time_base = TIME_BASE.with(|t| t.borrow().clone());
        let arrived: usize = frames.iter().map(|frame| frame.rows.len()).sum();
        let mut call_rows = vec![format!(
            r#"{{"time-base":"{time_base}","rows-in":{arrived}}}"#
        )];

        frames
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
            .collect()
    }
}

export!(Adapted050);
