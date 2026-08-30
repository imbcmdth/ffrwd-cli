//! A windowed module of the world before the format became kind-bearing, so
//! the adapter carrying that world is exercised.
//!
//! Frames pass through, two at a time. Each window's first frame carries away
//! the time base - which that world hands a module as a field of the stream
//! info rather than as a stamped-on tag - and how many rows arrived with the
//! window. Rows that arrived travel on beside it, and the trailing rows the
//! stream ends with are counted into one row of this module's own: that pair
//! is what the world added, and what its adapter has to keep working.

wit_bindgen::generate!({
    path: "../../worlds/0.6.0",
    world: "window-module",
});

use std::cell::RefCell;

use exports::ffrwd::av::window_filter::{
    FramePayload, Guest, InFrame, Meta, OutFrame, Processed, StreamInfo, WindowMeta,
};

const PARAMS_SCHEMA: &str = r#"{"type":"object","properties":{},"additionalProperties":false}"#;
const ROWS_SCHEMA: &str = r#"{"type":"object","properties":{"time-base":{"type":"string"},"rows-in":{"type":"integer"},"trailing-in":{"type":"integer"}},"additionalProperties":false}"#;

thread_local! {
    static TIME_BASE: RefCell<String> = const { RefCell::new(String::new()) };
}

struct Adapted060;

impl Guest for Adapted060 {
    fn describe() -> WindowMeta {
        WindowMeta {
            meta: Meta {
                name: "adapted-060".to_string(),
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
            forwards_rows: true,
        }
    }

    fn init(
        _width: u32,
        _height: u32,
        _pix_fmt: String,
        stream_info: StreamInfo,
        _params: String,
    ) -> Result<(), String> {
        let base = stream_info.time_base;
        if base.den <= 0 {
            return Err(format!(
                "the time base {}/{} is not a rate",
                base.num, base.den
            ));
        }
        TIME_BASE.with(|t| *t.borrow_mut() = format!("{}/{}", base.num, base.den));
        Ok(())
    }

    fn set_params(_params: String) -> Result<(), String> {
        Ok(())
    }

    fn process(frames: Vec<InFrame>, trailing: Vec<String>, last: bool) -> Processed {
        let time_base = TIME_BASE.with(|t| t.borrow().clone());
        let arrived: usize = frames.iter().map(|frame| frame.rows.len()).sum();
        let mut call_rows = vec![format!(
            r#"{{"time-base":"{time_base}","rows-in":{arrived}}}"#
        )];

        let out: Vec<OutFrame> = frames
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

export!(Adapted060);
