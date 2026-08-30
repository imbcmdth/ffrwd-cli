//! A module of the world that first let rows reach a filter, so the adapter
//! carrying `meta-filter` beside `filter` is exercised.
//!
//! Frames pass through. Each carries away how many rows arrived with it,
//! which is what proves the rows crossed the adapter.

wit_bindgen::generate!({
    path: "../../worlds/0.4.0",
    world: "meta-module",
});

use exports::ffrwd::av::filter::{FrameInfo, Guest, Meta, Outcome, Output, StreamInfo};
use exports::ffrwd::av::meta_filter::Guest as MetaGuest;

const PARAMS_SCHEMA: &str = r#"{"type":"object","properties":{},"additionalProperties":false}"#;
const ROWS_SCHEMA: &str = r#"{"type":"object","properties":{"rows-in":{"type":"integer"}},"required":["rows-in"],"additionalProperties":false}"#;

struct Adapted040;

impl Guest for Adapted040 {
    fn describe() -> Meta {
        Meta {
            name: "adapted-040".to_string(),
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
        _stream_info: StreamInfo,
        _params: String,
    ) -> Result<(), String> {
        Ok(())
    }

    fn set_params(_params: String) -> Result<(), String> {
        Ok(())
    }

    fn frame_independent() -> bool {
        true
    }

    /// Never reached: the host calls `process-meta` on a module exporting it.
    fn process(_info: FrameInfo, _frame: Vec<u8>) -> Outcome {
        Outcome {
            output: Output::Passthrough,
            rows: vec![],
        }
    }
}

impl MetaGuest for Adapted040 {
    fn process_meta(_info: FrameInfo, _frame: Vec<u8>, rows_in: Vec<String>) -> Outcome {
        Outcome {
            output: Output::Passthrough,
            rows: vec![format!(r#"{{"rows-in":{}}}"#, rows_in.len())],
        }
    }
}

export!(Adapted040);
