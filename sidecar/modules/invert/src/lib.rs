wit_bindgen::generate!({
    path: "../../wit",
    world: "video-module",
});

use exports::ffrwd::av::filter::{FrameInfo, Guest, Meta, Outcome, Output, StreamInfo};

struct Invert;

const PARAMS_SCHEMA: &str = r#"{"type":"object","properties":{},"additionalProperties":false}"#;

/// Validates that `params` is empty or `{}`; invert takes no parameters.
fn validate_params(params: &str) -> Result<(), String> {
    match params.trim() {
        "" | "{}" => Ok(()),
        other => Err(format!("invert takes no params, got: {other}")),
    }
}

impl Guest for Invert {
    fn describe() -> Meta {
        Meta {
            name: "invert".to_string(),
            version: "0.1.0".to_string(),
            params_schema: PARAMS_SCHEMA.to_string(),
            rows_schema: String::new(),
            pixel_formats: vec!["rgba".to_string()],
            // Not an audio module, so it names no sample formats.
            sample_formats: vec![],
            sample_rates: vec![],
            channel_counts: vec![],
            rows_language: vec![],
        }
    }

    fn init(
        _width: u32,
        _height: u32,
        _pix_fmt: String,
        _stream_info: StreamInfo,
        params: String,
    ) -> Result<(), String> {
        validate_params(&params)
    }

    fn set_params(params: String) -> Result<(), String> {
        validate_params(&params)
    }

    fn frame_independent() -> bool {
        true
    }

    fn process(_info: FrameInfo, frame: Vec<u8>) -> Outcome {
        let mut out = frame;
        let (pixels, _) = out.as_chunks_mut::<4>();
        for pixel in pixels {
            pixel[0] = 255 - pixel[0];
            pixel[1] = 255 - pixel[1];
            pixel[2] = 255 - pixel[2];
        }
        Outcome {
            output: Output::Frame(out),
            rows: vec![],
        }
    }
}

export!(Invert);
