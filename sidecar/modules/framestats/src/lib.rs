//! Mean brightness per frame, and how many frames there were.
//!
//! Frames pass through untouched. Each carries away a row with its time and
//! its mean; the count has no frame to ride, since it is not known until the
//! last one has gone by, so it leaves as a trailing row instead.

wit_bindgen::generate!({
    path: "../../worlds/0.10.0",
    world: "window-module",
});

use std::cell::RefCell;

use exports::ffrwd::av::window_filter::{
    Format, FramePayload, Guest, InFrame, Meta, OutFrame, Processed, StreamInfo, WindowMeta,
};
use serde::Serialize;

const PARAMS_SCHEMA: &str = r#"{"type":"object","properties":{},"additionalProperties":false}"#;
/// Two shapes: the per-frame stats, and the count the stream ends with.
const ROWS_SCHEMA: &str = r#"{"oneOf":[{"type":"object","properties":{"time":{"type":"number"},"mean":{"type":"integer"}},"required":["time","mean"],"additionalProperties":false},{"type":"object","properties":{"frames":{"type":"integer"}},"required":["frames"],"additionalProperties":false}]}"#;

#[derive(Serialize)]
struct Row {
    time: f64,
    mean: u64,
}

#[derive(Serialize)]
struct Summary {
    frames: u64,
}

struct State {
    width: u64,
    height: u64,
    /// The time base, kept as it arrived so a timestamp becomes seconds the
    /// same way the host's own conversion does.
    num: f64,
    den: f64,
    frames: u64,
}

thread_local! {
    static STATE: RefCell<Option<State>> = const { RefCell::new(None) };
}

/// Validates that `params` is empty or `{}`; framestats takes no parameters.
fn validate_params(params: &str) -> Result<(), String> {
    match params.trim() {
        "" | "{}" => Ok(()),
        other => Err(format!("framestats takes no params, got: {other}")),
    }
}

/// A timestamp in seconds.
fn seconds(pts: i64, num: f64, den: f64) -> f64 {
    pts as f64 * num / den
}

/// The mean of one frame's colour bytes, alpha left out.
fn mean(frame: &[u8], width: u64, height: u64) -> u64 {
    let mut sum: u64 = 0;
    let (pixels, _) = frame.as_chunks::<4>();
    for pixel in pixels {
        sum += pixel[0] as u64 + pixel[1] as u64 + pixel[2] as u64;
    }
    sum / (width * height * 3)
}

struct Framestats;

impl Guest for Framestats {
    fn describe() -> WindowMeta {
        WindowMeta {
            meta: Meta {
                name: "framestats".to_string(),
                version: "0.1.0".to_string(),
                params_schema: PARAMS_SCHEMA.to_string(),
                rows_schema: ROWS_SCHEMA.to_string(),
                pixel_formats: vec!["rgba".to_string()],
                // Not an audio module, so it names no sample formats.
                sample_formats: vec![],
                sample_rates: vec![],
                channel_counts: vec![],
                rows_language: vec![],
            },
            window: 1,
            stride: 1,
            // The running count carries over between calls.
            pure: false,
            one_to_one: true,
            reads_rows: false,
            // The rows leaving are this module's own.
            forwards_rows: false,
            // One stream in, which is every module here.
            inputs: 1,
        }
    }

    fn init(format: Format, stream_info: StreamInfo, params: String) -> Result<(), String> {
        validate_params(&params)?;
        let Format::Video(video) = format else {
            return Err("framestats reads pictures, and this stream is audio".to_string());
        };
        let (width, height) = (video.width, video.height);
        STATE.with(|s| {
            *s.borrow_mut() = Some(State {
                width: u64::from(width),
                height: u64::from(height),
                num: f64::from(stream_info.time_base.num),
                den: f64::from(stream_info.time_base.den),
                frames: 0,
            });
        });
        Ok(())
    }

    fn set_params(params: String) -> Result<(), String> {
        validate_params(&params)
    }

    fn process(frames: Vec<InFrame>, _trailing: Vec<String>, last: bool) -> Processed {
        STATE.with(|s| {
            let mut state_ref = s.borrow_mut();
            let state = state_ref.as_mut().expect("process called before init");

            let out: Vec<OutFrame> = frames
                .into_iter()
                .map(|frame| {
                    state.frames += 1;
                    let row = Row {
                        time: seconds(frame.pts, state.num, state.den),
                        mean: mean(&frame.frame, state.width, state.height),
                    };
                    OutFrame {
                        pts: frame.pts,
                        frame: FramePayload::Same,
                        rows: vec![serde_json::to_string(&row).expect("row serializes")],
                    }
                })
                .collect();

            let trailing = if last {
                let summary = Summary {
                    frames: state.frames,
                };
                vec![serde_json::to_string(&summary).expect("row serializes")]
            } else {
                Vec::new()
            };

            Processed {
                frames: out,
                trailing,
            }
        })
    }
}

export!(Framestats);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_timestamp_becomes_seconds_through_the_time_base() {
        assert!((seconds(25, 1.0, 25.0) - 1.0).abs() < 1e-12);
        assert!((seconds(65536, 1.0, 65536.0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn the_mean_leaves_the_alpha_byte_out() {
        // Opaque mid-grey: the colour bytes average to their own level
        // whatever the alpha byte holds.
        let frame: Vec<u8> = [90u8, 90, 90, 0].repeat(4 * 4);
        assert_eq!(mean(&frame, 4, 4), 90);
    }
}
