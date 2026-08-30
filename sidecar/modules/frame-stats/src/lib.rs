//! A sink: consumes video frames and writes a stats line per frame to
//! stderr. Nothing comes back — the module emits no frames and no rows, so
//! its output pad carries a null output. The lines are its whole product.

wit_bindgen::generate!({
    path: "../../wit",
    world: "window-module",
});

use std::cell::RefCell;

use exports::ffrwd::av::window_filter::{
    Format, Guest, InFrame, Meta, Processed, StreamInfo, WindowMeta,
};

const PARAMS_SCHEMA: &str = r#"{"type":"object","properties":{},"additionalProperties":false}"#;

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

/// Validates that `params` is empty or `{}`; frame_stats takes no parameters.
fn validate_params(params: &str) -> Result<(), String> {
    match params.trim() {
        "" | "{}" => Ok(()),
        other => Err(format!("frame_stats takes no params, got: {other}")),
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

struct FrameStats;

impl Guest for FrameStats {
    fn describe() -> WindowMeta {
        WindowMeta {
            meta: Meta {
                name: "frame_stats".to_string(),
                version: "0.1.0".to_string(),
                params_schema: PARAMS_SCHEMA.to_string(),
                // No rows leave: the stderr lines are the output.
                rows_schema: String::new(),
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
            // A sink: frames go in and none come out.
            one_to_one: false,
            reads_rows: false,
            forwards_rows: false,
            // One stream in, which is every module here.
            inputs: 1,
        }
    }

    fn init(format: Format, stream_info: StreamInfo, params: String) -> Result<(), String> {
        validate_params(&params)?;
        let Format::Video(video) = format else {
            return Err("frame_stats reads pictures, and this stream is audio".to_string());
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

            for frame in &frames {
                eprintln!(
                    "frame_stats frame={} time={:.3} mean={}",
                    state.frames,
                    seconds(frame.pts, state.num, state.den),
                    mean(&frame.frame, state.width, state.height),
                );
                state.frames += 1;
            }
            if last {
                eprintln!("frame_stats frames={}", state.frames);
            }

            Processed {
                frames: Vec::new(),
                trailing: Vec::new(),
            }
        })
    }
}

export!(FrameStats);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_timestamp_becomes_seconds_through_the_time_base() {
        assert!((seconds(25, 1.0, 25.0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn the_mean_leaves_the_alpha_byte_out() {
        let frame: Vec<u8> = [90u8, 90, 90, 0].repeat(4 * 4);
        assert_eq!(mean(&frame, 4, 4), 90);
    }
}
