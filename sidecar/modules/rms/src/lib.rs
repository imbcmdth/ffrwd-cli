//! How loud each third of a second was: one row per window naming the root
//! mean square of the samples in it, with the audio itself passed through
//! untouched.
//!
//! # The window
//!
//! 16000 samples, and the stride is the window, so the windows are disjoint
//! and tile the stream: a level is measured once over each stretch and never
//! twice over the same one. That is also what lets the audio pass through as
//! the very samples that arrived - overlapping windows could not, since every
//! sample would leave more than once.
//!
//! The window is many packets' worth whatever the producer chose, which is
//! the point: the host buffers the arrivals and hands this module one
//! contiguous piece, so a level is measured over a stretch of time rather than
//! over whatever ffmpeg happened to cut.
//!
//! The final call carries whatever the last stride left over. A stream whose
//! length is a multiple of the window ends with a call of no samples, and
//! there is no level to report over nothing, so it emits no row: the rows are
//! one per window and the windows are `ceil(samples / stride)` of them.
//!
//! Rows arriving with the audio stop here; what leaves is this module's own.

wit_bindgen::generate!({
    path: "../../wit",
    world: "window-module",
});

use std::cell::RefCell;

use exports::ffrwd::av::window_filter::{
    Format, FramePayload, Guest, InFrame, Meta, OutFrame, Processed, StreamInfo, WindowMeta,
};
use serde::Serialize;

const PARAMS_SCHEMA: &str = r#"{"type":"object","properties":{},"additionalProperties":false}"#;
const ROWS_SCHEMA: &str = r#"{"type":"object","properties":{"pts":{"type":"integer"},"samples":{"type":"integer"},"rms":{"type":"number"}},"required":["pts","samples","rms"],"additionalProperties":false}"#;

/// Samples one window covers. A third of a second at 48 kHz.
const WINDOW: u32 = 16_000;

#[derive(Serialize)]
struct Row {
    /// Where the window starts, in ticks of the stream's time base.
    pts: i64,
    /// How many samples it covers, which with `pts` is its span.
    samples: usize,
    rms: f64,
}

struct State {
    channels: usize,
}

thread_local! {
    static STATE: RefCell<Option<State>> = const { RefCell::new(None) };
}

/// Validates that `params` is empty or `{}`; rms takes no parameters.
fn validate_params(params: &str) -> Result<(), String> {
    match params.trim() {
        "" | "{}" => Ok(()),
        other => Err(format!("rms takes no params, got: {other}")),
    }
}

/// How many running totals the sum of squares is spread over. Four lanes that
/// never touch until the end, so the adds are independent instead of one
/// chain; the totals stay f64, which is the width the level is measured in.
const LANES: usize = 4;

/// Root mean square of interleaved f32 samples, every channel counted alike.
fn root_mean_square(samples: &[u8]) -> f64 {
    let (whole, _) = samples.as_chunks::<4>();
    if whole.is_empty() {
        return 0.0;
    }
    let mut totals = [0.0f64; LANES];
    let (groups, rest) = whole.as_chunks::<LANES>();
    for group in groups {
        for (total, bytes) in totals.iter_mut().zip(group) {
            let sample = f64::from(f32::from_le_bytes(*bytes));
            *total += sample * sample;
        }
    }
    for bytes in rest {
        let sample = f64::from(f32::from_le_bytes(*bytes));
        totals[0] += sample * sample;
    }
    let total: f64 = totals.iter().sum();
    (total / whole.len() as f64).sqrt()
}

struct Rms;

impl Guest for Rms {
    fn describe() -> WindowMeta {
        WindowMeta {
            meta: Meta {
                name: "rms".to_string(),
                version: "0.1.0".to_string(),
                params_schema: PARAMS_SCHEMA.to_string(),
                rows_schema: ROWS_SCHEMA.to_string(),
                // An audio module, so it names no pixel formats.
                pixel_formats: vec![],
                sample_formats: vec!["f32".to_string()],
                // A level is a level at any rate and any channel count.
                sample_rates: vec![],
                channel_counts: vec![],
                rows_language: vec![],
            },
            window: WINDOW,
            stride: WINDOW,
            // Each call reads only the samples it was handed.
            pure: true,
            // The samples pass through as they arrived.
            one_to_one: true,
            reads_rows: false,
            // What leaves is this module's own level and nothing else.
            forwards_rows: false,
            // One stream in, which is every module here.
            inputs: 1,
        }
    }

    fn init(format: Format, _stream_info: StreamInfo, params: String) -> Result<(), String> {
        validate_params(&params)?;
        let Format::Audio(audio) = format else {
            return Err("rms reads samples, and this stream is video".to_string());
        };
        if audio.sample_fmt != "f32" {
            return Err(format!("unsupported sample-fmt: {:?}", audio.sample_fmt));
        }
        STATE.with(|s| {
            *s.borrow_mut() = Some(State {
                channels: audio.channels as usize,
            });
        });
        Ok(())
    }

    fn set_params(params: String) -> Result<(), String> {
        validate_params(&params)
    }

    fn process(frames: Vec<InFrame>, _trailing: Vec<String>, _last: bool) -> Processed {
        STATE.with(|s| {
            let state_ref = s.borrow();
            let state = state_ref.as_ref().expect("process called before init");

            let out = frames
                .into_iter()
                .filter(|frame| !frame.frame.is_empty())
                .map(|frame| {
                    let row = Row {
                        pts: frame.pts,
                        samples: frame.frame.len() / (4 * state.channels),
                        rms: root_mean_square(&frame.frame),
                    };
                    OutFrame {
                        pts: frame.pts,
                        frame: FramePayload::Same,
                        rows: vec![serde_json::to_string(&row).expect("row serializes")],
                    }
                })
                .collect();
            Processed {
                frames: out,
                trailing: Vec::new(),
            }
        })
    }
}

export!(Rms);

#[cfg(test)]
mod tests {
    use super::*;

    fn bytes(samples: &[f32]) -> Vec<u8> {
        samples.iter().flat_map(|s| s.to_le_bytes()).collect()
    }

    #[test]
    fn a_flat_level_is_its_own_root_mean_square() {
        // Every sample the same size: the mean of the squares is that square,
        // and its root is the size back.
        assert!((root_mean_square(&bytes(&[0.5, -0.5, 0.5, -0.5])) - 0.5).abs() < 1e-9);
        assert_eq!(root_mean_square(&bytes(&[0.0, 0.0])), 0.0);
    }

    #[test]
    fn halving_every_sample_halves_the_level() {
        let loud = root_mean_square(&bytes(&[1.0, 0.5, -0.25, 0.75]));
        let quiet = root_mean_square(&bytes(&[0.5, 0.25, -0.125, 0.375]));
        assert!(
            (loud / 2.0 - quiet).abs() < 1e-9,
            "{loud} halved is not {quiet}"
        );
    }

    #[test]
    fn a_full_scale_sine_measures_the_root_of_a_half() {
        // The textbook figure, and what the integration test leans on: a sine
        // at amplitude one has an rms of 1/sqrt(2).
        let samples: Vec<f32> = (0..4_800)
            .map(|i| (std::f64::consts::TAU * f64::from(i) / 100.0).sin() as f32)
            .collect();
        let measured = root_mean_square(&bytes(&samples));
        assert!(
            (measured - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-4,
            "a full-scale sine measured {measured}"
        );
    }

    #[test]
    fn nothing_at_all_measures_nothing() {
        assert_eq!(root_mean_square(&[]), 0.0);
    }
}
