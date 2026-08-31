//! Gain: every sample multiplied by `gain`, nothing else touched.
//!
//! # The window
//!
//! Gain is a per-sample operation, so any window would do. 1024 samples is the
//! shape ffmpeg's own pcm packets arrive in, which makes it the least
//! surprising chunk to work in - but nothing lines the two up: the host cuts
//! windows out of whatever packets arrive, so a producer chunking differently
//! changes how many packets a window is spread over and not what leaves here.
//! The window is disjoint from the next, which is what lets a one-to-one
//! module return exactly the samples it was handed.
//!
//! Both wire formats this host carries are handled: f32 is multiplied as it
//! is, s16 is multiplied and clamped, since a gain above one can push a
//! sample past what sixteen bits hold.
//!
//! Rows arriving with the audio ride on out again, and so do trailing rows:
//! nothing here reads them.

wit_bindgen::generate!({
    path: "../../worlds/0.10.0",
    world: "window-module",
});

use std::cell::RefCell;

use exports::ffrwd::av::window_filter::{
    Format, FramePayload, Guest, InFrame, Meta, OutFrame, Processed, StreamInfo, WindowMeta,
};
use serde::Deserialize;

const PARAMS_SCHEMA: &str = r#"{"type":"object","properties":{"gain":{"type":"number","default":1.0}},"additionalProperties":false}"#;

/// Samples one call works in. One ffmpeg pcm packet's worth; see the note
/// above on why any number would serve.
const WINDOW: u32 = 1024;

/// Largest and smallest value sixteen bits hold, as the floats the samples are
/// multiplied as.
const S16_MAX: f32 = 32767.0;
const S16_MIN: f32 = -32768.0;

fn default_gain() -> f64 {
    1.0
}

#[derive(Deserialize)]
struct Params {
    #[serde(default = "default_gain")]
    gain: f64,
}

/// The sample format the host opened this instance for, fixed for its life.
#[derive(Clone, Copy)]
enum SampleFmt {
    F32,
    S16,
}

struct State {
    sample_fmt: SampleFmt,
    gain: f32,
}

thread_local! {
    static STATE: RefCell<Option<State>> = const { RefCell::new(None) };
}

/// Parses and validates params, shared by `init` and `set_params`.
fn parse_params(params: &str) -> Result<f32, String> {
    let trimmed = params.trim();
    let parsed: Params = if trimmed.is_empty() {
        Params {
            gain: default_gain(),
        }
    } else {
        serde_json::from_str(trimmed).map_err(|e| format!("invalid params: {e}"))?
    };
    if !parsed.gain.is_finite() {
        return Err(format!("gain must be a finite number, got {}", parsed.gain));
    }
    Ok(parsed.gain as f32)
}

/// One window's samples with the gain applied, in the format they arrived in.
fn amplify(samples: &[u8], state: &State) -> Vec<u8> {
    match state.sample_fmt {
        SampleFmt::F32 => {
            let (whole, _) = samples.as_chunks::<4>();
            whole
                .iter()
                .flat_map(|bytes| (f32::from_le_bytes(*bytes) * state.gain).to_le_bytes())
                .collect()
        }
        SampleFmt::S16 => {
            let (whole, _) = samples.as_chunks::<2>();
            whole
                .iter()
                .flat_map(|bytes| {
                    let sample = f32::from(i16::from_le_bytes(*bytes));
                    ((sample * state.gain).clamp(S16_MIN, S16_MAX) as i16).to_le_bytes()
                })
                .collect()
        }
    }
}

struct Again;

impl Guest for Again {
    fn describe() -> WindowMeta {
        WindowMeta {
            meta: Meta {
                name: "again".to_string(),
                version: "0.1.0".to_string(),
                params_schema: PARAMS_SCHEMA.to_string(),
                rows_schema: String::new(),
                // An audio module, so it names no pixel formats.
                pixel_formats: vec![],
                sample_formats: vec!["f32".to_string(), "s16".to_string()],
                // Gain is per sample: every rate and every channel count.
                sample_rates: vec![],
                channel_counts: vec![],
                rows_language: vec![],
            },
            window: WINDOW,
            stride: WINDOW,
            // Each call reads only the samples it was handed.
            pure: true,
            // The samples that arrive are the samples that leave.
            one_to_one: true,
            reads_rows: false,
            forwards_rows: true,
            // One stream in, which is every module here.
            inputs: 1,
        }
    }

    fn init(format: Format, _stream_info: StreamInfo, params: String) -> Result<(), String> {
        let gain = parse_params(&params)?;
        let Format::Audio(audio) = format else {
            return Err("again reads samples, and this stream is video".to_string());
        };
        let sample_fmt = match audio.sample_fmt.as_str() {
            "f32" => SampleFmt::F32,
            "s16" => SampleFmt::S16,
            other => return Err(format!("unsupported sample-fmt: {other:?}")),
        };
        STATE.with(|s| *s.borrow_mut() = Some(State { sample_fmt, gain }));
        Ok(())
    }

    fn set_params(params: String) -> Result<(), String> {
        let gain = parse_params(&params)?;
        STATE.with(|s| {
            let mut state_ref = s.borrow_mut();
            let state = state_ref.as_mut().expect("set_params called before init");
            state.gain = gain;
        });
        Ok(())
    }

    fn process(frames: Vec<InFrame>, trailing: Vec<String>, _last: bool) -> Processed {
        STATE.with(|s| {
            let state_ref = s.borrow();
            let state = state_ref.as_ref().expect("process called before init");

            let out = frames
                .into_iter()
                .map(|frame| OutFrame {
                    pts: frame.pts,
                    frame: FramePayload::New(amplify(&frame.frame, state)),
                    rows: frame.rows,
                })
                .collect();
            Processed {
                frames: out,
                trailing,
            }
        })
    }
}

export!(Again);

#[cfg(test)]
mod tests {
    use super::*;

    fn f32_bytes(samples: &[f32]) -> Vec<u8> {
        samples.iter().flat_map(|s| s.to_le_bytes()).collect()
    }

    fn f32_values(bytes: &[u8]) -> Vec<f32> {
        let (whole, _) = bytes.as_chunks::<4>();
        whole.iter().copied().map(f32::from_le_bytes).collect()
    }

    fn s16_bytes(samples: &[i16]) -> Vec<u8> {
        samples.iter().flat_map(|s| s.to_le_bytes()).collect()
    }

    fn s16_values(bytes: &[u8]) -> Vec<i16> {
        let (whole, _) = bytes.as_chunks::<2>();
        whole.iter().copied().map(i16::from_le_bytes).collect()
    }

    #[test]
    fn half_gain_halves_every_float_sample() {
        let state = State {
            sample_fmt: SampleFmt::F32,
            gain: 0.5,
        };
        let out = amplify(&f32_bytes(&[1.0, -0.5, 0.25, 0.0]), &state);
        assert_eq!(f32_values(&out), vec![0.5, -0.25, 0.125, 0.0]);
        assert_eq!(
            out.len(),
            16,
            "the samples that arrive are the ones that go"
        );
    }

    #[test]
    fn a_gain_past_what_sixteen_bits_hold_clamps_rather_than_wrapping() {
        let state = State {
            sample_fmt: SampleFmt::S16,
            gain: 4.0,
        };
        let out = amplify(&s16_bytes(&[10_000, -10_000, 100]), &state);
        assert_eq!(
            s16_values(&out),
            vec![32_767, -32_768, 400],
            "a sample driven past the range sits at the end of it"
        );
    }

    #[test]
    fn a_gain_of_one_leaves_the_bytes_alone() {
        let state = State {
            sample_fmt: SampleFmt::S16,
            gain: 1.0,
        };
        let samples = s16_bytes(&[0, 1, -1, 32_767, -32_768]);
        assert_eq!(amplify(&samples, &state), samples);
    }

    #[test]
    fn no_params_is_unity_gain_and_a_gain_that_is_not_a_number_is_refused() {
        assert_eq!(parse_params("").expect("no params is the default"), 1.0);
        assert_eq!(
            parse_params(r#"{"gain":0.5}"#).expect("a number is a gain"),
            0.5
        );
        let Err(err) = parse_params(r#"{"gain":"loud"}"#) else {
            panic!("a gain that is not a number should have been refused");
        };
        assert!(err.contains("invalid params"), "got: {err}");
    }
}
