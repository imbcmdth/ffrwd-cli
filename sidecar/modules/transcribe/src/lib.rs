//! What was said, and when: whisper base over one thirty-second window at a
//! time, with the audio passed through untouched.
//!
//! # The window
//!
//! 480000 samples - thirty seconds at the 16 kHz the model was trained on -
//! and the stride is the window, so the windows are disjoint and tile the
//! stream. Every stretch of audio is transcribed once and never twice. That is
//! also what lets the samples pass through as the very ones that arrived.
//!
//! The host conforms the stream to what this module publishes: 16 kHz, one
//! channel, f32. A window arrives whole, whatever packets the producer cut, so
//! the model never sees a boundary that is not its own.
//!
//! # The languages
//!
//! `language` says what is spoken and is required; the model is multilingual
//! and does not guess. `language_to` says what the rows come out in, and only
//! English is producible, so setting it to anything else is refused. Set to a
//! language other than the spoken one it makes whisper translate; otherwise
//! whisper transcribes.
//!
//! # The rows
//!
//! One row per decoded segment - `start_t`, `end_t`, `text` - in seconds from
//! the start of the STREAM, not the window: the window's own base comes from
//! its timestamp and the stream's time base, and whisper's within-window times
//! are added to it. A window the model hears as silence emits no rows at all.
//!
//! The final call adds one trailing row carrying the whole transcript, the
//! segments of every window joined in order.
//!
//! # The model
//!
//! The weights are compiled in, and they are too big for git. A module built
//! without them loads and describes itself, and refuses at init: see
//! `fetch-model.ps1`. `model/melfilters.bytes` is the mel filterbank whisper
//! was trained against, a fixed table of 80 by 201 floats taken from candle
//! (MIT/Apache-2.0); it is small, so it is in git with the source.

#![cfg_attr(not(have_model), allow(dead_code))]

wit_bindgen::generate!({
    path: "../../wit",
    world: "window-module",
});

mod mel;
mod whisper;

use std::cell::RefCell;

use exports::ffrwd::av::window_filter::{
    Format, FramePayload, Guest, InFrame, Meta, OutFrame, Processed, StreamInfo, WindowMeta,
};
use serde::{Deserialize, Serialize};
use whisper::{Task, Transcriber};

const PARAMS_SCHEMA: &str = r#"{"type":"object","properties":{"language":{"type":"string","minLength":2,"maxLength":3},"language_to":{"enum":["en"]}},"required":["language"],"additionalProperties":false}"#;
const ROWS_SCHEMA: &str = r#"{"oneOf":[{"type":"object","properties":{"start_t":{"type":"number"},"end_t":{"type":"number"},"text":{"type":"string"}},"required":["start_t","end_t","text"],"additionalProperties":false},{"type":"object","properties":{"transcript":{"type":"string"}},"required":["transcript"],"additionalProperties":false}]}"#;

/// Samples one window covers: thirty seconds at 16 kHz, which is the stretch
/// whisper was trained to hear at once.
const WINDOW: u32 = 480_000;

/// The rate the model works at, and the only one this module accepts.
const SAMPLE_RATE: u32 = 16_000;

/// The whisper files compiled into the module, absent when it was built
/// without them.
struct Files {
    weights: &'static [u8],
    tokenizer: &'static [u8],
    config: &'static [u8],
    melfilters: &'static [u8],
}

#[cfg(have_model)]
const FILES: Option<Files> = Some(Files {
    weights: include_bytes!(env!("TRANSCRIBE_WEIGHTS")),
    tokenizer: include_bytes!(env!("TRANSCRIBE_TOKENIZER")),
    config: include_bytes!(env!("TRANSCRIBE_CONFIG")),
    melfilters: include_bytes!(env!("TRANSCRIBE_MELFILTERS")),
});

#[cfg(not(have_model))]
const FILES: Option<Files> = None;

/// What is spoken, and what the rows come out in.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Params {
    language: String,
    #[serde(default)]
    language_to: Option<String>,
}

impl Params {
    /// Whisper's job here: translating when the rows are asked for in another
    /// language than the one spoken.
    fn task(&self) -> Task {
        match &self.language_to {
            Some(to) if *to != self.language => Task::Translate,
            _ => Task::Transcribe,
        }
    }
}

/// One decoded segment, in seconds from the start of the stream.
#[derive(Serialize)]
struct Row {
    start_t: f64,
    end_t: f64,
    text: String,
}

/// Everything said over the stream, on the final call.
#[derive(Serialize)]
struct Transcript {
    transcript: String,
}

struct State {
    transcriber: Transcriber,
    /// Seconds one timestamp tick is worth, from the stream's time base.
    tick: f64,
    /// Every segment's text so far, in the order the windows arrived.
    said: Vec<String>,
}

thread_local! {
    static STATE: RefCell<Option<State>> = const { RefCell::new(None) };
}

/// Parses and validates params, shared by `init` and `set_params`.
fn parse_params(params: &str) -> Result<Params, String> {
    let trimmed = params.trim();
    if trimmed.is_empty() {
        return Err("transcribe needs a language, and none was given".to_string());
    }
    let parsed: Params =
        serde_json::from_str(trimmed).map_err(|e| format!("transcribe: invalid params: {e}"))?;
    if !whisper::is_language(&parsed.language) {
        return Err(format!(
            "transcribe: whisper does not know the language {}",
            parsed.language
        ));
    }
    if let Some(to) = &parsed.language_to {
        if to != "en" {
            return Err(format!(
                "transcribe translates into English alone, and language_to is {to}"
            ));
        }
    }
    Ok(parsed)
}

/// Interleaved little-endian f32 samples as floats. One channel, so a sample
/// is a value.
fn samples(bytes: &[u8]) -> Vec<f32> {
    let (whole, _) = bytes.as_chunks::<4>();
    whole.iter().map(|b| f32::from_le_bytes(*b)).collect()
}

/// Seconds one timestamp tick is worth, which is what puts a window's segments
/// where they belong in the stream.
fn seconds_per_tick(time_base: ffrwd::av::types::Rational) -> f64 {
    f64::from(time_base.num) / f64::from(time_base.den)
}

struct Transcribe;

impl Guest for Transcribe {
    fn describe() -> WindowMeta {
        WindowMeta {
            meta: Meta {
                name: "transcribe".to_string(),
                version: "0.1.0".to_string(),
                params_schema: PARAMS_SCHEMA.to_string(),
                rows_schema: ROWS_SCHEMA.to_string(),
                // An audio module, so it names no pixel formats.
                pixel_formats: vec![],
                sample_formats: vec!["f32".to_string()],
                // What the model was trained on, and the host conforms to it.
                sample_rates: vec![SAMPLE_RATE],
                channel_counts: vec![1],
                // With `language_to` set the rows come out in it; otherwise in
                // the language spoken.
                rows_language: vec!["language_to".to_string(), "language".to_string()],
            },
            window: WINDOW,
            stride: WINDOW,
            // The transcript the final call carries gathers every window, so
            // a call is not answerable from what it was handed alone. The
            // decoding itself carries nothing over.
            pure: false,
            // The samples pass through as they arrived.
            one_to_one: true,
            reads_rows: false,
            // What leaves is this module's own words and nothing else.
            forwards_rows: false,
            // One stream in, which is every module here.
            inputs: 1,
        }
    }

    fn init(format: Format, stream_info: StreamInfo, params: String) -> Result<(), String> {
        let parsed = parse_params(&params)?;
        let Format::Audio(audio) = format else {
            return Err("transcribe listens, and this stream is video".to_string());
        };
        if audio.sample_fmt != "f32" {
            return Err(format!("unsupported sample-fmt: {:?}", audio.sample_fmt));
        }
        if audio.sample_rate != SAMPLE_RATE || audio.channels != 1 {
            return Err(format!(
                "transcribe hears 16000 Hz mono, and this stream is {} Hz across {} channel(s)",
                audio.sample_rate, audio.channels
            ));
        }
        let Some(files) = FILES else {
            return Err(
                "transcribe was built without its whisper model; run fetch-model.ps1 beside the \
                 module and build it again"
                    .to_string(),
            );
        };

        let transcriber = Transcriber::load(
            files.weights,
            files.tokenizer,
            files.config,
            files.melfilters,
            &parsed.language,
            parsed.task(),
        )?;
        STATE.with(|s| {
            *s.borrow_mut() = Some(State {
                transcriber,
                tick: seconds_per_tick(stream_info.time_base),
                said: Vec::new(),
            });
        });
        Ok(())
    }

    fn set_params(params: String) -> Result<(), String> {
        let parsed = parse_params(&params)?;
        STATE.with(|s| match s.borrow_mut().as_mut() {
            Some(state) => state.transcriber.retarget(&parsed.language, parsed.task()),
            None => Ok(()),
        })
    }

    fn process(frames: Vec<InFrame>, _trailing: Vec<String>, last: bool) -> Processed {
        STATE.with(|s| {
            let mut state_ref = s.borrow_mut();
            let state = state_ref.as_mut().expect("process called before init");

            let mut out = Vec::with_capacity(frames.len());
            for frame in frames {
                if frame.frame.is_empty() {
                    continue;
                }
                // Where this window starts in the stream, which its own
                // segment times are measured from.
                let base = frame.pts as f64 * state.tick;
                let pcm = samples(&frame.frame);

                let rows = match state.transcriber.window(&pcm) {
                    Ok(segments) => segments
                        .into_iter()
                        .map(|segment| {
                            state.said.push(segment.text.clone());
                            let row = Row {
                                start_t: base + segment.start,
                                end_t: base + segment.end,
                                text: segment.text,
                            };
                            serde_json::to_string(&row).expect("row serializes")
                        })
                        .collect(),
                    // A window that will not decode is reported and the audio
                    // still goes through; the stream is not lost over it.
                    Err(message) => {
                        eprintln!("{message}");
                        Vec::new()
                    }
                };

                out.push(OutFrame {
                    pts: frame.pts,
                    frame: FramePayload::Same,
                    rows,
                });
            }

            let trailing = if last {
                let transcript = Transcript {
                    transcript: state.said.join(" "),
                };
                vec![serde_json::to_string(&transcript).expect("row serializes")]
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

export!(Transcribe);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_window_is_thirty_seconds_of_the_rate_the_model_hears() {
        assert_eq!(WINDOW, 30 * SAMPLE_RATE);
    }

    /// The error a refusal produced, or a panic naming what should have been
    /// refused.
    fn refusal(params: &str) -> String {
        match parse_params(params) {
            Err(err) => err,
            Ok(_) => panic!("{params} should have been refused"),
        }
    }

    #[test]
    fn the_spoken_language_is_required() {
        assert_eq!(
            refusal(""),
            "transcribe needs a language, and none was given"
        );
        assert!(refusal("{}").contains("language"));
        assert_eq!(parse_params(r#"{"language":"es"}"#).unwrap().language, "es");
    }

    #[test]
    fn a_language_whisper_does_not_know_is_refused_by_name() {
        let err = refusal(r#"{"language":"klingon"}"#);
        assert!(err.contains("transcribe"), "got: {err}");
        assert!(err.contains("klingon"), "got: {err}");
    }

    #[test]
    fn english_is_the_only_language_whisper_can_produce() {
        assert_eq!(
            parse_params(r#"{"language":"es","language_to":"en"}"#)
                .unwrap()
                .language_to
                .as_deref(),
            Some("en")
        );
        let err = refusal(r#"{"language":"es","language_to":"fr"}"#);
        assert!(err.contains("transcribe"), "got: {err}");
        assert!(err.contains("fr"), "got: {err}");
    }

    #[test]
    fn a_param_the_module_does_not_take_is_refused() {
        let err = refusal(r#"{"language":"es","task":"translate"}"#);
        assert!(err.contains("transcribe"), "got: {err}");
    }

    #[test]
    fn the_task_follows_from_the_two_languages() {
        let task = |params: &str| parse_params(params).unwrap().task();
        assert_eq!(task(r#"{"language":"es"}"#), Task::Transcribe);
        assert_eq!(
            task(r#"{"language":"en","language_to":"en"}"#),
            Task::Transcribe
        );
        assert_eq!(
            task(r#"{"language":"es","language_to":"en"}"#),
            Task::Translate
        );
    }

    #[test]
    fn the_declared_row_languages_are_the_two_params_most_specific_first() {
        let meta = Transcribe::describe().meta;
        assert_eq!(meta.rows_language, vec!["language_to", "language"]);
        for name in &meta.rows_language {
            assert!(
                meta.params_schema.contains(&format!("\"{name}\"")),
                "{name} is declared but not in the params schema"
            );
        }
    }

    #[test]
    fn samples_read_back_as_little_endian_floats() {
        let bytes: Vec<u8> = [0.5f32, -0.25]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        assert_eq!(samples(&bytes), vec![0.5, -0.25]);
        assert_eq!(samples(&[]), Vec::<f32>::new());
    }

    #[test]
    fn the_time_base_field_becomes_seconds_per_tick() {
        use ffrwd::av::types::Rational;
        // At the natural base for audio one tick is one sample, so a window's
        // worth of ticks is a window's worth of seconds.
        let tick = seconds_per_tick(Rational {
            num: 1,
            den: SAMPLE_RATE as i32,
        });
        assert!((tick * f64::from(WINDOW) - 30.0).abs() < 1e-9);
    }
}
