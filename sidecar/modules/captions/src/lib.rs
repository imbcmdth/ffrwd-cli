//! Cue rows, so a subtitle output has something to read in the unit tier.
//!
//! One cue per frame - the frame's own time to the next frame's - saying which
//! frame it was, with the pixels passed through untouched. The stream ends
//! with one trailing record that is not a cue at all, which is the other arm
//! of the rows schema and the thing a subtitle output passes over.
//!
//! `malformed` makes the first cue leave out its end time, so the refusal a
//! subtitle output gives a cue that does not hold together is reachable from
//! outside.

wit_bindgen::generate!({
    path: "../../wit",
    world: "window-module",
});

use std::cell::RefCell;

use exports::ffrwd::av::window_filter::{
    Format, FramePayload, Guest, InFrame, Meta, OutFrame, Processed, StreamInfo, WindowMeta,
};
use serde::{Deserialize, Serialize};

const PARAMS_SCHEMA: &str = r#"{"type":"object","properties":{"malformed":{"type":"boolean","default":false}},"additionalProperties":false}"#;
const ROWS_SCHEMA: &str = r#"{"oneOf":[{"type":"object","properties":{"start_t":{"type":"number"},"end_t":{"type":"number"},"text":{"type":"string"}},"required":["start_t","end_t","text"],"additionalProperties":false},{"type":"object","properties":{"cues":{"type":"integer"}},"required":["cues"],"additionalProperties":false}]}"#;

/// Seconds one cue lasts, whatever the stream's own rate.
const CUE_SECONDS: f64 = 0.5;

#[derive(Deserialize)]
struct Params {
    #[serde(default)]
    malformed: bool,
}

/// One cue, in seconds from the start of the stream.
#[derive(Serialize)]
struct Cue {
    start_t: f64,
    end_t: f64,
    text: String,
}

/// A cue with no end time, for the refusal that names the missing field.
#[derive(Serialize)]
struct HalfCue {
    start_t: f64,
    text: String,
}

/// How many cues the stream carried, on the final call. Not a cue itself, so a
/// subtitle output passes it over.
#[derive(Serialize)]
struct Count {
    cues: usize,
}

struct State {
    /// Seconds one timestamp tick is worth, from the stream's time base.
    tick: f64,
    malformed: bool,
    said: usize,
}

thread_local! {
    static STATE: RefCell<Option<State>> = const { RefCell::new(None) };
}

fn parse_params(params: &str) -> Result<bool, String> {
    let trimmed = params.trim();
    if trimmed.is_empty() {
        return Ok(false);
    }
    let parsed: Params =
        serde_json::from_str(trimmed).map_err(|e| format!("captions: invalid params: {e}"))?;
    Ok(parsed.malformed)
}

struct Captions;

impl Guest for Captions {
    fn describe() -> WindowMeta {
        WindowMeta {
            meta: Meta {
                name: "captions".to_string(),
                version: "0.1.0".to_string(),
                params_schema: PARAMS_SCHEMA.to_string(),
                rows_schema: ROWS_SCHEMA.to_string(),
                pixel_formats: vec!["rgba".to_string()],
                sample_formats: vec![],
                sample_rates: vec![],
                channel_counts: vec![],
                rows_language: vec![],
            },
            window: 1,
            stride: 1,
            // The count the final call carries gathers the whole stream.
            pure: false,
            one_to_one: true,
            reads_rows: false,
            forwards_rows: false,
            // One stream in, which is every module here.
            inputs: 1,
        }
    }

    fn init(format: Format, stream_info: StreamInfo, params: String) -> Result<(), String> {
        let malformed = parse_params(&params)?;
        let Format::Video(_) = format else {
            return Err("captions writes over pictures, and this stream is audio".to_string());
        };
        let base = stream_info.time_base;
        if base.den <= 0 {
            return Err(format!(
                "captions: the time base {}/{} is not a rate",
                base.num, base.den
            ));
        }
        STATE.with(|s| {
            *s.borrow_mut() = Some(State {
                tick: f64::from(base.num) / f64::from(base.den),
                malformed,
                said: 0,
            });
        });
        Ok(())
    }

    fn set_params(params: String) -> Result<(), String> {
        let malformed = parse_params(&params)?;
        STATE.with(|s| {
            let mut state_ref = s.borrow_mut();
            let state = state_ref.as_mut().expect("set_params called before init");
            state.malformed = malformed;
        });
        Ok(())
    }

    fn process(frames: Vec<InFrame>, _trailing: Vec<String>, last: bool) -> Processed {
        STATE.with(|s| {
            let mut state_ref = s.borrow_mut();
            let state = state_ref.as_mut().expect("process called before init");

            let out: Vec<OutFrame> = frames
                .into_iter()
                .map(|frame| {
                    let start = frame.pts as f64 * state.tick;
                    let index = state.said;
                    state.said += 1;
                    let row = if state.malformed && index == 0 {
                        serde_json::to_string(&HalfCue {
                            start_t: start,
                            text: format!("cue {index}"),
                        })
                    } else {
                        serde_json::to_string(&Cue {
                            start_t: start,
                            end_t: start + CUE_SECONDS,
                            text: format!("cue {index}"),
                        })
                    };
                    OutFrame {
                        pts: frame.pts,
                        frame: FramePayload::Same,
                        rows: vec![row.expect("a row serializes")],
                    }
                })
                .collect();

            let trailing = if last {
                vec![serde_json::to_string(&Count { cues: state.said }).expect("a row serializes")]
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

export!(Captions);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_params_writes_whole_cues() {
        assert!(!parse_params("").expect("no params is the default"));
        assert!(parse_params(r#"{"malformed":true}"#).expect("a boolean"));
        let Err(err) = parse_params(r#"{"malformed":"yes"}"#) else {
            panic!("a malformed flag that is not a boolean should have been refused");
        };
        assert!(err.contains("captions"), "got: {err}");
    }

    #[test]
    fn a_cue_serializes_under_the_names_a_subtitle_output_reads() {
        let row = serde_json::to_string(&Cue {
            start_t: 1.0,
            end_t: 1.5,
            text: "cue 0".to_string(),
        })
        .expect("a row serializes");
        assert_eq!(row, r#"{"start_t":1.0,"end_t":1.5,"text":"cue 0"}"#);
    }
}
