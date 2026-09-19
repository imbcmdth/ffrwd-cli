//! A note every few frames, in the stream's own time base.
//!
//! The producer half of the packet-filter fixtures: `packet_sei` weaves rows
//! shaped `{"pts": <tick>, "note": "<text>"}` into a stream, and this is what
//! writes them. Frames pass through untouched; every `every`-th one carries
//! away a row whose `pts` is its own, so a note's place in the stream is a
//! fact the test can check rather than a guess.
//!
//! `label` goes in front of each note, which is what lets one query wire two
//! instances into two rows arguments and tell their notes apart afterwards.

wit_bindgen::generate!({
    path: "../../wit",
    world: "window-module",
});

use std::cell::RefCell;

use exports::ffrwd::av::window_filter::{
    Format, FramePayload, Guest, InWindow, Meta, OutFrame, Processed, StreamInfo, WindowMeta,
};
use serde::{Deserialize, Serialize};

const PARAMS_SCHEMA: &str = r#"{"type":"object","properties":{"label":{"type":"string"},"every":{"type":"integer"}},"additionalProperties":false}"#;
const ROWS_SCHEMA: &str = r#"{"type":"object","properties":{"pts":{"type":"integer"},"note":{"type":"string"}},"required":["pts","note"],"additionalProperties":false}"#;

#[derive(Deserialize)]
struct Params {
    #[serde(default = "default_label")]
    label: String,
    #[serde(default = "default_every")]
    every: u64,
}

fn default_label() -> String {
    "note".to_string()
}

fn default_every() -> u64 {
    5
}

/// One row this module writes: where the note belongs, and what it says.
#[derive(Serialize)]
struct Row {
    pts: i64,
    note: String,
}

struct State {
    label: String,
    every: u64,
    seen: u64,
}

thread_local! {
    static STATE: RefCell<Option<State>> = const { RefCell::new(None) };
}

fn read_params(params: &str) -> Result<Params, String> {
    match params.trim() {
        "" | "{}" => Ok(Params {
            label: default_label(),
            every: default_every(),
        }),
        written => {
            let read: Params = serde_json::from_str(written)
                .map_err(|error| format!("note_rows params: {error}"))?;
            if read.every == 0 {
                return Err("note_rows: 'every' counts frames, so it is at least 1".to_string());
            }
            Ok(read)
        }
    }
}

struct NoteRows;

impl Guest for NoteRows {
    fn describe() -> WindowMeta {
        WindowMeta {
            meta: Meta {
                name: "note_rows".to_string(),
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
            // The frame count carries over between calls.
            pure: false,
            one_to_one: true,
            reads_rows: false,
            // The rows leaving are this module's own.
            forwards_rows: false,
            inputs: 1,
        }
    }

    fn init(format: Format, _stream_info: StreamInfo, params: String) -> Result<(), String> {
        let read = read_params(&params)?;
        let Format::Video(_) = format else {
            return Err("note_rows reads pictures, and this stream is audio".to_string());
        };
        STATE.with(|s| {
            *s.borrow_mut() = Some(State {
                label: read.label,
                every: read.every,
                seen: 0,
            });
        });
        Ok(())
    }

    fn set_params(params: String) -> Result<(), String> {
        read_params(&params).map(|_| ())
    }

    fn process(window: &InWindow, _trailing: Vec<String>, last: bool) -> Processed {
        let _ = last;
        STATE.with(|s| {
            let mut held = s.borrow_mut();
            let state = held.as_mut().expect("process called before init");
            let mut out: Vec<OutFrame> = Vec::with_capacity(window.len() as usize);
            for i in 0..window.len() {
                let index = state.seen;
                state.seen += 1;
                let pts = window.pts(i);
                let rows = if index % state.every == 0 {
                    let row = Row {
                        pts,
                        note: format!("{}-{index}", state.label),
                    };
                    vec![serde_json::to_string(&row).expect("a row serializes")]
                } else {
                    Vec::new()
                };
                // The pictures are not this module's business: it reads the
                // times and hands every frame straight back.
                out.push(OutFrame {
                    pts,
                    frame: FramePayload::Same,
                    rows,
                });
            }
            Processed {
                frames: out,
                trailing: Vec::new(),
            }
        })
    }
}

export!(NoteRows);
