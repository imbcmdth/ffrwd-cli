//! `feed-probe` for sound: the module's own sound passes through untouched,
//! and a second sound, the feeder, arrives as NUT on a loopback port the
//! module listens on itself. Each packet of it is one row: its pts in the
//! feeder's own time base, the rate, channel count and sample format its
//! stream header names, and how many samples it carries.
//!
//! It reads f32 at 48 kHz in stereo, so a feeder is conformed to that before
//! it is written.

wit_bindgen::generate!({
    path: "../../wit",
    world: "window-module",
});

#[path = "../../feed-probe/src/feed.rs"]
mod feed;

use std::cell::RefCell;

use exports::ffrwd::av::window_filter::{
    Feeder, Format, Guest, InWindow, Meta, Processed, StreamInfo, WindowMeta,
};
use ffrwd_nut::Stream;
use serde::Serialize;

use feed::{Feed, Probe, PARAMS_SCHEMA};

const ROWS_SCHEMA: &str = r#"{"type":"object","properties":{"feed_pts":{"type":"integer"},"rate":{"type":"integer"},"channels":{"type":"integer"},"sample_fmt":{"type":"string"},"samples":{"type":"integer"}},"additionalProperties":false}"#;

/// Samples one call is handed.
const WINDOW: u32 = 1024;

#[derive(Serialize)]
struct Row<'a> {
    feed_pts: i64,
    rate: u32,
    channels: u32,
    sample_fmt: &'a str,
    samples: usize,
}

/// The feeder's first sound stream, a row per packet.
struct Sound;

impl Probe for Sound {
    const NAME: &'static str = "feed-probe-audio";

    fn counts(stream: &Stream) -> bool {
        stream.audio_geometry().is_some()
    }

    fn row(stream: &Stream, pts: i64, payload: &[u8]) -> String {
        let (rate, channels) = stream.audio_geometry().expect("counted as sound");
        let sample_fmt = stream.sample_fmt().unwrap_or("");
        let width = match sample_fmt {
            "f32" => 4,
            "s16" => 2,
            _ => 0,
        };
        let frame = width * channels as usize;
        let row = Row {
            feed_pts: pts,
            rate,
            channels,
            sample_fmt,
            samples: payload.len().checked_div(frame).unwrap_or(0),
        };
        serde_json::to_string(&row).expect("row serializes")
    }
}

thread_local! {
    static STATE: RefCell<Option<Feed<Sound>>> = const { RefCell::new(None) };
}

struct FeedProbeAudio;

impl Guest for FeedProbeAudio {
    fn describe() -> WindowMeta {
        WindowMeta {
            meta: Meta {
                name: "feed-probe-audio".to_string(),
                version: "0.1.0".to_string(),
                params_schema: PARAMS_SCHEMA.to_string(),
                rows_schema: ROWS_SCHEMA.to_string(),
                pixel_formats: vec![],
                sample_formats: vec!["f32".to_string()],
                sample_rates: vec![48000],
                channel_counts: vec![2],
                rows_language: vec![],
            },
            window: WINDOW,
            stride: WINDOW,
            // The connection is state carried from call to call.
            pure: false,
            one_to_one: true,
            reads_rows: false,
            forwards_rows: false,
            inputs: 1,
            feeders: vec![Feeder {
                input: 1,
                port_param: "port".to_string(),
                kind: "audio".to_string(),
                group: String::new(),
            }],
        }
    }

    fn init(_format: Format, _stream_info: StreamInfo, params_text: String) -> Result<(), String> {
        let state = Feed::open(&params_text)?;
        STATE.with(|cell| *cell.borrow_mut() = Some(state));
        Ok(())
    }

    fn set_params(params_text: String) -> Result<(), String> {
        feed::check_params(&params_text, Sound::NAME)
    }

    fn process(window: &InWindow, _trailing: Vec<String>, last: bool) -> Processed {
        STATE.with(|cell| feed::process(cell.borrow_mut().as_mut(), window, last))
    }
}

export!(FeedProbeAudio);
