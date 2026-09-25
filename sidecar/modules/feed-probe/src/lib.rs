//! A feeder, read and reported: the module's own stream passes through
//! untouched, and a second stream, the feeder, arrives as NUT on a loopback
//! port the module listens on itself (see `feed`). Each picture the feeder
//! sends is one row, its pts in the feeder's own time base and its size.
//!
//! The port is the `port` param, which the host fills with a port it picked
//! when a stream is written in the feeder's place.

wit_bindgen::generate!({
    path: "../../wit",
    world: "window-module",
});

mod feed;

use std::cell::RefCell;

use exports::ffrwd::av::window_filter::{
    Feeder, Format, Guest, InWindow, Meta, Processed, StreamInfo, WindowMeta,
};
use ffrwd_nut::Stream;
use serde::Serialize;

use feed::{Feed, Probe, PARAMS_SCHEMA};

const ROWS_SCHEMA: &str = r#"{"type":"object","properties":{"feed_pts":{"type":"integer"},"w":{"type":"integer"},"h":{"type":"integer"}},"additionalProperties":false}"#;

#[derive(Serialize)]
struct Row {
    feed_pts: i64,
    w: u32,
    h: u32,
}

/// The feeder's first picture stream, a row per picture.
struct Pictures;

impl Probe for Pictures {
    const NAME: &'static str = "feed-probe";

    fn counts(stream: &Stream) -> bool {
        stream.video_geometry().is_some()
    }

    fn row(stream: &Stream, pts: i64, _payload: &[u8]) -> String {
        let (w, h) = stream.video_geometry().expect("counted as a picture");
        serde_json::to_string(&Row {
            feed_pts: pts,
            w,
            h,
        })
        .expect("row serializes")
    }
}

thread_local! {
    static STATE: RefCell<Option<Feed<Pictures>>> = const { RefCell::new(None) };
}

struct FeedProbe;

impl Guest for FeedProbe {
    fn describe() -> WindowMeta {
        WindowMeta {
            meta: Meta {
                name: "feed-probe".to_string(),
                version: "0.1.0".to_string(),
                params_schema: PARAMS_SCHEMA.to_string(),
                rows_schema: ROWS_SCHEMA.to_string(),
                pixel_formats: vec!["yuv420p".to_string()],
                sample_formats: vec![],
                sample_rates: vec![],
                channel_counts: vec![],
                rows_language: vec![],
            },
            window: 1,
            stride: 1,
            // The connection is state carried from call to call.
            pure: false,
            one_to_one: true,
            reads_rows: false,
            forwards_rows: false,
            inputs: 1,
            feeders: vec![Feeder {
                input: 1,
                port_param: "port".to_string(),
                kind: "video".to_string(),
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
        feed::check_params(&params_text, Pictures::NAME)
    }

    fn process(window: &InWindow, _trailing: Vec<String>, last: bool) -> Processed {
        STATE.with(|cell| feed::process(cell.borrow_mut().as_mut(), window, last))
    }
}

export!(FeedProbe);
