//! A packet source publishing one DATA track: a handful of JSON messages,
//! each one packet at its own pts, compiled in the way `source-replay`
//! compiles in its h264 packets.
//!
//! It is the fixture a data stream's round trip is written against - source
//! to sink, and through a filter - so the messages are chosen to catch what
//! a wire could lose: two at one pts, a character outside ASCII, a gap
//! longer than a second (which NUT codes as a full pts rather than a step),
//! and a message whose bytes are not the canonical spelling of its JSON.

wit_bindgen::generate!({
    path: "../../wit",
    world: "packet-source-module",
});

use std::cell::Cell;

use crate::ffrwd::av::types::{CodedFormat, CodedStream, Packet, Rational};
use exports::ffrwd::av::packet_source::{
    Catalog, Guest, Meta, PadPackets, RenditionMeta, SourceTrack, StreamInfo,
};

const PARAMS_SCHEMA: &str = r#"{"type":"object","properties":{},"additionalProperties":false}"#;

/// Microseconds, the unit a data stream's pts are usually counted in.
const TIME_BASE: Rational = Rational {
    num: 1,
    den: 1_000_000,
};

/// The messages, in pts order: `(pts, message)`.
pub const MESSAGES: &[(i64, &str)] = &[
    (0, r#"{"kind":"start","n":0}"#),
    (40_000, r#"{"n":1,"text":"café"}"#),
    (40_000, r#"{"n":2,"text":"café"}"#),
    (3_500_000, r#"{ "n" : 3 , "spaced" : true }"#),
    (3_500_001, r#"{"n":4,"last":true}"#),
];

fn track() -> SourceTrack {
    SourceTrack {
        coded: CodedStream {
            codec: "json".to_string(),
            time_base: TIME_BASE,
            format: CodedFormat::Data,
            extradata: vec![],
            profile: None,
            level: None,
        },
        info: StreamInfo {
            index: 0,
            kind: "data".to_string(),
            codec: "json".to_string(),
            duration: None,
            tags: vec![],
            time_base: TIME_BASE,
        },
        row: 0,
        rendition: RenditionMeta {
            name: None,
            bandwidth: None,
            codecs: None,
            language: None,
        },
    }
}

/// The catalog restricted to `tracks`: this module publishes one track, so 0
/// is the only index it has.
fn subscribed(tracks: &[u32]) -> Result<Catalog, String> {
    let mut subscribed = Vec::with_capacity(tracks.len());
    for index in tracks {
        if *index != 0 {
            return Err(format!(
                "source_replay_data publishes 1 track, so track {index} is not one of them"
            ));
        }
        subscribed.push(track());
    }
    Ok(Catalog {
        tracks: subscribed,
        bounded: true,
    })
}

fn validate_params(params: &str) -> Result<(), String> {
    match params.trim() {
        "" | "{}" => Ok(()),
        other => Err(format!("source_replay_data takes no params, got: {other}")),
    }
}

thread_local! {
    /// The index into `MESSAGES` the next `next()` call hands out.
    static CURSOR: Cell<usize> = const { Cell::new(0) };
    /// How many pads `next()` answers on: one per track `open` was given.
    static PADS: Cell<usize> = const { Cell::new(0) };
}

struct SourceReplayData;

impl Guest for SourceReplayData {
    fn describe() -> Meta {
        Meta {
            name: "source_replay_data".to_string(),
            version: "0.1.0".to_string(),
            params_schema: PARAMS_SCHEMA.to_string(),
            rows_schema: String::new(),
            pixel_formats: vec![],
            sample_formats: vec![],
            sample_rates: vec![],
            channel_counts: vec![],
            rows_language: vec![],
        }
    }

    fn probe(params: String) -> Result<Catalog, String> {
        validate_params(&params)?;
        subscribed(&[0])
    }

    fn open(params: String, tracks: Vec<u32>) -> Result<Catalog, String> {
        validate_params(&params)?;
        let subscribed = subscribed(&tracks)?;
        CURSOR.with(|c| c.set(0));
        PADS.with(|c| c.set(subscribed.tracks.len()));
        Ok(subscribed)
    }

    /// One message per pull, the way a live source hands them on as they
    /// come.
    fn next() -> Result<Option<Vec<PadPackets>>, String> {
        let pads = PADS.with(|c| c.get());
        CURSOR.with(|c| {
            let index = c.get();
            let Some((pts, message)) = MESSAGES.get(index) else {
                return Ok(None);
            };
            c.set(index + 1);
            let packet = Packet {
                pts: *pts,
                dts: Some(*pts),
                duration: None,
                keyframe: true,
                data: message.as_bytes().to_vec(),
            };
            Ok(Some(
                (0..pads)
                    .map(|_| PadPackets {
                        packets: vec![packet.clone()],
                    })
                    .collect(),
            ))
        })
    }
}

export!(SourceReplayData);
