//! A packet source publishing one DATA track: a handful of JSON messages,
//! each one packet at its own pts, compiled in the way `source-replay`
//! compiles in its h264 packets.
//!
//! It is the fixture a data stream's round trip is written against - source
//! to sink, and through a filter - so the messages are chosen to catch what
//! a wire could lose: two at one pts, a character outside ASCII, a gap
//! longer than a second (which NUT codes as a full pts rather than a step),
//! and a message whose bytes are not the canonical spelling of its JSON.
//!
//! Track 1 is a picture beside them: the keyframe `source-replay` compiles
//! in, once every tenth of a second to 3.6 s. Subscribed with it, each pull
//! is one picture and the messages up to its time, so the messages are
//! sparse against a media clock that keeps moving.

wit_bindgen::generate!({
    path: "../../wit",
    world: "packet-source-module",
});

use std::cell::Cell;

use crate::ffrwd::av::types::{CodedFormat, CodedStream, CodedVideo, Packet, Rational};
use exports::ffrwd::av::packet_source::{
    Catalog, Guest, Meta, PadPackets, RenditionMeta, SourceTrack, StreamInfo,
};

const PARAMS_SCHEMA: &str = r#"{"type":"object","properties":{},"additionalProperties":false}"#;

/// Microseconds, the unit a data stream's pts are usually counted in.
const TIME_BASE: Rational = Rational {
    num: 1,
    den: 1_000_000,
};

// Only the keyframe is replayed, so the rest of the table goes unread.
#[allow(dead_code)]
mod generated {
    include!("../../source-replay/src/generated/packets.rs");
}
use generated::{EXTRADATA_LEN, HEIGHT, LEVEL, PACKET_LENS, PROFILE, RAW, WIDTH};

/// The picture's time base: a frame every tenth of a second.
const FRAME_BASE: Rational = Rational { num: 1, den: 10 };

/// How many pictures track 1 carries: to 3.6 s, one past the last message.
const FRAMES: i64 = 37;

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

/// The picture: `source-replay`'s keyframe, its own extradata and geometry.
fn picture() -> SourceTrack {
    SourceTrack {
        coded: CodedStream {
            codec: "h264".to_string(),
            time_base: FRAME_BASE,
            format: CodedFormat::Video(CodedVideo {
                width: WIDTH,
                height: HEIGHT,
                sample_aspect_ratio: None,
                color: None,
            }),
            extradata: RAW[..EXTRADATA_LEN].to_vec(),
            profile: Some(PROFILE),
            level: Some(LEVEL),
        },
        info: StreamInfo {
            index: 1,
            kind: "video".to_string(),
            codec: "h264".to_string(),
            duration: None,
            tags: vec![],
            time_base: FRAME_BASE,
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

/// The catalog restricted to `tracks`, in that order.
fn subscribed(tracks: &[u32]) -> Result<Catalog, String> {
    let mut subscribed = Vec::with_capacity(tracks.len());
    for index in tracks {
        subscribed.push(match index {
            0 => track(),
            1 => picture(),
            _ => {
                return Err(format!(
                    "source_replay_data publishes 2 tracks, so track {index} is not one of them"
                ))
            }
        });
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
    /// The picture the next `next()` call hands out, with the picture
    /// subscribed.
    static FRAME: Cell<i64> = const { Cell::new(0) };
    /// Which catalog track each pad carries, in `open`'s order.
    static PADS: Cell<[Option<u32>; 2]> = const { Cell::new([None, None]) };
}

fn message_packet(pts: i64, message: &str) -> Packet {
    Packet {
        pts,
        dts: Some(pts),
        duration: None,
        keyframe: true,
        data: message.as_bytes().to_vec(),
    }
}

/// The messages due by `until` microseconds, taken off the cursor.
fn messages_until(until: i64) -> Vec<Packet> {
    CURSOR.with(|c| {
        let mut taken = Vec::new();
        while let Some((pts, message)) = MESSAGES.get(c.get()) {
            if *pts > until {
                break;
            }
            taken.push(message_packet(*pts, message));
            c.set(c.get() + 1);
        }
        taken
    })
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
        subscribed(&[0, 1])
    }

    fn open(params: String, tracks: Vec<u32>) -> Result<Catalog, String> {
        validate_params(&params)?;
        let subscribed = subscribed(&tracks)?;
        CURSOR.with(|c| c.set(0));
        FRAME.with(|c| c.set(0));
        PADS.with(|c| c.set([tracks.first().copied(), tracks.get(1).copied()]));
        Ok(subscribed)
    }

    /// One message per pull, the way a live source hands them on as they
    /// come; with the picture, one picture per pull and the messages up to
    /// its time.
    fn next() -> Result<Option<Vec<PadPackets>>, String> {
        let pads = PADS.with(|c| c.get());
        let pads: Vec<u32> = pads.into_iter().flatten().collect();
        let (pictures, messages) = if pads.contains(&1) {
            let frame = FRAME.with(|c| c.get());
            if frame >= FRAMES {
                return Ok(None);
            }
            FRAME.with(|c| c.set(frame + 1));
            let keyframe = Packet {
                pts: frame,
                dts: Some(frame),
                duration: None,
                keyframe: true,
                data: RAW[EXTRADATA_LEN..EXTRADATA_LEN + PACKET_LENS[0]].to_vec(),
            };
            (vec![keyframe], messages_until(frame * 100_000))
        } else {
            let messages = CURSOR.with(|c| {
                MESSAGES
                    .get(c.get())
                    .map(|(pts, message)| {
                        c.set(c.get() + 1);
                        vec![message_packet(*pts, message)]
                    })
                    .unwrap_or_default()
            });
            if messages.is_empty() {
                return Ok(None);
            }
            (vec![], messages)
        };
        Ok(Some(
            pads.iter()
                .map(|track| PadPackets {
                    packets: if *track == 1 {
                        pictures.clone()
                    } else {
                        messages.clone()
                    },
                })
                .collect(),
        ))
    }
}

export!(SourceReplayData);
