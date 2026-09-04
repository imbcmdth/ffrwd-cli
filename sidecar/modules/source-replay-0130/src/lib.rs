//! `source_replay`, built against the vendored `ffrwd:av@0.13.0` world
//! instead of the current one: its `open` takes params alone, the shape
//! every packet source had before 0.15.0 told a source which tracks to pull.
//! It is `source-replay` line for line, its `wit_bindgen::generate!` pointed
//! at the older world instead, and it exists so the host's refusal of a
//! packet source built against an older world is proven against a module
//! actually built that way.
//!
//! The catalog and the seven packets it replays are `source-replay`'s own -
//! see `modules/source-replay/src/lib.rs` for what they are and why they are
//! compiled in rather than read from a file.

wit_bindgen::generate!({
    path: "../../worlds/0.13.0",
    world: "packet-source-module",
});

use std::cell::Cell;

use crate::ffrwd::av::types::{CodedFormat, CodedStream, CodedVideo, Packet, Rational};
use exports::ffrwd::av::packet_source::{
    Catalog, Guest, Meta, RenditionMeta, SourceTrack, StreamInfo,
};

mod generated {
    include!("../../source-replay/src/generated/packets.rs");
}
use generated::{EXTRADATA_LEN, LEVEL, PACKET_LENS, PACKET_TABLE, PROFILE, RAW};

const PARAMS_SCHEMA: &str = r#"{"type":"object","properties":{},"additionalProperties":false}"#;

fn extradata() -> &'static [u8] {
    &RAW[..EXTRADATA_LEN]
}

/// The real coded bytes for packet `index`, sliced out of `RAW` by
/// `PACKET_LENS` - each one starts with its own Annex-B start code.
fn packet_bytes(index: usize) -> &'static [u8] {
    let start = EXTRADATA_LEN + PACKET_LENS[..index].iter().sum::<usize>();
    &RAW[start..start + PACKET_LENS[index]]
}

fn coded_stream() -> CodedStream {
    CodedStream {
        codec: "h264".to_string(),
        time_base: Rational { num: 1, den: 25 },
        format: CodedFormat::Video(CodedVideo {
            width: generated::WIDTH,
            height: generated::HEIGHT,
            sample_aspect_ratio: None,
            color: None,
        }),
        extradata: extradata().to_vec(),
        profile: Some(PROFILE),
        level: Some(LEVEL),
    }
}

fn catalog() -> Catalog {
    Catalog {
        tracks: vec![SourceTrack {
            coded: coded_stream(),
            info: StreamInfo {
                index: 0,
                kind: "video".to_string(),
                codec: "h264".to_string(),
                duration: None,
                tags: vec![],
                time_base: Rational { num: 1, den: 25 },
            },
            row: 0,
            rendition: RenditionMeta {
                name: None,
                bandwidth: None,
                codecs: None,
                language: None,
            },
        }],
        bounded: true,
    }
}

/// `source_replay_0130` reads no parameters; anything but empty is refused
/// by name.
fn validate_params(params: &str) -> Result<(), String> {
    match params.trim() {
        "" | "{}" => Ok(()),
        other => Err(format!("source_replay_0130 takes no params, got: {other}")),
    }
}

thread_local! {
    /// The index into `PACKET_TABLE` the next `next()` call hands out.
    static CURSOR: Cell<usize> = const { Cell::new(0) };
}

struct SourceReplay0130;

impl Guest for SourceReplay0130 {
    fn describe() -> Meta {
        Meta {
            name: "source_replay_0130".to_string(),
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
        Ok(catalog())
    }

    fn open(params: String) -> Result<Catalog, String> {
        validate_params(&params)?;
        CURSOR.with(|c| c.set(0));
        Ok(catalog())
    }

    fn next() -> Result<Option<Vec<exports::ffrwd::av::packet_source::PadPackets>>, String> {
        CURSOR.with(|c| {
            let index = c.get();
            if index >= PACKET_TABLE.len() {
                return Ok(None);
            }
            c.set(index + 1);
            let (pts, dts, keyframe) = PACKET_TABLE[index];
            let packet = Packet {
                pts,
                dts,
                duration: None,
                keyframe,
                data: packet_bytes(index).to_vec(),
            };
            Ok(Some(vec![exports::ffrwd::av::packet_source::PadPackets {
                packets: vec![packet],
            }]))
        })
    }
}

export!(SourceReplay0130);
