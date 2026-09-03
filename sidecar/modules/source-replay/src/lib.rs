//! A packet source that replays a fixed set of coded packets compiled into
//! the module: the sidecar's own coded-edge shape (pts, dts, keyframe,
//! bytes), with no video encoder involved. `probe` and `open` both publish
//! the same single-track, bounded catalog; `next` hands the packets out one
//! at a time until they run out.
//!
//! The wasm sandbox this module runs in has no filesystem, so a literal
//! "read a NUT file named by params" is not reachable from inside it: the
//! packets replayed here are compiled in rather than read from a path. What
//! this proves is unchanged - run this module through the sidecar's
//! `run_packet_source` and the NUT it writes reads back through
//! `nut::Demuxer` with exactly these pts/dts/keyframe/bytes, and the same
//! reorder pattern `nut::mux`'s own coded-packet test pins.

wit_bindgen::generate!({
    path: "../../wit",
    world: "packet-source-module",
});

use std::cell::Cell;

use crate::ffrwd::av::types::{CodedFormat, CodedStream, CodedVideo, Packet, Rational};
use exports::ffrwd::av::packet_source::{
    Catalog, Guest, Meta, RenditionMeta, SourceTrack, StreamInfo,
};

const PARAMS_SCHEMA: &str = r#"{"type":"object","properties":{},"additionalProperties":false}"#;

/// Decode order I0 P3 B1 B2 P6 B4 B5, and the dts a reorder buffer of depth 2
/// settles for it - the exact pattern and numbers `nut::mux`'s own
/// coded-packet roundtrip test pins, so this module's proof rests on values
/// already known correct.
const PACKETS: &[(i64, Option<i64>, bool)] = &[
    (0, None, true),
    (3, None, false),
    (1, Some(0), false),
    (2, Some(1), false),
    (6, Some(2), false),
    (4, Some(3), false),
    (5, Some(4), false),
];

/// SPS/PPS shaped like a real h264 stream's, so `coded_stream_for`'s profile
/// and level are exercised the same way a real one would be.
const EXTRADATA: &[u8] = &[
    0x00, 0x00, 0x00, 0x01, 0x67, 0x64, 0x00, 0x0a, 0xac, 0xd9, 0x44, 0x7b, 0x01, 0x10, 0x00, 0x00,
    0x00, 0x01, 0x68, 0xeb, 0xe3, 0xcb, 0x22, 0xc0,
];

fn coded_stream() -> CodedStream {
    CodedStream {
        codec: "h264".to_string(),
        time_base: Rational { num: 1, den: 25 },
        format: CodedFormat::Video(CodedVideo {
            width: 32,
            height: 24,
            sample_aspect_ratio: None,
            color: None,
        }),
        extradata: EXTRADATA.to_vec(),
        profile: Some(0x64),
        level: Some(0x0a),
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

/// `source_replay` reads no parameters; anything but empty is refused by
/// name.
fn validate_params(params: &str) -> Result<(), String> {
    match params.trim() {
        "" | "{}" => Ok(()),
        other => Err(format!("source_replay takes no params, got: {other}")),
    }
}

thread_local! {
    /// The index into `PACKETS` the next `next()` call hands out.
    static CURSOR: Cell<usize> = const { Cell::new(0) };
}

struct SourceReplay;

impl Guest for SourceReplay {
    fn describe() -> Meta {
        Meta {
            name: "source_replay".to_string(),
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
            if index >= PACKETS.len() {
                return Ok(None);
            }
            c.set(index + 1);
            let (pts, dts, keyframe) = PACKETS[index];
            let packet = Packet {
                pts,
                dts,
                duration: None,
                keyframe,
                // Payload bytes carry no meaning; only shape and timing
                // matter for the roundtrip this module proves.
                data: vec![0x10u8.wrapping_add(index as u8); 24],
            };
            Ok(Some(vec![exports::ffrwd::av::packet_source::PadPackets {
                packets: vec![packet],
            }]))
        })
    }
}

export!(SourceReplay);

#[cfg(test)]
mod tests {
    use super::PACKETS;

    #[test]
    fn the_leading_none_dts_count_is_the_decode_delay_the_stream_needs() {
        let leading_none = PACKETS
            .iter()
            .take_while(|(_, dts, _)| dts.is_none())
            .count();
        assert_eq!(leading_none, 2, "an I-P-B-B GOP needs a reorder depth of 2");
    }

    #[test]
    fn every_settled_dts_is_non_decreasing() {
        let settled: Vec<i64> = PACKETS.iter().filter_map(|(_, dts, _)| *dts).collect();
        assert!(settled.windows(2).all(|pair| pair[0] <= pair[1]));
    }

    #[test]
    fn exactly_one_packet_is_a_keyframe() {
        assert_eq!(PACKETS.iter().filter(|(_, _, key)| *key).count(), 1);
    }
}
