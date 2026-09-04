//! A packet source that replays a real coded h264 stream compiled into the
//! module: seven frames of a 32x24 test pattern, actually encoded with
//! libx264 (see `cli/scripts/gen_replay_packets.py`), so both the
//! pts/dts/keyframe table and the packet bytes themselves are a real
//! stream's - not shaped to look like one.
//!
//! The wasm sandbox this module runs in has no filesystem, so a literal
//! "read a NUT file named by params" is not reachable from inside it: the
//! packets replayed here are compiled in rather than read from a path. What
//! this proves is unchanged - run this module through the sidecar's
//! `run_packet_source` and the NUT it writes reads back through
//! `nut::Demuxer` with exactly these pts/dts/keyframe/bytes, decodes with a
//! real h264 decoder, and settles into the same reorder pattern
//! `nut::mux`'s own coded-packet test pins.

wit_bindgen::generate!({
    path: "../../wit",
    world: "packet-source-module",
});

use std::cell::Cell;

use crate::ffrwd::av::types::{CodedFormat, CodedStream, CodedVideo, Packet, Rational};
use exports::ffrwd::av::packet_source::{
    Catalog, Guest, Meta, RenditionMeta, SourceTrack, StreamInfo,
};

mod generated {
    include!("generated/packets.rs");
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

/// The one track this module publishes, index 0 of its catalog.
fn track() -> SourceTrack {
    SourceTrack {
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
    }
}

/// The whole catalog, which `probe` reports.
fn catalog() -> Catalog {
    Catalog {
        tracks: vec![track()],
        bounded: true,
    }
}

/// The catalog restricted to `tracks`, in that order. This module publishes
/// one track, so 0 is the only index it has; anything else is refused by
/// name.
fn subscribed(tracks: &[u32]) -> Result<Catalog, String> {
    let mut subscribed = Vec::with_capacity(tracks.len());
    for index in tracks {
        if *index != 0 {
            return Err(format!(
                "source_replay publishes 1 track, so track {index} is not one of them"
            ));
        }
        subscribed.push(track());
    }
    Ok(Catalog {
        tracks: subscribed,
        bounded: true,
    })
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
    /// The index into `PACKET_TABLE` the next `next()` call hands out.
    static CURSOR: Cell<usize> = const { Cell::new(0) };

    /// How many pads `next()` answers on: one per track `open` was given.
    static PADS: Cell<usize> = const { Cell::new(0) };
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

    fn open(params: String, tracks: Vec<u32>) -> Result<Catalog, String> {
        validate_params(&params)?;
        let subscribed = subscribed(&tracks)?;
        CURSOR.with(|c| c.set(0));
        PADS.with(|c| c.set(subscribed.tracks.len()));
        Ok(subscribed)
    }

    fn next() -> Result<Option<Vec<exports::ffrwd::av::packet_source::PadPackets>>, String> {
        let pads = PADS.with(|c| c.get());
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
            // One pad per track `open` was given, in that order - which for
            // this module is the same track repeated.
            Ok(Some(
                (0..pads)
                    .map(|_| exports::ffrwd::av::packet_source::PadPackets {
                        packets: vec![packet.clone()],
                    })
                    .collect(),
            ))
        })
    }
}

export!(SourceReplay);

#[cfg(test)]
mod tests {
    use super::{generated::PACKET_LENS, packet_bytes, subscribed, PACKET_TABLE};

    #[test]
    fn opening_the_one_track_answers_a_catalog_of_it() {
        let catalog = subscribed(&[0]).expect("track 0 is the one this module publishes");
        assert_eq!(catalog.tracks.len(), 1);
        assert_eq!(catalog.tracks[0].coded.codec, "h264");
        assert!(catalog.bounded);
    }

    #[test]
    fn opening_a_track_this_module_does_not_publish_is_refused_by_name() {
        let err = subscribed(&[1]).expect_err("this module publishes one track");
        assert!(err.contains("track 1"), "{err}");
        assert!(err.contains("publishes 1 track"), "{err}");
    }

    #[test]
    fn opening_no_track_answers_an_empty_catalog() {
        assert!(subscribed(&[])
            .expect("nothing is asked for")
            .tracks
            .is_empty());
    }

    #[test]
    fn the_leading_none_dts_count_is_the_decode_delay_the_stream_needs() {
        let leading_none = PACKET_TABLE
            .iter()
            .take_while(|(_, dts, _)| dts.is_none())
            .count();
        assert_eq!(leading_none, 2, "an I-P-B-B GOP needs a reorder depth of 2");
    }

    #[test]
    fn every_settled_dts_is_non_decreasing() {
        let settled: Vec<i64> = PACKET_TABLE.iter().filter_map(|(_, dts, _)| *dts).collect();
        assert!(settled.windows(2).all(|pair| pair[0] <= pair[1]));
    }

    #[test]
    fn exactly_one_packet_is_a_keyframe() {
        assert_eq!(PACKET_TABLE.iter().filter(|(_, _, key)| *key).count(), 1);
    }

    #[test]
    fn every_packets_bytes_start_with_its_own_annex_b_start_code() {
        // libx264 writes a 3-byte start code (00 00 01) or a 4-byte one (00
        // 00 00 01) depending on the NAL - both are legal Annex-B.
        for (index, expected_len) in PACKET_LENS.iter().enumerate() {
            let bytes = packet_bytes(index);
            assert_eq!(bytes.len(), *expected_len);
            let starts_with_code = bytes.starts_with(&[0x00, 0x00, 0x01])
                || bytes.starts_with(&[0x00, 0x00, 0x00, 0x01]);
            assert!(
                starts_with_code,
                "packet {index} does not start with a start code"
            );
        }
    }
}
