//! `source_replay` with a second track, for the shape one sidecar feeding
//! TWO pipes into one reader takes: track 0 pours out more than a buffer's
//! worth before track 1 says anything at all. A reader opens its inputs one
//! at a time, so a host that wrote track 0's packets before track 1's header
//! wedges on exactly this - which is what the two-output test measures.
//!
//! Both tracks replay the coded h264 `source_replay` compiles in (see that
//! module for why the bytes are compiled in rather than read from a path);
//! the pair is a two-rendition catalog, the shape a live ladder publishes.

wit_bindgen::generate!({
    path: "../../wit",
    world: "packet-source-module",
});

use std::cell::Cell;

use crate::ffrwd::av::types::{CodedFormat, CodedStream, CodedVideo, Packet, Rational};
use exports::ffrwd::av::packet_source::{
    Catalog, Guest, Meta, PadPackets, RenditionMeta, SourceTrack, StreamInfo,
};

mod generated {
    include!("../../source-replay/src/generated/packets.rs");
}
use generated::{EXTRADATA_LEN, HEIGHT, LEVEL, PACKET_LENS, PACKET_TABLE, PROFILE, RAW, WIDTH};

const PARAMS_SCHEMA: &str = r#"{"type":"object","properties":{},"additionalProperties":false}"#;

/// How many times track 0 replays the seven-packet table, one replay per
/// `next()`. Sized past the megabyte a buffered writer holds, so a host that
/// writes track 0's packets before track 1's header blocks on a full pipe
/// rather than reaching track 1 at all.
pub(crate) const REPEATS: usize = 1200;

fn extradata() -> &'static [u8] {
    &RAW[..EXTRADATA_LEN]
}

fn packet_bytes(index: usize) -> &'static [u8] {
    let start = EXTRADATA_LEN + PACKET_LENS[..index].iter().sum::<usize>();
    &RAW[start..start + PACKET_LENS[index]]
}

fn coded_stream() -> CodedStream {
    CodedStream {
        codec: "h264".to_string(),
        time_base: Rational { num: 1, den: 25 },
        format: CodedFormat::Video(CodedVideo {
            width: WIDTH,
            height: HEIGHT,
            sample_aspect_ratio: None,
            color: None,
        }),
        extradata: extradata().to_vec(),
        profile: Some(PROFILE),
        level: Some(LEVEL),
    }
}

/// One of the two tracks, named by the rendition it stands for.
fn track(index: u32, name: &str) -> SourceTrack {
    SourceTrack {
        coded: coded_stream(),
        info: StreamInfo {
            index,
            kind: "video".to_string(),
            codec: "h264".to_string(),
            duration: None,
            tags: vec![],
            time_base: Rational { num: 1, den: 25 },
        },
        row: index,
        rendition: RenditionMeta {
            name: Some(name.to_string()),
            bandwidth: None,
            codecs: None,
            language: None,
        },
    }
}

fn catalog() -> Catalog {
    Catalog {
        tracks: vec![track(0, "wide"), track(1, "late")],
        bounded: true,
    }
}

/// The catalog restricted to `tracks`, in that order.
fn subscribed(tracks: &[u32]) -> Result<Catalog, String> {
    let published = catalog().tracks;
    let mut subscribed = Vec::with_capacity(tracks.len());
    for index in tracks {
        let Some(found) = published.get(*index as usize) else {
            return Err(format!(
                "source_replay_pair publishes {} tracks, so track {index} is not one of them",
                published.len()
            ));
        };
        subscribed.push(found.clone());
    }
    Ok(Catalog {
        tracks: subscribed,
        bounded: true,
    })
}

fn validate_params(params: &str) -> Result<(), String> {
    match params.trim() {
        "" | "{}" => Ok(()),
        other => Err(format!("source_replay_pair takes no params, got: {other}")),
    }
}

/// Track 0's packet at `entry` of replay `repeat`. The presentation order the
/// table records repeats every seven packets; decode order runs straight
/// through, so the dts is unsettled only for the two packets the very first
/// replay opens with.
pub(crate) fn track0_packet(repeat: usize, entry: usize) -> (i64, Option<i64>, bool) {
    let (pts, _, keyframe) = PACKET_TABLE[entry];
    let base = (repeat * PACKET_TABLE.len()) as i64;
    let dts = (repeat > 0 || entry >= 2).then(|| base + entry as i64 - 2);
    (base + pts, dts, keyframe)
}

/// Track 1's one packet, which arrives only once track 0 is spent: a settled
/// dts, so its own decode_delay is nothing.
pub(crate) fn track1_packet() -> (i64, Option<i64>, bool) {
    (0, Some(0), true)
}

thread_local! {
    /// Which `next()` call comes round: `REPEATS` of track 0, then track 1's
    /// one packet, then the end.
    static CALL: Cell<usize> = const { Cell::new(0) };

    /// Which catalog track each pad carries, in `open`'s order.
    static PADS: Cell<[Option<u32>; 2]> = const { Cell::new([None, None]) };
}

fn packet_for(index: usize, table: (i64, Option<i64>, bool)) -> Packet {
    let (pts, dts, keyframe) = table;
    Packet {
        pts,
        dts,
        duration: None,
        keyframe,
        data: packet_bytes(index).to_vec(),
    }
}

struct SourceReplayPair;

impl Guest for SourceReplayPair {
    fn describe() -> Meta {
        Meta {
            name: "source_replay_pair".to_string(),
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
        CALL.with(|c| c.set(0));
        let mut pads = [None, None];
        for (pad, index) in tracks.iter().enumerate().take(pads.len()) {
            pads[pad] = Some(*index);
        }
        PADS.with(|c| c.set(pads));
        Ok(subscribed)
    }

    fn next() -> Result<Option<Vec<PadPackets>>, String> {
        let pads = PADS.with(|c| c.get());
        let call = CALL.with(|c| {
            let call = c.get();
            c.set(call + 1);
            call
        });
        if call > REPEATS {
            return Ok(None);
        }
        let mut answer = Vec::with_capacity(pads.len());
        for pad in pads.iter().flatten() {
            let packets = match (*pad, call) {
                (0, repeat) if repeat < REPEATS => (0..PACKET_TABLE.len())
                    .map(|entry| packet_for(entry, track0_packet(repeat, entry)))
                    .collect(),
                (1, call) if call == REPEATS => vec![packet_for(0, track1_packet())],
                _ => vec![],
            };
            answer.push(PadPackets { packets });
        }
        Ok(Some(answer))
    }
}

export!(SourceReplayPair);

#[cfg(test)]
mod tests {
    use super::{generated::PACKET_LENS, subscribed, track0_packet, track1_packet, REPEATS};

    #[test]
    fn opening_both_tracks_answers_a_catalog_of_both_in_that_order() {
        let catalog = subscribed(&[0, 1]).expect("both tracks are published");
        assert_eq!(catalog.tracks.len(), 2);
        assert_eq!(
            catalog.tracks[0].rendition.name.as_deref(),
            Some("wide"),
            "track 0 first"
        );
        assert_eq!(catalog.tracks[1].rendition.name.as_deref(), Some("late"));
    }

    #[test]
    fn opening_a_track_this_module_does_not_publish_is_refused_by_name() {
        let err = subscribed(&[2]).expect_err("this module publishes two tracks");
        assert!(err.contains("track 2"), "{err}");
        assert!(err.contains("publishes 2 tracks"), "{err}");
    }

    #[test]
    fn track_0_pours_out_more_than_a_buffered_writer_holds() {
        let coded: usize = PACKET_LENS.iter().sum::<usize>() * REPEATS;
        assert!(
            coded > (1 << 20) + (1 << 16),
            "track 0 carries {coded} bytes, not past a megabyte buffer and a full pipe behind it"
        );
    }

    #[test]
    fn only_the_first_two_packets_of_track_0_leave_their_dts_unsettled() {
        let unsettled: Vec<(usize, usize)> = (0..3)
            .flat_map(|repeat| (0..7).map(move |entry| (repeat, entry)))
            .filter(|(repeat, entry)| track0_packet(*repeat, *entry).1.is_none())
            .collect();
        assert_eq!(unsettled, vec![(0, 0), (0, 1)]);
    }

    #[test]
    fn track_0s_settled_dts_never_decreases_and_never_passes_its_pts() {
        let mut previous = i64::MIN;
        for repeat in 0..REPEATS {
            for entry in 0..7 {
                let (pts, dts, _) = track0_packet(repeat, entry);
                let Some(dts) = dts else { continue };
                assert!(dts >= previous, "dts {dts} after {previous}");
                assert!(dts <= pts, "dts {dts} past pts {pts}");
                previous = dts;
            }
        }
    }

    #[test]
    fn track_1s_one_packet_settles_its_dts_at_once() {
        let (pts, dts, keyframe) = track1_packet();
        assert_eq!((pts, dts), (0, Some(0)));
        assert!(
            keyframe,
            "a track opening on a non-keyframe decodes nothing"
        );
    }
}
