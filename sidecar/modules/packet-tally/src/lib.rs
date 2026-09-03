//! A packet sink reading SEVERAL encoded streams at once: one instance, one
//! pad per stream, and a tally per pad.
//!
//! Each pad's coded stream is named at init - its codec, its frame size (0
//! for an audio pad), its extradata length, and the relation row and
//! rendition name `-pad` set on it - and every call adds that pad's packets
//! to its own counts. The final call emits one row per pad: how many packets
//! and bytes crossed, how many were keyframes, the geometry and extradata
//! length the pad opened with, and its row and rendition. Nothing pairs one
//! pad's packet with another's, which is what a sink over a rendition ladder
//! needs.

wit_bindgen::generate!({
    path: "../../wit",
    world: "packet-sink-module",
});

use std::cell::RefCell;

use crate::ffrwd::av::types::CodedFormat;
use exports::ffrwd::av::packet_sink::{
    Arity, Guest, InputStream, Meta, PacketSinkMeta, PadPackets, Processed,
};
use serde::Serialize;

const PARAMS_SCHEMA: &str = r#"{"type":"object","properties":{},"additionalProperties":false}"#;
const ROWS_SCHEMA: &str = r#"{"type":"object","properties":{"pad":{"type":"integer"},"row":{"type":"integer"},"rendition":{"type":"string"},"codec":{"type":"string"},"width":{"type":"integer"},"height":{"type":"integer"},"extradata":{"type":"integer"},"packets":{"type":"integer"},"keyframes":{"type":"integer"},"bytes":{"type":"integer"}},"additionalProperties":false}"#;

/// One pad's tally, emitted once at the end.
#[derive(Serialize)]
struct PadRow {
    pad: u32,
    /// The relation row this pad opened for, and the rendition name it was
    /// told, exactly as `-pad` set them on `input-stream`.
    row: u32,
    rendition: Option<String>,
    codec: String,
    width: u32,
    height: u32,
    /// Byte length of the codec's out-of-band header, e.g. h264's SPS/PPS or
    /// aac's AudioSpecificConfig. 0 where the stream carries none.
    extradata: u32,
    packets: u64,
    keyframes: u64,
    bytes: u64,
}

struct Pad {
    row: u32,
    rendition: Option<String>,
    codec: String,
    width: u32,
    height: u32,
    extradata: u32,
    packets: u64,
    keyframes: u64,
    bytes: u64,
}

thread_local! {
    static PADS: RefCell<Vec<Pad>> = const { RefCell::new(Vec::new()) };
}

/// Validates that `params` is empty or `{}`; packet_tally takes no parameters.
fn validate_params(params: &str) -> Result<(), String> {
    match params.trim() {
        "" | "{}" => Ok(()),
        other => Err(format!("packet_tally takes no params, got: {other}")),
    }
}

struct PacketTally;

impl Guest for PacketTally {
    fn describe() -> PacketSinkMeta {
        PacketSinkMeta {
            meta: Meta {
                name: "packet_tally".to_string(),
                version: "0.1.0".to_string(),
                params_schema: PARAMS_SCHEMA.to_string(),
                rows_schema: ROWS_SCHEMA.to_string(),
                // No decoded payload ever arrives, so no format list fills in.
                pixel_formats: vec![],
                sample_formats: vec![],
                sample_rates: vec![],
                channel_counts: vec![],
                rows_language: vec![],
            },
            // Counting needs no codec knowledge, so every codec is accepted.
            video_codecs: vec![],
            audio_codecs: vec![],
            video: Arity::Any,
            audio: Arity::Any,
        }
    }

    fn init(streams: Vec<InputStream>, params: String) -> Result<(), String> {
        validate_params(&params)?;
        if streams.is_empty() {
            return Err("packet_tally reads at least one stream".into());
        }
        let mut pads = Vec::with_capacity(streams.len());
        for stream in &streams {
            let (width, height) = match &stream.coded.format {
                CodedFormat::Video(video) => (video.width, video.height),
                // Audio carries no frame geometry; the row's width and
                // height stay 0 rather than a video pad's borrowed value.
                CodedFormat::Audio(_) => (0, 0),
            };
            pads.push(Pad {
                row: stream.row,
                rendition: stream.rendition.name.clone(),
                codec: stream.coded.codec.clone(),
                width,
                height,
                extradata: stream.coded.extradata.len() as u32,
                packets: 0,
                keyframes: 0,
                bytes: 0,
            });
        }
        PADS.with(|p| *p.borrow_mut() = pads);
        Ok(())
    }

    fn set_params(params: String) -> Result<(), String> {
        validate_params(&params)
    }

    fn process(pads: Vec<PadPackets>, last: bool) -> Processed {
        PADS.with(|held| {
            let mut state = held.borrow_mut();
            for (index, carried) in pads.iter().enumerate() {
                let Some(pad) = state.get_mut(index) else {
                    continue;
                };
                for packet in &carried.packets {
                    pad.packets += 1;
                    pad.bytes += packet.data.len() as u64;
                    if packet.keyframe {
                        pad.keyframes += 1;
                    }
                }
            }
            let mut trailing = Vec::new();
            if last {
                for (index, pad) in state.iter().enumerate() {
                    let row = PadRow {
                        pad: index as u32,
                        row: pad.row,
                        rendition: pad.rendition.clone(),
                        codec: pad.codec.clone(),
                        width: pad.width,
                        height: pad.height,
                        extradata: pad.extradata,
                        packets: pad.packets,
                        keyframes: pad.keyframes,
                        bytes: pad.bytes,
                    };
                    trailing.push(serde_json::to_string(&row).expect("a pad row serializes"));
                }
            }
            Processed {
                rows: vec![],
                trailing,
            }
        })
    }
}

export!(PacketTally);
