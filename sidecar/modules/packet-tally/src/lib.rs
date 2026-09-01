//! A packet sink reading SEVERAL encoded streams at once: one instance, one
//! pad per stream, and a tally per pad.
//!
//! Each pad's coded stream is named at init - its codec and its frame size -
//! and every call adds that pad's packets to its own counts. The final call
//! emits one row per pad: how many packets and bytes crossed, how many were
//! keyframes, and the geometry the pad opened with. Nothing pairs one pad's
//! packet with another's, which is what a sink over a rendition ladder needs.

wit_bindgen::generate!({
    path: "../../wit",
    world: "packet-sink-module",
});

use std::cell::RefCell;

use exports::ffrwd::av::packet_sink::{
    Arity, CodedFormat, Guest, InputStream, Meta, PacketSinkMeta, PadPackets, Processed,
};
use serde::Serialize;

const PARAMS_SCHEMA: &str = r#"{"type":"object","properties":{},"additionalProperties":false}"#;
const ROWS_SCHEMA: &str = r#"{"type":"object","properties":{"pad":{"type":"integer"},"codec":{"type":"string"},"width":{"type":"integer"},"height":{"type":"integer"},"packets":{"type":"integer"},"keyframes":{"type":"integer"},"bytes":{"type":"integer"}},"additionalProperties":false}"#;

/// One pad's tally, emitted once at the end.
#[derive(Serialize)]
struct PadRow {
    pad: u32,
    codec: String,
    width: u32,
    height: u32,
    packets: u64,
    keyframes: u64,
    bytes: u64,
}

struct Pad {
    codec: String,
    width: u32,
    height: u32,
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
            video: Arity::Many,
            audio: Arity::Zero,
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
                CodedFormat::Audio(_) => return Err("packet_tally reads video".into()),
            };
            pads.push(Pad {
                codec: stream.coded.codec.clone(),
                width,
                height,
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
                        codec: pad.codec.clone(),
                        width: pad.width,
                        height: pad.height,
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
