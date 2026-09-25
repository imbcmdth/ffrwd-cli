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
//!
//! `{"turns":true}` also emits a row for every call that carries no packets
//! and is not the last: how many such calls there have been, and how many
//! packets had arrived by then.

wit_bindgen::generate!({
    path: "../../wit",
    world: "packet-sink-module",
});

use std::cell::{Cell, RefCell};

use crate::ffrwd::av::types::CodedFormat;
use exports::ffrwd::av::packet_sink::{
    Arity, Guest, InputStream, Meta, PacketSinkMeta, PadPackets, Processed, Wants,
};
use serde::Serialize;

const PARAMS_SCHEMA: &str = r#"{"type":"object","properties":{"turns":{"type":"boolean","default":false,"description":"a row for every call that carries no packets"}},"additionalProperties":false}"#;
const ROWS_SCHEMA: &str = r#"{"type":"object","properties":{"turns":{"type":"integer"},"pad":{"type":"integer"},"row":{"type":"integer"},"rendition":{"type":"string"},"codec":{"type":"string"},"width":{"type":"integer"},"height":{"type":"integer"},"extradata":{"type":"integer"},"packets":{"type":"integer"},"keyframes":{"type":"integer"},"bytes":{"type":"integer"}},"additionalProperties":false}"#;

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

/// A call that carried no packets, when `{"turns":true}` asks for them.
#[derive(Serialize)]
struct TurnRow {
    turns: u64,
    packets: u64,
}

thread_local! {
    static PADS: RefCell<Vec<Pad>> = const { RefCell::new(Vec::new()) };
    /// How many calls have carried no packets, where they are counted.
    static TURNS: Cell<Option<u64>> = const { Cell::new(None) };
}

/// Whether `params` asks for the turns to be counted: `{}` or
/// `{"turns":true}`.
fn validate_params(params: &str) -> Result<bool, String> {
    let spelled: String = params.chars().filter(|c| !c.is_whitespace()).collect();
    match spelled.as_str() {
        "" | "{}" | r#"{"turns":false}"# => Ok(false),
        r#"{"turns":true}"# => Ok(true),
        _ => Err(format!(
            "packet_tally takes {{\"turns\":true}} or nothing, got: {}",
            params.trim()
        )),
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
            data: Arity::Any,
            // Every packet is the point: these count what crossed.
            wants: Wants::All,
        }
    }

    fn init(streams: Vec<InputStream>, params: String) -> Result<(), String> {
        let turns = validate_params(&params)?;
        TURNS.with(|t| t.set(turns.then_some(0)));
        if streams.is_empty() {
            return Err("packet_tally reads at least one stream".into());
        }
        let mut pads = Vec::with_capacity(streams.len());
        for stream in &streams {
            let (width, height) = match &stream.coded.format {
                CodedFormat::Video(video) => (video.width, video.height),
                // Audio and data carry no frame geometry; the row's width
                // and height stay 0 rather than a video pad's borrowed value.
                CodedFormat::Audio(_) | CodedFormat::Data => (0, 0),
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
        validate_params(&params).map(|_| ())
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
            let mut rows = Vec::new();
            let counted = TURNS.with(Cell::get);
            if let Some(turns) =
                counted.filter(|_| !last && pads.iter().all(|carried| carried.packets.is_empty()))
            {
                TURNS.with(|t| t.set(Some(turns + 1)));
                let row = TurnRow {
                    turns: turns + 1,
                    packets: state.iter().map(|pad| pad.packets).sum(),
                };
                rows.push(serde_json::to_string(&row).expect("a turn row serializes"));
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
            Processed { rows, trailing }
        })
    }
}

export!(PacketTally);
