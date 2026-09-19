//! The identity packet filter: every packet handed straight back, every
//! stream handed back as it arrived.
//!
//! It is what a byte-for-byte test is written against - what leaves this
//! module is what entered it, so anything the wire loses is the host's or
//! the container's - and it is the smallest thing a packet filter can be,
//! so it is also the worked example of the interface. It counts what
//! crossed and says so once, at the end.

wit_bindgen::generate!({
    path: "../../wit",
    world: "packet-filter-module",
});

use std::cell::RefCell;

use exports::ffrwd::av::packet_filter::{
    Arity, CodedStream, Filtered, Guest, InputStream, Meta, PacketFilterMeta, PadPackets,
};
use serde::Serialize;

const PARAMS_SCHEMA: &str = r#"{"type":"object","properties":{},"additionalProperties":false}"#;
const ROWS_SCHEMA: &str = r#"{"type":"object","properties":{"pad":{"type":"integer"},"packets":{"type":"integer"},"bytes":{"type":"integer"}},"additionalProperties":false}"#;

/// One pad's tally, emitted once at the end.
#[derive(Serialize)]
struct PadRow {
    pad: u32,
    packets: u64,
    bytes: u64,
}

thread_local! {
    static PADS: RefCell<Vec<PadRow>> = const { RefCell::new(Vec::new()) };
}

/// Validates that `params` is empty or `{}`; this filter takes no parameters.
fn validate_params(params: &str) -> Result<(), String> {
    match params.trim() {
        "" | "{}" => Ok(()),
        other => Err(format!("packet_passthrough takes no params, got: {other}")),
    }
}

struct PacketPassthrough;

impl Guest for PacketPassthrough {
    fn describe() -> PacketFilterMeta {
        PacketFilterMeta {
            meta: Meta {
                name: "packet_passthrough".to_string(),
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
            // Handing bytes back needs no codec knowledge, so every codec is
            // accepted.
            video_codecs: vec![],
            audio_codecs: vec![],
            video: Arity::Any,
            audio: Arity::Any,
            // Rows pass nowhere: this filter rewrites nothing.
            reads_rows: false,
        }
    }

    fn init(streams: Vec<InputStream>, params: String) -> Result<Vec<CodedStream>, String> {
        validate_params(&params)?;
        if streams.is_empty() {
            return Err("packet_passthrough reads at least one stream".into());
        }
        PADS.with(|p| {
            *p.borrow_mut() = (0..streams.len())
                .map(|pad| PadRow {
                    pad: pad as u32,
                    packets: 0,
                    bytes: 0,
                })
                .collect()
        });
        // The streams leaving are the streams that arrived, headers and all.
        Ok(streams.into_iter().map(|s| s.coded).collect())
    }

    fn set_params(params: String) -> Result<(), String> {
        validate_params(&params)
    }

    fn process(pads: Vec<PadPackets>, _rows: Vec<String>, last: bool) -> Filtered {
        PADS.with(|held| {
            let mut state = held.borrow_mut();
            for (index, carried) in pads.iter().enumerate() {
                let Some(pad) = state.get_mut(index) else {
                    continue;
                };
                for packet in &carried.packets {
                    pad.packets += 1;
                    pad.bytes += packet.data.len() as u64;
                }
            }
            let mut trailing = Vec::new();
            if last {
                for pad in state.iter() {
                    trailing.push(serde_json::to_string(pad).expect("a pad row serializes"));
                }
            }
            Filtered {
                // Nothing is held back, so what arrived leaves on the same
                // call, in the order it arrived.
                pads,
                rows: vec![],
                trailing,
            }
        })
    }
}

export!(PacketPassthrough);
