//! `packet_passthrough`, built against the vendored `ffrwd:av@0.16.0` world
//! instead of the current one: the world packet filters arrived in, before
//! a filter said how many data streams it reads. It is the module the host's
//! 0.16.0 packet-filter arm is proven against - a filter built before the
//! 0.17.0 bump still loads and hands its packets back untouched - and the one
//! a data stream is refused at, naming the world it was built for.
//!
//! It is `packet-passthrough` line for line but for the world and the name.

wit_bindgen::generate!({
    path: "../../worlds/0.16.0",
    world: "packet-filter-module",
});

use std::cell::RefCell;

use exports::ffrwd::av::packet_filter::{
    Arity, CodedStream, Filtered, Guest, InputStream, Meta, PacketFilterMeta, PadPackets,
};
use serde::Serialize;

const PARAMS_SCHEMA: &str = r#"{"type":"object","properties":{},"additionalProperties":false}"#;
const ROWS_SCHEMA: &str = r#"{"type":"object","properties":{"pad":{"type":"integer"},"packets":{"type":"integer"},"bytes":{"type":"integer"},"decode_delay":{"type":"integer"},"last_packets":{"type":"integer"}},"additionalProperties":false}"#;

/// One pad's tally, emitted once at the end.
#[derive(Serialize)]
struct PadRow {
    pad: u32,
    packets: u64,
    bytes: u64,
    /// The reorder depth `init` was told, straight back out: what says the
    /// host handed the wire's own bound to the module.
    decode_delay: u32,
    /// How many packets rode the FINAL call. A host that ended the run with
    /// an empty call would leave this 0, and a filter holding anything back
    /// would have nowhere to put it.
    last_packets: u64,
}

thread_local! {
    static PADS: RefCell<Vec<PadRow>> = const { RefCell::new(Vec::new()) };
}

/// Validates that `params` is empty or `{}`; this filter takes no parameters.
fn validate_params(params: &str) -> Result<(), String> {
    match params.trim() {
        "" | "{}" => Ok(()),
        other => Err(format!(
            "packet_passthrough_0160 takes no params, got: {other}"
        )),
    }
}

struct PacketPassthrough0160;

impl Guest for PacketPassthrough0160 {
    fn describe() -> PacketFilterMeta {
        PacketFilterMeta {
            meta: Meta {
                name: "packet_passthrough_0160".to_string(),
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
            return Err("packet_passthrough_0160 reads at least one stream".into());
        }
        PADS.with(|p| {
            *p.borrow_mut() = streams
                .iter()
                .enumerate()
                .map(|(pad, stream)| PadRow {
                    pad: pad as u32,
                    packets: 0,
                    bytes: 0,
                    decode_delay: stream.decode_delay,
                    last_packets: 0,
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
                if last {
                    pad.last_packets = carried.packets.len() as u64;
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

export!(PacketPassthrough0160);
