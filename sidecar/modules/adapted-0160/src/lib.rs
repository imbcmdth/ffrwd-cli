//! A packet sink of the world before a sink said how many data streams it
//! reads, so the adapter carrying that world is exercised by a module
//! actually shaped that way.
//!
//! One trailing row naming the codec each pad was opened for and how many
//! packets crossed it. It asks for keyframes alone, which only a world from
//! 0.16.0 on can say, so the description proves the field is read through.

wit_bindgen::generate!({
    path: "../../worlds/0.16.0",
    world: "packet-sink-module",
});

use std::cell::RefCell;

use exports::ffrwd::av::packet_sink::{
    Arity, Guest, InputStream, Meta, PacketSinkMeta, PadPackets, Processed, Wants,
};

const PARAMS_SCHEMA: &str = r#"{"type":"object","properties":{},"additionalProperties":false}"#;
const ROWS_SCHEMA: &str = r#"{"type":"object","properties":{"codec":{"type":"string"},"packets":{"type":"integer"}},"additionalProperties":false}"#;

thread_local! {
    static PADS: RefCell<Vec<(String, u64)>> = const { RefCell::new(Vec::new()) };
}

struct Adapted0160;

impl Guest for Adapted0160 {
    fn describe() -> PacketSinkMeta {
        PacketSinkMeta {
            meta: Meta {
                name: "adapted_0160".to_string(),
                version: "0.1.0".to_string(),
                params_schema: PARAMS_SCHEMA.to_string(),
                rows_schema: ROWS_SCHEMA.to_string(),
                pixel_formats: vec![],
                sample_formats: vec![],
                sample_rates: vec![],
                channel_counts: vec![],
                rows_language: vec![],
            },
            video_codecs: vec![],
            audio_codecs: vec![],
            video: Arity::Any,
            audio: Arity::Any,
            wants: Wants::Keyframes,
        }
    }

    fn init(streams: Vec<InputStream>, _params: String) -> Result<(), String> {
        PADS.with(|p| {
            *p.borrow_mut() = streams.into_iter().map(|s| (s.coded.codec, 0u64)).collect()
        });
        Ok(())
    }

    fn set_params(_params: String) -> Result<(), String> {
        Ok(())
    }

    fn process(pads: Vec<PadPackets>, last: bool) -> Processed {
        PADS.with(|held| {
            let mut state = held.borrow_mut();
            for (index, carried) in pads.iter().enumerate() {
                if let Some(pad) = state.get_mut(index) {
                    pad.1 += carried.packets.len() as u64;
                }
            }
            let mut trailing = Vec::new();
            if last {
                for (codec, packets) in state.iter() {
                    trailing.push(
                        serde_json::json!({ "codec": codec, "packets": packets }).to_string(),
                    );
                }
            }
            Processed {
                rows: vec![],
                trailing,
            }
        })
    }
}

export!(Adapted0160);
