//! A packet sink of the world before a sink read several streams, so the
//! adapter carrying that world is exercised by a module actually shaped that
//! way.
//!
//! One coded stream, one packet list, and a single trailing row naming the
//! codec it was opened for and how many packets crossed.

wit_bindgen::generate!({
    path: "../../worlds/0.11.0",
    world: "packet-module",
});

use std::cell::RefCell;

use exports::ffrwd::av::packet_sink::{
    CodedStream, Guest, Meta, Packet, PacketSinkMeta, Processed, StreamInfo,
};

const PARAMS_SCHEMA: &str = r#"{"type":"object","properties":{},"additionalProperties":false}"#;
const ROWS_SCHEMA: &str = r#"{"type":"object","properties":{"codec":{"type":"string"},"packets":{"type":"integer"}},"additionalProperties":false}"#;

thread_local! {
    static STATE: RefCell<(String, u64)> = const { RefCell::new((String::new(), 0)) };
}

struct Adapted0110;

impl Guest for Adapted0110 {
    fn describe() -> PacketSinkMeta {
        PacketSinkMeta {
            meta: Meta {
                name: "adapted_0110".to_string(),
                version: "0.1.0".to_string(),
                params_schema: PARAMS_SCHEMA.to_string(),
                rows_schema: ROWS_SCHEMA.to_string(),
                pixel_formats: vec![],
                sample_formats: vec![],
                sample_rates: vec![],
                channel_counts: vec![],
                rows_language: vec![],
            },
            codecs: vec![],
        }
    }

    fn init(
        coded_stream: CodedStream,
        _stream_info: StreamInfo,
        _params: String,
    ) -> Result<(), String> {
        STATE.with(|s| *s.borrow_mut() = (coded_stream.codec, 0));
        Ok(())
    }

    fn set_params(_params: String) -> Result<(), String> {
        Ok(())
    }

    fn process(packets: Vec<Packet>, last: bool) -> Processed {
        STATE.with(|s| {
            let mut state = s.borrow_mut();
            state.1 += packets.len() as u64;
            let mut trailing = Vec::new();
            if last {
                trailing
                    .push(serde_json::json!({ "codec": state.0, "packets": state.1 }).to_string());
            }
            Processed {
                rows: vec![],
                trailing,
            }
        })
    }
}

export!(Adapted0110);
