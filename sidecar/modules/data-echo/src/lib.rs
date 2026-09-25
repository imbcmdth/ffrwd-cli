//! A packet sink that says back every message a data pad carried: one row
//! per message, with the pad, the pts, dts and keyframe flag it arrived
//! with, and the message's own bytes as a string. Video and audio pads are
//! read and counted in nothing.
//!
//! It is what a data stream's round trip is checked against at the far end:
//! what the rows say is exactly what reached the sink.

wit_bindgen::generate!({
    path: "../../wit",
    world: "packet-sink-module",
});

use std::cell::RefCell;

use crate::ffrwd::av::types::CodedFormat;
use exports::ffrwd::av::packet_sink::{
    Arity, Guest, InputStream, Meta, PacketSinkMeta, PadPackets, Processed, Wants,
};
use serde::Serialize;

const PARAMS_SCHEMA: &str = r#"{"type":"object","properties":{},"additionalProperties":false}"#;
const ROWS_SCHEMA: &str = r#"{"type":"object","properties":{"pad":{"type":"integer"},"codec":{"type":"string"},"pts":{"type":"integer"},"dts":{"type":["integer","null"]},"keyframe":{"type":"boolean"},"message":{"type":"string"}},"additionalProperties":false}"#;

/// One message, said back.
#[derive(Serialize)]
struct Echo<'a> {
    pad: u32,
    codec: &'a str,
    pts: i64,
    dts: Option<i64>,
    keyframe: bool,
    message: String,
}

thread_local! {
    /// Each pad's codec, or None for a pad that is not a data stream.
    static PADS: RefCell<Vec<Option<String>>> = const { RefCell::new(Vec::new()) };
}

fn validate_params(params: &str) -> Result<(), String> {
    match params.trim() {
        "" | "{}" => Ok(()),
        other => Err(format!("data_echo takes no params, got: {other}")),
    }
}

struct DataEcho;

impl Guest for DataEcho {
    fn describe() -> PacketSinkMeta {
        PacketSinkMeta {
            meta: Meta {
                name: "data_echo".to_string(),
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
            data: Arity::Many,
            wants: Wants::All,
        }
    }

    fn init(streams: Vec<InputStream>, params: String) -> Result<(), String> {
        validate_params(&params)?;
        PADS.with(|p| {
            *p.borrow_mut() = streams
                .into_iter()
                .map(|s| match s.coded.format {
                    CodedFormat::Data => Some(s.coded.codec),
                    _ => None,
                })
                .collect()
        });
        Ok(())
    }

    fn set_params(params: String) -> Result<(), String> {
        validate_params(&params)
    }

    fn process(pads: Vec<PadPackets>, _last: bool) -> Processed {
        PADS.with(|held| {
            let codecs = held.borrow();
            let mut rows = Vec::new();
            for (pad, carried) in pads.iter().enumerate() {
                let Some(Some(codec)) = codecs.get(pad) else {
                    continue;
                };
                for packet in &carried.packets {
                    let echo = Echo {
                        pad: pad as u32,
                        codec,
                        pts: packet.pts,
                        dts: packet.dts,
                        keyframe: packet.keyframe,
                        message: String::from_utf8_lossy(&packet.data).into_owned(),
                    };
                    rows.push(serde_json::to_string(&echo).expect("an echo serializes"));
                }
            }
            Processed {
                rows,
                trailing: vec![],
            }
        })
    }
}

export!(DataEcho);
