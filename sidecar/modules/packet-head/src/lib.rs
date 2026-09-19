//! A packet sink that reads what a stream says about itself and stops: its
//! codec, the size of its out-of-band header, and when its first packet is
//! presented. One row, from the first packet alone, so it asks for `first`
//! and works the same when a host hands it the whole stream anyway.

wit_bindgen::generate!({
    path: "../../wit",
    world: "packet-sink-module",
});

use exports::ffrwd::av::packet_sink::{
    Arity, Guest, InputStream, Meta, PacketSinkMeta, PadPackets, Processed, Wants,
};
use serde::Serialize;
use std::cell::RefCell;

const PARAMS_SCHEMA: &str = r#"{"type":"object","properties":{},"additionalProperties":false}"#;
const ROWS_SCHEMA: &str = r#"{"type":"object","properties":{"codec":{"type":"string"},"start_t":{"type":"number"},"extradata":{"type":"integer"}},"additionalProperties":false}"#;

/// The head of the stream, once.
#[derive(Serialize)]
struct HeadRow {
    codec: String,
    /// Seconds: the first packet's pts through the stream's own time base.
    start_t: f64,
    /// How many bytes of out-of-band header the stream declared.
    extradata: u64,
}

struct State {
    codec: String,
    num: f64,
    den: f64,
    extradata: u64,
    written: bool,
}

thread_local! {
    static STATE: RefCell<Option<State>> = const { RefCell::new(None) };
}

/// Validates that `params` is empty or `{}`; packet_head takes no parameters.
fn validate_params(params: &str) -> Result<(), String> {
    match params.trim() {
        "" | "{}" => Ok(()),
        other => Err(format!("packet_head takes no params, got: {other}")),
    }
}

struct PacketHead;

impl Guest for PacketHead {
    fn describe() -> PacketSinkMeta {
        PacketSinkMeta {
            meta: Meta {
                name: "packet_head".to_string(),
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
            video: Arity::One,
            audio: Arity::Zero,
            // What the stream declares about itself is on the first packet.
            wants: Wants::First,
        }
    }

    fn init(streams: Vec<InputStream>, params: String) -> Result<(), String> {
        validate_params(&params)?;
        let stream = streams
            .first()
            .ok_or_else(|| "packet_head was opened on no stream".to_string())?;
        let (num, den) = if stream.info.time_base.den == 0 {
            (1.0, 1_000_000.0)
        } else {
            (
                stream.info.time_base.num as f64,
                stream.info.time_base.den as f64,
            )
        };
        let codec = stream.coded.codec.clone();
        let extradata = stream.coded.extradata.len() as u64;
        STATE.with(|s| {
            *s.borrow_mut() = Some(State {
                codec,
                num,
                den,
                extradata,
                written: false,
            });
        });
        Ok(())
    }

    fn set_params(params: String) -> Result<(), String> {
        validate_params(&params)
    }

    fn process(pads: Vec<PadPackets>, _last: bool) -> Processed {
        STATE.with(|s| {
            let mut state_ref = s.borrow_mut();
            let state = state_ref.as_mut().expect("process called before init");
            let first = pads.first().and_then(|pad| pad.packets.first());
            let mut rows = Vec::new();
            if let Some(packet) = first {
                if !state.written {
                    state.written = true;
                    rows.push(
                        serde_json::to_string(&HeadRow {
                            codec: state.codec.clone(),
                            start_t: packet.pts as f64 * state.num / state.den,
                            extradata: state.extradata,
                        })
                        .expect("a head row serializes"),
                    );
                }
            }
            Processed {
                rows,
                trailing: vec![],
            }
        })
    }
}

export!(PacketHead);
