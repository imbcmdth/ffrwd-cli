//! A packet sink that writes down each packet it was handed: when it is
//! presented, whether decoding can start there, how many bytes it carries,
//! and a vector that says which of eight buckets its position falls in. It
//! asks for the keyframes alone, so a host that can skip hands over a
//! fraction of the stream - and, since `wants` is a request, it also works
//! when it is handed all of it.
//!
//! The vector is the point of the column rather than of the numbers. Eight
//! components, unit length, so a query can compare one against a prompt's
//! the way a search over real embeddings does; what it means is only "this
//! packet, not that one".

wit_bindgen::generate!({
    path: "../../wit",
    world: "packet-sink-module",
});

use std::cell::RefCell;

use crate::ffrwd::av::types::Packet;
use exports::ffrwd::av::packet_sink::{
    Arity, Guest, InputStream, Meta, PacketSinkMeta, PadPackets, Processed, Wants,
};
use serde::Serialize;

const PARAMS_SCHEMA: &str = r#"{"type":"object","properties":{},"additionalProperties":false}"#;
/// `vector`'s two bounds are equal, which is what fixes its length for a
/// reader that has to type the column before it has a row.
const ROWS_SCHEMA: &str = r#"{"type":"object","properties":{"index":{"type":"integer"},"start_t":{"type":"number"},"keyframe":{"type":"boolean"},"bytes":{"type":"integer"},"vector":{"type":"array","items":{"type":"number"},"minItems":8,"maxItems":8}},"additionalProperties":false}"#;

/// The length `vector` always takes, matching the schema's bounds - and the
/// length `fauxlate`'s `embed_text` answers, so a query can score one of
/// these against a prompt without either side declaring the other's dims.
const DIMS: usize = 8;

/// One packet.
#[derive(Serialize)]
struct KeyRow {
    /// 1-based, counting the packets this module was handed rather than the
    /// packets the stream holds: a host that skipped some numbers what it
    /// did hand over.
    index: u64,
    /// Seconds, converted from the packet's pts through the stream's own
    /// time base, so nothing above this has to know the unit.
    start_t: f64,
    keyframe: bool,
    bytes: u64,
    vector: [f64; DIMS],
}

struct State {
    /// The stream's time base, kept as the pair it arrived as: a tick count
    /// times the numerator and then divided by the denominator is exact
    /// wherever the answer is, which multiplying by a precomputed
    /// seconds-per-tick is not.
    num: f64,
    den: f64,
    index: u64,
}

thread_local! {
    static STATE: RefCell<Option<State>> = const { RefCell::new(None) };
}

/// Validates that `params` is empty or `{}`; packet_keys takes no parameters.
fn validate_params(params: &str) -> Result<(), String> {
    match params.trim() {
        "" | "{}" => Ok(()),
        other => Err(format!("packet_keys takes no params, got: {other}")),
    }
}

/// Serializes a row; the schema above is hand-kept in step.
fn row<T: Serialize>(value: &T) -> String {
    serde_json::to_string(value).expect("a key row serializes")
}

impl State {
    fn write(&mut self, packet: &Packet) -> String {
        self.index += 1;
        let start_t = packet.pts as f64 * self.num / self.den;
        let bytes = packet.data.len() as u64;
        // One component set, the rest zero: unit length by construction, and
        // a different direction for each of eight consecutive packets, which
        // is all a query comparing it against a prompt needs of it.
        let mut vector = [0.0_f64; DIMS];
        vector[(self.index as usize - 1) % DIMS] = 1.0;
        row(&KeyRow {
            index: self.index,
            start_t,
            keyframe: packet.keyframe,
            bytes,
            vector,
        })
    }
}

/// `time_base` as a pair of floats, falling back on microseconds where the
/// host handed over a denominator that cannot divide.
fn ticks(num: i32, den: i32) -> (f64, f64) {
    if den == 0 {
        return (1.0, 1_000_000.0);
    }
    (num as f64, den as f64)
}

struct PacketKeys;

impl Guest for PacketKeys {
    fn describe() -> PacketSinkMeta {
        PacketSinkMeta {
            meta: Meta {
                name: "packet_keys".to_string(),
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
            // Reading a header and a length needs no codec knowledge.
            video_codecs: vec![],
            audio_codecs: vec![],
            video: Arity::One,
            audio: Arity::Zero,
            data: Arity::Zero,
            // What a writer puts on keyframes is all this reads.
            wants: Wants::Keyframes,
        }
    }

    fn init(streams: Vec<InputStream>, params: String) -> Result<(), String> {
        validate_params(&params)?;
        let time_base = streams
            .first()
            .map(|stream| stream.info.time_base)
            .ok_or_else(|| "packet_keys was opened on no stream".to_string())?;
        let (num, den) = ticks(time_base.num, time_base.den);
        STATE.with(|s| {
            *s.borrow_mut() = Some(State { num, den, index: 0 });
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
            let rows = pads
                .first()
                .map(|p| p.packets.as_slice())
                .unwrap_or(&[])
                .iter()
                .map(|packet| state.write(packet))
                .collect();
            Processed {
                rows,
                trailing: vec![],
            }
        })
    }
}

export!(PacketKeys);

#[cfg(test)]
mod tests {
    use super::*;

    fn packet(pts: i64, keyframe: bool, len: usize) -> Packet {
        Packet {
            pts,
            dts: Some(pts),
            duration: None,
            keyframe,
            data: vec![0; len],
        }
    }

    #[test]
    fn a_row_counts_what_it_was_handed_and_times_it_in_seconds() {
        let (num, den) = ticks(1, 1000);
        let mut state = State { num, den, index: 0 };
        assert_eq!(
            state.write(&packet(500, true, 3)),
            r#"{"index":1,"start_t":0.5,"keyframe":true,"bytes":3,"vector":[1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]}"#
        );
        assert_eq!(
            state.write(&packet(1500, false, 2)),
            r#"{"index":2,"start_t":1.5,"keyframe":false,"bytes":2,"vector":[0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0]}"#
        );
    }

    #[test]
    fn the_ninth_packet_points_the_way_the_first_did() {
        let (num, den) = ticks(1, 1000);
        let mut state = State { num, den, index: 8 };
        let ninth = state.write(&packet(0, true, 1));
        assert!(
            ninth.ends_with(r#""vector":[1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]}"#),
            "{ninth}"
        );
    }

    #[test]
    fn a_time_base_that_cannot_divide_falls_back_on_microseconds() {
        assert_eq!(ticks(1, 0), (1.0, 1_000_000.0));
    }
}
