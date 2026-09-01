//! A packet sink: counts the encoded packets of a stream without decoding
//! one. Each group of pictures leaves as a row - how many packets it held and
//! how many bytes, at which timestamp it opened - and the final call adds a
//! summary: totals, keyframes, and whether the timestamps ever stepped
//! backwards, which is what a reordering stream does.

wit_bindgen::generate!({
    path: "../../wit",
    world: "packet-sink-module",
});

use std::cell::RefCell;

use exports::ffrwd::av::packet_sink::{
    Arity, Guest, InputStream, Meta, Packet, PacketSinkMeta, PadPackets, Processed,
};
use serde::Serialize;

const PARAMS_SCHEMA: &str = r#"{"type":"object","properties":{},"additionalProperties":false}"#;
/// One schema covers both row shapes: a group row carries `gop`, the trailing
/// summary carries `keyframes`, and each leaves the other's fields out.
const ROWS_SCHEMA: &str = r#"{"type":"object","properties":{"gop":{"type":"integer"},"pts":{"type":"integer"},"packets":{"type":"integer"},"bytes":{"type":"integer"},"keyframes":{"type":"integer"},"gops":{"type":"integer"},"pts_monotonic":{"type":"boolean"}},"additionalProperties":false}"#;

/// One group of pictures, closed by the next keyframe or the end.
#[derive(Serialize)]
struct GroupRow {
    gop: u64,
    /// The timestamp the group opened at: its keyframe's, in the stream's
    /// time base.
    pts: i64,
    packets: u64,
    bytes: u64,
}

/// The trailing summary, once per stream.
#[derive(Serialize)]
struct SummaryRow {
    packets: u64,
    keyframes: u64,
    bytes: u64,
    gops: u64,
    /// Whether every packet's pts was at least the previous one's. False is
    /// a stream that reorders frames.
    pts_monotonic: bool,
}

/// The group currently open.
struct Group {
    index: u64,
    start_pts: i64,
    packets: u64,
    bytes: u64,
}

struct State {
    group: Option<Group>,
    packets: u64,
    keyframes: u64,
    bytes: u64,
    gops: u64,
    last_pts: Option<i64>,
    pts_monotonic: bool,
}

thread_local! {
    static STATE: RefCell<Option<State>> = const { RefCell::new(None) };
}

/// Validates that `params` is empty or `{}`; packet_stats takes no parameters.
fn validate_params(params: &str) -> Result<(), String> {
    match params.trim() {
        "" | "{}" => Ok(()),
        other => Err(format!("packet_stats takes no params, got: {other}")),
    }
}

/// Serializes a row; the schemas above are hand-kept in step.
fn row<T: Serialize>(value: &T) -> String {
    serde_json::to_string(value).expect("a stats row serializes")
}

impl State {
    /// Closes the open group into a row, if one is open.
    fn close_group(&mut self, rows: &mut Vec<String>) {
        if let Some(group) = self.group.take() {
            rows.push(row(&GroupRow {
                gop: group.index,
                pts: group.start_pts,
                packets: group.packets,
                bytes: group.bytes,
            }));
        }
    }

    fn count(&mut self, packet: &Packet, rows: &mut Vec<String>) {
        if packet.keyframe {
            self.keyframes += 1;
            self.close_group(rows);
        }
        let group = self.group.get_or_insert_with(|| {
            let index = self.gops;
            self.gops += 1;
            Group {
                index,
                start_pts: packet.pts,
                packets: 0,
                bytes: 0,
            }
        });
        group.packets += 1;
        group.bytes += packet.data.len() as u64;

        self.packets += 1;
        self.bytes += packet.data.len() as u64;
        if let Some(previous) = self.last_pts {
            if packet.pts < previous {
                self.pts_monotonic = false;
            }
        }
        self.last_pts = Some(packet.pts);
    }
}

struct PacketStats;

impl Guest for PacketStats {
    fn describe() -> PacketSinkMeta {
        PacketSinkMeta {
            meta: Meta {
                name: "packet_stats".to_string(),
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
            // One video stream: this module counts a stream, not a ladder.
            video: Arity::One,
            audio: Arity::Zero,
        }
    }

    fn init(_streams: Vec<InputStream>, params: String) -> Result<(), String> {
        validate_params(&params)?;
        STATE.with(|s| {
            *s.borrow_mut() = Some(State {
                group: None,
                packets: 0,
                keyframes: 0,
                bytes: 0,
                gops: 0,
                last_pts: None,
                pts_monotonic: true,
            });
        });
        Ok(())
    }

    fn set_params(params: String) -> Result<(), String> {
        validate_params(&params)
    }

    fn process(pads: Vec<PadPackets>, last: bool) -> Processed {
        STATE.with(|s| {
            let mut state_ref = s.borrow_mut();
            let state = state_ref.as_mut().expect("process called before init");

            let mut rows = Vec::new();
            for packet in pads.first().map(|p| p.packets.as_slice()).unwrap_or(&[]) {
                state.count(packet, &mut rows);
            }

            let mut trailing = Vec::new();
            if last {
                state.close_group(&mut rows);
                trailing.push(row(&SummaryRow {
                    packets: state.packets,
                    keyframes: state.keyframes,
                    bytes: state.bytes,
                    gops: state.gops,
                    pts_monotonic: state.pts_monotonic,
                }));
            }
            Processed { rows, trailing }
        })
    }
}

export!(PacketStats);

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

    fn fresh() -> State {
        State {
            group: None,
            packets: 0,
            keyframes: 0,
            bytes: 0,
            gops: 0,
            last_pts: None,
            pts_monotonic: true,
        }
    }

    #[test]
    fn a_keyframe_closes_the_group_before_it() {
        let mut state = fresh();
        let mut rows = Vec::new();
        state.count(&packet(0, true, 10), &mut rows);
        state.count(&packet(1, false, 5), &mut rows);
        state.count(&packet(2, true, 8), &mut rows);
        assert_eq!(rows, vec![r#"{"gop":0,"pts":0,"packets":2,"bytes":15}"#]);
        assert_eq!(state.gops, 2);
        assert_eq!(state.keyframes, 2);
    }

    #[test]
    fn a_backwards_pts_marks_the_stream_reordered() {
        let mut state = fresh();
        let mut rows = Vec::new();
        state.count(&packet(0, true, 1), &mut rows);
        state.count(&packet(2, false, 1), &mut rows);
        assert!(state.pts_monotonic);
        state.count(&packet(1, false, 1), &mut rows);
        assert!(!state.pts_monotonic);
    }

    #[test]
    fn a_stream_that_opens_without_a_keyframe_still_has_a_group() {
        let mut state = fresh();
        let mut rows = Vec::new();
        state.count(&packet(7, false, 3), &mut rows);
        assert_eq!(state.gops, 1);
        assert_eq!(state.keyframes, 0);
        state.close_group(&mut rows);
        assert_eq!(rows, vec![r#"{"gop":0,"pts":7,"packets":1,"bytes":3}"#]);
    }
}
