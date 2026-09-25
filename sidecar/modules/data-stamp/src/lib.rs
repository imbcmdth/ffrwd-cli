//! A data filter that stamps messages with the node that handled them, and
//! ticks on its clock.
//!
//! Every message on the call's first data pad is written back with a field
//! `"node": <node>` added, at the same time. With a clock pad and `every_s`
//! above 0 it also writes `{"kind":"tick","node":<node>}` each time `now`
//! crosses a multiple of `every_s` seconds, at that multiple. The output
//! counts microseconds. Nothing is held back: what a call brings, the call
//! writes, stamped messages and ticks merged in time order.

wit_bindgen::generate!({
    path: "../../wit",
    world: "data-filter-module",
});

use std::cell::RefCell;

use exports::ffrwd::av::data_filter::{
    DataFilterMeta, Guest, Message, Meta, PadInfo, PadKind, PadMessages, Processed, Rational,
};
use serde::Deserialize;

const PARAMS_SCHEMA: &str = r#"{"type":"object","properties":{"node":{"type":"string"},"every_s":{"type":"number","minimum":0,"default":0}},"required":["node"],"additionalProperties":false}"#;

/// Microseconds per second: the output's time base is 1/1000000.
const MICROS: i128 = 1_000_000;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Params {
    node: String,
    #[serde(default)]
    every_s: f64,
}

struct State {
    node: String,
    /// The tick interval in microseconds; 0 is no ticks.
    every_us: i64,
    /// The pad whose messages are stamped, and its time base.
    data: Option<(u32, Rational)>,
    /// The first clock pad's time base, which `now` counts in.
    clock: Option<Rational>,
    /// The last `now` seen, in microseconds.
    last_now_us: Option<i64>,
    /// The pts of the last message written, in microseconds: what a message
    /// that arrives after time has moved past it is written at instead.
    last_written_us: Option<i64>,
}

thread_local! {
    static STATE: RefCell<Option<State>> = const { RefCell::new(None) };
}

/// `pts` ticks of `time_base` in microseconds, rounded to the nearest.
fn to_micros(pts: i64, time_base: Rational) -> i64 {
    let num = i128::from(pts) * i128::from(time_base.num) * MICROS;
    let den = i128::from(time_base.den);
    let rounded = if num >= 0 {
        (num + den / 2) / den
    } else {
        (num - den / 2) / den
    };
    rounded as i64
}

/// The message with `"node": node` added, or an error naming what it is
/// instead of a JSON object. The rest of its bytes are left as written.
fn stamped(data: &[u8], node: &str) -> Result<Vec<u8>, String> {
    let text = std::str::from_utf8(data).map_err(|_| "a message is not UTF-8".to_string())?;
    let value: serde_json::Value =
        serde_json::from_str(text).map_err(|e| format!("a message is not JSON: {e}"))?;
    let object = value
        .as_object()
        .ok_or_else(|| "a message is not a JSON object".to_string())?;
    let node = serde_json::to_string(node).expect("a string serializes");
    if object.contains_key("node") {
        // Already stamped upstream: this node's name replaces it.
        let mut object = object.clone();
        object.insert(
            "node".to_string(),
            serde_json::from_str(&node).expect("valid"),
        );
        return Ok(serde_json::to_vec(&object).expect("an object serializes"));
    }
    let body = text.trim_end();
    let body = &body[..body.len() - 1];
    let separator = if object.is_empty() { "" } else { "," };
    Ok(format!("{body}{separator}\"node\":{node}}}").into_bytes())
}

struct DataStamp;

impl Guest for DataStamp {
    fn describe() -> DataFilterMeta {
        DataFilterMeta {
            meta: Meta {
                name: "data_stamp".to_string(),
                version: "0.1.0".to_string(),
                params_schema: PARAMS_SCHEMA.to_string(),
                rows_schema: String::new(),
                pixel_formats: vec![],
                sample_formats: vec![],
                sample_rates: vec![],
                channel_counts: vec![],
                rows_language: vec![],
            },
            outputs: vec!["json".to_string()],
            time_base: Rational {
                num: 1,
                den: MICROS as i32,
            },
        }
    }

    fn init(pads: Vec<PadInfo>, params: String) -> Result<(), String> {
        let params: Params =
            serde_json::from_str(&params).map_err(|e| format!("data_stamp params: {e}"))?;
        if !(params.every_s >= 0.0 && params.every_s.is_finite()) {
            return Err(format!(
                "data_stamp: every_s is {}, and a tick interval is 0 or more seconds",
                params.every_s
            ));
        }
        let data = pads
            .iter()
            .enumerate()
            .find(|(_, p)| p.kind == PadKind::Data)
            .map(|(index, p)| (index as u32, p.time_base));
        let clock = pads
            .iter()
            .find(|p| p.kind == PadKind::Clock)
            .map(|p| p.time_base);
        STATE.with(|s| {
            *s.borrow_mut() = Some(State {
                node: params.node,
                every_us: (params.every_s * MICROS as f64).round() as i64,
                data,
                clock,
                last_now_us: None,
                last_written_us: None,
            })
        });
        Ok(())
    }

    fn process(
        input: Vec<PadMessages>,
        now: Option<i64>,
        _last: bool,
    ) -> Result<Processed, String> {
        STATE.with(|s| {
            let mut guard = s.borrow_mut();
            let state = guard.as_mut().ok_or("data_stamp: process before init")?;
            // (pts, arrival order, message): sorted by time, ties in order.
            let mut written: Vec<(i64, usize, Vec<u8>)> = Vec::new();
            if let Some((pad, time_base)) = state.data {
                for carried in input.iter().filter(|p| p.pad == pad) {
                    for message in &carried.messages {
                        let order = written.len();
                        written.push((
                            to_micros(message.pts, time_base),
                            order,
                            stamped(&message.data, &state.node)?,
                        ));
                    }
                }
            }
            if let (Some(now), Some(clock)) = (now, state.clock) {
                let now_us = to_micros(now, clock);
                // Before the first call, `now` was just short of its first
                // value, so a clock starting on a multiple ticks there.
                let before = state.last_now_us.unwrap_or(now_us - 1);
                if state.every_us > 0 && now_us > before {
                    let tick = serde_json::json!({ "kind": "tick", "node": state.node });
                    let first = before.div_euclid(state.every_us) + 1;
                    let last = now_us.div_euclid(state.every_us);
                    for k in first..=last {
                        let order = written.len();
                        written.push((k * state.every_us, order, tick.to_string().into_bytes()));
                    }
                }
                state.last_now_us = Some(state.last_now_us.map_or(now_us, |t| t.max(now_us)));
            }
            written.sort_by_key(|(pts, order, _)| (*pts, *order));
            // An output's pts never go back: a message that reached this call
            // after the clock had moved past its own time is written now, at
            // the last time written, rather than behind it.
            for (pts, _, _) in &mut written {
                if let Some(floor) = state.last_written_us {
                    *pts = (*pts).max(floor);
                }
                state.last_written_us = Some(*pts);
            }
            Ok(Processed {
                outputs: vec![written
                    .into_iter()
                    .map(|(pts, _, data)| Message { pts, data })
                    .collect()],
                rows: vec![],
            })
        })
    }
}

export!(DataStamp);

#[cfg(test)]
mod tests {
    use super::{stamped, to_micros, Rational};

    #[test]
    fn a_stamp_is_added_without_respelling_the_rest() {
        assert_eq!(
            stamped(br#"{ "a" : 1 }"#, "n1").unwrap(),
            br#"{ "a" : 1 ,"node":"n1"}"#.to_vec()
        );
        assert_eq!(stamped(b"{}", "n1").unwrap(), br#"{"node":"n1"}"#.to_vec());
        assert!(stamped(b"[1]", "n1").is_err());
    }

    #[test]
    fn times_convert_to_microseconds() {
        assert_eq!(to_micros(3, Rational { num: 1, den: 25 }), 120_000);
        assert_eq!(to_micros(1, Rational { num: 1, den: 3 }), 333_333);
    }
}
