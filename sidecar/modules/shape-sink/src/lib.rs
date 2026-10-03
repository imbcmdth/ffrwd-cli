//! A sink of any number of pictures, taken as they arrive: it writes
//! nothing on any port, and on its last call says, as rows, how many frames
//! each stream brought and their first and last pts. Self-clocked, or with
//! `rate` a publisher's shape: a rate clock that only needs turns.

// Every shape module carries the same few helpers, and not every one uses
// each of them.
#![allow(dead_code, unused_imports)]

wit_bindgen::generate!({
    path: "../../wit",
    world: "node-module",
});

use std::cell::RefCell;

use exports::ffrwd::av::node::{Emission, Emitted, Guest, Payload};
use ffrwd::av::node_tick::Tick;
use ffrwd::av::node_types::{
    Accepts, BoundStream, Clock, InputPort, Message, NodeShape, OutputFormat, OutputPort, Pairing,
    PortKind, RowsUse,
};
use ffrwd::av::types::{Meta, Rational, Wants};
use serde::Deserialize;
use serde_json::json;

fn meta(name: &str, params_schema: &str) -> Meta {
    Meta {
        name: name.to_string(),
        version: "0.1.0".to_string(),
        params_schema: params_schema.to_string(),
        rows_schema: String::new(),
        pixel_formats: Vec::new(),
        sample_formats: Vec::new(),
        sample_rates: Vec::new(),
        channel_counts: Vec::new(),
        rows_language: Vec::new(),
    }
}

fn accepts() -> Accepts {
    Accepts {
        pixel_formats: Vec::new(),
        sample_formats: Vec::new(),
        sample_rates: Vec::new(),
        channel_counts: Vec::new(),
        codecs: Vec::new(),
        wants: Wants::All,
        like: None,
    }
}

fn input(name: &str, kind: PortKind, pairing: Pairing, rows: RowsUse) -> InputPort {
    InputPort {
        name: name.to_string(),
        kind,
        required: true,
        many: false,
        pairing,
        rows,
        window: 1,
        stride: 1,
        accepts: accepts(),
        schema: None,
    }
}

fn output(name: &str, kind: PortKind, format: Option<OutputFormat>) -> OutputPort {
    OutputPort {
        name: name.to_string(),
        kind,
        format,
        time_base: None,
        latency: 0.0,
        schema: None,
        row: None,
    }
}

fn data(name: &str) -> OutputPort {
    output(
        name,
        PortKind::Data,
        Some(OutputFormat::Data("json".to_string())),
    )
}

fn shape(inputs: Vec<InputPort>, outputs: Vec<OutputPort>, clock: Clock, pure: bool) -> NodeShape {
    NodeShape {
        inputs,
        outputs,
        clock,
        pure,
        one_to_one: false,
        bounded: true,
        relation: Vec::new(),
    }
}

fn parse<P: for<'a> Deserialize<'a> + Default>(text: &str) -> Result<P, String> {
    if text.trim().is_empty() {
        return Ok(P::default());
    }
    serde_json::from_str(text).map_err(|e| format!("params: {e}"))
}

fn message(port: &str, pts: i64, value: serde_json::Value) -> Emission {
    Emission {
        port: port.to_string(),
        payload: Payload::Message(Message {
            pts,
            data: value.to_string().into_bytes(),
        }),
    }
}

fn emitted(items: Vec<Emission>, finished: bool) -> Emitted {
    Emitted {
        items,
        rows: Vec::new(),
        finished,
    }
}

fn bound_ids(bound: &[BoundStream], port: &str) -> Vec<u32> {
    bound
        .iter()
        .filter(|b| b.port == port)
        .map(|b| b.id)
        .collect()
}

fn rate(num: i32) -> Rational {
    Rational { num, den: 1 }
}
use std::collections::BTreeMap;

#[derive(Deserialize, Default)]
struct Params {
    rate: Option<i32>,
}

thread_local! {
    static SEEN: RefCell<BTreeMap<u32, (u64, i64, i64)>> = const { RefCell::new(BTreeMap::new()) };
}

struct Node;

impl Guest for Node {
    fn describe() -> Meta {
        meta(
            "shape_sink",
            r#"{"type":"object","properties":{"rate":{"type":"integer"}}}"#,
        )
    }

    fn shape(
        params: String,
        _bound: Vec<ffrwd::av::node_types::Binding>,
    ) -> Result<NodeShape, String> {
        let params = parse::<Params>(&params)?;
        let mut v = input("v", PortKind::Video, Pairing::Arrival, RowsUse::Ignore);
        v.many = true;
        let clock = params
            .rate
            .map_or(Clock::SelfClocked, |r| Clock::Rate(rate(r)));
        Ok(shape(vec![v], Vec::new(), clock, false))
    }

    fn init(bound: Vec<BoundStream>, _latched: Vec<String>, _params: String) -> Result<(), String> {
        SEEN.with(|s| {
            let mut seen = s.borrow_mut();
            seen.clear();
            for id in bound_ids(&bound, "v") {
                seen.insert(id, (0, i64::MAX, i64::MIN));
            }
        });
        Ok(())
    }

    fn set_params(_params: String) -> Result<(), String> {
        Ok(())
    }

    fn process(tick: &Tick) -> Result<Emitted, String> {
        SEEN.with(|s| {
            let mut seen = s.borrow_mut();
            for id in tick.streams("v") {
                let entry = seen.entry(id).or_insert((0, i64::MAX, i64::MIN));
                for frame in tick.frames(id) {
                    entry.0 += 1;
                    entry.1 = entry.1.min(frame.pts);
                    entry.2 = entry.2.max(frame.pts);
                }
            }
            let rows = if tick.last() {
                seen.iter()
                    .map(|(id, (frames, first, last))| {
                        json!({"id": id, "frames": frames, "first": first, "last": last})
                            .to_string()
                    })
                    .collect()
            } else {
                Vec::new()
            };
            Ok(Emitted {
                items: Vec::new(),
                rows,
                finished: false,
            })
        })
    }
}

export!(Node);
