//! Coded packets as the clock: a tick per packet, in decode order, each
//! logged with its times, its key flag and its size, and handed on as it
//! came on `out`, a packets output that names no format of its own.

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
#[derive(Deserialize, Default)]
struct Params {}

thread_local! {
    static PACKETS: RefCell<(u32, bool)> = const { RefCell::new((0, false)) };
}

struct Node;

impl Guest for Node {
    fn describe() -> Meta {
        meta("shape_packets", "")
    }

    fn shape(
        params: String,
        _bound: Vec<ffrwd::av::node_types::Binding>,
    ) -> Result<NodeShape, String> {
        parse::<Params>(&params)?;
        let p = input("p", PortKind::Packets, Pairing::Lockstep, RowsUse::Ignore);
        Ok(shape(
            vec![p],
            vec![data("log"), output("out", PortKind::Packets, None)],
            Clock::Input("p".to_string()),
            false,
        ))
    }

    fn init(bound: Vec<BoundStream>, latched: Vec<String>, _params: String) -> Result<(), String> {
        let p = bound_ids(&bound, "p")
            .first()
            .copied()
            .ok_or("p is bound")?;
        let out = latched.iter().any(|port| port == "out");
        PACKETS.with(|s| *s.borrow_mut() = (p, out));
        Ok(())
    }

    fn set_params(_params: String) -> Result<(), String> {
        Ok(())
    }

    fn process(tick: &Tick) -> Result<Emitted, String> {
        let (p, out) = PACKETS.with(|s| *s.borrow());
        let mut items = Vec::new();
        for packet in tick.packets(p) {
            items.push(message(
                "log",
                tick.pts(),
                json!({
                    "tick": tick.pts(),
                    "pts": packet.pts,
                    "dts": packet.dts,
                    "key": packet.keyframe,
                    "bytes": packet.data.len(),
                    "last": tick.last(),
                }),
            ));
            if out {
                items.push(Emission {
                    port: "out".to_string(),
                    payload: Payload::Packet(packet),
                });
            }
        }
        Ok(emitted(items, false))
    }
}

export!(Node);
