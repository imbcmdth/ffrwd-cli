//! A sink of any number of pictures, taken as they arrive: it writes
//! nothing on any port, and on its last call says, as rows, how many frames
//! each stream brought, their first and last pts and the time base those are
//! counted in. Self-clocked, or with `rate` a publisher's shape: a rate clock
//! that only needs turns. With `packets` it takes what a publisher takes,
//! coded pictures on `video`, coded sound on `audio` and messages on `data`,
//! and counts packets and messages.

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
    #[serde(default)]
    packets: bool,
}

const ROWS_SCHEMA: &str = r#"{"type":"object","properties":{"id":{"type":"integer"},"port":{"type":"string"},"row":{"type":["integer","null"]},"num":{"type":"integer"},"den":{"type":"integer"},"frames":{"type":"integer"},"first":{"type":"integer"},"last":{"type":"integer"}}}"#;

/// What one stream brought: its port, the relation row it was bound for,
/// the time base its pts count in, how many frames or packets, and the
/// first and last pts.
struct Seen {
    port: String,
    row: Option<u32>,
    base: Rational,
    frames: u64,
    first: i64,
    last: i64,
}

thread_local! {
    static SEEN: RefCell<BTreeMap<u32, Seen>> = const { RefCell::new(BTreeMap::new()) };
    static PACKETS: RefCell<bool> = const { RefCell::new(false) };
}

/// The ports the sink reads, with `packets` or without.
fn ports(packets: bool) -> &'static [&'static str] {
    if packets {
        &["video", "audio", "data"]
    } else {
        &["v"]
    }
}

struct Node;

impl Guest for Node {
    fn describe() -> Meta {
        let mut described = meta(
            "shape_sink",
            r#"{"type":"object","properties":{"rate":{"type":"integer"},"packets":{"type":"boolean"}}}"#,
        );
        described.rows_schema = ROWS_SCHEMA.to_string();
        described
    }

    fn shape(
        params: String,
        _bound: Vec<ffrwd::av::node_types::Binding>,
    ) -> Result<NodeShape, String> {
        let params = parse::<Params>(&params)?;
        let inputs = ports(params.packets)
            .iter()
            .map(|name| {
                let kind = match (*name, params.packets) {
                    ("data", _) => PortKind::Data,
                    (_, true) => PortKind::Packets,
                    (_, false) => PortKind::Video,
                };
                let rows = if kind == PortKind::Data {
                    RowsUse::PerFrame
                } else {
                    RowsUse::Ignore
                };
                let mut port = input(name, kind, Pairing::Arrival, rows);
                port.many = true;
                port.required = !matches!(*name, "audio" | "data");
                port
            })
            .collect();
        let clock = params
            .rate
            .map_or(Clock::SelfClocked, |r| Clock::Rate(rate(r)));
        Ok(shape(inputs, Vec::new(), clock, false))
    }

    fn init(bound: Vec<BoundStream>, _latched: Vec<String>, params: String) -> Result<(), String> {
        let packets = parse::<Params>(&params)?.packets;
        SEEN.with(|s| {
            let mut seen = s.borrow_mut();
            seen.clear();
            for stream in bound
                .iter()
                .filter(|b| ports(packets).contains(&b.port.as_str()))
            {
                seen.insert(
                    stream.id,
                    Seen {
                        port: stream.port.clone(),
                        row: stream.row,
                        base: stream.info.time_base,
                        frames: 0,
                        first: i64::MAX,
                        last: i64::MIN,
                    },
                );
            }
        });
        PACKETS.with(|p| *p.borrow_mut() = packets);
        Ok(())
    }

    fn set_params(_params: String) -> Result<(), String> {
        Ok(())
    }

    fn process(tick: &Tick) -> Result<Emitted, String> {
        let packets = PACKETS.with(|p| *p.borrow());
        SEEN.with(|s| {
            let mut seen = s.borrow_mut();
            for port in ports(packets) {
                for id in tick.streams(port) {
                    let Some(entry) = seen.get_mut(&id) else {
                        continue;
                    };
                    let times: Vec<i64> = if *port == "data" {
                        tick.messages(id)
                            .iter()
                            .map(|message| message.pts)
                            .collect()
                    } else if packets {
                        tick.packets(id).iter().map(|packet| packet.pts).collect()
                    } else {
                        tick.frames(id).iter().map(|frame| frame.pts).collect()
                    };
                    for pts in times {
                        entry.frames += 1;
                        entry.first = entry.first.min(pts);
                        entry.last = entry.last.max(pts);
                    }
                }
            }
            let rows = if tick.last() {
                seen.iter()
                    .map(|(id, one)| {
                        json!({
                            "id": id,
                            "port": one.port,
                            "row": one.row,
                            "num": one.base.num,
                            "den": one.base.den,
                            "frames": one.frames,
                            "first": one.first,
                            "last": one.last,
                        })
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
