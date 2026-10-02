//! A tumbling window over 48 kHz sound: `window` samples a tick, none
//! shared, and one cue per window naming it, which leaves with the window,
//! so its port's latency is the window's length.

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
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Params {
    window: u32,
}

impl Default for Params {
    fn default() -> Params {
        Params { window: 48000 }
    }
}

const RATE: u32 = 48000;
const SCHEMA: &str = r#"{"type":"object","properties":{"window":{"type":"integer","minimum":1}}}"#;

thread_local! {
    static STATE: RefCell<(u32, u32, u64)> = const { RefCell::new((0, 1, 0)) };
}

struct Node;

impl Guest for Node {
    fn describe() -> Meta {
        meta("shape_window", SCHEMA)
    }

    fn shape(params: String, _bound: Vec<String>) -> Result<NodeShape, String> {
        let params: Params = parse(&params)?;
        let mut a = input("a", PortKind::Audio, Pairing::Lockstep, RowsUse::Ignore);
        a.window = params.window;
        a.stride = params.window;
        a.accepts.sample_formats = vec!["f32".to_string()];
        a.accepts.sample_rates = vec![RATE];
        let mut cues = data("cues");
        cues.latency = f64::from(params.window) / f64::from(RATE);
        Ok(shape(
            vec![a],
            vec![cues],
            Clock::Input("a".to_string()),
            false,
        ))
    }

    fn init(bound: Vec<BoundStream>, _latched: Vec<String>, _params: String) -> Result<(), String> {
        let a = bound.iter().find(|b| b.port == "a").ok_or("a is bound")?;
        let Some(OutputFormat::Audio(format)) = &a.format else {
            return Err("a is audio".to_string());
        };
        STATE.with(|s| *s.borrow_mut() = (a.id, 4 * format.channels, 0));
        Ok(())
    }

    fn set_params(_params: String) -> Result<(), String> {
        Ok(())
    }

    fn process(tick: &Tick) -> Result<Emitted, String> {
        STATE.with(|s| {
            let mut state = s.borrow_mut();
            let (a, sample, _) = *state;
            let mut items = Vec::new();
            for frame in tick.frames(a) {
                let samples = tick.fetch(a, frame.index).len() as u64 / u64::from(sample);
                let base = tick.time_base();
                let start = frame.pts as f64 * f64::from(base.num) / f64::from(base.den);
                let end = start + samples as f64 / f64::from(RATE);
                let window = state.2;
                state.2 += 1;
                items.push(message(
                    "cues",
                    frame.pts,
                    json!({"start_t": start, "end_t": end, "text": format!("window {window}"), "samples": samples}),
                ));
            }
            Ok(emitted(items, false))
        })
    }
}

export!(Node);
