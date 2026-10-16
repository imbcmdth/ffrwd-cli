//! A picture clock and words read by interval and kept as state: every tick
//! says how many words this instance has seen, its own and those of the
//! ticks it did not process, and when the newest of them arrived, read in
//! the words' time base. A count and a time that match the serial run's at
//! every worker count are the host handing each instance its earlier rows,
//! each at the time it arrived.

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
struct Params {
    /// The most the words wait, in seconds; none waits for their producer.
    latency: Option<f64>,
}

const SCHEMA: &str = r#"{"type":"object","properties":{"latency":{"type":"number"}}}"#;

struct State {
    words: Option<u32>,
    /// The words' time base.
    base: (i32, i32),
    seen: u64,
    /// When the newest word arrived, in the words' time base.
    newest: Option<i64>,
}

thread_local! {
    static STATE: RefCell<State> = const {
        RefCell::new(State { words: None, base: (1, 1), seen: 0, newest: None })
    };
}

struct Node;

impl Guest for Node {
    fn describe() -> Meta {
        meta("shape_state", SCHEMA)
    }

    fn shape(
        params: String,
        _bound: Vec<ffrwd::av::node_types::Binding>,
    ) -> Result<NodeShape, String> {
        let params: Params = parse(&params)?;
        let v = input("v", PortKind::Video, Pairing::Lockstep, RowsUse::Ignore);
        let mut words = input(
            "words",
            PortKind::Data,
            Pairing::Interval(ffrwd::av::node_types::Interval {
                latency: params.latency,
                ahead: 0.0,
                anchor: ffrwd::av::node_types::Anchor::SharedClock,
                group: None,
            }),
            RowsUse::State,
        );
        words.required = false;
        Ok(shape(
            vec![v, words],
            vec![data("seen")],
            Clock::Input("v".to_string()),
            true,
        ))
    }

    fn init(bound: Vec<BoundStream>, _latched: Vec<String>, _params: String) -> Result<(), String> {
        let words = bound.iter().find(|b| b.port == "words");
        let state = State {
            words: words.map(|b| b.id),
            base: words.map_or((1, 1), |b| (b.info.time_base.num, b.info.time_base.den)),
            seen: 0,
            newest: None,
        };
        STATE.with(|s| *s.borrow_mut() = state);
        Ok(())
    }

    fn set_params(_params: String) -> Result<(), String> {
        Ok(())
    }

    fn process(tick: &Tick) -> Result<Emitted, String> {
        STATE.with(|s| {
            let mut state = s.borrow_mut();
            let mut now = 0;
            if let Some(id) = state.words {
                for rows in tick.earlier_rows(id) {
                    state.seen += rows.rows.len() as u64;
                    state.newest = Some(rows.pts);
                }
                for m in tick.messages(id) {
                    now += 1;
                    state.seen += 1;
                    state.newest = Some(m.pts);
                }
            }
            let (num, den) = state.base;
            let newest = state
                .newest
                .map(|pts| pts as f64 * f64::from(num) / f64::from(den));
            let item = message(
                "seen",
                tick.pts(),
                json!({"pts": tick.pts(), "seen": state.seen, "now": now, "newest": newest,
                       "last": tick.last()}),
            );
            Ok(emitted(vec![item], false))
        })
    }
}

export!(Node);
