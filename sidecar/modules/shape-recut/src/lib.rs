//! A picture clock and its sound, lockstep: every tick says how many
//! samples it was handed and where they start, which at 29.97 under 48 kHz
//! is 1,601 or 1,602. On `v` and `a` it hands the picture and the sound on
//! as they came, one node writing both, as a compositor does.

// Every shape module carries the same few helpers, and not every one uses
// each of them.
#![allow(dead_code, unused_imports)]

wit_bindgen::generate!({
    path: "../../wit",
    world: "node-module",
});

use std::cell::RefCell;

use exports::ffrwd::av::node::{Emission, Emitted, Guest, Payload, SameFrame};
use ffrwd::av::node_tick::Tick;
use ffrwd::av::node_types::{
    Accepts, BoundStream, Clock, InputPort, LikeInput, Message, NodeShape, OutputFormat,
    OutputPort, Pairing, PortKind, RowsUse,
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

struct State {
    v: u32,
    a: u32,
    sample: usize,
    /// Whether `v` and `a` are read.
    hand_on: (bool, bool),
}

/// An output carrying what input `port` carries.
fn like(port: &str, kind: PortKind) -> OutputPort {
    output(
        port,
        kind,
        Some(OutputFormat::Like(LikeInput {
            port: port.to_string(),
            pixel_format: None,
            sample_format: None,
        })),
    )
}

/// Frame `index` of stream `id` this tick, leaving on `port` as it came.
fn same(port: &str, id: u32, frame: &ffrwd::av::node_types::Frame) -> Emission {
    Emission {
        port: port.to_string(),
        payload: Payload::Same(SameFrame {
            pts: frame.pts,
            duration: frame.duration,
            id,
            index: frame.index,
        }),
    }
}

thread_local! {
    static STATE: RefCell<Option<State>> = const { RefCell::new(None) };
}

struct Node;

impl Guest for Node {
    fn describe() -> Meta {
        meta("shape_recut", "")
    }

    fn shape(
        params: String,
        _bound: Vec<ffrwd::av::node_types::Binding>,
    ) -> Result<NodeShape, String> {
        parse::<Params>(&params)?;
        let v = input("v", PortKind::Video, Pairing::Lockstep, RowsUse::Ignore);
        let mut a = input("a", PortKind::Audio, Pairing::Lockstep, RowsUse::Ignore);
        a.accepts.sample_formats = vec!["f32".to_string()];
        Ok(shape(
            vec![v, a],
            vec![
                data("samples"),
                like("v", PortKind::Video),
                like("a", PortKind::Audio),
            ],
            Clock::Input("v".to_string()),
            true,
        ))
    }

    fn init(bound: Vec<BoundStream>, latched: Vec<String>, _params: String) -> Result<(), String> {
        let v = bound.iter().find(|b| b.port == "v").ok_or("v is bound")?;
        let a = bound.iter().find(|b| b.port == "a").ok_or("a is bound")?;
        let Some(OutputFormat::Audio(format)) = &a.format else {
            return Err("a is audio".to_string());
        };
        let read = |port: &str| latched.iter().any(|one| one == port);
        let state = State {
            v: v.id,
            a: a.id,
            sample: 4 * format.channels as usize,
            hand_on: (read("v"), read("a")),
        };
        STATE.with(|s| *s.borrow_mut() = Some(state));
        Ok(())
    }

    fn set_params(_params: String) -> Result<(), String> {
        Ok(())
    }

    fn process(tick: &Tick) -> Result<Emitted, String> {
        STATE.with(|s| {
            let state = s.borrow();
            let state = state.as_ref().ok_or("process before init")?;
            let frames = tick.frames(state.a);
            let samples: usize = frames
                .iter()
                .map(|f| tick.fetch(state.a, f.index).len() / state.sample)
                .sum();
            let first = frames.first().map(|f| f.pts);
            let item = message(
                "samples",
                tick.pts(),
                json!({"pts": tick.pts(), "samples": samples, "audio_pts": first, "last": tick.last()}),
            );
            let mut items = vec![item];
            if state.hand_on.0 {
                items.extend(tick.frames(state.v).iter().map(|f| same("v", state.v, f)));
            }
            if state.hand_on.1 {
                items.extend(frames.iter().map(|f| same("a", state.a, f)));
            }
            Ok(emitted(items, false))
        })
    }
}

export!(Node);
