//! Any number of optional pictures, held: at every tick of its rate it hands
//! back, with `same`, the newest frame of the input `pick` names, or of the
//! first that shows one, and draws a blank canvas where none does. With no
//! picture bound it ticks at `fps` and finishes after `frames`.

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
use exports::ffrwd::av::node::SameFrame;
use ffrwd::av::node_types::{Anchor, Hold};
use ffrwd::av::types::{RawFrame, VideoFormat};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Params {
    #[serde(default = "ten")]
    fps: i32,
    width: u32,
    height: u32,
    #[serde(default)]
    pick: usize,
    #[serde(default)]
    frames: Option<i64>,
    /// The ports the pictures arrive on instead of being bound, one feed
    /// each; they are conformed to the picture `c`, which is then the clock.
    #[serde(default)]
    ports: Option<serde_json::Value>,
}

fn ten() -> i32 {
    10
}

impl Default for Params {
    fn default() -> Params {
        Params {
            fps: 10,
            width: 4,
            height: 4,
            pick: 0,
            frames: None,
            ports: None,
        }
    }
}

const SCHEMA: &str = r#"{"type":"object","properties":{"fps":{"type":"integer"},"width":{"type":"integer"},"height":{"type":"integer"},"pick":{"type":"integer"},"frames":{"type":"integer"},"ports":{"type":["array","integer"],"items":{"type":"integer"}}}}"#;

struct State {
    params: Params,
    /// The bound pictures, and whether each is the output's size.
    inputs: Vec<(u32, bool)>,
}

thread_local! {
    static STATE: RefCell<Option<State>> = const { RefCell::new(None) };
}

fn node_shape(params: &Params, bound: &[String]) -> NodeShape {
    let mut v = input(
        "v",
        PortKind::Video,
        Pairing::Hold(Hold {
            anchor: Anchor::SharedClock,
            lead: 0.0,
            linger: None,
            timeout: None,
            group: None,
            port_param: params.ports.as_ref().map(|_| "ports".to_string()),
        }),
        RowsUse::Ignore,
    );
    v.required = false;
    v.many = true;
    v.accepts.pixel_formats = vec!["rgba".to_string()];
    let mut inputs = vec![v];
    if params.ports.is_some() {
        inputs[0].accepts.like = Some("c".to_string());
        let mut c = input("c", PortKind::Video, Pairing::Lockstep, RowsUse::Ignore);
        c.accepts.pixel_formats = vec!["rgba".to_string()];
        inputs.push(c);
    }
    let out = output(
        "out",
        PortKind::Video,
        Some(OutputFormat::Video(VideoFormat {
            width: params.width,
            height: params.height,
            pix_fmt: "rgba".to_string(),
            color: None,
        })),
    );
    let clock = if bound.iter().any(|b| b == "c") {
        Clock::Input("c".to_string())
    } else if bound.iter().any(|b| b == "v") {
        Clock::RateOf("v".to_string())
    } else {
        Clock::Rate(rate(params.fps))
    };
    shape(inputs, vec![out], clock, true)
}

struct Node;

impl Guest for Node {
    fn describe() -> Meta {
        meta("shape_hold", SCHEMA)
    }

    fn shape(
        params: String,
        bound: Vec<ffrwd::av::node_types::Binding>,
    ) -> Result<NodeShape, String> {
        let names: Vec<String> = bound.into_iter().map(|b| b.input).collect();
        Ok(node_shape(&parse(&params)?, &names))
    }

    fn init(bound: Vec<BoundStream>, _latched: Vec<String>, params: String) -> Result<(), String> {
        let params: Params = parse(&params)?;
        let inputs = bound
            .iter()
            .filter(|b| b.port == "v")
            .map(|b| {
                let fits = matches!(&b.format, Some(OutputFormat::Video(v))
                    if v.width == params.width && v.height == params.height);
                (b.id, fits)
            })
            .collect();
        STATE.with(|s| *s.borrow_mut() = Some(State { params, inputs }));
        Ok(())
    }

    fn set_params(params: String) -> Result<(), String> {
        parse::<Params>(&params).map(|_| ())
    }

    fn process(tick: &Tick) -> Result<Emitted, String> {
        STATE.with(|s| {
            let state = s.borrow();
            let state = state.as_ref().ok_or("process before init")?;
            let pts = tick.pts();
            let mut order: Vec<usize> = (0..state.inputs.len()).collect();
            if state.params.pick < order.len() {
                order.retain(|i| *i != state.params.pick);
                order.insert(0, state.params.pick);
            }
            for index in order {
                let (id, fits) = state.inputs[index];
                if let (true, Some(frame)) = (fits, tick.frames(id).last()) {
                    let same = Emission {
                        port: "out".to_string(),
                        payload: Payload::Same(SameFrame {
                            pts,
                            duration: Some(1),
                            id,
                            index: frame.index,
                        }),
                    };
                    return Ok(emitted(vec![same], false));
                }
            }
            if tick.last() {
                return Ok(emitted(Vec::new(), false));
            }
            let size = (state.params.width * state.params.height * 4) as usize;
            let blank = Emission {
                port: "out".to_string(),
                payload: Payload::Frame(RawFrame {
                    pts,
                    duration: Some(1),
                    data: vec![(pts % 251) as u8; size],
                }),
            };
            let finished =
                state.inputs.is_empty() && state.params.frames.is_some_and(|n| pts + 1 >= n);
            Ok(emitted(vec![blank], finished))
        })
    }
}

export!(Node);
