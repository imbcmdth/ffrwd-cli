//! No inputs: a picture per tick of its rate, numbered, until `frames` have
//! gone, and then it says it is finished.

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
use ffrwd::av::types::{RawFrame, VideoFormat};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Params {
    fps: i32,
    frames: i64,
    width: u32,
    height: u32,
}

impl Default for Params {
    fn default() -> Params {
        Params {
            fps: 25,
            frames: 25,
            width: 2,
            height: 2,
        }
    }
}

const SCHEMA: &str = r#"{"type":"object","properties":{"fps":{"type":"integer"},"frames":{"type":"integer"},"width":{"type":"integer"},"height":{"type":"integer"}}}"#;

thread_local! {
    static PARAMS: RefCell<Option<Params>> = const { RefCell::new(None) };
}

struct Node;

impl Guest for Node {
    fn describe() -> Meta {
        meta("shape_rate", SCHEMA)
    }

    fn shape(
        params: String,
        _bound: Vec<ffrwd::av::node_types::Binding>,
    ) -> Result<NodeShape, String> {
        let params: Params = parse(&params)?;
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
        Ok(shape(
            Vec::new(),
            vec![out],
            Clock::Rate(rate(params.fps)),
            true,
        ))
    }

    fn init(_bound: Vec<BoundStream>, _latched: Vec<String>, params: String) -> Result<(), String> {
        let params: Params = parse(&params)?;
        PARAMS.with(|p| *p.borrow_mut() = Some(params));
        Ok(())
    }

    fn set_params(_params: String) -> Result<(), String> {
        Ok(())
    }

    fn process(tick: &Tick) -> Result<Emitted, String> {
        PARAMS.with(|p| {
            let params = p.borrow();
            let params = params.as_ref().ok_or("process before init")?;
            if tick.last() {
                return Ok(emitted(Vec::new(), false));
            }
            let pts = tick.pts();
            let size = (params.width * params.height * 4) as usize;
            let frame = Emission {
                port: "out".to_string(),
                payload: Payload::Frame(RawFrame {
                    pts,
                    duration: Some(1),
                    data: vec![(pts % 256) as u8; size],
                }),
            };
            Ok(emitted(vec![frame], pts + 1 >= params.frames))
        })
    }
}

export!(Node);
