//! A picture redrawn at the size its params name, nearest pixel: an output
//! whose format is the call's, not its input's.

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

#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct Params {
    width: u32,
    height: u32,
}

const SCHEMA: &str = r#"{"type":"object","properties":{"width":{"type":"integer"},"height":{"type":"integer"}},"required":["width","height"]}"#;

struct State {
    v: u32,
    from: (u32, u32),
    to: (u32, u32),
    /// The input's colour as the host handed it, written as the row of the
    /// tick at pts 0: the test that a -pad reaches a node.
    colour: Option<String>,
}

thread_local! {
    static STATE: RefCell<Option<State>> = const { RefCell::new(None) };
}

struct Node;

impl Guest for Node {
    fn describe() -> Meta {
        meta("shape_canvas", SCHEMA)
    }

    fn shape(
        params: String,
        _bound: Vec<ffrwd::av::node_types::Binding>,
    ) -> Result<NodeShape, String> {
        let params: Params = parse(&params)?;
        if params.width == 0 || params.height == 0 {
            return Err("width and height are above zero".to_string());
        }
        let mut v = input("v", PortKind::Video, Pairing::Lockstep, RowsUse::Ignore);
        v.accepts.pixel_formats = vec!["rgba".to_string()];
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
        let mut shape = shape(vec![v], vec![out], Clock::Input("v".to_string()), true);
        shape.one_to_one = true;
        Ok(shape)
    }

    fn init(bound: Vec<BoundStream>, _latched: Vec<String>, params: String) -> Result<(), String> {
        let params: Params = parse(&params)?;
        let v = bound.iter().find(|b| b.port == "v").ok_or("v is bound")?;
        let Some(OutputFormat::Video(format)) = &v.format else {
            return Err("v is video".to_string());
        };
        let state = State {
            v: v.id,
            from: (format.width, format.height),
            to: (params.width, params.height),
            colour: format.color.as_ref().map(|c| {
                format!(
                    r#"{{"range":"{}","primaries":"{}","trc":"{}","space":"{}"}}"#,
                    c.range, c.primaries, c.trc, c.space
                )
            }),
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
            let mut items = Vec::new();
            let mut rows = Vec::new();
            if tick.pts() == 0 {
                if let Some(colour) = &state.colour {
                    rows.push(colour.clone());
                }
            }
            for frame in tick.frames(state.v) {
                let pixels = tick.fetch(state.v, frame.index);
                let (fw, fh) = (state.from.0 as usize, state.from.1 as usize);
                let (tw, th) = (state.to.0 as usize, state.to.1 as usize);
                let mut out = vec![0u8; tw * th * 4];
                for y in 0..th {
                    for x in 0..tw {
                        let (sx, sy) = (x * fw / tw, y * fh / th);
                        let from = (sy * fw + sx) * 4;
                        out[(y * tw + x) * 4..(y * tw + x) * 4 + 4]
                            .copy_from_slice(&pixels[from..from + 4]);
                    }
                }
                items.push(Emission {
                    port: "out".to_string(),
                    payload: Payload::Frame(RawFrame {
                        pts: frame.pts,
                        duration: frame.duration,
                        data: out,
                    }),
                });
            }
            let mut out = emitted(items, false);
            out.rows = rows;
            Ok(out)
        })
    }
}

export!(Node);
