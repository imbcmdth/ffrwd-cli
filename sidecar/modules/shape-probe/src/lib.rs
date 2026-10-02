//! A node whose shape reaches into every corner of the shape record, so the
//! host's `--shape` and its refusals can be checked against a real module.
//!
//! - `v`: video, the clock, lockstep, per-frame rows, accepts yuv420p and
//!   rgba. With `rate` set the clock is that rate instead and `v` is held.
//! - `feed`: video, optional, held on a host port named by `port`, anchored
//!   by the `smart_timed` tag.
//! - `words`: data, optional and many, by interval, folded as state, each
//!   stream placed on the clock by its first message.
//! - `a`: audio, optional, lockstep with `v`.
//! - `cues`: data, optional, by interval, on `feed`'s connection.
//! - `size`: video, optional, lockstep, read for its timing alone.
//! - `mask`: `v`'s size in gray, left out when `v` is not bound.
//! - `copy`: `v` itself, frame for frame.
//! - `canvas`: a video of the size `canvas` names, only when it does.
//! - `spots`: data, one message per tick, as late as twelve of `v`'s frames
//!   where the call says `v`'s rate and half a second where it does not.
//!
//! `refuse` asks for a shape the host must refuse, by name of the rule.

wit_bindgen::generate!({
    path: "../../wit",
    world: "node-module",
});

use exports::ffrwd::av::node::{Emission, Emitted, Guest, Payload, SameFrame};
use ffrwd::av::node_tick::Tick;
use ffrwd::av::node_types::{
    Accepts, Anchor, Binding, BoundStream, Clock, Hold, InputPort, Interval, LikeInput, Message,
    NodeShape, OutputFormat, OutputPort, Pairing, PortKind, RowsUse,
};
use ffrwd::av::types::{Meta, Rational, VideoFormat, Wants};
use std::sync::atomic::{AtomicBool, Ordering};

use serde::Deserialize;

const PARAMS_SCHEMA: &str = r#"{"type":"object","properties":{"rate":{"type":"integer","minimum":1},"canvas":{"type":"object","properties":{"width":{"type":"integer"},"height":{"type":"integer"}},"required":["width","height"]},"refuse":{"type":"string"},"port":{"type":"integer"}},"additionalProperties":false}"#;

#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct Params {
    rate: Option<i32>,
    canvas: Option<Canvas>,
    refuse: Option<String>,
    #[allow(dead_code)]
    port: Option<u32>,
}

#[derive(Deserialize)]
struct Canvas {
    width: u32,
    height: u32,
}

fn params(text: &str) -> Result<Params, String> {
    if text.trim().is_empty() {
        return Ok(Params::default());
    }
    serde_json::from_str(text).map_err(|e| format!("params: {e}"))
}

fn accepts(pixel_formats: &[&str]) -> Accepts {
    Accepts {
        pixel_formats: pixel_formats.iter().map(|f| f.to_string()).collect(),
        sample_formats: Vec::new(),
        sample_rates: Vec::new(),
        channel_counts: Vec::new(),
        codecs: Vec::new(),
        wants: Wants::All,
        like: None,
    }
}

fn input(name: &str, kind: PortKind, required: bool, pairing: Pairing, rows: RowsUse) -> InputPort {
    InputPort {
        name: name.to_string(),
        kind,
        required,
        many: false,
        pairing,
        rows,
        window: 1,
        stride: 1,
        accepts: accepts(&[]),
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

fn shape(params: &Params, bound: &[Binding]) -> NodeShape {
    let held = params.rate.is_some();
    let mut v = input(
        "v",
        PortKind::Video,
        !held,
        Pairing::Lockstep,
        RowsUse::PerFrame,
    );
    v.accepts = accepts(&["yuv420p", "rgba"]);
    if held {
        v.pairing = Pairing::Hold(Hold {
            anchor: Anchor::SharedClock,
            lead: 0.0,
            linger: None,
            timeout: None,
            group: None,
            port_param: None,
        });
    }
    let feed = input(
        "feed",
        PortKind::Video,
        false,
        Pairing::Hold(Hold {
            anchor: Anchor::Tagged("smart_timed".to_string()),
            lead: 0.5,
            linger: Some(1.0),
            timeout: None,
            group: Some("feeder".to_string()),
            port_param: Some("port".to_string()),
        }),
        RowsUse::Ignore,
    );
    let mut words = input(
        "words",
        PortKind::Data,
        false,
        Pairing::Interval(Interval {
            latency: Some(2.0),
            ahead: 0.25,
            anchor: Anchor::FirstFrame,
            group: None,
        }),
        RowsUse::State,
    );
    words.many = true;
    words.schema = Some(r#"{"type":"object","properties":{"text":{"type":"string"}}}"#.to_string());
    let audio_pairing = if held {
        Pairing::Hold(Hold {
            anchor: Anchor::FirstFrame,
            lead: 0.0,
            linger: None,
            timeout: Some(5.0),
            group: None,
            port_param: None,
        })
    } else {
        Pairing::Lockstep
    };
    let mut a = input("a", PortKind::Audio, false, audio_pairing, RowsUse::Ignore);
    a.accepts.sample_formats = vec!["f32".to_string()];

    let cues = input(
        "cues",
        PortKind::Data,
        false,
        Pairing::Interval(Interval {
            latency: None,
            ahead: 0.0,
            anchor: Anchor::SharedClock,
            group: Some("feeder".to_string()),
        }),
        RowsUse::PerFrame,
    );
    let mut size = input(
        "size",
        PortKind::Video,
        false,
        Pairing::Lockstep,
        RowsUse::Ignore,
    );
    size.accepts.wants = Wants::Timing;

    let v_binding = bound.iter().find(|b| b.input == "v");
    let v_bound = v_binding.is_some();
    let v_rate = v_binding
        .and_then(|b| b.streams.first())
        .and_then(|s| s.rate);
    let mut outputs = Vec::new();
    if v_bound {
        outputs.push(output(
            "mask",
            PortKind::Video,
            Some(OutputFormat::Like(LikeInput {
                port: "v".to_string(),
                pixel_format: Some("gray".to_string()),
                sample_format: None,
            })),
        ));
        outputs.push(output(
            "copy",
            PortKind::Video,
            Some(OutputFormat::Like(LikeInput {
                port: "v".to_string(),
                pixel_format: None,
                sample_format: None,
            })),
        ));
    }
    if let Some(canvas) = &params.canvas {
        outputs.push(output(
            "canvas",
            PortKind::Video,
            Some(OutputFormat::Video(VideoFormat {
                width: canvas.width,
                height: canvas.height,
                pix_fmt: "rgba".to_string(),
                color: None,
            })),
        ));
    }
    let mut spots = output(
        "spots",
        PortKind::Data,
        Some(OutputFormat::Data("json".to_string())),
    );
    spots.time_base = Some(Rational {
        num: 1,
        den: 1_000_000,
    });
    spots.latency = match v_rate {
        Some(rate) => 12.0 * f64::from(rate.den) / f64::from(rate.num),
        None => 0.5,
    };
    spots.schema =
        Some(r#"{"type":"object","properties":{"start_t":{"type":"number"}}}"#.to_string());
    outputs.push(spots);

    let mut shape = NodeShape {
        inputs: vec![v, feed, words, a, cues, size],
        outputs,
        clock: match params.rate {
            Some(rate) => Clock::Rate(Rational { num: rate, den: 1 }),
            None => Clock::Input("v".to_string()),
        },
        pure: true,
        one_to_one: false,
        bounded: true,
        relation: Vec::new(),
    };
    match params.refuse.as_deref() {
        Some("lockstep_on_rate") => {
            shape.clock = Clock::Rate(Rational { num: 25, den: 1 });
            shape.inputs[0].pairing = Pairing::Lockstep;
        }
        Some("hold_on_data") => {
            shape.inputs[2].pairing = Pairing::Hold(Hold {
                anchor: Anchor::SharedClock,
                lead: 0.0,
                linger: None,
                timeout: None,
                group: None,
                port_param: None,
            });
        }
        Some("data_ignores_rows") => shape.inputs[2].rows = RowsUse::Ignore,
        Some("stride_past_window") => {
            shape.inputs[0].window = 2;
            shape.inputs[0].stride = 3;
        }
        Some("optional_clock") => shape.inputs[0].required = false,
        Some("timing_on_data") => shape.inputs[4].accepts.wants = Wants::Timing,
        Some("group_not_held") => {
            if let Pairing::Interval(interval) = &mut shape.inputs[4].pairing {
                interval.group = Some("nobody".to_string());
            }
        }
        Some("group_first_frame") => {
            if let Pairing::Interval(interval) = &mut shape.inputs[4].pairing {
                interval.anchor = Anchor::FirstFrame;
            }
        }
        _ => {}
    }
    shape
}

struct ShapeProbe;

/// Whether the query reads `copy`, as `init` was told.
static LATCHED_COPY: AtomicBool = AtomicBool::new(false);

impl Guest for ShapeProbe {
    fn describe() -> Meta {
        Meta {
            name: "shape_probe".to_string(),
            version: "0.1.0".to_string(),
            params_schema: PARAMS_SCHEMA.to_string(),
            rows_schema: String::new(),
            pixel_formats: Vec::new(),
            sample_formats: Vec::new(),
            sample_rates: Vec::new(),
            channel_counts: Vec::new(),
            rows_language: Vec::new(),
        }
    }

    fn shape(params_text: String, bound: Vec<Binding>) -> Result<NodeShape, String> {
        Ok(shape(&params(&params_text)?, &bound))
    }

    fn init(
        _bound: Vec<BoundStream>,
        latched: Vec<String>,
        params_text: String,
    ) -> Result<(), String> {
        params(&params_text)?;
        LATCHED_COPY.store(latched.iter().any(|l| l == "copy"), Ordering::Relaxed);
        Ok(())
    }

    fn set_params(params_text: String) -> Result<(), String> {
        params(&params_text).map(|_| ())
    }

    fn process(tick: &Tick) -> Result<Emitted, String> {
        let mut items = Vec::new();
        if LATCHED_COPY.load(Ordering::Relaxed) {
            for id in tick.streams("v") {
                if let Some(frame) = tick.frames(id).first() {
                    items.push(Emission {
                        port: "copy".to_string(),
                        payload: Payload::Same(SameFrame {
                            pts: frame.pts,
                            duration: frame.duration,
                            id,
                            index: frame.index,
                        }),
                    });
                }
            }
        }
        let base = tick.time_base();
        let seconds = tick.pts() as f64 * f64::from(base.num) / f64::from(base.den);
        items.push(Emission {
            port: "spots".to_string(),
            payload: Payload::Message(Message {
                pts: (seconds * 1_000_000.0).round() as i64,
                data: format!(r#"{{"start_t":{seconds}}}"#).into_bytes(),
            }),
        });
        Ok(Emitted {
            items,
            rows: Vec::new(),
            finished: false,
        })
    }
}

export!(ShapeProbe);
