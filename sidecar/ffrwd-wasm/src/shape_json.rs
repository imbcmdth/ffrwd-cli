//! `--shape`: a node module's `shape(params, bound)` as one JSON line, the
//! contract a compiler reads a node's ports and clock from
//! (`sidecar/NODE-SHAPE.md`).
//!
//! The WIT's records keep their field names, in snake_case. A variant is an
//! object whose `kind` names its case, with the case's payload beside it:
//! a record's fields flattened in, anything else under a name of its own.

use anyhow::{anyhow, bail, Context, Result};
use ffrwd_wasm_runtime::node::{
    Accepts, Anchor, Binding, Clock, ColorSpec, InputPort, NodeShape, OutputFormat, OutputPort,
    Pairing, PortKind, Rational, RowsUse, StreamHint,
};
use ffrwd_wasm_runtime::runtime::{CodedFormat, CodedStream, Wants};
use serde::Deserialize;
use serde_json::{json, Map, Value};

/// `--shape <module> [-params <json> | -params-from <file>] [-bound <port,...> | -bound <json>]`,
/// grants taken out first the way `--probe` takes them.
pub fn run(module: &str, rest: &[String]) -> Result<String> {
    let (params, bound) = parse_args(rest)?;
    let exports_node = ffrwd_wasm_runtime::runtime::exports_node(module)
        .with_context(|| format!("asking {module} for its shape"))?;
    let shape = if exports_node {
        ffrwd_wasm_runtime::runtime::node_shape(module, &params, &bound)
    } else {
        crate::adapters::declared_shape(module, &params, &Binding::names(&bound))
    }
    .with_context(|| format!("asking {module} for its shape"))?;
    serde_json::to_string(&shape_json(&shape)).context("serializing the shape")
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct BindingJson {
    input: String,
    #[serde(default)]
    streams: Option<Vec<HintJson>>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct HintJson {
    #[serde(default)]
    rate: Option<RationalJson>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RationalJson {
    num: i32,
    den: i32,
}

/// `--bound`'s value: the comma list of the inputs' names, each a stream
/// with nothing known of it, or a JSON array of `binding` records.
pub fn parse_bound(text: &str) -> Result<Vec<Binding>> {
    let text = text.trim();
    if !text.starts_with('[') {
        let names: Vec<String> = text
            .split(',')
            .map(str::trim)
            .filter(|port| !port.is_empty())
            .map(str::to_string)
            .collect();
        return Ok(Binding::from_names(&names));
    }
    let parsed: Vec<BindingJson> =
        serde_json::from_str(text).map_err(|e| anyhow!("--bound {text}: {e}"))?;
    let mut bound: Vec<Binding> = Vec::with_capacity(parsed.len());
    for binding in parsed {
        if bound.iter().any(|b| b.input == binding.input) {
            bail!("--bound names input '{}' twice", binding.input);
        }
        let streams = match binding.streams {
            None => vec![StreamHint::default()],
            Some(streams) if streams.is_empty() => {
                bail!("--bound binds input '{}' to no stream", binding.input)
            }
            Some(streams) => streams
                .into_iter()
                .map(|hint| match hint.rate {
                    Some(RationalJson { num, den }) if num <= 0 || den <= 0 => bail!(
                        "--bound gives input '{}' the rate {num}/{den}, which is no rate",
                        binding.input
                    ),
                    rate => Ok(StreamHint {
                        rate: rate.map(|r| Rational {
                            num: r.num,
                            den: r.den,
                        }),
                    }),
                })
                .collect::<Result<_>>()?,
        };
        bound.push(Binding {
            input: binding.input,
            streams,
        });
    }
    Ok(bound)
}

fn parse_args(rest: &[String]) -> Result<(String, Vec<Binding>)> {
    let rest = crate::take_grant_args(rest.to_vec())?;
    let mut it = rest.iter();
    let mut params: Option<String> = None;
    let mut bound: Option<Vec<Binding>> = None;
    while let Some(arg) = it.next() {
        let mut next = |name: &str| -> Result<String> {
            it.next()
                .cloned()
                .ok_or_else(|| anyhow!("{name} requires a value"))
        };
        match arg.as_str() {
            "-params" | "--params" => {
                if params.is_some() {
                    bail!("second -params specified");
                }
                params = Some(next(arg)?);
            }
            "-params-from" | "--params-from" => {
                if params.is_some() {
                    bail!("-params-from and -params both name one module's parameters");
                }
                let path = next(arg)?;
                params = Some(
                    std::fs::read_to_string(&path)
                        .with_context(|| format!("reading -params-from {path}"))?,
                );
            }
            "-bound" | "--bound" => {
                if bound.is_some() {
                    bail!("second -bound specified");
                }
                bound = Some(parse_bound(&next(arg)?)?);
            }
            other => bail!("--shape: unknown flag {other}"),
        }
    }
    Ok((params.unwrap_or_default(), bound.unwrap_or_default()))
}

pub fn shape_json(shape: &NodeShape) -> Value {
    json!({
        "inputs": shape.inputs.iter().map(input_json).collect::<Vec<_>>(),
        "outputs": shape.outputs.iter().map(output_json).collect::<Vec<_>>(),
        "clock": clock_json(&shape.clock),
        "pure": shape.pure,
        "one_to_one": shape.one_to_one,
        "bounded": shape.bounded,
        "relation": shape.relation,
    })
}

fn kind_name(kind: PortKind) -> &'static str {
    kind.name()
}

fn rows_name(rows: RowsUse) -> &'static str {
    match rows {
        RowsUse::Ignore => "ignore",
        RowsUse::PerFrame => "per_frame",
        RowsUse::State => "state",
    }
}

fn wants_name(wants: Wants) -> &'static str {
    wants.written()
}

fn rational_json(r: Rational) -> Value {
    json!({"num": r.num, "den": r.den})
}

/// A variant: `kind`, then the payload's fields beside it.
fn variant(kind: &str, fields: Value) -> Value {
    let mut object = Map::new();
    object.insert("kind".to_string(), Value::from(kind));
    if let Value::Object(fields) = fields {
        object.extend(fields);
    }
    Value::Object(object)
}

fn input_json(port: &InputPort) -> Value {
    json!({
        "name": port.name,
        "kind": kind_name(port.kind),
        "required": port.required,
        "many": port.many,
        "pairing": pairing_json(&port.pairing),
        "rows": rows_name(port.rows),
        "window": port.window,
        "stride": port.stride,
        "accepts": accepts_json(&port.accepts),
        "schema": port.schema,
    })
}

fn pairing_json(pairing: &Pairing) -> Value {
    match pairing {
        Pairing::Lockstep => variant("lockstep", json!({})),
        Pairing::Hold(hold) => variant(
            "hold",
            json!({
                "anchor": anchor_json(&hold.anchor),
                "lead": hold.lead,
                "linger": hold.linger,
                "timeout": hold.timeout,
                "group": hold.group,
                "port_param": hold.port_param,
            }),
        ),
        Pairing::Interval(interval) => variant(
            "interval",
            json!({
                "latency": interval.latency,
                "ahead": interval.ahead,
                "anchor": anchor_json(&interval.anchor),
                "group": interval.group,
            }),
        ),
        Pairing::Arrival => variant("arrival", json!({})),
        Pairing::AtOrBefore => variant("at_or_before", json!({})),
    }
}

fn anchor_json(anchor: &Anchor) -> Value {
    match anchor {
        Anchor::SharedClock => variant("shared_clock", json!({})),
        Anchor::FirstFrame => variant("first_frame", json!({})),
        Anchor::Tagged(tag) => variant("tagged", json!({"tag": tag})),
    }
}

fn accepts_json(accepts: &Accepts) -> Value {
    json!({
        "pixel_formats": accepts.pixel_formats,
        "sample_formats": accepts.sample_formats,
        "sample_rates": accepts.sample_rates,
        "channel_counts": accepts.channel_counts,
        "codecs": accepts.codecs,
        "wants": wants_name(accepts.wants),
        "like": accepts.like,
    })
}

fn output_json(port: &OutputPort) -> Value {
    json!({
        "name": port.name,
        "kind": kind_name(port.kind),
        "format": port.format.as_ref().map(format_json),
        "time_base": port.time_base.map(rational_json),
        "latency": port.latency,
        "schema": port.schema,
        "row": port.row,
    })
}

fn color_json(color: &ColorSpec) -> Value {
    json!({
        "range": color.range,
        "primaries": color.primaries,
        "trc": color.trc,
        "space": color.space,
    })
}

fn format_json(format: &OutputFormat) -> Value {
    match format {
        OutputFormat::Video(v) => variant(
            "video",
            json!({
                "width": v.width,
                "height": v.height,
                "pix_fmt": v.pix_fmt,
                "color": v.color.as_ref().map(color_json),
            }),
        ),
        OutputFormat::Audio(a) => variant(
            "audio",
            json!({
                "sample_rate": a.sample_rate,
                "channels": a.channels,
                "sample_fmt": a.sample_fmt,
                "channel_layout": a.channel_layout,
            }),
        ),
        OutputFormat::Data(codec) => variant("data", json!({"codec": codec})),
        OutputFormat::Packets(coded) => variant("packets", coded_json(coded)),
        OutputFormat::Like(like) => variant(
            "like",
            json!({
                "port": like.port,
                "pixel_format": like.pixel_format,
                "sample_format": like.sample_format,
            }),
        ),
    }
}

fn coded_json(coded: &CodedStream) -> Value {
    let format = match &coded.format {
        CodedFormat::Video {
            width,
            height,
            sample_aspect_ratio,
            color,
        } => variant(
            "video",
            json!({
                "width": width,
                "height": height,
                "sample_aspect_ratio": sample_aspect_ratio.map(|(num, den)| json!({"num": num, "den": den})),
                "color": color.map(|c| json!({
                    "range": c.range,
                    "primaries": c.primaries,
                    "trc": c.trc,
                    "space": c.space,
                })),
            }),
        ),
        CodedFormat::Audio {
            sample_rate,
            channels,
            channel_layout,
        } => variant(
            "audio",
            json!({
                "sample_rate": sample_rate,
                "channels": channels,
                "channel_layout": channel_layout,
            }),
        ),
        CodedFormat::Data => variant("data", json!({})),
    };
    json!({
        "codec": coded.codec,
        "time_base": {"num": coded.time_base.num, "den": coded.time_base.den},
        "format": format,
        "extradata": crate::to_hex(&coded.extradata),
        "profile": coded.profile,
        "level": coded.level,
    })
}

fn clock_json(clock: &Clock) -> Value {
    match clock {
        Clock::Input(port) => variant("input", json!({"port": port})),
        Clock::Rate(rate) => variant("rate", rational_json(*rate)),
        Clock::RateOf(port) => variant("rate_of", json!({"port": port})),
        Clock::SelfClocked => variant("self_clocked", json!({})),
    }
}
