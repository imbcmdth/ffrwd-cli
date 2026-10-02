//! The host's own nodes on a data edge of a node network: `rowfilter`,
//! `rowmerge` over runs of rows, and `rowmerge=max_span=<s>`, the span
//! reducer, which turns rows written once per tick into one row per span.
//!
//! Each reads one data stream and writes one. The host hands it the
//! messages between two of its producer's progress marks as one tick, so a
//! tick's end is the end of the producer's tick that carried them.

use std::collections::{BTreeMap, HashMap};

use anyhow::{anyhow, bail, Result};
use ffrwd_wasm_runtime::node::{
    Accepts, Clock, Emission, Emitted, InputPort, Node, NodeShape, OutputFormat, OutputPort,
    Pairing, Payload, PortKind, RowsUse, Tick,
};
use ffrwd_wasm_runtime::runtime::{self, Message, TimeBase};
use serde_json::{Map, Value};

use crate::rowfilter::{self, RowFilter};
use crate::rowmerge::{self, RowMerge};

/// The option that makes `rowmerge` the span reducer.
const MAX_SPAN: &str = "max_span";

const START: &str = "start_t";
const END: &str = "end_t";
const ID: &str = "id";

/// Whether `module` names a node the host answers for on a data edge.
pub fn is_host_node(module: &str) -> bool {
    module == rowfilter::NODE || module == rowmerge::NODE
}

/// Opens the host node a chain names, reading `base` on its input.
pub fn open(module: &str, options: &[(String, String)], base: TimeBase) -> Result<Box<dyn Node>> {
    if module == rowfilter::NODE {
        return Ok(Box::new(HostNode::new(
            module,
            0.0,
            Reducer::Filter(RowFilter::open(options)?),
        )));
    }
    if options.iter().any(|(k, _)| k == MAX_SPAN) {
        let spans = Spans::open(options, base)?;
        let latency = spans.max_span;
        return Ok(Box::new(HostNode::new(
            module,
            latency,
            Reducer::Spans(spans),
        )));
    }
    Ok(Box::new(HostNode::new(
        module,
        0.0,
        Reducer::Runs(RowMerge::open(options)?, None),
    )))
}

enum Reducer {
    Filter(RowFilter),
    /// The run merge, and the pts of the last row it was handed.
    Runs(RowMerge, Option<i64>),
    Spans(Spans),
}

struct HostNode {
    name: String,
    shape: NodeShape,
    reducer: Reducer,
}

impl HostNode {
    fn new(name: &str, latency: f64, reducer: Reducer) -> HostNode {
        let data = Some(OutputFormat::Data(runtime::DATA_CODEC.into()));
        HostNode {
            name: name.to_string(),
            shape: NodeShape {
                inputs: vec![InputPort {
                    name: "in".into(),
                    kind: PortKind::Data,
                    required: true,
                    many: false,
                    pairing: Pairing::Arrival,
                    rows: RowsUse::PerFrame,
                    window: 1,
                    stride: 1,
                    accepts: Accepts::default(),
                    schema: None,
                }],
                outputs: vec![OutputPort {
                    name: "out".into(),
                    kind: PortKind::Data,
                    format: data,
                    time_base: None,
                    latency,
                    schema: None,
                    row: None,
                }],
                clock: Clock::SelfClocked,
                pure: false,
                one_to_one: false,
                bounded: true,
                relation: Vec::new(),
            },
            reducer,
        }
    }
}

fn message(pts: i64, row: String) -> Emission {
    Emission {
        port: 0,
        payload: Payload::Message(Message {
            pts,
            data: row.into_bytes(),
        }),
    }
}

impl Node for HostNode {
    fn name(&self) -> &str {
        &self.name
    }

    fn shape(&self) -> &NodeShape {
        &self.shape
    }

    fn set_params(&mut self, _params: &str) -> Result<()> {
        bail!(
            "{} takes its options from the command line and no params",
            self.name
        )
    }

    fn process(&mut self, tick: Tick) -> Result<Emitted> {
        let messages: Vec<Message> = tick.streams.into_iter().flat_map(|s| s.messages).collect();
        let mut items = Vec::new();
        match &mut self.reducer {
            Reducer::Filter(filter) => {
                for m in messages {
                    let row = String::from_utf8_lossy(&m.data).into_owned();
                    for kept in filter.keep(vec![row]) {
                        items.push(message(m.pts, kept));
                    }
                }
            }
            Reducer::Runs(merge, last) => {
                for m in messages {
                    *last = Some(m.pts);
                    let row = String::from_utf8_lossy(&m.data).into_owned();
                    for closed in merge.merged(vec![row]) {
                        items.push(message(m.pts, closed));
                    }
                }
                if tick.last {
                    let at = last.unwrap_or(tick.pts);
                    items.extend(merge.finish().into_iter().map(|row| message(at, row)));
                }
            }
            Reducer::Spans(spans) => {
                for m in &messages {
                    spans.row(m)?;
                }
                let end = if tick.last { None } else { Some(tick.pts) };
                for (pts, row) in spans.settle(end) {
                    items.push(message(pts, row));
                }
            }
        }
        Ok(Emitted {
            items,
            rows: Vec::new(),
            finished: false,
        })
    }
}

/// One span: the rows that share a `start_t` and an `id`, from `from` on.
struct Span {
    /// The `start_t` bits and the `id` text it was opened under.
    key: (u64, Option<String>),
    /// Where this span starts, in seconds: the rows' `start_t`, or the point
    /// an earlier stretch of them was cut at.
    from: f64,
    /// The newest row, whose fields the span keeps.
    row: Option<Map<String, Value>>,
    /// The pts of the newest row, and where the tick that carried it ended,
    /// once known.
    at: i64,
    end: Option<i64>,
}

/// The span reducer: rows sharing a `start_t`, and an `id` where they carry
/// one, are one span, which keeps the last row's fields and ends at the end
/// of the last tick that carried one; a tick with none is a gap inside it. A
/// span leaves once its producer is `max_span` past its start, cut there if
/// it is still going, and goes on from the cut as a new one.
struct Spans {
    max_span: f64,
    base: TimeBase,
    /// Open spans in the order they were first seen.
    open: BTreeMap<u64, Span>,
    /// Each open span's place in `open`, by its key.
    keys: HashMap<(u64, Option<String>), u64>,
    opened: u64,
    /// The newest tick length seen, for the rows of a last tick nothing
    /// ended.
    length: Option<i64>,
    /// The pts of the last row out, which none after it may precede.
    out: Option<i64>,
}

impl Spans {
    fn open(options: &[(String, String)], base: TimeBase) -> Result<Spans> {
        let mut max_span: Option<f64> = None;
        for (key, value) in options {
            if key != MAX_SPAN {
                bail!(
                    "{} has no option '{key}' beside {MAX_SPAN}; it takes {MAX_SPAN}=<seconds>",
                    rowmerge::NODE
                );
            }
            let seconds: f64 = value
                .parse()
                .map_err(|_| anyhow!("{}: {MAX_SPAN} is not a number: {value}", rowmerge::NODE))?;
            if !seconds.is_finite() || seconds <= 0.0 {
                bail!(
                    "{}: {MAX_SPAN} is a length in seconds above zero and cannot be {value}",
                    rowmerge::NODE
                );
            }
            max_span = Some(seconds);
        }
        Ok(Spans {
            max_span: max_span.expect("opened only with max_span given"),
            base,
            open: BTreeMap::new(),
            keys: HashMap::new(),
            opened: 0,
            length: None,
            out: None,
        })
    }

    fn pts(&self, seconds: f64) -> i64 {
        (seconds * self.base.den as f64 / self.base.num.max(1) as f64).floor() as i64
    }

    /// One row in. A row with no `start_t` is no span's, and is dropped.
    fn row(&mut self, message: &Message) -> Result<()> {
        let Ok(Value::Object(row)) = serde_json::from_slice::<Value>(&message.data) else {
            return Ok(());
        };
        let Some(start) = row.get(START).and_then(Value::as_f64) else {
            return Ok(());
        };
        let key = (start.to_bits(), row.get(ID).map(Value::to_string));
        let place = *self.keys.entry(key.clone()).or_insert_with(|| {
            self.opened += 1;
            self.opened
        });
        let span = self.open.entry(place).or_insert(Span {
            key,
            from: start,
            row: None,
            at: message.pts,
            end: None,
        });
        span.row = Some(row);
        span.at = message.pts;
        span.end = None;
        Ok(())
    }

    /// The tick ending at `end` has been handed whole (None: the input has
    /// ended). The spans it closes leave, oldest start first.
    fn settle(&mut self, end: Option<i64>) -> Vec<(i64, String)> {
        if let Some(end) = end {
            for span in self.open.values_mut() {
                if span.row.is_some() && span.end.is_none() && span.at < end {
                    span.end = Some(end);
                    self.length = Some(end - span.at);
                }
            }
        }
        let base = self.base;
        let now = end.map(|e| seconds(base, e));
        let length = self.length.unwrap_or(0);
        let mut leaving: Vec<(f64, Map<String, Value>)> = Vec::new();
        let mut gone = Vec::new();
        for (place, span) in self.open.iter_mut() {
            let cut = span.from + self.max_span;
            if now.is_some_and(|now| now < cut) {
                continue;
            }
            let Some(mut row) = span.row.take() else {
                gone.push(*place);
                continue;
            };
            let mut ended = seconds(base, span.end.unwrap_or(span.at + length));
            if now.is_some() {
                ended = ended.min(cut);
            }
            row.insert(START.into(), number(span.from));
            row.insert(END.into(), number(ended));
            leaving.push((span.from, row));
            if now.is_some() {
                span.from = cut;
                span.end = None;
            } else {
                gone.push(*place);
            }
        }
        for place in gone {
            if let Some(span) = self.open.remove(&place) {
                self.keys.remove(&span.key);
            }
        }
        leaving.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut out = Vec::with_capacity(leaving.len());
        for (from, row) in leaving {
            let pts = self.pts(from).max(self.out.unwrap_or(i64::MIN));
            self.out = Some(pts);
            out.push((pts, Value::Object(row).to_string()));
        }
        out
    }
}

fn seconds(base: TimeBase, pts: i64) -> f64 {
    pts as f64 * base.num as f64 / base.den.max(1) as f64
}

fn number(value: f64) -> Value {
    serde_json::Number::from_f64(value).map_or(Value::Null, Value::Number)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ffrwd_wasm_runtime::node::TickStream;

    const FRAMES: TimeBase = TimeBase { num: 1, den: 10 };

    fn tick(pts: i64, last: bool, rows: &[(i64, &str)]) -> Tick {
        Tick {
            pts,
            time_base: FRAMES,
            last,
            streams: vec![TickStream {
                id: 0,
                messages: rows
                    .iter()
                    .map(|(pts, row)| Message {
                        pts: *pts,
                        data: row.as_bytes().to_vec(),
                    })
                    .collect(),
                ..TickStream::default()
            }],
        }
    }

    fn rows(emitted: Emitted) -> Vec<(i64, Value)> {
        emitted
            .items
            .into_iter()
            .map(|item| match item.payload {
                Payload::Message(m) => (m.pts, serde_json::from_slice(&m.data).unwrap()),
                _ => panic!("a host node writes messages"),
            })
            .collect()
    }

    fn spans(max_span: &str) -> Box<dyn Node> {
        open(
            rowmerge::NODE,
            &[(MAX_SPAN.into(), max_span.into())],
            FRAMES,
        )
        .expect("opens")
    }

    #[test]
    fn rows_of_one_start_are_one_span_and_a_gap_stays_inside_it() {
        let mut node = spans("1");
        let a = r#"{"start_t":0.0,"id":0,"x":1}"#;
        let b = r#"{"start_t":0.0,"id":0,"x":2}"#;
        assert!(rows(node.process(tick(1, false, &[(0, a)])).unwrap()).is_empty());
        assert!(rows(node.process(tick(2, false, &[])).unwrap()).is_empty());
        assert!(rows(node.process(tick(3, false, &[(2, b)])).unwrap()).is_empty());
        let out = rows(node.process(tick(10, false, &[])).unwrap());
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].0, 0);
        assert_eq!(out[0].1["x"], 2);
        assert_eq!(out[0].1[END], 0.3);
    }

    #[test]
    fn two_ids_first_seen_on_one_tick_are_two_spans_and_no_id_is_one() {
        let mut node = spans("1");
        let a = r#"{"start_t":0.0,"id":0,"code":"a"}"#;
        let b = r#"{"start_t":0.0,"id":1,"code":"b"}"#;
        node.process(tick(1, false, &[(0, a), (0, b)])).unwrap();
        node.process(tick(2, false, &[(1, a)])).unwrap();
        let out = rows(node.process(tick(10, false, &[])).unwrap());
        let ends: Vec<(&str, f64)> = out
            .iter()
            .map(|(_, r)| (r["code"].as_str().unwrap(), r[END].as_f64().unwrap()))
            .collect();
        assert_eq!(ends, vec![("a", 0.2), ("b", 0.1)]);

        let mut node = spans("1");
        let a = r#"{"start_t":0.0,"code":"a"}"#;
        let b = r#"{"start_t":0.0,"code":"b"}"#;
        node.process(tick(1, false, &[(0, a), (0, b)])).unwrap();
        let out = rows(node.process(tick(10, false, &[])).unwrap());
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].1["code"], "b");
    }

    #[test]
    fn a_span_longer_than_max_span_is_cut_and_goes_on() {
        let mut node = spans("0.5");
        let row = r#"{"start_t":0.0,"id":0}"#;
        let mut out = Vec::new();
        for n in 0..8 {
            out.extend(rows(node.process(tick(n + 1, false, &[(n, row)])).unwrap()));
        }
        out.extend(rows(node.process(tick(8, true, &[])).unwrap()));
        let spans: Vec<(f64, f64)> = out
            .iter()
            .map(|(_, r)| (r[START].as_f64().unwrap(), r[END].as_f64().unwrap()))
            .collect();
        assert_eq!(spans, vec![(0.0, 0.5), (0.5, 0.8)]);
    }

    #[test]
    fn the_filter_keeps_what_its_predicate_keeps() {
        let mut node = open(
            rowfilter::NODE,
            &[("pred".into(), r#"{"ge":[{"field":"w"},{"lit":20}]}"#.into())],
            FRAMES,
        )
        .unwrap();
        let out = rows(
            node.process(tick(1, false, &[(0, r#"{"w":30}"#), (0, r#"{"w":3}"#)]))
                .unwrap(),
        );
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].1["w"], 30);
    }
}
