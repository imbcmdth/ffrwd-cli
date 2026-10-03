//! Every older world's module as a node.
//!
//! Each adapter has two shapes. The declared one is what `--shape` prints:
//! the old world in the node world's terms, so a compiler reads every module
//! through one record (an annotation column is a data input, annotation rows
//! a data output, a feeder a hold input). The other is what the host runs it
//! by, which keeps what that world's host did where the node world says it
//! differently: rows riding frames, a data filter's messages at or before
//! its clock, a packet filter's output per input stream.

use std::sync::Arc;

use anyhow::{anyhow, bail, Context, Result};
use ffrwd_wasm_runtime::node::{
    self, Accepts, Anchor, BoundStream, Clock, Emission, Emitted, Hold, InputPort, Interval,
    LikeInput, Node, NodeShape, OutFrame, OutputFormat, OutputPort, Pairing, Payload, PortKind,
    Rational, RowsUse, StreamFormat, Tick, TickStream,
};
use ffrwd_wasm_runtime::runtime::{
    self, Arity, Catalog, CodedFormat, CodedStream, DataFilter, DataPad, Filter, Format, Frame,
    Kind, Message, Packet, PacketFilter, PacketSink, PacketSource, PadKind, RowsModule, SinkInput,
    TimeBase,
};

use crate::heartbeat::{self, Beats};
use crate::tick::compare;

pub(crate) fn input(
    name: &str,
    kind: PortKind,
    required: bool,
    many: bool,
    pairing: Pairing,
    rows: RowsUse,
) -> InputPort {
    InputPort {
        name: name.to_string(),
        kind,
        required,
        many,
        pairing,
        rows,
        window: 1,
        stride: 1,
        accepts: Accepts::default(),
        schema: None,
    }
}

pub(crate) fn output(name: &str, kind: PortKind, format: Option<OutputFormat>) -> OutputPort {
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

/// A data output carrying a module's rows, one message per row at the time
/// it was made, in the clock's time base.
fn rows_output(name: &str, schema: &str) -> OutputPort {
    OutputPort {
        schema: (!schema.is_empty()).then(|| schema.to_string()),
        ..output(
            name,
            PortKind::Data,
            Some(OutputFormat::Data(runtime::DATA_CODEC.into())),
        )
    }
}

fn kind_of(kind: Kind) -> PortKind {
    match kind {
        Kind::Video => PortKind::Video,
        Kind::Audio => PortKind::Audio,
    }
}

fn accepts_of(meta: &runtime::Meta) -> Accepts {
    Accepts {
        pixel_formats: meta.pixel_formats.clone(),
        sample_formats: meta.sample_formats.clone(),
        sample_rates: meta.sample_rates.clone(),
        channel_counts: meta.channel_counts.clone(),
        ..Accepts::default()
    }
}

/// The shape `--shape` prints for a module of an older world.
pub fn declared_shape(module: &str, params: &str, bound: &[String]) -> Result<NodeShape> {
    let shape = if runtime::exports_window_filter(module)? || runtime::exports_filter(module)? {
        frame_shape(&runtime::describe(module)?)?
    } else if runtime::exports_packet_sink(module)? {
        sink_shape(&runtime::describe_packet_sink(module)?)
    } else if runtime::exports_packet_filter(module)? {
        filter_shape(&runtime::describe_packet_filter(module)?)
    } else if runtime::exports_packet_source(module)? {
        source_shape(&PacketSource::probe(module, params)?)
    } else if runtime::exports_data_filter(module)? {
        data_filter_shape(&runtime::describe_data_filter(module)?, bound)
    } else if runtime::exports_rows_module(module)? {
        rows_shape(&runtime::describe_rows_module(module)?)
    } else {
        let exports = runtime::exports(module)?;
        bail!(
            "{module} exports no node and no stream interface, so it has no shape to give; it \
             exports {}",
            if exports.is_empty() {
                "nothing".to_string()
            } else {
                exports.join(", ")
            }
        );
    };
    node::check_shape(&shape, &node::Binding::from_names(bound), module)?;
    Ok(shape)
}

/// A filter, meta-filter or window-filter: stream arguments in order, the
/// first that is not a feeder the clock with the module's window, the rest
/// lockstep with it, feeders as hold inputs on the port their param names;
/// the annotation column a data input and the rows a data output.
fn frame_shape(described: &runtime::Described) -> Result<NodeShape> {
    let kind = kind_of(described.meta.kind()?);
    let shape = described.shape.unwrap_or(runtime::Shape {
        window: 1,
        stride: 1,
        pure: false,
        one_to_one: true,
    });
    let arguments = described.inputs as usize + described.feeders.len();
    let mut inputs = Vec::with_capacity(arguments + 1);
    let mut clock: Option<String> = None;
    for argument in 0..arguments {
        let name = format!("in{argument}");
        let feeder = described
            .feeders
            .iter()
            .find(|f| f.input as usize == argument);
        if let Some(feeder) = feeder {
            let feeder_kind = if feeder.kind == "audio" {
                PortKind::Audio
            } else {
                PortKind::Video
            };
            inputs.push(input(
                &name,
                feeder_kind,
                false,
                false,
                Pairing::Hold(Hold {
                    anchor: Anchor::FirstFrame,
                    lead: 0.0,
                    linger: None,
                    timeout: None,
                    group: (!feeder.group.is_empty()).then(|| feeder.group.clone()),
                    port_param: Some(feeder.port_param.clone()),
                }),
                RowsUse::Ignore,
            ));
            continue;
        }
        let mut port = input(&name, kind, true, false, Pairing::Lockstep, RowsUse::Ignore);
        port.accepts = accepts_of(&described.meta);
        if clock.is_none() {
            port.window = shape.window;
            port.stride = shape.stride;
            clock = Some(name);
        }
        inputs.push(port);
    }
    if described.reads_rows {
        inputs.push(input(
            "rows",
            PortKind::Data,
            false,
            false,
            Pairing::Lockstep,
            RowsUse::PerFrame,
        ));
    }
    let mut outputs = vec![output("out", kind, None)];
    if !described.meta.rows_schema.is_empty() {
        outputs.push(rows_output("rows", &described.meta.rows_schema));
    }
    Ok(NodeShape {
        inputs,
        outputs,
        clock: Clock::Input(clock.ok_or_else(|| anyhow!("the module reads no stream"))?),
        pure: shape.pure,
        one_to_one: shape.one_to_one,
        bounded: true,
        relation: Vec::new(),
    })
}

/// Coded inputs by kind, each a many-port its arity allows, taken as they
/// arrive.
fn coded_inputs(
    video: (Arity, &[String]),
    audio: (Arity, &[String]),
    data: Arity,
) -> Vec<InputPort> {
    let mut inputs = Vec::new();
    for (name, arity, codecs, kind) in [
        ("video", video.0, video.1, PortKind::Packets),
        ("audio", audio.0, audio.1, PortKind::Packets),
        ("data", data, &[][..], PortKind::Data),
    ] {
        if arity == Arity::Zero {
            continue;
        }
        let rows = if kind == PortKind::Data {
            RowsUse::PerFrame
        } else {
            RowsUse::Ignore
        };
        let mut port = input(
            name,
            kind,
            matches!(arity, Arity::One | Arity::Many),
            arity != Arity::One,
            Pairing::Arrival,
            rows,
        );
        port.accepts.codecs = codecs.to_vec();
        inputs.push(port);
    }
    inputs
}

/// A packet sink: its coded inputs as they arrive, called at least every
/// fiftieth of a second; its rows are the run's.
fn sink_shape(described: &runtime::DescribedPacketSink) -> NodeShape {
    let mut inputs = coded_inputs(
        (described.video, &described.video_codecs),
        (described.audio, &described.audio_codecs),
        described.data,
    );
    for port in &mut inputs {
        port.accepts.wants = described.wants;
    }
    NodeShape {
        inputs,
        outputs: Vec::new(),
        clock: Clock::Rate(SINK_RATE),
        pure: false,
        one_to_one: false,
        bounded: true,
        relation: Vec::new(),
    }
}

/// How often a packet sink nothing reaches is called anyway.
pub const SINK_RATE: Rational = Rational { num: 50, den: 1 };

/// A packet filter: its coded inputs and rows as they arrive, each kind's
/// streams handed back on the output of that kind's name, one stream per
/// stream bound.
fn filter_shape(described: &runtime::DescribedPacketFilter) -> NodeShape {
    let mut inputs = coded_inputs(
        (described.video, &described.video_codecs),
        (described.audio, &described.audio_codecs),
        described.data,
    );
    let outputs = inputs
        .iter()
        .map(|port| output(&port.name, port.kind, None))
        .collect();
    inputs.push(input(
        "rows",
        PortKind::Data,
        described.reads_rows,
        true,
        Pairing::Arrival,
        RowsUse::PerFrame,
    ));
    NodeShape {
        inputs,
        outputs,
        clock: Clock::SelfClocked,
        pure: false,
        one_to_one: false,
        bounded: true,
        relation: Vec::new(),
    }
}

/// A packet source: no inputs, an output per catalog track, its rows the
/// relation.
fn source_shape(catalog: &Catalog) -> NodeShape {
    let mut relation: Vec<(u32, String)> = Vec::new();
    let mut outputs = Vec::with_capacity(catalog.tracks.len());
    for (index, track) in catalog.tracks.iter().enumerate() {
        if !relation.iter().any(|(row, _)| *row == track.row) {
            let r = &track.rendition;
            relation.push((
                track.row,
                serde_json::json!({
                    "row": track.row,
                    "name": r.name,
                    "bandwidth": r.bandwidth,
                    "codecs": r.codecs,
                    "language": r.language,
                })
                .to_string(),
            ));
        }
        let (kind, format) = match track.stream.format {
            CodedFormat::Data => (
                PortKind::Data,
                OutputFormat::Data(track.stream.codec.clone()),
            ),
            _ => (
                PortKind::Packets,
                OutputFormat::Packets(track.stream.clone()),
            ),
        };
        let row = relation
            .iter()
            .position(|(row, _)| *row == track.row)
            .map(|i| i as u32);
        outputs.push(OutputPort {
            row,
            ..output(&format!("track{index}"), kind, Some(format))
        });
    }
    NodeShape {
        inputs: Vec::new(),
        outputs,
        clock: Clock::SelfClocked,
        pure: false,
        one_to_one: false,
        bounded: catalog.bounded,
        relation: relation.into_iter().map(|(_, r)| r).collect(),
    }
}

/// A data filter: its data pads by interval with no wait, its first clock
/// pad the clock; with no clock bound, its pads as they arrive.
fn data_filter_shape(described: &runtime::DescribedDataFilter, bound: &[String]) -> NodeShape {
    let clocked = bound.iter().any(|b| b == "clock");
    let pairing = if clocked {
        Pairing::Interval(Interval::shared(Some(0.0), 0.0))
    } else {
        Pairing::Arrival
    };
    let mut inputs = vec![input(
        "data",
        PortKind::Data,
        false,
        true,
        pairing,
        RowsUse::PerFrame,
    )];
    if clocked {
        inputs.push(input(
            "clock",
            PortKind::Video,
            true,
            false,
            Pairing::Lockstep,
            RowsUse::Ignore,
        ));
    }
    let base = Rational {
        num: i32::try_from(described.time_base.num).unwrap_or(i32::MAX),
        den: i32::try_from(described.time_base.den).unwrap_or(i32::MAX),
    };
    let outputs = described
        .outputs
        .iter()
        .enumerate()
        .map(|(index, codec)| OutputPort {
            time_base: Some(base),
            ..output(
                &format!("out{index}"),
                PortKind::Data,
                Some(OutputFormat::Data(codec.clone())),
            )
        })
        .collect();
    NodeShape {
        inputs,
        outputs,
        clock: if clocked {
            Clock::Input("clock".into())
        } else {
            Clock::SelfClocked
        },
        pure: false,
        one_to_one: false,
        bounded: true,
        relation: Vec::new(),
    }
}

/// A rows module: one data input that is its own clock, one data output.
fn rows_shape(described: &runtime::DescribedRowsModule) -> NodeShape {
    let mut rows = input(
        "rows",
        PortKind::Data,
        true,
        false,
        Pairing::Lockstep,
        RowsUse::PerFrame,
    );
    rows.schema =
        (!described.input_rows_schema.is_empty()).then(|| described.input_rows_schema.clone());
    NodeShape {
        inputs: vec![rows],
        outputs: vec![rows_output("out", &described.meta.rows_schema)],
        clock: Clock::Input("rows".into()),
        pure: false,
        one_to_one: false,
        bounded: true,
        relation: Vec::new(),
    }
}

/// A frame module of any older world: its window over the clock pad, any
/// further pads at the clock's exact pts, rows riding the frames as that
/// world's host carried them.
pub struct FilterNode {
    filter: Filter,
    shape: NodeShape,
}

impl FilterNode {
    pub fn open(
        path: &str,
        format: &Format,
        info: &runtime::StreamInfo,
        params: &str,
    ) -> Result<FilterNode> {
        Ok(FilterNode::wrap(
            Filter::open(path, format, info, params)?,
            format,
        ))
    }

    pub fn wrap(filter: Filter, format: &Format) -> FilterNode {
        let kind = kind_of(format.kind());
        let shape = filter.shape();
        let pads = filter.inputs() as usize;
        let mut inputs = Vec::with_capacity(pads);
        for pad in 0..pads {
            let mut port = input(
                &format!("in{pad}"),
                kind,
                true,
                false,
                Pairing::Lockstep,
                if pad == 0 {
                    RowsUse::PerFrame
                } else {
                    RowsUse::Ignore
                },
            );
            port.accepts = accepts_of(filter.meta());
            if pad == 0 {
                port.window = shape.window;
                port.stride = shape.stride;
            }
            inputs.push(port);
        }
        FilterNode {
            shape: NodeShape {
                inputs,
                outputs: vec![output("out", kind, None)],
                clock: Clock::Input("in0".into()),
                pure: shape.pure,
                one_to_one: shape.one_to_one,
                bounded: true,
                relation: Vec::new(),
            },
            filter,
        }
    }
}

impl Node for FilterNode {
    fn name(&self) -> &str {
        self.filter.name()
    }

    fn shape(&self) -> &NodeShape {
        &self.shape
    }

    fn host_progress(&self) -> bool {
        false
    }

    fn set_params(&mut self, params: &str) -> Result<()> {
        self.filter.set_params(params)
    }

    fn process(&mut self, tick: Tick) -> Result<Emitted> {
        let mut streams = tick.streams.into_iter();
        let first = streams.next().unwrap_or_default();
        let trailing = first.trailing;
        let mut frames: Vec<Frame> = first.frames.into_iter().map(frame_of).collect();
        for stream in streams {
            frames.extend(stream.frames.into_iter().map(frame_of));
        }
        let processed = self.filter.process_window(&frames, &trailing, tick.last)?;
        let mut items: Vec<Emission> = processed
            .frames
            .into_iter()
            .map(|f| Emission {
                port: 0,
                payload: Payload::Frame(OutFrame {
                    pts: f.pts,
                    duration: None,
                    data: f.data,
                    rows: f.rows,
                }),
            })
            .collect();
        if !processed.trailing.is_empty() {
            items.push(Emission {
                port: 0,
                payload: Payload::Rows(processed.trailing),
            });
        }
        Ok(Emitted {
            items,
            rows: Vec::new(),
            finished: false,
        })
    }
}

fn frame_of(frame: node::TickFrame) -> Frame {
    Frame {
        pts: frame.pts,
        data: frame.data,
        rows: frame.rows,
    }
}

/// A call's frames and trailing rows as one tick: a window off the first
/// pad, or one frame off each of several.
pub fn frames_tick(
    frames: &[Frame],
    trailing: &[String],
    last: bool,
    pads: usize,
    base: TimeBase,
) -> Tick {
    let tick_frame = |f: &Frame| node::TickFrame {
        pts: f.pts,
        duration: None,
        data: Arc::clone(&f.data),
        rows: f.rows.clone(),
    };
    let mut streams: Vec<TickStream> = (0..pads.max(1))
        .map(|pad| TickStream {
            id: pad as u32,
            ..TickStream::default()
        })
        .collect();
    if pads <= 1 {
        streams[0].frames = frames.iter().map(tick_frame).collect();
    } else {
        for (pad, frame) in frames.iter().enumerate() {
            streams[pad].frames.push(tick_frame(frame));
        }
    }
    streams[0].trailing = trailing.to_vec();
    Tick {
        pts: frames.first().map_or(0, |f| f.pts),
        ordinal: 0,
        time_base: base,
        last,
        streams,
    }
}

/// A pad of a packet sink or filter as its world's `init` is told it.
pub fn sink_input(stream: &BoundStream) -> Result<SinkInput> {
    let coded = match &stream.format {
        StreamFormat::Packets(coded) => coded.clone(),
        StreamFormat::Data(codec) => CodedStream {
            codec: codec.clone(),
            time_base: stream.time_base,
            format: CodedFormat::Data,
            extradata: Vec::new(),
            profile: None,
            level: None,
        },
        StreamFormat::Video(_) | StreamFormat::Audio(_) => {
            bail!(
                "pad {} carries frames, and a packet module reads packets",
                stream.id
            )
        }
    };
    Ok(SinkInput {
        stream: coded,
        info: stream.info.clone(),
        row: stream.row.unwrap_or(stream.id),
        rendition: stream.rendition.clone(),
        decode_delay: stream.decode_delay,
    })
}

/// Each pad's packets taken out of a tick's streams, in pad order, a data
/// pad's messages as the packets an older world hands over: each a keyframe
/// at its own time.
fn pads_of(streams: &mut [TickStream], ids: &[u32]) -> Vec<Vec<Packet>> {
    ids.iter()
        .map(|id| {
            let Some(stream) = streams.iter_mut().find(|s| s.id == *id) else {
                return Vec::new();
            };
            let mut packets = std::mem::take(&mut stream.packets);
            packets.extend(
                std::mem::take(&mut stream.messages)
                    .into_iter()
                    .map(|m| Packet {
                        pts: m.pts,
                        dts: Some(m.pts),
                        duration: None,
                        keyframe: true,
                        data: m.data,
                    }),
            );
            packets
        })
        .collect()
}

/// The ports coded pads bind, by the kind they carry.
fn coded_ports(bound: &[BoundStream]) -> Vec<InputPort> {
    let mut ports: Vec<InputPort> = Vec::new();
    for stream in bound {
        let kind = stream.format.kind();
        let name = match &stream.format {
            StreamFormat::Packets(coded) => coded.format.kind(),
            _ => "data",
        };
        if ports.iter().any(|p| p.name == name) {
            continue;
        }
        let rows = if kind == PortKind::Data {
            RowsUse::PerFrame
        } else {
            RowsUse::Ignore
        };
        ports.push(input(name, kind, true, true, Pairing::Arrival, rows));
    }
    ports
}

/// A packet sink: whatever arrived on each pad, or nothing, at least every
/// fiftieth of a second; rows out.
pub struct PacketSinkNode {
    sink: PacketSink,
    shape: NodeShape,
    /// The bound streams' ids in pad order.
    pads: Vec<u32>,
}

impl PacketSinkNode {
    /// Opens the sink on `bound`, one stream per pad, ids its pad order.
    pub fn open(path: &str, bound: &[BoundStream], params: &str) -> Result<PacketSinkNode> {
        let inputs = bound.iter().map(sink_input).collect::<Result<Vec<_>>>()?;
        let sink = PacketSink::open(path, &inputs, params)?;
        Ok(PacketSinkNode {
            sink,
            shape: NodeShape {
                inputs: coded_ports(bound),
                outputs: Vec::new(),
                clock: Clock::Rate(SINK_RATE),
                pure: false,
                one_to_one: false,
                bounded: true,
                relation: Vec::new(),
            },
            pads: bound.iter().map(|b| b.id).collect(),
        })
    }
}

impl Node for PacketSinkNode {
    fn name(&self) -> &str {
        self.sink.name()
    }

    fn shape(&self) -> &NodeShape {
        &self.shape
    }

    fn host_progress(&self) -> bool {
        false
    }

    fn set_params(&mut self, params: &str) -> Result<()> {
        self.sink.set_params(params)
    }

    fn process(&mut self, mut tick: Tick) -> Result<Emitted> {
        let pads = pads_of(&mut tick.streams, &self.pads);
        let emitted = self.sink.process(&pads, tick.last)?;
        let mut rows = emitted.rows;
        rows.extend(emitted.trailing);
        Ok(Emitted {
            items: Vec::new(),
            rows,
            finished: false,
        })
    }
}

/// A packet filter: packets and rows as they arrive, each pad handed back
/// on the output of the same index, a data pad's messages placed on its
/// output's timeline with the heartbeats that arrived beside them.
pub struct PacketFilterNode {
    filter: PacketFilter,
    shape: NodeShape,
    pads: Vec<u32>,
    /// The stream the rows arrive on, if any do.
    rows: Option<u32>,
    /// Per pad, the timeline of a data pad's output.
    beats: Vec<Option<Beats>>,
}

impl PacketFilterNode {
    /// Opens the filter on `bound`'s pads, one stream per pad, ids their pad
    /// order, beside the stream `rows` names, if any.
    pub fn open(
        path: &str,
        bound: &[BoundStream],
        rows: Option<u32>,
        params: &str,
    ) -> Result<PacketFilterNode> {
        let pads: Vec<&BoundStream> = bound.iter().filter(|b| Some(b.id) != rows).collect();
        let inputs = pads
            .iter()
            .map(|b| sink_input(b))
            .collect::<Result<Vec<_>>>()?;
        let filter = PacketFilter::open(path, &inputs, params)?;
        let mut ports: Vec<InputPort> = Vec::new();
        let mut outputs = Vec::new();
        for stream in &pads {
            let name = format!("in{}", stream.id);
            let kind = stream.format.kind();
            let rows_use = if kind == PortKind::Data {
                RowsUse::PerFrame
            } else {
                RowsUse::Ignore
            };
            ports.push(input(&name, kind, true, false, Pairing::Arrival, rows_use));
            outputs.push(output(
                &format!("out{}", stream.id),
                kind,
                Some(OutputFormat::Like(LikeInput {
                    port: name,
                    pixel_format: None,
                    sample_format: None,
                })),
            ));
        }
        if rows.is_some() {
            ports.push(input(
                "rows",
                PortKind::Data,
                false,
                false,
                Pairing::Arrival,
                RowsUse::PerFrame,
            ));
        }
        let beats = pads
            .iter()
            .map(|b| (b.format.kind() == PortKind::Data).then(|| Beats::new(b.time_base)))
            .collect();
        Ok(PacketFilterNode {
            filter,
            shape: NodeShape {
                inputs: ports,
                outputs,
                clock: Clock::SelfClocked,
                pure: false,
                one_to_one: false,
                bounded: true,
                relation: Vec::new(),
            },
            pads: pads.iter().map(|b| b.id).collect(),
            rows,
            beats,
        })
    }

    pub fn reads_rows(&self) -> bool {
        self.filter.reads_rows()
    }

    /// The streams leaving, one per pad, as `init` answered them.
    pub fn streams(&self) -> &[CodedStream] {
        self.filter.streams()
    }
}

impl Node for PacketFilterNode {
    fn name(&self) -> &str {
        self.filter.name()
    }

    fn shape(&self) -> &NodeShape {
        &self.shape
    }

    fn settled_format(&self, port: usize) -> Option<StreamFormat> {
        let coded = self.filter.streams().get(port)?;
        Some(match coded.format {
            CodedFormat::Data => StreamFormat::Data(coded.codec.clone()),
            _ => StreamFormat::Packets(coded.clone()),
        })
    }

    fn host_progress(&self) -> bool {
        false
    }

    fn set_params(&mut self, params: &str) -> Result<()> {
        self.filter.set_params(params)
    }

    fn process(&mut self, mut tick: Tick) -> Result<Emitted> {
        let heard: Vec<Option<i64>> = self
            .pads
            .iter()
            .map(|id| tick.stream(*id).and_then(|s| s.progress))
            .collect();
        let rows: Vec<String> = self
            .rows
            .and_then(|id| tick.streams.iter_mut().find(|s| s.id == id))
            .map(|s| {
                std::mem::take(&mut s.messages)
                    .into_iter()
                    .map(|m| {
                        String::from_utf8(m.data)
                            .unwrap_or_else(|e| String::from_utf8_lossy(e.as_bytes()).into_owned())
                    })
                    .collect()
            })
            .unwrap_or_default();
        let pads = pads_of(&mut tick.streams, &self.pads);
        let mut items = Vec::new();
        if !tick.last && rows.is_empty() && pads.iter().all(Vec::is_empty) {
            // Nothing for the module: time moved on, and only the outputs of
            // the pads it moved on are told.
            for (pad, pts) in heard.into_iter().enumerate() {
                let Some(beats) = self.beats[pad].as_mut() else {
                    continue;
                };
                if let Some(pts) = pts.and_then(|pts| beats.pass(pts)) {
                    items.push(mark(pad, pts));
                }
            }
            return Ok(Emitted {
                items,
                rows: Vec::new(),
                finished: false,
            });
        }
        let filtered = self.filter.process(&pads, &rows, tick.last)?;
        for (pad, packets) in filtered.pads.into_iter().enumerate() {
            match self.beats[pad].as_mut() {
                Some(beats) => {
                    for packet in packets {
                        let pts = beats.place(packet.pts);
                        items.push(Emission {
                            port: pad,
                            payload: Payload::Message(Message {
                                pts,
                                data: packet.data,
                            }),
                        });
                    }
                    if !tick.last {
                        if let Some(pts) = heard[pad].and_then(|pts| beats.pass(pts)) {
                            items.push(mark(pad, pts));
                        }
                    }
                }
                None => items.extend(packets.into_iter().map(|p| Emission {
                    port: pad,
                    payload: Payload::Packet(p),
                })),
            }
        }
        let mut out_rows = filtered.rows;
        out_rows.extend(filtered.trailing);
        Ok(Emitted {
            items,
            rows: out_rows,
            finished: false,
        })
    }
}

/// A progress mark on output `port`.
fn mark(port: usize, pts: i64) -> Emission {
    Emission {
        port,
        payload: Payload::Message(Message {
            pts,
            data: Vec::new(),
        }),
    }
}

/// A data filter: messages at or before its clock, or as they arrive with
/// none; its outputs on their own timeline, beating while quiet.
pub struct DataFilterNode {
    filter: DataFilter,
    shape: NodeShape,
    /// Every pad's stream id and kind, in pad order.
    pads: Vec<(u32, PadKind)>,
    /// The first clock pad: its stream id and time base.
    clock: Option<(u32, TimeBase)>,
    beats: Vec<Beats>,
    /// How far each data pad has been read, messages and heartbeats both,
    /// in its own time base.
    read_to: Vec<Option<(i64, TimeBase)>>,
    /// Every pad's time base, in pad order.
    bases: Vec<TimeBase>,
}

impl DataFilterNode {
    /// Instantiates the filter and reads what it publishes, before its
    /// inputs have said what they carry: its outputs' headers go out first.
    pub fn load(path: &str) -> Result<DataFilterNode> {
        let filter = DataFilter::load(path)?;
        let described = filter.described();
        let beats = described
            .outputs
            .iter()
            .map(|_| {
                let mut beat = Beats::new(described.time_base);
                // The host's heartbeat at the start, already on the wire.
                beat.place(0);
                beat
            })
            .collect();
        let shape = data_filter_shape(described, &[]);
        Ok(DataFilterNode {
            filter,
            shape,
            pads: Vec::new(),
            clock: None,
            beats,
            read_to: Vec::new(),
            bases: Vec::new(),
        })
    }

    pub fn described(&self) -> &runtime::DescribedDataFilter {
        self.filter.described()
    }

    /// `init`, with every pad in order: a data stream is a data pad, any
    /// other stream a clock pad.
    pub fn init(&mut self, bound: &[BoundStream], params: &str) -> Result<()> {
        let mut pads = Vec::with_capacity(bound.len());
        for stream in bound {
            let kind = match stream.format {
                StreamFormat::Data(_) => PadKind::Data,
                _ => PadKind::Clock,
            };
            pads.push(DataPad {
                kind,
                codec: match &stream.format {
                    StreamFormat::Data(codec) => codec.clone(),
                    _ => String::new(),
                },
                time_base: stream.time_base,
            });
        }
        self.filter.init(&pads, params)?;
        self.pads = bound
            .iter()
            .zip(&pads)
            .map(|(stream, pad)| (stream.id, pad.kind))
            .collect();
        self.read_to = vec![None; bound.len()];
        self.bases = bound.iter().map(|b| b.time_base).collect();
        self.clock = bound
            .iter()
            .zip(&pads)
            .find(|(_, pad)| pad.kind == PadKind::Clock)
            .map(|(stream, _)| (stream.id, stream.time_base));
        let mut inputs = Vec::new();
        for (stream, pad) in bound.iter().zip(&pads) {
            let (kind, pairing) = match pad.kind {
                PadKind::Data => (PortKind::Data, Pairing::AtOrBefore),
                PadKind::Clock if Some(stream.id) == self.clock.map(|c| c.0) => {
                    (stream.format.kind(), Pairing::Lockstep)
                }
                PadKind::Clock => (stream.format.kind(), Pairing::Arrival),
            };
            let rows = if kind == PortKind::Data {
                RowsUse::PerFrame
            } else {
                RowsUse::Ignore
            };
            inputs.push(input(&stream.port, kind, true, false, pairing, rows));
        }
        self.shape.inputs = inputs;
        self.shape.clock = match self.clock {
            Some((id, _)) => Clock::Input(
                bound
                    .iter()
                    .find(|b| b.id == id)
                    .map(|b| b.port.clone())
                    .expect("the clock is bound"),
            ),
            None => Clock::SelfClocked,
        };
        Ok(())
    }
}

impl DataFilterNode {
    fn filter_base(&self, pad: usize) -> TimeBase {
        self.bases[pad]
    }
}

impl Node for DataFilterNode {
    fn name(&self) -> &str {
        self.filter.name()
    }

    fn shape(&self) -> &NodeShape {
        &self.shape
    }

    fn settled_format(&self, port: usize) -> Option<StreamFormat> {
        self.filter
            .described()
            .outputs
            .get(port)
            .map(|codec| StreamFormat::Data(codec.clone()))
    }

    fn host_progress(&self) -> bool {
        false
    }

    fn set_params(&mut self, _params: &str) -> Result<()> {
        bail!("{} takes no params between calls", self.filter.name())
    }

    /// One call: with a clock, `now` is the clock pad's frame, where the
    /// tick hands one; without, every message that arrived, and time as far
    /// as the data pads have been read.
    fn process(&mut self, tick: Tick) -> Result<Emitted> {
        let now = match self.clock {
            Some((id, _)) => tick
                .stream(id)
                .and_then(|s| s.frames.first())
                .map(|f| f.pts),
            None => None,
        };
        for (pad, (id, kind)) in self.pads.iter().enumerate() {
            let (Some(stream), PadKind::Data) = (tick.stream(*id), kind) else {
                continue;
            };
            let base = self.filter_base(pad);
            let newest = stream
                .messages
                .iter()
                .map(|m| m.pts)
                .chain(stream.progress)
                .max();
            if let Some(pts) = newest {
                let was = self.read_to[pad].map_or(pts, |(p, _)| p.max(pts));
                self.read_to[pad] = Some((was, base));
            }
        }
        let input: Vec<Vec<Message>> = self
            .pads
            .iter()
            .map(|(id, _)| {
                tick.stream(*id)
                    .map(|s| s.messages.clone())
                    .unwrap_or_default()
            })
            .collect();
        let mut items = Vec::new();
        let mut rows = Vec::new();
        let called = self.clock.is_some() || tick.last || input.iter().any(|m| !m.is_empty());
        if called {
            let processed = self.filter.process(&input, now, tick.last)?;
            let clocked = now
                .zip(self.clock.map(|(_, base)| base))
                .filter(|_| !tick.last);
            for (index, messages) in processed.outputs.into_iter().enumerate() {
                for message in messages {
                    let pts = self.beats[index].place(message.pts);
                    items.push(Emission {
                        port: index,
                        payload: Payload::Message(Message {
                            pts,
                            data: message.data,
                        }),
                    });
                }
                if let Some((now, base)) = clocked {
                    if let Some(pts) = self.beats[index].due(now, base) {
                        items.push(mark(index, pts));
                    }
                }
            }
            rows = processed.rows;
        }
        if self.clock.is_none() && !tick.last {
            let heard = self.read_to.iter().flatten().fold(
                None,
                |furthest: Option<(i64, TimeBase)>, &(pts, base)| match furthest {
                    Some((at, at_base)) if compare(pts, base, at, at_base).is_le() => furthest,
                    _ => Some((pts, base)),
                },
            );
            if let Some((heard, base)) = heard {
                for (index, beat) in self.beats.iter_mut().enumerate() {
                    if let Some(pts) = beat.due(heard, base) {
                        items.push(mark(index, pts));
                    }
                }
            }
        }
        Ok(Emitted {
            items,
            rows,
            finished: false,
        })
    }
}

/// A rows module: each tick's rows through `process`, `finish` the last.
pub struct RowsNode {
    module: RowsModule,
    shape: NodeShape,
}

impl RowsNode {
    pub fn open(path: &str, params: &str) -> Result<RowsNode> {
        let module = RowsModule::open(path, params)?;
        let shape = NodeShape {
            inputs: vec![input(
                "rows",
                PortKind::Data,
                true,
                false,
                Pairing::Lockstep,
                RowsUse::PerFrame,
            )],
            outputs: vec![rows_output("out", &module.meta().rows_schema)],
            clock: Clock::Input("rows".into()),
            pure: false,
            one_to_one: false,
            bounded: true,
            relation: Vec::new(),
        };
        Ok(RowsNode { module, shape })
    }
}

impl Node for RowsNode {
    fn name(&self) -> &str {
        self.module.name()
    }

    fn shape(&self) -> &NodeShape {
        &self.shape
    }

    fn host_progress(&self) -> bool {
        false
    }

    fn set_params(&mut self, _params: &str) -> Result<()> {
        bail!("{} takes no params between calls", self.module.name())
    }

    fn process(&mut self, tick: Tick) -> Result<Emitted> {
        let rows: Vec<String> = tick
            .streams
            .iter()
            .flat_map(|s| s.messages.iter())
            .map(|m| String::from_utf8_lossy(&m.data).into_owned())
            .collect();
        let mut out = Vec::new();
        if !tick.last || !rows.is_empty() {
            out.extend(
                self.module
                    .process(&rows)
                    .with_context(|| format!("{}: processing rows", self.module.name()))?,
            );
        }
        if tick.last {
            out.extend(
                self.module
                    .finish()
                    .with_context(|| format!("{}: finish", self.module.name()))?,
            );
        }
        Ok(Emitted {
            items: out
                .into_iter()
                .map(|row| Emission {
                    port: 0,
                    payload: Payload::Message(Message {
                        pts: tick.pts,
                        data: row.into_bytes(),
                    }),
                })
                .collect(),
            rows: Vec::new(),
            finished: false,
        })
    }
}

/// A packet source: one pull per call, an output per track it was told to
/// pull. Its first call pulls until every track's decode delay has settled,
/// since an output's header carries it and the source does not say it.
pub struct PacketSourceNode {
    source: PacketSource,
    catalog: Catalog,
    shape: NodeShape,
    delays: Vec<Option<u32>>,
    beats: Option<SourceBeats>,
    ended: bool,
}

impl PacketSourceNode {
    pub fn open(path: &str, params: &str, tracks: &[u32]) -> Result<PacketSourceNode> {
        let (source, catalog) = PacketSource::open(path, params, tracks)?;
        let shape = source_shape(&catalog);
        let delays = vec![None; catalog.tracks.len()];
        Ok(PacketSourceNode {
            source,
            catalog,
            shape,
            delays,
            beats: None,
            ended: false,
        })
    }

    pub fn catalog(&self) -> &Catalog {
        &self.catalog
    }

    fn pull(&mut self) -> Result<Option<Vec<Vec<Packet>>>> {
        let pads = self
            .source
            .next()
            .with_context(|| format!("{}: pulling packets", self.source.name()))?;
        Ok(pads.map(|pads| pads.into_iter().map(|pad| pad.packets).collect()))
    }

    fn emitted(&mut self, pads: Vec<Vec<Packet>>) -> Emitted {
        let beats = self.beats.as_mut().expect("set by the first call");
        let mut items = Vec::new();
        for (port, packets) in beats.pull(pads).into_iter().enumerate() {
            let data = self.catalog.tracks[port].stream.format == CodedFormat::Data;
            for packet in packets {
                items.push(Emission {
                    port,
                    payload: if data {
                        Payload::Message(Message {
                            pts: packet.pts,
                            data: if heartbeat::is_heartbeat(&packet.data) {
                                Vec::new()
                            } else {
                                packet.data
                            },
                        })
                    } else {
                        Payload::Packet(packet)
                    },
                });
            }
        }
        Emitted {
            items,
            rows: Vec::new(),
            finished: self.ended,
        }
    }
}

impl Node for PacketSourceNode {
    fn name(&self) -> &str {
        self.source.name()
    }

    fn decode_delay(&self, port: usize) -> Option<u32> {
        self.delays.get(port).copied().flatten()
    }

    fn shape(&self) -> &NodeShape {
        &self.shape
    }

    fn host_progress(&self) -> bool {
        false
    }

    fn set_params(&mut self, _params: &str) -> Result<()> {
        bail!("{} takes no params between calls", self.source.name())
    }

    fn process(&mut self, tick: Tick) -> Result<Emitted> {
        if tick.last || self.ended && self.beats.is_some() {
            return Ok(Emitted {
                finished: true,
                ..Emitted::default()
            });
        }
        if self.beats.is_none() {
            // The count of a track's leading packets with no dts is its
            // decode delay; a data track has none to count.
            for (slot, track) in self.catalog.tracks.iter().enumerate() {
                if track.stream.format == CodedFormat::Data {
                    self.delays[slot] = Some(0);
                }
            }
            let mut pending: Vec<Vec<Packet>> =
                self.catalog.tracks.iter().map(|_| Vec::new()).collect();
            while self.delays.iter().any(Option::is_none) {
                let Some(pads) = self.pull()? else {
                    self.ended = true;
                    break;
                };
                for (slot, packets) in pads.into_iter().enumerate() {
                    for packet in packets {
                        if self.delays[slot].is_none() && packet.dts.is_some() {
                            self.delays[slot] = Some(pending[slot].len() as u32);
                        }
                        pending[slot].push(packet);
                    }
                }
            }
            for (slot, delay) in self.delays.iter_mut().enumerate() {
                delay.get_or_insert(pending[slot].len() as u32);
            }
            self.beats = Some(SourceBeats::new(&self.catalog.tracks, &pending));
            return Ok(self.emitted(pending));
        }
        match self.pull()? {
            Some(pads) => Ok(self.emitted(pads)),
            None => {
                self.ended = true;
                Ok(Emitted {
                    finished: true,
                    ..Emitted::default()
                })
            }
        }
    }
}

/// The heartbeats a packet source's data tracks carry: one at start, at the
/// earliest time anything held back for the headers carries, and then those
/// the source hands on the track itself, at most one a tenth of a second.
///
/// The host claims no time of its own past the start. A heartbeat says no
/// message before its pts is still to come on the track, and only the source
/// can know that: tracks run independently, so a live source hands a message
/// whenever it reached it, and its media may be well past the message's pts
/// by then.
pub struct SourceBeats {
    /// Each track's time base, and its heartbeats where it is a data track.
    tracks: Vec<(TimeBase, Option<Beats>)>,
    /// The start heartbeat's time, until the first pull has carried it.
    start: Option<(i64, TimeBase)>,
}

impl SourceBeats {
    fn new(tracks: &[runtime::SourceTrack], held: &[Vec<Packet>]) -> SourceBeats {
        let tracks: Vec<(TimeBase, Option<Beats>)> = tracks
            .iter()
            .map(|t| {
                let base = t.stream.time_base;
                let data = t.stream.format == CodedFormat::Data;
                (base, data.then(|| Beats::new(base)))
            })
            .collect();
        let mut start: Option<(i64, TimeBase)> = None;
        for (packets, (base, _)) in held.iter().zip(&tracks) {
            if let Some(packet) = packets.first() {
                let at = (packet.dts.unwrap_or(packet.pts), *base);
                start = Some(match start {
                    Some((pts, b)) if compare(at.0, at.1, pts, b).is_ge() => (pts, b),
                    _ => at,
                });
            }
        }
        SourceBeats {
            tracks,
            start: Some(start.unwrap_or((0, heartbeat::EVERY))),
        }
    }

    /// One pull's packets, a list per track, with the data tracks' messages
    /// placed on their timeline and their heartbeats added: the start's, and
    /// the latest one the source handed on the track in this pull.
    fn pull(&mut self, mut pads: Vec<Vec<Packet>>) -> Vec<Vec<Packet>> {
        let start = self.start.take();
        for (packets, (base, beats)) in pads.iter_mut().zip(&mut self.tracks) {
            let Some(beats) = beats else { continue };
            let mut placed = Vec::with_capacity(packets.len() + 2);
            if let Some((pts, base)) = start {
                placed.extend(beats.due(pts, base).map(crate::heartbeat_packet));
            }
            let mut said = None;
            for mut packet in packets.drain(..) {
                if heartbeat::is_heartbeat(&packet.data) {
                    said = said.max(Some(packet.pts));
                    continue;
                }
                packet.pts = beats.place(packet.pts);
                packet.dts = Some(packet.pts);
                placed.push(packet);
            }
            if let Some(pts) = said {
                placed.extend(beats.due(pts, *base).map(crate::heartbeat_packet));
            }
            *packets = placed;
        }
        pads
    }
}
