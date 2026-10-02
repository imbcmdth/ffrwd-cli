//! A network that holds node modules: its chains read in the node spelling,
//! every node opened on the streams its pads bind, and the whole run on the
//! node lanes, its edges one NUT each.
//!
//! A node module's pads bind its ports by name. A frame module of an older
//! world takes its pads by position, as it does in any network, and runs as
//! the node its adapter makes of it. `rowfilter` and `rowmerge` read a data
//! edge. Every stream gets an id unique in the run, which is also the id its
//! readers are told at `init`.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, OnceLock};
use std::thread;

use anyhow::{anyhow, bail, Context, Result};
use ffrwd_wasm::nut;
use ffrwd_wasm_runtime::node::{
    BoundStream, Clock, Node, NodeShape, OutputFormat, PortKind, Rational, RowsUse, StreamFormat,
    TickFrame,
};
use ffrwd_wasm_runtime::runtime::{
    self, AudioFormat, Format, FormatOf, Media, Message, Packet, RenditionMeta, StreamInfo,
    TimeBase, VideoFormat, WitNode,
};

use crate::adapters::FilterNode;
use crate::edges::{self, Target};
use crate::graph::{self, NodeCall, NodePad, StreamClass, StreamRef, ROWS_PORT};
use crate::heartbeat;
use crate::host_nodes;
use crate::lanes::{Consumer, Intake, LaneSpec, Opener, Plan, PortOut, Scheduler};
use crate::network::{params_json, parse_schema, Binding};
use crate::tick::{Assembler, Item};
use crate::{Args, OutputKind};

/// Whether a network is one of nodes: any module it binds exports the node
/// world.
pub fn is_node_network(bindings: &[Binding]) -> Result<bool> {
    for binding in bindings {
        if runtime::exports_node(&binding.path)
            .with_context(|| format!("opening module {}", binding.path))?
        {
            return Ok(true);
        }
    }
    Ok(false)
}

/// One stream of the run: an input's, or a node output's.
#[derive(Clone)]
struct StreamDef {
    format: StreamFormat,
    base: TimeBase,
    info: StreamInfo,
    decode_delay: u32,
    /// How late a node's output may leave, which its readers are told.
    latency: Option<f64>,
    /// The header an input carried, kept for an output that writes the same
    /// stream on.
    header: Option<nut::Stream>,
    frame_rate: Option<(u64, u64)>,
    /// How the command line names it, for a refusal.
    spelling: String,
}

const KNOWN_PIX_FMTS: &[&str] = &["rgba", "yuv420p", "yuv422p", "yuv444p", "gray"];
const KNOWN_SAMPLE_FMTS: &[&str] = &["f32", "s16"];

fn intern(value: &str, known: &[&'static str], what: &str) -> Result<&'static str> {
    known
        .iter()
        .copied()
        .find(|k| *k == value)
        .ok_or_else(|| anyhow!("{what} {value}: only {} are carried", known.join(", ")))
}

fn base_of(tb: nut::TimeBase) -> TimeBase {
    TimeBase {
        num: tb.num,
        den: tb.den,
    }
}

fn nut_base(tb: TimeBase) -> nut::TimeBase {
    nut::TimeBase {
        num: tb.num,
        den: tb.den,
    }
}

/// What a stream of an input is to a node: its format, by its header.
fn input_stream(
    args: &Args,
    input: usize,
    index: usize,
    stream: &nut::Stream,
) -> Result<StreamDef> {
    let spelling = format!("stream {index} of input {input}");
    let base = base_of(stream.time_base);
    let format = if stream.is_json() {
        StreamFormat::Data(runtime::DATA_CODEC.to_string())
    } else if stream.pix_fmt().is_some() || stream.sample_fmt().is_some() {
        match crate::format_from_stream(stream)?.media {
            Media::Video(v) => StreamFormat::Video(v),
            Media::Audio(a) => StreamFormat::Audio(a),
        }
    } else if stream.codec_name().is_some() {
        let pad = crate::coded_pad(
            args,
            "this network",
            input,
            stream,
            index as u32,
            RenditionMeta::default(),
        )?;
        StreamFormat::Packets(pad.stream)
    } else {
        bail!(
            "{spelling} carries codec tag {}, which no node reads",
            stream.fourcc_name()
        );
    };
    let mut info = crate::stream_info_from(stream);
    info.index = index as u32;
    if let StreamFormat::Packets(coded) = &format {
        info.codec = coded.codec.clone();
    }
    if let StreamFormat::Data(codec) = &format {
        info.codec = codec.clone();
    }
    Ok(StreamDef {
        format,
        base,
        info,
        decode_delay: u32::try_from(stream.decode_delay).unwrap_or(u32::MAX),
        latency: None,
        header: Some(stream.clone()),
        frame_rate: stream.frame_rate,
        spelling,
    })
}

fn class_of(format: &StreamFormat) -> StreamClass {
    match format {
        StreamFormat::Video(_) => StreamClass::Video,
        StreamFormat::Audio(_) => StreamClass::Audio,
        StreamFormat::Data(_) => StreamClass::Data,
        StreamFormat::Packets(coded) => match coded.format.kind() {
            "audio" => StreamClass::Audio,
            _ => StreamClass::Video,
        },
    }
}

/// What a node is, by what its name binds.
enum Kind {
    Module { path: String },
    OldFilter { path: String },
    Host,
}

/// Everything a run needs once its network is opened.
struct Opened {
    plan: Plan,
    /// Per input, the stream id of each of its streams.
    input_ids: Vec<Vec<u32>>,
    streams: Vec<StreamDef>,
    writers: Vec<edges::Writer>,
}

pub fn run(args: &Args, bindings: &[Binding], wiring: &str) -> Result<()> {
    if args.annotations.input || args.annotations.output {
        bail!(
            "-annotations carries rows beside the frames of an older module's edge; a network \
             of node modules carries them as data streams of their own"
        );
    }
    if args.pads.iter().any(Option::is_some) {
        bail!("-pad follows a packet sink's -i, and a network of node modules hosts none");
    }
    if !args.rows_in.is_empty() {
        bail!(
            "-rows-in names a rows module's input; a network of node modules reads rows as a \
             data stream of an -i, [N:d]"
        );
    }
    if args.stream_info.is_some() {
        bail!(
            "-stream_info tells one module what its stream is; a network of node modules reads \
             each stream's own header"
        );
    }
    let calls = graph::parse_node_network(wiring)?;
    let inputs = edges::open_inputs(&args.inputs)?;
    let wake_slot: Arc<OnceLock<Arc<dyn Fn() + Send + Sync>>> = Arc::new(OnceLock::new());
    let wake: Arc<dyn Fn() + Send + Sync> = {
        let slot = Arc::clone(&wake_slot);
        Arc::new(move || {
            if let Some(wake) = slot.get() {
                wake();
            }
        })
    };
    let workers = crate::scheduler::worker_count(args.jobs);
    let headers: Vec<Vec<nut::Stream>> = inputs.iter().map(|i| i.streams.clone()).collect();
    let Opened {
        plan,
        input_ids,
        streams,
        mut writers,
    } = open(args, bindings, &calls, &headers, wake)?;
    let scheduler = Arc::new(Scheduler::start(plan, workers)?);
    let _ = wake_slot.set(scheduler.waker());

    let mut readers = Vec::with_capacity(inputs.len());
    for (index, input) in inputs.into_iter().enumerate() {
        let scheduler = Arc::clone(&scheduler);
        let ids = input_ids[index].clone();
        let defs: Vec<StreamDef> = ids.iter().map(|id| streams[*id as usize].clone()).collect();
        readers.push(thread::spawn(move || {
            let read = pump(input, &ids, &defs, &scheduler);
            match read {
                Ok(true) => {
                    for id in &ids {
                        if !scheduler.end(*id) {
                            break;
                        }
                    }
                }
                Ok(false) => {}
                Err(e) => scheduler.fail(e.context(format!("reading input {index}"))),
            }
        }));
    }
    let finished = scheduler.finish();
    drop(readers);
    finished?;
    for writer in &mut writers {
        writer.join();
    }
    for writer in &writers {
        if let Some(failure) = writer.queue.failure() {
            bail!("{failure}");
        }
    }
    Ok(())
}

/// Every frame of one input handed to the lanes reading it. True when the
/// input ended, false when the run stopped first.
fn pump(
    input: edges::Opened,
    ids: &[u32],
    defs: &[StreamDef],
    scheduler: &Scheduler,
) -> Result<bool> {
    let read: Vec<bool> = ids.iter().map(|id| scheduler.reads(*id)).collect();
    let mut counts = vec![0u64; ids.len()];
    let mut running = true;
    input.pump(|index, packet, payload| {
        let Some(&id) = ids.get(index) else {
            return Ok(true);
        };
        if !read[index] {
            return Ok(true);
        }
        let def = &defs[index];
        let number = counts[index];
        counts[index] += 1;
        running = match &def.format {
            StreamFormat::Video(_) | StreamFormat::Audio(_) => {
                let format = Format {
                    media: match &def.format {
                        StreamFormat::Video(v) => Media::Video(*v),
                        StreamFormat::Audio(a) => Media::Audio(*a),
                        _ => unreachable!("matched as frames"),
                    },
                    time_base: def.base,
                };
                crate::check_frame(&format, payload, number)
                    .with_context(|| def.spelling.clone())?;
                scheduler.arrive(
                    id,
                    Item::Frame(TickFrame {
                        pts: packet.pts,
                        duration: None,
                        data: Arc::new(payload.to_vec()),
                        rows: Vec::new(),
                    }),
                )
            }
            StreamFormat::Data(_) if heartbeat::is_heartbeat(payload) => {
                scheduler.progress(id, packet.pts)
            }
            StreamFormat::Data(_) => scheduler.arrive(
                id,
                Item::Message(Message {
                    pts: packet.pts,
                    data: payload.to_vec(),
                }),
            ),
            StreamFormat::Packets(_) => scheduler.arrive(
                id,
                Item::Packet(Packet {
                    pts: packet.pts,
                    dts: packet.dts,
                    duration: None,
                    keyframe: packet.keyframe,
                    data: payload.to_vec(),
                }),
            ),
        };
        Ok(running)
    })?;
    Ok(running)
}

fn open(
    args: &Args,
    bindings: &[Binding],
    calls: &[NodeCall],
    headers: &[Vec<nut::Stream>],
    wake: Arc<dyn Fn() + Send + Sync>,
) -> Result<Opened> {
    let mut streams: Vec<StreamDef> = Vec::new();
    let mut input_ids: Vec<Vec<u32>> = Vec::with_capacity(headers.len());
    let mut by_class: Vec<HashMap<StreamClass, Vec<u32>>> = Vec::with_capacity(headers.len());
    for (input, header) in headers.iter().enumerate() {
        let mut ids = Vec::with_capacity(header.len());
        let mut classes: HashMap<StreamClass, Vec<u32>> = HashMap::new();
        for (index, stream) in header.iter().enumerate() {
            let id = streams.len() as u32;
            match input_stream(args, input, index, stream) {
                Ok(def) => {
                    classes.entry(class_of(&def.format)).or_default().push(id);
                    streams.push(def);
                }
                Err(_) if stream.class() == nut::ANNOTATION_CLASS => {
                    streams.push(StreamDef {
                        format: StreamFormat::Data(String::new()),
                        base: base_of(stream.time_base),
                        info: StreamInfo::default(),
                        decode_delay: 0,
                        latency: None,
                        header: None,
                        frame_rate: None,
                        spelling: format!("the annotation stream of input {input}"),
                    });
                }
                Err(e) => return Err(e),
            }
            ids.push(id);
        }
        input_ids.push(ids);
        by_class.push(classes);
    }

    let paths: HashMap<&str, &str> = bindings
        .iter()
        .map(|b| (b.name.as_str(), b.path.as_str()))
        .collect();
    let mut labels: HashMap<&str, usize> = HashMap::new();
    for (index, call) in calls.iter().enumerate() {
        for (_, label) in &call.outputs {
            if labels.insert(label, index).is_some() {
                bail!("-filter_complex writes the label [{label}] twice");
            }
        }
    }
    let order = topological(calls, &labels)?;

    let mut label_ids: HashMap<String, u32> = HashMap::new();
    let mut consumers: HashMap<u32, Vec<Consumer>> = HashMap::new();
    let mut lanes: Vec<LaneSpec> = Vec::with_capacity(calls.len());
    for &index in &order {
        let call = &calls[index];
        let kind = match paths.get(call.module.as_str()) {
            Some(path) if runtime::exports_node(path)? => Kind::Module {
                path: path.to_string(),
            },
            Some(path)
                if runtime::exports_filter(path)? || runtime::exports_window_filter(path)? =>
            {
                Kind::OldFilter {
                    path: path.to_string(),
                }
            }
            Some(path) => bail!(
                "{}: {path} is a module of an older world that rides alone; a network of node \
                 modules hosts node modules, frame modules and the host's rowfilter and \
                 rowmerge",
                call.module
            ),
            None if host_nodes::is_host_node(&call.module) => Kind::Host,
            None => bail!("-filter_complex names '{}', which no -m binds", call.module),
        };

        let mut pads: Vec<(String, u32)> = Vec::with_capacity(call.inputs.len());
        for (position, (port, pad)) in call.inputs.iter().enumerate() {
            let id = match pad {
                NodePad::Input(r) => resolve_input(r, &by_class)?,
                NodePad::Label(label) => *label_ids.get(label.as_str()).ok_or_else(|| {
                    anyhow!("{} reads [{label}], which no chain writes", call.module)
                })?,
            };
            let port = match (&kind, port) {
                (Kind::Module { .. }, Some(port)) => port.clone(),
                (Kind::Module { .. }, None) => bail!(
                    "{}: the pad [{}] names no port; a node module's pads bind its ports by name, \
                     as [<port>={}]",
                    call.module,
                    spell(pad),
                    spell(pad)
                ),
                (_, Some(port)) => bail!(
                    "{}: the pad [{port}={}] names a port, and {} takes its pads by position",
                    call.module,
                    spell(pad),
                    call.module
                ),
                (Kind::Host, None) => "in".to_string(),
                (Kind::OldFilter { .. }, None) => format!("in{position}"),
            };
            pads.push((port, id));
        }

        let lane = match &kind {
            Kind::Module { path } => open_module(args, &call.module, path, call, &pads, &streams)?,
            Kind::OldFilter { path } => open_old_filter(&call.module, path, call, &pads, &streams)?,
            Kind::Host => open_host(call, &pads, &streams)?,
        };
        let Opening { mut spec, outputs } = lane;
        let lane_index = lanes.len();
        for (_, id) in &pads {
            consumers
                .entry(*id)
                .or_default()
                .push(Consumer::Lane(lane_index));
        }
        for (position, (port, label)) in call.outputs.iter().enumerate() {
            let name = match port {
                Some(name) => name.clone(),
                None => match spec.shape.outputs.get(position) {
                    Some(output) => output.name.clone(),
                    None => bail!(
                        "{} writes [{label}] as output {position}, and it has {} output(s)",
                        call.module,
                        spec.shape.outputs.len()
                    ),
                },
            };
            let id = streams.len() as u32;
            let def = if name == ROWS_PORT {
                let def = StreamDef {
                    format: StreamFormat::Data(runtime::DATA_CODEC.to_string()),
                    base: spec.tick_base,
                    info: info_for(&StreamFormat::Data(runtime::DATA_CODEC.into())),
                    decode_delay: 0,
                    latency: Some(0.0),
                    header: None,
                    frame_rate: None,
                    spelling: format!("[{label}], the rows of {}", call.module),
                };
                spec.rows = Some(PortOut {
                    stream: id,
                    base: def.base,
                    latency: 0.0,
                });
                def
            } else {
                let Some(port) = spec.shape.output_index(&name) else {
                    bail!(
                        "{} writes [{name}={label}], and it has no output '{name}'; it has {}",
                        call.module,
                        names(spec.shape.outputs.iter().map(|o| o.name.as_str()))
                    );
                };
                let Some(def) = outputs[port].clone() else {
                    bail!("{} output '{name}' has no format to write", call.module);
                };
                if spec.ports[port].is_some() {
                    bail!("{} output '{name}' is labelled twice", call.module);
                }
                spec.ports[port] = Some(PortOut {
                    stream: id,
                    base: def.base,
                    latency: def.latency.unwrap_or(0.0),
                });
                StreamDef {
                    spelling: format!("[{label}], output '{name}' of {}", call.module),
                    ..def
                }
            };
            streams.push(def);
            label_ids.insert(label.clone(), id);
        }
        lanes.push(spec);
    }

    let mut writers: Vec<edges::Writer> = Vec::new();
    for (w, output) in args.outputs.iter().enumerate() {
        let mapped: Vec<&String> = output.target.iter().chain(output.also.iter()).collect();
        if mapped.is_empty() {
            bail!(
                "the {} output has no -map naming which of the network's labels it writes",
                output.spelling
            );
        }
        let mut ids = Vec::with_capacity(mapped.len());
        for label in &mapped {
            let id = *label_ids.get(label.as_str()).ok_or_else(|| {
                anyhow!("-map [{label}] names a label the network does not write")
            })?;
            ids.push(id);
        }
        let target = match output.kind {
            OutputKind::Frames => Target::Nut(
                ids.iter()
                    .map(|id| header_for(&streams[*id as usize]))
                    .collect::<Result<Vec<_>>>()?,
            ),
            OutputKind::Rows | OutputKind::Subtitles(_) => {
                let [id] = ids.as_slice() else {
                    bail!(
                        "{} writes one data label, and {} are mapped to it",
                        output.spelling,
                        ids.len()
                    );
                };
                let def = &streams[*id as usize];
                if !matches!(def.format, StreamFormat::Data(_)) {
                    bail!(
                        "{} writes a data stream, and {} is {}",
                        output.spelling,
                        def.spelling,
                        def.format.kind().name()
                    );
                }
                match output.kind {
                    OutputKind::Subtitles(format) => Target::Subtitles(format),
                    _ => Target::Ndjson,
                }
            }
            OutputKind::Null => Target::Null,
        };
        for (position, id) in ids.iter().enumerate() {
            consumers
                .entry(*id)
                .or_default()
                .push(Consumer::Writer(w, position));
        }
        writers.push(edges::spawn_writer(
            output.path.clone(),
            output.spelling.clone(),
            target,
            Arc::clone(&wake),
        ));
    }

    let read: HashSet<u32> = consumers.keys().copied().collect();
    for spec in &mut lanes {
        for port in spec.ports.iter_mut() {
            if port.is_some_and(|p| !read.contains(&p.stream)) {
                *port = None;
            }
        }
        if spec.rows.is_some_and(|p| !read.contains(&p.stream)) {
            spec.rows = None;
        }
    }

    Ok(Opened {
        plan: Plan {
            lanes,
            consumers,
            writers: writers.iter().map(|w| Arc::clone(&w.queue)).collect(),
        },
        input_ids,
        streams,
        writers,
    })
}

fn names<'a>(names: impl Iterator<Item = &'a str>) -> String {
    let list: Vec<&str> = names.collect();
    if list.is_empty() {
        "none".to_string()
    } else {
        list.join(", ")
    }
}

fn spell(pad: &NodePad) -> String {
    match pad {
        NodePad::Input(r) => r.to_string(),
        NodePad::Label(label) => label.clone(),
    }
}

fn resolve_input(r: &StreamRef, by_class: &[HashMap<StreamClass, Vec<u32>>]) -> Result<u32> {
    let Some(classes) = by_class.get(r.input) else {
        bail!(
            "[{r}] reads input {}, and this command has {} -i",
            r.input,
            by_class.len()
        );
    };
    let of_class = classes.get(&r.class).map(Vec::as_slice).unwrap_or(&[]);
    of_class.get(r.nth).copied().ok_or_else(|| {
        anyhow!(
            "[{r}] reads {} stream {} of input {}, which carries {}",
            match r.class {
                StreamClass::Video => "video",
                StreamClass::Audio => "audio",
                StreamClass::Data => "data",
            },
            r.nth,
            r.input,
            of_class.len()
        )
    })
}

/// The chains in an order where every label is written before it is read.
fn topological(calls: &[NodeCall], labels: &HashMap<&str, usize>) -> Result<Vec<usize>> {
    let mut placed = vec![false; calls.len()];
    let mut order = Vec::with_capacity(calls.len());
    while order.len() < calls.len() {
        let before = order.len();
        for (index, call) in calls.iter().enumerate() {
            if placed[index] {
                continue;
            }
            let ready = call.inputs.iter().all(|(_, pad)| match pad {
                NodePad::Input(_) => true,
                NodePad::Label(label) => labels.get(label.as_str()).is_some_and(|c| placed[*c]),
            });
            if ready {
                placed[index] = true;
                order.push(index);
            }
        }
        if order.len() == before {
            let stuck = calls
                .iter()
                .enumerate()
                .find(|(i, _)| !placed[*i])
                .map(|(_, c)| c);
            let call = stuck.expect("something is unplaced");
            for (_, pad) in &call.inputs {
                if let NodePad::Label(label) = pad {
                    if !labels.contains_key(label.as_str()) {
                        bail!("{} reads [{label}], which no chain writes", call.module);
                    }
                }
            }
            bail!(
                "-filter_complex reads its own output round a loop through {}",
                call.module
            );
        }
    }
    Ok(order)
}

/// A lane, and per output port the stream it writes.
struct Opening {
    spec: LaneSpec,
    outputs: Vec<Option<StreamDef>>,
}

fn bound_stream(port: &str, id: u32, def: &StreamDef) -> BoundStream {
    BoundStream {
        port: port.to_string(),
        id,
        info: def.info.clone(),
        time_base: def.base,
        format: def.format.clone(),
        rendition: RenditionMeta::default(),
        row: None,
        decode_delay: def.decode_delay,
        latency: def.latency,
    }
}

fn info_for(format: &StreamFormat) -> StreamInfo {
    let (kind, codec) = match format {
        StreamFormat::Video(_) => ("video", "rawvideo".to_string()),
        StreamFormat::Audio(a) if a.sample_fmt == "s16" => ("audio", "pcm_s16le".to_string()),
        StreamFormat::Audio(_) => ("audio", "pcm_f32le".to_string()),
        StreamFormat::Data(codec) => ("data", codec.clone()),
        StreamFormat::Packets(coded) => (coded.format.kind(), coded.codec.clone()),
    };
    StreamInfo {
        index: 0,
        kind: kind.to_string(),
        codec,
        duration: None,
        tags: Vec::new(),
    }
}

/// A rate-of clock read off the first stream its port binds: the rate its
/// header states, or its time base's inverse.
fn resolve_rate(shape: &mut NodeShape, bound: &[BoundStream], defs: &[StreamDef]) -> Result<()> {
    let Clock::RateOf(port) = &shape.clock else {
        return Ok(());
    };
    let Some(first) = bound.iter().find(|b| &b.port == port) else {
        bail!("the clock is the rate of input '{port}', which this call leaves unbound");
    };
    let (num, den) = defs[first.id as usize]
        .frame_rate
        .unwrap_or((first.time_base.den, first.time_base.num));
    shape.clock = Clock::Rate(Rational {
        num: i32::try_from(num).unwrap_or(i32::MAX),
        den: i32::try_from(den).unwrap_or(1),
    });
    Ok(())
}

fn tick_base(shape: &NodeShape, bound: &[BoundStream]) -> Result<TimeBase> {
    Ok(match &shape.clock {
        Clock::Input(port) => bound
            .iter()
            .find(|b| &b.port == port)
            .map(|b| b.time_base)
            .ok_or_else(|| anyhow!("the clock input '{port}' is bound to no stream"))?,
        Clock::Rate(rate) => TimeBase {
            num: u64::try_from(rate.den).unwrap_or(1),
            den: u64::try_from(rate.num).unwrap_or(1),
        },
        Clock::RateOf(_) => unreachable!("resolved before the lane opens"),
        Clock::SelfClocked => TimeBase {
            num: 1,
            den: 1_000_000,
        },
    })
}

/// The stream each output port writes, as the shape, the bound streams and
/// the opened node settle it.
fn output_streams(
    name: &str,
    node: &dyn Node,
    shape: &NodeShape,
    bound: &[BoundStream],
    defs: &[StreamDef],
    tick: TimeBase,
) -> Result<Vec<Option<StreamDef>>> {
    let mut outputs = Vec::with_capacity(shape.outputs.len());
    for (port, output) in shape.outputs.iter().enumerate() {
        let like = match &output.format {
            Some(OutputFormat::Like(like)) => bound.iter().find(|b| b.port == like.port),
            _ => None,
        };
        let base = match output.time_base {
            Some(rate) => rate.time_base(&format!("{name} output '{}'", output.name))?,
            None => like.map_or(tick, |b| b.time_base),
        };
        let format = match node.settled_format(port) {
            Some(format) => format,
            None => match output.kind {
                PortKind::Video | PortKind::Audio => {
                    match runtime::output_format(shape, bound, port) {
                        Some(format) => frames_format(format)?,
                        None => {
                            outputs.push(None);
                            continue;
                        }
                    }
                }
                PortKind::Data => StreamFormat::Data(match &output.format {
                    Some(OutputFormat::Data(codec)) => codec.clone(),
                    _ => runtime::DATA_CODEC.to_string(),
                }),
                PortKind::Packets => match &output.format {
                    Some(OutputFormat::Packets(coded)) => StreamFormat::Packets(coded.clone()),
                    _ => {
                        outputs.push(None);
                        continue;
                    }
                },
            },
        };
        let source = like.or_else(|| match &shape.clock {
            Clock::Input(clock) => bound.iter().find(|b| &b.port == clock),
            _ => None,
        });
        let frame_rate = match &shape.clock {
            Clock::Rate(rate) if like.is_none() => Some((rate.num as u64, rate.den as u64)),
            _ => source.and_then(|b| defs[b.id as usize].frame_rate),
        };
        let header = source
            .and_then(|b| defs[b.id as usize].header.clone())
            .filter(|h| base_of(h.time_base) == base);
        outputs.push(Some(StreamDef {
            info: info_for(&format),
            decode_delay: node.decode_delay(port).unwrap_or(0),
            format,
            base,
            latency: Some(output.latency),
            header,
            frame_rate,
            spelling: String::new(),
        }));
    }
    Ok(outputs)
}

fn frames_format(format: FormatOf) -> Result<StreamFormat> {
    Ok(match format {
        FormatOf::Video {
            width,
            height,
            pix_fmt,
        } => {
            let pix_fmt = intern(&pix_fmt, KNOWN_PIX_FMTS, "pixel format")?;
            StreamFormat::Video(VideoFormat {
                width,
                height,
                pix_fmt,
                frame_len: crate::frame_len_for(pix_fmt, width, height)?,
                color: None,
            })
        }
        FormatOf::Audio {
            sample_rate,
            channels,
            sample_fmt,
        } => StreamFormat::Audio(AudioFormat {
            sample_rate,
            channels,
            sample_fmt: intern(&sample_fmt, KNOWN_SAMPLE_FMTS, "sample format")?,
            channel_layout: None,
        }),
    })
}

/// The stream ids whose rows a node keeps as state.
fn state_streams(shape: &NodeShape, bound: &[BoundStream]) -> Vec<u32> {
    bound
        .iter()
        .filter(|b| {
            shape
                .input(&b.port)
                .is_some_and(|p| p.rows == RowsUse::State)
        })
        .map(|b| b.id)
        .collect()
}

fn open_module(
    args: &Args,
    name: &str,
    path: &str,
    call: &NodeCall,
    pads: &[(String, u32)],
    defs: &[StreamDef],
) -> Result<Opening> {
    let params = match args.node_params.get(name) {
        Some(params) => params.clone(),
        None => {
            let meta = runtime::describe_node(path)
                .with_context(|| format!("describing module '{name}'"))?;
            let schema = parse_schema(&meta.params_schema, name)?;
            params_json(name, &schema, &call.options)?
        }
    };
    let mut wanted: Vec<String> = Vec::new();
    for (port, _) in pads {
        if !wanted.contains(port) {
            wanted.push(port.clone());
        }
    }
    let shape = runtime::node_shape(path, &params, &wanted)
        .with_context(|| format!("asking {name} for its shape"))?;
    let mut bound: Vec<BoundStream> = Vec::with_capacity(pads.len());
    for input in &shape.inputs {
        for (port, id) in pads.iter().filter(|(p, _)| *p == input.name) {
            bound.push(bound_stream(port, *id, &defs[*id as usize]));
        }
    }
    let latched: Vec<String> = call
        .outputs
        .iter()
        .filter_map(|(port, _)| port.clone())
        .filter(|port| port != ROWS_PORT)
        .collect();
    let node = WitNode::open(path, &params, bound.clone(), &latched)
        .with_context(|| format!("opening {name} from {path}"))?;
    let mut resolved = node.shape().clone();
    resolve_rate(&mut resolved, &bound, defs)?;
    let tick = tick_base(&resolved, &bound)?;
    let outputs = output_streams(name, &node, &resolved, &bound, defs, tick)?;
    let opener: Opener = {
        let (path, params, bound, latched) = (path.to_string(), params, bound.clone(), latched);
        Arc::new(move || {
            Ok(Box::new(WitNode::open(&path, &params, bound.clone(), &latched)?) as Box<dyn Node>)
        })
    };
    let assembler = Assembler::new(&resolved, &bound)?;
    Ok(Opening {
        spec: LaneSpec {
            name: name.to_string(),
            state: state_streams(&resolved, &bound),
            bound: bound.iter().map(|b| b.id).collect(),
            ports: vec![None; resolved.outputs.len()],
            shape: resolved,
            intake: Intake::Assembled(assembler),
            tick_base: tick,
            runners: vec![Box::new(node)],
            opener: Some(opener),
            rows: None,
        },
        outputs,
    })
}

fn open_old_filter(
    name: &str,
    path: &str,
    call: &NodeCall,
    pads: &[(String, u32)],
    defs: &[StreamDef],
) -> Result<Opening> {
    let Some((_, first)) = pads.first() else {
        bail!("{name} reads no stream; a frame module reads at least one");
    };
    let def = &defs[*first as usize];
    let media = match &def.format {
        StreamFormat::Video(v) => Media::Video(*v),
        StreamFormat::Audio(a) => Media::Audio(*a),
        other => bail!(
            "{name} reads decoded video or audio, and its first pad is {}",
            other.kind().name()
        ),
    };
    let format = Format {
        media,
        time_base: def.base,
    };
    let described =
        runtime::describe(path).with_context(|| format!("describing module '{name}'"))?;
    let schema = parse_schema(&described.meta.params_schema, name)?;
    let params = params_json(name, &schema, &call.options)?;
    let node = FilterNode::open(path, &format, &def.info, &params)
        .with_context(|| format!("opening module '{name}' from {path}"))?;
    let shape = node.shape().clone();
    if pads.len() != shape.inputs.len() {
        bail!(
            "{name} reads {} stream(s), and this chain wires {}",
            shape.inputs.len(),
            pads.len()
        );
    }
    let bound: Vec<BoundStream> = pads
        .iter()
        .map(|(port, id)| bound_stream(port, *id, &defs[*id as usize]))
        .collect();
    for stream in &bound {
        if stream.format != def.format {
            bail!("{name} reads its pads in one format, and they arrive in more than one");
        }
    }
    let tick = def.base;
    let outputs = output_streams(name, &node, &shape, &bound, defs, tick)?;
    let opener: Opener = {
        let (path, params, info) = (path.to_string(), params, def.info.clone());
        Arc::new(move || {
            Ok(Box::new(FilterNode::open(&path, &format, &info, &params)?) as Box<dyn Node>)
        })
    };
    let assembler = Assembler::new(&shape, &bound)?;
    Ok(Opening {
        spec: LaneSpec {
            name: name.to_string(),
            state: Vec::new(),
            bound: bound.iter().map(|b| b.id).collect(),
            ports: vec![None; shape.outputs.len()],
            shape,
            intake: Intake::Assembled(assembler),
            tick_base: tick,
            runners: vec![Box::new(node)],
            opener: Some(opener),
            rows: None,
        },
        outputs,
    })
}

fn open_host(call: &NodeCall, pads: &[(String, u32)], defs: &[StreamDef]) -> Result<Opening> {
    let [(_, id)] = pads else {
        bail!(
            "{} reads one data stream, and this chain wires {}",
            call.module,
            pads.len()
        );
    };
    let def = &defs[*id as usize];
    if !matches!(&def.format, StreamFormat::Data(codec) if codec == runtime::DATA_CODEC) {
        bail!(
            "{} in a network of node modules reads a data stream, and {} is {}",
            call.module,
            def.spelling,
            def.format.kind().name()
        );
    }
    let node = host_nodes::open(&call.module, &call.options, def.base)?;
    let shape = node.shape().clone();
    let latency = shape.outputs[0].latency;
    let output = StreamDef {
        format: StreamFormat::Data(runtime::DATA_CODEC.to_string()),
        base: def.base,
        info: info_for(&StreamFormat::Data(runtime::DATA_CODEC.into())),
        decode_delay: 0,
        latency: Some(def.latency.unwrap_or(0.0) + latency),
        header: None,
        frame_rate: None,
        spelling: String::new(),
    };
    Ok(Opening {
        spec: LaneSpec {
            name: call.module.clone(),
            state: Vec::new(),
            bound: vec![*id],
            ports: vec![None],
            shape,
            intake: Intake::Host { id: *id },
            tick_base: def.base,
            runners: vec![node],
            opener: None,
            rows: None,
        },
        outputs: vec![Some(output)],
    })
}

/// The NUT header an output writes a stream under.
fn header_for(def: &StreamDef) -> Result<nut::Stream> {
    let base = nut_base(def.base);
    let mut header = match &def.format {
        StreamFormat::Video(v) => match &def.header {
            Some(h)
                if h.pix_fmt() == Some(v.pix_fmt)
                    && h.video_geometry() == Some((v.width, v.height)) =>
            {
                h.clone()
            }
            _ => {
                let mut h =
                    nut::Stream::video(v.pix_fmt, v.width, v.height, base).ok_or_else(|| {
                        anyhow!(
                            "{} is {} video, which the wire does not carry; it carries {}",
                            def.spelling,
                            v.pix_fmt,
                            nut::supported_pix_fmts().join(", ")
                        )
                    })?;
                if let nut::Media::Video {
                    colorspace_type, ..
                } = &mut h.media
                {
                    *colorspace_type = crate::colorspace_type_for(v.color.as_ref());
                }
                h
            }
        },
        StreamFormat::Audio(a) => {
            let mut h =
                nut::Stream::audio(a.sample_fmt, a.sample_rate, a.channels).ok_or_else(|| {
                    anyhow!(
                        "{} is {} audio, which the wire does not carry",
                        def.spelling,
                        a.sample_fmt
                    )
                })?;
            h.time_base = base;
            h
        }
        StreamFormat::Data(_) => nut::Stream::json(base),
        StreamFormat::Packets(coded) => {
            crate::coded_stream_for(coded, u64::from(def.decode_delay))?
        }
    };
    header.time_base = base;
    header.frame_rate = match &def.format {
        StreamFormat::Video(_) => def.frame_rate,
        _ => header.frame_rate,
    };
    Ok(header)
}
