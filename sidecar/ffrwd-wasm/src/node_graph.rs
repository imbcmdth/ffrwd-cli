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
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::thread;

use anyhow::{anyhow, bail, Context, Result};
use ffrwd_wasm::nut;
use ffrwd_wasm_runtime::node::{
    Binding as PortBinding, BoundStream, Clock, Node, NodeShape, OutputFormat, Pairing, PortKind,
    Rational, RowsUse, StreamFormat, StreamHint, TickFrame,
};
use ffrwd_wasm_runtime::runtime::{
    self, AudioFormat, Format, FormatOf, Media, Message, Packet, RenditionMeta, StreamInfo,
    TimeBase, VideoFormat, Wants, WitNode,
};

use crate::adapters::FilterNode;
use crate::edges::{self, Target};
use crate::feeds::{self, FeedMember, FeedSpec};
use crate::graph::{self, NodeCall, NodePad, StreamClass, StreamRef, ROWS_PORT};
use crate::heartbeat;
use crate::host_nodes;
use crate::lanes::{Consumer, Intake, LaneSpec, Opener, Plan, PortOut, Scheduler};
use crate::leaky::{self, Leaky};
use crate::network::{params_json, parse_schema, Binding};
use crate::older;
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
    rendition: RenditionMeta,
    row: Option<u32>,
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
    let pad = args.pads.get(input).and_then(Option::as_ref);
    let rendition: RenditionMeta = pad.map(|p| p.rendition.clone().into()).unwrap_or_default();
    let format = if stream.is_json() {
        StreamFormat::Data(runtime::DATA_CODEC.to_string())
    } else if stream.pix_fmt().is_some() || stream.sample_fmt().is_some() {
        match crate::format_from_stream(stream)?.media {
            Media::Video(mut v) => {
                if let Some(c) = pad.and_then(|p| p.color.as_ref()) {
                    v.color = Some(runtime::ColorInfo {
                        range: crate::codec::parse_color_name("-pad color.range", &c.range)?,
                        primaries: crate::codec::parse_color_name(
                            "-pad color.primaries",
                            &c.primaries,
                        )?,
                        trc: crate::codec::parse_color_name("-pad color.trc", &c.trc)?,
                        space: crate::codec::parse_color_name("-pad color.space", &c.space)?,
                    });
                }
                StreamFormat::Video(v)
            }
            Media::Audio(a) => StreamFormat::Audio(a),
        }
    } else if stream.codec_name().is_some() {
        let coded = crate::coded_pad(
            args,
            "this network",
            input,
            stream,
            index as u32,
            rendition.clone(),
        )?;
        StreamFormat::Packets(coded.stream)
    } else {
        bail!(
            "{spelling} carries codec tag {}, which no node reads",
            stream.fourcc_name()
        );
    };
    let mut info = crate::stream_info_from(stream);
    info.index = index as u32;
    for (key, value) in pad.map(|p| &p.tags).into_iter().flatten() {
        match info.tags.iter_mut().find(|(k, _)| k == key) {
            Some(tag) => tag.1 = value.clone(),
            None => info.tags.push((key.clone(), value.clone())),
        }
    }
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
        rendition,
        row: pad.and_then(|p| p.row),
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
    Leaky,
}

/// Everything a run needs once its network is opened.
struct Opened {
    plan: Plan,
    /// Per input, the stream id of each of its streams.
    input_ids: Vec<Vec<u32>>,
    streams: Vec<StreamDef>,
    writers: Vec<edges::Writer>,
    /// The ports the host listens on for hold inputs given by one.
    feeds: Vec<FeedSpec>,
    /// The streams only inputs that want timing read: their bytes are never
    /// carried.
    timing: HashSet<u32>,
}

/// What a run carried of its inputs' frames: the bytes copied off the wire,
/// and the pictures and sounds a port feed conformed.
#[derive(Debug, Default)]
pub struct Carried {
    pub bytes: AtomicU64,
    pub conformed: AtomicU64,
}

pub fn run(args: &Args, bindings: &[Binding], wiring: &str) -> Result<()> {
    run_carrying(args, bindings, wiring).map(|_| ())
}

/// A run, and what it carried.
pub fn run_carrying(args: &Args, bindings: &[Binding], wiring: &str) -> Result<Arc<Carried>> {
    let carried = Arc::new(Carried::default());
    if args.annotations.input || args.annotations.output {
        bail!(
            "-annotations carries rows beside the frames of an older module's edge; a network \
             of node modules carries them as data streams of their own"
        );
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
    // The ports a feed is served on listen before any input is read: what
    // writes an input may itself wait for one of them to accept.
    let mut early = bind_feed_ports(args, bindings, &calls);
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
    let workers = crate::lanes::worker_count(args.jobs);
    let headers: Vec<Vec<nut::Stream>> = inputs.iter().map(|i| i.streams.clone()).collect();
    let ending = Arc::new(edges::Ending::default());
    let Opened {
        plan,
        input_ids,
        streams,
        mut writers,
        feeds,
        timing,
    } = open(args, bindings, &calls, &headers, wake, Arc::clone(&ending))?;
    let scheduler = Arc::new(Scheduler::start(plan, workers)?);
    let _ = wake_slot.set(scheduler.waker());
    ending.set_over(scheduler.over());

    let mut listeners = Vec::with_capacity(feeds.len());
    for spec in feeds {
        let scheduler = Arc::clone(&scheduler);
        let bound = early.remove(&spec.port);
        let carried = Arc::clone(&carried);
        listeners.push(thread::spawn(move || {
            if let Err(e) = feeds::serve(spec, bound, Arc::clone(&scheduler), &carried) {
                scheduler.fail(e);
            }
        }));
    }

    let mut readers = Vec::with_capacity(inputs.len());
    for (index, input) in inputs.into_iter().enumerate() {
        let scheduler = Arc::clone(&scheduler);
        let ids = input_ids[index].clone();
        let defs: Vec<StreamDef> = ids.iter().map(|id| streams[*id as usize].clone()).collect();
        let timed: Vec<bool> = ids.iter().map(|id| timing.contains(id)).collect();
        let carried = Arc::clone(&carried);
        readers.push(thread::spawn(move || {
            let read = pump(input, &ids, &defs, &timed, &scheduler, &carried);
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
    for listener in listeners {
        let _ = listener.join();
    }
    finished?;
    for writer in &mut writers {
        writer.join();
    }
    for writer in &writers {
        if let Some(failure) = writer.queue.failure() {
            bail!("{failure}");
        }
    }
    Ok(carried)
}

/// Every frame of one input handed to the lanes reading it. True when the
/// input ended, false when the run stopped first. A stream `timing` marks
/// is handed its frames' times alone: a picture with no bytes, a run of
/// samples with none and its length in samples as its duration.
fn pump(
    input: edges::Opened,
    ids: &[u32],
    defs: &[StreamDef],
    timing: &[bool],
    scheduler: &Scheduler,
    carried: &Carried,
) -> Result<bool> {
    let read: Vec<bool> = ids.iter().map(|id| scheduler.reads(*id)).collect();
    let mut counts = vec![0u64; ids.len()];
    let mut decoded: Vec<Option<i64>> = vec![None; ids.len()];
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
                let frame = match (&def.format, timing[index]) {
                    (StreamFormat::Audio(a), true) => TickFrame {
                        pts: packet.pts,
                        duration: Some((payload.len() / a.sample_len().max(1)) as i64),
                        data: Arc::new(Vec::new()),
                        rows: Vec::new(),
                    },
                    (_, true) => TickFrame {
                        pts: packet.pts,
                        duration: None,
                        data: Arc::new(Vec::new()),
                        rows: Vec::new(),
                    },
                    (_, false) => {
                        carried
                            .bytes
                            .fetch_add(payload.len() as u64, Ordering::Relaxed);
                        TickFrame {
                            pts: packet.pts,
                            duration: None,
                            data: Arc::new(payload.to_vec()),
                            rows: Vec::new(),
                        }
                    }
                };
                scheduler.arrive(id, Item::Frame(frame))
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
            StreamFormat::Packets(_) => {
                // A header that says nothing is reordered on a stream that is
                // (libx265's, written straight to NUT) makes every dts the
                // packet's pts; one that would go back is not known.
                let dts = packet
                    .dts
                    .filter(|dts| decoded[index].is_none_or(|last| *dts >= last));
                if dts.is_some() {
                    decoded[index] = dts;
                }
                scheduler.arrive(
                    id,
                    Item::Packet(Packet {
                        pts: packet.pts,
                        dts,
                        duration: None,
                        keyframe: packet.keyframe,
                        data: payload.to_vec(),
                    }),
                )
            }
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
    ending: Arc<edges::Ending>,
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
                        rendition: RenditionMeta::default(),
                        row: None,
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
    let given = args
        .node_bounds
        .iter()
        .map(|(name, lists)| ("-bound", name, lists.len()))
        .chain(
            args.node_params
                .iter()
                .map(|(name, files)| ("-params-from", name, files.len())),
        );
    for (flag, name, count) in given {
        let called = calls.iter().filter(|c| &c.module == name).count();
        if count > called {
            bail!(
                "{flag} {name}= is given {count} time(s), and -filter_complex calls '{name}' {called} time(s)"
            );
        }
    }

    let mut label_ids: HashMap<String, u32> = HashMap::new();
    let mut consumers: HashMap<u32, Vec<Consumer>> = HashMap::new();
    let mut mirrors: HashMap<u32, Vec<u32>> = HashMap::new();
    let mut lanes: Vec<LaneSpec> = Vec::with_capacity(calls.len());
    let mut feeds: Vec<FeedSpec> = Vec::new();
    let mut bytes_read: HashSet<u32> = HashSet::new();
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
                 modules hosts node modules, frame modules and the host's rowfilter, \
                 rowmerge and leaky",
                call.module
            ),
            None if host_nodes::is_host_node(&call.module) => Kind::Host,
            None if call.module == leaky::NODE => Kind::Leaky,
            None => bail!("-filter_complex names '{}', which no -m binds", call.module),
        };

        let mut pads: Vec<(String, u32)> = Vec::with_capacity(call.inputs.len());
        for (position, (port, pad)) in call.inputs.iter().enumerate() {
            let mut id = match pad {
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
                (Kind::OldFilter { .. } | Kind::Leaky, None) => format!("in{position}"),
            };
            if matches!(kind, Kind::Module { .. }) && pads.iter().any(|(_, bound)| *bound == id) {
                let mirror = streams.len() as u32;
                streams.push(streams[id as usize].clone());
                mirrors.entry(id).or_default().push(mirror);
                id = mirror;
            }
            pads.push((port, id));
        }

        let lane = match &kind {
            Kind::Module { path } => {
                open_module(args, &call.module, path, calls, index, &pads, &mut streams)?
            }
            Kind::OldFilter { path } => open_old_filter(&call.module, path, call, &pads, &streams)?,
            Kind::Host => open_host(call, &pads, &streams)?,
            Kind::Leaky => open_leaky(call, &pads, &streams)?,
        };
        let Opening {
            mut spec,
            outputs,
            feeds: listened,
        } = lane;
        let lane_index = lanes.len();
        for (_, id) in &pads {
            consumers
                .entry(*id)
                .or_default()
                .push(Consumer::Lane(lane_index));
        }
        for spec in &listened {
            for member in &spec.members {
                consumers
                    .entry(member.id)
                    .or_default()
                    .push(Consumer::Lane(lane_index));
            }
        }
        feeds.extend(listened);
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
                    rendition: RenditionMeta::default(),
                    row: None,
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
                    rendition: RenditionMeta::default(),
                    row: None,
                    ..def
                }
            };
            streams.push(def);
            label_ids.insert(label.clone(), id);
        }
        for (port, id) in &pads {
            let timing = matches!(kind, Kind::Module { .. })
                && spec
                    .shape
                    .input(port)
                    .is_some_and(|p| p.accepts.wants == Wants::Timing);
            if !timing {
                bytes_read.insert(*id);
            }
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
                    _ if host_timed(&lanes, *id) => Target::Ndjson(None),
                    _ => Target::Ndjson(Some(def.base)),
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
            Arc::clone(&ending),
        ));
    }

    for (id, readers) in &consumers {
        if readers.iter().any(|c| matches!(c, Consumer::Writer(..))) {
            bytes_read.insert(*id);
        }
    }
    for (original, copies) in &mirrors {
        if copies.iter().any(|copy| bytes_read.contains(copy)) {
            bytes_read.insert(*original);
        }
    }
    let timing: HashSet<u32> = input_ids
        .iter()
        .flatten()
        .copied()
        .filter(|id| !bytes_read.contains(id))
        .collect();
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
            mirrors,
        },
        input_ids,
        streams,
        writers,
        feeds,
        timing,
    })
}

/// Whether stream `id` is written by a node that ticks as things arrive,
/// whose times are the host's own clock and not the media's.
fn host_timed(lanes: &[LaneSpec], id: u32) -> bool {
    lanes.iter().any(|lane| {
        matches!(lane.shape.clock, Clock::SelfClocked)
            && !lane.bound.is_empty()
            && lane
                .ports
                .iter()
                .flatten()
                .chain(lane.rows.iter())
                .any(|port| port.stream == id)
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

/// A lane, per output port the stream it writes, and the ports its hold
/// inputs are served from.
struct Opening {
    spec: LaneSpec,
    outputs: Vec<Option<StreamDef>>,
    feeds: Vec<FeedSpec>,
}

fn bound_stream(port: &str, id: u32, def: &StreamDef, hint: StreamHint) -> BoundStream {
    BoundStream {
        port: port.to_string(),
        id,
        info: def.info.clone(),
        time_base: def.base,
        format: def.format.clone(),
        rendition: def.rendition.clone(),
        row: def.row,
        decode_delay: def.decode_delay,
        latency: def.latency,
        hint,
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
    let clock = match &shape.clock {
        Clock::Input(clock) => bound.iter().find(|b| &b.port == clock),
        _ => None,
    };
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
                PortKind::Packets => match (&output.format, clock) {
                    (Some(OutputFormat::Packets(coded)), _) => StreamFormat::Packets(coded.clone()),
                    (None, Some(clock)) if matches!(clock.format, StreamFormat::Packets(_)) => {
                        clock.format.clone()
                    }
                    _ => {
                        outputs.push(None);
                        continue;
                    }
                },
            },
        };
        let source = like.or(clock);
        let delay = match (&output.format, clock) {
            (None, Some(clock)) if matches!(format, StreamFormat::Packets(_)) => clock.decode_delay,
            _ => 0,
        };
        let frame_rate = match &shape.clock {
            Clock::Rate(rate) if like.is_none() => Some((rate.num as u64, rate.den as u64)),
            _ => source.and_then(|b| defs[b.id as usize].frame_rate),
        };
        let header = source
            .and_then(|b| defs[b.id as usize].header.clone())
            .filter(|h| base_of(h.time_base) == base);
        outputs.push(Some(StreamDef {
            info: info_for(&format),
            decode_delay: node.decode_delay(port).unwrap_or(delay),
            format,
            base,
            latency: Some(output.latency),
            header,
            frame_rate,
            spelling: String::new(),
            rendition: RenditionMeta::default(),
            row: None,
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

/// The ports the node calls' unbound hold inputs are served on, each bound
/// now. A port that will not bind, or a call whose shape cannot be had, is
/// left for opening the network to say so.
fn bind_feed_ports(
    args: &Args,
    bindings: &[Binding],
    calls: &[NodeCall],
) -> HashMap<u16, std::net::TcpListener> {
    let mut bound = HashMap::new();
    for (index, call) in calls.iter().enumerate() {
        let Some(binding) = bindings.iter().find(|b| b.name == call.module) else {
            continue;
        };
        if !runtime::exports_node(&binding.path).unwrap_or(false) {
            continue;
        }
        let Ok(meta) = runtime::describe_node(&binding.path) else {
            continue;
        };
        let Ok(schema) = parse_schema(&meta.params_schema, &call.module) else {
            continue;
        };
        let params = match given_params(args, calls, index) {
            Some(params) => params,
            None => match params_json(&call.module, &schema, &call.options) {
                Ok(params) => params,
                Err(_) => continue,
            },
        };
        let Ok(told) = bindings_of(args, calls, index) else {
            continue;
        };
        let wanted = padded_ports(call);
        let Ok(shape) = shape_of(&binding.path, &params, &told) else {
            continue;
        };
        for input in &shape.inputs {
            let Pairing::Hold(hold) = &input.pairing else {
                continue;
            };
            let Some(param) = &hold.port_param else {
                continue;
            };
            if wanted.contains(&input.name) {
                continue;
            }
            let Ok(ports) = port_params(&params, &schema, param, &call.module, &input.name) else {
                continue;
            };
            for port in ports {
                if let std::collections::hash_map::Entry::Vacant(slot) = bound.entry(port) {
                    if let Ok(listener) = feeds::bind(port) {
                        slot.insert(listener);
                    }
                }
            }
        }
    }
    bound
}

/// The ports a call's pads bind, in the order they first name each.
fn padded_ports(call: &NodeCall) -> Vec<String> {
    let mut ports: Vec<String> = Vec::new();
    for (port, _) in &call.inputs {
        if let Some(port) = port {
            if !ports.contains(port) {
                ports.push(port.clone());
            }
        }
    }
    ports
}

/// The inputs call `index` binds, as its shape is asked for them: the
/// `-bound` list the compiler asked with, for the k-th chain calling a name
/// the k-th given for it, checked against the pads; with none given, the
/// ports the pads name, a stream per pad with nothing known of it. An
/// input the list names and no pad binds is one a port serves.
fn bindings_of(args: &Args, calls: &[NodeCall], index: usize) -> Result<Vec<PortBinding>> {
    let call = &calls[index];
    let mut padded: Vec<PortBinding> = Vec::new();
    for (port, _) in &call.inputs {
        let Some(port) = port else { continue };
        match padded.iter_mut().find(|b| &b.input == port) {
            Some(binding) => binding.streams.push(StreamHint::default()),
            None => padded.push(PortBinding {
                input: port.clone(),
                streams: vec![StreamHint::default()],
            }),
        }
    }
    let Some(lists) = args.node_bounds.get(&call.module) else {
        return Ok(padded);
    };
    let Some(list) = lists.get(turn(calls, index)) else {
        bail!(
            "{}: -bound {}= is given {} time(s), and -filter_complex calls it more often; a \
             call takes the -bound given in its turn",
            call.module,
            call.module,
            lists.len()
        );
    };
    for wanted in &padded {
        match list.iter().find(|b| b.input == wanted.input) {
            None => bail!(
                "{}: -bound leaves out input '{}', which a pad binds",
                call.module,
                wanted.input
            ),
            Some(b) if b.streams.len() != wanted.streams.len() => bail!(
                "{}: -bound gives input '{}' {} stream(s), and the pads bind {}",
                call.module,
                wanted.input,
                b.streams.len(),
                wanted.streams.len()
            ),
            Some(_) => {}
        }
    }
    Ok(list.clone())
}

/// Which call of its name call `index` is: 0 for the first chain calling
/// it.
fn turn(calls: &[NodeCall], index: usize) -> usize {
    calls[..index]
        .iter()
        .filter(|c| c.module == calls[index].module)
        .count()
}

/// The params `-params-from` gives call `index`: the k-th given for its
/// name, for the k-th chain calling it.
fn given_params(args: &Args, calls: &[NodeCall], index: usize) -> Option<String> {
    args.node_params
        .get(&calls[index].module)
        .and_then(|given| given.get(turn(calls, index)))
        .cloned()
}

/// The hint the list gives stream `k` of `port`.
fn hint_of(told: &[PortBinding], port: &str, k: usize) -> StreamHint {
    told.iter()
        .find(|b| b.input == port)
        .and_then(|b| b.streams.get(k))
        .copied()
        .unwrap_or_default()
}

/// Inputs a `-bound` list names and no pad binds: each has to be one a port
/// serves, a hold input with a port param or a data input on a hold group.
fn check_port_served(
    name: &str,
    shape: &NodeShape,
    told: &[PortBinding],
    padded: &[String],
) -> Result<()> {
    for binding in told.iter().filter(|b| !padded.contains(&b.input)) {
        let served = shape
            .input(&binding.input)
            .is_some_and(|p| match &p.pairing {
                Pairing::Hold(hold) => hold.port_param.is_some(),
                Pairing::Interval(interval) => interval.group.is_some(),
                _ => false,
            });
        if !served {
            bail!(
                "{name}: -bound names input '{}', which no pad binds and no port serves",
                binding.input
            );
        }
    }
    Ok(())
}

/// A node's shape for these params and bound inputs, asked once a run.
fn shape_of(path: &str, params: &str, wanted: &[PortBinding]) -> Result<NodeShape> {
    type Key = (String, String, Vec<PortBinding>);
    static SHAPES: OnceLock<std::sync::Mutex<HashMap<Key, NodeShape>>> = OnceLock::new();
    let shapes = SHAPES.get_or_init(Default::default);
    let key = (path.to_string(), params.to_string(), wanted.to_vec());
    if let Some(shape) = shapes.lock().unwrap_or_else(|e| e.into_inner()).get(&key) {
        return Ok(shape.clone());
    }
    let shape = runtime::node_shape(path, params, wanted)?;
    shapes
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .insert(key, shape.clone());
    Ok(shape)
}

fn open_module(
    args: &Args,
    name: &str,
    path: &str,
    calls: &[NodeCall],
    index: usize,
    pads: &[(String, u32)],
    defs: &mut Vec<StreamDef>,
) -> Result<Opening> {
    let call = &calls[index];
    let meta =
        runtime::describe_node(path).with_context(|| format!("describing module '{name}'"))?;
    let schema = parse_schema(&meta.params_schema, name)?;
    let mut params = match given_params(args, calls, index) {
        Some(params) => params,
        None => params_json(name, &schema, &call.options)?,
    };
    let told = bindings_of(args, calls, index)?;
    let wanted = padded_ports(call);
    let shape =
        shape_of(path, &params, &told).with_context(|| format!("asking {name} for its shape"))?;
    check_port_served(name, &shape, &told, &wanted)?;
    let mut bound: Vec<BoundStream> = Vec::with_capacity(pads.len());
    for input in &shape.inputs {
        for (k, (port, id)) in pads.iter().filter(|(p, _)| *p == input.name).enumerate() {
            bound.push(bound_stream(
                port,
                *id,
                &defs[*id as usize],
                hint_of(&told, port, k),
            ));
        }
    }
    let feeds = listen_for(
        name,
        &shape,
        &wanted,
        &told,
        &mut bound,
        &mut params,
        &schema,
        defs,
    )?;
    let port_fed: Vec<u32> = feeds
        .iter()
        .flat_map(|f| f.members.iter().map(|m| m.id))
        .collect();
    let latched: Vec<String> = call
        .outputs
        .iter()
        .filter_map(|(port, _)| port.clone())
        .filter(|port| port != ROWS_PORT)
        .collect();
    let handed = restamped_bound(&shape, &bound, defs)?;
    let node = WitNode::open(path, &params, handed.clone(), &told, &latched)
        .with_context(|| format!("opening {name} from {path}"))?;
    let mut resolved = node.shape().clone();
    resolve_rate(&mut resolved, &bound, defs)?;
    let tick = tick_base(&resolved, &bound)?;
    let outputs = output_streams(name, &node, &resolved, &bound, defs, tick)?;
    let opener: Opener = {
        let (path, params, bound, latched) = (path.to_string(), params, handed, latched);
        let told = told.clone();
        Arc::new(move || {
            Ok(Box::new(WitNode::open(
                &path,
                &params,
                bound.clone(),
                &told,
                &latched,
            )?) as Box<dyn Node>)
        })
    };
    let assembler = Assembler::new(&resolved, &bound, name, &port_fed)?;
    Ok(Opening {
        spec: LaneSpec {
            name: name.to_string(),
            state: state_streams(&resolved, &bound),
            bound: bound.iter().map(|b| b.id).collect(),
            ports: vec![None; resolved.outputs.len()],
            shape: resolved,
            intake: Intake::Assembled(Box::new(assembler)),
            tick_base: tick,
            runners: vec![Box::new(node)],
            opener: Some(opener),
            rows: None,
        },
        outputs,
        feeds,
    })
}

/// The streams as the node is told them at `init`: one re-stamped onto the
/// clock (`crate::tick::restamped`) in the clock's time base.
fn restamped_bound(
    shape: &NodeShape,
    bound: &[BoundStream],
    defs: &[StreamDef],
) -> Result<Vec<BoundStream>> {
    let restamps = |b: &BoundStream| {
        shape.input(&b.port).is_some_and(|p| {
            crate::tick::restamped(p, b)
                || matches!(&p.pairing, Pairing::Interval(i) if i.group.is_some())
        })
    };
    if !bound.iter().any(restamps) {
        return Ok(bound.to_vec());
    }
    let mut planned = shape.clone();
    resolve_rate(&mut planned, bound, defs)?;
    let clock = tick_base(&planned, bound)?;
    Ok(bound
        .iter()
        .map(|b| {
            let mut told = b.clone();
            if restamps(b) {
                told.time_base = clock;
            }
            told
        })
        .collect())
}

/// The ports `name`'s hold inputs are served from: one listener per group
/// of hold inputs that name a port param and the call left unbound, each
/// bound a stream of the host's own in the port's format. A hold input
/// bound to a stream gets the port the host picked written into its param.
#[allow(clippy::too_many_arguments)]
fn listen_for(
    name: &str,
    shape: &NodeShape,
    wanted: &[String],
    told: &[PortBinding],
    bound: &mut Vec<BoundStream>,
    params: &mut String,
    schema: &serde_json::Value,
    defs: &mut Vec<StreamDef>,
) -> Result<Vec<FeedSpec>> {
    let mut feeds: Vec<FeedSpec> = Vec::new();
    let mut groups: Vec<(Option<String>, usize)> = Vec::new();
    for input in &shape.inputs {
        let Pairing::Hold(hold) = &input.pairing else {
            continue;
        };
        let Some(param) = &hold.port_param else {
            continue;
        };
        if wanted.contains(&input.name) {
            let streams = bound.iter().filter(|b| b.port == input.name).count().max(1);
            let ports: Vec<u16> = (0..streams).map(|_| free_port()).collect::<Result<_>>()?;
            *params = with_ports(params, schema, param, &ports)?;
            continue;
        }
        let ports = port_params(params, schema, param, name, &input.name)?;
        if ports.len() != 1 && !input.many {
            bail!(
                "{name} input '{}' takes one stream, and its param '{param}' names {} ports",
                input.name,
                ports.len()
            );
        }
        for port in ports {
            listen_on(
                name,
                shape,
                input,
                hold,
                port,
                told,
                bound,
                defs,
                &mut feeds,
                &mut groups,
            )?;
        }
    }
    for input in &shape.inputs {
        let Pairing::Interval(interval) = &input.pairing else {
            continue;
        };
        let Some(group) = &interval.group else {
            continue;
        };
        let Some(&(_, at)) = groups
            .iter()
            .find(|(g, _)| g.as_deref() == Some(group.as_str()))
        else {
            continue;
        };
        if wanted.contains(&input.name) {
            bail!(
                "{name} input '{}' arrives on the connection of group '{group}', which a port \
                 serves, and this call binds it a stream",
                input.name
            );
        }
        let format = StreamFormat::Data(runtime::DATA_CODEC.to_string());
        let id = defs.len() as u32;
        let def = StreamDef {
            info: info_for(&format),
            format,
            base: clock_base(shape, bound),
            decode_delay: 0,
            latency: None,
            header: None,
            frame_rate: None,
            spelling: format!(
                "the data on 127.0.0.1:{} for input '{}' of {name}",
                feeds[at].port, input.name
            ),
            rendition: RenditionMeta::default(),
            row: None,
        };
        bound.push(bound_stream(
            &input.name,
            id,
            &def,
            hint_of(told, &input.name, 0),
        ));
        feeds[at].members.push(FeedMember {
            id,
            name: input.name.clone(),
            format: def.format.clone(),
            timing: false,
        });
        defs.push(def);
    }
    for feed in &mut feeds {
        feed.members
            .sort_by_key(|m| !matches!(m.format, StreamFormat::Video(_)));
    }
    Ok(feeds)
}

/// The clock's time base as far as the shape and the streams bound so far
/// say it: a rate clock's, an input clock's stream's, or microseconds.
fn clock_base(shape: &NodeShape, bound: &[BoundStream]) -> TimeBase {
    match &shape.clock {
        Clock::Input(clock) => bound.iter().find(|b| &b.port == clock).map_or(
            TimeBase {
                num: 1,
                den: 1_000_000,
            },
            |b| b.time_base,
        ),
        Clock::Rate(rate) => TimeBase {
            num: u64::try_from(rate.den).unwrap_or(1),
            den: u64::try_from(rate.num).unwrap_or(1),
        },
        _ => TimeBase {
            num: 1,
            den: 1_000_000,
        },
    }
}

/// One port a hold input is served on: a stream of the host's own bound to
/// the input, and the listener it arrives on, shared with its group's other
/// members on that port.
#[allow(clippy::too_many_arguments)]
fn listen_on(
    name: &str,
    shape: &NodeShape,
    input: &ffrwd_wasm_runtime::node::InputPort,
    hold: &ffrwd_wasm_runtime::node::Hold,
    port: u16,
    told: &[PortBinding],
    bound: &mut Vec<BoundStream>,
    defs: &mut Vec<StreamDef>,
    feeds: &mut Vec<FeedSpec>,
    groups: &mut Vec<(Option<String>, usize)>,
) -> Result<()> {
    let format = conformed(name, shape, input, bound)?;
    let id = defs.len() as u32;
    let micros = TimeBase {
        num: 1,
        den: 1_000_000,
    };
    let base = match &shape.clock {
        Clock::Input(clock) => bound
            .iter()
            .find(|b| &b.port == clock)
            .map_or(micros, |b| b.time_base),
        _ => micros,
    };
    let def = StreamDef {
        info: info_for(&format),
        format,
        base,
        decode_delay: 0,
        latency: None,
        header: None,
        frame_rate: None,
        spelling: format!(
            "the feed on 127.0.0.1:{port} for input '{}' of {name}",
            input.name
        ),
        rendition: RenditionMeta::default(),
        row: None,
    };
    let k = bound.iter().filter(|b| b.port == input.name).count();
    bound.push(bound_stream(
        &input.name,
        id,
        &def,
        hint_of(told, &input.name, k),
    ));
    let member = FeedMember {
        id,
        name: input.name.clone(),
        format: def.format.clone(),
        timing: input.accepts.wants == Wants::Timing,
    };
    defs.push(def);
    let mine: Vec<usize> = match &hold.group {
        Some(g) => groups
            .iter()
            .filter(|(k, _)| k.as_ref() == Some(g))
            .map(|(_, index)| *index)
            .collect(),
        None => Vec::new(),
    };
    match mine.iter().find(|index| feeds[**index].port == port) {
        Some(index) => feeds[*index].members.push(member),
        None if !mine.is_empty() && !input.many => bail!(
            "{name} input '{}' is on port {port} and its group's picture on port {}; a \
             group arrives on one connection",
            input.name,
            feeds[mine[0]].port
        ),
        None => {
            groups.push((hold.group.clone(), feeds.len()));
            feeds.push(FeedSpec {
                node: name.to_string(),
                port,
                members: vec![member],
            });
        }
    }
    Ok(())
}

/// A free loopback port, for the param of a hold input bound to a stream.
fn free_port() -> Result<u16> {
    let listener = std::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0))
        .context("picking a loopback port")?;
    Ok(listener.local_addr()?.port())
}

fn params_object(params: &str) -> Result<serde_json::Map<String, serde_json::Value>> {
    if params.trim().is_empty() {
        return Ok(serde_json::Map::new());
    }
    match serde_json::from_str(params)? {
        serde_json::Value::Object(map) => Ok(map),
        _ => bail!("the params are not a JSON object"),
    }
}

/// `params` with the ports the host picked written into `key`: a list
/// where its schema takes one, the first port where it takes a number.
fn with_ports(
    params: &str,
    schema: &serde_json::Value,
    key: &str,
    ports: &[u16],
) -> Result<String> {
    let mut map = params_object(params)?;
    let types = schema
        .get("properties")
        .and_then(|p| p.get(key))
        .and_then(|p| p.get("type"));
    let list = match types {
        Some(serde_json::Value::Array(types)) => types.iter().any(|t| t == "array"),
        Some(t) => t == "array",
        None => false,
    };
    let value = if list {
        serde_json::Value::from(ports.to_vec())
    } else {
        serde_json::Value::from(ports[0])
    };
    map.insert(key.to_string(), value);
    Ok(serde_json::Value::Object(map).to_string())
}

/// The ports a hold input's param names, what the call wrote or the
/// param's default: one, or one per entry of a list.
fn port_params(
    params: &str,
    schema: &serde_json::Value,
    key: &str,
    name: &str,
    input: &str,
) -> Result<Vec<u16>> {
    let written = params_object(params)?.get(key).cloned();
    if let Some(serde_json::Value::Array(entries)) = &written {
        return entries
            .iter()
            .map(|entry| {
                let single = serde_json::json!({ key: entry }).to_string();
                port_param(&single, schema, key, name, input)
            })
            .collect();
    }
    Ok(vec![port_param(params, schema, key, name, input)?])
}

/// The port a hold input's param names: what the call wrote, or the
/// param's default.
fn port_param(
    params: &str,
    schema: &serde_json::Value,
    key: &str,
    name: &str,
    input: &str,
) -> Result<u16> {
    let written = params_object(params)?.get(key).cloned();
    let value = written.or_else(|| {
        schema
            .get("properties")
            .and_then(|p| p.get(key))
            .and_then(|p| p.get("default"))
            .cloned()
    });
    let number = value.as_ref().and_then(|v| v.as_u64());
    match number.and_then(|n| u16::try_from(n).ok()) {
        Some(port) if port > 0 => Ok(port),
        _ => bail!(
            "{name} input '{input}' is held on the port its param '{key}' names, and the call \
             wrote {} there; a loopback port is 1 to 65535",
            value.map_or("nothing".to_string(), |v| v.to_string())
        ),
    }
}

/// The format a port feed is conformed to: the input it follows (`like`),
/// or the clock input's where that is the same kind, with the port's own
/// pixel or sample format where it names one.
fn conformed(
    name: &str,
    shape: &NodeShape,
    input: &ffrwd_wasm_runtime::node::InputPort,
    bound: &[BoundStream],
) -> Result<StreamFormat> {
    let followed = input.accepts.like.clone().or_else(|| match &shape.clock {
        Clock::Input(clock) => Some(clock.clone()),
        _ => None,
    });
    let model = followed
        .as_ref()
        .and_then(|port| bound.iter().find(|b| &b.port == port))
        .filter(|b| b.format.kind() == input.kind)
        .map(|b| b.format.clone());
    let Some(model) = model else {
        bail!(
            "{name} input '{}' is held on a port and follows no bound {} input (accepts.like), \
             so the host cannot say what a feed is conformed to",
            input.name,
            input.kind.name()
        );
    };
    Ok(match model {
        StreamFormat::Video(v) => {
            let pix_fmt = match input.accepts.pixel_formats.first() {
                Some(first) => intern(first, KNOWN_PIX_FMTS, "pixel format")?,
                None => v.pix_fmt,
            };
            StreamFormat::Video(VideoFormat {
                pix_fmt,
                frame_len: crate::frame_len_for(pix_fmt, v.width, v.height)?,
                ..v
            })
        }
        StreamFormat::Audio(a) => {
            let sample_fmt = match input.accepts.sample_formats.first() {
                Some(first) => intern(first, KNOWN_SAMPLE_FMTS, "sample format")?,
                None => a.sample_fmt,
            };
            StreamFormat::Audio(AudioFormat { sample_fmt, ..a })
        }
        other => other,
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
        .map(|(port, id)| bound_stream(port, *id, &defs[*id as usize], StreamHint::default()))
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
    let assembler = Assembler::new(&shape, &bound, name, &[])?;
    Ok(Opening {
        spec: LaneSpec {
            name: name.to_string(),
            state: Vec::new(),
            bound: bound.iter().map(|b| b.id).collect(),
            ports: vec![None; shape.outputs.len()],
            shape,
            intake: Intake::Assembled(Box::new(assembler)),
            tick_base: tick,
            runners: vec![Box::new(node)],
            opener: Some(opener),
            rows: None,
        },
        outputs,
        feeds: Vec::new(),
    })
}

/// `leaky` over one picture stream, dropping what arrives too late.
fn open_leaky(call: &NodeCall, pads: &[(String, u32)], defs: &[StreamDef]) -> Result<Opening> {
    let name = &call.module;
    let [(port, id)] = pads else {
        bail!(
            "{name} reads one picture stream, and this chain wires {}",
            pads.len()
        );
    };
    let def = &defs[*id as usize];
    let format = |media| Format {
        media,
        time_base: def.base,
    };
    let (leaky, kind) = match &def.format {
        StreamFormat::Video(v) => (
            Leaky::open(&call.options, &format(Media::Video(*v)))?,
            PortKind::Video,
        ),
        StreamFormat::Audio(a) => (
            Leaky::open(&call.options, &format(Media::Audio(*a)))?,
            PortKind::Audio,
        ),
        StreamFormat::Packets(coded) => (
            Leaky::open_coded(&call.options, coded, def.base)?,
            PortKind::Packets,
        ),
        other => bail!(
            "{name} reads pictures, decoded or coded, and {} is {}",
            def.spelling,
            other.kind().name()
        ),
    };
    let node = older::leaky_node(name, leaky, kind);
    let shape = node.shape().clone();
    let bound = vec![bound_stream(port, *id, def, StreamHint::default())];
    let outputs = output_streams(name, node.as_ref(), &shape, &bound, defs, def.base)?;
    let assembler = Assembler::new(&shape, &bound, name, &[])?;
    Ok(Opening {
        spec: LaneSpec {
            name: name.clone(),
            state: Vec::new(),
            bound: vec![*id],
            ports: vec![None],
            shape,
            intake: Intake::Assembled(Box::new(assembler)),
            tick_base: def.base,
            runners: vec![node],
            opener: None,
            rows: None,
        },
        outputs,
        feeds: Vec::new(),
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
        rendition: RenditionMeta::default(),
        row: None,
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
        feeds: Vec::new(),
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
        StreamFormat::Packets(_) => def.frame_rate.or(header.frame_rate),
        _ => header.frame_rate,
    };
    Ok(header)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn picked_ports_are_written_as_a_list_where_the_schema_takes_one() {
        let listed = serde_json::json!({"properties": {"port": {"type": ["array", "integer"]}}});
        let single = serde_json::json!({"properties": {"port": {"type": "integer"}}});
        assert_eq!(
            with_ports("{}", &listed, "port", &[9100, 9101]).unwrap(),
            r#"{"port":[9100,9101]}"#
        );
        assert_eq!(
            with_ports("{}", &single, "port", &[9100, 9101]).unwrap(),
            r#"{"port":9100}"#
        );
        assert_eq!(
            port_params(r#"{"port":[9100,9101]}"#, &listed, "port", "n", "v").unwrap(),
            vec![9100, 9101]
        );
        assert_eq!(
            port_params(r#"{"port":9100}"#, &listed, "port", "n", "v").unwrap(),
            vec![9100]
        );
    }

    /// shape-probe, built for wasm32-wasip2 once per test binary; none
    /// where that target is not installed (CI's module-free job).
    fn probe() -> Option<String> {
        static BUILT: OnceLock<Option<String>> = OnceLock::new();
        BUILT
            .get_or_init(|| {
                let targets = std::process::Command::new("rustup")
                    .args(["target", "list", "--installed"])
                    .output()
                    .ok()?;
                if !String::from_utf8_lossy(&targets.stdout).contains("wasm32-wasip2") {
                    return None;
                }
                let modules = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .parent()
                    .expect("ffrwd-wasm/ has a parent")
                    .join("modules");
                let output = std::process::Command::new("cargo")
                    .args([
                        "build",
                        "--release",
                        "--target",
                        "wasm32-wasip2",
                        "-p",
                        "shape-probe",
                    ])
                    .current_dir(&modules)
                    .output()
                    .expect("spawn cargo build");
                assert!(
                    output.status.success(),
                    "{}",
                    String::from_utf8_lossy(&output.stderr)
                );
                Some(
                    modules
                        .join("target/wasm32-wasip2/release/shape_probe.wasm")
                        .display()
                        .to_string(),
                )
            })
            .clone()
    }

    #[test]
    fn a_picture_read_for_its_timing_alone_is_carried_without_its_bytes() {
        let Some(probe) = probe() else {
            eprintln!("wasm32-wasip2 is not installed: the timing test has no module to run");
            return;
        };
        let dir = std::env::temp_dir().join(format!("ffrwd-timing-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("a scratch directory");
        let tenths = nut::TimeBase { num: 1, den: 10 };
        let small = nut::Stream::video("rgba", 4, 4, tenths).expect("rgba");
        let big = nut::Stream::video("rgba", 64, 64, tenths).expect("rgba");
        let mut wire = Vec::new();
        {
            let mut muxer = nut::Muxer::with_streams(&mut wire, &[small, big]).expect("headers");
            for k in 0..10i64 {
                muxer.write_frame_to(0, k, &[k as u8; 64]).expect("a frame");
                muxer
                    .write_frame_to(1, k, &vec![k as u8; 64 * 64 * 4])
                    .expect("a frame");
            }
        }
        let input = dir.join("in.nut");
        std::fs::write(&input, &wire).expect("write the input");
        let port = {
            let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("a port");
            listener.local_addr().expect("its address").port()
        };
        let run = |chain: &str, out: &str| {
            let argv: Vec<String> = [
                "-f",
                "nut",
                "-i",
                &input.display().to_string(),
                "-m",
                &format!("shape_probe={probe}"),
                "-filter_complex",
                chain,
                "-map",
                "[s]",
                "-f",
                "ndjson",
                &dir.join(out).display().to_string(),
            ]
            .iter()
            .map(|a| a.to_string())
            .collect();
            let args = crate::parse_args(argv).expect("a command line");
            let crate::Modules::Network { bindings, wiring } = &args.modules else {
                panic!("a network");
            };
            let carried = run_carrying(&args, bindings, wiring).expect("the run");
            let rows: Vec<serde_json::Value> = std::fs::read_to_string(dir.join(out))
                .expect("spots")
                .lines()
                .map(|l| serde_json::from_str(l).expect("a row"))
                .collect();
            (carried.bytes.load(Ordering::Relaxed), rows)
        };

        let (bytes, rows) = run(
            &format!("[v=0:v][size=0:v:1]shape_probe=port={port}[spots=s]"),
            "timing.ndjson",
        );
        assert_eq!(bytes, 10 * 64, "the clock's pictures alone are carried");
        let sizes: Vec<serde_json::Value> = rows.iter().map(|r| r["size"].clone()).collect();
        assert_eq!(
            sizes.iter().take(3).cloned().collect::<Vec<_>>(),
            vec![
                serde_json::json!([0]),
                serde_json::json!([1]),
                serde_json::json!([2])
            ],
            "every frame of the timing input reaches the node, by its time"
        );

        let (bytes, _) = run(
            &format!("[v=0:v:1][size=0:v]shape_probe=port={port}[spots=s]"),
            "bytes.ndjson",
        );
        assert_eq!(
            bytes,
            10 * 64 * 64 * 4,
            "the large picture as the clock is carried whole, the small as timing not at all"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
