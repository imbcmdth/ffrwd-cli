//! The command line of each older world's module that rides alone, read
//! onto its adapter and the one node loop: which outputs it may have, the
//! order its inputs and outputs open in, and what it is told about each pad.
//! What each world's host did is kept, refusals word for word; only the
//! loop is shared.

use std::sync::Arc;
use std::time::Duration;

use anyhow::{bail, Context, Result};
use ffrwd_wasm::nut;
use ffrwd_wasm_runtime::node::{BoundStream, Node, Rational, StreamFormat};
use ffrwd_wasm_runtime::runtime::{self, CodedFormat, SinkInput, TimeBase};

use crate::adapters::{
    DataFilterNode, PacketFilterNode, PacketSinkNode, PacketSourceNode, RowsNode, SINK_RATE,
};
use crate::node_loop::{Intake, Outputs, Run};
use crate::tick::DataDrive;
use crate::{
    coded_pad, coded_stream_for, format_from_stream, heartbeat_packet, is_a_file,
    open_frame_output, open_input, read_ndjson_rows, read_rows, resolve_pad, source_tracks,
    spawn_pad_readers, write_coded_packet, Args, InputPath, OutputKind, OutputSpec, PadQueues,
    RowOutput, RowsQueue, ANNOTATIONS_IN, DATA_CODEC, EDGE_FORMAT, ROWS_FORMAT,
};

/// How long a packet sink goes without a call before it is called with no
/// packets at all. A module runs only inside a host call, and one that drives
/// something of its own there - a network session, whose acknowledgements
/// arrive between packets - would otherwise stand still for as long as its
/// packets do: a stream relayed from elsewhere arrives in lumps a second
/// apart. A fiftieth of a second is less than one AAC frame, so a sink fed
/// live sound is called about as often as it already was, and less than a
/// round trip to a public relay; a call with nothing to do costs a module
/// microseconds. It is the sink's rate clock.
///
/// A packet filter is not called idle: what it writes is packets, and those
/// come only with packets.
fn sink_idle() -> Duration {
    let rate: Rational = SINK_RATE;
    Duration::from_secs(rate.den as u64) / rate.num as u32
}

/// The rate clock's own time base, which its ticks are counted in.
fn rate_base(rate: Rational) -> TimeBase {
    TimeBase {
        num: rate.den as u64,
        den: rate.num as u64,
    }
}

const MICROS: TimeBase = TimeBase {
    num: 1,
    den: 1_000_000,
};

/// A sink or filter pad as the node is bound it: its kind's port, its pad
/// index the id it is named by.
fn bound_pad(pad: usize, input: SinkInput) -> BoundStream {
    let data = input.stream.format == CodedFormat::Data;
    BoundStream {
        port: input.stream.format.kind().to_string(),
        id: pad as u32,
        time_base: input.stream.time_base,
        format: if data {
            StreamFormat::Data(input.stream.codec.clone())
        } else {
            StreamFormat::Packets(input.stream.clone())
        },
        info: input.info,
        rendition: input.rendition,
        row: Some(input.row),
        decode_delay: input.decode_delay,
        latency: None,
        hint: Default::default(),
    }
}

/// Every pad's header as its reader reports it, in pad order.
fn pad_streams(
    headers: std::sync::mpsc::Receiver<(usize, Result<nut::Stream>)>,
    pads: usize,
) -> Result<Vec<nut::Stream>> {
    let mut streams: Vec<Option<nut::Stream>> = (0..pads).map(|_| None).collect();
    for _ in 0..pads {
        let (pad, stream) = headers.recv().expect("every reader reports its header");
        streams[pad] = Some(stream.with_context(|| format!("input {pad}"))?);
    }
    Ok(streams
        .into_iter()
        .map(|s| s.expect("every pad reported"))
        .collect())
}

/// The encoded inputs of a packet sink through ONE instance: packets handed
/// through untouched, in decode order per pad, rows to the row outputs. No
/// frames leave, so the only outputs are rows and null.
///
/// Pads run independently. Packets are not frames: nothing pairs a pad's
/// packet with another pad's, so there is no lockstep to hold and a call may
/// carry packets on some pads and none on others. One reader thread per pad
/// takes blocking reads off its pipe into a byte-bounded queue, and the
/// drive loop hands the module whatever has arrived: ONE producer feeding
/// pads at unequal packet rates interleaves its writes by dts, so a loop
/// that waited on a specific pad would deadlock against the producer's own
/// blocking write once the other pads' pipes filled. The queue bound is the
/// flow control: a stalled consumer stops the producer instead of buffering
/// it without limit. The wasm instance is called from this thread alone;
/// only the I/O grows threads.
pub fn run_packet_sink(args: &Args, module: &str, params: &str) -> Result<()> {
    if args.annotations.input {
        bail!(
            "a packet sink reads encoded packets and no rows arrive with them, so -annotations \
             {ANNOTATIONS_IN} has nothing to give it"
        );
    }
    let mut outputs = Outputs::new(0);
    for output in &args.outputs {
        match output.kind {
            OutputKind::Rows => outputs.add_rows(RowOutput::open(&output.path)?),
            // A null output opens nothing; the module's own effects are the
            // product.
            OutputKind::Null => {}
            _ => bail!(
                "{}: a packet sink emits rows alone; its outputs are -f {ROWS_FORMAT} and -f null",
                output.spelling
            ),
        }
    }

    // The readers start before anything else: each opens its own input and
    // pumps packets into its bounded queue from the first byte, so a fast
    // producer is drained while a slow one's whole chain still warms up, and
    // while the module's own open - which may dial a relay - takes its time.
    // The threads are not joined: on an error the process exits and takes a
    // reader blocked in a pipe read with it, which a join would wait on
    // forever.
    let pads = args.inputs.len();
    let queues = Arc::new(PadQueues::new(pads));
    let headers = spawn_pad_readers(&args.inputs, &queues, false);
    let streams = pad_streams(headers, pads)?;
    let mut bound = Vec::with_capacity(pads);
    for (pad, stream) in streams.iter().enumerate() {
        let (row, rendition) = resolve_pad(&args.pads, pad);
        bound.push(bound_pad(
            pad,
            coded_pad(args, module, pad, stream, row, rendition.into())?,
        ));
    }
    let node = PacketSinkNode::open(module, &bound, params)
        .with_context(|| format!("opening module {module}"))?;
    Run {
        node: Box::new(node),
        intake: Intake::Arrivals {
            queues,
            ids: (0..pads as u32).collect(),
            data: streams.iter().map(nut::Stream::is_json).collect(),
            rows: None,
            idle: Some(sink_idle()),
            base: rate_base(SINK_RATE),
        },
        outputs,
        doing: Some("processing packets"),
        final_on_failure: false,
    }
    .drive()
}

/// The encoded inputs of a packet filter through ONE instance: packets in,
/// packets out, with `-rows-in`'s rows arriving beside them.
///
/// The packet side is a sink's, pad for pad, and the difference is the other
/// end: each pad has an `-f nut` output, written from the header `init`
/// answered for it, and one writer thread per output so a pad whose consumer
/// is slow blocks alone. A data pad leaves as the JSON stream it arrived as,
/// its messages placed on its output's timeline and the heartbeats that
/// arrived beside them handed on.
///
/// Rows arrive on their own schedule. The reader thread fills a bounded
/// queue from the first line, and every call is handed whatever is in it.
/// Nothing here pairs a row with a packet: the row carries its own time and
/// the module decides where it belongs. On the final call the queue is
/// drained to the end of the rows input, so nothing written is left unseen.
pub fn run_packet_filter(args: &Args, module: &str, params: &str) -> Result<()> {
    if args.annotations.input || args.annotations.output {
        bail!(
            "a packet filter carries encoded packets, not frames, so -annotations has nothing \
             to give or take here; its rows arrive through -rows-in"
        );
    }

    let mut pad_outputs: Vec<&OutputSpec> = Vec::new();
    let mut row_outputs: Vec<RowOutput> = Vec::new();
    for output in &args.outputs {
        match output.kind {
            OutputKind::Frames => pad_outputs.push(output),
            OutputKind::Rows => row_outputs.push(RowOutput::open(&output.path)?),
            OutputKind::Null => {}
            _ => bail!(
                "{}: a packet filter writes encoded packets and rows; its outputs are \
                 -f {EDGE_FORMAT}, -f {ROWS_FORMAT} and -f null",
                output.spelling
            ),
        }
    }
    let pads = args.inputs.len();
    if pad_outputs.len() != pads {
        bail!(
            "{module} hands on the packets of every pad it reads: {pads} -i input(s) need \
             {pads} -f {EDGE_FORMAT} output(s), and this command gives {}",
            pad_outputs.len()
        );
    }

    // The rows reader starts here, with the pad readers and before the
    // module is opened: started after the open, a short input can be
    // entirely in the module's hands before the first row is read, and where
    // a module puts a row would depend on which thread won. Several
    // `-rows-in` fill ONE queue, so every argument's rows reach the module
    // through one list and each row says which argument it came from.
    let rows = Arc::new(RowsQueue::new(args.rows_in.len().max(1)));
    // Every input a file is what lets the rows settle before packet one; one
    // pipe among them and none of them wait, since packets never wait on a
    // pipe.
    let rows_from_file =
        !args.rows_in.is_empty() && args.rows_in.iter().all(|r| is_a_file(&r.path));
    if args.rows_in.is_empty() {
        rows.close_one();
    }
    for read in &args.rows_in {
        let read = read.clone();
        let queue = Arc::clone(&rows);
        std::thread::spawn(move || match open_input(&read.path) {
            Ok(reader) => read_rows(reader, &queue, read.arg.as_deref()),
            Err(error) => queue.fail(error.context("opening -rows-in")),
        });
    }

    let queues = Arc::new(PadQueues::new(pads));
    let headers = spawn_pad_readers(&args.inputs, &queues, false);
    let streams = pad_streams(headers, pads)?;
    let mut bound = Vec::with_capacity(pads + 1);
    for (pad, stream) in streams.iter().enumerate() {
        let (row, rendition) = resolve_pad(&args.pads, pad);
        bound.push(bound_pad(
            pad,
            coded_pad(args, module, pad, stream, row, rendition.into())?,
        ));
    }
    let rows_id = pads as u32;
    let node = PacketFilterNode::open(module, &bound, Some(rows_id), params)
        .with_context(|| format!("opening module {module}"))?;

    // A filter that acts on rows and is given none would run blind.
    if args.rows_in.is_empty() && node.reads_rows() {
        bail!(
            "{} reads the rows woven into its packets, and this command gives it none; \
             name them with -rows-in <path>",
            node.name()
        );
    }
    // A FILE's rows are all there to be read, so they are read before the
    // first call rather than raced against it.
    if rows_from_file {
        rows.settle()?;
    }

    let mut outputs = Outputs::new(pads);
    for output in row_outputs {
        outputs.add_rows(output);
    }
    for (pad, output) in pad_outputs.iter().enumerate() {
        // The codec, geometry and time base are the arriving stream's and
        // not a filter's to change, so the header written is the input's
        // own; only the out-of-band header is what `init` answered.
        let mut header = streams[pad].clone();
        header.extradata = node.streams()[pad].extradata.clone();
        let muxer = open_frame_output(&output.path, &header, false)
            .with_context(|| format!("opening output {}", output.spelling))?;
        outputs.track(pad, muxer, &output.spelling, true);
    }
    Run {
        node: Box::new(node),
        intake: Intake::Arrivals {
            queues,
            ids: (0..pads as u32).collect(),
            data: streams.iter().map(nut::Stream::is_json).collect(),
            rows: Some((rows, rows_id)),
            idle: None,
            base: MICROS,
        },
        outputs,
        doing: Some("processing packets"),
        final_on_failure: true,
    }
    .drive()
}

/// A data filter through ONE instance: messages on its data pads and the
/// time on its clock pads in, messages and rows out.
///
/// Every `-i` is a NUT, in the order the call names its arguments. A JSON
/// stream is a DATA pad; a video or audio stream, raw or coded, is a CLOCK
/// pad, read for its frames' pts alone. Each input is read on a thread of
/// its own into a bounded queue, the way a packet sink's are.
///
/// With a clock pad the module is called once each time the first clock
/// pad's pts advances, with `now` that pts, handed the messages that have
/// arrived and are not ahead of it; a data pad read from a regular file is
/// read past each frame before the frame is called for. Once the first
/// clock pad has ended each batch that arrives is a call. Without a clock
/// pad every batch that arrives is a call, with no `now`.
///
/// Each output is a JSON NUT in the module's own time base, written by a
/// thread of its own. Every output carries heartbeats: one at pts 0 the
/// moment its header is out, before any input has said what it carries,
/// since the ffmpeg reading it may be the one writing the clock; then one
/// each time the clock moves on with nothing written for a tenth of a
/// second, or, without a clock, as far as the data pads have been read.
pub fn run_data_filter(args: &Args, module: &str, params: &str) -> Result<()> {
    if args.annotations.input || args.annotations.output {
        bail!(
            "a data filter reads and writes data streams, not frames, so -annotations has \
             nothing to give or take here"
        );
    }
    if args.pads.iter().any(Option::is_some) {
        bail!("-pad follows a packet sink's -i; {module} is a data filter");
    }

    let pads = args.inputs.len();
    let queues = Arc::new(PadQueues::new(pads));
    let headers = spawn_pad_readers(&args.inputs, &queues, true);

    let mut node =
        DataFilterNode::load(module).with_context(|| format!("opening module {module}"))?;
    let described = node.described().clone();
    let mut data_outputs: Vec<&OutputSpec> = Vec::new();
    let mut row_outputs: Vec<RowOutput> = Vec::new();
    for output in &args.outputs {
        match output.kind {
            OutputKind::Frames => data_outputs.push(output),
            OutputKind::Rows => row_outputs.push(RowOutput::open(&output.path)?),
            OutputKind::Null => {}
            _ => bail!(
                "{}: a data filter writes data streams and rows; its outputs are \
                 -f {EDGE_FORMAT}, -f {ROWS_FORMAT} and -f null",
                output.spelling
            ),
        }
    }
    if data_outputs.len() != described.outputs.len() {
        bail!(
            "{} writes {} data stream(s), and this command gives {} -f {EDGE_FORMAT} output(s)",
            described.meta.name,
            described.outputs.len(),
            data_outputs.len()
        );
    }

    // Every output's header and first heartbeat go out before anything is
    // waited on.
    let out_base = nut::TimeBase {
        num: described.time_base.num,
        den: described.time_base.den,
    };
    let mut outputs = Outputs::new(data_outputs.len());
    for output in row_outputs {
        outputs.add_rows(output);
    }
    for (index, output) in data_outputs.iter().enumerate() {
        let mut muxer = open_frame_output(&output.path, &nut::Stream::json(out_base), false)
            .with_context(|| format!("opening output {}", output.spelling))?;
        write_coded_packet(&mut muxer, &heartbeat_packet(0))
            .and_then(|()| Ok(muxer.flush()?))
            .with_context(|| format!("writing output {}", output.spelling))?;
        outputs.track(index, muxer, &output.spelling, false);
    }

    let streams = pad_streams(headers, pads)?;
    let mut bound = Vec::with_capacity(pads);
    for (pad, stream) in streams.iter().enumerate() {
        bound.push(data_pad(module, pad, stream)?);
    }
    node.init(&bound, params)
        .with_context(|| format!("opening module {module}"))?;

    let data: Vec<bool> = streams.iter().map(nut::Stream::is_json).collect();
    let clock = data.iter().position(|is_data| !is_data);
    let settled = data
        .iter()
        .zip(&args.inputs)
        .map(|(is_data, path)| *is_data && is_a_file(path))
        .collect();
    let drive = DataDrive::new(
        bound
            .iter()
            .zip(&data)
            .map(|(b, is_data)| (b.time_base, *is_data))
            .collect(),
        settled,
        clock,
    );
    Run {
        node: Box::new(node),
        intake: Intake::Drive {
            queues,
            ids: (0..pads as u32).collect(),
            data,
            clock: clock.map(|pad| (pad, bound[pad].time_base)),
            drive,
        },
        outputs,
        doing: Some("processing messages"),
        final_on_failure: false,
    }
    .drive()
}

/// One argument of a data filter, from the NUT header its input opened with:
/// a JSON stream is a data pad, and a video or audio stream - raw or coded -
/// is a clock pad.
fn data_pad(module: &str, pad: usize, stream: &nut::Stream) -> Result<BoundStream> {
    let time_base = TimeBase {
        num: stream.time_base.num,
        den: stream.time_base.den,
    };
    let format = if stream.is_json() {
        StreamFormat::Data(DATA_CODEC.to_string())
    } else {
        match stream.media {
            nut::Media::Video { .. } | nut::Media::Audio { .. } => match format_from_stream(stream)
            {
                Ok(format) => match format.media {
                    runtime::Media::Video(video) => StreamFormat::Video(video),
                    runtime::Media::Audio(audio) => StreamFormat::Audio(audio),
                },
                // A coded clock is read for its times alone.
                Err(_) => StreamFormat::Packets(runtime::CodedStream {
                    codec: stream.codec_name().unwrap_or_default().to_string(),
                    time_base,
                    format: match stream.media {
                        nut::Media::Audio {
                            sample_rate,
                            channels,
                        } => CodedFormat::Audio {
                            sample_rate,
                            channels,
                            channel_layout: None,
                        },
                        _ => CodedFormat::Video {
                            width: 0,
                            height: 0,
                            sample_aspect_ratio: None,
                            color: None,
                        },
                    },
                    extradata: Vec::new(),
                    profile: None,
                    level: None,
                }),
            },
            nut::Media::Other { .. } => bail!(
                "{module} reads data streams and clocks, and input {pad} carries a {} stream \
                 tagged {}; a data pad is {DATA_CODEC}, a clock pad video or audio",
                stream.kind(),
                stream.fourcc_name()
            ),
        }
    };
    Ok(BoundStream {
        port: format!("in{pad}"),
        id: pad as u32,
        info: runtime::StreamInfo::default(),
        time_base,
        format,
        rendition: Default::default(),
        row: None,
        decode_delay: 0,
        latency: None,
        hint: Default::default(),
    })
}

/// A rows module through ONE instance: no stream at all. `-rows-in`'s whole
/// file is read first, then `process` is called once with every row, then
/// `finish`.
pub fn run_rows_module(
    args: &Args,
    module_path: &str,
    params: &str,
    rows_in: &InputPath,
) -> Result<()> {
    if args.annotations.input || args.annotations.output {
        bail!(
            "{module_path}: a rows module reads no stream and writes none, so -annotations has \
             nothing to carry rows on"
        );
    }
    let mut outputs = Outputs::new(1);
    outputs.rows_of(0);
    for output in &args.outputs {
        match output.kind {
            OutputKind::Rows => outputs.add_rows(RowOutput::open(&output.path)?),
            OutputKind::Null => {}
            _ => bail!(
                "{}: a rows module emits rows alone; its outputs are -f {ROWS_FORMAT} and -f null",
                output.spelling
            ),
        }
    }
    let rows = read_ndjson_rows(open_input(rows_in)?)?;
    let node = RowsNode::open(module_path, params)
        .with_context(|| format!("opening module {module_path}"))?;
    Run {
        node: Box::new(node),
        intake: Intake::Rows {
            id: 0,
            rows: Some(rows),
        },
        outputs,
        doing: None,
        final_on_failure: false,
    }
    .drive()
}

/// A packet source rides alone: no `-i`, one `-f nut` output per track the
/// command asked for, each naming its track with `-track`. Those indices are
/// what `open` subscribes to, so output i carries the opened catalog's track
/// i and nothing arrives that nobody reads.
///
/// An output's header carries its track's decode delay, which the source
/// does not say: the count of a track's leading packets with no dts is it,
/// and it takes packets to learn. So the source's first call pulls until
/// every track's has settled, and every header is written before a packet
/// goes anywhere. Past the headers each output gets its own writer: a reader
/// opens its inputs one at a time and reads packets off each to finish
/// opening it, so one loop writing every output would stop on the first
/// full pipe with the packets that would drain it unsent.
pub fn run_packet_source(args: &Args, module: &str, params: &str) -> Result<()> {
    if !args.inputs.is_empty() {
        bail!(
            "{module} is a packet source: it produces its own packets and reads no -i input; \
             this command gives it {}",
            args.inputs.len()
        );
    }
    if args.annotations.input || args.annotations.output {
        bail!(
            "a packet source's outputs carry encoded packets, not frames, so -annotations has \
             nothing to give or take here"
        );
    }
    for output in &args.outputs {
        if output.kind != OutputKind::Frames {
            bail!(
                "{}: a packet source writes its tracks as -f {EDGE_FORMAT}; -f {} is not that",
                output.spelling,
                output.kind.format()
            );
        }
    }

    let selected = source_tracks(&args.outputs, module)?;
    let node = PacketSourceNode::open(module, params, &selected)
        .with_context(|| format!("opening module {module}"))?;
    let mut outputs = Outputs::new(selected.len());
    outputs.flush_every_call();
    for (slot, output) in args.outputs.iter().enumerate() {
        let path = output.path.clone();
        let spelling = output.spelling.clone();
        let track = node.catalog().tracks[slot].clone();
        outputs.track_later(
            slot,
            Box::new(move |node: &dyn Node| {
                let decode_delay = u64::from(node.decode_delay(slot).unwrap_or(0));
                let stream = coded_stream_for(&track.stream, decode_delay)?;
                open_frame_output(&path, &stream, false)
                    .with_context(|| format!("opening output {spelling}"))
            }),
            &output.spelling,
            false,
        );
    }
    Run {
        node: Box::new(node),
        intake: Intake::Pull,
        outputs,
        doing: None,
        final_on_failure: false,
    }
    .drive()
}
