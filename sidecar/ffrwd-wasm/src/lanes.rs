//! Node lanes: the scheduler a node network runs on.
//!
//! Every node is a lane. What arrives on its bound streams goes into its
//! assembler, which cuts ticks by the node's pairing rules once, centrally,
//! before any tick goes to a worker; every tick is a task carrying an
//! ordinal. A pure node admits tasks concurrently, one instance per worker;
//! every other node admits one at a time, in order. Results are reassembled
//! by ordinal before anything leaves a lane, so what each lane emits, the
//! progress it sends down its data edges and the order things reach its
//! consumers are the same at every worker count.
//!
//! A lane whose node keeps rows as state opens every instance before the
//! first tick, since a live run cannot hold a tick while one starts, and
//! hands each instance, with its next tick, the rows of the ticks it did not
//! process. Every other pure lane opens instances as the run needs them.
//!
//! Backpressure is credit: a task is dispatched only while every lane and
//! output it feeds has room, and a reader waits for room on whatever reads
//! its streams. A generator, a node clocked by a rate with no inputs, is cut
//! a tick at a time as its queue has room, so it runs as fast as its
//! outputs drain. A rate clock whose inputs are all delivered as they arrive
//! (a publisher that only needs turns) is cut a tick when something has
//! arrived, or one period of its rate after its last tick, until they end;
//! then a tick takes what arrived untaken and its last call follows at
//! once, whatever its tick count. A rate clock with paired inputs runs on
//! past their newest time first.

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context, Result};
use ffrwd_wasm_runtime::node::{
    self as model, Emitted, Node, NodeShape, Payload, Tick, TickFrame, TickStream,
};
use ffrwd_wasm_runtime::profile::PROFILE;
use ffrwd_wasm_runtime::runtime::{Frame, Message, TimeBase};

use crate::adapters::frames_tick;
use crate::edges::{Out, Queue};
use crate::hold::SourceInfo;
use crate::tick::{Assembler, Item, Lockstep};

/// Opens one more instance of a lane's node.
pub type Opener = Arc<dyn Fn() -> Result<Box<dyn Node>> + Send + Sync>;

/// What waiting for room on a port feed came to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Room {
    Open,
    Full,
    Stopped,
}

/// An output port somebody reads: the stream it is, its time base, and the
/// latency its progress is held back by.
#[derive(Debug, Clone, Copy)]
pub struct PortOut {
    pub stream: u32,
    pub base: TimeBase,
    pub latency: f64,
}

/// How a lane's ticks are cut.
pub enum Intake {
    /// The node's own shape: its clock and every input's pairing.
    Assembled(Box<Assembler>),
    /// A host node on a data edge: the messages between two of its
    /// producer's progress marks, as one tick.
    Host { id: u32 },
    /// An older module in a network of older modules: its calls cut the way
    /// its world's host cut them, a window as soon as it is whole and a last
    /// call with what the strides left and the rows no frame carried.
    Windows {
        lockstep: Lockstep,
        /// The stream each pad reads.
        pads: Vec<u32>,
    },
}

/// One node of the network, opened, as the scheduler takes it over.
pub struct LaneSpec {
    pub name: String,
    pub shape: NodeShape,
    pub intake: Intake,
    /// The streams bound to its inputs, by id.
    pub bound: Vec<u32>,
    /// The streams bound to inputs held on the GPU, each with the format
    /// its frames are uploaded in and the size the input reads them at: a
    /// frame arriving in host memory is put on the GPU as it enters the
    /// lane, and one of another size is resized there.
    pub gpu: HashMap<u32, GpuSpec>,
    /// The stream ids whose rows the node keeps as state.
    pub state: Vec<u32>,
    pub tick_base: TimeBase,
    pub runners: Vec<Box<dyn Node>>,
    pub opener: Option<Opener>,
    pub ports: Vec<Option<PortOut>>,
    pub rows: Option<PortOut>,
    /// Runs only once everything it feeds has taken all it was handed, so
    /// nothing it hands on waits in a queue: `leaky` judges a picture late
    /// or not as it hands it on, and a picture queued after that is later
    /// than it was judged.
    pub unbuffered: bool,
}

/// The format a stream's frames are held on the GPU in, for a lane whose
/// input is there.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuSpec {
    /// The stream's format, which a frame arriving in host memory is
    /// uploaded in.
    pub pix_fmt: String,
    pub width: u32,
    pub height: u32,
    /// The size the input reads the frames at: the stream's, or that of
    /// the input it follows (`accepts.like`).
    pub reads: (u32, u32),
}

/// Who reads a stream: a lane, or one stream of an output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Consumer {
    Lane(usize),
    Writer(usize, usize),
}

pub struct Plan {
    pub lanes: Vec<LaneSpec>,
    pub consumers: HashMap<u32, Vec<Consumer>>,
    pub writers: Vec<Arc<Queue>>,
    /// Streams bound again under another id, as one node binding one stream
    /// on two ports needs: everything on the first reaches each of these.
    pub mirrors: HashMap<u32, Vec<u32>>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ClockKind {
    Input,
    Windows,
    /// A rate clock; whether any stream is bound to the node.
    Rate {
        inputs: bool,
    },
    SelfClocked,
    Host,
}

enum HostEvent {
    Message(Message),
    Progress(i64),
    End,
}

enum LaneIntake {
    Assembled(Box<Assembler>),
    Host {
        id: u32,
        events: VecDeque<HostEvent>,
    },
    Windows {
        lockstep: Lockstep,
        pads: Vec<u32>,
        /// The rows with no frame to ride that reached the first pad.
        trailing: Vec<String>,
    },
}

/// What a worker takes from a lane in one turn: the tasks it runs in
/// order, the instance that runs them, and that instance or the means to
/// open it.
struct Turn {
    tasks: Vec<Task>,
    instance: usize,
    runner: Option<Box<dyn Node>>,
    opener: Option<Opener>,
}

struct Task {
    ordinal: u64,
    tick: Tick,
    end: Option<i64>,
}

/// A tick's result waiting for its turn to leave.
struct Done {
    emitted: Emitted,
    pts: i64,
    end: Option<i64>,
    last: bool,
    instance: usize,
}

struct PortState {
    out: PortOut,
    /// The pts of the last message the port carried.
    last: Option<i64>,
    /// The progress last sent down the port.
    sent: Option<i64>,
}

struct Lane {
    name: String,
    intake: LaneIntake,
    clock: ClockKind,
    base: TimeBase,
    bound: Vec<u32>,
    gpu: HashMap<u32, GpuSpec>,
    /// The streams whose frames this lane has put on the GPU, said once
    /// each on stderr.
    #[cfg_attr(not(feature = "gpu"), allow(dead_code))]
    uploaded: HashSet<u32>,
    /// The streams whose frames this lane has resized on the GPU, said
    /// once each on stderr.
    #[cfg_attr(not(feature = "gpu"), allow(dead_code))]
    resized: HashSet<u32>,
    state: Vec<u32>,
    queue: VecDeque<Task>,
    next_ordinal: u64,
    in_flight: usize,
    width: usize,
    idle: Vec<(usize, Box<dyn Node>)>,
    created: usize,
    opener: Option<Opener>,
    done: BTreeMap<u64, Done>,
    next_flush: u64,
    ports: Vec<Option<PortState>>,
    rows: Option<PortState>,
    /// The last tick has been cut.
    last_cut: bool,
    /// The last tick has left.
    ended: bool,
    /// The node said it was finished: nothing arriving reaches it again.
    stopped: bool,
    /// The instance that said so, which makes the last call: any other may
    /// have run ticks past it.
    finisher: Option<usize>,
    started: Option<Instant>,
    /// When a rate clock whose inputs all arrive unpaired last cut a tick.
    turned: Option<Instant>,
    /// The next tick of an input clock whose cut stopped at the cap, and
    /// where its interval ends: while one is held back the lane counts as
    /// full to everything it reads.
    withheld: Option<(Tick, Option<i64>)>,
    unbuffered: bool,
}

impl Lane {
    /// `item` as this lane takes it: a frame of a stream bound to an input
    /// held on the GPU is put there as it arrives, once, when it is still
    /// in host memory, and the first such frame of each stream says so on
    /// stderr (`[gpu] <lane>: putting the frames of stream <n> on the GPU`).
    /// A frame of another size than the input reads is resized on the
    /// device, said the same way (`[gpu] <lane>: resizing the frames of
    /// stream <n> on the GPU to <w>x<h>`). A frame already there at its
    /// size (a decoder's picture) is handed on as it is, as is everything
    /// else.
    fn resident(&mut self, stream: u32, item: &Item) -> Result<Item> {
        let Some(spec) = self.gpu.get(&stream) else {
            return Ok(item.clone());
        };
        let Item::Frame(frame) = item else {
            return Ok(item.clone());
        };
        #[cfg(feature = "gpu")]
        {
            use ffrwd_wasm_runtime::gpu::frames::GpuFrame;
            use ffrwd_wasm_runtime::node::FrameData;
            let held = match &frame.data {
                FrameData::Host(bytes) if !bytes.is_empty() => {
                    let uploaded =
                        GpuFrame::upload(&spec.pix_fmt, spec.width, spec.height, Arc::clone(bytes))
                            .with_context(|| {
                                format!(
                                    "{}: putting a frame of stream {stream} on the GPU",
                                    self.name
                                )
                            })?;
                    if self.uploaded.insert(stream) {
                        eprintln!(
                            "[gpu] {}: putting the frames of stream {stream} on the GPU",
                            self.name
                        );
                    }
                    uploaded
                }
                FrameData::Gpu(gpu) if (gpu.width(), gpu.height()) != spec.reads => Arc::clone(gpu),
                _ => return Ok(item.clone()),
            };
            let (width, height) = spec.reads;
            let held = if (held.width(), held.height()) == spec.reads {
                held
            } else {
                let resized = held.resized(width, height).with_context(|| {
                    format!(
                        "{}: resizing a frame of stream {stream} to {width}x{height} on the GPU",
                        self.name
                    )
                })?;
                if self.resized.insert(stream) {
                    eprintln!(
                        "[gpu] {}: resizing the frames of stream {stream} on the GPU to \
                         {width}x{height}",
                        self.name
                    );
                }
                resized
            };
            Ok(Item::Frame(TickFrame {
                pts: frame.pts,
                duration: frame.duration,
                data: FrameData::Gpu(held),
                rows: frame.rows.clone(),
            }))
        }
        #[cfg(not(feature = "gpu"))]
        {
            let _ = (spec, frame);
            bail!(
                "{}: an input held on the GPU, and this ffrwd-wasm was built without GPU support",
                self.name
            )
        }
    }
}

struct State {
    lanes: Vec<Lane>,
    consumers: HashMap<u32, Vec<Consumer>>,
    writers: Vec<Arc<Queue>>,
    mirrors: HashMap<u32, Vec<u32>>,
    error: Option<anyhow::Error>,
    /// Every lane has ended, or every output has stopped taking anything.
    finished: bool,
    cap: usize,
}

struct Shared {
    state: Mutex<State>,
    work: Condvar,
    space: Condvar,
}

impl Shared {
    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn notify(&self) {
        self.work.notify_all();
        self.space.notify_all();
    }
}

fn clock_kind(shape: &NodeShape, intake: &Intake) -> ClockKind {
    match intake {
        Intake::Host { .. } => return ClockKind::Host,
        Intake::Windows { .. } => return ClockKind::Windows,
        Intake::Assembled(_) => {}
    }
    match &shape.clock {
        model::Clock::Input(_) => ClockKind::Input,
        model::Clock::Rate(_) | model::Clock::RateOf(_) => ClockKind::Rate {
            inputs: match intake {
                Intake::Assembled(a) => a.has_inputs(),
                Intake::Host { .. } | Intake::Windows { .. } => true,
            },
        },
        model::Clock::SelfClocked => ClockKind::SelfClocked,
    }
}

/// Opens, all at once, every instance a lane that hands rows round will run,
/// and takes away its way to open more.
fn open_every_instance(spec: &mut LaneSpec, width: usize) -> Result<()> {
    if spec.state.is_empty() || width <= 1 {
        return Ok(());
    }
    let Some(opener) = spec.opener.take() else {
        return Ok(());
    };
    let wanted = width.saturating_sub(spec.runners.len());
    let opened: Vec<Result<Box<dyn Node>>> = thread::scope(|scope| {
        let opening: Vec<_> = (0..wanted).map(|_| scope.spawn(|| opener())).collect();
        opening
            .into_iter()
            .map(|h| {
                h.join()
                    .unwrap_or_else(|_| Err(anyhow!("opening an instance panicked")))
            })
            .collect()
    });
    for runner in opened {
        spec.runners
            .push(runner.with_context(|| format!("opening {}", spec.name))?);
    }
    Ok(())
}

impl State {
    /// Whether everything in `consumers`, the readers of `stream`, takes
    /// more of it. A self-clocked reader is cut a task at a time, each
    /// taking one arrival an input, so what waits for it is the arrivals
    /// queued on its input, which count against the same bound as a task
    /// queue: without that, a decoder ahead of a slower encoder would queue
    /// every picture it made. An input clock is cut no further than the
    /// cap, and while its cut stopped there the lane counts as full, but
    /// to the reader of its audio clock: one packet of sound settles as
    /// many ticks as it has samples over the stride, so while ticks are
    /// held back what waits there is counted in packets, against the same
    /// bound, and that reader is held back only once the lane is that many
    /// packets behind.
    fn room(&self, stream: u32, consumers: &[Consumer]) -> bool {
        consumers.iter().all(|c| match c {
            Consumer::Lane(j) => {
                let lane = &self.lanes[*j];
                let queued = match (&lane.clock, &lane.intake) {
                    (ClockKind::SelfClocked, LaneIntake::Assembled(a)) => {
                        a.queued(stream).unwrap_or(0)
                    }
                    _ => 0,
                };
                let sound = match &lane.intake {
                    LaneIntake::Assembled(a) if lane.clock == ClockKind::Input => {
                        a.audio_backlog(stream)
                    }
                    _ => None,
                };
                lane.stopped
                    || match sound {
                        Some(packets) if lane.withheld.is_some() => packets < self.cap,
                        _ => {
                            lane.queue.len() < self.cap
                                && lane.withheld.is_none()
                                && queued < self.cap
                        }
                    }
            }
            Consumer::Writer(w, _) => self.writers[*w].len() < self.cap,
        })
    }

    /// Whether everything in `consumers` has taken all it was handed.
    fn drained(&self, consumers: &[Consumer]) -> bool {
        consumers.iter().all(|c| match c {
            Consumer::Lane(j) => {
                let lane = &self.lanes[*j];
                lane.stopped || lane.queue.is_empty()
            }
            Consumer::Writer(w, _) => self.writers[*w].idle(),
        })
    }

    fn consumers(&self, stream: u32) -> Vec<Consumer> {
        self.consumers.get(&stream).cloned().unwrap_or_default()
    }

    fn mirrors(&self, stream: u32) -> Vec<u32> {
        self.mirrors.get(&stream).cloned().unwrap_or_default()
    }

    /// One item on `stream`, handed to everything reading it.
    fn arrive(&mut self, stream: u32, item: Item) -> Result<()> {
        for mirror in self.mirrors(stream) {
            self.arrive(mirror, item.clone())?;
        }
        let consumers = self.consumers(stream);
        for consumer in &consumers {
            match *consumer {
                Consumer::Lane(j) => {
                    let lane = &mut self.lanes[j];
                    if lane.stopped {
                        continue;
                    }
                    let item = lane.resident(stream, &item)?;
                    if let Item::Frame(frame) = &item {
                        PROFILE.held.count(frame.data.len());
                    }
                    if let (Item::Message(m), true) = (&item, crate::trace::on()) {
                        let text = String::from_utf8_lossy(&m.data);
                        crate::trace::line(format_args!(
                            "rows in: {} stream {stream} at {}: {}",
                            lane.name,
                            m.pts,
                            text.chars().take(160).collect::<String>()
                        ));
                    }
                    match &mut lane.intake {
                        LaneIntake::Assembled(a) => a.arrive(stream, item)?,
                        LaneIntake::Host { events, .. } => {
                            if let Item::Message(m) = &item {
                                events.push_back(HostEvent::Message(m.clone()));
                            }
                        }
                        LaneIntake::Windows { lockstep, pads, .. } => {
                            let Item::Frame(f) = &item else { continue };
                            // An older module's window reads bytes: a frame
                            // on the GPU is read back here, once.
                            let frame = Frame {
                                pts: f.pts,
                                data: f.data.bytes()?,
                                rows: f.rows.clone(),
                            };
                            let count = pads.len();
                            let mut calls = Vec::new();
                            for pad in (0..count).filter(|p| pads[*p] == stream) {
                                calls.extend(lockstep.push(
                                    pad,
                                    std::slice::from_ref(&frame),
                                    &lane.name,
                                )?);
                            }
                            for call in calls {
                                let tick = frames_tick(&call, &[], false, count, lane.base);
                                Self::push_task(lane, tick, None);
                            }
                        }
                    }
                }
                Consumer::Writer(w, s) => self.writers[w].push(s, out_of(&item)),
            }
        }
        self.cut_all(&consumers)
    }

    fn progress(&mut self, stream: u32, pts: i64) -> Result<()> {
        for mirror in self.mirrors(stream) {
            self.progress(mirror, pts)?;
        }
        let consumers = self.consumers(stream);
        for consumer in &consumers {
            match *consumer {
                Consumer::Lane(j) => {
                    let lane = &mut self.lanes[j];
                    if lane.stopped {
                        continue;
                    }
                    match &mut lane.intake {
                        LaneIntake::Assembled(a) => a.progress(stream, pts)?,
                        LaneIntake::Host { events, .. } => {
                            events.push_back(HostEvent::Progress(pts))
                        }
                        LaneIntake::Windows { .. } => {}
                    }
                }
                Consumer::Writer(w, s) => self.writers[w].push(s, Out::Progress(pts)),
            }
        }
        self.cut_all(&consumers)
    }

    fn end(&mut self, stream: u32) -> Result<()> {
        for mirror in self.mirrors(stream) {
            self.end(mirror)?;
        }
        let consumers = self.consumers(stream);
        for consumer in &consumers {
            match *consumer {
                Consumer::Lane(j) => {
                    let lane = &mut self.lanes[j];
                    if lane.stopped {
                        continue;
                    }
                    match &mut lane.intake {
                        LaneIntake::Assembled(a) => a.end(stream)?,
                        LaneIntake::Host { events, .. } => events.push_back(HostEvent::End),
                        LaneIntake::Windows {
                            lockstep,
                            pads,
                            trailing,
                        } => {
                            let count = pads.len();
                            let mut last = None;
                            for pad in (0..count).filter(|p| pads[*p] == stream) {
                                if let Some(frames) = lockstep.end(pad, &lane.name)? {
                                    last = Some(frames);
                                }
                            }
                            if let Some(frames) = last {
                                let tick = frames_tick(&frames, trailing, true, count, lane.base);
                                Self::push_task(lane, tick, None);
                            }
                        }
                    }
                }
                Consumer::Writer(w, s) => self.writers[w].push(s, Out::End),
            }
        }
        self.cut_all(&consumers)
    }

    /// Rows with no frame to ride, on `stream` after its last frame: the
    /// first pad of an older module's lane takes them to its last call.
    fn trailing(&mut self, stream: u32, rows: &[String]) {
        for mirror in self.mirrors(stream) {
            self.trailing(mirror, rows);
        }
        for consumer in self.consumers(stream) {
            match consumer {
                Consumer::Lane(j) => {
                    if let LaneIntake::Windows { pads, trailing, .. } = &mut self.lanes[j].intake {
                        if pads.first() == Some(&stream) {
                            trailing.extend(rows.iter().cloned());
                        }
                    }
                }
                Consumer::Writer(w, s) => self.writers[w].push(s, Out::Trailing(rows.to_vec())),
            }
        }
    }

    /// A source starts or ends on a port feed's stream.
    fn source(&mut self, stream: u32, source: Option<SourceInfo>) -> Result<()> {
        for mirror in self.mirrors(stream) {
            self.source(mirror, source.clone())?;
        }
        let consumers = self.consumers(stream);
        for consumer in &consumers {
            if let Consumer::Lane(j) = *consumer {
                let lane = &mut self.lanes[j];
                if lane.stopped {
                    continue;
                }
                if let LaneIntake::Assembled(a) = &mut lane.intake {
                    match &source {
                        Some(info) => a.source_open(stream, info.clone())?,
                        None => a.source_close(stream)?,
                    }
                }
            }
        }
        self.cut_all(&consumers)
    }

    /// How many more frames a port feed on `stream` may send before it
    /// waits: the least room among its readers.
    fn hold_room(&self, stream: u32) -> usize {
        self.consumers(stream)
            .iter()
            .filter_map(|consumer| match consumer {
                Consumer::Lane(j) => match &self.lanes[*j].intake {
                    LaneIntake::Assembled(a) if !self.lanes[*j].stopped => a.hold_room(stream),
                    _ => None,
                },
                Consumer::Writer(..) => None,
            })
            .min()
            .unwrap_or(usize::MAX)
    }

    fn cut_all(&mut self, consumers: &[Consumer]) -> Result<()> {
        for consumer in consumers {
            if let Consumer::Lane(j) = consumer {
                self.cut(*j)?;
            }
        }
        Ok(())
    }

    fn push_task(lane: &mut Lane, tick: Tick, end: Option<i64>) {
        if tick.last {
            lane.last_cut = true;
        }
        lane.queue.push_back(Task {
            ordinal: lane.next_ordinal,
            tick,
            end,
        });
        lane.next_ordinal += 1;
    }

    /// Cuts whatever ticks lane `j`'s intake now settles.
    fn cut(&mut self, j: usize) -> Result<()> {
        let cap = self.cap;
        let lane = &mut self.lanes[j];
        if lane.last_cut || lane.stopped {
            return Ok(());
        }
        match lane.clock {
            ClockKind::Input => {
                let LaneIntake::Assembled(a) = &mut lane.intake else {
                    unreachable!("an input clock is assembled")
                };
                // One arrival can settle many ticks (an audio clock cuts
                // `window` samples a tick, so a packet of 1024 makes 1024
                // ticks of one), so the cut stops at the cap as every other
                // clock's does, holding the next tick back if one is
                // settled, and the rest is cut as the queue drains.
                let mut cut: Vec<(Tick, Option<i64>)> = Vec::new();
                let mut last = false;
                while lane.queue.len() + cut.len() < cap {
                    let next = match lane.withheld.take() {
                        Some(held) => held,
                        None => match a.next_clocked()? {
                            Some(tick) => {
                                let end = a.tick_end();
                                (tick, end)
                            }
                            None => break,
                        },
                    };
                    last = next.0.last;
                    cut.push(next);
                    if last {
                        break;
                    }
                }
                if !last && lane.withheld.is_none() && lane.queue.len() + cut.len() >= cap {
                    if let Some(tick) = a.next_clocked()? {
                        let end = a.tick_end();
                        lane.withheld = Some((tick, end));
                    }
                }
                for (tick, end) in cut {
                    Self::push_task(lane, tick, end);
                }
            }
            ClockKind::Rate { inputs: true } => {
                let base = lane.base;
                let LaneIntake::Assembled(a) = &mut lane.intake else {
                    unreachable!("a rate clock is assembled")
                };
                let paced = a.arrival_only() && !a.all_ended();
                let mut cut = Vec::new();
                while lane.queue.len() + cut.len() < cap {
                    if paced
                        && !a.has_arrivals()
                        && lane.turned.is_some_and(|t| t.elapsed() < period(base))
                    {
                        break;
                    }
                    let Some(tick) = a.next_rate(a.all_ended() && a.played_out(base))? else {
                        break;
                    };
                    let last = tick.last;
                    cut.push((tick, a.tick_end()));
                    if paced {
                        lane.turned = Some(Instant::now());
                    }
                    if last {
                        break;
                    }
                }
                for (tick, end) in cut {
                    Self::push_task(lane, tick, end);
                }
            }
            ClockKind::Rate { inputs: false } => {
                let LaneIntake::Assembled(a) = &mut lane.intake else {
                    unreachable!("a rate clock is assembled")
                };
                while lane.queue.len() < cap {
                    let Some(tick) = a.next_rate(false)? else {
                        break;
                    };
                    let end = a.tick_end();
                    lane.queue.push_back(Task {
                        ordinal: lane.next_ordinal,
                        tick,
                        end,
                    });
                    lane.next_ordinal += 1;
                }
            }
            ClockKind::SelfClocked => {
                if !lane.queue.is_empty() || lane.in_flight > 0 {
                    return Ok(());
                }
                let started = *lane.started.get_or_insert_with(Instant::now);
                let LaneIntake::Assembled(a) = &mut lane.intake else {
                    unreachable!("a self-clocked node is assembled")
                };
                let last = a.has_inputs() && a.all_ended();
                if a.has_inputs() && !last && !a.has_arrivals() {
                    return Ok(());
                }
                let pts = i64::try_from(started.elapsed().as_micros()).unwrap_or(i64::MAX);
                let tick = a.next_arrivals(pts, last)?;
                Self::push_task(lane, tick, None);
            }
            ClockKind::Windows => {}
            ClockKind::Host => {
                let LaneIntake::Host { id, events } = &mut lane.intake else {
                    unreachable!("a host node has a host intake")
                };
                let id = *id;
                let mut messages = Vec::new();
                let mut cuts = Vec::new();
                while let Some(event) = events.pop_front() {
                    match event {
                        HostEvent::Message(m) => messages.push(m),
                        HostEvent::Progress(p) => {
                            cuts.push((p, false, std::mem::take(&mut messages)));
                        }
                        HostEvent::End => {
                            cuts.push((0, true, std::mem::take(&mut messages)));
                            break;
                        }
                    }
                }
                for message in messages.into_iter().rev() {
                    events.push_front(HostEvent::Message(message));
                }
                let base = lane.base;
                for (pts, last, messages) in cuts {
                    let tick = Tick {
                        pts,
                        ordinal: 0,
                        time_base: base,
                        last,
                        streams: vec![TickStream {
                            id,
                            messages,
                            progress: (!last).then_some(pts),
                            ..TickStream::default()
                        }],
                    };
                    Self::push_task(lane, tick, (!last).then_some(pts));
                    if last {
                        break;
                    }
                }
            }
        }
        Ok(())
    }

    /// When the next idle turn of a rate clock whose inputs all arrive
    /// unpaired falls due, for a worker with nothing to do to wait until.
    fn next_turn(&self) -> Option<Instant> {
        self.lanes
            .iter()
            .filter(|lane| {
                lane.clock == ClockKind::Rate { inputs: true }
                    && !lane.last_cut
                    && !lane.stopped
                    && lane.queue.len() < self.cap
            })
            .filter_map(|lane| match &lane.intake {
                LaneIntake::Assembled(a) if a.arrival_only() && !a.all_ended() => {
                    lane.turned.map(|t| t + period(lane.base))
                }
                _ => None,
            })
            .min()
    }

    /// Whether lane `i` has a task a worker may start now: queued, under
    /// its width, an instance in hand or in reach, and room on everything
    /// it feeds. The last task waits for every other to finish.
    fn dispatchable(&self, i: usize) -> bool {
        let lane = &self.lanes[i];
        let Some(task) = lane.queue.front() else {
            return false;
        };
        if lane.in_flight >= lane.width || (task.tick.last && lane.in_flight > 0) {
            return false;
        }
        if let (true, Some(finisher)) = (task.tick.last, lane.finisher) {
            if !lane.idle.iter().any(|(i, _)| *i == finisher) {
                return false;
            }
        } else if lane.idle.is_empty() && !(lane.created < lane.width && lane.opener.is_some()) {
            return false;
        }
        let source = matches!(lane.clock, ClockKind::Rate { inputs: false });
        let ports = lane.ports.iter().flatten().chain(lane.rows.iter());
        for port in ports {
            // A stream's copies hold it back as its own readers do.
            let streams = std::iter::once(port.out.stream).chain(self.mirrors(port.out.stream));
            for (stream, consumers) in
                streams.filter_map(|stream| Some((stream, self.consumers.get(&stream)?)))
            {
                let room = if lane.unbuffered {
                    self.drained(consumers)
                } else {
                    self.room(stream, consumers)
                };
                if !room {
                    return false;
                }
                // A source has nothing upstream to wait on, so a hold input it
                // feeds holds it back by its own room, as a socket does.
                if source && consumers.iter().any(|c| self.held_full(*c, stream)) {
                    return false;
                }
            }
        }
        true
    }

    fn held_full(&self, consumer: Consumer, stream: u32) -> bool {
        let Consumer::Lane(j) = consumer else {
            return false;
        };
        match &self.lanes[j].intake {
            LaneIntake::Assembled(a) => a.held_room(stream) == Some(0),
            _ => false,
        }
    }

    /// A rate clock's ticks are cut as its queue has room, a self-clocked
    /// node's as it returns, and an input clock's that the cap withheld as
    /// its queue drains, so all are cut here, where a worker looks for
    /// work, as well as on arrivals. A lane holding a tick back holds back
    /// the readers of what it reads, so it goes first; then the lane with
    /// the longest queue.
    fn pick(&mut self) -> Result<Option<usize>> {
        for j in 0..self.lanes.len() {
            let lane = &self.lanes[j];
            if matches!(lane.clock, ClockKind::Rate { .. } | ClockKind::SelfClocked)
                || (lane.withheld.is_some() && lane.queue.len() < self.cap)
            {
                self.cut(j)?;
            }
        }
        Ok((0..self.lanes.len())
            .filter(|i| self.dispatchable(*i))
            .max_by_key(|i| {
                (
                    self.lanes[*i].withheld.is_some(),
                    self.lanes[*i].queue.len(),
                )
            }))
    }

    /// Takes lane `i`'s front task with an instance, or the means to open
    /// one, and hands a state input its earlier rows. A lane that runs one
    /// tick at a time takes the tasks queued behind the front one too, and
    /// the ticks its clock has settled past the cap, up to [`TURN`] and
    /// short of its last, and runs them in order in one turn: a clock of
    /// small ticks (an audio clock cut a sample a tick, 48000 a second)
    /// would otherwise wait for a worker, and pay a turn of the scheduler
    /// and a wake of every thread waiting on it, for each.
    fn dispatch(&mut self, i: usize) -> Result<Turn> {
        let lane = &mut self.lanes[i];
        let front = lane.queue.pop_front().expect("picked lanes have a task");
        lane.in_flight += 1;
        let wanted = match (front.tick.last, lane.finisher) {
            (true, Some(finisher)) => lane.idle.iter().position(|(i, _)| *i == finisher),
            _ => lane.idle.len().checked_sub(1),
        };
        let (instance, runner, opener) = match wanted.map(|at| lane.idle.remove(at)) {
            Some((instance, runner)) => (instance, Some(runner), None),
            None => {
                let instance = lane.created;
                lane.created += 1;
                (instance, None, lane.opener.clone())
            }
        };
        let mut tasks = vec![front];
        if lane.width == 1 && !lane.unbuffered && !tasks[0].tick.last {
            loop {
                let lane = &mut self.lanes[i];
                while tasks.len() < TURN && lane.queue.front().is_some_and(|t| !t.tick.last) {
                    tasks.extend(lane.queue.pop_front());
                }
                if tasks.len() >= TURN || !lane.queue.is_empty() || lane.withheld.is_none() {
                    break;
                }
                self.cut(i)?;
            }
        }
        let lane = &mut self.lanes[i];
        for task in &mut tasks {
            task.tick.ordinal = task.ordinal;
            if !lane.state.is_empty() {
                if let LaneIntake::Assembled(a) = &mut lane.intake {
                    for stream in &mut task.tick.streams {
                        if lane.state.contains(&stream.id) {
                            stream.earlier_rows = a.earlier.owed(instance, stream.id, task.ordinal);
                        }
                    }
                    a.earlier.processed(instance, task.ordinal);
                    if lane.created >= lane.width {
                        a.earlier.forget_told(lane.created);
                    }
                }
            }
            if let LaneIntake::Assembled(a) = &mut lane.intake {
                for stream in &mut task.tick.streams {
                    stream.ended_feeds = a.ended.owed(instance, stream.id, task.ordinal);
                }
                a.ended.processed(instance, task.ordinal);
                if lane.created >= lane.width {
                    a.ended.forget_told(lane.created);
                }
            }
        }
        Ok(Turn {
            tasks,
            instance,
            runner,
            opener,
        })
    }

    /// The results of one turn of lane `i`, by ordinal, in the order they
    /// ran.
    fn complete(
        &mut self,
        i: usize,
        results: Vec<(u64, Result<Done>)>,
        instance: usize,
        runner: Option<Box<dyn Node>>,
    ) {
        let lane = &mut self.lanes[i];
        lane.in_flight -= 1;
        let last = results
            .iter()
            .any(|(_, done)| done.as_ref().is_ok_and(|d| d.last));
        if let Some(runner) = runner {
            if !last {
                lane.idle.push((instance, runner));
            }
        }
        let mut failed = None;
        for (ordinal, done) in results {
            match done {
                Ok(done) => {
                    if ordinal >= lane.next_flush {
                        lane.done.insert(ordinal, done);
                    }
                }
                Err(e) => {
                    failed = Some(e.context(format!("in {}", lane.name)));
                    break;
                }
            }
        }
        let flushed = match failed {
            Some(e) => Err(e),
            None => self.flush(i),
        };
        if let Err(e) = flushed {
            if self.error.is_none() {
                self.error = Some(e);
            }
        }
    }

    /// Lane `i`'s results that are next in order leave: each item to its
    /// port's readers, the node's progress down every port, and on the last
    /// tick the end of every port.
    fn flush(&mut self, i: usize) -> Result<()> {
        let mut leaving: Vec<(u32, Delivery)> = Vec::new();
        loop {
            let lane = &mut self.lanes[i];
            let Some(done) = lane.done.remove(&lane.next_flush) else {
                break;
            };
            lane.next_flush += 1;
            let Done {
                emitted,
                pts,
                end,
                last,
                instance,
            } = done;
            let tick_base = lane.base;
            let windows = matches!(lane.intake, LaneIntake::Windows { .. });
            let mut deliveries: Vec<(u32, Delivery)> = Vec::new();
            let mut rows_out: Vec<(i64, String)> = Vec::new();
            let mut trailing: Vec<String> = Vec::new();
            for item in emitted.items {
                let Some(port) = lane.ports.get_mut(item.port).and_then(Option::as_mut) else {
                    if let Payload::Rows(rows) = item.payload {
                        rows_out.extend(rows.into_iter().map(|r| (pts, r)));
                    }
                    continue;
                };
                let stream = port.out.stream;
                if crate::trace::on() {
                    // A data output's messages: what a node hands a listener
                    // on another lane or process, rows by another name.
                    if let Payload::Message(m) = &item.payload {
                        if !m.data.is_empty() {
                            let text = String::from_utf8_lossy(&m.data);
                            crate::trace::line(format_args!(
                                "emit: {} stream {stream} message at {}: {}",
                                lane.name,
                                m.pts,
                                text.chars().take(160).collect::<String>()
                            ));
                        }
                    }
                    if let Payload::Frame(f) = &item.payload {
                        // A run of samples: 48 kHz or the like, a duration.
                        let base = port.out.base;
                        if base.num == 1 && base.den >= 8_000 && f.duration.is_some() {
                            if let Ok(bytes) = f.data.bytes() {
                                crate::trace::line(format_args!(
                                    "emit: {} stream {stream} sound {}+{} peak {:.3}",
                                    lane.name,
                                    f.pts,
                                    f.duration.unwrap_or(0),
                                    crate::trace::peak(&bytes, "f32")
                                ));
                            }
                        } else if let Some(l) = crate::trace::luma(&f.data) {
                            crate::trace::line(format_args!(
                                "emit: {} stream {stream} picture {} (base {}/{}) luma {l:.1}",
                                lane.name, f.pts, base.num, base.den
                            ));
                        }
                    }
                }
                match item.payload {
                    Payload::Frame(f) => {
                        rows_out.extend(f.rows.iter().cloned().map(|r| (f.pts, r)));
                        deliveries.push((
                            stream,
                            Delivery::Item(Item::Frame(TickFrame {
                                pts: f.pts,
                                duration: f.duration,
                                data: f.data,
                                rows: f.rows,
                            })),
                        ));
                    }
                    Payload::Message(m) if m.data.is_empty() => {
                        port.last = Some(port.last.map_or(m.pts, |l| l.max(m.pts)));
                        deliveries.push((stream, Delivery::Progress(m.pts)));
                    }
                    Payload::Message(m) => {
                        port.last = Some(m.pts);
                        deliveries.push((stream, Delivery::Item(Item::Message(m))));
                    }
                    Payload::Packet(p) => {
                        deliveries.push((stream, Delivery::Item(Item::Packet(p))));
                    }
                    // An older module's rows with no frame to ride leave with
                    // its last call alone, as its world's host had them.
                    Payload::Rows(rows) if windows => {
                        if last {
                            trailing.extend(rows);
                        }
                    }
                    Payload::Rows(rows) => rows_out.extend(rows.into_iter().map(|r| (pts, r))),
                }
            }
            rows_out.extend(emitted.rows.into_iter().map(|r| (pts, r)));
            if crate::trace::on() {
                for (at, row) in &rows_out {
                    crate::trace::line(format_args!(
                        "rows out: {} at {at} (base {}/{}): {}: {}",
                        lane.name,
                        tick_base.num,
                        tick_base.den,
                        match lane.rows.as_ref() {
                            Some(rows) => format!("on stream {}", rows.out.stream),
                            None => "no rows stream; it rides its frame or is dropped".to_string(),
                        },
                        row.chars().take(160).collect::<String>()
                    ));
                }
            }
            if let Some(rows) = lane.rows.as_mut() {
                for (at, row) in rows_out {
                    let at = model::rescale(at, tick_base, rows.out.base);
                    let at = rows.last.map_or(at, |l| l.max(at));
                    rows.last = Some(at);
                    deliveries.push((
                        rows.out.stream,
                        Delivery::Item(Item::Message(Message {
                            pts: at,
                            data: row.into_bytes(),
                        })),
                    ));
                }
            }
            if let Some(end) = end {
                let ports = lane.ports.iter_mut().flatten().chain(lane.rows.iter_mut());
                for port in ports {
                    let at = model::rescale(end, tick_base, port.out.base);
                    let at = model::progress(at, port.out.latency, port.out.base, port.last);
                    if port.sent.is_none_or(|sent| at > sent) {
                        port.sent = Some(at);
                        deliveries.push((port.out.stream, Delivery::Progress(at)));
                    }
                }
            }
            if windows {
                for port in lane.ports.iter().flatten() {
                    deliveries.push((port.out.stream, Delivery::Batch));
                    if last {
                        deliveries.push((port.out.stream, Delivery::Trailing(trailing.clone())));
                    }
                }
            }
            let finishing = emitted.finished && !last && !lane.stopped;
            if last {
                let ports = lane.ports.iter().flatten().chain(lane.rows.iter());
                for port in ports {
                    deliveries.push((port.out.stream, Delivery::End));
                }
                lane.ended = true;
                lane.stopped = true;
            }
            if finishing {
                lane.stopped = true;
                lane.finisher = Some(instance);
                lane.queue.clear();
                lane.withheld = None;
                lane.done.clear();
                lane.next_flush = lane.next_ordinal;
                let streams = lane
                    .bound
                    .iter()
                    .map(|&id| TickStream {
                        id,
                        ..TickStream::default()
                    })
                    .collect();
                let tick = Tick {
                    pts: end.unwrap_or(pts + 1),
                    ordinal: 0,
                    time_base: tick_base,
                    last: true,
                    streams,
                };
                lane.last_cut = true;
                lane.queue.push_back(Task {
                    ordinal: lane.next_ordinal,
                    tick,
                    end: None,
                });
                lane.next_ordinal += 1;
            }
            leaving.extend(deliveries);
        }
        for (stream, delivery) in superseded_progress_dropped(leaving) {
            match delivery {
                Delivery::Item(item) => self.arrive(stream, item)?,
                Delivery::Progress(at) => self.progress(stream, at)?,
                Delivery::End => self.end(stream)?,
                Delivery::Trailing(rows) => self.trailing(stream, &rows),
                Delivery::Batch => {
                    for consumer in self.consumers(stream) {
                        if let Consumer::Writer(w, s) = consumer {
                            self.writers[w].push(s, Out::Batch);
                        }
                    }
                }
            }
        }
        if self.lanes.iter().all(|l| l.ended) {
            self.finished = true;
        }
        Ok(())
    }

    /// Whether every output has stopped taking anything: the run has no
    /// reader left.
    fn unread(&self) -> bool {
        !self.writers.is_empty() && self.writers.iter().all(|w| w.settled())
    }
}

enum Delivery {
    Item(Item),
    Progress(i64),
    End,
    /// An older module's rows with no frame to ride, before its end.
    Trailing(Vec<String>),
    /// The end of what one call of an older module made.
    Batch,
}

/// What several ticks of one lane hand on, less each progress mark that a
/// later one on the same stream says again with nothing between them on
/// that stream: a turn of many small ticks moves its readers' progress once.
fn superseded_progress_dropped(deliveries: Vec<(u32, Delivery)>) -> Vec<(u32, Delivery)> {
    let mut marked: HashSet<u32> = HashSet::new();
    let mut kept: Vec<(u32, Delivery)> = deliveries
        .into_iter()
        .rev()
        .filter(|(stream, delivery)| match delivery {
            Delivery::Progress(_) => marked.insert(*stream),
            _ => {
                marked.remove(stream);
                true
            }
        })
        .collect();
    kept.reverse();
    kept
}

fn out_of(item: &Item) -> Out {
    match item {
        Item::Frame(f) => Out::Frame {
            pts: f.pts,
            data: f.data.clone(),
            rows: f.rows.clone(),
        },
        Item::Message(m) => Out::Message(m.clone()),
        Item::Packet(p) => Out::Packet(p.clone()),
    }
}

/// The worker threads a run gets when `-jobs` is not given. Four covers
/// most of a pure module's frame-parallel speed-up (a compositor rendering
/// 4K frames reached real time on four) without one sidecar taking every
/// core from the ffmpeg processes and encoders beside it.
pub const DEFAULT_WORKERS: usize = 4;

/// The most ticks one turn of a lane that runs a tick at a time takes: a
/// packet of sound's worth on a clock cut a sample a tick.
const TURN: usize = 1024;

/// How many worker threads a run gets: `-jobs` when it was given, else
/// `DEFAULT_WORKERS`, never more than the machine's effective core count.
/// `-jobs 1` is the serial escape hatch.
pub fn worker_count(jobs: Option<usize>) -> usize {
    let cores = thread::available_parallelism().map_or(1, |n| n.get());
    jobs.unwrap_or(DEFAULT_WORKERS).min(cores).max(1)
}

/// Where the scheduler puts what an output's writer calls once it has taken
/// something: the writers are opened before the scheduler, so they call
/// through this.
pub type WakeSlot = OnceLock<Arc<dyn Fn() + Send + Sync>>;

/// What an output's writer calls once it has taken something: room may
/// have opened, or the output may have stopped.
fn waker_of(shared: &Arc<Shared>) -> Arc<dyn Fn() + Send + Sync> {
    let shared = Arc::clone(shared);
    Arc::new(move || {
        let mut state = shared.lock();
        if state.unread() {
            state.finished = true;
        }
        drop(state);
        shared.notify();
    })
}

/// The running network: workers spawned, lanes wired, waiting to be fed.
pub struct Scheduler {
    shared: Arc<Shared>,
    handles: Mutex<Vec<thread::JoinHandle<()>>>,
}

impl Scheduler {
    /// Opens every instance a state lane needs, cuts what the lanes can cut
    /// before anything arrives, puts its waker in `wake`, and spawns
    /// `workers` threads. The waker is in place before any worker runs: a
    /// lane with no inputs runs at once, and a writer that drains what it
    /// made before the waker was there would wake nobody, leaving a worker
    /// that found the output full asleep for good.
    pub fn start(plan: Plan, workers: usize, wake: &WakeSlot) -> Result<Scheduler> {
        let Plan {
            lanes: specs,
            consumers,
            writers,
            mirrors,
        } = plan;
        let mut lanes = Vec::with_capacity(specs.len());
        for mut spec in specs {
            let clock = clock_kind(&spec.shape, &spec.intake);
            let width = if spec.shape.pure
                && matches!(
                    clock,
                    ClockKind::Input | ClockKind::Rate { .. } | ClockKind::Windows
                ) {
                workers
            } else {
                1
            };
            open_every_instance(&mut spec, width)?;
            let opener = if spec.state.is_empty() || width == 1 {
                spec.opener
            } else {
                None
            };
            let width = if opener.is_none() {
                width.min(spec.runners.len()).max(1)
            } else {
                width
            };
            let intake = match spec.intake {
                Intake::Assembled(a) => LaneIntake::Assembled(a),
                Intake::Host { id } => LaneIntake::Host {
                    id,
                    events: VecDeque::new(),
                },
                Intake::Windows { lockstep, pads } => LaneIntake::Windows {
                    lockstep,
                    pads,
                    trailing: Vec::new(),
                },
            };
            let port_state = |out: PortOut| PortState {
                out,
                last: None,
                sent: None,
            };
            lanes.push(Lane {
                name: spec.name,
                intake,
                clock,
                base: spec.tick_base,
                bound: spec.bound,
                gpu: spec.gpu,
                uploaded: HashSet::new(),
                resized: HashSet::new(),
                state: spec.state,
                queue: VecDeque::new(),
                next_ordinal: 0,
                in_flight: 0,
                width,
                created: spec.runners.len(),
                idle: spec.runners.into_iter().enumerate().collect(),
                opener,
                done: BTreeMap::new(),
                next_flush: 0,
                ports: spec.ports.into_iter().map(|p| p.map(port_state)).collect(),
                rows: spec.rows.map(port_state),
                last_cut: false,
                ended: false,
                stopped: false,
                finisher: None,
                started: None,
                turned: None,
                withheld: None,
                unbuffered: spec.unbuffered,
            });
        }
        let mut state = State {
            lanes,
            consumers,
            writers,
            mirrors,
            error: None,
            finished: false,
            cap: workers * 2 + 2,
        };
        for j in 0..state.lanes.len() {
            state.cut(j)?;
        }
        let shared = Arc::new(Shared {
            state: Mutex::new(state),
            work: Condvar::new(),
            space: Condvar::new(),
        });
        let _ = wake.set(waker_of(&shared));
        let handles = (0..workers)
            .map(|k| {
                let shared = Arc::clone(&shared);
                crate::named(format!("worker {k}"), move || worker(&shared))
            })
            .collect();
        Ok(Scheduler {
            shared,
            handles: Mutex::new(handles),
        })
    }

    /// Whether the run has nothing left to make: every lane has ended and
    /// nothing failed.
    pub fn over(&self) -> Arc<dyn Fn() -> bool + Send + Sync> {
        let shared = Arc::clone(&self.shared);
        Arc::new(move || {
            let state = shared.lock();
            state.error.is_none() && state.lanes.iter().all(|l| l.ended)
        })
    }

    /// Waits for room on whatever reads `stream`, then runs `act` on the
    /// state. False once the run has stopped.
    fn feed(&self, stream: u32, act: impl FnOnce(&mut State) -> Result<()>) -> bool {
        let mut state = self.shared.lock();
        let mut traced = false;
        loop {
            if state.error.is_some() || state.finished {
                return false;
            }
            let room = std::iter::once(stream)
                .chain(state.mirrors(stream))
                .all(|s| state.room(s, &state.consumers(s)));
            if room {
                break;
            }
            if !traced && crate::trace::on() {
                traced = true;
                let depths: Vec<String> = state
                    .consumers(stream)
                    .iter()
                    .map(|c| match c {
                        Consumer::Lane(j) => {
                            let lane = &state.lanes[*j];
                            let withheld = if lane.withheld.is_some() {
                                " and a tick withheld"
                            } else {
                                ""
                            };
                            format!("{} queue {}{withheld}", lane.name, lane.queue.len())
                        }
                        Consumer::Writer(w, _) => format!("writer {w} {}", state.writers[*w].len()),
                    })
                    .collect();
                crate::trace::line(format_args!(
                    "room: stream {stream} waits (cap {}): {}",
                    state.cap,
                    depths.join(", ")
                ));
            }
            state = self
                .shared
                .space
                .wait(state)
                .unwrap_or_else(|e| e.into_inner());
        }
        if let Err(e) = act(&mut state) {
            if state.error.is_none() {
                state.error = Some(e);
            }
        }
        drop(state);
        self.shared.notify();
        true
    }

    /// One item read off an input's stream.
    pub fn arrive(&self, stream: u32, item: Item) -> bool {
        self.feed(stream, |state| state.arrive(stream, item))
    }

    pub fn progress(&self, stream: u32, pts: i64) -> bool {
        self.feed(stream, |state| state.progress(stream, pts))
    }

    pub fn end(&self, stream: u32) -> bool {
        self.feed(stream, |state| state.end(stream))
    }

    /// `stream` has ended, leaving `trailing` rows with no frame to ride.
    pub fn end_with(&self, stream: u32, trailing: &[String]) -> bool {
        self.feed(stream, |state| {
            state.trailing(stream, trailing);
            state.end(stream)
        })
    }

    /// A port feed's connection on `stream` starts, told before its frames.
    pub fn source_open(&self, stream: u32, source: SourceInfo) -> bool {
        self.feed(stream, |state| state.source(stream, Some(source)))
    }

    /// A port feed's connection on `stream` has ended.
    pub fn source_close(&self, stream: u32) -> bool {
        self.feed(stream, |state| state.source(stream, None))
    }

    /// Waits until a port feed on `stream` has room for another frame, for
    /// `timeout` at most.
    pub fn wait_room(&self, stream: u32, timeout: Duration) -> Room {
        let mut state = self.shared.lock();
        let until = Instant::now() + timeout;
        loop {
            if state.error.is_some() || state.finished {
                return Room::Stopped;
            }
            if state.hold_room(stream) > 0 {
                return Room::Open;
            }
            let now = Instant::now();
            if now >= until {
                return Room::Full;
            }
            state = self
                .shared
                .space
                .wait_timeout(state, until - now)
                .unwrap_or_else(|e| e.into_inner())
                .0;
        }
    }

    /// Whether the run has stopped: finished, or failed.
    pub fn stopped(&self) -> bool {
        let state = self.shared.lock();
        state.error.is_some() || state.finished
    }

    /// Whether anything reads `stream`.
    pub fn reads(&self, stream: u32) -> bool {
        let state = self.shared.lock();
        // A stream's copies read it too.
        std::iter::once(stream)
            .chain(state.mirrors(stream))
            .any(|id| state.consumers.get(&id).is_some_and(|c| !c.is_empty()))
    }

    /// A failure outside the lanes stops them, unless one failed first.
    pub fn fail(&self, error: anyhow::Error) {
        let mut state = self.shared.lock();
        if state.error.is_none() {
            state.error = Some(error);
        }
        drop(state);
        self.shared.notify();
    }

    /// Waits until every lane has ended or no output takes anything more,
    /// joins the workers, and surfaces the first failure.
    pub fn finish(&self) -> Result<()> {
        {
            let mut state = self.shared.lock();
            while !state.finished && state.error.is_none() {
                if state.unread() {
                    state.finished = true;
                    break;
                }
                state = self
                    .shared
                    .space
                    .wait(state)
                    .unwrap_or_else(|e| e.into_inner());
            }
        }
        self.shared.notify();
        let mut panicked = false;
        let handles = std::mem::take(&mut *self.handles.lock().unwrap_or_else(|e| e.into_inner()));
        for handle in handles {
            panicked |= handle.join().is_err();
        }
        let mut state = self.shared.lock();
        // The failure is taken to be returned; the run stays stopped for a
        // port feed's listener, which is joined after this.
        state.finished = true;
        if let Some(e) = state.error.take() {
            return Err(e);
        }
        for writer in &state.writers {
            if let Some(failure) = writer.failure() {
                bail!("{failure}");
            }
        }
        if panicked {
            bail!("a worker thread panicked");
        }
        Ok(())
    }
}

/// One tick of a clock counted in `base`, in real time.
fn period(base: TimeBase) -> Duration {
    Duration::from_secs_f64(base.num as f64 / base.den.max(1) as f64)
}

fn worker(shared: &Shared) {
    let mut state = shared.lock();
    loop {
        if state.error.is_some() || state.finished {
            return;
        }
        let picked = match state.pick() {
            Ok(picked) => picked,
            Err(e) => {
                state.error = Some(e);
                shared.notify();
                return;
            }
        };
        let Some(i) = picked else {
            state = match state.next_turn() {
                Some(due) => {
                    let wait = due.saturating_duration_since(Instant::now());
                    match shared.work.wait_timeout(state, wait) {
                        Ok((state, _)) => state,
                        Err(e) => e.into_inner().0,
                    }
                }
                None => shared.work.wait(state).unwrap_or_else(|e| e.into_inner()),
            };
            continue;
        };
        let Turn {
            tasks,
            instance,
            runner,
            opener,
        } = match state.dispatch(i) {
            Ok(taken) => taken,
            Err(e) => {
                state.error = Some(e);
                shared.notify();
                return;
            }
        };
        drop(state);
        shared.notify();

        let opened = match runner {
            Some(runner) => Ok(runner),
            None => match opener {
                Some(open) => open(),
                None => Err(anyhow!("a lane ran out of instances")),
            },
        };
        let mut results = Vec::with_capacity(tasks.len());
        let runner = match opened {
            Ok(mut runner) => {
                for Task { ordinal, tick, end } in tasks {
                    let (pts, last) = (tick.pts, tick.last);
                    let started = Instant::now();
                    let result = runner.process(tick);
                    PROFILE.process.add(0, started);
                    // A tick the node says it finished on, or failed on, is
                    // the last of the turn: what was taken after it is not
                    // run, as a lane that stops drops its queue.
                    let stop = result.as_ref().map_or(true, |emitted| emitted.finished);
                    let result = result.map(|emitted| Done {
                        emitted,
                        pts,
                        end,
                        last,
                        instance,
                    });
                    results.push((ordinal, result));
                    if stop {
                        break;
                    }
                }
                Some(runner)
            }
            Err(e) => {
                results.push((tasks[0].ordinal, Err(e)));
                None
            }
        };

        state = shared.lock();
        state.complete(i, results, instance, runner);
        shared.notify();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ffrwd_wasm_runtime::node::{
        Accepts, BoundStream, Clock, InputPort, Pairing, PortKind, Rational, RowsUse, StreamFormat,
    };

    #[test]
    fn without_jobs_a_run_gets_four_workers_or_every_core_if_fewer() {
        let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
        assert_eq!(worker_count(None), DEFAULT_WORKERS.min(cores));
        assert_eq!(worker_count(Some(1)), 1);
        assert_eq!(worker_count(Some(usize::MAX)), cores);
    }
    use ffrwd_wasm_runtime::runtime::{AudioFormat, StreamInfo, VideoFormat};
    use std::sync::{mpsc, OnceLock};

    /// What a publisher's calls brought: frames per call, and whether each
    /// was the last.
    type Calls = Arc<Mutex<Vec<(usize, bool)>>>;

    struct Publisher {
        shape: NodeShape,
        calls: Calls,
    }

    impl Node for Publisher {
        fn name(&self) -> &str {
            "publish"
        }

        fn shape(&self) -> &NodeShape {
            &self.shape
        }

        fn set_params(&mut self, _params: &str) -> Result<()> {
            Ok(())
        }

        fn process(&mut self, tick: Tick) -> Result<Emitted> {
            let frames = tick.streams.iter().map(|s| s.frames.len()).sum();
            self.calls.lock().unwrap().push((frames, tick.last));
            Ok(Emitted::default())
        }
    }

    #[test]
    fn a_rate_clock_on_arrivals_takes_its_last_call_once_they_end_whatever_its_count() {
        let shape = NodeShape {
            inputs: vec![InputPort {
                name: "v".to_string(),
                kind: PortKind::Video,
                required: true,
                many: false,
                pairing: Pairing::Arrival,
                rows: RowsUse::Ignore,
                window: 1,
                stride: 1,
                accepts: Accepts::default(),
                schema: None,
            }],
            outputs: Vec::new(),
            clock: Clock::Rate(Rational { num: 50, den: 1 }),
            pure: false,
            one_to_one: false,
            bounded: true,
            relation: Vec::new(),
        };
        let tenths = TimeBase { num: 1, den: 10 };
        let bound = BoundStream {
            port: "v".to_string(),
            id: 0,
            info: StreamInfo::default(),
            time_base: tenths,
            format: StreamFormat::Video(VideoFormat {
                width: 2,
                height: 2,
                pix_fmt: "rgba",
                frame_len: 16,
                color: None,
            }),
            rendition: Default::default(),
            row: None,
            decode_delay: 0,
            latency: None,
            hint: Default::default(),
        };
        let assembler = Assembler::new(&shape, &[bound], "publish", &[]).expect("an assembler");
        let calls = Calls::default();
        let publisher = Publisher {
            shape: shape.clone(),
            calls: Arc::clone(&calls),
        };
        let plan = Plan {
            lanes: vec![LaneSpec {
                gpu: HashMap::new(),
                name: "publish".to_string(),
                shape,
                intake: Intake::Assembled(Box::new(assembler)),
                bound: vec![0],
                state: Vec::new(),
                tick_base: TimeBase { num: 1, den: 50 },
                runners: vec![Box::new(publisher)],
                opener: None,
                ports: Vec::new(),
                rows: None,
                unbuffered: false,
            }],
            consumers: HashMap::from([(0, vec![Consumer::Lane(0)])]),
            writers: Vec::new(),
            mirrors: HashMap::new(),
        };
        let scheduler = Scheduler::start(plan, 2, &WakeSlot::new()).expect("the lanes");
        let epoch = 17_909_860_700i64;
        for k in 0..30 {
            let frame = TickFrame {
                pts: epoch + k,
                duration: None,
                data: Arc::new(vec![0; 16]).into(),
                rows: Vec::new(),
            };
            assert!(scheduler.arrive(0, Item::Frame(frame)));
        }
        assert!(scheduler.end(0));
        let ended = Instant::now();
        let (done, finished) = mpsc::channel();
        thread::spawn(move || done.send(scheduler.finish()));
        finished
            .recv_timeout(Duration::from_secs(5))
            .expect("the lane ended with its input")
            .expect("the run");
        assert!(ended.elapsed() < Duration::from_secs(1));
        let calls = calls.lock().unwrap();
        let frames: usize = calls.iter().map(|c| c.0).sum();
        assert_eq!(frames, 30, "every arrival was handed");
        assert_eq!(
            calls.last(),
            Some(&(0, true)),
            "the last call follows the tick that took what was left"
        );
        assert_eq!(calls.iter().filter(|c| c.1).count(), 1);
        assert!(
            calls.len() <= 32,
            "a turn per arrival at most, the first and the last"
        );
    }

    /// A source of `count` messages, one a call, finished on the last.
    struct Counting {
        shape: NodeShape,
        made: i64,
        count: i64,
    }

    impl Node for Counting {
        fn name(&self) -> &str {
            "counting"
        }

        fn shape(&self) -> &NodeShape {
            &self.shape
        }

        fn set_params(&mut self, _params: &str) -> Result<()> {
            Ok(())
        }

        fn process(&mut self, tick: Tick) -> Result<Emitted> {
            if tick.last {
                return Ok(Emitted::default());
            }
            self.made += 1;
            Ok(Emitted {
                items: vec![model::Emission {
                    port: 0,
                    payload: Payload::Message(Message {
                        pts: self.made,
                        data: b"{}".to_vec(),
                    }),
                }],
                finished: self.made >= self.count,
                ..Emitted::default()
            })
        }
    }

    /// A source lane starts as the scheduler does and fills its output's
    /// queue at once, and the writer drains it while the thread that started
    /// the scheduler has not got further yet. The waker the writer calls is
    /// in place by then, so the worker that found the queue full is woken
    /// and the source runs to its end. Were the waker put in place only
    /// where the starting thread gets to after `start`, the writer would
    /// wake nobody and the run would never end.
    #[test]
    fn a_source_finishes_when_its_writer_drains_before_the_starting_thread_goes_on() {
        let shape = NodeShape {
            inputs: Vec::new(),
            outputs: Vec::new(),
            clock: Clock::SelfClocked,
            pure: false,
            one_to_one: false,
            bounded: true,
            relation: Vec::new(),
        };
        let assembler = Assembler::new(&shape, &[], "counting", &[]).expect("an assembler");
        let slot = Arc::new(WakeSlot::new());
        let wake: Arc<dyn Fn() + Send + Sync> = {
            let slot = Arc::clone(&slot);
            Arc::new(move || {
                if let Some(wake) = slot.get() {
                    wake();
                }
            })
        };
        let dir = std::env::temp_dir().join(format!("ffrwd-lanes-waker-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("a scratch directory");
        let mut writer = crate::edges::spawn_writer(
            crate::OutputPath::File(dir.join("out.ndjson").display().to_string()),
            "the test's output".to_string(),
            crate::edges::Target::Null,
            wake,
            Arc::new(crate::edges::Ending::default()),
        );
        let base = TimeBase { num: 1, den: 1000 };
        let plan = Plan {
            lanes: vec![LaneSpec {
                gpu: HashMap::new(),
                name: "counting".to_string(),
                shape: shape.clone(),
                intake: Intake::Assembled(Box::new(assembler)),
                bound: Vec::new(),
                state: Vec::new(),
                tick_base: base,
                runners: vec![Box::new(Counting {
                    shape,
                    made: 0,
                    count: 40,
                })],
                opener: None,
                ports: vec![Some(PortOut {
                    stream: 0,
                    base,
                    latency: 0.0,
                })],
                rows: None,
                unbuffered: false,
            }],
            consumers: HashMap::from([(0, vec![Consumer::Writer(0, 0)])]),
            writers: vec![Arc::clone(&writer.queue)],
            mirrors: HashMap::new(),
        };
        let scheduler = Arc::new(Scheduler::start(plan, 1, &slot).expect("the lanes"));
        // The starting thread held back where it used to put the waker in
        // place: by now the source has filled the queue and the writer has
        // drained it.
        thread::sleep(Duration::from_millis(300));
        let _ = slot.set(waker_of(&scheduler.shared));
        let (done, finished) = mpsc::channel();
        {
            let scheduler = Arc::clone(&scheduler);
            thread::spawn(move || done.send(scheduler.finish()));
        }
        finished
            .recv_timeout(Duration::from_secs(5))
            .expect("the source ran to its end")
            .expect("the run");
        writer.join();
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Opens when told to, for a node to wait on in its first call.
    #[derive(Default)]
    struct Gate {
        open: Mutex<bool>,
        opened: Condvar,
    }

    impl Gate {
        fn wait(&self) {
            let mut open = self.open.lock().unwrap();
            while !*open {
                open = self.opened.wait(open).unwrap();
            }
        }

        fn open(&self) {
            *self.open.lock().unwrap() = true;
            self.opened.notify_all();
        }
    }

    /// A node that notes the pts of each tick and, at each call, how long
    /// its lane's queue is; it waits on `gate` in its first call.
    struct Noting {
        shape: NodeShape,
        seen: Arc<Mutex<Vec<i64>>>,
        queues: Arc<Mutex<Vec<usize>>>,
        gate: Arc<Gate>,
        shared: Arc<OnceLock<Arc<Shared>>>,
    }

    impl Node for Noting {
        fn name(&self) -> &str {
            "noting"
        }

        fn shape(&self) -> &NodeShape {
            &self.shape
        }

        fn set_params(&mut self, _params: &str) -> Result<()> {
            Ok(())
        }

        fn process(&mut self, tick: Tick) -> Result<Emitted> {
            if let Some(shared) = self.shared.get() {
                let queue = shared.lock().lanes[0].queue.len();
                self.queues.lock().unwrap().push(queue);
            }
            let first = {
                let mut seen = self.seen.lock().unwrap();
                if !tick.last {
                    seen.push(tick.pts);
                }
                seen.len() == 1
            };
            if first {
                self.gate.wait();
            }
            Ok(Emitted::default())
        }
    }

    /// One lane running `node`, clocked by its one input `port` on stream
    /// 0, and the scheduler running it on one worker (a cap of 4).
    fn one_lane(shape: NodeShape, bound: BoundStream, node: Noting) -> Scheduler {
        let assembler = Assembler::new(&shape, &[bound], "noting", &[]).expect("an assembler");
        let plan = Plan {
            lanes: vec![LaneSpec {
                gpu: HashMap::new(),
                name: "noting".to_string(),
                shape,
                intake: Intake::Assembled(Box::new(assembler)),
                bound: vec![0],
                state: Vec::new(),
                tick_base: TimeBase {
                    num: 1,
                    den: 48_000,
                },
                runners: vec![Box::new(node)],
                opener: None,
                ports: Vec::new(),
                rows: None,
                unbuffered: false,
            }],
            consumers: HashMap::from([(0, vec![Consumer::Lane(0)])]),
            writers: Vec::new(),
            mirrors: HashMap::new(),
        };
        Scheduler::start(plan, 1, &WakeSlot::new()).expect("the lanes")
    }

    fn wait_until(what: &str, done: impl Fn() -> bool) {
        let until = Instant::now() + Duration::from_secs(5);
        while !done() {
            assert!(Instant::now() < until, "{what}");
            thread::sleep(Duration::from_millis(5));
        }
    }

    fn input_port(name: &str, kind: PortKind, pairing: Pairing) -> InputPort {
        InputPort {
            name: name.to_string(),
            kind,
            required: true,
            many: false,
            pairing,
            rows: RowsUse::Ignore,
            window: 1,
            stride: 1,
            accepts: Accepts::default(),
            schema: None,
        }
    }

    /// A node clocked by its sound a sample a tick (a listener written for
    /// pictures, bound to sound) is cut 1024 ticks by one packet of 1024
    /// samples. The cut stops at the cap, the reader waits while ticks are
    /// withheld, every tick still runs once and in order, and the ticks
    /// queued behind the first are taken in one turn.
    #[test]
    fn an_audio_clock_a_sample_a_tick_queues_no_more_than_the_cap_and_runs_every_tick() {
        let shape = NodeShape {
            inputs: vec![input_port("a", PortKind::Audio, Pairing::Lockstep)],
            outputs: Vec::new(),
            clock: Clock::Input("a".to_string()),
            pure: false,
            one_to_one: false,
            bounded: true,
            relation: Vec::new(),
        };
        let samples = TimeBase {
            num: 1,
            den: 48_000,
        };
        let bound = BoundStream {
            port: "a".to_string(),
            id: 0,
            info: StreamInfo::default(),
            time_base: samples,
            format: StreamFormat::Audio(AudioFormat {
                sample_rate: 48_000,
                channels: 1,
                sample_fmt: "f32",
                channel_layout: None,
            }),
            rendition: Default::default(),
            row: None,
            decode_delay: 0,
            latency: None,
            hint: Default::default(),
        };
        let seen = Arc::new(Mutex::new(Vec::new()));
        let queues = Arc::new(Mutex::new(Vec::new()));
        let gate = Arc::new(Gate::default());
        let shared = Arc::new(OnceLock::new());
        let node = Noting {
            shape: shape.clone(),
            seen: Arc::clone(&seen),
            queues: Arc::clone(&queues),
            gate: Arc::clone(&gate),
            shared: Arc::clone(&shared),
        };
        let scheduler = one_lane(shape, bound, node);
        let _ = shared.set(Arc::clone(&scheduler.shared));
        let packet = |pts: i64| {
            Item::Frame(TickFrame {
                pts,
                duration: Some(1024),
                data: Arc::new(vec![0u8; 1024 * 4]).into(),
                rows: Vec::new(),
            })
        };
        assert!(scheduler.arrive(0, packet(0)));
        wait_until("the first tick ran", || !seen.lock().unwrap().is_empty());
        // The first packet's ticks went in one turn; the second's wait, and
        // the reader goes on until the cap's worth of packets waits.
        assert!(scheduler.arrive(0, packet(1024)));
        {
            let state = scheduler.shared.lock();
            let lane = &state.lanes[0];
            assert_eq!(
                lane.queue.len(),
                state.cap,
                "the queue holds the cap's worth of the second packet's ticks"
            );
            assert!(
                lane.withheld.is_some(),
                "the rest of the packet's ticks wait"
            );
            assert!(
                state.room(0, &state.consumers(0)),
                "one packet behind, the reader goes on"
            );
        }
        for k in 2..5 {
            assert!(scheduler.arrive(0, packet(k * 1024)));
        }
        {
            let state = scheduler.shared.lock();
            assert!(
                state.lanes[0].queue.len() <= state.cap,
                "the queue holds no more than the cap"
            );
            assert!(
                !state.room(0, &state.consumers(0)),
                "the reader waits once the cap's worth of packets waits"
            );
        }
        gate.open();
        wait_until("every sample's tick ran", || {
            seen.lock().unwrap().len() == 5 * 1024
        });
        assert!(scheduler.end(0));
        let (done, finished) = mpsc::channel();
        let shared_queues = Arc::clone(&queues);
        thread::spawn(move || done.send(scheduler.finish()));
        finished
            .recv_timeout(Duration::from_secs(5))
            .expect("the lane ended with its input")
            .expect("the run");
        assert_eq!(*seen.lock().unwrap(), (0..5 * 1024).collect::<Vec<i64>>());
        let queues = shared_queues.lock().unwrap();
        assert!(
            queues[1..1024].iter().all(|q| *q == 4),
            "the first packet's ticks ran in one turn, the second's queued behind it: {:?}",
            &queues[..12]
        );
        assert!(
            queues.iter().all(|q| *q <= 4),
            "no call saw more than the cap"
        );
    }

    /// A self-clocked reader (a decoder) slower than what reaches it holds
    /// its writer back once the cap's worth of arrivals waits on its input,
    /// whatever else the scheduler bounds.
    #[test]
    fn a_self_clocked_reader_holds_its_writer_at_the_cap() {
        let shape = NodeShape {
            inputs: vec![input_port("v", PortKind::Video, Pairing::Arrival)],
            outputs: Vec::new(),
            clock: Clock::SelfClocked,
            pure: false,
            one_to_one: false,
            bounded: true,
            relation: Vec::new(),
        };
        let bound = BoundStream {
            port: "v".to_string(),
            id: 0,
            info: StreamInfo::default(),
            time_base: TimeBase { num: 1, den: 30 },
            format: StreamFormat::Video(VideoFormat {
                width: 2,
                height: 2,
                pix_fmt: "rgba",
                frame_len: 16,
                color: None,
            }),
            rendition: Default::default(),
            row: None,
            decode_delay: 0,
            latency: None,
            hint: Default::default(),
        };
        let seen = Arc::new(Mutex::new(Vec::new()));
        let gate = Arc::new(Gate::default());
        let node = Noting {
            shape: shape.clone(),
            seen: Arc::clone(&seen),
            queues: Arc::default(),
            gate: Arc::clone(&gate),
            shared: Arc::default(),
        };
        let scheduler = Arc::new(one_lane(shape, bound, node));
        let delivered = Arc::new(Mutex::new(0usize));
        let writer = {
            let scheduler = Arc::clone(&scheduler);
            let delivered = Arc::clone(&delivered);
            thread::spawn(move || {
                for k in 0..40 {
                    let frame = TickFrame {
                        pts: k,
                        duration: None,
                        data: Arc::new(vec![0; 16]).into(),
                        rows: Vec::new(),
                    };
                    if !scheduler.arrive(0, Item::Frame(frame)) {
                        return;
                    }
                    *delivered.lock().unwrap() += 1;
                }
                scheduler.end(0);
            })
        };
        wait_until("the first call started", || {
            !seen.lock().unwrap().is_empty()
        });
        thread::sleep(Duration::from_millis(200));
        let held = *delivered.lock().unwrap();
        assert!(
            held <= 4 + 1,
            "the writer waits once the cap's worth is queued ({held} delivered)"
        );
        gate.open();
        writer.join().expect("the writer");
        scheduler.finish().expect("the run");
        assert_eq!(*delivered.lock().unwrap(), 40);
    }

    #[test]
    fn a_progress_mark_a_later_one_says_again_is_dropped() {
        let marks = |d: &[(u32, Delivery)]| -> Vec<String> {
            d.iter()
                .map(|(s, d)| match d {
                    Delivery::Progress(at) => format!("{s}:p{at}"),
                    Delivery::End => format!("{s}:end"),
                    _ => format!("{s}:other"),
                })
                .collect()
        };
        let message = || {
            Delivery::Item(Item::Message(Message {
                pts: 0,
                data: Vec::new(),
            }))
        };
        let kept = superseded_progress_dropped(vec![
            (1, Delivery::Progress(1)),
            (1, Delivery::Progress(2)),
            (2, Delivery::Progress(2)),
            (1, message()),
            (1, Delivery::Progress(3)),
            (2, Delivery::Progress(3)),
            (1, Delivery::Progress(4)),
            (1, Delivery::End),
        ]);
        assert_eq!(
            marks(&kept),
            ["1:p2", "1:other", "2:p3", "1:p4", "1:end"],
            "each stream's marks up to the next thing on it, or the end, are the last of them"
        );
    }
}
