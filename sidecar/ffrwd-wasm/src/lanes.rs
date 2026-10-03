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

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context, Result};
use ffrwd_wasm_runtime::node::{
    self as model, Emitted, Node, NodeShape, Payload, Tick, TickFrame, TickStream,
};
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
    /// The stream ids whose rows the node keeps as state.
    pub state: Vec<u32>,
    pub tick_base: TimeBase,
    pub runners: Vec<Box<dyn Node>>,
    pub opener: Option<Opener>,
    pub ports: Vec<Option<PortOut>>,
    pub rows: Option<PortOut>,
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
    fn room(&self, consumers: &[Consumer]) -> bool {
        consumers.iter().all(|c| match c {
            Consumer::Lane(j) => {
                let lane = &self.lanes[*j];
                lane.stopped || lane.queue.len() < self.cap
            }
            Consumer::Writer(w, _) => self.writers[*w].len() < self.cap,
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
                    match &mut lane.intake {
                        LaneIntake::Assembled(a) => a.arrive(stream, item.clone())?,
                        LaneIntake::Host { events, .. } => {
                            if let Item::Message(m) = &item {
                                events.push_back(HostEvent::Message(m.clone()));
                            }
                        }
                        LaneIntake::Windows { lockstep, pads, .. } => {
                            let Item::Frame(f) = &item else { continue };
                            let frame = Frame {
                                pts: f.pts,
                                data: Arc::clone(&f.data),
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
                let mut cut = Vec::new();
                while let Some(tick) = a.next_clocked()? {
                    let last = tick.last;
                    cut.push((tick, a.tick_end()));
                    if last {
                        break;
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
            if let Some(consumers) = self.consumers.get(&port.out.stream) {
                if !self.room(consumers) {
                    return false;
                }
                // A source has nothing upstream to wait on, so a hold input it
                // feeds holds it back by its own room, as a socket does.
                if source
                    && consumers
                        .iter()
                        .any(|c| self.held_full(*c, port.out.stream))
                {
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

    /// A rate clock's ticks are cut as its queue has room, and a
    /// self-clocked node's as it returns, so both are cut here, where a
    /// worker looks for work, as well as on arrivals.
    fn pick(&mut self) -> Result<Option<usize>> {
        for j in 0..self.lanes.len() {
            if matches!(
                self.lanes[j].clock,
                ClockKind::Rate { .. } | ClockKind::SelfClocked
            ) {
                self.cut(j)?;
            }
        }
        Ok((0..self.lanes.len())
            .filter(|i| self.dispatchable(*i))
            .max_by_key(|i| self.lanes[*i].queue.len()))
    }

    /// Takes lane `i`'s front task with an instance, or the means to open
    /// one, and hands a state input its earlier rows.
    fn dispatch(&mut self, i: usize) -> (Task, usize, Option<Box<dyn Node>>, Option<Opener>) {
        let lane = &mut self.lanes[i];
        let mut task = lane.queue.pop_front().expect("picked lanes have a task");
        task.tick.ordinal = task.ordinal;
        lane.in_flight += 1;
        let wanted = match (task.tick.last, lane.finisher) {
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
        (task, instance, runner, opener)
    }

    fn complete(
        &mut self,
        i: usize,
        ordinal: u64,
        instance: usize,
        runner: Option<Box<dyn Node>>,
        done: Result<Done>,
    ) {
        let lane = &mut self.lanes[i];
        lane.in_flight -= 1;
        let last = done.as_ref().is_ok_and(|d| d.last);
        if let Some(runner) = runner {
            if !last {
                lane.idle.push((instance, runner));
            }
        }
        let flushed = match done {
            Ok(done) => {
                if ordinal >= lane.next_flush {
                    lane.done.insert(ordinal, done);
                }
                self.flush(i)
            }
            Err(e) => Err(e.context(format!("in {}", self.lanes[i].name))),
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
            for (stream, delivery) in deliveries {
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

fn out_of(item: &Item) -> Out {
    match item {
        Item::Frame(f) => Out::Frame {
            pts: f.pts,
            data: Arc::clone(&f.data),
            rows: f.rows.clone(),
        },
        Item::Message(m) => Out::Message(m.clone()),
        Item::Packet(p) => Out::Packet(p.clone()),
    }
}

/// How many worker threads a run gets: the machine's effective core count,
/// capped by `-jobs` when it was given. `-jobs 1` is the serial escape hatch.
pub fn worker_count(jobs: Option<usize>) -> usize {
    let cores = thread::available_parallelism().map_or(1, |n| n.get());
    jobs.unwrap_or(cores).min(cores).max(1)
}

/// The running network: workers spawned, lanes wired, waiting to be fed.
pub struct Scheduler {
    shared: Arc<Shared>,
    handles: Mutex<Vec<thread::JoinHandle<()>>>,
}

impl Scheduler {
    /// Opens every instance a state lane needs, cuts what the lanes can cut
    /// before anything arrives, and spawns `workers` threads.
    pub fn start(plan: Plan, workers: usize) -> Result<Scheduler> {
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
        let handles = (0..workers)
            .map(|_| {
                let shared = Arc::clone(&shared);
                thread::spawn(move || worker(&shared))
            })
            .collect();
        Ok(Scheduler {
            shared,
            handles: Mutex::new(handles),
        })
    }

    /// What an output's writer calls once it has taken something: room may
    /// have opened, or the output may have stopped.
    pub fn waker(&self) -> Arc<dyn Fn() + Send + Sync> {
        let shared = Arc::clone(&self.shared);
        Arc::new(move || {
            let mut state = shared.lock();
            if state.unread() {
                state.finished = true;
            }
            drop(state);
            shared.notify();
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
        loop {
            if state.error.is_some() || state.finished {
                return false;
            }
            let mut consumers = state.consumers(stream);
            for mirror in state.mirrors(stream) {
                consumers.extend(state.consumers(mirror));
            }
            if state.room(&consumers) {
                break;
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
        self.shared
            .lock()
            .consumers
            .get(&stream)
            .is_some_and(|c| !c.is_empty())
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
        let (task, instance, runner, opener) = state.dispatch(i);
        drop(state);
        shared.notify();

        let opened = match runner {
            Some(runner) => Ok(runner),
            None => match opener {
                Some(open) => open(),
                None => Err(anyhow!("a lane ran out of instances")),
            },
        };
        let Task { ordinal, tick, end } = task;
        let (pts, last) = (tick.pts, tick.last);
        let (runner, done) = match opened {
            Ok(mut runner) => {
                let result = runner.process(tick).map(|emitted| Done {
                    emitted,
                    pts,
                    end,
                    last,
                    instance,
                });
                (Some(runner), result)
            }
            Err(e) => (None, Err(e)),
        };

        state = shared.lock();
        state.complete(i, ordinal, instance, runner, done);
        shared.notify();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ffrwd_wasm_runtime::node::{
        Accepts, BoundStream, Clock, InputPort, Pairing, PortKind, Rational, RowsUse, StreamFormat,
    };
    use ffrwd_wasm_runtime::runtime::{StreamInfo, VideoFormat};
    use std::sync::mpsc;

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
            }],
            consumers: HashMap::from([(0, vec![Consumer::Lane(0)])]),
            writers: Vec::new(),
            mirrors: HashMap::new(),
        };
        let scheduler = Scheduler::start(plan, 2).expect("the lanes");
        let epoch = 17_909_860_700i64;
        for k in 0..30 {
            let frame = TickFrame {
                pts: epoch + k,
                duration: None,
                data: Arc::new(vec![0; 16]),
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
}
