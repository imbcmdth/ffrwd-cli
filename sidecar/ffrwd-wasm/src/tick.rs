//! Tick assembly: what one call of a node is handed, cut out of whatever has
//! arrived on its inputs, as the node's shape pairs each one with the clock.
//!
//! - [`Assembler`] is the node world's rules: the clock's window and stride,
//!   lockstep inputs (the clock's interval, so audio under a video clock is
//!   re-cut to each tick, 1,601 or 1,602 samples at 29.97), hold inputs as
//!   the newest frame at or before the tick, interval inputs settled on
//!   their producer's progress, `ahead` and `latency`, and arrival.
//! - [`Lockstep`] is the same thing for a lane of an older frame module: a
//!   clock pad with its window and further pads at the clock's exact pts,
//!   rows riding pad 0 alone.
//! - [`DataDrive`] is the adapted 0.17 data filter's: messages at or before
//!   the clock, a pad read from a file waited for.
//! - [`EarlierRows`] is what a state input's instance is owed of the ticks
//!   it did not process.
//!
//! Pairing is resolved here, once, before a tick goes to any instance, so
//! every instance of a node sees the same ticks.

use std::cmp::Ordering;
use std::collections::{BTreeMap, VecDeque};
use std::sync::Arc;

use anyhow::{anyhow, bail, Result};
use ffrwd_wasm_runtime::node::{
    Anchor, BoundStream, Clock, Feed, Hold, InputPort, Interval, NodeShape, Pairing, PortKind,
    RowsUse, StreamFormat, Tick, TickFrame, TickStream, TimedRows,
};
use ffrwd_wasm_runtime::runtime::{AudioFormat, Format, Frame, Message, Packet, Shape, TimeBase};

use crate::heartbeat;
use crate::hold::{self, Grid, Group, Handed, Member, Report, SourceInfo};
use crate::windows::Windows;

/// Whether `a` in `base_a` is before, at or after `b` in `base_b`, exactly.
pub fn compare(a: i64, base_a: TimeBase, b: i64, base_b: TimeBase) -> Ordering {
    let left = i128::from(a) * i128::from(base_a.num) * i128::from(base_b.den);
    let right = i128::from(b) * i128::from(base_b.num) * i128::from(base_a.den);
    left.cmp(&right)
}

/// `seconds` added to `pts`, in `base`, rounded up.
fn plus_seconds(pts: i64, base: TimeBase, seconds: f64) -> i64 {
    let ticks = (seconds * base.den as f64 / base.num.max(1) as f64).ceil() as i64;
    pts.saturating_add(ticks)
}

/// A lane of an older frame module: the clock pad cut into windows, every
/// further pad one frame at the clock's exact timestamp.
pub struct Lockstep {
    pads: Vec<VecDeque<Frame>>,
    ended: Vec<bool>,
    windows: Windows,
}

impl Lockstep {
    pub fn new(shape: Shape, format: &Format, pads: usize) -> Lockstep {
        Lockstep {
            pads: (0..pads).map(|_| VecDeque::new()).collect(),
            ended: vec![false; pads],
            windows: Windows::new(shape, format),
        }
    }

    /// Frames arriving on one pad, and every call they complete, in order.
    pub fn push(&mut self, pad: usize, frames: &[Frame], name: &str) -> Result<Vec<Vec<Frame>>> {
        let mut calls = Vec::new();
        if self.pads.len() == 1 {
            for frame in frames {
                calls.extend(self.windows.push(frame.clone(), name)?);
            }
            return Ok(calls);
        }
        self.pads[pad].extend(frames.iter().cloned());
        while self.pads.iter().all(|q| !q.is_empty()) {
            calls.push(self.take(name)?);
        }
        Ok(calls)
    }

    /// One pad has ended. Once every one has, the final call's frames: the
    /// tail the strides left over.
    pub fn end(&mut self, pad: usize, name: &str) -> Result<Option<Vec<Frame>>> {
        self.ended[pad] = true;
        if !self.ended.iter().all(|e| *e) {
            return Ok(None);
        }
        for (pad, queue) in self.pads.iter().enumerate() {
            if !queue.is_empty() {
                bail!(
                    "{name}: pad {pad} ends with {} frame(s) that never paired with the other pads; \
                     a module reading several streams reads them in lockstep",
                    queue.len()
                );
            }
        }
        Ok(Some(self.windows.tail()))
    }

    /// One frame off every pad, at one timestamp. Rows ride pad 0; what
    /// arrived on any other pad is dropped here.
    fn take(&mut self, name: &str) -> Result<Vec<Frame>> {
        let head = self.pads[0]
            .front()
            .map(|f| f.pts)
            .ok_or_else(|| anyhow!("{name}: a queue emptied mid-window"))?;
        for (pad, queue) in self.pads.iter().enumerate().skip(1) {
            let other = queue.front().map(|f| f.pts).unwrap_or(head);
            if other != head {
                bail!(
                    "{name}: pad 0 is at pts {head} and pad {pad} at pts {other}; a module reading \
                     several streams reads them in lockstep, one frame per pad at one timestamp"
                );
            }
        }
        let mut call = Vec::with_capacity(self.pads.len());
        for (pad, queue) in self.pads.iter_mut().enumerate() {
            let mut frame = queue.pop_front().expect("every head matched pts");
            if pad > 0 {
                frame.rows.clear();
            }
            call.push(frame);
        }
        Ok(call)
    }
}

/// One call an adapted data filter makes.
pub struct DataCall {
    /// One list per pad, in pad order; empty for a clock pad.
    pub input: Vec<Vec<Message>>,
    pub now: Option<i64>,
    pub last: bool,
}

/// What an adapted data filter holds between batches: the messages that
/// have arrived and not yet been handed over, the clock frames not yet
/// called for, and where the clock is.
pub struct DataDrive {
    /// Every pad's time base, and whether it is a data pad.
    pads: Vec<(TimeBase, bool)>,
    /// The data pads read from a regular file. Their messages are all there
    /// to be read, so the clock waits for them rather than racing them.
    settled: Vec<bool>,
    /// The first clock pad, which drives the calls; None without one.
    clock: Option<usize>,
    /// Messages arrived and not yet handed over, per pad, in arrival order.
    held: Vec<VecDeque<Message>>,
    /// The pts of the last packet each pad has delivered, heartbeats
    /// included: how far it has been read.
    read_to: Vec<Option<i64>>,
    ended: Vec<bool>,
    /// The first clock pad's frames not yet called for, by pts.
    ticks: VecDeque<i64>,
    /// The first clock pad's latest pts, which is `now`.
    now: Option<i64>,
}

impl DataDrive {
    pub fn new(pads: Vec<(TimeBase, bool)>, settled: Vec<bool>, clock: Option<usize>) -> DataDrive {
        let count = pads.len();
        DataDrive {
            pads,
            settled,
            clock,
            held: (0..count).map(|_| VecDeque::new()).collect(),
            read_to: vec![None; count],
            ended: vec![false; count],
            ticks: VecDeque::new(),
            now: None,
        }
    }

    /// Takes one batch off the queues and answers the calls it makes, in
    /// order. `ended` says which pads have ended by this batch, and `last`
    /// that every one has, in which case the final call is among those
    /// answered.
    pub fn arrive(
        &mut self,
        carried: Vec<Vec<Packet>>,
        ended: &[bool],
        last: bool,
    ) -> Vec<DataCall> {
        for (pad, packets) in carried.into_iter().enumerate() {
            if self.pads[pad].1 {
                if let Some(packet) = packets.last() {
                    self.read_to[pad] = Some(packet.pts);
                }
                self.held[pad].extend(
                    packets
                        .into_iter()
                        .filter(|p| !heartbeat::is_heartbeat(&p.data))
                        .map(|p| Message {
                            pts: p.pts,
                            data: p.data,
                        }),
                );
            } else if Some(pad) == self.clock {
                self.ticks.extend(packets.into_iter().map(|p| p.pts));
            }
        }
        self.ended.copy_from_slice(ended);

        let mut calls = Vec::new();
        while let Some(&pts) = self.ticks.front() {
            // A call is made as the clock advances; a frame at or behind the
            // last one moves nothing.
            if self.now.is_some_and(|now| pts <= now) {
                self.ticks.pop_front();
                continue;
            }
            if !self.caught_up(pts) {
                break;
            }
            self.ticks.pop_front();
            self.now = Some(pts);
            let input = self.release(Some(pts));
            calls.push(DataCall {
                input,
                now: Some(pts),
                last: false,
            });
        }
        // Once the clock has ended, nothing is held for it any more.
        let clock_done = self
            .clock
            .is_none_or(|clock| self.ended[clock] && self.ticks.is_empty());
        if clock_done {
            let input = self.release(None);
            if input.iter().any(|m| !m.is_empty()) {
                calls.push(DataCall {
                    input,
                    now: self.now,
                    last: false,
                });
            }
        }
        if last {
            // The final call rides the last one this batch made, or is one
            // of its own when the batch made none.
            match calls.last_mut() {
                Some(call) => call.last = true,
                None => calls.push(DataCall {
                    input: self.release(None),
                    now: self.now,
                    last: true,
                }),
            }
        }
        calls
    }

    /// Whether every data pad read from a file has been read past `now` on
    /// the first clock pad's clock, or has ended.
    fn caught_up(&self, now: i64) -> bool {
        let Some(clock) = self.clock.map(|clock| self.pads[clock].0) else {
            return true;
        };
        (0..self.pads.len())
            .filter(|pad| self.settled[*pad] && !self.ended[*pad])
            .all(|pad| {
                self.read_to[pad]
                    .is_some_and(|pts| compare(pts, self.pads[pad].0, now, clock).is_gt())
            })
    }

    /// The held messages up to `now` on the first clock pad's clock, one
    /// list per pad, or all of them without one. A pad's messages leave in
    /// the order they arrived, so one ahead of the clock holds back those
    /// behind it too.
    fn release(&mut self, now: Option<i64>) -> Vec<Vec<Message>> {
        let clock_base = self.clock.map(|clock| self.pads[clock].0);
        let mut released: Vec<Vec<Message>> = Vec::with_capacity(self.held.len());
        for (pad, held) in self.held.iter_mut().enumerate() {
            let base = self.pads[pad].0;
            let mut out = Vec::new();
            while let Some(message) = held.front() {
                let due = match (now, clock_base) {
                    (Some(now), Some(clock)) => compare(message.pts, base, now, clock).is_le(),
                    _ => true,
                };
                if !due {
                    break;
                }
                out.push(held.pop_front().expect("front is some"));
            }
            released.push(out);
        }
        released
    }
}

/// The rows a state input's instance is owed: those of every tick it did
/// not process since its previous call, oldest first. A frame-parallel
/// node's instances each see every row, whichever ticks they were handed.
#[derive(Default)]
pub struct EarlierRows {
    /// Per stream id, every tick's rows, by tick number.
    ticks: BTreeMap<u32, Vec<(u64, TimedRows)>>,
    /// Per instance, the next tick number whose rows it has not been told.
    told: BTreeMap<usize, u64>,
}

impl EarlierRows {
    /// The rows stream `id` brought at tick `number`, at `pts`.
    pub fn record(&mut self, id: u32, number: u64, pts: i64, rows: Vec<String>) {
        if rows.is_empty() {
            return;
        }
        self.ticks
            .entry(id)
            .or_default()
            .push((number, TimedRows { pts, rows }));
    }

    /// What `instance`, about to process tick `number`, is owed on stream
    /// `id`: the rows of every tick before `number` it has not been told of
    /// and did not process itself.
    pub fn owed(&mut self, instance: usize, id: u32, number: u64) -> Vec<TimedRows> {
        let from = self.told.get(&instance).copied().unwrap_or(0);
        let owed = self
            .ticks
            .get(&id)
            .map(|ticks| {
                ticks
                    .iter()
                    .filter(|(n, _)| *n >= from && *n < number)
                    .map(|(_, rows)| rows.clone())
                    .collect()
            })
            .unwrap_or_default();
        owed
    }

    /// `instance` has processed tick `number`: it knows everything up to it.
    pub fn processed(&mut self, instance: usize, number: u64) {
        self.told.insert(instance, number + 1);
    }

    /// The rows every one of `instances` has been told of go.
    pub fn forget_told(&mut self, instances: usize) {
        let past = (0..instances)
            .map(|i| self.told.get(&i).copied().unwrap_or(0))
            .min()
            .unwrap_or(0);
        for ticks in self.ticks.values_mut() {
            ticks.retain(|(n, _)| *n >= past);
        }
    }
}

/// The feeds a hold input's instance is owed: those that ended on any tick
/// since its previous call, the tick it processes included, oldest first.
/// A frame-parallel node's instances each hear of every end, whichever
/// ticks they were handed.
#[derive(Default)]
pub struct EndedFeeds {
    /// Per stream id, every feed that ended, by the tick it ended on.
    ended: BTreeMap<u32, Vec<(u64, Feed)>>,
    /// Per instance, the first tick number whose ends it has not been told.
    told: BTreeMap<usize, u64>,
}

impl EndedFeeds {
    pub fn record(&mut self, id: u32, number: u64, feed: Feed) {
        self.ended.entry(id).or_default().push((number, feed));
    }

    /// What `instance`, about to process tick `number`, is owed on `id`.
    pub fn owed(&self, instance: usize, id: u32, number: u64) -> Vec<Feed> {
        let from = self.told.get(&instance).copied().unwrap_or(0);
        self.ended
            .get(&id)
            .map(|ended| {
                ended
                    .iter()
                    .filter(|(n, _)| *n >= from && *n <= number)
                    .map(|(_, feed)| feed.clone())
                    .collect()
            })
            .unwrap_or_default()
    }

    pub fn processed(&mut self, instance: usize, number: u64) {
        self.told.insert(instance, number + 1);
    }

    /// The ends every one of `instances` has been told of go.
    pub fn forget_told(&mut self, instances: usize) {
        let past = (0..instances)
            .map(|i| self.told.get(&i).copied().unwrap_or(0))
            .min()
            .unwrap_or(0);
        for ended in self.ended.values_mut() {
            ended.retain(|(n, _)| *n >= past);
        }
    }
}

/// One thing arriving on a bound stream.
#[derive(Debug, Clone)]
pub enum Item {
    Frame(TickFrame),
    Message(Message),
    Packet(Packet),
}

/// One bound stream as the assembler holds it.
struct Input {
    id: u32,
    kind: PortKind,
    pairing: Pairing,
    rows: RowsUse,
    is_clock: bool,
    base: TimeBase,
    audio: Option<AudioFormat>,
    /// Arrived and not yet handed over, oldest first.
    queue: VecDeque<Item>,
    /// The newest time the producer has said it is done to, by a progress
    /// mark or by an item's own time.
    progress: Option<i64>,
    ended: bool,
    /// A hold input: its group, and which member of it.
    hold: Option<(usize, usize)>,
    /// An interval input on a time origin of its own, re-stamped onto the
    /// clock, `base` being the clock's.
    restamp: Option<Restamp>,
}

/// A stream placed on the clock by its first message: its own time base,
/// and once that message has arrived the offset, in the clock's base, that
/// lays it at the tick it arrived on.
#[derive(Debug, Clone, Copy)]
struct Restamp {
    from: TimeBase,
    offset: Option<i64>,
}

impl Restamp {
    fn waiting(&self) -> bool {
        self.offset.is_none()
    }
}

/// Whether `stream` on `port` counts on an origin of its own and is
/// re-stamped onto the clock: an interval input anchored `first-frame`, or
/// `tagged` where the stream's tags do not set the name to 1.
pub fn restamped(port: &InputPort, stream: &BoundStream) -> bool {
    let Pairing::Interval(interval) = &port.pairing else {
        return false;
    };
    match &interval.anchor {
        Anchor::SharedClock => false,
        Anchor::FirstFrame => true,
        Anchor::Tagged(name) => !stream.info.tags.iter().any(|(k, v)| k == name && v == "1"),
    }
}

/// `item`'s times moved by `to`.
fn retime(item: &mut Item, to: impl Fn(i64) -> i64) {
    match item {
        Item::Frame(f) => f.pts = to(f.pts),
        Item::Message(m) => m.pts = to(m.pts),
        Item::Packet(p) => {
            p.pts = to(p.pts);
            p.dts = p.dts.map(&to);
        }
    }
}

impl Input {
    /// A time of this stream's own, on the clock once it has an offset.
    fn placed(&self, time: i64) -> Option<i64> {
        match self.restamp {
            None => Some(time),
            Some(Restamp { offset: None, .. }) => None,
            Some(Restamp {
                from,
                offset: Some(offset),
            }) => Some(ffrwd_wasm_runtime::node::rescale(time, from, self.base) + offset),
        }
    }

    /// The first message's offset: it stands at `pts`, the tick being made.
    fn anchor_at(&mut self, pts: i64) {
        let Some(restamp) = self.restamp.filter(Restamp::waiting) else {
            return;
        };
        let Some(first) = self.queue.front().map(|item| self.item_time(item)) else {
            return;
        };
        let offset = pts - ffrwd_wasm_runtime::node::rescale(first, restamp.from, self.base);
        self.restamp = Some(Restamp {
            offset: Some(offset),
            ..restamp
        });
        let base = self.base;
        let to = |t: i64| ffrwd_wasm_runtime::node::rescale(t, restamp.from, base) + offset;
        for item in self.queue.iter_mut() {
            retime(item, &to);
        }
        self.progress = self.queue.iter().map(|item| self.item_time(item)).max();
    }

    fn item_time(&self, item: &Item) -> i64 {
        match item {
            Item::Frame(f) => f.pts,
            Item::Message(m) => m.pts,
            Item::Packet(p) => p.dts.unwrap_or(p.pts),
        }
    }

    /// Whether this input has said everything before `end` (in `base`).
    fn settled_to(&self, end: i64, base: TimeBase) -> bool {
        if self.ended {
            return true;
        }
        self.progress
            .is_some_and(|p| compare(p, self.base, end, base).is_ge())
    }
}

/// The clock's own state.
enum ClockState {
    /// An input clock, the index of its input.
    Input(usize),
    /// A rate clock: ticks so far, and the rate's time base.
    Rate {
        base: TimeBase,
    },
    SelfClocked,
}

/// A node's ticks, cut from what arrives on its bound streams.
pub struct Assembler {
    inputs: Vec<Input>,
    clock: ClockState,
    window: usize,
    stride: usize,
    /// Ticks made so far.
    made: u64,
    /// A packets clock's running maximum of dts.
    packets_time: Option<i64>,
    /// An audio clock's samples not yet consumed, the first at `audio_pts`:
    /// how many, and their bytes, which a timing clock carries none of.
    audio_buffer: Vec<u8>,
    audio_buffered: usize,
    audio_pts: Option<i64>,
    /// Whether the last tick has been made.
    done: bool,
    /// Where the interval of the tick made last ends, in its time base.
    made_end: Option<i64>,
    /// Rows of state inputs, for instances that did not see every tick.
    pub earlier: EarlierRows,
    /// Feeds of hold inputs that ended, for every instance to hear of.
    pub ended: EndedFeeds,
    /// Feeds that ended while the tick being made was cut, by stream id.
    ending: Vec<(u32, Feed)>,
    /// The hold inputs, grouped.
    holds: Vec<Group>,
    /// The clock's ticks as a grid, for foretelling where a feed ends.
    grid: Grid,
    /// The last tick's time, for a jump in the clock.
    last_pts: Option<i64>,
    name: String,
    report: Report,
}

impl Assembler {
    /// The ticks of `name`, a node of `shape` bound to `bound`. `port_fed`
    /// names the hold inputs a port feed serves, which never hold a tick.
    pub fn new(
        shape: &NodeShape,
        bound: &[BoundStream],
        name: &str,
        port_fed: &[u32],
    ) -> Result<Assembler> {
        let clock_base = match &shape.clock {
            Clock::Input(name) => bound.iter().find(|b| &b.port == name).map(|b| b.time_base),
            Clock::Rate(rate) => Some(TimeBase {
                num: u64::try_from(rate.den).unwrap_or(1),
                den: u64::try_from(rate.num).unwrap_or(1),
            }),
            _ => None,
        };
        let mut inputs = Vec::with_capacity(bound.len());
        let mut holds: Vec<Group> = Vec::new();
        let mut on_groups: Vec<(usize, String, Member, bool)> = Vec::new();
        let mut grouped: Vec<(Option<String>, Hold, Vec<Member>, bool)> = Vec::new();
        for stream in bound {
            let port = shape
                .input(&stream.port)
                .ok_or_else(|| anyhow!("no input port '{}'", stream.port))?;
            if port.pairing == Pairing::AtOrBefore {
                bail!(
                    "input '{}' pairs at or before the clock, which only an adapted data filter does",
                    port.name
                );
            }
            let is_clock = matches!(&shape.clock, Clock::Input(c) if *c == port.name);
            let audio = match &stream.format {
                StreamFormat::Audio(audio) => Some(*audio),
                _ => None,
            };
            let hold = match &port.pairing {
                Pairing::Hold(hold) => {
                    let member = Member {
                        name: port.name.clone(),
                        kind: port.kind,
                        base: stream.time_base,
                        info: stream.info.clone(),
                        audio,
                        ahead: 0.0,
                    };
                    let fed = port_fed.contains(&stream.id);
                    let key = hold.group.clone();
                    let group = match key
                        .as_ref()
                        .and_then(|k| grouped.iter().position(|(g, ..)| g.as_ref() == Some(k)))
                    {
                        Some(group) => group,
                        None => {
                            grouped.push((key, hold.clone(), Vec::new(), fed));
                            grouped.len() - 1
                        }
                    };
                    grouped[group].2.push(member);
                    grouped[group].3 |= fed;
                    Some((group, grouped[group].2.len() - 1))
                }
                Pairing::Interval(Interval {
                    group: Some(group),
                    ahead,
                    ..
                }) => {
                    on_groups.push((
                        inputs.len(),
                        group.clone(),
                        Member {
                            name: port.name.clone(),
                            kind: port.kind,
                            base: stream.time_base,
                            info: stream.info.clone(),
                            audio: None,
                            ahead: *ahead,
                        },
                        port_fed.contains(&stream.id),
                    ));
                    None
                }
                _ => None,
            };
            let restamp = match clock_base {
                Some(_) if restamped(port, stream) => Some(Restamp {
                    from: stream.time_base,
                    offset: None,
                }),
                _ => None,
            };
            inputs.push(Input {
                id: stream.id,
                kind: port.kind,
                pairing: port.pairing.clone(),
                rows: port.rows,
                is_clock,
                base: match restamp {
                    Some(_) => clock_base.unwrap_or(stream.time_base),
                    None => stream.time_base,
                },
                audio,
                queue: VecDeque::new(),
                progress: None,
                ended: false,
                hold,
                restamp,
            });
        }
        for (index, key, member, fed) in on_groups {
            let Some(group) = grouped
                .iter()
                .position(|(g, ..)| g.as_deref() == Some(key.as_str()))
            else {
                bail!(
                    "input '{}' arrives on group '{key}', and this call binds none of that \
                     group's hold inputs",
                    member.name
                );
            };
            grouped[group].2.push(member);
            grouped[group].3 |= fed;
            inputs[index].hold = Some((group, grouped[group].2.len() - 1));
        }
        for (_, hold, members, fed) in grouped {
            holds.push(Group::new(hold, name, members, fed));
        }
        let (clock, window, stride) = match &shape.clock {
            Clock::Input(name) => {
                let index = inputs
                    .iter()
                    .position(|i| i.is_clock)
                    .ok_or_else(|| anyhow!("the clock input '{name}' is bound to no stream"))?;
                let port = shape.input(name).expect("checked with the shape");
                (
                    ClockState::Input(index),
                    port.window as usize,
                    port.stride as usize,
                )
            }
            Clock::Rate(rate) => (
                ClockState::Rate {
                    base: TimeBase {
                        num: u64::try_from(rate.den).unwrap_or(1),
                        den: u64::try_from(rate.num).unwrap_or(1),
                    },
                },
                1,
                1,
            ),
            Clock::RateOf(_) => bail!(
                "a clock at an input's rate is read off that input by the compiler, which hands \
                 the host a rate"
            ),
            Clock::SelfClocked => (ClockState::SelfClocked, 1, 1),
        };
        let grid = match clock {
            ClockState::Input(_) => Grid::learned(),
            _ => Grid::exact(),
        };
        Ok(Assembler {
            inputs,
            clock,
            window,
            stride,
            made: 0,
            packets_time: None,
            audio_buffer: Vec::new(),
            audio_buffered: 0,
            audio_pts: None,
            done: false,
            made_end: None,
            earlier: EarlierRows::default(),
            ended: EndedFeeds::default(),
            ending: Vec::new(),
            holds,
            grid,
            last_pts: None,
            name: name.to_string(),
            report: Box::new(|line| eprintln!("{line}")),
        })
    }

    fn input(&mut self, id: u32) -> Result<&mut Input> {
        self.inputs
            .iter_mut()
            .find(|i| i.id == id)
            .ok_or_else(|| anyhow!("stream {id} is bound to no input"))
    }

    /// One item arriving on stream `id`.
    pub fn arrive(&mut self, id: u32, mut item: Item) -> Result<()> {
        let input = self.input(id)?;
        if input.restamp.is_some_and(|r| !r.waiting()) {
            let raw = input.item_time(&item);
            let placed = input.placed(raw).expect("anchored");
            retime(&mut item, |t| t + placed - raw);
        }
        let time = input.item_time(&item);
        if !input.restamp.is_some_and(|r| r.waiting()) {
            input.progress = Some(input.progress.map_or(time, |p| p.max(time)));
        }
        match (input.hold, item) {
            (Some((group, member)), Item::Frame(frame)) => {
                self.holds[group].arrive(member, frame);
            }
            (Some((group, member)), Item::Message(message)) => {
                self.holds[group].arrive(
                    member,
                    TickFrame {
                        pts: message.pts,
                        duration: None,
                        data: Arc::new(message.data),
                        rows: Vec::new(),
                    },
                );
            }
            (Some(_), _) => {}
            (None, item) => input.queue.push_back(item),
        }
        Ok(())
    }

    /// A progress mark on stream `id`: nothing more will arrive before
    /// `pts`.
    pub fn progress(&mut self, id: u32, pts: i64) -> Result<()> {
        let input = self.input(id)?;
        if let (Some((group, member)), PortKind::Data) = (input.hold, input.kind) {
            self.holds[group].mark(member, pts);
            return Ok(());
        }
        if let Some(pts) = input.placed(pts) {
            input.progress = Some(input.progress.map_or(pts, |p| p.max(pts)));
        }
        Ok(())
    }

    /// Stream `id` has ended.
    pub fn end(&mut self, id: u32) -> Result<()> {
        let input = self.input(id)?;
        input.ended = true;
        if let Some((group, member)) = input.hold {
            self.holds[group].source_close(member);
        }
        Ok(())
    }

    /// A source starts on hold input `id`: a port feed's connection.
    pub fn source_open(&mut self, id: u32, source: SourceInfo) -> Result<()> {
        let input = self.input(id)?;
        let Some((group, member)) = input.hold else {
            bail!("stream {id} is not a hold input, and only one takes a source");
        };
        self.holds[group].source_open(member, source);
        Ok(())
    }

    /// The source on hold input `id` has closed; what it sent plays out.
    pub fn source_close(&mut self, id: u32) -> Result<()> {
        let input = self.input(id)?;
        let Some((group, member)) = input.hold else {
            bail!("stream {id} is not a hold input, and only one takes a source");
        };
        self.holds[group].source_close(member);
        Ok(())
    }

    /// How many more frames a port feed on hold input `id` may send before
    /// it waits; None for a stream that is not one.
    pub fn hold_room(&self, id: u32) -> Option<usize> {
        let input = self.inputs.iter().find(|i| i.id == id)?;
        let (group, _) = input.hold?;
        let group = &self.holds[group];
        group.port_fed().then(|| group.room())
    }

    /// How many more frames the hold input bound to stream `id` takes before
    /// its buffer is full, whoever feeds it; None where `id` is not held.
    pub fn held_room(&self, id: u32) -> Option<usize> {
        let input = self.inputs.iter().find(|i| i.id == id)?;
        let (group, _) = input.hold?;
        Some(self.holds[group].room())
    }

    fn port_fed(&self, input: &Input) -> bool {
        input
            .hold
            .is_some_and(|(group, _)| self.holds[group].port_fed())
    }

    /// Whether every bound stream has ended. A port feed never ends: the
    /// run's end is its end.
    pub fn all_ended(&self) -> bool {
        self.inputs.iter().all(|i| i.ended && !self.port_fed(i))
    }

    #[cfg(test)]
    pub fn is_done(&self) -> bool {
        self.done
    }

    /// The next tick of an input clock, if what has arrived settles it.
    pub fn next_clocked(&mut self) -> Result<Option<Tick>> {
        let ClockState::Input(clock) = self.clock else {
            bail!("the node has no input clock");
        };
        if self.done {
            return Ok(None);
        }
        let Some(next) = self.peek_clock(clock)? else {
            return Ok(None);
        };
        let base = self.inputs[clock].base;
        let end = if next.last { None } else { next.end };
        for input in &self.inputs {
            if input.is_clock {
                continue;
            }
            if !self.input_ready(input, next.pts, end, base) {
                return Ok(None);
            }
        }
        let clock_stream = self.take_clock(clock, &next)?;
        let mut streams = Vec::with_capacity(self.inputs.len());
        streams.push(clock_stream);
        let mut held = self.hold_ticks(next.pts, end, base);
        for index in 0..self.inputs.len() {
            if index == clock {
                continue;
            }
            streams.push(self.hand(index, next.pts, end, base, &mut held)?);
        }
        if next.last {
            self.done = true;
        }
        Ok(Some(self.made(next.pts, end, base, next.last, streams)))
    }

    /// Tick `self.made` of a rate clock, with whatever its inputs pair to
    /// it; None while a held or interval input has not settled it. `last`
    /// asks for the final call.
    pub fn next_rate(&mut self, last: bool) -> Result<Option<Tick>> {
        let ClockState::Rate { base } = self.clock else {
            bail!("the node has no rate clock");
        };
        if self.done {
            return Ok(None);
        }
        let pts = self.made as i64;
        let end = if last { None } else { Some(pts + 1) };
        for input in &self.inputs {
            if !last && !self.input_ready(input, pts, end, base) {
                return Ok(None);
            }
        }
        let mut streams = Vec::with_capacity(self.inputs.len());
        let mut held = self.hold_ticks(pts, end, base);
        for index in 0..self.inputs.len() {
            streams.push(self.hand(index, pts, end, base, &mut held)?);
        }
        if last {
            self.done = true;
        }
        Ok(Some(self.made(pts, end, base, last, streams)))
    }

    /// A self-clocked node's next call: whatever has arrived. `pts` is the
    /// microseconds since its first call.
    pub fn next_arrivals(&mut self, pts: i64, last: bool) -> Result<Tick> {
        let base = TimeBase {
            num: 1,
            den: 1_000_000,
        };
        let mut streams = Vec::with_capacity(self.inputs.len());
        let mut held = Vec::new();
        for index in 0..self.inputs.len() {
            streams.push(self.hand(index, pts, None, base, &mut held)?);
        }
        if last {
            self.done = true;
        }
        Ok(self.made(pts, None, base, last, streams))
    }

    /// Every hold group's share of the tick from `pts` to `end`: a jump in
    /// the clock ends every feed first.
    fn hold_ticks(&mut self, pts: i64, end: Option<i64>, base: TimeBase) -> Vec<Vec<Handed>> {
        if let Some(last) = self.last_pts {
            let step = pts.saturating_sub(last);
            if step < 0 || step > hold::ticks_of(hold::MAX_STEP_SECONDS, base) {
                for group in &mut self.holds {
                    group.clock_jumped(pts, base);
                }
            }
        }
        self.last_pts = Some(pts);
        let grid = self.grid;
        let handed: Vec<Vec<Handed>> = self
            .holds
            .iter_mut()
            .map(|group| group.tick(pts, end, base, &grid))
            .collect();
        for (index, group) in self.holds.iter_mut().enumerate() {
            for (member, feed) in group.take_ended() {
                let id = self
                    .inputs
                    .iter()
                    .find(|i| i.hold == Some((index, member)))
                    .map(|i| i.id);
                if let Some(id) = id {
                    self.ending.push((id, feed));
                }
            }
        }
        handed
    }

    /// Where the interval of the tick made last ends, in its time base:
    /// None for a last call that takes whatever is left, and for a
    /// self-clocked node's.
    pub fn tick_end(&self) -> Option<i64> {
        self.made_end
    }

    /// The newest time any input has said it is done to, in `base`: a hold
    /// input's as its feed maps it.
    pub fn latest(&self, base: TimeBase) -> Option<i64> {
        let groups = self.holds.iter().filter_map(|g| g.progress(base));
        self.inputs
            .iter()
            .filter(|i| i.hold.is_none())
            .filter_map(|i| {
                i.progress
                    .map(|p| ffrwd_wasm_runtime::node::rescale(p, i.base, base))
            })
            .chain(groups)
            .max()
    }

    /// How far the clock has arrived, in `base`, which bounds how long an
    /// input is waited for: the clock input's newest time, or on a rate
    /// clock the newest time any input has reached.
    fn reference(&self, base: TimeBase) -> Option<i64> {
        match self.clock {
            ClockState::Input(clock) => {
                let input = &self.inputs[clock];
                input
                    .progress
                    .map(|p| ffrwd_wasm_runtime::node::rescale(p, input.base, base))
            }
            _ => self.latest(base),
        }
    }

    /// Whether anything has arrived that no tick has taken.
    pub fn has_arrivals(&self) -> bool {
        self.inputs.iter().any(|i| !i.queue.is_empty())
    }

    /// Whether every input is delivered as it arrives, so nothing it brings
    /// settles a tick and its turns are bounded only by time.
    pub fn arrival_only(&self) -> bool {
        self.inputs.iter().all(|i| i.pairing == Pairing::Arrival)
    }

    /// Whether a rate clock whose inputs have all ended has handed them
    /// everything: nothing delivered as it arrives is waiting, and its
    /// ticks have passed the newest time of every paired input. An arrival
    /// input's times may sit on any origin, so they never hold the clock.
    pub fn played_out(&self, base: TimeBase) -> bool {
        let unpaired = |i: &Input| i.pairing == Pairing::Arrival;
        if self
            .inputs
            .iter()
            .any(|i| unpaired(i) && !i.queue.is_empty())
        {
            return false;
        }
        let groups = self.holds.iter().filter_map(|g| g.progress(base));
        let latest = self
            .inputs
            .iter()
            .filter(|i| i.hold.is_none() && !unpaired(i))
            .filter_map(|i| {
                i.progress
                    .map(|p| ffrwd_wasm_runtime::node::rescale(p, i.base, base))
            })
            .chain(groups)
            .max();
        latest.is_none_or(|latest| self.made as i64 > latest)
    }

    /// Whether any stream is bound to the node.
    pub fn has_inputs(&self) -> bool {
        !self.inputs.is_empty()
    }

    fn made(
        &mut self,
        pts: i64,
        end: Option<i64>,
        base: TimeBase,
        last: bool,
        streams: Vec<TickStream>,
    ) -> Tick {
        let number = self.made;
        self.made += 1;
        self.made_end = end;
        self.grid.observe(pts);
        for (id, feed) in std::mem::take(&mut self.ending) {
            self.ended.record(id, number, feed);
        }
        for stream in &streams {
            let state = self
                .inputs
                .iter()
                .any(|i| i.id == stream.id && i.rows == RowsUse::State);
            if !state {
                continue;
            }
            let rows: Vec<String> = stream
                .frames
                .iter()
                .flat_map(|f| f.rows.iter().cloned())
                .chain(
                    stream
                        .messages
                        .iter()
                        .map(|m| String::from_utf8_lossy(&m.data).into_owned()),
                )
                .collect();
            self.earlier.record(stream.id, number, pts, rows);
        }
        Tick {
            pts,
            ordinal: number,
            time_base: base,
            last,
            streams,
        }
    }

    /// Where the clock's next tick stands, without taking it: None while
    /// the clock has not arrived far enough to say.
    ///
    /// A tick's interval ends where the next one starts, or where its frame
    /// says it does, so a clock waits for that before it ticks: the progress
    /// a node sends, and what an interval or lockstep input is handed, are
    /// then the same however the input arrived.
    fn peek_clock(&mut self, clock: usize) -> Result<Option<NextTick>> {
        let window = self.window;
        let stride = self.stride;
        let input = &mut self.inputs[clock];
        match input.kind {
            PortKind::Video => {
                let have = input.queue.len().min(window);
                if have < window && !input.ended {
                    return Ok(None);
                }
                let Some(first) = input.queue.front() else {
                    return Ok(Some(empty_last(input)));
                };
                let pts = input.item_time(first);
                let next = input.queue.get(stride).map(|item| input.item_time(item));
                let last = input.ended && next.is_none();
                let end = match next {
                    Some(t) => Some(t),
                    None if last => None,
                    None => match input.queue.get(stride - 1) {
                        Some(Item::Frame(TickFrame {
                            pts,
                            duration: Some(d),
                            ..
                        })) if window == stride => Some(pts + d),
                        _ => return Ok(None),
                    },
                };
                let take = if last { input.queue.len() } else { stride };
                Ok(Some(NextTick {
                    pts,
                    end,
                    last,
                    take,
                }))
            }
            PortKind::Audio => {
                let audio = input
                    .audio
                    .ok_or_else(|| anyhow!("the audio clock has no sample format"))?;
                while let Some(Item::Frame(frame)) = input.queue.front() {
                    if self.audio_buffered == 0 {
                        self.audio_pts = Some(frame.pts);
                    }
                    self.audio_buffered += sample_count(frame, audio.sample_len());
                    self.audio_buffer.extend_from_slice(&frame.data);
                    input.queue.pop_front();
                }
                let have = self.audio_buffered;
                if have < window && !input.ended {
                    return Ok(None);
                }
                let Some(pts) = self.audio_pts.filter(|_| have > 0) else {
                    return Ok(Some(empty_last(input)));
                };
                let last = input.ended && have <= window;
                let take = if last { have } else { stride.min(have) };
                let step = samples_to_ticks(take as i64, audio.sample_rate, input.base);
                Ok(Some(NextTick {
                    pts,
                    end: Some(pts + step),
                    last,
                    take,
                }))
            }
            PortKind::Data => {
                let Some(first) = input.queue.front().map(|i| input.item_time(i)) else {
                    return Ok(if input.ended {
                        Some(empty_last(input))
                    } else {
                        None
                    });
                };
                let count = input
                    .queue
                    .iter()
                    .take_while(|i| input.item_time(i) == first)
                    .count();
                let next = input.queue.get(count).map(|i| input.item_time(i));
                let passed = input.progress.is_some_and(|p| p > first);
                let last = input.ended && next.is_none();
                if next.is_none() && !last && !passed {
                    return Ok(None);
                }
                Ok(Some(NextTick {
                    pts: first,
                    end: next.or(input.progress.filter(|p| *p > first)),
                    last,
                    take: count,
                }))
            }
            PortKind::Packets => {
                let Some(first) = input.queue.front().map(|i| input.item_time(i)) else {
                    return Ok(if input.ended {
                        Some(empty_last(input))
                    } else {
                        None
                    });
                };
                let pts = self.packets_time.map_or(first, |t| t.max(first));
                let next = input.queue.get(1).map(|i| input.item_time(i).max(pts));
                let last = input.ended && next.is_none();
                if next.is_none() && !last {
                    return Ok(None);
                }
                Ok(Some(NextTick {
                    pts,
                    end: next,
                    last,
                    take: 1,
                }))
            }
        }
    }

    /// Takes the clock's tick `next` described.
    fn take_clock(&mut self, clock: usize, next: &NextTick) -> Result<TickStream> {
        let window = self.window;
        let input = &mut self.inputs[clock];
        let mut stream = TickStream {
            id: input.id,
            progress: input.progress,
            ..TickStream::default()
        };
        match input.kind {
            PortKind::Video => {
                stream.frames = input
                    .queue
                    .iter()
                    .take(if next.last { input.queue.len() } else { window })
                    .filter_map(|item| match item {
                        Item::Frame(f) => Some(f.clone()),
                        _ => None,
                    })
                    .collect();
                input.queue.drain(..next.take.min(input.queue.len()));
            }
            PortKind::Audio => {
                let audio = input.audio.expect("peeked as audio");
                let width = audio.sample_len();
                let have = self.audio_buffered;
                let handed = if next.last { have } else { window.min(have) };
                let data = match self.audio_buffer.is_empty() {
                    true => Vec::new(),
                    false => self.audio_buffer[..handed * width].to_vec(),
                };
                let drained = (next.take * width).min(self.audio_buffer.len());
                self.audio_buffer.drain(..drained);
                self.audio_buffered -= next.take.min(self.audio_buffered);
                let step = samples_to_ticks(next.take as i64, audio.sample_rate, input.base);
                self.audio_pts = Some(next.pts + step);
                if self.audio_buffered == 0 {
                    self.audio_pts = None;
                }
                if handed > 0 {
                    stream.frames.push(TickFrame {
                        pts: next.pts,
                        duration: Some(samples_to_ticks(
                            handed as i64,
                            audio.sample_rate,
                            input.base,
                        )),
                        data: Arc::new(data),
                        rows: Vec::new(),
                    });
                }
            }
            PortKind::Data | PortKind::Packets => {
                if input.kind == PortKind::Packets && next.take > 0 {
                    self.packets_time = Some(next.pts);
                }
                for item in input.queue.drain(..next.take.min(input.queue.len())) {
                    push_item(&mut stream, item);
                }
            }
        }
        stream.frames = strip_rows(stream.frames, input.rows);
        Ok(stream)
    }

    /// Whether `input` has said enough to be handed for the interval from
    /// `pts` to `end` (None: to the end of the inputs) in `base`.
    fn input_ready(&self, input: &Input, pts: i64, end: Option<i64>, base: TimeBase) -> bool {
        if input.ended {
            return true;
        }
        match &input.pairing {
            Pairing::Arrival => true,
            Pairing::Lockstep => match end {
                Some(end) => input.settled_to(end, base),
                None => false,
            },
            Pairing::Hold(_) => match input.hold {
                Some((group, _)) => self.holds[group].ready(pts, end, base, self.reference(base)),
                None => true,
            },
            Pairing::Interval(interval) => {
                if input.restamp.is_some_and(|r| r.waiting()) {
                    return true;
                }
                if let Some((group, _)) = input.hold {
                    return self.holds[group].ready(pts, end, base, self.reference(base));
                }
                let Some(end) = end else {
                    return false;
                };
                let until = plus_seconds(end, base, interval.ahead);
                if input.settled_to(until, base) {
                    return true;
                }
                match (interval.latency, self.reference(base)) {
                    (Some(latency), Some(arrived)) => arrived >= plus_seconds(end, base, latency),
                    _ => false,
                }
            }
            Pairing::AtOrBefore => true,
        }
    }

    /// Input `index`'s share of the tick at `pts` whose interval ends at
    /// `end` (None: everything left). `held` is what every hold group hands
    /// this tick.
    fn hand(
        &mut self,
        index: usize,
        pts: i64,
        end: Option<i64>,
        base: TimeBase,
        held: &mut [Vec<Handed>],
    ) -> Result<TickStream> {
        let first = self.made == 0;
        let input = &mut self.inputs[index];
        if matches!(input.pairing, Pairing::Interval(_)) {
            input.anchor_at(pts);
        }
        let mut stream = TickStream {
            id: input.id,
            progress: input.progress,
            ..TickStream::default()
        };
        let before = |time: i64, input_base: TimeBase, limit: Option<i64>| match limit {
            Some(limit) => compare(time, input_base, limit, base).is_lt(),
            None => true,
        };
        match &input.pairing {
            Pairing::Arrival | Pairing::AtOrBefore => {
                for item in input.queue.drain(..) {
                    push_item(&mut stream, item);
                }
            }
            Pairing::Lockstep => {
                if input.kind == PortKind::Audio {
                    let audio = input
                        .audio
                        .ok_or_else(|| anyhow!("an audio input has no sample format"))?;
                    if let Some(frame) = recut(&mut input.queue, audio, input.base, end, base)? {
                        stream.frames.push(frame);
                    }
                } else {
                    while let Some(item) = input.queue.front() {
                        if !before(input.item_time(item), input.base, end) {
                            break;
                        }
                        let item = input.queue.pop_front().expect("front is some");
                        push_item(&mut stream, item);
                    }
                }
            }
            Pairing::Hold(_) => {
                if let Some((group, member)) = input.hold {
                    let handed = std::mem::take(&mut held[group][member]);
                    stream.frames = handed.frames;
                    stream.feed = handed.feed;
                    stream.info = handed.info;
                }
            }
            Pairing::Interval(_) if input.hold.is_some() => {
                if let Some((group, member)) = input.hold {
                    let handed = std::mem::take(&mut held[group][member]);
                    stream.messages = handed.messages;
                    stream.info = handed.info;
                }
            }
            Pairing::Interval(interval) => {
                let until = end.map(|end| plus_seconds(end, base, interval.ahead));
                let mut late = 0;
                let mut earliest: Option<i64> = None;
                while let Some(item) = input.queue.front() {
                    if !before(input.item_time(item), input.base, until) {
                        break;
                    }
                    let item = input.queue.pop_front().expect("front is some");
                    let time = input.item_time(&item);
                    if !first && compare(time, input.base, pts, base).is_lt() {
                        late += 1;
                        earliest = Some(earliest.map_or(time, |e: i64| e.min(time)));
                    }
                    push_item(&mut stream, item);
                }
                if late > 0 {
                    let seconds = |t: i64, b: TimeBase| t as f64 * b.num as f64 / b.den as f64;
                    let earliest = earliest.map_or(0.0, |t| seconds(t, input.base));
                    let at = seconds(pts, base);
                    let id = input.id;
                    (self.report)(format!(
                        "interval: {}: {late} message(s) stamped before the tick at {at:.3}s \
                         arrived after it, the earliest at {earliest:.3}s; delivered now",
                        self.name
                    ));
                    let row = serde_json::json!({
                        "kind": "late",
                        "node": self.name,
                        "stream": id,
                        "at": (at * 1000.0).round() / 1000.0,
                        "late": late,
                        "earliest": (earliest * 1000.0).round() / 1000.0,
                    });
                    (self.report)(format!("{}{row}", crate::leaky::ROW_PREFIX));
                }
            }
        }
        stream.frames = strip_rows(stream.frames, input.rows);
        Ok(stream)
    }
}

/// Where an input clock's next tick stands: its time, where its interval
/// ends (None where nothing yet says), whether it is the last, and how many
/// frames, samples or messages it consumes.
struct NextTick {
    pts: i64,
    end: Option<i64>,
    last: bool,
    take: usize,
}

/// The final call of a clock with nothing left: at the time just past its
/// last.
fn empty_last(input: &Input) -> NextTick {
    NextTick {
        pts: input.progress.map_or(0, |p| p + 1),
        end: None,
        last: true,
        take: 0,
    }
}

/// An input that ignores its rows is handed none.
fn strip_rows(mut frames: Vec<TickFrame>, rows: RowsUse) -> Vec<TickFrame> {
    if rows == RowsUse::Ignore {
        for frame in &mut frames {
            frame.rows.clear();
        }
    }
    frames
}

fn push_item(stream: &mut TickStream, item: Item) {
    match item {
        Item::Frame(f) => stream.frames.push(f),
        Item::Message(m) => stream.messages.push(m),
        Item::Packet(p) => stream.packets.push(p),
    }
}

/// How many samples a run holds: its bytes' worth, or for a run of an input
/// read for its timing alone, which carries none, its duration in samples.
pub fn sample_count(frame: &TickFrame, width: usize) -> usize {
    if frame.data.is_empty() {
        frame
            .duration
            .map_or(0, |d| usize::try_from(d).unwrap_or(0))
    } else {
        frame.data.len() / width.max(1)
    }
}

/// `samples` at `rate` as ticks of `base`, rounded down.
fn samples_to_ticks(samples: i64, rate: u32, base: TimeBase) -> i64 {
    let num = i128::from(samples) * i128::from(base.den);
    let den = i128::from(rate) * i128::from(base.num);
    (num / den.max(1)) as i64
}

/// The samples of `queue` whose time falls before `end` (in `base`), as
/// one run at the first one's pts; everything left with no end. The rest
/// stays queued, its pts advanced to the first sample left.
fn recut(
    queue: &mut VecDeque<Item>,
    audio: AudioFormat,
    audio_base: TimeBase,
    end: Option<i64>,
    base: TimeBase,
) -> Result<Option<TickFrame>> {
    let width = audio.sample_len();
    let mut data = Vec::new();
    let mut total = 0usize;
    let mut first: Option<i64> = None;
    while let Some(Item::Frame(frame)) = queue.front() {
        let samples = sample_count(frame, width);
        let take = match end {
            None => samples,
            Some(end) => samples_before(frame.pts, audio_base, audio.sample_rate, end, base)
                .clamp(0, samples as i64) as usize,
        };
        if take == 0 {
            break;
        }
        first.get_or_insert(frame.pts);
        total += take;
        let timing = frame.data.is_empty();
        if !timing {
            data.extend_from_slice(&frame.data[..take * width]);
        }
        if take == samples {
            queue.pop_front();
            continue;
        }
        let left = TickFrame {
            pts: frame.pts + samples_to_ticks(take as i64, audio.sample_rate, audio_base),
            duration: timing.then_some((samples - take) as i64),
            data: match timing {
                true => Arc::new(Vec::new()),
                false => Arc::new(frame.data[take * width..].to_vec()),
            },
            rows: Vec::new(),
        };
        queue.pop_front();
        queue.push_front(Item::Frame(left));
        break;
    }
    let Some(pts) = first else {
        return Ok(None);
    };
    let samples = total as i64;
    Ok(Some(TickFrame {
        pts,
        duration: Some(samples_to_ticks(samples, audio.sample_rate, audio_base)),
        data: Arc::new(data),
        rows: Vec::new(),
    }))
}

/// How many samples of a run starting at `pts` (in `audio_base`, at `rate`)
/// fall before `end` (in `base`): the first whole sample at or past `end` is
/// not one of them.
fn samples_before(pts: i64, audio_base: TimeBase, rate: u32, end: i64, base: TimeBase) -> i64 {
    // (end*base - pts*audio_base) * rate, as one fraction, rounded up.
    let num = (i128::from(end) * i128::from(base.num) * i128::from(audio_base.den)
        - i128::from(pts) * i128::from(audio_base.num) * i128::from(base.den))
        * i128::from(rate);
    let den = i128::from(base.den) * i128::from(audio_base.den);
    if num <= 0 {
        return 0;
    }
    ((num + den - 1) / den) as i64
}

#[cfg(test)]
mod tests {
    use super::*;
    use ffrwd_wasm_runtime::node::{Accepts, InputPort, Interval, NodeShape, Rational};
    use ffrwd_wasm_runtime::runtime::{StreamInfo, VideoFormat};

    const NTSC: TimeBase = TimeBase { num: 1, den: 30000 };
    const KHZ48: TimeBase = TimeBase { num: 1, den: 48000 };
    const MICROS: TimeBase = TimeBase {
        num: 1,
        den: 1_000_000,
    };

    fn port(name: &str, kind: PortKind, pairing: Pairing, rows: RowsUse) -> InputPort {
        InputPort {
            name: name.to_string(),
            kind,
            required: true,
            many: false,
            pairing,
            rows,
            window: 1,
            stride: 1,
            accepts: Accepts::default(),
            schema: None,
        }
    }

    fn video() -> StreamFormat {
        StreamFormat::Video(VideoFormat {
            width: 2,
            height: 2,
            pix_fmt: "rgba",
            frame_len: 16,
            color: None,
        })
    }

    fn audio() -> StreamFormat {
        StreamFormat::Audio(AudioFormat {
            sample_rate: 48000,
            channels: 1,
            sample_fmt: "f32",
            channel_layout: None,
        })
    }

    fn bound(port: &str, id: u32, base: TimeBase, format: StreamFormat) -> BoundStream {
        BoundStream {
            port: port.to_string(),
            id,
            info: StreamInfo::default(),
            time_base: base,
            format,
            rendition: Default::default(),
            row: None,
            decode_delay: 0,
            latency: None,
            hint: Default::default(),
        }
    }

    fn shape(inputs: Vec<InputPort>, clock: Clock) -> NodeShape {
        NodeShape {
            inputs,
            outputs: Vec::new(),
            clock,
            pure: true,
            one_to_one: false,
            bounded: true,
            relation: Vec::new(),
        }
    }

    fn frame(pts: i64, rows: &[&str]) -> Item {
        Item::Frame(TickFrame {
            pts,
            duration: None,
            data: Arc::new(vec![0; 16]),
            rows: rows.iter().map(|r| r.to_string()).collect(),
        })
    }

    fn samples(pts: i64, count: usize) -> Item {
        Item::Frame(TickFrame {
            pts,
            duration: None,
            data: Arc::new(vec![0; count * 4]),
            rows: Vec::new(),
        })
    }

    fn message(pts: i64, text: &str) -> Item {
        Item::Message(Message {
            pts,
            data: text.as_bytes().to_vec(),
        })
    }

    fn drain(assembler: &mut Assembler) -> Vec<Tick> {
        let mut ticks = Vec::new();
        while let Some(tick) = assembler.next_clocked().expect("a tick") {
            ticks.push(tick);
        }
        ticks
    }

    #[test]
    fn a_lone_clock_ticks_on_every_frame_and_the_last_on_its_end() {
        let s = shape(
            vec![port(
                "v",
                PortKind::Video,
                Pairing::Lockstep,
                RowsUse::Ignore,
            )],
            Clock::Input("v".into()),
        );
        let mut a =
            Assembler::new(&s, &[bound("v", 7, NTSC, video())], "m", &[]).expect("assembler");
        for k in 0..3 {
            a.arrive(7, frame(k * 1001, &["r"])).expect("arrive");
        }
        let ticks = drain(&mut a);
        assert_eq!(
            ticks.iter().map(|t| t.pts).collect::<Vec<_>>(),
            [0, 1001],
            "a tick waits for the frame that ends its interval"
        );
        assert_eq!(a.tick_end(), Some(2002));
        assert!(
            ticks[0].streams[0].frames[0].rows.is_empty(),
            "ignore drops rows"
        );
        a.end(7).expect("end");
        let last = drain(&mut a);
        assert_eq!(last.len(), 1);
        assert!(last[0].last);
        assert_eq!(last[0].pts, 2002, "the last frame rides the last call");
        assert_eq!(last[0].streams[0].frames.len(), 1);
        assert_eq!(a.tick_end(), None);
    }

    #[test]
    fn a_window_and_stride_hand_overlapping_frames() {
        let mut v = port("v", PortKind::Video, Pairing::Lockstep, RowsUse::PerFrame);
        v.window = 3;
        v.stride = 1;
        let s = shape(vec![v], Clock::Input("v".into()));
        let mut a =
            Assembler::new(&s, &[bound("v", 0, NTSC, video())], "m", &[]).expect("assembler");
        for k in 0..5 {
            a.arrive(0, frame(k, &[])).expect("arrive");
        }
        let ticks = drain(&mut a);
        let windows: Vec<Vec<i64>> = ticks
            .iter()
            .map(|t| t.streams[0].frames.iter().map(|f| f.pts).collect())
            .collect();
        assert_eq!(windows, vec![vec![0, 1, 2], vec![1, 2, 3], vec![2, 3, 4]]);
        a.end(0).expect("end");
        let last = drain(&mut a);
        assert_eq!(
            last[0].streams[0]
                .frames
                .iter()
                .map(|f| f.pts)
                .collect::<Vec<_>>(),
            vec![3, 4],
            "the final call carries what the strides left"
        );
    }

    #[test]
    fn audio_under_a_ntsc_clock_is_recut_to_1602_and_1601_samples() {
        let s = shape(
            vec![
                port("v", PortKind::Video, Pairing::Lockstep, RowsUse::Ignore),
                port("a", PortKind::Audio, Pairing::Lockstep, RowsUse::Ignore),
            ],
            Clock::Input("v".into()),
        );
        let mut a = Assembler::new(
            &s,
            &[bound("v", 0, NTSC, video()), bound("a", 1, KHZ48, audio())],
            "m",
            &[],
        )
        .expect("assembler");
        for k in 0..6 {
            a.arrive(0, frame(k * 1001, &[])).expect("video");
        }
        // 1024-sample packets, as ffmpeg writes pcm, past the sixth frame.
        let mut at = 0;
        while at < 6 * 1602 {
            a.arrive(1, samples(at, 1024)).expect("audio");
            at += 1024;
        }
        let ticks = drain(&mut a);
        let counts: Vec<usize> = ticks
            .iter()
            .map(|t| t.streams[1].frames[0].data.len() / 4)
            .collect();
        assert_eq!(counts, vec![1602, 1602, 1601, 1602, 1601]);
        let starts: Vec<i64> = ticks.iter().map(|t| t.streams[1].frames[0].pts).collect();
        assert_eq!(starts, vec![0, 1602, 3204, 4805, 6407]);
    }

    #[test]
    fn an_interval_input_waits_for_progress_past_the_end_and_its_ahead() {
        let words = Pairing::Interval(Interval::shared(None, 0.0));
        let s = shape(
            vec![
                port("v", PortKind::Video, Pairing::Lockstep, RowsUse::Ignore),
                port("w", PortKind::Data, words, RowsUse::PerFrame),
            ],
            Clock::Input("v".into()),
        );
        let tb = TimeBase { num: 1, den: 10 };
        let mut a = Assembler::new(
            &s,
            &[
                bound("v", 0, tb, video()),
                bound("w", 1, MICROS, StreamFormat::Data("json".into())),
            ],
            "m",
            &[],
        )
        .expect("assembler");
        for k in 0..3 {
            a.arrive(0, frame(k, &[])).expect("video");
        }
        a.arrive(1, message(50_000, "{\"a\":1}")).expect("word");
        assert!(drain(&mut a).is_empty(), "no progress past 0.1 s yet");
        a.progress(1, 100_000).expect("progress");
        let ticks = drain(&mut a);
        assert_eq!(ticks.len(), 1);
        assert_eq!(ticks[0].streams[1].messages.len(), 1);
        a.arrive(1, message(150_000, "{\"b\":2}")).expect("word");
        a.progress(1, 200_000).expect("progress");
        let ticks = drain(&mut a);
        assert_eq!(ticks[0].streams[1].messages[0].pts, 150_000);
    }

    #[test]
    fn an_interval_latency_settles_on_the_clock_alone() {
        let words = Pairing::Interval(Interval::shared(Some(0.2), 0.0));
        let s = shape(
            vec![
                port("v", PortKind::Video, Pairing::Lockstep, RowsUse::Ignore),
                port("w", PortKind::Data, words, RowsUse::PerFrame),
            ],
            Clock::Input("v".into()),
        );
        let tb = TimeBase { num: 1, den: 10 };
        let mut a = Assembler::new(
            &s,
            &[
                bound("v", 0, tb, video()),
                bound("w", 1, MICROS, StreamFormat::Data("json".into())),
            ],
            "m",
            &[],
        )
        .expect("assembler");
        for k in 0..3 {
            a.arrive(0, frame(k, &[])).expect("video");
        }
        assert!(drain(&mut a).is_empty(), "the clock is 0.2 s in, 0.1 short");
        a.arrive(0, frame(3, &[])).expect("video");
        assert_eq!(
            drain(&mut a).len(),
            1,
            "0.3 s is the interval's end and its latency"
        );
    }

    #[test]
    fn a_rate_clock_holds_its_input_and_counts_its_ticks() {
        let hold = Pairing::Hold(ffrwd_wasm_runtime::node::Hold {
            anchor: ffrwd_wasm_runtime::node::Anchor::SharedClock,
            lead: 0.0,
            linger: None,
            timeout: None,
            group: None,
            port_param: None,
        });
        let s = shape(
            vec![port("v", PortKind::Video, hold, RowsUse::Ignore)],
            Clock::Rate(Rational { num: 10, den: 1 }),
        );
        let tb = TimeBase { num: 1, den: 20 };
        let mut a = Assembler::new(&s, &[bound("v", 0, tb, video())], "m", &[]).expect("assembler");
        a.arrive(0, frame(0, &[])).expect("video");
        assert!(
            a.next_rate(false).expect("tick").is_none(),
            "nothing past 0 yet"
        );
        a.arrive(0, frame(1, &[])).expect("video");
        let tick = a.next_rate(false).expect("tick").expect("settled");
        assert_eq!((tick.pts, tick.streams[0].frames[0].pts), (0, 0));
        a.arrive(0, frame(5, &[])).expect("video");
        let tick = a.next_rate(false).expect("tick").expect("settled");
        assert_eq!(tick.pts, 1);
        assert_eq!(
            tick.streams[0].frames[0].pts, 1,
            "the newest at or before 0.1 s"
        );
        let tick = a.next_rate(false).expect("tick").expect("settled");
        assert_eq!(
            tick.streams[0].frames[0].pts, 1,
            "held while the source is behind"
        );
    }

    #[test]
    fn a_port_fed_group_hands_its_sound_cut_to_the_tick_beside_its_picture() {
        let hold = |group: &str| {
            Pairing::Hold(ffrwd_wasm_runtime::node::Hold {
                anchor: ffrwd_wasm_runtime::node::Anchor::FirstFrame,
                lead: 0.0,
                linger: None,
                timeout: None,
                group: Some(group.to_string()),
                port_param: Some("port".to_string()),
            })
        };
        let s = shape(
            vec![
                port("v", PortKind::Video, Pairing::Lockstep, RowsUse::Ignore),
                port("feed", PortKind::Video, hold("g"), RowsUse::Ignore),
                port("feed_audio", PortKind::Audio, hold("g"), RowsUse::Ignore),
            ],
            Clock::Input("v".into()),
        );
        let tb = TimeBase { num: 1, den: 30 };
        let mut a = Assembler::new(
            &s,
            &[
                bound("v", 0, tb, video()),
                bound("feed", 1, tb, video()),
                bound("feed_audio", 2, KHZ48, audio()),
            ],
            "m",
            &[1, 2],
        )
        .expect("assembler");
        let source = |base| crate::hold::SourceInfo {
            connection: 1,
            tags: Vec::new(),
            base,
            info: StreamInfo::default(),
        };
        a.source_open(1, source(TimeBase { num: 1, den: 25 }))
            .expect("open");
        a.source_open(2, source(KHZ48)).expect("open");
        for k in 0..3 {
            a.arrive(0, frame(k, &[])).expect("video");
        }
        a.arrive(1, frame(0, &[])).expect("feed");
        a.arrive(2, samples(0, 1024)).expect("sound");
        a.arrive(2, samples(1024, 1024)).expect("sound");
        let ticks = drain(&mut a);
        assert_eq!(ticks.len(), 2);
        let picture = &ticks[0].streams[1];
        assert_eq!(
            picture.feed.as_ref().map(|f| f.start.at),
            Some(0),
            "lead 0: up at once"
        );
        assert_eq!(picture.frames.len(), 1);
        let sound = &ticks[0].streams[2];
        assert_eq!(sound.feed.as_ref().map(|f| f.start.at), Some(0));
        assert_eq!(sound.frames.len(), 1, "{:?}", sound.frames);
        assert_eq!(sound.frames[0].data.len(), 1600 * 4);
        assert_eq!(sound.info.as_ref().map(|(_, base)| *base), Some(KHZ48));
    }

    #[test]
    fn a_data_clock_ticks_once_per_pts_and_a_packets_clock_on_its_running_dts() {
        let s = shape(
            vec![port(
                "d",
                PortKind::Data,
                Pairing::Lockstep,
                RowsUse::PerFrame,
            )],
            Clock::Input("d".into()),
        );
        let mut a = Assembler::new(
            &s,
            &[bound("d", 0, MICROS, StreamFormat::Data("json".into()))],
            "m",
            &[],
        )
        .expect("assembler");
        a.arrive(0, message(0, "{}")).expect("m");
        a.arrive(0, message(0, "{}")).expect("m");
        a.arrive(0, message(5, "{}")).expect("m");
        let ticks = drain(&mut a);
        assert_eq!(ticks.len(), 1);
        assert_eq!(ticks[0].streams[0].messages.len(), 2);

        let s = shape(
            vec![port(
                "p",
                PortKind::Packets,
                Pairing::Lockstep,
                RowsUse::Ignore,
            )],
            Clock::Input("p".into()),
        );
        let coded = StreamFormat::Packets(ffrwd_wasm_runtime::runtime::CodedStream {
            codec: "h264".into(),
            time_base: MICROS,
            format: ffrwd_wasm_runtime::runtime::CodedFormat::Data,
            extradata: Vec::new(),
            profile: None,
            level: None,
        });
        let mut a =
            Assembler::new(&s, &[bound("p", 0, MICROS, coded)], "m", &[]).expect("assembler");
        for (pts, dts) in [(20, Some(0)), (0, Some(10)), (10, None)] {
            a.arrive(
                0,
                Item::Packet(Packet {
                    pts,
                    dts,
                    duration: None,
                    keyframe: false,
                    data: Vec::new(),
                }),
            )
            .expect("packet");
        }
        a.end(0).expect("end");
        let times: Vec<i64> = drain(&mut a).iter().map(|t| t.pts).collect();
        assert_eq!(times, vec![0, 10, 10]);
    }

    #[test]
    fn a_self_clocked_node_is_handed_whatever_arrived_until_every_input_ends() {
        let s = shape(
            vec![port(
                "d",
                PortKind::Data,
                Pairing::Arrival,
                RowsUse::PerFrame,
            )],
            Clock::SelfClocked,
        );
        let mut a = Assembler::new(
            &s,
            &[bound("d", 3, MICROS, StreamFormat::Data("json".into()))],
            "m",
            &[],
        )
        .expect("assembler");
        a.arrive(3, message(40, "{}")).expect("m");
        a.arrive(3, message(10, "{}")).expect("m");
        let tick = a.next_arrivals(5, false).expect("tick");
        let pts: Vec<i64> = tick.streams[0].messages.iter().map(|m| m.pts).collect();
        assert_eq!(pts, vec![40, 10], "as they arrived, not by time");
        assert!(!a.all_ended());
        a.end(3).expect("end");
        assert!(a.all_ended());
        let last = a.next_arrivals(9, true).expect("tick");
        assert!(last.last && last.streams[0].messages.is_empty());
        assert!(a.is_done());
    }

    #[test]
    fn earlier_rows_are_what_an_instance_missed() {
        let mut earlier = EarlierRows::default();
        for n in 0..4u64 {
            earlier.record(1, n, n as i64 * 10, vec![format!("r{n}")]);
        }
        assert!(earlier.owed(0, 1, 0).is_empty());
        earlier.processed(0, 0);
        earlier.processed(1, 1);
        let owed: Vec<i64> = earlier.owed(0, 1, 2).iter().map(|r| r.pts).collect();
        assert_eq!(owed, vec![10], "instance 0 saw tick 0 and missed tick 1");
        let owed: Vec<i64> = earlier.owed(2, 1, 3).iter().map(|r| r.pts).collect();
        assert_eq!(
            owed,
            vec![0, 10, 20],
            "a first call is owed every one before it"
        );
    }

    #[test]
    fn an_instance_that_skipped_the_tick_a_feed_ended_on_hears_of_it_on_its_next_call() {
        let hold = Pairing::Hold(ffrwd_wasm_runtime::node::Hold {
            anchor: ffrwd_wasm_runtime::node::Anchor::SharedClock,
            lead: 0.0,
            linger: None,
            timeout: Some(0.1),
            group: None,
            port_param: Some("port".to_string()),
        });
        let s = shape(
            vec![port("feed", PortKind::Video, hold, RowsUse::Ignore)],
            Clock::Rate(Rational { num: 30, den: 1 }),
        );
        let tb = TimeBase { num: 1, den: 30 };
        let mut a =
            Assembler::new(&s, &[bound("feed", 4, tb, video())], "m", &[4]).expect("assembler");
        a.arrive(4, frame(0, &[])).expect("a frame");
        let mut heard: Vec<Vec<(u64, usize, i64)>> = vec![Vec::new(); 3];
        for _ in 0..10 {
            let tick = a
                .next_rate(false)
                .expect("tick")
                .expect("a port feed never holds one");
            let ordinal = tick.ordinal;
            // Two workers take turns, and a third opens at tick 8.
            let instance = if ordinal >= 8 {
                2
            } else {
                (ordinal % 2) as usize
            };
            for feed in a.ended.owed(instance, 4, ordinal) {
                heard[instance].push((ordinal, instance, feed.ends.expect("ends is set")));
            }
            a.ended.processed(instance, ordinal);
        }
        assert_eq!(
            heard[1],
            vec![(3, 1, 2)],
            "the timeout ends the feed on tick 3, after tick 2"
        );
        assert_eq!(
            heard[0],
            vec![(4, 0, 2)],
            "instance 0 skipped tick 3 and hears of it on 4"
        );
        assert_eq!(
            heard[2],
            vec![(8, 2, 2)],
            "a first call hears of every end before it"
        );
    }

    #[test]
    fn a_first_frame_data_stream_is_restamped_from_the_tick_its_first_message_arrives_on() {
        let mut interval = Interval::shared(None, 0.0);
        interval.anchor = ffrwd_wasm_runtime::node::Anchor::FirstFrame;
        let s = shape(
            vec![
                port("v", PortKind::Video, Pairing::Lockstep, RowsUse::Ignore),
                port(
                    "cues",
                    PortKind::Data,
                    Pairing::Interval(interval),
                    RowsUse::PerFrame,
                ),
            ],
            Clock::Input("v".into()),
        );
        let thirtieths = TimeBase { num: 1, den: 30 };
        let millis = TimeBase { num: 1, den: 1000 };
        let json = StreamFormat::Data("json".into());
        let mut a = Assembler::new(
            &s,
            &[
                bound("v", 0, thirtieths, video()),
                bound("cues", 1, millis, json),
            ],
            "m",
            &[],
        )
        .expect("assembler");
        for k in 0..3 {
            a.arrive(0, frame(k, &[])).expect("a frame");
        }
        let before = drain(&mut a);
        assert_eq!(
            before.iter().map(|t| t.pts).collect::<Vec<_>>(),
            [0, 1],
            "a stream with no message yet holds no tick"
        );
        // An hour and a half into its own origin.
        a.arrive(1, message(5_400_000, "a")).expect("a message");
        a.arrive(1, message(5_400_100, "b")).expect("a message");
        for k in 3..8 {
            a.arrive(0, frame(k, &[])).expect("a frame");
        }
        let after = drain(&mut a);
        let handed = |ticks: &[Tick]| -> Vec<(i64, Vec<(i64, String)>)> {
            ticks
                .iter()
                .map(|t| {
                    let messages = t.streams[1]
                        .messages
                        .iter()
                        .map(|m| (m.pts, String::from_utf8_lossy(&m.data).into_owned()))
                        .collect();
                    (t.pts, messages)
                })
                .collect()
        };
        assert_eq!(
            handed(&after),
            vec![(2, vec![(2, "a".to_string())]), (3, vec![]), (4, vec![])],
            "the first message stands at the tick it arrived on; tick 5 waits for the \
             stream to say it is done past 6"
        );
        a.progress(1, 5_400_200).expect("a progress mark");
        assert_eq!(
            handed(&drain(&mut a)),
            vec![(5, vec![(5, "b".to_string())]), (6, vec![])],
            "0.1 s later on its own origin is three ticks later on the clock's, and its \
             progress is restamped too"
        );
    }
}
