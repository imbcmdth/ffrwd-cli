//! The one loop a node runs in when it rides alone: ticks cut from what its
//! inputs have brought, each handed to the node, and what it emitted handed
//! to the outputs of the ports it names.
//!
//! What arrives, and how it becomes ticks, is the [`Intake`]: coded pads and
//! rows as they arrive (a self-clocked node, or one on a rate clock that is
//! also called when nothing arrives), an adapted data filter's pads
//! released at or before its clock, rows read whole, or nothing at all for
//! a source that pulls its own. Where the ticks go is the [`Outputs`]: each
//! port's stream on a writer of its own, and the rows a node emits beside
//! its ports to the run's rows outputs.

use std::io::Write;
use std::sync::{mpsc, Arc};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use ffrwd_wasm_runtime::node::{Node, Payload, Tick, TickFrame, TickStream};
use ffrwd_wasm_runtime::runtime::{Message, Packet, TimeBase};

use crate::tick::DataDrive;
use crate::{heartbeat, heartbeat_packet, message_packet, PadQueues, RowOutput, RowsQueue};

/// Where a run's ticks come from.
pub enum Intake {
    /// Pads read as they arrive, one stream id per pad. `rows` is the rows
    /// a packet filter reads beside its pads, polled rather than waited on,
    /// and the stream they are handed on. `idle` calls the node with nothing
    /// once that long has passed without a call.
    Arrivals {
        queues: Arc<PadQueues>,
        ids: Vec<u32>,
        data: Vec<bool>,
        rows: Option<(Arc<RowsQueue>, u32)>,
        idle: Option<Duration>,
        base: TimeBase,
    },
    /// An adapted data filter's pads: with a clock, a tick each time the
    /// clock moves on with every message at or before it; without, a tick
    /// per batch of whatever arrived.
    Drive {
        queues: Arc<PadQueues>,
        ids: Vec<u32>,
        data: Vec<bool>,
        /// The clock pad and its time base, if there is one.
        clock: Option<(usize, TimeBase)>,
        drive: DataDrive,
    },
    /// Rows read whole before the node was opened, handed on one tick.
    Rows { id: u32, rows: Option<Vec<String>> },
    /// Nothing arrives: the node makes its own.
    Pull,
}

/// One output port's writer: batches of packets onto a NUT of its own.
struct Track {
    sender: Option<Sender>,
    data: bool,
}

enum Sender {
    Bounded(mpsc::SyncSender<Vec<Packet>>),
    Unbounded(mpsc::Sender<Vec<Packet>>),
}

impl Sender {
    fn send(&self, packets: Vec<Packet>) -> bool {
        match self {
            Sender::Bounded(s) => s.send(packets).is_ok(),
            Sender::Unbounded(s) => s.send(packets).is_ok(),
        }
    }
}

/// An output port not yet opened, with what opens it once the node has
/// settled what its header needs.
pub(crate) type Opener = Box<dyn FnOnce(&dyn Node) -> Result<crate::FrameOutput>>;

/// Where a run's emissions go.
#[derive(Default)]
pub struct Outputs {
    tracks: Vec<Option<Track>>,
    /// Ports opened only once every one of them can be: their headers wait
    /// on the node, and no packet goes anywhere before every header.
    pending: Vec<(usize, Opener, String, bool)>,
    /// Whether a port's messages go to the rows outputs instead of a track.
    to_rows: Vec<bool>,
    rows: Vec<RowOutput>,
    writers: Vec<thread::JoinHandle<Result<()>>>,
    /// Whether every writer still takes what it is handed.
    sending: bool,
    /// Whether every call hands every track a batch, an empty one included,
    /// so each is flushed: a header sits in its buffer until then, and a
    /// reader opening its inputs one at a time waits on it.
    flush_every_call: bool,
}

impl Outputs {
    pub fn new(ports: usize) -> Outputs {
        Outputs {
            tracks: (0..ports).map(|_| None).collect(),
            to_rows: vec![false; ports],
            sending: true,
            ..Outputs::default()
        }
    }

    /// The run's rows outputs: the node's rows, and the messages of every
    /// port handed to [`Outputs::rows_of`].
    pub(crate) fn add_rows(&mut self, output: RowOutput) {
        self.rows.push(output);
    }

    /// Every call flushes every track, whether it wrote anything or not.
    pub fn flush_every_call(&mut self) {
        self.flush_every_call = true;
    }

    /// Port `port`'s messages are rows, written to the rows outputs.
    pub fn rows_of(&mut self, port: usize) {
        self.to_rows[port] = true;
    }

    /// Port `port` written to `muxer` by a thread of its own. A bounded
    /// channel holds one batch, which overlaps a write with the next call
    /// and bounds what a stalled reader can pile up; an unbounded one never
    /// holds the node up on a reader that opens its inputs one at a time.
    pub(crate) fn track(
        &mut self,
        port: usize,
        muxer: crate::FrameOutput,
        spelling: &str,
        bounded: bool,
    ) {
        let data = muxer.stream().is_json();
        let spelling = spelling.to_string();
        let sender = if bounded {
            let (sender, batches) = mpsc::sync_channel::<Vec<Packet>>(1);
            self.writers.push(thread::spawn(move || {
                crate::write_track(muxer, &batches)
                    .with_context(|| format!("writing output {spelling}"))
            }));
            Sender::Bounded(sender)
        } else {
            let (sender, batches) = mpsc::channel::<Vec<Packet>>();
            self.writers.push(thread::spawn(move || {
                crate::write_track(muxer, &batches)
                    .with_context(|| format!("writing output {spelling}"))
            }));
            Sender::Unbounded(sender)
        };
        self.tracks[port] = Some(Track {
            sender: Some(sender),
            data,
        });
    }

    /// Port `port` opened by `opener` once the node can say what its header
    /// carries.
    pub(crate) fn track_later(
        &mut self,
        port: usize,
        opener: Opener,
        spelling: &str,
        bounded: bool,
    ) {
        self.pending
            .push((port, opener, spelling.to_string(), bounded));
    }

    fn open_pending(&mut self, node: &dyn Node) -> Result<()> {
        if self.pending.is_empty() {
            return Ok(());
        }
        let pending = std::mem::take(&mut self.pending);
        let mut opened = Vec::with_capacity(pending.len());
        for (port, opener, spelling, bounded) in pending {
            opened.push((port, opener(node)?, spelling, bounded));
        }
        for (port, muxer, spelling, bounded) in opened {
            self.track(port, muxer, &spelling, bounded);
        }
        Ok(())
    }

    /// One call's emissions: each port's as one batch on its track, then
    /// rows.
    fn route(&mut self, node: &dyn Node, emitted: ffrwd_wasm_runtime::node::Emitted) -> Result<()> {
        self.open_pending(node)?;
        let mut batches: Vec<Vec<Packet>> = self.tracks.iter().map(|_| Vec::new()).collect();
        let mut rows: Vec<String> = Vec::new();
        for item in emitted.items {
            let port = item.port;
            if self.to_rows[port] {
                if let Payload::Message(m) = item.payload {
                    if !m.data.is_empty() {
                        rows.push(String::from_utf8_lossy(&m.data).into_owned());
                    }
                }
                continue;
            }
            let Some(Some(track)) = self.tracks.get(port) else {
                continue;
            };
            let packet = match item.payload {
                Payload::Message(Message { pts, data }) if data.is_empty() && track.data => {
                    heartbeat_packet(pts)
                }
                Payload::Message(Message { pts, data }) => message_packet(pts, data),
                Payload::Packet(p) => p,
                Payload::Frame(_) | Payload::Rows(_) => continue,
            };
            batches[port].push(packet);
        }
        for (port, batch) in batches.into_iter().enumerate() {
            if batch.is_empty() && !self.flush_every_call {
                continue;
            }
            if let Some(Some(Track {
                sender: Some(sender),
                ..
            })) = self.tracks.get(port)
            {
                self.sending &= sender.send(batch);
            }
        }
        rows.extend(emitted.rows);
        for output in &mut self.rows {
            output.write_batch(&rows)?;
        }
        Ok(())
    }

    /// Every writer closed and joined, the rows flushed; the first failure.
    fn finish(mut self) -> Result<()> {
        for track in self.tracks.iter_mut().flatten() {
            track.sender = None;
        }
        for output in &mut self.rows {
            output.flush()?;
        }
        let mut wrote = Ok(());
        for writer in self.writers.drain(..) {
            let written = writer
                .join()
                .unwrap_or_else(|panic| std::panic::resume_unwind(panic));
            if wrote.is_ok() {
                wrote = written;
            }
        }
        wrote
    }
}

/// One node riding alone: its ticks, its outputs, and how a failed call is
/// named.
pub struct Run {
    pub node: Box<dyn Node>,
    pub intake: Intake,
    pub outputs: Outputs,
    /// What a call that is not the final one is said to be doing, for its
    /// failure: "processing packets". None leaves the node's own words.
    pub doing: Option<&'static str>,
    /// Whether the node is still handed its final call when an output's
    /// writer has stopped.
    pub final_on_failure: bool,
}

impl Run {
    pub fn drive(mut self) -> Result<()> {
        let outcome = self.ticks();
        // A reader still waiting for queue space must wake and stop.
        self.intake.close();
        outcome?;
        self.outputs.finish()
    }

    fn call(&mut self, tick: Tick) -> Result<ffrwd_wasm_runtime::node::Emitted> {
        let last = tick.last;
        let result = self.node.process(tick);
        let Some(doing) = self.doing else {
            return result;
        };
        let name = self.node.name().to_string();
        result.with_context(|| {
            if last {
                format!("{name}: the final call")
            } else {
                format!("{name}: {doing}")
            }
        })
    }

    fn ticks(&mut self) -> Result<()> {
        let started = Instant::now();
        let mut called = Instant::now();
        let mut made: i64 = 0;
        loop {
            let ticks = self.intake.next(&mut called, &mut made, started)?;
            for tick in ticks {
                let last = tick.last;
                let emitted = self.call(tick)?;
                called = Instant::now();
                let finished = emitted.finished;
                self.outputs.route(&*self.node, emitted)?;
                if last {
                    return Ok(());
                }
                if finished || !self.outputs.sending {
                    if finished || self.final_on_failure {
                        let base = TimeBase {
                            num: 1,
                            den: 1_000_000,
                        };
                        let final_tick = Tick {
                            pts: started.elapsed().as_micros() as i64,
                            time_base: base,
                            last: true,
                            streams: self.intake.empty_streams(),
                        };
                        let emitted = self.call(final_tick)?;
                        self.outputs.route(&*self.node, emitted)?;
                    }
                    return Ok(());
                }
            }
        }
    }
}

impl Intake {
    /// Wake every reader still waiting so it can stop.
    fn close(&self) {
        match self {
            Intake::Arrivals { queues, rows, .. } => {
                queues.close();
                if let Some((rows, _)) = rows {
                    rows.close();
                }
            }
            Intake::Drive { queues, .. } => queues.close(),
            Intake::Rows { .. } | Intake::Pull => {}
        }
    }

    /// An empty stream per bound stream, for a final call nothing brought.
    fn empty_streams(&self) -> Vec<TickStream> {
        let ids: Vec<u32> = match self {
            Intake::Arrivals { ids, rows, .. } => ids
                .iter()
                .copied()
                .chain(rows.as_ref().map(|(_, id)| *id))
                .collect(),
            Intake::Drive { ids, .. } => ids.clone(),
            Intake::Rows { id, .. } => vec![*id],
            Intake::Pull => Vec::new(),
        };
        ids.into_iter()
            .map(|id| TickStream {
                id,
                ..TickStream::default()
            })
            .collect()
    }

    /// The ticks the next batch makes, in order; the last of a run carries
    /// `last`.
    fn next(
        &mut self,
        called: &mut Instant,
        made: &mut i64,
        started: Instant,
    ) -> Result<Vec<Tick>> {
        match self {
            Intake::Arrivals {
                queues,
                ids,
                data,
                rows,
                idle,
                base,
            } => loop {
                let deadline = idle.map(|idle| *called + idle);
                let (mut carried, last) = queues.take_until(deadline)?;
                let heard = crate::drop_heartbeats(&mut carried, data);
                let arrived: Vec<String> = match rows {
                    Some((rows, _)) if last => rows.drain_to_end()?,
                    Some((rows, _)) => rows.take()?.0,
                    None => Vec::new(),
                };
                if let Some(idle) = idle {
                    let quiet = carried.iter().all(Vec::is_empty) && arrived.is_empty();
                    if !last && quiet && called.elapsed() < *idle {
                        continue;
                    }
                }
                let mut streams: Vec<TickStream> = Vec::with_capacity(ids.len() + 1);
                for (pad, packets) in carried.into_iter().enumerate() {
                    let mut stream = TickStream {
                        id: ids[pad],
                        progress: heard[pad],
                        ..TickStream::default()
                    };
                    if data[pad] {
                        stream.messages = packets
                            .into_iter()
                            .map(|p| Message {
                                pts: p.pts,
                                data: p.data,
                            })
                            .collect();
                    } else {
                        stream.packets = packets;
                    }
                    streams.push(stream);
                }
                if let Some((_, id)) = rows {
                    streams.push(TickStream {
                        id: *id,
                        messages: arrived
                            .into_iter()
                            .map(|row| Message {
                                pts: 0,
                                data: row.into_bytes(),
                            })
                            .collect(),
                        ..TickStream::default()
                    });
                }
                let pts = match idle {
                    Some(_) => *made,
                    None => started.elapsed().as_micros() as i64,
                };
                *made += 1;
                return Ok(vec![Tick {
                    pts,
                    time_base: *base,
                    last,
                    streams,
                }]);
            },
            Intake::Drive {
                queues,
                ids,
                data,
                clock,
                drive,
            } => {
                let (carried, ended, last) = queues.take_ended()?;
                match clock {
                    Some((clock, base)) => {
                        let calls = drive.arrive(carried, &ended, last);
                        Ok(calls
                            .into_iter()
                            .map(|call| {
                                let streams = ids
                                    .iter()
                                    .enumerate()
                                    .map(|(pad, id)| {
                                        let mut stream = TickStream {
                                            id: *id,
                                            ..TickStream::default()
                                        };
                                        if pad == *clock {
                                            stream.frames = call
                                                .now
                                                .map(|now| TickFrame {
                                                    pts: now,
                                                    duration: None,
                                                    data: Arc::new(Vec::new()),
                                                    rows: Vec::new(),
                                                })
                                                .into_iter()
                                                .collect();
                                        } else {
                                            stream.messages = call.input[pad].clone();
                                        }
                                        stream
                                    })
                                    .collect();
                                Tick {
                                    pts: call.now.unwrap_or(0),
                                    time_base: *base,
                                    last: call.last,
                                    streams,
                                }
                            })
                            .collect())
                    }
                    None => {
                        let mut streams = Vec::with_capacity(ids.len());
                        for (pad, packets) in carried.into_iter().enumerate() {
                            let mut stream = TickStream {
                                id: ids[pad],
                                ..TickStream::default()
                            };
                            if data[pad] {
                                stream.progress = packets.last().map(|p| p.pts);
                                stream.messages = packets
                                    .into_iter()
                                    .filter(|p| !heartbeat::is_heartbeat(&p.data))
                                    .map(|p| Message {
                                        pts: p.pts,
                                        data: p.data,
                                    })
                                    .collect();
                            }
                            streams.push(stream);
                        }
                        *made += 1;
                        Ok(vec![Tick {
                            pts: 0,
                            time_base: TimeBase {
                                num: 1,
                                den: 1_000_000,
                            },
                            last,
                            streams,
                        }])
                    }
                }
            }
            Intake::Rows { id, rows } => {
                let base = TimeBase {
                    num: 1,
                    den: 1_000_000,
                };
                Ok(match rows.take() {
                    Some(rows) => vec![
                        Tick {
                            pts: 0,
                            time_base: base,
                            last: false,
                            streams: vec![TickStream {
                                id: *id,
                                messages: rows
                                    .into_iter()
                                    .map(|row| Message {
                                        pts: 0,
                                        data: row.into_bytes(),
                                    })
                                    .collect(),
                                ..TickStream::default()
                            }],
                        },
                        Tick {
                            pts: 0,
                            time_base: base,
                            last: true,
                            streams: vec![TickStream {
                                id: *id,
                                ..TickStream::default()
                            }],
                        },
                    ],
                    None => Vec::new(),
                })
            }
            Intake::Pull => {
                *made += 1;
                Ok(vec![Tick {
                    pts: started.elapsed().as_micros() as i64,
                    time_base: TimeBase {
                        num: 1,
                        den: 1_000_000,
                    },
                    last: false,
                    streams: Vec::new(),
                }])
            }
        }
    }
}
