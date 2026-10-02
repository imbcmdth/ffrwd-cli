//! The wire at a node network's edges.
//!
//! An `-i` is one NUT carrying every stream of its edge: its headers are read
//! first, every input at once, and then a thread per input demuxes the rest
//! and hands each frame on. An output is one thread writing from a queue of
//! its own: one NUT carrying every label mapped to it, a data label as
//! NDJSON, cues as a subtitle document, or nothing. Several streams in one
//! NUT are written in time order across streams, so what one network writes
//! is the same bytes however its lanes were scheduled.

use std::collections::VecDeque;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock};
use std::thread;

use anyhow::{anyhow, bail, Context, Result};
use ffrwd_wasm::nut::{self, Event, Limits, PushDemuxer};
use ffrwd_wasm_runtime::runtime::{self, Message, Packet, TimeBase};

use crate::heartbeat::{self, Beats};
use crate::tick::compare;
use crate::{open_input, open_output, subtitles, InputPath, InputReader, OutputPath, OutputWriter};

/// One input with its headers read and its frames still to come.
pub struct Opened {
    pub streams: Vec<nut::Stream>,
    reader: BufReader<InputReader>,
    demuxer: PushDemuxer,
}

const READ_CHUNK: usize = 1 << 16;

/// Opens every input and reads its headers, all at once: one producer's
/// first bytes can wait on another's whole chain starting up.
pub fn open_inputs(paths: &[InputPath]) -> Result<Vec<Opened>> {
    thread::scope(|scope| {
        let handles: Vec<_> = paths
            .iter()
            .enumerate()
            .map(|(index, path)| {
                scope.spawn(move || {
                    open_one(path)
                        .with_context(|| format!("reading the NUT headers of input {index}"))
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| {
                h.join()
                    .unwrap_or_else(|panic| std::panic::resume_unwind(panic))
            })
            .collect()
    })
}

fn open_one(path: &InputPath) -> Result<Opened> {
    let mut reader = BufReader::with_capacity(1 << 20, open_input(path)?);
    let mut demuxer = PushDemuxer::new(Limits::default());
    let mut chunk = vec![0u8; READ_CHUNK];
    loop {
        match demuxer.next_event()? {
            Some(Event::EndOfHeaders) | Some(Event::EndOfInput) => break,
            Some(_) => {}
            None => {
                let n = reader.read(&mut chunk)?;
                if n == 0 {
                    demuxer.finish();
                } else {
                    demuxer.feed(&chunk[..n]);
                }
            }
        }
    }
    let streams = demuxer
        .streams()
        .iter()
        .enumerate()
        .map(|(index, s)| {
            s.clone()
                .ok_or_else(|| anyhow!("stream {index} was declared and its header never came"))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Opened {
        streams,
        reader,
        demuxer,
    })
}

impl Opened {
    /// Every frame after the headers, handed to `each` as its stream index,
    /// its packet and its payload, until the input ends or `each` says to
    /// stop.
    pub fn pump(
        mut self,
        mut each: impl FnMut(usize, nut::Packet, &[u8]) -> Result<bool>,
    ) -> Result<()> {
        let mut chunk = vec![0u8; READ_CHUNK];
        loop {
            match self.demuxer.next_event()? {
                Some(Event::Frame { stream, packet }) => {
                    if !each(stream, packet, self.demuxer.payload())? {
                        return Ok(());
                    }
                }
                Some(Event::EndOfInput) => return Ok(()),
                Some(_) => {}
                None => {
                    let n = self.reader.read(&mut chunk)?;
                    if n == 0 {
                        self.demuxer.finish();
                    } else {
                        self.demuxer.feed(&chunk[..n]);
                    }
                }
            }
        }
    }
}

/// One thing an output is handed for one of its streams.
#[derive(Debug, Clone)]
pub enum Out {
    Frame {
        pts: i64,
        data: Arc<Vec<u8>>,
        /// The rows riding it, between older modules.
        rows: Vec<String>,
    },
    Packet(Packet),
    Message(Message),
    /// Nothing more on the stream before this pts.
    Progress(i64),
    /// Everything one call of an older module made has been handed over.
    Batch,
    /// An older module's rows with no frame to ride, before its end.
    Trailing(Vec<String>),
    End,
}

/// What an output writes.
pub enum Target {
    /// One NUT, a stream per label in the order they were mapped.
    Nut(Vec<nut::Stream>),
    /// One data label, a message per line, stamped with its time in the
    /// label's time base where it has one.
    Ndjson(Option<TimeBase>),
    /// One data label of cues, written whole once it ends.
    Subtitles(subtitles::Format),
    Null,
}

#[derive(Default)]
struct QueueState {
    items: VecDeque<(usize, Out)>,
    /// The reader went away: what is handed on is dropped.
    closed: bool,
    failed: Option<String>,
    /// Everything it was handed is written and the output is closed.
    done: bool,
}

/// An output's queue: pushing never waits, and the scheduler reads its
/// length as credit before it hands on more.
pub struct Queue {
    state: Mutex<QueueState>,
    ready: Condvar,
}

impl Queue {
    fn lock(&self) -> MutexGuard<'_, QueueState> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub fn push(&self, stream: usize, out: Out) {
        let mut state = self.lock();
        if state.closed || state.failed.is_some() || state.done {
            return;
        }
        state.items.push_back((stream, out));
        self.ready.notify_all();
    }

    pub fn len(&self) -> usize {
        self.lock().items.len()
    }

    /// Whether this output takes nothing more: its reader went away, it
    /// failed, or it is written whole.
    pub fn settled(&self) -> bool {
        let state = self.lock();
        state.closed || state.failed.is_some() || state.done
    }

    pub fn failure(&self) -> Option<String> {
        self.lock().failed.clone()
    }
}

pub struct Writer {
    pub queue: Arc<Queue>,
    handle: Option<thread::JoinHandle<()>>,
}

impl Writer {
    pub fn join(&mut self) {
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

/// What the writers of one run share about its end: how many are still
/// writing, and whether the run has anything left to make.
#[derive(Default)]
pub struct Ending {
    writing: AtomicUsize,
    over: OnceLock<Arc<dyn Fn() -> bool + Send + Sync>>,
}

impl Ending {
    /// Says how to ask whether the run has anything left to make.
    pub fn set_over(&self, over: Arc<dyn Fn() -> bool + Send + Sync>) {
        let _ = self.over.set(over);
    }

    fn over(&self) -> bool {
        self.over.get().is_some_and(|over| over())
    }
}

/// The last output a run finishes, left open for the process's exit to
/// close: its reader sees the end only once the process is going, never
/// while it tears its modules down. Any earlier one closes as it finishes,
/// since a reader of two outputs may need the first one's end to take the
/// rest of the second.
static PARKED: Mutex<Vec<OutputWriter>> = Mutex::new(Vec::new());

/// Starts the thread that writes `target` to `path`. `wake` is called each
/// time something leaves the queue, without the queue's lock held.
pub fn spawn_writer(
    path: OutputPath,
    spelling: String,
    target: Target,
    wake: Arc<dyn Fn() + Send + Sync>,
    ending: Arc<Ending>,
) -> Writer {
    let queue = Arc::new(Queue {
        state: Mutex::new(QueueState::default()),
        ready: Condvar::new(),
    });
    let shared = Arc::clone(&queue);
    ending.writing.fetch_add(1, Ordering::SeqCst);
    let handle = thread::spawn(move || {
        let result = write_all(&path, target, &shared, &*wake);
        let last = ending.writing.fetch_sub(1, Ordering::SeqCst) == 1;
        let result = result.map(|out| {
            if last && ending.over() {
                PARKED.lock().unwrap_or_else(|e| e.into_inner()).push(out);
            }
        });
        {
            let mut state = shared.lock();
            match result {
                Ok(()) => state.done = true,
                Err(e) if is_closed(&e) => state.closed = true,
                Err(e) => state.failed = Some(format!("writing output {spelling}: {e:#}")),
            }
            state.items.clear();
        }
        wake();
    });
    Writer {
        queue,
        handle: Some(handle),
    }
}

/// Whether a write failed because the reader closed its end.
fn is_closed(error: &anyhow::Error) -> bool {
    error.chain().any(|cause| {
        cause
            .downcast_ref::<io::Error>()
            .is_some_and(|e| e.kind() == io::ErrorKind::BrokenPipe || crate::is_pipe_eof(e))
    })
}

/// The next thing handed to the queue, or None once nothing will be.
fn next(queue: &Queue, wake: &dyn Fn()) -> Option<(usize, Out)> {
    let item = {
        let mut state = queue.lock();
        loop {
            if let Some(item) = state.items.pop_front() {
                break Some(item);
            }
            if state.closed || state.failed.is_some() || state.done {
                break None;
            }
            state = queue.ready.wait(state).unwrap_or_else(|e| e.into_inner());
        }
    };
    wake();
    item
}

/// Writes everything `target` is handed and hands back the output, flushed
/// and still open.
fn write_all(
    path: &OutputPath,
    target: Target,
    queue: &Queue,
    wake: &dyn Fn(),
) -> Result<OutputWriter> {
    let writer = BufWriter::with_capacity(1 << 20, open_output(path)?);
    match target {
        Target::Nut(streams) => {
            let live = crate::output_is_pipe(path);
            let writer = write_nut(writer, streams, live, queue, wake)?;
            Ok(writer.into_inner().map_err(|e| e.into_error())?)
        }
        Target::Ndjson(base) => {
            let mut writer = writer;
            while let Some((_, out)) = next(queue, wake) {
                match out {
                    Out::Message(m) => {
                        match base {
                            Some(base) => {
                                let base = nut::TimeBase {
                                    num: base.num,
                                    den: base.den,
                                };
                                let row = String::from_utf8_lossy(&m.data);
                                writer.write_all(crate::stamp_row(&row, m.pts, base).as_bytes())?;
                            }
                            None => writer.write_all(&m.data)?,
                        }
                        writer.write_all(b"\n")?;
                        writer.flush()?;
                    }
                    Out::End => break,
                    _ => {}
                }
            }
            Ok(writer.into_inner().map_err(|e| e.into_error())?)
        }
        Target::Subtitles(format) => {
            let mut document = subtitles::Document::new(format);
            while let Some((_, out)) = next(queue, wake) {
                match out {
                    Out::Message(m) => {
                        document.push_row(&String::from_utf8_lossy(&m.data), format.name())?
                    }
                    Out::End => break,
                    _ => {}
                }
            }
            let mut writer = writer;
            writer.write_all(document.render().as_bytes())?;
            Ok(writer.into_inner().map_err(|e| e.into_error())?)
        }
        Target::Null => {
            while let Some((_, out)) = next(queue, wake) {
                if matches!(out, Out::End) {
                    break;
                }
            }
            Ok(writer.into_inner().map_err(|e| e.into_error())?)
        }
    }
}

/// One stream of a NUT output as the interleaver holds it.
struct Track {
    base: TimeBase,
    coded: bool,
    /// For a data stream, its timeline as written.
    beats: Option<Beats>,
    held: VecDeque<Out>,
    /// The newest time the stream has said it is done to.
    settled: Option<i64>,
    ended: bool,
}

fn time_of(out: &Out) -> Option<i64> {
    match out {
        Out::Frame { pts, .. } => Some(*pts),
        Out::Packet(p) => Some(p.dts.unwrap_or(p.pts)),
        Out::Message(m) => Some(m.pts),
        Out::Progress(p) => Some(*p),
        Out::Batch | Out::Trailing(_) | Out::End => None,
    }
}

/// `live` where something reads the output as it is written, which is who
/// a data stream's heartbeats are for; a file is read once it is whole, and
/// carries the messages alone.
fn write_nut<W: Write>(
    writer: W,
    streams: Vec<nut::Stream>,
    live: bool,
    queue: &Queue,
    wake: &dyn Fn(),
) -> Result<W> {
    let mut tracks: Vec<Track> = streams
        .iter()
        .map(|s| Track {
            base: TimeBase {
                num: s.time_base.num,
                den: s.time_base.den,
            },
            coded: s.codec_name().is_some(),
            beats: s.is_json().then(|| {
                Beats::new(TimeBase {
                    num: s.time_base.num,
                    den: s.time_base.den,
                })
            }),
            held: VecDeque::new(),
            settled: None,
            ended: false,
        })
        .collect();
    let mut muxer = nut::Muxer::with_streams(writer, &streams).context("writing the NUT header")?;
    while !tracks.iter().all(|t| t.ended && t.held.is_empty()) {
        let Some((stream, out)) = next(queue, wake) else {
            break;
        };
        let track = &mut tracks[stream];
        if let Some(time) = time_of(&out) {
            track.settled = Some(track.settled.map_or(time, |s| s.max(time)));
        }
        match out {
            Out::End => track.ended = true,
            Out::Batch | Out::Trailing(_) => {}
            out => track.held.push_back(out),
        }
        while let Some(index) = writable(&tracks) {
            let out = tracks[index]
                .held
                .pop_front()
                .expect("writable has one held");
            write_one(&mut muxer, index, &mut tracks[index], out, live)?;
        }
    }
    muxer.finish()?;
    Ok(muxer.into_inner())
}

/// The stream whose held item is next in time across every stream, once no
/// other stream can still bring one earlier; ties go to the lower stream.
fn writable(tracks: &[Track]) -> Option<usize> {
    let mut best: Option<(usize, i64)> = None;
    for (index, track) in tracks.iter().enumerate() {
        let Some(time) = track.held.front().and_then(time_of) else {
            continue;
        };
        let earlier = match best {
            None => true,
            Some((b, bt)) => compare(time, track.base, bt, tracks[b].base).is_lt(),
        };
        if earlier {
            best = Some((index, time));
        }
    }
    let (index, time) = best?;
    let base = tracks[index].base;
    let blocked = tracks.iter().enumerate().any(|(other, track)| {
        if other == index || track.ended || !track.held.is_empty() {
            return false;
        }
        match track.settled {
            None => true,
            Some(settled) => match compare(settled, track.base, time, base) {
                std::cmp::Ordering::Less => true,
                std::cmp::Ordering::Equal => other < index,
                std::cmp::Ordering::Greater => false,
            },
        }
    });
    (!blocked).then_some(index)
}

fn write_one<W: Write>(
    muxer: &mut nut::Muxer<W>,
    index: usize,
    track: &mut Track,
    out: Out,
    live: bool,
) -> Result<()> {
    match out {
        Out::Frame { pts, data, .. } => {
            if track.coded {
                bail!("a frame reached output stream {index}, which carries coded packets");
            }
            muxer.write_frame_to(index, pts, &data)?;
        }
        Out::Packet(packet) => {
            let framed = nut::Packet {
                pts: packet.pts,
                dts: packet.dts,
                keyframe: packet.keyframe,
            };
            muxer.write_coded_to(index, &framed, &packet.data)?;
        }
        Out::Message(message) => {
            let pts = match &mut track.beats {
                Some(beats) => beats.place(message.pts),
                None => message.pts,
            };
            let framed = nut::Packet {
                pts,
                dts: Some(pts),
                keyframe: true,
            };
            muxer.write_coded_to(index, &framed, &message.data)?;
        }
        Out::Progress(_) if !live => return Ok(()),
        Out::Progress(pts) => {
            let base = track.base;
            if let Some(due) = track.beats.as_mut().and_then(|b| b.due(pts, base)) {
                let framed = nut::Packet {
                    pts: due,
                    dts: Some(due),
                    keyframe: true,
                };
                muxer.write_coded_to(index, &framed, heartbeat::PAYLOAD)?;
            } else {
                return Ok(());
            }
        }
        Out::Batch | Out::Trailing(_) | Out::End => return Ok(()),
    }
    muxer.flush()?;
    Ok(())
}

/// Starts the thread that hands one older module's output to `sink`, a call
/// at a time, as that world's host wrote it: the call's frames, then its
/// rows with no frame to ride, then the end.
pub fn spawn_sink(
    mut sink: crate::Sink,
    module: String,
    wake: Arc<dyn Fn() + Send + Sync>,
) -> Writer {
    let queue = Arc::new(Queue {
        state: Mutex::new(QueueState::default()),
        ready: Condvar::new(),
    });
    let shared = Arc::clone(&queue);
    let handle = thread::spawn(move || {
        let mut batch: Vec<runtime::Frame> = Vec::new();
        let result = (|| -> Result<()> {
            while let Some((_, out)) = next(&shared, &*wake) {
                match out {
                    Out::Frame { pts, data, rows } => {
                        batch.push(runtime::Frame { pts, data, rows })
                    }
                    Out::Batch => sink.write(&module, &std::mem::take(&mut batch))?,
                    Out::Trailing(rows) => sink.write_trailing(&rows)?,
                    Out::End => return sink.finish(),
                    _ => {}
                }
            }
            Ok(())
        })();
        {
            let mut state = shared.lock();
            match result {
                Ok(()) => state.done = true,
                Err(e) => state.failed = Some(format!("{e:#}")),
            }
            state.items.clear();
        }
        wake();
    });
    Writer {
        queue,
        handle: Some(handle),
    }
}
