//! The lane scheduler: a fixed pool of worker threads driving every node of
//! the network at once.
//!
//! Every node is a LANE. A lane cuts its arriving stream into windows -
//! serially, since a window is consecutive frames - and queues each window as
//! a task carrying an ordinal. Workers pull tasks from whichever ready lane
//! has the longest queue. A pure video node admits tasks concurrently, up to
//! one instance per worker, instances opened lazily under pressure; every
//! other lane - impure, audio, the host's own rows node - admits one task at
//! a time, in ordinal order, executed by whichever worker takes it.
//!
//! Output is byte-identical at every worker count: each lane reassembles its
//! results by ordinal before anything leaves it, so downstream cutters, row
//! consumers and sinks always see the serial order. Rows ride inside their
//! carrier frames, so their order is the frames' order.
//!
//! Backpressure is credit, not blocking: a task is dispatched only while
//! every lane downstream of it has queue room, and the feeder waits for room
//! on the lanes that read its input. Completions always land, so a worker
//! never blocks holding work, and the graph being acyclic is what rules a
//! standstill out.

use std::collections::{BTreeMap, HashSet, VecDeque};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread;

use anyhow::{anyhow, bail, Context, Result};
use ffrwd_wasm_runtime::runtime::{Filter, Format, Frame, Processed, Shape, StreamInfo};

use crate::network::Source;
use crate::rowfilter::RowFilter;
use crate::windows::Windows;
use crate::Sink;

/// How many worker threads a run gets: the machine's effective core count,
/// capped by `-jobs` when it was given. `-jobs 1` is the serial escape hatch.
pub fn worker_count(jobs: Option<usize>) -> usize {
    let cores = thread::available_parallelism().map_or(1, |n| n.get());
    jobs.unwrap_or(cores).min(cores).max(1)
}

/// How to open one more instance of a lane's module, for a pure lane growing
/// under pressure.
pub struct Reopen {
    pub path: String,
    pub params: String,
    pub format: Format,
    pub info: StreamInfo,
}

/// A stand-in for a module instance, for driving the scheduler without wasm.
#[cfg(test)]
type StubFn = Box<dyn FnMut(&[Frame], &[String], bool) -> Result<Processed> + Send>;

/// What executes one lane's tasks. A module instance is one wasmtime store;
/// the rows node is the host's own and needs none.
pub enum Runner {
    Module(Box<Filter>),
    Rows(RowFilter),
    #[cfg(test)]
    Stub(StubFn),
}

impl Runner {
    fn run(&mut self, frames: &[Frame], trailing: &[String], last: bool) -> Result<Processed> {
        match self {
            Runner::Module(filter) => filter.process_window(frames, trailing, last),
            Runner::Rows(rows) => {
                let out = frames.iter().cloned().map(|f| rows.pass(f)).collect();
                let trailing = if last {
                    rows.keep(trailing.to_vec())
                } else {
                    Vec::new()
                };
                Ok(Processed {
                    frames: out,
                    trailing,
                })
            }
            #[cfg(test)]
            Runner::Stub(f) => f(frames, trailing, last),
        }
    }
}

/// One node of the opened network, as the scheduler takes it over.
pub struct LaneSeed {
    pub name: String,
    /// The instances opened so far; ordinarily one.
    pub runners: Vec<Runner>,
    pub shape: Shape,
    pub sources: Vec<Source>,
    pub format: Format,
    /// Set only for a lane that may grow more instances.
    pub reopen: Option<Reopen>,
}

impl LaneSeed {
    /// How many tasks this lane admits at once. Only a pure module over video
    /// frames spreads across workers: an impure one's calls are ordered by
    /// its own declaration, and an audio module's output is checked as one
    /// continuous run of samples per instance, so it keeps one.
    fn width(&self, workers: usize) -> usize {
        let module = !matches!(self.runners.first(), Some(Runner::Rows(_)));
        if module && self.shape.pure && self.format.video().is_some() {
            workers
        } else {
            1
        }
    }
}

/// One window on its way to a worker. `last` marks the lane's final call,
/// which carries the tail the strides left over and the trailing rows.
struct Task {
    ordinal: u64,
    frames: Vec<Frame>,
    trailing: Vec<String>,
    last: bool,
}

struct Lane {
    name: String,
    // Intake: where arriving frames wait to become tasks.
    sources: Vec<Source>,
    /// One buffer per pad, used only by a lane reading several streams: a
    /// pad's frames wait here until every other pad has one at the same
    /// timestamp.
    pads: Vec<VecDeque<Frame>>,
    pad_eof: Vec<bool>,
    /// The window this lane buffers toward, for a lane reading one stream.
    windows: Windows,
    /// Rows that arrived on an input wire with no frame to ride, per pad.
    trailing_by_pad: Vec<Vec<String>>,
    /// Trailing rows each upstream lane ended with, keyed by lane index so
    /// they concatenate in topological order.
    trailing_upstream: BTreeMap<usize, Vec<String>>,
    // Tasks: cut windows waiting for a worker, and what is in flight.
    queue: VecDeque<Task>,
    next_ordinal: u64,
    final_ordinal: Option<u64>,
    in_flight: usize,
    /// Concurrent tasks this lane admits: the pool size for a pure video
    /// module, 1 for everything else.
    width: usize,
    idle: Vec<Runner>,
    created: usize,
    reopen: Option<Reopen>,
    // Output: results reassembled by ordinal before anything leaves.
    done: BTreeMap<u64, Processed>,
    next_flush: u64,
    /// The lanes reading this one, as `(lane, pad)`.
    downstream: Vec<(usize, usize)>,
    sinks: Vec<Sink>,
    flushed_final: bool,
}

struct State {
    lanes: Vec<Lane>,
    error: Option<anyhow::Error>,
    finished: bool,
}

struct Shared {
    state: Mutex<State>,
    /// Workers wait here for a dispatchable task.
    work: Condvar,
    /// The feeder waits here for queue room on the lanes it feeds.
    space: Condvar,
    /// Per-lane queue bound, the credit unit.
    cap: usize,
}

impl Shared {
    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }
}

impl State {
    /// Frames arriving on one pad of one lane, cut into whatever tasks they
    /// complete.
    fn push_frames(&mut self, lane_idx: usize, pad: usize, frames: &[Frame]) -> Result<()> {
        let lane = &mut self.lanes[lane_idx];
        if lane.sources.len() == 1 {
            for frame in frames {
                for window in lane.windows.push(frame.clone(), &lane.name)? {
                    lane.queue.push_back(Task {
                        ordinal: lane.next_ordinal,
                        frames: window,
                        trailing: Vec::new(),
                        last: false,
                    });
                    lane.next_ordinal += 1;
                }
            }
            return Ok(());
        }
        lane.pads[pad].extend(frames.iter().cloned());
        while lane.pads.iter().all(|q| !q.is_empty()) {
            let call = take_lockstep(lane)?;
            lane.queue.push_back(Task {
                ordinal: lane.next_ordinal,
                frames: call,
                trailing: Vec::new(),
                last: false,
            });
            lane.next_ordinal += 1;
        }
        Ok(())
    }

    /// One pad's stream has ended. When they all have, the lane's final task
    /// is queued: the tail the strides left over, and every trailing row that
    /// reached this lane.
    fn mark_pad_eof(&mut self, lane_idx: usize, pad: usize) -> Result<()> {
        let lane = &mut self.lanes[lane_idx];
        lane.pad_eof[pad] = true;
        if !lane.pad_eof.iter().all(|eof| *eof) {
            return Ok(());
        }
        for (pad, queue) in lane.pads.iter().enumerate() {
            if !queue.is_empty() {
                bail!(
                    "{}: pad {pad} ends with {} frame(s) that never paired with the other pads; \
                     a module reading several streams reads them in lockstep",
                    lane.name,
                    queue.len()
                );
            }
        }
        let mut trailing: Vec<String> = lane.trailing_by_pad.iter().flatten().cloned().collect();
        for rows in lane.trailing_upstream.values() {
            trailing.extend(rows.iter().cloned());
        }
        let frames = lane.windows.tail();
        lane.final_ordinal = Some(lane.next_ordinal);
        lane.queue.push_back(Task {
            ordinal: lane.next_ordinal,
            frames,
            trailing,
            last: true,
        });
        lane.next_ordinal += 1;
        Ok(())
    }

    /// Whether lane `i` has a task a worker may start right now: input
    /// queued, admission under its width, an instance in hand or in reach,
    /// and queue room on every lane downstream. The final task also waits
    /// for the lane's other tasks to finish, so it reaches one instance and
    /// reaches it last.
    fn dispatchable(&self, i: usize, cap: usize) -> bool {
        let lane = &self.lanes[i];
        let Some(task) = lane.queue.front() else {
            return false;
        };
        if lane.in_flight >= lane.width {
            return false;
        }
        if task.last && lane.in_flight > 0 {
            return false;
        }
        if lane.idle.is_empty() && !(lane.created < lane.width && lane.reopen.is_some()) {
            return false;
        }
        lane.downstream
            .iter()
            .all(|(j, _)| self.lanes[*j].queue.len() < cap)
    }

    /// The dispatchable lane with the longest queue, if any is.
    fn pick(&self, cap: usize) -> Option<usize> {
        (0..self.lanes.len())
            .filter(|i| self.dispatchable(*i, cap))
            .max_by_key(|i| self.lanes[*i].queue.len())
    }

    /// Takes lane `i`'s front task. `None` for the runner means the worker
    /// opens a fresh instance from the returned `Reopen` outside the lock.
    fn dispatch(&mut self, i: usize) -> (Task, Option<Runner>, Option<Reopen>) {
        let lane = &mut self.lanes[i];
        let task = lane.queue.pop_front().expect("picked lanes have a task");
        lane.in_flight += 1;
        match lane.idle.pop() {
            Some(runner) => (task, Some(runner), None),
            None => {
                lane.created += 1;
                let reopen = lane
                    .reopen
                    .as_ref()
                    .expect("dispatchable without an instance only with reopen");
                (
                    task,
                    None,
                    Some(Reopen {
                        path: reopen.path.clone(),
                        params: reopen.params.clone(),
                        format: reopen.format,
                        info: reopen.info.clone(),
                    }),
                )
            }
        }
    }

    /// One task's result lands: the instance goes back to the pool, the
    /// result waits under its ordinal, and everything now contiguous flushes.
    fn complete(
        &mut self,
        i: usize,
        ordinal: u64,
        last: bool,
        runner: Option<Runner>,
        result: Result<Processed>,
    ) {
        let lane = &mut self.lanes[i];
        lane.in_flight -= 1;
        // The final call happens once per instance, so the instance that took
        // it retires rather than rejoining the pool.
        if let Some(runner) = runner {
            if !last {
                lane.idle.push(runner);
            }
        }
        let flushed = match result {
            Ok(processed) => {
                lane.done.insert(ordinal, processed);
                self.flush(i)
            }
            Err(e) => Err(e),
        };
        if let Err(e) = flushed {
            if self.error.is_none() {
                self.error = Some(e);
            }
        }
    }

    /// Flushes lane `i`'s completed results in ordinal order: to its sinks,
    /// and into the lanes that read it. The final result also carries the
    /// lane's trailing rows on, and ends the streams downstream.
    fn flush(&mut self, i: usize) -> Result<()> {
        loop {
            let lane = &mut self.lanes[i];
            let next = lane.next_flush;
            let Some(processed) = lane.done.remove(&next) else {
                return Ok(());
            };
            lane.next_flush += 1;
            let is_final = lane.final_ordinal == Some(next);
            let Processed { frames, trailing } = processed;

            for sink in &mut lane.sinks {
                sink.write(&lane.name, &frames)?;
            }
            if is_final {
                for sink in &mut lane.sinks {
                    sink.write_trailing(&trailing)?;
                    sink.finish()?;
                }
                lane.flushed_final = true;
            }

            let downstream = lane.downstream.clone();
            for (j, pad) in &downstream {
                self.push_frames(*j, *pad, &frames)?;
            }
            if is_final {
                let mut handed: HashSet<usize> = HashSet::new();
                for (j, pad) in &downstream {
                    if handed.insert(*j) {
                        self.lanes[*j].trailing_upstream.insert(i, trailing.clone());
                    }
                    self.mark_pad_eof(*j, *pad)?;
                }
                if self.lanes.iter().all(|l| l.flushed_final) {
                    self.finished = true;
                }
            }
        }
    }
}

/// One frame off every pad, at one timestamp, the way a module reading
/// several streams is called. Rows ride pad 0; what arrived on any other pad
/// is dropped here.
fn take_lockstep(lane: &mut Lane) -> Result<Vec<Frame>> {
    let head = lane.pads[0]
        .front()
        .map(|f| f.pts)
        .ok_or_else(|| anyhow!("{}: a queue emptied mid-window", lane.name))?;
    for (pad, queue) in lane.pads.iter().enumerate().skip(1) {
        let other = queue.front().map(|f| f.pts).unwrap_or(head);
        if other != head {
            bail!(
                "{}: pad 0 is at pts {head} and pad {pad} at pts {other}; a module reading \
                 several streams reads them in lockstep, one frame per pad at one timestamp",
                lane.name
            );
        }
    }
    let mut call = Vec::with_capacity(lane.pads.len());
    for (pad, queue) in lane.pads.iter_mut().enumerate() {
        let mut frame = queue.pop_front().expect("every head matched pts");
        if pad > 0 {
            frame.rows.clear();
        }
        call.push(frame);
    }
    Ok(call)
}

/// The running pool: workers spawned, lanes wired, waiting to be fed.
pub struct Scheduler {
    shared: Arc<Shared>,
    handles: Vec<thread::JoinHandle<()>>,
    /// Per input index, the `(lane, pad)` pairs reading it.
    input_pads: Vec<Vec<(usize, usize)>>,
}

impl Scheduler {
    /// Wires the lanes, hands each sink to the lane it is bound to, and
    /// spawns `workers` threads. `inputs` is how many `-i` streams the
    /// feeder will push.
    pub fn start(
        seeds: Vec<LaneSeed>,
        sinks: Vec<Sink>,
        inputs: usize,
        workers: usize,
    ) -> Scheduler {
        let mut input_pads: Vec<Vec<(usize, usize)>> = vec![Vec::new(); inputs];
        let mut downstream: Vec<Vec<(usize, usize)>> = vec![Vec::new(); seeds.len()];
        for (index, seed) in seeds.iter().enumerate() {
            for (pad, source) in seed.sources.iter().enumerate() {
                match source {
                    Source::Input(input) => input_pads[*input].push((index, pad)),
                    Source::Node(upstream) => downstream[*upstream].push((index, pad)),
                }
            }
        }

        let mut lane_sinks: Vec<Vec<Sink>> = (0..seeds.len()).map(|_| Vec::new()).collect();
        for sink in sinks {
            lane_sinks[sink.node].push(sink);
        }

        let lanes: Vec<Lane> = seeds
            .into_iter()
            .zip(downstream)
            .zip(lane_sinks)
            .map(|((seed, downstream), sinks)| {
                let width = seed.width(workers);
                let pad_count = seed.sources.len();
                let windows = Windows::new(seed.shape, &seed.format);
                Lane {
                    name: seed.name,
                    sources: seed.sources,
                    pads: (0..pad_count).map(|_| VecDeque::new()).collect(),
                    pad_eof: vec![false; pad_count],
                    windows,
                    trailing_by_pad: vec![Vec::new(); pad_count],
                    trailing_upstream: BTreeMap::new(),
                    queue: VecDeque::new(),
                    next_ordinal: 0,
                    final_ordinal: None,
                    in_flight: 0,
                    width,
                    created: seed.runners.len(),
                    idle: seed.runners,
                    reopen: seed.reopen,
                    done: BTreeMap::new(),
                    next_flush: 0,
                    downstream,
                    sinks,
                    flushed_final: false,
                }
            })
            .collect();

        let shared = Arc::new(Shared {
            state: Mutex::new(State {
                lanes,
                error: None,
                finished: false,
            }),
            work: Condvar::new(),
            space: Condvar::new(),
            cap: workers * 2 + 2,
        });

        let handles = (0..workers)
            .map(|_| {
                let shared = Arc::clone(&shared);
                thread::spawn(move || worker(&shared))
            })
            .collect();

        Scheduler {
            shared,
            handles,
            input_pads,
        }
    }

    /// One frame read off input `input`, handed to every lane reading it.
    /// Waits for queue room first. Returns false when the scheduler has
    /// stopped on an error, which `finish` will surface.
    pub fn push_input(&self, input: usize, frame: Frame) -> bool {
        let mut state = self.shared.lock();
        loop {
            if state.error.is_some() {
                return false;
            }
            let room = self.input_pads[input]
                .iter()
                .all(|(lane, _)| state.lanes[*lane].queue.len() < self.shared.cap);
            if room {
                break;
            }
            state = self
                .shared
                .space
                .wait(state)
                .unwrap_or_else(|e| e.into_inner());
        }
        for (lane, pad) in &self.input_pads[input] {
            if let Err(e) = state.push_frames(*lane, *pad, std::slice::from_ref(&frame)) {
                if state.error.is_none() {
                    state.error = Some(e);
                }
                self.notify();
                return false;
            }
        }
        self.notify();
        true
    }

    /// Input `input` has ended, its wire leaving `trailing` rows with no
    /// frame to ride. Every lane reading it learns both.
    pub fn input_eof(&self, input: usize, trailing: &[String]) -> bool {
        let mut state = self.shared.lock();
        for (lane, pad) in &self.input_pads[input] {
            state.lanes[*lane].trailing_by_pad[*pad].extend(trailing.iter().cloned());
            if let Err(e) = state.mark_pad_eof(*lane, *pad) {
                if state.error.is_none() {
                    state.error = Some(e);
                }
                self.notify();
                return false;
            }
        }
        self.notify();
        true
    }

    /// A failure outside the scheduler (reading an input, checking a frame):
    /// stops the workers and surfaces `error`, unless a worker failed first.
    pub fn abort(self, error: anyhow::Error) -> Result<()> {
        {
            let mut state = self.shared.lock();
            if state.error.is_none() {
                state.error = Some(error);
            }
        }
        self.notify();
        self.finish()
    }

    /// Waits for every lane to flush its final result, joins the workers,
    /// and surfaces the first error if there was one.
    pub fn finish(self) -> Result<()> {
        {
            let mut state = self.shared.lock();
            while !state.finished && state.error.is_none() {
                state = self
                    .shared
                    .space
                    .wait(state)
                    .unwrap_or_else(|e| e.into_inner());
            }
        }
        let mut panicked = false;
        for handle in self.handles {
            panicked |= handle.join().is_err();
        }
        let mut state = self.shared.lock();
        if let Some(e) = state.error.take() {
            return Err(e);
        }
        if panicked {
            bail!("a worker thread panicked");
        }
        Ok(())
    }

    fn notify(&self) {
        self.shared.work.notify_all();
        self.shared.space.notify_all();
    }
}

/// One worker: pull a task from the readiest lane, run it unlocked, land the
/// result, repeat until everything has flushed or something failed.
fn worker(shared: &Shared) {
    let mut state = shared.lock();
    loop {
        if state.error.is_some() || state.finished {
            return;
        }
        let Some(i) = state.pick(shared.cap) else {
            state = shared.work.wait(state).unwrap_or_else(|e| e.into_inner());
            continue;
        };
        let (task, runner, reopen) = state.dispatch(i);
        // Popping the task made queue room; upstream lanes and the feeder
        // may be waiting on it.
        shared.work.notify_all();
        shared.space.notify_all();
        drop(state);

        let opened = match runner {
            Some(runner) => Ok(runner),
            None => {
                let reopen = reopen.expect("dispatch hands a runner or a reopen");
                Filter::open(&reopen.path, &reopen.format, &reopen.info, &reopen.params)
                    .map(|filter| Runner::Module(Box::new(filter)))
                    .with_context(|| format!("opening another instance of {}", reopen.path))
            }
        };
        let (runner, result) = match opened {
            Ok(mut runner) => {
                let result = runner.run(&task.frames, &task.trailing, task.last);
                (Some(runner), result)
            }
            Err(e) => (None, Err(e)),
        };

        state = shared.lock();
        state.complete(i, task.ordinal, task.last, runner, result);
        shared.work.notify_all();
        shared.space.notify_all();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ffrwd_wasm_runtime::runtime::{Media, TimeBase, VideoFormat};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::mpsc;
    use std::time::Duration;

    fn video() -> Format {
        Format {
            media: Media::Video(VideoFormat {
                width: 2,
                height: 2,
                pix_fmt: "rgba",
                frame_len: 16,
            }),
            time_base: TimeBase { num: 1, den: 25 },
        }
    }

    fn shape(pure: bool) -> Shape {
        Shape {
            window: 1,
            stride: 1,
            pure,
            one_to_one: true,
        }
    }

    fn frame(pts: i64) -> Frame {
        Frame {
            pts,
            data: Arc::new(Vec::new()),
            rows: Vec::new(),
        }
    }

    fn lane(name: &str, pure: bool, sources: Vec<Source>, runners: Vec<Runner>) -> LaneSeed {
        LaneSeed {
            name: name.to_string(),
            runners,
            shape: shape(pure),
            sources,
            format: video(),
            reopen: None,
        }
    }

    /// A stub that passes frames through, recording the pts it saw and how
    /// many calls ran at once.
    fn recording(
        seen: Arc<Mutex<Vec<i64>>>,
        running: Arc<AtomicUsize>,
        peak: Arc<AtomicUsize>,
        delay: fn(i64) -> u64,
    ) -> Runner {
        Runner::Stub(Box::new(move |frames, _trailing, _last| {
            let now = running.fetch_add(1, Ordering::SeqCst) + 1;
            peak.fetch_max(now, Ordering::SeqCst);
            for f in frames {
                seen.lock().unwrap().push(f.pts);
            }
            if let Some(f) = frames.first() {
                thread::sleep(Duration::from_millis(delay(f.pts)));
            }
            running.fetch_sub(1, Ordering::SeqCst);
            Ok(Processed {
                frames: frames.to_vec(),
                trailing: Vec::new(),
            })
        }))
    }

    #[test]
    fn an_impure_lane_admits_one_task_at_a_time_in_order() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let running = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let runner = recording(
            Arc::clone(&seen),
            Arc::clone(&running),
            Arc::clone(&peak),
            |_| 1,
        );

        let seeds = vec![lane("impure", false, vec![Source::Input(0)], vec![runner])];
        let sched = Scheduler::start(seeds, Vec::new(), 1, 8);
        for pts in 0..32 {
            assert!(sched.push_input(0, frame(pts)));
        }
        assert!(sched.input_eof(0, &[]));
        sched.finish().expect("the run drains");

        assert_eq!(peak.load(Ordering::SeqCst), 1, "one task in flight, ever");
        assert_eq!(
            *seen.lock().unwrap(),
            (0..32).collect::<Vec<i64>>(),
            "an impure lane's calls happen in ordinal order"
        );
    }

    #[test]
    fn a_pure_lane_finishing_out_of_order_still_delivers_in_order() {
        // Four instances of a pure lane, each call sleeping LONGER for
        // earlier frames, so completions land out of order on purpose. The
        // exclusive lane downstream must still see the input order.
        let upstream_seen = Arc::new(Mutex::new(Vec::new()));
        let running = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let runners = (0..4)
            .map(|_| {
                recording(
                    Arc::clone(&upstream_seen),
                    Arc::clone(&running),
                    Arc::clone(&peak),
                    |pts| 3 * (7 - (pts as u64) % 8),
                )
            })
            .collect();

        let downstream_seen = Arc::new(Mutex::new(Vec::new()));
        let downstream = recording(
            Arc::clone(&downstream_seen),
            Arc::new(AtomicUsize::new(0)),
            Arc::new(AtomicUsize::new(0)),
            |_| 0,
        );

        let seeds = vec![
            lane("pure", true, vec![Source::Input(0)], runners),
            lane("sink", false, vec![Source::Node(0)], vec![downstream]),
        ];
        let sched = Scheduler::start(seeds, Vec::new(), 1, 4);
        for pts in 0..24 {
            assert!(sched.push_input(0, frame(pts)));
        }
        assert!(sched.input_eof(0, &[]));
        sched.finish().expect("the run drains");

        assert!(
            peak.load(Ordering::SeqCst) > 1,
            "the pure lane actually ran concurrently"
        );
        assert_eq!(
            *downstream_seen.lock().unwrap(),
            (0..24).collect::<Vec<i64>>(),
            "results reassemble by ordinal before anything leaves the lane"
        );
    }

    #[test]
    fn a_stalled_lane_does_not_starve_the_others() {
        // Lane 0 blocks its worker on the first frame; lane 1, on its own
        // input, must drain completely with the one worker left.
        let (release_tx, release_rx) = mpsc::channel::<()>();
        let stalled = Runner::Stub(Box::new(move |frames, _trailing, last| {
            if !last && !frames.is_empty() {
                release_rx
                    .recv()
                    .expect("the test releases the stalled lane");
            }
            Ok(Processed {
                frames: frames.to_vec(),
                trailing: Vec::new(),
            })
        }));

        let seen = Arc::new(Mutex::new(Vec::new()));
        let busy = recording(
            Arc::clone(&seen),
            Arc::new(AtomicUsize::new(0)),
            Arc::new(AtomicUsize::new(0)),
            |_| 0,
        );

        let seeds = vec![
            lane("stalled", false, vec![Source::Input(0)], vec![stalled]),
            lane("busy", false, vec![Source::Input(1)], vec![busy]),
        ];
        let sched = Scheduler::start(seeds, Vec::new(), 2, 2);
        assert!(sched.push_input(0, frame(0)));
        for pts in 0..10 {
            assert!(sched.push_input(1, frame(pts)));
        }
        assert!(sched.input_eof(0, &[]));
        assert!(sched.input_eof(1, &[]));

        // The busy lane drains while the stalled one holds its worker.
        while seen.lock().unwrap().len() < 10 {
            thread::sleep(Duration::from_millis(1));
        }
        release_tx.send(()).expect("a worker is waiting on this");
        sched.finish().expect("the run drains");
        assert_eq!(*seen.lock().unwrap(), (0..10).collect::<Vec<i64>>());
    }
}
