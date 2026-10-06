//! `leaky`: the host's node that keeps a live picture near the wall clock.
//!
//! It is spelled like a module -
//! `[a]leaky=max_lateness=<seconds>:max_spread=<seconds>[b]` - and wired like
//! one, but nothing is compiled and nothing is instantiated. The name and
//! the first option are GStreamer's: a leaky queue, and a sink's
//! max-lateness.
//!
//! A live pipeline stamps its pts onto the Unix epoch, so a frame's LATENESS
//! is the wall clock less its pts, in seconds. Part of every frame's
//! lateness is the path's own offset: what a sender that started late, or a
//! relay in between, adds to each frame alike. The FLOOR is that offset as
//! the node sees it now: the smallest lateness read at arrival in the last
//! [`FLOOR_SECONDS`] of wall time. A frame later than the floor by more
//! than its SPREAD plus the BOUND is dropped; every other frame passes at
//! once, pixels, pts and rows untouched. Nothing is held and nothing is
//! reordered, so what leaves is the stream that arrived with its late
//! frames taken out. A frame behind one already passed is dropped too: the
//! output's timestamps never decrease.
//!
//! The floor moves with the path. It falls the moment a frame is read
//! earlier, and it rises once every lower reading has left the window. A
//! source whose clock drifts is followed, microseconds a window, with
//! nothing dropped; a source that slips behind for good costs one window of
//! drops, after which every frame passes again, that much later. Nothing
//! resets the floor, since a source that has fallen behind never catches up
//! by itself, and a row says when the floor rises (below).
//!
//! A frame counts toward the floor only when it was read AT ARRIVAL: the
//! node waited for it, so its lateness is the path's and not the node's
//! own. A frame queued behind a stage that cannot keep up is read later
//! than it arrived, and a floor that counted it would absorb the very
//! backlog the node is there to shed. The node cannot see its queue, but it
//! can tell when it waited. A frame read at least half a pts step after one
//! it dropped was waited for, since a drop frees the node at once. A frame
//! whose lateness is no more than the least of the run before it, plus what
//! a drifting clock adds in between ([`DRIFT`]), was waited for too, since a
//! node held up by its stage never reads one earlier; after a group handed
//! on at once, the group's width is allowed for, as the next group's first
//! picture is that much older than the freshest before it. The first frame
//! after the floor starts over counts as read at arrival. A reading below
//! the floor lowers it at once. One above it waits until its run has
//! closed, and the run's freshest picture counts then, unless something in
//! the run passed after something was dropped: that run was a backlog the
//! node read down to the bound and stopped in, and the queue went on. Under
//! a stage that cannot keep up, no read qualifies, and the floor holds where
//! it was; once nothing in the window qualifies, it holds at the last
//! reading that did.
//!
//! The BOUND is what the path's own delay varies by. Above the floor,
//! lateness has a long right tail, near log-normal, so the node keeps a
//! running mean and deviation of ln(excess), the excess being how far above
//! the floor a frame came, as exponential averages over the stream's own
//! time with [`TAIL_SECONDS`] for their time constant, and the bound is
//! [`TAIL_SIGMAS`] deviations above the mean. The excess is never taken as
//! less than [`TAIL_LEAST`], so timer noise near zero does not blow the
//! logarithm up. Dropped frames feed the estimate as passed ones do: an
//! estimate fed only what passed never sees past its own bound, so it stays
//! blind to a source that slipped by less than `max_lateness` until the
//! floor has turned over, and sits tighter than the path on one with
//! hiccups. Each run (below) feeds it once, with its first picture's excess
//! less the width of the group before it, and a group read at arrival
//! feeds it again with its freshest picture's.
//! The bound is never more than `max_lateness`, so a flapping path cannot
//! loosen it without limit, and until [`TAIL_SAMPLES`] samples have fed the
//! estimate the bound is `max_lateness` alone. A steady path ends with a
//! bound of a few tens of milliseconds; a bursty one with a looser bound,
//! up to `max_lateness`.
//!
//! The spread is what the input's own delivery adds. A relay that hands on a
//! whole group of pictures at once (MoQ hands a subscriber a second of them
//! in a few milliseconds) makes the group's first picture a second later
//! than its last, and none of that is the node falling behind. So the node
//! cuts what it reads into RUNS, pictures each arriving less than half its
//! own pts step after the one before, and a run's WIDTH is how far its
//! lateness ranges. A run counts only when it ends caught up, its freshest
//! picture within half of `max_lateness` of the floor: a backlog drained
//! after a slow stage behind the node also arrives at once, but ends near
//! the edge of the budget, and so teaches nothing. The spread is the widest
//! of the last [`KEEP`] counted runs, each capped at `max_spread`. A steady
//! feed's runs are single pictures, so its spread is 0 and it is judged by
//! the floor and the bound alone. Two kinds of run stand only until a
//! narrower one counts: the first, and one as wide as `max_spread`. Those
//! are a reader's own probe backlog as a rule, handed on at once when it
//! starts, or a stall.
//!
//! A leaf's first delivery is shaped by its own start more than by the
//! relay's pace: the piece of a group made so far when it joined, handed
//! on at once, and as a rule read before the decoder's queue has filled.
//! It is narrower than the groups after it, and fresher: each later
//! group's freshest picture is read behind the pictures the decoder holds
//! back and the ones handed on before it. Judged against the floor that
//! piece set, a leaf of the SMART demo found no group ending within half of
//! `max_lateness` of it for 24 s, so it kept the piece's spread and dropped
//! the oldest 12 pictures of every group. So a first run of more than one
//! picture stands for its freshness as it does for its width: when it
//! ends, the floor starts over from the picture after it.
//!
//! And the node LEARNS first: until [`LEARN`] runs have counted, or a run
//! of one picture after the first, and for no longer than `max_spread` plus
//! `max_lateness` seconds from its first picture. Meanwhile it drops only a
//! picture later than the floor by more than `max_spread` plus the bound,
//! and a run begun then counts when its freshest picture is within
//! `max_lateness` of the floor, one that would pass with no spread at all.
//! A steady feed, and a stage too slow from its first picture, read runs of
//! one picture, so the node stops learning at its third picture and is
//! judged as it always was.
//!
//! Once a second of wall time it writes a row saying what it did, on stderr
//! behind [`ROW_PREFIX`], where the run that started it reads it:
//! `{"kind":"leaky","node":<id>,"passed":<n>,"dropped":<n>,"lateness_s":<s>,
//! "baseline_s":<s>,"spread_s":<s>,"bound_s":<s>}`. The counts are the
//! window's own; the times are the latest frame's lateness, the floor, the
//! spread and the bound, in seconds. When the floor has risen by more than
//! [`SLIP`] since it was last reported, it writes
//! `{"kind":"leaky","node":<id>,"event":"slip","by_s":<s>,"baseline_s":<s>}`
//! the same way, so a source that slipped is seen in the log as it happens.
//!
//! Over a coded picture it reads packets, in decode order, and judges each
//! by its dts (its pts where the wire gives none) as it would a picture.
//! What it drops is whole groups: a keyframe passes or not on its own
//! lateness, and every packet after it passes only while it and all before
//! it in the group did, since each needs the ones before it to decode. A
//! group dropped stays dropped up to the next keyframe that passes.
//!
//! The name is reserved: no `-m` binds it, because the host answers for it.

use std::collections::VecDeque;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, bail, Result};
use ffrwd_wasm_runtime::runtime::{
    CodedFormat, CodedStream, Format, Frame, Packet, Shape, TimeBase,
};
use serde_json::json;

/// The node name the grammar reserves.
pub const NODE: &str = "leaky";

/// The most the bound may be, in seconds: how far past the floor and the
/// spread a frame may ever be.
const MAX_LATENESS: &str = "max_lateness";

/// The most the spread may grow to, in seconds.
const MAX_SPREAD: &str = "max_spread";

/// The node the rows name: the id the compiler gave it.
const NAME: &str = "node";

/// What `max_lateness` is when the node is not given one.
const DEFAULT_MAX_LATENESS: f64 = 0.5;

/// What `max_spread` is when the node is not given one: room for a relay
/// handing on two seconds of pictures at once.
const DEFAULT_MAX_SPREAD: f64 = 2.0;

/// How many counted runs the spread is the widest of.
pub const KEEP: usize = 8;

/// How many runs the node counts while it learns: its first delivery and
/// two after it.
pub const LEARN: usize = 3;

/// How far back the floor looks, in seconds of wall time: a frame read at
/// arrival lowers or raises it for this long. Long enough that a stall of a
/// second is still a stall, short enough that a source that slipped is
/// passing again before a viewer gives up.
pub const FLOOR_SECONDS: f64 = 4.0;

/// How much later past its pts a frame may be read than the run before it,
/// per second between them, and still count as read at arrival: a
/// millisecond a second. A clock drifting faster is broken, and a stage
/// that much slower than its input is behind by a tenth of a percent.
const DRIFT: f64 = 1e-3;

/// How far the floor rises before a row says so, in seconds.
pub const SLIP: f64 = 0.02;

/// The time constant of the tail's running mean and deviation, in seconds.
pub const TAIL_SECONDS: f64 = 10.0;

/// How many samples feed the tail before its bound is used.
pub const TAIL_SAMPLES: usize = 300;

/// How many deviations above the mean of ln(excess) the bound sits.
const TAIL_SIGMAS: f64 = 3.0;

/// The least an excess is taken as when it feeds the tail, in seconds.
const TAIL_LEAST: f64 = 0.005;

/// What a row on stderr starts with, so a reader tells it from the log.
pub const ROW_PREFIX: &str = "ffrwd:row ";

/// How much wall time one row covers, in seconds.
const REPORT_EVERY: f64 = 1.0;

/// How the host drives it: a frame at a time, and it hands on fewer frames
/// than it reads. Its floor is state, so one instance sees every frame.
pub const SHAPE: Shape = Shape {
    window: 1,
    stride: 1,
    pure: false,
    one_to_one: false,
};

/// The wall clock, in seconds since the Unix epoch.
pub type Clock = Box<dyn FnMut() -> f64 + Send>;

/// Where a row goes, as one line of JSON.
pub type Report = Box<dyn FnMut(String) + Send>;

/// The wall clock the node reads outside a test.
pub fn wall() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0.0, |elapsed| elapsed.as_secs_f64())
}

fn to_stderr(row: String) {
    eprintln!("{ROW_PREFIX}{row}");
}

/// The frames one row counts, and when its wall time started.
struct Window {
    start: f64,
    passed: u64,
    dropped: u64,
}

/// Pictures arriving faster than they were made: how many, when the latest
/// was read and made, the most and least lateness among them, whether the
/// node was learning when the first was read, and what became of them.
struct Run {
    pictures: usize,
    wall: f64,
    at: f64,
    most: f64,
    least: f64,
    learning: bool,
    /// Whether the first picture was read at arrival.
    arrived: bool,
    /// The width of the group before this run, where there was one: how
    /// much older its first picture is than the freshest before it.
    allowance: f64,
    /// Whether a picture of it was dropped.
    dropped: bool,
    /// Whether a picture of it passed after one was dropped: the node read
    /// its way down a backlog to the bound and stopped there.
    drained: bool,
    /// Whether its latest picture passed.
    kept: bool,
}

impl Run {
    /// Whether it is a group handed on at once, and not a backlog the node
    /// read down to the bound.
    fn group(&self) -> bool {
        self.pictures > 1 && !self.drained
    }
}

/// A lateness read at arrival, and when it was read.
struct Sample {
    wall: f64,
    lateness: f64,
}

/// The running mean and deviation of ln(excess), as exponential averages
/// over the stream's own time, and how many samples fed them.
struct Tail {
    mean: f64,
    variance: f64,
    samples: usize,
    /// When the picture of the latest sample was made.
    last: Option<f64>,
}

impl Tail {
    fn new() -> Tail {
        Tail {
            mean: 0.0,
            variance: 0.0,
            samples: 0,
            last: None,
        }
    }

    /// One more excess, from the picture made at `at`: a stall feeds it one
    /// picture's worth, not the stall's.
    fn feed(&mut self, excess: f64, at: f64) {
        let x = excess.max(TAIL_LEAST).ln();
        let alpha = self.last.map_or(1.0, |last| {
            1.0 - (-(at - last).max(0.0) / TAIL_SECONDS).exp()
        });
        let delta = x - self.mean;
        self.mean += alpha * delta;
        self.variance = (1.0 - alpha) * (self.variance + alpha * delta * delta);
        self.samples += 1;
        self.last = Some(at);
    }

    /// How far above the floor a frame may come, in seconds, never more
    /// than `ceiling`, and `ceiling` alone until enough samples have fed it.
    fn bound(&self, ceiling: f64) -> f64 {
        if self.samples < TAIL_SAMPLES {
            return ceiling;
        }
        (self.mean + TAIL_SIGMAS * self.variance.sqrt())
            .exp()
            .min(ceiling)
    }
}

/// What a node is opened with, in seconds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Limits {
    /// The most the bound may be: how far past the floor and the spread a
    /// picture may ever be.
    pub max_lateness: f64,
    /// The most the spread may grow to; 0 learns none.
    pub max_spread: f64,
}

impl Default for Limits {
    fn default() -> Limits {
        Limits {
            max_lateness: DEFAULT_MAX_LATENESS,
            max_spread: DEFAULT_MAX_SPREAD,
        }
    }
}

pub struct Leaky {
    limits: Limits,
    time_base: TimeBase,
    node: String,
    /// The smallest lateness among [`Leaky::recent`], or the last one once
    /// they have all left the window.
    floor: Option<f64>,
    /// The floor as a row last reported it, or as it last fell to.
    reported: Option<f64>,
    /// The latenesses read at arrival in the last [`FLOOR_SECONDS`], oldest
    /// first, each smaller than the one before it: the front is the least.
    recent: VecDeque<Sample>,
    tail: Tail,
    /// The latest frame's lateness.
    lateness: f64,
    last_passed: Option<i64>,
    /// The run the latest frame belongs to, still open.
    run: Option<Run>,
    /// The widths of the last [`KEEP`] counted runs, each capped, and
    /// whether each stands only until a narrower one counts.
    widths: VecDeque<(f64, bool)>,
    /// How many runs have counted, up to [`LEARN`].
    counted: usize,
    /// When the first picture was read.
    first: Option<f64>,
    window: Option<Window>,
    /// Over packets: whether the group the latest one belongs to still
    /// passes.
    group: bool,
    clock: Clock,
    report: Report,
}

/// The seconds an option names: a finite number, never below zero, and
/// above it where `positive`.
fn seconds(key: &str, value: &str, positive: bool) -> Result<f64> {
    let seconds: f64 = value
        .parse()
        .map_err(|_| anyhow!("{NODE}: {key} is not a number of seconds: {value}"))?;
    if !seconds.is_finite() || seconds < 0.0 || (positive && seconds == 0.0) {
        bail!("{NODE}: {key} is a number of seconds, and cannot be {value}");
    }
    Ok(seconds)
}

impl Leaky {
    /// Reads a node's options against the stream it is opened for, with the
    /// wall clock and stderr.
    pub fn open(options: &[(String, String)], format: &Format) -> Result<Leaky> {
        if format.video().is_none() {
            bail!(
                "{NODE} drops late pictures, and its input carries audio; a sound stream is \
                 never dropped"
            );
        }
        Leaky::configured(options, format.time_base)
    }

    /// Reads a node's options against a coded stream in `time_base`.
    pub fn open_coded(
        options: &[(String, String)],
        coded: &CodedStream,
        time_base: TimeBase,
    ) -> Result<Leaky> {
        if !matches!(coded.format, CodedFormat::Video { .. }) {
            bail!(
                "{NODE} drops late pictures, and its input is coded sound ({}); a sound stream \
                 is never dropped",
                coded.codec
            );
        }
        Leaky::configured(options, time_base)
    }

    fn configured(options: &[(String, String)], time_base: TimeBase) -> Result<Leaky> {
        let mut max_lateness: Option<f64> = None;
        let mut max_spread: Option<f64> = None;
        let mut node: Option<String> = None;
        for (key, value) in options {
            match key.as_str() {
                MAX_LATENESS => {
                    if max_lateness.is_some() {
                        bail!("{NODE} is given the option '{MAX_LATENESS}' twice");
                    }
                    max_lateness = Some(seconds(MAX_LATENESS, value, true)?);
                }
                MAX_SPREAD => {
                    if max_spread.is_some() {
                        bail!("{NODE} is given the option '{MAX_SPREAD}' twice");
                    }
                    max_spread = Some(seconds(MAX_SPREAD, value, false)?);
                }
                NAME => {
                    if node.is_some() {
                        bail!("{NODE} is given the option '{NAME}' twice");
                    }
                    node = Some(value.clone());
                }
                other => bail!(
                    "{NODE} has no option '{other}'; it takes {MAX_LATENESS}=<seconds>, \
                     {MAX_SPREAD}=<seconds> and {NAME}=<name>"
                ),
            }
        }
        let limits = Limits {
            max_lateness: max_lateness.unwrap_or(DEFAULT_MAX_LATENESS),
            max_spread: max_spread.unwrap_or(DEFAULT_MAX_SPREAD),
        };
        Ok(Leaky::with(
            limits,
            time_base,
            node.unwrap_or_else(|| NODE.to_string()),
            Box::new(wall),
            Box::new(to_stderr),
        ))
    }

    /// A node reading `clock` and writing its rows to `report`.
    pub fn with(
        limits: Limits,
        time_base: TimeBase,
        node: String,
        clock: Clock,
        report: Report,
    ) -> Leaky {
        Leaky {
            limits,
            time_base,
            node,
            floor: None,
            reported: None,
            recent: VecDeque::new(),
            tail: Tail::new(),
            lateness: 0.0,
            last_passed: None,
            run: None,
            widths: VecDeque::with_capacity(KEEP),
            counted: 0,
            first: None,
            window: None,
            group: false,
            clock,
            report,
        }
    }

    /// The path's offset as the node sees it: the smallest lateness read at
    /// arrival in the last [`FLOOR_SECONDS`], in seconds, once a picture has
    /// been read.
    pub fn floor(&self) -> Option<f64> {
        self.floor
    }

    /// How far above the floor and the spread a picture may come, in
    /// seconds: what the tail allows, never more than `max_lateness`.
    pub fn bound(&self) -> f64 {
        self.tail.bound(self.limits.max_lateness)
    }

    /// How much later than the floor the input's own delivery makes a
    /// picture: the widest of the last counted runs, 0 before any.
    pub fn spread(&self) -> f64 {
        self.widths
            .iter()
            .map(|(width, _)| *width)
            .fold(0.0, f64::max)
    }

    /// Whether the node is still learning its spread at `now`: fewer than
    /// [`LEARN`] runs have counted, none of one picture but the first, and
    /// its first picture was read less than `max_spread` plus `max_lateness`
    /// seconds ago.
    pub fn learning(&self, now: f64) -> bool {
        let Limits {
            max_lateness,
            max_spread,
        } = self.limits;
        max_spread > 0.0
            && self.counted < LEARN
            && self
                .first
                .is_none_or(|first| now - first < max_spread + max_lateness)
    }

    /// Whether the frame at `pts` passes, read against the clock now.
    pub fn judge(&mut self, pts: i64) -> bool {
        let now = (self.clock)();
        let passes = self.weigh(pts, now, true);
        self.tally(passes, now);
        passes
    }

    /// Whether the picture at `pts`, read at `now`, is in time, learning from
    /// it. Only an `ordered` time can be behind the last one passed.
    fn weigh(&mut self, pts: i64, now: f64, ordered: bool) -> bool {
        let at = pts as f64 * self.time_base.num as f64 / self.time_base.den.max(1) as f64;
        let lateness = now - at;
        self.first.get_or_insert(now);
        self.follow(now, at, lateness);
        self.settle(now);
        let floor = *self.floor.get_or_insert(lateness);
        self.lateness = lateness;
        let behind = ordered && self.last_passed.is_some_and(|last| pts < last);
        // While it learns, it drops only what no spread it may learn would pass.
        let spread = if self.learning(now) {
            self.limits.max_spread
        } else {
            self.spread()
        };
        let budget = spread + self.bound();
        let passes = !behind && lateness <= floor + budget;
        if passes && ordered {
            self.last_passed = Some(pts);
        }
        if let Some(run) = &mut self.run {
            run.kept = passes;
            if passes {
                run.drained |= run.dropped;
            } else {
                run.dropped = true;
            }
            if run.pictures == 1 {
                let allowance = run.allowance;
                self.tail.feed(lateness - floor - allowance, at);
            }
        }
        passes
    }

    /// One more picture passed or dropped, in the row's counts.
    fn tally(&mut self, passes: bool, now: f64) {
        let window = self.window.get_or_insert(Window {
            start: now,
            passed: 0,
            dropped: 0,
        });
        if passes {
            window.passed += 1;
        } else {
            window.dropped += 1;
        }
        if now - window.start >= REPORT_EVERY {
            self.write_row();
            self.window = Some(Window {
                start: now,
                passed: 0,
                dropped: 0,
            });
        }
    }

    /// The frame made at `at` and read at `now`, both in seconds, carries on
    /// the open run, or closes it and starts the next.
    fn follow(&mut self, now: f64, at: f64, lateness: f64) {
        if let Some(run) = &mut self.run {
            // Read less than half its own pts step after the one before: the
            // two arrived faster than they were made, together.
            if at > run.at && now - run.wall < (at - run.at) / 2.0 {
                run.pictures += 1;
                run.wall = now;
                run.at = at;
                run.most = run.most.max(lateness);
                run.least = run.least.min(lateness);
                return;
            }
        }
        let previous = self.run.take();
        if let Some(run) = &previous {
            self.close(run);
        }
        let allowance = previous
            .as_ref()
            .filter(|run| run.group())
            .map_or(0.0, |run| run.most - run.least);
        let arrived = match &previous {
            _ if self.floor.is_none() => true,
            None => true,
            Some(run) if !run.kept && at > run.at => true,
            Some(run) => lateness - allowance <= run.least + DRIFT * (now - run.wall),
        };
        self.run = Some(Run {
            pictures: 1,
            wall: now,
            at,
            most: lateness,
            least: lateness,
            learning: self.learning(now),
            arrived,
            allowance,
            dropped: false,
            drained: false,
            kept: false,
        });
    }

    /// A run is over. It counts when its freshest picture came within half
    /// of `max_lateness` of the floor: the node had caught up. A run begun
    /// while the node learned counts within all of `max_lateness`: its
    /// freshest picture would have passed with no spread.
    fn close(&mut self, run: &Run) {
        let Some(floor) = self.floor else {
            return;
        };
        if self.counted == 0 && run.pictures > 1 && self.limits.max_spread > 0.0 {
            // The first run of more than one picture is the node's own start
            // as a rule: after it, the floor starts over.
            self.recent.clear();
            self.floor = None;
            self.reported = None;
        } else if run.arrived && !run.drained {
            // Its freshest picture, now that the run is known not to have
            // been a backlog read down to the bound.
            self.remember(run.wall, run.least);
            if run.pictures > 1 {
                self.tail.feed(run.least - floor, run.at);
            }
        }
        let margin = if run.learning {
            self.limits.max_lateness
        } else {
            self.limits.max_lateness / 2.0
        };
        if run.least > floor + margin {
            return;
        }
        let width = (run.most - run.least).min(self.limits.max_spread);
        // The first run, and one as wide as the cap: a reader's probe backlog
        // as a rule, or a stall. Each stands until a narrower run counts.
        let provisional = self.counted == 0 || width >= self.limits.max_spread;
        // A picture on its own after the first: the input hands them on one at
        // a time, and there is nothing more to learn.
        self.counted = if self.counted > 0 && run.pictures == 1 {
            LEARN
        } else {
            (self.counted + 1).min(LEARN)
        };
        if !provisional {
            self.widths.retain(|(_, standing)| !standing);
        }
        if self.widths.len() == KEEP {
            self.widths.pop_front();
        }
        self.widths.push_back((width, provisional));
    }

    /// One more lateness read at arrival at `wall`. Any earlier one it is no
    /// larger than can never be the least again while it stands.
    fn remember(&mut self, wall: f64, lateness: f64) {
        while self.recent.back().is_some_and(|s| s.lateness >= lateness) {
            self.recent.pop_back();
        }
        self.recent.push_back(Sample { wall, lateness });
    }

    /// The floor at `now`: the least of what was read at arrival in the
    /// window, falling at once and rising as older readings leave it, with a
    /// row when it has risen by more than [`SLIP`].
    fn settle(&mut self, now: f64) {
        while self
            .recent
            .front()
            .is_some_and(|s| s.wall < now - FLOOR_SECONDS)
        {
            self.recent.pop_front();
        }
        // The open run counts while it may still close as read at arrival;
        // its least joins the window when it does.
        let pending = self
            .run
            .as_ref()
            .filter(|run| run.arrived && !run.drained)
            .map(|run| run.least);
        let recent = self.recent.front().map(|s| s.lateness);
        let Some(least) = recent.into_iter().chain(pending).reduce(f64::min) else {
            return;
        };
        match self.floor {
            Some(floor) if least > floor => {
                self.floor = Some(least);
                let reported = self.reported.unwrap_or(floor);
                if least - reported > SLIP {
                    self.reported = Some(least);
                    let row = json!({
                        "kind": NODE,
                        "node": self.node,
                        "event": "slip",
                        "by_s": millis(least - reported),
                        "baseline_s": millis(least),
                    });
                    (self.report)(row.to_string());
                }
            }
            Some(floor) if least < floor => {
                self.floor = Some(least);
                self.reported = Some(least);
            }
            Some(_) => {}
            None => {
                self.floor = Some(least);
                self.reported = Some(least);
            }
        }
    }

    /// One frame through, or None where it was too late.
    pub fn pass(&mut self, frame: Frame) -> Option<Frame> {
        self.judge(frame.pts).then_some(frame)
    }

    /// One packet through, or None where it or its group was too late.
    pub fn pass_packet(&mut self, packet: Packet) -> Option<Packet> {
        let now = (self.clock)();
        let fresh = self.weigh(packet.dts.unwrap_or(packet.pts), now, packet.dts.is_some());
        self.group = fresh && (packet.keyframe || self.group);
        self.tally(self.group, now);
        self.group.then_some(packet)
    }

    /// The stream has ended: the row for the window still open.
    pub fn finish(&mut self) {
        if self.window.is_some() {
            self.write_row();
            self.window = None;
        }
    }

    fn write_row(&mut self) {
        let Some(window) = &self.window else {
            return;
        };
        if window.passed == 0 && window.dropped == 0 {
            return;
        }
        let row = json!({
            "kind": NODE,
            "node": self.node,
            "passed": window.passed,
            "dropped": window.dropped,
            "lateness_s": millis(self.lateness),
            "baseline_s": millis(self.floor().unwrap_or(0.0)),
            "spread_s": millis(self.spread()),
            "bound_s": millis(self.bound()),
        });
        (self.report)(row.to_string());
    }
}

/// Seconds to the millisecond, which is what a row is read at.
fn millis(seconds: f64) -> f64 {
    (seconds * 1000.0).round() / 1000.0
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;
    use ffrwd_wasm_runtime::runtime::{AudioFormat, Media, VideoFormat};

    /// Pts in milliseconds.
    const MILLIS: TimeBase = TimeBase { num: 1, den: 1000 };

    /// Where the clock stands, and the rows written, both shared with the node.
    struct Harness {
        now: Arc<Mutex<f64>>,
        rows: Arc<Mutex<Vec<serde_json::Value>>>,
        leaky: Leaky,
    }

    impl Harness {
        fn new(max_lateness: f64) -> Harness {
            Harness::limited(Limits {
                max_lateness,
                ..Limits::default()
            })
        }

        fn limited(limits: Limits) -> Harness {
            let now = Arc::new(Mutex::new(0.0));
            let rows = Arc::new(Mutex::new(Vec::new()));
            let clock = Arc::clone(&now);
            let written = Arc::clone(&rows);
            let leaky = Leaky::with(
                limits,
                MILLIS,
                "n3".to_string(),
                Box::new(move || *clock.lock().unwrap()),
                Box::new(move |row| {
                    written
                        .lock()
                        .unwrap()
                        .push(serde_json::from_str(&row).unwrap());
                }),
            );
            Harness { now, rows, leaky }
        }

        /// A frame at `pts` seconds, judged with the wall at `wall`.
        fn at(&mut self, wall: f64, pts: f64) -> bool {
            *self.now.lock().unwrap() = wall;
            self.leaky.judge((pts * 1000.0).round() as i64)
        }

        fn rows(&self) -> Vec<serde_json::Value> {
            self.rows.lock().unwrap().clone()
        }

        /// The rows that counted a second's pictures.
        fn reports(&self) -> Vec<serde_json::Value> {
            self.rows()
                .into_iter()
                .filter(|row| row["event"].is_null())
                .collect()
        }

        /// The rows that said the floor rose.
        fn slips(&self) -> Vec<serde_json::Value> {
            self.rows()
                .into_iter()
                .filter(|row| row["event"] == "slip")
                .collect()
        }

        /// Pictures `(arriving, pts)` in seconds, read in turn by a node that
        /// then spends `cost(wall)` seconds handing on each one it passes, as
        /// a stage behind it takes it, and next to nothing on one it drops.
        /// What each picture came to, and the spread when it was judged.
        fn serve(
            &mut self,
            pictures: &[(f64, f64)],
            cost: impl Fn(f64) -> f64,
        ) -> Vec<(bool, f64)> {
            let mut wall = f64::MIN;
            pictures
                .iter()
                .map(|&(arriving, pts)| {
                    wall = wall.max(arriving);
                    let passed = self.at(wall, pts);
                    wall += if passed { cost(wall) } else { 1e-5 };
                    (passed, self.leaky.spread())
                })
                .collect()
        }

        /// Pictures `k` of a steady feed made at 30 a second, each read
        /// `late(k)` seconds after it was made by a node that keeps up; which
        /// of them were dropped.
        fn steady(
            &mut self,
            pictures: impl Iterator<Item = u32>,
            late: impl Fn(u32) -> f64,
        ) -> Vec<u32> {
            pictures
                .filter(|&k| !self.at(made(k) + late(k), made(k)))
                .collect()
        }
    }

    /// Picture `k` of a feed made at 30 a second, in seconds.
    fn made(k: u32) -> f64 {
        1_000.0 + f64::from(k) / 30.0
    }

    /// Timer noise on picture `k`: 0 to 12 ms, in no pattern a window sees.
    fn noise(k: u32) -> f64 {
        f64::from((k * 7) % 13) / 1_000.0
    }

    /// A relay's delivery: each second's 30 pictures handed on together,
    /// 0.1 ms apart, 10.25 s after the second they were made in ended, as
    /// the SMART demo's subscriber measured Cloudflare's. The first delivery
    /// is a reader's probe backlog: its first `probed` seconds at once.
    fn bursts(seconds: u32, probed: u32) -> Vec<(f64, f64)> {
        (0..seconds * 30)
            .map(|k| {
                let delivered = (k / 30).max(probed - 1) + 1;
                let first = if k / 30 < probed { 0 } else { (k / 30) * 30 };
                let arriving = 1_000.0 + f64::from(delivered) + 10.25;
                (arriving + f64::from(k - first) * 1e-4, made(k))
            })
            .collect()
    }

    fn dropped(served: &[(bool, f64)]) -> usize {
        served.iter().filter(|(passed, _)| !passed).count()
    }

    #[test]
    fn a_late_packet_drops_the_rest_of_its_group_up_to_a_keyframe_in_time() {
        let mut h = Harness::limited(Limits {
            max_lateness: 0.5,
            max_spread: 0.0,
        });
        let passed: Vec<bool> = (0..30i64)
            .map(|k| {
                let pts = 1_000.0 + k as f64 / 30.0;
                let stalled = if (13..17).contains(&k) { 2.0 } else { 0.0 };
                *h.now.lock().unwrap() = pts + 7.0 + stalled;
                let ms = (pts * 1000.0).round() as i64;
                let packet = Packet {
                    pts: ms + 66,
                    dts: Some(ms),
                    duration: None,
                    keyframe: k % 10 == 0,
                    data: vec![k as u8],
                };
                h.leaky.pass_packet(packet).is_some()
            })
            .collect();
        let dropped: Vec<usize> = (0..30).filter(|k| !passed[*k]).collect();
        assert_eq!(dropped, (13..20).collect::<Vec<_>>());
    }

    #[test]
    fn a_sender_that_starts_late_is_absorbed_by_the_floor() {
        let mut h = Harness::new(0.5);
        // Every frame arrives 7 s after its stamp: late, but alike.
        let passed: Vec<bool> = (0..90)
            .map(|k| {
                let pts = 1_000.0 + f64::from(k) / 30.0;
                h.at(pts + 7.0, pts)
            })
            .collect();
        assert!(passed.iter().all(|p| *p));
        // Seven seconds, to the millisecond the pts are stamped in.
        assert!((h.leaky.floor().unwrap() - 7.0).abs() < 1e-3);
    }

    #[test]
    fn the_floor_is_the_least_lateness_read_at_arrival_in_the_window() {
        let mut h = Harness::new(0.5);
        assert!(h.at(10.3, 10.0));
        assert!(h.at(10.4, 10.2)); // 0.2: earlier than any before it
        assert!((h.leaky.floor().unwrap() - 0.2).abs() < 1e-9);
        // 0.6 is past 0.2 by 0.4, inside the budget: it does not move the
        // floor, which rises only as what it stands on leaves the window.
        assert!(h.at(11.0, 10.4));
        assert!((h.leaky.floor().unwrap() - 0.2).abs() < 1e-9);
        // The 0.2 reading leaves the window; the 0.6 one, read later past
        // its pts than the one before, never counted. The floor stands on
        // what came after.
        assert!(h.at(11.1, 10.6));
        assert!(h.at(10.4 + FLOOR_SECONDS + 0.1, 10.4 + FLOOR_SECONDS - 0.2));
        assert!((h.leaky.floor().unwrap() - 0.3).abs() < 1e-9);
    }

    #[test]
    fn a_frame_later_than_the_floor_by_more_than_the_budget_is_dropped() {
        let mut h = Harness::new(0.5);
        assert!(h.at(100.1, 100.0)); // the floor: 0.1
        assert!(h.at(100.633, 100.033)); // 0.6: past it by exactly 0.5
        assert!(!h.at(100.667, 100.066)); // 0.601: past it by more
                                          // The stall is over: the next frame is on time again and passes.
        assert!(h.at(100.2, 100.1));
    }

    #[test]
    fn a_stall_drops_the_backlog_and_keeps_what_is_on_time() {
        let mut h = Harness::new(0.5);
        let mut kept = Vec::new();
        for k in 0..30 {
            let pts = f64::from(k) / 30.0;
            // The first ten arrive as they are made; the other twenty all at
            // once, when a stall ends at 1.09 s.
            let wall = if k < 10 { pts } else { 1.09 };
            if h.at(wall, pts) {
                kept.push(k);
            }
        }
        // What was more than half a second old when the stall ended is gone:
        // everything before 0.59 s. The rest leaves in the order it came.
        let expected: Vec<i32> = (0..10).chain(18..30).collect();
        assert_eq!(kept, expected);
    }

    #[test]
    fn a_frame_behind_one_already_passed_is_dropped_whatever_its_lateness() {
        let mut h = Harness::new(0.5);
        assert!(h.at(5.0, 5.0));
        assert!(!h.at(5.01, 4.99));
        assert!(h.at(5.04, 5.033));
    }

    #[test]
    fn frames_pass_untouched_and_in_order() {
        let mut h = Harness::new(0.5);
        *h.now.lock().unwrap() = 1.0;
        let frame = Frame {
            pts: 1000,
            data: Arc::new(vec![1, 2, 3]),
            rows: vec!["{\"a\":1}".to_string()],
        };
        let out = h
            .leaky
            .pass(frame.clone())
            .expect("an on-time frame passes");
        assert_eq!(out.pts, frame.pts);
        assert!(Arc::ptr_eq(&out.data, &frame.data));
        assert_eq!(out.rows, frame.rows);
    }

    #[test]
    fn a_row_reports_each_second_of_wall_time() {
        let mut h = Harness::new(0.5);
        for k in 0..45 {
            let pts = f64::from(k) / 30.0;
            // From the thirtieth frame on, each arrives 2 s late.
            let wall = if k < 30 { pts } else { pts + 2.0 };
            h.at(wall, pts);
        }
        // The frame that closed the first second is counted in its row.
        let rows = h.rows();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0]["kind"], "leaky");
        assert_eq!(rows[0]["node"], "n3");
        assert_eq!(rows[0]["passed"], 30);
        assert_eq!(rows[0]["dropped"], 1);
        assert_eq!(rows[0]["lateness_s"], 2.0);
        assert_eq!(rows[0]["baseline_s"], 0.0);
        assert_eq!(rows[0]["spread_s"], 0.0);
        // Too few samples have fed the tail: the bound is max_lateness.
        assert_eq!(rows[0]["bound_s"], 0.5);
        h.leaky.finish();
        let rows = h.rows();
        assert_eq!(rows.len(), 2);
        assert_eq!(
            (rows[1]["passed"].clone(), rows[1]["dropped"].clone()),
            (0.into(), 14.into())
        );
        // Nothing is left to report twice.
        h.leaky.finish();
        assert_eq!(h.rows().len(), 2);
    }

    #[test]
    fn a_steady_feed_learns_no_spread_and_is_judged_as_it_always_was() {
        // Thirty pictures a second, each read up to 12 ms off its pace; from
        // the fourth second to the sixth the stage behind the node takes
        // 60 ms over each picture it is handed, more than the feed allows.
        let feed: Vec<(f64, f64)> = (0..300)
            .map(|k| (made(k) + 0.25 + noise(k), made(k)))
            .collect();
        let cost = |wall: f64| {
            if (1_003.0..1_006.0).contains(&wall) {
                0.06
            } else {
                1e-3
            }
        };
        let learning = Harness::new(0.5).serve(&feed, cost);
        let blind = Harness::limited(Limits {
            max_lateness: 0.5,
            max_spread: 0.0,
        })
        .serve(&feed, cost);

        let passed = |served: &[(bool, f64)]| -> Vec<bool> {
            served.iter().map(|(passed, _)| *passed).collect()
        };
        assert_eq!(passed(&learning), passed(&blind));
        assert!(dropped(&learning) > 20, "{}", dropped(&learning));
        // Nothing was learned while it kept up, nor while it fell behind. The
        // backlog it drained when the stage recovered arrived at once and is
        // a run, which stands for the KEEP pictures after it.
        let learned: Vec<usize> = (0..learning.len())
            .filter(|k| learning[*k].1 > 0.0)
            .collect();
        assert_eq!(learned.len(), KEEP, "{learned:?}");
        assert!(learned[0] > 160 && learned[KEEP - 1] == learned[0] + KEEP - 1);
        assert_eq!(learning.last().unwrap().1, 0.0);
    }

    #[test]
    fn a_bursty_feed_learns_its_spread_and_drops_nothing() {
        let feed = bursts(20, 1);
        let mut h = Harness::new(0.5);
        let served = h.serve(&feed, |_| 1e-4);

        assert_eq!(dropped(&served), 0);
        // The first burst teaches it, and every later one is judged with it:
        // a second's pictures less the 3 ms they took to arrive.
        assert_eq!(served[29].1, 0.0);
        for (passed, spread) in &served[30..] {
            assert!(*passed);
            assert!((spread - 0.9637).abs() < 1e-3, "{spread}");
        }
        let rows = h.reports();
        assert!(rows.len() >= 18, "{rows:?}");
        assert!(rows[2..]
            .iter()
            .all(|row| row["spread_s"] == 0.964 && row["dropped"] == 0));

        // Without it, the older half of every burst is past the budget.
        let blind = Harness::limited(Limits {
            max_lateness: 0.5,
            max_spread: 0.0,
        })
        .serve(&feed, |_| 1e-4);
        assert!(dropped(&blind) > 250, "{}", dropped(&blind));
    }

    #[test]
    fn the_readers_probe_backlog_is_forgotten_once_a_burst_counts() {
        // A reader that probed four seconds hands them on at once: a run
        // four seconds wide, capped, which stands only until the next.
        let served = Harness::new(0.5).serve(&bursts(12, 4), |_| 1e-4);
        assert_eq!(dropped(&served), 0);
        assert_eq!(served[120].1, 2.0);
        assert!((served[150].1 - 0.9637).abs() < 1e-3, "{}", served[150].1);
        assert!((served.last().unwrap().1 - 0.9637).abs() < 1e-3);
    }

    #[test]
    fn a_backlog_as_wide_as_the_cap_stands_only_until_a_narrower_run_counts() {
        // A reader starting hands on the eight seconds it probed in two
        // pieces, a second's worth and then the other seven, and after them a
        // group a second as each is made.
        let feed: Vec<(f64, f64)> = (0..480)
            .map(|k| {
                let arriving = match k {
                    0..=29 => 1_008.3 + f64::from(k) * 1e-4,
                    30..=239 => 1_008.35 + f64::from(k - 30) * 1e-4,
                    _ => 1_000.0 + f64::from(k / 30 + 1) + 0.3 + f64::from(k % 30) * 1e-4,
                };
                (arriving, made(k))
            })
            .collect();
        let served = Harness::new(0.5).serve(&feed, |_| 1e-4);
        assert_eq!(dropped(&served), 0);
        // Judged with the backlog's capped width for one group, then with
        // the groups' own.
        assert_eq!(served[240].1, 2.0);
        assert!((served[270].1 - 0.9637).abs() < 1e-3, "{}", served[270].1);
    }

    #[test]
    fn a_feed_whose_groups_are_wider_than_the_cap_keeps_the_cap() {
        // Two-second groups, 60 pictures each, against a cap of 1.5 s.
        let feed: Vec<(f64, f64)> = (0..600)
            .map(|k| {
                let group = k / 60;
                let arriving = 1_000.0 + f64::from(group * 2 + 2) + 10.25;
                (arriving + f64::from(k % 60) * 1e-4, made(k))
            })
            .collect();
        let limits = Limits {
            max_lateness: 0.5,
            max_spread: 1.5,
        };
        let served = Harness::limited(limits).serve(&feed, |_| 1e-4);
        assert!(served[120..].iter().all(|(_, spread)| *spread == 1.5));
    }

    #[test]
    fn a_bursty_feed_behind_an_overloaded_stage_still_drops_and_its_spread_holds() {
        // Ten seconds keeping up, then the stage behind takes 0.1 s over each
        // picture: 10 a second of the 30 arriving.
        let feed = bursts(40, 1);
        let mut h = Harness::new(0.5);
        let served = h.serve(&feed, |wall| if wall < 1_021.0 { 1e-4 } else { 0.1 });

        let (before, after) = served.split_at(300);
        assert_eq!(dropped(before), 0);
        let late = &after[300..];
        let share = dropped(late) as f64 / late.len() as f64;
        assert!((0.6..0.72).contains(&share), "{share}");
        // Falling behind teaches it nothing: the spread is the bursts'.
        assert!(after
            .iter()
            .all(|(_, spread)| (spread - 0.9637).abs() < 1e-3));
    }

    #[test]
    fn a_stall_widens_the_spread_to_its_cap_and_no_further_and_not_for_long() {
        // The stage behind stops for five seconds once, and keeps up again.
        let feed = bursts(40, 1);
        let stalled = std::cell::Cell::new(false);
        let served = Harness::new(0.5).serve(&feed, |wall| {
            if wall > 1_021.0 && !stalled.replace(true) {
                5.0
            } else {
                1e-4
            }
        });
        assert!(dropped(&served) > 0);
        let widest = served.iter().map(|(_, spread)| *spread).fold(0.0, f64::max);
        assert_eq!(widest, 2.0);
        // KEEP bursts later, the spread is the feed's own again.
        assert!((served.last().unwrap().1 - 0.9637).abs() < 1e-3);
    }

    #[test]
    fn a_feed_that_turns_steady_forgets_its_spread_after_keep_pictures() {
        let mut feed = bursts(5, 1);
        feed.extend((150..200).map(|k| (made(k) + 10.3, made(k))));
        let served = Harness::new(0.5).serve(&feed, |_| 1e-4);
        assert_eq!(dropped(&served), 0);
        assert!(served[150].1 > 0.9);
        assert_eq!(served[150 + KEEP + 1].1, 0.0);
    }

    #[test]
    fn max_spread_caps_what_a_run_teaches() {
        let limits = Limits {
            max_lateness: 0.5,
            max_spread: 0.25,
        };
        let served = Harness::limited(limits).serve(&bursts(6, 1), |_| 1e-4);
        assert_eq!(served.last().unwrap().1, 0.25);
        assert!(dropped(&served) > 0);
    }

    /// What it costs the stage behind to take a picture in the SMART demo's
    /// leaf: a group of 30 all handed on reads as a run 0.865 s wide.
    const HANDING_ON: f64 = 0.0035;

    /// A leaf joining a relay as the SMART demo's did on Cloudflare. Its
    /// first delivery is a piece of a group: 11 pictures read 0.3 s wide,
    /// the freshest 2.6 s after it was made. Then the rest of that group,
    /// and a group of 30 once a second, each out of the decoder 0.5 ms a
    /// picture apart, its oldest read `late` s past 2.6 s plus the group's
    /// own length, so that its freshest is read `late` s past 2.6 s plus
    /// what handing on the pictures before it takes.
    fn a_leaf_joining(seconds: u32, late: f64) -> Vec<(f64, f64)> {
        let piece = (0..11).map(|k| (made(10) + 2.6 - f64::from(10 - k) * 0.0033, made(k)));
        let groups = (11..seconds * 30).map(|k| {
            let oldest = if k < 30 { 11 } else { k / 30 * 30 };
            let newest = k / 30 * 30 + 29;
            let arriving = made(newest) + 2.6 + late + f64::from(k - oldest) * 5e-4;
            (arriving, made(k))
        });
        piece.chain(groups).collect()
    }

    #[test]
    fn a_leaf_whose_first_delivery_is_narrow_learns_its_groups_and_drops_nothing() {
        // Its groups' freshest pictures are read 0.27 s to 0.31 s past the
        // floor the first piece set, more than half of max_lateness.
        // Judged against that piece from the first picture on, as 0.25.2
        // judged it, the node kept the piece's 0.3 s and dropped the oldest
        // 12 of every group, 349 of 900 in 30 s, as the demo's leaf did for
        // its first 24 s.
        let feed = a_leaf_joining(30, 0.21);
        let mut h = Harness::new(0.5);
        let served = h.serve(&feed, |_| HANDING_ON);
        assert_eq!(dropped(&served), 0);
        // The rest of the first group teaches it 0.54 s, and the first whole
        // group, its third delivery, the groups' own width, by which every
        // group after is judged.
        assert!(served[30..60]
            .iter()
            .all(|(_, spread)| (spread - 0.537).abs() < 2e-3));
        assert!(served[60..]
            .iter()
            .all(|(_, spread)| (spread - 0.865).abs() < 2e-3));
        assert!(!h.leaky.learning(feed[60].0));
        assert!(h.reports().iter().all(|row| row["dropped"] == 0));
        // The floor started over after the piece.
        assert!(h.leaky.floor().unwrap() > 2.6 + 0.2);
    }

    #[test]
    fn a_first_delivery_fresher_than_every_group_after_it_sets_no_floor() {
        // Two pictures out of the decoder before its queue has filled, read
        // 0.8 s fresher than any group after them. Against their floor every
        // picture after would be late by more than the budget, however wide
        // the spread learned.
        let mut feed: Vec<(f64, f64)> = (9..11)
            .map(|k| (made(10) + 2.0 - f64::from(10 - k) * 1e-4, made(k)))
            .collect();
        feed.extend(a_leaf_joining(12, 0.21).into_iter().skip(11));
        let mut h = Harness::new(0.5);
        let served = h.serve(&feed, |_| HANDING_ON);
        assert_eq!(dropped(&served), 0);
        // The groups' own freshest pictures, read 0.865 s of handing on
        // after their oldest, are what the floor stands on once the rest of
        // the first group has left the window.
        let floor = h.leaky.floor().unwrap();
        assert!((2.6 + 0.21..2.6 + 0.32).contains(&floor), "{floor}");
        assert!((h.leaky.spread() - 0.865).abs() < 2e-3);
    }

    #[test]
    fn a_steady_feed_stops_learning_at_its_third_picture() {
        let mut h = Harness::new(0.5);
        for k in 0..3 {
            let wall = made(k) + 0.25;
            assert!(h.leaky.learning(wall));
            assert!(h.at(wall, made(k)));
        }
        assert!(!h.leaky.learning(made(3) + 0.25));
    }

    #[test]
    fn a_stage_too_slow_from_the_first_picture_is_judged_as_without_learning() {
        // A steady feed, and bursts, behind a stage that never keeps up.
        let steady: Vec<(f64, f64)> = (0..300).map(|k| (made(k) + 0.25, made(k))).collect();
        for (feed, cost) in [(steady, 0.06), (bursts(10, 1), 0.1)] {
            let learning = Harness::new(0.5).serve(&feed, |_| cost);
            let blind = Harness::limited(Limits {
                max_lateness: 0.5,
                max_spread: 0.0,
            })
            .serve(&feed, |_| cost);
            let passed = |served: &[(bool, f64)]| -> Vec<bool> {
                served.iter().map(|(passed, _)| *passed).collect()
            };
            assert_eq!(passed(&learning), passed(&blind));
            assert!(
                dropped(&learning) > feed.len() / 3,
                "{}",
                dropped(&learning)
            );
        }
    }

    #[test]
    fn learning_lasts_no_longer_than_max_spread_and_max_lateness() {
        // After a first picture on time, every group's freshest picture is
        // read 0.6 s past it: no run ends fresh enough to count. Up to 2.5 s
        // from the first picture it drops nothing; then it stops learning,
        // with nothing learned, and drops what is past max_lateness over the
        // first picture's floor. That floor stands for one window. Then the
        // groups' own freshest pictures are the floor, a row says the source
        // slipped 0.6 s, the groups count, and they pass.
        let mut feed = vec![(made(0) + 2.6, made(0))];
        feed.extend(a_leaf_joining(8, 0.6).into_iter().skip(11));
        let mut h = Harness::new(0.5);
        let served = h.serve(&feed, |_| 1e-4);
        let start = feed[0].0;
        let outcome = |from: f64, until: f64| -> (usize, usize) {
            feed.iter()
                .zip(&served)
                .filter(|((arriving, _), _)| (from..until).contains(arriving))
                .fold((0, 0), |(passed, dropped), (_, (kept, _))| {
                    if *kept {
                        (passed + 1, dropped)
                    } else {
                        (passed, dropped + 1)
                    }
                })
        };
        assert_eq!(outcome(start, start + 2.5), (1 + 19, 0));
        assert!(!h.leaky.learning(start + 2.5));
        assert_eq!(outcome(start + 2.5, start + FLOOR_SECONDS), (0, 60));
        assert_eq!(outcome(start + FLOOR_SECONDS + 1.0, f64::MAX).1, 0);
        // The groups' freshest picture is read 29 arrivals after the oldest.
        let floor = h.leaky.floor().unwrap();
        assert!((3.2..3.22).contains(&floor), "{floor}");
        let slips = h.slips();
        assert_eq!(slips.len(), 1, "{slips:?}");
        let by = slips[0]["by_s"].as_f64().unwrap();
        assert!((0.6..0.62).contains(&by), "{by}");
        assert!(h.leaky.spread() > 0.5, "{}", h.leaky.spread());
    }

    /// Thirty seconds of a steady feed read 0.25 s late with timer noise:
    /// enough for the tail to be in use, with a bound of tens of
    /// milliseconds.
    fn settled() -> Harness {
        let mut h = Harness::new(0.5);
        assert!(h.steady(0..900, |k| 0.25 + noise(k)).is_empty());
        assert!(h.leaky.tail.samples >= TAIL_SAMPLES);
        let bound = h.leaky.bound();
        assert!((0.01..0.1).contains(&bound), "{bound}");
        h
    }

    #[test]
    fn a_transient_burst_is_dropped_and_the_floor_holds() {
        let mut h = settled();
        // The network holds a second of pictures and hands them on at once,
        // 1 s late; then the feed is as it was.
        let dropped = h.steady(900..930, |k| 0.25 + noise(k) + f64::from(930 - k) / 30.0);
        assert!(dropped.len() >= 28 && dropped[0] == 900, "{dropped:?}");
        assert!(h.steady(930..1_200, |k| 0.25 + noise(k)).is_empty());
        let floor = h.leaky.floor().unwrap();
        assert!((floor - 0.25).abs() < 2e-3, "{floor}");
        assert!(h.slips().is_empty());
    }

    #[test]
    fn a_permanent_step_costs_one_window_of_drops_and_a_row_and_then_passes() {
        let mut h = settled();
        // From picture 900 on, the source is 0.6 s further behind, for good:
        // past max_lateness over the floor it had, whatever the tail allows.
        let dropped = h.steady(900..1_800, |k| 0.85 + noise(k));
        assert!(dropped.contains(&900));
        // The drops end once the window has turned over: the last reading
        // from before the step was at picture 899, and the step itself reads
        // every picture after it 0.5 s later, so FLOOR_SECONDS of wall time
        // is 105 pictures.
        let last = *dropped.last().unwrap();
        assert!((1_000..=1_010).contains(&last), "{last}");
        assert_eq!(dropped.len(), (last - 900 + 1) as usize, "{dropped:?}");
        let floor = h.leaky.floor().unwrap();
        assert!((floor - 0.85).abs() < 2e-3, "{floor}");
        // One row said so, with the size of the step.
        let slips = h.slips();
        assert_eq!(slips.len(), 1, "{slips:?}");
        assert_eq!(slips[0]["node"], "n3");
        let by = slips[0]["by_s"].as_f64().unwrap();
        assert!((by - 0.6).abs() < 0.015, "{by}");
        assert_eq!(slips[0]["baseline_s"].as_f64().unwrap(), millis(floor));
        // A second, smaller step is another row.
        let dropped = h.steady(1_800..2_400, |k| 1.05 + noise(k));
        assert!(
            !dropped.is_empty() && dropped.len() < 130,
            "{}",
            dropped.len()
        );
        let slips = h.slips();
        assert_eq!(slips.len(), 2, "{slips:?}");
        let by = slips[1]["by_s"].as_f64().unwrap();
        assert!((by - 0.2).abs() < 0.015, "{by}");
    }

    #[test]
    fn a_source_drifting_a_minute_an_hour_is_followed_all_day_with_no_drop() {
        // Ten pictures a second for a day, each 60 ms an hour later than
        // the last by the two clocks alone. The floor follows the drift a
        // few microseconds a window; the bound never comes into it.
        let mut h = Harness::new(0.5);
        let rate = 0.06 / 3_600.0;
        let mut dropped = 0u32;
        for k in 0..24 * 3_600 * 10 {
            let at = 1_000.0 + f64::from(k) / 10.0;
            if !h.at(at + 0.25 + f64::from(k) / 10.0 * rate, at) {
                dropped += 1;
            }
        }
        assert_eq!(dropped, 0);
        let floor = h.leaky.floor().unwrap();
        assert!((floor - (0.25 + 24.0 * 0.06)).abs() < 1e-3, "{floor}");
        // The rows add up to the day's drift, SLIP at a time.
        let slipped: f64 = h
            .slips()
            .iter()
            .map(|row| row["by_s"].as_f64().unwrap())
            .sum();
        assert!((slipped - 24.0 * 0.06).abs() < 2.0 * SLIP, "{slipped}");
    }

    #[test]
    fn a_source_that_comes_back_earlier_is_followed_at_once() {
        let mut h = Harness::new(0.5);
        assert!(h.steady(0..300, |k| 0.75 + noise(k)).is_empty());
        assert!((h.leaky.floor().unwrap() - 0.75).abs() < 2e-3);
        // The source's clock jumps half a second ahead: from picture 300 on
        // each is stamped 0.5 s later and read at the wall it would have
        // been anyway, so each is 0.25 s late. The first passes and the
        // floor is its lateness at once; the rest pass.
        let mut dropped = 0;
        for k in 300..600 {
            if !h.at(made(k) + 0.75 + noise(k), made(k) + 0.5) {
                dropped += 1;
            }
            if k == 300 {
                let floor = h.leaky.floor().unwrap();
                assert!((floor - 0.25 - noise(300)).abs() < 1e-9, "{floor}");
            }
        }
        assert_eq!(dropped, 0);
        assert!((h.leaky.floor().unwrap() - 0.25).abs() < 2e-3);
        assert!(h.slips().is_empty());
    }

    #[test]
    fn a_stage_that_cannot_keep_up_does_not_raise_the_floor() {
        // Ten seconds keeping up, then the stage behind takes 60 ms over
        // every picture, for longer than the window is wide. Every picture
        // the node reads meanwhile was queued, and none of them counts: the
        // floor holds, within the noise of the last reading that did count,
        // and what passes is never further past it than the spread, the
        // bound and a pts step.
        let feed: Vec<(f64, f64)> = (0..1_200)
            .map(|k| (made(k) + 0.25 + noise(k), made(k)))
            .collect();
        let mut h = Harness::new(0.5);
        let served = h.serve(&feed, |wall| if wall < 1_010.0 { 1e-3 } else { 0.06 });
        let floor = h.leaky.floor().unwrap();
        assert!((0.25..0.263).contains(&floor), "{floor}");
        assert!(h.slips().is_empty(), "{:?}", h.slips());
        assert!(dropped(&served[300..]) > 300, "{}", dropped(&served[300..]));
        assert!(h.reports().iter().all(|row| {
            let field = |key: &str| row[key].as_f64().unwrap();
            field("lateness_s") <= field("baseline_s") + field("spread_s") + field("bound_s") + 0.04
        }));
    }

    #[test]
    fn a_steady_path_ends_with_a_tight_bound_and_a_bursty_one_with_a_looser_one() {
        let steady = settled().leaky.bound();
        assert!((0.01..0.1).contains(&steady), "{steady}");

        // The same feed with every fifth picture held back up to 0.3 s more.
        let mut h = Harness::new(0.5);
        h.steady(0..900, |k| {
            0.25 + noise(k)
                + if k % 5 == 0 {
                    f64::from(k % 7) * 0.05
                } else {
                    0.0
                }
        });
        let bursty = h.leaky.bound();
        assert!(bursty > 2.0 * steady, "{bursty} against {steady}");
        assert!(bursty <= 0.5, "{bursty}");

        // A path late by whole seconds now and then can loosen it no further
        // than max_lateness.
        let mut h = Harness::new(0.5);
        h.steady(0..900, |k| {
            0.25 + noise(k)
                + if k % 3 == 0 {
                    f64::from(k % 11) * 0.3
                } else {
                    0.0
                }
        });
        assert_eq!(h.leaky.bound(), 0.5);
    }

    #[test]
    fn dropped_pictures_feed_the_tail_and_one_fed_only_what_passed_stays_blind() {
        // A settled feed slips 0.3 s: under max_lateness over the floor it
        // had, so the tail may let it through once it has learned that
        // pictures now come that late. What the node drops teaches it that
        // within a couple of seconds, well before the floor turns over. An
        // estimate fed only with what it would itself pass never learns it,
        // and drops every picture until the floor has turned.
        let mut h = settled();
        let mut censored = Tail::new();
        censored.feed(0.0, made(0));
        for k in 1..900 {
            censored.feed(noise(k), made(k));
        }
        assert!((0.01..0.1).contains(&censored.bound(0.5)));
        let mut first_passed = None;
        let mut dropped = 0;
        let mut censored_dropped = 0;
        for k in 900..1_800 {
            let wall = made(k) + 0.55 + noise(k);
            if h.at(wall, made(k)) {
                first_passed.get_or_insert(k);
            } else {
                dropped += 1;
            }
            let excess = wall - made(k) - h.leaky.floor().unwrap();
            if excess <= censored.bound(0.5) {
                censored.feed(excess, made(k));
            } else {
                censored_dropped += 1;
            }
        }
        let first = first_passed.expect("the stream resumed");
        assert!((910..990).contains(&first), "{first}");
        assert!(dropped < 60, "{dropped}");
        // The censored estimate dropped the 103 pictures read before the
        // floor turned over, and holds a bound for the path as it was.
        assert!(censored_dropped >= 100, "{censored_dropped}");
        assert!(censored.bound(0.5) < 0.05, "{}", censored.bound(0.5));
        // With the floor caught up, the excess is noise again and the
        // node's bound tightens back.
        assert!((h.leaky.floor().unwrap() - 0.55).abs() < 2e-3);
        assert!(h.leaky.bound() < 0.1, "{}", h.leaky.bound());
    }

    fn video() -> Format {
        Format {
            media: Media::Video(VideoFormat {
                width: 16,
                height: 16,
                pix_fmt: "yuv420p",
                frame_len: 384,
                color: None,
            }),
            time_base: MILLIS,
        }
    }

    #[test]
    fn a_sound_stream_is_refused() {
        let sound = Format {
            media: Media::Audio(AudioFormat {
                sample_rate: 48_000,
                channels: 2,
                sample_fmt: "f32",
                channel_layout: None,
            }),
            time_base: MILLIS,
        };
        let refused = Leaky::open(&[], &sound).err().expect("sound is refused");
        assert!(refused.to_string().contains("never dropped"), "{refused}");
    }

    fn refuse(options: &[(&str, &str)]) -> String {
        let options: Vec<(String, String)> = options
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        match Leaky::open(&options, &video()) {
            Ok(_) => panic!("{options:?} opened"),
            Err(e) => e.to_string(),
        }
    }

    #[test]
    fn its_options_are_read_and_refused_by_name() {
        let leaky = Leaky::open(&[], &video()).unwrap();
        assert_eq!(leaky.limits, Limits::default());
        assert_eq!(
            (leaky.limits.max_lateness, leaky.limits.max_spread),
            (0.5, 2.0)
        );
        assert_eq!(leaky.node, "leaky");
        let given = [
            ("max_lateness".to_string(), "0.25".to_string()),
            ("max_spread".to_string(), "0".to_string()),
            ("node".to_string(), "n7".to_string()),
        ];
        let leaky = Leaky::open(&given, &video()).unwrap();
        assert_eq!(
            (leaky.limits.max_lateness, leaky.limits.max_spread),
            (0.25, 0.0)
        );
        assert_eq!(leaky.node.as_str(), "n7");

        assert!(refuse(&[("max_lateness", "0")]).contains("cannot be 0"));
        assert!(refuse(&[("max_lateness", "-1")]).contains("cannot be -1"));
        assert!(refuse(&[("max_lateness", "soon")]).contains("not a number"));
        assert!(refuse(&[("max_lateness", "1"), ("max_lateness", "2")]).contains("twice"));
        assert!(refuse(&[("latency", "1")]).contains("no option 'latency'"));
        assert!(refuse(&[("max_spread", "-1")]).contains("cannot be -1"));
        assert!(refuse(&[("max_spread", "inf")]).contains("cannot be inf"));
        assert!(refuse(&[("max_spread", "1"), ("max_spread", "1")]).contains("twice"));
    }
}
