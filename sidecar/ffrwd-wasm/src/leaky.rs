//! `leaky`: the host's node that keeps a live picture near the wall clock.
//!
//! It is spelled like a module -
//! `[a]leaky=max_lateness=<seconds>:max_spread=<seconds>[b]` - and wired like
//! one, but nothing is compiled and nothing is instantiated. The name and
//! the first option are GStreamer's: a leaky queue, and a sink's
//! max-lateness.
//!
//! A live pipeline stamps its pts onto the Unix epoch, so a frame's LATENESS
//! is the wall clock less its pts, in seconds. The smallest lateness seen so
//! far is the BASELINE: what a sender that started late, or a relay in
//! between, adds to every frame alike. A frame later than the baseline by
//! more than its SPREAD plus `max_lateness` is dropped; every other frame
//! passes at once, pixels, pts and rows untouched. Nothing is held and
//! nothing is reordered, so what leaves is the stream that arrived with its
//! late frames taken out. A frame behind one already passed is dropped too:
//! the output's timestamps never decrease.
//!
//! The spread is what the input's own delivery adds. A relay that hands on a
//! whole group of pictures at once (MoQ hands a subscriber a second of them
//! in a few milliseconds) makes the group's first picture a second later
//! than its last, and none of that is the node falling behind. So the node
//! cuts what it reads into RUNS, pictures each arriving less than half its
//! own pts step after the one before, and a run's WIDTH is how far its
//! lateness ranges. A run counts only when it ends caught up, its freshest
//! picture within half of `max_lateness` of the baseline: a backlog drained
//! after a slow stage behind the node also arrives at once, but ends near
//! the edge of the budget, and so teaches nothing. The spread is the widest
//! of the last [`KEEP`] counted runs, each capped at `max_spread`. A steady
//! feed's runs are single pictures, so its spread is 0 and it is judged as
//! it always was; the first run, the reader's own probe backlog as a rule,
//! is forgotten as soon as another counts.
//!
//! Once a second of wall time it writes a row saying what it did, on stderr
//! behind [`ROW_PREFIX`], where the run that started it reads it:
//! `{"kind":"leaky","node":<id>,"passed":<n>,"dropped":<n>,"lateness_s":<s>,
//! "baseline_s":<s>,"spread_s":<s>}`. The counts are the window's own; the
//! times are the latest frame's lateness, the baseline and the spread, in
//! seconds.
//!
//! The name is reserved: no `-m` binds it, because the host answers for it.

use std::collections::VecDeque;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, bail, Result};
use ffrwd_wasm_runtime::runtime::{Format, Frame, Shape, TimeBase};
use serde_json::json;

/// The node name the grammar reserves.
pub const NODE: &str = "leaky";

/// How late a frame may be past the baseline and the spread, in seconds.
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

/// What a row on stderr starts with, so a reader tells it from the log.
pub const ROW_PREFIX: &str = "ffrwd:row ";

/// How much wall time one row covers, in seconds.
const REPORT_EVERY: f64 = 1.0;

/// How the host drives it: a frame at a time, and it hands on fewer frames
/// than it reads. Its baseline is state, so one instance sees every frame.
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

/// Pictures arriving faster than they were made: when the latest was read
/// and made, and the most and least lateness among them.
struct Run {
    wall: f64,
    at: f64,
    most: f64,
    least: f64,
}

/// What a node is opened with, in seconds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Limits {
    /// How late past the baseline and the spread a picture may be.
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
    baseline: Option<f64>,
    /// The latest frame's lateness.
    lateness: f64,
    last_passed: Option<i64>,
    /// The run the latest frame belongs to, still open.
    run: Option<Run>,
    /// The widths of the last [`KEEP`] counted runs, each capped.
    widths: VecDeque<f64>,
    /// How many runs have counted.
    counted: u64,
    window: Option<Window>,
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
            format.time_base,
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
            baseline: None,
            lateness: 0.0,
            last_passed: None,
            run: None,
            widths: VecDeque::with_capacity(KEEP),
            counted: 0,
            window: None,
            clock,
            report,
        }
    }

    /// How much later than the baseline the input's own delivery makes a
    /// picture: the widest of the last counted runs, 0 before any.
    pub fn spread(&self) -> f64 {
        self.widths.iter().copied().fold(0.0, f64::max)
    }

    /// Whether the frame at `pts` passes, read against the clock now.
    pub fn judge(&mut self, pts: i64) -> bool {
        let now = (self.clock)();
        let at = pts as f64 * self.time_base.num as f64 / self.time_base.den.max(1) as f64;
        let lateness = now - at;
        self.follow(now, at, lateness);
        let baseline = self.baseline.map_or(lateness, |seen| seen.min(lateness));
        self.baseline = Some(baseline);
        self.lateness = lateness;
        let behind = self.last_passed.is_some_and(|last| pts < last);
        let budget = self.spread() + self.limits.max_lateness;
        let passes = !behind && lateness <= baseline + budget;
        if passes {
            self.last_passed = Some(pts);
        }
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
        passes
    }

    /// The frame made at `at` and read at `now`, both in seconds, carries on
    /// the open run, or closes it and starts the next.
    fn follow(&mut self, now: f64, at: f64, lateness: f64) {
        if let Some(run) = &mut self.run {
            // Read less than half its own pts step after the one before: the
            // two arrived faster than they were made, together.
            if at > run.at && now - run.wall < (at - run.at) / 2.0 {
                run.wall = now;
                run.at = at;
                run.most = run.most.max(lateness);
                run.least = run.least.min(lateness);
                return;
            }
        }
        if let Some(run) = self.run.take() {
            self.close(&run);
        }
        self.run = Some(Run {
            wall: now,
            at,
            most: lateness,
            least: lateness,
        });
    }

    /// A run is over. It counts when its freshest picture came within half
    /// of `max_lateness` of the baseline: the node had caught up.
    fn close(&mut self, run: &Run) {
        let Some(baseline) = self.baseline else {
            return;
        };
        if run.least > baseline + self.limits.max_lateness / 2.0 {
            return;
        }
        if self.counted == 1 {
            // The first run is the reader's own probe backlog as a rule, as
            // wide as it probed: it stands only until another counts.
            self.widths.clear();
        }
        self.counted += 1;
        if self.widths.len() == KEEP {
            self.widths.pop_front();
        }
        self.widths
            .push_back((run.most - run.least).min(self.limits.max_spread));
    }

    /// One frame through, or None where it was too late.
    pub fn pass(&mut self, frame: Frame) -> Option<Frame> {
        self.judge(frame.pts).then_some(frame)
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
            "baseline_s": millis(self.baseline.unwrap_or(0.0)),
            "spread_s": millis(self.spread()),
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
    }

    /// Picture `k` of a feed made at 30 a second, in seconds.
    fn made(k: u32) -> f64 {
        1_000.0 + f64::from(k) / 30.0
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
    fn a_sender_that_starts_late_is_absorbed_by_the_baseline() {
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
        assert!((h.leaky.baseline.unwrap() - 7.0).abs() < 1e-3);
    }

    #[test]
    fn the_baseline_is_the_smallest_lateness_seen_so_far() {
        let mut h = Harness::new(0.5);
        assert!(h.at(10.3, 10.0));
        assert!(h.at(10.4, 10.2)); // 0.2: earlier than any before it
        assert!((h.leaky.baseline.unwrap() - 0.2).abs() < 1e-9);
        // 0.6 is past 0.2 by 0.4, inside the budget: it does not move the
        // baseline, which never rises.
        assert!(h.at(11.0, 10.4));
        assert!((h.leaky.baseline.unwrap() - 0.2).abs() < 1e-9);
    }

    #[test]
    fn a_frame_later_than_the_baseline_by_more_than_the_budget_is_dropped() {
        let mut h = Harness::new(0.5);
        assert!(h.at(100.1, 100.0)); // the baseline: 0.1
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
            .map(|k| (made(k) + 0.25 + f64::from((k * 7) % 13) / 1_000.0, made(k)))
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
        let rows = h.rows();
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
