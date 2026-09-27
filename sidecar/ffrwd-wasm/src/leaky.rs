//! `leaky`: the host's node that keeps a live picture near the wall clock.
//!
//! It is spelled like a module - `[a]leaky=max_lateness=<seconds>[b]` - and
//! wired like one, but nothing is compiled and nothing is instantiated. The
//! name and the option are GStreamer's: a leaky queue, and a sink's
//! max-lateness.
//!
//! A live pipeline stamps its pts onto the Unix epoch, so a frame's LATENESS
//! is the wall clock less its pts, in seconds. The smallest lateness seen so
//! far is the BASELINE: what a sender that started late, or a relay in
//! between, adds to every frame alike. A frame later than the baseline by
//! more than `max_lateness` is dropped; every other frame passes at once,
//! pixels, pts and rows untouched. Nothing is held and nothing is reordered,
//! so what leaves is the stream that arrived with its late frames taken out.
//! A frame behind one already passed is dropped too: the output's timestamps
//! never decrease.
//!
//! Once a second of wall time it writes a row saying what it did, on stderr
//! behind [`ROW_PREFIX`], where the run that started it reads it:
//! `{"kind":"leaky","node":<id>,"passed":<n>,"dropped":<n>,"lateness_s":<s>,
//! "baseline_s":<s>}`. The counts are the window's own; the two times are the
//! latest frame's lateness and the baseline, in seconds.
//!
//! The name is reserved: no `-m` binds it, because the host answers for it.

use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, bail, Result};
use ffrwd_wasm_runtime::runtime::{Format, Frame, Shape, TimeBase};
use serde_json::json;

/// The node name the grammar reserves.
pub const NODE: &str = "leaky";

/// How late a frame may be past the baseline, in seconds.
const MAX_LATENESS: &str = "max_lateness";

/// The node the rows name: the id the compiler gave it.
const NAME: &str = "node";

/// What `max_lateness` is when the node is not given one.
const DEFAULT_MAX_LATENESS: f64 = 0.5;

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

pub struct Leaky {
    max_lateness: f64,
    time_base: TimeBase,
    node: String,
    baseline: Option<f64>,
    /// The latest frame's lateness.
    lateness: f64,
    last_passed: Option<i64>,
    window: Option<Window>,
    clock: Clock,
    report: Report,
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
        let mut node: Option<String> = None;
        for (key, value) in options {
            match key.as_str() {
                MAX_LATENESS => {
                    if max_lateness.is_some() {
                        bail!("{NODE} is given the option '{MAX_LATENESS}' twice");
                    }
                    let seconds: f64 = value.parse().map_err(|_| {
                        anyhow!("{NODE}: {MAX_LATENESS} is not a number of seconds: {value}")
                    })?;
                    if !seconds.is_finite() || seconds <= 0.0 {
                        bail!(
                            "{NODE}: {MAX_LATENESS} is how many seconds late a picture may be, \
                             and cannot be {value}"
                        );
                    }
                    max_lateness = Some(seconds);
                }
                NAME => {
                    if node.is_some() {
                        bail!("{NODE} is given the option '{NAME}' twice");
                    }
                    node = Some(value.clone());
                }
                other => bail!(
                    "{NODE} has no option '{other}'; it takes {MAX_LATENESS}=<seconds> and \
                     {NAME}=<name>"
                ),
            }
        }
        Ok(Leaky::with(
            max_lateness.unwrap_or(DEFAULT_MAX_LATENESS),
            format.time_base,
            node.unwrap_or_else(|| NODE.to_string()),
            Box::new(wall),
            Box::new(to_stderr),
        ))
    }

    /// A node reading `clock` and writing its rows to `report`.
    pub fn with(
        max_lateness: f64,
        time_base: TimeBase,
        node: String,
        clock: Clock,
        report: Report,
    ) -> Leaky {
        Leaky {
            max_lateness,
            time_base,
            node,
            baseline: None,
            lateness: 0.0,
            last_passed: None,
            window: None,
            clock,
            report,
        }
    }

    /// Whether the frame at `pts` passes, read against the clock now.
    pub fn judge(&mut self, pts: i64) -> bool {
        let now = (self.clock)();
        let at = pts as f64 * self.time_base.num as f64 / self.time_base.den.max(1) as f64;
        let lateness = now - at;
        let baseline = self.baseline.map_or(lateness, |seen| seen.min(lateness));
        self.baseline = Some(baseline);
        self.lateness = lateness;
        let behind = self.last_passed.is_some_and(|last| pts < last);
        let passes = !behind && lateness <= baseline + self.max_lateness;
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
            let now = Arc::new(Mutex::new(0.0));
            let rows = Arc::new(Mutex::new(Vec::new()));
            let clock = Arc::clone(&now);
            let written = Arc::clone(&rows);
            let leaky = Leaky::with(
                max_lateness,
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
        assert_eq!(leaky.max_lateness, 0.5);
        assert_eq!(leaky.node, "leaky");
        let given = [
            ("max_lateness".to_string(), "0.25".to_string()),
            ("node".to_string(), "n7".to_string()),
        ];
        let leaky = Leaky::open(&given, &video()).unwrap();
        assert_eq!((leaky.max_lateness, leaky.node.as_str()), (0.25, "n7"));

        assert!(refuse(&[("max_lateness", "0")]).contains("cannot be 0"));
        assert!(refuse(&[("max_lateness", "-1")]).contains("cannot be -1"));
        assert!(refuse(&[("max_lateness", "soon")]).contains("not a number"));
        assert!(refuse(&[("max_lateness", "1"), ("max_lateness", "2")]).contains("twice"));
        assert!(refuse(&[("latency", "1")]).contains("no option 'latency'"));
    }
}
