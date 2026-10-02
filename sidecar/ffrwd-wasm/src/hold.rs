//! Hold pairing: a source's frames paired by time onto a node's clock.
//!
//! One [`Group`] serves one hold input, or every hold input of one `group`
//! (a feeder's picture and sound on one connection). Frames arrive as the
//! source delivers them and queue per member; each tick, a video member
//! hands the newest frame at or before the tick, repeating the last one
//! while the source is behind and skipping forward when it catches up, and
//! an audio member hands the tick's samples re-cut from what arrived, or
//! none while the source is behind.
//!
//! A FEED is one source from the tick its start is fixed to the last tick
//! its last frame shows on. The offset between source time and clock time
//! is fixed once per feed, by the group's lead member (its picture, or
//! where the source brings none its first member that it does bring): a
//! `shared-clock` source's pts are clock time, so its first frame waits for
//! the clock to reach it; a `first-frame` source is primed until `lead`
//! seconds of it are held, or its end, and then scheduled `lead` ahead of
//! the clock; `tagged` chooses between the two by the source's tags. A
//! source that jumps backwards, or forwards by more than a second, is a new
//! source; so is each connection of a port feed. `linger` keeps the last
//! frame after the source ends, and `timeout` ends a feed that stops
//! sending. Where the host can foretell the last tick a feed shows on, it
//! says so (`ends`), counted on the clock's grid.
//!
//! What each feed did is said on stderr, as a line and as a row behind
//! [`crate::leaky::ROW_PREFIX`].

use std::collections::VecDeque;
use std::sync::Arc;

use ffrwd_wasm_runtime::node::{Anchor, Feed, FeedStart, Hold, PortKind, TickFrame};
use ffrwd_wasm_runtime::runtime::{AudioFormat, StreamInfo, TimeBase};
use serde_json::json;

use crate::leaky::ROW_PREFIX;

/// How many frames of a source are held ahead of the clock, and how many
/// bytes: a port feed that outruns the clock waits on its socket past
/// either.
pub const MAX_HELD_FRAMES: usize = 600;
pub const MAX_HELD_BYTES: usize = 96 << 20;

/// A step in a source's time, or the clock's, that is backwards or further
/// forward than this is a discontinuity.
pub const MAX_STEP_SECONDS: f64 = 1.0;

/// Where a report goes, one line at a time.
pub type Report = Box<dyn FnMut(String) + Send>;

fn to_stderr(line: String) {
    eprintln!("{line}");
}

/// What a source says about itself when it starts.
#[derive(Debug, Clone)]
pub struct SourceInfo {
    /// The connection it arrived on, so a group's members attach to one
    /// source; 0 for a stream bound in the plan.
    pub connection: u64,
    pub tags: Vec<(String, String)>,
    pub base: TimeBase,
    pub info: StreamInfo,
}

/// Why a feed ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Why {
    /// The source ended and its last frame has shown.
    Ended,
    /// Nothing arrived for `timeout` seconds of clock time.
    Timeout,
    /// The clock jumped.
    Jumped,
}

impl Why {
    fn said(self) -> &'static str {
        match self {
            Why::Ended => "the source ended",
            Why::Timeout => "the source stopped sending",
            Why::Jumped => "the clock jumped",
        }
    }
}

/// The clock's ticks as a grid: where a tick falls that has not been made
/// yet. A rate clock's is exact; an input clock's is learned from its
/// frames, and starts again at a step that is not one frame.
#[derive(Debug, Clone, Copy)]
pub struct Grid {
    exact: bool,
    origin: i64,
    count: i64,
    last: i64,
}

impl Grid {
    /// A rate clock's: tick `k` at `k`.
    pub fn exact() -> Grid {
        Grid {
            exact: true,
            origin: 0,
            count: 0,
            last: 0,
        }
    }

    /// An input clock's, before any tick.
    pub fn learned() -> Grid {
        Grid {
            exact: false,
            origin: 0,
            count: -1,
            last: 0,
        }
    }

    /// A tick made at `pts`.
    pub fn observe(&mut self, pts: i64) {
        if self.exact {
            return;
        }
        if self.count < 0 {
            self.origin = pts;
            self.count = 0;
            self.last = pts;
            return;
        }
        let step = pts.saturating_sub(self.last);
        let one = if self.count > 0 {
            (self.last - self.origin) / self.count
        } else {
            step
        };
        if step > 0 && (self.count == 0 || (step - one).abs() * 4 < one) {
            self.count += 1;
        } else {
            self.origin = pts;
            self.count = 0;
        }
        self.last = pts;
    }

    fn known(&self) -> bool {
        self.exact || self.count >= 1
    }

    fn tick(&self, k: i128) -> i64 {
        if self.exact {
            return k.clamp(i64::MIN as i128, i64::MAX as i128) as i64;
        }
        let span = (self.last - self.origin) as i128;
        let n = self.count as i128;
        (self.origin as i128 + (k * span + n / 2).div_euclid(n))
            .clamp(i64::MIN as i128, i64::MAX as i128) as i64
    }

    fn index_near(&self, at: i64) -> i128 {
        if self.exact {
            return at as i128;
        }
        let span = ((self.last - self.origin) as i128).max(1);
        ((at - self.origin) as i128 * self.count as i128).div_euclid(span)
    }

    /// The first tick at or past `at`.
    pub fn at_or_after(&self, at: i64) -> Option<i64> {
        if !self.known() {
            return None;
        }
        let mut k = self.index_near(at);
        while self.tick(k) >= at && self.tick(k - 1) >= at {
            k -= 1;
        }
        while self.tick(k) < at {
            k += 1;
        }
        Some(self.tick(k))
    }

    /// The last tick before `at`.
    pub fn before(&self, at: i64) -> Option<i64> {
        let next = self.at_or_after(at)?;
        let mut k = self.index_near(next);
        while self.tick(k) < next {
            k += 1;
        }
        while self.tick(k) >= next {
            k -= 1;
        }
        Some(self.tick(k))
    }

    /// The first tick after `at`.
    pub fn after(&self, at: i64) -> Option<i64> {
        self.at_or_after(at.saturating_add(1))
    }
}

/// `seconds` as ticks of `base`, rounded up.
pub fn ticks_of(seconds: f64, base: TimeBase) -> i64 {
    (seconds * base.den as f64 / base.num.max(1) as f64).ceil() as i64
}

fn gcd(a: i128, b: i128) -> i128 {
    let (mut a, mut b) = (a.abs(), b.abs());
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a.max(1)
}

/// A feed's one offset, clock time less source time, in seconds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Offset {
    num: i128,
    den: i128,
}

impl Offset {
    const ZERO: Offset = Offset { num: 0, den: 1 };

    /// The offset that lays `src` (in `src_base`) at `clock` (in `clock_base`).
    fn between(src: i64, src_base: TimeBase, clock: i64, clock_base: TimeBase) -> Offset {
        let num = clock as i128 * clock_base.num as i128 * src_base.den as i128
            - src as i128 * src_base.num as i128 * clock_base.den as i128;
        let den = clock_base.den as i128 * src_base.den as i128;
        let g = gcd(num, den);
        Offset {
            num: num / g,
            den: den / g,
        }
    }

    /// Source time `t` in `base` on the clock, in `clock_base`, rounded down.
    fn to_clock(self, t: i64, base: TimeBase, clock_base: TimeBase) -> i64 {
        let num = (t as i128 * base.num as i128 * self.den + self.num * base.den as i128)
            * clock_base.den as i128;
        let den = base.den as i128 * self.den * clock_base.num as i128;
        num.div_euclid(den.max(1))
            .clamp(i64::MIN as i128, i64::MAX as i128) as i64
    }

    /// The first clock time at or past source time `t`: where a frame at
    /// `t` first shows.
    fn to_clock_up(self, t: i64, base: TimeBase, clock_base: TimeBase) -> i64 {
        let num = (t as i128 * base.num as i128 * self.den + self.num * base.den as i128)
            * clock_base.den as i128;
        let den = (base.den as i128 * self.den * clock_base.num as i128).max(1);
        (num + den - 1)
            .div_euclid(den)
            .clamp(i64::MIN as i128, i64::MAX as i128) as i64
    }

    /// Clock time `p` in `clock_base` as source time in `base`, rounded down.
    fn to_source(self, p: i64, clock_base: TimeBase, base: TimeBase) -> i64 {
        let num = (p as i128 * clock_base.num as i128 * self.den
            - self.num * clock_base.den as i128)
            * base.den as i128;
        let den = clock_base.den as i128 * self.den * base.num as i128;
        num.div_euclid(den.max(1))
            .clamp(i64::MIN as i128, i64::MAX as i128) as i64
    }
}

/// One member's frames of one source.
struct MemberQueue {
    base: TimeBase,
    info: StreamInfo,
    queue: VecDeque<TickFrame>,
    bytes: usize,
    /// The last frame's pts, or the end of the last run of samples.
    last: Option<i64>,
}

impl MemberQueue {
    fn new(base: TimeBase, info: StreamInfo) -> MemberQueue {
        MemberQueue {
            base,
            info,
            queue: VecDeque::new(),
            bytes: 0,
            last: None,
        }
    }

    fn push(&mut self, frame: TickFrame) {
        self.bytes += frame.data.len();
        self.queue.push_back(frame);
    }

    fn pop(&mut self) -> Option<TickFrame> {
        let frame = self.queue.pop_front()?;
        self.bytes -= frame.data.len();
        Some(frame)
    }
}

/// One source: one connection, or one stretch of a stream between
/// discontinuities.
struct Source {
    connection: u64,
    tags: Vec<(String, String)>,
    members: Vec<MemberQueue>,
    /// Per member, whether the source brings it: a port feed's connection
    /// whose sound was refused never does.
    brings: Vec<bool>,
    /// The member that fixes this source's offset.
    lead: usize,
    closed: bool,
}

/// One member of a group: a hold input.
pub struct Member {
    pub name: String,
    pub kind: PortKind,
    pub base: TimeBase,
    pub info: StreamInfo,
    pub audio: Option<AudioFormat>,
}

/// The feed on the group's front source.
struct FeedState {
    offset: Offset,
    /// The lead member's first frame, and the clock time it stands at.
    first_pts: i64,
    at: i64,
    /// The clock time of the tick the offset was fixed on, or `at` where
    /// that is earlier: a source the clock has passed shows at once.
    known: i64,
    shown: Vec<Option<TickFrame>>,
    /// The clock time the lead member last moved on, for the timeout.
    advanced: i64,
    /// The clock time the material ran out, when it has: the linger runs
    /// from here, and `why` is why.
    drained: Option<(i64, Why)>,
    ends: Option<i64>,
    frames_shown: u64,
    repeated: u64,
    skipped: u64,
}

/// One hold input, or one group of them.
pub struct Group {
    hold: Hold,
    node: String,
    port_fed: bool,
    members: Vec<Member>,
    lead: usize,
    sources: VecDeque<Source>,
    feed: Option<FeedState>,
    /// The last tick's time and the clock's base, for the room a held
    /// source gets.
    now: Option<(i64, TimeBase)>,
    report: Report,
}

/// What one member hands for one tick.
#[derive(Debug, Clone, Default)]
pub struct Handed {
    pub frames: Vec<TickFrame>,
    pub feed: Option<Feed>,
    pub info: Option<(StreamInfo, TimeBase)>,
}

impl Group {
    pub fn new(hold: Hold, node: &str, members: Vec<Member>, port_fed: bool) -> Group {
        let lead = members
            .iter()
            .position(|m| m.kind == PortKind::Video)
            .unwrap_or(0);
        Group {
            hold,
            node: node.to_string(),
            port_fed,
            members,
            lead,
            sources: VecDeque::new(),
            feed: None,
            now: None,
            report: Box::new(to_stderr),
        }
    }

    #[cfg(test)]
    pub fn with_report(mut self, report: Report) -> Group {
        self.report = report;
        self
    }

    pub fn port_fed(&self) -> bool {
        self.port_fed
    }

    fn name(&self) -> String {
        let names: Vec<&str> = self.members.iter().map(|m| m.name.as_str()).collect();
        format!("{} '{}'", self.node, names.join("', '"))
    }

    fn open_source(&mut self, connection: u64, tags: Vec<(String, String)>) {
        if let Some(back) = self.sources.back_mut() {
            back.closed = true;
        }
        let members = self
            .members
            .iter()
            .map(|m| MemberQueue::new(m.base, m.info.clone()))
            .collect();
        self.sources.push_back(Source {
            connection,
            tags,
            members,
            brings: vec![connection == 0; self.members.len()],
            lead: self.lead,
            closed: false,
        });
    }

    /// A source starts on `member`'s stream: a port feed's connection, told
    /// before its frames.
    pub fn source_open(&mut self, member: usize, source: SourceInfo) {
        let attached = self
            .sources
            .back()
            .is_some_and(|s| s.connection == source.connection && !s.closed);
        if !attached {
            self.open_source(source.connection, source.tags.clone());
        }
        let back = self.sources.back_mut().expect("opened");
        back.brings[member] = true;
        back.lead = if back.brings[self.lead] {
            self.lead
        } else {
            back.brings.iter().position(|b| *b).unwrap_or(self.lead)
        };
        let queue = &mut back.members[member];
        queue.base = source.base;
        queue.info = source.info;
    }

    /// The source on `member`'s stream has ended; what it sent still plays
    /// out.
    pub fn source_close(&mut self, member: usize) {
        if let Some(back) = self.sources.back_mut().filter(|s| s.lead == member) {
            back.closed = true;
        }
    }

    /// The source that follows a jump in the current one: the same
    /// connection, tags and time bases, from the frame after the jump.
    fn split_source(&mut self) {
        let back = self.sources.back_mut().expect("a source to split");
        back.closed = true;
        let next = Source {
            connection: back.connection,
            tags: back.tags.clone(),
            members: back
                .members
                .iter()
                .map(|q| MemberQueue::new(q.base, q.info.clone()))
                .collect(),
            brings: back.brings.clone(),
            lead: back.lead,
            closed: false,
        };
        self.sources.push_back(next);
    }

    /// One frame of `member`'s stream.
    pub fn arrive(&mut self, member: usize, frame: TickFrame) {
        let back_open = self.sources.back().is_some_and(|s| !s.closed);
        if !back_open {
            let tags = self.members[member].info.tags.clone();
            self.open_source(0, tags);
        }
        let back = self.sources.back().expect("opened");
        if member == back.lead {
            let queue = &back.members[member];
            let step_limit = ticks_of(MAX_STEP_SECONDS, queue.base);
            if let Some(last) = queue.last {
                if frame.pts < last || frame.pts.saturating_sub(last) > step_limit {
                    self.split_source();
                }
            }
        }
        let queue = &mut self.sources.back_mut().expect("opened").members[member];
        queue.last = Some(match self.members[member].kind {
            PortKind::Audio => frame.pts.saturating_add(frame.duration.unwrap_or(0)),
            _ => frame.pts,
        });
        queue.push(frame);
    }

    /// How many more frames the source may send before it waits: none while
    /// a `shared-clock` source's first frame is held for its time, past its
    /// first picture and its first sound.
    pub fn room(&self) -> usize {
        let Some(source) = self.sources.back() else {
            return MAX_HELD_FRAMES;
        };
        let held = source.members.iter().map(|m| m.queue.len()).sum::<usize>();
        let bytes = source.members.iter().map(|m| m.bytes).sum::<usize>();
        if held >= MAX_HELD_FRAMES || bytes >= MAX_HELD_BYTES {
            return 0;
        }
        let waiting = match (&self.feed, self.now) {
            (Some(feed), Some((now, _))) => now < feed.at,
            (None, Some((now, clock_base))) => {
                let lead = &source.members[source.lead];
                lead.queue.front().is_none_or(|first| {
                    now < Offset::ZERO.to_clock_up(first.pts, lead.base, clock_base)
                })
            }
            _ => true,
        };
        let shared = self.anchor_of(source) == Anchor::SharedClock;
        if self.port_fed && shared && waiting && self.sources.len() == 1 {
            if source.members[source.lead].queue.is_empty() {
                return 1;
            }
            let sound_seen = source
                .members
                .iter()
                .zip(&self.members)
                .all(|(q, m)| m.kind != PortKind::Audio || !q.queue.is_empty());
            return usize::from(!sound_seen);
        }
        MAX_HELD_FRAMES - held
    }

    fn anchor_of(&self, source: &Source) -> Anchor {
        match &self.hold.anchor {
            Anchor::Tagged(name) => {
                let tagged = source.tags.iter().any(|(k, v)| k == name && v == "1");
                if tagged {
                    Anchor::SharedClock
                } else {
                    Anchor::FirstFrame
                }
            }
            other => other.clone(),
        }
    }

    /// Whether a `first-frame` source holds `lead` seconds of frames, or
    /// has ended with some.
    fn primed(&self, source: &Source) -> bool {
        let lead = &source.members[source.lead];
        let (Some(first), Some(last)) = (lead.queue.front(), lead.queue.back()) else {
            return false;
        };
        if source.closed {
            return true;
        }
        last.pts.saturating_sub(first.pts) >= ticks_of(self.hold.lead, lead.base)
    }

    /// Whether what has arrived settles the tick from `pts` to `end` (None:
    /// to the end of the inputs), for a source bound in the plan: the feed
    /// has a frame past the interval, or its source ended, or nothing has
    /// arrived and `reference`, how far the clock has arrived, is `timeout`
    /// past the interval. A port feed never holds a tick.
    pub fn ready(
        &self,
        pts: i64,
        end: Option<i64>,
        clock_base: TimeBase,
        reference: Option<i64>,
    ) -> bool {
        if self.port_fed {
            return true;
        }
        let timed_out = || match (self.hold.timeout, reference, end) {
            (Some(timeout), Some(reference), Some(end)) => {
                reference >= end.saturating_add(ticks_of(timeout, clock_base))
            }
            _ => false,
        };
        let Some(source) = self.sources.front() else {
            return timed_out();
        };
        if source.closed || self.sources.len() > 1 {
            return true;
        }
        let until = |kind: PortKind| match kind {
            PortKind::Audio => end.unwrap_or(pts),
            _ => pts,
        };
        match &self.feed {
            Some(feed) => {
                let settled = source
                    .members
                    .iter()
                    .zip(&self.members)
                    .zip(&source.brings)
                    .all(|((q, m), brings)| {
                        let wanted = feed.offset.to_source(until(m.kind), clock_base, q.base);
                        !brings || q.last.is_some_and(|last| last > wanted)
                    });
                settled || timed_out()
            }
            None => match self.anchor_of(source) {
                Anchor::FirstFrame => self.primed(source) || timed_out(),
                _ => {
                    let lead = &source.members[source.lead];
                    let kind = self.members[source.lead].kind;
                    let wanted = Offset::ZERO.to_source(until(kind), clock_base, lead.base);
                    lead.last.is_some_and(|last| last > wanted) || timed_out()
                }
            },
        }
    }

    fn end_feed(&mut self, pts: i64, clock_base: TimeBase, why: Why) {
        let Some(feed) = self.feed.take() else {
            return;
        };
        let lead = self.sources.front().map_or(self.lead, |s| s.lead);
        self.sources.pop_front();
        let seconds = |t: i64| t as f64 * clock_base.num as f64 / clock_base.den as f64;
        let line = format!(
            "hold: {}: the feed ended at {:.3}s ({}), {} frames shown, {} repeated, {} skipped",
            self.name(),
            seconds(pts),
            why.said(),
            feed.frames_shown,
            feed.repeated,
            feed.skipped
        );
        (self.report)(line);
        let row = json!({
            "kind": "feed",
            "node": self.node,
            "input": self.members[lead].name,
            "event": "end",
            "at": millis(seconds(pts)),
            "why": match why {
                Why::Ended => "ended",
                Why::Timeout => "timeout",
                Why::Jumped => "jumped",
            },
            "shown": feed.frames_shown,
            "repeated": feed.repeated,
            "skipped": feed.skipped,
        });
        (self.report)(format!("{ROW_PREFIX}{row}"));
    }

    /// The clock jumped: every feed ends.
    pub fn clock_jumped(&mut self, pts: i64, clock_base: TimeBase) {
        self.end_feed(pts, clock_base, Why::Jumped);
    }

    fn start_feed(&mut self, pts: i64, clock_base: TimeBase, grid: &Grid) {
        let Some(source) = self.sources.front() else {
            return;
        };
        let anchor = self.anchor_of(source);
        let lead = &source.members[source.lead];
        let Some(first) = lead.queue.front() else {
            if source.closed {
                self.sources.pop_front();
            }
            return;
        };
        let (first_pts, lead_base) = (first.pts, lead.base);
        let (offset, at) = match anchor {
            Anchor::SharedClock => (
                Offset::ZERO,
                Offset::ZERO.to_clock_up(first_pts, lead_base, clock_base),
            ),
            _ => {
                if !self.primed(source) {
                    return;
                }
                let scheduled = pts.saturating_add(ticks_of(self.hold.lead, clock_base));
                let at = grid.at_or_after(scheduled).unwrap_or(scheduled);
                (Offset::between(first_pts, lead_base, at, clock_base), at)
            }
        };
        let seconds = |t: i64, base: TimeBase| t as f64 * base.num as f64 / base.den as f64;
        let line = format!(
            "hold: {}: the feed comes up at {:.3}s, mapped from {:.3}s ({})",
            self.name(),
            seconds(at, clock_base),
            seconds(first_pts, lead_base),
            match anchor {
                Anchor::SharedClock => "its pts are clock time".to_string(),
                _ => format!("lead {}s", self.hold.lead),
            }
        );
        let mut row = json!({
            "kind": "feed",
            "node": self.node,
            "input": self.members[source.lead].name,
            "event": "start",
            "at": millis(seconds(at, clock_base)),
            "first_pts": millis(seconds(first_pts, lead_base)),
            "anchor": match anchor {
                Anchor::SharedClock => "shared-clock",
                _ => "first-frame",
            },
            "tags": source.tags.iter().map(|(k, v)| json!([k, v])).collect::<Vec<_>>(),
        });
        let absent: Vec<&str> = self
            .members
            .iter()
            .zip(&source.brings)
            .enumerate()
            .filter(|(index, (_, brings))| *index != source.lead && !**brings)
            .map(|(_, (member, _))| member.name.as_str())
            .collect();
        if !absent.is_empty() {
            row["absent"] = json!(absent);
        }
        (self.report)(line);
        (self.report)(format!("{ROW_PREFIX}{row}"));
        self.feed = Some(FeedState {
            offset,
            first_pts,
            at,
            known: pts.min(at),
            shown: vec![None; self.members.len()],
            advanced: pts.max(at),
            drained: None,
            ends: None,
            frames_shown: 0,
            repeated: 0,
            skipped: 0,
        });
    }

    /// Where the feed's last frame shows for the last time, once the source
    /// has ended or timed out, on the clock's grid.
    fn foretell(&self, feed: &FeedState, clock_base: TimeBase, grid: &Grid) -> Option<i64> {
        let linger = self.hold.linger.map(|s| ticks_of(s, clock_base));
        if let Some((drained, _)) = feed.drained {
            return match linger {
                Some(linger) => grid.before(drained.saturating_add(linger)),
                None => grid.before(drained),
            };
        }
        let source = self.sources.front()?;
        if !source.closed {
            return None;
        }
        let lead = &source.members[source.lead];
        let last = match lead.queue.back() {
            Some(frame) => frame.pts,
            None => feed.shown[source.lead].as_ref()?.pts,
        };
        let turn = feed
            .offset
            .to_clock_up(last, lead.base, clock_base)
            .max(feed.at);
        let shows = grid.at_or_after(turn)?;
        match linger {
            Some(linger) => {
                let drained = grid.after(shows)?;
                grid.before(drained.saturating_add(linger))
            }
            None => Some(shows),
        }
    }

    /// The tick from `pts` to `end` in `clock_base`: what every member hands,
    /// in member order.
    pub fn tick(
        &mut self,
        pts: i64,
        end: Option<i64>,
        clock_base: TimeBase,
        grid: &Grid,
    ) -> Vec<Handed> {
        self.now = Some((pts, clock_base));
        if self.feed.is_none() {
            self.start_feed(pts, clock_base, grid);
        }
        let Some(mut feed) = self.feed.take() else {
            return vec![Handed::default(); self.members.len()];
        };
        let source = self.sources.front_mut().expect("a feed has a source");
        let lead = source.lead;
        let live = pts >= feed.at;
        let mut handed: Vec<Handed> = Vec::with_capacity(self.members.len());
        let mut lead_moved = false;
        for (index, member) in self.members.iter().enumerate() {
            if index != lead && !source.brings[index] {
                handed.push(Handed::default());
                continue;
            }
            let queue = &mut source.members[index];
            let mut frames = Vec::new();
            if live {
                match member.kind {
                    PortKind::Audio => {
                        let audio = member.audio.expect("an audio member has a format");
                        let s0 = feed.offset.to_source(pts, clock_base, queue.base);
                        let s1 = end.map(|end| feed.offset.to_source(end, clock_base, queue.base));
                        if let Some(run) = recut_window(queue, audio, s0, s1) {
                            if index == lead {
                                lead_moved = true;
                            }
                            frames.push(run);
                        }
                    }
                    _ => {
                        let target = feed.offset.to_source(pts, clock_base, queue.base);
                        let mut popped = 0;
                        while queue.queue.front().is_some_and(|f| f.pts <= target) {
                            feed.shown[index] = queue.pop();
                            popped += 1;
                        }
                        if index == lead {
                            if popped > 0 {
                                lead_moved = true;
                                feed.skipped += popped - 1;
                            } else if feed.shown[index].is_some() {
                                feed.repeated += 1;
                            }
                        }
                        if let Some(shown) = &feed.shown[index] {
                            frames.push(shown.clone());
                        }
                    }
                }
            }
            // Every member's pair is the lead's offset, so a member that
            // starts off the clock's grid still maps its pts exactly.
            let first = if index == lead {
                feed.first_pts
            } else {
                feed.offset.to_source(feed.at, clock_base, queue.base)
            };
            handed.push(Handed {
                frames,
                feed: Some(Feed {
                    start: FeedStart {
                        tags: source.tags.clone(),
                        first_pts: first,
                        at: feed.at,
                        known: feed.known,
                    },
                    ends: None,
                }),
                info: Some((
                    StreamInfo {
                        tags: source.tags.clone(),
                        ..queue.info.clone()
                    },
                    queue.base,
                )),
            });
        }
        if live && lead_moved {
            feed.advanced = pts;
            feed.frames_shown += 1;
        }
        let lead_empty = source.members[lead].queue.is_empty();
        let timeout = self.hold.timeout.map(|s| ticks_of(s, clock_base));
        let linger = self.hold.linger.map(|s| ticks_of(s, clock_base));
        let mut over = None;
        if live {
            if let Some((drained, why)) = feed.drained {
                if linger.is_none_or(|l| pts.saturating_sub(drained) >= l) {
                    over = Some(why);
                }
            } else if source.closed && lead_empty && !lead_moved {
                feed.drained = Some((pts, Why::Ended));
                if linger.is_none() || feed.shown[lead].is_none() {
                    over = Some(Why::Ended);
                }
            } else if !lead_moved && timeout.is_some_and(|t| pts.saturating_sub(feed.advanced) >= t)
            {
                feed.drained = Some((pts, Why::Timeout));
                if linger.is_none() || feed.shown[lead].is_none() {
                    over = Some(Why::Timeout);
                }
            }
        }
        if let Some(why) = over {
            self.feed = Some(feed);
            self.end_feed(pts, clock_base, why);
            return vec![Handed::default(); self.members.len()];
        }
        feed.ends = self.foretell(&feed, clock_base, grid);
        for hand in &mut handed {
            if let Some(record) = &mut hand.feed {
                record.ends = feed.ends;
            }
        }
        self.feed = Some(feed);
        handed
    }

    /// The newest time the group has heard of, on the clock: how far its
    /// source has arrived.
    pub fn progress(&self, clock_base: TimeBase) -> Option<i64> {
        let source = self.sources.back()?;
        let lead = &source.members[source.lead];
        let last = lead.last?;
        let offset = self.feed.as_ref().map_or(Offset::ZERO, |f| f.offset);
        Some(offset.to_clock(last, lead.base, clock_base))
    }
}

fn millis(seconds: f64) -> f64 {
    (seconds * 1000.0).round() / 1000.0
}

/// Sample `index` of a run at `rate` as a time in `base`, rounded to nearest.
fn pts_of_sample(index: i128, rate: u32, base: TimeBase) -> i64 {
    let num = index * base.den as i128;
    let den = rate as i128 * base.num as i128;
    ((num + den / 2).div_euclid(den.max(1))).clamp(i64::MIN as i128, i64::MAX as i128) as i64
}

/// The sample of a run at `rate` that time `pts` in `base` falls on, rounded
/// to nearest.
fn sample_of(pts: i64, rate: u32, base: TimeBase) -> i128 {
    let num = pts as i128 * base.num as i128 * rate as i128;
    let den = base.den as i128;
    (num + den / 2).div_euclid(den.max(1))
}

/// The samples of `queue` from time `s0` to `s1` (None: everything from
/// `s0`), as one run: samples before `s0` are dropped, a hole inside the
/// window is silence, and the run ends where the source has sent no further
/// or at `s1`. None where the source has nothing in the window.
fn recut_window(
    queue: &mut MemberQueue,
    audio: AudioFormat,
    s0: i64,
    s1: Option<i64>,
) -> Option<TickFrame> {
    let width = audio.sample_len().max(1);
    let rate = audio.sample_rate;
    let base = queue.base;
    let w0 = sample_of(s0, rate, base);
    let w1 = s1.map(|s1| sample_of(s1, rate, base));
    while let Some(front) = queue.queue.front() {
        let start = sample_of(front.pts, rate, base);
        let count = (front.data.len() / width) as i128;
        if start + count <= w0 {
            queue.pop();
            continue;
        }
        if start < w0 {
            let skip = (w0 - start) as usize;
            let front = queue.pop().expect("front is some");
            let left = TickFrame {
                pts: pts_of_sample(w0, rate, base),
                duration: Some(count as i64 - skip as i64),
                data: Arc::new(front.data[skip * width..].to_vec()),
                rows: Vec::new(),
            };
            queue.bytes += left.data.len();
            queue.queue.push_front(left);
        }
        break;
    }
    let mut data: Vec<u8> = Vec::new();
    let mut first: Option<i128> = None;
    let mut cursor: i128 = 0;
    while let Some(front) = queue.queue.front() {
        let start = sample_of(front.pts, rate, base);
        if w1.is_some_and(|w1| start >= w1) {
            break;
        }
        let count = (front.data.len() / width) as i128;
        if first.is_none() {
            first = Some(start);
            cursor = start;
        }
        if start > cursor {
            let hole = match w1 {
                Some(w1) => (start - cursor).min(w1 - cursor),
                None => start - cursor,
            };
            data.resize(data.len() + hole as usize * width, 0);
            cursor += hole;
            if w1.is_some_and(|w1| cursor >= w1) {
                break;
            }
        }
        let take = match w1 {
            Some(w1) => count.min(w1 - start),
            None => count,
        };
        data.extend_from_slice(&front.data[..take as usize * width]);
        cursor = start + take;
        if take == count {
            queue.pop();
        } else {
            let front = queue.pop().expect("front is some");
            let left = TickFrame {
                pts: pts_of_sample(start + take, rate, base),
                duration: Some((count - take) as i64),
                data: Arc::new(front.data[take as usize * width..].to_vec()),
                rows: Vec::new(),
            };
            queue.bytes += left.data.len();
            queue.queue.push_front(left);
            break;
        }
    }
    let first = first?;
    if data.is_empty() {
        return None;
    }
    Some(TickFrame {
        pts: pts_of_sample(first, rate, base),
        duration: Some((data.len() / width) as i64),
        data: Arc::new(data),
        rows: Vec::new(),
    })
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;

    const THIRTIETHS: TimeBase = TimeBase { num: 1, den: 30 };
    const KHZ48: TimeBase = TimeBase { num: 1, den: 48000 };
    const MILLIS: TimeBase = TimeBase { num: 1, den: 1000 };

    fn hold(anchor: Anchor, lead: f64, linger: Option<f64>, timeout: Option<f64>) -> Hold {
        Hold {
            anchor,
            lead,
            linger,
            timeout,
            group: None,
            port_param: None,
        }
    }

    fn video_member(id: u32, base: TimeBase) -> Member {
        Member {
            name: format!("in{id}"),
            kind: PortKind::Video,
            base,
            info: StreamInfo::default(),
            audio: None,
        }
    }

    fn audio_member(id: u32) -> Member {
        Member {
            name: format!("in{id}"),
            kind: PortKind::Audio,
            base: KHZ48,
            info: StreamInfo::default(),
            audio: Some(AudioFormat {
                sample_rate: 48000,
                channels: 1,
                sample_fmt: "f32",
                channel_layout: None,
            }),
        }
    }

    fn frame(pts: i64, mark: u8) -> TickFrame {
        TickFrame {
            pts,
            duration: None,
            data: Arc::new(vec![mark]),
            rows: Vec::new(),
        }
    }

    /// `count` samples from `pts`, each the sample's own index as f32.
    fn samples(pts: i64, count: usize) -> TickFrame {
        let data: Vec<u8> = (0..count)
            .flat_map(|k| ((pts as usize + k) as f32).to_le_bytes())
            .collect();
        TickFrame {
            pts,
            duration: Some(count as i64),
            data: Arc::new(data),
            rows: Vec::new(),
        }
    }

    fn quiet() -> (Report, Arc<Mutex<Vec<String>>>) {
        let lines = Arc::new(Mutex::new(Vec::new()));
        let written = Arc::clone(&lines);
        (
            Box::new(move |line| written.lock().unwrap().push(line)),
            lines,
        )
    }

    fn group(hold: Hold, members: Vec<Member>, port_fed: bool) -> (Group, Arc<Mutex<Vec<String>>>) {
        let (report, lines) = quiet();
        (
            Group::new(hold, "n", members, port_fed).with_report(report),
            lines,
        )
    }

    fn mark(handed: &Handed) -> Option<u8> {
        handed.frames.first().map(|f| f.data[0])
    }

    #[test]
    fn a_first_frame_feed_is_primed_by_lead_and_scheduled_lead_ahead() {
        let (mut g, lines) = group(
            hold(Anchor::FirstFrame, 0.3, None, Some(1.0)),
            vec![video_member(1, MILLIS)],
            false,
        );
        let grid = Grid::exact();
        g.arrive(0, frame(0, 10));
        g.arrive(0, frame(40, 11));
        assert!(!g.ready(100, Some(101), THIRTIETHS, Some(101)), "priming");
        g.arrive(0, frame(300, 12));
        assert!(g.ready(100, Some(101), THIRTIETHS, Some(101)));
        let handed = g.tick(100, Some(101), THIRTIETHS, &grid);
        let feed = handed[0].feed.clone().expect("scheduled");
        assert_eq!((feed.start.first_pts, feed.start.at), (0, 109));
        assert!(handed[0].frames.is_empty(), "nothing shows before `at`");
        assert!(lines.lock().unwrap()[0].contains("comes up at 3.633s, mapped from 0.000s"));
        let handed = g.tick(108, Some(109), THIRTIETHS, &grid);
        assert!(handed[0].frames.is_empty());
        let handed = g.tick(109, Some(110), THIRTIETHS, &grid);
        assert_eq!(mark(&handed[0]), Some(10));
        let handed = g.tick(110, Some(111), THIRTIETHS, &grid);
        assert_eq!(
            mark(&handed[0]),
            Some(10),
            "33 ms in: the frame at 40 ms is ahead"
        );
        let handed = g.tick(111, Some(112), THIRTIETHS, &grid);
        assert_eq!(mark(&handed[0]), Some(11));
    }

    #[test]
    fn a_shared_clock_feed_waits_for_the_clock_and_skips_what_the_clock_passed() {
        let (mut g, _) = group(
            hold(Anchor::SharedClock, 0.0, None, None),
            vec![video_member(1, MILLIS)],
            false,
        );
        let grid = Grid::exact();
        for k in 0..10 {
            g.arrive(0, frame(5000 + k * 40, k as u8));
        }
        let handed = g.tick(100, Some(101), THIRTIETHS, &grid);
        let feed = handed[0].feed.clone().expect("held");
        assert_eq!(feed.start.at, 150);
        assert!(handed[0].frames.is_empty());
        let handed = g.tick(150, Some(151), THIRTIETHS, &grid);
        assert_eq!(mark(&handed[0]), Some(0));
        let handed = g.tick(156, Some(157), THIRTIETHS, &grid);
        assert_eq!(
            mark(&handed[0]),
            Some(5),
            "0.2 s in: frames 1 to 4 are skipped"
        );
        g.source_close(0);
        let handed = g.tick(157, Some(158), THIRTIETHS, &grid);
        assert_eq!(mark(&handed[0]), Some(5));
        assert_eq!(handed[0].feed.as_ref().unwrap().ends, Some(161));
    }

    #[test]
    fn the_last_frame_shows_once_then_the_feed_ends_and_a_linger_keeps_it() {
        for (linger, last_tick) in [(None, 2), (Some(0.1), 5)] {
            let (mut g, lines) = group(
                hold(Anchor::SharedClock, 0.0, linger, None),
                vec![video_member(1, THIRTIETHS)],
                false,
            );
            let grid = Grid::exact();
            g.arrive(0, frame(0, 1));
            g.arrive(0, frame(2, 2));
            g.source_close(0);
            let shown: Vec<Option<u8>> = (0..8)
                .map(|k| mark(&g.tick(k, Some(k + 1), THIRTIETHS, &grid)[0]))
                .collect();
            let wanted: Vec<Option<u8>> = (0..8)
                .map(|k| match k {
                    0 | 1 => Some(1),
                    k if k <= last_tick => Some(2),
                    _ => None,
                })
                .collect();
            assert_eq!(shown, wanted, "linger {linger:?}");
            assert!(lines
                .lock()
                .unwrap()
                .iter()
                .any(|l| l.contains("the source ended")));
        }
    }

    #[test]
    fn a_source_that_stops_sending_times_out_and_a_new_one_starts_the_next_feed() {
        let (mut g, lines) = group(
            hold(Anchor::SharedClock, 0.0, None, Some(0.1)),
            vec![video_member(1, THIRTIETHS)],
            true,
        );
        let grid = Grid::exact();
        g.arrive(0, frame(0, 1));
        let shown: Vec<Option<u8>> = (0..5)
            .map(|k| mark(&g.tick(k, Some(k + 1), THIRTIETHS, &grid)[0]))
            .collect();
        assert_eq!(shown, vec![Some(1), Some(1), Some(1), None, None]);
        assert!(lines
            .lock()
            .unwrap()
            .iter()
            .any(|l| l.contains("stopped sending")));
        g.arrive(0, frame(7, 2));
        let handed = g.tick(7, Some(8), THIRTIETHS, &grid);
        assert_eq!(mark(&handed[0]), Some(2));
        assert_eq!(handed[0].feed.as_ref().unwrap().start.first_pts, 7);
    }

    #[test]
    fn a_jump_in_the_source_is_a_new_feed_on_the_next_tick() {
        let (mut g, _) = group(
            hold(Anchor::SharedClock, 0.0, None, None),
            vec![video_member(1, THIRTIETHS)],
            false,
        );
        let grid = Grid::exact();
        g.arrive(0, frame(0, 1));
        g.arrive(0, frame(1, 2));
        g.arrive(0, frame(100, 3));
        g.arrive(0, frame(101, 4));
        assert!(g.ready(0, Some(1), THIRTIETHS, None));
        assert_eq!(mark(&g.tick(0, Some(1), THIRTIETHS, &grid)[0]), Some(1));
        let handed = g.tick(1, Some(2), THIRTIETHS, &grid);
        assert_eq!(mark(&handed[0]), Some(2));
        assert_eq!(handed[0].feed.as_ref().unwrap().ends, Some(1));
        assert_eq!(
            mark(&g.tick(2, Some(3), THIRTIETHS, &grid)[0]),
            None,
            "ended"
        );
        let handed = g.tick(3, Some(4), THIRTIETHS, &grid);
        assert_eq!(handed[0].feed.as_ref().unwrap().start.first_pts, 100);
        assert!(
            handed[0].frames.is_empty(),
            "held until the clock reaches 100"
        );
    }

    #[test]
    fn a_sound_that_starts_off_the_grid_is_told_the_pictures_offset() {
        let (mut g, _) = group(
            hold(Anchor::FirstFrame, 0.0, None, None),
            vec![video_member(1, MILLIS), audio_member(2)],
            false,
        );
        g.arrive(0, frame(1000, 1));
        g.arrive(1, samples(48480, 1000));
        g.arrive(0, frame(1040, 2));
        let handed = g.tick(300, Some(301), THIRTIETHS, &Grid::exact());
        let sound = handed[1].feed.clone().unwrap();
        assert_eq!((sound.start.first_pts, sound.start.at), (48000, 300));
    }

    #[test]
    fn a_member_the_connection_does_not_bring_gets_no_feed() {
        let (mut g, lines) = group(
            hold(Anchor::FirstFrame, 0.0, None, None),
            vec![video_member(1, MILLIS), audio_member(2)],
            true,
        );
        g.source_open(
            0,
            SourceInfo {
                connection: 1,
                tags: Vec::new(),
                base: MILLIS,
                info: StreamInfo::default(),
            },
        );
        g.arrive(0, frame(1000, 1));
        g.arrive(0, frame(1040, 2));
        let handed = g.tick(300, Some(301), THIRTIETHS, &Grid::exact());
        assert!(handed[0].feed.is_some());
        assert!(handed[1].feed.is_none() && handed[1].frames.is_empty());
        let said = lines.lock().unwrap().join("\n");
        assert!(said.contains(r#""absent":["in2"]"#), "{said}");
    }

    #[test]
    fn a_connection_that_brings_only_sound_is_led_by_its_sound() {
        let (mut g, lines) = group(
            hold(Anchor::FirstFrame, 0.0, None, None),
            vec![video_member(1, MILLIS), audio_member(2)],
            true,
        );
        g.source_open(
            1,
            SourceInfo {
                connection: 1,
                tags: Vec::new(),
                base: KHZ48,
                info: StreamInfo::default(),
            },
        );
        g.arrive(1, samples(48000, 1600));
        g.arrive(1, samples(49600, 1600));
        let handed = g.tick(300, Some(301), THIRTIETHS, &Grid::exact());
        assert!(handed[0].feed.is_none() && handed[0].frames.is_empty());
        let sound = handed[1].feed.clone().expect("the sound has a feed");
        assert_eq!((sound.start.first_pts, sound.start.at), (48000, 300));
        assert_eq!(handed[1].frames[0].pts, 48000);
        let said = lines.lock().unwrap().join("\n");
        assert!(said.contains(r#""input":"in2""#), "{said}");
        assert!(said.contains(r#""absent":["in1"]"#), "{said}");
        g.source_close(1);
        let handed = g.tick(301, Some(302), THIRTIETHS, &Grid::exact());
        assert_eq!(handed[1].frames[0].pts, 49600);
    }

    #[test]
    fn a_group_shares_one_offset_and_its_sound_is_cut_to_the_tick() {
        let (mut g, _) = group(
            hold(Anchor::FirstFrame, 0.0, None, None),
            vec![video_member(1, MILLIS), audio_member(2)],
            false,
        );
        let grid = Grid::exact();
        g.arrive(0, frame(1000, 1));
        g.arrive(1, samples(48000, 1000));
        g.arrive(1, samples(49000, 1000));
        g.arrive(1, samples(50000, 1000));
        g.arrive(0, frame(1040, 2));
        let handed = g.tick(300, Some(301), THIRTIETHS, &grid);
        let picture = handed[0].feed.clone().unwrap();
        let sound = handed[1].feed.clone().unwrap();
        assert_eq!((picture.start.first_pts, picture.start.at), (1000, 300));
        assert_eq!((sound.start.first_pts, sound.start.at), (48000, 300));
        assert_eq!(mark(&handed[0]), Some(1));
        let run = &handed[1].frames[0];
        assert_eq!((run.pts, run.duration), (48000, Some(1600)));
        let value = |k: usize| f32::from_le_bytes(run.data[k * 4..k * 4 + 4].try_into().unwrap());
        assert_eq!(value(0), 48000.0);
        assert_eq!(value(1599), 49599.0);
        let handed = g.tick(301, Some(302), THIRTIETHS, &grid);
        let run = &handed[1].frames[0];
        assert_eq!(
            (run.pts, run.duration),
            (49600, Some(1400)),
            "behind: a short run"
        );
        let handed = g.tick(302, Some(303), THIRTIETHS, &grid);
        assert!(handed[1].frames.is_empty(), "nothing in the window: none");
        g.arrive(1, samples(53000, 1000));
        let handed = g.tick(303, Some(304), THIRTIETHS, &grid);
        let run = &handed[1].frames[0];
        assert_eq!(
            (run.pts, run.duration),
            (53000, Some(1000)),
            "a late run starts where it is"
        );
    }

    #[test]
    fn a_hole_inside_the_sound_is_silence_and_samples_before_the_window_go() {
        let (mut g, _) = group(
            hold(Anchor::SharedClock, 0.0, None, None),
            vec![audio_member(2)],
            false,
        );
        let grid = Grid::exact();
        g.arrive(0, samples(0, 800));
        g.arrive(0, samples(1000, 800));
        let handed = g.tick(0, Some(1), THIRTIETHS, &grid);
        let run = &handed[0].frames[0];
        assert_eq!((run.pts, run.duration), (0, Some(1600)));
        let value = |k: usize| f32::from_le_bytes(run.data[k * 4..k * 4 + 4].try_into().unwrap());
        assert_eq!(value(799), 799.0);
        assert_eq!(value(800), 0.0, "the hole is silence");
        assert_eq!(value(1000), 1000.0);
        g.arrive(0, samples(1800, 10000));
        let handed = g.tick(3, Some(4), THIRTIETHS, &grid);
        let run = &handed[0].frames[0];
        assert_eq!((run.pts, run.duration), (4800, Some(1600)));
    }

    #[test]
    fn a_tagged_source_is_timed_by_its_tag() {
        let (mut g, _) = group(
            hold(Anchor::Tagged("smart_timed".into()), 0.0, None, None),
            vec![video_member(1, THIRTIETHS)],
            true,
        );
        let grid = Grid::exact();
        g.source_open(
            0,
            SourceInfo {
                connection: 1,
                tags: vec![("smart_timed".into(), "1".into())],
                base: THIRTIETHS,
                info: StreamInfo::default(),
            },
        );
        g.arrive(0, frame(50, 1));
        let handed = g.tick(10, Some(11), THIRTIETHS, &grid);
        assert_eq!(handed[0].feed.as_ref().unwrap().start.at, 50);
        assert_eq!(handed[0].info.as_ref().unwrap().0.tags[0].0, "smart_timed");
        assert_eq!(g.room(), 0, "held: the source waits on its socket");
        g.source_close(0);
        g.source_open(
            0,
            SourceInfo {
                connection: 2,
                tags: Vec::new(),
                base: THIRTIETHS,
                info: StreamInfo::default(),
            },
        );
        g.arrive(0, frame(0, 2));
        let handed = g.tick(50, Some(51), THIRTIETHS, &grid);
        assert_eq!(mark(&handed[0]), Some(1));
        let handed = g.tick(51, Some(52), THIRTIETHS, &grid);
        assert!(handed[0].frames.is_empty(), "the first feed ended");
        let handed = g.tick(52, Some(53), THIRTIETHS, &grid);
        assert_eq!(
            handed[0].feed.as_ref().unwrap().start.at,
            52,
            "untimed: lead 0"
        );
        assert_eq!(mark(&handed[0]), Some(2));
    }

    #[test]
    fn a_learned_grid_foretells_the_end_on_the_clocks_frames() {
        let mut grid = Grid::learned();
        for k in 0..5 {
            grid.observe(1000 + k * 1001);
        }
        assert_eq!(grid.at_or_after(1000 + 3 * 1001 + 1), Some(1000 + 4 * 1001));
        assert_eq!(grid.at_or_after(1000 + 9 * 1001), Some(1000 + 9 * 1001));
        assert_eq!(grid.before(1000 + 9 * 1001), Some(1000 + 8 * 1001));
        assert_eq!(grid.after(1000 + 9 * 1001), Some(1000 + 10 * 1001));
        grid.observe(50_000);
        assert_eq!(
            grid.at_or_after(50_001),
            None,
            "a jump starts the grid again"
        );
        grid.observe(51_001);
        assert_eq!(grid.at_or_after(50_001), Some(51_001));
    }

    #[test]
    fn the_offset_maps_both_ways_exactly_at_the_anchors() {
        let offset = Offset::between(
            7_000_000,
            TimeBase {
                num: 1,
                den: 1_000_000,
            },
            303,
            THIRTIETHS,
        );
        assert_eq!(
            offset.to_clock(
                7_000_000,
                TimeBase {
                    num: 1,
                    den: 1_000_000
                },
                THIRTIETHS
            ),
            303
        );
        assert_eq!(offset.to_source(303, THIRTIETHS, KHZ48), 7 * 48000);
        assert_eq!(offset.to_source(304, THIRTIETHS, KHZ48), 7 * 48000 + 1600);
        assert_eq!(Offset::ZERO.to_clock(48000, KHZ48, THIRTIETHS), 30);
    }
}
