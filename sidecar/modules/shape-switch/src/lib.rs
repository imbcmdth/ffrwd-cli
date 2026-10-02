//! What the live switch did, as one node: the programme's picture and sound
//! pass through until a feed is on, and the feed's stand in for them while
//! it is. The feed is a hold group on the loopback port `port` names, its
//! picture conformed to the programme's and its sound to the programme's
//! sound; `smart_timed=1` in its tags makes it a timed feed. Every edge is
//! the host's: the lead, the linger, the timeout and the return are the hold
//! input's, and the node only reads what each tick hands it.
//!
//! Beside the switched picture and sound it writes two data outputs: `clock`,
//! one row a second of programme time pairing it with the wall, and `feeds`,
//! a row whenever what the host says of the feed changes: its start, the end
//! it foretells, and its end.

use ffrwd_node::{Anchor, Bound, Init, Input, Node, Out, Output, Rational, Result, Shape, Tick};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Params {
    lead: f64,
    linger: f64,
    timeout: f64,
    layer: bool,
}

#[derive(Serialize, PartialEq, Clone)]
struct FeedRow {
    pts: i64,
    event: &'static str,
    first_pts: i64,
    at: i64,
    ends: Option<i64>,
    timed: bool,
}

#[derive(Serialize)]
struct ClockRow {
    event: &'static str,
    pts: f64,
    wall: f64,
}

struct Switch {
    v: u32,
    a: Option<u32>,
    feed: Option<u32>,
    feed_audio: Option<u32>,
    layer: bool,
    frame_len: usize,
    sample_len: usize,
    rate: u32,
    /// The programme time of the last clock row.
    last_row: Option<i64>,
    /// What the last feeds row said, so a tick that changes nothing says
    /// nothing.
    said: Option<FeedRow>,
}

fn seconds(seconds: f64) -> Option<f64> {
    (seconds > 0.0).then_some(seconds)
}

impl Node for Switch {
    const NAME: &'static str = "shape_switch";
    const VERSION: &'static str = "0.1.0";
    const PARAMS_SCHEMA: &'static str = r#"{"type":"object","properties":{"port":{"type":"integer","minimum":1,"maximum":65535,"default":9000},"lead":{"type":"number","minimum":0,"default":0.3},"linger":{"type":"number","minimum":0,"default":0},"timeout":{"type":"number","minimum":0,"default":1},"layer":{"type":"boolean","default":false}},"additionalProperties":false}"#;
    type Params = Params;

    fn shape(params: &Params, _: &Bound) -> Result<Shape> {
        let feed = |input: Input| {
            let mut input = input
                .optional()
                .hold()
                .anchor(Anchor::Tagged("smart_timed".to_owned()))
                .lead(params.lead)
                .group("switch")
                .port_param("port");
            if let Some(linger) = seconds(params.linger) {
                input = input.linger(linger);
            }
            if let Some(timeout) = seconds(params.timeout) {
                input = input.timeout(timeout);
            }
            input
        };
        Ok(Shape::new()
            .input(Input::video("v").clock().pixel_formats(&["rgba"]))
            .input(Input::audio("a").optional().sample_formats(&["f32"]))
            .input(feed(Input::video("feed")).like("v").pixel_formats(&["rgba"]))
            .input(feed(Input::audio("feed_audio")).like("a").sample_formats(&["f32"]))
            .output(Output::like("v"))
            .output(Output::like("a"))
            .output(Output::rows("clock").schema_json(
                r#"{"type":"object","properties":{"event":{"type":"string"},"pts":{"type":"number"},"wall":{"type":"number"}}}"#,
            ))
            .output(Output::rows("feeds").schema_json(
                r#"{"type":"object","properties":{"pts":{"type":"integer"},"event":{"type":"string"},"first_pts":{"type":"integer"},"at":{"type":"integer"},"ends":{"type":["integer","null"]},"timed":{"type":"boolean"}}}"#,
            ))
            .one_to_one())
    }

    fn init(params: Params, init: &Init) -> Result<Switch> {
        let v = init.stream("v")?;
        let video = v.video_format().ok_or("`v` is a video input")?;
        let a = init.optional("a");
        let audio = a.and_then(|a| a.audio_format());
        Ok(Switch {
            v: v.id,
            a: a.map(|a| a.id),
            feed: init.optional("feed").map(|f| f.id),
            feed_audio: init.optional("feed_audio").map(|f| f.id),
            layer: params.layer,
            frame_len: (video.width * video.height * 4) as usize,
            sample_len: audio.map_or(4, |a| a.channels as usize * 4),
            rate: audio.map_or(48_000, |a| a.sample_rate),
            last_row: None,
            said: None,
        })
    }

    fn process(&mut self, tick: &Tick, out: &mut Out) -> Result<()> {
        let pts = tick.pts();
        let clock = tick.time_base();
        let Some(frame) = tick.frame(self.v) else {
            return Ok(());
        };
        let feed = self
            .feed
            .and_then(|id| tick.feed(id).map(|feed| (id, feed)));
        let live = feed.as_ref().is_some_and(|(_, feed)| pts >= feed.start.at);
        let shown = feed
            .as_ref()
            .and_then(|(id, _)| tick.frame(*id).map(|f| (*id, f)));
        match (self.layer, shown) {
            (_, Some((id, fed))) => out.same("v", frame.pts, frame.duration, id, fed.index)?,
            (false, None) => out.pass("v", self.v, &frame)?,
            (true, None) => out.frame("v", frame.pts, frame.duration, vec![0; self.frame_len])?,
        }

        if let Some(a) = self.a {
            if let Some(sound) = tick.frame(a) {
                let mut samples = tick.fetch(a, sound.index);
                if live {
                    samples.iter_mut().for_each(|b| *b = 0);
                    let fed = self
                        .feed_audio
                        .and_then(|id| tick.frame(id).map(|run| (id, run)));
                    if let (Some((id, run)), Some((_, feed))) = (fed, &feed) {
                        let run_feed = tick.feed(id).unwrap_or_else(|| feed.clone());
                        let base = tick.info(id).time_base;
                        let offset =
                            position(run.pts, base, &run_feed.start, pts, clock, self.rate);
                        let data = tick.fetch(id, run.index);
                        let width = self.sample_len;
                        let count = (samples.len() / width) as i64;
                        for (k, sample) in data.chunks_exact(width).enumerate() {
                            let at = offset + k as i64;
                            if at < 0 || at >= count {
                                continue;
                            }
                            let slot = at as usize * width;
                            samples[slot..slot + width].copy_from_slice(sample);
                        }
                    }
                }
                out.frame("a", sound.pts, sound.duration, samples)?;
            }
        }

        let row = feed.as_ref().map(|(_, feed)| FeedRow {
            pts,
            event: match feed.ends {
                Some(_) => "ending",
                None if live => "live",
                None => "start",
            },
            first_pts: feed.start.first_pts,
            at: feed.start.at,
            ends: feed.ends,
            timed: feed
                .start
                .tags
                .iter()
                .any(|(k, v)| k == "smart_timed" && v == "1"),
        });
        let changed = match (&self.said, &row) {
            (None, None) => None,
            (Some(_), None) => Some(FeedRow {
                pts,
                event: "end",
                first_pts: 0,
                at: 0,
                ends: None,
                timed: false,
            }),
            (Some(said), Some(row)) => {
                (said.event != row.event || said.ends != row.ends || said.at != row.at)
                    .then(|| row.clone())
            }
            (None, Some(row)) => Some(row.clone()),
        };
        if let Some(row) = changed {
            out.row("feeds", pts, &row)?;
        }
        self.said = row;

        let due = self
            .last_row
            .is_none_or(|last| pts < last || clock.seconds(pts - last) >= 1.0);
        if due {
            self.last_row = Some(pts);
            let wall = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0.0, |d| d.as_secs_f64());
            out.row(
                "clock",
                pts,
                &ClockRow {
                    event: "clock",
                    pts: clock.seconds(pts),
                    wall,
                },
            )?;
        }
        Ok(())
    }
}

/// Where a run of the feed's samples starting at `run_pts` (in `base`)
/// lands in the tick's samples, as a sample index from `tick_pts`: through
/// the feed's start, which maps its first sample's pts onto the clock.
fn position(
    run_pts: i64,
    base: Rational,
    start: &ffrwd_node::FeedStart,
    tick_pts: i64,
    clock: Rational,
    rate: u32,
) -> i64 {
    let into_feed = base.seconds(run_pts - start.first_pts);
    let into_tick = clock.seconds(start.at - tick_pts) + into_feed;
    (into_tick * rate as f64).round() as i64
}

ffrwd_node::export!(Switch);
