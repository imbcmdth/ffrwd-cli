//! A program with someone speaking now and then: a picture saying the time,
//! and the cue rows a streaming transcriber writes beside it. A source, at
//! `fps`, quiet between utterances. Every `every` seconds an utterance
//! starts, a quarter of a second in, and lasts 1.2 s; it is sent growing
//! (`final` false) half a second in, and whole (`final` true) `late` seconds
//! after it ends, Ann and Bob speaking in turn. It ends after `seconds`, or
//! never when that is 0. Its clock reads `start` at its first picture, as a
//! program's does for a reader that joins it that far in.

use ffrwd_node::{Bound, Init, Node, Out, Output, Rational, Result, Shape, Tick};
use serde::{Deserialize, Serialize};
use stand_ins::Picture;

const WIDTH: u32 = 320;
const HEIGHT: u32 = 180;
const OPENS: f64 = 0.25;
const LASTS: f64 = 1.2;

#[derive(Deserialize)]
struct Params {
    fps: f64,
    every: f64,
    late: f64,
    seconds: f64,
    start: f64,
}

#[derive(Default, Serialize)]
struct Said {
    start_t: f64,
    end_t: f64,
    text: String,
    #[serde(rename = "final")]
    finished: bool,
    voice: String,
}

struct Speaker {
    every: f64,
    late: f64,
    seconds: f64,
    start: f64,
    /// The next utterance to send whole, and the next to send growing.
    said: u32,
    grown: u32,
}

impl Speaker {
    fn utterance(&self, n: u32) -> (f64, f64, &'static str) {
        let start = self.start + f64::from(n) * self.every + OPENS;
        let voice = if n.is_multiple_of(2) { "Ann" } else { "Bob" };
        (start, start + LASTS, voice)
    }
}

impl Node for Speaker {
    const NAME: &'static str = "speaker";
    const VERSION: &'static str = "0.1.0";
    const PARAMS_SCHEMA: &'static str = r#"{"type":"object","properties":{"fps":{"type":"number","exclusiveMinimum":0,"default":15},"every":{"type":"number","exclusiveMinimum":0,"default":12},"late":{"type":"number","minimum":0,"default":0},"seconds":{"type":"number","minimum":0,"default":0},"start":{"type":"number","minimum":0,"default":0}},"additionalProperties":false}"#;
    type Params = Params;

    fn shape(params: &Params, _: &Bound) -> Result<Shape> {
        Ok(Shape::new()
            .rate(Rational::approximate(params.fps, 1001))
            .output(
                Output::video("video")
                    .size(WIDTH, HEIGHT)
                    .pixel_format("rgba")
                    .row(0),
            )
            .output(Output::rows("captions").schema::<Said>().row(0))
            .relation_row(&format!(r#"{{"width":{WIDTH},"height":{HEIGHT}}}"#))
            .bounded(params.seconds > 0.0))
    }

    fn init(params: Params, _: &Init) -> Result<Speaker> {
        Ok(Speaker {
            every: params.every,
            late: params.late,
            seconds: params.seconds,
            start: params.start,
            said: 0,
            grown: 0,
        })
    }

    fn process(&mut self, tick: &Tick, out: &mut Out) -> Result<()> {
        if self.seconds > 0.0 && tick.seconds() >= self.seconds {
            out.finish();
            return Ok(());
        }
        let base = tick.time_base();
        let pts =
            tick.pts() + (self.start * f64::from(base.den) / f64::from(base.num)).round() as i64;
        let now = base.seconds(pts);
        let mut canvas = Picture::filled(WIDTH as usize, HEIGHT as usize, [16, 24, 48, 255]);
        canvas.text(10, 80, 2, &format!("{now:.2}"), [255, 255, 255, 255]);
        out.frame("video", pts, Some(1), canvas.data)?;
        let (start, _, voice) = self.utterance(self.grown);
        if now >= start + 0.5 {
            let row = Said {
                start_t: start,
                end_t: start + 0.5,
                text: format!("cue {} so", self.grown),
                finished: false,
                voice: voice.into(),
            };
            out.row("captions", pts, &row)?;
            self.grown += 1;
        }
        let (start, end, voice) = self.utterance(self.said);
        if now >= end + self.late {
            let row = Said {
                start_t: start,
                end_t: end,
                text: format!("cue {} said at {start:.2}", self.said),
                finished: true,
                voice: voice.into(),
            };
            out.row("captions", pts, &row)?;
            self.said += 1;
        }
        Ok(())
    }
}

ffrwd_node::export!(Speaker);

#[cfg(test)]
mod tests {
    use super::*;
    use ffrwd_node::mock::Harness;

    #[derive(Deserialize)]
    struct Heard {
        start_t: f64,
        #[serde(rename = "final")]
        finished: bool,
        voice: String,
    }

    #[test]
    fn a_cue_grows_and_then_comes_whole_late() {
        let mut speaker =
            Harness::<Speaker>::new(r#"{"fps":10,"every":12,"late":3}"#, vec![]).unwrap();
        let mut rows = Vec::new();
        for pts in 0..50 {
            let tick = speaker.tick(pts);
            rows.extend(speaker.process(&tick).unwrap().messages("captions"));
        }
        let read: Vec<(i64, Heard)> = rows
            .iter()
            .map(|(pts, json)| (*pts, ffrwd_node::parse(json).unwrap()))
            .collect();
        assert_eq!(read.len(), 2);
        assert!(!read[0].1.finished);
        // Whole 3 s after it ended at 1.45 s, still saying when it started.
        assert_eq!(read[1].0, 45);
        assert!(read[1].1.finished);
        assert_eq!(read[1].1.start_t, 0.25);
        assert_eq!(read[1].1.voice, "Ann");
    }
}
