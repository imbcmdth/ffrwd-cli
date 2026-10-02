//! Burns what it hears and reads onto the picture: a level meter for the
//! sound beside each frame, and the text of every cue showing at the frame's
//! time. The picture is the clock; the sound comes frame for frame and the
//! words by their time, and either may be left out. With neither, the
//! picture leaves untouched. Recipes 147, 148, 149 and 153.

use ffrwd_node::{
    Bound, Cue, Cues, Init, Input, NoParams, Node, Out, Output, Result, Shape, StateRow, Tick,
};
use stand_ins::{text_size, Picture, Rect};

const METER: [u8; 4] = [64, 220, 64, 255];
const SHADE: [u8; 4] = [0, 0, 0, 160];
const TEXT: [u8; 4] = [255, 255, 255, 255];

struct Burn {
    v: u32,
    a: Option<(u32, usize)>,
    width: usize,
    height: usize,
    cues: Cues,
}

/// The level of `samples` from 0 at -60 dB of full scale or quieter to 1 at
/// full scale.
fn level(samples: &[f32]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let power = samples
        .iter()
        .map(|s| (*s as f64) * (*s as f64))
        .sum::<f64>()
        / samples.len() as f64;
    let db = 10.0 * power.max(1e-12).log10();
    ((db + 60.0) / 60.0).clamp(0.0, 1.0)
}

impl Burn {
    /// Where the meter's track lies: along the bottom left, a third of the
    /// picture wide.
    fn track(&self) -> Rect {
        let margin = (self.width / 40).max(2);
        let tall = (self.height / 40).max(4);
        let y1 = self.height.saturating_sub(margin);
        Rect {
            x0: margin,
            y0: y1.saturating_sub(tall),
            x1: margin + self.width / 3,
            y1,
        }
    }

    fn meter(&self, picture: &mut Picture, level: f64) {
        let track = self.track();
        picture.fill(track, SHADE);
        let lit = track.x0 + (track.width() as f64 * level).round() as usize;
        picture.fill(Rect { x1: lit, ..track }, METER);
    }

    /// `lines` centred above the meter, each on a shaded band.
    fn caption(&self, picture: &mut Picture, lines: &[&str]) {
        let scale = (self.height / 120).max(1);
        let gap = scale * 2;
        let (_, line) = text_size("", scale);
        let bottom = self.track().y0.saturating_sub(gap * 2);
        let mut y = bottom.saturating_sub(lines.len() * (line + gap));
        for text in lines {
            let (w, h) = text_size(text, scale);
            let x = self.width.saturating_sub(w) / 2;
            let band = Rect {
                x0: x.saturating_sub(gap),
                y0: y.saturating_sub(gap / 2),
                x1: x + w + gap,
                y1: y + h + gap / 2,
            };
            picture.fill(band, SHADE);
            picture.text(x as i64, y as i64, scale, text, TEXT);
            y += line + gap;
        }
    }
}

impl Node for Burn {
    const NAME: &'static str = "burn";
    const VERSION: &'static str = "0.1.0";
    type Params = NoParams;

    fn shape(_: &NoParams, _: &Bound) -> Result<Shape> {
        Ok(Shape::new()
            .input(Input::video("v").clock().pixel_formats(&["rgba"]))
            .input(Input::audio("a").optional().sample_formats(&["f32"]))
            .input(
                Input::rows("words")
                    .optional()
                    .interval()
                    .state()
                    .schema::<Cue>(),
            )
            .output(Output::like("v"))
            .pure()
            .one_to_one())
    }

    fn init(_: NoParams, init: &Init) -> Result<Burn> {
        let v = init.stream("v")?;
        let video = v.video_format().ok_or("`v` is a video input")?;
        let a = match init.optional("a") {
            Some(a) => {
                let audio = a.audio_format().ok_or("`a` is an audio input")?;
                Some((a.id, audio.channels.max(1) as usize))
            }
            None => None,
        };
        Ok(Burn {
            v: v.id,
            a,
            width: video.width as usize,
            height: video.height as usize,
            cues: Cues::new(),
        })
    }

    fn fold(&mut self, row: StateRow) -> Result<()> {
        self.cues.add(row.row::<Cue>()?);
        Ok(())
    }

    fn process(&mut self, tick: &Tick, out: &mut Out) -> Result<()> {
        let Some(frame) = tick.frame(self.v) else {
            return Ok(());
        };
        let t = tick.time_base().seconds(frame.pts);
        self.cues.drop_ended(t);
        let lines: Vec<&str> = self.cues.at(t).map(|cue| cue.text.as_str()).collect();
        let sound = self.a.map(|(a, _)| match tick.frame(a) {
            Some(heard) => {
                let bytes = tick.fetch(a, heard.index);
                let (chunks, _) = bytes.as_chunks::<4>();
                let samples: Vec<f32> = chunks
                    .iter()
                    .map(|chunk| f32::from_le_bytes(*chunk))
                    .collect();
                level(&samples)
            }
            None => 0.0,
        });
        if sound.is_none() && lines.is_empty() {
            return Ok(out.pass("v", self.v, &frame)?);
        }
        let mut picture = Picture::new(tick.fetch(self.v, frame.index), self.width, self.height)?;
        if let Some(level) = sound {
            self.meter(&mut picture, level);
        }
        self.caption(&mut picture, &lines);
        Ok(out.frame("v", frame.pts, frame.duration, picture.data)?)
    }
}

ffrwd_node::export!(Burn);

#[cfg(test)]
mod tests {
    use super::*;
    use ffrwd_node::mock::Harness;
    use ffrwd_node::{BoundStream, Payload, Rational};

    const TB: Rational = Rational::new(1, 10);

    #[test]
    fn the_picture_alone_passes() {
        let v = BoundStream::video("v", 0, 32, 24, "rgba", TB);
        let mut burn = Harness::<Burn>::new("", vec![v]).unwrap();
        assert_eq!(burn.shape().inputs.len(), 3);
        let emitted = burn
            .process(&burn.tick(0).frame(0, 0, vec![0; 32 * 24 * 4]))
            .unwrap();
        assert!(matches!(emitted.on("v")[..], [Payload::Same { .. }]));
    }

    #[test]
    fn a_cue_shows_until_it_ends_on_any_worker() {
        let bound = vec![
            BoundStream::video("v", 0, 120, 80, "rgba", TB),
            BoundStream::rows("words", 1, TB),
        ];
        let mut burn = Harness::<Burn>::new("", bound).unwrap();
        let cue = r#"{"start_t":0.0,"end_t":2.0,"text":"-20 dB"}"#;
        let first = burn
            .tick(15)
            .frame(0, 15, vec![0; 120 * 80 * 4])
            .earlier(1, 0, &[cue]);
        let shown = burn.process(&first).unwrap();
        assert!(matches!(shown.on("v")[..], [Payload::Frame { .. }]));
        assert_eq!(burn.node().cues.len(), 1);
        let after = burn.tick(20).frame(0, 20, vec![0; 120 * 80 * 4]);
        let gone = burn.process(&after).unwrap();
        assert!(matches!(gone.on("v")[..], [Payload::Same { .. }]));
        assert!(burn.node().cues.is_empty());
    }

    #[test]
    fn the_meter_follows_the_sound() {
        assert_eq!(level(&[1.0, -1.0]), 1.0);
        assert_eq!(level(&[0.0; 4]), 0.0);
        assert!((level(&[0.1; 4]) - 2.0 / 3.0).abs() < 1e-6);
    }
}
