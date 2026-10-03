//! Listens two seconds of sound at a time and writes one cue a window saying
//! how loud it was: `-23 dB`. A tumbling window whose cues leave with it, so
//! its output's latency is 0. Recipes 147, 148 and 153.

use ffrwd_node::{Bound, Cue, Init, Input, NoParams, Node, Out, Output, Result, Shape, Tick};

const RATE: u32 = 48_000;
const WINDOW: u32 = 2 * RATE;

struct Hear {
    a: u32,
    channels: usize,
}

/// How loud `samples` are, as a cue's text: their RMS in dB of full scale.
fn loudness(samples: &[f32]) -> String {
    if samples.is_empty() {
        return "silence".to_owned();
    }
    let power = samples
        .iter()
        .map(|s| (*s as f64) * (*s as f64))
        .sum::<f64>()
        / samples.len() as f64;
    let db = 10.0 * power.log10();
    if db < -90.0 {
        "silence".to_owned()
    } else {
        format!("{db:.0} dB")
    }
}

fn samples(bytes: &[u8]) -> Vec<f32> {
    let (chunks, _) = bytes.as_chunks::<4>();
    chunks
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect()
}

impl Node for Hear {
    const NAME: &'static str = "hear";
    const VERSION: &'static str = "0.1.0";
    type Params = NoParams;

    fn shape(_: &NoParams, _: &Bound) -> Result<Shape> {
        Ok(Shape::new()
            .input(
                Input::audio("a")
                    .clock()
                    .window(WINDOW, WINDOW)
                    .sample_formats(&["f32"])
                    .sample_rates(&[RATE]),
            )
            .output(Output::rows("cues").schema::<Cue>())
            .pure())
    }

    fn init(_: NoParams, init: &Init) -> Result<Hear> {
        let a = init.stream("a")?;
        let audio = a.audio_format().ok_or("`a` is an audio input")?;
        Ok(Hear {
            a: a.id,
            channels: audio.channels.max(1) as usize,
        })
    }

    fn process(&mut self, tick: &Tick, out: &mut Out) -> Result<()> {
        let Some(frame) = tick.frame(self.a) else {
            return Ok(());
        };
        let samples = samples(&tick.fetch(self.a, frame.index));
        let start = tick.time_base().seconds(frame.pts);
        let length = (samples.len() / self.channels) as f64 / RATE as f64;
        let cue = Cue::new(start, start + length, loudness(&samples));
        Ok(out.row("cues", frame.pts, &cue)?)
    }
}

ffrwd_node::export!(Hear);

#[cfg(test)]
mod tests {
    use super::*;
    use ffrwd_node::mock::Harness;
    use ffrwd_node::BoundStream;

    fn bytes(samples: &[f32]) -> Vec<u8> {
        samples.iter().flat_map(|s| s.to_le_bytes()).collect()
    }

    #[test]
    fn loudness_in_db_of_full_scale() {
        assert_eq!(loudness(&[1.0, -1.0]), "0 dB");
        assert_eq!(loudness(&[0.1; 8]), "-20 dB");
        assert_eq!(loudness(&[0.0; 8]), "silence");
        assert_eq!(loudness(&[]), "silence");
    }

    #[test]
    fn a_cue_a_window() {
        let a = BoundStream::audio("a", 0, RATE, 2, "f32");
        let mut hear = Harness::<Hear>::new("", vec![a]).unwrap();
        assert_eq!(hear.shape().inputs[0].window, WINDOW);
        let mut cues = Vec::new();
        for (window, length) in [(0, WINDOW), (1, WINDOW), (2, RATE / 2)] {
            let pts = (window * WINDOW) as i64;
            let tick = hear
                .tick(pts)
                .frame(0, pts, bytes(&vec![0.1; length as usize * 2]));
            let tick = if length < WINDOW { tick.last() } else { tick };
            cues.extend(hear.process(&tick).unwrap().messages("cues"));
        }
        let read: Vec<Cue> = cues
            .iter()
            .map(|(_, json)| ffrwd_node::parse(json).unwrap())
            .collect();
        assert_eq!(
            cues.iter().map(|(pts, _)| *pts).collect::<Vec<_>>(),
            [0, 96_000, 192_000]
        );
        assert_eq!(read[1], Cue::new(2.0, 4.0, "-20 dB"));
        assert_eq!(read[2], Cue::new(4.0, 4.5, "-20 dB"));
        let none = hear.process(&hear.tick(288_000).last()).unwrap();
        assert!(none.items.is_empty());
    }
}
