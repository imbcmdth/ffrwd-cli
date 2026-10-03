//! A line of text crossing a `width` x `height` canvas from right to left, a
//! quarter of the canvas a second, at `fps`. A source: no inputs, a rate
//! clock, one relation row, and it never ends by itself. Recipe 154.

use ffrwd_node::{Bound, Init, Node, Out, Output, Rational, Result, Shape, Tick};
use serde::Deserialize;
use stand_ins::{text_size, Picture};

const BACKGROUND: [u8; 4] = [16, 24, 48, 255];
const TEXT: [u8; 4] = [255, 255, 255, 255];

#[derive(Deserialize)]
struct Params {
    text: String,
    width: u32,
    height: u32,
    fps: f64,
}

struct Ticker {
    text: String,
    width: usize,
    height: usize,
}

impl Ticker {
    fn scale(&self) -> usize {
        (self.height / 90).max(1)
    }

    /// Where the text's left edge stands `seconds` in: it enters at the
    /// right edge and comes round again once it has left at the left.
    fn left(&self, seconds: f64) -> i64 {
        let (wide, _) = text_size(&self.text, self.scale());
        let lap = (self.width + wide) as f64;
        let travelled = (seconds * self.width as f64 / 4.0).rem_euclid(lap);
        self.width as i64 - travelled.floor() as i64
    }
}

impl Node for Ticker {
    const NAME: &'static str = "ticker";
    const VERSION: &'static str = "0.1.0";
    const PARAMS_SCHEMA: &'static str = r#"{"type":"object","properties":{"text":{"type":"string"},"width":{"type":"integer","minimum":16,"maximum":8192,"default":1280},"height":{"type":"integer","minimum":16,"maximum":8192,"default":720},"fps":{"type":"number","exclusiveMinimum":0,"maximum":240,"default":30}},"required":["text"],"additionalProperties":false}"#;
    type Params = Params;

    fn shape(params: &Params, _: &Bound) -> Result<Shape> {
        let (width, height) = (params.width, params.height);
        Ok(Shape::new()
            .rate(Rational::approximate(params.fps, 1001))
            .output(
                Output::video("video")
                    .size(width, height)
                    .pixel_format("rgba")
                    .row(0),
            )
            .relation_row(&format!(r#"{{"width":{width},"height":{height}}}"#))
            .bounded(false)
            .pure())
    }

    fn init(params: Params, _: &Init) -> Result<Ticker> {
        Ok(Ticker {
            text: params.text,
            width: params.width as usize,
            height: params.height as usize,
        })
    }

    fn process(&mut self, tick: &Tick, out: &mut Out) -> Result<()> {
        let mut canvas = Picture::filled(self.width, self.height, BACKGROUND);
        let scale = self.scale();
        let (_, tall) = text_size(&self.text, scale);
        let top = (self.height.saturating_sub(tall) / 2) as i64;
        canvas.text(self.left(tick.seconds()), top, scale, &self.text, TEXT);
        Ok(out.frame("video", tick.pts(), Some(1), canvas.data)?)
    }
}

ffrwd_node::export!(Ticker);

#[cfg(test)]
mod tests {
    use super::*;
    use ffrwd_node::mock::Harness;
    use ffrwd_node::{Clock, Format, Payload, VideoFormat};

    #[test]
    fn a_source_at_its_rate() {
        let ticker = Harness::<Ticker>::new(r#"{"text":"Nothing to see here"}"#, vec![]).unwrap();
        let shape = ticker.shape();
        assert_eq!(shape.clock, Some(Clock::Rate(Rational::new(30, 1))));
        assert!(!shape.bounded && shape.inputs.is_empty());
        assert_eq!(shape.relation, [r#"{"width":1280,"height":720}"#]);
        assert!(matches!(
            shape.outputs[0].format,
            Some(Format::Video(VideoFormat {
                width: 1280,
                height: 720,
                ..
            }))
        ));
        assert!(Harness::<Ticker>::new("", vec![]).is_err());
    }

    #[test]
    fn the_text_crosses_from_the_right() {
        let ticker = Ticker {
            text: "abc".to_owned(),
            width: 400,
            height: 90,
        };
        assert_eq!(ticker.left(0.0), 400);
        assert_eq!(ticker.left(1.0), 300);
        assert_eq!(ticker.left(4.0), 0);
        let lap = (400 + text_size("abc", 1).0) as f64 / 100.0;
        assert!((ticker.left(lap + 1.0) - ticker.left(1.0)).abs() <= 1);
    }

    #[test]
    fn a_frame_a_tick() {
        let mut ticker =
            Harness::<Ticker>::new(r#"{"text":"hi","width":64,"height":32,"fps":10}"#, vec![])
                .unwrap();
        let emitted = ticker.process(&ticker.tick(25)).unwrap();
        let [Payload::Frame {
            pts: 25,
            duration: Some(1),
            data,
        }] = emitted.on("video")[..]
        else {
            panic!("no frame")
        };
        assert_eq!(data.len(), 64 * 32 * 4);
        assert!(data.chunks(4).any(|pixel| pixel == TEXT));
    }
}
