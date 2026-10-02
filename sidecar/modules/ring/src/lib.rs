//! Draws every six-field row it is handed on the picture the rows came from:
//! a box outline at `x`, `y`, `w`, `h`, its colour picked by `id`. A frame
//! with no rows leaves untouched. Recipe 145.

use ffrwd_node::{Bound, Init, Input, NoParams, Node, Out, Output, Result, Shape, Tick};
use serde::{Deserialize, Serialize};
use stand_ins::{Picture, Rect, PALETTE};

/// The row `ring` reads: what `spot` writes, every field a number.
#[derive(Default, Serialize, Deserialize)]
struct Ringed {
    start_t: f64,
    id: f64,
    x: f64,
    y: f64,
    w: f64,
    h: f64,
}

struct Ring {
    v: u32,
    spots: u32,
    width: usize,
    height: usize,
}

/// The outline's width: two pixels on a small picture, more on a large one.
fn thickness(width: usize) -> usize {
    (width / 160).max(2)
}

impl Node for Ring {
    const NAME: &'static str = "ring";
    const VERSION: &'static str = "0.1.0";
    type Params = NoParams;

    fn shape(_: &NoParams, _: &Bound) -> Result<Shape> {
        Ok(Shape::new()
            .input(Input::video("v").clock().pixel_formats(&["rgba"]))
            .input(Input::rows("spots").schema::<Ringed>())
            .output(Output::like("v"))
            .pure()
            .one_to_one())
    }

    fn init(_: NoParams, init: &Init) -> Result<Ring> {
        let v = init.stream("v")?;
        let video = v.video_format().ok_or("`v` is a video input")?;
        Ok(Ring {
            v: v.id,
            spots: init.stream("spots")?.id,
            width: video.width as usize,
            height: video.height as usize,
        })
    }

    fn process(&mut self, tick: &Tick, out: &mut Out) -> Result<()> {
        let Some(frame) = tick.frame(self.v) else {
            return Ok(());
        };
        let rows: Vec<Ringed> = tick.rows(self.spots)?;
        if rows.is_empty() {
            return Ok(out.pass("v", self.v, &frame)?);
        }
        let mut picture = Picture::new(tick.fetch(self.v, frame.index), self.width, self.height)?;
        for row in &rows {
            let Some(rect) = Rect::padded(row.x, row.y, row.w, row.h, 0.0, self.width, self.height)
            else {
                continue;
            };
            let colour = PALETTE[row.id.max(0.0) as usize % PALETTE.len()];
            picture.outline(rect, thickness(self.width), colour);
        }
        Ok(out.frame("v", frame.pts, frame.duration, picture.data)?)
    }
}

ffrwd_node::export!(Ring);

#[cfg(test)]
mod tests {
    use super::*;
    use ffrwd_node::mock::Harness;
    use ffrwd_node::{BoundStream, Payload, Rational};

    fn harness() -> Harness<Ring> {
        let tb = Rational::new(1, 15);
        let bound = vec![
            BoundStream::video("v", 0, 32, 24, "rgba", tb),
            BoundStream::rows("spots", 1, tb),
        ];
        Harness::new("", bound).unwrap()
    }

    #[test]
    fn a_frame_without_rows_passes_untouched() {
        let mut ring = harness();
        let tick = ring
            .tick(4)
            .frame_with(0, 4, Some(1), &[], vec![0; 32 * 24 * 4]);
        let emitted = ring.process(&tick).unwrap();
        assert_eq!(
            emitted.on("v"),
            [&Payload::Same {
                pts: 4,
                duration: Some(1),
                id: 0,
                index: 0
            }]
        );
    }

    #[test]
    fn a_row_is_outlined_in_its_colour() {
        let mut ring = harness();
        let row = r#"{"start_t":0.2,"id":1,"x":4,"y":4,"w":10,"h":8}"#;
        let tick = ring
            .tick(3)
            .frame(0, 3, vec![0; 32 * 24 * 4])
            .message(1, 3, row.as_bytes());
        let emitted = ring.process(&tick).unwrap();
        let [Payload::Frame { pts: 3, data, .. }] = emitted.on("v")[..] else {
            panic!("no frame: {emitted:?}")
        };
        let at = |x: usize, y: usize| &data[(y * 32 + x) * 4..(y * 32 + x) * 4 + 4];
        assert_eq!(at(4, 4)[..3], PALETTE[1][..3]);
        assert_eq!(at(13, 11)[..3], PALETTE[1][..3]);
        assert_eq!(at(8, 8), [0, 0, 0, 0]);
    }
}
