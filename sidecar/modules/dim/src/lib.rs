//! Dims the picture inside every box it is handed by `amount`: 0 leaves it,
//! 1 makes it black. It reads four fields, `{x, y, w, h}`, and any row
//! carrying them will do. Recipes 146 and 151.

use ffrwd_node::{Bound, Init, Input, Node, Out, Output, Result, Shape, Tick};
use serde::{Deserialize, Serialize};
use stand_ins::{Picture, Rect};

#[derive(Deserialize)]
struct Params {
    amount: f64,
}

#[derive(Default, Serialize, Deserialize)]
struct Box {
    x: f64,
    y: f64,
    w: f64,
    h: f64,
}

struct Dim {
    v: u32,
    boxes: u32,
    width: usize,
    height: usize,
    amount: f64,
}

/// One flag a pixel: inside any of `boxes`.
fn covered(boxes: &[Box], width: usize, height: usize) -> Vec<bool> {
    let mut inside = vec![false; width * height];
    for found in boxes {
        let Some(rect) = Rect::padded(found.x, found.y, found.w, found.h, 0.0, width, height)
        else {
            continue;
        };
        for y in rect.y0..rect.y1 {
            inside[y * width + rect.x0..y * width + rect.x1].fill(true);
        }
    }
    inside
}

impl Node for Dim {
    const NAME: &'static str = "dim";
    const VERSION: &'static str = "0.1.0";
    const PARAMS_SCHEMA: &'static str = r#"{"type":"object","properties":{"amount":{"type":"number","minimum":0,"maximum":1,"default":0.5}},"additionalProperties":false}"#;
    type Params = Params;

    fn shape(_: &Params, _: &Bound) -> Result<Shape> {
        Ok(Shape::new()
            .input(Input::video("v").clock().pixel_formats(&["rgba"]))
            .input(Input::rows("boxes").schema::<Box>())
            .output(Output::like("v"))
            .pure()
            .one_to_one())
    }

    fn init(params: Params, init: &Init) -> Result<Dim> {
        let v = init.stream("v")?;
        let video = v.video_format().ok_or("`v` is a video input")?;
        Ok(Dim {
            v: v.id,
            boxes: init.stream("boxes")?.id,
            width: video.width as usize,
            height: video.height as usize,
            amount: params.amount,
        })
    }

    fn set_params(&mut self, params: Params) -> Result<()> {
        self.amount = params.amount;
        Ok(())
    }

    fn process(&mut self, tick: &Tick, out: &mut Out) -> Result<()> {
        let Some(frame) = tick.frame(self.v) else {
            return Ok(());
        };
        let boxes: Vec<Box> = tick.rows(self.boxes)?;
        if boxes.is_empty() || self.amount == 0.0 {
            return Ok(out.pass("v", self.v, &frame)?);
        }
        let mut picture = Picture::new(tick.fetch(self.v, frame.index), self.width, self.height)?;
        picture.darken(&covered(&boxes, self.width, self.height), self.amount);
        Ok(out.frame("v", frame.pts, frame.duration, picture.data)?)
    }
}

ffrwd_node::export!(Dim);

#[cfg(test)]
mod tests {
    use super::*;
    use ffrwd_node::mock::Harness;
    use ffrwd_node::{BoundStream, Payload, Rational};

    #[test]
    fn overlapping_boxes_dim_once() {
        let inside = covered(
            &[
                Box {
                    x: 1.0,
                    y: 1.0,
                    w: 2.0,
                    h: 2.0,
                },
                Box {
                    x: 2.0,
                    y: 2.0,
                    w: 9.0,
                    h: 9.0,
                },
            ],
            4,
            4,
        );
        let count = inside.iter().filter(|inside| **inside).count();
        assert_eq!(count, 3 + 4);
    }

    #[test]
    fn six_field_rows_are_read_for_their_box() {
        let tb = Rational::new(1, 15);
        let bound = vec![
            BoundStream::video("v", 0, 4, 4, "rgba", tb),
            BoundStream::rows("boxes", 1, tb),
        ];
        let mut dim = Harness::<Dim>::new(r#"{"amount":0.75}"#, bound).unwrap();
        let spot = r#"{"start_t":0,"id":0,"x":0,"y":0,"w":2,"h":1}"#;
        let tick = dim
            .tick(0)
            .frame(0, 0, vec![200; 4 * 4 * 4])
            .message(1, 0, spot.as_bytes());
        let emitted = dim.process(&tick).unwrap();
        let [Payload::Frame { data, .. }] = emitted.on("v")[..] else {
            panic!("no frame")
        };
        assert_eq!(&data[..4], &[50, 50, 50, 200]);
        assert_eq!(&data[8..12], &[200, 200, 200, 200]);
    }
}
