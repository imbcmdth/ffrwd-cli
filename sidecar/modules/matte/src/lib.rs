//! Makes a matte of the mark it finds and a row per mark, both from one
//! node: `mask`, the picture's size in gray, white inside the mark and black
//! elsewhere, and `spots`, the rows `spot` writes. An output the query does
//! not read is not made. Recipe 151.

use ffrwd_node::{Bound, Init, Input, Node, Out, Output, Result, Shape, Tick};
use serde::Deserialize;
use stand_ins::{Rgba, Spot, Spotter};

#[derive(Deserialize)]
struct Params {
    every: u64,
}

struct Matte {
    v: u32,
    width: usize,
    height: usize,
    spotter: Spotter,
    mask: bool,
    spots: bool,
}

/// A gray matte of `width` x `height`, white inside `spot`.
fn matte(spot: Option<&Spot>, width: usize, height: usize) -> Vec<u8> {
    let mut mask = vec![0u8; width * height];
    if let Some(spot) = spot {
        let (x0, y0) = (spot.x as usize, spot.y as usize);
        let x1 = (x0 + spot.w as usize).min(width);
        for y in y0..(y0 + spot.h as usize).min(height) {
            mask[y * width + x0.min(x1)..y * width + x1].fill(255);
        }
    }
    mask
}

impl Node for Matte {
    const NAME: &'static str = "matte";
    const VERSION: &'static str = "0.1.0";
    const PARAMS_SCHEMA: &'static str = r#"{"type":"object","properties":{"every":{"type":"integer","minimum":1,"default":30}},"additionalProperties":false}"#;
    type Params = Params;

    fn shape(_: &Params, _: &Bound) -> Result<Shape> {
        Ok(Shape::new()
            .input(Input::video("v").clock().pixel_formats(&["rgba"]))
            .output(Output::video("mask").pixel_format("gray"))
            .output(Output::rows("spots").schema::<Spot>())
            .one_to_one())
    }

    fn init(params: Params, init: &Init) -> Result<Matte> {
        let v = init.stream("v")?;
        let video = v.video_format().ok_or("`v` is a video input")?;
        Ok(Matte {
            v: v.id,
            width: video.width as usize,
            height: video.height as usize,
            spotter: Spotter::new(params.every),
            mask: init.latched("mask"),
            spots: init.latched("spots"),
        })
    }

    fn process(&mut self, tick: &Tick, out: &mut Out) -> Result<()> {
        let Some(frame) = tick.frame(self.v) else {
            return Ok(());
        };
        let bytes = tick.fetch(self.v, frame.index);
        let picture = Rgba::new(&bytes, self.width, self.height)?;
        let spot = self
            .spotter
            .see(tick.time_base().seconds(frame.pts), &picture);
        if self.mask {
            let mask = matte(spot.as_ref(), self.width, self.height);
            out.frame("mask", frame.pts, frame.duration, mask)?;
        }
        if let Some(spot) = spot.filter(|_| self.spots) {
            out.row("spots", frame.pts, &spot)?;
        }
        Ok(())
    }
}

ffrwd_node::export!(Matte);

#[cfg(test)]
mod tests {
    use super::*;
    use ffrwd_node::mock::Harness;
    use ffrwd_node::{BoundStream, Format, Payload, Rational};
    use stand_ins::{Picture, Rect};

    #[test]
    fn the_mask_follows_the_picture_in_gray() {
        let v = BoundStream::video("v", 0, 64, 48, "rgba", Rational::new(1, 15));
        let matte = Harness::<Matte>::new("", vec![v]).unwrap();
        let mask = matte.shape().find_output("mask").unwrap();
        let like = mask.like.as_ref().unwrap();
        assert_eq!(
            (like.port.as_deref(), like.pixel_format.as_deref()),
            (Some("v"), Some("gray"))
        );
        let spots = matte.shape().find_output("spots").unwrap();
        assert_eq!(spots.format, Some(Format::Data("json".to_owned())));
    }

    #[test]
    fn a_mask_and_a_row_from_one_frame() {
        let v = BoundStream::video("v", 0, 64, 48, "rgba", Rational::new(1, 15));
        let mut matte = Harness::<Matte>::new("", vec![v]).unwrap();
        let mut picture = Picture::filled(64, 48, [200, 30, 30, 255]);
        picture.fill(
            Rect {
                x0: 8,
                y0: 4,
                x1: 24,
                y1: 20,
            },
            [128, 128, 128, 255],
        );
        let emitted = matte
            .process(&matte.tick(2).frame(0, 2, picture.data))
            .unwrap();
        let [Payload::Frame { pts: 2, data, .. }] = emitted.on("mask")[..] else {
            panic!("no mask")
        };
        assert_eq!(data.len(), 64 * 48);
        assert_eq!(
            (data[4 * 64 + 8], data[4 * 64 + 7], data[20 * 64 + 8]),
            (255, 0, 0)
        );
        let rows = emitted.messages("spots");
        let spot: Spot = ffrwd_node::parse(&rows[0].1).unwrap();
        assert_eq!((spot.x, spot.y, spot.w, spot.h), (8, 4, 16, 16));
    }
}
