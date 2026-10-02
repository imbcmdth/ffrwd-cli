//! Finds a mark in each frame and returns rows alone: one a frame while the
//! mark is in view, `{start_t, id, x, y, w, h}`, every row of one sighting
//! carrying the time its block began as `start_t`, a new sighting every
//! `every` frames of the run. Counted by the tick's ordinal, so it is pure
//! and a run split across workers names every sighting alike. Recipes 145,
//! 146 and 152.

use ffrwd_node::{Bound, Init, Input, Node, Out, Output, Result, Shape, Tick};
use serde::Deserialize;
use stand_ins::{Rgba, Spot, Spotter};

#[derive(Deserialize)]
struct Params {
    every: u64,
}

struct SpotNode {
    v: u32,
    width: usize,
    height: usize,
    spotter: Spotter,
}

impl Node for SpotNode {
    const NAME: &'static str = "spot";
    const VERSION: &'static str = "0.1.0";
    const PARAMS_SCHEMA: &'static str = r#"{"type":"object","properties":{"every":{"type":"integer","minimum":1,"default":30}},"additionalProperties":false}"#;
    type Params = Params;

    fn shape(_: &Params, _: &Bound) -> Result<Shape> {
        Ok(Shape::new()
            .input(Input::video("v").clock().pixel_formats(&["rgba"]))
            .output(Output::rows("spots").schema::<Spot>())
            .pure())
    }

    fn init(params: Params, init: &Init) -> Result<SpotNode> {
        let v = init.stream("v")?;
        let video = v.video_format().ok_or("`v` is a video input")?;
        Ok(SpotNode {
            v: v.id,
            width: video.width as usize,
            height: video.height as usize,
            spotter: Spotter::new(params.every, v),
        })
    }

    fn process(&mut self, tick: &Tick, out: &mut Out) -> Result<()> {
        for frame in tick.frames(self.v) {
            let bytes = tick.fetch(self.v, frame.index);
            let picture = Rgba::new(&bytes, self.width, self.height)?;
            if let Some(spot) = self.spotter.see(tick.ordinal(), frame.pts, &picture) {
                out.row("spots", frame.pts, &spot)?;
            }
        }
        Ok(())
    }
}

ffrwd_node::export!(SpotNode);

#[cfg(test)]
mod tests {
    use super::*;
    use ffrwd_node::mock::Harness;
    use ffrwd_node::{BoundStream, Rational};
    use stand_ins::{Picture, Rect};

    fn frame(x0: usize) -> Vec<u8> {
        let mut picture = Picture::filled(64, 48, [200, 30, 30, 255]);
        let mark = Rect {
            x0,
            y0: 8,
            x1: x0 + 12,
            y1: 20,
        };
        picture.fill(mark, [128, 128, 128, 255]);
        picture.data
    }

    #[test]
    fn a_row_a_frame_named_by_the_sighting() {
        let v = BoundStream::video("v", 0, 64, 48, "rgba", Rational::new(1, 10));
        let mut spot = Harness::<SpotNode>::new(r#"{"every":3}"#, vec![v]).unwrap();
        let mut rows = Vec::new();
        for n in 0..4 {
            let tick = spot.tick(n).frame(0, n, frame(4 + n as usize));
            rows.extend(spot.process(&tick).unwrap().messages("spots"));
        }
        let read: Vec<Spot> = rows
            .iter()
            .map(|(_, json)| ffrwd_node::parse(json).unwrap())
            .collect();
        assert_eq!(
            rows.iter().map(|(pts, _)| *pts).collect::<Vec<_>>(),
            [0, 1, 2, 3]
        );
        assert_eq!(
            read.iter()
                .map(|spot| (spot.start_t, spot.id))
                .collect::<Vec<_>>(),
            [(0.0, 0), (0.0, 0), (0.0, 0), (0.3, 1)]
        );
        assert_eq!((read[2].x, read[2].y, read[2].w, read[2].h), (6, 8, 12, 12));
    }

    #[test]
    fn no_mark_no_row() {
        let v = BoundStream::video("v", 0, 64, 48, "rgba", Rational::new(1, 10));
        let mut spot = Harness::<SpotNode>::new("", vec![v]).unwrap();
        let blank = Picture::filled(64, 48, [0, 0, 0, 255]).data;
        let emitted = spot.process(&spot.tick(0).frame(0, 0, blank)).unwrap();
        assert!(emitted.items.is_empty());
    }
}
