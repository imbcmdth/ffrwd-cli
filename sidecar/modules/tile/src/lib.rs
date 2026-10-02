//! Lays any number of pictures out in a grid of `columns`, on a `width` x
//! `height` canvas, each scaled to fit its cell. It ticks at the first
//! picture's rate, or at `fps` when the call gives one, and shows each
//! picture's newest frame, so the pictures need not share a rate or a start.
//! Recipe 150.

use ffrwd_node::{Anchor, Bound, Init, Input, Node, Out, Output, Rational, Result, Shape, Tick};
use serde::Deserialize;
use stand_ins::{Picture, Rect};

const BLACK: [u8; 4] = [0, 0, 0, 255];

#[derive(Deserialize)]
struct Params {
    columns: usize,
    fps: Option<f64>,
    width: u32,
    height: u32,
}

struct Tiled {
    id: u32,
    width: usize,
    height: usize,
    at: Rect,
}

struct Tile {
    tiles: Vec<Tiled>,
    width: usize,
    height: usize,
}

/// The cells of a grid of `count` pictures in `columns` across a `width` x
/// `height` canvas, in reading order.
fn cells(count: usize, columns: usize, width: usize, height: usize) -> Vec<Rect> {
    let columns = columns.max(1);
    let rows = count.div_ceil(columns).max(1);
    (0..count)
        .map(|n| {
            let (column, row) = (n % columns, n / columns);
            Rect {
                x0: column * width / columns,
                y0: row * height / rows,
                x1: (column + 1) * width / columns,
                y1: (row + 1) * height / rows,
            }
        })
        .collect()
}

/// The largest `width` x `height` picture that fits `cell` keeping its
/// shape, centred in it.
fn fit(width: usize, height: usize, cell: Rect) -> Rect {
    let (cw, ch) = (cell.width(), cell.height());
    let (w, h) = if width * ch <= height * cw {
        ((width * ch / height.max(1)).max(1), ch)
    } else {
        (cw, (height * cw / width.max(1)).max(1))
    };
    let x0 = cell.x0 + (cw - w.min(cw)) / 2;
    let y0 = cell.y0 + (ch - h.min(ch)) / 2;
    Rect {
        x0,
        y0,
        x1: x0 + w,
        y1: y0 + h,
    }
}

impl Node for Tile {
    const NAME: &'static str = "tile";
    const VERSION: &'static str = "0.1.0";
    const PARAMS_SCHEMA: &'static str = r#"{"type":"object","properties":{"columns":{"type":"integer","minimum":1,"default":2},"fps":{"type":["number","null"],"exclusiveMinimum":0},"width":{"type":"integer","minimum":16,"default":1280},"height":{"type":"integer","minimum":16,"default":720}},"additionalProperties":false}"#;
    type Params = Params;

    fn shape(params: &Params, _: &Bound) -> Result<Shape> {
        let shape = Shape::new()
            .input(
                Input::video("v")
                    .many()
                    .hold()
                    .anchor(Anchor::SharedClock)
                    .pixel_formats(&["rgba"]),
            )
            .output(
                Output::video("v")
                    .size(params.width, params.height)
                    .pixel_format("rgba"),
            )
            .pure();
        Ok(match params.fps {
            Some(fps) => shape.rate(Rational::approximate(fps, 1001)),
            None => shape.rate_of("v"),
        })
    }

    fn init(params: Params, init: &Init) -> Result<Tile> {
        let (width, height) = (params.width as usize, params.height as usize);
        let streams = init.streams("v");
        let at = cells(streams.len(), params.columns, width, height);
        let tiles = streams
            .iter()
            .zip(at)
            .map(|(stream, cell)| {
                let video = stream
                    .video_format()
                    .ok_or_else(|| format!("picture {} of `v` has no size", stream.id))?;
                let (w, h) = (video.width as usize, video.height as usize);
                Ok(Tiled {
                    id: stream.id,
                    width: w,
                    height: h,
                    at: fit(w, h, cell),
                })
            })
            .collect::<Result<Vec<Tiled>, String>>()?;
        Ok(Tile {
            tiles,
            width,
            height,
        })
    }

    fn process(&mut self, tick: &Tick, out: &mut Out) -> Result<()> {
        let mut canvas = Picture::filled(self.width, self.height, BLACK);
        let mut shown = false;
        for tile in &self.tiles {
            let Some(frame) = tick.frame(tile.id) else {
                continue;
            };
            let picture = Picture::new(tick.fetch(tile.id, frame.index), tile.width, tile.height)?;
            canvas.put(
                tile.at.x0,
                tile.at.y0,
                &picture.resized(tile.at.width(), tile.at.height()),
            );
            shown = true;
        }
        if !shown && tick.last() {
            return Ok(());
        }
        Ok(out.frame("v", tick.pts(), Some(1), canvas.data)?)
    }
}

ffrwd_node::export!(Tile);

#[cfg(test)]
mod tests {
    use super::*;
    use ffrwd_node::mock::Harness;
    use ffrwd_node::{BoundStream, Clock, Payload};

    #[test]
    fn a_grid_of_cells_in_reading_order() {
        let grid = cells(3, 2, 1280, 720);
        assert_eq!(
            grid.iter()
                .map(|cell| (cell.x0, cell.y0, cell.x1, cell.y1))
                .collect::<Vec<_>>(),
            [(0, 0, 640, 360), (640, 0, 1280, 360), (0, 360, 640, 720)]
        );
        let row = cells(3, 3, 1280, 720);
        assert_eq!((row[2].x0, row[2].x1, row[2].y1), (853, 1280, 720));
    }

    #[test]
    fn a_picture_fits_its_cell_keeping_its_shape() {
        let cell = Rect {
            x0: 0,
            y0: 0,
            x1: 426,
            y1: 720,
        };
        let placed = fit(320, 240, cell);
        assert_eq!((placed.width(), placed.height()), (426, 319));
        assert_eq!((placed.x0, placed.y0), (0, 200));
        let wide = Rect {
            x0: 640,
            y0: 0,
            x1: 1280,
            y1: 360,
        };
        assert_eq!(
            fit(320, 240, wide),
            Rect {
                x0: 720,
                y0: 0,
                x1: 1200,
                y1: 360
            }
        );
    }

    #[test]
    fn the_rate_is_the_first_picture_s_unless_fps_says() {
        let pictures = |count: u32| {
            (0..count)
                .map(|id| BoundStream::video("v", id, 32, 24, "rgba", Rational::new(1, 15360)))
                .collect::<Vec<_>>()
        };
        let follows = Harness::<Tile>::new("", pictures(2)).unwrap();
        assert_eq!(follows.shape().clock, Some(Clock::RateOf("v".to_owned())));
        let fixed = Harness::<Tile>::new(r#"{"fps":29.97}"#, pictures(2)).unwrap();
        assert_eq!(
            fixed.shape().clock,
            Some(Clock::Rate(Rational::new(2997, 100)))
        );
    }

    #[test]
    fn each_picture_lands_in_its_cell() {
        let bound = (0..3)
            .map(|id| BoundStream::video("v", id, 16, 12, "rgba", Rational::new(1, 15)))
            .collect();
        let mut tile = Harness::<Tile>::new(r#"{"columns":2,"width":64,"height":48}"#, bound)
            .unwrap()
            .clock(Rational::new(1, 15));
        let red = Picture::filled(16, 12, [255, 0, 0, 255]).data;
        let blue = Picture::filled(16, 12, [0, 0, 255, 255]).data;
        let tick = tile.tick(5).frame(0, 77, red).frame(2, 3, blue);
        let emitted = tile.process(&tick).unwrap();
        let [Payload::Frame { pts: 5, data, .. }] = emitted.on("v")[..] else {
            panic!("no frame")
        };
        let at = |x: usize, y: usize| &data[(y * 64 + x) * 4..(y * 64 + x) * 4 + 4];
        assert_eq!(at(16, 12), [255, 0, 0, 255]);
        assert_eq!(at(48, 12), [0, 0, 0, 255]);
        assert_eq!(at(16, 36), [0, 0, 255, 255]);
        let ended = tile.process(&tile.tick(6).last()).unwrap();
        assert!(ended.items.is_empty());
    }
}
