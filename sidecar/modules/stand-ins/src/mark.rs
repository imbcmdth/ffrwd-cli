use ffrwd_frame::{Rect, Rgba};
use ffrwd_node::{BoundStream, Rational};
use serde::{Deserialize, Serialize};

/// Fewer pixels than this are noise, not the mark.
const SMALLEST: usize = 32;

fn grey(pixel: &[u8]) -> bool {
    let (r, g, b) = (pixel[0], pixel[1], pixel[2]);
    let (lo, hi) = (r.min(g).min(b), r.max(g).max(b));
    hi - lo < 24 && lo > 96 && hi < 160
}

/// The mark the stand-ins track: the box around the largest patch of mid
/// grey in the picture, which in ffmpeg's `testsrc2` is the grey shape that
/// grows and shrinks at the lower left. None when no patch is big enough.
pub fn find(frame: &Rgba) -> Option<Rect> {
    let (width, height) = (frame.width, frame.height);
    let (pixels, _) = frame.data.as_chunks::<4>();
    let mut open: Vec<bool> = pixels.iter().map(|pixel| grey(pixel)).collect();
    let mut best: Option<(usize, Rect)> = None;
    let mut stack = Vec::new();
    for start in 0..open.len() {
        if !open[start] {
            continue;
        }
        open[start] = false;
        stack.push(start);
        let mut count = 0;
        let (x, y) = (start % width, start / width);
        let mut rect = Rect {
            x0: x,
            y0: y,
            x1: x + 1,
            y1: y + 1,
        };
        while let Some(at) = stack.pop() {
            count += 1;
            let (x, y) = (at % width, at / width);
            rect.x0 = rect.x0.min(x);
            rect.y0 = rect.y0.min(y);
            rect.x1 = rect.x1.max(x + 1);
            rect.y1 = rect.y1.max(y + 1);
            let neighbours = [
                (x > 0).then(|| at - 1),
                (x + 1 < width).then(|| at + 1),
                (y > 0).then(|| at - width),
                (y + 1 < height).then(|| at + width),
            ];
            for next in neighbours.into_iter().flatten() {
                if open[next] {
                    open[next] = false;
                    stack.push(next);
                }
            }
        }
        if count >= SMALLEST && best.is_none_or(|(most, _)| count > most) {
            best = Some((count, rect));
        }
    }
    best.map(|(_, rect)| rect)
}

/// One row of the mark: where it is on this frame, and the sighting it
/// belongs to.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Spot {
    /// When this sighting began, in seconds.
    pub start_t: f64,
    /// The sighting's number, from 0.
    pub id: u64,
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

/// The mark named by frame number: frames `0..every` of the run are
/// sighting 0, the next `every` sighting 1, and so on, each starting at its
/// first frame's time. A function of the frame and its number alone, so
/// every worker of a split run names a frame's sighting alike.
pub struct Spotter {
    every: u64,
    /// One frame of the picture, in its time base's ticks.
    step: i64,
    time_base: Rational,
}

impl Spotter {
    /// Sightings of `every` frames of `v`, a frame being the length its rate
    /// says, or one tick of its time base where the call gave no rate.
    pub fn new(every: u64, v: &BoundStream) -> Spotter {
        let time_base = v.info.time_base;
        let step = v.hint.rate.map_or(1, |rate| {
            let num = i64::from(time_base.den) * i64::from(rate.den);
            let den = (i64::from(time_base.num) * i64::from(rate.num)).max(1);
            ((num + den / 2) / den).max(1)
        });
        Spotter {
            every: every.max(1),
            step,
            time_base,
        }
    }

    /// Frame `ordinal` of the run, at `pts`: its row, when the mark is in
    /// view.
    pub fn see(&self, ordinal: u64, pts: i64, frame: &Rgba) -> Option<Spot> {
        let rect = find(frame)?;
        let into = (ordinal % self.every) as i64;
        Some(Spot {
            start_t: self.time_base.seconds(pts - into * self.step),
            id: ordinal / self.every,
            x: rect.x0 as u32,
            y: rect.y0 as u32,
            w: rect.width() as u32,
            h: rect.height() as u32,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Picture;

    fn with_mark(rect: Rect) -> Picture {
        let mut picture = Picture::filled(64, 48, [200, 30, 30, 255]);
        picture.fill(rect, [128, 128, 128, 255]);
        picture.fill(
            Rect {
                x0: 60,
                y0: 0,
                x1: 62,
                y1: 2,
            },
            [128, 128, 128, 255],
        );
        picture
    }

    #[test]
    fn the_largest_grey_patch_is_the_mark() {
        let rect = Rect {
            x0: 10,
            y0: 20,
            x1: 26,
            y1: 28,
        };
        assert_eq!(find(&with_mark(rect).view()), Some(rect));
        assert_eq!(
            find(&Picture::filled(64, 48, [200, 30, 30, 255]).view()),
            None
        );
    }

    #[test]
    fn a_sighting_is_split_every_so_many_frames() {
        let picture = with_mark(Rect {
            x0: 4,
            y0: 4,
            x1: 20,
            y1: 20,
        });
        let v = BoundStream::video("v", 0, 64, 48, "rgba", Rational::new(1, 10));
        let spotter = Spotter::new(2, &v);
        let rows: Vec<(f64, u64)> = (0..5)
            .map(|n| {
                let spot = spotter.see(n, n as i64, &picture.view()).unwrap();
                (spot.start_t, spot.id)
            })
            .collect();
        assert_eq!(rows, [(0.0, 0), (0.0, 0), (0.2, 1), (0.2, 1), (0.4, 2)]);
        let fine = BoundStream::video("v", 0, 64, 48, "rgba", Rational::new(1, 90_000))
            .rate(Rational::new(30, 1));
        let spot = Spotter::new(3, &fine)
            .see(7, 7 * 3000, &picture.view())
            .unwrap();
        assert_eq!(
            (spot.id, spot.start_t),
            (2, 0.2),
            "frame 7 is in the third sighting, which began at frame 6, 0.2 s in"
        );
    }
}
