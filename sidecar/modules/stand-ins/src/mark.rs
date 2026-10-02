use ffrwd_frame::{Rect, Rgba};
use ffrwd_node::Spans;
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

/// The mark followed frame by frame: a sighting lasts while the mark stays
/// in view and at most `every` frames, so a mark always in view is a new
/// sighting every `every` frames.
pub struct Spotter {
    spans: Spans<()>,
}

impl Spotter {
    pub fn new(every: u64) -> Spotter {
        Spotter {
            spans: Spans::new().longest(every),
        }
    }

    /// The frame at `t` seconds: its row, when the mark is in view.
    pub fn see(&mut self, t: f64, frame: &Rgba) -> Option<Spot> {
        self.spans.tick(t);
        let rect = find(frame)?;
        let span = self.spans.see(());
        Some(Spot {
            start_t: span.start_t,
            id: span.number,
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
        let mut spotter = Spotter::new(2);
        let rows: Vec<(f64, u64)> = (0..5)
            .map(|n| {
                let spot = spotter.see(n as f64 / 10.0, &picture.view()).unwrap();
                (spot.start_t, spot.id)
            })
            .collect();
        assert_eq!(rows, [(0.0, 0), (0.0, 0), (0.2, 1), (0.2, 1), (0.4, 2)]);
    }
}
