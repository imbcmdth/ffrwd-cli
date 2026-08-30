//! The box history and the union it emits, with no detector anywhere in it.
//!
//! A detector that flickers - a box a few pixels off every frame, and gone
//! entirely on some - is steadied by remembering what it saw. Every detection
//! is kept for a window of time; what leaves is the union of everything still
//! remembered, which is bigger, steadier, and survives the frames a detection
//! is missing from. A shot index that changes clears the memory, since a box
//! from the previous shot names nothing in this one.

/// One rectangle, in frame pixels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rect {
    pub x: i32,
    pub y: i32,
    pub w: u32,
    pub h: u32,
}

impl Rect {
    fn right(&self) -> i32 {
        self.x + self.w as i32
    }

    fn bottom(&self) -> i32 {
        self.y + self.h as i32
    }

    fn is_empty(&self) -> bool {
        self.w == 0 || self.h == 0
    }
}

/// Whether two rectangles share at least one pixel. Edges that merely touch
/// do not.
fn overlaps(a: Rect, b: Rect) -> bool {
    a.x < b.right() && b.x < a.right() && a.y < b.bottom() && b.y < a.bottom()
}

/// The smallest rectangle holding both.
fn bounding(a: Rect, b: Rect) -> Rect {
    let x = a.x.min(b.x);
    let y = a.y.min(b.y);
    let right = a.right().max(b.right());
    let bottom = a.bottom().max(b.bottom());
    Rect {
        x,
        y,
        w: (right - x) as u32,
        h: (bottom - y) as u32,
    }
}

/// Every rectangle merged with the ones it overlaps, to a fixed point: one
/// bounding rectangle per cluster of overlapping rectangles, in reading order.
///
/// Absorbing a rectangle grows the cluster, which may then reach one that was
/// left alone a moment ago - so each insertion sweeps until a whole pass
/// absorbs nothing.
fn union(rects: impl IntoIterator<Item = Rect>) -> Vec<Rect> {
    let mut clusters: Vec<Rect> = Vec::new();
    for rect in rects {
        let mut merged = rect;
        loop {
            let mut absorbed = false;
            let mut apart = Vec::with_capacity(clusters.len());
            for cluster in clusters.drain(..) {
                if overlaps(merged, cluster) {
                    merged = bounding(merged, cluster);
                    absorbed = true;
                } else {
                    apart.push(cluster);
                }
            }
            clusters = apart;
            if !absorbed {
                break;
            }
        }
        clusters.push(merged);
    }
    clusters.sort_by_key(|r| (r.y, r.x, r.w, r.h));
    clusters
}

/// One detection and the frame time it was seen at.
struct Seen {
    rect: Rect,
    at: f64,
}

/// The history and the rules over it.
pub struct Smoother {
    /// Seconds a detection is remembered for.
    window: f64,
    history: Vec<Seen>,
    /// The last shot index seen, absent until one arrives.
    shot: Option<i64>,
}

impl Smoother {
    pub fn new(window: f64) -> Smoother {
        Smoother {
            window,
            history: Vec::new(),
            shot: None,
        }
    }

    /// Changes the window without disturbing what is already remembered.
    pub fn set_window(&mut self, window: f64) {
        self.window = window;
    }

    /// One frame: the shot it belongs to if anything upstream said so, and
    /// what the detector found in it. Returns the rectangles to emit.
    ///
    /// A shot index differing from the last one seen clears the history first.
    /// Then the detections join it, everything older than the window leaves,
    /// and the union of what remains is the answer.
    pub fn step(&mut self, at: f64, shot: Option<i64>, detections: &[Rect]) -> Vec<Rect> {
        if let Some(shot) = shot {
            if self.shot.is_some_and(|previous| previous != shot) {
                self.history.clear();
            }
            self.shot = Some(shot);
        }

        self.history.extend(
            detections
                .iter()
                .filter(|rect| !rect.is_empty())
                .map(|rect| Seen { rect: *rect, at }),
        );
        let window = self.window;
        self.history.retain(|seen| at - seen.at <= window);

        union(self.history.iter().map(|seen| seen.rect))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The window facebox smooths over by default, and the step between frames
    /// at 25fps.
    const WINDOW: f64 = 0.5;
    const STEP: f64 = 0.04;

    fn rect(x: i32, y: i32, w: u32, h: u32) -> Rect {
        Rect { x, y, w, h }
    }

    #[test]
    fn overlapping_boxes_leave_as_one_bounding_rectangle() {
        let mut smoother = Smoother::new(WINDOW);
        let out = smoother.step(0.0, None, &[rect(0, 0, 10, 10), rect(5, 5, 10, 10)]);
        assert_eq!(out, vec![rect(0, 0, 15, 15)]);
    }

    #[test]
    fn boxes_that_do_not_meet_stay_apart() {
        let mut smoother = Smoother::new(WINDOW);
        // The second starts exactly where the first ends, so they share no
        // pixel and are two clusters.
        let out = smoother.step(0.0, None, &[rect(0, 0, 10, 10), rect(10, 0, 10, 10)]);
        assert_eq!(out, vec![rect(0, 0, 10, 10), rect(10, 0, 10, 10)]);
    }

    #[test]
    fn a_box_bridging_two_clusters_merges_all_three() {
        // The first two are far apart; the third overlaps both, so a single
        // pass that stopped at the first absorption would leave two rectangles.
        let mut smoother = Smoother::new(WINDOW);
        let out = smoother.step(
            0.0,
            None,
            &[rect(0, 0, 10, 10), rect(16, 0, 10, 10), rect(8, 0, 10, 10)],
        );
        assert_eq!(out, vec![rect(0, 0, 26, 10)]);
    }

    #[test]
    fn a_box_survives_the_frames_the_detector_misses() {
        let mut smoother = Smoother::new(WINDOW);
        let seen = rect(20, 20, 30, 30);
        assert_eq!(smoother.step(0.0, None, &[seen]), vec![seen]);
        for frame in 1..5 {
            assert_eq!(
                smoother.step(frame as f64 * STEP, None, &[]),
                vec![seen],
                "the box is still remembered {frame} frames after the last detection"
            );
        }
    }

    #[test]
    fn a_box_leaves_the_history_once_the_window_has_passed() {
        let mut smoother = Smoother::new(WINDOW);
        let seen = rect(20, 20, 30, 30);
        smoother.step(0.0, None, &[seen]);
        assert_eq!(
            smoother.step(WINDOW, None, &[]),
            vec![seen],
            "a detection exactly the window old is still remembered"
        );
        assert!(
            smoother.step(WINDOW + STEP, None, &[]).is_empty(),
            "a detection older than the window is gone"
        );
    }

    #[test]
    fn the_union_grows_as_a_jittering_box_moves_and_shrinks_as_it_expires() {
        let mut smoother = Smoother::new(WINDOW);
        smoother.step(0.0, None, &[rect(0, 0, 10, 10)]);
        assert_eq!(
            smoother.step(STEP, None, &[rect(4, 0, 10, 10)]),
            vec![rect(0, 0, 14, 10)],
            "both detections are remembered, so the union spans them"
        );
        assert_eq!(
            smoother.step(WINDOW + STEP, None, &[]),
            vec![rect(4, 0, 10, 10)],
            "the first has expired and only the second is left"
        );
    }

    #[test]
    fn a_new_shot_clears_the_history_before_the_frame_is_processed() {
        let mut smoother = Smoother::new(WINDOW);
        let seen = rect(20, 20, 30, 30);
        assert_eq!(smoother.step(0.0, Some(0), &[seen]), vec![seen]);
        assert!(
            smoother.step(STEP, Some(1), &[]).is_empty(),
            "a box from the previous shot names nothing in this one"
        );
    }

    #[test]
    fn a_new_shot_keeps_what_that_frame_itself_detected() {
        let mut smoother = Smoother::new(WINDOW);
        smoother.step(0.0, Some(0), &[rect(0, 0, 10, 10)]);
        let fresh = rect(40, 40, 10, 10);
        assert_eq!(
            smoother.step(STEP, Some(1), &[fresh]),
            vec![fresh],
            "the history is cleared before the frame's own detections join it"
        );
    }

    #[test]
    fn the_same_shot_index_clears_nothing() {
        let mut smoother = Smoother::new(WINDOW);
        let seen = rect(20, 20, 30, 30);
        smoother.step(0.0, Some(7), &[seen]);
        assert_eq!(smoother.step(STEP, Some(7), &[]), vec![seen]);
    }

    #[test]
    fn frames_that_carry_no_shot_index_never_clear() {
        // Nothing upstream is counting shots, which is facebox on its own:
        // smoothing, and no clearing ever.
        let mut smoother = Smoother::new(WINDOW);
        let seen = rect(20, 20, 30, 30);
        smoother.step(0.0, None, &[seen]);
        for frame in 1..5 {
            assert_eq!(smoother.step(frame as f64 * STEP, None, &[]), vec![seen]);
        }
    }

    #[test]
    fn an_empty_detection_is_not_remembered() {
        let mut smoother = Smoother::new(WINDOW);
        assert!(smoother
            .step(0.0, None, &[rect(5, 5, 0, 10), rect(5, 5, 10, 0)])
            .is_empty());
    }

    #[test]
    fn a_window_of_zero_remembers_only_this_frame() {
        let mut smoother = Smoother::new(0.0);
        let first = rect(0, 0, 10, 10);
        assert_eq!(smoother.step(0.0, None, &[first]), vec![first]);
        assert!(smoother.step(STEP, None, &[]).is_empty());
    }

    #[test]
    fn a_wider_window_takes_effect_without_losing_the_history() {
        let mut smoother = Smoother::new(WINDOW);
        let seen = rect(20, 20, 30, 30);
        smoother.step(0.0, None, &[seen]);
        smoother.set_window(2.0);
        assert_eq!(
            smoother.step(1.0, None, &[]),
            vec![seen],
            "a detection a second old outlives the window it was made under"
        );
    }
}
