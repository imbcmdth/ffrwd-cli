//! Shows a feed over the picture, a third of its size in the lower right
//! corner, while the feed has a frame to show; the picture alone otherwise.
//! The feed is a hold input on the loopback port `port` names, shown from
//! `lead` seconds after its first frame arrives, and the host conforms it to
//! the picture's size. Recipe 149.

use ffrwd_node::{Bound, Init, Input, Node, Out, Output, Result, Shape, Tick};
use serde::Deserialize;
use stand_ins::{Picture, Rect};

const BORDER: [u8; 4] = [255, 255, 255, 255];

#[derive(Deserialize)]
struct Params {
    lead: f64,
}

struct Inset {
    v: u32,
    feed: Option<u32>,
    width: usize,
    height: usize,
}

/// Where the feed goes on a `width` x `height` picture: a third of it, a
/// fortieth of its width in from the lower right corner.
fn placed(width: usize, height: usize) -> Rect {
    let margin = (width / 40).max(1);
    let (w, h) = ((width / 3).max(1), (height / 3).max(1));
    let x1 = width.saturating_sub(margin);
    let y1 = height.saturating_sub(margin);
    Rect {
        x0: x1.saturating_sub(w),
        y0: y1.saturating_sub(h),
        x1,
        y1,
    }
}

impl Node for Inset {
    const NAME: &'static str = "inset";
    const VERSION: &'static str = "0.1.0";
    const PARAMS_SCHEMA: &'static str = r#"{"type":"object","properties":{"port":{"type":"integer","minimum":0,"maximum":65535,"default":9000},"lead":{"type":"number","minimum":0,"default":0.5}},"additionalProperties":false}"#;
    type Params = Params;

    fn shape(params: &Params, _: &Bound) -> Result<Shape> {
        Ok(Shape::new()
            .input(Input::video("v").clock().pixel_formats(&["rgba"]))
            .input(
                Input::video("feed")
                    .optional()
                    .hold()
                    .lead(params.lead)
                    .port_param("port")
                    .like("v")
                    .pixel_formats(&["rgba"]),
            )
            .output(Output::like("v"))
            .pure()
            .one_to_one())
    }

    fn init(_: Params, init: &Init) -> Result<Inset> {
        let v = init.stream("v")?;
        let video = v.video_format().ok_or("`v` is a video input")?;
        Ok(Inset {
            v: v.id,
            feed: init.optional("feed").map(|feed| feed.id),
            width: video.width as usize,
            height: video.height as usize,
        })
    }

    fn process(&mut self, tick: &Tick, out: &mut Out) -> Result<()> {
        let Some(frame) = tick.frame(self.v) else {
            return Ok(());
        };
        let shown = self.feed.and_then(|feed| Some((feed, tick.frame(feed)?)));
        let Some((feed, held)) = shown else {
            return Ok(out.pass("v", self.v, &frame)?);
        };
        let mut picture = Picture::new(tick.fetch(self.v, frame.index), self.width, self.height)?;
        let fed = Picture::new(tick.fetch(feed, held.index), self.width, self.height)?;
        let at = placed(self.width, self.height);
        let border = (self.width / 320).max(1);
        picture.fill(
            Rect {
                x0: at.x0.saturating_sub(border),
                y0: at.y0.saturating_sub(border),
                x1: at.x1 + border,
                y1: at.y1 + border,
            },
            BORDER,
        );
        picture.put(at.x0, at.y0, &fed.resized(at.width(), at.height()));
        Ok(out.frame("v", frame.pts, frame.duration, picture.data)?)
    }
}

ffrwd_node::export!(Inset);

#[cfg(test)]
mod tests {
    use super::*;
    use ffrwd_node::mock::Harness;
    use ffrwd_node::{BoundStream, Pairing, Payload, Rational};

    const TB: Rational = Rational::new(1, 15);

    #[test]
    fn the_feed_is_held_on_its_port() {
        let v = BoundStream::video("v", 0, 120, 90, "rgba", TB);
        let inset = Harness::<Inset>::new(r#"{"port":9100,"lead":0.25}"#, vec![v]).unwrap();
        let feed = inset.shape().find_input("feed").unwrap();
        let Pairing::Hold(hold) = &feed.pairing else {
            panic!("not held")
        };
        assert_eq!(
            (hold.lead, hold.port_param.as_deref()),
            (0.25, Some("port"))
        );
        assert_eq!(feed.accepts.like.as_deref(), Some("v"));
    }

    #[test]
    fn the_feed_shows_in_the_corner_while_it_has_a_frame() {
        let bound = vec![
            BoundStream::video("v", 0, 120, 90, "rgba", TB),
            BoundStream::video("feed", 1, 120, 90, "rgba", TB),
        ];
        let mut inset = Harness::<Inset>::new("", bound).unwrap();
        let picture = Picture::filled(120, 90, [0, 0, 0, 255]).data;
        let fed = Picture::filled(120, 90, [0, 0, 255, 255]).data;
        let alone = inset.tick(0).frame(0, 0, picture.clone());
        assert!(matches!(
            inset.process(&alone).unwrap().on("v")[..],
            [Payload::Same { .. }]
        ));
        let both = inset.tick(1).frame(0, 1, picture).frame(1, 7, fed);
        let emitted = inset.process(&both).unwrap();
        let [Payload::Frame { data, .. }] = emitted.on("v")[..] else {
            panic!("no frame")
        };
        let at = |x: usize, y: usize| &data[(y * 120 + x) * 4..(y * 120 + x) * 4 + 4];
        let corner = placed(120, 90);
        assert_eq!(at(corner.x0 + 5, corner.y0 + 5), [0, 0, 255, 255]);
        assert_eq!(at(5, 5), [0, 0, 0, 255]);
    }
}
