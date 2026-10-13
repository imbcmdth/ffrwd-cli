//! Reports the colour its picture was declared in: one row a frame, saying
//! the pixel format it read and the range, primaries, transfer and matrix
//! the host told it at `init`. Reads the pixel format `pix_fmt` names,
//! yuv420p unless it says rgba, so two calls read one stream two ways. The
//! rows another call made, handed to `seen`, are written on beside its own,
//! so one rows file holds what both were told.

use ffrwd_node::{Bound, Init, Input, Node, Out, Output, Result, Shape, Tick};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Params {
    #[serde(default = "yuv420p")]
    pix_fmt: String,
}

fn yuv420p() -> String {
    "yuv420p".to_string()
}

/// One row: the format a frame was read in and the colour it was declared in,
/// "none" where the host declared none.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Colour {
    pub pix_fmt: String,
    pub range: String,
    pub primaries: String,
    pub trc: String,
    pub space: String,
}

struct ColourProbe {
    v: u32,
    seen: Option<u32>,
    colour: Colour,
}

impl Node for ColourProbe {
    const NAME: &'static str = "colour_probe";
    const VERSION: &'static str = "0.1.0";
    const PARAMS_SCHEMA: &'static str = r#"{"type":"object","properties":{"pix_fmt":{"type":"string","enum":["yuv420p","rgba"],"default":"yuv420p"}},"additionalProperties":false}"#;
    type Params = Params;

    fn shape(params: &Params, _: &Bound) -> Result<Shape> {
        Ok(Shape::new()
            .input(
                Input::video("v")
                    .clock()
                    .pixel_formats(&[params.pix_fmt.as_str()]),
            )
            .input(Input::rows("seen").optional().schema::<Colour>())
            .output(Output::rows("colour").schema::<Colour>())
            .pure())
    }

    fn init(_: Params, init: &Init) -> Result<ColourProbe> {
        let v = init.stream("v")?;
        let video = v.video_format().ok_or("`v` is a video input")?;
        let none = || "none".to_string();
        let colour = match &video.color {
            Some(c) => Colour {
                pix_fmt: video.pix_fmt.clone(),
                range: c.range.clone(),
                primaries: c.primaries.clone(),
                trc: c.trc.clone(),
                space: c.space.clone(),
            },
            None => Colour {
                pix_fmt: video.pix_fmt.clone(),
                range: none(),
                primaries: none(),
                trc: none(),
                space: none(),
            },
        };
        Ok(ColourProbe {
            v: v.id,
            seen: init.optional("seen").map(|seen| seen.id),
            colour,
        })
    }

    fn process(&mut self, tick: &Tick, out: &mut Out) -> Result<()> {
        for frame in tick.frames(self.v) {
            if let Some(seen) = self.seen {
                let rows: Vec<Colour> = tick.rows(seen)?;
                for row in &rows {
                    out.row("colour", frame.pts, row)?;
                }
            }
            out.row("colour", frame.pts, &self.colour)?;
        }
        Ok(())
    }
}

ffrwd_node::export!(ColourProbe);
