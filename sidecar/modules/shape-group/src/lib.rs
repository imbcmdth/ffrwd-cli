//! Many pictures and many sounds held on one group, as a compositor takes
//! each lateral's picture and sound: `v` and `s` take a stream per port
//! `ports` lists, the picture and the sound on one port arriving on one
//! connection. `cues` takes rows on the group's first connection.
//!
//! It passes the clock `c` through. `feeds` has a row whenever what the
//! host says of one stream's feed changes: its input (`member`), its place
//! among the input's streams, whether a feed is live and, while one is,
//! its start. `cues` has a row per message `cues` is handed, on the tick it is.

use ffrwd_node::{Anchor, Bound, Init, Input, Node, Out, Output, Result, Shape, Tick};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Params {
    lead: f64,
    timeout: f64,
}

#[derive(Serialize)]
struct FeedRow<'a> {
    pts: i64,
    member: &'a str,
    k: usize,
    live: bool,
    at_pts: i64,
    first_pts: i64,
}

#[derive(Serialize)]
struct CueRow {
    pts: i64,
    cue_pts: i64,
    text: String,
}

/// One stream of `v` or `s`: its input, its place, its id and the start
/// said last.
struct Held {
    input: &'static str,
    k: usize,
    id: u32,
    said: Option<(i64, i64)>,
}

struct Group {
    c: u32,
    held: Vec<Held>,
    cues: Option<u32>,
}

impl Node for Group {
    const NAME: &'static str = "shape_group";
    const VERSION: &'static str = "0.1.0";
    const PARAMS_SCHEMA: &'static str = r#"{"type":"object","properties":{"ports":{"type":"array","items":{"type":"integer","minimum":1,"maximum":65535}},"lead":{"type":"number","minimum":0,"default":0.3},"timeout":{"type":"number","exclusiveMinimum":0,"default":1}},"additionalProperties":false}"#;
    type Params = Params;

    fn shape(params: &Params, _: &Bound) -> Result<Shape> {
        let held = |input: Input| {
            input
                .optional()
                .many()
                .hold()
                .anchor(Anchor::FirstFrame)
                .lead(params.lead)
                .timeout(params.timeout)
                .group("lateral")
                .port_param("ports")
        };
        Ok(Shape::new()
            .input(Input::video("c").clock().pixel_formats(&["rgba"]))
            .input(Input::audio("a").optional().sample_formats(&["f32"]))
            .input(held(Input::video("v")).like("c").pixel_formats(&["rgba"]))
            .input(held(Input::audio("s")).like("a").sample_formats(&["f32"]))
            .input(Input::rows("cues").optional().group("lateral").per_frame())
            .output(Output::like("c"))
            .output(Output::rows("feeds").schema_json(
                r#"{"type":"object","properties":{"pts":{"type":"integer"},"member":{"type":"string"},"k":{"type":"integer"},"live":{"type":"boolean"},"at_pts":{"type":"integer"},"first_pts":{"type":"integer"}}}"#,
            ))
            .output(Output::rows("cues").schema_json(
                r#"{"type":"object","properties":{"pts":{"type":"integer"},"cue_pts":{"type":"integer"},"text":{"type":"string"}}}"#,
            ))
            .one_to_one())
    }

    fn init(_: Params, init: &Init) -> Result<Group> {
        let mut held = Vec::new();
        for input in ["v", "s"] {
            for (k, stream) in init.streams(input).iter().enumerate() {
                held.push(Held {
                    input,
                    k,
                    id: stream.id,
                    said: None,
                });
            }
        }
        Ok(Group {
            c: init.stream("c")?.id,
            held,
            cues: init.optional("cues").map(|b| b.id),
        })
    }

    fn process(&mut self, tick: &Tick, out: &mut Out) -> Result<()> {
        let pts = tick.pts();
        let Some(frame) = tick.frame(self.c) else {
            return Ok(());
        };
        out.pass("c", self.c, &frame)?;
        for held in &mut self.held {
            let start = tick
                .feed(held.id)
                .map(|feed| (feed.start.at, feed.start.first_pts));
            if start == held.said {
                continue;
            }
            held.said = start;
            let (at, first_pts) = start.unwrap_or_default();
            out.row(
                "feeds",
                pts,
                &FeedRow {
                    pts,
                    member: held.input,
                    k: held.k,
                    live: start.is_some(),
                    at_pts: at,
                    first_pts,
                },
            )?;
        }
        for message in self.cues.map(|id| tick.messages(id)).unwrap_or_default() {
            out.row(
                "cues",
                pts,
                &CueRow {
                    pts,
                    cue_pts: message.pts,
                    text: String::from_utf8_lossy(&message.data).into_owned(),
                },
            )?;
        }
        Ok(())
    }
}

ffrwd_node::export!(Group);
