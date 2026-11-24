//! The cue rows a streaming transcriber writes beside a program where
//! someone speaks now and then, over a picture read for its timing alone.
//! Every `every` seconds of the picture an utterance starts, a quarter of a
//! second in, and lasts `lasts` seconds; its row goes out whole `late`
//! seconds after it ends, Ann and Bob speaking in turn. Between utterances
//! it writes nothing.

use ffrwd_node::{Bound, Init, Input, Node, Out, Output, Result, Shape, Tick};
use serde::{Deserialize, Serialize};

const OPENS: f64 = 0.25;

#[derive(Deserialize)]
struct Params {
    every: f64,
    lasts: f64,
    late: f64,
}

#[derive(Default, Serialize)]
struct Said {
    start_t: f64,
    end_t: f64,
    text: String,
    #[serde(rename = "final")]
    finished: bool,
    voice: String,
}

struct Pauses {
    every: f64,
    lasts: f64,
    late: f64,
    /// The next utterance to send.
    said: u32,
}

impl Node for Pauses {
    const NAME: &'static str = "pauses";
    const VERSION: &'static str = "0.1.0";
    const PARAMS_SCHEMA: &'static str = r#"{"type":"object","properties":{"every":{"type":"number","exclusiveMinimum":0,"default":12},"lasts":{"type":"number","exclusiveMinimum":0,"default":2.5},"late":{"type":"number","minimum":0,"default":1}},"additionalProperties":false}"#;
    type Params = Params;

    fn shape(_: &Params, _: &Bound) -> Result<Shape> {
        Ok(Shape::new()
            .input(Input::video("v").timing().clock())
            .output(Output::rows("said").schema::<Said>()))
    }

    fn init(params: Params, init: &Init) -> Result<Pauses> {
        init.stream("v")?;
        Ok(Pauses {
            every: params.every,
            lasts: params.lasts,
            late: params.late,
            said: 0,
        })
    }

    fn process(&mut self, tick: &Tick, out: &mut Out) -> Result<()> {
        let now = tick.seconds();
        loop {
            let start = f64::from(self.said) * self.every + OPENS;
            let end = start + self.lasts;
            if now < end + self.late {
                return Ok(());
            }
            let row = Said {
                start_t: start,
                end_t: end,
                text: format!("cue {} said at {start:.2}", self.said),
                finished: true,
                voice: if self.said.is_multiple_of(2) {
                    "Ann"
                } else {
                    "Bob"
                }
                .into(),
            };
            out.row("said", tick.pts(), &row)?;
            self.said += 1;
        }
    }
}

ffrwd_node::export!(Pauses);
