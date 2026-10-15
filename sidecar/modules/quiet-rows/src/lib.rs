//! A rows producer that writes nothing: clocked by its picture, one rows
//! output of cues, not pure, and silent on every tick. The host's progress
//! after each tick is all its reader hears of it.

use ffrwd_node::{Bound, Cue, Init, Input, NoParams, Node, Out, Output, Result, Shape, Tick};

struct QuietRows;

impl Node for QuietRows {
    const NAME: &'static str = "quiet_rows";
    const VERSION: &'static str = "0.1.0";
    type Params = NoParams;

    fn shape(_: &NoParams, _: &Bound) -> Result<Shape> {
        Ok(Shape::new()
            .input(Input::video("v").clock())
            .output(Output::rows("cues").schema::<Cue>()))
    }

    fn init(_: NoParams, init: &Init) -> Result<QuietRows> {
        init.stream("v")?;
        Ok(QuietRows)
    }

    fn process(&mut self, _: &Tick, _: &mut Out) -> Result<()> {
        Ok(())
    }
}

ffrwd_node::export!(QuietRows);
