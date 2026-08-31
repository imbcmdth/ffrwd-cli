//! Twice the frames, the same span of time: every frame is emitted at its own
//! timestamp and again halfway to its neighbour.
//!
//! # Where the second copy lands
//!
//! Halfway across the gap to the *previous* frame - which is the same gap as
//! to the next one whenever the frames are evenly spaced, and the honest
//! choice when they are not, since the module has already seen it. The first
//! frame has no previous one, so it borrows the gap to the next: that is why
//! nothing comes out until the second frame arrives, and why the first
//! frame's pixels are the only ones this module ever copies. A stream of a
//! single frame has no gap at all, and its second copy lands one tick of the
//! stream's time base after the first.
//!
//! The rows arriving with a frame ride on its first copy only, so a row is
//! not doubled along with the picture.

wit_bindgen::generate!({
    path: "../../worlds/0.10.0",
    world: "window-module",
});

use std::cell::RefCell;

use exports::ffrwd::av::window_filter::{
    Format, FramePayload, Guest, InFrame, Meta, OutFrame, Processed, StreamInfo, WindowMeta,
};

const PARAMS_SCHEMA: &str = r#"{"type":"object","properties":{},"additionalProperties":false}"#;

/// How far behind its first copy the second copy of a lone frame lands, in
/// ticks of the stream's time base. One tick is the smallest step that keeps
/// the two apart.
const LONE_FRAME_OFFSET: i64 = 1;

/// The first frame, held until the next one settles the gap it splits.
struct Held {
    pts: i64,
    frame: Vec<u8>,
    rows: Vec<String>,
}

#[derive(Default)]
struct State {
    held: Option<Held>,
    previous_pts: Option<i64>,
}

thread_local! {
    static STATE: RefCell<State> = RefCell::new(State::default());
}

/// Validates that `params` is empty or `{}`; double takes no parameters.
fn validate_params(params: &str) -> Result<(), String> {
    match params.trim() {
        "" | "{}" => Ok(()),
        other => Err(format!("double takes no params, got: {other}")),
    }
}

/// The held frame's pair, now that the `offset` between the copies is known.
/// Its pixels are copied, because the call emitting them is not the call that
/// received them.
fn emit_held(held: Held, offset: i64) -> Vec<OutFrame> {
    vec![
        OutFrame {
            pts: held.pts,
            frame: FramePayload::New(held.frame.clone()),
            rows: held.rows,
        },
        OutFrame {
            pts: held.pts + offset,
            frame: FramePayload::New(held.frame),
            rows: vec![],
        },
    ]
}

struct Double;

impl Guest for Double {
    fn describe() -> WindowMeta {
        WindowMeta {
            meta: Meta {
                name: "double".to_string(),
                version: "0.1.0".to_string(),
                params_schema: PARAMS_SCHEMA.to_string(),
                rows_schema: String::new(),
                pixel_formats: vec!["rgba".to_string(), "yuv420p".to_string()],
                // Not an audio module, so it names no sample formats.
                sample_formats: vec![],
                sample_rates: vec![],
                channel_counts: vec![],
                rows_language: vec![],
            },
            window: 1,
            stride: 1,
            // The gap a frame is split across comes from the frame before it.
            pure: false,
            // Two outputs per frame is the whole point.
            one_to_one: false,
            // Rows ride along on a frame's first copy; nothing here reads them.
            reads_rows: false,
            // A frame's own rows leave on its first copy.
            forwards_rows: true,
            // One stream in, which is every module here.
            inputs: 1,
        }
    }

    fn init(_format: Format, _stream_info: StreamInfo, params: String) -> Result<(), String> {
        validate_params(&params)?;
        STATE.with(|s| *s.borrow_mut() = State::default());
        Ok(())
    }

    fn set_params(params: String) -> Result<(), String> {
        validate_params(&params)
    }

    fn process(frames: Vec<InFrame>, _trailing: Vec<String>, last: bool) -> Processed {
        STATE.with(|s| {
            let mut state = s.borrow_mut();
            let mut out = Vec::new();

            for frame in frames {
                let Some(previous) = state.previous_pts else {
                    // Nothing to measure a gap against yet.
                    state.previous_pts = Some(frame.pts);
                    state.held = Some(Held {
                        pts: frame.pts,
                        frame: frame.frame,
                        rows: frame.rows,
                    });
                    continue;
                };

                let gap = frame.pts - previous;
                state.previous_pts = Some(frame.pts);
                if let Some(held) = state.held.take() {
                    out.extend(emit_held(held, gap / 2));
                }
                out.push(OutFrame {
                    pts: frame.pts,
                    frame: FramePayload::Same,
                    rows: frame.rows,
                });
                out.push(OutFrame {
                    pts: frame.pts + gap / 2,
                    frame: FramePayload::Same,
                    rows: vec![],
                });
            }

            // A stream of one frame ends with that frame still held.
            if last {
                if let Some(held) = state.held.take() {
                    out.extend(emit_held(held, LONE_FRAME_OFFSET));
                }
            }
            Processed {
                frames: out,
                trailing: Vec::new(),
            }
        })
    }
}

export!(Double);
