//! Where one shot ends and the next begins. Each frame's luma is reduced to a
//! small grid of cell averages and compared with the previous frame's: a mean
//! absolute difference above `threshold` is a cut, and the shot counter steps.
//! The first frame is shot 0.
//!
//! Frames pass through untouched. The row a frame carries away, `{"shot": n}`,
//! is what a downstream module reads to know a cut happened, and it is the
//! only row that leaves: rows arriving with a frame stop here.

wit_bindgen::generate!({
    path: "../../worlds/0.10.0",
    world: "window-module",
});

use std::cell::RefCell;

use exports::ffrwd::av::window_filter::{
    Format, FramePayload, Guest, InFrame, Meta, OutFrame, Processed, StreamInfo, WindowMeta,
};
use serde::{Deserialize, Serialize};

const PARAMS_SCHEMA: &str = r#"{"type":"object","properties":{"threshold":{"type":"number","exclusiveMinimum":0,"default":12.0}},"additionalProperties":false}"#;
const ROWS_SCHEMA: &str = r#"{"type":"object","properties":{"shot":{"type":"integer"}},"required":["shot"],"additionalProperties":false}"#;

/// Cells a frame is reduced to, per side. Small enough that a moving subject
/// barely moves the average, large enough that a new picture moves all of them.
const GRID: usize = 32;

/// Well clear of the frame-to-frame difference ordinary motion produces, and
/// well below what a hard cut produces.
fn default_threshold() -> f64 {
    12.0
}

#[derive(Deserialize)]
struct Params {
    #[serde(default = "default_threshold")]
    threshold: f64,
}

impl Default for Params {
    fn default() -> Self {
        Params {
            threshold: default_threshold(),
        }
    }
}

#[derive(Serialize)]
struct Row {
    shot: i64,
}

/// The pixel format the host chose at `init`, fixed for the instance's life.
#[derive(Clone, Copy, PartialEq)]
enum PixFmt {
    Yuv420p,
    Rgba,
}

struct State {
    width: usize,
    height: usize,
    pix_fmt: PixFmt,
    threshold: f64,
    /// The previous frame's cells, absent until a frame has been seen.
    previous: Option<Vec<u8>>,
    shot: i64,
    /// The current frame's cells, reused every frame.
    cells: Vec<u8>,
}

thread_local! {
    static STATE: RefCell<Option<State>> = const { RefCell::new(None) };
}

/// Parses and validates params, shared by `init` and `set_params`.
fn parse_params(params: &str) -> Result<Params, String> {
    let trimmed = params.trim();
    let parsed: Params = if trimmed.is_empty() {
        Params::default()
    } else {
        serde_json::from_str(trimmed).map_err(|e| format!("invalid params: {e}"))?
    };
    if parsed.threshold <= 0.0 || !parsed.threshold.is_finite() {
        return Err(format!(
            "threshold must be greater than 0, got {}",
            parsed.threshold
        ));
    }
    Ok(parsed)
}

/// The luma of one run of one row, totalled. yuv420p carries luma in the
/// first plane; rgba is converted with the standard weights, per pixel, so
/// each one rounds where it always did.
fn row_luma_sum(
    frame: &[u8],
    pix_fmt: PixFmt,
    width: usize,
    y: usize,
    x0: usize,
    x1: usize,
) -> u32 {
    match pix_fmt {
        PixFmt::Yuv420p => frame[y * width + x0..y * width + x1]
            .iter()
            .map(|sample| u32::from(*sample))
            .sum(),
        PixFmt::Rgba => {
            let base = (y * width + x0) * 4;
            frame[base..base + (x1 - x0) * 4]
                .as_chunks::<4>()
                .0
                .iter()
                .map(|pixel| {
                    let r = u32::from(pixel[0]);
                    let g = u32::from(pixel[1]);
                    let b = u32::from(pixel[2]);
                    (r * 299 + g * 587 + b * 114) / 1000
                })
                .sum()
        }
    }
}

/// The frame's luma as `GRID`x`GRID` cell averages, written into `cells`. A
/// frame smaller than the grid repeats pixels rather than leaving cells empty.
///
/// A cell's columns are one contiguous run of each of its rows, so it is
/// totalled a run at a time and the format is decided once per run rather
/// than once per pixel.
fn downsample(frame: &[u8], pix_fmt: PixFmt, width: usize, height: usize, cells: &mut Vec<u8>) {
    cells.clear();
    cells.resize(GRID * GRID, 0);
    let columns: Vec<(usize, usize)> = (0..GRID)
        .map(|cx| {
            let x0 = cx * width / GRID;
            (x0, ((cx + 1) * width / GRID).max(x0 + 1))
        })
        .collect();

    for cy in 0..GRID {
        let y0 = cy * height / GRID;
        let y1 = ((cy + 1) * height / GRID).max(y0 + 1);
        for (cx, (x0, x1)) in columns.iter().enumerate() {
            let mut total: u32 = 0;
            for y in y0..y1 {
                total += row_luma_sum(frame, pix_fmt, width, y, *x0, *x1);
            }
            let count = ((y1 - y0) * (x1 - x0)) as u32;
            cells[cy * GRID + cx] = (total / count.max(1)) as u8;
        }
    }
}

/// Mean absolute difference between two frames' cells, in luma steps.
fn mean_abs_diff(a: &[u8], b: &[u8]) -> f64 {
    if a.is_empty() {
        return 0.0;
    }
    let total: u32 = a
        .iter()
        .zip(b)
        .map(|(p, q)| u32::from(p.abs_diff(*q)))
        .sum();
    f64::from(total) / a.len() as f64
}

struct Shots;

impl Guest for Shots {
    fn describe() -> WindowMeta {
        WindowMeta {
            meta: Meta {
                name: "shots".to_string(),
                version: "0.1.0".to_string(),
                params_schema: PARAMS_SCHEMA.to_string(),
                rows_schema: ROWS_SCHEMA.to_string(),
                pixel_formats: vec!["yuv420p".to_string(), "rgba".to_string()],
                // Not an audio module, so it names no sample formats.
                sample_formats: vec![],
                sample_rates: vec![],
                channel_counts: vec![],
                rows_language: vec![],
            },
            window: 1,
            stride: 1,
            // The previous frame's cells carry over between calls.
            pure: false,
            one_to_one: true,
            reads_rows: false,
            // What leaves a frame is this module's shot index and nothing else.
            forwards_rows: false,
            // One stream in, which is every module here.
            inputs: 1,
        }
    }

    fn init(format: Format, _stream_info: StreamInfo, params: String) -> Result<(), String> {
        let parsed = parse_params(&params)?;
        let Format::Video(video) = format else {
            return Err("shots reads pictures, and this stream is audio".to_string());
        };
        let (width, height) = (video.width, video.height);

        let pix_fmt = match video.pix_fmt.as_str() {
            "yuv420p" => PixFmt::Yuv420p,
            "rgba" => PixFmt::Rgba,
            other => return Err(format!("unsupported pix-fmt: {other:?}")),
        };

        STATE.with(|s| {
            *s.borrow_mut() = Some(State {
                width: width as usize,
                height: height as usize,
                pix_fmt,
                threshold: parsed.threshold,
                previous: None,
                shot: 0,
                cells: Vec::with_capacity(GRID * GRID),
            });
        });
        Ok(())
    }

    fn set_params(params: String) -> Result<(), String> {
        let parsed = parse_params(&params)?;
        STATE.with(|s| {
            let mut state_ref = s.borrow_mut();
            let state = state_ref.as_mut().expect("set_params called before init");
            state.threshold = parsed.threshold;
        });
        Ok(())
    }

    fn process(frames: Vec<InFrame>, _trailing: Vec<String>, _last: bool) -> Processed {
        STATE.with(|s| {
            let mut state_ref = s.borrow_mut();
            let state = state_ref.as_mut().expect("process called before init");

            let out = frames
                .into_iter()
                .map(|frame| {
                    let mut cells = std::mem::take(&mut state.cells);
                    downsample(
                        &frame.frame,
                        state.pix_fmt,
                        state.width,
                        state.height,
                        &mut cells,
                    );
                    let cut = match &state.previous {
                        Some(previous) => mean_abs_diff(previous, &cells) > state.threshold,
                        None => false,
                    };
                    if cut {
                        state.shot += 1;
                    }
                    // These cells become the previous frame's; the ones they
                    // replace go back to being the scratch buffer.
                    state.cells = state.previous.replace(cells).unwrap_or_default();

                    let rows =
                        vec![serde_json::to_string(&Row { shot: state.shot })
                            .expect("row serializes")];
                    OutFrame {
                        pts: frame.pts,
                        frame: FramePayload::Same,
                        rows,
                    }
                })
                .collect();

            Processed {
                frames: out,
                trailing: Vec::new(),
            }
        })
    }
}

export!(Shots);

#[cfg(test)]
mod tests {
    use super::*;

    /// A `width`x`height` rgba frame filled with one grey level.
    fn flat_rgba(width: usize, height: usize, level: u8) -> Vec<u8> {
        let mut frame = vec![255u8; width * height * 4];
        for pixel in frame.chunks_mut(4) {
            pixel[0] = level;
            pixel[1] = level;
            pixel[2] = level;
        }
        frame
    }

    #[test]
    fn a_flat_frame_reduces_to_one_level_in_every_cell() {
        let mut cells = Vec::new();
        downsample(&flat_rgba(64, 64, 90), PixFmt::Rgba, 64, 64, &mut cells);
        assert_eq!(cells.len(), GRID * GRID);
        assert!(
            cells.iter().all(|c| *c == 90),
            "every cell of a flat frame holds the frame's level"
        );
    }

    #[test]
    fn a_frame_smaller_than_the_grid_still_fills_it() {
        let mut cells = Vec::new();
        downsample(&flat_rgba(8, 8, 40), PixFmt::Rgba, 8, 8, &mut cells);
        assert_eq!(cells.len(), GRID * GRID, "cells repeat rather than vanish");
    }

    #[test]
    fn the_difference_is_the_luma_step_between_two_flat_frames() {
        let mut dark = Vec::new();
        let mut light = Vec::new();
        downsample(&flat_rgba(64, 64, 10), PixFmt::Rgba, 64, 64, &mut dark);
        downsample(&flat_rgba(64, 64, 210), PixFmt::Rgba, 64, 64, &mut light);
        assert!((mean_abs_diff(&dark, &light) - 200.0).abs() < 1e-9);
        assert_eq!(mean_abs_diff(&dark, &dark), 0.0);
    }

    #[test]
    fn a_threshold_of_zero_or_less_is_refused_by_value() {
        for bad in ["0", "-1.5"] {
            let Err(err) = parse_params(&format!(r#"{{"threshold":{bad}}}"#)) else {
                panic!("a threshold of {bad} should have been refused");
            };
            assert!(err.contains("greater than 0"), "got: {err}");
        }
        assert_eq!(
            parse_params("")
                .expect("no params is the default")
                .threshold,
            12.0
        );
    }
}
