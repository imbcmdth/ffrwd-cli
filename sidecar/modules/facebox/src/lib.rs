//! Face detection, steadied. The detector runs on every frame; what leaves is
//! the union of everything it has found over the last `smooth-window` seconds,
//! so a box that jitters comes out as one steady rectangle and a frame the
//! detector misses is covered by its neighbours.
//!
//! A `{"shot": n}` row arriving with a frame - from a module that finds cuts -
//! empties that memory when the index changes, since a box from the previous
//! shot names nothing in this one. Those rows are read, not passed on: what
//! leaves a frame is this module's rectangles and nothing else.

wit_bindgen::generate!({
    path: "../../wit",
    world: "window-module",
});

mod smooth;

use std::cell::RefCell;
use std::io::Cursor;

use ffrwd::av::types::Rational;

use exports::ffrwd::av::window_filter::{
    Format, FramePayload, Guest, InFrame, Meta, OutFrame, Processed, StreamInfo, WindowMeta,
};
use serde::{Deserialize, Serialize};
use smooth::{Rect, Smoother};

static MODEL: &[u8] = include_bytes!(env!("FACEBOX_MODEL"));

const PARAMS_SCHEMA: &str = r#"{"type":"object","properties":{"min-face-size":{"type":"integer","minimum":20,"default":40},"score-threshold":{"type":"number","exclusiveMinimum":0,"default":2.0},"smooth-window":{"type":"number","minimum":0,"default":0.5}},"additionalProperties":false}"#;
const ROWS_SCHEMA: &str = r#"{"type":"object","properties":{"x":{"type":"integer"},"y":{"type":"integer"},"w":{"type":"integer"},"h":{"type":"integer"}},"required":["x","y","w","h"],"additionalProperties":false}"#;

fn default_min_face_size() -> u32 {
    40
}

fn default_score_threshold() -> f64 {
    2.0
}

fn default_smooth_window() -> f64 {
    0.5
}

#[derive(Deserialize)]
#[serde(rename_all = "kebab-case")]
struct Params {
    #[serde(default = "default_min_face_size")]
    min_face_size: u32,
    #[serde(default = "default_score_threshold")]
    score_threshold: f64,
    #[serde(default = "default_smooth_window")]
    smooth_window: f64,
}

impl Default for Params {
    fn default() -> Self {
        Params {
            min_face_size: default_min_face_size(),
            score_threshold: default_score_threshold(),
            smooth_window: default_smooth_window(),
        }
    }
}

#[derive(Serialize)]
struct Row {
    x: i32,
    y: i32,
    w: u32,
    h: u32,
}

/// A row naming which shot its frame belongs to. A row shaped any other way -
/// a rectangle, say - is not one of these and is ignored.
#[derive(Deserialize)]
struct ShotRow {
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
    /// Seconds one timestamp tick is worth, from the stream's time base.
    tick: f64,
    detector: Box<dyn rustface::Detector>,
    /// Grayscale conversion buffer for the rgba path, reused every frame.
    gray: Vec<u8>,
    smoother: Smoother,
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
    if parsed.min_face_size < 20 {
        return Err(format!(
            "min-face-size must be at least 20, got {}",
            parsed.min_face_size
        ));
    }
    if parsed.score_threshold <= 0.0 {
        return Err(format!(
            "score-threshold must be greater than 0, got {}",
            parsed.score_threshold
        ));
    }
    if parsed.smooth_window < 0.0 || !parsed.smooth_window.is_finite() {
        return Err(format!(
            "smooth-window must be at least 0, got {}",
            parsed.smooth_window
        ));
    }
    Ok(parsed)
}

/// Seconds one timestamp tick is worth, which is what ages the box history. A
/// windowed module is handed timestamps rather than seconds, and the host
/// always supplies the time base they are counted in.
fn seconds_per_tick(time_base: Rational) -> f64 {
    f64::from(time_base.num) / f64::from(time_base.den)
}

/// Converts packed RGBA into 8-bit grayscale using standard luma weights.
fn to_grayscale(frame: &[u8], gray: &mut [u8]) {
    for (i, g) in gray.iter_mut().enumerate() {
        let base = i * 4;
        let r = frame[base] as u32;
        let gg = frame[base + 1] as u32;
        let b = frame[base + 2] as u32;
        *g = ((r * 299 + gg * 587 + b * 114) / 1000) as u8;
    }
}

/// Clamps a detected box to the frame bounds.
fn clamp_box(b: Rect, width: usize, height: usize) -> Rect {
    let width = width as i32;
    let height = height as i32;
    let x0 = b.x.clamp(0, width);
    let y0 = b.y.clamp(0, height);
    let x1 = (b.x + b.w as i32).clamp(0, width);
    let y1 = (b.y + b.h as i32).clamp(0, height);
    Rect {
        x: x0,
        y: y0,
        w: (x1 - x0) as u32,
        h: (y1 - y0) as u32,
    }
}

/// The shot index a frame's rows name, if any of them does. The last one wins.
fn shot_of(rows: &[String]) -> Option<i64> {
    rows.iter()
        .filter_map(|row| serde_json::from_str::<ShotRow>(row).ok())
        .map(|row| row.shot)
        .next_back()
}

/// What the detector finds in one frame, clamped to it.
fn detect(state: &mut State, frame: &[u8]) -> Vec<Rect> {
    let (width, height) = (state.width, state.height);
    let faces = match state.pix_fmt {
        PixFmt::Yuv420p => {
            let luma = &frame[..width * height];
            let image = rustface::ImageData::new(luma, width as u32, height as u32);
            state.detector.detect(&image)
        }
        PixFmt::Rgba => {
            to_grayscale(frame, &mut state.gray);
            let image = rustface::ImageData::new(&state.gray, width as u32, height as u32);
            state.detector.detect(&image)
        }
    };

    faces
        .iter()
        .map(|face| {
            let bbox = face.bbox();
            clamp_box(
                Rect {
                    x: bbox.x(),
                    y: bbox.y(),
                    w: bbox.width(),
                    h: bbox.height(),
                },
                width,
                height,
            )
        })
        .collect()
}

struct Facebox;

impl Guest for Facebox {
    fn describe() -> WindowMeta {
        WindowMeta {
            meta: Meta {
                name: "facebox".to_string(),
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
            // The box history carries over between calls.
            pure: false,
            one_to_one: true,
            // A shot index arriving with a frame empties that history.
            reads_rows: true,
            // Those rows are read, not passed on: what leaves a frame is this
            // module's rectangles and nothing else.
            forwards_rows: false,
            // One stream in, which is every module here.
            inputs: 1,
        }
    }

    fn init(format: Format, stream_info: StreamInfo, params: String) -> Result<(), String> {
        let parsed = parse_params(&params)?;
        let tick = seconds_per_tick(stream_info.time_base);
        let Format::Video(video) = format else {
            return Err("facebox reads pictures, and this stream is audio".to_string());
        };
        let (width, height) = (video.width, video.height);

        let pix_fmt = match video.pix_fmt.as_str() {
            "yuv420p" => PixFmt::Yuv420p,
            "rgba" => PixFmt::Rgba,
            other => return Err(format!("unsupported pix-fmt: {other:?}")),
        };

        let model = rustface::read_model(Cursor::new(MODEL))
            .map_err(|e| format!("failed to read model: {e}"))?;
        let mut detector = rustface::create_detector_with_model(model);
        detector.set_min_face_size(parsed.min_face_size);
        detector.set_score_thresh(parsed.score_threshold);

        let w = width as usize;
        let h = height as usize;
        STATE.with(|s| {
            *s.borrow_mut() = Some(State {
                width: w,
                height: h,
                pix_fmt,
                tick,
                detector,
                gray: vec![0u8; w * h],
                smoother: Smoother::new(parsed.smooth_window),
            });
        });
        Ok(())
    }

    fn set_params(params: String) -> Result<(), String> {
        let parsed = parse_params(&params)?;
        STATE.with(|s| {
            let mut state_ref = s.borrow_mut();
            let state = state_ref.as_mut().expect("set_params called before init");
            state.detector.set_min_face_size(parsed.min_face_size);
            state.detector.set_score_thresh(parsed.score_threshold);
            state.smoother.set_window(parsed.smooth_window);
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
                    let shot = shot_of(&frame.rows);
                    let detections = detect(state, &frame.frame);
                    let at = frame.pts as f64 * state.tick;

                    let rows = state
                        .smoother
                        .step(at, shot, &detections)
                        .into_iter()
                        .map(|rect| {
                            serde_json::to_string(&Row {
                                x: rect.x,
                                y: rect.y,
                                w: rect.w,
                                h: rect.h,
                            })
                            .expect("row serializes")
                        })
                        .collect();

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

export!(Facebox);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_time_base_field_becomes_seconds_per_tick() {
        assert!((seconds_per_tick(Rational { num: 1, den: 65536 }) * 65536.0 - 1.0).abs() < 1e-12);
        assert!((seconds_per_tick(Rational { num: 1, den: 25 }) * 25.0 - 1.0).abs() < 1e-12);
        assert!(
            (seconds_per_tick(Rational {
                num: 1001,
                den: 30000
            }) * 30000.0
                - 1001.0)
                .abs()
                < 1e-9
        );
    }

    #[test]
    fn a_shot_row_is_read_and_a_rectangle_row_is_not() {
        assert_eq!(shot_of(&[r#"{"shot":3}"#.to_string()]), Some(3));
        assert_eq!(shot_of(&[r#"{"x":1,"y":2,"w":3,"h":4}"#.to_string()]), None);
        assert_eq!(shot_of(&[]), None);
        assert_eq!(
            shot_of(&[r#"{"shot":1}"#.to_string(), r#"{"shot":2}"#.to_string()]),
            Some(2),
            "the last shot row a frame carries is the one that counts"
        );
    }

    #[test]
    fn the_smoothing_window_defaults_to_half_a_second_and_refuses_a_negative_one() {
        assert_eq!(
            parse_params("")
                .expect("no params is the default")
                .smooth_window,
            0.5
        );
        let Err(err) = parse_params(r#"{"smooth-window":-1}"#) else {
            panic!("a negative smoothing window should have been refused");
        };
        assert!(err.contains("smooth-window"), "got: {err}");
    }

    #[test]
    fn a_box_is_cut_to_the_frame_it_was_found_in() {
        let clamped = clamp_box(
            Rect {
                x: -4,
                y: -4,
                w: 20,
                h: 20,
            },
            8,
            8,
        );
        assert_eq!(
            clamped,
            Rect {
                x: 0,
                y: 0,
                w: 8,
                h: 8
            }
        );
    }
}
