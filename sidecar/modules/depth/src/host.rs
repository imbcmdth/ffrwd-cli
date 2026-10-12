//! The host build: a window filter, one frame in and one out at the same
//! geometry and in the same pixel format. The frame is letterboxed into the
//! square the graph takes and normalized the way the model was trained, here,
//! on the CPU; the graph is run, and its map min-max normalized over the
//! frame's own range and sampled back out to the frame's own size.

// `generate_all`: the world's interfaces come from two other packages -
// ffrwd:av and wasi:nn - and without it bindgen expects them to have been
// generated somewhere else.
wit_bindgen::generate!({
    path: ["../../worlds/0.10.0", "wit"],
    // Fully qualified: three packages are in scope, and each has worlds.
    world: "ffrwd:depth/depth",
    generate_all,
});

use std::cell::RefCell;

use exports::ffrwd::av::window_filter::{
    Format, FramePayload, Guest, InFrame, Meta, OutFrame, Processed, StreamInfo, WindowMeta,
};
use wasi::nn::graph::{load_by_name, Graph};
use wasi::nn::inference::GraphExecutionContext;
use wasi::nn::tensor::{Tensor, TensorType};

use crate::common::{
    depth_grid, to_frame, to_grey, Letterbox, PixFmt, INPUT_NAME, MEAN, MODEL, OUTPUT_NAME, SIDE,
    STD,
};

const PARAMS_SCHEMA: &str = r#"{"type":"object","properties":{},"additionalProperties":false}"#;

/// What `init` settled, plus the graph it loaded.
struct Opened {
    width: usize,
    height: usize,
    pix_fmt: PixFmt,
    letterbox: Letterbox,
    /// Held for the life of the instance: building it once is what keeps a
    /// provider's kernels from being chosen again per frame.
    context: GraphExecutionContext,
    /// Kept alive because the context is only valid while its graph is.
    _graph: Graph,
}

thread_local! {
    static OPENED: RefCell<Option<Opened>> = const { RefCell::new(None) };
}

struct Depth;

fn validate_params(params: &str) -> Result<(), String> {
    match params.trim() {
        "" | "{}" => Ok(()),
        other => Err(format!("depth takes no params, got: {other}")),
    }
}

/// The spec's spelling of an error code, so a message says what actually
/// went wrong rather than how this module happens to format things.
fn failed(what: &str, error: &wasi::nn::errors::Error) -> String {
    use wasi::nn::errors::ErrorCode;
    let code = match error.code() {
        ErrorCode::InvalidArgument => "invalid-argument",
        ErrorCode::InvalidEncoding => "invalid-encoding",
        ErrorCode::Timeout => "timeout",
        ErrorCode::RuntimeError => "runtime-error",
        ErrorCode::UnsupportedOperation => "unsupported-operation",
        ErrorCode::TooLarge => "too-large",
        ErrorCode::NotFound => "not-found",
        ErrorCode::Security => "security",
        ErrorCode::Unknown => "unknown",
    };
    format!("depth: {what}: {code} ({})", error.data())
}

/// Where one square pixel reads from along an axis: the two frame samples it
/// falls between, and how far along it sits. Every square row reads the same
/// columns, so the column map is built once rather than per row.
struct Taps {
    low: Vec<usize>,
    high: Vec<usize>,
    fraction: Vec<f32>,
}

/// The map from `count` square steps back onto `source` frame samples.
fn taps(count: usize, source: usize, scale: f32) -> Taps {
    let mut map = Taps {
        low: Vec::with_capacity(count),
        high: Vec::with_capacity(count),
        fraction: Vec::with_capacity(count),
    };
    for step in 0..count {
        let f = ((step as f32 + 0.5) / scale - 0.5).clamp(0.0, (source - 1) as f32);
        let base = f.floor() as usize;
        map.low.push(base);
        map.high.push((base + 1).min(source - 1));
        map.fraction.push(f - base as f32);
    }
    map
}

/// One frame row as red, green and blue, a channel at a time so each is
/// contiguous.
fn row_to_rgb(
    frame: &[u8],
    pix_fmt: PixFmt,
    width: usize,
    height: usize,
    y: usize,
    out: &mut [f32],
) {
    let (red, rest) = out.split_at_mut(width);
    let (green, blue) = rest.split_at_mut(width);
    match pix_fmt {
        PixFmt::Rgba => {
            for (x, pixel) in frame[y * width * 4..(y + 1) * width * 4]
                .as_chunks::<4>()
                .0
                .iter()
                .enumerate()
            {
                red[x] = f32::from(pixel[0]);
                green[x] = f32::from(pixel[1]);
                blue[x] = f32::from(pixel[2]);
            }
        }
        PixFmt::Yuv420p => {
            let pixels = width * height;
            let (cw, ch) = (width.div_ceil(2), height.div_ceil(2));
            let chroma = cw * ch;
            let luma = &frame[y * width..(y + 1) * width];
            let crow = (y / 2).min(ch - 1) * cw;
            for x in 0..width {
                let l = f32::from(luma[x]);
                let ci = crow + (x / 2).min(cw - 1);
                let u = f32::from(frame[pixels + ci]) - 128.0;
                let v = f32::from(frame[pixels + chroma + ci]) - 128.0;
                // The usual BT.601 inverse, in full range: the frames a module
                // is handed are what the host decoded, not studio-swing video.
                red[x] = l + 1.402 * v;
                green[x] = l - 0.344_136 * u - 0.714_136 * v;
                blue[x] = l + 1.772 * u;
            }
        }
    }
}

/// One frame row resized to the square's columns, channel by channel.
fn resize_row(rgb: &[f32], columns: &Taps, width: usize, out: &mut [f32]) {
    let count = columns.low.len();
    for channel in 0..3 {
        let source = &rgb[channel * width..(channel + 1) * width];
        let target = &mut out[channel * count..(channel + 1) * count];
        for (sx, sample) in target.iter_mut().enumerate() {
            let a = source[columns.low[sx]];
            let b = source[columns.high[sx]];
            *sample = a + (b - a) * columns.fraction[sx];
        }
    }
}

/// The frame scaled into the square the graph takes, normalized and laid out
/// as the planar fp32 tensor it expects. Everything outside the letterbox is
/// black, which the depth of the frame itself never reads back.
///
/// The resize is separable, so each frame row is turned into the square's
/// columns once and the two square rows that read it mix the same numbers.
/// Slots go by parity, and a square row mixes frame rows `y` and `y + 1`,
/// which never share one.
fn to_input(
    frame: &[u8],
    pix_fmt: PixFmt,
    width: usize,
    height: usize,
    box_: Letterbox,
) -> Vec<u8> {
    let plane = SIDE * SIDE;
    // Black, standardized, is where the padding sits.
    let mut planes = vec![0f32; plane * 3];
    for channel in 0..3 {
        let pad = (0.0 - MEAN[channel]) / STD[channel];
        planes[channel * plane..(channel + 1) * plane].fill(pad);
    }

    let columns = taps(box_.width, width, box_.scale);
    let rows = taps(box_.height, height, box_.scale);
    let mut rgb = vec![0f32; width * 3];
    let mut resized = [vec![0f32; box_.width * 3], vec![0f32; box_.width * 3]];
    let mut held: [Option<usize>; 2] = [None, None];

    for sy in 0..box_.height {
        for y in [rows.low[sy], rows.high[sy]] {
            let slot = y % 2;
            if held[slot] != Some(y) {
                row_to_rgb(frame, pix_fmt, width, height, y, &mut rgb);
                resize_row(&rgb, &columns, width, &mut resized[slot]);
                held[slot] = Some(y);
            }
        }
        let (top_row, bottom_row) = (&resized[rows.low[sy] % 2], &resized[rows.high[sy] % 2]);
        let ty = rows.fraction[sy];

        let at = (box_.offset_y as usize + sy) * SIDE + box_.offset_x as usize;
        for channel in 0..3 {
            let (mean, std) = (MEAN[channel], STD[channel]);
            let top = &top_row[channel * box_.width..(channel + 1) * box_.width];
            let bottom = &bottom_row[channel * box_.width..(channel + 1) * box_.width];
            let target = &mut planes[channel * plane + at..channel * plane + at + box_.width];
            for ((sample, a), b) in target.iter_mut().zip(top).zip(bottom) {
                let value = (a + (b - a) * ty).clamp(0.0, 255.0) / 255.0;
                *sample = (value - mean) / std;
            }
        }
    }

    let mut bytes = vec![0u8; planes.len() * 4];
    let (words, _) = bytes.as_chunks_mut::<4>();
    for (word, value) in words.iter_mut().zip(&planes) {
        *word = value.to_le_bytes();
    }
    bytes
}

/// One frame through the graph.
fn run(opened: &Opened, frame: &[u8]) -> Result<Vec<u8>, String> {
    let input = to_input(
        frame,
        opened.pix_fmt,
        opened.width,
        opened.height,
        opened.letterbox,
    );
    let tensor = Tensor::new(&[1, 3, SIDE as u32, SIDE as u32], TensorType::Fp32, &input);
    let outputs = opened
        .context
        .compute(vec![(INPUT_NAME.to_string(), tensor)])
        .map_err(|e| failed("compute", &e))?;

    let out = outputs
        .into_iter()
        .find(|(name, _)| name == OUTPUT_NAME)
        .map(|(_, tensor)| tensor)
        .ok_or_else(|| format!("depth: the graph returned no tensor named {OUTPUT_NAME}"))?;
    let (grid, grid_h, grid_w) = depth_grid(&out.dimensions(), &out.data())?;
    let grey = to_grey(
        &grid,
        grid_w,
        grid_h,
        opened.width,
        opened.height,
        opened.letterbox,
    );
    Ok(to_frame(&grey, opened.pix_fmt, opened.width, opened.height))
}

impl Guest for Depth {
    fn describe() -> WindowMeta {
        WindowMeta {
            meta: Meta {
                name: "depth".to_string(),
                version: "0.1.0".to_string(),
                params_schema: PARAMS_SCHEMA.to_string(),
                rows_schema: String::new(),
                pixel_formats: vec!["yuv420p".to_string(), "rgba".to_string()],
                sample_formats: vec![],
                sample_rates: vec![],
                channel_counts: vec![],
                rows_language: vec![],
            },
            window: 1,
            stride: 1,
            pure: true,
            one_to_one: true,
            reads_rows: false,
            forwards_rows: true,
            // One stream in: the frame it measures.
            inputs: 1,
        }
    }

    fn init(format: Format, _stream_info: StreamInfo, params: String) -> Result<(), String> {
        let Format::Video(video) = format else {
            return Err("depth reads frames, and this stream is audio".to_string());
        };
        let pix_fmt = match video.pix_fmt.as_str() {
            "yuv420p" => PixFmt::Yuv420p,
            "rgba" => PixFmt::Rgba,
            other => return Err(format!("depth does not accept pixel format {other}")),
        };
        validate_params(&params)?;

        // The graph is loaded once per instance, and the session built once:
        // the first frame is what a provider picks its kernels on, and every
        // frame after it reuses them.
        let graph =
            load_by_name(MODEL).map_err(|e| failed(&format!("load-by-name({MODEL:?})"), &e))?;
        let context = graph
            .init_execution_context()
            .map_err(|e| failed("init-execution-context", &e))?;

        OPENED.with(|o| {
            *o.borrow_mut() = Some(Opened {
                width: video.width as usize,
                height: video.height as usize,
                pix_fmt,
                letterbox: Letterbox::new(video.width as usize, video.height as usize),
                context,
                _graph: graph,
            });
        });
        Ok(())
    }

    fn set_params(params: String) -> Result<(), String> {
        validate_params(&params)
    }

    fn process(frames: Vec<InFrame>, _trailing: Vec<String>, _last: bool) -> Processed {
        // The final call carries nothing: window and stride are 1, so no frame
        // is ever left over.
        let mut out = Vec::with_capacity(frames.len());
        OPENED.with(|opened| {
            let borrowed = opened.borrow();
            let opened = borrowed
                .as_ref()
                .expect("init loads the graph before any frame arrives");
            for frame in &frames {
                match run(opened, &frame.frame) {
                    Ok(map) => out.push(OutFrame {
                        pts: frame.pts,
                        frame: FramePayload::New(map),
                        rows: frame.rows.clone(),
                    }),
                    // `process` has no way to say no, so a graph that failed
                    // mid-stream stops the run rather than passing a frame off
                    // as a depth map.
                    Err(message) => panic!("{message}"),
                }
            }
        });
        Processed {
            frames: out,
            trailing: vec![],
        }
    }
}

export!(Depth);
