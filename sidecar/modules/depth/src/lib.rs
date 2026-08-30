//! Monocular depth: every frame leaves as a greyscale map of how far away it
//! is, near bright and far dark.
//!
//! The graph is Depth Anything V2 (small), run through `wasi:nn`. The module
//! never opens a file - the host binds the graph to a name with
//! `-nn depth=<path>` and this module asks for that name and nothing else.
//!
//! One frame in, one out, at the same geometry and in the same pixel format,
//! so the map can be fed straight back into a module that reads the frame
//! beside it. Between the two the frame is letterboxed into the square the
//! graph takes, normalized the way the model was trained, run, min-max
//! normalized over the frame's own range, and sampled back out to the frame's
//! own size.

// `generate_all`: the world's interfaces come from two other packages -
// ffrwd:av and wasi:nn - and without it bindgen expects them to have been
// generated somewhere else.
wit_bindgen::generate!({
    path: ["../../wit", "wit"],
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

/// The name the host binds the graph to. `-nn depth=<path>`.
const MODEL: &str = "depth";

/// The square the graph is run at. Depth Anything works in patches of 14, and
/// 518 is 37 of them - the size its own preprocessing uses.
const SIDE: usize = 518;

/// The graph's own names for the tensors it takes and returns.
const INPUT_NAME: &str = "pixel_values";
const OUTPUT_NAME: &str = "predicted_depth";

/// What the model was trained on: each channel rescaled to 0..1, then
/// standardized by these.
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

const PARAMS_SCHEMA: &str = r#"{"type":"object","properties":{},"additionalProperties":false}"#;

/// The pixel format the host chose at `init`, fixed for the instance's life.
#[derive(Clone, Copy, PartialEq)]
enum PixFmt {
    Yuv420p,
    Rgba,
}

/// Where the frame sits inside the square the graph is run at, once it has
/// been scaled to fit with its shape kept.
#[derive(Clone, Copy)]
struct Letterbox {
    /// Square pixels per frame pixel.
    scale: f32,
    /// Where the scaled frame starts inside the square.
    offset_x: f32,
    offset_y: f32,
    /// How much of the square the scaled frame covers.
    width: usize,
    height: usize,
}

impl Letterbox {
    fn new(width: usize, height: usize) -> Letterbox {
        let scale = (SIDE as f32 / width as f32).min(SIDE as f32 / height as f32);
        let scaled_w = ((width as f32 * scale).round() as usize).clamp(1, SIDE);
        let scaled_h = ((height as f32 * scale).round() as usize).clamp(1, SIDE);
        Letterbox {
            scale,
            offset_x: ((SIDE - scaled_w) / 2) as f32,
            offset_y: ((SIDE - scaled_h) / 2) as f32,
            width: scaled_w,
            height: scaled_h,
        }
    }

    /// Where a frame pixel lands in the square, in square coordinates.
    fn to_square(self, x: usize, y: usize) -> (f32, f32) {
        (
            self.offset_x + (x as f32 + 0.5) * self.scale - 0.5,
            self.offset_y + (y as f32 + 0.5) * self.scale - 0.5,
        )
    }
}

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

/// The graph's output as a plain grid of floats, whichever rank it came back
/// with: [1, h, w] as this graph spells it, or [1, 1, h, w] as an export that
/// keeps the channel does.
fn depth_grid(dimensions: &[u32], data: &[u8]) -> Result<(Vec<f32>, usize, usize), String> {
    let (h, w) = match dimensions {
        [_, h, w] => (*h as usize, *w as usize),
        [_, _, h, w] => (*h as usize, *w as usize),
        other => {
            return Err(format!(
                "depth: the graph returned a depth map of {} dimension(s), expected 3 or 4",
                other.len()
            ))
        }
    };
    let (whole, _) = data.as_chunks::<4>();
    let values: Vec<f32> = whole.iter().copied().map(f32::from_le_bytes).collect();
    if values.len() < h * w {
        return Err(format!(
            "depth: the graph returned {} value(s) for a {h}x{w} depth map",
            values.len()
        ));
    }
    Ok((values, h, w))
}

/// The depth map sampled back out to the frame's own size and scaled to fill
/// a byte. The range is the frame's own: the nearest thing in it is 255 and
/// the furthest 0, so a flat scene still uses the whole scale.
fn to_grey(
    grid: &[f32],
    grid_w: usize,
    grid_h: usize,
    width: usize,
    height: usize,
    box_: Letterbox,
) -> Vec<u8> {
    // The graph's grid may not be the square it was given, so square
    // coordinates are carried onto it by ratio.
    let gx_per_square = grid_w as f32 / SIDE as f32;
    let gy_per_square = grid_h as f32 / SIDE as f32;

    // Where each frame column and row reads from on the grid. Neither depends
    // on the other axis, so both are built once and the per-pixel work is
    // what is left: two rows of the grid mixed along their length.
    let place = |g: f32, extent: usize| -> (usize, usize, f32) {
        let g = g.clamp(0.0, (extent - 1) as f32);
        let base = g.floor() as usize;
        (base, (base + 1).min(extent - 1), g - base as f32)
    };
    let columns: Vec<(usize, usize, f32)> = (0..width)
        .map(|x| place(box_.to_square(x, 0).0 * gx_per_square, grid_w))
        .collect();

    // The range is taken over what the frame covers, so the black bars never
    // stretch it.
    let mut raw = vec![0f32; width * height];
    let (mut low, mut high) = (f32::INFINITY, f32::NEG_INFINITY);
    for y in 0..height {
        let (y0, y1, ty) = place(box_.to_square(0, y).1 * gy_per_square, grid_h);
        let top_row = &grid[y0 * grid_w..(y0 + 1) * grid_w];
        let bottom_row = &grid[y1 * grid_w..(y1 + 1) * grid_w];
        let target = &mut raw[y * width..(y + 1) * width];
        for (sample, (x0, x1, tx)) in target.iter_mut().zip(&columns) {
            let top = top_row[*x0] + (top_row[*x1] - top_row[*x0]) * tx;
            let bottom = bottom_row[*x0] + (bottom_row[*x1] - bottom_row[*x0]) * tx;
            let value = top + (bottom - top) * ty;
            *sample = value;
            low = low.min(value);
            high = high.max(value);
        }
    }

    let span = high - low;
    let mut out = vec![0u8; width * height];
    if span <= 0.0 {
        // A frame the model read as one flat distance. Mid grey says so,
        // rather than a scale that divides by nothing.
        out.fill(128);
        return out;
    }
    for (sample, value) in out.iter_mut().zip(&raw) {
        *sample = (((value - low) / span) * 255.0).round().clamp(0.0, 255.0) as u8;
    }
    out
}

/// A greyscale map written as a frame of the instance's own format: the luma
/// plane with neutral chroma, or the same value in red, green and blue.
fn to_frame(grey: &[u8], pix_fmt: PixFmt, width: usize, height: usize, len: usize) -> Vec<u8> {
    let mut out = vec![0u8; len];
    match pix_fmt {
        PixFmt::Yuv420p => {
            out[..width * height].copy_from_slice(grey);
            // 128 in both chroma planes is no colour at all.
            out[width * height..].fill(128);
        }
        PixFmt::Rgba => {
            for (index, value) in grey.iter().enumerate() {
                let base = index * 4;
                out[base] = *value;
                out[base + 1] = *value;
                out[base + 2] = *value;
                out[base + 3] = 255;
            }
        }
    }
    out
}

/// One frame through the graph.
fn run(opened: &Opened, frame: &[u8], len: usize) -> Result<Vec<u8>, String> {
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
    Ok(to_frame(
        &grey,
        opened.pix_fmt,
        opened.width,
        opened.height,
        len,
    ))
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
                match run(opened, &frame.frame, frame.frame.len()) {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_wide_frame_is_letterboxed_with_bars_above_and_below() {
        let box_ = Letterbox::new(1024, 512);
        assert_eq!(box_.width, SIDE, "the wide side fills the square");
        // 512 frame pixels at 518/1024 to a pixel.
        assert_eq!(box_.height, 259, "and the other is scaled to fit");
        assert_eq!(box_.offset_x, 0.0);
        assert!(box_.offset_y > 0.0, "the bars are above and below");
    }

    #[test]
    fn a_square_frame_fills_the_square() {
        let box_ = Letterbox::new(256, 256);
        assert_eq!((box_.width, box_.height), (SIDE, SIDE));
        assert_eq!((box_.offset_x, box_.offset_y), (0.0, 0.0));
    }

    #[test]
    fn a_depth_map_comes_back_from_three_dimensions_or_four() {
        let data: Vec<u8> = (0..6).flat_map(|v| (v as f32).to_le_bytes()).collect();
        let (three, h, w) = depth_grid(&[1, 2, 3], &data).expect("rank 3");
        assert_eq!((h, w), (2, 3));
        assert_eq!(three.len(), 6);
        let (four, h, w) = depth_grid(&[1, 1, 2, 3], &data).expect("rank 4");
        assert_eq!((h, w), (2, 3));
        assert_eq!(four, three, "the same grid, however it was shaped");
    }

    #[test]
    fn a_map_with_no_range_at_all_comes_out_mid_grey() {
        let flat = vec![7.0f32; SIDE * SIDE];
        let grey = to_grey(&flat, SIDE, SIDE, 4, 4, Letterbox::new(4, 4));
        assert!(
            grey.iter().all(|v| *v == 128),
            "one flat distance is no scale to normalize onto"
        );
    }

    #[test]
    fn the_range_is_stretched_to_fill_a_byte() {
        // A ramp across the square: whatever its raw values, the frame it
        // becomes runs from 0 to 255.
        let mut grid = vec![0f32; SIDE * SIDE];
        for y in 0..SIDE {
            for x in 0..SIDE {
                grid[y * SIDE + x] = 100.0 + x as f32 * 0.5;
            }
        }
        let grey = to_grey(&grid, SIDE, SIDE, 64, 64, Letterbox::new(64, 64));
        assert_eq!(*grey.iter().min().expect("pixels"), 0);
        assert_eq!(*grey.iter().max().expect("pixels"), 255);
    }

    #[test]
    fn a_grey_map_writes_neutral_chroma_and_opaque_alpha() {
        let grey = vec![40u8; 4 * 4];
        let yuv = to_frame(&grey, PixFmt::Yuv420p, 4, 4, 4 * 4 + 2 * 2 * 2);
        assert!(
            yuv[..16].iter().all(|v| *v == 40),
            "the luma plane is the map"
        );
        assert!(
            yuv[16..].iter().all(|v| *v == 128),
            "and the chroma is neutral"
        );

        let rgba = to_frame(&grey, PixFmt::Rgba, 4, 4, 4 * 4 * 4);
        let (pixels, _) = rgba.as_chunks::<4>();
        for pixel in pixels {
            assert_eq!(
                *pixel,
                [40, 40, 40, 255],
                "equal in every channel, and opaque"
            );
        }
    }
}
