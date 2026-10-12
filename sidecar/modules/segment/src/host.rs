//! The host build: a window filter. The frame is letterboxed into the square
//! the graph takes and laid out as its planar fp32 input here, on the CPU,
//! and the map leaves in the instance's own pixel format with the rows riding
//! it.

// `generate_all`: the world's interfaces come from two other packages -
// ffrwd:av and wasi:nn - and without it bindgen expects them to have been
// generated somewhere else.
wit_bindgen::generate!({
    path: ["../../worlds/0.10.0", "wit"],
    // Fully qualified: three packages are in scope, and each has worlds.
    world: "ffrwd:segment/segment",
    generate_all,
});

use std::cell::{Cell, RefCell};

use exports::ffrwd::av::window_filter::{
    Format, FramePayload, Guest, InFrame, Meta, OutFrame, Processed, StreamInfo, WindowMeta,
};
use wasi::nn::graph::{load_by_name, Graph};
use wasi::nn::inference::GraphExecutionContext;
use wasi::nn::tensor::{Tensor, TensorType};

use crate::common::{
    decode, floats_of, index_map, outputs, parse_params, suppress, to_frame, to_row, Letterbox,
    Params, PixFmt, Taps, INPUT_INDEX, INPUT_NAME, MODEL, PAD, ROWS_SCHEMA, SIDE,
};

const PARAMS_SCHEMA: &str = r#"{"type":"object","properties":{"confidence":{"type":"number","default":0.25},"iou":{"type":"number","default":0.45}},"additionalProperties":false}"#;

/// What `init` settled, plus the graph it loaded.
struct Opened {
    width: usize,
    height: usize,
    pix_fmt: PixFmt,
    letterbox: Letterbox,
    params: Params,
    /// What the graph calls its input, settled by the first call that works.
    input_name: Cell<&'static str>,
    /// Held for the life of the instance: building it once is what keeps a
    /// provider's kernels from being chosen again per frame.
    context: GraphExecutionContext,
    /// Kept alive because the context is only valid while its graph is.
    _graph: Graph,
}

thread_local! {
    static OPENED: RefCell<Option<Opened>> = const { RefCell::new(None) };
}

struct Segment;

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
    format!("segment: {what}: {code} ({})", error.data())
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
fn resize_rgb_row(rgb: &[f32], columns: &Taps, width: usize, out: &mut [f32]) {
    let count = columns.low.len();
    for channel in 0..3 {
        let source = &rgb[channel * width..(channel + 1) * width];
        let target = &mut out[channel * count..(channel + 1) * count];
        for (((sample, low), high), fraction) in target
            .iter_mut()
            .zip(&columns.low)
            .zip(&columns.high)
            .zip(&columns.fraction)
        {
            let (a, b) = (source[*low], source[*high]);
            *sample = a + (b - a) * fraction;
        }
    }
}

/// The frame scaled into the square the graph takes and laid out as the planar
/// fp32 tensor it expects: red, green and blue in turn, each rescaled to 0..1.
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
    let mut planes = vec![PAD; plane * 3];

    let columns = Taps::build(box_.width, width, |sx| (sx as f32 + 0.5) / box_.scale - 0.5);
    let rows = Taps::build(box_.height, height, |sy| {
        (sy as f32 + 0.5) / box_.scale - 0.5
    });
    let mut rgb = vec![0f32; width * 3];
    let mut resized = [vec![0f32; box_.width * 3], vec![0f32; box_.width * 3]];
    let mut held: [Option<usize>; 2] = [None, None];

    for sy in 0..box_.height {
        for y in [rows.low[sy], rows.high[sy]] {
            let slot = y % 2;
            if held[slot] != Some(y) {
                row_to_rgb(frame, pix_fmt, width, height, y, &mut rgb);
                resize_rgb_row(&rgb, &columns, width, &mut resized[slot]);
                held[slot] = Some(y);
            }
        }
        let (top_row, bottom_row) = (&resized[rows.low[sy] % 2], &resized[rows.high[sy] % 2]);
        let ty = rows.fraction[sy];

        let at = (box_.offset_y + sy) * SIDE + box_.offset_x;
        for channel in 0..3 {
            let top = &top_row[channel * box_.width..(channel + 1) * box_.width];
            let bottom = &bottom_row[channel * box_.width..(channel + 1) * box_.width];
            let target = &mut planes[channel * plane + at..channel * plane + at + box_.width];
            for ((sample, a), b) in target.iter_mut().zip(top).zip(bottom) {
                *sample = (a + (b - a) * ty).clamp(0.0, 255.0) / 255.0;
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

/// A tensor's floats, whatever it arrived as bytes.
fn floats(tensor: &Tensor) -> Vec<f32> {
    floats_of(&tensor.data())
}

/// One frame through the graph, however the graph names its input.
fn compute(opened: &Opened, input: &[u8]) -> Result<Vec<(String, Tensor)>, String> {
    let dimensions = [1, 3, SIDE as u32, SIDE as u32];
    let name = opened.input_name.get();
    let tensor = Tensor::new(&dimensions, TensorType::Fp32, input);
    match opened.context.compute(vec![(name.to_string(), tensor)]) {
        Ok(returned) => Ok(returned),
        // An export whose input is not called what ultralytics calls it. The
        // host takes a position where it takes a name, so the retry names none,
        // and the name that worked is kept for every frame after this one.
        Err(_) if name == INPUT_NAME => {
            opened.input_name.set(INPUT_INDEX);
            let tensor = Tensor::new(&dimensions, TensorType::Fp32, input);
            opened
                .context
                .compute(vec![(INPUT_INDEX.to_string(), tensor)])
                .map_err(|e| failed("compute", &e))
        }
        Err(e) => Err(failed("compute", &e)),
    }
}

/// One frame in, an index map and its rows out.
fn run(opened: &Opened, frame: &[u8], len: usize) -> Result<(Vec<u8>, Vec<String>), String> {
    let input = to_input(
        frame,
        opened.pix_fmt,
        opened.width,
        opened.height,
        opened.letterbox,
    );
    let returned = compute(opened, &input)?;
    let tensors: Vec<Tensor> = returned.into_iter().map(|(_, tensor)| tensor).collect();
    let shapes: Vec<Vec<u32>> = tensors.iter().map(Tensor::dimensions).collect();
    let (predictions, prototypes) = outputs(&shapes)?;

    let channels = shapes[predictions][1] as usize;
    let anchors = shapes[predictions][2] as usize;
    let proto_h = shapes[prototypes][2] as usize;
    let proto_w = shapes[prototypes][3] as usize;

    let found = decode(
        &floats(&tensors[predictions]),
        channels,
        anchors,
        opened.params.confidence as f32,
    );
    let kept = suppress(found, opened.params.iou as f32);

    let map = index_map(
        &kept,
        &floats(&tensors[prototypes]),
        (proto_w, proto_h),
        (opened.width, opened.height),
        opened.letterbox,
    );
    let rows = kept
        .iter()
        .enumerate()
        .map(|(index, detection)| {
            to_row(
                detection,
                (index + 1) as u32,
                opened.letterbox,
                opened.width,
                opened.height,
            )
        })
        .collect();
    Ok((
        to_frame(&map, opened.pix_fmt, opened.width, opened.height, len),
        rows,
    ))
}

impl Guest for Segment {
    fn describe() -> WindowMeta {
        WindowMeta {
            meta: Meta {
                name: "segment".to_string(),
                version: "0.1.0".to_string(),
                params_schema: PARAMS_SCHEMA.to_string(),
                rows_schema: ROWS_SCHEMA.to_string(),
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
            // The rows leaving are this module's own detections.
            forwards_rows: false,
            // One stream in: the frame it segments.
            inputs: 1,
        }
    }

    fn init(format: Format, _stream_info: StreamInfo, params: String) -> Result<(), String> {
        let Format::Video(video) = format else {
            return Err("segment reads frames, and this stream is audio".to_string());
        };
        let pix_fmt = match video.pix_fmt.as_str() {
            "yuv420p" => PixFmt::Yuv420p,
            "rgba" => PixFmt::Rgba,
            other => return Err(format!("segment does not accept pixel format {other}")),
        };
        let parsed = parse_params(&params)?;

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
                params: parsed,
                input_name: Cell::new(INPUT_NAME),
                context,
                _graph: graph,
            });
        });
        Ok(())
    }

    fn set_params(params: String) -> Result<(), String> {
        let parsed = parse_params(&params)?;
        OPENED.with(|o| {
            if let Some(opened) = o.borrow_mut().as_mut() {
                opened.params = parsed;
            }
        });
        Ok(())
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
                    Ok((map, rows)) => out.push(OutFrame {
                        pts: frame.pts,
                        frame: FramePayload::New(map),
                        rows,
                    }),
                    // `process` has no way to say no, so a graph that failed
                    // mid-stream stops the run rather than passing a frame off
                    // as an index map.
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

export!(Segment);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_padding_of_a_letterboxed_input_is_the_grey_the_model_was_trained_with() {
        // A tall frame: the padding is the columns either side of it.
        let (width, height) = (32usize, 64usize);
        let box_ = Letterbox::new(width, height);
        let frame = vec![0u8; width * height * 4];
        let bytes = to_input(&frame, PixFmt::Rgba, width, height, box_);
        let (words, _) = bytes.as_chunks::<4>();
        let value = |channel: usize, x: usize, y: usize| {
            f32::from_le_bytes(words[channel * SIDE * SIDE + y * SIDE + x])
        };
        assert!((value(0, 0, 0) - PAD).abs() < 1e-6, "the padding is grey");
        assert!(
            value(1, SIDE / 2, SIDE / 2).abs() < 1e-6,
            "and the black frame inside it is black"
        );
    }
}
