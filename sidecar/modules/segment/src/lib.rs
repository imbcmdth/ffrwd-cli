//! Instance segmentation: every frame leaves as an index map naming which
//! object owns each pixel, and one row per object beside it.
//!
//! The graph is YOLOv8n-seg, run through `wasi:nn`. The module never opens a
//! file - the host binds the graph to a name with `-nn segment=<path>` and this
//! module asks for that name and nothing else.
//!
//! # The id-as-luma encoding
//!
//! The frame that leaves is greyscale, the same geometry and pixel format the
//! instance was opened for, and the id of the detection owning a pixel goes
//! straight into that pixel's luma. 0 is background, 1 is the first detection,
//! 2 the second, and so on: id 1 is luma 1, id 3 is luma 3. The picture is
//! therefore almost black to look at - it is a lookup table, not a rendering,
//! and a downstream module reads a pixel's value as an index into the rows.
//! In yuv420p the ids are the Y plane and the chroma is neutral; in rgba the
//! id is in red, green and blue alike, opaque. Ids run in descending
//! confidence, so id 1 is the frame's surest object, and where two detections
//! claim a pixel the surer one keeps it.
//!
//! At most 255 detections fit in a byte, and a frame with more is refused
//! rather than wrapped or truncated.
//!
//! The rows carry `id`, `class`, `score` and the box as `x`, `y`, `w`, `h` in
//! the frame's own pixels, so an id read out of the map has somewhere to go.

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
use serde::{Deserialize, Serialize};
use wasi::nn::graph::{load_by_name, Graph};
use wasi::nn::inference::GraphExecutionContext;
use wasi::nn::tensor::{Tensor, TensorType};

/// The name the host binds the graph to. `-nn segment=<path>`.
const MODEL: &str = "segment";

/// The square the graph is run at. The ultralytics export is static at this
/// size.
const SIDE: usize = 640;

/// What ultralytics calls the input tensor.
const INPUT_NAME: &str = "images";

/// The host accepts a position where it accepts a name, which is what an
/// export that named its input something else is reached by.
const INPUT_INDEX: &str = "0";

/// Mask coefficients per detection, and prototype planes to spend them on.
const COEFFICIENTS: usize = 32;

/// Box channels ahead of the class scores: centre x, centre y, width, height.
const BOX_CHANNELS: usize = 4;

/// What the letterbox pads with. The model was trained and is evaluated
/// against this grey, and a marginal detection at the edge of the picture
/// turns on it.
const PAD: f32 = 114.0 / 255.0;

/// Ids are a byte of luma, and 0 is background.
const MAX_DETECTIONS: usize = 255;

const PARAMS_SCHEMA: &str = r#"{"type":"object","properties":{"confidence":{"type":"number","default":0.25},"iou":{"type":"number","default":0.45}},"additionalProperties":false}"#;

const ROWS_SCHEMA: &str = r#"{"type":"object","properties":{"id":{"type":"integer"},"class":{"type":"string"},"score":{"type":"number"},"x":{"type":"integer"},"y":{"type":"integer"},"w":{"type":"integer"},"h":{"type":"integer"}},"required":["id","class","score","x","y","w","h"],"additionalProperties":false}"#;

/// The classes the graph was trained on, in its own order.
const COCO: [&str; 80] = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
];

/// The name of a class by the graph's own index. An export trained on
/// something else is named by number rather than guessed at.
fn class_name(index: usize) -> String {
    COCO.get(index)
        .map_or_else(|| index.to_string(), |name| (*name).to_string())
}

fn default_confidence() -> f64 {
    0.25
}

fn default_iou() -> f64 {
    0.45
}

#[derive(Clone, Copy, Debug, Deserialize)]
// The schema says these two and no others, and this is what makes that true.
#[serde(deny_unknown_fields)]
struct Params {
    #[serde(default = "default_confidence")]
    confidence: f64,
    #[serde(default = "default_iou")]
    iou: f64,
}

impl Default for Params {
    fn default() -> Self {
        Params {
            confidence: default_confidence(),
            iou: default_iou(),
        }
    }
}

/// One row per detection.
#[derive(Serialize)]
struct Row {
    id: u32,
    class: String,
    score: f64,
    x: u32,
    y: u32,
    w: u32,
    h: u32,
}

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
    offset_x: usize,
    offset_y: usize,
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
            offset_x: (SIDE - scaled_w) / 2,
            offset_y: (SIDE - scaled_h) / 2,
            width: scaled_w,
            height: scaled_h,
        }
    }

    /// A square coordinate back on the frame's own horizontal axis.
    fn frame_x(self, square: f32) -> f32 {
        (square - self.offset_x as f32) / self.scale
    }

    /// A square coordinate back on the frame's own vertical axis.
    fn frame_y(self, square: f32) -> f32 {
        (square - self.offset_y as f32) / self.scale
    }
}

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

fn parse_params(params: &str) -> Result<Params, String> {
    let trimmed = params.trim();
    let parsed: Params = if trimmed.is_empty() {
        Params::default()
    } else {
        serde_json::from_str(trimmed).map_err(|e| format!("segment cannot read its params: {e}"))?
    };
    for (name, value) in [("confidence", parsed.confidence), ("iou", parsed.iou)] {
        if !value.is_finite() || !(0.0..=1.0).contains(&value) {
            return Err(format!("segment needs {name} between 0 and 1, got {value}"));
        }
    }
    Ok(parsed)
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
    format!("segment: {what}: {code} ({})", error.data())
}

/// Where each of `count` output steps reads from along one axis: the two
/// samples it falls between, and how far along it sits. Every row of a resize
/// reads the same columns, so a column map is built once rather than per row.
struct Taps {
    low: Vec<usize>,
    high: Vec<usize>,
    fraction: Vec<f32>,
}

impl Taps {
    /// `at` gives the source coordinate of each step, on a source `extent`
    /// samples long.
    fn build(count: usize, extent: usize, at: impl Fn(usize) -> f32) -> Taps {
        let mut map = Taps {
            low: Vec::with_capacity(count),
            high: Vec::with_capacity(count),
            fraction: Vec::with_capacity(count),
        };
        for step in 0..count {
            let f = at(step).clamp(0.0, (extent - 1) as f32);
            let base = f.floor() as usize;
            map.low.push(base);
            map.high.push((base + 1).min(extent - 1));
            map.fraction.push(f - base as f32);
        }
        map
    }
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

/// One object the graph found, in the square's coordinates.
#[derive(Clone)]
struct Detection {
    class: usize,
    score: f32,
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    coefficients: Vec<f32>,
}

impl Detection {
    fn area(&self) -> f32 {
        (self.x2 - self.x1).max(0.0) * (self.y2 - self.y1).max(0.0)
    }

    fn intersection_over_union(&self, other: &Detection) -> f32 {
        let w = (self.x2.min(other.x2) - self.x1.max(other.x1)).max(0.0);
        let h = (self.y2.min(other.y2) - self.y1.max(other.y1)).max(0.0);
        let overlap = w * h;
        let union = self.area() + other.area() - overlap;
        if union <= 0.0 {
            0.0
        } else {
            overlap / union
        }
    }
}

/// Every anchor whose best class scores at least `confidence`, as boxes.
///
/// `predictions` is [1, channels, anchors]: four box channels, one score per
/// class, then the mask coefficients. The best class is found a class at a
/// time rather than an anchor at a time, so each of the score channels is one
/// contiguous walk of the anchors.
fn decode(predictions: &[f32], channels: usize, anchors: usize, confidence: f32) -> Vec<Detection> {
    let classes = channels - BOX_CHANNELS - COEFFICIENTS;
    let mut best = vec![f32::NEG_INFINITY; anchors];
    let mut which = vec![0usize; anchors];

    for class in 0..classes {
        let start = (BOX_CHANNELS + class) * anchors;
        let row = &predictions[start..start + anchors];
        for ((top, index), score) in best.iter_mut().zip(which.iter_mut()).zip(row) {
            if *score > *top {
                *top = *score;
                *index = class;
            }
        }
    }

    let channel = |c: usize, anchor: usize| predictions[c * anchors + anchor];
    let first = BOX_CHANNELS + classes;
    let mut found = Vec::new();
    for (anchor, score) in best.iter().enumerate() {
        if *score < confidence {
            continue;
        }
        let (cx, cy) = (channel(0, anchor), channel(1, anchor));
        let (w, h) = (channel(2, anchor), channel(3, anchor));
        found.push(Detection {
            class: which[anchor],
            score: *score,
            x1: cx - w / 2.0,
            y1: cy - h / 2.0,
            x2: cx + w / 2.0,
            y2: cy + h / 2.0,
            coefficients: (0..COEFFICIENTS)
                .map(|k| channel(first + k, anchor))
                .collect(),
        });
    }
    found
}

/// The detections left once each one has knocked out the weaker boxes of its
/// own class that it overlaps by `iou` or more. The survivors come back in
/// descending score, which is the order their ids are handed out in.
fn suppress(mut found: Vec<Detection>, iou: f32) -> Vec<Detection> {
    found.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut kept: Vec<Detection> = Vec::new();
    for candidate in found {
        let beaten = kept.iter().any(|other| {
            other.class == candidate.class && other.intersection_over_union(&candidate) >= iou
        });
        if !beaten {
            kept.push(candidate);
        }
    }
    kept
}

/// One prototype row weighted by a detection's coefficients. Each plane
/// contributes one contiguous run, so the 32-term reduction is 32 walks of the
/// row rather than a gather per pixel.
fn weigh_row(
    prototypes: &[f32],
    plane: usize,
    width: usize,
    coefficients: &[f32],
    row: usize,
    out: &mut [f32],
) {
    out.fill(0.0);
    for (index, weight) in coefficients.iter().enumerate() {
        let start = index * plane + row * width;
        let source = &prototypes[start..start + width];
        for (slot, value) in out.iter_mut().zip(source) {
            *slot += weight * value;
        }
    }
}

/// One row of weighted prototypes resized to a run of frame columns.
fn resize_row(source: &[f32], columns: &Taps, out: &mut [f32]) {
    for (((sample, low), high), fraction) in out
        .iter_mut()
        .zip(&columns.low)
        .zip(&columns.high)
        .zip(&columns.fraction)
    {
        let (a, b) = (source[*low], source[*high]);
        *sample = a + (b - a) * fraction;
    }
}

/// A detection's box on the frame's own axes, clipped to the picture. Empty
/// when the box falls entirely in the letterbox padding.
fn frame_box(
    detection: &Detection,
    box_: Letterbox,
    width: usize,
    height: usize,
) -> (usize, usize, usize, usize) {
    let clip = |value: f32, limit: usize| value.clamp(0.0, limit as f32) as usize;
    let x0 = clip(box_.frame_x(detection.x1).floor(), width);
    let x1 = clip(box_.frame_x(detection.x2).ceil(), width);
    let y0 = clip(box_.frame_y(detection.y1).floor(), height);
    let y1 = clip(box_.frame_y(detection.y2).ceil(), height);
    (x0, y0, x1.max(x0), y1.max(y0))
}

/// The index map: one byte a pixel, naming which detection owns it.
///
/// The detections arrive in descending score and are painted in that order
/// into pixels nothing has claimed yet, so the surest one keeps a pixel two of
/// them cover.
///
/// A mask is `sigmoid(coefficients . prototypes)` thresholded at a half.
/// Sigmoid rises with its argument and a half is where it crosses zero, so the
/// weighted sum is compared against zero directly and no sigmoid is computed
/// at all. The resize is separable: a prototype row is weighted and stretched
/// to the box's columns once, and the two frame rows reading it mix the same
/// numbers, which leaves the per-pixel work one mix, one compare and one store
/// along a contiguous run.
fn index_map(
    detections: &[Detection],
    prototypes: &[f32],
    proto: (usize, usize),
    frame: (usize, usize),
    box_: Letterbox,
) -> Vec<u8> {
    if detections.len() > MAX_DETECTIONS {
        panic!(
            "segment: {} detections in one frame, and an index map names at most \
             {MAX_DETECTIONS}; raise confidence",
            detections.len()
        );
    }
    let (proto_w, proto_h) = proto;
    let (width, height) = frame;
    let plane = proto_w * proto_h;

    let mut map = vec![0u8; width * height];
    let mut weighted = vec![0f32; proto_w];
    // Where a square coordinate lands on the prototype grid, each axis by its
    // own ratio because the grid is the square scaled down.
    let per_square_x = proto_w as f32 / SIDE as f32;
    let per_square_y = proto_h as f32 / SIDE as f32;

    for (index, detection) in detections.iter().enumerate() {
        let id = (index + 1) as u8;
        let (x0, y0, x1, y1) = frame_box(detection, box_, width, height);
        if x0 == x1 || y0 == y1 {
            continue;
        }
        let run = x1 - x0;

        let columns = Taps::build(run, proto_w, |step| {
            let square = box_.offset_x as f32 + (x0 + step) as f32 * box_.scale;
            (square + 0.5) * per_square_x - 0.5
        });
        let rows = Taps::build(y1 - y0, proto_h, |step| {
            let square = box_.offset_y as f32 + (y0 + step) as f32 * box_.scale;
            (square + 0.5) * per_square_y - 0.5
        });

        let mut resized = [vec![0f32; run], vec![0f32; run]];
        let mut held: [Option<usize>; 2] = [None, None];

        for step in 0..rows.low.len() {
            for row in [rows.low[step], rows.high[step]] {
                let slot = row % 2;
                if held[slot] != Some(row) {
                    weigh_row(
                        prototypes,
                        plane,
                        proto_w,
                        &detection.coefficients,
                        row,
                        &mut weighted,
                    );
                    resize_row(&weighted, &columns, &mut resized[slot]);
                    held[slot] = Some(row);
                }
            }
            let (top, bottom) = (&resized[rows.low[step] % 2], &resized[rows.high[step] % 2]);
            let ty = rows.fraction[step];

            let at = (y0 + step) * width + x0;
            let target = &mut map[at..at + run];
            for ((slot, a), b) in target.iter_mut().zip(top).zip(bottom) {
                let logit = a + (b - a) * ty;
                if logit > 0.0 && *slot == 0 {
                    *slot = id;
                }
            }
        }
    }
    map
}

/// An index map written as a frame of the instance's own format: the luma
/// plane with neutral chroma, or the same value in red, green and blue.
fn to_frame(map: &[u8], pix_fmt: PixFmt, width: usize, height: usize, len: usize) -> Vec<u8> {
    let mut out = vec![0u8; len];
    match pix_fmt {
        PixFmt::Yuv420p => {
            out[..width * height].copy_from_slice(map);
            // 128 in both chroma planes is no colour at all.
            out[width * height..].fill(128);
        }
        PixFmt::Rgba => {
            for (index, value) in map.iter().enumerate() {
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

/// One detection's row, its box in the frame's own pixels.
fn to_row(detection: &Detection, id: u32, box_: Letterbox, width: usize, height: usize) -> String {
    let (x0, y0, x1, y1) = frame_box(detection, box_, width, height);
    serde_json::to_string(&Row {
        id,
        class: class_name(detection.class),
        // To four places: the graph's own precision is nowhere near the
        // sixteen digits an f32 widened to an f64 prints.
        score: (f64::from(detection.score) * 10_000.0).round() / 10_000.0,
        x: x0 as u32,
        y: y0 as u32,
        w: (x1 - x0) as u32,
        h: (y1 - y0) as u32,
    })
    .expect("row serializes")
}

/// A tensor's floats, whatever it arrived as bytes.
fn floats(tensor: &Tensor) -> Vec<f32> {
    let data = tensor.data();
    let (whole, _) = data.as_chunks::<4>();
    whole.iter().copied().map(f32::from_le_bytes).collect()
}

/// Which returned tensor is the predictions and which the prototypes, by
/// shape: the prototypes are the rank-4 one carrying the 32 planes, and the
/// predictions the rank-3 one carrying a channel per box coordinate, class and
/// coefficient. Names are not read, so an export that spells them differently
/// still resolves.
fn outputs(shapes: &[Vec<u32>]) -> Result<(usize, usize), String> {
    let predictions = shapes.iter().position(|dimensions| {
        matches!(dimensions.as_slice(), [_, channels, _]
            if *channels as usize > BOX_CHANNELS + COEFFICIENTS)
    });
    let prototypes = shapes.iter().position(|dimensions| {
        matches!(dimensions.as_slice(), [_, planes, _, _]
            if *planes as usize == COEFFICIENTS)
    });
    match (predictions, prototypes) {
        (Some(a), Some(b)) => Ok((a, b)),
        _ => Err(format!(
            "segment: the graph returned {shapes:?}, and this module wants \
             [1, channels, anchors] beside [1, {COEFFICIENTS}, height, width]"
        )),
    }
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

    const PROTO: usize = 160;

    fn detection(class: usize, score: f32, x1: f32, y1: f32, x2: f32, y2: f32) -> Detection {
        Detection {
            class,
            score,
            x1,
            y1,
            x2,
            y2,
            coefficients: vec![0.0; COEFFICIENTS],
        }
    }

    /// Prototypes whose first plane is all ones: a coefficient of one is then
    /// a positive logit everywhere, and a negative one a negative logit.
    fn flat_prototypes() -> Vec<f32> {
        let mut planes = vec![0f32; COEFFICIENTS * PROTO * PROTO];
        planes[..PROTO * PROTO].fill(1.0);
        planes
    }

    /// A detection whose mask is its whole box.
    fn covering(mut detection: Detection) -> Detection {
        detection.coefficients[0] = 1.0;
        detection
    }

    #[test]
    fn a_wide_frame_is_letterboxed_with_bars_above_and_below() {
        let box_ = Letterbox::new(1280, 640);
        assert_eq!(box_.width, SIDE, "the wide side fills the square");
        assert_eq!(box_.height, SIDE / 2, "and the other is scaled to fit");
        assert_eq!(box_.offset_x, 0);
        assert_eq!(box_.offset_y, SIDE / 4, "the bars are above and below");
    }

    #[test]
    fn a_tall_frame_is_letterboxed_with_bars_left_and_right() {
        // The picture the model's own examples use.
        let box_ = Letterbox::new(810, 1080);
        assert_eq!(box_.height, SIDE);
        assert_eq!(box_.width, 480);
        assert_eq!(box_.offset_x, 80, "80 columns of padding either side");
        assert_eq!(box_.offset_y, 0);
    }

    #[test]
    fn a_square_frame_fills_the_square() {
        let box_ = Letterbox::new(256, 256);
        assert_eq!((box_.width, box_.height), (SIDE, SIDE));
        assert_eq!((box_.offset_x, box_.offset_y), (0, 0));
    }

    #[test]
    fn a_square_coordinate_comes_back_to_the_frame_it_came_from() {
        let box_ = Letterbox::new(810, 1080);
        assert!(box_.frame_x(box_.offset_x as f32).abs() < 0.01);
        assert!((box_.frame_x((box_.offset_x + box_.width) as f32) - 810.0).abs() < 0.01);
        assert!(box_.frame_y(0.0).abs() < 0.01);
        assert!((box_.frame_y(SIDE as f32) - 1080.0).abs() < 0.01);
    }

    #[test]
    fn overlapping_boxes_of_one_class_leave_only_the_surest() {
        let kept = suppress(
            vec![
                detection(0, 0.6, 0.0, 0.0, 100.0, 100.0),
                detection(0, 0.9, 5.0, 5.0, 105.0, 105.0),
            ],
            0.45,
        );
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].score, 0.9, "and the survivor is the surer one");
    }

    #[test]
    fn disjoint_boxes_both_survive() {
        let kept = suppress(
            vec![
                detection(0, 0.9, 0.0, 0.0, 100.0, 100.0),
                detection(0, 0.6, 200.0, 200.0, 300.0, 300.0),
            ],
            0.45,
        );
        assert_eq!(kept.len(), 2);
        assert_eq!(
            [kept[0].score, kept[1].score],
            [0.9, 0.6],
            "in descending score, which is the order ids are handed out in"
        );
    }

    #[test]
    fn a_box_only_knocks_out_its_own_class() {
        let kept = suppress(
            vec![
                detection(0, 0.9, 0.0, 0.0, 100.0, 100.0),
                detection(5, 0.6, 0.0, 0.0, 100.0, 100.0),
            ],
            0.45,
        );
        assert_eq!(kept.len(), 2, "a person inside a bus is not the bus");
    }

    #[test]
    fn boxes_overlapping_less_than_the_threshold_both_survive() {
        let a = detection(0, 0.9, 0.0, 0.0, 100.0, 100.0);
        let b = detection(0, 0.8, 60.0, 0.0, 160.0, 100.0);
        let overlap = a.intersection_over_union(&b);
        assert!(overlap > 0.0 && overlap < 0.45, "overlapping, but not much");
        assert_eq!(suppress(vec![a, b], 0.45).len(), 2);
    }

    #[test]
    fn the_best_class_of_an_anchor_is_the_one_that_scored_highest() {
        // Three anchors and four classes.
        let (classes, anchors) = (4usize, 3usize);
        let channels = BOX_CHANNELS + classes + COEFFICIENTS;
        let mut predictions = vec![0f32; channels * anchors];
        for anchor in 0..anchors {
            predictions[2 * anchors + anchor] = 20.0;
            predictions[3 * anchors + anchor] = 20.0;
        }
        // Anchor 0 is class 2, anchor 1 is class 0, anchor 2 scores nothing.
        predictions[(BOX_CHANNELS + 2) * anchors] = 0.8;
        predictions[(BOX_CHANNELS + 1) * anchors] = 0.3;
        predictions[BOX_CHANNELS * anchors + 1] = 0.7;
        predictions[(BOX_CHANNELS + 3) * anchors + 2] = 0.1;

        let found = decode(&predictions, channels, anchors, 0.25);
        assert_eq!(found.len(), 2, "the third anchor is under the threshold");
        assert_eq!((found[0].class, found[1].class), (2, 0));
        assert_eq!(found[0].score, 0.8);
    }

    #[test]
    fn a_box_decodes_from_its_centre_and_size() {
        let (classes, anchors) = (1usize, 1usize);
        let channels = BOX_CHANNELS + classes + COEFFICIENTS;
        let mut predictions = vec![0f32; channels * anchors];
        predictions[0] = 100.0;
        predictions[1] = 200.0;
        predictions[2] = 40.0;
        predictions[3] = 60.0;
        predictions[BOX_CHANNELS] = 0.9;
        predictions[(BOX_CHANNELS + classes) * anchors] = 0.5;

        let found = decode(&predictions, channels, anchors, 0.25);
        assert_eq!(
            [found[0].x1, found[0].y1, found[0].x2, found[0].y2],
            [80.0, 170.0, 120.0, 230.0]
        );
        assert_eq!(found[0].coefficients[0], 0.5, "and the coefficients follow");
    }

    #[test]
    fn the_index_map_paints_a_detection_over_its_own_box() {
        let map = index_map(
            &[covering(detection(0, 0.9, 100.0, 100.0, 300.0, 300.0))],
            &flat_prototypes(),
            (PROTO, PROTO),
            (SIDE, SIDE),
            Letterbox::new(SIDE, SIDE),
        );
        assert_eq!(map[200 * SIDE + 200], 1, "inside the box is the detection");
        assert_eq!(map[50 * SIDE + 50], 0, "outside it is background");
        assert_eq!(map[400 * SIDE + 400], 0);
    }

    #[test]
    fn a_negative_mask_paints_nothing_even_inside_its_box() {
        let mut found = detection(0, 0.9, 100.0, 100.0, 300.0, 300.0);
        found.coefficients[0] = -1.0;
        let map = index_map(
            &[found],
            &flat_prototypes(),
            (PROTO, PROTO),
            (SIDE, SIDE),
            Letterbox::new(SIDE, SIDE),
        );
        assert!(map.iter().all(|value| *value == 0));
    }

    #[test]
    fn where_two_detections_overlap_the_surer_one_owns_the_pixel() {
        // Ids follow the order they arrive in, which is descending score.
        let map = index_map(
            &[
                covering(detection(0, 0.9, 100.0, 100.0, 300.0, 300.0)),
                covering(detection(1, 0.4, 200.0, 200.0, 400.0, 400.0)),
            ],
            &flat_prototypes(),
            (PROTO, PROTO),
            (SIDE, SIDE),
            Letterbox::new(SIDE, SIDE),
        );
        assert_eq!(map[250 * SIDE + 250], 1, "the overlap is the surer one's");
        assert_eq!(
            map[150 * SIDE + 150],
            1,
            "and each keeps what only it covers"
        );
        assert_eq!(map[350 * SIDE + 350], 2);
    }

    #[test]
    #[should_panic(expected = "segment: 256 detections in one frame")]
    fn more_detections_than_a_byte_holds_are_refused() {
        let crowd: Vec<Detection> = (0..=MAX_DETECTIONS)
            .map(|_| detection(0, 0.9, 0.0, 0.0, 10.0, 10.0))
            .collect();
        index_map(
            &crowd,
            &flat_prototypes(),
            (PROTO, PROTO),
            (SIDE, SIDE),
            Letterbox::new(SIDE, SIDE),
        );
    }

    #[test]
    fn exactly_as_many_detections_as_a_byte_holds_are_painted() {
        // Two-pixel columns side by side, so the last id is its own stripe.
        let crowd: Vec<Detection> = (0..MAX_DETECTIONS)
            .map(|index| {
                covering(detection(
                    0,
                    0.9,
                    index as f32 * 2.0,
                    0.0,
                    index as f32 * 2.0 + 2.0,
                    SIDE as f32,
                ))
            })
            .collect();
        let map = index_map(
            &crowd,
            &flat_prototypes(),
            (PROTO, PROTO),
            (SIDE, SIDE),
            Letterbox::new(SIDE, SIDE),
        );
        assert_eq!(
            map[300 * SIDE + 509],
            MAX_DETECTIONS as u8,
            "the last id still fits in the byte"
        );
    }

    #[test]
    fn a_class_is_named_by_the_graphs_own_index() {
        assert_eq!(class_name(0), "person");
        assert_eq!(class_name(5), "bus");
        assert_eq!(class_name(36), "skateboard");
        assert_eq!(class_name(79), "toothbrush");
        assert_eq!(
            class_name(80),
            "80",
            "an export trained on something else is numbered, not guessed at"
        );
    }

    #[test]
    fn a_row_carries_the_box_in_the_frames_own_pixels() {
        // A tall frame, so the square's padding is on the horizontal axis.
        let box_ = Letterbox::new(810, 1080);
        let found = detection(0, 0.8642, 80.0, 0.0, 560.0, 640.0);
        assert_eq!(
            to_row(&found, 1, box_, 810, 1080),
            r#"{"id":1,"class":"person","score":0.8642,"x":0,"y":0,"w":810,"h":1080}"#,
            "the whole picture, named in its own pixels"
        );
    }

    #[test]
    fn a_row_clips_a_box_that_runs_off_the_picture() {
        let box_ = Letterbox::new(SIDE, SIDE);
        let found = detection(2, 0.5, -50.0, -50.0, 100.0, 100.0);
        assert_eq!(
            to_row(&found, 3, box_, SIDE, SIDE),
            r#"{"id":3,"class":"car","score":0.5,"x":0,"y":0,"w":100,"h":100}"#
        );
    }

    #[test]
    fn the_two_returned_tensors_are_told_apart_by_shape() {
        assert_eq!(
            outputs(&[vec![1, 116, 8400], vec![1, 32, 160, 160]]).expect("both found"),
            (0, 1)
        );
        // The same two, in the other order.
        assert_eq!(
            outputs(&[vec![1, 32, 160, 160], vec![1, 116, 8400]]).expect("both found"),
            (1, 0)
        );
    }

    #[test]
    fn a_graph_returning_something_else_is_refused_by_name() {
        let error = outputs(&[vec![1, 25200, 85]]).expect_err("not a segmentation graph");
        assert!(error.starts_with("segment: "), "{error}");
    }

    #[test]
    fn params_default_to_the_thresholds_the_schema_publishes() {
        let parsed = parse_params("").expect("empty is the defaults");
        assert_eq!((parsed.confidence, parsed.iou), (0.25, 0.45));
        let braces = parse_params("{}").expect("and so is an empty object");
        assert_eq!((braces.confidence, braces.iou), (0.25, 0.45));
    }

    #[test]
    fn params_outside_zero_to_one_are_refused_by_name() {
        for bad in [
            r#"{"confidence":1.5}"#,
            r#"{"confidence":-0.1}"#,
            r#"{"iou":2}"#,
            r#"{"iou":-1}"#,
        ] {
            let error = parse_params(bad).expect_err(bad);
            assert!(error.starts_with("segment "), "{error}");
        }
        assert!(
            parse_params(r#"{"radius":3}"#).is_err(),
            "and so is a param this module has none of"
        );
    }

    #[test]
    fn a_map_writes_neutral_chroma_and_opaque_alpha() {
        let map = vec![3u8; 4 * 4];
        let yuv = to_frame(&map, PixFmt::Yuv420p, 4, 4, 4 * 4 + 2 * 2 * 2);
        assert!(
            yuv[..16].iter().all(|v| *v == 3),
            "the luma plane is the ids"
        );
        assert!(
            yuv[16..].iter().all(|v| *v == 128),
            "and the chroma is neutral"
        );

        let rgba = to_frame(&map, PixFmt::Rgba, 4, 4, 4 * 4 * 4);
        let (pixels, _) = rgba.as_chunks::<4>();
        for pixel in pixels {
            assert_eq!(*pixel, [3, 3, 3, 255], "equal in every channel, and opaque");
        }
    }

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
