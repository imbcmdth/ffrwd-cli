wit_bindgen::generate!({
    path: "../../wit",
    world: "meta-module",
});

use std::cell::RefCell;

use exports::ffrwd::av::filter::{FrameInfo, Guest, Meta, Outcome, Output, StreamInfo};
use exports::ffrwd::av::meta_filter::Guest as MetaGuest;
use serde::Deserialize;

const PARAMS_SCHEMA: &str = r#"{"type":"object","properties":{"radius":{"type":"integer","minimum":1,"maximum":256,"default":8},"passes":{"type":"integer","minimum":1,"maximum":8,"default":3}},"additionalProperties":false}"#;

fn default_radius() -> u32 {
    8
}

fn default_passes() -> u32 {
    3
}

#[derive(Deserialize)]
struct Params {
    #[serde(default = "default_radius")]
    radius: u32,
    #[serde(default = "default_passes")]
    passes: u32,
}

impl Default for Params {
    fn default() -> Self {
        Params {
            radius: default_radius(),
            passes: default_passes(),
        }
    }
}

/// One rectangle to blur, as an upstream detector reports it. Extra keys are
/// ignored: a row carrying more than the four coordinates is still a
/// rectangle.
#[derive(Deserialize)]
struct Rect {
    x: i64,
    y: i64,
    w: i64,
    h: i64,
}

/// A rectangle cut down to the frame. Empty rectangles never reach here.
#[derive(Clone, Copy)]
struct Region {
    x: usize,
    y: usize,
    w: usize,
    h: usize,
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
    blur: Blur,
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
    if !(1..=256).contains(&parsed.radius) {
        return Err(format!("radius must be 1..=256, got {}", parsed.radius));
    }
    if !(1..=8).contains(&parsed.passes) {
        return Err(format!("passes must be 1..=8, got {}", parsed.passes));
    }
    Ok(parsed)
}

/// The rectangles in `rows`, cut to a `width`x`height` frame. A row that is
/// not a rectangle - one an upstream module emitted for something else - is
/// skipped rather than refused.
fn regions(rows: &[String], width: usize, height: usize) -> Vec<Region> {
    rows.iter()
        .filter_map(|row| serde_json::from_str::<Rect>(row).ok())
        .filter_map(|rect| clamp(rect, width, height))
        .collect()
}

/// A rectangle inside the frame, or None when it falls outside it entirely.
fn clamp(rect: Rect, width: usize, height: usize) -> Option<Region> {
    let w = width as i64;
    let h = height as i64;
    let x0 = rect.x.clamp(0, w);
    let y0 = rect.y.clamp(0, h);
    let x1 = rect.x.saturating_add(rect.w).clamp(0, w);
    let y1 = rect.y.saturating_add(rect.h).clamp(0, h);
    if x1 <= x0 || y1 <= y0 {
        return None;
    }
    Some(Region {
        x: x0 as usize,
        y: y0 as usize,
        w: (x1 - x0) as usize,
        h: (y1 - y0) as usize,
    })
}

/// The same region in a half-resolution chroma plane. Rounds outward, so a
/// blurred luma rectangle never keeps sharp colour at its edge.
fn chroma_region(r: Region, width: usize, height: usize) -> Region {
    let x = r.x / 2;
    let y = r.y / 2;
    let x1 = ((r.x + r.w).div_ceil(2)).min(width);
    let y1 = ((r.y + r.h).div_ceil(2)).min(height);
    Region {
        x,
        y,
        w: x1.saturating_sub(x),
        h: y1.saturating_sub(y),
    }
}

/// Exact `sum / window` as a multiply and a shift, so a blur pass does not
/// divide per pixel. The radius is capped at 256, so `255 * window` always
/// fits the shift and a single correction step lands on the true quotient.
#[derive(Clone, Copy)]
struct DivideBy {
    window: u32,
    /// `floor(2^DIVIDE_SHIFT / window)`. Rounded down, so the quotient it
    /// gives is the true one or the one below it, never above.
    magic: u32,
}

/// Where `DivideBy` puts its binary point: high enough that rounding the
/// magic down costs at most one, low enough that `sum * magic` stays inside
/// 32 bits.
const DIVIDE_SHIFT: u32 = 24;

impl DivideBy {
    fn new(radius: usize) -> DivideBy {
        let window = (2 * radius + 1) as u32;
        DivideBy {
            window,
            magic: (1u32 << DIVIDE_SHIFT) / window,
        }
    }

    #[inline(always)]
    fn apply(self, sum: u32) -> u32 {
        let quotient = (sum.wrapping_mul(self.magic)) >> DIVIDE_SHIFT;
        quotient + u32::from(sum - quotient * self.window >= self.window)
    }
}

/// A row of window sums written out as the samples they average to. No lane
/// depends on another, which is what lets the whole row go at once.
fn emit_row(sums: &[u32], div: DivideBy, dst: &mut [u8]) {
    for (sample, sum) in dst.iter_mut().zip(sums) {
        *sample = div.apply(*sum) as u8;
    }
}

/// One row's worth of box blur, window `2*radius+1`, edges clamped to the
/// region so nothing outside it is read.
///
/// The row's running total goes in first - `totals[i]` is the sum of its
/// first `i` samples - so each window is one subtraction and no sum depends
/// on the one before it.
fn blur_rows(
    src: &[u8],
    dst: &mut [u8],
    w: usize,
    h: usize,
    radius: usize,
    totals: &mut [u32],
    sums: &mut [u32],
) {
    let div = DivideBy::new(radius);
    for y in 0..h {
        let row = &src[y * w..y * w + w];
        totals[0] = 0;
        let mut running = 0u32;
        for (i, sample) in row.iter().enumerate() {
            running += u32::from(*sample);
            totals[i + 1] = running;
        }

        let first = u32::from(row[0]);
        let last = u32::from(row[w - 1]);
        let clamped = |x: usize| -> u32 {
            totals[(x + radius + 1).min(w)] - totals[x.saturating_sub(radius)]
                + (radius - x.min(radius)) as u32 * first
                + ((x + radius + 1).saturating_sub(w)) as u32 * last
        };

        // Only the ends need clamping; between them the window sits inside
        // the row and every sum is one subtraction of two contiguous walks.
        let inside_start = radius.min(w);
        let inside_end = w.saturating_sub(radius).max(inside_start);
        for (x, sum) in sums[..inside_start].iter_mut().enumerate() {
            *sum = clamped(x);
        }
        for (i, sum) in sums[inside_end..].iter_mut().enumerate() {
            *sum = clamped(inside_end + i);
        }
        // A radius wider than the row leaves no interior at all, and the ends
        // above have already covered every column.
        if inside_start < inside_end {
            let ahead = &totals[inside_start + radius + 1..inside_end + radius + 1];
            let behind = &totals[inside_start - radius..inside_end - radius];
            for ((sum, a), b) in sums[inside_start..inside_end]
                .iter_mut()
                .zip(ahead)
                .zip(behind)
            {
                *sum = *a - *b;
            }
        }

        emit_row(sums, div, &mut dst[y * w..y * w + w]);
    }
}

/// `blur_rows` down the columns instead, as one running total per column, so
/// the rows it reads are contiguous and a row of its output is `w`
/// independent lanes.
fn blur_columns(src: &[u8], dst: &mut [u8], w: usize, h: usize, radius: usize, sums: &mut [u32]) {
    let div = DivideBy::new(radius);
    let row = |y: usize| {
        let y = y.min(h - 1);
        &src[y * w..y * w + w]
    };

    // The window at row 0, everything above the edge clamped to it.
    for (sum, sample) in sums.iter_mut().zip(row(0)) {
        *sum = u32::from(*sample) * (radius + 1) as u32;
    }
    for y in 1..=radius {
        for (sum, sample) in sums.iter_mut().zip(row(y)) {
            *sum += u32::from(*sample);
        }
    }
    for y in 0..h {
        emit_row(sums, div, &mut dst[y * w..y * w + w]);
        let (leaving, arriving) = (row(y.saturating_sub(radius)), row(y + radius + 1));
        for ((sum, out), into) in sums.iter_mut().zip(leaving).zip(arriving) {
            *sum = *sum - u32::from(*out) + u32::from(*into);
        }
    }
}

/// One 8-bit plane, as a region blur reads it: a pixel at `(x, y)` sits at
/// `y * stride + x * step + offset`, which covers both a packed RGBA channel
/// and a planar YUV one.
struct Plane<'a> {
    data: &'a mut [u8],
    stride: usize,
    step: usize,
    offset: usize,
}

impl Plane<'_> {
    fn at(&self, region: Region, row: usize, col: usize) -> usize {
        (region.y + row) * self.stride + (region.x + col) * self.step + self.offset
    }
}

/// The blur's settings and the buffers it works in, reused across rectangles
/// and frames.
struct Blur {
    radius: usize,
    passes: u32,
    region: Vec<u8>,
    scratch: Vec<u8>,
    /// One row's running total, `w + 1` long.
    totals: Vec<u32>,
    /// The window sums of the row being written, one per column.
    sums: Vec<u32>,
}

impl Blur {
    /// Blurs one region of one plane in place, at `radius` - the chroma
    /// planes take a smaller one than the luma.
    fn apply(&mut self, plane: Plane<'_>, region: Region, radius: usize) {
        let (w, h) = (region.w, region.h);
        if w == 0 || h == 0 {
            return;
        }
        self.region.clear();
        self.region.resize(w * h, 0);
        self.scratch.clear();
        self.scratch.resize(w * h, 0);
        self.totals.clear();
        self.totals.resize(w + 1, 0);
        self.sums.clear();
        self.sums.resize(w, 0);

        // A planar plane's rows are contiguous, so the region comes out a row
        // at a time; a packed one is read a pixel at a time because its
        // channel is strided.
        for row in 0..h {
            let at = plane.at(region, row, 0);
            let into = &mut self.region[row * w..row * w + w];
            if plane.step == 1 {
                into.copy_from_slice(&plane.data[at..at + w]);
            } else {
                for (col, sample) in into.iter_mut().enumerate() {
                    *sample = plane.data[at + col * plane.step];
                }
            }
        }

        // A few box passes approximate a Gaussian, which is what makes the
        // result unreadable rather than merely soft.
        for _ in 0..self.passes {
            blur_rows(
                &self.region,
                &mut self.scratch,
                w,
                h,
                radius,
                &mut self.totals,
                &mut self.sums,
            );
            blur_columns(
                &self.scratch,
                &mut self.region,
                w,
                h,
                radius,
                &mut self.sums,
            );
        }

        for row in 0..h {
            let at = plane.at(region, row, 0);
            let from = &self.region[row * w..row * w + w];
            if plane.step == 1 {
                plane.data[at..at + w].copy_from_slice(from);
            } else {
                for (col, sample) in from.iter().enumerate() {
                    plane.data[at + col * plane.step] = *sample;
                }
            }
        }
    }
}

/// Blurs every region of one frame, leaving the alpha channel alone.
fn blur_frame(state: &mut State, frame: &mut [u8], regions: &[Region]) {
    let (width, height) = (state.width, state.height);
    let radius = state.blur.radius;

    match state.pix_fmt {
        PixFmt::Rgba => {
            for region in regions {
                for offset in 0..3 {
                    state.blur.apply(
                        Plane {
                            data: frame,
                            stride: width * 4,
                            step: 4,
                            offset,
                        },
                        *region,
                        radius,
                    );
                }
            }
        }
        PixFmt::Yuv420p => {
            let chroma_w = width / 2;
            let chroma_h = height / 2;
            let (luma, chroma) = frame.split_at_mut(width * height);
            let (u, v) = chroma.split_at_mut(chroma_w * chroma_h);
            // Chroma is half resolution both ways, so the radius halves with
            // it and the blur covers the same picture area.
            let chroma_radius = radius.div_ceil(2);
            for region in regions {
                state.blur.apply(
                    Plane {
                        data: luma,
                        stride: width,
                        step: 1,
                        offset: 0,
                    },
                    *region,
                    radius,
                );
                let small = chroma_region(*region, chroma_w, chroma_h);
                for plane in [&mut *u, &mut *v] {
                    state.blur.apply(
                        Plane {
                            data: plane,
                            stride: chroma_w,
                            step: 1,
                            offset: 0,
                        },
                        small,
                        chroma_radius,
                    );
                }
            }
        }
    }
}

struct BlurBoxes;

impl Guest for BlurBoxes {
    fn describe() -> Meta {
        Meta {
            name: "blur-boxes".to_string(),
            version: "0.1.0".to_string(),
            params_schema: PARAMS_SCHEMA.to_string(),
            rows_schema: String::new(),
            pixel_formats: vec!["yuv420p".to_string(), "rgba".to_string()],
            // Not an audio module, so it names no sample formats.
            sample_formats: vec![],
            sample_rates: vec![],
            channel_counts: vec![],
            rows_language: vec![],
        }
    }

    fn init(
        width: u32,
        height: u32,
        pix_fmt: String,
        _stream_info: StreamInfo,
        params: String,
    ) -> Result<(), String> {
        let parsed = parse_params(&params)?;

        let pix_fmt = match pix_fmt.as_str() {
            "yuv420p" => PixFmt::Yuv420p,
            "rgba" => PixFmt::Rgba,
            other => return Err(format!("unsupported pix-fmt: {other:?}")),
        };

        STATE.with(|s| {
            *s.borrow_mut() = Some(State {
                width: width as usize,
                height: height as usize,
                pix_fmt,
                blur: Blur {
                    radius: parsed.radius as usize,
                    passes: parsed.passes,
                    region: Vec::new(),
                    scratch: Vec::new(),
                    totals: Vec::new(),
                    sums: Vec::new(),
                },
            });
        });
        Ok(())
    }

    fn set_params(params: String) -> Result<(), String> {
        let parsed = parse_params(&params)?;
        STATE.with(|s| {
            let mut state_ref = s.borrow_mut();
            let state = state_ref.as_mut().expect("set_params called before init");
            state.blur.radius = parsed.radius as usize;
            state.blur.passes = parsed.passes;
        });
        Ok(())
    }

    fn frame_independent() -> bool {
        true
    }

    /// Without rows there is nothing to blur. A host that hands this module
    /// frames through `process` gets them back untouched.
    fn process(_info: FrameInfo, _frame: Vec<u8>) -> Outcome {
        Outcome {
            output: Output::Passthrough,
            rows: vec![],
        }
    }
}

impl MetaGuest for BlurBoxes {
    fn process_meta(_info: FrameInfo, frame: Vec<u8>, rows_in: Vec<String>) -> Outcome {
        STATE.with(|s| {
            let mut state_ref = s.borrow_mut();
            let state = state_ref.as_mut().expect("process called before init");

            let rects = regions(&rows_in, state.width, state.height);
            if rects.is_empty() {
                return Outcome {
                    output: Output::Passthrough,
                    rows: vec![],
                };
            }

            let mut out = frame;
            blur_frame(state, &mut out, &rects);
            Outcome {
                output: Output::Frame(out),
                rows: vec![],
            }
        })
    }
}

export!(BlurBoxes);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_radius_wider_than_the_region_averages_the_whole_row() {
        // Every window then overhangs both ends of its row, which leaves the
        // pass no interior between the clamped ends to walk.
        let (w, h) = (4usize, 4usize);
        let src: Vec<u8> = (0..(w * h) as u8).map(|i| i * 17).collect();
        let mut dst = vec![0u8; w * h];
        let mut totals = vec![0u32; w + 1];
        let mut sums = vec![0u32; w];
        blur_rows(&src, &mut dst, w, h, 9, &mut totals, &mut sums);

        // The window at column 0 is 19 wide: nine steps clamp onto the row's
        // first sample, six onto its last, and the row itself is the four
        // between them.
        let row: u32 = src[..w].iter().map(|s| u32::from(*s)).sum();
        let first = u32::from(src[0]);
        let last = u32::from(src[w - 1]);
        let expected = (9 * first + row + 6 * last) / 19;
        assert_eq!(u32::from(dst[0]), expected);
    }
}
