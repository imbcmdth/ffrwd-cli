//! Depth of field: pad 0's frame, blurred where pad 1's mask says to.
//!
//! Two streams in, one out. Pad 0 carries the frame; pad 1 carries a
//! greyscale mask of the same geometry, where black is sharp and white is as
//! blurred as `max_radius` allows. `invert` swaps those ends.
//!
//! The blur is a pyramid rather than one variable-radius pass. Each level is
//! the level below it box-blurred again, at radii 1, 2, 4, ... up to
//! `max_radius`, so the levels get geometrically softer for a cost that grows
//! with their number and not with the radius. Every pixel then reads its
//! mask value as a position along that stack and mixes the two levels it
//! falls between - so the softness varies per pixel, continuously, while the
//! frame is blurred only a handful of times.
//!
//! Both formats blur the same way: yuv420p blurs its three planes, and rgba
//! blurs red, green and blue and carries alpha through untouched. Chroma is
//! blurred at half the luma radius, so the softness covers the same distance
//! on the picture in either plane.

wit_bindgen::generate!({
    path: "../../worlds/0.10.0",
    world: "window-module",
});

use std::cell::RefCell;

use exports::ffrwd::av::window_filter::{
    Format, FramePayload, Guest, InFrame, Meta, OutFrame, Processed, StreamInfo, WindowMeta,
};
use serde::Deserialize;

/// The streams this module reads: the frame, then the mask.
const INPUTS: u32 = 2;

/// Pads by the name the refusals use.
const FRAME_PAD: usize = 0;
const MASK_PAD: usize = 1;

const PARAMS_SCHEMA: &str = r#"{"type":"object","properties":{"max_radius":{"type":"number","default":16},"invert":{"type":"boolean","default":false}},"additionalProperties":false}"#;

fn default_max_radius() -> f64 {
    16.0
}

#[derive(Clone, Copy, Deserialize)]
struct Params {
    #[serde(default = "default_max_radius")]
    max_radius: f64,
    #[serde(default)]
    invert: bool,
}

impl Default for Params {
    fn default() -> Self {
        Params {
            max_radius: default_max_radius(),
            invert: false,
        }
    }
}

/// The pixel format the host chose at `init`, fixed for the instance's life.
#[derive(Clone, Copy, PartialEq)]
enum PixFmt {
    Yuv420p,
    Rgba,
}

/// What `init` settled: the geometry, the format, and the parameters.
#[derive(Clone, Copy)]
struct Opened {
    width: usize,
    height: usize,
    pix_fmt: PixFmt,
    params: Params,
}

thread_local! {
    static OPENED: RefCell<Option<Opened>> = const { RefCell::new(None) };
}

struct BlurMask;

fn parse_params(params: &str) -> Result<Params, String> {
    let trimmed = params.trim();
    let parsed: Params = if trimmed.is_empty() {
        Params::default()
    } else {
        serde_json::from_str(trimmed).map_err(|e| format!("invalid params: {e}"))?
    };
    if !parsed.max_radius.is_finite() || parsed.max_radius < 0.0 {
        return Err(format!(
            "max_radius must be zero or more, got {}",
            parsed.max_radius
        ));
    }
    Ok(parsed)
}

/// The radii the pyramid is built with: 1, 2, 4, ... while they stay within
/// `max_radius`. Empty when the radius leaves no room for even one step,
/// which makes the module an identity.
fn step_radii(max_radius: f64) -> Vec<usize> {
    let limit = max_radius.floor() as usize;
    let mut radii = Vec::new();
    let mut radius = 1usize;
    while radius <= limit {
        radii.push(radius);
        radius *= 2;
    }
    radii
}

/// Exact `sum / divisor` as a multiply and a shift, so neither blur pass
/// divides per pixel. One correction step lands on the true quotient while
/// `255 * divisor` fits the shift, which every window a real frame carries
/// does.
#[derive(Clone, Copy)]
struct DivideBy {
    divisor: u32,
    /// `floor(2^DIVIDE_SHIFT / divisor)`. Rounded down, so the quotient it
    /// gives is the true one or the one below it, never above.
    magic: u32,
}

/// Where `DivideBy` puts its binary point: high enough that rounding the
/// magic down costs at most one, low enough that `sum * magic` stays inside
/// 32 bits.
const DIVIDE_SHIFT: u32 = 24;

impl DivideBy {
    fn new(divisor: u32) -> DivideBy {
        DivideBy {
            divisor,
            magic: (1u32 << DIVIDE_SHIFT) / divisor,
        }
    }

    /// Whether `apply` lands on the true quotient for this divisor. A window
    /// wider than any real frame divides outright instead.
    fn is_exact(self) -> bool {
        self.divisor.saturating_mul(255) <= 1 << DIVIDE_SHIFT
    }

    #[inline(always)]
    fn apply(self, sum: u32) -> u32 {
        let quotient = (sum.wrapping_mul(self.magic)) >> DIVIDE_SHIFT;
        quotient + u32::from(sum - quotient * self.divisor >= self.divisor)
    }
}

/// A row of window sums written out as the samples they average to. No lane
/// depends on another, which is what lets the whole row go at once.
fn emit_row(sums: &[u32], div: DivideBy, dst: &mut [u8]) {
    if div.is_exact() {
        for (sample, sum) in dst.iter_mut().zip(sums) {
            *sample = div.apply(*sum) as u8;
        }
    } else {
        for (sample, sum) in dst.iter_mut().zip(sums) {
            *sample = (*sum / div.divisor) as u8;
        }
    }
}

/// One row's horizontal window sums. `totals` carries the row's running
/// total - `totals[i]` is the sum of its first `i` samples - so each window
/// is one subtraction and no sum depends on the one before it. A window
/// reaching past an edge reads that edge's own sample once per step it
/// overhangs.
fn row_sums(row: &[u8], r: usize, totals: &mut [u32], sums: &mut [u32]) {
    let w = row.len();
    totals[0] = 0;
    let mut running = 0u32;
    for (i, sample) in row.iter().enumerate() {
        running += u32::from(*sample);
        totals[i + 1] = running;
    }

    let first = u32::from(row[0]);
    let last = u32::from(row[w - 1]);
    let clamped = |x: usize| -> u32 {
        totals[(x + r + 1).min(w)] - totals[x.saturating_sub(r)]
            + (r - x.min(r)) as u32 * first
            + ((x + r + 1).saturating_sub(w)) as u32 * last
    };

    // Only the ends need clamping; between them the window sits inside the
    // row and every sum is one subtraction of two contiguous walks.
    let inside_start = r.min(w);
    let inside_end = w.saturating_sub(r).max(inside_start);
    for (x, sum) in sums[..inside_start].iter_mut().enumerate() {
        *sum = clamped(x);
    }
    for (i, sum) in sums[inside_end..].iter_mut().enumerate() {
        *sum = clamped(inside_end + i);
    }
    // A radius wider than the row leaves no interior at all, and the ends
    // above have already covered every column.
    if inside_start < inside_end {
        let ahead = &totals[inside_start + r + 1..inside_end + r + 1];
        let behind = &totals[inside_start - r..inside_end - r];
        for ((sum, a), b) in sums[inside_start..inside_end]
            .iter_mut()
            .zip(ahead)
            .zip(behind)
        {
            *sum = *a - *b;
        }
    }
}

/// Buffers a blur reuses, so a frame's pyramid allocates per frame rather
/// than per level.
#[derive(Default)]
struct Scratch {
    /// One row's running total, `w + 1` long.
    totals: Vec<u32>,
    /// The window sums of the row being written, one per column.
    sums: Vec<u32>,
    /// What the horizontal pass wrote, which the vertical pass reads.
    across: Vec<u8>,
}

impl Scratch {
    fn fit(&mut self, w: usize, h: usize) {
        self.totals.clear();
        self.totals.resize(w + 1, 0);
        self.sums.clear();
        self.sums.resize(w, 0);
        self.across.clear();
        self.across.resize(w * h, 0);
    }
}

/// A box blur of radius `r`, two separable passes with a running sum, so its
/// cost does not grow with the radius. Edges clamp to the nearest sample.
///
/// Both passes walk rows, so every load is contiguous: the horizontal pass
/// reads each window out of its row's running total, and the vertical pass
/// carries one running total per column, which makes a row of its output `w`
/// independent lanes.
fn box_blur(src: &[u8], w: usize, h: usize, r: usize, scratch: &mut Scratch, dst: &mut [u8]) {
    if r == 0 {
        dst.copy_from_slice(src);
        return;
    }
    let div = DivideBy::new((2 * r + 1) as u32);
    scratch.fit(w, h);

    for y in 0..h {
        row_sums(
            &src[y * w..y * w + w],
            r,
            &mut scratch.totals,
            &mut scratch.sums,
        );
        emit_row(&scratch.sums, div, &mut scratch.across[y * w..y * w + w]);
    }

    let across = &scratch.across;
    let sums = &mut scratch.sums;
    let row = |y: usize| {
        let y = y.min(h - 1);
        &across[y * w..y * w + w]
    };

    // The window at row 0, everything above the edge clamped to it.
    for (sum, sample) in sums.iter_mut().zip(row(0)) {
        *sum = u32::from(*sample) * (r + 1) as u32;
    }
    for y in 1..=r {
        for (sum, sample) in sums.iter_mut().zip(row(y)) {
            *sum += u32::from(*sample);
        }
    }
    for y in 0..h {
        emit_row(sums, div, &mut dst[y * w..y * w + w]);
        let (leaving, entering) = (row(y.saturating_sub(r)), row(y + r + 1));
        for ((sum, out), into) in sums.iter_mut().zip(leaving).zip(entering) {
            *sum = *sum - u32::from(*out) + u32::from(*into);
        }
    }
}

/// The stack of progressively blurred versions of one plane: level 0 is the
/// plane itself, and each level after it is the one below blurred again.
fn pyramid(
    plane: &[u8],
    w: usize,
    h: usize,
    radii: &[usize],
    scratch: &mut Scratch,
) -> Vec<Vec<u8>> {
    let mut levels: Vec<Vec<u8>> = Vec::with_capacity(radii.len() + 1);
    levels.push(plane.to_vec());
    for radius in radii {
        let mut next = vec![0u8; w * h];
        let below = levels.last().expect("level 0 is there before any step");
        box_blur(below, w, h, *radius, scratch, &mut next);
        levels.push(next);
    }
    levels
}

/// One plane composited out of its pyramid, each pixel reading `mask` as a
/// position along the stack and mixing the two levels it falls between.
///
/// `mask` runs one value per pixel of this plane, so a subsampled plane is
/// handed a subsampled mask.
fn composite(levels: &[Vec<u8>], mask: &[u8], invert: bool, out: &mut [u8]) {
    let top = levels.len() - 1;
    if top == 0 {
        out.copy_from_slice(&levels[0]);
        return;
    }
    // The position is carried in 1/255ths of a level, so the mix is integer
    // throughout and a mask at either end lands on a level exactly.
    for (index, sample) in out.iter_mut().enumerate() {
        let amount = if invert {
            255 - mask[index]
        } else {
            mask[index]
        };
        let position = u32::from(amount) * top as u32;
        let lower = (position / 255) as usize;
        // A mask at the very top lands exactly on the last level, and there is
        // nothing above it to mix with.
        let (lower, fraction) = if lower >= top {
            (top - 1, 255)
        } else {
            (lower, position - lower as u32 * 255)
        };
        let a = i32::from(levels[lower][index]);
        let b = i32::from(levels[lower + 1][index]);
        *sample = ((a * 255 + (b - a) * fraction as i32 + 127) / 255) as u8;
    }
}

/// The mask's luma at every pixel of a full-resolution frame.
fn mask_luma(mask: &[u8], pix_fmt: PixFmt, width: usize, height: usize) -> Vec<u8> {
    match pix_fmt {
        // The Y plane is the luma already.
        PixFmt::Yuv420p => mask[..width * height].to_vec(),
        PixFmt::Rgba => (0..width * height)
            .map(|index| {
                let base = index * 4;
                let r = u32::from(mask[base]);
                let g = u32::from(mask[base + 1]);
                let b = u32::from(mask[base + 2]);
                // The standard weights, so a mask that is not quite grey still
                // reads as the grey a viewer would see.
                ((299 * r + 587 * g + 114 * b) / 1000) as u8
            })
            .collect(),
    }
}

/// A full-resolution mask sampled down to a chroma plane's grid.
fn subsample(mask: &[u8], width: usize, chroma_width: usize, chroma_height: usize) -> Vec<u8> {
    let mut out = vec![0u8; chroma_width * chroma_height];
    for y in 0..chroma_height {
        for x in 0..chroma_width {
            out[y * chroma_width + x] = mask[(y * 2) * width + x * 2];
        }
    }
    out
}

/// The radii a chroma plane is blurred with: half the luma ones, so the blur
/// covers the same distance on the picture. A step that would round to
/// nothing is kept at 1.
fn chroma_radii(radii: &[usize]) -> Vec<usize> {
    radii.iter().map(|r| (r / 2).max(1)).collect()
}

/// One frame blurred by one mask.
fn blur(opened: &Opened, frame: &[u8], mask: &[u8]) -> Vec<u8> {
    let Opened {
        width,
        height,
        pix_fmt,
        params,
    } = *opened;
    let radii = step_radii(params.max_radius);
    let luma = mask_luma(mask, pix_fmt, width, height);
    let mut out = frame.to_vec();
    let mut scratch = Scratch::default();

    match pix_fmt {
        PixFmt::Rgba => {
            // Deinterleaved, blurred, and put back with alpha as it arrived.
            let pixels = width * height;
            let mut plane = vec![0u8; pixels];
            let mut mixed = vec![0u8; pixels];
            for channel in 0..3 {
                for (sample, pixel) in plane.iter_mut().zip(frame.as_chunks::<4>().0) {
                    *sample = pixel[channel];
                }
                let levels = pyramid(&plane, width, height, &radii, &mut scratch);
                composite(&levels, &luma, params.invert, &mut mixed);
                for (pixel, value) in out.as_chunks_mut::<4>().0.iter_mut().zip(&mixed) {
                    pixel[channel] = *value;
                }
            }
        }
        PixFmt::Yuv420p => {
            let pixels = width * height;
            let (cw, ch) = (width.div_ceil(2), height.div_ceil(2));
            let chroma = cw * ch;

            let levels = pyramid(&frame[..pixels], width, height, &radii, &mut scratch);
            composite(&levels, &luma, params.invert, &mut out[..pixels]);

            let small = subsample(&luma, width, cw, ch);
            let small_radii = chroma_radii(&radii);
            for plane_index in 0..2 {
                let start = pixels + plane_index * chroma;
                let levels = pyramid(
                    &frame[start..start + chroma],
                    cw,
                    ch,
                    &small_radii,
                    &mut scratch,
                );
                composite(
                    &levels,
                    &small,
                    params.invert,
                    &mut out[start..start + chroma],
                );
            }
        }
    }
    out
}

impl Guest for BlurMask {
    fn describe() -> WindowMeta {
        WindowMeta {
            meta: Meta {
                name: "blur_mask".to_string(),
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
            inputs: INPUTS,
        }
    }

    fn init(format: Format, _stream_info: StreamInfo, params: String) -> Result<(), String> {
        let Format::Video(video) = format else {
            return Err("blur_mask reads frames, and this stream is audio".to_string());
        };
        let pix_fmt = match video.pix_fmt.as_str() {
            "yuv420p" => PixFmt::Yuv420p,
            "rgba" => PixFmt::Rgba,
            other => return Err(format!("blur_mask does not accept pixel format {other}")),
        };
        let parsed = parse_params(&params)?;
        OPENED.with(|o| {
            *o.borrow_mut() = Some(Opened {
                width: video.width as usize,
                height: video.height as usize,
                pix_fmt,
                params: parsed,
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
        if frames.is_empty() {
            return Processed {
                frames: vec![],
                trailing: vec![],
            };
        }
        if frames.len() != INPUTS as usize {
            panic!(
                "blur_mask was handed {} pad(s) and reads {INPUTS}: the frame on pad {FRAME_PAD} \
                 and its mask on pad {MASK_PAD}",
                frames.len()
            );
        }
        let opened = OPENED
            .with(|o| *o.borrow())
            .expect("init settles the geometry before any frame arrives");

        let frame = &frames[FRAME_PAD];
        let mask = &frames[MASK_PAD];
        if mask.frame.len() != frame.frame.len() {
            panic!(
                "blur_mask was handed {} bytes on pad {FRAME_PAD} and {} on pad {MASK_PAD}; the \
                 mask has the frame's own geometry",
                frame.frame.len(),
                mask.frame.len()
            );
        }

        let blurred = blur(&opened, &frame.frame, &mask.frame);
        Processed {
            frames: vec![OutFrame {
                pts: frame.pts,
                frame: FramePayload::New(blurred),
                // Pad 0's rows travel on; pad 1 carried none by the time this
                // module saw it.
                rows: frame.rows.clone(),
            }],
            trailing: vec![],
        }
    }
}

export!(BlurMask);

#[cfg(test)]
mod tests {
    use super::*;

    const W: usize = 32;
    const H: usize = 32;

    fn opened(max_radius: f64, invert: bool) -> Opened {
        Opened {
            width: W,
            height: H,
            pix_fmt: PixFmt::Rgba,
            params: Params { max_radius, invert },
        }
    }

    /// A frame of hard vertical stripes, which any blur at all softens.
    fn stripes() -> Vec<u8> {
        let mut frame = vec![0u8; W * H * 4];
        for y in 0..H {
            for x in 0..W {
                let value = if x % 2 == 0 { 0 } else { 255 };
                let base = (y * W + x) * 4;
                frame[base] = value;
                frame[base + 1] = value;
                frame[base + 2] = value;
                frame[base + 3] = 255;
            }
        }
        frame
    }

    /// An rgba mask of one grey level everywhere.
    fn flat_mask(level: u8) -> Vec<u8> {
        let mut mask = vec![255u8; W * H * 4];
        for index in 0..W * H {
            mask[index * 4] = level;
            mask[index * 4 + 1] = level;
            mask[index * 4 + 2] = level;
        }
        mask
    }

    /// An rgba mask that is black on the left half and white on the right.
    fn half_mask() -> Vec<u8> {
        let mut mask = vec![255u8; W * H * 4];
        for y in 0..H {
            for x in 0..W {
                let level = if x < W / 2 { 0 } else { 255 };
                let base = (y * W + x) * 4;
                mask[base] = level;
                mask[base + 1] = level;
                mask[base + 2] = level;
            }
        }
        mask
    }

    /// The variance of the red channel over the columns `xs`, which is what
    /// says how hard the stripes still are.
    fn variance(frame: &[u8], xs: std::ops::Range<usize>) -> f64 {
        let values: Vec<f64> = (0..H)
            .flat_map(|y| xs.clone().map(move |x| (y, x)))
            .map(|(y, x)| f64::from(frame[(y * W + x) * 4]))
            .collect();
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64
    }

    #[test]
    fn a_mask_of_zero_leaves_the_frame_exactly_as_it_arrived() {
        let frame = stripes();
        let out = blur(&opened(16.0, false), &frame, &flat_mask(0));
        assert_eq!(out, frame, "nothing is masked, so nothing is blurred");
    }

    #[test]
    fn a_mask_of_white_blurs_as_far_as_the_radius_allows() {
        let frame = stripes();
        let full = blur(&opened(16.0, false), &frame, &flat_mask(255));
        let sharp = variance(&frame, 0..W);
        let blurred = variance(&full, 0..W);
        assert!(
            blurred < sharp / 10.0,
            "the stripes must flatten: {sharp} sharp, {blurred} blurred"
        );
    }

    #[test]
    fn invert_swaps_which_end_of_the_mask_is_sharp() {
        let frame = stripes();
        let inverted = blur(&opened(16.0, true), &frame, &flat_mask(0));
        assert!(
            variance(&inverted, 0..W) < variance(&frame, 0..W) / 10.0,
            "a black mask is the blurred end once inverted"
        );
        assert_eq!(
            blur(&opened(16.0, true), &frame, &flat_mask(255)),
            frame,
            "and white is the sharp one"
        );
    }

    #[test]
    fn half_a_mask_blurs_half_the_frame() {
        let frame = stripes();
        let out = blur(&opened(16.0, false), &frame, &half_mask());
        let left = &out[..];
        assert_eq!(
            &left[..(W / 2) * 4],
            &frame[..(W / 2) * 4],
            "the unmasked half is untouched, row for row"
        );
        // Away from the seam, so neither half reads the other's pixels.
        assert!(
            variance(&out, W / 2 + 8..W) < variance(&frame, W / 2 + 8..W) / 10.0,
            "the masked half is blurred"
        );
        assert_eq!(
            variance(&out, 0..W / 2 - 8),
            variance(&frame, 0..W / 2 - 8),
            "and the unmasked half is exactly as sharp as it arrived"
        );
    }

    #[test]
    fn a_gradient_mask_blurs_further_the_higher_it_climbs() {
        // Columns of rising mask value: each band must come out no sharper
        // than the band before it.
        let frame = stripes();
        let mut mask = vec![255u8; W * H * 4];
        for y in 0..H {
            for x in 0..W {
                let level = (x * 255 / (W - 1)) as u8;
                let base = (y * W + x) * 4;
                mask[base] = level;
                mask[base + 1] = level;
                mask[base + 2] = level;
            }
        }
        let out = blur(&opened(16.0, false), &frame, &mask);

        let bands: Vec<f64> = (0..4)
            .map(|band| variance(&out, band * 8..band * 8 + 8))
            .collect();
        for pair in bands.windows(2) {
            assert!(
                pair[1] <= pair[0],
                "each band must be no sharper than the one before it: {bands:?}"
            );
        }
        assert!(
            bands[3] < bands[0] / 10.0,
            "and the far end must be properly soft: {bands:?}"
        );
    }

    #[test]
    fn the_radii_climb_in_powers_of_two_up_to_the_limit() {
        assert_eq!(step_radii(16.0), vec![1, 2, 4, 8, 16]);
        assert_eq!(step_radii(5.0), vec![1, 2, 4]);
        assert_eq!(step_radii(1.0), vec![1]);
        assert_eq!(step_radii(0.0), Vec::<usize>::new(), "no room for a step");
    }

    #[test]
    fn a_radius_wider_than_the_frame_blurs_it_rather_than_running_off_the_end() {
        // Every window then overhangs both ends of its row, which leaves the
        // pass no interior between the clamped ends to walk.
        let frame = stripes();
        let out = blur(&opened(64.0, false), &frame, &flat_mask(255));
        assert_eq!(out.len(), frame.len());
        assert!(
            variance(&out, 0..W) < variance(&frame, 0..W) / 10.0,
            "a radius past the frame's own width still flattens the stripes"
        );
    }

    #[test]
    fn a_radius_of_zero_is_an_identity_whatever_the_mask_says() {
        let frame = stripes();
        assert_eq!(blur(&opened(0.0, false), &frame, &flat_mask(255)), frame);
    }
}
