//! What both builds compute the same way: where the frame sits in the square
//! the graph is run at, and the map the graph returns, sampled back out to
//! the frame and written in its format.

/// The name the host binds the graph to. `-nn depth=<path>`.
pub const MODEL: &str = "depth";

/// The square the graph is run at. Depth Anything works in patches of 14, and
/// 518 is 37 of them - the size its own preprocessing uses.
pub const SIDE: usize = 518;

/// The graph's own names for the tensors it takes and returns.
pub const INPUT_NAME: &str = "pixel_values";
pub const OUTPUT_NAME: &str = "predicted_depth";

/// What the model was trained on: each channel rescaled to 0..1, then
/// standardized by these.
pub const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
pub const STD: [f32; 3] = [0.229, 0.224, 0.225];

/// The pixel format of the instance's frames in host memory.
#[derive(Clone, Copy, PartialEq)]
pub enum PixFmt {
    Yuv420p,
    Rgba,
}

/// Where the frame sits inside the square the graph is run at, once it has
/// been scaled to fit with its shape kept: a letterbox.
#[derive(Clone, Copy)]
pub struct Letterbox {
    /// Square pixels per frame pixel.
    pub scale: f32,
    /// Where the scaled frame starts inside the square.
    pub offset_x: f32,
    pub offset_y: f32,
    /// How much of the square the scaled frame covers; the host build's
    /// own resize reads it.
    #[cfg_attr(feature = "gpu", allow(dead_code))]
    pub width: usize,
    #[cfg_attr(feature = "gpu", allow(dead_code))]
    pub height: usize,
}

impl Letterbox {
    pub fn new(width: usize, height: usize) -> Letterbox {
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
    pub fn to_square(self, x: usize, y: usize) -> (f32, f32) {
        (
            self.offset_x + (x as f32 + 0.5) * self.scale - 0.5,
            self.offset_y + (y as f32 + 0.5) * self.scale - 0.5,
        )
    }
}

/// The graph's output as a plain grid of floats, whichever rank it came back
/// with: [1, h, w] as this graph spells it, or [1, 1, h, w] as an export that
/// keeps the channel does.
pub fn depth_grid(dimensions: &[u32], data: &[u8]) -> Result<(Vec<f32>, usize, usize), String> {
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
pub fn to_grey(
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
pub fn to_frame(grey: &[u8], pix_fmt: PixFmt, width: usize, height: usize) -> Vec<u8> {
    match pix_fmt {
        PixFmt::Yuv420p => {
            let chroma = width.div_ceil(2) * height.div_ceil(2);
            let mut out = vec![128u8; width * height + 2 * chroma];
            out[..width * height].copy_from_slice(grey);
            out
        }
        PixFmt::Rgba => {
            let mut out = vec![0u8; width * height * 4];
            for (index, value) in grey.iter().enumerate() {
                let base = index * 4;
                out[base] = *value;
                out[base + 1] = *value;
                out[base + 2] = *value;
                out[base + 3] = 255;
            }
            out
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_wide_frame_is_letterboxed_with_bars_above_and_below() {
        let box_ = Letterbox::new(1024, 512);
        assert_eq!(box_.width, SIDE, "the wide side fills the square");
        // 512 frame pixels at 518/1024 to a pixel are 259, centred.
        assert_eq!(box_.height, 259, "and the other is scaled to fit");
        assert_eq!(box_.offset_x, 0.0);
        assert_eq!(box_.offset_y, 129.0, "the bars are above and below");
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
        let yuv = to_frame(&grey, PixFmt::Yuv420p, 4, 4);
        assert_eq!(yuv.len(), 4 * 4 + 2 * 2 * 2);
        assert!(
            yuv[..16].iter().all(|v| *v == 40),
            "the luma plane is the map"
        );
        assert!(
            yuv[16..].iter().all(|v| *v == 128),
            "and the chroma is neutral"
        );

        let rgba = to_frame(&grey, PixFmt::Rgba, 4, 4);
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
