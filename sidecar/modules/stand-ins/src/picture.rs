use ffrwd_frame::{planes, Filter, Norm, Rect, Rgba};

use crate::font;

/// What `planes` multiplies by to hand back eight-bit values unchanged:
/// scaled to 0..1, then divided by 1/255.
const UNCHANGED: Norm = Norm {
    mean: [0.0; 3],
    std: [1.0 / 255.0; 3],
};

/// An RGBA picture to draw on: four bytes a pixel, row after row, no
/// padding.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Picture {
    pub data: Vec<u8>,
    pub width: usize,
    pub height: usize,
}

impl Picture {
    /// The bytes a host handed over for a `width` x `height` rgba frame.
    pub fn new(data: Vec<u8>, width: usize, height: usize) -> Result<Picture, String> {
        Rgba::new(&data, width, height)?;
        Ok(Picture {
            data,
            width,
            height,
        })
    }

    /// An opaque picture of one colour.
    pub fn filled(width: usize, height: usize, colour: [u8; 4]) -> Picture {
        Picture {
            data: colour.repeat(width * height),
            width,
            height,
        }
    }

    pub fn view(&self) -> Rgba<'_> {
        Rgba {
            data: &self.data,
            width: self.width,
            height: self.height,
        }
    }

    fn clip(&self, rect: Rect) -> Rect {
        let x1 = rect.x1.min(self.width);
        let y1 = rect.y1.min(self.height);
        Rect {
            x0: rect.x0.min(x1),
            y0: rect.y0.min(y1),
            x1,
            y1,
        }
    }

    /// `rect` painted over, alpha `colour[3]` of 255.
    pub fn fill(&mut self, rect: Rect, colour: [u8; 4]) {
        let rect = self.clip(rect);
        let alpha = colour[3] as u32;
        for y in rect.y0..rect.y1 {
            let row =
                &mut self.data[(y * self.width + rect.x0) * 4..(y * self.width + rect.x1) * 4];
            let (pixels, _) = row.as_chunks_mut::<4>();
            for pixel in pixels {
                for channel in 0..3 {
                    let under = pixel[channel] as u32;
                    pixel[channel] =
                        ((colour[channel] as u32 * alpha + under * (255 - alpha) + 127) / 255)
                            as u8;
                }
            }
        }
    }

    /// `rect`'s edge, `thickness` pixels wide, drawn inside it.
    pub fn outline(&mut self, rect: Rect, thickness: usize, colour: [u8; 4]) {
        let rect = self.clip(rect);
        let t = thickness
            .min(rect.width().div_ceil(2))
            .min(rect.height().div_ceil(2));
        let band = |x0, y0, x1, y1| Rect { x0, y0, x1, y1 };
        self.fill(band(rect.x0, rect.y0, rect.x1, rect.y0 + t), colour);
        self.fill(band(rect.x0, rect.y1 - t, rect.x1, rect.y1), colour);
        self.fill(band(rect.x0, rect.y0, rect.x0 + t, rect.y1), colour);
        self.fill(band(rect.x1 - t, rect.y0, rect.x1, rect.y1), colour);
    }

    /// Every pixel `covered` names, one flag a pixel, darkened by `amount`
    /// of its value: 0 leaves it, 1 makes it black.
    pub fn darken(&mut self, covered: &[bool], amount: f64) {
        let keep = ((1.0 - amount.clamp(0.0, 1.0)) * 256.0).round() as u32;
        let (pixels, _) = self.data.as_chunks_mut::<4>();
        for (pixel, _) in pixels
            .iter_mut()
            .zip(covered)
            .filter(|(_, covered)| **covered)
        {
            for channel in &mut pixel[..3] {
                *channel = ((*channel as u32 * keep) >> 8) as u8;
            }
        }
    }

    /// `other` copied in with its top left corner at `x`, `y`, cut to fit.
    pub fn put(&mut self, x: usize, y: usize, other: &Picture) {
        if x >= self.width || y >= self.height {
            return;
        }
        let w = other.width.min(self.width - x);
        for row in 0..other.height.min(self.height - y) {
            let to = ((y + row) * self.width + x) * 4;
            let from = row * other.width * 4;
            self.data[to..to + w * 4].copy_from_slice(&other.data[from..from + w * 4]);
        }
    }

    /// The picture resized to `width` x `height` with Pillow's bilinear
    /// filter, by way of ffrwd-frame. Alpha comes out opaque.
    pub fn resized(&self, width: usize, height: usize) -> Picture {
        let planar = planes(
            &self.view(),
            Rect::whole(self.width, self.height),
            width,
            height,
            Filter::Bilinear,
            UNCHANGED,
        );
        let area = width * height;
        let mut data = Vec::with_capacity(area * 4);
        for at in 0..area {
            for channel in 0..3 {
                data.push(planar[channel * area + at].round().clamp(0.0, 255.0) as u8);
            }
            data.push(255);
        }
        Picture {
            data,
            width,
            height,
        }
    }

    /// `text` in the bitmap font, each font pixel `scale` pixels square,
    /// its first glyph's top left corner at `x`, `y`; whatever falls off the
    /// picture is left out.
    pub fn text(&mut self, x: i64, y: i64, scale: usize, text: &str, colour: [u8; 4]) {
        let scale = scale.max(1);
        let step = (font::ADVANCE * scale) as i64;
        for (n, c) in text.chars().enumerate() {
            let left = x + n as i64 * step;
            if left >= self.width as i64 {
                break;
            }
            if left + step <= 0 {
                continue;
            }
            for (row, bits) in font::glyph(c).iter().enumerate() {
                for column in 0..font::WIDTH {
                    if bits & (1 << (font::WIDTH - 1 - column)) == 0 {
                        continue;
                    }
                    let px = left + (column * scale) as i64;
                    let py = y + (row * scale) as i64;
                    if px + scale as i64 <= 0 || py + scale as i64 <= 0 {
                        continue;
                    }
                    let (x0, y0) = (px.max(0) as usize, py.max(0) as usize);
                    let x1 = (px + scale as i64).max(0) as usize;
                    let y1 = (py + scale as i64).max(0) as usize;
                    self.fill(Rect { x0, y0, x1, y1 }, colour);
                }
            }
        }
    }
}

/// How wide and tall `text` stands in the bitmap font at `scale`.
pub fn text_size(text: &str, scale: usize) -> (usize, usize) {
    let count = text.chars().count();
    let width = (count * font::ADVANCE).saturating_sub(1) * scale.max(1);
    (width, font::HEIGHT * scale.max(1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fill_blends_by_alpha() {
        let mut picture = Picture::filled(4, 4, [0, 0, 0, 255]);
        picture.fill(Rect::whole(2, 2), [255, 255, 255, 255]);
        picture.fill(
            Rect {
                x0: 2,
                y0: 0,
                x1: 9,
                y1: 1,
            },
            [200, 100, 0, 128],
        );
        assert_eq!(&picture.data[..4], &[255, 255, 255, 255]);
        assert_eq!(&picture.data[8..12], &[100, 50, 0, 255]);
        assert_eq!(&picture.data[12..16], &[100, 50, 0, 255]);
        assert_eq!(&picture.data[16 * 2..16 * 2 + 4], &[0, 0, 0, 255]);
    }

    #[test]
    fn resized_keeps_a_flat_colour() {
        let picture = Picture::filled(8, 6, [10, 120, 250, 255]);
        let half = picture.resized(4, 3);
        assert_eq!((half.width, half.height), (4, 3));
        assert!(half
            .data
            .chunks(4)
            .all(|pixel| pixel == [10, 120, 250, 255]));
    }

    #[test]
    fn text_lands_where_it_says() {
        let mut picture = Picture::filled(20, 12, [0, 0, 0, 255]);
        picture.text(1, 1, 1, "I", [255, 255, 255, 255]);
        let lit = |x: usize, y: usize| picture.data[(y * 20 + x) * 4] == 255;
        assert!(lit(2, 1) && lit(3, 1) && lit(4, 1));
        assert!(lit(3, 4) && !lit(2, 4));
        assert_eq!(text_size("Ii", 2), (22, 18));
        picture.text(-100, 1, 1, "far off", [255, 0, 0, 255]);
    }
}
