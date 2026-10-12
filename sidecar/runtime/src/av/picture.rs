//! How the sidecar's wire lays a picture out: the formats it carries and
//! the planes each is, tightly packed, one after another.

/// The pixel formats the wire carries, by ffmpeg's names for them.
pub const WIRE_PIX_FMTS: &[&str] = &["rgba", "yuv420p", "yuv422p", "yuv444p", "gray"];

/// One plane: the bytes in a row, and how many rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Plane {
    pub row_bytes: usize,
    pub rows: usize,
}

/// The planes of a `width`x`height` picture in `pix_fmt`, in the order the
/// wire writes them: rgba interleaved in one, yuv as Y then U then V. A
/// chroma plane's size rounds up, as ffmpeg's own rawvideo does. None for a
/// format the wire does not carry.
pub fn planes(pix_fmt: &str, width: u32, height: u32) -> Option<Vec<Plane>> {
    let w = width as usize;
    let h = height as usize;
    let plane = |row_bytes, rows| Plane { row_bytes, rows };
    let planes = match pix_fmt {
        "rgba" => vec![plane(w * 4, h)],
        "gray" => vec![plane(w, h)],
        "yuv420p" => {
            let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
            vec![plane(w, h), plane(cw, ch), plane(cw, ch)]
        }
        "yuv422p" => {
            let cw = w.div_ceil(2);
            vec![plane(w, h), plane(cw, h), plane(cw, h)]
        }
        "yuv444p" => vec![plane(w, h), plane(w, h), plane(w, h)],
        _ => return None,
    };
    Some(planes)
}

/// The bytes one packed picture takes.
pub fn frame_len(pix_fmt: &str, width: u32, height: u32) -> Option<usize> {
    planes(pix_fmt, width, height).map(|p| p.iter().map(|p| p.row_bytes * p.rows).sum())
}

/// The wire's spelling of `pix_fmt`, as a `'static` str, or None for one it
/// does not carry.
pub fn wire_pix_fmt(pix_fmt: &str) -> Option<&'static str> {
    WIRE_PIX_FMTS.iter().copied().find(|&f| f == pix_fmt)
}

/// Whether the format is rgb rather than yuv, which decides the colour
/// matrix a converted picture is tagged with.
pub fn is_rgb(pix_fmt: &str) -> bool {
    pix_fmt == "rgba"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_wire_sizes_match_the_ones_main_frames_with() {
        assert_eq!(frame_len("rgba", 64, 48), Some(64 * 48 * 4));
        assert_eq!(frame_len("yuv420p", 64, 48), Some(64 * 48 * 3 / 2));
        assert_eq!(frame_len("yuv422p", 64, 48), Some(64 * 48 * 2));
        assert_eq!(frame_len("yuv444p", 64, 48), Some(64 * 48 * 3));
        assert_eq!(frame_len("gray", 64, 48), Some(64 * 48));
        assert_eq!(frame_len("nv12", 64, 48), None);
    }

    #[test]
    fn an_odd_chroma_plane_rounds_up() {
        let p = planes("yuv420p", 5, 3).unwrap();
        assert_eq!(
            p[1],
            Plane {
                row_bytes: 3,
                rows: 2
            }
        );
    }
}
