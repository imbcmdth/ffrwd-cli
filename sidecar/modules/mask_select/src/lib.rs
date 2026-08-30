//! Picks instances out of an index map: a binary mask of the ones that
//! survived.
//!
//! One video stream in, carrying an index map - every pixel is the id of the
//! instance that owns it, as a luma value, with 0 for background. The rows
//! arriving with the frame say which ids are still wanted; a module upstream
//! has usually already dropped the rest. Out comes a mask: 255 where the pixel
//! belongs to a surviving id, 0 everywhere else.
//!
//! A row is read for its `id` and nothing else. Rows carrying no `id` are
//! another module's and are skipped, the way every consumer skips the rows it
//! was not written for. No rows at all means nothing survived, so the mask is
//! entirely black.
//!
//! The rows are the selection, not the output: this module emits none of its
//! own and passes none on.

wit_bindgen::generate!({
    path: "../../wit",
    world: "window-module",
});

use std::cell::RefCell;

use exports::ffrwd::av::window_filter::{
    Format, FramePayload, Guest, InFrame, Meta, OutFrame, Processed, StreamInfo, WindowMeta,
};
use serde::Deserialize;

const PARAMS_SCHEMA: &str = r#"{"type":"object","properties":{},"additionalProperties":false}"#;

/// What a selected pixel and a rejected one carry.
const KEEP: u8 = 255;
const DROP: u8 = 0;

/// Ids an index map can spell: one byte of luma, so 0 through 255.
const IDS: usize = 256;

/// The pixel format the host chose at `init`, fixed for the instance's life.
#[derive(Clone, Copy, PartialEq)]
enum PixFmt {
    Yuv420p,
    Rgba,
}

/// What `init` settled.
#[derive(Clone, Copy)]
struct Opened {
    width: usize,
    height: usize,
    pix_fmt: PixFmt,
}

thread_local! {
    static OPENED: RefCell<Option<Opened>> = const { RefCell::new(None) };
}

/// One row's id. Every other field is another module's business.
#[derive(Deserialize)]
struct Selected {
    id: i64,
}

struct MaskSelect;

fn validate_params(params: &str) -> Result<(), String> {
    match params.trim() {
        "" | "{}" => Ok(()),
        other => Err(format!("mask_select takes no params, got: {other}")),
    }
}

/// One lookup per id an index map can carry: `KEEP` for a surviving id and
/// `DROP` for the rest. Built once per frame, so the plane below is a walk
/// with no branch and no search in it.
///
/// A row that is not an object, or carries no `id`, is skipped. An id no byte
/// can spell is skipped too: no pixel of an index map could be carrying it.
fn keep_table(rows: &[String]) -> [u8; IDS] {
    let mut table = [DROP; IDS];
    for row in rows {
        let Ok(selected) = serde_json::from_str::<Selected>(row) else {
            continue;
        };
        if let Ok(id) = usize::try_from(selected.id) {
            if id < IDS {
                table[id] = KEEP;
            }
        }
    }
    table
}

/// An index plane mapped through the table, in contiguous runs. The index is
/// a byte and the table has an entry per byte, so no lookup is bounds checked
/// and no lane depends on the one before it.
fn select_plane(ids: &[u8], table: &[u8; IDS], out: &mut [u8]) {
    const LANES: usize = 16;
    let (whole, tail) = out.as_chunks_mut::<LANES>();
    let (source, source_tail) = ids.as_chunks::<LANES>();
    for (run, ids) in whole.iter_mut().zip(source) {
        for (sample, id) in run.iter_mut().zip(ids) {
            *sample = table[*id as usize];
        }
    }
    for (sample, id) in tail.iter_mut().zip(source_tail) {
        *sample = table[*id as usize];
    }
}

/// The mask one index map frame becomes, in the format the instance was
/// opened for.
fn select(opened: &Opened, frame: &[u8], rows: &[String]) -> Vec<u8> {
    let table = keep_table(rows);
    let pixels = opened.width * opened.height;
    let mut out = vec![DROP; frame.len()];

    match opened.pix_fmt {
        PixFmt::Yuv420p => {
            select_plane(&frame[..pixels], &table, &mut out[..pixels]);
            // 128 in both chroma planes is no colour at all.
            out[pixels..].fill(128);
        }
        PixFmt::Rgba => {
            // The id sits in every colour channel of an index map, so red is
            // read and all three are written.
            let mut mask = vec![DROP; pixels];
            let ids: Vec<u8> = frame.as_chunks::<4>().0.iter().map(|p| p[0]).collect();
            select_plane(&ids, &table, &mut mask);
            for (pixel, value) in out.as_chunks_mut::<4>().0.iter_mut().zip(&mask) {
                *pixel = [*value, *value, *value, 255];
            }
        }
    }
    out
}

impl Guest for MaskSelect {
    fn describe() -> WindowMeta {
        WindowMeta {
            meta: Meta {
                name: "mask_select".to_string(),
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
            // The rows are the selection.
            reads_rows: true,
            // And they are consumed by it: what leaves is the mask alone.
            forwards_rows: false,
            inputs: 1,
        }
    }

    fn init(format: Format, _stream_info: StreamInfo, params: String) -> Result<(), String> {
        let Format::Video(video) = format else {
            return Err("mask_select reads frames, and this stream is audio".to_string());
        };
        let pix_fmt = match video.pix_fmt.as_str() {
            "yuv420p" => PixFmt::Yuv420p,
            "rgba" => PixFmt::Rgba,
            other => return Err(format!("mask_select does not accept pixel format {other}")),
        };
        validate_params(&params)?;

        OPENED.with(|o| {
            *o.borrow_mut() = Some(Opened {
                width: video.width as usize,
                height: video.height as usize,
                pix_fmt,
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
        let opened = OPENED
            .with(|o| *o.borrow())
            .expect("init settles the geometry before any frame arrives");

        let out = frames
            .iter()
            .map(|frame| OutFrame {
                pts: frame.pts,
                frame: FramePayload::New(select(&opened, &frame.frame, &frame.rows)),
                // The selection was the rows' whole purpose; none travel on.
                rows: vec![],
            })
            .collect();
        Processed {
            frames: out,
            trailing: vec![],
        }
    }
}

export!(MaskSelect);

#[cfg(test)]
mod tests {
    use super::*;

    const W: usize = 8;
    const H: usize = 4;

    fn opened(pix_fmt: PixFmt) -> Opened {
        Opened {
            width: W,
            height: H,
            pix_fmt,
        }
    }

    fn rows(items: &[&str]) -> Vec<String> {
        items.iter().map(|r| r.to_string()).collect()
    }

    /// A yuv420p index map whose luma is `ids`, one byte a pixel.
    fn yuv_map(ids: &[u8]) -> Vec<u8> {
        let mut frame = vec![128u8; W * H + 2 * (W / 2) * (H / 2)];
        frame[..W * H].copy_from_slice(ids);
        frame
    }

    /// Ids laid out in vertical bands: column x carries id `x / 2`.
    fn bands() -> Vec<u8> {
        (0..W * H).map(|i| (i % W / 2) as u8).collect()
    }

    #[test]
    fn a_surviving_id_paints_its_pixels_and_nothing_else() {
        let out = select(
            &opened(PixFmt::Yuv420p),
            &yuv_map(&bands()),
            &rows(&[r#"{"id":2,"class":"person","score":0.9}"#]),
        );
        let ids = bands();
        for (index, id) in ids.iter().enumerate() {
            let expected = if *id == 2 { KEEP } else { DROP };
            assert_eq!(out[index], expected, "pixel {index} carries id {id}");
        }
    }

    #[test]
    fn several_ids_survive_together() {
        let out = select(
            &opened(PixFmt::Yuv420p),
            &yuv_map(&bands()),
            &rows(&[r#"{"id":1}"#, r#"{"id":3}"#]),
        );
        let ids = bands();
        let kept: Vec<u8> = ids
            .iter()
            .map(|id| if *id == 1 || *id == 3 { KEEP } else { DROP })
            .collect();
        assert_eq!(out[..W * H], kept[..]);
    }

    #[test]
    fn no_rows_at_all_is_an_entirely_black_mask() {
        let out = select(&opened(PixFmt::Yuv420p), &yuv_map(&bands()), &[]);
        assert!(
            out[..W * H].iter().all(|v| *v == DROP),
            "nothing survived, so nothing is selected"
        );
    }

    #[test]
    fn rows_of_another_module_are_skipped_rather_than_refused() {
        let out = select(
            &opened(PixFmt::Yuv420p),
            &yuv_map(&bands()),
            &rows(&[
                r#"{"shot":4,"score":0.2}"#,
                "not json at all",
                r#"{"id":2}"#,
            ]),
        );
        let ids = bands();
        for (index, id) in ids.iter().enumerate() {
            let expected = if *id == 2 { KEEP } else { DROP };
            assert_eq!(out[index], expected);
        }
    }

    #[test]
    fn an_id_no_index_map_could_carry_selects_nothing() {
        let table = keep_table(&rows(&[r#"{"id":900}"#, r#"{"id":-3}"#]));
        assert!(
            table.iter().all(|v| *v == DROP),
            "no pixel of a one-byte index map holds either id"
        );
    }

    #[test]
    fn the_mask_keeps_neutral_chroma() {
        let out = select(
            &opened(PixFmt::Yuv420p),
            &yuv_map(&bands()),
            &rows(&[r#"{"id":2}"#]),
        );
        assert!(
            out[W * H..].iter().all(|v| *v == 128),
            "a mask carries no colour"
        );
    }

    #[test]
    fn an_rgba_map_reads_its_id_and_writes_every_channel() {
        let ids = bands();
        let mut frame = vec![255u8; W * H * 4];
        for (pixel, id) in frame.as_chunks_mut::<4>().0.iter_mut().zip(&ids) {
            *pixel = [*id, *id, *id, 255];
        }
        let out = select(&opened(PixFmt::Rgba), &frame, &rows(&[r#"{"id":3}"#]));
        for (pixel, id) in out.as_chunks::<4>().0.iter().zip(&ids) {
            let expected = if *id == 3 { KEEP } else { DROP };
            assert_eq!(*pixel, [expected, expected, expected, 255]);
        }
    }

    #[test]
    fn a_run_that_is_not_a_whole_number_of_lanes_is_still_mapped() {
        // The plane's tail is what a lane-wise walk leaves behind.
        let mut table = [DROP; IDS];
        table[5] = KEEP;
        let ids: Vec<u8> = (0..37).map(|i| (i % 8) as u8).collect();
        let mut out = vec![0u8; ids.len()];
        select_plane(&ids, &table, &mut out);
        for (index, id) in ids.iter().enumerate() {
            assert_eq!(out[index], if *id == 5 { KEEP } else { DROP });
        }
    }
}
