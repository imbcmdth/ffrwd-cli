//! One filter built twice - against the list-shaped windowed world and the
//! current borrowed one - driven straight through
//! `ffrwd_wasm_runtime::runtime` with the same frames, to prove the two
//! arrivals land on identical output. The filter reads only the newest frame
//! of a fifteen-frame window, which is the shape the borrow exists for: the
//! list world copies every frame of every call in, the borrowed one hands
//! over just what is fetched.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, OnceLock};

use ffrwd_wasm_runtime::runtime::{
    describe, Filter, Format, Frame, Media, StreamInfo, TimeBase, VideoFormat,
};

const WIDTH: u32 = 8;
const HEIGHT: u32 = 8;
const FRAME_LEN: usize = (WIDTH * HEIGHT * 4) as usize;
const WINDOW: usize = 15;
const FRAMES: i64 = 25;

fn sidecar_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("runtime/ has a parent directory")
        .to_path_buf()
}

/// Builds both halves of the pair for wasm32-wasip2, once per test binary.
/// `modules/` is a separate cargo workspace with its own build lock, so this
/// does not deadlock against the `cargo test` run driving this binary.
fn built(artifact: &'static str) -> PathBuf {
    static BUILT: OnceLock<()> = OnceLock::new();
    BUILT.get_or_init(|| {
        let output = Command::new("cargo")
            .args([
                "build",
                "--release",
                "--target",
                "wasm32-wasip2",
                "-p",
                "newest-list",
                "-p",
                "newest-borrow",
            ])
            .current_dir(sidecar_root().join("modules"))
            .output()
            .expect("spawn cargo build for the newest pair");
        assert!(
            output.status.success(),
            "building the newest pair failed (status {:?}):\n{}",
            output.status.code(),
            String::from_utf8_lossy(&output.stderr)
        );
    });
    sidecar_root().join(format!(
        "modules/target/wasm32-wasip2/release/{artifact}.wasm"
    ))
}

fn format() -> Format {
    Format {
        media: Media::Video(VideoFormat {
            width: WIDTH,
            height: HEIGHT,
            pix_fmt: "rgba",
            frame_len: FRAME_LEN,
            color: None,
        }),
        time_base: TimeBase { num: 1, den: 25 },
    }
}

fn stream() -> StreamInfo {
    StreamInfo {
        index: 0,
        kind: "video".to_string(),
        codec: "rawvideo".to_string(),
        duration: None,
        tags: Vec::new(),
    }
}

/// A frame whose every byte names its timestamp.
fn frame(pts: i64) -> Frame {
    Frame {
        pts,
        data: Arc::new(vec![pts as u8; FRAME_LEN]),
        rows: Vec::new(),
    }
}

/// Drives one module over the stream the way the host cuts it: window 15,
/// stride 1, then whatever is left as the final call.
fn drive(module: &Path) -> Vec<(i64, Vec<u8>, Vec<String>)> {
    let mut filter =
        Filter::open(&module.display().to_string(), &format(), &stream(), "{}").expect("open");

    let mut out = Vec::new();
    let mut buffered: Vec<Frame> = Vec::new();
    for pts in 0..FRAMES {
        buffered.push(frame(pts));
        if buffered.len() == WINDOW {
            let produced = filter
                .process_window(&buffered, &[], false)
                .expect("a full window");
            for f in produced.frames {
                out.push((f.pts, f.data.as_ref().clone(), f.rows));
            }
            buffered.remove(0);
        }
    }
    let produced = filter
        .process_window(&buffered, &[], true)
        .expect("the final call");
    for f in produced.frames {
        out.push((f.pts, f.data.as_ref().clone(), f.rows));
    }
    assert!(produced.trailing.is_empty(), "no rows were left to trail");
    out
}

#[test]
fn the_borrowed_window_and_the_list_land_on_identical_output() {
    let list = built("newest_list");
    let borrow = built("newest_borrow");

    let from_list = drive(&list);
    let from_borrow = drive(&borrow);

    // Every frame leaves exactly once, when it is newest in a full window.
    let expected: Vec<i64> = ((WINDOW as i64 - 1)..FRAMES).collect();
    assert_eq!(
        from_list.iter().map(|(pts, _, _)| *pts).collect::<Vec<_>>(),
        expected
    );
    assert_eq!(from_list, from_borrow);

    // The row proves the fetched bytes were the newest frame's.
    let (pts, data, rows) = &from_list[0];
    assert_eq!(data, &vec![*pts as u8; FRAME_LEN]);
    assert_eq!(
        rows,
        &vec![format!("{{\"sum\":{}}}", *pts as u64 * FRAME_LEN as u64)]
    );
}

#[test]
fn the_pair_describes_the_same_shape_in_two_worlds() {
    let list = describe(&built("newest_list").display().to_string()).expect("describe newest-list");
    let borrow =
        describe(&built("newest_borrow").display().to_string()).expect("describe newest-borrow");

    assert_eq!(list.world, "0.10.0");
    assert_eq!(borrow.world, "0.13.0");
    assert_eq!(list.shape, borrow.shape);
    assert_eq!(list.inputs, borrow.inputs);
    assert_eq!(list.meta.pixel_formats, borrow.meta.pixel_formats);
}
