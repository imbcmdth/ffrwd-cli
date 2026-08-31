//! A module reading several streams, driven straight through
//! `ffrwd_wasm_runtime::runtime` with no network and no ffmpeg.
//!
//! The network buffers per pad and matches by timestamp before it ever calls
//! in here; these are the rules the runtime holds a caller to whatever drove
//! it, so the contract does not rest on one caller getting it right.

use std::path::PathBuf;
use std::process::Command;
use std::sync::OnceLock;

use ffrwd_wasm_runtime::runtime::{
    Filter, Format, Frame, Media, StreamInfo, TimeBase, VideoFormat,
};

const WIDTH: u32 = 8;
const HEIGHT: u32 = 8;
const FRAME_LEN: usize = (WIDTH * HEIGHT * 4) as usize;

fn sidecar_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("runtime/ has a parent directory")
        .to_path_buf()
}

/// Builds the two-pad fixture for wasm32-wasip2, once per test binary.
/// `modules/` is a separate cargo workspace with its own build lock, so this
/// does not deadlock against the `cargo test` run driving this binary.
fn pick_first() -> &'static PathBuf {
    static BUILT: OnceLock<PathBuf> = OnceLock::new();
    BUILT.get_or_init(|| {
        let output = Command::new("cargo")
            .args([
                "build",
                "--release",
                "--target",
                "wasm32-wasip2",
                "-p",
                "pick-first",
            ])
            .current_dir(sidecar_root().join("modules"))
            .output()
            .expect("spawn cargo build for pick-first");
        assert!(
            output.status.success(),
            "building pick-first failed (status {:?}):\n{}",
            output.status.code(),
            String::from_utf8_lossy(&output.stderr)
        );
        sidecar_root().join("modules/target/wasm32-wasip2/release/pick_first.wasm")
    })
}

fn format() -> Format {
    Format {
        media: Media::Video(VideoFormat {
            width: WIDTH,
            height: HEIGHT,
            pix_fmt: "rgba",
            frame_len: FRAME_LEN,
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

/// A frame whose every byte says which pad and timestamp it came from.
fn frame(pts: i64, fill: u8, rows: Vec<String>) -> Frame {
    Frame {
        pts,
        data: vec![fill; FRAME_LEN].into(),
        rows,
    }
}

fn open() -> Filter {
    Filter::open(&pick_first().to_string_lossy(), &format(), &stream(), "")
        .expect("opening pick_first")
}

#[test]
fn a_call_carries_one_frame_per_pad_in_pad_order() {
    let mut filter = open();
    assert_eq!(filter.inputs(), 2, "pick_first reads two streams");

    let pads = [frame(0, 11, Vec::new()), frame(0, 22, Vec::new())];
    let out = filter
        .process_window(&pads, &[], false)
        .expect("two pads at one timestamp");

    assert_eq!(out.frames.len(), 1, "one frame back per call");
    assert_eq!(
        *out.frames[0].data,
        vec![11u8; FRAME_LEN],
        "pad 0 is the one the module kept, so pad order is what it says it is"
    );
}

#[test]
fn pads_at_different_timestamps_are_refused_naming_the_module_and_the_pad() {
    let mut filter = open();
    let pads = [frame(0, 11, Vec::new()), frame(3, 22, Vec::new())];
    let err = filter
        .process_window(&pads, &[], false)
        .expect_err("the pads carry different timestamps");
    let message = err.to_string();
    assert!(message.contains("pick_first"), "got: {message}");
    assert!(
        message.contains("pad 0") && message.contains("pad 1"),
        "both pads are named: {message}"
    );
    assert!(message.contains("same timestamp"), "got: {message}");
}

#[test]
fn a_call_short_of_a_pad_is_refused_naming_the_module() {
    let mut filter = open();
    let err = filter
        .process_window(&[frame(0, 11, Vec::new())], &[], false)
        .expect_err("one frame is not two pads");
    let message = err.to_string();
    assert!(message.contains("pick_first"), "got: {message}");
    assert!(
        message.contains("1 frame(s) for the 2 stream(s)"),
        "got: {message}"
    );
}

#[test]
fn rows_ride_pad_zero_and_the_other_pads_carry_none() {
    // The host clears them before the call, so what a module sees on pad 0 is
    // all there is. Driven here with pad 1 already empty, which is what the
    // module is promised.
    let mut filter = open();
    let pads = [
        frame(0, 11, vec![r#"{"from":"pad0"}"#.to_string()]),
        frame(0, 22, Vec::new()),
    ];
    let out = filter
        .process_window(&pads, &[], false)
        .expect("two pads at one timestamp");

    // pick_first reports what it was handed, then passes pad 0's rows on.
    assert_eq!(
        out.frames[0].rows,
        vec![
            r#"{"pads":2,"pts":0,"rows-in":1}"#.to_string(),
            r#"{"from":"pad0"}"#.to_string(),
        ],
        "the module saw one row, and it was pad 0's"
    );
}

#[test]
fn the_final_call_of_a_multi_pad_module_carries_nothing() {
    // Window 1 and stride 1 leave nothing over, so the last call is empty and
    // the module has nothing to hand back.
    let mut filter = open();
    filter
        .process_window(
            &[frame(0, 11, Vec::new()), frame(0, 22, Vec::new())],
            &[],
            false,
        )
        .expect("one pair");
    let out = filter
        .process_window(&[], &[], true)
        .expect("the final call");
    assert!(out.frames.is_empty(), "nothing was left over");
    assert!(out.trailing.is_empty());
}
