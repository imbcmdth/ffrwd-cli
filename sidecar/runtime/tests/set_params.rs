//! `Filter::set_params` coverage, driven directly through
//! `ffrwd_wasm_runtime::runtime` with no ffmpeg involved.

use std::path::PathBuf;
use std::process::Command;
use std::sync::OnceLock;

use ffrwd_wasm_runtime::runtime::{
    Filter, Format, Frame, Media, StreamInfo, TimeBase, VideoFormat,
};

const WIDTH: u32 = 8;
const HEIGHT: u32 = 8;
const PIX_FMT: &str = "rgba";
const FRAME_LEN: usize = (WIDTH * HEIGHT * 4) as usize;

/// 25fps, which is what the seconds an adapted module is told are counted in.
const FORMAT: Format = Format {
    media: Media::Video(VideoFormat {
        width: WIDTH,
        height: HEIGHT,
        pix_fmt: PIX_FMT,
        frame_len: FRAME_LEN,
    }),
    time_base: TimeBase { num: 1, den: 25 },
};

/// Repo root, the parent of `runtime/`.
fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("host/ has a parent directory")
        .to_path_buf()
}

/// Absolute path to a module's built `.wasm` component.
fn module_path(name: &str) -> PathBuf {
    repo_root()
        .join("modules/target/wasm32-wasip2/release")
        .join(format!("{name}.wasm"))
}

/// Builds the wasm modules for wasm32-wasip2, once per test binary.
/// `modules/` is a separate cargo workspace with its own build lock, so this
/// does not deadlock against the `cargo test` run driving this binary.
fn ensure_modules_built() {
    static BUILT: OnceLock<()> = OnceLock::new();
    BUILT.get_or_init(|| {
        let output = Command::new("cargo")
            .args(["build", "--release", "--target", "wasm32-wasip2"])
            .current_dir(repo_root().join("modules"))
            .output()
            .expect("spawn cargo build for modules");
        assert!(
            output.status.success(),
            "building modules failed (status {:?}):\n{}",
            output.status.code(),
            String::from_utf8_lossy(&output.stderr)
        );
    });
}

/// One frame through a window-1 module, returning the pixels it produced.
fn one_frame(filter: &mut Filter, pts: i64, frame: &[u8]) -> Vec<u8> {
    let window = [Frame {
        pts,
        data: frame.to_vec(),
        rows: Vec::new(),
    }];
    let out = filter
        .process_window(&window, &[], false)
        .unwrap_or_else(|e| panic!("processing the frame at pts {pts}: {e}"));
    assert_eq!(out.frames.len(), 1, "a one-to-one module returns one frame");
    out.frames[0].data.clone()
}

#[test]
#[ignore = "needs the trails module (stateful decay params), not vendored here"]
fn set_params_keeps_state_and_rejects_bad_json() {
    ensure_modules_built();
    let module = module_path("trails");
    let module_str = module.to_str().expect("module path is valid UTF-8");

    let stream = StreamInfo::default();
    let mut filter =
        Filter::open(module_str, &FORMAT, &stream, r#"{"decay":0.8}"#).expect("opening trails");

    let white = vec![255u8; FRAME_LEN];
    let black = vec![0u8; FRAME_LEN];

    one_frame(&mut filter, 0, &white);

    assert_eq!(
        one_frame(&mut filter, 1, &black),
        vec![204u8; FRAME_LEN],
        "decay 0.8 fading the previous white frame (255 * 0.8 = 204)"
    );

    filter
        .set_params(r#"{"decay":0.5}"#)
        .expect("set_params to decay 0.5");

    assert_eq!(
        one_frame(&mut filter, 2, &black),
        vec![102u8; FRAME_LEN],
        "decay 0.5 fading the previous 204 (204 * 0.5 = 102)"
    );

    let rejected = filter.set_params("not json");
    assert!(
        rejected.is_err(),
        "malformed JSON must be rejected by set_params"
    );

    assert_eq!(
        one_frame(&mut filter, 3, &black),
        vec![51u8; FRAME_LEN],
        "decay is still 0.5 after the rejected set_params (102 * 0.5 = 51)"
    );
}
