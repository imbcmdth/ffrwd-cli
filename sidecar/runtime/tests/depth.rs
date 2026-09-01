//! Real frames through the depth module, driven directly through
//! `ffrwd_wasm_runtime::runtime` with no ffmpeg involved.
//!
//! The graph is not in git and the ONNX Runtime that loads it is fetched, so
//! this skips - loudly - when either is absent.
//! `sidecar/modules/depth/fetch-model.ps1` downloads the graph and
//! `ffrwd setup nn` the runtime.

use std::path::PathBuf;
use std::process::Command;
use std::sync::OnceLock;
use std::time::Instant;

use ffrwd_wasm_runtime::nn;
use ffrwd_wasm_runtime::runtime::{
    Filter, Format, Frame, Media, StreamInfo, TimeBase, VideoFormat,
};

/// The name the module asks the host for.
const MODEL: &str = "depth";

/// A frame small enough to build by hand and large enough that a depth map
/// over it has somewhere to vary.
const WIDTH: u32 = 64;
const HEIGHT: u32 = 64;

/// The sidecar's directory, the parent of `runtime/`.
fn sidecar_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("runtime/ has a parent directory")
        .to_path_buf()
}

/// Where `fetch-model.ps1` puts the graph.
fn model_path() -> PathBuf {
    sidecar_root().join("modules/depth/model/model.onnx")
}
/// Where `ffrwd setup nn` puts a runtime: the user's cache, keyed by the
/// version this build demands and the platform it runs on.
fn cached_runtime_dir() -> Option<PathBuf> {
    let home = std::env::var_os(if cfg!(windows) { "USERPROFILE" } else { "HOME" })?;
    Some(
        PathBuf::from(home)
            .join(".cache/ffrwd/nn-runtime")
            .join(nn::ort_version())
            .join(nn::platform()),
    )
}

/// The runtime directory the environment names, else the cached one.
fn runtime_dir() -> Option<PathBuf> {
    let named = std::env::var_os(nn::RUNTIME_DIR_VAR).map(PathBuf::from);
    let candidates = [named, cached_runtime_dir()];
    candidates.into_iter().flatten().find(|dir| {
        [
            "onnxruntime.dll",
            "libonnxruntime.so",
            "libonnxruntime.dylib",
        ]
        .iter()
        .any(|lib| dir.join(lib).is_file())
    })
}

/// Builds the depth module for wasm32-wasip2 and returns its component.
/// `modules/` is a separate cargo workspace with its own build lock, so this
/// does not deadlock against the `cargo test` run driving this binary.
fn build_module() -> PathBuf {
    let output = Command::new("cargo")
        .args([
            "build",
            "--release",
            "--target",
            "wasm32-wasip2",
            "-p",
            "depth",
        ])
        .current_dir(sidecar_root().join("modules"))
        .output()
        .expect("spawn cargo build for depth");
    assert!(
        output.status.success(),
        "building depth failed (status {:?}):\n{}",
        output.status.code(),
        String::from_utf8_lossy(&output.stderr)
    );
    sidecar_root().join("modules/target/wasm32-wasip2/release/depth.wasm")
}

/// Binds the graph and builds the module, or says why neither happened. The
/// models are process-global and load once, so every test here shares one
/// answer.
fn configured() -> &'static Result<PathBuf, String> {
    static READY: OnceLock<Result<PathBuf, String>> = OnceLock::new();
    READY.get_or_init(|| {
        let model = model_path();
        if !model.is_file() {
            return Err("depth's graph is absent. Run \
                        sidecar/modules/depth/fetch-model.ps1 to download it."
                .to_string());
        }
        let Some(dir) = runtime_dir() else {
            return Err("depth needs an ONNX Runtime. Run \
                        `ffrwd setup nn` to download one."
                .to_string());
        };
        let target = match std::env::var("FFRWD_NN_DEPTH_TARGET").as_deref() {
            Ok("gpu") => nn::Target::Gpu,
            _ => nn::Target::Cpu,
        };
        nn::configure(&nn::Config {
            models: vec![nn::ModelSpec {
                name: MODEL.to_string(),
                path: model,
            }],
            runtime_dir: Some(dir),
            target,
        })
        .map_err(|e| format!("loading the depth graph: {e}"))?;
        Ok(build_module())
    })
}

/// The built module, or `None` after saying out loud why the test did
/// nothing.
fn module_or_skip() -> Option<&'static PathBuf> {
    match configured() {
        Ok(module) => Some(module),
        Err(why) => {
            eprintln!("SKIPPED: {why}");
            None
        }
    }
}

/// rgba geometry, one tick to the frame at 25 fps.
fn format() -> Format {
    Format {
        media: Media::Video(VideoFormat {
            width: WIDTH,
            height: HEIGHT,
            pix_fmt: "rgba",
            frame_len: (WIDTH * HEIGHT * 4) as usize,
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

/// A frame with structure in it: a bright disc on a dark ground, which any
/// depth model reads as something in front of something else.
fn a_scene() -> Vec<u8> {
    let (w, h) = (WIDTH as usize, HEIGHT as usize);
    let mut frame = vec![0u8; w * h * 4];
    for y in 0..h {
        for x in 0..w {
            let dx = x as f32 - w as f32 / 2.0;
            let dy = y as f32 - h as f32 / 2.0;
            let inside = (dx * dx + dy * dy).sqrt() < w as f32 / 4.0;
            let base = (y * w + x) * 4;
            let (r, g, b) = if inside {
                (230, 200, 160)
            } else {
                // A vertical gradient behind it, so the ground is not flat.
                let shade = (30 + y * 120 / h) as u8;
                (shade, shade, shade + 20)
            };
            frame[base] = r;
            frame[base + 1] = g;
            frame[base + 2] = b;
            frame[base + 3] = 255;
        }
    }
    frame
}

#[test]
fn a_real_frame_comes_back_as_a_depth_map_with_range_in_it() {
    let Some(module) = module_or_skip() else {
        return;
    };
    let format = format();
    let mut filter =
        Filter::open(&module.to_string_lossy(), &format, &stream(), "").expect("opening depth");

    let frame = Frame {
        pts: 0,
        data: a_scene().into(),
        rows: Vec::new(),
    };
    let started = Instant::now();
    let out = filter
        .process_window(std::slice::from_ref(&frame), &[], false)
        .expect("one frame through depth");
    let elapsed = started.elapsed();

    assert_eq!(out.frames.len(), 1, "one frame in, one out");
    let map = &out.frames[0];
    assert_eq!(map.pts, 0, "and at the timestamp it arrived on");
    assert_eq!(
        map.data.len(),
        frame.data.len(),
        "a depth map is a frame of the geometry it was opened for"
    );

    // Greyscale: equal in every channel, and opaque.
    let (pixels, _) = map.data.as_chunks::<4>();
    for pixel in pixels {
        assert_eq!(
            (pixel[0], pixel[1], pixel[3]),
            (pixel[1], pixel[2], 255),
            "a depth map is grey and opaque"
        );
    }

    // Normalized per frame, so the range is the whole scale rather than
    // whatever the graph's raw units happened to be.
    let greys: Vec<u8> = map.data.iter().step_by(4).copied().collect();
    let low = *greys.iter().min().expect("pixels");
    let high = *greys.iter().max().expect("pixels");
    assert_eq!((low, high), (0, 255), "min-max normalized onto the byte");
    let distinct = greys.iter().collect::<std::collections::HashSet<_>>().len();
    assert!(
        distinct > 16,
        "a scene with a disc in front of a gradient is not a step: {distinct} level(s)"
    );

    eprintln!("depth: {:.0} ms for the first frame", elapsed.as_millis());
}

#[test]
fn the_map_moves_when_the_scene_does() {
    let Some(module) = module_or_skip() else {
        return;
    };
    let format = format();
    let mut filter =
        Filter::open(&module.to_string_lossy(), &format, &stream(), "").expect("opening depth");

    let scene = Frame {
        pts: 0,
        data: a_scene().into(),
        rows: Vec::new(),
    };
    // The same geometry with nothing in it: one flat wall, which cannot read
    // as the same depth map as a disc in front of a gradient.
    let flat = Frame {
        pts: 1,
        data: vec![128u8; (WIDTH * HEIGHT * 4) as usize].into(),
        rows: Vec::new(),
    };

    let first = filter
        .process_window(std::slice::from_ref(&scene), &[], false)
        .expect("the scene");
    let second = filter
        .process_window(std::slice::from_ref(&flat), &[], true)
        .expect("the flat wall");

    assert_ne!(
        first.frames[0].data, second.frames[0].data,
        "a different picture is a different depth map"
    );
}
