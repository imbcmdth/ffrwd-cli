//! Real frames through `segment` and on into `mask_select`, driven directly
//! through `ffrwd_wasm_runtime::runtime` with no ffmpeg involved.
//!
//! The graph is not in git and the ONNX Runtime that loads it is fetched, so
//! this skips - loudly - when either is absent.
//! `sidecar/modules/segment/fetch-model.ps1` downloads the graph and
//! `ffrwd setup nn` the runtime.
//!
//! The picture is not in git either: a segmenter proves nothing against
//! shapes drawn by hand, so `FFRWD_SEGMENT_FRAME` names one raw yuv420p
//! 640x640 frame with something in it that the model knows. ffmpeg writes
//! one:
//!
//!     ffmpeg -i photo.jpg -frames:v 1 -vf scale=640:640 \
//!       -f rawvideo -pix_fmt yuv420p frame.yuv

use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

use ffrwd_wasm_runtime::nn;
use ffrwd_wasm_runtime::runtime::{
    Filter, Format, Frame, Media, StreamInfo, TimeBase, VideoFormat,
};

/// The name the module asks the host for.
const MODEL: &str = "segment";

/// The square the graph runs at, so a frame of this size is letterboxed by
/// nothing and every id lands where the graph put it.
const WIDTH: u32 = 640;
const HEIGHT: u32 = 640;

/// What `mask_select` writes for a selected pixel and a rejected one.
const KEEP: u8 = 255;
const DROP: u8 = 0;

/// The sidecar's directory, the parent of `runtime/`.
fn sidecar_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("runtime/ has a parent directory")
        .to_path_buf()
}

/// Where `fetch-model.ps1` puts the graph.
fn model_path() -> PathBuf {
    sidecar_root().join("modules/segment/model/yolov8n-seg.onnx")
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

/// Builds one module for wasm32-wasip2 and returns its component.
/// `modules/` is a separate cargo workspace with its own build lock, so this
/// does not deadlock against the `cargo test` run driving this binary.
fn build_module(name: &str) -> PathBuf {
    let output = Command::new("cargo")
        .args([
            "build",
            "--release",
            "--target",
            "wasm32-wasip2",
            "-p",
            name,
        ])
        .current_dir(sidecar_root().join("modules"))
        .output()
        .unwrap_or_else(|e| panic!("spawn cargo build for {name}: {e}"));
    assert!(
        output.status.success(),
        "building {name} failed (status {:?}):\n{}",
        output.status.code(),
        String::from_utf8_lossy(&output.stderr)
    );
    sidecar_root().join(format!("modules/target/wasm32-wasip2/release/{name}.wasm"))
}

/// The environment variable naming a real frame to run the graph over.
const FRAME_VAR: &str = "FFRWD_SEGMENT_FRAME";

/// The frame that variable names, read and checked for size.
fn frame_data() -> Result<Vec<u8>, String> {
    let Some(path) = std::env::var_os(FRAME_VAR).map(PathBuf::from) else {
        return Err(format!(
            "{FRAME_VAR} is not set. Name one raw yuv420p {WIDTH}x{HEIGHT} frame \
             with something in it the model knows; a segmenter proves nothing \
             against shapes drawn by hand."
        ));
    };
    let data = std::fs::read(&path).map_err(|e| format!("{FRAME_VAR}={}: {e}", path.display()))?;
    let wanted = (WIDTH * HEIGHT) as usize * 3 / 2;
    if data.len() != wanted {
        return Err(format!(
            "{FRAME_VAR}={} is {} bytes; one raw yuv420p {WIDTH}x{HEIGHT} frame is {wanted}",
            path.display(),
            data.len()
        ));
    }
    Ok(data)
}

/// The two built modules, or why neither happened. The models are
/// process-global and load once, so every test here shares one answer.
fn configured() -> &'static Result<(PathBuf, PathBuf), String> {
    static READY: OnceLock<Result<(PathBuf, PathBuf), String>> = OnceLock::new();
    READY.get_or_init(|| {
        let model = model_path();
        if !model.is_file() {
            return Err("segment's graph is absent. Run \
                        sidecar/modules/segment/fetch-model.ps1 to download it."
                .to_string());
        }
        let Some(dir) = runtime_dir() else {
            return Err("segment needs an ONNX Runtime. Run \
                        `ffrwd setup nn` to download one."
                .to_string());
        };
        let target = match nn::Target::from_env() {
            Ok(Some(target)) => target,
            _ => nn::Target::Cpu,
        };
        nn::configure(&nn::Config {
            models: vec![nn::ModelSpec {
                name: MODEL.to_string(),
                path: model,
            }],
            runtime_dir: Some(dir),
            target,
            exclude: Vec::new(),
        })
        .map_err(|e| format!("loading the segment graph: {e}"))?;
        Ok((build_module("segment"), build_module("mask_select")))
    })
}

/// The built modules and the frame to run them over, or `None` after saying
/// out loud why the test did nothing.
fn ready_or_skip() -> Option<(&'static PathBuf, &'static PathBuf, Vec<u8>)> {
    let modules = match configured() {
        Ok(modules) => modules,
        Err(why) => {
            eprintln!("SKIPPED: {why}");
            return None;
        }
    };
    match frame_data() {
        Ok(frame) => Some((&modules.0, &modules.1, frame)),
        Err(why) => {
            eprintln!("SKIPPED: {why}");
            None
        }
    }
}

/// yuv420p geometry, one tick to the frame at 25 fps.
fn format() -> Format {
    Format {
        media: Media::Video(VideoFormat {
            width: WIDTH,
            height: HEIGHT,
            pix_fmt: "yuv420p",
            frame_len: (WIDTH * HEIGHT) as usize * 3 / 2,
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

/// One frame through a module, as the one-to-one window-1 shape it declares.
fn one_frame(filter: &mut Filter, frame: Frame) -> Frame {
    let out = filter
        .process_window(std::slice::from_ref(&frame), &[], false)
        .expect("one frame through the module");
    assert_eq!(out.frames.len(), 1, "one frame in, one out");
    out.frames.into_iter().next().expect("the one frame")
}

fn open(module: &Path) -> Filter {
    Filter::open(&module.to_string_lossy(), &format(), &stream(), "")
        .unwrap_or_else(|e| panic!("opening {}: {e}", module.display()))
}

/// Every row's `id`, in the order they arrived.
fn row_ids(rows: &[String]) -> Vec<i64> {
    rows.iter()
        .map(|row| {
            let value: serde_json::Value =
                serde_json::from_str(row).unwrap_or_else(|e| panic!("row {row} is not JSON: {e}"));
            value
                .get("id")
                .and_then(|v| v.as_i64())
                .unwrap_or_else(|| panic!("row {row} carries no id"))
        })
        .collect()
}

#[test]
fn an_index_map_paints_only_the_ids_its_rows_declare() {
    let Some((segment, _, scene)) = ready_or_skip() else {
        return;
    };
    let mut filter = open(segment);
    let map = one_frame(
        &mut filter,
        Frame {
            pts: 0,
            data: scene.into(),
            rows: Vec::new(),
        },
    );

    let pixels = (WIDTH * HEIGHT) as usize;
    assert_eq!(map.pts, 0, "at the timestamp it arrived on");
    assert_eq!(map.data.len(), pixels * 3 / 2, "one yuv420p frame");
    assert!(
        map.data[pixels..].iter().all(|v| *v == 128),
        "an index map carries no colour"
    );

    let ids = row_ids(&map.rows);
    assert!(
        !ids.is_empty(),
        "the named frame is meant to carry something the model knows, and \
         nothing was found in it"
    );
    assert_eq!(
        ids,
        (1..=ids.len() as i64).collect::<Vec<_>>(),
        "ids are one-based and run without a gap: {ids:?}"
    );

    // Every id the map paints has a row, and 0 is the background.
    let painted: BTreeSet<u8> = map.data[..pixels].iter().copied().collect();
    let declared: BTreeSet<u8> = ids
        .iter()
        .map(|id| u8::try_from(*id).expect("an id fits a luma byte"))
        .chain(std::iter::once(0))
        .collect();
    assert!(
        painted.is_subset(&declared),
        "the map paints {painted:?} and the rows declare {declared:?}"
    );

    // And each id sits inside the box its own row reports.
    let boxes: HashMap<i64, (i64, i64, i64, i64)> = map
        .rows
        .iter()
        .map(|row| {
            let v: serde_json::Value = serde_json::from_str(row).expect("a row is JSON");
            let at = |k: &str| v.get(k).and_then(|n| n.as_i64()).expect("a box coordinate");
            (at("id"), (at("x"), at("y"), at("w"), at("h")))
        })
        .collect();
    for y in 0..HEIGHT as usize {
        for x in 0..WIDTH as usize {
            let id = i64::from(map.data[y * WIDTH as usize + x]);
            if id == 0 {
                continue;
            }
            let (bx, by, bw, bh) = boxes[&id];
            assert!(
                (bx..bx + bw).contains(&(x as i64)) && (by..by + bh).contains(&(y as i64)),
                "id {id} is painted at ({x}, {y}), outside its box ({bx}, {by}, {bw}, {bh})"
            );
        }
    }
}

#[test]
fn a_mask_is_exactly_the_ids_the_rows_that_reached_it_name() {
    let Some((segment, mask_select, scene)) = ready_or_skip() else {
        return;
    };
    let mut filter = open(segment);
    let map = one_frame(
        &mut filter,
        Frame {
            pts: 0,
            data: scene.into(),
            rows: Vec::new(),
        },
    );
    let ids = row_ids(&map.rows);
    assert!(
        !ids.is_empty(),
        "a mask of nothing at all proves nothing; the named frame must carry \
         something the model knows"
    );

    // Every other row kept, which is what a predicate upstream would have
    // left: the mask must follow the rows it was handed and nothing else.
    let kept: Vec<String> = map
        .rows
        .iter()
        .enumerate()
        .filter(|(index, _)| index % 2 == 0)
        .map(|(_, row)| row.clone())
        .collect();
    let selected: BTreeSet<u8> = row_ids(&kept)
        .iter()
        .map(|id| u8::try_from(*id).expect("an id fits a luma byte"))
        .collect();

    let mut selector = open(mask_select);
    let mask = one_frame(
        &mut selector,
        Frame {
            pts: 0,
            data: map.data.clone(),
            rows: kept,
        },
    );

    let pixels = (WIDTH * HEIGHT) as usize;
    assert_eq!(mask.pts, 0);
    assert_eq!(mask.data.len(), pixels * 3 / 2);
    assert!(
        mask.data[pixels..].iter().all(|v| *v == 128),
        "a mask carries no colour"
    );
    assert!(
        mask.rows.is_empty(),
        "the rows were the selection, and none travel on"
    );

    for (index, (out, id)) in mask.data[..pixels]
        .iter()
        .zip(&map.data[..pixels])
        .enumerate()
    {
        let want = if selected.contains(id) { KEEP } else { DROP };
        assert_eq!(
            *out, want,
            "pixel {index} carries id {id}, and the mask should be {want}"
        );
    }

    // A frame carrying at least one instance is what makes the above more
    // than two empty planes agreeing.
    eprintln!(
        "segment found {} instance(s); {} id(s) selected",
        ids.len(),
        selected.len()
    );
}
