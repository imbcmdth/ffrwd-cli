//! Integration tests for the `ffrwd-wasm` binary. The NUT on its stdin and
//! stdout is written and read by this crate's own wire, no ffmpeg involved,
//! so these stay fast and deterministic. That real ffmpeg reads and writes
//! the same wire is what `ffmpeg.rs` is for.

use std::env;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Output, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

use ffrwd_wasm::nut::{Demuxer, Muxer, Stream, TimeBase};

const WIDTH: u32 = 8;
const HEIGHT: u32 = 8;
const FRAME_LEN: usize = (WIDTH * HEIGHT * 4) as usize;

/// The time base ffmpeg gives a NUT stream by default, and the step one
/// frame takes at 25fps in it.
const TIME_BASE: TimeBase = TimeBase { num: 1, den: 65536 };
const PTS_STEP: i64 = 65536 / 25;

/// Sidecar root, the parent of `ffrwd-wasm/`.
fn sidecar_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("ffrwd-wasm/ has a parent directory")
        .to_path_buf()
}

/// The cargo workspace every module is built in: the sidecar's own fleet.
fn module_workspace() -> PathBuf {
    sidecar_root().join("modules")
}

/// Absolute path to a module's built `.wasm` component.
fn module_path(name: &str) -> PathBuf {
    module_workspace()
        .join("target/wasm32-wasip2/release")
        .join(format!("{name}.wasm"))
}

/// Builds the wasm modules for wasm32-wasip2, once per test binary.
/// The workspace is separate from the one driving this binary and has its own
/// build lock, so this does not deadlock against the `cargo test` run.
fn ensure_modules_built() {
    static BUILT: OnceLock<()> = OnceLock::new();
    BUILT.get_or_init(|| {
        let workspace = module_workspace();
        let output = Command::new("cargo")
            .args(["build", "--release", "--target", "wasm32-wasip2"])
            .current_dir(&workspace)
            .output()
            .expect("spawn cargo build for modules");
        assert!(
            output.status.success(),
            "building {} failed (status {:?}):\n{}",
            workspace.display(),
            output.status.code(),
            String::from_utf8_lossy(&output.stderr)
        );
    });
}

fn a_stream() -> Stream {
    Stream::video("rgba", WIDTH, HEIGHT, TIME_BASE).expect("rgba is carried")
}

/// Frame `index` filled with a byte pattern distinct per position and per
/// frame: `(index * 41 + offset) % 256`.
fn synthetic_frame(index: u8) -> Vec<u8> {
    (0..FRAME_LEN)
        .map(|offset| index.wrapping_mul(41).wrapping_add(offset as u8))
        .collect()
}

/// A NUT stream carrying `frames`, one every `PTS_STEP` from zero.
fn nut_stream(frames: &[Vec<u8>]) -> Vec<u8> {
    let mut wire = Vec::new();
    {
        let mut muxer = Muxer::new(&mut wire, &a_stream()).expect("write NUT headers");
        for (i, frame) in frames.iter().enumerate() {
            muxer
                .write_frame(i as i64 * PTS_STEP, frame)
                .expect("write NUT frame");
        }
        muxer.finish().expect("finish the NUT stream");
    }
    wire
}

/// Every frame in a NUT stream, with the PTS it carries.
fn read_nut(wire: &[u8]) -> Vec<(i64, Vec<u8>)> {
    let mut demuxer = Demuxer::open(wire).expect("read NUT headers");
    let mut frames = Vec::new();
    let mut buf = Vec::new();
    while let Some(pts) = demuxer.read_frame(&mut buf).expect("read a NUT frame") {
        frames.push((pts, buf.clone()));
    }
    frames
}

struct FfrwdWasmRun {
    stdout: Vec<u8>,
    stderr: String,
    output: Output,
}

impl FfrwdWasmRun {
    fn success(&self) -> bool {
        self.output.status.success()
    }

    /// The frames on stdout, with their timestamps.
    fn frames(&self) -> Vec<(i64, Vec<u8>)> {
        read_nut(&self.stdout)
    }
}

/// Runs `ffrwd-wasm` with the given argv (everything after the binary name),
/// feeding `stdin_bytes` and collecting stdout/stderr.
fn run_ffrwd_wasm(args: &[&str], stdin_bytes: &[u8]) -> FfrwdWasmRun {
    let exe = env!("CARGO_BIN_EXE_ffrwd-wasm");
    let mut cmd = Command::new(exe);
    cmd.args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = cmd.spawn().expect("spawn ffrwd-wasm");
    let mut stdin = child.stdin.take().expect("child stdin");
    // A refusal can close stdin before it is all written, which is a broken
    // pipe rather than a test failure.
    let _ = stdin.write_all(stdin_bytes);
    drop(stdin);
    let output = child.wait_with_output().expect("wait for ffrwd-wasm");
    FfrwdWasmRun {
        stdout: output.stdout.clone(),
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        output,
    }
}

/// Runs `ffrwd-wasm` on `module` with NUT on stdin and stdout, inserting
/// `module_args` between `-m module` and the output spec (e.g.
/// `["-jobs", "4"]`).
fn run_filter(module: &std::path::Path, module_args: &[&str], stdin_bytes: &[u8]) -> FfrwdWasmRun {
    ensure_modules_built();
    let module_str = module.to_str().expect("module path is valid UTF-8");
    let mut args: Vec<&str> = vec!["-f", "nut", "-i", "-", "-m", module_str];
    args.extend_from_slice(module_args);
    args.extend_from_slice(&["-f", "nut", "-"]);
    run_ffrwd_wasm(&args, stdin_bytes)
}

/// Runs `ffrwd-wasm` as a module network: one `-m name=path` per module, the
/// `-filter_complex` wiring them, then `extra` and `outputs` as written. NUT
/// on stdin throughout.
fn run_network(
    modules: &[(&str, &str)],
    wiring: &str,
    extra: &[&str],
    outputs: &[&str],
    stdin_bytes: &[u8],
) -> FfrwdWasmRun {
    ensure_modules_built();
    let mut owned: Vec<String> = ["-f", "nut", "-i", "-"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    for (name, module) in modules {
        owned.push("-m".to_string());
        let path = module_path(module);
        owned.push(format!(
            "{name}={}",
            path.to_str().expect("module path is valid UTF-8")
        ));
    }
    owned.push("-filter_complex".to_string());
    owned.push(wiring.to_string());
    owned.extend(extra.iter().map(|s| s.to_string()));
    owned.extend(outputs.iter().map(|s| s.to_string()));

    let args: Vec<&str> = owned.iter().map(String::as_str).collect();
    run_ffrwd_wasm(&args, stdin_bytes)
}

/// The one mapped NUT output on stdout every network test but the fan-out
/// writes.
const ONE_OUTPUT: &[&str] = &["-map", "[out0]", "-f", "nut", "-"];

fn assert_run_ok(run: &FfrwdWasmRun, what: &str) {
    assert!(
        run.success(),
        "{what} exited with {:?}\nstderr:\n{}",
        run.output.status.code(),
        run.stderr
    );
}

#[test]
fn invert_round_trips_frames() {
    let frames: Vec<Vec<u8>> = (0..3u8).map(synthetic_frame).collect();

    let run = run_filter(&module_path("invert"), &[], &nut_stream(&frames));
    assert_run_ok(&run, "ffrwd-wasm");

    let got = run.frames();
    assert_eq!(got.len(), 3, "expected exactly 3 frames back");

    for (i, input) in frames.iter().enumerate() {
        let (pts, output) = &got[i];
        assert_eq!(*pts, i as i64 * PTS_STEP, "frame {i} timestamp");
        for pixel in 0..(WIDTH * HEIGHT) as usize {
            let p = pixel * 4;
            assert_eq!(
                output[p],
                255 - input[p],
                "frame {i} pixel {pixel} red byte"
            );
            assert_eq!(
                output[p + 1],
                255 - input[p + 1],
                "frame {i} pixel {pixel} green byte"
            );
            assert_eq!(
                output[p + 2],
                255 - input[p + 2],
                "frame {i} pixel {pixel} blue byte"
            );
            assert_eq!(
                output[p + 3],
                input[p + 3],
                "frame {i} pixel {pixel} alpha byte must be unchanged"
            );
        }
    }
}

#[test]
fn the_stream_header_survives_the_hop() {
    let stream = Stream::video(
        "rgba",
        WIDTH,
        HEIGHT,
        TimeBase {
            num: 1001,
            den: 30000,
        },
    )
    .expect("rgba is carried");
    let mut wire = Vec::new();
    {
        let mut muxer = Muxer::new(&mut wire, &stream).expect("write NUT headers");
        muxer
            .write_frame(7, &synthetic_frame(0))
            .expect("write NUT frame");
        muxer.finish().expect("finish the NUT stream");
    }

    let run = run_filter(&module_path("invert"), &[], &wire);
    assert_run_ok(&run, "ffrwd-wasm");

    let demuxer = Demuxer::open(&run.stdout[..]).expect("read the output headers");
    assert_eq!(demuxer.stream().time_base, stream.time_base);
    assert_eq!(demuxer.stream().video_geometry(), Some((WIDTH, HEIGHT)));
    assert_eq!(demuxer.stream().pix_fmt(), Some("rgba"));
    assert_eq!(run.frames()[0].0, 7, "an odd timestamp rides through");
}

#[test]
fn jobs_preserve_frame_order() {
    let frame_count = 32u16;
    let frames: Vec<Vec<u8>> = (0..frame_count).map(|i| synthetic_frame(i as u8)).collect();
    let wire = nut_stream(&frames);

    let baseline = run_filter(&module_path("invert"), &["-jobs", "1"], &wire);
    assert_run_ok(&baseline, "jobs=1");
    assert_eq!(baseline.frames().len(), frame_count as usize);

    // 4 is the job count measured against; 6 is a non-power-of-two count that
    // does not evenly divide 32, so round-robin dispatch leaves a ragged
    // final round - reordering bugs hide behind neat divisions.
    for &jobs in &[4u16, 6] {
        let run = run_filter(&module_path("invert"), &["-jobs", &jobs.to_string()], &wire);
        assert_run_ok(&run, &format!("jobs={jobs}"));
        assert_eq!(
            run.stdout, baseline.stdout,
            "jobs={jobs} output must be byte-identical to jobs=1"
        );
        for (i, (pts, output)) in run.frames().iter().enumerate() {
            assert_eq!(*pts, i as i64 * PTS_STEP, "jobs={jobs} frame {i} timestamp");
            let expected_input = &frames[i];
            for pixel in 0..(WIDTH * HEIGHT) as usize {
                let p = pixel * 4;
                assert_eq!(
                    output[p],
                    255 - expected_input[p],
                    "jobs={jobs} frame {i} pixel {pixel} must derive from input frame {i}"
                );
            }
        }
    }
}

#[test]
fn a_truncated_frame_is_an_error() {
    let mut wire = nut_stream(&[synthetic_frame(0)]);
    wire.truncate(wire.len() - FRAME_LEN / 2);

    let run = run_filter(&module_path("invert"), &[], &wire);
    assert!(
        !run.success(),
        "a truncated frame should be a non-zero exit, got success with stdout len {}",
        run.stdout.len()
    );
    assert!(
        run.stderr.contains("ends inside a frame"),
        "expected an incomplete-frame message on stderr, got:\n{}",
        run.stderr
    );
}

#[test]
fn input_that_is_not_nut_is_an_error() {
    let run = run_filter(&module_path("invert"), &[], &synthetic_frame(0));
    assert!(!run.success(), "raw frames are no longer the wire");
    assert!(
        run.stderr.contains("not NUT"),
        "expected the error to say what the input is not, got:\n{}",
        run.stderr
    );
}

#[test]
fn a_missing_module_is_an_error() {
    let missing = sidecar_root().join("modules/target/wasm32-wasip2/release/does_not_exist.wasm");
    let run = run_filter(&missing, &[], &nut_stream(&[synthetic_frame(0)]));
    assert!(!run.success(), "a missing module should be a non-zero exit");
    assert!(
        run.stderr.contains(missing.to_str().unwrap()),
        "expected the missing path in the stderr message, got:\n{}",
        run.stderr
    );
}

#[test]
fn geometry_flags_are_refused() {
    ensure_modules_built();
    let module_str = module_path("invert")
        .to_str()
        .expect("module path is valid UTF-8")
        .to_string();

    for (flag, value) in [("-s", "8x8"), ("-r", "25"), ("-pix_fmt", "rgba")] {
        let run = run_ffrwd_wasm(
            &[
                "-f",
                "nut",
                flag,
                value,
                "-i",
                "-",
                "-m",
                &module_str,
                "-f",
                "nut",
                "-",
            ],
            &[],
        );
        assert!(!run.success(), "{flag} should no longer be accepted");
        assert!(
            run.stderr.contains(flag) && run.stderr.contains("stream header"),
            "expected {flag}'s refusal to name the stream header, got:\n{}",
            run.stderr
        );
    }
}

#[test]
fn unknown_flags_are_errors() {
    ensure_modules_built();
    let run = run_ffrwd_wasm(
        &[
            "-f",
            "nut",
            "-i",
            "-",
            "-m",
            module_path("invert").to_str().unwrap(),
            "-frobnicate",
            "yes",
            "-f",
            "nut",
            "-",
        ],
        &[],
    );
    assert!(
        !run.success(),
        "a misspelled flag should be a non-zero exit"
    );
    assert!(
        run.stderr.contains("-frobnicate"),
        "expected the offending flag named in the error, got:\n{}",
        run.stderr
    );
}

#[test]
fn an_unsupported_edge_format_is_an_error() {
    ensure_modules_built();
    let run = run_ffrwd_wasm(
        &[
            "-f",
            "rawvideo",
            "-i",
            "-",
            "-m",
            module_path("invert").to_str().unwrap(),
            "-f",
            "nut",
            "-",
        ],
        &[],
    );
    assert!(!run.success(), "rawvideo is no longer an edge format");
    assert!(
        run.stderr.contains("rawvideo") && run.stderr.contains("nut"),
        "expected the error to name both formats, got:\n{}",
        run.stderr
    );
}

/// The face detection model facebox compiles in, by content. A different
/// model detects different rectangles, so it is pinned rather than trusted.
#[test]
fn the_face_model_is_the_one_facebox_compiles_in() {
    use sha2::{Digest, Sha256};

    let path = sidecar_root().join("modules/facebox/model/seeta_fd_frontal_v1.0.bin");
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));
    let digest = Sha256::digest(&bytes);
    assert_eq!(
        format!("{digest:x}"),
        "c4619d066ed35e84d9a8e842860b0dff567aba0cbb139881075538761db3ff5d"
    );
}

/// A NUT stream carrying `frames` with each frame's rows on the annotation
/// stream, one every `PTS_STEP` from zero.
fn annotated_nut_stream(frames: &[(Vec<u8>, Vec<String>)]) -> Vec<u8> {
    let mut wire = Vec::new();
    {
        let mut muxer = Muxer::with_annotations(&mut wire, &a_stream()).expect("write NUT headers");
        for (i, (frame, rows)) in frames.iter().enumerate() {
            let pts = i as i64 * PTS_STEP;
            muxer.write_rows(pts, rows).expect("write NUT rows");
            muxer.write_frame(pts, frame).expect("write NUT frame");
        }
        muxer.finish().expect("finish the NUT stream");
    }
    wire
}

/// `annotated_nut_stream`, plus the one trailing record a sender writes after
/// every frame's rows.
fn annotated_nut_stream_with_trailing(
    frames: &[(Vec<u8>, Vec<String>)],
    trailing: &[String],
) -> Vec<u8> {
    let mut wire = Vec::new();
    {
        let mut muxer = Muxer::with_annotations(&mut wire, &a_stream()).expect("write NUT headers");
        let mut last = 0i64;
        for (i, (frame, rows)) in frames.iter().enumerate() {
            let pts = i as i64 * PTS_STEP;
            muxer.write_rows(pts, rows).expect("write NUT rows");
            muxer.write_frame(pts, frame).expect("write NUT frame");
            last = pts;
        }
        muxer
            .write_trailing(last, trailing)
            .expect("write the trailing record");
        muxer.finish().expect("finish the NUT stream");
    }
    wire
}

/// A path in the system temp directory, removed when the guard drops. Unique
/// per call: the tests run in parallel, so two asking for the same name must
/// not land on one file.
struct TempFile(PathBuf);

impl TempFile {
    fn new(name: &str) -> TempFile {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = env::temp_dir().join(format!("ffrwd-wasm-{}-{n}-{name}", std::process::id()));
        let _ = std::fs::remove_file(&path);
        TempFile(path)
    }

    fn path(&self) -> &std::path::Path {
        &self.0
    }

    fn text(&self) -> String {
        std::fs::read_to_string(&self.0).expect("read the output file")
    }
}

impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// One rectangle as facebox reports it.
fn rect_row(x: u32, y: u32, w: u32, h: u32) -> String {
    format!(r#"{{"x":{x},"y":{y},"w":{w},"h":{h}}}"#)
}

/// Whether `(x, y)` falls inside the rectangle `(x, y, w, h)`.
fn inside(x: u32, y: u32, rect: (u32, u32, u32, u32)) -> bool {
    let (rx, ry, rw, rh) = rect;
    x >= rx && x < rx + rw && y >= ry && y < ry + rh
}

#[test]
fn blur_boxes_changes_only_the_rectangle_its_own_frames_rows_name() {
    // A different rectangle per frame, so a host that handed every frame the
    // same rows - or the wrong frame's - would blur the wrong pixels. The
    // third frame carries no rows at all and must come back untouched.
    let rects = [Some((0u32, 0u32, 2u32, 2u32)), Some((4, 4, 2, 2)), None];
    let frames: Vec<(Vec<u8>, Vec<String>)> = rects
        .iter()
        .enumerate()
        .map(|(i, rect)| {
            let rows = match rect {
                Some((x, y, w, h)) => vec![rect_row(*x, *y, *w, *h)],
                None => vec![],
            };
            (synthetic_frame(i as u8), rows)
        })
        .collect();

    let run = run_filter(
        &module_path("blur_boxes"),
        &["-annotations", "in"],
        &annotated_nut_stream(&frames),
    );
    assert_run_ok(&run, "blur-boxes reading rows");

    let got = run.frames();
    assert_eq!(got.len(), frames.len(), "every frame comes back");

    for (i, rect) in rects.iter().enumerate() {
        let input = &frames[i].0;
        let (pts, output) = &got[i];
        assert_eq!(*pts, i as i64 * PTS_STEP, "frame {i} timestamp");

        let Some(rect) = *rect else {
            assert_eq!(
                output, input,
                "frame {i} carries no rows, so there is nothing to blur"
            );
            continue;
        };

        let mut changed_inside = false;
        for y in 0..HEIGHT {
            for x in 0..WIDTH {
                let p = ((y * WIDTH + x) * 4) as usize;
                if inside(x, y, rect) {
                    changed_inside |= output[p..p + 3] != input[p..p + 3];
                } else {
                    assert_eq!(
                        output[p..p + 4],
                        input[p..p + 4],
                        "frame {i} pixel ({x}, {y}) is outside {rect:?} and must be untouched"
                    );
                }
                assert_eq!(
                    output[p + 3],
                    input[p + 3],
                    "frame {i} pixel ({x}, {y}) alpha must be unchanged"
                );
            }
        }
        assert!(
            changed_inside,
            "frame {i} rectangle {rect:?} should have been blurred"
        );
    }
}

#[test]
fn annotations_in_is_refused_for_a_module_that_reads_no_rows() {
    let frames = vec![(synthetic_frame(0), vec![rect_row(0, 0, 2, 2)])];
    let run = run_filter(
        &module_path("invert"),
        &["-annotations", "in"],
        &annotated_nut_stream(&frames),
    );
    assert!(
        !run.success(),
        "invert reads no rows, so -annotations in has nothing to give it"
    );
    assert!(
        run.stderr.contains("invert") && run.stderr.contains("does not read rows"),
        "expected the refusal to name the module, got:\n{}",
        run.stderr
    );
}

#[test]
fn an_annotated_input_is_refused_without_the_flag() {
    let frames = vec![(synthetic_frame(0), vec![rect_row(0, 0, 2, 2)])];
    let run = run_filter(
        &module_path("blur_boxes"),
        &[],
        &annotated_nut_stream(&frames),
    );
    assert!(
        !run.success(),
        "the ffmpeg-facing wire carries one stream whatever the module reads"
    );
    assert!(
        run.stderr.contains("2 streams"),
        "expected the single-stream refusal, got:\n{}",
        run.stderr
    );
}

#[derive(serde::Deserialize)]
struct FunctionDescription {
    name: String,
    params_schema: serde_json::Value,
    result_schema: serde_json::Value,
}

#[derive(serde::Deserialize)]
struct Description {
    world: String,
    name: Option<String>,
    #[allow(dead_code)]
    version: Option<String>,
    params_schema: Option<serde_json::Value>,
    rows_schema: Option<serde_json::Value>,
    pixel_formats: Option<Vec<String>>,
    sample_formats: Option<Vec<String>>,
    sample_rates: Option<Vec<u32>>,
    channel_counts: Option<Vec<u32>>,
    /// No `serde(default)`: this one is always there, so a missing key is a
    /// parse failure rather than an empty list.
    rows_language: Vec<String>,
    /// No `serde(default)` either: how many streams a module reads is always
    /// there, and 1 for one that never said.
    inputs: u32,
    #[serde(default)]
    window: Option<u32>,
    #[serde(default)]
    stride: Option<u32>,
    #[serde(default)]
    pure: Option<bool>,
    #[serde(default)]
    one_to_one: Option<bool>,
    reads_rows: Option<bool>,
    #[serde(default)]
    forwards_rows: Option<bool>,
    #[serde(default)]
    functions: Vec<FunctionDescription>,
    #[serde(default)]
    meta: bool,
}

/// Raw `--describe` stdout, trimmed. Kept separate from the parsed form so a
/// test can pin the literal absence of a key, not just its default value.
fn describe_raw(module: &std::path::Path) -> String {
    ensure_modules_built();
    let module_str = module
        .to_str()
        .expect("module path is valid UTF-8")
        .to_string();

    let run = run_ffrwd_wasm(&["--describe", &module_str], &[]);
    assert_run_ok(&run, "--describe");

    String::from_utf8(run.stdout)
        .expect("--describe output is valid UTF-8")
        .trim()
        .to_string()
}

fn describe(module: &std::path::Path) -> Description {
    let stdout = describe_raw(module);
    serde_json::from_str(&stdout).unwrap_or_else(|e| panic!("parsing {stdout:?}: {e}"))
}

#[test]
fn describe_prints_one_json_object() {
    let parsed = describe(&module_path("invert"));
    assert_eq!(parsed.world, "ffrwd:av@0.14.0");
    assert_eq!(parsed.name, Some("invert".to_string()));
    assert_eq!(parsed.pixel_formats, Some(vec!["rgba".to_string()]));
    assert_eq!(parsed.inputs, 1, "invert reads one stream");
    assert!(
        parsed.functions.is_empty(),
        "invert exports no value functions"
    );
}

#[test]
fn describe_on_a_values_only_module_has_no_filter_fields() {
    let parsed = describe(&module_path("brand"));
    assert_eq!(parsed.world, "ffrwd:av@0.14.0");
    assert_eq!(parsed.name, None, "brand exports no filter, so no name");
    assert_eq!(
        parsed.pixel_formats, None,
        "brand publishes no pixel formats"
    );

    assert_eq!(parsed.functions.len(), 1);
    let f = &parsed.functions[0];
    assert_eq!(f.name, "append-brand");
    assert_eq!(f.params_schema["type"], "object");
    assert_eq!(f.result_schema["type"], "string");
}

#[test]
fn describe_marks_a_module_that_reads_annotations() {
    let parsed = describe(&module_path("blur_boxes"));
    assert_eq!(parsed.name, Some("blur-boxes".to_string()));
    assert_eq!(
        parsed.reads_rows,
        Some(true),
        "blur_boxes exports meta-filter, which is what reading rows is"
    );
    assert!(
        parsed.meta,
        "blur_boxes reads rows, so the older meta key is set too"
    );
}

#[test]
fn describe_omits_meta_for_a_module_that_does_not_read_annotations() {
    let raw = describe_raw(&module_path("invert"));
    assert!(
        !raw.contains("\"meta\""),
        "invert exports no meta-filter, so the meta key should be absent, got:\n{raw}"
    );
    assert_eq!(
        describe(&module_path("invert")).reads_rows,
        Some(false),
        "a module that exports a frame interface always answers reads_rows"
    );
}

#[test]
fn describe_always_carries_rows_language_and_it_is_empty_when_none_is_declared() {
    for name in ["invert", "brand", "adapted_070"] {
        let raw = describe_raw(&module_path(name));
        assert!(
            raw.contains("\"rows_language\":[]"),
            "{name} declares no language for its rows, so the key is there and empty, got:\n{raw}"
        );
        assert!(
            describe(&module_path(name)).rows_language.is_empty(),
            "{name} declares no language for its rows"
        );
    }
}

#[test]
fn describe_publishes_the_params_that_settle_a_modules_rows_language() {
    // transcribe's rows come out in `language_to` when that is set, and in
    // `language` otherwise, which is the order it names them in.
    let transcribe = describe(&module_path("transcribe"));
    assert_eq!(
        transcribe.rows_language,
        vec!["language_to".to_string(), "language".to_string()],
        "the first of these params that is set at the call is the rows' language"
    );
}

#[test]
fn describe_always_says_how_many_streams_a_module_reads() {
    // A module of the world before the declaration existed, one of the
    // per-frame interface, and one with no frame interface at all: each of
    // them reads one stream, and the host answers so on their behalf.
    for name in ["adapted_080", "adapted_070", "invert", "brand"] {
        let raw = describe_raw(&module_path(name));
        assert!(
            raw.contains("\"inputs\":1"),
            "{name} reads one stream, so the key is there and 1, got:\n{raw}"
        );
        assert_eq!(describe(&module_path(name)).inputs, 1, "{name} reads one");
    }
}

#[test]
fn describe_says_whether_a_module_asks_for_inference() {
    // Read off the component's imports, so the answer comes back with no
    // model bound and no ONNX Runtime on the machine - which is what lets a
    // caller decide to bind one.
    for (name, wants) in [
        ("depth", true),
        ("nn_probe", true),
        ("invert", false),
        ("blur_mask", false),
        ("brand", false),
    ] {
        let raw = describe_raw(&module_path(name));
        assert!(
            raw.contains(&format!("\"nn\":{wants}")),
            "{name} should describe as nn:{wants}, got:\n{raw}"
        );
    }
}

#[test]
fn describe_carries_the_pads_a_multi_stream_module_declares() {
    let blur_mask = describe(&module_path("blur_mask"));
    assert_eq!(
        blur_mask.inputs, 2,
        "blur_mask reads the frame and the mask beside it"
    );
    // A module reading several streams takes one frame off each pad and hands
    // one back, which is the only shape the host drives it in.
    assert_eq!((blur_mask.window, blur_mask.stride), (Some(1), Some(1)));
}

#[test]
fn a_windowed_module_of_a_world_without_inputs_reads_one_stream() {
    // adapted-080 declares the language of its rows - what that world added -
    // and has no place to say how many streams it reads, so its adapter says
    // one for it.
    let parsed = describe(&module_path("adapted_080"));
    assert_eq!(parsed.rows_language, vec!["language".to_string()]);
    assert_eq!(parsed.inputs, 1);
    assert_eq!(parsed.reads_rows, Some(true));
    assert_eq!(parsed.forwards_rows, Some(true));
}

#[test]
fn a_windowed_module_of_the_090_world_still_runs() {
    // adapted-090 is built against the vendored 0.9.0 world - the one before
    // the packet sink existed - and loads through that world's adapter with
    // everything it declared crossing intact.
    let parsed = describe(&module_path("adapted_090"));
    assert_eq!(parsed.name, Some("adapted-090".to_string()));
    assert_eq!(parsed.inputs, 1);

    let frames: Vec<Vec<u8>> = (0..3u8).map(synthetic_frame).collect();
    let run = run_filter(
        &module_path("adapted_090"),
        &["-annotations", "out"],
        &nut_stream(&frames),
    );
    assert_run_ok(&run, "adapted-090");
    let (got, trailing) = annotated_frames_and_trailing(&run.stdout);
    assert_eq!(got.len(), frames.len(), "every frame passes through");
    for (i, (pts, data, rows)) in got.iter().enumerate() {
        assert_eq!(*pts, i as i64 * PTS_STEP, "frame {i} timestamp");
        assert_eq!(data, &frames[i], "frame {i} pixels");
        // adapted-090's row already names `pts`, so the host leaves it alone
        // and adds only `time` - not the `{"pts":{pts}}` alone this pinned
        // before the host started stamping rows: a module's own pts wins,
        // but the row still gains the field it never had.
        assert_eq!(rows.len(), 1, "frame {i} row");
        let row: serde_json::Value =
            serde_json::from_str(&rows[0]).unwrap_or_else(|e| panic!("frame {i} row: {e}"));
        assert_eq!(
            row["pts"], *pts,
            "frame {i}: adapted-090's own pts is not clobbered"
        );
        let time = row["time"]
            .as_f64()
            .unwrap_or_else(|| panic!("frame {i}: expected the host-stamped time, row was {row}"));
        assert!(
            (time - TIME_BASE.seconds(*pts)).abs() < 1e-9,
            "frame {i}: time {time} does not agree with pts {pts} through the stream's time base"
        );
        assert_eq!(
            row.as_object().map(|o| o.len()),
            Some(2),
            "frame {i}: exactly pts and time, nothing else"
        );
    }
    assert!(
        trailing.is_empty(),
        "adapted-090 ends with no trailing rows"
    );
}

#[test]
fn describe_says_a_values_only_module_reads_no_rows_either_way() {
    let parsed = describe(&module_path("brand"));
    assert_eq!(
        parsed.reads_rows, None,
        "brand exports no frame interface, so reads_rows is null like its other frame fields"
    );
}

#[test]
fn describe_publishes_how_a_windowed_module_must_be_driven() {
    let doubled = describe(&module_path("double"));
    assert_eq!(doubled.name, Some("double".to_string()));
    assert_eq!(doubled.window, Some(1), "double sees one frame at a time");
    assert_eq!(doubled.stride, Some(1));
    assert_eq!(
        doubled.pure,
        Some(false),
        "double carries a frame between calls"
    );
    assert_eq!(
        doubled.one_to_one,
        Some(false),
        "double emits two frames per frame"
    );
    assert_eq!(
        doubled.reads_rows,
        Some(false),
        "double carries the rows arriving with a frame and reads none of them"
    );
    assert_eq!(
        doubled.forwards_rows,
        Some(true),
        "a frame's own rows leave on its first copy"
    );

    let tail = describe(&module_path("tail3"));
    assert_eq!(tail.window, Some(3));
    assert_eq!(tail.stride, Some(3), "tail3's windows do not overlap");
    assert_eq!(
        tail.pure,
        Some(true),
        "nothing carries between tail3's calls"
    );
    assert_eq!(
        tail.one_to_one,
        Some(true),
        "tail3 passes each frame through at its own timestamp"
    );
    assert_eq!(
        tail.reads_rows,
        Some(false),
        "tail3 passes incoming rows on beside its own and reads none of them"
    );
    assert_eq!(
        tail.forwards_rows,
        Some(true),
        "passing them on is what tail3 declares"
    );
}

#[test]
fn describe_publishes_faceboxs_window_and_that_it_reads_rows() {
    let faces = describe(&module_path("facebox"));
    assert_eq!(faces.name, Some("facebox".to_string()));
    assert_eq!(faces.window, Some(1), "facebox sees one frame at a time");
    assert_eq!(faces.stride, Some(1));
    assert_eq!(
        faces.pure,
        Some(false),
        "facebox carries a box history between calls"
    );
    assert_eq!(
        faces.one_to_one,
        Some(true),
        "facebox passes each frame through at its own timestamp"
    );
    assert_eq!(
        faces.reads_rows,
        Some(true),
        "facebox acts on the shot index arriving with a frame"
    );
    assert!(
        faces.meta,
        "facebox reads rows, so the older meta key is set"
    );
    assert_eq!(
        faces.forwards_rows,
        Some(false),
        "facebox consumes the rows it reads: what leaves is its own rectangles"
    );

    let params = faces.params_schema.expect("facebox declares parameters");
    let window = &params["properties"]["smooth-window"];
    assert_eq!(window["type"], "number");
    assert_eq!(window["default"], 0.5, "half a second of box history");
}

#[test]
fn describe_publishes_what_shots_emits_and_how_it_is_driven() {
    let shots = describe(&module_path("shots"));
    assert_eq!(shots.name, Some("shots".to_string()));
    assert_eq!(shots.window, Some(1));
    assert_eq!(shots.stride, Some(1));
    assert_eq!(
        shots.pure,
        Some(false),
        "shots carries the previous frame between calls"
    );
    assert_eq!(shots.one_to_one, Some(true));
    assert_eq!(
        shots.reads_rows,
        Some(false),
        "shots reads pixels, not the rows arriving with them"
    );
    assert_eq!(
        shots.forwards_rows,
        Some(false),
        "what leaves a frame is the shot index and nothing else"
    );
    assert_eq!(
        shots.pixel_formats,
        Some(vec!["yuv420p".to_string(), "rgba".to_string()]),
        "shots reads luma from both formats facebox does"
    );

    let rows = shots.rows_schema.expect("shots emits rows");
    assert_eq!(rows["properties"]["shot"]["type"], "integer");
    assert_eq!(rows["required"], serde_json::json!(["shot"]));
}

#[test]
fn describe_omits_meta_for_a_windowed_module_that_reads_no_rows() {
    // A window is not a claim on the rows: the two questions are separate
    // keys, and a module that only carries rows through answers no to this one.
    for name in ["double", "tail3"] {
        let raw = describe_raw(&module_path(name));
        assert!(
            !raw.contains("\"meta\""),
            "{name} reads no rows, so the meta key should be absent, got:\n{raw}"
        );
    }
}

#[test]
fn describe_omits_the_window_fields_for_a_per_frame_module() {
    let raw = describe_raw(&module_path("invert"));
    for key in ["\"window\"", "\"stride\"", "\"pure\"", "\"one_to_one\""] {
        assert!(
            !raw.contains(key),
            "invert exports the per-frame interface, so {key} should be absent, got:\n{raw}"
        );
    }
}

#[test]
fn describe_omits_meta_for_a_values_only_module() {
    let raw = describe_raw(&module_path("brand"));
    assert!(
        !raw.contains("\"meta\""),
        "brand exports no filter at all, so the meta key should be absent, got:\n{raw}"
    );
}

#[test]
fn invoke_returns_the_module_result() {
    ensure_modules_built();
    let module_str = module_path("brand")
        .to_str()
        .expect("module path is valid UTF-8")
        .to_string();

    let run = run_ffrwd_wasm(
        &[
            "--invoke",
            &module_str,
            "append-brand",
            r#"{"title":"clip","suffix":"_v2"}"#,
        ],
        &[],
    );
    assert_run_ok(&run, "--invoke");
    let stdout = String::from_utf8(run.stdout).expect("--invoke output is valid UTF-8");
    assert_eq!(stdout.trim(), "\"clip_v2\"");
}

#[test]
fn invoke_surfaces_the_modules_own_validation_error() {
    ensure_modules_built();
    let module_str = module_path("brand")
        .to_str()
        .expect("module path is valid UTF-8")
        .to_string();

    let run = run_ffrwd_wasm(
        &[
            "--invoke",
            &module_str,
            "append-brand",
            r#"{"title":"clip"}"#,
        ],
        &[],
    );
    assert!(!run.success(), "a missing required key should be an error");
    assert!(
        run.stderr.contains("suffix"),
        "expected the missing key named in the error, got:\n{}",
        run.stderr
    );
}

#[test]
fn invoke_on_an_unknown_function_names_what_is_exported() {
    ensure_modules_built();
    let module_str = module_path("brand")
        .to_str()
        .expect("module path is valid UTF-8")
        .to_string();

    let run = run_ffrwd_wasm(&["--invoke", &module_str, "not-a-function", "{}"], &[]);
    assert!(!run.success(), "an unknown function should be an error");
    assert!(
        run.stderr.contains("append-brand"),
        "expected the module's actual function named in the error, got:\n{}",
        run.stderr
    );
}

#[test]
fn invoke_on_a_module_without_values_is_an_error() {
    ensure_modules_built();
    let module_str = module_path("invert")
        .to_str()
        .expect("module path is valid UTF-8")
        .to_string();

    let run = run_ffrwd_wasm(&["--invoke", &module_str, "append-brand", "{}"], &[]);
    assert!(
        !run.success(),
        "a module without the values interface should be an error"
    );
    assert!(
        run.stderr.contains("filter"),
        "expected the error to name what invert does export instead, got:\n{}",
        run.stderr
    );
}

// The windowed engine: a module that changes the frame count, and one whose
// window does not divide the stream.

/// Every frame on stdout with the rows riding beside it, for a run that was
/// given `-annotations out`.
fn annotated_frames(wire: &[u8]) -> Vec<(i64, Vec<u8>, Vec<String>)> {
    let mut demuxer = Demuxer::open_annotated(wire).expect("read the annotated NUT headers");
    let mut frames = Vec::new();
    let mut buf = Vec::new();
    while let Some(pts) = demuxer.read_frame(&mut buf).expect("read a NUT frame") {
        frames.push((pts, buf.clone(), demuxer.take_rows(pts)));
    }
    frames
}

/// One frame read back off an annotated stream: its timestamp, its pixels and
/// the rows riding beside it.
type AnnotatedFrame = (i64, Vec<u8>, Vec<String>);

/// Every frame on stdout with the rows riding beside it, and the trailing
/// rows the stream ended with.
fn annotated_frames_and_trailing(wire: &[u8]) -> (Vec<AnnotatedFrame>, Vec<String>) {
    let mut demuxer = Demuxer::open_annotated(wire).expect("read the annotated NUT headers");
    let mut frames = Vec::new();
    let mut buf = Vec::new();
    while let Some(pts) = demuxer.read_frame(&mut buf).expect("read a NUT frame") {
        frames.push((pts, buf.clone(), demuxer.take_rows(pts)));
    }
    let trailing = demuxer.take_trailing();
    (frames, trailing)
}

/// One row as tail3 reports a call.
#[derive(serde::Deserialize)]
struct TailRow {
    frames: usize,
    last: bool,
}

#[test]
fn double_emits_every_frame_twice_halfway_apart() {
    let frames: Vec<Vec<u8>> = (0..3u8).map(synthetic_frame).collect();
    let run = run_filter(&module_path("double"), &[], &nut_stream(&frames));
    assert_run_ok(&run, "double");

    // Each frame at its own timestamp, and again half the gap to its
    // neighbour later. The input is evenly spaced, so that is half a step.
    let half = PTS_STEP / 2;
    let expected: Vec<(i64, &Vec<u8>)> = frames
        .iter()
        .enumerate()
        .flat_map(|(i, frame)| {
            let pts = i as i64 * PTS_STEP;
            [(pts, frame), (pts + half, frame)]
        })
        .collect();

    let got = run.frames();
    assert_eq!(got.len(), 2 * frames.len(), "twice as many frames come out");
    for (i, ((pts, data), (want_pts, want_data))) in got.iter().zip(expected).enumerate() {
        assert_eq!(*pts, want_pts, "output frame {i} timestamp");
        assert_eq!(data, want_data, "output frame {i} pixels");
    }

    // Twice the frames at half the spacing is the same stretch of time: the
    // timestamps span one extra half-step, not one extra frame.
    let in_span = (frames.len() as i64 - 1) * PTS_STEP;
    let out_span = got[got.len() - 1].0 - got[0].0;
    assert_eq!(out_span, in_span + half);
}

#[test]
fn double_splits_a_lone_frame_by_a_single_tick() {
    let run = run_filter(
        &module_path("double"),
        &[],
        &nut_stream(&[synthetic_frame(0)]),
    );
    assert_run_ok(&run, "double over one frame");

    let got = run.frames();
    assert_eq!(got.len(), 2, "one frame in, two out");
    assert_eq!(got[0].0, 0, "the first copy keeps the input's timestamp");
    assert_eq!(
        got[1].0, 1,
        "a lone frame has no gap to halve, so the second copy is one tick later"
    );
    assert_eq!(got[0].1, got[1].1, "both copies are the same picture");
}

#[test]
fn the_final_call_carries_whatever_the_stride_left_over() {
    // Ten frames through a window of three: three full calls, then a final
    // call holding the one frame left. Nine leave nothing over, and the final
    // call - which still happens - carries no frames, so no row.
    let cases = [
        (10usize, vec![(3, false), (3, false), (3, false), (1, true)]),
        (9, vec![(3, false), (3, false), (3, false)]),
    ];

    for (count, expected) in cases {
        let frames: Vec<Vec<u8>> = (0..count).map(|i| synthetic_frame(i as u8)).collect();
        let run = run_filter(
            &module_path("tail3"),
            &["-annotations", "out"],
            &nut_stream(&frames),
        );
        assert_run_ok(&run, &format!("tail3 over {count} frames"));

        let got = annotated_frames(&run.stdout);
        assert_eq!(got.len(), count, "tail3 passes every frame through");
        for (i, (pts, data, _)) in got.iter().enumerate() {
            assert_eq!(*pts, i as i64 * PTS_STEP, "frame {i} timestamp");
            assert_eq!(data, &frames[i], "frame {i} pixels");
        }

        let calls: Vec<(usize, bool)> = got
            .iter()
            .flat_map(|(_, _, rows)| rows.iter())
            .map(|row| {
                let parsed: TailRow = serde_json::from_str(row)
                    .unwrap_or_else(|e| panic!("parsing tail3 row {row:?}: {e}"));
                (parsed.frames, parsed.last)
            })
            .collect();
        assert_eq!(calls, expected, "{count} frames through a window of three");
    }
}

#[test]
fn an_impure_module_runs_under_jobs_and_matches_serial() {
    // double carries a frame between calls, so its lane admits one task at a
    // time whatever -jobs says; the flag caps the pool and refuses nothing.
    let frames: Vec<Vec<u8>> = (0..6u8).map(synthetic_frame).collect();
    let wire = nut_stream(&frames);

    let baseline = run_filter(&module_path("double"), &["-jobs", "1"], &wire);
    assert_run_ok(&baseline, "double jobs=1");

    let run = run_filter(&module_path("double"), &["-jobs", "2"], &wire);
    assert_run_ok(&run, "double jobs=2");
    assert_eq!(
        run.stdout, baseline.stdout,
        "an impure module's output must be byte-identical at any -jobs"
    );
}

#[test]
fn a_pure_windowed_module_gives_the_same_output_under_jobs() {
    // Twenty frames make six full windows and a tail of two, so the workers
    // divide the stream unevenly and the last one carries the final call.
    let frames: Vec<Vec<u8>> = (0..20u8).map(synthetic_frame).collect();
    let wire = nut_stream(&frames);

    let baseline = run_filter(
        &module_path("tail3"),
        &["-annotations", "out", "-jobs", "1"],
        &wire,
    );
    assert_run_ok(&baseline, "tail3 jobs=1");

    for jobs in ["3", "4"] {
        let run = run_filter(
            &module_path("tail3"),
            &["-annotations", "out", "-jobs", jobs],
            &wire,
        );
        assert_run_ok(&run, &format!("tail3 jobs={jobs}"));
        assert_eq!(
            run.stdout, baseline.stdout,
            "jobs={jobs} output must be byte-identical to jobs=1"
        );
    }
}

// The module network: several modules in one process, wired by a
// -filter_complex string over the names an -m table binds.

#[test]
fn the_single_module_spelling_runs_as_a_network_of_one() {
    let frames: Vec<Vec<u8>> = (0..4u8).map(synthetic_frame).collect();
    let wire = nut_stream(&frames);

    let short = run_filter(&module_path("invert"), &[], &wire);
    assert_run_ok(&short, "-m <path>");

    let network = run_network(
        &[("invert", "invert")],
        "[0:v]invert[out0]",
        &[],
        ONE_OUTPUT,
        &wire,
    );
    assert_run_ok(&network, "-m invert=<path>");
    assert_eq!(
        short.stdout, network.stdout,
        "the two spellings of one module must produce the same bytes"
    );
}

#[test]
fn rows_ride_from_one_module_to_the_next_inside_the_process() {
    // tail3 carries the rows arriving with a frame on through, and blur-boxes
    // downstream acts on them - so what is blurred proves the rows crossed
    // between two modules of one process, with no wire between them. A
    // different rectangle per frame, so the wrong frame's rows would blur the
    // wrong pixels; the third frame carries none and must come back untouched.
    let rects = [
        Some((0u32, 0u32, 2u32, 2u32)),
        Some((4, 4, 2, 2)),
        None,
        Some((2, 2, 3, 3)),
    ];
    let frames: Vec<(Vec<u8>, Vec<String>)> = rects
        .iter()
        .enumerate()
        .map(|(i, rect)| {
            let rows = match rect {
                Some((x, y, w, h)) => vec![rect_row(*x, *y, *w, *h)],
                None => vec![],
            };
            (synthetic_frame(i as u8), rows)
        })
        .collect();

    let run = run_network(
        &[("tail3", "tail3"), ("blur", "blur_boxes")],
        "[0:v]tail3[a];[a]blur[out0]",
        &["-annotations", "in"],
        ONE_OUTPUT,
        &annotated_nut_stream(&frames),
    );
    assert_run_ok(&run, "tail3 into blur-boxes");

    let got = run.frames();
    assert_eq!(got.len(), frames.len(), "every frame comes back");

    for (i, rect) in rects.iter().enumerate() {
        let input = &frames[i].0;
        let (pts, output) = &got[i];
        assert_eq!(*pts, i as i64 * PTS_STEP, "frame {i} timestamp");

        let Some(rect) = *rect else {
            assert_eq!(
                output, input,
                "frame {i} carries no rectangle, so there is nothing to blur"
            );
            continue;
        };

        let mut changed_inside = false;
        for y in 0..HEIGHT {
            for x in 0..WIDTH {
                let p = ((y * WIDTH + x) * 4) as usize;
                if inside(x, y, rect) {
                    changed_inside |= output[p..p + 3] != input[p..p + 3];
                } else {
                    assert_eq!(
                        output[p..p + 4],
                        input[p..p + 4],
                        "frame {i} pixel ({x}, {y}) is outside {rect:?} and must be untouched"
                    );
                }
            }
        }
        assert!(
            changed_inside,
            "frame {i} rectangle {rect:?} should have been blurred"
        );
    }
}

#[test]
fn a_nodes_options_reach_its_module_as_the_parameters_it_declares() {
    // blur-boxes rejects a radius outside 1..=256, so a refusal naming the
    // value is proof the option arrived as a number rather than as text.
    let frames = vec![(synthetic_frame(0), vec![rect_row(0, 0, 4, 4)])];
    let wire = annotated_nut_stream(&frames);

    let accepted = run_network(
        &[("tail3", "tail3"), ("blur", "blur_boxes")],
        "[0:v]tail3[a];[a]blur=radius=2:passes=1[out0]",
        &["-annotations", "in"],
        ONE_OUTPUT,
        &wire,
    );
    assert_run_ok(&accepted, "blur-boxes with options");

    let refused = run_network(
        &[("tail3", "tail3"), ("blur", "blur_boxes")],
        "[0:v]tail3[a];[a]blur=radius=999[out0]",
        &["-annotations", "in"],
        ONE_OUTPUT,
        &wire,
    );
    assert!(!refused.success(), "999 is outside blur-boxes' own range");
    assert!(
        refused.stderr.contains("999"),
        "expected the module's own complaint about the value, got:\n{}",
        refused.stderr
    );
}

#[test]
fn an_option_the_module_does_not_declare_is_refused_by_name() {
    let run = run_network(
        &[("invert", "invert")],
        "[0:v]invert=strength=3[out0]",
        &[],
        ONE_OUTPUT,
        &nut_stream(&[synthetic_frame(0)]),
    );
    assert!(!run.success(), "invert declares no parameters at all");
    assert!(
        run.stderr.contains("strength") && run.stderr.contains("invert"),
        "expected the option and the module named, got:\n{}",
        run.stderr
    );
}

#[test]
fn a_module_name_no_binding_covers_is_refused_by_name() {
    let run = run_network(
        &[("invert", "invert")],
        "[0:v]invert[a];[a]blur[out0]",
        &[],
        ONE_OUTPUT,
        &nut_stream(&[synthetic_frame(0)]),
    );
    assert!(!run.success(), "nothing binds the name blur");
    assert!(
        run.stderr.contains("blur") && run.stderr.contains("invert"),
        "expected the unbound name and what is bound, got:\n{}",
        run.stderr
    );
}

#[test]
fn a_label_nothing_reads_is_refused_by_name() {
    let run = run_network(
        &[("invert", "invert"), ("double", "double")],
        "[0:v]invert[a];[0:v]double[out0]",
        &[],
        ONE_OUTPUT,
        &nut_stream(&[synthetic_frame(0)]),
    );
    assert!(!run.success(), "[a] is written and never read");
    assert!(
        run.stderr.contains("[a]") && run.stderr.contains("invert"),
        "expected the dangling label and its module named, got:\n{}",
        run.stderr
    );
}

#[test]
fn a_label_two_modules_write_is_refused_by_name() {
    let run = run_network(
        &[("invert", "invert"), ("double", "double")],
        "[0:v]invert[a];[0:v]double[a];[a]invert[out0]",
        &[],
        ONE_OUTPUT,
        &nut_stream(&[synthetic_frame(0)]),
    );
    assert!(!run.success(), "two modules cannot write one label");
    assert!(
        run.stderr.contains("[a]")
            && run.stderr.contains("invert")
            && run.stderr.contains("double"),
        "expected both writers of the label named, got:\n{}",
        run.stderr
    );
}

#[test]
fn an_input_index_with_no_input_is_refused_by_name() {
    let run = run_network(
        &[("invert", "invert")],
        "[1:v]invert[out0]",
        &[],
        ONE_OUTPUT,
        &nut_stream(&[synthetic_frame(0)]),
    );
    assert!(!run.success(), "there is no second -i");
    assert!(
        run.stderr.contains("[1:v]") && run.stderr.contains("invert"),
        "expected the input label and the module named, got:\n{}",
        run.stderr
    );
}

#[test]
fn a_map_naming_a_label_the_network_does_not_write_is_refused() {
    let run = run_network(
        &[("invert", "invert")],
        "[0:v]invert[out0]",
        &[],
        &["-map", "[nope]", "-f", "nut", "-"],
        &nut_stream(&[synthetic_frame(0)]),
    );
    assert!(!run.success(), "nothing writes [nope]");
    assert!(
        run.stderr.contains("nope"),
        "expected the missing label named, got:\n{}",
        run.stderr
    );
}

#[test]
fn a_network_runs_under_jobs_and_matches_serial() {
    let frames: Vec<Vec<u8>> = (0..8u8).map(synthetic_frame).collect();
    let wire = nut_stream(&frames);

    let baseline = run_network(
        &[("invert", "invert")],
        "[0:v]invert[out0]",
        &["-jobs", "1"],
        ONE_OUTPUT,
        &wire,
    );
    assert_run_ok(&baseline, "network jobs=1");

    let run = run_network(
        &[("invert", "invert")],
        "[0:v]invert[out0]",
        &["-jobs", "4"],
        ONE_OUTPUT,
        &wire,
    );
    assert_run_ok(&run, "network jobs=4");
    assert_eq!(
        run.stdout, baseline.stdout,
        "a network's output must be byte-identical at any -jobs"
    );
}

// Determinism across thread counts: composed graphs at 1, 2 and 8 workers
// produce the same bytes and the same row order, which is the contract that
// lets the pool default on.

/// One network run per `-jobs` in `counts`, each asserted byte-identical to
/// the first.
fn assert_jobs_match(
    modules: &[(&str, &str)],
    wiring: &str,
    extra: &[&str],
    outputs: &[&str],
    wire: &[u8],
    what: &str,
) -> Vec<u8> {
    let mut baseline: Option<Vec<u8>> = None;
    for jobs in ["1", "2", "8"] {
        let mut with_jobs: Vec<&str> = vec!["-jobs", jobs];
        with_jobs.extend_from_slice(extra);
        let run = run_network(modules, wiring, &with_jobs, outputs, wire);
        assert_run_ok(&run, &format!("{what} jobs={jobs}"));
        match &baseline {
            None => baseline = Some(run.stdout),
            Some(expected) => assert_eq!(
                &run.stdout, expected,
                "{what} jobs={jobs} output must be byte-identical to jobs=1"
            ),
        }
    }
    baseline.expect("at least one count ran")
}

#[test]
fn a_pure_into_pure_graph_is_identical_at_every_thread_count() {
    // tail3 is windowed (three frames a call) and pure, so both lanes may
    // spread across workers; the reassembly is what keeps the bytes equal.
    let frames: Vec<Vec<u8>> = (0..20u8).map(synthetic_frame).collect();
    assert_jobs_match(
        &[("invert", "invert"), ("tail3", "tail3")],
        "[0:v]invert[a];[a]tail3[out0]",
        &["-annotations", "out"],
        ONE_OUTPUT,
        &nut_stream(&frames),
        "invert into tail3",
    );
}

#[test]
fn a_pure_into_impure_graph_keeps_frame_and_row_order() {
    // framestats is impure and writes one row per frame, each stamped with
    // that frame's time, plus a trailing count; any frame reaching it out of
    // order reorders the file, so identical ndjson at every thread count is
    // frame-order and row-order proof at once.
    let frames: Vec<Vec<u8>> = (0..20u8).map(synthetic_frame).collect();
    let wire = nut_stream(&frames);

    let mut baseline: Option<Vec<u8>> = None;
    for jobs in ["1", "2", "8"] {
        let rows_file = TempFile::new(&format!("determinism-{jobs}.ndjson"));
        let rows_path = rows_file
            .path()
            .to_str()
            .expect("temp path is valid UTF-8")
            .to_string();
        let run = run_network(
            &[("invert", "invert"), ("framestats", "framestats")],
            "[0:v]invert[a];[a]framestats[b]",
            &["-jobs", jobs],
            &["-map", "[b]", "-f", "ndjson", &rows_path],
            &wire,
        );
        assert_run_ok(&run, &format!("invert into framestats jobs={jobs}"));
        let rows = std::fs::read(rows_file.path()).expect("read the ndjson output");
        assert!(!rows.is_empty(), "framestats writes a row per frame");
        match &baseline {
            None => baseline = Some(rows),
            Some(expected) => assert_eq!(
                &rows, expected,
                "jobs={jobs} rows must be identical to jobs=1, order included"
            ),
        }
    }
}

// Rows with no frame to ride: what a module says after its last frame, and
// how it travels.

/// The summary row framestats ends with.
#[derive(serde::Deserialize)]
struct FramestatsSummary {
    frames: u64,
}

#[test]
fn a_trailing_row_crosses_from_one_module_to_the_next() {
    // framestats' count is not known until its last frame has gone by, so it
    // leaves as a trailing row. tail3 downstream passes on whatever trailing
    // rows it is handed and invents none, so what the network ends with is
    // proof the row crossed: the output is mapped to tail3, and tail3 has
    // nothing else to put there.
    let frames: Vec<Vec<u8>> = (0..7u8).map(synthetic_frame).collect();

    let run = run_network(
        &[("framestats", "framestats"), ("tail3", "tail3")],
        "[0:v]framestats[a];[a]tail3[out0]",
        &["-annotations", "out"],
        ONE_OUTPUT,
        &nut_stream(&frames),
    );
    assert_run_ok(&run, "framestats into tail3");

    let (got, trailing) = annotated_frames_and_trailing(&run.stdout);
    assert_eq!(got.len(), frames.len(), "every frame passes through");

    assert_eq!(trailing.len(), 1, "one row arrived with no frame to ride");
    let summary: FramestatsSummary = serde_json::from_str(&trailing[0])
        .unwrap_or_else(|e| panic!("parsing the trailing row {:?}: {e}", trailing[0]));
    assert_eq!(
        summary.frames, 7,
        "the count framestats ended with is the one that left the network"
    );
}

#[test]
fn a_trailing_record_on_the_wire_reaches_the_module_on_its_last_call() {
    // The record arrives in final position on the annotation stream, and the
    // module reading that input is handed it on its last call. tail3 passes
    // on whatever it is handed, so its own rows output holding the row is
    // proof it arrived.
    let frames = vec![(synthetic_frame(0), vec![rect_row(0, 0, 2, 2)])];
    let sent = r#"{"frames":1}"#.to_string();
    let wire = annotated_nut_stream_with_trailing(&frames, std::slice::from_ref(&sent));

    let rows_file = TempFile::new("wire-trailing.ndjson");
    let rows_path = rows_file
        .path()
        .to_str()
        .expect("temp path is valid UTF-8")
        .to_string();

    let run = run_network(
        &[("tail3", "tail3"), ("blur", "blur_boxes")],
        "[0:v]tail3[a];[a]blur[out0]",
        &["-annotations", "in"],
        &[
            "-map", "[a]", "-f", "ndjson", &rows_path, "-map", "[out0]", "-f", "nut", "-",
        ],
        &wire,
    );
    assert_run_ok(&run, "a trailing record on the wire");

    let written = rows_file.text();
    assert!(
        written.lines().any(|line| line == sent),
        "expected the arriving trailing row carried on, got:\n{written}"
    );
}

#[test]
fn a_module_that_drops_rows_ends_the_path_the_annotations_travel() {
    // facebox reads the rows arriving with a frame and puts its own
    // rectangles there instead, so nothing that came in on the wire reaches
    // blur-boxes behind it.
    let frames = vec![(synthetic_frame(0), vec![r#"{"shot":0}"#.to_string()])];

    let run = run_network(
        &[("facebox", "facebox"), ("blur_boxes", "blur_boxes")],
        "[0:v]facebox[a];[a]blur_boxes[out0]",
        &["-annotations", "in"],
        ONE_OUTPUT,
        &annotated_nut_stream(&frames),
    );
    assert!(
        !run.success(),
        "the rows arriving on the wire stop at facebox"
    );
    assert!(
        run.stderr.contains("facebox") && run.stderr.contains("blur-boxes"),
        "expected the module that stops the rows and the one that wanted them named, got:\n{}",
        run.stderr
    );
}

// One module per world older than the current one, each built against that
// world, so every adapter is driven by something actually shaped that way.

/// What a module of a world without the time-base field reports back.
#[derive(serde::Deserialize)]
struct TimeBaseRow {
    #[serde(rename = "time-base")]
    time_base: String,
    #[serde(default)]
    time: f64,
    #[serde(default, rename = "rows-in")]
    rows_in: usize,
}

/// What a module of the world that first read rows reports back.
#[derive(serde::Deserialize)]
struct RowsInRow {
    #[serde(rename = "rows-in")]
    rows_in: usize,
}

#[test]
fn a_per_frame_module_of_the_oldest_world_still_runs() {
    let frames: Vec<Vec<u8>> = (0..2u8).map(synthetic_frame).collect();
    let run = run_filter(
        &module_path("adapted_020"),
        &["-annotations", "out"],
        &nut_stream(&frames),
    );
    assert_run_ok(&run, "adapted-020");

    let (got, _) = annotated_frames_and_trailing(&run.stdout);
    assert_eq!(got.len(), frames.len(), "every frame passes through");
    for (i, (pts, data, rows)) in got.iter().enumerate() {
        assert_eq!(*pts, i as i64 * PTS_STEP, "frame {i} timestamp");
        assert_eq!(data, &frames[i], "frame {i} pixels");
        assert_eq!(rows.len(), 1, "frame {i} carries the module's one row");
        let row: TimeBaseRow =
            serde_json::from_str(&rows[0]).unwrap_or_else(|e| panic!("parsing {:?}: {e}", rows[0]));
        assert_eq!(
            row.time_base, "1/65536",
            "a world without the field is told the time base as a tag"
        );
        let expected = (i as i64 * PTS_STEP) as f64 / 65536.0;
        assert!(
            (row.time - expected).abs() < 1e-12,
            "frame {i} time: expected {expected}, got {}",
            row.time
        );
    }
}

#[test]
fn a_values_module_of_an_older_world_still_answers() {
    ensure_modules_built();
    let module_str = module_path("adapted_030")
        .to_str()
        .expect("module path is valid UTF-8")
        .to_string();

    let run = run_ffrwd_wasm(&["--invoke", &module_str, "world-of", "{}"], &[]);
    assert_run_ok(&run, "--invoke on adapted-030");
    let stdout = String::from_utf8(run.stdout).expect("--invoke output is valid UTF-8");
    assert_eq!(stdout.trim(), "\"ffrwd:av@0.3.0\"");
}

#[test]
fn a_meta_module_of_an_older_world_is_still_handed_the_rows() {
    let frames = vec![
        (
            synthetic_frame(0),
            vec![rect_row(0, 0, 2, 2), rect_row(4, 4, 2, 2)],
        ),
        (synthetic_frame(1), vec![]),
    ];
    let run = run_filter(
        &module_path("adapted_040"),
        &["-annotations", "in", "-annotations", "out"],
        &annotated_nut_stream(&frames),
    );
    assert_run_ok(&run, "adapted-040");

    let (got, trailing) = annotated_frames_and_trailing(&run.stdout);
    let counted: Vec<usize> = got
        .iter()
        .flat_map(|(_, _, rows)| rows.iter())
        .map(|row| {
            let parsed: RowsInRow =
                serde_json::from_str(row).unwrap_or_else(|e| panic!("parsing {row:?}: {e}"));
            parsed.rows_in
        })
        .collect();
    assert_eq!(
        counted,
        vec![2, 0],
        "the rows crossed the adapter unchanged"
    );
    assert!(
        trailing.is_empty(),
        "a module of that world has no trailing rows to emit"
    );
}

/// `row` parsed as JSON, for comparing rows structurally rather than as
/// exact bytes - key order is not part of what a row promises.
fn parsed_row(row: &str) -> serde_json::Value {
    serde_json::from_str(row).unwrap_or_else(|e| panic!("parsing {row:?}: {e}"))
}

/// `row`, as the host's stamp leaves it once it has crossed a `-annotations
/// out` wire riding the frame at `pts` on `time_base`: `pts` and `time`
/// added, since none of this file's fixture rows name either key of their
/// own.
fn stamped_row(row: &str, pts: i64, time_base: TimeBase) -> serde_json::Value {
    let mut value = parsed_row(row);
    let object = value.as_object_mut().expect("row is a JSON object");
    object.insert("pts".to_string(), serde_json::json!(pts));
    object.insert(
        "time".to_string(),
        serde_json::json!(time_base.seconds(pts)),
    );
    value
}

#[test]
fn a_windowed_module_of_the_previous_world_still_runs() {
    // Four frames through a window of two: two calls, each reporting the time
    // base it was told as a tag and how many rows it was handed. The rows
    // arriving travel on beside its own, which is what the host assumed of
    // every windowed module then; and a trailing record on the wire reaches
    // it as nothing, since its world has no spelling for one.
    let frames: Vec<(Vec<u8>, Vec<String>)> = (0..4u8)
        .map(|i| (synthetic_frame(i), vec![rect_row(u32::from(i), 0, 2, 2)]))
        .collect();
    let wire = annotated_nut_stream_with_trailing(&frames, &[r#"{"frames":4}"#.to_string()]);

    let run = run_filter(
        &module_path("adapted_050"),
        &["-annotations", "in", "-annotations", "out"],
        &wire,
    );
    assert_run_ok(&run, "adapted-050");

    let (got, trailing) = annotated_frames_and_trailing(&run.stdout);
    assert_eq!(got.len(), frames.len(), "every frame passes through");

    // The call's own row rides its first frame; the arriving rows ride the
    // frames they came in on.
    let first: TimeBaseRow = serde_json::from_str(&got[0].2[0])
        .unwrap_or_else(|e| panic!("parsing {:?}: {e}", got[0].2[0]));
    assert_eq!(
        first.time_base, "1/65536",
        "a world without the field is told the time base as a tag"
    );
    assert_eq!(first.rows_in, 2, "both frames of the window carried a row");
    assert_eq!(
        parsed_row(&got[0].2[1]),
        stamped_row(&frames[0].1[0], got[0].0, TIME_BASE),
        "the arriving row travels on beside the module's own, gaining the \
         pts/time stamp every row that rides a frame gets"
    );
    let expected: Vec<serde_json::Value> = frames[1]
        .1
        .iter()
        .map(|row| stamped_row(row, got[1].0, TIME_BASE))
        .collect();
    let actual: Vec<serde_json::Value> = got[1].2.iter().map(|row| parsed_row(row)).collect();
    assert_eq!(actual, expected, "and so does the next frame's");

    assert!(
        trailing.is_empty(),
        "a module of that world cannot emit trailing rows"
    );
}

// Audio. The wire is built here the same way the video one is - `Stream::audio`
// writes the header real ffmpeg writes - so these stay in the unit tier; that
// real ffmpeg reads and writes it is `ffmpeg.rs`'s business.

/// 48 kHz, the rate `sine` produces: at the natural time base one sample is
/// one tick.
const RATE: u32 = 48_000;

fn an_audio_stream(sample_fmt: &str, channels: u32) -> Stream {
    Stream::audio(sample_fmt, RATE, channels).expect("the pcm formats are carried")
}

/// A NUT audio stream of `packets` samples each, filled with a ramp so a
/// window's contents say which samples it holds. Every packet carries the
/// timestamp of its first sample, which at this time base is its offset.
fn audio_wire(packets: &[usize]) -> Vec<u8> {
    let stream = an_audio_stream("f32", 1);
    let mut wire = Vec::new();
    {
        let mut muxer = Muxer::new(&mut wire, &stream).expect("write NUT headers");
        let mut offset = 0usize;
        for samples in packets {
            let data: Vec<u8> = (offset..offset + samples)
                .flat_map(|s| (s as f32).to_le_bytes())
                .collect();
            muxer
                .write_frame(offset as i64, &data)
                .expect("write NUT packet");
            offset += samples;
        }
        muxer.finish().expect("finish the NUT stream");
    }
    wire
}

/// Every packet on an audio wire as its f32 samples, with the timestamp it
/// carries.
fn audio_packets(wire: &[u8]) -> Vec<(i64, Vec<f32>)> {
    let mut demuxer = Demuxer::open(wire).expect("read NUT headers");
    let mut out = Vec::new();
    let mut buf = Vec::new();
    while let Some(pts) = demuxer.read_frame(&mut buf).expect("read a NUT packet") {
        out.push((pts, {
            let (whole, _) = buf.as_chunks::<4>();
            whole.iter().copied().map(f32::from_le_bytes).collect()
        }));
    }
    out
}

/// Every sample on an audio wire, in order, with the packet boundaries gone.
fn audio_samples(wire: &[u8]) -> Vec<f32> {
    audio_packets(wire)
        .into_iter()
        .flat_map(|(_, samples)| samples)
        .collect()
}

/// One row rms emits per window. `time` is not rms' own key - the host
/// stamps it on from the frame the row rode - so it is the collision case's
/// other side: a row that already names `pts` still gets `time` added.
#[derive(serde::Deserialize)]
struct RmsRow {
    pts: i64,
    samples: usize,
    rms: f64,
    time: f64,
}

/// The rows on stdout of a run whose only output was `-f ndjson -`.
fn ndjson_rows<T: serde::de::DeserializeOwned>(stdout: Vec<u8>) -> Vec<T> {
    String::from_utf8(stdout)
        .expect("ndjson is valid UTF-8")
        .lines()
        .filter(|line| !line.is_empty())
        .map(|line| serde_json::from_str(line).unwrap_or_else(|e| panic!("parsing {line:?}: {e}")))
        .collect()
}

#[test]
fn describe_publishes_what_an_audio_module_accepts() {
    let again = describe(&module_path("again"));
    assert_eq!(again.name, Some("again".to_string()));
    assert_eq!(
        again.pixel_formats,
        Some(vec![]),
        "an audio module publishes no pixel formats, which is what says it is one"
    );
    assert_eq!(
        again.sample_formats,
        Some(vec!["f32".to_string(), "s16".to_string()]),
        "gain works on both wire formats"
    );
    assert_eq!(
        (again.sample_rates, again.channel_counts),
        (Some(vec![]), Some(vec![])),
        "empty is every rate and every channel count"
    );
    assert_eq!(
        (again.window, again.stride),
        (Some(1024), Some(1024)),
        "for an audio module a window is samples, and disjoint windows are what let it be one-to-one"
    );
    assert_eq!(again.one_to_one, Some(true));
}

#[test]
fn describe_leaves_a_video_modules_audio_declarations_empty() {
    let invert = describe(&module_path("invert"));
    assert_eq!(invert.pixel_formats, Some(vec!["rgba".to_string()]));
    assert_eq!(
        (
            invert.sample_formats,
            invert.sample_rates,
            invert.channel_counts
        ),
        (Some(vec![]), Some(vec![]), Some(vec![])),
        "a video module publishes no sample formats, which is what says it is one"
    );
}

#[test]
fn gain_scales_every_sample_and_returns_the_samples_it_was_handed() {
    // Packets of 700 samples against a window of 1024: the host re-cuts them,
    // so what leaves is windows and not the packets that arrived.
    let wire = audio_wire(&[700, 700, 700, 700]);
    let run = run_filter(
        &module_path("again"),
        &["-params", r#"{"gain":0.5}"#],
        &wire,
    );
    assert_run_ok(&run, "again");

    let before = audio_samples(&wire);
    let after = audio_samples(&run.stdout);
    assert_eq!(
        after.len(),
        before.len(),
        "a one-to-one audio module returns the samples it was handed"
    );
    for (index, (was, now)) in before.iter().zip(&after).enumerate() {
        assert!(
            (was * 0.5 - now).abs() < 1e-6,
            "sample {index}: {was} at half gain is not {now}"
        );
    }

    let packets = audio_packets(&run.stdout);
    let starts: Vec<i64> = packets.iter().map(|(pts, _)| *pts).collect();
    assert_eq!(
        starts,
        vec![0, 1024, 2048],
        "the output is windows of 1024, whatever the packets that arrived were"
    );
    assert_eq!(
        packets.last().expect("a final window").1.len(),
        2800 - 2048,
        "the final call carries whatever the last stride left over"
    );
}

#[test]
fn a_window_is_gathered_out_of_many_packets_and_carries_one_row() {
    // rms has a window of 16000 and the packets are 1500, so eleven of them
    // make a window: nothing about the packetization reaches the module.
    let packets = vec![1500usize; 32];
    let wire = audio_wire(&packets);
    let total: usize = packets.iter().sum();

    let module = module_path("rms");
    let module_str = module.to_str().expect("module path is valid UTF-8");
    let run = run_ffrwd_wasm(
        &[
            "-f", "nut", "-i", "-", "-m", module_str, "-f", "ndjson", "-",
        ],
        &wire,
    );
    assert_run_ok(&run, "rms");

    let rows: Vec<RmsRow> = ndjson_rows(run.stdout);
    assert_eq!(
        rows.len(),
        total.div_ceil(16_000),
        "one row per window, and the windows are ceil(samples / stride) of them"
    );
    for (index, row) in rows.iter().enumerate() {
        assert_eq!(
            row.pts,
            index as i64 * 16_000,
            "row {index} starts where the stride left off, rms' own value untouched by the host"
        );
        assert_eq!(
            row.samples,
            16_000.min(total - index * 16_000),
            "row {index} spans to the next window, or to the end of the stream"
        );
        // rms names no time of its own, so the host adds it: the window's
        // pts (a sample count) through the stream's 1/48000 time base.
        let expected_time = (index * 16_000) as f64 / RATE as f64;
        assert!(
            (row.time - expected_time).abs() < 1e-9,
            "row {index} time: expected {expected_time}, got {}",
            row.time
        );
    }
}

#[test]
fn an_audio_edge_into_a_video_module_is_refused_naming_both() {
    let wire = audio_wire(&[1024]);
    let run = run_network(
        &[("facebox", "facebox")],
        "[0:a]facebox[out0]",
        &[],
        ONE_OUTPUT,
        &wire,
    );
    assert!(!run.success(), "facebox reads pictures, not samples");
    assert!(
        run.stderr.contains("[0:a]") && run.stderr.contains("facebox"),
        "expected the edge and the module named, got:\n{}",
        run.stderr
    );
}

#[test]
fn a_video_edge_into_an_audio_module_is_refused_naming_both() {
    let wire = nut_stream(&[synthetic_frame(0)]);
    let run = run_network(
        &[("again", "again")],
        "[0:v]again[out0]",
        &[],
        ONE_OUTPUT,
        &wire,
    );
    assert!(!run.success(), "again reads samples, not pictures");
    assert!(
        run.stderr.contains("[0:v]") && run.stderr.contains("again"),
        "expected the edge and the module named, got:\n{}",
        run.stderr
    );
}

#[test]
fn an_input_label_that_names_the_wrong_kind_is_refused_naming_the_edge() {
    let wire = audio_wire(&[1024]);
    let run = run_network(
        &[("again", "again")],
        "[0:v]again[out0]",
        &[],
        ONE_OUTPUT,
        &wire,
    );
    assert!(!run.success(), "input 0 is audio and the label says video");
    assert!(
        run.stderr.contains("[0:v]") && run.stderr.contains("audio"),
        "expected the edge and what the input carries named, got:\n{}",
        run.stderr
    );
}

#[test]
fn a_module_of_an_older_world_handed_audio_is_refused_naming_its_world() {
    // Every world before this one opens an instance with a frame size and a
    // pixel format, so there is nowhere for a sample rate to go.
    let wire = audio_wire(&[1024]);
    let run = run_filter(&module_path("adapted_060"), &[], &wire);
    assert!(!run.success(), "a 0.6.0 module hosts video alone");
    assert!(
        run.stderr.contains("adapted-060") && run.stderr.contains("ffrwd:av@0.6.0"),
        "expected the module and its world named, got:\n{}",
        run.stderr
    );
}

#[test]
fn rows_ride_audio_from_one_module_to_the_next_inside_the_process() {
    // rms measures each window and again carries the rows on: the levels reach
    // the ndjson output through a module that never read them.
    let packets = vec![1500usize; 32];
    let wire = audio_wire(&packets);
    let total: usize = packets.iter().sum();

    let run = run_network(
        &[("rms", "rms"), ("again", "again")],
        "[0:a]rms[a];[a]again=gain=0.5[out0]",
        &[],
        &["-map", "[out0]", "-f", "ndjson", "-"],
        &wire,
    );
    assert_run_ok(&run, "the rms-into-again network");

    let rows: Vec<RmsRow> = ndjson_rows(run.stdout);
    assert_eq!(
        rows.len(),
        total.div_ceil(16_000),
        "every level rms measured reaches the output through again"
    );
    assert_eq!(rows[0].pts, 0);
    assert_eq!(
        rows[0].time, 0.0,
        "the host stamps time even through a forwarding module"
    );
    assert!(rows[0].rms > 0.0, "a ramp is not silence");
}

#[test]
fn a_trailing_row_crosses_an_audio_chain() {
    // The record arrives on the wire with no packet to ride, reaches again on
    // its final call, and leaves the same way - which is what a module that
    // forwards rows does with one.
    let stream = an_audio_stream("f32", 1);
    let mut wire = Vec::new();
    {
        let mut muxer = Muxer::with_annotations(&mut wire, &stream).expect("write NUT headers");
        let data: Vec<u8> = (0..2000u32)
            .flat_map(|s| (s as f32).to_le_bytes())
            .collect();
        muxer.write_frame(0, &data).expect("write NUT packet");
        muxer
            .write_trailing(0, &[r#"{"level":"quiet"}"#.to_string()])
            .expect("write the trailing record");
        muxer.finish().expect("finish the NUT stream");
    }

    let run = run_filter(
        &module_path("again"),
        &["-annotations", "in", "-annotations", "out"],
        &wire,
    );
    assert_run_ok(&run, "again");

    let mut demuxer = Demuxer::open_annotated(&run.stdout[..]).expect("read the annotated headers");
    let mut buf = Vec::new();
    while demuxer
        .read_frame(&mut buf)
        .expect("read a packet")
        .is_some()
    {}
    assert_eq!(
        demuxer.take_trailing(),
        vec![r#"{"level":"quiet"}"#.to_string()],
        "the record crosses a module that forwards rows unstamped: a trailing \
         row rides no frame, so the host adds no pts or time to it"
    );
}

#[test]
fn an_audio_chain_and_a_video_chain_share_one_process() {
    // Two inputs, two outputs, nothing in common but the process: what proves
    // they are separate is that each output is the shape its own chain makes.
    let audio = audio_wire(&[1024, 1024]);
    let video = nut_stream(&[synthetic_frame(0), synthetic_frame(1)]);

    let video_file = TempFile::new("mixed-video-in.nut");
    std::fs::write(video_file.path(), &video).expect("write the video input");
    let video_out = TempFile::new("mixed-video-out.nut");
    let audio_out = TempFile::new("mixed-audio-out.nut");

    let invert = module_path("invert");
    let again = module_path("again");
    let run = run_ffrwd_wasm(
        &[
            "-f",
            "nut",
            "-i",
            video_file.path().to_str().expect("path is valid UTF-8"),
            "-f",
            "nut",
            "-i",
            "-",
            "-m",
            &format!("invert={}", invert.to_str().expect("valid UTF-8")),
            "-m",
            &format!("again={}", again.to_str().expect("valid UTF-8")),
            "-filter_complex",
            "[0:v]invert[outv];[1:a]again=gain=0.5[outa]",
            "-map",
            "[outv]",
            "-f",
            "nut",
            video_out.path().to_str().expect("path is valid UTF-8"),
            "-map",
            "[outa]",
            "-f",
            "nut",
            audio_out.path().to_str().expect("path is valid UTF-8"),
        ],
        &audio,
    );
    assert_run_ok(&run, "the mixed network");

    let frames = read_nut(&std::fs::read(video_out.path()).expect("read the video output"));
    assert_eq!(frames.len(), 2, "both frames come through invert");
    for (index, (_, data)) in frames.iter().enumerate() {
        let want = synthetic_frame(index as u8);
        assert_eq!(data[0], 255 - want[0], "frame {index} is inverted");
    }

    let out = std::fs::read(audio_out.path()).expect("read the audio output");
    let before = audio_samples(&audio);
    let after = audio_samples(&out);
    assert_eq!(
        after.len(),
        before.len(),
        "every sample comes through again"
    );
    for (index, (was, now)) in before.iter().zip(&after).enumerate() {
        assert!(
            (was * 0.5 - now).abs() < 1e-6,
            "sample {index}: {was} at half gain is not {now}"
        );
    }
}

#[test]
fn a_windowed_module_of_the_060_world_still_runs() {
    // adapted-060 reads the time base as a field, forwards the rows arriving
    // with its frames, and answers the trailing record with one of its own -
    // the three things that world added, all through its adapter.
    let frames: Vec<(Vec<u8>, Vec<String>)> = (0..4u8)
        .map(|i| (synthetic_frame(i), vec![rect_row(u32::from(i), 0, 2, 2)]))
        .collect();
    let wire = annotated_nut_stream_with_trailing(&frames, &[r#"{"frames":4}"#.to_string()]);

    let run = run_filter(
        &module_path("adapted_060"),
        &["-annotations", "in", "-annotations", "out"],
        &wire,
    );
    assert_run_ok(&run, "adapted-060");

    let (got, trailing) = annotated_frames_and_trailing(&run.stdout);
    assert_eq!(got.len(), frames.len(), "every frame passes through");

    let first: TimeBaseRow = serde_json::from_str(&got[0].2[0])
        .unwrap_or_else(|e| panic!("parsing {:?}: {e}", got[0].2[0]));
    assert_eq!(
        first.time_base, "1/65536",
        "that world reads the time base as a field of the stream info"
    );
    assert_eq!(first.rows_in, 2, "both frames of the window carried a row");
    assert_eq!(
        parsed_row(&got[0].2[1]),
        stamped_row(&frames[0].1[0], got[0].0, TIME_BASE),
        "the arriving row travels on beside the module's own, gaining the \
         pts/time stamp every row that rides a frame gets"
    );
    assert_eq!(
        trailing,
        vec![r#"{"trailing-in":1}"#.to_string()],
        "the record on the wire reached its final call, and its answer left the same way, \
         unstamped since it rides no frame"
    );
}

// Subtitle outputs. NUT keeps no per-packet duration, so the cues do not ride
// the edge stream: they are gathered and written whole to an output of their
// own. `captions` is the fixture that emits them.

/// Runs `captions` over `frames` frames and returns what its subtitle output
/// wrote, plus the run itself.
fn captions_run(frames: u8, format: &str, params: &[&str]) -> (FfrwdWasmRun, TempFile) {
    ensure_modules_built();
    let wire = nut_stream(&(0..frames).map(synthetic_frame).collect::<Vec<_>>());
    let out = TempFile::new(&format!("captions.{format}"));
    let module = module_path("captions");
    let module_str = module.to_str().expect("module path is valid UTF-8");
    let out_str = out.path().to_str().expect("path is valid UTF-8");

    let mut args: Vec<&str> = vec!["-f", "nut", "-i", "-", "-m", module_str];
    args.extend_from_slice(params);
    args.extend_from_slice(&["-f", "nut", "-", "-f", format, out_str]);
    (run_ffrwd_wasm(&args, &wire), out)
}

#[test]
fn cue_rows_become_an_srt_document() {
    let (run, out) = captions_run(3, "srt", &[]);
    assert_run_ok(&run, "captions into an srt output");

    // 25fps in a 1/65536 base, so the frames sit 0.04s apart and each cue runs
    // half a second from where its frame is.
    assert_eq!(
        out.text(),
        "1\n00:00:00,000 --> 00:00:00,500\ncue 0\n\n\
         2\n00:00:00,040 --> 00:00:00,540\ncue 1\n\n\
         3\n00:00:00,080 --> 00:00:00,580\ncue 2\n\n",
        "the cues are numbered from one in the order their rows arrived"
    );
    assert_eq!(
        run.frames().len(),
        3,
        "the frames still leave on the nut output"
    );
}

#[test]
fn a_trailing_record_that_is_not_a_cue_is_passed_over() {
    // captions ends the stream with `{"cues":N}`, the other arm of its rows
    // schema. It reaches the subtitle output and is not a cue, so it is not
    // one of them and it is not a refusal either.
    let (run, out) = captions_run(2, "srt", &[]);
    assert_run_ok(&run, "captions into an srt output");
    let document = out.text();
    assert!(
        !document.contains("cues"),
        "the record that is not a cue is not in the document, got:\n{document}"
    );
    assert_eq!(
        document.matches(" --> ").count(),
        2,
        "two frames, two cues, got:\n{document}"
    );
}

#[test]
fn cue_rows_become_a_webvtt_document() {
    let (run, out) = captions_run(1, "webvtt", &[]);
    assert_run_ok(&run, "captions into a webvtt output");
    assert_eq!(
        out.text(),
        "WEBVTT\n\n1\n00:00:00.000 --> 00:00:00.500\ncue 0\n\n"
    );
}

#[test]
fn a_row_that_is_not_a_whole_cue_is_refused_naming_the_output_and_the_field() {
    let (run, _out) = captions_run(2, "srt", &["-params", r#"{"malformed":true}"#]);
    assert!(!run.success(), "a cue with no end time is not a cue");
    assert!(
        run.stderr.contains("-f srt") && run.stderr.contains("end_t"),
        "expected the output and the missing field named, got:\n{}",
        run.stderr
    );
}

#[test]
fn a_second_subtitle_output_of_the_same_format_is_refused() {
    ensure_modules_built();
    let module = module_path("captions");
    let module_str = module.to_str().expect("module path is valid UTF-8");
    let first = TempFile::new("captions-first.srt");
    let second = TempFile::new("captions-second.srt");
    let run = run_ffrwd_wasm(
        &[
            "-f",
            "nut",
            "-i",
            "-",
            "-m",
            module_str,
            "-f",
            "srt",
            first.path().to_str().expect("path is valid UTF-8"),
            "-f",
            "srt",
            second.path().to_str().expect("path is valid UTF-8"),
        ],
        &nut_stream(&[synthetic_frame(0)]),
    );
    assert!(!run.success(), "a single module writes one of each");
    assert!(
        run.stderr.contains("second -f srt output"),
        "expected the second output named, got:\n{}",
        run.stderr
    );
}

#[test]
fn an_output_format_that_is_not_supported_is_refused_listing_the_ones_that_are() {
    ensure_modules_built();
    let module = module_path("captions");
    let module_str = module.to_str().expect("module path is valid UTF-8");
    let run = run_ffrwd_wasm(
        &["-f", "nut", "-i", "-", "-m", module_str, "-f", "ass", "-"],
        &nut_stream(&[synthetic_frame(0)]),
    );
    assert!(!run.success(), "ass is not an output this writes");
    for format in ["nut", "ndjson", "srt", "webvtt"] {
        assert!(
            run.stderr.contains(format),
            "expected {format} among the ones listed, got:\n{}",
            run.stderr
        );
    }
}

#[test]
fn a_network_names_the_label_its_subtitle_output_writes() {
    ensure_modules_built();
    let out = TempFile::new("network-captions.srt");
    let run = run_network(
        &[("captions", "captions")],
        "[0:v]captions[out0]",
        &[],
        &[
            "-map",
            "[out0]",
            "-f",
            "srt",
            out.path().to_str().expect("path is valid UTF-8"),
        ],
        &nut_stream(&(0..2u8).map(synthetic_frame).collect::<Vec<_>>()),
    );
    assert_run_ok(&run, "a network writing a subtitle output");
    assert_eq!(
        out.text().matches(" --> ").count(),
        2,
        "both cues reached the document, got:\n{}",
        out.text()
    );
}

/// What a module of the world that first carried audio reports back.
#[derive(serde::Deserialize)]
struct OpenedAsRow {
    #[serde(rename = "opened-as")]
    opened_as: String,
    #[serde(rename = "rows-in")]
    rows_in: usize,
}

#[test]
fn a_windowed_audio_module_of_the_world_before_this_one_still_runs() {
    // adapted-070 is opened with a kind-bearing format - the thing that world
    // added - and its windows count samples. It passes the samples through,
    // forwards the rows arriving with them, and answers the trailing record
    // with one of its own, all through its adapter.
    let stream = an_audio_stream("f32", 1);
    let packets = [0usize, 1024];
    let mut wire = Vec::new();
    {
        let mut muxer = Muxer::with_annotations(&mut wire, &stream).expect("write NUT headers");
        for offset in packets {
            let data: Vec<u8> = (offset..offset + 1024)
                .flat_map(|s| (s as f32).to_le_bytes())
                .collect();
            muxer
                .write_rows(offset as i64, &[format!(r#"{{"at":{offset}}}"#)])
                .expect("write the rows");
            muxer
                .write_frame(offset as i64, &data)
                .expect("write NUT packet");
        }
        muxer
            .write_trailing(1024, &[r#"{"level":"quiet"}"#.to_string()])
            .expect("write the trailing record");
        muxer.finish().expect("finish the NUT stream");
    }

    let run = run_filter(
        &module_path("adapted_070"),
        &["-annotations", "in", "-annotations", "out"],
        &wire,
    );
    assert_run_ok(&run, "adapted-070");

    let (got, trailing) = annotated_frames_and_trailing(&run.stdout);
    assert_eq!(got.len(), packets.len(), "every window passes through");

    let first: OpenedAsRow = serde_json::from_str(&got[0].2[0])
        .unwrap_or_else(|e| panic!("parsing {:?}: {e}", got[0].2[0]));
    assert_eq!(
        first.opened_as, "48000 Hz, 1 channel(s), f32",
        "that world is handed the audio arm of a kind-bearing format"
    );
    assert_eq!(first.rows_in, 1, "the window carried the row that arrived");
    let audio_time_base = TimeBase {
        num: 1,
        den: RATE as u64,
    };
    assert_eq!(
        parsed_row(&got[0].2[1]),
        stamped_row(r#"{"at":0}"#, got[0].0, audio_time_base),
        "the arriving row travels on beside the module's own, gaining the \
         pts/time stamp every row that rides a frame gets"
    );
    assert_eq!(
        trailing,
        vec![r#"{"trailing-in":1}"#.to_string()],
        "the record on the wire reached its final call, and its answer left the same way, \
         unstamped since it rides no frame"
    );
}

/// What a module of the world before this one reports back.
#[derive(serde::Deserialize)]
struct LanguageRow {
    language: String,
    #[serde(rename = "rows-in")]
    rows_in: usize,
}

#[test]
fn a_windowed_video_module_of_the_world_before_this_one_still_runs() {
    // adapted-080 is of the world that named the language of a module's rows
    // and had no place to say how many streams it reads. It is driven as one
    // stream, a frame at a time, forwarding the rows arriving with each and
    // answering the trailing record with one of its own - all through its
    // adapter.
    let frames: Vec<(Vec<u8>, Vec<String>)> = (0..3u8)
        .map(|i| (synthetic_frame(i), vec![rect_row(u32::from(i), 0, 2, 2)]))
        .collect();
    let wire = annotated_nut_stream_with_trailing(&frames, &[r#"{"frames":3}"#.to_string()]);

    let run = run_filter(
        &module_path("adapted_080"),
        &[
            "-params",
            r#"{"language":"cy"}"#,
            "-annotations",
            "in",
            "-annotations",
            "out",
        ],
        &wire,
    );
    assert_run_ok(&run, "adapted-080");

    let (got, trailing) = annotated_frames_and_trailing(&run.stdout);
    assert_eq!(got.len(), frames.len(), "every frame passes through");

    let first: LanguageRow = serde_json::from_str(&got[0].2[0])
        .unwrap_or_else(|e| panic!("parsing {:?}: {e}", got[0].2[0]));
    assert_eq!(
        first.language, "cy",
        "that world names the param its rows' language comes from"
    );
    assert_eq!(first.rows_in, 1, "the frame carried the row that arrived");
    assert_eq!(
        parsed_row(&got[0].2[1]),
        stamped_row(&frames[0].1[0], got[0].0, TIME_BASE),
        "the arriving row travels on beside the module's own, gaining the \
         pts/time stamp every row that rides a frame gets"
    );
    assert_eq!(
        trailing,
        vec![r#"{"trailing-in":1}"#.to_string()],
        "the record on the wire reached its final call, and its answer left the same way, \
         unstamped since it rides no frame"
    );
}

/// Runs the misbehaving fixture over a second of audio in `wrong` mode, and
/// returns what the host said about it.
fn broken_audio(wrong: &str) -> FfrwdWasmRun {
    let wire = audio_wire(&[1024; 8]);
    let params = format!(r#"{{"wrong":"{wrong}"}}"#);
    let run = run_filter(&module_path("broken_audio"), &["-params", &params], &wire);
    assert!(
        !run.success(),
        "{wrong} breaks a rule the host holds an audio module to"
    );
    run
}

#[test]
fn passing_an_overlapping_window_through_is_refused_naming_the_module() {
    let run = broken_audio("same");
    assert!(
        run.stderr.contains("broken-audio")
            && run.stderr.contains("stride")
            && run.stderr.contains("window"),
        "expected the module and the rule named, got:\n{}",
        run.stderr
    );
}

#[test]
fn a_hole_in_a_one_to_one_modules_output_is_refused_naming_the_module() {
    // A quarter of each window leaves, so the second call starts past where
    // the first one ended and the samples between are nowhere.
    let run = broken_audio("gap");
    assert!(
        run.stderr.contains("broken-audio") && run.stderr.contains("no gap and no overlap"),
        "expected the module and the rule named, got:\n{}",
        run.stderr
    );
}

#[test]
fn a_one_to_one_module_that_ends_short_is_refused_naming_the_module() {
    // Each call looks continuous on its own; what does not add up is the
    // instance's whole life, which is where the host catches it.
    let run = broken_audio("short");
    assert!(
        run.stderr.contains("broken-audio")
            && run.stderr.contains("returns the samples it was handed"),
        "expected the module and the rule named, got:\n{}",
        run.stderr
    );
}

#[test]
fn a_packet_that_is_not_whole_samples_is_refused() {
    // Thirteen bytes is three f32 samples and a byte over, which names no
    // sample at all.
    let stream = an_audio_stream("f32", 1);
    let mut wire = Vec::new();
    {
        let mut muxer = Muxer::new(&mut wire, &stream).expect("write NUT headers");
        muxer
            .write_frame(0, &[0u8; 13])
            .expect("write a short NUT packet");
        muxer.finish().expect("finish the NUT stream");
    }

    let run = run_filter(&module_path("again"), &[], &wire);
    assert!(!run.success(), "13 bytes is not a whole number of samples");
    assert!(
        run.stderr.contains("13 bytes") && run.stderr.contains("whole number of samples"),
        "expected the byte count and the rule named, got:\n{}",
        run.stderr
    );
}

#[test]
fn a_stereo_sample_is_every_channel_at_one_instant() {
    // Two channels of s16 is four bytes a sample, so the same byte count is
    // half as many samples - which is what the window counts.
    let stream = an_audio_stream("s16", 2);
    let mut wire = Vec::new();
    {
        let mut muxer = Muxer::new(&mut wire, &stream).expect("write NUT headers");
        // 3000 samples: 1024 twice over, then 952 on the final call.
        let data: Vec<u8> = (0..3000i16)
            .flat_map(|s| [s.to_le_bytes(), s.to_le_bytes()])
            .flatten()
            .collect();
        muxer.write_frame(0, &data).expect("write NUT packet");
        muxer.finish().expect("finish the NUT stream");
    }

    let run = run_filter(&module_path("again"), &["-params", r#"{"gain":2}"#], &wire);
    assert_run_ok(&run, "again on a stereo wire");

    let mut demuxer = Demuxer::open(&run.stdout[..]).expect("read NUT headers");
    assert_eq!(demuxer.stream().audio_geometry(), Some((RATE, 2)));
    let mut spans = Vec::new();
    let mut buf = Vec::new();
    while let Some(pts) = demuxer.read_frame(&mut buf).expect("read a packet") {
        spans.push((pts, buf.len() / 4));
    }
    assert_eq!(
        spans,
        vec![(0, 1024), (1024, 1024), (2048, 952)],
        "a window of 1024 stereo samples is 4096 bytes, and the tail is what is left"
    );
}

/// One option value escaped the way a caller must write it into a network
/// string: once for the option list, then again for the graph.
fn escape_option(value: &str) -> String {
    /// The option list splits on ':', the graph on its own set, and both
    /// honour the same quoting and escaping.
    const OPTION_LEVEL: [char; 3] = ['\\', '\'', ':'];
    const GRAPH_LEVEL: [char; 6] = ['\\', '\'', '[', ']', ',', ';'];

    fn escape(value: &str, level: &[char]) -> String {
        let mut out = String::with_capacity(value.len());
        for c in value.chars() {
            if level.contains(&c) {
                out.push('\\');
            }
            out.push(c);
        }
        out
    }

    escape(&escape(value, &OPTION_LEVEL), &GRAPH_LEVEL)
}

/// A `rowfilter` node carrying `pred`, spelled as a network writes it.
fn rows_node(input: &str, pred: &str, output: &str) -> String {
    format!("[{input}]rowfilter=pred={}[{output}]", escape_option(pred))
}

#[test]
fn the_rows_node_keeps_only_the_rows_its_predicate_matches() {
    // framestats writes one row a frame and one trailing row counting them,
    // so a predicate naming a field only the trailing row carries separates
    // the two without touching the frames.
    let frames: Vec<Vec<u8>> = (0..5u8).map(synthetic_frame).collect();
    let wiring = format!(
        "[0:v]framestats[a];{}",
        rows_node("a", r#"{"ge":[{"field":"frames"},{"lit":1}]}"#, "out0")
    );

    let run = run_network(
        &[("framestats", "framestats")],
        &wiring,
        &[],
        &["-map", "[out0]", "-f", "ndjson", "-"],
        &nut_stream(&frames),
    );
    assert_run_ok(&run, "framestats into the rows node");

    let rows = String::from_utf8(run.stdout).expect("ndjson is valid UTF-8");
    let rows: Vec<&str> = rows.lines().filter(|l| !l.is_empty()).collect();
    assert_eq!(
        rows,
        vec![r#"{"frames":5}"#],
        "the per-frame rows carry no 'frames' at all, so only the trailing one is kept"
    );
}

#[test]
fn the_rows_node_judges_trailing_rows_by_the_same_predicate() {
    // The other way about: a field every per-frame row carries and the
    // trailing one does not.
    let frames: Vec<Vec<u8>> = (0..5u8).map(synthetic_frame).collect();
    let wiring = format!(
        "[0:v]framestats[a];{}",
        rows_node("a", r#"{"ge":[{"field":"time"},{"lit":0}]}"#, "out0")
    );

    let run = run_network(
        &[("framestats", "framestats")],
        &wiring,
        &[],
        &["-map", "[out0]", "-f", "ndjson", "-"],
        &nut_stream(&frames),
    );
    assert_run_ok(&run, "framestats into the rows node");

    let rows = String::from_utf8(run.stdout).expect("ndjson is valid UTF-8");
    let kept: Vec<&str> = rows.lines().filter(|l| !l.is_empty()).collect();
    assert_eq!(
        kept.len(),
        5,
        "one row a frame, and the trailing row dropped"
    );
    assert!(
        kept.iter().all(|row| row.contains("\"time\"")),
        "got: {kept:?}"
    );
}

#[test]
fn the_rows_node_hands_on_the_frames_it_was_given() {
    // The same network with and without it: the pixels must not move.
    let frames: Vec<Vec<u8>> = (0..4u8).map(synthetic_frame).collect();
    let wire = nut_stream(&frames);

    let plain = run_network(
        &[("invert", "invert")],
        "[0:v]invert[out0]",
        &[],
        ONE_OUTPUT,
        &wire,
    );
    assert_run_ok(&plain, "invert alone");

    let wiring = format!(
        "[0:v]invert[a];{}",
        rows_node("a", r#"{"eq":[{"field":"nothing"},{"lit":1}]}"#, "out0")
    );
    let filtered = run_network(&[("invert", "invert")], &wiring, &[], ONE_OUTPUT, &wire);
    assert_run_ok(&filtered, "invert into the rows node");

    assert_eq!(
        read_nut(&filtered.stdout),
        read_nut(&plain.stdout),
        "the rows node judges rows and nothing else"
    );
}

#[test]
fn a_module_bound_as_the_rows_node_is_refused_naming_it() {
    let frames: Vec<Vec<u8>> = (0..2u8).map(synthetic_frame).collect();
    let run = run_network(
        &[("rowfilter", "invert")],
        "[0:v]rowfilter[out0]",
        &[],
        ONE_OUTPUT,
        &nut_stream(&frames),
    );
    assert!(!run.success(), "the name is the host's own");
    assert!(
        run.stderr.contains("rowfilter") && run.stderr.contains("no module is bound to it"),
        "expected the reserved name named, got:\n{}",
        run.stderr
    );
}

#[test]
fn a_rows_node_with_no_predicate_is_refused_naming_the_option() {
    let frames: Vec<Vec<u8>> = (0..2u8).map(synthetic_frame).collect();
    let run = run_network(
        &[("invert", "invert")],
        "[0:v]invert[a];[a]rowfilter[out0]",
        &[],
        ONE_OUTPUT,
        &nut_stream(&frames),
    );
    assert!(!run.success(), "pred is required");
    assert!(
        run.stderr.contains("pred=<json>"),
        "expected the option named, got:\n{}",
        run.stderr
    );
}

#[test]
fn a_predicate_that_is_not_a_predicate_is_refused_before_any_frame() {
    let frames: Vec<Vec<u8>> = (0..2u8).map(synthetic_frame).collect();
    let wiring = format!(
        "[0:v]invert[a];{}",
        rows_node("a", r#"{"between":[1,2]}"#, "out0")
    );
    let run = run_network(
        &[("invert", "invert")],
        &wiring,
        &[],
        ONE_OUTPUT,
        &nut_stream(&frames),
    );
    assert!(!run.success(), "the operator does not exist");
    assert!(
        run.stderr.contains("has no operator") && run.stderr.contains("rowfilter"),
        "expected the node and the operators named, got:\n{}",
        run.stderr
    );
}

#[test]
fn a_predicate_carrying_quotes_and_colons_survives_the_network_string() {
    // brand takes a title, so a row language is not needed to prove the
    // predicate arrived intact: a field name with a colon in it can only
    // match if both levels of escaping were undone.
    let frames: Vec<Vec<u8>> = (0..3u8).map(synthetic_frame).collect();
    let wiring = format!(
        "[0:v]framestats[a];{}",
        rows_node("a", r#"{"eq":[{"field":"a:b'c"},{"lit":"x,y;z"}]}"#, "out0")
    );

    let run = run_network(
        &[("framestats", "framestats")],
        &wiring,
        &[],
        &["-map", "[out0]", "-f", "ndjson", "-"],
        &nut_stream(&frames),
    );
    // No row carries that field, so everything is dropped - but the run must
    // reach the frames at all, which a mis-escaped predicate never would.
    assert_run_ok(&run, "a predicate full of separators");
    assert!(
        run.stdout.is_empty(),
        "no row carries the field, so none survive"
    );
}

/// A tiny HTTP endpoint: accepts connections until dropped, answers 200 to
/// every request and hands back each request's body line.
fn collecting_endpoint() -> (String, std::sync::mpsc::Receiver<String>) {
    use std::io::{BufRead, BufReader, Read, Write};

    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind endpoint");
    let address = listener.local_addr().expect("endpoint address");
    let (sender, receiver) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(stream) = stream else { break };
            let mut reader = BufReader::new(stream);
            let mut line = String::new();
            let mut length = 0usize;
            if reader.read_line(&mut line).is_err() {
                continue;
            }
            loop {
                let mut header = String::new();
                if reader.read_line(&mut header).is_err() {
                    break;
                }
                let header = header.trim_end();
                if header.is_empty() {
                    break;
                }
                if let Some(value) = header.to_ascii_lowercase().strip_prefix("content-length:") {
                    length = value.trim().parse().unwrap_or(0);
                }
            }
            let mut body = vec![0u8; length];
            if reader.read_exact(&mut body).is_err() {
                continue;
            }
            let _ = sender.send(String::from_utf8_lossy(&body).into_owned());
            let mut stream = reader.into_inner();
            let _ = stream
                .write_all(b"HTTP/1.1 200 OK\r\ncontent-length: 0\r\nconnection: close\r\n\r\n");
        }
    });
    (format!("http://{address}/rows"), receiver)
}

#[test]
fn a_sink_module_consumes_frames_and_a_null_output_carries_nothing() {
    ensure_modules_built();
    let frames: Vec<Vec<u8>> = (0..3u8).map(synthetic_frame).collect();
    let module = module_path("frame_stats");
    let module_str = module.to_str().expect("module path is valid UTF-8");

    let run = run_ffrwd_wasm(
        &["-f", "nut", "-i", "-", "-m", module_str, "-f", "null", "-"],
        &nut_stream(&frames),
    );
    assert_run_ok(&run, "frame_stats to a null output");
    assert!(
        run.stdout.is_empty(),
        "a null output writes nothing, got {} bytes",
        run.stdout.len()
    );
    for i in 0..3 {
        assert!(
            run.stderr.contains(&format!("frame_stats frame={i}")),
            "expected a stats line for frame {i}, got:\n{}",
            run.stderr
        );
    }
    assert!(
        run.stderr.contains("frame_stats frames=3"),
        "expected the closing count, got:\n{}",
        run.stderr
    );
}

#[test]
fn an_http_module_is_refused_without_the_grant() {
    ensure_modules_built();
    let frames = vec![(synthetic_frame(0), vec![])];
    let module = module_path("post_rows");
    let module_str = module.to_str().expect("module path is valid UTF-8");

    let run = run_ffrwd_wasm(
        &[
            "-f",
            "nut",
            "-i",
            "-",
            "-annotations",
            "in",
            "-m",
            module_str,
            "-params",
            r#"{"url":"http://127.0.0.1:9/rows"}"#,
            "-f",
            "null",
            "-",
        ],
        &annotated_nut_stream(&frames),
    );
    assert!(
        !run.success(),
        "post_rows imports wasi:http and this run granted nothing"
    );
    assert!(
        run.stderr.contains("wasi:http") && run.stderr.contains("-http"),
        "expected the refusal to name the missing grant, got:\n{}",
        run.stderr
    );
}

#[test]
fn an_http_grant_naming_another_module_grants_this_one_nothing() {
    ensure_modules_built();
    let frames = vec![(synthetic_frame(0), vec![])];
    let module = module_path("post_rows");
    let module_str = module.to_str().expect("module path is valid UTF-8");
    let other = module_path("invert");
    let other_str = other.to_str().expect("module path is valid UTF-8");

    let run = run_ffrwd_wasm(
        &[
            "-http",
            other_str,
            "-f",
            "nut",
            "-i",
            "-",
            "-annotations",
            "in",
            "-m",
            module_str,
            "-params",
            r#"{"url":"http://127.0.0.1:9/rows"}"#,
            "-f",
            "null",
            "-",
        ],
        &annotated_nut_stream(&frames),
    );
    assert!(
        !run.success(),
        "the grant is per module, and this one names a different module"
    );
    assert!(
        run.stderr.contains("wasi:http"),
        "expected the missing-grant refusal, got:\n{}",
        run.stderr
    );
}

#[test]
fn an_http_module_posts_each_row_under_its_grant() {
    ensure_modules_built();
    let (url, received) = collecting_endpoint();
    let frames = vec![
        (synthetic_frame(0), vec![rect_row(0, 0, 2, 2)]),
        (synthetic_frame(1), vec![]),
        (synthetic_frame(2), vec![rect_row(4, 4, 2, 2)]),
    ];
    let module = module_path("post_rows");
    let module_str = module.to_str().expect("module path is valid UTF-8");
    let params = format!(r#"{{"url":"{url}"}}"#);

    let run = run_ffrwd_wasm(
        &[
            "-http",
            module_str,
            "-f",
            "nut",
            "-i",
            "-",
            "-annotations",
            "in",
            "-m",
            module_str,
            "-params",
            &params,
            "-f",
            "null",
            "-",
        ],
        &annotated_nut_stream(&frames),
    );
    assert_run_ok(&run, "post_rows under its grant");
    assert!(run.stdout.is_empty(), "a null output writes nothing");

    let mut bodies: Vec<String> = received.try_iter().collect();
    bodies.sort();
    let mut expected = vec![rect_row(0, 0, 2, 2), rect_row(4, 4, 2, 2)];
    expected.sort();
    assert_eq!(bodies, expected, "one POST per row, the row as the body");
}

// A module written in Go, built with TinyGo for wasm32-wasip2. It exports the
// same interface the Rust fleet exports and the host is told nothing about
// where it came from. TinyGo is not a prerequisite of this checkout, so the
// cases below skip on a machine without it rather than failing.

/// The Go modules' workspace, beside the cargo one.
fn go_module_workspace() -> PathBuf {
    sidecar_root().join("modules-go")
}

/// Absolute path to a Go module's built `.wasm` component.
fn go_module_path(name: &str) -> PathBuf {
    go_module_workspace()
        .join("build")
        .join(format!("{name}.wasm"))
}

/// Builds the Go modules once, and says whether they are there to run. The
/// recipe lives in `build.sh` beside them, so this runs that rather than
/// keeping a second copy of it.
fn go_modules_built() -> bool {
    static BUILT: OnceLock<bool> = OnceLock::new();
    *BUILT.get_or_init(|| {
        let workspace = go_module_workspace();
        match Command::new("sh")
            .arg("build.sh")
            .current_dir(&workspace)
            .output()
        {
            Ok(output) if output.status.success() => true,
            Ok(output) => {
                eprintln!(
                    "skipping the Go modules: build.sh exited {:?}\n{}",
                    output.status.code(),
                    String::from_utf8_lossy(&output.stderr)
                );
                false
            }
            Err(e) => {
                eprintln!("skipping the Go modules: nothing to run build.sh with: {e}");
                false
            }
        }
    })
}

/// Runs `ffrwd-wasm` on a Go module, the way `run_filter` runs a Rust one.
fn run_go_filter(name: &str, module_args: &[&str], stdin_bytes: &[u8]) -> FfrwdWasmRun {
    let module_str = go_module_path(name);
    let module_str = module_str.to_str().expect("module path is valid UTF-8");
    let mut args: Vec<&str> = vec!["-f", "nut", "-i", "-", "-m", module_str];
    args.extend_from_slice(module_args);
    args.extend_from_slice(&["-f", "nut", "-"]);
    run_ffrwd_wasm(&args, stdin_bytes)
}

#[test]
fn a_go_module_inverts_the_frames_the_rust_one_inverts() {
    if !go_modules_built() {
        return;
    }
    let frames: Vec<Vec<u8>> = (0..3u8).map(synthetic_frame).collect();
    let wire = nut_stream(&frames);

    let go = run_go_filter("invert_go", &[], &wire);
    assert_run_ok(&go, "invert-go");

    // What inverting is, said again here rather than read off either module:
    // the three colour bytes complemented, the alpha byte left alone.
    let got = go.frames();
    assert_eq!(got.len(), 3, "expected exactly 3 frames back");
    for (i, input) in frames.iter().enumerate() {
        let (pts, output) = &got[i];
        assert_eq!(*pts, i as i64 * PTS_STEP, "frame {i} timestamp");
        for pixel in 0..(WIDTH * HEIGHT) as usize {
            let p = pixel * 4;
            assert_eq!(output[p], 255 - input[p], "frame {i} pixel {pixel} red");
            assert_eq!(
                output[p + 1],
                255 - input[p + 1],
                "frame {i} pixel {pixel} green"
            );
            assert_eq!(
                output[p + 2],
                255 - input[p + 2],
                "frame {i} pixel {pixel} blue"
            );
            assert_eq!(
                output[p + 3],
                input[p + 3],
                "frame {i} pixel {pixel} alpha must be unchanged"
            );
        }
    }

    // And the whole wire matches the Rust module's, so the two agree on the
    // stream header and the timestamps as well as the pixels.
    let rust = run_filter(&module_path("invert"), &[], &wire);
    assert_run_ok(&rust, "invert");
    assert_eq!(
        go.stdout, rust.stdout,
        "the Go module's output must be byte-identical to the Rust one's"
    );
}

#[test]
fn describe_reads_a_go_module_like_any_other() {
    if !go_modules_built() {
        return;
    }
    let parsed = describe(&go_module_path("invert_go"));
    assert_eq!(parsed.world, "ffrwd:av@0.11.0");
    assert_eq!(
        parsed.name,
        Some("invert-go".to_string()),
        "a module names itself, and this one is not the Rust invert"
    );
    assert_eq!(parsed.pixel_formats, Some(vec!["rgba".to_string()]));
    assert_eq!(parsed.window, Some(1), "a per-frame filter is window 1");
    assert_eq!(parsed.stride, Some(1));
    assert_eq!(parsed.inputs, 1, "invert-go reads one stream");
    assert_eq!(parsed.rows_schema, None, "invert-go emits no rows");
}

/// One row as window3-go reports a call: the frames it saw and the
/// timestamps at the two ends of that window.
#[derive(serde::Deserialize)]
struct WindowRow {
    saw: usize,
    first: i64,
    last: i64,
}

#[test]
fn a_go_module_is_driven_with_a_window_wider_than_its_stride() {
    if !go_modules_built() {
        return;
    }
    // window3-go declares window 3, stride 1: a call sees three frames and
    // consumes the oldest, so a stream of N frames makes N-2 full calls, each
    // spanning two steps, and a final call holding the two the strides left.
    // Two frames is the short stream that never fills a window at all: no
    // full call happens, and the final one carries the whole stream.
    for count in [6usize, 2] {
        let frames: Vec<Vec<u8>> = (0..count).map(|i| synthetic_frame(i as u8)).collect();
        let run = run_go_filter("window3_go", &["-annotations", "out"], &nut_stream(&frames));
        assert_run_ok(&run, &format!("window3-go over {count} frames"));

        let got = annotated_frames(&run.stdout);
        assert_eq!(got.len(), count, "every frame passes through, once");
        for (i, (pts, data, _)) in got.iter().enumerate() {
            assert_eq!(*pts, i as i64 * PTS_STEP, "frame {i} timestamp");
            assert_eq!(data, &frames[i], "frame {i} passes through untouched");
        }

        let mut expected: Vec<(usize, i64, i64)> = (0..count.saturating_sub(2))
            .map(|i| (3, i as i64 * PTS_STEP, (i + 2) as i64 * PTS_STEP))
            .collect();
        let tail = count.min(2);
        if tail > 0 {
            let first = (count - tail) as i64 * PTS_STEP;
            expected.push((tail, first, (count - 1) as i64 * PTS_STEP));
        }

        let calls: Vec<(usize, i64, i64)> = got
            .iter()
            .flat_map(|(_, _, rows)| rows.iter())
            .map(|row| {
                let parsed: WindowRow = serde_json::from_str(row)
                    .unwrap_or_else(|e| panic!("parsing window3-go row {row:?}: {e}"));
                (parsed.saw, parsed.first, parsed.last)
            })
            .collect();
        assert_eq!(
            calls, expected,
            "{count} frames through a window of three striding one"
        );
    }
}

#[test]
fn a_go_module_holds_up_at_a_real_frame_size() {
    if !go_modules_built() {
        return;
    }
    // The cases above run on 8x8 frames, which are small enough that a Go
    // module never reaches for more memory. A frame of a size something might
    // actually carry is what makes it, so one runs here - still against what
    // inverting is, and still beside the Rust module.
    const W: u32 = 320;
    const H: u32 = 240;
    let stream = Stream::video("rgba", W, H, TIME_BASE).expect("rgba is carried");
    let frames: Vec<Vec<u8>> = (0..4u8)
        .map(|i| {
            (0..(W * H * 4) as usize)
                .map(|offset| i.wrapping_mul(41).wrapping_add(offset as u8))
                .collect()
        })
        .collect();

    let mut wire = Vec::new();
    {
        let mut muxer = Muxer::new(&mut wire, &stream).expect("write NUT headers");
        for (i, frame) in frames.iter().enumerate() {
            muxer
                .write_frame(i as i64 * PTS_STEP, frame)
                .expect("write NUT frame");
        }
        muxer.finish().expect("finish the NUT stream");
    }

    let go = run_go_filter("invert_go", &[], &wire);
    assert_run_ok(&go, "invert-go on 320x240");

    let got = go.frames();
    assert_eq!(got.len(), frames.len(), "every frame comes back");
    for (i, input) in frames.iter().enumerate() {
        let (pts, output) = &got[i];
        assert_eq!(*pts, i as i64 * PTS_STEP, "frame {i} timestamp");
        assert_eq!(output.len(), input.len(), "frame {i} length");
        for pixel in 0..(W * H) as usize {
            let p = pixel * 4;
            assert_eq!(output[p], 255 - input[p], "frame {i} pixel {pixel} red");
            assert_eq!(
                output[p + 3],
                input[p + 3],
                "frame {i} pixel {pixel} alpha must be unchanged"
            );
        }
    }

    let rust = run_filter(&module_path("invert"), &[], &wire);
    assert_run_ok(&rust, "invert on 320x240");
    assert_eq!(
        go.stdout, rust.stdout,
        "the two modules agree on a real frame as well as a small one"
    );
}

// The same Go modules, built for the same world through componentize-go
// instead of TinyGo: mainline `go build` for GOOS=wasip1 in reactor mode
// (`//go:wasmexport`), wrapped to a wasip2 component with the
// wasi_snapshot_preview1 adapter. No TinyGo anywhere in this chain, and no
// `-gc=leaking` workaround - mainline Go's own collector is what is on trial.
// build-big.sh is not a prerequisite of this checkout either, so these skip
// the same way the TinyGo cases above do.

/// Absolute path to a Go module built by componentize-go instead of TinyGo.
fn go_big_module_path(name: &str) -> PathBuf {
    go_module_workspace()
        .join("build-big")
        .join(format!("{name}.wasm"))
}

/// Builds the componentize-go road's Go modules once, and says whether they
/// are there to run. The recipe lives in `build-big.sh` beside them.
fn go_big_modules_built() -> bool {
    static BUILT: OnceLock<bool> = OnceLock::new();
    *BUILT.get_or_init(|| {
        let workspace = go_module_workspace();
        match Command::new("sh")
            .arg("build-big.sh")
            .current_dir(&workspace)
            .output()
        {
            Ok(output) if output.status.success() => true,
            Ok(output) => {
                eprintln!(
                    "skipping the componentize-go modules: build-big.sh exited {:?}\n{}",
                    output.status.code(),
                    String::from_utf8_lossy(&output.stderr)
                );
                false
            }
            Err(e) => {
                eprintln!(
                    "skipping the componentize-go modules: nothing to run build-big.sh with: {e}"
                );
                false
            }
        }
    })
}

/// Runs `ffrwd-wasm` on a componentize-go module, the way `run_go_filter`
/// runs a TinyGo one.
fn run_go_big_filter(name: &str, module_args: &[&str], stdin_bytes: &[u8]) -> FfrwdWasmRun {
    let module_str = go_big_module_path(name);
    let module_str = module_str.to_str().expect("module path is valid UTF-8");
    let mut args: Vec<&str> = vec!["-f", "nut", "-i", "-", "-m", module_str];
    args.extend_from_slice(module_args);
    args.extend_from_slice(&["-f", "nut", "-"]);
    run_ffrwd_wasm(&args, stdin_bytes)
}

#[test]
fn a_componentize_go_module_inverts_the_frames_the_rust_one_inverts() {
    if !go_big_modules_built() {
        return;
    }
    let frames: Vec<Vec<u8>> = (0..3u8).map(synthetic_frame).collect();
    let wire = nut_stream(&frames);

    let go = run_go_big_filter("invert_go", &[], &wire);
    assert_run_ok(&go, "invert-go/componentize-go");

    let got = go.frames();
    assert_eq!(got.len(), 3, "expected exactly 3 frames back");
    for (i, input) in frames.iter().enumerate() {
        let (pts, output) = &got[i];
        assert_eq!(*pts, i as i64 * PTS_STEP, "frame {i} timestamp");
        for pixel in 0..(WIDTH * HEIGHT) as usize {
            let p = pixel * 4;
            assert_eq!(output[p], 255 - input[p], "frame {i} pixel {pixel} red");
            assert_eq!(
                output[p + 3],
                input[p + 3],
                "frame {i} pixel {pixel} alpha must be unchanged"
            );
        }
    }

    let rust = run_filter(&module_path("invert"), &[], &wire);
    assert_run_ok(&rust, "invert");
    assert_eq!(
        go.stdout, rust.stdout,
        "the componentize-go module's output must be byte-identical to the Rust one's"
    );
}

#[test]
fn describe_reads_a_componentize_go_module_like_any_other() {
    if !go_big_modules_built() {
        return;
    }
    let parsed = describe(&go_big_module_path("invert_go"));
    assert_eq!(parsed.world, "ffrwd:av@0.11.0");
    assert_eq!(parsed.name, Some("invert-go".to_string()));
    assert_eq!(parsed.pixel_formats, Some(vec!["rgba".to_string()]));
    assert_eq!(parsed.window, Some(1));
    assert_eq!(parsed.stride, Some(1));
    assert_eq!(parsed.inputs, 1);
    assert_eq!(parsed.rows_schema, None);
}

#[test]
fn a_componentize_go_modules_window_survives_the_same_road() {
    if !go_big_modules_built() {
        return;
    }
    // window3-go's overlapping window - the shape no module in the Rust
    // fleet declares - re-verified on the componentize-go road: window 3,
    // stride 1, so N frames make N-2 full calls plus a final call holding
    // the two the strides left.
    for count in [6usize, 2] {
        let frames: Vec<Vec<u8>> = (0..count).map(|i| synthetic_frame(i as u8)).collect();
        let run = run_go_big_filter("window3_go", &["-annotations", "out"], &nut_stream(&frames));
        assert_run_ok(
            &run,
            &format!("window3-go/componentize-go over {count} frames"),
        );

        let got = annotated_frames(&run.stdout);
        assert_eq!(got.len(), count, "every frame passes through, once");
        for (i, (pts, data, _)) in got.iter().enumerate() {
            assert_eq!(*pts, i as i64 * PTS_STEP, "frame {i} timestamp");
            assert_eq!(data, &frames[i], "frame {i} passes through untouched");
        }

        let mut expected: Vec<(usize, i64, i64)> = (0..count.saturating_sub(2))
            .map(|i| (3, i as i64 * PTS_STEP, (i + 2) as i64 * PTS_STEP))
            .collect();
        let tail = count.min(2);
        if tail > 0 {
            let first = (count - tail) as i64 * PTS_STEP;
            expected.push((tail, first, (count - 1) as i64 * PTS_STEP));
        }

        let calls: Vec<(usize, i64, i64)> = got
            .iter()
            .flat_map(|(_, _, rows)| rows.iter())
            .map(|row| {
                let parsed: WindowRow = serde_json::from_str(row)
                    .unwrap_or_else(|e| panic!("parsing window3-go row {row:?}: {e}"));
                (parsed.saw, parsed.first, parsed.last)
            })
            .collect();
        assert_eq!(
            calls, expected,
            "{count} frames through a window of three striding one, on the componentize-go road"
        );
    }
}

#[test]
fn a_componentize_go_module_matches_the_rust_module_over_many_calls_at_a_real_frame_size() {
    if !go_big_modules_built() {
        return;
    }
    // Not 4 frames, hundreds: a small call count is exactly what let the
    // TinyGo road's fleet-harness trap hide its collector's corruption, so
    // this is the case that trap is for. Streamed rather than built as one
    // in-memory wire: at 320x240 for 300 calls the output is tens of
    // megabytes, past what the buffered helpers' write-then-read can move
    // through a pipe without both sides blocking on each other.
    const W: u32 = 320;
    const H: u32 = 240;
    const CALLS: usize = 300;
    let (seen, ok, stderr) = stream_invert_endurance(&go_big_module_path("invert_go"), W, H, CALLS);
    assert!(
        ok,
        "expected all {CALLS} calls at {W}x{H} to verify correct, but only {seen} did\nstderr:\n{stderr}"
    );
}

/// Streams `total` synthetic `w`x`h` RGBA frames through `module`'s invert
/// without holding the whole run in memory, verifying each frame's inverted
/// bytes and that pts strictly increases as they arrive - this is what a
/// TinyGo build's `-gc=leaking` cannot survive past roughly a thousand
/// 640x480 frames, so this is built to find where a run stops rather than to
/// assume it. Returns how many frames verified correct before the run ended,
/// whether the process then exited zero having produced exactly `total`, and
/// its stderr.
fn stream_invert_endurance(
    module: &std::path::Path,
    w: u32,
    h: u32,
    total: usize,
) -> (usize, bool, String) {
    let exe = env!("CARGO_BIN_EXE_ffrwd-wasm");
    let module_str = module
        .to_str()
        .expect("module path is valid UTF-8")
        .to_string();
    let mut child = Command::new(exe)
        .args(["-f", "nut", "-i", "-", "-m", &module_str, "-f", "nut", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn ffrwd-wasm");

    let stdin = child.stdin.take().expect("child stdin");
    let mut stderr_pipe = child.stderr.take().expect("child stderr");
    let stderr_reader = std::thread::spawn(move || {
        use std::io::Read;
        let mut s = String::new();
        let _ = stderr_pipe.read_to_string(&mut s);
        s
    });

    let frame_len = (w * h * 4) as usize;
    let stream = Stream::video("rgba", w, h, TIME_BASE).expect("rgba is carried");
    let writer = std::thread::spawn(move || -> bool {
        let mut muxer = match Muxer::new(stdin, &stream) {
            Ok(m) => m,
            Err(_) => return false,
        };
        let mut frame = vec![0u8; frame_len];
        for i in 0..total {
            let base = (i as u8).wrapping_mul(41);
            for (offset, byte) in frame.iter_mut().enumerate() {
                *byte = base.wrapping_add(offset as u8);
            }
            if muxer.write_frame(i as i64 * PTS_STEP, &frame).is_err() {
                return false;
            }
        }
        muxer.finish().is_ok()
    });

    let stdout = child.stdout.take().expect("child stdout");
    let mut seen = 0usize;
    if let Ok(mut demuxer) = Demuxer::open(stdout) {
        {
            let mut buf = Vec::new();
            let mut last_pts = i64::MIN;
            'read: while seen < total {
                match demuxer.read_frame(&mut buf) {
                    Ok(Some(pts)) => {
                        if seen > 0 && pts <= last_pts {
                            break 'read;
                        }
                        last_pts = pts;
                        let base = (seen as u8).wrapping_mul(41);
                        if buf.len() != frame_len {
                            break 'read;
                        }
                        for (offset, &byte) in buf.iter().enumerate() {
                            let input = base.wrapping_add(offset as u8);
                            let want = if offset % 4 == 3 { input } else { 255 - input };
                            if byte != want {
                                break 'read;
                            }
                        }
                        seen += 1;
                    }
                    Ok(None) | Err(_) => break 'read,
                }
            }
        }
    }

    let _ = writer.join();
    let stderr = stderr_reader.join().unwrap_or_default();
    let status = child.wait().expect("wait for ffrwd-wasm");
    (seen, status.success() && seen == total, stderr)
}

#[test]
fn a_componentize_go_module_survives_a_thousand_frames_where_tinygo_would_not() {
    if !go_big_modules_built() {
        return;
    }
    // 1000+ frames of 640x480 through invert is what caps a TinyGo build at
    // `-gc=leaking`: nothing it allocates is ever freed, so the instance runs
    // out of memory around there. Mainline Go's collector is what is on
    // trial here, so this goes well past that cap.
    const TOTAL: usize = 1200;
    let (seen, ok, stderr) =
        stream_invert_endurance(&go_big_module_path("invert_go"), 640, 480, TOTAL);
    assert!(
        ok,
        "expected all {TOTAL} frames of 640x480 to verify correct on the componentize-go road, \
         but only {seen} did\nstderr:\n{stderr}"
    );
}

#[test]
fn a_componentize_go_module_holds_up_well_past_where_tinygo_dies() {
    if !go_big_modules_built() {
        return;
    }
    // The endurance case the committed TinyGo report measured: a leaking
    // collector caps that road at roughly a thousand 640x480 frames. This
    // goes to 3000 - three times that cap - checking the same things: every
    // frame's bytes, and that pts never stops increasing.
    const TOTAL: usize = 3000;
    let (seen, ok, stderr) =
        stream_invert_endurance(&go_big_module_path("invert_go"), 640, 480, TOTAL);
    assert!(
        ok,
        "expected all {TOTAL} frames of 640x480 to verify correct on the componentize-go road, \
         but only {seen} did\nstderr:\n{stderr}"
    );
}

/// Not a `#[test]`: this finds, rather than assumes, the frame count a
/// TinyGo `-gc=leaking` build actually stops at on this machine. Run
/// explicitly with `cargo test -- --ignored --nocapture
/// tinygo_road_endurance_ceiling` to reproduce the number the report above
/// cites; skipped otherwise so an ordinary run of this file does not pay for
/// a search that ends in a crash by design.
#[test]
#[ignore]
fn tinygo_road_endurance_ceiling() {
    if !go_modules_built() {
        eprintln!("skipping: the TinyGo modules are not built");
        return;
    }
    const TOTAL: usize = 4000;
    let (seen, ok, stderr) = stream_invert_endurance(&go_module_path("invert_go"), 640, 480, TOTAL);
    eprintln!(
        "TinyGo road (-gc=leaking): {seen}/{TOTAL} frames of 640x480 verified correct, \
         ok={ok}\nstderr tail:\n{}",
        stderr
            .chars()
            .rev()
            .take(2000)
            .collect::<String>()
            .chars()
            .rev()
            .collect::<String>()
    );
}

/// Not a `#[test]` in the ordinary run, for the same reason as
/// `tinygo_road_endurance_ceiling`: this finds where the componentize-go
/// road's own workaround for the reentrancy trap - disabling the collector
/// entirely, in `invert-go`'s `init()` - runs into its own ceiling. Nothing
/// is ever freed once the collector is off, so this is a search for where
/// wasm32's linear memory actually runs out, not a guess. Run explicitly with
/// `cargo test -- --ignored --nocapture componentize_go_road_endurance_ceiling`.
#[test]
#[ignore]
fn componentize_go_road_endurance_ceiling() {
    if !go_big_modules_built() {
        eprintln!("skipping: the componentize-go modules are not built");
        return;
    }
    const TOTAL: usize = 20_000;
    let (seen, ok, stderr) =
        stream_invert_endurance(&go_big_module_path("invert_go"), 640, 480, TOTAL);
    eprintln!(
        "componentize-go road (collector off): {seen}/{TOTAL} frames of 640x480 verified correct, \
         ok={ok}\nstderr tail:\n{}",
        stderr
            .chars()
            .rev()
            .take(2000)
            .collect::<String>()
            .chars()
            .rev()
            .collect::<String>()
    );
}

/// Wall-clock frames/second for one `invert`-shaped module over `count`
/// frames of `w`x`h` RGBA, wire construction excluded: only the
/// spawn-through-exit of `ffrwd-wasm` itself is timed, since that is what a
/// build road can move.
fn invert_fps(module: &std::path::Path, w: u32, h: u32, count: usize) -> f64 {
    let start = std::time::Instant::now();
    // Streamed for the same reason the endurance helper is: at 640x480 for
    // 200 frames the output is tens of megabytes, past what a
    // write-everything-then-read helper can move through a pipe without
    // both sides blocking on each other.
    let (seen, ok, stderr) = stream_invert_endurance(module, w, h, count);
    let elapsed = start.elapsed();
    assert!(
        ok,
        "expected all {count} frames of {w}x{h} to verify correct on {}, but only {seen} did\nstderr:\n{stderr}",
        module.display()
    );
    count as f64 / elapsed.as_secs_f64()
}

/// Not a `#[test]` in the ordinary run: prints frames/second for the Rust
/// invert and both Go roads over the same frames, in one process invocation
/// each, back to back in this one session - interleaved rather than
/// separate-session numbers, which is what a fair comparison needs. Run
/// explicitly with `cargo test -- --ignored --nocapture fps_measurements`.
#[test]
#[ignore]
fn fps_measurements() {
    const W: u32 = 640;
    const H: u32 = 480;
    const COUNT: usize = 200;

    let rust_fps = invert_fps(&module_path("invert"), W, H, COUNT);
    eprintln!("rust invert:                 {rust_fps:.1} fps ({COUNT} frames of {W}x{H})");

    if go_modules_built() {
        let tinygo_fps = invert_fps(&go_module_path("invert_go"), W, H, COUNT);
        eprintln!("invert-go / TinyGo:          {tinygo_fps:.1} fps");
    } else {
        eprintln!("invert-go / TinyGo:          skipped, not built");
    }

    if go_big_modules_built() {
        let big_fps = invert_fps(&go_big_module_path("invert_go"), W, H, COUNT);
        eprintln!("invert-go / componentize-go: {big_fps:.1} fps");
    } else {
        eprintln!("invert-go / componentize-go: skipped, not built");
    }
}
