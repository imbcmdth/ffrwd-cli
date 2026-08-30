//! Real-ffmpeg coverage through the pipelines that ship:
//! `ffmpeg (lavfi source) | ffrwd-wasm -m <module> | ffmpeg (consumer)`, the
//! same with two sidecars in the middle sharing one annotated wire, and the
//! same with one sidecar hosting a whole network of modules. NUT on every
//! seam. ffmpeg must be on PATH; a missing ffmpeg fails the test rather than
//! skipping it.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStderr, Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use std::thread;

use ffrwd_wasm::nut::Demuxer;
use serde::Deserialize;

const WIDTH: u32 = 8;
const HEIGHT: u32 = 8;
const FRAME_LEN: usize = (WIDTH * HEIGHT * 4) as usize;

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

fn read_to_string_lossy(mut r: impl Read) -> String {
    let mut buf = Vec::new();
    r.read_to_end(&mut buf).expect("read stderr pipe");
    String::from_utf8_lossy(&buf).into_owned()
}

/// Spawns a thread that drains a child's stderr into a `String`, so a
/// filling pipe can never stall the stage feeding it.
fn drain_stderr(stderr: ChildStderr) -> thread::JoinHandle<String> {
    thread::spawn(move || read_to_string_lossy(stderr))
}

struct StageResult {
    status: ExitStatus,
    stderr: String,
}

fn assert_stage_ok(result: &StageResult, name: &str) {
    assert!(
        result.status.success(),
        "{name} exited with {:?}\nstderr:\n{}",
        result.status.code(),
        result.stderr
    );
}

/// A path under the OS temp dir, unique per call, removed when dropped.
struct TempFile(PathBuf);

impl TempFile {
    fn new(label: &str) -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "ffrwd-wasm-test-{}-{}-{label}",
            std::process::id(),
            n
        ));
        TempFile(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// Runs a plain `ffmpeg -f lavfi -i <source> -f rawvideo -pix_fmt rgba -`,
/// with no ffrwd-wasm in the loop, to get known-good reference pixels.
fn run_reference(source: &str, frames: u32) -> Vec<u8> {
    let output = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            source,
            "-frames:v",
            &frames.to_string(),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgba",
            "-",
        ])
        .stdin(Stdio::null())
        .output()
        .expect("spawn reference ffmpeg");
    assert!(
        output.status.success(),
        "reference ffmpeg exited with {:?}\nstderr:\n{}",
        output.status.code(),
        String::from_utf8_lossy(&output.stderr)
    );
    output.stdout
}

/// The argv of an ffmpeg that turns a lavfi source into NUT-wrapped
/// uncompressed frames on `target` - the shape every producer here uses.
fn producer_args(source: &str, frames: u32, pix_fmt: &str, target: &str) -> Vec<String> {
    [
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        source,
        "-frames:v",
        &frames.to_string(),
        "-c:v",
        "rawvideo",
        "-pix_fmt",
        pix_fmt,
        "-f",
        "nut",
        target,
    ]
    .iter()
    .map(|s| s.to_string())
    .collect()
}

/// Spawns the NUT producer with its stdout on a pipe.
fn spawn_producer(source: &str, frames: u32, pix_fmt: &str) -> Child {
    Command::new("ffmpeg")
        .args(producer_args(source, frames, pix_fmt, "-"))
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn producer ffmpeg")
}

/// Writes a NUT file with the producer's shape, for tests that need to feed
/// ffrwd-wasm a file and probe it afterwards.
fn write_nut(source: &str, frames: u32, pix_fmt: &str, target: &Path) {
    let output = Command::new("ffmpeg")
        .args(producer_args(
            source,
            frames,
            pix_fmt,
            target.to_str().expect("path is valid UTF-8"),
        ))
        .stdin(Stdio::null())
        .output()
        .expect("spawn producer ffmpeg");
    assert!(
        output.status.success(),
        "producer ffmpeg exited with {:?}\nstderr:\n{}",
        output.status.code(),
        String::from_utf8_lossy(&output.stderr)
    );
}

/// Every packet timestamp in a file, in seconds, as ffprobe reads them.
fn packet_times(path: &Path) -> Vec<f64> {
    let output = Command::new("ffprobe")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "packet=pts_time",
            "-of",
            "csv=p=0",
        ])
        .arg(path)
        .stdin(Stdio::null())
        .output()
        .expect("spawn ffprobe");
    assert!(
        output.status.success(),
        "ffprobe exited with {:?}\nstderr:\n{}",
        output.status.code(),
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(|line| {
            line.trim_end_matches(',')
                .parse()
                .unwrap_or_else(|e| panic!("parsing ffprobe timestamp {line:?}: {e}"))
        })
        .collect()
}

/// Last timestamp minus first, which is what a stream's timestamps say its
/// span is however they were made.
fn span(times: &[f64]) -> f64 {
    match (times.first(), times.last()) {
        (Some(first), Some(last)) => last - first,
        _ => 0.0,
    }
}

/// The argv ffrwd-wasm gets: NUT in, a module, then whatever outputs the
/// caller asks for.
fn ffrwd_wasm_args(module: &Path, module_args: &[&str]) -> Vec<String> {
    let mut args: Vec<String> = ["-f", "nut", "-i", "-", "-m"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    args.push(
        module
            .to_str()
            .expect("module path is valid UTF-8")
            .to_string(),
    );
    args.extend(module_args.iter().map(|s| s.to_string()));
    args
}

struct PipelineRun {
    producer: StageResult,
    ffrwd_wasm: StageResult,
    consumer: StageResult,
    stdout: Vec<u8>,
}

/// Wires `ffmpeg (lavfi producer) | ffrwd-wasm -m <module> | ffmpeg (consumer)`
/// with real OS pipes: each stage's stdout feeds the next stage's stdin
/// directly, and the two non-final stages have their stderr drained
/// concurrently so a full pipe can't stall the chain. The consumer unwraps
/// the NUT back to raw pixels, which is what the caller asserts on.
fn run_pipeline(source: &str, frames: u32, module: &Path, module_args: &[&str]) -> PipelineRun {
    let mut args = ffrwd_wasm_args(module, module_args);
    args.extend(["-f".to_string(), "nut".to_string(), "-".to_string()]);
    run_pipeline_args(source, frames, args)
}

/// `run_pipeline` for a sidecar spelled out in full - a module network, say -
/// where the caller writes the whole argv itself.
fn run_pipeline_args(source: &str, frames: u32, args: Vec<String>) -> PipelineRun {
    ensure_modules_built();

    let mut producer = spawn_producer(source, frames, "rgba");
    let producer_stdout = producer.stdout.take().expect("producer stdout");
    let producer_stderr_handle = drain_stderr(producer.stderr.take().expect("producer stderr"));

    let mut ffrwd_wasm: Child = Command::new(env!("CARGO_BIN_EXE_ffrwd-wasm"))
        .args(&args)
        .stdin(Stdio::from(producer_stdout))
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn ffrwd-wasm");
    let ffrwd_wasm_stdout = ffrwd_wasm.stdout.take().expect("ffrwd-wasm stdout");
    let ffrwd_wasm_stderr_handle =
        drain_stderr(ffrwd_wasm.stderr.take().expect("ffrwd-wasm stderr"));

    let consumer_output = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "nut",
            "-i",
            "-",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgba",
            "-",
        ])
        .stdin(Stdio::from(ffrwd_wasm_stdout))
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .expect("spawn consumer ffmpeg");

    let producer_status = producer.wait().expect("wait for producer ffmpeg");
    let ffrwd_wasm_status = ffrwd_wasm.wait().expect("wait for ffrwd-wasm");
    let producer_stderr = producer_stderr_handle
        .join()
        .expect("producer stderr thread panicked");
    let ffrwd_wasm_stderr = ffrwd_wasm_stderr_handle
        .join()
        .expect("ffrwd-wasm stderr thread panicked");

    PipelineRun {
        producer: StageResult {
            status: producer_status,
            stderr: producer_stderr,
        },
        ffrwd_wasm: StageResult {
            status: ffrwd_wasm_status,
            stderr: ffrwd_wasm_stderr,
        },
        consumer: StageResult {
            status: consumer_output.status,
            stderr: String::from_utf8_lossy(&consumer_output.stderr).into_owned(),
        },
        stdout: consumer_output.stdout,
    }
}

fn assert_pipeline_ok(run: &PipelineRun) {
    assert_stage_ok(&run.producer, "producer ffmpeg");
    assert_stage_ok(&run.ffrwd_wasm, "ffrwd-wasm");
    assert_stage_ok(&run.consumer, "consumer ffmpeg");
}

fn frame_at(bytes: &[u8], index: usize) -> &[u8] {
    &bytes[index * FRAME_LEN..(index + 1) * FRAME_LEN]
}

/// Red channel of row 0, one byte per pixel.
fn red_row0(frame: &[u8]) -> Vec<u8> {
    (0..WIDTH as usize).map(|x| frame[x * 4]).collect()
}

#[test]
fn invert_inverts_rgb_and_keeps_alpha() {
    let source = "color=c=red:s=8x8:r=1:d=1";
    let reference = run_reference(source, 1);
    assert_eq!(reference.len(), FRAME_LEN);

    let run = run_pipeline(source, 1, &module_path("invert"), &[]);
    assert_pipeline_ok(&run);
    assert_eq!(
        run.stdout.len(),
        FRAME_LEN,
        "expected exactly one frame back"
    );

    for i in (0..FRAME_LEN).step_by(4) {
        assert_eq!(
            run.stdout[i],
            255 - reference[i],
            "red byte at pixel {}",
            i / 4
        );
        assert_eq!(
            run.stdout[i + 1],
            255 - reference[i + 1],
            "green byte at pixel {}",
            i / 4
        );
        assert_eq!(
            run.stdout[i + 2],
            255 - reference[i + 2],
            "blue byte at pixel {}",
            i / 4
        );
        assert_eq!(
            run.stdout[i + 3],
            reference[i + 3],
            "alpha byte at pixel {} must be unchanged",
            i / 4
        );
    }
}

#[test]
fn every_frame_makes_it_through() {
    // 64x64 frames put more than the syncpoint distance on the wire, so the
    // output carries a dozen syncpoints rather than the single one a tiny
    // stream needs, and real ffmpeg reads across all of them.
    let source = "testsrc2=s=64x64:r=25:d=1";
    let frames = 25;
    let frame_len = 64 * 64 * 4;

    let run = run_pipeline(source, frames, &module_path("invert"), &[]);
    assert_pipeline_ok(&run);
    assert_eq!(
        run.stdout.len(),
        frames as usize * frame_len,
        "expected exactly {frames} frames back"
    );

    let reference = run_reference(source, frames);
    for i in 0..frames as usize {
        let got = &run.stdout[i * frame_len..(i + 1) * frame_len];
        let want = &reference[i * frame_len..(i + 1) * frame_len];
        for p in (0..frame_len).step_by(4) {
            assert_eq!(
                got[p],
                255 - want[p],
                "frame {i} byte {p} must derive from source frame {i}"
            );
        }
    }
}

#[test]
fn timestamps_ride_through_unchanged() {
    ensure_modules_built();
    let frames = 10;
    let plain_source = "testsrc2=s=8x8:r=25:d=1";
    let stretched_source = "testsrc2=s=8x8:r=25:d=1,setpts=2.0*PTS";

    let plain = TempFile::new("plain.nut");
    write_nut(plain_source, frames, "rgba", plain.path());
    let upstream = TempFile::new("stretched.nut");
    write_nut(stretched_source, frames, "rgba", upstream.path());

    let plain_times = packet_times(plain.path());
    let upstream_times = packet_times(upstream.path());
    assert_eq!(upstream_times.len(), frames as usize);
    assert!(
        (span(&upstream_times) - 2.0 * span(&plain_times)).abs() < 1e-6,
        "the upstream filter should have doubled the span: {} against {}",
        span(&upstream_times),
        span(&plain_times)
    );

    let filtered = TempFile::new("filtered.nut");
    let mut args = ffrwd_wasm_args(&module_path("invert"), &[]);
    args.push("-f".to_string());
    args.push("nut".to_string());
    args.push(
        filtered
            .path()
            .to_str()
            .expect("path is valid UTF-8")
            .to_string(),
    );
    // The input is a file here rather than stdin, so nothing but the wire is
    // between the two ffprobe readings.
    let input_index = args.iter().position(|a| a == "-i").expect("-i is there") + 1;
    args[input_index] = upstream
        .path()
        .to_str()
        .expect("path is valid UTF-8")
        .to_string();

    let output = Command::new(env!("CARGO_BIN_EXE_ffrwd-wasm"))
        .args(&args)
        .stdin(Stdio::null())
        .output()
        .expect("spawn ffrwd-wasm");
    assert!(
        output.status.success(),
        "ffrwd-wasm exited with {:?}\nstderr:\n{}",
        output.status.code(),
        String::from_utf8_lossy(&output.stderr)
    );

    let filtered_times = packet_times(filtered.path());
    assert_eq!(
        filtered_times, upstream_times,
        "every frame must come out at the timestamp it came in with"
    );
    assert!(
        (span(&filtered_times) - span(&upstream_times)).abs() < 1e-9,
        "the filtered stream's span must be the upstream's"
    );
}

#[test]
#[ignore = "needs the box_blur module, not vendored here"]
fn box_blur_softens_a_hard_edge() {
    let source = "color=c=black:s=8x8:r=1:d=1,drawbox=x=4:y=0:w=4:h=8:color=white:t=fill";

    let reference = run_reference(source, 1);
    let ref_row0 = red_row0(&reference);
    assert!(
        ref_row0.iter().all(|&b| b == 0 || b == 255),
        "reference row 0 should hold only the two extreme values, got {ref_row0:?}"
    );

    let run = run_pipeline(
        source,
        1,
        &module_path("box_blur"),
        &["-params", r#"{"radius":1}"#],
    );
    assert_pipeline_ok(&run);
    assert_eq!(run.stdout.len(), FRAME_LEN);

    let blurred_row0 = red_row0(frame_at(&run.stdout, 0));
    assert_eq!(
        blurred_row0,
        vec![0, 0, 0, 85, 170, 255, 255, 255],
        "measured first row of the red channel"
    );
}

// Row and refusal coverage below. These tests read ffrwd-wasm's file outputs
// directly instead of piping to a consumer ffmpeg, so they get their own
// runner rather than reusing `run_pipeline`.

struct AnalysisRun {
    producer: StageResult,
    ffrwd_wasm: StageResult,
}

/// Runs `ffmpeg (lavfi producer) | ffrwd-wasm -m <module>` for tests whose
/// results come from ffrwd-wasm's output files rather than its stdout.
/// ffrwd-wasm's stdout is always unused here since every output goes to a
/// file.
fn run_analysis_pipeline(
    source: &str,
    frames: u32,
    module: &Path,
    module_args: &[&str],
    jobs: Option<u32>,
    frame_out: Option<&Path>,
    rows_out: Option<&Path>,
) -> AnalysisRun {
    ensure_modules_built();

    let mut producer = spawn_producer(source, frames, "rgba");
    let producer_stdout = producer.stdout.take().expect("producer stdout");
    let producer_stderr_handle = drain_stderr(producer.stderr.take().expect("producer stderr"));

    let mut args = ffrwd_wasm_args(module, module_args);
    if let Some(j) = jobs {
        args.push("-jobs".into());
        args.push(j.to_string());
    }
    if let Some(p) = frame_out {
        args.push("-f".into());
        args.push("nut".into());
        args.push(p.to_str().expect("path is valid UTF-8").into());
    }
    if let Some(p) = rows_out {
        args.push("-f".into());
        args.push("ndjson".into());
        args.push(p.to_str().expect("path is valid UTF-8").into());
    }

    let mut ffrwd_wasm: Child = Command::new(env!("CARGO_BIN_EXE_ffrwd-wasm"))
        .args(&args)
        .stdin(Stdio::from(producer_stdout))
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn ffrwd-wasm");
    let ffrwd_wasm_stderr_handle =
        drain_stderr(ffrwd_wasm.stderr.take().expect("ffrwd-wasm stderr"));

    let producer_status = producer.wait().expect("wait for producer ffmpeg");
    let ffrwd_wasm_status = ffrwd_wasm.wait().expect("wait for ffrwd-wasm");
    let producer_stderr = producer_stderr_handle
        .join()
        .expect("producer stderr thread panicked");
    let ffrwd_wasm_stderr = ffrwd_wasm_stderr_handle
        .join()
        .expect("ffrwd-wasm stderr thread panicked");

    AnalysisRun {
        producer: StageResult {
            status: producer_status,
            stderr: producer_stderr,
        },
        ffrwd_wasm: StageResult {
            status: ffrwd_wasm_status,
            stderr: ffrwd_wasm_stderr,
        },
    }
}

/// `pts` is not framestats' own key - the host stamps it on from the frame
/// the row rode - which is the collision case's other side from rms' row (in
/// the wasm unit tests): a row that already names `time` still gets `pts`
/// added.
#[derive(Deserialize)]
struct FramestatsRow {
    pts: i64,
    time: f64,
    mean: u64,
}

/// The row framestats ends with, which no frame carries.
#[derive(Deserialize)]
struct FramestatsSummary {
    frames: u64,
}

/// One row per call, as tail3 reports its windows.
#[derive(Deserialize)]
struct TailRow {
    frames: usize,
    last: bool,
}

/// framestats' ndjson output: a row per frame, then the summary the stream
/// ends with, which has no frame to ride and so comes last.
fn parse_framestats_output(path: &Path) -> (Vec<FramestatsRow>, u64) {
    let text = std::fs::read_to_string(path).expect("read ndjson output file");
    let mut lines: Vec<&str> = text.lines().filter(|line| !line.is_empty()).collect();
    let summary_line = lines.pop().expect("framestats ends with its summary row");
    let summary: FramestatsSummary = serde_json::from_str(summary_line)
        .unwrap_or_else(|e| panic!("parsing framestats summary row {summary_line:?}: {e}"));
    let rows = lines
        .iter()
        .map(|line| {
            serde_json::from_str(line)
                .unwrap_or_else(|e| panic!("parsing ndjson row {line:?}: {e}"))
        })
        .collect();
    (rows, summary.frames)
}

/// Every window tail3 reported, as `(frames, last)`.
fn parse_tail_rows(path: &Path) -> Vec<(usize, bool)> {
    let text = std::fs::read_to_string(path).expect("read ndjson output file");
    text.lines()
        .filter(|line| !line.is_empty())
        .map(|line| {
            let row: TailRow = serde_json::from_str(line)
                .unwrap_or_else(|e| panic!("parsing tail3 row {line:?}: {e}"));
            (row.frames, row.last)
        })
        .collect()
}

/// Asserts row times start at 0.0 and strictly increase, frame to frame.
fn assert_times_strictly_increasing(rows: &[FramestatsRow]) {
    let mut prev: Option<f64> = None;
    for (i, row) in rows.iter().enumerate() {
        if let Some(p) = prev {
            assert!(
                row.time > p,
                "row {i} time {} does not exceed previous row's time {p}",
                row.time
            );
        }
        prev = Some(row.time);
    }
}

/// Asserts the host-stamped `pts` starts at 0 and strictly increases, frame
/// to frame - the same shape `time` takes, since both come from the frame
/// the row rode.
fn assert_pts_strictly_increasing(rows: &[FramestatsRow]) {
    assert_eq!(
        rows.first().map(|r| r.pts),
        Some(0),
        "a fresh stream's first frame carries pts 0"
    );
    let mut prev: Option<i64> = None;
    for (i, row) in rows.iter().enumerate() {
        if let Some(p) = prev {
            assert!(
                row.pts > p,
                "row {i} pts {} does not exceed previous row's pts {p}",
                row.pts
            );
        }
        prev = Some(row.pts);
    }
}

#[test]
fn framestats_emits_a_row_per_frame() {
    let source = "color=c=red:s=8x8:r=25:d=0.2";
    let rows_file = TempFile::new("framestats-rows");

    let run = run_analysis_pipeline(
        source,
        5,
        &module_path("framestats"),
        &[],
        None,
        None,
        Some(rows_file.path()),
    );
    assert_stage_ok(&run.producer, "producer ffmpeg");
    assert_stage_ok(&run.ffrwd_wasm, "ffrwd-wasm");

    let (rows, counted) = parse_framestats_output(rows_file.path());
    assert_eq!(rows.len(), 5, "expected exactly one row per frame");
    assert_eq!(counted, 5, "the summary counts every frame that went by");

    for (i, row) in rows.iter().enumerate() {
        assert_eq!(row.mean, 84, "row {i} mean");
    }

    let step = 0.04;
    for (i, row) in rows.iter().enumerate() {
        let expected = i as f64 * step;
        assert!(
            (row.time - expected).abs() < 1e-9,
            "row {i} time: expected {expected}, got {}",
            row.time
        );
    }
    assert_times_strictly_increasing(&rows);
    assert_pts_strictly_increasing(&rows);
}

#[test]
fn dual_output_passes_frames_and_writes_rows() {
    let source = "color=c=red:s=8x8:r=25:d=0.2";
    let reference = run_reference(source, 5);
    assert_eq!(reference.len(), 5 * FRAME_LEN);

    let frames_file = TempFile::new("dual-frames.nut");
    let rows_file = TempFile::new("dual-rows");

    let run = run_analysis_pipeline(
        source,
        5,
        &module_path("framestats"),
        &[],
        None,
        Some(frames_file.path()),
        Some(rows_file.path()),
    );
    assert_stage_ok(&run.producer, "producer ffmpeg");
    assert_stage_ok(&run.ffrwd_wasm, "ffrwd-wasm");

    let unwrapped = Command::new("ffmpeg")
        .args(["-hide_banner", "-loglevel", "error", "-f", "nut", "-i"])
        .arg(frames_file.path())
        .args(["-f", "rawvideo", "-pix_fmt", "rgba", "-"])
        .stdin(Stdio::null())
        .output()
        .expect("spawn ffmpeg to unwrap the frame output");
    assert!(unwrapped.status.success(), "unwrapping the frame output");
    assert_eq!(
        unwrapped.stdout, reference,
        "framestats is passthrough: frames must be unchanged"
    );

    let (rows, counted) = parse_framestats_output(rows_file.path());
    assert_eq!(rows.len(), 5, "expected exactly one row per frame");
    assert_eq!(counted, 5, "the summary counts every frame that went by");
    for (i, row) in rows.iter().enumerate() {
        assert_eq!(row.mean, 84, "row {i} mean");
    }
    assert_times_strictly_increasing(&rows);
    assert_pts_strictly_increasing(&rows);

    // The frame output is the same pts sequence the ndjson rows were stamped
    // from, read independently off the wire rather than trusted by assertion.
    let mut demuxer =
        Demuxer::open(std::fs::File::open(frames_file.path()).expect("open the frame output"))
            .expect("read the NUT headers");
    let time_base = demuxer.stream().time_base;
    let mut buf = Vec::new();
    let mut frame_pts = Vec::new();
    while let Some(pts) = demuxer.read_frame(&mut buf).expect("read a NUT frame") {
        frame_pts.push(pts);
    }
    assert_eq!(frame_pts.len(), rows.len(), "one frame per row");
    for (i, (row, pts)) in rows.iter().zip(&frame_pts).enumerate() {
        assert_eq!(row.pts, *pts, "row {i}'s stamped pts is the frame it rode");
        assert!(
            (row.time - time_base.seconds(*pts)).abs() < 1e-9,
            "row {i}'s time agrees with pts through the stream's time base"
        );
    }
}

#[test]
fn rows_keep_frame_order_under_jobs() {
    // tail3, because only a pure module may be spread across workers. Its row
    // per call says how many frames that call carried, so the file read in
    // order is the window order.
    let source = "testsrc2=s=64x64:r=25:d=1";
    let frames = 25;

    let rows_serial = TempFile::new("jobs1-rows");
    let run_serial = run_analysis_pipeline(
        source,
        frames,
        &module_path("tail3"),
        &[],
        Some(1),
        None,
        Some(rows_serial.path()),
    );
    assert_stage_ok(&run_serial.producer, "producer ffmpeg (-jobs 1)");
    assert_stage_ok(&run_serial.ffrwd_wasm, "ffrwd-wasm (-jobs 1)");

    let rows_parallel = TempFile::new("jobs4-rows");
    let run_parallel = run_analysis_pipeline(
        source,
        frames,
        &module_path("tail3"),
        &[],
        Some(4),
        None,
        Some(rows_parallel.path()),
    );
    assert_stage_ok(&run_parallel.producer, "producer ffmpeg (-jobs 4)");
    assert_stage_ok(&run_parallel.ffrwd_wasm, "ffrwd-wasm (-jobs 4)");

    let bytes_serial = std::fs::read(rows_serial.path()).expect("read -jobs 1 ndjson output");
    let bytes_parallel = std::fs::read(rows_parallel.path()).expect("read -jobs 4 ndjson output");
    assert_eq!(
        bytes_serial, bytes_parallel,
        "ndjson output must be byte-identical regardless of -jobs"
    );

    // 25 frames through a window of three: eight full calls, then the one
    // frame left over, which is the call marked last.
    let mut expected = vec![(3usize, false); 8];
    expected.push((1, true));
    assert_eq!(parse_tail_rows(rows_serial.path()), expected);
}

/// Where `needle` sits in `wire`, for pinning the order two records were
/// written in.
fn byte_offset_of(wire: &[u8], needle: &[u8]) -> usize {
    wire.windows(needle.len())
        .position(|window| window == needle)
        .unwrap_or_else(|| {
            panic!(
                "{:?} is nowhere in the output",
                String::from_utf8_lossy(needle)
            )
        })
}

#[test]
fn the_frame_count_leaves_as_one_trailing_record_after_every_frames_rows() {
    // framestats' count is not known until the last frame has gone by, so it
    // has no frame to ride: it leaves as the one trailing record, after every
    // per-frame record on the annotation stream.
    let source = "color=c=red:s=8x8:r=25:d=0.2";
    let frames_file = TempFile::new("trailing-frames.nut");

    let run = run_analysis_pipeline(
        source,
        5,
        &module_path("framestats"),
        &["-annotations", "out"],
        None,
        Some(frames_file.path()),
        None,
    );
    assert_stage_ok(&run.producer, "producer ffmpeg");
    assert_stage_ok(&run.ffrwd_wasm, "ffrwd-wasm");

    let wire = std::fs::read(frames_file.path()).expect("read the annotated frame output");
    let mut demuxer = Demuxer::open_annotated(&wire[..]).expect("read the annotated NUT headers");
    assert!(demuxer.has_annotations(), "the rows ride beside the frames");

    let mut per_frame: Vec<String> = Vec::new();
    let mut buf = Vec::new();
    while let Some(pts) = demuxer.read_frame(&mut buf).expect("read a NUT frame") {
        let rows = demuxer.take_rows(pts);
        assert_eq!(rows.len(), 1, "one row per frame, at pts {pts}");
        per_frame.extend(rows);
    }
    assert_eq!(per_frame.len(), 5, "a row for every frame");
    for row in &per_frame {
        let parsed: FramestatsRow =
            serde_json::from_str(row).unwrap_or_else(|e| panic!("parsing {row:?}: {e}"));
        assert_eq!(parsed.mean, 84);
    }

    let trailing = demuxer.take_trailing();
    assert_eq!(trailing.len(), 1, "one trailing row, and it is the count");
    let summary: FramestatsSummary = serde_json::from_str(&trailing[0])
        .unwrap_or_else(|e| panic!("parsing the trailing row {:?}: {e}", trailing[0]));
    assert_eq!(summary.frames, 5, "every frame is counted");

    let last_per_frame = byte_offset_of(&wire, per_frame[4].as_bytes());
    let record = byte_offset_of(&wire, br#"{"trailing":"#);
    assert!(
        record > last_per_frame,
        "the trailing record is written after every per-frame record, got {record} then          {last_per_frame}"
    );
}

// Shot detection over a real cut: one lavfi stream whose two halves are
// different pictures, so the only large frame-to-frame change in it is the
// join.

/// A second of testsrc followed by a second of smptebars, concatenated into
/// one stream: 50 frames with a hard cut in the middle and nothing else in
/// them that looks like one.
const CUT_SOURCE: &str =
    "testsrc=s=64x64:r=25:d=1[a];smptebars=s=64x64:r=25:d=1[b];[a][b]concat=n=2:v=1:a=0";
const CUT_FRAMES: u32 = 50;
/// The first frame of the second half.
const CUT_AT: usize = 25;

#[derive(Deserialize)]
struct ShotRow {
    shot: i64,
}

/// The shot index each row names, in frame order.
fn parse_shot_rows(path: &Path) -> Vec<i64> {
    let text = std::fs::read_to_string(path).expect("read ndjson output file");
    text.lines()
        .filter(|line| !line.is_empty())
        .map(|line| {
            let row: ShotRow = serde_json::from_str(line)
                .unwrap_or_else(|e| panic!("parsing shots row {line:?}: {e}"));
            row.shot
        })
        .collect()
}

/// Runs shots over the cut source at `params`, returning the shot index per
/// frame.
fn shots_over_the_cut(params: &[&str]) -> Vec<i64> {
    let rows_file = TempFile::new("shots-rows");
    let run = run_analysis_pipeline(
        CUT_SOURCE,
        CUT_FRAMES,
        &module_path("shots"),
        params,
        None,
        None,
        Some(rows_file.path()),
    );
    assert_stage_ok(&run.producer, "producer ffmpeg");
    assert_stage_ok(&run.ffrwd_wasm, "ffrwd-wasm hosting shots");
    parse_shot_rows(rows_file.path())
}

/// Shot 0 up to `CUT_AT`, shot 1 from there on.
fn one_cut_halfway() -> Vec<i64> {
    (0..CUT_FRAMES as usize)
        .map(|frame| i64::from(frame >= CUT_AT))
        .collect()
}

#[test]
fn shots_steps_once_at_a_real_cut() {
    assert_eq!(
        shots_over_the_cut(&[]),
        one_cut_halfway(),
        "at the default threshold the index steps at the join and nowhere else"
    );
}

#[test]
fn the_default_threshold_sits_between_the_cut_and_the_motion() {
    // The same stream either side of the default: five times higher still
    // catches the cut, a third of it still catches nothing else. The default
    // is not perched on the edge of either.
    assert_eq!(
        shots_over_the_cut(&["-params", r#"{"threshold":60}"#]),
        one_cut_halfway(),
        "the cut is far above the default"
    );
    assert_eq!(
        shots_over_the_cut(&["-params", r#"{"threshold":4}"#]),
        one_cut_halfway(),
        "everything that is not the cut is far below the default"
    );
}

// The sidecar-to-sidecar edge: two modules on one pipe, with the rows the
// first emits riding beside the frames.

/// Unwraps a NUT file to raw rgba pixels through real ffmpeg.
fn unwrap_nut(path: &Path) -> Vec<u8> {
    let output = Command::new("ffmpeg")
        .args(["-hide_banner", "-loglevel", "error", "-f", "nut", "-i"])
        .arg(path)
        .args(["-f", "rawvideo", "-pix_fmt", "rgba", "-"])
        .stdin(Stdio::null())
        .output()
        .expect("spawn ffmpeg to unwrap a NUT file");
    assert!(
        output.status.success(),
        "unwrapping {} exited with {:?}\nstderr:\n{}",
        path.display(),
        output.status.code(),
        String::from_utf8_lossy(&output.stderr)
    );
    output.stdout
}

/// Runs one ffrwd-wasm over a NUT file, writing a NUT file.
fn run_stage(module: &Path, module_args: &[&str], input: &Path, output: &Path) -> StageResult {
    ensure_modules_built();
    let mut args = ffrwd_wasm_args(module, module_args);
    let input_index = args.iter().position(|a| a == "-i").expect("-i is there") + 1;
    args[input_index] = input.to_str().expect("path is valid UTF-8").to_string();
    args.push("-f".into());
    args.push("nut".into());
    args.push(output.to_str().expect("path is valid UTF-8").into());

    let out = Command::new(env!("CARGO_BIN_EXE_ffrwd-wasm"))
        .args(&args)
        .stdin(Stdio::null())
        .output()
        .expect("spawn ffrwd-wasm");
    StageResult {
        status: out.status,
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
    }
}

#[test]
fn rows_ride_beside_the_frames_from_one_sidecar_to_the_next() {
    let source = "color=c=red:s=8x8:r=25:d=0.2";
    let frames = 5;
    let reference = run_reference(source, frames);

    let produced = TempFile::new("chain-source.nut");
    write_nut(source, frames, "rgba", produced.path());

    // First sidecar: framestats emits a row per frame, and -annotations out
    // puts them on the wire beside the frames it passes through.
    let annotated = TempFile::new("chain-annotated.nut");
    let first = run_stage(
        &module_path("framestats"),
        &["-annotations", "out"],
        produced.path(),
        annotated.path(),
    );
    assert_stage_ok(&first, "framestats sidecar");

    // What actually reached the wire, read back frame by frame.
    let wire = std::fs::File::open(annotated.path()).expect("open the annotated wire");
    let mut demuxer = Demuxer::open_annotated(wire).expect("read the annotated NUT headers");
    assert!(
        demuxer.has_annotations(),
        "-annotations out should declare the second stream"
    );
    let mut buf = Vec::new();
    let mut seen = 0;
    while let Some(pts) = demuxer.read_frame(&mut buf).expect("read a frame") {
        let rows = demuxer.take_rows(pts);
        assert_eq!(
            rows.len(),
            1,
            "frame {seen} should carry framestats' one row"
        );
        let row: FramestatsRow = serde_json::from_str(&rows[0])
            .unwrap_or_else(|e| panic!("parsing row {:?}: {e}", rows[0]));
        assert_eq!(row.mean, 84, "frame {seen} mean");
        // The row's pts is the host's stamp - framestats declares none of its
        // own - and it must be the exact pts the frame carried on the wire,
        // not the frame's index among the ones seen so far.
        assert_eq!(
            row.pts, pts,
            "frame {seen}'s row carries the pts of the frame it rode"
        );
        assert!(
            (row.time - seen as f64 * 0.04).abs() < 1e-9,
            "frame {seen} row time {} is not the frame's",
            row.time
        );
        seen += 1;
    }
    assert_eq!(seen, frames as usize, "every frame comes back");

    // Second sidecar: blur-boxes reads those rows. None of them is a
    // rectangle, so every frame passes through untouched.
    let blurred = TempFile::new("chain-blurred.nut");
    let second = run_stage(
        &module_path("blur_boxes"),
        &["-annotations", "in"],
        annotated.path(),
        blurred.path(),
    );
    assert_stage_ok(&second, "blur-boxes sidecar");
    assert_eq!(
        unwrap_nut(blurred.path()),
        reference,
        "rows that name no rectangle leave the frames alone"
    );
}

#[test]
fn two_sidecars_run_as_one_pipeline() {
    // The whole topology on real OS pipes: ffmpeg, two sidecars sharing one
    // annotated wire, then ffmpeg again.
    ensure_modules_built();
    let source = "color=c=red:s=8x8:r=25:d=0.2";
    let frames = 5;
    let reference = run_reference(source, frames);

    let mut producer = spawn_producer(source, frames, "rgba");
    let producer_stdout = producer.stdout.take().expect("producer stdout");
    let producer_stderr = drain_stderr(producer.stderr.take().expect("producer stderr"));

    let mut first_args = ffrwd_wasm_args(&module_path("facebox"), &["-annotations", "out"]);
    first_args.extend(["-f".to_string(), "nut".to_string(), "-".to_string()]);
    let mut first: Child = Command::new(env!("CARGO_BIN_EXE_ffrwd-wasm"))
        .args(&first_args)
        .stdin(Stdio::from(producer_stdout))
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn the detecting sidecar");
    let first_stdout = first.stdout.take().expect("first sidecar stdout");
    let first_stderr = drain_stderr(first.stderr.take().expect("first sidecar stderr"));

    let mut second_args = ffrwd_wasm_args(&module_path("blur_boxes"), &["-annotations", "in"]);
    second_args.extend(["-f".to_string(), "nut".to_string(), "-".to_string()]);
    let mut second: Child = Command::new(env!("CARGO_BIN_EXE_ffrwd-wasm"))
        .args(&second_args)
        .stdin(Stdio::from(first_stdout))
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn the blurring sidecar");
    let second_stdout = second.stdout.take().expect("second sidecar stdout");
    let second_stderr = drain_stderr(second.stderr.take().expect("second sidecar stderr"));

    let consumer = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "nut",
            "-i",
            "-",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgba",
            "-",
        ])
        .stdin(Stdio::from(second_stdout))
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .expect("spawn consumer ffmpeg");

    let stages = [
        (
            "producer ffmpeg",
            producer.wait().expect("wait for producer"),
            producer_stderr,
        ),
        (
            "facebox sidecar",
            first.wait().expect("wait for the detecting sidecar"),
            first_stderr,
        ),
        (
            "blur-boxes sidecar",
            second.wait().expect("wait for the blurring sidecar"),
            second_stderr,
        ),
    ];
    for (name, status, stderr) in stages {
        let stderr = stderr.join().expect("stderr thread panicked");
        assert_stage_ok(&StageResult { status, stderr }, name);
    }
    assert_stage_ok(
        &StageResult {
            status: consumer.status,
            stderr: String::from_utf8_lossy(&consumer.stderr).into_owned(),
        },
        "consumer ffmpeg",
    );

    // A flat colour field holds no faces, so nothing is detected and nothing
    // is blurred: what comes out is what went in.
    assert_eq!(consumer.stdout, reference);
}

#[test]
fn unpublished_pixel_formats_are_refused() {
    // The stream header says yuv420p, which the invert module does not
    // publish. The refusal must name what it does publish.
    ensure_modules_built();
    let yuv_source = TempFile::new("yuv-source.nut");
    write_nut("color=c=red:s=8x8:r=1:d=1", 1, "yuv420p", yuv_source.path());

    let mut args = ffrwd_wasm_args(&module_path("invert"), &[]);
    let input_index = args.iter().position(|a| a == "-i").expect("-i is there") + 1;
    args[input_index] = yuv_source
        .path()
        .to_str()
        .expect("path is valid UTF-8")
        .to_string();
    args.extend(["-f".to_string(), "ndjson".to_string()]);
    let rows_file = TempFile::new("invert-yuv-rows");
    args.push(
        rows_file
            .path()
            .to_str()
            .expect("path is valid UTF-8")
            .to_string(),
    );

    let output = Command::new(env!("CARGO_BIN_EXE_ffrwd-wasm"))
        .args(&args)
        .stdin(Stdio::null())
        .output()
        .expect("spawn ffrwd-wasm");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !output.status.success(),
        "ffrwd-wasm should refuse a pixel format the module does not publish"
    );
    assert!(
        stderr.contains("rgba"),
        "stderr should contain the module's published pixel format list, got:\n{stderr}"
    );
}

#[test]
fn double_doubles_the_frames_and_keeps_the_stretch_of_time() {
    ensure_modules_built();
    let source = "testsrc2=s=8x8:r=25:d=1.2";
    let frames = 30;

    let produced = TempFile::new("double-source.nut");
    write_nut(source, frames, "rgba", produced.path());
    let doubled = TempFile::new("double-out.nut");
    let stage = run_stage(&module_path("double"), &[], produced.path(), doubled.path());
    assert_stage_ok(&stage, "double sidecar");

    let before = packet_times(produced.path());
    let after = packet_times(doubled.path());
    assert_eq!(before.len(), frames as usize);
    assert_eq!(
        after.len(),
        2 * frames as usize,
        "every frame should come out twice"
    );

    // Each frame keeps its own timestamp and its copy lands between it and
    // the next frame, so the picture rate doubles without the stream covering
    // a different stretch of time.
    for (i, time) in before.iter().enumerate() {
        assert!(
            (after[2 * i] - time).abs() < 1e-9,
            "frame {i} should keep its timestamp: {} against {time}",
            after[2 * i]
        );
        assert!(
            after[2 * i + 1] > after[2 * i],
            "frame {i}'s copy should come after it"
        );
        if let Some(next) = before.get(i + 1) {
            assert!(
                after[2 * i + 1] < *next,
                "frame {i}'s copy should land before frame {}",
                i + 1
            );
        }
    }

    let interval = span(&before) / (before.len() as f64 - 1.0);
    assert!(
        span(&after) - span(&before) < interval,
        "the doubled stream spans less than one extra frame: {} against {}",
        span(&after),
        span(&before)
    );

    // Real ffmpeg decodes all of them, not just the packets ffprobe counted.
    assert_eq!(
        unwrap_nut(doubled.path()).len(),
        2 * frames as usize * FRAME_LEN
    );
}

// The module network: several modules in one process, wired the way an ffmpeg
// filtergraph is. Real ffmpeg on both ends of every one of these.

/// The `-m name=path` table and the `-filter_complex` wiring, as one argv for
/// a network sidecar reading NUT on stdin.
fn network_args(modules: &[(&str, &str)], wiring: &str) -> Vec<String> {
    let mut args: Vec<String> = ["-f", "nut", "-i", "-"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    for (name, module) in modules {
        args.push("-m".to_string());
        let path = module_path(module);
        args.push(format!(
            "{name}={}",
            path.to_str().expect("module path is valid UTF-8")
        ));
    }
    args.push("-filter_complex".to_string());
    args.push(wiring.to_string());
    args
}

#[test]
fn the_faces_pair_runs_as_one_network_process() {
    // The wiring the compiler emits for `blur_boxes(detect_faces(v))`, byte for
    // byte, hosted by a single sidecar between two ffmpegs. The rows facebox
    // writes reach blur-boxes inside the process, so neither edge carries an
    // annotation stream and neither ffmpeg has to know about one.
    let source = "color=c=red:s=8x8:r=25:d=0.2";
    let frames = 5;
    let reference = run_reference(source, frames);

    let mut args = network_args(
        &[("facebox", "facebox"), ("blur_boxes", "blur_boxes")],
        "[0:v]facebox[n1];[n1]blur_boxes[out0]",
    );
    args.extend(
        ["-map", "[out0]", "-f", "nut", "-"]
            .iter()
            .map(|s| s.to_string()),
    );

    let run = run_pipeline_args(source, frames, args);
    assert_pipeline_ok(&run);
    assert!(
        !run.ffrwd_wasm.stderr.contains("annotations"),
        "no annotation stream should be involved, got:\n{}",
        run.ffrwd_wasm.stderr
    );

    // A flat colour field holds no faces, so nothing is detected and nothing
    // is blurred: every frame comes through as it went in.
    assert_eq!(
        run.stdout, reference,
        "every frame must reach the far ffmpeg unchanged"
    );
}

#[test]
fn the_whole_face_chain_runs_as_one_network_over_a_cut() {
    // shots into facebox into blur-boxes, one process, over a stream with a
    // real cut in it: the shot rows reach facebox in memory and are consumed
    // there, and the rectangles facebox would emit reach blur-boxes the same
    // way. Neither edge carries an annotation stream.
    let reference = run_reference(CUT_SOURCE, CUT_FRAMES);

    let mut args = network_args(
        &[
            ("shots", "shots"),
            ("facebox", "facebox"),
            ("blur_boxes", "blur_boxes"),
        ],
        "[0:v]shots[a];[a]facebox[b];[b]blur_boxes[out0]",
    );
    args.extend(
        ["-map", "[out0]", "-f", "nut", "-"]
            .iter()
            .map(|s| s.to_string()),
    );

    let run = run_pipeline_args(CUT_SOURCE, CUT_FRAMES, args);
    assert_pipeline_ok(&run);
    assert!(
        !run.ffrwd_wasm.stderr.contains("annotations"),
        "no annotation stream should be involved, got:\n{}",
        run.ffrwd_wasm.stderr
    );

    // Neither half of the stream holds a face, so nothing is detected and
    // nothing is blurred: every frame reaches the far ffmpeg as it left the
    // source, cut and all.
    assert_eq!(
        run.stdout, reference,
        "every frame must cross the three modules unchanged"
    );
}

#[test]
fn a_network_hands_one_modules_frames_to_two_chains() {
    // [a] is read twice. An ffmpeg graph would need a `split` here; a network
    // hands the frames to both readers in memory, and the two chains change
    // the frame count differently - so what each output holds proves it got
    // its own copy.
    ensure_modules_built();
    let source = "testsrc2=s=8x8:r=25:d=0.4";
    let frames = 10;

    let produced = TempFile::new("fanout-source.nut");
    write_nut(source, frames, "rgba", produced.path());
    let doubled = TempFile::new("fanout-doubled.nut");
    let threes = TempFile::new("fanout-threes.nut");

    let mut args = network_args(
        &[
            ("invert", "invert"),
            ("double", "double"),
            ("tail3", "tail3"),
        ],
        "[0:v]invert[a];[a]double[out0];[a]tail3[out1]",
    );
    let input_index = args.iter().position(|a| a == "-i").expect("-i is there") + 1;
    args[input_index] = produced
        .path()
        .to_str()
        .expect("path is valid UTF-8")
        .to_string();
    for (label, target) in [("[out0]", doubled.path()), ("[out1]", threes.path())] {
        args.push("-map".into());
        args.push(label.into());
        args.push("-f".into());
        args.push("nut".into());
        args.push(target.to_str().expect("path is valid UTF-8").into());
    }

    let output = Command::new(env!("CARGO_BIN_EXE_ffrwd-wasm"))
        .args(&args)
        .stdin(Stdio::null())
        .output()
        .expect("spawn ffrwd-wasm");
    assert!(
        output.status.success(),
        "the fan-out network exited with {:?}\nstderr:\n{}",
        output.status.code(),
        String::from_utf8_lossy(&output.stderr)
    );

    // Both chains start from the inverted frames, so both must match the
    // source inverted - one of them twice over.
    let mut inverted = run_reference(source, frames);
    for (index, byte) in inverted.iter_mut().enumerate() {
        if index % 4 != 3 {
            *byte = 255 - *byte;
        }
    }

    let through_tail3 = unwrap_nut(threes.path());
    assert_eq!(
        through_tail3, inverted,
        "tail3 passes every inverted frame through untouched"
    );

    let through_double = unwrap_nut(doubled.path());
    assert_eq!(
        through_double.len(),
        2 * inverted.len(),
        "double emits two frames for every one it is handed"
    );
    for index in 0..frames as usize {
        let want = frame_at(&inverted, index);
        assert_eq!(
            frame_at(&through_double, 2 * index),
            want,
            "the first copy of frame {index}"
        );
        assert_eq!(
            frame_at(&through_double, 2 * index + 1),
            want,
            "the second copy of frame {index}"
        );
    }
}

#[test]
fn a_module_that_changes_the_frame_count_runs_inside_a_network() {
    // invert into double: M frames in, 2M out, through two nodes of one
    // process, and the timestamps still cover the stretch of time they did.
    ensure_modules_built();
    let source = "testsrc2=s=8x8:r=25:d=0.4";
    let frames = 10;

    let produced = TempFile::new("chain-double-source.nut");
    write_nut(source, frames, "rgba", produced.path());
    let out = TempFile::new("chain-double-out.nut");

    let mut args = network_args(
        &[("invert", "invert"), ("double", "double")],
        "[0:v]invert[a];[a]double[out0]",
    );
    let input_index = args.iter().position(|a| a == "-i").expect("-i is there") + 1;
    args[input_index] = produced
        .path()
        .to_str()
        .expect("path is valid UTF-8")
        .to_string();
    args.push("-map".into());
    args.push("[out0]".into());
    args.push("-f".into());
    args.push("nut".into());
    args.push(out.path().to_str().expect("path is valid UTF-8").into());

    let output = Command::new(env!("CARGO_BIN_EXE_ffrwd-wasm"))
        .args(&args)
        .stdin(Stdio::null())
        .output()
        .expect("spawn ffrwd-wasm");
    assert!(
        output.status.success(),
        "the invert-into-double network exited with {:?}\nstderr:\n{}",
        output.status.code(),
        String::from_utf8_lossy(&output.stderr)
    );

    let before = packet_times(produced.path());
    let after = packet_times(out.path());
    assert_eq!(before.len(), frames as usize);
    assert_eq!(after.len(), 2 * frames as usize, "M in, 2M out");
    for (index, time) in before.iter().enumerate() {
        assert!(
            (after[2 * index] - time).abs() < 1e-9,
            "frame {index} keeps its own timestamp"
        );
    }
    let interval = span(&before) / (before.len() as f64 - 1.0);
    assert!(
        span(&after) - span(&before) < interval,
        "the doubled stream spans less than one extra frame: {} against {}",
        span(&after),
        span(&before)
    );

    // Real ffmpeg decodes all of them, and every one is the inverted picture.
    let decoded = unwrap_nut(out.path());
    assert_eq!(decoded.len(), 2 * frames as usize * FRAME_LEN);
    let reference = run_reference(source, frames);
    for index in 0..frames as usize {
        let want = frame_at(&reference, index);
        let got = frame_at(&decoded, 2 * index);
        for pixel in (0..FRAME_LEN).step_by(4) {
            assert_eq!(
                got[pixel],
                255 - want[pixel],
                "frame {index} red byte {pixel} must come from source frame {index}"
            );
        }
    }
}

/// Runs one network sidecar over `inputs` as files, writing one NUT file, and
/// returns how it exited.
fn run_network_stage(
    modules: &[(&str, &str)],
    wiring: &str,
    inputs: &[&Path],
    output: &Path,
) -> StageResult {
    ensure_modules_built();
    let mut args: Vec<String> = Vec::new();
    for input in inputs {
        args.push("-f".into());
        args.push("nut".into());
        args.push("-i".into());
        args.push(input.to_str().expect("path is valid UTF-8").into());
    }
    let tail = network_args(modules, wiring);
    // `network_args` opens with its own single stdin input, which the files
    // above replace.
    args.extend(tail.into_iter().skip(4));
    args.push("-map".into());
    args.push("[out0]".into());
    args.push("-f".into());
    args.push("nut".into());
    args.push(output.to_str().expect("path is valid UTF-8").into());

    let out = Command::new(env!("CARGO_BIN_EXE_ffrwd-wasm"))
        .args(&args)
        .stdin(Stdio::null())
        .output()
        .expect("spawn ffrwd-wasm");
    StageResult {
        status: out.status,
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
    }
}

/// The source and frame count every two-pad test here runs over.
const TWO_PAD_SOURCE: &str = "testsrc2=s=8x8:r=25:d=0.2";
const TWO_PAD_FRAMES: u32 = 5;

#[test]
fn a_module_reading_two_streams_reads_them_in_lockstep() {
    ensure_modules_built();

    let together = TempFile::new("lockstep-source.nut");
    write_nut(TWO_PAD_SOURCE, TWO_PAD_FRAMES, "rgba", together.path());
    let out = TempFile::new("lockstep-out.nut");

    // The same file on both pads: every timestamp matches its twin, so the
    // module is fed once per pair. pick_first keeps pad 0, so an output equal
    // to the source says the pads arrived paired and in pad order.
    let stage = run_network_stage(
        &[("pick", "pick_first")],
        "[0:v][1:v]pick[out0]",
        &[together.path(), together.path()],
        out.path(),
    );
    assert_stage_ok(&stage, "a two-pad node in lockstep");
    assert_eq!(
        unwrap_nut(out.path()),
        run_reference(TWO_PAD_SOURCE, TWO_PAD_FRAMES),
        "pad 0 is what leaves, byte for byte"
    );
}

#[test]
fn one_input_read_twice_arrives_on_both_pads() {
    ensure_modules_built();

    // What the compiler emits when one boundary read feeds two pads: the same
    // label twice. Both pads carry the same frame at the same timestamp, and
    // neither may swallow the other's copy.
    let source = TempFile::new("repeated-source.nut");
    write_nut(TWO_PAD_SOURCE, TWO_PAD_FRAMES, "rgba", source.path());
    let out = TempFile::new("repeated-out.nut");

    let stage = run_network_stage(
        &[("pick", "pick_first")],
        "[0:v][0:v]pick[out0]",
        &[source.path()],
        out.path(),
    );
    assert_stage_ok(&stage, "one input on two pads");
    assert_eq!(
        unwrap_nut(out.path()),
        run_reference(TWO_PAD_SOURCE, TWO_PAD_FRAMES),
        "every frame comes through, so no pad went hungry"
    );
}

#[test]
fn two_streams_that_drift_apart_are_refused_by_name() {
    ensure_modules_built();
    let plain = TempFile::new("drift-plain.nut");
    write_nut(TWO_PAD_SOURCE, TWO_PAD_FRAMES, "rgba", plain.path());
    let stretched = TempFile::new("drift-stretched.nut");
    write_nut(
        "testsrc2=s=8x8:r=25:d=0.2,setpts=2.0*PTS",
        TWO_PAD_FRAMES,
        "rgba",
        stretched.path(),
    );
    let out = TempFile::new("drift-out.nut");

    let stage = run_network_stage(
        &[("pick", "pick_first")],
        "[0:v][1:v]pick[out0]",
        &[plain.path(), stretched.path()],
        out.path(),
    );
    assert!(
        !stage.status.success(),
        "the second stream runs at half the rate, so the pads never line up"
    );
    assert!(
        stage.stderr.contains("pick_first") && stage.stderr.contains("lockstep"),
        "expected the module and the rule named, got:\n{}",
        stage.stderr
    );
    assert!(
        stage.stderr.contains("pad 0") && stage.stderr.contains("pad 1"),
        "expected both pads named, got:\n{}",
        stage.stderr
    );
}

// blur_mask, the other half of the depth-of-field pair: a frame on pad 0 and
// a mask on pad 1, both made by ffmpeg. The frame is big enough that a blur
// has somewhere to spread.

/// A picture with hard detail in it, which any blur softens.
const MASKED_SOURCE: &str = "testsrc2=s=64x64:r=25:d=0.2";
const MASKED_FRAMES: u32 = 5;

/// A mask running black on the left to white on the right, so one run covers
/// every amount of blur at once.
const GRADIENT_MASK: &str = "color=c=black:s=64x64:r=25,format=rgba,\
                             geq=r='X*255/(W-1)':g='X*255/(W-1)':b='X*255/(W-1)':a=255";

/// A mask of nothing at all: black everywhere is no blur anywhere.
const BLACK_MASK: &str = "color=c=black:s=64x64:r=25";

#[test]
fn a_mask_of_black_leaves_every_frame_exactly_as_it_arrived() {
    ensure_modules_built();
    let frames = TempFile::new("blur-mask-frames.nut");
    write_nut(MASKED_SOURCE, MASKED_FRAMES, "rgba", frames.path());
    let mask = TempFile::new("blur-mask-black.nut");
    write_nut(BLACK_MASK, MASKED_FRAMES, "rgba", mask.path());
    let out = TempFile::new("blur-mask-black-out.nut");

    let stage = run_network_stage(
        &[("blur_mask", "blur_mask")],
        "[0:v][1:v]blur_mask[out0]",
        &[frames.path(), mask.path()],
        out.path(),
    );
    assert_stage_ok(&stage, "blur_mask under a black mask");
    assert_eq!(
        unwrap_nut(out.path()),
        run_reference(MASKED_SOURCE, MASKED_FRAMES),
        "nothing is masked, so the frames come through byte for byte"
    );
}

#[test]
fn a_gradient_mask_blurs_the_end_it_points_at() {
    ensure_modules_built();
    let frames = TempFile::new("blur-mask-gradient-frames.nut");
    write_nut(MASKED_SOURCE, MASKED_FRAMES, "rgba", frames.path());
    let mask = TempFile::new("blur-mask-gradient.nut");
    write_nut(GRADIENT_MASK, MASKED_FRAMES, "rgba", mask.path());
    let out = TempFile::new("blur-mask-gradient-out.nut");

    let stage = run_network_stage(
        &[("blur_mask", "blur_mask")],
        "[0:v][1:v]blur_mask[out0]",
        &[frames.path(), mask.path()],
        out.path(),
    );
    assert_stage_ok(&stage, "blur_mask under a gradient mask");

    let blurred = unwrap_nut(out.path());
    let sharp = run_reference(MASKED_SOURCE, MASKED_FRAMES);
    assert_eq!(blurred.len(), sharp.len(), "one frame out per frame in");
    assert_ne!(blurred, sharp, "a mask that is not black changes the frame");

    // The left edge is under a black mask and the right under a white one, so
    // whatever detail the picture has must survive on one side and not the
    // other. Read on the first frame's first row.
    let row = |bytes: &[u8], from: usize, to: usize| -> Vec<i32> {
        (from..to).map(|x| i32::from(bytes[x * 4])).collect()
    };
    let roughness =
        |values: &[i32]| -> i32 { values.windows(2).map(|p| (p[1] - p[0]).abs()).sum() };
    assert_eq!(
        row(&blurred, 0, 4),
        row(&sharp, 0, 4),
        "the black end of the mask leaves the pixels alone"
    );
    assert!(
        roughness(&row(&blurred, 48, 64)) < roughness(&row(&sharp, 48, 64)),
        "and the white end is softer than it arrived"
    );
}

#[test]
fn a_module_wired_more_pads_than_it_reads_is_refused_by_name() {
    ensure_modules_built();
    let source = TempFile::new("arity-source.nut");
    write_nut(TWO_PAD_SOURCE, TWO_PAD_FRAMES, "rgba", source.path());
    let out = TempFile::new("arity-out.nut");

    // invert reads one stream, and the wiring gives it two.
    let stage = run_network_stage(
        &[("inv", "invert")],
        "[0:v][0:v]inv[out0]",
        &[source.path()],
        out.path(),
    );
    assert!(
        !stage.status.success(),
        "a one-stream module handed two pads has no answer for the second"
    );
    assert!(
        stage.stderr.contains("inv") && stage.stderr.contains("reads 1"),
        "expected the module and what it reads named, got:\n{}",
        stage.stderr
    );
}

// Audio, with real ffmpeg on both ends: an lavfi source encoded as pcm inside
// NUT, through a sidecar, and read back by ffmpeg again. What the sidecar is
// handed is whatever ffmpeg chose to packetize into - 1024 samples for pcm,
// and `asetnsamples` where a test wants a different shape - which is the point
// of the re-chunker.

/// The rate every audio test runs at, and the length of the tone.
const AUDIO_RATE: u32 = 48_000;
const AUDIO_SECONDS: f64 = 1.0;
const AUDIO_SAMPLES: usize = 48_000;

/// A one-second 440 Hz tone, optionally re-packetized by `filter`.
fn sine_source(filter: Option<&str>) -> (String, String) {
    let source = format!("sine=f=440:r={AUDIO_RATE}:d={AUDIO_SECONDS}");
    let chain = filter.unwrap_or("anull").to_string();
    (source, chain)
}

/// Writes a NUT file of pcm in `sample_fmt`, the shape ffmpeg puts on a pipe.
fn write_audio_nut(filter: Option<&str>, sample_fmt: &str, target: &Path) {
    let (source, chain) = sine_source(filter);
    let output = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            &source,
            "-af",
            &chain,
            "-c:a",
            &format!("pcm_{sample_fmt}"),
            "-f",
            "nut",
        ])
        .arg(target)
        .stdin(Stdio::null())
        .output()
        .expect("spawn producer ffmpeg");
    assert!(
        output.status.success(),
        "producer ffmpeg exited with {:?}\nstderr:\n{}",
        output.status.code(),
        String::from_utf8_lossy(&output.stderr)
    );
}

/// Every sample in a NUT audio file, decoded to f32 by real ffmpeg so both
/// wire formats are compared the same way.
fn decode_samples(path: &Path) -> Vec<f32> {
    let output = Command::new("ffmpeg")
        .args(["-hide_banner", "-loglevel", "error", "-f", "nut", "-i"])
        .arg(path)
        .args(["-f", "f32le", "-ac", "1", "-"])
        .stdin(Stdio::null())
        .output()
        .expect("spawn ffmpeg to decode audio");
    assert!(
        output.status.success(),
        "decoding {} exited with {:?}\nstderr:\n{}",
        path.display(),
        output.status.code(),
        String::from_utf8_lossy(&output.stderr)
    );
    let (whole, _) = output.stdout.as_chunks::<4>();
    whole.iter().copied().map(f32::from_le_bytes).collect()
}

/// Root mean square of a run of samples.
fn rms(samples: &[f32]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let total: f64 = samples.iter().map(|s| f64::from(*s) * f64::from(*s)).sum();
    (total / samples.len() as f64).sqrt()
}

/// How many samples each packet of an audio file holds, as ffprobe reads the
/// durations. At the natural time base a packet's duration is its samples.
fn packet_samples(path: &Path) -> Vec<i64> {
    let output = Command::new("ffprobe")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "packet=duration",
            "-of",
            "csv=p=0",
        ])
        .arg(path)
        .stdin(Stdio::null())
        .output()
        .expect("spawn ffprobe");
    assert!(output.status.success(), "ffprobe on {}", path.display());
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(|line| {
            line.trim_end_matches(',')
                .parse()
                .unwrap_or_else(|e| panic!("parsing ffprobe duration {line:?}: {e}"))
        })
        .collect()
}

/// Runs one ffrwd-wasm over a NUT file, writing a NUT file and, when asked, an
/// ndjson file beside it.
fn run_audio_stage(
    module: &Path,
    module_args: &[&str],
    input: &Path,
    frames_out: Option<&Path>,
    rows_out: Option<&Path>,
) -> StageResult {
    ensure_modules_built();
    let mut args = ffrwd_wasm_args(module, module_args);
    let input_index = args.iter().position(|a| a == "-i").expect("-i is there") + 1;
    args[input_index] = input.to_str().expect("path is valid UTF-8").to_string();
    if let Some(p) = frames_out {
        args.push("-f".into());
        args.push("nut".into());
        args.push(p.to_str().expect("path is valid UTF-8").into());
    }
    if let Some(p) = rows_out {
        args.push("-f".into());
        args.push("ndjson".into());
        args.push(p.to_str().expect("path is valid UTF-8").into());
    }

    let out = Command::new(env!("CARGO_BIN_EXE_ffrwd-wasm"))
        .args(&args)
        .stdin(Stdio::null())
        .output()
        .expect("spawn ffrwd-wasm");
    StageResult {
        status: out.status,
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
    }
}

#[test]
fn gain_halves_a_tone_on_both_wires_without_losing_a_sample() {
    for sample_fmt in ["f32le", "s16le"] {
        let source = TempFile::new("gain-source.nut");
        write_audio_nut(None, sample_fmt, source.path());
        let quieter = TempFile::new("gain-out.nut");
        let stage = run_audio_stage(
            &module_path("again"),
            &["-params", r#"{"gain":0.5}"#],
            source.path(),
            Some(quieter.path()),
            None,
        );
        assert_stage_ok(&stage, &format!("again on the {sample_fmt} wire"));

        let before = decode_samples(source.path());
        let after = decode_samples(quieter.path());
        assert_eq!(
            before.len(),
            AUDIO_SAMPLES,
            "the source is a second of {sample_fmt}"
        );
        assert_eq!(
            after.len(),
            before.len(),
            "{sample_fmt}: every sample that went in comes out"
        );

        let (loud, quiet) = (rms(&before), rms(&after));
        assert!(
            (quiet - loud / 2.0).abs() < loud * 1e-3,
            "{sample_fmt}: {loud} at half gain measured {quiet}"
        );
    }
}

#[test]
fn a_window_is_cut_out_of_whatever_packets_ffmpeg_sent() {
    // The producer packetizes in 1500s and rms works in 16000s, so no window
    // is a packet and no packet is a window: eleven arrivals make a window,
    // and the eleventh is split across two of them.
    let source = TempFile::new("rechunk-source.nut");
    write_audio_nut(Some("asetnsamples=n=1500"), "f32le", source.path());
    let arriving = packet_samples(source.path());
    assert_eq!(
        arriving.first().copied(),
        Some(1500),
        "the producer was asked for 1500-sample packets, got {arriving:?}"
    );
    assert_eq!(arriving.iter().sum::<i64>(), AUDIO_SAMPLES as i64);

    let measured = TempFile::new("rechunk-out.nut");
    let rows_file = TempFile::new("rechunk-rows");
    let stage = run_audio_stage(
        &module_path("rms"),
        &[],
        source.path(),
        Some(measured.path()),
        Some(rows_file.path()),
    );
    assert_stage_ok(&stage, "rms over re-cut packets");

    let text = std::fs::read_to_string(rows_file.path()).expect("read the ndjson output");
    let rows: Vec<serde_json::Value> = text
        .lines()
        .filter(|line| !line.is_empty())
        .map(|line| serde_json::from_str(line).unwrap_or_else(|e| panic!("parsing {line:?}: {e}")))
        .collect();
    assert_eq!(
        rows.len(),
        AUDIO_SAMPLES.div_ceil(16_000),
        "one row per window, and the windows are ceil(samples / stride) of them"
    );
    for (index, row) in rows.iter().enumerate() {
        assert_eq!(
            row["pts"].as_i64(),
            Some(index as i64 * 16_000),
            "row {index} starts where the stride left off"
        );
        assert_eq!(
            row["samples"].as_u64(),
            Some(16_000),
            "row {index} spans a whole window"
        );
    }

    // A full-scale sine measures 1/sqrt(2); this one is quieter than that, and
    // every window of the same tone measures the same.
    let levels: Vec<f64> = rows
        .iter()
        .map(|row| row["rms"].as_f64().expect("a level"))
        .collect();
    let whole = rms(&decode_samples(source.path()));
    for (index, level) in levels.iter().enumerate() {
        assert!(
            (level - whole).abs() < whole * 1e-2,
            "window {index} measured {level} where the whole tone measures {whole}"
        );
    }

    // The audio itself passes through: the windows leave as the samples that
    // arrived, so the packets on the way out are the windows.
    assert_eq!(
        packet_samples(measured.path()),
        vec![16_000; 3],
        "the output is windows, whatever arrived"
    );
    assert_eq!(decode_samples(measured.path()).len(), AUDIO_SAMPLES);
}

#[test]
fn rows_and_a_trailing_record_ride_audio_from_one_sidecar_to_the_next() {
    // rms puts a level on the wire beside the samples; again reads that wire,
    // passes the levels on, and hands the trailing record on with them.
    let source = TempFile::new("audio-chain-source.nut");
    write_audio_nut(Some("asetnsamples=n=1500"), "f32le", source.path());

    let annotated = TempFile::new("audio-chain-annotated.nut");
    let first = run_audio_stage(
        &module_path("rms"),
        &["-annotations", "out"],
        source.path(),
        Some(annotated.path()),
        None,
    );
    assert_stage_ok(&first, "the rms sidecar");

    // What actually reached the wire, window by window.
    let wire = std::fs::read(annotated.path()).expect("read the annotated wire");
    let mut demuxer = Demuxer::open_annotated(&wire[..]).expect("read the annotated NUT headers");
    assert!(
        demuxer.has_annotations(),
        "-annotations out should declare the second stream"
    );
    let mut buf = Vec::new();
    let mut seen = 0;
    while let Some(pts) = demuxer.read_frame(&mut buf).expect("read a packet") {
        let rows = demuxer.take_rows(pts);
        assert_eq!(rows.len(), 1, "window {seen} carries rms' one row");
        assert_eq!(
            pts,
            seen as i64 * 16_000,
            "window {seen} starts on a stride"
        );
        seen += 1;
    }
    assert_eq!(seen, AUDIO_SAMPLES.div_ceil(16_000), "every window");

    // Second sidecar: again neither reads the levels nor invents any, and the
    // rows come out the far side unchanged.
    let quieter = TempFile::new("audio-chain-quieter.nut");
    let second = run_audio_stage(
        &module_path("again"),
        &[
            "-params",
            r#"{"gain":0.5}"#,
            "-annotations",
            "in",
            "-annotations",
            "out",
        ],
        annotated.path(),
        Some(quieter.path()),
        None,
    );
    assert_stage_ok(&second, "the again sidecar");

    let wire = std::fs::read(quieter.path()).expect("read the far wire");
    let mut demuxer = Demuxer::open_annotated(&wire[..]).expect("read the annotated NUT headers");
    let mut carried = Vec::new();
    let mut buf = Vec::new();
    while let Some(pts) = demuxer.read_frame(&mut buf).expect("read a packet") {
        carried.extend(demuxer.take_rows(pts));
    }
    assert_eq!(
        carried.len(),
        AUDIO_SAMPLES.div_ceil(16_000),
        "every level crosses the module that forwards rows"
    );

    let before = rms(&decode_samples(source.path()));
    let after = rms(&decode_samples(quieter.path()));
    assert!(
        (after - before / 2.0).abs() < before * 1e-3,
        "the samples were quietened on the way through: {before} became {after}"
    );
}

#[test]
fn an_audio_chain_and_a_video_chain_run_in_one_sidecar() {
    ensure_modules_built();
    let video = TempFile::new("mixed-video.nut");
    write_nut("testsrc2=s=8x8:r=25:d=0.4", 10, "rgba", video.path());
    let audio = TempFile::new("mixed-audio.nut");
    write_audio_nut(None, "f32le", audio.path());

    let video_out = TempFile::new("mixed-video-out.nut");
    let audio_out = TempFile::new("mixed-audio-out.nut");

    let mut args: Vec<String> = Vec::new();
    for input in [video.path(), audio.path()] {
        args.push("-f".into());
        args.push("nut".into());
        args.push("-i".into());
        args.push(input.to_str().expect("path is valid UTF-8").into());
    }
    for (name, module) in [("invert", "invert"), ("again", "again")] {
        args.push("-m".into());
        args.push(format!(
            "{name}={}",
            module_path(module).to_str().expect("valid UTF-8")
        ));
    }
    args.push("-filter_complex".into());
    args.push("[0:v]invert[outv];[1:a]again=gain=0.5[outa]".into());
    for (label, target) in [("[outv]", video_out.path()), ("[outa]", audio_out.path())] {
        args.push("-map".into());
        args.push(label.into());
        args.push("-f".into());
        args.push("nut".into());
        args.push(target.to_str().expect("path is valid UTF-8").into());
    }

    let out = Command::new(env!("CARGO_BIN_EXE_ffrwd-wasm"))
        .args(&args)
        .stdin(Stdio::null())
        .output()
        .expect("spawn ffrwd-wasm");
    assert_stage_ok(
        &StageResult {
            status: out.status,
            stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
        },
        "the mixed network",
    );

    // The video chain: every frame inverted, and the frame count unchanged.
    let mut inverted = run_reference("testsrc2=s=8x8:r=25:d=0.4", 10);
    for (index, byte) in inverted.iter_mut().enumerate() {
        if index % 4 != 3 {
            *byte = 255 - *byte;
        }
    }
    assert_eq!(
        unwrap_nut(video_out.path()),
        inverted,
        "the video chain is untouched by the audio one beside it"
    );

    // The audio chain: every sample halved, and the sample count unchanged.
    let before = decode_samples(audio.path());
    let after = decode_samples(audio_out.path());
    assert_eq!(after.len(), before.len());
    assert!(
        (rms(&after) - rms(&before) / 2.0).abs() < rms(&before) * 1e-3,
        "the audio chain is untouched by the video one beside it"
    );
}

#[test]
fn a_stream_of_the_wrong_kind_is_refused_naming_the_module() {
    ensure_modules_built();
    let audio = TempFile::new("kind-audio.nut");
    write_audio_nut(None, "f32le", audio.path());
    let video = TempFile::new("kind-video.nut");
    write_nut("testsrc2=s=8x8:r=25:d=0.2", 5, "rgba", video.path());
    let out = TempFile::new("kind-out.nut");

    // An audio edge into a video module, and a video edge into an audio one:
    // both name the edge and the module.
    for (wiring, input, module, wanted) in [
        (
            "[0:a]facebox[out0]",
            audio.path(),
            ("facebox", "facebox"),
            ["[0:a]", "facebox"],
        ),
        (
            "[0:v]again[out0]",
            video.path(),
            ("again", "again"),
            ["[0:v]", "again"],
        ),
    ] {
        let stage = run_network_stage(&[module], wiring, &[input], out.path());
        assert!(!stage.status.success(), "{wiring} crosses the kinds");
        for needle in wanted {
            assert!(
                stage.stderr.contains(needle),
                "expected {needle:?} in the refusal for {wiring}, got:\n{}",
                stage.stderr
            );
        }
    }

    // A module of a world that has no spelling for audio at all.
    let stage = run_audio_stage(
        &module_path("adapted_060"),
        &[],
        audio.path(),
        Some(out.path()),
        None,
    );
    assert!(!stage.status.success(), "a 0.6.0 module hosts video alone");
    assert!(
        stage.stderr.contains("adapted-060") && stage.stderr.contains("ffrwd:av@0.6.0"),
        "expected the module and its world named, got:\n{}",
        stage.stderr
    );
}

// Subtitles, both ends real. The cues do not ride the NUT edge - it keeps no
// per-packet duration - so the sidecar writes a document, and the terminal
// ffmpeg reads that document as an input like any other.

/// The whisper files transcribe compiles in, which a checkout may not have.
const MODEL_FILES: [&str; 4] = [
    "model.safetensors",
    "tokenizer.json",
    "config.json",
    "melfilters.bytes",
];

/// Whether the whisper files are where transcribe's build script looks.
fn model_present() -> bool {
    let dir = sidecar_root().join("modules/transcribe/model");
    MODEL_FILES.iter().all(|name| dir.join(name).is_file())
}

/// Writes the speech fixture as 16 kHz mono f32 pcm in NUT, which is what
/// transcribe publishes and what the host conforms a stream to.
fn write_speech_nut(fixture: &Path, target: &Path) {
    let output = Command::new("ffmpeg")
        .args(["-hide_banner", "-loglevel", "error", "-y", "-i"])
        .arg(fixture)
        .args([
            "-vn",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-c:a",
            "pcm_f32le",
            "-f",
            "nut",
        ])
        .arg(target)
        .stdin(Stdio::null())
        .output()
        .expect("spawn ffmpeg to write the speech NUT");
    assert!(
        output.status.success(),
        "writing the speech NUT exited with {:?}\nstderr:\n{}",
        output.status.code(),
        String::from_utf8_lossy(&output.stderr)
    );
}

/// The cues of a subtitle document, as `(start, end, text)` triples.
fn cues(srt: &str) -> Vec<(String, String, String)> {
    let mut out = Vec::new();
    let mut lines = srt.lines().peekable();
    while let Some(line) = lines.next() {
        let Some((start, end)) = line.split_once(" --> ") else {
            continue;
        };
        let mut text = Vec::new();
        while let Some(next) = lines.peek() {
            if next.trim().is_empty() {
                break;
            }
            text.push(lines.next().expect("peeked a line").to_string());
        }
        out.push((
            start.trim().to_string(),
            end.trim().to_string(),
            text.join("\n"),
        ));
    }
    out
}

/// The subtitle stream ffprobe reads off a file: its codec and its language
/// tag.
fn subtitle_stream(path: &Path) -> (String, String) {
    let output = Command::new("ffprobe")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-select_streams",
            "s:0",
            "-show_entries",
            "stream=codec_name:stream_tags=language",
            "-of",
            "csv=p=0",
        ])
        .arg(path)
        .stdin(Stdio::null())
        .output()
        .expect("spawn ffprobe");
    assert!(output.status.success(), "ffprobe on {}", path.display());
    let text = String::from_utf8_lossy(&output.stdout);
    let mut fields = text.trim().split(',');
    (
        fields.next().unwrap_or_default().to_string(),
        fields.next().unwrap_or_default().to_string(),
    )
}

/// The subtitles inside a container, pulled back out as an srt document.
fn extract_subtitles(path: &Path) -> String {
    let output = Command::new("ffmpeg")
        .args(["-hide_banner", "-loglevel", "error", "-i"])
        .arg(path)
        .args(["-map", "0:s:0", "-f", "srt", "-"])
        .stdin(Stdio::null())
        .output()
        .expect("spawn ffmpeg to extract subtitles");
    assert!(
        output.status.success(),
        "extracting subtitles from {} exited with {:?}\nstderr:\n{}",
        path.display(),
        output.status.code(),
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8_lossy(&output.stdout).into_owned()
}

#[test]
fn speech_becomes_cues_and_ffmpeg_muxes_them_into_an_mp4() {
    if !model_present() {
        eprintln!(
            "SKIPPED: transcribe's whisper model is absent. Run \
             sidecar/modules/transcribe/fetch-model.ps1 to download it."
        );
        return;
    }
    ensure_modules_built();

    let fixture = sidecar_root().join("modules/transcribe/tests/data/speech-es.wav");
    let source = TempFile::new("speech.nut");
    write_speech_nut(&fixture, source.path());

    // Spanish in, English out: the rows are cues, and the cues are a document.
    let subs = TempFile::new("speech.srt");
    let passed = TempFile::new("speech-out.nut");
    let module = module_path("transcribe");
    let out = Command::new(env!("CARGO_BIN_EXE_ffrwd-wasm"))
        .args(["-f", "nut", "-i"])
        .arg(source.path())
        .args(["-m"])
        .arg(&module)
        .args([
            "-params",
            r#"{"language":"es","language_to":"en"}"#,
            "-f",
            "nut",
        ])
        .arg(passed.path())
        .args(["-f", "srt"])
        .arg(subs.path())
        .stdin(Stdio::null())
        .output()
        .expect("spawn ffrwd-wasm");
    assert!(
        out.status.success(),
        "ffrwd-wasm exited with {:?}\nstderr:\n{}",
        out.status.code(),
        String::from_utf8_lossy(&out.stderr)
    );

    let document = std::fs::read_to_string(subs.path()).expect("read the subtitle document");
    let written = cues(&document);
    assert!(
        !written.is_empty(),
        "thirty seconds of speech decodes to at least one cue, got:\n{document}"
    );
    assert!(
        !document.contains("transcript"),
        "the trailing record is not a cue, so it is not in the document, got:\n{document}"
    );
    for (index, (start, end, text)) in written.iter().enumerate() {
        assert!(
            start.starts_with("00:00:") && end.starts_with("00:00:"),
            "cue {index} runs {start} to {end}, and the fixture is half a minute"
        );
        assert!(!text.trim().is_empty(), "cue {index} says nothing");
    }
    eprintln!("the sidecar wrote {} cues:\n{document}", written.len());

    // The terminal ffmpeg reads the document as an input and encodes it into
    // the container. The language tag is the compiler's to write; that the
    // plumbing takes one is what this proves.
    let muxed = TempFile::new("speech.mp4");
    let mux = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=64x64:r=25:d=30",
            "-i",
        ])
        .arg(subs.path())
        .args([
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-c:s",
            "mov_text",
            "-metadata:s:s:0",
            "language=eng",
            "-f",
            "mp4",
        ])
        .arg(muxed.path())
        .stdin(Stdio::null())
        .output()
        .expect("spawn ffmpeg to mux the subtitles");
    assert!(
        mux.status.success(),
        "muxing exited with {:?}\nstderr:\n{}",
        mux.status.code(),
        String::from_utf8_lossy(&mux.stderr)
    );

    let (codec, language) = subtitle_stream(muxed.path());
    assert_eq!(codec, "mov_text", "the mp4 carries a subtitle stream");
    assert_eq!(language, "eng", "and the tag ffmpeg was given is on it");

    let back = cues(&extract_subtitles(muxed.path()));
    assert_eq!(
        back.len(),
        written.len(),
        "every cue the sidecar wrote is in the mp4"
    );
    for (index, (was, now)) in written.iter().zip(&back).enumerate() {
        assert_eq!(was.2, now.2, "cue {index} says the same thing in the mp4");
        assert_eq!(was.0, now.0, "cue {index} starts where it started");
        // mov_text shows one cue at a time, so it clips a cue at the next
        // one's start. It never carries one further than the document did.
        assert!(
            now.1 <= was.1,
            "cue {index} ran to {} in the document and to {} in the mp4",
            was.1,
            now.1
        );
    }
}

// The encoded edge: `ffmpeg -c:v libx264 -f nut pipe:` straight into a packet
// sink, nothing decoded in between.

/// One group-of-pictures row as packet_stats reports it.
#[derive(serde::Deserialize)]
struct GopRow {
    gop: u64,
    packets: u64,
    bytes: u64,
}

/// The trailing summary packet_stats ends with.
#[derive(serde::Deserialize)]
struct PacketSummary {
    packets: u64,
    keyframes: u64,
    bytes: u64,
    gops: u64,
}

#[test]
fn a_live_encoder_feeds_the_packet_sink() {
    ensure_modules_built();
    const FRAMES: u64 = 25;
    const GOP: u64 = 10;

    let mut producer = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=64x48:rate=25",
            "-frames:v",
            &FRAMES.to_string(),
            "-c:v",
            "libx264",
            "-g",
            &GOP.to_string(),
            "-pix_fmt",
            "yuv420p",
            "-f",
            "nut",
            "-",
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn encoder ffmpeg");
    let producer_stdout = producer.stdout.take().expect("producer stdout");
    let producer_stderr_handle = drain_stderr(producer.stderr.take().expect("producer stderr"));

    let output = Command::new(env!("CARGO_BIN_EXE_ffrwd-wasm"))
        .args(["-f", "nut", "-i", "-", "-m"])
        .arg(module_path("packet_stats"))
        .args(["-f", "ndjson", "-"])
        .stdin(Stdio::from(producer_stdout))
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .expect("spawn ffrwd-wasm");

    let producer_status = producer.wait().expect("wait for encoder ffmpeg");
    let producer_stderr = producer_stderr_handle.join().expect("join stderr thread");
    assert!(
        producer_status.success(),
        "encoder ffmpeg exited with {:?}\nstderr:\n{producer_stderr}",
        producer_status.code()
    );
    assert!(
        output.status.success(),
        "ffrwd-wasm exited with {:?}\nstderr:\n{}",
        output.status.code(),
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let mut lines: Vec<&str> = stdout.lines().collect();
    let summary: PacketSummary = serde_json::from_str(lines.pop().expect("a trailing summary row"))
        .unwrap_or_else(|e| panic!("parsing the summary: {e}\nstdout:\n{stdout}"));
    let gops: Vec<GopRow> = lines
        .iter()
        .map(|line| {
            serde_json::from_str(line).unwrap_or_else(|e| panic!("parsing group row {line:?}: {e}"))
        })
        .collect();

    // Every frame encodes to one packet; a group is capped at GOP frames, so
    // at least ceil(FRAMES / GOP) keyframes open that many groups - and the
    // encoder's first packet is a keyframe, so groups and keyframes agree.
    assert_eq!(summary.packets, FRAMES);
    assert!(
        summary.keyframes >= FRAMES.div_ceil(GOP),
        "-g {GOP} caps a group, so {FRAMES} frames need at least {} keyframes, got {}",
        FRAMES.div_ceil(GOP),
        summary.keyframes
    );
    assert_eq!(summary.gops, summary.keyframes);
    assert_eq!(gops.len() as u64, summary.gops);
    assert_eq!(gops.iter().map(|g| g.packets).sum::<u64>(), summary.packets);
    assert_eq!(gops.iter().map(|g| g.bytes).sum::<u64>(), summary.bytes);
    for (index, group) in gops.iter().enumerate() {
        assert_eq!(group.gop, index as u64, "group rows arrive in order");
        assert!(
            group.packets <= GOP,
            "group {index} holds {} packets, more than -g {GOP} allows",
            group.packets
        );
    }
}
