//! Frames held on the GPU: a node module beside `ffrwd:gpu` reads its
//! input as textures and hands textures back, and the sidecar carries them
//! to the wire. The GPU path is checked against the same module's host
//! path byte for byte, on a machine with an adapter; where there is none,
//! the GPU runs are skipped and say so, and the host path still runs.

use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::sync::OnceLock;

use ffrwd_wasm::nut::{Demuxer, Muxer, Stream, TimeBase};
use std::io::Write;

const WIDTH: u32 = 8;
const HEIGHT: u32 = 8;
const FRAME_LEN: usize = (WIDTH * HEIGHT * 4) as usize;
const TIME_BASE: TimeBase = TimeBase { num: 1, den: 25 };
/// The box's side, under the frame's so the corner is checkable.
const SIDE: usize = 3;

fn sidecar_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("ffrwd-wasm/ has a parent directory")
        .to_path_buf()
}

/// The gpu-box module's path, built once per test binary.
fn module() -> String {
    static BUILT: OnceLock<()> = OnceLock::new();
    BUILT.get_or_init(|| {
        let output = Command::new("cargo")
            .args([
                "build",
                "--release",
                "--target",
                "wasm32-wasip2",
                "-p",
                "gpu-box",
            ])
            .current_dir(sidecar_root().join("modules"))
            .output()
            .expect("spawn cargo build");
        assert!(
            output.status.success(),
            "building gpu-box failed:\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
    });
    sidecar_root()
        .join("modules/target/wasm32-wasip2/release/gpu_box.wasm")
        .to_str()
        .expect("a UTF-8 path")
        .to_string()
}

/// Frame `index`: a byte pattern distinct per position and per frame.
fn picture(index: u8) -> Vec<u8> {
    (0..FRAME_LEN)
        .map(|offset| index.wrapping_mul(41).wrapping_add(offset as u8))
        .collect()
}

fn nut_stream(frames: &[Vec<u8>]) -> Vec<u8> {
    let stream = Stream::video("rgba", WIDTH, HEIGHT, TIME_BASE).expect("rgba is carried");
    let mut wire = Vec::new();
    {
        let mut muxer = Muxer::new(&mut wire, &stream).expect("write NUT headers");
        for (i, frame) in frames.iter().enumerate() {
            muxer.write_frame(i as i64, frame).expect("write a frame");
        }
        muxer.finish().expect("finish the stream");
    }
    wire
}

fn read_nut(wire: &[u8]) -> Vec<(i64, Vec<u8>)> {
    let mut demuxer = Demuxer::open(wire).expect("read NUT headers");
    let mut frames = Vec::new();
    let mut buf = Vec::new();
    while let Some(pts) = demuxer.read_frame(&mut buf).expect("read a frame") {
        frames.push((pts, buf.clone()));
    }
    frames
}

/// `picture(index)` with the box drawn on it.
fn boxed(index: u8) -> Vec<u8> {
    let mut pixels = picture(index);
    for y in 0..SIDE {
        for x in 0..SIDE {
            let at = (y * WIDTH as usize + x) * 4;
            pixels[at..at + 4].copy_from_slice(&[255, 0, 0, 255]);
        }
    }
    pixels
}

/// Runs gpu_box over `frames` with `memory` as its params say, at `jobs`.
fn run(memory: &str, jobs: &str, frames: &[Vec<u8>]) -> Output {
    run_with(memory, jobs, frames, &[])
}

/// `run` with `extra` ahead of the inputs.
fn run_with(memory: &str, jobs: &str, frames: &[Vec<u8>], extra: &[&str]) -> Output {
    let module = module();
    let wiring = format!("[v=0:v]gpu_box=memory={memory}:size={SIDE}[out=o]");
    let mut child = Command::new(env!("CARGO_BIN_EXE_ffrwd-wasm"))
        .args(extra)
        .args(["-gpu", &module, "-jobs", jobs, "-f", "nut", "-i", "-", "-m"])
        .arg(format!("gpu_box={module}"))
        .args(["-filter_complex", &wiring, "-map", "[o]", "-f", "nut", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn ffrwd-wasm");
    let wire = nut_stream(frames);
    let mut stdin = child.stdin.take().expect("stdin is piped");
    std::thread::spawn(move || {
        let _ = stdin.write_all(&wire);
    });
    child.wait_with_output().expect("wait for ffrwd-wasm")
}

/// True when `output` is the refusal of a machine with no GPU, which these
/// tests skip rather than fail: the path is the owner's machine's to prove.
fn no_gpu(output: &Output) -> bool {
    let stderr = String::from_utf8_lossy(&output.stderr);
    !output.status.success() && (stderr.contains("no GPU") || stderr.contains("no adapter"))
}

fn frames_of(output: &Output, what: &str) -> Vec<(i64, Vec<u8>)> {
    assert!(
        output.status.success(),
        "{what} exited {:?}:\n{}",
        output.status.code(),
        String::from_utf8_lossy(&output.stderr)
    );
    read_nut(&output.stdout)
}

#[test]
fn the_shape_says_which_ports_are_held_on_the_gpu() {
    let module = module();
    for (memory, held) in [("gpu", "gpu"), ("host", "host")] {
        let output = Command::new(env!("CARGO_BIN_EXE_ffrwd-wasm"))
            .args([
                "-gpu", &module, "--shape", &module, "--bound", "v", "--params",
            ])
            .arg(format!(r#"{{"memory":"{memory}"}}"#))
            .output()
            .expect("spawn ffrwd-wasm");
        assert!(
            output.status.success(),
            "--shape with memory={memory} exited {:?}:\n{}",
            output.status.code(),
            String::from_utf8_lossy(&output.stderr)
        );
        let shape: serde_json::Value =
            serde_json::from_slice(&output.stdout).expect("one JSON line");
        assert_eq!(
            shape["inputs"][0]["accepts"]["memory"], held,
            "memory={memory}: {shape}"
        );
        assert_eq!(
            shape["outputs"][0]["memory"], held,
            "memory={memory}: {shape}"
        );
    }
}

#[test]
fn frames_in_host_memory_come_back_with_the_box_drawn() {
    let frames: Vec<Vec<u8>> = (0..3u8).map(picture).collect();
    let got = frames_of(&run("host", "1", &frames), "gpu_box in host memory");
    assert_eq!(got.len(), 3);
    for (index, (pts, data)) in got.iter().enumerate() {
        assert_eq!(*pts, index as i64);
        assert_eq!(*data, boxed(index as u8), "frame {index}");
    }
}

#[test]
fn a_profiled_run_counts_each_stage_the_frames_crossed() {
    let frames: Vec<Vec<u8>> = (0..3u8).map(picture).collect();
    let output = run_with("host", "1", &frames, &["-profile"]);
    let got = frames_of(&output, "gpu_box in host memory with -profile");
    assert_eq!(got.len(), 3);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let row: serde_json::Value = stderr
        .lines()
        .filter_map(|line| line.strip_prefix("ffrwd:row "))
        .map(|line| serde_json::from_str::<serde_json::Value>(line).expect("a row is JSON"))
        .find(|row| row["kind"] == "profile")
        .unwrap_or_else(|| panic!("no profile row on stderr:\n{stderr}"));
    let total = (3 * FRAME_LEN) as u64;
    for stage in ["held", "fetch", "emit", "write"] {
        assert_eq!(row[stage]["bytes"], total, "{stage} in {row}");
        assert_eq!(row[stage]["calls"], 3, "{stage} in {row}");
    }
    assert!(
        row["read"]["bytes"]
            .as_u64()
            .is_some_and(|read| read > total),
        "the wire carries the frames and their headers: {row}"
    );
    assert!(
        row["process"]["calls"]
            .as_u64()
            .is_some_and(|calls| calls >= 3),
        "one process call per tick at least: {row}"
    );
    for stage in ["upload", "download"] {
        assert_eq!(
            row[stage]["bytes"], 0,
            "nothing on the GPU in host memory: {row}"
        );
    }
}

#[test]
fn frames_held_on_the_gpu_come_back_with_the_same_box() {
    let frames: Vec<Vec<u8>> = (0..3u8).map(picture).collect();
    let on_gpu = run("gpu", "1", &frames);
    if no_gpu(&on_gpu) {
        eprintln!("skipped: this machine has no GPU adapter");
        return;
    }
    let got = frames_of(&on_gpu, "gpu_box on the GPU");
    let host = frames_of(&run("host", "1", &frames), "gpu_box in host memory");
    assert_eq!(got, host, "the GPU path writes what the host path writes");
    let again = frames_of(&run("gpu", "4", &frames), "gpu_box on the GPU at -jobs 4");
    assert_eq!(got, again, "-jobs 1 and -jobs 4 write the same bytes");
}

// A decoder on the process's own device: `decode` with `hw=d3d12va`, read
// by an input held on the GPU, leaves its pictures there, and gpu_box reads
// them as textures with nothing uploaded at its lane.

/// The h264 fixture: 64x48, B-frames.
fn h264_fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/h264.nut")
}

/// The libav directory `FFRWD_AV_RUNTIME` names, where it holds libav;
/// None, having said why the test is skipped, otherwise or off Windows.
fn d3d12va_libav(test: &str) -> Option<PathBuf> {
    if !cfg!(windows) {
        eprintln!(
            "SKIPPED: {test}. A decoder shares the process's device on Windows (D3D12) only."
        );
        return None;
    }
    let dir = std::env::var_os("FFRWD_AV_RUNTIME").map(PathBuf::from);
    match dir.filter(|d| d.join("avcodec-63.dll").is_file()) {
        Some(dir) => Some(dir),
        None => {
            eprintln!(
                "SKIPPED: {test}. Point FFRWD_AV_RUNTIME at a directory holding FFmpeg 9.0's \
                 shared libraries to run it."
            );
            None
        }
    }
}

/// `chain` over the h264 fixture, with gpu_box bound, each `(label, path)`
/// mapped to a NUT file.
fn run_decoded(dir: &Path, chain: &str, maps: &[(&str, &Path)]) -> Output {
    let module = module();
    let mut command = Command::new(env!("CARGO_BIN_EXE_ffrwd-wasm"));
    command
        .env("FFRWD_AV_RUNTIME", dir)
        .args(["-gpu", &module, "-f", "nut", "-i"])
        .arg(h264_fixture())
        .arg("-m")
        .arg(format!("gpu_box={module}"))
        .args(["-filter_complex", chain]);
    for (label, path) in maps {
        command
            .args(["-map", &format!("[{label}]"), "-f", "nut"])
            .arg(path);
    }
    command
        .stdin(Stdio::null())
        .output()
        .expect("spawn ffrwd-wasm")
}

fn scratch(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("ffrwd_gpu_{}_{name}", std::process::id()))
}

fn frames_in(path: &Path) -> Vec<(i64, Vec<u8>)> {
    let wire = std::fs::read(path).expect("read the output");
    let _ = std::fs::remove_file(path);
    read_nut(&wire)
}

const DECODED_LINE: &str = "[gpu] decode h264: d3d12va, frames ";
const UPLOAD_LINE: &str = "putting the frames of stream";

#[test]
fn pictures_decoded_on_the_device_reach_gpu_box_unuploaded_and_as_software_decodes_them() {
    let test =
        "pictures_decoded_on_the_device_reach_gpu_box_unuploaded_and_as_software_decodes_them";
    let Some(dir) = d3d12va_libav(test) else {
        return;
    };
    let chain = |hw: &str, memory: &str| {
        format!("[0:v]decode=format=yuv420p:hw={hw}[d];[v=d]gpu_box=memory={memory}:out=host:size=5[out=o]")
    };
    let on_device = scratch("on_device.nut");
    let output = run_decoded(&dir, &chain("d3d12va", "gpu"), &[("o", &on_device)]);
    if no_gpu(&output) {
        eprintln!("skipped: this machine has no GPU adapter");
        return;
    }
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    assert!(
        output.status.success(),
        "decode on the device exited {:?}:\n{stderr}",
        output.status.code()
    );
    assert!(
        stderr.contains(&format!("{DECODED_LINE}wrapped"))
            || stderr.contains(&format!("{DECODED_LINE}copied")),
        "the decoder says how its pictures are taken:\n{stderr}"
    );
    assert!(
        !stderr.contains(UPLOAD_LINE),
        "nothing is uploaded at the lane:\n{stderr}"
    );
    let got = frames_in(&on_device);

    let uploaded = scratch("uploaded.nut");
    let output = run_decoded(&dir, &chain("none", "gpu"), &[("o", &uploaded)]);
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    assert!(
        output.status.success(),
        "software decode exited {:?}:\n{stderr}",
        output.status.code()
    );
    assert!(
        stderr.contains(UPLOAD_LINE),
        "a software decoder's pictures are uploaded at the lane:\n{stderr}"
    );
    let host = scratch("host.nut");
    let output = run_decoded(&dir, &chain("none", "host"), &[("o", &host)]);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );

    let software = frames_in(&host);
    assert_eq!(software.len(), got.len(), "as many pictures either way");
    assert!(!got.is_empty());
    assert_eq!(
        frames_in(&uploaded),
        software,
        "the uploaded path writes what the host path writes"
    );
    for (index, (a, b)) in got.iter().zip(&software).enumerate() {
        assert_eq!(a.0, b.0, "picture {index}'s pts");
        assert!(
            a.1 == b.1,
            "picture {index}: the device's decode is the software decode, byte for byte"
        );
    }
}

#[test]
fn encode_reads_a_picture_decoded_on_the_device_back_once_as_ffmpeg_decodes_it() {
    let test = "encode_reads_a_picture_decoded_on_the_device_back_once_as_ffmpeg_decodes_it";
    let Some(dir) = d3d12va_libav(test) else {
        return;
    };
    if Command::new("ffmpeg")
        .arg("-version")
        .stdin(Stdio::null())
        .output()
        .is_err()
    {
        eprintln!("SKIPPED: {test}. Put ffmpeg on PATH to run it.");
        return;
    }
    // gpu_box holds [d] on the GPU, so the decoder leaves its pictures
    // there; encode reads the same frames and reads each back.
    let chain = "[0:v]decode=format=yuv420p:hw=d3d12va[d];\
                 [v=d]gpu_box=memory=gpu:out=host:size=5[out=o];[d]encode=codec=ffv1[e]";
    let (boxed, encoded) = (scratch("boxed.nut"), scratch("encoded.nut"));
    let output = run_decoded(&dir, chain, &[("o", &boxed), ("e", &encoded)]);
    if no_gpu(&output) {
        eprintln!("skipped: this machine has no GPU adapter");
        return;
    }
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    assert!(
        output.status.success(),
        "exited {:?}:\n{stderr}",
        output.status.code()
    );
    assert!(
        stderr.contains(DECODED_LINE),
        "the decoder ran on the device:\n{stderr}"
    );
    let _ = std::fs::remove_file(&boxed);
    let rawvideo = |path: &Path| {
        let output = Command::new("ffmpeg")
            .args(["-v", "error", "-i"])
            .arg(path)
            .args([
                "-fps_mode",
                "passthrough",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "yuv420p",
                "-",
            ])
            .stdin(Stdio::null())
            .output()
            .expect("spawn ffmpeg");
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        output.stdout
    };
    let through_encode = rawvideo(&encoded);
    let _ = std::fs::remove_file(&encoded);
    let decoded = rawvideo(&h264_fixture());
    assert!(!decoded.is_empty());
    assert!(
        through_encode == decoded,
        "ffv1 of the pictures read back is ffmpeg's own decode, byte for byte ({} and {} bytes)",
        through_encode.len(),
        decoded.len()
    );
}
