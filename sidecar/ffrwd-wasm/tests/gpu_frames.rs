//! Frames held on the GPU: a node module beside `ffrwd:gpu` reads its
//! input as textures and hands textures back, and the sidecar carries them
//! to the wire. The GPU path is checked against the same module's host
//! path byte for byte, on a machine with an adapter; where there is none,
//! the GPU runs are skipped and say so, and the host path still runs.

use std::path::PathBuf;
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
    let module = module();
    let wiring = format!("[v=0:v]gpu_box=memory={memory}:size={SIDE}[out=o]");
    let mut child = Command::new(env!("CARGO_BIN_EXE_ffrwd-wasm"))
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
