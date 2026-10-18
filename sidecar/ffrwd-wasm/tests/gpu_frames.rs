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
    run_in(memory, jobs, frames, extra, None)
}

/// The environment variable naming the adapter when the argv does not.
const ADAPTER_VAR: &str = "FFRWD_GPU_ADAPTER";

/// `run_with`, with [`ADAPTER_VAR`] set to `adapter` or, for None, unset.
fn run_in(
    memory: &str,
    jobs: &str,
    frames: &[Vec<u8>],
    extra: &[&str],
    adapter: Option<&str>,
) -> Output {
    let module = module();
    let wiring = format!("[v=0:v]gpu_box=memory={memory}:size={SIDE}[out=o]");
    let mut command = Command::new(env!("CARGO_BIN_EXE_ffrwd-wasm"));
    match adapter {
        Some(spec) => command.env(ADAPTER_VAR, spec),
        None => command.env_remove(ADAPTER_VAR),
    };
    let mut child = command
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

/// Set to anything but blank where a GPU is promised (CI's software Vulkan
/// driver): a run that finds no adapter then fails rather than skips.
const GPU_REQUIRED: &str = "FFRWD_GPU_REQUIRED";

/// True when `output` is the refusal of a machine with no GPU, which these
/// tests skip rather than fail, unless [`GPU_REQUIRED`] says one is there.
fn no_gpu(output: &Output) -> bool {
    let stderr = String::from_utf8_lossy(&output.stderr);
    let missing =
        !output.status.success() && (stderr.contains("no GPU") || stderr.contains("no adapter"));
    let required = std::env::var_os(GPU_REQUIRED).is_some_and(|v| !v.is_empty());
    assert!(
        !(missing && required),
        "{GPU_REQUIRED} is set, and the run found no GPU adapter:\n{stderr}"
    );
    missing
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

/// What `--gpu-info` lists on this machine, None where it lists nothing
/// (and [`GPU_REQUIRED`] does not say there is something).
fn adapters() -> Option<Vec<serde_json::Value>> {
    let output = Command::new(env!("CARGO_BIN_EXE_ffrwd-wasm"))
        .arg("--gpu-info")
        .env_remove(ADAPTER_VAR)
        .output()
        .expect("spawn ffrwd-wasm --gpu-info");
    assert!(
        output.status.success(),
        "--gpu-info exited {:?}:\n{}",
        output.status.code(),
        String::from_utf8_lossy(&output.stderr)
    );
    let info: serde_json::Value = serde_json::from_slice(&output.stdout).expect("one JSON line");
    let listed = info["adapters"]
        .as_array()
        .expect("a list of adapters")
        .clone();
    let required = std::env::var_os(GPU_REQUIRED).is_some_and(|v| !v.is_empty());
    assert!(
        !(listed.is_empty() && required),
        "{GPU_REQUIRED} is set, and --gpu-info lists no adapter"
    );
    (!listed.is_empty()).then_some(listed)
}

/// The adapter a run's `[gpu] device:` line names, as `<name> (<Backend>)`.
fn device_line(output: &Output) -> String {
    let stderr = String::from_utf8_lossy(&output.stderr);
    stderr
        .lines()
        .find_map(|line| line.strip_prefix("[gpu] device: "))
        .unwrap_or_else(|| panic!("no [gpu] device: line on stderr:\n{stderr}"))
        .to_string()
}

/// How the `[gpu] device:` line names `adapter`: wgpu's Debug spelling of
/// its backend.
fn named_as(adapter: &serde_json::Value) -> String {
    let backend = match adapter["backend"].as_str().expect("a backend") {
        "dx12" => "Dx12",
        "vulkan" => "Vulkan",
        "metal" => "Metal",
        "gl" => "Gl",
        other => panic!("a backend wgpu does not name: {other}"),
    };
    format!("{} ({backend})", adapter["name"].as_str().expect("a name"))
}

#[test]
fn gpu_info_lists_each_adapter_and_marks_the_one_opened_by_default() {
    let Some(listed) = adapters() else {
        eprintln!("skipped: this machine has no GPU adapter");
        return;
    };
    let rank = |kind: &str| match kind {
        "discrete" => 1,
        "integrated" => 2,
        "other" => 3,
        "virtual" => 4,
        "software" => 5,
        other => panic!("a type --gpu-info does not spell: {other}"),
    };
    for (index, adapter) in listed.iter().enumerate() {
        assert_eq!(adapter["index"], index, "{adapter}");
        assert!(
            adapter["vendor"].is_u64() && adapter["device"].is_u64(),
            "{adapter}"
        );
        rank(adapter["type"].as_str().expect("a type"));
    }
    let defaults: Vec<&serde_json::Value> =
        listed.iter().filter(|a| a["default"] == true).collect();
    let [default] = defaults[..] else {
        panic!("one adapter is the default: {listed:?}");
    };
    // The best kind the first backend has, the first of it where several are.
    let first = &listed[0]["backend"];
    let best = listed
        .iter()
        .filter(|a| &a["backend"] == first)
        .min_by_key(|a| rank(a["type"].as_str().unwrap()))
        .unwrap();
    assert_eq!(default, best, "{listed:?}");

    let frames: Vec<Vec<u8>> = (0..2u8).map(picture).collect();
    let output = run("gpu", "1", &frames);
    assert_eq!(device_line(&output), named_as(default));
}

#[test]
fn an_adapter_named_by_index_or_by_name_is_the_one_opened() {
    let Some(listed) = adapters() else {
        eprintln!("skipped: this machine has no GPU adapter");
        return;
    };
    let frames: Vec<Vec<u8>> = (0..3u8).map(picture).collect();
    let host = frames_of(&run("host", "1", &frames), "gpu_box in host memory");
    for adapter in &listed {
        let index = adapter["index"].to_string();
        let output = run_with("gpu", "1", &frames, &["-gpu-adapter", &index]);
        assert_eq!(
            device_line(&output),
            named_as(adapter),
            "-gpu-adapter {index}"
        );
        assert_eq!(frames_of(&output, &named_as(adapter)), host);
    }
    // A name in another case names the first backend's adapter of that name.
    let first = &listed[0];
    let name = first["name"].as_str().unwrap().to_uppercase();
    let alone = listed
        .iter()
        .filter(|a| a["backend"] == first["backend"] && a["name"] == first["name"])
        .count()
        == 1;
    if alone {
        let output = run_with("gpu", "1", &frames, &["-gpu-adapter", &name]);
        assert_eq!(device_line(&output), named_as(first), "-gpu-adapter {name}");
        let output = run_in("gpu", "1", &frames, &[], Some(&name));
        assert_eq!(
            device_line(&output),
            named_as(first),
            "{ADAPTER_VAR}={name}"
        );
    }
}

#[test]
fn a_software_adapter_named_runs_the_module_on_itself() {
    let Some(listed) = adapters() else {
        eprintln!("skipped: this machine has no GPU adapter");
        return;
    };
    let Some(software) = listed.iter().find(|a| a["type"] == "software") else {
        eprintln!("skipped: this machine lists no software adapter");
        return;
    };
    let name = software["name"].as_str().unwrap();
    let frames: Vec<Vec<u8>> = (0..3u8).map(picture).collect();
    let output = run_with("gpu", "1", &frames, &["-gpu-adapter", name]);
    assert_eq!(device_line(&output), named_as(software));
    let got = frames_of(&output, name);
    assert_eq!(got.len(), 3);
    for (index, (_, data)) in got.iter().enumerate() {
        assert_eq!(*data, boxed(index as u8), "frame {index} on {name}");
    }
}

#[test]
fn an_adapter_spec_naming_none_ends_the_run_rather_than_open_another() {
    if adapters().is_none() {
        eprintln!("skipped: this machine has no GPU adapter");
        return;
    }
    let frames: Vec<Vec<u8>> = (0..2u8).map(picture).collect();
    let hint = "`ffrwd-wasm --gpu-info` lists them";
    for (output, said) in [
        (
            run_with("gpu", "1", &frames, &["-gpu-adapter", "no such card"]),
            "ffrwd-wasm: -gpu-adapter no such card: no adapter's name contains 'no such card'; \
             the adapters are 0 ",
        ),
        (
            run_with("gpu", "1", &frames, &["-gpu-adapter", "987654321"]),
            "ffrwd-wasm: -gpu-adapter 987654321: there is no adapter 987654321, they are \
             numbered from 0 to ",
        ),
        (
            run_in("gpu", "1", &frames, &[], Some("no such card")),
            "ffrwd-wasm: FFRWD_GPU_ADAPTER=no such card: no adapter's name contains",
        ),
    ] {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert_eq!(output.status.code(), Some(2), "{stderr}");
        assert!(stderr.contains(said) && stderr.contains(hint), "{stderr}");
        assert!(!stderr.contains("[gpu] device:"), "{stderr}");
        assert!(output.stdout.is_empty(), "nothing is written");
    }
    // The flag replaces the variable.
    let output = run_in(
        "gpu",
        "1",
        &frames,
        &["-gpu-adapter", "0"],
        Some("no such card"),
    );
    assert_eq!(frames_of(&output, "-gpu-adapter 0").len(), 2);
}

// A decoder on the process's own device: `decode` with `hw=d3d12va` on
// Windows or `hw=videotoolbox` on macOS, read by an input held on the GPU,
// leaves its pictures there, and gpu_box reads them as textures with
// nothing uploaded at its lane.

/// The h264 fixture: 64x48, B-frames. On macOS 96x64, since VideoToolbox
/// decodes nothing as small as 64x48 and libavcodec falls back to software.
fn h264_fixture() -> PathBuf {
    let name = if cfg!(target_os = "macos") {
        "h264-96x64.nut"
    } else {
        "h264.nut"
    };
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name)
}

/// The h264 fixture's size.
fn h264_fixture_size() -> (u32, u32) {
    if cfg!(target_os = "macos") {
        (96, 64)
    } else {
        (64, 48)
    }
}

/// The libav directory `FFRWD_AV_RUNTIME` names, where it holds libav;
/// None, having said why the test is skipped, otherwise or off Windows.
fn d3d12va_libav(test: &str) -> Option<PathBuf> {
    if !cfg!(windows) {
        eprintln!("SKIPPED: {test}. The D3D12 encoders run on Windows only.");
        return None;
    }
    libav(test)
}

/// The libav directory `FFRWD_AV_RUNTIME` names, where it holds libav;
/// None, having said why the test is skipped, otherwise.
fn libav(test: &str) -> Option<PathBuf> {
    let library = if cfg!(windows) {
        "avcodec-63.dll"
    } else if cfg!(target_os = "macos") {
        "libavcodec.63.dylib"
    } else {
        "libavcodec.so.63"
    };
    let dir = std::env::var_os("FFRWD_AV_RUNTIME").map(PathBuf::from);
    match dir.filter(|d| d.join(library).is_file()) {
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

const UPLOAD_LINE: &str = "putting the frames of stream";

/// The device a decoder shares the process's device through, and the
/// libav directory: `d3d12va` on Windows, `videotoolbox` on macOS, `vulkan`
/// on Linux. None, having said why the test is skipped, elsewhere.
fn shared_decoder(test: &str) -> Option<(&'static str, PathBuf)> {
    let device = if cfg!(windows) {
        "d3d12va"
    } else if cfg!(target_os = "macos") {
        "videotoolbox"
    } else if cfg!(target_os = "linux") {
        "vulkan"
    } else {
        eprintln!(
            "SKIPPED: {test}. A decoder shares the process's device on Windows (D3D12), \
             macOS (Metal) and Linux (Vulkan) only."
        );
        return None;
    };
    Some((device, libav(test)?))
}

/// The device a decoder shares the process's device through where the
/// 64x48 fixture decodes on it, and the libav directory: `d3d12va` on
/// Windows, `vulkan` on Linux. None, having said why the test is skipped,
/// elsewhere: VideoToolbox decodes nothing that small.
fn small_shared_decoder(test: &str) -> Option<(&'static str, PathBuf)> {
    if !cfg!(windows) && !cfg!(target_os = "linux") {
        eprintln!(
            "SKIPPED: {test}. Its 64x48 fixture decodes on the process's device on Windows \
             and Linux only."
        );
        return None;
    }
    shared_decoder(test)
}

/// What the sidecar says on Linux where the process's Vulkan device has no
/// decoder: a software adapter, or a card without the video extensions.
const NO_DEVICE_DECODER: &str = "[gpu] no decoder or encoder on the process's Vulkan device";

/// True, having said why the test is skipped, where `output` is a run whose
/// device has no decoder on it to share; what these tests skip rather than
/// fail, as they skip a machine with no adapter.
fn no_device_decoder(test: &str, output: &Output) -> bool {
    let stderr = String::from_utf8_lossy(&output.stderr);
    let Some(line) = stderr.lines().find(|l| l.starts_with(NO_DEVICE_DECODER)) else {
        return false;
    };
    eprintln!("SKIPPED: {test}. {line}");
    true
}

/// What the decoder says of how its pictures are taken on `device`.
fn decoded_line(device: &str) -> String {
    format!("[gpu] decode h264: {device}, frames ")
}

/// The device a decoder shares the process's device through, the h264
/// encoder that reads its pool there, and the libav directory:
/// `h264_d3d12va` on Windows, `h264_videotoolbox` on macOS, `h264_nvenc` on
/// Linux, fed from the Vulkan device through CUDA.
fn shared_encoder(test: &str) -> Option<(&'static str, &'static str, PathBuf)> {
    let (device, dir) = shared_decoder(test)?;
    let encoder = if cfg!(windows) {
        "h264_d3d12va"
    } else if cfg!(target_os = "linux") {
        "h264_nvenc"
    } else {
        "h264_videotoolbox"
    };
    Some((device, encoder, dir))
}

#[test]
fn pictures_decoded_on_the_device_reach_gpu_box_unuploaded_and_as_software_decodes_them() {
    let test =
        "pictures_decoded_on_the_device_reach_gpu_box_unuploaded_and_as_software_decodes_them";
    let Some((device, dir)) = shared_decoder(test) else {
        return;
    };
    let chain = |hw: &str, memory: &str| {
        format!("[0:v]decode=format=yuv420p:hw={hw}[d];[v=d]gpu_box=memory={memory}:out=host:size=5[out=o]")
    };
    let on_device = scratch("on_device.nut");
    let output = run_decoded(&dir, &chain(device, "gpu"), &[("o", &on_device)]);
    if no_gpu(&output) {
        eprintln!("skipped: this machine has no GPU adapter");
        return;
    }
    if no_device_decoder(test, &output) {
        return;
    }
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    assert!(
        output.status.success(),
        "decode on the device exited {:?}:\n{stderr}",
        output.status.code()
    );
    let decoded = decoded_line(device);
    assert!(
        stderr.contains(&format!("{decoded}wrapped"))
            || stderr.contains(&format!("{decoded}copied")),
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
    let Some((device, dir)) = shared_decoder(test) else {
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
    let chain = format!(
        "[0:v]decode=format=yuv420p:hw={device}[d];\
         [v=d]gpu_box=memory=gpu:out=host:size=5[out=o];[d]encode=codec=ffv1[e]"
    );
    let (boxed, encoded) = (scratch("boxed.nut"), scratch("encoded.nut"));
    let output = run_decoded(&dir, &chain, &[("o", &boxed), ("e", &encoded)]);
    if no_gpu(&output) {
        eprintln!("skipped: this machine has no GPU adapter");
        return;
    }
    if no_device_decoder(test, &output) {
        return;
    }
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    assert!(
        output.status.success(),
        "exited {:?}:\n{stderr}",
        output.status.code()
    );
    assert!(
        stderr.contains(&decoded_line(device)),
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

// An encoder on the process's own device: `h264_d3d12va` reads pictures of
// a pool on the device the decoder and the modules use, so a frame there
// reaches it by one copy on the device and no byte crosses to the host.

/// ffmpeg on PATH; None, having said the test is skipped, otherwise.
fn ffmpeg(test: &str) -> Option<()> {
    let found = Command::new("ffmpeg")
        .arg("-version")
        .stdin(Stdio::null())
        .output()
        .is_ok_and(|o| o.status.success());
    if !found {
        eprintln!("SKIPPED: {test}. Put ffmpeg and ffprobe on PATH to run it.");
        return None;
    }
    Some(())
}

/// Runs ffmpeg with `args`, failing the test with its stderr if it fails.
fn ffmpeg_ok(args: &[&str], what: &str) -> Vec<u8> {
    let output = Command::new("ffmpeg")
        .args(["-v", "error", "-y"])
        .args(args)
        .stdin(Stdio::null())
        .output()
        .expect("spawn ffmpeg");
    assert!(
        output.status.success(),
        "{what}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    output.stdout
}

fn path_str(path: &Path) -> &str {
    path.to_str().expect("a UTF-8 path")
}

/// `chain` over the NUT file `input`, its `[out0]` written to `out`, with
/// `-profile`; gpu_box bound when `module` is given.
fn run_chain(dir: &Path, input: &Path, chain: &str, out: &Path, module: Option<&str>) -> Output {
    chain_command(dir, input, chain, out, module)
        .output()
        .expect("spawn ffrwd-wasm")
}

/// The command [`run_chain`] runs.
fn chain_command(
    dir: &Path,
    input: &Path,
    chain: &str,
    out: &Path,
    module: Option<&str>,
) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_ffrwd-wasm"));
    command.env("FFRWD_AV_RUNTIME", dir).arg("-profile");
    if let Some(module) = module {
        command.args(["-gpu", module]);
    }
    command.args(["-f", "nut", "-i"]).arg(input);
    if let Some(module) = module {
        command.arg("-m").arg(format!("gpu_box={module}"));
    }
    command
        .args(["-filter_complex", chain, "-map", "[out0]", "-f", "nut"])
        .arg(out)
        .stdin(Stdio::null());
    command
}

/// What the D3D12 debug layer's errors start with on stderr, where wgpu's
/// validation turns the layer on.
const DEBUG_LAYER_ERROR: &str = "[gpu] D3D12 debug layer: ";

/// The profile row a `-profile` run wrote on stderr.
fn profile_row(stderr: &str) -> serde_json::Value {
    stderr
        .lines()
        .filter_map(|line| line.strip_prefix("ffrwd:row "))
        .map(|line| serde_json::from_str::<serde_json::Value>(line).expect("a row is JSON"))
        .find(|row| row["kind"] == "profile")
        .unwrap_or_else(|| panic!("no profile row on stderr:\n{stderr}"))
}

/// What ffmpeg decodes `path` to, as `pix_fmt` pictures one after another.
fn decoded_by_ffmpeg(path: &Path, pix_fmt: &str) -> Vec<u8> {
    ffmpeg_ok(
        &[
            "-i",
            path_str(path),
            "-fps_mode",
            "passthrough",
            "-f",
            "rawvideo",
            "-pix_fmt",
            pix_fmt,
            "-",
        ],
        "decoding with ffmpeg",
    )
}

/// The mean of each of a yuv420p picture's three planes.
fn plane_means(picture: &[u8], width: usize, height: usize) -> [f64; 3] {
    let luma = width * height;
    let chroma = luma / 4;
    let mean = |plane: &[u8]| plane.iter().map(|&b| f64::from(b)).sum::<f64>() / plane.len() as f64;
    [
        mean(&picture[..luma]),
        mean(&picture[luma..luma + chroma]),
        mean(&picture[luma + chroma..luma + 2 * chroma]),
    ]
}

const SIZE: (usize, usize) = (320, 240);
/// What `encoder` says when it is opened on the device for 320x240.
fn from_the_device(encoder: &str) -> String {
    format!("[av] encode {encoder}: 320x240 nv12, from the device")
}

#[test]
fn an_encoder_on_the_device_takes_the_decoders_pictures_with_nothing_read_back() {
    let test = "an_encoder_on_the_device_takes_the_decoders_pictures_with_nothing_read_back";
    let Some((device, encoder, dir)) = shared_encoder(test) else {
        return;
    };
    let Some(()) = ffmpeg(test) else {
        return;
    };
    let source = scratch("d3d12va_source.nut");
    let made = Command::new("ffmpeg")
        .args([
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=s=320x240:r=25",
        ])
        .args([
            "-frames:v",
            "60",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-f",
            "nut",
        ])
        .arg(&source)
        .stdin(Stdio::null())
        .output()
        .expect("spawn ffmpeg");
    if !made.status.success() {
        eprintln!(
            "SKIPPED: {test}. The ffmpeg on PATH cannot make an h264 stream with libx264: {}",
            String::from_utf8_lossy(&made.stderr)
        );
        return;
    }
    let out = scratch("d3d12va_out.nut");
    let chain =
        format!("[0:v]decode=format=yuv420p:hw={device}[d];[d]encode=codec={encoder}:b=2M[out0]");
    let output = run_chain(&dir, &source, &chain, &out, None);
    if no_gpu(&output) {
        eprintln!("skipped: this machine has no GPU adapter");
        return;
    }
    if no_device_decoder(test, &output) {
        return;
    }
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    assert!(
        output.status.success(),
        "exited {:?}:\n{stderr}",
        output.status.code()
    );
    assert!(stderr.contains(&from_the_device(encoder)), "{stderr}");
    assert!(
        stderr.contains(&format!(
            "[gpu] encode {encoder}: from the device, each frame "
        )),
        "the copy is said:\n{stderr}"
    );
    assert!(stderr.contains(&decoded_line(device)), "{stderr}");
    let row = profile_row(&stderr);
    for stage in ["upload", "download"] {
        assert_eq!(row[stage]["bytes"], 0, "nothing crosses to the host: {row}");
    }

    let wire = std::fs::read(&out).expect("read the output");
    let mut demuxer = Demuxer::open(wire.as_slice()).expect("read NUT headers");
    let header = demuxer.stream().clone();
    assert_eq!(header.fourcc, b"H264", "carried as h264");
    assert!(
        !header.extradata.is_empty(),
        "the encoder's header rides in the stream's"
    );
    let mut packets = 0;
    let mut buf = Vec::new();
    while demuxer
        .read_packet(&mut buf)
        .expect("read a packet")
        .is_some()
    {
        packets += 1;
    }
    assert_eq!(packets, 60, "one packet a picture");

    let mp4 = scratch("d3d12va_out.mp4");
    ffmpeg_ok(
        &[
            "-f",
            "nut",
            "-i",
            path_str(&out),
            "-c",
            "copy",
            path_str(&mp4),
        ],
        "ffmpeg -c copy",
    );
    let probed = Command::new("ffprobe")
        .args(["-v", "error", "-count_frames", "-select_streams", "v:0"])
        .args([
            "-show_entries",
            "stream=codec_name,width,height,nb_read_frames",
        ])
        .args(["-of", "csv=p=0"])
        .arg(&mp4)
        .stdin(Stdio::null())
        .output()
        .expect("spawn ffprobe");
    assert_eq!(
        String::from_utf8_lossy(&probed.stdout).trim(),
        "h264,320,240,60"
    );

    let ours = decoded_by_ffmpeg(&mp4, "yuv420p");
    let theirs = decoded_by_ffmpeg(&source, "yuv420p");
    for path in [&source, &out, &mp4] {
        let _ = std::fs::remove_file(path);
    }
    let len = SIZE.0 * SIZE.1 * 3 / 2;
    assert_eq!(ours.len(), theirs.len(), "as many pictures");
    let mut worst = 0f64;
    for (a, b) in ours.chunks(len).zip(theirs.chunks(len)) {
        let (a, b) = (
            plane_means(a, SIZE.0, SIZE.1),
            plane_means(b, SIZE.0, SIZE.1),
        );
        for plane in 0..3 {
            worst = worst.max((a[plane] - b[plane]).abs());
        }
    }
    eprintln!("each picture's mean colour is within {worst:.3} of the source's");
    assert!(worst < 2.0, "a picture's mean colour is {worst} levels off");
}

#[test]
fn an_encoder_on_the_device_takes_a_modules_rgba_as_nv12_made_there() {
    let test = "an_encoder_on_the_device_takes_a_modules_rgba_as_nv12_made_there";
    let Some((_, encoder, dir)) = shared_encoder(test) else {
        return;
    };
    let Some(()) = ffmpeg(test) else {
        return;
    };
    let module = module();
    let source = scratch("d3d12va_rgba.nut");
    ffmpeg_ok(
        &[
            "-f",
            "lavfi",
            "-i",
            "testsrc2=s=320x240:r=25",
            "-frames:v",
            "30",
            "-c:v",
            "rawvideo",
            "-pix_fmt",
            "rgba",
            "-f",
            "nut",
            path_str(&source),
        ],
        "making an rgba stream",
    );
    let encoded = scratch("d3d12va_boxed.nut");
    let chain = format!(
        "[v=0:v]gpu_box=memory=gpu:out=gpu:size=40[out=b];[b]encode=codec={encoder}:b=8M[out0]"
    );
    let output = run_chain(&dir, &source, &chain, &encoded, Some(&module));
    if no_gpu(&output) {
        eprintln!("skipped: this machine has no GPU adapter");
        return;
    }
    if no_device_decoder(test, &output) {
        return;
    }
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    assert!(
        output.status.success(),
        "exited {:?}:\n{stderr}",
        output.status.code()
    );
    assert!(stderr.contains(&from_the_device(encoder)), "{stderr}");
    assert!(
        stderr.contains("made NV12 from rgba first"),
        "the conversion is said:\n{stderr}"
    );
    let row = profile_row(&stderr);
    assert_eq!(row["download"]["bytes"], 0, "nothing is read back: {row}");

    let wire = std::fs::read(&encoded).expect("read the output");
    let header = Demuxer::open(wire.as_slice())
        .expect("read NUT headers")
        .stream()
        .clone();
    match header.media {
        ffrwd_wasm::nut::Media::Video {
            colorspace_type, ..
        } => assert_eq!(colorspace_type, 2, "the header says BT.709, limited range"),
        other => panic!("not video: {other:?}"),
    }

    // The host path's pictures, made NV12 by ffmpeg, are what the device's
    // conversion matches before the encode.
    let host = scratch("d3d12va_host.nut");
    let output = run_chain(
        &dir,
        &source,
        "[v=0:v]gpu_box=memory=host:size=40[out=out0]",
        &host,
        Some(&module),
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let wanted = ffmpeg_ok(
        &[
            "-i",
            path_str(&host),
            "-vf",
            "scale=out_color_matrix=bt709:out_range=tv,format=nv12",
            "-f",
            "rawvideo",
            "-",
        ],
        "ffmpeg's rgba to NV12",
    );
    let got = decoded_by_ffmpeg(&encoded, "nv12");
    for path in [&source, &encoded, &host] {
        let _ = std::fs::remove_file(path);
    }
    let (width, height) = SIZE;
    let len = width * height * 3 / 2;
    assert_eq!(got.len(), wanted.len(), "as many pictures");
    assert_eq!(got.len(), 30 * len);
    let luma = width * height;
    let mean = |x: &[u8], y: &[u8]| {
        x.iter()
            .zip(y)
            .map(|(p, q)| f64::from(p.abs_diff(*q)))
            .sum::<f64>()
            / x.len() as f64
    };
    // The box, inside its edges: red's luma in BT.709 limited range is 63.
    let box_luma = |p: &[u8]| {
        let mut sum = 0u32;
        for y in 5..35 {
            sum += p[y * width + 5..y * width + 35]
                .iter()
                .map(|&b| u32::from(b))
                .sum::<u32>();
        }
        f64::from(sum) / 900.0
    };
    let mut worst = [0f64; 2];
    for (index, (a, b)) in got.chunks(len).zip(wanted.chunks(len)).enumerate() {
        worst[0] = worst[0].max(mean(&a[..luma], &b[..luma]));
        worst[1] = worst[1].max(mean(&a[luma..], &b[luma..]));
        assert!(
            (box_luma(a) - box_luma(b)).abs() < 2.0 && (box_luma(a) - 63.0).abs() < 2.0,
            "picture {index}: the box's luma is {} here and {} in ffmpeg's",
            box_luma(a),
            box_luma(b)
        );
    }
    eprintln!(
        "against ffmpeg's conversion of the host path: luma {:.3}, chroma {:.3} levels apart \
         on average in the worst picture",
        worst[0], worst[1]
    );
    assert!(worst[0] < 2.0 && worst[1] < 2.0, "{worst:?}");
}

#[test]
fn h264_d3d12va_uploads_a_frame_in_host_memory_and_says_so() {
    let test = "h264_d3d12va_uploads_a_frame_in_host_memory_and_says_so";
    let Some(dir) = d3d12va_libav(test) else {
        return;
    };
    let Some(()) = ffmpeg(test) else {
        return;
    };
    let source = scratch("d3d12va_host_rgba.nut");
    ffmpeg_ok(
        &[
            "-f",
            "lavfi",
            "-i",
            "testsrc2=s=320x240:r=25",
            "-frames:v",
            "10",
            "-c:v",
            "rawvideo",
            "-pix_fmt",
            "rgba",
            "-f",
            "nut",
            path_str(&source),
        ],
        "making an rgba stream",
    );
    let out = scratch("d3d12va_uploaded.nut");
    let output = run_chain(
        &dir,
        &source,
        "[0:v]encode=codec=h264_d3d12va:b=8M[out0]",
        &out,
        None,
    );
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    if !output.status.success() && stderr.contains("av_hwframe_ctx_init") {
        eprintln!("skipped: this machine has no D3D12 video encoder: {stderr}");
        return;
    }
    assert!(
        output.status.success(),
        "exited {:?}:\n{stderr}",
        output.status.code()
    );
    assert!(
        stderr.contains("[av] encode h264_d3d12va: 320x240 nv12, uploaded from host memory"),
        "{stderr}"
    );
    let row = profile_row(&stderr);
    assert_eq!(
        row["upload"]["bytes"],
        10 * 320 * 240 * 3 / 2,
        "each picture is uploaded once, as NV12: {row}"
    );
    // Converted by libswscale before the upload, in the colour the header
    // declares: BT.709, limited range.
    let header = Demuxer::open(std::fs::read(&out).expect("read the output").as_slice())
        .expect("read NUT headers")
        .stream()
        .clone();
    assert!(
        !header.extradata.is_empty(),
        "the header has the parameter sets"
    );
    match header.media {
        ffrwd_wasm::nut::Media::Video {
            colorspace_type, ..
        } => assert_eq!(colorspace_type, 2, "BT.709, limited range"),
        other => panic!("not video: {other:?}"),
    }
    let got = decoded_by_ffmpeg(&out, "nv12");
    let wanted = ffmpeg_ok(
        &[
            "-i",
            path_str(&source),
            "-vf",
            "scale=out_color_matrix=bt709:out_range=tv,format=nv12",
            "-f",
            "rawvideo",
            "-",
        ],
        "ffmpeg's rgba to NV12",
    );
    let _ = std::fs::remove_file(&out);
    let _ = std::fs::remove_file(&source);
    assert_eq!(got.len(), wanted.len(), "as many pictures");
    let apart = got
        .iter()
        .zip(&wanted)
        .map(|(a, b)| f64::from(a.abs_diff(*b)))
        .sum::<f64>()
        / got.len() as f64;
    assert!(apart < 1.0, "{apart} levels apart on average");
}

// The host's own conversions on the device: a decoder's NV12 handed to a
// reader that asks for rgba, and an input that follows another's size
// (`accepts.like`) resized, whether it is bound to a stream or fed on a
// port.

/// The largest and mean difference between two `width`-wide rgba pictures,
/// the `side`-wide box gpu_box draws in the corner left out.
fn apart_outside_the_box(a: &[u8], b: &[u8], width: usize, side: usize) -> (u8, f64) {
    assert_eq!(a.len(), b.len());
    let (mut max, mut sum, mut count) = (0u8, 0u64, 0u64);
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        let pixel = i / 4;
        if pixel % width < side && pixel / width < side {
            continue;
        }
        let d = x.abs_diff(*y);
        max = max.max(d);
        sum += u64::from(d);
        count += 1;
    }
    (max, sum as f64 / count.max(1) as f64)
}

/// Whether the red box is at the corner of a `width`-wide rgba picture,
/// `side` wide.
fn has_the_box(picture: &[u8], width: usize, side: usize) -> bool {
    (0..side).all(|y| {
        (0..side).all(|x| {
            let at = (y * width + x) * 4;
            picture[at..at + 4] == [255, 0, 0, 255]
        })
    })
}

#[test]
fn pictures_decoded_on_the_device_reach_an_rgba_reader_converted_there() {
    let test = "pictures_decoded_on_the_device_reach_an_rgba_reader_converted_there";
    let Some((device, dir)) = small_shared_decoder(test) else {
        return;
    };
    let Some(()) = ffmpeg(test) else {
        return;
    };
    let module = module();
    let out = scratch("rgba_from_the_device.nut");
    let output = run_chain(
        &dir,
        &h264_fixture(),
        &format!(
            "[0:v]decode=format=rgba:hw={device}[d];[v=d]gpu_box=memory=gpu:out=gpu:size=5[out=out0]"
        ),
        &out,
        Some(&module),
    );
    if no_gpu(&output) {
        eprintln!("skipped: this machine has no GPU adapter");
        return;
    }
    if no_device_decoder(test, &output) {
        return;
    }
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    assert!(
        output.status.success(),
        "exited {:?}:\n{stderr}",
        output.status.code()
    );
    assert!(stderr.contains(&decoded_line(device)), "{stderr}");
    assert!(
        stderr.contains("[gpu] convert nv12 64x48 to rgba 64x48"),
        "the decoder's NV12 is made rgba on the device:\n{stderr}"
    );
    assert!(
        !stderr.contains(UPLOAD_LINE),
        "nothing is uploaded:\n{stderr}"
    );
    let pictures = frames_in(&out);
    assert!(!pictures.is_empty());
    let row = profile_row(&stderr);
    assert_eq!(row["upload"]["bytes"], 0, "{row}");
    assert_eq!(
        row["convert"]["calls"],
        pictures.len(),
        "one conversion a picture: {row}"
    );
    assert_eq!(
        row["download"]["bytes"],
        pictures.len() * 64 * 48 * 4,
        "only what leaves on the wire is read back: {row}"
    );

    // ffmpeg's bilinear conversion, with the chroma interpolated rather
    // than repeated and rounded accurately, is what the device's is.
    let theirs = ffmpeg_ok(
        &[
            "-i",
            path_str(&h264_fixture()),
            "-fps_mode",
            "passthrough",
            "-vf",
            "scale=flags=bilinear+full_chroma_int+accurate_rnd,format=rgba",
            "-f",
            "rawvideo",
            "-",
        ],
        "ffmpeg's yuv420p to rgba",
    );
    let len = 64 * 48 * 4;
    assert_eq!(theirs.len(), pictures.len() * len, "as many pictures");
    let (mut max, mut mean) = (0u8, 0f64);
    for ((_, ours), theirs) in pictures.iter().zip(theirs.chunks(len)) {
        assert!(has_the_box(ours, 64, 5));
        let (m, a) = apart_outside_the_box(ours, theirs, 64, 5);
        max = max.max(m);
        mean = mean.max(a);
    }
    eprintln!(
        "against ffmpeg's conversion: at most {max} apart, mean {mean:.3} in the worst picture"
    );
    // Measured on an RTX 4090: at most 1 apart, a mean of 0.018.
    assert!(max <= 1, "{max} apart");
    assert!(mean < 0.03, "mean {mean}");
}

/// `source`, h264, decoded to rgba by the sidecar with a BT.709 `-pad`, on
/// `hw`, through gpu_box on the GPU: the pictures, and the run's stderr.
fn decoded_bt709(
    dir: &Path,
    module: &str,
    source: &Path,
    hw: &str,
) -> (Vec<(i64, Vec<u8>)>, Output) {
    let out = scratch(&format!("bt709_{hw}.nut"));
    let output = Command::new(env!("CARGO_BIN_EXE_ffrwd-wasm"))
        .env("FFRWD_AV_RUNTIME", dir)
        .args(["-gpu", module, "-f", "nut", "-i"])
        .arg(source)
        .args([
            "-pad",
            r#"{"color": [{"range": "tv", "primaries": "bt709", "trc": "bt709", "space": "bt709"}]}"#,
            "-m",
            &format!("gpu_box={module}"),
            "-filter_complex",
            &format!(
                "[0:v]decode=format=rgba:hw={hw}[d];\
                 [v=d]gpu_box=memory=gpu:out=gpu:size=5[out=out0]"
            ),
            "-map",
            "[out0]",
            "-f",
            "nut",
        ])
        .arg(&out)
        .stdin(Stdio::null())
        .output()
        .expect("spawn ffrwd-wasm");
    let pictures = if output.status.success() {
        frames_in(&out)
    } else {
        Vec::new()
    };
    (pictures, output)
}

#[test]
fn a_picture_decoded_on_the_device_is_made_rgba_in_the_colour_the_software_decoder_reads() {
    let test =
        "a_picture_decoded_on_the_device_is_made_rgba_in_the_colour_the_software_decoder_reads";
    let Some((hw, dir)) = small_shared_decoder(test) else {
        return;
    };
    let Some(()) = ffmpeg(test) else {
        return;
    };
    let module = module();
    // The fixture's bitstream says nothing of its colour, so both read it in
    // the pad's BT.709 limited; one whose bitstream says BT.601 is read in
    // BT.601 by both, whatever the pad says.
    let said = scratch("bitstream_bt601.nut");
    ffmpeg_ok(
        &[
            "-i",
            path_str(&h264_fixture()),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-color_range",
            "tv",
            "-colorspace",
            "bt470bg",
            "-f",
            "nut",
            path_str(&said),
        ],
        "re-encoding the fixture with BT.601 in its bitstream",
    );
    for (source, read_in, other) in [
        (h264_fixture(), "bt709", "bt601"),
        (said.clone(), "bt601", "bt709"),
    ] {
        let (device, output) = decoded_bt709(&dir, &module, &source, hw);
        if no_gpu(&output) {
            eprintln!("skipped: this machine has no GPU adapter");
            return;
        }
        if no_device_decoder(test, &output) {
            return;
        }
        let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
        assert!(output.status.success(), "{stderr}");
        assert!(
            stderr.contains("[gpu] convert nv12 64x48 to rgba 64x48"),
            "the decoder's NV12 is made rgba on the device:\n{stderr}"
        );
        let (software, output) = decoded_bt709(&dir, &module, &source, "none");
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        // ffmpeg's conversion in `matrix`, with `flags` the scaler's.
        let in_matrix = |matrix: &str, flags: &str| {
            ffmpeg_ok(
                &[
                    "-i",
                    path_str(&source),
                    "-fps_mode",
                    "passthrough",
                    "-vf",
                    &format!("scale=in_color_matrix={matrix}:in_range=tv{flags},format=rgba"),
                    "-f",
                    "rawvideo",
                    "-",
                ],
                "ffmpeg's yuv420p to rgba",
            )
        };
        // The device's conversion is ffmpeg's bilinear one with the chroma
        // interpolated; the software decoder's is libswscale's default.
        let bilinear = ":flags=bilinear+full_chroma_int+accurate_rnd";
        let len = 64 * 48 * 4;
        let worst = |ours: &[(i64, Vec<u8>)], theirs: &[u8]| {
            assert_eq!(theirs.len(), ours.len() * len, "as many pictures");
            let (mut max, mut mean) = (0u8, 0f64);
            for ((_, ours), theirs) in ours.iter().zip(theirs.chunks(len)) {
                let (m, a) = apart_outside_the_box(ours, theirs, 64, 5);
                max = max.max(m);
                mean = mean.max(a);
            }
            (max, mean)
        };
        let on_device = worst(&device, &in_matrix(read_in, bilinear));
        let on_device_other = worst(&device, &in_matrix(other, bilinear));
        let in_software = worst(&software, &in_matrix(read_in, ""));
        let in_software_other = worst(&software, &in_matrix(other, ""));
        eprintln!(
            "against ffmpeg's {read_in} ({other}), at most apart and mean in the worst \
             picture: the device {on_device:?} ({on_device_other:?}), the software decode \
             {in_software:?} ({in_software_other:?})"
        );
        // Measured on an RTX 4090: the device at most 1 apart, a mean of
        // 0.031; the software decode the same bytes; the other matrix over 3
        // apart on average.
        assert!(on_device.0 <= 1 && on_device.1 < 0.05, "{on_device:?}");
        assert_eq!(
            in_software.0, 0,
            "the software decode is ffmpeg's {read_in}"
        );
        assert!(on_device_other.1 > 1.0 && in_software_other.1 > 1.0);
    }
    let _ = std::fs::remove_file(&said);
}

/// `frames` of `width` x `height` rgba as a NUT file at `path`, 25 a
/// second.
fn write_rgba_nut(path: &Path, frames: &[Vec<u8>], width: u32, height: u32) {
    let mut stream = Stream::video("rgba", width, height, TIME_BASE).expect("rgba is carried");
    stream.frame_rate = Some((25, 1));
    let mut wire = Vec::new();
    {
        let mut muxer = Muxer::new(&mut wire, &stream).expect("write NUT headers");
        for (i, frame) in frames.iter().enumerate() {
            muxer.write_frame(i as i64, frame).expect("write a frame");
        }
        muxer.finish().expect("finish the stream");
    }
    std::fs::write(path, wire).expect("write the NUT file");
}

/// A `width` x `height` rgba picture with ramps, hard edges and a
/// checkerboard, opaque.
fn pattern(width: u32, height: u32) -> Vec<u8> {
    let mut rgba = Vec::with_capacity((width * height * 4) as usize);
    for y in 0..height {
        for x in 0..width {
            rgba.extend_from_slice(&[
                (x * 255 / (width - 1)) as u8,
                (y * 255 / (height - 1)) as u8,
                (((x / 8 + y / 8) % 2) * 200 + 20) as u8,
                255,
            ]);
        }
    }
    rgba
}

/// A flat green `width` x `height` rgba picture.
fn green(width: u32, height: u32) -> Vec<u8> {
    [0, 200, 0, 255].repeat((width * height) as usize)
}

/// ffmpeg's bilinear resize of a `from`-sized rgba picture to `to`, with
/// rgba's chroma kept at full size, which the device's resize is measured
/// against.
fn resized_by_ffmpeg(rgba: &[u8], from: (u32, u32), to: (u32, u32), name: &str) -> Vec<u8> {
    let path = scratch(name);
    std::fs::write(&path, rgba).expect("write the picture");
    let resized = ffmpeg_ok(
        &[
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgba",
            "-s",
            &format!("{}x{}", from.0, from.1),
            "-i",
            path_str(&path),
            "-vf",
            &format!(
                "scale={}:{}:flags=bilinear+full_chroma_inp+full_chroma_int,format=rgba",
                to.0, to.1
            ),
            "-f",
            "rawvideo",
            "-",
        ],
        "ffmpeg's bilinear resize",
    );
    let _ = std::fs::remove_file(&path);
    resized
}

#[test]
fn an_input_that_follows_another_is_resized_on_the_device() {
    let test = "an_input_that_follows_another_is_resized_on_the_device";
    let Some(()) = ffmpeg(test) else {
        return;
    };
    let module = module();
    let (small, large) = ((32u32, 24u32), (64u32, 48u32));
    let (v, w) = (scratch("follow_v.nut"), scratch("follow_w.nut"));
    write_rgba_nut(&v, &vec![green(small.0, small.1); 5], small.0, small.1);
    let picture = pattern(large.0, large.1);
    write_rgba_nut(&w, &vec![picture.clone(); 5], large.0, large.1);
    let out = scratch("follow_out.nut");
    let output = Command::new(env!("CARGO_BIN_EXE_ffrwd-wasm"))
        .args(["-profile", "-gpu", &module, "-f", "nut", "-i"])
        .arg(&v)
        .args(["-f", "nut", "-i"])
        .arg(&w)
        .arg("-m")
        .arg(format!("gpu_box={module}"))
        .args([
            "-filter_complex",
            "[v=0:v][w=1:v]gpu_box=memory=gpu:out=gpu:w=pad:size=3[out=o]",
            "-map",
            "[o]",
            "-f",
            "nut",
        ])
        .arg(&out)
        .stdin(Stdio::null())
        .output()
        .expect("spawn ffrwd-wasm");
    let _ = (std::fs::remove_file(&v), std::fs::remove_file(&w));
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
        stderr.contains("[gpu] gpu_box: resizing the frames of stream 1 on the GPU to 32x24"),
        "{stderr}"
    );
    assert!(
        stderr.contains("[gpu] convert rgba 64x48 to rgba 32x24"),
        "{stderr}"
    );
    let row = profile_row(&stderr);
    assert_eq!(row["convert"]["calls"], 5, "one resize a frame: {row}");
    let pictures = frames_in(&out);
    assert_eq!(pictures.len(), 5);
    let theirs = resized_by_ffmpeg(&picture, large, small, "follow_picture.rgba");
    for (pts, ours) in &pictures {
        assert!(has_the_box(ours, small.0 as usize, 3));
        let (max, mean) = apart_outside_the_box(ours, &theirs, small.0 as usize, 3);
        eprintln!("picture {pts} against ffmpeg's resize: at most {max} apart, mean {mean:.3}");
        // Measured on an RTX 4090: at most 1 apart, a mean of 0.047.
        assert!(
            max <= 1 && mean < 0.05,
            "picture {pts}: {max} apart, mean {mean}"
        );
    }
}

/// How long a run may take to put its feed's port up while the programme's
/// first frame is the only one written: spawning the sidecar, opening the
/// device and compiling the module come first, and take what the machine
/// makes them (2.4 s on an M1 Max, 8 s for a debug build on one core). It
/// only has to be past any start.
const LISTENING_BOUND: std::time::Duration = std::time::Duration::from_secs(60);

/// Reads `child`'s stderr to its end on a thread of its own: the moment
/// its feed's port is up, sent once, and the whole of it when the run ends.
fn watch_for_listening(
    child: &mut std::process::Child,
) -> (
    std::sync::mpsc::Receiver<std::time::Instant>,
    std::thread::JoinHandle<String>,
) {
    use std::io::BufRead;
    let stderr = child.stderr.take().expect("stderr is piped");
    let (up, listening) = std::sync::mpsc::channel();
    let reader = std::thread::spawn(move || {
        let mut all = String::new();
        for line in std::io::BufReader::new(stderr)
            .lines()
            .map_while(Result::ok)
        {
            if line.contains("\"kind\":\"listen\"") {
                let _ = up.send(std::time::Instant::now());
            }
            all.push_str(&line);
            all.push('\n');
        }
        all
    });
    (listening, reader)
}

#[test]
fn a_feed_held_on_the_gpu_is_conformed_on_the_device() {
    use std::net::{TcpListener, TcpStream};
    use std::time::{Duration, Instant};
    let test = "a_feed_held_on_the_gpu_is_conformed_on_the_device";
    let Some(()) = ffmpeg(test) else {
        return;
    };
    let module = module();
    let (small, large) = ((32u32, 24u32), (64u32, 48u32));
    let port = {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind a free port");
        listener.local_addr().expect("its address").port()
    };
    let wiring = format!("[v=0:v]gpu_box=memory=gpu:out=gpu:w=port:port={port}:size=3[out=o]");
    let mut child = Command::new(env!("CARGO_BIN_EXE_ffrwd-wasm"))
        .args(["-gpu", &module, "-f", "nut", "-i", "-", "-m"])
        .arg(format!("gpu_box={module}"))
        .args(["-filter_complex", &wiring, "-map", "[o]", "-f", "nut", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn ffrwd-wasm");

    // The programme, flat green, at 25 a second for three seconds: its first
    // frame at once, the rest paced from when the feed's port is up, so the
    // sidecar's own start is not in the three seconds.
    let (listening, stderr_reader) = watch_for_listening(&mut child);
    let mut stdin = child.stdin.take().expect("stdin is piped");
    let programme = std::thread::spawn(move || {
        let mut stream = Stream::video("rgba", small.0, small.1, TIME_BASE).expect("rgba");
        stream.frame_rate = Some((25, 1));
        let mut muxer = Muxer::new(&mut stdin, &stream).expect("headers");
        if muxer.write_frame(0, &green(small.0, small.1)).is_err() {
            return;
        }
        let _ = muxer.flush();
        let started = listening
            .recv_timeout(LISTENING_BOUND)
            .unwrap_or_else(|_| Instant::now());
        for k in 1..75i64 {
            let due = started + Duration::from_millis(40 * k as u64);
            if let Some(wait) = due.checked_duration_since(Instant::now()) {
                std::thread::sleep(wait);
            }
            if muxer.write_frame(k, &green(small.0, small.1)).is_err() {
                return;
            }
            let _ = muxer.flush();
        }
        let _ = muxer.finish();
    });
    // The feeder: the pattern at twice the size, for most of the run.
    let picture = pattern(large.0, large.1);
    let sent = picture.clone();
    let feeder = std::thread::spawn(move || {
        let until = Instant::now() + Duration::from_secs(10);
        let socket = loop {
            std::thread::sleep(Duration::from_millis(100));
            if let Ok(socket) = TcpStream::connect(("127.0.0.1", port)) {
                break socket;
            }
            if Instant::now() > until {
                return;
            }
        };
        let mut stream = Stream::video("rgba", large.0, large.1, TIME_BASE).expect("rgba");
        stream.frame_rate = Some((25, 1));
        let Ok(mut muxer) = Muxer::new(socket, &stream) else {
            return;
        };
        for k in 0..40i64 {
            if muxer.write_frame(k, &sent).is_err() || muxer.flush().is_err() {
                return;
            }
            std::thread::sleep(Duration::from_millis(40));
        }
        let _ = muxer.finish();
    });
    let output = child.wait_with_output().expect("wait for ffrwd-wasm");
    let _ = (programme.join(), feeder.join());
    let stderr = stderr_reader.join().expect("read stderr");
    if no_gpu(&output) {
        eprintln!("skipped: this machine has no GPU adapter");
        return;
    }
    assert!(
        output.status.success(),
        "exited {:?}:\n{stderr}",
        output.status.code()
    );
    let conformed = "feed: gpu_box 'w': the feeder's picture is 64x48 rgba, conformed to \
                     32x24 rgba on the GPU";
    assert!(stderr.contains(conformed), "{stderr}");
    assert!(
        stderr.contains("[gpu] convert rgba 64x48 to rgba 32x24"),
        "{stderr}"
    );
    let pictures = read_nut(&output.stdout);
    assert_eq!(pictures.len(), 75, "a picture a programme frame");
    let theirs = resized_by_ffmpeg(&picture, large, small, "feed_picture.rgba");
    // A picture the feed did not reach is the programme's flat green.
    let fed: Vec<&Vec<u8>> = pictures
        .iter()
        .map(|(_, p)| p)
        .filter(|p| p[(5 * small.0 as usize + 5) * 4 + 1] != 200)
        .collect();
    assert!(
        fed.len() >= 10,
        "the feed shows on {} pictures of 75:\n{stderr}",
        fed.len()
    );
    let (mut max, mut mean) = (0u8, 0f64);
    for ours in &fed {
        assert!(has_the_box(ours, small.0 as usize, 3));
        let (m, a) = apart_outside_the_box(ours, &theirs, small.0 as usize, 3);
        max = max.max(m);
        mean = mean.max(a);
    }
    eprintln!(
        "{} fed pictures against ffmpeg's resize: at most {max} apart, mean {mean:.3}",
        fed.len()
    );
    // Measured on an RTX 4090: at most 1 apart, a mean of 0.047.
    assert!(max <= 1 && mean < 0.05, "{max} apart, mean {mean}");
}

#[test]
fn a_coded_feed_held_on_the_gpu_is_decoded_on_the_device() {
    use std::net::{TcpListener, TcpStream};
    use std::time::{Duration, Instant};
    let test = "a_coded_feed_held_on_the_gpu_is_decoded_on_the_device";
    let Some((device, dir)) = shared_decoder(test) else {
        return;
    };
    let module = module();
    let small = (32u32, 24u32);
    let port = {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind a free port");
        listener.local_addr().expect("its address").port()
    };
    let wiring = format!("[v=0:v]gpu_box=memory=gpu:out=gpu:w=port:port={port}:size=3[out=o]");
    let mut child = Command::new(env!("CARGO_BIN_EXE_ffrwd-wasm"))
        .env("FFRWD_AV_RUNTIME", &dir)
        .args(["-profile", "-gpu", &module, "-f", "nut", "-i", "-", "-m"])
        .arg(format!("gpu_box={module}"))
        .args(["-filter_complex", &wiring, "-map", "[o]", "-f", "nut", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn ffrwd-wasm");

    // The programme, flat green, at 25 a second for three seconds: its first
    // frame at once, the rest paced from when the feed's port is up, so the
    // sidecar's own start is not in the three seconds.
    let (listening, stderr_reader) = watch_for_listening(&mut child);
    let mut stdin = child.stdin.take().expect("stdin is piped");
    let programme = std::thread::spawn(move || {
        let mut stream = Stream::video("rgba", small.0, small.1, TIME_BASE).expect("rgba");
        stream.frame_rate = Some((25, 1));
        let mut muxer = Muxer::new(&mut stdin, &stream).expect("headers");
        if muxer.write_frame(0, &green(small.0, small.1)).is_err() {
            return;
        }
        let _ = muxer.flush();
        let started = listening
            .recv_timeout(LISTENING_BOUND)
            .unwrap_or_else(|_| Instant::now());
        for k in 1..75i64 {
            let due = started + Duration::from_millis(40 * k as u64);
            if let Some(wait) = due.checked_duration_since(Instant::now()) {
                std::thread::sleep(wait);
            }
            if muxer.write_frame(k, &green(small.0, small.1)).is_err() {
                return;
            }
            let _ = muxer.flush();
        }
        let _ = muxer.finish();
    });
    // The feeder: the h264 fixture's packets as they are.
    let feeder = std::thread::spawn(move || {
        let until = Instant::now() + Duration::from_secs(10);
        let socket = loop {
            std::thread::sleep(Duration::from_millis(100));
            if let Ok(socket) = TcpStream::connect(("127.0.0.1", port)) {
                break socket;
            }
            if Instant::now() > until {
                return;
            }
        };
        let file = std::fs::File::open(h264_fixture()).expect("open the h264 fixture");
        let mut demuxer = Demuxer::open(file).expect("read the fixture's NUT headers");
        let stream = demuxer.stream().clone();
        let Ok(mut muxer) = Muxer::with_streams(std::io::BufWriter::new(socket), &[stream]) else {
            return;
        };
        let mut buf = Vec::new();
        while let Ok(Some(packet)) = demuxer.read_packet(&mut buf) {
            if muxer.write_coded_to(0, &packet, &buf).is_err() || muxer.flush().is_err() {
                return;
            }
            std::thread::sleep(Duration::from_millis(40));
        }
        let _ = muxer.finish();
    });
    let output = child.wait_with_output().expect("wait for ffrwd-wasm");
    let _ = (programme.join(), feeder.join());
    let stderr = stderr_reader.join().expect("read stderr");
    if no_gpu(&output) {
        eprintln!("skipped: this machine has no GPU adapter");
        return;
    }
    if let Some(line) = stderr.lines().find(|l| l.starts_with(NO_DEVICE_DECODER)) {
        eprintln!("SKIPPED: {test}. {line}");
        return;
    }
    assert!(
        output.status.success(),
        "exited {:?}:\n{stderr}",
        output.status.code()
    );
    assert!(stderr.contains(&decoded_line(device)), "{stderr}");
    let (fw, fh) = h264_fixture_size();
    assert!(
        stderr.contains(&format!(
            "feed: gpu_box 'w': the feeder's picture is {fw}x{fh} h264, conformed to 32x24 \
             rgba on the GPU"
        )),
        "{stderr}"
    );
    // The programme alone is uploaded; the feed's pictures never were in
    // host memory.
    let programme_bytes = 75 * small.0 as usize * small.1 as usize * 4;
    assert!(
        stderr.contains(&format!("\"upload\":{{\"bytes\":{programme_bytes},")),
        "{stderr}"
    );
    let pictures = read_nut(&output.stdout);
    assert_eq!(pictures.len(), 75, "a picture a programme frame");
    let fed = pictures
        .iter()
        .filter(|(_, p)| p[(5 * small.0 as usize + 5) * 4 + 1] != 200)
        .count();
    assert!(
        fed >= 10,
        "the feed shows on {fed} pictures of 75:\n{stderr}"
    );
    assert!(pictures
        .iter()
        .all(|(_, p)| has_the_box(p, small.0 as usize, 3)));
}

// The host's `filter` node: a libavfilter graph on the device, between a
// decoder and an encoder on it, or over pictures in host memory.

/// The libav directory `FFRWD_AV_RUNTIME` names, where it holds libavfilter
/// beside libavcodec; None, having said why the test is skipped, otherwise.
fn d3d12va_libavfilter(test: &str) -> Option<PathBuf> {
    let dir = d3d12va_libav(test)?;
    if !dir.join("avfilter-12.dll").is_file() {
        eprintln!(
            "SKIPPED: {test}. {} holds no avfilter-12.dll; point FFRWD_AV_RUNTIME at a libav \
             built with libavfilter to run it.",
            dir.display()
        );
        return None;
    }
    Some(dir)
}

/// Whether the ffmpeg on PATH has `filter`; says the test is skipped where
/// it has not.
fn ffmpeg_has_filter(test: &str, filter: &str) -> bool {
    let listed = Command::new("ffmpeg")
        .args(["-hide_banner", "-filters"])
        .stdin(Stdio::null())
        .output()
        .expect("spawn ffmpeg");
    let has = String::from_utf8_lossy(&listed.stdout)
        .lines()
        .any(|line| line.split_whitespace().nth(1) == Some(filter));
    if !has {
        eprintln!("SKIPPED: {test}. The ffmpeg on PATH has no {filter}.");
    }
    has
}

/// What ffmpeg's own `scale_d3d12` makes of `source` (`args` ahead of the
/// input naming how the pictures reach it), as yuv420p pictures one after
/// another.
fn scaled_by_ffmpeg(source: &Path, args: &[&str], graph: &str) -> Vec<u8> {
    let mut all = args.to_vec();
    all.extend([
        "-i",
        path_str(source),
        "-vf",
        graph,
        "-fps_mode",
        "passthrough",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "yuv420p",
        "-",
    ]);
    ffmpeg_ok(&all, "ffmpeg's own scale_d3d12")
}

const SCALED: (usize, usize) = (160, 120);
const FILTER_LINE: &str = "[av] filter scale_d3d12: on the host's device";

#[test]
fn scale_d3d12_runs_between_a_decoder_and_an_encoder_on_the_device() {
    let test = "scale_d3d12_runs_between_a_decoder_and_an_encoder_on_the_device";
    let Some(dir) = d3d12va_libavfilter(test) else {
        return;
    };
    let Some(()) = ffmpeg(test) else {
        return;
    };
    if !ffmpeg_has_filter(test, "scale_d3d12") {
        return;
    }
    let source = scratch("filter_source.nut");
    let made = Command::new("ffmpeg")
        .args(["-v", "error", "-y", "-f", "lavfi", "-i"])
        .arg("testsrc2=s=320x240:r=25")
        .args(["-frames:v", "60", "-c:v", "libx264", "-pix_fmt", "yuv420p"])
        .args(["-f", "nut"])
        .arg(&source)
        .stdin(Stdio::null())
        .output()
        .expect("spawn ffmpeg");
    if !made.status.success() {
        eprintln!(
            "SKIPPED: {test}. The ffmpeg on PATH cannot make an h264 stream with libx264: {}",
            String::from_utf8_lossy(&made.stderr)
        );
        return;
    }
    let out = scratch("filter_out.nut");
    let chain = "[0:v]decode=format=yuv420p:hw=d3d12va[d];\
                 [d]filter=graph=scale_d3d12=w=160\\\\:h=120[f];\
                 [f]encode=codec=h264_d3d12va:b=2M[out0]";
    // Under the D3D12 debug layer, whatever the build: a filter whose
    // commands are recorded again before the device has run them is an
    // error there, and without the layer it goes unseen.
    let output = chain_command(&dir, &source, chain, &out, None)
        .env("WGPU_VALIDATION", "1")
        .output()
        .expect("spawn ffrwd-wasm");
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
    assert!(stderr.contains(&decoded_line("d3d12va")), "{stderr}");
    assert!(!stderr.contains(DEBUG_LAYER_ERROR), "{stderr}");
    assert!(stderr.contains(FILTER_LINE), "{stderr}");
    assert!(
        stderr.contains("[av] filter scale_d3d12: frames copied on the device"),
        "{stderr}"
    );
    assert!(
        stderr.contains("[gpu] filter scale_d3d12: d3d12va, frames wrapped"),
        "{stderr}"
    );
    assert!(
        stderr.contains("[av] encode h264_d3d12va: 160x120 nv12, from the device"),
        "{stderr}"
    );
    let row = profile_row(&stderr);
    for stage in ["upload", "download"] {
        assert_eq!(row[stage]["bytes"], 0, "nothing crosses to the host: {row}");
    }

    let mp4 = scratch("filter_out.mp4");
    ffmpeg_ok(
        &[
            "-f",
            "nut",
            "-i",
            path_str(&out),
            "-c",
            "copy",
            path_str(&mp4),
        ],
        "ffmpeg -c copy",
    );
    let probed = Command::new("ffprobe")
        .args(["-v", "error", "-count_frames", "-select_streams", "v:0"])
        .args([
            "-show_entries",
            "stream=codec_name,width,height,nb_read_frames",
        ])
        .args(["-of", "csv=p=0"])
        .arg(&mp4)
        .stdin(Stdio::null())
        .output()
        .expect("spawn ffprobe");
    assert_eq!(
        String::from_utf8_lossy(&probed.stdout).trim(),
        "h264,160,120,60"
    );

    let ours = decoded_by_ffmpeg(&mp4, "yuv420p");
    // ffmpeg decodes in software and uploads: the device's decode is the
    // software decode byte for byte, and ffmpeg's own d3d12va decode into
    // scale_d3d12 made one picture 9 levels off the rest in one of seven
    // runs on the machine this was written on.
    let theirs = scaled_by_ffmpeg(
        &source,
        &["-init_hw_device", "d3d12va=d", "-filter_hw_device", "d"],
        "format=nv12,hwupload,scale_d3d12=w=160:h=120,hwdownload,format=nv12",
    );
    for path in [&source, &out, &mp4] {
        let _ = std::fs::remove_file(path);
    }
    let len = SCALED.0 * SCALED.1 * 3 / 2;
    assert_eq!(ours.len(), theirs.len(), "as many pictures");
    assert_eq!(ours.len(), 60 * len, "sixty 160x120 pictures");
    // The encoder at 2 Mb/s over a 160x120 picture leaves next to nothing
    // of its own: within a level on average, and each plane's mean within
    // one.
    let mut worst = 0f64;
    for (a, b) in ours.chunks(len).zip(theirs.chunks(len)) {
        let (a, b) = (
            plane_means(a, SCALED.0, SCALED.1),
            plane_means(b, SCALED.0, SCALED.1),
        );
        for plane in 0..3 {
            worst = worst.max((a[plane] - b[plane]).abs());
        }
    }
    let apart = ours
        .iter()
        .zip(&theirs)
        .map(|(a, b)| f64::from(a.abs_diff(*b)))
        .sum::<f64>()
        / ours.len() as f64;
    eprintln!(
        "against ffmpeg's own scale_d3d12: {apart:.3} levels apart on average, plane means \
         within {worst:.3}"
    );
    assert!(apart < 1.0, "{apart} levels apart on average");
    assert!(worst < 1.0, "a plane's mean is {worst} levels off");
}

#[test]
fn scale_d3d12_takes_pictures_in_host_memory_uploaded_and_reads_them_back() {
    let test = "scale_d3d12_takes_pictures_in_host_memory_uploaded_and_reads_them_back";
    let Some(dir) = d3d12va_libavfilter(test) else {
        return;
    };
    let Some(()) = ffmpeg(test) else {
        return;
    };
    if !ffmpeg_has_filter(test, "scale_d3d12") {
        return;
    }
    let source = scratch("filter_host.nut");
    ffmpeg_ok(
        &[
            "-f",
            "lavfi",
            "-i",
            "testsrc2=s=320x240:r=25",
            "-frames:v",
            "10",
            "-c:v",
            "rawvideo",
            "-pix_fmt",
            "yuv420p",
            "-f",
            "nut",
            path_str(&source),
        ],
        "making a yuv420p stream",
    );
    let out = scratch("filter_host_out.nut");
    let chain = "[0:v]filter=graph=scale_d3d12=w=160\\\\:h=120[out0]";
    let output = run_chain(&dir, &source, chain, &out, None);
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    if !output.status.success() && stderr.contains("av_hwframe_ctx_init") {
        eprintln!("skipped: this machine has no D3D12 device: {stderr}");
        return;
    }
    assert!(
        output.status.success(),
        "exited {:?}:\n{stderr}",
        output.status.code()
    );
    assert!(
        stderr.contains(
            "[av] filter scale_d3d12: on a d3d12va device of its own, pictures read back to \
             host memory"
        ),
        "{stderr}"
    );
    assert!(
        stderr.contains("[av] filter scale_d3d12: uploaded from host memory"),
        "{stderr}"
    );
    let row = profile_row(&stderr);
    assert_eq!(
        row["upload"]["bytes"],
        10 * 320 * 240 * 3 / 2,
        "each picture is uploaded once, as NV12: {row}"
    );
    assert_eq!(
        row["download"]["bytes"],
        10 * 160 * 120 * 3 / 2,
        "each scaled picture is read back once: {row}"
    );

    let ours = frames_in(&out);
    let theirs = scaled_by_ffmpeg(
        &source,
        &["-init_hw_device", "d3d12va=d", "-filter_hw_device", "d"],
        "format=nv12,hwupload,scale_d3d12=w=160:h=120,hwdownload,format=nv12",
    );
    let _ = std::fs::remove_file(&source);
    let ours: Vec<u8> = ours.into_iter().flat_map(|(_, picture)| picture).collect();
    assert_eq!(
        ours.len(),
        10 * SCALED.0 * SCALED.1 * 3 / 2,
        "ten 160x120 pictures"
    );
    // The same video processor scaled both, and every conversion around it
    // is a repacking of NV12 and yuv420p.
    assert!(
        ours == theirs,
        "the pictures differ from ffmpeg's own scale_d3d12"
    );
}
