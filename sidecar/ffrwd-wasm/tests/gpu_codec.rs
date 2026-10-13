//! A codec package's module built on `ffrwd:gpu`: its encoder is handed the
//! planes of a picture held on the GPU as textures, and its decoder hands
//! its pictures back the same way, so neither is read back or uploaded.
//!
//! `testcodec_gpu` is `testcodec` with `ffrwd:gpu`'s encoder and decoder
//! beside `ffrwd:av`'s. Each path is checked against `testcodec`, which
//! reads and writes bytes: the packets, and the pictures they decode to,
//! are the same byte for byte, and `-profile` counts what crossed host
//! memory. Without a GPU adapter the runs are skipped and say so.

use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Output, Stdio};
use std::sync::OnceLock;

use ffrwd_wasm::nut::{Demuxer, Muxer, Stream, TimeBase};

const WIDTH: u32 = 16;
const HEIGHT: u32 = 8;
const FRAME_LEN: usize = (WIDTH * HEIGHT * 4) as usize;
const TIME_BASE: TimeBase = TimeBase { num: 1, den: 25 };
const FRAMES: u8 = 4;

fn sidecar_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("ffrwd-wasm/ has a parent directory")
        .to_path_buf()
}

/// The built path of module `name`, after building gpu-box and both test
/// codecs once per test binary.
fn module(name: &str) -> String {
    static BUILT: OnceLock<()> = OnceLock::new();
    BUILT.get_or_init(|| {
        let output = Command::new("cargo")
            .args(["build", "--release", "--target", "wasm32-wasip2"])
            .args(["-p", "gpu-box", "-p", "testcodec", "-p", "testcodec-gpu"])
            .current_dir(sidecar_root().join("modules"))
            .output()
            .expect("spawn cargo build");
        assert!(
            output.status.success(),
            "building the modules failed:\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
    });
    sidecar_root()
        .join("modules/target/wasm32-wasip2/release")
        .join(format!("{name}.wasm"))
        .to_str()
        .expect("a UTF-8 path")
        .to_string()
}

/// Frame `index`: rgba with a distinct byte pattern per position and frame,
/// opaque.
fn picture(index: u8) -> Vec<u8> {
    (0..FRAME_LEN)
        .map(|offset| {
            if offset % 4 == 3 {
                255
            } else {
                index.wrapping_mul(53).wrapping_add((offset * 7) as u8)
            }
        })
        .collect()
}

fn rgba_wire() -> Vec<u8> {
    let stream = Stream::video("rgba", WIDTH, HEIGHT, TIME_BASE).expect("rgba is carried");
    let mut wire = Vec::new();
    {
        let mut muxer = Muxer::new(&mut wire, &stream).expect("write NUT headers");
        for index in 0..FRAMES {
            muxer
                .write_frame(i64::from(index), &picture(index))
                .expect("write a frame");
        }
        muxer.finish().expect("finish the stream");
    }
    wire
}

/// Every packet's or frame's pts and bytes.
fn items(wire: &[u8]) -> Vec<(i64, Vec<u8>)> {
    let mut demuxer = Demuxer::open(wire).expect("read NUT headers");
    let mut items = Vec::new();
    let mut buf = Vec::new();
    while let Some(packet) = demuxer.read_packet(&mut buf).expect("read a packet") {
        items.push((packet.pts, buf.clone()));
    }
    items
}

/// Runs a network over `wire` with gpu-box bound as `gpu_box` and `codec`
/// as `tc`, granted the GPU, with `-profile`.
fn run(wire: &[u8], codec: &str, network: &str, extra: &[&str]) -> Output {
    let gpu_box = module("gpu_box");
    let codec = module(codec);
    let mut child = Command::new(env!("CARGO_BIN_EXE_ffrwd-wasm"))
        .args(["-profile", "-gpu", &gpu_box, "-gpu", &codec])
        .args(["-f", "nut", "-i", "-", "-m"])
        .arg(format!("gpu_box={gpu_box}"))
        .arg("-m")
        .arg(format!("tc={codec}"))
        .args(["-filter_complex", network])
        .args(extra)
        .args(["-map", "[o]", "-f", "nut", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn ffrwd-wasm");
    let mut stdin = child.stdin.take().expect("stdin is piped");
    let wire = wire.to_vec();
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
    if missing {
        eprintln!("SKIPPED: no GPU adapter");
    }
    missing
}

/// The profile row's bytes for `stage`.
fn profiled(output: &Output, stage: &str) -> u64 {
    let stderr = String::from_utf8_lossy(&output.stderr);
    let row: serde_json::Value = stderr
        .lines()
        .filter_map(|line| line.strip_prefix("ffrwd:row "))
        .map(|line| serde_json::from_str::<serde_json::Value>(line).expect("a row is JSON"))
        .find(|row| row["kind"] == "profile")
        .unwrap_or_else(|| panic!("no profile row on stderr:\n{stderr}"));
    row[stage]["bytes"].as_u64().expect("a byte count")
}

fn ok(output: &Output, what: &str) -> Vec<(i64, Vec<u8>)> {
    assert!(
        output.status.success(),
        "{what} exited {:?}:\n{}",
        output.status.code(),
        String::from_utf8_lossy(&output.stderr)
    );
    items(&output.stdout)
}

const ENCODING: &str = "[v=0:v]gpu_box=memory=gpu:out=gpu:size=3[out=b];[b]tc[o]";

#[test]
fn an_encoder_on_textures_codes_a_picture_on_the_gpu_with_nothing_read_back() {
    let wire = rgba_wire();
    let textures = run(&wire, "testcodec_gpu", ENCODING, &[]);
    if no_gpu(&textures) {
        return;
    }
    let bytes = run(&wire, "testcodec", ENCODING, &[]);
    let coded = ok(&textures, "encoding from textures");
    assert_eq!(coded.len(), usize::from(FRAMES));
    assert_eq!(
        coded,
        ok(&bytes, "encoding from bytes"),
        "the packets are the ones the bytes make"
    );
    let picture = (WIDTH * HEIGHT * 3 / 2) as u64 * u64::from(FRAMES);
    assert_eq!(profiled(&textures, "download"), 0, "nothing is read back");
    assert_eq!(
        profiled(&bytes, "download"),
        picture,
        "the bytes build reads each yuv420p picture back once"
    );
}

#[test]
fn a_decoder_on_textures_leaves_its_pictures_on_the_gpu_with_nothing_uploaded() {
    let encoded = run(&rgba_wire(), "testcodec", ENCODING, &[]);
    if no_gpu(&encoded) {
        return;
    }
    ok(&encoded, "encoding");
    let coded = encoded.stdout;
    for (format, out) in [("yuv420p", "host"), ("rgba", "gpu")] {
        let network = format!("[0:v]tc[d];[v=d]gpu_box=memory=gpu:out={out}:size=3[out=o]");
        let told = format!(r#"tc=[{{"pix_fmt": "{format}"}}]"#);
        let textures = run(&coded, "testcodec_gpu", &network, &["-pads", &told]);
        let bytes = run(&coded, "testcodec", &network, &["-pads", &told]);
        let decoded = ok(&textures, &format!("decoding into textures, made {format}"));
        assert_eq!(decoded.len(), usize::from(FRAMES));
        assert_eq!(
            decoded,
            ok(&bytes, &format!("decoding into bytes, made {format}")),
            "{format}: the pictures are the ones the bytes decode to"
        );
        assert_eq!(
            profiled(&textures, "upload"),
            0,
            "{format}: nothing is uploaded"
        );
        assert!(
            profiled(&bytes, "upload") > 0,
            "{format}: the bytes build's pictures are uploaded"
        );
    }
}
