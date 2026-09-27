//! A codec package's module, hosted where ffmpeg's own codec would stand.
//!
//! `testcodec` codes every frame as one run-length keyframe under the tag
//! FTST, which no table in the wire names. What is checked here is the
//! host's half: the describe the compiler reads, a raw stream coded and
//! decoded back byte for byte at the same timestamps, the coded wire's tag
//! and extradata, and the refusals a run makes before a module is handed
//! something it cannot take.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::sync::OnceLock;

use ffrwd_wasm::nut::{Demuxer, Muxer, Packet, Stream, TimeBase};

const WIDTH: u32 = 32;
const HEIGHT: u32 = 16;
const TIME_BASE: TimeBase = TimeBase { num: 1, den: 25 };
/// Timestamps with a hole in them, so a host that renumbered frames would
/// be caught.
const PTS: &[i64] = &[0, 1, 2, 5, 6, 9];

fn sidecar_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("ffrwd-wasm/ has a parent directory")
        .to_path_buf()
}

/// Absolute path to the test codec's built component, built once per test
/// binary. `modules/` is a separate cargo workspace with its own build lock,
/// so this does not deadlock against the `cargo test` run driving this
/// binary.
fn codec_module() -> PathBuf {
    static BUILT: OnceLock<()> = OnceLock::new();
    BUILT.get_or_init(|| {
        let workspace = sidecar_root().join("modules");
        let output = Command::new("cargo")
            .args([
                "build",
                "--release",
                "--target",
                "wasm32-wasip2",
                "-p",
                "testcodec",
                "-p",
                "invert",
            ])
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
    sidecar_root().join("modules/target/wasm32-wasip2/release/testcodec.wasm")
}

fn module_arg() -> String {
    codec_module().to_str().expect("path is UTF-8").to_string()
}

struct Run {
    stdout: Vec<u8>,
    stderr: String,
    output: Output,
}

impl Run {
    fn assert_ok(&self, what: &str) {
        assert!(
            self.output.status.success(),
            "{what} exited with {:?}\nstderr:\n{}",
            self.output.status.code(),
            self.stderr
        );
    }

    fn assert_refused(&self, what: &str, needles: &[&str]) {
        assert!(
            !self.output.status.success(),
            "{what} was expected to be refused, and exited 0"
        );
        for needle in needles {
            assert!(
                self.stderr.contains(needle),
                "{what}: stderr does not say {needle:?}:\n{}",
                self.stderr
            );
        }
    }
}

/// Runs `ffrwd-wasm` with the given argv, feeding `stdin_bytes`.
fn run_ffrwd_wasm(args: &[&str], stdin_bytes: &[u8]) -> Run {
    let exe = env!("CARGO_BIN_EXE_ffrwd-wasm");
    let mut child = Command::new(exe)
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn ffrwd-wasm");
    let mut stdin = child.stdin.take().expect("child stdin");
    // A refusal can close stdin before it is all written, which is a broken
    // pipe rather than a test failure.
    let _ = stdin.write_all(stdin_bytes);
    drop(stdin);
    let output = child.wait_with_output().expect("wait for ffrwd-wasm");
    Run {
        stdout: output.stdout.clone(),
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        output,
    }
}

/// One yuv420p picture: flat runs, longer than one run byte can count, and
/// a stretch that changes every byte, so both ends of the run-length coding
/// are crossed.
fn yuv_frame(index: usize) -> Vec<u8> {
    let luma = (WIDTH * HEIGHT) as usize;
    let len = luma * 3 / 2;
    let mut frame = vec![0u8; len];
    for (i, byte) in frame.iter_mut().enumerate() {
        *byte = if i < 300 {
            16 + index as u8
        } else if i < luma {
            (i * 7 + index) as u8
        } else {
            128
        };
    }
    frame
}

/// A raw NUT stream of `frames` at `pts`, in `pix_fmt`.
fn raw_wire(pix_fmt: &str, frames: &[(i64, Vec<u8>)]) -> Vec<u8> {
    let stream = Stream::video(pix_fmt, WIDTH, HEIGHT, TIME_BASE).expect("pix_fmt is carried");
    let mut wire = Vec::new();
    {
        let mut muxer = Muxer::new(&mut wire, &stream).expect("write NUT headers");
        for (pts, data) in frames {
            muxer.write_frame(*pts, data).expect("write NUT frame");
        }
        muxer.finish().expect("flush NUT");
    }
    wire
}

fn yuv_frames() -> Vec<(i64, Vec<u8>)> {
    PTS.iter()
        .enumerate()
        .map(|(i, pts)| (*pts, yuv_frame(i)))
        .collect()
}

/// Every packet of a NUT stream, and the stream header it was read under.
fn read_wire(wire: &[u8]) -> (Stream, Vec<(Packet, Vec<u8>)>) {
    let mut demuxer = Demuxer::open(wire).expect("read the NUT headers");
    let mut packets = Vec::new();
    let mut buf = Vec::new();
    while let Some(packet) = demuxer.read_packet(&mut buf).expect("read a NUT packet") {
        packets.push((packet, buf.clone()));
    }
    (demuxer.stream().clone(), packets)
}

fn encode(raw: &[u8]) -> Vec<u8> {
    let module = module_arg();
    let run = run_ffrwd_wasm(
        &[
            "-codec", "encode", "-m", &module, "-f", "nut", "-i", "pipe:0", "-f", "nut", "pipe:1",
        ],
        raw,
    );
    run.assert_ok("encode");
    run.stdout
}

fn decode(coded: &[u8]) -> Vec<u8> {
    let module = module_arg();
    let run = run_ffrwd_wasm(
        &[
            "-codec", "decode", "-m", &module, "-f", "nut", "-i", "pipe:0", "-f", "nut", "pipe:1",
        ],
        coded,
    );
    run.assert_ok("decode");
    run.stdout
}

/// A path in the temp directory, named after this process and the test, so
/// two tests running at once never share one.
fn scratch(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("ffrwd_codec_{}_{name}", std::process::id()))
}

fn ffmpeg_on_path() -> bool {
    Command::new("ffmpeg")
        .arg("-version")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}

fn announce_skip(what: &str) {
    eprintln!("SKIPPED: {what}. Install ffmpeg and put it on PATH to run this test.");
}

#[test]
fn describe_prints_the_encoder_and_the_decoder() {
    let module = module_arg();
    let run = run_ffrwd_wasm(&["--describe", &module], b"");
    run.assert_ok("describe");
    let text = String::from_utf8(run.stdout).expect("describe is UTF-8");
    assert_eq!(text.trim().lines().count(), 1, "one JSON line:\n{text}");
    let d: serde_json::Value = serde_json::from_str(text.trim()).expect("describe is JSON");

    let encoder_schema = serde_json::json!({
        "type": "object",
        "properties": {"level": {"type": "number", "minimum": 0, "maximum": 9}},
        "additionalProperties": false
    });
    let decoder_schema = serde_json::json!({
        "type": "object",
        "properties": {"pix_fmt": {"type": "string", "enum": ["yuv420p", "gray"]}},
        "additionalProperties": false
    });
    assert_eq!(d["world"], "ffrwd:av@0.18.0");
    assert_eq!(d["name"], "testcodec");
    assert_eq!(d["version"], "0.1.0");
    // The top level is the encoder's, since the module has one.
    assert_eq!(d["params_schema"], encoder_schema);
    assert_eq!(d["pixel_formats"], serde_json::json!(["yuv420p", "gray"]));
    assert_eq!(d["sample_formats"], serde_json::json!([]));
    assert_eq!(
        d["encoder"],
        serde_json::json!({
            "codec": "testcodec",
            "fourcc": "FTST",
            "delay": 0,
            "decode_delay": 0,
            "frame_samples": 0,
            "params_schema": encoder_schema,
            "pixel_formats": ["yuv420p", "gray"],
            "sample_formats": []
        })
    );
    assert_eq!(
        d["decoder"],
        serde_json::json!({
            "fourccs": ["FTST"],
            "delay": 0,
            "params_schema": decoder_schema,
            "pixel_formats": ["yuv420p", "gray"],
            "sample_formats": []
        })
    );
    // A codec is none of the stream interfaces the other flags mark.
    for flag in [
        "packet_sink",
        "packet_filter",
        "source",
        "rows_module",
        "data_filter",
        "gpu",
    ] {
        assert_eq!(d[flag], false, "{flag}");
    }
}

#[test]
fn a_raw_stream_round_trips_byte_for_byte_at_its_own_timestamps() {
    let frames = yuv_frames();
    let coded = encode(&raw_wire("yuv420p", &frames));
    let decoded = decode(&coded);

    let (stream, packets) = read_wire(&decoded);
    assert_eq!(stream.pix_fmt(), Some("yuv420p"));
    assert_eq!(stream.video_geometry(), Some((WIDTH, HEIGHT)));
    assert_eq!(stream.time_base, TIME_BASE);
    let back: Vec<(i64, Vec<u8>)> = packets.into_iter().map(|(p, d)| (p.pts, d)).collect();
    assert_eq!(back.len(), frames.len(), "one frame back per frame in");
    for ((pts, data), (in_pts, in_data)) in back.iter().zip(&frames) {
        assert_eq!(pts, in_pts, "pts preserved");
        assert!(data == in_data, "frame at pts {pts} differs from the input");
    }
}

#[test]
fn the_coded_wire_carries_the_tag_and_the_extradata() {
    let frames = yuv_frames();
    let coded = encode(&raw_wire("yuv420p", &frames));
    let (stream, packets) = read_wire(&coded);
    assert_eq!(stream.fourcc, b"FTST");
    assert_eq!(stream.codec_name(), Some("FTST"));
    assert_eq!(stream.extradata, b"FTST\x01");
    assert_eq!(stream.decode_delay, 0);
    assert_eq!(stream.video_geometry(), Some((WIDTH, HEIGHT)));
    assert_eq!(stream.time_base, TIME_BASE);
    assert_eq!(packets.len(), frames.len());
    for ((packet, data), (pts, _)) in packets.iter().zip(&frames) {
        assert_eq!(packet.pts, *pts);
        assert_eq!(packet.dts, Some(*pts), "no reordering, so dts is pts");
        assert!(packet.keyframe, "every packet is a keyframe");
        assert_eq!(&data[..2], b"FT", "the codec's own header");
        // Run-length coding a frame this flat makes it smaller.
        assert!(data.len() < WIDTH as usize * HEIGHT as usize * 3 / 2);
    }
}

#[test]
fn ffmpeg_reads_the_tag_and_copies_the_stream() {
    if !ffmpeg_on_path() {
        announce_skip("ffmpeg_reads_the_tag_and_copies_the_stream");
        return;
    }
    let coded = encode(&raw_wire("yuv420p", &yuv_frames()));
    let coded_path = scratch("coded.nut");
    let copied_path = scratch("copied.nut");
    std::fs::write(&coded_path, &coded).expect("write the coded stream");

    let probe = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_tag_string,width,height",
            "-of",
            "default=nk=0:nw=1",
            coded_path.to_str().expect("path is UTF-8"),
        ])
        .output()
        .expect("spawn ffprobe");
    let said = String::from_utf8_lossy(&probe.stdout).replace("\r\n", "\n");
    assert!(
        probe.status.success(),
        "ffprobe exited with {:?}\n{}",
        probe.status.code(),
        String::from_utf8_lossy(&probe.stderr)
    );
    assert!(
        said.contains("codec_tag_string=FTST"),
        "ffprobe said:\n{said}"
    );
    assert!(
        said.contains(&format!("width={WIDTH}")),
        "ffprobe said:\n{said}"
    );

    let copy = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            coded_path.to_str().expect("path is UTF-8"),
            "-c",
            "copy",
            copied_path.to_str().expect("path is UTF-8"),
        ])
        .output()
        .expect("spawn ffmpeg");
    let _ = std::fs::remove_file(&coded_path);
    let copied = std::fs::read(&copied_path);
    let _ = std::fs::remove_file(&copied_path);
    assert!(
        copy.status.success(),
        "ffmpeg -c copy exited with {:?}\n{}",
        copy.status.code(),
        String::from_utf8_lossy(&copy.stderr)
    );
    // What ffmpeg copied still decodes back to the input through the codec.
    let copied = copied.expect("ffmpeg wrote the copy");
    let (stream, packets) = read_wire(&copied);
    assert_eq!(stream.codec_name(), Some("FTST"));
    assert_eq!(packets.len(), PTS.len());
}

#[test]
fn a_codec_module_with_both_halves_needs_codec() {
    let module = module_arg();
    let run = run_ffrwd_wasm(
        &[
            "-m", &module, "-f", "nut", "-i", "pipe:0", "-f", "nut", "pipe:1",
        ],
        &raw_wire("yuv420p", &yuv_frames()),
    );
    run.assert_refused(
        "a codec module without -codec",
        &[
            "exports both an encoder and a decoder",
            "-codec encode",
            "-codec decode",
        ],
    );
}

#[test]
fn codec_names_encode_or_decode_and_nothing_else() {
    let module = module_arg();
    let run = run_ffrwd_wasm(
        &[
            "-codec",
            "transcode",
            "-m",
            &module,
            "-f",
            "nut",
            "-i",
            "pipe:0",
            "-f",
            "nut",
            "pipe:1",
        ],
        b"",
    );
    run.assert_refused(
        "-codec transcode",
        &["-codec transcode", "encode and decode"],
    );
}

#[test]
fn a_stream_under_another_tag_is_refused_by_the_decoder() {
    let h264 = std::fs::read(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/h264.nut"))
        .expect("read the h264 fixture");
    let module = module_arg();
    let run = run_ffrwd_wasm(
        &[
            "-codec", "decode", "-m", &module, "-f", "nut", "-i", "pipe:0", "-f", "nut", "pipe:1",
        ],
        &h264,
    );
    run.assert_refused("decoding an h264 stream", &["testcodec", "FTST", "H264"]);
}

#[test]
fn a_raw_stream_is_refused_by_the_decoder() {
    let module = module_arg();
    let run = run_ffrwd_wasm(
        &[
            "-codec", "decode", "-m", &module, "-f", "nut", "-i", "pipe:0", "-f", "nut", "pipe:1",
        ],
        &raw_wire("yuv420p", &yuv_frames()),
    );
    run.assert_refused("decoding a raw stream", &["testcodec", "raw yuv420p"]);
}

#[test]
fn a_pixel_format_the_encoder_does_not_list_is_refused() {
    let rgba: Vec<(i64, Vec<u8>)> = vec![(0, vec![7u8; (WIDTH * HEIGHT * 4) as usize])];
    let module = module_arg();
    let run = run_ffrwd_wasm(
        &[
            "-codec", "encode", "-m", &module, "-f", "nut", "-i", "pipe:0", "-f", "nut", "pipe:1",
        ],
        &raw_wire("rgba", &rgba),
    );
    run.assert_refused("encoding rgba", &["testcodec", "rgba", "yuv420p, gray"]);
}

#[test]
fn a_coded_stream_is_refused_by_the_encoder() {
    let coded = encode(&raw_wire("yuv420p", &yuv_frames()));
    let module = module_arg();
    let run = run_ffrwd_wasm(
        &[
            "-codec", "encode", "-m", &module, "-f", "nut", "-i", "pipe:0", "-f", "nut", "pipe:1",
        ],
        &coded,
    );
    run.assert_refused(
        "encoding a coded stream",
        &["testcodec", "raw frames", "FTST"],
    );
}

#[test]
fn the_encoders_own_refusal_names_it() {
    let module = module_arg();
    let run = run_ffrwd_wasm(
        &[
            "-codec",
            "encode",
            "-m",
            &module,
            "-params",
            r#"{"level": 12}"#,
            "-f",
            "nut",
            "-i",
            "pipe:0",
            "-f",
            "nut",
            "pipe:1",
        ],
        &raw_wire("yuv420p", &yuv_frames()),
    );
    run.assert_refused("level 12", &["testcodec refused to open", "level"]);
}

#[test]
fn codec_is_refused_for_a_module_that_is_not_a_codec() {
    codec_module();
    let invert = sidecar_root().join("modules/target/wasm32-wasip2/release/invert.wasm");
    let invert = invert.to_str().expect("path is UTF-8");
    let run = run_ffrwd_wasm(
        &[
            "-codec", "encode", "-m", invert, "-f", "nut", "-i", "pipe:0", "-f", "nut", "pipe:1",
        ],
        &raw_wire("yuv420p", &yuv_frames()),
    );
    run.assert_refused(
        "-codec on a frame filter",
        &["-codec encode", "neither an encoder nor a decoder"],
    );
}

#[test]
fn files_work_as_well_as_pipes_and_jobs_is_accepted() {
    let raw_path = scratch("raw.nut");
    let coded_path = scratch("coded_file.nut");
    let back_path = scratch("back.nut");
    let frames = yuv_frames();
    std::fs::write(&raw_path, raw_wire("yuv420p", &frames)).expect("write raw input");
    let module = module_arg();
    let path = |p: &Path| p.to_str().expect("path is UTF-8").to_string();
    let (raw_arg, coded_arg, back_arg) = (path(&raw_path), path(&coded_path), path(&back_path));
    run_ffrwd_wasm(
        &[
            "-jobs", "4", "-codec", "encode", "-m", &module, "-f", "nut", "-i", &raw_arg, "-f",
            "nut", &coded_arg,
        ],
        b"",
    )
    .assert_ok("encode to a file");
    run_ffrwd_wasm(
        &[
            "-codec", "decode", "-m", &module, "-f", "nut", "-i", &coded_arg, "-f", "nut",
            &back_arg,
        ],
        b"",
    )
    .assert_ok("decode from a file");
    let back = std::fs::read(&back_path).expect("read the decoded file");
    for p in [&raw_path, &coded_path, &back_path] {
        let _ = std::fs::remove_file(p);
    }
    let (_, packets) = read_wire(&back);
    let back: Vec<(i64, Vec<u8>)> = packets.into_iter().map(|(p, d)| (p.pts, d)).collect();
    assert!(back == frames, "the file round trip differs from the input");
}

#[test]
fn a_nominal_frame_rate_is_taken_on_an_encode_run() {
    let frames = yuv_frames();
    let module = module_arg();
    let run = run_ffrwd_wasm(
        &[
            "-codec",
            "encode",
            "-frame_rate",
            "30000/1001",
            "-m",
            &module,
            "-f",
            "nut",
            "-i",
            "pipe:0",
            "-f",
            "nut",
            "pipe:1",
        ],
        &raw_wire("yuv420p", &frames),
    );
    run.assert_ok("encode with -frame_rate");
    let (_, packets) = read_wire(&decode(&run.stdout));
    let back: Vec<(i64, Vec<u8>)> = packets.into_iter().map(|(p, d)| (p.pts, d)).collect();
    assert!(back == frames, "the round trip differs from the input");
}

#[test]
fn a_malformed_frame_rate_is_refused_by_name() {
    let module = module_arg();
    for bad in ["30", "30/0", "-30/1", "thirty/1", "30/1/1"] {
        let run = run_ffrwd_wasm(
            &[
                "-codec",
                "encode",
                "-frame_rate",
                bad,
                "-m",
                &module,
                "-f",
                "nut",
                "-i",
                "pipe:0",
                "-f",
                "nut",
                "pipe:1",
            ],
            b"",
        );
        run.assert_refused(&format!("-frame_rate {bad}"), &["-frame_rate", bad]);
    }
}

#[test]
fn a_frame_rate_on_a_decode_run_is_refused() {
    let coded = encode(&raw_wire("yuv420p", &yuv_frames()));
    let module = module_arg();
    let run = run_ffrwd_wasm(
        &[
            "-codec",
            "decode",
            "-frame_rate",
            "25/1",
            "-m",
            &module,
            "-f",
            "nut",
            "-i",
            "pipe:0",
            "-f",
            "nut",
            "pipe:1",
        ],
        &coded,
    );
    run.assert_refused("-frame_rate on a decode run", &["-frame_rate", "decodes"]);
}
