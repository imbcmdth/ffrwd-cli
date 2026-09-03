//! The encoded edge: the NUT reader over a stream ffmpeg itself muxed, and a
//! packet sink driven over it.
//!
//! `tests/data/h264.nut` was generated once with
//! `ffmpeg -f lavfi -i testsrc2=size=64x48:rate=25:duration=2 -c:v libx264
//! -g 30 -pix_fmt yuv420p -f nut tests/data/h264.nut`. Every expectation
//! below is pinned from ffprobe's account of that same file - packet count,
//! keyframe positions, timestamps, sizes, extradata - never from this
//! parser's own output.

use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Output, Stdio};
use std::sync::OnceLock;

use ffrwd_wasm::nut::{Demuxer, Muxer, Packet, Stream, TimeBase};

/// The committed fixture, one encoded h264 stream in NUT.
const FIXTURE: &[u8] = include_bytes!("data/h264.nut");

/// ffprobe: 50 packets, 4250 bytes across them.
const PACKETS: usize = 50;
const TOTAL_BYTES: usize = 4250;

/// ffprobe: the packets flagged K sit at these indices, at these timestamps.
const KEYFRAMES: &[(usize, i64)] = &[(0, 4096), (30, 65536)];

/// ffprobe: pts and dts of the first packets, in the order they arrive. The
/// first two dts are N/A - the wire does not settle them until the reorder
/// buffer of `has_b_frames = 2` entries fills.
const FIRST_TIMESTAMPS: &[(i64, Option<i64>)] = &[
    (4096, None),
    (12288, None),
    (8192, Some(4096)),
    (6144, Some(6144)),
    (10240, Some(8192)),
    (20480, Some(10240)),
    (16384, Some(12288)),
    (14336, Some(14336)),
];

/// ffprobe: the last packet.
const LAST_TIMESTAMP: (i64, Option<i64>) = (104448, Some(100352));

/// ffprobe: the stream's extradata, 38 bytes of SPS and PPS.
const EXTRADATA: &[u8] = &[
    0x00, 0x00, 0x00, 0x01, 0x67, 0x64, 0x00, 0x0a, 0xac, 0xd9, 0x44, 0x7b, 0x01, 0x10, 0x00, 0x00,
    0x03, 0x00, 0x10, 0x00, 0x00, 0x03, 0x03, 0x20, 0xf1, 0x22, 0x59, 0x60, 0x00, 0x00, 0x00, 0x01,
    0x68, 0xeb, 0xe3, 0xcb, 0x22, 0xc0,
];

fn sidecar_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("ffrwd-wasm/ has a parent directory")
        .to_path_buf()
}

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/h264.nut")
}

/// Absolute path to a module's built `.wasm` component, built once per test
/// binary. `modules/` is a separate cargo workspace with its own build lock,
/// so this does not deadlock against the `cargo test` run driving this
/// binary.
fn module_path(name: &str) -> PathBuf {
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
                "packet-stats",
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
    sidecar_root()
        .join("modules/target/wasm32-wasip2/release")
        .join(format!("{name}.wasm"))
}

struct Run {
    stdout: String,
    stderr: String,
    output: Output,
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
        stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        output,
    }
}

/// Whether ffmpeg is on PATH, so the one test here that shells out to real
/// ffmpeg can skip rather than fail where it is not installed.
fn ffmpeg_on_path() -> bool {
    Command::new("ffmpeg")
        .arg("-version")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}

/// Says why a test did nothing, since a skip is otherwise silent.
fn announce_skip(what: &str) {
    eprintln!("SKIPPED: {what}. Install ffmpeg and put it on PATH to run this test.");
}

/// Every packet of the fixture, as the reader hands them out.
fn read_fixture() -> Vec<(Packet, Vec<u8>)> {
    let mut demuxer = Demuxer::open(FIXTURE).expect("read the fixture's NUT headers");
    let mut packets = Vec::new();
    let mut buf = Vec::new();
    while let Some(packet) = demuxer.read_packet(&mut buf).expect("read a NUT packet") {
        packets.push((packet, buf.clone()));
    }
    packets
}

#[test]
fn the_stream_header_matches_ffprobes_account() {
    let demuxer = Demuxer::open(FIXTURE).expect("read the fixture's NUT headers");
    let stream = demuxer.stream();
    assert_eq!(stream.codec_name(), Some("h264"));
    assert_eq!(stream.video_geometry(), Some((64, 48)));
    assert_eq!(stream.time_base, TimeBase { num: 1, den: 51200 });
    // ffprobe: has_b_frames=2.
    assert_eq!(stream.decode_delay, 2);
    assert_eq!(stream.extradata, EXTRADATA);
}

#[test]
fn the_packets_match_ffprobes_account() {
    let packets = read_fixture();
    assert_eq!(packets.len(), PACKETS);

    let keyframes: Vec<(usize, i64)> = packets
        .iter()
        .enumerate()
        .filter(|(_, (packet, _))| packet.keyframe)
        .map(|(index, (packet, _))| (index, packet.pts))
        .collect();
    assert_eq!(keyframes, KEYFRAMES);

    let timestamps: Vec<(i64, Option<i64>)> = packets
        .iter()
        .map(|(packet, _)| (packet.pts, packet.dts))
        .collect();
    assert_eq!(&timestamps[..FIRST_TIMESTAMPS.len()], FIRST_TIMESTAMPS);
    assert_eq!(*timestamps.last().expect("fifty packets"), LAST_TIMESTAMP);

    // dts never decreases once the wire settles it.
    let settled: Vec<i64> = timestamps.iter().filter_map(|(_, dts)| *dts).collect();
    assert!(settled.windows(2).all(|pair| pair[0] <= pair[1]));

    let total: usize = packets.iter().map(|(_, data)| data.len()).sum();
    assert_eq!(total, TOTAL_BYTES);
}

#[test]
fn packet_stats_reports_the_groups_ffprobe_counted() {
    let module = module_path("packet_stats");
    let run = run_ffrwd_wasm(
        &[
            "-f",
            "nut",
            "-i",
            fixture_path().to_str().expect("fixture path is UTF-8"),
            "-m",
            module.to_str().expect("module path is UTF-8"),
            "-f",
            "ndjson",
            "-",
        ],
        &[],
    );
    assert!(
        run.output.status.success(),
        "packet_stats exited with {:?}\nstderr:\n{}",
        run.output.status.code(),
        run.stderr
    );
    // ffprobe: packets 0-29 total 2687 bytes, packets 30-49 total 1563; the
    // pts sequence steps backwards wherever a B-frame follows its references.
    let rows: Vec<&str> = run.stdout.lines().collect();
    assert_eq!(
        rows,
        vec![
            r#"{"gop":0,"pts":4096,"packets":30,"bytes":2687}"#,
            r#"{"gop":1,"pts":65536,"packets":20,"bytes":1563}"#,
            r#"{"packets":50,"keyframes":2,"bytes":4250,"gops":2,"pts_monotonic":false}"#,
        ]
    );
}

#[test]
fn an_encoded_stream_is_refused_by_a_frame_module() {
    let module = module_path("invert");
    let run = run_ffrwd_wasm(
        &[
            "-f",
            "nut",
            "-i",
            "-",
            "-m",
            module.to_str().expect("module path is UTF-8"),
            "-f",
            "nut",
            "-",
        ],
        FIXTURE,
    );
    assert!(!run.output.status.success());
    assert!(
        run.stderr.contains("input carries encoded h264") && run.stderr.contains("packet sink"),
        "stderr does not name the encoded stream:\n{}",
        run.stderr
    );
}

#[test]
fn a_raw_stream_is_refused_by_a_packet_sink() {
    let stream =
        Stream::video("rgba", 2, 2, TimeBase { num: 1, den: 25 }).expect("rgba is carried");
    let mut wire = Vec::new();
    {
        let mut muxer = Muxer::new(&mut wire, &stream).expect("write NUT headers");
        muxer
            .write_frame(0, &[0u8; 16])
            .expect("write one raw frame");
        muxer.finish().expect("finish the NUT stream");
    }

    let module = module_path("packet_stats");
    let run = run_ffrwd_wasm(
        &[
            "-f",
            "nut",
            "-i",
            "-",
            "-m",
            module.to_str().expect("module path is UTF-8"),
            "-f",
            "ndjson",
            "-",
        ],
        &wire,
    );
    assert!(!run.output.status.success());
    assert!(
        run.stderr.contains("consumes encoded packets")
            && run.stderr.contains("decoded rgba video"),
        "stderr does not name what the input carries:\n{}",
        run.stderr
    );
}

#[test]
fn a_packet_sink_refuses_a_frame_output() {
    let module = module_path("packet_stats");
    let run = run_ffrwd_wasm(
        &[
            "-f",
            "nut",
            "-i",
            "-",
            "-m",
            module.to_str().expect("module path is UTF-8"),
            "-f",
            "nut",
            "-",
        ],
        FIXTURE,
    );
    assert!(!run.output.status.success());
    assert!(
        run.stderr.contains("a packet sink emits rows alone"),
        "stderr does not name the refused output:\n{}",
        run.stderr
    );
}

#[test]
fn a_packet_sink_joins_no_network() {
    let module = module_path("packet_stats");
    let binding = format!("stats={}", module.to_str().expect("module path is UTF-8"));
    let run = run_ffrwd_wasm(
        &[
            "-f",
            "nut",
            "-i",
            "-",
            "-m",
            &binding,
            "-filter_complex",
            "[0]stats[out0]",
            "-map",
            "[out0]",
            "-f",
            "ndjson",
            "-",
        ],
        FIXTURE,
    );
    assert!(!run.output.status.success());
    assert!(
        run.stderr.contains("joins no network"),
        "stderr does not refuse the network:\n{}",
        run.stderr
    );
}

#[test]
fn real_ffmpeg_accepts_a_stream_write_coded_wrote() {
    // The other tests here prove this crate's own Demuxer reads back what
    // write_coded writes; this one proves real ffmpeg's NUT demuxer does
    // too, over packets ffmpeg itself encoded and this reader parsed off the
    // fixture - not synthetic bytes.
    if !ffmpeg_on_path() {
        announce_skip("real ffmpeg cannot demux a write_coded stream");
        return;
    }

    // Read order off a NUT stream is decode order, which is exactly the
    // order write_coded wants them handed back in. A prefix that spans both
    // keyframes carries the mid-GOP B-frame reordering as well as the cut
    // from one GOP to the next.
    let packets: Vec<(Packet, Vec<u8>)> = read_fixture().into_iter().take(35).collect();
    let stream = Demuxer::open(FIXTURE)
        .expect("read the fixture's NUT headers")
        .stream()
        .clone();

    let mut wire = Vec::new();
    {
        let mut muxer = Muxer::new(&mut wire, &stream).expect("write headers");
        for (packet, data) in &packets {
            muxer.write_coded(packet, data).expect("write coded packet");
        }
        muxer.finish().expect("finish");
    }

    let mut child = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "nut",
            "-i",
            "-",
            "-c",
            "copy",
            "-f",
            "null",
            "-",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn ffmpeg");
    child
        .stdin
        .take()
        .expect("ffmpeg stdin")
        .write_all(&wire)
        .expect("write the NUT stream to ffmpeg's stdin");
    let output = child.wait_with_output().expect("wait for ffmpeg");
    assert!(
        output.status.success(),
        "ffmpeg exited with {:?}\nstderr:\n{}",
        output.status.code(),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn describe_reports_the_packet_sink() {
    let module = module_path("packet_stats");
    let run = run_ffrwd_wasm(
        &["--describe", module.to_str().expect("module path is UTF-8")],
        &[],
    );
    assert!(
        run.output.status.success(),
        "describe exited with {:?}\nstderr:\n{}",
        run.output.status.code(),
        run.stderr
    );
    let description: serde_json::Value =
        serde_json::from_str(run.stdout.trim()).expect("describe prints one JSON object");
    assert_eq!(description["world"], "ffrwd:av@0.13.0");
    assert_eq!(description["name"], "packet_stats");
    // The codecs list is what reports the packet-sink export; empty accepts
    // every codec.
    assert_eq!(description["video_codecs"], serde_json::json!([]));
    // No frame interface: none of the windowed fields appear.
    assert!(description.get("window").is_none());
    assert_eq!(description["reads_rows"], serde_json::Value::Null);
}
