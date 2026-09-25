//! Data streams through the packet interfaces: a JSON track a packet source
//! publishes, written as a JSON NUT; the same NUT handed to a packet sink
//! and a packet filter as a data pad; and refused, by name, at a module
//! built against a world with no data arm. Each data edge also carries
//! heartbeats, packets holding a single space saying how far time has got, which no module
//! is ever handed.
//!
//! `source_replay_data` publishes the messages mirrored in `MESSAGES` (see
//! `modules/source-replay-data/src/lib.rs`), chosen for what a wire could
//! lose: two at one pts, a character outside ASCII, a gap NUT codes as a
//! full pts, and bytes that are not the canonical spelling of their JSON.
//! Every check below is byte for byte against them.

use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::sync::OnceLock;

use ffrwd_wasm::nut::{Demuxer, TimeBase};

/// `source_replay_data`'s own messages, in pts order.
const MESSAGES: &[(i64, &str)] = &[
    (0, r#"{"kind":"start","n":0}"#),
    (40_000, r#"{"n":1,"text":"café"}"#),
    (40_000, r#"{"n":2,"text":"café"}"#),
    (3_500_000, r#"{ "n" : 3 , "spaced" : true }"#),
    (3_500_001, r#"{"n":4,"last":true}"#),
];

/// The unit the source counts its messages in.
const MICROS: TimeBase = TimeBase {
    num: 1,
    den: 1_000_000,
};

fn sidecar_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("ffrwd-wasm/ has a parent directory")
        .to_path_buf()
}

fn h264_fixture() -> PathBuf {
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
                "source-replay-data",
                "-p",
                "data-echo",
                "-p",
                "packet-passthrough",
                "-p",
                "packet-passthrough-0160",
                "-p",
                "packet-stats",
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
    stdout: Vec<u8>,
    stderr: String,
    output: Output,
}

/// Runs `ffrwd-wasm` with the given argv and nothing on stdin.
fn run_ffrwd_wasm(args: &[&str]) -> Run {
    let exe = env!("CARGO_BIN_EXE_ffrwd-wasm");
    let mut child = Command::new(exe)
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn ffrwd-wasm");
    drop(child.stdin.take());
    let output = child.wait_with_output().expect("wait for ffrwd-wasm");
    Run {
        stdout: output.stdout.clone(),
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        output,
    }
}

fn assert_ok(run: &Run, what: &str) {
    assert!(
        run.output.status.success(),
        "{what} exited with {:?}\nstderr:\n{}",
        run.output.status.code(),
        run.stderr
    );
}

/// A path in the temp directory, named after this process and the test, so
/// two tests running at once never share one.
fn scratch(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("ffrwd_data_streams_{}_{name}", std::process::id()))
}

fn path_str(path: &Path) -> &str {
    path.to_str().expect("path is UTF-8")
}

/// Every packet of a NUT stream as (pts, dts, keyframe, bytes).
fn read_packets(wire: &[u8]) -> Vec<(i64, Option<i64>, bool, Vec<u8>)> {
    let mut demuxer = Demuxer::open(wire).expect("read the NUT headers");
    let mut packets = Vec::new();
    let mut buf = Vec::new();
    while let Some(packet) = demuxer.read_packet(&mut buf).expect("read a NUT packet") {
        packets.push((packet.pts, packet.dts, packet.keyframe, buf.clone()));
    }
    packets
}

/// Whether a packet is a heartbeat: nothing but whitespace.
fn is_heartbeat(data: &[u8]) -> bool {
    data.iter().all(u8::is_ascii_whitespace)
}

/// The pts of every heartbeat a JSON NUT carries.
fn heartbeats(wire: &[u8]) -> Vec<i64> {
    read_packets(wire)
        .into_iter()
        .filter(|(_, _, _, data)| is_heartbeat(data))
        .map(|(pts, _, _, _)| pts)
        .collect()
}

/// The source's data track, written to `path` as a JSON NUT.
fn write_source_track(path: &Path) {
    let module = module_path("source_replay_data");
    let run = run_ffrwd_wasm(&[
        "-m",
        path_str(&module),
        "-track",
        "0",
        "-f",
        "nut",
        path_str(path),
    ]);
    assert_ok(&run, "source_replay_data");
}

/// Checks a JSON NUT carries exactly `MESSAGES`: a data stream in
/// microseconds, each message its own keyframe packet at its own pts, bytes
/// untouched. Heartbeats between them are not messages.
fn assert_carries_the_messages(wire: &[u8]) {
    let demuxer = Demuxer::open(wire).expect("read the NUT headers");
    assert!(
        demuxer.stream().is_json(),
        "the stream is a JSON data stream"
    );
    assert_eq!(demuxer.stream().codec_name(), Some("json"));
    assert_eq!(demuxer.stream().time_base, MICROS);

    let packets: Vec<_> = read_packets(wire)
        .into_iter()
        .filter(|(_, _, _, data)| !is_heartbeat(data))
        .collect();
    assert_eq!(packets.len(), MESSAGES.len(), "one packet per message");
    for (index, ((pts, dts, keyframe, data), (want_pts, want))) in
        packets.iter().zip(MESSAGES).enumerate()
    {
        assert_eq!(pts, want_pts, "message {index} pts");
        assert_eq!(*dts, Some(*want_pts), "message {index} dts is its pts");
        assert!(keyframe, "message {index} is a keyframe");
        assert_eq!(data.as_slice(), want.as_bytes(), "message {index} bytes");
    }
}

#[test]
fn a_sources_data_track_is_written_as_a_json_nut() {
    let path = scratch("source.nut");
    write_source_track(&path);
    let wire = std::fs::read(&path).expect("read what the source wrote");
    std::fs::remove_file(&path).ok();
    assert_carries_the_messages(&wire);
    // No media moves time on, so the one heartbeat is the start's.
    assert_eq!(heartbeats(&wire), vec![0]);
}

#[test]
fn a_sources_data_track_beats_while_its_picture_moves_on() {
    // With the picture subscribed, a pull is one picture a tenth of a second
    // on and the messages up to it: the track says where time has got to at
    // the start and whenever it has been quiet for a tenth of a second.
    let data = scratch("beats_data.nut");
    let video = scratch("beats_video.nut");
    let module = module_path("source_replay_data");
    let run = run_ffrwd_wasm(&[
        "-m",
        path_str(&module),
        "-track",
        "0",
        "-f",
        "nut",
        path_str(&data),
        "-track",
        "1",
        "-f",
        "nut",
        path_str(&video),
    ]);
    let wire = std::fs::read(&data).expect("read the data track");
    std::fs::remove_file(&data).ok();
    std::fs::remove_file(&video).ok();
    assert_ok(&run, "source_replay_data");
    assert_carries_the_messages(&wire);
    let quiet = (2..=34).map(|tenth| tenth * 100_000);
    assert_eq!(
        heartbeats(&wire),
        std::iter::once(0).chain(quiet).collect::<Vec<i64>>()
    );
}

#[test]
fn a_probed_data_track_says_it_is_one() {
    let module = module_path("source_replay_data");
    let run = run_ffrwd_wasm(&["--probe", path_str(&module)]);
    assert_ok(&run, "--probe");
    let catalog: serde_json::Value =
        serde_json::from_slice(&run.stdout).expect("--probe prints one JSON object");
    let track = &catalog["tracks"][0];
    assert_eq!(track["kind"], "data");
    assert_eq!(track["codec"], "json");
    assert_eq!(track["format"], serde_json::json!({"data": {}}));
    assert_eq!(track["time_base"], serde_json::json!([1, 1_000_000]));
}

#[test]
fn a_json_nut_reaches_a_sink_as_a_data_pad() {
    let path = scratch("to_sink.nut");
    write_source_track(&path);
    // The sink is handed the messages and never the heartbeat among them.
    let wire = std::fs::read(&path).expect("read what the source wrote");
    assert_eq!(heartbeats(&wire), vec![0]);
    let module = module_path("data_echo");
    let run = run_ffrwd_wasm(&[
        "-f",
        "nut",
        "-i",
        path_str(&path),
        "-m",
        path_str(&module),
        "-f",
        "ndjson",
        "-",
    ]);
    std::fs::remove_file(&path).ok();
    assert_ok(&run, "data_echo");

    let rows: Vec<serde_json::Value> = String::from_utf8_lossy(&run.stdout)
        .lines()
        .map(|line| serde_json::from_str(line).expect("each row is JSON"))
        .collect();
    assert_eq!(rows.len(), MESSAGES.len(), "one row per message");
    for (index, (row, (pts, message))) in rows.iter().zip(MESSAGES).enumerate() {
        assert_eq!(
            row,
            &serde_json::json!({
                "pad": 0,
                "codec": "json",
                "pts": pts,
                "dts": pts,
                "keyframe": true,
                "message": message,
            }),
            "message {index}"
        );
    }
}

#[test]
fn a_filter_hands_a_data_pad_back_beside_a_coded_one() {
    // Video first, then data, the order the pads are handed over in: each
    // leaves on its own output, the h264 as it arrived and the messages as
    // a JSON NUT carrying exactly what the source wrote.
    let messages = scratch("filter_in.nut");
    write_source_track(&messages);
    let video_out = scratch("filter_video.nut");
    let data_out = scratch("filter_data.nut");
    let module = module_path("packet_passthrough");
    let fixture = h264_fixture();
    let run = run_ffrwd_wasm(&[
        "-f",
        "nut",
        "-i",
        path_str(&fixture),
        "-f",
        "nut",
        "-i",
        path_str(&messages),
        "-m",
        path_str(&module),
        "-f",
        "nut",
        path_str(&video_out),
        "-f",
        "nut",
        path_str(&data_out),
        "-f",
        "ndjson",
        "-",
    ]);
    std::fs::remove_file(&messages).ok();
    let video = std::fs::read(&video_out).expect("read the video output");
    let data = std::fs::read(&data_out).expect("read the data output");
    std::fs::remove_file(&video_out).ok();
    std::fs::remove_file(&data_out).ok();
    assert_ok(&run, "packet_passthrough");

    assert_carries_the_messages(&data);
    let fixture_bytes = std::fs::read(&fixture).expect("read the h264 fixture");
    assert_eq!(read_packets(&video), read_packets(&fixture_bytes));

    // The filter's own tally says the data pad was opened as one, and it
    // was handed the messages and not the heartbeat among them.
    let tallies: Vec<serde_json::Value> = String::from_utf8_lossy(&run.stdout)
        .lines()
        .map(|line| serde_json::from_str(line).expect("each row is JSON"))
        .collect();
    assert_eq!(tallies.len(), 2);
    assert_eq!(tallies[1]["pad"], 1);
    assert_eq!(tallies[1]["packets"], MESSAGES.len());
    assert_eq!(tallies[1]["decode_delay"], 0);
}

#[test]
fn a_data_stream_is_refused_where_it_cannot_go() {
    let path = scratch("refused.nut");
    write_source_track(&path);

    // A filter built against a world with no data arm, named with the world.
    let older = module_path("packet_passthrough_0160");
    let out = scratch("refused_out.nut");
    let run = run_ffrwd_wasm(&[
        "-f",
        "nut",
        "-i",
        path_str(&path),
        "-m",
        path_str(&older),
        "-f",
        "nut",
        path_str(&out),
    ]);
    std::fs::remove_file(&out).ok();
    assert!(!run.output.status.success());
    assert!(
        run.stderr.contains(
            "packet_passthrough_0160 is built against ffrwd:av@0.16.0, which carries no data \
             stream; rebuild it against ffrwd:av@0.17.0"
        ),
        "{}",
        run.stderr
    );

    // A current sink that declares it reads none, handed the video it does
    // read and the messages beside it.
    let sink = module_path("packet_stats");
    let fixture = h264_fixture();
    let run = run_ffrwd_wasm(&[
        "-f",
        "nut",
        "-i",
        path_str(&fixture),
        "-f",
        "nut",
        "-i",
        path_str(&path),
        "-m",
        path_str(&sink),
        "-f",
        "ndjson",
        "-",
    ]);
    std::fs::remove_file(&path).ok();
    assert!(!run.output.status.success());
    assert!(
        run.stderr
            .contains("packet_stats reads no data stream, and this query hands it 1"),
        "{}",
        run.stderr
    );
}

#[test]
fn describe_reports_how_many_data_streams_a_module_reads() {
    for (name, streams) in [("packet_passthrough", "any"), ("data_echo", "many")] {
        let module = module_path(name);
        let run = run_ffrwd_wasm(&["--describe", path_str(&module)]);
        assert_ok(&run, "--describe");
        let description: serde_json::Value =
            serde_json::from_slice(&run.stdout).expect("describe prints one JSON object");
        assert_eq!(description["data_streams"], streams, "{name}");
    }
}
