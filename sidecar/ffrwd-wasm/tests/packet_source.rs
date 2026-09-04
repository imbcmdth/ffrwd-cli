//! Integration tests for the sidecar's packet-source HEAD mode: no `-i`, one
//! `-track <index> -f nut` output per track the command reads, driven by
//! `source_replay` - a fixture
//! module that hands back a fixed, compiled-in set of coded packets (see
//! `modules/source-replay/src/lib.rs` for why it is compiled in rather than
//! read from a file: the wasm sandbox these modules run in has no
//! filesystem).

use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use ffrwd_wasm::nut::Demuxer;

/// Sidecar root, the parent of `ffrwd-wasm/`.
fn sidecar_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("ffrwd-wasm/ has a parent directory")
        .to_path_buf()
}

/// Absolute path to `source_replay`'s built `.wasm` component, built once
/// per test binary. `modules/` is a separate cargo workspace with its own
/// build lock, so this does not deadlock against the `cargo test` run
/// driving this binary.
fn built_module(name: &str) -> PathBuf {
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
                "source-replay",
                "-p",
                "source-replay-pair",
                "-p",
                "source-replay-0130",
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

fn module_path() -> PathBuf {
    built_module("source_replay")
}

/// The two-track fixture, whose track 0 pours out more than a buffered writer
/// holds before track 1 says anything at all.
fn pair_module_path() -> PathBuf {
    built_module("source_replay_pair")
}

/// The same fixture built against the vendored `worlds/0.13.0`, whose `open`
/// takes params alone: what a packet source looked like before it was told
/// which tracks to pull.
fn older_module_path() -> PathBuf {
    built_module("source_replay_0130")
}

struct Run {
    stdout: Vec<u8>,
    stderr: String,
    output: Output,
}

/// Runs `ffrwd-wasm` with the given argv and no stdin - a packet source
/// takes none.
fn run_ffrwd_wasm(args: &[&str]) -> Run {
    let exe = env!("CARGO_BIN_EXE_ffrwd-wasm");
    let output = Command::new(exe)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .expect("spawn ffrwd-wasm");
    Run {
        stdout: output.stdout.clone(),
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        output,
    }
}

/// `source_replay`'s own packets, mirrored from
/// `modules/source-replay/src/lib.rs`. The module cannot be a host
/// dependency (it targets wasm32-wasip2), so what it publishes is pinned
/// here the same way any other fixture's expectations are pinned from an
/// outside description, not from output.
const EXPECTED: &[(i64, Option<i64>, bool)] = &[
    (0, None, true),
    (3, None, false),
    (1, Some(0), false),
    (2, Some(1), false),
    (6, Some(2), false),
    (4, Some(3), false),
    (5, Some(4), false),
];

const EXPECTED_EXTRADATA: &[u8] = &[
    0x00, 0x00, 0x00, 0x01, 0x67, 0x64, 0x00, 0x0a, 0xac, 0xd9, 0x49, 0x7e, 0x5c, 0x04, 0x40, 0x00,
    0x00, 0x03, 0x00, 0x40, 0x00, 0x00, 0x0c, 0x83, 0xc4, 0x89, 0x65, 0x80, 0x00, 0x00, 0x00, 0x01,
    0x68, 0xef, 0x8f, 0x2c, 0x8b,
];

/// `RAW` is the module's own checked-in fixture, `sidecar/modules/source-
/// replay/src/generated/packets.bin` (built by `cli/scripts/
/// gen_replay_packets.py`) - extradata followed by the seven packets' coded
/// bytes, back to back. Reading the exact same file the module `include!`s
/// keeps this test from re-typing over a kilobyte of coded video by hand;
/// `EXTRADATA_LEN` and `PACKET_LENS` are the small facts that describe how
/// to slice it, pinned the same way `EXPECTED` and `EXPECTED_EXTRADATA` are.
const RAW: &[u8] = include_bytes!("../../modules/source-replay/src/generated/packets.bin");
const EXTRADATA_LEN: usize = 37;
const PACKET_LENS: &[usize] = &[1077, 12, 12, 12, 13, 14, 12];

/// The real coded bytes for packet `index`, decode order, sliced out of
/// `RAW` the same way the module itself slices it.
fn expected_packet_bytes(index: usize) -> &'static [u8] {
    let start = EXTRADATA_LEN + PACKET_LENS[..index].iter().sum::<usize>();
    &RAW[start..start + PACKET_LENS[index]]
}

#[test]
fn a_packet_source_refuses_an_input() {
    let module = module_path();
    let run = run_ffrwd_wasm(&[
        "-f",
        "nut",
        "-i",
        "-",
        "-m",
        module.to_str().expect("module path is UTF-8"),
        "-track",
        "0",
        "-f",
        "nut",
        "-",
    ]);
    assert!(!run.output.status.success());
    assert!(
        run.stderr.contains("packet source") && run.stderr.contains("no -i input"),
        "stderr does not refuse the input:\n{}",
        run.stderr
    );
}

#[test]
fn an_output_naming_no_track_is_refused() {
    let module = module_path();
    let run = run_ffrwd_wasm(&[
        "-m",
        module.to_str().expect("module path is UTF-8"),
        "-f",
        "nut",
        "-",
    ]);
    assert!(!run.output.status.success());
    assert!(
        run.stderr.contains("writes the tracks it is told to") && run.stderr.contains("-track"),
        "stderr does not say how to name the track:\n{}",
        run.stderr
    );
}

#[test]
fn a_track_the_catalog_does_not_publish_is_refused() {
    let module = module_path();
    let run = run_ffrwd_wasm(&[
        "-m",
        module.to_str().expect("module path is UTF-8"),
        "-track",
        "1",
        "-f",
        "nut",
        "-",
    ]);
    assert!(!run.output.status.success());
    assert!(
        run.stderr.contains("track 1") && run.stderr.contains("publishes 1 track"),
        "stderr does not name the track the source lacks:\n{}",
        run.stderr
    );
}

#[test]
fn a_packet_source_built_against_an_older_world_is_refused_at_open() {
    let module = older_module_path();
    let run = run_ffrwd_wasm(&[
        "-m",
        module.to_str().expect("module path is UTF-8"),
        "-track",
        "0",
        "-f",
        "nut",
        "-",
    ]);
    assert!(!run.output.status.success());
    assert!(
        run.stderr.contains("rebuild it against ffrwd:av@0.15.0"),
        "stderr does not name the world to rebuild against:\n{}",
        run.stderr
    );
}

#[test]
fn probe_prints_the_catalog_a_run_would_open() {
    let module = module_path();
    let run = run_ffrwd_wasm(&["--probe", module.to_str().expect("module path is UTF-8")]);
    assert!(
        run.output.status.success(),
        "probe exited with {:?}\nstderr:\n{}",
        run.output.status.code(),
        run.stderr
    );
    let stdout = String::from_utf8(run.stdout).expect("probe prints UTF-8");
    let catalog: serde_json::Value =
        serde_json::from_str(stdout.trim()).expect("probe prints one JSON object");
    assert_eq!(catalog["bounded"], true);
    let tracks = catalog["tracks"].as_array().expect("a tracks array");
    assert_eq!(tracks.len(), 1);
    let track = &tracks[0];
    assert_eq!(track["codec"], "h264");
    assert_eq!(track["time_base"], serde_json::json!([1, 25]));
    assert_eq!(
        track["format"],
        serde_json::json!({"video": {"width": 32, "height": 24}})
    );
    assert_eq!(
        track["extradata"],
        EXPECTED_EXTRADATA
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>()
    );
    assert_eq!(track["profile"], 0x64);
    assert_eq!(track["level"], 0x0a);
    assert_eq!(track["row"], 0);
    assert_eq!(
        track["rendition"],
        serde_json::json!({"name": null, "bandwidth": null, "codecs": null, "language": null})
    );
}

#[test]
fn probe_and_open_agree_on_the_track_the_run_reads() {
    // `probe` reads the whole catalog at compile time and `open` the tracks
    // the run named, both off the same track in the module, so the run's own
    // NUT header must match what probe printed for it.
    let module = module_path();
    let probed = run_ffrwd_wasm(&["--probe", module.to_str().expect("module path is UTF-8")]);
    assert!(probed.output.status.success());
    let probed_catalog: serde_json::Value =
        serde_json::from_str(String::from_utf8(probed.stdout).unwrap().trim()).unwrap();

    let run = run_ffrwd_wasm(&[
        "-m",
        module.to_str().expect("module path is UTF-8"),
        "-track",
        "0",
        "-f",
        "nut",
        "-",
    ]);
    assert!(
        run.output.status.success(),
        "run exited with {:?}\nstderr:\n{}",
        run.output.status.code(),
        run.stderr
    );
    let demuxer = Demuxer::open(&run.stdout[..]).expect("read the NUT headers this run wrote");
    let stream = demuxer.stream();
    assert_eq!(stream.codec_name(), Some("h264"));
    assert_eq!(stream.video_geometry(), Some((32, 24)));
    assert_eq!(
        probed_catalog["tracks"][0]["format"],
        serde_json::json!({"video": {"width": 32, "height": 24}})
    );
}

#[test]
fn a_run_writes_back_exactly_the_packets_the_module_published() {
    let module = module_path();
    let run = run_ffrwd_wasm(&[
        "-m",
        module.to_str().expect("module path is UTF-8"),
        "-track",
        "0",
        "-f",
        "nut",
        "-",
    ]);
    assert!(
        run.output.status.success(),
        "run exited with {:?}\nstderr:\n{}",
        run.output.status.code(),
        run.stderr
    );

    let mut demuxer = Demuxer::open(&run.stdout[..]).expect("read the NUT headers this run wrote");
    // The module never states its own decode_delay - the wit coded-stream
    // has no field for it - so the sidecar settles it from the leading
    // `dts: None` packets. Two of `EXPECTED`'s seven are None.
    assert_eq!(demuxer.stream().decode_delay, 2);
    assert_eq!(demuxer.stream().extradata, EXPECTED_EXTRADATA);

    let mut buf = Vec::new();
    let mut got = Vec::new();
    while let Some(packet) = demuxer.read_packet(&mut buf).expect("read a NUT packet") {
        got.push((packet.pts, packet.dts, packet.keyframe, buf.clone()));
    }
    assert_eq!(got.len(), EXPECTED.len());
    for (index, ((pts, dts, keyframe, data), (want_pts, want_dts, want_keyframe))) in
        got.iter().zip(EXPECTED).enumerate()
    {
        assert_eq!(pts, want_pts, "packet {index} pts");
        assert_eq!(dts, want_dts, "packet {index} dts");
        assert_eq!(keyframe, want_keyframe, "packet {index} keyframe");
        assert_eq!(
            data.as_slice(),
            expected_packet_bytes(index),
            "packet {index} data"
        );
    }
}

#[test]
fn annotations_have_nothing_to_give_or_take_on_a_packet_source() {
    let module = module_path();
    let run = run_ffrwd_wasm(&[
        "-m",
        module.to_str().expect("module path is UTF-8"),
        "-annotations",
        "out",
        "-track",
        "0",
        "-f",
        "nut",
        "-",
    ]);
    assert!(!run.output.status.success());
    assert!(
        run.stderr.contains("packet source") && run.stderr.contains("annotations"),
        "stderr does not refuse annotations:\n{}",
        run.stderr
    );
}

#[test]
fn describe_reports_the_packet_source() {
    let module = module_path();
    let run = run_ffrwd_wasm(&["--describe", module.to_str().expect("module path is UTF-8")]);
    assert!(
        run.output.status.success(),
        "describe exited with {:?}\nstderr:\n{}",
        run.output.status.code(),
        run.stderr
    );
    let stdout = String::from_utf8(run.stdout).expect("describe prints UTF-8");
    let description: serde_json::Value =
        serde_json::from_str(stdout.trim()).expect("describe prints one JSON object");
    assert_eq!(description["world"], "ffrwd:av@0.15.0");
    assert_eq!(description["name"], "source_replay");
    assert_eq!(description["source"], true);
    // No frame interface and no packet-sink export alongside it.
    assert!(description.get("window").is_none());
    assert!(description.get("video_codecs").is_none());
}

/// A path in the temp directory nothing else in this run uses.
fn scratch(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("ffrwd-wasm-{}-{name}", std::process::id()))
}

/// What has reached `path` so far: its NUT headers, and how many packets
/// follow. None while the headers are not there yet.
fn arrived(path: &Path) -> Option<(ffrwd_wasm::nut::Stream, usize)> {
    let file = std::fs::File::open(path).ok()?;
    let mut demuxer = Demuxer::open(std::io::BufReader::new(file)).ok()?;
    let stream = demuxer.stream().clone();
    let mut buf = Vec::new();
    let mut packets = 0;
    while let Ok(Some(_)) = demuxer.read_packet(&mut buf) {
        packets += 1;
    }
    Some((stream, packets))
}

/// The two-track fixture with track 0 writing to a pipe nobody reads, so it
/// fills and stays full, and track 1 writing to a file. Answers what reached
/// track 1's output once `enough` is true of it, or None by `deadline`; the
/// child is killed either way.
fn while_the_first_output_stands_full(
    late: &Path,
    deadline: Duration,
    enough: impl Fn(&(ffrwd_wasm::nut::Stream, usize)) -> bool,
) -> Option<(ffrwd_wasm::nut::Stream, usize)> {
    let module = pair_module_path();
    let _ = std::fs::remove_file(late);
    let mut child = Command::new(env!("CARGO_BIN_EXE_ffrwd-wasm"))
        .args([
            "-m",
            module.to_str().expect("module path is UTF-8"),
            "-track",
            "0",
            "-f",
            "nut",
            "-",
            "-track",
            "1",
            "-f",
            "nut",
            late.to_str().expect("scratch path is UTF-8"),
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn ffrwd-wasm");

    let start = Instant::now();
    let mut reached = None;
    while start.elapsed() < deadline {
        if let Some(state) = arrived(late) {
            if enough(&state) {
                reached = Some(state);
                break;
            }
        }
        std::thread::sleep(Duration::from_millis(25));
    }
    let _ = child.kill();
    let _ = child.wait();
    let _ = std::fs::remove_file(late);
    reached
}

#[test]
fn a_two_output_source_writes_both_headers_before_it_writes_a_packet() {
    // A host that wrote track 0's packets before track 1's header would block
    // on track 0's full pipe with track 1's header never written - which is a
    // reader that can never open its second input.
    let (stream, _) = while_the_first_output_stands_full(
        &scratch("headers-late.nut"),
        Duration::from_secs(30),
        |_| true,
    )
    .expect("track 1's NUT header never reached its output while track 0's pipe stood full");
    assert_eq!(stream.codec_name(), Some("h264"));
    assert_eq!(stream.video_geometry(), Some((32, 24)));
    // Track 1's one packet settles its dts at once, so it reorders nothing.
    assert_eq!(stream.decode_delay, 0);
}

#[test]
fn a_full_output_does_not_hold_up_another_outputs_packets() {
    // The reader reads packets off its second input to finish opening it, so
    // the header alone is not enough: track 1's packet has to get out while
    // track 0's pipe stands full, which one loop writing both outputs cannot
    // do. Track 1 speaks only after track 0 has published everything, so this
    // is reached only by a loop that never blocked on track 0.
    let (_, packets) = while_the_first_output_stands_full(
        &scratch("packets-late.nut"),
        Duration::from_secs(30),
        |(_, packets)| *packets > 0,
    )
    .expect("track 1's packet never reached its output while track 0's pipe stood full");
    assert_eq!(packets, 1, "track 1 publishes one packet");
}

#[test]
fn a_two_output_source_writes_each_track_to_its_own_output() {
    let module = pair_module_path();
    let wide = scratch("wide.nut");
    let late = scratch("pair-late.nut");

    let run = run_ffrwd_wasm(&[
        "-m",
        module.to_str().expect("module path is UTF-8"),
        "-track",
        "0",
        "-f",
        "nut",
        wide.to_str().expect("scratch path is UTF-8"),
        "-track",
        "1",
        "-f",
        "nut",
        late.to_str().expect("scratch path is UTF-8"),
    ]);
    assert!(
        run.output.status.success(),
        "run exited with {:?}\nstderr:\n{}",
        run.output.status.code(),
        run.stderr
    );

    let mut wide_demuxer = Demuxer::open(std::io::BufReader::new(
        std::fs::File::open(&wide).expect("track 0's output"),
    ))
    .expect("read track 0's NUT headers");
    // Two leading packets leave their dts unsettled, exactly as on the
    // one-track fixture this one replays.
    assert_eq!(wide_demuxer.stream().decode_delay, 2);
    let mut buf = Vec::new();
    let mut wide_packets = 0usize;
    while wide_demuxer
        .read_packet(&mut buf)
        .expect("read a NUT packet")
        .is_some()
    {
        wide_packets += 1;
    }

    let mut late_demuxer = Demuxer::open(std::io::BufReader::new(
        std::fs::File::open(&late).expect("track 1's output"),
    ))
    .expect("read track 1's NUT headers");
    assert_eq!(late_demuxer.stream().decode_delay, 0);
    let mut late_packets = 0usize;
    while late_demuxer
        .read_packet(&mut buf)
        .expect("read a NUT packet")
        .is_some()
    {
        late_packets += 1;
    }

    let _ = std::fs::remove_file(&wide);
    let _ = std::fs::remove_file(&late);

    // 1200 replays of the seven-packet table on track 0, one packet on
    // track 1 - the fixture's own description.
    assert_eq!(wide_packets, 1200 * 7);
    assert_eq!(late_packets, 1);
}
