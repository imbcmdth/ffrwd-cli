//! Integration tests for the sidecar's packet-source HEAD mode: no `-i`, one
//! `-f nut` output per catalog track, driven by `source_replay` - a fixture
//! module that hands back a fixed, compiled-in set of coded packets (see
//! `modules/source-replay/src/lib.rs` for why it is compiled in rather than
//! read from a file: the wasm sandbox these modules run in has no
//! filesystem).

use std::path::PathBuf;
use std::process::{Command, Output, Stdio};
use std::sync::OnceLock;

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
fn module_path() -> PathBuf {
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
    sidecar_root().join("modules/target/wasm32-wasip2/release/source_replay.wasm")
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
    0x00, 0x00, 0x00, 0x01, 0x67, 0x64, 0x00, 0x0a, 0xac, 0xd9, 0x44, 0x7b, 0x01, 0x10, 0x00, 0x00,
    0x00, 0x01, 0x68, 0xeb, 0xe3, 0xcb, 0x22, 0xc0,
];

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
fn the_output_count_must_match_the_catalogs_track_count() {
    let module = module_path();
    let dir = tempdir();
    let extra = dir.join("extra.nut");
    let run = run_ffrwd_wasm(&[
        "-m",
        module.to_str().expect("module path is UTF-8"),
        "-f",
        "nut",
        "-",
        "-f",
        "nut",
        extra.to_str().expect("temp path is UTF-8"),
    ]);
    assert!(!run.output.status.success());
    assert!(
        run.stderr.contains("names 1 track") && run.stderr.contains("2 output"),
        "stderr does not name both counts:\n{}",
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
fn probe_and_open_agree_the_source_never_changed_shape() {
    // `probe` (compile time) and the catalog `run_packet_source`'s `open`
    // reads (run time) both come from the same `catalog()` in the module,
    // so the run's own NUT header must match what probe printed.
    let module = module_path();
    let probed = run_ffrwd_wasm(&["--probe", module.to_str().expect("module path is UTF-8")]);
    assert!(probed.output.status.success());
    let probed_catalog: serde_json::Value =
        serde_json::from_str(String::from_utf8(probed.stdout).unwrap().trim()).unwrap();

    let run = run_ffrwd_wasm(&[
        "-m",
        module.to_str().expect("module path is UTF-8"),
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
            data,
            &vec![0x10u8.wrapping_add(index as u8); 24],
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
    assert_eq!(description["world"], "ffrwd:av@0.13.0");
    assert_eq!(description["name"], "source_replay");
    assert_eq!(description["source"], true);
    // No frame interface and no packet-sink export alongside it.
    assert!(description.get("window").is_none());
    assert!(description.get("video_codecs").is_none());
}

/// A fresh temp directory this test owns, cleaned up on drop by the OS's own
/// temp-directory sweep - these tests write small files, never read them
/// back from disk, and the crate carries no tempfile dependency to add one
/// just for this.
fn tempdir() -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "ffrwd-wasm-packet-source-test-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("the clock is past 1970")
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).expect("create a temp directory");
    dir
}
