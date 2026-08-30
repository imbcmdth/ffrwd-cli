//! The packet sink driven straight through `ffrwd_wasm_runtime::runtime`,
//! with no NUT wire and no ffmpeg: the contract the runtime holds a caller
//! to, whatever drove it.

use std::path::PathBuf;
use std::process::Command;
use std::sync::OnceLock;

use ffrwd_wasm_runtime::runtime::{CodedStream, Packet, PacketSink, StreamInfo, TimeBase};

/// Repo root, the parent of `runtime/`.
fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("runtime/ has a parent directory")
        .to_path_buf()
}

/// Absolute path to a module's built `.wasm` component, built once per test
/// binary. `modules/` is a separate cargo workspace with its own build lock,
/// so this does not deadlock against the `cargo test` run driving this
/// binary.
fn module_path(name: &str) -> PathBuf {
    static BUILT: OnceLock<()> = OnceLock::new();
    BUILT.get_or_init(|| {
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
            .current_dir(repo_root().join("modules"))
            .output()
            .expect("spawn cargo build for modules");
        assert!(
            output.status.success(),
            "building modules failed (status {:?}):\n{}",
            output.status.code(),
            String::from_utf8_lossy(&output.stderr)
        );
    });
    repo_root()
        .join("modules/target/wasm32-wasip2/release")
        .join(format!("{name}.wasm"))
}

/// A synthetic encoded stream: the sink never looks inside the bytes.
fn coded_stream() -> CodedStream {
    CodedStream {
        codec: "h264".to_string(),
        time_base: TimeBase { num: 1, den: 25 },
        width: 8,
        height: 8,
        extradata: vec![0, 0, 0, 1, 0x67],
    }
}

fn packet(pts: i64, keyframe: bool, len: usize) -> Packet {
    Packet {
        pts,
        dts: Some(pts),
        keyframe,
        data: vec![0xAB; len],
    }
}

fn open_stats() -> PacketSink {
    let module = module_path("packet_stats");
    let module_str = module.to_str().expect("module path is valid UTF-8");
    PacketSink::open(module_str, &coded_stream(), &StreamInfo::default(), "")
        .expect("opening packet_stats")
}

#[test]
fn rows_leave_per_group_and_the_summary_trails() {
    let mut sink = open_stats();

    let first = sink
        .process(&[packet(0, true, 10), packet(1, false, 5)], false)
        .expect("the first call");
    assert!(first.rows.is_empty(), "the first group is still open");
    assert!(first.trailing.is_empty());

    let second = sink
        .process(&[packet(2, true, 8)], false)
        .expect("the second call");
    assert_eq!(
        second.rows,
        vec![r#"{"gop":0,"pts":0,"packets":2,"bytes":15}"#],
        "the keyframe closes the group before it"
    );
    assert!(second.trailing.is_empty());

    let last = sink.process(&[], true).expect("the final call");
    assert_eq!(
        last.rows,
        vec![r#"{"gop":1,"pts":2,"packets":1,"bytes":8}"#]
    );
    assert_eq!(
        last.trailing,
        vec![r#"{"packets":3,"keyframes":2,"bytes":23,"gops":2,"pts_monotonic":true}"#]
    );
}

#[test]
fn a_call_after_the_final_one_is_refused() {
    let mut sink = open_stats();
    sink.process(&[], true).expect("the final call");
    let refused = sink.process(&[], false);
    assert!(
        refused
            .as_ref()
            .err()
            .is_some_and(|e| e.to_string().contains("final call")),
        "a call after the final one must be refused, got {refused:?}"
    );
}

#[test]
fn a_frames_module_is_refused_by_name() {
    let module = module_path("invert");
    let module_str = module.to_str().expect("module path is valid UTF-8");
    let Err(error) = PacketSink::open(module_str, &coded_stream(), &StreamInfo::default(), "")
    else {
        panic!("invert exports no packet sink and must be refused");
    };
    let message = format!("{error:#}");
    assert!(
        message.contains("does not export") && message.contains("packet-sink"),
        "the refusal names what is missing and what is there:\n{message}"
    );
}
