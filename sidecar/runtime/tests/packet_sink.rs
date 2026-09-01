//! The packet sink driven straight through `ffrwd_wasm_runtime::runtime`,
//! with no NUT wire and no ffmpeg: the contract the runtime holds a caller
//! to, whatever drove it.

use std::path::PathBuf;
use std::process::Command;
use std::sync::OnceLock;

use ffrwd_wasm_runtime::runtime::{
    CodedFormat, CodedStream, Packet, PacketSink, SinkInput, StreamInfo, TimeBase,
};

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
                "packet-tally",
                "-p",
                "adapted-0110",
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

/// One pad's synthetic encoded stream: the sink never looks inside the bytes.
fn pad(width: u32, height: u32) -> SinkInput {
    SinkInput {
        stream: CodedStream {
            codec: "h264".to_string(),
            time_base: TimeBase { num: 1, den: 25 },
            format: CodedFormat::Video { width, height },
            extradata: vec![0, 0, 0, 1, 0x67],
        },
        info: StreamInfo::default(),
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

fn open(name: &str, pads: &[SinkInput]) -> anyhow::Result<PacketSink> {
    let module = module_path(name);
    let module_str = module.to_str().expect("module path is valid UTF-8");
    PacketSink::open(module_str, pads, "")
}

fn open_stats() -> PacketSink {
    open("packet_stats", &[pad(8, 8)]).expect("opening packet_stats")
}

#[test]
fn rows_leave_per_group_and_the_summary_trails() {
    let mut sink = open_stats();

    let first = sink
        .process(&[vec![packet(0, true, 10), packet(1, false, 5)]], false)
        .expect("the first call");
    assert!(first.rows.is_empty(), "the first group is still open");
    assert!(first.trailing.is_empty());

    let second = sink
        .process(&[vec![packet(2, true, 8)]], false)
        .expect("the second call");
    assert_eq!(
        second.rows,
        vec![r#"{"gop":0,"pts":0,"packets":2,"bytes":15}"#],
        "the keyframe closes the group before it"
    );
    assert!(second.trailing.is_empty());

    let last = sink.process(&[vec![]], true).expect("the final call");
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
    sink.process(&[vec![]], true).expect("the final call");
    let refused = sink.process(&[vec![]], false);
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
    let Err(error) = open("invert", &[pad(8, 8)]) else {
        panic!("invert exports no packet sink and must be refused");
    };
    let message = format!("{error:#}");
    assert!(
        message.contains("does not export") && message.contains("packet-sink"),
        "the refusal names what is missing and what is there:\n{message}"
    );
}

#[test]
fn each_pad_keeps_its_own_stream() {
    let pads = [pad(320, 240), pad(160, 120), pad(64, 48)];
    let mut sink = open("packet_tally", &pads).expect("opening packet_tally");
    assert_eq!(sink.pads(), 3);

    // Packets are not frames: a call may carry some pads and not others.
    sink.process(
        &[vec![packet(0, true, 10)], vec![], vec![packet(0, true, 4)]],
        false,
    )
    .expect("the first call");
    sink.process(
        &[vec![], vec![packet(0, true, 7)], vec![packet(1, false, 2)]],
        false,
    )
    .expect("the second call");

    let last = sink
        .process(&[vec![], vec![], vec![]], true)
        .expect("the final call");
    assert_eq!(
        last.trailing,
        vec![
            r#"{"pad":0,"codec":"h264","width":320,"height":240,"packets":1,"keyframes":1,"bytes":10}"#,
            r#"{"pad":1,"codec":"h264","width":160,"height":120,"packets":1,"keyframes":1,"bytes":7}"#,
            r#"{"pad":2,"codec":"h264","width":64,"height":48,"packets":2,"keyframes":1,"bytes":6}"#,
        ]
    );
}

#[test]
fn a_sink_reading_one_stream_refuses_several() {
    let Err(error) = open("packet_stats", &[pad(8, 8), pad(4, 4)]) else {
        panic!("packet_stats reads one stream and must refuse two");
    };
    let message = format!("{error:#}");
    assert!(
        message.contains("one video stream") && message.contains("hands it 2"),
        "the refusal names the shape and what arrived:\n{message}"
    );
}

#[test]
fn a_sink_of_the_previous_world_still_loads() {
    let mut sink = open("adapted_0110", &[pad(8, 8)]).expect("opening adapted_0110");
    sink.process(&[vec![packet(0, true, 3), packet(1, false, 3)]], false)
        .expect("the first call");
    let last = sink.process(&[vec![]], true).expect("the final call");
    assert_eq!(last.trailing, vec![r#"{"codec":"h264","packets":2}"#]);
}

#[test]
fn a_sink_of_the_previous_world_refuses_several() {
    let Err(error) = open("adapted_0110", &[pad(8, 8), pad(4, 4)]) else {
        panic!("a 0.11 sink reads one stream and must refuse two");
    };
    let message = format!("{error:#}");
    assert!(
        message.contains("one video stream"),
        "the refusal names the shape:\n{message}"
    );
}
