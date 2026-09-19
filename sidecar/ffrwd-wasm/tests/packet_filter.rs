//! The packet filter: encoded packets in, encoded packets out, with rows
//! arriving beside them.
//!
//! `tests/data/h264.nut` is the same fixture `packets.rs` pins - one h264
//! stream ffmpeg encoded, 50 packets, two keyframes - and the expectations
//! here are that file's, read back through this crate's own Demuxer before
//! any filter runs. What an identity filter writes is compared against it
//! packet for packet; what the SEI filter writes is compared against it
//! through real ffmpeg, which is the only thing that can say whether a
//! rewritten packet still decodes to the same picture.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::sync::OnceLock;

use ffrwd_wasm::nut::{Demuxer, Packet};

/// The committed fixture, one encoded h264 stream in NUT.
const FIXTURE: &[u8] = include_bytes!("data/h264.nut");

/// ffprobe: 50 packets, 4250 bytes across them, keyframes at these pts.
const PACKETS: usize = 50;
const TOTAL_BYTES: usize = 4250;
const KEYFRAME_PTS: &[i64] = &[4096, 65536];

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
                "packet-passthrough",
                "-p",
                "packet-sei",
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

/// Whether ffmpeg is on PATH, so the tests that shell out to real ffmpeg can
/// skip rather than fail where it is not installed.
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

/// A path in the temp directory, named after this process and the test, so
/// two tests running at once never share one.
fn scratch(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("ffrwd_packet_filter_{}_{name}", std::process::id()))
}

/// Every packet of a NUT stream, as the reader hands them out.
fn read_packets(wire: &[u8]) -> Vec<(Packet, Vec<u8>)> {
    let mut demuxer = Demuxer::open(wire).expect("read the NUT headers");
    let mut packets = Vec::new();
    let mut buf = Vec::new();
    while let Some(packet) = demuxer.read_packet(&mut buf).expect("read a NUT packet") {
        packets.push((packet, buf.clone()));
    }
    packets
}

/// Every packet's pts, dts and duration, as ffprobe reads them back. It is
/// the whole of what a container is told about timing, so two files that
/// agree here hold the same stream however they were written.
fn packet_timing(path: &Path) -> String {
    let output = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-show_packets",
            "-of",
            "compact=nk=1",
            "-show_entries",
            "packet=pts,dts,duration",
            path.to_str().expect("path is UTF-8"),
        ])
        .output()
        .expect("spawn ffprobe for packet timing");
    assert!(
        output.status.success(),
        "ffprobe of {} exited with {:?}\nstderr:\n{}",
        path.display(),
        output.status.code(),
        String::from_utf8_lossy(&output.stderr)
    );
    let text = String::from_utf8_lossy(&output.stdout).replace("\r\n", "\n");
    assert!(
        text.lines().count() > 1,
        "ffprobe read no packets out of {}",
        path.display()
    );
    text
}

/// ffmpeg's framemd5 of a file, one line per decoded frame: what says two
/// encodings decode to the same pictures.
fn framemd5(path: &Path) -> String {
    let output = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            path.to_str().expect("path is UTF-8"),
            "-f",
            "framemd5",
            "-",
        ])
        .output()
        .expect("spawn ffmpeg for framemd5");
    assert!(
        output.status.success(),
        "framemd5 of {} exited with {:?}\nstderr:\n{}",
        path.display(),
        output.status.code(),
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8_lossy(&output.stdout)
        .lines()
        // The header carries the input's own file name, which differs
        // between the two files being compared; the frame lines do not.
        .filter(|line| !line.starts_with('#'))
        .collect::<Vec<&str>>()
        .join("\n")
}

/// Muxes a NUT file to MP4 with `-c copy`: no re-encode, so the packets in
/// the MP4 are the packets the filter wrote.
fn mux_to_mp4(nut: &Path, mp4: &Path) {
    let output = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "nut",
            "-i",
            nut.to_str().expect("path is UTF-8"),
            "-c",
            "copy",
            mp4.to_str().expect("path is UTF-8"),
        ])
        .output()
        .expect("spawn ffmpeg to mux");
    assert!(
        output.status.success(),
        "muxing {} exited with {:?}\nstderr:\n{}",
        nut.display(),
        output.status.code(),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn an_identity_filter_hands_back_the_packets_it_was_given() {
    let module = module_path("packet_passthrough");
    let written = scratch("identity.nut");
    let run = run_ffrwd_wasm(
        &[
            "-f",
            "nut",
            "-i",
            fixture_path().to_str().expect("fixture path is UTF-8"),
            "-m",
            module.to_str().expect("module path is UTF-8"),
            "-f",
            "nut",
            written.to_str().expect("output path is UTF-8"),
            "-f",
            "ndjson",
            "-",
        ],
        &[],
    );
    assert!(
        run.output.status.success(),
        "packet_passthrough exited with {:?}\nstderr:\n{}",
        run.output.status.code(),
        run.stderr
    );

    let wire = std::fs::read(&written).expect("read what the filter wrote");
    std::fs::remove_file(&written).ok();

    // The stream header the filter's output carries is the one it read: the
    // codec, the geometry, the time base, the reorder depth and the SPS/PPS
    // a decoder needs before the first packet.
    let before = Demuxer::open(FIXTURE).expect("read the fixture's headers");
    let after = Demuxer::open(&wire[..]).expect("read the written headers");
    assert_eq!(after.stream().codec_name(), before.stream().codec_name());
    assert_eq!(
        after.stream().video_geometry(),
        before.stream().video_geometry()
    );
    assert_eq!(after.stream().time_base, before.stream().time_base);
    assert_eq!(after.stream().decode_delay, before.stream().decode_delay);
    assert_eq!(after.stream().extradata, before.stream().extradata);

    let original = read_packets(FIXTURE);
    let filtered = read_packets(&wire);
    assert_eq!(original.len(), PACKETS);
    assert_eq!(filtered.len(), original.len(), "one packet in, one out");
    for (index, ((was, was_data), (now, now_data))) in
        original.iter().zip(filtered.iter()).enumerate()
    {
        assert_eq!(now.pts, was.pts, "packet {index} pts");
        assert_eq!(now.dts, was.dts, "packet {index} dts");
        assert_eq!(now.keyframe, was.keyframe, "packet {index} keyframe");
        assert_eq!(now_data, was_data, "packet {index} bytes");
    }

    // The tally, and with it the two things only the module can report: the
    // reorder depth `init` was told, and that the final call carried the
    // last packets rather than nothing. ffprobe: has_b_frames = 2.
    let tally: serde_json::Value =
        serde_json::from_str(run.stdout.trim()).expect("one trailing row");
    assert_eq!(tally["pad"], 0);
    assert_eq!(tally["packets"], PACKETS);
    assert_eq!(tally["bytes"], TOTAL_BYTES);
    assert_eq!(
        tally["decode_delay"], 2,
        "the wire's reorder depth reached the module"
    );
    assert!(
        tally["last_packets"].as_u64().expect("a packet count") > 0,
        "the final call carried no packets, so a filter holding one back \
         would have nowhere to put it"
    );
}

#[test]
fn real_ffmpeg_plays_what_an_identity_filter_wrote() {
    if !ffmpeg_on_path() {
        announce_skip("real ffmpeg cannot demux what a packet filter wrote");
        return;
    }
    let module = module_path("packet_passthrough");
    let written = scratch("playable.nut");
    let run = run_ffrwd_wasm(
        &[
            "-f",
            "nut",
            "-i",
            fixture_path().to_str().expect("fixture path is UTF-8"),
            "-m",
            module.to_str().expect("module path is UTF-8"),
            "-f",
            "nut",
            written.to_str().expect("output path is UTF-8"),
            "-f",
            "null",
            "-",
        ],
        &[],
    );
    assert!(
        run.output.status.success(),
        "packet_passthrough exited with {:?}\nstderr:\n{}",
        run.output.status.code(),
        run.stderr
    );

    // Straight off the wire the two streams are the same to the timestamp:
    // every frame, at the same time, for the same length.
    assert_eq!(
        framemd5(&written),
        framemd5(&fixture_path()),
        "the identity filter's NUT decodes to the fixture frame for frame"
    );

    // Every packet's pts, dts and duration, straight off the wire. NUT has
    // no duration field, so the only thing that carries one is the frame
    // rate the stream states, and a filter that dropped it would leave a
    // reordering stream with no durations at all.
    assert_eq!(
        packet_timing(&written),
        packet_timing(&fixture_path()),
        "the identity filter's NUT states the timing the fixture states"
    );

    // And through a real muxer with -c copy, which is where a filter that
    // broke the packet count or the order would show up, and where a lost
    // duration moves the first reordered packet's dts.
    let mp4 = scratch("playable.mp4");
    mux_to_mp4(&written, &mp4);
    let source = scratch("source.mp4");
    mux_to_mp4(&fixture_path(), &source);
    assert_eq!(
        packet_timing(&mp4),
        packet_timing(&source),
        "the identity filter's mp4 holds the fixture's own packet timing"
    );
    assert_eq!(
        framemd5(&mp4),
        framemd5(&source),
        "the identity filter's mp4 decodes to the fixture's own frames"
    );
    for path in [&written, &mp4, &source] {
        std::fs::remove_file(path).ok();
    }
}

#[test]
fn a_reordering_hevc_stream_keeps_its_timing_through_an_identity_filter() {
    // The h264 fixture is committed; hevc is encoded here, because what is
    // under test is the wire and not the codec, and a second committed
    // fixture would only pin the same thing twice.
    if !ffmpeg_on_path() {
        announce_skip("real ffmpeg cannot encode the hevc stream this reads");
        return;
    }
    let source = scratch("hevc.nut");
    let output = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=64x48:rate=25:duration=2",
            "-c:v",
            "libx265",
            "-x265-params",
            "bframes=3:log-level=none",
            "-g",
            "30",
            "-pix_fmt",
            "yuv420p",
            "-f",
            "nut",
            source.to_str().expect("path is UTF-8"),
        ])
        .output()
        .expect("spawn ffmpeg to encode hevc");
    if !output.status.success() {
        announce_skip("this ffmpeg has no libx265 to encode a reordering hevc stream with");
        return;
    }

    let module = module_path("packet_passthrough");
    let written = scratch("hevc_out.nut");
    let run = run_ffrwd_wasm(
        &[
            "-f",
            "nut",
            "-i",
            source.to_str().expect("path is UTF-8"),
            "-m",
            module.to_str().expect("module path is UTF-8"),
            "-f",
            "nut",
            written.to_str().expect("path is UTF-8"),
            "-f",
            "null",
            "-",
        ],
        &[],
    );
    assert!(
        run.output.status.success(),
        "packet_passthrough exited with {:?}\nstderr:\n{}",
        run.output.status.code(),
        run.stderr
    );

    assert_eq!(
        packet_timing(&written),
        packet_timing(&source),
        "the identity filter's hevc NUT states the timing the encode states"
    );
    let filtered_mp4 = scratch("hevc_out.mp4");
    mux_to_mp4(&written, &filtered_mp4);
    let source_mp4 = scratch("hevc_src.mp4");
    mux_to_mp4(&source, &source_mp4);
    assert_eq!(
        packet_timing(&filtered_mp4),
        packet_timing(&source_mp4),
        "the identity filter's hevc mp4 holds the encode's own packet timing"
    );
    assert_eq!(
        framemd5(&filtered_mp4),
        framemd5(&source_mp4),
        "the identity filter's hevc mp4 decodes to the encode's own frames"
    );
    for path in [&source, &written, &filtered_mp4, &source_mp4] {
        std::fs::remove_file(path).ok();
    }
}

#[test]
fn rows_reach_a_filter_and_the_packets_it_rewrote_still_decode() {
    if !ffmpeg_on_path() {
        announce_skip("real ffmpeg cannot mux and decode what the SEI filter wrote");
        return;
    }
    // Two notes, in the stream's own time base: one before the first
    // keyframe (pts 4096) and one between the two (pts 65536). Which
    // keyframe carries which depends on whether the rows reader is ahead of
    // the packets, which is a race by design - so what is pinned is that
    // both notes were woven in and none was left over.
    let rows = scratch("notes.ndjson");
    std::fs::write(
        &rows,
        format!(
            "{{\"pts\":0,\"note\":\"alpha\"}}\n{{\"pts\":{},\"note\":\"beta\"}}\n",
            KEYFRAME_PTS[0] + 1
        ),
    )
    .expect("write the rows file");

    let module = module_path("packet_sei");
    let written = scratch("sei.nut");
    let run = run_ffrwd_wasm(
        &[
            "-f",
            "nut",
            "-i",
            fixture_path().to_str().expect("fixture path is UTF-8"),
            "-m",
            module.to_str().expect("module path is UTF-8"),
            "-rows-in",
            rows.to_str().expect("rows path is UTF-8"),
            "-f",
            "nut",
            written.to_str().expect("output path is UTF-8"),
            "-f",
            "ndjson",
            "-",
        ],
        &[],
    );
    assert!(
        run.output.status.success(),
        "packet_sei exited with {:?}\nstderr:\n{}",
        run.output.status.code(),
        run.stderr
    );

    let (emitted, summary) = sei_rows(&run.stdout);
    assert!(
        !emitted.is_empty(),
        "the filter reported no woven rows, so no row reached it:\n{}",
        run.stdout
    );
    let woven: u64 = emitted
        .iter()
        .map(|row| row["notes"].as_u64().expect("a note count"))
        .sum();
    assert_eq!(woven, 2, "both notes were woven in");
    assert_eq!(
        summary["rows_total"], 2,
        "both rows reached the module:\n{}",
        run.stdout
    );

    // The packets themselves: count and timestamps untouched, and the
    // keyframes that took a note longer than they were.
    let wire = std::fs::read(&written).expect("read what the filter wrote");
    let original = read_packets(FIXTURE);
    let filtered = read_packets(&wire);
    assert_eq!(filtered.len(), original.len(), "one packet in, one out");
    let mut grew = 0;
    for (index, ((was, was_data), (now, now_data))) in
        original.iter().zip(filtered.iter()).enumerate()
    {
        assert_eq!(now.pts, was.pts, "packet {index} pts");
        assert_eq!(now.dts, was.dts, "packet {index} dts");
        assert_eq!(now.keyframe, was.keyframe, "packet {index} keyframe");
        if now_data.len() != was_data.len() {
            assert!(now.keyframe, "packet {index} grew and is not a keyframe");
            assert!(
                now_data.ends_with(was_data),
                "packet {index} lost the bytes the encoder wrote"
            );
            grew += 1;
        }
    }
    assert_eq!(grew, emitted.len(), "one rewritten packet per woven row");

    // Through a real muxer with -c copy, and decoded: the pictures are the
    // pictures the encoder wrote, whatever rode beside them.
    let filtered_mp4 = scratch("sei.mp4");
    mux_to_mp4(&written, &filtered_mp4);
    let source_mp4 = scratch("sei_source.mp4");
    mux_to_mp4(&fixture_path(), &source_mp4);
    assert_eq!(
        packet_timing(&filtered_mp4),
        packet_timing(&source_mp4),
        "weaving a message in moves no packet in time"
    );
    assert_eq!(
        framemd5(&filtered_mp4),
        framemd5(&source_mp4),
        "the woven stream decodes to the unfiltered encode's frames"
    );

    for path in [&rows, &written, &filtered_mp4, &source_mp4] {
        std::fs::remove_file(path).ok();
    }
}

#[test]
fn params_come_out_of_a_file_as_readily_as_off_the_line() {
    // A module's parameters can be long, and a command line is a poor place
    // to keep one. The two spellings name the same value, and naming both
    // is refused.
    let module = module_path("packet_passthrough");
    let params = scratch("params.json");
    std::fs::write(&params, "{}").expect("write the params file");
    let written = scratch("params_out.nut");
    let argv = |from: &str| {
        vec![
            "-f".to_string(),
            "nut".to_string(),
            "-i".to_string(),
            fixture_path()
                .to_str()
                .expect("fixture path is UTF-8")
                .to_string(),
            "-m".to_string(),
            module.to_str().expect("module path is UTF-8").to_string(),
            from.to_string(),
            params.to_str().expect("params path is UTF-8").to_string(),
            "-f".to_string(),
            "nut".to_string(),
            written.to_str().expect("output path is UTF-8").to_string(),
        ]
    };
    let flags: Vec<String> = argv("-params-from");
    let borrowed: Vec<&str> = flags.iter().map(String::as_str).collect();
    let run = run_ffrwd_wasm(&borrowed, &[]);
    assert!(
        run.output.status.success(),
        "-params-from exited with {:?}\nstderr:\n{}",
        run.output.status.code(),
        run.stderr
    );

    let mut both = flags.clone();
    both.insert(6, "-params".to_string());
    both.insert(7, "{}".to_string());
    let borrowed: Vec<&str> = both.iter().map(String::as_str).collect();
    let run = run_ffrwd_wasm(&borrowed, &[]);
    assert!(!run.output.status.success());
    assert!(
        run.stderr.contains("-params-from and -params"),
        "stderr does not refuse the pair:\n{}",
        run.stderr
    );

    for path in [&params, &written] {
        std::fs::remove_file(path).ok();
    }
}

#[test]
fn a_rows_input_that_cannot_be_opened_stops_the_run() {
    // The rows input is opened on its reader's own thread, so a live run
    // does not wait on a named pipe before its packets move. That puts the
    // open's failure off the main thread, and it still has to reach it.
    let module = module_path("packet_sei");
    let missing = scratch("no_such_rows.ndjson");
    let unwritten = scratch("unwritten.nut");
    std::fs::remove_file(&missing).ok();
    let run = run_ffrwd_wasm(
        &[
            "-f",
            "nut",
            "-i",
            fixture_path().to_str().expect("fixture path is UTF-8"),
            "-m",
            module.to_str().expect("module path is UTF-8"),
            "-rows-in",
            missing.to_str().expect("rows path is UTF-8"),
            "-f",
            "nut",
            unwritten.to_str().expect("output path is UTF-8"),
        ],
        &[],
    );
    assert!(!run.output.status.success());
    assert!(
        run.stderr.contains("-rows-in"),
        "stderr does not name the rows input that failed:\n{}",
        run.stderr
    );
    std::fs::remove_file(&unwritten).ok();
}

/// `packet_sei`'s rows, split into the woven ones and the single summary
/// the last call carries. A leftover note - one no keyframe came along to
/// carry - is neither, and fails the split: nothing a caller sent should go
/// unwoven in these tests.
fn sei_rows(stdout: &str) -> (Vec<serde_json::Value>, serde_json::Value) {
    let mut woven = Vec::new();
    let mut summary = None;
    for line in stdout.lines() {
        let row: serde_json::Value =
            serde_json::from_str(line).expect("packet_sei emits one JSON row per line");
        if row.get("calls").is_some() {
            assert!(summary.is_none(), "one summary row, not two:\n{stdout}");
            summary = Some(row);
        } else if row.get("notes").is_some() {
            woven.push(row);
        } else {
            panic!("a note was left with no keyframe to ride:\n{stdout}");
        }
    }
    (woven, summary.expect("the last call carries a summary row"))
}

#[test]
fn a_files_rows_are_all_in_hand_before_the_first_packet() {
    // The rows reader starts with the pad readers, not after the module is
    // open, and a FILE's rows are read to the end before the first call. Run
    // after the open, the reader loses the race on an input this short: the
    // pads fill their queues while the module is still being instantiated,
    // and the rows turn up call by call or not until the last one.
    //
    // Repeated, because a race that comes out right once has proven nothing:
    // every run has to put every row on call one.
    let module = module_path("packet_sei");
    let rows = scratch("first_call.ndjson");
    let written = scratch("first_call.nut");
    // Enough rows that reading them is not instantaneous. Three would be
    // in hand before the module finished opening whatever the host did,
    // and a test that cannot tell the two orderings apart proves nothing.
    // They all sit at pts 0, so the first keyframe carries the lot.
    const NOTES: usize = 4000;
    let mut text = String::new();
    for index in 0..NOTES {
        text.push_str(&format!("{{\"pts\":0,\"note\":\"note-{index:06}\"}}\n"));
    }
    std::fs::write(&rows, &text).expect("write the rows file");

    for attempt in 1..=5 {
        let run = run_ffrwd_wasm(
            &[
                "-f",
                "nut",
                "-i",
                fixture_path().to_str().expect("fixture path is UTF-8"),
                "-m",
                module.to_str().expect("module path is UTF-8"),
                "-rows-in",
                rows.to_str().expect("rows path is UTF-8"),
                "-f",
                "nut",
                written.to_str().expect("output path is UTF-8"),
                "-f",
                "ndjson",
                "-",
            ],
            &[],
        );
        assert!(
            run.output.status.success(),
            "attempt {attempt} exited with {:?}\nstderr:\n{}",
            run.output.status.code(),
            run.stderr
        );
        let (_, summary) = sei_rows(&run.stdout);
        assert_eq!(
            summary["rows_total"], NOTES,
            "attempt {attempt}: every row reached the module\n{}",
            run.stdout
        );
        assert_eq!(
            summary["rows_first_call"], NOTES,
            "attempt {attempt}: every row was there before the first packet\n{}",
            run.stdout
        );
    }

    for path in [&rows, &written] {
        std::fs::remove_file(path).ok();
    }
}

#[test]
fn a_filter_that_reads_rows_is_refused_without_them() {
    let module = module_path("packet_sei");
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
        run.stderr.contains("-rows-in"),
        "stderr does not name the missing rows input:\n{}",
        run.stderr
    );
}

#[test]
fn a_filter_needs_one_frame_output_per_pad() {
    let module = module_path("packet_passthrough");
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
        FIXTURE,
    );
    assert!(!run.output.status.success());
    assert!(
        run.stderr.contains("hands on the packets of every pad"),
        "stderr does not name the missing output:\n{}",
        run.stderr
    );
}

#[test]
fn rows_in_on_a_module_that_is_neither_is_refused() {
    let module = module_path("packet_stats");
    let run = run_ffrwd_wasm(
        &[
            "-f",
            "nut",
            "-i",
            "-",
            "-m",
            module.to_str().expect("module path is UTF-8"),
            "-rows-in",
            "-",
            "-f",
            "ndjson",
            "-",
        ],
        FIXTURE,
    );
    assert!(!run.output.status.success());
    assert!(
        run.stderr.contains("-rows-in") && run.stderr.contains("is neither"),
        "stderr does not refuse -rows-in:\n{}",
        run.stderr
    );
}

#[test]
fn describe_reports_the_packet_filter() {
    let module = module_path("packet_sei");
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
    assert_eq!(description["world"], "ffrwd:av@0.16.0");
    assert_eq!(description["name"], "packet_sei");
    // The flag is what tells a filter from a sink; both carry the codec and
    // arity fields beside it.
    assert_eq!(description["packet_filter"], true);
    assert_eq!(description["video_codecs"], serde_json::json!(["h264"]));
    assert_eq!(description["video_streams"], "one");
    assert_eq!(description["audio_streams"], "none");
    assert_eq!(description["reads_rows"], true);
    // No frame interface: none of the windowed fields appear.
    assert!(description.get("window").is_none());
}
