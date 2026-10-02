//! The shape modules, one per distinct shape a node module can declare, run
//! through the sidecar as networks of node modules. Each run is made at
//! `-jobs` 1, 2 and 4 and has to write the same bytes every time; then the
//! first, a middle and the last tick are checked against what the WIT says
//! that tick holds.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

use ffrwd_wasm::nut::{Event, Limits, Media, Muxer, Packet, PushDemuxer, Stream, TimeBase};
use serde_json::Value;

const SHAPES: &[&str] = &[
    "shape-hold",
    "shape-state",
    "shape-canvas",
    "shape-rate",
    "shape-recut",
    "shape-self",
    "shape-sink",
    "shape-window",
    "shape-packets",
    "shape-switch",
    "shape-probe",
    "spot",
    "ring",
    "matte",
];

fn sidecar_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("ffrwd-wasm/ has a parent directory")
        .to_path_buf()
}

/// `-m name=path` for one shape module, building them all once.
fn module(name: &str) -> String {
    static BUILT: OnceLock<()> = OnceLock::new();
    BUILT.get_or_init(|| {
        let mut args = vec!["build", "--release", "--target", "wasm32-wasip2"];
        for shape in SHAPES {
            args.extend(["-p", shape]);
        }
        let output = Command::new("cargo")
            .args(&args)
            .current_dir(sidecar_root().join("modules"))
            .output()
            .expect("spawn cargo build");
        assert!(
            output.status.success(),
            "building the shape modules failed:\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
    });
    let path = sidecar_root().join(format!("modules/target/wasm32-wasip2/release/{name}.wasm"));
    format!("{name}={}", path.display())
}

fn scratch(test: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("ffrwd-node-shapes-{}-{test}", std::process::id()));
    fs::create_dir_all(&dir).expect("make a scratch directory");
    dir
}

/// Runs `args` at -jobs 1, 2 and 4, each writing `outputs` into a directory
/// of its own, and checks the three wrote the same bytes. `{dir}` in an
/// argument is that directory. Returns the outputs of the run at 1.
fn at_every_jobs(test: &str, args: &[String], outputs: &[&str]) -> Vec<Vec<u8>> {
    let base = scratch(test);
    let mut runs: Vec<Vec<Vec<u8>>> = Vec::new();
    for jobs in ["1", "2", "4"] {
        let dir = base.join(format!("j{jobs}"));
        fs::create_dir_all(&dir).expect("make a run directory");
        let spelled: Vec<String> = args
            .iter()
            .map(|a| a.replace("{dir}", &dir.display().to_string()))
            .collect();
        let output = Command::new(env!("CARGO_BIN_EXE_ffrwd-wasm"))
            .arg("-jobs")
            .arg(jobs)
            .args(&spelled)
            .output()
            .expect("spawn ffrwd-wasm");
        assert!(
            output.status.success(),
            "-jobs {jobs} exited {:?}:\n{}",
            output.status.code(),
            String::from_utf8_lossy(&output.stderr)
        );
        runs.push(
            outputs
                .iter()
                .map(|name| fs::read(dir.join(name)).expect("the output was written"))
                .collect(),
        );
    }
    assert_eq!(
        runs[0], runs[1],
        "{test}: -jobs 1 and -jobs 2 wrote different bytes"
    );
    assert_eq!(
        runs[0], runs[2],
        "{test}: -jobs 1 and -jobs 4 wrote different bytes"
    );
    let _ = fs::remove_dir_all(&base);
    runs.swap_remove(0)
}

fn args(list: &[&str]) -> Vec<String> {
    list.iter().map(|a| a.to_string()).collect()
}

/// One thing to write on a stream of a test input.
enum Write {
    Frame(i64, Vec<u8>),
    Message(i64, String),
    Coded(i64, bool, Vec<u8>),
}

fn write_nut(path: &Path, streams: &[Stream], items: &[(usize, Write)]) {
    let mut wire = Vec::new();
    {
        let mut muxer = Muxer::with_streams(&mut wire, streams).expect("write headers");
        for (stream, item) in items {
            match item {
                Write::Frame(pts, data) => muxer.write_frame_to(*stream, *pts, data),
                Write::Message(pts, text) => muxer.write_coded_to(
                    *stream,
                    &Packet {
                        pts: *pts,
                        dts: Some(*pts),
                        keyframe: true,
                    },
                    text.as_bytes(),
                ),
                Write::Coded(pts, keyframe, data) => muxer.write_coded_to(
                    *stream,
                    &Packet {
                        pts: *pts,
                        dts: Some(*pts),
                        keyframe: *keyframe,
                    },
                    data,
                ),
            }
            .expect("write an item");
        }
        muxer.finish().expect("finish");
    }
    fs::write(path, wire).expect("write the input");
}

fn video(width: u32, height: u32, base: TimeBase, rate: (u64, u64)) -> Stream {
    let mut stream = Stream::video("rgba", width, height, base).expect("rgba is carried");
    stream.frame_rate = Some(rate);
    stream
}

/// One frame of an output: its stream, its pts and its bytes.
type Written = (usize, i64, Vec<u8>);

/// Every frame of a NUT output.
fn frames(wire: &[u8]) -> (Vec<Stream>, Vec<Written>) {
    let mut demuxer = PushDemuxer::new(Limits::default());
    demuxer.feed(wire);
    demuxer.finish();
    let mut out = Vec::new();
    while let Some(event) = demuxer.next_event().expect("the output parses") {
        match event {
            Event::Frame { stream, packet } => {
                out.push((stream, packet.pts, demuxer.payload().to_vec()))
            }
            Event::EndOfInput => break,
            _ => {}
        }
    }
    let streams = demuxer.streams().iter().flatten().cloned().collect();
    (streams, out)
}

fn lines(bytes: &[u8]) -> Vec<Value> {
    String::from_utf8_lossy(bytes)
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).expect("a JSON line"))
        .collect()
}

const TENTHS: TimeBase = TimeBase { num: 1, den: 10 };

#[test]
fn a_held_picture_is_handed_back_from_whichever_input_the_call_picks() {
    let dir = scratch("hold-inputs");
    let full: Vec<(usize, Write)> = (0..10)
        .map(|k| (0, Write::Frame(k, vec![10 + k as u8; 64])))
        .collect();
    let half: Vec<(usize, Write)> = (0..5)
        .map(|k| (0, Write::Frame(2 * k, vec![100 + k as u8; 64])))
        .collect();
    write_nut(&dir.join("a.nut"), &[video(4, 4, TENTHS, (10, 1))], &full);
    write_nut(&dir.join("b.nut"), &[video(4, 4, TENTHS, (5, 1))], &half);
    let a = dir.join("a.nut").display().to_string();
    let b = dir.join("b.nut").display().to_string();
    let out = at_every_jobs(
        "hold",
        &args(&[
            "-f",
            "nut",
            "-i",
            &a,
            "-f",
            "nut",
            "-i",
            &b,
            "-m",
            &module("shape_hold"),
            "-filter_complex",
            "[v=0:v][v=1:v]shape_hold=width=4:height=4:pick=1[out=o]",
            "-map",
            "[o]",
            "-f",
            "nut",
            "{dir}/out.nut",
        ]),
        &["out.nut"],
    );
    let (streams, got) = frames(&out[0]);
    assert_eq!(
        streams[0].frame_rate,
        Some((10, 1)),
        "the rate of the first picture"
    );
    assert_eq!(
        got.len(),
        10,
        "a tick per frame of the clock's picture, none on the last"
    );
    let at = |n: usize| (got[n].1, got[n].2[0]);
    assert_eq!(
        at(0),
        (0, 100),
        "the first tick shows the picked input's first frame"
    );
    assert_eq!(at(5), (5, 102), "tick 5 holds the frame at 4");
    assert_eq!(at(8), (8, 104), "tick 8 is the last frame's turn");
    assert_eq!(
        at(9),
        (9, 19),
        "the picked input's feed ended after its last frame showed, so the other stands in"
    );
}

#[test]
fn with_no_picture_bound_the_held_node_ticks_at_its_rate_and_finishes() {
    let out = at_every_jobs(
        "hold-none",
        &args(&[
            "-m",
            &module("shape_hold"),
            "-filter_complex",
            "shape_hold=width=2:height=2:fps=5:frames=3[out=o]",
            "-map",
            "[o]",
            "-f",
            "nut",
            "{dir}/out.nut",
        ]),
        &["out.nut"],
    );
    let (streams, got) = frames(&out[0]);
    assert_eq!(streams[0].time_base, TimeBase { num: 1, den: 5 });
    let ticks: Vec<(i64, u8)> = got.iter().map(|(_, pts, data)| (*pts, data[0])).collect();
    assert_eq!(ticks, vec![(0, 0), (1, 1), (2, 2)]);
}

#[test]
fn every_instance_of_a_state_node_sees_every_row() {
    let dir = scratch("state-input");
    let mut items: Vec<(usize, Write)> = Vec::new();
    for k in 0..20i64 {
        if k % 3 == 0 {
            items.push((1, Write::Message(k, format!(r#"{{"text":"w{k}"}}"#))));
        }
        items.push((0, Write::Frame(k, vec![k as u8; 16])));
    }
    write_nut(
        &dir.join("in.nut"),
        &[video(2, 2, TENTHS, (10, 1)), Stream::json(TENTHS)],
        &items,
    );
    let input = dir.join("in.nut").display().to_string();
    let out = at_every_jobs(
        "state",
        &args(&[
            "-f",
            "nut",
            "-i",
            &input,
            "-m",
            &module("shape_state"),
            "-filter_complex",
            "[v=0:v][words=0:d]shape_state[seen=s]",
            "-map",
            "[s]",
            "-f",
            "ndjson",
            "{dir}/seen.ndjson",
        ]),
        &["seen.ndjson"],
    );
    let ticks = lines(&out[0]);
    assert_eq!(ticks.len(), 20);
    let seen = |n: usize| {
        (
            ticks[n]["pts"].as_i64(),
            ticks[n]["seen"].as_u64(),
            ticks[n]["now"].as_u64(),
        )
    };
    assert_eq!(seen(0), (Some(0), Some(1), Some(1)));
    assert_eq!(seen(10), (Some(10), Some(4), Some(0)));
    assert_eq!(seen(19), (Some(19), Some(7), Some(0)));
    assert_eq!(ticks[19]["last"], true);
}

#[test]
fn an_output_sized_by_the_params_is_redrawn_at_that_size() {
    let dir = scratch("canvas");
    let picture = |k: i64| -> Vec<u8> {
        (0..64usize)
            .flat_map(|p| [p as u8, k as u8, 0, 255])
            .collect()
    };
    let items: Vec<(usize, Write)> = (0..6).map(|k| (0, Write::Frame(k, picture(k)))).collect();
    write_nut(&dir.join("in.nut"), &[video(8, 8, TENTHS, (10, 1))], &items);
    let input = dir.join("in.nut").display().to_string();
    let out = at_every_jobs(
        "canvas",
        &args(&[
            "-f",
            "nut",
            "-i",
            &input,
            "-m",
            &module("shape_canvas"),
            "-filter_complex",
            "[v=0:v]shape_canvas=width=2:height=2[out=o]",
            "-map",
            "[o]",
            "-f",
            "nut",
            "{dir}/out.nut",
        ]),
        &["out.nut"],
    );
    let (streams, got) = frames(&out[0]);
    assert_eq!(streams[0].video_geometry(), Some((2, 2)));
    assert_eq!(got.len(), 6);
    for n in [0usize, 3, 5] {
        let (_, pts, data) = &got[n];
        assert_eq!(*pts, n as i64);
        assert_eq!(data.len(), 16);
        assert_eq!(
            (data[0], data[1]),
            (0, n as u8),
            "pixel 0,0 is the input's 0,0"
        );
        assert_eq!(data[12], 36, "pixel 1,1 is the input's 4,4");
    }
}

#[test]
fn leaky_is_a_host_node_of_a_node_network() {
    let dir = scratch("leaky-node");
    let items: Vec<(usize, Write)> = (0..6)
        .map(|k| (0, Write::Frame(k, vec![k as u8; 8 * 8 * 4])))
        .collect();
    write_nut(&dir.join("in.nut"), &[video(8, 8, TENTHS, (10, 1))], &items);
    let input = dir.join("in.nut").display().to_string();
    let out = at_every_jobs(
        "leaky-node",
        &args(&[
            "-f",
            "nut",
            "-i",
            &input,
            "-m",
            &module("shape_canvas"),
            "-filter_complex",
            "[v=0:v]shape_canvas=width=2:height=2[out=o];[o]leaky=max_lateness=5[l]",
            "-map",
            "[l]",
            "-f",
            "nut",
            "{dir}/out.nut",
        ]),
        &["out.nut"],
    );
    let (streams, got) = frames(&out[0]);
    assert_eq!(streams[0].video_geometry(), Some((2, 2)));
    let pts: Vec<i64> = got.iter().map(|(_, pts, _)| *pts).collect();
    assert_eq!(
        pts,
        vec![0, 1, 2, 3, 4, 5],
        "a file read at once is never late"
    );
    assert!(got
        .iter()
        .enumerate()
        .all(|(k, (_, _, data))| data[0] == k as u8));
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_pad_hands_a_node_the_colour_the_wire_does_not_carry() {
    let dir = scratch("pad-colour");
    let items: Vec<(usize, Write)> = (0..3)
        .map(|k| (0, Write::Frame(k, vec![7u8; 8 * 8 * 4])))
        .collect();
    write_nut(&dir.join("in.nut"), &[video(8, 8, TENTHS, (10, 1))], &items);
    let input = dir.join("in.nut").display().to_string();
    let out = at_every_jobs(
        "pad-colour",
        &args(&[
            "-f",
            "nut",
            "-i",
            &input,
            "-pad",
            r#"{"color": {"range": "pc", "primaries": "bt709", "trc": "bt709", "space": "bt709"}}"#,
            "-m",
            &module("shape_canvas"),
            "-filter_complex",
            "[v=0:v]shape_canvas=width=2:height=2[out=o][@rows=r]",
            "-map",
            "[o]",
            "-f",
            "nut",
            "{dir}/out.nut",
            "-map",
            "[r]",
            "-f",
            "ndjson",
            "{dir}/rows.ndjson",
        ]),
        &["out.nut", "rows.ndjson"],
    );
    let rows = lines(&out[1]);
    assert_eq!(rows.len(), 1, "one row, the colour, on the first tick");
    assert_eq!(rows[0]["range"], "pc");
    assert_eq!(rows[0]["space"], "bt709");
    assert_eq!(rows[0]["primaries"], "bt709");
}

#[test]
fn a_rate_with_no_inputs_ticks_until_the_node_finishes() {
    let out = at_every_jobs(
        "rate",
        &args(&[
            "-m",
            &module("shape_rate"),
            "-filter_complex",
            "shape_rate=fps=25:frames=10:width=2:height=2[out=o]",
            "-map",
            "[o]",
            "-f",
            "nut",
            "{dir}/out.nut",
        ]),
        &["out.nut"],
    );
    let (streams, got) = frames(&out[0]);
    assert_eq!(streams[0].frame_rate, Some((25, 1)));
    assert_eq!(got.len(), 10, "nothing past the tick that finished");
    for n in [0usize, 4, 9] {
        assert_eq!((got[n].1, got[n].2[0]), (n as i64, n as u8));
    }
}

#[test]
fn sound_under_a_picture_clock_is_recut_to_each_tick() {
    let dir = scratch("recut");
    let ntsc = TimeBase { num: 1, den: 30000 };
    let audio = Stream::audio("f32", 48000, 1).expect("f32 is carried");
    let mut items: Vec<(usize, Write)> = Vec::new();
    let mut sample = 0i64;
    for k in 0..30i64 {
        let until = (k + 1) * 1001 * 48000 / 30000;
        while sample < until + 1024 && sample < 48128 {
            items.push((1, Write::Frame(sample, vec![0u8; 1024 * 4])));
            sample += 1024;
        }
        items.push((0, Write::Frame(k * 1001, vec![0u8; 16])));
    }
    write_nut(
        &dir.join("in.nut"),
        &[video(2, 2, ntsc, (30000, 1001)), audio],
        &items,
    );
    let input = dir.join("in.nut").display().to_string();
    let out = at_every_jobs(
        "recut",
        &args(&[
            "-f",
            "nut",
            "-i",
            &input,
            "-m",
            &module("shape_recut"),
            "-filter_complex",
            "[v=0:v][a=0:a]shape_recut[samples=s]",
            "-map",
            "[s]",
            "-f",
            "ndjson",
            "{dir}/samples.ndjson",
        ]),
        &["samples.ndjson"],
    );
    let ticks = lines(&out[0]);
    assert_eq!(ticks.len(), 30);
    let samples: Vec<u64> = ticks
        .iter()
        .map(|t| t["samples"].as_u64().unwrap())
        .collect();
    assert_eq!(
        (samples[0], ticks[0]["audio_pts"].as_i64()),
        (1602, Some(0))
    );
    assert_eq!(samples[1..29].iter().filter(|s| **s == 1601).count(), 11);
    assert!(samples[1..29].iter().all(|s| *s == 1601 || *s == 1602));
    let before: u64 = samples[..15].iter().sum();
    assert_eq!(ticks[15]["audio_pts"].as_i64(), Some(before as i64));
    assert_eq!(ticks[29]["last"], true);
    assert_eq!(
        samples.iter().sum::<u64>(),
        48128,
        "the last tick takes what is left"
    );
}

#[test]
fn a_self_clocked_source_runs_until_it_finishes() {
    let out = at_every_jobs(
        "self",
        &args(&[
            "-m",
            &module("shape_self"),
            "-filter_complex",
            "shape_self=count=4[msgs=m]",
            "-map",
            "[m]",
            "-f",
            "nut",
            "{dir}/out.nut",
        ]),
        &["out.nut"],
    );
    let (streams, got) = frames(&out[0]);
    assert!(streams[0].is_json());
    let messages: Vec<(i64, String)> = got
        .into_iter()
        .filter(|(_, _, data)| !data.iter().all(u8::is_ascii_whitespace))
        .map(|(_, pts, data)| (pts, String::from_utf8(data).unwrap()))
        .collect();
    assert_eq!(
        messages,
        vec![
            (0, r#"{"n":0}"#.to_string()),
            (100, r#"{"n":1}"#.to_string()),
            (200, r#"{"n":2}"#.to_string()),
            (300, r#"{"n":3}"#.to_string()),
        ]
    );
}

#[test]
fn a_sink_of_many_pictures_takes_every_frame_of_each() {
    let dir = scratch("sink");
    let mut inputs = Vec::new();
    for (index, count) in [3i64, 7, 5].into_iter().enumerate() {
        let items: Vec<(usize, Write)> = (0..count)
            .map(|k| (0, Write::Frame(k, vec![0u8; 16])))
            .collect();
        let path = dir.join(format!("{index}.nut"));
        write_nut(&path, &[video(2, 2, TENTHS, (10, 1))], &items);
        inputs.push(path.display().to_string());
    }
    let out = at_every_jobs(
        "sink",
        &args(&[
            "-f",
            "nut",
            "-i",
            &inputs[0],
            "-f",
            "nut",
            "-i",
            &inputs[1],
            "-f",
            "nut",
            "-i",
            &inputs[2],
            "-m",
            &module("shape_sink"),
            "-filter_complex",
            "[v=0:v][v=1:v][v=2:v]shape_sink[@rows=r]",
            "-map",
            "[r]",
            "-f",
            "ndjson",
            "{dir}/rows.ndjson",
        ]),
        &["rows.ndjson"],
    );
    let rows = lines(&out[0]);
    let counts: Vec<(u64, u64, i64, i64)> = rows
        .iter()
        .map(|r| {
            (
                r["id"].as_u64().unwrap(),
                r["frames"].as_u64().unwrap(),
                r["first"].as_i64().unwrap(),
                r["last"].as_i64().unwrap(),
            )
        })
        .collect();
    assert_eq!(counts, vec![(0, 3, 0, 2), (1, 7, 0, 6), (2, 5, 0, 4)]);
}

#[test]
fn a_reader_by_interval_waits_for_the_window_that_holds_its_time() {
    let dir = scratch("window");
    let audio = Stream::audio("f32", 48000, 1).expect("f32 is carried");
    let mut items: Vec<(usize, Write)> = Vec::new();
    for k in 0..40i64 {
        items.push((1, Write::Frame(k * 4800, vec![0u8; 4800 * 4])));
        items.push((0, Write::Frame(k, vec![0u8; 16])));
    }
    write_nut(
        &dir.join("in.nut"),
        &[video(2, 2, TENTHS, (10, 1)), audio],
        &items,
    );
    let input = dir.join("in.nut").display().to_string();
    let out = at_every_jobs(
        "window",
        &args(&[
            "-f",
            "nut",
            "-i",
            &input,
            "-m",
            &module("shape_window"),
            "-m",
            &module("shape_state"),
            "-filter_complex",
            "[a=0:a]shape_window=window=48000[cues=c];[v=0:v][words=c]shape_state[seen=s]",
            "-map",
            "[s]",
            "-f",
            "ndjson",
            "{dir}/seen.ndjson",
            "-map",
            "[c]",
            "-f",
            "ndjson",
            "{dir}/cues.ndjson",
        ]),
        &["seen.ndjson", "cues.ndjson"],
    );
    let cues = lines(&out[1]);
    let windows: Vec<(f64, f64)> = cues
        .iter()
        .map(|c| (c["start_t"].as_f64().unwrap(), c["end_t"].as_f64().unwrap()))
        .collect();
    assert_eq!(
        windows,
        vec![(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.0)]
    );
    let ticks = lines(&out[0]);
    assert_eq!(ticks.len(), 40);
    let seen = |n: usize| (ticks[n]["seen"].as_u64(), ticks[n]["now"].as_u64());
    assert_eq!(
        seen(0),
        (Some(1), Some(1)),
        "the first window's cue starts at the first tick"
    );
    assert_eq!(seen(9), (Some(1), Some(0)));
    assert_eq!(
        seen(10),
        (Some(2), Some(1)),
        "the second window's cue lands on its own tick"
    );
    assert_eq!(seen(39), (Some(4), Some(0)));
}

#[test]
fn words_thirty_seconds_late_hold_the_picture_in_the_host_and_not_in_the_node() {
    let dir = scratch("words-late");
    let audio = Stream::audio("f32", 48000, 1).expect("f32 is carried");
    let mut items: Vec<(usize, Write)> = Vec::new();
    for k in 0..750i64 {
        items.push((1, Write::Frame(k * 4800, vec![0u8; 4800 * 4])));
        items.push((0, Write::Frame(k, vec![0u8; 16])));
    }
    write_nut(
        &dir.join("in.nut"),
        &[video(2, 2, TENTHS, (10, 1)), audio],
        &items,
    );
    let input = dir.join("in.nut").display().to_string();
    let out = at_every_jobs(
        "words-late",
        &args(&[
            "-f",
            "nut",
            "-i",
            &input,
            "-m",
            &module("shape_window"),
            "-m",
            &module("shape_state"),
            "-filter_complex",
            "[a=0:a]shape_window=window=1440000[cues=c];[v=0:v][words=c]shape_state[seen=s]",
            "-map",
            "[s]",
            "-f",
            "ndjson",
            "{dir}/seen.ndjson",
            "-map",
            "[c]",
            "-f",
            "ndjson",
            "{dir}/cues.ndjson",
        ]),
        &["seen.ndjson", "cues.ndjson"],
    );
    let cues = lines(&out[1]);
    assert_eq!(
        cues.len(),
        3,
        "75 s of sound is two whole windows and a last one"
    );
    let ticks = lines(&out[0]);
    assert_eq!(ticks.len(), 750);
    let seen = |n: usize| (ticks[n]["seen"].as_u64(), ticks[n]["now"].as_u64());
    assert_eq!(
        seen(0),
        (Some(1), Some(1)),
        "the first cue lands on the first tick"
    );
    assert_eq!(seen(299), (Some(1), Some(0)));
    assert_eq!(
        seen(300),
        (Some(2), Some(1)),
        "the second window's cue, 30 s later"
    );
    assert_eq!(seen(600), (Some(3), Some(1)));
    assert_eq!(seen(749), (Some(3), Some(0)));
}

#[test]
fn a_message_later_than_the_latency_bound_is_delivered_at_the_next_tick_and_reported() {
    let dir = scratch("late-message");
    let mut items: Vec<(usize, Write)> = Vec::new();
    for k in 0..30i64 {
        items.push((0, Write::Frame(k, vec![k as u8; 16])));
        if k == 2 {
            items.push((1, Write::Message(2, r#"{"text":"on time"}"#.to_string())));
        }
        if k == 20 {
            items.push((1, Write::Message(5, r#"{"text":"late"}"#.to_string())));
        }
    }
    write_nut(
        &dir.join("in.nut"),
        &[video(2, 2, TENTHS, (10, 1)), Stream::json(TENTHS)],
        &items,
    );
    let input = dir.join("in.nut").display().to_string();
    let base = scratch("late-message-run");
    let seen = base.join("seen.ndjson");
    let output = Command::new(env!("CARGO_BIN_EXE_ffrwd-wasm"))
        .args([
            "-f",
            "nut",
            "-i",
            &input,
            "-m",
            &module("shape_state"),
            "-filter_complex",
            "[v=0:v][words=0:d]shape_state=latency=0.5[seen=s]",
            "-map",
            "[s]",
            "-f",
            "ndjson",
            &seen.display().to_string(),
        ])
        .output()
        .expect("spawn ffrwd-wasm");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "{stderr}");
    let ticks = lines(&fs::read(&seen).expect("the output was written"));
    let now = |n: usize| ticks[n]["now"].as_u64();
    assert_eq!(now(2), Some(1), "the message on time lands on its tick");
    assert_eq!(
        now(5),
        Some(0),
        "the tick at 0.5 s was cut once the clock reached 1.0 s"
    );
    let delivered: Vec<usize> = (0..30).filter(|n| now(*n) == Some(1)).collect();
    assert_eq!(
        delivered,
        vec![2, 15],
        "the late message reaches the next tick cut after it arrived, which trails the clock by the bound"
    );
    assert!(
        stderr
            .contains("stamped before the tick at 1.500s arrived after it, the earliest at 0.500s"),
        "{stderr}"
    );
    assert!(
        stderr.contains(r#"ffrwd:row {"at":1.5,"earliest":0.5,"kind":"late""#),
        "{stderr}"
    );
    let _ = fs::remove_dir_all(&dir);
    let _ = fs::remove_dir_all(&base);
}

#[test]
fn coded_packets_clock_a_node_one_tick_each() {
    let dir = scratch("packets");
    let coded = Stream {
        fourcc: b"H264".to_vec(),
        time_base: TimeBase { num: 1, den: 25 },
        msb_pts_shift: 14,
        max_pts_distance: 25,
        decode_delay: 0,
        extradata: Vec::new(),
        frame_rate: Some((25, 1)),
        media: Media::Video {
            width: 16,
            height: 16,
            sample_width: 1,
            sample_height: 1,
            colorspace_type: 0,
        },
    };
    let items: Vec<(usize, Write)> = (0..10i64)
        .map(|k| (0, Write::Coded(k, k % 5 == 0, vec![0u8; 10 + k as usize])))
        .collect();
    write_nut(&dir.join("in.nut"), &[coded], &items);
    let input = dir.join("in.nut").display().to_string();
    let out = at_every_jobs(
        "packets",
        &args(&[
            "-f",
            "nut",
            "-i",
            &input,
            "-m",
            &module("shape_packets"),
            "-filter_complex",
            "[p=0:v]shape_packets[log=l]",
            "-map",
            "[l]",
            "-f",
            "ndjson",
            "{dir}/log.ndjson",
        ]),
        &["log.ndjson"],
    );
    let log = lines(&out[0]);
    assert_eq!(log.len(), 10);
    let entry = |n: usize| {
        (
            log[n]["tick"].as_i64(),
            log[n]["dts"].as_i64(),
            log[n]["key"].as_bool(),
            log[n]["bytes"].as_u64(),
        )
    };
    assert_eq!(entry(0), (Some(0), Some(0), Some(true), Some(10)));
    assert_eq!(entry(5), (Some(5), Some(5), Some(true), Some(15)));
    assert_eq!(entry(9), (Some(9), Some(9), Some(false), Some(19)));
    assert_eq!(log[9]["last"], true, "the last packet rides the last call");
}

#[test]
fn a_packets_output_with_no_format_carries_its_clocks_stream_and_rate() {
    let dir = scratch("packets-out");
    // The header says nothing is reordered and the pts do reorder, as
    // libx265 written straight to NUT leaves them: every dts the wire
    // implies is its pts, and the ones that would go back are not known.
    let coded = Stream {
        fourcc: b"H264".to_vec(),
        time_base: TimeBase { num: 1, den: 25 },
        msb_pts_shift: 14,
        max_pts_distance: 25,
        decode_delay: 0,
        extradata: vec![1, 2, 3, 4],
        frame_rate: Some((25, 1)),
        media: Media::Video {
            width: 16,
            height: 16,
            sample_width: 1,
            sample_height: 1,
            colorspace_type: 0,
        },
    };
    let order = [0i64, 3, 1, 2, 6, 4, 5];
    let items: Vec<(usize, Write)> = order
        .iter()
        .map(|&pts| (0, Write::Coded(pts, pts == 0, vec![pts as u8; 8])))
        .collect();
    write_nut(&dir.join("in.nut"), &[coded], &items);
    let input = dir.join("in.nut").display().to_string();
    let out = at_every_jobs(
        "packets-out",
        &args(&[
            "-f",
            "nut",
            "-i",
            &input,
            "-m",
            &module("shape_packets"),
            "-filter_complex",
            "[p=0:v]shape_packets[log=l][out=o]",
            "-map",
            "[l]",
            "-f",
            "ndjson",
            "{dir}/log.ndjson",
            "-map",
            "[o]",
            "-f",
            "nut",
            "{dir}/out.nut",
        ]),
        &["log.ndjson", "out.nut"],
    );
    let dts: Vec<Option<i64>> = lines(&out[0]).iter().map(|l| l["dts"].as_i64()).collect();
    assert_eq!(dts, vec![Some(0), Some(3), None, None, Some(6), None, None]);
    let (streams, packets) = frames(&out[1]);
    assert_eq!(streams.len(), 1);
    assert_eq!(streams[0].fourcc, b"H264".to_vec());
    assert_eq!(streams[0].extradata, vec![1, 2, 3, 4]);
    assert_eq!(streams[0].frame_rate, Some((25, 1)));
    let written: Vec<(i64, Vec<u8>)> = packets.into_iter().map(|(_, pts, d)| (pts, d)).collect();
    let wanted: Vec<(i64, Vec<u8>)> = order.iter().map(|&pts| (pts, vec![pts as u8; 8])).collect();
    assert_eq!(written, wanted);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn leaky_over_a_coded_stream_passes_its_packets_in_time_whole() {
    let dir = scratch("leaky-packets");
    let coded = Stream {
        fourcc: b"H264".to_vec(),
        time_base: TimeBase { num: 1, den: 25 },
        msb_pts_shift: 14,
        max_pts_distance: 25,
        decode_delay: 0,
        extradata: vec![1, 2, 3, 4],
        frame_rate: Some((25, 1)),
        media: Media::Video {
            width: 16,
            height: 16,
            sample_width: 1,
            sample_height: 1,
            colorspace_type: 0,
        },
    };
    let order = [0i64, 3, 1, 2, 6, 4, 5, 7, 10, 8, 9];
    let items: Vec<(usize, Write)> = order
        .iter()
        .map(|&pts| {
            (
                0,
                Write::Coded(pts, pts == 0 || pts == 7, vec![pts as u8; 8]),
            )
        })
        .collect();
    write_nut(&dir.join("in.nut"), &[coded], &items);
    let input = dir.join("in.nut").display().to_string();
    let out = at_every_jobs(
        "leaky-packets",
        &args(&[
            "-f",
            "nut",
            "-i",
            &input,
            "-m",
            &module("shape_packets"),
            "-filter_complex",
            "[p=0:v]shape_packets[out=o];[o]leaky=max_lateness=5[l]",
            "-map",
            "[l]",
            "-f",
            "nut",
            "{dir}/out.nut",
        ]),
        &["out.nut"],
    );
    let (streams, packets) = frames(&out[0]);
    assert_eq!(streams[0].fourcc, b"H264".to_vec());
    let written: Vec<(i64, Vec<u8>)> = packets.into_iter().map(|(_, pts, d)| (pts, d)).collect();
    let wanted: Vec<(i64, Vec<u8>)> = order.iter().map(|&pts| (pts, vec![pts as u8; 8])).collect();
    assert_eq!(written, wanted, "a file read at once is never late");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_pad_tags_a_held_stream_and_a_tagged_anchor_reads_it() {
    let dir = scratch("pad-tags");
    // A tenth of a second of picture and of sound per step, from `from`.
    let steps = |from: i64, to: i64, mark: u8| -> Vec<(usize, Write)> {
        (from..to)
            .flat_map(|k| {
                [
                    (1, Write::Frame(k * 4800, vec![0u8; 4800 * 4])),
                    (0, Write::Frame(k, vec![mark; 64])),
                ]
            })
            .collect()
    };
    let streams = || {
        [
            video(4, 4, TENTHS, (10, 1)),
            Stream::audio("f32", 48000, 1).expect("f32 is carried"),
        ]
    };
    write_nut(&dir.join("prog.nut"), &streams(), &steps(0, 20, 10));
    write_nut(&dir.join("feed.nut"), &streams(), &steps(5, 10, 200));
    let prog = dir.join("prog.nut").display().to_string();
    let ad = dir.join("feed.nut").display().to_string();
    let first_row = |test: &str, pad: &[&str]| {
        let out_dir = dir.join(test);
        fs::create_dir_all(&out_dir).expect("make a run directory");
        let rows = out_dir.join("feeds.ndjson").display().to_string();
        let mut list = vec!["-f", "nut", "-i", &prog, "-f", "nut", "-i", &ad];
        list.extend_from_slice(pad);
        let switch = module("shape_switch");
        list.extend_from_slice(&[
            "-m",
            &switch,
            "-filter_complex",
            "[v=0:v][a=0:a][feed=1:v][feed_audio=1:a]shape_switch=lead=0.3[feeds=f]",
            "-map",
            "[f]",
            "-f",
            "ndjson",
            &rows,
        ]);
        let run = Command::new(env!("CARGO_BIN_EXE_ffrwd-wasm"))
            .args(&list)
            .output()
            .expect("spawn ffrwd-wasm");
        assert!(
            run.status.success(),
            "{}",
            String::from_utf8_lossy(&run.stderr)
        );
        lines(&fs::read(&rows).expect("the rows were written"))[0].clone()
    };
    let timed = first_row(
        "pad-tags-timed",
        &["-pad", r#"{"tags":{"smart_timed":"1"}}"#],
    );
    assert_eq!(timed["timed"], true);
    assert_eq!(
        timed["at"], 5,
        "a timed feed shows when the programme reaches its pts"
    );
    let untimed = first_row("pad-tags-untimed", &[]);
    assert_eq!(untimed["timed"], false);
    assert_ne!(
        untimed["at"], 5,
        "an untimed feed is scheduled by its first frame"
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_bound_feeds_end_is_told_lead_ahead_on_every_run_and_worker_count() {
    let dir = scratch("feed-ends");
    let steps = |to: i64| -> Vec<(usize, Write)> {
        (0..to)
            .flat_map(|k| {
                [
                    (1, Write::Frame(k * 4800, vec![0u8; 4800 * 4])),
                    (0, Write::Frame(k, vec![k as u8; 64])),
                ]
            })
            .collect()
    };
    let streams = || {
        [
            video(4, 4, TENTHS, (10, 1)),
            Stream::audio("f32", 48000, 1).expect("f32 is carried"),
        ]
    };
    write_nut(&dir.join("prog.nut"), &streams(), &steps(80));
    write_nut(&dir.join("ad.nut"), &streams(), &steps(30));
    let prog = dir.join("prog.nut").display().to_string();
    let ad = dir.join("ad.nut").display().to_string();
    let run = |test: &str| {
        at_every_jobs(
            test,
            &args(&[
                "-f",
                "nut",
                "-i",
                &prog,
                "-f",
                "nut",
                "-i",
                &ad,
                "-m",
                &module("shape_switch"),
                "-filter_complex",
                "[v=0:v][a=0:a][feed=1:v][feed_audio=1:a]shape_switch=lead=0.5:timeout=0[feeds=f]",
                "-map",
                "[f]",
                "-f",
                "ndjson",
                "{dir}/feeds.ndjson",
            ]),
            &["feeds.ndjson"],
        )
    };
    let first = run("feed-ends-1");
    assert_eq!(first, run("feed-ends-2"), "a second run differs");
    assert_eq!(first, run("feed-ends-3"), "a third run differs");
    let rows: Vec<(i64, String, Option<i64>)> = lines(&first[0])
        .iter()
        .map(|r| {
            (
                r["pts"].as_i64().unwrap(),
                r["event"].as_str().unwrap().to_string(),
                r["ends"].as_i64(),
            )
        })
        .collect();
    // Primed on the first tick, shown half a second later, its last frame
    // at 5 + 29; the end is told half a second, the lead, before it.
    assert_eq!(
        rows,
        vec![
            (0, "start".to_string(), None),
            (5, "live".to_string(), None),
            (29, "ending".to_string(), Some(34)),
            (35, "end".to_string(), None),
        ]
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_grouped_data_input_bound_to_a_stream_is_placed_with_its_groups_offset() {
    let dir = scratch("group-data");
    let prog: Vec<(usize, Write)> = (0..40)
        .map(|k| (0, Write::Frame(k, vec![0u8; 16])))
        .collect();
    write_nut(
        &dir.join("prog.nut"),
        &[video(2, 2, TENTHS, (10, 1))],
        &prog,
    );
    let mut fed: Vec<(usize, Write)> = (100..120)
        .map(|k| (0, Write::Frame(k, vec![k as u8; 16])))
        .collect();
    fed.insert(3, (1, Write::Message(10_300, r#"{"cue":"a"}"#.to_string())));
    fed.insert(
        11,
        (1, Write::Message(11_000, r#"{"cue":"b"}"#.to_string())),
    );
    write_nut(
        &dir.join("fed.nut"),
        &[
            video(2, 2, TENTHS, (10, 1)),
            Stream::json(TimeBase { num: 1, den: 1000 }),
        ],
        &fed,
    );
    let prog = dir.join("prog.nut").display().to_string();
    let fed = dir.join("fed.nut").display().to_string();
    // `calls` counts an instance's own calls, so the rows are compared
    // without it.
    let run = |jobs: &str| -> Vec<Value> {
        let spots = dir.join(format!("spots-{jobs}.ndjson"));
        run_at(
            jobs,
            &args(&[
                "-f",
                "nut",
                "-i",
                &prog,
                "-f",
                "nut",
                "-i",
                &fed,
                "-m",
                &module("shape_probe"),
                "-filter_complex",
                "[v=0:v][feed=1:v][cues=1:d]shape_probe[spots=s]",
                "-map",
                "[s]",
                "-f",
                "ndjson",
                &spots.display().to_string(),
            ]),
        );
        let mut rows = lines(&fs::read(&spots).expect("the rows were written"));
        for row in &mut rows {
            row.as_object_mut()
                .expect("a row is an object")
                .remove("calls");
        }
        rows
    };
    let rows = run("1");
    assert_eq!(rows, run("2"), "-jobs 1 and -jobs 2 differ");
    assert_eq!(rows, run("4"), "-jobs 1 and -jobs 4 differ");
    assert_eq!(rows.len(), 40);
    // Untimed and primed on the first tick, the picture at 10 s shows half a
    // second, the lead, later: at 5. The rows ride the same offset, from a
    // time base of their own.
    assert_eq!(rows[5]["feed"]["at"], 5);
    assert_eq!(rows[5]["feed"]["first_pts"], 100);
    // Its last picture shows at 24 and lingers a second; that is told the
    // lead before the last picture's turn.
    assert_eq!(rows[18]["feed"]["ends"], Value::Null);
    assert_eq!(rows[19]["feed"]["ends"], 34);
    let cues: Vec<(usize, Value)> = rows
        .iter()
        .enumerate()
        .filter_map(|(n, r)| r.get("cues").map(|c| (n, c.clone())))
        .collect();
    assert_eq!(
        cues,
        vec![
            (8, serde_json::json!([[8, {"cue": "a"}]])),
            (15, serde_json::json!([[15, {"cue": "b"}]])),
        ]
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn one_stream_bound_twice_reaches_both_bindings() {
    let dir = scratch("bound-twice");
    let items: Vec<(usize, Write)> = (0..10)
        .map(|k| (0, Write::Frame(k, vec![10 + k as u8; 64])))
        .collect();
    write_nut(&dir.join("in.nut"), &[video(4, 4, TENTHS, (10, 1))], &items);
    let input = dir.join("in.nut").display().to_string();
    let out = at_every_jobs(
        "bound-twice",
        &args(&[
            "-f",
            "nut",
            "-i",
            &input,
            "-m",
            &module("shape_hold"),
            "-filter_complex",
            "[v=0:v][v=0:v]shape_hold=width=4:height=4:pick=1[out=o]",
            "-map",
            "[o]",
            "-f",
            "nut",
            "{dir}/out.nut",
        ]),
        &["out.nut"],
    );
    let (_, got) = frames(&out[0]);
    let marks: Vec<u8> = got.iter().map(|(_, _, data)| data[0]).collect();
    assert_eq!(
        marks,
        (10..20).collect::<Vec<u8>>(),
        "the second binding hands every frame"
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_source_held_by_a_node_beside_it_runs_no_further_ahead_than_the_hold_takes() {
    let out = at_every_jobs(
        "held-source",
        &args(&[
            "-m",
            &module("shape_rate"),
            "-m",
            &module("shape_hold"),
            "-filter_complex",
            "shape_rate=fps=25:frames=1500:width=8:height=8[out=r];[v=r]shape_hold=width=8:height=8:pick=0[out=o]",
            "-map",
            "[o]",
            "-f",
            "nut",
            "{dir}/out.nut",
        ]),
        &["out.nut"],
    );
    let (_, got) = frames(&out[0]);
    assert_eq!(got.len(), 1500, "every frame of the source is shown once");
    assert!(got
        .iter()
        .enumerate()
        .all(|(n, (_, pts, _))| *pts == n as i64));
}

#[test]
fn a_data_output_written_to_a_file_carries_its_messages_alone() {
    let dir = scratch("quiet-file");
    let input = ten_frames(&dir);
    let out = at_every_jobs(
        "quiet-file",
        &args(&[
            "-f",
            "nut",
            "-i",
            &input,
            "-m",
            &module("shape_state"),
            "-filter_complex",
            "[v=0:v]shape_state[seen=s]",
            "-map",
            "[s]",
            "-f",
            "nut",
            "{dir}/rows.nut",
        ]),
        &["rows.nut"],
    );
    let (_, packets) = frames(&out[0]);
    assert!(!packets.is_empty());
    assert!(
        packets
            .iter()
            .all(|(_, _, data)| !data.iter().all(u8::is_ascii_whitespace)),
        "no progress mark reaches a file"
    );
    let _ = fs::remove_dir_all(&dir);
}

fn ten_frames(dir: &Path) -> String {
    let items: Vec<(usize, Write)> = (0..10)
        .map(|k| (0, Write::Frame(k, vec![k as u8; 64])))
        .collect();
    write_nut(&dir.join("in.nut"), &[video(4, 4, TENTHS, (10, 1))], &items);
    dir.join("in.nut").display().to_string()
}

#[test]
fn several_maps_before_one_output_write_one_nut_in_time_order() {
    let dir = scratch("one-nut");
    let input = ten_frames(&dir);
    let out = at_every_jobs(
        "one-nut",
        &args(&[
            "-f",
            "nut",
            "-i",
            &input,
            "-m",
            &module("shape_canvas"),
            "-m",
            &module("shape_state"),
            "-filter_complex",
            "[v=0:v]shape_canvas=width=2:height=2[out=o];[v=0:v]shape_state[seen=s]",
            "-map",
            "[o]",
            "-map",
            "[s]",
            "-f",
            "nut",
            "{dir}/out.nut",
        ]),
        &["out.nut"],
    );
    let (streams, got) = frames(&out[0]);
    assert_eq!(streams.len(), 2);
    assert_eq!(streams[0].video_geometry(), Some((2, 2)));
    assert!(streams[1].is_json());
    let order: Vec<(usize, i64)> = got
        .iter()
        .filter(|(stream, _, data)| *stream == 0 || !data.iter().all(u8::is_ascii_whitespace))
        .map(|(stream, pts, _)| (*stream, *pts))
        .collect();
    let expected: Vec<(usize, i64)> = (0..10).flat_map(|k| [(0, k), (1, k)]).collect();
    assert_eq!(order, expected, "each tick's frame, then its message");
}

#[test]
fn a_nodes_params_may_come_whole_from_a_file() {
    let dir = scratch("params-from");
    let input = ten_frames(&dir);
    let params = dir.join("canvas.json");
    fs::write(&params, r#"{"width":1,"height":3}"#).expect("write params");
    let out = at_every_jobs(
        "params-from",
        &args(&[
            "-f",
            "nut",
            "-i",
            &input,
            "-m",
            &module("shape_canvas"),
            "-params-from",
            &format!("shape_canvas={}", params.display()),
            "-filter_complex",
            "[v=0:v]shape_canvas=width=2:height=2[out=o]",
            "-map",
            "[o]",
            "-f",
            "nut",
            "{dir}/out.nut",
        ]),
        &["out.nut"],
    );
    let (streams, _) = frames(&out[0]);
    assert_eq!(
        streams[0].video_geometry(),
        Some((1, 3)),
        "the file's params win"
    );
}

#[test]
fn a_node_pad_that_names_no_port_is_refused_naming_the_pad() {
    let dir = scratch("no-port");
    let input = ten_frames(&dir);
    let output = Command::new(env!("CARGO_BIN_EXE_ffrwd-wasm"))
        .args([
            "-f",
            "nut",
            "-i",
            &input,
            "-m",
            &module("shape_canvas"),
            "-filter_complex",
            "[0:v]shape_canvas=width=2:height=2[out=o]",
            "-map",
            "[o]",
            "-f",
            "nut",
        ])
        .arg(dir.join("out.nut"))
        .output()
        .expect("spawn ffrwd-wasm");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success());
    assert!(
        stderr.contains("[0:v] names no port") && stderr.contains("[<port>=0:v]"),
        "{stderr}"
    );
}

/// A loopback port nothing listens on now.
fn free_port() -> u16 {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind a port");
    listener.local_addr().expect("its address").port()
}

/// One run of the sidecar at `jobs`, which has to succeed.
fn run_at(jobs: &str, args: &[String]) -> std::process::Output {
    let output = Command::new(env!("CARGO_BIN_EXE_ffrwd-wasm"))
        .arg("-jobs")
        .arg(jobs)
        .args(args)
        .output()
        .expect("spawn ffrwd-wasm");
    assert!(
        output.status.success(),
        "-jobs {jobs} exited {:?}:
{}",
        output.status.code(),
        String::from_utf8_lossy(&output.stderr)
    );
    output
}

#[test]
fn every_worker_numbers_a_tick_by_its_ordinal_in_the_run() {
    let dir = scratch("ordinal");
    let items: Vec<(usize, Write)> = (0..40)
        .map(|k| (0, Write::Frame(k, vec![k as u8; 64])))
        .collect();
    write_nut(&dir.join("in.nut"), &[video(4, 4, TENTHS, (10, 1))], &items);
    let input = dir.join("in.nut").display().to_string();
    let chain = format!("[v=0:v]shape_probe=port={}[spots=s]", free_port());
    let mut runs = Vec::new();
    for jobs in ["1", "4"] {
        let spots = dir.join(format!("spots-{jobs}.ndjson"));
        run_at(
            jobs,
            &args(&[
                "-f",
                "nut",
                "-i",
                &input,
                "-m",
                &module("shape_probe"),
                "-filter_complex",
                &chain,
                "-map",
                "[s]",
                "-f",
                "ndjson",
                &spots.display().to_string(),
            ]),
        );
        runs.push(lines(&fs::read(&spots).expect("spots written")));
    }
    for rows in &runs {
        let ordinals: Vec<u64> = rows
            .iter()
            .map(|r| r["ordinal"].as_u64().expect("an ordinal"))
            .collect();
        assert_eq!(ordinals, (0..40).collect::<Vec<u64>>());
    }
    let one: Vec<u64> = runs[0]
        .iter()
        .map(|r| r["calls"].as_u64().unwrap())
        .collect();
    assert_eq!(
        one,
        (1..=40).collect::<Vec<u64>>(),
        "one instance counts every tick"
    );
    let split = runs[1]
        .iter()
        .filter(|r| r["calls"].as_u64().unwrap() != r["ordinal"].as_u64().unwrap() + 1)
        .count();
    assert!(
        split > 0,
        "at -jobs 4 the ticks were spread over instances, whose own counts disagree"
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_frame_of_an_input_read_for_its_timing_alone_cannot_be_fetched() {
    let dir = scratch("timing-fetch");
    let input = ten_frames(&dir);
    let chain = format!(
        "[v=0:v][size=0:v]shape_probe=port={}:fetch_size=true[spots=s]",
        free_port()
    );
    let output = Command::new(env!("CARGO_BIN_EXE_ffrwd-wasm"))
        .args(args(&[
            "-f",
            "nut",
            "-i",
            &input,
            "-m",
            &module("shape_probe"),
            "-filter_complex",
            &chain,
            "-map",
            "[s]",
            "-f",
            "null",
            "-",
        ]))
        .output()
        .expect("spawn ffrwd-wasm");
    assert_eq!(output.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("asked of input 'size', which wants timing"),
        "{stderr}"
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_bound_list_is_what_the_shape_is_asked_with_and_each_stream_is_handed_its_hint() {
    let dir = scratch("bound-list");
    let input = ten_frames(&dir);
    let chain = format!("[v=0:v]shape_probe=port={}[spots=s]", free_port());
    let spelled = |bound: &[&str]| {
        let mut line = args(&[
            "-f",
            "nut",
            "-i",
            &input,
            "-m",
            &module("shape_probe"),
            "-filter_complex",
            &chain,
        ]);
        for list in bound {
            line.push("-bound".to_string());
            line.push(format!("shape_probe={list}"));
        }
        line.extend(args(&["-map", "[s]", "-f", "ndjson", "-"]));
        Command::new(env!("CARGO_BIN_EXE_ffrwd-wasm"))
            .args(line)
            .output()
            .expect("spawn ffrwd-wasm")
    };
    let hints = |output: &std::process::Output| -> Vec<Option<String>> {
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        lines(&output.stdout)
            .iter()
            .map(|r| r["hint"].as_str().map(str::to_string))
            .collect()
    };
    let told = hints(&spelled(&[
        r#"[{"input":"v","streams":[{"rate":{"num":25,"den":1}}]},{"input":"feed"}]"#,
    ]));
    assert_eq!(told.len(), 10);
    assert!(
        told.iter().all(|h| h.as_deref() == Some("25/1")),
        "v is handed the rate the list gives it: {told:?}"
    );
    let untold = hints(&spelled(&[]));
    assert!(
        untold.iter().all(Option::is_none),
        "with no list the pads name the inputs and nothing is known of their rates"
    );

    for (bound, said) in [
        (vec!["[]"], "-bound leaves out input 'v', which a pad binds"),
        (
            vec![r#"[{"input":"v","streams":[{},{}]}]"#],
            "-bound gives input 'v' 2 stream(s), and the pads bind 1",
        ),
        (
            vec![r#"[{"input":"v"},{"input":"a"}]"#],
            "names input 'a', which no pad binds and no port serves",
        ),
        (
            vec![r#"[{"input":"v"}]"#, r#"[{"input":"v"}]"#],
            "-bound shape_probe= is given 2 time(s)",
        ),
        (
            vec![r#"[{"input":"v"},{"input":"nope"}]"#],
            "bound input 'nope'",
        ),
    ] {
        let output = spelled(&bound);
        assert_eq!(output.status.code(), Some(1), "{bound:?}");
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(stderr.contains(said), "{bound:?}: {stderr}");
    }
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_name_called_twice_takes_its_params_files_in_turn() {
    let dir = scratch("params-from-twice");
    let input = ten_frames(&dir);
    let tall = dir.join("tall.json");
    let square = dir.join("square.json");
    fs::write(&tall, r#"{"width":1,"height":3}"#).expect("write params");
    fs::write(&square, r#"{"width":2,"height":2}"#).expect("write params");
    let line = |files: &[&std::path::Path]| {
        let mut line = args(&[
            "-f",
            "nut",
            "-i",
            &input,
            "-m",
            &module("shape_canvas"),
            "-filter_complex",
            "[v=0:v]shape_canvas[out=o1];[v=0:v]shape_canvas[out=o2]",
        ]);
        for file in files {
            line.push("-params-from".to_string());
            line.push(format!("shape_canvas={}", file.display()));
        }
        line.extend(args(&[
            "-map",
            "[o1]",
            "-map",
            "[o2]",
            "-f",
            "nut",
            "{dir}/out.nut",
        ]));
        line
    };
    let spelled: Vec<String> = line(&[&tall, &square, &tall])
        .into_iter()
        .map(|a| a.replace("{dir}", &dir.display().to_string()))
        .collect();
    let output = Command::new(env!("CARGO_BIN_EXE_ffrwd-wasm"))
        .args(spelled)
        .output()
        .expect("spawn ffrwd-wasm");
    assert_eq!(output.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("-params-from shape_canvas= is given 3 time(s), and -filter_complex calls 'shape_canvas' 2 time(s)"),
        "{stderr}"
    );
    let out = at_every_jobs("params-from-twice", &line(&[&tall, &square]), &["out.nut"]);
    let (streams, _) = frames(&out[0]);
    let sizes: Vec<Option<(u32, u32)>> = streams.iter().map(|s| s.video_geometry()).collect();
    assert_eq!(
        sizes,
        vec![Some((1, 3)), Some((2, 2))],
        "the first file to the first chain, the second to the second"
    );

    let _ = fs::remove_dir_all(&dir);
}

/// `count` 64x48 red pictures at 30 fps, each with a grey mark that moves a
/// pixel a frame.
fn marked(dir: &Path, count: i64) -> String {
    let items: Vec<(usize, Write)> = (0..count)
        .map(|k| {
            let mut picture = [200u8, 30, 30, 255].repeat(64 * 48);
            for y in 8..20 {
                for x in (4 + k as usize % 40)..(16 + k as usize % 40) {
                    picture[(y * 64 + x) * 4..(y * 64 + x) * 4 + 4]
                        .copy_from_slice(&[128, 128, 128, 255]);
                }
            }
            (0, Write::Frame(k, picture))
        })
        .collect();
    let thirtieths = TimeBase { num: 1, den: 30 };
    write_nut(
        &dir.join("marked.nut"),
        &[video(64, 48, thirtieths, (30, 1))],
        &items,
    );
    dir.join("marked.nut").display().to_string()
}

#[test]
fn recipe_145_split_across_workers_names_every_sighting_alike() {
    let dir = scratch("recipe-145");
    let input = marked(&dir, 90);
    let bound = r#"[{"input":"v","streams":[{"rate":{"num":30,"den":1}}]}]"#;
    let out = at_every_jobs(
        "recipe-145",
        &args(&[
            "-f",
            "nut",
            "-i",
            &input,
            "-m",
            &module("spot"),
            "-m",
            &module("ring"),
            "-filter_complex",
            "[v=0:v]spot=every=30[spots=n1];[v=0:v][spots=n1]ring[v=out0]",
            "-bound",
            &format!("spot={bound}"),
            "-bound",
            &format!("ring={}", bound.replace("]}]", "]},{\"input\":\"spots\"}]")),
            "-map",
            "[out0]",
            "-f",
            "nut",
            "{dir}/ringed.nut",
            "-map",
            "[n1]",
            "-f",
            "ndjson",
            "{dir}/spots.ndjson",
        ]),
        &["ringed.nut", "spots.ndjson"],
    );
    let spots = lines(&out[1]);
    assert_eq!(spots.len(), 90, "a row a frame while the mark is in view");
    let named: Vec<(u64, f64)> = spots
        .iter()
        .map(|r| (r["id"].as_u64().unwrap(), r["start_t"].as_f64().unwrap()))
        .collect();
    assert_eq!(named[29], (0, 0.0));
    assert_eq!(
        named[30],
        (1, 1.0),
        "a new sighting every 30 frames, from its first"
    );
    assert_eq!(named[89], (2, 2.0));
    let (_, pictures) = frames(&out[0]);
    assert_eq!(pictures.len(), 90);
    let _ = fs::remove_dir_all(&dir);
}
