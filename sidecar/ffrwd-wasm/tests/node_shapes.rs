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
    assert_eq!(
        at(9),
        (9, 104),
        "the last tick holds the picked input's last frame"
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
