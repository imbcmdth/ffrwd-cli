//! The live half of the node world, end to end: a hold input served by a
//! port the host listens on, fed over TCP by this test as a feeder process
//! would, against a programme paced in real time on the sidecar's stdin.
//! What the switch's live cases checked, checked here against the host's own
//! hold pairing and the node test modules: `shape_switch` stands in for the
//! switch and `inset` for recipe 149.
//!
//! Every picture is flat, so one pixel says whose frame it is: the
//! programme's frames are green, the feed's red, and the green channel
//! counts frames. The programme's sound is -1 everywhere and the feed's
//! sound is its own sample index, so a sample of the output says which
//! source it came from and, for the feed, which sample. Nothing here reads
//! the clock to decide: expectations come from the rows the node writes
//! about the feed the host handed it.

use std::fs;
use std::io::{BufWriter, Write};
use std::net::{Ipv4Addr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Mutex, MutexGuard, OnceLock};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use ffrwd_wasm::nut::{self, Event, Limits, Muxer, PushDemuxer, Stream, TimeBase};
use serde_json::Value;

const MODULES: &[&str] = &[
    "shape-switch",
    "inset",
    "shape-state",
    "shape-window",
    "shape-hold",
    "shape-probe",
];

const RATE: u32 = 48_000;

/// One case at a time: each paces a programme and a feeder on the wall
/// clock, and a machine running fifteen of them at once keeps none of
/// them on time.
fn serial() -> MutexGuard<'static, ()> {
    static LOCK: Mutex<()> = Mutex::new(());
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

fn sidecar_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("ffrwd-wasm/ has a parent directory")
        .to_path_buf()
}

/// `-m name=path` for one module, building them all once.
fn module(name: &str) -> String {
    static BUILT: OnceLock<()> = OnceLock::new();
    BUILT.get_or_init(|| {
        let mut args = vec!["build", "--release", "--target", "wasm32-wasip2"];
        for shape in MODULES {
            args.extend(["-p", shape]);
        }
        let output = Command::new("cargo")
            .args(&args)
            .current_dir(sidecar_root().join("modules"))
            .output()
            .expect("spawn cargo build");
        assert!(
            output.status.success(),
            "building the modules failed:\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
    });
    let path = sidecar_root().join(format!("modules/target/wasm32-wasip2/release/{name}.wasm"));
    format!("{name}={}", path.display())
}

fn scratch(test: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("ffrwd-live-{}-{test}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("make a scratch directory");
    dir
}

fn free_port() -> u16 {
    TcpListener::bind((Ipv4Addr::LOCALHOST, 0))
        .expect("a free port")
        .local_addr()
        .expect("its address")
        .port()
}

/// Waits for the sidecar to listen on `port`: a connect that is closed at
/// once, which the host drops as a feeder that never sent a byte.
fn wait_for_port(run: &mut Run, port: u16) {
    let until = Instant::now() + Duration::from_secs(60);
    loop {
        if TcpStream::connect((Ipv4Addr::LOCALHOST, port)).is_ok() {
            return;
        }
        if let Ok(Some(status)) = run.child.try_wait() {
            let stderr = run.stderr.take().map(|t| t.join().unwrap_or_default());
            panic!(
                "ffrwd-wasm exited {status:?} before listening on {port}:
{}",
                stderr.unwrap_or_default()
            );
        }
        assert!(Instant::now() < until, "nothing listened on {port}");
        thread::sleep(Duration::from_millis(20));
    }
}

/// The programme: flat green pictures counting frames, a -1 tone.
#[derive(Clone)]
struct Program {
    width: u32,
    height: u32,
    fps: u32,
    seconds: f64,
    sound: bool,
    /// Programme seconds the picture and sound skip over: a hole in the
    /// clock, the timestamps after it left where they were.
    holes: Vec<(f64, f64)>,
}

impl Program {
    fn new(fps: u32, seconds: f64) -> Program {
        Program {
            width: 32,
            height: 24,
            fps,
            seconds,
            sound: true,
            holes: Vec::new(),
        }
    }

    fn frames(&self) -> i64 {
        (self.seconds * self.fps as f64).round() as i64
    }

    fn streams(&self) -> Vec<Stream> {
        let base = TimeBase {
            num: 1,
            den: u64::from(self.fps),
        };
        let mut video = Stream::video("rgba", self.width, self.height, base).expect("rgba");
        video.frame_rate = Some((u64::from(self.fps), 1));
        let mut streams = vec![video];
        if self.sound {
            let mut audio = Stream::audio("f32", RATE, 1).expect("f32");
            audio.time_base = TimeBase {
                num: 1,
                den: u64::from(RATE),
            };
            streams.push(audio);
        }
        streams
    }

    fn in_hole(&self, seconds: f64) -> bool {
        self.holes
            .iter()
            .any(|(a, b)| seconds >= *a && seconds < *b)
    }

    fn frame(&self, k: i64) -> Vec<u8> {
        let pixel = [0, (k & 255) as u8, ((k >> 8) & 255) as u8, 255];
        pixel.repeat((self.width * self.height) as usize)
    }

    /// Writes the programme to `out` in real time: each frame at its own
    /// wall time from the start, the sound in 1024-sample packets ahead of
    /// the picture it belongs with, as ffmpeg interleaves them.
    fn pace(&self, out: impl Write, started: Instant) {
        let mut muxer = Muxer::with_streams(BufWriter::new(out), &self.streams()).expect("headers");
        let mut sample = 0i64;
        let packet = 1024i64;
        let silence = vec![(-1.0f32).to_le_bytes(); packet as usize].concat();
        for k in 0..self.frames() {
            let at = k as f64 / self.fps as f64;
            if self.in_hole(at) {
                continue;
            }
            let due = started + Duration::from_secs_f64(at);
            let now = Instant::now();
            if due > now {
                thread::sleep(due - now);
            }
            if self.sound {
                let upto = ((k + 1) as f64 / self.fps as f64 * RATE as f64) as i64;
                while sample + packet <= upto {
                    let seconds = sample as f64 / RATE as f64;
                    if !self.in_hole(seconds) && muxer.write_frame_to(1, sample, &silence).is_err()
                    {
                        return;
                    }
                    sample += packet;
                }
            }
            if muxer.write_frame_to(0, k, &self.frame(k)).is_err() {
                return;
            }
            if muxer.flush().is_err() {
                return;
            }
        }
        let _ = muxer.finish();
    }
}

/// A feeder: flat red pictures counting frames, a sound whose every sample
/// is its own index.
#[derive(Clone)]
struct Feeder {
    width: u32,
    height: u32,
    fps: u32,
    seconds: f64,
    /// Its first pts, in seconds of its own clock.
    start: f64,
    /// `smart_timed=1` on its picture: its pts are programme time.
    timed: bool,
    sound: bool,
    /// Whether it sends a picture at all.
    picture: bool,
    pix_fmt: &'static str,
    s16: bool,
    /// Written at its own rate when true, as fast as the socket takes it
    /// otherwise.
    paced: bool,
    /// A pause in the writing: at this many seconds of its material, for
    /// this long.
    stall: Option<(f64, f64)>,
    /// What happens after the last frame: the connection closes, or stays
    /// open and silent for this long.
    hold_open: Option<Duration>,
    /// Stops writing after this many frames and holds the connection
    /// open: a feeder that stalls for good.
    stop_after: Option<i64>,
    /// Rows on a data stream of their own after the picture and sound, each
    /// at a time in milliseconds of the feeder's own clock.
    rows: Vec<(i64, String)>,
}

impl Feeder {
    fn new(fps: u32, seconds: f64) -> Feeder {
        Feeder {
            width: 32,
            height: 24,
            fps,
            seconds,
            start: 0.0,
            timed: false,
            sound: true,
            picture: true,
            pix_fmt: "rgba",
            s16: false,
            paced: true,
            stall: None,
            hold_open: None,
            stop_after: None,
            rows: Vec::new(),
        }
    }

    fn frame(&self, j: i64) -> Vec<u8> {
        let pixels = (self.width * self.height) as usize;
        match self.pix_fmt {
            "rgba" => [255, (j & 255) as u8, ((j >> 8) & 255) as u8, 255].repeat(pixels),
            _ => {
                let mut data = vec![81u8; pixels];
                data.extend(vec![90u8; pixels / 4]);
                data.extend(vec![240u8; pixels / 4]);
                data
            }
        }
    }

    fn streams(&self) -> Vec<Stream> {
        let base = TimeBase {
            num: 1,
            den: u64::from(self.fps),
        };
        let mut video =
            Stream::video(self.pix_fmt, self.width, self.height, base).expect("carried");
        video.frame_rate = Some((u64::from(self.fps), 1));
        let mut streams = if self.picture {
            vec![video]
        } else {
            Vec::new()
        };
        if self.sound {
            let fmt = if self.s16 { "s16" } else { "f32" };
            let mut audio = Stream::audio(fmt, RATE, 1).expect("carried");
            audio.time_base = TimeBase {
                num: 1,
                den: u64::from(RATE),
            };
            streams.push(audio);
        }
        if !self.rows.is_empty() {
            streams.push(Stream::json(TimeBase { num: 1, den: 1000 }));
        }
        streams
    }

    fn samples(&self, from: i64, count: i64) -> Vec<u8> {
        (from..from + count)
            .flat_map(|s| {
                if self.s16 {
                    ((s % 32_000) as i16).to_le_bytes().to_vec()
                } else {
                    (s as f32).to_le_bytes().to_vec()
                }
            })
            .collect()
    }
}

/// What a feeder's thread reports when it is done.
struct Fed {
    /// When its last frame had been written, and how many were.
    written_at: Instant,
    frames: i64,
}

fn put_v(out: &mut Vec<u8>, mut value: u64) {
    let mut groups = Vec::new();
    loop {
        groups.push((value & 0x7f) as u8);
        value >>= 7;
        if value == 0 {
            break;
        }
    }
    while let Some(group) = groups.pop() {
        out.push(if groups.is_empty() {
            group
        } else {
            group | 0x80
        });
    }
}

fn put_s(out: &mut Vec<u8>, value: i64) {
    put_v(out, value.unsigned_abs() * 2 - u64::from(value > 0));
}

fn put_vb(out: &mut Vec<u8>, bytes: &[u8]) {
    put_v(out, bytes.len() as u64);
    out.extend_from_slice(bytes);
}

/// The info packet ffmpeg writes for `-metadata:s:0 smart_timed=1`.
fn timed_info() -> Vec<u8> {
    let mut body = Vec::new();
    put_v(&mut body, 1);
    put_s(&mut body, 0);
    put_v(&mut body, 0);
    put_v(&mut body, 0);
    put_v(&mut body, 1);
    put_vb(&mut body, b"smart_timed");
    put_s(&mut body, -1);
    put_vb(&mut body, b"1");
    let mut packet = Vec::new();
    packet.extend_from_slice(&nut::INFO_STARTCODE.to_be_bytes());
    put_v(&mut packet, body.len() as u64 + 4);
    packet.extend_from_slice(&body);
    packet.extend_from_slice(&nut::crc32(&body).to_be_bytes());
    packet
}

/// Connects to `port` after `after` and writes the feeder.
fn feed(port: u16, feeder: Feeder, after: Duration) -> JoinHandle<Fed> {
    thread::spawn(move || {
        thread::sleep(after);
        let socket = TcpStream::connect((Ipv4Addr::LOCALHOST, port)).expect("connect");
        socket.set_nodelay(true).ok();
        let mut raw = socket.try_clone().expect("clone");
        let mut muxer =
            Muxer::with_streams(BufWriter::new(socket), &feeder.streams()).expect("headers");
        if feeder.timed {
            raw.write_all(&timed_info()).expect("the timed tag");
        }
        let started = Instant::now();
        let sound = usize::from(feeder.picture);
        let frames = (feeder.seconds * feeder.fps as f64).round() as i64;
        let mut sample = (feeder.start * RATE as f64).round() as i64;
        let first_sample = sample;
        let packet = 1024i64;
        let mut written = 0;
        let mut stalled = false;
        let data = usize::from(feeder.picture) + usize::from(feeder.sound);
        let mut rows = feeder.rows.iter().peekable();
        for j in 0..frames {
            if feeder.stop_after.is_some_and(|n| j >= n) {
                break;
            }
            let at = j as f64 / feeder.fps as f64;
            if let Some((stall_at, stall_for)) = feeder.stall {
                if at >= stall_at && !stalled {
                    stalled = true;
                    thread::sleep(Duration::from_secs_f64(stall_for));
                }
            }
            if feeder.paced {
                let due = started + Duration::from_secs_f64(at);
                let now = Instant::now();
                if due > now {
                    thread::sleep(due - now);
                }
            }
            let pts = ((feeder.start + at) * feeder.fps as f64).round() as i64;
            if feeder.sound {
                let upto = first_sample + ((j + 1) as f64 / feeder.fps as f64 * RATE as f64) as i64;
                while sample + packet <= upto {
                    if muxer
                        .write_frame_to(sound, sample, &feeder.samples(sample, packet))
                        .is_err()
                    {
                        return Fed {
                            written_at: Instant::now(),
                            frames: written,
                        };
                    }
                    sample += packet;
                }
            }
            let millis = ((feeder.start + at) * 1000.0).round() as i64;
            while let Some((row_at, text)) = rows.next_if(|row| row.0 <= millis) {
                let packet = nut::Packet {
                    pts: *row_at,
                    dts: Some(*row_at),
                    keyframe: true,
                };
                if muxer
                    .write_coded_to(data, &packet, text.as_bytes())
                    .is_err()
                {
                    break;
                }
            }
            if feeder.picture && muxer.write_frame_to(0, pts, &feeder.frame(j)).is_err() {
                break;
            }
            if muxer.flush().is_err() {
                break;
            }
            written += 1;
        }
        let total = first_sample + (feeder.seconds * RATE as f64).round() as i64;
        if feeder.sound && feeder.stop_after.is_none() && sample < total {
            let _ = muxer.write_frame_to(sound, sample, &feeder.samples(sample, total - sample));
            let _ = muxer.flush();
        }
        let written_at = Instant::now();
        match feeder.hold_open {
            Some(open) => thread::sleep(open),
            None => {
                let _ = muxer.finish();
            }
        }
        Fed {
            written_at,
            frames: written,
        }
    })
}

/// Whose frame a picture of the output is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Shown {
    Program(i64),
    Feed(i64),
    Blank,
}

/// What a run wrote.
struct Output {
    /// Every picture, as (pts, whose).
    frames: Vec<(i64, Shown)>,
    /// Per picture, whether its lower right corner is the feed's: where
    /// `inset` draws the feed.
    corners: Vec<(i64, bool)>,
    /// The sound laid out by sample index from 0: -1 is the programme's, a
    /// feed sample is its own index, 0 is silence, NaN never arrived.
    sound: Vec<f32>,
    feeds: Vec<Value>,
    clock: Vec<Value>,
    stderr: String,
}

impl Output {
    /// The pts of every frame showing the feed, in order.
    fn feed_frames(&self) -> Vec<(i64, i64)> {
        self.frames
            .iter()
            .filter_map(|(pts, shown)| match shown {
                Shown::Feed(j) => Some((*pts, *j)),
                _ => None,
            })
            .collect()
    }

    /// Stretches of consecutive frames showing the feed: (first pts, last pts).
    fn insertions(&self) -> Vec<(i64, i64)> {
        let mut out: Vec<(i64, i64)> = Vec::new();
        let mut open: Option<(i64, i64)> = None;
        for (pts, shown) in &self.frames {
            match (shown, open) {
                (Shown::Feed(_), None) => open = Some((*pts, *pts)),
                (Shown::Feed(_), Some((a, _))) => open = Some((a, *pts)),
                (_, Some(span)) => {
                    out.push(span);
                    open = None;
                }
                _ => {}
            }
        }
        if let Some(span) = open {
            out.push(span);
        }
        out
    }

    fn feed_rows(&self, event: &str) -> Vec<&Value> {
        self.feeds.iter().filter(|r| r["event"] == event).collect()
    }

    fn sample(&self, index: i64) -> f32 {
        self.sound.get(index as usize).copied().unwrap_or(f32::NAN)
    }
}

fn lines(path: &Path) -> Vec<Value> {
    fs::read_to_string(path)
        .unwrap_or_default()
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).expect("a JSON line"))
        .collect()
}

fn read_output(dir: &Path, stderr: String) -> Output {
    let wire = fs::read(dir.join("out.nut")).unwrap_or_default();
    let mut demuxer = PushDemuxer::new(Limits::default());
    demuxer.feed(&wire);
    if wire.is_empty() {
        return Output {
            frames: Vec::new(),
            corners: Vec::new(),
            sound: Vec::new(),
            feeds: lines(&dir.join("feeds.ndjson")),
            clock: lines(&dir.join("clock.ndjson")),
            stderr,
        };
    }
    demuxer.finish();
    let mut frames = Vec::new();
    let mut corners = Vec::new();
    let mut packets: Vec<(i64, Vec<u8>)> = Vec::new();
    let mut audio_index = None;
    while let Some(event) = demuxer.next_event().expect("the output parses") {
        match event {
            Event::EndOfHeaders => {
                audio_index = demuxer
                    .streams()
                    .iter()
                    .position(|s| s.as_ref().is_some_and(|s| s.sample_fmt().is_some()));
            }
            Event::Frame { stream, packet } => {
                let data = demuxer.payload();
                if Some(stream) == audio_index {
                    packets.push((packet.pts, data.to_vec()));
                } else {
                    let shown = match (data[0], data[3]) {
                        (250.., _) => Shown::Feed(i64::from(data[1]) | (i64::from(data[2]) << 8)),
                        (0, 0) => Shown::Blank,
                        _ => Shown::Program(i64::from(data[1]) | (i64::from(data[2]) << 8)),
                    };
                    frames.push((packet.pts, shown));
                    let corner = (22 * 32 + 30) * 4;
                    corners.push((packet.pts, data.len() > corner && data[corner] >= 250));
                }
            }
            Event::EndOfInput => break,
            _ => {}
        }
    }
    let mut sound: Vec<f32> = Vec::new();
    for (pts, data) in packets {
        let at = pts as usize;
        let count = data.len() / 4;
        if sound.len() < at + count {
            sound.resize(at + count, f32::NAN);
        }
        for (k, chunk) in data.as_chunks::<4>().0.iter().enumerate() {
            sound[at + k] = f32::from_le_bytes(*chunk);
        }
    }
    Output {
        frames,
        corners,
        sound,
        feeds: lines(&dir.join("feeds.ndjson")),
        clock: lines(&dir.join("clock.ndjson")),
        stderr,
    }
}

/// One run of `shape_switch` (or whatever `chain` names) on a paced
/// programme, with its outputs under `dir`.
struct Run {
    child: Child,
    dir: PathBuf,
    pacer: JoinHandle<()>,
    stderr: Option<JoinHandle<String>>,
}

fn start(test: &str, program: &Program, jobs: &str, modules: &[&str], chain: &str) -> (Run, u16) {
    start_padded(test, program, jobs, modules, chain, &[])
}

/// `start`, with `pad` after the programme's `-i`.
fn start_padded(
    test: &str,
    program: &Program,
    jobs: &str,
    modules: &[&str],
    chain: &str,
    pad: &[&str],
) -> (Run, u16) {
    let dir = scratch(test);
    let port = free_port();
    let chain = chain.replace("{port}", &port.to_string());
    let mut command = Command::new(env!("CARGO_BIN_EXE_ffrwd-wasm"));
    command.args(["-jobs", jobs, "-f", "nut", "-i", "pipe:0"]);
    command.args(pad);
    for name in modules {
        command.arg("-m").arg(module(name));
    }
    let out = dir.join("out.nut").display().to_string();
    let feeds = dir.join("feeds.ndjson").display().to_string();
    let clock = dir.join("clock.ndjson").display().to_string();
    command.args(["-filter_complex", &chain]);
    command.args(["-map", "[o]"]);
    if program.sound && chain.contains("[a=p]") {
        command.args(["-map", "[p]"]);
    }
    if program.width >= 1280 {
        command.args(["-f", "null", &out]);
    } else {
        command.args(["-f", "nut", &out]);
    }
    if chain.contains("[feeds=f]") || chain.contains("[spots=f]") {
        command.args(["-map", "[f]", "-f", "ndjson", &feeds]);
    }
    if chain.contains("[clock=c]") {
        command.args(["-map", "[c]", "-f", "ndjson", &clock]);
    }
    command.stdin(Stdio::piped());
    command.stdout(Stdio::null());
    command.stderr(Stdio::piped());
    let mut child = command.spawn().expect("spawn ffrwd-wasm");
    let stdin = child.stdin.take().expect("stdin");
    let stderr = child.stderr.take().expect("stderr");
    let stderr = thread::spawn(move || {
        let mut text = String::new();
        let mut reader = std::io::BufReader::new(stderr);
        std::io::Read::read_to_string(&mut reader, &mut text).ok();
        text
    });
    let program = program.clone();
    let started = Instant::now();
    let pacer = thread::spawn(move || program.pace(stdin, started));
    (
        Run {
            child,
            dir,
            pacer,
            stderr: Some(stderr),
        },
        port,
    )
}

fn finish(run: Run) -> Output {
    let Run {
        mut child,
        dir,
        pacer,
        stderr,
    } = run;
    let _ = pacer.join();
    let status = child.wait().expect("wait");
    let stderr = stderr
        .map(|t| t.join().unwrap_or_default())
        .unwrap_or_default();
    assert!(status.success(), "ffrwd-wasm exited {status:?}:\n{stderr}");
    let out = read_output(&dir, stderr);
    let _ = fs::remove_dir_all(&dir);
    out
}

const SWITCH: &str = "[v=0:v][a=0:a]shape_switch=port={port}:lead=0.3[v=o][a=p][feeds=f][clock=c]";

fn switch_chain(options: &str) -> String {
    format!("[v=0:v][a=0:a]shape_switch=port={{port}}:{options}[v=o][a=p][feeds=f][clock=c]")
}

/// The feeds row that starts the first feed, and what it says.
fn first_start(out: &Output) -> (i64, i64, bool) {
    let start = out
        .feeds
        .iter()
        .find(|r| r["event"] != "end")
        .unwrap_or_else(|| panic!("no feed started:\n{}", out.stderr));
    (
        start["at"].as_i64().unwrap(),
        start["first_pts"].as_i64().unwrap(),
        start["timed"].as_bool().unwrap(),
    )
}

#[test]
fn a_feed_takes_over_picture_and_sound_together_and_the_programme_comes_back() {
    let _serial = serial();
    let program = Program::new(30, 8.0);
    let (mut run, port) = start("replace", &program, "1", &["shape_switch"], SWITCH);
    wait_for_port(&mut run, port);
    let fed = feed(port, Feeder::new(25, 3.0), Duration::from_secs(2));
    let out = finish(run);
    fed.join().expect("the feeder");

    assert_eq!(out.frames.len() as i64, program.frames(), "{}", out.stderr);
    assert!(
        out.frames
            .iter()
            .enumerate()
            .all(|(k, (pts, _))| *pts == k as i64),
        "every frame carries the programme's own timestamp"
    );
    let (at, first_pts, timed) = first_start(&out);
    assert!(!timed);
    assert_eq!(first_pts, 0, "mapped from the feeder's first picture");
    let insertions = out.insertions();
    assert_eq!(
        insertions.len(),
        1,
        "the feed was on the screen exactly once ({insertions:?})\n{}",
        out.stderr
    );
    let (first, last) = insertions[0];
    assert_eq!(first, at, "the feed comes up on the tick its start names");
    let lasted = (last - first + 1) as f64 / 30.0;
    assert!(
        (lasted - 3.0).abs() < 0.2,
        "the insertion lasted about 3 s ({lasted:.2}s)"
    );
    let shown = out.feed_frames();
    assert_eq!(shown[0].1, 0, "its first frame first");
    assert!(
        shown
            .windows(2)
            .all(|w| w[1].1 >= w[0].1 && w[1].1 - w[0].1 <= 1),
        "the feed's frames show in order, none skipped: {shown:?}"
    );
    assert!(shown.iter().any(|(_, j)| *j == 74), "its last frame showed");

    let ends = out.feed_rows("ending");
    assert!(
        !ends.is_empty(),
        "the return was foretold once the feeder closed\n{}",
        out.stderr
    );
    let foretold = ends.last().unwrap()["ends"].as_i64().unwrap();
    assert_eq!(
        foretold, last,
        "where the picture returned is where it was foretold"
    );

    let cut = at * 1600;
    assert_eq!(
        out.sample(cut - 1),
        -1.0,
        "the sample before the cut is the programme's"
    );
    for k in [0i64, 1, 1000, 47_999, 48_000, 100_000, 143_999] {
        assert_eq!(
            out.sample(cut + k),
            k as f32,
            "programme sample {k} past the cut is the feeder's sample {k}"
        );
    }
    let back = (last + 1) * 1600;
    assert_eq!(
        out.sample(back),
        -1.0,
        "the programme's sound is back on the frame its picture is"
    );
    assert_ne!(out.sample(back - 1), -1.0, "and not before");
    assert!(
        out.clock.len() >= 7,
        "a clock row a second ({})",
        out.clock.len()
    );
}

#[test]
fn a_feeder_that_sends_only_sound_feeds_the_sound_and_leaves_the_picture() {
    let _serial = serial();
    let program = Program::new(30, 8.0);
    let (mut run, port) = start("sound-only", &program, "1", &["shape_switch"], SWITCH);
    wait_for_port(&mut run, port);
    let feeder = Feeder {
        picture: false,
        ..Feeder::new(25, 3.0)
    };
    let fed = feed(port, feeder, Duration::from_secs(2));
    let out = finish(run);
    fed.join().expect("the feeder");

    assert!(!out.stderr.contains("refused"), "{}", out.stderr);
    assert_eq!(out.frames.len() as i64, program.frames(), "{}", out.stderr);
    assert!(
        out.frames
            .iter()
            .all(|(pts, shown)| *shown == Shown::Program(*pts)),
        "the programme's picture throughout"
    );
    assert!(
        out.stderr.contains(r#""input":"feed_audio""#),
        "{}",
        out.stderr
    );
    assert!(
        out.stderr.contains(r#""absent":["feed"]"#),
        "{}",
        out.stderr
    );
    let (at, first_pts, _) = first_start(&out);
    assert_eq!(first_pts, 0, "mapped from the feeder's first sample");
    let cut = at * 1600;
    assert_eq!(
        out.sample(cut - 1),
        -1.0,
        "the programme's sound before the cut"
    );
    for k in [0i64, 1, 1000, 47_999, 48_000, 100_000] {
        assert_eq!(
            out.sample(cut + k),
            k as f32,
            "programme sample {k} past the cut is the feeder's sample {k}"
        );
    }
}

#[test]
fn a_stall_repeats_the_last_frame_and_the_mapping_holds_through_it() {
    let _serial = serial();
    let program = Program::new(30, 9.0);
    let (mut run, port) = start("sync", &program, "1", &["shape_switch"], SWITCH);
    wait_for_port(&mut run, port);
    let mut feeder = Feeder::new(25, 4.0);
    feeder.stall = Some((1.5, 0.9));
    let fed = feed(port, feeder, Duration::from_secs(2));
    let out = finish(run);
    fed.join().expect("the feeder");

    let (at, _, _) = first_start(&out);
    let shown = out.feed_frames();
    let mut longest = 0;
    let mut run = 1;
    for w in shown.windows(2) {
        run = if w[1].1 == w[0].1 { run + 1 } else { 1 };
        longest = longest.max(run);
    }
    assert!(
        longest >= 6,
        "the stall repeated the last frame ({longest} ticks)\n{}",
        out.stderr
    );
    let skipped = shown.windows(2).filter(|w| w[1].1 - w[0].1 > 3).count();
    assert!(skipped >= 1, "and the catch-up skipped forward ({skipped})");
    for (pts, j) in &shown {
        let into = (*pts - at) as f64 / 30.0;
        let frame = *j as f64 / 25.0;
        assert!(
            frame <= into + 1e-9 && into - frame < 0.9 + 1.0 / 25.0,
            "frame {j} at tick {pts} is the newest at or before the tick, or the one held through the stall"
        );
    }
    let cut = at * 1600;
    for k in [10_000i64, 40_000, 120_000, 190_000] {
        let value = out.sample(cut + k);
        assert!(
            value == k as f32 || value == 0.0,
            "the feeder's sample {k} is on its own sample or silence, never shifted (got {value})"
        );
    }
    assert_eq!(
        out.sample(cut + 185_000),
        185_000.0,
        "after the stall the sound is back on its mapping"
    );
    assert!(out.stderr.contains("repeated"), "{}", out.stderr);
}

#[test]
fn a_feeder_that_closes_early_plays_out_and_one_that_wedges_times_out() {
    let _serial = serial();
    let program = Program::new(30, 8.0);
    let (mut run, port) = start(
        "kill",
        &program,
        "1",
        &["shape_switch"],
        &switch_chain("lead=0.3:timeout=1"),
    );
    wait_for_port(&mut run, port);
    let mut feeder = Feeder::new(25, 6.0);
    feeder.stop_after = Some(40);
    feeder.hold_open = Some(Duration::from_secs(6));
    let fed = feed(port, feeder, Duration::from_secs(2));
    let out = finish(run);
    fed.join().expect("the feeder");
    let insertions = out.insertions();
    assert_eq!(insertions.len(), 1, "{insertions:?}\n{}", out.stderr);
    let (first, last) = insertions[0];
    let lasted = (last - first + 1) as f64 / 30.0;
    assert!(
        (lasted - (40.0 / 25.0 + 1.0)).abs() < 0.15,
        "the last frame held for the timeout and no longer ({lasted:.2}s)"
    );
    assert!(out.stderr.contains("stopped sending"), "{}", out.stderr);
    assert!(
        out.feed_rows("ending").is_empty(),
        "a timeout cannot be foretold"
    );

    let program = Program::new(30, 8.0);
    let (mut run, port) = start("kill-eof", &program, "1", &["shape_switch"], SWITCH);
    wait_for_port(&mut run, port);
    let mut feeder = Feeder::new(25, 6.0);
    feeder.stop_after = Some(40);
    let fed = feed(port, feeder, Duration::from_secs(2));
    let out = finish(run);
    fed.join().expect("the feeder");
    let insertions = out.insertions();
    assert_eq!(insertions.len(), 1, "{insertions:?}\n{}", out.stderr);
    let (first, last) = insertions[0];
    let lasted = (last - first + 1) as f64 / 30.0;
    assert!(
        (lasted - 40.0 / 25.0).abs() < 0.1,
        "a closed feeder plays out what it sent ({lasted:.2}s)"
    );
    assert!(out.stderr.contains("the source ended"), "{}", out.stderr);
}

#[test]
fn a_timed_feed_is_held_until_the_programme_reaches_its_pts_and_waits_on_its_socket() {
    let _serial = serial();
    let program = Program::new(30, 10.0);
    let (mut run, port) = start("timed", &program, "1", &["shape_switch"], SWITCH);
    wait_for_port(&mut run, port);
    let mut feeder = Feeder::new(25, 6.0);
    feeder.timed = true;
    feeder.start = 5.0;
    feeder.width = 320;
    feeder.height = 240;
    feeder.paced = false;
    let connected = Instant::now() + Duration::from_secs(1);
    let fed = feed(port, feeder, Duration::from_secs(1));
    let out = finish(run);
    let fed = fed.join().expect("the feeder");

    let (at, first_pts, timed) = first_start(&out);
    assert!(timed);
    assert_eq!((at, first_pts), (150, 125), "held at its own pts");
    let told = out
        .feeds
        .iter()
        .find(|r| r["event"] == "start")
        .expect("a start row");
    assert!(
        told["pts"].as_i64().unwrap() < 120,
        "told well before its time ({})",
        told["pts"]
    );
    let insertions = out.insertions();
    assert_eq!(insertions.len(), 1, "{insertions:?}\n{}", out.stderr);
    assert_eq!(
        insertions[0].0, 150,
        "the picture came in on programme frame 150"
    );
    assert_eq!(
        out.sample(150 * 1600),
        5.0 * 48_000.0,
        "the sound came in at its own pts"
    );
    assert_eq!(out.sample(150 * 1600 - 1), -1.0);
    assert!(
        fed.written_at > connected + Duration::from_secs(3),
        "the feeder waited on its socket while it was held ({:?})",
        fed.written_at - connected
    );
    assert!(
        fed.frames == 150,
        "and sent everything once read on ({})",
        fed.frames
    );
}

#[test]
fn a_timed_feed_whose_time_has_passed_comes_up_at_once_and_skips_to_where_the_programme_is() {
    let _serial = serial();
    let program = Program::new(30, 8.0);
    let (mut run, port) = start("timed-late", &program, "1", &["shape_switch"], SWITCH);
    wait_for_port(&mut run, port);
    let mut feeder = Feeder::new(25, 6.0);
    feeder.timed = true;
    feeder.start = 1.0;
    feeder.paced = false;
    let fed = feed(port, feeder, Duration::from_secs(5));
    let out = finish(run);
    fed.join().expect("the feeder");
    let (at, first_pts, timed) = first_start(&out);
    assert!(timed);
    assert_eq!((at, first_pts), (30, 25));
    let insertions = out.insertions();
    assert_eq!(insertions.len(), 1, "{insertions:?}\n{}", out.stderr);
    let (first, last) = insertions[0];
    assert!(
        first > 60,
        "it came up where the programme was ({first})\n{}",
        out.stderr
    );
    let shown = out.feed_frames();
    assert!(
        shown[0].1 > 5,
        "the frames the programme had passed were skipped ({})\n{}",
        shown[0].1,
        out.stderr
    );
    for (p, j) in shown.iter().skip(2) {
        let into = *p as f64 / 30.0 - 1.0;
        assert!(
            *j as f64 / 25.0 <= into + 1e-9 && into < (*j + 1) as f64 / 25.0 + 1e-9,
            "frame {j} at tick {p} is the newest at or before the tick on the shared clock"
        );
    }
    assert_eq!(last, 209, "and it ended at its own last pts");
    let settled = (first + 10) * 1600;
    assert_eq!(
        out.sample(settled),
        settled as f32,
        "the sound is on its own pts"
    );
}

#[test]
fn a_timed_feed_sent_whole_before_its_time_comes_up_on_its_pts() {
    let _serial = serial();
    let program = Program::new(30, 6.0);
    let (mut run, port) = start("timed-gone", &program, "1", &["shape_switch"], SWITCH);
    wait_for_port(&mut run, port);
    let mut feeder = Feeder::new(25, 0.6);
    feeder.timed = true;
    feeder.start = 3.0;
    feeder.paced = false;
    let fed = feed(port, feeder, Duration::from_secs(1));
    let out = finish(run);
    fed.join().expect("the feeder");
    let insertions = out.insertions();
    assert_eq!(insertions.len(), 1, "{insertions:?}\n{}", out.stderr);
    assert_eq!(insertions[0].0, 90);
    assert_eq!(out.sample(90 * 1600), 3.0 * 48_000.0);
    let ends = out.feed_rows("ending");
    assert!(
        !ends.is_empty(),
        "the end was foretold with the start\n{}",
        out.stderr
    );
    assert_eq!(ends[0]["ends"].as_i64(), Some(insertions[0].1));
}

#[test]
fn a_hole_in_the_programme_goes_out_where_it_was_and_a_jump_ends_the_feed() {
    let _serial = serial();
    let mut program = Program::new(30, 9.0);
    program.holes = vec![(4.0, 4.5)];
    let (mut run, port) = start("holes", &program, "1", &["shape_switch"], SWITCH);
    wait_for_port(&mut run, port);
    let fed = feed(port, Feeder::new(25, 4.0), Duration::from_secs(2));
    let out = finish(run);
    fed.join().expect("the feeder");
    let pts: Vec<i64> = out.frames.iter().map(|(p, _)| *p).collect();
    assert!(
        !pts.contains(&125) && pts.contains(&119) && pts.contains(&135),
        "the hole is where it was"
    );
    assert!(
        out.sound[(4.25 * 48_000.0) as usize].is_nan(),
        "the hole in the sound is where it was"
    );
    assert!(!out.sound[5 * 48_000].is_nan());
    let insertions = out.insertions();
    assert_eq!(
        insertions.len(),
        1,
        "the feed ran over the hole {insertions:?}\n{}",
        out.stderr
    );
    let (at, _, _) = first_start(&out);
    let shown = out.feed_frames();
    for (p, j) in &shown {
        let into = (*p - at) as f64 / 30.0;
        assert!(
            (*j as f64 / 25.0) <= into + 1e-9 && into - (*j as f64 / 25.0) < 0.1,
            "frame {j} at {p} keeps the mapping over the hole"
        );
    }

    let mut program = Program::new(30, 9.0);
    program.holes = vec![(4.0, 5.5)];
    let (mut run, port) = start("jump", &program, "1", &["shape_switch"], SWITCH);
    wait_for_port(&mut run, port);
    let fed = feed(port, Feeder::new(25, 5.0), Duration::from_secs(2));
    let out = finish(run);
    fed.join().expect("the feeder");
    let insertions = out.insertions();
    assert!(
        !insertions.is_empty() && insertions[0].1 < 120,
        "the jump ended the feed {insertions:?}\n{}",
        out.stderr
    );
    assert!(out.stderr.contains("the clock jumped"), "{}", out.stderr);
}

#[test]
fn a_layer_is_blank_while_idle_and_the_feeds_own_bytes_while_live() {
    let _serial = serial();
    let program = Program::new(30, 6.0);
    let (mut run, port) = start(
        "layer",
        &program,
        "1",
        &["shape_switch"],
        &switch_chain("lead=0.3:layer=true"),
    );
    wait_for_port(&mut run, port);
    let fed = feed(port, Feeder::new(25, 2.0), Duration::from_secs(2));
    let out = finish(run);
    fed.join().expect("the feeder");
    assert!(out
        .frames
        .iter()
        .all(|(_, s)| matches!(s, Shown::Blank | Shown::Feed(_))));
    let live = out
        .frames
        .iter()
        .filter(|(_, s)| matches!(s, Shown::Feed(_)))
        .count();
    assert!(
        (50..=62).contains(&live),
        "the feed was on the layer for the insertion ({live})"
    );
}

#[test]
fn a_feeder_at_another_size_and_format_is_conformed_to_the_port() {
    let _serial = serial();
    let program = Program::new(30, 6.0);
    let (mut run, port) = start("conform", &program, "1", &["shape_switch"], SWITCH);
    wait_for_port(&mut run, port);
    let mut feeder = Feeder::new(25, 2.0);
    feeder.width = 64;
    feeder.height = 48;
    feeder.pix_fmt = "yuv420p";
    feeder.s16 = true;
    let fed = feed(port, feeder, Duration::from_secs(2));
    let out = finish(run);
    fed.join().expect("the feeder");
    assert!(
        out.stderr.contains("conformed to 32x24 rgba"),
        "{}",
        out.stderr
    );
    let insertions = out.insertions();
    assert_eq!(insertions.len(), 1, "{insertions:?}\n{}", out.stderr);
    let (at, _, _) = first_start(&out);
    let value = out.sample(at * 1600 + 5000);
    assert!(
        (value - 5000.0 / 32_768.0).abs() < 1e-4,
        "s16 sound arrives as f32 ({value})"
    );
}

#[test]
fn an_rgba_feed_into_an_rgba_programme_needs_no_matrix() {
    let _serial = serial();
    // The compiler names an rgba picture's matrix `gbr`, which converts no
    // yuv; between two rgba pictures nothing is converted at all.
    let program = Program::new(30, 6.0);
    let pad =
        r#"{"color": {"range": "pc", "primaries": "unknown", "trc": "unknown", "space": "gbr"}}"#;
    let (mut run, port) = start_padded(
        "rgba-gbr",
        &program,
        "1",
        &["shape_switch"],
        SWITCH,
        &["-pad", pad],
    );
    wait_for_port(&mut run, port);
    let mut feeder = Feeder::new(25, 2.0);
    feeder.width = 64;
    feeder.height = 48;
    let fed = feed(port, feeder, Duration::from_secs(2));
    let out = finish(run);
    fed.join().expect("the feeder");
    assert!(
        out.stderr.contains("conformed to 32x24 rgba"),
        "{}",
        out.stderr
    );
    let insertions = out.insertions();
    assert_eq!(
        insertions.len(),
        1,
        "{insertions:?}
{}",
        out.stderr
    );
}

#[test]
fn a_many_hold_port_listens_on_every_port_its_list_names() {
    let _serial = serial();
    let program = Program::new(30, 5.0);
    let (first, second) = (free_port(), free_port());
    let dir = scratch("port-list-params");
    let params = dir.join("params.json");
    fs::write(
        &params,
        format!(r#"{{"width":32,"height":24,"pick":1,"ports":[{first},{second}]}}"#),
    )
    .expect("write the params");
    let from = format!("shape_hold={}", params.display());
    let (mut run, _) = start_padded(
        "port-list",
        &program,
        "1",
        &["shape_hold"],
        "[c=0:v]shape_hold[out=o]",
        &["-params-from", &from],
    );
    wait_for_port(&mut run, first);
    wait_for_port(&mut run, second);
    let fed = feed(second, Feeder::new(25, 2.0), Duration::from_secs(1));
    let out = finish(run);
    fed.join().expect("the feeder");
    for port in [first, second] {
        assert!(
            out.stderr.contains(&format!("listens on 127.0.0.1:{port}")),
            "{}",
            out.stderr
        );
    }
    assert!(
        !out.feed_frames().is_empty(),
        "the second port's feed is shown
{}",
        out.stderr
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_second_feeder_is_refused_and_a_silent_one_is_replaced() {
    let _serial = serial();
    let program = Program::new(30, 8.0);
    let (mut run, port) = start("refuse", &program, "1", &["shape_switch"], SWITCH);
    wait_for_port(&mut run, port);
    let mut first = Feeder::new(25, 3.0);
    first.hold_open = Some(Duration::from_secs(2));
    let one = feed(port, first, Duration::from_secs(1));
    let two = feed(port, Feeder::new(25, 1.0), Duration::from_secs(2));
    let out = finish(run);
    one.join().expect("one");
    two.join().expect("two");
    assert!(
        out.stderr.contains("was refused; one at a time"),
        "{}",
        out.stderr
    );
    assert_eq!(out.insertions().len(), 1, "{}", out.stderr);
}

#[test]
fn a_feeder_that_sends_one_frame_and_stalls_is_never_switched_in() {
    let _serial = serial();
    let program = Program::new(30, 5.0);
    let (mut run, port) = start("one-frame", &program, "1", &["shape_switch"], SWITCH);
    wait_for_port(&mut run, port);
    let mut feeder = Feeder::new(25, 3.0);
    feeder.stop_after = Some(1);
    feeder.hold_open = Some(Duration::from_secs(9));
    let _fed = feed(port, feeder, Duration::from_secs(1));
    let out = finish(run);
    assert!(
        out.insertions().is_empty(),
        "{:?}\n{}",
        out.insertions(),
        out.stderr
    );
    assert!(out.feeds.is_empty());
}

#[test]
fn a_feed_that_reconnects_within_one_tick_is_two_ticks_apart() {
    let _serial = serial();
    let program = Program::new(30, 8.0);
    let (mut run, port) = start(
        "reconnect",
        &program,
        "1",
        &["shape_switch"],
        &switch_chain("lead=0"),
    );
    wait_for_port(&mut run, port);
    let mut one = Feeder::new(25, 1.0);
    one.paced = false;
    let mut two = Feeder::new(25, 1.0);
    two.paced = false;
    two.start = 10.0;
    let a = feed(port, one, Duration::from_secs(2));
    let b = feed(port, two, Duration::from_millis(2080));
    let out = finish(run);
    a.join().expect("a");
    b.join().expect("b");
    let insertions = out.insertions();
    assert_eq!(
        insertions.len(),
        2,
        "two feeds {insertions:?}\n{}",
        out.stderr
    );
    assert!(
        insertions[1].0 >= insertions[0].1 + 2,
        "a tick between them {insertions:?}"
    );
    assert_eq!(out.feed_rows("end").len(), 2, "{:?}", out.feeds);
}

#[test]
fn the_feed_port_listens_before_the_programme_has_sent_a_byte() {
    let _serial = serial();
    // What writes the programme here waits for the feed's port to accept, as
    // a lateral's writer waits for the switch it feeds, so the port has to
    // listen before the host has read a byte of its inputs.
    let program = Program::new(30, 1.0);
    let dir = scratch("listens-first");
    let port = free_port();
    let out = dir.join("out.nut").display().to_string();
    let mut child = Command::new(env!("CARGO_BIN_EXE_ffrwd-wasm"))
        .args(["-f", "nut", "-i", "pipe:0", "-m", &module("shape_switch")])
        .args([
            "-filter_complex",
            &format!("[v=0:v][a=0:a]shape_switch=port={port}[v=o]"),
            "-map",
            "[o]",
            "-f",
            "nut",
            &out,
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn ffrwd-wasm");
    let stdin = child.stdin.take().expect("stdin");
    let writer = {
        let program = program.clone();
        thread::spawn(move || {
            let until = Instant::now() + Duration::from_secs(20);
            while TcpStream::connect((Ipv4Addr::LOCALHOST, port)).is_err() {
                if Instant::now() > until {
                    return false;
                }
                thread::sleep(Duration::from_millis(20));
            }
            program.pace(stdin, Instant::now());
            true
        })
    };
    let listened = writer.join().expect("the writer finished");
    let finished = child.wait_with_output().expect("wait for ffrwd-wasm");
    let stderr = String::from_utf8_lossy(&finished.stderr).into_owned();
    assert!(
        listened,
        "nothing listened on {port} before the programme came:
{stderr}"
    );
    assert!(finished.status.success(), "{stderr}");
    let written = read_output(&dir, stderr);
    assert_eq!(written.frames.len() as i64, program.frames());
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_programme_without_a_feeder_passes_through_untouched() {
    let _serial = serial();
    let program = Program::new(30, 3.0);
    let (mut run, port) = start("alone", &program, "1", &["shape_switch"], SWITCH);
    wait_for_port(&mut run, port);
    let out = finish(run);
    assert_eq!(out.frames.len() as i64, program.frames());
    assert!(out
        .frames
        .iter()
        .enumerate()
        .all(|(k, (p, s))| *p == k as i64 && *s == Shown::Program(k as i64)));
    assert!(out.sound.iter().all(|s| *s == -1.0));
    assert!(out.feeds.is_empty());
    assert_eq!(out.clock.len(), 3);
}

#[test]
fn recipe_149_runs_with_a_feed_and_without_one_at_every_jobs() {
    let _serial = serial();
    for jobs in ["1", "2", "4"] {
        let program = Program::new(30, 5.0);
        let chain = "[v=0:v]inset=port={port}:lead=0.5[v=o]";
        let (mut run, port) = start(&format!("inset-{jobs}"), &program, jobs, &["inset"], chain);
        wait_for_port(&mut run, port);
        let fed = feed(port, Feeder::new(25, 1.5), Duration::from_secs(1));
        let out = finish(run);
        fed.join().expect("the feeder");
        assert_eq!(out.frames.len() as i64, program.frames(), "{}", out.stderr);
        let drawn = out.corners.iter().filter(|(_, red)| *red).count();
        assert!(
            (40..=50).contains(&drawn),
            "the inset was drawn for the feed's 1.5 s ({drawn})\n{}",
            out.stderr
        );
        assert!(out
            .frames
            .iter()
            .all(|(_, s)| matches!(s, Shown::Program(_))));
        assert!(out.stderr.contains("comes up at"), "{}", out.stderr);

        let idle = Program::new(30, 2.0);
        let (mut run, port) = start(
            &format!("inset-idle-{jobs}"),
            &idle,
            jobs,
            &["inset"],
            chain,
        );
        wait_for_port(&mut run, port);
        let out = finish(run);
        assert_eq!(out.frames.len() as i64, idle.frames(), "{}", out.stderr);
        assert!(out.corners.iter().all(|(_, red)| !red));
    }
}

#[test]
fn a_720p_feeder_at_the_programmes_rate_plays_out_in_its_own_time() {
    let _serial = serial();
    throughput(1280, 720);
}

/// The size the suite does not run every time: `cargo test --test live
/// a_1080p -- --ignored --nocapture`.
#[test]
#[ignore]
fn a_1080p_feeder_at_the_programmes_rate_plays_out_in_its_own_time() {
    let _serial = serial();
    throughput(1920, 1080);
}

/// A feeder at the programme's own rate and size, and whether it keeps up:
/// a frame bigger than what the receive buffer holds between two reads
/// would throttle it, and the insertion would stutter and last longer than
/// its material.
fn throughput(width: u32, height: u32) {
    let mut program = Program::new(30, 9.0);
    program.width = width;
    program.height = height;
    let (mut run, port) = start(
        &format!("throughput-{height}"),
        &program,
        "1",
        &["shape_switch"],
        SWITCH,
    );
    wait_for_port(&mut run, port);
    let mut feeder = Feeder::new(30, 5.0);
    feeder.width = width;
    feeder.height = height;
    let started = Instant::now();
    let fed = feed(port, feeder, Duration::from_secs(2));
    let out = finish(run);
    let fed = fed.join().expect("the feeder");
    let (at, _, _) = first_start(&out);
    let end = out.feed_rows("end");
    assert_eq!(end.len(), 1, "{}", out.stderr);
    let lasted = (end[0]["pts"].as_i64().unwrap() - at) as f64 / 30.0;
    assert!(
        (lasted - 5.0).abs() < 0.2,
        "the feeder played out in its own time ({lasted:.2}s)\n{}",
        out.stderr
    );
    let line = out
        .stderr
        .lines()
        .find(|l| l.contains("the feed ended at"))
        .expect("the host said how the feed ended");
    let count = |word: &str| -> u64 {
        line.split(',')
            .find(|part| part.contains(word))
            .and_then(|part| part.split_whitespace().next())
            .and_then(|n| n.parse().ok())
            .unwrap_or(u64::MAX)
    };
    let (shown, repeated) = (count("frames shown"), count("repeated"));
    assert!(
        repeated < 3,
        "almost no picture repeats ({repeated} of {shown})\n{}",
        out.stderr
    );
    let wrote = fed.written_at.duration_since(started).as_secs_f64() - 2.0;
    eprintln!(
        "throughput: 150 frames of {width}x{height} rgba written in {wrote:.2}s, {shown} shown, \
         {repeated} repeated, the insertion lasted {lasted:.2}s"
    );
}

#[test]
fn rows_a_feeder_writes_beside_its_picture_and_sound_land_with_the_groups_offset() {
    let _serial = serial();
    let program = Program::new(30, 3.0);
    let (mut run, port) = start(
        "group-rows",
        &program,
        "2",
        &["shape_probe"],
        "[v=0:v]shape_probe=port={port}[copy=o][spots=f]",
    );
    wait_for_port(&mut run, port);
    let mut feeder = Feeder::new(30, 1.2);
    feeder.start = 10.0;
    feeder.rows = vec![
        (10_200, r#"{"text":"a"}"#.to_string()),
        (10_600, r#"{"text":"b"}"#.to_string()),
    ];
    let fed = feed(port, feeder, Duration::from_millis(300));
    let out = finish(run);
    let _ = fed.join();
    let start = out
        .feeds
        .iter()
        .find_map(|r| r.get("feed").cloned())
        .unwrap_or_else(|| {
            panic!(
                "no feed came up:
{}",
                out.stderr
            )
        });
    let at = start["at"].as_i64().expect("at");
    let first_pts = start["first_pts"].as_i64().expect("first_pts");
    assert_eq!(first_pts, 300, "the feeder's own origin, ten seconds in");
    assert!(
        start["known"].as_i64().expect("known") < at,
        "an untimed feed is fixed a lead before it shows"
    );
    let mut cues: Vec<(i64, i64, String)> = Vec::new();
    for row in &out.feeds {
        let tick = (row["start_t"].as_f64().expect("a tick") * 30.0).round() as i64;
        for cue in row["cues"].as_array().into_iter().flatten() {
            cues.push((
                tick,
                cue[0].as_i64().expect("a pts"),
                cue[1]["text"].as_str().expect("its text").to_string(),
            ));
        }
    }
    assert_eq!(
        cues,
        vec![
            (at + 6, at + 6, "a".to_string()),
            (at + 18, at + 18, "b".to_string()),
        ],
        "each row at the clock time its feeder's picture of that moment shows, on that tick:
{}",
        out.stderr
    );
}
