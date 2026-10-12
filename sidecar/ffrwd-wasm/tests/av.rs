//! libav through the runtime's shim: found, refused, and used to decode
//! the wire's h264 fixture and to encode it again, each checked against
//! what ffmpeg itself makes of the same bytes. Then the host's `decode`
//! and `encode` nodes on a network, and a port feed that arrives coded.
//!
//! libav is not in git, so the cases that need it skip - loudly - when
//! neither `FFRWD_AV_RUNTIME` nor the directory the `ffrwd-wasm` binary was
//! built into holds the four libraries. The cases that compare against
//! ffmpeg skip the same way without ffmpeg and ffprobe on PATH.

use std::net::{Ipv4Addr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use ffrwd_wasm::nut::{Demuxer, Muxer, Packet, Stream, TimeBase};
use ffrwd_wasm_runtime::av::{self, Decoder, Encoder, Hw, Library};

/// The h264 fixture: 64x48, B-frames, a time base of 1/51200.
fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/h264.nut")
}

/// A path in the temp directory, named after this process and the test, so
/// two tests running at once never share one.
fn scratch(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("ffrwd_av_{}_{name}", std::process::id()))
}

fn holds_libav(dir: &Path) -> bool {
    Library::LOAD_ORDER
        .iter()
        .all(|lib| dir.join(lib.file_name()).is_file())
}

/// Where libav is: what the environment says, else beside the binary. The
/// shim in this process is pointed at it once. `None` when neither holds
/// the libraries.
fn runtime_dir() -> Option<PathBuf> {
    static DIR: OnceLock<Option<PathBuf>> = OnceLock::new();
    DIR.get_or_init(|| {
        let named = std::env::var_os(av::RUNTIME_DIR_VAR).map(PathBuf::from);
        let beside = Path::new(env!("CARGO_BIN_EXE_ffrwd-wasm"))
            .parent()
            .map(Path::to_path_buf);
        let dir = [named, beside]
            .into_iter()
            .flatten()
            .find(|dir| holds_libav(dir));
        if let Some(dir) = &dir {
            av::configure(&av::Config {
                runtime_dir: Some(dir.clone()),
            });
        }
        dir
    })
    .clone()
}

/// Says why a test did nothing, since a skip is otherwise silent.
fn announce_skip(what: &str) {
    eprintln!(
        "SKIPPED: {what}. Point {} at a directory holding FFmpeg 9.0's shared \
         libraries ({}) to run it.",
        av::RUNTIME_DIR_VAR,
        Library::LOAD_ORDER.map(|l| l.file_name()).join(", ")
    );
}

fn on_path(tool: &str) -> bool {
    Command::new(tool)
        .arg("-version")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}

fn ffmpeg_on_path() -> bool {
    on_path("ffmpeg") && on_path("ffprobe")
}

fn announce_skip_ffmpeg(what: &str) {
    eprintln!("SKIPPED: {what}. Install ffmpeg and put it (and ffprobe) on PATH to run this test.");
}

fn run_ok(command: &mut Command, what: &str) -> Output {
    let output = command
        .output()
        .unwrap_or_else(|e| panic!("spawn {what}: {e}"));
    assert!(
        output.status.success(),
        "{what} failed (status {:?}):\n{}",
        output.status.code(),
        String::from_utf8_lossy(&output.stderr)
    );
    output
}

/// What ffmpeg decodes `path` to: every picture as yuv420p, one after
/// another, none duplicated or dropped.
fn ffmpeg_rawvideo(path: &Path) -> Vec<u8> {
    run_ok(
        Command::new("ffmpeg")
            .args(["-v", "error", "-i"])
            .arg(path)
            .args([
                "-fps_mode",
                "passthrough",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "yuv420p",
                "-",
            ])
            .stdin(Stdio::null()),
        "ffmpeg",
    )
    .stdout
}

/// The timestamps of the pictures ffmpeg decodes from `path`, in the order
/// it decodes them.
fn ffprobe_pts(path: &Path) -> Vec<i64> {
    let output = run_ok(
        Command::new("ffprobe")
            .args([
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "frame=pts",
                "-of",
                "csv=p=0",
            ])
            .arg(path)
            .stdin(Stdio::null()),
        "ffprobe",
    );
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(|line| line.trim().trim_end_matches(','))
        .filter(|line| !line.is_empty())
        .map(|line| line.parse().expect("ffprobe prints integer pts"))
        .collect()
}

/// Every picture the shim decodes from the fixture, as yuv420p, with its
/// timestamp, and the stream header it came under.
fn decode_fixture(hw: Hw) -> (Stream, Vec<av::DecodedFrame>, String) {
    let file = std::fs::File::open(fixture()).expect("open the h264 fixture");
    let mut demuxer = Demuxer::open(file).expect("read the fixture's NUT headers");
    let stream = demuxer.stream().clone();
    assert_eq!(stream.codec_name(), Some("h264"));

    let mut decoder =
        Decoder::open("h264", &stream.extradata, hw, "yuv420p").expect("open an h264 decoder");
    let mut frames = Vec::new();
    let mut buf = Vec::new();
    while let Some(packet) = demuxer.read_packet(&mut buf).expect("read a NUT packet") {
        decoder
            .send(packet.pts, packet.dts, None, packet.keyframe, &buf)
            .expect("send a packet");
        while let Some(frame) = decoder.receive().expect("receive a picture") {
            frames.push(frame);
        }
    }
    decoder.flush().expect("flush the decoder");
    while let Some(frame) = decoder.receive().expect("receive a picture") {
        frames.push(frame);
    }

    let format = decoder.format().expect("a picture was decoded").clone();
    let (width, height) = stream.video_geometry().expect("a video stream");
    assert_eq!((format.width, format.height), (width, height));
    assert_eq!(format.pix_fmt, "yuv420p");
    let engaged = decoder.engaged().to_string();
    (stream, frames, engaged)
}

/// The pictures decoded, one after another.
fn concatenated(frames: &[av::DecodedFrame]) -> Vec<u8> {
    frames.iter().flat_map(|f| f.data.iter().copied()).collect()
}

#[test]
fn av_info_with_no_libav_prints_the_refusal_as_json() {
    let empty = scratch("empty");
    std::fs::create_dir_all(&empty).expect("create an empty directory");
    let output = Command::new(env!("CARGO_BIN_EXE_ffrwd-wasm"))
        .env(av::RUNTIME_DIR_VAR, &empty)
        .arg("--av-info")
        .output()
        .expect("spawn ffrwd-wasm");
    let _ = std::fs::remove_dir(&empty);
    assert!(
        output.status.success(),
        "--av-info failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );

    let text = String::from_utf8(output.stdout).expect("--av-info is UTF-8");
    assert_eq!(text.trim().lines().count(), 1, "one JSON line:\n{text}");
    let info: serde_json::Value = serde_json::from_str(text.trim()).expect("--av-info is JSON");

    assert_eq!(info["demands"]["avcodec"], 63);
    assert_eq!(info["demands"]["avutil"], 61);
    assert_eq!(info["demands"]["swscale"], 10);
    assert_eq!(info["demands"]["swresample"], 7);
    assert_eq!(info["platform"], ffrwd_wasm_runtime::nn::platform());
    assert_eq!(info["from"], av::RUNTIME_DIR_VAR);
    assert_eq!(info["dir"], empty.to_string_lossy().as_ref());
    assert!(info["avcodec"].is_null(), "{info}");
    assert_eq!(info["decoders"], serde_json::json!([]));

    let error = info["error"].as_str().expect("the refusal is a string");
    let wanted = Library::Avutil.file_name();
    assert!(error.contains(&format!("no {wanted} in")), "{error}");
    assert!(error.contains(&*empty.to_string_lossy()), "{error}");
    assert!(
        error.contains("pip install --force-reinstall ffrwd-wasm"),
        "{error}"
    );
}

#[test]
fn av_info_names_what_the_loaded_libav_offers() {
    let Some(dir) = runtime_dir() else {
        announce_skip("av_info_names_what_the_loaded_libav_offers");
        return;
    };
    let output = run_ok(
        Command::new(env!("CARGO_BIN_EXE_ffrwd-wasm"))
            .env_remove(av::RUNTIME_DIR_VAR)
            .args(["--av-info", "-av-runtime"])
            .arg(&dir),
        "ffrwd-wasm --av-info",
    );
    let text = String::from_utf8(output.stdout).expect("--av-info is UTF-8");
    let info: serde_json::Value = serde_json::from_str(text.trim()).expect("--av-info is JSON");
    assert!(info["error"].is_null(), "{info}");
    assert_eq!(info["from"], "-av-runtime");
    let avcodec = info["avcodec"].as_str().expect("a version string");
    assert!(avcodec.starts_with("63."), "{avcodec}");
    let has = |list: &str, name: &str| {
        info[list]
            .as_array()
            .expect("a list")
            .iter()
            .any(|v| v == name)
    };
    assert!(has("decoders", "h264"), "{info}");
    assert!(has("encoders", "ffv1"), "{info}");
    assert!(info["hw_device_types"].is_array());
}

#[test]
fn a_library_of_another_major_is_refused_naming_both_versions() {
    // Not reachable without a second build of libav, so the check itself.
    av::check_major(Library::Avcodec, (63 << 16) | (1 << 8) | 102).expect("63 is accepted");
    av::check_major(Library::Swscale, 10 << 16).expect("10 is accepted");

    let err =
        av::check_major(Library::Avcodec, (62 << 16) | (28 << 8) | 100).expect_err("62 is refused");
    assert!(matches!(
        err,
        av::Error::Version {
            library: "avcodec",
            demanded: 63,
            ..
        }
    ));
    let text = err.to_string();
    assert!(text.contains("62.28.100"), "{text}");
    assert!(text.contains("demands major 63"), "{text}");

    let err = av::check_major(Library::Avutil, 60 << 16).expect_err("60 is refused");
    assert!(err.to_string().contains("libavutil"), "{err}");
}

#[test]
fn the_shim_decodes_the_fixture_exactly_as_ffmpeg_does() {
    if runtime_dir().is_none() {
        announce_skip("the_shim_decodes_the_fixture_exactly_as_ffmpeg_does");
        return;
    }
    if !ffmpeg_on_path() {
        announce_skip_ffmpeg("the_shim_decodes_the_fixture_exactly_as_ffmpeg_does");
        return;
    }
    let (stream, frames, engaged) = decode_fixture(Hw::None);
    assert_eq!(engaged, "software");

    let (width, height) = stream.video_geometry().expect("a video stream");
    let size = av::frame_len("yuv420p", width, height).expect("a wire format");
    let theirs = ffmpeg_rawvideo(&fixture());
    assert!(!frames.is_empty(), "the fixture decodes to something");
    assert_eq!(theirs.len() % size, 0, "ffmpeg wrote whole pictures");
    assert_eq!(
        frames.len(),
        theirs.len() / size,
        "the same number of pictures"
    );
    for frame in &frames {
        assert_eq!((frame.width, frame.height), (width, height));
        assert_eq!(frame.data.len(), size);
    }

    let pts: Vec<i64> = frames.iter().map(|f| f.pts).collect();
    assert_eq!(
        pts,
        ffprobe_pts(&fixture()),
        "the same timestamps, in order"
    );

    assert!(concatenated(&frames) == theirs, "the same bytes");
}

#[test]
fn hw_auto_decodes_the_fixture_to_the_same_bytes() {
    if runtime_dir().is_none() {
        announce_skip("hw_auto_decodes_the_fixture_to_the_same_bytes");
        return;
    }
    // On a machine with no device this is the software decode again, which
    // is the fall back under test as much as the device is.
    let (_, auto, engaged) = decode_fixture(Hw::Auto);
    let (_, software, _) = decode_fixture(Hw::None);
    eprintln!("hw auto engaged: {engaged}");
    assert!(
        engaged == "software" || Hw::AUTO_ORDER.contains(&engaged.as_str()),
        "{engaged}"
    );
    let pts = |frames: &[av::DecodedFrame]| frames.iter().map(|f| f.pts).collect::<Vec<_>>();
    assert_eq!(pts(&auto), pts(&software));
    assert!(
        concatenated(&auto) == concatenated(&software),
        "the same bytes"
    );
}

#[test]
fn a_named_device_this_libav_lacks_is_refused_by_name() {
    if runtime_dir().is_none() {
        announce_skip("a_named_device_this_libav_lacks_is_refused_by_name");
        return;
    }
    let err = Decoder::open("h264", &[], Hw::Device("nosuchdevice".into()), "yuv420p")
        .err()
        .expect("no such device");
    assert!(
        matches!(&err, av::Error::Hardware { device, .. } if device == "nosuchdevice"),
        "{err}"
    );
}

#[test]
fn an_encoder_option_it_refuses_names_the_option_and_the_encoder() {
    if runtime_dir().is_none() {
        announce_skip("an_encoder_option_it_refuses_names_the_option_and_the_encoder");
        return;
    }
    let options = vec![("nosuchoption".to_string(), "1".to_string())];
    let err = Encoder::open("ffv1", 64, 48, "yuv420p", (1, 25), None, None, &options)
        .err()
        .expect("an unknown option");
    match &err {
        av::Error::Option { encoder, key, .. } => {
            assert_eq!(encoder, "ffv1");
            assert_eq!(key, "nosuchoption");
        }
        other => panic!("expected an option refusal, got {other}"),
    }
}

#[test]
fn ffv1_through_the_shim_reads_back_through_ffmpeg_byte_for_byte() {
    if runtime_dir().is_none() {
        announce_skip("ffv1_through_the_shim_reads_back_through_ffmpeg_byte_for_byte");
        return;
    }
    if !ffmpeg_on_path() {
        announce_skip_ffmpeg("ffv1_through_the_shim_reads_back_through_ffmpeg_byte_for_byte");
        return;
    }
    let (stream, frames, _) = decode_fixture(Hw::None);
    let (width, height) = stream.video_geometry().expect("a video stream");
    let tb = stream.time_base;
    let rate = stream.frame_rate.map(|(n, d)| (n as i64, d as i64));

    let mut encoder = Encoder::open(
        "ffv1",
        width,
        height,
        "yuv420p",
        (tb.num as i64, tb.den as i64),
        rate,
        None,
        &[],
    )
    .expect("open an ffv1 encoder");
    assert_eq!(encoder.pix_fmt(), "yuv420p", "ffv1's first format");
    assert_eq!(encoder.fourcc(), Some(*b"FFV1"));
    assert_eq!(av::fourcc_for_codec(encoder.codec()), Some(*b"FFV1"));

    let mut packets = Vec::new();
    for frame in &frames {
        encoder
            .send(frame.pts, frame.duration, &frame.data)
            .expect("send a picture");
        while let Some(packet) = encoder.receive().expect("receive a packet") {
            packets.push(packet);
        }
    }
    encoder.flush().expect("flush the encoder");
    while let Some(packet) = encoder.receive().expect("receive a packet") {
        packets.push(packet);
    }
    assert_eq!(packets.len(), frames.len(), "ffv1 is one packet a picture");
    assert!(packets[0].keyframe, "the first packet is a keyframe");

    let out = Stream::coded_fourcc(
        "video",
        &encoder.fourcc().expect("a tag"),
        (width, height),
        TimeBase {
            num: tb.num,
            den: tb.den,
        },
        encoder.extradata().to_vec(),
        u64::from(encoder.delay()),
    )
    .expect("a coded stream header");
    let path = scratch("ffv1.nut");
    {
        let file = std::fs::File::create(&path).expect("create the ffv1 NUT");
        let mut muxer = Muxer::new(file, &out).expect("write NUT headers");
        for packet in &packets {
            let header = Packet {
                pts: packet.pts,
                dts: packet.dts,
                keyframe: packet.keyframe,
            };
            muxer
                .write_coded(&header, &packet.data)
                .expect("write a packet");
        }
        muxer.finish().expect("finish the NUT");
    }

    let theirs = ffmpeg_rawvideo(&path);
    let pts = ffprobe_pts(&path);
    let _ = std::fs::remove_file(&path);
    assert_eq!(
        pts,
        frames.iter().map(|f| f.pts).collect::<Vec<_>>(),
        "the same timestamps"
    );
    assert!(theirs == concatenated(&frames), "the same bytes");
}

// The host's `decode` and `encode` nodes, run by the binary on a network
// the way the compiler spells one.

fn sidecar_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("ffrwd-wasm/ has a parent directory")
        .to_path_buf()
}

/// Builds the modules these tests run, once per test binary, in the
/// modules workspace (its own build lock, so no deadlock with this run).
fn built_modules() {
    static BUILT: OnceLock<()> = OnceLock::new();
    BUILT.get_or_init(|| {
        let output = Command::new("cargo")
            .args([
                "build",
                "--release",
                "--target",
                "wasm32-wasip2",
                "-p",
                "invert",
                "-p",
                "shape-switch",
            ])
            .current_dir(sidecar_root().join("modules"))
            .output()
            .expect("spawn cargo build for modules");
        assert!(
            output.status.success(),
            "building the modules failed:\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
    });
}

/// `-m name=path` for a module of the modules workspace.
fn module(name: &str) -> String {
    built_modules();
    let path = sidecar_root().join(format!("modules/target/wasm32-wasip2/release/{name}.wasm"));
    format!("{name}={}", path.display())
}

/// The sidecar on `args`, with libav from `dir` where there is one.
fn sidecar(args: &[&str], dir: Option<&Path>) -> Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_ffrwd-wasm"));
    command.args(args).stdin(Stdio::null());
    match dir {
        Some(dir) => command.env(av::RUNTIME_DIR_VAR, dir),
        None => command.env_remove(av::RUNTIME_DIR_VAR),
    };
    command.output().expect("spawn ffrwd-wasm")
}

fn stderr_of(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

/// Every packet of a NUT file, with its timestamp, and the stream header.
fn demux(path: &Path) -> (Stream, Vec<(i64, Vec<u8>)>) {
    let file = std::fs::File::open(path).expect("open the output");
    let mut demuxer = Demuxer::open(file).expect("read the output's NUT headers");
    let stream = demuxer.stream().clone();
    let mut packets = Vec::new();
    let mut buf = Vec::new();
    while let Some(packet) = demuxer.read_packet(&mut buf).expect("read a NUT packet") {
        packets.push((packet.pts, std::mem::take(&mut buf)));
    }
    (stream, packets)
}

/// What ffmpeg decodes `path` to as rgba, every picture once.
fn ffmpeg_rgba(path: &Path) -> Vec<u8> {
    run_ok(
        Command::new("ffmpeg")
            .args(["-v", "error", "-i"])
            .arg(path)
            .args([
                "-fps_mode",
                "passthrough",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgba",
                "-",
            ])
            .stdin(Stdio::null()),
        "ffmpeg",
    )
    .stdout
}

/// The libav directory, for a test that needs libav and ffmpeg; None,
/// having said why, where either is missing.
fn libav_and_ffmpeg(test: &str) -> Option<PathBuf> {
    let Some(dir) = runtime_dir() else {
        announce_skip(test);
        return None;
    };
    if !ffmpeg_on_path() {
        announce_skip_ffmpeg(test);
        return None;
    }
    Some(dir)
}

/// The fixture's own stream header.
fn fixture_stream() -> Stream {
    let file = std::fs::File::open(fixture()).expect("open the h264 fixture");
    Demuxer::open(file)
        .expect("read the fixture's NUT headers")
        .stream()
        .clone()
}

#[test]
fn decode_alone_writes_the_pictures_ffmpeg_decodes_at_the_same_times() {
    let test = "decode_alone_writes_the_pictures_ffmpeg_decodes_at_the_same_times";
    let Some(dir) = libav_and_ffmpeg(test) else {
        return;
    };
    let out = scratch("decode_alone.nut");
    let output = sidecar(
        &[
            "-f",
            "nut",
            "-i",
            fixture().to_str().expect("UTF-8"),
            "-filter_complex",
            "[0:v]decode=format=yuv420p[out0]",
            "-map",
            "[out0]",
            "-f",
            "nut",
            out.to_str().expect("UTF-8"),
        ],
        Some(&dir),
    );
    let stderr = stderr_of(&output);
    assert!(output.status.success(), "{stderr}");
    assert_eq!(
        stderr.matches("[av] decode h264: software").count(),
        1,
        "one line says where it decodes:\n{stderr}"
    );

    let (header, frames) = demux(&out);
    let _ = std::fs::remove_file(&out);
    let input = fixture_stream();
    assert_eq!(header.pix_fmt(), Some("yuv420p"));
    assert_eq!(header.video_geometry(), Some((64, 48)));
    assert_eq!(header.time_base, input.time_base);
    assert!(input.frame_rate.is_some(), "the fixture states its rate");
    assert_eq!(header.frame_rate, input.frame_rate);

    let pts: Vec<i64> = frames.iter().map(|(pts, _)| *pts).collect();
    assert_eq!(
        pts,
        ffprobe_pts(&fixture()),
        "the same timestamps, in order"
    );
    let ours: Vec<u8> = frames.into_iter().flat_map(|(_, data)| data).collect();
    assert!(ours == ffmpeg_rawvideo(&fixture()), "the same bytes");
}

#[test]
fn coded_in_coded_out_through_an_old_module_is_the_inverted_picture() {
    let test = "coded_in_coded_out_through_an_old_module_is_the_inverted_picture";
    let Some(dir) = libav_and_ffmpeg(test) else {
        return;
    };
    let out = scratch("inverted.nut");
    let invert = module("invert");
    // ffv1 codes rgba input as bgra, losslessly, where it is told to; its
    // first format is yuv420p.
    let chain = "[0:v]decode=format=rgba[decode0];[decode0]invert[n1];\
                 [n1]encode=codec=ffv1:pix_fmt=bgra[out0]";
    let output = sidecar(
        &[
            "-f",
            "nut",
            "-i",
            fixture().to_str().expect("UTF-8"),
            "-m",
            &invert,
            "-filter_complex",
            chain,
            "-map",
            "[out0]",
            "-f",
            "nut",
            out.to_str().expect("UTF-8"),
        ],
        Some(&dir),
    );
    let stderr = stderr_of(&output);
    assert!(output.status.success(), "{stderr}");
    assert_eq!(
        stderr.matches("[av] decode h264: software").count(),
        1,
        "{stderr}"
    );
    assert_eq!(
        stderr.matches("[av] encode ffv1: 64x48 bgra").count(),
        1,
        "{stderr}"
    );

    let (header, packets) = demux(&out);
    assert_eq!(header.fourcc, b"FFV1", "carried under ffv1's own tag");
    assert_eq!(header.pix_fmt(), None, "coded, not raw");
    assert!(
        !header.extradata.is_empty(),
        "ffv1's header rides in the stream's"
    );
    assert_eq!(header.frame_rate, fixture_stream().frame_rate);
    assert_eq!(
        packets.len(),
        ffprobe_pts(&fixture()).len(),
        "one packet a picture"
    );

    let ours = ffmpeg_rgba(&out);
    let _ = std::fs::remove_file(&out);
    // invert flips the three colour channels and keeps alpha.
    let mut wanted = ffmpeg_rgba(&fixture());
    for (index, byte) in wanted.iter_mut().enumerate() {
        if index % 4 != 3 {
            *byte = 255 - *byte;
        }
    }
    assert_eq!(ours.len(), wanted.len(), "as many pictures");
    assert!(ours == wanted, "the inverted pictures, byte for byte");
}

#[test]
fn what_encode_writes_ffmpeg_copies_into_a_file() {
    let Some(dir) = libav_and_ffmpeg("what_encode_writes_ffmpeg_copies_into_a_file") else {
        return;
    };
    let out = scratch("copied.nut");
    let mkv = scratch("copied.mkv");
    let output = sidecar(
        &[
            "-f",
            "nut",
            "-i",
            fixture().to_str().expect("UTF-8"),
            "-filter_complex",
            "[0:v]decode=format=yuv420p[d];[d]encode=codec=ffv1:g=1[out0]",
            "-map",
            "[out0]",
            "-f",
            "nut",
            out.to_str().expect("UTF-8"),
        ],
        Some(&dir),
    );
    assert!(output.status.success(), "{}", stderr_of(&output));
    run_ok(
        Command::new("ffmpeg")
            .args(["-v", "error", "-y", "-f", "nut", "-i"])
            .arg(&out)
            .args(["-c", "copy"])
            .arg(&mkv)
            .stdin(Stdio::null()),
        "ffmpeg -c copy",
    );
    let probed = run_ok(
        Command::new("ffprobe")
            .args([
                "-v",
                "error",
                "-count_frames",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name,width,height,nb_read_frames",
                "-of",
                "csv=p=0",
            ])
            .arg(&mkv)
            .stdin(Stdio::null()),
        "ffprobe",
    );
    let _ = std::fs::remove_file(&out);
    let _ = std::fs::remove_file(&mkv);
    let frames = ffprobe_pts(&fixture()).len();
    assert_eq!(
        String::from_utf8_lossy(&probed.stdout).trim(),
        format!("ffv1,64,48,{frames}")
    );
}

/// Paces a programme onto `out` in real time, for `seconds`: flat 64x48
/// rgba pictures at 25 a second, and a silent 48 kHz sound in 1920-sample
/// packets, one ahead of each picture.
fn pace_programme(out: impl std::io::Write, seconds: u32) {
    let mut video = Stream::video("rgba", 64, 48, TimeBase { num: 1, den: 25 }).expect("rgba");
    video.frame_rate = Some((25, 1));
    let mut audio = Stream::audio("f32", 48_000, 1).expect("f32");
    audio.time_base = TimeBase {
        num: 1,
        den: 48_000,
    };
    let mut muxer = Muxer::with_streams(std::io::BufWriter::new(out), &[video, audio])
        .expect("programme headers");
    let picture = [0u8, 255, 0, 255].repeat(64 * 48);
    let silence = vec![0u8; 1920 * 4];
    let started = Instant::now();
    for k in 0..i64::from(seconds * 25) {
        let due = started + Duration::from_millis(k as u64 * 40);
        if let Some(wait) = due.checked_duration_since(Instant::now()) {
            std::thread::sleep(wait);
        }
        let written = muxer
            .write_frame_to(1, k * 1920, &silence)
            .and_then(|_| muxer.write_frame_to(0, k, &picture))
            .and_then(|_| muxer.flush());
        if written.is_err() {
            return;
        }
    }
    let _ = muxer.finish();
}

#[test]
fn a_port_feed_of_coded_h264_reaches_its_hold_input_as_rgba_pictures() {
    let test = "a_port_feed_of_coded_h264_reaches_its_hold_input_as_rgba_pictures";
    let Some(dir) = libav_and_ffmpeg(test) else {
        return;
    };
    let switch = module("shape_switch");
    let port = TcpListener::bind((Ipv4Addr::LOCALHOST, 0))
        .expect("a free port")
        .local_addr()
        .expect("its address")
        .port();
    let out = scratch("fed.nut");
    // `layer` shows the feed's own pictures while it is live and black
    // otherwise.
    let chain = format!("[v=0:v][a=0:a]shape_switch=port={port}:lead=0.3:layer=true[v=o]");
    let mut child = Command::new(env!("CARGO_BIN_EXE_ffrwd-wasm"))
        .env(av::RUNTIME_DIR_VAR, &dir)
        .args(["-jobs", "1", "-f", "nut", "-i", "pipe:0", "-m", &switch])
        .args(["-filter_complex", &chain, "-map", "[o]", "-f", "nut"])
        .arg(&out)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn ffrwd-wasm");
    let stdin = child.stdin.take().expect("stdin");
    let mut stderr = child.stderr.take().expect("stderr");
    let stderr = std::thread::spawn(move || {
        let mut text = String::new();
        std::io::Read::read_to_string(&mut stderr, &mut text).ok();
        text
    });
    let programme = std::thread::spawn(move || pace_programme(stdin, 5));

    // The feeder writes the fixture's coded stream as it is, once the port
    // listens. The connects that find it listening send nothing, and the
    // host drops each for the next.
    let until = Instant::now() + Duration::from_secs(60);
    while TcpStream::connect((Ipv4Addr::LOCALHOST, port)).is_err() {
        if let Ok(Some(status)) = child.try_wait() {
            let stderr = stderr.join().unwrap_or_default();
            panic!("ffrwd-wasm exited {status:?} before listening on {port}:\n{stderr}");
        }
        assert!(Instant::now() < until, "nothing listened on {port}");
        std::thread::sleep(Duration::from_millis(20));
    }
    std::thread::sleep(Duration::from_millis(500));
    {
        let file = std::fs::File::open(fixture()).expect("open the h264 fixture");
        let mut demuxer = Demuxer::open(file).expect("read the fixture's NUT headers");
        let socket = TcpStream::connect((Ipv4Addr::LOCALHOST, port)).expect("connect the feeder");
        let stream = demuxer.stream().clone();
        let mut muxer = Muxer::with_streams(std::io::BufWriter::new(socket), &[stream])
            .expect("the feeder's headers");
        let mut buf = Vec::new();
        while let Some(packet) = demuxer.read_packet(&mut buf).expect("read a NUT packet") {
            muxer
                .write_coded_to(0, &packet, &buf)
                .expect("write a coded packet");
        }
        muxer.finish().expect("finish the feed");
    }

    programme.join().expect("the programme");
    let status = child.wait().expect("wait for ffrwd-wasm");
    let stderr = stderr.join().unwrap_or_default();
    assert!(status.success(), "{stderr}");
    assert!(
        stderr.contains("the feeder's picture is 64x48 h264, conformed to 64x48 rgba"),
        "the report names the codec:\n{stderr}"
    );
    assert!(stderr.contains("[av] decode h264: software"), "{stderr}");

    let (header, frames) = demux(&out);
    let _ = std::fs::remove_file(&out);
    assert_eq!(header.pix_fmt(), Some("rgba"));
    let size = 64 * 48 * 4;
    let decoded = ffmpeg_rgba(&fixture());
    let pictures: Vec<&[u8]> = decoded.chunks(size).collect();
    let mut live = 0;
    for (pts, data) in &frames {
        assert_eq!(data.len(), size, "every picture is the port's size");
        if data.iter().all(|b| *b == 0) {
            continue;
        }
        live += 1;
        assert!(
            pictures.contains(&data.as_slice()),
            "the picture at {pts} is one of the fixture's, decoded to rgba"
        );
    }
    assert!(
        live >= pictures.len() / 2,
        "the feed was shown ({live} of {} pictures)\n{stderr}",
        pictures.len()
    );
}

#[test]
fn decode_and_encode_refuse_what_they_cannot_take_naming_it() {
    // These are refused before libav is asked for, so they need none.
    let raw = scratch("refusals_raw.nut");
    {
        let base = TimeBase { num: 1, den: 25 };
        let stream = Stream::video("rgba", 4, 4, base).expect("rgba");
        let file = std::fs::File::create(&raw).expect("create the raw input");
        let mut muxer = Muxer::with_streams(file, &[stream]).expect("headers");
        for k in 0..3 {
            muxer.write_frame_to(0, k, &[k as u8; 64]).expect("a frame");
        }
        muxer.finish().expect("finish the raw input");
    }
    let raw_path = raw.to_str().expect("UTF-8").to_string();
    let coded_path = fixture().to_str().expect("UTF-8").to_string();
    let cases: &[(&str, &str, &[&str])] = &[
        (
            &raw_path,
            "[0:v]decode=format=rgba[o]",
            &["decode reads a coded picture stream", "raw rgba video"],
        ),
        (
            &raw_path,
            "[0:v]encode=b=6M[o]",
            &["encode names its encoder with codec=<name>", "names none"],
        ),
        (
            &coded_path,
            "[0:v]encode=codec=ffv1[o]",
            &["encode reads raw pictures", "coded h264 video"],
        ),
        (
            &coded_path,
            "[0:v]decode=format=bgra[o]",
            &[
                "decode=format=bgra",
                "rgba, yuv420p, yuv422p, yuv444p, gray",
            ],
        ),
        (
            &coded_path,
            "[0:v]decode[o]",
            &["decode names the pixel format it writes with format=<pix_fmt>"],
        ),
        (
            &coded_path,
            "[0:v]decode=format=rgba:scale=2[o]",
            &["decode has no option 'scale'"],
        ),
    ];
    let out = scratch("refusals_out.nut");
    for (input, chain, needles) in cases {
        let output = sidecar(
            &[
                "-f",
                "nut",
                "-i",
                input,
                "-filter_complex",
                chain,
                "-map",
                "[o]",
                "-f",
                "nut",
                out.to_str().expect("UTF-8"),
            ],
            None,
        );
        let stderr = stderr_of(&output);
        assert!(!output.status.success(), "{chain} was refused:\n{stderr}");
        for needle in *needles {
            assert!(stderr.contains(needle), "{chain}: {needle:?} in\n{stderr}");
        }
    }
    let _ = std::fs::remove_file(&raw);
    let _ = std::fs::remove_file(&out);

    let output = sidecar(
        &[
            "-f",
            "nut",
            "-i",
            &coded_path,
            "-m",
            "decode=x.wasm",
            "-filter_complex",
            "[0:v]decode[o]",
            "-map",
            "[o]",
            "-f",
            "nut",
            out.to_str().expect("UTF-8"),
        ],
        None,
    );
    let stderr = stderr_of(&output);
    assert!(!output.status.success());
    assert!(
        stderr.contains("'decode' is the network's own node"),
        "{stderr}"
    );
}
