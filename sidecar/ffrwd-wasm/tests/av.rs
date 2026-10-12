//! libav through the runtime's shim: found, refused, and used to decode
//! the wire's h264 fixture and to encode it again, each checked against
//! what ffmpeg itself makes of the same bytes.
//!
//! libav is not in git, so the cases that need it skip - loudly - when
//! neither `FFRWD_AV_RUNTIME` nor the directory the `ffrwd-wasm` binary was
//! built into holds the four libraries. The cases that compare against
//! ffmpeg skip the same way without ffmpeg and ffprobe on PATH.

use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::sync::OnceLock;

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
