//! Real speech through the transcribe module, driven directly through
//! `ffrwd_wasm_runtime::runtime` with no ffmpeg involved.
//!
//! The whisper weights are not in git, so this skips - loudly - when they are
//! absent. `sidecar/modules/transcribe/fetch-model.ps1` puts them where the
//! module's build script looks.

use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;

use ffrwd_wasm_runtime::runtime::{
    AudioFormat, Filter, Format, Frame, Media, StreamInfo, TimeBase,
};

/// What the module publishes, and what the host conforms a stream to.
const SAMPLE_RATE: u32 = 16_000;

/// The module's window: thirty seconds.
const WINDOW: usize = 30 * SAMPLE_RATE as usize;

/// One tick is one sample, the natural base for audio.
const FORMAT: Format = Format {
    media: Media::Audio(AudioFormat {
        sample_rate: SAMPLE_RATE,
        channels: 1,
        sample_fmt: "f32",
    }),
    time_base: TimeBase {
        num: 1,
        den: SAMPLE_RATE as u64,
    },
};

/// The files the module compiles in, which a checkout may not have.
const MODEL_FILES: [&str; 4] = [
    "model.safetensors",
    "tokenizer.json",
    "config.json",
    "melfilters.bytes",
];

/// The sidecar's directory, the parent of `runtime/`.
fn sidecar_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("runtime/ has a parent directory")
        .to_path_buf()
}

/// Whether the whisper files are where the module's build script looks.
fn model_present() -> bool {
    let dir = sidecar_root().join("modules/transcribe/model");
    MODEL_FILES.iter().all(|name| dir.join(name).is_file())
}

/// Says why the test did nothing, since a skip is otherwise silent.
fn announce_skip() {
    eprintln!(
        "SKIPPED: transcribe's whisper model is absent. Run \
         sidecar/modules/transcribe/fetch-model.ps1 to download it."
    );
}

/// Builds the transcribe module for wasm32-wasip2 and returns its component.
/// `modules/` is a separate cargo workspace with its own build lock, so this
/// does not deadlock against the `cargo test` run driving this binary.
fn build_module() -> PathBuf {
    let output = Command::new("cargo")
        .args([
            "build",
            "--release",
            "--target",
            "wasm32-wasip2",
            "-p",
            "transcribe",
        ])
        .current_dir(sidecar_root().join("modules"))
        .output()
        .expect("spawn cargo build for transcribe");
    assert!(
        output.status.success(),
        "building transcribe failed (status {:?}):\n{}",
        output.status.code(),
        String::from_utf8_lossy(&output.stderr)
    );
    sidecar_root().join("modules/target/wasm32-wasip2/release/transcribe.wasm")
}

/// The stream every instance here is attached to.
fn stream() -> StreamInfo {
    StreamInfo {
        kind: "audio".to_string(),
        codec: "pcm_f32le".to_string(),
        ..StreamInfo::default()
    }
}

/// The samples of a 16-bit mono wav, as the floats the module is handed.
/// Walks the RIFF chunks rather than assuming a fixed header, since the writer
/// may have put its own chunks before the data.
fn wav_samples(path: &PathBuf) -> Vec<f32> {
    let bytes = std::fs::read(path).expect("reading the speech fixture");
    assert_eq!(&bytes[0..4], b"RIFF", "the fixture is a RIFF file");
    assert_eq!(&bytes[8..12], b"WAVE", "the fixture is a wav");

    let mut at = 12;
    while at + 8 <= bytes.len() {
        let id = &bytes[at..at + 4];
        let len =
            u32::from_le_bytes(bytes[at + 4..at + 8].try_into().expect("four bytes")) as usize;
        let body = at + 8;
        if id == b"data" {
            let (whole, _) = bytes[body..body + len].as_chunks::<2>();
            return whole
                .iter()
                .map(|b| f32::from(i16::from_le_bytes(*b)) / 32768.0)
                .collect();
        }
        at = body + len + (len % 2);
    }
    panic!("the fixture has no data chunk");
}

/// One fixture's samples, padded with silence to the module's window.
fn fixture(name: &str) -> Vec<f32> {
    let path = sidecar_root()
        .join("modules/transcribe/tests/data")
        .join(name);
    let mut pcm = wav_samples(&path);
    assert!(
        pcm.len() <= WINDOW,
        "{name} is longer than one window: {} samples",
        pcm.len()
    );
    pcm.resize(WINDOW, 0.0);
    pcm
}

/// One decoded row: the span it covers in the stream, and what was said.
struct Said {
    start_t: f64,
    end_t: f64,
    text: String,
}

/// One window of samples through a fresh instance, as rows. Checks the shape
/// every window's rows have to have: named spans, inside the window, in order,
/// and saying something.
fn decode(module: &str, pcm: &[f32], params: &str) -> Vec<Said> {
    let opening = Instant::now();
    let mut filter = Filter::open(module, &FORMAT, &stream(), params).expect("opening transcribe");
    eprintln!(
        "transcribe {params}: the model loaded in {:.1}s",
        opening.elapsed().as_secs_f64()
    );

    let bytes: Vec<u8> = pcm.iter().flat_map(|s| s.to_le_bytes()).collect();
    let window = [Frame {
        pts: 0,
        data: bytes.clone().into(),
        rows: Vec::new(),
    }];

    let started = Instant::now();
    let out = filter
        .process_window(&window, &[], true)
        .unwrap_or_else(|e| panic!("transcribing with {params}: {e}"));
    let seconds = WINDOW as f64 / f64::from(SAMPLE_RATE);
    eprintln!(
        "transcribe {params}: {seconds:.1}s of audio in {:.1}s",
        started.elapsed().as_secs_f64()
    );

    assert_eq!(out.frames.len(), 1, "one window in, one window out");
    assert_eq!(
        *out.frames[0].data, bytes,
        "a one-to-one module returns the samples it was handed"
    );
    assert!(
        !out.frames[0].rows.is_empty(),
        "a window of clear speech decodes to at least one segment"
    );

    let mut said = Vec::new();
    let mut previous_end = f64::NEG_INFINITY;
    for row in &out.frames[0].rows {
        let value: serde_json::Value =
            serde_json::from_str(row).unwrap_or_else(|e| panic!("row {row} is not JSON: {e}"));
        let object = value.as_object().expect("a row is a JSON object");
        let mut keys: Vec<&str> = object.keys().map(String::as_str).collect();
        keys.sort_unstable();
        assert_eq!(
            keys,
            ["end_t", "start_t", "text"],
            "a segment row names its span and its words"
        );

        let start_t = value["start_t"].as_f64().expect("start_t is a number");
        let end_t = value["end_t"].as_f64().expect("end_t is a number");
        let text = value["text"]
            .as_str()
            .expect("text is a string")
            .to_string();

        assert!(
            start_t < end_t,
            "a segment ends after it starts, got {start_t} to {end_t}"
        );
        assert!(
            start_t >= -1e-9,
            "a segment starts no earlier than its own window, got {start_t}"
        );
        assert!(
            end_t <= seconds + 1e-9,
            "a segment ends inside the window it came from, got {end_t}"
        );
        assert!(!text.trim().is_empty(), "a segment says something");
        assert!(
            start_t >= previous_end - 1e-9,
            "segments do not overlap: {start_t} starts before {previous_end}"
        );
        previous_end = end_t;
        eprintln!("  {start_t:.2}s-{end_t:.2}s: {text}");

        said.push(Said {
            start_t,
            end_t,
            text,
        });
    }
    said
}

/// The words of every row, lowercased, with the punctuation dropped.
fn words(said: &[Said]) -> Vec<String> {
    said.iter()
        .flat_map(|s| s.text.split(|c: char| !c.is_alphanumeric()))
        .filter(|w| !w.is_empty())
        .map(str::to_lowercase)
        .collect()
}

/// How many of `common` the rows use, which is what tells one language from
/// another over a stretch of ordinary speech.
fn stopwords(said: &[Said], common: &[&str]) -> usize {
    let used = words(said);
    common
        .iter()
        .filter(|w| used.iter().any(|u| u == *w))
        .count()
}

#[test]
fn real_speech_comes_back_as_timed_rows_and_one_transcript() {
    if !model_present() {
        announce_skip();
        return;
    }

    let module = build_module();
    let module_str = module.to_str().expect("module path is valid UTF-8");

    let fixture = sidecar_root().join("modules/transcribe/tests/data/speech.wav");
    let pcm = wav_samples(&fixture);
    assert_eq!(pcm.len(), WINDOW, "the fixture is exactly one window");

    let opening = Instant::now();
    let mut filter = Filter::open(module_str, &FORMAT, &stream(), r#"{"language":"en"}"#)
        .expect("opening transcribe");
    eprintln!(
        "transcribe: the model loaded in {:.1}s",
        opening.elapsed().as_secs_f64()
    );

    let bytes: Vec<u8> = pcm.iter().flat_map(|s| s.to_le_bytes()).collect();

    // The same speech twice over, so the second window's rows must come back a
    // whole window later than the first's: the times are the stream's, not the
    // window's. At this time base one tick is one sample.
    let mut texts: Vec<Vec<String>> = Vec::new();
    let mut spans: Vec<Vec<(f64, f64)>> = Vec::new();
    let mut trailing = Vec::new();

    for (index, last) in [(0usize, false), (1usize, true)] {
        let pts = (index * WINDOW) as i64;
        let window = [Frame {
            pts,
            data: bytes.clone().into(),
            rows: Vec::new(),
        }];

        let started = Instant::now();
        let out = filter
            .process_window(&window, &[], last)
            .unwrap_or_else(|e| panic!("transcribing the window at pts {pts}: {e}"));
        let elapsed = started.elapsed();
        let seconds = WINDOW as f64 / f64::from(SAMPLE_RATE);
        eprintln!(
            "transcribe: window {index}: {seconds:.1}s of audio in {:.1}s, {:.2}x realtime",
            elapsed.as_secs_f64(),
            seconds / elapsed.as_secs_f64()
        );

        assert_eq!(out.frames.len(), 1, "one window in, one window out");
        let produced = &out.frames[0];
        assert_eq!(produced.pts, pts, "the audio keeps its timestamp");
        assert_eq!(
            *produced.data, bytes,
            "a one-to-one module returns the samples it was handed"
        );
        assert!(
            !produced.rows.is_empty(),
            "a window of clear speech decodes to at least one segment"
        );

        let base = index as f64 * WINDOW as f64 / f64::from(SAMPLE_RATE);
        let mut previous_end = f64::NEG_INFINITY;
        let mut window_texts = Vec::new();
        let mut window_spans = Vec::new();
        for row in &produced.rows {
            let value: serde_json::Value =
                serde_json::from_str(row).unwrap_or_else(|e| panic!("row {row} is not JSON: {e}"));
            let start_t = value["start_t"].as_f64().expect("start_t is a number");
            let end_t = value["end_t"].as_f64().expect("end_t is a number");
            let text = value["text"].as_str().expect("text is a string");

            assert!(
                start_t < end_t,
                "a segment ends after it starts, got {start_t} to {end_t}"
            );
            assert!(
                start_t >= base - 1e-9,
                "a segment starts no earlier than its own window, got {start_t} under {base}"
            );
            assert!(
                end_t <= base + WINDOW as f64 / f64::from(SAMPLE_RATE) + 1e-9,
                "a segment ends inside the window it came from, got {end_t}"
            );
            assert!(!text.trim().is_empty(), "a segment says something");
            assert!(
                start_t >= previous_end - 1e-9,
                "segments do not overlap: {start_t} starts before {previous_end}"
            );
            previous_end = end_t;
            eprintln!("  {start_t:.2}s-{end_t:.2}s: {text}");

            window_texts.push(text.to_string());
            window_spans.push((start_t, end_t));
        }

        if last {
            assert_eq!(
                out.trailing.len(),
                1,
                "the final call carries one transcript row"
            );
            trailing = out.trailing.clone();
        } else {
            assert!(
                out.trailing.is_empty(),
                "only the final call carries trailing rows, got {:?}",
                out.trailing
            );
        }
        texts.push(window_texts);
        spans.push(window_spans);
    }

    // The same audio decodes the same way twice - the decoding carries nothing
    // between windows - and the second window's times are a window later.
    assert_eq!(
        texts[0], texts[1],
        "the same window decodes to the same words wherever it sits in the stream"
    );
    let window_seconds = WINDOW as f64 / f64::from(SAMPLE_RATE);
    for ((a0, a1), (b0, b1)) in spans[0].iter().zip(&spans[1]) {
        assert!(
            (b0 - a0 - window_seconds).abs() < 1e-6 && (b1 - a1 - window_seconds).abs() < 1e-6,
            "the second window's segment {b0}-{b1} is not {window_seconds}s past {a0}-{a1}"
        );
    }

    let value: serde_json::Value =
        serde_json::from_str(&trailing[0]).expect("the transcript row is JSON");
    let transcript = value["transcript"]
        .as_str()
        .expect("transcript is a string");
    assert!(
        !transcript.trim().is_empty(),
        "the transcript says something"
    );
    let expected = texts.concat().join(" ");
    assert_eq!(
        transcript, expected,
        "the transcript is every segment of every window, in order"
    );
    eprintln!("transcript: {transcript}");
}

#[test]
fn spanish_speech_comes_back_in_spanish_or_in_english_as_it_is_asked_for() {
    if !model_present() {
        announce_skip();
        return;
    }

    let module = build_module();
    let module_str = module.to_str().expect("module path is valid UTF-8");
    let pcm = fixture("speech-es.wav");

    eprintln!("spanish, into english:");
    let english = decode(module_str, &pcm, r#"{"language":"es","language_to":"en"}"#);
    eprintln!("spanish, as spoken:");
    let spanish = decode(module_str, &pcm, r#"{"language":"es"}"#);

    // Both decodings cover the speech rather than a corner of it.
    for (asked, said) in [("english", &english), ("spanish", &spanish)] {
        let covered: f64 = said.iter().map(|s| s.end_t - s.start_t).sum();
        assert!(
            covered > 10.0,
            "the {asked} rows cover only {covered:.1}s of a thirty-second window"
        );
    }

    // Common words carry the language over a stretch of ordinary speech in a
    // way any one sentence does not.
    assert!(
        stopwords(&english, &["the", "of", "is", "and", "we"]) >= 2,
        "the rows asked for in English are not English: {:?}",
        english.iter().map(|s| &s.text).collect::<Vec<_>>()
    );
    assert!(
        stopwords(&spanish, &["de", "la", "el", "en", "que", "los", "un"]) >= 2,
        "the rows of Spanish speech are not Spanish: {:?}",
        spanish.iter().map(|s| &s.text).collect::<Vec<_>>()
    );
    assert_eq!(stopwords(&english, &["the"]), 1, "English speech says the");
    assert_eq!(
        stopwords(&spanish, &["the"]),
        0,
        "Spanish speech does not say the"
    );

    let english_text: Vec<&str> = english.iter().map(|s| s.text.as_str()).collect();
    let spanish_text: Vec<&str> = spanish.iter().map(|s| s.text.as_str()).collect();
    assert_ne!(
        english_text, spanish_text,
        "translating and transcribing the same speech gave the same words"
    );
}

#[test]
fn a_language_whisper_does_not_know_is_refused_by_name() {
    let module = build_module();
    let module_str = module.to_str().expect("module path is valid UTF-8");

    let Err(err) = Filter::open(module_str, &FORMAT, &stream(), r#"{"language":"klingon"}"#) else {
        panic!("a language whisper does not know should have been refused");
    };
    let message = format!("{err:#}");
    assert!(message.contains("transcribe"), "got: {message}");
    assert!(message.contains("klingon"), "got: {message}");
}

#[test]
fn a_window_of_silence_says_nothing() {
    if !model_present() {
        announce_skip();
        return;
    }

    let module = build_module();
    let module_str = module.to_str().expect("module path is valid UTF-8");

    let mut filter = Filter::open(module_str, &FORMAT, &stream(), r#"{"language":"en"}"#)
        .expect("opening transcribe");

    let bytes = vec![0u8; WINDOW * 4];
    let window = [Frame {
        pts: 0,
        data: bytes.clone().into(),
        rows: Vec::new(),
    }];
    let out = filter
        .process_window(&window, &[], true)
        .expect("transcribing silence");

    assert_eq!(out.frames.len(), 1, "the silence still passes through");
    assert_eq!(*out.frames[0].data, bytes, "and passes through unchanged");
    assert!(
        out.frames[0].rows.is_empty(),
        "silence decodes to no rows, got {:?}",
        out.frames[0].rows
    );
    assert_eq!(out.trailing.len(), 1, "the transcript row still arrives");
    let value: serde_json::Value =
        serde_json::from_str(&out.trailing[0]).expect("the transcript row is JSON");
    assert_eq!(
        value["transcript"].as_str().expect("a string"),
        "",
        "nothing was said, so the transcript is empty"
    );
}
