//! Integration tests for a rows module through `ffrwd-wasm`'s own argv:
//! `-rows-in` in, `-f ndjson` out, no `-i` at all - and, further down, a
//! rows module chained onto a stream module in one process with
//! `-m <path> -rows-from <index>`. `fauxlate` is the fixture for both - see
//! `modules/fauxlate/src/lib.rs` - and `captions` (`modules/captions/src/lib.rs`)
//! is the stream module the chained tests read rows from.

use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Output, Stdio};
use std::sync::OnceLock;

use ffrwd_wasm::nut::{Muxer, Stream, TimeBase};

/// Sidecar root, the parent of `ffrwd-wasm/`.
fn sidecar_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("ffrwd-wasm/ has a parent directory")
        .to_path_buf()
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
                "fauxlate",
                "-p",
                "invert",
                "-p",
                "captions",
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
    stdout: Vec<u8>,
    stderr: String,
    output: Output,
}

/// Runs `ffrwd-wasm` with the given argv and no stdin - a rows module reads
/// its rows off `-rows-in`, never off stdin.
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

/// Runs `ffrwd-wasm` with the given argv, feeding `stdin_bytes` - for a
/// stream module chained to a rows module, which reads its stream off
/// stdin the ordinary way.
fn run_ffrwd_wasm_stdin(args: &[&str], stdin_bytes: &[u8]) -> Run {
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
        stdout: output.stdout.clone(),
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        output,
    }
}

/// The time base ffmpeg gives a NUT stream by default, and the step one
/// frame takes at 25fps in it - matching `ffrwd_wasm.rs`'s own fixture.
const TIME_BASE: TimeBase = TimeBase { num: 1, den: 65536 };
const PTS_STEP: i64 = 65536 / 25;
const WIDTH: u32 = 8;
const HEIGHT: u32 = 8;
const FRAME_LEN: usize = (WIDTH * HEIGHT * 4) as usize;

fn a_stream() -> Stream {
    Stream::video("rgba", WIDTH, HEIGHT, TIME_BASE).expect("rgba is carried")
}

/// Frame `index` filled with a byte pattern distinct per position and per
/// frame - captions never reads the pixels, only the frame count, so the
/// pattern itself does not matter here.
fn synthetic_frame(index: u8) -> Vec<u8> {
    (0..FRAME_LEN)
        .map(|offset| index.wrapping_mul(41).wrapping_add(offset as u8))
        .collect()
}

/// A NUT stream carrying `frames`, one every `PTS_STEP` from zero.
fn nut_stream(frames: &[Vec<u8>]) -> Vec<u8> {
    let mut wire = Vec::new();
    {
        let mut muxer = Muxer::new(&mut wire, &a_stream()).expect("write NUT headers");
        for (i, frame) in frames.iter().enumerate() {
            muxer
                .write_frame(i as i64 * PTS_STEP, frame)
                .expect("write NUT frame");
        }
        muxer.finish().expect("finish the NUT stream");
    }
    wire
}

/// A fresh temp directory this test owns, cleaned up on drop by the OS's own
/// temp-directory sweep - these tests write small files, never read them
/// back from disk except this run's own output, and the crate carries no
/// tempfile dependency to add one just for this.
fn tempdir() -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "ffrwd-wasm-rows-module-test-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("the clock is past 1970")
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).expect("create a temp directory");
    dir
}

/// Writes `lines` as NDJSON (one object per line) to a fresh temp file and
/// returns its path.
fn ndjson_file(dir: &std::path::Path, name: &str, lines: &[&str]) -> PathBuf {
    let path = dir.join(name);
    let mut file = std::fs::File::create(&path).expect("create the NDJSON fixture");
    for line in lines {
        writeln!(file, "{line}").expect("write an NDJSON line");
    }
    path
}

#[test]
fn a_run_reads_rows_in_and_writes_the_word_ruled_cues() {
    let module = module_path("fauxlate");
    let dir = tempdir();
    let rows_in = ndjson_file(
        &dir,
        "cues.ndjson",
        &[
            r#"{"start_t":0.0,"end_t":1.5,"text":"Cue one."}"#,
            r#"{"start_t":1.5,"end_t":3.0,"text":"Wait!"}"#,
        ],
    );
    let rows_out = dir.join("out.ndjson");

    let run = run_ffrwd_wasm(&[
        "-m",
        module.to_str().expect("module path is UTF-8"),
        "-params",
        "{}",
        "-rows-in",
        rows_in.to_str().expect("temp path is UTF-8"),
        "-f",
        "ndjson",
        rows_out.to_str().expect("temp path is UTF-8"),
    ]);
    assert!(
        run.output.status.success(),
        "run exited with {:?}\nstderr:\n{}",
        run.output.status.code(),
        run.stderr
    );

    let written = std::fs::read_to_string(&rows_out).expect("read the rows this run wrote");
    let lines: Vec<&str> = written.lines().collect();
    assert_eq!(
        lines,
        vec![
            r#"{"start_t":0.0,"end_t":1.5,"text":"Cue-a one-o."}"#,
            r#"{"start_t":1.5,"end_t":3.0,"text":"Wait-a!"}"#,
        ]
    );
}

#[test]
fn a_rows_module_given_an_input_is_refused_by_name() {
    let module = module_path("fauxlate");
    let dir = tempdir();
    let rows_in = ndjson_file(
        &dir,
        "cues.ndjson",
        &[r#"{"start_t":0.0,"end_t":1.0,"text":"a"}"#],
    );
    let run = run_ffrwd_wasm(&[
        "-f",
        "nut",
        "-i",
        "-",
        "-m",
        module.to_str().expect("module path is UTF-8"),
        "-rows-in",
        rows_in.to_str().expect("temp path is UTF-8"),
        "-f",
        "ndjson",
        "-",
    ]);
    assert!(!run.output.status.success());
    assert!(
        run.stderr.contains("rows module") && run.stderr.contains("-i"),
        "stderr does not refuse the input:\n{}",
        run.stderr
    );
}

#[test]
fn a_stream_module_given_rows_in_is_refused_by_name() {
    let module = module_path("invert");
    let dir = tempdir();
    let rows_in = ndjson_file(
        &dir,
        "cues.ndjson",
        &[r#"{"start_t":0.0,"end_t":1.0,"text":"a"}"#],
    );
    let run = run_ffrwd_wasm(&[
        "-m",
        module.to_str().expect("module path is UTF-8"),
        "-rows-in",
        rows_in.to_str().expect("temp path is UTF-8"),
        "-f",
        "nut",
        "-",
    ]);
    assert!(!run.output.status.success());
    assert!(
        run.stderr.contains("-rows-in") && run.stderr.contains("invert"),
        "stderr does not name the module refusing -rows-in:\n{}",
        run.stderr
    );
}

#[test]
fn describe_reports_the_rows_kind_and_both_schemas() {
    let module = module_path("fauxlate");
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
    assert_eq!(description["world"], "ffrwd:av@0.14.0");
    assert_eq!(description["name"], "fauxlate");
    assert_eq!(description["rows_module"], true);
    let output_schema = &description["rows_schema"];
    let input_schema = &description["input_rows_schema"];
    assert_eq!(
        output_schema, input_schema,
        "fauxlate reads and writes the same cue shape"
    );
    assert_eq!(
        output_schema["required"],
        serde_json::json!(["start_t", "end_t", "text"])
    );
    // fauxlate also exports `values` -- both `translate` and `embed_text`
    // -- alongside the rows module, and the description carries all of it.
    let functions = description["functions"]
        .as_array()
        .expect("a functions array");
    assert_eq!(functions.len(), 2);
    let names: Vec<&str> = functions
        .iter()
        .map(|f| f["name"].as_str().expect("a function name"))
        .collect();
    assert_eq!(names, vec!["translate", "embed_text"]);
}

// A rows module chained onto a stream module in one process:
// `-m <stream> -m <rows> -rows-from <index>`, the module-network form
// `docs/examples.md`'s "Translate captions as they are produced" pins.

#[test]
fn a_stream_module_chained_into_a_rows_module_translates_every_cue() {
    let captions = module_path("captions");
    let fauxlate = module_path("fauxlate");
    let dir = tempdir();
    let out = dir.join("translated.webvtt");

    let frames: Vec<Vec<u8>> = (0..3u8).map(synthetic_frame).collect();
    let wire = nut_stream(&frames);

    let run = run_ffrwd_wasm_stdin(
        &[
            "-f",
            "nut",
            "-i",
            "-",
            "-m",
            captions.to_str().expect("module path is UTF-8"),
            "-m",
            fauxlate.to_str().expect("module path is UTF-8"),
            "-rows-from",
            "0",
            "-f",
            "webvtt",
            out.to_str().expect("temp path is UTF-8"),
        ],
        &wire,
    );
    assert!(
        run.output.status.success(),
        "run exited with {:?}\nstderr:\n{}",
        run.output.status.code(),
        run.stderr
    );

    let document = std::fs::read_to_string(&out).expect("read the webvtt document");
    let cue_count = document.matches(" --> ").count();
    assert_eq!(
        cue_count, 3,
        "captions writes one cue per frame, and none should be dropped or added, got:\n{document}"
    );
    for index in 0..3 {
        // captions writes "cue {index}"; fauxlate's word rule turns that
        // into "cue-a {index}-o".
        assert!(
            document.contains(&format!("cue-a {index}-o")),
            "cue {index} was not translated, got:\n{document}"
        );
    }
}

#[test]
fn a_rows_module_on_the_line_without_rows_from_is_refused_by_name() {
    let captions = module_path("captions");
    let fauxlate = module_path("fauxlate");
    let run = run_ffrwd_wasm(&[
        "-f",
        "nut",
        "-i",
        "-",
        "-m",
        captions.to_str().expect("module path is UTF-8"),
        "-m",
        fauxlate.to_str().expect("module path is UTF-8"),
        "-f",
        "webvtt",
        "-",
    ]);
    assert!(!run.output.status.success());
    assert!(
        run.stderr.contains("-rows-from") && run.stderr.contains("fauxlate"),
        "stderr does not name the module needing -rows-from:\n{}",
        run.stderr
    );
}

#[test]
fn a_rows_from_naming_no_m_on_the_line_is_refused_by_name() {
    let captions = module_path("captions");
    let fauxlate = module_path("fauxlate");
    let run = run_ffrwd_wasm(&[
        "-f",
        "nut",
        "-i",
        "-",
        "-m",
        captions.to_str().expect("module path is UTF-8"),
        "-m",
        fauxlate.to_str().expect("module path is UTF-8"),
        "-rows-from",
        "5",
        "-f",
        "webvtt",
        "-",
    ]);
    assert!(!run.output.status.success());
    assert!(
        run.stderr.contains("-rows-from 5") && run.stderr.contains("fauxlate"),
        "stderr does not name the module and the index, got:\n{}",
        run.stderr
    );
}

#[test]
fn a_rows_from_naming_its_own_slot_is_refused_by_name() {
    let captions = module_path("captions");
    let fauxlate = module_path("fauxlate");
    let run = run_ffrwd_wasm(&[
        "-f",
        "nut",
        "-i",
        "-",
        "-m",
        captions.to_str().expect("module path is UTF-8"),
        "-m",
        fauxlate.to_str().expect("module path is UTF-8"),
        "-rows-from",
        "1",
        "-f",
        "webvtt",
        "-",
    ]);
    assert!(!run.output.status.success());
    assert!(
        run.stderr.contains("-rows-from 1") && run.stderr.contains("fauxlate"),
        "stderr does not name the module and the index, got:\n{}",
        run.stderr
    );
}

// Two rows documents off one line: `-rows <index>` per output says whose
// rows each holds, the way `-rows-from` says whose rows a module reads.

#[test]
fn two_rows_outputs_hold_the_modules_their_rows_flags_name() {
    let captions = module_path("captions");
    let fauxlate = module_path("fauxlate");
    let dir = tempdir();
    let written = dir.join("written.webvtt");
    let translated = dir.join("translated.webvtt");

    let frames: Vec<Vec<u8>> = (0..3u8).map(synthetic_frame).collect();
    let wire = nut_stream(&frames);

    let run = run_ffrwd_wasm_stdin(
        &[
            "-f",
            "nut",
            "-i",
            "-",
            "-m",
            captions.to_str().expect("module path is UTF-8"),
            "-m",
            fauxlate.to_str().expect("module path is UTF-8"),
            "-rows-from",
            "0",
            "-rows",
            "0",
            "-f",
            "webvtt",
            written.to_str().expect("temp path is UTF-8"),
            "-rows",
            "1",
            "-f",
            "webvtt",
            translated.to_str().expect("temp path is UTF-8"),
        ],
        &wire,
    );
    assert!(
        run.output.status.success(),
        "run exited with {:?}\nstderr:\n{}",
        run.output.status.code(),
        run.stderr
    );

    let first = std::fs::read_to_string(&written).expect("read the producer's document");
    let second = std::fs::read_to_string(&translated).expect("read the chained document");
    assert_eq!(
        first.matches(" --> ").count(),
        second.matches(" --> ").count(),
        "the two documents hold the same cues, got:\n{first}\nand:\n{second}"
    );
    for index in 0..3 {
        assert!(
            first.contains(&format!("cue {index}")),
            "cue {index} is not in the producer's own document, got:\n{first}"
        );
        assert!(
            second.contains(&format!("cue-a {index}-o")),
            "cue {index} was not translated in the chained document, got:\n{second}"
        );
    }
    assert!(
        !first.contains("cue-a"),
        "the producer's document went through the rows module, got:\n{first}"
    );
}

#[test]
fn a_rows_naming_no_m_on_the_line_is_refused_by_name() {
    let captions = module_path("captions");
    let run = run_ffrwd_wasm(&[
        "-f",
        "nut",
        "-i",
        "-",
        "-m",
        captions.to_str().expect("module path is UTF-8"),
        "-rows",
        "3",
        "-f",
        "webvtt",
        "-",
    ]);
    assert!(!run.output.status.success());
    assert!(
        run.stderr.contains("-rows 3"),
        "stderr does not name the index, got:\n{}",
        run.stderr
    );
}

#[test]
fn a_rows_on_a_frame_output_is_refused_by_name() {
    let captions = module_path("captions");
    let run = run_ffrwd_wasm(&[
        "-f",
        "nut",
        "-i",
        "-",
        "-m",
        captions.to_str().expect("module path is UTF-8"),
        "-rows",
        "0",
        "-f",
        "nut",
        "-",
    ]);
    assert!(!run.output.status.success());
    assert!(
        run.stderr.contains("-rows") && run.stderr.contains("writes none"),
        "stderr does not say the output carries no rows, got:\n{}",
        run.stderr
    );
}

#[test]
fn a_rows_module_no_output_names_is_refused_by_name() {
    let captions = module_path("captions");
    let fauxlate = module_path("fauxlate");
    let dir = tempdir();
    let out = dir.join("plain.webvtt");
    let run = run_ffrwd_wasm(&[
        "-f",
        "nut",
        "-i",
        "-",
        "-m",
        captions.to_str().expect("module path is UTF-8"),
        "-m",
        fauxlate.to_str().expect("module path is UTF-8"),
        "-rows-from",
        "0",
        "-rows",
        "0",
        "-f",
        "webvtt",
        out.to_str().expect("temp path is UTF-8"),
        "-rows",
        "0",
        "-f",
        "srt",
        "-",
    ]);
    assert!(!run.output.status.success());
    assert!(
        run.stderr.contains("fauxlate") && run.stderr.contains("reach no output"),
        "stderr does not name the module nothing reads, got:\n{}",
        run.stderr
    );
}
