//! Integration tests for a rows module through `ffrwd-wasm`'s own argv:
//! `-rows-in` in, `-f ndjson` out, no `-i` at all. `fauxlate` is the fixture
//! - see `modules/fauxlate/src/lib.rs`.

use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Output, Stdio};
use std::sync::OnceLock;

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
    // fauxlate also exports `values`, alongside the rows module, and the
    // description carries both.
    let functions = description["functions"]
        .as_array()
        .expect("a functions array");
    assert_eq!(functions.len(), 1);
    assert_eq!(functions[0]["name"], "translate");
}
