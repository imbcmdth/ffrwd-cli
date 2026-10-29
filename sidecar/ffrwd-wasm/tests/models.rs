//! Files bound with `-model`, read as bytes through `ffrwd:models/files`,
//! driven through the binary's own argv.
//!
//! Every case spawns `ffrwd-wasm`: the bindings are recorded once per
//! process.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::OnceLock;

use sha2::{Digest, Sha256};

/// The sidecar's directory, the parent of `ffrwd-wasm/`.
fn sidecar_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("ffrwd-wasm/ has a parent directory")
        .to_path_buf()
}

/// Builds models-probe and nn-probe for wasm32-wasip2, once per test
/// binary. `modules/` is a separate cargo workspace with its own build lock,
/// so this does not deadlock against the `cargo test` run driving this
/// binary.
fn module(name: &str) -> PathBuf {
    static BUILT: OnceLock<()> = OnceLock::new();
    BUILT.get_or_init(|| {
        let output = Command::new("cargo")
            .args([
                "build",
                "--release",
                "--target",
                "wasm32-wasip2",
                "-p",
                "models-probe",
                "-p",
                "nn-probe",
            ])
            .current_dir(sidecar_root().join("modules"))
            .output()
            .expect("spawn cargo build for models-probe");
        assert!(
            output.status.success(),
            "building models-probe failed (status {:?}):\n{}",
            output.status.code(),
            String::from_utf8_lossy(&output.stderr)
        );
    });
    sidecar_root()
        .join("modules/target/wasm32-wasip2/release")
        .join(format!("{name}.wasm"))
}

/// A file of `len` bytes, distinct per position, written under a temporary
/// directory of this process and this test's own.
fn bytes_file(test: &str, name: &str, len: usize) -> (PathBuf, Vec<u8>) {
    let dir = std::env::temp_dir().join(format!("ffrwd_models_{}_{test}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("make the temporary directory");
    let bytes: Vec<u8> = (0..len).map(|i| (i * 7 % 251) as u8).collect();
    let path = dir.join(name);
    std::fs::write(&path, &bytes).expect("write the file");
    (path, bytes)
}

fn run(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_ffrwd-wasm"))
        .args(args)
        .output()
        .expect("spawn ffrwd-wasm")
}

fn stdout(output: &Output) -> String {
    String::from_utf8_lossy(&output.stdout).to_string()
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).to_string()
}

fn binding(name: &str, path: &Path) -> String {
    format!("{name}={}", path.display())
}

#[test]
fn describe_says_whether_a_module_reads_its_files() {
    for (name, reads) in [("models_probe", true), ("nn_probe", false)] {
        let path = module(name);
        let output = run(&["--describe", &path.to_string_lossy()]);
        assert!(output.status.success(), "{}", stderr(&output));
        let described: serde_json::Value =
            serde_json::from_str(stdout(&output).trim()).expect("describe prints JSON");
        assert_eq!(described["models"], serde_json::json!(reads), "{name}");
    }
}

#[test]
fn a_module_reads_the_bytes_bound_for_it_by_name() {
    let probe = module("models_probe");
    let probe = probe.to_string_lossy().to_string();
    let (weights, bytes) = bytes_file("reads", "w.bin", 70_000);
    let (other, _) = bytes_file("reads", "other.bin", 10);
    let (config, _) = bytes_file("reads", "c.json", 3);
    let output = run(&[
        "-model",
        &binding("w.bin", &other),
        "-model",
        &binding("c.json", &config),
        "-model-for",
        &probe,
        "-model",
        &binding("w.bin", &weights),
        "-model-for",
        &probe,
        "-model",
        &binding("x.bin", &other),
        "-model-for",
        "elsewhere.wasm",
        "--invoke",
        &probe,
        "read",
        r#"{"name":"w.bin"}"#,
    ]);
    assert!(
        output.status.success(),
        "the run failed: {}",
        stderr(&output)
    );
    let result: serde_json::Value =
        serde_json::from_str(stdout(&output).trim()).expect("the result is JSON");
    let digest: String = Sha256::digest(&bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    assert_eq!(
        result,
        serde_json::json!({
            "len": bytes.len(),
            "sha256": digest,
            "names": ["c.json", "w.bin"],
        })
    );
}

#[test]
fn a_name_not_bound_for_this_module_is_the_hosts_err() {
    let probe = module("models_probe");
    let probe = probe.to_string_lossy().to_string();
    let (weights, _) = bytes_file("unbound", "w.bin", 16);
    let output = run(&[
        "-model",
        &binding("w.bin", &weights),
        "-model",
        &binding("v.bin", &weights),
        "-model-for",
        "elsewhere.wasm",
        "--invoke",
        &probe,
        "read",
        r#"{"name":"v.bin"}"#,
    ]);
    assert!(!output.status.success(), "{}", stdout(&output));
    assert!(
        stderr(&output).contains(
            "no file named 'v.bin' is bound for this module; the files bound for it are w.bin"
        ),
        "got: {}",
        stderr(&output)
    );

    let output = run(&["--invoke", &probe, "read", r#"{"name":"v.bin"}"#]);
    assert!(!output.status.success(), "{}", stdout(&output));
    assert!(
        stderr(&output)
            .contains("no file named 'v.bin' is bound for this module; no file is bound for it"),
        "got: {}",
        stderr(&output)
    );
}

#[test]
fn a_file_that_is_not_there_is_refused_before_anything_runs() {
    let output = run(&["-model", "nope=missing.bin", "--describe", "x.wasm"]);
    assert_eq!(output.status.code(), Some(2), "{}", stdout(&output));
    assert!(
        stderr(&output).contains(
            "-model nope=missing.bin: 'nope' is bound to missing.bin, which does not exist"
        ),
        "got: {}",
        stderr(&output)
    );
}
