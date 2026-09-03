//! A rows module driven straight through `ffrwd_wasm_runtime::runtime`, with
//! no stream and no ffmpeg: the contract the runtime holds a caller to,
//! whatever drove it. `fauxlate` is the fixture - see
//! `modules/fauxlate/src/lib.rs`.

use std::path::PathBuf;
use std::process::Command;
use std::sync::OnceLock;

use ffrwd_wasm_runtime::runtime::RowsModule;

/// Repo root, the parent of `runtime/`.
fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("runtime/ has a parent directory")
        .to_path_buf()
}

/// Absolute path to a module's built `.wasm` component, built once per test
/// binary. `modules/` is a separate cargo workspace with its own build lock,
/// so this does not deadlock against the `cargo test` run driving this
/// binary.
fn module_path(name: &str) -> PathBuf {
    static BUILT: OnceLock<()> = OnceLock::new();
    BUILT.get_or_init(|| {
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
            .current_dir(repo_root().join("modules"))
            .output()
            .expect("spawn cargo build for modules");
        assert!(
            output.status.success(),
            "building modules failed (status {:?}):\n{}",
            output.status.code(),
            String::from_utf8_lossy(&output.stderr)
        );
    });
    repo_root()
        .join("modules/target/wasm32-wasip2/release")
        .join(format!("{name}.wasm"))
}

fn open(name: &str) -> anyhow::Result<RowsModule> {
    let module = module_path(name);
    let module_str = module.to_str().expect("module path is valid UTF-8");
    RowsModule::open(module_str, "{}")
}

fn open_fauxlate() -> RowsModule {
    open("fauxlate").expect("opening fauxlate")
}

#[test]
fn translate_rows_carries_start_t_and_end_t_and_word_rules_the_text() {
    let mut module = open_fauxlate();

    let rows_in = vec![
        r#"{"start_t":0.0,"end_t":1.5,"text":"Cue one."}"#.to_string(),
        r#"{"start_t":1.5,"end_t":3.0,"text":"Wait!"}"#.to_string(),
    ];
    let rows_out = module.process(&rows_in).expect("processing cue rows");
    assert_eq!(
        rows_out,
        vec![
            r#"{"start_t":0.0,"end_t":1.5,"text":"Cue-a one-o."}"#.to_string(),
            r#"{"start_t":1.5,"end_t":3.0,"text":"Wait-a!"}"#.to_string(),
        ]
    );

    let trailing = module.finish().expect("finish");
    assert!(
        trailing.is_empty(),
        "fauxlate translates on process and holds nothing back, got {trailing:?}"
    );
}

#[test]
fn each_rows_own_text_restarts_the_alternation_at_a() {
    let mut module = open_fauxlate();

    // Two separate calls, each with one row: the second row's first word
    // still gets "-a", the same as the first row's did - the count is per
    // row, not carried across the module's whole life.
    let first = module
        .process(&[r#"{"start_t":0.0,"end_t":1.0,"text":"one two"}"#.to_string()])
        .expect("first call");
    assert_eq!(
        first,
        vec![r#"{"start_t":0.0,"end_t":1.0,"text":"one-a two-o"}"#.to_string()]
    );

    let second = module
        .process(&[r#"{"start_t":1.0,"end_t":2.0,"text":"three four"}"#.to_string()])
        .expect("second call");
    assert_eq!(
        second,
        vec![r#"{"start_t":1.0,"end_t":2.0,"text":"three-a four-o"}"#.to_string()]
    );
}

#[test]
fn a_row_that_is_not_a_cue_is_refused_naming_the_row() {
    let mut module = open_fauxlate();
    let refused = module.process(&[r#"{"text":"missing timestamps"}"#.to_string()]);
    let Err(error) = refused else {
        panic!("a row missing start_t/end_t must be refused");
    };
    let message = format!("{error:#}");
    assert!(
        message.contains("missing timestamps"),
        "the refusal does not name the row, got: {message}"
    );
}

#[test]
fn a_call_after_finish_is_refused() {
    let mut module = open_fauxlate();
    module.finish().expect("finish");
    let refused = module.process(&[]);
    assert!(
        refused
            .as_ref()
            .err()
            .is_some_and(|e| e.to_string().contains("finish")),
        "a call after finish must be refused, got {refused:?}"
    );
}

#[test]
fn a_frames_module_is_refused_by_name() {
    let Err(error) = open("invert") else {
        panic!("invert exports no rows module and must be refused");
    };
    let message = format!("{error:#}");
    assert!(
        message.contains("does not export") && message.contains("rows-module"),
        "the refusal does not name what invert actually exports, got: {message}"
    );
}
