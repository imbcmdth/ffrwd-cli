//! `--shape` and `--describe` on a node module, and `--shape` on a module of
//! an older world, as the compiler reads them.

use std::path::PathBuf;
use std::process::{Command, Output};
use std::sync::OnceLock;

fn sidecar_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("ffrwd-wasm/ has a parent directory")
        .to_path_buf()
}

/// Builds the two modules these tests read, once per test binary.
fn module(name: &str) -> String {
    static BUILT: OnceLock<()> = OnceLock::new();
    BUILT.get_or_init(|| {
        let output = Command::new("cargo")
            .args([
                "build",
                "--release",
                "--target",
                "wasm32-wasip2",
                "-p",
                "shape-probe",
                "-p",
                "facebox",
            ])
            .current_dir(sidecar_root().join("modules"))
            .output()
            .expect("spawn cargo build");
        assert!(
            output.status.success(),
            "building the modules failed:\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
    });
    sidecar_root()
        .join(format!("modules/target/wasm32-wasip2/release/{name}.wasm"))
        .to_str()
        .expect("a UTF-8 path")
        .to_string()
}

fn sidecar(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_ffrwd-wasm"))
        .args(args)
        .output()
        .expect("spawn ffrwd-wasm")
}

fn json(output: &Output) -> serde_json::Value {
    assert!(
        output.status.success(),
        "exited {:?}:\n{}",
        output.status.code(),
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).expect("one JSON line")
}

#[test]
fn describe_names_a_node_module_by_its_world() {
    let probe = module("shape_probe");
    let d = json(&sidecar(&["--describe", &probe]));
    assert_eq!(d["world"], "node-module");
    assert_eq!(d["node"], true);
    assert_eq!(d["name"], "shape_probe");
    assert_eq!(d["pixel_formats"], serde_json::json!([]));

    let facebox = module("facebox");
    let d = json(&sidecar(&["--describe", &facebox]));
    assert_eq!(d["world"], "ffrwd:av@0.18.0");
    assert!(d.get("node").is_none(), "absent for every other module");
}

#[test]
fn shape_prints_the_record_with_its_variants_as_kinds() {
    let probe = module("shape_probe");
    let s = json(&sidecar(&[
        "--shape",
        &probe,
        "--params",
        r#"{"canvas":{"width":640,"height":360}}"#,
        "--bound",
        "v,feed",
    ]));
    assert_eq!(
        s["clock"],
        serde_json::json!({"kind": "input", "port": "v"})
    );
    assert_eq!(s["one_to_one"], false);
    let feed = &s["inputs"][1];
    assert_eq!(feed["name"], "feed");
    assert_eq!(feed["pairing"]["kind"], "hold");
    assert_eq!(
        feed["pairing"]["anchor"],
        serde_json::json!({"kind": "tagged", "tag": "smart_timed"})
    );
    assert_eq!(feed["pairing"]["port_param"], "port");
    assert_eq!(s["inputs"][2]["rows"], "state");
    let outputs: Vec<&str> = s["outputs"]
        .as_array()
        .expect("a list")
        .iter()
        .map(|o| o["name"].as_str().expect("a name"))
        .collect();
    assert_eq!(outputs, ["mask", "copy", "canvas", "spots"]);
    assert_eq!(
        s["outputs"][2]["format"],
        serde_json::json!({"kind": "video", "width": 640, "height": 360, "pix_fmt": "rgba", "color": null})
    );
    assert_eq!(
        s["outputs"][3]["time_base"],
        serde_json::json!({"num": 1, "den": 1000000})
    );
}

#[test]
fn shape_refuses_what_the_wit_refuses_and_a_port_not_declared() {
    let probe = module("shape_probe");
    let run = sidecar(&[
        "--shape",
        &probe,
        "--params",
        r#"{"refuse":"data_ignores_rows"}"#,
    ]);
    assert_eq!(run.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&run.stderr);
    assert!(stderr.contains("ignores its rows"), "{stderr}");

    let run = sidecar(&["--shape", &probe, "--bound", "v,nope"]);
    let stderr = String::from_utf8_lossy(&run.stderr);
    assert!(stderr.contains("bound input 'nope'"), "{stderr}");
}

#[test]
fn shape_on_an_older_module_is_the_shape_it_runs_as() {
    let facebox = module("facebox");
    let s = json(&sidecar(&["--shape", &facebox, "--bound", "in0,rows"]));
    assert_eq!(
        s["clock"],
        serde_json::json!({"kind": "input", "port": "in0"})
    );
    let inputs: Vec<(&str, &str)> = s["inputs"]
        .as_array()
        .expect("a list")
        .iter()
        .map(|i| (i["name"].as_str().unwrap(), i["kind"].as_str().unwrap()))
        .collect();
    assert_eq!(inputs, [("in0", "video"), ("rows", "data")]);
    let outputs: Vec<(&str, &str)> = s["outputs"]
        .as_array()
        .expect("a list")
        .iter()
        .map(|o| (o["name"].as_str().unwrap(), o["kind"].as_str().unwrap()))
        .collect();
    assert_eq!(outputs, [("out", "video"), ("rows", "data")]);
    assert!(s["outputs"][1]["schema"]
        .as_str()
        .expect("the rows schema")
        .contains("\"x\""));
}
