//! Inference through `wasi:nn`, driven through the binary's own argv.
//!
//! Every case spawns `ffrwd-wasm`, so each gets a fresh process: models load
//! once per process, and a run that registers none is a different run rather
//! than a different order of these tests.
//!
//! ONNX Runtime is not in git, so the cases that need one skip - loudly - when
//! neither the cache nor `FFRWD_NN_RUNTIME` holds one. `ffrwd setup nn`
//! downloads one.

use std::path::PathBuf;
use std::process::{Command, Output};
use std::sync::OnceLock;

/// The sidecar's directory, the parent of `ffrwd-wasm/`.
fn sidecar_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("ffrwd-wasm/ has a parent directory")
        .to_path_buf()
}

/// Builds the probe module for wasm32-wasip2, once per test binary.
/// `modules/` is a separate cargo workspace with its own build lock, so this
/// does not deadlock against the `cargo test` run driving this binary.
fn probe_module() -> PathBuf {
    static BUILT: OnceLock<()> = OnceLock::new();
    BUILT.get_or_init(|| {
        let output = Command::new("cargo")
            .args([
                "build",
                "--release",
                "--target",
                "wasm32-wasip2",
                "-p",
                "nn-probe",
            ])
            .current_dir(sidecar_root().join("modules"))
            .output()
            .expect("spawn cargo build for nn-probe");
        assert!(
            output.status.success(),
            "building nn-probe failed (status {:?}):\n{}",
            output.status.code(),
            String::from_utf8_lossy(&output.stderr)
        );
    });
    sidecar_root().join("modules/target/wasm32-wasip2/release/nn_probe.wasm")
}

fn tiny_model() -> PathBuf {
    sidecar_root().join("modules/nn-probe/model/tiny.onnx")
}
/// Where `ffrwd setup nn` puts a runtime: the user's cache, keyed by the
/// version this build demands and the platform it runs on.
fn cached_runtime_dir() -> Option<PathBuf> {
    let home = std::env::var_os(if cfg!(windows) { "USERPROFILE" } else { "HOME" })?;
    Some(
        PathBuf::from(home)
            .join(".cache/ffrwd/nn-runtime")
            .join(ffrwd_wasm_runtime::nn::ort_version())
            .join(ffrwd_wasm_runtime::nn::platform()),
    )
}

/// Where ONNX Runtime is: what the environment says, else the cache.
/// `None` when neither holds a runtime library.
fn runtime_dir() -> Option<PathBuf> {
    let named = std::env::var_os("FFRWD_NN_RUNTIME").map(PathBuf::from);
    let candidates = [named, cached_runtime_dir()];
    candidates.into_iter().flatten().find(|dir| {
        [
            "onnxruntime.dll",
            "libonnxruntime.so",
            "libonnxruntime.dylib",
        ]
        .iter()
        .any(|lib| dir.join(lib).is_file())
    })
}

/// Says why a test did nothing, since a skip is otherwise silent.
fn announce_skip(what: &str) {
    eprintln!("SKIPPED: {what}. Run `ffrwd setup nn` to download an ONNX Runtime.");
}

/// Runs the binary with `args`, with the runtime directory named explicitly so
/// an environment variable set outside cannot change the outcome.
fn run_with_runtime(runtime: &std::path::Path, args: &[&str]) -> Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_ffrwd-wasm"));
    command
        .env_remove("FFRWD_NN_RUNTIME")
        .arg("-nn-runtime")
        .arg(runtime);
    command.args(args);
    command.output().expect("spawn ffrwd-wasm")
}

/// Runs the binary with no runtime directory anywhere.
fn run_bare(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_ffrwd-wasm"))
        .env_remove("FFRWD_NN_RUNTIME")
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

#[test]
fn a_module_that_asks_for_inference_in_a_run_with_no_model_is_refused_by_name() {
    let module = probe_module();
    let module = module.to_string_lossy().to_string();
    let output = run_bare(&[
        "--invoke",
        &module,
        "run",
        r#"{"model":"tiny","input":[1,2,3,4]}"#,
    ]);

    assert!(
        !output.status.success(),
        "a run with no model should refuse"
    );
    let message = stderr(&output);
    assert!(
        message.contains("imports wasi:nn") && message.contains("-nn name=path"),
        "the refusal should name the interface and the option: {message}"
    );
    assert!(
        message.contains("nn_probe.wasm"),
        "the refusal should name the module: {message}"
    );
}

#[test]
fn a_model_with_no_runtime_to_load_it_is_refused_before_anything_loads() {
    let module = probe_module();
    let output = run_bare(&[
        "-nn",
        &format!("tiny={}", tiny_model().display()),
        "--invoke",
        &module.to_string_lossy(),
        "run",
        r#"{"model":"tiny","input":[1,2,3,4]}"#,
    ]);

    assert!(
        !output.status.success(),
        "a model with no runtime should refuse"
    );
    let message = stderr(&output);
    assert!(
        message.contains("-nn-runtime") && message.contains("FFRWD_NN_RUNTIME"),
        "the refusal should name both ways to say where the runtime is: {message}"
    );
    assert!(
        message.contains("ffrwd setup nn"),
        "the refusal should say what to fetch: {message}"
    );
}

#[test]
fn a_model_name_is_spelled_the_way_a_module_name_is() {
    let output = run_bare(&["-nn", "bad-name=x.onnx", "--describe", "x.wasm"]);
    assert!(!output.status.success());
    assert!(
        stderr(&output).contains("is not a model name"),
        "got: {}",
        stderr(&output)
    );

    let output = run_bare(&["-nn", "tiny", "--describe", "x.wasm"]);
    assert!(!output.status.success());
    assert!(
        stderr(&output).contains("-nn name=path"),
        "got: {}",
        stderr(&output)
    );

    let output = run_bare(&["-nn-target", "tpu", "--describe", "x.wasm"]);
    assert!(!output.status.success());
    assert!(
        stderr(&output).contains("the targets are cpu, gpu, cuda, directml and coreml"),
        "got: {}",
        stderr(&output)
    );
}

#[test]
fn nn_info_answers_with_no_runtime_anywhere() {
    // The whole point of it: what to fetch is asked of a machine that has
    // fetched nothing yet, so nothing here may touch an ONNX Runtime.
    let output = run_bare(&["--nn-info"]);
    assert!(
        output.status.success(),
        "--nn-info failed: {}",
        stderr(&output)
    );
    let info: serde_json::Value =
        serde_json::from_str(&stdout(&output)).expect("--nn-info prints JSON");

    let version = info["ort_version"]
        .as_str()
        .expect("ort_version is a string");
    assert!(
        version.starts_with("1.") && version.split('.').count() == 3,
        "ort_version reads 1.<minor>.<patch>, got {version}"
    );

    let providers: Vec<&str> = info["providers"]
        .as_array()
        .expect("providers is an array")
        .iter()
        .map(|p| p.as_str().expect("a provider is a string"))
        .collect();
    // DirectML asks for nothing installed and CUDA asks for a toolkit, so on
    // Windows the one that works on any Direct3D 12 adapter is tried first.
    let expected: &[&str] = if cfg!(windows) {
        &["directml", "cuda", "cpu"]
    } else if cfg!(target_os = "macos") {
        &["coreml", "cpu"]
    } else {
        &["cuda", "cpu"]
    };
    assert_eq!(providers, expected);

    let platform = info["platform"].as_str().expect("platform is a string");
    assert!(
        platform.starts_with(if cfg!(windows) {
            "win-"
        } else if cfg!(target_os = "macos") {
            "osx-"
        } else {
            "linux-"
        }),
        "the platform key names this operating system, got {platform}"
    );
}

#[test]
fn every_provider_has_a_spelling_of_its_own() {
    // A named provider is accepted everywhere `cpu` is, whether or not this
    // machine can engage it: what it cannot do is warn and run on the CPU.
    for target in ["cpu", "gpu", "cuda", "directml", "coreml"] {
        let output = run_bare(&["-nn-target", target, "--describe", "x.wasm"]);
        let message = stderr(&output);
        assert!(
            !message.contains("-nn-target"),
            "-nn-target {target} should be a target: {message}"
        );
    }
}

#[test]
fn the_probe_module_runs_the_graph_the_host_registered() {
    let Some(runtime) = runtime_dir() else {
        announce_skip("the probe module cannot run without a runtime");
        return;
    };
    let module = probe_module();
    let output = run_with_runtime(
        &runtime,
        &[
            "-nn",
            &format!("tiny={}", tiny_model().display()),
            "--invoke",
            &module.to_string_lossy(),
            "run",
            r#"{"model":"tiny","input":[1,2,3,4]}"#,
        ],
    );

    assert!(
        output.status.success(),
        "the run failed:\n{}",
        stderr(&output)
    );
    // The graph is y = a @ w + b. With w's rows [1,0] [0,1] [1,1] [2,-1] and
    // b [0.5,-0.5], the input [1,2,3,4] gives 1+3+8+0.5 and 2+3-4-0.5.
    let result: serde_json::Value =
        serde_json::from_str(stdout(&output).trim()).expect("the result is JSON");
    assert_eq!(result["dimensions"], serde_json::json!([1, 2]));
    assert_eq!(result["output"], serde_json::json!([12.5, 0.5]));
}

#[test]
fn the_provider_that_engaged_is_reported() {
    let Some(runtime) = runtime_dir() else {
        announce_skip("there is no session to report a provider for");
        return;
    };
    let module = probe_module();
    let output = run_with_runtime(
        &runtime,
        &[
            "-nn",
            &format!("tiny={}", tiny_model().display()),
            "--invoke",
            &module.to_string_lossy(),
            "run",
            r#"{"model":"tiny","input":[1,2,3,4]}"#,
        ],
    );

    let message = stderr(&output);
    assert!(
        message.contains("[nn] execution provider:"),
        "a session that ran should say which provider did it: {message}"
    );
    // The runtime that answered is named, with where it came from.
    assert!(
        message.contains("onnxruntime") && message.contains("(fetched)"),
        "the verdict should attribute the library it loaded: {message}"
    );
}

#[test]
fn a_gpu_target_that_ended_up_on_the_cpu_says_so() {
    let Some(runtime) = runtime_dir() else {
        announce_skip("there is no session to establish a provider for");
        return;
    };
    let module = probe_module();
    let output = run_with_runtime(
        &runtime,
        &[
            "-nn",
            &format!("tiny={}", tiny_model().display()),
            "-nn-target",
            "gpu",
            "--invoke",
            &module.to_string_lossy(),
            "run",
            r#"{"model":"tiny","input":[1,2,3,4]}"#,
        ],
    );

    assert!(
        output.status.success(),
        "a gpu target falls back rather than failing:\n{}",
        stderr(&output)
    );
    let message = stderr(&output);
    // A machine with an accelerator this runtime directory can reach engages
    // it, and there is no fall back to report.
    if !message.contains("execution provider: CPU") {
        return;
    }
    assert!(
        message.contains("-nn-target gpu was requested") && message.contains("tiny"),
        "a fall back to the CPU is never silent, and names the model: {message}"
    );
}

#[test]
fn a_name_that_was_never_bound_comes_back_not_found() {
    let Some(runtime) = runtime_dir() else {
        announce_skip("there is no registry to miss in");
        return;
    };
    let module = probe_module();
    let output = run_with_runtime(
        &runtime,
        &[
            "-nn",
            &format!("tiny={}", tiny_model().display()),
            "--invoke",
            &module.to_string_lossy(),
            "run",
            r#"{"model":"nope","input":[1,2,3,4]}"#,
        ],
    );

    assert!(!output.status.success(), "an unbound name is a failure");
    let message = stderr(&output);
    assert!(
        message.contains("not-found") && message.contains("nope"),
        "the interface's own error code should reach the caller: {message}"
    );
}

#[test]
fn the_guest_doing_inference_can_still_read_nothing() {
    let Some(runtime) = runtime_dir() else {
        announce_skip("the sandbox probe needs a session to run beside");
        return;
    };
    let module = probe_module();
    let output = run_with_runtime(
        &runtime,
        &[
            "-nn",
            &format!("tiny={}", tiny_model().display()),
            "--invoke",
            &module.to_string_lossy(),
            "sandbox",
            "{}",
        ],
    );

    assert!(
        output.status.success(),
        "the sandbox probe failed:\n{}",
        stderr(&output)
    );
    let result: serde_json::Value =
        serde_json::from_str(stdout(&output).trim()).expect("the result is JSON");
    assert_eq!(
        result["reachable"],
        serde_json::json!([]),
        "a module that does inference has no filesystem: {result}"
    );
    assert!(
        result["denied"].as_array().is_some_and(|d| d.len() == 4),
        "every path the guest tried should have been denied: {result}"
    );
}

// The depth-of-field pair end to end: depth measures the frame, blur_mask
// blurs it by what depth found. Real ffmpeg on both ends, one sidecar in the
// middle hosting both modules, and the graph reached by the name `-nn` bound
// it to.

/// Builds every module for wasm32-wasip2, which the chain needs two of.
fn chain_modules() -> PathBuf {
    static BUILT: OnceLock<()> = OnceLock::new();
    BUILT.get_or_init(|| {
        let output = Command::new("cargo")
            .args(["build", "--release", "--target", "wasm32-wasip2"])
            .current_dir(sidecar_root().join("modules"))
            .output()
            .expect("spawn cargo build for modules");
        assert!(
            output.status.success(),
            "building modules failed (status {:?}):\n{}",
            output.status.code(),
            String::from_utf8_lossy(&output.stderr)
        );
    });
    sidecar_root().join("modules/target/wasm32-wasip2/release")
}

/// Where `modules/depth/fetch-model.ps1` puts the graph.
fn depth_model() -> PathBuf {
    sidecar_root().join("modules/depth/model/model.onnx")
}

/// Frames a chain run puts through, and the picture they come from.
const CHAIN_FRAMES: u32 = 6;
const CHAIN_SOURCE: &str = "testsrc2=s=64x64:r=25:d=0.24";

/// Writes a NUT file of raw frames from an lavfi source.
fn write_chain_input(target: &std::path::Path) {
    let output = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            CHAIN_SOURCE,
            "-frames:v",
            &CHAIN_FRAMES.to_string(),
            "-c:v",
            "rawvideo",
            "-pix_fmt",
            "yuv420p",
            "-f",
            "nut",
        ])
        .arg(target)
        .stdin(std::process::Stdio::null())
        .output()
        .expect("spawn ffmpeg");
    assert!(
        output.status.success(),
        "writing the chain input failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

/// How many video packets a file holds, as ffprobe counts them.
fn frame_count(path: &std::path::Path) -> usize {
    let output = Command::new("ffprobe")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-select_streams",
            "v:0",
            "-count_packets",
            "-show_entries",
            "stream=nb_read_packets",
            "-of",
            "csv=p=0",
        ])
        .arg(path)
        .stdin(std::process::Stdio::null())
        .output()
        .expect("spawn ffprobe");
    String::from_utf8_lossy(&output.stdout)
        .trim()
        .trim_end_matches(',')
        .parse()
        .unwrap_or_else(|e| {
            panic!(
                "parsing ffprobe's packet count {:?}: {e}",
                String::from_utf8_lossy(&output.stdout)
            )
        })
}

#[test]
fn the_depth_of_field_chain_runs_and_keeps_every_frame() {
    let Some(runtime) = runtime_dir() else {
        announce_skip("the depth chain needs an ONNX Runtime");
        return;
    };
    let model = depth_model();
    if !model.is_file() {
        eprintln!(
            "SKIPPED: depth's graph is absent. Run \
             sidecar/modules/depth/fetch-model.ps1 to download it."
        );
        return;
    }
    let built = chain_modules();

    let dir = std::env::temp_dir().join(format!("ffrwd-wasm-chain-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("a directory to work in");
    let input = dir.join("in.nut");
    let output_path = dir.join("out.nut");
    write_chain_input(&input);

    // The wiring the compiler emits for blur_mask(v, depth(v)): one boundary
    // read feeding both the model and the blur.
    let result = run_with_runtime(
        &runtime,
        &[
            "-nn",
            &format!("depth={}", model.display()),
            "-f",
            "nut",
            "-i",
            &input.to_string_lossy(),
            "-m",
            &format!("depth={}", built.join("depth.wasm").display()),
            "-m",
            &format!("blur_mask={}", built.join("blur_mask.wasm").display()),
            "-filter_complex",
            "[0:v]depth[n1];[0:v][n1]blur_mask[out0]",
            "-map",
            "[out0]",
            "-f",
            "nut",
            &output_path.to_string_lossy(),
        ],
    );
    assert!(
        result.status.success(),
        "the depth chain failed:\n{}",
        stderr(&result)
    );
    assert!(
        stderr(&result).contains("[nn] execution provider:"),
        "the run must say which provider engaged:\n{}",
        stderr(&result)
    );
    assert_eq!(
        frame_count(&output_path),
        CHAIN_FRAMES as usize,
        "one frame out per frame in"
    );

    let _ = std::fs::remove_dir_all(&dir);
}
