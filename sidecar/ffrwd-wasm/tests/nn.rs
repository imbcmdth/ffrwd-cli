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
        message.contains("[nn] tiny: execution provider "),
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
    if !message.contains("[nn] tiny: execution provider CPU") {
        return;
    }
    assert!(
        message.contains("[nn] tiny: execution provider CPU, "),
        "a fall back to the CPU is never silent, and names the model and why: {message}"
    );
}

#[test]
fn nn_exclude_refuses_cpu_and_a_misspelling() {
    let output = run_bare(&["-nn-exclude", "cpu", "--describe", "x.wasm"]);
    assert!(!output.status.success());
    assert!(
        stderr(&output).contains("cannot be excluded"),
        "got: {}",
        stderr(&output)
    );

    let output = run_bare(&["-nn-exclude", "dml", "--describe", "x.wasm"]);
    assert!(!output.status.success());
    assert!(
        stderr(&output).contains("cuda, directml and coreml"),
        "got: {}",
        stderr(&output)
    );
}

#[test]
fn nn_exclude_takes_a_model_name_and_refuses_one_never_bound() {
    let model = format!("tiny={}", tiny_model().display());

    let output = run_bare(&[
        "-nn",
        &model,
        "-nn-exclude",
        "tiny=directml",
        "--describe",
        "x.wasm",
    ]);
    let message = stderr(&output);
    assert!(
        !message.contains("-nn-exclude"),
        "an exclusion naming a bound model should be accepted: {message}"
    );

    // The pair may come before the binding it names.
    let output = run_bare(&[
        "-nn-exclude",
        "tiny=cuda",
        "-nn",
        &model,
        "--describe",
        "x.wasm",
    ]);
    let message = stderr(&output);
    assert!(
        !message.contains("-nn-exclude"),
        "the order of -nn and -nn-exclude should not matter: {message}"
    );

    let output = run_bare(&[
        "-nn",
        &model,
        "-nn-exclude",
        "tinny=cuda",
        "--describe",
        "x.wasm",
    ]);
    assert!(!output.status.success());
    assert!(
        stderr(&output).contains("-nn-exclude tinny=cuda: no -nn binds tinny"),
        "got: {}",
        stderr(&output)
    );

    let output = run_bare(&[
        "-nn",
        &model,
        "-nn-exclude",
        "tiny=cpu",
        "--describe",
        "x.wasm",
    ]);
    assert!(!output.status.success());
    assert!(
        stderr(&output).contains("cannot be excluded"),
        "got: {}",
        stderr(&output)
    );
}

#[test]
fn nn_exclude_accepts_every_deniable_provider() {
    // A named provider is accepted whether or not this machine can offer it:
    // only cpu, and a misspelling, are refused.
    for provider in ["cuda", "directml", "coreml"] {
        let output = run_bare(&["-nn-exclude", provider, "--describe", "x.wasm"]);
        let message = stderr(&output);
        assert!(
            !message.contains("-nn-exclude"),
            "-nn-exclude {provider} should be accepted: {message}"
        );
    }
}

#[test]
fn a_gpu_walk_with_directml_excluded_never_engages_it() {
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
            "-nn-exclude",
            "directml",
            "--invoke",
            &module.to_string_lossy(),
            "run",
            r#"{"model":"tiny","input":[1,2,3,4]}"#,
        ],
    );

    assert!(
        output.status.success(),
        "excluding a provider from the walk still runs, on whatever is left:\n{}",
        stderr(&output)
    );
    let message = stderr(&output);
    assert!(
        message.contains("[nn] -nn-exclude: directml"),
        "the excluded provider is named once, regardless of the walk's outcome: {message}"
    );
    assert!(
        !message.contains("execution provider DIRECTML"),
        "an excluded provider never engages: {message}"
    );
}

#[test]
fn a_model_excluding_the_walks_provider_runs_on_the_cpu_beside_one_that_does_not() {
    let Some(runtime) = runtime_dir() else {
        announce_skip("there is no session to establish a provider for");
        return;
    };
    let module = probe_module();
    let module_arg = module.to_string_lossy().to_string();
    let model = tiny_model();
    let pinned_arg = format!("pinned={}", model.display());
    let free_arg = format!("free={}", model.display());
    // The platform's first GPU provider: DirectML on Windows, CUDA on Linux,
    // CoreML on macOS.
    let first = ffrwd_wasm_runtime::nn::providers()[0].clone();
    let exclusion = format!("pinned={first}");
    let first = first.to_ascii_uppercase();

    for asked in ["pinned", "free"] {
        let input = format!(r#"{{"model":"{asked}","input":[1,2,3,4]}}"#);
        let output = run_with_runtime(
            &runtime,
            &[
                "-nn",
                &pinned_arg,
                "-nn",
                &free_arg,
                "-nn-target",
                "gpu",
                "-nn-exclude",
                &exclusion,
                "--invoke",
                &module_arg,
                "run",
                &input,
            ],
        );
        let message = stderr(&output);
        assert!(
            output.status.success(),
            "{asked} should answer whichever provider it runs on:\n{message}"
        );
        let result: serde_json::Value =
            serde_json::from_str(stdout(&output).trim()).expect("the result is JSON");
        assert_eq!(result["output"], serde_json::json!([12.5, 0.5]), "{asked}");

        assert!(
            message.contains(&format!("[nn] -nn-exclude: {exclusion}")),
            "the exclusion is named as given: {message}"
        );
        // One model still takes the first provider, so the walk runs its
        // preflight rather than passing it over for a pin.
        assert!(
            !message.contains("every model's pin"),
            "one model's exclusion does not exclude it for the walk: {message}"
        );
        if message.contains(&format!("-nn-target gpu: not {first},")) {
            announce_skip(&format!(
                "{first} is not available here, so the walk did not take it"
            ));
            return;
        }
        assert!(
            message.contains(&format!(
                "[nn] pinned: execution provider CPU, {first} excluded by its pin"
            )),
            "the model that excludes the walk's provider runs on the CPU: {message}"
        );
        let free = message
            .lines()
            .find(|line| line.starts_with("[nn] free: execution provider "))
            .unwrap_or_else(|| panic!("the other model's verdict is named: {message}"));
        assert!(
            free == format!("[nn] free: execution provider {first}")
                || free == format!("[nn] free: execution provider CPU, {first} did not engage"),
            "the model that does not exclude it asks for it: {free}"
        );
    }
}

#[test]
fn every_gpu_provider_excluded_runs_on_cpu_and_says_why() {
    let Some(runtime) = runtime_dir() else {
        announce_skip("there is no session to establish a provider for");
        return;
    };
    let module = probe_module();
    let model_arg = format!("tiny={}", tiny_model().display());
    let module_arg = module.to_string_lossy().to_string();
    // Every provider `-nn-target gpu` could reach on this platform, so the
    // walk has nowhere left to land but the CPU.
    let output = run_with_runtime(
        &runtime,
        &[
            "-nn",
            &model_arg,
            "-nn-target",
            "gpu",
            "-nn-exclude",
            "directml",
            "-nn-exclude",
            "cuda",
            "-nn-exclude",
            "coreml",
            "--invoke",
            &module_arg,
            "run",
            r#"{"model":"tiny","input":[1,2,3,4]}"#,
        ],
    );

    assert!(
        output.status.success(),
        "a walk with nowhere left still runs, on the cpu:\n{}",
        stderr(&output)
    );
    let message = stderr(&output);
    let verdict = message
        .lines()
        .find(|line| line.starts_with("[nn] tiny: execution provider "))
        .unwrap_or_else(|| panic!("the model's verdict is named: {message}"));
    assert!(
        verdict.starts_with("[nn] tiny: execution provider CPU, ")
            && verdict.ends_with("excluded by its pin"),
        "a walk left with only cpu by exclusion says why it is slow: {message}"
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
        stderr(&result).contains("[nn] depth: execution provider "),
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

// Frames made into a model's input by the host (`ffrwd:gpu/nn`): nn_frames
// asks for each frame as a tensor and runs a graph of each channel's mean
// and largest value on it. The same frames give the same rows whichever
// way the tensor was made: on the CPU from host memory, on the device and
// read back, or on the device and bound to a CUDA session in place.

/// Builds the nn-frames module, once per test binary.
fn frames_module() -> PathBuf {
    static BUILT: OnceLock<()> = OnceLock::new();
    BUILT.get_or_init(|| {
        let output = Command::new("cargo")
            .args([
                "build",
                "--release",
                "--target",
                "wasm32-wasip2",
                "-p",
                "nn-frames",
            ])
            .current_dir(sidecar_root().join("modules"))
            .output()
            .expect("spawn cargo build for nn-frames");
        assert!(
            output.status.success(),
            "building nn-frames failed:\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
    });
    sidecar_root().join("modules/target/wasm32-wasip2/release/nn_frames.wasm")
}

/// `channel-means.onnx`, or its NHWC twin.
fn means_model(nhwc: bool) -> PathBuf {
    sidecar_root().join(if nhwc {
        "modules/nn-frames/model/channel-means-nhwc.onnx"
    } else {
        "modules/nn-frames/model/channel-means.onnx"
    })
}

const MEANS_WIDTH: u32 = 64;
const MEANS_HEIGHT: u32 = 48;
const MEANS_FRAMES: usize = 4;

/// A NUT file of `MEANS_FRAMES` yuv420p frames, each a different picture
/// of ramps, so every channel's mean and largest value move.
fn means_input(dir: &std::path::Path) -> PathBuf {
    use ffrwd_wasm::nut::{Muxer, Stream, TimeBase};
    let (w, h) = (MEANS_WIDTH as usize, MEANS_HEIGHT as usize);
    let (cw, ch) = (w / 2, h / 2);
    let stream = Stream::video(
        "yuv420p",
        MEANS_WIDTH,
        MEANS_HEIGHT,
        TimeBase { num: 1, den: 25 },
    )
    .expect("yuv420p is carried");
    let path = dir.join("means.nut");
    let mut wire = Vec::new();
    {
        let mut muxer = Muxer::new(&mut wire, &stream).expect("write NUT headers");
        for k in 0..MEANS_FRAMES {
            let mut frame = Vec::with_capacity(w * h + 2 * cw * ch);
            for y in 0..h {
                for x in 0..w {
                    frame.push((16 + (x * 3 + y * 2 + k * 29) % 220) as u8);
                }
            }
            for plane in 0..2 {
                for y in 0..ch {
                    for x in 0..cw {
                        let ramp = if plane == 0 {
                            x * 7 + k * 11
                        } else {
                            y * 9 + k * 5
                        };
                        frame.push((16 + ramp % 225) as u8);
                    }
                }
            }
            muxer.write_frame(k as i64, &frame).expect("write a frame");
        }
        muxer.finish().expect("finish the stream");
    }
    std::fs::write(&path, wire).expect("write the input");
    path
}

/// One nn_frames run over `input`, its rows and its stderr. `params` are the
/// module's after `model=means`.
fn means_run(
    runtime: &std::path::Path,
    dir: &std::path::Path,
    input: &std::path::Path,
    target: &str,
    nhwc: bool,
    params: &str,
    extra: &[&str],
) -> (Vec<serde_json::Value>, String) {
    let module = frames_module();
    let module = module.to_string_lossy().to_string();
    let rows = dir.join(format!(
        "rows-{target}-{}.ndjson",
        params.replace([':', '='], "_")
    ));
    let _ = std::fs::remove_file(&rows);
    let mut args: Vec<String> = extra.iter().map(|a| a.to_string()).collect();
    args.extend([
        "-nn".to_string(),
        format!("means={}", means_model(nhwc).display()),
        "-nn-target".to_string(),
        target.to_string(),
        "-gpu".to_string(),
        module.clone(),
        "-f".to_string(),
        "nut".to_string(),
        "-i".to_string(),
        input.display().to_string(),
        "-m".to_string(),
        format!("nn_frames={module}"),
        "-filter_complex".to_string(),
        format!("[v=0:v]nn_frames=model=means:{params}[@rows=r]"),
        "-map".to_string(),
        "[r]".to_string(),
        "-f".to_string(),
        "ndjson".to_string(),
        rows.display().to_string(),
    ]);
    let refs: Vec<&str> = args.iter().map(String::as_str).collect();
    let output = run_with_runtime(runtime, &refs);
    let message = stderr(&output);
    assert!(
        output.status.success() || message.contains("no GPU"),
        "nn_frames ({target}, {params}) exited {:?}:\n{message}",
        output.status.code()
    );
    let written = std::fs::read_to_string(&rows).unwrap_or_default();
    let parsed = written
        .lines()
        .map(|line| serde_json::from_str(line).expect("a row is JSON"))
        .collect();
    (parsed, message)
}

/// Every output value of every row, in order.
fn output_values(rows: &[serde_json::Value]) -> Vec<f64> {
    rows.iter()
        .flat_map(|row| {
            ["mean", "max"].into_iter().flat_map(move |name| {
                row["outputs"][name]
                    .as_array()
                    .unwrap_or_else(|| panic!("{name} is a list: {row}"))
                    .iter()
                    .map(|v| v.as_f64().expect("a number"))
                    .collect::<Vec<_>>()
            })
        })
        .collect()
}

/// The largest difference between two runs' numbers.
fn apart(a: &[serde_json::Value], b: &[serde_json::Value]) -> f64 {
    output_values(a)
        .iter()
        .zip(&output_values(b))
        .map(|(p, q)| (p - q).abs())
        .fold(0f64, f64::max)
}

/// Asserts two runs' rows hold the same numbers within float rounding.
fn assert_same_rows(a: &[serde_json::Value], b: &[serde_json::Value], what: &str) {
    assert_eq!(a.len(), MEANS_FRAMES, "{what}: one row per frame: {a:?}");
    assert_eq!(a.len(), b.len(), "{what}: as many rows");
    let worst = apart(a, b);
    assert!(
        worst < 1e-4,
        "{what}: apart by up to {worst}:\n{a:?}\n{b:?}"
    );
    assert!(
        output_values(a).iter().all(|v| (0.0..=1.0).contains(v)),
        "{what}: 0 to 1 in, 0 to 1 out: {a:?}"
    );
}

/// The `download` stage's bytes in the run's profile row.
fn downloaded_bytes(message: &str) -> u64 {
    let row: serde_json::Value = message
        .lines()
        .filter_map(|line| line.strip_prefix("ffrwd:row "))
        .map(|line| serde_json::from_str::<serde_json::Value>(line).expect("a row is JSON"))
        .find(|row| row["kind"] == "profile")
        .unwrap_or_else(|| panic!("no profile row on stderr:\n{message}"));
    row["download"]["bytes"].as_u64().expect("download bytes")
}

/// A scratch directory of the test's own.
fn means_dir(test: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("ffrwd-nn-{test}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("a directory to work in");
    dir
}

#[test]
fn a_frame_made_into_a_tensor_gives_the_same_rows_on_every_route() {
    let Some(runtime) = runtime_dir() else {
        announce_skip("nn_frames cannot run a model without a runtime");
        return;
    };
    let dir = means_dir("routes");
    let input = means_input(&dir);
    let tensor = u64::from(MEANS_WIDTH / 2 * MEANS_HEIGHT / 2 * 3 * 4);
    let size = format!("width={}:height={}", MEANS_WIDTH / 2, MEANS_HEIGHT / 2);
    let profile = ["-profile"];

    // On the CPU, from host memory: the reference.
    let (host, message) = means_run(
        &runtime,
        &dir,
        &input,
        "cpu",
        false,
        &format!("memory=host:{size}"),
        &profile,
    );
    assert!(
        message.contains("[nn] means: input in host memory, made on the CPU"),
        "{message}"
    );
    assert_eq!(downloaded_bytes(&message), 0, "nothing is on the GPU");
    assert_eq!(host.len(), MEANS_FRAMES, "{message}");

    // On the device and read back, for a CPU session.
    let (downloaded, message) = means_run(
        &runtime,
        &dir,
        &input,
        "cpu",
        false,
        &format!("memory=gpu:{size}"),
        &profile,
    );
    if message.contains("no GPU") {
        eprintln!("SKIPPED: the GPU routes. This machine has no GPU adapter.");
        let _ = std::fs::remove_dir_all(&dir);
        return;
    }
    assert!(
        message.contains("[nn] means: input downloaded, CPU takes host memory"),
        "{message}"
    );
    assert_same_rows(&host, &downloaded, "downloaded against the CPU");
    assert_eq!(
        downloaded_bytes(&message),
        tensor * MEANS_FRAMES as u64,
        "each frame's tensor is read back once, and nothing else: {message}"
    );

    // On the device and bound in place, for a CUDA session.
    let (bound, message) = means_run(
        &runtime,
        &dir,
        &input,
        "cuda",
        false,
        &format!("memory=gpu:{size}"),
        &profile,
    );
    if !message.contains("[nn] means: execution provider CUDA") {
        eprintln!("SKIPPED: the bound route. The CUDA provider did not engage here:\n{message}");
        let _ = std::fs::remove_dir_all(&dir);
        return;
    }
    if !cfg!(windows) {
        assert!(
            message.contains("[nn] means: input downloaded, CUDA takes host memory here"),
            "{message}"
        );
        let _ = std::fs::remove_dir_all(&dir);
        return;
    }
    assert!(
        message.contains("[nn] means: input from the pool, bound on CUDA"),
        "a CUDA session binds the pool's memory on Windows: {message}"
    );
    assert_same_rows(&host, &bound, "bound against the CPU");
    assert_eq!(
        downloaded_bytes(&message),
        0,
        "the bound route reads nothing back: {message}"
    );

    // NHWC, letterboxed, BGR, scaled: the device pass and the CPU agree on
    // every option, through the other graph.
    let options = format!("{size}:layout=nhwc:fit=letterbox:order=bgr:scale=0.5");
    let (host, _) = means_run(
        &runtime,
        &dir,
        &input,
        "cpu",
        true,
        &format!("memory=host:{options}"),
        &[],
    );
    let (bound, message) = means_run(
        &runtime,
        &dir,
        &input,
        "cuda",
        true,
        &format!("memory=gpu:{options}"),
        &[],
    );
    assert!(
        message.contains("[nn] means: input from the pool, bound on CUDA"),
        "{message}"
    );
    assert_same_rows(&host, &bound, "NHWC, letterboxed, bound against the CPU");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn pictures_decoded_on_the_device_are_bound_to_cuda_and_never_read_back() {
    let Some(runtime) = runtime_dir() else {
        announce_skip("nn_frames cannot run a model without a runtime");
        return;
    };
    if !cfg!(windows) {
        eprintln!("SKIPPED: a decoder shares the process's device on Windows (D3D12) only.");
        return;
    }
    let Some(av) = std::env::var_os("FFRWD_AV_RUNTIME")
        .map(PathBuf::from)
        .filter(|d| d.join("avcodec-63.dll").is_file())
    else {
        eprintln!(
            "SKIPPED: decoded pictures into a model. Point FFRWD_AV_RUNTIME at a directory \
             holding FFmpeg 9.0's shared libraries to run it."
        );
        return;
    };
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/h264.nut");
    let module = frames_module();
    let module = module.to_string_lossy().to_string();
    let dir = means_dir("decoded");
    let run = |target: &str| {
        let rows = dir.join(format!("rows-{target}.ndjson"));
        let output = Command::new(env!("CARGO_BIN_EXE_ffrwd-wasm"))
            .env_remove("FFRWD_NN_RUNTIME")
            .env("FFRWD_AV_RUNTIME", &av)
            .args(["-profile", "-nn-runtime"])
            .arg(&runtime)
            .args(["-nn", &format!("means={}", means_model(false).display())])
            .args(["-nn-target", target, "-gpu", &module, "-f", "nut", "-i"])
            .arg(&fixture)
            .args(["-m", &format!("nn_frames={module}"), "-filter_complex"])
            .arg(
                "[0:v]decode=format=yuv420p:hw=d3d12va[d];\
                 [v=d]nn_frames=model=means:memory=gpu:width=32:height=24[@rows=r]",
            )
            .args(["-map", "[r]", "-f", "ndjson"])
            .arg(&rows)
            .output()
            .expect("spawn ffrwd-wasm");
        let message = stderr(&output);
        assert!(
            output.status.success() || message.contains("no GPU"),
            "{target}: {message}"
        );
        let rows: Vec<serde_json::Value> = std::fs::read_to_string(&rows)
            .unwrap_or_default()
            .lines()
            .map(|line| serde_json::from_str(line).expect("a row is JSON"))
            .collect();
        (rows, message)
    };
    let (downloaded, message) = run("cpu");
    if message.contains("no GPU") {
        eprintln!("SKIPPED: this machine has no GPU adapter.");
        return;
    }
    assert!(
        message.contains("[gpu] decode h264: d3d12va, frames "),
        "the pictures stay on the device: {message}"
    );
    assert!(
        message.contains("[nn] means: input downloaded, CPU takes host memory"),
        "{message}"
    );
    let frames = downloaded.len() as u64;
    assert!(frames > 0, "{message}");
    assert_eq!(
        downloaded_bytes(&message),
        frames * 32 * 24 * 3 * 4,
        "the tensor is read back, never the picture: {message}"
    );
    let (bound, message) = run("cuda");
    if !message.contains("[nn] means: execution provider CUDA") {
        eprintln!("SKIPPED: the bound route. The CUDA provider did not engage here:\n{message}");
        return;
    }
    assert!(
        message.contains("[nn] means: input from the pool, bound on CUDA"),
        "{message}"
    );
    assert_eq!(downloaded_bytes(&message), 0, "{message}");
    assert_eq!(bound.len(), downloaded.len());
    let worst = apart(&downloaded, &bound);
    assert!(worst < 1e-4, "apart by up to {worst}");
    let _ = std::fs::remove_dir_all(&dir);
}
