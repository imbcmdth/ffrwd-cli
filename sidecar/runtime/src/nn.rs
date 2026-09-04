//! `wasi:nn` for components that import it: ONNX models the host loads and
//! the guest names.
//!
//! A module never sees a model file. The host reads the bytes, builds the
//! session, and registers it under a name; the guest calls `load-by-name` and
//! gets a graph back, so a module doing inference still has no filesystem
//! access at all.
//!
//! ONNX Runtime is loaded at run time from a directory the caller names. The
//! sidecar has no link-time dependency on it, and a checkout without one
//! builds and runs everything that does not ask for inference.

pub mod dlls;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

use anyhow::{bail, Context, Result};
use wasmtime_wasi_nn::backend::onnx::{OnnxBackend, Provider};
use wasmtime_wasi_nn::wit::{ExecutionTarget, WasiNnCtx};
use wasmtime_wasi_nn::{Backend, Graph, GraphRegistry, Registry};

/// The environment variable naming the ONNX Runtime directory when the
/// command line does not.
pub const RUNTIME_DIR_VAR: &str = "FFRWD_NN_RUNTIME";

/// The environment variable naming the target when the command line does not.
pub const TARGET_VAR: &str = "FFRWD_NN_TARGET";

/// Where the runtime comes from, named in every refusal that wants one.
const FETCH_HINT: &str = "run `ffrwd setup nn` to download one";

/// The interface namespace a component imports to ask for inference.
pub const IMPORT_PREFIX: &str = "wasi:nn/";

/// The valid `-nn-target` spellings, listed in every refusal of one that is
/// not.
const TARGETS: &str = "the targets are cpu, gpu, cuda, directml and coreml";

/// The valid `-nn-exclude` spellings, listed in every refusal of one that is
/// not. `cpu` is not among them -- every model runs somewhere, and CPU is
/// the one provider ONNX Runtime always offers.
const EXCLUDABLE: &str = "the providers -nn-exclude can name are cuda, directml and coreml";

/// Where a session runs. `Gpu` is whichever provider this platform and this
/// runtime directory can actually offer; the rest name one outright.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Target {
    #[default]
    Cpu,
    Gpu,
    Cuda,
    DirectMl,
    CoreMl,
}

impl Target {
    /// The spellings `-nn-target` accepts, and `None` for anything else.
    fn spell(raw: &str) -> Option<Target> {
        match raw {
            "cpu" => Some(Target::Cpu),
            "gpu" => Some(Target::Gpu),
            "cuda" => Some(Target::Cuda),
            "directml" => Some(Target::DirectMl),
            "coreml" => Some(Target::CoreMl),
            _ => None,
        }
    }

    /// The spelling `-nn-target` accepts.
    pub fn parse(raw: &str) -> Result<Target> {
        match Target::spell(raw) {
            Some(target) => Ok(target),
            None => bail!("-nn-target {raw}: {TARGETS}"),
        }
    }

    /// The target the environment names, for a run whose argv does not. An
    /// unset variable is no answer; a set one that is not a target is a
    /// refusal, since a misspelling would otherwise run on the CPU in silence.
    pub fn from_env() -> Result<Option<Target>> {
        let Some(raw) = std::env::var_os(TARGET_VAR) else {
            return Ok(None);
        };
        let raw = raw.to_string_lossy().to_string();
        match Target::spell(&raw) {
            Some(target) => Ok(Some(target)),
            None => bail!("{TARGET_VAR}={raw}: {TARGETS}"),
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Target::Cpu => "cpu",
            Target::Gpu => "gpu",
            Target::Cuda => "cuda",
            Target::DirectMl => "directml",
            Target::CoreMl => "coreml",
        }
    }

    /// Anything but `cpu` is a GPU session as far as the backend is concerned;
    /// which provider it asks for is the resolved list, not this.
    fn execution(self) -> ExecutionTarget {
        match self {
            Target::Cpu => ExecutionTarget::Cpu,
            _ => ExecutionTarget::Gpu,
        }
    }
}

/// The provider `-nn-exclude` names. `cpu` and anything not a provider at
/// all are refused: a process always keeps somewhere to run, and a
/// misspelling would otherwise be silently ignored.
pub fn parse_exclude(raw: &str) -> Result<Provider> {
    match raw {
        "cuda" => Ok(Provider::Cuda),
        "directml" => Ok(Provider::DirectMl),
        "coreml" => Ok(Provider::CoreMl),
        "cpu" => bail!("-nn-exclude cpu: cpu cannot be excluded"),
        _ => bail!("-nn-exclude {raw}: {EXCLUDABLE}"),
    }
}

/// What `gpu` means here: the providers this platform can offer, most wanted
/// first. The first whose preflight is clean is the one a session asks for.
///
/// Windows takes DirectML ahead of CUDA. DirectML runs on any Direct3D 12
/// adapter and asks for nothing installed, where CUDA needs a CUDA 12 runtime
/// and cuDNN 9 on the machine; the measured difference between them is small
/// enough that the one that works everywhere goes first.
#[cfg(windows)]
const GPU_PRIORITY: &[Provider] = &[Provider::DirectMl, Provider::Cuda];
#[cfg(target_os = "macos")]
const GPU_PRIORITY: &[Provider] = &[Provider::CoreMl];
#[cfg(all(unix, not(target_os = "macos")))]
const GPU_PRIORITY: &[Provider] = &[Provider::Cuda];

/// The operating system and architecture a runtime directory is filed under.
/// The spellings are the ones ONNX Runtime names its own release archives
/// with, so a directory and the artifact that filled it read the same.
pub const fn platform() -> &'static str {
    #[cfg(all(windows, target_arch = "x86_64"))]
    {
        "win-x64"
    }
    #[cfg(all(windows, target_arch = "aarch64"))]
    {
        "win-arm64"
    }
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        "osx-arm64"
    }
    #[cfg(all(target_os = "macos", target_arch = "x86_64"))]
    {
        "osx-x86_64"
    }
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    {
        "linux-x64"
    }
    #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
    {
        "linux-aarch64"
    }
}

/// The ONNX Runtime version this build demands.
///
/// `ort` refuses a library whose version string does not read `1.<minor>.x`,
/// naming both, so this is the one number a fetch has to match. It is read off
/// the crate rather than written down, which is what keeps a bootstrap that
/// asks for it from drifting past a crate upgrade.
pub fn ort_version() -> String {
    format!("1.{}.0", ort::MINOR_VERSION)
}

/// The providers `-nn-target gpu` walks, most wanted first, with the CPU it
/// falls back to last. The spellings are `-nn-target`'s own.
pub fn providers() -> Vec<String> {
    GPU_PRIORITY
        .iter()
        .map(|p| p.name().to_ascii_lowercase())
        .chain(std::iter::once("cpu".to_string()))
        .collect()
}

/// What `--nn-info` prints: enough for a caller to fetch the right runtime for
/// this build on this machine without knowing anything else about either.
pub fn info() -> String {
    let providers: Vec<String> = providers()
        .into_iter()
        .map(|name| format!("\"{name}\""))
        .collect();
    format!(
        "{{\"ort_version\":\"{}\",\"providers\":[{}],\"platform\":\"{}\"}}",
        ort_version(),
        providers.join(","),
        platform()
    )
}

/// One `-nn` binding: the name a guest asks for, and the file behind it.
#[derive(Clone, Debug)]
pub struct ModelSpec {
    pub name: String,
    pub path: PathBuf,
}

/// What the command line said about inference.
#[derive(Clone, Debug, Default)]
pub struct Config {
    pub models: Vec<ModelSpec>,
    pub runtime_dir: Option<PathBuf>,
    pub target: Target,
    /// Providers a `-nn-target gpu` walk skips for this process, the union
    /// of every bound model's own pin. Ignored for a target named outright.
    pub exclude: Vec<Provider>,
}

/// A name to ONNX graph map.
///
/// The crate ships `InMemoryRegistry`, but it takes a directory whose basename
/// becomes the registered name and whose contents must include a file called
/// `model.onnx`, and it hardcodes the CPU target. None of that expresses
/// `-nn name=path`, and it makes a GPU-resident preloaded graph impossible.
/// `GraphRegistry` is a public trait, so a few lines of our own replace it.
#[derive(Clone, Default)]
struct ModelRegistry(Arc<HashMap<String, Graph>>);

impl ModelRegistry {
    /// Reads the file here in the host and hands the bytes to the backend.
    /// The guest learns neither the path nor the bytes.
    fn register(
        &mut self,
        backend: &mut Backend,
        name: &str,
        path: &Path,
        target: ExecutionTarget,
    ) -> Result<()> {
        let bytes = std::fs::read(path)
            .with_context(|| format!("reading the model file {}", path.display()))?;
        let graph = backend
            .load(&[&bytes], target)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        Arc::get_mut(&mut self.0)
            .context("the model registry is already shared")?
            .insert(name.to_string(), graph);
        Ok(())
    }
}

impl GraphRegistry for ModelRegistry {
    fn get(&self, name: &str) -> Option<&Graph> {
        self.0.get(name)
    }

    fn get_mut(&mut self, _name: &str) -> Option<&mut Graph> {
        // A loaded graph is immutable; nothing on the wit path asks for this.
        None
    }
}

/// The models this process loaded, and the provider they were loaded for.
struct Models {
    registry: ModelRegistry,
    names: Vec<String>,
    provider: Provider,
}

/// Loaded once per process: building a session is expensive and `Graph` is
/// reference-counted, so every store shares the one session. `Backend` is not
/// shared, and is rebuilt for each store.
fn graph_registry() -> &'static OnceLock<Models> {
    static MODELS: OnceLock<Models> = OnceLock::new();
    &MODELS
}

/// Whether any model is registered, and so whether a component that imports
/// `wasi:nn` can be run at all.
pub fn is_configured() -> bool {
    graph_registry().get().is_some()
}

/// The names bound by `-nn`, for messages listing what a guest could have
/// asked for.
pub fn model_names() -> Vec<String> {
    graph_registry()
        .get()
        .map(|m| m.names.clone())
        .unwrap_or_default()
}

/// Refuses, before instantiation, a component that asks for inference in a
/// run where no model is registered.
pub fn require_configured(module_path: &str) -> Result<()> {
    if is_configured() {
        return Ok(());
    }
    bail!(
        "{module_path} imports wasi:nn, and this run registers no model; \
         bind one with -nn name=path"
    );
}

/// The ONNX backend, told which provider a `gpu` session should ask for. The
/// crate's own `backend::list()` builds one that always means CUDA.
fn onnx_backend(provider: Provider) -> Backend {
    Backend::from(OnnxBackend::with_gpu_providers(vec![provider]))
}

/// The wasi-nn context one store gets. Backends are consumed by the context,
/// so each store builds its own; the graphs behind them are shared.
pub fn store_ctx() -> WasiNnCtx {
    match graph_registry().get() {
        Some(models) => WasiNnCtx::new(
            vec![onnx_backend(models.provider)],
            Registry::from(models.registry.clone()),
        ),
        None => empty_ctx(),
    }
}

/// A context with no backends and no graphs, for the stores of the modules
/// that never asked for inference.
pub fn empty_ctx() -> WasiNnCtx {
    WasiNnCtx::new(
        Vec::<Backend>::new(),
        Registry::from(ModelRegistry::default()),
    )
}

/// Where the ONNX Runtime directory comes from: the command line, else the
/// environment. Absent is a refusal that says what to fetch.
fn resolve_runtime_dir(config: &Config) -> Result<PathBuf> {
    let dir = config
        .runtime_dir
        .clone()
        .or_else(|| std::env::var_os(RUNTIME_DIR_VAR).map(PathBuf::from));
    let Some(dir) = dir else {
        bail!(
            "-nn needs an ONNX Runtime: name its directory with -nn-runtime <dir> \
             or set {RUNTIME_DIR_VAR}, then {FETCH_HINT}"
        );
    };
    if !dir.is_dir() {
        bail!(
            "-nn-runtime {}: not a directory; {FETCH_HINT}",
            dir.display()
        );
    }
    Ok(dir)
}

/// Loads the runtime library named by full path, having first checked it is
/// there: `ort` panics from inside its own initialisation when the library
/// will not load, and a panic is not a message anyone can act on.
fn open_runtime(dir: &Path) -> Result<PathBuf> {
    let lib = dir.join(dlls::RUNTIME_LIB);
    if !lib.is_file() {
        bail!(
            "no {} in {}; {FETCH_HINT}",
            dlls::RUNTIME_LIB,
            dir.display()
        );
    }
    // Never a bare name: Windows ships an onnxruntime of its own in System32
    // and it is the one a bare name finds.
    dlls::add_search_dir(dir)
        .with_context(|| format!("adding {} to the library search path", dir.display()))?;
    if let Err(e) = unsafe { libloading::Library::new(&lib) } {
        bail!("{} could not be loaded: {e}", lib.display());
    }
    Ok(lib)
}

/// Resolves what the CUDA provider needs before a session is created, so a
/// missing prerequisite is named rather than discovered as a silent fall back
/// to the CPU. Every directory that answered is added to the search path.
/// Returns the complaints, which are warnings: the run continues on the CPU.
fn preflight_cuda(dir: &Path) -> Vec<String> {
    let mut complaints = Vec::new();
    let dirs = dlls::search_dirs(dir);

    for lib in dlls::PROVIDER_LIBS {
        if !dir.join(lib).is_file() {
            complaints.push(format!(
                "{lib} is not in {}; `ffrwd setup nn --cuda` adds it",
                dir.display()
            ));
        }
    }

    for dep in dlls::CUDA_DEPS {
        match dlls::locate(&dirs, dep.file) {
            Some(found) => {
                let _ = dlls::add_search_dir(&found);
                if dep.file.starts_with("cudnn64") || dep.file.starts_with("libcudnn.so") {
                    let absent: Vec<&str> = dlls::CUDNN_SUBLIBS
                        .iter()
                        .copied()
                        .filter(|sub| !found.join(sub).is_file())
                        .collect();
                    if !absent.is_empty() {
                        complaints.push(format!(
                            "{} has {} but not {}",
                            found.display(),
                            dep.file,
                            absent.join(", ")
                        ));
                    }
                }
            }
            None => {
                let others = dlls::siblings(&dirs, dep);
                let instead = match others.first() {
                    Some((name, where_)) => format!("; {name} is in {}", where_.display()),
                    None => String::new(),
                };
                complaints.push(format!("{} was not found{instead}", dep.file));
            }
        }
    }

    if !complaints.is_empty() {
        complaints.push(format!(
            "the CUDA provider needs a CUDA 12 runtime and cuDNN 9. Install them, \
             or run `ffrwd setup nn --cuda --full` to put a pinned set in {}",
            dir.display()
        ));
    }
    complaints
}

/// Resolves what DirectML needs: its own build of ONNX Runtime, DirectML
/// itself, and a Direct3D 12 device to run on.
fn preflight_directml(dir: &Path) -> Vec<String> {
    let mut complaints = Vec::new();
    let dml = dir.join(dlls::DIRECTML_SUBDIR);

    if !dml.join(dlls::RUNTIME_LIB).is_file() {
        complaints.push(format!(
            "no {} in {}; `ffrwd setup nn` puts the DirectML build there",
            dlls::RUNTIME_LIB,
            dml.display()
        ));
    }

    // The pinned DirectML beside that runtime, else the one Windows ships;
    // the pinned one is found first because its directory is added by hand.
    if dlls::locate(&dlls::search_dirs(&dml), dlls::DIRECTML_LIB).is_none() {
        complaints.push(format!(
            "{} was not found in {} or on the search path; \
             `ffrwd setup nn` downloads it",
            dlls::DIRECTML_LIB,
            dml.display()
        ));
    }

    match dlls::d3d12_available() {
        Some(true) => {}
        Some(false) => {
            complaints.push("no Direct3D 12 device answered; DirectML needs one".to_string())
        }
        None => {
            complaints.push("d3d12.dll could not be loaded, so DirectML has no device".to_string())
        }
    }

    complaints
}

/// CoreML is part of the operating system, so there is nothing to fetch and
/// nothing to resolve - only somewhere to be.
fn preflight_coreml() -> Vec<String> {
    if cfg!(target_os = "macos") {
        Vec::new()
    } else {
        vec!["CoreML is macOS'; this is not macOS".to_string()]
    }
}

/// What a provider needs before a session is built, so a missing prerequisite
/// is named rather than discovered as a silent fall back to the CPU.
fn preflight(provider: Provider, dir: &Path) -> Vec<String> {
    match provider {
        Provider::Cpu => Vec::new(),
        Provider::Cuda => preflight_cuda(dir),
        Provider::DirectMl => preflight_directml(dir),
        Provider::CoreMl => preflight_coreml(),
    }
}

/// The provider a run asks for, and what to say on the way there.
///
/// `gpu` walks the platform's priority order and takes the first whose
/// preflight is clean, saying in one line why it passed over each earlier one.
/// `exclude` drops a candidate from that walk before its preflight ever
/// runs -- a model's own pin, not this machine, ruled it out. A provider
/// named outright is taken as named regardless of `exclude`: its complaints
/// are warnings, and ONNX Runtime's own fall back to the CPU is what the
/// verdict reports.
fn resolve(target: Target, dir: &Path, exclude: &[Provider]) -> (Provider, Vec<String>) {
    match target {
        Target::Cpu => (Provider::Cpu, Vec::new()),
        Target::Cuda => (Provider::Cuda, preflight(Provider::Cuda, dir)),
        Target::DirectMl => (Provider::DirectMl, preflight(Provider::DirectMl, dir)),
        Target::CoreMl => (Provider::CoreMl, preflight(Provider::CoreMl, dir)),
        Target::Gpu => {
            let mut notes = Vec::new();
            for &candidate in GPU_PRIORITY {
                if exclude.contains(&candidate) {
                    notes.push(format!(
                        "-nn-target gpu: not {}, excluded by -nn-exclude",
                        candidate.name()
                    ));
                    continue;
                }
                let complaints = preflight(candidate, dir);
                let Some(first) = complaints.first() else {
                    return (candidate, notes);
                };
                notes.push(format!("-nn-target gpu: not {}, {first}", candidate.name()));
            }
            (Provider::Cpu, notes)
        }
    }
}

/// Which ONNX Runtime answers for a provider. DirectML's is a different binary
/// carrying the same name as the CPU and CUDA one, so it sits in its own
/// directory and only that directory is put on the search path. Absent, the
/// base directory answers and the session falls back to the CPU the way one
/// asking for a CUDA provider that is not there does.
fn runtime_lib_dir(dir: &Path, provider: Provider) -> PathBuf {
    if provider == Provider::DirectMl {
        let dml = dir.join(dlls::DIRECTML_SUBDIR);
        if dml.join(dlls::RUNTIME_LIB).is_file() {
            return dml;
        }
    }
    dir.to_path_buf()
}

/// Names the provider that engaged and every native library that answered,
/// with where each came from. A GPU session that quietly became a CPU one is
/// otherwise indistinguishable from a slow GPU.
fn report_verdict(dir: &Path, config: &Config, asked: Provider) {
    // Named once regardless of the walk's outcome, so `-nn-exclude` is never
    // a silent input.
    if !config.exclude.is_empty() {
        let names: Vec<String> = config
            .exclude
            .iter()
            .map(|p| p.name().to_ascii_lowercase())
            .collect();
        eprintln!("[nn] -nn-exclude: {}", names.join(", "));
    }

    let engaged = dlls::engaged_provider();
    let provider = match engaged {
        Some(provider) => provider.name(),
        None => "unknown on this platform",
    };
    eprintln!("[nn] execution provider: {provider}");

    if let Some(paths) = dlls::loaded() {
        for path in paths.iter().filter(|p| dlls::interesting(p)) {
            eprintln!(
                "[nn]   {} ({})",
                path.display(),
                dlls::origin(path, dir).label()
            );
        }
    }

    if config.target != Target::Cpu && engaged == Some(Provider::Cpu) {
        let names: Vec<&str> = config.models.iter().map(|m| m.name.as_str()).collect();
        let why = if asked == Provider::Cpu {
            "no GPU execution provider was available".to_string()
        } else {
            format!("the {} execution provider did not load", asked.name())
        };
        eprintln!(
            "[nn] -nn-target {} was requested but {why}; {} {} on the CPU",
            config.target.label(),
            names.join(", "),
            if names.len() == 1 { "runs" } else { "run" }
        );
    }

    // The walk landed on CPU with every GPU candidate excluded by a bound
    // model's own pin, rather than by a preflight failure: named once at
    // startup, so a user knows why a run that binds no `-nn-target` of its
    // own is slow.
    if config.target == Target::Gpu && asked == Provider::Cpu && !config.exclude.is_empty() {
        let excluded: Vec<String> = GPU_PRIORITY
            .iter()
            .filter(|candidate| config.exclude.contains(candidate))
            .map(|candidate| candidate.name().to_ascii_lowercase())
            .collect();
        if !excluded.is_empty() {
            let names: Vec<&str> = config.models.iter().map(|m| m.name.as_str()).collect();
            eprintln!(
                "[nn] {} {} on cpu: {} excluded by its pin",
                names.join(", "),
                if names.len() == 1 { "runs" } else { "run" },
                excluded.join(", ")
            );
        }
    }
}

/// Loads every `-nn` model, once, before any component is instantiated.
///
/// A run that binds no model does nothing here and touches no ONNX Runtime,
/// which is what lets a checkout without one run everything else.
pub fn configure(config: &Config) -> Result<()> {
    if config.models.is_empty() {
        return Ok(());
    }

    let dir = resolve_runtime_dir(config)?;

    // Which provider before which runtime: DirectML's ONNX Runtime is a
    // different binary, so the library to load follows from the answer.
    let (provider, complaints) = resolve(config.target, &dir, &config.exclude);
    for complaint in complaints {
        eprintln!("[nn] {complaint}");
    }
    let lib = open_runtime(&runtime_lib_dir(&dir, provider))?;

    ort::init_from(lib.to_string_lossy().to_string())
        .commit()
        .map_err(|e| anyhow::anyhow!("initialising ONNX Runtime from {}: {e}", lib.display()))?;

    let mut onnx = onnx_backend(provider);

    let mut registry = ModelRegistry::default();
    let mut names = Vec::new();
    for spec in &config.models {
        registry
            .register(&mut onnx, &spec.name, &spec.path, config.target.execution())
            .with_context(|| {
                format!(
                    "-nn {}={}: loading the model",
                    spec.name,
                    spec.path.display()
                )
            })?;
        names.push(spec.name.clone());
    }

    report_verdict(&dir, config, provider);

    graph_registry()
        .set(Models {
            registry,
            names,
            provider,
        })
        .map_err(|_| anyhow::anyhow!("the models were already loaded"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_exclude_accepts_the_three_deniable_providers() {
        assert_eq!(parse_exclude("cuda").unwrap(), Provider::Cuda);
        assert_eq!(parse_exclude("directml").unwrap(), Provider::DirectMl);
        assert_eq!(parse_exclude("coreml").unwrap(), Provider::CoreMl);
    }

    #[test]
    fn parse_exclude_refuses_cpu() {
        let err = parse_exclude("cpu").unwrap_err().to_string();
        assert!(err.contains("cannot be excluded"), "{err}");
    }

    #[test]
    fn parse_exclude_refuses_a_misspelling() {
        let err = parse_exclude("dml").unwrap_err().to_string();
        assert!(err.contains(EXCLUDABLE), "{err}");
    }

    /// An empty directory, so every real provider's own preflight fails on
    /// its own -- what is under test is that an excluded candidate is
    /// skipped before its preflight ever runs, named as the reason rather
    /// than folded into whatever the preflight would have said.
    fn empty_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("ffrwd-nn-test-{name}"));
        let _ = std::fs::create_dir_all(&dir);
        dir
    }

    #[test]
    fn the_gpu_walk_skips_an_excluded_candidate_before_its_preflight() {
        // The platform's own first candidate: DirectML on Windows, CUDA on
        // Linux, CoreML on macOS.
        let first = GPU_PRIORITY[0];
        let dir = empty_dir("exclude-one");
        let (provider, notes) = resolve(Target::Gpu, &dir, &[first]);
        assert_eq!(provider, Provider::Cpu);
        let named = format!("not {}", format!("{first:?}").to_uppercase());
        assert!(
            notes
                .iter()
                .any(|n| n.contains(&named) && n.contains("excluded by -nn-exclude")),
            "{notes:?}"
        );
    }

    #[test]
    fn the_gpu_walk_with_every_candidate_excluded_lands_on_cpu() {
        let dir = empty_dir("exclude-all");
        let exclude = GPU_PRIORITY.to_vec();
        let (provider, notes) = resolve(Target::Gpu, &dir, &exclude);
        assert_eq!(provider, Provider::Cpu);
        for candidate in GPU_PRIORITY {
            assert!(
                notes
                    .iter()
                    .any(|n| n.contains(candidate.name()) && n.contains("-nn-exclude")),
                "{candidate:?} should be named as excluded: {notes:?}"
            );
        }
    }

    #[test]
    fn excluding_nothing_leaves_the_walk_exactly_as_it_was() {
        let dir = empty_dir("exclude-none");
        let (with_none, notes_a) = resolve(Target::Gpu, &dir, &[]);
        let (without_arg, notes_b) = resolve(Target::Gpu, &dir, &Vec::new());
        assert_eq!(with_none, without_arg);
        assert_eq!(notes_a, notes_b);
    }

    #[test]
    fn a_named_target_ignores_exclude() {
        // Naming a provider outright is unaffected by `-nn-exclude`: only the
        // `gpu` walk consults it.
        let dir = empty_dir("named-target");
        let (provider, _) = resolve(Target::DirectMl, &dir, &[Provider::DirectMl]);
        assert_eq!(provider, Provider::DirectMl);
    }
}
