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

/// The providers `-nn-exclude` can name, listed in every refusal of one that
/// is not. `cpu` is not among them -- every model runs somewhere, and CPU is
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
}

/// How the argv spells a provider: `-nn-target`'s and `-nn-exclude`'s word
/// for it.
fn spelling(provider: Provider) -> String {
    provider.name().to_ascii_lowercase()
}

/// One `-nn-exclude`: a provider one model's session never asks for, or
/// every model's when it names no model.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Exclusion {
    /// The `-nn` name it applies to; `None` is every model.
    pub model: Option<String>,
    pub provider: Provider,
}

impl Exclusion {
    /// Whether this keeps `provider` from the model bound as `name`.
    fn denies(&self, name: &str, provider: Provider) -> bool {
        self.provider == provider && self.model.as_deref().is_none_or(|model| model == name)
    }
}

/// Spelled the way `-nn-exclude` took it.
impl std::fmt::Display for Exclusion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.model {
            Some(name) => write!(f, "{name}={}", spelling(self.provider)),
            None => write!(f, "{}", spelling(self.provider)),
        }
    }
}

/// What one `-nn-exclude` says: `<name>=<provider>` for the model bound as
/// `name`, or a bare `<provider>` for every model. `cpu` and anything not a
/// provider at all are refused: a model always keeps somewhere to run, and a
/// misspelling would otherwise be silently ignored. Whether `name` is bound
/// is [`Config::check_exclude`]'s to say, once every `-nn` has been read.
pub fn parse_exclude(raw: &str) -> Result<Exclusion> {
    let (model, spelled) = match raw.split_once('=') {
        Some((name, provider)) => (Some(name.to_string()), provider),
        None => (None, raw),
    };
    let provider = match spelled {
        "cuda" => Provider::Cuda,
        "directml" => Provider::DirectMl,
        "coreml" => Provider::CoreMl,
        "cpu" => bail!("-nn-exclude {raw}: cpu cannot be excluded"),
        _ => bail!("-nn-exclude {raw}: {EXCLUDABLE}"),
    };
    Ok(Exclusion { model, provider })
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
        .map(|&p| spelling(p))
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
    /// What `-nn-exclude` said, in the order it said it. A `-nn-target gpu`
    /// walk passes over a provider every bound model excludes, and a model
    /// that excludes the provider the walk took runs on the CPU. Ignored for
    /// a target named outright.
    pub exclude: Vec<Exclusion>,
}

impl Config {
    /// Refuses a `-nn-exclude <name>=<provider>` whose name no `-nn` binds,
    /// since a misspelt name would otherwise exclude nothing in silence.
    pub fn check_exclude(&self) -> Result<()> {
        for exclusion in &self.exclude {
            let Some(name) = &exclusion.model else {
                continue;
            };
            if !self.models.iter().any(|m| &m.name == name) {
                bail!("-nn-exclude {exclusion}: no -nn binds {name}");
            }
        }
        Ok(())
    }

    /// Whether the model bound as `name` excludes `provider`, by its own
    /// exclusion or a bare one.
    fn excludes(&self, name: &str, provider: Provider) -> bool {
        self.exclude.iter().any(|e| e.denies(name, provider))
    }

    /// The candidates of a `gpu` walk that every bound model excludes, and
    /// so that the walk passes over.
    fn excluded_by_every_model(&self) -> Vec<Provider> {
        GPU_PRIORITY
            .iter()
            .copied()
            .filter(|&p| self.models.iter().all(|m| self.excludes(&m.name, p)))
            .collect()
    }
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
    /// Hands a model file's bytes, read here in the host, to the backend.
    /// The guest learns neither the path nor the bytes.
    fn register(
        &mut self,
        backend: &mut Backend,
        name: &str,
        bytes: &[u8],
        target: ExecutionTarget,
    ) -> Result<()> {
        // The backend's own message names only the kind of failure; what
        // ONNX Runtime said is its cause, kept so a refusal carries it.
        let graph = backend.load(&[bytes], target).map_err(anyhow::Error::new)?;
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

/// The models this process loaded, and the provider the walk took for it.
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

/// The provider this process loads a runtime for, and what to say on the way
/// there.
///
/// One process holds one ONNX Runtime binary, and DirectML's is a different
/// binary from the CPU and CUDA one, so with the pinned artifacts a DirectML
/// session and a CUDA one cannot share a process. The walk therefore settles
/// one provider for the whole process, and each model's session then asks
/// for it or for the CPU: a model that excludes the provider the walk took
/// runs on the CPU.
///
/// `gpu` walks the platform's priority order and takes the first candidate
/// whose preflight is clean, saying in one line why it passed over each
/// earlier one. `excluded` is the candidates every bound model excludes,
/// dropped before their preflight ever runs -- the models' own pins, not
/// this machine, ruled them out. A provider named outright is taken as named
/// and nothing is excluded: its complaints are warnings, and a session that
/// cannot engage it runs on the CPU and says so.
fn resolve(target: Target, dir: &Path, excluded: &[Provider]) -> (Provider, Vec<String>) {
    match target {
        Target::Cpu => (Provider::Cpu, Vec::new()),
        Target::Cuda => (Provider::Cuda, preflight(Provider::Cuda, dir)),
        Target::DirectMl => (Provider::DirectMl, preflight(Provider::DirectMl, dir)),
        Target::CoreMl => (Provider::CoreMl, preflight(Provider::CoreMl, dir)),
        Target::Gpu => {
            let mut notes = Vec::new();
            for &candidate in GPU_PRIORITY {
                if excluded.contains(&candidate) {
                    notes.push(format!(
                        "-nn-target gpu: not {}, excluded by every model's pin",
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
/// base directory answers, and a session asking it for DirectML does not
/// engage and runs on the CPU, the way one asking for a CUDA provider that is
/// not there does.
fn runtime_lib_dir(dir: &Path, provider: Provider) -> PathBuf {
    if provider == Provider::DirectMl {
        let dml = dir.join(dlls::DIRECTML_SUBDIR);
        if dml.join(dlls::RUNTIME_LIB).is_file() {
            return dml;
        }
    }
    dir.to_path_buf()
}

/// What one model's session asks for once the walk has settled the process's
/// provider.
#[derive(Debug, PartialEq, Eq)]
enum Ask {
    /// That provider; a session that cannot engage it runs on the CPU.
    Gpu(Provider),
    /// The CPU, and why when a GPU was asked for.
    Cpu(Option<String>),
}

/// The provider the walk took goes to every model that does not exclude it,
/// and the CPU to every model that does. A provider named outright goes to
/// every model, exclusions or not.
fn ask(config: &Config, provider: Provider, name: &str) -> Ask {
    if config.target == Target::Cpu {
        return Ask::Cpu(None);
    }
    if provider == Provider::Cpu {
        // Only a `gpu` walk lands here: a provider named outright is taken.
        let pinned: Vec<&str> = GPU_PRIORITY
            .iter()
            .filter(|&&p| config.excludes(name, p))
            .map(|p| p.name())
            .collect();
        let why = if pinned.len() == GPU_PRIORITY.len() {
            format!("{} excluded by its pin", pinned.join(", "))
        } else {
            "no GPU execution provider was available".to_string()
        };
        return Ask::Cpu(Some(why));
    }
    if config.target == Target::Gpu && config.excludes(name, provider) {
        return Ask::Cpu(Some(format!("{} excluded by its pin", provider.name())));
    }
    Ask::Gpu(provider)
}

/// Where one model's session runs, and why it is the CPU when a GPU was
/// asked for.
struct Placement {
    name: String,
    provider: Provider,
    why_cpu: Option<String>,
}

/// An error that may span several lines, as one.
fn one_line(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Names the provider each model's session runs on and every native library
/// that answered, with where each came from. A GPU session that quietly
/// became a CPU one is otherwise indistinguishable from a slow GPU.
fn report_verdict(dir: &Path, config: &Config, placements: &[Placement]) {
    // Named once regardless of the walk's outcome, so `-nn-exclude` is never
    // a silent input.
    if !config.exclude.is_empty() {
        let named: Vec<String> = config.exclude.iter().map(|e| e.to_string()).collect();
        eprintln!("[nn] -nn-exclude: {}", named.join(", "));
    }

    for placement in placements {
        match &placement.why_cpu {
            Some(why) => eprintln!(
                "[nn] {}: execution provider {}, {why}",
                placement.name,
                placement.provider.name()
            ),
            None => eprintln!(
                "[nn] {}: execution provider {}",
                placement.name,
                placement.provider.name()
            ),
        }
    }

    if let Some(paths) = dlls::loaded() {
        for path in paths.iter().filter(|p| dlls::interesting(p)) {
            eprintln!(
                "[nn]   {} ({})",
                path.display(),
                dlls::origin(path, dir).label()
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
    config.check_exclude()?;

    let dir = resolve_runtime_dir(config)?;

    // Which provider before which runtime: DirectML's ONNX Runtime is a
    // different binary, so the library to load follows from the answer.
    let excluded = match config.target {
        Target::Gpu => config.excluded_by_every_model(),
        _ => Vec::new(),
    };
    let (provider, complaints) = resolve(config.target, &dir, &excluded);
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
    let mut placements = Vec::new();
    for spec in &config.models {
        let loading = || {
            format!(
                "-nn {}={}: loading the model",
                spec.name,
                spec.path.display()
            )
        };
        let bytes = std::fs::read(&spec.path)
            .with_context(|| format!("reading the model file {}", spec.path.display()))
            .with_context(loading)?;

        let (placed, why_cpu) = match ask(config, provider, &spec.name) {
            Ask::Gpu(asked) => {
                match registry.register(&mut onnx, &spec.name, &bytes, ExecutionTarget::Gpu) {
                    Ok(()) => (asked, None),
                    // The session refused rather than falling back in silence;
                    // the CPU answers the same, and slower.
                    Err(e) => {
                        eprintln!(
                            "[nn] {}: {} did not engage ({}); runs on the CPU",
                            spec.name,
                            asked.name(),
                            one_line(&e.root_cause().to_string())
                        );
                        registry
                            .register(&mut onnx, &spec.name, &bytes, ExecutionTarget::Cpu)
                            .with_context(loading)?;
                        (
                            Provider::Cpu,
                            Some(format!("{} did not engage", asked.name())),
                        )
                    }
                }
            }
            Ask::Cpu(why) => {
                registry
                    .register(&mut onnx, &spec.name, &bytes, ExecutionTarget::Cpu)
                    .with_context(loading)?;
                (Provider::Cpu, why)
            }
        };
        placements.push(Placement {
            name: spec.name.clone(),
            provider: placed,
            why_cpu,
        });
        names.push(spec.name.clone());
    }

    report_verdict(&dir, config, &placements);

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

    /// A bare exclusion, for every model.
    fn every(provider: Provider) -> Exclusion {
        Exclusion {
            model: None,
            provider,
        }
    }

    /// One model's exclusion.
    fn one(name: &str, provider: Provider) -> Exclusion {
        Exclusion {
            model: Some(name.to_string()),
            provider,
        }
    }

    /// A `gpu` run binding these names, excluding these.
    fn config(names: &[&str], exclude: Vec<Exclusion>) -> Config {
        Config {
            models: names
                .iter()
                .map(|name| ModelSpec {
                    name: name.to_string(),
                    path: PathBuf::from(format!("{name}.onnx")),
                })
                .collect(),
            runtime_dir: None,
            target: Target::Gpu,
            exclude,
        }
    }

    #[test]
    fn parse_exclude_accepts_the_three_deniable_providers() {
        assert_eq!(parse_exclude("cuda").unwrap(), every(Provider::Cuda));
        assert_eq!(
            parse_exclude("directml").unwrap(),
            every(Provider::DirectMl)
        );
        assert_eq!(parse_exclude("coreml").unwrap(), every(Provider::CoreMl));
    }

    #[test]
    fn parse_exclude_takes_a_model_name_before_the_provider() {
        let exclusion = parse_exclude("depth=directml").unwrap();
        assert_eq!(exclusion, one("depth", Provider::DirectMl));
        assert_eq!(exclusion.to_string(), "depth=directml");
        assert_eq!(every(Provider::Cuda).to_string(), "cuda");
    }

    #[test]
    fn parse_exclude_refuses_cpu() {
        let err = parse_exclude("cpu").unwrap_err().to_string();
        assert!(err.contains("cannot be excluded"), "{err}");
        let err = parse_exclude("depth=cpu").unwrap_err().to_string();
        assert!(
            err.contains("depth=cpu") && err.contains("cannot be excluded"),
            "{err}"
        );
    }

    #[test]
    fn parse_exclude_refuses_a_misspelling() {
        let err = parse_exclude("dml").unwrap_err().to_string();
        assert!(err.contains(EXCLUDABLE), "{err}");
        let err = parse_exclude("depth=dml").unwrap_err().to_string();
        assert!(err.contains(EXCLUDABLE), "{err}");
    }

    #[test]
    fn an_exclusion_naming_a_model_no_nn_binds_is_refused() {
        let unbound = config(&["depth"], vec![one("dpeth", Provider::Cuda)]);
        let err = unbound.check_exclude().unwrap_err().to_string();
        assert!(
            err.contains("-nn-exclude dpeth=cuda") && err.contains("no -nn binds dpeth"),
            "{err}"
        );

        let bound = config(
            &["depth"],
            vec![one("depth", Provider::Cuda), every(Provider::DirectMl)],
        );
        bound.check_exclude().unwrap();
    }

    #[test]
    fn a_bare_exclusion_reaches_every_model_and_a_named_one_only_its_own() {
        let first = GPU_PRIORITY[0];
        let both = config(&["a", "b"], vec![every(first)]);
        assert!(both.excludes("a", first) && both.excludes("b", first));
        assert_eq!(both.excluded_by_every_model(), vec![first]);

        let only_a = config(&["a", "b"], vec![one("a", first)]);
        assert!(only_a.excludes("a", first) && !only_a.excludes("b", first));
        assert!(only_a.excluded_by_every_model().is_empty());

        let each = config(&["a", "b"], vec![one("a", first), one("b", first)]);
        assert_eq!(each.excluded_by_every_model(), vec![first]);
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
        let named = format!("not {}", first.name());
        assert!(
            notes
                .iter()
                .any(|n| n.contains(&named) && n.contains("excluded by every model's pin")),
            "{notes:?}"
        );
    }

    #[test]
    fn the_gpu_walk_does_not_skip_a_candidate_one_model_still_takes() {
        // One of two models excludes the first candidate, so the walk still
        // runs its preflight, and what that says is the note.
        let first = GPU_PRIORITY[0];
        let dir = empty_dir("exclude-one-model");
        let only_a = config(&["a", "b"], vec![one("a", first)]);
        let (_, notes) = resolve(Target::Gpu, &dir, &only_a.excluded_by_every_model());
        let named = format!("not {}", first.name());
        let note = notes
            .iter()
            .find(|n| n.contains(&named))
            .unwrap_or_else(|| panic!("the first candidate is named: {notes:?}"));
        assert!(!note.contains("pin"), "{note}");
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
                    .any(|n| n.contains(candidate.name()) && n.contains("every model's pin")),
                "{candidate:?} should be named as excluded: {notes:?}"
            );
        }
    }

    #[test]
    fn a_model_excluding_the_walks_provider_asks_for_the_cpu() {
        let first = GPU_PRIORITY[0];
        let only_a = config(&["a", "b"], vec![one("a", first)]);
        assert_eq!(
            ask(&only_a, first, "a"),
            Ask::Cpu(Some(format!("{} excluded by its pin", first.name())))
        );
        assert_eq!(ask(&only_a, first, "b"), Ask::Gpu(first));
    }

    #[test]
    fn a_walk_that_found_no_gpu_says_why_for_each_model() {
        // `a` excludes every candidate, so its pin is why; `b` excludes
        // none, so nothing was available.
        let all: Vec<Exclusion> = GPU_PRIORITY.iter().map(|&p| one("a", p)).collect();
        let pinned = config(&["a", "b"], all);
        let Ask::Cpu(Some(why)) = ask(&pinned, Provider::Cpu, "a") else {
            panic!("a runs on the CPU with a reason");
        };
        assert!(why.ends_with("excluded by its pin"), "{why}");
        assert_eq!(
            ask(&pinned, Provider::Cpu, "b"),
            Ask::Cpu(Some("no GPU execution provider was available".to_string()))
        );
    }

    #[test]
    fn a_named_target_asks_for_its_provider_whatever_a_model_excludes() {
        let mut named = config(&["a"], vec![one("a", Provider::Cuda)]);
        named.target = Target::Cuda;
        assert_eq!(ask(&named, Provider::Cuda, "a"), Ask::Gpu(Provider::Cuda));

        named.target = Target::Cpu;
        assert_eq!(ask(&named, Provider::Cpu, "a"), Ask::Cpu(None));
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
