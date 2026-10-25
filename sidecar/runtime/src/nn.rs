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

#[cfg(all(windows, feature = "gpu"))]
pub mod cuda;
#[cfg(all(windows, feature = "gpu"))]
pub mod directml;
pub mod dlls;
#[cfg(feature = "gpu")]
pub mod frame_tensor;
#[cfg(feature = "gpu")]
pub mod output_texture;

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

/// The environment variable naming each session's intra-op threads when the
/// command line does not.
pub const THREADS_VAR: &str = "FFRWD_NN_THREADS";

/// What `-nn-threads` and its variable say: a whole number of threads, one or
/// more. A set value that is not one is refused, since a misspelling would
/// otherwise leave every core spinning in silence.
pub fn parse_threads(raw: &str) -> Option<usize> {
    raw.trim()
        .parse::<usize>()
        .ok()
        .filter(|threads| *threads > 0)
}

/// The intra-op threads the environment names, for a run whose argv does
/// not; None when the variable is unset.
pub fn threads_from_env() -> Result<Option<usize>> {
    let Some(raw) = std::env::var_os(THREADS_VAR) else {
        return Ok(None);
    };
    let raw = raw.to_string_lossy().to_string();
    match parse_threads(&raw) {
        Some(threads) => Ok(Some(threads)),
        None => bail!("{THREADS_VAR}={raw}: a whole number of threads, 1 or more"),
    }
}

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
    /// The modules it is bound for, each a `-nn-for` after it; empty for
    /// every module.
    pub modules: Vec<PathBuf>,
}

impl ModelSpec {
    /// What the binding is called in the registry and in what the run says:
    /// its name, and the modules it is for where it is bound for some.
    pub fn key(&self) -> String {
        if self.modules.is_empty() {
            return self.name.clone();
        }
        let modules: Vec<String> = self
            .modules
            .iter()
            .map(|m| m.display().to_string())
            .collect();
        format!("{} for {}", self.name, modules.join(", "))
    }

    /// Whether a module at `module` loads this binding under its name.
    fn binds(&self, module: &Path) -> bool {
        self.modules.is_empty() || self.modules.iter().any(|m| canonical(m) == module)
    }
}

/// `path` as the file it names, where it names one.
fn canonical(path: &Path) -> PathBuf {
    path.canonicalize().unwrap_or_else(|_| path.to_path_buf())
}

/// The registry key each name a module asks for resolves to, for one
/// module's store: a binding for that module alone wins over one for every
/// module.
#[derive(Clone, Debug, Default)]
pub struct Scope(HashMap<String, String>);

impl Scope {
    /// The registry key `name` resolves to here; `name` itself where nothing
    /// binds it, which then names nothing.
    pub fn key<'a>(&'a self, name: &'a str) -> &'a str {
        self.0.get(name).map(String::as_str).unwrap_or(name)
    }
}

/// What the command line said about inference.
#[derive(Clone, Debug, Default)]
pub struct Config {
    pub models: Vec<ModelSpec>,
    pub runtime_dir: Option<PathBuf>,
    pub target: Target,
    /// What `-nn-exclude` said, in the order it said it. A `-nn-target gpu`
    /// walk prefers a provider no bound model excludes, never takes one
    /// every bound model excludes, and a model that excludes the provider
    /// the walk took runs on the CPU. Ignored for a target named outright.
    pub exclude: Vec<Exclusion>,
    /// The intra-op threads every session runs on (`-nn-threads`); ONNX
    /// Runtime's default when None.
    pub intra_threads: Option<usize>,
}

impl Config {
    /// Refuses a name bound twice for one module: twice for every module, or
    /// twice for a module named by two `-nn-for`.
    pub fn check_bindings(&self) -> Result<()> {
        for (at, spec) in self.models.iter().enumerate() {
            for earlier in &self.models[..at] {
                if earlier.name != spec.name {
                    continue;
                }
                if earlier.modules.is_empty() && spec.modules.is_empty() {
                    bail!("-nn {}: bound twice", spec.name);
                }
                if let Some(module) = spec.modules.iter().find(|m| earlier.modules.contains(m)) {
                    bail!("-nn {}: bound twice for {}", spec.name, module.display());
                }
            }
        }
        Ok(())
    }

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

    /// The candidates of a `gpu` walk some bound model excludes, and which
    /// models those are.
    fn pins(&self) -> Vec<(Provider, Pinned)> {
        GPU_PRIORITY
            .iter()
            .copied()
            .filter_map(|p| {
                let by: Vec<String> = self
                    .models
                    .iter()
                    .filter(|m| self.excludes(&m.name, p))
                    .map(ModelSpec::key)
                    .collect();
                if by.is_empty() {
                    None
                } else if by.len() == self.models.len() {
                    Some((p, Pinned::Every))
                } else {
                    Some((p, Pinned::By(by)))
                }
            })
            .collect()
    }
}

/// Which bound models' pins exclude a candidate of the `gpu` walk.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Pinned {
    /// Every one's.
    Every,
    /// These, by name, and not the rest.
    By(Vec<String>),
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
        backend.name_next(name);
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
    /// Every session, by its binding's key.
    registry: ModelRegistry,
    /// The bindings, in `-nn` order.
    bindings: Vec<ModelSpec>,
    names: Vec<String>,
    provider: Provider,
    /// The intra-op threads each session runs on (`-nn-threads`).
    intra_threads: Option<usize>,
    /// The provider each model's session engaged, by its binding's key.
    placed: HashMap<String, Provider>,
    /// The CUDA runtime the CUDA provider loads, where the walk took CUDA
    /// and its preflight found one.
    cuda_runtime: Option<PathBuf>,
}

/// Loaded once per process: building a session is expensive and `Graph` is
/// reference-counted, so every store shares the one session. `Backend` is not
/// shared, and is rebuilt for each store.
/// Whether this process has loaded ONNX Runtime.
static LOADED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Whether this process has loaded ONNX Runtime: a run binding no model never
/// does.
pub fn runtime_loaded() -> bool {
    LOADED.load(std::sync::atomic::Ordering::Acquire)
}

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

/// The provider the session of the binding keyed `key` engaged
/// ([`Scope::key`]); None for a key no `-nn` makes.
pub fn provider_of(key: &str) -> Option<Provider> {
    graph_registry().get()?.placed.get(key).copied()
}

/// The key of the `-nn` binding whose session `key` is
/// ([`wasmtime_wasi_nn::backend::BackendGraph::session_key`]).
pub fn model_of_session(key: usize) -> Option<String> {
    let models = graph_registry().get()?;
    models
        .names
        .iter()
        .find(|name| {
            models
                .registry
                .get(name)
                .is_some_and(|graph| key != 0 && graph.session_key() == key)
        })
        .cloned()
}

/// The CUDA runtime library the CUDA provider loads, where the process's
/// provider is CUDA and its preflight found one.
pub fn cuda_runtime() -> Option<PathBuf> {
    graph_registry().get()?.cuda_runtime.clone()
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
fn onnx_backend(provider: Provider, intra_threads: Option<usize>) -> OnnxBackend {
    OnnxBackend::with_gpu_providers(vec![provider]).with_intra_threads(intra_threads)
}

/// The stream every CUDA session is built on, where one was made: the
/// waits for a frame's tensor and the signals after an output is written
/// are queued in it, so the GPU orders them with the sessions' work.
#[cfg(all(windows, feature = "gpu"))]
pub fn session_stream() -> Option<&'static cuda::CudaStream> {
    session_stream_cell().get()
}

#[cfg(all(windows, feature = "gpu"))]
fn session_stream_cell() -> &'static OnceLock<cuda::CudaStream> {
    static STREAM: OnceLock<cuda::CudaStream> = OnceLock::new();
    &STREAM
}

/// `onnx` with the providers the host registers itself.
///
/// A CUDA session is built on a stream of the host's, so a wait for a
/// frame's tensor is queued ahead of the run on the GPU rather than waited
/// for by the CPU. Where the stream cannot be made, CUDA is registered as
/// `ort` registers it and the CPU waits instead.
///
/// A DirectML session, where a module is granted the process's device and
/// that device is D3D12, is built on that device and its queue, so buffers
/// of the device bind to it. Where that cannot be done, DirectML is
/// registered as `ort` registers it, on a device of its own.
#[cfg(all(windows, feature = "gpu"))]
fn registered(
    onnx: OnnxBackend,
    provider: Provider,
    cuda_runtime: Option<&Path>,
    directml_lib: Option<&Path>,
) -> OnnxBackend {
    use ort::execution_providers::{CUDAExecutionProvider, ExecutionProvider};
    match provider {
        Provider::Cuda => {
            let made = cuda_runtime
                .ok_or_else(|| anyhow::anyhow!("the CUDA runtime the provider loads was not found"))
                .and_then(cuda::Cuda::open)
                .and_then(|cuda| cuda.stream());
            let stream = match made {
                Ok(stream) => session_stream_cell().get_or_init(|| stream),
                Err(e) => {
                    eprintln!(
                        "[nn] CUDA sessions run on the provider's own stream ({}); a frame's                          tensor is waited for by the CPU",
                        one_line(&format!("{e:#}"))
                    );
                    return onnx;
                }
            };
            let raw = stream.raw() as usize;
            onnx.with_registrar(
                Provider::Cuda,
                Arc::new(move |builder| {
                    // SAFETY: the stream is the process's, never destroyed.
                    let cuda = unsafe {
                        CUDAExecutionProvider::default().with_compute_stream(raw as *mut ())
                    };
                    cuda.register(builder).map_err(|e| e.to_string())
                }),
            )
        }
        Provider::DirectMl if crate::runtime::any_gpu_granted() => {
            let (dml, queue) = match directml::prepare(directml_lib) {
                Ok(made) => made,
                Err(e) => {
                    eprintln!(
                        "[nn] DIRECTML sessions run on a device of their own ({}); a frame's                          tensor is read back for them",
                        one_line(&format!("{e:#}"))
                    );
                    return onnx;
                }
            };
            onnx.with_registrar(
                Provider::DirectMl,
                Arc::new(move |builder| {
                    use ort::AsPointer;
                    // SAFETY: live options, and what `prepare` made, held
                    // for the process.
                    unsafe { directml::register(builder.ptr_mut(), dml, queue) }
                        .map_err(|e| format!("{e:#}"))
                }),
            )
        }
        _ => onnx,
    }
}

#[cfg(not(all(windows, feature = "gpu")))]
fn registered(
    onnx: OnnxBackend,
    _provider: Provider,
    _cuda_runtime: Option<&Path>,
    _directml_lib: Option<&Path>,
) -> OnnxBackend {
    onnx
}

/// `onnx` with CoreML registered as the host wants it: in the MLProgram
/// format, which runs more of a model than the default and does so on the
/// GPU, on the CPU and GPU only, with each compiled model cached under the
/// runtime's directory, so a run after the first skips the compile. The
/// Neural Engine is left out: CoreML compiles a model for it again on every
/// load, which no cache keeps (45 s for X-CLIP's video tower, every run). The backend builds a CoreML
/// session for each shape a model's inputs arrive in, with the model's free
/// dimensions fixed to it, since CoreML runs a graph with free dimensions
/// mostly on the CPU.
#[cfg(target_os = "macos")]
fn coreml_registered(onnx: OnnxBackend, provider: Provider, dir: &Path) -> OnnxBackend {
    use ort::execution_providers::coreml::{
        CoreMLComputeUnits, CoreMLExecutionProvider, CoreMLModelFormat,
    };
    use ort::execution_providers::ExecutionProvider;
    if provider != Provider::CoreMl {
        return onnx;
    }
    let cache = dir.join("coreml-cache");
    let cache = std::fs::create_dir_all(&cache)
        .ok()
        .map(|()| cache.to_string_lossy().into_owned());
    onnx.with_registrar(
        Provider::CoreMl,
        Arc::new(move |builder| {
            let mut coreml = CoreMLExecutionProvider::default()
                .with_model_format(CoreMLModelFormat::MLProgram)
                .with_compute_units(CoreMLComputeUnits::CPUAndGPU);
            if let Some(cache) = &cache {
                coreml = coreml.with_model_cache_dir(cache);
            }
            coreml.register(builder).map_err(|e| e.to_string())
        }),
    )
}

/// The wasi-nn context the store of the module at `module_path` gets, and
/// the key each name it asks for resolves to. Backends are consumed by the
/// context, so each store builds its own; the graphs behind them are shared.
pub fn store_ctx(module_path: &str) -> (WasiNnCtx, Scope) {
    let Some(models) = graph_registry().get() else {
        return (empty_ctx(), Scope::default());
    };
    let module = canonical(Path::new(module_path));
    let mut scope = HashMap::new();
    for for_every in [true, false] {
        for spec in &models.bindings {
            if spec.modules.is_empty() == for_every && spec.binds(&module) {
                scope.insert(spec.name.clone(), spec.key());
            }
        }
    }
    let graphs: HashMap<String, Graph> = scope
        .iter()
        .filter_map(|(name, key)| Some((name.clone(), models.registry.0.get(key)?.clone())))
        .collect();
    let ctx = WasiNnCtx::new(
        vec![Backend::from(onnx_backend(
            models.provider,
            models.intra_threads,
        ))],
        Registry::from(ModelRegistry(Arc::new(graphs))),
    );
    (ctx, Scope(scope))
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
/// `gpu` walks the platform's priority order ([`walk`]), saying in one line
/// why it passed over each candidate it did not take. `pins` is the
/// candidates bound models exclude, and which. A provider named outright is
/// taken as named and nothing is excluded: its complaints are warnings, and
/// a session that cannot engage it runs on the CPU and says so.
fn resolve(target: Target, dir: &Path, pins: &[(Provider, Pinned)]) -> (Provider, Vec<String>) {
    match target {
        Target::Cpu => (Provider::Cpu, Vec::new()),
        Target::Cuda => (Provider::Cuda, preflight(Provider::Cuda, dir)),
        Target::DirectMl => (Provider::DirectMl, preflight(Provider::DirectMl, dir)),
        Target::CoreMl => (Provider::CoreMl, preflight(Provider::CoreMl, dir)),
        Target::Gpu => walk(GPU_PRIORITY, pins, |candidate| preflight(candidate, dir)),
    }
}

/// The `gpu` walk over `candidates`, in order: the first whose preflight is
/// clean and that no bound model excludes; failing that, the first whose
/// preflight is clean and that some model still takes, the models that
/// exclude it running on the CPU; failing that, the CPU. A candidate every
/// model excludes is dropped before its preflight ever runs -- the models'
/// own pins, not this machine, ruled it out. The notes name each candidate
/// passed over and why, one line each.
fn walk(
    candidates: &[Provider],
    pins: &[(Provider, Pinned)],
    preflight: impl Fn(Provider) -> Vec<String>,
) -> (Provider, Vec<String>) {
    let mut notes = Vec::new();
    // The first clean candidate some model excludes, and where its note is.
    let mut second: Option<(Provider, usize)> = None;
    for &candidate in candidates {
        let pinned = pins.iter().find(|(p, _)| *p == candidate).map(|(_, by)| by);
        let why = match pinned {
            Some(Pinned::Every) => "excluded by every model's pin".to_string(),
            _ => match (preflight(candidate).first(), pinned) {
                (Some(first), _) => first.clone(),
                (None, Some(Pinned::By(names))) => {
                    if second.is_none() {
                        second = Some((candidate, notes.len()));
                    }
                    let pin = if names.len() == 1 { "pin" } else { "pins" };
                    format!("excluded by the {pin} of {}", names.join(", "))
                }
                (None, _) => return (candidate, notes),
            },
        };
        notes.push(format!("-nn-target gpu: not {}, {why}", candidate.name()));
    }
    match second {
        Some((candidate, at)) => {
            notes.remove(at);
            (candidate, notes)
        }
        None => (Provider::Cpu, notes),
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
    config.check_bindings()?;
    config.check_exclude()?;

    let dir = resolve_runtime_dir(config)?;

    // Which provider before which runtime: DirectML's ONNX Runtime is a
    // different binary, so the library to load follows from the answer.
    let pins = match config.target {
        Target::Gpu => config.pins(),
        _ => Vec::new(),
    };
    let (provider, complaints) = resolve(config.target, &dir, &pins);
    for complaint in complaints {
        eprintln!("[nn] {complaint}");
    }
    let lib = open_runtime(&runtime_lib_dir(&dir, provider))?;

    ort::init_from(lib.to_string_lossy().to_string())
        .commit()
        .map_err(|e| anyhow::anyhow!("initialising ONNX Runtime from {}: {e}", lib.display()))?;
    LOADED.store(true, std::sync::atomic::Ordering::Release);

    let cuda_runtime = match provider {
        Provider::Cuda => dlls::locate(&dlls::search_dirs(&dir), dlls::CUDA_RUNTIME)
            .map(|found| found.join(dlls::CUDA_RUNTIME)),
        _ => None,
    };
    let directml_lib = match provider {
        Provider::DirectMl => {
            let dml = runtime_lib_dir(&dir, provider);
            dlls::locate(&dlls::search_dirs(&dml), dlls::DIRECTML_LIB)
                .map(|found| found.join(dlls::DIRECTML_LIB))
        }
        _ => None,
    };
    let onnx = registered(
        onnx_backend(provider, config.intra_threads),
        provider,
        cuda_runtime.as_deref(),
        directml_lib.as_deref(),
    );
    #[cfg(target_os = "macos")]
    let onnx = coreml_registered(onnx, provider, &dir);
    let mut onnx = Backend::from(onnx);

    let mut registry = ModelRegistry::default();
    let mut names = Vec::new();
    let mut placements = Vec::new();
    for spec in &config.models {
        let key = spec.key();
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
                match registry.register(&mut onnx, &key, &bytes, ExecutionTarget::Gpu) {
                    Ok(()) => (asked, None),
                    // The session refused rather than falling back in silence;
                    // the CPU answers the same, and slower.
                    Err(e) => {
                        eprintln!(
                            "[nn] {}: {} did not engage ({}); runs on the CPU",
                            key,
                            asked.name(),
                            one_line(&e.root_cause().to_string())
                        );
                        registry
                            .register(&mut onnx, &key, &bytes, ExecutionTarget::Cpu)
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
                    .register(&mut onnx, &key, &bytes, ExecutionTarget::Cpu)
                    .with_context(loading)?;
                (Provider::Cpu, why)
            }
        };
        placements.push(Placement {
            name: key.clone(),
            provider: placed,
            why_cpu,
        });
        names.push(key);
    }

    report_verdict(&dir, config, &placements);

    let placed = placements
        .iter()
        .map(|p| (p.name.clone(), p.provider))
        .collect();
    graph_registry()
        .set(Models {
            registry,
            bindings: config.models.clone(),
            names,
            provider,
            intra_threads: config.intra_threads,
            placed,
            cuda_runtime,
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
                    modules: Vec::new(),
                })
                .collect(),
            runtime_dir: None,
            target: Target::Gpu,
            exclude,
            intra_threads: None,
        }
    }

    #[test]
    fn threads_are_a_whole_number_of_one_or_more() {
        assert_eq!(parse_threads("1"), Some(1));
        assert_eq!(parse_threads(" 4 "), Some(4));
        for refused in ["0", "-1", "1.5", "one", ""] {
            assert_eq!(parse_threads(refused), None, "{refused:?}");
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
        assert_eq!(both.pins(), vec![(first, Pinned::Every)]);

        let only_a = config(&["a", "b"], vec![one("a", first)]);
        assert!(only_a.excludes("a", first) && !only_a.excludes("b", first));
        assert_eq!(
            only_a.pins(),
            vec![(first, Pinned::By(vec!["a".to_string()]))]
        );

        let each = config(&["a", "b"], vec![one("a", first), one("b", first)]);
        assert_eq!(each.pins(), vec![(first, Pinned::Every)]);
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
        let (provider, notes) = resolve(Target::Gpu, &dir, &[(first, Pinned::Every)]);
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
        let (provider, notes) = resolve(Target::Gpu, &dir, &only_a.pins());
        if preflight(first, &dir).is_empty() {
            // A provider needing nothing on disk (CoreML) passes its
            // preflight even here, and is taken.
            assert_eq!(provider, first, "{notes:?}");
            return;
        }
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
        let exclude: Vec<(Provider, Pinned)> =
            GPU_PRIORITY.iter().map(|&p| (p, Pinned::Every)).collect();
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

    /// A preflight that is clean for `clean` and complains of the rest.
    fn clean_for(clean: &[Provider]) -> impl Fn(Provider) -> Vec<String> + '_ {
        move |p| {
            if clean.contains(&p) {
                Vec::new()
            } else {
                vec![format!("{} is not on this machine", p.name())]
            }
        }
    }

    #[test]
    fn the_gpu_walk_prefers_a_candidate_no_model_excludes() {
        let candidates = [Provider::DirectMl, Provider::Cuda];
        let both = clean_for(&candidates);
        let ages = vec![(Provider::DirectMl, Pinned::By(vec!["ages".to_string()]))];
        assert_eq!(
            walk(&candidates, &ages, &both),
            (
                Provider::Cuda,
                vec!["-nn-target gpu: not DIRECTML, excluded by the pin of ages".to_string()]
            )
        );
        let two = vec![(
            Provider::DirectMl,
            Pinned::By(vec!["ages".to_string(), "faces".to_string()]),
        )];
        assert_eq!(
            walk(&candidates, &two, &both).1,
            vec!["-nn-target gpu: not DIRECTML, excluded by the pins of ages, faces".to_string()]
        );
        // Nothing excluded: the first clean candidate, and nothing said.
        assert_eq!(
            walk(&candidates, &[], &both),
            (Provider::DirectMl, Vec::new())
        );
    }

    #[test]
    fn the_gpu_walk_takes_a_candidate_some_model_excludes_when_no_other_is_clean() {
        // CUDA's preflight complains, so DirectML is taken though `ages`
        // excludes it, and `ages` runs on the CPU; the notes say why not
        // CUDA, and nothing of DirectML.
        let candidates = [Provider::DirectMl, Provider::Cuda];
        let ages = vec![(Provider::DirectMl, Pinned::By(vec!["ages".to_string()]))];
        assert_eq!(
            walk(&candidates, &ages, clean_for(&[Provider::DirectMl])),
            (
                Provider::DirectMl,
                vec!["-nn-target gpu: not CUDA, CUDA is not on this machine".to_string()]
            )
        );
        // A candidate every model excludes is never taken.
        let every = vec![
            (Provider::DirectMl, Pinned::Every),
            (Provider::Cuda, Pinned::By(vec!["ages".to_string()])),
        ];
        assert_eq!(
            walk(&candidates, &every, clean_for(&candidates)),
            (
                Provider::Cuda,
                vec!["-nn-target gpu: not DIRECTML, excluded by every model's pin".to_string()]
            )
        );
        let (provider, notes) = walk(&candidates, &every, clean_for(&[Provider::DirectMl]));
        assert_eq!(provider, Provider::Cpu);
        assert_eq!(notes.len(), 2, "{notes:?}");
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
        let (provider, _) = resolve(
            Target::DirectMl,
            &dir,
            &[(Provider::DirectMl, Pinned::Every)],
        );
        assert_eq!(provider, Provider::DirectMl);
    }
}
