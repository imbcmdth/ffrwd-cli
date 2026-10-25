//! Implements a `wasi-nn` [`BackendInner`] using ONNX via the `ort` crate.
//!
//! This file carries the vendored copy's changes to the ONNX backend, which
//! are upstreaming candidates: `ExecutionTarget::Gpu` resolves through a
//! [`Provider`] priority list the embedder supplies rather than always meaning
//! CUDA, a `Gpu` session whose providers all fail to register is an error
//! rather than a silent fall back to the CPU, integer tensors are
//! accepted alongside fp32, an input whose elements are in device memory
//! the embedder holds ([`crate::DeviceInput`]) is bound to a CUDA or
//! DirectML session through an `IoBinding` instead of being copied into a
//! CPU tensor, an output can be bound to device memory the embedder holds
//! ([`crate::DeviceOutput`]) the same way, and the embedder can register a
//! provider on a session itself ([`OnnxBackend::with_registrar`]), which is
//! how a session is given a CUDA stream or a D3D12 device of the
//! embedder's, and the embedder can name how many intra-op threads every
//! session runs on ([`OnnxBackend::with_intra_threads`]). Everything else
//! here is upstream's.

use super::{
    BackendError, BackendExecutionContext, BackendFromDir, BackendGraph, BackendInner, NamedTensor,
};
use crate::backend::{Id, read};
use crate::wit::types::{ExecutionTarget, GraphEncoding, Tensor, TensorType};
use crate::{DeviceHandle, DeviceInput, DeviceMemory, DeviceOutputHandle, ExecutionContext, Graph};
use ort::{
    execution_providers::{CPUExecutionProvider, ExecutionProviderDispatch},
    memory::{AllocationDevice, AllocatorType, MemoryInfo, MemoryType},
    session::{Input, Output},
    session::{
        Session, SessionInputValue,
        builder::{GraphOptimizationLevel, SessionBuilder},
    },
    tensor::TensorElementType,
    value::{DynValue, Tensor as OrtTensor, TensorRefMut, ValueType},
};

#[cfg(feature = "onnx-cuda")]
use ort::execution_providers::CUDAExecutionProvider;
#[cfg(feature = "onnx-coreml")]
use ort::execution_providers::CoreMLExecutionProvider;
#[cfg(feature = "onnx-directml")]
use ort::execution_providers::DirectMLExecutionProvider;

use std::path::Path;
use std::sync::{Arc, Mutex};

/// An execution provider a session can be put on.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Provider {
    Cpu,
    Cuda,
    DirectMl,
    CoreMl,
}

impl Provider {
    /// What this provider is called in a message to whoever asked for it.
    pub fn name(self) -> &'static str {
        match self {
            Provider::Cpu => "CPU",
            Provider::Cuda => "CUDA",
            Provider::DirectMl => "DIRECTML",
            Provider::CoreMl => "COREML",
        }
    }

    /// The `ort` provider this names, or `None` when this build does not
    /// carry it. `None` is passed over rather than refused: the list is a
    /// priority order, so the next entry answers.
    fn dispatch(self) -> Option<ExecutionProviderDispatch> {
        match self {
            Provider::Cpu => Some(CPUExecutionProvider::default().build()),
            Provider::Cuda => {
                #[cfg(feature = "onnx-cuda")]
                {
                    Some(CUDAExecutionProvider::default().build())
                }
                #[cfg(not(feature = "onnx-cuda"))]
                {
                    None
                }
            }
            Provider::DirectMl => {
                #[cfg(feature = "onnx-directml")]
                {
                    Some(DirectMLExecutionProvider::default().build())
                }
                #[cfg(not(feature = "onnx-directml"))]
                {
                    None
                }
            }
            Provider::CoreMl => {
                #[cfg(feature = "onnx-coreml")]
                {
                    Some(CoreMLExecutionProvider::default().build())
                }
                #[cfg(not(feature = "onnx-coreml"))]
                {
                    None
                }
            }
        }
    }
}

/// What a `Gpu` session asks for when the embedder names nothing: the
/// providers this platform can have, most wanted first.
fn platform_gpu_providers() -> Vec<Provider> {
    if cfg!(target_vendor = "apple") {
        vec![Provider::CoreMl]
    } else if cfg!(windows) {
        vec![Provider::Cuda, Provider::DirectMl]
    } else {
        vec![Provider::Cuda]
    }
}

/// Registers a provider on a session being built, as the embedder wants it
/// registered; an error is the provider refusing, and the session is not
/// built on it.
pub type Registrar = Arc<dyn Fn(&mut SessionBuilder) -> Result<(), String> + Send + Sync>;

pub struct OnnxBackend {
    /// What an `ExecutionTarget::Gpu` session asks for, most wanted first.
    gpu_providers: Vec<Provider>,
    /// Providers the embedder registers itself, in place of `ort`'s default
    /// registration of them.
    registrars: Vec<(Provider, Registrar)>,
    /// What the next graph loaded is called on stderr.
    next_name: String,
    /// The intra-op threads every session runs on; ONNX Runtime's default
    /// when None.
    intra_threads: Option<usize>,
}

impl Default for OnnxBackend {
    fn default() -> Self {
        Self {
            gpu_providers: platform_gpu_providers(),
            registrars: Vec::new(),
            next_name: String::new(),
            intra_threads: None,
        }
    }
}

impl OnnxBackend {
    /// A backend whose `Gpu` sessions ask for these providers, in this order.
    ///
    /// An embedder that resolved one itself - it knows which libraries it put
    /// on disk and which of them loaded - names just that one, and the session
    /// either engages it or fails to load, saying why. Loading the same bytes
    /// again with `ExecutionTarget::Cpu` is how it then runs on the CPU.
    pub fn with_gpu_providers(gpu_providers: Vec<Provider>) -> Self {
        Self {
            gpu_providers,
            registrars: Vec::new(),
            next_name: String::new(),
            intra_threads: None,
        }
    }

    /// Builds every session on `threads` intra-op threads rather than ONNX
    /// Runtime's default, a pool as wide as the machine whose threads keep
    /// spinning after each run returns. One thread is no pool at all, which
    /// is what a small model run many times a second wants: a 160 ms speech
    /// chunk took 8% of a core on one thread and eight cores on the default.
    pub fn with_intra_threads(mut self, threads: Option<usize>) -> Self {
        self.intra_threads = threads;
        self
    }

    /// Has `provider` registered on every session by `registrar` rather than
    /// by `ort`'s defaults: a CUDA session on a stream of the embedder's, a
    /// DirectML one on its D3D12 device and queue.
    pub fn with_registrar(mut self, provider: Provider, registrar: Registrar) -> Self {
        self.registrars.retain(|(p, _)| *p != provider);
        self.registrars.push((provider, registrar));
        self
    }

    fn registrar(&self, provider: Provider) -> Option<&Registrar> {
        self.registrars
            .iter()
            .find(|(p, _)| *p == provider)
            .map(|(_, r)| r)
    }
}

unsafe impl Send for OnnxBackend {}
unsafe impl Sync for OnnxBackend {}

impl BackendInner for OnnxBackend {
    fn encoding(&self) -> GraphEncoding {
        GraphEncoding::Onnx
    }

    fn load(&mut self, builders: &[&[u8]], target: ExecutionTarget) -> Result<Graph, BackendError> {
        if builders.len() != 1 {
            return Err(BackendError::InvalidNumberOfBuilders(1, builders.len()));
        }

        // One provider at a time, in order, each told to fail rather than
        // leave ONNX Runtime to fall back to the CPU in silence. The first
        // that registers builds the session; none registering fails the load.
        let mut refused: Vec<(Provider, String)> = Vec::new();
        for provider in session_providers(target, &self.gpu_providers) {
            if let Some(registrar) = self.registrar(provider) {
                if let Some(graph) = specialized(
                    provider,
                    registrar,
                    builders[0],
                    &self.next_name,
                    self.intra_threads,
                )? {
                    return Ok(graph);
                }
                let mut builder = session_builder(self.intra_threads)?;
                if let Err(e) = registrar(&mut builder) {
                    tracing::debug!(
                        "{} execution provider did not register: {e}",
                        provider.name()
                    );
                    refused.push((provider, e));
                    continue;
                }
                let session = builder
                    .with_optimization_level(GraphOptimizationLevel::Level3)?
                    .commit_from_memory(builders[0])?;
                let box_: Box<dyn BackendGraph> = Box::new(OnnxGraph {
                    session: Arc::new(Mutex::new(session)),
                    provider,
                    shapes: None,
                });
                return Ok(box_.into());
            }
            let Some(dispatch) = provider.dispatch() else {
                tracing::debug!(
                    "{} execution provider is not enabled in this build",
                    provider.name()
                );
                refused.push((provider, "not enabled in this build".to_string()));
                continue;
            };
            let builder = match session_builder(self.intra_threads)?
                .with_execution_providers([dispatch.error_on_failure()])
            {
                Ok(builder) => builder,
                Err(e) => {
                    tracing::debug!(
                        "{} execution provider did not register: {e}",
                        provider.name()
                    );
                    refused.push((provider, e.to_string()));
                    continue;
                }
            };
            tracing::debug!("Using {} execution provider", provider.name());
            let session = builder
                .with_optimization_level(GraphOptimizationLevel::Level3)?
                .commit_from_memory(builders[0])?;

            let box_: Box<dyn BackendGraph> = Box::new(OnnxGraph {
                session: Arc::new(Mutex::new(session)),
                provider,
                shapes: None,
            });
            return Ok(box_.into());
        }

        let why = match refused.as_slice() {
            [] => "no GPU execution provider was named".to_string(),
            [(_, only)] => only.clone(),
            all => all
                .iter()
                .map(|(provider, e)| format!("{}: {e}", provider.name()))
                .collect::<Vec<_>>()
                .join("; "),
        };
        Err(BackendError::BackendAccess(wasmtime::format_err!("{why}")))
    }

    fn as_dir_loadable<'a>(&'a mut self) -> Option<&'a mut dyn BackendFromDir> {
        Some(self)
    }

    fn name_next(&mut self, name: &str) {
        self.next_name = name.to_string();
    }
}

/// Whether a session on `provider` is built again for each shape its inputs
/// arrive in. CoreML takes a graph with free dimensions only in part, and
/// runs the rest on the CPU, so the session is built with them fixed.
fn by_shape(provider: Provider) -> bool {
    provider == Provider::CoreMl
}

/// A graph whose session is built for each shape its inputs arrive in, where
/// `provider` is one ([`by_shape`]) and every free dimension of the model's
/// inputs is named; None otherwise. The session read for the inputs' names
/// and types runs on the CPU and is never run.
fn specialized(
    provider: Provider,
    registrar: &Registrar,
    model: &[u8],
    name: &str,
    intra_threads: Option<usize>,
) -> Result<Option<Graph>, BackendError> {
    if !by_shape(provider) {
        return Ok(None);
    }
    let described = Session::builder()?.commit_from_memory(model)?;
    let mut free = Vec::new();
    for (input, slot) in described.inputs.iter().enumerate() {
        let ValueType::Tensor {
            shape,
            dimension_symbols,
            ..
        } = &slot.input_type
        else {
            return Ok(None);
        };
        for (dimension, (&size, symbol)) in shape.iter().zip(dimension_symbols.iter()).enumerate() {
            if size >= 0 {
                continue;
            }
            if symbol.is_empty() {
                return Ok(None);
            }
            free.push((input, dimension, symbol.clone()));
        }
    }
    if free.is_empty() {
        return Ok(None);
    }
    let box_: Box<dyn BackendGraph> = Box::new(OnnxGraph {
        session: Arc::new(Mutex::new(described)),
        provider,
        shapes: Some(Arc::new(ByShape {
            name: name.to_string(),
            model: model.to_vec(),
            provider,
            registrar: registrar.clone(),
            free,
            intra_threads,
            sessions: Mutex::new(Vec::new()),
            fallen: Mutex::new(None),
        })),
    });
    Ok(Some(box_.into()))
}

/// A session builder on `intra_threads` intra-op threads, or on ONNX
/// Runtime's default with None.
fn session_builder(intra_threads: Option<usize>) -> ort::Result<SessionBuilder> {
    let builder = Session::builder()?;
    match intra_threads {
        Some(threads) => builder.with_intra_threads(threads),
        None => Ok(builder),
    }
}

/// A model's sessions, one for each shape its inputs arrived in, or, once
/// one of them could not be built or failed its first run, the one session
/// the provider builds for the model by default.
struct ByShape {
    /// What the model is called on stderr.
    name: String,
    model: Vec<u8>,
    provider: Provider,
    registrar: Registrar,
    /// Each free dimension of the inputs: the input, the dimension, and its
    /// name.
    free: Vec<(usize, usize, String)>,
    /// The intra-op threads its sessions run on.
    intra_threads: Option<usize>,
    /// The sizes the free dimensions took, and the session built for them.
    sessions: Mutex<Vec<(Vec<i64>, Arc<Mutex<Session>>)>>,
    /// The session every shape runs on once a session for a shape could not
    /// be built or failed its first run.
    fallen: Mutex<Option<Arc<Mutex<Session>>>>,
}

impl ByShape {
    /// The session for the shapes `inputs` hold, built the first time they
    /// arrive in them, and whether it was built just now.
    fn session(&self, inputs: &[TensorSlot]) -> Result<(Arc<Mutex<Session>>, bool), BackendError> {
        if let Some(fallen) = self.fallen.lock().unwrap().as_ref() {
            return Ok((fallen.clone(), false));
        }
        let mut sizes = Vec::with_capacity(self.free.len());
        for (input, dimension, name) in &self.free {
            let size = inputs
                .get(*input)
                .and_then(|slot| slot.tensor.as_ref())
                .and_then(|tensor| tensor.dimensions.get(*dimension))
                .ok_or_else(|| {
                    BackendError::BackendAccess(wasmtime::format_err!(
                        "no input gives the model's dimension {name}"
                    ))
                })?;
            sizes.push(i64::from(*size));
        }
        let mut sessions = self.sessions.lock().unwrap();
        if let Some((_, session)) = sessions.iter().find(|(made, _)| *made == sizes) {
            return Ok((session.clone(), false));
        }
        let mut builder = session_builder(self.intra_threads)?;
        (self.registrar)(&mut builder).map_err(|e| {
            BackendError::BackendAccess(wasmtime::format_err!("the provider did not register: {e}"))
        })?;
        for ((_, _, name), size) in self.free.iter().zip(&sizes) {
            builder = builder.with_dimension_override(name, *size)?;
        }
        let session = Arc::new(Mutex::new(
            builder
                .with_optimization_level(GraphOptimizationLevel::Level3)?
                .commit_from_memory(&self.model)?,
        ));
        sessions.push((sizes, session.clone()));
        Ok((session, true))
    }

    /// The session the provider builds for the model by default, which
    /// every shape runs on from now on: what a session for a shape gives
    /// way to when it could not be built or failed its first run (`why`).
    /// Said once on stderr.
    fn fall_back(&self, why: &BackendError) -> Result<Arc<Mutex<Session>>, BackendError> {
        let mut fallen = self.fallen.lock().unwrap();
        if let Some(session) = fallen.as_ref() {
            return Ok(session.clone());
        }
        eprintln!(
            "[nn] {}: a {} session built for its inputs' shapes failed ({why}); the model \
             runs on the session {} builds by default",
            self.name,
            self.provider.name(),
            self.provider.name()
        );
        let dispatch = self.provider.dispatch().ok_or_else(|| {
            BackendError::BackendAccess(wasmtime::format_err!(
                "{} is not enabled in this build",
                self.provider.name()
            ))
        })?;
        let session = Arc::new(Mutex::new(
            session_builder(self.intra_threads)?
                .with_execution_providers([dispatch.error_on_failure()])?
                .with_optimization_level(GraphOptimizationLevel::Level3)?
                .commit_from_memory(&self.model)?,
        ));
        self.sessions.lock().unwrap().clear();
        *fallen = Some(session.clone());
        Ok(session)
    }
}

/// The providers a session tries, in order, for a target.
fn session_providers(target: ExecutionTarget, gpu: &[Provider]) -> Vec<Provider> {
    match target {
        ExecutionTarget::Cpu => vec![Provider::Cpu],
        ExecutionTarget::Gpu => gpu.to_vec(),
        ExecutionTarget::Tpu => {
            tracing::warn!(
                "TPU execution target is not supported for ONNX backend yet, falling back to CPU"
            );
            vec![Provider::Cpu]
        }
    }
}

impl BackendFromDir for OnnxBackend {
    fn load_from_dir(
        &mut self,
        path: &Path,
        target: ExecutionTarget,
    ) -> Result<Graph, BackendError> {
        let model = read(&path.join("model.onnx"))?;
        self.load(&[&model], target)
    }
}

struct OnnxGraph {
    session: Arc<Mutex<Session>>,
    /// The provider the session was built on, which decides whether an
    /// input in device memory can be bound to it.
    provider: Provider,
    /// The sessions built for each shape the inputs arrive in, where the
    /// provider takes fixed shapes only; `session` then only describes the
    /// model.
    shapes: Option<Arc<ByShape>>,
}
unsafe impl Send for OnnxGraph {}
unsafe impl Sync for OnnxGraph {}

impl BackendGraph for OnnxGraph {
    fn init_execution_context(&self) -> Result<ExecutionContext, BackendError> {
        let session = self.session.lock().unwrap();
        // We need to hold on to the names of the inputs in order for
        // `set_input` to work with both indexes and names. Having the
        // dimensions and type around is useful for validation but could be
        // retrieved from the session.
        let mut inputs = vec![];
        for input in &session.inputs {
            let shape = Shape::from_onnx_input(input)?;
            inputs.push(TensorSlot {
                shape,
                tensor: None,
                bound: None,
            });
        }
        // We need to keep track of the output shapes since they are used for
        // creating the output tensor.
        let mut outputs = vec![];
        for output in &session.outputs {
            let shape = Shape::from_onnx_output(output)?;
            outputs.push(TensorSlot {
                shape,
                tensor: None,
                bound: None,
            });
        }
        let box_: Box<dyn BackendExecutionContext> = Box::new(OnnxExecutionContext {
            session: self.session.clone(),
            provider: self.provider,
            shapes: self.shapes.clone(),
            inputs,
            outputs,
        });
        Ok(box_.into())
    }

    fn session_key(&self) -> usize {
        Arc::as_ptr(&self.session) as usize
    }
}

struct OnnxExecutionContext {
    session: Arc<Mutex<Session>>,
    provider: Provider,
    shapes: Option<Arc<ByShape>>,
    inputs: Vec<TensorSlot>,
    outputs: Vec<TensorSlot>,
}

unsafe impl Send for OnnxExecutionContext {}
unsafe impl Sync for OnnxExecutionContext {}

impl OnnxExecutionContext {
    /// Runs the session over the inputs set, and reads every output back as
    /// the tensor its type asks for, in the session's order.
    ///
    /// With every input in host memory and no output bound to device memory
    /// the session is run as upstream runs it. Otherwise the inputs and
    /// outputs are bound instead: an input in device memory as it is, the
    /// others as CPU tensors; an output bound to device memory there, the
    /// others to host memory, where they are read as before. An output
    /// bound to device memory comes back as a tensor on the device, whose
    /// bytes are read back only when asked for.
    fn run(&mut self) -> Result<Vec<Tensor>, BackendError> {
        let on_device = self
            .inputs
            .iter()
            .any(|slot| slot.tensor.as_ref().is_some_and(|t| t.device.is_some()))
            || self.outputs.iter().any(|slot| {
                slot.bound
                    .as_ref()
                    .is_some_and(|b| b.output.0.memory().is_some())
            });
        for slot in &self.outputs {
            if let Some(bound) = &slot.bound {
                bound.output.0.ready()?;
            }
        }
        let Some(shapes) = self.shapes.clone() else {
            return self.run_on(&self.session, on_device);
        };
        let (session, built) = match shapes.session(&self.inputs) {
            Ok(made) => made,
            Err(why) => (shapes.fall_back(&why)?, false),
        };
        match self.run_on(&session, on_device) {
            Err(why) if built => {
                let session = shapes.fall_back(&why)?;
                self.run_on(&session, on_device)
            }
            ran => ran,
        }
    }

    /// Runs `session` over the inputs set, as [`run`](Self::run) says.
    fn run_on(
        &self,
        session: &Arc<Mutex<Session>>,
        on_device: bool,
    ) -> Result<Vec<Tensor>, BackendError> {
        let mut session = session.lock().unwrap();
        if !on_device {
            let mut session_inputs: Vec<SessionInputValue<'_>> = vec![];
            for i in &self.inputs {
                session_inputs.extend(to_input_value(i)?);
            }
            let session_outputs = session.run(session_inputs.as_slice())?;
            let mut read = Vec::with_capacity(self.outputs.len());
            for (i, output) in self.outputs.iter().enumerate() {
                let (dimensions, data) = from_output_value(&session_outputs[i], output.shape.ty)?;
                if let Some(bound) = &output.bound {
                    bound.output.0.written(Some(&data))?;
                }
                read.push(Tensor {
                    dimensions,
                    ty: output.shape.ty,
                    data,
                    device: None,
                });
            }
            return Ok(read);
        }

        let mut binding = session.create_binding()?;
        for slot in &self.inputs {
            let Some(tensor) = &slot.tensor else {
                return Err(BackendError::BackendAccess(wasmtime::format_err!(
                    "missing input tensor: {}",
                    slot.shape.name
                )));
            };
            let Some(device) = &tensor.device else {
                binding.bind_input(&slot.shape.name, &host_value(tensor)?)?;
                continue;
            };
            if tensor.ty != TensorType::Fp32 {
                return Err(BackendError::BackendAccess(wasmtime::format_err!(
                    "an input in device memory is fp32, and {} is {:?}",
                    slot.shape.name,
                    tensor.ty
                )));
            }
            let (info, pointer) = memory_info(self.provider, device.0.memory(), &slot.shape.name)?;
            device.0.ready()?;
            let shape = ort::tensor::Shape::new(tensor.dimensions.iter().map(|&d| i64::from(d)));
            // SAFETY: `DeviceInput` promises device memory of the kind its
            // `memory` names holding the tensor's elements, alive as long as
            // the tensor is; the tensor is in `slot` until after the run.
            let value =
                unsafe { TensorRefMut::<f32>::from_raw(info, pointer as usize as *mut _, shape)? };
            binding.bind_input(&slot.shape.name, &*value)?;
        }
        let host = MemoryInfo::new(
            AllocationDevice::CPU,
            0,
            AllocatorType::Device,
            MemoryType::CPUOutput,
        )?;
        for output in &self.outputs {
            let device = output
                .bound
                .as_ref()
                .and_then(|b| b.output.0.memory().map(|m| (b, m)));
            match device {
                Some((bound, memory)) => {
                    let (info, pointer) = memory_info(self.provider, memory, &output.shape.name)?;
                    // SAFETY: `DeviceOutput` promises device memory of the
                    // kind `memory` names with room for the output's
                    // elements, alive as long as the binding is, which
                    // outlives the run.
                    let value = unsafe { device_value(&info, pointer, &bound.dimensions)? };
                    binding.bind_output(&output.shape.name, value)?;
                }
                None => binding.bind_output_to_device(&output.shape.name, &host)?,
            }
        }
        let session_outputs = session.run_binding(&binding)?;
        let mut read = Vec::with_capacity(self.outputs.len());
        for (i, output) in self.outputs.iter().enumerate() {
            match &output.bound {
                Some(bound) if bound.output.0.memory().is_some() => {
                    bound.output.0.written(None)?;
                    let dimensions = bound
                        .dimensions
                        .iter()
                        .map(convert_i64)
                        .collect::<Result<Vec<u32>, _>>()?;
                    read.push(Tensor::on_device(
                        dimensions,
                        TensorType::Fp32,
                        DeviceHandle(Arc::new(OutputRead(bound.output.clone()))),
                    ));
                }
                bound => {
                    let (dimensions, data) =
                        from_output_value(&session_outputs[i], output.shape.ty)?;
                    if let Some(bound) = bound {
                        bound.output.0.written(Some(&data))?;
                    }
                    read.push(Tensor {
                        dimensions,
                        ty: output.shape.ty,
                        data,
                        device: None,
                    });
                }
            }
        }
        Ok(read)
    }

    /// Helper function for finding the internal index of a tensor by [`Id`].
    fn find(&self, id: Id, list: &[TensorSlot]) -> Result<usize, BackendError> {
        let index = match id {
            Id::Index(i) => {
                let i = i as usize;
                if i < list.len() {
                    i
                } else {
                    return Err(BackendError::BackendAccess(wasmtime::format_err!(
                        "incorrect tensor index: {i} >= {}",
                        list.len()
                    )));
                }
            }
            Id::Name(n) => list.iter().position(|s| s.shape.name == n).ok_or_else(|| {
                BackendError::BackendAccess(wasmtime::format_err!("unknown tensor name: {n}"))
            })?,
        };
        Ok(index)
    }
}

impl BackendExecutionContext for OnnxExecutionContext {
    fn set_input(&mut self, id: Id, tensor: &Tensor) -> Result<(), BackendError> {
        let index = self.find(id, &self.inputs)?;
        let input = &mut self.inputs[index];
        if let Err(e) = input.shape.matches(tensor) {
            return Err(e.into());
        }
        // Hold the tensor data on the context until `compute` is called.
        input.tensor.replace(tensor.clone());
        Ok(())
    }

    fn compute(
        &mut self,
        inputs: Option<Vec<NamedTensor>>,
    ) -> Result<Option<Vec<NamedTensor>>, BackendError> {
        match inputs {
            // WIT
            Some(inputs) => {
                for slot in &mut self.inputs {
                    slot.tensor = None;
                }
                for input in &inputs {
                    let index = self
                        .inputs
                        .iter()
                        .position(|slot| slot.shape.name == input.name);
                    let index = match index {
                        Some(idx) => idx,
                        None => {
                            // Try to convert name to index
                            if let Ok(idx) = input.name.parse::<usize>() {
                                if idx < self.inputs.len() {
                                    idx
                                } else {
                                    return Err(BackendError::BackendAccess(
                                        wasmtime::format_err!("Input index out of range: {idx}"),
                                    ));
                                }
                            } else {
                                return Err(BackendError::BackendAccess(wasmtime::format_err!(
                                    "Unknown input tensor name: {}",
                                    input.name
                                )));
                            }
                        }
                    };

                    let input_slot = &mut self.inputs[index];
                    if let Err(e) = input_slot.shape.matches(&input.tensor) {
                        return Err(e.into());
                    }
                    input_slot.tensor.replace(input.tensor.clone());
                }

                let read = self.run()?;

                let mut output_tensors = Vec::new();
                for (output, tensor) in self.outputs.iter_mut().zip(read) {
                    output.tensor.replace(tensor.clone());
                    output_tensors.push(NamedTensor {
                        name: output.shape.name.clone(),
                        tensor,
                    });
                }
                Ok(Some(output_tensors))
            }

            // WITX
            None => {
                let read = self.run()?;
                for (output, tensor) in self.outputs.iter_mut().zip(read) {
                    output.tensor.replace(tensor);
                }
                Ok(None)
            }
        }
    }

    fn get_output(&mut self, id: Id) -> Result<Tensor, BackendError> {
        let index = self.find(id, &self.outputs)?;
        let output = &self.outputs[index];
        if let Some(tensor) = &output.tensor {
            Ok(tensor.clone())
        } else {
            Err(BackendError::BackendAccess(wasmtime::format_err!(
                "missing output tensor: {}; has `compute` been called?",
                output.shape.name
            )))
        }
    }

    fn bind_output_device(
        &mut self,
        name: &str,
        output: DeviceOutputHandle,
    ) -> Result<(), BackendError> {
        let index = self.find(Id::Name(name.to_string()), &self.outputs)?;
        let provider = self.provider;
        let slot = &mut self.outputs[index];
        if slot.shape.ty != TensorType::Fp32 {
            return Err(BackendError::BackendAccess(wasmtime::format_err!(
                "output {name} is {:?}, and only an fp32 output is bound to device memory",
                slot.shape.ty
            )));
        }
        let (width, height) = output.0.size();
        let dimensions = resolve(&slot.shape.dimensions, width, height);
        let elements: i64 = dimensions.iter().product();
        if elements != i64::from(width) * i64::from(height) {
            return Err(BackendError::BackendAccess(wasmtime::format_err!(
                "output {name} is {:?}, taken as {dimensions:?}: {elements} elements, \
                 where a {width}x{height} texture holds {}",
                slot.shape.dimensions,
                u64::from(width) * u64::from(height)
            )));
        }
        if let Some(memory) = output.0.memory() {
            memory_info(provider, memory, name)?;
        }
        slot.bound = Some(BoundOutput { output, dimensions });
        Ok(())
    }

    fn execution_provider(&self) -> Option<&'static str> {
        Some(self.provider.name())
    }

    fn session_key(&self) -> usize {
        Arc::as_ptr(&self.session) as usize
    }
}

/// How a session on `provider` takes device memory of kind `memory`, and
/// the pointer to hand it; refused where the session is not on that device.
fn memory_info(
    provider: Provider,
    memory: DeviceMemory,
    name: &str,
) -> Result<(MemoryInfo, u64), BackendError> {
    match (memory, provider) {
        (DeviceMemory::Cuda { pointer, ordinal }, Provider::Cuda) => Ok((
            MemoryInfo::new(
                AllocationDevice::CUDA,
                ordinal,
                AllocatorType::Device,
                MemoryType::Default,
            )?,
            pointer,
        )),
        (DeviceMemory::DirectMl { allocation }, Provider::DirectMl) => Ok((
            MemoryInfo::new(
                AllocationDevice::DIRECTML,
                0,
                AllocatorType::Device,
                MemoryType::Default,
            )?,
            allocation,
        )),
        (memory, provider) => Err(BackendError::BackendAccess(wasmtime::format_err!(
            "{name} is in device memory ({memory:?}), which a session on {} cannot reach",
            provider.name()
        ))),
    }
}

/// An output bound to device memory, and the shape it is bound with.
struct BoundOutput {
    output: DeviceOutputHandle,
    dimensions: Vec<i64>,
}

/// A graph's output dimensions with the dynamic ones given: the last is
/// `width`, the one before it `height`, and any others 1.
fn resolve(dimensions: &[i64], width: u32, height: u32) -> Vec<i64> {
    let last = dimensions.len();
    dimensions
        .iter()
        .enumerate()
        .map(|(i, &d)| match (is_dynamic_dimension(d), last - i) {
            (false, _) => d,
            (true, 1) => i64::from(width),
            (true, 2) => i64::from(height),
            (true, _) => 1,
        })
        .collect()
}

/// A value over device memory the embedder owns, for a session to write.
///
/// # Safety
/// `pointer` is memory of the kind `info` names, with room for the fp32
/// elements `dimensions` count, alive as long as the value is.
unsafe fn device_value(
    info: &MemoryInfo,
    pointer: u64,
    dimensions: &[i64],
) -> Result<DynValue, BackendError> {
    use ort::AsPointer;
    let elements: i64 = dimensions.iter().product();
    let mut value: *mut ort::sys::OrtValue = std::ptr::null_mut();
    // SAFETY: as the caller promises; the shape outlives the call.
    let status = unsafe {
        (ort::api().CreateTensorWithDataAsOrtValue)(
            info.ptr(),
            pointer as usize as *mut std::ffi::c_void,
            elements as usize * 4,
            dimensions.as_ptr(),
            dimensions.len(),
            ort::sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &mut value,
        )
    };
    // SAFETY: a status the API returned.
    unsafe { ort::error::status_to_result(status) }?;
    let value = std::ptr::NonNull::new(value).ok_or_else(|| {
        BackendError::BackendAccess(wasmtime::format_err!(
            "CreateTensorWithDataAsOrtValue made nothing"
        ))
    })?;
    // SAFETY: a live value made above, owned by the result.
    Ok(unsafe { DynValue::from_ptr(value, None) })
}

/// An output in device memory, as a tensor's elements a guest may read.
struct OutputRead(DeviceOutputHandle);

impl DeviceInput for OutputRead {
    fn memory(&self) -> DeviceMemory {
        self.0
            .0
            .memory()
            .expect("only an output written to device memory is handed out this way")
    }

    fn ready(&self) -> wasmtime::Result<()> {
        Ok(())
    }

    fn read(&self) -> wasmtime::Result<Vec<u8>> {
        self.0.0.read()
    }
}

impl From<ort::Error> for BackendError {
    fn from(e: ort::Error) -> Self {
        BackendError::BackendAccess(wasmtime::format_err!("{e}"))
    }
}

/// Holds a slot for ONNX session inputs and outputs.
///
/// TODO: it seems unfortunate that we have to "hold" some extra data per
/// session but in the input case, this is necessary for name-based indexing.
struct TensorSlot {
    shape: Shape,
    tensor: Option<Tensor>,
    /// An output bound to device memory the embedder holds.
    bound: Option<BoundOutput>,
}

/// Describes a tensor in ONNX terms.
struct Shape {
    name: String,
    dimensions: Vec<i64>,
    ty: TensorType,
}

impl Shape {
    fn from_onnx_input(input: &Input) -> Result<Self, BackendError> {
        let name = input.name.clone();
        let (dimensions, ty) = convert_value_type(&input.input_type)?;
        Ok(Self {
            name,
            dimensions,
            ty,
        })
    }

    fn from_onnx_output(output: &Output) -> Result<Self, BackendError> {
        let name = output.name.clone();
        let (dimensions, ty) = convert_value_type(&output.output_type)?;
        Ok(Self {
            name,
            dimensions,
            ty,
        })
    }

    fn matches(&self, tensor: &Tensor) -> wasmtime::Result<()> {
        if self.dimensions.len() != tensor.dimensions.len() {
            return Err(wasmtime::format_err!(
                "input tensor cardinality does not match model: {:?} != {:?}",
                self.dimensions,
                tensor.dimensions
            ));
        } else {
            for (&shape_dim, &tensor_dim) in self.dimensions.iter().zip(tensor.dimensions.iter()) {
                let tensor_dim = tensor_dim as i64;
                if !is_dynamic_dimension(shape_dim) && shape_dim != tensor_dim {
                    return Err(wasmtime::format_err!(
                        "input tensor dimensions do not match model: {:?} != {:?}",
                        self.dimensions,
                        tensor.dimensions
                    ));
                }
            }
        }
        if self.ty != tensor.ty {
            return Err(wasmtime::format_err!(
                "input tensor type does not match model: {:?} != {:?}",
                self.ty,
                tensor.ty
            ));
        }
        Ok(())
    }
}

fn convert_value_type(vt: &ValueType) -> Result<(Vec<i64>, TensorType), BackendError> {
    match vt {
        ValueType::Tensor { ty, shape, .. } => {
            let dimensions = shape.to_vec();
            let ty = (*ty).try_into()?;
            Ok((dimensions, ty))
        }
        _ => Err(BackendError::BackendAccess(wasmtime::format_err!(
            "unsupported input type: {vt:?}"
        ))),
    }
}

fn convert_i64(i: &i64) -> Result<u32, BackendError> {
    u32::try_from(*i).map_err(|d| -> BackendError {
        wasmtime::format_err!("unable to convert dimension to u32: {d}").into()
    })
}

impl TryFrom<TensorElementType> for TensorType {
    type Error = BackendError;
    fn try_from(ty: TensorElementType) -> Result<Self, Self::Error> {
        match ty {
            TensorElementType::Float32 => Ok(TensorType::Fp32),
            TensorElementType::Float64 => Ok(TensorType::Fp64),
            TensorElementType::Uint8 => Ok(TensorType::U8),
            TensorElementType::Int32 => Ok(TensorType::I32),
            TensorElementType::Int64 => Ok(TensorType::I64),
            _ => Err(BackendError::BackendAccess(wasmtime::format_err!(
                "unsupported tensor type: {ty:?}"
            ))),
        }
    }
}

/// A host tensor as a value an `IoBinding` takes.
fn host_value(tensor: &Tensor) -> Result<DynValue, BackendError> {
    let dimensions: Vec<i64> = tensor.dimensions.iter().map(|&d| i64::from(d)).collect();
    let value = match tensor.ty {
        TensorType::Fp32 => {
            OrtTensor::from_array((dimensions, bytes_to_f32_vec(tensor.data.clone())))
                .map(|t| t.into_dyn())
        }
        TensorType::I64 => {
            OrtTensor::from_array((dimensions, bytes_to_i64_vec(tensor.data.clone())))
                .map(|t| t.into_dyn())
        }
        TensorType::I32 => {
            OrtTensor::from_array((dimensions, bytes_to_i32_vec(tensor.data.clone())))
                .map(|t| t.into_dyn())
        }
        other => {
            return Err(BackendError::BackendAccess(wasmtime::format_err!(
                "ONNX inputs of type {other:?} are not supported"
            )));
        }
    };
    value.map_err(|e| {
        BackendError::BackendAccess(wasmtime::format_err!(
            "failed to create ONNX session input: {e}"
        ))
    })
}

fn to_input_value(slot: &TensorSlot) -> Result<[SessionInputValue<'_>; 1], BackendError> {
    match &slot.tensor {
        Some(tensor) => {
            let dimensions: Vec<i64> = tensor
                .dimensions
                .iter()
                .map(|d| *d as i64) // TODO: fewer conversions
                .collect();
            // An integer input is as ordinary as a float one: a model reading
            // a sample rate or a token id takes i64, and refusing it refuses
            // the model.
            let ort_tensor = match tensor.ty {
                TensorType::Fp32 => {
                    OrtTensor::from_array((dimensions, bytes_to_f32_vec(tensor.data.to_vec())))
                        .map(SessionInputValue::from)
                }
                TensorType::I64 => {
                    OrtTensor::from_array((dimensions, bytes_to_i64_vec(tensor.data.to_vec())))
                        .map(SessionInputValue::from)
                }
                TensorType::I32 => {
                    OrtTensor::from_array((dimensions, bytes_to_i32_vec(tensor.data.to_vec())))
                        .map(SessionInputValue::from)
                }
                other => {
                    return Err(BackendError::BackendAccess(wasmtime::format_err!(
                        "ONNX inputs of type {other:?} are not supported"
                    )));
                }
            }
            .map_err(|e| {
                BackendError::BackendAccess(wasmtime::format_err!(
                    "failed to create ONNX session input: {e}"
                ))
            })?;
            Ok([ort_tensor])
        }
        None => {
            return Err(BackendError::BackendAccess(wasmtime::format_err!(
                "missing input tensor: {}",
                slot.shape.name
            )));
        }
    }
}

pub fn f32_vec_to_bytes(data: Vec<f32>) -> Vec<u8> {
    data.into_iter().flat_map(f32::to_le_bytes).collect()
}

pub fn bytes_to_f32_vec(data: Vec<u8>) -> Vec<f32> {
    data.chunks(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

pub fn i64_vec_to_bytes(data: Vec<i64>) -> Vec<u8> {
    data.into_iter().flat_map(i64::to_le_bytes).collect()
}

pub fn bytes_to_i64_vec(data: Vec<u8>) -> Vec<i64> {
    data.chunks(8)
        .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

pub fn i32_vec_to_bytes(data: Vec<i32>) -> Vec<u8> {
    data.into_iter().flat_map(i32::to_le_bytes).collect()
}

pub fn bytes_to_i32_vec(data: Vec<u8>) -> Vec<i32> {
    data.chunks(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn dimensions_as_u32(shape: &ort::tensor::Shape) -> Result<Vec<u32>, BackendError> {
    (*shape)
        .iter()
        .map(|d| if *d == -1 { Ok(1) } else { convert_i64(d) })
        .collect()
}

/// One session output read back as the shape and little-endian bytes the
/// guest's own tensor type asks for.
fn from_output_value(
    value: &ort::value::Value,
    ty: TensorType,
) -> Result<(Vec<u32>, Vec<u8>), BackendError> {
    match ty {
        TensorType::Fp32 => {
            let (shape, data): (&ort::tensor::Shape, &[f32]) = value.try_extract_tensor()?;
            Ok((dimensions_as_u32(shape)?, f32_vec_to_bytes(data.to_vec())))
        }
        TensorType::I64 => {
            let (shape, data): (&ort::tensor::Shape, &[i64]) = value.try_extract_tensor()?;
            Ok((dimensions_as_u32(shape)?, i64_vec_to_bytes(data.to_vec())))
        }
        TensorType::I32 => {
            let (shape, data): (&ort::tensor::Shape, &[i32]) = value.try_extract_tensor()?;
            Ok((dimensions_as_u32(shape)?, i32_vec_to_bytes(data.to_vec())))
        }
        other => Err(BackendError::BackendAccess(wasmtime::format_err!(
            "ONNX outputs of type {other:?} are not supported"
        ))),
    }
}

/// Returns whether the dimension is dynamic.
///
/// ONNX uses [dimensional variables] (i.e., name strings) to indicate that the
/// value of a tensor dimension is user-defined, not fixed by the model. This is
/// useful for batching up several inference requests, e.g. When `ort` returns a
/// dimension of this kind, though, it uses `-1` to indicate that the dimension
/// is dynamic.
///
/// [dimensional variables]:
///     https://onnx.ai/onnx/repo-docs/IR.html#static-tensor-shapes
fn is_dynamic_dimension(d: i64) -> bool {
    d == -1
}
