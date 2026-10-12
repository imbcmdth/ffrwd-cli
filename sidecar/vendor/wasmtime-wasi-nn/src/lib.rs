pub mod backend;
mod registry;
pub mod wit;
pub mod witx;

use crate::backend::{BackendError, Id, NamedTensor as BackendNamedTensor};
use crate::wit::generated_::wasi::nn::tensor::TensorType;
use core::fmt;
pub use registry::{GraphRegistry, InMemoryRegistry};
use std::path::Path;
use std::sync::Arc;
use wasmtime::format_err;

/// Construct an in-memory registry from the available backends and a list of
/// `(<backend name>, <graph directory>)`. This assumes graphs can be loaded
/// from a local directory, which is a safe assumption currently for the current
/// model types.
pub fn preload(preload_graphs: &[(String, String)]) -> wasmtime::Result<(Vec<Backend>, Registry)> {
    let mut backends = backend::list();
    let mut registry = InMemoryRegistry::new();
    for (kind, path) in preload_graphs {
        let kind_ = kind.parse()?;
        let backend = backends
            .iter_mut()
            .find(|b| b.encoding() == kind_)
            .ok_or(format_err!("unsupported backend: {kind}"))?
            .as_dir_loadable()
            .ok_or(format_err!("{kind} does not support directory loading"))?;
        registry.load(backend, Path::new(path))?;
    }
    Ok((backends, Registry::from(registry)))
}

/// A machine learning backend.
pub struct Backend(Box<dyn backend::BackendInner>);
impl std::ops::Deref for Backend {
    type Target = dyn backend::BackendInner;
    fn deref(&self) -> &Self::Target {
        self.0.as_ref()
    }
}
impl std::ops::DerefMut for Backend {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.0.as_mut()
    }
}
impl<T: backend::BackendInner + 'static> From<T> for Backend {
    fn from(value: T) -> Self {
        Self(Box::new(value))
    }
}

/// A backend-defined graph (i.e., ML model).
#[derive(Clone)]
pub struct Graph(Arc<dyn backend::BackendGraph>);
impl From<Box<dyn backend::BackendGraph>> for Graph {
    fn from(value: Box<dyn backend::BackendGraph>) -> Self {
        Self(value.into())
    }
}
impl std::ops::Deref for Graph {
    type Target = dyn backend::BackendGraph;
    fn deref(&self) -> &Self::Target {
        self.0.as_ref()
    }
}

/// A host-side tensor.
///
/// Eventually, this may be defined in each backend as they gain the ability to
/// hold tensors on various devices (TODO:
/// https://github.com/WebAssembly/wasi-nn/pull/70).
#[derive(Clone, PartialEq)]
pub struct Tensor {
    pub dimensions: Vec<u32>,
    pub ty: TensorType,
    pub data: Vec<u8>,
    /// Where the elements are when they are in device memory the embedder
    /// holds rather than in `data`, which is then empty. A backend that can
    /// bind such memory reads it in place; see [`DeviceInput`].
    pub device: Option<DeviceHandle>,
}
impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tensor")
            .field("dimensions", &self.dimensions)
            .field("ty", &self.ty)
            .field("data (bytes)", &self.data.len())
            .field("on a device", &self.device.is_some())
            .finish()
    }
}

/// Where a tensor's elements are in device memory the embedder holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceMemory {
    /// A CUDA device pointer, on the device of this ordinal.
    Cuda { pointer: u64, ordinal: i32 },
    /// An allocation DirectML made over a D3D12 resource
    /// (`CreateGPUAllocationFromD3DResource`), as the pointer it returned.
    DirectMl { allocation: u64 },
}

/// An input tensor's elements in device memory the embedder owns, which a
/// backend binds to its session in place of bytes. The tensor carries a
/// handle to it, so the wasi-nn tensor itself stays bytes or nothing.
pub trait DeviceInput: Send + Sync {
    /// Where the elements are, packed as the tensor's dimensions and type
    /// say.
    fn memory(&self) -> DeviceMemory;

    /// Called before the session runs: whatever writes the elements is
    /// ordered before the run, by a wait the run's own queue or stream
    /// takes or by returning once they are written.
    fn ready(&self) -> wasmtime::Result<()>;

    /// The elements in host memory, little-endian, for a reader asking for
    /// the tensor's bytes.
    fn read(&self) -> wasmtime::Result<Vec<u8>>;
}

/// An output the embedder holds device memory for, bound to an execution
/// context so every run writes it there rather than into host memory.
pub trait DeviceOutput: Send + Sync {
    /// Where the session writes the elements; `None` when the session
    /// cannot write device memory, so the run writes host memory and hands
    /// the bytes to [`DeviceOutput::written`].
    fn memory(&self) -> Option<DeviceMemory>;

    /// The picture the elements are, width and height: the output has
    /// `width * height` fp32 elements, and its dynamic dimensions are
    /// taken as the last the width, the one before it the height, and any
    /// others 1.
    fn size(&self) -> (u32, u32);

    /// Called before the session runs: whatever reads the last run's
    /// elements is ordered before this run writes them.
    fn ready(&self) -> wasmtime::Result<()>;

    /// Called once the run is given to the device: the elements, in host
    /// memory when [`DeviceOutput::memory`] is `None`.
    fn written(&self, host: Option<&[u8]>) -> wasmtime::Result<()>;

    /// The elements of the last run in host memory, little-endian.
    fn read(&self) -> wasmtime::Result<Vec<u8>>;
}

/// An execution context's [`DeviceOutput`].
#[derive(Clone)]
pub struct DeviceOutputHandle(pub Arc<dyn DeviceOutput>);

/// A tensor's [`DeviceInput`], shared by every copy of the tensor.
#[derive(Clone)]
pub struct DeviceHandle(pub Arc<dyn DeviceInput>);

impl PartialEq for DeviceHandle {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

/// A backend-defined execution context.
pub struct ExecutionContext(Box<dyn backend::BackendExecutionContext>);
impl From<Box<dyn backend::BackendExecutionContext>> for ExecutionContext {
    fn from(value: Box<dyn backend::BackendExecutionContext>) -> Self {
        Self(value)
    }
}
impl std::ops::Deref for ExecutionContext {
    type Target = dyn backend::BackendExecutionContext;
    fn deref(&self) -> &Self::Target {
        self.0.as_ref()
    }
}
impl std::ops::DerefMut for ExecutionContext {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.0.as_mut()
    }
}

/// A container for graphs.
pub struct Registry(Box<dyn GraphRegistry>);
impl std::ops::Deref for Registry {
    type Target = dyn GraphRegistry;
    fn deref(&self) -> &Self::Target {
        self.0.as_ref()
    }
}
impl std::ops::DerefMut for Registry {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.0.as_mut()
    }
}
impl<T> From<T> for Registry
where
    T: GraphRegistry + 'static,
{
    fn from(value: T) -> Self {
        Self(Box::new(value))
    }
}

impl ExecutionContext {
    pub fn set_input(&mut self, id: Id, tensor: &Tensor) -> Result<(), BackendError> {
        self.0.set_input(id, tensor)
    }

    pub fn compute(&mut self) -> Result<(), BackendError> {
        self.0.compute(None).map(|_| ())
    }

    pub fn get_output(&mut self, id: Id) -> Result<Tensor, BackendError> {
        self.0.get_output(id)
    }

    /// Binds output `name` to device memory the embedder holds, for every
    /// later run; see [`DeviceOutput`].
    pub fn bind_output_device(
        &mut self,
        name: &str,
        output: DeviceOutputHandle,
    ) -> Result<(), BackendError> {
        self.0.bind_output_device(name, output)
    }

    pub fn compute_with_io(
        &mut self,
        inputs: Vec<BackendNamedTensor>,
    ) -> Result<Vec<BackendNamedTensor>, BackendError> {
        match self.0.compute(Some(inputs))? {
            Some(outputs) => Ok(outputs),
            None => Ok(Vec::new()),
        }
    }
}

impl Tensor {
    pub fn new(dimensions: Vec<u32>, ty: TensorType, data: Vec<u8>) -> Self {
        Self {
            dimensions,
            ty,
            data,
            device: None,
        }
    }

    /// A tensor whose elements are in device memory: `dimensions` and `ty`
    /// describe them, and `device` reaches them.
    pub fn on_device(dimensions: Vec<u32>, ty: TensorType, device: DeviceHandle) -> Self {
        Self {
            dimensions,
            ty,
            data: Vec::new(),
            device: Some(device),
        }
    }
}
