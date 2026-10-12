//! Implements a `wasi-nn` [`BackendInner`] using ONNX via the `ort` crate.
//!
//! This file carries the vendored copy's changes to the ONNX backend, which
//! are upstreaming candidates: `ExecutionTarget::Gpu` resolves through a
//! [`Provider`] priority list the embedder supplies rather than always meaning
//! CUDA, a `Gpu` session whose providers all fail to register is an error
//! rather than a silent fall back to the CPU, integer tensors are
//! accepted alongside fp32, and an input whose elements are in CUDA memory
//! the embedder holds ([`crate::DeviceInput`]) is bound to a CUDA session
//! through an `IoBinding` instead of being copied into a CPU tensor.
//! Everything else here is upstream's.

use super::{
    BackendError, BackendExecutionContext, BackendFromDir, BackendGraph, BackendInner, NamedTensor,
};
use crate::backend::{Id, read};
use crate::wit::types::{ExecutionTarget, GraphEncoding, Tensor, TensorType};
use crate::{ExecutionContext, Graph};
use ort::{
    execution_providers::{CPUExecutionProvider, ExecutionProviderDispatch},
    memory::{AllocationDevice, AllocatorType, MemoryInfo, MemoryType},
    session::{Input, Output},
    session::{Session, SessionInputValue, builder::GraphOptimizationLevel},
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

pub struct OnnxBackend {
    /// What an `ExecutionTarget::Gpu` session asks for, most wanted first.
    gpu_providers: Vec<Provider>,
}

impl Default for OnnxBackend {
    fn default() -> Self {
        Self {
            gpu_providers: platform_gpu_providers(),
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
        Self { gpu_providers }
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
            let Some(dispatch) = provider.dispatch() else {
                tracing::debug!(
                    "{} execution provider is not enabled in this build",
                    provider.name()
                );
                refused.push((provider, "not enabled in this build".to_string()));
                continue;
            };
            let builder =
                match Session::builder()?.with_execution_providers([dispatch.error_on_failure()]) {
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
            });
        }
        let box_: Box<dyn BackendExecutionContext> = Box::new(OnnxExecutionContext {
            session: self.session.clone(),
            provider: self.provider,
            inputs,
            outputs,
        });
        Ok(box_.into())
    }
}

struct OnnxExecutionContext {
    session: Arc<Mutex<Session>>,
    provider: Provider,
    inputs: Vec<TensorSlot>,
    outputs: Vec<TensorSlot>,
}

unsafe impl Send for OnnxExecutionContext {}
unsafe impl Sync for OnnxExecutionContext {}

impl OnnxExecutionContext {
    /// Runs the session over the inputs set, and reads every output back as
    /// the shape and bytes its type asks for, in the session's order.
    ///
    /// With every input in host memory the session is run as upstream runs
    /// it. With any input in device memory the inputs are bound instead:
    /// that input's memory as it is, the others as CPU tensors, and the
    /// outputs to host memory, where they are read as before.
    fn run(&mut self) -> Result<Vec<(Vec<u32>, Vec<u8>)>, BackendError> {
        let on_device = self
            .inputs
            .iter()
            .any(|slot| slot.tensor.as_ref().is_some_and(|t| t.device.is_some()));
        let mut session = self.session.lock().unwrap();
        if !on_device {
            let mut session_inputs: Vec<SessionInputValue<'_>> = vec![];
            for i in &self.inputs {
                session_inputs.extend(to_input_value(i)?);
            }
            let session_outputs = session.run(session_inputs.as_slice())?;
            return self
                .outputs
                .iter()
                .enumerate()
                .map(|(i, output)| from_output_value(&session_outputs[i], output.shape.ty))
                .collect();
        }

        if self.provider != Provider::Cuda {
            return Err(BackendError::BackendAccess(wasmtime::format_err!(
                "an input in device memory is bound to a CUDA session, and this one is on {}",
                self.provider.name()
            )));
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
            let (pointer, ordinal) = device.0.cuda_pointer();
            device.0.ready()?;
            let memory = MemoryInfo::new(
                AllocationDevice::CUDA,
                ordinal,
                AllocatorType::Device,
                MemoryType::Default,
            )?;
            let shape = ort::tensor::Shape::new(tensor.dimensions.iter().map(|&d| i64::from(d)));
            // SAFETY: `DeviceInput` promises a CUDA allocation on device
            // `ordinal` holding the tensor's elements, alive as long as the
            // tensor is; the tensor is in `slot` until after the run below.
            let value = unsafe {
                TensorRefMut::<f32>::from_raw(memory, pointer as usize as *mut _, shape)?
            };
            binding.bind_input(&slot.shape.name, &*value)?;
        }
        let host = MemoryInfo::new(
            AllocationDevice::CPU,
            0,
            AllocatorType::Device,
            MemoryType::CPUOutput,
        )?;
        for output in &self.outputs {
            binding.bind_output_to_device(&output.shape.name, &host)?;
        }
        let session_outputs = session.run_binding(&binding)?;
        self.outputs
            .iter()
            .enumerate()
            .map(|(i, output)| from_output_value(&session_outputs[i], output.shape.ty))
            .collect()
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
                for (output, (dimensions, data)) in self.outputs.iter_mut().zip(read) {
                    let tensor = Tensor {
                        dimensions,
                        ty: output.shape.ty,
                        data,
                        device: None,
                    };
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
                for (output, (dimensions, data)) in self.outputs.iter_mut().zip(read) {
                    output.tensor.replace(Tensor {
                        dimensions,
                        ty: output.shape.ty,
                        data,
                        device: None,
                    });
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
