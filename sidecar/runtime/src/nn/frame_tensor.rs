//! A tick's frame, or crops of it, made into a model's input tensor, by
//! whichever route the model's session can take, and one line on stderr per
//! model saying which.
//!
//! A frame held on the GPU is made into the tensor by a pass on the device
//! ([`crate::gpu::tensor`]). Where the model's session is CUDA, the
//! process's device is D3D12 and the CUDA runtime finds the same adapter,
//! the pass writes a buffer on a shared heap that CUDA has imported, and
//! the tensor handed to `wasi:nn` is that buffer, bound to the session with
//! no copy: `[nn] <model>: input from the pool, bound on CUDA`. The session
//! runs on a stream of the host's, in which a wait for the pass is queued
//! ahead of the run, so the GPU orders the two and the CPU waits for
//! neither. Where the model's session is DirectML on the process's own
//! device and queue ([`crate::nn::directml`]), the pass's buffer is handed
//! to the session as a DirectML allocation, and the queue orders the pass
//! and the run: `[nn] <model>: input from the pool, bound on DIRECTML`.
//! Otherwise the pass's output is read back once and the tensor is ordinary
//! bytes: `[nn] <model>: input downloaded, <why>`.
//!
//! A frame in host memory is made into the tensor on the CPU, the same way
//! the pass makes it: `[nn] <model>: input in host memory, made on the CPU`.
//!
//! `FFRWD_NN_BIND=off` takes the downloaded route where the bound one would
//! be taken, so the two can be compared on one session.
//!
//! [`stack`] joins tensors made on several ticks into one: on the device,
//! bound the way its parts were, where every part is bound in place; in host
//! memory otherwise.

use std::collections::{HashMap, HashSet};
use std::sync::{Mutex, OnceLock};

use anyhow::{bail, Result};
use wasmtime_wasi_nn::backend::onnx::Provider;
use wasmtime_wasi_nn::wit::TensorType;
use wasmtime_wasi_nn::Tensor;

use crate::gpu::tensor::{host_tensor, to_bytes, Rect, TensorSpec};
use crate::node::FrameData;

/// How a model's inputs from frames on the GPU reach its session.
#[derive(Clone)]
enum Route {
    /// The pass's buffer, bound to a CUDA session in place.
    #[cfg(windows)]
    Bound(&'static bound::Binder),
    /// The pass's buffer, bound to a DirectML session on the process's
    /// device in place.
    #[cfg(windows)]
    DirectMl,
    /// The pass's buffer read back once; why it is not bound.
    Downloaded(String),
}

/// The route each model took, settled the first time it is asked for.
fn routes() -> &'static Mutex<HashMap<String, Route>> {
    static ROUTES: OnceLock<Mutex<HashMap<String, Route>>> = OnceLock::new();
    ROUTES.get_or_init(Default::default)
}

/// Says `[nn] <model>: <line>` on stderr, the first time it is said of the
/// model.
fn say(model: &str, line: &str) {
    static SAID: OnceLock<Mutex<HashSet<(String, String)>>> = OnceLock::new();
    let mut said = SAID
        .get_or_init(Default::default)
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    if said.insert((model.to_string(), line.to_string())) {
        eprintln!("[nn] {model}: {line}");
    }
}

/// The route for `model`, whose session engaged `provider`.
fn route(model: &str, provider: Provider) -> Route {
    let mut routes = routes().lock().unwrap_or_else(|e| e.into_inner());
    routes
        .entry(model.to_string())
        .or_insert_with(|| settle(provider))
        .clone()
}

/// The environment variable that, set to `off` (or `0`), keeps every model
/// on the downloaded route.
pub const BIND_VAR: &str = "FFRWD_NN_BIND";

fn settle(provider: Provider) -> Route {
    let off = std::env::var(BIND_VAR).is_ok_and(|v| v == "off" || v == "0");
    if off {
        return Route::Downloaded(format!("{BIND_VAR} is off"));
    }
    match provider {
        Provider::Cuda => {
            #[cfg(windows)]
            {
                match bound::Binder::get() {
                    Ok(binder) => Route::Bound(binder),
                    Err(e) => {
                        Route::Downloaded(format!("CUDA could not reach the pool's memory ({e:#})"))
                    }
                }
            }
            #[cfg(not(windows))]
            {
                Route::Downloaded(
                    "CUDA takes host memory here: the pool is imported into CUDA on Windows \
                     (D3D12) only, so far"
                        .to_string(),
                )
            }
        }
        #[cfg(windows)]
        Provider::DirectMl if crate::nn::directml::on_shared_device() => Route::DirectMl,
        Provider::DirectMl => Route::Downloaded(
            "DIRECTML takes host memory (its session runs on a device of its own, not the \
             pool's)"
                .to_string(),
        ),
        other => Route::Downloaded(format!("{} takes host memory", other.name())),
    }
}

/// Frame `frame`, a `width` x `height` `pix_fmt` frame of a stream, made into
/// the tensor `spec` says for the model whose binding is keyed `model`
/// ([`crate::nn::Scope::key`]): the whole frame,
/// or each of `crops` cut to the frame, one picture after another.
pub fn frame_tensor(
    frame: &FrameData,
    pix_fmt: &str,
    width: u32,
    height: u32,
    model: &str,
    spec: &TensorSpec,
    crops: Option<&[Rect]>,
) -> Result<Tensor> {
    let crops = match crops {
        Some(crops) => crate::gpu::tensor::cut(crops, width, height)?,
        None => vec![Rect::whole(width, height)],
    };
    spec.check(crops.len())?;
    let Some(provider) = crate::nn::provider_of(model) else {
        let bound = crate::nn::model_names();
        bail!(
            "no -nn binds a model called {model:?}; this run binds {}",
            if bound.is_empty() {
                "none".to_string()
            } else {
                bound.join(", ")
            }
        );
    };
    let dimensions = spec.dimensions(crops.len() as u32);
    let gpu = match frame {
        FrameData::Host(bytes) => {
            let values = host_tensor(bytes, pix_fmt, width, height, spec, &crops)?;
            say(model, "input in host memory, made on the CPU");
            return Ok(Tensor::new(dimensions, TensorType::Fp32, to_bytes(&values)));
        }
        FrameData::Gpu(gpu) => gpu,
    };
    let why = match route(model, provider) {
        #[cfg(windows)]
        Route::Bound(binder) => match binder.bind(gpu, spec, &crops) {
            Ok(tensor) => {
                say(model, "input from the pool, bound on CUDA");
                return Ok(tensor);
            }
            Err(e) => {
                // One refusal settles the model's route: every later frame
                // is downloaded, and the line says why.
                let why = format!("CUDA could not bind the pool's memory ({e:#})");
                routes()
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .insert(model.to_string(), Route::Downloaded(why.clone()));
                why
            }
        },
        #[cfg(windows)]
        Route::DirectMl => match bound::bind_directml(gpu, spec, &crops) {
            Ok(tensor) => {
                say(model, "input from the pool, bound on DIRECTML");
                return Ok(tensor);
            }
            Err(e) => {
                let why = format!("DIRECTML could not bind the pool's memory ({e:#})");
                routes()
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .insert(model.to_string(), Route::Downloaded(why.clone()));
                why
            }
        },
        Route::Downloaded(why) => why,
    };
    let made = gpu.tensor(spec, &crops, false)?;
    let bytes = made.read_back()?;
    say(model, &format!("input downloaded, {why}"));
    Ok(Tensor::new(dimensions, TensorType::Fp32, bytes))
}

/// How many elements a tensor of `dimensions` holds.
fn elements(dimensions: &[u32]) -> u64 {
    dimensions.iter().map(|&d| u64::from(d)).product()
}

/// `tensors`' elements one after another, in order, as one tensor of
/// `dimensions`. Where every tensor is bound to its session in place, the
/// same way, the joined tensor is copied together on the device and bound
/// that way; otherwise each tensor's elements are taken where they are, a
/// tensor on the device read back, and joined in host memory.
pub fn stack(tensors: &[&Tensor], dimensions: Vec<u32>) -> Result<Tensor> {
    let Some(first) = tensors.first() else {
        bail!("nothing to stack: the list is empty");
    };
    if let Some(at) = tensors.iter().position(|t| t.ty != first.ty) {
        bail!(
            "tensor {at} is {:?} and tensor 0 is {:?}; a stack is of one element type",
            tensors[at].ty,
            first.ty
        );
    }
    let held: u64 = tensors.iter().map(|t| elements(&t.dimensions)).sum();
    let asked = elements(&dimensions);
    if asked != held {
        bail!(
            "dimensions {dimensions:?} hold {asked} elements, and the {} tensors hold {held}",
            tensors.len()
        );
    }
    #[cfg(windows)]
    if let Some(joined) = bound::stack(tensors, &dimensions)? {
        return Ok(joined);
    }
    let mut bytes = Vec::new();
    for tensor in tensors {
        match &tensor.device {
            Some(device) => bytes.extend(device.0.read().map_err(|e| anyhow::anyhow!("{e:#}"))?),
            None => bytes.extend_from_slice(&tensor.data),
        }
    }
    Ok(Tensor::new(dimensions, first.ty, bytes))
}

/// The bound route: the process's D3D12 device, the CUDA runtime the CUDA
/// provider loaded, and the import of the device's shared sync object.
#[cfg(windows)]
pub(crate) mod bound {
    use std::sync::{Arc, OnceLock};
    use std::time::Instant;

    use anyhow::{anyhow, bail, Result};
    use wasmtime_wasi_nn::wit::TensorType;
    use wasmtime_wasi_nn::{DeviceHandle, DeviceInput, DeviceMemory, Tensor};

    use crate::gpu::d3d12::{adapter_luid, SharedSync};
    use crate::gpu::frames::GpuFrame;
    use crate::gpu::tensor::{DeviceTensor, Rect, TensorSpec};
    use crate::nn::cuda::{Cuda, CudaBuffer, CudaSemaphore, SESSION_DEVICE};
    use crate::profile::PROFILE;

    /// What binding a frame's tensor takes, made once a process.
    pub struct Binder {
        pub(crate) cuda: &'static Cuda,
        /// The device's shared sync object, as CUDA waits on it.
        pub(crate) semaphore: CudaSemaphore,
    }

    impl Binder {
        /// The process's binder, or why there is none: the first answer is
        /// every later one.
        pub fn get() -> Result<&'static Binder> {
            static MADE: OnceLock<std::result::Result<Binder, String>> = OnceLock::new();
            MADE.get_or_init(|| Binder::make().map_err(|e| format!("{e:#}")))
                .as_ref()
                .map_err(|e| anyhow!("{e}"))
        }

        fn make() -> Result<Binder> {
            let luid = adapter_luid().ok_or_else(|| {
                anyhow!("the process's device is not a D3D12 one, and only D3D12 is imported")
            })?;
            let path = crate::nn::cuda_runtime()
                .ok_or_else(|| anyhow!("the CUDA runtime the provider loads was not found"))?;
            let cuda = Cuda::open(&path)?;
            let ordinal = cuda.ordinal_of(luid)?;
            // The CUDA provider runs every session on device 0.
            if ordinal != SESSION_DEVICE {
                bail!(
                    "the pool's adapter is CUDA device {ordinal}, and the CUDA session runs on \
                     device {SESSION_DEVICE}"
                );
            }
            let sync = SharedSync::get()?;
            let semaphore = cuda.import_semaphore(sync.handle().raw())?;
            Ok(Binder { cuda, semaphore })
        }

        /// `crops` of `frame` made into `spec`'s tensor in a buffer CUDA has
        /// mapped, as a tensor whose elements are that buffer.
        pub fn bind(
            &'static self,
            frame: &GpuFrame,
            spec: &TensorSpec,
            crops: &[Rect],
        ) -> Result<Tensor> {
            let made = frame.tensor(spec, crops, true)?;
            let dimensions = made.dimensions();
            self.wrap(made, dimensions)
        }

        /// `made`, a tensor in a buffer on a shared heap, as a tensor of
        /// `dimensions` whose elements are that buffer as CUDA maps it,
        /// mapped the first time.
        fn wrap(&'static self, made: Arc<DeviceTensor>, dimensions: Vec<u32>) -> Result<Tensor> {
            let buffer = &made.buffer;
            let mapped = buffer
                .cuda
                .get_or_init(|| {
                    let handle = buffer
                        .shared
                        .as_ref()
                        .ok_or_else(|| "the buffer is not on a shared heap".to_string())?;
                    self.cuda
                        .import_buffer(handle.raw(), buffer.size)
                        .map_err(|e| format!("{e:#}"))
                })
                .as_ref()
                .map_err(|e| anyhow!("{e}"))?;
            let input = Bound {
                pointer: mapped.pointer,
                tensor: made.clone(),
                binder: self,
            };
            Ok(Tensor::on_device(
                dimensions,
                TensorType::Fp32,
                DeviceHandle(Arc::new(input)),
            ))
        }
    }

    /// `crops` of `frame` made into `spec`'s tensor in a buffer of the
    /// process's device, as a tensor whose elements are that buffer handed
    /// to DirectML.
    pub fn bind_directml(frame: &GpuFrame, spec: &TensorSpec, crops: &[Rect]) -> Result<Tensor> {
        let made = frame.tensor(spec, crops, true)?;
        let dimensions = made.dimensions();
        wrap_directml(made, dimensions)
    }

    /// `made`, a tensor in a buffer of the process's device, as a tensor of
    /// `dimensions` whose elements are that buffer handed to DirectML.
    fn wrap_directml(made: Arc<DeviceTensor>, dimensions: Vec<u32>) -> Result<Tensor> {
        let allocation = crate::nn::directml::allocation_of_buffer(&made.buffer)?;
        Ok(Tensor::on_device(
            dimensions,
            TensorType::Fp32,
            DeviceHandle(Arc::new(DirectMlInput {
                allocation: allocation.pointer(),
                tensor: made,
            })),
        ))
    }

    /// How a tensor is bound in place.
    #[derive(Clone, Copy)]
    enum Kind {
        Cuda(&'static Binder),
        DirectMl,
    }

    /// `tensors` copied together on the device as a tensor of `dimensions`,
    /// bound the way every one of them is; None where one is not bound in
    /// place, or they are bound two ways.
    pub fn stack(tensors: &[&Tensor], dimensions: &[u32]) -> Result<Option<Tensor>> {
        let mut parts = Vec::with_capacity(tensors.len());
        let mut kind = None;
        for tensor in tensors {
            let Some(device) = &tensor.device else {
                return Ok(None);
            };
            let input: &dyn std::any::Any = device.0.as_ref();
            let (this, part) = if let Some(bound) = input.downcast_ref::<Bound>() {
                (Kind::Cuda(bound.binder), &bound.tensor)
            } else if let Some(dml) = input.downcast_ref::<DirectMlInput>() {
                (Kind::DirectMl, &dml.tensor)
            } else {
                return Ok(None);
            };
            match (kind, this) {
                (None, _) => kind = Some(this),
                (Some(Kind::Cuda(_)), Kind::Cuda(_)) | (Some(Kind::DirectMl), Kind::DirectMl) => {}
                _ => return Ok(None),
            }
            parts.push(Arc::clone(part));
        }
        let joined = Arc::new(crate::gpu::tensor::join(&parts, true)?);
        let dimensions = dimensions.to_vec();
        match kind {
            Some(Kind::Cuda(binder)) => binder.wrap(joined, dimensions).map(Some),
            Some(Kind::DirectMl) => wrap_directml(joined, dimensions).map(Some),
            None => Ok(None),
        }
    }

    /// A frame's tensor as a DirectML session binds it. The session runs on
    /// the queue the pass was given to, after it, so nothing waits.
    struct DirectMlInput {
        allocation: u64,
        tensor: Arc<DeviceTensor>,
    }

    impl DeviceInput for DirectMlInput {
        fn memory(&self) -> DeviceMemory {
            DeviceMemory::DirectMl {
                allocation: self.allocation,
            }
        }

        fn ready(&self) -> wasmtime::Result<()> {
            Ok(())
        }

        fn read(&self) -> wasmtime::Result<Vec<u8>> {
            self.tensor
                .read_back()
                .map_err(|e| wasmtime::format_err!("{e:#}"))
        }
    }

    /// A frame's tensor as a session binds it.
    struct Bound {
        pointer: u64,
        tensor: Arc<DeviceTensor>,
        binder: &'static Binder,
    }

    impl Bound {
        fn mapped(&self) -> Result<&CudaBuffer> {
            match self.tensor.buffer.cuda.get() {
                Some(Ok(mapped)) => Ok(mapped),
                _ => bail!("the tensor's buffer is not mapped into CUDA"),
            }
        }
    }

    impl DeviceInput for Bound {
        fn memory(&self) -> DeviceMemory {
            DeviceMemory::Cuda {
                pointer: self.pointer,
                ordinal: SESSION_DEVICE,
            }
        }

        /// The session waits for the pass in its own stream, so the GPU
        /// orders the two; without that stream the CPU waits.
        fn ready(&self) -> wasmtime::Result<()> {
            let Some(value) = self.tensor.written_at else {
                return Ok(());
            };
            let waited = match crate::nn::session_stream() {
                Some(stream) => self.binder.semaphore.wait_in(stream, value),
                None => self.binder.semaphore.wait(value),
            };
            waited.map_err(|e| wasmtime::format_err!("{e:#}"))
        }

        fn read(&self) -> wasmtime::Result<Vec<u8>> {
            if let Some(value) = self.tensor.written_at {
                self.binder
                    .semaphore
                    .wait(value)
                    .map_err(|e| wasmtime::format_err!("{e:#}"))?;
            }
            let started = Instant::now();
            let bytes = self
                .mapped()
                .and_then(|m| m.read())
                .map_err(|e| wasmtime::format_err!("{e:#}"))?;
            let len = self.tensor.byte_len() as usize;
            PROFILE.download.add(len, started);
            Ok(bytes[..len].to_vec())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn host(dimensions: &[u32], first: u8) -> Tensor {
        let len = 4 * elements(dimensions) as usize;
        Tensor::new(
            dimensions.to_vec(),
            TensorType::Fp32,
            (0..len).map(|i| first.wrapping_add(i as u8)).collect(),
        )
    }

    #[test]
    fn tensors_in_host_memory_are_joined_in_the_order_named_with_the_dimensions_asked() {
        let (a, b) = (host(&[1, 3, 2, 2], 0), host(&[2, 3, 2, 2], 100));
        let joined = stack(&[&a, &b, &a], vec![1, 4, 3, 2, 2]).unwrap();
        assert_eq!(joined.dimensions, [1, 4, 3, 2, 2]);
        assert_eq!(joined.ty, TensorType::Fp32);
        assert!(joined.device.is_none());
        let want: Vec<u8> = [&a.data, &b.data, &a.data]
            .into_iter()
            .flatten()
            .copied()
            .collect();
        assert_eq!(joined.data, want);
    }

    #[test]
    fn a_stack_of_nothing_of_two_types_or_of_the_wrong_size_is_refused() {
        let a = host(&[1, 3, 2, 2], 0);
        let err = stack(&[], vec![1]).unwrap_err().to_string();
        assert!(err.contains("the list is empty"), "{err}");
        let mut other = host(&[1, 3, 2, 2], 0);
        other.ty = TensorType::Fp16;
        let err = stack(&[&a, &other], vec![2, 3, 2, 2])
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("tensor 1") && err.contains("one element type"),
            "{err}"
        );
        let err = stack(&[&a, &a], vec![1, 3, 2, 2]).unwrap_err().to_string();
        assert!(
            err.contains("hold 12 elements") && err.contains("hold 24"),
            "{err}"
        );
    }
}
