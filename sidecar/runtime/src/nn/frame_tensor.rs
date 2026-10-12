//! A tick's frame made into a model's input tensor, by whichever route the
//! model's session can take, and one line on stderr per model saying which.
//!
//! A frame held on the GPU is made into the tensor by a pass on the device
//! ([`crate::gpu::tensor`]). Where the model's session is CUDA, the
//! process's device is D3D12 and the CUDA runtime finds the same adapter,
//! the pass writes a buffer on a shared heap that CUDA has imported, and
//! the tensor handed to `wasi:nn` is that buffer, bound to the session with
//! no copy: `[nn] <model>: input from the pool, bound on CUDA`. Otherwise
//! the pass's output is read back once and the tensor is ordinary bytes:
//! `[nn] <model>: input downloaded, <why>`. DirectML is one of those: the
//! pinned `ort` (2.0.0-rc.10) carries none of DirectML's interop API, and
//! its DirectML provider takes a device id and never the caller's D3D12
//! device, so a DirectML session cannot be handed a D3D12 buffer.
//!
//! A frame in host memory is made into the tensor on the CPU, the same way
//! the pass makes it: `[nn] <model>: input in host memory, made on the CPU`.

use std::collections::{HashMap, HashSet};
use std::sync::{Mutex, OnceLock};

use anyhow::{bail, Result};
use wasmtime_wasi_nn::backend::onnx::Provider;
use wasmtime_wasi_nn::wit::TensorType;
use wasmtime_wasi_nn::Tensor;

use crate::gpu::tensor::{host_tensor, to_bytes, TensorSpec};
use crate::node::FrameData;

/// How a model's inputs from frames on the GPU reach its session.
#[derive(Clone)]
enum Route {
    /// The pass's buffer, bound to a CUDA session in place.
    #[cfg(windows)]
    Bound(&'static bound::Binder),
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

fn settle(provider: Provider) -> Route {
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
        Provider::DirectMl => Route::Downloaded(
            "DIRECTML takes host memory (the pinned ort cannot hand a D3D12 buffer to a \
             DirectML session)"
                .to_string(),
        ),
        other => Route::Downloaded(format!("{} takes host memory", other.name())),
    }
}

/// Frame `frame`, a `width` x `height` `pix_fmt` frame of a stream, made into
/// the tensor `spec` says for the model bound as `model`.
pub fn frame_tensor(
    frame: &FrameData,
    pix_fmt: &str,
    width: u32,
    height: u32,
    model: &str,
    spec: &TensorSpec,
) -> Result<Tensor> {
    spec.check()?;
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
    let dimensions = spec.dimensions();
    let gpu = match frame {
        FrameData::Host(bytes) => {
            let values = host_tensor(bytes, pix_fmt, width, height, spec)?;
            say(model, "input in host memory, made on the CPU");
            return Ok(Tensor::new(dimensions, TensorType::Fp32, to_bytes(&values)));
        }
        FrameData::Gpu(gpu) => gpu,
    };
    let why = match route(model, provider) {
        #[cfg(windows)]
        Route::Bound(binder) => match binder.bind(gpu, spec) {
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
        Route::Downloaded(why) => why,
    };
    let made = gpu.tensor(spec, false)?;
    let bytes = made.read_back()?;
    say(model, &format!("input downloaded, {why}"));
    Ok(Tensor::new(dimensions, TensorType::Fp32, bytes))
}

/// The bound route: the process's D3D12 device, the CUDA runtime the CUDA
/// provider loaded, and the import of the device's shared sync object.
#[cfg(windows)]
mod bound {
    use std::sync::{Arc, OnceLock};
    use std::time::Instant;

    use anyhow::{anyhow, bail, Result};
    use wasmtime_wasi_nn::wit::TensorType;
    use wasmtime_wasi_nn::{DeviceHandle, DeviceInput, Tensor};

    use crate::gpu::d3d12::{adapter_luid, SharedSync};
    use crate::gpu::frames::GpuFrame;
    use crate::gpu::tensor::{DeviceTensor, TensorSpec};
    use crate::nn::cuda::{Cuda, CudaBuffer, CudaSemaphore};
    use crate::profile::PROFILE;

    /// What binding a frame's tensor takes, made once a process.
    pub struct Binder {
        cuda: &'static Cuda,
        /// The device's shared sync object, as CUDA waits on it.
        semaphore: CudaSemaphore,
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
            let cuda = Cuda::open(&path, luid)?;
            // The CUDA provider runs every session on device 0.
            if cuda.device != 0 {
                bail!(
                    "the pool's adapter is CUDA device {}, and the CUDA session runs on device 0",
                    cuda.device
                );
            }
            let sync = SharedSync::get()?;
            let semaphore = cuda.import_semaphore(sync.handle().raw())?;
            Ok(Binder { cuda, semaphore })
        }

        /// `frame` made into `spec`'s tensor in a buffer CUDA has mapped, as
        /// a tensor whose elements are that buffer.
        pub fn bind(&'static self, frame: &GpuFrame, spec: &TensorSpec) -> Result<Tensor> {
            let made = frame.tensor(spec, true)?;
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
                spec.dimensions(),
                TensorType::Fp32,
                DeviceHandle(Arc::new(input)),
            ))
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
        fn cuda_pointer(&self) -> (u64, i32) {
            (self.pointer, self.binder.cuda.device)
        }

        fn ready(&self) -> wasmtime::Result<()> {
            if let Some(value) = self.tensor.written_at {
                self.binder
                    .semaphore
                    .wait(value)
                    .map_err(|e| wasmtime::format_err!("{e:#}"))?;
            }
            Ok(())
        }

        fn read(&self) -> wasmtime::Result<Vec<u8>> {
            self.ready()?;
            let started = Instant::now();
            let bytes = self
                .mapped()
                .and_then(|m| m.read())
                .map_err(|e| wasmtime::format_err!("{e:#}"))?;
            let len = self.tensor.spec.len() as usize;
            PROFILE.download.add(len, started);
            Ok(bytes[..len].to_vec())
        }
    }
}
