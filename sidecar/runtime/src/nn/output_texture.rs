//! A model's output bound to a texture of the process's device
//! (`ffrwd:gpu/nn.output-texture`), by whichever route the session can
//! take, and one line on stderr per model and output saying which.
//!
//! On a CUDA session, where the process's device is D3D12 and the CUDA
//! runtime finds the same adapter, the session writes the output into a
//! buffer on a shared heap that CUDA imported, in the stream it runs on;
//! the stream then signals a shared sync object the queue waits on, and a
//! pass copies the buffer into the texture: `[nn] <model>: output <name>
//! to a texture, bound on CUDA`. Before the next run the stream waits for
//! that pass, so the run never writes over what is still being copied.
//!
//! On a DirectML session on the process's own device and queue, the
//! session writes a buffer of that device and the pass copies it in; the
//! queue orders the two: `... bound on DIRECTML`.
//!
//! Elsewhere the output comes back to host memory and is uploaded into the
//! texture once: `[nn] <model>: output <name> uploaded to a texture,
//! <why>`.

use std::sync::{Arc, Mutex, OnceLock};

use anyhow::{anyhow, bail, Result};
use wasmtime_wasi_nn::backend::onnx::Provider;
use wasmtime_wasi_nn::{DeviceMemory, DeviceOutput, DeviceOutputHandle, ExecutionContext};

use crate::gpu::output::OutputTexture;

/// How an output reaches its texture.
enum Route {
    /// Written by a CUDA session into a buffer CUDA mapped.
    #[cfg(windows)]
    Cuda(cuda::CudaOutput),
    /// Written by a DirectML session into a buffer of the device.
    #[cfg(windows)]
    DirectMl(u64),
    /// Uploaded from host memory; the last run's elements.
    Host(Mutex<Vec<u8>>),
}

/// One output bound to a texture.
pub struct BoundTexture {
    pub target: OutputTexture,
    route: Route,
    /// Keeps the DirectML allocation over the buffer alive.
    #[cfg(windows)]
    _allocation: Option<Arc<crate::nn::directml::Allocation>>,
}

impl BoundTexture {
    pub fn size(&self) -> (u32, u32) {
        (self.target.width, self.target.height)
    }
}

/// Binds output `name` of `context` to a `width` x `height` texture.
pub fn bind(
    context: &mut ExecutionContext,
    name: &str,
    width: u32,
    height: u32,
) -> Result<Arc<BoundTexture>> {
    if width == 0 || height == 0 {
        bail!("a {width}x{height} texture has no texels");
    }
    let model =
        crate::nn::model_of_session(context.session_key()).unwrap_or_else(|| "a model".to_string());
    let provider = match context.execution_provider() {
        Some("CUDA") => Provider::Cuda,
        Some("DIRECTML") => Provider::DirectMl,
        Some("COREML") => Provider::CoreMl,
        _ => Provider::Cpu,
    };
    let (bound, line) = make(provider, width, height)?;
    let bound = Arc::new(bound);
    context
        .bind_output_device(name, DeviceOutputHandle(bound.clone()))
        .map_err(|e| anyhow!("{e}"))?;
    say(&model, &format!("output {name} {line}"));
    Ok(bound)
}

/// Says `[nn] <model>: <line>` on stderr, the first time it is said.
fn say(model: &str, line: &str) {
    static SAID: OnceLock<Mutex<std::collections::HashSet<String>>> = OnceLock::new();
    let mut said = SAID
        .get_or_init(Default::default)
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let line = format!("[nn] {model}: {line}");
    if said.insert(line.clone()) {
        eprintln!("{line}");
    }
}

/// The texture and its route for a session on `provider`, and what the
/// line says of it.
fn make(provider: Provider, width: u32, height: u32) -> Result<(BoundTexture, String)> {
    let host = |why: String| -> Result<(BoundTexture, String)> {
        Ok((
            BoundTexture {
                target: OutputTexture::new(width, height, false)?,
                route: Route::Host(Mutex::new(Vec::new())),
                #[cfg(windows)]
                _allocation: None,
            },
            format!("uploaded to a texture, {why}"),
        ))
    };
    if std::env::var(super::frame_tensor::BIND_VAR).is_ok_and(|v| v == "off" || v == "0") {
        return host(format!("{} is off", super::frame_tensor::BIND_VAR));
    }
    match provider {
        #[cfg(windows)]
        Provider::Cuda => {
            let target = OutputTexture::new(width, height, true)?;
            match cuda::CudaOutput::new(&target) {
                Ok(output) => Ok((
                    BoundTexture {
                        target,
                        route: Route::Cuda(output),
                        _allocation: None,
                    },
                    "to a texture, bound on CUDA".to_string(),
                )),
                Err(e) => host(format!("CUDA could not reach the device's memory ({e:#})")),
            }
        }
        #[cfg(windows)]
        Provider::DirectMl => {
            let target = OutputTexture::new(width, height, true)?;
            match crate::nn::directml::allocation_of(&target) {
                Ok(allocation) => Ok((
                    BoundTexture {
                        target,
                        route: Route::DirectMl(allocation.pointer()),
                        _allocation: Some(allocation),
                    },
                    "to a texture, bound on DIRECTML".to_string(),
                )),
                Err(e) => host(format!("DIRECTML writes host memory ({e:#})")),
            }
        }
        #[cfg(not(windows))]
        Provider::Cuda => host(
            "CUDA writes host memory here: the device's memory is imported into CUDA on \
             Windows (D3D12) only, so far"
                .to_string(),
        ),
        other => host(format!("{} writes host memory", other.name())),
    }
}

impl DeviceOutput for BoundTexture {
    fn memory(&self) -> Option<DeviceMemory> {
        match &self.route {
            #[cfg(windows)]
            Route::Cuda(output) => Some(DeviceMemory::Cuda {
                pointer: output.pointer(),
                ordinal: crate::nn::cuda::SESSION_DEVICE,
            }),
            #[cfg(windows)]
            Route::DirectMl(allocation) => Some(DeviceMemory::DirectMl {
                allocation: *allocation,
            }),
            Route::Host(_) => None,
        }
    }

    fn size(&self) -> (u32, u32) {
        BoundTexture::size(self)
    }

    fn ready(&self) -> wasmtime::Result<()> {
        match &self.route {
            #[cfg(windows)]
            Route::Cuda(output) => output.ready().map_err(|e| wasmtime::format_err!("{e:#}")),
            _ => Ok(()),
        }
    }

    fn written(&self, host: Option<&[u8]>) -> wasmtime::Result<()> {
        let done = match (&self.route, host) {
            #[cfg(windows)]
            (Route::Cuda(output), _) => output.written(&self.target),
            #[cfg(windows)]
            (Route::DirectMl(_), _) => self.target.copy_in(),
            (Route::Host(kept), Some(bytes)) => {
                *kept.lock().unwrap_or_else(|e| e.into_inner()) = bytes.to_vec();
                self.target.upload(bytes)
            }
            (Route::Host(_), None) => Err(anyhow!("the run wrote no host memory to upload")),
        };
        done.map_err(|e| wasmtime::format_err!("{e:#}"))
    }

    fn read(&self) -> wasmtime::Result<Vec<u8>> {
        let read: Result<Vec<u8>> = match &self.route {
            #[cfg(windows)]
            Route::Cuda(output) => output.read(),
            #[cfg(windows)]
            Route::DirectMl(_) => self.target.read_buffer(),
            Route::Host(kept) => Ok(kept.lock().unwrap_or_else(|e| e.into_inner()).clone()),
        };
        read.map_err(|e| wasmtime::format_err!("{e:#}"))
    }
}

/// The CUDA route: the buffer CUDA mapped, and the sync objects each way.
#[cfg(windows)]
mod cuda {
    use std::sync::{Arc, Mutex, OnceLock};
    use std::time::Instant;

    use anyhow::{anyhow, Result};

    use crate::gpu::d3d12::{ForeignSync, SharedSync};
    use crate::gpu::output::OutputTexture;
    use crate::gpu::tensor::PooledBuffer;
    use crate::nn::cuda::CudaSemaphore;
    use crate::nn::frame_tensor::bound::Binder;
    use crate::profile::PROFILE;

    /// The sync object CUDA signals after a run, as CUDA signals it: made
    /// once a process.
    fn from_cuda(binder: &'static Binder) -> Result<&'static CudaSemaphore> {
        static MADE: OnceLock<std::result::Result<CudaSemaphore, String>> = OnceLock::new();
        MADE.get_or_init(|| {
            let sync = ForeignSync::get().map_err(|e| format!("{e:#}"))?;
            binder
                .cuda
                .import_semaphore(sync.handle().raw())
                .map_err(|e| format!("{e:#}"))
        })
        .as_ref()
        .map_err(|e| anyhow!("{e}"))
    }

    pub struct CudaOutput {
        binder: &'static Binder,
        buffer: Arc<PooledBuffer>,
        signals: &'static CudaSemaphore,
        pointer: u64,
        /// The shared sync object's value once the last copy has run.
        copied_at: Mutex<Option<u64>>,
    }

    impl CudaOutput {
        pub fn new(target: &OutputTexture) -> Result<CudaOutput> {
            let binder = Binder::get()?;
            let buffer = target
                .buffer
                .as_ref()
                .ok_or_else(|| anyhow!("the output has no buffer"))?;
            let mapped = buffer
                .cuda
                .get_or_init(|| {
                    let handle = buffer
                        .shared
                        .as_ref()
                        .ok_or_else(|| "the buffer is not on a shared heap".to_string())?;
                    binder
                        .cuda
                        .import_buffer(handle.raw(), buffer.size)
                        .map_err(|e| format!("{e:#}"))
                })
                .as_ref()
                .map_err(|e| anyhow!("{e}"))?;
            Ok(CudaOutput {
                binder,
                buffer: Arc::clone(buffer),
                signals: from_cuda(binder)?,
                pointer: mapped.pointer,
                copied_at: Mutex::new(None),
            })
        }

        pub fn pointer(&self) -> u64 {
            self.pointer
        }

        /// The run waits, in the session's stream, for the last copy out of
        /// the buffer; without that stream the CPU waits.
        pub fn ready(&self) -> Result<()> {
            let Some(value) = *self.copied_at.lock().unwrap_or_else(|e| e.into_inner()) else {
                return Ok(());
            };
            match crate::nn::session_stream() {
                Some(stream) => self.binder.semaphore.wait_in(stream, value),
                None => self.binder.semaphore.wait(value),
            }
        }

        /// The stream signals once the run has written the buffer, the
        /// queue waits for that, copies the buffer into the texture, and
        /// signals in turn for the next run to wait on.
        pub fn written(&self, target: &OutputTexture) -> Result<()> {
            let device = crate::gpu::shared_device()
                .ok_or_else(|| anyhow!("this machine has no GPU adapter"))?;
            let global = crate::gpu::shared();
            if let Some(stream) = crate::nn::session_stream() {
                ForeignSync::get()?.signal_then_wait(global, device, |value| {
                    self.signals.signal_in(stream, value)
                })?;
            }
            // Without the session's stream the run has finished by now: the
            // session waits for its own stream before it returns.
            target.copy_in()?;
            let value = SharedSync::get()?.signal(global, device)?;
            *self.copied_at.lock().unwrap_or_else(|e| e.into_inner()) = Some(value);
            Ok(())
        }

        /// The buffer's elements, once the stream has written them.
        pub fn read(&self) -> Result<Vec<u8>> {
            if let Some(stream) = crate::nn::session_stream() {
                stream.synchronize()?;
            }
            let started = Instant::now();
            let Some(Ok(mapped)) = self.buffer.cuda.get() else {
                return Err(anyhow!("the output's buffer is not mapped into CUDA"));
            };
            let bytes = mapped.read()?;
            PROFILE.download.add(bytes.len(), started);
            Ok(bytes)
        }
    }
}
