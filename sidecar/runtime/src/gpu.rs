//! `wasi:webgpu` for components that import it: GPU compute through wgpu,
//! for a module the argv granted `-gpu`, and the device the host itself
//! draws on.
//!
//! The interface is `wasi:webgpu@0.0.1`, the synchronous WASI 0.2 version,
//! hosted by the copy of wasi-gfx's host under `vendor/wasi-webgpu-wasmtime`
//! (it needs a GPU-capable build: the `gpu` cargo feature).
//!
//! A module the argv did not grant never reaches a GPU: a run refuses it
//! before instantiation (see `runtime::link`), and a describe, which does
//! instantiate it, links the interface over an instance with no backends, so
//! `request-adapter` finds nothing.
//!
//! One wgpu instance and ONE device serve the whole process, created the
//! first time a granted module is instantiated or a frame is put on the GPU.
//! Every module's `request-adapter` and `request-device` answer that device,
//! so a module's textures and the frames the host holds on the GPU
//! ([`frames`]) live on one device and nothing is copied between two. The
//! backend is the platform's own, D3D12 on Windows, Metal on macOS, Vulkan
//! elsewhere, since every decoder there writes that API's images;
//! `WGPU_BACKEND` (wgpu's own variable, a comma list such as `vulkan` or
//! `dx12`) narrows what is tried.

use anyhow::Result;
use wasmtime::component::{Component, Linker};

#[cfg(feature = "gpu")]
pub mod frames;

/// A build without the `gpu` feature holds no frame on a GPU; the type
/// exists so [`crate::node::FrameData`] has its variant, and is never made.
#[cfg(not(feature = "gpu"))]
pub mod frames {
    use std::sync::Arc;

    pub struct GpuFrame(std::convert::Infallible);

    impl GpuFrame {
        pub fn bytes(&self) -> anyhow::Result<Arc<Vec<u8>>> {
            match self.0 {}
        }

        pub fn wire_len(&self) -> usize {
            match self.0 {}
        }
    }

    impl std::fmt::Debug for GpuFrame {
        fn fmt(&self, _f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self.0 {}
        }
    }
}

/// The interface namespace a component imports to ask for a GPU.
pub const IMPORT_PREFIX: &str = "wasi:webgpu/";

/// Whether this build can host `wasi:webgpu` at all.
pub const AVAILABLE: bool = cfg!(feature = "gpu");

/// Whether the component imports `wasi:webgpu`. Read off the component
/// type, so nothing is instantiated to find out.
pub(crate) fn imports_webgpu(component: &Component, engine: &wasmtime::Engine) -> bool {
    component
        .component_type()
        .imports(engine)
        .any(|(name, _)| name.starts_with(IMPORT_PREFIX))
}

#[cfg(feature = "gpu")]
mod enabled {
    use std::sync::{Arc, OnceLock};

    use wasi_webgpu_wasmtime::reexports::{wgpu_core, wgpu_types};
    use wasi_webgpu_wasmtime::SharedDevice;

    /// A store's GPU: the wgpu instance its module's calls go to, and the
    /// process's device when the module is granted one.
    pub struct StoreGpu {
        pub(crate) instance: Arc<wgpu_core::global::Global>,
        pub(crate) shared: Option<Arc<SharedDevice>>,
    }

    /// The process's wgpu instance, for granted modules and the host's own
    /// frames.
    pub fn shared() -> &'static Arc<wgpu_core::global::Global> {
        static INSTANCE: OnceLock<Arc<wgpu_core::global::Global>> = OnceLock::new();
        INSTANCE.get_or_init(|| {
            let descriptor = wgpu_types::InstanceDescriptor::new_without_display_handle_from_env();
            Arc::new(wgpu_core::global::Global::new(
                "ffrwd-wasm",
                descriptor,
                None,
            ))
        })
    }

    /// An instance with no backends, whose adapter requests find nothing:
    /// what a module the argv did not grant is linked against.
    fn denied() -> &'static Arc<wgpu_core::global::Global> {
        static INSTANCE: OnceLock<Arc<wgpu_core::global::Global>> = OnceLock::new();
        INSTANCE.get_or_init(|| {
            let mut descriptor = wgpu_types::InstanceDescriptor::new_without_display_handle();
            descriptor.backends = wgpu_types::Backends::empty();
            Arc::new(wgpu_core::global::Global::new(
                "ffrwd-wasm-no-gpu",
                descriptor,
                None,
            ))
        })
    }

    /// The backends tried for the process's device, the platform's own
    /// first: every hardware decoder on Windows writes D3D11 or D3D12
    /// images, every one on macOS writes Metal textures, and on Linux and
    /// Android the decoders that share at all share Vulkan images.
    #[cfg(windows)]
    const PREFERENCE: &[wgpu_types::Backends] = &[
        wgpu_types::Backends::DX12,
        wgpu_types::Backends::VULKAN,
        wgpu_types::Backends::GL,
    ];
    #[cfg(target_os = "macos")]
    const PREFERENCE: &[wgpu_types::Backends] = &[wgpu_types::Backends::METAL];
    #[cfg(not(any(windows, target_os = "macos")))]
    const PREFERENCE: &[wgpu_types::Backends] =
        &[wgpu_types::Backends::VULKAN, wgpu_types::Backends::GL];

    /// The features the device is opened with, of those the adapter has:
    /// what a guest may ask for in `request-device` and what the host's own
    /// passes use. Nothing experimental.
    fn wanted_features() -> wgpu_types::Features {
        wgpu_types::Features::TIMESTAMP_QUERY
            | wgpu_types::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS
            | wgpu_types::Features::TIMESTAMP_QUERY_INSIDE_PASSES
            | wgpu_types::Features::SUBGROUP
            | wgpu_types::Features::SHADER_F16
            | wgpu_types::Features::TEXTURE_COMPRESSION_BC
            | wgpu_types::Features::TEXTURE_COMPRESSION_ETC2
            | wgpu_types::Features::TEXTURE_COMPRESSION_ASTC
            | wgpu_types::Features::FLOAT32_FILTERABLE
            | wgpu_types::Features::BGRA8UNORM_STORAGE
            | wgpu_types::Features::RG11B10UFLOAT_RENDERABLE
            | wgpu_types::Features::DEPTH32FLOAT_STENCIL8
            | wgpu_types::Features::DEPTH_CLIP_CONTROL
            | wgpu_types::Features::INDIRECT_FIRST_INSTANCE
            | wgpu_types::Features::DUAL_SOURCE_BLENDING
            | wgpu_types::Features::CLEAR_TEXTURE
            | wgpu_types::Features::TEXTURE_FORMAT_16BIT_NORM
    }

    /// Opens the process's device on the first adapter a preferred backend
    /// finds. None when the machine has no adapter at all, which every
    /// caller treats as "no GPU".
    fn open_device() -> Option<Arc<SharedDevice>> {
        let instance = shared();
        let options = wgpu_types::RequestAdapterOptions {
            power_preference: wgpu_types::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        };
        let mut found = None;
        for backends in PREFERENCE {
            match instance.request_adapter(&options, *backends, None) {
                Ok(adapter) => {
                    found = Some(adapter);
                    break;
                }
                Err(wgpu_types::RequestAdapterError::NotFound { .. }) => continue,
                Err(e) => {
                    eprintln!("[gpu] asking for an adapter: {e}");
                    continue;
                }
            }
        }
        let adapter = found?;
        let features = instance.adapter_features(adapter) & wanted_features();
        let limits = instance.adapter_limits(adapter);
        let info = instance.adapter_get_info(adapter);
        let desc = wgpu_types::DeviceDescriptor {
            label: Some(std::borrow::Cow::Borrowed("ffrwd-wasm")),
            required_features: features,
            required_limits: limits.clone(),
            ..Default::default()
        };
        let (device, queue) = match instance.adapter_request_device(adapter, &desc, None, None) {
            Ok(ids) => ids,
            Err(e) => {
                eprintln!("[gpu] opening {} ({:?}): {e}", info.name, info.backend);
                instance.adapter_drop(adapter);
                return None;
            }
        };
        eprintln!("[gpu] device: {} ({:?})", info.name, info.backend);
        Some(Arc::new(SharedDevice {
            adapter: Arc::new(adapter),
            device,
            queue: Arc::new(queue),
            features,
            limits,
            info,
        }))
    }

    /// The process's device, opened on first use. None on a machine with
    /// no adapter.
    pub fn shared_device() -> Option<&'static Arc<SharedDevice>> {
        static DEVICE: OnceLock<Option<Arc<SharedDevice>>> = OnceLock::new();
        DEVICE.get_or_init(open_device).as_ref()
    }

    impl StoreGpu {
        pub fn new(granted: bool) -> StoreGpu {
            if granted {
                StoreGpu {
                    instance: Arc::clone(shared()),
                    shared: shared_device().cloned(),
                }
            } else {
                StoreGpu {
                    instance: Arc::clone(denied()),
                    shared: None,
                }
            }
        }
    }
}

#[cfg(feature = "gpu")]
pub use enabled::{shared, shared_device, StoreGpu};

/// A build without the `gpu` feature holds nothing per store.
#[cfg(not(feature = "gpu"))]
pub struct StoreGpu;

#[cfg(not(feature = "gpu"))]
impl StoreGpu {
    pub fn new(_granted: bool) -> StoreGpu {
        StoreGpu
    }
}

/// Adds `wasi:webgpu` (and the `wasi:graphics-context` its types name) to a
/// linker whose store data hands out a webgpu view.
#[cfg(feature = "gpu")]
pub(crate) fn add_to_linker<T>(linker: &mut Linker<T>, _component: &Component) -> Result<()>
where
    T: wasi_webgpu_wasmtime::WasiWebGpuView + 'static,
{
    wasi_webgpu_wasmtime::add_to_linker(linker).map_err(|e| anyhow::anyhow!("{e:#}"))
}

/// A build without the `gpu` feature has no implementation to link, so the
/// component's remaining imports (webgpu's) are defined as traps. Only a
/// describe gets here: a run of such a module is refused before linking.
#[cfg(not(feature = "gpu"))]
pub(crate) fn add_to_linker<T: 'static>(
    linker: &mut Linker<T>,
    component: &Component,
) -> Result<()> {
    linker
        .define_unknown_imports_as_traps(component)
        .map_err(|e| anyhow::anyhow!("{e:#}"))
}

/// Why a build without the `gpu` feature refuses to run a module importing
/// `wasi:webgpu`.
pub(crate) const NOT_BUILT: &str = "this ffrwd-wasm was built without GPU support \
     (the `gpu` cargo feature), so it cannot run a module that imports wasi:webgpu";
