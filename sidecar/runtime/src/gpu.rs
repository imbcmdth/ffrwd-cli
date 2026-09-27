//! `wasi:webgpu` for components that import it: GPU compute through wgpu,
//! for a module the argv granted `-gpu`.
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
//! One wgpu instance serves the whole process, created the first time a
//! granted module is instantiated. `WGPU_BACKEND` (wgpu's own variable, a
//! comma list such as `vulkan` or `dx12`) narrows the backends it tries;
//! otherwise adapters are asked for Vulkan, then Metal, then D3D12, then GL.

use anyhow::Result;
use wasmtime::component::{Component, Linker};

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

    /// A store's GPU: the wgpu instance its module's calls go to.
    pub struct StoreGpu {
        pub(crate) instance: Arc<wgpu_core::global::Global>,
    }

    /// The process's wgpu instance, for granted modules.
    fn shared() -> &'static Arc<wgpu_core::global::Global> {
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

    impl StoreGpu {
        pub fn new(granted: bool) -> StoreGpu {
            let instance = if granted { shared() } else { denied() };
            StoreGpu {
                instance: Arc::clone(instance),
            }
        }
    }
}

#[cfg(feature = "gpu")]
pub use enabled::StoreGpu;

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
