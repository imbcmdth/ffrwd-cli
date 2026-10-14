//! Wasmtime host for `wasi:webgpu@0.0.1` over wgpu-core.
//!
//! Adapted from wasi-gfx-runtime's `wasi-webgpu-wasmtime` at 301716b. What
//! changed from upstream, in short (README.md has the list):
//!
//! - Synchronous throughout: `map-async` blocks until the map lands instead
//!   of being a wasmtime async function, so a store linking this stays a
//!   synchronous store.
//! - Compute only: no surfaces, canvases or window handles. The
//!   `wasi:graphics-context` types webgpu names are linked, and every
//!   function on them refuses.
//! - The context is a borrowed view (`WasiWebGpuCtx`) the way upstream's
//!   later 0.3 crate has it, holding a shared `wgpu_core::global::Global`.
//! - `subgroups` maps to wgpu's own `SUBGROUP` feature, and the record
//!   resources, pipeline constants, required limits and
//!   `on-submitted-work-done` are implemented.

#![allow(clippy::unwrap_or_default)]
#![allow(clippy::new_without_default)]

use std::sync::Arc;

use wasmtime::component::HasData;

mod enum_conversions;
mod graphics_context;
mod to_core_conversions;
mod trait_impls;
mod wrapper_types;

pub use graphics_context::{AbstractBuffer, GraphicsContext};
pub use wrapper_types::Texture;

/// Re-export of `wgpu_core` and `wgpu_types`, so a runtime does not have to
/// track which version of wgpu this crate is built on.
pub mod reexports {
    pub use wgpu_core;
    pub use wgpu_types;
}

// https://searchfox.org/mozilla-central/source/dom/webgpu/Instance.h#68
#[cfg(target_os = "android")]
const PREFERRED_CANVAS_FORMAT: wasi::webgpu::webgpu::GpuTextureFormat =
    wasi::webgpu::webgpu::GpuTextureFormat::Rgba8unorm;
#[cfg(not(target_os = "android"))]
const PREFERRED_CANVAS_FORMAT: wasi::webgpu::webgpu::GpuTextureFormat =
    wasi::webgpu::webgpu::GpuTextureFormat::Bgra8unorm;

wasmtime::component::bindgen!({
    path: "wit",
    world: "ffrwd:webgpu-host/host",
    require_store_data_send: true,
    imports: {
        default: trappable,
    },
    with: {
        "wasi:io": wasmtime_wasi_io::bindings::wasi::io,
        "wasi:webgpu/webgpu.gpu-adapter": wrapper_types::Adapter,
        "wasi:webgpu/webgpu.gpu-device": wrapper_types::Device,
        "wasi:webgpu/webgpu.gpu-queue": wrapper_types::Queue,
        "wasi:webgpu/webgpu.gpu-command-encoder": wrapper_types::CommandEncoder,
        "wasi:webgpu/webgpu.gpu-render-pass-encoder": wrapper_types::RenderPassEncoder,
        "wasi:webgpu/webgpu.gpu-compute-pass-encoder": wrapper_types::ComputePassEncoder,
        "wasi:webgpu/webgpu.gpu-shader-module": wrapper_types::ShaderModule,
        "wasi:webgpu/webgpu.gpu-compilation-info": wrapper_types::CompilationInfo,
        "wasi:webgpu/webgpu.gpu-compilation-message": wrapper_types::CompilationMessage,
        "wasi:webgpu/webgpu.gpu-render-pipeline": wrapper_types::RenderPipeline,
        "wasi:webgpu/webgpu.gpu-render-bundle-encoder": wrapper_types::RenderBundleEncoder,
        "wasi:webgpu/webgpu.gpu-render-bundle": wgpu_core::id::RenderBundleId,
        "wasi:webgpu/webgpu.gpu-command-buffer": wgpu_core::id::CommandBufferId,
        "wasi:webgpu/webgpu.gpu-buffer": wrapper_types::Buffer,
        "wasi:webgpu/webgpu.gpu-pipeline-layout": wgpu_core::id::PipelineLayoutId,
        "wasi:webgpu/webgpu.gpu-bind-group-layout": wgpu_core::id::BindGroupLayoutId,
        "wasi:webgpu/webgpu.gpu-sampler": wgpu_core::id::SamplerId,
        "wasi:webgpu/webgpu.gpu-supported-features": wgpu_types::Features,
        "wasi:webgpu/webgpu.gpu-texture": wrapper_types::Texture,
        "wasi:webgpu/webgpu.gpu-compute-pipeline": wrapper_types::ComputePipeline,
        "wasi:webgpu/webgpu.gpu-bind-group": wgpu_core::id::BindGroupId,
        "wasi:webgpu/webgpu.gpu-texture-view": wgpu_core::id::TextureViewId,
        "wasi:webgpu/webgpu.gpu-adapter-info": wgpu_types::AdapterInfo,
        "wasi:webgpu/webgpu.gpu-query-set": wgpu_core::id::QuerySetId,
        "wasi:webgpu/webgpu.gpu-supported-limits": wgpu_types::Limits,
        "wasi:webgpu/webgpu.record-gpu-pipeline-constant-value": wrapper_types::RecordGpuPipelineConstantValue,
        "wasi:webgpu/webgpu.record-option-gpu-size64": wrapper_types::RecordOptionGpuSize64,
        "wasi:webgpu/webgpu.gpu-error": wrapper_types::GpuError,
        "wasi:webgpu/webgpu.wgsl-language-features": wrapper_types::WgslLanguageFeatures,
        "wasi:graphics-context/graphics-context.context": graphics_context::GraphicsContext,
        "wasi:graphics-context/graphics-context.abstract-buffer": graphics_context::AbstractBuffer,
    },
});

/// Adds `wasi:webgpu/webgpu` and `wasi:graphics-context/graphics-context` to
/// a linker. `wasi:io/poll`, which webgpu's uncaptured-error subscription
/// returns, is WASI's own and is expected to be linked already.
pub fn add_to_linker<T>(l: &mut wasmtime::component::Linker<T>) -> wasmtime::Result<()>
where
    T: WasiWebGpuView + 'static,
{
    wasi::webgpu::webgpu::add_to_linker::<_, HasWasiWebGpuCtx>(l, T::webgpu)?;
    wasi::graphics_context::graphics_context::add_to_linker::<_, HasWasiWebGpuCtx>(l, T::webgpu)?;
    Ok(())
}

/// Implemented by a store's data to hand this crate its view.
pub trait WasiWebGpuView: Send {
    fn webgpu(&mut self) -> WasiWebGpuCtx<'_>;
}

/// What the host functions borrow from the store for one call: the wgpu
/// instance the store may use, the device every store shares when the
/// host opened one, and the store's resource table.
pub struct WasiWebGpuCtx<'a> {
    pub instance: &'a Arc<wgpu_core::global::Global>,
    /// The process's device: `request-adapter` and `request-device` answer
    /// it, so every module and the host's own frames are on one device.
    /// None links a store whose requests find nothing.
    pub shared: Option<&'a Arc<SharedDevice>>,
    pub table: &'a mut wasmtime::component::ResourceTable,
}

/// The one device a process opens, shared by every store and by the host.
/// The adapter and queue ids are held here so a guest dropping its handle
/// never drops the device's; the device itself is dropped by nobody.
pub struct SharedDevice {
    pub adapter: Arc<wgpu_core::id::AdapterId>,
    pub device: wgpu_core::id::DeviceId,
    pub queue: Arc<wgpu_core::id::QueueId>,
    /// What the device was opened with: the most a guest may require.
    pub features: wgpu_types::Features,
    pub limits: wgpu_types::Limits,
    pub info: wgpu_types::AdapterInfo,
}

/// Whether wgpu-core is built with `static-dxc`: true on the targets
/// Cargo.toml turns it on for.
const STATIC_DXC: bool = cfg!(all(
    target_os = "windows",
    target_env = "msvc",
    not(target_arch = "aarch64")
));

/// Whether wgpu's D3D12 backend compiles shaders with DXC, as
/// `WGPU_DX12_COMPILER` and the build choose it. Without the variable, wgpu
/// uses static DXC where it is built in, and elsewhere `dxcompiler.dll` if
/// it loads and FXC if not; that last choice is taken as FXC.
pub fn dx12_compiler_is_dxc() -> bool {
    match wgpu_types::Dx12Compiler::from_env() {
        Some(wgpu_types::Dx12Compiler::Fxc) => false,
        Some(wgpu_types::Dx12Compiler::DynamicDxc { .. } | wgpu_types::Dx12Compiler::StaticDxc) => {
            true
        }
        Some(wgpu_types::Dx12Compiler::Auto) | None => STATIC_DXC,
    }
}

/// How a D3D12 device without DXC gets subgroups.
const SUBGROUPS_HINT: &str = "The D3D12 device has no subgroups without DXC; put \
     dxcompiler.dll on the PATH and set WGPU_DX12_COMPILER=dxc to get them.";

/// The message a refused `request-device` carries: `refusal` as it stands,
/// followed by how to get subgroups when the request asked for them and the
/// adapter is a D3D12 one without them because DXC is not compiling its
/// shaders.
pub fn request_device_message(
    refusal: &str,
    requested_subgroups: bool,
    backend: wgpu_types::Backend,
    adapter_has_subgroups: bool,
    compiler_is_dxc: bool,
) -> String {
    if requested_subgroups && d3d12_without_dxc(backend, adapter_has_subgroups, compiler_is_dxc) {
        format!("{refusal}. {SUBGROUPS_HINT}")
    } else {
        refusal.to_string()
    }
}

/// Whether the adapter is a D3D12 one without subgroups because DXC is not
/// compiling its shaders.
fn d3d12_without_dxc(
    backend: wgpu_types::Backend,
    adapter_has_subgroups: bool,
    compiler_is_dxc: bool,
) -> bool {
    backend == wgpu_types::Backend::Dx12 && !adapter_has_subgroups && !compiler_is_dxc
}

/// Prints how to get subgroups on stderr, once per process, when a module
/// asks whether the shared device has them and it is a D3D12 one without
/// them for want of DXC. A module that checks before it asks for a device
/// refuses with its own message, which cannot carry the fix.
pub(crate) fn note_missing_subgroups(shared: &SharedDevice) {
    static NOTED: std::sync::Once = std::sync::Once::new();
    if d3d12_without_dxc(
        shared.info.backend,
        shared.features.contains(wgpu_types::Features::SUBGROUP),
        dx12_compiler_is_dxc(),
    ) {
        NOTED.call_once(|| eprintln!("wasi:webgpu: {SUBGROUPS_HINT}"));
    }
}

struct HasWasiWebGpuCtx;

impl HasData for HasWasiWebGpuCtx {
    type Data<'a> = WasiWebGpuCtx<'a>;
}
