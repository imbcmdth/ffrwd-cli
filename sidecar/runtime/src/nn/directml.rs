//! DirectML's own API, which the pinned `ort` does not carry: a DirectML
//! session on the process's D3D12 device and queue, and buffers of that
//! device handed to the session as its own allocations.
//!
//! ONNX Runtime publishes the API as a table of functions,
//! `OrtDmlApi` in `dml_provider_factory.h`, reached through
//! `GetExecutionProviderApi("DML", ...)` on the DirectML build of the
//! runtime. The table is declared here as the 1.22.0 header lays it out.
//! The `IDMLDevice` the session needs is made by `DMLCreateDevice` over the
//! process's `ID3D12Device`, from the DirectML library the provider loads.
//!
//! A session registered this way runs its work on the queue every module
//! and the host submit to, so a pass given to the queue before the run and
//! one given after are ordered with it there, and nothing waits.

use std::ffi::{c_int, c_void};
use std::path::Path;
use std::sync::{Arc, OnceLock};

use anyhow::{anyhow, bail, Context, Result};
use ort::sys::{OrtAllocator, OrtSessionOptions, OrtStatusPtr};
use windows::core::{Interface, GUID, HRESULT};
use windows::Win32::Graphics::Direct3D12::{ID3D12CommandQueue, ID3D12Device, ID3D12Resource};

use crate::gpu::tensor::PooledBuffer;

/// `OrtDmlApi`, as ONNX Runtime 1.22.0's `dml_provider_factory.h` declares
/// it: six functions, in this order.
#[repr(C)]
pub struct OrtDmlApi {
    pub session_options_append_execution_provider_dml: unsafe extern "system" fn(
        options: *mut OrtSessionOptions,
        device_id: c_int,
    )
        -> OrtStatusPtr,
    pub session_options_append_execution_provider_dml1: unsafe extern "system" fn(
        options: *mut OrtSessionOptions,
        dml_device: *mut c_void,
        cmd_queue: *mut c_void,
    )
        -> OrtStatusPtr,
    pub create_gpu_allocation_from_d3d_resource: unsafe extern "system" fn(
        d3d_resource: *mut c_void,
        dml_resource: *mut *mut c_void,
    ) -> OrtStatusPtr,
    pub free_gpu_allocation: unsafe extern "system" fn(dml_resource: *mut c_void) -> OrtStatusPtr,
    pub get_d3d12_resource_from_allocation: unsafe extern "system" fn(
        provider: *mut OrtAllocator,
        dml_resource: *mut c_void,
        d3d_resource: *mut *mut c_void,
    ) -> OrtStatusPtr,
    pub session_options_append_execution_provider_dml2: unsafe extern "system" fn(
        options: *mut OrtSessionOptions,
        device_opts: *mut c_void,
    )
        -> OrtStatusPtr,
}

/// `IID_IDMLDevice`.
const IID_IDML_DEVICE: GUID = GUID::from_u128(0x6dbd6437_96fd_423f_a98c_ae5e7c2a573f);

/// `DMLCreateDevice`.
type FnCreateDevice = unsafe extern "system" fn(
    d3d12_device: *mut c_void,
    flags: u32,
    riid: *const GUID,
    device: *mut *mut c_void,
) -> HRESULT;

/// A status the API returned, as a result.
fn checked(call: &str, status: OrtStatusPtr) -> Result<()> {
    // SAFETY: a status ONNX Runtime returned, or null.
    unsafe { ort::error::status_to_result(status) }.map_err(|e| anyhow!("{call}: {e}"))
}

/// The table, from the runtime the process loaded. The first answer, a
/// refusal included, is every later one.
pub fn api() -> Result<&'static OrtDmlApi> {
    static FOUND: OnceLock<std::result::Result<usize, String>> = OnceLock::new();
    let found = FOUND.get_or_init(|| {
        let mut table: *const c_void = std::ptr::null();
        // SAFETY: the name is a C string; the version is the one `ort` was
        // built against, which the loaded runtime answers to.
        let status = unsafe {
            (ort::api().GetExecutionProviderApi)(
                c"DML".as_ptr(),
                ort::sys::ORT_API_VERSION,
                &mut table,
            )
        };
        checked("GetExecutionProviderApi(\"DML\")", status).map_err(|e| format!("{e:#}"))?;
        if table.is_null() {
            return Err("GetExecutionProviderApi(\"DML\") answered nothing".to_string());
        }
        Ok(table as usize)
    });
    match found {
        // SAFETY: the runtime's own table, alive as long as the library is,
        // which is the process.
        Ok(table) => Ok(unsafe { &*(*table as *const OrtDmlApi) }),
        Err(e) => Err(anyhow!("{e}")),
    }
}

/// The DirectML device over the process's D3D12 device, and the queue the
/// session runs on, held for the life of the process.
struct SessionDevice {
    dml: windows::core::IUnknown,
    queue: ID3D12CommandQueue,
}

// SAFETY: COM objects DirectML and D3D12 document as free-threaded.
unsafe impl Send for SessionDevice {}
unsafe impl Sync for SessionDevice {}

fn session_device_cell() -> &'static OnceLock<SessionDevice> {
    static DEVICE: OnceLock<SessionDevice> = OnceLock::new();
    &DEVICE
}

/// Whether DirectML sessions run on the process's device, so a buffer of
/// that device can be bound to one.
pub fn on_shared_device() -> bool {
    session_device_cell().get().is_some()
}

/// Makes the DirectML device over the process's D3D12 device, with
/// `DirectML.dll` loaded from `library` where one was found beside the
/// runtime, else by name. Returns the raw pointers a registration takes.
pub fn prepare(library: Option<&Path>) -> Result<(usize, usize)> {
    if let Some(made) = session_device_cell().get() {
        return Ok((made.dml.as_raw() as usize, made.queue.as_raw() as usize));
    }
    api()?;
    let device: ID3D12Device = crate::gpu::d3d12::device()
        .ok_or_else(|| anyhow!("the process's device is not a D3D12 one"))?;
    let queue = crate::gpu::d3d12::raw_queue()?;
    // SAFETY: loading DirectML runs its initialisers, which the DirectML
    // provider runs in this process too.
    let loaded = unsafe {
        match library {
            Some(path) => libloading::Library::new(path),
            None => libloading::Library::new("DirectML.dll"),
        }
    }
    .context("loading DirectML.dll")?;
    // SAFETY: the signature DirectML.h declares.
    let create: FnCreateDevice = unsafe {
        *loaded
            .get::<FnCreateDevice>(b"DMLCreateDevice\0")
            .context("DirectML.dll exports no DMLCreateDevice")?
    };
    // DirectML stays loaded for the process: the device made below is its.
    std::mem::forget(loaded);
    let mut raw: *mut c_void = std::ptr::null_mut();
    // SAFETY: a live device, no flags, the interface's id, an out pointer.
    unsafe { create(device.as_raw(), 0, &IID_IDML_DEVICE, &mut raw) }
        .ok()
        .context("DMLCreateDevice")?;
    if raw.is_null() {
        bail!("DMLCreateDevice made nothing");
    }
    // SAFETY: an interface pointer DMLCreateDevice made, whose reference
    // the result owns.
    let dml = unsafe { windows::core::IUnknown::from_raw(raw) };
    let made = session_device_cell().get_or_init(|| SessionDevice { dml, queue });
    Ok((made.dml.as_raw() as usize, made.queue.as_raw() as usize))
}

/// Registers DirectML on a session's options with the device and queue
/// [`prepare`] made.
///
/// # Safety
/// `options` is a live session options object; `dml` and `queue` are what
/// [`prepare`] returned.
pub unsafe fn register(options: *mut OrtSessionOptions, dml: usize, queue: usize) -> Result<()> {
    let api = api()?;
    // SAFETY: as the caller promises; the session takes its own references.
    let status = unsafe {
        (api.session_options_append_execution_provider_dml1)(
            options,
            dml as *mut c_void,
            queue as *mut c_void,
        )
    };
    checked("SessionOptionsAppendExecutionProvider_DML1", status)
}

/// A D3D12 buffer as an allocation of DirectML's, which a tensor bound to
/// a DirectML session is made over; freed when dropped.
pub struct Allocation {
    pointer: usize,
    /// The buffer, kept alive as long as the allocation is.
    _resource: ID3D12Resource,
}

// SAFETY: an opaque pointer DirectML owns, used only through its API.
unsafe impl Send for Allocation {}
unsafe impl Sync for Allocation {}

impl Allocation {
    pub fn pointer(&self) -> u64 {
        self.pointer as u64
    }

    fn of(resource: &ID3D12Resource) -> Result<Allocation> {
        let api = api()?;
        let mut pointer: *mut c_void = std::ptr::null_mut();
        // SAFETY: a live resource of the device the sessions run on, and an
        // out pointer.
        let status = unsafe {
            (api.create_gpu_allocation_from_d3d_resource)(resource.as_raw(), &mut pointer)
        };
        checked("CreateGPUAllocationFromD3DResource", status)?;
        if pointer.is_null() {
            bail!("CreateGPUAllocationFromD3DResource made nothing");
        }
        Ok(Allocation {
            pointer: pointer as usize,
            _resource: resource.clone(),
        })
    }
}

impl Drop for Allocation {
    fn drop(&mut self) {
        if let Ok(api) = api() {
            // SAFETY: made by CreateGPUAllocationFromD3DResource, freed once.
            let status = unsafe { (api.free_gpu_allocation)(self.pointer as *mut c_void) };
            let _ = checked("FreeGPUAllocation", status);
        }
    }
}

/// `buffer` as a DirectML allocation, made the first time it is asked for
/// and kept as long as the buffer is.
pub fn allocation_of_buffer(buffer: &PooledBuffer) -> Result<Arc<Allocation>> {
    if !on_shared_device() {
        bail!("DirectML sessions do not run on the process's device");
    }
    buffer
        .dml
        .get_or_init(|| {
            let resource = buffer
                .resource
                .as_ref()
                .ok_or_else(|| "the buffer is not one the host made itself".to_string())?;
            Allocation::of(resource)
                .map(Arc::new)
                .map_err(|e| format!("{e:#}"))
        })
        .clone()
        .map_err(|e| anyhow!("{e}"))
}

/// The buffer of an output's texture as a DirectML allocation.
pub fn allocation_of(target: &crate::gpu::output::OutputTexture) -> Result<Arc<Allocation>> {
    let buffer = target
        .buffer
        .as_ref()
        .ok_or_else(|| anyhow!("the output has no buffer"))?;
    allocation_of_buffer(buffer)
}
