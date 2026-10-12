//! The CUDA runtime, loaded at run time, for the few calls that let a CUDA
//! session read and write memory on the process's D3D12 device: importing
//! a shared D3D12 buffer as a CUDA device pointer, importing a shared D3D12
//! sync object as a CUDA external semaphore, waiting on one and signalling
//! one in a stream, and the stream every CUDA session runs on, so those
//! waits and signals are ordered with the sessions' work on the GPU.
//!
//! The library is the CUDA 12 runtime ONNX Runtime's CUDA provider already
//! loaded (`cudart64_12.dll`, found where the provider's preflight found
//! it), so the pointers and the session share one CUDA context: the
//! device's primary context, which both reach through the runtime API.
//! Nothing here is linked at build time; a function the library does not
//! export refuses the load, naming it.

use std::ffi::CStr;
use std::os::raw::{c_char, c_int, c_uint, c_void};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use anyhow::{anyhow, bail, Result};
use libloading::Library;

/// The runtime API's error code; 0 is success.
type CudaError = c_int;

type FnGetDeviceCount = unsafe extern "C" fn(*mut c_int) -> CudaError;
type FnGetDeviceProperties = unsafe extern "C" fn(*mut u8, c_int) -> CudaError;
type FnSetDevice = unsafe extern "C" fn(c_int) -> CudaError;
type FnImportExternalMemory =
    unsafe extern "C" fn(*mut *mut c_void, *const MemoryHandleDesc) -> CudaError;
type FnGetMappedBuffer =
    unsafe extern "C" fn(*mut *mut c_void, *mut c_void, *const BufferDesc) -> CudaError;
type FnDestroy = unsafe extern "C" fn(*mut c_void) -> CudaError;
type FnImportExternalSemaphore =
    unsafe extern "C" fn(*mut *mut c_void, *const SemaphoreHandleDesc) -> CudaError;
type FnWaitExternalSemaphores =
    unsafe extern "C" fn(*const *mut c_void, *const WaitParams, c_uint, *mut c_void) -> CudaError;
type FnSignalExternalSemaphores =
    unsafe extern "C" fn(*const *mut c_void, *const SignalParams, c_uint, *mut c_void) -> CudaError;
type FnStreamCreateWithFlags = unsafe extern "C" fn(*mut *mut c_void, c_uint) -> CudaError;
type FnStreamSynchronize = unsafe extern "C" fn(*mut c_void) -> CudaError;
type FnMemcpy = unsafe extern "C" fn(*mut c_void, *const c_void, usize, c_int) -> CudaError;
type FnGetErrorString = unsafe extern "C" fn(CudaError) -> *const c_char;

/// `cudaExternalMemoryHandleTypeD3D12Resource`.
const MEMORY_D3D12_RESOURCE: c_int = 5;
/// `cudaExternalMemoryDedicated`: the handle is a committed resource.
const MEMORY_DEDICATED: c_uint = 1;
/// The handle type of a D3D12 sync object (4).
const SEMAPHORE_D3D12: c_int = 4;
/// `cudaStreamDefault`: a blocking stream, which the legacy default stream
/// is ordered with.
const STREAM_BLOCKING: c_uint = 0;
/// `cudaStreamNonBlocking`.
const STREAM_NON_BLOCKING: c_uint = 1;
/// `cudaMemcpyDeviceToHost`.
const DEVICE_TO_HOST: c_int = 2;
/// Bytes `cudaGetDeviceProperties_v2` may write: the struct is about 1 KB
/// in CUDA 12 and 13; this leaves room.
const PROPERTIES_LEN: usize = 4096;
/// Where `luid` is in `cudaDeviceProp`: after `name[256]` and the 16-byte
/// `uuid`.
const LUID_OFFSET: usize = 272;

/// The `win32` arm of the handle union: an NT handle, or a name.
#[repr(C)]
#[derive(Clone, Copy)]
struct Win32Handle {
    handle: *mut c_void,
    name: *const c_void,
}

/// `cudaExternalMemoryHandleDesc`, with the reserved words CUDA 13 adds,
/// zeroed, which a CUDA 12 runtime does not read.
#[repr(C)]
struct MemoryHandleDesc {
    kind: c_int,
    handle: Win32Handle,
    size: u64,
    flags: c_uint,
    reserved: [c_uint; 16],
}

/// `cudaExternalMemoryBufferDesc`.
#[repr(C)]
struct BufferDesc {
    offset: u64,
    size: u64,
    flags: c_uint,
    reserved: [c_uint; 16],
}

/// `cudaExternalSemaphoreHandleDesc`.
#[repr(C)]
struct SemaphoreHandleDesc {
    kind: c_int,
    handle: Win32Handle,
    flags: c_uint,
    reserved: [c_uint; 16],
}

/// `cudaExternalSemaphoreWaitParams`, the layout the `_v2` entry point
/// takes: a D3D12 sync object waits for `value`. The keyed-mutex
/// member is a struct of a u64 and a u32, so it ends in four bytes of
/// padding, written out here so every later field lands where C has it;
/// the runtime refuses a wait whose reserved words are not zero.
#[repr(C)]
struct WaitParams {
    value: u64,
    nv_sci_sync: u64,
    keyed_mutex_key: u64,
    keyed_mutex_timeout_ms: c_uint,
    keyed_mutex_padding: c_uint,
    params_reserved: [c_uint; 10],
    flags: c_uint,
    reserved: [c_uint; 16],
}

/// `cudaExternalSemaphoreSignalParams`, the layout the `_v2` entry point
/// takes: a D3D12 sync object is set to `value`. Its parameters end on a
/// u32 after 8-byte members, so C pads the struct to 144 bytes, as this
/// one is.
#[repr(C)]
struct SignalParams {
    value: u64,
    nv_sci_sync: u64,
    keyed_mutex_key: u64,
    params_reserved: [c_uint; 12],
    flags: c_uint,
    reserved: [c_uint; 16],
}

// The sizes the CUDA 12 and 13 headers give these structs on 64-bit
// Windows.
const _: () = assert!(std::mem::size_of::<MemoryHandleDesc>() == 104);
const _: () = assert!(std::mem::size_of::<BufferDesc>() == 88);
const _: () = assert!(std::mem::size_of::<SemaphoreHandleDesc>() == 96);
const _: () = assert!(std::mem::size_of::<WaitParams>() == 144);
const _: () = assert!(std::mem::size_of::<SignalParams>() == 144);

/// The CUDA device every session runs on: the CUDA provider's default.
pub const SESSION_DEVICE: c_int = 0;

/// The functions resolved from the runtime, each named as it is exported.
struct Fns {
    get_device_count: FnGetDeviceCount,
    get_device_properties: FnGetDeviceProperties,
    set_device: FnSetDevice,
    import_external_memory: FnImportExternalMemory,
    get_mapped_buffer: FnGetMappedBuffer,
    destroy_external_memory: FnDestroy,
    free: FnDestroy,
    import_external_semaphore: FnImportExternalSemaphore,
    wait_external_semaphores: FnWaitExternalSemaphores,
    signal_external_semaphores: FnSignalExternalSemaphores,
    stream_create: FnStreamCreateWithFlags,
    stream_synchronize: FnStreamSynchronize,
    stream_destroy: FnDestroy,
    destroy_external_semaphore: FnDestroy,
    memcpy: FnMemcpy,
    get_error_string: FnGetErrorString,
}

/// The loaded runtime and its functions.
pub struct Cuda {
    _library: Library,
    fns: Fns,
    /// Where the library was loaded from.
    pub path: PathBuf,
}

// SAFETY: the runtime API is thread-safe; the struct holds only the
// library handle and function pointers into it.
unsafe impl Send for Cuda {}
unsafe impl Sync for Cuda {}

/// Resolves `name` from `library` as `T`.
///
/// # Safety
/// `T` is the signature the runtime's headers declare for `name`.
unsafe fn symbol<T: Copy>(library: &Library, path: &Path, name: &str) -> Result<T> {
    let mut cname = name.as_bytes().to_vec();
    cname.push(0);
    match unsafe { library.get::<T>(&cname) } {
        Ok(found) => Ok(*found),
        Err(e) => bail!("{} exports no {name}: {e}", path.display()),
    }
}

impl Cuda {
    /// The runtime, loaded once per process from `path`. The first answer,
    /// a refusal included, is every later one.
    pub fn open(path: &Path) -> Result<&'static Cuda> {
        static OPENED: OnceLock<std::result::Result<Cuda, String>> = OnceLock::new();
        OPENED
            .get_or_init(|| Cuda::load(path).map_err(|e| format!("{e:#}")))
            .as_ref()
            .map_err(|e| anyhow!("{e}"))
    }

    fn load(path: &Path) -> Result<Cuda> {
        // SAFETY: loading the CUDA runtime runs its initialisers, which the
        // CUDA provider runs in this process too.
        let library = unsafe { Library::new(path) }
            .map_err(|e| anyhow!("{} could not be loaded: {e}", path.display()))?;
        // SAFETY: each type is the signature the CUDA 12 runtime's headers
        // declare for the name.
        let fns = unsafe {
            Fns {
                get_device_count: symbol(&library, path, "cudaGetDeviceCount")?,
                get_device_properties: symbol(&library, path, "cudaGetDeviceProperties_v2")?,
                set_device: symbol(&library, path, "cudaSetDevice")?,
                import_external_memory: symbol(&library, path, "cudaImportExternalMemory")?,
                get_mapped_buffer: symbol(&library, path, "cudaExternalMemoryGetMappedBuffer")?,
                destroy_external_memory: symbol(&library, path, "cudaDestroyExternalMemory")?,
                free: symbol(&library, path, "cudaFree")?,
                import_external_semaphore: symbol(&library, path, "cudaImportExternalSemaphore")?,
                wait_external_semaphores: symbol(
                    &library,
                    path,
                    "cudaWaitExternalSemaphoresAsync_v2",
                )?,
                signal_external_semaphores: symbol(
                    &library,
                    path,
                    "cudaSignalExternalSemaphoresAsync_v2",
                )?,
                stream_create: symbol(&library, path, "cudaStreamCreateWithFlags")?,
                stream_synchronize: symbol(&library, path, "cudaStreamSynchronize")?,
                stream_destroy: symbol(&library, path, "cudaStreamDestroy")?,
                destroy_external_semaphore: symbol(&library, path, "cudaDestroyExternalSemaphore")?,
                memcpy: symbol(&library, path, "cudaMemcpy")?,
                get_error_string: symbol(&library, path, "cudaGetErrorString")?,
            }
        };
        Ok(Cuda {
            _library: library,
            fns,
            path: path.to_path_buf(),
        })
    }

    /// The CUDA ordinal of the adapter whose LUID is `luid`.
    pub fn ordinal_of(&self, luid: [u8; 8]) -> Result<c_int> {
        let mut count: c_int = 0;
        // SAFETY: a call with a valid out pointer.
        self.check("cudaGetDeviceCount", unsafe {
            (self.fns.get_device_count)(&mut count)
        })?;
        let mut properties = vec![0u8; PROPERTIES_LEN];
        for ordinal in 0..count {
            // SAFETY: the buffer is larger than any `cudaDeviceProp`.
            self.check("cudaGetDeviceProperties", unsafe {
                (self.fns.get_device_properties)(properties.as_mut_ptr(), ordinal)
            })?;
            if properties[LUID_OFFSET..LUID_OFFSET + 8] == luid {
                return Ok(ordinal);
            }
        }
        bail!("none of the {count} CUDA device(s) is the adapter the process's D3D12 device is on")
    }

    /// A stream of its own on the session device, which a session is built
    /// on so the waits and signals queued in it are ordered with its work.
    ///
    /// It is a blocking stream. ONNX Runtime's CUDA provider does some of a
    /// run's copies on the legacy default stream rather than the session's,
    /// and only a blocking stream orders that stream's work after a wait
    /// queued in it: on a non-blocking one the copies read a bound input
    /// before the pass that writes it has run.
    pub fn stream(&'static self) -> Result<CudaStream> {
        self.current()?;
        let mut stream: *mut c_void = std::ptr::null_mut();
        // SAFETY: a valid out pointer.
        self.check("cudaStreamCreateWithFlags", unsafe {
            (self.fns.stream_create)(&mut stream, STREAM_BLOCKING)
        })?;
        Ok(CudaStream { cuda: self, stream })
    }

    /// `code` as an error naming `call`, or nothing for success.
    fn check(&self, call: &str, code: CudaError) -> Result<()> {
        if code == 0 {
            return Ok(());
        }
        // SAFETY: the runtime returns a static string for every code.
        let text = unsafe {
            let text = (self.fns.get_error_string)(code);
            if text.is_null() {
                String::new()
            } else {
                CStr::from_ptr(text).to_string_lossy().into_owned()
            }
        };
        bail!("{call} failed: {text} (error {code})")
    }

    /// Makes the session device current on the calling thread.
    fn current(&self) -> Result<()> {
        // SAFETY: a call with a valid ordinal.
        self.check("cudaSetDevice", unsafe {
            (self.fns.set_device)(SESSION_DEVICE)
        })
    }

    /// The committed D3D12 buffer behind the NT handle `handle`, `size`
    /// bytes, as a CUDA device pointer. The handle stays the caller's to
    /// close; CUDA holds its own reference to the memory.
    pub fn import_buffer(&'static self, handle: *mut c_void, size: u64) -> Result<CudaBuffer> {
        self.current()?;
        let desc = MemoryHandleDesc {
            kind: MEMORY_D3D12_RESOURCE,
            handle: Win32Handle {
                handle,
                name: std::ptr::null(),
            },
            size,
            flags: MEMORY_DEDICATED,
            reserved: [0; 16],
        };
        let mut memory: *mut c_void = std::ptr::null_mut();
        // SAFETY: valid descriptor and out pointer.
        self.check("cudaImportExternalMemory", unsafe {
            (self.fns.import_external_memory)(&mut memory, &desc)
        })?;
        let buffer = BufferDesc {
            offset: 0,
            size,
            flags: 0,
            reserved: [0; 16],
        };
        let mut pointer: *mut c_void = std::ptr::null_mut();
        // SAFETY: `memory` was just imported; valid descriptor and out
        // pointer.
        let mapped = self.check("cudaExternalMemoryGetMappedBuffer", unsafe {
            (self.fns.get_mapped_buffer)(&mut pointer, memory, &buffer)
        });
        if let Err(e) = mapped {
            // SAFETY: imported above and mapped nowhere.
            unsafe { (self.fns.destroy_external_memory)(memory) };
            return Err(e);
        }
        Ok(CudaBuffer {
            cuda: self,
            memory,
            pointer: pointer as u64,
            size,
        })
    }

    /// The D3D12 sync object behind the NT handle `handle` as a CUDA
    /// external semaphore, with a stream of its own to wait on it in where
    /// no session stream is given. The handle stays the caller's to close.
    pub fn import_semaphore(&'static self, handle: *mut c_void) -> Result<CudaSemaphore> {
        self.current()?;
        let desc = SemaphoreHandleDesc {
            kind: SEMAPHORE_D3D12,
            handle: Win32Handle {
                handle,
                name: std::ptr::null(),
            },
            flags: 0,
            reserved: [0; 16],
        };
        let mut semaphore: *mut c_void = std::ptr::null_mut();
        // SAFETY: valid descriptor and out pointer.
        self.check("cudaImportExternalSemaphore", unsafe {
            (self.fns.import_external_semaphore)(&mut semaphore, &desc)
        })?;
        let mut stream: *mut c_void = std::ptr::null_mut();
        // SAFETY: a valid out pointer.
        self.check("cudaStreamCreateWithFlags", unsafe {
            (self.fns.stream_create)(&mut stream, STREAM_NON_BLOCKING)
        })?;
        Ok(CudaSemaphore {
            cuda: self,
            semaphore,
            stream,
        })
    }
}

/// A CUDA stream, destroyed when dropped.
pub struct CudaStream {
    cuda: &'static Cuda,
    stream: *mut c_void,
}

// SAFETY: a stream handle, used through the thread-safe runtime API.
unsafe impl Send for CudaStream {}
unsafe impl Sync for CudaStream {}

impl CudaStream {
    /// The `cudaStream_t`, for a session to be built on.
    pub fn raw(&self) -> *mut c_void {
        self.stream
    }

    /// Returns once everything queued in the stream has run.
    pub fn synchronize(&self) -> Result<()> {
        self.cuda.current()?;
        // SAFETY: a live stream.
        self.cuda.check("cudaStreamSynchronize", unsafe {
            (self.cuda.fns.stream_synchronize)(self.stream)
        })
    }
}

impl Drop for CudaStream {
    fn drop(&mut self) {
        // SAFETY: made here, released once.
        unsafe { (self.cuda.fns.stream_destroy)(self.stream) };
    }
}

/// A D3D12 buffer mapped into CUDA: freed and released when dropped.
pub struct CudaBuffer {
    cuda: &'static Cuda,
    memory: *mut c_void,
    /// The device pointer to the buffer's first byte.
    pub pointer: u64,
    pub size: u64,
}

// SAFETY: a device pointer and an import handle, used through the
// thread-safe runtime API.
unsafe impl Send for CudaBuffer {}
unsafe impl Sync for CudaBuffer {}

impl CudaBuffer {
    /// The buffer's bytes, copied to host memory once whatever writes it
    /// has been waited for.
    pub fn read(&self) -> Result<Vec<u8>> {
        self.cuda.current()?;
        let mut out = vec![0u8; self.size as usize];
        // SAFETY: both ranges are `size` bytes.
        self.cuda.check("cudaMemcpy", unsafe {
            (self.cuda.fns.memcpy)(
                out.as_mut_ptr().cast(),
                self.pointer as usize as *const c_void,
                out.len(),
                DEVICE_TO_HOST,
            )
        })?;
        Ok(out)
    }
}

impl Drop for CudaBuffer {
    fn drop(&mut self) {
        // SAFETY: the pointer was mapped from `memory`, which was imported
        // here; each is released once.
        unsafe {
            (self.cuda.fns.free)(self.pointer as usize as *mut c_void);
            (self.cuda.fns.destroy_external_memory)(self.memory);
        }
    }
}

/// A D3D12 sync object imported as a CUDA external semaphore, and the
/// stream it is waited on in.
pub struct CudaSemaphore {
    cuda: &'static Cuda,
    semaphore: *mut c_void,
    stream: *mut c_void,
}

// SAFETY: as for `CudaBuffer`.
unsafe impl Send for CudaSemaphore {}
unsafe impl Sync for CudaSemaphore {}

impl CudaSemaphore {
    /// Has `stream`'s later work wait until the sync object has reached
    /// `value`, without the CPU waiting.
    pub fn wait_in(&self, stream: &CudaStream, value: u64) -> Result<()> {
        self.queue_wait(stream.stream, value)
    }

    /// Has `stream` set the sync object to `value` once the work queued in
    /// it before has run.
    pub fn signal_in(&self, stream: &CudaStream, value: u64) -> Result<()> {
        self.cuda.current()?;
        let params = SignalParams {
            value,
            nv_sci_sync: 0,
            keyed_mutex_key: 0,
            params_reserved: [0; 12],
            flags: 0,
            reserved: [0; 16],
        };
        // SAFETY: one imported semaphore, its parameters, and a live stream.
        self.cuda
            .check("cudaSignalExternalSemaphoresAsync", unsafe {
                (self.cuda.fns.signal_external_semaphores)(
                    &self.semaphore,
                    &params,
                    1,
                    stream.stream,
                )
            })
    }

    /// Returns once the sync object has reached `value`: the wait is queued
    /// on the semaphore's own stream, and the stream is waited for.
    pub fn wait(&self, value: u64) -> Result<()> {
        self.queue_wait(self.stream, value)?;
        // SAFETY: a live stream.
        self.cuda.check("cudaStreamSynchronize", unsafe {
            (self.cuda.fns.stream_synchronize)(self.stream)
        })
    }

    fn queue_wait(&self, stream: *mut c_void, value: u64) -> Result<()> {
        self.cuda.current()?;
        let params = WaitParams {
            value,
            nv_sci_sync: 0,
            keyed_mutex_key: 0,
            keyed_mutex_timeout_ms: 0,
            keyed_mutex_padding: 0,
            params_reserved: [0; 10],
            flags: 0,
            reserved: [0; 16],
        };
        // SAFETY: one imported semaphore, its parameters, and a live stream.
        self.cuda.check("cudaWaitExternalSemaphoresAsync", unsafe {
            (self.cuda.fns.wait_external_semaphores)(&self.semaphore, &params, 1, stream)
        })
    }
}

impl Drop for CudaSemaphore {
    fn drop(&mut self) {
        // SAFETY: both made here, each released once.
        unsafe {
            (self.cuda.fns.stream_destroy)(self.stream);
            (self.cuda.fns.destroy_external_semaphore)(self.semaphore);
        }
    }
}
