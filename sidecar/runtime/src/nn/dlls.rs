//! The native libraries an ONNX session pulls in, and where each came from.
//!
//! Nothing here is loaded by bare name. Windows ships an onnxruntime of its
//! own in System32 and PATH is whatever the machine happens to have, so the
//! runtime is named by full path, and the directories its dependencies come
//! from are resolved and added to the search path one at a time.

use std::path::{Path, PathBuf};

pub use wasmtime_wasi_nn::backend::onnx::Provider;

/// The ONNX Runtime shared library's file name on this platform.
#[cfg(windows)]
pub const RUNTIME_LIB: &str = "onnxruntime.dll";
#[cfg(target_os = "macos")]
pub const RUNTIME_LIB: &str = "libonnxruntime.dylib";
#[cfg(all(unix, not(target_os = "macos")))]
pub const RUNTIME_LIB: &str = "libonnxruntime.so";

/// The CUDA execution provider itself, which ships with ONNX Runtime and is
/// fetched beside the runtime.
#[cfg(windows)]
pub const PROVIDER_LIBS: &[&str] = &[
    "onnxruntime_providers_shared.dll",
    "onnxruntime_providers_cuda.dll",
];
#[cfg(not(windows))]
pub const PROVIDER_LIBS: &[&str] = &[
    "libonnxruntime_providers_shared.so",
    "libonnxruntime_providers_cuda.so",
];

/// Where the DirectML build of ONNX Runtime lives, under the runtime
/// directory. It is a different binary from the CPU and CUDA one and carries
/// the same name, so it cannot share a directory with them.
pub const DIRECTML_SUBDIR: &str = "directml";

/// DirectML itself, which the DirectML build of ONNX Runtime loads by name.
pub const DIRECTML_LIB: &str = "DirectML.dll";

/// The library whose presence in the process proves a provider engaged, most
/// wanted first. ONNX Runtime falls back to its CPU provider in silence, so
/// this is the only thing that establishes what actually ran.
#[cfg(windows)]
pub const PROVIDER_MARKERS: &[(Provider, &str)] = &[
    (Provider::Cuda, "onnxruntime_providers_cuda.dll"),
    (Provider::DirectMl, DIRECTML_LIB),
];
#[cfg(all(unix, not(target_os = "macos")))]
pub const PROVIDER_MARKERS: &[(Provider, &str)] =
    &[(Provider::Cuda, "libonnxruntime_providers_cuda.so")];
/// CoreML is linked into ONNX Runtime rather than loaded beside it, and the
/// loaded images cannot be enumerated here anyway, so there is no marker to
/// look for and the verdict says so rather than guessing.
#[cfg(target_os = "macos")]
pub const PROVIDER_MARKERS: &[(Provider, &str)] = &[];

/// One library the CUDA provider needs, and what every version of it is
/// called, so an absent one can be reported alongside the version that is
/// actually installed.
pub struct CudaDep {
    pub file: &'static str,
    pub family: &'static str,
}

/// What the CUDA provider links against. These belong to the machine, like
/// the driver: a CUDA 12 runtime and cuDNN 9 are prerequisites, not something
/// the sidecar carries.
#[cfg(windows)]
pub const CUDA_DEPS: &[CudaDep] = &[
    CudaDep {
        file: "cudart64_12.dll",
        family: "cudart64_",
    },
    CudaDep {
        file: "cublas64_12.dll",
        family: "cublas64_",
    },
    CudaDep {
        file: "cublasLt64_12.dll",
        family: "cublasLt64_",
    },
    CudaDep {
        file: "cufft64_11.dll",
        family: "cufft64_",
    },
    CudaDep {
        file: "cudnn64_9.dll",
        family: "cudnn64_",
    },
];
#[cfg(not(windows))]
pub const CUDA_DEPS: &[CudaDep] = &[
    CudaDep {
        file: "libcudart.so.12",
        family: "libcudart.so.",
    },
    CudaDep {
        file: "libcublas.so.12",
        family: "libcublas.so.",
    },
    CudaDep {
        file: "libcublasLt.so.12",
        family: "libcublasLt.so.",
    },
    CudaDep {
        file: "libcufft.so.11",
        family: "libcufft.so.",
    },
    CudaDep {
        file: "libcudnn.so.9",
        family: "libcudnn.so.",
    },
];

/// cuDNN 9 loads these itself, by bare name, out of its own directory.
/// `_adv` carries the recurrent ops -- an RNN or LSTM node needs it, where a
/// plain convolutional graph never touches it -- and its absence does not
/// register as a missing library until a session tries to build that node,
/// which is a crash rather than a refusal.
#[cfg(windows)]
pub const CUDNN_SUBLIBS: &[&str] = &[
    "cudnn_adv64_9.dll",
    "cudnn_graph64_9.dll",
    "cudnn_ops64_9.dll",
    "cudnn_heuristic64_9.dll",
    "cudnn_engines_precompiled64_9.dll",
    "cudnn_engines_runtime_compiled64_9.dll",
];
#[cfg(not(windows))]
pub const CUDNN_SUBLIBS: &[&str] = &[
    "libcudnn_adv.so.9",
    "libcudnn_graph.so.9",
    "libcudnn_ops.so.9",
    "libcudnn_heuristic.so.9",
    "libcudnn_engines_precompiled.so.9",
    "libcudnn_engines_runtime_compiled.so.9",
];

/// Where a library was found.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Origin {
    /// Under the runtime directory the caller named.
    Fetched,
    /// Shipped by the operating system.
    System,
    /// Anywhere else the machine had it, typically a CUDA toolkit install.
    Machine,
}

impl Origin {
    pub fn label(self) -> &'static str {
        match self {
            Origin::Fetched => "fetched",
            Origin::System => "system",
            Origin::Machine => "machine",
        }
    }
}

/// Directories whose contents belong to the operating system rather than to
/// anyone who installed something. Spelled with the separator each platform
/// writes, since these are compared against paths the loader reported.
#[cfg(windows)]
const SYSTEM_DIRS: &[&str] = &["\\windows\\system32", "\\windows\\winsxs"];
#[cfg(not(windows))]
const SYSTEM_DIRS: &[&str] = &["/usr/lib", "/lib", "/system/library"];

/// A path as text, comparable with another. Windows accepts either separator
/// and is case-insensitive, so a directory typed with forward slashes on the
/// command line still matches what the loader reports with backslashes.
fn comparable(path: &Path) -> String {
    let text = path.to_string_lossy().to_ascii_lowercase();
    if cfg!(windows) {
        text.replace('/', "\\")
    } else {
        text
    }
}

/// Which of the three a loaded library's path belongs to.
pub fn origin(path: &Path, runtime_dir: &Path) -> Origin {
    let path = comparable(path);
    let root = comparable(runtime_dir);
    if !root.is_empty() && path.starts_with(&root) {
        Origin::Fetched
    } else if SYSTEM_DIRS.iter().any(|dir| path.contains(dir)) {
        Origin::System
    } else {
        Origin::Machine
    }
}

/// The directories a dependency may legitimately come from, most specific
/// first: what we fetched, then what the machine advertises.
pub fn search_dirs(runtime_dir: &Path) -> Vec<PathBuf> {
    let mut dirs = vec![runtime_dir.to_path_buf()];
    let var = if cfg!(windows) {
        "PATH"
    } else {
        "LD_LIBRARY_PATH"
    };
    if let Some(raw) = std::env::var_os(var) {
        for dir in std::env::split_paths(&raw) {
            if !dir.as_os_str().is_empty() && !dirs.contains(&dir) {
                dirs.push(dir);
            }
        }
    }
    dirs
}

/// The first directory in `dirs` holding a file called `file`.
pub fn locate(dirs: &[PathBuf], file: &str) -> Option<PathBuf> {
    dirs.iter().find(|d| d.join(file).is_file()).cloned()
}

/// Other versions of the same library, for saying what is installed when the
/// wanted one is not. Each entry is a file name and the directory holding it.
pub fn siblings(dirs: &[PathBuf], dep: &CudaDep) -> Vec<(String, PathBuf)> {
    let mut out = Vec::new();
    for dir in dirs {
        let Ok(entries) = std::fs::read_dir(dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.starts_with(dep.family) && name != dep.file {
                out.push((name, dir.clone()));
            }
        }
    }
    out
}

/// Make `dir` a place dependent libraries are found, without putting it on
/// PATH and without letting the rest of PATH answer.
///
/// The CUDA provider's own imports and cuDNN's runtime-loaded sublibraries are
/// resolved by bare name through the normal search order. Switching the
/// process to the restricted order and adding directories to it explicitly
/// means only the directories named here can answer.
#[cfg(windows)]
pub fn add_search_dir(dir: &Path) -> anyhow::Result<()> {
    use windows_sys::Win32::System::LibraryLoader::{
        AddDllDirectory, SetDefaultDllDirectories, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS,
    };
    let abs = std::fs::canonicalize(dir)?;
    let text = abs.to_string_lossy();
    let text = text.strip_prefix("\\\\?\\").unwrap_or(&text);
    let wide: Vec<u16> = text.encode_utf16().chain(std::iter::once(0)).collect();
    unsafe {
        if SetDefaultDllDirectories(LOAD_LIBRARY_SEARCH_DEFAULT_DIRS) == 0 {
            anyhow::bail!("SetDefaultDllDirectories failed");
        }
        if AddDllDirectory(wide.as_ptr()).is_null() {
            anyhow::bail!("AddDllDirectory({text}) failed");
        }
    }
    Ok(())
}

/// Elsewhere the loader takes its directories from the environment, which the
/// caller sets before the process starts; there is nothing to pin here.
#[cfg(not(windows))]
pub fn add_search_dir(_dir: &Path) -> anyhow::Result<()> {
    Ok(())
}

/// Every native library the process has actually loaded, by full path.
#[cfg(windows)]
pub fn loaded() -> Option<Vec<PathBuf>> {
    use windows_sys::Win32::Foundation::HMODULE;
    use windows_sys::Win32::System::ProcessStatus::{EnumProcessModules, GetModuleFileNameExW};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;

    let mut out = Vec::new();
    unsafe {
        let process = GetCurrentProcess();
        let mut needed: u32 = 0;
        let mut modules: Vec<HMODULE> = vec![std::ptr::null_mut(); 2048];
        let bytes = (modules.len() * std::mem::size_of::<HMODULE>()) as u32;
        if EnumProcessModules(process, modules.as_mut_ptr(), bytes, &mut needed) == 0 {
            return None;
        }
        let count = (needed as usize / std::mem::size_of::<HMODULE>()).min(modules.len());
        for &module in &modules[..count] {
            let mut buf = [0u16; 1024];
            let len = GetModuleFileNameExW(process, module, buf.as_mut_ptr(), buf.len() as u32);
            if len > 0 {
                out.push(PathBuf::from(String::from_utf16_lossy(
                    &buf[..len as usize],
                )));
            }
        }
    }
    out.sort();
    Some(out)
}

#[cfg(target_os = "linux")]
pub fn loaded() -> Option<Vec<PathBuf>> {
    let maps = std::fs::read_to_string("/proc/self/maps").ok()?;
    let mut out: Vec<PathBuf> = maps
        .lines()
        .filter_map(|line| {
            line.split_once(" /")
                .map(|(_, rest)| PathBuf::from(format!("/{rest}")))
        })
        .collect();
    out.sort();
    out.dedup();
    Some(out)
}

/// Nothing here enumerates the loaded images, so the provider cannot be
/// established and callers are told so rather than guessing.
#[cfg(not(any(windows, target_os = "linux")))]
pub fn loaded() -> Option<Vec<PathBuf>> {
    None
}

/// Whether Direct3D 12 can give DirectML a device.
///
/// d3d12.dll is the operating system's and is taken by name, unlike everything
/// else here: it is the graphics stack, not something anyone installed beside
/// a runtime. A null device pointer asks whether a device could be created
/// without creating one. The library is left loaded, since a DirectML session
/// wants it next.
#[cfg(windows)]
pub fn d3d12_available() -> Option<bool> {
    #[repr(C)]
    struct Guid {
        data1: u32,
        data2: u16,
        data3: u16,
        data4: [u8; 8],
    }
    // ID3D12Device, and the feature level every DirectML-capable adapter has.
    const ID3D12DEVICE: Guid = Guid {
        data1: 0x189819f1,
        data2: 0x1db6,
        data3: 0x4b57,
        data4: [0xbe, 0x54, 0x18, 0x21, 0x33, 0x9b, 0x85, 0xf7],
    };
    const FEATURE_LEVEL_11_0: u32 = 0xb000;
    type CreateDevice = unsafe extern "system" fn(
        *mut std::ffi::c_void,
        u32,
        *const Guid,
        *mut *mut std::ffi::c_void,
    ) -> i32;

    let lib = unsafe { libloading::Library::new("d3d12.dll") }.ok()?;
    let create: libloading::Symbol<CreateDevice> =
        unsafe { lib.get(b"D3D12CreateDevice\0") }.ok()?;
    let hr = unsafe {
        create(
            std::ptr::null_mut(),
            FEATURE_LEVEL_11_0,
            &ID3D12DEVICE,
            std::ptr::null_mut(),
        )
    };
    let available = hr >= 0;
    std::mem::forget(lib);
    Some(available)
}

/// Direct3D 12 is Windows', so nowhere else has a device for DirectML.
#[cfg(not(windows))]
pub fn d3d12_available() -> Option<bool> {
    Some(false)
}

/// The libraries worth naming in a provider verdict.
pub fn interesting(path: &Path) -> bool {
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default();
    let name = name.strip_prefix("lib").unwrap_or(&name);
    [
        "onnxruntime",
        "cudart",
        "cublas",
        "cudnn",
        "cufft",
        "nvcuda",
        "nvrtc",
        "directml",
        "d3d12",
        "dxcore",
    ]
    .iter()
    .any(|stem| name.starts_with(stem))
}

/// Whether a library of this name is loaded in the process.
fn is_loaded(loaded: &[PathBuf], file: &str) -> bool {
    let want = file.to_ascii_lowercase();
    loaded.iter().any(|p| {
        p.file_name()
            .map(|n| n.to_string_lossy().to_ascii_lowercase() == want)
            .unwrap_or(false)
    })
}

/// The execution provider that actually engaged. `None` when the process's
/// loaded images cannot be enumerated, which is not the same as `Cpu`: ONNX
/// Runtime falls back to its CPU provider in silence, so a guess here would be
/// a guess reported as a fact.
pub fn engaged_provider() -> Option<Provider> {
    let loaded = loaded()?;
    for (provider, marker) in PROVIDER_MARKERS {
        if is_loaded(&loaded, marker) {
            return Some(*provider);
        }
    }
    Some(Provider::Cpu)
}
