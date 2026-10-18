//! Decoding and encoding through libavcodec, loaded at run time.
//!
//! The sidecar has no link-time dependency on libav. libavutil,
//! libswresample, libavcodec and libswscale are loaded by full path, in
//! that order, then libavfilter where it is there, from a directory
//! resolved when something first asks for them: the `-av-runtime` argv flag, else `FFRWD_AV_RUNTIME`, else the
//! directory the running executable is in, which is where the sidecar's
//! wheel puts them. A run that decodes and encodes nothing never loads
//! them, the way `nn` loads no ONNX Runtime for a run binding no model.
//!
//! Only FFmpeg 9.0's majors are accepted (libavcodec 63, libavutil 61,
//! libswscale 10, libswresample 7, libavfilter 12). A directory without
//! libavfilter runs everything but [`Filter`], which refuses by name. The structs this shim reads are
//! transcribed from that release's headers, so a library of any other major
//! is refused by name rather than read with the wrong layout.
//!
//! Pictures cross this API as the sidecar's wire lays them out (see
//! [`WIRE_PIX_FMTS`]), in host memory. A decoder may run on a hardware
//! device; its frames are copied back to host memory and converted with
//! libswscale like any other. A decoder opened on a device the caller
//! already works on ([`Decoder::open_shared`]) can instead hand its
//! pictures over as that device's own textures ([`DeviceFrame`]).

mod decode;
mod encode;
mod error;
mod ffi;
mod filter;
mod fourcc;
mod picture;
mod sys;
mod vulkan;

use std::path::PathBuf;
use std::sync::OnceLock;

pub use decode::{
    ComDevice, D3d12Picture, DecodedFormat, DecodedFrame, Decoder, DeviceFrame, Hw, Received,
    Shared, ON_DEVICE,
};
pub use encode::{EncodedPacket, Encoder};
pub use error::{Error, Result, INSTALL_HINT};
pub use filter::Filter;
pub use fourcc::{codec_for_fourcc, fourcc_for_codec};
pub use picture::{frame_len, WIRE_PIX_FMTS};
pub use vulkan::{QueueFamily, VkImageState, VkPictureLock, VulkanDevice, VulkanHandles};

/// The environment variable naming the libav directory when the command
/// line does not.
pub const RUNTIME_DIR_VAR: &str = "FFRWD_AV_RUNTIME";

/// The majors this build was written against, FFmpeg 9.0's.
pub const AVCODEC_MAJOR: u32 = 63;
pub const AVUTIL_MAJOR: u32 = 61;
pub const SWSCALE_MAJOR: u32 = 10;
pub const SWRESAMPLE_MAJOR: u32 = 7;
pub const AVFILTER_MAJOR: u32 = 12;

/// One of the libraries the shim loads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Library {
    Avutil,
    Swresample,
    Avcodec,
    Swscale,
    Avfilter,
}

impl Library {
    /// The four every run that decodes or encodes needs, in the order they
    /// load: each after the ones it imports. libavfilter loads after them.
    pub const LOAD_ORDER: [Library; 4] = [
        Library::Avutil,
        Library::Swresample,
        Library::Avcodec,
        Library::Swscale,
    ];

    /// Every library this build demands a major of.
    pub const ALL: [Library; 5] = [
        Library::Avutil,
        Library::Swresample,
        Library::Avcodec,
        Library::Swscale,
        Library::Avfilter,
    ];

    /// The name without its `lib` prefix: `avcodec`.
    pub fn name(self) -> &'static str {
        match self {
            Library::Avutil => "avutil",
            Library::Swresample => "swresample",
            Library::Avcodec => "avcodec",
            Library::Swscale => "swscale",
            Library::Avfilter => "avfilter",
        }
    }

    /// The major this build demands.
    pub fn major(self) -> u32 {
        match self {
            Library::Avutil => AVUTIL_MAJOR,
            Library::Swresample => SWRESAMPLE_MAJOR,
            Library::Avcodec => AVCODEC_MAJOR,
            Library::Swscale => SWSCALE_MAJOR,
            Library::Avfilter => AVFILTER_MAJOR,
        }
    }

    /// The file it is loaded from on this platform: the name FFmpeg's own
    /// build gives the shared library of the demanded major.
    pub fn file_name(self) -> String {
        let (name, major) = (self.name(), self.major());
        if cfg!(windows) {
            format!("{name}-{major}.dll")
        } else if cfg!(target_os = "macos") {
            format!("lib{name}.{major}.dylib")
        } else {
            format!("lib{name}.so.{major}")
        }
    }

    /// The function that answers the library's version.
    fn version_symbol(self) -> &'static str {
        match self {
            Library::Avutil => "avutil_version",
            Library::Swresample => "swresample_version",
            Library::Avcodec => "avcodec_version",
            Library::Swscale => "swscale_version",
            Library::Avfilter => "avfilter_version",
        }
    }
}

/// A version int as `major.minor.micro`, the way libav packs it.
pub fn version_text(version: u32) -> String {
    format!(
        "{}.{}.{}",
        version >> 16,
        (version >> 8) & 0xff,
        version & 0xff
    )
}

/// Refuses a library whose version, as its `*_version()` returns it, is of
/// another major than this build demands.
pub fn check_major(library: Library, version: u32) -> Result<()> {
    if version >> 16 == library.major() {
        return Ok(());
    }
    Err(Error::Version {
        library: library.name(),
        found: version_text(version),
        demanded: library.major(),
    })
}

/// What the command line said about libav.
#[derive(Clone, Debug, Default)]
pub struct Config {
    pub runtime_dir: Option<PathBuf>,
}

fn configured() -> &'static OnceLock<Config> {
    static CONFIG: OnceLock<Config> = OnceLock::new();
    &CONFIG
}

/// Records the argv's choice of directory. Called before anything is
/// instantiated; nothing is loaded until something asks for libav, and the
/// first call is the one that counts.
pub fn configure(config: &Config) {
    let _ = configured().set(config.clone());
}

/// The directory libav loads from, and how it was chosen: the argv, else
/// the environment, else the executable's own directory.
fn resolve_dir() -> Result<(PathBuf, &'static str)> {
    if let Some(dir) = configured().get().and_then(|c| c.runtime_dir.clone()) {
        return Ok((dir, "-av-runtime"));
    }
    if let Some(dir) = std::env::var_os(RUNTIME_DIR_VAR) {
        return Ok((PathBuf::from(dir), RUNTIME_DIR_VAR));
    }
    let exe = std::env::current_exe().map_err(|e| Error::NoDirectory {
        why: format!("the executable's own path is unknown: {e}"),
    })?;
    match exe.parent() {
        Some(dir) => Ok((dir.to_path_buf(), "the executable's directory")),
        None => Err(Error::NoDirectory {
            why: format!("{} has no parent directory", exe.display()),
        }),
    }
}

/// The libraries, loaded once on first use. A load that failed fails the
/// same way for every later caller.
fn runtime() -> Result<&'static ffi::Lib> {
    static LIB: OnceLock<Result<ffi::Lib>> = OnceLock::new();
    LIB.get_or_init(|| {
        let (dir, from) = resolve_dir()?;
        ffi::Lib::load(&dir, from)
    })
    .as_ref()
    .map_err(Clone::clone)
}

/// Loads libav now, if nothing has yet; the error is the refusal.
pub fn load() -> Result<()> {
    runtime().map(|_| ())
}

/// The names of the decoders the loaded build has.
pub fn decoders() -> Result<Vec<String>> {
    Ok(runtime()?
        .codecs()
        .into_iter()
        .filter(|(_, decodes, _)| *decodes)
        .map(|(codec, _, _)| codec.name())
        .collect())
}

/// The names of the encoders the loaded build has.
pub fn encoders() -> Result<Vec<String>> {
    Ok(runtime()?
        .codecs()
        .into_iter()
        .filter(|(_, _, encodes)| *encodes)
        .map(|(codec, _, _)| codec.name())
        .collect())
}

/// The hardware device types the loaded build can create, by ffmpeg's
/// names for them: `cuda`, `d3d11va`, `vaapi`.
pub fn hw_device_types() -> Result<Vec<String>> {
    Ok(runtime()?.hw_types())
}

/// A picture's colour, by the names ffmpeg's command line gives each value
/// (`-color_range tv -color_primaries bt709 ...`). A decoder reports
/// `unknown` for what the stream does not say; an encoder leaves an empty
/// field, or `unknown`, unspecified.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Color {
    /// `tv` or `pc`.
    pub range: String,
    pub primaries: String,
    pub trc: String,
    pub space: String,
}

/// A string as JSON.
fn json_string(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 2);
    out.push('"');
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn json_list(items: &[String]) -> String {
    let items: Vec<String> = items.iter().map(|i| json_string(i)).collect();
    format!("[{}]", items.join(","))
}

/// The names of the filters the loaded libavfilter has; empty where the
/// directory has none.
pub fn filters() -> Result<Vec<String>> {
    Ok(runtime()?.filter_names())
}

/// What `--av-info` prints, one line of JSON: the majors this build demands,
/// the platform, the directory resolved, and what the libav there is and
/// offers. It answers on a machine with no libav at all: the refusal is in
/// `error`, and what could not be read is null or empty. libavfilter is
/// answered for on its own: its version in `avfilter`, its filters in
/// `filters`, and why it is not loaded in `avfilter_error`.
pub fn info() -> String {
    let demands = Library::ALL
        .iter()
        .map(|l| format!("\"{}\":{}", l.name(), l.major()))
        .collect::<Vec<_>>()
        .join(",");
    let (dir, from) = match resolve_dir() {
        Ok((dir, from)) => (json_string(&dir.to_string_lossy()), json_string(from)),
        Err(_) => ("null".to_string(), "null".to_string()),
    };
    let null = || "null".to_string();
    let (avfilter, avfilter_error, filters) = match runtime() {
        Ok(lib) => match lib.filter_lib() {
            Ok(_) => (
                json_string(&version_text(lib.version(Library::Avfilter))),
                null(),
                json_list(&lib.filter_names()),
            ),
            Err(e) => (null(), json_string(&e.to_string()), "[]".to_string()),
        },
        Err(_) => (null(), null(), "[]".to_string()),
    };
    let (avcodec, ffmpeg, license, error, hw, decoders, encoders) = match runtime() {
        Ok(lib) => {
            let codecs = lib.codecs();
            let names = |pick: fn(&(ffi::Codec, bool, bool)) -> bool| {
                codecs
                    .iter()
                    .filter(|c| pick(c))
                    .map(|(c, _, _)| c.name())
                    .collect::<Vec<_>>()
            };
            (
                json_string(&version_text(lib.version(Library::Avcodec))),
                json_string(&lib.version_info()),
                json_string(&lib.license()),
                null(),
                json_list(&lib.hw_types()),
                json_list(&names(|c| c.1)),
                json_list(&names(|c| c.2)),
            )
        }
        Err(e) => (
            null(),
            null(),
            null(),
            json_string(&e.to_string()),
            "[]".to_string(),
            "[]".to_string(),
            "[]".to_string(),
        ),
    };
    format!(
        "{{\"demands\":{{{demands}}},\"platform\":\"{}\",\"dir\":{dir},\"from\":{from},\
         \"avcodec\":{avcodec},\"ffmpeg\":{ffmpeg},\"license\":{license},\"error\":{error},\
         \"hw_device_types\":{hw},\"decoders\":{decoders},\"encoders\":{encoders},\
         \"avfilter\":{avfilter},\"avfilter_error\":{avfilter_error},\"filters\":{filters}}}",
        crate::nn::platform()
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_pinned_majors_are_accepted() {
        for lib in Library::LOAD_ORDER {
            check_major(lib, (lib.major() << 16) | (1 << 8) | 102).unwrap();
        }
    }

    #[test]
    fn another_major_is_refused_naming_both() {
        let err = check_major(Library::Avcodec, (62 << 16) | (11 << 8) | 100).unwrap_err();
        let text = err.to_string();
        assert!(text.contains("libavcodec is version 62.11.100"), "{text}");
        assert!(text.contains("demands major 63"), "{text}");
        assert!(
            text.contains("pip install --force-reinstall ffrwd-wasm"),
            "{text}"
        );
    }

    #[test]
    fn the_file_names_are_ffmpegs_own() {
        let name = Library::Avcodec.file_name();
        if cfg!(windows) {
            assert_eq!(name, "avcodec-63.dll");
        } else if cfg!(target_os = "macos") {
            assert_eq!(name, "libavcodec.63.dylib");
        } else {
            assert_eq!(name, "libavcodec.so.63");
        }
    }

    #[test]
    fn json_strings_escape_what_json_requires() {
        assert_eq!(json_string("a\"b\\c\nd"), "\"a\\\"b\\\\c\\nd\"");
    }
}
