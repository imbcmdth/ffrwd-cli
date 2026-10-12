//! What the shim refuses, by kind, each with the hint that goes with it.

use std::fmt;
use std::path::PathBuf;

/// What puts the libraries back, named in every refusal that wants them.
/// The sidecar's wheel carries the libav it was built against, beside the
/// binary; the cli says "reinstall ffrwd" for a missing sidecar the same way.
pub const INSTALL_HINT: &str = "reinstall the sidecar, which carries the libav it was built \
     against (`pip install --force-reinstall ffrwd-wasm`), or name a directory holding them \
     with -av-runtime <dir> or FFRWD_AV_RUNTIME";

/// One refusal from the shim.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    /// No directory to look for the libraries in: neither the argv nor the
    /// environment named one, and the executable's own could not be found.
    NoDirectory { why: String },
    /// A library is not in the directory, or is there and would not load.
    /// `from` says how the directory was chosen.
    Missing {
        dir: PathBuf,
        from: &'static str,
        file: String,
        why: Option<String>,
    },
    /// A library of a major other than the one this build was written
    /// against. `found` is its whole version, `major.minor.micro`.
    Version {
        library: &'static str,
        found: String,
        demanded: u32,
    },
    /// A function the library should export and does not.
    Symbol {
        library: String,
        symbol: &'static str,
        why: String,
    },
    /// The loaded libraries lay a struct out differently from the headers
    /// this build was transcribed from.
    Layout { what: String },
    /// The loaded build has no decoder or encoder by that name.
    Codec { role: &'static str, name: String },
    /// A pixel format the wire does not carry, or one an encoder refuses.
    PixelFormat { name: String, why: String },
    /// A hardware device that was named and could not be had.
    Hardware {
        codec: String,
        device: String,
        why: String,
    },
    /// An option the encoder refused, under the name ffmpeg's command line
    /// gives it.
    Option {
        encoder: String,
        key: String,
        value: String,
        why: String,
    },
    /// A frame whose bytes are not the size its format and geometry make.
    FrameSize {
        pix_fmt: String,
        width: u32,
        height: u32,
        expected: usize,
        got: usize,
    },
    /// A filter graph libavfilter would not make, or a device it runs on
    /// that cannot be had: the graph as written, the call, and why.
    Graph {
        graph: String,
        call: &'static str,
        why: String,
    },
    /// A call into libav that failed, with libav's own words for why.
    Call {
        call: &'static str,
        codec: String,
        why: String,
    },
}

impl Error {
    /// What to do about it, where there is something to do.
    pub fn hint(&self) -> Option<String> {
        match self {
            Error::NoDirectory { .. }
            | Error::Missing { .. }
            | Error::Version { .. }
            | Error::Symbol { .. }
            | Error::Layout { .. } => Some(INSTALL_HINT.to_string()),
            Error::Codec { role, .. } => Some(format!(
                "`ffrwd-wasm --av-info` lists the {role}s this build has"
            )),
            Error::PixelFormat { .. } => None,
            Error::Hardware { .. } => Some(
                "`ffrwd-wasm --av-info` lists the hardware device types this build has; \
                 decode with hw auto to fall back to software instead"
                    .to_string(),
            ),
            Error::Option { encoder, .. } => Some(format!(
                "`ffmpeg -h encoder={encoder}` lists the options it takes"
            )),
            Error::Graph { .. } => Some(
                "`ffrwd-wasm --av-info` lists the filters this build has; libav's own line \
                 above names the filter or option it refused"
                    .to_string(),
            ),
            Error::FrameSize { .. } | Error::Call { .. } => None,
        }
    }

    fn message(&self) -> String {
        match self {
            Error::NoDirectory { why } => {
                format!("no directory to load libav from: {why}")
            }
            Error::Missing {
                dir,
                from,
                file,
                why: None,
            } => format!("no {file} in {} ({from})", dir.display()),
            Error::Missing {
                dir,
                from,
                file,
                why: Some(why),
            } => format!(
                "{} ({from}) could not be loaded: {why}",
                dir.join(file).display()
            ),
            Error::Version {
                library,
                found,
                demanded,
            } => {
                format!("lib{library} is version {found}, and this build demands major {demanded}")
            }
            Error::Symbol {
                library,
                symbol,
                why,
            } => format!("{library} does not export {symbol}: {why}"),
            Error::Layout { what } => format!(
                "the loaded libav lays its structs out differently from the headers this \
                 build was written against: {what}"
            ),
            Error::Codec { role, name } => {
                format!("this libav has no {role} called {name}")
            }
            Error::PixelFormat { name, why } => format!("pixel format {name}: {why}"),
            Error::Hardware { codec, device, why } => {
                format!("decoding {codec} on {device}: {why}")
            }
            Error::Option {
                encoder,
                key,
                value,
                why,
            } => format!("{encoder} refused the option {key}={value}: {why}"),
            Error::FrameSize {
                pix_fmt,
                width,
                height,
                expected,
                got,
            } => format!(
                "a {width}x{height} {pix_fmt} frame is {expected} bytes, and this one is {got}"
            ),
            Error::Graph { graph, call, why } => {
                format!("the filter graph '{graph}': {call}: {why}")
            }
            Error::Call { call, codec, why } => format!("{call} ({codec}): {why}"),
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.hint() {
            Some(hint) => write!(f, "{}; {hint}", self.message()),
            None => f.write_str(&self.message()),
        }
    }
}

impl std::error::Error for Error {}

/// The shim's own result.
pub type Result<T> = std::result::Result<T, Error>;
