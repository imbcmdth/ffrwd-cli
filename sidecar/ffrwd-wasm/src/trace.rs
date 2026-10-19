//! The feed trace (`FFRWD_TRACE_FEED`), kept in the runtime crate so the
//! host functions a module calls can write to it too.

pub use ffrwd_wasm_runtime::trace::{line, on};

use ffrwd_wasm_runtime::node::FrameData;

/// The largest sample of a run, as a fraction of full scale: `f32` or `s16`
/// interleaved, the run's own format.
pub fn peak(bytes: &[u8], sample_fmt: &str) -> f32 {
    match sample_fmt {
        "s16" => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| (i16::from_le_bytes(*b) as f32 / 32768.0).abs())
            .fold(0.0, f32::max),
        _ => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b).abs())
            .filter(|v| v.is_finite())
            .fold(0.0, f32::max),
    }
}

/// The mean luma of an rgba picture: read back whole where it is at most
/// 64x64, and where a larger one is on the device, made 32x32 there first
/// so the read-back stays small; None for a large picture in host memory or
/// another layout.
pub fn luma(data: &FrameData) -> Option<f64> {
    const SMALL: usize = 64 * 64 * 4;
    let bytes = match data {
        FrameData::Gpu(frame) => on_device(frame, data, SMALL)?,
        FrameData::Host(bytes) => {
            if bytes.len() > SMALL || bytes.len() % 4 != 0 {
                return None;
            }
            std::sync::Arc::clone(bytes)
        }
    };
    let px = bytes.as_chunks::<4>().0;
    if px.is_empty() {
        return None;
    }
    let sum: f64 = px
        .iter()
        .map(|p| 0.299 * p[0] as f64 + 0.587 * p[1] as f64 + 0.114 * p[2] as f64)
        .sum();
    Some(sum / px.len() as f64)
}

/// An rgba frame on the device as bytes: read back whole where it is at
/// most `small` bytes, made 32x32 there first where it is larger.
#[cfg(feature = "gpu")]
fn on_device(
    frame: &std::sync::Arc<ffrwd_wasm_runtime::gpu::frames::GpuFrame>,
    data: &FrameData,
    small: usize,
) -> Option<std::sync::Arc<Vec<u8>>> {
    use ffrwd_wasm_runtime::gpu::frames::{Layout, Target, Yuv};
    let (w, h) = frame.size();
    if frame.wire_len() != (w * h * 4) as usize {
        return None;
    }
    if (w * h * 4) as usize <= small {
        return data.bytes().ok();
    }
    let reduced = frame
        .converted(Target::new(Layout::Rgba, 32, 32, Yuv::DEFAULT))
        .ok()?;
    reduced.bytes().ok()
}

#[cfg(not(feature = "gpu"))]
fn on_device(
    _frame: &std::sync::Arc<ffrwd_wasm_runtime::gpu::frames::GpuFrame>,
    _data: &FrameData,
    _small: usize,
) -> Option<std::sync::Arc<Vec<u8>>> {
    None
}

/// The wgpu ids of a frame's planes on the device, as ` textures ...`, for
/// the hold's tick line; nothing for a frame in host memory.
#[cfg(feature = "gpu")]
pub fn textures(data: &FrameData) -> String {
    data.gpu().map_or(String::new(), |g| {
        let ids: Vec<String> = g
            .planes()
            .iter()
            .map(|p| format!("{:?}", p.texture))
            .collect();
        format!(" textures {}", ids.join(" "))
    })
}

#[cfg(not(feature = "gpu"))]
pub fn textures(_data: &FrameData) -> String {
    String::new()
}

/// A trace line, its arguments formatted only while the trace is on.
macro_rules! trace_feed {
    ($($arg:tt)*) => {
        if $crate::trace::on() {
            $crate::trace::line(format_args!($($arg)*));
        }
    };
}
pub(crate) use trace_feed;
