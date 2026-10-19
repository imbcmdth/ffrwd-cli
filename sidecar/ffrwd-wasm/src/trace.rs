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

/// The mean luma of a small rgba picture (at most 64x64), read back where it
/// is on the device; None for a larger one or another layout, which would
/// cost the trace a frame's read-back.
pub fn luma(data: &FrameData) -> Option<f64> {
    const SMALL: usize = 64 * 64 * 4;
    let bytes = match data {
        FrameData::Gpu(frame) => {
            let (w, h) = frame.size();
            if (w * h * 4) as usize > SMALL || frame.wire_len() != (w * h * 4) as usize {
                return None;
            }
            data.bytes().ok()?
        }
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

/// A trace line, its arguments formatted only while the trace is on.
macro_rules! trace_feed {
    ($($arg:tt)*) => {
        if $crate::trace::on() {
            $crate::trace::line(format_args!($($arg)*));
        }
    };
}
pub(crate) use trace_feed;
