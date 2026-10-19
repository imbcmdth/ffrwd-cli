//! The feed trace (`FFRWD_TRACE_FEED`), kept in the runtime crate so the
//! host functions a module calls can write to it too.

pub use ffrwd_wasm_runtime::trace::{line, on};

/// A trace line, its arguments formatted only while the trace is on.
macro_rules! trace_feed {
    ($($arg:tt)*) => {
        if $crate::trace::on() {
            $crate::trace::line(format_args!($($arg)*));
        }
    };
}
pub(crate) use trace_feed;
