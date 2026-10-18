//! The feed trace: with `FFRWD_TRACE_FEED` naming a file, a line for each
//! thing a port feed and its hold do, appended to that file as it happens,
//! so a run that is killed, or whose stderr nobody keeps, still leaves it.
//! Each line is the wall time in seconds since the epoch, the milliseconds
//! since the process started, the process id, the thread's name, and what
//! happened. Off, it costs one check of a static per call.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::sync::{Mutex, OnceLock};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

struct Trace {
    file: Mutex<File>,
    start: Instant,
}

static TRACE: OnceLock<Option<Trace>> = OnceLock::new();

fn trace() -> Option<&'static Trace> {
    TRACE
        .get_or_init(|| {
            let path = std::env::var_os("FFRWD_TRACE_FEED")?;
            match OpenOptions::new().create(true).append(true).open(&path) {
                Ok(file) => Some(Trace {
                    file: Mutex::new(file),
                    start: Instant::now(),
                }),
                Err(e) => {
                    eprintln!(
                        "ffrwd-wasm: FFRWD_TRACE_FEED: opening {}: {e}; no trace",
                        path.to_string_lossy()
                    );
                    None
                }
            }
        })
        .as_ref()
}

/// Whether the trace is on.
pub fn on() -> bool {
    trace().is_some()
}

/// Appends one line, written whole by one call so lines from several
/// threads never interleave.
pub fn line(what: std::fmt::Arguments<'_>) {
    let Some(trace) = trace() else {
        return;
    };
    let wall = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0.0, |d| d.as_secs_f64());
    let thread = std::thread::current();
    let line = format!(
        "{wall:.6} {:.3} {} [{}] {what}\n",
        trace.start.elapsed().as_secs_f64() * 1000.0,
        std::process::id(),
        thread.name().unwrap_or("main"),
    );
    if let Ok(mut file) = trace.file.lock() {
        let _ = file.write_all(line.as_bytes());
    }
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
