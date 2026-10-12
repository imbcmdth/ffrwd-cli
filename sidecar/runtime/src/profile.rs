//! Where a run's bytes and time go: counted per stage as the run goes, and
//! written as one row at the end behind `-profile`.
//!
//! Each stage counts the bytes that crossed it, how many times, and the
//! time that took. The counters are the process's and relaxed, so counting
//! costs a few atomic adds per frame whether or not the row is asked for,
//! and a run at any `-jobs` sums to the same bytes.

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::time::Instant;

/// One stage's counters.
pub struct Stage {
    name: &'static str,
    bytes: AtomicU64,
    calls: AtomicU64,
    nanos: AtomicU64,
}

impl Stage {
    const fn new(name: &'static str) -> Stage {
        Stage {
            name,
            bytes: AtomicU64::new(0),
            calls: AtomicU64::new(0),
            nanos: AtomicU64::new(0),
        }
    }

    /// Counts `bytes` crossing once, taking from `since` until now.
    pub fn add(&self, bytes: usize, since: Instant) {
        self.count(bytes);
        let nanos = u64::try_from(since.elapsed().as_nanos()).unwrap_or(u64::MAX);
        self.nanos.fetch_add(nanos, Relaxed);
    }

    /// Counts `bytes` crossing once, with no time of its own.
    pub fn count(&self, bytes: usize) {
        self.bytes.fetch_add(bytes as u64, Relaxed);
        self.calls.fetch_add(1, Relaxed);
    }

    pub fn bytes(&self) -> u64 {
        self.bytes.load(Relaxed)
    }

    pub fn calls(&self) -> u64 {
        self.calls.load(Relaxed)
    }

    pub fn seconds(&self) -> f64 {
        self.nanos.load(Relaxed) as f64 / 1e9
    }

    fn json(&self) -> String {
        format!(
            r#""{}":{{"bytes":{},"calls":{},"time_s":{:?}}}"#,
            self.name,
            self.bytes(),
            self.calls(),
            self.seconds()
        )
    }
}

/// The stages a frame crosses between the input pipes and the output pipes.
pub struct Profile {
    /// Bytes read off the input pipes.
    pub read: Stage,
    /// Frames placed in ticks, as the host holds them.
    pub held: Stage,
    /// Bytes handed to modules by `fetch`.
    pub fetch: Stage,
    /// Time inside modules' `process` calls; no bytes of its own.
    pub process: Stage,
    /// Frames lifted out of modules.
    pub emit: Stage,
    /// Host frames put on the GPU.
    pub upload: Stage,
    /// GPU frames read back.
    pub download: Stage,
    /// Bytes written to the output pipes.
    pub write: Stage,
}

/// The process's counters.
pub static PROFILE: Profile = Profile {
    read: Stage::new("read"),
    held: Stage::new("held"),
    fetch: Stage::new("fetch"),
    process: Stage::new("process"),
    emit: Stage::new("emit"),
    upload: Stage::new("upload"),
    download: Stage::new("download"),
    write: Stage::new("write"),
};

impl Profile {
    fn stages(&self) -> [&Stage; 8] {
        [
            &self.read,
            &self.held,
            &self.fetch,
            &self.process,
            &self.emit,
            &self.upload,
            &self.download,
            &self.write,
        ]
    }

    /// The row: `{"kind":"profile","read":{"bytes":N,"calls":N,"time_s":S},...}`,
    /// one object per stage in the order a frame crosses them.
    pub fn row(&self) -> String {
        let stages: Vec<String> = self.stages().iter().map(|s| s.json()).collect();
        format!(r#"{{"kind":"profile",{}}}"#, stages.join(","))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_stage_sums_what_crossed_it_and_the_row_lists_every_stage() {
        let stage = Stage::new("test");
        stage.count(10);
        stage.add(5, Instant::now());
        assert_eq!((stage.bytes(), stage.calls()), (15, 2));
        assert!(stage.seconds() >= 0.0);
        assert!(stage
            .json()
            .starts_with(r#""test":{"bytes":15,"calls":2,"time_s":"#));

        let row: serde_json::Value =
            serde_json::from_str(&PROFILE.row()).expect("the row is one JSON object");
        assert_eq!(row["kind"], "profile");
        for name in [
            "read", "held", "fetch", "process", "emit", "upload", "download", "write",
        ] {
            assert!(row[name]["bytes"].is_u64(), "{name} in {row}");
            assert!(row[name]["calls"].is_u64(), "{name} in {row}");
            assert!(
                row[name]["time_s"].is_f64(),
                "{name} in {row}: a time is written with its point"
            );
        }
    }
}
