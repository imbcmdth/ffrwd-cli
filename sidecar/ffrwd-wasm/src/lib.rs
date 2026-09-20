//! What the `ffrwd-wasm` binary is built from. The binary's own frame loop
//! and argument handling stay in `main.rs`; the wire itself is now the
//! `ffrwd-nut` crate, re-exported here so this crate's tests and the binary
//! reach it by the name they always used.

pub use ffrwd_nut as nut;
