//! What the `ffrwd-wasm` binary is built from. Only the wire lives here so
//! far: the binary's own frame loop and argument handling stay in `main.rs`,
//! and tests reach the wire through this crate.

pub mod nut;
