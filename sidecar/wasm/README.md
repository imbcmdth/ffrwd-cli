# ffrwd/wasm

The `ffrwd:av` wit world, versioned as this package is versioned:
`ffrwd/wasm@0.12.0` carries `ffrwd:av@0.12.0`, immutable, resolvable
forever. Depend on it and your build has the interface a module
targets — no vendored copy to drift, no path into anyone's checkout.

## Using it from Rust

Declare the dependency and install it:

```
"dependencies": { "ffrwd/wasm": "0.12.0" }
```

```
ffrwd install ffrwd/wasm
```

Then a `build.rs` puts the wit where `wit_bindgen::generate!` reads
it:

```rust
use std::{env, fs, path::PathBuf, process::Command};

fn main() {
    let source = match env::var_os("FFRWD_WIT_DIR") {
        Some(dir) => PathBuf::from(dir),
        None => {
            let out = Command::new("ffrwd")
                .args(["path", "ffrwd/wasm"])
                .output()
                .expect("ffrwd on PATH");
            assert!(out.status.success(), "ffrwd path failed");
            PathBuf::from(String::from_utf8(out.stdout).unwrap().trim()).join("wit")
        }
    }
    .join("av.wit");
    let wit = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("wit");
    fs::create_dir_all(&wit).unwrap();
    fs::copy(&source, wit.join("av.wit")).unwrap();
}
```

with `wit_bindgen::generate!({ path: "wit" })` in the crate and
`wit/` in `.gitignore`. `FFRWD_WIT_DIR` overrides the lookup for
builds that already have the wit on disk.

Or skip all of it: `ffrwd init --rust` scaffolds a working module —
this dependency, that build script, a buildable kernel, a recipe —
ready for `cargo build --target wasm32-wasip2 --release` and
`ffrwd publish`.

## The GPU interface

A module granted `-gpu` imports `wasi:webgpu@0.0.1`, the synchronous
version of the wasi-gfx proposal, which the sidecar hosts over wgpu-core.
The copy the sidecar hosts is a superset of the proposal's: two functions
the proposal's IDL does not have, `gpu-command-encoder.write-timestamp`
and `gpu-queue.get-timestamp-period`, and answers where the proposal's
host trapped (`get-compilation-info`, the texture getters). The package
keeps the upstream name and version, so a module built against the
proposal's own files links against the host unchanged, and one built
against this copy may call the additions.

That copy ships here, under `wit/deps/`, beside `av.wit`: `webgpu`,
`graphics-context` (which webgpu's types name; the host refuses every
call of it) and `wasi:io@0.2.0`. A guest binds the package, not a path
into the sidecar's checkout:

```rust
wit_bindgen::generate!({
    path: "wit",
    world: "gpu-module",   // a world of the guest's own, importing wasi:webgpu/webgpu@0.0.1
});
```

with the same `build.rs` as above copying `wit/` whole, `deps/` included.
The tree's one copy is `sidecar/vendor/wasi-webgpu-wasmtime/wit/deps`;
the release workflow stages it at pack time, as it stages `av.wit`.

## Versions

A new world version is published here as a new package version, and
older versions stay: a module built against `0.8.0` keeps resolving
`ffrwd/wasm@0.8.0` as long as the registry stands. The sidecar hosts
every world it ever shipped, so old modules keep running too.
