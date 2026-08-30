# ffrwd/wasm

The `ffrwd:av` wit world, versioned as this package is versioned:
`ffrwd/wasm@0.10.0` carries `ffrwd:av@0.10.0`, immutable, resolvable
forever. Depend on it and your build has the interface a module
targets — no vendored copy to drift, no path into anyone's checkout.

## Using it from Rust

Declare the dependency and install it:

```
"dependencies": { "ffrwd/wasm": "0.10.0" }
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

## Versions

A new world version is published here as a new package version, and
older versions stay: a module built against `0.8.0` keeps resolving
`ffrwd/wasm@0.8.0` as long as the registry stands. The sidecar hosts
every world it ever shipped, so old modules keep running too.
