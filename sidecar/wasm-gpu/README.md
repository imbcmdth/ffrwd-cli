# ffrwd/gpu

The `ffrwd:gpu` wit package, versioned as this package is versioned:
`ffrwd/gpu@0.1.0` carries `ffrwd:gpu@0.1.0`, immutable, resolvable
forever. A node module includes its `gpu-module` world beside
`ffrwd:av/node-module` from `ffrwd/wasm` to read its video inputs as
textures of the sidecar's device and hand textures back.

## Using it from Rust

Depend on both packages and install them:

```
"dependencies": { "ffrwd/wasm": "0.19.1", "ffrwd/gpu": "0.1.0" }
```

```
ffrwd install ffrwd/wasm ffrwd/gpu
```

A `build.rs` like `ffrwd/wasm`'s copies each package's `wit/` into the
crate, and the crate's own world includes both:

```wit
package my:module;

world module {
  include ffrwd:av/node-module@0.19.1;
  include ffrwd:gpu/gpu-module@0.1.0;
}
```

```rust
wit_bindgen::generate!({
    path: ["wit/av", "wit/gpu", "wit"],
    world: "my:module/module",
    generate_all,
});
```

The module then exports `ffrwd:gpu/ports.memory`, saying which of its
video ports are held on the GPU, and calls `ffrwd:gpu/frames.planes` for
an input frame's planes as textures and `ffrwd:gpu/frames.emit` to hand
an output texture over. `sidecar/NODE-SHAPE.md`, "Frames on the GPU",
has what the host holds a port to; `sidecar/modules/gpu-box` is a
module doing both.

This package carries `wit/gpu.wit` and, under `wit/deps/`, the same
`wasi:webgpu`, `wasi:graphics-context` and `wasi:io` copies `ffrwd/wasm`
carries. The tree's one copy of `gpu.wit` is `sidecar/wit-gpu/gpu.wit`;
the release workflow stages it at pack time, as it stages `av.wit`.
