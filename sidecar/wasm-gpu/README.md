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

A module that runs a model on its frames includes `gpu-nn-module`
instead, and imports the `wasi:nn` interfaces it calls beside it; it
then asks `ffrwd:gpu/nn.frame-tensor` for a frame as the model's input
tensor, made by the host at the size, resampling and normalisation it
names, or `ffrwd:gpu/nn.crops-tensor` for rectangles of a frame as one
batched tensor, and passes that tensor to `compute`; `ffrwd:gpu/nn.stack`
joins tensors kept from several ticks into one. `sidecar/NN-EDGE.md`, "Inputs from the
pool", has the routes; `sidecar/modules/nn-frames` is such a module.

A codec package's module, beside `ffrwd:av/codec-module`, includes
`codec-module` (or `encoder-module` or `decoder-module` for one half) and
exports `ffrwd:gpu/encoder` and `ffrwd:gpu/decoder`: where the sidecar
holds the pictures on the GPU, `encode` is handed each picture's planes as
textures and `decode` hands its pictures back as textures it made, in
place of `ffrwd:av`'s `encode` and `decode`, which still serve elsewhere.
`ffrwd:av`'s `describe` and `init` open it as for host memory.
`sidecar/modules/testcodec`, built as `testcodec_gpu`, is such a module.

A module's `request-device` is answered with the sidecar's one device,
and a feature that device lacks is refused. The device is on the adapter
wgpu picks for high performance, unless `-gpu-adapter` or
`FFRWD_GPU_ADAPTER` names another (`sidecar/NODE-CLI.md`, "The GPU
adapter"); a module's `request-adapter` answers that adapter whatever it
asks for, and the decoder, the encoders and the filters that hold
pictures on the GPU use the same device. On Windows the device is
D3D12, whose `subgroups` need DXC: the release wheel's sidecar for
Windows x86_64 has it compiled in, and any other build needs
`dxcompiler.dll` (the Windows SDK's, or the DirectX Shader Compiler
release's) on the PATH and `WGPU_DX12_COMPILER=dxc`. A request for
`subgroups` refused for want of DXC says so, and so does the sidecar's
stderr, once, when a module asks whether the device has them.

This package carries `wit/gpu.wit` and, under `wit/deps/`, the same
`wasi:webgpu`, `wasi:graphics-context` and `wasi:io` copies `ffrwd/wasm`
carries, and `wasi:nn`, whose tensor `nn` hands out. The tree's one
copy of `gpu.wit` is `sidecar/wit-gpu/gpu.wit`, with its deps beside it;
the release workflow stages them at pack time, as it stages `av.wit`.
