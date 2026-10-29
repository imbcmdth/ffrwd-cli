# ffrwd/models

The `ffrwd:models` wit package, versioned as this package is versioned:
`ffrwd/models@0.1.0` carries `ffrwd:models@0.1.0`, immutable, resolvable
forever. A module includes its `models-module` world beside
`ffrwd:av/node-module` from `ffrwd/wasm` to read the files its manifest
pins under `models` as bytes.

A module that runs its model through `wasi:nn` never needs this: the
host builds the session and the module loads it by name. A module that
runs its model in its own code, on the sidecar's device through
`wasi:webgpu` or in its own arithmetic, needs the weights themselves,
and this is the one way it gets them. It still has no directory, no
path and no file but the ones its manifest pinned.

## Using it

Depend on both packages and install them:

```
"dependencies": { "ffrwd/wasm": "0.19.1", "ffrwd/models": "0.1.0" }
```

```
ffrwd install ffrwd/wasm ffrwd/models
```

The module's own world includes both:

```wit
package my:module;

world module {
  include ffrwd:av/node-module@0.19.1;
  include ffrwd:models/models-module@0.1.0;
}
```

The module then calls `ffrwd:models/files.read(name)` for the bytes of
one pinned file, `name` being the file's name as the hub spells it (the
last segment of the pin's `file`), and `files.names()` for the names it
may ask for. `ffrwd compile` binds every entry of the export's `models`
list for the module with `-model <name>=<path>`, and `ffrwd install`
fetches them. A name nothing bound is an Err naming what is; the host
reads the file on each call, so a module keeps what it reads.

`sidecar/NN-EDGE.md`, "Files a module reads itself", has the sidecar's
argv and refusals; `sidecar/modules/models-probe` is a module reading
one file and answering its length and digest.

This package carries `wit/models.wit` and nothing under `wit/deps/`: the
interface uses no other package. The tree's one copy is
`sidecar/wit-models/models.wit`; the release workflow stages it at pack
time, as it stages `av.wit` and `gpu.wit`.
