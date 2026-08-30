# The inference edge

What the sidecar accepts on the command line for `wasi:nn`, and what a module
gets back. The compiler builds this argv.

## What a module asks for

A module imports `wasi:nn` - the standard interface, not one of ours - and
calls `load-by-name` with a name. The host reads the model file, builds the
session, and registers it under that name before anything is instantiated. The
module never sees a path and never opens a file: it has no preopened
directory, so the model it just ran is as unreachable to it as everything else
on the disk.

A component that does not import `wasi:nn` gets an unchanged host. The
interface is linked only for the components whose type declares it.

## The argv

    -nn <name>=<path>       bind a model file to a name, repeatable
    -nn-runtime <dir>       where ONNX Runtime is, else FFRWD_NN_RUNTIME
    -nn-target <provider>   where sessions run, else FFRWD_NN_TARGET; cpu is
                            the default

`-nn` is spelled like `-m` and validated the same way: a name is letters,
digits and underscores, and each name may be bound once.

    ffrwd-wasm -f nut -i - \
      -nn yolo=models/yolov8n.onnx \
      -nn-runtime <dir> \
      -m detect=detect.wasm \
      -filter_complex "[0:v]detect[out0]" \
      -map [out0] -f ndjson rows.ndjson

The target is one flag for the run, not a suffix on each binding: a path on
Windows starts `C:\`, so a `name=path:target` spelling cannot be split on a
colon without guessing.

## The targets

| target | what it means |
|---|---|
| `cpu` | the CPU provider; the default |
| `gpu` | the best provider this platform and this runtime directory can offer |
| `cuda` | NVIDIA, through the CUDA execution provider |
| `directml` | any Direct3D 12 adapter, through DirectML; Windows only |
| `coreml` | Apple's, through CoreML; macOS only |

`gpu` walks a priority order and takes the first whose prerequisites resolve:

| platform | order |
|---|---|
| Windows | DirectML, CUDA, CPU |
| macOS | CoreML, CPU |
| Linux | CUDA, CPU |

Windows takes DirectML first. It runs on any Direct3D 12 adapter and needs
nothing installed, where CUDA needs a CUDA 12 runtime and cuDNN 9 on the
machine; CUDA is still reached by naming it.

It says in one line why it passed over each provider it did not take:

    [nn] -nn-target gpu: not DIRECTML, no onnxruntime.dll in
    C:\...\1.22.0\win-x64\directml; `ffrwd setup nn` puts the DirectML build there
    [nn] execution provider: CUDA

A provider named outright is taken as named. One that cannot engage is a
warning and a run on the CPU, never a refusal - the answer is the same, and
slower:

    [nn] -nn-target directml was requested but the DIRECTML execution provider
    did not load; yolo runs on the CPU

A spelling that is not a target is refused, since a misspelling would
otherwise run on the CPU in silence:

    -nn-target dml: the targets are cpu, gpu, cuda, directml and coreml

The three options are read before the argv is dispatched, so `--describe` and
`--invoke` take them too:

    ffrwd-wasm -nn tiny=model.onnx -nn-runtime <dir> --invoke probe.wasm run '{...}'

## Which modules need one

`--describe` carries `"nn"`, always present: whether the component imports
`wasi:nn`, and so whether a run of it needs a `-nn` binding at all. It is read
off the component's imports, so it is answered with no model bound and no
ONNX Runtime on the machine - which is what lets the compiler decide to emit a
binding before it has one.

    "reads_rows": false, "forwards_rows": true, "inputs": 1, "nn": true

## Where the model comes from

A module's fetch script leaves the graph in its own `model/` directory and
also puts a copy beside the built wasm, named after the module:

    modules/target/wasm32-wasip2/release/depth.wasm    depth.onnx
    modules/target/wasm32-wasip2/release/segment.wasm  segment.onnx

That is the convention `-nn` bindings are emitted from: the model a module
asks for by name sits next to it, under that name.

## What this build demands

    $ ffrwd-wasm --nn-info
    {"ort_version":"1.22.0","providers":["directml","cuda","cpu"],"platform":"win-x64"}

`ort_version` is the ONNX Runtime this build loads, read off the `ort` crate
rather than written down twice - `ort` refuses a library whose version string
is not `1.<minor>.x`, naming both. `providers` is the order `-nn-target gpu`
walks here. `platform` is the key a fetched runtime is filed under, spelled
the way ONNX Runtime names its own release archives.

It is answered before any argv is read and with no runtime on the machine,
which is what lets a caller ask what to fetch before it has anything.

## Where the runtime comes from

`-nn-runtime <dir>`, else `FFRWD_NN_RUNTIME`. Absent, a run binding a model is
refused before anything loads. The target is `-nn-target`, else
`FFRWD_NN_TARGET`; argv wins, and a variable that names no target is refused
rather than quietly taken as the default.

The compiler fills that directory and names it. A query reaching a module that
runs a model provisions one before spawning anything; `ffrwd setup nn` does
the same ahead of time, for a CI image or a machine about to lose its network;
`ffrwd install` does it for a package whose modules run models. Setting
`FFRWD_NN_RUNTIME` stops all of it - the variable is a directory of your own,
and the sidecar reads it directly.

    ~/.cache/ffrwd/nn-runtime/<ort-version>/<platform>/

Version-scoped, so a sidecar upgrade cannot load the set fetched for the
version before it; under the cache, because every byte of it is
redownloadable.

## The tiers

| tier | what | download |
|---|---|---|
| cpu | `onnxruntime.dll` | 72 MB |
| directml | `directml/onnxruntime.dll` and `directml/DirectML.dll` | 220 MB |
| cuda | the CUDA build of `onnxruntime.dll`, `onnxruntime_providers_shared.dll`, `onnxruntime_providers_cuda.dll` | 298 MB |
| full | also a pinned CUDA 12 and cuDNN 9 set | 1.4 GB |

What a query fetches for itself is per platform: Windows takes cpu and
directml, 47 MB on disk; Linux takes cpu, and cuda beside it when an NVIDIA
driver is there; macOS takes its stock archive, which carries CoreML inside
it. The `cuda` tier on Windows is `ffrwd setup nn --cuda` and nothing else -
it is six times the DirectML download for a provider the machine may not have
the prerequisites for.

    ffrwd setup nn                  what this platform takes anyway
    ffrwd setup nn --cuda           also the CUDA execution provider
    ffrwd setup nn --cuda --full    also the CUDA 12 and cuDNN 9 it needs

The CUDA 12 runtime and cuDNN 9 are the machine's, like the driver. `--cuda`
carries the provider and expects to find them installed; `--full` is the
escape hatch for a machine that has neither. DirectML asks for nothing
installed beyond a Direct3D 12 adapter, which is every recent Windows machine.

Every artifact is pinned - a URL, a sha256, and its exact byte count, per
runtime version and platform, in the compiler's `nn.py`. The download stops at
the first block past the pinned size, the archive is verified whole before
anything is opened, each library is written through a temporary file and moved
onto its name, the archive is deleted after, and a tier lands completely or
not at all.

Pinned today: ONNX Runtime 1.22.0, DirectML 1.15.4, CUDA 12.9.1
redistributables, cuDNN 9.10.2.

## What DirectML needs on disk

The DirectML execution provider is compiled into ONNX Runtime rather than
loaded beside it the way the CUDA one is. Its `onnxruntime.dll` is therefore a
different binary from the CPU and CUDA one, carrying the same name, and it
cannot share a directory with them:

    <dir>/onnxruntime.dll                 CPU, or the CUDA build with --cuda
    <dir>/onnxruntime_providers_cuda.dll  the CUDA provider
    <dir>/directml/onnxruntime.dll        the DirectML build
    <dir>/directml/DirectML.dll           DirectML itself

`ort` under `load-dynamic` needs exactly two files for the provider: the
DirectML build of the runtime, named by full path, and a `DirectML.dll` the
loader can find from it. Nothing else - no import library, no
`onnxruntime_providers_shared.dll`, which belongs to the CUDA provider.

So the host resolves the provider before the runtime, and loads the
`onnxruntime.dll` that goes with the answer. Only that directory is added to
the search path, so the other build cannot answer for it.

Windows ships a `DirectML.dll` of its own in System32 and a recent one works.
The fetched copy is the version ONNX Runtime 1.22.0 was built against, and it
is found first because its directory is added by hand and user directories are
searched before System32.

The two DirectML artifacts come from nuget rather than the ONNX Runtime
release: `Microsoft.ML.OnnxRuntime.DirectML` 1.22.0 for the runtime, and
`Microsoft.AI.DirectML` 1.15.4 - the version that package names as its
dependency - for DirectML. Each is a zip, and one library is taken out of each
by its full path inside it: both packages carry a build per architecture under
the same leaf name.

## What the other platforms take

The same ONNX Runtime release publishes the rest, and the host already looks
for these names on these platforms:

| platform | archive | libraries |
|---|---|---|
| linux-x64 | `onnxruntime-linux-x64-1.22.0.tgz` | `libonnxruntime.so` |
| linux-x64, cuda | `onnxruntime-linux-x64-gpu-1.22.0.tgz` | also `libonnxruntime_providers_shared.so`, `libonnxruntime_providers_cuda.so` |
| osx-arm64 | `onnxruntime-osx-arm64-1.22.0.tgz` | `libonnxruntime.dylib`, CoreML inside it |
| osx-x86_64 | `onnxruntime-osx-x86_64-1.22.0.tgz` | the same |

Those archives carry one real file and a chain of symlinks to it. What is
written out is the same set under the other arrangement: the real bytes land
at the name the host loads, `libonnxruntime.so`, and the versioned names an
soname lookup may ask for - `libonnxruntime.so.1`, `libonnxruntime.so.1.22.0` -
are links to it. The runtime is loaded by full path either way; the links are
for anything that goes looking by soname.

## Loading discipline

The runtime is named by full path, never by bare name: Windows ships an
onnxruntime of its own in System32, and that is the one a bare name finds. The
directory is added to the process's library search path directly rather than
to `PATH`, and so is each directory that answered for a CUDA dependency, so
only directories the host resolved can satisfy a load.

Every provider is resolved before a session is built, so a missing
prerequisite is named rather than discovered as a silent fall back to the CPU.

CUDA wants `cudart64_12`, `cublas64_12`, `cublasLt64_12`, `cufft64_11`,
`cudnn64_9` and the cuDNN sublibraries, and says what is missing and what is
installed instead:

    [nn] cudart64_12.dll was not found; cudart64_13.dll is in C:\...\CUDA\v13.3\bin\x64

DirectML wants its own build of the runtime, a `DirectML.dll`, and a Direct3D
12 device. The device is asked for by loading `d3d12.dll` - the operating
system's, and the one thing here taken by name rather than by path - and
calling `D3D12CreateDevice` with a null device pointer, which answers whether
one could be created without creating it.

After the session is built, the host reports the provider that actually
engaged and where each native library came from, because ONNX Runtime falls
back to its CPU provider in silence:

    [nn] execution provider: DIRECTML
    [nn]   C:\...\1.22.0\win-x64\directml\DirectML.dll (fetched)
    [nn]   C:\...\1.22.0\win-x64\directml\onnxruntime.dll (fetched)
    [nn]   C:\WINDOWS\SYSTEM32\d3d12.dll (system)

The verdict is read off the process's loaded images, not off what was asked
for: `onnxruntime_providers_cuda.dll` means CUDA, `DirectML.dll` means
DirectML, neither means CPU. Where the images cannot be enumerated the verdict
says `unknown on this platform` rather than guessing, which is what macOS gets.

`fetched` is under the runtime directory, `system` is the operating system's,
`machine` is anywhere else - typically a CUDA toolkit install. A target that
asked for a provider and ended up on the CPU says so, naming the models that
ran there. The run continues; the answer is the same, and slower.

## Refusals

A model bound with no runtime to load it:

    -nn needs an ONNX Runtime: name its directory with -nn-runtime <dir> or set
    FFRWD_NN_RUNTIME, then run `ffrwd setup nn` to download one

A component that imports `wasi:nn` in a run that binds no model, refused
before instantiation rather than as a link failure:

    <module> imports wasi:nn, and this run registers no model; bind one with
    -nn name=path

A name the guest asks for that was never bound comes back through the
interface as the spec's own `not-found`, not as a trap.

## The probe module

`modules/nn-probe` is the fixture: it exports the value interface and imports
`wasi:nn`, so it runs without a stream. `run` names a model and returns its
output; `sandbox` reports which paths the guest could read, and the answer is
none.

Its graph is `model/tiny.onnx`, 178 bytes: `y = a @ w + b` over fp32, `a` of
shape [1, 4] and `y` of shape [1, 2], with small integer weights so any
expected output is arithmetic on paper. It is in git, so the tests run on a
bare checkout; `model/make-tiny-onnx.py` regenerates it and is the only thing
that needs python.

## The depth module

`modules/depth` is the one that ships. It exports the stream interface rather
than the value one, so frames reach it the way they reach any module, and it
asks for the name **`depth`**:

    -nn depth=modules/depth/model/model.onnx

The graph is Depth Anything V2 (small), fp32, 94 MB, and is not in git.
`modules/depth/fetch-model.ps1` downloads it against a pinned revision and
sha256. Unlike transcribe the module does not compile it in, so nothing is
rebuilt after a fetch.

It runs the graph at 518x518 - the size the model's own preprocessing uses -
takes `pixel_values` as a planar fp32 tensor and reads `predicted_depth` back.

    ffrwd-wasm -f nut -i in.nut \
      -nn depth=modules/depth/model/model.onnx \
      -nn-runtime <dir> -nn-target gpu \
      -m depth=depth.wasm -m blur_mask=blur_mask.wasm \
      -filter_complex "[0:v]depth[n1];[0:v][n1]blur_mask[out0]" \
      -map [out0] -f nut out.nut

On an RTX 4090, 64x64 frames, the graph itself is what the time goes on:
178 ms a frame on the CPU provider against 20 ms on CUDA, after a fixed
230 ms and 870 ms respectively to start up and build the session.
