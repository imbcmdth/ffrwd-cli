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
    -nn-exclude <name>=<provider>
                            keep one model off a provider, repeatable;
                            the bare <provider> keeps every model off it

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

The provider is chosen per model. `-nn-exclude <name>=<provider>` keeps
the model bound as `<name>` off that provider; the bare `-nn-exclude
<provider>` keeps every model off it. `cuda`, `directml` and `coreml` are
the providers it can name, `cpu` is refused, since a model always keeps
somewhere to run, and a name no `-nn` binds is refused. It is repeatable.
The compiler passes one pair per provider a model's own pin denies. A
provider named outright with `-nn-target` is unaffected: `-nn-exclude`
only narrows the `gpu` walk.

    ffrwd-wasm -nn transcribe=whisper.onnx -nn detect=yolo.onnx \
      -nn-runtime <dir> -nn-target gpu -nn-exclude transcribe=directml \
      --invoke transcribe.wasm run '{...}'

One process loads one ONNX Runtime binary, and DirectML's is a different
binary from the CPU and CUDA one, so the walk still settles one provider
for the process: the first in the platform's order whose preflight is
clean and that at least one bound model does not exclude. It says in one
line why it passed over each earlier candidate, whether preflight ruled
it out or every model's pin did:

    [nn] -nn-target gpu: not DIRECTML, no onnxruntime.dll in
    C:\...\1.22.0\win-x64\directml; `ffrwd setup nn` puts the DirectML build there
    [nn] -nn-target gpu: not DIRECTML, excluded by every model's pin

Each model's session then asks for that provider, or for the CPU where
the model excludes it, and the ask does not fall back in silence: the
provider is registered so that a session it cannot engage for fails to
build, and the host says so and builds the session again on the CPU. The
verdict is one line per model:

    [nn] -nn-exclude: transcribe=directml
    [nn] transcribe: execution provider CPU, DIRECTML excluded by its pin
    [nn] detect: execution provider DIRECTML

    [nn] detect: CUDA did not engage (Error loading "...\onnxruntime_providers_shared.dll" which is missing); runs on the CPU
    [nn] detect: execution provider CPU, CUDA did not engage

    [nn] detect: execution provider CPU, no GPU execution provider was available

A provider that registers is reported as engaged; ONNX Runtime may still
run a node it cannot place on the CPU (see "A silent fall back to CPU"
below), which no verdict here catches.

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

### A silent fall back to CPU

cuDNN's version matters more than the CUDA provider admits. A model whose
encoder opens with a convolution -- whisper's does -- fails to build that
node's graph on cuDNN 9.25, and ONNX Runtime does not refuse the session: it
falls back to the CPU provider and runs. The only symptom is the speed, which
is why it survives a smoke test. cuDNN 9.8 builds it. When a `gpu` target
performs like `cpu`, check the cuDNN version before anything else.

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

What a query fetches for itself is what the machine turns out to have,
looked at once by the compiler's `nn.detect`: Windows takes cpu and directml,
DirectML asking for nothing beyond a Direct3D 12 adapter; Linux takes cpu;
either takes cuda wherever an NVIDIA driver shows up - the kernel module's
directory, `nvidia-smi` on PATH, `nvcuda.dll` in the system directory or
`libcuda.so.1` on the loader's path - and full beside it where no CUDA 12
runtime and cuDNN 9 are already on PATH, under CUDA_PATH, in ldconfig's
cache or under LD_LIBRARY_PATH. macOS takes its stock archive, which carries
CoreML inside it. A run fetches the same set on its way, naming what and how
big before it does.

    ffrwd setup nn                  what this machine takes, said in one line first
    ffrwd setup nn --cuda           the CUDA execution provider whatever detection said
    ffrwd setup nn --cuda --full    and the CUDA 12 and cuDNN 9 it needs

The CUDA 12 runtime and cuDNN 9 are the machine's, like the driver, and the
pinned set is for a machine that has neither. Detection decides that; the
two flags add a tier it did not take.

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

    <dir>/onnxruntime.dll                 CPU, or the CUDA build when the cuda tier is taken
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

## Inputs from the pool

A node module that holds its video on the GPU (`ffrwd:gpu`, see
`NODE-SHAPE.md`, "Frames on the GPU") can have the host make a frame into
its model's input tensor, so the module never fetches the frame's bytes. It
includes `ffrwd:gpu/gpu-nn-module` beside the node world, imports the
`wasi:nn` interfaces it calls, and asks:

    frame-tensor: func(id: u32, index: u32, options: frame-tensor-options)
        -> result<tensor, string>;

    record frame-tensor-options {
      model: string,                    // the -nn name
      width: u32, height: u32,          // the tensor's picture size
      layout: tensor-layout,            // nchw or nhwc
      order: channel-order,             // rgb or bgr
      fit: fit,                         // stretch or letterbox
      scale: f32,                       // (value * scale - mean) / std
      mean: tuple<f32, f32, f32>,
      std: tuple<f32, f32, f32>,
    }

The answer is an ordinary `wasi:nn` tensor, fp32, which the module passes
to `compute` like any other. `id` and `index` are the tick's, as
`frames.planes` takes them. The frame is resized bilinearly from pixel
centres (the chroma of a subsampled frame sited at the centre of the pixels
it covers), its YUV made RGB with the stream's matrix and range, and each
channel, read as 0 to 1, becomes `(value * scale - mean) / std`. A
letterbox fills the bars with 0 before the normalisation.

How the tensor reaches the session depends on where the frame is and on the
provider the model's session engaged. The host says which, once per model:

| frame | session | route | line |
|---|---|---|---|
| on the GPU (D3D12) | CUDA, Windows | a compute pass on the shared device writes the tensor into a buffer on a shared heap, which CUDA imported once; the session is handed that memory through an `IoBinding` | `[nn] depth: input from the pool, bound on CUDA` |
| on the GPU | DirectML | the same pass, read back once | `[nn] depth: input downloaded, DIRECTML takes host memory (the pinned ort cannot hand a D3D12 buffer to a DirectML session)` |
| on the GPU | CPU, CoreML | the same pass, read back once | `[nn] depth: input downloaded, CPU takes host memory` |
| on the GPU | CUDA where the import failed | the same pass, read back once | `[nn] depth: input downloaded, CUDA could not reach the pool's memory (<why>)` |
| in host memory | any | the same arithmetic on the CPU, from the frame's bytes | `[nn] depth: input in host memory, made on the CPU` |

On the bound route the queue signals a shared D3D12 sync object after the
pass, CUDA imports that sync object as an external semaphore, and the wait
on it is queued and finished before the session runs. Neither the frame nor
the tensor crosses to host memory: a `-profile` run counts no `download`
bytes. On the
downloaded route each frame costs one read back of the tensor, never of the
frame, and the profile's `download` counts the tensor's bytes.
`FFRWD_NN_BIND=off` takes the downloaded route where the bound one would be
taken (`input downloaded, FFRWD_NN_BIND is off`), so the two can be
compared on one session.

DirectML cannot be bound with the pinned `ort` 2.0.0-rc.10: it carries none
of DirectML's interop API (`OrtDmlApi`,
`CreateGPUAllocationFromD3DResource`), and its DirectML provider takes a
device id, never the caller's D3D12 device, so a DirectML session cannot be
handed a buffer of the shared device. Linux has no bound route yet: the
pool is on Vulkan there, and the Vulkan import is not written.

Outputs come back as before: `compute` returns them in host memory.

The CUDA runtime the binding uses is the one the CUDA provider loaded
(`cudart64_12.dll`, where the provider's preflight found it), loaded at run
time and never linked. The CUDA device is the one whose LUID is the shared
D3D12 adapter's, and it has to be device 0, the one the CUDA provider runs
every session on.

`modules/nn-frames` is the test module: a sink that runs a model on each
frame through `frame-tensor` and writes each output as rows, its values when
there are 16 or fewer, else their count, mean, least and greatest. Its
graphs, `model/channel-means.onnx` (NCHW) and `channel-means-nhwc.onnx`,
return each channel's mean and largest value; `model/make-channel-means.py`
regenerates them. The same frames give the same rows on every route, to
within float rounding.

    ffrwd-wasm -nn means=modules/nn-frames/model/channel-means.onnx \
      -nn-runtime <dir> -nn-target cuda -gpu nn_frames.wasm \
      -f nut -i in.nut -m nn_frames=nn_frames.wasm \
      -filter_complex "[0:v]decode=format=yuv420p:hw=d3d12va[d];[v=d]nn_frames=model=means:memory=gpu:width=518:height=518[@rows=r]" \
      -map [r] -f ndjson rows.ndjson

On the RTX 4090, release build, h264 decoded on the device into the pool,
the tensor 518x518 letterboxed, median per frame over 230 frames, the time
inside the module for `frame-tensor` and `compute` together:

| model | input | bound on CUDA | downloaded, CUDA session |
|---|---|---|---|
| channel-means | 1080p | 1.33 ms | 6.74 ms |
| channel-means | 4K | 1.41 ms | 7.41 ms |
| Depth Anything V2 small | 1080p | 29.4 ms | 34.2 ms |
| Depth Anything V2 small | 4K | 30.7 ms | 40.3 ms |

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
