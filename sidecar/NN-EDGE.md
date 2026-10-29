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
    -nn-for <module>        bind the -nn before it for this module alone,
                            repeatable
    -nn-runtime <dir>       where ONNX Runtime is, else FFRWD_NN_RUNTIME
    -nn-target <provider>   where sessions run, else FFRWD_NN_TARGET; cpu is
                            the default
    -nn-threads <n>         intra-op threads each session runs on, else
                            FFRWD_NN_THREADS; ONNX Runtime's default pool
                            when neither says
    -nn-exclude <name>=<provider>
                            keep one model off a provider, repeatable;
                            the bare <provider> keeps every model off it

`-nn` is spelled like `-m` and validated the same way: a name is letters,
digits and underscores, and each name may be bound once for every module.
`-nn-for <module>` after a `-nn` narrows it to that module, named by the
path its `-m` gives, so two modules asking for one name can each load a
file of its own; a module's own binding wins over one for every module,
and a name bound twice for one module is refused. `-nn-exclude` names
every binding of its name.

    ffrwd-wasm ... -nn ages=a/ages.onnx -nn-for a/ages.wasm \
      -nn ages=b/ages.onnx -nn-for b/ages_gpu.wasm ...

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
for the process. It takes the first in the platform's order whose
preflight is clean and that no bound model excludes, so that every model
runs on it; where there is none, the first whose preflight is clean and
that at least one bound model does not exclude, the models excluding it
then running on the CPU. A process binding `detect_faces`, which pins
nothing, beside `ages`, which keeps off DirectML, settles CUDA and runs
both there, rather than DirectML with `ages` on the CPU. It says in one
line why it passed over each earlier candidate, whether preflight ruled
it out or a model's pin did:

    [nn] -nn-target gpu: not DIRECTML, no onnxruntime.dll in
    C:\...\1.22.0\win-x64\directml; `ffrwd setup nn` puts the DirectML build there
    [nn] -nn-target gpu: not DIRECTML, excluded by the pin of ages
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

## Files a module reads itself

A module that runs its model in its own code, rather than through a
`wasi:nn` session the host builds, needs the weights as bytes. It imports
`ffrwd:models/files@0.1.0` (`sidecar/wit-models/models.wit`) and the host
hands over exactly the files bound on the command line:

    -model <name>=<path>    bind a file under a name, repeatable
    -model-for <module>     bind the -model before it for this module alone,
                            repeatable

`<name>` is one path segment - no `/` or `\`, not `.` or `..` - and the
file must exist when the argv is read; the run is refused otherwise, before
anything is instantiated. `-model-for` narrows a binding the way `-nn-for`
does: a module's own binding of a name wins over one bound for every
module, and a name bound twice for one module is refused. The flags are
read before dispatch, so `--describe`, `--invoke` and the frame loop all
see them.

The module calls `files.read(name)` and gets the whole file, read on each
call, or an Err naming what is bound for it:

    no file named 'v.bin' is bound for this module; the files bound for it are w.bin
    no file named 'v.bin' is bound for this module; no file is bound for it

`files.names()` lists the bound names in argv order. The module still has
no directory and no path: the path behind a name never reaches it, and a
read that fails on the host says `'<name>' could not be read` and no more.
The interface is linked only for components whose type imports it.

`--describe` carries `"models"`, always present, read off the imports like
`"nn"`: whether the component reads its files this way. The compiler
writes one `-model` per file the module's package pins for its export,
named by the pin's file name (`docs/dialect.md`, "Models").
`modules/models-probe` is the test module: `--invoke models_probe.wasm
read '{"name":"w.bin"}'` answers the length and sha256 of what it read
beside the names bound for it.

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

The intra-op threads are `-nn-threads`, else `FFRWD_NN_THREADS`, a whole
number of one or more; anything else is refused. Unset, a session gets ONNX
Runtime's default pool, one thread a core, whose threads spin for some
tens of milliseconds after every run. A small model run many times a second
then keeps every core busy between runs: a 160 ms speech step took eight
cores of a twelve-thread machine on the default pool and a third of one core
on one thread, at the same speed. One thread is no pool, so nothing spins.
Pinning the process does not help: ONNX Runtime pins its pool threads to
cores of its own choosing.

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

## CoreML

On macOS the GPU walk takes CoreML, which ONNX Runtime carries inside its
one library. The host registers it in the MLProgram format, which runs more
of a model than the default NeuralNetwork format and runs it on the GPU, on
the CPU and GPU only, and gives it a model cache, `coreml-cache` in the
runtime's directory. The first run that loads a model compiles it, which
takes from a few seconds to two minutes; later runs read the compiled
model back in about a second. The Neural Engine is left out: CoreML
compiles a model for it again on every load, which no cache keeps, 45 s
for X-CLIP's video tower on every run, and it ran neither depth nor
X-CLIP faster than the GPU.

CoreML runs a graph with free dimensions mostly on the CPU. Where every
free dimension of a model's inputs is named (`batch_size`, `height`,
`width`), the session is built at the first run instead, with those
dimensions fixed to the shapes of that run's inputs, and again for each
other shape that arrives. A model with an unnamed free dimension is built
as it is. A model whose batch changes from run to run compiles once per
batch size.

A session built for a shape that cannot be built, or fails its first run,
gives way to the session CoreML builds for the model by default, for every
shape from then on, and stderr says so once:

    [nn] ages: a COREML session built for its inputs' shapes failed (...);
    the model runs on the session COREML builds by default

faceage's age model takes that route: CoreML cannot parse it as an ML
program ("Required param 'pad' is missing").

| Depth Anything V2 small, 518x518, M1 Max | a frame |
|---|---|
| CoreML, free dimensions left | 600 ms |
| CPU | 330 ms |
| CoreML, built for the input's shape | 52 ms |

CoreML reads and writes host memory: a tensor made on the device from a
frame in the pool is read back once (`input downloaded, COREML takes host
memory`), and an output bound to a texture is uploaded from what the
session wrote. Each is a copy of the tensor, never of the frame.

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
      pad: f32,                         // what a letterbox's bars hold, 0 to 1
      scale: f32,                       // (value * scale - mean) / std
      mean: tuple<f32, f32, f32>,
      std: tuple<f32, f32, f32>,
      filter: resize-filter,            // bilinear or bicubic
      antialias: bool,                  // widen the kernel where it shrinks
    }

The answer is an ordinary `wasi:nn` tensor, fp32, which the module passes
to `compute` like any other. `id` and `index` are the tick's, as
`frames.planes` takes them. Each sample of the frame is made RGB, its YUV
with the stream's matrix and range (the chroma of a subsampled frame read
bilinearly, sited at the centre of the pixels it covers), the samples
are weighed by the resampling kernel, and the sum is clamped to 0 to 1.
The kernel is `bilinear`, or `bicubic`, which is
Keys' cubic with a = -0.5 (Pillow's BICUBIC). With `antialias` the kernel
is widened by the scale where the picture shrinks, so every source pixel
under an output pixel counts, as in Pillow's `resize` and PyTorch's
`antialias`; without it the kernel reads two samples across for
`bilinear` and four for `bicubic` at every scale. Near an edge the weights
of the samples past it are dropped and the rest scaled to sum to 1, as
Pillow does. `bilinear` without `antialias` is what `frame-tensor` did
before the two fields existed. Each channel, read as 0 to 1, then becomes
`(value * scale - mean) / std`. A letterbox fills the bars with `pad`
before the normalisation: 0 is black, 114/255 the grey YOLO models are
trained on.

A module that runs a model on parts of a frame, such as an age model on
each face a detector found, asks for the parts as one batched tensor:

    record rect { x: u32, y: u32, width: u32, height: u32 }

    crops-tensor: func(id: u32, index: u32, crops: list<rect>,
        options: frame-tensor-options) -> result<tensor, string>;

Each rectangle, in the frame's pixels, is made into one picture of the
tensor the way `frame-tensor` makes a whole frame, and the pictures are
stacked in the order given: `[N, 3, height, width]` for `nchw`,
`[N, height, width, 3]` for `nhwc`. A crop's edges are the edges the
resampling stops at, so it reads only its own pixels, as a crop cut out
and then resized does; `fit` and `pad` apply to each crop on its own. A
rectangle reaching past the frame is cut to it. An empty list and a
rectangle with nothing of it on the frame are refused. The tensor takes
the same routes to the session as `frame-tensor`'s, and the same line
names the route.

How the tensor reaches the session depends on where the frame is and on the
provider the model's session engaged. The host says which, once per model:

| frame | session | route | line |
|---|---|---|---|
| on the GPU (D3D12) | CUDA, Windows | a compute pass on the shared device writes the tensor into a buffer on a shared heap, which CUDA imported once; the session is handed that memory through an `IoBinding` | `[nn] depth: input from the pool, bound on CUDA` |
| on the GPU (D3D12) | DirectML | the same pass; the buffer is handed to the session as a DirectML allocation | `[nn] depth: input from the pool, bound on DIRECTML` |
| on the GPU | DirectML on a device of its own | the same pass, read back once | `[nn] depth: input downloaded, DIRECTML takes host memory (its session runs on a device of its own, not the pool's)` |
| on the GPU | CPU, CoreML | the same pass, read back once | `[nn] depth: input downloaded, CPU takes host memory` |
| on the GPU | CUDA where the import failed | the same pass, read back once | `[nn] depth: input downloaded, CUDA could not reach the pool's memory (<why>)` |
| in host memory | any | the same arithmetic on the CPU, from the frame's bytes | `[nn] depth: input in host memory, made on the CPU` |

On the CUDA route every CUDA session runs on a stream the host made
(`user_compute_stream`). It is a blocking stream: ONNX Runtime makes some
of a run's copies on CUDA's legacy default stream, which is ordered only
against blocking streams. The queue signals a shared D3D12 sync object after
the pass, CUDA imports that sync object as an external semaphore, and the
wait on it is queued in the session's stream ahead of the run, so the GPU
orders the pass and the run and the CPU waits for neither. Where the stream
cannot be made, CUDA is registered with its own stream and the CPU waits for
the pass before each run, and stderr says so once. On the DirectML route the
session runs on the queue every module and the host submit to, so the pass,
given to the queue first, runs first, and nothing waits. Neither the frame
nor the tensor crosses to host memory: a `-profile` run counts no `download`
bytes. On the downloaded route each frame costs one read back of the tensor,
never of the frame, and the profile's `download` counts the tensor's bytes.
`FFRWD_NN_BIND=off` takes the downloaded route where a bound one would be
taken (`input downloaded, FFRWD_NN_BIND is off`), so the routes can be
compared on one session.

### Tensors of several ticks

A tensor `frame-tensor` or `crops-tensor` answers is the module's until it
drops it or hands it to `compute`, across calls of `process`. A module that
runs a model on several frames at once, such as a video model taking eight
frames of a shot, keeps the frames' tensors as they arrive and joins them
when it runs the model:

    stack: func(tensors: list<borrow<tensor>>, dimensions: list<u32>)
        -> result<tensor, string>;

The answer holds the tensors' elements one after another, in the order
given, as one tensor of `dimensions`: eight `[1, 3, 224, 224]` frames as
`[1, 8, 3, 224, 224]`, for example. A tensor may be named more than once,
and none is consumed, so a batch of slots is a list of tensors the module
holds. A slot is rewritten by dropping its tensor and holding a new one, and
the host reuses the dropped tensor's buffer. An empty list, tensors of
different element types, and `dimensions` that do not hold as many elements
as the tensors together are refused.

| tensors | joined |
|---|---|
| every one bound in place on CUDA, or every one on DirectML | copied together on the device into a buffer of the pool, after the passes that made them, and bound the way they were; nothing is read back |
| any other mix | in host memory, each tensor's elements where they are, a tensor on the device read back |

On the downloaded route each tensor was read back when it was made, so the
join reads nothing more. Each tensor a module keeps holds its buffer: eight
frames at 224x224 hold eight buffers of 602 KB.

### DirectML's route

The pinned `ort` 2.0.0-rc.10 carries none of DirectML's own API, and its
DirectML provider takes a device id, so its session would make a D3D12
device of its own that no buffer of the pool belongs to. The host declares
the API itself (`runtime/src/nn/directml.rs`): the `OrtDmlApi` table of
ONNX Runtime 1.22.0's `dml_provider_factory.h`, reached through
`GetExecutionProviderApi("DML", ...)` on the DirectML build of the runtime.
Where some module is granted `-gpu` and the shared device is D3D12, every
DirectML session is registered with `SessionOptionsAppendExecutionProvider_DML1`
over an `IDMLDevice` that `DMLCreateDevice` (from the `DirectML.dll` the
provider loads) made on the shared `ID3D12Device`, and the shared queue.
A buffer of that device becomes a DirectML allocation through
`CreateGPUAllocationFromD3DResource`, made once per buffer and kept, and a
tensor over it with the `DML` memory info is bound like any other. Where
that registration cannot be made, DirectML is registered by device id as
before, stderr says so once, and inputs take the downloaded route.

Since the session runs on the shared device, a debug build's D3D12 debug
layer no longer leaves DirectML and the shared device on different
adapters.

### Outputs on the device

An output that is a picture (a depth map, a mask) can stay on the device:
the module binds it to a texture of the shared device and reads that
texture through `ffrwd:gpu` like any other.

    record output-texture-options {
      output: string,               // the graph's name for the output
      width: u32, height: u32,      // the texture's size
    }

    output-texture: func(context: borrow<graph-execution-context>,
        options: output-texture-options) -> result<gpu-texture, string>;

The texture is `r32float`, one fp32 element a texel, row by row, so the
output has `width * height` elements; its dynamic dimensions are taken as
the last the width, the one before it the height, and any others 1. From
the next `compute` of that context on, each writes the output into the
texture before it returns, as far as the device can tell: a pass the module
submits afterwards reads what that `compute` wrote. `compute` still lists
the output, and its `data` reads the elements back. Every other output
comes back in host memory, as before. The interface is the two calls and
the size and no more: an output's elements are fp32 whatever the model, so
the format is fixed rather than asked for, and a module that wants the map
in another format makes it in its own pass, as `modules/depth` does.

| session | route | line |
|---|---|---|
| CUDA, Windows | the session writes a buffer on a shared heap CUDA imported; its stream signals a shared sync object the queue waits on, and a pass copies the buffer into the texture; before the next run the stream waits for that pass | `[nn] depth: output predicted_depth to a texture, bound on CUDA` |
| DirectML on the shared device | the session writes a buffer of the device, as a DirectML allocation, and a pass copies it into the texture; the queue orders the two | `[nn] depth: output predicted_depth to a texture, bound on DIRECTML` |
| CPU, CoreML, DirectML on a device of its own | the output comes back to host memory and is uploaded into the texture once | `[nn] depth: output predicted_depth uploaded to a texture, CPU writes host memory` |
| CUDA where the import failed | the same | `... uploaded to a texture, CUDA could not reach the device's memory (<why>)` |

On the bound routes a `-profile` run counts no `download` and no `upload`
bytes for the output; the uploaded route counts the output's bytes as
`upload`. `FFRWD_NN_BIND=off` takes the uploaded route here too.

Linux has no bound route yet, for inputs or outputs: the pool is on Vulkan
there, and the Vulkan import is not written.

The CUDA runtime the binding uses is the one the CUDA provider loaded
(`cudart64_12.dll`, where the provider's preflight found it), loaded at run
time and never linked. The CUDA device is the one whose LUID is the shared
D3D12 adapter's, and it has to be device 0, the one the CUDA provider runs
every session on.

`modules/nn-frames` is the test module: a sink that runs a model on each
frame through `frame-tensor` and writes each output as rows, its values when
there are 16 or fewer, else their count, mean, least and greatest. With
`texture=<output>:texture_width=<w>:texture_height=<h>` it binds that output
to a texture and reads its values from there. With `stack=<n>` it keeps n
slots, frame k's tensor in slot k mod n, and runs the model on the slots
joined by `stack`, `[n, 3, h, w]`, or with `clip=true` `[1, n, 3, h, w]`,
which `model/channel-means-clip.onnx` takes. Its graphs,
`model/channel-means.onnx` (NCHW) and `channel-means-nhwc.onnx`, return
each channel's mean and largest value; `model/make-channel-means.py`
regenerates them. The same frames give the same rows on every route, to
within float rounding.

    ffrwd-wasm -nn means=modules/nn-frames/model/channel-means.onnx \
      -nn-runtime <dir> -nn-target cuda -gpu nn_frames.wasm \
      -f nut -i in.nut -m nn_frames=nn_frames.wasm \
      -filter_complex "[0:v]decode=format=yuv420p:hw=d3d12va[d];[v=d]nn_frames=model=means:memory=gpu:width=518:height=518[@rows=r]" \
      -map [r] -f ndjson rows.ndjson

On the RTX 4090, release build, h264 decoded on the device into the pool,
the tensor 518x518 letterboxed, median per frame over 230 frames, the time
inside the module for `frame-tensor` and `compute` together, and on the
bound routes the output bound to a texture and then read back from it
(`output_ms`; a module that uses the texture on the device reads nothing
back). The parts are medians of their own, so they need not sum to the
whole:

| model | input | CUDA, bound | CUDA, downloaded | DirectML, bound | DirectML, downloaded |
|---|---|---|---|---|---|
| channel-means | 1080p | 1.61 ms (1.34 + 0.34) | 6.57 ms | 0.74 ms (0.59 + 0.13) | 6.54 ms |
| channel-means | 4K | 1.80 ms (1.43 + 0.36) | 6.82 ms | 0.87 ms (0.71 + 0.14) | 6.63 ms |
| Depth Anything V2 small | 1080p | 28.6 ms (27.8 + 0.89) | 34.8 ms | 28.2 ms (18.7 + 9.45) | 61.1 ms |
| Depth Anything V2 small | 4K | 28.9 ms (28.0 + 0.87) | 36.3 ms | 28.3 ms (18.7 + 9.49) | 63.2 ms |

A DirectML run on bound memory returns before the device has finished, so
its time is in the read back; the sum is what a frame costs. Without the
texture, a CUDA session on the host's stream takes 0.80 ms for
channel-means at 1080p and 0.82 to 0.89 ms at 4K, where the CPU waiting for
the pass took 1.25 to 1.36 ms; Depth Anything at 4K takes 29.1 to 29.3 ms
against 28.1 to 28.6 ms, so for it the run, not the wait, is the cost.

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

`modules/depth` is the one that ships, in two builds from one source. Both
ask for the name **`depth`**, so both read `depth.onnx` beside them:

    -nn depth=modules/depth/model/model.onnx

The graph is Depth Anything V2 (small), fp32, 94 MB, and is not in git.
`modules/depth/fetch-model.ps1` downloads it against a pinned revision and
sha256. Unlike transcribe the module does not compile it in, so nothing is
rebuilt after a fetch.

Both run the graph at 518x518 - the size the model's own preprocessing uses -
take `pixel_values` as a planar fp32 tensor, letterboxed and normalised the
way the model was trained, and read `predicted_depth` back.

`depth.wasm` exports the stream interface, so frames reach it the way they
reach any module, and makes its tensor itself, on the CPU. It imports
nothing of the GPU's, so a sidecar built without the `gpu` feature runs it,
and nothing grants it the device:

    ffrwd-wasm -f nut -i in.nut       -nn depth=modules/depth/model/model.onnx       -nn-runtime <dir> -nn-target gpu       -m depth=depth.wasm -m blur_mask=blur_mask.wasm       -filter_complex "[0:v]depth[n1];[0:v][n1]blur_mask[out0]"       -map [out0] -f nut out.nut

`depth_gpu.wasm`, the `gpu` feature built by the `depth-gpu` crate, is a
node of the node world beside `ffrwd:gpu`. It imports `wasi:webgpu`, so it
runs only where it is granted the device (`-gpu`), even where its frames
stay in host memory; a sidecar built without the feature refuses it by name.
The host makes its tensor with `frame-tensor`. `memory` says where its
frames are held. On `host`, the default, the tensor is made on the CPU and
the map comes back as bytes; the frame leaves in the input's pixel format,
yuv420p or rgba, and no adapter is needed. On `gpu`, the input is held on
the device, the map is bound to a texture with `output-texture`, and two
passes on the device find its range over the frame and paint the frame,
which leaves as an rgba texture. The two paths paint the same frames to
within a level.

    ffrwd-wasm -f nut -i in.nut       -nn depth=modules/depth/model/model.onnx       -nn-runtime <dir> -nn-target gpu -gpu depth_gpu.wasm       -m depth=depth_gpu.wasm -m blur_mask=blur_mask.wasm       -filter_complex "[v=0:v]depth=memory=gpu[out=n1];[0:v][n1]blur_mask[out0]"       -map [out0] -f nut out.nut

On an RTX 4090, 64x64 frames, the graph itself is what the time goes on:
178 ms a frame on the CPU provider against 20 ms on CUDA, after a fixed
230 ms and 870 ms respectively to start up and build the session.

## The segment module

`modules/segment` runs YOLOv8n-seg at 640x640 and asks for the name
**`segment`**, in two builds of one source as depth does. `segment.wasm` is a
window filter that letterboxes the frame onto 114/255 grey itself, on the
CPU, and writes an index map with its rows riding it; it imports nothing of
the GPU's. `segment_gpu.wasm` is a node beside `ffrwd:gpu` whose map leaves
on `map` and rows on `objects`, a data port of their own, and is granted
`-gpu`. The host makes its tensor with `frame-tensor` (`pad` 114/255). On
`memory=gpu` the mask prototypes, the graph's rank-4 output of 32 planes,
are bound to a texture with `output-texture`, its planes stacked down it,
and a pass on the device paints the map from them and each detection's
coefficients; the predictions come back to host memory, where the
detections are decoded and thinned, so the rows are made on the CPU on
every route. Its output line names the export's own output:

    [nn] segment: output onnx::Shape_356 to a texture, bound on CUDA

Which output is the prototypes is known only by shape, so a run on
`memory=gpu` runs the graph once over a grey square at `init` to learn it.
On the 4090 at 1080p the module takes about 24 ms a frame bound on CUDA,
19 ms bound on DirectML, 44 ms on either downloaded, and 43 ms on
`memory=host` with CUDA.
