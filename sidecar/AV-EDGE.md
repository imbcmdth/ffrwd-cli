# The codec edge

What the sidecar accepts on the command line for its own `decode`,
`encode` and `filter` nodes, where the libav they run on comes from, and
what it refuses. The compiler builds this argv; `NODE-CLI.md` has the
nodes' spelling inside a network.

## What runs

`decode` and `encode` are the host's own nodes, like `leaky`: no `-m` binds
them, the network string names them, and the host answers for them with
libavcodec. A region holding one reads its input as coded packets and the
ffmpeg writing that pipe copies the stream (`-c:v copy`); a region holding
the other writes coded packets and the ffmpeg reading that pipe copies them
into the file. Raw frames exist only inside the region, in host memory, in
the format the module between them reads.

libavcodec is loaded at run time, never linked. The sidecar loads
`libavutil`, `libswresample`, `libavcodec` and `libswscale` by full path,
in that order, from one directory, then `libavfilter` where the directory
has it. `libswscale` converts the decoder's native format to the wire's
and the wire's to the encoder's; there is no `libavformat`, since the
sidecar carries NUT itself and ffmpeg keeps every container. A run that
holds none of the three nodes, and serves no port feed whose picture
arrives coded, never loads any of it.

`filter` runs a graph in libavfilter's own syntax, on a D3D12 device
(below). A directory without `libavfilter`, which a wheel from before it
was bundled is, runs everything else and refuses `filter` by name.

## The argv

    -av-runtime <dir>       where the libraries are, else FFRWD_AV_RUNTIME,
                            else the directory the ffrwd-wasm binary is in

The flag is read before the argv is dispatched, the way `-nn-runtime` is,
so `--describe` and `--invoke` take it too. The libraries ship inside each
platform's `ffrwd-wasm` wheel, under `ffrwd_wasm_av/` beside the package,
and the cli passes that directory as `-av-runtime` to every process that
needs it; a printed command never carries the flag, since its reader has
their own machine's copy. A checkout points at a build of its own with
`FFRWD_AV_RUNTIME`, which stands when the argv says nothing.

## The pinned build

The sidecar demands one major version of each library and refuses any
other, naming what it found and what it wanted:

| library | major | file on Windows | on Linux | on macOS |
|---|---|---|---|---|
| libavutil | 61 | `avutil-61.dll` | `libavutil.so.61` | `libavutil.61.dylib` |
| libswresample | 7 | `swresample-7.dll` | `libswresample.so.7` | `libswresample.7.dylib` |
| libavcodec | 63 | `avcodec-63.dll` | `libavcodec.so.63` | `libavcodec.63.dylib` |
| libswscale | 10 | `swscale-10.dll` | `libswscale.so.10` | `libswscale.10.dylib` |
| libavfilter | 12 | `avfilter-12.dll` | `libavfilter.so.12` | `libavfilter.12.dylib` |

Those are FFmpeg 9.0's. The builds are the project's own, LGPL only:
`--disable-everything`, the decoders for h264, hevc, av1, vp9,
mpeg2video, prores, mjpeg, ffv1 and mpeg4, the software encoders ffv1,
mpeg4, mpeg2video and mjpeg, and the hardware codecs the platform has
(NVDEC and NVENC on Linux and Windows; D3D11VA, D3D12VA, DXVA2, Media
Foundation and AMF on Windows; VideoToolbox on macOS; Vulkan on Linux,
opened at run time, where VA-API is left out because libavutil would link
libva at load time and a wheel cannot assume it). libavfilter carries
`buffer`, `buffersink`, `format`, `hwupload`, `hwdownload` and `null`
everywhere, and the hardware filters the platform's build can make:
`scale_d3d12` and `deinterlace_d3d12` on Windows; on Linux glibc the
`_cuda` filters (`scale`, `overlay`, `pad`, `transpose`, `yadif`,
`bwdif`, `chromakey`, `colorspace`), their kernels compiled by clang, and
the `_vulkan` ones (`scale`, `overlay`, `blend`, `transpose`, the flips,
`bwdif`, `gblur`, `avgblur`, `nlmeans`, `xfade`, `chromaber`), their
shaders compiled at build time; none on musl or macOS. Never
`--enable-gpl`, never `--enable-nonfree`. The musl wheels carry software codecs only. The native av1 decoder decodes only through a
hardware device; there is no software AV1 in this build. The LGPL text
sits beside the libraries in the wheel, with the MIT notices of the
NVIDIA and AMF header sets on the platforms that compile them in.
`sidecar/ffrwd-wasm/av-pins.json`
pins each platform's archive by URL, sha256 and byte count, and the
release workflow fetches it before the wheel is built.

## `--av-info`

    ffrwd-wasm --av-info

prints one line of JSON: the majors this build demands, the platform, the
directory it resolved, the version it loaded, and the hardware device
types, decoders and encoders that libavcodec has. It answers on a machine
with no libraries at all, carrying the refusal instead of a version, which
is how a caller learns what to reinstall without running anything else.
libavfilter is answered for on its own: its version in `avfilter`, every
filter it has in `filters`, and in `avfilter_error` why it is not loaded.
The compiler reads `filters` to decide which of a query's filters run on
the device (`docs/dialect.md`).

## Refusals

Every refusal names the directory looked in, the file wanted, and the
reinstall hint:

    no avcodec-63.dll in C:\...\ffrwd_wasm_av; reinstall the sidecar with
    `pip install --force-reinstall ffrwd-wasm`

A library of another major is refused by version:

    libavcodec is version 62.3.100, and this build demands major 63

A directory without libavfilter refuses `filter` the same way, and
nothing else:

    filter stream 0 of input 0: no avfilter-12.dll in C:\...\ffrwd_wasm_av
    (-av-runtime); reinstall the sidecar, ...

A graph libavfilter will not parse or configure is refused naming the
graph and the call, after libav's own line naming the filter or option:

    [Parsed_scale_d3d12_0 @ ...] Could not set non-existent option 'wx' to value '4'
    filter stream 0 of input 0: the filter graph 'scale_d3d12=wx=4':
    avfilter_graph_parse_ptr: Option not found; `ffrwd-wasm --av-info` lists
    the filters this build has; ...

A decoder or encoder the build does not have is refused by name, pointing
at `--av-info` for what the build has. An encoder option the encoder does
not take is refused naming the option and the encoder. `encode` codes in
the format `pix_fmt` names, else the encoder's first; an rgba picture
into `ffv1` keeps every bit only with `pix_fmt=bgra`.

## Hardware

`decode` takes `hw=<device>`: `none`, `auto`, or one of `cuda`,
`d3d11va`, `d3d12va`, `dxva2`, `videotoolbox`, `vaapi`, `vulkan`. `auto`
tries the platform's order (Windows: d3d11va, cuda; Linux: vulkan, cuda;
macOS: videotoolbox), skipping a device type the build does not have, and
falls back to software. The frames come back to host memory, with one
exception: when the node reading them holds its frames on the GPU
(`NODE-SHAPE.md`) and the device is the sidecar's own, `d3d12va` on
Windows, the decoder writes straight into the sidecar's frame pool and
nothing comes back. A coded port feed into an input held on the GPU is
decoded the same way, where the device allows. One line on stderr says
what engaged, and a second says whether the decoder's own images were
taken as they are or copied once on the device:

    [av] decode h264: d3d11va
    [av] decode h264: software (d3d11va: no device)
    [av] decode h264: d3d12va, on the host's device
    [gpu] decode h264: d3d12va, frames wrapped

That holds for `format=yuv420p` and `format=rgba`. The decoder writes
NV12 either way; a reader of yuv420p is handed its planes split on the
device, byte for byte the software decode, and a reader of rgba is handed
it converted on the device once per picture, in the stream's matrix and
range (BT.709, or BT.601 where the stream says so or says nothing, as
libswscale reads it), its chroma interpolated between samples. That is
within one level of ffmpeg's `scale=flags=bilinear+full_chroma_int+accurate_rnd`;
the software path's default conversion repeats each chroma sample across
two pixels instead, so the two differ at sharp colour edges. A stream in
another matrix, or another `format`, is converted in host memory and put
on the GPU by its reader, as before. The conversion says so once:

    [gpu] convert nv12 1920x1080 to rgba 1920x1080

An encoder is named outright (`codec=h264_nvenc`) and engages or refuses;
there is no fallback from a hardware encoder to a software one, since the
query named it. `h264_d3d12va` and `hevc_d3d12va` take their frames from
the pool with one copy on the device and no readback; an rgba frame a
module rendered is made NV12 on the device first (BT.709 limited range
unless the stream says BT.601 or full range), and the header says what
was written. These two encoders in FFmpeg 9.0.1 fail on a B-frame on
NVIDIA's driver and write no extradata, so `bf=0` is set unless asked
otherwise, and the parameter sets for the header are taken from a first
packet coded ahead of the stream. They keep 16 pictures in flight unless
`async_depth` is given, since each picture reaches them through a copy on
another queue. Every other encoder reads a GPU frame back once.

    [av] encode h264_d3d12va: 1920x1080 nv12, from the device
    [gpu] encode h264_d3d12va: from the device, each frame one copy

`filter` runs its graph on a `d3d12va` device: the D3D12 filters read and
write D3D12 textures, so the graph's source is a frames context of NV12
textures of the stream's size. Where something on the sidecar's device
reads the node's pictures or writes the pictures it reads (a node module
holding a port on the GPU, a `decode` there, an `encode` reading D3D12
pictures) the graph runs on that device: each frame is copied into a
picture of the source on the device (a module's rgba made NV12 first, as
for the encoder), and the sink's pictures become the node's frames as
they are, wrapped as the decoder's are, with no copy. A frame in host
memory is converted to NV12 by libswscale and uploaded. With no such
reader the graph runs on a device of its own and its pictures are read
back to host memory. The output's size comes from the configured graph's
sink. One line says where the graph runs, one how frames reach it:

    [av] filter scale_d3d12: on the host's device
    [av] filter scale_d3d12: frames copied on the device
    [gpu] filter scale_d3d12: d3d12va, frames wrapped

A D3D12 filter signals one sync object of its own for every picture it
writes and writes a texture of its pool again without waiting on anyone,
so a picture taken from the graph is kept until the device has finished
reading it rather than handed back by a signal, and a frame is copied
into the source only once the filter has finished with what it read
before. `deinterlace_d3d12` does not run on NVIDIA's driver: creating its
video processor removes the device (0x887A0005), ffmpeg's own command
line included.
