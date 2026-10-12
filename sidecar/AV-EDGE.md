# The codec edge

What the sidecar accepts on the command line for its own `decode` and
`encode` nodes, where the libavcodec they run on comes from, and what it
refuses. The compiler builds this argv; `NODE-CLI.md` has the two nodes'
spelling inside a network.

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
in that order, from one directory. `libswscale` converts the decoder's
native format to the wire's and the wire's to the encoder's; there is no
`libavformat`, since the sidecar carries NUT itself and ffmpeg keeps every
container. A run that holds neither node never loads any of it.

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

Those are FFmpeg 9.0's. The builds are the project's own, LGPL only:
`--disable-everything`, the decoders for h264, hevc, av1, vp9,
mpeg2video, prores, mjpeg, ffv1 and mpeg4, the software encoders ffv1,
mpeg4, mpeg2video and mjpeg, and the hardware codecs the platform has
(NVDEC and NVENC on Linux and Windows; D3D11VA, D3D12VA, DXVA2, Media
Foundation and AMF on Windows; VideoToolbox on macOS; Vulkan on Linux,
opened at run time, where VA-API is left out because libavutil would link
libva at load time and a wheel cannot assume it). Never `--enable-gpl`,
never `--enable-nonfree`. The musl wheels carry software codecs only. The native av1 decoder decodes only through a
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

## Refusals

Every refusal names the directory looked in, the file wanted, and the
reinstall hint:

    no avcodec-63.dll in C:\...\ffrwd_wasm_av; reinstall the sidecar with
    `pip install --force-reinstall ffrwd-wasm`

A library of another major is refused by version:

    libavcodec is version 62.3.100, and this build demands major 63

A decoder or encoder the build does not have is refused by name, listing
what the build has. An encoder option the encoder does not take is refused
naming the option and the encoder.

## Hardware

`decode` takes `hw=<device>`: `none`, `auto`, or one of `cuda`,
`d3d11va`, `d3d12va`, `dxva2`, `videotoolbox`, `vaapi`, `vulkan`. `auto`
tries the platform's order (Windows: d3d11va, cuda; Linux: vulkan, cuda;
macOS: videotoolbox), skipping a device type the build does not have, and
falls back to software. Either way the
frames come back to host memory. One line on stderr says what engaged:

    [av] decode h264: d3d11va
    [av] decode h264: software (d3d11va: no device)

An encoder is named outright (`codec=h264_nvenc`) and engages or refuses;
there is no fallback from a hardware encoder to a software one, since the
query named it.
