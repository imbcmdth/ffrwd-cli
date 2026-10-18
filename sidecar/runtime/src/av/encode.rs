//! Encoding the wire's pictures into coded video.

use std::os::raw::c_int;
use std::time::Instant;

use super::decode::{DeviceFrame, Shared};
use super::error::{Error, Result};
use super::ffi::{AvErr, BufferRef, CodecCtx, Frame, Lib, Packet, RawColor, Scaler, Status};
use super::fourcc;
use super::picture::{self, WIRE_PIX_FMTS};
use super::sys::{AVRational, AV_CODEC_FLAG_GLOBAL_HEADER};
use super::Color;

/// One coded packet, in decode order, timestamps in the encoder's time
/// base.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EncodedPacket {
    pub pts: i64,
    pub dts: Option<i64>,
    pub duration: Option<i64>,
    pub keyframe: bool,
    pub data: Vec<u8>,
}

/// An encoder for one video stream.
///
/// Pictures go in with [`send`](Encoder::send); after each, call
/// [`receive`](Encoder::receive) until it answers None. At the end of the
/// stream, [`flush`](Encoder::flush), then `receive` until None again.
pub struct Encoder {
    lib: &'static Lib,
    name: String,
    codec: String,
    ctx: CodecCtx,
    input: &'static str,
    input_fmt: c_int,
    input_color: RawColor,
    pix_fmt: c_int,
    pix_fmt_name: String,
    color: RawColor,
    width: u32,
    height: u32,
    frame: Frame,
    scaler: Option<Scaler>,
    packet: Packet,
    extradata: Vec<u8>,
    delay: u32,
    codec_tag: u32,
    /// Set for an encoder that reads D3D12 pictures from a pool of its own:
    /// the picture of the pool a host picture is uploaded into.
    pool: Option<Frame>,
    /// What the encoder codes in place of the `pix_fmt` asked, where it
    /// does: an encoder on the device asked for yuv420p codes nv12.
    substituted: Option<&'static str>,
    /// Set for an encoder that reads CUDA pictures and is fed from a Vulkan
    /// device: the pool of Vulkan pictures a frame on that device is written
    /// into, each copied into one of the encoder's on the card.
    staging: Option<BufferRef>,
}

/// The device type the encoders that read D3D12 pictures run on.
const D3D12VA: &str = "d3d12va";

/// The device type the VideoToolbox encoders read pictures of.
const VIDEOTOOLBOX: &str = "videotoolbox";

/// The device type NVENC reads pictures of.
const CUDA: &str = "cuda";

/// The pictures an encoder that reads D3D12 pictures has in flight before
/// it waits for one. Each picture reaches it through a copy on another
/// queue; at libavcodec's default of 2 the wait for that copy held a
/// decode-to-encode chain to two thirds of ffmpeg's rate.
const ASYNC_DEPTH: &str = "16";

fn call<'a>(call: &'static str, codec: &'a str) -> impl FnOnce(AvErr) -> Error + 'a {
    move |e| Error::Call {
        call,
        codec: codec.to_string(),
        why: e.why,
    }
}

/// A fraction as libav's, refusing one that is not positive or does not
/// fit.
fn rational(encoder: &str, key: &str, (num, den): (i64, i64)) -> Result<AVRational> {
    let refuse = |why: &str| Error::Option {
        encoder: encoder.to_string(),
        key: key.to_string(),
        value: format!("{num}/{den}"),
        why: why.to_string(),
    };
    if num <= 0 || den <= 0 {
        return Err(refuse("it is not a positive fraction"));
    }
    match (c_int::try_from(num), c_int::try_from(den)) {
        (Ok(num), Ok(den)) => Ok(AVRational { num, den }),
        _ => Err(refuse("it does not fit libav's 32-bit fractions")),
    }
}

/// A codec's parameter sets: how a NAL unit's type is read from its first
/// byte, and the types that are parameter sets.
#[derive(Clone, Copy)]
struct ParameterSets {
    kind: fn(u8) -> u8,
    kinds: &'static [u8],
}

/// h264's SPS and PPS, hevc's VPS, SPS and PPS; None for a codec whose
/// header is not made of NAL units.
fn parameter_set_types(codec: &str) -> Option<ParameterSets> {
    match codec {
        "h264" => Some(ParameterSets {
            kind: |b| b & 0x1f,
            kinds: &[7, 8],
        }),
        "hevc" => Some(ParameterSets {
            kind: |b| (b >> 1) & 0x3f,
            kinds: &[32, 33, 34],
        }),
        _ => None,
    }
}

/// The NAL units of an Annex B stream that are parameter sets, each behind
/// a four-byte start code and without the zero bytes that may follow it.
fn annex_b_units(data: &[u8], sets: ParameterSets) -> Vec<u8> {
    let mut starts = Vec::new();
    let mut i = 0;
    while i + 3 <= data.len() {
        if data[i..i + 3] == [0, 0, 1] {
            starts.push(i + 3);
            i += 3;
        } else {
            i += 1;
        }
    }
    let mut out = Vec::new();
    for (n, &start) in starts.iter().enumerate() {
        let end = starts.get(n + 1).map_or(data.len(), |next| next - 3);
        let mut unit = &data[start..end.max(start)];
        while let [rest @ .., 0] = unit {
            unit = rest;
        }
        if unit
            .first()
            .is_some_and(|&b| sets.kinds.contains(&(sets.kind)(b)))
        {
            out.extend_from_slice(&[0, 0, 0, 1]);
            out.extend_from_slice(unit);
        }
    }
    out
}

/// What `avcodec_send_frame` came back with, as a result.
fn sent(status: Status, name: &str) -> Result<()> {
    match status {
        Status::Done => Ok(()),
        Status::Again => Err(Error::Call {
            call: "avcodec_send_frame",
            codec: name.to_string(),
            why: "the encoder has packets waiting; receive until it answers None, then send"
                .to_string(),
        }),
        Status::Eof => Err(Error::Call {
            call: "avcodec_send_frame",
            codec: name.to_string(),
            why: "the encoder was already flushed".to_string(),
        }),
        Status::Failed(e) => Err(call("avcodec_send_frame", name)(e)),
    }
}

/// What an encoder reading NV12 pictures of a pool on a device codes for
/// the `pix_fmt` asked: nv12 as asked, or for yuv420p, the same 4:2:0
/// sampling at 8 bits and the same coded stream, nv12 with the note saying
/// so. Anything else is refused, naming what it takes.
fn pool_pix_fmt(asked: &str) -> std::result::Result<Option<&'static str>, String> {
    match asked {
        "nv12" => Ok(None),
        "yuv420p" => Ok(Some("yuv420p asked, nv12 on the device")),
        _ => Err(
            "it reads NV12 pictures on the device, so it takes yuv420p (coded as nv12) or nv12"
                .to_string(),
        ),
    }
}

/// Whether a colour name leaves the value unspecified.
pub(super) fn unset(name: &str) -> bool {
    matches!(name, "" | "unknown" | "unspecified")
}

impl Encoder {
    /// An encoder called `codec` (`libx264`, `ffv1`, `prores_ks`), or
    /// libavcodec's preferred encoder for the codec called that (`h264`).
    ///
    /// Pictures arrive `width`x`height` in the wire format `in_pix_fmt`,
    /// timestamped in `time_base`. The encoder codes them in the `pix_fmt`
    /// option's format if given, else the first it lists, converting with
    /// libswscale where that is not `in_pix_fmt`. `color` tags the stream;
    /// an rgba input is tagged rgb and converted with its matrix.
    ///
    /// `options` are set by the names ffmpeg's command line gives them
    /// (`crf`, `preset`, `g`, `b`, `maxrate`, `bufsize`, `keyint_min`,
    /// `profile`, `level`, `tune`, `sc_threshold`), the encoder's own first;
    /// one it refuses is an error naming it. The global header flag is set,
    /// so the codec's header comes out as [`extradata`](Encoder::extradata)
    /// for a container that carries it out of band.
    #[allow(clippy::too_many_arguments)]
    pub fn open(
        codec: &str,
        width: u32,
        height: u32,
        in_pix_fmt: &str,
        time_base: (i64, i64),
        frame_rate: Option<(i64, i64)>,
        color: Option<&Color>,
        options: &[(String, String)],
    ) -> Result<Encoder> {
        Encoder::open_with(
            codec, width, height, in_pix_fmt, time_base, frame_rate, color, options, None,
        )
    }

    /// An encoder that reads D3D12 pictures (`h264_d3d12va`,
    /// `hevc_d3d12va`, `av1_d3d12va`), opened as [`open`](Encoder::open)
    /// opens one, with a pool of NV12 textures of the stream's size on
    /// `device`, or on a `d3d12va` device of its own when None. A
    /// VideoToolbox encoder given [`Shared::VideoToolbox`] is opened the
    /// same way on a pool of IOSurface-backed NV12 pictures, which any
    /// Metal device reads and writes. An NVENC encoder given
    /// [`Shared::Vulkan`] is opened on a pool of CUDA pictures on the same
    /// card, and is fed through a pool of Vulkan pictures on that device:
    /// a frame is written into one of those on the device, and libavutil
    /// copies it into a CUDA picture on the card, the two APIs sharing the
    /// image's memory and its timeline semaphore.
    ///
    /// A picture is written into one of the pool's textures on the device
    /// ([`device_picture`](Encoder::device_picture), then
    /// [`send_picture`](Encoder::send_picture)), or handed over in host
    /// memory with [`send`](Encoder::send), which converts it to NV12 with
    /// `color`'s matrix and uploads it, since the encoder reads nothing
    /// else.
    ///
    /// B-frames are off unless `bf` is among `options`: this libav's D3D12
    /// encoders fail on the first B-frame on the NVIDIA driver they were
    /// tried on, ffmpeg's own command line included.
    ///
    /// The encoder keeps [`ASYNC_DEPTH`] pictures in flight unless
    /// `async_depth` is among `options`.
    ///
    /// These encoders write the codec's header (h264's and hevc's
    /// parameter sets) into each keyframe and leave the out-of-band one
    /// empty, so [`extradata`](Encoder::extradata) is read from the first
    /// packet of a second encoder opened the same way, which codes one
    /// picture and is closed.
    #[allow(clippy::too_many_arguments)]
    pub fn open_on_device(
        codec: &str,
        width: u32,
        height: u32,
        in_pix_fmt: &str,
        time_base: (i64, i64),
        frame_rate: Option<(i64, i64)>,
        color: Option<&Color>,
        options: &[(String, String)],
        device: Option<Shared>,
    ) -> Result<Encoder> {
        let again = device.as_ref().map(Shared::share);
        let open = |device| {
            Encoder::open_with(
                codec,
                width,
                height,
                in_pix_fmt,
                time_base,
                frame_rate,
                color,
                options,
                Some(device),
            )
        };
        let mut encoder = open(device)?;
        if encoder.extradata.is_empty() && parameter_set_types(&encoder.codec).is_some() {
            let mut probe = open(again)?;
            encoder.extradata = probe.parameter_sets()?;
        }
        Ok(encoder)
    }

    /// The parameter sets the first packet carries, for an encoder on the
    /// device that wrote no out-of-band header: one picture of its pool,
    /// unwritten, coded, and the encoder flushed.
    fn parameter_sets(&mut self) -> Result<Vec<u8>> {
        let Some(kinds) = parameter_set_types(&self.codec) else {
            return Ok(Vec::new());
        };
        let picture = self.device_picture()?;
        self.send_picture(picture, 0, None)?;
        self.flush()?;
        while let Some(packet) = self.receive()? {
            let sets = annex_b_units(&packet.data, kinds);
            if !sets.is_empty() {
                return Ok(sets);
            }
        }
        Ok(Vec::new())
    }

    /// Whether `codec` names an encoder that reads D3D12 pictures and
    /// nothing else, which [`open_on_device`](Encoder::open_on_device)
    /// opens.
    pub fn reads_d3d12(codec: &str) -> bool {
        codec.ends_with("_d3d12va")
    }

    /// Whether `codec` names an NVENC encoder, which reads pictures in host
    /// memory or a pool of CUDA pictures that
    /// [`open_on_device`](Encoder::open_on_device) gives it, fed from a
    /// Vulkan device.
    pub fn reads_cuda(codec: &str) -> bool {
        codec.ends_with("_nvenc")
    }

    /// Whether `codec` names a VideoToolbox encoder, which reads pictures
    /// in host memory or a pool of IOSurface-backed pictures that
    /// [`open_on_device`](Encoder::open_on_device) gives it.
    pub fn reads_videotoolbox(codec: &str) -> bool {
        codec.ends_with("_videotoolbox")
    }

    #[allow(clippy::too_many_arguments)]
    fn open_with(
        codec: &str,
        width: u32,
        height: u32,
        in_pix_fmt: &str,
        time_base: (i64, i64),
        frame_rate: Option<(i64, i64)>,
        color: Option<&Color>,
        options: &[(String, String)],
        device: Option<Option<Shared>>,
    ) -> Result<Encoder> {
        let lib = super::runtime()?;
        let Some(found) = lib.find_encoder(codec) else {
            return Err(Error::Codec {
                role: "encoder",
                name: codec.to_string(),
            });
        };
        let name = found.name();
        let Some(input) = picture::wire_pix_fmt(in_pix_fmt) else {
            return Err(Error::PixelFormat {
                name: in_pix_fmt.to_string(),
                why: format!("an encoder takes one of {}", WIRE_PIX_FMTS.join(", ")),
            });
        };
        let input_fmt = lib
            .wire_pix_fmt(input)
            .expect("every wire format was found at load");
        let dims = (c_int::try_from(width), c_int::try_from(height));
        let (Ok(w), Ok(h)) = dims else {
            return Err(Error::Option {
                encoder: name,
                key: "s".to_string(),
                value: format!("{width}x{height}"),
                why: "the size does not fit".to_string(),
            });
        };
        if width == 0 || height == 0 {
            return Err(Error::Option {
                encoder: name,
                key: "s".to_string(),
                value: format!("{width}x{height}"),
                why: "a picture has no area".to_string(),
            });
        }
        let time_base = rational(&name, "time_base", time_base)?;
        let frame_rate = frame_rate
            .map(|rate| rational(&name, "r", rate))
            .transpose()?;

        let mut ctx = CodecCtx::new(found).map_err(call("avcodec_alloc_context3", &name))?;

        let supported = ctx.supported_pix_fmts();
        // An encoder on the device lists its device's format (`d3d12`,
        // `videotoolbox_vld`); what it codes is its pool's format, NV12.
        let on_device = match &device {
            None => None,
            Some(shared) => {
                let refuse = |why: String| Error::Call {
                    call: "av_hwframe_ctx_init",
                    codec: name.clone(),
                    why,
                };
                let (format, device_name, api) = match shared {
                    Some(Shared::VideoToolbox) => {
                        ("videotoolbox_vld", VIDEOTOOLBOX, "VideoToolbox")
                    }
                    Some(Shared::Vulkan(_)) => ("cuda", CUDA, "CUDA"),
                    _ => ("d3d12", D3D12VA, "D3D12"),
                };
                let reads = lib.pix_fmt(format).is_some_and(|f| supported.contains(&f));
                if !reads {
                    return Err(refuse(format!("it does not read {api} pictures")));
                }
                let Some(device_type) = lib.hw_type(device_name) else {
                    return Err(refuse(format!(
                        "this libav cannot create a {device_name} device"
                    )));
                };
                if !width.is_multiple_of(2) || !height.is_multiple_of(2) {
                    return Err(refuse(format!(
                        "an NV12 texture has an even size, and the stream is {width}x{height}"
                    )));
                }
                let nv12 = lib
                    .pix_fmt("nv12")
                    .ok_or_else(|| refuse("libavutil has no nv12 pixel format".to_string()))?;
                Some((device_type, nv12))
            }
        };
        let supported = match on_device {
            Some((_, nv12)) => vec![nv12],
            None => supported,
        };
        let asked = options
            .iter()
            .rev()
            .find(|(k, _)| k == "pix_fmt")
            .map(|(_, v)| v.as_str());
        let mut substituted = None;
        let pix_fmt = match asked {
            Some(asked) if on_device.is_some() => {
                let (_, nv12) = on_device.expect("matched as some");
                match pool_pix_fmt(asked) {
                    Ok(note) => {
                        substituted = note;
                        nv12
                    }
                    Err(why) => {
                        return Err(Error::Option {
                            encoder: name,
                            key: "pix_fmt".to_string(),
                            value: asked.to_string(),
                            why,
                        })
                    }
                }
            }
            Some(asked) => {
                let refuse = |why: String| Error::Option {
                    encoder: name.clone(),
                    key: "pix_fmt".to_string(),
                    value: asked.to_string(),
                    why,
                };
                let Some(fmt) = lib.pix_fmt(asked) else {
                    return Err(refuse("libavutil has no such pixel format".to_string()));
                };
                if !supported.is_empty() && !supported.contains(&fmt) {
                    let takes: Vec<String> =
                        supported.iter().map(|&f| lib.pix_fmt_name(f)).collect();
                    return Err(refuse(format!("it takes {}", takes.join(", "))));
                }
                fmt
            }
            // The first the encoder lists that pictures in host memory are
            // made in: a VideoToolbox encoder lists its device's first.
            None => supported
                .iter()
                .copied()
                .find(|&f| !lib.pix_fmt_is_hw(f))
                .unwrap_or(input_fmt),
        };

        let color = match color {
            None => RawColor::UNSPECIFIED,
            Some(c) => {
                let names = [&c.range, &c.primaries, &c.trc, &c.space].map(|n| {
                    if unset(n) {
                        ""
                    } else {
                        n.as_str()
                    }
                });
                lib.color_values(names)
                    .map_err(|(key, value)| Error::Option {
                        encoder: name.clone(),
                        key: key.to_string(),
                        value,
                        why: "libavutil has no colour value of that name".to_string(),
                    })?
            }
        };
        // The picture as it arrives, and as the encoder codes it: rgb has no
        // matrix and is full range, so the conversion applies the stream's.
        let input_color = if picture::is_rgb(input) {
            color.as_rgb()
        } else {
            color
        };
        let coded_color = if lib.pix_fmt_is_rgb(pix_fmt) {
            color.as_rgb()
        } else {
            color
        };

        ctx.set_video(w, h, pix_fmt, time_base, frame_rate);
        ctx.set_color(coded_color);
        ctx.add_flags(AV_CODEC_FLAG_GLOBAL_HEADER);
        // ffmpeg's command line encodes with as many threads as it has cores
        // unless told otherwise; an explicit `threads` below replaces this.
        ctx.set_option("threads", "auto")
            .map_err(call("av_opt_set threads", &name))?;
        let mut staging = None;
        if let (Some((_, nv12)), Some(Some(Shared::Vulkan(vulkan)))) = (on_device, &device) {
            let refuse = |why: String| Error::Call {
                call: "av_hwframe_ctx_init",
                codec: name.clone(),
                why,
            };
            let cuda = vulkan.cuda().map_err(refuse)?;
            let format = lib
                .pix_fmt("cuda")
                .expect("checked above that the encoder reads it");
            let frames = lib
                .frames_on_device(cuda, format, nv12, w, h)
                .map_err(call("av_hwframe_ctx_init", &name))?;
            ctx.attach_frames(frames, format, nv12);
            staging = Some(
                lib.vulkan_frames(vulkan.device(), nv12, w, h)
                    .map_err(|e| {
                        refuse(format!("the Vulkan pictures it is fed from: {}", e.why))
                    })?,
            );
        } else if let (Some((device_type, nv12)), Some(Some(Shared::VideoToolbox))) =
            (on_device, &device)
        {
            let format = lib
                .pix_fmt("videotoolbox_vld")
                .expect("checked above that the encoder reads it");
            ctx.attach_device_frames(device_type, format, nv12, w, h)
                .map_err(call("av_hwframe_ctx_init", &name))?;
        } else if let (Some((device_type, nv12)), Some(device)) = (on_device, device) {
            let raw = match device {
                Some(Shared::D3d12(raw)) => raw.into_raw(),
                Some(Shared::VideoToolbox | Shared::Vulkan(_)) => {
                    unreachable!("taken by the branches above")
                }
                None => std::ptr::null_mut(),
            };
            ctx.attach_d3d12_frames(device_type, raw, nv12, w, h)
                .map_err(call("av_hwframe_ctx_init", &name))?;
            if !options.iter().any(|(k, _)| k == "bf") {
                ctx.set_option("bf", "0")
                    .map_err(call("av_opt_set bf", &name))?;
            }
            if !options.iter().any(|(k, _)| k == "async_depth") {
                ctx.set_option("async_depth", ASYNC_DEPTH)
                    .map_err(call("av_opt_set async_depth", &name))?;
            }
        }
        for (key, value) in options {
            if key == "pix_fmt" {
                continue;
            }
            if let Err(e) = ctx.set_option(key, value) {
                let why = if e.is_option_not_found() {
                    "it has no such option".to_string()
                } else {
                    e.why
                };
                return Err(Error::Option {
                    encoder: name,
                    key: key.clone(),
                    value: value.clone(),
                    why,
                });
            }
        }
        ctx.open().map_err(call("avcodec_open2", &name))?;

        let params = ctx
            .parameters()
            .map_err(call("avcodec_parameters_from_context", &name))?;
        let delay = if params.video_delay > 0 {
            params.video_delay
        } else {
            ctx.max_b_frames().max(0)
        } as u32;
        let codec_name = found.codec_name();

        Ok(Encoder {
            lib,
            codec: codec_name,
            ctx,
            input,
            input_fmt,
            input_color,
            pix_fmt,
            pix_fmt_name: lib.pix_fmt_name(pix_fmt),
            color: coded_color,
            width,
            height,
            frame: Frame::new(lib).map_err(call("av_frame_alloc", &name))?,
            scaler: None,
            packet: Packet::new(lib).map_err(call("av_packet_alloc", &name))?,
            extradata: params.extradata,
            delay,
            codec_tag: params.codec_tag,
            pool: match on_device {
                Some(_) => Some(Frame::new(lib).map_err(call("av_frame_alloc", &name))?),
                None => None,
            },
            staging,
            substituted,
            name,
        })
    }

    /// Whether the encoder reads pictures from a pool on a device: D3D12
    /// textures, or IOSurface-backed VideoToolbox pictures.
    pub fn on_device(&self) -> bool {
        self.pool.is_some()
    }

    /// A picture from the encoder's pool, for the caller to write on the
    /// device: NV12 of the stream's size. A D3D12 one is a texture whose
    /// objects [`DeviceFrame::d3d12`] names; the caller records the write,
    /// has the sync object signalled once it has run, and says the value
    /// with [`DeviceFrame::written_at`]. A VideoToolbox one is a
    /// `CVPixelBuffer` [`DeviceFrame::cv_pixel_buffer`] names, which the
    /// caller writes and waits for. Either is then handed to
    /// [`send_picture`](Encoder::send_picture).
    pub fn device_picture(&mut self) -> Result<DeviceFrame> {
        let mut frame = Frame::new(self.lib).map_err(call("av_frame_alloc", &self.name))?;
        match &self.staging {
            Some(staging) => staging.get_buffer(&mut frame),
            None => self.ctx.pool_frame(&mut frame),
        }
        .map_err(call("av_hwframe_get_buffer", &self.name))?;
        Ok(DeviceFrame {
            pts: 0,
            duration: None,
            width: self.width,
            height: self.height,
            pix_fmt: "nv12",
            frame,
        })
    }

    /// Hands the encoder a picture of its pool, written on the device,
    /// timestamped in its time base.
    pub fn send_picture(
        &mut self,
        mut picture: DeviceFrame,
        pts: i64,
        duration: Option<i64>,
    ) -> Result<()> {
        if picture.frame.d3d12().is_none()
            && picture.frame.videotoolbox().is_none()
            && picture.frame.vulkan().is_none()
        {
            return Err(Error::Call {
                call: "avcodec_send_frame",
                codec: self.name.clone(),
                why: "the picture is not one of the encoder's pool".to_string(),
            });
        }
        if picture.frame.vulkan().is_some() {
            // A Vulkan picture is copied into one of the encoder's CUDA
            // pictures on the card, after what was written into it.
            let Some(cuda) = self.pool.as_mut() else {
                return Err(Error::Call {
                    call: "avcodec_send_frame",
                    codec: self.name.clone(),
                    why: "the encoder has no pool of CUDA pictures".to_string(),
                });
            };
            let copied = self
                .ctx
                .pool_frame(cuda)
                .map_err(call("av_hwframe_get_buffer", &self.name))
                .and_then(|()| {
                    cuda.transfer_from(&picture.frame)
                        .map_err(call("av_hwframe_transfer_data", &self.name))
                });
            picture.frame.unref();
            if let Err(e) = copied {
                cuda.unref();
                return Err(e);
            }
            cuda.set_color(self.color);
            cuda.set_timing(pts, duration);
            let status = self.ctx.send_frame(Some(cuda));
            cuda.unref();
            return sent(status, &self.name);
        }
        picture.frame.set_color(self.color);
        picture.frame.set_timing(pts, duration);
        let status = self.ctx.send_frame(Some(&picture.frame));
        picture.frame.unref();
        sent(status, &self.name)
    }

    /// The encoder's own name: `libx264`, `ffv1`.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The codec it codes, by ffmpeg's name for it: `h264` for `libx264`.
    pub fn codec(&self) -> &str {
        &self.codec
    }

    /// The codec's out-of-band header, for a container that carries one
    /// (NUT does, in its stream header). Empty where the codec has none.
    pub fn extradata(&self) -> &[u8] {
        &self.extradata
    }

    /// How many packets the decoder will hold back before its first
    /// picture: the reorder depth B-frames give, zero without them.
    pub fn delay(&self) -> u32 {
        self.delay
    }

    /// The pixel format the encoder codes in.
    pub fn pix_fmt(&self) -> &str {
        &self.pix_fmt_name
    }

    /// What the encoder says of a `pix_fmt` it codes as another, where it
    /// was given one: `yuv420p asked, nv12 on the device`.
    pub fn pix_fmt_note(&self) -> Option<&'static str> {
        self.substituted
    }

    /// The tag to carry the stream under on the NUT wire: the one the
    /// encoder chose for itself where it chose one (prores_ks picks one per
    /// profile), else the one ffmpeg's NUT muxer writes for the codec.
    pub fn fourcc(&self) -> Option<[u8; 4]> {
        let own = self.codec_tag.to_le_bytes();
        if self.codec_tag != 0 && own.iter().all(|b| b.is_ascii_graphic() || *b == b' ') {
            return Some(own);
        }
        fourcc::fourcc_for_codec(&self.codec)
    }

    /// Hands the encoder one picture, packed as the wire lays out the
    /// format it was opened with, timestamped in its time base.
    pub fn send(&mut self, pts: i64, duration: Option<i64>, data: &[u8]) -> Result<()> {
        let expected = picture::frame_len(self.input, self.width, self.height)
            .expect("the input is a wire format");
        if data.len() != expected {
            return Err(Error::FrameSize {
                pix_fmt: self.input.to_string(),
                width: self.width,
                height: self.height,
                expected,
                got: data.len(),
            });
        }

        let Encoder {
            lib,
            name,
            ctx,
            input,
            input_fmt,
            input_color,
            pix_fmt,
            color,
            width,
            height,
            frame,
            scaler,
            pool,
            ..
        } = self;

        // A fresh buffer for every picture, in the encoder's own format: the
        // encoder may still hold the last one. The picture crosses into it
        // in one pass: copied in row by row where the input is already that
        // format, else converted by libswscale straight from the input's
        // planes.
        frame.set_video(*width, *height, *pix_fmt);
        frame.set_color(*color);
        frame.alloc().map_err(call("av_frame_get_buffer", name))?;
        let written = if *pix_fmt == *input_fmt {
            frame.fill(input, data)
        } else {
            if scaler.is_none() {
                *scaler = Some(Scaler::new(lib).map_err(call("sws_alloc_context", name))?);
            }
            let sws = scaler.as_mut().expect("set just above");
            sws.scale_from_packed(frame, input, *width, *height, *input_color, data)
                .map_err(call("sws_scale_frame", name))?
        };
        if !written {
            frame.unref();
            return Err(Error::Call {
                call: "av_frame_get_buffer",
                codec: name.clone(),
                why: format!("the picture is not laid out as {input}"),
            });
        }
        frame.set_timing(pts, duration);

        // An encoder on the device reads its pool only: the picture goes up
        // into one of its textures, with its timestamps and colour.
        let status = match pool {
            None => ctx.send_frame(Some(frame)),
            Some(device) => {
                let started = Instant::now();
                let uploaded = ctx
                    .pool_frame(device)
                    .map_err(call("av_hwframe_get_buffer", name))
                    .and_then(|()| {
                        device
                            .upload(frame)
                            .map_err(call("av_hwframe_transfer_data", name))
                    });
                if let Err(e) = uploaded {
                    frame.unref();
                    device.unref();
                    return Err(e);
                }
                // What crossed is the NV12 picture.
                let nv12 = picture::frame_len("yuv420p", *width, *height).unwrap_or(expected);
                crate::profile::PROFILE.upload.add(nv12, started);
                device.set_color(*color);
                device.set_timing(pts, duration);
                let status = ctx.send_frame(Some(device));
                device.unref();
                status
            }
        };
        frame.unref();
        sent(status, name)
    }

    /// Ends the stream. The packets the encoder still holds come out of
    /// `receive` after this.
    pub fn flush(&mut self) -> Result<()> {
        match self.ctx.send_frame(None) {
            Status::Done | Status::Eof => Ok(()),
            Status::Again => Err(Error::Call {
                call: "avcodec_send_frame",
                codec: self.name.clone(),
                why: "the encoder has packets waiting; receive until it answers None, \
                      then flush"
                    .to_string(),
            }),
            Status::Failed(e) => Err(call("avcodec_send_frame", &self.name)(e)),
        }
    }

    /// The next coded packet, or None when the encoder wants another
    /// picture (or, after `flush`, has none left).
    pub fn receive(&mut self) -> Result<Option<EncodedPacket>> {
        match self.ctx.receive_packet(&mut self.packet) {
            Status::Done => {}
            Status::Again | Status::Eof => return Ok(None),
            Status::Failed(e) => return Err(call("avcodec_receive_packet", &self.name)(e)),
        }
        let packet = EncodedPacket {
            pts: self.packet.pts().unwrap_or(0),
            dts: self.packet.dts(),
            duration: self.packet.duration(),
            keyframe: self.packet.is_key(),
            data: self.packet.data(),
        };
        self.packet.unref();
        Ok(Some(packet))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_encoder_on_the_device_codes_yuv420p_as_nv12_and_refuses_the_rest() {
        assert_eq!(pool_pix_fmt("nv12"), Ok(None));
        assert_eq!(
            pool_pix_fmt("yuv420p"),
            Ok(Some("yuv420p asked, nv12 on the device"))
        );
        for other in ["yuv444p", "yuv420p10le", "p010le", "rgba", "bgr0"] {
            let why = pool_pix_fmt(other).unwrap_err();
            assert!(
                why.contains("yuv420p") && why.contains("nv12"),
                "{other}: {why}"
            );
        }
    }

    #[test]
    fn the_parameter_sets_are_read_out_of_an_annex_b_packet() {
        let h264 = parameter_set_types("h264").expect("h264 has parameter sets");
        // SPS, PPS with trailing zeros, then a slice behind a three-byte
        // start code.
        let packet = [
            0, 0, 0, 1, 0x67, 1, 2, 0, 0, 0, 1, 0x68, 3, 0, 0, 0, 0, 1, 0x65, 9, 9,
        ];
        assert_eq!(
            annex_b_units(&packet, h264),
            vec![0, 0, 0, 1, 0x67, 1, 2, 0, 0, 0, 1, 0x68, 3]
        );
        let hevc = parameter_set_types("hevc").expect("hevc has parameter sets");
        let packet = [0, 0, 1, 0x40, 1, 0, 0, 1, 0x42, 1, 7, 0, 0, 1, 0x26, 1, 5];
        assert_eq!(
            annex_b_units(&packet, hevc),
            vec![0, 0, 0, 1, 0x40, 1, 0, 0, 0, 1, 0x42, 1, 7]
        );
        assert!(parameter_set_types("av1").is_none());
    }
}
