//! Decoding coded video into the wire's pictures.

use std::os::raw::c_void;

use super::encode::unset;
use super::error::{Error, Result};
use super::ffi::{
    add_ref_com, release_com, AvErr, Codec, CodecCtx, Frame, Lib, Packet, RawColor, Scaler, Status,
};
use super::picture::{self, WIRE_PIX_FMTS};
use super::Color;

/// The wire formats a decoder opened on the caller's device leaves its
/// `nv12` pictures there for: yuv420p, an exact repacking of them, and
/// rgba, which the caller converts on the device.
pub const ON_DEVICE: &[&str] = &["yuv420p", "rgba"];

/// Where a decoder runs.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum Hw {
    /// In software.
    #[default]
    None,
    /// On the first of the platform's devices that opens (see
    /// [`Hw::AUTO_ORDER`]), else in software.
    Auto,
    /// On the hardware device type ffmpeg calls this (`cuda`, `d3d11va`,
    /// `d3d12va`, `dxva2`, `videotoolbox`, `vaapi`, `vulkan`), or not at all.
    Device(String),
}

impl Hw {
    /// The devices `Auto` tries, in order, on this platform.
    #[cfg(windows)]
    pub const AUTO_ORDER: &'static [&'static str] = &["d3d11va", "cuda"];
    #[cfg(target_os = "macos")]
    pub const AUTO_ORDER: &'static [&'static str] = &["videotoolbox"];
    #[cfg(all(unix, not(target_os = "macos")))]
    pub const AUTO_ORDER: &'static [&'static str] = &["vulkan", "vaapi", "cuda"];

    /// `none` (or nothing), `auto`, or a device type's name.
    pub fn parse(raw: &str) -> Hw {
        match raw {
            "" | "none" => Hw::None,
            "auto" => Hw::Auto,
            device => Hw::Device(device.to_string()),
        }
    }
}

/// A device the caller already works on, which a decoder can be opened on
/// so that the pictures it writes are that device's own resources.
pub enum Shared {
    /// An `ID3D12Device`, which becomes the `d3d12va` device.
    D3d12(ComDevice),
    /// A `videotoolbox` device. VideoToolbox takes no device of the
    /// caller's: it writes IOSurface-backed pictures, which any Metal device
    /// on the machine reads.
    VideoToolbox,
    /// A Vulkan device the caller made, which becomes the `vulkan` device.
    Vulkan(super::VulkanDevice),
}

impl Shared {
    /// ffmpeg's name for the device type the decoder runs on.
    pub fn device_type(&self) -> &'static str {
        match self {
            Shared::D3d12(_) => "d3d12va",
            Shared::VideoToolbox => "videotoolbox",
            Shared::Vulkan(_) => "vulkan",
        }
    }

    /// Another reference to the same device.
    pub fn share(&self) -> Shared {
        match self {
            Shared::D3d12(device) => Shared::D3d12(device.share()),
            Shared::VideoToolbox => Shared::VideoToolbox,
            Shared::Vulkan(device) => Shared::Vulkan(device.share()),
        }
    }
}

/// One reference to a COM device object, given up when dropped unless a
/// decoder took it.
pub struct ComDevice(*mut c_void);

// SAFETY: D3D12 devices are free-threaded; the reference is only moved.
unsafe impl Send for ComDevice {}

impl ComDevice {
    /// Takes over one reference to `device`, an `ID3D12Device *`.
    ///
    /// # Safety
    /// `device` is a live `ID3D12Device` the caller holds a reference to,
    /// which this takes.
    pub unsafe fn from_raw(device: *mut c_void) -> ComDevice {
        ComDevice(device)
    }

    /// Another reference to the device.
    pub fn share(&self) -> ComDevice {
        // SAFETY: the reference this holds keeps the device alive; the new
        // one is the returned handle's.
        unsafe { add_ref_com(self.0) };
        ComDevice(self.0)
    }

    pub(super) fn into_raw(self) -> *mut c_void {
        let raw = self.0;
        std::mem::forget(self);
        raw
    }
}

impl Drop for ComDevice {
    fn drop(&mut self) {
        // SAFETY: the reference `from_raw` took.
        unsafe { release_com(self.0) };
    }
}

/// A picture on a [`Shared`] device: one a decoder opened on it left
/// there, or one of an encoder's pool waiting to be written
/// ([`Encoder::device_picture`](super::Encoder::device_picture)). The
/// device's own texture, in `pix_fmt`, kept from reuse until this is
/// dropped or handed to the encoder.
pub struct DeviceFrame {
    pub pts: i64,
    pub duration: Option<i64>,
    pub width: u32,
    pub height: u32,
    /// What the texture holds: `nv12`.
    pub pix_fmt: &'static str,
    pub(super) frame: Frame,
}

/// The D3D12 objects of a [`DeviceFrame`], valid while it is.
#[derive(Clone, Copy, Debug)]
pub struct D3d12Picture {
    /// `ID3D12Resource *`: the texture, an NV12 one.
    pub texture: *mut c_void,
    /// The array slice the picture is in, for a decoder writing a texture
    /// array; 0 otherwise.
    pub slice: u32,
    /// The D3D12 sync object the decoder signals when the
    /// picture is written.
    pub sync: *mut c_void,
    /// The value it signals then.
    pub sync_value: u64,
}

impl DeviceFrame {
    /// The picture's D3D12 objects; None for a picture of another device.
    pub fn d3d12(&self) -> Option<D3d12Picture> {
        let picture = self.frame.d3d12()?;
        // SAFETY: the frame holds the picture, so the struct is alive.
        unsafe {
            let p = &*picture;
            Some(D3d12Picture {
                texture: p.texture,
                slice: p.subresource_index.max(0) as u32,
                sync: p.sync_ctx.sync,
                sync_value: p.sync_ctx.sync_value,
            })
        }
    }

    /// The picture's Vulkan images, under libav's lock on the picture until
    /// the result is dropped; None for a picture of another device.
    pub fn vulkan(&mut self) -> Option<super::VkPictureLock<'_>> {
        let (frame, frames) = self.frame.vulkan()?;
        // SAFETY: the frame holds the picture and a reference to its frames
        // context, both alive while it is, which the lock borrows.
        Some(unsafe { super::VkPictureLock::lock(frame, frames) })
    }

    /// The picture's `CVPixelBufferRef`, valid while it is; None for a
    /// picture of another device.
    pub fn cv_pixel_buffer(&self) -> Option<*mut c_void> {
        self.frame.videotoolbox()
    }

    /// Says the reader signals the picture's sync object with `value` once
    /// it is done reading the texture: the decoder waits for that value
    /// before it writes the texture again. Set before the frame is dropped.
    pub fn read_until(&mut self, value: u64) {
        self.set_sync_value(value);
    }

    /// Says the texture of an encoder's picture is written once its sync
    /// object reaches `value`: the encoder waits for that value before it
    /// reads it. Set before the picture is sent.
    pub fn written_at(&mut self, value: u64) {
        self.set_sync_value(value);
    }

    fn set_sync_value(&mut self, value: u64) {
        if let Some(picture) = self.frame.d3d12() {
            // SAFETY: the frame holds the picture; the codec reads the value
            // only once this reference has been dropped or handed to it.
            unsafe { (*picture).sync_ctx.sync_value = value };
        }
    }
}

/// What [`Decoder::receive_any`] hands back: a picture in host memory, or
/// one left on the shared device.
pub enum Received {
    Host(DecodedFrame),
    Device(DeviceFrame),
}

/// One decoded picture, packed as the wire lays out the format the decoder
/// was opened to produce.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecodedFrame {
    pub pts: i64,
    pub duration: Option<i64>,
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
}

/// What the stream turned out to be, once its first picture has decoded.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecodedFormat {
    pub width: u32,
    pub height: u32,
    /// The wire format pictures come out in: what the decoder was asked for.
    pub pix_fmt: String,
    /// The format the decoder produced them in, before conversion; for a
    /// picture copied back from a device, the format it was copied as.
    pub coded_pix_fmt: String,
    pub color: Color,
}

/// A decoder for one coded video stream.
///
/// Packets go in with [`send`](Decoder::send); after each, call
/// [`receive`](Decoder::receive) until it answers None. At the end of the
/// stream, [`flush`](Decoder::flush), then `receive` until None again.
pub struct Decoder {
    lib: &'static Lib,
    codec: String,
    want: &'static str,
    want_fmt: i32,
    ctx: CodecCtx,
    engaged: String,
    /// The device asked for, until the first picture says whether it
    /// engaged.
    unconfirmed: Option<String>,
    format: Option<DecodedFormat>,
    frame: Frame,
    host: Frame,
    packet: Packet,
    scaler: Option<Scaler>,
    last: Option<(i64, Option<i64>)>,
    /// Opened on a [`Shared`] device: pictures stay on it where the wanted
    /// format allows.
    shared: bool,
}

/// A failed libav call, as the shim's error.
fn call<'a>(call: &'static str, codec: &'a str) -> impl FnOnce(AvErr) -> Error + 'a {
    move |e| Error::Call {
        call,
        codec: codec.to_string(),
        why: e.why,
    }
}

/// `color` as libavutil's values, a field it leaves unsaid unspecified.
fn raw_color(lib: &Lib, color: Option<&Color>, codec: &str) -> Result<RawColor> {
    let Some(c) = color else {
        return Ok(RawColor::UNSPECIFIED);
    };
    let names =
        [&c.range, &c.primaries, &c.trc, &c.space].map(|n| if unset(n) { "" } else { n.as_str() });
    lib.color_values(names).map_err(|(key, value)| Error::Call {
        call: "avcodec_parameters_to_context",
        codec: codec.to_string(),
        why: format!("libavutil has no {key} called {value}"),
    })
}

/// Opens a software decoder.
fn open_software(codec: Codec, extradata: &[u8], color: RawColor, name: &str) -> Result<CodecCtx> {
    let mut ctx = CodecCtx::new(codec).map_err(call("avcodec_alloc_context3", name))?;
    ctx.set_stream(extradata, color)
        .map_err(call("avcodec_parameters_to_context", name))?;
    // ffmpeg's command line decodes with as many threads as it has cores.
    ctx.set_option("threads", "auto")
        .map_err(call("av_opt_set threads", name))?;
    ctx.open().map_err(call("avcodec_open2", name))?;
    Ok(ctx)
}

/// Opens a decoder on a device of type `device`, or says why not: a decoder
/// for the codec that takes that device, the device itself, and the open.
fn open_on_device(
    lib: &'static Lib,
    preferred: Codec,
    extradata: &[u8],
    color: RawColor,
    name: &str,
    device: &str,
) -> std::result::Result<CodecCtx, String> {
    let Some(device_type) = lib.hw_type(device) else {
        return Err(format!("this libav cannot create a {device} device"));
    };
    let Some(codec) = lib
        .decoders_like(preferred)
        .into_iter()
        .find(|c| !c.is_wrapper_or_experimental() && c.takes_device(device_type))
    else {
        return Err(format!("no {name} decoder here decodes on {device}"));
    };
    let mut ctx = CodecCtx::new(codec).map_err(|e| e.why)?;
    ctx.set_stream(extradata, color).map_err(|e| e.why)?;
    ctx.attach_device(device_type)
        .map_err(|e| format!("no device: {}", e.why))?;
    ctx.set_option("threads", "auto").map_err(|e| e.why)?;
    ctx.open().map_err(|e| e.why)?;
    Ok(ctx)
}

impl Decoder {
    /// A decoder for `codec`, by ffmpeg's name for it (`h264`, `hevc`,
    /// `av1`, `vp9`, `mpeg2video`, `prores`, `mjpeg`, `ffv1`, `mpeg4`), or a
    /// particular decoder's own name (`libdav1d`). `extradata` is the
    /// codec's out-of-band header, as the container carried it; `want` is
    /// the wire format pictures come out in. `color` is the stream's colour
    /// as the container said it, which the decoder's pictures carry where
    /// the bitstream says none, as ffmpeg's own decode does.
    ///
    /// Says on stderr, once, where it decodes:
    /// `[av] decode h264: d3d11va`, or
    /// `[av] decode h264: software (d3d11va: <why>; cuda: <why>)`.
    /// A device named outright that cannot be had is a refusal, not a fall
    /// back.
    pub fn open(
        codec: &str,
        extradata: &[u8],
        hw: Hw,
        want: &str,
        color: Option<&Color>,
    ) -> Result<Decoder> {
        let lib = super::runtime()?;
        let Some(want) = picture::wire_pix_fmt(want) else {
            return Err(Error::PixelFormat {
                name: want.to_string(),
                why: format!("a decoder produces one of {}", WIRE_PIX_FMTS.join(", ")),
            });
        };
        let want_fmt = lib
            .wire_pix_fmt(want)
            .expect("every wire format was found at load");
        let Some(preferred) = lib.find_decoder(codec) else {
            return Err(Error::Codec {
                role: "decoder",
                name: codec.to_string(),
            });
        };
        let color = raw_color(lib, color, codec)?;

        let (ctx, engaged, unconfirmed, passed_over) = match &hw {
            Hw::None => (
                open_software(preferred, extradata, color, codec)?,
                "software".to_string(),
                None,
                Vec::new(),
            ),
            Hw::Device(device) => {
                match open_on_device(lib, preferred, extradata, color, codec, device) {
                    Ok(ctx) => (ctx, device.clone(), Some(device.clone()), Vec::new()),
                    Err(why) => {
                        return Err(Error::Hardware {
                            codec: codec.to_string(),
                            device: device.clone(),
                            why,
                        })
                    }
                }
            }
            Hw::Auto => {
                let mut passed_over = Vec::new();
                let mut opened = None;
                for &device in Hw::AUTO_ORDER {
                    match open_on_device(lib, preferred, extradata, color, codec, device) {
                        Ok(ctx) => {
                            opened = Some((ctx, device));
                            break;
                        }
                        Err(why) => passed_over.push(format!("{device}: {why}")),
                    }
                }
                match opened {
                    Some((ctx, device)) => (
                        ctx,
                        device.to_string(),
                        Some(device.to_string()),
                        passed_over,
                    ),
                    None => (
                        open_software(preferred, extradata, color, codec)?,
                        "software".to_string(),
                        None,
                        passed_over,
                    ),
                }
            }
        };

        if engaged == "software" && !passed_over.is_empty() {
            eprintln!("[av] decode {codec}: software ({})", passed_over.join("; "));
        } else {
            eprintln!("[av] decode {codec}: {engaged}");
        }

        Ok(Decoder {
            lib,
            codec: codec.to_string(),
            want,
            want_fmt,
            ctx,
            engaged,
            unconfirmed,
            format: None,
            frame: Frame::new(lib).map_err(call("av_frame_alloc", codec))?,
            host: Frame::new(lib).map_err(call("av_frame_alloc", codec))?,
            packet: Packet::new(lib).map_err(call("av_packet_alloc", codec))?,
            scaler: None,
            last: None,
            shared: false,
        })
    }

    /// A decoder on `device`, a device the caller already works on, so the
    /// pictures it writes are that device's resources and
    /// [`receive_any`](Decoder::receive_any) can hand them over where they
    /// are. An `nv12` picture stays there for a `want` the caller makes of
    /// it on the device ([`ON_DEVICE`]); any other is copied back and
    /// converted as [`open`](Decoder::open)'s are. `color` is as
    /// [`open`](Decoder::open) takes it.
    ///
    /// Says on stderr, once: `[av] decode h264: d3d12va, on the host's
    /// device`. A decoder or device that refuses is a refusal, and the
    /// reference to `device` is given up either way.
    pub fn open_shared(
        codec: &str,
        extradata: &[u8],
        device: Shared,
        want: &str,
        color: Option<&Color>,
    ) -> Result<Decoder> {
        let lib = super::runtime()?;
        let Some(want) = picture::wire_pix_fmt(want) else {
            return Err(Error::PixelFormat {
                name: want.to_string(),
                why: format!("a decoder produces one of {}", WIRE_PIX_FMTS.join(", ")),
            });
        };
        let want_fmt = lib
            .wire_pix_fmt(want)
            .expect("every wire format was found at load");
        let Some(preferred) = lib.find_decoder(codec) else {
            return Err(Error::Codec {
                role: "decoder",
                name: codec.to_string(),
            });
        };
        let device_name = device.device_type();
        let refused = |why: String| Error::Hardware {
            codec: codec.to_string(),
            device: device_name.to_string(),
            why,
        };
        let Some(device_type) = lib.hw_type(device_name) else {
            return Err(refused(format!(
                "this libav cannot create a {device_name} device"
            )));
        };
        let Some(found) = lib
            .decoders_like(preferred)
            .into_iter()
            .find(|c| !c.is_wrapper_or_experimental() && c.takes_device(device_type))
        else {
            return Err(refused(format!(
                "no {codec} decoder here decodes on {device_name}"
            )));
        };
        let mut ctx = CodecCtx::new(found).map_err(|e| refused(e.why))?;
        let color = raw_color(lib, color, codec)?;
        ctx.set_stream(extradata, color)
            .map_err(|e| refused(e.why))?;
        match device {
            Shared::D3d12(raw) => ctx
                .attach_d3d12_device(device_type, raw.into_raw())
                .map_err(|e| refused(format!("the host's device: {}", e.why)))?,
            Shared::VideoToolbox => ctx.attach_device(device_type).map_err(|e| refused(e.why))?,
            Shared::Vulkan(device) => ctx
                .attach_shared_device(device.device())
                .map_err(|e| refused(format!("the host's device: {}", e.why)))?,
        }
        ctx.set_option("threads", "auto")
            .map_err(|e| refused(e.why))?;
        ctx.open().map_err(|e| refused(e.why))?;
        eprintln!("[av] decode {codec}: {device_name}, on the host's device");
        Ok(Decoder {
            lib,
            codec: codec.to_string(),
            want,
            want_fmt,
            ctx,
            engaged: device_name.to_string(),
            unconfirmed: Some(device_name.to_string()),
            format: None,
            frame: Frame::new(lib).map_err(call("av_frame_alloc", codec))?,
            host: Frame::new(lib).map_err(call("av_frame_alloc", codec))?,
            packet: Packet::new(lib).map_err(call("av_packet_alloc", codec))?,
            scaler: None,
            last: None,
            shared: true,
        })
    }

    /// `software`, or the device decoding runs on. A device that opened
    /// but whose first picture came back in host memory reads `software`
    /// from then on, and stderr says so.
    pub fn engaged(&self) -> &str {
        &self.engaged
    }

    /// What the stream is, once its first picture has decoded; None before.
    pub fn format(&self) -> Option<&DecodedFormat> {
        self.format.as_ref()
    }

    /// The decoder libavcodec chose: `h264`, or `libdav1d` for `av1`.
    pub fn decoder_name(&self) -> String {
        self.ctx.codec().name()
    }

    /// Hands the decoder one packet. An empty packet is skipped: libavcodec
    /// reads an empty packet as the end of the stream, which is `flush`'s.
    pub fn send(
        &mut self,
        pts: i64,
        dts: Option<i64>,
        duration: Option<i64>,
        keyframe: bool,
        data: &[u8],
    ) -> Result<()> {
        if data.is_empty() {
            return Ok(());
        }
        self.packet
            .fill(data, pts, dts, duration, keyframe)
            .map_err(call("av_new_packet", &self.codec))?;
        let status = self.ctx.send_packet(Some(&self.packet));
        self.packet.unref();
        match status {
            Status::Done => Ok(()),
            Status::Again => Err(Error::Call {
                call: "avcodec_send_packet",
                codec: self.codec.clone(),
                why: "the decoder has pictures waiting; receive until it answers None, \
                      then send"
                    .to_string(),
            }),
            Status::Eof => Err(Error::Call {
                call: "avcodec_send_packet",
                codec: self.codec.clone(),
                why: "the decoder was already flushed".to_string(),
            }),
            Status::Failed(e) => Err(call("avcodec_send_packet", &self.codec)(e)),
        }
    }

    /// Ends the stream. The pictures the decoder still holds come out of
    /// `receive` after this.
    pub fn flush(&mut self) -> Result<()> {
        match self.ctx.send_packet(None) {
            Status::Done | Status::Eof => Ok(()),
            Status::Again => Err(Error::Call {
                call: "avcodec_send_packet",
                codec: self.codec.clone(),
                why: "the decoder has pictures waiting; receive until it answers None, \
                      then flush"
                    .to_string(),
            }),
            Status::Failed(e) => Err(call("avcodec_send_packet", &self.codec)(e)),
        }
    }

    /// The next decoded picture, or None when the decoder wants another
    /// packet (or, after `flush`, has none left).
    pub fn receive(&mut self) -> Result<Option<DecodedFrame>> {
        match self.ctx.receive_frame(&mut self.frame) {
            Status::Done => {}
            Status::Again | Status::Eof => return Ok(None),
            Status::Failed(e) => return Err(call("avcodec_receive_frame", &self.codec)(e)),
        }
        let picture = self.convert();
        self.frame.unref();
        picture.map(Some)
    }

    /// As [`receive`](Decoder::receive), except that a decoder opened with
    /// [`open_shared`](Decoder::open_shared) hands a picture over where it
    /// is, on the device, when the wanted format is an exact repacking of
    /// it.
    pub fn receive_any(&mut self) -> Result<Option<Received>> {
        match self.ctx.receive_frame(&mut self.frame) {
            Status::Done => {}
            Status::Again | Status::Eof => return Ok(None),
            Status::Failed(e) => return Err(call("avcodec_receive_frame", &self.codec)(e)),
        }
        if let Some(held) = self.held_as() {
            let picture = self.keep(held);
            self.frame.unref();
            return picture.map(|p| Some(Received::Device(p)));
        }
        let picture = self.convert();
        self.frame.unref();
        picture.map(|p| Some(Received::Host(p)))
    }

    /// The format a picture just received stays on the device in, where
    /// it does: an `nv12` D3D12, VideoToolbox or Vulkan picture, wanted as one of
    /// [`ON_DEVICE`].
    fn held_as(&self) -> Option<&'static str> {
        if !self.shared || !ON_DEVICE.contains(&self.want) {
            return None;
        }
        if self.frame.d3d12().is_none()
            && self.frame.videotoolbox().is_none()
            && self.frame.vulkan().is_none()
        {
            return None;
        }
        let nv12 = self.lib.pix_fmt("nv12")?;
        (self.frame.sw_format()? == nv12).then_some("nv12")
    }

    /// The picture just received, kept on the device as `pix_fmt`.
    fn keep(&mut self, pix_fmt: &'static str) -> Result<DeviceFrame> {
        self.unconfirmed = None;
        let (width, height) = (self.frame.width(), self.frame.height());
        if self.format.is_none() {
            let [range, primaries, trc, space] = self.lib.color_names(self.frame.color());
            self.format = Some(DecodedFormat {
                width,
                height,
                pix_fmt: self.want.to_string(),
                coded_pix_fmt: pix_fmt.to_string(),
                color: Color {
                    range,
                    primaries,
                    trc,
                    space,
                },
            });
        }
        let (pts, duration) = self.timing();
        let mut frame = Frame::new(self.lib).map_err(call("av_frame_alloc", &self.codec))?;
        frame
            .take(&mut self.frame)
            .map_err(call("av_frame_ref", &self.codec))?;
        Ok(DeviceFrame {
            pts,
            duration,
            width,
            height,
            pix_fmt,
            frame,
        })
    }

    /// The picture's own timestamp, else libavcodec's best guess at it,
    /// else one frame on from the last; and its duration.
    fn timing(&mut self) -> (i64, Option<i64>) {
        let duration = self.frame.duration();
        let last = self.last;
        let pts = self
            .frame
            .pts()
            .or(self.frame.best_effort_timestamp())
            .or_else(|| last.map(|(p, d)| p + d.unwrap_or(1)))
            .unwrap_or(0);
        self.last = Some((pts, duration));
        (pts, duration)
    }

    /// The picture just received, in host memory, in the wanted format,
    /// packed.
    fn convert(&mut self) -> Result<DecodedFrame> {
        let on_device = self.frame.is_hardware();
        if let Some(device) = self.unconfirmed.take() {
            if !on_device {
                eprintln!(
                    "[av] decode {}: software ({device} opened, and the first picture \
                     came back in host memory)",
                    self.codec
                );
                self.engaged = "software".to_string();
            }
        }

        let Decoder {
            lib,
            codec,
            want,
            want_fmt,
            frame,
            host,
            scaler,
            format,
            ..
        } = self;

        let source: &Frame = if on_device {
            host.download(frame)
                .map_err(call("av_hwframe_transfer_data", codec))?;
            host
        } else {
            frame
        };
        let (width, height) = (source.width(), source.height());

        if format.is_none() {
            let [range, primaries, trc, space] = lib.color_names(source.color());
            *format = Some(DecodedFormat {
                width,
                height,
                pix_fmt: want.to_string(),
                coded_pix_fmt: lib.pix_fmt_name(source.format()),
                color: Color {
                    range,
                    primaries,
                    trc,
                    space,
                },
            });
        }

        // One pass over the picture either way: copied out row by row where
        // it is already the wanted format, else converted by libswscale
        // straight into the packed vector.
        let mut data = Vec::new();
        let packed = if source.format() == *want_fmt {
            data.reserve_exact(picture::frame_len(want, width, height).unwrap_or(0));
            source.pack(want, &mut data)
        } else {
            let mut color = source.color();
            if picture::is_rgb(want) {
                color = color.as_rgb();
            }
            if scaler.is_none() {
                *scaler = Some(Scaler::new(lib).map_err(call("sws_alloc_context", codec))?);
            }
            let sws = scaler.as_mut().expect("set just above");
            sws.scale_into_packed(source, want, color, &mut data)
                .map_err(call("sws_scale_frame", codec))?
        };
        if !packed {
            return Err(Error::Call {
                call: "sws_scale_frame",
                codec: codec.clone(),
                why: format!("the converted {width}x{height} picture is not laid out as {want}"),
            });
        }
        if on_device {
            host.unref();
        }

        let (pts, duration) = self.timing();
        Ok(DecodedFrame {
            pts,
            duration,
            width,
            height,
            data,
        })
    }
}
