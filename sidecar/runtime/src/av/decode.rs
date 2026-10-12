//! Decoding coded video into the wire's pictures.

use super::error::{Error, Result};
use super::ffi::{AvErr, Codec, CodecCtx, Frame, Lib, Packet, Scaler, Status};
use super::picture::{self, WIRE_PIX_FMTS};
use super::Color;

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
    out: Frame,
    packet: Packet,
    scaler: Option<Scaler>,
    last: Option<(i64, Option<i64>)>,
}

/// A failed libav call, as the shim's error.
fn call<'a>(call: &'static str, codec: &'a str) -> impl FnOnce(AvErr) -> Error + 'a {
    move |e| Error::Call {
        call,
        codec: codec.to_string(),
        why: e.why,
    }
}

/// Opens a software decoder.
fn open_software(codec: Codec, extradata: &[u8], name: &str) -> Result<CodecCtx> {
    let mut ctx = CodecCtx::new(codec).map_err(call("avcodec_alloc_context3", name))?;
    ctx.set_stream(extradata)
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
    ctx.set_stream(extradata).map_err(|e| e.why)?;
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
    /// the wire format pictures come out in.
    ///
    /// Says on stderr, once, where it decodes:
    /// `[av] decode h264: d3d11va`, or
    /// `[av] decode h264: software (d3d11va: <why>; cuda: <why>)`.
    /// A device named outright that cannot be had is a refusal, not a fall
    /// back.
    pub fn open(codec: &str, extradata: &[u8], hw: Hw, want: &str) -> Result<Decoder> {
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

        let (ctx, engaged, unconfirmed, passed_over) = match &hw {
            Hw::None => (
                open_software(preferred, extradata, codec)?,
                "software".to_string(),
                None,
                Vec::new(),
            ),
            Hw::Device(device) => match open_on_device(lib, preferred, extradata, codec, device) {
                Ok(ctx) => (ctx, device.clone(), Some(device.clone()), Vec::new()),
                Err(why) => {
                    return Err(Error::Hardware {
                        codec: codec.to_string(),
                        device: device.clone(),
                        why,
                    })
                }
            },
            Hw::Auto => {
                let mut passed_over = Vec::new();
                let mut opened = None;
                for &device in Hw::AUTO_ORDER {
                    match open_on_device(lib, preferred, extradata, codec, device) {
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
                        open_software(preferred, extradata, codec)?,
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
            out: Frame::new(lib).map_err(call("av_frame_alloc", codec))?,
            packet: Packet::new(lib).map_err(call("av_packet_alloc", codec))?,
            scaler: None,
            last: None,
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
            out,
            scaler,
            format,
            last,
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

        let mut data = Vec::with_capacity(picture::frame_len(want, width, height).unwrap_or(0));
        let packed = if source.format() == *want_fmt {
            source.pack(want, &mut data)
        } else {
            let mut color = source.color();
            if picture::is_rgb(want) {
                color = color.as_rgb();
            }
            out.set_video(width, height, *want_fmt);
            out.set_color(color);
            if scaler.is_none() {
                *scaler = Some(Scaler::new(lib).map_err(call("sws_alloc_context", codec))?);
            }
            let sws = scaler.as_mut().expect("set just above");
            sws.scale(out, source)
                .map_err(call("sws_scale_frame", codec))?;
            let packed = out.pack(want, &mut data);
            out.unref();
            packed
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

        // The picture's own timestamp, else libavcodec's best guess at it,
        // else one frame on from the last.
        let duration = frame.duration();
        let pts = frame
            .pts()
            .or(frame.best_effort_timestamp())
            .or_else(|| last.map(|(p, d)| p + d.unwrap_or(1)))
            .unwrap_or(0);
        *last = Some((pts, duration));

        Ok(DecodedFrame {
            pts,
            duration,
            width,
            height,
            data,
        })
    }
}
