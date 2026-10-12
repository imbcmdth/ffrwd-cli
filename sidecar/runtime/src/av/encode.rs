//! Encoding the wire's pictures into coded video.

use std::os::raw::c_int;

use super::error::{Error, Result};
use super::ffi::{AvErr, CodecCtx, Frame, Lib, Packet, RawColor, Scaler, Status};
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
}

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

/// Whether a colour name leaves the value unspecified.
fn unset(name: &str) -> bool {
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
        let asked = options
            .iter()
            .rev()
            .find(|(k, _)| k == "pix_fmt")
            .map(|(_, v)| v.as_str());
        let pix_fmt = match asked {
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
            None => supported.first().copied().unwrap_or(input_fmt),
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
            name,
        })
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

        let status = ctx.send_frame(Some(frame));
        frame.unref();
        match status {
            Status::Done => Ok(()),
            Status::Again => Err(Error::Call {
                call: "avcodec_send_frame",
                codec: name.clone(),
                why: "the encoder has packets waiting; receive until it answers None, then send"
                    .to_string(),
            }),
            Status::Eof => Err(Error::Call {
                call: "avcodec_send_frame",
                codec: name.clone(),
                why: "the encoder was already flushed".to_string(),
            }),
            Status::Failed(e) => Err(call("avcodec_send_frame", name)(e)),
        }
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
