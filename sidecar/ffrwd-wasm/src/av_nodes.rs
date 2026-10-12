//! The host's own `decode` and `encode`: a coded picture stream turned into
//! frames through libavcodec, and frames turned back into a coded stream.
//!
//! Each is spelled like a module and wired by position -
//! `[0:v]decode=format=rgba[d]`, `[n1]encode=codec=h264_nvenc:b=6M[out0]` -
//! but no `-m` binds it and nothing is instantiated. Each is one instance,
//! called in order, since libavcodec keeps the stream's state from call to
//! call, and each call is handed whatever has arrived: `decode` its packets
//! in decode order, `encode` its frames in presentation order. A frame
//! leaves `decode` in presentation order as soon as the decoder makes it,
//! and a packet leaves `encode` in decode order the same way. The last call
//! flushes the codec and hands on what it still held.

use std::sync::Arc;

use anyhow::{anyhow, bail, Result};
use ffrwd_wasm::nut;
use ffrwd_wasm_runtime::av::{self, Decoder, Encoder, Hw, WIRE_PIX_FMTS};
use ffrwd_wasm_runtime::node::{
    Accepts, Clock, Emission, Emitted, FrameMemory, InputPort, Node, NodeShape, OutFrame,
    OutputPort, Pairing, Payload, PortKind, Rational, RowsUse, StreamFormat, Tick,
};
use ffrwd_wasm_runtime::runtime::{
    CodedFormat, CodedStream, ColorInfo, Packet, TimeBase, VideoFormat,
};

use crate::codec::frame_ticks;

/// The node that decodes.
pub const DECODE: &str = "decode";
/// The node that encodes.
pub const ENCODE: &str = "encode";

const FORMAT: &str = "format";
const HW: &str = "hw";
const CODEC: &str = "codec";

/// Whether `module` names `decode` or `encode`.
pub fn is_av_node(module: &str) -> bool {
    module == DECODE || module == ENCODE
}

/// The shape both have: one input by position, one output, ticking as
/// things arrive.
fn shape(input: PortKind, output: PortKind, base: TimeBase, latency: f64) -> NodeShape {
    let time_base = match (i32::try_from(base.num), i32::try_from(base.den)) {
        (Ok(num), Ok(den)) => Some(Rational { num, den }),
        _ => None,
    };
    NodeShape {
        inputs: vec![InputPort {
            name: "in0".into(),
            kind: input,
            required: true,
            many: false,
            pairing: Pairing::Arrival,
            rows: RowsUse::Ignore,
            window: 1,
            stride: 1,
            accepts: Accepts::default(),
            schema: None,
        }],
        outputs: vec![OutputPort {
            name: "out".into(),
            kind: output,
            format: None,
            time_base,
            latency,
            schema: None,
            memory: FrameMemory::Host,
            row: None,
        }],
        clock: Clock::SelfClocked,
        pure: false,
        one_to_one: false,
        bounded: true,
        relation: Vec::new(),
    }
}

/// What a stream is, for a refusal: `raw yuv420p video`, `coded h264
/// video`.
fn carried(format: &StreamFormat) -> String {
    match format {
        StreamFormat::Video(v) => format!("raw {} video", v.pix_fmt),
        StreamFormat::Audio(a) => format!("raw {} audio", a.sample_fmt),
        StreamFormat::Data(codec) => format!("a {codec} data stream"),
        StreamFormat::Packets(coded) => format!("coded {} {}", coded.codec, coded.format.kind()),
    }
}

/// A colour name read back from libav, as the `'static` the host's formats
/// hold. Read once a run per stream, so the few bytes stay for the process.
fn leaked(name: &str) -> &'static str {
    Box::leak(name.to_string().into_boxed_str())
}

/// The codec libavcodec opens for a coded stream: the one its tag on the
/// wire names, else the name it already carries (`ffv1`, from an `encode`
/// upstream).
fn decoder_codec(coded: &CodedStream, fourcc: Option<&[u8]>) -> String {
    let by_tag = |tag: &[u8]| {
        <&[u8; 4]>::try_from(tag)
            .ok()
            .and_then(av::codec_for_fourcc)
    };
    fourcc
        .and_then(by_tag)
        .or_else(|| by_tag(coded.codec.as_bytes()))
        .map_or_else(|| coded.codec.clone(), str::to_string)
}

/// `decode`: one coded picture stream in, its pictures out in `format`.
pub struct Decode {
    name: String,
    shape: NodeShape,
    decoder: Decoder,
    /// The pictures as the stream's header declares them.
    format: VideoFormat,
    /// How long a picture lasts, where the frame rate says in whole ticks.
    duration: Option<i64>,
}

impl Decode {
    /// Opens the decoder for the coded stream `input` in `base`, carried
    /// under `fourcc` where its header names one. `delay` is the packets
    /// the header says the decoder holds back.
    pub fn open(
        options: &[(String, String)],
        input: &StreamFormat,
        spelling: &str,
        fourcc: Option<&[u8]>,
        base: TimeBase,
        frame_rate: Option<(u64, u64)>,
        delay: u32,
    ) -> Result<Decode> {
        let mut want: Option<&'static str> = None;
        let mut hw: Option<Hw> = None;
        for (key, value) in options {
            match key.as_str() {
                FORMAT if want.is_some() => bail!("{DECODE} is given the option '{FORMAT}' twice"),
                FORMAT => {
                    let Some(found) = WIRE_PIX_FMTS.iter().copied().find(|f| f == value) else {
                        bail!(
                            "{DECODE}={FORMAT}={value}: a decoder writes one of {}",
                            WIRE_PIX_FMTS.join(", ")
                        );
                    };
                    want = Some(found);
                }
                HW if hw.is_some() => bail!("{DECODE} is given the option '{HW}' twice"),
                HW => hw = Some(Hw::parse(value)),
                other => bail!(
                    "{DECODE} has no option '{other}'; it takes {FORMAT}=<pix_fmt> ({}) and \
                     {HW}=<none|auto|device>",
                    WIRE_PIX_FMTS.join(", ")
                ),
            }
        }
        let Some(want) = want else {
            bail!(
                "{DECODE} names the pixel format it writes with {FORMAT}=<pix_fmt>, one of {}; \
                 this call names none",
                WIRE_PIX_FMTS.join(", ")
            );
        };
        let coded = match input {
            StreamFormat::Packets(coded) if matches!(coded.format, CodedFormat::Video { .. }) => {
                coded
            }
            other => bail!(
                "{DECODE} reads a coded picture stream, and {spelling} is {}; a raw picture \
                 needs no {DECODE}",
                carried(other)
            ),
        };
        let CodedFormat::Video {
            width,
            height,
            color,
            ..
        } = coded.format
        else {
            unreachable!("matched as video above");
        };
        let codec = decoder_codec(coded, fourcc);
        let decoder = Decoder::open(&codec, &coded.extradata, hw.unwrap_or_default(), want)
            .map_err(|e| anyhow!("{DECODE} {spelling}: {e}"))?;
        let format = VideoFormat {
            width,
            height,
            pix_fmt: want,
            frame_len: crate::frame_len_for(want, width, height)?,
            color,
        };
        // The pictures the decoder holds back, as seconds of the stream: a
        // lockstep reader is told to wait that long past a tick.
        let latency = match frame_rate {
            Some((num, den)) if num > 0 => f64::from(delay) * den as f64 / num as f64,
            _ => 0.0,
        };
        Ok(Decode {
            name: DECODE.to_string(),
            shape: shape(PortKind::Packets, PortKind::Video, base, latency),
            decoder,
            format,
            duration: frame_ticks(frame_rate, base),
        })
    }

    /// The pictures this node writes, as the header declared them.
    pub fn format(&self) -> VideoFormat {
        self.format
    }

    /// Every picture the decoder has ready, onto `items`.
    fn drain(&mut self, items: &mut Vec<Emission>) -> Result<()> {
        while let Some(frame) = self
            .decoder
            .receive()
            .map_err(|e| anyhow!("{}: {e}", self.name))?
        {
            if (frame.width, frame.height) != (self.format.width, self.format.height) {
                bail!(
                    "{}: the stream's header says {}x{}, and the decoder made a {}x{} picture at \
                     pts {}; a stream that changes size mid-way is not carried",
                    self.name,
                    self.format.width,
                    self.format.height,
                    frame.width,
                    frame.height,
                    frame.pts
                );
            }
            items.push(Emission {
                port: 0,
                payload: Payload::Frame(OutFrame {
                    pts: frame.pts,
                    duration: frame.duration.or(self.duration),
                    data: Arc::new(frame.data).into(),
                    rows: Vec::new(),
                }),
            });
        }
        Ok(())
    }
}

impl Node for Decode {
    fn name(&self) -> &str {
        &self.name
    }

    fn shape(&self) -> &NodeShape {
        &self.shape
    }

    /// The header's geometry and colour until a picture has decoded, and
    /// what the decoder made of the stream after.
    fn settled_format(&self, _port: usize) -> Option<StreamFormat> {
        let Some(decoded) = self.decoder.format() else {
            return Some(StreamFormat::Video(self.format));
        };
        let color = ColorInfo {
            range: leaked(&decoded.color.range),
            primaries: leaked(&decoded.color.primaries),
            trc: leaked(&decoded.color.trc),
            space: leaked(&decoded.color.space),
        };
        Some(StreamFormat::Video(VideoFormat {
            width: decoded.width,
            height: decoded.height,
            frame_len: crate::frame_len_for(self.format.pix_fmt, decoded.width, decoded.height)
                .ok()?,
            color: Some(color),
            ..self.format
        }))
    }

    fn decode_delay(&self, _port: usize) -> Option<u32> {
        Some(0)
    }

    fn set_params(&mut self, _params: &str) -> Result<()> {
        bail!(
            "{} takes its options from the command line and no params",
            self.name
        )
    }

    fn process(&mut self, tick: Tick) -> Result<Emitted> {
        let mut items = Vec::new();
        for stream in tick.streams {
            for packet in stream.packets {
                self.decoder
                    .send(
                        packet.pts,
                        packet.dts,
                        packet.duration.or(self.duration),
                        packet.keyframe,
                        &packet.data,
                    )
                    .map_err(|e| anyhow!("{}: {e}", self.name))?;
                self.drain(&mut items)?;
            }
        }
        if tick.last {
            self.decoder
                .flush()
                .map_err(|e| anyhow!("{}: {e}", self.name))?;
            self.drain(&mut items)?;
        }
        Ok(Emitted {
            items,
            rows: Vec::new(),
            finished: false,
        })
    }
}

/// `encode`: one raw picture stream in, the encoder's packets out.
pub struct Encode {
    name: String,
    shape: NodeShape,
    encoder: Encoder,
    coded: CodedStream,
    duration: Option<i64>,
}

impl Encode {
    /// Opens the encoder `codec=` names for the raw pictures `input` in
    /// `base`, handing it every other option by ffmpeg's name for it.
    /// `aspect` is the input's pixel aspect, which the coded stream
    /// declares in turn.
    pub fn open(
        options: &[(String, String)],
        input: &StreamFormat,
        spelling: &str,
        base: TimeBase,
        frame_rate: Option<(u64, u64)>,
        aspect: Option<(i32, i32)>,
    ) -> Result<Encode> {
        let mut codec: Option<&str> = None;
        let mut handed: Vec<(String, String)> = Vec::with_capacity(options.len());
        for (key, value) in options {
            if key == CODEC {
                if codec.is_some() {
                    bail!("{ENCODE} is given the option '{CODEC}' twice");
                }
                codec = Some(value);
            } else {
                handed.push((key.clone(), value.clone()));
            }
        }
        let Some(codec) = codec.filter(|c| !c.is_empty()) else {
            bail!(
                "{ENCODE} names its encoder with {CODEC}=<name> (codec=h264_nvenc, \
                 codec=ffv1); this call names none"
            );
        };
        let StreamFormat::Video(video) = input else {
            bail!(
                "{ENCODE} reads raw pictures, and {spelling} is {}",
                carried(input)
            );
        };
        let color = video.color.map(|c| av::Color {
            range: c.range.to_string(),
            primaries: c.primaries.to_string(),
            trc: c.trc.to_string(),
            space: c.space.to_string(),
        });
        let rate = frame_rate
            .and_then(|(num, den)| Some((i64::try_from(num).ok()?, i64::try_from(den).ok()?)));
        let time_base = (
            i64::try_from(base.num).unwrap_or(i64::MAX),
            i64::try_from(base.den).unwrap_or(i64::MAX),
        );
        let encoder = Encoder::open(
            codec,
            video.width,
            video.height,
            video.pix_fmt,
            time_base,
            rate,
            color.as_ref(),
            &handed,
        )
        .map_err(|e| anyhow!("{ENCODE} {spelling}: {e}"))?;
        eprintln!(
            "[av] encode {}: {}x{} {}",
            encoder.name(),
            video.width,
            video.height,
            encoder.pix_fmt()
        );
        let coded = CodedStream {
            codec: encoder.codec().to_string(),
            time_base: base,
            format: CodedFormat::Video {
                width: video.width,
                height: video.height,
                sample_aspect_ratio: aspect,
                color: video.color,
            },
            extradata: encoder.extradata().to_vec(),
            profile: None,
            level: None,
        };
        Ok(Encode {
            name: ENCODE.to_string(),
            shape: shape(PortKind::Video, PortKind::Packets, base, 0.0),
            encoder,
            coded,
            duration: frame_ticks(frame_rate, base),
        })
    }

    /// The coded stream this node writes.
    pub fn coded(&self) -> &CodedStream {
        &self.coded
    }

    /// The packets a decoder of this stream holds back.
    pub fn delay(&self) -> u32 {
        self.encoder.delay()
    }

    /// The NUT header the packets leave under: the wire's own for a codec
    /// its table names, else the encoder's tag, with the extradata, the
    /// reorder depth, the pixel aspect and the colour either way.
    pub fn header(&self) -> Result<nut::Stream> {
        let delay = u64::from(self.delay());
        if nut::fourcc_for_coded("video", &self.coded.codec).is_some() {
            return crate::coded_stream_for(&self.coded, delay);
        }
        let Some(fourcc) = self.encoder.fourcc() else {
            bail!(
                "{}: {} writes {}, which has no tag this wire carries it under",
                self.name,
                self.encoder.name(),
                self.coded.codec
            );
        };
        let CodedFormat::Video {
            width,
            height,
            sample_aspect_ratio,
            color,
        } = &self.coded.format
        else {
            unreachable!("an encoder writes video");
        };
        let mut stream = nut::Stream::coded_fourcc(
            "video",
            &fourcc,
            (*width, *height),
            nut::TimeBase {
                num: self.coded.time_base.num,
                den: self.coded.time_base.den,
            },
            self.coded.extradata.clone(),
            delay,
        )
        .ok_or_else(|| {
            anyhow!(
                "{}: {} writes under the tag {}, which this wire carries as something other \
                 than a video codec of its own",
                self.name,
                self.encoder.name(),
                String::from_utf8_lossy(&fourcc)
            )
        })?;
        if let nut::Media::Video {
            sample_width,
            sample_height,
            colorspace_type,
            ..
        } = &mut stream.media
        {
            if let Some((num, den)) = sample_aspect_ratio.filter(|(n, d)| *n > 0 && *d > 0) {
                (*sample_width, *sample_height) = (num as u64, den as u64);
            }
            *colorspace_type = crate::colorspace_type_for(color.as_ref());
        }
        Ok(stream)
    }

    /// Every packet the encoder has ready, onto `items`.
    fn drain(&mut self, items: &mut Vec<Emission>) -> Result<()> {
        while let Some(packet) = self
            .encoder
            .receive()
            .map_err(|e| anyhow!("{}: {e}", self.name))?
        {
            items.push(Emission {
                port: 0,
                payload: Payload::Packet(Packet {
                    pts: packet.pts,
                    dts: packet.dts,
                    duration: packet.duration,
                    keyframe: packet.keyframe,
                    data: packet.data,
                }),
            });
        }
        Ok(())
    }
}

impl Node for Encode {
    fn name(&self) -> &str {
        &self.name
    }

    fn shape(&self) -> &NodeShape {
        &self.shape
    }

    fn settled_format(&self, _port: usize) -> Option<StreamFormat> {
        Some(StreamFormat::Packets(self.coded.clone()))
    }

    fn decode_delay(&self, _port: usize) -> Option<u32> {
        Some(self.delay())
    }

    fn set_params(&mut self, _params: &str) -> Result<()> {
        bail!(
            "{} takes its options from the command line and no params",
            self.name
        )
    }

    fn process(&mut self, tick: Tick) -> Result<Emitted> {
        let mut items = Vec::new();
        for stream in tick.streams {
            for frame in stream.frames {
                // A frame on the GPU is read back here; the encoder takes
                // bytes until it reads the device's own frames.
                let bytes = frame
                    .data
                    .bytes()
                    .map_err(|e| anyhow!("{}: {e}", self.name))?;
                self.encoder
                    .send(frame.pts, frame.duration.or(self.duration), &bytes)
                    .map_err(|e| anyhow!("{}: {e}", self.name))?;
                self.drain(&mut items)?;
            }
        }
        if tick.last {
            self.encoder
                .flush()
                .map_err(|e| anyhow!("{}: {e}", self.name))?;
            self.drain(&mut items)?;
        }
        Ok(Emitted {
            items,
            rows: Vec::new(),
            finished: false,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn coded(codec: &str) -> CodedStream {
        CodedStream {
            codec: codec.to_string(),
            time_base: TimeBase { num: 1, den: 25 },
            format: CodedFormat::Video {
                width: 64,
                height: 48,
                sample_aspect_ratio: None,
                color: None,
            },
            extradata: Vec::new(),
            profile: None,
            level: None,
        }
    }

    #[test]
    fn the_decoder_is_named_by_the_tag_else_by_the_stream() {
        assert_eq!(decoder_codec(&coded("h264"), Some(b"H264")), "h264");
        assert_eq!(decoder_codec(&coded("FFV1"), Some(b"FFV1")), "ffv1");
        assert_eq!(decoder_codec(&coded("FFV1"), None), "ffv1");
        assert_eq!(decoder_codec(&coded("ffv1"), None), "ffv1");
        assert_eq!(decoder_codec(&coded("vp8"), Some(b"VP80")), "vp8");
    }
}
